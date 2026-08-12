"""Deep Step-33 ranking module joining retrieval, LambdaMART and semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from time import perf_counter
from typing import Protocol

from yelp_agent.business_profiles.schema import BusinessProfileV1
from yelp_agent.learning_to_rank.runtime import HybridV2ScoredCandidate
from yelp_agent.query.schema import RecommendationRequest
from yelp_agent.query_retrieval import QueryRetrievalResult
from yelp_agent.retrieval import RetrievalResult
from yelp_agent.semantic_ranking import (
    RankingIntentCompiler,
    apply_ranking_policy,
    score_candidates,
)

from .candidate_pool import ProtectedCandidatePool, hybrid_candidate_rows
from .config import QueryAwareRankingPolicy
from .schema import (
    CandidateEvidence,
    PreparedQueryAwareCase,
    QueryAwareRankingResult,
)
from .scoring import apply_coarse_policy, rank_percentile


class HybridRankingService(Protocol):
    def rank(
        self,
        *,
        request_id: str,
        user_id: str,
        cutoff_time: datetime,
        candidates: Sequence[Mapping[str, object]],
    ) -> Sequence[Mapping[str, object]]: ...


class EmbeddingMatcher(Protocol):
    def match(
        self,
        *,
        query_text: str,
        business_ids: Sequence[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> object: ...


class CrossEncoderReranker(Protocol):
    def rerank(
        self,
        *,
        query_text: str,
        business_ids: Sequence[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> object: ...


class BusinessProfileReader(Protocol):
    def get(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> dict[str, BusinessProfileV1]: ...


class ExternalModelCallError(RuntimeError):
    """Raised when the local-only formal experiment observes API usage."""


class QueryAwareRecommendationEngine:
    """Expose preparation and finalization without ever accepting target labels."""

    def __init__(
        self,
        *,
        ranking_service: HybridRankingService,
        embedding_matcher: EmbeddingMatcher,
        cross_encoder: CrossEncoderReranker,
        business_profiles: BusinessProfileReader,
        policy: QueryAwareRankingPolicy,
        compiler: RankingIntentCompiler | None = None,
    ) -> None:
        self._ranking = ranking_service
        self._embedding = embedding_matcher
        self._cross = cross_encoder
        self._profiles = business_profiles
        self._policy = policy
        self._compiler = compiler or RankingIntentCompiler()
        self._pool = ProtectedCandidatePool(
            candidate_limit=policy.union_candidate_limit
        )

    @property
    def policy(self) -> QueryAwareRankingPolicy:
        return self._policy

    def prepare(
        self,
        *,
        case_id: str,
        split: str,
        request: RecommendationRequest,
        history: RetrievalResult,
        query: QueryRetrievalResult,
        rrf_fusion_ranking: Sequence[str],
        usage_scope: str,
    ) -> PreparedQueryAwareCase:
        started = perf_counter()
        pool = self._pool.build(history, query)
        ids = [item.business_id for item in pool.items]
        feature_rows = hybrid_candidate_rows(
            pool,
            history_candidate_count=len(history.candidates),
        )
        hybrid_rows = list(
            self._ranking.rank(
                request_id=request.request_id,
                user_id=request.user_id,
                cutoff_time=request.cutoff_time,
                candidates=feature_rows,
            )
        )
        hybrid = {
            str(row.get("business_id")): HybridV2ScoredCandidate.model_validate(row)
            for row in hybrid_rows
        }
        if set(hybrid) != set(ids):
            raise ValueError("LambdaMART did not score the complete protected union")
        intent = self._compiler.compile(request)
        embedding_result = self._embedding.match(
            query_text=intent.document,
            business_ids=ids,
            cutoff_time=request.cutoff_time,
            usage_scope=f"{usage_scope}:union-embedding",
        )
        if query.usage.provider_calls != 0 or embedding_result.usage.api_calls != 0:
            raise ExternalModelCallError(
                "Step 33 formal ranking must not call an external model"
            )
        embedding_by_id = {item.business_id: item for item in embedding_result.matches}
        if set(embedding_by_id) != set(ids):
            raise ValueError("embedding matcher did not score the protected union")
        query_size = len(query.candidates)
        candidate_evidence = []
        for item in pool.items:
            model = hybrid[item.business_id]
            embedding = embedding_by_id[item.business_id]
            candidate_evidence.append(
                CandidateEvidence(
                    business_id=item.business_id,
                    source_channels=item.source_channels,
                    history_rank=item.history_rank,
                    query_rank=item.query_rank,
                    lightgbm_rank=model.rank,
                    lightgbm_model_score=model.model_score,
                    lightgbm_percentile=rank_percentile(model.rank, len(ids)),
                    embedding_rank=embedding.semantic_rank,
                    embedding_score=embedding.normalized_score,
                    embedding_percentile=rank_percentile(
                        embedding.semantic_rank, len(ids)
                    ),
                    query_retrieval_percentile=(
                        0.0
                        if item.query_rank is None
                        else rank_percentile(item.query_rank, query_size)
                    ),
                )
            )
        history_ids = {item.business_id for item in history.candidates}
        history_lightgbm = [
            item.business_id
            for item in sorted(
                (item for item in candidate_evidence if item.business_id in history_ids),
                key=lambda value: (
                    -value.lightgbm_model_score,
                    value.business_id,
                ),
            )
        ]
        return PreparedQueryAwareCase(
            case_id=case_id,
            split=split,
            request=request,
            history_ranking=[item.business_id for item in history.candidates],
            query_ranking=[item.business_id for item in query.candidates],
            rrf_fusion_ranking=list(rrf_fusion_ranking),
            history_lightgbm_ranking=history_lightgbm,
            union_candidate_count=pool.union_candidate_count,
            eligible_candidate_count=len(pool.items),
            overlap_count=pool.overlap_count,
            exclusions=list(pool.exclusions),
            candidates=candidate_evidence,
            retrieval_latency_ms=history.latency_ms + query.latency_ms,
            preparation_latency_ms=(perf_counter() - started) * 1000.0,
            embedding_input_tokens=(
                query.usage.embedding_input_tokens
                + embedding_result.usage.input_tokens
            ),
            embedding_logical_tokens=(
                query.usage.embedding_logical_tokens
                + embedding_result.usage.logical_input_tokens
            ),
            embedding_cache_hits=(
                query.usage.cache_hits + embedding_result.usage.cache_hits
            ),
            embedding_cache_misses=(
                query.usage.cache_misses + embedding_result.usage.cache_misses
            ),
        )

    def finalize(
        self,
        prepared: PreparedQueryAwareCase,
        *,
        usage_scope: str,
    ) -> QueryAwareRankingResult:
        started = perf_counter()
        coarse = apply_coarse_policy(prepared, self._policy)
        coarse_ranking = [item.business_id for item in coarse]
        limit = min(self._policy.semantic_candidate_limit, len(coarse_ranking))
        prefix = coarse_ranking[:limit]
        semantic_scores = []
        final_ranking = list(coarse_ranking)
        fallback = False
        fallback_reason = None
        cross_input_tokens = 0
        cross_logical_tokens = 0
        cross_cache_hits = 0
        cross_cache_misses = 0
        try:
            intent = self._compiler.compile(prepared.request)
            profiles = self._profiles.get(prefix, prepared.request.cutoff_time)
            evidence_by_id = {
                item.business_id: item for item in prepared.candidates
            }
            cross = self._cross.rerank(
                query_text=intent.document,
                business_ids=prefix,
                cutoff_time=prepared.request.cutoff_time,
                usage_scope=f"{usage_scope}:cross-encoder",
            )
            if cross.usage.api_calls != 0:
                raise ExternalModelCallError(
                    "Step 33 formal reranking must not call an external model"
                )
            cross_by_id = {
                item.business_id: item.relevance_score for item in cross.matches
            }
            if set(cross_by_id) != set(prefix):
                raise ValueError("Cross-Encoder did not score the complete prefix")
            rows = score_candidates(
                request=prepared.request,
                base_ranking=prefix,
                profiles=profiles,
                embedding_scores={
                    business_id: evidence_by_id[business_id].embedding_score
                    for business_id in prefix
                },
                cross_encoder_scores=cross_by_id,
                policy=self._policy.semantic_policy,
            )
            final_ranking, semantic_scores, _, _ = apply_ranking_policy(
                base_ranking=coarse_ranking,
                rows=rows,
                intent=intent,
                mode="protected",
                policy=self._policy.semantic_policy,
            )
            cross_input_tokens = cross.usage.input_tokens
            cross_logical_tokens = cross.usage.logical_input_tokens
            cross_cache_hits = cross.usage.cache_hits
            cross_cache_misses = cross.usage.cache_misses
        except ExternalModelCallError:
            raise
        except Exception as exc:
            fallback = True
            fallback_reason = f"SEMANTIC_RANKING_FALLBACK:{type(exc).__name__}"
        final_rank = {
            business_id: rank
            for rank, business_id in enumerate(final_ranking, start=1)
        }
        updated_coarse = [
            item.model_copy(update={"final_rank": final_rank[item.business_id]})
            for item in coarse
        ]
        elapsed = (perf_counter() - started) * 1000.0
        return QueryAwareRankingResult(
            case_id=prepared.case_id,
            request_id=prepared.request.request_id,
            split=prepared.split,
            ranking=final_ranking,
            top_10=final_ranking[: self._policy.internal_result_limit],
            displayed_top_5=final_ranking[: self._policy.display_limit],
            coarse_scores=updated_coarse,
            semantic_scores=semantic_scores,
            hard_exclusions=prepared.exclusions,
            query_weight=self._policy.selected_query_weight,
            semantic_candidate_limit=self._policy.semantic_candidate_limit,
            fallback=fallback,
            fallback_reason=fallback_reason,
            latency_ms=prepared.retrieval_latency_ms
            + prepared.preparation_latency_ms
            + elapsed,
            embedding_input_tokens=prepared.embedding_input_tokens,
            cross_encoder_input_tokens=cross_input_tokens,
            logical_input_tokens=prepared.embedding_logical_tokens
            + cross_logical_tokens,
            cache_hits=prepared.embedding_cache_hits + cross_cache_hits,
            cache_misses=prepared.embedding_cache_misses + cross_cache_misses,
        )
