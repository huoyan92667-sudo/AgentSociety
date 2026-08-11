"""Deep Step-30 module joining intent, evidence, scoring, and rank policy."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

from yelp_agent.business_profiles.schema import BusinessProfileV1
from yelp_agent.cross_encoder.schema import CrossEncoderMatchResult
from yelp_agent.query.schema import RecommendationRequest
from yelp_agent.semantic_embedding.schema import SemanticMatchResult

from .config import SemanticRankingPolicy
from .intent_compiler import RankingIntentCompiler
from .policy import apply_ranking_policy
from .schema import (
    SemanticRankingMode,
    SemanticRankingResult,
    SemanticRankingUsage,
)
from .scoring import score_candidates


class BusinessProfileReader(Protocol):
    def get(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> dict[str, BusinessProfileV1]: ...


class EmbeddingMatcher(Protocol):
    def match(
        self,
        *,
        query_text: str,
        business_ids: Sequence[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> SemanticMatchResult: ...


class CrossEncoderReranker(Protocol):
    def rerank(
        self,
        *,
        query_text: str,
        business_ids: Sequence[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> CrossEncoderMatchResult: ...


class SemanticRankingEngine:
    """Expose one ranking interface while hiding every scoring detail."""

    def __init__(
        self,
        *,
        profiles: BusinessProfileReader,
        embedding_matcher: EmbeddingMatcher,
        cross_encoder_reranker: CrossEncoderReranker,
        policy: SemanticRankingPolicy,
        mode: SemanticRankingMode,
        candidate_limit: int,
        compiler: RankingIntentCompiler | None = None,
    ) -> None:
        if candidate_limit != policy.candidate_limit:
            raise ValueError("config and policy candidate limits must match")
        self._profiles = profiles
        self._embedding = embedding_matcher
        self._cross = cross_encoder_reranker
        self._policy = policy
        self._mode = mode
        self._candidate_limit = candidate_limit
        self._compiler = compiler or RankingIntentCompiler()
        self._diagnostics: list[SemanticRankingResult] = []

    @property
    def mode(self) -> SemanticRankingMode:
        return self._mode

    @property
    def policy(self) -> SemanticRankingPolicy:
        return self._policy

    def rank(
        self,
        *,
        request: RecommendationRequest,
        base_ranking: Sequence[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> SemanticRankingResult:
        base = list(base_ranking)
        if not base or len(base) != len(set(base)):
            raise ValueError("base ranking must be nonempty and unique")
        started = perf_counter()
        prefix = base[: self._candidate_limit]
        intent = self._compiler.compile(request)
        if (
            intent.rankable_condition_count == 0
            or intent.mean_confidence < self._policy.minimum_intent_confidence
        ):
            result = SemanticRankingResult(
                request_id=request.request_id,
                context_id=usage_scope,
                mode=self._mode,
                intent=intent,
                base_ranking=base,
                ranking=base,
                candidate_scores=[],
                effective_alpha=0.0,
                no_op_reason=(
                    "NO_RANKABLE_SEMANTIC_CONDITIONS"
                    if intent.rankable_condition_count == 0
                    else "INTENT_CONFIDENCE_BELOW_THRESHOLD"
                ),
                usage=SemanticRankingUsage(
                    embedding_input_tokens=0,
                    cross_encoder_input_tokens=0,
                    logical_input_tokens=0,
                    cache_hits=0,
                    cache_misses=0,
                    provider_calls=0,
                    latency_ms=(perf_counter() - started) * 1000.0,
                ),
            )
            self._diagnostics.append(result)
            return result
        try:
            profiles = self._profiles.get(prefix, cutoff_time)
            embedding = self._embedding.match(
                query_text=intent.document,
                business_ids=prefix,
                cutoff_time=cutoff_time,
                usage_scope=f"{usage_scope or request.request_id}:step30-embedding",
            )
            cross = self._cross.rerank(
                query_text=intent.document,
                business_ids=prefix,
                cutoff_time=cutoff_time,
                usage_scope=f"{usage_scope or request.request_id}:step30-cross",
            )
            embedding_scores = {
                item.business_id: item.normalized_score for item in embedding.matches
            }
            cross_scores = {
                item.business_id: item.relevance_score for item in cross.matches
            }
            if set(embedding_scores) != set(prefix) or set(cross_scores) != set(prefix):
                raise ValueError("semantic models did not score the complete prefix")
            rows = score_candidates(
                request=request,
                base_ranking=prefix,
                profiles=profiles,
                embedding_scores=embedding_scores,
                cross_encoder_scores=cross_scores,
                policy=self._policy,
            )
            ranking, rows, alpha, no_op = apply_ranking_policy(
                base_ranking=base,
                rows=rows,
                intent=intent,
                mode=self._mode,
                policy=self._policy,
            )
            usage = SemanticRankingUsage(
                embedding_input_tokens=embedding.usage.input_tokens,
                cross_encoder_input_tokens=cross.usage.input_tokens,
                logical_input_tokens=(
                    embedding.usage.logical_input_tokens
                    + cross.usage.logical_input_tokens
                ),
                cache_hits=embedding.usage.cache_hits + cross.usage.cache_hits,
                cache_misses=embedding.usage.cache_misses + cross.usage.cache_misses,
                provider_calls=embedding.usage.api_calls + cross.usage.api_calls,
                latency_ms=(perf_counter() - started) * 1000.0,
            )
            result = SemanticRankingResult(
                request_id=request.request_id,
                context_id=usage_scope,
                mode=self._mode,
                intent=intent,
                base_ranking=base,
                ranking=ranking,
                candidate_scores=rows,
                effective_alpha=alpha,
                no_op_reason=no_op,
                usage=usage,
            )
        except Exception as exc:
            result = SemanticRankingResult(
                request_id=request.request_id,
                context_id=usage_scope,
                mode=self._mode,
                intent=intent,
                base_ranking=base,
                ranking=base,
                candidate_scores=[],
                effective_alpha=0.0,
                no_op_reason="SEMANTIC_RANKING_FALLBACK",
                fallback=True,
                fallback_reason=type(exc).__name__,
                usage=SemanticRankingUsage(
                    embedding_input_tokens=0,
                    cross_encoder_input_tokens=0,
                    logical_input_tokens=0,
                    cache_hits=0,
                    cache_misses=0,
                    provider_calls=0,
                    latency_ms=(perf_counter() - started) * 1000.0,
                ),
            )
        self._diagnostics.append(result)
        return result

    def write_diagnostics(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for result in self._diagnostics:
                handle.write(
                    json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
                    + "\n"
                )
        return target
