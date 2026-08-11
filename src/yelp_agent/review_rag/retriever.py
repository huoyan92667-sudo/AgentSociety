"""Aspect, BM25, and lazy local-Embedding Review retrieval with RRF fusion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np

from yelp_agent.config import ReviewAspectVocabulary
from yelp_agent.semantic_embedding.matcher import CachedEmbeddingGateway
from yelp_agent.semantic_embedding.schema import (
    EmbeddingUsage,
    SemanticDocument,
)

from .bm25 import bm25_scores
from .config import ReviewRAGConfig, ReviewRAGPolicy
from .schema import (
    ReviewEvidenceHit,
    ReviewSearchRequest,
    ReviewSearchResult,
    ReviewSegment,
    SegmentAspectEvidence,
)
from .store import ReviewRAGStore


@dataclass(frozen=True, slots=True)
class _RankedRoutes:
    segments: tuple[ReviewSegment, ...]
    aspect_evidence: Mapping[str, tuple[SegmentAspectEvidence, ...]]
    aspect_ranks: Mapping[str, int]
    bm25_ranks: Mapping[str, int]
    embedding_ranks: Mapping[str, int]
    usage: EmbeddingUsage
    provider: str | None
    model: str | None


class ReviewRetriever:
    """The one deep module interface used by tools, scripts, and tests."""

    def __init__(
        self,
        *,
        store: ReviewRAGStore,
        config: ReviewRAGConfig,
        policy: ReviewRAGPolicy,
        vocabulary: ReviewAspectVocabulary,
        embedding_gateway: CachedEmbeddingGateway | None = None,
    ) -> None:
        self._store = store
        self._config = config
        self._policy = policy
        self._vocabulary = vocabulary
        self._embedding_gateway = embedding_gateway

    def search(self, request: ReviewSearchRequest) -> ReviewSearchResult:
        """Return Top-5 unique Review IDs inside the locked scope and cutoff."""

        per_business: list[list[ReviewEvidenceHit]] = []
        total_eligible = 0
        route_counts = {"aspect": 0, "bm25": 0, "embedding": 0}
        usage = _empty_usage()
        provider: str | None = None
        model: str | None = None
        for business_id in request.business_ids:
            routes = self._routes_for_business(request, business_id)
            total_eligible += len(routes.segments)
            route_counts["aspect"] += len(routes.aspect_ranks)
            route_counts["bm25"] += len(routes.bm25_ranks)
            route_counts["embedding"] += len(routes.embedding_ranks)
            usage = _add_usage(usage, routes.usage)
            provider = routes.provider or provider
            model = routes.model or model
            per_business.append(
                self._fuse_business(
                    business_id,
                    routes,
                    top_k=request.top_k,
                )
            )
        merged = _round_robin(per_business, request.top_k)
        hits = [item.model_copy(update={"rank": rank}) for rank, item in enumerate(merged, 1)]
        payload = {
            "query_text": request.query_text,
            "business_ids": request.business_ids,
            "cutoff_time": request.cutoff_time.isoformat(),
            "aspects": request.aspects,
            "top_k": request.top_k,
        }
        request_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return ReviewSearchResult(
            request_sha256=request_hash,
            business_ids=request.business_ids,
            cutoff_time=request.cutoff_time,
            hits=hits,
            route_result_counts=route_counts,
            eligible_segment_count=total_eligible,
            embedding_provider=provider,  # type: ignore[arg-type]
            embedding_model=model,
            embedding_usage=usage,
        )

    def _routes_for_business(
        self,
        request: ReviewSearchRequest,
        business_id: str,
    ) -> _RankedRoutes:
        segments = self._store.segments_before(business_id, request.cutoff_time)
        evidence = self._store.aspect_evidence(
            segments,
            aspects=list(request.aspects),
            cutoff_time=request.cutoff_time,
        )
        aspect_order = sorted(
            evidence,
            key=lambda segment_id: (
                -max(item.confidence for item in evidence[segment_id]),
                -_segment_by_id(segments)[segment_id].useful,
                segment_id,
            ),
        )[: self._config.aspect_candidate_limit]
        aspect_ranks = {value: rank for rank, value in enumerate(aspect_order, 1)}
        expansions = self._aspect_expansions(request.aspects)
        lexical_scores = bm25_scores(
            request.query_text,
            [item.text for item in segments],
            expansions=expansions,
            k1=self._config.bm25_k1,
            b=self._config.bm25_b,
        )
        bm25_order = sorted(
            (index for index, score in enumerate(lexical_scores) if score > 0),
            key=lambda index: (
                -lexical_scores[index],
                -segments[index].useful,
                segments[index].segment_id,
            ),
        )[: self._config.bm25_candidate_limit]
        bm25_ranks = {
            segments[index].segment_id: rank
            for rank, index in enumerate(bm25_order, 1)
        }
        embedding_candidates = _ordered_union(
            [aspect_order, [segments[index].segment_id for index in bm25_order]],
            limit=self._config.embedding_candidate_limit,
        )
        if len(embedding_candidates) < self._config.embedding_candidate_limit:
            embedding_candidates = _ordered_union(
                [embedding_candidates, [item.segment_id for item in segments]],
                limit=self._config.embedding_candidate_limit,
            )
        embedding_ranks: dict[str, int] = {}
        usage = _empty_usage()
        provider = None
        model = None
        if (
            self._embedding_gateway is not None
            and embedding_candidates
            and self._policy.embedding_weight > 0
        ):
            by_id = _segment_by_id(segments)
            query_text = " ".join((request.query_text, *expansions)).strip()
            query_document = _document(
                source_id=request.usage_scope,
                source_kind="query",
                text=query_text,
                version=self._config.review_document_version,
            )
            passage_documents = [
                _document(
                    source_id=segment_id,
                    source_kind="review_chunk",
                    text=by_id[segment_id].text,
                    version=self._config.review_document_version,
                )
                for segment_id in embedding_candidates
            ]
            query_vectors, query_usage = self._embedding_gateway.embed(
                [query_document],
                input_type="query",
                usage_scope=request.usage_scope,
            )
            passage_vectors, passage_usage = self._embedding_gateway.embed(
                passage_documents,
                input_type="document",
                usage_scope=request.usage_scope,
            )
            scores = [
                float(np.clip(np.dot(query_vectors[0], vector), -1.0, 1.0))
                for vector in passage_vectors
            ]
            order = sorted(
                range(len(embedding_candidates)),
                key=lambda index: (-scores[index], embedding_candidates[index]),
            )
            embedding_ranks = {
                embedding_candidates[index]: rank
                for rank, index in enumerate(order, 1)
            }
            usage = _add_usage(query_usage, passage_usage)
            provider = self._embedding_gateway.encoder.provider
            model = self._embedding_gateway.encoder.model
        return _RankedRoutes(
            segments=segments,
            aspect_evidence=evidence,
            aspect_ranks=aspect_ranks,
            bm25_ranks=bm25_ranks,
            embedding_ranks=embedding_ranks,
            usage=usage,
            provider=provider,
            model=model,
        )

    def _fuse_business(
        self,
        business_id: str,
        routes: _RankedRoutes,
        *,
        top_k: int,
    ) -> list[ReviewEvidenceHit]:
        candidates = set(routes.aspect_ranks) | set(routes.bm25_ranks) | set(
            routes.embedding_ranks
        )
        by_id = _segment_by_id(routes.segments)
        maximum = sum(
            weight / (self._policy.rrf_k + 1)
            for weight in (
                self._policy.aspect_weight,
                self._policy.bm25_weight,
                self._policy.embedding_weight,
            )
            if weight > 0
        ) or 1.0
        rows: list[tuple[float, ReviewSegment]] = []
        for segment_id in candidates:
            score = 0.0
            for rank, weight in (
                (routes.aspect_ranks.get(segment_id), self._policy.aspect_weight),
                (routes.bm25_ranks.get(segment_id), self._policy.bm25_weight),
                (routes.embedding_ranks.get(segment_id), self._policy.embedding_weight),
            ):
                if rank is not None and weight > 0:
                    score += weight / (self._policy.rrf_k + rank)
            rows.append((score, by_id[segment_id]))
        rows.sort(
            key=lambda item: (
                -item[0],
                -item[1].useful,
                -item[1].review_time.timestamp(),
                item[1].segment_id,
            )
        )
        result: list[ReviewEvidenceHit] = []
        seen_reviews: set[str] = set()
        for score, segment in rows:
            if segment.review_id in seen_reviews:
                continue
            seen_reviews.add(segment.review_id)
            evidence = routes.aspect_evidence.get(segment.segment_id, ())
            result.append(
                ReviewEvidenceHit(
                    rank=len(result) + 1,
                    segment_id=segment.segment_id,
                    review_id=segment.review_id,
                    business_id=business_id,
                    review_time=segment.review_time,
                    stars=segment.stars,
                    useful=segment.useful,
                    text=segment.text[:2000],
                    text_sha256=segment.text_sha256,
                    matched_aspects=list(dict.fromkeys(item.aspect for item in evidence)),
                    aspect_sentiments=list(
                        dict.fromkeys(item.sentiment for item in evidence)
                    ),
                    aspect_rank=routes.aspect_ranks.get(segment.segment_id),
                    bm25_rank=routes.bm25_ranks.get(segment.segment_id),
                    embedding_rank=routes.embedding_ranks.get(segment.segment_id),
                    rrf_score=score,
                    relevance_score=float(np.clip(score / maximum, 0.0, 1.0)),
                )
            )
            if len(result) >= top_k:
                break
        return result

    def _aspect_expansions(self, aspects: Sequence[str]) -> list[str]:
        result: list[str] = []
        for aspect in aspects:
            terms = self._vocabulary.aspects.get(aspect)  # type: ignore[arg-type]
            if terms is None:
                continue
            result.extend((aspect.replace("_", " "), *terms.positive, *terms.negative))
        return result


def _segment_by_id(segments: Sequence[ReviewSegment]) -> dict[str, ReviewSegment]:
    return {item.segment_id: item for item in segments}


def _document(
    *,
    source_id: str,
    source_kind: str,
    text: str,
    version: str,
) -> SemanticDocument:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return SemanticDocument(
        source_id=source_id,
        source_kind=source_kind,  # type: ignore[arg-type]
        text=text,
        text_sha256=digest,
        document_version=version,
    )


def _ordered_union(routes: Sequence[Sequence[str]], *, limit: int) -> list[str]:
    result: list[str] = []
    max_length = max((len(route) for route in routes), default=0)
    for index in range(max_length):
        for route in routes:
            if index < len(route) and route[index] not in result:
                result.append(route[index])
                if len(result) >= limit:
                    return result
    return result


def _round_robin(values: Sequence[Sequence[ReviewEvidenceHit]], limit: int) -> list[ReviewEvidenceHit]:
    result: list[ReviewEvidenceHit] = []
    max_length = max((len(value) for value in values), default=0)
    for index in range(max_length):
        for value in values:
            if index < len(value):
                result.append(value[index])
                if len(result) >= limit:
                    return result
    return result


def _empty_usage() -> EmbeddingUsage:
    return EmbeddingUsage(
        encoder_calls=0,
        api_calls=0,
        input_tokens=0,
        logical_input_tokens=0,
        cache_saved_tokens=0,
        truncated_text_count=0,
        cache_hits=0,
        cache_misses=0,
        estimated_cost_cny=0.0,
        provider_latency_ms=0.0,
    )


def _add_usage(first: EmbeddingUsage, second: EmbeddingUsage) -> EmbeddingUsage:
    return EmbeddingUsage(
        encoder_calls=first.encoder_calls + second.encoder_calls,
        api_calls=first.api_calls + second.api_calls,
        input_tokens=first.input_tokens + second.input_tokens,
        logical_input_tokens=first.logical_input_tokens + second.logical_input_tokens,
        cache_saved_tokens=first.cache_saved_tokens + second.cache_saved_tokens,
        truncated_text_count=first.truncated_text_count + second.truncated_text_count,
        cache_hits=first.cache_hits + second.cache_hits,
        cache_misses=first.cache_misses + second.cache_misses,
        estimated_cost_cny=first.estimated_cost_cny + second.estimated_cost_cny,
        provider_latency_ms=first.provider_latency_ms + second.provider_latency_ms,
    )
