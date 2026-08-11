"""Cached static-document Cross-Encoder reranking service."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from yelp_agent.data.temporal_view import BusinessRecord
from yelp_agent.semantic_embedding.documents import build_business_document, build_query_document

from .cache import CachedCrossEncoderScore, SqliteCrossEncoderCache
from .config import CrossEncoderConfig
from .schema import (
    CrossEncoderBusinessMatch,
    CrossEncoderMatchResult,
    CrossEncoderUsage,
    CrossEncoderUsageEvent,
)
from .scorer import PairScorer


class StaticBusinessReader(Protocol):
    def business(self, business_id: str) -> BusinessRecord: ...


class CachedCrossEncoderReranker:
    """Score exact visible candidates while hiding local model mechanics."""

    def __init__(
        self,
        *,
        businesses: StaticBusinessReader,
        scorer: PairScorer,
        cache: SqliteCrossEncoderCache,
        config: CrossEncoderConfig,
    ) -> None:
        self._businesses = businesses
        self._scorer = scorer
        self._cache = cache
        self._config = config
        self._instruction_sha256 = hashlib.sha256(
            config.instruction.encode("utf-8")
        ).hexdigest()

    def rerank(
        self,
        *,
        query_text: str,
        business_ids: Sequence[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> CrossEncoderMatchResult:
        ids = list(business_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("business IDs must be a nonempty unique sequence")
        query = build_query_document(query_text)
        businesses = [self._businesses.business(business_id) for business_id in ids]
        if [business.business_id for business in businesses] != ids:
            raise ValueError("static business reader returned an incorrect scope")
        documents = [
            build_business_document(
                business,
                document_version=self._config.business_document_version,
            )
            for business in businesses
        ]
        keys = [self._cache_key(query.text_sha256, document.text_sha256) for document in documents]
        cached = self._cache.get_many(keys)
        missing_indices = [index for index, key in enumerate(keys) if key not in cached]
        scored_tokens = 0
        scorer_calls = 0
        truncated = 0
        latency_ms = 0.0
        for offset in range(0, len(missing_indices), self._scorer.batch_size):
            indices = missing_indices[offset : offset + self._scorer.batch_size]
            response = self._scorer.score(
                query.text,
                [documents[index].text for index in indices],
            )
            scorer_calls += 1
            scored_tokens += response.input_tokens
            truncated += response.truncated_pair_count
            latency_ms += response.latency_ms
            records = []
            for index, score, token_count in zip(
                indices, response.scores, response.per_pair_input_tokens, strict=True
            ):
                record = CachedCrossEncoderScore(
                    cache_key=keys[index], model=self._scorer.model,
                    instruction_sha256=self._instruction_sha256,
                    document_version=documents[index].document_version,
                    query_sha256=query.text_sha256,
                    document_sha256=documents[index].text_sha256,
                    max_sequence_length=self._config.max_sequence_length,
                    score=score, input_tokens=token_count,
                )
                records.append(record)
                cached[keys[index]] = record
            self._cache.put_many(records)
        logical_tokens = sum(cached[key].input_tokens for key in keys)
        misses = len(missing_indices)
        usage = CrossEncoderUsage(
            scorer_calls=scorer_calls,
            api_calls=0,
            input_tokens=scored_tokens,
            logical_input_tokens=logical_tokens,
            cache_saved_tokens=max(0, logical_tokens - scored_tokens),
            truncated_pair_count=truncated,
            cache_hits=len(keys) - misses,
            cache_misses=misses,
            provider_latency_ms=latency_ms,
        )
        order = sorted(range(len(ids)), key=lambda index: (-cached[keys[index]].score, ids[index]))
        rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
        matches = [
            CrossEncoderBusinessMatch(
                business_id=business_id,
                cutoff_time=cutoff_time,
                document_sha256=documents[index].text_sha256,
                relevance_score=cached[keys[index]].score,
                cross_encoder_rank=rank_by_index[index],
            )
            for index, business_id in enumerate(ids)
        ]
        matches.sort(key=lambda item: (item.cross_encoder_rank, item.business_id))
        self._cache.record_usage(
            CrossEncoderUsageEvent(
                usage_scope=usage_scope or query.text_sha256,
                model=self._scorer.model,
                requested_pair_count=len(keys), unique_pair_count=len(set(keys)),
                cache_hits=usage.cache_hits, cache_misses=usage.cache_misses,
                logical_input_tokens=usage.logical_input_tokens,
                scored_input_tokens=usage.input_tokens,
                cache_saved_tokens=usage.cache_saved_tokens,
                truncated_pair_count=usage.truncated_pair_count,
                scorer_calls=usage.scorer_calls, api_calls=0,
                latency_ms=usage.provider_latency_ms,
            )
        )
        self._cache.write_manifest(model=self._scorer.model)
        return CrossEncoderMatchResult(
            query_sha256=query.text_sha256,
            model=self._scorer.model,
            matches=matches,
            usage=usage,
        )

    def _cache_key(self, query_sha256: str, document_sha256: str) -> str:
        payload = {
            "provider": "local", "model": self._scorer.model,
            "instruction_sha256": self._instruction_sha256,
            "document_version": self._config.business_document_version,
            "query_sha256": query_sha256, "document_sha256": document_sha256,
            "max_sequence_length": self._config.max_sequence_length,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
