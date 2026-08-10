"""Deep semantic matching Module with local cache and remote encoding hidden."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from time import perf_counter
from typing import Protocol

import numpy as np

from yelp_agent.data.temporal_view import BusinessRecord

from .cache import CachedEmbedding, SqliteEmbeddingCache
from .config import SemanticEmbeddingConfig
from .documents import build_business_document, build_query_document
from .encoder import EmbeddingEncoder
from .schema import (
    EmbeddingInputType,
    EmbeddingUsage,
    SemanticBusinessMatch,
    SemanticDocument,
    SemanticMatchResult,
)


class StaticBusinessReader(Protocol):
    def business(self, business_id: str) -> BusinessRecord: ...


class CachedEmbeddingGateway:
    """Resolve documents to normalized vectors with transparent persistence."""

    def __init__(
        self,
        *,
        encoder: EmbeddingEncoder,
        cache: SqliteEmbeddingCache,
        config: SemanticEmbeddingConfig,
    ) -> None:
        self.encoder = encoder
        self.cache = cache
        self.config = config
        self._instruction_sha256 = hashlib.sha256(
            config.query_instruction.encode("utf-8")
        ).hexdigest()

    def embed(
        self,
        documents: Sequence[SemanticDocument],
        *,
        input_type: EmbeddingInputType,
    ) -> tuple[list[np.ndarray], EmbeddingUsage]:
        if not documents:
            raise ValueError("documents cannot be empty")
        keys = [self._cache_key(document, input_type) for document in documents]
        cached = self.cache.get_many(keys)
        missing_by_key: dict[str, SemanticDocument] = {}
        for key, document in zip(keys, documents, strict=True):
            if key not in cached:
                missing_by_key.setdefault(key, document)
        input_tokens = 0
        api_calls = 0
        provider_latency_ms = 0.0
        missing_items = list(missing_by_key.items())
        for offset in range(0, len(missing_items), self.encoder.batch_size):
            batch = missing_items[offset : offset + self.encoder.batch_size]
            response = self.encoder.encode(
                [document.text for _, document in batch],
                input_type=input_type,
            )
            if response.model != self.encoder.model:
                raise ValueError("embedding response model does not match configured model")
            api_calls += 1
            input_tokens += response.input_tokens
            provider_latency_ms += response.latency_ms
            records: list[CachedEmbedding] = []
            token_share = response.input_tokens // max(1, len(batch))
            for (key, document), vector in zip(batch, response.vectors, strict=True):
                normalized = _normalize(vector)
                record = CachedEmbedding(
                    cache_key=key,
                    provider=self.encoder.provider,
                    model=self.encoder.model,
                    dimension=self.encoder.dimension,
                    input_type=input_type,
                    instruction_sha256=(
                        self._instruction_sha256 if input_type == "query" else "0" * 64
                    ),
                    document_version=document.document_version,
                    text_sha256=document.text_sha256,
                    vector=normalized,
                    input_tokens=token_share,
                )
                records.append(record)
                cached[key] = record
            self.cache.put_many(records)
        vectors = [cached[key].vector for key in keys]
        self.cache.write_manifest(
            model=self.encoder.model,
            dimension=self.encoder.dimension,
        )
        misses = len(missing_by_key)
        return vectors, EmbeddingUsage(
            api_calls=api_calls,
            input_tokens=input_tokens,
            cache_hits=len(documents) - misses,
            cache_misses=misses,
            estimated_cost_cny=(
                input_tokens
                / 1000.0
                * self.config.price_cny_per_1000_input_tokens
            ),
            provider_latency_ms=provider_latency_ms,
        )

    def _cache_key(
        self,
        document: SemanticDocument,
        input_type: EmbeddingInputType,
    ) -> str:
        payload = {
            "provider": self.encoder.provider,
            "model": self.encoder.model,
            "dimension": self.encoder.dimension,
            "input_type": input_type,
            "instruction_sha256": (
                self._instruction_sha256 if input_type == "query" else "0" * 64
            ),
            "document_version": document.document_version,
            "text_sha256": document.text_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class SemanticEmbeddingMatcher:
    """Match one visible query against an exact-cutoff scoped business set."""

    def __init__(
        self,
        *,
        businesses: StaticBusinessReader,
        gateway: CachedEmbeddingGateway,
        config: SemanticEmbeddingConfig,
    ) -> None:
        self._businesses = businesses
        self._gateway = gateway
        self._config = config

    def match(
        self,
        *,
        query_text: str,
        business_ids: Sequence[str],
        cutoff_time: datetime,
    ) -> SemanticMatchResult:
        ids = list(business_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("business IDs must be a nonempty unique sequence")
        started = perf_counter()
        businesses = [self._businesses.business(business_id) for business_id in ids]
        if [business.business_id for business in businesses] != ids:
            raise ValueError("static business reader returned an incorrect scope")
        query = build_query_document(query_text)
        business_documents = [
            build_business_document(
                business,
                document_version=self._config.business_document_version,
            )
            for business in businesses
        ]
        query_vectors, query_usage = self._gateway.embed([query], input_type="query")
        business_vectors, business_usage = self._gateway.embed(
            business_documents,
            input_type="document",
        )
        query_vector = query_vectors[0]
        scores = [
            float(np.clip(np.dot(query_vector, vector), -1.0, 1.0))
            for vector in business_vectors
        ]
        order = sorted(range(len(ids)), key=lambda index: (-scores[index], ids[index]))
        rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
        matches = [
            SemanticBusinessMatch(
                business_id=business_id,
                cutoff_time=cutoff_time,
                document_sha256=business_documents[index].text_sha256,
                cosine_similarity=scores[index],
                normalized_score=(scores[index] + 1.0) / 2.0,
                semantic_rank=rank_by_index[index],
            )
            for index, business_id in enumerate(ids)
        ]
        matches.sort(key=lambda item: (item.semantic_rank, item.business_id))
        elapsed_ms = (perf_counter() - started) * 1000.0
        return SemanticMatchResult(
            query_sha256=query.text_sha256,
            model=self._gateway.encoder.model,
            dimension=self._gateway.encoder.dimension,
            matches=matches,
            usage=EmbeddingUsage(
                api_calls=query_usage.api_calls + business_usage.api_calls,
                input_tokens=query_usage.input_tokens + business_usage.input_tokens,
                cache_hits=query_usage.cache_hits + business_usage.cache_hits,
                cache_misses=query_usage.cache_misses + business_usage.cache_misses,
                estimated_cost_cny=(
                    query_usage.estimated_cost_cny
                    + business_usage.estimated_cost_cny
                ),
                provider_latency_ms=min(
                    elapsed_ms,
                    query_usage.provider_latency_ms
                    + business_usage.provider_latency_ms,
                ),
            ),
        )


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("embedding vector must have a positive finite norm")
    return value / norm
