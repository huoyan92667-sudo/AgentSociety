"""把正反描述变成向量，并从硬筛商家范围内召回、合并和判定评论。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Protocol

import numpy as np

from yelp_agent.semantic_embedding.encoder import EncodedBatch

from .full_reviews import FullReviewStore
from .qdrant_store import QdrantReviewSegmentStore
from .schema import (
    EvidenceDirection,
    PreferenceSearchDescription,
    QdrantSegmentHit,
    ReviewRetrievalMetrics,
    ReviewSimilarityCandidate,
)


class QueryEncoder(Protocol):
    """只负责把少量正反检索说法编码成查询向量。"""

    def encode(self, texts: list[str], *, input_type: str) -> EncodedBatch: ...

    def close(self) -> None: ...


class SegmentVectorReader(Protocol):
    """按 Qdrant 稳定点编号读取本地评论片段向量。"""

    def get_many(self, point_ids: list[int]) -> dict[int, np.ndarray]: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class ReviewRetrievalBatch:
    """全部偏好的评论候选和本轮可观察性能数据。"""

    by_requirement: dict[str, dict[str, list[ReviewSimilarityCandidate]]]
    metrics: ReviewRetrievalMetrics


class ReviewEvidenceRetriever:
    """每家先各取正反候选，再按 P、N 差值直接判定方向。"""

    def __init__(
        self,
        *,
        store: QdrantReviewSegmentStore,
        encoder: QueryEncoder,
        segment_vectors: SegmentVectorReader,
        full_reviews: FullReviewStore,
        recall_threshold: float = 0.55,
        acceptance_threshold: float = 0.60,
        direction_margin: float = 0.05,
        recall_each_side: int = 15,
        initial_segment_group_size: int = 15,
        middle_segment_group_size: int = 30,
        final_segment_group_size: int = 60,
        minimum_clear_evidence: int = 5,
        search_concurrency: int = 4,
    ) -> None:
        if not -1 <= recall_threshold <= acceptance_threshold <= 1:
            raise ValueError("review similarity thresholds are invalid")
        if not 0 <= direction_margin <= 2:
            raise ValueError("direction margin is invalid")
        if (
            recall_each_side < 1
            or initial_segment_group_size < recall_each_side
            or middle_segment_group_size < initial_segment_group_size
            or final_segment_group_size < middle_segment_group_size
        ):
            raise ValueError("review recall limits are invalid")
        if minimum_clear_evidence < 1 or search_concurrency < 1:
            raise ValueError("review retrieval controls must be positive")
        self._store = store
        self._encoder = encoder
        self._segment_vectors = segment_vectors
        self._full_reviews = full_reviews
        self.recall_threshold = recall_threshold
        self.acceptance_threshold = acceptance_threshold
        self.direction_margin = direction_margin
        self._recall_each_side = recall_each_side
        self._initial_segment_group_size = initial_segment_group_size
        self._middle_segment_group_size = middle_segment_group_size
        self._final_segment_group_size = final_segment_group_size
        self._minimum_clear_evidence = minimum_clear_evidence
        self._search_concurrency = search_concurrency

    def close(self) -> None:
        self._full_reviews.close()
        self._segment_vectors.close()
        self._encoder.close()
        self._store.close()

    def retrieve(
        self,
        requirement: PreferenceSearchDescription,
        business_ids: list[str],
        *,
        cutoff_time: datetime | None = None,
    ) -> dict[str, list[ReviewSimilarityCandidate]]:
        """兼容单条要求调用；在线排序会使用一次处理全部要求的入口。"""

        batch = self.retrieve_many(
            [requirement],
            business_ids,
            cutoff_time=cutoff_time,
        )
        return batch.by_requirement.get(requirement.requirement_id, {})

    def retrieve_many(
        self,
        requirements: list[PreferenceSearchDescription],
        business_ids: list[str],
        *,
        cutoff_time: datetime | None = None,
    ) -> ReviewRetrievalBatch:
        """一次编码并批量检索全部要求，证据确实不足时才扩大召回。"""

        if not requirements or not business_ids:
            return ReviewRetrievalBatch(
                by_requirement={item.requirement_id: {} for item in requirements},
                metrics=ReviewRetrievalMetrics(),
            )

        descriptions: list[str] = []
        requirement_slices: dict[str, tuple[int, int, int]] = {}
        for requirement in requirements:
            start = len(descriptions)
            descriptions.extend(requirement.positive_descriptions)
            positive_end = len(descriptions)
            descriptions.extend(requirement.negative_descriptions)
            requirement_slices[requirement.requirement_id] = (
                start,
                positive_end,
                len(descriptions),
            )

        query_vectors, embedding_latency_ms, embedding_batches = self._encode_all(
            descriptions
        )
        first_started = perf_counter()
        first_results = self._search_many(
            query_vectors,
            business_ids,
            cutoff_time=cutoff_time,
            group_size=self._initial_segment_group_size,
        )
        first_search_ms = (perf_counter() - first_started) * 1000

        segments = self._empty_segment_map(requirements, business_ids)
        self._merge_search_results(
            segments,
            requirements,
            requirement_slices,
            first_results,
        )
        # Qdrant 只负责找编号和片段事实。这里跨所有偏好统一去重，
        # 再从本地内存映射文件读取真正参与点积的向量。
        vectors_by_point: dict[int, np.ndarray] = {}
        vector_load_ms = self._load_missing_vectors(segments, vectors_by_point)
        middle_ids = self._businesses_needing_expansion(
            requirements,
            business_ids,
            requirement_slices,
            query_vectors,
            segments,
            first_results,
            group_size=self._initial_segment_group_size,
            vectors_by_point=vectors_by_point,
        )

        middle_search_ms = 0.0
        middle_hit_count = 0
        middle_results: list[dict[str, list[QdrantSegmentHit]]] = []
        if middle_ids:
            middle_started = perf_counter()
            middle_results = self._search_many(
                query_vectors,
                middle_ids,
                cutoff_time=cutoff_time,
                group_size=self._middle_segment_group_size,
            )
            middle_search_ms = (perf_counter() - middle_started) * 1000
            middle_hit_count = _search_hit_count(middle_results)
            self._merge_search_results(
                segments,
                requirements,
                requirement_slices,
                middle_results,
            )
            vector_load_ms += self._load_missing_vectors(segments, vectors_by_point)

        final_ids = (
            self._businesses_needing_expansion(
                requirements,
                middle_ids,
                requirement_slices,
                query_vectors,
                segments,
                middle_results,
                group_size=self._middle_segment_group_size,
                vectors_by_point=vectors_by_point,
            )
            if middle_results
            else []
        )
        final_search_ms = 0.0
        final_hit_count = 0
        if final_ids:
            final_started = perf_counter()
            final_results = self._search_many(
                query_vectors,
                final_ids,
                cutoff_time=cutoff_time,
                group_size=self._final_segment_group_size,
            )
            final_search_ms = (perf_counter() - final_started) * 1000
            final_hit_count = _search_hit_count(final_results)
            self._merge_search_results(
                segments,
                requirements,
                requirement_slices,
                final_results,
            )
            vector_load_ms += self._load_missing_vectors(segments, vectors_by_point)

        raw_by_requirement: dict[str, dict[str, list[_ReviewAggregate]]] = {}
        requested_review_ids: list[str] = []
        for requirement in requirements:
            start, positive_end, end = requirement_slices[requirement.requirement_id]
            positive_vectors = query_vectors[start:positive_end]
            negative_vectors = query_vectors[positive_end:end]
            by_business: dict[str, list[_ReviewAggregate]] = {}
            for business_id in business_ids:
                aggregates = self._aggregate_reviews(
                    list(segments[requirement.requirement_id][business_id].values()),
                    positive_vectors,
                    negative_vectors,
                    vectors_by_point,
                )
                selected = self._select_each_side(aggregates)
                by_business[business_id] = selected
                requested_review_ids.extend(item.review_id for item in selected)
            raw_by_requirement[requirement.requirement_id] = by_business

        full_review_started = perf_counter()
        full_texts = self._full_reviews.get_many(requested_review_ids)
        full_review_ms = (perf_counter() - full_review_started) * 1000
        output: dict[
            str, dict[str, list[ReviewSimilarityCandidate]]
        ] = {}
        for requirement in requirements:
            by_business = {}
            for business_id in business_ids:
                candidates = [
                    self._materialize(item, full_texts[item.review_id])
                    for item in raw_by_requirement[requirement.requirement_id].get(
                        business_id, []
                    )
                ]
                by_business[business_id] = self._deduplicate_full_reviews(candidates)
            output[requirement.requirement_id] = by_business

        return ReviewRetrievalBatch(
            by_requirement=output,
            metrics=ReviewRetrievalMetrics(
                query_vector_count=len(query_vectors),
                embedding_batch_count=embedding_batches,
                embedding_latency_ms=embedding_latency_ms,
                first_pass_search_latency_ms=first_search_ms,
                middle_pass_search_latency_ms=middle_search_ms,
                final_pass_search_latency_ms=final_search_ms,
                local_vector_load_latency_ms=vector_load_ms,
                full_review_load_latency_ms=full_review_ms,
                first_pass_segment_hit_count=_search_hit_count(first_results),
                middle_pass_segment_hit_count=middle_hit_count,
                final_pass_segment_hit_count=final_hit_count,
                middle_pass_business_count=len(middle_ids),
                final_pass_business_count=len(final_ids),
                loaded_vector_count=len(vectors_by_point),
                requirement_segment_relation_count=sum(
                    len(hits)
                    for requirement_map in segments.values()
                    for hits in requirement_map.values()
                ),
                unique_segment_count=len(vectors_by_point),
                full_review_count=len(full_texts),
            ),
        )

    def _encode_all(
        self,
        descriptions: list[str],
    ) -> tuple[np.ndarray, float, int]:
        """按本地模型的批量上限编码，但不再按偏好重复调用。"""

        batch_size = int(getattr(self._encoder, "batch_size", 16))
        vectors: list[np.ndarray] = []
        latency_ms = 0.0
        batch_count = 0
        for offset in range(0, len(descriptions), batch_size):
            encoded = self._encoder.encode(
                descriptions[offset : offset + batch_size],
                input_type="query",
            )
            vectors.extend(np.asarray(item, dtype=np.float32) for item in encoded.vectors)
            latency_ms += float(getattr(encoded, "latency_ms", 0.0))
            batch_count += 1
        return _normalize_matrix(np.asarray(vectors, dtype=np.float32)), latency_ms, batch_count

    def _search_many(
        self,
        query_vectors: np.ndarray,
        business_ids: list[str],
        *,
        cutoff_time: datetime | None,
        group_size: int,
    ) -> list[dict[str, list[QdrantSegmentHit]]]:
        method = getattr(self._store, "search_grouped_many", None)
        if callable(method):
            return method(
                [vector for vector in query_vectors],
                business_ids,
                score_threshold=self.recall_threshold,
                cutoff_time=cutoff_time,
                group_size=group_size,
                max_concurrency=self._search_concurrency,
            )
        # 小型测试替身和旧调用方仍可以只实现单次查询接口。
        return [
            self._store.search_grouped(
                vector,
                business_ids,
                score_threshold=self.recall_threshold,
                cutoff_time=cutoff_time,
                group_size=group_size,
            )
            for vector in query_vectors
        ]

    @staticmethod
    def _empty_segment_map(
        requirements: list[PreferenceSearchDescription],
        business_ids: list[str],
    ) -> dict[str, dict[str, dict[str, QdrantSegmentHit]]]:
        return {
            requirement.requirement_id: {
                business_id: {} for business_id in business_ids
            }
            for requirement in requirements
        }

    @staticmethod
    def _merge_search_results(
        target: dict[str, dict[str, dict[str, QdrantSegmentHit]]],
        requirements: list[PreferenceSearchDescription],
        requirement_slices: dict[str, tuple[int, int, int]],
        search_results: list[dict[str, list[QdrantSegmentHit]]],
    ) -> None:
        for requirement in requirements:
            start, _, end = requirement_slices[requirement.requirement_id]
            for grouped in search_results[start:end]:
                for business_id, hits in grouped.items():
                    destination = target[requirement.requirement_id].get(business_id)
                    if destination is None:
                        continue
                    for hit in hits:
                        previous = destination.get(hit.segment_id)
                        if previous is None or hit.route_similarity > previous.route_similarity:
                            destination[hit.segment_id] = hit

    def _businesses_needing_expansion(
        self,
        requirements: list[PreferenceSearchDescription],
        business_ids: list[str],
        requirement_slices: dict[str, tuple[int, int, int]],
        query_vectors: np.ndarray,
        segments: dict[str, dict[str, dict[str, QdrantSegmentHit]]],
        current_results: list[dict[str, list[QdrantSegmentHit]]],
        *,
        group_size: int,
        vectors_by_point: dict[int, np.ndarray],
    ) -> list[str]:
        """只有当前一档截满且明确证据不足的商家才进入下一档。"""

        needed: set[str] = set()
        for requirement in requirements:
            start, positive_end, end = requirement_slices[requirement.requirement_id]
            for business_id in business_ids:
                hit_limit_reached = any(
                    len(grouped.get(business_id, []))
                    >= group_size
                    for grouped in current_results[start:end]
                )
                if not hit_limit_reached:
                    continue
                aggregates = self._aggregate_reviews(
                    list(segments[requirement.requirement_id][business_id].values()),
                    query_vectors[start:positive_end],
                    query_vectors[positive_end:end],
                    vectors_by_point,
                )
                clear_count = sum(
                    self._direction(item.positive, item.negative) != "ambiguous"
                    for item in aggregates
                )
                if clear_count < self._minimum_clear_evidence:
                    needed.add(business_id)
        return [business_id for business_id in business_ids if business_id in needed]

    def _aggregate_reviews(
        self,
        hits: list[QdrantSegmentHit],
        positive_vectors: np.ndarray,
        negative_vectors: np.ndarray,
        vectors_by_point: dict[int, np.ndarray],
    ) -> list[_ReviewAggregate]:
        by_review: dict[str, _ReviewAggregate] = {}
        for hit in hits:
            vector = np.asarray(vectors_by_point[hit.point_id], dtype=np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            positive = float((positive_vectors @ vector).max())
            negative = float((negative_vectors @ vector).max())
            aggregate = by_review.get(hit.review_id)
            if aggregate is None:
                aggregate = _ReviewAggregate(hit)
                by_review[hit.review_id] = aggregate
            aggregate.add(hit, positive=positive, negative=negative)
        return list(by_review.values())

    def _load_missing_vectors(
        self,
        segments: dict[str, dict[str, dict[str, QdrantSegmentHit]]],
        destination: dict[int, np.ndarray],
    ) -> float:
        """跨偏好、正反说法和商家统一去重后，再从本地文件读取。"""

        point_ids = [
            hit.point_id
            for requirement_map in segments.values()
            for business_map in requirement_map.values()
            for hit in business_map.values()
            if hit.point_id not in destination
        ]
        unique_ids = list(dict.fromkeys(point_ids))
        if not unique_ids:
            return 0.0
        started = perf_counter()
        destination.update(self._segment_vectors.get_many(unique_ids))
        return (perf_counter() - started) * 1000

    def _select_each_side(
        self,
        aggregates: list[_ReviewAggregate],
    ) -> list[_ReviewAggregate]:
        positive = sorted(
            (item for item in aggregates if item.positive >= self.recall_threshold),
            key=lambda item: (-item.positive, -item.useful, item.review_id),
        )[: self._recall_each_side]
        negative = sorted(
            (item for item in aggregates if item.negative >= self.recall_threshold),
            key=lambda item: (-item.negative, -item.useful, item.review_id),
        )[: self._recall_each_side]
        selected = {item.review_id: item for item in positive}
        selected.update({item.review_id: item for item in negative})
        return list(selected.values())

    def _materialize(
        self,
        item: _ReviewAggregate,
        review_text: str,
    ) -> ReviewSimilarityCandidate:
        return ReviewSimilarityCandidate(
            review_id=item.review_id,
            business_id=item.business_id,
            user_id=item.user_id,
            review_time=item.review_time,
            stars=item.stars,
            useful=item.useful,
            review_text=review_text,
            review_text_sha256=item.review_text_sha256,
            matched_segment_id=item.best_hit.segment_id,
            matched_segment_text=item.best_hit.segment_text,
            positive_similarity=item.positive,
            negative_similarity=item.negative,
            direction=self._direction(item.positive, item.negative),
        )

    def _direction(self, positive: float, negative: float) -> EvidenceDirection:
        if (
            positive >= self.acceptance_threshold
            and positive - negative >= self.direction_margin
        ):
            return "positive"
        if (
            negative >= self.acceptance_threshold
            and negative - positive >= self.direction_margin
        ):
            return "negative"
        return "ambiguous"

    @staticmethod
    def _deduplicate_full_reviews(
        candidates: list[ReviewSimilarityCandidate],
    ) -> list[ReviewSimilarityCandidate]:
        """评论编号不同但正文相同或只差标点空白时，也只算一条。"""

        ordered = sorted(
            candidates,
            key=lambda item: (
                -max(item.positive_similarity, item.negative_similarity),
                -item.useful,
                item.review_id,
            ),
        )
        seen_exact: set[str] = set()
        seen_normalized: set[str] = set()
        result: list[ReviewSimilarityCandidate] = []
        for item in ordered:
            normalized = re.sub(r"[^\w]+", "", item.review_text.casefold())
            normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if (
                item.review_text_sha256 in seen_exact
                or normalized_hash in seen_normalized
            ):
                continue
            seen_exact.add(item.review_text_sha256)
            seen_normalized.add(normalized_hash)
            result.append(item)
        return result


class _ReviewAggregate:
    """片段级相似度合并到原评论级；正反最高分可以来自不同片段。"""

    def __init__(self, hit: QdrantSegmentHit) -> None:
        self.review_id = hit.review_id
        self.business_id = hit.business_id
        self.user_id = hit.user_id
        self.review_time = hit.review_time
        self.stars = hit.stars
        self.useful = hit.useful
        self.review_text_sha256 = hit.review_text_sha256
        self.positive = -1.0
        self.negative = -1.0
        self.best_hit = hit
        self._best_similarity = -1.0

    def add(self, hit: QdrantSegmentHit, *, positive: float, negative: float) -> None:
        self.positive = max(self.positive, positive)
        self.negative = max(self.negative, negative)
        best = max(positive, negative)
        if best > self._best_similarity:
            self._best_similarity = best
            self.best_hit = hit


def _normalize_matrix(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _search_hit_count(
    results: list[dict[str, list[QdrantSegmentHit]]],
) -> int:
    """统计查询返回关系数；同一片段被不同说法命中会分别计数。"""

    return sum(len(hits) for grouped in results for hits in grouped.values())
