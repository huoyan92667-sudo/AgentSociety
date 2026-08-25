"""把正反描述变成向量，并从硬筛商家范围内召回、合并和判定评论。"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Protocol

import numpy as np

from yelp_agent.semantic_embedding.encoder import EncodedBatch

from .full_reviews import FullReviewStore
from .qdrant_store import QdrantReviewSegmentStore
from .schema import (
    EvidenceDirection,
    PreferenceSearchDescription,
    QdrantSegmentHit,
    ReviewSimilarityCandidate,
)


class QueryEncoder(Protocol):
    """只负责把少量正反检索说法编码成查询向量。"""

    def encode(self, texts: list[str], *, input_type: str) -> EncodedBatch: ...

    def close(self) -> None: ...


class ReviewEvidenceRetriever:
    """每家先各取正反候选，再按 P、N 差值直接判定方向。"""

    def __init__(
        self,
        *,
        store: QdrantReviewSegmentStore,
        encoder: QueryEncoder,
        full_reviews: FullReviewStore,
        recall_threshold: float = 0.55,
        acceptance_threshold: float = 0.60,
        direction_margin: float = 0.05,
        recall_each_side: int = 15,
        segment_group_size: int = 60,
    ) -> None:
        if not -1 <= recall_threshold <= acceptance_threshold <= 1:
            raise ValueError("review similarity thresholds are invalid")
        if not 0 <= direction_margin <= 2:
            raise ValueError("direction margin is invalid")
        if recall_each_side < 1 or segment_group_size < recall_each_side:
            raise ValueError("review recall limits are invalid")
        self._store = store
        self._encoder = encoder
        self._full_reviews = full_reviews
        self.recall_threshold = recall_threshold
        self.acceptance_threshold = acceptance_threshold
        self.direction_margin = direction_margin
        self._recall_each_side = recall_each_side
        self._segment_group_size = segment_group_size

    def close(self) -> None:
        self._full_reviews.close()
        self._encoder.close()
        self._store.close()

    def retrieve(
        self,
        requirement: PreferenceSearchDescription,
        business_ids: list[str],
        *,
        cutoff_time: datetime | None = None,
    ) -> dict[str, list[ReviewSimilarityCandidate]]:
        """返回每家最多30条去重评论，模糊项仍保留计数但不参与证据分。"""

        if not business_ids:
            return {}
        descriptions = [
            *requirement.positive_descriptions,
            *requirement.negative_descriptions,
        ]
        encoded = self._encoder.encode(descriptions, input_type="query")
        query_vectors = _normalize_matrix(np.asarray(encoded.vectors, dtype=np.float32))
        positive_count = len(requirement.positive_descriptions)
        positive_vectors = query_vectors[:positive_count]
        negative_vectors = query_vectors[positive_count:]

        # 每个说法独立查一次，再合并片段。这样某个具体表达不会被另一个宽泛表达淹没。
        by_business_segment: dict[str, dict[str, QdrantSegmentHit]] = {
            business_id: {} for business_id in business_ids
        }
        for query_vector in query_vectors:
            grouped = self._store.search_grouped(
                query_vector,
                business_ids,
                score_threshold=self.recall_threshold,
                cutoff_time=cutoff_time,
                group_size=self._segment_group_size,
            )
            for business_id, hits in grouped.items():
                for hit in hits:
                    previous = by_business_segment[business_id].get(hit.segment_id)
                    if previous is None or hit.route_similarity > previous.route_similarity:
                        by_business_segment[business_id][hit.segment_id] = hit

        raw_reviews: dict[str, list[_ReviewAggregate]] = {}
        requested_review_ids: list[str] = []
        for business_id, hits_by_id in by_business_segment.items():
            aggregates = self._aggregate_reviews(
                list(hits_by_id.values()),
                positive_vectors,
                negative_vectors,
            )
            selected = self._select_each_side(aggregates)
            raw_reviews[business_id] = selected
            requested_review_ids.extend(item.review_id for item in selected)

        full_texts = self._full_reviews.get_many(requested_review_ids)
        output: dict[str, list[ReviewSimilarityCandidate]] = {}
        for business_id in business_ids:
            candidates = [
                self._materialize(item, full_texts[item.review_id])
                for item in raw_reviews.get(business_id, [])
            ]
            output[business_id] = self._deduplicate_full_reviews(candidates)
        return output

    def _aggregate_reviews(
        self,
        hits: list[QdrantSegmentHit],
        positive_vectors: np.ndarray,
        negative_vectors: np.ndarray,
    ) -> list[_ReviewAggregate]:
        by_review: dict[str, _ReviewAggregate] = {}
        for hit in hits:
            vector = np.asarray(hit.vector, dtype=np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            positive = float((positive_vectors @ vector).max())
            negative = float((negative_vectors @ vector).max())
            aggregate = by_review.get(hit.review_id)
            if aggregate is None:
                aggregate = _ReviewAggregate(hit)
                by_review[hit.review_id] = aggregate
            aggregate.add(hit, positive=positive, negative=negative)
        return list(by_review.values())

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
