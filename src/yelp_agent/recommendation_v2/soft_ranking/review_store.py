"""在当前餐厅的全部评论向量里，分别寻找满足和违反要求的评论。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Protocol

import duckdb
import numpy as np

from yelp_agent.semantic_embedding.encoder import EncodedBatch

from .schema import RetrievedReview

_RECOMMENDATION_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_INDEX_ROOT = _RECOMMENDATION_ROOT / "data" / "review_index" / "v1"
_FULL_REVIEW_SOURCE = _PROJECT_ROOT / "data" / "processed" / "reviews.parquet"


class QueryEncoder(Protocol):
    """只编码少量检索说法；全部评论向量已经离线保存。"""

    def encode(self, texts: list[str], *, input_type: str) -> EncodedBatch: ...

    def close(self) -> None: ...


class _CachedReview:
    """一条完整评论及它在向量文件中的全部片段行号。"""

    __slots__ = (
        "business_id",
        "review_id",
        "review_text",
        "review_time",
        "row_indices",
        "stars",
        "useful",
    )

    def __init__(self, row: dict[str, object]) -> None:
        self.business_id = str(row["business_id"])
        self.review_id = str(row["review_id"])
        self.review_time = row["review_time"].isoformat()  # type: ignore[union-attr]
        self.stars = float(row["stars"])
        self.useful = int(row["useful"])
        self.review_text = str(row["review_text"])
        self.row_indices: list[int] = []


class ReviewVectorStore:
    """限定商家范围后，用正反说法在全部评论片段中各取最多五条。"""

    def __init__(
        self,
        encoder: QueryEncoder,
        *,
        index_root: str | Path = _INDEX_ROOT,
        full_review_source: str | Path = _FULL_REVIEW_SOURCE,
        similarity_threshold: float = 0.55,
        full_weight_similarity: float = 0.82,
    ) -> None:
        if not -1 <= similarity_threshold < full_weight_similarity <= 1:
            raise ValueError("similarity thresholds are invalid")
        root = Path(index_root)
        segments = root / "review_segments.parquet"
        embeddings = root / "segment_embeddings.npy"
        reviews = Path(full_review_source)
        if not segments.is_file() or not embeddings.is_file() or not reviews.is_file():
            raise FileNotFoundError("full review vector artifacts are incomplete")
        self._encoder = encoder
        self._vectors = np.load(embeddings, mmap_mode="r")
        self._threshold = similarity_threshold
        self._full_weight_similarity = full_weight_similarity
        self._connection = duckdb.connect(database=":memory:")
        self._lock = Lock()
        self._closed = False
        self._cache: dict[str, list[_CachedReview]] = {}
        segment_path = str(segments.resolve()).replace("'", "''")
        review_path = str(reviews.resolve()).replace("'", "''")
        self._connection.execute(
            "CREATE VIEW review_segments AS "
            f"SELECT * FROM read_parquet('{segment_path}')"
        )
        self._connection.execute(
            f"CREATE VIEW full_reviews AS SELECT * FROM read_parquet('{review_path}')"
        )

    def close(self) -> None:
        """关闭数据库连接和本地向量模型进程。"""

        if not self._closed:
            self._connection.close()
            self._encoder.close()
            self._closed = True

    def retrieve(
        self,
        business_ids: list[str],
        satisfying_anchors: list[str],
        contradicting_anchors: list[str],
        *,
        limit_each_side: int = 5,
    ) -> dict[str, list[RetrievedReview]]:
        """正面最多五条、反面最多五条；同一评论同时命中时只返回一次。"""

        if self._closed:
            raise RuntimeError("review vector store is closed")
        if not business_ids:
            return {}
        if len(business_ids) != len(set(business_ids)):
            raise ValueError("review retrieval business IDs must be unique")
        if len(satisfying_anchors) != 2 or len(contradicting_anchors) != 2:
            raise ValueError("each retrieval side requires exactly two anchors")
        if limit_each_side < 1:
            raise ValueError("review limit must be positive")

        self._ensure_cached(business_ids)
        encoded = self._encoder.encode(
            [*satisfying_anchors, *contradicting_anchors],
            input_type="query",
        )
        query_vectors = np.asarray(encoded.vectors, dtype=np.float32)
        query_vectors /= np.maximum(
            np.linalg.norm(query_vectors, axis=1, keepdims=True),
            1e-12,
        )
        output: dict[str, list[RetrievedReview]] = {}
        for business_id in business_ids:
            output[business_id] = self._retrieve_business(
                self._cache.get(business_id, []),
                query_vectors,
                limit_each_side=limit_each_side,
            )
        return output

    def _ensure_cached(self, business_ids: list[str]) -> None:
        missing = [value for value in business_ids if value not in self._cache]
        if not missing:
            return
        marks = ",".join("?" for _ in missing)
        query = f"""
            SELECT
                s.row_index,
                s.review_id,
                s.business_id,
                s.review_time,
                s.stars,
                s.useful,
                r.text AS review_text
            FROM review_segments s
            JOIN full_reviews r USING (review_id, business_id)
            WHERE s.business_id IN ({marks})
            ORDER BY s.business_id, s.review_id, s.segment_index
        """
        with self._lock:
            rows = self._connection.execute(query, missing).to_arrow_table().to_pylist()
        grouped: dict[tuple[str, str], _CachedReview] = {}
        for row in rows:
            key = (str(row["business_id"]), str(row["review_id"]))
            review = grouped.setdefault(key, _CachedReview(row))
            review.row_indices.append(int(row["row_index"]))
        by_business: dict[str, list[_CachedReview]] = defaultdict(list)
        for review in grouped.values():
            by_business[review.business_id].append(review)
        for business_id in missing:
            self._cache[business_id] = by_business.get(business_id, [])

    def _retrieve_business(
        self,
        reviews: list[_CachedReview],
        query_vectors: np.ndarray,
        *,
        limit_each_side: int,
    ) -> list[RetrievedReview]:
        if not reviews:
            return []
        row_indices = [index for review in reviews for index in review.row_indices]
        matrix = np.asarray(self._vectors[row_indices], dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        similarities = matrix @ query_vectors.T

        scored: list[tuple[_CachedReview, float, float]] = []
        offset = 0
        for review in reviews:
            count = len(review.row_indices)
            review_scores = similarities[offset : offset + count]
            offset += count
            positive = float(review_scores[:, :2].max())
            negative = float(review_scores[:, 2:].max())
            scored.append((review, positive, negative))

        positive_hits = sorted(
            (item for item in scored if item[1] >= self._threshold),
            key=lambda item: (-item[1], -item[0].useful, item[0].review_id),
        )[:limit_each_side]
        negative_hits = sorted(
            (item for item in scored if item[2] >= self._threshold),
            key=lambda item: (-item[2], -item[0].useful, item[0].review_id),
        )[:limit_each_side]
        selected: dict[str, tuple[_CachedReview, float, float, set[str]]] = {}
        for side, hits in (("positive", positive_hits), ("negative", negative_hits)):
            for review, positive, negative in hits:
                existing = selected.get(review.review_id)
                if existing is None:
                    selected[review.review_id] = (
                        review,
                        positive,
                        negative,
                        {side},
                    )
                else:
                    existing[3].add(side)

        result: list[RetrievedReview] = []
        for review, positive, negative, sides in selected.values():
            result.append(
                RetrievedReview(
                    review_id=review.review_id,
                    business_id=review.business_id,
                    review_time=review.review_time,
                    stars=review.stars,
                    useful=review.useful,
                    review_text=review.review_text,
                    retrieval_side="both" if len(sides) == 2 else next(iter(sides)),
                    positive_similarity=positive,
                    negative_similarity=negative,
                    positive_weight=self._evidence_weight(positive),
                    negative_weight=self._evidence_weight(negative),
                )
            )
        return sorted(
            result,
            key=lambda item: (
                -max(item.positive_similarity or -1, item.negative_similarity or -1),
                item.review_id,
            ),
        )

    def _evidence_weight(self, similarity: float) -> float:
        """把已过阈值的相似度换成0到1的证据权重。"""

        if similarity < self._threshold:
            return 0.0
        return min(
            1.0,
            (similarity - self._threshold)
            / (self._full_weight_similarity - self._threshold),
        )


# 保留旧名字，避免只引用类型名的外部代码立刻中断。
ReviewCandidateStore = ReviewVectorStore
