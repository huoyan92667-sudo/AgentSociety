from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace

import numpy as np

from yelp_agent.recommendation_v2.review_evidence.retrieval import (
    ReviewEvidenceRetriever,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
    QdrantSegmentHit,
)


class _Encoder:
    batch_size = 16

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], *, input_type: str) -> object:
        assert input_type == "query"
        self.calls.append(list(texts))
        return SimpleNamespace(
            vectors=tuple(
                np.asarray([1.0, 0.0]) if "positive" in text else np.asarray([0.0, 1.0])
                for text in texts
            ),
            latency_ms=2.0,
        )

    def close(self) -> None:
        pass


class _Store:
    def __init__(self, hits: list[QdrantSegmentHit]) -> None:
        self.hits = hits

    def search_grouped(self, *args: object, **kwargs: object) -> dict[str, list[QdrantSegmentHit]]:
        return {"business-1": self.hits}

    def close(self) -> None:
        pass


class _FullReviews:
    def get_many(self, review_ids: list[str]) -> dict[str, str]:
        return {review_id: f"full text {review_id}" for review_id in set(review_ids)}

    def close(self) -> None:
        pass


_TEST_VECTORS: dict[int, np.ndarray] = {}


class _VectorStore:
    """模拟按 Qdrant 点编号从本地向量文件批量读取。"""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def get_many(self, point_ids: list[int]) -> dict[int, np.ndarray]:
        self.calls.append(list(point_ids))
        return {point_id: _TEST_VECTORS[point_id] for point_id in point_ids}

    def close(self) -> None:
        pass


class _BatchStore:
    """记录批量查询规模，并可模拟前两档都被截满。"""

    def __init__(self, *, truncate_first_pass: bool = False) -> None:
        self.truncate_first_pass = truncate_first_pass
        self.calls: list[dict[str, object]] = []

    def search_grouped_many(
        self,
        query_vectors: list[np.ndarray],
        business_ids: list[str],
        **kwargs: object,
    ) -> list[dict[str, list[QdrantSegmentHit]]]:
        self.calls.append(
            {
                "query_count": len(query_vectors),
                "business_ids": list(business_ids),
                **kwargs,
            }
        )
        group_size = int(kwargs["group_size"])
        if not self.truncate_first_pass or group_size not in {15, 30}:
            return [{business_id: [] for business_id in business_ids} for _ in query_vectors]
        repeated = [
            _hit("same-review", [1.0, 0.0], "d").model_copy(
                update={"segment_id": sha256(f"segment-{index}".encode()).hexdigest()}
            )
            for index in range(group_size)
        ]
        return [
            {business_id: list(repeated) for business_id in business_ids}
            for _ in query_vectors
        ]

    def close(self) -> None:
        pass


def _hit(review_id: str, vector: list[float], character: str) -> QdrantSegmentHit:
    point_id = int(sha256(f"{review_id}:{character}".encode()).hexdigest()[:8], 16)
    _TEST_VECTORS[point_id] = np.asarray(vector, dtype=np.float32)
    return QdrantSegmentHit(
        point_id=point_id,
        segment_id=character * 64,
        review_id=review_id,
        business_id="business-1",
        user_id=f"user-{review_id}",
        review_time=datetime(2022, 1, 1, tzinfo=UTC),
        stars=4,
        useful=1,
        segment_index=0,
        segment_text=f"segment {review_id}",
        review_text_sha256=character * 64,
        route_similarity=0.9,
    )


def test_direct_p_n_judgment_keeps_ambiguous_out_of_evidence_roles() -> None:
    retriever = ReviewEvidenceRetriever(
        store=_Store(
            [
                _hit("positive", [1.0, 0.0], "a"),
                _hit("negative", [0.0, 1.0], "b"),
                _hit("ambiguous", [0.71, 0.70], "c"),
            ]
        ),  # type: ignore[arg-type]
        encoder=_Encoder(),
        segment_vectors=_VectorStore(),
        full_reviews=_FullReviews(),  # type: ignore[arg-type]
    )
    requirement = PreferenceSearchDescription(
        requirement_id="tail",
        requirement_text="tail",
        kind="long_tail",
        priority=1,
        preference_strength=100,
        positive_descriptions=["positive one", "positive two"],
        negative_descriptions=["negative one", "negative two"],
    )

    result = retriever.retrieve(
        requirement,
        ["business-1"],
        cutoff_time=datetime(2023, 1, 1, tzinfo=UTC),
    )["business-1"]

    assert {item.review_id: item.direction for item in result} == {
        "positive": "positive",
        "negative": "negative",
        "ambiguous": "ambiguous",
    }


def test_multiple_requirements_share_one_embedding_and_batched_search() -> None:
    encoder = _Encoder()
    store = _BatchStore()
    retriever = ReviewEvidenceRetriever(
        store=store,  # type: ignore[arg-type]
        encoder=encoder,
        segment_vectors=_VectorStore(),
        full_reviews=_FullReviews(),  # type: ignore[arg-type]
    )
    requirements = [
        PreferenceSearchDescription(
            requirement_id=f"requirement-{index}",
            requirement_text=f"requirement-{index}",
            kind="long_tail",
            priority=index,
            preference_strength=100,
            positive_descriptions=["positive one", "positive two"],
            negative_descriptions=["negative one", "negative two"],
        )
        for index in (1, 2)
    ]
    current_time = datetime(2026, 8, 26, tzinfo=UTC)

    batch = retriever.retrieve_many(
        requirements,
        ["business-1"],
        cutoff_time=current_time,
    )

    assert len(encoder.calls) == 1
    assert len(encoder.calls[0]) == 8
    assert len(store.calls) == 1
    assert store.calls[0]["query_count"] == 8
    assert store.calls[0]["group_size"] == 15
    assert store.calls[0]["cutoff_time"] == current_time
    assert batch.metrics.query_vector_count == 8
    assert batch.metrics.embedding_batch_count == 1
    assert batch.metrics.middle_pass_business_count == 0


def test_only_truncated_business_with_too_little_clear_evidence_is_supplemented() -> None:
    store = _BatchStore(truncate_first_pass=True)
    retriever = ReviewEvidenceRetriever(
        store=store,  # type: ignore[arg-type]
        encoder=_Encoder(),
        segment_vectors=_VectorStore(),
        full_reviews=_FullReviews(),  # type: ignore[arg-type]
    )
    requirement = PreferenceSearchDescription(
        requirement_id="tail",
        requirement_text="tail",
        kind="long_tail",
        priority=1,
        preference_strength=100,
        positive_descriptions=["positive one", "positive two"],
        negative_descriptions=["negative one", "negative two"],
    )

    batch = retriever.retrieve_many(
        [requirement],
        ["business-1"],
        cutoff_time=datetime(2026, 8, 26, tzinfo=UTC),
    )

    assert [call["group_size"] for call in store.calls] == [15, 30, 60]
    assert store.calls[1]["business_ids"] == ["business-1"]
    assert batch.metrics.middle_pass_business_count == 1
    assert batch.metrics.final_pass_business_count == 1


def test_yelp_review_time_without_timezone_is_treated_as_utc() -> None:
    hit = QdrantSegmentHit(
        point_id=1,
        segment_id="e" * 64,
        review_id="review-naive-time",
        business_id="business-1",
        user_id="user-1",
        review_time=datetime(2022, 1, 1),
        stars=4,
        useful=0,
        segment_index=0,
        segment_text="A useful review segment.",
        review_text_sha256="f" * 64,
        route_similarity=0.8,
    )

    assert hit.review_time.tzinfo is UTC
