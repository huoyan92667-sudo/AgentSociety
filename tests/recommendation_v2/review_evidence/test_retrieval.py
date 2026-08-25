from datetime import UTC, datetime
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
    def encode(self, texts: list[str], *, input_type: str) -> object:
        assert input_type == "query"
        return SimpleNamespace(
            vectors=tuple(
                np.asarray([1.0, 0.0]) if "positive" in text else np.asarray([0.0, 1.0])
                for text in texts
            )
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


def _hit(review_id: str, vector: list[float], character: str) -> QdrantSegmentHit:
    return QdrantSegmentHit(
        point_id=len(review_id),
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
        vector=vector,
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
