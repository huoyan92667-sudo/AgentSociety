from datetime import UTC, datetime

import numpy as np
from qdrant_client import QdrantClient, models

from yelp_agent.recommendation_v2.review_evidence.qdrant_store import (
    QdrantReviewSegmentStore,
    _fuse_grouped_results,
)
from yelp_agent.recommendation_v2.review_evidence.schema import QdrantSegmentHit


def _payload(segment: str, review: str, business: str, timestamp: int) -> dict[str, object]:
    return {
        "segment_id": segment * 64,
        "review_id": review,
        "business_id": business,
        "user_id": f"user-{review}",
        "review_time": datetime.fromtimestamp(timestamp, tz=UTC).isoformat(),
        "review_timestamp": timestamp,
        "stars": 4.0,
        "useful": 1,
        "segment_index": 0,
        "segment_text": f"text for {review}",
        "review_text_sha256": "f" * 64,
    }


def test_search_is_filtered_and_grouped_per_business() -> None:
    review_timestamp = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    client = QdrantClient(":memory:")
    client.create_collection(
        "test",
        vectors_config={
            "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
    )
    client.upsert(
        "test",
        points=[
            models.PointStruct(
                id=1,
                vector={"dense": [1.0, 0.0]},
                payload=_payload("a", "review-1", "business-1", review_timestamp),
            ),
            models.PointStruct(
                id=2,
                vector={"dense": [0.9, 0.1]},
                payload=_payload("b", "review-2", "business-2", review_timestamp),
            ),
            models.PointStruct(
                id=3,
                vector={"dense": [1.0, 0.0]},
                payload=_payload("c", "review-3", "outside", review_timestamp),
            ),
        ],
        wait=True,
    )
    store = QdrantReviewSegmentStore(client, collection_name="test")

    result = store.search_grouped(
        np.asarray([1.0, 0.0]),
        ["business-1", "business-2"],
        score_threshold=0.55,
        cutoff_time=datetime(2021, 1, 1, tzinfo=UTC),
        group_size=2,
    )

    assert [item.review_id for item in result["business-1"]] == ["review-1"]
    assert [item.review_id for item in result["business-2"]] == ["review-2"]
    assert "outside" not in result
    store.close()


def test_rrf_fusion_rewards_segments_found_by_both_routes() -> None:
    timestamp = datetime(2020, 1, 1, tzinfo=UTC)

    def hit(review_id: str, character: str) -> QdrantSegmentHit:
        return QdrantSegmentHit(
            point_id=ord(character),
            segment_id=character * 64,
            review_id=review_id,
            business_id="business-1",
            user_id="user-1",
            review_time=timestamp,
            stars=4,
            useful=0,
            segment_index=0,
            segment_text=review_id,
            review_text_sha256="f" * 64,
            route_similarity=0.8,
        )

    dense = [{"business-1": [hit("dense-only", "a"), hit("both", "b")]}]
    bm25 = [{"business-1": [hit("both", "b"), hit("bm25-only", "c")]}]

    fused = _fuse_grouped_results(
        dense,
        bm25,
        ["business-1"],
        group_size=3,
        rrf_k=60,
    )[0]["business-1"]

    assert [item.review_id for item in fused] == [
        "both",
        "dense-only",
        "bm25-only",
    ]
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[1].dense_rank == 1
    assert fused[1].bm25_rank is None
    assert fused[2].dense_rank is None
    assert fused[2].bm25_rank == 2


def test_segment_hit_accepts_keyword_scores_above_one() -> None:
    hit = QdrantSegmentHit(
        point_id=1,
        segment_id="a" * 64,
        review_id="review-1",
        business_id="business-1",
        user_id="user-1",
        review_time=datetime(2020, 1, 1, tzinfo=UTC),
        stars=4,
        useful=0,
        segment_index=0,
        segment_text="authentic Szechuan",
        review_text_sha256="f" * 64,
        route_similarity=33.38567,
    )

    assert hit.route_similarity == 33.38567
