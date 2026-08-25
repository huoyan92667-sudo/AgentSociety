from datetime import UTC, datetime

import numpy as np
from qdrant_client import QdrantClient, models

from yelp_agent.recommendation_v2.review_evidence.qdrant_store import (
    QdrantReviewSegmentStore,
)


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
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        "test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload=_payload("a", "review-1", "business-1", review_timestamp),
            ),
            models.PointStruct(
                id=2,
                vector=[0.9, 0.1],
                payload=_payload("b", "review-2", "business-2", review_timestamp),
            ),
            models.PointStruct(
                id=3,
                vector=[1.0, 0.0],
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
