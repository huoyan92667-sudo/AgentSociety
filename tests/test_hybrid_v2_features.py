from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.business_profiles.schema import (
    BUSINESS_ASPECT_EVENT_SCHEMA,
    BUSINESS_RATING_EVENT_SCHEMA,
)
from yelp_agent.config import load_business_profile_config
from yelp_agent.learning_to_rank.features import (
    HybridV1Weights,
    HybridV2FeatureSources,
    build_hybrid_v2_features,
)
from yelp_agent.profiles.schema import (
    PREFERENCE_SIGNAL_SCHEMA,
    PROFILE_SNAPSHOT_SCHEMA,
    TASK_PROFILE_LINK_SCHEMA,
)
from yelp_agent.retrieval.benchmark import CANDIDATE_SCHEMA


def _candidate(business_id: str, rank: int) -> dict[str, object]:
    return {
        "task_id": "validation:u1",
        "rank": rank,
        "business_id": business_id,
        "fusion_score": 1.0 / rank,
        "route_count": 4,
        "quality_rank": rank,
        "quality_score": 0.7,
        "category_rank": rank,
        "category_score": 0.6,
        "text_rank": rank,
        "text_score": 0.5,
        "location_rank": rank,
        "location_score": 0.4,
        "distance_km": 2.0,
        "item_knn_rank": rank,
        "item_knn_positive_score": 0.3,
        "item_knn_negative_evidence": 0.1,
        "item_knn_positive_support_count": 4,
        "item_knn_negative_support_count": 1,
        "item_knn_positive_neighbor_count": 2,
        "item_knn_negative_neighbor_count": 1,
        "item_knn_missing": False,
    }


def _write_sources(root: Path, *, include_future: bool) -> HybridV2FeatureSources:
    cutoff = datetime(2020, 1, 10)
    candidates = root / "candidates.parquet"
    user_root = root / "users"
    business_root = root / "businesses"
    user_root.mkdir(parents=True)
    business_root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [_candidate("b-steak", 1), _candidate("b-other", 2)],
            schema=CANDIDATE_SCHEMA,
        ),
        candidates,
    )
    profile_id = "a" * 64
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "profile_id": profile_id,
                    "user_id": "u1",
                    "cutoff_time": cutoff,
                    "history_length": 10,
                    "average_rating": 4.0,
                    "rating_1_count": 0,
                    "rating_2_count": 1,
                    "rating_3_count": 1,
                    "rating_4_count": 5,
                    "rating_5_count": 3,
                    "location_latitude": 39.9,
                    "location_longitude": -75.1,
                    "reliability": 0.8,
                    "category_evidence_count": 8,
                    "aspect_evidence_count": 4,
                    "price_evidence_count": 0,
                    "area_evidence_count": 2,
                    "first_interaction": datetime(2019, 1, 1),
                    "last_interaction": datetime(2020, 1, 5),
                    "profile_version": "1.0.0",
                }
            ],
            schema=PROFILE_SNAPSHOT_SCHEMA,
        ),
        user_root / "profile_snapshots.parquet",
    )
    signals = [
        {
            "profile_id": profile_id,
            "user_id": "u1",
            "cutoff_time": cutoff,
            "kind": "category",
            "value": "Steakhouses",
            "score": 0.8,
            "confidence": 0.9,
            "evidence_count": 3,
            "effective_evidence": 2.5,
            "first_seen": datetime(2019, 1, 1),
            "last_confirmed": datetime(2020, 1, 5),
            "source": "rating_category",
        },
        {
            "profile_id": profile_id,
            "user_id": "u1",
            "cutoff_time": cutoff,
            "kind": "aspect",
            "value": "service",
            "score": 0.7,
            "confidence": 0.8,
            "evidence_count": 3,
            "effective_evidence": 2.0,
            "first_seen": datetime(2019, 1, 1),
            "last_confirmed": datetime(2020, 1, 5),
            "source": "review_aspect",
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(signals, schema=PREFERENCE_SIGNAL_SCHEMA),
        user_root / "preference_signals.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "validation:u1",
                    "split": "validation",
                    "user_id": "u1",
                    "cutoff_time": cutoff,
                    "profile_id": profile_id,
                    "expected_history_count": 10,
                    "fold": None,
                    "sample_weight": None,
                }
            ],
            schema=TASK_PROFILE_LINK_SCHEMA,
        ),
        user_root / "task_profile_map.parquet",
    )
    business_schema = pa.schema(
        [
            pa.field("business_id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("address", pa.string()),
            pa.field("city", pa.string()),
            pa.field("state", pa.string()),
            pa.field("postal_code", pa.string()),
            pa.field("latitude", pa.float64()),
            pa.field("longitude", pa.float64()),
            pa.field("categories", pa.list_(pa.string())),
            pa.field("attributes_json", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "business_id": "b-steak",
                    "name": "Steak",
                    "address": "",
                    "city": "Philadelphia",
                    "state": "PA",
                    "postal_code": "",
                    "latitude": 39.9,
                    "longitude": -75.1,
                    "categories": ["Restaurants", "Steakhouses"],
                    "attributes_json": "{}",
                },
                {
                    "business_id": "b-other",
                    "name": "Other",
                    "address": "",
                    "city": "Philadelphia",
                    "state": "PA",
                    "postal_code": "",
                    "latitude": 39.9,
                    "longitude": -75.1,
                    "categories": ["Restaurants", "Pizza"],
                    "attributes_json": "{}",
                },
            ],
            schema=business_schema,
        ),
        business_root / "businesses.parquet",
    )
    rating_rows = [
        {
            "review_id": "r1",
            "business_id": "b-steak",
            "user_id": "x1",
            "stars": 5.0,
            "review_time": datetime(2020, 1, 1),
        }
    ]
    aspect_rows = [
        {
            "review_id": f"a{index}",
            "business_id": "b-steak",
            "user_id": f"x{index}",
            "review_time": datetime(2020, 1, index),
            "aspect": "service",
            "sentiment": "positive",
            "confidence": 0.9,
            "source_text_sha256": f"{index}" * 64,
            "extractor_version": "1.2.0",
        }
        for index in range(1, 4)
    ]
    if include_future:
        rating_rows.append(
            {
                "review_id": "future-rating",
                "business_id": "b-steak",
                "user_id": "future",
                "stars": 1.0,
                "review_time": datetime(2020, 1, 20),
            }
        )
        aspect_rows.append(
            {
                "review_id": "future-aspect",
                "business_id": "b-steak",
                "user_id": "future",
                "review_time": datetime(2020, 1, 20),
                "aspect": "service",
                "sentiment": "negative",
                "confidence": 1.0,
                "source_text_sha256": "f" * 64,
                "extractor_version": "1.2.0",
            }
        )
    pq.write_table(
        pa.Table.from_pylist(rating_rows, schema=BUSINESS_RATING_EVENT_SCHEMA),
        business_root / "rating_events.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(aspect_rows, schema=BUSINESS_ASPECT_EVENT_SCHEMA),
        business_root / "aspect_events.parquet",
    )
    return HybridV2FeatureSources(
        candidates=candidates,
        user_profile_root=user_root,
        business_profile_root=business_root,
    )


def test_feature_builder_is_point_in_time_and_marks_missing_evidence(
    tmp_path: Path,
) -> None:
    weights = HybridV1Weights(category=0.0, text=0.5, quality=0.1, location=0.4)
    before = tmp_path / "before.parquet"
    after = tmp_path / "after.parquet"
    build_hybrid_v2_features(
        _write_sources(tmp_path / "without-future", include_future=False),
        before,
        split="validation",
        weights=weights,
        broad_categories={"Restaurants", "Food", "Nightlife", "Shopping"},
        business_profile_config=load_business_profile_config("configs"),
        batch_size=100,
    )
    build_hybrid_v2_features(
        _write_sources(tmp_path / "with-future", include_future=True),
        after,
        split="validation",
        weights=weights,
        broad_categories={"Restaurants", "Food", "Nightlife", "Shopping"},
        business_profile_config=load_business_profile_config("configs"),
        batch_size=100,
    )

    assert pq.read_table(before).to_pylist() == pq.read_table(after).to_pylist()
    rows = {row["business_id"]: row for row in pq.read_table(before).to_pylist()}
    steak = rows["b-steak"]
    other = rows["b-other"]
    assert steak["user_category_positive_match"] == 0.72
    assert steak["business_rating_count_log"] > 0
    assert steak["aspect_positive_match"] > 0
    assert steak["aspect_evidence_missing"] == 0.0
    assert other["aspect_evidence_missing"] == 1.0
