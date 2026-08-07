from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yelp_agent.config import load_user_profile_config
from yelp_agent.data.businesses import BUSINESS_SCHEMA
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.data.rolling_training import ROLLING_CONTEXT_SCHEMA
from yelp_agent.data.temporal import CONTEXT_SCHEMA
from yelp_agent.profiles.artifacts import (
    UserProfileArtifactError,
    build_user_profile_artifacts,
)
from yelp_agent.profiles.store import UserProfileNotFound, UserProfileStore
from yelp_agent.reviews.schema import REVIEW_ASPECT_SCHEMA

PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "businesses": root / "businesses.parquet",
        "interactions": root / "interactions.parquet",
        "aspects": root / "aspects.parquet",
        "train": root / "train_contexts.parquet",
        "evaluation": root / "evaluation_contexts.parquet",
    }
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "business_id": "business-1",
                    "name": "Bistro",
                    "address": "1 Test Street",
                    "city": "Philadelphia",
                    "state": "PA",
                    "postal_code": "19107",
                    "latitude": 39.95,
                    "longitude": -75.16,
                    "categories": ["Restaurants", "Bistros"],
                    "attributes_json": '{"RestaurantsPriceRange2":"2"}',
                }
            ],
            schema=BUSINESS_SCHEMA,
        ),
        paths["businesses"],
    )
    interaction_rows = [
        {
            "review_id": "review-1",
            "user_id": "user-1",
            "business_id": "business-1",
            "stars": 5.0,
            "useful": 0,
            "funny": 0,
            "cool": 0,
            "text": "The food was good.",
            "date": datetime(2020, 1, 1),
        },
        {
            "review_id": "review-2",
            "user_id": "user-1",
            "business_id": "business-1",
            "stars": 4.0,
            "useful": 0,
            "funny": 0,
            "cool": 0,
            "text": "Still good.",
            "date": datetime(2020, 2, 1),
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(interaction_rows, schema=REVIEW_SCHEMA),
        paths["interactions"],
    )
    evidence = "The food was good"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "review_id": "review-1",
                    "business_id": "business-1",
                    "user_id": "user-1",
                    "review_time": datetime(2020, 1, 1),
                    "aspect": "food_quality",
                    "sentiment": "positive",
                    "confidence": 0.85,
                    "evidence_span": evidence,
                    "evidence_start": 0,
                    "evidence_end": len(evidence),
                    "source_text_sha256": "a" * 64,
                    "extractor_name": "rule_based",
                    "extractor_version": "1.2.0",
                }
            ],
            schema=REVIEW_ASPECT_SCHEMA,
        ),
        paths["aspects"],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "train:user-1:1",
                    "split": "train",
                    "user_id": "user-1",
                    "cutoff_time": datetime(2020, 1, 15),
                    "history_count": 1,
                    "history_max_time": datetime(2020, 1, 1),
                    "target_position": 2,
                    "user_interaction_count": 4,
                    "fold": 1,
                    "sample_weight": 0.5,
                }
            ],
            schema=ROLLING_CONTEXT_SCHEMA,
        ),
        paths["train"],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "validation:user-1",
                    "split": "validation",
                    "user_id": "user-1",
                    "cutoff_time": datetime(2020, 3, 1),
                    "history_count": 2,
                    "history_max_time": datetime(2020, 2, 1),
                }
            ],
            schema=CONTEXT_SCHEMA,
        ),
        paths["evaluation"],
    )
    return paths


def test_artifact_builder_freezes_task_time_profiles_without_ground_truth(
    tmp_path: Path,
) -> None:
    paths = _write_sources(tmp_path)

    result = build_user_profile_artifacts(
        businesses_path=paths["businesses"],
        interactions_path=paths["interactions"],
        aspect_records_path=paths["aspects"],
        rolling_contexts_path=paths["train"],
        evaluation_contexts_path=paths["evaluation"],
        output_root=tmp_path / "profiles",
        config=load_user_profile_config(PROJECT_CONFIG_DIR),
        broad_categories={"Restaurants"},
    )

    assert result.status == "written"
    assert result.profile_snapshots == 2
    assert result.task_links == 2
    assert result.split_task_counts == {"train": 1, "validation": 1}
    assert result.history_count_mismatches == 0
    assert (tmp_path / "profiles" / "profile_snapshots.parquet").is_file()
    assert (tmp_path / "profiles" / "preference_signals.parquet").is_file()
    assert (tmp_path / "profiles" / "task_profile_map.parquet").is_file()
    assert (tmp_path / "profiles" / "manifest.json").is_file()

    artifact_paths = (
        tmp_path / "profiles" / "profile_snapshots.parquet",
        tmp_path / "profiles" / "preference_signals.parquet",
        tmp_path / "profiles" / "task_profile_map.parquet",
    )
    first_bytes = tuple(path.read_bytes() for path in artifact_paths)
    second = build_user_profile_artifacts(
        businesses_path=paths["businesses"],
        interactions_path=paths["interactions"],
        aspect_records_path=paths["aspects"],
        rolling_contexts_path=paths["train"],
        evaluation_contexts_path=paths["evaluation"],
        output_root=tmp_path / "profiles",
        config=load_user_profile_config(PROJECT_CONFIG_DIR),
        broad_categories={"Restaurants"},
    )

    assert second.status == "skipped"
    assert tuple(path.read_bytes() for path in artifact_paths) == first_bytes


def test_profile_store_reads_only_exact_cutoff_or_bound_task(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path)
    output_root = tmp_path / "profiles"
    build_user_profile_artifacts(
        businesses_path=paths["businesses"],
        interactions_path=paths["interactions"],
        aspect_records_path=paths["aspects"],
        rolling_contexts_path=paths["train"],
        evaluation_contexts_path=paths["evaluation"],
        output_root=output_root,
        config=load_user_profile_config(PROJECT_CONFIG_DIR),
        broad_categories={"Restaurants"},
    )

    with UserProfileStore(output_root) as store:
        train_profile = store.get("user-1", datetime(2020, 1, 15))
        validation_profile = store.for_task("validation:user-1")
        with pytest.raises(UserProfileNotFound):
            store.get("user-1", datetime(2020, 2, 15))

    assert train_profile.history_length == 1
    assert train_profile.aspect_preferences[0].value == "food_quality"
    assert validation_profile.history_length == 2
    assert validation_profile.cutoff_time == datetime(2020, 3, 1)


def test_history_count_mismatch_leaves_no_partial_artifacts(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path)
    bad_train = pq.read_table(paths["train"]).to_pylist()
    bad_train[0]["history_count"] = 99
    pq.write_table(
        pa.Table.from_pylist(bad_train, schema=ROLLING_CONTEXT_SCHEMA),
        paths["train"],
    )
    output_root = tmp_path / "profiles"

    with pytest.raises(UserProfileArtifactError, match="History count mismatch"):
        build_user_profile_artifacts(
            businesses_path=paths["businesses"],
            interactions_path=paths["interactions"],
            aspect_records_path=paths["aspects"],
            rolling_contexts_path=paths["train"],
            evaluation_contexts_path=paths["evaluation"],
            output_root=output_root,
            config=load_user_profile_config(PROJECT_CONFIG_DIR),
            broad_categories={"Restaurants"},
        )

    assert not list(output_root.glob("*.parquet"))
    assert not list(output_root.glob("*.partial"))
