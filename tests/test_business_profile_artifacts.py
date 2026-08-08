from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yelp_agent.business_profiles.artifacts import (
    BusinessKnowledgeArtifactError,
    build_business_knowledge_artifacts,
)
from yelp_agent.business_profiles.schema import (
    BUSINESS_ASPECT_EVENT_SCHEMA,
    BUSINESS_RATING_EVENT_SCHEMA,
)
from yelp_agent.business_profiles.store import (
    BusinessKnowledgeError,
    BusinessKnowledgeStore,
)
from yelp_agent.config import load_business_profile_config
from yelp_agent.data.businesses import BUSINESS_SCHEMA
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.reviews.schema import REVIEW_ASPECT_SCHEMA

PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _write_sources(root: Path) -> tuple[Path, Path, Path]:
    businesses_path = root / "businesses.parquet"
    reviews_path = root / "reviews.parquet"
    aspects_path = root / "aspects.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "business_id": "business-a",
                    "name": "Test Bistro",
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
        businesses_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "review_id": "review-1",
                    "user_id": "user-1",
                    "business_id": "business-a",
                    "stars": 5.0,
                    "useful": 0,
                    "funny": 0,
                    "cool": 0,
                    "text": "This long text must not enter the compact index.",
                    "date": datetime(2020, 1, 1),
                }
            ],
            schema=REVIEW_SCHEMA,
        ),
        reviews_path,
    )
    evidence = "quiet room"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "review_id": "review-1",
                    "business_id": "business-a",
                    "user_id": "user-1",
                    "review_time": datetime(2020, 1, 1),
                    "aspect": "quiet_environment",
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
        aspects_path,
    )
    return businesses_path, reviews_path, aspects_path


def test_artifacts_project_out_review_text_and_are_reusable(tmp_path: Path) -> None:
    businesses, reviews, aspects = _write_sources(tmp_path)
    output_root = tmp_path / "business_profiles"

    first = build_business_knowledge_artifacts(
        businesses_path=businesses,
        reviews_path=reviews,
        aspect_records_path=aspects,
        output_root=output_root,
        config=load_business_profile_config(PROJECT_CONFIG_DIR),
    )

    assert first.status == "written"
    assert first.businesses == 1
    assert first.rating_events == 1
    assert first.aspect_events == 1
    assert pq.ParquetFile(output_root / "rating_events.parquet").schema_arrow.equals(
        BUSINESS_RATING_EVENT_SCHEMA,
        check_metadata=False,
    )
    assert pq.ParquetFile(output_root / "aspect_events.parquet").schema_arrow.equals(
        BUSINESS_ASPECT_EVENT_SCHEMA,
        check_metadata=False,
    )
    assert "text" not in BUSINESS_RATING_EVENT_SCHEMA.names
    assert "evidence_span" not in BUSINESS_ASPECT_EVENT_SCHEMA.names
    artifact_paths = tuple(sorted(output_root.glob("*.parquet")))
    first_bytes = tuple(path.read_bytes() for path in artifact_paths)

    second = build_business_knowledge_artifacts(
        businesses_path=businesses,
        reviews_path=reviews,
        aspect_records_path=aspects,
        output_root=output_root,
        config=load_business_profile_config(PROJECT_CONFIG_DIR),
    )

    assert second.status == "skipped"
    assert tuple(path.read_bytes() for path in artifact_paths) == first_bytes


def test_store_loads_self_contained_artifacts_and_reads_exact_cutoff(
    tmp_path: Path,
) -> None:
    businesses, reviews, aspects = _write_sources(tmp_path)
    output_root = tmp_path / "business_profiles"
    config = load_business_profile_config(PROJECT_CONFIG_DIR)
    build_business_knowledge_artifacts(
        businesses_path=businesses,
        reviews_path=reviews,
        aspect_records_path=aspects,
        output_root=output_root,
        config=config,
    )

    store = BusinessKnowledgeStore.from_artifacts(output_root, config=config)
    profile = store.get(["business-a"], datetime(2021, 1, 1))["business-a"]

    assert profile.name == "Test Bistro"
    assert profile.quality.review_count == 1
    assert profile.aspect_summaries["quiet_environment"].evidence_count == 1
    assert profile.aspect_summaries["quiet_environment"].status == "unknown"
    assert profile.source_scope == "selected_user_interactions"


def test_artifacts_reject_aspect_evidence_that_cannot_trace_to_its_review(
    tmp_path: Path,
) -> None:
    businesses, reviews, aspects = _write_sources(tmp_path)
    corrupted = pq.read_table(aspects).to_pylist()
    corrupted[0]["review_id"] = "not-the-source-review"
    pq.write_table(
        pa.Table.from_pylist(corrupted, schema=REVIEW_ASPECT_SCHEMA),
        aspects,
    )
    output_root = tmp_path / "business_profiles"

    with pytest.raises(BusinessKnowledgeArtifactError, match="source review"):
        build_business_knowledge_artifacts(
            businesses_path=businesses,
            reviews_path=reviews,
            aspect_records_path=aspects,
            output_root=output_root,
            config=load_business_profile_config(PROJECT_CONFIG_DIR),
        )

    assert not list(output_root.glob("*.parquet"))
    assert not list(output_root.glob("*.partial"))


def test_store_rejects_same_schema_artifact_with_wrong_manifest_hash(
    tmp_path: Path,
) -> None:
    businesses, reviews, aspects = _write_sources(tmp_path)
    output_root = tmp_path / "business_profiles"
    config = load_business_profile_config(PROJECT_CONFIG_DIR)
    build_business_knowledge_artifacts(
        businesses_path=businesses,
        reviews_path=reviews,
        aspect_records_path=aspects,
        output_root=output_root,
        config=config,
    )
    ratings_path = output_root / "rating_events.parquet"
    corrupted = pq.read_table(ratings_path).to_pylist()
    corrupted[0]["stars"] = 1.0
    pq.write_table(
        pa.Table.from_pylist(corrupted, schema=BUSINESS_RATING_EVENT_SCHEMA),
        ratings_path,
    )

    with pytest.raises(BusinessKnowledgeError, match="hash"):
        BusinessKnowledgeStore.from_artifacts(output_root, config=config)
