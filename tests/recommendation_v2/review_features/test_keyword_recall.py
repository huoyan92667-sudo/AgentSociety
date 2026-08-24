from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.config import load_review_aspect_settings
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.recommendation_v2.business_facts import BUSINESS_FACT_SCHEMA
from yelp_agent.recommendation_v2.review_features import (
    CANDIDATE_ASPECT_SCHEMA,
    CANDIDATE_REVIEW_SCHEMA,
    KeywordAspectMatcher,
    build_keyword_review_candidates,
)
from yelp_agent.recommendation_v2.review_features.definitions import (
    build_aspect_recall_definitions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _business_row(business_id: str) -> dict[str, object]:
    return {
        "business_id": business_id,
        "name": "Test Restaurant",
        "address": "1 Main St",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19107",
        "latitude": 39.95,
        "longitude": -75.16,
        "categories": ["Restaurants"],
        "price_level": None,
        "price_lower_usd": None,
        "price_upper_usd": None,
        "rating": 4.0,
        "review_count": 2,
        "accepts_reservations": None,
        "delivery": None,
        "takeout": None,
        "outdoor_seating": None,
        "good_for_kids": None,
        "good_for_groups": None,
        "wheelchair_accessible": None,
        "dogs_allowed": None,
        "parking_available": None,
        "parking_garage": None,
        "parking_street": None,
        "parking_validated": None,
        "parking_lot": None,
        "parking_valet": None,
    }


def _review_row(
    review_id: str,
    business_id: str,
    text: str,
) -> dict[str, object]:
    return {
        "review_id": review_id,
        "user_id": f"user-{review_id}",
        "business_id": business_id,
        "stars": 4.0,
        "useful": 0,
        "funny": 0,
        "cool": 0,
        "text": text,
        "date": datetime(2025, 1, 1, tzinfo=UTC),
    }


def test_old_vocabulary_only_marks_candidate_aspects() -> None:
    _, vocabulary = load_review_aspect_settings(PROJECT_ROOT / "configs")
    matcher = KeywordAspectMatcher(build_aspect_recall_definitions(vocabulary))

    matches = matcher.match(
        "The food was delicious, but it was too loud and hard to park."
    )

    assert list(matches) == ["food_quality", "quiet_environment", "parking"]
    assert matches["food_quality"] == ["delicious"]
    assert matches["quiet_environment"] == ["too loud"]
    assert matches["parking"] == ["hard to park"]


def test_builds_complete_reviews_and_review_aspect_links(tmp_path: Path) -> None:
    review_path = tmp_path / "reviews.parquet"
    business_path = tmp_path / "business_facts.parquet"
    output_root = tmp_path / "output"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _review_row(
                    "r1",
                    "b1",
                    "The food was delicious, but it was too loud and hard to park.",
                ),
                _review_row("r2", "b1", "There was no wait and great service."),
                _review_row("r3", "outside", "The restaurant was filthy."),
            ],
            schema=REVIEW_SCHEMA,
        ),
        review_path,
    )
    pq.write_table(
        pa.Table.from_pylist([_business_row("b1")], schema=BUSINESS_FACT_SCHEMA),
        business_path,
    )

    result = build_keyword_review_candidates(
        review_path,
        business_path,
        PROJECT_ROOT / "configs",
        output_root,
    )
    review_table = pq.read_table(output_root / "keyword_candidate_reviews.parquet")
    aspect_table = pq.read_table(output_root / "keyword_candidate_aspects.parquet")

    assert result.manifest.source_review_count == 2
    assert result.manifest.candidate_review_count == 2
    assert result.manifest.candidate_aspect_count == 5
    assert review_table.schema.equals(CANDIDATE_REVIEW_SCHEMA, check_metadata=False)
    assert aspect_table.schema.equals(CANDIDATE_ASPECT_SCHEMA, check_metadata=False)
    assert set(review_table.column("review_id").to_pylist()) == {"r1", "r2"}
    assert set(zip(
        aspect_table.column("review_id").to_pylist(),
        aspect_table.column("aspect").to_pylist(),
        strict=True,
    )) == {
        ("r1", "food_quality"),
        ("r1", "quiet_environment"),
        ("r1", "parking"),
        ("r2", "service"),
        ("r2", "queue_time"),
    }
    assert aspect_table.column("semantic_score").null_count == 5


def test_reuses_unchanged_keyword_build(tmp_path: Path) -> None:
    review_path = tmp_path / "reviews.parquet"
    business_path = tmp_path / "business_facts.parquet"
    output_root = tmp_path / "output"
    pq.write_table(
        pa.Table.from_pylist(
            [_review_row("r1", "b1", "The food was delicious.")],
            schema=REVIEW_SCHEMA,
        ),
        review_path,
    )
    pq.write_table(
        pa.Table.from_pylist([_business_row("b1")], schema=BUSINESS_FACT_SCHEMA),
        business_path,
    )

    first = build_keyword_review_candidates(
        review_path,
        business_path,
        PROJECT_ROOT / "configs",
        output_root,
    )
    second = build_keyword_review_candidates(
        review_path,
        business_path,
        PROJECT_ROOT / "configs",
        output_root,
    )

    assert first.status == "written"
    assert second.status == "skipped"
    assert second.manifest == first.manifest
