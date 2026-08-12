from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from yelp_agent.business_profiles.schema import (
    BusinessAspectEvent,
    BusinessRatingEvent,
)
from yelp_agent.business_profiles.store import BusinessKnowledgeStore
from yelp_agent.config import load_business_profile_config
from yelp_agent.data.temporal_view import BusinessRecord

PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _business(business_id: str) -> BusinessRecord:
    return BusinessRecord(
        business_id=business_id,
        name=f"Name {business_id}",
        address="1 Test Street",
        city="Philadelphia",
        state="PA",
        postal_code="19107",
        latitude=39.95,
        longitude=-75.16,
        categories=("Restaurants", "Steakhouses"),
        attributes_json='{"RestaurantsPriceRange2":"3"}',
    )


def _aspect(
    review_id: str,
    user_id: str,
    sentiment: str,
    review_time: datetime,
    *,
    aspect: str = "quiet_environment",
) -> BusinessAspectEvent:
    return BusinessAspectEvent.model_validate(
        {
            "review_id": review_id,
            "business_id": "business-a",
            "user_id": user_id,
            "review_time": review_time,
            "aspect": aspect,
            "sentiment": sentiment,
            "confidence": 0.85,
            "source_text_sha256": "a" * 64,
            "extractor_version": "1.2.0",
        }
    )


def test_store_builds_business_profile_from_only_evidence_before_cutoff() -> None:
    store = BusinessKnowledgeStore.from_records(
        businesses=(_business("business-a"), _business("business-b")),
        rating_events=(
            BusinessRatingEvent(
                review_id="rating-past-a",
                business_id="business-a",
                user_id="user-1",
                stars=5.0,
                review_time=datetime(2020, 1, 1),
            ),
            BusinessRatingEvent(
                review_id="rating-past-b",
                business_id="business-b",
                user_id="user-2",
                stars=3.0,
                review_time=datetime(2020, 1, 1),
            ),
            BusinessRatingEvent(
                review_id="rating-future-a",
                business_id="business-a",
                user_id="user-3",
                stars=1.0,
                review_time=datetime(2030, 1, 1),
            ),
        ),
        aspect_events=(
            _aspect("aspect-1", "user-1", "positive", datetime(2020, 1, 1)),
            _aspect("aspect-2", "user-2", "positive", datetime(2020, 1, 2)),
            _aspect("aspect-3", "user-1", "positive", datetime(2020, 1, 3)),
            _aspect("future-1", "user-3", "negative", datetime(2030, 1, 1)),
        ),
        config=load_business_profile_config(PROJECT_CONFIG_DIR),
    )

    profile = store.get(["business-a"], datetime(2021, 1, 1))["business-a"]
    quiet = profile.aspect_summaries["quiet_environment"]

    assert profile.quality.review_count == 1
    assert profile.quality.mean_rating == 5.0
    assert profile.quality.bayesian_rating == pytest.approx(85 / 21)
    assert profile.structured_attributes == {"RestaurantsPriceRange2": "3"}
    assert quiet.status == "known"
    assert quiet.positive_count == 3
    assert quiet.negative_count == 0
    assert quiet.positive_ratio == 1.0
    assert quiet.latest_evidence_time == datetime(2020, 1, 3)
    assert profile.evidence_summary.latest_rating_time == datetime(2020, 1, 1)
    assert profile.cutoff_time == datetime(2021, 1, 1)


def test_unknown_threshold_is_applied_per_aspect_to_directional_evidence() -> None:
    events = (
        _aspect(
            "food-1",
            "user-1",
            "positive",
            datetime(2020, 1, 1),
            aspect="food_quality",
        ),
        _aspect(
            "food-2",
            "user-2",
            "positive",
            datetime(2020, 1, 2),
            aspect="food_quality",
        ),
        _aspect(
            "food-3",
            "user-1",
            "negative",
            datetime(2020, 1, 3),
            aspect="food_quality",
        ),
        _aspect("quiet-1", "user-1", "positive", datetime(2020, 1, 1)),
        _aspect("quiet-2", "user-2", "mixed", datetime(2020, 1, 2)),
        _aspect("quiet-3", "user-2", "neutral", datetime(2020, 1, 3)),
    )
    store = BusinessKnowledgeStore.from_records(
        businesses=(_business("business-a"),),
        rating_events=(),
        aspect_events=events,
        config=load_business_profile_config(PROJECT_CONFIG_DIR),
    )

    profile = store.get(["business-a"], datetime(2021, 1, 1))["business-a"]

    assert profile.aspect_summaries["food_quality"].status == "known"
    quiet = profile.aspect_summaries["quiet_environment"]
    assert quiet.status == "unknown"
    assert quiet.evidence_count == 3
    assert quiet.positive_ratio is None
    assert quiet.confidence == 0.0


def test_narrow_aspect_reader_matches_the_complete_profile_at_the_same_cutoff() -> None:
    store = BusinessKnowledgeStore.from_records(
        businesses=(_business("business-a"),),
        rating_events=(),
        aspect_events=(
            _aspect("quiet-1", "user-1", "positive", datetime(2020, 1, 1)),
            _aspect("quiet-2", "user-2", "positive", datetime(2020, 1, 2)),
            _aspect("quiet-3", "user-3", "negative", datetime(2020, 1, 3)),
        ),
        config=load_business_profile_config(PROJECT_CONFIG_DIR),
    )
    cutoff = datetime(2021, 1, 1)

    narrow = store.get_aspects(
        ["business-a"],
        ["quiet_environment"],
        cutoff,
    )
    complete = store.get(["business-a"], cutoff)

    assert store.source_scope == "selected_user_interactions"
    assert narrow["business-a"] == {
        "quiet_environment": complete["business-a"].aspect_summaries[
            "quiet_environment"
        ]
    }


def test_conflicting_reviews_are_flagged_without_leaking_to_another_business() -> None:
    conflict_events = tuple(
        _aspect(
            f"review-{index}",
            f"user-{index}",
            "positive" if index < 3 else "negative",
            datetime(2020, 1, index + 1),
        )
        for index in range(6)
    )
    store = BusinessKnowledgeStore.from_records(
        businesses=(_business("business-a"), _business("business-b")),
        rating_events=(),
        aspect_events=conflict_events,
        config=load_business_profile_config(PROJECT_CONFIG_DIR),
    )

    profiles = store.get(
        ["business-a", "business-b"],
        datetime(2021, 1, 1),
    )

    conflicted = profiles["business-a"].aspect_summaries["quiet_environment"]
    isolated = profiles["business-b"].aspect_summaries["quiet_environment"]
    assert conflicted.status == "known"
    assert conflicted.conflict is True
    assert conflicted.positive_ratio == 0.5
    assert conflicted.negative_ratio == 0.5
    assert isolated.status == "unknown"
    assert isolated.evidence_count == 0
    assert (
        profiles["business-a"].profile_reliability
        > profiles["business-b"].profile_reliability
    )
