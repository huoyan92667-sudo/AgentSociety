from __future__ import annotations

from datetime import datetime
from pathlib import Path

from yelp_agent.config import load_user_profile_config
from yelp_agent.data.temporal_view import BusinessRecord, InteractionRecord
from yelp_agent.profiles.adapter import to_agent_user_profile
from yelp_agent.profiles.builder import UserProfileBuilder
from yelp_agent.reviews.schema import ReviewAspectRecord

PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _business(
    business_id: str,
    *categories: str,
    postal_code: str = "19107",
    latitude: float | None = 39.95,
    longitude: float | None = -75.16,
    price: int | None = 2,
) -> BusinessRecord:
    attributes_json = (
        "{}" if price is None else f'{{"RestaurantsPriceRange2":"{price}"}}'
    )
    return BusinessRecord(
        business_id=business_id,
        name=business_id,
        address="",
        city="Philadelphia",
        state="PA",
        postal_code=postal_code,
        latitude=latitude,
        longitude=longitude,
        categories=categories,
        attributes_json=attributes_json,
    )


def test_builder_uses_only_past_history_for_category_preferences() -> None:
    config = load_user_profile_config(PROJECT_CONFIG_DIR)
    builder = UserProfileBuilder.from_records(
        businesses=(
            _business("steak", "Restaurants", "Steakhouses"),
            _business("sushi", "Restaurants", "Sushi Bars"),
        ),
        interactions=(
            InteractionRecord(
                review_id="past",
                user_id="user-1",
                business_id="steak",
                stars=5.0,
                text="Excellent steak.",
                date=datetime(2020, 1, 1),
            ),
            InteractionRecord(
                review_id="future",
                user_id="user-1",
                business_id="sushi",
                stars=5.0,
                text="Excellent sushi.",
                date=datetime(2030, 1, 1),
            ),
        ),
        aspect_records=(),
        config=config,
        broad_categories={"Restaurants"},
    )

    profile = builder.build("user-1", datetime(2021, 1, 1))

    assert profile.history_length == 1
    assert [signal.value for signal in profile.category_preferences] == ["Steakhouses"]
    assert profile.category_dislikes == []
    assert "Sushi Bars" not in profile.model_dump_json()


def _aspect(
    review_id: str,
    aspect: str,
    sentiment: str,
    review_time: datetime,
) -> ReviewAspectRecord:
    evidence = f"evidence for {aspect}"
    return ReviewAspectRecord.model_validate(
        {
            "review_id": review_id,
            "business_id": "steak",
            "user_id": "user-1",
            "review_time": review_time,
            "aspect": aspect,
            "sentiment": sentiment,
            "confidence": 0.85,
            "evidence_span": evidence,
            "evidence_start": 0,
            "evidence_end": len(evidence),
            "source_text_sha256": "a" * 64,
            "extractor_name": "rule_based",
            "extractor_version": "1.2.0",
        }
    )


def test_builder_aggregates_only_past_review_aspects() -> None:
    config = load_user_profile_config(PROJECT_CONFIG_DIR)
    builder = UserProfileBuilder.from_records(
        businesses=(_business("steak", "Restaurants", "Steakhouses"),),
        interactions=(
            InteractionRecord(
                review_id="past",
                user_id="user-1",
                business_id="steak",
                stars=4.0,
                text="Past review.",
                date=datetime(2020, 1, 1),
            ),
        ),
        aspect_records=(
            _aspect(
                "past",
                "quiet_environment",
                "positive",
                datetime(2020, 1, 1),
            ),
            _aspect(
                "past",
                "queue_time",
                "negative",
                datetime(2020, 1, 1),
            ),
            _aspect(
                "future",
                "parking",
                "positive",
                datetime(2030, 1, 1),
            ),
        ),
        config=config,
        broad_categories={"Restaurants"},
    )

    profile = builder.build("user-1", datetime(2021, 1, 1))

    assert [signal.value for signal in profile.aspect_preferences] == [
        "quiet_environment"
    ]
    assert [signal.value for signal in profile.aspect_dislikes] == ["queue_time"]
    assert profile.evidence_summary.aspect_evidence_count == 2
    assert "parking" not in profile.model_dump_json()


def test_builder_derives_recent_price_area_and_location_preferences() -> None:
    config = load_user_profile_config(PROJECT_CONFIG_DIR)
    builder = UserProfileBuilder.from_records(
        businesses=(
            _business(
                "old",
                "Restaurants",
                "Bakeries",
                postal_code="19107",
                latitude=39.90,
                longitude=-75.20,
                price=1,
            ),
            _business(
                "recent",
                "Restaurants",
                "Bakeries",
                postal_code="19106",
                latitude=40.00,
                longitude=-75.10,
                price=2,
            ),
        ),
        interactions=(
            InteractionRecord(
                review_id="old",
                user_id="user-1",
                business_id="old",
                stars=5.0,
                text="Old visit.",
                date=datetime(2018, 1, 1),
            ),
            InteractionRecord(
                review_id="recent",
                user_id="user-1",
                business_id="recent",
                stars=5.0,
                text="Recent visit.",
                date=datetime(2020, 12, 1),
            ),
        ),
        aspect_records=(),
        config=config,
        broad_categories={"Restaurants"},
    )

    profile = builder.build("user-1", datetime(2021, 1, 1))

    assert profile.price_preference is not None
    assert profile.price_preference.value == "2"
    assert [signal.value for signal in profile.frequent_areas] == ["19106", "19107"]
    assert profile.location_center is not None
    assert profile.location_center.latitude > 39.95
    assert profile.location_center.longitude > -75.15
    assert profile.evidence_summary.price_evidence_count == 2
    assert profile.evidence_summary.area_evidence_count == 2


def test_agent_adapter_preserves_long_term_profile_signals() -> None:
    config = load_user_profile_config(PROJECT_CONFIG_DIR)
    cutoff = datetime(2021, 1, 1)
    builder = UserProfileBuilder.from_records(
        businesses=(
            _business(
                "steak",
                "Restaurants",
                "Steakhouses",
                postal_code="19107",
                price=3,
            ),
        ),
        interactions=(
            InteractionRecord(
                review_id="review-1",
                user_id="user-1",
                business_id="steak",
                stars=5.0,
                text="Quiet and delicious.",
                date=datetime(2020, 12, 1),
            ),
        ),
        aspect_records=(
            _aspect(
                "review-1",
                "quiet_environment",
                "positive",
                datetime(2020, 12, 1),
            ),
        ),
        config=config,
        broad_categories={"Restaurants"},
    )

    rich_profile = builder.build("user-1", cutoff)
    agent_profile = to_agent_user_profile(rich_profile)

    assert agent_profile.profile_id == rich_profile.profile_id
    assert agent_profile.cutoff_time == cutoff
    assert agent_profile.history_count == 1
    assert agent_profile.preferred_categories == {"Steakhouses": 1.0}
    assert agent_profile.category_confidences["Steakhouses"] > 0
    assert agent_profile.aspect_preferences == {"quiet_environment": 1.0}
    assert agent_profile.aspect_confidences["quiet_environment"] > 0
    assert agent_profile.price_preference == "3"
    assert agent_profile.frequent_areas == ["19107"]
    assert agent_profile.profile_reliability == rich_profile.reliability
    assert agent_profile.profile_version == "1.0.0"


def test_future_evidence_cannot_rewrite_an_existing_profile() -> None:
    config = load_user_profile_config(PROJECT_CONFIG_DIR)
    cutoff = datetime(2021, 1, 1)
    business = _business("steak", "Restaurants", "Steakhouses")
    past = InteractionRecord(
        review_id="past",
        user_id="user-1",
        business_id="steak",
        stars=5.0,
        text="Past visit.",
        date=datetime(2020, 1, 1),
    )
    common = {
        "businesses": (business,),
        "config": config,
        "broad_categories": {"Restaurants"},
    }
    before = UserProfileBuilder.from_records(
        interactions=(past,),
        aspect_records=(_aspect("past", "food_quality", "positive", past.date),),
        **common,
    ).build("user-1", cutoff)
    after = UserProfileBuilder.from_records(
        interactions=(
            past,
            InteractionRecord(
                review_id="future",
                user_id="user-1",
                business_id="steak",
                stars=1.0,
                text="Future visit.",
                date=datetime(2030, 1, 1),
            ),
        ),
        aspect_records=(
            _aspect("past", "food_quality", "positive", past.date),
            _aspect(
                "future",
                "food_quality",
                "negative",
                datetime(2030, 1, 1),
            ),
        ),
        **common,
    ).build("user-1", cutoff)

    assert after == before
    assert after.model_dump_json() == before.model_dump_json()


def test_more_history_increases_profile_reliability() -> None:
    config = load_user_profile_config(PROJECT_CONFIG_DIR)
    businesses = tuple(
        _business(f"business-{index}", "Restaurants", "Bakeries") for index in range(5)
    )
    interactions = tuple(
        InteractionRecord(
            review_id=f"review-{index}",
            user_id="user-1",
            business_id=f"business-{index}",
            stars=5.0,
            text="Good.",
            date=datetime(2020, index + 1, 1),
        )
        for index in range(5)
    )
    one = UserProfileBuilder.from_records(
        businesses=businesses,
        interactions=interactions[:1],
        aspect_records=(),
        config=config,
        broad_categories={"Restaurants"},
    ).build("user-1", datetime(2021, 1, 1))
    five = UserProfileBuilder.from_records(
        businesses=businesses,
        interactions=interactions,
        aspect_records=(),
        config=config,
        broad_categories={"Restaurants"},
    ).build("user-1", datetime(2021, 1, 1))

    assert five.reliability > one.reliability
    assert (
        five.category_preferences[0].confidence > one.category_preferences[0].confidence
    )


def test_missing_price_and_coordinates_are_explicit_not_errors() -> None:
    config = load_user_profile_config(PROJECT_CONFIG_DIR)
    builder = UserProfileBuilder.from_records(
        businesses=(
            _business(
                "unknown",
                "Restaurants",
                "Bakeries",
                price=None,
                latitude=None,
                longitude=None,
            ),
        ),
        interactions=(
            InteractionRecord(
                review_id="review-1",
                user_id="user-1",
                business_id="unknown",
                stars=4.0,
                text="Good.",
                date=datetime(2020, 1, 1),
            ),
        ),
        aspect_records=(),
        config=config,
        broad_categories={"Restaurants"},
    )

    profile = builder.build("user-1", datetime(2021, 1, 1))

    assert profile.price_preference is None
    assert profile.location_center is None
