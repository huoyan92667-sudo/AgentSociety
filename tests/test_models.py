from datetime import datetime

import pytest
from pydantic import ValidationError

from yelp_agent.models import (
    GroundTruth,
    LocationCenter,
    Prediction,
    RecommendationTask,
    ScoreBreakdown,
    UserProfile,
)
from yelp_agent.protocols import Ranker


def test_recommendation_task_round_trips_through_json() -> None:
    task = RecommendationTask(
        task_id="test:user-1:review-20",
        user_id="user-1",
        cutoff_time=datetime(2024, 1, 20, 12, 30),
        candidate_business_ids=[f"business-{index}" for index in range(20)],
    )

    restored = RecommendationTask.model_validate_json(task.model_dump_json())

    assert restored == task
    assert len(restored.candidate_business_ids) == 20


@pytest.mark.parametrize(
    "candidate_ids",
    [
        [f"business-{index}" for index in range(19)],
        [f"business-{index}" for index in range(21)],
        ["same-business"] * 20,
        ["", *[f"business-{index}" for index in range(19)]],
    ],
)
def test_recommendation_task_requires_twenty_unique_nonempty_candidates(
    candidate_ids: list[str],
) -> None:
    with pytest.raises(ValidationError):
        RecommendationTask(
            task_id="test:user-1:review-20",
            user_id="user-1",
            cutoff_time=datetime(2024, 1, 20, 12, 30),
            candidate_business_ids=candidate_ids,
        )


def test_ground_truth_is_a_separate_contract_from_ranker_task() -> None:
    truth = GroundTruth(
        task_id="test:user-1:review-20",
        target_business_id="business-7",
    )

    assert truth.target_business_id == "business-7"
    with pytest.raises(ValidationError):
        RecommendationTask(
            task_id=truth.task_id,
            user_id="user-1",
            cutoff_time=datetime(2024, 1, 20, 12, 30),
            candidate_business_ids=[f"business-{index}" for index in range(20)],
            target_business_id=truth.target_business_id,
        )


def test_user_profile_round_trips_with_dynamic_preferences() -> None:
    profile = UserProfile(
        user_id="user-1",
        history_count=3,
        average_rating=11 / 3,
        rating_distribution={"1": 0, "2": 1, "3": 0, "4": 1, "5": 1},
        preferred_categories={"Italian": 0.85},
        disliked_categories={"Nightlife": 0.4},
        preferred_city="Philadelphia",
        location_center=LocationCenter(latitude=39.9526, longitude=-75.1652),
        positive_keywords=["pasta", "friendly"],
        negative_keywords=["slow"],
    )

    restored = UserProfile.model_validate_json(profile.model_dump_json())

    assert restored == profile


@pytest.mark.parametrize(
    "rating_distribution",
    [
        {"1": 0, "2": 1, "3": 0, "4": 1},
        {"1": 0, "2": 1, "3": 0, "4": 1, "5": 0},
    ],
)
def test_user_profile_rating_distribution_matches_history(
    rating_distribution: dict[str, int],
) -> None:
    with pytest.raises(ValidationError):
        UserProfile(
            user_id="user-1",
            history_count=3,
            average_rating=11 / 3,
            rating_distribution=rating_distribution,
            preferred_categories={},
            disliked_categories={},
        )


@pytest.mark.parametrize(
    "keywords",
    [
        [f"keyword-{index}" for index in range(11)],
        ["pasta", "pasta"],
        [""],
    ],
)
def test_user_profile_keyword_lists_are_small_unique_and_nonempty(
    keywords: list[str],
) -> None:
    with pytest.raises(ValidationError):
        UserProfile(
            user_id="user-1",
            history_count=0,
            average_rating=0,
            rating_distribution={"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
            preferred_categories={},
            disliked_categories={},
            positive_keywords=keywords,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("category_score", -0.01),
        ("text_score", 1.01),
        ("quality_score", -1),
        ("location_score", 2),
        ("hybrid_score", 1.5),
    ],
)
def test_score_breakdown_rejects_values_outside_unit_interval(
    field_name: str,
    invalid_value: float,
) -> None:
    values = {
        "business_id": "business-7",
        "category_score": 0.5,
        "text_score": 0.5,
        "quality_score": 0.5,
        "location_score": 0.5,
        "hybrid_score": 0.5,
    }
    values[field_name] = invalid_value

    with pytest.raises(ValidationError):
        ScoreBreakdown(**values)


def test_prediction_round_trips_with_operational_metadata() -> None:
    prediction = Prediction(
        task_id="test:user-1:review-20",
        ranking=[f"business-{index}" for index in range(20)],
        latency_ms=12.5,
        fallback=False,
        tool_calls=4,
        llm_tokens=128,
        metadata={"method": "agent"},
    )

    restored = Prediction.model_validate_json(prediction.model_dump_json())

    assert restored == prediction
    assert restored.fallback_reason is None


@pytest.mark.parametrize(
    "ranking",
    [
        [f"business-{index}" for index in range(19)],
        [f"business-{index}" for index in range(21)],
        ["same-business"] * 20,
    ],
)
def test_prediction_requires_a_complete_unique_twenty_item_ranking(
    ranking: list[str],
) -> None:
    with pytest.raises(ValidationError):
        Prediction(
            task_id="test:user-1:review-20",
            ranking=ranking,
            latency_ms=1,
            fallback=False,
        )


@pytest.mark.parametrize(
    ("fallback", "fallback_reason"),
    [
        (True, None),
        (False, "llm_disabled"),
    ],
)
def test_prediction_fallback_flag_and_reason_must_agree(
    fallback: bool,
    fallback_reason: str | None,
) -> None:
    with pytest.raises(ValidationError):
        Prediction(
            task_id="test:user-1:review-20",
            ranking=[f"business-{index}" for index in range(20)],
            latency_ms=1,
            fallback=fallback,
            fallback_reason=fallback_reason,
        )


def test_ranker_interface_accepts_any_adapter_with_rank_method() -> None:
    class EchoRanker:
        def rank(self, task: RecommendationTask) -> Prediction:
            return Prediction(
                task_id=task.task_id,
                ranking=task.candidate_business_ids,
                latency_ms=0,
                fallback=False,
            )

    task = RecommendationTask(
        task_id="test:user-1:review-20",
        user_id="user-1",
        cutoff_time=datetime(2024, 1, 20, 12, 30),
        candidate_business_ids=[f"business-{index}" for index in range(20)],
    )
    ranker = EchoRanker()

    assert isinstance(ranker, Ranker)
    assert ranker.rank(task).ranking == task.candidate_business_ids
