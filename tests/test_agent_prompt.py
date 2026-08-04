from __future__ import annotations

import json

import pytest

from yelp_agent.agent.prompt import AgentPromptError, build_rerank_prompt
from yelp_agent.agent.tools import (
    BusinessDetails,
    BusinessDetailsResult,
    HistoryReview,
    HybridRankingResult,
    UserHistoryResult,
)
from yelp_agent.features.quality import BusinessQuality
from yelp_agent.models import (
    RecommendationTask,
    ScoreBreakdown,
    UserProfile,
)


def _prompt_fixture() -> tuple[
    RecommendationTask,
    UserHistoryResult,
    UserProfile,
    HybridRankingResult,
    BusinessDetailsResult,
]:
    candidates = [f"business-{index:02d}" for index in range(20)]
    task = RecommendationTask(
        task_id="test:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )
    positive = HistoryReview(
        review_id="review-positive",
        business_id="history-positive",
        business_name="Liked Place",
        categories=["Restaurants", "Noodles"],
        stars=5,
        text="Cozy noodles and friendly service",
        date="2020-01-03T00:00:00",
    )
    negative = HistoryReview(
        review_id="review-negative",
        business_id="history-negative",
        business_name="Disliked Place",
        categories=["Restaurants", "Burgers"],
        stars=1,
        text="Very noisy and slow",
        date="2020-01-02T00:00:00",
    )
    history = UserHistoryResult(
        user_id=task.user_id,
        cutoff_time=task.cutoff_time,
        reviews=[positive, negative],
        recent_positive=[positive],
        recent_negative=[negative],
    )
    profile = UserProfile(
        user_id=task.user_id,
        history_count=2,
        average_rating=3.0,
        rating_distribution={"1": 1, "2": 0, "3": 0, "4": 0, "5": 1},
        preferred_categories={"Noodles": 0.9},
        disliked_categories={"Burgers": 0.2},
        positive_keywords=["cozy", "noodles"],
        negative_keywords=["noisy", "slow"],
    )
    breakdowns = {
        business_id: ScoreBreakdown(
            business_id=business_id,
            category_score=0.8,
            text_score=0.7,
            quality_score=0.6,
            location_score=0.5,
            hybrid_score=round(0.9 - index * 0.02, 2),
        )
        for index, business_id in enumerate(candidates)
    }
    hybrid = HybridRankingResult(
        task_id=task.task_id,
        ranking=candidates,
        score_breakdowns=breakdowns,
    )
    details = BusinessDetailsResult(
        task_id=task.task_id,
        cutoff_time=task.cutoff_time,
        businesses=[
            BusinessDetails(
                business_id=business_id,
                name=f"Name {business_id}",
                address=f"{index} Test Street",
                city="Philadelphia",
                state="PA",
                postal_code="19101",
                latitude=39.95,
                longitude=-75.16,
                categories=["Restaurants", "Noodles"],
                attributes={"WiFi": "free"},
                quality=BusinessQuality(
                    business_id=business_id,
                    review_count=100 + index,
                    mean_rating=4.0,
                    bayesian_rating=3.9,
                    normalized_bayesian_rating=0.725,
                    normalized_popularity=0.8,
                    quality_score=0.6,
                ),
                score_breakdown=breakdowns[business_id],
            )
            for index, business_id in reversed(
                list(enumerate(candidates[:8]))
            )
        ],
    )
    return task, history, profile, hybrid, details


def test_prompt_contains_only_hybrid_top_eight_in_hybrid_order() -> None:
    task, history, profile, hybrid, details = _prompt_fixture()

    messages = build_rerank_prompt(
        task=task,
        history=history,
        profile=profile,
        hybrid=hybrid,
        business_details=details,
    )

    assert [message.role for message in messages] == ["system", "user"]
    payload = json.loads(messages[1].content)
    assert [candidate["business_id"] for candidate in payload["candidates"]] == (
        task.candidate_business_ids[:8]
    )
    assert [candidate["hybrid_rank"] for candidate in payload["candidates"]] == (
        list(range(1, 9))
    )
    assert all(
        business_id not in messages[1].content
        for business_id in task.candidate_business_ids[8:]
    )
    assert payload["candidates"][0]["scores"]["hybrid"] == 0.9
    assert payload["candidates"][0]["historical_quality"]["review_count"] == 100
    assert payload["user_profile"]["preferred_categories"] == {"Noodles": 0.9}
    assert "ground_truth" not in messages[1].content.lower()


def test_prompt_rejects_mislabeled_representative_history() -> None:
    task, history, profile, hybrid, details = _prompt_fixture()
    malformed = history.model_copy(
        update={"recent_positive": [history.recent_negative[0]]}
    )

    with pytest.raises(AgentPromptError, match="Representative history"):
        build_rerank_prompt(
            task=task,
            history=malformed,
            profile=profile,
            hybrid=hybrid,
            business_details=details,
        )


def test_prompt_uses_only_eight_representatives_and_truncates_long_text() -> None:
    task, _, profile, hybrid, details = _prompt_fixture()
    positives = [
        HistoryReview(
            review_id=f"positive-{index}",
            business_id=f"history-positive-{index}",
            business_name=f"Liked {index}",
            categories=["Noodles"],
            stars=5,
            text=("x" * 700 if index == 0 else f"positive text {index}"),
            date=f"2020-01-{10 - index:02d}T00:00:00",
        )
        for index in range(4)
    ]
    negatives = [
        HistoryReview(
            review_id=f"negative-{index}",
            business_id=f"history-negative-{index}",
            business_name=f"Disliked {index}",
            categories=["Burgers"],
            stars=1,
            text=f"negative text {index}",
            date=f"2020-01-{6 - index:02d}T00:00:00",
        )
        for index in range(4)
    ]
    neutral = HistoryReview(
        review_id="neutral",
        business_id="history-neutral",
        business_name="Neutral Place",
        categories=["Coffee"],
        stars=3,
        text="NEUTRAL_TEXT_MUST_NOT_ENTER_PROMPT",
        date="2020-01-01T00:00:00",
    )
    history = UserHistoryResult(
        user_id=task.user_id,
        cutoff_time=task.cutoff_time,
        reviews=[*positives, *negatives, neutral],
        recent_positive=positives,
        recent_negative=negatives,
    )

    first = build_rerank_prompt(
        task=task,
        history=history,
        profile=profile,
        hybrid=hybrid,
        business_details=details,
    )
    second = build_rerank_prompt(
        task=task,
        history=history,
        profile=profile,
        hybrid=hybrid,
        business_details=details,
    )

    payload = json.loads(first[1].content)
    representatives = payload["representative_history"]
    assert len(representatives["positive"]) == 4
    assert len(representatives["negative"]) == 4
    assert len(representatives["positive"][0]["text"]) == 500
    assert representatives["positive"][0]["text"].endswith("…")
    assert "NEUTRAL_TEXT_MUST_NOT_ENTER_PROMPT" not in first[1].content
    assert first == second
