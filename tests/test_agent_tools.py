from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.agent.tools import AgentToolError, AgentToolbox
from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.features.quality import BusinessQuality
from yelp_agent.models import (
    Prediction,
    RecommendationTask,
    ScoreBreakdown,
    UserProfile,
)
from yelp_agent.rankers.hybrid_ranker import HybridTaskScore


class UnusedRanker:
    pass


class UnusedQualityStore:
    pass


class FixedRanker:
    def score(self, task: RecommendationTask) -> HybridTaskScore:
        return HybridTaskScore(
            profile=UserProfile(
                user_id=task.user_id,
                history_count=3,
                average_rating=3.0,
                rating_distribution={
                    "1": 1,
                    "2": 0,
                    "3": 1,
                    "4": 0,
                    "5": 1,
                },
                preferred_categories={"Test Cuisine": 0.8},
                disliked_categories={},
                positive_keywords=["cozy"],
                negative_keywords=["noisy"],
            ),
            score_breakdowns={
                business_id: ScoreBreakdown(
                    business_id=business_id,
                    category_score=0.8,
                    text_score=0.7,
                    quality_score=0.6,
                    location_score=0.5,
                    hybrid_score=0.65,
                )
                for business_id in task.candidate_business_ids
            },
        )

    def rank(self, task: RecommendationTask) -> Prediction:
        return Prediction(
            task_id=task.task_id,
            ranking=sorted(task.candidate_business_ids),
            latency_ms=1.0,
            fallback=False,
        )


class FixedQualityStore:
    def score_businesses(
        self,
        business_ids: list[str],
        cutoff_time: object,
    ) -> dict[str, BusinessQuality]:
        return {
            business_id: BusinessQuality(
                business_id=business_id,
                review_count=7,
                mean_rating=4.0,
                bayesian_rating=3.8,
                normalized_bayesian_rating=0.7,
                normalized_popularity=0.5,
                quality_score=0.6,
            )
            for business_id in business_ids
        }


def _write_agent_tool_fixture(
    root: Path,
) -> tuple[Path, Path, Path, RecommendationTask]:
    businesses_path = root / "businesses.parquet"
    interactions_path = root / "interactions.parquet"
    histories_path = root / "histories.parquet"
    candidates = [f"candidate-{index:02d}" for index in range(20)]
    business_ids = ["history-low", "history-neutral", "history-high", *candidates]
    pd.DataFrame(
        [
            {
                "business_id": business_id,
                "name": f"Name {business_id}",
                "address": "1 Test Street",
                "city": "Philadelphia",
                "state": "PA",
                "postal_code": "19101",
                "latitude": 39.95,
                "longitude": -75.16,
                "categories": ["Restaurants", "Test Cuisine"],
                "attributes_json": '{"OutdoorSeating": "True"}',
            }
            for business_id in business_ids
        ]
    ).to_parquet(businesses_path, index=False)
    pd.DataFrame(
        [
            {
                "review_id": "review-low",
                "user_id": "user-1",
                "business_id": "history-low",
                "stars": 1.0,
                "text": "Too noisy",
                "date": pd.Timestamp("2020-01-01"),
            },
            {
                "review_id": "review-neutral",
                "user_id": "user-1",
                "business_id": "history-neutral",
                "stars": 3.0,
                "text": "It was fine",
                "date": pd.Timestamp("2020-01-02"),
            },
            {
                "review_id": "review-high",
                "user_id": "user-1",
                "business_id": "history-high",
                "stars": 5.0,
                "text": "Cozy and delicious",
                "date": pd.Timestamp("2020-01-03"),
            },
            {
                "review_id": "future-unreferenced",
                "user_id": "user-1",
                "business_id": "history-high",
                "stars": 5.0,
                "text": "This must stay invisible",
                "date": pd.Timestamp("2030-01-01"),
            },
        ]
    ).to_parquet(interactions_path, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "test:user-1",
                "position": position,
                "review_id": review_id,
            }
            for position, review_id in enumerate(
                ["review-low", "review-neutral", "review-high"],
                start=1,
            )
        ]
    ).to_parquet(histories_path, index=False)
    task = RecommendationTask(
        task_id="test:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )
    return businesses_path, interactions_path, histories_path, task


def test_history_tool_returns_only_frozen_history_and_representative_reviews(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, task = _write_agent_tool_fixture(
        tmp_path
    )
    session = AgentToolbox(
        TemporalDataView(businesses, interactions, interactions),
        hybrid_ranker=UnusedRanker(),
        quality_store=UnusedQualityStore(),
    ).for_task(task)

    result = session.get_user_history(task.user_id, task.cutoff_time)

    assert [item.review_id for item in result.reviews] == [
        "review-high",
        "review-neutral",
        "review-low",
    ]
    assert [item.review_id for item in result.recent_positive] == [
        "review-high"
    ]
    assert [item.review_id for item in result.recent_negative] == [
        "review-low"
    ]
    assert all(item.date < task.cutoff_time for item in result.reviews)
    assert session.call_count == 1


def test_business_details_are_point_in_time_and_restricted_to_candidates(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, task = _write_agent_tool_fixture(
        tmp_path
    )
    session = AgentToolbox(
        TemporalDataView(businesses, interactions, interactions),
        hybrid_ranker=FixedRanker(),
        quality_store=FixedQualityStore(),
    ).for_task(task)

    result = session.get_business_details(
        task.candidate_business_ids[:2],
        task.cutoff_time,
    )

    assert [item.business_id for item in result.businesses] == [
        "candidate-00",
        "candidate-01",
    ]
    assert result.businesses[0].attributes == {"OutdoorSeating": "True"}
    assert result.businesses[0].quality.review_count == 7
    assert result.businesses[0].score_breakdown.hybrid_score == 0.65

    with pytest.raises(AgentToolError, match="outside the bound candidates"):
        session.get_business_details(["history-high"], task.cutoff_time)
    with pytest.raises(AgentToolError, match="cutoff_time"):
        session.get_business_details(
            ["candidate-00"],
            pd.Timestamp("2030-01-01").to_pydatetime(),
        )
    assert session.call_count == 3


def test_profile_tool_returns_bound_task_profile_and_rejects_another_user(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, task = _write_agent_tool_fixture(
        tmp_path
    )
    session = AgentToolbox(
        TemporalDataView(businesses, interactions, interactions),
        hybrid_ranker=FixedRanker(),
        quality_store=FixedQualityStore(),
    ).for_task(task)

    profile = session.get_user_profile(task.user_id, task.cutoff_time)

    assert profile.user_id == "user-1"
    assert profile.preferred_categories == {"Test Cuisine": 0.8}
    assert profile.positive_keywords == ["cozy"]
    assert profile.negative_keywords == ["noisy"]
    with pytest.raises(AgentToolError, match="user_id"):
        session.get_user_profile("another-user", task.cutoff_time)
    assert session.call_count == 2


def test_hybrid_ranking_tool_requires_the_complete_bound_candidate_set(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, task = _write_agent_tool_fixture(
        tmp_path
    )
    session = AgentToolbox(
        TemporalDataView(businesses, interactions, interactions),
        hybrid_ranker=FixedRanker(),
        quality_store=FixedQualityStore(),
    ).for_task(task)

    result = session.get_hybrid_ranking(
        task.user_id,
        list(reversed(task.candidate_business_ids)),
        task.cutoff_time,
    )

    assert result.ranking == sorted(task.candidate_business_ids)
    assert set(result.score_breakdowns) == set(task.candidate_business_ids)
    assert result.score_breakdowns["candidate-00"].hybrid_score == 0.65
    with pytest.raises(AgentToolError, match="complete bound candidate set"):
        session.get_hybrid_ranking(
            task.user_id,
            task.candidate_business_ids[:-1],
            task.cutoff_time,
        )
    with pytest.raises(AgentToolError, match="user_id"):
        session.get_hybrid_ranking(
            "another-user",
            task.candidate_business_ids,
            task.cutoff_time,
        )
    assert session.call_count == 3
