from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.models import RecommendationTask


def _business(business_id: str, categories: list[str]) -> dict[str, object]:
    return {
        "business_id": business_id,
        "name": business_id,
        "address": "",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "",
        "latitude": 39.95,
        "longitude": -75.16,
        "categories": categories,
        "attributes_json": "{}",
    }


def _write_category_fixture(root: Path) -> tuple[Path, Path, Path, list[str]]:
    businesses_path = root / "businesses.parquet"
    interactions_path = root / "interactions.parquet"
    histories_path = root / "histories.parquet"
    candidates = [f"candidate-{index:02d}" for index in range(20)]
    businesses = [
        _business("history-mexican-1", ["Restaurants", "Mexican"]),
        _business("history-mexican-2", ["Restaurants", "Mexican"]),
        _business("history-coffee", ["Food", "Coffee & Tea"]),
        _business(candidates[0], ["Restaurants", "Mexican"]),
        _business(candidates[1], ["Food", "Coffee & Tea"]),
        *[
            _business(business_id, ["Shopping", "Books"])
            for business_id in candidates[2:]
        ],
    ]
    pd.DataFrame(businesses).to_parquet(businesses_path, index=False)
    pd.DataFrame(
        [
            {
                "review_id": "history-1",
                "user_id": "user-1",
                "business_id": "history-mexican-1",
                "stars": 5.0,
                "text": "great mexican",
                "date": pd.Timestamp("2020-01-01"),
            },
            {
                "review_id": "history-2",
                "user_id": "user-1",
                "business_id": "history-mexican-2",
                "stars": 3.0,
                "text": "average mexican",
                "date": pd.Timestamp("2020-01-02"),
            },
            {
                "review_id": "history-3",
                "user_id": "user-1",
                "business_id": "history-coffee",
                "stars": 4.0,
                "text": "good coffee",
                "date": pd.Timestamp("2020-01-03"),
            },
        ]
    ).to_parquet(interactions_path, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "validation:user-1",
                "position": position,
                "review_id": review_id,
            }
            for position, review_id in enumerate(
                ["history-1", "history-2", "history-3"],
                start=1,
            )
        ]
    ).to_parquet(histories_path, index=False)
    return businesses_path, interactions_path, histories_path, candidates


def test_builds_category_profile_and_candidate_scores_from_task_history(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, candidates = (
        _write_category_fixture(tmp_path)
    )
    store = TemporalCategoryStore(
        TemporalDataView(businesses, interactions, interactions),
        broad_categories={"Restaurants", "Food", "Nightlife", "Shopping"},
    )
    task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )

    features = store.features_for(task)

    assert features.profile.history_count == 3
    assert features.profile.average_rating == pytest.approx(4.0)
    assert features.profile.rating_distribution == {
        "1": 0,
        "2": 0,
        "3": 1,
        "4": 1,
        "5": 1,
    }
    assert features.profile.preferred_categories == {
        "Coffee & Tea": pytest.approx(0.675),
        "Mexican": pytest.approx(0.825),
    }
    assert features.category_scores[candidates[0]] == pytest.approx(0.825)
    assert features.category_scores[candidates[1]] == pytest.approx(0.675)
    assert features.category_scores[candidates[2]] == pytest.approx(0.0)


def test_test_profile_includes_the_validation_behavior(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, candidates = (
        _write_category_fixture(tmp_path)
    )
    business_rows = pd.read_parquet(businesses)
    pd.concat(
        [
            business_rows,
            pd.DataFrame(
                [
                    _business(
                        "validation-business",
                        ["Restaurants", "Italian"],
                    )
                ]
            ),
        ],
        ignore_index=True,
    ).to_parquet(businesses, index=False)
    interaction_rows = pd.read_parquet(interactions)
    pd.concat(
        [
            interaction_rows,
            pd.DataFrame(
                [
                    {
                        "review_id": "validation-review",
                        "user_id": "user-1",
                        "business_id": "validation-business",
                        "stars": 5.0,
                        "text": "great italian",
                        "date": pd.Timestamp("2020-02-01"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).to_parquet(interactions, index=False)
    history_rows = pd.read_parquet(histories)
    test_history = pd.DataFrame(
        [
            {
                "task_id": "test:user-1",
                "position": position,
                "review_id": review_id,
            }
            for position, review_id in enumerate(
                [
                    "history-1",
                    "history-2",
                    "history-3",
                    "validation-review",
                ],
                start=1,
            )
        ]
    )
    pd.concat([history_rows, test_history], ignore_index=True).to_parquet(
        histories,
        index=False,
    )
    store = TemporalCategoryStore(
        TemporalDataView(businesses, interactions, interactions),
        broad_categories={"Restaurants", "Food", "Nightlife", "Shopping"},
    )
    validation_task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )
    test_task = RecommendationTask(
        task_id="test:user-1",
        user_id="user-1",
        cutoff_time="2020-03-01T00:00:00",
        candidate_business_ids=candidates,
    )

    validation_profile = store.features_for(validation_task).profile
    test_profile = store.features_for(test_task).profile

    assert validation_profile.history_count == 3
    assert "Italian" not in validation_profile.preferred_categories
    assert test_profile.history_count == 4
    assert test_profile.preferred_categories["Italian"] == pytest.approx(0.85)


def test_unreferenced_future_interaction_cannot_change_frozen_profile(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, candidates = (
        _write_category_fixture(tmp_path)
    )
    task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )
    kwargs = {
        "broad_categories": {
            "Restaurants",
            "Food",
            "Nightlife",
            "Shopping",
        }
    }
    before = TemporalCategoryStore(
        TemporalDataView(businesses, interactions, interactions),
        **kwargs,
    ).features_for(task)
    interaction_rows = pd.read_parquet(interactions)
    future = pd.DataFrame(
        [
            {
                "review_id": "future-review",
                "user_id": "user-1",
                "business_id": "history-coffee",
                "stars": 1.0,
                "text": "future",
                "date": pd.Timestamp("2030-01-01"),
            }
        ]
    )
    pd.concat([interaction_rows, future], ignore_index=True).to_parquet(
        interactions,
        index=False,
    )

    after = TemporalCategoryStore(
        TemporalDataView(businesses, interactions, interactions),
        **kwargs,
    ).features_for(task)

    assert after == before
