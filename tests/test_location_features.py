import math
from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.features.location import TemporalLocationStore
from yelp_agent.models import RecommendationTask


def _business(
    business_id: str,
    latitude: float | None,
    longitude: float | None,
) -> dict[str, object]:
    return {
        "business_id": business_id,
        "name": business_id,
        "address": "",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "",
        "latitude": latitude,
        "longitude": longitude,
        "categories": ["Restaurants"],
        "attributes_json": "{}",
    }


def _write_location_fixture(
    root: Path,
    *,
    history_coordinates: tuple[float | None, float | None] = (
        39.9526,
        -75.1652,
    ),
) -> tuple[Path, Path, Path, list[str]]:
    businesses_path = root / "businesses.parquet"
    interactions_path = root / "interactions.parquet"
    histories_path = root / "histories.parquet"
    candidates = [f"candidate-{index:02d}" for index in range(20)]
    center_latitude, center_longitude = history_coordinates
    pd.DataFrame(
        [
            _business(
                "history-business",
                center_latitude,
                center_longitude,
            ),
            _business(candidates[0], 39.9526, -75.1652),
            _business(candidates[1], 39.9526 + 0.089932, -75.1652),
            *[
                _business(business_id, 39.9526, -75.1652)
                for business_id in candidates[2:]
            ],
        ]
    ).to_parquet(businesses_path, index=False)
    pd.DataFrame(
        [
            {
                "review_id": "history-review",
                "user_id": "user-1",
                "business_id": "history-business",
                "stars": 5.0,
                "text": "history",
                "date": pd.Timestamp("2020-01-01"),
            }
        ]
    ).to_parquet(interactions_path, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "validation:user-1",
                "position": 1,
                "review_id": "history-review",
            }
        ]
    ).to_parquet(histories_path, index=False)
    return businesses_path, interactions_path, histories_path, candidates


def test_scores_candidate_distance_from_dynamic_history_center(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, candidates = (
        _write_location_fixture(tmp_path)
    )
    store = TemporalLocationStore(
        TemporalDataView(businesses, interactions, interactions),
        scale_km=10.0,
    )
    task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )

    features = store.features_for(task)
    lightweight = store.score_candidates(task)

    assert features.location_center is not None
    assert features.location_center.latitude == pytest.approx(39.9526)
    assert features.location_center.longitude == pytest.approx(-75.1652)
    assert features.history_coordinate_count == 1
    assert features.business_scores[candidates[0]].distance_km == pytest.approx(
        0.0
    )
    assert features.business_scores[candidates[0]].location_score == pytest.approx(
        1.0
    )
    assert features.business_scores[candidates[1]].distance_km == pytest.approx(
        10.0,
        rel=0.02,
    )
    assert features.business_scores[candidates[1]].location_score == pytest.approx(
        math.exp(-1.0),
        rel=0.02,
    )
    for index, business_id in enumerate(lightweight.business_ids):
        expected = features.business_scores[business_id]
        assert lightweight.location_scores[index] == pytest.approx(
            expected.location_score
        )
        assert lightweight.distances_km[index] == pytest.approx(
            expected.distance_km
        )


def test_missing_history_or_candidate_coordinates_return_neutral_score(
    tmp_path: Path,
) -> None:
    missing_history_root = tmp_path / "missing-history"
    missing_history_root.mkdir()
    businesses, interactions, histories, candidates = _write_location_fixture(
        missing_history_root,
        history_coordinates=(None, None),
    )
    task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )
    missing_history_features = TemporalLocationStore(
        TemporalDataView(businesses, interactions, interactions),
    ).features_for(task)

    assert missing_history_features.location_center is None
    assert missing_history_features.history_coordinate_count == 0
    assert all(
        score.location_score == pytest.approx(0.5)
        and score.distance_km is None
        for score in missing_history_features.business_scores.values()
    )

    missing_candidate_root = tmp_path / "missing-candidate"
    missing_candidate_root.mkdir()
    businesses, interactions, histories, candidates = _write_location_fixture(
        missing_candidate_root
    )
    business_rows = pd.read_parquet(businesses)
    business_rows.loc[
        business_rows["business_id"] == candidates[0],
        ["latitude", "longitude"],
    ] = None
    business_rows.to_parquet(businesses, index=False)
    candidate_features = TemporalLocationStore(
        TemporalDataView(businesses, interactions, interactions),
    ).features_for(task)

    missing_candidate = candidate_features.business_scores[candidates[0]]
    assert candidate_features.location_center is not None
    assert missing_candidate.distance_km is None
    assert missing_candidate.location_score == pytest.approx(0.5)


def test_test_center_adds_validation_behavior_without_changing_old_center(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, candidates = (
        _write_location_fixture(tmp_path)
    )
    business_rows = pd.read_parquet(businesses)
    pd.concat(
        [
            business_rows,
            pd.DataFrame(
                [
                    _business(
                        "validation-business",
                        40.0526,
                        -75.1652,
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
                        "text": "validation",
                        "date": pd.Timestamp("2020-02-01"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).to_parquet(interactions, index=False)
    history_rows = pd.read_parquet(histories)
    pd.concat(
        [
            history_rows,
            pd.DataFrame(
                [
                    {
                        "task_id": "test:user-1",
                        "position": 1,
                        "review_id": "history-review",
                    },
                    {
                        "task_id": "test:user-1",
                        "position": 2,
                        "review_id": "validation-review",
                    },
                ]
            ),
        ],
        ignore_index=True,
    ).to_parquet(histories, index=False)
    store = TemporalLocationStore(
        TemporalDataView(businesses, interactions, interactions),
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

    validation_features = store.features_for(validation_task)
    test_features = store.features_for(test_task)

    assert validation_features.location_center is not None
    assert test_features.location_center is not None
    assert validation_features.location_center.latitude == pytest.approx(39.9526)
    assert test_features.location_center.latitude == pytest.approx(40.0026)
    assert validation_features.business_scores[
        candidates[0]
    ].location_score == pytest.approx(1.0)
    assert (
        test_features.business_scores[candidates[0]].location_score
        < validation_features.business_scores[candidates[0]].location_score
    )
