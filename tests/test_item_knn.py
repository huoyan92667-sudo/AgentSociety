from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from yelp_agent.collaborative.artifacts import build_item_knn_artifacts
from yelp_agent.collaborative.item_knn import (
    ItemKNNHistoryEvent,
    ItemKNNRequest,
    TemporalItemKNNStore,
)
from yelp_agent.config import ItemKNNConfig


def _config(**updates: object) -> ItemKNNConfig:
    values: dict[str, object] = {
        "shrinkage_beta": 1.0,
        "half_life_days": None,
        "reserved_tail_interactions": 0,
    }
    values.update(updates)
    return ItemKNNConfig.model_validate(values)


def _write_interactions(path: Path) -> None:
    rows = [
        ("positive-1", "p1-a", "a", 5.0, "2020-01-01"),
        ("positive-1", "p1-b", "b", 4.0, "2020-01-02"),
        ("positive-2", "p2-a", "a", 4.0, "2020-01-03"),
        ("positive-2", "p2-b", "b", 5.0, "2020-01-04"),
        ("negative-1", "n1-c", "c", 1.0, "2020-01-01"),
        ("negative-1", "n1-d", "d", 2.0, "2020-01-02"),
        ("negative-2", "n2-c", "c", 2.0, "2020-01-03"),
        ("negative-2", "n2-d", "d", 1.0, "2020-01-04"),
        ("neutral-1", "m1-e", "e", 3.0, "2020-01-01"),
        ("neutral-1", "m1-f", "f", 3.0, "2020-01-02"),
        ("target-user", "target-a", "a", 5.0, "2020-01-10"),
        ("target-user", "target-c", "c", 1.0, "2020-01-11"),
    ]
    pd.DataFrame(
        [
            {
                "user_id": user_id,
                "review_id": review_id,
                "business_id": business_id,
                "stars": stars,
                "text": "",
                "date": pd.Timestamp(date),
            }
            for user_id, review_id, business_id, stars, date in rows
        ]
    ).to_parquet(path, index=False)


def test_item_knn_keeps_positive_and_negative_evidence_separate(
    tmp_path: Path,
) -> None:
    interactions = tmp_path / "interactions.parquet"
    _write_interactions(interactions)
    store = TemporalItemKNNStore(interactions, _config())

    result = store.score_candidates(
        ItemKNNRequest(
            user_id="target-user",
            cutoff_time=datetime(2020, 2, 1),
            candidate_business_ids=("b", "d", "f"),
            history=(
                ItemKNNHistoryEvent(
                    business_id="a",
                    stars=5.0,
                    date=datetime(2020, 1, 10),
                ),
                ItemKNNHistoryEvent(
                    business_id="c",
                    stars=1.0,
                    date=datetime(2020, 1, 11),
                ),
            ),
        )
    )
    by_business = {score.business_id: score for score in result.scores}

    expected_similarity = 2.0 / (3.0 * 2.0) ** 0.5 * (2.0 / 3.0)
    assert by_business["b"].positive_score == pytest.approx(expected_similarity)
    assert by_business["b"].positive_support_count == 2
    assert by_business["b"].negative_evidence == 0.0
    assert by_business["d"].negative_evidence == pytest.approx(expected_similarity)
    assert by_business["d"].negative_support_count == 2
    assert by_business["d"].positive_score == 0.0
    assert by_business["f"].positive_score == 0.0
    assert by_business["f"].negative_evidence == 0.0
    assert result.positive_history_count == 1
    assert result.negative_history_count == 1
    assert result.missing is False


def test_item_knn_half_life_prefers_recent_collaborative_evidence(
    tmp_path: Path,
) -> None:
    interactions = tmp_path / "interactions.parquet"
    rows = [
        ("old-user", "old-a", "a", 5.0, "2018-01-01"),
        ("old-user", "old-b", "b", 5.0, "2018-01-02"),
        ("recent-user", "recent-a", "a", 5.0, "2020-12-01"),
        ("recent-user", "recent-c", "c", 5.0, "2020-12-02"),
        ("target-user", "target-a", "a", 5.0, "2020-12-15"),
    ]
    pd.DataFrame(
        [
            {
                "user_id": user_id,
                "review_id": review_id,
                "business_id": business_id,
                "stars": stars,
                "date": pd.Timestamp(date),
            }
            for user_id, review_id, business_id, stars, date in rows
        ]
    ).to_parquet(interactions, index=False)

    store = TemporalItemKNNStore(
        interactions,
        _config(half_life_days=365),
    )
    result = store.score_candidates(
        ItemKNNRequest(
            user_id="target-user",
            cutoff_time=datetime(2021, 1, 1),
            candidate_business_ids=("b", "c"),
            history=(
                ItemKNNHistoryEvent(
                    business_id="a",
                    stars=5.0,
                    date=datetime(2020, 12, 15),
                ),
            ),
        )
    )
    scores = {item.business_id: item.positive_score for item in result.scores}

    assert scores["c"] > scores["b"]


def test_item_knn_uses_latest_rating_before_each_cutoff_in_any_call_order(
    tmp_path: Path,
) -> None:
    interactions = tmp_path / "interactions.parquet"
    rows = [
        ("changing-user", "change-a", "a", 5.0, "2020-01-01"),
        ("changing-user", "change-b-like", "b", 5.0, "2020-01-02"),
        ("stable-user", "stable-a", "a", 5.0, "2020-01-01"),
        ("stable-user", "stable-b", "b", 5.0, "2020-01-02"),
        ("target-user", "target-a", "a", 5.0, "2020-01-03"),
        ("changing-user", "change-b-dislike", "b", 1.0, "2020-01-20"),
        ("future-user", "future-a", "a", 5.0, "2030-01-01"),
        ("future-user", "future-b", "b", 5.0, "2030-01-02"),
    ]
    pd.DataFrame(
        [
            {
                "user_id": user_id,
                "review_id": review_id,
                "business_id": business_id,
                "stars": stars,
                "date": pd.Timestamp(date),
            }
            for user_id, review_id, business_id, stars, date in rows
        ]
    ).to_parquet(interactions, index=False)
    store = TemporalItemKNNStore(interactions, _config())

    later = store.score_candidates(
        ItemKNNRequest(
            user_id="target-user",
            cutoff_time=datetime(2020, 2, 1),
            candidate_business_ids=("b",),
            history=(
                ItemKNNHistoryEvent(
                    business_id="a",
                    stars=5.0,
                    date=datetime(2020, 1, 3),
                ),
            ),
        )
    )
    earlier = store.score_candidates(
        ItemKNNRequest(
            user_id="target-user",
            cutoff_time=datetime(2020, 1, 10),
            candidate_business_ids=("b",),
            history=(
                ItemKNNHistoryEvent(
                    business_id="a",
                    stars=5.0,
                    date=datetime(2020, 1, 3),
                ),
            ),
        )
    )

    assert earlier.scores[0].positive_support_count == 2
    assert later.scores[0].positive_support_count == 1
    assert earlier.scores[0].positive_score > later.scores[0].positive_score


def test_graph_exclusions_do_not_remove_visible_request_history(
    tmp_path: Path,
) -> None:
    interactions = tmp_path / "interactions.parquet"
    rows = [
        ("graph-user", "graph-a", "a", 5.0, "2020-01-01"),
        ("graph-user", "graph-b", "b", 5.0, "2020-01-02"),
        ("graph-user", "reserved-x", "x", 5.0, "2020-01-03"),
        ("graph-user", "reserved-y", "y", 5.0, "2020-01-04"),
        ("excluded-user", "excluded-a", "a", 5.0, "2020-01-01"),
        ("excluded-user", "excluded-b", "b", 5.0, "2020-01-02"),
    ]
    pd.DataFrame(
        [
            {
                "user_id": user_id,
                "review_id": review_id,
                "business_id": business_id,
                "stars": stars,
                "date": pd.Timestamp(date),
            }
            for user_id, review_id, business_id, stars, date in rows
        ]
    ).to_parquet(interactions, index=False)
    store = TemporalItemKNNStore(
        interactions,
        _config(reserved_tail_interactions=2),
        excluded_user_ids={"excluded-user", "request-user"},
    )

    result = store.score_candidates(
        ItemKNNRequest(
            user_id="request-user",
            cutoff_time=datetime(2020, 2, 1),
            candidate_business_ids=("b", "y"),
            history=(
                ItemKNNHistoryEvent(
                    business_id="a",
                    stars=5.0,
                    date=datetime(2020, 1, 10),
                ),
                ItemKNNHistoryEvent(
                    business_id="x",
                    stars=5.0,
                    date=datetime(2020, 1, 11),
                ),
            ),
        )
    )
    scores = {item.business_id: item for item in result.scores}

    assert scores["b"].positive_support_count == 1
    assert scores["b"].positive_score > 0.0
    assert scores["y"].positive_support_count == 0
    assert scores["y"].positive_score == 0.0


def test_item_knn_marks_missing_when_history_has_no_collaborative_neighbors(
    tmp_path: Path,
) -> None:
    interactions = tmp_path / "interactions.parquet"
    _write_interactions(interactions)
    store = TemporalItemKNNStore(interactions, _config())

    result = store.score_candidates(
        ItemKNNRequest(
            user_id="cold-user",
            cutoff_time=datetime(2020, 2, 1),
            candidate_business_ids=("b", "d"),
            history=(
                ItemKNNHistoryEvent(
                    business_id="unseen-business",
                    stars=5.0,
                    date=datetime(2020, 1, 20),
                ),
            ),
        )
    )

    assert result.missing is True
    assert all(item.positive_score == 0.0 for item in result.scores)
    assert all(item.negative_evidence == 0.0 for item in result.scores)


def test_item_knn_artifact_excludes_reserved_targets_and_users(
    tmp_path: Path,
) -> None:
    interactions = tmp_path / "interactions.parquet"
    rows = [
        ("kept-user", "kept-positive", "a", 5.0, "2020-01-01"),
        ("kept-user", "kept-negative", "b", 1.0, "2020-01-02"),
        ("kept-user", "reserved-neutral", "c", 3.0, "2020-01-03"),
        ("kept-user", "reserved-positive", "d", 5.0, "2020-01-04"),
        ("excluded-user", "excluded-positive", "e", 5.0, "2020-01-01"),
        ("excluded-user", "excluded-negative", "f", 1.0, "2020-01-02"),
    ]
    pd.DataFrame(
        [
            {
                "user_id": user_id,
                "review_id": review_id,
                "business_id": business_id,
                "stars": stars,
                "date": pd.Timestamp(date),
            }
            for user_id, review_id, business_id, stars, date in rows
        ]
    ).to_parquet(interactions, index=False)

    first = build_item_knn_artifacts(
        interactions,
        tmp_path / "item-knn",
        _config(reserved_tail_interactions=2),
        excluded_user_ids={"excluded-user"},
    )
    reused = build_item_knn_artifacts(
        interactions,
        tmp_path / "item-knn",
        _config(reserved_tail_interactions=2),
        excluded_user_ids={"excluded-user"},
    )

    assert first.status == "written"
    assert reused.status == "skipped"
    assert first.retained_interactions == 2
    assert first.reserved_interactions == 2
    assert first.excluded_user_interactions == 2
    assert pq.read_table(first.positive_events_path).column(
        "review_id"
    ).to_pylist() == ["kept-positive"]
    assert pq.read_table(first.negative_events_path).column(
        "review_id"
    ).to_pylist() == ["kept-negative"]
    assert pq.read_table(first.neutral_events_path).num_rows == 0
