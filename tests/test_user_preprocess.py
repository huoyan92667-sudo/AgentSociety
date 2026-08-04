import hashlib
import json
from pathlib import Path

import pandas as pd

from yelp_agent.config import load_config
from yelp_agent.data.users import (
    UserPreprocessResult,
    preprocess_users_and_interactions,
    write_user_preprocess_report,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def interaction(
    review_id: str,
    user_id: str,
    business_id: str,
    stars: float,
    date: str,
) -> dict:
    return {
        "review_id": review_id,
        "user_id": user_id,
        "business_id": business_id,
        "stars": stars,
        "useful": 0,
        "funny": 0,
        "cool": 0,
        "text": review_id,
        "date": pd.Timestamp(date),
    }


def user(user_id: str, name: str) -> dict:
    return {
        "user_id": user_id,
        "name": name,
        "yelping_since": "2010-01-02 03:04:05",
        "review_count": 999,
        "average_stars": 5.0,
    }


def test_keeps_only_users_meeting_all_interaction_thresholds(
    tmp_path: Path,
) -> None:
    reviews_path = tmp_path / "reviews.parquet"
    raw_users_path = tmp_path / "users.json"
    users_output_path = tmp_path / "processed" / "users.parquet"
    interactions_output_path = tmp_path / "processed" / "interactions.parquet"
    pd.DataFrame(
        [
            interaction("r3", "eligible", "b3", 5, "2020-01-03"),
            interaction("r1", "eligible", "b1", 1, "2020-01-01"),
            interaction("r2", "eligible", "b2", 3, "2020-01-02"),
            interaction("r4", "too-few-businesses", "b1", 1, "2020-01-01"),
            interaction("r5", "too-few-businesses", "b1", 3, "2020-01-02"),
            interaction("r6", "too-few-businesses", "b2", 5, "2020-01-03"),
            interaction("r7", "too-few-reviews", "b1", 1, "2020-01-01"),
            interaction("r8", "too-few-reviews", "b2", 3, "2020-01-02"),
        ]
    ).to_parquet(reviews_path, index=False)
    write_jsonl(
        raw_users_path,
        [
            user("too-few-reviews", "Few"),
            user("eligible", "Selected"),
            user("too-few-businesses", "Narrow"),
        ],
    )
    config = load_config().data.model_copy(
        update={
            "min_user_reviews": 3,
            "min_distinct_businesses": 3,
            "min_distinct_ratings": 3,
            "max_users": 10,
        }
    )

    result = preprocess_users_and_interactions(
        reviews_path,
        raw_users_path,
        users_output_path,
        interactions_output_path,
        config,
    )

    users = pd.read_parquet(users_output_path)
    interactions = pd.read_parquet(interactions_output_path)
    assert result.status == "written"
    assert result.eligible_users == 1
    assert result.selected_users == 1
    assert result.selected_interactions == 3
    assert users["user_id"].tolist() == ["eligible"]
    assert users["name"].tolist() == ["Selected"]
    assert users["interaction_count"].tolist() == [3]
    assert users["distinct_business_count"].tolist() == [3]
    assert users["distinct_rating_count"].tolist() == [3]
    assert interactions["review_id"].tolist() == ["r1", "r2", "r3"]


def test_reuses_matching_frozen_outputs_without_rescanning_raw_users(
    tmp_path: Path,
) -> None:
    reviews_path = tmp_path / "reviews.parquet"
    raw_users_path = tmp_path / "users.json"
    users_output_path = tmp_path / "users.parquet"
    interactions_output_path = tmp_path / "interactions.parquet"
    pd.DataFrame(
        [
            interaction("r1", "eligible", "b1", 1, "2020-01-01"),
            interaction("r2", "eligible", "b2", 3, "2020-01-02"),
            interaction("r3", "eligible", "b3", 5, "2020-01-03"),
        ]
    ).to_parquet(reviews_path, index=False)
    write_jsonl(raw_users_path, [user("eligible", "Selected")])
    config = load_config().data.model_copy(
        update={
            "min_user_reviews": 3,
            "min_distinct_businesses": 3,
            "min_distinct_ratings": 3,
        }
    )
    preprocess_users_and_interactions(
        reviews_path,
        raw_users_path,
        users_output_path,
        interactions_output_path,
        config,
    )
    users_mtime = users_output_path.stat().st_mtime_ns
    interactions_mtime = interactions_output_path.stat().st_mtime_ns
    raw_users_path.write_text("{invalid-now}\n", encoding="utf-8")

    result = preprocess_users_and_interactions(
        reviews_path,
        raw_users_path,
        users_output_path,
        interactions_output_path,
        config,
    )

    assert result.status == "skipped"
    assert result.source_users is None
    assert result.eligible_users is None
    assert result.selected_users == 1
    assert result.selected_interactions == 3
    assert users_output_path.stat().st_mtime_ns == users_mtime
    assert interactions_output_path.stat().st_mtime_ns == interactions_mtime


def test_sampling_is_seeded_and_independent_of_input_order(tmp_path: Path) -> None:
    reviews_path = tmp_path / "reviews.parquet"
    raw_users_path = tmp_path / "users.json"
    user_ids = ["user-d", "user-b", "user-a", "user-c"]
    rows = []
    for user_id in user_ids:
        rows.extend(
            [
                interaction(f"{user_id}-3", user_id, "b3", 5, "2020-01-03"),
                interaction(f"{user_id}-1", user_id, "b1", 1, "2020-01-01"),
                interaction(f"{user_id}-2", user_id, "b2", 3, "2020-01-02"),
            ]
        )
    pd.DataFrame(list(reversed(rows))).to_parquet(reviews_path, index=False)
    write_jsonl(raw_users_path, [user(user_id, user_id) for user_id in user_ids])
    config = load_config().data.model_copy(
        update={
            "min_user_reviews": 3,
            "min_distinct_businesses": 3,
            "min_distinct_ratings": 3,
            "max_users": 2,
            "random_seed": 42,
        }
    )

    preprocess_users_and_interactions(
        reviews_path,
        raw_users_path,
        tmp_path / "first-users.parquet",
        tmp_path / "first-interactions.parquet",
        config,
    )
    write_jsonl(
        raw_users_path,
        [user(user_id, user_id) for user_id in reversed(user_ids)],
    )
    preprocess_users_and_interactions(
        reviews_path,
        raw_users_path,
        tmp_path / "second-users.parquet",
        tmp_path / "second-interactions.parquet",
        config,
    )

    expected = sorted(
        user_ids,
        key=lambda user_id: (
            hashlib.sha256(f"42:{user_id}".encode()).hexdigest(),
            user_id,
        ),
    )[:2]
    first_users = pd.read_parquet(tmp_path / "first-users.parquet")
    second_users = pd.read_parquet(tmp_path / "second-users.parquet")
    assert first_users["user_id"].tolist() == sorted(expected)
    assert second_users["user_id"].tolist() == sorted(expected)
    assert (
        (tmp_path / "first-users.parquet").read_bytes()
        == (tmp_path / "second-users.parquet").read_bytes()
    )
    assert (
        (tmp_path / "first-interactions.parquet").read_bytes()
        == (tmp_path / "second-interactions.parquet").read_bytes()
    )


def test_writes_machine_readable_user_preprocess_report(tmp_path: Path) -> None:
    report_path = tmp_path / "runs" / "user_preprocess_report.json"
    result = UserPreprocessResult(
        status="written",
        reviews_path="reviews.parquet",
        raw_users_path="users.json",
        users_output_path="users.parquet",
        interactions_output_path="interactions.parquet",
        source_users=100,
        eligible_users=12,
        selected_users=10,
        selected_interactions=80,
    )

    write_user_preprocess_report(result, report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["eligible_users"] == 12
    assert payload["selected_users"] == 10
    assert payload["selected_interactions"] == 80
