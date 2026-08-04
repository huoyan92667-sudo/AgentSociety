from __future__ import annotations

from pathlib import Path

import pytest

from yelp_agent.config import load_config
from yelp_agent.evaluation.data_usage import (
    DataUsageViolation,
    UserFoldSummary,
    assign_development_task_folds,
    assign_user_fold,
    build_validation_user_fold_summary,
    deterministic_bootstrap_users,
    write_user_fold_summary,
)
from yelp_agent.models import RecommendationTask


PROJECT_ROOT = Path(__file__).parents[1]


def _task(task_id: str, user_id: str) -> RecommendationTask:
    return RecommendationTask(
        task_id=task_id,
        user_id=user_id,
        cutoff_time="2020-01-01T00:00:00",
        candidate_business_ids=[
            f"business-{index:02d}" for index in range(20)
        ],
    )


def test_user_folds_are_stable_and_keep_one_user_together() -> None:
    policy = load_config().evaluation_data_usage
    tasks = [
        _task("validation:user-a:1", "user-a"),
        _task("validation:user-b:1", "user-b"),
        _task("validation:user-a:2", "user-a"),
    ]

    forward = assign_development_task_folds(tasks, policy)
    reversed_order = assign_development_task_folds(
        list(reversed(tasks)),
        policy,
    )

    assert forward == reversed_order
    assert forward["validation:user-a:1"] == forward["validation:user-a:2"]
    assert forward["validation:user-a:1"] == assign_user_fold(
        "user-a", policy
    )
    assert all(1 <= fold <= 5 for fold in forward.values())


def test_model_selection_rejects_legacy_test_tasks() -> None:
    policy = load_config().evaluation_data_usage

    with pytest.raises(DataUsageViolation, match="validation tasks only"):
        assign_development_task_folds(
            [_task("test:user-a", "user-a")],
            policy,
        )


def test_bootstrap_samples_users_reproducibly_not_tasks() -> None:
    policy = load_config().evaluation_data_usage
    users = ["user-d", "user-b", "user-a", "user-c"]

    first = deterministic_bootstrap_users(
        users,
        replicate_index=17,
        policy=policy,
    )
    reordered = deterministic_bootstrap_users(
        list(reversed(users)),
        replicate_index=17,
        policy=policy,
    )

    assert first == reordered
    assert len(first) == len(users)
    assert set(first).issubset(users)


def test_fold_summary_is_anonymous_and_byte_stable(tmp_path: Path) -> None:
    tasks_path = tmp_path / "validation_tasks.jsonl"
    tasks = [
        _task("validation:user-a:1", "user-a"),
        _task("validation:user-b:1", "user-b"),
        _task("validation:user-a:2", "user-a"),
    ]
    tasks_path.write_text(
        "".join(task.model_dump_json() + "\n" for task in tasks),
        encoding="utf-8",
    )
    policy = load_config().evaluation_data_usage

    summary = build_validation_user_fold_summary(tasks_path, policy)
    output = tmp_path / "current_validation_user_fold_summary.json"
    write_user_fold_summary(summary, output)
    first_bytes = output.read_bytes()
    first_mtime = output.stat().st_mtime_ns
    write_user_fold_summary(summary, output)

    assert summary.task_count == 3
    assert summary.user_count == 2
    assert summary.multi_task_user_count == 1
    assert sum(fold.user_count for fold in summary.folds.values()) == 2
    assert sum(fold.task_count for fold in summary.folds.values()) == 3
    assert summary.user_ids_included is False
    assert summary.task_ids_included is False
    assert b"user-a" not in first_bytes
    assert b"user-b" not in first_bytes
    assert output.read_bytes() == first_bytes
    assert output.stat().st_mtime_ns == first_mtime


def test_tracked_validation_fold_summary_matches_current_dataset() -> None:
    summary_path = (
        PROJECT_ROOT
        / "docs"
        / "evaluation"
        / "current_validation_user_fold_summary.json"
    )
    summary = UserFoldSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )

    assert summary.task_count == 4_971
    assert summary.user_count == 4_971
    assert summary.multi_task_user_count == 0
    assert sum(fold.user_count for fold in summary.folds.values()) == 4_971
    assert sum(fold.task_count for fold in summary.folds.values()) == 4_971
    assert summary.user_ids_included is False
    assert summary.task_ids_included is False
    assert summary.legacy_test_status == "previously_observed"
    assert summary.legacy_test_usage == "historical_comparison_only"
    assert summary.strict_blind_holdout is False
