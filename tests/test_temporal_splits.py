import json
from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.data.temporal import (
    TemporalSplitError,
    TemporalSplitResult,
    build_temporal_splits,
    write_temporal_split_report,
)


def interaction(
    review_id: str,
    user_id: str,
    business_id: str,
    date: str,
) -> dict:
    return {
        "review_id": review_id,
        "user_id": user_id,
        "business_id": business_id,
        "stars": 4.0,
        "useful": 0,
        "funny": 0,
        "cool": 0,
        "text": review_id,
        "date": pd.Timestamp(date),
    }


def test_builds_validation_and_test_with_isolated_targets(
    tmp_path: Path,
) -> None:
    interactions_path = tmp_path / "interactions.parquet"
    output_root = tmp_path / "task_dataset"
    pd.DataFrame(
        [
            interaction("r4", "user-1", "business-4", "2020-01-04"),
            interaction("r2", "user-1", "business-2", "2020-01-02"),
            interaction("r1", "user-1", "business-1", "2020-01-01"),
            interaction("r3", "user-1", "business-3", "2020-01-03"),
        ]
    ).to_parquet(interactions_path, index=False)

    result = build_temporal_splits(interactions_path, output_root)

    contexts = pd.read_parquet(result.contexts_path)
    histories = pd.read_parquet(result.histories_path)
    ground_truth = pd.read_parquet(result.ground_truth_path)
    validation_id = "validation:user-1"
    test_id = "test:user-1"
    validation = contexts.set_index("task_id").loc[validation_id]
    test = contexts.set_index("task_id").loc[test_id]

    assert result.status == "written"
    assert result.users == 1
    assert result.validation_tasks == 1
    assert result.test_tasks == 1
    assert validation["cutoff_time"] == pd.Timestamp("2020-01-03")
    assert validation["history_count"] == 2
    assert test["cutoff_time"] == pd.Timestamp("2020-01-04")
    assert test["history_count"] == 3

    histories_by_task = (
        histories.sort_values(["task_id", "position"])
        .groupby("task_id")["review_id"]
        .apply(list)
        .to_dict()
    )
    assert histories_by_task[validation_id] == ["r1", "r2"]
    assert histories_by_task[test_id] == ["r1", "r2", "r3"]

    targets = ground_truth.set_index("task_id")["target_business_id"].to_dict()
    assert targets == {
        validation_id: "business-3",
        test_id: "business-4",
    }
    assert "target_business_id" not in contexts.columns
    assert "target_business_id" not in histories.columns


def test_rejects_a_target_tied_with_its_history_cutoff(tmp_path: Path) -> None:
    interactions_path = tmp_path / "interactions.parquet"
    output_root = tmp_path / "task_dataset"
    pd.DataFrame(
        [
            interaction("r1", "user-1", "business-1", "2020-01-01"),
            interaction("r2", "user-1", "business-2", "2020-01-02"),
            interaction("r3", "user-1", "business-3", "2020-01-02"),
        ]
    ).to_parquet(interactions_path, index=False)

    with pytest.raises(TemporalSplitError, match="cutoff tie"):
        build_temporal_splits(interactions_path, output_root)

    assert not output_root.exists()


def test_reuses_consistent_frozen_temporal_outputs(tmp_path: Path) -> None:
    interactions_path = tmp_path / "interactions.parquet"
    output_root = tmp_path / "task_dataset"
    pd.DataFrame(
        [
            interaction("r1", "user-1", "business-1", "2020-01-01"),
            interaction("r2", "user-1", "business-2", "2020-01-02"),
            interaction("r3", "user-1", "business-3", "2020-01-03"),
        ]
    ).to_parquet(interactions_path, index=False)
    first = build_temporal_splits(interactions_path, output_root)
    paths = [
        Path(first.contexts_path),
        Path(first.histories_path),
        Path(first.ground_truth_path),
    ]
    mtimes = [path.stat().st_mtime_ns for path in paths]

    second = build_temporal_splits(interactions_path, output_root)

    assert second.status == "skipped"
    assert second.users == 1
    assert second.validation_tasks == 1
    assert second.test_tasks == 1
    assert second.history_rows == 3
    assert [path.stat().st_mtime_ns for path in paths] == mtimes


def test_writes_temporal_leakage_audit_report(tmp_path: Path) -> None:
    report_path = tmp_path / "runs" / "temporal_split_report.json"
    result = TemporalSplitResult(
        status="written",
        interactions_path="interactions.parquet",
        contexts_path="contexts.parquet",
        histories_path="histories.parquet",
        ground_truth_path="ground_truth.parquet",
        users=2,
        validation_tasks=2,
        test_tasks=2,
        history_rows=30,
        target_history_leaks=0,
        history_cutoff_violations=0,
        test_missing_validation=0,
    )

    write_temporal_split_report(result, report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["validation_tasks"] == 2
    assert payload["test_tasks"] == 2
    assert payload["target_history_leaks"] == 0
    assert payload["history_cutoff_violations"] == 0
