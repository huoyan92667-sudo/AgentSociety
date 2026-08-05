from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.config import RollingTrainingConfig, load_config
from yelp_agent.data.rolling_training import (
    MinimalTrainTaskReference,
    RollingTrainingError,
    RollingTrainingManifest,
    build_rolling_training_tasks,
    write_rolling_training_report,
)
from yelp_agent.data.temporal import build_temporal_splits
from yelp_agent.evaluation.data_usage import assign_user_fold


PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _interaction(
    review_id: str,
    user_id: str,
    business_id: str,
    date: str,
) -> dict[str, object]:
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


def _user_rows(user_id: str, count: int) -> list[dict[str, object]]:
    return [
        _interaction(
            f"{user_id}-r{index}",
            user_id,
            f"business-{user_id}-{index}",
            f"2020-01-{index:02d}",
        )
        for index in range(1, count + 1)
    ]


def _training_config(**updates: object) -> RollingTrainingConfig:
    values: dict[str, object] = {
        "minimum_history_count": 2,
        "maximum_tasks_per_user": 3,
        "minimum_target_gap": 2,
        "minimum_sample_weight": 0.5,
        "maximum_sample_weight": 1.0,
        "selection_strategy": "evenly_spaced_include_latest",
        "weighting_strategy": "linear_by_lifecycle_position",
    }
    values.update(updates)
    return RollingTrainingConfig.model_validate(values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(
    tmp_path: Path,
    rows: list[dict[str, object]],
) -> tuple[Path, Path, Path]:
    interactions = tmp_path / "data" / "processed" / "interactions.parquet"
    interactions.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(interactions, index=False)
    task_root = tmp_path / "data" / "task_dataset"
    temporal = build_temporal_splits(interactions, task_root)
    return interactions, Path(temporal.contexts_path), task_root


def _build(
    interactions: Path,
    frozen_contexts: Path,
    task_root: Path,
    config: RollingTrainingConfig | None = None,
    *,
    force: bool = False,
):
    policy = load_config(PROJECT_CONFIG_DIR).evaluation_data_usage
    return build_rolling_training_tasks(
        interactions,
        frozen_contexts,
        task_root,
        config or _training_config(),
        policy,
        force=force,
    )


def test_builds_multiple_time_safe_tasks_and_isolates_labels(
    tmp_path: Path,
) -> None:
    interactions, frozen_contexts, task_root = _sources(
        tmp_path,
        [*_user_rows("user-a", 10), *_user_rows("user-b", 5)],
    )
    protected = {
        path: _sha256(path)
        for path in task_root.rglob("*")
        if path.is_file()
    }

    result = _build(interactions, frozen_contexts, task_root)

    contexts = pd.read_parquet(result.contexts_path).sort_values(
        ["user_id", "target_position"]
    )
    histories = pd.read_parquet(result.histories_path)
    truth = pd.read_parquet(result.ground_truth_path)
    minimal = [
        MinimalTrainTaskReference.model_validate_json(line)
        for line in Path(result.minimal_train_tasks_path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    policy = load_config(PROJECT_CONFIG_DIR).evaluation_data_usage

    assert result.status == "written"
    assert result.source_users == 2
    assert result.users_with_tasks == 2
    assert result.train_tasks == 4
    assert result.minimal_train_tasks == 2
    assert contexts.groupby("user_id")["history_count"].apply(list).to_dict() == {
        "user-a": [2, 4, 7],
        "user-b": [2],
    }
    assert contexts.groupby("user_id")["sample_weight"].apply(list).to_dict() == {
        "user-a": [0.5, 0.7, 1.0],
        "user-b": [1.0],
    }
    assert set(contexts["fold"]) == {
        assign_user_fold("user-a", policy),
        assign_user_fold("user-b", policy),
    }
    assert "target_business_id" not in contexts.columns
    assert "target_review_id" not in contexts.columns
    assert "target_business_id" not in histories.columns
    assert "target_review_id" not in histories.columns

    history_ids = set(histories["review_id"])
    target_ids = set(truth["target_review_id"])
    reserved_ids = {
        "user-a-r9",
        "user-a-r10",
        "user-b-r4",
        "user-b-r5",
    }
    assert not reserved_ids.intersection(history_ids)
    assert not reserved_ids.intersection(target_ids)
    for task_id, target_review_id in truth[
        ["task_id", "target_review_id"]
    ].itertuples(index=False):
        task_history = set(
            histories.loc[histories["task_id"] == task_id, "review_id"]
        )
        assert target_review_id not in task_history

    newest_by_user = set(
        contexts.sort_values("target_position")
        .groupby("user_id")
        .tail(1)["task_id"]
    )
    assert {item.task_id for item in minimal} == newest_by_user
    assert all(_sha256(path) == digest for path, digest in protected.items())
    assert not (task_root / "legacy_candidates").exists()


def test_reuses_and_force_rebuilds_byte_identical_outputs(tmp_path: Path) -> None:
    interactions, frozen_contexts, task_root = _sources(
        tmp_path,
        _user_rows("user-a", 12),
    )
    first = _build(interactions, frozen_contexts, task_root)
    artifact_paths = [
        Path(first.contexts_path),
        Path(first.histories_path),
        Path(first.ground_truth_path),
        Path(first.minimal_train_tasks_path),
        Path(first.manifest_path),
    ]
    first_bytes = {path: path.read_bytes() for path in artifact_paths}
    mtimes = {path: path.stat().st_mtime_ns for path in artifact_paths}

    second = _build(interactions, frozen_contexts, task_root)

    assert second.status == "skipped"
    assert second.manifest_sha256 == first.manifest_sha256
    assert all(path.stat().st_mtime_ns == mtimes[path] for path in artifact_paths)

    rebuilt = _build(interactions, frozen_contexts, task_root, force=True)
    assert rebuilt.status == "written"
    assert all(path.read_bytes() == first_bytes[path] for path in artifact_paths)


def test_rejects_configuration_drift_without_force(tmp_path: Path) -> None:
    interactions, frozen_contexts, task_root = _sources(
        tmp_path,
        _user_rows("user-a", 12),
    )
    _build(interactions, frozen_contexts, task_root)

    with pytest.raises(RollingTrainingError, match="configuration changed"):
        _build(
            interactions,
            frozen_contexts,
            task_root,
            _training_config(maximum_tasks_per_user=2),
        )


def test_caps_long_histories_while_covering_early_and_recent_lifecycle(
    tmp_path: Path,
) -> None:
    interactions, frozen_contexts, task_root = _sources(
        tmp_path,
        _user_rows("user-a", 20),
    )

    result = _build(interactions, frozen_contexts, task_root)
    contexts = pd.read_parquet(result.contexts_path).sort_values(
        "target_position"
    )

    assert list(contexts["history_count"]) == [2, 10, 17]
    assert list(contexts["sample_weight"]) == [0.5, 0.766666666667, 1.0]
    assert len(contexts) == 3
    assert contexts.iloc[-1]["target_position"] == 18


def test_rejects_interactions_that_no_longer_match_frozen_splits(
    tmp_path: Path,
) -> None:
    rows = _user_rows("user-a", 10)
    interactions, frozen_contexts, task_root = _sources(tmp_path, rows)
    pd.DataFrame(
        [
            *rows,
            _interaction(
                "user-a-r11",
                "user-a",
                "business-user-a-11",
                "2020-01-11",
            ),
        ]
    ).to_parquet(interactions, index=False)

    with pytest.raises(RollingTrainingError, match="contexts disagree"):
        _build(interactions, frozen_contexts, task_root)


def test_skips_unsafe_early_cutoff_ties(tmp_path: Path) -> None:
    rows = _user_rows("user-a", 8)
    rows[2]["date"] = rows[1]["date"]
    interactions, frozen_contexts, task_root = _sources(tmp_path, rows)

    result = _build(interactions, frozen_contexts, task_root)
    contexts = pd.read_parquet(result.contexts_path)

    assert result.skipped_cutoff_ties == 1
    assert (contexts["history_max_time"] < contexts["cutoff_time"]).all()


def test_writes_manifest_and_human_facing_report(tmp_path: Path) -> None:
    interactions, frozen_contexts, task_root = _sources(
        tmp_path,
        _user_rows("user-a", 10),
    )
    result = _build(interactions, frozen_contexts, task_root)
    report_path = tmp_path / "runs" / "rolling_training_build_report.json"

    write_rolling_training_report(result, report_path)

    manifest = RollingTrainingManifest.model_validate_json(
        Path(result.manifest_path).read_text(encoding="utf-8")
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert manifest.reserved_splits == ["validation", "test"]
    assert manifest.legacy_candidate_files_read is False
    assert manifest.target_history_leaks == 0
    assert manifest.reserved_review_leaks == 0
    assert report["train_tasks"] == result.train_tasks
    assert report["manifest_sha256"] == result.manifest_sha256
