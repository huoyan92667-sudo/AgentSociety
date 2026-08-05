from __future__ import annotations

from pathlib import Path

import pytest

from yelp_agent.experiments import (
    TaskFileError,
    TaskSplitError,
    read_recommendation_tasks,
    write_json_artifact,
    write_jsonl_artifact,
    write_text_artifact,
)
from yelp_agent.models import Prediction, RecommendationTask


def _task(task_id: str) -> RecommendationTask:
    return RecommendationTask(
        task_id=task_id,
        user_id="user-1",
        cutoff_time="2020-01-01T00:00:00",
        candidate_business_ids=[f"business-{index:02d}" for index in range(20)],
    )


def test_task_reader_preserves_order_and_supports_split_and_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.jsonl"
    tasks = [_task("validation:user-2"), _task("validation:user-1")]
    path.write_text(
        "\n" + "".join(task.model_dump_json() + "\n" for task in tasks),
        encoding="utf-8",
    )

    loaded = read_recommendation_tasks(
        path,
        required_split="validation",
        limit=1,
    )

    assert loaded == tasks[:1]


def test_task_reader_rejects_wrong_split_duplicate_and_empty(
    tmp_path: Path,
) -> None:
    wrong_split = tmp_path / "wrong.jsonl"
    wrong_split.write_text(
        _task("test:user-1").model_dump_json() + "\n",
        encoding="utf-8",
    )
    duplicate = tmp_path / "duplicate.jsonl"
    task = _task("validation:user-1")
    duplicate.write_text(
        task.model_dump_json() + "\n" + task.model_dump_json() + "\n",
        encoding="utf-8",
    )
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")

    with pytest.raises(TaskSplitError, match="validation tasks only"):
        read_recommendation_tasks(wrong_split, required_split="validation")
    with pytest.raises(TaskFileError, match="Duplicate task_id"):
        read_recommendation_tasks(duplicate)
    with pytest.raises(TaskFileError, match="contains no tasks"):
        read_recommendation_tasks(empty)


def test_artifact_writers_are_atomic_byte_stable_and_typed(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "nested" / "report.md"
    first = write_text_artifact(text_path, "report\n")
    first_mtime = text_path.stat().st_mtime_ns
    second = write_text_artifact(text_path, "report\n")

    task = _task("validation:user-1")
    json_path = tmp_path / "task.json"
    jsonl_path = tmp_path / "predictions.jsonl"
    prediction = Prediction(
        task_id=task.task_id,
        ranking=task.candidate_business_ids,
        latency_ms=0,
        fallback=False,
    )
    write_json_artifact(json_path, task)
    write_jsonl_artifact(jsonl_path, [prediction])

    assert first.status == "written"
    assert second.status == "skipped"
    assert text_path.stat().st_mtime_ns == first_mtime
    assert first.sha256 == second.sha256
    assert RecommendationTask.model_validate_json(
        json_path.read_text(encoding="utf-8")
    ) == task
    assert Prediction.model_validate_json(
        jsonl_path.read_text(encoding="utf-8").strip()
    ) == prediction
    assert not list(tmp_path.rglob("*.partial"))
