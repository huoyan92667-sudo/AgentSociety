"""Shared execution and validation for all recommendation rankers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from yelp_agent.models import Prediction, RecommendationTask, StrictModel
from yelp_agent.protocols import Ranker


class RankerRunError(RuntimeError):
    """Raised when tasks or ranker outputs violate the public contract."""


class RankerRunResult(StrictModel):
    status: Literal["written", "skipped"]
    tasks_path: str
    predictions_path: str
    task_count: int = Field(ge=0)


def _read_tasks(tasks_path: Path) -> list[RecommendationTask]:
    if not tasks_path.is_file():
        raise RankerRunError(f"task file does not exist: {tasks_path}")

    tasks: list[RecommendationTask] = []
    seen_task_ids: set[str] = set()
    try:
        with tasks_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                task = RecommendationTask.model_validate_json(line)
                if task.task_id in seen_task_ids:
                    raise RankerRunError(
                        f"duplicate task_id at line {line_number}: {task.task_id}"
                    )
                seen_task_ids.add(task.task_id)
                tasks.append(task)
    except RankerRunError:
        raise
    except Exception as exc:
        raise RankerRunError(f"invalid task file {tasks_path}: {exc}") from exc

    if not tasks:
        raise RankerRunError(f"task file contains no tasks: {tasks_path}")
    return tasks


def _validate_prediction(
    task: RecommendationTask,
    prediction: Prediction,
) -> None:
    if prediction.task_id != task.task_id:
        raise RankerRunError(
            f"ranker returned task_id {prediction.task_id!r} "
            f"for task {task.task_id!r}"
        )
    if set(prediction.ranking) != set(task.candidate_business_ids):
        raise RankerRunError(
            f"ranking for task {task.task_id!r} is not a complete "
            "permutation of its candidates"
        )


def _validate_existing_predictions(
    predictions_path: Path,
    tasks: list[RecommendationTask],
) -> None:
    predictions: list[Prediction] = []
    try:
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    predictions.append(Prediction.model_validate_json(line))
    except Exception as exc:
        raise RankerRunError(
            f"invalid existing prediction file {predictions_path}: {exc}"
        ) from exc

    if len(predictions) != len(tasks):
        raise RankerRunError(
            f"existing prediction count {len(predictions)} does not match "
            f"task count {len(tasks)}"
        )
    for task, prediction in zip(tasks, predictions, strict=True):
        _validate_prediction(task, prediction)


def run_ranker(
    tasks_path: str | Path,
    ranker: Ranker,
    predictions_path: str | Path,
    *,
    force: bool = False,
) -> RankerRunResult:
    """Run a ranker over frozen tasks or safely reuse a complete output file."""

    resolved_tasks_path = Path(tasks_path)
    resolved_predictions_path = Path(predictions_path)
    tasks = _read_tasks(resolved_tasks_path)

    if resolved_predictions_path.is_file() and not force:
        _validate_existing_predictions(resolved_predictions_path, tasks)
        return RankerRunResult(
            status="skipped",
            tasks_path=str(resolved_tasks_path),
            predictions_path=str(resolved_predictions_path),
            task_count=len(tasks),
        )

    resolved_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = resolved_predictions_path.with_suffix(
        resolved_predictions_path.suffix + ".partial"
    )

    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as handle:
            for task in tasks:
                try:
                    prediction = ranker.rank(task)
                except Exception as exc:
                    raise RankerRunError(
                        f"ranker failed for task {task.task_id!r}: {exc}"
                    ) from exc
                if not isinstance(prediction, Prediction):
                    raise RankerRunError(
                        f"ranker returned {type(prediction).__name__} instead "
                        f"of Prediction for task {task.task_id!r}"
                    )
                _validate_prediction(task, prediction)
                handle.write(prediction.model_dump_json() + "\n")
        os.replace(partial_path, resolved_predictions_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    return RankerRunResult(
        status="written",
        tasks_path=str(resolved_tasks_path),
        predictions_path=str(resolved_predictions_path),
        task_count=len(tasks),
    )
