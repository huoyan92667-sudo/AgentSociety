"""Shared execution and validation for all recommendation rankers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from yelp_agent.experiments import (
    TaskFileError,
    read_recommendation_tasks,
    write_jsonl_artifact,
)
from yelp_agent.models import Prediction, RecommendationTask, StrictModel
from yelp_agent.protocols import Ranker


class RankerRunError(RuntimeError):
    """Raised when tasks or ranker outputs violate the public contract."""


class RankerRunResult(StrictModel):
    status: Literal["written", "skipped"]
    tasks_path: str
    predictions_path: str
    task_count: int = Field(ge=0)


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
    try:
        tasks = read_recommendation_tasks(resolved_tasks_path)
    except (FileNotFoundError, TaskFileError) as exc:
        raise RankerRunError(str(exc)) from exc

    if resolved_predictions_path.is_file() and not force:
        _validate_existing_predictions(resolved_predictions_path, tasks)
        return RankerRunResult(
            status="skipped",
            tasks_path=str(resolved_tasks_path),
            predictions_path=str(resolved_predictions_path),
            task_count=len(tasks),
        )

    predictions: list[Prediction] = []
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
        predictions.append(prediction)
    write_jsonl_artifact(resolved_predictions_path, predictions)

    return RankerRunResult(
        status="written",
        tasks_path=str(resolved_tasks_path),
        predictions_path=str(resolved_predictions_path),
        task_count=len(tasks),
    )
