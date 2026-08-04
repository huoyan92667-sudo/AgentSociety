"""Compare target-rank movement between two prediction artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from yelp_agent.evaluation.evaluator import (
    _load_ground_truth,
    _load_predictions,
    _load_tasks,
)
from yelp_agent.models import Prediction, RecommendationTask, StrictModel


class PredictionComparisonError(RuntimeError):
    """Raised when two prediction files cannot be compared safely."""


class TaskRankChange(StrictModel):
    task_id: str = Field(min_length=1)
    baseline_rank: int = Field(ge=1)
    challenger_rank: int = Field(ge=1)
    rank_improvement: int


class PredictionComparison(StrictModel):
    task_count: int = Field(gt=0)
    improved_count: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    worsened_count: int = Field(ge=0)
    mean_baseline_rank: float = Field(ge=1)
    mean_challenger_rank: float = Field(ge=1)
    entered_top_1: int = Field(ge=0)
    left_top_1: int = Field(ge=0)
    entered_top_3: int = Field(ge=0)
    left_top_3: int = Field(ge=0)
    entered_top_5: int = Field(ge=0)
    left_top_5: int = Field(ge=0)
    largest_improvements: list[TaskRankChange]
    largest_worsenings: list[TaskRankChange]


def _load_complete_predictions(
    path: Path,
    *,
    tasks: dict[str, RecommendationTask],
    known_task_ids: set[str],
) -> dict[str, Prediction]:
    predictions, invalid, missing, unexpected, _ = _load_predictions(
        path,
        tasks,
        known_task_ids=known_task_ids,
    )
    if invalid or missing or unexpected or len(predictions) != len(tasks):
        raise PredictionComparisonError(
            f"prediction artifact is incomplete or invalid: {path}"
        )
    return predictions


def compare_prediction_files(
    tasks_path: str | Path,
    ground_truth_path: str | Path,
    baseline_predictions_path: str | Path,
    challenger_predictions_path: str | Path,
    *,
    task_limit: int | None = None,
) -> PredictionComparison:
    """Compare target positions without exposing target IDs in the report."""

    if task_limit is not None and task_limit <= 0:
        raise ValueError("task_limit must be positive")
    all_tasks = _load_tasks(Path(tasks_path))
    tasks = (
        all_tasks
        if task_limit is None
        else dict(list(all_tasks.items())[:task_limit])
    )
    truth = _load_ground_truth(Path(ground_truth_path), tasks)
    known_task_ids = set(all_tasks)
    baseline = _load_complete_predictions(
        Path(baseline_predictions_path),
        tasks=tasks,
        known_task_ids=known_task_ids,
    )
    challenger = _load_complete_predictions(
        Path(challenger_predictions_path),
        tasks=tasks,
        known_task_ids=known_task_ids,
    )

    changes: list[TaskRankChange] = []
    for task_id in tasks:
        target = truth[task_id]
        baseline_rank = baseline[task_id].ranking.index(target) + 1
        challenger_rank = challenger[task_id].ranking.index(target) + 1
        changes.append(
            TaskRankChange(
                task_id=task_id,
                baseline_rank=baseline_rank,
                challenger_rank=challenger_rank,
                rank_improvement=baseline_rank - challenger_rank,
            )
        )

    improved = [change for change in changes if change.rank_improvement > 0]
    worsened = [change for change in changes if change.rank_improvement < 0]
    unchanged_count = sum(change.rank_improvement == 0 for change in changes)
    count = len(changes)
    return PredictionComparison(
        task_count=count,
        improved_count=len(improved),
        unchanged_count=unchanged_count,
        worsened_count=len(worsened),
        mean_baseline_rank=sum(change.baseline_rank for change in changes) / count,
        mean_challenger_rank=(
            sum(change.challenger_rank for change in changes) / count
        ),
        entered_top_1=sum(
            change.baseline_rank > 1 and change.challenger_rank <= 1
            for change in changes
        ),
        left_top_1=sum(
            change.baseline_rank <= 1 and change.challenger_rank > 1
            for change in changes
        ),
        entered_top_3=sum(
            change.baseline_rank > 3 and change.challenger_rank <= 3
            for change in changes
        ),
        left_top_3=sum(
            change.baseline_rank <= 3 and change.challenger_rank > 3
            for change in changes
        ),
        entered_top_5=sum(
            change.baseline_rank > 5 and change.challenger_rank <= 5
            for change in changes
        ),
        left_top_5=sum(
            change.baseline_rank <= 5 and change.challenger_rank > 5
            for change in changes
        ),
        largest_improvements=sorted(
            improved,
            key=lambda change: (-change.rank_improvement, change.task_id),
        )[:3],
        largest_worsenings=sorted(
            worsened,
            key=lambda change: (change.rank_improvement, change.task_id),
        )[:3],
    )


def write_prediction_comparison(
    comparison: PredictionComparison,
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_text(
            comparison.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
