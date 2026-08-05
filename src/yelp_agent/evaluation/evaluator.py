"""Evaluate prediction JSONL against isolated single-positive ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pydantic import Field, ValidationError

from yelp_agent.evaluation.metrics import compute_single_positive_metrics
from yelp_agent.experiments import (
    TaskFileError,
    read_recommendation_tasks,
    write_json_artifact,
    write_jsonl_artifact,
)
from yelp_agent.models import Prediction, RecommendationTask, StrictModel


class EvaluationDataError(RuntimeError):
    """Raised when frozen tasks or ground truth are internally inconsistent."""


class EvaluationIssue(StrictModel):
    task_id: str | None = None
    line_number: int | None = Field(default=None, ge=1)
    code: str
    message: str


class EvaluationMetrics(StrictModel):
    task_count: int = Field(ge=0)
    valid_prediction_count: int = Field(ge=0)
    missing_prediction_count: int = Field(ge=0)
    invalid_prediction_count: int = Field(ge=0)
    unexpected_prediction_count: int = Field(ge=0)
    valid_output_rate: float = Field(ge=0, le=1)
    hr_at_1: float = Field(ge=0, le=1)
    hr_at_3: float = Field(ge=0, le=1)
    hr_at_5: float = Field(ge=0, le=1)
    avg_hr: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    mean_tool_calls: float = Field(ge=0)
    mean_llm_tokens: float | None = Field(default=None, ge=0)
    fallback_count: int = Field(ge=0)
    fallback_rate: float = Field(ge=0, le=1)
    llm_attempted_count: int = Field(ge=0)
    llm_failure_count: int = Field(ge=0)
    llm_failure_rate: float = Field(ge=0, le=1)


class EvaluationReport(StrictModel):
    metrics: EvaluationMetrics
    issues: list[EvaluationIssue]


def load_evaluation_tasks(path: Path) -> dict[str, RecommendationTask]:
    """Load frozen tasks while preserving their file order."""

    try:
        tasks = read_recommendation_tasks(path)
    except TaskFileError as exc:
        raise EvaluationDataError(str(exc)) from exc
    return {task.task_id: task for task in tasks}


def load_ground_truth(
    path: Path,
    tasks: dict[str, RecommendationTask],
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth Parquet does not exist: {path}")
    try:
        rows = pq.read_table(
            path,
            columns=["task_id", "target_business_id"],
        ).to_pylist()
    except Exception as exc:
        raise EvaluationDataError(f"Could not read ground truth: {path}") from exc
    truth: dict[str, str] = {}
    for row in rows:
        task_id = row.get("task_id")
        target = row.get("target_business_id")
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(target, str)
            or not target
        ):
            raise EvaluationDataError("Ground truth contains an invalid row")
        if task_id in truth:
            raise EvaluationDataError(
                f"Duplicate task_id in ground truth: {task_id!r}"
            )
        truth[task_id] = target
    missing = sorted(set(tasks).difference(truth))
    if missing:
        raise EvaluationDataError(
            "Ground truth is missing expected task IDs "
            f"(missing={missing[:3]})"
        )
    selected_truth = {
        task_id: truth[task_id]
        for task_id in tasks
    }
    for task_id, target in selected_truth.items():
        if target not in tasks[task_id].candidate_business_ids:
            raise EvaluationDataError(
                f"Ground truth target is outside candidates for {task_id!r}"
            )
    return selected_truth


def load_predictions(
    path: Path,
    tasks: dict[str, RecommendationTask],
    *,
    known_task_ids: set[str],
) -> tuple[
    dict[str, Prediction],
    set[str],
    set[str],
    int,
    list[EvaluationIssue],
]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction JSONL does not exist: {path}")
    valid: dict[str, Prediction] = {}
    referenced_task_ids: set[str] = set()
    invalid_task_ids: set[str] = set()
    unexpected_count = 0
    issues: list[EvaluationIssue] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                unexpected_count += 1
                issues.append(
                    EvaluationIssue(
                        line_number=line_number,
                        code="invalid_json",
                        message=exc.msg,
                    )
                )
                continue
            if not isinstance(payload, dict):
                unexpected_count += 1
                issues.append(
                    EvaluationIssue(
                        line_number=line_number,
                        code="not_an_object",
                        message="Prediction line must be a JSON object",
                    )
                )
                continue
            task_id = payload.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                unexpected_count += 1
                issues.append(
                    EvaluationIssue(
                        line_number=line_number,
                        code="missing_task_id",
                        message="Prediction has no usable task_id",
                    )
                )
                continue
            if task_id not in tasks:
                if task_id in known_task_ids:
                    # A valid prediction outside a requested evaluation subset.
                    continue
                unexpected_count += 1
                issues.append(
                    EvaluationIssue(
                        task_id=task_id,
                        line_number=line_number,
                        code="unexpected_task_id",
                        message="Prediction task_id is not in the task dataset",
                    )
                )
                continue
            if task_id in referenced_task_ids:
                invalid_task_ids.add(task_id)
                valid.pop(task_id, None)
                issues.append(
                    EvaluationIssue(
                        task_id=task_id,
                        line_number=line_number,
                        code="duplicate_prediction",
                        message="Task has more than one prediction",
                    )
                )
                continue
            referenced_task_ids.add(task_id)
            try:
                prediction = Prediction.model_validate(payload)
            except ValidationError as exc:
                invalid_task_ids.add(task_id)
                issues.append(
                    EvaluationIssue(
                        task_id=task_id,
                        line_number=line_number,
                        code="invalid_prediction",
                        message=str(exc),
                    )
                )
                continue
            if set(prediction.ranking) != set(
                tasks[task_id].candidate_business_ids
            ):
                invalid_task_ids.add(task_id)
                issues.append(
                    EvaluationIssue(
                        task_id=task_id,
                        line_number=line_number,
                        code="candidate_mismatch",
                        message=(
                            "Ranking must be a complete permutation of task candidates"
                        ),
                    )
                )
                continue
            valid[task_id] = prediction

    missing_task_ids = set(tasks).difference(referenced_task_ids)
    for task_id in sorted(missing_task_ids):
        issues.append(
            EvaluationIssue(
                task_id=task_id,
                code="missing_prediction",
                message="Task has no prediction",
            )
        )
    return (
        valid,
        invalid_task_ids,
        missing_task_ids,
        unexpected_count,
        issues,
    )


def _mean(values: list[float]) -> float:
    return 0.0 if not values else float(np.mean(values))


def evaluate_prediction_file(
    tasks_path: str | Path,
    ground_truth_path: str | Path,
    predictions_path: str | Path,
    *,
    task_limit: int | None = None,
) -> EvaluationReport:
    """Evaluate predictions; missing or invalid expected tasks receive zero."""

    if task_limit is not None and task_limit <= 0:
        raise ValueError("task_limit must be positive")
    all_tasks = load_evaluation_tasks(Path(tasks_path))
    tasks = (
        all_tasks
        if task_limit is None
        else dict(list(all_tasks.items())[:task_limit])
    )
    truth = load_ground_truth(Path(ground_truth_path), tasks)
    (
        predictions,
        invalid_task_ids,
        missing_task_ids,
        unexpected_count,
        issues,
    ) = load_predictions(
        Path(predictions_path),
        tasks,
        known_task_ids=set(all_tasks),
    )

    task_metrics = [
        compute_single_positive_metrics(
            truth[task_id],
            predictions[task_id].ranking if task_id in predictions else [],
        )
        for task_id in tasks
    ]
    task_count = len(tasks)
    hr_at_1 = _mean([metrics.hit_at_1 for metrics in task_metrics])
    hr_at_3 = _mean([metrics.hit_at_3 for metrics in task_metrics])
    hr_at_5 = _mean([metrics.hit_at_5 for metrics in task_metrics])
    valid_predictions = list(predictions.values())
    latencies = [prediction.latency_ms for prediction in valid_predictions]
    llm_tokens = [
        float(prediction.llm_tokens)
        for prediction in valid_predictions
        if prediction.llm_tokens is not None
    ]
    fallback_count = sum(
        prediction.fallback for prediction in valid_predictions
    )
    llm_attempted = [
        prediction
        for prediction in valid_predictions
        if prediction.metadata.get("llm_attempted") is True
    ]
    llm_failure_count = sum(
        prediction.fallback
        and prediction.fallback_reason != "llm_disabled"
        for prediction in llm_attempted
    )
    valid_count = len(valid_predictions)
    metrics = EvaluationMetrics(
        task_count=task_count,
        valid_prediction_count=valid_count,
        missing_prediction_count=len(missing_task_ids),
        invalid_prediction_count=len(invalid_task_ids),
        unexpected_prediction_count=unexpected_count,
        valid_output_rate=valid_count / task_count,
        hr_at_1=hr_at_1,
        hr_at_3=hr_at_3,
        hr_at_5=hr_at_5,
        avg_hr=(hr_at_1 + hr_at_3 + hr_at_5) / 3.0,
        mrr=_mean([metric.reciprocal_rank for metric in task_metrics]),
        ndcg_at_5=_mean([metric.ndcg_at_5 for metric in task_metrics]),
        mean_latency_ms=_mean(latencies),
        p95_latency_ms=(
            0.0 if not latencies else float(np.percentile(latencies, 95))
        ),
        mean_tool_calls=_mean(
            [float(prediction.tool_calls) for prediction in valid_predictions]
        ),
        mean_llm_tokens=None if not llm_tokens else _mean(llm_tokens),
        fallback_count=fallback_count,
        fallback_rate=0.0 if not valid_count else fallback_count / valid_count,
        llm_attempted_count=len(llm_attempted),
        llm_failure_count=llm_failure_count,
        llm_failure_rate=(
            0.0
            if not llm_attempted
            else llm_failure_count / len(llm_attempted)
        ),
    )
    return EvaluationReport(metrics=metrics, issues=issues)


def write_evaluation_report(
    report: EvaluationReport,
    metrics_path: str | Path,
    issues_path: str | Path,
) -> None:
    """Write aggregate metrics separately from task-level validation issues."""

    write_json_artifact(metrics_path, report.metrics)
    write_jsonl_artifact(issues_path, report.issues)
