"""Atomic batch execution for Agent predictions, traces, and runtime metrics."""

from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field

from yelp_agent.evaluation.agent_runtime import (
    summarize_agent_trace_file,
    write_agent_runtime_metrics,
)
from yelp_agent.models import Prediction, RecommendationTask, StrictModel
from yelp_agent.rankers.agent_ranker import AgentTrace


class AgentRunError(RuntimeError):
    """Raised when Agent batch inputs or outputs violate the contract."""


class AgentFailure(StrictModel):
    task_id: str = Field(min_length=1)
    fallback_reason: str = Field(min_length=1)
    llm_status: Literal["success", "disabled", "failure", "not_called"]
    model: str | None = None
    attempt_count: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    llm_attempted: bool
    llm_latency_ms: float | None = Field(default=None, ge=0)
    timeout_attempts: int = Field(default=0, ge=0)
    unknown_usage_attempts: int = Field(default=0, ge=0)


class AgentRunResult(StrictModel):
    status: Literal["written", "skipped"]
    tasks_path: str
    output_dir: str
    predictions_path: str
    traces_path: str
    failures_path: str
    runtime_metrics_path: str
    task_count: int = Field(gt=0)
    failure_count: int = Field(ge=0)


class _TraceableRanker(Protocol):
    def rank(self, task: RecommendationTask) -> Prediction: ...

    def trace_for(self, task_id: str) -> AgentTrace: ...


def _read_tasks(path: Path, limit: int | None) -> list[RecommendationTask]:
    if not path.is_file():
        raise AgentRunError(f"task file does not exist: {path}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    tasks: list[RecommendationTask] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                task = RecommendationTask.model_validate_json(line)
                if task.task_id in seen:
                    raise AgentRunError(
                        f"duplicate task_id at line {line_number}: {task.task_id}"
                    )
                seen.add(task.task_id)
                tasks.append(task)
                if limit is not None and len(tasks) == limit:
                    break
    except AgentRunError:
        raise
    except Exception as exc:
        raise AgentRunError(f"invalid task file {path}: {exc}") from exc
    if not tasks:
        raise AgentRunError(f"task file contains no tasks: {path}")
    return tasks


def _validate_task_output(
    task: RecommendationTask,
    prediction: Prediction,
    trace: AgentTrace,
) -> None:
    if prediction.task_id != task.task_id or trace.task_id != task.task_id:
        raise AgentRunError(
            f"Agent output task_id does not match {task.task_id!r}"
        )
    if set(prediction.ranking) != set(task.candidate_business_ids):
        raise AgentRunError(
            f"Agent ranking is not a candidate permutation for {task.task_id!r}"
        )
    if prediction.fallback != trace.fallback:
        raise AgentRunError(
            f"Prediction and trace fallback disagree for {task.task_id!r}"
        )
    if prediction.fallback_reason != trace.fallback_reason:
        raise AgentRunError(
            f"Prediction and trace fallback reason disagree for {task.task_id!r}"
        )


def _failure_from(
    prediction: Prediction,
    trace: AgentTrace,
) -> AgentFailure:
    if not prediction.fallback or prediction.fallback_reason is None:
        raise AgentRunError("cannot create a failure row from a successful prediction")
    return AgentFailure(
        task_id=prediction.task_id,
        fallback_reason=prediction.fallback_reason,
        llm_status=trace.llm_status,
        model=trace.model,
        attempt_count=trace.attempt_count,
        tool_calls=trace.tool_calls,
        total_tokens=trace.total_tokens,
        llm_attempted=prediction.metadata.get("llm_attempted") is True,
        llm_latency_ms=trace.llm_latency_ms,
        timeout_attempts=sum(
            attempt.failure_reason == "timeout"
            for attempt in trace.llm_attempts
        ),
        unknown_usage_attempts=trace.unknown_usage_attempts,
    )


def _read_models(path: Path, model_type: type[StrictModel]) -> list[StrictModel]:
    values: list[StrictModel] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    values.append(model_type.model_validate_json(line))
    except Exception as exc:
        raise AgentRunError(f"invalid existing Agent artifact: {path}") from exc
    return values


def _validate_existing(
    tasks: list[RecommendationTask],
    paths: dict[str, Path],
) -> int:
    predictions = [
        value
        for value in _read_models(paths["predictions"], Prediction)
        if isinstance(value, Prediction)
    ]
    traces = [
        value
        for value in _read_models(paths["traces"], AgentTrace)
        if isinstance(value, AgentTrace)
    ]
    failures = [
        value
        for value in _read_models(paths["failures"], AgentFailure)
        if isinstance(value, AgentFailure)
    ]
    if len(predictions) != len(tasks) or len(traces) != len(tasks):
        raise AgentRunError(
            "existing Agent predictions/traces do not match task count"
        )
    expected_failures: list[AgentFailure] = []
    for task, prediction, trace in zip(
        tasks,
        predictions,
        traces,
        strict=True,
    ):
        _validate_task_output(task, prediction, trace)
        if prediction.fallback:
            expected_failures.append(_failure_from(prediction, trace))
    if failures != expected_failures:
        raise AgentRunError(
            "existing Agent failures do not match the fallback subset"
        )
    return len(failures)


def run_agent_ranker(
    tasks_path: str | Path,
    ranker: _TraceableRanker,
    output_dir: str | Path,
    *,
    force: bool = False,
    limit: int | None = None,
) -> AgentRunResult:
    """Run Agent tasks and publish consistent JSONL plus runtime metrics."""

    resolved_tasks = Path(tasks_path)
    resolved_output = Path(output_dir)
    tasks = _read_tasks(resolved_tasks, limit)
    paths = {
        "predictions": resolved_output / "predictions.jsonl",
        "traces": resolved_output / "traces.jsonl",
        "failures": resolved_output / "failures.jsonl",
        "runtime_metrics": resolved_output / "runtime_metrics.json",
    }
    jsonl_names = ("predictions", "traces", "failures")
    existing = [paths[name] for name in jsonl_names if paths[name].exists()]
    if existing and not force:
        if len(existing) != len(jsonl_names):
            raise AgentRunError(
                "Agent output is incomplete; use force=True to rebuild"
            )
        failure_count = _validate_existing(tasks, paths)
        runtime_metrics = summarize_agent_trace_file(paths["traces"])
        write_agent_runtime_metrics(
            runtime_metrics,
            paths["runtime_metrics"],
        )
        return AgentRunResult(
            status="skipped",
            tasks_path=str(resolved_tasks),
            output_dir=str(resolved_output),
            predictions_path=str(paths["predictions"]),
            traces_path=str(paths["traces"]),
            failures_path=str(paths["failures"]),
            runtime_metrics_path=str(paths["runtime_metrics"]),
            task_count=len(tasks),
            failure_count=failure_count,
        )

    resolved_output.mkdir(parents=True, exist_ok=True)
    partials = {
        name: path.with_name(path.name + ".partial")
        for name, path in paths.items()
    }
    for partial in partials.values():
        partial.unlink(missing_ok=True)

    failure_count = 0
    try:
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context(
                    partial.open("w", encoding="utf-8", newline="\n")
                )
                for name, partial in partials.items()
                if name in jsonl_names
            }
            for task in tasks:
                try:
                    prediction = ranker.rank(task)
                    trace = ranker.trace_for(task.task_id)
                except Exception as exc:
                    raise AgentRunError(
                        f"Agent failed for task {task.task_id!r}"
                    ) from exc
                if not isinstance(prediction, Prediction):
                    raise AgentRunError(
                        f"Agent returned {type(prediction).__name__} instead of Prediction"
                    )
                if not isinstance(trace, AgentTrace):
                    raise AgentRunError(
                        f"Agent returned {type(trace).__name__} instead of AgentTrace"
                    )
                _validate_task_output(task, prediction, trace)
                handles["predictions"].write(prediction.model_dump_json() + "\n")
                handles["traces"].write(trace.model_dump_json() + "\n")
                if prediction.fallback:
                    handles["failures"].write(
                        _failure_from(prediction, trace).model_dump_json() + "\n"
                    )
                    failure_count += 1
        runtime_metrics = summarize_agent_trace_file(partials["traces"])
        partials["runtime_metrics"].write_text(
            runtime_metrics.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        for name in (*jsonl_names, "runtime_metrics"):
            os.replace(partials[name], paths[name])
    except Exception:
        for partial in partials.values():
            partial.unlink(missing_ok=True)
        raise

    return AgentRunResult(
        status="written",
        tasks_path=str(resolved_tasks),
        output_dir=str(resolved_output),
        predictions_path=str(paths["predictions"]),
        traces_path=str(paths["traces"]),
        failures_path=str(paths["failures"]),
        runtime_metrics_path=str(paths["runtime_metrics"]),
        task_count=len(tasks),
        failure_count=failure_count,
    )
