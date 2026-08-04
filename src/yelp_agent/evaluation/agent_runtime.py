"""Aggregate operational metrics from sanitized Agent traces."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from pydantic import Field, ValidationError

from yelp_agent.models import StrictModel
from yelp_agent.rankers.agent_ranker import AgentTrace


class AgentRuntimeMetricsError(RuntimeError):
    """Raised when an Agent trace artifact cannot be summarized safely."""


class AgentRuntimeMetrics(StrictModel):
    task_count: int = Field(gt=0)
    llm_attempted_count: int = Field(ge=0)
    llm_response_success_count: int = Field(ge=0)
    llm_response_success_rate: float = Field(ge=0, le=1)
    agent_rerank_success_count: int = Field(ge=0)
    agent_rerank_success_rate: float = Field(ge=0, le=1)
    fallback_count: int = Field(ge=0)
    fallback_rate: float = Field(ge=0, le=1)
    total_attempt_count: int = Field(ge=0)
    mean_attempt_count: float = Field(ge=0)
    timeout_attempt_count: int = Field(ge=0)
    unknown_usage_attempts: int = Field(ge=0)
    attempt_failure_reason_counts: dict[str, int]
    fallback_reason_counts: dict[str, int]
    model_counts: dict[str, int]
    mean_agent_latency_ms: float = Field(ge=0)
    p95_agent_latency_ms: float = Field(ge=0)
    mean_llm_latency_ms: float | None = Field(default=None, ge=0)
    p95_llm_latency_ms: float | None = Field(default=None, ge=0)
    mean_tool_calls: float = Field(ge=0)
    observed_input_tokens: int = Field(ge=0)
    observed_output_tokens: int = Field(ge=0)
    observed_total_tokens: int = Field(ge=0)
    observed_token_task_count: int = Field(ge=0)
    mean_observed_total_tokens: float | None = Field(default=None, ge=0)
    p95_observed_total_tokens: float | None = Field(default=None, ge=0)
    top_k_changed_count: int = Field(ge=0)
    top_k_changed_rate: float = Field(ge=0, le=1)
    timing_coverage_count: int = Field(ge=0)
    mean_hybrid_ms: float | None = Field(default=None, ge=0)
    mean_tools_ms: float | None = Field(default=None, ge=0)
    mean_prompt_ms: float | None = Field(default=None, ge=0)
    mean_timed_llm_ms: float | None = Field(default=None, ge=0)
    mean_parse_merge_ms: float | None = Field(default=None, ge=0)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _p95(values: Sequence[float]) -> float | None:
    return None if not values else float(np.percentile(values, 95))


def _sorted_counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _validate_attempts(trace: AgentTrace) -> None:
    if not trace.llm_attempts:
        return
    if len(trace.llm_attempts) != trace.attempt_count:
        raise AgentRuntimeMetricsError(
            f"attempt count mismatch for task {trace.task_id!r}"
        )
    expected_indexes = list(range(1, trace.attempt_count + 1))
    actual_indexes = [
        attempt.attempt_index for attempt in trace.llm_attempts
    ]
    if actual_indexes != expected_indexes:
        raise AgentRuntimeMetricsError(
            f"attempt indexes are not consecutive for task {trace.task_id!r}"
        )


def summarize_agent_traces(
    traces: Sequence[AgentTrace],
) -> AgentRuntimeMetrics:
    """Summarize provider, fallback, latency, token, and timing behavior."""

    if not traces:
        raise AgentRuntimeMetricsError("Agent trace collection is empty")
    seen: set[str] = set()
    for trace in traces:
        if trace.task_id in seen:
            raise AgentRuntimeMetricsError(
                f"duplicate Agent trace task_id: {trace.task_id!r}"
            )
        seen.add(trace.task_id)
        _validate_attempts(trace)

    task_count = len(traces)
    llm_attempted = [
        trace
        for trace in traces
        if trace.llm_status in {"success", "failure"}
        and trace.attempt_count > 0
    ]
    llm_success_count = sum(
        trace.llm_status == "success" for trace in llm_attempted
    )
    rerank_success_count = sum(not trace.fallback for trace in traces)
    fallback_count = sum(trace.fallback for trace in traces)
    total_attempt_count = sum(trace.attempt_count for trace in llm_attempted)

    attempt_failure_reasons = [
        attempt.failure_reason
        for trace in traces
        for attempt in trace.llm_attempts
        if attempt.failure_reason is not None
    ]
    fallback_reasons = [
        trace.fallback_reason
        for trace in traces
        if trace.fallback_reason is not None
    ]
    models = [trace.model for trace in traces if trace.model is not None]

    agent_latencies = [trace.latency_ms for trace in traces]
    llm_latencies = [
        trace.llm_latency_ms
        for trace in llm_attempted
        if trace.llm_latency_ms is not None
    ]

    observed_input_tokens = 0
    observed_output_tokens = 0
    observed_total_tokens = 0
    per_task_total_tokens: list[float] = []
    for trace in traces:
        if trace.llm_attempts:
            known_input = [
                attempt.input_tokens
                for attempt in trace.llm_attempts
                if attempt.input_tokens is not None
            ]
            known_output = [
                attempt.output_tokens
                for attempt in trace.llm_attempts
                if attempt.output_tokens is not None
            ]
            known_total = [
                attempt.total_tokens
                for attempt in trace.llm_attempts
                if attempt.total_tokens is not None
            ]
            observed_input_tokens += sum(known_input)
            observed_output_tokens += sum(known_output)
            if known_total:
                task_total = sum(known_total)
                observed_total_tokens += task_total
                per_task_total_tokens.append(float(task_total))
        elif trace.total_tokens is not None:
            # Backward compatibility for traces written before attempt logging.
            observed_input_tokens += trace.input_tokens or 0
            observed_output_tokens += trace.output_tokens or 0
            observed_total_tokens += trace.total_tokens
            per_task_total_tokens.append(float(trace.total_tokens))

    changed_count = sum(
        trace.top_k_before != trace.top_k_after for trace in traces
    )
    timed = [trace.timing for trace in traces if trace.timing is not None]
    mean_agent_latency = _mean(agent_latencies)
    p95_agent_latency = _p95(agent_latencies)
    if mean_agent_latency is None or p95_agent_latency is None:
        raise AssertionError("non-empty traces must have Agent latencies")

    return AgentRuntimeMetrics(
        task_count=task_count,
        llm_attempted_count=len(llm_attempted),
        llm_response_success_count=llm_success_count,
        llm_response_success_rate=(
            0.0 if not llm_attempted else llm_success_count / len(llm_attempted)
        ),
        agent_rerank_success_count=rerank_success_count,
        agent_rerank_success_rate=rerank_success_count / task_count,
        fallback_count=fallback_count,
        fallback_rate=fallback_count / task_count,
        total_attempt_count=total_attempt_count,
        mean_attempt_count=(
            0.0 if not llm_attempted else total_attempt_count / len(llm_attempted)
        ),
        timeout_attempt_count=sum(
            reason == "timeout" for reason in attempt_failure_reasons
        ),
        unknown_usage_attempts=sum(
            trace.unknown_usage_attempts for trace in traces
        ),
        attempt_failure_reason_counts=_sorted_counts(
            attempt_failure_reasons
        ),
        fallback_reason_counts=_sorted_counts(fallback_reasons),
        model_counts=_sorted_counts(models),
        mean_agent_latency_ms=mean_agent_latency,
        p95_agent_latency_ms=p95_agent_latency,
        mean_llm_latency_ms=_mean(llm_latencies),
        p95_llm_latency_ms=_p95(llm_latencies),
        mean_tool_calls=float(np.mean([trace.tool_calls for trace in traces])),
        observed_input_tokens=observed_input_tokens,
        observed_output_tokens=observed_output_tokens,
        observed_total_tokens=observed_total_tokens,
        observed_token_task_count=len(per_task_total_tokens),
        mean_observed_total_tokens=_mean(per_task_total_tokens),
        p95_observed_total_tokens=_p95(per_task_total_tokens),
        top_k_changed_count=changed_count,
        top_k_changed_rate=changed_count / task_count,
        timing_coverage_count=len(timed),
        mean_hybrid_ms=_mean([value.hybrid_ms for value in timed]),
        mean_tools_ms=_mean([value.tools_ms for value in timed]),
        mean_prompt_ms=_mean([value.prompt_ms for value in timed]),
        mean_timed_llm_ms=_mean([value.llm_ms for value in timed]),
        mean_parse_merge_ms=_mean(
            [value.parse_merge_ms for value in timed]
        ),
    )


def load_agent_traces(path: str | Path) -> list[AgentTrace]:
    """Load one JSONL trace artifact with duplicate and schema checks."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Agent trace JSONL does not exist: {source}")
    traces: list[AgentTrace] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    traces.append(AgentTrace.model_validate_json(line))
                except ValidationError as exc:
                    raise AgentRuntimeMetricsError(
                        f"invalid Agent trace at line {line_number}"
                    ) from exc
    except AgentRuntimeMetricsError:
        raise
    except OSError as exc:
        raise AgentRuntimeMetricsError(
            f"could not read Agent trace JSONL: {source}"
        ) from exc
    return traces


def summarize_agent_trace_file(path: str | Path) -> AgentRuntimeMetrics:
    return summarize_agent_traces(load_agent_traces(path))


def write_agent_runtime_metrics(
    metrics: AgentRuntimeMetrics,
    path: str | Path,
) -> None:
    """Atomically write a deterministic JSON runtime report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = metrics.model_dump_json(indent=2) + "\n"
    if destination.is_file():
        try:
            if destination.read_text(encoding="utf-8") == rendered:
                return
        except OSError:
            pass
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_text(
            rendered,
            encoding="utf-8",
        )
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
