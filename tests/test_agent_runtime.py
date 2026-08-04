from __future__ import annotations

from pathlib import Path

import pytest

from yelp_agent.agent.llm import LLMAttemptTrace
from yelp_agent.evaluation.agent_runtime import (
    AgentRuntimeMetrics,
    AgentRuntimeMetricsError,
    load_agent_traces,
    summarize_agent_trace_file,
    summarize_agent_traces,
    write_agent_runtime_metrics,
)
from yelp_agent.rankers.agent_ranker import (
    AgentTimingBreakdown,
    AgentTrace,
)


def _base_trace(**updates: object) -> AgentTrace:
    trace = AgentTrace(
        task_id="test:user-1",
        profile=None,
        representative_review_ids=[],
        hybrid_ranking=[f"business-{index}" for index in range(20)],
        hybrid_score_breakdowns={},
        top_k_before=[f"business-{index}" for index in range(8)],
        top_k_after=[f"business-{index}" for index in range(8)],
        llm_status="success",
        model="deepseek-v4-flash",
        attempt_count=1,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        observed_total_tokens=150,
        unknown_usage_attempts=0,
        llm_latency_ms=40.0,
        llm_attempts=[
            LLMAttemptTrace(
                attempt_index=1,
                status="success",
                latency_ms=40.0,
                retryable=False,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                usage_unknown=False,
            )
        ],
        tool_calls=4,
        llm_reason="valid",
        fallback=False,
        fallback_reason=None,
        latency_ms=50.0,
        timing=AgentTimingBreakdown(
            hybrid_ms=1.0,
            tools_ms=2.0,
            prompt_ms=3.0,
            llm_ms=40.0,
            parse_merge_ms=4.0,
            total_ms=50.0,
        ),
    )
    return trace.model_copy(update=updates)


def _three_runtime_traces() -> list[AgentTrace]:
    success = _base_trace(
        top_k_after=list(reversed([f"business-{index}" for index in range(8)]))
    )
    timeout = _base_trace(
        task_id="test:user-2",
        llm_status="failure",
        attempt_count=2,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        observed_total_tokens=None,
        unknown_usage_attempts=2,
        llm_latency_ms=180.0,
        llm_attempts=[
            LLMAttemptTrace(
                attempt_index=index,
                status="failure",
                latency_ms=90.0,
                failure_reason="timeout",
                retryable=True,
                usage_unknown=True,
            )
            for index in (1, 2)
        ],
        llm_reason=None,
        fallback=True,
        fallback_reason="timeout",
        latency_ms=190.0,
        timing=AgentTimingBreakdown(
            hybrid_ms=1.0,
            tools_ms=2.0,
            prompt_ms=3.0,
            llm_ms=180.0,
            parse_merge_ms=0.0,
            total_ms=190.0,
        ),
    )
    disabled = _base_trace(
        task_id="test:user-3",
        llm_status="disabled",
        attempt_count=0,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        observed_total_tokens=None,
        llm_latency_ms=0.0,
        llm_attempts=[],
        llm_reason=None,
        fallback=True,
        fallback_reason="llm_disabled",
        latency_ms=5.0,
        timing=None,
    )
    return [success, timeout, disabled]


def test_summarizes_fake_provider_and_agent_runtime_separately() -> None:
    metrics = summarize_agent_traces(_three_runtime_traces())

    assert metrics.task_count == 3
    assert metrics.llm_attempted_count == 2
    assert metrics.llm_response_success_count == 1
    assert metrics.llm_response_success_rate == pytest.approx(0.5)
    assert metrics.agent_rerank_success_count == 1
    assert metrics.agent_rerank_success_rate == pytest.approx(1 / 3)
    assert metrics.fallback_count == 2
    assert metrics.fallback_rate == pytest.approx(2 / 3)
    assert metrics.total_attempt_count == 3
    assert metrics.mean_attempt_count == pytest.approx(1.5)
    assert metrics.timeout_attempt_count == 2
    assert metrics.unknown_usage_attempts == 2
    assert metrics.attempt_failure_reason_counts == {"timeout": 2}
    assert metrics.fallback_reason_counts == {
        "llm_disabled": 1,
        "timeout": 1,
    }
    assert metrics.model_counts == {"deepseek-v4-flash": 3}
    assert metrics.mean_agent_latency_ms == pytest.approx(245 / 3)
    assert metrics.p95_agent_latency_ms == pytest.approx(176.0)
    assert metrics.mean_llm_latency_ms == pytest.approx(110.0)
    assert metrics.p95_llm_latency_ms == pytest.approx(173.0)
    assert metrics.mean_tool_calls == pytest.approx(4.0)
    assert metrics.observed_input_tokens == 100
    assert metrics.observed_output_tokens == 50
    assert metrics.observed_total_tokens == 150
    assert metrics.observed_token_task_count == 1
    assert metrics.mean_observed_total_tokens == pytest.approx(150.0)
    assert metrics.p95_observed_total_tokens == pytest.approx(150.0)
    assert metrics.top_k_changed_count == 1
    assert metrics.top_k_changed_rate == pytest.approx(1 / 3)
    assert metrics.timing_coverage_count == 2
    assert metrics.mean_hybrid_ms == pytest.approx(1.0)
    assert metrics.mean_tools_ms == pytest.approx(2.0)
    assert metrics.mean_prompt_ms == pytest.approx(3.0)
    assert metrics.mean_timed_llm_ms == pytest.approx(110.0)
    assert metrics.mean_parse_merge_ms == pytest.approx(2.0)


def test_trace_file_round_trip_and_runtime_report_are_deterministic(
    tmp_path: Path,
) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "".join(
            trace.model_dump_json() + "\n"
            for trace in _three_runtime_traces()
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "runtime_metrics.json"

    metrics = summarize_agent_trace_file(traces_path)
    write_agent_runtime_metrics(metrics, output_path)
    first_mtime = output_path.stat().st_mtime_ns
    write_agent_runtime_metrics(metrics, output_path)

    restored = AgentRuntimeMetrics.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert restored == metrics
    assert output_path.stat().st_mtime_ns == first_mtime
    assert not (tmp_path / "runtime_metrics.json.partial").exists()
    assert load_agent_traces(traces_path) == _three_runtime_traces()


def test_rejects_duplicate_tasks_and_inconsistent_attempt_indexes() -> None:
    trace = _base_trace()
    with pytest.raises(AgentRuntimeMetricsError, match="duplicate"):
        summarize_agent_traces([trace, trace])

    inconsistent = trace.model_copy(update={"attempt_count": 2})
    with pytest.raises(AgentRuntimeMetricsError, match="attempt count"):
        summarize_agent_traces([inconsistent])
