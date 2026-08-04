from __future__ import annotations

from pathlib import Path

import pytest

from yelp_agent.agent.llm import LLMAttemptTrace
from yelp_agent.evaluation.agent_runtime import AgentRuntimeMetrics
from yelp_agent.models import Prediction, RecommendationTask
from yelp_agent.rankers.agent_ranker import AgentTrace
from yelp_agent.rankers.agent_runner import (
    AgentFailure,
    AgentRunError,
    run_agent_ranker,
)


class FixedTraceableRanker:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._traces: dict[str, AgentTrace] = {}

    def rank(self, task: RecommendationTask) -> Prediction:
        self.calls.append(task.task_id)
        fallback = task.user_id == "user-2"
        reason = "timeout" if fallback else None
        trace = AgentTrace(
            task_id=task.task_id,
            profile=None,
            representative_review_ids=[],
            hybrid_ranking=task.candidate_business_ids,
            hybrid_score_breakdowns={},
            top_k_before=task.candidate_business_ids[:8],
            top_k_after=task.candidate_business_ids[:8],
            llm_status="failure" if fallback else "success",
            model="deepseek-v4-flash",
            attempt_count=3 if fallback else 1,
            input_tokens=None if fallback else 100,
            output_tokens=None if fallback else 20,
            total_tokens=None if fallback else 120,
            observed_total_tokens=None if fallback else 120,
            unknown_usage_attempts=3 if fallback else 0,
            llm_latency_ms=30.0 if fallback else 10.0,
            llm_attempts=(
                [
                    LLMAttemptTrace(
                        attempt_index=index,
                        status="failure",
                        latency_ms=10.0,
                        failure_reason="timeout",
                        retryable=True,
                        usage_unknown=True,
                    )
                    for index in range(1, 4)
                ]
                if fallback
                else [
                    LLMAttemptTrace(
                        attempt_index=1,
                        status="success",
                        latency_ms=10.0,
                        retryable=False,
                        input_tokens=100,
                        output_tokens=20,
                        total_tokens=120,
                        usage_unknown=False,
                    )
                ]
            ),
            tool_calls=4,
            llm_reason=None if fallback else "valid",
            fallback=fallback,
            fallback_reason=reason,
            latency_ms=10.0,
        )
        self._traces[task.task_id] = trace
        return Prediction(
            task_id=task.task_id,
            ranking=task.candidate_business_ids,
            latency_ms=10.0,
            fallback=fallback,
            fallback_reason=reason,
            tool_calls=4,
            llm_tokens=None if fallback else 120,
            metadata={
                "method": "hybrid_agent",
                "llm_attempted": True,
            },
        )

    def trace_for(self, task_id: str) -> AgentTrace:
        return self._traces[task_id]


class FailsOnSecondTaskRanker(FixedTraceableRanker):
    def rank(self, task: RecommendationTask) -> Prediction:
        if task.user_id == "user-2":
            raise RuntimeError("batch interruption")
        return super().rank(task)


def _write_tasks(path: Path) -> list[RecommendationTask]:
    candidates = [f"business-{index:02d}" for index in range(20)]
    tasks = [
        RecommendationTask(
            task_id=f"test:user-{index}",
            user_id=f"user-{index}",
            cutoff_time="2020-02-01T00:00:00",
            candidate_business_ids=candidates,
        )
        for index in (1, 2)
    ]
    path.write_text(
        "".join(task.model_dump_json() + "\n" for task in tasks),
        encoding="utf-8",
    )
    return tasks


def test_agent_runner_writes_predictions_traces_and_failure_subset(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    tasks = _write_tasks(tasks_path)
    output_dir = tmp_path / "agent-output"
    ranker = FixedTraceableRanker()

    result = run_agent_ranker(tasks_path, ranker, output_dir)

    assert result.status == "written"
    assert result.task_count == 2
    assert result.failure_count == 1
    assert Path(result.runtime_metrics_path) == output_dir / "runtime_metrics.json"
    predictions = [
        Prediction.model_validate_json(line)
        for line in (output_dir / "predictions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    traces = [
        AgentTrace.model_validate_json(line)
        for line in (output_dir / "traces.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    failures = [
        AgentFailure.model_validate_json(line)
        for line in (output_dir / "failures.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    runtime_metrics = AgentRuntimeMetrics.model_validate_json(
        (output_dir / "runtime_metrics.json").read_text(encoding="utf-8")
    )
    assert [prediction.task_id for prediction in predictions] == [
        task.task_id for task in tasks
    ]
    assert [trace.task_id for trace in traces] == [task.task_id for task in tasks]
    assert len(failures) == 1
    assert failures[0].task_id == "test:user-2"
    assert failures[0].fallback_reason == "timeout"
    assert failures[0].llm_latency_ms == 30.0
    assert failures[0].timeout_attempts == 3
    assert failures[0].unknown_usage_attempts == 3
    assert runtime_metrics.task_count == 2
    assert runtime_metrics.llm_response_success_count == 1
    assert runtime_metrics.fallback_count == 1
    assert runtime_metrics.timeout_attempt_count == 3
    assert runtime_metrics.observed_total_tokens == 120
    assert ranker.calls == [task.task_id for task in tasks]


def test_agent_runner_reuses_only_a_complete_consistent_artifact_set(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    output_dir = tmp_path / "agent-output"
    run_agent_ranker(tasks_path, FixedTraceableRanker(), output_dir)
    mtimes = {
        name: (output_dir / name).stat().st_mtime_ns
        for name in (
            "predictions.jsonl",
            "traces.jsonl",
            "failures.jsonl",
            "runtime_metrics.json",
        )
    }
    unused_ranker = FixedTraceableRanker()

    result = run_agent_ranker(tasks_path, unused_ranker, output_dir)

    assert result.status == "skipped"
    assert result.task_count == 2
    assert result.failure_count == 1
    assert unused_ranker.calls == []
    assert {
        name: (output_dir / name).stat().st_mtime_ns
        for name in mtimes
    } == mtimes


def test_agent_runner_removes_all_partial_files_after_interruption(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    output_dir = tmp_path / "agent-output"

    with pytest.raises(AgentRunError, match="Agent failed"):
        run_agent_ranker(tasks_path, FailsOnSecondTaskRanker(), output_dir)

    for name in (
        "predictions.jsonl",
        "traces.jsonl",
        "failures.jsonl",
        "runtime_metrics.json",
    ):
        assert not (output_dir / name).exists()
        assert not (output_dir / f"{name}.partial").exists()
