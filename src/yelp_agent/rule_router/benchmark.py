"""Run and evaluate the frozen Step 24 Rule Agent baseline."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from yelp_agent.agent_benchmark import (
    VisibleAgentScenario,
    load_evidence_labels,
    load_scenario_ground_truth,
    load_visible_scenarios,
)
from yelp_agent.agent_evaluation import (
    AgentEvaluationReport,
    AgentScenarioRun,
    evaluate_agent_scenario_runs,
    load_agent_scenario_runs,
    write_agent_evaluation_report,
    write_agent_scenario_runs,
)
from yelp_agent.agent_harness import AgentHarness, BenchmarkSessionDriver


BenchmarkSplit = Literal["development", "validation", "all"]


@dataclass(frozen=True, slots=True)
class RuleAgentBenchmarkResult:
    output_root: Path
    runs_path: Path
    metrics_path: Path
    failures_path: Path
    driver_path: Path
    runtime_metrics_path: Path
    summary_path: Path
    report: AgentEvaluationReport


def merge_rule_agent_benchmark_outputs(
    *,
    development_root: str | Path,
    validation_root: str | Path,
    benchmark_root: str | Path,
    output_root: str | Path,
) -> RuleAgentBenchmarkResult:
    """Combine already-run frozen splits without executing the Agent again."""

    development = Path(development_root)
    validation = Path(validation_root)
    output = Path(output_root)
    runs = (
        *load_agent_scenario_runs(development / "scenario_runs.jsonl"),
        *load_agent_scenario_runs(validation / "scenario_runs.jsonl"),
    )
    driver_rows = [
        *_load_jsonl(development / "driver_results.jsonl"),
        *_load_jsonl(validation / "driver_results.jsonl"),
    ]
    root = Path(benchmark_root)
    report = evaluate_agent_scenario_runs(
        runs,
        visible_scenarios=load_visible_scenarios(
            root / "visible" / "scenarios.jsonl"
        ),
        ground_truth=load_scenario_ground_truth(
            root / "hidden" / "ground_truth.jsonl"
        ),
        evidence_labels=load_evidence_labels(
            root / "hidden" / "evidence_labels.parquet"
        ),
    )
    return _publish_results(
        runs=runs,
        driver_rows=driver_rows,
        report=report,
        output=output,
        split="all",
    )


def run_rule_agent_benchmark(
    harness: AgentHarness,
    *,
    benchmark_root: str | Path,
    output_root: str | Path,
    split: BenchmarkSplit = "all",
    progress: Callable[[int, int, VisibleAgentScenario], None] | None = None,
    scenario_ids: set[str] | None = None,
) -> RuleAgentBenchmarkResult:
    """Run visible inputs, release hidden turns on trigger, then evaluate offline."""

    root = Path(benchmark_root)
    output = Path(output_root)
    visible = _select(
        load_visible_scenarios(root / "visible" / "scenarios.jsonl"), split
    )
    if scenario_ids is not None:
        known = {item.scenario_id for item in visible}
        unknown = scenario_ids - known
        if unknown:
            raise ValueError(
                f"requested scenario IDs are outside split {split}: {sorted(unknown)[:3]}"
            )
        visible = tuple(item for item in visible if item.scenario_id in scenario_ids)
        if not visible:
            raise ValueError("scenario ID filter selected no benchmark scenarios")
    visible_ids = {item.scenario_id for item in visible}
    truth = tuple(
        item
        for item in load_scenario_ground_truth(
            root / "hidden" / "ground_truth.jsonl"
        )
        if item.scenario_id in visible_ids
    )
    evidence = tuple(
        item
        for item in load_evidence_labels(
            root / "hidden" / "evidence_labels.parquet"
        )
        if item.scenario_id in visible_ids
    )
    truth_by_id = {item.scenario_id: item for item in truth}
    if len(visible) != len(truth) or set(truth_by_id) != visible_ids:
        raise ValueError("visible and hidden benchmark scenarios do not align")

    ordered = tuple(sorted(visible, key=_execution_order))
    driver = BenchmarkSessionDriver()
    runs: list[AgentScenarioRun] = []
    driver_rows: list[dict[str, object]] = []
    for index, scenario in enumerate(ordered, start=1):
        driven = driver.drive(
            harness,
            scenario,
            truth_by_id[scenario.scenario_id].scripted_user_turns,
        )
        run = driven.result.run or _snapshot_awaiting_run(driven.result.session)
        runs.append(run)
        driver_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "split": scenario.split,
                "stop_reason": driven.stop_reason,
                "released_turn_indices": driven.released_turn_indices,
                "status": driven.result.session.status,
                "fallback_reason": driven.result.session.fallback_reason,
            }
        )
        if progress is not None:
            progress(index, len(ordered), scenario)

    report = evaluate_agent_scenario_runs(
        runs,
        visible_scenarios=visible,
        ground_truth=truth,
        evidence_labels=evidence,
    )
    return _publish_results(
        runs=runs,
        driver_rows=driver_rows,
        report=report,
        output=output,
        split=split,
    )


def replace_rule_agent_benchmark_outputs(
    *,
    base_root: str | Path,
    replacement_root: str | Path,
    benchmark_root: str | Path,
    output_root: str | Path,
) -> RuleAgentBenchmarkResult:
    """Replace explicitly rerun scenarios and reevaluate the complete benchmark."""

    base_path = Path(base_root)
    replacement_path = Path(replacement_root)
    base_runs = {
        item.scenario_id: item
        for item in load_agent_scenario_runs(base_path / "scenario_runs.jsonl")
    }
    replacements = {
        item.scenario_id: item
        for item in load_agent_scenario_runs(replacement_path / "scenario_runs.jsonl")
    }
    if not replacements or not set(replacements).issubset(base_runs):
        raise ValueError("replacement scenarios must be a nonempty subset of base runs")
    base_driver = {
        str(item["scenario_id"]): item
        for item in _load_jsonl(base_path / "driver_results.jsonl")
    }
    replacement_driver = {
        str(item["scenario_id"]): item
        for item in _load_jsonl(replacement_path / "driver_results.jsonl")
    }
    if set(replacements) != set(replacement_driver):
        raise ValueError("replacement runs and driver rows do not align")
    base_runs.update(replacements)
    base_driver.update(replacement_driver)
    root = Path(benchmark_root)
    visible = load_visible_scenarios(root / "visible" / "scenarios.jsonl")
    truth = load_scenario_ground_truth(root / "hidden" / "ground_truth.jsonl")
    evidence = load_evidence_labels(root / "hidden" / "evidence_labels.parquet")
    if set(base_runs) != {item.scenario_id for item in visible}:
        raise ValueError("patched runs do not cover the complete benchmark")
    ordered_runs = tuple(base_runs[key] for key in sorted(base_runs))
    ordered_driver = tuple(base_driver[key] for key in sorted(base_driver))
    report = evaluate_agent_scenario_runs(
        ordered_runs,
        visible_scenarios=visible,
        ground_truth=truth,
        evidence_labels=evidence,
    )
    return _publish_results(
        runs=ordered_runs,
        driver_rows=ordered_driver,
        report=report,
        output=Path(output_root),
        split="all",
    )


def _publish_results(
    *,
    runs: Sequence[AgentScenarioRun],
    driver_rows: Sequence[dict[str, object]],
    report: AgentEvaluationReport,
    output: Path,
    split: BenchmarkSplit,
) -> RuleAgentBenchmarkResult:
    runs_path = write_agent_scenario_runs(runs, output / "scenario_runs.jsonl")
    metrics_path = write_agent_evaluation_report(report, output / "metrics.json")
    driver_path = _write_jsonl(output / "driver_results.jsonl", driver_rows)
    runtime_metrics = _runtime_metrics(runs)
    runtime_metrics_path = _atomic_write(
        output / "runtime_metrics.json",
        json.dumps(runtime_metrics, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    run_by_id = {item.scenario_id: item for item in runs}
    driver_by_id = {str(item["scenario_id"]): item for item in driver_rows}
    failures = [
        {
            "scenario_id": result.scenario_id,
            "split": result.split,
            "category": result.category,
            "fallback": run_by_id[result.scenario_id].fallback,
            "fallback_reason": run_by_id[result.scenario_id].fallback_reason,
            "driver_stop_reason": driver_by_id[result.scenario_id]["stop_reason"],
            "violations": result.violations,
        }
        for result in report.scenario_results
        if (
            result.violations
            or run_by_id[result.scenario_id].fallback
            or driver_by_id[result.scenario_id]["stop_reason"] != "completed"
        )
    ]
    failures_path = _write_jsonl(output / "failures.jsonl", failures)
    summary_path = _atomic_write(
        output / "summary.md",
        _markdown_summary(
            report,
            split=split,
            failure_count=len(failures),
            runtime_metrics=runtime_metrics,
        ),
    )
    return RuleAgentBenchmarkResult(
        output_root=output,
        runs_path=runs_path,
        metrics_path=metrics_path,
        failures_path=failures_path,
        driver_path=driver_path,
        runtime_metrics_path=runtime_metrics_path,
        summary_path=summary_path,
        report=report,
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Rule Agent driver output does not exist: {path}")
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Rule Agent driver row must be an object: {path}")
            rows.append(value)
    return rows


def _runtime_metrics(
    runs: Sequence[AgentScenarioRun],
) -> dict[str, int | float]:
    scenario_count = len(runs)
    step_count = sum(
        len(turn.actions) for run in runs for turn in run.turns
    )
    tool_call_count = sum(
        len(turn.tool_calls) for run in runs for turn in run.turns
    )
    turn_count = sum(len(run.turns) for run in runs)
    tool_calls = [
        call for run in runs for turn in run.turns for call in turn.tool_calls
    ]
    semantic_calls = [call for call in tool_calls if call.tool_kind == "semantic"]
    semantic_input_tokens = sum(call.input_tokens or 0 for call in semantic_calls)
    return {
        "scenario_count": scenario_count,
        "turn_count": turn_count,
        "step_count": step_count,
        "tool_call_count": tool_call_count,
        "average_steps_per_scenario": step_count / scenario_count,
        "average_tool_calls_per_scenario": tool_call_count / scenario_count,
        "average_turns_per_scenario": turn_count / scenario_count,
        "llm_call_count": 0,
        "llm_cost_usd": 0.0,
        "semantic_tool_call_count": len(semantic_calls),
        "semantic_input_tokens": semantic_input_tokens,
        "semantic_cache_hit_rate": (
            sum(call.cache_hit for call in semantic_calls) / len(semantic_calls)
            if semantic_calls
            else 0.0
        ),
    }


def _snapshot_awaiting_run(session: object) -> AgentScenarioRun:
    from yelp_agent.agent_harness import AgentSession

    state = AgentSession.model_validate(session)
    if state.status != "awaiting_user" or not state.turns:
        raise RuntimeError(
            f"scenario {state.scenario_id} produced neither a run "
            "nor a visible question"
        )
    return AgentScenarioRun(
        scenario_id=state.scenario_id,
        agent_version=state.agent_version,
        turns=state.turns,
        fallback=False,
        latency_ms=state.elapsed_ms,
        input_tokens=state.input_tokens if state.token_usage_observed else None,
        output_tokens=state.output_tokens if state.token_usage_observed else None,
        cost_usd=state.cost_usd if state.cost_usd > 0 else None,
    )
def _select(
    scenarios: Sequence[VisibleAgentScenario],
    split: BenchmarkSplit,
) -> tuple[VisibleAgentScenario, ...]:
    if split == "all":
        return tuple(scenarios)
    selected = tuple(item for item in scenarios if item.split == split)
    if not selected:
        raise ValueError(f"benchmark split contains no scenarios: {split}")
    return selected


def _execution_order(scenario: VisibleAgentScenario) -> tuple[int, str]:
    # Development always runs first. The frozen config is then used unchanged
    # for validation, which prevents validation-label rule tuning.
    return (0 if scenario.split == "development" else 1, scenario.scenario_id)


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return _atomic_write(path, content)


def _atomic_write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_text(content, encoding="utf-8", newline="\n")
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return path


def _metric(report: AgentEvaluationReport, name: str) -> str:
    score = report.metrics.get(name)
    if score is None or score.value is None:
        return "N/A"
    if score.unit == "ratio":
        return f"{score.value:.2%}"
    return f"{score.value:.4f}"


def _markdown_summary(
    report: AgentEvaluationReport,
    *,
    split: BenchmarkSplit,
    failure_count: int,
    runtime_metrics: dict[str, int | float],
) -> str:
    key_metrics = (
        "action_accuracy",
        "tool_selection_accuracy",
        "invalid_action_rate",
        "repeated_tool_call_rate",
        "direct_return_precision",
        "missing_field_detection_precision",
        "missing_field_detection_recall",
        "unnecessary_question_rate",
        "question_answerability_rate",
        "fallback_rate",
        "valid_candidate_rate",
        "empty_result_rate",
        "mean_latency_ms",
        "p95_latency_ms",
    )
    lines = [
        "# 第 24 步 Rule Agent V1 实验摘要",
        "",
        f"- 评测范围：`{split}`",
        f"- 场景数：{report.scenario_count}",
        f"- 含违规或回退的场景：{failure_count}",
        "- Router：确定性规则，不调用 LLM",
        "- Review RAG：未实现（计划第 27 步）",
        "- 官网实时核验：未实现，因此相关问题保守回答",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {_metric(report, name)} |" for name in key_metrics)
    lines.extend(
        [
            "| `average_steps_per_scenario` | "
            f"{float(runtime_metrics['average_steps_per_scenario']):.4f} |",
            "| `average_tool_calls_per_scenario` | "
            f"{float(runtime_metrics['average_tool_calls_per_scenario']):.4f} |",
            "| `average_turns_per_scenario` | "
            f"{float(runtime_metrics['average_turns_per_scenario']):.4f} |",
            "| `llm_call_count` | 0 |",
            "| `llm_cost_usd` | 0.0000 |",
        ]
    )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "本结果是后续 Constrained LLM Router 与 Cost-aware LLM Router 的零 LLM 基线。",
            "Review 证据检索和官网核验尚未安装，所以对应场景的低分属于已知能力边界，",
            "不会通过伪造评论引用或把静态 Yelp 字段说成实时政策来抬高指标。",
            "",
        ]
    )
    return "\n".join(lines)
