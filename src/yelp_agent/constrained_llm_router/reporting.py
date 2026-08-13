"""Trace-derived Step 35 Router usage and failure reporting."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence

from yelp_agent.agent_evaluation.schema import AgentScenarioRun, RouterDecisionTrace

from .schema import RouterExperimentSummary


def summarize_router_runs(
    runs: Sequence[AgentScenarioRun],
) -> RouterExperimentSummary:
    traces = [
        action.router_decision
        for run in runs
        for turn in run.turns
        for action in turn.actions
        if action.router_decision is not None
    ]
    typed: list[RouterDecisionTrace] = [item for item in traces if item is not None]
    model = [item for item in typed if item.router_kind == "constrained_llm"]
    provider = [item for item in typed if item.provider_called]
    known = [item for item in provider if item.total_tokens is not None]
    return RouterExperimentSummary(
        decision_count=len(typed),
        model_decision_count=len(model),
        single_choice_bypass_count=sum(
            item.router_kind == "single_choice_bypass" for item in typed
        ),
        rule_fallback_count=sum(item.router_kind == "rule_fallback" for item in typed),
        task_correction_count=sum(
            item.input_task_type is not None
            and item.selected_task_type is not None
            and item.input_task_type != item.selected_task_type
            for item in typed
        ),
        information_gap_correction_count=sum(
            item.input_information_gaps != item.selected_information_gaps
            for item in typed
        ),
        invalid_output_count=sum(item.status == "invalid_output" for item in typed),
        low_confidence_count=sum(item.status == "low_confidence" for item in typed),
        provider_call_count=sum(item.attempt_count for item in provider),
        cache_hit_count=sum(item.cache_hit for item in typed),
        input_tokens=sum(item.input_tokens or 0 for item in known),
        output_tokens=sum(item.output_tokens or 0 for item in known),
        total_tokens=sum(item.total_tokens or 0 for item in known),
        usage_unknown_count=sum(
            item.provider_called and item.total_tokens is None for item in typed
        ),
        mean_latency_ms=(
            sum(item.latency_ms for item in provider) / len(provider)
            if provider
            else 0.0
        ),
    )


def write_router_report(
    runs: Sequence[AgentScenarioRun],
    root: str | Path,
) -> tuple[Path, Path]:
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_router_runs(runs)
    json_path = output / "router_metrics.json"
    md_path = output / "router_summary.md"
    json_path.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        "\n".join(
            [
                "# Constrained LLM Router 运行摘要",
                "",
                f"- 决策总数：{summary.decision_count}",
                f"- LLM 选择：{summary.model_decision_count}",
                f"- 单选项直接执行：{summary.single_choice_bypass_count}",
                f"- Rule Router 回退：{summary.rule_fallback_count}",
                f"- 任务类型修正：{summary.task_correction_count}",
                f"- Provider 尝试：{summary.provider_call_count}",
                f"- 输入 / 输出 / 总 Token：{summary.input_tokens} / {summary.output_tokens} / {summary.total_tokens}",
                f"- 平均 Provider 延迟：{summary.mean_latency_ms:.2f} ms",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, md_path
