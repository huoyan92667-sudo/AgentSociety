"""Visible/hidden-separated evaluation of Step 34 multi-turn memory behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from yelp_agent.agent_benchmark import (
    ScriptedUserTurn,
    VisibleAgentScenario,
    load_scenario_ground_truth,
    load_visible_scenarios,
)
from yelp_agent.agent_harness import RuleBasedRequestInterpreter
from yelp_agent.query import QueryParseInput

from .manager import SessionMemoryManager
from .reducer import record_memory_observation
from .schema import MemoryTurnInput, SessionMemory


@dataclass(frozen=True, slots=True)
class MemoryBenchmarkResult:
    output_root: Path
    cases_path: Path
    metrics_path: Path
    summary_path: Path
    metrics: dict[str, object]


def run_session_memory_benchmark(
    manager: SessionMemoryManager,
    *,
    benchmark_root: str | Path,
    output_root: str | Path,
) -> MemoryBenchmarkResult:
    """Evaluate the 130 scripted scenarios without exposing labels to memory code."""

    root = Path(benchmark_root)
    output = Path(output_root)
    visible = {
        item.scenario_id: item
        for item in load_visible_scenarios(root / "visible" / "scenarios.jsonl")
    }
    truths = load_scenario_ground_truth(root / "hidden" / "ground_truth.jsonl")
    rows: list[dict[str, object]] = []
    for truth in sorted(truths, key=lambda item: item.scenario_id):
        if not truth.scripted_user_turns:
            continue
        scenario = visible[truth.scenario_id]
        memory = _start(manager, scenario)
        if truth.business_scope:
            display = list(truth.business_scope[:5])
            memory = record_memory_observation(
                memory,
                turn_index=1,
                business_scope=list(truth.business_scope),
                presented_business_ids=display,
            )
            assert memory is not None
        for scripted in sorted(truth.scripted_user_turns, key=lambda item: item.turn_index):
            result = _continue(manager, scenario, memory, scripted)
            memory = result.memory
            expected_rejected = set(scripted.rejected_business_ids)
            predicted_rejected = set(memory.rejected_business_ids)
            expected_added = {_expected_condition_key(item) for item in scripted.added_conditions}
            predicted = {_condition_key(item) for item in result.request.conditions}
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "split": scenario.split,
                    "turn_index": scripted.turn_index,
                    "task_type_correct": (
                        result.readiness.task_type == scripted.expected_task_type
                    ),
                    "information_gaps_correct": (
                        set(result.readiness.information_gaps)
                        == set(scripted.expected_information_gaps)
                    ),
                    "condition_recall": (
                        1.0
                        if not expected_added
                        else len(expected_added & predicted) / len(expected_added)
                    ),
                    "rejected_business_recall": (
                        1.0
                        if not expected_rejected
                        else len(expected_rejected & predicted_rejected)
                        / len(expected_rejected)
                    ),
                    "reference_scope_valid": (
                        set(result.request.referenced_business_ids)
                        .issubset(set(truth.business_scope))
                    ),
                    "memory_revision": memory.revision,
                    "provider_called": result.extraction.provider_called,
                    "rule_fallback": result.extraction.status == "rule_fallback",
                    "input_tokens": result.extraction.input_tokens,
                    "output_tokens": result.extraction.output_tokens,
                    "latency_ms": result.extraction.latency_ms,
                    "accepted_changes": result.accepted_changes,
                    "rejected_changes": result.rejected_changes,
                }
            )
    metrics = _metrics(rows)
    cases_path = _write_jsonl(output / "memory_cases.jsonl", rows)
    metrics_path = _write_text(
        output / "metrics.json",
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    summary_path = _write_text(output / "summary.md", _summary(metrics))
    return MemoryBenchmarkResult(
        output_root=output,
        cases_path=cases_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        metrics=metrics,
    )


def _start(
    manager: SessionMemoryManager,
    scenario: VisibleAgentScenario,
) -> SessionMemory:
    interpreted = RuleBasedRequestInterpreter().interpret(
        QueryParseInput(
            user_id=scenario.user_id,
            session_id=scenario.session_id,
            cutoff_time=scenario.cutoff_time,
            query_text=scenario.query_text,
            user_latitude=scenario.user_latitude,
            user_longitude=scenario.user_longitude,
            referenced_business_ids=scenario.referenced_business_ids,
        )
    )
    return manager.update(
        MemoryTurnInput(
            query_text=scenario.query_text,
            language=scenario.language,
            current_turn=1,
            base_request=interpreted.request,
            base_readiness=interpreted.readiness,
            explicit_referenced_business_ids=scenario.referenced_business_ids,
        )
    ).memory


def _continue(
    manager: SessionMemoryManager,
    scenario: VisibleAgentScenario,
    memory: SessionMemory,
    scripted: ScriptedUserTurn,
):
    latitude = scripted.state_updates.get("user_latitude")
    longitude = scripted.state_updates.get("user_longitude")
    value = QueryParseInput(
        user_id=scenario.user_id,
        session_id=scenario.session_id,
        cutoff_time=scenario.cutoff_time,
        query_text=scripted.query_text,
        user_latitude=float(latitude) if isinstance(latitude, (int, float)) else None,
        user_longitude=(
            float(longitude) if isinstance(longitude, (int, float)) else None
        ),
    )
    interpreted = RuleBasedRequestInterpreter().interpret(value)
    return manager.update(
        MemoryTurnInput(
            query_text=scripted.query_text,
            language=scenario.language,
            current_turn=scripted.turn_index,
            base_request=interpreted.request,
            base_readiness=interpreted.readiness,
            previous_memory=memory,
        )
    )


def _condition_key(item: object) -> tuple[str, str, str]:
    return (
        str(getattr(item, "field")),
        str(getattr(item, "operator")),
        str(getattr(item, "value")).casefold(),
    )


def _expected_condition_key(item: object) -> tuple[str, str, str]:
    return (
        str(getattr(item, "field")),
        str(getattr(item, "operator")),
        str(getattr(item, "value")).casefold(),
    )


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    if count == 0:
        raise ValueError("memory benchmark selected no scripted turns")
    provider = [item for item in rows if item["provider_called"]]
    return {
        "schema_version": 1,
        "scripted_turn_count": count,
        "task_type_accuracy": sum(bool(item["task_type_correct"]) for item in rows)
        / count,
        "information_gap_exact_match": sum(
            bool(item["information_gaps_correct"]) for item in rows
        )
        / count,
        "condition_recall": sum(float(item["condition_recall"]) for item in rows)
        / count,
        "rejected_business_recall": sum(
            float(item["rejected_business_recall"]) for item in rows
        )
        / count,
        "reference_scope_valid_rate": sum(
            bool(item["reference_scope_valid"]) for item in rows
        )
        / count,
        "provider_call_count": len(provider),
        "rule_fallback_rate": sum(bool(item["rule_fallback"]) for item in rows)
        / count,
        "input_tokens": sum(int(item["input_tokens"] or 0) for item in provider),
        "output_tokens": sum(int(item["output_tokens"] or 0) for item in provider),
        "mean_provider_latency_ms": (
            sum(float(item["latency_ms"]) for item in provider) / len(provider)
            if provider
            else 0.0
        ),
        "hidden_labels_visible_to_manager": False,
    }


def _summary(metrics: dict[str, object]) -> str:
    def pct(name: str) -> str:
        return f"{float(metrics[name]):.2%}"

    return "\n".join(
        [
            "# 第34步：统一会话记忆评测",
            "",
            f"- 多轮脚本数：{metrics['scripted_turn_count']}",
            f"- 任务类型准确率：{pct('task_type_accuracy')}",
            f"- 信息缺口完全匹配：{pct('information_gap_exact_match')}",
            f"- 新增条件召回率：{pct('condition_recall')}",
            f"- 拒绝商家召回率：{pct('rejected_business_recall')}",
            f"- 引用范围合法率：{pct('reference_scope_valid_rate')}",
            f"- API 调用数：{metrics['provider_call_count']}",
            f"- 规则回退率：{pct('rule_fallback_rate')}",
            f"- API 输入 Token：{metrics['input_tokens']}",
            f"- API 输出 Token：{metrics['output_tokens']}",
            "- Ground Truth 只由离线评测器读取，Memory Manager 不接收隐藏标签。",
            "",
        ]
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    return _write_text(
        path,
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows),
    )


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(content, encoding="utf-8", newline="\n")
    partial.replace(path)
    return path
