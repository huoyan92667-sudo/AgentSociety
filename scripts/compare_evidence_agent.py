"""Create the formal Step 27 versus Step 28 500-scenario report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from yelp_agent.agent_benchmark import load_scenario_ground_truth


METRICS = (
    "review_retrieval_recall_at_1",
    "review_retrieval_recall_at_3",
    "review_retrieval_recall_at_5",
    "evidence_precision_at_1",
    "evidence_precision_at_3",
    "evidence_precision_at_5",
    "business_scope_isolation_rate",
    "grounded_answer_rate",
    "unsupported_claim_rate",
    "citation_correctness",
    "evidence_recency_reporting_rate",
    "conflict_detection_accuracy",
    "official_policy_caution_accuracy",
    "unnecessary_rag_call_rate",
    "fallback_rate",
    "mean_latency_ms",
    "p95_latency_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    step27_root = root / "runs" / "rule_agent_review_rag_v1"
    step28_root = root / "runs" / "rule_agent_evidence_v1"
    step27 = _read_json(step27_root / "metrics.json")
    step28 = _read_json(step28_root / "metrics.json")
    runs = _jsonl(step28_root / "scenario_runs.jsonl")
    truth = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(
            root / "benchmarks" / "agent_scenarios_v1" / "hidden" / "ground_truth.jsonl"
        )
    }
    required = {
        scenario_id
        for scenario_id, item in truth.items()
        if "retrieve_business_reviews" in item.required_actions
    }
    search_calls = _tool_calls(runs, "SEARCH_BUSINESS_REVIEWS")
    aggregate_calls = _tool_calls(runs, "AGGREGATE_REVIEW_EVIDENCE")
    searched = {scenario_id for scenario_id, _ in search_calls}
    aggregated = {scenario_id for scenario_id, _ in aggregate_calls}
    missing = sorted(required - aggregated)
    aggregate_latencies = [float(call.get("latency_ms") or 0) for _, call in aggregate_calls]
    aggregate_claims = [
        claim
        for run in runs.values()
        for turn in run.get("turns", [])
        for claim in turn.get("claims", [])
        if str(claim.get("claim_id", "")).startswith("aggregate:")
    ]
    development = _read_json(
        root / "runs" / "evidence_aggregator_v1" / "development_tuning.json"
    )
    validation = _read_json(
        root / "runs" / "evidence_aggregator_v1" / "validation_aggregation_metrics.json"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "scenario_count": 500,
        "development_count": 400,
        "validation_count": 100,
        "execution_strategy": {
            "rerun_scenarios": len(required),
            "reused_unchanged_step27_scenarios": 500 - len(required),
            "reason": (
                "Only scenarios whose frozen contract requires Review retrieval can "
                "reach the new aggregation tool; all other Step 27 traces are reused."
            ),
        },
        "review_required_scenario_count": len(required),
        "review_search_scenario_count": len(searched),
        "evidence_aggregation_scenario_count": len(aggregated),
        "evidence_aggregation_coverage": len(aggregated) / len(required),
        "missing_aggregation_count": len(missing),
        "missing_aggregation_by_category": dict(
            sorted(Counter(truth[item].scenario_category for item in missing).items())
        ),
        "selected_policy": development["selected_policy"],
        "direct_development_metrics": next(
            item["metrics"]
            for item in development["candidates"]
            if item["policy"]["policy_version"]
            == development["selected_policy"]["policy_version"]
        ),
        "direct_validation_metrics": validation["metrics"],
        "metrics": {
            "step27": {name: _metric(step27, name) for name in METRICS},
            "step28": {name: _metric(step28, name) for name in METRICS},
        },
        "deltas": {
            name: _delta(_metric(step28, name), _metric(step27, name))
            for name in METRICS
        },
        "step28_by_split": {
            split: {name: _metric(step28["by_split"][split], name) for name in METRICS}
            for split in ("development", "validation")
        },
        "aggregation_usage": {
            "tool_call_count": len(aggregate_calls),
            "completed_count": sum(
                call.get("status") == "completed" for _, call in aggregate_calls
            ),
            "cache_hit_count": sum(bool(call.get("cache_hit")) for _, call in aggregate_calls),
            "aggregate_claim_count": len(aggregate_claims),
            "aggregate_citation_count": sum(
                len(claim.get("evidence_refs", [])) for claim in aggregate_claims
            ),
            "mean_tool_latency_ms": (
                sum(aggregate_latencies) / len(aggregate_latencies)
                if aggregate_latencies
                else 0.0
            ),
            "p95_tool_latency_ms": _percentile(aggregate_latencies, 0.95),
            "external_api_calls": 0,
            "billed_tokens": 0,
            "cost_cny": 0.0,
        },
        "validation_used_for_tuning": False,
        "label_quality": "Step 13 traceable silver labels, not human gold labels",
    }
    json_path = step28_root / "evidence_aggregation_comparison.json"
    md_path = step28_root / "evidence_aggregation_comparison.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))


def _tool_calls(
    runs: dict[str, dict[str, Any]], tool_name: str
) -> list[tuple[str, dict[str, Any]]]:
    return [
        (scenario_id, call)
        for scenario_id, run in runs.items()
        for turn in run.get("turns", [])
        for call in turn.get("tool_calls", [])
        if call.get("tool_name") == tool_name
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["scenario_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _metric(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get("metrics", {}).get(name, {}).get("value")
    return None if value is None else float(value)


def _delta(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else first - second


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 第 28 步正式 500 场景 Evidence Aggregator 报告",
        "",
        "流程：Business-scoped Review RAG → 证据原子 → 时间/相关性加权 → 共识与冲突 → 可信回答策略。",
        "策略仅在 Development 选择；Validation 冻结运行。没有调用 LLM 或外部 API。",
        "",
        "## Step 27 与 Step 28",
        "",
        "| 指标 | Step27 | Step28 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for name in METRICS:
        before = report["metrics"]["step27"][name]
        after = report["metrics"]["step28"][name]
        delta = report["deltas"][name]
        if name.endswith("latency_ms"):
            lines.append(f"| `{name}` | {before:.2f} | {after:.2f} | {delta:+.2f} |")
        else:
            lines.append(f"| `{name}` | {_pct(before)} | {_pct(after)} | {_pct(delta)} |")
    direct_dev = report["direct_development_metrics"]
    direct_val = report["direct_validation_metrics"]
    usage = report["aggregation_usage"]
    lines.extend(
        [
            "",
            "## 聚合策略与覆盖",
            "",
            f"- 冻结策略：`{report['selected_policy']['policy_version']}`。",
            f"- Development：回答策略准确率 {_pct(direct_dev['response_policy_correct'])}，冲突准确率 {_pct(direct_dev['conflict_correct'])}。",
            f"- Validation：回答策略准确率 {_pct(direct_val['response_policy_correct'])}，冲突准确率 {_pct(direct_val['conflict_correct'])}，证据立场准确率 {_pct(direct_val['stance_accuracy'])}。",
            f"- 130 道应使用 Review 的题中，{report['evidence_aggregation_scenario_count']} 道实际进入聚合，覆盖率 {_pct(report['evidence_aggregation_coverage'])}。",
            f"- 未进入聚合的 {report['missing_aggregation_count']} 道分桶为 `{report['missing_aggregation_by_category']}`，沿用第 27 步已知的语义解析限制。",
            "",
            "## 成本与解释",
            "",
            f"- 聚合工具调用 {usage['tool_call_count']} 次，产生 {usage['aggregate_claim_count']} 条结构化结论和 {usage['aggregate_citation_count']} 个 Review 引用。",
            f"- 聚合工具平均/P95 延迟 {usage['mean_tool_latency_ms']:.2f}/{usage['p95_tool_latency_ms']:.2f} ms。",
            "- 外部 API=0，计费 token=0，费用=0 元。",
            "- Review Recall 没变化是正确现象：第 28 步不重新检索，只解释第 27 步已经找到的证据。",
            "- 标签是可追溯银标签，不是人工金标；Validation 结果只报告，不反向调整阈值。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
