"""Create the formal Step 26 versus Step 27 500-scenario report."""

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
    "citation_correctness",
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
    step26_root = root / "runs" / "rule_agent_cross_encoder_v1"
    step27_root = root / "runs" / "rule_agent_review_rag_v1"
    step26 = _read_json(step26_root / "metrics.json")
    step27 = _read_json(step27_root / "metrics.json")
    runs = _jsonl(step27_root / "scenario_runs.jsonl")
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
    calls = [
        (scenario_id, call)
        for scenario_id, run in runs.items()
        for turn in run.get("turns", [])
        for call in turn.get("tool_calls", [])
        if call.get("tool_name") == "SEARCH_BUSINESS_REVIEWS"
    ]
    called_scenarios = {scenario_id for scenario_id, _ in calls}
    evidence = [
        (scenario_id, item["evidence"])
        for scenario_id, call in calls
        for item in call.get("retrieved_evidence", [])
    ]
    latencies = [float(call.get("latency_ms") or 0.0) for _, call in calls]
    missing = sorted(required - called_scenarios)
    category_counts = Counter(truth[scenario_id].scenario_category for scenario_id in missing)
    direct_tuning = _read_json(root / "runs" / "review_rag_v1" / "development_tuning.json")
    selected_metrics = next(
        item["metrics"]
        for item in direct_tuning["candidates"]
        if item["policy"]["policy_version"]
        == direct_tuning["selected_policy"]["policy_version"]
    )
    direct_validation = _read_json(
        root / "runs" / "review_rag_v1" / "validation_retrieval_metrics.json"
    )["metrics"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "scenario_count": 500,
        "development_count": 400,
        "validation_count": 100,
        "review_required_scenario_count": len(required),
        "review_tool_called_scenario_count": len(called_scenarios),
        "review_tool_coverage": len(called_scenarios) / len(required),
        "missing_review_tool_scenario_count": len(missing),
        "missing_review_tool_by_category": dict(sorted(category_counts.items())),
        "retrieved_evidence_count": len(evidence),
        "actual_retrieved_business_scope_isolation": (
            sum(
                item["business_id"] in truth[scenario_id].business_scope
                for scenario_id, item in evidence
            )
            / len(evidence)
            if evidence
            else 0.0
        ),
        "policy": direct_tuning["selected_policy"],
        "direct_development_retriever_metrics": selected_metrics,
        "direct_validation_retriever_metrics": direct_validation,
        "metrics": {
            "step26": {name: _metric(step26, name) for name in METRICS},
            "step27": {name: _metric(step27, name) for name in METRICS},
        },
        "deltas": {
            name: _delta(_metric(step27, name), _metric(step26, name))
            for name in METRICS
        },
        "step27_by_split": {
            split: {
                name: _metric(step27["by_split"][split], name)
                for name in METRICS
            }
            for split in ("development", "validation")
        },
        "review_rag_usage": {
            "tool_call_count": len(calls),
            "completed_count": sum(call.get("status") == "completed" for _, call in calls),
            "cache_hit_count": sum(bool(call.get("cache_hit")) for _, call in calls),
            "cache_hit_rate": (
                sum(bool(call.get("cache_hit")) for _, call in calls) / len(calls)
                if calls
                else 0.0
            ),
            "actual_local_input_tokens": sum(
                int(call.get("input_tokens") or 0) for _, call in calls
            ),
            "mean_tool_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "p95_tool_latency_ms": _percentile(latencies, 0.95),
            "external_api_calls": 0,
            "billed_tokens": 0,
            "cost_cny": 0.0,
        },
        "label_quality": "Step 13 traceable silver labels, not human gold labels",
        "validation_used_for_tuning": False,
    }
    json_path = step27_root / "review_rag_comparison.json"
    md_path = step27_root / "review_rag_comparison.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))


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
    metrics = report["metrics"]
    lines = [
        "# 第 27 步正式 500 场景 Review RAG 报告",
        "",
        "冻结流程：明确商家作用域 → 严格 cutoff → Aspect/BM25 候选 → 本地 Qwen Embedding → RRF → Top-5 Review ID。",
        "Development 用于选择 RRF 权重；Validation 只运行一次并报告。外部 API、计费 token 和费用均为 0。",
        "",
        "## Step 26 与 Step 27",
        "",
        "| 指标 | Step26 无 Review RAG | Step27 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for name in METRICS:
        before = metrics["step26"][name]
        after = metrics["step27"][name]
        delta = report["deltas"][name]
        if name.endswith("latency_ms"):
            lines.append(
                f"| `{name}` | {before:.2f} | {after:.2f} | {delta:+.2f} |"
            )
        else:
            lines.append(
                f"| `{name}` | {_pct(before)} | {_pct(after)} | {_pct(delta)} |"
            )
    usage = report["review_rag_usage"]
    direct = report["direct_development_retriever_metrics"]
    direct_validation = report["direct_validation_retriever_metrics"]
    lines.extend(
        [
            "",
            "## 检索器与完整 Agent 的差别",
            "",
            f"- 直接运行 104 道 Development RAG 题：Recall@1/3/5={_pct(direct['recall_at_1'])}/{_pct(direct['recall_at_3'])}/{_pct(direct['recall_at_5'])}。",
            f"- 冻结后直接运行 26 道 Validation RAG 题：Recall@1/3/5={_pct(direct_validation['recall_at_1'])}/{_pct(direct_validation['recall_at_3'])}/{_pct(direct_validation['recall_at_5'])}。",
            f"- 完整 Agent 应调用 RAG 的题共 {report['review_required_scenario_count']} 道，实际调用 {report['review_tool_called_scenario_count']} 道，覆盖率 {_pct(report['review_tool_coverage'])}。",
            f"- 漏调用 {report['missing_review_tool_scenario_count']} 道，分桶：`{report['missing_review_tool_by_category']}`。这是语义解析/路由问题，不是 Store 串店。",
            f"- 对实际返回的 {report['retrieved_evidence_count']} 条证据，真实商家作用域隔离率 {_pct(report['actual_retrieved_business_scope_isolation'])}。正式 evaluator 会把应调用却没调用的题也按失败计入，因此总体指标更低。",
            "",
            "## 成本与边界",
            "",
            f"- Review RAG 工具调用 {usage['tool_call_count']} 次；缓存命中率 {_pct(usage['cache_hit_rate'])}。",
            f"- 正式运行新增本地输入 token {usage['actual_local_input_tokens']:,}；平均/P95 工具延迟 {usage['mean_tool_latency_ms']:.2f}/{usage['p95_tool_latency_ms']:.2f} ms。",
            "- 外部 API=0，计费 token=0，费用=0 元。",
            "- 标签来自第 13 步高置信度规则抽取，是可追溯银标签，不是人工金标；Precision 会低估未被银标签覆盖但实际上相关的评论。",
            "- 第 27 步返回原始证据和确定性引用，不负责跨评论事实聚合；冲突聚合属于第 28 步。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
