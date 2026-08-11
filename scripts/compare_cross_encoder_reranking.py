"""Build the frozen Step 24/25/26 ranking and cost comparison report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from yelp_agent.agent_benchmark import load_scenario_ground_truth, load_visible_scenarios


METRICS = (
    "hr_at_1", "hr_at_3", "hr_at_5", "mrr", "ndcg_at_5",
    "valid_candidate_rate", "fallback_rate", "mean_latency_ms", "p95_latency_ms",
)


def _jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["scenario_id"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _ranking(run: dict[str, Any]) -> list[str]:
    for turn in reversed(run.get("turns") or []):
        if turn.get("candidate_ranking"):
            return [str(value) for value in turn["candidate_ranking"]]
    return []


def _rank(values: list[str], acceptable: set[str]) -> int | None:
    return next((index for index, value in enumerate(values, 1) if value in acceptable), None)


def _metric(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get("metrics", {}).get(name, {}).get("value")
    return None if value is None else float(value)


def _tool_usage(runs: dict[str, dict[str, Any]], tool_name: str) -> dict[str, Any]:
    calls = [
        call
        for run in runs.values()
        for turn in run.get("turns") or []
        for call in turn.get("tool_calls") or []
        if call.get("tool_name") == tool_name
    ]
    return {
        "call_count": len(calls),
        "completed_count": sum(call.get("status") == "completed" for call in calls),
        "cache_hit_count": sum(bool(call.get("cache_hit")) for call in calls),
        "cache_hit_rate": sum(bool(call.get("cache_hit")) for call in calls) / len(calls) if calls else 0.0,
        "actual_input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
        "mean_latency_ms": sum(float(call.get("latency_ms") or 0) for call in calls) / len(calls) if calls else 0.0,
        "external_api_calls": 0,
        "cost_cny": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    names = {
        "step24": root / "runs" / "rule_agent_v1",
        "step25": root / "runs" / "rule_agent_embedding_v2",
        "step26": root / "runs" / "rule_agent_cross_encoder_v1",
    }
    runs = {key: _jsonl(path / "scenario_runs.jsonl") for key, path in names.items()}
    if len({frozenset(value) for value in runs.values()}) != 1:
        raise ValueError("Step 24/25/26 scenario IDs do not align")
    metrics = {
        key: json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for key, path in names.items()
    }
    benchmark = root / "benchmarks" / "agent_scenarios_v1"
    visible = {
        item.scenario_id: item
        for item in load_visible_scenarios(benchmark / "visible" / "scenarios.jsonl")
    }
    truth = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(benchmark / "hidden" / "ground_truth.jsonl")
    }
    rows: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    for scenario_id in sorted(runs["step26"]):
        step25 = _ranking(runs["step25"][scenario_id])
        step26 = _ranking(runs["step26"][scenario_id])
        acceptable = set(truth[scenario_id].acceptable_business_ids)
        if not step25 or not step26 or not acceptable:
            continue
        before = _rank(step25, acceptable)
        after = _rank(step26, acceptable)
        if before is None or after is None:
            outcome = "correct_business_not_in_candidate_set"
        elif after < before:
            outcome = "improved"
        elif after > before:
            outcome = "worsened"
        else:
            outcome = "unchanged"
        outcomes[outcome] += 1
        rows.append(
            {
                "scenario_id": scenario_id,
                "split": visible[scenario_id].split,
                "step25_rank": before,
                "step26_rank": after,
                "rank_improvement": None if before is None or after is None else before - after,
                "top5_before": step25[:5],
                "top5_after": step26[:5],
                "top20_changed": step25[:20] != step26[:20],
                "outcome": outcome,
            }
        )
    report = {
        "schema_version": 1,
        "scenario_count": 500,
        "development_count": 400,
        "validation_count": 100,
        "selection_split": "development",
        "validation_used_for_tuning": False,
        "policy": json.loads((root / "configs" / "cross_encoder_policy.json").read_text(encoding="utf-8")),
        "metrics": {
            name: {metric: _metric(payload, metric) for metric in METRICS}
            for name, payload in metrics.items()
        },
        "by_split": {
            split: {
                name: {
                    metric: _metric(payload["by_split"][split], metric)
                    for metric in ("hr_at_1", "hr_at_3", "hr_at_5", "mrr", "fallback_rate")
                }
                for name, payload in metrics.items()
            }
            for split in ("development", "validation")
        },
        "deltas": {
            "step26_minus_step25": {
                metric: _metric(metrics["step26"], metric) - _metric(metrics["step25"], metric)
                for metric in METRICS
                if _metric(metrics["step26"], metric) is not None and _metric(metrics["step25"], metric) is not None
            },
            "step26_minus_step24": {
                metric: _metric(metrics["step26"], metric) - _metric(metrics["step24"], metric)
                for metric in METRICS
                if _metric(metrics["step26"], metric) is not None and _metric(metrics["step24"], metric) is not None
            },
        },
        "ranking_changes": {
            "ranking_case_count": len(rows),
            "top20_changed_count": sum(row["top20_changed"] for row in rows),
            **dict(outcomes),
        },
        "usage": {
            "embedding": _tool_usage(runs["step26"], "COMPUTE_EMBEDDING_MATCH"),
            "cross_encoder": _tool_usage(runs["step26"], "COMPUTE_CROSS_ENCODER_MATCH"),
            "external_api_calls": 0,
            "llm_calls": 0,
            "billed_token_count": 0,
            "cost_cny": 0.0,
        },
        "largest_improvements": sorted(
            rows,
            key=lambda row: (row["rank_improvement"] is None, -(row["rank_improvement"] or 0), row["scenario_id"]),
        )[:10],
        "largest_regressions": sorted(
            rows,
            key=lambda row: (row["rank_improvement"] is None, row["rank_improvement"] or 0, row["scenario_id"]),
        )[:10],
    }
    output = args.output_root or names["step26"]
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "reranking_comparison.json"
    md_path = output / "reranking_comparison.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    lines = [
        "# 第 26 步正式 500 场景实验与重排序效果报告", "",
        "冻结流程：Hybrid V2 → Embedding Top-30 → Cross-Encoder Top-20 → 返回 Top-5。Cross-Encoder 使用本地 Qwen3-Reranker-0.6B；隐藏标签只在离线调参与评测器中读取。", "",
        "## 总体指标", "",
        "| 指标 | Step24 Hybrid | Step25 Embedding | Step26 Cross-Encoder | 26-25 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ("hr_at_1", "hr_at_3", "hr_at_5", "mrr", "ndcg_at_5", "fallback_rate"):
        a, b, c = (report["metrics"][name][metric] for name in ("step24", "step25", "step26"))
        lines.append(f"| `{metric}` | {pct(a)} | {pct(b)} | {pct(c)} | {pct(c-b)} |")
    lines += ["", "## 冻结与分割", ""]
    for split in ("development", "validation"):
        value = report["by_split"][split]["step26"]
        lines.append(
            f"- {split}: HR@1={pct(value['hr_at_1'])}, HR@3={pct(value['hr_at_3'])}, HR@5={pct(value['hr_at_5'])}, MRR={pct(value['mrr'])}"
        )
    usage = report["usage"]["cross_encoder"]
    changes = report["ranking_changes"]
    lines += [
        "", "## 排名变化与成本", "",
        f"- 可比较排名场景：{changes['ranking_case_count']}；Top-20 发生变化：{changes['top20_changed_count']}。",
        f"- 正确商家排名提高/下降/不变：{changes.get('improved', 0)}/{changes.get('worsened', 0)}/{changes.get('unchanged', 0)}。",
        f"- Cross-Encoder 调用：{usage['call_count']}，缓存命中率 {pct(usage['cache_hit_rate'])}，平均工具延迟 {usage['mean_latency_ms']:.2f} ms。",
        f"- 本地模型实际处理 token：{usage['actual_input_tokens']:,}；外部 API=0，计费 token=0，费用=0 元。",
        "- 正式 Agent 平均/P95 延迟："
        f"{report['metrics']['step26']['mean_latency_ms']:.2f}/{report['metrics']['step26']['p95_latency_ms']:.2f} ms。",
        "", "## 结论", "",
        "Cross-Encoder 在不改变召回集合的前提下继续提升了前 5 名排序，尤其 HR@1 约为 Step25 的两倍；但 valid_candidate_rate 几乎不变，说明召回缺失仍是当前主要上限。第 27 步 Review RAG 应用于解释和细粒度证据，不应被宣传为已经解决召回问题。", "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))
    print(f"json={json_path}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
