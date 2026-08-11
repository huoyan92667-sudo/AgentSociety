"""Compare frozen Step 24 Hybrid and Step 25 semantic reranking offline.

This is an evaluation-only script. Hidden labels are loaded here for scoring and
are never passed to the Agent runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from yelp_agent.agent_benchmark import load_scenario_ground_truth, load_visible_scenarios


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["scenario_id"])] = row
    return rows


def _final_turn(run: dict[str, Any]) -> dict[str, Any]:
    turns = run.get("turns") or []
    for turn in reversed(turns):
        if turn.get("candidate_ranking"):
            return turn
    return turns[-1] if turns else {}


def _rank(ranking: list[str], acceptable: set[str]) -> int | None:
    positions = [index + 1 for index, value in enumerate(ranking) if value in acceptable]
    return min(positions) if positions else None


def _hit(ranking: list[str], acceptable: set[str], cutoff: int) -> bool:
    return any(value in acceptable for value in ranking[:cutoff])


def _metric_value(payload: dict[str, Any], name: str) -> float | None:
    metric = payload.get("metrics", {}).get(name, {})
    value = metric.get("value")
    return None if value is None else float(value)


def _metric_delta(
    baseline: dict[str, Any], semantic: dict[str, Any], name: str
) -> dict[str, float | None]:
    before = _metric_value(baseline, name)
    after = _metric_value(semantic, name)
    return {"step24": before, "step25": after, "delta": None if before is None or after is None else after - before}


def build_report(
    *,
    baseline_runs: dict[str, dict[str, Any]],
    semantic_runs: dict[str, dict[str, Any]],
    baseline_metrics: dict[str, Any],
    semantic_metrics: dict[str, Any],
    visible: dict[str, Any],
    truth: dict[str, Any],
) -> dict[str, Any]:
    if set(baseline_runs) != set(semantic_runs):
        raise ValueError("Step 24 and Step 25 scenario IDs do not match")
    if set(baseline_runs) != set(truth):
        raise ValueError("scenario runs and hidden labels do not match")

    rank_rows: list[dict[str, Any]] = []
    semantic_calls = 0
    semantic_successes = 0
    semantic_fallbacks = 0
    top30_changed = 0
    improvement_counts: Counter[str] = Counter()
    for scenario_id in sorted(semantic_runs):
        baseline_turn = _final_turn(baseline_runs[scenario_id])
        semantic_turn = _final_turn(semantic_runs[scenario_id])
        before = list(baseline_turn.get("candidate_ranking") or [])
        after = list(semantic_turn.get("candidate_ranking") or [])
        acceptable = set(truth[scenario_id].get("acceptable_business_ids") or [])
        if before and after:
            before_rank = _rank(before, acceptable)
            after_rank = _rank(after, acceptable)
            if before[:30] != after[:30]:
                top30_changed += 1
            if before_rank is not None and after_rank is not None:
                if after_rank < before_rank:
                    outcome = "improved"
                elif after_rank > before_rank:
                    outcome = "worsened"
                else:
                    outcome = "unchanged"
                improvement_counts[outcome] += 1
            else:
                outcome = "not_in_candidate_ranking"
                improvement_counts[outcome] += 1
            rank_rows.append(
                {
                    "scenario_id": scenario_id,
                    "split": visible[scenario_id]["split"],
                    "query_text": visible[scenario_id]["query_text"],
                    "acceptable_count": len(acceptable),
                    "hybrid_rank": before_rank,
                    "semantic_rank": after_rank,
                    "rank_delta": (
                        None
                        if before_rank is None or after_rank is None
                        else before_rank - after_rank
                    ),
                    "outcome": outcome,
                    "top3_before": before[:3],
                    "top3_after": after[:3],
                }
            )
        for turn in semantic_runs[scenario_id].get("turns", []):
            for call in turn.get("tool_calls", []):
                if call.get("tool_name") != "COMPUTE_EMBEDDING_MATCH":
                    continue
                semantic_calls += 1
                if call.get("status") == "completed":
                    semantic_successes += 1
                if call.get("status") != "completed" or semantic_runs[scenario_id].get("fallback"):
                    semantic_fallbacks += 1

    rank_rows.sort(
        key=lambda row: (
            row["rank_delta"] is None,
            -(row["rank_delta"] or 0),
            row["scenario_id"],
        )
    )
    ranking_cases = len(rank_rows)
    rank_pairs = [
        row for row in rank_rows if row["hybrid_rank"] is not None and row["semantic_rank"] is not None
    ]

    ranking_metrics: dict[str, Any] = {
        "ranking_case_count": ranking_cases,
        "semantic_call_count": semantic_calls,
        "semantic_success_rate": semantic_successes / semantic_calls if semantic_calls else 0.0,
        "semantic_fallback_count": semantic_fallbacks,
        "top30_changed_rate": top30_changed / ranking_cases if ranking_cases else 0.0,
        "improved_count": improvement_counts["improved"],
        "worsened_count": improvement_counts["worsened"],
        "unchanged_count": improvement_counts["unchanged"],
        "not_in_candidate_ranking_count": improvement_counts["not_in_candidate_ranking"],
    }
    if rank_pairs:
        ranking_metrics["mean_hybrid_rank"] = sum(row["hybrid_rank"] for row in rank_pairs) / len(rank_pairs)
        ranking_metrics["mean_semantic_rank"] = sum(row["semantic_rank"] for row in rank_pairs) / len(rank_pairs)
        ranking_metrics["mean_rank_delta"] = sum(row["rank_delta"] for row in rank_pairs) / len(rank_pairs)
        for cutoff in (1, 3, 5, 10, 30):
            ranking_metrics[f"hybrid_hit_at_{cutoff}"] = sum(
                row["hybrid_rank"] is not None and row["hybrid_rank"] <= cutoff for row in rank_pairs
            ) / len(rank_pairs)
            ranking_metrics[f"semantic_hit_at_{cutoff}"] = sum(
                row["semantic_rank"] is not None and row["semantic_rank"] <= cutoff for row in rank_pairs
            ) / len(rank_pairs)
            ranking_metrics[f"hybrid_mrr_at_{cutoff}"] = sum(
                1.0 / row["hybrid_rank"] if row["hybrid_rank"] is not None and row["hybrid_rank"] <= cutoff else 0.0
                for row in rank_pairs
            ) / len(rank_pairs)
            ranking_metrics[f"semantic_mrr_at_{cutoff}"] = sum(
                1.0 / row["semantic_rank"] if row["semantic_rank"] is not None and row["semantic_rank"] <= cutoff else 0.0
                for row in rank_pairs
            ) / len(rank_pairs)

    metric_names = sorted(set(baseline_metrics.get("metrics", {})) | set(semantic_metrics.get("metrics", {})))
    return {
        "schema_version": 1,
        "scenario_count": len(semantic_runs),
        "split_counts": dict(Counter(item["split"] for item in visible.values())),
        "step24_metrics": {name: _metric_value(baseline_metrics, name) for name in metric_names},
        "step25_metrics": {name: _metric_value(semantic_metrics, name) for name in metric_names},
        "metric_deltas_step25_minus_step24": {
            name: _metric_delta(baseline_metrics, semantic_metrics, name) for name in metric_names
        },
        "ranking_metrics": ranking_metrics,
        "largest_improvements": rank_rows[:10],
        "largest_regressions": [
            row for row in sorted(
                rank_rows,
                key=lambda value: (
                    value["rank_delta"] is None,
                    value["rank_delta"] if value["rank_delta"] is not None else math.inf,
                    value["scenario_id"],
                ),
            )[:10]
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    delta = report["metric_deltas_step25_minus_step24"]
    ranking = report["ranking_metrics"]
    lines = [
        "# Step 25 正式 500 场景实验与重排序效果报告",
        "",
        "本报告比较冻结的 Step 24 Rule Agent Hybrid 与 Step 25 本地 Qwen3 Embedding Top-30 语义重排。隐藏标签只在离线评测脚本中读取，没有传入 Agent。",
        "",
        f"- 场景数：{report['scenario_count']}（development {report['split_counts'].get('development', 0)}，validation {report['split_counts'].get('validation', 0)}）",
        "- Step 25：静态商家文档 Embedding，Hybrid 前 30 家融合，31 名以后保持 Hybrid 顺序",
        "",
        "## 官方 Agent 指标差值",
        "",
        "| 指标 | Step 24 | Step 25 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in delta.items():
        before = values["step24"]
        after = values["step25"]
        change = values["delta"]
        fmt = lambda value: "N/A" if value is None else f"{value:.4f}"
        lines.append(f"| `{name}` | {fmt(before)} | {fmt(after)} | {fmt(change)} |")
    lines.extend(
        [
            "",
            "## 候选排名变化",
            "",
            f"- 可比较候选排名场景：{ranking['ranking_case_count']}",
            f"- Embedding 调用成功率：{ranking['semantic_success_rate']:.2%}",
            f"- Top-30 顺序发生变化：{ranking['top30_changed_rate']:.2%}",
            f"- 正确候选排名提高：{ranking['improved_count']}",
            f"- 正确候选排名下降：{ranking['worsened_count']}",
            f"- 正确候选排名不变：{ranking['unchanged_count']}",
            f"- 平均排名变化（Hybrid 排名 - Step25 排名）：{ranking.get('mean_rank_delta', 0.0):.4f}",
            "",
            "| 截止名次 | Hybrid Hit | Step25 Semantic Hit | Hybrid MRR | Step25 Semantic MRR |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for cutoff in (1, 3, 5, 10, 30):
        lines.append(
            f"| {cutoff} | {ranking.get(f'hybrid_hit_at_{cutoff}', 0):.2%} | "
            f"{ranking.get(f'semantic_hit_at_{cutoff}', 0):.2%} | "
            f"{ranking.get(f'hybrid_mrr_at_{cutoff}', 0):.4f} | "
            f"{ranking.get(f'semantic_mrr_at_{cutoff}', 0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "Step 25 的目标不是改变召回集合，而是在已有 Hybrid 候选中理解当前 query 的语义。若 Hit/MRR 提升，说明语义重排把更符合即时需求的商家推到了前面；若不变或下降，也说明当前静态商家文档和 Embedding 仍不足，下一步应分析失败案例，而不是直接宣称 Agent 有提升。",
            "",
            "完整逐场景排名变化保存在 `reranking_comparison.json`，其中包含最大提升和最大回退案例。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--semantic-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    baseline_root = args.baseline_root or root / "runs" / "rule_agent_v1"
    semantic_root = args.semantic_root or root / "runs" / "rule_agent_embedding_v2"
    output_root = args.output_root or semantic_root
    sources = root / "benchmarks" / "agent_scenarios_v1"
    baseline_runs = _read_jsonl(baseline_root / "scenario_runs.jsonl")
    semantic_runs = _read_jsonl(semantic_root / "scenario_runs.jsonl")
    visible = {
        item.scenario_id: item.model_dump(mode="json")
        for item in load_visible_scenarios(sources / "visible" / "scenarios.jsonl")
    }
    truth = {
        item.scenario_id: item.model_dump(mode="json")
        for item in load_scenario_ground_truth(sources / "hidden" / "ground_truth.jsonl")
    }
    baseline_metrics = json.loads((baseline_root / "metrics.json").read_text(encoding="utf-8"))
    semantic_metrics = json.loads((semantic_root / "metrics.json").read_text(encoding="utf-8"))
    report = build_report(
        baseline_runs=baseline_runs,
        semantic_runs=semantic_runs,
        baseline_metrics=baseline_metrics,
        semantic_metrics=semantic_metrics,
        visible=visible,
        truth=truth,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reranking_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "reranking_comparison.md").write_text(
        _markdown(report), encoding="utf-8", newline="\n"
    )
    print(json.dumps(report["ranking_metrics"], ensure_ascii=False, sort_keys=True))
    print(f"json={output_root / 'reranking_comparison.json'}")
    print(f"markdown={output_root / 'reranking_comparison.md'}")


if __name__ == "__main__":
    main()
