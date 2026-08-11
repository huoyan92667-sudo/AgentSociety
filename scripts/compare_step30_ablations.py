"""Compare Step 29 and Step 30 semantic-ranking ablations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from yelp_agent.agent_benchmark import load_scenario_ground_truth
from yelp_agent.semantic_ranking import SemanticRankingResult


KEY_METRICS = (
    "action_accuracy",
    "tool_selection_accuracy",
    "direct_return_precision",
    "hr_at_1",
    "hr_at_3",
    "hr_at_5",
    "mrr",
    "ndcg_at_5",
    "valid_candidate_rate",
    "fallback_rate",
    "mean_latency_ms",
    "p95_latency_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME=OUTPUT_ROOT; repeat for every ablation",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    variants = dict(_parse_variant(root, value) for value in args.variant)
    truth = {
        item.scenario_id: set(item.acceptable_business_ids)
        for item in load_scenario_ground_truth(
            root / "benchmarks" / "agent_scenarios_v1" / "hidden" / "ground_truth.jsonl"
        )
    }
    summaries: dict[str, dict[str, object]] = {}
    for name, path in variants.items():
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        summary: dict[str, object] = {
            "scenario_count": metrics.get("scenario_count"),
            "metrics": {
                key: _metric_value(metrics, key) for key in KEY_METRICS
            },
        }
        diagnostics_path = path / "semantic_ranking_diagnostics.jsonl"
        if diagnostics_path.is_file():
            diagnostics = [
                SemanticRankingResult.model_validate_json(line)
                for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summary["semantic_ranking"] = _diagnostic_summary(diagnostics, truth)
        summaries[name] = summary
    output = _resolve(root, args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "variants": summaries}
    json_path = output / "step30_ablation_comparison.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = output / "step30_ablation_comparison.md"
    markdown_path.write_text(_markdown(summaries), encoding="utf-8")
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")


def _diagnostic_summary(
    diagnostics: list[SemanticRankingResult],
    truth: dict[str, set[str]],
) -> dict[str, float | int]:
    latest_by_context = {
        str(item.context_id): item
        for item in diagnostics
        if item.context_id is not None
    }
    diagnostics = list(latest_by_context.values())
    changed = 0
    fallback = 0
    no_op = 0
    llm_signal = 0
    improved = 0
    harmed = 0
    comparable = 0
    protected = 0
    movements: list[int] = []
    input_tokens = 0
    logical_tokens = 0
    for item in diagnostics:
        changed += item.ranking != item.base_ranking
        fallback += item.fallback
        no_op += item.no_op_reason is not None
        llm_signal += item.intent.semantic_model_condition_count > 0
        protected += any(
            row.protection_reason_codes for row in item.candidate_scores
        )
        movements.extend(abs(row.rank_movement) for row in item.candidate_scores)
        input_tokens += item.usage.actual_input_tokens
        logical_tokens += item.usage.logical_input_tokens
        acceptable = truth.get(str(item.context_id), set())
        if acceptable:
            comparable += 1
            base_rank = _first_rank(item.base_ranking, acceptable)
            final_rank = _first_rank(item.ranking, acceptable)
            improved += final_rank is not None and (
                base_rank is None or final_rank < base_rank
            )
            harmed += base_rank is not None and (
                final_rank is None or final_rank > base_rank
            )
    total = len(diagnostics)
    return {
        "diagnostic_count": total,
        "changed_ranking_rate": changed / total if total else 0,
        "no_op_rate": no_op / total if total else 0,
        "fallback_rate": fallback / total if total else 0,
        "llm_signal_rate": llm_signal / total if total else 0,
        "rank_protection_trigger_rate": protected / total if total else 0,
        "comparable_count": comparable,
        "target_improvement_rate": improved / comparable if comparable else 0,
        "target_harm_rate": harmed / comparable if comparable else 0,
        "mean_absolute_movement": (
            sum(movements) / len(movements) if movements else 0
        ),
        "local_model_input_tokens": input_tokens,
        "local_model_logical_tokens": logical_tokens,
    }


def _first_rank(ranking: list[str], acceptable: set[str]) -> int | None:
    return next(
        (index for index, business_id in enumerate(ranking, 1) if business_id in acceptable),
        None,
    )


def _metric_value(payload: dict[str, object], key: str) -> float | None:
    metrics = payload.get("metrics")
    row = metrics.get(key) if isinstance(metrics, dict) else None
    value = row.get("value") if isinstance(row, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _markdown(summaries: dict[str, dict[str, object]]) -> str:
    names = list(summaries)
    lines = [
        "# Step 30 语义排序与保守融合消融",
        "",
        "| 指标 | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for metric in KEY_METRICS:
        values = []
        for name in names:
            metrics = summaries[name].get("metrics")
            value = metrics.get(metric) if isinstance(metrics, dict) else None
            if value is None:
                values.append("N/A")
            elif metric.endswith("latency_ms"):
                values.append(f"{float(value):.2f}")
            else:
                values.append(f"{float(value):.2%}")
        lines.append(f"| `{metric}` | " + " | ".join(values) + " |")
    lines.extend(["", "## 排名诊断", ""])
    for name in names:
        diagnostic = summaries[name].get("semantic_ranking")
        if not isinstance(diagnostic, dict):
            continue
        lines.extend(
            [
                f"### {name}",
                "",
                f"- 发生排序变化：{float(diagnostic['changed_ranking_rate']):.2%}",
                f"- 正确商家改善：{float(diagnostic['target_improvement_rate']):.2%}",
                f"- 正确商家受损：{float(diagnostic['target_harm_rate']):.2%}",
                f"- Rank Protection 触发：{float(diagnostic['rank_protection_trigger_rate']):.2%}",
                f"- 平均绝对名次变化：{float(diagnostic['mean_absolute_movement']):.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def _parse_variant(root: Path, value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("variant must use NAME=OUTPUT_ROOT")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise ValueError("variant name cannot be blank")
    return name.strip(), _resolve(root, Path(raw_path))


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
