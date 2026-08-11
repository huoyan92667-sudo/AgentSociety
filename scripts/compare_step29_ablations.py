"""Compare frozen Step 29 Agent variants without rerunning any scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_METRICS = (
    "action_accuracy",
    "tool_selection_accuracy",
    "direct_return_precision",
    "missing_field_detection_precision",
    "missing_field_detection_recall",
    "unnecessary_question_rate",
    "grounded_answer_rate",
    "citation_correctness",
    "unsupported_claim_rate",
    "conflict_detection_accuracy",
    "official_policy_caution_accuracy",
    "valid_candidate_rate",
    "hr_at_1",
    "hr_at_3",
    "hr_at_5",
    "mrr",
    "fallback_rate",
    "mean_latency_ms",
    "mean_tokens",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant in NAME=OUTPUT_DIRECTORY form; repeat for every variant.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/controlled_llm_v1/ablation"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    variants = [_parse_variant(item, root) for item in args.variant]
    names = [name for name, _ in variants]
    if len(names) != len(set(names)):
        raise ValueError("variant names must be unique")
    rows = [_load_variant(name, path) for name, path in variants]
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "comparison_kind": "step29_controlled_llm_ablation",
        "variants": rows,
    }
    json_path = output / "ablation_comparison.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = output / "ablation_comparison.md"
    markdown_path.write_text(_markdown(rows), encoding="utf-8")
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")


def _parse_variant(value: str, root: Path) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise ValueError("--variant must use NAME=OUTPUT_DIRECTORY")
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = root / path
    return name.strip(), path


def _load_variant(name: str, path: Path) -> dict[str, object]:
    metrics_path = path / "metrics.json"
    runtime_path = path / "runtime_metrics.json"
    if not metrics_path.is_file() or not runtime_path.is_file():
        raise FileNotFoundError(f"variant {name!r} is incomplete: {path}")
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    measured = metrics_payload.get("metrics", {})
    values = {
        metric: _metric_value(measured.get(metric)) for metric in DEFAULT_METRICS
    }
    return {
        "name": name,
        "output_root": str(path),
        "scenario_count": metrics_payload.get("scenario_count"),
        "metrics": values,
        "runtime": {
            "average_steps_per_scenario": runtime_payload.get(
                "average_steps_per_scenario"
            ),
            "average_tool_calls_per_scenario": runtime_payload.get(
                "average_tool_calls_per_scenario"
            ),
            "llm_call_count": runtime_payload.get("llm_call_count", 0),
            "llm_input_tokens": runtime_payload.get("llm_input_tokens", 0),
            "llm_output_tokens": runtime_payload.get("llm_output_tokens", 0),
            "llm_total_tokens": runtime_payload.get("llm_total_tokens", 0),
            "llm_cache_hit_count": runtime_payload.get("llm_cache_hit_count", 0),
            "llm_failure_count": runtime_payload.get("llm_failure_count", 0),
        },
    }


def _metric_value(value: object) -> float | None:
    if not isinstance(value, dict) or value.get("status") != "measured":
        return None
    raw = value.get("value")
    return float(raw) if isinstance(raw, (int, float)) else None


def _markdown(rows: list[dict[str, object]]) -> str:
    headline = (
        "action_accuracy",
        "grounded_answer_rate",
        "citation_correctness",
        "unsupported_claim_rate",
        "hr_at_5",
        "fallback_rate",
        "mean_latency_ms",
    )
    lines = [
        "# Step 29 受控大模型消融对比",
        "",
        "| 变体 | 场景数 | 动作准确率 | 有依据回答率 | 引用正确率 | 无依据声明率 | HR@5 | 回退率 | 平均延迟(ms) | Provider Token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        runtime = row["runtime"]
        assert isinstance(metrics, dict) and isinstance(runtime, dict)
        values = [_format_metric(metrics.get(key), key) for key in headline]
        lines.append(
            "| {name} | {count} | {values} | {tokens} |".format(
                name=row["name"],
                count=row["scenario_count"],
                values=" | ".join(values),
                tokens=runtime.get("llm_total_tokens", 0),
            )
        )
    lines.extend(
        [
            "",
            "说明：Provider Token 只统计该次正式运行真正发往 API 的调用；缓存命中不会重复计费。",
            "",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: object, name: str) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    if name == "mean_latency_ms":
        return f"{value:.1f}"
    return f"{100 * value:.2f}%"


if __name__ == "__main__":
    main()
