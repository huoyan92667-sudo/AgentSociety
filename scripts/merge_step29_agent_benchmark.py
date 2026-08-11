"""Merge frozen Step 29 splits and preserve LLM traces/usage accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.controlled_llm import augment_runtime_metrics
from yelp_agent.rule_router import (
    RuleAgentSourcePaths,
    merge_rule_agent_benchmark_outputs,
)

COUNT_KEYS = (
    "logical_call_count",
    "provider_call_count",
    "success_count",
    "failure_count",
    "cache_hit_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "usage_unknown_count",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    development = _resolve(root, args.development_root)
    validation = _resolve(root, args.validation_root)
    output = _resolve(root, args.output_root)
    sources = RuleAgentSourcePaths.from_project_root(root)
    result = merge_rule_agent_benchmark_outputs(
        development_root=development,
        validation_root=validation,
        benchmark_root=sources.benchmark_root,
        output_root=output,
    )
    usages = [_load_usage(path) for path in (development, validation)]
    merged_usage = _merge_usages(usages)
    usage_path = output / "llm_usage.json"
    usage_path.write_text(
        json.dumps(merged_usage, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    calls_path = output / "llm_calls.jsonl"
    call_lines: list[str] = []
    for split_root in (development, validation):
        source = split_root / "llm_calls.jsonl"
        if source.is_file():
            call_lines.extend(source.read_text(encoding="utf-8").splitlines())
    calls_path.write_text(
        "" if not call_lines else "\n".join(call_lines) + "\n",
        encoding="utf-8",
    )
    augment_runtime_metrics(result.runtime_metrics_path, merged_usage)
    print(f"scenarios={result.report.scenario_count}")
    print(f"metrics={result.metrics_path}")
    print(f"calls={calls_path}")
    print(f"usage={usage_path}")


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _load_usage(path: Path) -> dict[str, object]:
    source = path / "llm_usage.json"
    if not source.is_file():
        raise FileNotFoundError(f"missing Step 29 usage file: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"usage file must contain an object: {source}")
    return payload


def _merge_usages(usages: list[dict[str, object]]) -> dict[str, object]:
    merged: dict[str, object] = {
        "schema_version": 1,
        "accounting_kind": "merged_view_not_additional_provider_usage",
    }
    for key in COUNT_KEYS:
        merged[key] = sum(int(item.get(key, 0) or 0) for item in usages)
    capabilities = sorted(
        {
            capability
            for usage in usages
            for capability in (
                usage.get("by_capability", {}).keys()
                if isinstance(usage.get("by_capability"), dict)
                else ()
            )
        }
    )
    by_capability: dict[str, object] = {}
    for capability in capabilities:
        rows = []
        for usage in usages:
            by_value = usage.get("by_capability", {})
            if isinstance(by_value, dict) and isinstance(by_value.get(capability), dict):
                rows.append(by_value[capability])
        combined = {key: sum(int(row.get(key, 0) or 0) for row in rows) for key in COUNT_KEYS}
        latency_denominator = sum(int(row.get("logical_call_count", 0) or 0) for row in rows)
        latency_numerator = sum(
            float(row.get("mean_latency_ms", 0) or 0)
            * int(row.get("logical_call_count", 0) or 0)
            for row in rows
        )
        combined["mean_latency_ms"] = (
            latency_numerator / latency_denominator if latency_denominator else 0.0
        )
        by_capability[capability] = combined
    merged["by_capability"] = by_capability
    return merged


if __name__ == "__main__":
    main()
