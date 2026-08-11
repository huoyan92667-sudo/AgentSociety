"""Select the semantic acceptance threshold from Development cache only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.config import load_llm_environment
from yelp_agent.controlled_llm import (
    CacheOnlyChatGenerator,
    ControlledJSONCaller,
    ControlledLLMUsageLedger,
    ControlledParserAdapter,
    ControlledRequestInterpreter,
    ControlledSemanticEnhancer,
    SqliteControlledLLMCache,
    load_controlled_llm_config,
)
from yelp_agent.query import build_rule_based_request_parser
from yelp_agent.query.benchmark import evaluate_request_parser, load_query_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.72, 0.78, 0.82, 0.86, 0.90]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/controlled_llm_v1/semantic_tuning.json")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_controlled_llm_config(root / "configs" / "controlled_llm.yaml")
    model = load_llm_environment().model
    if not model:
        raise ValueError("OPENAI_MODEL is required to address the existing cache")
    cases = tuple(
        item
        for item in load_query_benchmark(
            root / "benchmarks" / "query_aware_v2" / "queries_500.jsonl"
        )
        if item.split == "development"
    )
    baseline = evaluate_request_parser(build_rule_based_request_parser(), cases)
    rows: list[dict[str, object]] = []
    for threshold in args.thresholds:
        semantic_config = config.semantic.model_copy(
            update={"minimum_signal_confidence": threshold}
        )
        ledger = ControlledLLMUsageLedger()
        with SqliteControlledLLMCache(
            root / config.cache_relative_path
        ) as cache:
            caller = ControlledJSONCaller(
                generator=CacheOnlyChatGenerator(),
                model_name=model,
                cache=cache,
                ledger=ledger,
            )
            parser_adapter = ControlledParserAdapter(
                ControlledRequestInterpreter(
                    ControlledSemanticEnhancer(
                        config=semantic_config,
                        caller=caller,
                    )
                )
            )
            report = evaluate_request_parser(parser_adapter, cases)  # type: ignore[arg-type]
        usage = ledger.summary()
        if usage["provider_call_count"] != 0 or usage["failure_count"] != 0:
            raise RuntimeError("semantic tuning requires complete validated cache coverage")
        rows.append(
            {
                "minimum_signal_confidence": threshold,
                "condition_f1": report.condition_f1,
                "exact_match_rate": report.exact_match_rate,
                "condition_precision": report.condition_precision,
                "condition_recall": report.condition_recall,
                "missing_fields_exact_match_rate": report.missing_fields_exact_match_rate,
                "cache_hit_count": usage["cache_hit_count"],
            }
        )
    selected = max(
        rows,
        key=lambda item: (
            float(item["condition_f1"]),
            float(item["exact_match_rate"]),
            float(item["condition_precision"]),
            float(item["minimum_signal_confidence"]),
        ),
    )
    payload = {
        "schema_version": 1,
        "selection_split": "development",
        "validation_used_for_selection": False,
        "selection_objective": [
            "condition_f1",
            "exact_match_rate",
            "condition_precision",
            "higher_threshold",
        ],
        "baseline": {
            "condition_f1": baseline.condition_f1,
            "exact_match_rate": baseline.exact_match_rate,
            "condition_precision": baseline.condition_precision,
            "condition_recall": baseline.condition_recall,
            "missing_fields_exact_match_rate": baseline.missing_fields_exact_match_rate,
        },
        "candidates": rows,
        "selected": selected,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
