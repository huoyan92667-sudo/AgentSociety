"""Evaluate the controlled semantic parser on the frozen 500-query benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.agent_harness import RuleBasedRequestInterpreter
from yelp_agent.controlled_llm import (
    ControlledParserAdapter,
    ControlledRequestInterpreter,
    SemanticEnhancementInput,
    SemanticEscalationPolicy,
    build_controlled_llm_runtime,
    load_controlled_llm_config,
)
from yelp_agent.query.benchmark import (
    evaluate_request_parser,
    load_query_benchmark,
    write_query_parser_benchmark_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--split", choices=("development", "validation", "all"), default="development"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--triggered-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--controlled-llm-config",
        type=Path,
        default=Path("configs/controlled_llm.yaml"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.controlled_llm_config
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_controlled_llm_config(config_path)
    cases = load_query_benchmark(
        root / "benchmarks" / "query_aware_v2" / "queries_500.jsonl"
    )
    if args.split != "all":
        cases = tuple(item for item in cases if item.split == args.split)
    if args.triggered_only:
        policy = SemanticEscalationPolicy(config.semantic)
        baseline = RuleBasedRequestInterpreter()
        cases = tuple(
            item
            for item in cases
            if policy.should_call(
                SemanticEnhancementInput(
                    base_request=(parsed := baseline.interpret(item.parse_input())).request,
                    base_readiness=parsed.readiness,
                    language=item.language,
                )
            )
        )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        cases = cases[: args.limit]
    output = args.output_root or root / "runs" / "controlled_llm_v1" / "query" / args.split
    with build_controlled_llm_runtime(
        project_root=root,
        config=config,
    ) as runtime:
        controlled = ControlledParserAdapter(
            ControlledRequestInterpreter(runtime.semantic_enhancer)
        )
        report = evaluate_request_parser(controlled, cases)  # type: ignore[arg-type]
        report_path = write_query_parser_benchmark_report(
            report, output / "parser_benchmark.json"
        )
        traces_path, usage_path = runtime.ledger.write(output)
    print(f"report={report_path}")
    print(f"calls={traces_path}")
    print(f"usage={usage_path}")
    print(
        f"cases={report.case_count} f1={report.condition_f1:.4f} "
        f"exact={report.exact_match_rate:.4f}"
    )


if __name__ == "__main__":
    main()
