"""Evaluate the deterministic Step 18 request parser on frozen seed queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.config import load_query_aware_config
from yelp_agent.query import build_rule_based_request_parser
from yelp_agent.query.benchmark import (
    evaluate_request_parser,
    load_query_benchmark,
    write_query_parser_benchmark_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/query_aware_v1/parser_benchmark.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_query_aware_config(args.config_dir)
    parser = build_rule_based_request_parser()
    if parser.version != config.rule_parser_version:
        raise RuntimeError("Query parser version disagrees with frozen configuration")
    cases = load_query_benchmark(config.benchmark_path)
    report = evaluate_request_parser(parser, cases)
    path = write_query_parser_benchmark_report(report, args.output)
    print(report.model_dump_json(indent=2))
    print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()
