"""Summarize one Agent traces.jsonl into runtime_metrics.json."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.evaluation.agent_runtime import (
    summarize_agent_trace_file,
    write_agent_runtime_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Agent traces.jsonl to summarize",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to runtime_metrics.json beside the trace file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.traces.with_name("runtime_metrics.json")
    metrics = summarize_agent_trace_file(args.traces)
    write_agent_runtime_metrics(metrics, output)
    print(metrics.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
