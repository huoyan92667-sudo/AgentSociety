"""Compare target-rank movement for two recommendation prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.evaluation.comparison import (
    compare_prediction_files,
    write_prediction_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = compare_prediction_files(
        args.tasks,
        args.ground_truth,
        args.baseline,
        args.challenger,
        task_limit=args.limit,
    )
    write_prediction_comparison(comparison, args.output)
    print(comparison.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
