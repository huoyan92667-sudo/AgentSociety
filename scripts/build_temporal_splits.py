"""Build leak-resistant validation and test targets from interactions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.data.temporal import (
    build_temporal_splits,
    write_temporal_split_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("data/processed/interactions.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/task_dataset"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/temporal_split_report.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_temporal_splits(
        args.interactions,
        args.output_root,
        force=args.force,
    )
    write_temporal_split_report(result, args.report)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
