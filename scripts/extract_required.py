"""Safely extract only business, review, and user JSONL from Yelp TAR."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.data.extract import (
    extract_required_members,
    write_extraction_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/yelp_dataset.tar"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/extraction_report.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing required JSON files after atomic extraction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = extract_required_members(
        args.archive,
        args.output_dir,
        force=args.force,
    )
    write_extraction_report(result, args.report)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
