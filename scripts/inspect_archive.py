"""Inspect the Yelp TAR without extracting any member."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.data.archive import inspect_archive, write_inspection_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/yelp_dataset.tar"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/data_inspection.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inspection = inspect_archive(args.archive)
    write_inspection_report(inspection, args.output)
    print(inspection.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
