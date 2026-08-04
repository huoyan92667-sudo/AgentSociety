"""Filter Yelp business JSONL into static Philadelphia Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.config import load_config
from yelp_agent.data.businesses import (
    preprocess_businesses,
    write_business_preprocess_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/yelp_academic_dataset_business.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/businesses.parquet"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/business_preprocess_report.json"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config_dir)
    result = preprocess_businesses(
        args.input,
        args.output,
        config.data,
        force=args.force,
    )
    write_business_preprocess_report(result, args.report)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
