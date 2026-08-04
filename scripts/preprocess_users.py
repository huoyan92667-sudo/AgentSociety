"""Select eligible Yelp users and freeze their interaction histories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.config import load_config
from yelp_agent.data.users import (
    preprocess_users_and_interactions,
    write_user_preprocess_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/processed/reviews.parquet"),
    )
    parser.add_argument(
        "--raw-users",
        type=Path,
        default=Path("data/raw/yelp_academic_dataset_user.json"),
    )
    parser.add_argument(
        "--users-output",
        type=Path,
        default=Path("data/processed/users.parquet"),
    )
    parser.add_argument(
        "--interactions-output",
        type=Path,
        default=Path("data/processed/interactions.parquet"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/user_preprocess_report.json"),
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
    result = preprocess_users_and_interactions(
        args.reviews,
        args.raw_users,
        args.users_output,
        args.interactions_output,
        config.data,
        force=args.force,
    )
    write_user_preprocess_report(result, args.report)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
