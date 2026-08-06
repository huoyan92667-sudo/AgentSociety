"""Freeze target-safe positive, negative and neutral Item-KNN events."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from yelp_agent.collaborative.artifacts import build_item_knn_artifacts
from yelp_agent.config import load_item_knn_config
from yelp_agent.experiments import write_json_artifact


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
        default=Path("data/features/item_knn"),
    )
    parser.add_argument(
        "--excluded-users",
        type=Path,
        help="optional Parquet containing a user_id column",
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/item_knn_v1/artifact_build_report.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    excluded_user_ids: set[str] = set()
    if args.excluded_users is not None:
        rows = (
            pq.read_table(
                args.excluded_users,
                columns=["user_id"],
            )
            .column("user_id")
            .to_pylist()
        )
        excluded_user_ids = {str(user_id) for user_id in rows}
    result = build_item_knn_artifacts(
        args.interactions,
        args.output_root,
        load_item_knn_config(args.config_dir),
        excluded_user_ids=excluded_user_ids,
        force=args.force,
    )
    write_json_artifact(args.report, result)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
