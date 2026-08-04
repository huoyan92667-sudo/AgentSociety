"""Run the point-in-time Popularity baseline on frozen tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.config import load_config
from yelp_agent.features.quality import TemporalQualityStore
from yelp_agent.rankers.popularity_ranker import PopularityRanker
from yelp_agent.rankers.runner import run_ranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/task_dataset/tasks/test_tasks.jsonl"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/processed/reviews.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/popularity/test/predictions.jsonl"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing valid prediction file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config_dir)
    quality_store = TemporalQualityStore(
        args.reviews,
        prior_count=config.hybrid.bayesian_prior_count,
    )
    result = run_ranker(
        args.tasks,
        PopularityRanker(quality_store),
        args.output,
        force=args.force,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
