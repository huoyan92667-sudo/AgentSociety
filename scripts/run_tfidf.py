"""Fit if needed and run the personalized TF-IDF baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.config import load_config
from yelp_agent.features.text import TemporalTextStore, fit_tfidf_model
from yelp_agent.rankers.runner import run_ranker
from yelp_agent.rankers.tfidf_ranker import TfidfRanker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/task_dataset/tasks/test_tasks.jsonl"),
    )
    parser.add_argument(
        "--businesses",
        type=Path,
        default=Path("data/processed/businesses.parquet"),
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("data/processed/interactions.parquet"),
    )
    parser.add_argument(
        "--histories",
        type=Path,
        default=Path(
            "data/task_dataset/tasks/temporal_histories.parquet"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/features/tfidf_vectorizer.joblib"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/features/tfidf_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/tfidf/test/predictions.jsonl"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
    )
    parser.add_argument(
        "--force-fit",
        action="store_true",
        help="rebuild the vectorizer even if a matching artifact exists",
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
    fit_tfidf_model(
        args.businesses,
        args.interactions,
        args.histories,
        args.artifact,
        args.manifest,
        config.tfidf,
        force=args.force_fit,
    )
    text_store = TemporalTextStore(
        args.businesses,
        args.interactions,
        args.histories,
        args.artifact,
        args.manifest,
    )
    result = run_ranker(
        args.tasks,
        TfidfRanker(text_store),
        args.output,
        force=args.force,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
