"""Build target-blind multi-route candidates for train and validation tasks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import load_config, load_retrieval_config
from yelp_agent.experiments.artifacts import write_json_artifact
from yelp_agent.retrieval.benchmark import (
    RetrievalSourcePaths,
    build_full_retrieval_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
    )
    parser.add_argument(
        "--train-contexts",
        type=Path,
        default=Path(
            "data/task_dataset/training/rolling_train_contexts.parquet"
        ),
    )
    parser.add_argument(
        "--temporal-contexts",
        type=Path,
        default=Path("data/task_dataset/tasks/temporal_contexts.parquet"),
    )
    parser.add_argument(
        "--businesses",
        type=Path,
        default=Path("data/processed/businesses.parquet"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/processed/reviews.parquet"),
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("data/processed/interactions.parquet"),
    )
    parser.add_argument(
        "--tfidf-artifact",
        type=Path,
        default=Path("data/features/tfidf_vectorizer.joblib"),
    )
    parser.add_argument(
        "--tfidf-manifest",
        type=Path,
        default=Path("data/features/tfidf_manifest.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/task_dataset/full_retrieval_v1"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/full_retrieval_v1/build_report.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split_contexts = {
        split: args.train_contexts if split == "train" else args.temporal_contexts
        for split in sorted(set(args.splits))
    }
    result = build_full_retrieval_benchmark(
        RetrievalSourcePaths(
            businesses=args.businesses,
            reviews=args.reviews,
            interactions=args.interactions,
            tfidf_artifact=args.tfidf_artifact,
            tfidf_manifest=args.tfidf_manifest,
        ),
        split_contexts,
        args.output_root,
        load_config(args.config_dir),
        load_retrieval_config(args.config_dir),
        force=args.force,
    )
    write_json_artifact(args.report, result)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
