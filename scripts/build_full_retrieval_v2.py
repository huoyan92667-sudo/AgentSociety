"""Build Full Retrieval V2 with the validation-frozen Item-KNN route."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import (
    ItemKNNConfig,
    load_config,
    load_retrieval_config,
)
from yelp_agent.experiments import write_json_artifact
from yelp_agent.retrieval.benchmark import (
    RetrievalSourcePaths,
    build_full_retrieval_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument(
        "--train-contexts",
        type=Path,
        default=Path("data/task_dataset/training/rolling_train_contexts.parquet"),
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
        "--item-knn-root",
        type=Path,
        default=Path("data/features/item_knn"),
    )
    parser.add_argument(
        "--selected-config",
        type=Path,
        default=Path("data/features/item_knn/tuning/selected_config.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/task_dataset/full_retrieval_v2_item_knn"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/item_knn_v1/full_retrieval_build_report.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_config = ItemKNNConfig.model_validate_json(
        args.selected_config.read_text(encoding="utf-8")
    )
    split_contexts = {
        split: (args.train_contexts if split == "train" else args.temporal_contexts)
        for split in sorted(set(args.splits))
    }
    result = build_full_retrieval_benchmark(
        RetrievalSourcePaths(
            businesses=args.businesses,
            reviews=args.reviews,
            interactions=args.interactions,
            tfidf_artifact=args.tfidf_artifact,
            tfidf_manifest=args.tfidf_manifest,
            item_knn_positive_events=(args.item_knn_root / "positive_events.parquet"),
            item_knn_negative_events=(args.item_knn_root / "negative_events.parquet"),
            item_knn_neutral_events=(args.item_knn_root / "neutral_events.parquet"),
            item_knn_manifest=args.item_knn_root / "manifest.json",
        ),
        split_contexts,
        args.output_root,
        load_config(args.config_dir),
        load_retrieval_config(args.config_dir),
        item_knn_config=selected_config,
        force=args.force,
    )
    write_json_artifact(args.report, result)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
