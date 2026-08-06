"""Select the Item-KNN half-life on validation-only fused Recall."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.collaborative.tuning import (
    ItemKNNTuningSources,
    tune_item_knn,
)
from yelp_agent.config import load_item_knn_config, load_retrieval_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--contexts",
        type=Path,
        default=Path("data/task_dataset/tasks/temporal_contexts.parquet"),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/task_dataset/ground_truth/ground_truth.parquet"),
    )
    parser.add_argument(
        "--base-route-provenance",
        type=Path,
        default=Path(
            "data/task_dataset/full_retrieval_v1/validation_route_provenance.parquet"
        ),
    )
    parser.add_argument(
        "--item-knn-root",
        type=Path,
        default=Path("data/features/item_knn"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/features/item_knn/tuning"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = ItemKNNTuningSources(
        businesses=args.businesses,
        reviews=args.reviews,
        interactions=args.interactions,
        contexts=args.contexts,
        ground_truth=args.ground_truth,
        base_route_provenance=args.base_route_provenance,
        positive_events=args.item_knn_root / "positive_events.parquet",
        negative_events=args.item_knn_root / "negative_events.parquet",
        neutral_events=args.item_knn_root / "neutral_events.parquet",
    )
    result = tune_item_knn(
        sources,
        args.output_root,
        load_item_knn_config(args.config_dir),
        load_retrieval_config(args.config_dir),
        progress=lambda message: print(message, flush=True),
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
