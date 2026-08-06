"""Compare Full Retrieval V1 with the Item-KNN V2 validation run."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import load_retrieval_config
from yelp_agent.evaluation.item_knn import compare_item_knn_retrieval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path("runs/full_retrieval_v1/validation_task_results.parquet"),
    )
    parser.add_argument(
        "--item-knn-results",
        type=Path,
        default=Path("runs/item_knn_v1/validation_task_results.parquet"),
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("data/task_dataset/tasks/temporal_contexts.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/item_knn_v1/comparison.json"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_item_knn_retrieval(
        args.baseline_results,
        args.item_knn_results,
        args.contexts,
        metric_cutoffs=load_retrieval_config(args.config_dir).metric_cutoffs,
        output_path=args.output,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
