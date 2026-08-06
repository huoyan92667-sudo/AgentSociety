"""Evaluate frozen full-retrieval validation candidates with isolated labels."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import load_retrieval_config
from yelp_agent.evaluation.retrieval import evaluate_full_retrieval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
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
        "--candidates",
        type=Path,
        default=Path(
            "data/task_dataset/full_retrieval_v1/validation_candidates.parquet"
        ),
    )
    parser.add_argument(
        "--task-audit",
        type=Path,
        default=Path(
            "data/task_dataset/full_retrieval_v1/validation_task_audit.parquet"
        ),
    )
    parser.add_argument(
        "--route-provenance",
        type=Path,
        default=Path(
            "data/task_dataset/full_retrieval_v1/validation_route_provenance.parquet"
        ),
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
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/full_retrieval_v1/validation_metrics.json"),
    )
    parser.add_argument(
        "--task-results",
        type=Path,
        default=Path("runs/full_retrieval_v1/validation_task_results.parquet"),
    )
    parser.add_argument(
        "--benchmark-name",
        choices=(
            "Full Retrieval Benchmark V1",
            "Full Retrieval Benchmark V2 + Item-KNN",
        ),
        default="Full Retrieval Benchmark V1",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_retrieval_config(args.config_dir)
    metrics = evaluate_full_retrieval(
        split=args.split,
        contexts_path=args.contexts,
        ground_truth_path=args.ground_truth,
        candidates_path=args.candidates,
        task_audit_path=args.task_audit,
        reviews_path=args.reviews,
        interactions_path=args.interactions,
        metric_cutoffs=config.metric_cutoffs,
        route_provenance_path=args.route_provenance,
        metrics_output_path=args.metrics,
        task_results_output_path=args.task_results,
        benchmark_name=args.benchmark_name,
    )
    print(metrics.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
