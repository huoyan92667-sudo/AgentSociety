"""Train, select, and freeze Hybrid V2-B on Train and Validation only."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import load_hybrid_v2_b_config
from yelp_agent.learning_to_rank.lambdamart_experiment import (
    HybridV2BExperimentSources,
    run_hybrid_v2_b_validation_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--hybrid-v2-a-data-root",
        type=Path,
        default=Path("data/features/hybrid_v2_a"),
    )
    parser.add_argument(
        "--hybrid-v2-a-report",
        type=Path,
        default=Path("runs/hybrid_v2_a/validation_report.json"),
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path("data/task_dataset"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/features/hybrid_v2_b"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/hybrid_v2_b"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_hybrid_v2_b_validation_experiment(
        HybridV2BExperimentSources(
            train_features=args.hybrid_v2_a_data_root / "train_features.parquet",
            validation_features=(
                args.hybrid_v2_a_data_root / "validation_features.parquet"
            ),
            validation_contexts=(
                args.task_root / "tasks" / "temporal_contexts.parquet"
            ),
            validation_ground_truth=(
                args.task_root / "ground_truth" / "ground_truth.parquet"
            ),
            reviews=args.processed_root / "reviews.parquet",
            interactions=args.processed_root / "interactions.parquet",
            hybrid_v2_a_validation_report=args.hybrid_v2_a_report,
        ),
        load_hybrid_v2_b_config(args.config_dir),
        data_root=args.data_root,
        run_root=args.run_root,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
