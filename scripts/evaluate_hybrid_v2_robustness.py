"""Audit frozen Hybrid V2-A with user folds and paired bootstrap."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import load_config
from yelp_agent.learning_to_rank.robustness import (
    HybridV2RobustnessSources,
    run_hybrid_v2_validation_robustness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/features/hybrid_v2_a"),
    )
    parser.add_argument("--task-root", type=Path, default=Path("data/task_dataset"))
    parser.add_argument(
        "--frozen-model-root",
        type=Path,
        default=Path("runs/hybrid_v2_a/frozen"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/hybrid_v2_a/validation_robustness_report.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_config = load_config(args.config_dir)
    result = run_hybrid_v2_validation_robustness(
        HybridV2RobustnessSources(
            train_features=args.data_root / "train_features.parquet",
            validation_features=args.data_root / "validation_features.parquet",
            validation_predictions=(args.data_root / "validation_predictions.parquet"),
            validation_contexts=(
                args.task_root / "tasks" / "temporal_contexts.parquet"
            ),
            validation_ground_truth=(
                args.task_root / "ground_truth" / "ground_truth.parquet"
            ),
            reviews=Path("data/processed/reviews.parquet"),
            interactions=Path("data/processed/interactions.parquet"),
            frozen_model_root=args.frozen_model_root,
        ),
        fold_output_root=args.data_root / "validation_folds",
        report_path=args.report,
        policy=app_config.evaluation_data_usage,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
