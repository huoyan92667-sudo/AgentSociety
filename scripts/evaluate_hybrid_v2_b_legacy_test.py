"""Evaluate frozen Hybrid V2-B on the previously observed Legacy Test."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import load_hybrid_v2_b_config
from yelp_agent.learning_to_rank.lambdamart_legacy import (
    LambdaMARTLegacyTestSources,
    run_lambdamart_legacy_test,
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
        "--hybrid-v2-b-data-root",
        type=Path,
        default=Path("data/features/hybrid_v2_b"),
    )
    parser.add_argument("--task-root", type=Path, default=Path("data/task_dataset"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--frozen-model-root",
        type=Path,
        default=Path("runs/hybrid_v2_b/frozen"),
    )
    parser.add_argument(
        "--hybrid-v2-a-report",
        type=Path,
        default=Path("runs/hybrid_v2_a/legacy_test_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/hybrid_v2_b/legacy_test_report.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_hybrid_v2_b_config(args.config_dir)
    report = run_lambdamart_legacy_test(
        LambdaMARTLegacyTestSources(
            test_features=args.hybrid_v2_a_data_root / "test_features.parquet",
            test_contexts=args.task_root / "tasks" / "temporal_contexts.parquet",
            test_ground_truth=(
                args.task_root / "ground_truth" / "ground_truth.parquet"
            ),
            reviews=args.processed_root / "reviews.parquet",
            interactions=args.processed_root / "interactions.parquet",
            frozen_model_root=args.frozen_model_root,
            hybrid_v2_a_legacy_report=args.hybrid_v2_a_report,
        ),
        score_output_path=args.hybrid_v2_b_data_root / "test_scores.parquet",
        prediction_output_path=(
            args.hybrid_v2_b_data_root / "test_predictions.parquet"
        ),
        report_output_path=args.output,
        prediction_batch_size=config.prediction_batch_size,
    )
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
