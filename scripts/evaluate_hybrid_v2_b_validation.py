"""Build segmented and paired Validation diagnostics for Hybrid V2-B."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import load_hybrid_v2_b_config
from yelp_agent.learning_to_rank.lambdamart_diagnostics import (
    LambdaMARTDiagnosticSources,
    build_lambdamart_validation_diagnostics,
    write_lambdamart_validation_diagnostics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--logistic-predictions",
        type=Path,
        default=Path("data/features/hybrid_v2_a/validation_predictions.parquet"),
    )
    parser.add_argument(
        "--lambdamart-predictions",
        type=Path,
        default=Path("data/features/hybrid_v2_b/validation_predictions.parquet"),
    )
    parser.add_argument("--task-root", type=Path, default=Path("data/task_dataset"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/hybrid_v2_b/validation_diagnostics.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_hybrid_v2_b_config(args.config_dir)
    report = build_lambdamart_validation_diagnostics(
        LambdaMARTDiagnosticSources(
            logistic_predictions=args.logistic_predictions,
            lambdamart_predictions=args.lambdamart_predictions,
            validation_contexts=(
                args.task_root / "tasks" / "temporal_contexts.parquet"
            ),
            validation_ground_truth=(
                args.task_root / "ground_truth" / "ground_truth.parquet"
            ),
            reviews=args.processed_root / "reviews.parquet",
            interactions=args.processed_root / "interactions.parquet",
            businesses=args.processed_root / "businesses.parquet",
        ),
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
    )
    write_lambdamart_validation_diagnostics(args.output, report)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
