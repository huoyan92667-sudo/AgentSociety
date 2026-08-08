"""Train, validate, and freeze Hybrid V2-A without reading Legacy Test."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import (
    load_business_profile_config,
    load_config,
    load_hybrid_v2_config,
)
from yelp_agent.learning_to_rank.experiment import (
    HybridV2ExperimentSources,
    run_hybrid_v2_validation_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--retrieval-root",
        type=Path,
        default=Path("data/task_dataset/full_retrieval_v2_item_knn"),
    )
    parser.add_argument("--task-root", type=Path, default=Path("data/task_dataset"))
    parser.add_argument(
        "--user-profile-root",
        type=Path,
        default=Path("data/features/user_profiles/v1"),
    )
    parser.add_argument(
        "--business-profile-root",
        type=Path,
        default=Path("data/features/business_profiles/v1"),
    )
    parser.add_argument(
        "--hybrid-v1-weights",
        type=Path,
        default=Path("runs/hybrid/hybrid_weights.json"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/features/hybrid_v2_a"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("runs/hybrid_v2_a/frozen"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/hybrid_v2_a/validation_report.json"),
    )
    parser.add_argument("--reuse-prepared", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_config = load_config(args.config_dir)
    result = run_hybrid_v2_validation_experiment(
        HybridV2ExperimentSources(
            train_candidates=args.retrieval_root / "train_candidates.parquet",
            validation_candidates=(
                args.retrieval_root / "validation_candidates.parquet"
            ),
            train_ground_truth=(
                args.task_root / "ground_truth" / "rolling_train_ground_truth.parquet"
            ),
            validation_contexts=(
                args.task_root / "tasks" / "temporal_contexts.parquet"
            ),
            validation_ground_truth=(
                args.task_root / "ground_truth" / "ground_truth.parquet"
            ),
            user_profile_root=args.user_profile_root,
            business_profile_root=args.business_profile_root,
            reviews=Path("data/processed/reviews.parquet"),
            interactions=Path("data/processed/interactions.parquet"),
            hybrid_v1_weights=args.hybrid_v1_weights,
            retrieval_manifest=args.retrieval_root / "manifest.json",
            user_profile_manifest=args.user_profile_root / "manifest.json",
            business_profile_manifest=args.business_profile_root / "manifest.json",
        ),
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        report_path=args.report,
        model_config=load_hybrid_v2_config(args.config_dir),
        business_profile_config=load_business_profile_config(args.config_dir),
        broad_categories=set(app_config.data.broad_categories),
        reuse_prepared=args.reuse_prepared,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
