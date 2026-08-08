"""Run post-freeze Hybrid V2-A on the previously observed Legacy Test."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from yelp_agent.config import (
    ItemKNNConfig,
    load_business_profile_config,
    load_config,
    load_hybrid_v2_config,
    load_retrieval_config,
)
from yelp_agent.learning_to_rank.legacy_test import (
    HybridV2LegacyTestSources,
    run_hybrid_v2_legacy_test,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--task-root", type=Path, default=Path("data/task_dataset"))
    parser.add_argument(
        "--item-knn-root",
        type=Path,
        default=Path("data/features/item_knn"),
    )
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
        "--frozen-model-root",
        type=Path,
        default=Path("runs/hybrid_v2_a/frozen"),
    )
    parser.add_argument(
        "--retrieval-output-root",
        type=Path,
        default=Path("data/task_dataset/full_retrieval_v2_item_knn_legacy_test"),
    )
    parser.add_argument(
        "--feature-output",
        type=Path,
        default=Path("data/features/hybrid_v2_a/test_features.parquet"),
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path("data/features/hybrid_v2_a/test_predictions.parquet"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/hybrid_v2_a/legacy_test_report.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_item_knn = ItemKNNConfig.model_validate_json(
        (args.item_knn_root / "tuning" / "selected_config.json").read_text(
            encoding="utf-8"
        )
    )
    result = run_hybrid_v2_legacy_test(
        HybridV2LegacyTestSources(
            contexts=args.task_root / "tasks" / "temporal_contexts.parquet",
            ground_truth=(args.task_root / "ground_truth" / "ground_truth.parquet"),
            businesses=Path("data/processed/businesses.parquet"),
            reviews=Path("data/processed/reviews.parquet"),
            interactions=Path("data/processed/interactions.parquet"),
            tfidf_artifact=Path("data/features/tfidf_vectorizer.joblib"),
            tfidf_manifest=Path("data/features/tfidf_manifest.json"),
            item_knn_root=args.item_knn_root,
            user_profile_root=args.user_profile_root,
            business_profile_root=args.business_profile_root,
            hybrid_v1_weights=Path("runs/hybrid/hybrid_weights.json"),
            frozen_model_root=args.frozen_model_root,
        ),
        retrieval_output_root=args.retrieval_output_root,
        feature_output_path=args.feature_output,
        prediction_output_path=args.prediction_output,
        report_path=args.report,
        app_config=load_config(args.config_dir),
        retrieval_config=load_retrieval_config(args.config_dir),
        item_knn_config=selected_item_knn,
        model_config=load_hybrid_v2_config(args.config_dir),
        business_profile_config=load_business_profile_config(args.config_dir),
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
