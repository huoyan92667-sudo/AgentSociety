"""Diagnose frozen Hybrid V1 on validation tasks only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.config import load_config
from yelp_agent.evaluation.hybrid_diagnostics import (
    diagnose_hybrid_v1_validation,
    write_hybrid_diagnosis,
)
from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.features.hybrid import HybridFeatureStore
from yelp_agent.features.location import TemporalLocationStore
from yelp_agent.features.quality import TemporalQualityStore
from yelp_agent.features.text import TemporalTextStore
from yelp_agent.tuning.hybrid import (
    fingerprint_hybrid_feature_sources,
    load_validated_hybrid_weights,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-tasks",
        type=Path,
        default=Path("data/task_dataset/tasks/validation_tasks.jsonl"),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(
            "data/task_dataset/ground_truth/candidate_ground_truth.parquet"
        ),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/task_dataset/audit/candidate_provenance.parquet"),
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
        "--histories",
        type=Path,
        default=Path("data/task_dataset/tasks/temporal_histories.parquet"),
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
        "--weights",
        type=Path,
        default=Path("runs/hybrid/hybrid_weights.json"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--task-diagnostics",
        type=Path,
        default=Path(
            "runs/hybrid_v1_diagnosis/validation_task_diagnostics.jsonl"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "docs/evaluation/hybrid_v1_validation_diagnosis_summary.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "docs/evaluation/hybrid_v1_validation_diagnosis.md"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config_dir)
    feature_sources_sha256 = fingerprint_hybrid_feature_sources(
        {
            "businesses": args.businesses,
            "reviews": args.reviews,
            "interactions": args.interactions,
            "histories": args.histories,
            "tfidf_artifact": args.tfidf_artifact,
            "tfidf_manifest": args.tfidf_manifest,
            "data_config": args.config_dir / "data.yaml",
            "hybrid_config": args.config_dir / "hybrid.yaml",
            "tfidf_config": args.config_dir / "tfidf.yaml",
        }
    )
    weights = load_validated_hybrid_weights(
        args.weights,
        feature_sources_sha256=feature_sources_sha256,
    )
    quality_store = TemporalQualityStore(
        args.reviews,
        prior_count=config.hybrid.bayesian_prior_count,
    )
    location_store = TemporalLocationStore(
        args.businesses,
        args.interactions,
        args.histories,
        scale_km=config.hybrid.location_scale_km,
    )
    feature_store = HybridFeatureStore(
        category_store=TemporalCategoryStore(
            args.businesses,
            args.interactions,
            args.histories,
            broad_categories=set(config.data.broad_categories),
        ),
        text_store=TemporalTextStore(
            args.businesses,
            args.interactions,
            args.histories,
            args.tfidf_artifact,
            args.tfidf_manifest,
        ),
        quality_store=quality_store,
        location_store=location_store,
    )
    diagnostics, summary = diagnose_hybrid_v1_validation(
        validation_tasks_path=args.validation_tasks,
        ground_truth_path=args.ground_truth,
        provenance_path=args.provenance,
        feature_store=feature_store,
        quality_store=quality_store,
        location_store=location_store,
        weights=weights,
        policy=config.evaluation_data_usage,
    )
    write_hybrid_diagnosis(
        diagnostics,
        summary,
        task_diagnostics_path=args.task_diagnostics,
        summary_path=args.summary,
        markdown_path=args.report,
    )
    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
