"""Tune Hybrid weights on validation tasks and freeze one weight artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.config import load_config
from yelp_agent.experiments import write_resolved_configuration
from yelp_agent.features.hybrid import HybridWeights
from yelp_agent.features.text import fit_tfidf_model
from yelp_agent.ranking.assembly import (
    HybridSourcePaths,
    build_hybrid_assembly,
)
from yelp_agent.tuning.hybrid import tune_hybrid_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
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
        default=Path(
            "data/task_dataset/tasks/temporal_histories.parquet"
        ),
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
        "--output",
        type=Path,
        default=Path("runs/hybrid/hybrid_weights.json"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
    )
    parser.add_argument(
        "--force-tfidf",
        action="store_true",
        help="rebuild the TF-IDF artifact before tuning",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing matching frozen weight artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config_dir)
    fit_tfidf_model(
        args.businesses,
        args.interactions,
        args.histories,
        args.tfidf_artifact,
        args.tfidf_manifest,
        config.tfidf,
        force=args.force_tfidf,
    )
    assembly = build_hybrid_assembly(
        config,
        HybridSourcePaths(
            businesses=args.businesses,
            reviews=args.reviews,
            interactions=args.interactions,
            histories=args.histories,
            tfidf_artifact=args.tfidf_artifact,
            tfidf_manifest=args.tfidf_manifest,
            config_dir=args.config_dir,
        ),
    )
    result = tune_hybrid_weights(
        args.tasks,
        args.ground_truth,
        assembly.feature_store,
        args.output,
        initial_weights=HybridWeights.from_config(config.hybrid),
        feature_sources_sha256=assembly.feature_sources_sha256,
        step=config.hybrid.tuning_step,
        force=args.force,
    )
    write_resolved_configuration(
        args.output.parent / "resolved_config.json",
        config,
        force=args.force,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
