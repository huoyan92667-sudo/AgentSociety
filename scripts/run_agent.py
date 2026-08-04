"""Run Hybrid+Agent with atomic predictions, traces, and failure outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.agent.tools import AgentToolbox
from yelp_agent.config import load_config
from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.features.hybrid import HybridFeatureStore
from yelp_agent.features.location import TemporalLocationStore
from yelp_agent.features.quality import TemporalQualityStore
from yelp_agent.features.text import TemporalTextStore
from yelp_agent.rankers.agent_ranker import AgentRanker
from yelp_agent.rankers.agent_runner import run_agent_ranker
from yelp_agent.rankers.hybrid_ranker import HybridRanker
from yelp_agent.tuning.hybrid import (
    fingerprint_hybrid_feature_sources,
    load_validated_hybrid_weights,
)


DEFAULT_OUTPUT_DIR = Path("runs/agent/test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/task_dataset/tasks/test_tasks.jsonl"),
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="run only the first N tasks; requires an explicit output directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing Agent artifact set",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.output_dir == DEFAULT_OUTPUT_DIR:
        raise SystemExit("--limit requires an explicit --output-dir")
    config = load_config(args.config_dir)
    if (
        config.agent.top_k_to_rerank != 8
        or config.agent.history_limit != 30
        or config.agent.representative_review_count != 8
    ):
        raise ValueError(
            "MVP Agent requires top_k=8, history_limit=30, "
            "and representative_review_count=8"
        )
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
        location_store=TemporalLocationStore(
            args.businesses,
            args.interactions,
            args.histories,
            scale_km=config.hybrid.location_scale_km,
        ),
    )
    hybrid_ranker = HybridRanker(feature_store, weights)
    toolbox = AgentToolbox(
        args.businesses,
        args.interactions,
        args.histories,
        hybrid_ranker=hybrid_ranker,
        quality_store=quality_store,
    )
    agent_ranker = AgentRanker(
        hybrid_ranker=hybrid_ranker,
        toolbox=toolbox,
        llm=OpenAICompatibleLLM.from_environment(config.agent),
    )
    result = run_agent_ranker(
        args.tasks,
        agent_ranker,
        args.output_dir,
        force=args.force,
        limit=args.limit,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
