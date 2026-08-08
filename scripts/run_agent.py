"""Run Hybrid+Agent with atomic predictions, traces, and failure outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.agent.tools import AgentToolbox
from yelp_agent.business_profiles.store import BusinessKnowledgeStore
from yelp_agent.config import load_business_profile_config, load_config
from yelp_agent.profiles.store import UserProfileStore
from yelp_agent.rankers.agent_ranker import AgentRanker
from yelp_agent.rankers.agent_runner import run_agent_ranker
from yelp_agent.ranking.assembly import (
    HybridSourcePaths,
    build_frozen_hybrid_runtime,
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
        "--profiles",
        type=Path,
        default=Path("data/features/user_profiles/v1"),
    )
    parser.add_argument(
        "--business-profiles",
        type=Path,
        default=Path("data/features/business_profiles/v1"),
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
    business_profile_config = load_business_profile_config(args.config_dir)
    runtime = build_frozen_hybrid_runtime(
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
        args.weights,
    )
    with UserProfileStore(args.profiles) as profile_store:
        business_profile_store = BusinessKnowledgeStore.from_artifacts(
            args.business_profiles,
            config=business_profile_config,
        )
        toolbox = AgentToolbox(
            runtime.assembly.data_view,
            hybrid_ranker=runtime.ranker,
            quality_store=runtime.assembly.quality_store,
            profile_store=profile_store,
            business_profile_store=business_profile_store,
        )
        agent_ranker = AgentRanker(
            hybrid_ranker=runtime.ranker,
            toolbox=toolbox,
            llm=OpenAICompatibleLLM.from_environment(config.agent),
        )
        result = run_agent_ranker(
            args.tasks,
            agent_ranker,
            args.output_dir,
            force=args.force,
            limit=args.limit,
            configuration=config,
        )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
