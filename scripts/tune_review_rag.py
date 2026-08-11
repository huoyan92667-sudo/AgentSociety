"""Tune Review RAG fusion strictly on the 104 Development evidence scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.config import load_review_aspect_settings
from yelp_agent.review_rag import (
    ReviewRAGStore,
    ReviewRetriever,
    load_review_rag_config,
    tune_review_rag_policy,
)
from yelp_agent.semantic_embedding import (
    CachedEmbeddingGateway,
    LocalQwenEmbeddingEncoder,
    SqliteEmbeddingCache,
)
from yelp_agent.semantic_embedding.config import LocalEmbeddingEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_review_rag_config(root / "configs" / "review_rag.yaml")
    semantic = config.semantic_config()
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        semantic,
        LocalEmbeddingEnvironment(
            model_path=args.model_path,
            python_executable=args.model_python,
            device=args.device,
        ),
    )
    store = ReviewRAGStore(
        root / "data" / "features" / "review_rag" / "v1" / "review_segments.parquet",
        root / "data" / "features" / "review_aspects" / "development" / "aspect_records.parquet",
        config,
    )
    _, vocabulary = load_review_aspect_settings(root / "configs")
    gateway = CachedEmbeddingGateway(
        encoder=encoder,
        cache=SqliteEmbeddingCache(root / config.embedding_cache_relative_path),
        config=semantic,
    )
    try:
        result = tune_review_rag_policy(
            benchmark_root=root / "benchmarks" / "agent_scenarios_v1",
            output_root=root / "runs" / "review_rag_v1",
            policy_path=root / config.policy_relative_path,
            retriever_factory=lambda policy: ReviewRetriever(
                store=store,
                config=config,
                policy=policy,
                vocabulary=vocabulary,
                embedding_gateway=gateway,
            ),
            progress=lambda index, total, scenario: print(
                f"development {index}/{total} {scenario.scenario_id[:8]}",
                flush=True,
            ) if index % 10 == 0 or index == total else None,
        )
        print(f"selected_policy={result.selected_policy.model_dump_json()}")
        print(f"report={result.report_path}")
    finally:
        store.close()
        encoder.close()


if __name__ == "__main__":
    main()
