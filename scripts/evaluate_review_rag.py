"""Evaluate the frozen Review RAG policy on one split without retuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.config import load_review_aspect_settings
from yelp_agent.review_rag import (
    ReviewRAGStore,
    ReviewRetriever,
    evaluate_frozen_review_retriever,
    load_review_rag_config,
    load_review_rag_policy,
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
    parser.add_argument("--split", choices=("development", "validation"), default="validation")
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
    try:
        report = evaluate_frozen_review_retriever(
            ReviewRetriever(
                store=store,
                config=config,
                policy=load_review_rag_policy(root, config),
                vocabulary=vocabulary,
                embedding_gateway=CachedEmbeddingGateway(
                    encoder=encoder,
                    cache=SqliteEmbeddingCache(root / config.embedding_cache_relative_path),
                    config=semantic,
                ),
            ),
            benchmark_root=root / "benchmarks" / "agent_scenarios_v1",
            split=args.split,
            output_root=root / "runs" / "review_rag_v1",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        store.close()
        encoder.close()


if __name__ == "__main__":
    main()
