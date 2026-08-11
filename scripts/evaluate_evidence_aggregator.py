"""Evaluate one Development-frozen Step 28 policy without retuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.config import load_review_aspect_settings
from yelp_agent.evidence_aggregation import (
    EvidenceAggregator,
    evaluate_frozen_evidence_aggregator,
    load_evidence_aggregation_config,
    load_evidence_aggregation_policy,
)
from yelp_agent.review_rag import (
    ReviewRAGStore,
    ReviewRetriever,
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
    rag_config = load_review_rag_config(root / "configs" / "review_rag.yaml")
    evidence_config = load_evidence_aggregation_config(
        root / "configs" / "evidence_aggregator.yaml"
    )
    semantic = rag_config.semantic_config()
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
        rag_config,
    )
    _, vocabulary = load_review_aspect_settings(root / "configs")
    try:
        retriever = ReviewRetriever(
            store=store,
            config=rag_config,
            policy=load_review_rag_policy(root, rag_config),
            vocabulary=vocabulary,
            embedding_gateway=CachedEmbeddingGateway(
                encoder=encoder,
                cache=SqliteEmbeddingCache(root / rag_config.embedding_cache_relative_path),
                config=semantic,
            ),
        )
        report = evaluate_frozen_evidence_aggregator(
            EvidenceAggregator(
                load_evidence_aggregation_policy(root, evidence_config)
            ),
            retriever,
            benchmark_root=root / "benchmarks" / "agent_scenarios_v1",
            split=args.split,
            output_root=root / "runs" / "evidence_aggregator_v1",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        store.close()
        encoder.close()


if __name__ == "__main__":
    main()
