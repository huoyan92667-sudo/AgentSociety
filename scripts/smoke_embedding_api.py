"""Small real-API smoke test that never prints secrets or raw vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from yelp_agent.semantic_embedding import (
    CachedEmbeddingGateway,
    DashScopeEmbeddingEncoder,
    SemanticDocument,
    SqliteEmbeddingCache,
    build_query_document,
    load_dashscope_embedding_environment,
    load_semantic_embedding_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--query",
        default="I want a quiet romantic steakhouse for a date night.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_semantic_embedding_config(root / "configs" / "embedding.yaml")
    environment = load_dashscope_embedding_environment()
    if not environment.enabled:
        raise SystemExit("DashScope embedding environment is incomplete")
    cache = SqliteEmbeddingCache(root / config.cache_relative_path)
    gateway = CachedEmbeddingGateway(
        encoder=DashScopeEmbeddingEncoder.from_environment(config, environment),
        cache=cache,
        config=config,
    )
    query = build_query_document(args.query)
    business_text = (
        "Business name: Example Steakhouse\n"
        "Categories: Restaurants, Steakhouses\n"
        "Structured attributes: ambience romantic=True; noise level=quiet"
    )
    document = SemanticDocument(
        source_id="smoke-business",
        source_kind="business_static",
        text=business_text,
        text_sha256=hashlib.sha256(business_text.encode("utf-8")).hexdigest(),
        document_version=config.business_document_version,
    )
    query_vectors, query_usage = gateway.embed([query], input_type="query")
    document_vectors, document_usage = gateway.embed(
        [document], input_type="document"
    )
    similarity = float(query_vectors[0] @ document_vectors[0])
    print(
        json.dumps(
            {
                "success": True,
                "provider": config.provider,
                "model": gateway.encoder.model,
                "dimension": gateway.encoder.dimension,
                "cosine_similarity": similarity,
                "api_calls": query_usage.api_calls + document_usage.api_calls,
                "input_tokens": query_usage.input_tokens
                + document_usage.input_tokens,
                "cache_hits": query_usage.cache_hits + document_usage.cache_hits,
                "cache_misses": query_usage.cache_misses
                + document_usage.cache_misses,
                "estimated_cost_cny": query_usage.estimated_cost_cny
                + document_usage.estimated_cost_cny,
                "cache_records": cache.count(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
