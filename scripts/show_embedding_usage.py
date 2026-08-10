"""Print exact cumulative semantic-embedding inference usage."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from yelp_agent.semantic_embedding import SqliteEmbeddingCache, load_semantic_embedding_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--usage-scope", default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_semantic_embedding_config(root / "configs" / "embedding.yaml")
    cache = SqliteEmbeddingCache(root / config.cache_relative_path)
    payload = {
        "provider": config.provider,
        "cache_records": cache.count(),
        "usage": cache.usage_summary(usage_scope=args.usage_scope).model_dump(),
    }
    if args.usage_scope is None:
        legacy_path = (
            root
            / "data"
            / "features"
            / "semantic_embeddings"
            / "v1"
            / "embedding_cache.sqlite3"
        )
        if legacy_path.is_file():
            with sqlite3.connect(f"file:{legacy_path}?mode=ro", uri=True) as connection:
                legacy_row = connection.execute(
                    "SELECT count(*), coalesce(sum(input_tokens), 0) FROM embeddings"
                ).fetchone()
            legacy_tokens = int(legacy_row[1])
            payload["historical_remote_v1_cache"] = {
                "cache_records": int(legacy_row[0]),
                "cached_input_token_attribution": legacy_tokens,
                "exact_provider_billing_total": False,
            }
            payload["combined_observed_input_tokens"] = (
                legacy_tokens + int(payload["usage"]["encoded_input_tokens"])
            )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
