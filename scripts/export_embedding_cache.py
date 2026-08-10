"""Export the local SQLite embedding cache to a portable Parquet artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.semantic_embedding import (
    SqliteEmbeddingCache,
    load_semantic_embedding_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_semantic_embedding_config(root / "configs" / "embedding.yaml")
    cache = SqliteEmbeddingCache(root / config.cache_relative_path)
    output = args.output or root / config.cache_relative_path / "embedding_cache.parquet"
    path = cache.export_parquet(output)
    print(f"records={cache.count()}")
    print(f"parquet={path}")


if __name__ == "__main__":
    main()
