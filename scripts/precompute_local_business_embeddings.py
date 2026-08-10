"""Generate and cache every static Philadelphia business embedding locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.semantic_embedding import (
    CachedEmbeddingGateway,
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
    SqliteEmbeddingCache,
    build_business_document,
    load_semantic_embedding_config,
)
from yelp_agent.semantic_embedding.precompute import load_static_business_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = load_semantic_embedding_config(root / "configs" / "embedding.yaml")
    records = load_static_business_records(
        root / "data" / "processed" / "businesses.parquet"
    )
    documents = [
        build_business_document(
            record,
            document_version=config.business_document_version,
        )
        for record in records
    ]
    cache = SqliteEmbeddingCache(root / config.cache_relative_path)
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        config,
        LocalEmbeddingEnvironment(
            model_path=args.model_path,
            python_executable=args.python,
            device=args.device,
        ),
    )
    try:
        _, usage = CachedEmbeddingGateway(
            encoder=encoder,
            cache=cache,
            config=config,
        ).embed(
            documents,
            input_type="document",
            usage_scope="static-business-precompute-v2",
        )
    finally:
        encoder.close()

    report = {
        "business_count": len(documents),
        "provider": encoder.provider,
        "model": encoder.model,
        "dimension": encoder.dimension,
        "cache_record_count": cache.count(),
        "run_usage": usage.model_dump(),
        "cumulative_usage": cache.usage_summary().model_dump(),
    }
    output = args.output or (
        root / "runs" / "semantic_embedding_v2" / "precompute.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    partial.replace(output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print(f"output={output}")


if __name__ == "__main__":
    main()
