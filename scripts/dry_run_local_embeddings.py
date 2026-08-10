"""Count all static-business tokens locally without generating vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.semantic_embedding import (
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
    load_semantic_embedding_config,
)
from yelp_agent.semantic_embedding.precompute import (
    estimate_static_embedding_tokens,
    load_static_business_records,
)


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
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        config,
        LocalEmbeddingEnvironment(
            model_path=args.model_path,
            python_executable=args.python,
            device=args.device,
        ),
    )
    try:
        result = estimate_static_embedding_tokens(
            load_static_business_records(root / "data" / "processed" / "businesses.parquet"),
            encoder=encoder,
            config=config,
        )
    finally:
        encoder.close()
    output = args.output or root / "runs" / "semantic_embedding_v2" / "dry_run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    partial.replace(output)
    print(json.dumps(result.model_dump(), ensure_ascii=False, sort_keys=True))
    print(f"output={output}")


if __name__ == "__main__":
    main()
