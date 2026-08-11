"""Build deterministic full-corpus Review passages for Step 27."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.review_rag import build_review_rag_artifacts, load_review_rag_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=Path("data/processed/reviews.parquet"))
    parser.add_argument("--config", type=Path, default=Path("configs/review_rag.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/features/review_rag/v1"))
    parser.add_argument("--report", type=Path, default=Path("runs/review_rag_v1/build_report.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_review_rag_artifacts(
        args.reviews,
        args.output_root,
        load_review_rag_config(args.config),
        force=args.force,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
