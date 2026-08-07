"""Build the frozen rule-based Review Aspect V1 artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.config import load_review_aspect_settings
from yelp_agent.experiments import write_json_artifact
from yelp_agent.reviews.artifacts import build_review_aspect_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/processed/interactions.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/features/review_aspects/development"),
    )
    parser.add_argument(
        "--source-scope",
        choices=(
            "selected_user_interactions",
            "full_business_reviews",
            "custom",
        ),
        default="selected_user_interactions",
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/review_aspect_v1/build_report.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, vocabulary = load_review_aspect_settings(args.config_dir)
    result = build_review_aspect_artifacts(
        args.reviews,
        args.output_root,
        config,
        vocabulary,
        source_scope=args.source_scope,
        force=args.force,
    )
    write_json_artifact(args.report, result)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
