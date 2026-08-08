"""Build compact point-in-time business knowledge for Step 15."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.business_profiles.artifacts import (
    build_business_knowledge_artifacts,
)
from yelp_agent.config import load_business_profile_config
from yelp_agent.experiments import write_json_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--businesses",
        type=Path,
        default=Path("data/processed/businesses.parquet"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/processed/reviews.parquet"),
    )
    parser.add_argument(
        "--review-aspects",
        type=Path,
        default=Path("data/features/review_aspects/development/aspect_records.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/features/business_profiles/v1"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/business_profiles_v1/build_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_business_knowledge_artifacts(
        businesses_path=args.businesses,
        reviews_path=args.reviews,
        aspect_records_path=args.review_aspects,
        output_root=args.output_root,
        config=load_business_profile_config(args.config_dir),
    )
    write_json_artifact(args.report, result)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
