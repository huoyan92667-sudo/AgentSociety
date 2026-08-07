"""Build frozen train, validation, and test user profiles for Step 14."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.config import load_config, load_user_profile_config
from yelp_agent.experiments import write_json_artifact
from yelp_agent.profiles.artifacts import build_user_profile_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--businesses",
        type=Path,
        default=Path("data/processed/businesses.parquet"),
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("data/processed/interactions.parquet"),
    )
    parser.add_argument(
        "--review-aspects",
        type=Path,
        default=Path("data/features/review_aspects/development/aspect_records.parquet"),
    )
    parser.add_argument(
        "--rolling-contexts",
        type=Path,
        default=Path("data/task_dataset/training/rolling_train_contexts.parquet"),
    )
    parser.add_argument(
        "--evaluation-contexts",
        type=Path,
        default=Path("data/task_dataset/tasks/temporal_contexts.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/features/user_profiles/v1"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/user_profiles_v1/build_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = load_config(args.config_dir)
    profile_config = load_user_profile_config(args.config_dir)
    result = build_user_profile_artifacts(
        businesses_path=args.businesses,
        interactions_path=args.interactions,
        aspect_records_path=args.review_aspects,
        rolling_contexts_path=args.rolling_contexts,
        evaluation_contexts_path=args.evaluation_contexts,
        output_root=args.output_root,
        config=profile_config,
        broad_categories=set(app_config.data.broad_categories),
    )
    write_json_artifact(args.report, result)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
