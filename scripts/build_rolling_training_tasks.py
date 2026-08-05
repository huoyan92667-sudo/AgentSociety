"""Build rolling temporal train examples without touching legacy candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.config import load_config, load_rolling_training_config
from yelp_agent.data.rolling_training import (
    build_rolling_training_tasks,
    write_rolling_training_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("data/processed/interactions.parquet"),
    )
    parser.add_argument(
        "--frozen-contexts",
        type=Path,
        default=Path(
            "data/task_dataset/tasks/temporal_contexts.parquet"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/task_dataset"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/rolling_training_build_report.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild the complete rolling training artifact set",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_config = load_config(args.config_dir)
    training_config = load_rolling_training_config(args.config_dir)
    result = build_rolling_training_tasks(
        args.interactions,
        args.frozen_contexts,
        args.output_root,
        training_config,
        app_config.evaluation_data_usage,
        force=args.force,
    )
    write_rolling_training_report(result, args.report)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
