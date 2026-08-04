"""Summarize deterministic validation user folds without publishing IDs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.config import load_config
from yelp_agent.evaluation.data_usage import (
    build_validation_user_fold_summary,
    write_user_fold_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-tasks",
        type=Path,
        default=Path("data/task_dataset/tasks/validation_tasks.jsonl"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/evaluation/current_validation_user_fold_summary.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = load_config(args.config_dir).evaluation_data_usage
    summary = build_validation_user_fold_summary(
        args.validation_tasks,
        policy,
    )
    write_user_fold_summary(summary, args.output)
    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
