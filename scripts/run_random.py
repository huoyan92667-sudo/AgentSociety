"""Run the deterministic Random baseline on frozen recommendation tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.rankers.random_ranker import RandomRanker
from yelp_agent.rankers.runner import run_ranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/task_dataset/tasks/test_tasks.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/random/test/predictions.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing valid prediction file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_ranker(
        args.tasks,
        RandomRanker(seed=args.seed),
        args.output,
        force=args.force,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
