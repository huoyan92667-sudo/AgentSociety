"""Evaluate a prediction JSONL file against isolated Yelp ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.evaluation.evaluator import (
    evaluate_prediction_file,
    write_evaluation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/task_dataset/tasks/test_tasks.jsonl"),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(
            "data/task_dataset/ground_truth/candidate_ground_truth.parquet"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("runs/predictions.jsonl"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/metrics.json"),
    )
    parser.add_argument(
        "--issues",
        type=Path,
        default=Path("runs/evaluation_issues.jsonl"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate only the first N frozen tasks; ignore other known tasks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_prediction_file(
        args.tasks,
        args.ground_truth,
        args.predictions,
        task_limit=args.limit,
    )
    write_evaluation_report(report, args.metrics, args.issues)
    print(report.metrics.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
