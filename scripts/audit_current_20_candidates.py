"""Audit the frozen current 20-candidate reranking benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yelp_agent.config import load_config
from yelp_agent.evaluation.candidate_audit import (
    audit_20_candidate_benchmark,
    write_candidate_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-tasks",
        type=Path,
        default=Path("data/task_dataset/tasks/validation_tasks.jsonl"),
    )
    parser.add_argument(
        "--legacy-test-tasks",
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
        "--provenance",
        type=Path,
        default=Path("data/task_dataset/audit/candidate_provenance.parquet"),
    )
    parser.add_argument(
        "--dropped-tasks",
        type=Path,
        default=Path("data/task_dataset/audit/dropped_tasks.parquet"),
    )
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
        "--interactions",
        type=Path,
        default=Path("data/processed/interactions.parquet"),
    )
    parser.add_argument(
        "--histories",
        type=Path,
        default=Path("data/task_dataset/tasks/temporal_histories.parquet"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/evaluation/current_20_candidate_audit.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_20_candidate_benchmark(
        validation_tasks_path=args.validation_tasks,
        test_tasks_path=args.legacy_test_tasks,
        ground_truth_path=args.ground_truth,
        provenance_path=args.provenance,
        dropped_tasks_path=args.dropped_tasks,
        businesses_path=args.businesses,
        reviews_path=args.reviews,
        interactions_path=args.interactions,
        histories_path=args.histories,
        config=load_config(args.config_dir).data,
    )
    write_candidate_audit(report, args.output)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
