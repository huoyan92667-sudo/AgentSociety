"""Audit full Review passages against sources and hidden silver labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.review_rag import audit_review_rag_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=Path("data/processed/reviews.parquet"))
    parser.add_argument("--segments", type=Path, default=Path("data/features/review_rag/v1/review_segments.parquet"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmarks/agent_scenarios_v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/review_rag_v1/audit_report.json"))
    args = parser.parse_args()
    report = audit_review_rag_artifacts(
        reviews_path=args.reviews,
        segments_path=args.segments,
        visible_scenarios_path=args.benchmark_root / "visible" / "scenarios.jsonl",
        evidence_labels_path=args.benchmark_root / "hidden" / "evidence_labels.parquet",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(report.model_dump_json(indent=2))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
