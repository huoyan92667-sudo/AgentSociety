"""运行首个全量评论召回试验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import SingleCaseBenchmarkConfig, run_single_case_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-project-root",
        type=Path,
        default=Path(r"C:\Users\29072\PycharmProjects\AgentSociety"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-reviews-per-batch", type=int, default=50)
    parser.add_argument("--max-chars-per-batch", type=int, default=60_000)
    parser.add_argument("--annotation-concurrency", type=int, default=4)
    args = parser.parse_args()
    report = run_single_case_benchmark(
        SingleCaseBenchmarkConfig(
            source_project_root=args.source_project_root.resolve(),
            output_root=(
                None if args.output_root is None else args.output_root.resolve()
            ),
            max_reviews_per_batch=args.max_reviews_per_batch,
            max_chars_per_batch=args.max_chars_per_batch,
            annotation_concurrency=args.annotation_concurrency,
        )
    )
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
