"""Rescore saved multi-turn Agent traces against their actual visible context."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.session_memory_benchmark.contextual_evaluation import (
    evaluate_context_grounded_replay_v3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_project_root.resolve()
    result = evaluate_context_grounded_replay_v3(
        runs_path=args.runs,
        benchmark_root=args.benchmark_root,
        businesses_path=source / "data" / "processed" / "businesses.parquet",
        profile_snapshots_path=(
            source / "data" / "features" / "user_profiles" / "v1" / "profile_snapshots.parquet"
        ),
        preference_signals_path=(
            source / "data" / "features" / "user_profiles" / "v1" / "preference_signals.parquet"
        ),
        output_root=args.output_root,
    )
    print(f"metrics={result.metrics_path}")
    print(f"cases={result.cases_path}")
    print(f"explorer={result.explorer_path}")
    print(f"failures={result.failures_path}")
    print(f"explorer_index={result.explorer_index_path}")


if __name__ == "__main__":
    main()
