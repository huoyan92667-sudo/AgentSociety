"""Build and audit the complete Step 20 Agent Scenario Benchmark V1."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.agent_benchmark import AgentBenchmarkSources, build_agent_benchmark
from yelp_agent.config import load_agent_benchmark_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--businesses",
        type=Path,
        default=Path("data/features/business_profiles/v1/businesses.parquet"),
    )
    parser.add_argument(
        "--rating-events",
        type=Path,
        default=Path("data/features/business_profiles/v1/rating_events.parquet"),
    )
    parser.add_argument(
        "--aspect-records",
        type=Path,
        default=Path(
            "data/features/review_aspects/development/aspect_records.parquet"
        ),
    )
    parser.add_argument(
        "--profile-snapshots",
        type=Path,
        default=Path("data/features/user_profiles/v1/profile_snapshots.parquet"),
    )
    parser.add_argument(
        "--preference-signals",
        type=Path,
        default=Path("data/features/user_profiles/v1/preference_signals.parquet"),
    )
    parser.add_argument(
        "--task-profile-map",
        type=Path,
        default=Path("data/features/user_profiles/v1/task_profile_map.parquet"),
    )
    parser.add_argument(
        "--query-benchmark",
        type=Path,
        default=Path("benchmarks/query_aware_v2/queries_500.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/agent_scenarios_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_agent_benchmark_config(args.config_dir)
    if config.rewriter == "openai_compatible":
        raise SystemExit(
            "Real rewriting requires an explicitly injected provider adapter; "
            "the frozen Step 20 build never calls an API implicitly."
        )
    result = build_agent_benchmark(
        AgentBenchmarkSources(
            businesses=args.businesses,
            rating_events=args.rating_events,
            aspect_records=args.aspect_records,
            profile_snapshots=args.profile_snapshots,
            preference_signals=args.preference_signals,
            task_profile_map=args.task_profile_map,
            query_benchmark=args.query_benchmark,
        ),
        config,
        output_root=args.output_root,
    )
    print(f"status={result.status}")
    print(f"scenarios={result.manifest.scenario_count}")
    print(f"split_counts={result.manifest.split_counts}")
    print(f"category_counts={result.manifest.category_counts}")
    print(f"evidence_labels={result.audit.evidence_label_count}")
    print(f"scripted_turns={result.audit.scripted_user_turn_count}")
    print(f"root={result.root}")


if __name__ == "__main__":
    main()
