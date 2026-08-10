"""Thin CLI for the complete Step 24 Rule Agent benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.rule_router import (
    RuleAgentSourcePaths,
    build_real_rule_agent_runtime,
    run_rule_agent_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=("development", "validation", "all"),
        default="all",
    )
    args = parser.parse_args()
    sources = RuleAgentSourcePaths.from_project_root(args.project_root)
    output = args.output_root or sources.project_root / "runs" / "rule_agent_v1"

    def progress(index: int, total: int, scenario: object) -> None:
        if index == 1 or index % 25 == 0 or index == total:
            split = getattr(scenario, "split", "unknown")
            print(f"[{index}/{total}] completed ({split})", flush=True)

    with build_real_rule_agent_runtime(sources) as runtime:
        result = run_rule_agent_benchmark(
            runtime.harness,
            benchmark_root=sources.benchmark_root,
            output_root=output,
            split=args.split,
            progress=progress,
        )
    print(f"runs={result.runs_path}")
    print(f"metrics={result.metrics_path}")
    print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
