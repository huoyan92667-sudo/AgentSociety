"""Replace targeted reruns in one complete Rule Agent benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.rule_router import (
    RuleAgentSourcePaths,
    replace_rule_agent_benchmark_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sources = RuleAgentSourcePaths.from_project_root(args.project_root)
    result = replace_rule_agent_benchmark_outputs(
        base_root=args.base_root,
        replacement_root=args.replacement_root,
        benchmark_root=sources.benchmark_root,
        output_root=args.output_root,
    )
    print(f"scenarios={result.report.scenario_count}")
    print(f"runs={result.runs_path}")
    print(f"metrics={result.metrics_path}")


if __name__ == "__main__":
    main()
