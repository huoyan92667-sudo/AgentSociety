"""Merge frozen Step 24 development and validation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.rule_router import (
    RuleAgentSourcePaths,
    merge_rule_agent_benchmark_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--development-root", type=Path, default=None)
    parser.add_argument("--validation-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    sources = RuleAgentSourcePaths.from_project_root(args.project_root)
    output = args.output_root or sources.project_root / "runs" / "rule_agent_v1"
    development = args.development_root or output / "development"
    validation = args.validation_root or output / "validation"
    result = merge_rule_agent_benchmark_outputs(
        development_root=development,
        validation_root=validation,
        benchmark_root=sources.benchmark_root,
        output_root=output,
    )
    print(f"scenarios={result.report.scenario_count}")
    print(f"metrics={result.metrics_path}")
    print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
