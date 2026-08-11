"""Merge frozen Step 30 Development and Validation outputs without rerunning."""

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
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sources = RuleAgentSourcePaths.from_project_root(root)
    development = _resolve(root, args.development_root)
    validation = _resolve(root, args.validation_root)
    output = _resolve(root, args.output_root)
    result = merge_rule_agent_benchmark_outputs(
        development_root=development,
        validation_root=validation,
        benchmark_root=sources.benchmark_root,
        output_root=output,
    )
    for filename in ("semantic_ranking_diagnostics.jsonl", "llm_calls.jsonl"):
        contents = []
        for source in (development / filename, validation / filename):
            if source.is_file():
                contents.append(source.read_text(encoding="utf-8"))
        if contents:
            (output / filename).write_text("".join(contents), encoding="utf-8")
    print(f"runs={result.runs_path}")
    print(f"metrics={result.metrics_path}")


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
