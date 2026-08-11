"""Run the Step 26 Rule Agent with local Embedding and Cross-Encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

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
        "--split", choices=("development", "validation", "all"), default="all"
    )
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--cross-encoder-model-path", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--scenario-ids-from-runs",
        type=Path,
        default=None,
        help="Rerun only scenario IDs in this run file matching --fallback-reason.",
    )
    parser.add_argument("--fallback-reason", default="max_total_tokens_exceeded")
    args = parser.parse_args()
    root = args.project_root.resolve()
    sources = RuleAgentSourcePaths.from_project_root(root)
    output = args.output_root or root / "runs" / "rule_agent_cross_encoder_v1" / args.split
    embedding_environment = {
        "LOCAL_EMBEDDING_MODEL_PATH": str(args.embedding_model_path),
        "LOCAL_EMBEDDING_PYTHON": str(args.model_python),
        "LOCAL_EMBEDDING_DEVICE": args.device,
    }
    cross_environment = {
        "LOCAL_CROSS_ENCODER_MODEL_PATH": str(args.cross_encoder_model_path),
        "LOCAL_CROSS_ENCODER_PYTHON": str(args.model_python),
        "LOCAL_CROSS_ENCODER_DEVICE": args.device,
    }
    scenario_ids = None
    if args.scenario_ids_from_runs is not None:
        scenario_ids = set()
        with args.scenario_ids_from_runs.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("fallback_reason") == args.fallback_reason:
                    scenario_ids.add(str(row["scenario_id"]))
        if not scenario_ids:
            raise ValueError("no scenarios matched the requested fallback reason")

    def progress(index: int, total: int, scenario: object) -> None:
        if index == 1 or index % 25 == 0 or index == total:
            split = getattr(scenario, "split", "unknown")
            print(f"[{index}/{total}] completed ({split})", flush=True)

    with build_real_rule_agent_runtime(
        sources,
        embedding_config_path=root / "configs" / "embedding.yaml",
        embedding_environment=embedding_environment,
        cross_encoder_config_path=root / "configs" / "cross_encoder.yaml",
        cross_encoder_environment=cross_environment,
    ) as runtime:
        result = run_rule_agent_benchmark(
            runtime.harness,
            benchmark_root=sources.benchmark_root,
            output_root=output,
            split=args.split,
            progress=progress,
            scenario_ids=scenario_ids,
        )
    print(f"runs={result.runs_path}")
    print(f"metrics={result.metrics_path}")
    print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
