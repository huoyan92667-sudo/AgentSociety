"""Evaluate typed Agent runs without exposing hidden labels to the Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.agent_evaluation import (
    evaluate_agent_scenario_files,
    freeze_agent_evaluation_contract,
)
from yelp_agent.config import load_agent_evaluation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmarks/agent_scenarios_v1"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("benchmarks/agent_scenarios_v1/evaluation_contract.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/agent_scenarios_v1/agent_evaluation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.contract.is_file():
        raise SystemExit(
            "Step 21 contract is missing; run freeze_agent_evaluation_contract.py first"
        )
    verification = freeze_agent_evaluation_contract(
        args.benchmark_root,
        load_agent_evaluation_config(args.config_dir),
        output_path=args.contract,
    )
    if verification.status != "reused":
        raise SystemExit("evaluation requires an already-frozen contract")
    report = evaluate_agent_scenario_files(
        args.runs,
        benchmark_root=args.benchmark_root,
        output_path=args.output,
    )
    print(f"scenarios={report.scenario_count}")
    print(f"task_type_accuracy={report.metrics['task_type_accuracy'].value}")
    print(f"action_accuracy={report.metrics['action_accuracy'].value}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
