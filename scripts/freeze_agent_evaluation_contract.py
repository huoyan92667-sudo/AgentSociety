"""Freeze the Step 21 metric catalog against Agent Scenario Benchmark V1."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.agent_evaluation import freeze_agent_evaluation_contract
from yelp_agent.config import load_agent_evaluation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmarks/agent_scenarios_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/agent_scenarios_v1/evaluation_contract.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_agent_evaluation_contract(
        args.benchmark_root,
        load_agent_evaluation_config(args.config_dir),
        output_path=args.output,
    )
    print(f"status={result.status}")
    print(f"contract_version={result.manifest.contract_version}")
    print(f"metrics={result.manifest.metric_count}")
    print(f"output={result.output_path}")


if __name__ == "__main__":
    main()
