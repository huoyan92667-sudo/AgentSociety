"""Run the Step 19 state baseline on the frozen Step 20 benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.agent_benchmark.baseline import (
    evaluate_decision_readiness_benchmark,
    write_decision_readiness_benchmark_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visible",
        type=Path,
        default=Path("benchmarks/agent_scenarios_v1/visible/scenarios.jsonl"),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("benchmarks/agent_scenarios_v1/hidden/ground_truth.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/agent_scenarios_v1/decision_readiness_baseline.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_decision_readiness_benchmark(
        args.visible,
        args.ground_truth,
    )
    write_decision_readiness_benchmark_report(report, args.output)
    print(f"scenarios={report.scenario_count}")
    print(f"task_type_accuracy={report.task_type_accuracy:.6f}")
    print(f"information_gap_f1={report.information_gap_f1:.6f}")
    print(
        "query_aware_confidence_refusal_rate="
        f"{report.query_aware_confidence_refusal_rate:.6f}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
