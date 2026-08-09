"""Train and freeze Step 19 ranking-confidence calibration on Validation folds."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.config import load_config, load_decision_readiness_config
from yelp_agent.decision_readiness import (
    DecisionReadinessSources,
    run_decision_readiness_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--validation-features",
        type=Path,
        default=Path("data/features/hybrid_v2_a/validation_features.parquet"),
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path("data/features/hybrid_v2_b/validation_predictions.parquet"),
    )
    parser.add_argument(
        "--validation-ground-truth",
        type=Path,
        default=Path("data/task_dataset/ground_truth/ground_truth.parquet"),
    )
    parser.add_argument(
        "--feature-output",
        type=Path,
        default=Path(
            "data/features/decision_readiness/v1/calibration_examples.parquet"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/decision_readiness_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_decision_readiness_config(args.config_dir)
    evaluation_policy = load_config(args.config_dir).evaluation_data_usage
    result = run_decision_readiness_experiment(
        DecisionReadinessSources(
            validation_features=args.validation_features,
            validation_predictions=args.validation_predictions,
            validation_ground_truth=args.validation_ground_truth,
        ),
        config,
        evaluation_policy,
        feature_output_path=args.feature_output,
        run_root=args.run_root,
    )
    report = result.report
    selected = report.candidate_metrics[report.selected_calibrator]
    print(f"status={result.status}")
    print(f"tasks={report.source_task_count}")
    print(f"selected_calibrator={report.selected_calibrator}")
    print(f"brier_score={selected.brier_score:.8f}")
    print(f"base_rate_brier_score={report.base_rate_brier_score:.8f}")
    print(
        "brier_improvement_vs_base_rate="
        f"{report.selected_brier_improvement_vs_base_rate:.8%}"
    )
    print(
        "expected_calibration_error="
        f"{selected.expected_calibration_error:.8f}"
    )
    print(f"positive_rate={selected.positive_rate:.8f}")
    print(f"artifact_root={result.artifact_root}")
    print(f"report={result.report_path}")
    print(f"reliability_diagram={result.reliability_diagram_path}")
    print(f"coverage_risk={result.coverage_risk_path}")


if __name__ == "__main__":
    main()
