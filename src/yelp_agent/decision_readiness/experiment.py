"""Run the complete Step 19 calibration experiment without accepting Test inputs."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from yelp_agent.config import DecisionReadinessConfig, EvaluationDataUsageConfig
from yelp_agent.experiments import write_json_artifact, write_text_artifact
from yelp_agent.models import StrictModel

from .artifacts import (
    FrozenConfidenceManifest,
    configuration_sha256,
    load_frozen_confidence_calibrator,
    save_frozen_confidence_calibrator,
    sha256_file,
)
from .calibration import (
    FrozenConfidenceCalibrator,
    derive_uncertainty_thresholds,
    select_confidence_calibrator,
)
from .dataset import (
    CalibrationDatasetBuildResult,
    build_calibration_dataset,
    load_calibration_batch,
)
from .schema import (
    CALIBRATION_FEATURE_NAMES,
    CalibrationMetrics,
    DecisionReadinessExperimentReport,
)


@dataclass(frozen=True, slots=True)
class DecisionReadinessSources:
    validation_features: Path
    validation_predictions: Path
    validation_ground_truth: Path


class DecisionReadinessExperimentResult(StrictModel):
    status: Literal["written", "reused"]
    calibration_dataset: CalibrationDatasetBuildResult
    artifact_root: str
    report_path: str
    reliability_diagram_path: str
    coverage_risk_path: str
    report: DecisionReadinessExperimentReport


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _metric_csv(metrics: CalibrationMetrics) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "lower_bound",
            "upper_bound",
            "task_count",
            "mean_confidence",
            "observed_accuracy",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for item in metrics.reliability_bins:
        writer.writerow(item.model_dump())
    return buffer.getvalue()


def _coverage_csv(metrics: CalibrationMetrics) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "requested_coverage",
            "actual_coverage",
            "selected_task_count",
            "minimum_confidence",
            "observed_accuracy",
            "risk",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for item in metrics.coverage_risk:
        writer.writerow(item.model_dump())
    return buffer.getvalue()


def _reliability_svg(metrics: CalibrationMetrics) -> str:
    width = 720
    height = 520
    left = 80
    top = 40
    plot = 400
    points = []
    for item in metrics.reliability_bins:
        if item.mean_confidence is None or item.observed_accuracy is None:
            continue
        x = left + item.mean_confidence * plot
        y = top + (1.0 - item.observed_accuracy) * plot
        points.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2563eb">'
            f"<title>n={item.task_count}, confidence={item.mean_confidence:.4f}, "
            f"accuracy={item.observed_accuracy:.4f}</title></circle>"
        )
    ticks = []
    for index in range(6):
        value = index / 5
        x = left + value * plot
        y = top + (1 - value) * plot
        ticks.append(
            f'<text x="{x:.1f}" y="{top + plot + 24}" text-anchor="middle" '
            f'font-size="12">{value:.1f}</text>'
        )
        ticks.append(
            f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12">{value:.1f}</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="520" '
        'viewBox="0 0 720 520">'
        '<rect width="720" height="520" fill="white"/>'
        '<text x="360" y="24" text-anchor="middle" font-size="18" '
        'font-family="sans-serif">Step 19 Reliability Diagram</text>'
        f'<rect x="{left}" y="{top}" width="{plot}" height="{plot}" '
        'fill="#f8fafc" stroke="#334155"/>'
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top}" '
        'stroke="#94a3b8" stroke-dasharray="6 5"/>'
        + "".join(points)
        + "".join(ticks)
        + f'<text x="{left + plot / 2}" y="{top + plot + 54}" '
        'text-anchor="middle" font-size="14">Predicted Top-1 probability</text>'
        + f'<text x="22" y="{top + plot / 2}" text-anchor="middle" '
        'font-size="14" transform="rotate(-90 22 '
        f'{top + plot / 2})">Observed Top-1 accuracy</text>'
        + f'<text x="{left + plot + 35}" y="{top + 30}" font-size="13">'
        f'Method: {metrics.method}</text>'
        + f'<text x="{left + plot + 35}" y="{top + 52}" font-size="13">'
        f'Brier: {metrics.brier_score:.6f}</text>'
        + f'<text x="{left + plot + 35}" y="{top + 74}" font-size="13">'
        f'ECE: {metrics.expected_calibration_error:.6f}</text>'
        "</svg>\n"
    )


def _dataset_result_from_existing(path: Path) -> CalibrationDatasetBuildResult:
    batch = load_calibration_batch(path)
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["fold", "top1_correct", "target_retrieved"])
    folds = [int(value) for value in table["fold"].to_pylist()]
    return CalibrationDatasetBuildResult(
        output_path=str(path),
        output_sha256=sha256_file(path),
        task_count=len(batch.task_ids),
        positive_count=int(batch.labels.sum()),
        target_retrieved_count=int(
            sum(bool(value) for value in table["target_retrieved"].to_pylist())
        ),
        fold_counts={str(fold): folds.count(fold) for fold in sorted(set(folds))},
    )


def run_decision_readiness_experiment(
    sources: DecisionReadinessSources,
    config: DecisionReadinessConfig,
    evaluation_policy: EvaluationDataUsageConfig,
    *,
    feature_output_path: str | Path,
    run_root: str | Path,
) -> DecisionReadinessExperimentResult:
    """Build, cross-validate, select, freeze, and report Step 19."""

    source_paths = {
        "validation_features": _required(
            sources.validation_features,
            "Validation features",
        ),
        "validation_predictions": _required(
            sources.validation_predictions,
            "Validation predictions",
        ),
        "validation_ground_truth": _required(
            sources.validation_ground_truth,
            "Validation ground truth",
        ),
    }
    feature_output = Path(feature_output_path)
    run = Path(run_root)
    artifact_root = run / "frozen"
    report_path = run / "evaluation_report.json"
    reliability_csv_path = run / "reliability_diagram.csv"
    reliability_svg_path = run / "reliability_diagram.svg"
    coverage_path = run / "coverage_risk.csv"
    required_outputs = (
        feature_output,
        artifact_root / "model.joblib",
        artifact_root / "manifest.json",
        report_path,
        reliability_csv_path,
        reliability_svg_path,
        coverage_path,
    )
    if all(path.is_file() for path in required_outputs):
        _, manifest = load_frozen_confidence_calibrator(artifact_root)
        current_source_sha256 = {
            name: sha256_file(path) for name, path in source_paths.items()
        }
        if manifest.source_sha256 != current_source_sha256:
            raise ValueError(
                "Step 19 inputs changed; use a new run path or rebuild the run"
            )
        if manifest.configuration_sha256 != configuration_sha256(config):
            raise ValueError(
                "Step 19 configuration changed; use a new run path or rebuild the run"
            )
        report = DecisionReadinessExperimentReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        if (
            report.selected_calibrator != manifest.selected_calibrator
            or report.source_task_count != manifest.training_task_count
            or report.fold_counts != manifest.fold_counts
        ):
            raise ValueError("Step 19 report and frozen manifest disagree")
        return DecisionReadinessExperimentResult(
            status="reused",
            calibration_dataset=_dataset_result_from_existing(feature_output),
            artifact_root=str(artifact_root),
            report_path=str(report_path),
            reliability_diagram_path=str(reliability_svg_path),
            coverage_risk_path=str(coverage_path),
            report=report,
        )
    if any(path.exists() for path in required_outputs):
        raise FileExistsError("Step 19 output set is incomplete; do not mix runs")

    dataset = build_calibration_dataset(
        validation_features_path=source_paths["validation_features"],
        validation_predictions_path=source_paths["validation_predictions"],
        ground_truth_path=source_paths["validation_ground_truth"],
        output_path=feature_output,
        evaluation_policy=evaluation_policy,
    )
    batch = load_calibration_batch(feature_output)
    selection = select_confidence_calibrator(
        batch,
        random_seed=config.random_seed,
        maximum_iterations=config.maximum_iterations,
        ece_bin_count=config.ece_bin_count,
        coverage_points=tuple(config.coverage_points),
    )
    thresholds = derive_uncertainty_thresholds(
        batch,
        sparse_history_count_max=config.sparse_history_count_max,
        unseen_category_min=config.unseen_category_min,
        feature_disagreement_min=config.feature_disagreement_min,
        small_top_margin_quantile=config.small_top_margin_quantile,
        weak_collaborative_support_quantile=(
            config.weak_collaborative_support_quantile
        ),
        low_profile_reliability_max=config.low_profile_reliability_max,
    )
    calibrator = FrozenConfidenceCalibrator(
        model=selection.selected_model,
        thresholds=thresholds,
        version=config.calibrator_version,
    )
    base_rate = float(batch.labels.mean())
    base_rate_brier = base_rate * (1.0 - base_rate)
    selected_brier = selection.candidate_metrics[
        selection.selected_kind
    ].brier_score
    report = DecisionReadinessExperimentReport(
        source_task_count=len(batch.task_ids),
        fold_counts=dataset.fold_counts,
        selected_calibrator=selection.selected_kind,
        candidate_metrics=selection.candidate_metrics,
        base_rate_brier_score=base_rate_brier,
        selected_brier_improvement_vs_base_rate=(
            (base_rate_brier - selected_brier) / base_rate_brier
        ),
        confidence_target="hybrid_v2_b_next_business_top1",
        query_aware_limitation=(
            "No Query x business relevance labels exist; Hybrid V2 next-business "
            "confidence must not be presented as Query-aware correctness."
        ),
    )
    manifest = FrozenConfidenceManifest(
        artifact_name="Step 19 Hybrid V2-B Top-1 Confidence Calibrator",
        model_version=config.calibrator_version,
        selected_calibrator=selection.selected_kind,
        feature_names=list(CALIBRATION_FEATURE_NAMES),
        uncertainty_thresholds=thresholds,
        candidate_metrics=selection.candidate_metrics,
        source_sha256={
            name: sha256_file(path) for name, path in source_paths.items()
        },
        configuration_sha256=configuration_sha256(config),
        model_sha256="0" * 64,
        training_task_count=dataset.task_count,
        top1_correct_count=dataset.positive_count,
        target_retrieved_count=dataset.target_retrieved_count,
        fold_counts=dataset.fold_counts,
        confidence_target="hybrid_v2_b_next_business_top1",
    )
    save_frozen_confidence_calibrator(artifact_root, calibrator, manifest)
    selected_metrics = selection.candidate_metrics[selection.selected_kind]
    write_json_artifact(report_path, report)
    write_text_artifact(reliability_csv_path, _metric_csv(selected_metrics))
    write_text_artifact(reliability_svg_path, _reliability_svg(selected_metrics))
    write_text_artifact(coverage_path, _coverage_csv(selected_metrics))
    return DecisionReadinessExperimentResult(
        status="written",
        calibration_dataset=dataset,
        artifact_root=str(artifact_root),
        report_path=str(report_path),
        reliability_diagram_path=str(reliability_svg_path),
        coverage_risk_path=str(coverage_path),
        report=report,
    )
