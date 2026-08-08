"""End-to-end Hybrid V2-B validation selection with no Legacy Test input."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb
from pydantic import Field

from yelp_agent.config import HybridV2BConfig
from yelp_agent.learning_to_rank.evaluation import (
    HybridV2RankingMetrics,
    HybridV2RankingWriteResult,
)
from yelp_agent.learning_to_rank.lambdamart import LambdaMARTParameters
from yelp_agent.learning_to_rank.lambdamart_artifacts import (
    FrozenLambdaMARTManifest,
    load_frozen_lambdamart,
    save_frozen_lambdamart,
)
from yelp_agent.learning_to_rank.lambdamart_selection import (
    LambdaMARTSelectionSources,
    LambdaMARTSelectionTrial,
    select_lambdamart_model,
    selection_key,
)
from yelp_agent.learning_to_rank.scored_ranking import (
    CandidateScoreWriteResult,
    write_scored_rankings,
)
from yelp_agent.models import StrictModel


@dataclass(frozen=True, slots=True)
class HybridV2BExperimentSources:
    train_features: Path
    validation_features: Path
    validation_contexts: Path
    validation_ground_truth: Path
    reviews: Path
    interactions: Path
    hybrid_v2_a_validation_report: Path


class HybridV2BValidationReport(StrictModel):
    experiment_name: Literal["Hybrid V2-B LambdaMART validation selection"]
    retrieval_configuration_changed: Literal[False] = False
    candidate_limit: Literal[500] = 500
    baseline: HybridV2RankingMetrics
    hybrid_v2_a_logistic: HybridV2RankingMetrics
    fair_comparison: HybridV2RankingMetrics
    selected: HybridV2RankingMetrics
    selected_parameters: LambdaMARTParameters
    selected_feature_set: str = Field(min_length=1)
    selected_blend_alpha: float = Field(ge=0, le=1)
    parameter_trials: list[LambdaMARTSelectionTrial]
    blend_trials: list[LambdaMARTSelectionTrial]
    ablations: dict[str, HybridV2RankingMetrics]
    absolute_delta_vs_logistic: dict[str, float]
    relative_delta_vs_logistic: dict[str, float]
    validation_champion: Literal["hybrid_v2_a_logistic", "hybrid_v2_b_lambdamart"]
    legacy_test_used_for_training: Literal[False] = False
    legacy_test_used_for_selection: Literal[False] = False
    feature_importance_is_causal: Literal[False] = False


class HybridV2BExperimentResult(StrictModel):
    status: Literal["written", "reused"]
    artifact_root: str
    validation_scores: CandidateScoreWriteResult
    validation_predictions: HybridV2RankingWriteResult
    validation_report: HybridV2BValidationReport


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha256(config: HybridV2BConfig) -> str:
    canonical = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _metric_summary(metrics: HybridV2RankingMetrics) -> dict[str, float]:
    return {
        "hr_at_1": metrics.hr_at_1,
        "hr_at_3": metrics.hr_at_3,
        "hr_at_5": metrics.hr_at_5,
        "avg_hr": metrics.avg_hr,
        "mrr": metrics.mrr,
        "ndcg_at_5": metrics.ndcg_at_5,
    }


def _load_logistic_metrics(
    path: Path,
) -> tuple[HybridV2RankingMetrics, HybridV2RankingMetrics]:
    payload = json.loads(_required(path, "Hybrid V2-A report").read_text("utf-8"))
    return (
        HybridV2RankingMetrics.model_validate(payload["baseline"]),
        HybridV2RankingMetrics.model_validate(payload["selected"]),
    )


def _delta(
    challenger: HybridV2RankingMetrics,
    baseline: HybridV2RankingMetrics,
) -> tuple[dict[str, float], dict[str, float]]:
    names = ("hr_at_1", "hr_at_3", "hr_at_5", "avg_hr", "mrr", "ndcg_at_5")
    absolute = {
        name: float(getattr(challenger, name) - getattr(baseline, name))
        for name in names
    }
    relative = {
        name: (
            0.0
            if float(getattr(baseline, name)) == 0.0
            else absolute[name] / float(getattr(baseline, name))
        )
        for name in names
    }
    return absolute, relative


def _write_importance(path: Path, model) -> None:
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("feature", "gain", "split"),
        )
        writer.writeheader()
        for name in sorted(gain, key=lambda value: (-gain[value], value)):
            writer.writerow({"feature": name, "gain": gain[name], "split": split[name]})
    os.replace(partial, path)


def _write_report(path: Path, report: HybridV2BValidationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _artifact_counts(path: Path) -> tuple[int, int]:
    with duckdb.connect() as connection:
        tasks, rows = connection.execute(
            "SELECT count(DISTINCT task_id), count(*) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
    return int(tasks), int(rows)


def run_hybrid_v2_b_validation_experiment(
    sources: HybridV2BExperimentSources,
    config: HybridV2BConfig,
    *,
    data_root: str | Path,
    run_root: str | Path,
) -> HybridV2BExperimentResult:
    """Select and freeze V2-B without accepting any Test source path."""

    source_paths = {
        "train_features": _required(sources.train_features, "Train features"),
        "validation_features": _required(
            sources.validation_features, "Validation features"
        ),
        "validation_contexts": _required(
            sources.validation_contexts, "Validation contexts"
        ),
        "validation_ground_truth": _required(
            sources.validation_ground_truth, "Validation ground truth"
        ),
        "reviews": _required(sources.reviews, "Reviews"),
        "interactions": _required(sources.interactions, "Interactions"),
        "hybrid_v2_a_validation_report": _required(
            sources.hybrid_v2_a_validation_report, "Hybrid V2-A report"
        ),
    }
    data = Path(data_root)
    run = Path(run_root)
    artifact_root = run / "frozen"
    report_path = run / "validation_report.json"
    score_path = data / "validation_scores.parquet"
    prediction_path = data / "validation_predictions.parquet"
    if report_path.is_file() and score_path.is_file() and prediction_path.is_file():
        load_frozen_lambdamart(artifact_root)
        score_tasks, score_rows = _artifact_counts(score_path)
        prediction_tasks, prediction_rows = _artifact_counts(prediction_path)
        return HybridV2BExperimentResult(
            status="reused",
            artifact_root=str(artifact_root),
            validation_scores=CandidateScoreWriteResult(
                output_path=str(score_path),
                task_count=score_tasks,
                row_count=score_rows,
            ),
            validation_predictions=HybridV2RankingWriteResult(
                output_path=str(prediction_path),
                task_count=prediction_tasks,
                row_count=prediction_rows,
            ),
            validation_report=HybridV2BValidationReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            ),
        )
    if any(path.exists() for path in (report_path, score_path, prediction_path)):
        raise FileExistsError("Hybrid V2-B output set is incomplete; do not mix runs")
    baseline, logistic = _load_logistic_metrics(sources.hybrid_v2_a_validation_report)
    selection = select_lambdamart_model(
        LambdaMARTSelectionSources(
            train_features=sources.train_features,
            validation_features=sources.validation_features,
            validation_contexts=sources.validation_contexts,
            validation_ground_truth=sources.validation_ground_truth,
            reviews=sources.reviews,
            interactions=sources.interactions,
        ),
        config,
        scratch_root=data / "scratch",
        final_scores_path=score_path,
    )
    predictions = write_scored_rankings(
        scores_path=score_path,
        output_path=prediction_path,
        blend_alpha=selection.selected_blend_alpha,
    )
    absolute, relative = _delta(selection.selected_metrics, logistic)
    champion: Literal["hybrid_v2_a_logistic", "hybrid_v2_b_lambdamart"] = (
        "hybrid_v2_b_lambdamart"
        if selection_key(selection.selected_metrics) > selection_key(logistic)
        else "hybrid_v2_a_logistic"
    )
    gain_importance = selection.model.feature_importance(importance_type="gain")
    split_importance = selection.model.feature_importance(importance_type="split")
    manifest = FrozenLambdaMARTManifest(
        artifact_name="Hybrid V2-B LambdaMART",
        model_version=config.model_version,
        feature_version="1.0.0",
        selected_parameters=selection.selected_parameters,
        selected_blend_alpha=selection.selected_blend_alpha,
        selected_feature_set=selection.selected_feature_set,
        feature_names=list(selection.model.feature_names),
        source_sha256={name: _sha256(path) for name, path in source_paths.items()},
        configuration_sha256=_configuration_sha256(config),
        model_sha256="0" * 64,
        validation_summary=_metric_summary(selection.selected_metrics),
        fair_comparison_summary=_metric_summary(selection.fair_comparison_metrics),
        parameter_trials=[
            trial.model_dump(mode="json") for trial in selection.parameter_trials
        ],
        blend_trials=[
            trial.model_dump(mode="json") for trial in selection.blend_trials
        ],
        ablation_summaries={
            name: _metric_summary(metrics)
            for name, metrics in selection.ablation_metrics.items()
        },
        feature_importance_gain=gain_importance,
        feature_importance_split=split_importance,
        test_data_used_for_training=False,
        test_data_used_for_selection=False,
    )
    save_frozen_lambdamart(artifact_root, selection.model, manifest)
    report = HybridV2BValidationReport(
        experiment_name="Hybrid V2-B LambdaMART validation selection",
        baseline=baseline,
        hybrid_v2_a_logistic=logistic,
        fair_comparison=selection.fair_comparison_metrics,
        selected=selection.selected_metrics,
        selected_parameters=selection.selected_parameters,
        selected_feature_set=selection.selected_feature_set,
        selected_blend_alpha=selection.selected_blend_alpha,
        parameter_trials=list(selection.parameter_trials),
        blend_trials=list(selection.blend_trials),
        ablations=selection.ablation_metrics,
        absolute_delta_vs_logistic=absolute,
        relative_delta_vs_logistic=relative,
        validation_champion=champion,
    )
    _write_importance(run / "feature_importance.csv", selection.model)
    _write_report(report_path, report)
    return HybridV2BExperimentResult(
        status="written",
        artifact_root=str(artifact_root),
        validation_scores=selection.validation_scores,
        validation_predictions=predictions,
        validation_report=report,
    )
