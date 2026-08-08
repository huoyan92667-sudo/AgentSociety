"""One resumable validation-only experiment that freezes Hybrid V2-A."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from yelp_agent.config import BusinessProfileConfig, HybridV2Config
from yelp_agent.experiments import write_json_artifact
from yelp_agent.learning_to_rank.artifacts import (
    FrozenHybridV2Manifest,
    load_frozen_hybrid_v2,
    save_frozen_hybrid_v2,
)
from yelp_agent.learning_to_rank.evaluation import (
    HybridV2RankingMetrics,
    HybridV2RankingWriteResult,
    write_hybrid_v2_rankings,
)
from yelp_agent.learning_to_rank.features import (
    ALL_FEATURE_NAMES,
    HybridV1Weights,
    HybridV2FeatureBuildResult,
    HybridV2FeatureSources,
    build_hybrid_v2_features,
)
from yelp_agent.learning_to_rank.sampling import (
    TrainingSelectionResult,
    build_training_selection,
)
from yelp_agent.learning_to_rank.selection import (
    HybridV2SelectionSources,
    HybridV2SelectionTrial,
    select_hybrid_v2_model,
)
from yelp_agent.models import StrictModel


@dataclass(frozen=True, slots=True)
class HybridV2ExperimentSources:
    train_candidates: Path
    validation_candidates: Path
    train_ground_truth: Path
    validation_contexts: Path
    validation_ground_truth: Path
    user_profile_root: Path
    business_profile_root: Path
    reviews: Path
    interactions: Path
    hybrid_v1_weights: Path
    retrieval_manifest: Path
    user_profile_manifest: Path
    business_profile_manifest: Path


class HybridV2ValidationReport(StrictModel):
    experiment_name: Literal["Hybrid V2-A validation selection"]
    training_selection: TrainingSelectionResult
    train_features: HybridV2FeatureBuildResult
    validation_features: HybridV2FeatureBuildResult
    validation_predictions: HybridV2RankingWriteResult
    baseline: HybridV2RankingMetrics
    selected: HybridV2RankingMetrics
    selected_feature_set: str = Field(default="full", min_length=1)
    regularization_trials: list[HybridV2SelectionTrial]
    blend_trials: list[HybridV2SelectionTrial]
    ablations: dict[str, HybridV2RankingMetrics]
    frozen_manifest_path: str
    legacy_test_used_for_training: Literal[False] = False
    legacy_test_used_for_selection: Literal[False] = False


class HybridV2ExperimentResult(StrictModel):
    status: Literal["written", "skipped"]
    artifact_root: str
    data_root: str
    report_path: str
    selected_feature_set: str
    selected_regularization_c: float = Field(gt=0)
    selected_blend_alpha: float = Field(ge=0, le=1)
    validation_avg_hr: float = Field(ge=0, le=1)
    baseline_avg_hr: float = Field(ge=0, le=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha256(
    model_config: HybridV2Config,
    business_config: BusinessProfileConfig,
    weights: HybridV1Weights,
    broad_categories: set[str],
) -> str:
    payload = {
        "business_profile": business_config.model_dump(mode="json"),
        "feature_names": ALL_FEATURE_NAMES,
        "hybrid_v1_weights": weights.model_dump(mode="json"),
        "hybrid_v2": model_config.model_dump(mode="json"),
        "broad_categories": sorted(broad_categories),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_hybrid_v1_weights(path: Path) -> HybridV1Weights:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen Hybrid V1 weights do not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_weights")
    if not isinstance(selected, dict):
        raise TypeError("Hybrid V1 artifact does not contain selected_weights")
    return HybridV1Weights.model_validate(selected)


def _metric_summary(metrics: HybridV2RankingMetrics) -> dict[str, float]:
    return {
        "hr_at_1": metrics.hr_at_1,
        "hr_at_3": metrics.hr_at_3,
        "hr_at_5": metrics.hr_at_5,
        "avg_hr": metrics.avg_hr,
        "mrr": metrics.mrr,
        "ndcg_at_5": metrics.ndcg_at_5,
        "retrieved_target_mrr": metrics.retrieved_target_mrr,
    }


def _required_sources(sources: HybridV2ExperimentSources) -> dict[str, Path]:
    values = {
        "train_candidates": sources.train_candidates,
        "validation_candidates": sources.validation_candidates,
        "train_ground_truth": sources.train_ground_truth,
        "validation_contexts": sources.validation_contexts,
        "validation_ground_truth": sources.validation_ground_truth,
        "reviews": sources.reviews,
        "interactions": sources.interactions,
        "hybrid_v1_weights": sources.hybrid_v1_weights,
        "retrieval_manifest": sources.retrieval_manifest,
        "user_profile_manifest": sources.user_profile_manifest,
        "business_profile_manifest": sources.business_profile_manifest,
    }
    missing = [str(path) for path in values.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Hybrid V2 sources are missing: " + ", ".join(missing))
    return values


def run_hybrid_v2_validation_experiment(
    sources: HybridV2ExperimentSources,
    *,
    data_root: str | Path,
    artifact_root: str | Path,
    report_path: str | Path,
    model_config: HybridV2Config,
    business_profile_config: BusinessProfileConfig,
    broad_categories: set[str],
    reuse_prepared: bool = False,
) -> HybridV2ExperimentResult:
    """Prepare, select on validation, and freeze without reading Legacy Test."""

    source_paths = _required_sources(sources)
    data = Path(data_root)
    artifact = Path(artifact_root)
    report = Path(report_path)
    if (artifact / "manifest.json").is_file():
        _, manifest = load_frozen_hybrid_v2(artifact)
        existing_report = HybridV2ValidationReport.model_validate_json(
            report.read_text(encoding="utf-8")
        )
        return HybridV2ExperimentResult(
            status="skipped",
            artifact_root=str(artifact),
            data_root=str(data),
            report_path=str(report),
            selected_feature_set=manifest.selected_feature_set,
            selected_regularization_c=manifest.selected_regularization_c,
            selected_blend_alpha=manifest.selected_blend_alpha,
            validation_avg_hr=existing_report.selected.avg_hr,
            baseline_avg_hr=existing_report.baseline.avg_hr,
        )
    generated = {
        "selection": data / "training_selection.parquet",
        "train_features": data / "train_features.parquet",
        "validation_features": data / "validation_features.parquet",
        "validation_predictions": data / "validation_predictions.parquet",
    }
    weights = _load_hybrid_v1_weights(sources.hybrid_v1_weights)
    prepared_names = ("selection", "train_features", "validation_features")
    if reuse_prepared:
        missing_prepared = [
            str(generated[name])
            for name in prepared_names
            if not generated[name].is_file()
        ]
        if missing_prepared or not report.is_file():
            raise FileNotFoundError(
                "Reusable Hybrid V2 preparation is incomplete: "
                + ", ".join(missing_prepared)
            )
        if generated["validation_predictions"].exists():
            raise FileExistsError(
                "Move the previous validation predictions before reselection"
            )
        previous = HybridV2ValidationReport.model_validate_json(
            report.read_text(encoding="utf-8")
        )
        training_selection = previous.training_selection
        train_features = previous.train_features
        validation_features = previous.validation_features
        expected_hashes = {
            "selection": training_selection.output_sha256,
            "train_features": train_features.output_sha256,
            "validation_features": validation_features.output_sha256,
        }
        changed = [
            name
            for name, expected in expected_hashes.items()
            if _sha256(generated[name]) != expected
        ]
        if changed:
            raise ValueError(
                "Reusable Hybrid V2 preparation hash changed: " + ", ".join(changed)
            )
    else:
        existing = [str(path) for path in generated.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Incomplete Hybrid V2 preparation already exists: "
                + ", ".join(existing)
            )
        training_selection = build_training_selection(
            sources.train_candidates,
            sources.train_ground_truth,
            generated["selection"],
            model_config,
        )
        train_features = build_hybrid_v2_features(
            HybridV2FeatureSources(
                candidates=generated["selection"],
                user_profile_root=sources.user_profile_root,
                business_profile_root=sources.business_profile_root,
            ),
            generated["train_features"],
            split="train",
            weights=weights,
            broad_categories=broad_categories,
            business_profile_config=business_profile_config,
            batch_size=model_config.feature_batch_size,
        )
        validation_features = build_hybrid_v2_features(
            HybridV2FeatureSources(
                candidates=sources.validation_candidates,
                user_profile_root=sources.user_profile_root,
                business_profile_root=sources.business_profile_root,
            ),
            generated["validation_features"],
            split="validation",
            weights=weights,
            broad_categories=broad_categories,
            business_profile_config=business_profile_config,
            batch_size=model_config.feature_batch_size,
        )
    selection = select_hybrid_v2_model(
        HybridV2SelectionSources(
            train_features=generated["train_features"],
            validation_features=generated["validation_features"],
            validation_contexts=sources.validation_contexts,
            validation_ground_truth=sources.validation_ground_truth,
            reviews=sources.reviews,
            interactions=sources.interactions,
        ),
        model_config,
    )
    predictions = write_hybrid_v2_rankings(
        features_path=generated["validation_features"],
        output_path=generated["validation_predictions"],
        model=selection.model,
        blend_alpha=selection.selected_blend_alpha,
    )
    source_sha256 = {name: _sha256(path) for name, path in sorted(source_paths.items())}
    source_sha256.update(
        {
            "training_selection": _sha256(generated["selection"]),
            "train_features": _sha256(generated["train_features"]),
            "validation_features": _sha256(generated["validation_features"]),
        }
    )
    manifest = FrozenHybridV2Manifest(
        artifact_name="Hybrid V2-A Pairwise Logistic",
        model_version=model_config.model_version,
        feature_version="1.0.0",
        selected_regularization_c=selection.selected_regularization_c,
        selected_blend_alpha=selection.selected_blend_alpha,
        selected_feature_set=selection.selected_feature_set,
        feature_names=list(selection.model.feature_names),
        source_sha256=source_sha256,
        configuration_sha256=_configuration_sha256(
            model_config, business_profile_config, weights, broad_categories
        ),
        model_sha256="0" * 64,
        validation_summary=_metric_summary(selection.selected_metrics),
        regularization_trials=[
            trial.model_dump(mode="json") for trial in selection.c_trials
        ],
        blend_trials=[
            trial.model_dump(mode="json") for trial in selection.alpha_trials
        ],
        ablation_summaries={
            name: _metric_summary(metrics)
            for name, metrics in selection.ablation_metrics.items()
        },
        coefficient_report={
            name: float(value)
            for name, value in zip(
                selection.model.feature_names,
                selection.model.coefficients,
                strict=True,
            )
        },
        test_data_used_for_training=False,
        test_data_used_for_selection=False,
    )
    save_frozen_hybrid_v2(artifact, selection.model, manifest)
    validation_report = HybridV2ValidationReport(
        experiment_name="Hybrid V2-A validation selection",
        training_selection=training_selection,
        train_features=train_features,
        validation_features=validation_features,
        validation_predictions=predictions,
        baseline=selection.baseline_metrics,
        selected=selection.selected_metrics,
        selected_feature_set=selection.selected_feature_set,
        regularization_trials=list(selection.c_trials),
        blend_trials=list(selection.alpha_trials),
        ablations=selection.ablation_metrics,
        frozen_manifest_path=str(artifact / "manifest.json"),
    )
    write_json_artifact(report, validation_report)
    return HybridV2ExperimentResult(
        status="written",
        artifact_root=str(artifact),
        data_root=str(data),
        report_path=str(report),
        selected_feature_set=selection.selected_feature_set,
        selected_regularization_c=selection.selected_regularization_c,
        selected_blend_alpha=selection.selected_blend_alpha,
        validation_avg_hr=selection.selected_metrics.avg_hr,
        baseline_avg_hr=selection.baseline_metrics.avg_hr,
    )
