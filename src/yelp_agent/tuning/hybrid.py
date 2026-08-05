"""Validation-only exhaustive tuning for Hybrid recommendation weights."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.experiments import (
    TaskFileError,
    TaskSplitError,
    read_recommendation_tasks,
    write_json_artifact,
)
from yelp_agent.features.hybrid import (
    HybridComponentFeatures,
    HybridWeights,
)
from yelp_agent.models import RecommendationTask, StrictModel


class HybridTuningError(RuntimeError):
    """Raised when Hybrid weights cannot be tuned safely."""


class HybridValidationMetrics(StrictModel):
    hr_at_1: float = Field(ge=0, le=1)
    hr_at_3: float = Field(ge=0, le=1)
    hr_at_5: float = Field(ge=0, le=1)
    avg_hr: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)


class FrozenHybridWeights(StrictModel):
    format_version: int = Field(ge=1)
    validation_tasks_sha256: str
    validation_truth_sha256: str
    feature_sources_sha256: dict[str, str]
    tuning_step: float = Field(gt=0, le=1)
    task_count: int = Field(gt=0)
    combination_count: int = Field(gt=0)
    initial_weights: HybridWeights
    selected_weights: HybridWeights
    validation_metrics: HybridValidationMetrics
    tie_break_policy: list[str]


class HybridTuneResult(StrictModel):
    status: Literal["written", "skipped"]
    weights_path: str
    task_count: int = Field(gt=0)
    combination_count: int = Field(gt=0)
    selected_weights: HybridWeights
    validation_metrics: HybridValidationMetrics


class _HybridComponentProvider(Protocol):
    def features_for(
        self,
        task: RecommendationTask,
    ) -> HybridComponentFeatures: ...


TIE_BREAK_POLICY = [
    "maximize_avg_hr",
    "maximize_mrr",
    "maximize_hr_at_1",
    "minimize_l1_distance_from_initial_weights",
    "maximize_lexicographic_category_text_quality_location",
]


def enumerate_hybrid_weights(step: float) -> list[HybridWeights]:
    """Enumerate all four-part nonnegative weights summing exactly to one."""

    if not math.isfinite(step) or step <= 0 or step > 1:
        raise ValueError("tuning step must be in (0, 1]")
    unit_count = round(1.0 / step)
    if not math.isclose(unit_count * step, 1.0, abs_tol=1e-9):
        raise ValueError("tuning step must divide 1.0 exactly")

    combinations: list[HybridWeights] = []
    for category_units in range(unit_count, -1, -1):
        remaining_after_category = unit_count - category_units
        for text_units in range(remaining_after_category, -1, -1):
            remaining_after_text = remaining_after_category - text_units
            for quality_units in range(remaining_after_text, -1, -1):
                location_units = remaining_after_text - quality_units
                combinations.append(
                    HybridWeights(
                        category=category_units / unit_count,
                        text=text_units / unit_count,
                        quality=quality_units / unit_count,
                        location=location_units / unit_count,
                    )
                )
    return combinations


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_hybrid_feature_sources(
    sources: dict[str, str | Path],
) -> dict[str, str]:
    """Hash every data, model, and config source used by Hybrid features."""

    if not sources:
        raise ValueError("at least one Hybrid feature source is required")
    fingerprints: dict[str, str] = {}
    for name, raw_path in sorted(sources.items()):
        if not name:
            raise ValueError("Hybrid feature source names cannot be empty")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Hybrid feature source {name!r} does not exist: {path}"
            )
        fingerprints[name] = _sha256_file(path)
    return fingerprints


def _load_validation_tasks(path: Path) -> list[RecommendationTask]:
    try:
        return read_recommendation_tasks(path, required_split="validation")
    except TaskSplitError as exc:
        raise HybridTuningError(
            "Hybrid tuning accepts validation tasks only"
        ) from exc
    except TaskFileError as exc:
        raise HybridTuningError(str(exc)) from exc


def _load_validation_truth(
    path: Path,
    tasks: list[RecommendationTask],
) -> tuple[list[str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth Parquet does not exist: {path}")
    expected = {task.task_id: task for task in tasks}
    selected: dict[str, str] = {}
    try:
        rows = pq.read_table(
            path,
            columns=["task_id", "target_business_id"],
        ).to_pylist()
    except Exception as exc:
        raise HybridTuningError(f"Could not read validation ground truth: {path}") from exc
    for row in rows:
        task_id = row.get("task_id")
        if task_id not in expected:
            continue
        target = row.get("target_business_id")
        if not isinstance(target, str) or not target:
            raise HybridTuningError(
                f"Validation ground truth is invalid for {task_id!r}"
            )
        if task_id in selected:
            raise HybridTuningError(
                f"Duplicate validation ground truth for {task_id!r}"
            )
        if target not in expected[task_id].candidate_business_ids:
            raise HybridTuningError(
                f"Validation target is outside candidates for {task_id!r}"
            )
        selected[str(task_id)] = target
    missing = sorted(set(expected).difference(selected))
    if missing:
        raise HybridTuningError(
            f"Ground truth is missing validation task IDs: {missing[:3]}"
        )
    targets = [selected[task.task_id] for task in tasks]
    digest = hashlib.sha256()
    for task, target in zip(tasks, targets, strict=True):
        digest.update(task.task_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(target.encode("utf-8"))
        digest.update(b"\n")
    return targets, digest.hexdigest()


def _precompute_components(
    tasks: list[RecommendationTask],
    targets: list[str],
    feature_store: _HybridComponentProvider,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = np.empty((len(tasks), 20, 4), dtype=np.float64)
    target_positions = np.empty(len(tasks), dtype=np.int64)
    tie_precedes_target = np.empty((len(tasks), 20), dtype=bool)
    for task_index, (task, target) in enumerate(
        zip(tasks, targets, strict=True)
    ):
        features = feature_store.features_for(task)
        if set(features.business_scores) != set(task.candidate_business_ids):
            raise HybridTuningError(
                f"Hybrid features do not match candidates for {task.task_id!r}"
            )
        target_positions[task_index] = task.candidate_business_ids.index(target)
        tie_precedes_target[task_index] = [
            business_id < target
            for business_id in task.candidate_business_ids
        ]
        for candidate_index, business_id in enumerate(
            task.candidate_business_ids
        ):
            score = features.business_scores[business_id]
            components[task_index, candidate_index] = (
                score.category_score,
                score.text_score,
                score.quality_score,
                score.location_score,
            )
    return components, target_positions, tie_precedes_target


def _metrics_for_weights(
    components: np.ndarray,
    target_positions: np.ndarray,
    tie_precedes_target: np.ndarray,
    weights: HybridWeights,
) -> HybridValidationMetrics:
    weight_vector = np.asarray(
        [
            weights.category,
            weights.text,
            weights.quality,
            weights.location,
        ],
        dtype=np.float64,
    )
    scores = components @ weight_vector
    task_indices = np.arange(scores.shape[0])
    target_scores = scores[task_indices, target_positions]
    strictly_higher = scores > target_scores[:, None]
    equal_and_preceding = (
        (scores == target_scores[:, None]) & tie_precedes_target
    )
    ranks = (
        1
        + np.sum(strictly_higher, axis=1)
        + np.sum(equal_and_preceding, axis=1)
    )
    hr_at_1 = float(np.mean(ranks <= 1))
    hr_at_3 = float(np.mean(ranks <= 3))
    hr_at_5 = float(np.mean(ranks <= 5))
    ndcg = np.where(
        ranks <= 5,
        1.0 / np.log2(ranks + 1),
        0.0,
    )
    return HybridValidationMetrics(
        hr_at_1=hr_at_1,
        hr_at_3=hr_at_3,
        hr_at_5=hr_at_5,
        avg_hr=(hr_at_1 + hr_at_3 + hr_at_5) / 3.0,
        mrr=float(np.mean(1.0 / ranks)),
        ndcg_at_5=float(np.mean(ndcg)),
    )


def _selection_key(
    metrics: HybridValidationMetrics,
    weights: HybridWeights,
    initial_weights: HybridWeights,
) -> tuple[float, ...]:
    l1_distance = sum(
        abs(weights.as_dict[name] - initial_weights.as_dict[name])
        for name in ("category", "text", "quality", "location")
    )
    return (
        metrics.avg_hr,
        metrics.mrr,
        metrics.hr_at_1,
        -l1_distance,
        weights.category,
        weights.text,
        weights.quality,
        weights.location,
    )


def _load_frozen_hybrid_artifact(path: str | Path) -> FrozenHybridWeights:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Frozen Hybrid weights do not exist: {source}")
    try:
        frozen = FrozenHybridWeights.model_validate_json(
            source.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise HybridTuningError(f"Frozen Hybrid weights are invalid: {source}") from exc
    if frozen.format_version != 1 or frozen.tie_break_policy != TIE_BREAK_POLICY:
        raise HybridTuningError(
            "Frozen Hybrid weights use an unsupported format or tie policy"
        )
    return frozen


def load_validated_hybrid_weights(
    path: str | Path,
    *,
    feature_sources_sha256: dict[str, str],
) -> HybridWeights:
    """Load weights only when all current Hybrid feature sources still match."""

    if not feature_sources_sha256:
        raise ValueError("feature_sources_sha256 cannot be empty")
    frozen = _load_frozen_hybrid_artifact(path)
    if frozen.feature_sources_sha256 != feature_sources_sha256:
        expected_names = set(frozen.feature_sources_sha256)
        current_names = set(feature_sources_sha256)
        changed = sorted(
            name
            for name in expected_names & current_names
            if frozen.feature_sources_sha256[name]
            != feature_sources_sha256[name]
        )
        missing = sorted(expected_names - current_names)
        unexpected = sorted(current_names - expected_names)
        details = []
        if changed:
            details.append(f"changed={changed}")
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise HybridTuningError(
            "Frozen Hybrid weights do not match current feature sources"
            + (f": {', '.join(details)}" if details else "")
        )
    return frozen.selected_weights


def _result(
    status: Literal["written", "skipped"],
    output_path: Path,
    frozen: FrozenHybridWeights,
) -> HybridTuneResult:
    return HybridTuneResult(
        status=status,
        weights_path=str(output_path),
        task_count=frozen.task_count,
        combination_count=frozen.combination_count,
        selected_weights=frozen.selected_weights,
        validation_metrics=frozen.validation_metrics,
    )


def tune_hybrid_weights(
    validation_tasks_path: str | Path,
    ground_truth_path: str | Path,
    feature_store: _HybridComponentProvider,
    output_path: str | Path,
    *,
    initial_weights: HybridWeights,
    feature_sources_sha256: dict[str, str],
    step: float = 0.1,
    force: bool = False,
) -> HybridTuneResult:
    """Select Hybrid weights using validation tasks and freeze one artifact."""

    tasks_path = Path(validation_tasks_path)
    truth_path = Path(ground_truth_path)
    output = Path(output_path)
    tasks = _load_validation_tasks(tasks_path)
    targets, validation_truth_sha256 = _load_validation_truth(truth_path, tasks)
    combinations = enumerate_hybrid_weights(step)
    validation_tasks_sha256 = _sha256_file(tasks_path)
    if not feature_sources_sha256:
        raise ValueError("feature_sources_sha256 cannot be empty")

    if output.is_file() and not force:
        try:
            frozen = FrozenHybridWeights.model_validate_json(
                output.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise HybridTuningError(
                f"Existing frozen Hybrid weights are invalid: {output}"
            ) from exc
        if (
            frozen.format_version != 1
            or frozen.validation_tasks_sha256 != validation_tasks_sha256
            or frozen.validation_truth_sha256 != validation_truth_sha256
            or frozen.feature_sources_sha256 != feature_sources_sha256
            or not math.isclose(frozen.tuning_step, step, abs_tol=1e-12)
            or frozen.initial_weights != initial_weights
            or frozen.combination_count != len(combinations)
            or frozen.tie_break_policy != TIE_BREAK_POLICY
        ):
            raise HybridTuningError(
                "Existing Hybrid weights do not match validation inputs or config; "
                "use force=True to rebuild"
            )
        return _result("skipped", output, frozen)

    components, target_positions, tie_precedes_target = (
        _precompute_components(tasks, targets, feature_store)
    )
    best_weights: HybridWeights | None = None
    best_metrics: HybridValidationMetrics | None = None
    best_key: tuple[float, ...] | None = None
    for weights in combinations:
        metrics = _metrics_for_weights(
            components,
            target_positions,
            tie_precedes_target,
            weights,
        )
        selection_key = _selection_key(metrics, weights, initial_weights)
        if best_key is None or selection_key > best_key:
            best_weights = weights
            best_metrics = metrics
            best_key = selection_key
    if best_weights is None or best_metrics is None:
        raise HybridTuningError("Hybrid tuning produced no weight candidate")

    frozen = FrozenHybridWeights(
        format_version=1,
        validation_tasks_sha256=validation_tasks_sha256,
        validation_truth_sha256=validation_truth_sha256,
        feature_sources_sha256=feature_sources_sha256,
        tuning_step=step,
        task_count=len(tasks),
        combination_count=len(combinations),
        initial_weights=initial_weights,
        selected_weights=best_weights,
        validation_metrics=best_metrics,
        tie_break_policy=TIE_BREAK_POLICY,
    )
    write_json_artifact(output, frozen)
    return _result("written", output, frozen)
