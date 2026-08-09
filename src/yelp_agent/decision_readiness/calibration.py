"""Cross-validated Top-1 calibration hidden behind one runtime estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .schema import (
    CALIBRATION_FEATURE_NAMES,
    ISOTONIC_INPUT_FEATURE,
    CalibrationMetrics,
    CalibratorKind,
    CoverageRiskPoint,
    RankingConfidenceEstimate,
    RankingSignals,
    RankingUncertaintyReason,
    ReliabilityBin,
    UncertaintyThresholds,
)


@dataclass(frozen=True, slots=True)
class CalibrationBatch:
    """Aligned, user-folded task features and the isolated Top-1 label."""

    task_ids: tuple[str, ...]
    folds: np.ndarray
    features: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        folds = np.asarray(self.folds, dtype=np.int32)
        features = np.asarray(self.features, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int8)
        row_count = len(self.task_ids)
        if row_count < 2 or len(set(self.task_ids)) != row_count:
            raise ValueError("calibration task IDs must be unique and nonempty")
        if folds.shape != (row_count,) or len(set(folds.tolist())) < 2:
            raise ValueError("calibration requires two or more aligned folds")
        if features.shape != (row_count, len(CALIBRATION_FEATURE_NAMES)):
            raise ValueError("calibration features do not match the frozen feature order")
        if labels.shape != (row_count,) or set(labels.tolist()) != {0, 1}:
            raise ValueError("calibration labels must contain both Top-1 outcomes")
        if not np.all(np.isfinite(features)):
            raise ValueError("calibration features must be finite")
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)


class ConfidenceModel(Protocol):
    kind: CalibratorKind
    feature_names: tuple[str, ...]

    def predict_probability(self, features: np.ndarray) -> np.ndarray: ...


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True, slots=True)
class LogisticConfidenceModel:
    kind: CalibratorKind
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def __post_init__(self) -> None:
        count = len(self.feature_names)
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if self.kind != "logistic" or self.feature_names != CALIBRATION_FEATURE_NAMES:
            raise ValueError("logistic model must use the frozen calibration features")
        if mean.shape != (count,) or scale.shape != (count,) or coefficients.shape != (
            count,
        ):
            raise ValueError("logistic model vectors do not match its features")
        if (
            not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or not np.all(np.isfinite(coefficients))
            or not math.isfinite(self.intercept)
            or np.any(scale <= 0)
        ):
            raise ValueError("logistic model parameters must be finite")
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("runtime calibration features have an invalid shape")
        standardized = (matrix - self.feature_mean) / self.feature_scale
        return _sigmoid(standardized @ self.coefficients + self.intercept)


@dataclass(frozen=True, slots=True)
class IsotonicConfidenceModel:
    kind: CalibratorKind
    feature_names: tuple[str, ...]
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x_thresholds, dtype=np.float64)
        y = np.asarray(self.y_thresholds, dtype=np.float64)
        if self.kind != "isotonic" or self.feature_names != (ISOTONIC_INPUT_FEATURE,):
            raise ValueError("isotonic model must use only the frozen margin feature")
        if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
            raise ValueError("isotonic thresholds must be aligned vectors")
        if (
            not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or np.any(np.diff(x) < 0)
            or np.any(np.diff(y) < 0)
            or np.any(y < 0)
            or np.any(y > 1)
        ):
            raise ValueError("isotonic thresholds must be finite and monotonic")
        object.__setattr__(self, "x_thresholds", x)
        object.__setattr__(self, "y_thresholds", y)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(CALIBRATION_FEATURE_NAMES):
            raise ValueError("runtime calibration features have an invalid shape")
        position = CALIBRATION_FEATURE_NAMES.index(ISOTONIC_INPUT_FEATURE)
        return np.interp(
            matrix[:, position],
            self.x_thresholds,
            self.y_thresholds,
            left=float(self.y_thresholds[0]),
            right=float(self.y_thresholds[-1]),
        )


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    random_seed: int,
    maximum_iterations: int,
) -> LogisticConfidenceModel:
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    estimator = LogisticRegression(
        random_state=random_seed,
        max_iter=maximum_iterations,
        solver="lbfgs",
    ).fit(standardized, labels)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale == 0] = 1.0
    return LogisticConfidenceModel(
        kind="logistic",
        feature_names=CALIBRATION_FEATURE_NAMES,
        feature_mean=np.asarray(scaler.mean_, dtype=np.float64),
        feature_scale=scale,
        coefficients=np.asarray(estimator.coef_[0], dtype=np.float64),
        intercept=float(estimator.intercept_[0]),
    )


def _fit_isotonic(features: np.ndarray, labels: np.ndarray) -> IsotonicConfidenceModel:
    position = CALIBRATION_FEATURE_NAMES.index(ISOTONIC_INPUT_FEATURE)
    estimator = IsotonicRegression(out_of_bounds="clip").fit(
        features[:, position],
        labels,
    )
    return IsotonicConfidenceModel(
        kind="isotonic",
        feature_names=(ISOTONIC_INPUT_FEATURE,),
        x_thresholds=np.asarray(estimator.X_thresholds_, dtype=np.float64),
        y_thresholds=np.asarray(estimator.y_thresholds_, dtype=np.float64),
    )


def _fit_model(
    kind: CalibratorKind,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    random_seed: int,
    maximum_iterations: int,
) -> ConfidenceModel:
    if kind == "logistic":
        return _fit_logistic(
            features,
            labels,
            random_seed=random_seed,
            maximum_iterations=maximum_iterations,
        )
    return _fit_isotonic(features, labels)


def _reliability_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bin_count: int,
) -> tuple[list[ReliabilityBin], float]:
    values: list[ReliabilityBin] = []
    ece = 0.0
    order = np.argsort(probabilities, kind="stable")
    for selected_indices in np.array_split(order, bin_count):
        count = len(selected_indices)
        selected_probabilities = probabilities[selected_indices]
        selected_labels = labels[selected_indices]
        mean_confidence = (
            float(selected_probabilities.mean()) if count else None
        )
        accuracy = float(selected_labels.mean()) if count else None
        if count and mean_confidence is not None and accuracy is not None:
            ece += (count / len(labels)) * abs(accuracy - mean_confidence)
        values.append(
            ReliabilityBin(
                lower_bound=(
                    float(selected_probabilities.min()) if count else 0.0
                ),
                upper_bound=(
                    float(selected_probabilities.max()) if count else 0.0
                ),
                task_count=count,
                mean_confidence=mean_confidence,
                observed_accuracy=accuracy,
            )
        )
    return values, ece


def _coverage_risk(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    coverage_points: tuple[float, ...],
) -> list[CoverageRiskPoint]:
    order = np.argsort(-probabilities, kind="stable")
    values: list[CoverageRiskPoint] = []
    for requested in coverage_points:
        count = max(1, int(math.ceil(len(labels) * requested)))
        selected = order[:count]
        accuracy = float(labels[selected].mean())
        values.append(
            CoverageRiskPoint(
                requested_coverage=requested,
                actual_coverage=count / len(labels),
                selected_task_count=count,
                minimum_confidence=float(probabilities[selected].min()),
                observed_accuracy=accuracy,
                risk=1.0 - accuracy,
            )
        )
    return values


def calibration_metrics(
    kind: CalibratorKind,
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    ece_bin_count: int,
    coverage_points: tuple[float, ...],
) -> CalibrationMetrics:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    outcomes = np.asarray(labels, dtype=np.int8)
    if values.shape != outcomes.shape or values.ndim != 1 or len(values) == 0:
        raise ValueError("calibration metrics require aligned nonempty vectors")
    bins, ece = _reliability_bins(values, outcomes, bin_count=ece_bin_count)
    positive_count = int(outcomes.sum())
    return CalibrationMetrics(
        method=kind,
        task_count=len(outcomes),
        positive_count=positive_count,
        positive_rate=positive_count / len(outcomes),
        brier_score=float(np.mean((values - outcomes) ** 2)),
        expected_calibration_error=ece,
        reliability_bins=bins,
        coverage_risk=_coverage_risk(
            values,
            outcomes,
            coverage_points=coverage_points,
        ),
    )


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    selected_kind: CalibratorKind
    selected_model: ConfidenceModel
    candidate_metrics: dict[CalibratorKind, CalibrationMetrics]
    out_of_fold_probabilities: dict[CalibratorKind, np.ndarray]


def select_confidence_calibrator(
    batch: CalibrationBatch,
    *,
    random_seed: int,
    maximum_iterations: int,
    ece_bin_count: int,
    coverage_points: tuple[float, ...],
) -> CalibrationSelection:
    """Select by user-fold OOF Brier, then ECE, without touching Test."""

    predictions: dict[CalibratorKind, np.ndarray] = {
        "logistic": np.zeros(len(batch.labels), dtype=np.float64),
        "isotonic": np.zeros(len(batch.labels), dtype=np.float64),
    }
    for fold in sorted(set(batch.folds.tolist())):
        held_out = batch.folds == fold
        training = ~held_out
        if set(batch.labels[training].tolist()) != {0, 1}:
            raise ValueError(f"calibration training fold complement {fold} lacks a class")
        for kind in ("logistic", "isotonic"):
            model = _fit_model(
                kind,
                batch.features[training],
                batch.labels[training],
                random_seed=random_seed,
                maximum_iterations=maximum_iterations,
            )
            predictions[kind][held_out] = model.predict_probability(
                batch.features[held_out]
            )
    metrics = {
        kind: calibration_metrics(
            kind,
            probabilities,
            batch.labels,
            ece_bin_count=ece_bin_count,
            coverage_points=coverage_points,
        )
        for kind, probabilities in predictions.items()
    }
    selected_kind: CalibratorKind = min(
        metrics,
        key=lambda kind: (
            metrics[kind].brier_score,
            metrics[kind].expected_calibration_error,
            0 if kind == "logistic" else 1,
        ),
    )
    selected_model = _fit_model(
        selected_kind,
        batch.features,
        batch.labels,
        random_seed=random_seed,
        maximum_iterations=maximum_iterations,
    )
    return CalibrationSelection(
        selected_kind=selected_kind,
        selected_model=selected_model,
        candidate_metrics=metrics,
        out_of_fold_probabilities=predictions,
    )


def derive_uncertainty_thresholds(
    batch: CalibrationBatch,
    *,
    sparse_history_count_max: int,
    unseen_category_min: float,
    feature_disagreement_min: float,
    small_top_margin_quantile: float,
    weak_collaborative_support_quantile: float,
    low_profile_reliability_max: float,
) -> UncertaintyThresholds:
    by_name = {
        name: batch.features[:, index]
        for index, name in enumerate(CALIBRATION_FEATURE_NAMES)
    }
    available_collaborative = by_name["top1_item_knn_missing"] < 0.5
    collaborative_scores = by_name["top1_item_knn_positive_score"][
        available_collaborative
    ]
    weak_threshold = (
        float(np.quantile(collaborative_scores, weak_collaborative_support_quantile))
        if len(collaborative_scores)
        else 0.0
    )
    return UncertaintyThresholds(
        sparse_history_log_max=math.log1p(sparse_history_count_max),
        unseen_category_min=unseen_category_min,
        feature_disagreement_min=feature_disagreement_min,
        small_top_margin_max=float(
            np.quantile(
                by_name["blend_score_margin"],
                small_top_margin_quantile,
            )
        ),
        weak_collaborative_support_max=weak_threshold,
        low_profile_reliability_max=low_profile_reliability_max,
    )


def uncertainty_reasons(
    signals: RankingSignals,
    thresholds: UncertaintyThresholds,
) -> list[RankingUncertaintyReason]:
    reasons: list[RankingUncertaintyReason] = []
    if signals.user_history_length_log <= thresholds.sparse_history_log_max:
        reasons.append("sparse_history")
    if signals.top1_user_category_novelty >= thresholds.unseen_category_min:
        reasons.append("unseen_category")
    if signals.component_disagreement >= thresholds.feature_disagreement_min:
        reasons.append("feature_disagreement")
    if signals.blend_score_margin <= thresholds.small_top_margin_max:
        reasons.append("small_top_margin")
    if (
        signals.top1_item_knn_missing >= 0.5
        or signals.top1_item_knn_positive_score
        <= thresholds.weak_collaborative_support_max
    ):
        reasons.append("weak_collaborative_support")
    if signals.user_profile_reliability <= thresholds.low_profile_reliability_max:
        reasons.append("low_profile_reliability")
    return reasons


@dataclass(frozen=True, slots=True)
class FrozenConfidenceCalibrator:
    """The only calibration interface needed by online request analysis."""

    model: ConfidenceModel
    thresholds: UncertaintyThresholds
    version: str

    def estimate(self, signals: RankingSignals) -> RankingConfidenceEstimate:
        values = signals.as_feature_dict()
        matrix = np.asarray(
            [[values[name] for name in CALIBRATION_FEATURE_NAMES]],
            dtype=np.float64,
        )
        probability = float(self.model.predict_probability(matrix)[0])
        return RankingConfidenceEstimate(
            probability_top1_correct=probability,
            target="hybrid_v2_b_next_business_top1",
            calibrator_kind=self.model.kind,
            calibrator_version=self.version,
            uncertainty_reasons=uncertainty_reasons(signals, self.thresholds),
        )
