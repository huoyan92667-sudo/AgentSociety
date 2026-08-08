"""Deterministic pairwise Logistic learning behind a small ranking seam."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True, slots=True)
class PairwiseTrainingBatch:
    """Aligned positive/negative feature rows for pairwise supervision."""

    feature_names: tuple[str, ...]
    positive_features: np.ndarray
    negative_features: np.ndarray
    sample_weights: np.ndarray

    def __post_init__(self) -> None:
        positive = np.asarray(self.positive_features, dtype=np.float64)
        negative = np.asarray(self.negative_features, dtype=np.float64)
        weights = np.asarray(self.sample_weights, dtype=np.float64)
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("feature_names must be nonempty and unique")
        if positive.ndim != 2 or negative.shape != positive.shape:
            raise ValueError("positive and negative features must be aligned matrices")
        if positive.shape[0] == 0 or positive.shape[1] != len(self.feature_names):
            raise ValueError("training features do not match feature_names")
        if weights.shape != (positive.shape[0],) or np.any(weights <= 0):
            raise ValueError("sample_weights must be positive and pair-aligned")
        if not (
            np.all(np.isfinite(positive))
            and np.all(np.isfinite(negative))
            and np.all(np.isfinite(weights))
        ):
            raise ValueError("pairwise training data must be finite")
        object.__setattr__(self, "positive_features", positive)
        object.__setattr__(self, "negative_features", negative)
        object.__setattr__(self, "sample_weights", weights)


@dataclass(frozen=True, slots=True)
class PairwiseLogisticModel:
    """Frozen linear scorer learned from feature differences."""

    feature_names: tuple[str, ...]
    feature_scale: np.ndarray
    coefficients: np.ndarray
    regularization_c: float

    def __post_init__(self) -> None:
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        expected = (len(self.feature_names),)
        if scale.shape != expected or coefficients.shape != expected:
            raise ValueError("model vectors must match feature_names")
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("feature_scale must be finite and positive")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must be finite")
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)

    def score(
        self,
        features: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        """Score candidate rows; higher values are ranked first."""

        if feature_names != self.feature_names:
            raise ValueError("feature order does not match the frozen model")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("candidate feature matrix has an invalid shape")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("candidate features must be finite")
        return (matrix / self.feature_scale) @ self.coefficients


def train_pairwise_logistic(
    batch: PairwiseTrainingBatch,
    *,
    regularization_c: float,
) -> PairwiseLogisticModel:
    """Fit a symmetric pairwise Logistic model without a meaningless intercept."""

    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive")
    differences = batch.positive_features - batch.negative_features
    scale = np.sqrt(np.mean(np.square(differences), axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    symmetric = np.concatenate((differences, -differences), axis=0) / scale
    labels = np.concatenate(
        (
            np.ones(len(differences), dtype=np.int8),
            np.zeros(len(differences), dtype=np.int8),
        )
    )
    weights = np.concatenate((batch.sample_weights, batch.sample_weights))
    estimator = LogisticRegression(
        C=float(regularization_c),
        l1_ratio=0.0,
        solver="lbfgs",
        fit_intercept=False,
        max_iter=1000,
        tol=1e-8,
    )
    estimator.fit(symmetric, labels, sample_weight=weights)
    return PairwiseLogisticModel(
        feature_names=batch.feature_names,
        feature_scale=scale,
        coefficients=np.asarray(estimator.coef_[0], dtype=np.float64),
        regularization_c=float(regularization_c),
    )


def load_pairwise_training_batch(
    features_path: str | Path,
    *,
    feature_names: tuple[str, ...],
    excluded_fold: int | None = None,
) -> PairwiseTrainingBatch:
    """Join every negative to the one positive row in its frozen task."""

    path = Path(features_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training feature artifact does not exist: {path}")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be nonempty and unique")
    available = set(pq.ParquetFile(path).schema_arrow.names)
    required = {"task_id", "business_id", "label", "sample_weight", *feature_names}
    if excluded_fold is not None:
        if excluded_fold < 1:
            raise ValueError("excluded_fold must be positive")
        required.add("fold")
    if not required.issubset(available):
        raise ValueError("Training feature artifact is missing required columns")

    def quoted(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    positive_columns = ", ".join(
        f"positive.{quoted(name)} AS {quoted('positive_' + name)}"
        for name in feature_names
    )
    negative_columns = ", ".join(
        f"negative.{quoted(name)} AS {quoted('negative_' + name)}"
        for name in feature_names
    )
    with duckdb.connect() as connection:
        invalid_tasks = connection.execute(
            """
            SELECT count(*)
            FROM (
                SELECT task_id,
                       count(*) FILTER (WHERE label) AS positives,
                       count(*) FILTER (WHERE NOT label) AS negatives,
                       min(sample_weight) AS minimum_weight,
                       max(sample_weight) AS maximum_weight
                FROM read_parquet(?)
                GROUP BY task_id
                HAVING positives <> 1 OR negatives < 1
                    OR minimum_weight <= 0 OR minimum_weight <> maximum_weight
            )
            """,
            [str(path)],
        ).fetchone()[0]
        if int(invalid_tasks) != 0:
            raise ValueError(
                "Every training task must have one target, negatives, and one weight"
            )
        fold_filter = "" if excluded_fold is None else "AND negative.fold <> ?"
        query = f"""
            SELECT
                {positive_columns},
                {negative_columns},
                positive.sample_weight
            FROM read_parquet(?) AS negative
            JOIN read_parquet(?) AS positive USING (task_id)
            WHERE NOT negative.label AND positive.label
                {fold_filter}
            ORDER BY negative.task_id, negative.business_id
        """
        parameters: list[object] = [str(path), str(path)]
        if excluded_fold is not None:
            parameters.append(excluded_fold)
        table = connection.execute(query, parameters).arrow().read_all()
    positive = np.column_stack(
        [
            table[f"positive_{name}"].to_numpy(zero_copy_only=False)
            for name in feature_names
        ]
    ).astype(np.float64, copy=False)
    negative = np.column_stack(
        [
            table[f"negative_{name}"].to_numpy(zero_copy_only=False)
            for name in feature_names
        ]
    ).astype(np.float64, copy=False)
    weights = (
        table["sample_weight"]
        .to_numpy(zero_copy_only=False)
        .astype(np.float64, copy=False)
    )
    return PairwiseTrainingBatch(
        feature_names=feature_names,
        positive_features=positive,
        negative_features=negative,
        sample_weights=weights,
    )
