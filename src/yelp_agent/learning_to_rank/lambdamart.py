"""Deterministic grouped LambdaMART learning behind one ranking interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class LambdaMARTParameters:
    """One named, bounded challenger configuration."""

    name: str
    num_leaves: int
    learning_rate: float
    num_boost_round: int
    min_child_samples: int
    reg_lambda: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LambdaMART parameter name cannot be empty")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least two")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_boost_round < 1:
            raise ValueError("num_boost_round must be positive")
        if self.min_child_samples < 1:
            raise ValueError("min_child_samples must be positive")
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda cannot be negative")


@dataclass(frozen=True, slots=True)
class LambdaMARTTrainingBatch:
    """Contiguous ranking groups with one relevant business per task."""

    feature_names: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray
    group_sizes: np.ndarray
    sample_weights: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int8)
        groups = np.asarray(self.group_sizes, dtype=np.int32)
        weights = np.asarray(self.sample_weights, dtype=np.float64)
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("feature_names must be nonempty and unique")
        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != len(self.feature_names)
        ):
            raise ValueError("training features do not match feature_names")
        if labels.shape != (len(features),) or not set(labels.tolist()) <= {0, 1}:
            raise ValueError("labels must be an aligned binary vector")
        if groups.ndim != 1 or len(groups) == 0 or np.any(groups < 2):
            raise ValueError("group_sizes must contain ranking groups of size two+")
        if int(groups.sum()) != len(features):
            raise ValueError("group_sizes must cover every training row")
        if weights.shape != (len(features),) or np.any(weights <= 0):
            raise ValueError("sample_weights must be positive and row-aligned")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(weights)):
            raise ValueError("LambdaMART training data must be finite")
        offset = 0
        for size in groups:
            group_labels = labels[offset : offset + int(size)]
            if int(group_labels.sum()) != 1:
                raise ValueError("every ranking group must contain one target")
            offset += int(size)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "group_sizes", groups)
        object.__setattr__(self, "sample_weights", weights)


@dataclass(frozen=True, slots=True)
class LambdaMARTModel:
    """Frozen nonlinear scorer with an immutable feature order."""

    feature_names: tuple[str, ...]
    parameters: LambdaMARTParameters
    random_seed: int
    booster: lgb.Booster

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
        return np.asarray(
            self.booster.predict(
                matrix,
                num_iteration=self.parameters.num_boost_round,
            ),
            dtype=np.float64,
        )

    def feature_importance(self, *, importance_type: str) -> dict[str, float]:
        """Return diagnostic importance; callers must not treat it as causal."""

        if importance_type not in {"gain", "split"}:
            raise ValueError("importance_type must be gain or split")
        values = self.booster.feature_importance(importance_type=importance_type)
        return {
            name: float(value)
            for name, value in zip(self.feature_names, values, strict=True)
        }


def load_lambdamart_training_batch(
    features_path: str | Path,
    *,
    feature_names: tuple[str, ...],
    excluded_fold: int | None = None,
) -> LambdaMARTTrainingBatch:
    """Load task-contiguous rows without exposing grouping details to callers."""

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

    fold_filter = "" if excluded_fold is None else "WHERE fold <> ?"
    parameters: list[object] = [str(path)]
    if excluded_fold is not None:
        parameters.append(excluded_fold)
    selected = ", ".join(quoted(name) for name in feature_names)
    query = f"""
        SELECT task_id, business_id, label, sample_weight, {selected}
        FROM read_parquet(?)
        {fold_filter}
        ORDER BY task_id, label DESC, business_id
    """
    with duckdb.connect() as connection:
        table = connection.execute(query, parameters).to_arrow_table()
    rows = table.num_rows
    if rows == 0:
        raise ValueError("LambdaMART training population is empty")
    task_ids = table["task_id"].to_pylist()
    group_sizes: list[int] = []
    previous = task_ids[0]
    current_size = 0
    for task_id in task_ids:
        if task_id != previous:
            group_sizes.append(current_size)
            previous = task_id
            current_size = 0
        current_size += 1
    group_sizes.append(current_size)
    features = np.column_stack(
        [table[name].to_numpy(zero_copy_only=False) for name in feature_names]
    ).astype(np.float64, copy=False)
    labels = table["label"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    weights = (
        table["sample_weight"]
        .to_numpy(zero_copy_only=False)
        .astype(np.float64, copy=False)
    )
    return LambdaMARTTrainingBatch(
        feature_names=feature_names,
        features=features,
        labels=labels,
        group_sizes=np.asarray(group_sizes, dtype=np.int32),
        sample_weights=weights,
    )


def train_lambdamart(
    batch: LambdaMARTTrainingBatch,
    *,
    parameters: LambdaMARTParameters,
    random_seed: int,
) -> LambdaMARTModel:
    """Fit one deterministic LambdaRank challenger on frozen ranking groups."""

    dataset = lgb.Dataset(
        batch.features,
        label=batch.labels,
        group=batch.group_sizes,
        weight=batch.sample_weights,
        feature_name=list(batch.feature_names),
        free_raw_data=False,
    )
    booster = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "None",
            "label_gain": [0, 1],
            "num_leaves": parameters.num_leaves,
            "learning_rate": parameters.learning_rate,
            "min_data_in_leaf": parameters.min_child_samples,
            "lambda_l2": parameters.reg_lambda,
            "seed": int(random_seed),
            "feature_fraction_seed": int(random_seed),
            "bagging_seed": int(random_seed),
            "data_random_seed": int(random_seed),
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=parameters.num_boost_round,
    )
    return LambdaMARTModel(
        feature_names=batch.feature_names,
        parameters=parameters,
        random_seed=int(random_seed),
        booster=booster,
    )
