"""Build label-isolated, task-level inputs for Step 19 confidence calibration."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.config import EvaluationDataUsageConfig
from yelp_agent.evaluation.data_usage import assign_user_fold
from yelp_agent.models import StrictModel

from .calibration import CalibrationBatch
from .schema import CALIBRATION_FEATURE_NAMES


CALIBRATION_EXAMPLE_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("fold", pa.int32(), nullable=False),
        *(pa.field(name, pa.float64(), nullable=False) for name in CALIBRATION_FEATURE_NAMES),
        pa.field("top1_correct", pa.bool_(), nullable=False),
        pa.field("target_retrieved", pa.bool_(), nullable=False),
    ]
)


class CalibrationDatasetBuildResult(StrictModel):
    output_path: str
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    positive_count: int = Field(ge=1)
    target_retrieved_count: int = Field(ge=1)
    fold_counts: dict[str, int]
    target_business_fields_available: bool = False
    test_tasks_available: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    return value


def _task_feature_query() -> str:
    return """
        WITH feature_rows AS (
            SELECT
                task_id, user_id, business_id,
                quality_score, category_score, text_score, location_score,
                item_knn_positive_score, item_knn_missing,
                user_history_length_log, user_profile_reliability,
                user_category_novelty, route_coverage
            FROM read_parquet(?)
            WHERE split = 'validation'
        ), prediction_rows AS (
            SELECT
                task_id, business_id, rank, model_rank, v1_rank,
                model_score, hybrid_v1_score, blend_score
            FROM read_parquet(?)
        ), truth_rows AS (
            SELECT task_id, target_business_id
            FROM read_parquet(?)
            WHERE starts_with(task_id, 'validation:')
        ), top_rows AS (
            SELECT
                p.task_id,
                max(f.user_id) AS user_id,
                max(CASE WHEN p.rank = 1 THEN p.business_id END) AS top1_business_id,
                max(CASE WHEN p.rank = 1 THEN p.blend_score END) AS top1_blend_score,
                max(CASE WHEN p.rank = 2 THEN p.blend_score END) AS top2_blend_score,
                max(CASE WHEN p.rank = 1 THEN p.model_score END) AS top1_model_score,
                max(CASE WHEN p.rank = 2 THEN p.model_score END) AS top2_model_score,
                max(CASE WHEN p.rank = 1 THEN p.hybrid_v1_score END) AS top1_v1_score,
                max(CASE WHEN p.rank = 2 THEN p.hybrid_v1_score END) AS top2_v1_score,
                max(CASE WHEN p.rank = 1 THEN p.model_rank END) AS top1_model_rank,
                max(CASE WHEN p.rank = 1 THEN p.v1_rank END) AS top1_v1_rank,
                max(CASE WHEN p.rank = 1 THEN f.quality_score END) AS top1_quality,
                max(CASE WHEN p.rank = 1 THEN f.category_score END) AS top1_category,
                max(CASE WHEN p.rank = 1 THEN f.text_score END) AS top1_text,
                max(CASE WHEN p.rank = 1 THEN f.location_score END) AS top1_location,
                max(CASE WHEN p.rank = 1 THEN f.item_knn_positive_score END) AS top1_item,
                max(CASE WHEN p.rank = 1 THEN f.item_knn_missing END) AS top1_item_missing,
                max(CASE WHEN p.rank = 1 THEN f.user_history_length_log END) AS history_log,
                max(CASE WHEN p.rank = 1 THEN f.user_profile_reliability END) AS profile_reliability,
                max(CASE WHEN p.rank = 1 THEN f.user_category_novelty END) AS category_novelty,
                max(CASE WHEN p.rank = 1 THEN f.route_coverage END) AS top1_route_coverage,
                count(*) AS candidate_count
            FROM prediction_rows p
            INNER JOIN feature_rows f USING (task_id, business_id)
            GROUP BY p.task_id
        ), task_maxima AS (
            SELECT
                f.task_id,
                max(f.quality_score) AS max_quality,
                max(f.category_score) AS max_category,
                max(f.text_score) AS max_text,
                max(f.location_score) AS max_location,
                max(f.item_knn_positive_score) AS max_item,
                max(CASE WHEN f.item_knn_missing < 0.5 THEN 1 ELSE 0 END) AS item_available,
                max(CASE WHEN f.business_id = g.target_business_id THEN 1 ELSE 0 END) AS target_retrieved,
                max(g.target_business_id) AS target_business_id
            FROM feature_rows f
            INNER JOIN truth_rows g USING (task_id)
            GROUP BY f.task_id
        )
        SELECT
            t.task_id,
            t.user_id,
            CAST(t.top1_blend_score AS DOUBLE) AS top1_blend_score,
            CAST(t.top1_blend_score - t.top2_blend_score AS DOUBLE) AS blend_score_margin,
            CAST(t.top1_model_score - t.top2_model_score AS DOUBLE) AS model_score_margin,
            CAST(t.top1_v1_score - t.top2_v1_score AS DOUBLE) AS hybrid_v1_score_margin,
            CAST(abs(t.top1_model_rank - t.top1_v1_rank) / greatest(t.candidate_count - 1, 1) AS DOUBLE)
                AS model_v1_rank_disagreement,
            CAST((
                CASE WHEN t.top1_quality + 1e-12 < m.max_quality THEN 1 ELSE 0 END
                + CASE WHEN t.top1_category + 1e-12 < m.max_category THEN 1 ELSE 0 END
                + CASE WHEN t.top1_text + 1e-12 < m.max_text THEN 1 ELSE 0 END
                + CASE WHEN t.top1_location + 1e-12 < m.max_location THEN 1 ELSE 0 END
                + CASE WHEN m.item_available = 1 AND t.top1_item + 1e-12 < m.max_item THEN 1 ELSE 0 END
            ) / CAST(4 + m.item_available AS DOUBLE) AS DOUBLE) AS component_disagreement,
            CAST(t.top1_item AS DOUBLE) AS top1_item_knn_positive_score,
            CAST(t.top1_item_missing AS DOUBLE) AS top1_item_knn_missing,
            CAST(t.history_log AS DOUBLE) AS user_history_length_log,
            CAST(t.profile_reliability AS DOUBLE) AS user_profile_reliability,
            CAST(t.category_novelty AS DOUBLE) AS top1_user_category_novelty,
            CAST(t.top1_route_coverage AS DOUBLE) AS top1_route_coverage,
            CAST(t.top1_business_id = m.target_business_id AS BOOLEAN) AS top1_correct,
            CAST(m.target_retrieved = 1 AS BOOLEAN) AS target_retrieved
        FROM top_rows t
        INNER JOIN task_maxima m USING (task_id)
        ORDER BY t.task_id
    """


def build_calibration_dataset(
    *,
    validation_features_path: str | Path,
    validation_predictions_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path,
    evaluation_policy: EvaluationDataUsageConfig,
) -> CalibrationDatasetBuildResult:
    """Publish target-blind runtime signals plus a separately derived fit label."""

    features = _required(validation_features_path, "Validation ranking features")
    predictions = _required(
        validation_predictions_path,
        "Validation ranking predictions",
    )
    truth = _required(ground_truth_path, "Ground truth")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Calibration dataset already exists: {output}")
    connection = duckdb.connect()
    try:
        table = connection.execute(
            _task_feature_query(),
            [str(features), str(predictions), str(truth)],
        ).to_arrow_table()
    finally:
        connection.close()
    if len(table) == 0:
        raise ValueError("Calibration dataset query returned no tasks")
    task_ids = [str(value) for value in table["task_id"].to_pylist()]
    user_ids = [str(value) for value in table["user_id"].to_pylist()]
    if any(not task_id.startswith("validation:") for task_id in task_ids):
        raise ValueError("Calibration dataset may contain Validation tasks only")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Calibration dataset contains duplicate tasks")
    folds = [assign_user_fold(user_id, evaluation_policy) for user_id in user_ids]
    arrays = [
        table["task_id"],
        table["user_id"],
        pa.array(folds, type=pa.int32()),
        *(table[name].cast(pa.float64()) for name in CALIBRATION_FEATURE_NAMES),
        table["top1_correct"].cast(pa.bool_()),
        table["target_retrieved"].cast(pa.bool_()),
    ]
    output_table = pa.Table.from_arrays(arrays, schema=CALIBRATION_EXAMPLE_SCHEMA)
    positive_count = int(sum(bool(value) for value in table["top1_correct"].to_pylist()))
    retrieved_count = int(
        sum(bool(value) for value in table["target_retrieved"].to_pylist())
    )
    if positive_count == 0 or retrieved_count == 0:
        raise ValueError("Calibration dataset has no positive outcomes")
    fold_counts = {
        str(fold): folds.count(fold) for fold in sorted(set(folds))
    }
    if len(fold_counts) != evaluation_policy.cross_validation.folds:
        raise ValueError("Calibration dataset does not cover every configured fold")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        pq.write_table(output_table, partial, compression="zstd")
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return CalibrationDatasetBuildResult(
        output_path=str(output),
        output_sha256=_sha256(output),
        task_count=len(output_table),
        positive_count=positive_count,
        target_retrieved_count=retrieved_count,
        fold_counts=fold_counts,
    )


def load_calibration_batch(path: str | Path) -> CalibrationBatch:
    source = _required(path, "Calibration dataset")
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow != CALIBRATION_EXAMPLE_SCHEMA:
        raise ValueError("Calibration dataset schema does not match Step 19")
    table = parquet.read()
    return CalibrationBatch(
        task_ids=tuple(str(value) for value in table["task_id"].to_pylist()),
        folds=table["fold"].to_numpy(zero_copy_only=False),
        features=np.column_stack(
            [
                table[name].to_numpy(zero_copy_only=False)
                for name in CALIBRATION_FEATURE_NAMES
            ]
        ),
        labels=table["top1_correct"].to_numpy(zero_copy_only=False),
    )
