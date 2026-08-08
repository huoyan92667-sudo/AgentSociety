"""Ground-truth-isolated, deterministic negative selection for training only."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.config import HybridV2Config
from yelp_agent.models import StrictModel
from yelp_agent.retrieval.benchmark import CANDIDATE_SCHEMA

TRAINING_SELECTION_SCHEMA = pa.schema(
    [
        *CANDIDATE_SCHEMA,
        pa.field("label", pa.bool_(), nullable=False),
        pa.field("negative_kind", pa.string(), nullable=False),
        pa.field("selection_order", pa.int32(), nullable=False),
    ]
)


class TrainingSelectionError(RuntimeError):
    """Raised when frozen candidates cannot produce safe pairwise examples."""


class TrainingSelectionResult(StrictModel):
    status: Literal["written"] = "written"
    output_path: str
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_tasks: int = Field(ge=1)
    selected_tasks: int = Field(ge=1)
    target_not_retrieved_tasks: int = Field(ge=0)
    selected_rows: int = Field(ge=1)
    negatives_per_task: int = Field(ge=1)


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


def build_training_selection(
    candidates_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path,
    config: HybridV2Config,
) -> TrainingSelectionResult:
    """Select one retrieved target and exactly 20 unique negatives per task."""

    candidates = _required(candidates_path, "Training candidates")
    truth = _required(ground_truth_path, "Training ground truth")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Training selection already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    hard = config.hard_negative_count
    similar = config.profile_similar_negative_count
    random_count = config.random_negative_count
    seed = config.random_seed

    try:
        with duckdb.connect() as connection:
            duplicate_truth, source_tasks = connection.execute(
                """
                SELECT count(*) - count(DISTINCT task_id), count(DISTINCT task_id)
                FROM read_parquet(?)
                """,
                [str(truth)],
            ).fetchone()
            if int(duplicate_truth) != 0 or int(source_tasks) == 0:
                raise TrainingSelectionError(
                    "Training ground truth must contain one row per task"
                )
            duplicate_candidates = connection.execute(
                """
                SELECT count(*) - count(DISTINCT task_id || chr(0) || business_id)
                FROM read_parquet(?)
                """,
                [str(candidates)],
            ).fetchone()[0]
            if int(duplicate_candidates) != 0:
                raise TrainingSelectionError(
                    "Training candidates contain duplicate task-business rows"
                )
            hit_tasks, insufficient = connection.execute(
                """
                WITH truth AS (
                    SELECT task_id, target_business_id FROM read_parquet(?)
                ), counts AS (
                    SELECT
                        truth.task_id,
                        count(candidate.business_id) AS candidate_count,
                        count(*) FILTER (
                            WHERE candidate.business_id = truth.target_business_id
                        ) AS target_count
                    FROM truth
                    LEFT JOIN read_parquet(?) AS candidate USING (task_id)
                    GROUP BY truth.task_id
                )
                SELECT
                    count(*) FILTER (WHERE target_count = 1),
                    count(*) FILTER (
                        WHERE target_count = 1 AND candidate_count < ?
                    )
                FROM counts
                """,
                [str(truth), str(candidates), config.negative_count + 1],
            ).fetchone()
            if int(insufficient) != 0:
                raise TrainingSelectionError(
                    "A target-retrieved task has too few unique negatives"
                )
            if int(hit_tasks) == 0:
                raise TrainingSelectionError(
                    "No training target appears in the frozen candidates"
                )

            query = f"""
                WITH truth AS (
                    SELECT task_id, target_business_id FROM read_parquet(?)
                ), eligible AS (
                    SELECT truth.task_id, truth.target_business_id
                    FROM truth
                    JOIN read_parquet(?) AS candidate
                      ON candidate.task_id = truth.task_id
                     AND candidate.business_id = truth.target_business_id
                ), negatives AS (
                    SELECT candidate.*
                    FROM read_parquet(?) AS candidate
                    JOIN eligible USING (task_id)
                    WHERE candidate.business_id <> eligible.target_business_id
                ), hard AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY task_id ORDER BY rank, business_id
                    ) AS bucket_order
                    FROM negatives
                    QUALIFY bucket_order <= {hard}
                ), profile_similar AS (
                    SELECT negative.*, row_number() OVER (
                        PARTITION BY negative.task_id
                        ORDER BY
                            coalesce(negative.category_score, 0.0) DESC,
                            coalesce(negative.text_score, 0.0) DESC,
                            negative.rank,
                            negative.business_id
                    ) AS bucket_order
                    FROM negatives AS negative
                    ANTI JOIN hard
                      ON hard.task_id = negative.task_id
                     AND hard.business_id = negative.business_id
                    QUALIFY bucket_order <= {similar}
                ), random_selected AS (
                    SELECT negative.*, row_number() OVER (
                        PARTITION BY negative.task_id
                        ORDER BY sha256(
                            '{seed}' || chr(0) || negative.task_id
                            || chr(0) || negative.business_id
                        ), negative.business_id
                    ) AS bucket_order
                    FROM negatives AS negative
                    ANTI JOIN hard
                      ON hard.task_id = negative.task_id
                     AND hard.business_id = negative.business_id
                    ANTI JOIN profile_similar
                      ON profile_similar.task_id = negative.task_id
                     AND profile_similar.business_id = negative.business_id
                    QUALIFY bucket_order <= {random_count}
                ), chosen AS (
                    SELECT
                        candidate.*,
                        true AS label,
                        'target' AS negative_kind,
                        0 AS selection_order
                    FROM read_parquet(?) AS candidate
                    JOIN eligible
                      ON eligible.task_id = candidate.task_id
                     AND eligible.target_business_id = candidate.business_id
                    UNION ALL BY NAME
                    SELECT
                        hard.* EXCLUDE (bucket_order),
                        false AS label,
                        'hard' AS negative_kind,
                        hard.bucket_order AS selection_order
                    FROM hard
                    UNION ALL BY NAME
                    SELECT
                        profile_similar.* EXCLUDE (bucket_order),
                        false AS label,
                        'profile_similar' AS negative_kind,
                        {hard} + profile_similar.bucket_order AS selection_order
                    FROM profile_similar
                    UNION ALL BY NAME
                    SELECT
                        random_selected.* EXCLUDE (bucket_order),
                        false AS label,
                        'random' AS negative_kind,
                        {hard + similar}
                            + random_selected.bucket_order AS selection_order
                    FROM random_selected
                )
                SELECT * FROM chosen
                ORDER BY task_id, selection_order, business_id
            """
            reader = connection.execute(
                query,
                [
                    str(truth),
                    str(candidates),
                    str(candidates),
                    str(candidates),
                ],
            ).arrow()
            table = reader.read_all().cast(TRAINING_SELECTION_SCHEMA)
    except duckdb.Error as exc:
        raise TrainingSelectionError(
            f"Could not construct deterministic training negatives: {exc}"
        ) from exc

    expected_rows = int(hit_tasks) * (config.negative_count + 1)
    if table.num_rows != expected_rows:
        raise TrainingSelectionError(
            "Training selection did not produce the configured row count"
        )
    pq.write_table(
        table,
        partial,
        compression="zstd",
        use_dictionary=["task_id", "business_id", "negative_kind"],
    )
    os.replace(partial, output)
    return TrainingSelectionResult(
        output_path=str(output),
        output_sha256=_sha256(output),
        source_tasks=int(source_tasks),
        selected_tasks=int(hit_tasks),
        target_not_retrieved_tasks=int(source_tasks) - int(hit_tasks),
        selected_rows=table.num_rows,
        negatives_per_task=config.negative_count,
    )
