"""Model-agnostic candidate scoring with labels kept outside the seam."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.learning_to_rank.evaluation import (
    HybridV2RankingMetrics,
    HybridV2RankingWriteResult,
    evaluate_hybrid_v2_ranking,
    write_hybrid_v2_rankings,
)
from yelp_agent.learning_to_rank.model import PairwiseLogisticModel
from yelp_agent.models import StrictModel

SCORED_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("model_score", pa.float64(), nullable=False),
        pa.field("hybrid_v1_score", pa.float64(), nullable=False),
    ]
)


class CandidateScorer(Protocol):
    """The shared seam implemented by linear and nonlinear rankers."""

    feature_names: tuple[str, ...]

    def score(
        self,
        features: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> np.ndarray: ...


class CandidateScoreWriteResult(StrictModel):
    output_path: str
    task_count: int = Field(ge=1)
    row_count: int = Field(ge=1)


_PRECOMPUTED_SCORE_ADAPTER = PairwiseLogisticModel(
    feature_names=("model_score",),
    feature_scale=np.asarray([1.0], dtype=np.float64),
    coefficients=np.asarray([1.0], dtype=np.float64),
    regularization_c=1.0,
)


def write_candidate_scores(
    *,
    features_path: str | Path,
    output_path: str | Path,
    scorer: CandidateScorer,
    batch_size: int,
) -> CandidateScoreWriteResult:
    """Stream a frozen feature artifact through one target-blind scorer."""

    source = Path(features_path)
    if not source.is_file():
        raise FileNotFoundError(f"Ranking features do not exist: {source}")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Candidate score artifact already exists: {output}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not scorer.feature_names or len(set(scorer.feature_names)) != len(
        scorer.feature_names
    ):
        raise ValueError("scorer feature_names must be nonempty and unique")
    parquet = pq.ParquetFile(source)
    available = set(parquet.schema_arrow.names)
    required = {
        "task_id",
        "business_id",
        "hybrid_v1_score",
        *scorer.feature_names,
    }
    if not required.issubset(available):
        raise ValueError("Ranking feature artifact does not match the scorer")
    forbidden = {"label", "target_business_id", "target_review_id", "ground_truth"}
    if available.intersection(forbidden):
        raise ValueError("Ranking features expose forbidden ground truth")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    columns = [
        "task_id",
        "business_id",
        "hybrid_v1_score",
        *scorer.feature_names,
    ]
    task_ids: set[str] = set()
    row_count = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(partial, SCORED_CANDIDATE_SCHEMA, compression="zstd")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            table = pa.Table.from_batches([batch])
            matrix = np.column_stack(
                [
                    table[name].to_numpy(zero_copy_only=False)
                    for name in scorer.feature_names
                ]
            ).astype(np.float64, copy=False)
            scores = np.asarray(
                scorer.score(matrix, feature_names=scorer.feature_names),
                dtype=np.float64,
            )
            if scores.shape != (len(table),) or not np.all(np.isfinite(scores)):
                raise ValueError("scorer returned invalid candidate scores")
            ids = table["task_id"].to_pylist()
            task_ids.update(str(task_id) for task_id in ids)
            output_table = pa.Table.from_arrays(
                [
                    table["task_id"],
                    table["business_id"],
                    pa.array(scores, type=pa.float64()),
                    table["hybrid_v1_score"].cast(pa.float64()),
                ],
                schema=SCORED_CANDIDATE_SCHEMA,
            )
            writer.write_table(output_table)
            row_count += len(table)
        writer.close()
        writer = None
        if row_count == 0 or not task_ids:
            raise ValueError("Ranking feature artifact is empty")
        os.replace(partial, output)
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise
    return CandidateScoreWriteResult(
        output_path=str(output),
        task_count=len(task_ids),
        row_count=row_count,
    )


def evaluate_candidate_scores(
    *,
    scores_path: str | Path,
    contexts_path: str | Path,
    ground_truth_path: str | Path,
    reviews_path: str | Path,
    interactions_path: str | Path,
    split: Literal["validation", "test"],
    model_name: str,
    blend_alpha: float,
    task_filter_path: str | Path | None = None,
) -> HybridV2RankingMetrics:
    """Reuse the frozen metric definition for precomputed nonlinear scores."""

    return evaluate_hybrid_v2_ranking(
        features_path=scores_path,
        contexts_path=contexts_path,
        ground_truth_path=ground_truth_path,
        reviews_path=reviews_path,
        interactions_path=interactions_path,
        split=split,
        model_name=model_name,
        model=_PRECOMPUTED_SCORE_ADAPTER,
        blend_alpha=blend_alpha,
        task_filter_path=task_filter_path,
    )


def write_scored_rankings(
    *,
    scores_path: str | Path,
    output_path: str | Path,
    blend_alpha: float,
) -> HybridV2RankingWriteResult:
    """Publish deterministic complete rankings from precomputed model scores."""

    return write_hybrid_v2_rankings(
        features_path=scores_path,
        output_path=output_path,
        model=_PRECOMPUTED_SCORE_ADAPTER,
        blend_alpha=blend_alpha,
    )
