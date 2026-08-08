from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yelp_agent.learning_to_rank.scored_ranking import (
    evaluate_candidate_scores,
    write_candidate_scores,
    write_scored_rankings,
)


@dataclass(frozen=True)
class _SignalScorer:
    feature_names: tuple[str, ...] = ("signal",)

    def score(
        self,
        features: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        assert feature_names == self.feature_names
        return features[:, 0]


def test_candidate_score_writer_is_target_blind_complete_and_batched(tmp_path) -> None:
    features_path = tmp_path / "features.parquet"
    output_path = tmp_path / "scores.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "t1",
                    "business_id": "b2",
                    "hybrid_v1_score": 0.8,
                    "signal": 0.2,
                },
                {
                    "task_id": "t1",
                    "business_id": "b1",
                    "hybrid_v1_score": 0.1,
                    "signal": 0.9,
                },
                {
                    "task_id": "t2",
                    "business_id": "b3",
                    "hybrid_v1_score": 0.4,
                    "signal": 0.6,
                },
            ]
        ),
        features_path,
    )

    result = write_candidate_scores(
        features_path=features_path,
        output_path=output_path,
        scorer=_SignalScorer(),
        batch_size=2,
    )

    table = pq.read_table(output_path)
    assert result.task_count == 2
    assert result.row_count == 3
    assert table.schema.names == [
        "task_id",
        "business_id",
        "model_score",
        "hybrid_v1_score",
    ]
    assert "label" not in table.schema.names
    np.testing.assert_allclose(table["model_score"].to_pylist(), [0.2, 0.9, 0.6])


def test_candidate_score_writer_rejects_training_labels(tmp_path) -> None:
    features_path = tmp_path / "features_with_label.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "t1",
                    "business_id": "b1",
                    "hybrid_v1_score": 0.1,
                    "signal": 0.9,
                    "label": 1,
                }
            ]
        ),
        features_path,
    )

    with pytest.raises(ValueError, match="forbidden ground truth"):
        write_candidate_scores(
            features_path=features_path,
            output_path=tmp_path / "scores.parquet",
            scorer=_SignalScorer(),
            batch_size=2,
        )


def test_scored_candidates_use_shared_metrics_and_deterministic_blending(
    tmp_path,
) -> None:
    cutoff = datetime(2020, 1, 10)
    scores = tmp_path / "scores.parquet"
    contexts = tmp_path / "contexts.parquet"
    truth = tmp_path / "truth.parquet"
    reviews = tmp_path / "reviews.parquet"
    interactions = tmp_path / "interactions.parquet"
    rankings = tmp_path / "rankings.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "t1",
                    "business_id": "target",
                    "model_score": 0.9,
                    "hybrid_v1_score": 0.1,
                },
                {
                    "task_id": "t1",
                    "business_id": "other",
                    "model_score": 0.1,
                    "hybrid_v1_score": 0.9,
                },
                {
                    "task_id": "t2",
                    "business_id": "only",
                    "model_score": 0.5,
                    "hybrid_v1_score": 0.5,
                },
            ]
        ),
        scores,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "t1",
                    "split": "validation",
                    "user_id": "u1",
                    "cutoff_time": cutoff,
                },
                {
                    "task_id": "t2",
                    "split": "validation",
                    "user_id": "u2",
                    "cutoff_time": cutoff,
                },
            ]
        ),
        contexts,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_id": "t1", "target_business_id": "target"},
                {"task_id": "t2", "target_business_id": "missing"},
            ]
        ),
        truth,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"business_id": "target", "date": datetime(2019, 1, 1)},
                {"business_id": "missing", "date": datetime(2019, 1, 1)},
            ]
        ),
        reviews,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "user_id": "someone",
                    "business_id": "other",
                    "date": datetime(2019, 1, 1),
                }
            ]
        ),
        interactions,
    )

    metrics = evaluate_candidate_scores(
        scores_path=scores,
        contexts_path=contexts,
        ground_truth_path=truth,
        reviews_path=reviews,
        interactions_path=interactions,
        split="validation",
        model_name="nonlinear",
        blend_alpha=1.0,
    )
    write_scored_rankings(
        scores_path=scores,
        output_path=rankings,
        blend_alpha=1.0,
    )

    assert metrics.hr_at_1 == 0.5
    assert metrics.retrieved_target_hr_at_1 == 1.0
    target = pq.read_table(rankings).to_pylist()[0]
    assert target["business_id"] == "target"
    assert target["rank"] == 1
