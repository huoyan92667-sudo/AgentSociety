from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.learning_to_rank.evaluation import evaluate_hybrid_v2_ranking
from yelp_agent.learning_to_rank.model import PairwiseLogisticModel


def test_ranking_metrics_separate_retrieval_failure_from_ranking_quality(
    tmp_path: Path,
) -> None:
    cutoff = datetime(2020, 1, 10)
    features = tmp_path / "features.parquet"
    contexts = tmp_path / "contexts.parquet"
    truth = tmp_path / "truth.parquet"
    reviews = tmp_path / "reviews.parquet"
    interactions = tmp_path / "interactions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "t1",
                    "business_id": "good",
                    "preference": 0.9,
                    "hybrid_v1_score": 0.2,
                    "label": None,
                },
                {
                    "task_id": "t1",
                    "business_id": "bad",
                    "preference": 0.1,
                    "hybrid_v1_score": 0.8,
                    "label": None,
                },
                {
                    "task_id": "t2",
                    "business_id": "other",
                    "preference": 0.5,
                    "hybrid_v1_score": 0.5,
                    "label": None,
                },
            ]
        ),
        features,
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
                {"task_id": "t1", "target_business_id": "good"},
                {"task_id": "t2", "target_business_id": "missing"},
            ]
        ),
        truth,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"business_id": "good", "date": datetime(2019, 1, 1)},
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
    model = PairwiseLogisticModel(
        feature_names=("preference",),
        feature_scale=pa.array([1.0]).to_numpy(),
        coefficients=pa.array([1.0]).to_numpy(),
        regularization_c=1.0,
    )

    metrics = evaluate_hybrid_v2_ranking(
        features_path=features,
        contexts_path=contexts,
        ground_truth_path=truth,
        reviews_path=reviews,
        interactions_path=interactions,
        split="validation",
        model_name="synthetic",
        model=model,
        blend_alpha=1.0,
    )

    assert metrics.primary_task_count == 2
    assert metrics.target_retrieved_count == 1
    assert metrics.hr_at_1 == 0.5
    assert metrics.avg_hr == 0.5
    assert metrics.mrr == 0.5
    assert metrics.retrieved_target_hr_at_1 == 1.0
