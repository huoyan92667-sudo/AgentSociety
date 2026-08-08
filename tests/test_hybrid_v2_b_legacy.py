from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.learning_to_rank.evaluation import HybridV2RankingMetrics
from yelp_agent.learning_to_rank.lambdamart import (
    LambdaMARTParameters,
    LambdaMARTTrainingBatch,
    train_lambdamart,
)
from yelp_agent.learning_to_rank.lambdamart_artifacts import (
    FrozenLambdaMARTManifest,
    save_frozen_lambdamart,
)
from yelp_agent.learning_to_rank.lambdamart_legacy import (
    LambdaMARTLegacyTestSources,
    run_lambdamart_legacy_test,
)


def _metric(name: str, value: float) -> dict[str, object]:
    return HybridV2RankingMetrics(
        model_name=name,
        split="test",
        blend_alpha=0.5,
        task_count=1,
        primary_task_count=1,
        catalog_ineligible_task_count=0,
        target_in_history_task_count=0,
        target_retrieved_count=1,
        target_not_retrieved_count=0,
        hr_at_1=value,
        hr_at_3=value,
        hr_at_5=value,
        avg_hr=value,
        mrr=value,
        ndcg_at_5=value,
        retrieved_target_hr_at_1=value,
        retrieved_target_hr_at_3=value,
        retrieved_target_hr_at_5=value,
        retrieved_target_mrr=value,
        all_task_hr_at_1=value,
        all_task_hr_at_3=value,
        all_task_hr_at_5=value,
    ).model_dump(mode="json")


def test_legacy_test_loads_frozen_model_without_selecting_on_test(tmp_path) -> None:
    parameters = LambdaMARTParameters(
        name="tiny",
        num_leaves=3,
        learning_rate=0.2,
        num_boost_round=10,
        min_child_samples=1,
        reg_lambda=0.0,
    )
    model = train_lambdamart(
        LambdaMARTTrainingBatch(
            feature_names=("signal",),
            features=np.asarray([[1.0], [0.0], [0.9], [0.1]], dtype=np.float64),
            labels=np.asarray([1, 0, 1, 0], dtype=np.int8),
            group_sizes=np.asarray([2, 2], dtype=np.int32),
            sample_weights=np.ones(4, dtype=np.float64),
        ),
        parameters=parameters,
        random_seed=42,
    )
    frozen = tmp_path / "frozen"
    save_frozen_lambdamart(
        frozen,
        model,
        FrozenLambdaMARTManifest(
            artifact_name="Hybrid V2-B LambdaMART",
            model_version="2.0.0-b",
            feature_version="1.0.0",
            selected_parameters=parameters,
            selected_blend_alpha=1.0,
            selected_feature_set="full",
            feature_names=["signal"],
            source_sha256={"train": "a" * 64},
            configuration_sha256="b" * 64,
            model_sha256="0" * 64,
            validation_summary={"avg_hr": 1.0},
            test_data_used_for_training=False,
            test_data_used_for_selection=False,
        ),
    )
    cutoff = datetime(2020, 1, 10)
    features = tmp_path / "test_features.parquet"
    contexts = tmp_path / "contexts.parquet"
    truth = tmp_path / "truth.parquet"
    reviews = tmp_path / "reviews.parquet"
    interactions = tmp_path / "interactions.parquet"
    logistic_report = tmp_path / "logistic_test.json"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "t1",
                    "business_id": "target",
                    "hybrid_v1_score": 0.1,
                    "signal": 1.0,
                },
                {
                    "task_id": "t1",
                    "business_id": "other",
                    "hybrid_v1_score": 0.9,
                    "signal": 0.0,
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
                    "split": "test",
                    "user_id": "u1",
                    "cutoff_time": cutoff,
                }
            ]
        ),
        contexts,
    )
    pq.write_table(
        pa.Table.from_pylist([{"task_id": "t1", "target_business_id": "target"}]),
        truth,
    )
    pq.write_table(
        pa.Table.from_pylist([{"business_id": "target", "date": datetime(2019, 1, 1)}]),
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
    logistic_report.write_text(
        json.dumps(
            {
                "hybrid_v1": _metric("hybrid_v1", 0.0),
                "hybrid_v2": _metric("hybrid_v2_a", 0.0),
            }
        ),
        encoding="utf-8",
    )

    report = run_lambdamart_legacy_test(
        LambdaMARTLegacyTestSources(
            test_features=features,
            test_contexts=contexts,
            test_ground_truth=truth,
            reviews=reviews,
            interactions=interactions,
            frozen_model_root=frozen,
            hybrid_v2_a_legacy_report=logistic_report,
        ),
        score_output_path=tmp_path / "test_scores.parquet",
        prediction_output_path=tmp_path / "test_predictions.parquet",
        report_output_path=tmp_path / "legacy_report.json",
        prediction_batch_size=1000,
    )

    assert report.evaluation_status == "historical_comparison_only"
    assert report.model_selected_before_test is True
    assert report.hybrid_v2_b_lambdamart.hr_at_1 == 1.0
    assert report.test_used_for_selection is False
