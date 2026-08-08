from __future__ import annotations

import json
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.config import HybridV2BConfig, LambdaMARTTrialConfig
from yelp_agent.learning_to_rank.evaluation import HybridV2RankingMetrics
from yelp_agent.learning_to_rank.features import ALL_FEATURE_NAMES
from yelp_agent.learning_to_rank.lambdamart_experiment import (
    HybridV2BExperimentSources,
    run_hybrid_v2_b_validation_experiment,
)


def _feature_values(signal: float) -> dict[str, float]:
    return {
        name: signal if name == "category_score" else 0.1 for name in ALL_FEATURE_NAMES
    }


def _metrics(name: str, avg_hr: float) -> dict[str, object]:
    value = HybridV2RankingMetrics(
        model_name=name,
        split="validation",
        blend_alpha=0.5,
        task_count=2,
        primary_task_count=2,
        catalog_ineligible_task_count=0,
        target_in_history_task_count=0,
        target_retrieved_count=2,
        target_not_retrieved_count=0,
        hr_at_1=avg_hr,
        hr_at_3=avg_hr,
        hr_at_5=avg_hr,
        avg_hr=avg_hr,
        mrr=avg_hr,
        ndcg_at_5=avg_hr,
        retrieved_target_hr_at_1=avg_hr,
        retrieved_target_hr_at_3=avg_hr,
        retrieved_target_hr_at_5=avg_hr,
        retrieved_target_mrr=avg_hr,
        all_task_hr_at_1=avg_hr,
        all_task_hr_at_3=avg_hr,
        all_task_hr_at_5=avg_hr,
    )
    return value.model_dump(mode="json")


def test_validation_experiment_freezes_challenger_without_test_input(tmp_path) -> None:
    cutoff = datetime(2020, 1, 10)
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    contexts = tmp_path / "contexts.parquet"
    truth = tmp_path / "truth.parquet"
    reviews = tmp_path / "reviews.parquet"
    interactions = tmp_path / "interactions.parquet"
    logistic_report = tmp_path / "logistic_report.json"
    train_rows: list[dict[str, object]] = []
    for task_index in range(6):
        task_id = f"train-{task_index}"
        for candidate_index, signal in enumerate((0.95, 0.2, 0.1)):
            train_rows.append(
                {
                    "task_id": task_id,
                    "business_id": f"b-{candidate_index}",
                    "label": candidate_index == 0,
                    "sample_weight": 1.0,
                    **_feature_values(signal),
                }
            )
    pq.write_table(pa.Table.from_pylist(train_rows), train)
    validation_rows: list[dict[str, object]] = []
    for task_index in range(2):
        task_id = f"validation-{task_index}"
        for business_id, signal, v1 in (
            (f"target-{task_index}", 0.95, 0.1),
            (f"other-{task_index}", 0.1, 0.9),
        ):
            validation_rows.append(
                {
                    "task_id": task_id,
                    "business_id": business_id,
                    "hybrid_v1_score": v1,
                    **_feature_values(signal),
                }
            )
    pq.write_table(pa.Table.from_pylist(validation_rows), validation)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": f"validation-{index}",
                    "split": "validation",
                    "user_id": f"u-{index}",
                    "cutoff_time": cutoff,
                }
                for index in range(2)
            ]
        ),
        contexts,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": f"validation-{index}",
                    "target_business_id": f"target-{index}",
                }
                for index in range(2)
            ]
        ),
        truth,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "business_id": f"target-{index}",
                    "date": datetime(2019, 1, 1),
                }
                for index in range(2)
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
    logistic_report.write_text(
        json.dumps(
            {
                "baseline": _metrics("hybrid_v1", 0.0),
                "selected": _metrics("hybrid_v2_a", 0.5),
            }
        ),
        encoding="utf-8",
    )
    config = HybridV2BConfig(
        schema_version=1,
        model_version="2.0.0-b",
        random_seed=42,
        fair_comparison_feature_set="without_business_profile",
        parameter_trials=[
            LambdaMARTTrialConfig(
                name="tiny",
                num_leaves=3,
                learning_rate=0.2,
                num_boost_round=10,
                min_child_samples=1,
                reg_lambda=0.0,
            )
        ],
        blend_alphas=[0.0, 1.0],
        ablation_feature_sets=["without_business_profile", "full"],
        primary_metric="avg_hr",
        prediction_batch_size=1000,
        bootstrap_samples=100,
    )

    result = run_hybrid_v2_b_validation_experiment(
        HybridV2BExperimentSources(
            train_features=train,
            validation_features=validation,
            validation_contexts=contexts,
            validation_ground_truth=truth,
            reviews=reviews,
            interactions=interactions,
            hybrid_v2_a_validation_report=logistic_report,
        ),
        config,
        data_root=tmp_path / "data",
        run_root=tmp_path / "run",
    )

    assert result.status == "written"
    assert (tmp_path / "run" / "frozen" / "model.joblib").is_file()
    assert (tmp_path / "run" / "validation_report.json").is_file()
    assert (tmp_path / "data" / "validation_predictions.parquet").is_file()
    assert result.validation_report.legacy_test_used_for_selection is False

    reused = run_hybrid_v2_b_validation_experiment(
        HybridV2BExperimentSources(
            train_features=train,
            validation_features=validation,
            validation_contexts=contexts,
            validation_ground_truth=truth,
            reviews=reviews,
            interactions=interactions,
            hybrid_v2_a_validation_report=logistic_report,
        ),
        config,
        data_root=tmp_path / "data",
        run_root=tmp_path / "run",
    )
    assert reused.status == "reused"
    assert reused.validation_scores.row_count == 4
