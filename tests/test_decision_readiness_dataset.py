from pathlib import Path
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yelp_agent.config import load_config, load_decision_readiness_config
from yelp_agent.decision_readiness.dataset import (
    build_calibration_dataset,
    load_calibration_batch,
)
from yelp_agent.decision_readiness.experiment import (
    DecisionReadinessSources,
    run_decision_readiness_experiment,
)
from yelp_agent.learning_to_rank.features import HYBRID_V2_FEATURE_SCHEMA


def _write_sources(root: Path) -> tuple[Path, Path, Path]:
    feature_rows = []
    prediction_rows = []
    truth_rows = []
    for task_index in range(100):
        task_id = f"validation:user-{task_index}"
        target = f"b-{task_index}-0" if task_index % 2 == 0 else f"b-{task_index}-2"
        truth_rows.append({"task_id": task_id, "target_business_id": target})
        for candidate_index in range(3):
            business_id = f"b-{task_index}-{candidate_index}"
            gap = 0.4 if task_index % 2 == 0 else 0.05
            base = (1.0, 1.0 - gap, 0.4)[candidate_index]
            row = {
                "task_id": task_id,
                "split": "validation",
                "user_id": f"user-{task_index}",
                "cutoff_time": datetime(2024, 1, 1),
                "business_id": business_id,
                "retrieval_rank": candidate_index + 1,
                "label": business_id == target,
                "negative_kind": None,
                "sample_weight": 1.0,
                "fold": None,
            }
            for name in HYBRID_V2_FEATURE_SCHEMA.names[10:]:
                row[name] = 0.0
            row.update(
                {
                    "quality_score": base,
                    "category_score": base,
                    "text_score": base,
                    "location_score": base,
                    "item_knn_positive_score": base,
                    "item_knn_missing": 0.0,
                    "user_history_length_log": 2.0,
                    "user_profile_reliability": 0.7,
                    "user_category_novelty": 0.0,
                    "route_coverage": 4.0,
                }
            )
            feature_rows.append(row)
            prediction_rows.append(
                {
                    "task_id": task_id,
                    "business_id": business_id,
                    "rank": candidate_index + 1,
                    "model_rank": candidate_index + 1,
                    "v1_rank": candidate_index + 1,
                    "model_score": base,
                    "hybrid_v1_score": base,
                    "blend_score": base,
                }
            )
    features = root / "features.parquet"
    predictions = root / "predictions.parquet"
    truth = root / "truth.parquet"
    pq.write_table(pa.Table.from_pylist(feature_rows, schema=HYBRID_V2_FEATURE_SCHEMA), features)
    prediction_schema = pa.schema(
        [
            pa.field("task_id", pa.string()),
            pa.field("business_id", pa.string()),
            pa.field("rank", pa.int64()),
            pa.field("model_rank", pa.int64()),
            pa.field("v1_rank", pa.int64()),
            pa.field("model_score", pa.float64()),
            pa.field("hybrid_v1_score", pa.float64()),
            pa.field("blend_score", pa.float64()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(prediction_rows, schema=prediction_schema), predictions)
    pq.write_table(pa.Table.from_pylist(truth_rows), truth)
    return features, predictions, truth


def test_dataset_builder_keeps_target_ids_out_of_runtime_features(tmp_path: Path) -> None:
    features, predictions, truth = _write_sources(tmp_path)
    output = tmp_path / "calibration.parquet"

    result = build_calibration_dataset(
        validation_features_path=features,
        validation_predictions_path=predictions,
        ground_truth_path=truth,
        output_path=output,
        evaluation_policy=load_config().evaluation_data_usage,
    )
    batch = load_calibration_batch(output)

    assert result.task_count == 100
    assert result.positive_count == 50
    assert len(batch.task_ids) == 100
    assert "target_business_id" not in pq.ParquetFile(output).schema_arrow.names
    assert set(batch.labels.tolist()) == {0, 1}


def test_complete_experiment_writes_and_reuses_one_consistent_run(
    tmp_path: Path,
) -> None:
    features, predictions, truth = _write_sources(tmp_path)
    sources = DecisionReadinessSources(
        validation_features=features,
        validation_predictions=predictions,
        validation_ground_truth=truth,
    )
    feature_output = tmp_path / "derived" / "calibration.parquet"
    run_root = tmp_path / "run"
    policy = load_config().evaluation_data_usage
    config = load_decision_readiness_config()

    first = run_decision_readiness_experiment(
        sources,
        config,
        policy,
        feature_output_path=feature_output,
        run_root=run_root,
    )
    second = run_decision_readiness_experiment(
        sources,
        config,
        policy,
        feature_output_path=feature_output,
        run_root=run_root,
    )

    assert first.status == "written"
    assert second.status == "reused"
    assert first.report.source_task_count == 100
    assert first.report.test_data_used_for_fit is False
    assert first.report.query_aware_confidence_calibrated is False
    assert (run_root / "frozen" / "manifest.json").is_file()
    assert (run_root / "evaluation_report.json").is_file()
    assert (run_root / "reliability_diagram.svg").is_file()
    assert (run_root / "coverage_risk.csv").is_file()

    changed_config = config.model_copy(update={"random_seed": 43})
    with pytest.raises(ValueError, match="configuration changed"):
        run_decision_readiness_experiment(
            sources,
            changed_config,
            policy,
            feature_output_path=feature_output,
            run_root=run_root,
        )
