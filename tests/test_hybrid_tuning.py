from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.features.hybrid import (
    HybridComponentFeatures,
    HybridComponentScore,
    HybridWeights,
)
from yelp_agent.models import RecommendationTask, UserProfile
from yelp_agent.tuning.hybrid import (
    HybridTuningError,
    enumerate_hybrid_weights,
    load_validated_hybrid_weights,
    tune_hybrid_weights,
)


def test_enumerates_all_step_point_one_weight_combinations() -> None:
    weights = enumerate_hybrid_weights(0.1)

    tuples = {
        (
            weight.category,
            weight.text,
            weight.quality,
            weight.location,
        )
        for weight in weights
    }
    assert len(weights) == 286
    assert len(tuples) == 286
    assert (0.4, 0.3, 0.2, 0.1) in tuples
    assert all(abs(sum(values) - 1.0) < 1e-9 for values in tuples)


class CountingTextSignalStore:
    def __init__(self) -> None:
        self.call_count = 0

    def features_for(self, task: RecommendationTask) -> HybridComponentFeatures:
        self.call_count += 1
        return HybridComponentFeatures(
            profile=UserProfile(
                user_id=task.user_id,
                history_count=1,
                average_rating=5.0,
                rating_distribution={
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "5": 1,
                },
                preferred_categories={},
                disliked_categories={},
            ),
            business_scores={
                business_id: HybridComponentScore(
                    business_id=business_id,
                    category_score=(
                        1.0 if business_id == "business-00" else 0.0
                    ),
                    text_score=(
                        1.0 if business_id == "business-01" else 0.0
                    ),
                    quality_score=0.0,
                    location_score=0.0,
                )
                for business_id in task.candidate_business_ids
            },
        )


def test_tunes_on_precomputed_validation_features_and_freezes_unique_weights(
    tmp_path: Path,
) -> None:
    candidates = [f"business-{index:02d}" for index in range(20)]
    task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )
    tasks_path = tmp_path / "validation_tasks.jsonl"
    tasks_path.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    ground_truth_path = tmp_path / "ground_truth.parquet"
    pd.DataFrame(
        [
            {
                "task_id": task.task_id,
                "target_business_id": "business-01",
            }
        ]
    ).to_parquet(ground_truth_path, index=False)
    output_path = tmp_path / "weights" / "hybrid_weights.json"
    feature_store = CountingTextSignalStore()
    initial_weights = HybridWeights(
        category=0.4,
        text=0.3,
        quality=0.2,
        location=0.1,
    )

    result = tune_hybrid_weights(
        tasks_path,
        ground_truth_path,
        feature_store,
        output_path,
        initial_weights=initial_weights,
        feature_sources_sha256={"synthetic": "0" * 64},
        step=0.1,
    )

    assert result.status == "written"
    assert result.task_count == 1
    assert result.combination_count == 286
    assert result.selected_weights == HybridWeights(
        category=0.3,
        text=0.4,
        quality=0.2,
        location=0.1,
    )
    assert result.validation_metrics.avg_hr == 1.0
    assert result.validation_metrics.mrr == 1.0
    assert feature_store.call_count == 1
    assert load_validated_hybrid_weights(
        output_path,
        feature_sources_sha256={"synthetic": "0" * 64},
    ) == result.selected_weights

    with pytest.raises(HybridTuningError, match="feature sources"):
        load_validated_hybrid_weights(
            output_path,
            feature_sources_sha256={"synthetic": "1" * 64},
        )

    initial_mtime = output_path.stat().st_mtime_ns
    unused_store = CountingTextSignalStore()
    reused = tune_hybrid_weights(
        tasks_path,
        ground_truth_path,
        unused_store,
        output_path,
        initial_weights=initial_weights,
        feature_sources_sha256={"synthetic": "0" * 64},
        step=0.1,
    )
    assert reused.status == "skipped"
    assert unused_store.call_count == 0
    assert output_path.stat().st_mtime_ns == initial_mtime


def test_tuner_rejects_test_tasks(tmp_path: Path) -> None:
    candidates = [f"business-{index:02d}" for index in range(20)]
    task = RecommendationTask(
        task_id="test:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )
    tasks_path = tmp_path / "test_tasks.jsonl"
    tasks_path.write_text(task.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(
        HybridTuningError,
        match="accepts validation tasks only",
    ):
        tune_hybrid_weights(
            tasks_path,
            tmp_path / "unused_ground_truth.parquet",
            CountingTextSignalStore(),
            tmp_path / "weights.json",
            initial_weights=HybridWeights(
                category=0.4,
                text=0.3,
                quality=0.2,
                location=0.1,
            ),
            feature_sources_sha256={"synthetic": "0" * 64},
        )
