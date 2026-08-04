from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.config import load_config
from yelp_agent.evaluation.data_usage import (
    DataUsageViolation,
    assign_user_fold,
)
from yelp_agent.evaluation.hybrid_diagnostics import (
    HybridDiagnosisSummary,
    diagnose_hybrid_v1_validation,
    write_hybrid_diagnosis,
)
from yelp_agent.features.hybrid import (
    HybridComponentFeatures,
    HybridComponentScore,
    HybridWeights,
)
from yelp_agent.features.location import (
    LocationBusinessScore,
    LocationTaskFeatures,
)
from yelp_agent.features.quality import BusinessQuality
from yelp_agent.models import LocationCenter, RecommendationTask, UserProfile


CANDIDATES = [f"business-{index:02d}" for index in range(20)]
TARGET = CANDIDATES[0]
PROJECT_ROOT = Path(__file__).parents[1]


class FixedComponentStore:
    def features_for(
        self,
        task: RecommendationTask,
    ) -> HybridComponentFeatures:
        return HybridComponentFeatures(
            profile=UserProfile(
                user_id=task.user_id,
                history_count=12,
                average_rating=4.0,
                rating_distribution={
                    "1": 1,
                    "2": 1,
                    "3": 2,
                    "4": 4,
                    "5": 4,
                },
                preferred_categories={"Noodles": 0.9},
                disliked_categories={},
                location_center=LocationCenter(
                    latitude=39.95,
                    longitude=-75.16,
                ),
            ),
            business_scores={
                business_id: HybridComponentScore(
                    business_id=business_id,
                    category_score=(
                        0.9
                        if business_id == TARGET
                        else 0.1 if business_id == CANDIDATES[1] else 0.0
                    ),
                    text_score=(
                        0.8
                        if business_id == TARGET
                        else 0.9 if business_id == CANDIDATES[1] else 0.1
                    ),
                    quality_score=(
                        0.2
                        if business_id == TARGET
                        else 0.8 if business_id == CANDIDATES[1] else 0.1
                    ),
                    location_score=(
                        0.1
                        if business_id == TARGET
                        else 0.9 if business_id == CANDIDATES[1] else 0.1
                    ),
                )
                for business_id in task.candidate_business_ids
            },
        )


class FixedQualityStore:
    def score_businesses(
        self,
        business_ids: list[str],
        cutoff_time: object,
    ) -> dict[str, BusinessQuality]:
        return {
            business_id: BusinessQuality(
                business_id=business_id,
                review_count=10 if business_id == TARGET else 30,
                mean_rating=4.0,
                bayesian_rating=4.0,
                normalized_bayesian_rating=0.75,
                normalized_popularity=0.2 if business_id == TARGET else 0.8,
                quality_score=0.64 if business_id == TARGET else 0.76,
            )
            for business_id in business_ids
        }


class FixedLocationStore:
    def features_for(
        self,
        task: RecommendationTask,
    ) -> LocationTaskFeatures:
        return LocationTaskFeatures(
            location_center=LocationCenter(
                latitude=39.95,
                longitude=-75.16,
            ),
            history_coordinate_count=12,
            business_scores={
                business_id: LocationBusinessScore(
                    business_id=business_id,
                    distance_km=float(index + 1),
                    location_score=0.9,
                )
                for index, business_id in enumerate(task.candidate_business_ids)
            },
        )


def _policy():
    policy = load_config().evaluation_data_usage
    return policy.model_copy(
        update={
            "bootstrap": policy.bootstrap.model_copy(
                update={"samples": 100}
            )
        }
    )


def _one_user_for_each_fold() -> list[str]:
    policy = _policy()
    users_by_fold: dict[int, str] = {}
    index = 0
    while len(users_by_fold) < policy.cross_validation.folds:
        user_id = f"user-{index}"
        users_by_fold.setdefault(assign_user_fold(user_id, policy), user_id)
        index += 1
    return [
        users_by_fold[fold]
        for fold in range(1, policy.cross_validation.folds + 1)
    ]


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    tasks_path = root / "validation_tasks.jsonl"
    tasks = [
        RecommendationTask(
            task_id=f"validation:{user_id}",
            user_id=user_id,
            cutoff_time="2020-02-01T00:00:00",
            candidate_business_ids=CANDIDATES,
        )
        for user_id in _one_user_for_each_fold()
    ]
    tasks_path.write_text(
        "".join(task.model_dump_json() + "\n" for task in tasks),
        encoding="utf-8",
    )
    truth_path = root / "ground_truth.parquet"
    pd.DataFrame(
        [
            {"task_id": task.task_id, "target_business_id": TARGET}
            for task in tasks
        ]
    ).to_parquet(truth_path, index=False)
    provenance_path = root / "candidate_provenance.parquet"
    provenance_rows = []
    for task_index, task in enumerate(tasks):
        for position, business_id in enumerate(CANDIDATES, start=1):
            source_bucket = "target" if business_id == TARGET else "same_fine"
            if task_index == 0 and business_id == CANDIDATES[2]:
                source_bucket = "refill_same_fine"
            provenance_rows.append(
                {
                    "task_id": task.task_id,
                    "business_id": business_id,
                    "source_bucket": source_bucket,
                    "final_position": position,
                }
            )
    pd.DataFrame(provenance_rows).to_parquet(provenance_path, index=False)
    return tasks_path, truth_path, provenance_path


def test_diagnoses_frozen_hybrid_by_fold_and_segment(tmp_path: Path) -> None:
    tasks_path, truth_path, provenance_path = _write_inputs(tmp_path)

    diagnostics, summary = diagnose_hybrid_v1_validation(
        validation_tasks_path=tasks_path,
        ground_truth_path=truth_path,
        provenance_path=provenance_path,
        feature_store=FixedComponentStore(),
        quality_store=FixedQualityStore(),
        location_store=FixedLocationStore(),
        weights=HybridWeights(
            category=0.0,
            text=0.5,
            quality=0.1,
            location=0.4,
        ),
        policy=_policy(),
    )

    assert summary.development_split == "validation"
    assert summary.legacy_test_loaded is False
    assert summary.task_count == 5
    assert summary.user_count == 5
    assert summary.zero_weight_components == ["category"]
    assert summary.fold_task_counts == {str(fold): 1 for fold in range(1, 6)}
    assert summary.overall.hr_at_3 == 1.0
    assert summary.overall.mean_target_rank == 2.0
    assert summary.bootstrap_intervals["avg_hr"].lower == pytest.approx(
        summary.overall.avg_hr
    )
    assert len(diagnostics) == 5
    assert diagnostics[0].candidate_refill is True
    assert diagnostics[0].target_category_seen is True
    assert diagnostics[0].primary_disadvantage == "location"
    assert summary.segment_metrics["candidate_refill"][
        "refill_used"
    ].task_count == 1


def test_writes_local_identifiers_but_anonymous_publishable_files(
    tmp_path: Path,
) -> None:
    tasks_path, truth_path, provenance_path = _write_inputs(tmp_path)
    diagnostics, summary = diagnose_hybrid_v1_validation(
        validation_tasks_path=tasks_path,
        ground_truth_path=truth_path,
        provenance_path=provenance_path,
        feature_store=FixedComponentStore(),
        quality_store=FixedQualityStore(),
        location_store=FixedLocationStore(),
        weights=HybridWeights(
            category=0.0,
            text=0.5,
            quality=0.1,
            location=0.4,
        ),
        policy=_policy(),
    )
    task_output = tmp_path / "runs" / "task_diagnostics.jsonl"
    summary_output = tmp_path / "docs" / "summary.json"
    markdown_output = tmp_path / "docs" / "report.md"

    write_hybrid_diagnosis(
        diagnostics,
        summary,
        task_diagnostics_path=task_output,
        summary_path=summary_output,
        markdown_path=markdown_output,
    )
    first_bytes = {
        path: path.read_bytes()
        for path in (task_output, summary_output, markdown_output)
    }
    write_hybrid_diagnosis(
        diagnostics,
        summary,
        task_diagnostics_path=task_output,
        summary_path=summary_output,
        markdown_path=markdown_output,
    )

    user_id = diagnostics[0].user_id.encode()
    task_id = diagnostics[0].task_id.encode()
    assert user_id in task_output.read_bytes()
    assert task_id in task_output.read_bytes()
    assert user_id not in summary_output.read_bytes()
    assert task_id not in summary_output.read_bytes()
    assert user_id not in markdown_output.read_bytes()
    assert task_id not in markdown_output.read_bytes()
    assert {
        path: path.read_bytes()
        for path in (task_output, summary_output, markdown_output)
    } == first_bytes


def test_rejects_legacy_test_tasks_before_loading_answers(
    tmp_path: Path,
) -> None:
    task = RecommendationTask(
        task_id="test:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=CANDIDATES,
    )
    tasks_path = tmp_path / "test_tasks.jsonl"
    tasks_path.write_text(task.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(DataUsageViolation, match="validation tasks only"):
        diagnose_hybrid_v1_validation(
            validation_tasks_path=tasks_path,
            ground_truth_path=tmp_path / "not-read.parquet",
            provenance_path=tmp_path / "not-read-provenance.parquet",
            feature_store=FixedComponentStore(),
            quality_store=FixedQualityStore(),
            location_store=FixedLocationStore(),
            weights=HybridWeights(
                category=0.0,
                text=0.5,
                quality=0.1,
                location=0.4,
            ),
            policy=_policy(),
        )


def test_tracked_validation_diagnosis_is_current_and_anonymous() -> None:
    summary_path = (
        PROJECT_ROOT
        / "docs"
        / "evaluation"
        / "hybrid_v1_validation_diagnosis_summary.json"
    )
    report_path = (
        PROJECT_ROOT
        / "docs"
        / "evaluation"
        / "hybrid_v1_validation_diagnosis.md"
    )
    summary = HybridDiagnosisSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    report = report_path.read_text(encoding="utf-8")

    assert summary.task_count == 4_971
    assert summary.user_count == 4_971
    assert summary.legacy_test_loaded is False
    assert summary.published_summary_includes_identifiers is False
    assert summary.zero_weight_components == ["category"]
    assert summary.source_sha256["task_diagnostics"] == (
        "26bde2442629362d90bfeddde49ed4ea662ad3327f17f1fba1ec5b081b1f3dc5"
    )
    assert "最值得继续验证的线索" in report
    assert "对 Hybrid V2 的建议顺序" in report
