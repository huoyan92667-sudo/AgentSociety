from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.evaluation.evaluator import (
    evaluate_prediction_file,
    write_evaluation_report,
)
from yelp_agent.models import Prediction, RecommendationTask


def write_models(path: Path, models: list[object]) -> None:
    path.write_text(
        "".join(model.model_dump_json() + "\n" for model in models),
        encoding="utf-8",
    )


def test_evaluates_complete_valid_prediction_file(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    truth_path = tmp_path / "ground_truth.parquet"
    predictions_path = tmp_path / "predictions.jsonl"
    candidates = [f"business-{index}" for index in range(1, 21)]
    target_positions = [1, 3, 6, 20]
    tasks = [
        RecommendationTask(
            task_id=f"task-{index}",
            user_id=f"user-{index}",
            cutoff_time="2020-01-01T00:00:00",
            candidate_business_ids=candidates,
        )
        for index in range(1, 5)
    ]
    write_models(tasks_path, tasks)
    pd.DataFrame(
        [
            {
                "task_id": f"task-{index}",
                "target_business_id": f"business-{target_position}",
            }
            for index, target_position in enumerate(target_positions, start=1)
        ]
    ).to_parquet(truth_path, index=False)
    predictions = [
        Prediction(
            task_id="task-1",
            ranking=candidates,
            latency_ms=10,
            fallback=False,
            tool_calls=0,
            metadata={"llm_attempted": True},
        ),
        Prediction(
            task_id="task-2",
            ranking=candidates,
            latency_ms=20,
            fallback=False,
            tool_calls=1,
            llm_tokens=100,
            metadata={"llm_attempted": True},
        ),
        Prediction(
            task_id="task-3",
            ranking=candidates,
            latency_ms=30,
            fallback=True,
            fallback_reason="llm_disabled",
            tool_calls=2,
            llm_tokens=200,
            metadata={"llm_attempted": False},
        ),
        Prediction(
            task_id="task-4",
            ranking=candidates,
            latency_ms=40,
            fallback=True,
            fallback_reason="network_error",
            tool_calls=3,
            metadata={"llm_attempted": True},
        ),
    ]
    write_models(predictions_path, predictions)

    report = evaluate_prediction_file(
        tasks_path,
        truth_path,
        predictions_path,
    )

    metrics = report.metrics
    assert metrics.task_count == 4
    assert metrics.valid_prediction_count == 4
    assert metrics.valid_output_rate == 1.0
    assert metrics.hr_at_1 == pytest.approx(0.25)
    assert metrics.hr_at_3 == pytest.approx(0.5)
    assert metrics.hr_at_5 == pytest.approx(0.5)
    assert metrics.avg_hr == pytest.approx((0.25 + 0.5 + 0.5) / 3)
    assert metrics.mrr == pytest.approx(0.3875)
    assert metrics.ndcg_at_5 == pytest.approx(0.375)
    assert metrics.mean_latency_ms == pytest.approx(25)
    assert metrics.p95_latency_ms == pytest.approx(38.5)
    assert metrics.mean_tool_calls == pytest.approx(1.5)
    assert metrics.mean_llm_tokens == pytest.approx(150)
    assert metrics.fallback_count == 2
    assert metrics.fallback_rate == pytest.approx(0.5)
    assert metrics.llm_attempted_count == 3
    assert metrics.llm_failure_count == 1
    assert metrics.llm_failure_rate == pytest.approx(1 / 3)
    assert report.issues == []


def test_invalid_and_missing_outputs_score_zero_and_are_reported(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    truth_path = tmp_path / "ground_truth.parquet"
    predictions_path = tmp_path / "predictions.jsonl"
    candidates = [f"business-{index}" for index in range(1, 21)]
    tasks = [
        RecommendationTask(
            task_id=f"task-{index}",
            user_id=f"user-{index}",
            cutoff_time="2020-01-01T00:00:00",
            candidate_business_ids=candidates,
        )
        for index in range(1, 5)
    ]
    write_models(tasks_path, tasks)
    pd.DataFrame(
        [
            {
                "task_id": f"task-{index}",
                "target_business_id": "business-1",
            }
            for index in range(1, 5)
        ]
    ).to_parquet(truth_path, index=False)
    valid = Prediction(
        task_id="task-1",
        ranking=candidates,
        latency_ms=10,
        fallback=False,
    )
    mismatch = Prediction(
        task_id="task-2",
        ranking=[*candidates[:-1], "outside-business"],
        latency_ms=20,
        fallback=False,
    )
    duplicate_first = Prediction(
        task_id="task-4",
        ranking=candidates,
        latency_ms=30,
        fallback=False,
    )
    duplicate_second = duplicate_first.model_copy(update={"latency_ms": 40})
    unexpected = Prediction(
        task_id="unknown-task",
        ranking=candidates,
        latency_ms=50,
        fallback=False,
    )
    predictions_path.write_text(
        valid.model_dump_json()
        + "\n"
        + mismatch.model_dump_json()
        + "\n"
        + duplicate_first.model_dump_json()
        + "\n"
        + duplicate_second.model_dump_json()
        + "\n"
        + unexpected.model_dump_json()
        + "\n"
        + "{not-json}\n",
        encoding="utf-8",
    )

    report = evaluate_prediction_file(
        tasks_path,
        truth_path,
        predictions_path,
    )

    metrics = report.metrics
    assert metrics.task_count == 4
    assert metrics.valid_prediction_count == 1
    assert metrics.invalid_prediction_count == 2
    assert metrics.missing_prediction_count == 1
    assert metrics.unexpected_prediction_count == 2
    assert metrics.valid_output_rate == pytest.approx(0.25)
    assert metrics.hr_at_1 == pytest.approx(0.25)
    assert metrics.hr_at_3 == pytest.approx(0.25)
    assert metrics.hr_at_5 == pytest.approx(0.25)
    assert metrics.avg_hr == pytest.approx(0.25)
    assert metrics.mrr == pytest.approx(0.25)
    assert metrics.ndcg_at_5 == pytest.approx(0.25)
    assert {issue.code for issue in report.issues} == {
        "candidate_mismatch",
        "duplicate_prediction",
        "missing_prediction",
        "unexpected_task_id",
        "invalid_json",
    }


def test_writes_metrics_and_issues_as_separate_artifacts(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    truth_path = tmp_path / "ground_truth.parquet"
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "run" / "metrics.json"
    issues_path = tmp_path / "run" / "evaluation_issues.jsonl"
    candidates = [f"business-{index}" for index in range(1, 21)]
    task = RecommendationTask(
        task_id="task-1",
        user_id="user-1",
        cutoff_time="2020-01-01T00:00:00",
        candidate_business_ids=candidates,
    )
    write_models(tasks_path, [task])
    pd.DataFrame(
        [{"task_id": "task-1", "target_business_id": "business-1"}]
    ).to_parquet(truth_path, index=False)
    predictions_path.write_text("", encoding="utf-8")
    report = evaluate_prediction_file(
        tasks_path,
        truth_path,
        predictions_path,
    )

    write_evaluation_report(report, metrics_path, issues_path)

    assert '"task_count": 1' in metrics_path.read_text(encoding="utf-8")
    issue_lines = issues_path.read_text(encoding="utf-8").splitlines()
    assert len(issue_lines) == 1
    assert '"code":"missing_prediction"' in issue_lines[0]


def test_combined_ground_truth_can_evaluate_one_split(tmp_path: Path) -> None:
    tasks_path = tmp_path / "test_tasks.jsonl"
    truth_path = tmp_path / "combined_ground_truth.parquet"
    predictions_path = tmp_path / "predictions.jsonl"
    candidates = [f"business-{index}" for index in range(1, 21)]
    task = RecommendationTask(
        task_id="test:user-1",
        user_id="user-1",
        cutoff_time="2020-01-01T00:00:00",
        candidate_business_ids=candidates,
    )
    prediction = Prediction(
        task_id=task.task_id,
        ranking=candidates,
        latency_ms=1,
        fallback=False,
    )
    write_models(tasks_path, [task])
    write_models(predictions_path, [prediction])
    pd.DataFrame(
        [
            {
                "task_id": "test:user-1",
                "target_business_id": "business-1",
            },
            {
                "task_id": "validation:user-1",
                "target_business_id": "business-2",
            },
        ]
    ).to_parquet(truth_path, index=False)

    report = evaluate_prediction_file(
        tasks_path,
        truth_path,
        predictions_path,
    )

    assert report.metrics.task_count == 1
    assert report.metrics.hr_at_1 == 1.0


def test_task_limit_ignores_predictions_for_other_known_tasks(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    truth_path = tmp_path / "ground_truth.parquet"
    predictions_path = tmp_path / "predictions.jsonl"
    candidates = [f"business-{index}" for index in range(1, 21)]
    tasks = [
        RecommendationTask(
            task_id=f"task-{index}",
            user_id=f"user-{index}",
            cutoff_time="2020-01-01T00:00:00",
            candidate_business_ids=candidates,
        )
        for index in range(1, 4)
    ]
    write_models(tasks_path, tasks)
    pd.DataFrame(
        [
            {
                "task_id": task.task_id,
                "target_business_id": "business-1",
            }
            for task in tasks
        ]
    ).to_parquet(truth_path, index=False)
    predictions = [
        Prediction(
            task_id=task.task_id,
            ranking=candidates,
            latency_ms=1,
            fallback=False,
        )
        for task in tasks
    ]
    write_models(predictions_path, predictions)

    report = evaluate_prediction_file(
        tasks_path,
        truth_path,
        predictions_path,
        task_limit=2,
    )

    assert report.metrics.task_count == 2
    assert report.metrics.valid_prediction_count == 2
    assert report.metrics.unexpected_prediction_count == 0
    assert report.metrics.hr_at_1 == 1.0
    assert report.issues == []


def test_task_limit_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task_limit must be positive"):
        evaluate_prediction_file(
            tmp_path / "tasks.jsonl",
            tmp_path / "ground_truth.parquet",
            tmp_path / "predictions.jsonl",
            task_limit=0,
        )
