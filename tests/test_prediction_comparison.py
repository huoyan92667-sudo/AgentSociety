from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.evaluation.comparison import compare_prediction_files
from yelp_agent.models import Prediction, RecommendationTask


def _write_models(path: Path, models: list[object]) -> None:
    path.write_text(
        "".join(model.model_dump_json() + "\n" for model in models),
        encoding="utf-8",
    )


def test_compares_target_rank_changes_without_exposing_targets(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    truth_path = tmp_path / "truth.parquet"
    baseline_path = tmp_path / "baseline.jsonl"
    challenger_path = tmp_path / "challenger.jsonl"
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
    _write_models(tasks_path, tasks)
    pd.DataFrame(
        [
            {"task_id": "task-1", "target_business_id": "business-1"},
            {"task_id": "task-2", "target_business_id": "business-4"},
            {"task_id": "task-3", "target_business_id": "business-2"},
        ]
    ).to_parquet(truth_path, index=False)
    _write_models(
        baseline_path,
        [
            Prediction(
                task_id=task.task_id,
                ranking=candidates,
                latency_ms=1,
                fallback=False,
            )
            for task in tasks
        ],
    )
    moved_down = [*candidates[1:4], candidates[0], *candidates[4:]]
    moved_up = [candidates[3], *candidates[:3], *candidates[4:]]
    _write_models(
        challenger_path,
        [
            Prediction(
                task_id="task-1",
                ranking=moved_down,
                latency_ms=1,
                fallback=False,
            ),
            Prediction(
                task_id="task-2",
                ranking=moved_up,
                latency_ms=1,
                fallback=False,
            ),
            Prediction(
                task_id="task-3",
                ranking=candidates,
                latency_ms=1,
                fallback=False,
            ),
        ],
    )

    comparison = compare_prediction_files(
        tasks_path,
        truth_path,
        baseline_path,
        challenger_path,
    )

    assert comparison.task_count == 3
    assert comparison.improved_count == 1
    assert comparison.unchanged_count == 1
    assert comparison.worsened_count == 1
    assert comparison.mean_baseline_rank == pytest.approx(7 / 3)
    assert comparison.mean_challenger_rank == pytest.approx(7 / 3)
    assert comparison.entered_top_1 == 1
    assert comparison.left_top_1 == 1
    assert comparison.entered_top_3 == 1
    assert comparison.left_top_3 == 1
    assert comparison.entered_top_5 == 0
    assert comparison.left_top_5 == 0
    assert comparison.largest_improvements[0].task_id == "task-2"
    assert comparison.largest_improvements[0].rank_improvement == 3
    assert comparison.largest_worsenings[0].task_id == "task-1"
    assert comparison.largest_worsenings[0].rank_improvement == -3
    assert "business-" not in comparison.model_dump_json()
