from pathlib import Path

from yelp_agent.models import Prediction, RecommendationTask
from yelp_agent.rankers.random_ranker import RandomRanker
from yelp_agent.rankers.runner import run_ranker


def test_runner_writes_and_reuses_complete_predictions(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "run" / "predictions.jsonl"
    candidates = [f"business-{index}" for index in range(1, 21)]
    tasks = [
        RecommendationTask(
            task_id=f"test:user-{index}",
            user_id=f"user-{index}",
            cutoff_time="2020-01-01T00:00:00",
            candidate_business_ids=candidates,
        )
        for index in range(1, 3)
    ]
    tasks_path.write_text(
        "".join(task.model_dump_json() + "\n" for task in tasks),
        encoding="utf-8",
    )

    first = run_ranker(
        tasks_path,
        RandomRanker(seed=42),
        output_path,
    )

    predictions = [
        Prediction.model_validate_json(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    initial_mtime = output_path.stat().st_mtime_ns
    second = run_ranker(
        tasks_path,
        RandomRanker(seed=42),
        output_path,
    )

    assert first.status == "written"
    assert first.task_count == 2
    assert [prediction.task_id for prediction in predictions] == [
        task.task_id for task in tasks
    ]
    assert all(
        set(prediction.ranking) == set(candidates)
        for prediction in predictions
    )
    assert second.status == "skipped"
    assert second.task_count == 2
    assert output_path.stat().st_mtime_ns == initial_mtime
