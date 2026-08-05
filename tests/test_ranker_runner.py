from pathlib import Path

import pytest

from yelp_agent.config import ResolvedConfiguration, load_config
from yelp_agent.models import Prediction, RecommendationTask
from yelp_agent.rankers.random_ranker import RandomRanker
from yelp_agent.rankers.runner import RankerRunError, run_ranker


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


def test_runner_records_and_checks_effective_configuration(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "run" / "predictions.jsonl"
    task = RecommendationTask(
        task_id="test:user-1",
        user_id="user-1",
        cutoff_time="2020-01-01T00:00:00",
        candidate_business_ids=[
            f"business-{index}" for index in range(1, 21)
        ],
    )
    tasks_path.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    config = load_config(Path(__file__).parents[1] / "configs")

    first = run_ranker(
        tasks_path,
        RandomRanker(seed=config.data.random_seed),
        output_path,
        configuration=config,
    )

    snapshot_path = output_path.parent / "resolved_config.json"
    snapshot = ResolvedConfiguration.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )
    assert first.resolved_config_path == str(snapshot_path)
    assert first.config_fingerprint == snapshot.fingerprint

    changed = config.model_copy(
        update={
            "agent": config.agent.model_copy(
                update={"timeout_seconds": 89},
            )
        }
    )
    with pytest.raises(RankerRunError, match="different configuration"):
        run_ranker(
            tasks_path,
            RandomRanker(seed=config.data.random_seed),
            output_path,
            configuration=changed,
        )
