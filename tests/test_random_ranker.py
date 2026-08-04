from yelp_agent.models import RecommendationTask
from yelp_agent.protocols import Ranker
from yelp_agent.rankers.random_ranker import RandomRanker


def test_random_ranker_is_a_deterministic_candidate_permutation() -> None:
    candidates = [f"business-{index}" for index in range(1, 21)]
    task = RecommendationTask(
        task_id="test:user-1",
        user_id="user-1",
        cutoff_time="2020-01-01T00:00:00",
        candidate_business_ids=candidates,
    )
    ranker = RandomRanker(seed=42)

    first = ranker.rank(task)
    second = ranker.rank(task)

    assert isinstance(ranker, Ranker)
    assert first.task_id == task.task_id
    assert first.ranking == second.ranking
    assert set(first.ranking) == set(candidates)
    assert first.ranking != candidates
    assert task.candidate_business_ids == candidates
    assert first.fallback is False
    assert first.tool_calls == 0
    assert first.llm_tokens is None
    assert first.metadata == {
        "method": "random",
        "seed": 42,
        "llm_attempted": False,
    }
