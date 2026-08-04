from pathlib import Path

from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.models import RecommendationTask
from yelp_agent.protocols import Ranker
from yelp_agent.rankers.category_ranker import CategoryRanker

from test_category_features import _write_category_fixture


def test_category_ranker_orders_candidates_by_personal_affinity(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, candidates = (
        _write_category_fixture(tmp_path)
    )
    store = TemporalCategoryStore(
        businesses,
        interactions,
        histories,
        broad_categories={"Restaurants", "Food", "Nightlife", "Shopping"},
    )
    ranker = CategoryRanker(store)
    task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=list(reversed(candidates)),
    )

    prediction = ranker.rank(task)

    assert isinstance(ranker, Ranker)
    assert prediction.ranking == candidates
    assert prediction.fallback is False
    assert prediction.tool_calls == 0
    assert prediction.llm_tokens is None
    assert prediction.metadata == {
        "method": "category",
        "history_count": 3,
        "llm_attempted": False,
    }
