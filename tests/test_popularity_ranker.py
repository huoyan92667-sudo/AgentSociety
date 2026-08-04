from pathlib import Path

import pandas as pd

from yelp_agent.features.quality import TemporalQualityStore
from yelp_agent.models import RecommendationTask
from yelp_agent.protocols import Ranker
from yelp_agent.rankers.popularity_ranker import PopularityRanker


def test_popularity_ranker_orders_by_point_in_time_quality(
    tmp_path: Path,
) -> None:
    reviews_path = tmp_path / "reviews.parquet"
    candidates = [f"business-{index:02d}" for index in range(20)]
    reviews = [
        {
            "business_id": business_id,
            "stars": 3.0,
            "date": pd.Timestamp("2020-01-01"),
        }
        for business_id in candidates
    ]
    reviews.extend(
        {
            "business_id": "business-00",
            "stars": 5.0,
            "date": pd.Timestamp("2020-01-02"),
        }
        for _ in range(10)
    )
    pd.DataFrame(reviews).to_parquet(reviews_path, index=False)
    ranker = PopularityRanker(TemporalQualityStore(reviews_path))
    task = RecommendationTask(
        task_id="test:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=list(reversed(candidates)),
    )

    prediction = ranker.rank(task)

    assert isinstance(ranker, Ranker)
    assert prediction.task_id == task.task_id
    assert prediction.ranking == candidates
    assert prediction.fallback is False
    assert prediction.tool_calls == 0
    assert prediction.llm_tokens is None
    assert prediction.metadata == {
        "method": "popularity",
        "prior_count": 20,
        "llm_attempted": False,
    }
