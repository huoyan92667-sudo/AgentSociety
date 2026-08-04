from pathlib import Path

from yelp_agent.features.text import TemporalTextStore, fit_tfidf_model
from yelp_agent.models import RecommendationTask
from yelp_agent.protocols import Ranker
from yelp_agent.rankers.tfidf_ranker import TfidfRanker

from test_tfidf_training import tfidf_test_config, write_text_fixture


def test_tfidf_ranker_orders_by_personal_text_affinity(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, candidates = write_text_fixture(
        tmp_path
    )
    artifact = tmp_path / "features" / "tfidf_vectorizer.joblib"
    manifest = tmp_path / "features" / "tfidf_manifest.json"
    fit_tfidf_model(
        businesses,
        interactions,
        histories,
        artifact,
        manifest,
        tfidf_test_config(),
    )
    ranker = TfidfRanker(
        TemporalTextStore(
            businesses,
            interactions,
            histories,
            artifact,
            manifest,
        )
    )
    task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=list(reversed(candidates)),
    )

    prediction = ranker.rank(task)

    assert isinstance(ranker, Ranker)
    assert prediction.ranking == [
        candidates[0],
        *candidates[2:],
        candidates[1],
    ]
    assert prediction.fallback is False
    assert prediction.tool_calls == 0
    assert prediction.llm_tokens is None
    assert prediction.metadata == {
        "method": "tfidf",
        "positive_review_count": 1,
        "negative_review_count": 1,
        "llm_attempted": False,
    }
