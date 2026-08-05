from pathlib import Path

import pytest

from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.features.text import (
    TemporalTextStore,
    fit_tfidf_model,
)
from yelp_agent.models import RecommendationTask

from test_tfidf_training import tfidf_test_config, write_text_fixture


def test_scores_positive_and_negative_text_affinity_and_extracts_keywords(
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
    store = TemporalTextStore(
        TemporalDataView(businesses, interactions, interactions),
        artifact,
        manifest,
    )
    task = RecommendationTask(
        task_id="validation:user-1",
        user_id="user-1",
        cutoff_time="2020-02-01T00:00:00",
        candidate_business_ids=candidates,
    )

    features = store.features_for(task)

    liked = features.business_scores[candidates[0]]
    disliked = features.business_scores[candidates[1]]
    neutral = features.business_scores[candidates[2]]
    assert liked.positive_similarity > 0
    assert liked.text_score > 0.5
    assert disliked.negative_similarity > 0
    assert disliked.text_score < 0.5
    assert neutral.text_score == pytest.approx(0.5)
    assert any("cozy" in keyword for keyword in features.positive_keywords)
    assert any("noisy" in keyword for keyword in features.negative_keywords)
    assert len(features.positive_keywords) <= 10
    assert len(features.negative_keywords) <= 10
