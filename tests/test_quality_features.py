from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.features.quality import TemporalQualityStore


def test_scores_business_quality_from_reviews_before_cutoff(
    tmp_path: Path,
) -> None:
    reviews_path = tmp_path / "reviews.parquet"
    business_ids = [f"business-{index:02d}" for index in range(20)]
    pd.DataFrame(
        [
            {
                "business_id": business_id,
                "stars": 5.0 if business_id == "business-00" else 3.0,
                "date": pd.Timestamp("2020-01-01"),
            }
            for business_id in business_ids
        ]
    ).to_parquet(reviews_path, index=False)
    store = TemporalQualityStore(reviews_path, prior_count=20)

    scores = store.score_businesses(
        business_ids,
        pd.Timestamp("2020-02-01").to_pydatetime(),
    )

    global_mean = 3.1
    expected_bayesian_rating = (5.0 + 20 * global_mean) / 21
    expected_normalized_rating = (expected_bayesian_rating - 1.0) / 4.0
    assert set(scores) == set(business_ids)
    assert scores["business-00"].review_count == 1
    assert scores["business-00"].bayesian_rating == pytest.approx(
        expected_bayesian_rating
    )
    assert scores["business-00"].normalized_popularity == pytest.approx(1.0)
    assert scores["business-00"].quality_score == pytest.approx(
        0.8 * expected_normalized_rating + 0.2
    )
    assert (
        scores["business-00"].quality_score
        > scores["business-01"].quality_score
    )


def test_future_reviews_cannot_change_quality_scores(
    tmp_path: Path,
) -> None:
    reviews_path = tmp_path / "reviews.parquet"
    business_ids = [f"business-{index:02d}" for index in range(20)]
    historical_reviews = pd.DataFrame(
        [
            {
                "business_id": business_id,
                "stars": float(index % 5 + 1),
                "date": pd.Timestamp("2020-01-01"),
            }
            for index, business_id in enumerate(business_ids)
        ]
    )
    historical_reviews.to_parquet(reviews_path, index=False)
    cutoff = pd.Timestamp("2020-02-01").to_pydatetime()
    before = TemporalQualityStore(reviews_path).score_businesses(
        business_ids,
        cutoff,
    )

    future_reviews = pd.DataFrame(
        [
            {
                "business_id": business_ids[index % 2],
                "stars": 1.0 if index % 2 == 0 else 5.0,
                "date": pd.Timestamp("2030-01-01"),
            }
            for index in range(200)
        ]
    )
    pd.concat(
        [historical_reviews, future_reviews],
        ignore_index=True,
    ).to_parquet(reviews_path, index=False)
    after = TemporalQualityStore(reviews_path).score_businesses(
        business_ids,
        cutoff,
    )

    assert after == before
