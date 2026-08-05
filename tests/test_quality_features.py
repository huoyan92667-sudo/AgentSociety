from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.features.quality import TemporalQualityStore


def write_quality_view(
    root: Path,
    reviews: pd.DataFrame,
) -> TemporalDataView:
    businesses_path = root / "businesses.parquet"
    reviews_path = root / "reviews.parquet"
    interactions_path = root / "interactions.parquet"
    business_ids = list(dict.fromkeys(reviews["business_id"].tolist()))
    pd.DataFrame(
        [
            {
                "business_id": business_id,
                "name": business_id,
                "address": "",
                "city": "Philadelphia",
                "state": "PA",
                "postal_code": "",
                "latitude": 39.95,
                "longitude": -75.16,
                "categories": ["Restaurants"],
                "attributes_json": "{}",
            }
            for business_id in business_ids
        ]
    ).to_parquet(businesses_path, index=False)
    review_rows = reviews.copy()
    if "review_id" not in review_rows:
        review_rows.insert(
            0,
            "review_id",
            [f"review-{index}" for index in range(len(review_rows))],
        )
    review_rows.to_parquet(reviews_path, index=False)
    first = review_rows.iloc[0]
    pd.DataFrame(
        [
            {
                "review_id": "interaction-1",
                "user_id": "user-1",
                "business_id": first["business_id"],
                "stars": first["stars"],
                "text": "history",
                "date": first["date"],
            }
        ]
    ).to_parquet(interactions_path, index=False)
    return TemporalDataView(
        businesses_path,
        reviews_path,
        interactions_path,
    )


def test_scores_business_quality_from_reviews_before_cutoff(
    tmp_path: Path,
) -> None:
    business_ids = [f"business-{index:02d}" for index in range(20)]
    reviews = pd.DataFrame(
        [
            {
                "business_id": business_id,
                "stars": 5.0 if business_id == "business-00" else 3.0,
                "date": pd.Timestamp("2020-01-01"),
            }
            for business_id in business_ids
        ]
    )
    store = TemporalQualityStore(
        write_quality_view(tmp_path, reviews),
        prior_count=20,
    )

    scores = store.score_businesses(
        business_ids,
        pd.Timestamp("2020-02-01").to_pydatetime(),
    )
    catalog = store.score_catalog(
        pd.Timestamp("2020-02-01").to_pydatetime()
    )
    catalog_scores = dict(
        zip(catalog.business_ids, catalog.quality_scores, strict=True)
    )
    catalog_counts = dict(
        zip(catalog.business_ids, catalog.review_counts, strict=True)
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
    assert catalog_counts["business-00"] == 1
    assert catalog_scores == pytest.approx(
        {business_id: value.quality_score for business_id, value in scores.items()}
    )


def test_future_reviews_cannot_change_quality_scores(
    tmp_path: Path,
) -> None:
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
    cutoff = pd.Timestamp("2020-02-01").to_pydatetime()
    before = TemporalQualityStore(
        write_quality_view(tmp_path, historical_reviews)
    ).score_businesses(
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
    combined = pd.concat(
        [historical_reviews, future_reviews],
        ignore_index=True,
    )
    after = TemporalQualityStore(
        write_quality_view(tmp_path, combined)
    ).score_businesses(
        business_ids,
        cutoff,
    )

    assert after == before
