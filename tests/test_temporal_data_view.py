from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.data.temporal_view import TemporalDataView


def _write_view_fixture(root: Path) -> tuple[Path, Path, Path]:
    businesses = root / "businesses.parquet"
    reviews = root / "reviews.parquet"
    interactions = root / "interactions.parquet"
    pd.DataFrame(
        [
            {
                "business_id": "business-1",
                "name": "First Place",
                "address": "1 Main St",
                "city": "Philadelphia",
                "state": "PA",
                "postal_code": "19100",
                "latitude": 39.95,
                "longitude": -75.16,
                "categories": ["Restaurants", "Mexican"],
                "attributes_json": '{"OutdoorSeating":true}',
                "stars": 1.0,
                "review_count": 999,
            },
            {
                "business_id": "business-2",
                "name": "Second Place",
                "address": "2 Main St",
                "city": "Philadelphia",
                "state": "PA",
                "postal_code": "19101",
                "latitude": 40.0,
                "longitude": -75.2,
                "categories": ["Food", "Coffee & Tea"],
                "attributes_json": "{}",
                "stars": 5.0,
                "review_count": 999,
            },
        ]
    ).to_parquet(businesses, index=False)
    pd.DataFrame(
        [
            {
                "review_id": "review-1",
                "user_id": "user-1",
                "business_id": "business-1",
                "stars": 5.0,
                "text": "old positive",
                "date": pd.Timestamp("2020-01-01"),
            },
            {
                "review_id": "review-2",
                "user_id": "user-1",
                "business_id": "business-2",
                "stars": 3.0,
                "text": "at cutoff",
                "date": pd.Timestamp("2020-02-01"),
            },
            {
                "review_id": "review-3",
                "user_id": "user-1",
                "business_id": "business-1",
                "stars": 1.0,
                "text": "future negative",
                "date": pd.Timestamp("2030-01-01"),
            },
        ]
    ).to_parquet(reviews, index=False)
    pd.read_parquet(reviews).to_parquet(interactions, index=False)
    return businesses, reviews, interactions


def test_static_businesses_exclude_snapshot_scores_and_are_read_only(
    tmp_path: Path,
) -> None:
    view = TemporalDataView(*_write_view_fixture(tmp_path))

    business = view.business("business-1")
    first_attributes = business.attributes_dict()
    first_attributes["OutdoorSeating"] = False

    assert not hasattr(business, "stars")
    assert not hasattr(business, "review_count")
    assert business.attributes_dict() == {"OutdoorSeating": True}
    with pytest.raises(FrozenInstanceError):
        business.name = "changed"  # type: ignore[misc]
    assert not hasattr(view, "all_reviews")


def test_every_temporal_query_is_strictly_before_cutoff(tmp_path: Path) -> None:
    view = TemporalDataView(*_write_view_fixture(tmp_path))
    cutoff = pd.Timestamp("2020-02-01").to_pydatetime()

    history = view.user_history("user-1", cutoff)
    reviews = view.reviews_before("business-1", cutoff)
    interactions = view.interactions_before(cutoff)
    statistics = view.review_statistics_before(["business-1"], cutoff)

    assert [item.review_id for item in history] == ["review-1"]
    assert [item.stars for item in reviews] == [5.0]
    assert [item.review_id for item in interactions] == ["review-1"]
    assert all(item.date < cutoff for item in (*history, *reviews, *interactions))
    assert statistics.global_count == 1
    assert statistics.global_mean_rating == pytest.approx(5.0)
    assert statistics.businesses["business-1"].count == 1


def test_review_limit_returns_latest_prior_rows_in_chronological_order(
    tmp_path: Path,
) -> None:
    view = TemporalDataView(*_write_view_fixture(tmp_path))

    reviews = view.reviews_before(
        "business-1",
        pd.Timestamp("2040-01-01").to_pydatetime(),
        limit=1,
    )

    assert len(reviews) == 1
    assert reviews[0].stars == 1.0
    assert reviews[0].date == pd.Timestamp("2030-01-01")


def test_same_time_reviews_use_review_id_as_stable_tie_break(
    tmp_path: Path,
) -> None:
    businesses, reviews_path, interactions = _write_view_fixture(tmp_path)
    rows = pd.read_parquet(reviews_path)
    tied = rows.iloc[[0]].copy()
    tied.loc[:, "review_id"] = "review-0"
    pd.concat([rows, tied], ignore_index=True).to_parquet(
        reviews_path,
        index=False,
    )
    view = TemporalDataView(businesses, reviews_path, interactions)

    reviews = view.reviews_before(
        "business-1",
        pd.Timestamp("2020-02-01").to_pydatetime(),
    )

    assert [review.review_id for review in reviews] == [
        "review-0",
        "review-1",
    ]
