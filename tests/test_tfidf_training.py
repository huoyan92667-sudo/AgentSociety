import json
from pathlib import Path

import pandas as pd

from yelp_agent.config import TfidfConfig
from yelp_agent.features.text import (
    fit_tfidf_model,
    load_tfidf_vectorizer,
)


def tfidf_test_config() -> TfidfConfig:
    return TfidfConfig(
        stop_words="english",
        ngram_min=1,
        ngram_max=2,
        min_df=1,
        sublinear_tf=True,
        max_features=1_000,
        keyword_count=10,
    )


def write_text_fixture(
    root: Path,
) -> tuple[Path, Path, Path, list[str]]:
    businesses_path = root / "businesses.parquet"
    interactions_path = root / "interactions.parquet"
    histories_path = root / "histories.parquet"
    candidates = [f"candidate-{index:02d}" for index in range(20)]
    businesses = [
        {
            "business_id": "history-positive",
            "name": "Positive History",
            "categories": ["Restaurants", "Noodles"],
            "attributes_json": "{}",
        },
        {
            "business_id": "history-negative",
            "name": "Negative History",
            "categories": ["Restaurants", "Burgers"],
            "attributes_json": "{}",
        },
        {
            "business_id": "validation-business",
            "name": "Validation Business",
            "categories": ["Restaurants", "Desserts"],
            "attributes_json": "{}",
        },
        {
            "business_id": candidates[0],
            "name": "Cozy Noodle House",
            "categories": ["Restaurants", "Noodles"],
            "attributes_json": json.dumps(
                {"Ambience": "quiet cozy", "RestaurantsTakeOut": "True"}
            ),
        },
        {
            "business_id": candidates[1],
            "name": "Noisy Burger Grill",
            "categories": ["Restaurants", "Burgers"],
            "attributes_json": json.dumps(
                {"NoiseLevel": "loud", "RestaurantsTakeOut": "True"}
            ),
        },
        *[
            {
                "business_id": business_id,
                "name": f"Other Business {index}",
                "categories": ["Shopping", "Books"],
                "attributes_json": "{}",
            }
            for index, business_id in enumerate(candidates[2:], start=2)
        ],
    ]
    for business in businesses:
        business.update(
            {
                "address": "",
                "city": "Philadelphia",
                "state": "PA",
                "postal_code": "",
                "latitude": 39.95,
                "longitude": -75.16,
            }
        )
    pd.DataFrame(businesses).to_parquet(businesses_path, index=False)
    pd.DataFrame(
        [
            {
                "review_id": "positive-review",
                "user_id": "user-1",
                "business_id": "history-positive",
                "stars": 5.0,
                "text": "cozy noodles friendly service trainingphrase",
                "date": pd.Timestamp("2020-01-01"),
            },
            {
                "review_id": "negative-review",
                "user_id": "user-1",
                "business_id": "history-negative",
                "stars": 1.0,
                "text": "noisy overpriced terrible service",
                "date": pd.Timestamp("2020-01-02"),
            },
            {
                "review_id": "validation-review",
                "user_id": "user-1",
                "business_id": "validation-business",
                "stars": 5.0,
                "text": "targetleakword secret discovery",
                "date": pd.Timestamp("2020-02-01"),
            },
        ]
    ).to_parquet(interactions_path, index=False)
    pd.DataFrame(
        [
            {
                "task_id": task_id,
                "position": position,
                "review_id": review_id,
            }
            for task_id, review_ids in (
                (
                    "validation:user-1",
                    ["positive-review", "negative-review"],
                ),
                (
                    "test:user-1",
                    [
                        "positive-review",
                        "negative-review",
                        "validation-review",
                    ],
                ),
            )
            for position, review_id in enumerate(review_ids, start=1)
        ]
    ).to_parquet(histories_path, index=False)
    return businesses_path, interactions_path, histories_path, candidates


def test_fits_vocabulary_from_validation_history_but_not_test_addition(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, _ = write_text_fixture(tmp_path)
    artifact_path = tmp_path / "features" / "tfidf_vectorizer.joblib"
    manifest_path = tmp_path / "features" / "tfidf_manifest.json"

    result = fit_tfidf_model(
        businesses,
        interactions,
        histories,
        artifact_path,
        manifest_path,
        tfidf_test_config(),
    )
    vectorizer = load_tfidf_vectorizer(artifact_path)
    vocabulary = set(vectorizer.get_feature_names_out())

    assert result.status == "written"
    assert result.training_review_count == 2
    assert result.business_document_count == 23
    assert "trainingphrase" in vocabulary
    assert "targetleakword" not in vocabulary
    assert artifact_path.is_file()
    assert manifest_path.is_file()


def test_reuses_matching_tfidf_artifact_without_retraining(
    tmp_path: Path,
) -> None:
    businesses, interactions, histories, _ = write_text_fixture(tmp_path)
    artifact_path = tmp_path / "features" / "tfidf_vectorizer.joblib"
    manifest_path = tmp_path / "features" / "tfidf_manifest.json"
    config = tfidf_test_config()
    first = fit_tfidf_model(
        businesses,
        interactions,
        histories,
        artifact_path,
        manifest_path,
        config,
    )
    mtimes = (
        artifact_path.stat().st_mtime_ns,
        manifest_path.stat().st_mtime_ns,
    )

    second = fit_tfidf_model(
        businesses,
        interactions,
        histories,
        artifact_path,
        manifest_path,
        config,
    )

    assert first.status == "written"
    assert second.status == "skipped"
    assert second.vocabulary_size == first.vocabulary_size
    assert (
        artifact_path.stat().st_mtime_ns,
        manifest_path.stat().st_mtime_ns,
    ) == mtimes
