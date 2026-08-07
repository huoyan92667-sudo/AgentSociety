from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from yelp_agent.config import load_review_aspect_settings
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.reviews.artifacts import build_review_aspect_artifacts
from yelp_agent.reviews.extractors.rule_based import RuleBasedAspectExtractor
from yelp_agent.reviews.schema import (
    ASPECT_NAMES,
    REVIEW_ASPECT_SCHEMA,
    ReviewAspectRecord,
    ReviewDocument,
)
from yelp_agent.reviews.store import ReviewAspectStore

PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


def test_review_aspect_schema_exposes_only_the_frozen_general_aspects() -> None:
    assert ASPECT_NAMES == (
        "food_quality",
        "service",
        "price_value",
        "quiet_environment",
        "crowded",
        "queue_time",
        "portion_size",
        "parking",
        "pet_friendly",
        "family_friendly",
        "date_suitable",
        "group_suitable",
        "spiciness",
        "cleanliness",
    )
    assert "fatty_meat" not in ASPECT_NAMES
    assert "lean_meat" not in ASPECT_NAMES

    record = ReviewAspectRecord(
        review_id="review-1",
        business_id="business-1",
        user_id="user-1",
        review_time=datetime(2020, 1, 1),
        aspect="quiet_environment",
        sentiment="positive",
        confidence=0.9,
        evidence_span="The room was quiet",
        evidence_start=0,
        evidence_end=18,
        source_text_sha256="a" * 64,
        extractor_name="rule_based",
        extractor_version="1.0.0",
    )

    assert record.aspect == "quiet_environment"

    with pytest.raises(ValidationError):
        ReviewAspectRecord.model_validate(
            {
                **record.model_dump(),
                "aspect": "fatty_meat",
            }
        )


def test_review_aspect_vocabulary_is_versioned_and_covers_exact_taxonomy() -> None:
    config, vocabulary = load_review_aspect_settings(PROJECT_CONFIG_DIR)

    assert config.schema_version == 1
    assert config.audit.max_tokens == 2000
    assert config.audit.response_format_json is True
    assert config.audit.thinking == "disabled"
    assert config.extractor_version == "1.1.0"
    assert vocabulary.vocabulary_version == 2
    assert config.audit.batch_size == 10
    assert tuple(vocabulary.aspects) == ASPECT_NAMES
    assert "quiet" in vocabulary.aspects["quiet_environment"].positive
    assert "noisy" in vocabulary.aspects["quiet_environment"].negative
    assert "fatty_meat" not in vocabulary.aspects
    assert "lean_meat" not in vocabulary.aspects


def test_rule_extractor_separates_contrasting_aspects_with_exact_evidence() -> None:
    config, vocabulary = load_review_aspect_settings(PROJECT_CONFIG_DIR)
    extractor = RuleBasedAspectExtractor(config, vocabulary)
    text = "The food was delicious, but the service was terrible."

    records = extractor.extract(
        ReviewDocument(
            review_id="review-contrast",
            business_id="business-1",
            user_id="user-1",
            review_time=datetime(2020, 1, 1),
            text=text,
        )
    )

    assert [(record.aspect, record.sentiment) for record in records] == [
        ("food_quality", "positive"),
        ("service", "negative"),
    ]
    assert [record.evidence_span for record in records] == [
        "The food was delicious",
        "the service was terrible",
    ]
    assert all(
        text[record.evidence_start : record.evidence_end] == record.evidence_span
        for record in records
    )


def test_rule_extractor_flips_a_locally_negated_term_and_is_deterministic() -> None:
    config, vocabulary = load_review_aspect_settings(PROJECT_CONFIG_DIR)
    extractor = RuleBasedAspectExtractor(config, vocabulary)
    review = ReviewDocument(
        review_id="review-negation",
        business_id="business-1",
        user_id="user-1",
        review_time=datetime(2020, 1, 1),
        text="The room was not quiet. Dr. Lee said the restaurant was very clean.",
    )

    first = extractor.extract(review)
    second = extractor.extract(review)

    assert first == second
    assert [(record.aspect, record.sentiment) for record in first] == [
        ("quiet_environment", "negative"),
        ("cleanliness", "positive"),
    ]
    assert first[0].confidence == config.negated_confidence
    assert first[1].evidence_span == "Dr. Lee said the restaurant was very clean"


def test_crowded_rule_requires_business_context_for_very_busy() -> None:
    config, vocabulary = load_review_aspect_settings(PROJECT_CONFIG_DIR)
    extractor = RuleBasedAspectExtractor(config, vocabulary)

    business_records = extractor.extract(
        ReviewDocument(
            review_id="review-busy-business",
            business_id="business-1",
            user_id="user-1",
            review_time=datetime(2020, 1, 1),
            text="This place can get very busy on weekends.",
        )
    )
    personal_records = extractor.extract(
        ReviewDocument(
            review_id="review-busy-person",
            business_id="business-1",
            user_id="user-1",
            review_time=datetime(2020, 1, 1),
            text="I had a very busy day of meetings.",
        )
    )

    assert [(row.aspect, row.sentiment) for row in business_records] == [
        ("crowded", "negative")
    ]
    assert all(row.aspect != "crowded" for row in personal_records)


def test_aspect_artifact_builder_is_atomic_counted_and_reusable(tmp_path: Path) -> None:
    reviews_path = tmp_path / "reviews.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "review_id": "r1",
                    "user_id": "u1",
                    "business_id": "b1",
                    "stars": 4.0,
                    "useful": 0,
                    "funny": 0,
                    "cool": 0,
                    "text": "The food was delicious, but the service was terrible.",
                    "date": datetime(2020, 1, 1),
                },
                {
                    "review_id": "r2",
                    "user_id": "u2",
                    "business_id": "b1",
                    "stars": 3.0,
                    "useful": 0,
                    "funny": 0,
                    "cool": 0,
                    "text": "I visited on Tuesday.",
                    "date": datetime(2020, 1, 2),
                },
                {
                    "review_id": "r3",
                    "user_id": "u3",
                    "business_id": "b2",
                    "stars": 4.0,
                    "useful": 0,
                    "funny": 0,
                    "cool": 0,
                    "text": "It was not quiet. The bathroom was very clean.",
                    "date": datetime(2020, 1, 3),
                },
            ],
            schema=REVIEW_SCHEMA,
        ),
        reviews_path,
    )
    config, vocabulary = load_review_aspect_settings(PROJECT_CONFIG_DIR)
    output_root = tmp_path / "aspects"

    written = build_review_aspect_artifacts(
        reviews_path,
        output_root,
        config,
        vocabulary,
        source_scope="selected_user_interactions",
    )
    reused = build_review_aspect_artifacts(
        reviews_path,
        output_root,
        config,
        vocabulary,
        source_scope="selected_user_interactions",
    )

    assert written.status == "written"
    assert written.source_reviews == 3
    assert written.source_scope == "selected_user_interactions"
    assert written.reviews_with_aspects == 2
    assert written.aspect_records == 4
    assert written.aspect_counts == {
        "cleanliness": 1,
        "food_quality": 1,
        "quiet_environment": 1,
        "service": 1,
    }
    assert reused.status == "skipped"
    assert reused.model_dump(exclude={"status"}) == written.model_dump(
        exclude={"status"}
    )
    assert Path(written.records_path).is_file()
    assert Path(written.manifest_path).is_file()
    records = pq.read_table(written.records_path).to_pylist()
    assert all(row["evidence_span"] for row in records)
    assert not (output_root / "aspect_records.parquet.partial").exists()


def test_aspect_store_requires_a_strict_cutoff_for_business_and_user_reads(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "aspect_records.parquet"

    def record(
        review_id: str,
        user_id: str,
        review_time: datetime,
        aspect: str,
    ) -> dict[str, object]:
        return ReviewAspectRecord(
            review_id=review_id,
            business_id="business-1",
            user_id=user_id,
            review_time=review_time,
            aspect=aspect,
            sentiment="positive",
            confidence=0.85,
            evidence_span="clean",
            evidence_start=0,
            evidence_end=5,
            source_text_sha256="b" * 64,
            extractor_name="rule_based",
            extractor_version="1.0.0",
        ).model_dump(mode="python")

    pq.write_table(
        pa.Table.from_pylist(
            [
                record("past", "user-1", datetime(2020, 1, 1), "cleanliness"),
                record("at-cutoff", "user-1", datetime(2020, 2, 1), "service"),
                record("future", "user-2", datetime(2030, 1, 1), "food_quality"),
            ],
            schema=REVIEW_ASPECT_SCHEMA,
        ),
        records_path,
    )

    with ReviewAspectStore(records_path) as store:
        business = store.for_business_before(
            "business-1",
            datetime(2020, 2, 1),
        )
        user = store.for_user_before("user-1", datetime(2020, 2, 1))

    assert [item.review_id for item in business] == ["past"]
    assert [item.review_id for item in user] == ["past"]
    assert not hasattr(ReviewAspectStore, "all_records")
