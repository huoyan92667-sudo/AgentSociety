from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.config import load_review_aspect_settings
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.review_rag import (
    ReviewRAGConfig,
    ReviewRAGPolicy,
    ReviewRAGStore,
    ReviewRetriever,
    ReviewSearchRequest,
    build_review_rag_artifacts,
    infer_review_aspects,
)
from yelp_agent.reviews.schema import REVIEW_ASPECT_SCHEMA


def _config() -> ReviewRAGConfig:
    return ReviewRAGConfig(
        agent_version="test-review-rag",
        segment_max_chars=120,
        segment_min_chars=5,
        max_segments_per_review=4,
        parquet_batch_size=100,
        minimum_aspect_confidence=0.85,
        bm25_candidate_limit=10,
        aspect_candidate_limit=10,
        embedding_candidate_limit=5,
        max_store_segments_per_business=100,
        embedding_dimension=1024,
        query_instruction="Retrieve relevant Yelp review evidence.",
        review_document_version="test-v1",
        embedding_cache_relative_path="data/test-review-cache",
        policy_relative_path="configs/test-review-policy.json",
    )


def _write_sources(root: Path) -> tuple[Path, Path]:
    reviews = root / "reviews.parquet"
    rows = [
        {
            "review_id": "r1",
            "user_id": "u1",
            "business_id": "b1",
            "stars": 5.0,
            "useful": 3,
            "funny": 0,
            "cool": 1,
            "text": "A quiet peaceful dining room. Easy to talk with friends.",
            "date": datetime(2020, 1, 1),
        },
        {
            "review_id": "r2",
            "user_id": "u2",
            "business_id": "b1",
            "stars": 2.0,
            "useful": 2,
            "funny": 0,
            "cool": 0,
            "text": "The room was noisy and very loud during dinner.",
            "date": datetime(2020, 2, 1),
        },
        {
            "review_id": "future",
            "user_id": "u3",
            "business_id": "b1",
            "stars": 5.0,
            "useful": 99,
            "funny": 0,
            "cool": 0,
            "text": "Future quiet evidence must never leak.",
            "date": datetime(2022, 1, 1),
        },
        {
            "review_id": "decoy",
            "user_id": "u4",
            "business_id": "b2",
            "stars": 5.0,
            "useful": 100,
            "funny": 0,
            "cool": 0,
            "text": "Perfect quiet peaceful calm atmosphere.",
            "date": datetime(2020, 1, 1),
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=REVIEW_SCHEMA), reviews)
    aspects = root / "aspects.parquet"
    aspect_rows = [
        {
            "review_id": "r1",
            "business_id": "b1",
            "user_id": "u1",
            "review_time": datetime(2020, 1, 1),
            "aspect": "quiet_environment",
            "sentiment": "positive",
            "confidence": 0.9,
            "evidence_span": "quiet",
            "evidence_start": 2,
            "evidence_end": 7,
            "source_text_sha256": "0" * 64,
            "extractor_name": "test",
            "extractor_version": "1",
        },
        {
            "review_id": "r2",
            "business_id": "b1",
            "user_id": "u2",
            "review_time": datetime(2020, 2, 1),
            "aspect": "quiet_environment",
            "sentiment": "negative",
            "confidence": 0.9,
            "evidence_span": "noisy",
            "evidence_start": 13,
            "evidence_end": 18,
            "source_text_sha256": "1" * 64,
            "extractor_name": "test",
            "extractor_version": "1",
        },
    ]
    pq.write_table(pa.Table.from_pylist(aspect_rows, schema=REVIEW_ASPECT_SCHEMA), aspects)
    return reviews, aspects


def test_review_rag_is_business_scoped_cutoff_safe_and_deduplicated(
    tmp_path: Path,
) -> None:
    reviews, aspects = _write_sources(tmp_path)
    artifact_root = tmp_path / "rag"
    first = build_review_rag_artifacts(reviews, artifact_root, _config())
    second = build_review_rag_artifacts(reviews, artifact_root, _config())
    assert first.status == "written"
    assert second.status == "skipped"
    store = ReviewRAGStore(first.segments_path, aspects, _config())
    _, vocabulary = load_review_aspect_settings("configs")
    retriever = ReviewRetriever(
        store=store,
        config=_config(),
        policy=ReviewRAGPolicy(
            policy_version="test",
            aspect_weight=1,
            bm25_weight=1,
            embedding_weight=0,
        ),
        vocabulary=vocabulary,
    )
    result = retriever.search(
        ReviewSearchRequest(
            query_text="Is this place quiet?",
            business_ids=["b1"],
            cutoff_time=datetime(2021, 1, 1),
            aspects=["quiet_environment"],
            usage_scope="scenario-test",
        )
    )
    repeated = retriever.search(
        ReviewSearchRequest(
            query_text="Is this place quiet?",
            business_ids=["b1"],
            cutoff_time=datetime(2021, 1, 1),
            aspects=["quiet_environment"],
            usage_scope="scenario-test",
        )
    )
    assert {item.review_id for item in result.hits} == {"r1", "r2"}
    assert all(item.business_id == "b1" for item in result.hits)
    assert all(item.review_time < datetime(2021, 1, 1) for item in result.hits)
    assert len({item.review_id for item in result.hits}) == len(result.hits)
    assert result.embedding_usage.api_calls == 0
    assert result.route_result_counts["aspect"] == 2
    assert result.model_dump_json() == repeated.model_dump_json()
    store.close()


def test_review_rag_multibusiness_top_five_preserves_scope(tmp_path: Path) -> None:
    reviews, aspects = _write_sources(tmp_path)
    artifact_root = tmp_path / "rag"
    result = build_review_rag_artifacts(reviews, artifact_root, _config())
    store = ReviewRAGStore(result.segments_path, aspects, _config())
    _, vocabulary = load_review_aspect_settings("configs")
    retriever = ReviewRetriever(
        store=store,
        config=_config(),
        policy=ReviewRAGPolicy(
            policy_version="test",
            aspect_weight=1,
            bm25_weight=1,
            embedding_weight=0,
        ),
        vocabulary=vocabulary,
    )
    found = retriever.search(
        ReviewSearchRequest(
            query_text="quiet atmosphere",
            business_ids=["b1", "b2"],
            cutoff_time=datetime(2021, 1, 1),
            aspects=["quiet_environment"],
            usage_scope="comparison-test",
        )
    )
    assert len(found.hits) <= 5
    assert {item.business_id for item in found.hits} == {"b1", "b2"}
    store.close()


def test_visible_query_aspect_hints_cover_chinese_and_english() -> None:
    assert infer_review_aspects("这家店停车方便吗？") == ["parking"]
    assert infer_review_aspects("Which is reviewed better for group suitable?") == [
        "group_suitable"
    ]
    assert infer_review_aspects("Reviews conflict over food quality") == [
        "food_quality"
    ]
