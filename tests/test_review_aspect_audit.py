from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.agent.llm import LLMCallResult
from yelp_agent.reviews.audit import (
    AspectAuditItem,
    ReviewAspectAuditor,
    run_review_aspect_audit,
    sample_aspect_audit_items,
    sample_unmatched_aspect_discovery_items,
)
from yelp_agent.reviews.schema import REVIEW_ASPECT_SCHEMA, ReviewAspectRecord


class RecordingAuditLLM:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def generate(self, messages: object) -> LLMCallResult:
        self.calls.append(messages)
        return LLMCallResult(
            status="success",
            content=(
                '{"decisions":[{'
                '"item_id":"item-001",'
                '"aspect_correct":true,'
                '"sentiment_correct":true,'
                '"evidence_supported":true,'
                '"missing_aspects":[],'
                '"error_code":"NONE",'
                '"confidence":0.96}]}'
            ),
            model="deepseek-v4-flash",
            latency_ms=123.0,
            attempt_count=1,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            observed_total_tokens=120,
        )


def test_auditor_validates_structured_decisions_and_reuses_cache(
    tmp_path: Path,
) -> None:
    llm = RecordingAuditLLM()
    auditor = ReviewAspectAuditor(
        llm,
        model_name="deepseek-v4-flash",
        cache_dir=tmp_path / "cache",
    )
    items = (
        AspectAuditItem(
            item_id="item-001",
            sentence="The room was not quiet.",
            candidate_aspect="quiet_environment",
            candidate_sentiment="negative",
            evidence_span="The room was not quiet",
        ),
    )

    first = auditor.audit(items)
    second = auditor.audit(items)

    assert first.status == "success"
    assert first.total_tokens == 120
    assert first.latency_ms == 123.0
    assert first.decisions[0].aspect_correct is True
    assert first.decisions[0].error_code == "NONE"
    assert second.status == "cache_hit"
    assert second.decisions == first.decisions
    assert second.latency_ms == 0
    assert second.total_tokens == 0
    assert second.attempt_count == 0
    assert len(llm.calls) == 1
    serialized_request = str(llm.calls[0])
    assert "business_id" not in serialized_request
    assert "user_id" not in serialized_request
    assert "review_id" not in serialized_request


def test_auditor_cache_changes_when_request_profile_changes(
    tmp_path: Path,
) -> None:
    llm = RecordingAuditLLM()
    item = AspectAuditItem(
        item_id="item-001",
        sentence="The room was not quiet.",
        candidate_aspect="quiet_environment",
        candidate_sentiment="negative",
        evidence_span="The room was not quiet",
    )
    first = ReviewAspectAuditor(
        llm,
        model_name="deepseek-v4-flash",
        cache_dir=tmp_path / "cache",
        request_profile='{"thinking":"enabled"}',
    )
    second = ReviewAspectAuditor(
        llm,
        model_name="deepseek-v4-flash",
        cache_dir=tmp_path / "cache",
        request_profile='{"thinking":"disabled"}',
    )

    assert first.audit((item,)).status == "success"
    assert second.audit((item,)).status == "success"
    assert len(llm.calls) == 2


class InvalidAuditLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def generate(self, messages: object) -> LLMCallResult:
        return LLMCallResult(
            status="success",
            content=self.content,
            model="fake-model",
            latency_ms=1.0,
            attempt_count=1,
        )


def test_auditor_rejects_non_json_and_wrong_item_ids_without_caching(
    tmp_path: Path,
) -> None:
    item = AspectAuditItem(
        item_id="expected",
        sentence="The restaurant was noisy.",
        candidate_aspect="quiet_environment",
        candidate_sentiment="negative",
        evidence_span="The restaurant was noisy",
    )
    non_json = ReviewAspectAuditor(
        InvalidAuditLLM("not-json"),
        model_name="fake-model",
        cache_dir=tmp_path / "cache-1",
    ).audit((item,))
    wrong_id = ReviewAspectAuditor(
        InvalidAuditLLM(
            '{"decisions":[{'
            '"item_id":"unexpected",'
            '"aspect_correct":true,'
            '"sentiment_correct":true,'
            '"evidence_supported":true,'
            '"missing_aspects":[],'
            '"error_code":"NONE",'
            '"confidence":0.9}]}'
        ),
        model_name="fake-model",
        cache_dir=tmp_path / "cache-2",
    ).audit((item,))

    assert non_json.status == "failure"
    assert non_json.failure_reason == "non_json"
    assert wrong_id.status == "failure"
    assert wrong_id.failure_reason == "item_id_mismatch"
    assert list(tmp_path.rglob("*.json")) == []


def test_auditor_discards_only_suggestions_without_exact_evidence(
    tmp_path: Path,
) -> None:
    item = AspectAuditItem(
        item_id="discovery-item",
        sentence="The noodles were generally good.",
    )
    response = {
        "decisions": [
            {
                "item_id": "discovery-item",
                "aspect_correct": None,
                "sentiment_correct": None,
                "evidence_supported": None,
                "missing_aspects": [],
                "suggestions": [
                    {
                        "aspect": "food_quality",
                        "sentiment": "positive",
                        "evidence_span": "noodles were generally good",
                        "suggested_phrase": "generally good",
                        "confidence": 0.9,
                    },
                    {
                        "aspect": "food_quality",
                        "sentiment": "positive",
                        "evidence_span": "excellent noodles",
                        "suggested_phrase": "excellent noodles",
                        "confidence": 0.8,
                    },
                ],
                "error_code": "NONE",
                "confidence": 0.9,
            }
        ]
    }
    auditor = ReviewAspectAuditor(
        InvalidAuditLLM(json.dumps(response)),
        model_name="fake-model",
        cache_dir=tmp_path / "cache",
    )

    result = auditor.audit((item,))

    assert result.status == "success"
    assert result.discarded_suggestion_count == 1
    assert [row.evidence_span for row in result.decisions[0].suggestions] == [
        "noodles were generally good"
    ]


def test_audit_sampling_is_deterministic_stratified_and_anonymous(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for index, (aspect, sentiment) in enumerate(
        [
            ("service", "positive"),
            ("service", "positive"),
            ("service", "negative"),
            ("service", "negative"),
            ("cleanliness", "positive"),
            ("cleanliness", "positive"),
        ]
    ):
        evidence = f"anonymous evidence {index}"
        rows.append(
            ReviewAspectRecord(
                review_id=f"private-review-{index}",
                business_id="private-business",
                user_id="private-user",
                review_time=datetime(2020, 1, index + 1),
                aspect=aspect,
                sentiment=sentiment,
                confidence=0.85,
                evidence_span=evidence,
                evidence_start=0,
                evidence_end=len(evidence),
                source_text_sha256="c" * 64,
                extractor_name="rule_based",
                extractor_version="1.0.0",
            ).model_dump(mode="python")
        )
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=REVIEW_ASPECT_SCHEMA), path)

    first = sample_aspect_audit_items(path, sample_size=4, seed=42)
    second = sample_aspect_audit_items(path, sample_size=4, seed=42)

    assert first == second
    assert len(first) == 4
    assert len({item.item_id for item in first}) == 4
    assert {item.candidate_sentiment for item in first} == {
        "positive",
        "negative",
    }
    serialized = "".join(item.model_dump_json() for item in first)
    assert "private-review" not in serialized
    assert "private-business" not in serialized
    assert "private-user" not in serialized


class EchoAuditLLM:
    def generate(self, messages: object) -> LLMCallResult:
        user_message = messages[1]
        item = json.loads(user_message.content)["items"][0]
        return LLMCallResult(
            status="success",
            content=json.dumps(
                {
                    "decisions": [
                        {
                            "item_id": item["item_id"],
                            "aspect_correct": True,
                            "sentiment_correct": True,
                            "evidence_supported": True,
                            "missing_aspects": [],
                            "error_code": "NONE",
                            "confidence": 0.9,
                        }
                    ]
                }
            ),
            model="fake-model",
            latency_ms=10.0,
            attempt_count=1,
            input_tokens=8,
            output_tokens=2,
            total_tokens=10,
            observed_total_tokens=10,
        )


def test_audit_run_aggregates_batch_agreement_usage_and_latency(
    tmp_path: Path,
) -> None:
    items = tuple(
        AspectAuditItem(
            item_id=f"item-{index}",
            sentence="The room was noisy.",
            candidate_aspect="quiet_environment",
            candidate_sentiment="negative",
            evidence_span="The room was noisy",
        )
        for index in range(2)
    )
    auditor = ReviewAspectAuditor(
        EchoAuditLLM(),
        model_name="fake-model",
        cache_dir=tmp_path / "cache",
    )

    result = run_review_aspect_audit(items, auditor, batch_size=1)

    assert result.summary.status == "complete"
    assert result.summary.sample_size == 2
    assert result.summary.success_batches == 2
    assert result.summary.audited_items == 2
    assert result.summary.aspect_agreement_rate == 1.0
    assert result.summary.sentiment_agreement_rate == 1.0
    assert result.summary.evidence_support_rate == 1.0
    assert result.summary.total_tokens == 20
    assert result.summary.mean_api_latency_ms == 10.0
    assert len(result.traces) == 2


def test_audit_run_continues_with_a_structured_failed_batch(tmp_path: Path) -> None:
    item = AspectAuditItem(
        item_id="failed-item",
        sentence="The room was noisy.",
        candidate_aspect="quiet_environment",
        candidate_sentiment="negative",
        evidence_span="The room was noisy",
    )
    auditor = ReviewAspectAuditor(
        InvalidAuditLLM("not-json"),
        model_name="fake-model",
        cache_dir=tmp_path / "cache",
    )

    result = run_review_aspect_audit((item,), auditor, batch_size=1)

    assert result.summary.status == "failed"
    assert result.summary.failed_batches == 1
    assert result.summary.audited_items == 0
    assert result.batches[0].failure_reason == "non_json"


def test_unmatched_review_sampling_creates_anonymous_discovery_items(
    tmp_path: Path,
) -> None:
    reviews_path = tmp_path / "reviews.parquet"
    from yelp_agent.data.reviews import REVIEW_SCHEMA

    review_rows = []
    for index, text in enumerate(
        [
            "Our waiter disappeared for forty minutes.",
            "We had to shout to hear each other.",
            "Nothing relevant happened here.",
        ]
    ):
        review_rows.append(
            {
                "review_id": f"private-review-{index}",
                "user_id": "private-user",
                "business_id": "private-business",
                "stars": 3.0,
                "useful": 0,
                "funny": 0,
                "cool": 0,
                "text": text,
                "date": datetime(2020, 1, index + 1),
            }
        )
    pq.write_table(
        pa.Table.from_pylist(review_rows, schema=REVIEW_SCHEMA),
        reviews_path,
    )
    records_path = tmp_path / "records.parquet"
    pq.write_table(
        pa.Table.from_pylist([], schema=REVIEW_ASPECT_SCHEMA),
        records_path,
    )

    items = sample_unmatched_aspect_discovery_items(
        reviews_path,
        records_path,
        sample_size=2,
        seed=42,
    )

    assert len(items) == 2
    assert all(item.candidate_aspect is None for item in items)
    assert all(item.candidate_sentiment is None for item in items)
    assert all(item.evidence_span is None for item in items)
    serialized = "".join(item.model_dump_json() for item in items)
    assert "private-review" not in serialized
    assert "private-user" not in serialized
    assert "private-business" not in serialized
