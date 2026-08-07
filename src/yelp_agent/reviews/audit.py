"""Cached, schema-validated LLM review of rule-based Aspect assertions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError, field_validator, model_validator

from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.experiments import write_json_artifact
from yelp_agent.models import StrictModel
from yelp_agent.reviews.extractors.rule_based import segment_review_clauses
from yelp_agent.reviews.schema import REVIEW_ASPECT_SCHEMA, AspectName, AspectSentiment

_PROMPT_VERSION = "review-aspect-audit-v1"
_SYSTEM_PROMPT = """You audit rule-based labels for English Yelp review clauses.
Return one direct JSON object only, with key \"decisions\". Do not add markdown.
For every input item return exactly one decision with the same item_id.
Check whether the candidate aspect, sentiment, and quoted evidence are supported
by the sentence. List other frozen aspects clearly expressed in the same sentence.
Do not infer facts not stated in the sentence. Do not provide hidden reasoning.
Allowed aspects: food_quality, service, price_value, quiet_environment, crowded,
queue_time, portion_size, parking, pet_friendly, family_friendly, date_suitable,
group_suitable, spiciness, cleanliness.
Allowed error_code values: NONE, WRONG_ASPECT, WRONG_SENTIMENT,
UNSUPPORTED_EVIDENCE, MISSING_ASPECT, AMBIGUOUS.
For discovery items without a candidate assertion, set the three correctness
fields to null and return any clearly supported labels in suggestions.
Every suggestion contains aspect, sentiment, evidence_span, suggested_phrase,
and confidence. suggested_phrase should be a short reusable phrase copied from
the evidence, not a newly invented description.
Each decision must contain: item_id, aspect_correct, sentiment_correct,
evidence_supported, missing_aspects, suggestions, error_code, confidence."""


class _LLM(Protocol):
    def generate(self, messages: Sequence[LLMMessage]) -> LLMCallResult: ...


class AspectAuditItem(StrictModel):
    """An anonymous sentence and one candidate rule assertion."""

    item_id: str = Field(min_length=1)
    sentence: str = Field(min_length=1, max_length=3000)
    candidate_aspect: AspectName | None = None
    candidate_sentiment: AspectSentiment | None = None
    evidence_span: str | None = Field(default=None, min_length=1, max_length=3000)

    @model_validator(mode="after")
    def validate_evidence(self) -> "AspectAuditItem":
        candidate_values = (
            self.candidate_aspect,
            self.candidate_sentiment,
            self.evidence_span,
        )
        if any(value is None for value in candidate_values) and any(
            value is not None for value in candidate_values
        ):
            raise ValueError("audit candidate fields must be all present or all absent")
        if self.evidence_span is not None and self.evidence_span not in self.sentence:
            raise ValueError("audit evidence_span must be an exact sentence substring")
        return self


AuditErrorCode = Literal[
    "NONE",
    "WRONG_ASPECT",
    "WRONG_SENTIMENT",
    "UNSUPPORTED_EVIDENCE",
    "MISSING_ASPECT",
    "AMBIGUOUS",
]


class AspectAuditSuggestion(StrictModel):
    aspect: AspectName
    sentiment: Literal["positive", "negative", "mixed"]
    evidence_span: str = Field(min_length=1, max_length=3000)
    suggested_phrase: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)


class AspectAuditDecision(StrictModel):
    item_id: str = Field(min_length=1)
    aspect_correct: bool | None
    sentiment_correct: bool | None
    evidence_supported: bool | None
    missing_aspects: list[AspectName]
    suggestions: list[AspectAuditSuggestion] = Field(default_factory=list)
    error_code: AuditErrorCode
    confidence: float = Field(ge=0, le=1)

    @field_validator("missing_aspects")
    @classmethod
    def validate_missing_aspects(cls, values: list[AspectName]) -> list[AspectName]:
        if len(set(values)) != len(values):
            raise ValueError("missing_aspects must be unique")
        return values


class _AuditResponse(StrictModel):
    decisions: list[AspectAuditDecision]


class AspectAuditBatchResult(StrictModel):
    status: Literal["success", "cache_hit", "disabled", "failure"]
    model: str | None = None
    prompt_version: Literal["review-aspect-audit-v1"] = _PROMPT_VERSION
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_count: int = Field(ge=1, le=25)
    decisions: list[AspectAuditDecision]
    latency_ms: float = Field(ge=0)
    attempt_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    observed_total_tokens: int | None = Field(default=None, ge=0)
    unknown_usage_attempts: int = Field(default=0, ge=0)
    failure_reason: str | None = None


class AspectAuditTrace(StrictModel):
    batch_index: int = Field(ge=1)
    item: AspectAuditItem
    decision: AspectAuditDecision


class AspectAuditRunSummary(StrictModel):
    status: Literal["complete", "partial", "failed"]
    sample_size: int = Field(ge=1)
    batch_size: int = Field(ge=1, le=25)
    batch_count: int = Field(ge=1)
    success_batches: int = Field(ge=0)
    cache_hit_batches: int = Field(ge=0)
    failed_batches: int = Field(ge=0)
    audited_items: int = Field(ge=0)
    candidate_items: int = Field(ge=0)
    discovery_items: int = Field(ge=0)
    suggestion_count: int = Field(ge=0)
    models: list[str]
    aspect_agreement_rate: float = Field(ge=0, le=1)
    sentiment_agreement_rate: float = Field(ge=0, le=1)
    evidence_support_rate: float = Field(ge=0, le=1)
    error_code_counts: dict[str, int]
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    unknown_usage_attempts: int = Field(ge=0)
    total_api_latency_ms: float = Field(ge=0)
    mean_api_latency_ms: float = Field(ge=0)
    p95_api_latency_ms: float = Field(ge=0)


class AspectAuditExecution(StrictModel):
    summary: AspectAuditRunSummary
    batches: list[AspectAuditBatchResult]
    traces: list[AspectAuditTrace]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class ReviewAspectAuditor:
    """Hide prompt construction, strict parsing, failures, and disk caching."""

    def __init__(
        self,
        llm: _LLM,
        *,
        model_name: str | None,
        cache_dir: str | Path,
        request_profile: str = "default",
    ) -> None:
        self._llm = llm
        self._model_name = model_name
        self._cache_dir = Path(cache_dir)
        self._prompt_sha256 = _sha256_text(_SYSTEM_PROMPT)
        self._request_profile_sha256 = _sha256_text(request_profile)

    def _cache_key(self, items: tuple[AspectAuditItem, ...]) -> str:
        return _sha256_text(
            _canonical(
                {
                    "items": [item.model_dump(mode="json") for item in items],
                    "model": self._model_name,
                    "prompt_sha256": self._prompt_sha256,
                    "prompt_version": _PROMPT_VERSION,
                    "request_profile_sha256": self._request_profile_sha256,
                }
            )
        )

    def _messages(self, items: tuple[AspectAuditItem, ...]) -> list[LLMMessage]:
        payload = {
            "items": [item.model_dump(mode="json") for item in items],
            "required_output": {"decisions": "one decision for every item_id"},
        }
        return [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=_canonical(payload)),
        ]

    def audit(
        self,
        items: Sequence[AspectAuditItem],
    ) -> AspectAuditBatchResult:
        frozen = tuple(items)
        if not frozen or len(frozen) > 25:
            raise ValueError("an audit batch must contain between 1 and 25 items")
        expected_ids = [item.item_id for item in frozen]
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("audit item IDs must be unique")
        cache_key = self._cache_key(frozen)
        cache_path = self._cache_dir / f"{cache_key}.json"
        if cache_path.is_file():
            try:
                cached = AspectAuditBatchResult.model_validate_json(
                    cache_path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError):
                pass
            else:
                if cached.cache_key == cache_key and cached.status == "success":
                    return cached.model_copy(
                        update={
                            "status": "cache_hit",
                            "latency_ms": 0.0,
                            "attempt_count": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "observed_total_tokens": 0,
                            "unknown_usage_attempts": 0,
                        }
                    )

        call = self._llm.generate(self._messages(frozen))
        base = {
            "model": call.model,
            "prompt_sha256": self._prompt_sha256,
            "cache_key": cache_key,
            "item_count": len(frozen),
            "latency_ms": call.latency_ms,
            "attempt_count": call.attempt_count,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "total_tokens": call.total_tokens,
            "observed_total_tokens": call.observed_total_tokens,
            "unknown_usage_attempts": call.unknown_usage_attempts,
        }
        if call.status != "success" or call.content is None:
            return AspectAuditBatchResult(
                status="disabled" if call.status == "disabled" else "failure",
                decisions=[],
                failure_reason=call.failure_reason,
                **base,
            )
        try:
            payload = json.loads(call.content)
        except json.JSONDecodeError:
            return AspectAuditBatchResult(
                status="failure",
                decisions=[],
                failure_reason="non_json",
                **base,
            )
        try:
            parsed = _AuditResponse.model_validate(payload)
        except ValidationError:
            return AspectAuditBatchResult(
                status="failure",
                decisions=[],
                failure_reason="schema_validation",
                **base,
            )
        actual_ids = [decision.item_id for decision in parsed.decisions]
        if actual_ids != expected_ids:
            return AspectAuditBatchResult(
                status="failure",
                decisions=[],
                failure_reason="item_id_mismatch",
                **base,
            )
        for item, decision in zip(frozen, parsed.decisions, strict=True):
            correctness = (
                decision.aspect_correct,
                decision.sentiment_correct,
                decision.evidence_supported,
            )
            if item.candidate_aspect is None:
                if any(value is not None for value in correctness):
                    return AspectAuditBatchResult(
                        status="failure",
                        decisions=[],
                        failure_reason="discovery_schema_validation",
                        **base,
                    )
            elif any(value is None for value in correctness):
                return AspectAuditBatchResult(
                    status="failure",
                    decisions=[],
                    failure_reason="candidate_schema_validation",
                    **base,
                )
            if any(
                suggestion.evidence_span not in item.sentence
                for suggestion in decision.suggestions
            ):
                return AspectAuditBatchResult(
                    status="failure",
                    decisions=[],
                    failure_reason="unsupported_suggestion",
                    **base,
                )
        result = AspectAuditBatchResult(
            status="success",
            decisions=parsed.decisions,
            failure_reason=None,
            **base,
        )
        write_json_artifact(cache_path, result)
        return result


def sample_aspect_audit_items(
    records_path: str | Path,
    *,
    sample_size: int,
    seed: int,
) -> tuple[AspectAuditItem, ...]:
    """Select balanced rule outputs without exposing source identifiers."""

    source = Path(records_path)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"Review Aspect Parquet does not exist: {source}")
    try:
        schema = pq.ParquetFile(source).schema_arrow
    except (OSError, pa.ArrowException) as exc:
        raise ValueError("Could not read Review Aspect Parquet") from exc
    if not schema.equals(REVIEW_ASPECT_SCHEMA, check_metadata=False):
        raise ValueError("Review Aspect Parquet has an unexpected schema")

    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                WITH ranked AS (
                    SELECT
                        review_id,
                        aspect,
                        sentiment,
                        evidence_span,
                        evidence_start,
                        row_number() OVER (
                            PARTITION BY aspect, sentiment
                            ORDER BY sha256(
                                review_id || ':' ||
                                CAST(evidence_start AS VARCHAR) || ':' ||
                                aspect || ':' || CAST(? AS VARCHAR)
                            )
                        ) AS stratum_rank
                    FROM read_parquet(?)
                    WHERE sentiment IN ('positive', 'negative', 'mixed')
                )
                SELECT
                    review_id,
                    aspect,
                    sentiment,
                    evidence_span,
                    evidence_start
                FROM ranked
                ORDER BY
                    stratum_rank,
                    sha256(aspect || ':' || sentiment || ':' || CAST(? AS VARCHAR)),
                    review_id,
                    evidence_start
                LIMIT ?
                """,
                [seed, str(source), seed, sample_size],
            ).fetchall()
    except duckdb.Error as exc:
        raise ValueError("Could not sample Review Aspect audit items") from exc

    items: list[AspectAuditItem] = []
    for review_id, aspect, sentiment, evidence_span, evidence_start in rows:
        item_id = _sha256_text(f"{review_id}:{aspect}:{sentiment}:{evidence_start}")[
            :24
        ]
        items.append(
            AspectAuditItem.model_validate(
                {
                    "item_id": item_id,
                    "sentence": evidence_span,
                    "candidate_aspect": aspect,
                    "candidate_sentiment": sentiment,
                    "evidence_span": evidence_span,
                }
            )
        )
    if not items:
        raise ValueError("Review Aspect artifact contains no auditable records")
    return tuple(items)


def sample_unmatched_aspect_discovery_items(
    reviews_path: str | Path,
    records_path: str | Path,
    *,
    sample_size: int,
    seed: int,
) -> tuple[AspectAuditItem, ...]:
    """Sample anonymous clauses from reviews where V1 found no Aspect at all."""

    reviews = Path(reviews_path)
    records = Path(records_path)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    for path, expected in (
        (reviews, REVIEW_SCHEMA),
        (records, REVIEW_ASPECT_SCHEMA),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required Parquet does not exist: {path}")
        try:
            schema = pq.ParquetFile(path).schema_arrow
        except (OSError, pa.ArrowException) as exc:
            raise ValueError(f"Could not read Parquet: {path}") from exc
        if not schema.equals(expected, check_metadata=False):
            raise ValueError(f"Parquet has an unexpected schema: {path}")
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                SELECT r.review_id, r.text
                FROM read_parquet(?) AS r
                WHERE length(trim(r.text)) >= 20
                  AND NOT EXISTS (
                      SELECT 1
                      FROM read_parquet(?) AS a
                      WHERE a.review_id = r.review_id
                  )
                ORDER BY sha256(r.review_id || ':' || CAST(? AS VARCHAR))
                LIMIT ?
                """,
                [str(reviews), str(records), seed, max(sample_size * 4, 20)],
            ).fetchall()
    except duckdb.Error as exc:
        raise ValueError("Could not sample unmatched Review Aspect clauses") from exc

    candidates: list[tuple[str, str]] = []
    for review_id, text in rows:
        for clause_index, clause in enumerate(segment_review_clauses(str(text))):
            if 20 <= len(clause) <= 1000:
                anonymous_id = _sha256_text(
                    f"discovery:{review_id}:{clause_index}"
                )[:24]
                candidates.append((anonymous_id, clause))
    candidates.sort(key=lambda row: _sha256_text(f"{seed}:{row[0]}"))
    items = tuple(
        AspectAuditItem(item_id=item_id, sentence=clause)
        for item_id, clause in candidates[:sample_size]
    )
    if len(items) < sample_size:
        raise ValueError("Not enough unmatched review clauses for requested sample")
    return items


def run_review_aspect_audit(
    items: Sequence[AspectAuditItem],
    auditor: ReviewAspectAuditor,
    *,
    batch_size: int,
) -> AspectAuditExecution:
    """Audit all sampled items, continuing safely after failed batches."""

    frozen = tuple(items)
    if not frozen:
        raise ValueError("audit run requires at least one item")
    if not 1 <= batch_size <= 25:
        raise ValueError("batch_size must be between 1 and 25")
    batches: list[AspectAuditBatchResult] = []
    traces: list[AspectAuditTrace] = []
    for start in range(0, len(frozen), batch_size):
        batch_index = len(batches) + 1
        batch_items = frozen[start : start + batch_size]
        result = auditor.audit(batch_items)
        batches.append(result)
        if not result.decisions:
            continue
        for item, decision in zip(
            batch_items,
            result.decisions,
            strict=True,
        ):
            traces.append(
                AspectAuditTrace(
                    batch_index=batch_index,
                    item=item,
                    decision=decision,
                )
            )

    audited_items = len(traces)
    success_batches = sum(batch.status == "success" for batch in batches)
    cache_hit_batches = sum(batch.status == "cache_hit" for batch in batches)
    failed_batches = len(batches) - success_batches - cache_hit_batches
    if audited_items == len(frozen):
        status: Literal["complete", "partial", "failed"] = "complete"
    elif audited_items:
        status = "partial"
    else:
        status = "failed"

    def rate(attribute: str) -> float:
        candidates = [
            trace
            for trace in traces
            if getattr(trace.decision, attribute) is not None
        ]
        if not candidates:
            return 0.0
        return sum(
            bool(getattr(trace.decision, attribute)) for trace in candidates
        ) / len(candidates)

    error_counts: dict[str, int] = {}
    for trace in traces:
        code = trace.decision.error_code
        error_counts[code] = error_counts.get(code, 0) + 1
    api_latencies = [
        batch.latency_ms
        for batch in batches
        if batch.status != "cache_hit" and batch.attempt_count > 0
    ]
    ordered_latencies = sorted(api_latencies)
    p95_index = max(0, (95 * len(ordered_latencies) + 99) // 100 - 1)
    total_latency = sum(api_latencies)
    summary = AspectAuditRunSummary(
        status=status,
        sample_size=len(frozen),
        batch_size=batch_size,
        batch_count=len(batches),
        success_batches=success_batches,
        cache_hit_batches=cache_hit_batches,
        failed_batches=failed_batches,
        audited_items=audited_items,
        candidate_items=sum(
            trace.item.candidate_aspect is not None for trace in traces
        ),
        discovery_items=sum(
            trace.item.candidate_aspect is None for trace in traces
        ),
        suggestion_count=sum(
            len(trace.decision.suggestions) for trace in traces
        ),
        models=sorted({batch.model for batch in batches if batch.model}),
        aspect_agreement_rate=rate("aspect_correct"),
        sentiment_agreement_rate=rate("sentiment_correct"),
        evidence_support_rate=rate("evidence_supported"),
        error_code_counts=dict(sorted(error_counts.items())),
        total_input_tokens=sum(batch.input_tokens or 0 for batch in batches),
        total_output_tokens=sum(batch.output_tokens or 0 for batch in batches),
        total_tokens=sum(batch.total_tokens or 0 for batch in batches),
        unknown_usage_attempts=sum(batch.unknown_usage_attempts for batch in batches),
        total_api_latency_ms=total_latency,
        mean_api_latency_ms=(
            total_latency / len(api_latencies) if api_latencies else 0.0
        ),
        p95_api_latency_ms=(ordered_latencies[p95_index] if ordered_latencies else 0.0),
    )
    return AspectAuditExecution(summary=summary, batches=batches, traces=traces)
