"""Identity-blind Query rendering and local ownership of hidden labels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel

from .frames import PlannedQueryRecommendationCase
from .schema import (
    QueryRecommendationBenchmarkBundle,
    QueryRecommendationGroundTruth,
    VisibleQueryRecommendationCase,
)


class GeneratedQuery(StrictModel):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_text: str = Field(min_length=3, max_length=2000)


class GeneratedQueryBatch(StrictModel):
    queries: list[GeneratedQuery] = Field(min_length=1)


class QueryFidelityAudit(StrictModel):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool
    reason_codes: list[str]


class QueryFidelityAuditBatch(StrictModel):
    audits: list[QueryFidelityAudit] = Field(min_length=1)


class QueryRewrite(StrictModel):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    language: Literal["zh-CN", "en-US"]
    query_text: str = Field(min_length=3, max_length=2000)
    condition_fidelity_passed: bool
    audit_reason_codes: list[str]

    @model_validator(mode="after")
    def validate_audit(self) -> QueryRewrite:
        if self.condition_fidelity_passed and self.audit_reason_codes:
            raise ValueError("passing rewrites cannot carry audit failures")
        if not self.condition_fidelity_passed and not self.audit_reason_codes:
            raise ValueError("failed rewrites require audit reason codes")
        return self


class QueryGenerationReport(StrictModel):
    schema_version: Literal[1] = 1
    generator_model: str = Field(min_length=1)
    case_count: int = Field(ge=1)
    batch_count: int = Field(ge=1)
    provider_call_count: int = Field(ge=1)
    attempt_count: int = Field(ge=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    total_latency_ms: float = Field(ge=0)
    cumulative_cache_call_count: int = Field(ge=1)
    cumulative_cache_total_tokens: int | None = Field(default=None, ge=0)
    cumulative_cache_latency_ms: float = Field(ge=0)
    generation_prompt_sha256: list[str]
    audit_prompt_sha256: list[str]
    all_condition_fidelity_passed: Literal[True] = True
    target_identity_sent_to_provider: Literal[False] = False
    target_review_text_sent_to_provider: Literal[False] = False


_GENERATION_SYSTEM = """You render canonical Yelp recommendation requirements as one natural user request.
Return one JSON object only, without Markdown. Each case must appear exactly once.
Never add, remove, weaken, strengthen, or contradict a condition. Preserve exact numbers.
Use only the requested language. Mandatory/filter conditions must sound non-negotiable;
strong/rank conditions must sound like clear preferences. Do not mention field names,
coordinates, reviews, datasets, labels, IDs, or any specific business name.
Output: {"queries":[{"case_id":"64 hex chars","query_text":"..."}]}.
"""

_AUDIT_SYSTEM = """You audit whether a generated Yelp request preserves its canonical requirements.
Return one JSON object only, without Markdown. Check category, every condition, direction,
importance, exact numbers, and party size. Also reject invented requirements, business names,
IDs, review references, labels, or dataset language. Use stable uppercase reason codes.
Output: {"audits":[{"case_id":"64 hex chars","passed":true,"reason_codes":[]}]}.
"""


def _canonical_rows(
    drafts: Sequence[PlannedQueryRecommendationCase],
) -> list[dict[str, object]]:
    return [
        {
            "case_id": draft.frame.case_id,
            "language": draft.language,
            "conditions": [
                item.model_dump(mode="json") for item in draft.frame.conditions
            ],
            "party_size": draft.frame.party_size,
        }
        for draft in drafts
    ]


def build_query_rewrite_prompt(
    drafts: Sequence[PlannedQueryRecommendationCase],
) -> tuple[str, str, str]:
    """Serialize meanings only; target/user identity never crosses this seam."""

    if not drafts:
        raise ValueError("rewrite batch cannot be empty")
    user = "Render every canonical case exactly once:\n" + json.dumps(
        _canonical_rows(drafts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256((_GENERATION_SYSTEM + "\n" + user).encode()).hexdigest()
    return _GENERATION_SYSTEM, user, digest


def build_query_audit_prompt(
    drafts: Sequence[PlannedQueryRecommendationCase],
    generated: GeneratedQueryBatch,
) -> tuple[str, str, str]:
    expected = {draft.frame.case_id for draft in drafts}
    actual = {item.case_id for item in generated.queries}
    if len(generated.queries) != len(expected) or actual != expected:
        raise ValueError("generated batch must contain every requested case exactly once")
    query_by_id = {item.case_id: item.query_text for item in generated.queries}
    rows = _canonical_rows(drafts)
    for row in rows:
        row["query_text"] = query_by_id[str(row["case_id"])]
    user = "Audit every generated case:\n" + json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256((_AUDIT_SYSTEM + "\n" + user).encode()).hexdigest()
    return _AUDIT_SYSTEM, user, digest


def parse_generated_queries(
    content: str,
    drafts: Sequence[PlannedQueryRecommendationCase],
) -> GeneratedQueryBatch:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    batch = GeneratedQueryBatch.model_validate_json(stripped)
    expected = {draft.frame.case_id for draft in drafts}
    ids = [item.case_id for item in batch.queries]
    if len(ids) != len(expected) or set(ids) != expected or len(ids) != len(set(ids)):
        raise ValueError("provider must return every requested case exactly once")
    return batch


def parse_fidelity_audits(
    content: str,
    drafts: Sequence[PlannedQueryRecommendationCase],
) -> QueryFidelityAuditBatch:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    batch = QueryFidelityAuditBatch.model_validate_json(stripped)
    expected = {draft.frame.case_id for draft in drafts}
    ids = [item.case_id for item in batch.audits]
    if len(ids) != len(expected) or set(ids) != expected or len(ids) != len(set(ids)):
        raise ValueError("audit must return every requested case exactly once")
    return batch


def combine_generation_and_audit(
    drafts: Sequence[PlannedQueryRecommendationCase],
    generated: GeneratedQueryBatch,
    audited: QueryFidelityAuditBatch,
) -> tuple[QueryRewrite, ...]:
    generated_by_id = {item.case_id: item for item in generated.queries}
    audit_by_id = {item.case_id: item for item in audited.audits}
    return tuple(
        QueryRewrite(
            case_id=draft.frame.case_id,
            language=draft.language,
            query_text=generated_by_id[draft.frame.case_id].query_text,
            condition_fidelity_passed=audit_by_id[draft.frame.case_id].passed,
            audit_reason_codes=audit_by_id[draft.frame.case_id].reason_codes,
        )
        for draft in drafts
    )


def assemble_query_recommendation_bundle(
    drafts: Sequence[PlannedQueryRecommendationCase],
    rewrites: Sequence[QueryRewrite],
    *,
    generator_model: str,
    generator_prompt_sha256: str,
) -> QueryRecommendationBenchmarkBundle:
    """Attach code-owned real positives after provider work is complete."""

    expected = {draft.frame.case_id for draft in drafts}
    rewrite_ids = [item.case_id for item in rewrites]
    if len(rewrite_ids) != len(expected) or set(rewrite_ids) != expected:
        raise ValueError("rewrites must align one-to-one with planned cases")
    if len(rewrite_ids) != len(set(rewrite_ids)):
        raise ValueError("rewrite case IDs must be unique")
    by_id = {item.case_id: item for item in rewrites}
    visible = []
    truth = []
    frames = []
    normalized_queries: set[str] = set()
    for draft in drafts:
        rewrite = by_id[draft.frame.case_id]
        if rewrite.language != draft.language:
            raise ValueError("rewrite language does not match the frame")
        if not rewrite.condition_fidelity_passed:
            raise ValueError(
                f"rewrite failed condition audit: {draft.frame.case_id}"
            )
        normalized = rewrite.query_text.strip().casefold()
        forbidden = (
            draft.anchor.target_business_id.casefold(),
            draft.anchor.target_review_id.casefold(),
        )
        if any(value in normalized for value in forbidden):
            raise ValueError("rewrite leaked hidden target identity")
        if normalized in normalized_queries:
            raise ValueError("visible Query texts must be globally unique")
        normalized_queries.add(normalized)
        visible.append(
            VisibleQueryRecommendationCase(
                case_id=draft.frame.case_id,
                split=draft.anchor.benchmark_split,
                language=draft.language,
                user_id=draft.anchor.user_id,
                session_id=f"query-recommendation:{draft.frame.case_id}",
                cutoff_time=draft.anchor.cutoff_time,
                query_text=rewrite.query_text,
                user_latitude=draft.user_latitude,
                user_longitude=draft.user_longitude,
                generator_kind="openai_compatible",
                generator_model=generator_model,
                generator_prompt_sha256=generator_prompt_sha256,
            )
        )
        truth.append(
            QueryRecommendationGroundTruth(
                case_id=draft.frame.case_id,
                source_task_id=draft.anchor.source_task_id,
                target_business_id=draft.anchor.target_business_id,
                target_review_id=draft.anchor.target_review_id,
                target_stars=draft.anchor.target_stars,
                target_time=draft.anchor.target_time,
            )
        )
        frames.append(draft.frame)
    return QueryRecommendationBenchmarkBundle(
        visible_cases=tuple(visible),
        ground_truth=tuple(truth),
        frames=tuple(frames),
    )
