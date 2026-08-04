"""Strict parsing and safe merging of LLM Top-8 reranking output."""

from __future__ import annotations

import json

from pydantic import Field, ValidationError, field_validator

from yelp_agent.models import StrictModel


TOP_K_TO_RERANK = 8
FINAL_RANKING_COUNT = 20


class AgentResponseError(RuntimeError):
    """A safe, categorized LLM response failure used for Hybrid fallback."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ParsedRerank(StrictModel):
    ranking: list[str]
    reason: str | None = Field(default=None, min_length=1, max_length=1000)

    @field_validator("ranking")
    @classmethod
    def validate_ranking_ids(cls, values: list[str]) -> list[str]:
        if any(not value or value != value.strip() for value in values):
            raise ValueError(
                "ranking IDs must be nonempty and contain no surrounding whitespace"
            )
        return values


def _validate_expected_ids(expected_business_ids: list[str]) -> None:
    if len(expected_business_ids) != TOP_K_TO_RERANK:
        raise ValueError("expected_business_ids must contain exactly eight IDs")
    if len(set(expected_business_ids)) != len(expected_business_ids):
        raise ValueError("expected_business_ids must be unique")
    if any(
        not business_id or business_id != business_id.strip()
        for business_id in expected_business_ids
    ):
        raise ValueError("expected_business_ids contain an invalid ID")


def parse_rerank_response(
    content: str,
    *,
    expected_business_ids: list[str],
) -> ParsedRerank:
    """Parse a direct JSON object and require an exact Top-8 permutation."""

    _validate_expected_ids(expected_business_ids)
    if not isinstance(content, str) or not content.strip():
        raise AgentResponseError("empty_response")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AgentResponseError("non_json") from exc
    try:
        parsed = ParsedRerank.model_validate(payload)
    except ValidationError as exc:
        raise AgentResponseError("schema_validation") from exc
    if parsed.ranking != list(dict.fromkeys(parsed.ranking)):
        raise AgentResponseError("duplicate_id")
    unexpected = set(parsed.ranking).difference(expected_business_ids)
    if unexpected:
        raise AgentResponseError("unknown_id")
    missing = set(expected_business_ids).difference(parsed.ranking)
    if missing:
        raise AgentResponseError("missing_id")
    if len(parsed.ranking) != len(expected_business_ids):
        raise AgentResponseError("candidate_count_mismatch")
    return parsed


def merge_reranked_top_k(
    parsed: ParsedRerank,
    hybrid_ranking: list[str],
) -> list[str]:
    """Replace only Hybrid Top-8 and preserve positions 9 through 20."""

    if (
        len(hybrid_ranking) != FINAL_RANKING_COUNT
        or len(set(hybrid_ranking)) != FINAL_RANKING_COUNT
        or any(
            not business_id or business_id != business_id.strip()
            for business_id in hybrid_ranking
        )
    ):
        raise ValueError("hybrid_ranking must contain 20 unique valid IDs")
    expected_top = hybrid_ranking[:TOP_K_TO_RERANK]
    if len(parsed.ranking) != TOP_K_TO_RERANK or set(parsed.ranking) != set(
        expected_top
    ):
        raise AgentResponseError("top_k_mismatch")
    final_ranking = [*parsed.ranking, *hybrid_ranking[TOP_K_TO_RERANK:]]
    if (
        len(final_ranking) != FINAL_RANKING_COUNT
        or len(set(final_ranking)) != FINAL_RANKING_COUNT
        or set(final_ranking) != set(hybrid_ranking)
    ):
        raise AgentResponseError("final_ranking_invalid")
    return final_ranking
