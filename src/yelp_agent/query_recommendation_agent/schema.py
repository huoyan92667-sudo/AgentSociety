"""Target-blind contracts for full Agent runs on Query Recommendation V1."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.agent_evaluation.schema import AgentTurnTrace
from yelp_agent.models import StrictModel
from yelp_agent.query import RecommendationRequest
from yelp_agent.recommendation_evidence.schema import RecommendationEvidenceCard


class QueryRecommendationAgentPrediction(StrictModel):
    """One frozen Agent output produced without loading hidden benchmark labels."""

    schema_version: Literal[1] = 1
    benchmark_id: Literal["query_recommendation_v1"] = "query_recommendation_v1"
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["development", "validation"]
    agent_version: str = Field(min_length=1)
    status: Literal["completed", "awaiting_user", "fallback"]
    response_kind: Literal[
        "none",
        "clarification",
        "recommendation",
        "grounded_answer",
        "uncertain_answer",
        "fallback",
    ]
    request: RecommendationRequest
    retrieval_ranking: list[str] = Field(default_factory=list, max_length=500)
    final_ranking: list[str] = Field(default_factory=list, max_length=1000)
    displayed_business_ids: list[str] = Field(default_factory=list, max_length=5)
    evidence_cards: list[RecommendationEvidenceCard] = Field(
        default_factory=list,
        max_length=5,
    )
    query_aware_result: dict[str, Any] | None = None
    turns: list[AgentTurnTrace] = Field(min_length=1)
    fallback: bool = False
    fallback_reason: str | None = None
    failure_codes: list[str] = Field(default_factory=list)
    action_count: int = Field(ge=0)
    invalid_action_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    hidden_labels_loaded: Literal[False] = False

    @field_validator(
        "retrieval_ranking",
        "final_ranking",
        "displayed_business_ids",
    )
    @classmethod
    def unique_business_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not item for item in values):
            raise ValueError("business rankings must contain unique nonempty IDs")
        return values

    @model_validator(mode="after")
    def validate_prediction(self) -> Self:
        if self.invalid_action_count > self.action_count:
            raise ValueError("invalid actions cannot exceed all actions")
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("input and output tokens must appear together")
        if self.fallback != (self.fallback_reason is not None):
            raise ValueError("fallback and fallback_reason must appear together")
        if self.status == "fallback" and not self.fallback:
            raise ValueError("fallback status requires fallback metadata")
        if self.displayed_business_ids != self.final_ranking[
            : len(self.displayed_business_ids)
        ]:
            raise ValueError("displayed businesses must be the final ranking prefix")
        card_ids = [item.business_id for item in self.evidence_cards]
        if card_ids and card_ids != self.displayed_business_ids[: len(card_ids)]:
            raise ValueError("evidence cards must align with displayed businesses")
        if self.response_kind == "recommendation" and (
            not self.final_ranking or not self.displayed_business_ids
        ):
            raise ValueError("recommendation responses require a ranking and display")
        if self.status == "awaiting_user" and self.response_kind != "clarification":
            raise ValueError("awaiting_user predictions must contain a clarification")
        if (
            self.query_aware_result is not None
            and self.status == "completed"
            and self.response_kind == "recommendation"
        ):
            ranking = self.query_aware_result.get("ranking")
            if isinstance(ranking, list) and self.final_ranking != ranking:
                raise ValueError(
                    "Agent final ranking must equal the Query-aware Tool ranking"
                )
        return self


class QueryRecommendationAgentRunManifest(StrictModel):
    """Hash-addressed proof that the run phase saw only visible cases."""

    schema_version: Literal[1] = 1
    benchmark_id: Literal["query_recommendation_v1"] = "query_recommendation_v1"
    case_count: int = Field(ge=1)
    split_counts: dict[str, int]
    agent_versions: list[str] = Field(min_length=1)
    visible_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hidden_labels_loaded: Literal[False] = False
