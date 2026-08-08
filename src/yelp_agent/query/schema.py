"""Stable contracts for one current, point-in-time recommendation request."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import Field, computed_field, field_validator, model_validator

from yelp_agent.models import LocationCenter, StrictModel

type RequestIntent = Literal[
    "recommendation_request",
    "business_detail_question",
    "candidate_comparison",
    "feedback_refinement",
    "unknown",
]
type ConditionField = Literal[
    "category",
    "distance_km",
    "budget_per_person",
    "price_level",
    "quiet_environment",
    "crowded",
    "queue_time",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "spiciness",
    "cleanliness",
    "food_quality",
    "service",
    "price_value",
]
type ConditionOperator = Literal[
    "includes",
    "excludes",
    "equals",
    "less_than_or_equal",
    "greater_than_or_equal",
    "prefer",
    "avoid",
]
type RequirementImportance = Literal["mandatory", "strong", "preferred"]
type EnforcementMode = Literal["filter", "rank", "clarify", "evidence"]
type UnknownPolicy = Literal[
    "exclude",
    "ask",
    "allow_with_warning",
    "not_applicable",
]
type ConditionSource = Literal["rule", "semantic_model"]
type MissingField = Literal[
    "user_location",
    "budget_precision",
    "desired_category",
    "party_size",
    "ambiguous_requirement",
]
type ConditionValue = str | int | float | bool


class QueryParseInput(StrictModel):
    """One raw user turn plus deterministic context known before parsing."""

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    cutoff_time: datetime
    query_text: str = Field(min_length=1, max_length=2000)
    user_latitude: float | None = Field(default=None, ge=-90, le=90)
    user_longitude: float | None = Field(default=None, ge=-180, le=180)
    referenced_business_ids: list[str] = Field(default_factory=list)

    @field_validator("query_text")
    @classmethod
    def validate_query_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query_text cannot be blank")
        return stripped

    @field_validator("referenced_business_ids")
    @classmethod
    def validate_references(cls, values: list[str]) -> list[str]:
        if any(not value or value != value.strip() for value in values):
            raise ValueError("referenced business IDs must be nonempty")
        if len(set(values)) != len(values):
            raise ValueError("referenced business IDs must be unique")
        return values

    @model_validator(mode="after")
    def validate_location(self) -> QueryParseInput:
        if (self.user_latitude is None) != (self.user_longitude is None):
            raise ValueError("user coordinates must be both present or both absent")
        return self


class RequestCondition(StrictModel):
    """One traceable user requirement and its deterministic execution policy."""

    field: ConditionField
    operator: ConditionOperator
    value: ConditionValue
    importance: RequirementImportance
    enforcement: EnforcementMode
    explicit: bool
    confidence: float = Field(ge=0, le=1)
    evidence_span: str = Field(min_length=1)
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(gt=0)
    source: ConditionSource
    unknown_policy: UnknownPolicy

    @model_validator(mode="after")
    def validate_policy_and_evidence(self) -> RequestCondition:
        if self.evidence_end - self.evidence_start != len(self.evidence_span):
            raise ValueError("condition evidence offsets must match evidence_span")
        if self.enforcement == "filter" and self.importance != "mandatory":
            raise ValueError("only mandatory requirements may filter candidates")
        if self.enforcement == "filter" and self.unknown_policy == "not_applicable":
            raise ValueError("filters require an explicit unknown-data policy")
        return self


class RecommendationRequest(StrictModel):
    """Canonical Agent input produced from one current user request."""

    schema_version: Literal[1] = 1
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    cutoff_time: datetime
    query_text: str = Field(min_length=1)
    intent: RequestIntent
    conditions: list[RequestCondition]
    party_size: int | None = Field(default=None, ge=1, le=100)
    location_center: LocationCenter | None = None
    missing_fields: list[MissingField] = Field(default_factory=list)
    referenced_business_ids: list[str] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)
    parser_version: str = Field(min_length=1)

    @computed_field
    @property
    def hard_constraints(self) -> list[RequestCondition]:
        return [item for item in self.conditions if item.enforcement == "filter"]

    @computed_field
    @property
    def soft_preferences(self) -> list[RequestCondition]:
        return [item for item in self.conditions if item.enforcement == "rank"]

    @computed_field
    @property
    def evidence_requirements(self) -> list[RequestCondition]:
        return [item for item in self.conditions if item.enforcement == "evidence"]

    @computed_field
    @property
    def clarification_requirements(self) -> list[RequestCondition]:
        return [item for item in self.conditions if item.enforcement == "clarify"]

    @computed_field
    @property
    def desired_categories(self) -> list[str]:
        return sorted(
            {
                str(item.value)
                for item in self.conditions
                if item.field == "category" and item.operator == "includes"
            }
        )

    @computed_field
    @property
    def excluded_categories(self) -> list[str]:
        return sorted(
            {
                str(item.value)
                for item in self.conditions
                if item.field == "category" and item.operator == "excludes"
            }
        )

    @model_validator(mode="after")
    def validate_request(self) -> RecommendationRequest:
        keys = [
            (item.field, item.operator, str(item.value), item.enforcement)
            for item in self.conditions
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("request conditions must be unique")
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("missing_fields must be unique")
        return self


def stable_request_id(value: QueryParseInput, *, parser_version: str) -> str:
    """Derive an id without business labels, target data, or environment state."""

    payload = (
        f"{value.user_id}\x1f{value.session_id}\x1f"
        f"{value.cutoff_time.isoformat()}\x1f{value.query_text}\x1f{parser_version}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()
