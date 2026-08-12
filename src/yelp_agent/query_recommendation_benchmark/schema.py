"""Visible and hidden contracts for the Query recommendation benchmark."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.query.benchmark import ExpectedRequestCondition


class VisibleQueryRecommendationCase(StrictModel):
    """The complete benchmark input a retriever or Agent may receive."""

    schema_version: Literal[1] = 1
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["development", "validation"]
    language: Literal["zh-CN", "en-US"]
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    cutoff_time: datetime
    query_text: str = Field(min_length=1, max_length=2000)
    user_latitude: float | None = Field(default=None, ge=-90, le=90)
    user_longitude: float | None = Field(default=None, ge=-180, le=180)
    generator_kind: Literal["deterministic", "openai_compatible"]
    generator_model: str | None = None
    generator_prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator("query_text")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_case(self) -> VisibleQueryRecommendationCase:
        if (self.user_latitude is None) != (self.user_longitude is None):
            raise ValueError("case coordinates must be both present or absent")
        generated = self.generator_kind == "openai_compatible"
        metadata = self.generator_model is not None
        if metadata != (self.generator_prompt_sha256 is not None):
            raise ValueError("generator model and prompt hash must appear together")
        if generated != metadata:
            raise ValueError("only generated cases may carry generator metadata")
        return self


class QueryRecommendationGroundTruth(StrictModel):
    """One behavior-anchored known positive hidden from every retrieval module."""

    schema_version: Literal[1] = 1
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_task_id: str = Field(min_length=1)
    target_business_id: str = Field(min_length=1)
    target_review_id: str = Field(min_length=1)
    target_stars: float = Field(ge=4, le=5)
    target_time: datetime


class QueryRecommendationFrame(StrictModel):
    """Hidden canonical meaning used to render and audit one visible Query."""

    schema_version: Literal[1] = 1
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_family: Literal[
        "category_only",
        "category_distance",
        "category_price",
        "category_single_aspect",
        "category_two_aspects",
        "category_distance_aspect",
        "occasion_context",
    ]
    conditions: list[ExpectedRequestCondition] = Field(min_length=1)
    party_size: int | None = Field(default=None, ge=1, le=20)
    location_source: Literal["none", "history_centroid"]
    target_support_sources: list[
        Literal["static_category", "static_price", "history_location", "cutoff_aspect"]
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_frame(self) -> QueryRecommendationFrame:
        condition_keys = [
            (item.field, item.operator, str(item.value), item.enforcement)
            for item in self.conditions
        ]
        if len(condition_keys) != len(set(condition_keys)):
            raise ValueError("frame conditions must be unique")
        if len(self.target_support_sources) != len(set(self.target_support_sources)):
            raise ValueError("target support sources must be unique")
        has_distance = any(item.field == "distance_km" for item in self.conditions)
        if has_distance != (self.location_source == "history_centroid"):
            raise ValueError("distance conditions require a history centroid")
        return self


class QueryRecommendationBenchmarkBundle(StrictModel):
    visible_cases: tuple[VisibleQueryRecommendationCase, ...]
    ground_truth: tuple[QueryRecommendationGroundTruth, ...]
    frames: tuple[QueryRecommendationFrame, ...]

    @model_validator(mode="after")
    def validate_alignment(self) -> QueryRecommendationBenchmarkBundle:
        groups = [
            [item.case_id for item in values]
            for values in (self.visible_cases, self.ground_truth, self.frames)
        ]
        if not groups[0]:
            raise ValueError("benchmark bundle cannot be empty")
        if any(len(values) != len(set(values)) for values in groups):
            raise ValueError("case IDs must be unique within every artifact")
        if any(set(values) != set(groups[0]) for values in groups[1:]):
            raise ValueError("visible, truth, and frame case IDs must align")
        users_by_split: dict[str, set[str]] = {}
        for item in self.visible_cases:
            users_by_split.setdefault(item.split, set()).add(item.user_id)
        if users_by_split.get("development", set()).intersection(
            users_by_split.get("validation", set())
        ):
            raise ValueError("users cannot cross benchmark splits")
        return self
