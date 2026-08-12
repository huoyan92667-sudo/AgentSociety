"""Deterministic selection of real future high-rating behavior anchors."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel


class QueryRecommendationBenchmarkConfig(StrictModel):
    schema_version: Literal[1] = 1
    benchmark_version: Literal["1.0.0"] = "1.0.0"
    seed: int = 42
    development_count: int = Field(default=400, ge=1)
    validation_count: int = Field(default=100, ge=1)
    minimum_target_stars: float = Field(default=4.0, ge=4, le=5)
    minimum_target_pre_cutoff_reviews: int = Field(default=1, ge=1)
    aspect_source_scope: Literal["selected_user_interactions"] = (
        "selected_user_interactions"
    )


class BehaviorAnchorCandidate(StrictModel):
    """A real next interaction before it is assigned to a benchmark split."""

    source_task_id: str = Field(min_length=1)
    source_split: Literal["train", "validation"]
    user_id: str = Field(min_length=1)
    cutoff_time: datetime
    history_business_ids: tuple[str, ...]
    target_business_id: str = Field(min_length=1)
    target_review_id: str = Field(min_length=1)
    target_stars: float = Field(ge=1, le=5)
    target_time: datetime
    target_pre_cutoff_review_count: int = Field(ge=0)

    @field_validator("history_business_ids")
    @classmethod
    def validate_history(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or any(not value for value in values):
            raise ValueError("anchor history must contain business IDs")
        return values

    @model_validator(mode="after")
    def validate_time(self) -> BehaviorAnchorCandidate:
        if self.target_time != self.cutoff_time:
            raise ValueError("target time must equal the recommendation cutoff")
        return self


class BehaviorAnchor(BehaviorAnchorCandidate):
    benchmark_split: Literal["development", "validation"]


def _selection_key(
    candidate: BehaviorAnchorCandidate,
    *,
    seed: int,
    benchmark_split: str,
) -> tuple[str, str]:
    payload = (
        f"{seed}\0{benchmark_split}\0{candidate.user_id}\0"
        f"{candidate.source_task_id}"
    ).encode()
    return hashlib.sha256(payload).hexdigest(), candidate.source_task_id


def _eligible(
    candidate: BehaviorAnchorCandidate,
    config: QueryRecommendationBenchmarkConfig,
) -> bool:
    return (
        candidate.target_stars >= config.minimum_target_stars
        and candidate.target_pre_cutoff_review_count
        >= config.minimum_target_pre_cutoff_reviews
        and candidate.target_business_id not in set(candidate.history_business_ids)
    )


def _select_unique_users(
    candidates: list[BehaviorAnchorCandidate],
    *,
    count: int,
    split: Literal["development", "validation"],
    seed: int,
    excluded_users: set[str],
) -> list[BehaviorAnchor]:
    selected: list[BehaviorAnchor] = []
    seen = set(excluded_users)
    for candidate in sorted(
        candidates,
        key=lambda item: _selection_key(item, seed=seed, benchmark_split=split),
    ):
        if candidate.user_id in seen:
            continue
        seen.add(candidate.user_id)
        selected.append(
            BehaviorAnchor(
                **candidate.model_dump(),
                benchmark_split=split,
            )
        )
        if len(selected) == count:
            return selected
    raise ValueError(f"not enough eligible unique users for {split}: {len(selected)}/{count}")


def select_behavior_anchors(
    candidates: tuple[BehaviorAnchorCandidate, ...],
    config: QueryRecommendationBenchmarkConfig,
) -> tuple[BehaviorAnchor, ...]:
    """Select validation first, then a disjoint development user population."""

    eligible = [item for item in candidates if _eligible(item, config)]
    validation = _select_unique_users(
        [item for item in eligible if item.source_split == "validation"],
        count=config.validation_count,
        split="validation",
        seed=config.seed,
        excluded_users=set(),
    )
    validation_users = {item.user_id for item in validation}
    development = _select_unique_users(
        [item for item in eligible if item.source_split == "train"],
        count=config.development_count,
        split="development",
        seed=config.seed,
        excluded_users=validation_users,
    )
    return tuple(development + validation)
