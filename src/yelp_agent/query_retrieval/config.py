"""Configuration for Query-only and later dual-channel candidate retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from yelp_agent.config import ConfigModel


class QueryRetrievalConfig(ConfigModel):
    schema_version: Literal[1]
    agent_version: Literal["step31-dual-channel-query-recall-v1"]
    enabled: bool
    candidate_limit: int = Field(ge=1)
    per_route_limit: int = Field(ge=1)
    rrf_constant: float = Field(gt=0)
    category_weight: float = Field(ge=0)
    embedding_weight: float = Field(ge=0)
    aspect_weight: float = Field(ge=0)
    location_weight: float = Field(ge=0)
    location_scale_km: float = Field(gt=0)
    dual_history_weight: float = Field(ge=0)
    dual_query_weight: float = Field(ge=0)
    aspect_source_scope: Literal["selected_user_interactions"]

    @model_validator(mode="after")
    def validate_limits_and_weights(self) -> QueryRetrievalConfig:
        if self.per_route_limit < self.candidate_limit:
            raise ValueError("per_route_limit must be at least candidate_limit")
        if not any(
            value > 0
            for value in (
                self.category_weight,
                self.embedding_weight,
                self.aspect_weight,
                self.location_weight,
            )
        ):
            raise ValueError("at least one Query route weight must be positive")
        if self.dual_history_weight == self.dual_query_weight == 0:
            raise ValueError("at least one dual-channel weight must be positive")
        return self


def load_query_retrieval_config(path: str | Path) -> QueryRetrievalConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Query retrieval config does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Query retrieval config must contain a mapping")
    return QueryRetrievalConfig.model_validate(payload)
