"""Frozen construction settings for Query Recommendation Benchmark V1."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from yelp_agent.models import StrictModel

from .frames import QueryFramePlanConfig
from .selection import QueryRecommendationBenchmarkConfig


class QueryGenerationConfig(StrictModel):
    batch_size: int = Field(default=20, ge=1, le=25)
    timeout_seconds: float = Field(default=90, gt=0, le=120)
    max_retries: int = Field(default=1, ge=0, le=2)
    max_output_tokens: int = Field(default=12000, ge=1000, le=16000)
    maximum_generation_attempts: int = Field(default=2, ge=1, le=3)
    temperature: Literal[0.0] = 0.0
    thinking: Literal["disabled"] = "disabled"


class QueryRecommendationBuildConfig(StrictModel):
    schema_version: Literal[1] = 1
    selection: QueryRecommendationBenchmarkConfig = Field(
        default_factory=QueryRecommendationBenchmarkConfig
    )
    frames: QueryFramePlanConfig = Field(default_factory=QueryFramePlanConfig)
    generation: QueryGenerationConfig = Field(default_factory=QueryGenerationConfig)


def load_query_recommendation_build_config(
    path: str | Path,
) -> QueryRecommendationBuildConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Query benchmark config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Query benchmark config must contain a mapping")
    return QueryRecommendationBuildConfig.model_validate(payload)
