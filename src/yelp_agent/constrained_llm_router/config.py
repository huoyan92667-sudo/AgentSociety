"""Frozen Step 35 Router configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel


class ConstrainedLLMRouterConfig(StrictModel):
    schema_version: Literal[1] = 1
    agent_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    enabled: bool = True
    temperature: Literal[0.0] = 0.0
    timeout_seconds: float = Field(gt=0, le=180)
    max_retries: int = Field(default=0, ge=0, le=1)
    max_output_tokens: int = Field(default=300, ge=64, le=1000)
    thinking: Literal["enabled", "disabled"] = "disabled"
    response_format_json: Literal[True] = True
    cache_relative_path: str = Field(min_length=1)
    minimum_confidence: float = Field(default=0.60, ge=0, le=1)
    repair_invalid_output_once: bool = True
    bypass_single_choice: bool = True
    maximum_choices: int = Field(default=10, ge=2, le=12)
    tuning_split: Literal["development"] = "development"
    validation_used_for_selection: Literal[False] = False

    @model_validator(mode="after")
    def validate_provider_attempts(self) -> ConstrainedLLMRouterConfig:
        if self.max_retries and self.repair_invalid_output_once:
            raise ValueError(
                "transport retry and invalid-output repair cannot both be enabled"
            )
        return self


def load_constrained_llm_router_config(
    path: str | Path,
) -> ConstrainedLLMRouterConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"constrained Router config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("constrained Router config must contain a mapping")
    return ConstrainedLLMRouterConfig.model_validate(payload)
