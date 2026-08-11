"""Configuration for controlled Step 29 semantic and answer calls."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel


class ControlledCapabilityConfig(StrictModel):
    enabled: bool
    prompt_version: str = Field(min_length=1)
    max_output_tokens: int = Field(ge=64, le=4000)


class SemanticEscalationConfig(ControlledCapabilityConfig):
    mode: Literal["never", "gated", "always"] = "gated"
    minimum_signal_confidence: float = Field(ge=0, le=1)
    minimum_task_type_confidence: float = Field(ge=0, le=1)
    call_for_task_types: list[str] = Field(default_factory=list)
    complex_markers: list[str] = Field(default_factory=list)


class AnswerComposerConfig(ControlledCapabilityConfig):
    compose_grounded_answers: bool = True
    compose_uncertain_answers: bool = True
    maximum_evidence_items: int = Field(default=12, ge=1, le=20)


class ControlledLLMConfig(StrictModel):
    schema_version: Literal[1] = 1
    agent_version: str = Field(min_length=1)
    temperature: Literal[0.0] = 0.0
    timeout_seconds: float = Field(gt=0, le=180)
    max_retries: int = Field(default=0, ge=0, le=1)
    thinking: Literal["enabled", "disabled"] = "disabled"
    response_format_json: Literal[True] = True
    cache_relative_path: str = Field(min_length=1)
    maximum_provider_calls_per_turn: int = Field(default=2, ge=1, le=2)
    semantic: SemanticEscalationConfig
    answer: AnswerComposerConfig
    tuning_split: Literal["development"] = "development"
    validation_used_for_selection: Literal[False] = False

    @model_validator(mode="after")
    def validate_versions(self) -> ControlledLLMConfig:
        if self.semantic.prompt_version == self.answer.prompt_version:
            raise ValueError("semantic and answer prompts require distinct versions")
        return self


def load_controlled_llm_config(path: str | Path) -> ControlledLLMConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"controlled LLM config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("controlled LLM config must contain a mapping")
    return ControlledLLMConfig.model_validate(payload)
