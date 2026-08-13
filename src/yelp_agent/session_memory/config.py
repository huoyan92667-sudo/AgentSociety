"""Configuration for the Step 34 LLM-first session memory module."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from yelp_agent.models import StrictModel


class SessionMemoryConfig(StrictModel):
    schema_version: Literal[1] = 1
    memory_version: str = Field(min_length=1)
    enabled: bool = True
    prompt_version: str = Field(min_length=1)
    temperature: Literal[0.0] = 0.0
    timeout_seconds: float = Field(default=90, gt=0, le=180)
    max_retries: int = Field(default=2, ge=0, le=2)
    max_output_tokens: int = Field(default=1200, ge=128, le=4000)
    thinking: Literal["enabled", "disabled"] = "disabled"
    response_format_json: Literal[True] = True
    cache_relative_path: str = Field(min_length=1)
    call_mode: Literal["always", "followups_only"] = "always"
    rule_fallback_enabled: Literal[True] = True
    minimum_task_type_confidence: float = Field(default=0.75, ge=0, le=1)
    minimum_patch_confidence: float = Field(default=0.65, ge=0, le=1)
    max_recent_turns: int = Field(default=8, ge=1, le=20)
    max_presented_sets: int = Field(default=5, ge=1, le=10)


def load_session_memory_config(path: str | Path) -> SessionMemoryConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"session-memory config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("session-memory config must contain a mapping")
    return SessionMemoryConfig.model_validate(payload)
