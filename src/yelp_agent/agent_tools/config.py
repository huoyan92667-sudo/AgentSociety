"""Bounded runtime configuration for the Agent tool registry."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from yelp_agent.models import StrictModel


class AgentToolRuntimeConfig(StrictModel):
    registry_version: Literal["1.0.0"] = "1.0.0"
    cache_max_entries: int = Field(default=512, ge=0, le=10_000)
    max_retry_attempts: int = Field(default=2, ge=1, le=3)
    max_timeout_ms: float = Field(default=30_000, gt=0, le=90_000)


def load_agent_tool_runtime_config(
    path: str | Path,
) -> AgentToolRuntimeConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Agent tool config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Agent tool config must contain a mapping")
    return AgentToolRuntimeConfig.model_validate(payload)
