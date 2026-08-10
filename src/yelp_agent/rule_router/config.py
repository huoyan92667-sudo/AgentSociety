"""Frozen configuration for the zero-LLM Rule Router baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from yelp_agent.models import StrictModel


class RuleRouterConfig(StrictModel):
    schema_version: Literal[1] = 1
    router_version: Literal["1.0.0"] = "1.0.0"
    agent_version: str = Field(min_length=1)
    display_limit: int = Field(default=3, ge=1, le=10)
    rules_frozen: Literal[True] = True
    llm_enabled: Literal[False] = False
    review_rag_enabled: Literal[False] = False
    official_source_enabled: Literal[False] = False


def load_rule_router_config(path: str | Path) -> RuleRouterConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Rule Router config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Rule Router config must contain a mapping")
    return RuleRouterConfig.model_validate(payload)
