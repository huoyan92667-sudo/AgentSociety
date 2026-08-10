"""Configuration loader for the Step 22 Agent harness."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from yelp_agent.models import StrictModel

from .schema import HarnessBudget


class AgentHarnessConfig(StrictModel):
    schema_version: Literal[1] = 1
    agent_version: str = Field(min_length=1)
    budget: HarnessBudget = Field(default_factory=HarnessBudget)


def load_agent_harness_config(path: Path) -> AgentHarnessConfig:
    """Load one strict YAML document without reading environment secrets."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Agent harness config must be a YAML mapping")
    return AgentHarnessConfig.model_validate(payload)
