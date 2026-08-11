"""Configuration and frozen policy loading for Step 30."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel

from .schema import SemanticRankingMode


class SemanticRankingConfig(StrictModel):
    schema_version: Literal[1] = 1
    agent_version: str = Field(min_length=1)
    enabled: bool = True
    mode: SemanticRankingMode = "protected"
    candidate_limit: int = Field(default=20, ge=2, le=100)
    policy_relative_path: str = Field(min_length=1)
    diagnostics_relative_path: str = Field(min_length=1)
    tuning_split: Literal["development"] = "development"
    validation_used_for_selection: Literal[False] = False


class SemanticRankingPolicy(StrictModel):
    """Development-frozen semantic scoring and rank-protection policy."""

    schema_version: Literal[1] = 1
    candidate_limit: int = Field(ge=2, le=100)
    embedding_weight: float = Field(ge=0, le=1)
    cross_encoder_weight: float = Field(ge=0, le=1)
    structured_weight: float = Field(ge=0, le=1)
    minimum_intent_confidence: float = Field(ge=0, le=1)
    maximum_fusion_alpha: float = Field(ge=0, le=1)
    evidence_floor: float = Field(ge=0, le=1)
    maximum_upward_move: int = Field(ge=0, le=100)
    maximum_downward_move: int = Field(ge=0, le=100)
    protected_top_k: int = Field(ge=0, le=20)
    displacement_margin: float = Field(ge=0, le=1)
    objective: Literal["HR@5"] = "HR@5"
    tie_breakers: list[str] = Field(
        default_factory=lambda: ["MRR", "HR@1", "harm_rate", "mean_movement"]
    )
    development_scenario_count: int = Field(ge=0)
    development_metrics: dict[str, float]
    validation_used_for_selection: Literal[False] = False

    @model_validator(mode="after")
    def validate_weights_and_protection(self) -> Self:
        total = (
            self.embedding_weight
            + self.cross_encoder_weight
            + self.structured_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("semantic score weights must sum to one")
        if self.protected_top_k > self.candidate_limit:
            raise ValueError("protected_top_k cannot exceed candidate_limit")
        return self


def load_semantic_ranking_config(path: str | Path) -> SemanticRankingConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Semantic-ranking config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Semantic-ranking config must contain a mapping")
    return SemanticRankingConfig.model_validate(payload)


def load_semantic_ranking_policy(
    project_root: str | Path,
    config: SemanticRankingConfig,
) -> SemanticRankingPolicy:
    path = Path(project_root) / config.policy_relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Semantic-ranking policy does not exist: {path}")
    return SemanticRankingPolicy.model_validate_json(path.read_text(encoding="utf-8"))


def write_semantic_ranking_policy(
    policy: SemanticRankingPolicy,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target
