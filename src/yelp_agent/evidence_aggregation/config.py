"""Configuration and Development-frozen policy for evidence aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel


class EvidenceAggregationPolicy(StrictModel):
    schema_version: Literal[1] = 1
    policy_version: str = Field(min_length=1)
    selection_split: Literal["development"] = "development"
    validation_used_for_tuning: Literal[False] = False
    recency_half_life_days: int = Field(default=730, ge=30, le=3650)
    minimum_relevance: float = Field(default=0.05, ge=0, le=1)
    minimum_extraction_confidence: float = Field(default=0.8, ge=0, le=1)
    conflict_minimum_count_per_side: int = Field(default=1, ge=1, le=5)
    conflict_minority_mass_share: float = Field(default=0.25, gt=0, lt=0.5)
    consensus_mass_share: float = Field(default=0.67, gt=0.5, le=1)
    diversity_target_users: int = Field(default=3, ge=1, le=10)
    sample_prior: float = Field(default=3.0, gt=0, le=20)
    grounded_minimum_evidence: int = Field(default=1, ge=1, le=5)
    grounded_minimum_unique_users: int = Field(default=1, ge=1, le=5)
    medium_confidence_threshold: float = Field(default=0.08, ge=0, le=1)
    high_confidence_threshold: float = Field(default=0.2, ge=0, le=1)
    maximum_citations_per_aspect: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "EvidenceAggregationPolicy":
        if self.high_confidence_threshold <= self.medium_confidence_threshold:
            raise ValueError("high confidence threshold must exceed medium")
        return self


class EvidenceAggregationConfig(StrictModel):
    schema_version: Literal[1] = 1
    aggregation_version: Literal["1.0.0"] = "1.0.0"
    agent_version: str = Field(min_length=1)
    policy_relative_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_relative_path(self) -> "EvidenceAggregationConfig":
        path = Path(self.policy_relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("evidence policy path must stay inside project")
        return self

    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_evidence_aggregation_config(path: str | Path) -> EvidenceAggregationConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"evidence aggregation config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence aggregation config must contain a mapping")
    return EvidenceAggregationConfig.model_validate(payload)


def load_evidence_aggregation_policy(
    project_root: str | Path,
    config: EvidenceAggregationConfig,
) -> EvidenceAggregationPolicy:
    path = Path(project_root) / config.policy_relative_path
    if not path.is_file():
        raise FileNotFoundError(f"evidence aggregation policy does not exist: {path}")
    return EvidenceAggregationPolicy.model_validate_json(path.read_text(encoding="utf-8"))
