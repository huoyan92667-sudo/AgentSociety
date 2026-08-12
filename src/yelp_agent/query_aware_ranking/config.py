"""Frozen configuration and development-selected policy for Step 33."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.semantic_ranking import SemanticRankingPolicy


class QueryAwareRankingConfig(StrictModel):
    schema_version: Literal[1] = 1
    agent_version: Literal["step33-query-aware-ranking-v1"]
    union_candidate_limit: int = Field(default=1000, ge=500, le=2000)
    coarse_diagnostic_limit: int = Field(default=100, ge=10, le=500)
    semantic_candidate_limit: int = Field(default=30, ge=5, le=100)
    internal_result_limit: int = Field(default=10, ge=10, le=100)
    display_limit: Literal[5] = 5
    query_weight_candidates: list[float] = Field(min_length=1)
    query_retrieval_signal_weight: float = Field(ge=0, le=1)
    embedding_signal_weight: float = Field(ge=0, le=1)
    semantic_policy_source: str = Field(min_length=1)
    policy_relative_path: str = Field(min_length=1)
    tuning_split: Literal["development"] = "development"
    validation_used_for_selection: Literal[False] = False

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if self.coarse_diagnostic_limit > self.union_candidate_limit:
            raise ValueError("coarse diagnostic limit exceeds union limit")
        if self.semantic_candidate_limit > self.coarse_diagnostic_limit:
            raise ValueError("semantic limit exceeds coarse diagnostic limit")
        if self.internal_result_limit > self.coarse_diagnostic_limit:
            raise ValueError("internal result limit exceeds coarse diagnostic limit")
        if len(set(self.query_weight_candidates)) != len(
            self.query_weight_candidates
        ):
            raise ValueError("query-weight candidates must be unique")
        if any(not 0 <= value <= 1 for value in self.query_weight_candidates):
            raise ValueError("query-weight candidates must be between zero and one")
        if abs(
            self.query_retrieval_signal_weight + self.embedding_signal_weight - 1.0
        ) > 1e-9:
            raise ValueError("query signal weights must sum to one")
        for value in (self.semantic_policy_source, self.policy_relative_path):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Step 33 paths must stay inside the project")
        return self


class QueryAwareRankingPolicy(StrictModel):
    """Policy selected using development labels and frozen before validation."""

    schema_version: Literal[1] = 1
    policy_version: Literal["1.0.0"] = "1.0.0"
    selected_query_weight: float = Field(ge=0, le=1)
    query_retrieval_signal_weight: float = Field(ge=0, le=1)
    embedding_signal_weight: float = Field(ge=0, le=1)
    union_candidate_limit: int = Field(ge=500, le=2000)
    coarse_diagnostic_limit: int = Field(ge=10, le=500)
    semantic_candidate_limit: int = Field(ge=5, le=100)
    internal_result_limit: int = Field(ge=10, le=100)
    display_limit: Literal[5] = 5
    objective: Literal["AvgHR@1,3,5,10"] = "AvgHR@1,3,5,10"
    tie_breakers: list[str]
    development_case_count: int = Field(ge=1)
    development_metrics: dict[str, float]
    semantic_policy: SemanticRankingPolicy
    validation_used_for_selection: Literal[False] = False

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if abs(
            self.query_retrieval_signal_weight + self.embedding_signal_weight - 1.0
        ) > 1e-9:
            raise ValueError("query signal weights must sum to one")
        if self.semantic_candidate_limit != self.semantic_policy.candidate_limit:
            raise ValueError("semantic policy candidate limit does not match Step 33")
        return self


def load_query_aware_ranking_config(path: str | Path) -> QueryAwareRankingConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Step 33 config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Step 33 config must contain a mapping")
    return QueryAwareRankingConfig.model_validate(payload)


def load_query_aware_ranking_policy(path: str | Path) -> QueryAwareRankingPolicy:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Step 33 policy does not exist: {source}")
    return QueryAwareRankingPolicy.model_validate_json(
        source.read_text(encoding="utf-8")
    )


def write_query_aware_ranking_policy(
    policy: QueryAwareRankingPolicy,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            policy.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target
