"""Typed configuration loaded from the project's YAML files."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(ConfigModel):
    random_seed: int
    city: str
    min_business_reviews: int
    min_user_reviews: int
    min_distinct_businesses: int
    min_distinct_ratings: int
    max_users: int
    candidate_count: int
    hard_negative_same_category: int
    hard_negative_related_category: int
    preference_negative_count: int
    random_negative_count: int
    review_chunk_size: int
    allowed_categories: list[str]
    broad_categories: list[str]
    category_groups: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_candidate_layout(self) -> "DataConfig":
        if self.candidate_count != 20:
            raise ValueError("candidate_count must be exactly 20 for the MVP")
        negative_count = (
            self.hard_negative_same_category
            + self.hard_negative_related_category
            + self.preference_negative_count
            + self.random_negative_count
        )
        if negative_count != 19:
            raise ValueError("negative candidate buckets must sum to 19")
        return self


class HybridConfig(ConfigModel):
    category_weight: float
    text_weight: float
    quality_weight: float
    location_weight: float
    location_scale_km: float
    bayesian_prior_count: int
    tuning_step: float

    @model_validator(mode="after")
    def validate_weight_sum(self) -> "HybridConfig":
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("hybrid weights cannot be negative")
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("hybrid weights must sum to 1")
        return self

    @property
    def weights(self) -> dict[str, float]:
        return {
            "category": self.category_weight,
            "text": self.text_weight,
            "quality": self.quality_weight,
            "location": self.location_weight,
        }


class AgentConfig(ConfigModel):
    enabled: bool
    top_k_to_rerank: int = Field(ge=2, le=20)
    history_limit: int
    representative_review_count: int
    temperature: float
    timeout_seconds: float = Field(gt=0)
    max_retries: int


class EvaluationConfig(ConfigModel):
    primary_metric: str
    hit_cutoffs: tuple[int, ...]
    ndcg_cutoff: int


class TfidfConfig(ConfigModel):
    stop_words: str
    ngram_min: int = Field(ge=1)
    ngram_max: int = Field(ge=1)
    min_df: int = Field(ge=1)
    sublinear_tf: bool
    max_features: int = Field(gt=0)
    keyword_count: int = Field(ge=1, le=10)

    @model_validator(mode="after")
    def validate_ngram_range(self) -> "TfidfConfig":
        if self.ngram_max < self.ngram_min:
            raise ValueError("ngram_max cannot be smaller than ngram_min")
        return self

    @property
    def ngram_range(self) -> tuple[int, int]:
        return (self.ngram_min, self.ngram_max)


class AppConfig(ConfigModel):
    data: DataConfig
    hybrid: HybridConfig
    agent: AgentConfig
    evaluation: EvaluationConfig
    tfidf: TfidfConfig


class LLMEnvironment(ConfigModel):
    api_key: SecretStr | None = None
    base_url: str | None = None
    model: str | None = None

    @property
    def llm_enabled(self) -> bool:
        return self.api_key is not None and bool(self.model)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration file must contain a mapping: {path}")
    return payload


def load_config(config_dir: str | Path = "configs") -> AppConfig:
    root = Path(config_dir)
    return AppConfig(
        data=DataConfig.model_validate(_read_yaml(root / "data.yaml")),
        hybrid=HybridConfig.model_validate(_read_yaml(root / "hybrid.yaml")),
        agent=AgentConfig.model_validate(_read_yaml(root / "agent.yaml")),
        evaluation=EvaluationConfig.model_validate(
            _read_yaml(root / "evaluation.yaml")
        ),
        tfidf=TfidfConfig.model_validate(_read_yaml(root / "tfidf.yaml")),
    )


def load_llm_environment(
    environment: Mapping[str, str] | None = None,
) -> LLMEnvironment:
    if environment is None:
        from dotenv import load_dotenv

        load_dotenv()
        environment = os.environ

    def optional_value(name: str) -> str | None:
        value = environment.get(name, "").strip()
        return value or None

    api_key = optional_value("OPENAI_API_KEY")
    return LLMEnvironment(
        api_key=SecretStr(api_key) if api_key else None,
        base_url=optional_value("OPENAI_BASE_URL"),
        model=optional_value("OPENAI_MODEL"),
    )
