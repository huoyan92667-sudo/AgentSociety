"""Typed configuration loaded from the project's YAML files."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from yelp_agent.models import RECOMMENDATION_CANDIDATE_COUNT
from yelp_agent.reviews.schema import ASPECT_NAMES, AspectName


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(ConfigModel):
    random_seed: int
    city: str = Field(min_length=1)
    min_business_reviews: int = Field(ge=0)
    min_user_reviews: int = Field(ge=1)
    min_distinct_businesses: int = Field(ge=1)
    min_distinct_ratings: int = Field(ge=1, le=5)
    max_users: int = Field(ge=1)
    candidate_count: int
    hard_negative_same_category: int = Field(ge=0)
    hard_negative_related_category: int = Field(ge=0)
    preference_negative_count: int = Field(ge=0)
    random_negative_count: int = Field(ge=0)
    review_chunk_size: int = Field(ge=1)
    allowed_categories: list[str]
    broad_categories: list[str]
    category_groups: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_candidate_layout(self) -> "DataConfig":
        if self.candidate_count != RECOMMENDATION_CANDIDATE_COUNT:
            raise ValueError(
                "candidate_count must be exactly "
                f"{RECOMMENDATION_CANDIDATE_COUNT} for the MVP"
            )
        negative_count = (
            self.hard_negative_same_category
            + self.hard_negative_related_category
            + self.preference_negative_count
            + self.random_negative_count
        )
        required_negatives = RECOMMENDATION_CANDIDATE_COUNT - 1
        if negative_count != required_negatives:
            raise ValueError(
                "negative candidate buckets must sum to "
                f"{required_negatives}"
            )
        return self

    @model_validator(mode="after")
    def validate_category_taxonomy(self) -> "DataConfig":
        def validate_names(values: list[str], *, label: str) -> None:
            if not values:
                raise ValueError(f"{label} cannot be empty")
            if any(not value or value != value.strip() for value in values):
                raise ValueError(f"{label} contain an invalid name")
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique")

        if self.city != self.city.strip():
            raise ValueError("city cannot contain surrounding whitespace")
        validate_names(self.allowed_categories, label="allowed_categories")
        validate_names(self.broad_categories, label="broad_categories")
        allowed = set(self.allowed_categories)
        if not set(self.broad_categories).issubset(allowed):
            raise ValueError("broad_categories must be allowed categories")

        required_groups = {
            "dining",
            "nightlife",
            "retail",
            "personal_care",
            "entertainment",
        }
        if set(self.category_groups) != required_groups:
            raise ValueError(
                "category_groups must define dining, nightlife, retail, "
                "personal_care, and entertainment"
            )
        for group, categories in self.category_groups.items():
            validate_names(categories, label=f"category_groups.{group}")
            if not set(categories).issubset(allowed):
                raise ValueError(
                    f"category_groups.{group} contains a category outside "
                    "allowed_categories"
                )
        return self


class HybridConfig(ConfigModel):
    category_weight: float
    text_weight: float
    quality_weight: float
    location_weight: float
    location_scale_km: float = Field(gt=0)
    bayesian_prior_count: int = Field(gt=0)
    tuning_step: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_weight_sum(self) -> "HybridConfig":
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("hybrid weights cannot be negative")
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("hybrid weights must sum to 1")
        tuning_units = round(1.0 / self.tuning_step)
        if not math.isclose(
            tuning_units * self.tuning_step,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("tuning_step must divide 1.0 exactly")
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
    temperature: Literal[0.0]
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    response_format_json: bool = False
    thinking: Literal["enabled", "disabled"] | None = None


class TfidfConfig(ConfigModel):
    stop_words: Literal["english"]
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


class RollingTrainingConfig(ConfigModel):
    minimum_history_count: int = Field(ge=1)
    maximum_tasks_per_user: int = Field(ge=1, le=8)
    minimum_target_gap: int = Field(ge=1)
    minimum_sample_weight: float = Field(gt=0, le=1)
    maximum_sample_weight: Literal[1.0]
    selection_strategy: Literal["evenly_spaced_include_latest"]
    weighting_strategy: Literal["linear_by_lifecycle_position"]

    @model_validator(mode="after")
    def validate_weight_range(self) -> "RollingTrainingConfig":
        if self.minimum_sample_weight > self.maximum_sample_weight:
            raise ValueError(
                "minimum_sample_weight cannot exceed maximum_sample_weight"
            )
        return self


class RetrievalConfig(ConfigModel):
    """Settings for target-blind full-catalog candidate retrieval."""

    candidate_limit: int = Field(ge=1)
    per_route_limit: int = Field(ge=1)
    rrf_constant: float = Field(gt=0)
    exclude_history_businesses: bool
    bayesian_prior_count: int = Field(gt=0)
    location_scale_km: float = Field(gt=0)
    metric_cutoffs: list[int]
    provenance_splits: list[Literal["train", "validation", "test"]]

    @model_validator(mode="after")
    def validate_retrieval_limits(self) -> "RetrievalConfig":
        if self.per_route_limit < self.candidate_limit:
            raise ValueError(
                "per_route_limit must be at least candidate_limit"
            )
        if not self.metric_cutoffs:
            raise ValueError("metric_cutoffs cannot be empty")
        if (
            any(cutoff <= 0 for cutoff in self.metric_cutoffs)
            or self.metric_cutoffs != sorted(self.metric_cutoffs)
            or len(set(self.metric_cutoffs)) != len(self.metric_cutoffs)
        ):
            raise ValueError(
                "metric_cutoffs must be unique positive values in ascending order"
            )
        if self.metric_cutoffs[-1] > self.candidate_limit:
            raise ValueError(
                "metric_cutoffs cannot exceed candidate_limit"
            )
        if len(set(self.provenance_splits)) != len(self.provenance_splits):
            raise ValueError("provenance_splits must be unique")
        return self


class ItemKNNConfig(ConfigModel):
    """Settings for point-in-time item-item collaborative evidence."""

    shrinkage_beta: float = Field(gt=0)
    half_life_days: Literal[180, 365, 730] | None = None
    reserved_tail_interactions: int = Field(ge=0)


class ReviewAspectAuditConfig(ConfigModel):
    enabled: bool
    sample_size: int = Field(ge=1)
    batch_size: int = Field(ge=1, le=25)
    seed: int
    temperature: Literal[0.0]
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0, le=2)
    max_tokens: int = Field(ge=1, le=16000)
    response_format_json: Literal[True]
    thinking: Literal["default", "enabled", "disabled"]

    @property
    def provider_thinking(self) -> Literal["enabled", "disabled"] | None:
        """Translate human-readable default into an omitted API parameter."""

        return None if self.thinking == "default" else self.thinking


class ReviewAspectConfig(ConfigModel):
    schema_version: Literal[1]
    extractor_name: Literal["rule_based"]
    extractor_version: str = Field(min_length=1)
    rule_confidence: float = Field(gt=0, le=1)
    negated_confidence: float = Field(gt=0, le=1)
    mixed_confidence: float = Field(gt=0, le=1)
    negation_window_tokens: int = Field(ge=1, le=5)
    chunk_size: int = Field(ge=1)
    audit: ReviewAspectAuditConfig


class ReviewAspectTerms(ConfigModel):
    positive: list[str]
    negative: list[str]

    @model_validator(mode="after")
    def validate_terms(self) -> "ReviewAspectTerms":
        for name, values in (
            ("positive", self.positive),
            ("negative", self.negative),
        ):
            if not values:
                raise ValueError(f"{name} review aspect terms cannot be empty")
            if any(not value or value != value.strip() for value in values):
                raise ValueError(f"{name} review aspect terms contain an invalid value")
            normalized = [value.casefold() for value in values]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{name} review aspect terms must be unique")
        return self


class ReviewAspectVocabulary(ConfigModel):
    vocabulary_version: Literal[3]
    aspects: dict[AspectName, ReviewAspectTerms]

    @model_validator(mode="after")
    def validate_taxonomy(self) -> "ReviewAspectVocabulary":
        if tuple(self.aspects) != ASPECT_NAMES:
            raise ValueError(
                "review aspect vocabulary must define the frozen taxonomy in order"
            )
        return self


class UserProfileConfig(ConfigModel):
    """Settings for deterministic, point-in-time user profiles."""

    schema_version: Literal[1]
    profile_version: Literal["1.0.0"]
    half_life_days: int = Field(gt=0)
    confidence_saturation: float = Field(gt=0)
    reliability_history_saturation: int = Field(gt=0)
    reliability_aspect_saturation: int = Field(gt=0)
    max_category_preferences: int = Field(gt=0)
    max_area_preferences: int = Field(gt=0)
    artifact_batch_size: int = Field(gt=0)


class BusinessProfileConfig(ConfigModel):
    """Settings for point-in-time shared business knowledge."""

    schema_version: Literal[1]
    profile_version: Literal["1.0.0"]
    source_scope: Literal["selected_user_interactions"]
    aspect_half_life_days: int = Field(gt=0)
    minimum_aspect_evidence: int = Field(gt=0)
    minimum_aspect_users: int = Field(gt=0)
    aspect_confidence_saturation: float = Field(gt=0)
    aspect_user_saturation: float = Field(gt=0)
    conflict_ratio_threshold: float = Field(gt=0, le=0.5)
    rating_reliability_saturation: int = Field(gt=0)
    aspect_reliability_saturation: int = Field(gt=0)
    bayesian_prior_count: int = Field(gt=0)
    cache_max_entries: int = Field(gt=0)


class LegacyTestConfig(ConfigModel):
    name: str = Field(min_length=1)
    status: Literal["previously_observed"]
    usage: Literal["historical_comparison_only"]


class CrossValidationConfig(ConfigModel):
    strategy: Literal["deterministic_user_level_sha256"]
    folds: int = Field(ge=2, le=20)
    seed: int


class BootstrapConfig(ConfigModel):
    unit: Literal["user"]
    samples: int = Field(ge=100)
    confidence_level: float = Field(gt=0, lt=1)
    seed: int


class ModelSelectionConfig(ConfigModel):
    primary_metric: Literal["avg_hr"]
    tie_break_metrics: list[Literal["mrr", "hr_at_1"]]

    @model_validator(mode="after")
    def validate_tie_break_metrics(self) -> "ModelSelectionConfig":
        if len(set(self.tie_break_metrics)) != len(self.tie_break_metrics):
            raise ValueError("tie_break_metrics must be unique")
        return self


class EvaluationReportingConfig(ConfigModel):
    fold_mean: bool
    fold_standard_deviation: bool
    paired_model_delta: bool
    bootstrap_confidence_interval: bool


class EvaluationDataUsageConfig(ConfigModel):
    development_split: Literal["validation"]
    strict_blind_holdout: Literal[False]
    legacy_test: LegacyTestConfig
    cross_validation: CrossValidationConfig
    bootstrap: BootstrapConfig
    model_selection: ModelSelectionConfig
    reporting: EvaluationReportingConfig
    segment_dimensions: list[str]

    @model_validator(mode="after")
    def validate_segment_dimensions(self) -> "EvaluationDataUsageConfig":
        if not self.segment_dimensions:
            raise ValueError("segment_dimensions cannot be empty")
        if any(
            not dimension or dimension != dimension.strip()
            for dimension in self.segment_dimensions
        ):
            raise ValueError("segment_dimensions contain an invalid name")
        if len(set(self.segment_dimensions)) != len(self.segment_dimensions):
            raise ValueError("segment_dimensions must be unique")
        return self


class AppConfig(ConfigModel):
    data: DataConfig
    hybrid: HybridConfig
    agent: AgentConfig
    tfidf: TfidfConfig
    evaluation_data_usage: EvaluationDataUsageConfig


class ResolvedConfiguration(ConfigModel):
    """Serializable, self-validating snapshot of effective YAML settings."""

    format_version: Literal[1] = 1
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: AppConfig

    @model_validator(mode="after")
    def validate_fingerprint(self) -> "ResolvedConfiguration":
        expected = configuration_fingerprint(self.configuration)
        if self.fingerprint != expected:
            raise ValueError("configuration fingerprint does not match values")
        return self


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
        tfidf=TfidfConfig.model_validate(_read_yaml(root / "tfidf.yaml")),
        evaluation_data_usage=EvaluationDataUsageConfig.model_validate(
            _read_yaml(root / "evaluation_data_usage.yaml")
        ),
    )


def load_rolling_training_config(
    config_dir: str | Path = "configs",
) -> RollingTrainingConfig:
    """Load settings used only when constructing rolling train examples."""

    root = Path(config_dir)
    return RollingTrainingConfig.model_validate(
        _read_yaml(root / "training.yaml")
    )


def load_retrieval_config(
    config_dir: str | Path = "configs",
) -> RetrievalConfig:
    """Load settings used only by the full retrieval benchmark."""

    root = Path(config_dir)
    return RetrievalConfig.model_validate(
        _read_yaml(root / "retrieval.yaml")
    )


def load_item_knn_config(
    config_dir: str | Path = "configs",
) -> ItemKNNConfig:
    """Load settings used only by point-in-time Item-KNN."""

    root = Path(config_dir)
    return ItemKNNConfig.model_validate(
        _read_yaml(root / "item_knn.yaml")
    )


def load_review_aspect_settings(
    config_dir: str | Path = "configs",
) -> tuple[ReviewAspectConfig, ReviewAspectVocabulary]:
    """Load the versioned Review Aspect protocol and seed vocabulary."""

    root = Path(config_dir)
    return (
        ReviewAspectConfig.model_validate(
            _read_yaml(root / "review_aspects.yaml")
        ),
        ReviewAspectVocabulary.model_validate(
            _read_yaml(root / "review_aspect_vocabulary.yaml")
        ),
    )


def load_user_profile_config(
    config_dir: str | Path = "configs",
) -> UserProfileConfig:
    """Load deterministic long-term user-profile settings."""

    root = Path(config_dir)
    return UserProfileConfig.model_validate(_read_yaml(root / "user_profiles.yaml"))


def load_business_profile_config(
    config_dir: str | Path = "configs",
) -> BusinessProfileConfig:
    """Load shared business-knowledge settings."""

    root = Path(config_dir)
    return BusinessProfileConfig.model_validate(
        _read_yaml(root / "business_profiles.yaml")
    )


def configuration_fingerprint(config: AppConfig) -> str:
    """Hash effective values, independent of YAML formatting and file paths."""

    canonical = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_resolved_configuration(config: AppConfig) -> ResolvedConfiguration:
    """Build the public, secret-free configuration recorded with a run."""

    return ResolvedConfiguration(
        fingerprint=configuration_fingerprint(config),
        configuration=config,
    )


def load_resolved_configuration(
    path: str | Path,
) -> ResolvedConfiguration:
    """Read and validate a previously recorded configuration snapshot."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"resolved configuration does not exist: {resolved}"
        )
    return ResolvedConfiguration.model_validate_json(
        resolved.read_text(encoding="utf-8")
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
