"""Configuration for the frozen Step 34.5 benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel

from .schema import BenchmarkLanguage, BenchmarkSplit, TurnFamily


class LanguagePlan(StrictModel):
    recommendation_sessions: int = Field(ge=0)
    recommendation_families: dict[TurnFamily, int]
    support_families: dict[TurnFamily, int]

    @model_validator(mode="after")
    def validate_turn_counts(self) -> Self:
        recommendation_turns = sum(self.recommendation_families.values())
        if recommendation_turns != self.recommendation_sessions * 3:
            raise ValueError(
                "recommendation family counts must equal three turns per session"
            )
        unsupported = set(self.support_families) - {
            "clarification_answer",
            "conflict_resolution",
            "no_state_change",
        }
        if unsupported:
            raise ValueError(f"unsupported support families: {sorted(unsupported)}")
        return self


class SplitPlan(StrictModel):
    languages: dict[BenchmarkLanguage, LanguagePlan]

    @model_validator(mode="after")
    def validate_languages(self) -> Self:
        if set(self.languages) != {"zh-CN", "en-US"}:
            raise ValueError("every split requires Chinese and English plans")
        return self


class SessionMemoryBenchmarkV2Config(StrictModel):
    formal: bool = True
    benchmark_version: str = Field(min_length=1)
    seed: int = 42
    pipeline_version: str = Field(min_length=1)
    generator_prompt_version: str = Field(min_length=1)
    reviewer_prompt_version: str = Field(min_length=1)
    batch_size: int = Field(default=5, ge=1, le=20)
    maximum_generation_rounds: int = Field(default=3, ge=1, le=5)
    generation_timeout_seconds: int = Field(default=90, ge=1, le=300)
    generation_max_retries: int = Field(default=2, ge=0, le=5)
    generation_max_output_tokens: int = Field(default=12_000, ge=100)
    duplicate_similarity_threshold: float = Field(default=0.94, gt=0, le=1)
    split_plans: dict[BenchmarkSplit, SplitPlan]

    @model_validator(mode="after")
    def validate_frozen_shape(self) -> Self:
        if set(self.split_plans) != {"development", "validation"}:
            raise ValueError("benchmark requires development and validation plans")
        total_turns = 0
        split_turns: dict[str, int] = {}
        language_turns = {"zh-CN": 0, "en-US": 0}
        for split, plan in self.split_plans.items():
            count = 0
            for language, language_plan in plan.languages.items():
                turns = sum(language_plan.recommendation_families.values()) + sum(
                    language_plan.support_families.values()
                )
                count += turns
                language_turns[language] += turns
            split_turns[split] = count
            total_turns += count
        if not self.formal:
            return self
        if total_turns != 500:
            raise ValueError("formal Benchmark V2 must contain exactly 500 turns")
        if split_turns != {"development": 400, "validation": 100}:
            raise ValueError("formal Benchmark V2 split must be 400/100")
        if language_turns != {"zh-CN": 300, "en-US": 200}:
            raise ValueError("formal Benchmark V2 language split must be 300/200")
        return self


def load_session_memory_benchmark_v2_config(
    path: str | Path,
) -> SessionMemoryBenchmarkV2Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SessionMemoryBenchmarkV2Config.model_validate(payload)
