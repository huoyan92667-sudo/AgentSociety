"""Configuration and secret-safe environment loading for Step 25."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, SecretStr, field_validator, model_validator

from yelp_agent.models import StrictModel


_SUPPORTED_DIMENSIONS = {256, 512, 768, 1024, 1536, 2048, 2560}


class SemanticEmbeddingConfig(StrictModel):
    schema_version: Literal[1] = 1
    embedding_version: Literal["2.0.0"] = "2.0.0"
    provider: Literal["dashscope", "local"] = "local"
    agent_version: str = Field(min_length=1)
    dimension: int = 1024
    batch_size: int = Field(default=16, ge=1, le=64)
    max_sequence_length: int = Field(default=512, ge=32, le=32_768)
    timeout_seconds: float = Field(default=90, gt=0, le=120)
    max_retries: int = Field(default=2, ge=0, le=3)
    candidate_limit: int = Field(default=30, ge=1, le=500)
    max_total_tokens_per_turn: int = Field(default=12_000, ge=12_000)
    fusion_alpha: float = Field(default=0.3, ge=0, le=1)
    query_instruction: str = Field(min_length=1, max_length=1000)
    business_document_version: str = Field(min_length=1)
    cache_relative_path: str = Field(min_length=1)
    price_cny_per_1000_input_tokens: float = Field(default=0.0, ge=0)
    output_type: Literal["dense"] = "dense"

    @model_validator(mode="after")
    def validate_provider_limits(self) -> SemanticEmbeddingConfig:
        if self.provider == "dashscope" and self.batch_size > 20:
            raise ValueError("DashScope embedding batch size cannot exceed 20")
        if self.provider == "local" and self.price_cny_per_1000_input_tokens != 0:
            raise ValueError("local embedding must have zero API token price")
        return self

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value not in _SUPPORTED_DIMENSIONS:
            raise ValueError("unsupported embedding dimension")
        return value

    @field_validator("cache_relative_path")
    @classmethod
    def validate_relative_cache(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("embedding cache path must stay inside the project")
        return path.as_posix()


class DashScopeEmbeddingEnvironment(StrictModel):
    api_key: SecretStr | None = None
    base_url: str | None = None
    model: str | None = None

    @property
    def enabled(self) -> bool:
        return all((self.api_key is not None, self.base_url, self.model))

    @model_validator(mode="after")
    def validate_complete_or_empty(self) -> DashScopeEmbeddingEnvironment:
        configured = (self.api_key is not None, bool(self.base_url), bool(self.model))
        if any(configured) and not all(configured):
            raise ValueError(
                "DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, and DASHSCOPE_MODEL "
                "must be configured together"
            )
        if self.base_url is not None and not self.base_url.startswith(
            ("https://", "http://")
        ):
            raise ValueError("DASHSCOPE_BASE_URL must be an HTTP(S) URL")
        return self


class LocalEmbeddingEnvironment(StrictModel):
    model_path: Path | None = None
    python_executable: Path = Field(default_factory=lambda: Path(sys.executable))
    device: Literal["cuda", "cpu"] = "cuda"

    @property
    def enabled(self) -> bool:
        return self.model_path is not None

    @model_validator(mode="after")
    def validate_model_path(self) -> LocalEmbeddingEnvironment:
        if self.model_path is None:
            return self
        if not self.python_executable.is_file():
            raise ValueError("local embedding Python executable does not exist")
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (self.model_path / name).is_file()]
        if missing:
            raise ValueError(
                "local embedding model is incomplete: " + ", ".join(missing)
            )
        return self


def load_semantic_embedding_config(
    path: str | Path,
) -> SemanticEmbeddingConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Embedding config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Embedding config must contain a mapping")
    return SemanticEmbeddingConfig.model_validate(payload)


def load_dashscope_embedding_environment(
    environment: Mapping[str, str] | None = None,
) -> DashScopeEmbeddingEnvironment:
    if environment is None:
        from dotenv import load_dotenv

        load_dotenv()
        environment = os.environ

    def optional(name: str) -> str | None:
        value = environment.get(name, "").strip()
        return value or None

    api_key = optional("DASHSCOPE_API_KEY")
    return DashScopeEmbeddingEnvironment(
        api_key=SecretStr(api_key) if api_key else None,
        base_url=optional("DASHSCOPE_BASE_URL"),
        model=optional("DASHSCOPE_MODEL"),
    )


def load_local_embedding_environment(
    environment: Mapping[str, str] | None = None,
) -> LocalEmbeddingEnvironment:
    if environment is None:
        from dotenv import load_dotenv

        load_dotenv()
        environment = os.environ
    raw_path = environment.get("LOCAL_EMBEDDING_MODEL_PATH", "").strip()
    raw_python = environment.get("LOCAL_EMBEDDING_PYTHON", "").strip()
    raw_device = environment.get("LOCAL_EMBEDDING_DEVICE", "cuda").strip().casefold()
    return LocalEmbeddingEnvironment(
        model_path=Path(raw_path) if raw_path else None,
        python_executable=Path(raw_python) if raw_python else Path(sys.executable),
        device=raw_device,
    )
