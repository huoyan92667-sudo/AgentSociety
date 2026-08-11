"""Configuration and local runtime loading for Step 26."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel

from .schema import CrossEncoderPolicy


class CrossEncoderConfig(StrictModel):
    schema_version: Literal[1] = 1
    cross_encoder_version: Literal["1.0.0"] = "1.0.0"
    provider: Literal["local"] = "local"
    agent_version: str = Field(min_length=1)
    batch_size: int = Field(default=8, ge=1, le=32)
    max_sequence_length: int = Field(default=512, ge=64, le=32_768)
    timeout_seconds: float = Field(default=90, gt=0, le=180)
    max_total_tokens_per_turn: int = Field(default=12_000, ge=12_000)
    instruction: str = Field(min_length=1, max_length=2000)
    business_document_version: str = Field(min_length=1)
    cache_relative_path: str = Field(min_length=1)
    policy_relative_path: str = Field(min_length=1)

    @field_validator("cache_relative_path", "policy_relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Cross-Encoder paths must stay inside the project")
        return path.as_posix()


class LocalCrossEncoderEnvironment(StrictModel):
    model_path: Path | None = None
    python_executable: Path = Field(default_factory=lambda: Path(sys.executable))
    device: Literal["cuda", "cpu"] = "cuda"

    @property
    def enabled(self) -> bool:
        return self.model_path is not None

    @model_validator(mode="after")
    def validate_runtime(self) -> LocalCrossEncoderEnvironment:
        if self.model_path is None:
            return self
        if not self.python_executable.is_file():
            raise ValueError("local Cross-Encoder Python executable does not exist")
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (self.model_path / name).is_file()]
        if missing:
            raise ValueError("local Cross-Encoder model is incomplete: " + ", ".join(missing))
        return self


def load_cross_encoder_config(path: str | Path) -> CrossEncoderConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Cross-Encoder config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cross-Encoder config must contain a mapping")
    return CrossEncoderConfig.model_validate(payload)


def load_cross_encoder_policy(
    project_root: str | Path, config: CrossEncoderConfig
) -> CrossEncoderPolicy:
    path = Path(project_root) / config.policy_relative_path
    if not path.is_file():
        raise FileNotFoundError(f"frozen Cross-Encoder policy does not exist: {path}")
    return CrossEncoderPolicy.model_validate_json(path.read_text(encoding="utf-8"))


def load_local_cross_encoder_environment(
    environment: Mapping[str, str] | None = None,
) -> LocalCrossEncoderEnvironment:
    if environment is None:
        from dotenv import load_dotenv

        load_dotenv()
        environment = os.environ
    raw_path = environment.get("LOCAL_CROSS_ENCODER_MODEL_PATH", "").strip()
    raw_python = environment.get("LOCAL_CROSS_ENCODER_PYTHON", "").strip()
    raw_device = environment.get("LOCAL_CROSS_ENCODER_DEVICE", "cuda").strip().casefold()
    return LocalCrossEncoderEnvironment(
        model_path=Path(raw_path) if raw_path else None,
        python_executable=Path(raw_python) if raw_python else Path(sys.executable),
        device=raw_device,
    )
