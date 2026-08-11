"""Configuration and frozen fusion policy for Review RAG V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.semantic_embedding.config import SemanticEmbeddingConfig


class ReviewRAGPolicy(StrictModel):
    schema_version: Literal[1] = 1
    policy_version: str = Field(min_length=1)
    selection_split: Literal["development"] = "development"
    validation_used_for_tuning: Literal[False] = False
    rrf_k: int = Field(default=60, ge=1, le=200)
    aspect_weight: float = Field(default=1.0, ge=0, le=5)
    bm25_weight: float = Field(default=1.0, ge=0, le=5)
    embedding_weight: float = Field(default=1.0, ge=0, le=5)
    top_k: Literal[5] = 5

    @model_validator(mode="after")
    def validate_active_route(self) -> "ReviewRAGPolicy":
        if self.aspect_weight + self.bm25_weight + self.embedding_weight <= 0:
            raise ValueError("at least one Review RAG route must be enabled")
        return self


class ReviewRAGConfig(StrictModel):
    schema_version: Literal[1] = 1
    rag_version: Literal["1.0.0"] = "1.0.0"
    agent_version: str = Field(min_length=1)
    segment_max_chars: int = Field(default=900, ge=100, le=4000)
    segment_min_chars: int = Field(default=20, ge=1, le=500)
    max_segments_per_review: int = Field(default=8, ge=1, le=50)
    parquet_batch_size: int = Field(default=5000, ge=100, le=100_000)
    minimum_aspect_confidence: float = Field(default=0.85, ge=0, le=1)
    bm25_candidate_limit: int = Field(default=60, ge=5, le=500)
    aspect_candidate_limit: int = Field(default=60, ge=5, le=500)
    embedding_candidate_limit: int = Field(default=30, ge=5, le=100)
    max_store_segments_per_business: int = Field(default=20_000, ge=100)
    bm25_k1: float = Field(default=1.5, gt=0, le=5)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    embedding_dimension: int = 1024
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    embedding_max_sequence_length: int = Field(default=512, ge=32, le=4096)
    query_instruction: str = Field(min_length=1, max_length=1000)
    review_document_version: str = Field(min_length=1)
    embedding_cache_relative_path: str = Field(min_length=1)
    policy_relative_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_segment_limits(self) -> "ReviewRAGConfig":
        if self.segment_min_chars >= self.segment_max_chars:
            raise ValueError("segment_min_chars must be below segment_max_chars")
        for value in (self.embedding_cache_relative_path, self.policy_relative_path):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Review RAG artifact paths must stay inside project")
        return self

    def semantic_config(self) -> SemanticEmbeddingConfig:
        """Reuse the local encoder/cache seam with a Review-specific instruction."""

        return SemanticEmbeddingConfig(
            provider="local",
            agent_version=self.agent_version,
            dimension=self.embedding_dimension,
            batch_size=self.embedding_batch_size,
            max_sequence_length=self.embedding_max_sequence_length,
            timeout_seconds=90,
            max_retries=0,
            candidate_limit=self.embedding_candidate_limit,
            max_total_tokens_per_turn=12_000,
            fusion_alpha=0.0,
            query_instruction=self.query_instruction,
            business_document_version=self.review_document_version,
            cache_relative_path=self.embedding_cache_relative_path,
            price_cny_per_1000_input_tokens=0.0,
        )

    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_review_rag_config(path: str | Path) -> ReviewRAGConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Review RAG config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Review RAG config must contain a mapping")
    return ReviewRAGConfig.model_validate(payload)


def load_review_rag_policy(project_root: str | Path, config: ReviewRAGConfig) -> ReviewRAGPolicy:
    path = Path(project_root) / config.policy_relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Review RAG policy does not exist: {path}")
    return ReviewRAGPolicy.model_validate_json(path.read_text(encoding="utf-8"))
