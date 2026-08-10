"""Static business document loading and local-token dry-run planning."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.data.temporal_view import BusinessRecord
from yelp_agent.models import StrictModel

from .config import SemanticEmbeddingConfig
from .documents import build_business_document
from .local_encoder import LocalQwenEmbeddingEncoder


class StaticEmbeddingDryRun(StrictModel):
    business_count: int = Field(ge=1)
    total_tokens: int = Field(ge=1)
    minimum_tokens: int = Field(ge=1)
    maximum_tokens: int = Field(ge=1)
    average_tokens: float = Field(gt=0)
    truncated_text_count: int = Field(ge=0)
    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    document_version: str = Field(min_length=1)


def load_static_business_records(path: str | Path) -> tuple[BusinessRecord, ...]:
    rows = pq.read_table(path).to_pylist()
    records = tuple(
        BusinessRecord(
            business_id=str(row["business_id"]),
            name=str(row["name"]),
            address=str(row["address"]),
            city=str(row["city"]),
            state=str(row["state"]),
            postal_code=str(row["postal_code"]),
            latitude=None if row["latitude"] is None else float(row["latitude"]),
            longitude=None if row["longitude"] is None else float(row["longitude"]),
            categories=tuple(str(value) for value in row["categories"]),
            attributes_json=str(row["attributes_json"]),
        )
        for row in rows
    )
    if not records or len({record.business_id for record in records}) != len(records):
        raise ValueError("static business records must be nonempty and unique")
    return records


def estimate_static_embedding_tokens(
    records: Sequence[BusinessRecord],
    *,
    encoder: LocalQwenEmbeddingEncoder,
    config: SemanticEmbeddingConfig,
    count_batch_size: int = 256,
) -> StaticEmbeddingDryRun:
    documents = [
        build_business_document(
            record,
            document_version=config.business_document_version,
        )
        for record in records
    ]
    token_counts: list[int] = []
    truncated = 0
    for offset in range(0, len(documents), count_batch_size):
        result = encoder.count_tokens(
            [document.text for document in documents[offset : offset + count_batch_size]],
            input_type="document",
        )
        token_counts.extend(result.per_text_tokens)
        truncated += result.truncated_text_count
    return StaticEmbeddingDryRun(
        business_count=len(documents),
        total_tokens=sum(token_counts),
        minimum_tokens=min(token_counts),
        maximum_tokens=max(token_counts),
        average_tokens=sum(token_counts) / len(token_counts),
        truncated_text_count=truncated,
        model=encoder.model,
        dimension=encoder.dimension,
        document_version=config.business_document_version,
    )
