"""Streaming filter from Yelp business JSONL to static Parquet."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.config import DataConfig
from yelp_agent.models import StrictModel


BUSINESS_SCHEMA = pa.schema(
    [
        ("business_id", pa.string()),
        ("name", pa.string()),
        ("address", pa.string()),
        ("city", pa.string()),
        ("state", pa.string()),
        ("postal_code", pa.string()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("categories", pa.list_(pa.string())),
        ("attributes_json", pa.string()),
    ]
)

REJECTION_REASONS = (
    "wrong_city",
    "closed",
    "insufficient_reviews",
    "missing_categories",
    "outside_allowed_categories",
)


class BusinessPreprocessError(RuntimeError):
    """Raised when the business source cannot produce a safe Parquet file."""


class BusinessPreprocessResult(StrictModel):
    status: Literal["written", "skipped"]
    input_path: str
    output_path: str
    source_rows: int | None = Field(default=None, ge=0)
    selected_rows: int = Field(ge=0)
    rejection_counts: dict[str, int]


def _parse_categories(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [
        category.strip()
        for category in str(value).split(",")
        if category.strip()
    ]


def _write_chunk(
    writer: pq.ParquetWriter,
    rows: list[dict[str, object]],
) -> None:
    writer.write_table(pa.Table.from_pylist(rows, schema=BUSINESS_SCHEMA))


def preprocess_businesses(
    input_path: str | Path,
    output_path: str | Path,
    config: DataConfig,
    *,
    force: bool = False,
) -> BusinessPreprocessResult:
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"business JSONL does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    if destination.exists() and not force:
        try:
            parquet_file = pq.ParquetFile(destination)
        except Exception as exc:
            raise BusinessPreprocessError(
                f"existing business Parquet is unreadable: {destination}"
            ) from exc
        if not parquet_file.schema_arrow.equals(BUSINESS_SCHEMA):
            raise BusinessPreprocessError(
                f"existing business Parquet has an unexpected schema: {destination}"
            )
        return BusinessPreprocessResult(
            status="skipped",
            input_path=str(source.resolve()),
            output_path=str(destination.resolve()),
            source_rows=None,
            selected_rows=parquet_file.metadata.num_rows,
            rejection_counts={reason: 0 for reason in REJECTION_REASONS},
        )

    source_rows = 0
    selected_rows = 0
    rejection_counts: Counter[str] = Counter()
    selected_ids: set[str] = set()
    allowed_categories = set(config.allowed_categories)
    chunk: list[dict[str, object]] = []

    try:
        with (
            source.open("r", encoding="utf-8") as input_file,
            pq.ParquetWriter(
                temporary,
                BUSINESS_SCHEMA,
                compression="zstd",
            ) as writer,
        ):
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                source_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BusinessPreprocessError(
                        f"invalid JSON at line {line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise BusinessPreprocessError(
                        f"business row must be an object at line {line_number}"
                    )
                if str(row.get("city", "")) != config.city:
                    rejection_counts["wrong_city"] += 1
                    continue
                if int(row.get("is_open", 0) or 0) != 1:
                    rejection_counts["closed"] += 1
                    continue
                if int(row.get("review_count", 0) or 0) < config.min_business_reviews:
                    rejection_counts["insufficient_reviews"] += 1
                    continue
                categories = _parse_categories(row.get("categories"))
                if not categories:
                    rejection_counts["missing_categories"] += 1
                    continue
                if not allowed_categories.intersection(categories):
                    rejection_counts["outside_allowed_categories"] += 1
                    continue

                business_id = str(row.get("business_id", "")).strip()
                if not business_id:
                    raise BusinessPreprocessError(
                        f"eligible business has no business_id at line {line_number}"
                    )
                if business_id in selected_ids:
                    raise BusinessPreprocessError(
                        f"duplicate eligible business_id at line {line_number}: "
                        f"{business_id}"
                    )
                selected_ids.add(business_id)
                chunk.append(
                    {
                        "business_id": business_id,
                        "name": str(row.get("name", "") or ""),
                        "address": str(row.get("address", "") or ""),
                        "city": str(row.get("city", "") or ""),
                        "state": str(row.get("state", "") or ""),
                        "postal_code": str(row.get("postal_code", "") or ""),
                        "latitude": row.get("latitude"),
                        "longitude": row.get("longitude"),
                        "categories": categories,
                        "attributes_json": json.dumps(
                            row.get("attributes") or {},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
                selected_rows += 1
                if len(chunk) >= config.review_chunk_size:
                    _write_chunk(writer, chunk)
                    chunk.clear()
            if chunk:
                _write_chunk(writer, chunk)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return BusinessPreprocessResult(
        status="written",
        input_path=str(source.resolve()),
        output_path=str(destination.resolve()),
        source_rows=source_rows,
        selected_rows=selected_rows,
        rejection_counts={
            reason: rejection_counts[reason] for reason in REJECTION_REASONS
        },
    )


def write_business_preprocess_report(
    result: BusinessPreprocessResult,
    output_path: str | Path,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
