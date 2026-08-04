"""Stream Yelp reviews into a target-business Parquet dataset."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from yelp_agent.config import DataConfig


REVIEW_SCHEMA = pa.schema(
    [
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("stars", pa.float64(), nullable=False),
        pa.field("useful", pa.int64(), nullable=False),
        pa.field("funny", pa.int64(), nullable=False),
        pa.field("cool", pa.int64(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("date", pa.timestamp("us"), nullable=False),
    ]
)


class ReviewPreprocessError(RuntimeError):
    """Raised when the raw review data cannot be safely preprocessed."""


class ReviewPreprocessResult(BaseModel):
    """Serializable summary of one review preprocessing run."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["written", "skipped"]
    input_path: str
    businesses_path: str
    output_path: str
    source_rows: int | None
    selected_rows: int
    outside_target_businesses: int | None


def _load_business_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Business Parquet file does not exist: {path}")

    try:
        table = pq.read_table(path, columns=["business_id"])
    except (OSError, pa.ArrowException) as exc:
        raise ReviewPreprocessError(
            f"Could not read business IDs from {path}: {exc}"
        ) from exc

    business_ids = table.column("business_id").to_pylist()
    if not business_ids:
        raise ReviewPreprocessError(f"Business Parquet contains no business IDs: {path}")
    if any(not isinstance(value, str) or not value.strip() for value in business_ids):
        raise ReviewPreprocessError(f"Business Parquet contains an invalid business ID: {path}")
    if len(business_ids) != len(set(business_ids)):
        raise ReviewPreprocessError(f"Business Parquet contains duplicate business IDs: {path}")
    return set(business_ids)


def _validate_existing_output(path: Path) -> int:
    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as exc:
        raise ReviewPreprocessError(
            f"Existing review Parquet is unreadable; use force=True to rebuild it: {path}"
        ) from exc

    if not parquet_file.schema_arrow.equals(REVIEW_SCHEMA, check_metadata=False):
        raise ReviewPreprocessError(
            f"Existing review Parquet has an unexpected schema; "
            f"use force=True to rebuild it: {path}"
        )
    return parquet_file.metadata.num_rows


def _required_string(record: dict[str, object], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ReviewPreprocessError(
            f"Invalid {field!r} at review JSONL line {line_number}"
        )
    return value


def _parse_selected_review(
    record: dict[str, object], line_number: int
) -> dict[str, object]:
    review_id = _required_string(record, "review_id", line_number)
    user_id = _required_string(record, "user_id", line_number)
    business_id = _required_string(record, "business_id", line_number)

    try:
        stars = float(record["stars"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewPreprocessError(
            f"Invalid 'stars' at review JSONL line {line_number}"
        ) from exc
    if not 1.0 <= stars <= 5.0:
        raise ReviewPreprocessError(
            f"Review stars must be between 1 and 5 at JSONL line {line_number}"
        )

    counters: dict[str, int] = {}
    for field in ("useful", "funny", "cool"):
        value = record.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReviewPreprocessError(
                f"Invalid {field!r} at review JSONL line {line_number}"
            )
        if value < 0:
            raise ReviewPreprocessError(
                f"Review {field!r} cannot be negative at JSONL line {line_number}"
            )
        counters[field] = value

    text = record.get("text", "")
    if not isinstance(text, str):
        raise ReviewPreprocessError(
            f"Invalid 'text' at review JSONL line {line_number}"
        )

    raw_date = record.get("date")
    if not isinstance(raw_date, str):
        raise ReviewPreprocessError(
            f"Invalid 'date' at review JSONL line {line_number}"
        )
    try:
        date = datetime.fromisoformat(raw_date)
    except ValueError as exc:
        raise ReviewPreprocessError(
            f"Invalid 'date' at review JSONL line {line_number}: {raw_date!r}"
        ) from exc

    return {
        "review_id": review_id,
        "user_id": user_id,
        "business_id": business_id,
        "stars": stars,
        **counters,
        "text": text,
        "date": date,
    }


def _write_chunk(writer: pq.ParquetWriter, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows, schema=REVIEW_SCHEMA)
    writer.write_table(table)


def preprocess_reviews(
    input_path: str | Path,
    businesses_path: str | Path,
    output_path: str | Path,
    config: DataConfig,
    *,
    force: bool = False,
) -> ReviewPreprocessResult:
    """Filter raw Yelp reviews to the frozen target-business universe.

    The input is scanned line-by-line. Selected reviews are written in bounded
    chunks to an atomic ``.partial`` file before replacing the final output.
    """

    source = Path(input_path)
    business_source = Path(businesses_path)
    destination = Path(output_path)

    if not source.is_file():
        raise FileNotFoundError(f"Review JSONL file does not exist: {source}")

    if destination.exists() and not force:
        selected_rows = _validate_existing_output(destination)
        return ReviewPreprocessResult(
            status="skipped",
            input_path=str(source),
            businesses_path=str(business_source),
            output_path=str(destination),
            source_rows=None,
            selected_rows=selected_rows,
            outside_target_businesses=None,
        )

    target_business_ids = _load_business_ids(business_source)
    chunk_size = config.review_chunk_size
    if chunk_size <= 0:
        raise ValueError("review_chunk_size must be greater than zero")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.partial")
    if partial_path.exists():
        partial_path.unlink()

    source_rows = 0
    selected_rows = 0
    outside_target_businesses = 0
    selected_review_ids: set[str] = set()
    rows: list[dict[str, object]] = []
    writer: pq.ParquetWriter | None = None

    try:
        writer = pq.ParquetWriter(
            partial_path,
            REVIEW_SCHEMA,
            compression="zstd",
            use_dictionary=["user_id", "business_id"],
        )
        with source.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                source_rows += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReviewPreprocessError(
                        f"Invalid JSON at review JSONL line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ReviewPreprocessError(
                        f"Expected a JSON object at review JSONL line {line_number}"
                    )

                business_id = record.get("business_id")
                if business_id not in target_business_ids:
                    outside_target_businesses += 1
                    continue

                parsed = _parse_selected_review(record, line_number)
                review_id = str(parsed["review_id"])
                if review_id in selected_review_ids:
                    raise ReviewPreprocessError(
                        f"Duplicate selected review_id {review_id!r} "
                        f"at JSONL line {line_number}"
                    )
                selected_review_ids.add(review_id)
                rows.append(parsed)
                selected_rows += 1

                if len(rows) >= chunk_size:
                    _write_chunk(writer, rows)
                    rows.clear()

        if rows:
            _write_chunk(writer, rows)
        writer.close()
        writer = None
        os.replace(partial_path, destination)
    except Exception:
        if writer is not None:
            writer.close()
        partial_path.unlink(missing_ok=True)
        raise

    return ReviewPreprocessResult(
        status="written",
        input_path=str(source),
        businesses_path=str(business_source),
        output_path=str(destination),
        source_rows=source_rows,
        selected_rows=selected_rows,
        outside_target_businesses=outside_target_businesses,
    )


def write_review_preprocess_report(
    result: ReviewPreprocessResult, path: str | Path
) -> Path:
    """Persist a preprocessing result as human-readable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
