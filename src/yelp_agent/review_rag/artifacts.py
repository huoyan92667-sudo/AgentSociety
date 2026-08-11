"""Build the full Review passage artifact without calling any model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.models import StrictModel

from .config import ReviewRAGConfig
from .schema import REVIEW_SEGMENT_SCHEMA, ReviewRAGManifest
from .segmenter import segment_review


class ReviewRAGBuildResult(StrictModel):
    status: Literal["written", "skipped"]
    segments_path: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_review_count: int = Field(ge=1)
    segment_count: int = Field(ge=1)
    business_count: int = Field(ge=1)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_review_rag_artifacts(
    reviews_path: str | Path,
    output_root: str | Path,
    config: ReviewRAGConfig,
    *,
    force: bool = False,
) -> ReviewRAGBuildResult:
    """Segment every nonempty processed Review into an atomic Parquet artifact."""

    source = Path(reviews_path)
    root = Path(output_root)
    segments_path = root / "review_segments.parquet"
    manifest_path = root / "manifest.json"
    if not source.is_file():
        raise FileNotFoundError(f"processed Review Parquet does not exist: {source}")
    source_file = pq.ParquetFile(source)
    if not source_file.schema_arrow.equals(REVIEW_SCHEMA, check_metadata=False):
        raise ValueError("processed Review Parquet schema is incompatible")
    source_hash = sha256_file(source)
    if not force and segments_path.is_file() and manifest_path.is_file():
        try:
            manifest = ReviewRAGManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            manifest = None
        if (
            manifest is not None
            and manifest.source_reviews_sha256 == source_hash
            and manifest.config_sha256 == config.sha256()
            and manifest.segments_sha256 == sha256_file(segments_path)
            and pq.ParquetFile(segments_path).schema_arrow.equals(
                REVIEW_SEGMENT_SCHEMA, check_metadata=False
            )
        ):
            return _result("skipped", segments_path, manifest_path, manifest)

    root.mkdir(parents=True, exist_ok=True)
    partial = segments_path.with_name(segments_path.name + ".partial")
    partial.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    source_count = 0
    nonempty_count = 0
    segment_count = 0
    truncated_count = 0
    business_ids: set[str] = set()
    try:
        writer = pq.ParquetWriter(
            partial,
            REVIEW_SEGMENT_SCHEMA,
            compression="zstd",
            use_dictionary=["business_id", "user_id", "review_id"],
        )
        for batch in source_file.iter_batches(batch_size=config.parquet_batch_size):
            rows: list[dict[str, object]] = []
            for source_row in batch.to_pylist():
                source_count += 1
                built = segment_review(source_row, config)
                if built.segments:
                    nonempty_count += 1
                    business_ids.add(str(source_row["business_id"]))
                truncated_count += int(built.truncated)
                rows.extend(item.model_dump(mode="python") for item in built.segments)
            if rows:
                segment_count += len(rows)
                writer.write_table(
                    pa.Table.from_pylist(rows, schema=REVIEW_SEGMENT_SCHEMA)
                )
        writer.close()
        writer = None
        os.replace(partial, segments_path)
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise
    if source_count != source_file.metadata.num_rows or segment_count < 1:
        segments_path.unlink(missing_ok=True)
        raise RuntimeError("Review RAG artifact build produced inconsistent counts")
    manifest = ReviewRAGManifest(
        source_reviews_sha256=source_hash,
        config_sha256=config.sha256(),
        segments_sha256=sha256_file(segments_path),
        source_review_count=source_count,
        nonempty_review_count=nonempty_count,
        segment_count=segment_count,
        business_count=len(business_ids),
        truncated_review_count=truncated_count,
    )
    partial_manifest = manifest_path.with_name(manifest_path.name + ".partial")
    partial_manifest.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(partial_manifest, manifest_path)
    return _result("written", segments_path, manifest_path, manifest)


def _result(
    status: Literal["written", "skipped"],
    segments_path: Path,
    manifest_path: Path,
    manifest: ReviewRAGManifest,
) -> ReviewRAGBuildResult:
    return ReviewRAGBuildResult(
        status=status,
        segments_path=str(segments_path),
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        source_review_count=manifest.source_review_count,
        segment_count=manifest.segment_count,
        business_count=manifest.business_count,
    )
