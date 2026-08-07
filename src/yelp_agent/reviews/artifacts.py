"""Build and freeze deterministic Review Aspect Parquet artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError

from yelp_agent.config import ReviewAspectConfig, ReviewAspectVocabulary
from yelp_agent.experiments import write_json_artifact
from yelp_agent.models import StrictModel
from yelp_agent.reviews.extractors.rule_based import RuleBasedAspectExtractor
from yelp_agent.reviews.schema import REVIEW_ASPECT_SCHEMA, ReviewDocument


class ReviewAspectArtifactError(RuntimeError):
    """Raised when source reviews or frozen outputs are inconsistent."""


class ReviewAspectManifest(StrictModel):
    format_version: Literal[1] = 1
    artifact_name: Literal["Review Aspect Records V1"] = "Review Aspect Records V1"
    source_scope: Literal[
        "selected_user_interactions",
        "full_business_reviews",
        "custom",
    ]
    source_reviews_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: Literal[1]
    extractor_name: str
    extractor_version: str
    source_reviews: int = Field(ge=1)
    reviews_with_aspects: int = Field(ge=0)
    aspect_records: int = Field(ge=0)
    aspect_counts: dict[str, int]
    sentiment_counts: dict[str, int]


class ReviewAspectBuildResult(StrictModel):
    status: Literal["written", "skipped"]
    source_scope: Literal[
        "selected_user_interactions",
        "full_business_reviews",
        "custom",
    ]
    records_path: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_reviews: int = Field(ge=1)
    reviews_with_aspects: int = Field(ge=0)
    aspect_records: int = Field(ge=0)
    aspect_counts: dict[str, int]
    sentiment_counts: dict[str, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_sha256(value: StrictModel) -> str:
    payload = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _result(
    status: Literal["written", "skipped"],
    records_path: Path,
    manifest_path: Path,
    manifest: ReviewAspectManifest,
) -> ReviewAspectBuildResult:
    return ReviewAspectBuildResult(
        status=status,
        source_scope=manifest.source_scope,
        records_path=str(records_path),
        manifest_path=str(manifest_path),
        manifest_sha256=_sha256_file(manifest_path),
        source_reviews=manifest.source_reviews,
        reviews_with_aspects=manifest.reviews_with_aspects,
        aspect_records=manifest.aspect_records,
        aspect_counts=manifest.aspect_counts,
        sentiment_counts=manifest.sentiment_counts,
    )


def _reusable_manifest(
    manifest_path: Path,
    records_path: Path,
    *,
    source_sha256: str,
    configuration_sha256: str,
    vocabulary_sha256: str,
    source_scope: str,
) -> ReviewAspectManifest | None:
    if not manifest_path.is_file() or not records_path.is_file():
        return None
    try:
        manifest = ReviewAspectManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    if (
        manifest.source_reviews_sha256 != source_sha256
        or manifest.source_scope != source_scope
        or manifest.configuration_sha256 != configuration_sha256
        or manifest.vocabulary_sha256 != vocabulary_sha256
        or manifest.records_sha256 != _sha256_file(records_path)
    ):
        return None
    try:
        parquet_file = pq.ParquetFile(records_path)
    except (OSError, pa.ArrowException):
        return None
    if not parquet_file.schema_arrow.equals(
        REVIEW_ASPECT_SCHEMA,
        check_metadata=False,
    ):
        return None
    return manifest


def build_review_aspect_artifacts(
    reviews_path: str | Path,
    output_root: str | Path,
    config: ReviewAspectConfig,
    vocabulary: ReviewAspectVocabulary,
    *,
    source_scope: Literal[
        "selected_user_interactions",
        "full_business_reviews",
        "custom",
    ],
    force: bool = False,
) -> ReviewAspectBuildResult:
    """Extract all row-local evidence and atomically freeze one local artifact."""

    source = Path(reviews_path)
    root = Path(output_root)
    records_path = root / "aspect_records.parquet"
    manifest_path = root / "manifest.json"
    if not source.is_file():
        raise FileNotFoundError(f"Review Parquet does not exist: {source}")
    try:
        source_file = pq.ParquetFile(source)
    except (OSError, pa.ArrowException) as exc:
        raise ReviewAspectArtifactError(
            f"Could not read source reviews: {source}"
        ) from exc
    source_reviews = source_file.metadata.num_rows
    if source_reviews < 1:
        raise ReviewAspectArtifactError("Source review Parquet contains no rows")

    source_sha256 = _sha256_file(source)
    configuration_sha256 = _model_sha256(config)
    vocabulary_sha256 = _model_sha256(vocabulary)
    if not force:
        reusable = _reusable_manifest(
            manifest_path,
            records_path,
            source_sha256=source_sha256,
            configuration_sha256=configuration_sha256,
            vocabulary_sha256=vocabulary_sha256,
            source_scope=source_scope,
        )
        if reusable is not None:
            return _result("skipped", records_path, manifest_path, reusable)

    root.mkdir(parents=True, exist_ok=True)
    partial = records_path.with_name(records_path.name + ".partial")
    partial.unlink(missing_ok=True)
    extractor = RuleBasedAspectExtractor(config, vocabulary)
    writer: pq.ParquetWriter | None = None
    aspect_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    reviews_with_aspects = 0
    aspect_records = 0

    try:
        writer = pq.ParquetWriter(
            partial,
            REVIEW_ASPECT_SCHEMA,
            compression="zstd",
            use_dictionary=[
                "business_id",
                "user_id",
                "aspect",
                "sentiment",
                "extractor_name",
                "extractor_version",
            ],
        )
        for batch in source_file.iter_batches(
            batch_size=config.chunk_size,
            columns=["review_id", "user_id", "business_id", "text", "date"],
        ):
            output_rows: list[dict[str, object]] = []
            for row in batch.to_pylist():
                text = str(row["text"] or "")
                if not text.strip():
                    continue
                review = ReviewDocument(
                    review_id=str(row["review_id"]),
                    business_id=str(row["business_id"]),
                    user_id=str(row["user_id"]),
                    review_time=row["date"],
                    text=text,
                )
                extracted = extractor.extract(review)
                if extracted:
                    reviews_with_aspects += 1
                for record in extracted:
                    output_rows.append(record.model_dump(mode="python"))
                    aspect_counts[record.aspect] += 1
                    sentiment_counts[record.sentiment] += 1
                aspect_records += len(extracted)
            if output_rows:
                writer.write_table(
                    pa.Table.from_pylist(output_rows, schema=REVIEW_ASPECT_SCHEMA)
                )
        writer.close()
        writer = None
        os.replace(partial, records_path)
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise

    manifest = ReviewAspectManifest(
        source_scope=source_scope,
        source_reviews_sha256=source_sha256,
        configuration_sha256=configuration_sha256,
        vocabulary_sha256=vocabulary_sha256,
        records_sha256=_sha256_file(records_path),
        schema_version=config.schema_version,
        extractor_name=config.extractor_name,
        extractor_version=config.extractor_version,
        source_reviews=source_reviews,
        reviews_with_aspects=reviews_with_aspects,
        aspect_records=aspect_records,
        aspect_counts=dict(sorted(aspect_counts.items())),
        sentiment_counts=dict(sorted(sentiment_counts.items())),
    )
    write_json_artifact(manifest_path, manifest)
    return _result("written", records_path, manifest_path, manifest)
