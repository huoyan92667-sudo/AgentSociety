"""Read-only, cutoff-required access to frozen Review Aspect records."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.reviews.schema import (
    ASPECT_NAMES,
    REVIEW_ASPECT_SCHEMA,
    AspectName,
    ReviewAspectRecord,
)

_COLUMNS = (
    "review_id",
    "business_id",
    "user_id",
    "review_time",
    "aspect",
    "sentiment",
    "confidence",
    "evidence_span",
    "evidence_start",
    "evidence_end",
    "source_text_sha256",
    "extractor_name",
    "extractor_version",
)


class ReviewAspectStoreError(RuntimeError):
    """Raised when a frozen aspect artifact cannot be queried safely."""


class ReviewAspectStore:
    """Expose only entity-scoped reads that require a strict cutoff."""

    def __init__(self, records_path: str | Path) -> None:
        self._path = Path(records_path)
        if not self._path.is_file():
            raise FileNotFoundError(
                f"Review Aspect Parquet does not exist: {self._path}"
            )
        try:
            schema = pq.ParquetFile(self._path).schema_arrow
        except (OSError, pa.ArrowException) as exc:
            raise ReviewAspectStoreError(
                f"Could not read Review Aspect Parquet: {self._path}"
            ) from exc
        if not schema.equals(REVIEW_ASPECT_SCHEMA, check_metadata=False):
            raise ReviewAspectStoreError(
                "Review Aspect Parquet has an unexpected schema"
            )
        self._connection = duckdb.connect()
        self._connection.from_parquet(str(self._path)).create_view(
            "review_aspect_records"
        )
        self._closed = False

    def __enter__(self) -> "ReviewAspectStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def _query_before(
        self,
        field: str,
        entity_id: str,
        cutoff_time: datetime,
        aspects: tuple[AspectName, ...] | None,
    ) -> tuple[ReviewAspectRecord, ...]:
        if self._closed:
            raise ReviewAspectStoreError("Review Aspect Store is closed")
        if not entity_id or entity_id != entity_id.strip():
            raise ValueError(
                "entity ID must be nonempty without surrounding whitespace"
            )
        if not isinstance(cutoff_time, datetime):
            raise TypeError("cutoff_time must be a datetime")
        parameters: list[object] = [entity_id, cutoff_time]
        aspect_clause = ""
        if aspects is not None:
            if (
                not aspects
                or len(set(aspects)) != len(aspects)
                or any(aspect not in ASPECT_NAMES for aspect in aspects)
            ):
                raise ValueError("aspects must be unique frozen Aspect names")
            placeholders = ", ".join("?" for _ in aspects)
            aspect_clause = f" AND aspect IN ({placeholders})"
            parameters.extend(aspects)
        query = f"""
            SELECT {", ".join(_COLUMNS)}
            FROM review_aspect_records
            WHERE {field} = ?
              AND review_time < ?
              {aspect_clause}
            ORDER BY review_time, review_id, evidence_start, aspect
        """
        try:
            rows = self._connection.execute(query, parameters).fetchall()
        except duckdb.Error as exc:
            raise ReviewAspectStoreError("Could not query Review Aspect Store") from exc
        return tuple(
            ReviewAspectRecord.model_validate(dict(zip(_COLUMNS, row, strict=True)))
            for row in rows
        )

    def for_business_before(
        self,
        business_id: str,
        cutoff_time: datetime,
        *,
        aspects: tuple[AspectName, ...] | None = None,
    ) -> tuple[ReviewAspectRecord, ...]:
        """Return only one business's evidence strictly before cutoff."""

        return self._query_before(
            "business_id",
            business_id,
            cutoff_time,
            aspects,
        )

    def for_user_before(
        self,
        user_id: str,
        cutoff_time: datetime,
        *,
        aspects: tuple[AspectName, ...] | None = None,
    ) -> tuple[ReviewAspectRecord, ...]:
        """Return only one user's evidence strictly before cutoff."""

        return self._query_before("user_id", user_id, cutoff_time, aspects)
