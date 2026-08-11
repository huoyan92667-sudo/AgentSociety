"""Content-addressed local score cache without raw query persistence."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .schema import CrossEncoderUsageEvent


@dataclass(frozen=True, slots=True)
class CachedCrossEncoderScore:
    cache_key: str
    model: str
    instruction_sha256: str
    document_version: str
    query_sha256: str
    document_sha256: str
    max_sequence_length: int
    score: float
    input_tokens: int


class SqliteCrossEncoderCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "cross_encoder_cache.sqlite3"
        self.manifest_path = self.root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cross_encoder_scores (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    instruction_sha256 TEXT NOT NULL,
                    document_version TEXT NOT NULL,
                    query_sha256 TEXT NOT NULL,
                    document_sha256 TEXT NOT NULL,
                    max_sequence_length INTEGER NOT NULL,
                    score REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cross_encoder_usage_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usage_scope TEXT NOT NULL,
                    model TEXT NOT NULL,
                    requested_pair_count INTEGER NOT NULL,
                    unique_pair_count INTEGER NOT NULL,
                    cache_hits INTEGER NOT NULL,
                    cache_misses INTEGER NOT NULL,
                    logical_input_tokens INTEGER NOT NULL,
                    scored_input_tokens INTEGER NOT NULL,
                    cache_saved_tokens INTEGER NOT NULL,
                    truncated_pair_count INTEGER NOT NULL,
                    scorer_calls INTEGER NOT NULL,
                    api_calls INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def get_many(self, keys: Iterable[str]) -> dict[str, CachedCrossEncoderScore]:
        values = list(dict.fromkeys(keys))
        result: dict[str, CachedCrossEncoderScore] = {}
        if not values:
            return result
        with self._connect() as connection:
            for offset in range(0, len(values), 500):
                batch = values[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT cache_key, model, instruction_sha256, document_version, "
                    "query_sha256, document_sha256, max_sequence_length, score, "
                    f"input_tokens FROM cross_encoder_scores WHERE cache_key IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    score = float(row[7])
                    if not 0 <= score <= 1:
                        continue
                    result[str(row[0])] = CachedCrossEncoderScore(
                        cache_key=str(row[0]), model=str(row[1]),
                        instruction_sha256=str(row[2]), document_version=str(row[3]),
                        query_sha256=str(row[4]), document_sha256=str(row[5]),
                        max_sequence_length=int(row[6]), score=score,
                        input_tokens=int(row[8]),
                    )
        return result

    def put_many(self, records: Iterable[CachedCrossEncoderScore]) -> None:
        timestamp = datetime.now(UTC).isoformat()
        rows = [
            (
                item.cache_key, item.model, item.instruction_sha256,
                item.document_version, item.query_sha256, item.document_sha256,
                item.max_sequence_length, item.score, item.input_tokens, timestamp,
            )
            for item in records
        ]
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO cross_encoder_scores (
                    cache_key, model, instruction_sha256, document_version,
                    query_sha256, document_sha256, max_sequence_length,
                    score, input_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def record_usage(self, event: CrossEncoderUsageEvent) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cross_encoder_usage_events (
                    usage_scope, model, requested_pair_count, unique_pair_count,
                    cache_hits, cache_misses, logical_input_tokens,
                    scored_input_tokens, cache_saved_tokens, truncated_pair_count,
                    scorer_calls, api_calls, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.usage_scope, event.model, event.requested_pair_count,
                    event.unique_pair_count, event.cache_hits, event.cache_misses,
                    event.logical_input_tokens, event.scored_input_tokens,
                    event.cache_saved_tokens, event.truncated_pair_count,
                    event.scorer_calls, event.api_calls, event.latency_ms,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT count(*) FROM cross_encoder_scores").fetchone()[0])

    def write_manifest(self, *, model: str) -> Path:
        payload = {
            "schema_version": 1,
            "provider": "local",
            "model": model,
            "record_count": self.count(),
            "raw_query_persisted": False,
            "raw_document_persisted": False,
            "external_api_calls": 0,
            "cache_file": self.path.name,
        }
        partial = self.manifest_path.with_name(self.manifest_path.name + ".partial")
        partial.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(partial, self.manifest_path)
        return self.manifest_path
