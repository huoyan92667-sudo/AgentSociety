"""Persistent local cache for vectors returned by the remote provider."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class CachedEmbedding:
    cache_key: str
    provider: str
    model: str
    dimension: int
    input_type: str
    instruction_sha256: str
    document_version: str
    text_sha256: str
    vector: np.ndarray
    input_tokens: int


class SqliteEmbeddingCache:
    """Store vectors atomically without persisting raw user query text."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "embedding_cache.sqlite3"
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
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    input_type TEXT NOT NULL,
                    instruction_sha256 TEXT NOT NULL,
                    document_version TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def get_many(self, keys: Iterable[str]) -> dict[str, CachedEmbedding]:
        values = list(dict.fromkeys(keys))
        if not values:
            return {}
        result: dict[str, CachedEmbedding] = {}
        with self._connect() as connection:
            for offset in range(0, len(values), 500):
                batch = values[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT cache_key, provider, model, dimension, input_type, "
                    "instruction_sha256, document_version, text_sha256, vector, "
                    f"input_tokens FROM embeddings WHERE cache_key IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    dimension = int(row[3])
                    vector = np.frombuffer(row[8], dtype="<f4").copy()
                    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
                        continue
                    result[str(row[0])] = CachedEmbedding(
                        cache_key=str(row[0]),
                        provider=str(row[1]),
                        model=str(row[2]),
                        dimension=dimension,
                        input_type=str(row[4]),
                        instruction_sha256=str(row[5]),
                        document_version=str(row[6]),
                        text_sha256=str(row[7]),
                        vector=vector,
                        input_tokens=int(row[9]),
                    )
        return result

    def put_many(self, records: Iterable[CachedEmbedding]) -> None:
        rows = []
        timestamp = datetime.now(UTC).isoformat()
        for record in records:
            vector = np.asarray(record.vector, dtype="<f4")
            if vector.shape != (record.dimension,) or not np.all(np.isfinite(vector)):
                raise ValueError("cached embedding vector is invalid")
            rows.append(
                (
                    record.cache_key,
                    record.provider,
                    record.model,
                    record.dimension,
                    record.input_type,
                    record.instruction_sha256,
                    record.document_version,
                    record.text_sha256,
                    vector.tobytes(),
                    record.input_tokens,
                    timestamp,
                )
            )
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO embeddings (
                    cache_key, provider, model, dimension, input_type,
                    instruction_sha256, document_version, text_sha256,
                    vector, input_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT count(*) FROM embeddings").fetchone()[0])

    def write_manifest(self, *, model: str, dimension: int) -> Path:
        payload = {
            "schema_version": 1,
            "provider": "dashscope",
            "model": model,
            "dimension": dimension,
            "record_count": self.count(),
            "raw_text_persisted": False,
            "cache_file": self.path.name,
        }
        partial = self.manifest_path.with_name(self.manifest_path.name + ".partial")
        partial.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(partial, self.manifest_path)
        return self.manifest_path

    def export_parquet(self, path: str | Path) -> Path:
        output = Path(path)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT cache_key, provider, model, dimension, input_type, "
                "instruction_sha256, document_version, text_sha256, vector, "
                "input_tokens, created_at FROM embeddings ORDER BY cache_key"
            ).fetchall()
        payload = []
        for row in rows:
            vector = np.frombuffer(row[8], dtype="<f4").astype(float).tolist()
            payload.append(
                {
                    "cache_key": row[0],
                    "provider": row[1],
                    "model": row[2],
                    "dimension": row[3],
                    "input_type": row[4],
                    "instruction_sha256": row[5],
                    "document_version": row[6],
                    "text_sha256": row[7],
                    "embedding": vector,
                    "input_tokens": row[9],
                    "created_at": row[10],
                }
            )
        table = pa.Table.from_pylist(payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        pq.write_table(table, partial, compression="zstd")
        os.replace(partial, output)
        return output
