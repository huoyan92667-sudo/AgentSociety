"""Small durable SQLite cache for validated controlled-LLM JSON outputs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Self


class SqliteControlledLLMCache:
    """Cache only validated response payloads; provider secrets are never stored."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS controlled_llm_cache (
                cache_key TEXT PRIMARY KEY,
                capability TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def get(self, cache_key: str) -> str | None:
        row = self._connection.execute(
            "SELECT payload_json FROM controlled_llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        return None if row is None else str(row[0])

    def put(
        self,
        *,
        cache_key: str,
        capability: str,
        model: str,
        prompt_version: str,
        payload_json: str,
        created_at: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO controlled_llm_cache
                (cache_key, capability, model, prompt_version, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                capability,
                model,
                prompt_version,
                payload_json,
                created_at,
            ),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
