"""Atomic, byte-stable writers for local experiment artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from yelp_agent.models import StrictModel


class ArtifactWriteResult(StrictModel):
    """Describe whether an artifact changed without exposing file internals."""

    status: Literal["written", "skipped"]
    path: str
    byte_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _write_bytes(path: Path, payload: bytes) -> ArtifactWriteResult:
    digest = hashlib.sha256(payload).hexdigest()
    if path.is_file():
        try:
            if path.read_bytes() == payload:
                return ArtifactWriteResult(
                    status="skipped",
                    path=str(path),
                    byte_count=len(payload),
                    sha256=digest,
                )
        except OSError:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_bytes(payload)
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return ArtifactWriteResult(
        status="written",
        path=str(path),
        byte_count=len(payload),
        sha256=digest,
    )


def write_text_artifact(
    path: str | Path,
    payload: str,
) -> ArtifactWriteResult:
    """Atomically publish UTF-8 text and avoid rewriting identical bytes."""

    return _write_bytes(Path(path), payload.encode("utf-8"))


def write_json_artifact(
    path: str | Path,
    value: BaseModel,
) -> ArtifactWriteResult:
    """Atomically publish one Pydantic model as indented JSON plus LF."""

    return write_text_artifact(
        path,
        value.model_dump_json(indent=2) + "\n",
    )


def write_jsonl_artifact(
    path: str | Path,
    values: Iterable[BaseModel],
) -> ArtifactWriteResult:
    """Atomically publish Pydantic models as compact JSONL in input order."""

    payload = "".join(value.model_dump_json() + "\n" for value in values)
    return write_text_artifact(path, payload)
