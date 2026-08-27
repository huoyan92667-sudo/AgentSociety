"""超过数据库内联阈值的大结果内容存储。"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class StoredContent:
    uri: str
    sha256: str
    size_bytes: int


class ArtifactContentStore(Protocol):
    """保存和读取完整大结果正文的接口。"""

    async def put(self, *, result_id: str, content: Any) -> StoredContent: ...

    async def get(self, uri: str) -> Any: ...

    async def delete(self, uri: str) -> None: ...


class LocalJsonContentStore:
    """把大结果压缩成 JSON 文件；生产环境可替换为对象存储。"""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    async def put(self, *, result_id: str, content: Any) -> StoredContent:
        return await asyncio.to_thread(self._put_sync, result_id, content)

    async def get(self, uri: str) -> Any:
        return await asyncio.to_thread(self._get_sync, uri)

    async def delete(self, uri: str) -> None:
        await asyncio.to_thread(self._delete_sync, uri)

    def _put_sync(self, result_id: str, content: Any) -> StoredContent:
        if not result_id or any(
            character not in "0123456789abcdef" for character in result_id
        ):
            raise ValueError("result_id must be a lowercase hexadecimal identifier")
        raw = json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        relative = Path(result_id[:2]) / f"{result_id}.json.gz"
        target = (self._root / relative).resolve()
        self._require_inside_root(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        with gzip.open(temporary, "wb") as stream:
            stream.write(raw)
        temporary.replace(target)
        return StoredContent(
            uri=relative.as_posix(),
            sha256=digest,
            size_bytes=len(raw),
        )

    def _get_sync(self, uri: str) -> Any:
        target = (self._root / Path(uri)).resolve()
        self._require_inside_root(target)
        with gzip.open(target, "rb") as stream:
            raw = stream.read()
        return json.loads(raw.decode("utf-8"))

    def _delete_sync(self, uri: str) -> None:
        target = (self._root / Path(uri)).resolve()
        self._require_inside_root(target)
        target.unlink(missing_ok=True)

    def _require_inside_root(self, target: Path) -> None:
        if target != self._root and self._root not in target.parents:
            raise ValueError("artifact path escapes configured root")
