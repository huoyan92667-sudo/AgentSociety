"""按评论编号读取完整原评论，避免在每个 Qdrant 片段里重复保存长文本。"""

from __future__ import annotations

from pathlib import Path

import duckdb

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FULL_REVIEW_SOURCE = _PROJECT_ROOT / "data" / "processed" / "reviews.parquet"


class FullReviewStore:
    """只读查询完整评论；Qdrant 负责找片段，这里负责还原原文。"""

    def __init__(self, source: str | Path = DEFAULT_FULL_REVIEW_SOURCE) -> None:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"full review source does not exist: {path}")
        self._connection = duckdb.connect(database=":memory:")
        escaped = str(path.resolve()).replace("'", "''")
        self._connection.execute(
            f"CREATE VIEW reviews AS SELECT * FROM read_parquet('{escaped}')"
        )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def get_many(self, review_ids: list[str]) -> dict[str, str]:
        """批量还原完整原文；返回值严格覆盖请求中的评论编号。"""

        if self._closed:
            raise RuntimeError("full review store is closed")
        unique_ids = list(dict.fromkeys(review_ids))
        if not unique_ids:
            return {}
        marks = ",".join("?" for _ in unique_ids)
        rows = self._connection.execute(
            f"SELECT review_id, text FROM reviews WHERE review_id IN ({marks})",
            unique_ids,
        ).fetchall()
        result = {str(review_id): str(text) for review_id, text in rows}
        missing = set(unique_ids) - set(result)
        if missing:
            raise KeyError(f"full review text is missing for {len(missing)} reviews")
        return result
