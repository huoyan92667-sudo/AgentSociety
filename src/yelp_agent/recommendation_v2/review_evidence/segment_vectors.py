"""按 Qdrant 点编号从本地向量文件读取评论片段向量。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ReviewSegmentVectorStore:
    """内存映射大向量文件，避免 Qdrant 为每次命中重复传回 1024 个数。"""

    def __init__(self, embeddings_path: str | Path) -> None:
        self.path = Path(embeddings_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"review segment embeddings not found: {self.path}")
        vectors = np.load(self.path, mmap_mode="r")
        if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] < 1:
            raise ValueError("review segment embeddings must be a nonempty matrix")
        self._vectors: np.ndarray | None = vectors
        self.dimension = int(vectors.shape[1])
        self.point_count = int(vectors.shape[0])

    def get_many(self, point_ids: list[int]) -> dict[int, np.ndarray]:
        """每个编号只读取一次；返回 float32 供后续点积计算。"""

        vectors = self._vectors
        if vectors is None:
            raise RuntimeError("review segment vector store is closed")
        unique_ids = list(dict.fromkeys(point_ids))
        if any(point_id < 0 or point_id >= self.point_count for point_id in unique_ids):
            raise IndexError("review segment point ID is outside the local vector file")
        if not unique_ids:
            return {}
        loaded = np.asarray(vectors[unique_ids], dtype=np.float32)
        return {
            point_id: loaded[index]
            for index, point_id in enumerate(unique_ids)
        }

    def close(self) -> None:
        """释放内存映射；真正的文件内容不会被修改。"""

        vectors = self._vectors
        self._vectors = None
        mmap = getattr(vectors, "_mmap", None)
        if mmap is not None:
            mmap.close()
