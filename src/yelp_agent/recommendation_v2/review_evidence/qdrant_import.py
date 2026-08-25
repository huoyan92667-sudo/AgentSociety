"""命令行入口：把新版评论片段和向量导入本地 Qdrant。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .builder import DEFAULT_EVIDENCE_ROOT
from .qdrant_store import QdrantReviewSegmentStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导入评论证据向量到 Qdrant")
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument(
        "--index-root",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "index",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recreate", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    # 大批量向量写入使用二进制传输，避免 HTTP JSON 编码数百万浮点数。
    store = QdrantReviewSegmentStore.from_url(args.url, prefer_grpc=True)
    try:
        result = store.import_index(
            args.index_root,
            recreate=args.recreate,
            batch_size=args.batch_size,
        )
    finally:
        store.close()
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
