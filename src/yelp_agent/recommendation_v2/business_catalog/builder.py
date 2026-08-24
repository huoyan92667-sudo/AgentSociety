"""从现有商家全集生成严格的餐厅子集和原始类别清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from yelp_agent.data.businesses import BUSINESS_SCHEMA

from .schema import (
    DINING_CATEGORY_SCHEMA,
    DiningCatalogBuildResult,
    DiningCatalogManifest,
)

DINING_SCOPE_CATEGORY = "Restaurants"


class DiningCatalogError(RuntimeError):
    """输入商家数据或已有餐厅产物不完整时抛出。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category_count(table: pa.Table) -> int:
    return len(
        {
            str(category)
            for categories in table.column("categories").to_pylist()
            for category in categories
        }
    )


def _read_source(source_path: Path) -> pa.Table:
    if not source_path.is_file():
        raise FileNotFoundError(f"business source does not exist: {source_path}")
    try:
        parquet = pq.ParquetFile(source_path)
    except (OSError, pa.ArrowException) as exc:
        raise DiningCatalogError("business source is not readable parquet") from exc
    if not parquet.schema_arrow.equals(BUSINESS_SCHEMA, check_metadata=False):
        raise DiningCatalogError("business source does not use BUSINESS_SCHEMA")
    table = parquet.read()
    business_ids = [str(value) for value in table.column("business_id").to_pylist()]
    if not business_ids or any(not value for value in business_ids):
        raise DiningCatalogError("business source must contain valid business IDs")
    if len(business_ids) != len(set(business_ids)):
        raise DiningCatalogError("business source contains duplicate business IDs")
    return table


def _select_dining_businesses(source: pa.Table) -> pa.Table:
    """只按真实 Restaurants 标记选择，暂不猜测咖啡店或食品店。"""

    category_rows = source.column("categories").to_pylist()
    mask = pa.array(
        [DINING_SCOPE_CATEGORY in categories for categories in category_rows],
        type=pa.bool_(),
    )
    selected = source.filter(mask).sort_by([("business_id", "ascending")])
    if selected.num_rows == 0:
        raise DiningCatalogError("source contains no Restaurants businesses")
    return selected


def _category_table(dining: pa.Table) -> pa.Table:
    counts: Counter[str] = Counter()
    for categories in dining.column("categories").to_pylist():
        counts.update(str(category) for category in categories)
    business_count = dining.num_rows
    rows = [
        {
            "category": category,
            "business_count": count,
            "business_share": count / business_count,
            "is_scope_marker": category == DINING_SCOPE_CATEGORY,
        }
        for category, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].casefold(), item[0]),
        )
    ]
    return pa.Table.from_pylist(rows, schema=DINING_CATEGORY_SCHEMA)


def _load_existing(
    output_root: Path,
    *,
    source_sha256: str,
) -> DiningCatalogManifest | None:
    manifest_path = output_root / "manifest.json"
    business_path = output_root / "businesses.parquet"
    category_path = output_root / "categories.parquet"
    if not all(path.is_file() for path in (manifest_path, business_path, category_path)):
        return None
    try:
        manifest = DiningCatalogManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    if manifest.source_sha256 != source_sha256:
        return None
    actual_hashes = {
        "businesses": _sha256_file(business_path),
        "categories": _sha256_file(category_path),
    }
    if actual_hashes != manifest.output_sha256:
        return None
    return manifest


def build_dining_catalog(
    source_path: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> DiningCatalogBuildResult:
    """生成餐厅专用商家文件和下一步固定类别表需要的类别清单。"""

    source = Path(source_path)
    destination = Path(output_root)
    source_hash = _sha256_file(source) if source.is_file() else ""
    if not force and source_hash:
        existing = _load_existing(destination, source_sha256=source_hash)
        if existing is not None:
            return DiningCatalogBuildResult(
                status="skipped",
                output_root=str(destination.resolve()),
                manifest=existing,
            )

    source_table = _read_source(source)
    dining_table = _select_dining_businesses(source_table)
    categories_table = _category_table(dining_table)
    destination.mkdir(parents=True, exist_ok=True)
    business_path = destination / "businesses.parquet"
    category_path = destination / "categories.parquet"
    manifest_path = destination / "manifest.json"
    temporary_business = destination / "businesses.parquet.partial"
    temporary_category = destination / "categories.parquet.partial"
    temporary_manifest = destination / "manifest.json.partial"
    temporary_paths = (
        temporary_business,
        temporary_category,
        temporary_manifest,
    )

    for path in temporary_paths:
        path.unlink(missing_ok=True)
    try:
        pq.write_table(dining_table, temporary_business, compression="zstd")
        pq.write_table(categories_table, temporary_category, compression="zstd")
        output_hashes = {
            "businesses": _sha256_file(temporary_business),
            "categories": _sha256_file(temporary_category),
        }
        manifest = DiningCatalogManifest(
            source_path=str(source.resolve()),
            source_sha256=source_hash,
            source_business_count=source_table.num_rows,
            dining_business_count=dining_table.num_rows,
            source_category_count=_category_count(source_table),
            dining_category_count=categories_table.num_rows,
            output_sha256=output_hashes,
        )
        temporary_manifest.write_text(
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_business, business_path)
        os.replace(temporary_category, category_path)
        os.replace(temporary_manifest, manifest_path)
    except Exception:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise

    return DiningCatalogBuildResult(
        status="written",
        output_root=str(destination.resolve()),
        manifest=manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成新版餐厅专用商家数据集")
    parser.add_argument("source_path", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_dining_catalog(
        args.source_path,
        args.output_root,
        force=args.force,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
