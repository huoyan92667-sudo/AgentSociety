"""用餐厅子集的319种真实类别生成固定、可校验的类别表。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.recommendation_v2.business_catalog import DINING_CATEGORY_SCHEMA

from .schema import (
    FIXED_CATEGORY_SCHEMA,
    FixedCategoryBuildResult,
    FixedCategoryDocument,
    FixedCategoryManifest,
    FixedCategoryRow,
)
from .taxonomy import GROUPS, classify_category

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = _PACKAGE_ROOT / "data" / "dining_catalog" / "v1" / "categories.parquet"
DEFAULT_OUTPUT_ROOT = _PACKAGE_ROOT / "data" / "category_catalog" / "v1"


class FixedCategoryCatalogError(RuntimeError):
    """餐厅类别统计或生成后的固定表不完整时抛出。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_inventory(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"dining category inventory does not exist: {path}")
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(DINING_CATEGORY_SCHEMA, check_metadata=False):
        raise FixedCategoryCatalogError("unexpected dining category schema")
    rows = parquet.read().to_pylist()
    names = [str(row["category"]) for row in rows]
    if not names or len(names) != len(set(names)):
        raise FixedCategoryCatalogError("dining categories must be nonempty and unique")
    return rows


def _document(rows: list[dict[str, object]]) -> FixedCategoryDocument:
    fixed_rows = []
    for row in rows:
        category = str(row["category"])
        classification = classify_category(category)
        fixed_rows.append(
            FixedCategoryRow(
                category=category,
                business_count=int(row["business_count"]),
                business_share=float(row["business_share"]),
                selectable=classification.selectable,
                category_kind=classification.category_kind,
                parent=classification.parent,
                parent_is_category=classification.parent_is_category,
            )
        )
    return FixedCategoryDocument(
        groups=list(GROUPS),
        categories=fixed_rows,
    )


def _load_existing(
    output_root: Path,
    *,
    source_hash: str,
) -> FixedCategoryManifest | None:
    manifest_path = output_root / "manifest.json"
    catalog_path = output_root / "catalog.json"
    categories_path = output_root / "categories.parquet"
    if not all(path.is_file() for path in (manifest_path, catalog_path, categories_path)):
        return None
    try:
        manifest = FixedCategoryManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if manifest.source_sha256 != source_hash:
        return None
    if manifest.output_sha256 != {
        "catalog": _sha256_file(catalog_path),
        "categories": _sha256_file(categories_path),
    }:
        return None
    return manifest


def build_fixed_category_catalog(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    force: bool = False,
) -> FixedCategoryBuildResult:
    """生成固定类别表；319种输入类别一条不少，每条都标明能否选择。"""

    source = Path(source_path)
    destination = Path(output_root)
    source_hash = _sha256_file(source) if source.is_file() else ""
    if not force and source_hash:
        existing = _load_existing(destination, source_hash=source_hash)
        if existing is not None:
            return FixedCategoryBuildResult(
                status="skipped",
                output_root=str(destination.resolve()),
                manifest=existing,
            )

    rows = _read_inventory(source)
    document = _document(rows)
    destination.mkdir(parents=True, exist_ok=True)
    catalog_path = destination / "catalog.json"
    categories_path = destination / "categories.parquet"
    manifest_path = destination / "manifest.json"
    temporary_catalog = destination / "catalog.json.partial"
    temporary_categories = destination / "categories.parquet.partial"
    temporary_manifest = destination / "manifest.json.partial"
    temporary_paths = (
        temporary_catalog,
        temporary_categories,
        temporary_manifest,
    )
    for path in temporary_paths:
        path.unlink(missing_ok=True)
    try:
        temporary_catalog.write_text(
            document.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        category_table = pa.Table.from_pylist(
            [item.model_dump(mode="python") for item in document.categories],
            schema=FIXED_CATEGORY_SCHEMA,
        )
        pq.write_table(category_table, temporary_categories, compression="zstd")
        output_hashes = {
            "catalog": _sha256_file(temporary_catalog),
            "categories": _sha256_file(temporary_categories),
        }
        selectable_count = sum(item.selectable for item in document.categories)
        manifest = FixedCategoryManifest(
            source_path=str(source.resolve()),
            source_sha256=source_hash,
            source_category_count=len(document.categories),
            selectable_category_count=selectable_count,
            non_selectable_category_count=len(document.categories) - selectable_count,
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
        os.replace(temporary_catalog, catalog_path)
        os.replace(temporary_categories, categories_path)
        os.replace(temporary_manifest, manifest_path)
    except Exception:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    return FixedCategoryBuildResult(
        status="written",
        output_root=str(destination.resolve()),
        manifest=manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成固定餐饮类别表")
    parser.add_argument("source_path", nargs="?", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("output_root", nargs="?", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_fixed_category_catalog(
        args.source_path,
        args.output_root,
        force=args.force,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
