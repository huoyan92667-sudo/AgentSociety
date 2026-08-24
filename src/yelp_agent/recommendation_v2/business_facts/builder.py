"""从餐厅范围和 Yelp 原始商家文件生成统一商家基础事实。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from yelp_agent.data.businesses import BUSINESS_SCHEMA

from .normalize import normalize_business_fact
from .schema import (
    BOOLEAN_FACT_FIELDS,
    BUSINESS_FACT_SCHEMA,
    PRICE_BAND_DOCUMENT,
    BusinessFact,
    BusinessFactBuildResult,
    BusinessFactManifest,
)

_RECOMMENDATION_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DINING_SOURCE = (
    _RECOMMENDATION_ROOT / "data" / "dining_catalog" / "v1" / "businesses.parquet"
)
DEFAULT_RAW_BUSINESS_SOURCE = (
    _PROJECT_ROOT / "data" / "raw" / "yelp_academic_dataset_business.json"
)
DEFAULT_OUTPUT_ROOT = _RECOMMENDATION_ROOT / "data" / "business_facts" / "v1"


class BusinessFactBuildError(RuntimeError):
    """餐厅范围和原始商家无法形成完整的一一对应关系时抛出。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dining_scope(path: Path) -> dict[str, set[str]]:
    """只从已经确认的2447家餐厅中读取编号和原始类别。"""

    if not path.is_file():
        raise FileNotFoundError(f"dining business file does not exist: {path}")
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(BUSINESS_SCHEMA, check_metadata=False):
        raise BusinessFactBuildError("dining business file has an unexpected schema")
    rows = parquet.read(columns=["business_id", "categories"]).to_pylist()
    result = {
        str(row["business_id"]): {str(item) for item in row["categories"]}
        for row in rows
    }
    if not result or len(result) != len(rows):
        raise BusinessFactBuildError("dining business IDs must be nonempty and unique")
    return result


def _read_raw_facts(
    path: Path,
    dining_scope: dict[str, set[str]],
) -> list[BusinessFact]:
    """流式扫描原始商家文件，只保留已经确认属于餐厅范围的记录。"""

    if not path.is_file():
        raise FileNotFoundError(f"raw Yelp business file does not exist: {path}")
    facts: list[BusinessFact] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BusinessFactBuildError(
                    f"invalid raw business JSON at line {line_number}"
                ) from exc
            if not isinstance(raw, dict):
                raise BusinessFactBuildError(
                    f"raw business line {line_number} is not an object"
                )
            business_id = str(raw.get("business_id", "")).strip()
            expected_categories = dining_scope.get(business_id)
            if expected_categories is None:
                continue
            if business_id in seen:
                raise BusinessFactBuildError(
                    f"duplicate target business in raw source: {business_id}"
                )
            fact = normalize_business_fact(raw)
            if set(fact.categories) != expected_categories:
                raise BusinessFactBuildError(
                    f"raw categories changed for target business: {business_id}"
                )
            facts.append(fact)
            seen.add(business_id)

    missing = set(dining_scope).difference(seen)
    if missing:
        preview = ", ".join(sorted(missing)[:3])
        raise BusinessFactBuildError(
            f"raw source is missing {len(missing)} dining businesses: {preview}"
        )
    return sorted(facts, key=lambda item: item.business_id)


def _known_value_counts(facts: list[BusinessFact]) -> dict[str, int]:
    fields = ("price_level", *BOOLEAN_FACT_FIELDS)
    return {
        field: sum(getattr(fact, field) is not None for fact in facts)
        for field in fields
    }


def _load_existing(
    output_root: Path,
    *,
    source_hashes: dict[str, str],
) -> BusinessFactManifest | None:
    manifest_path = output_root / "manifest.json"
    fact_path = output_root / "business_facts.parquet"
    price_path = output_root / "price_bands.json"
    if not all(path.is_file() for path in (manifest_path, fact_path, price_path)):
        return None
    try:
        manifest = BusinessFactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    if manifest.source_sha256 != source_hashes:
        return None
    actual_hashes = {
        "business_facts": _sha256_file(fact_path),
        "price_bands": _sha256_file(price_path),
    }
    if manifest.output_sha256 != actual_hashes:
        return None
    return manifest


def build_business_facts(
    dining_source: str | Path = DEFAULT_DINING_SOURCE,
    raw_business_source: str | Path = DEFAULT_RAW_BUSINESS_SOURCE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    force: bool = False,
) -> BusinessFactBuildResult:
    """生成一行一个商家的基础事实；不会构造评论特征或评论证据。"""

    dining_path = Path(dining_source)
    raw_path = Path(raw_business_source)
    destination = Path(output_root)
    if not dining_path.is_file():
        raise FileNotFoundError(f"dining business file does not exist: {dining_path}")
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw Yelp business file does not exist: {raw_path}")
    source_hashes = {
        "dining_businesses": _sha256_file(dining_path),
        "raw_businesses": _sha256_file(raw_path),
    }
    if not force:
        existing = _load_existing(destination, source_hashes=source_hashes)
        if existing is not None:
            return BusinessFactBuildResult(
                status="skipped",
                output_root=str(destination.resolve()),
                manifest=existing,
            )

    facts = _read_raw_facts(raw_path, _dining_scope(dining_path))
    destination.mkdir(parents=True, exist_ok=True)
    fact_path = destination / "business_facts.parquet"
    price_path = destination / "price_bands.json"
    manifest_path = destination / "manifest.json"
    temporary_fact = destination / "business_facts.parquet.partial"
    temporary_price = destination / "price_bands.json.partial"
    temporary_manifest = destination / "manifest.json.partial"
    temporary_paths = (temporary_fact, temporary_price, temporary_manifest)
    for path in temporary_paths:
        path.unlink(missing_ok=True)
    try:
        table = pa.Table.from_pylist(
            [fact.model_dump(mode="python") for fact in facts],
            schema=BUSINESS_FACT_SCHEMA,
        )
        pq.write_table(table, temporary_fact, compression="zstd")
        temporary_price.write_text(
            PRICE_BAND_DOCUMENT.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            "business_facts": _sha256_file(temporary_fact),
            "price_bands": _sha256_file(temporary_price),
        }
        manifest = BusinessFactManifest(
            dining_source_path=str(dining_path.resolve()),
            raw_business_source_path=str(raw_path.resolve()),
            source_sha256=source_hashes,
            business_count=len(facts),
            known_value_counts=_known_value_counts(facts),
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
        os.replace(temporary_fact, fact_path)
        os.replace(temporary_price, price_path)
        os.replace(temporary_manifest, manifest_path)
    except Exception:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise

    return BusinessFactBuildResult(
        status="written",
        output_root=str(destination.resolve()),
        manifest=manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成统一餐厅商家基础事实")
    parser.add_argument("dining_source", nargs="?", type=Path, default=DEFAULT_DINING_SOURCE)
    parser.add_argument(
        "raw_business_source",
        nargs="?",
        type=Path,
        default=DEFAULT_RAW_BUSINESS_SOURCE,
    )
    parser.add_argument("output_root", nargs="?", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_business_facts(
        args.dining_source,
        args.raw_business_source,
        args.output_root,
        force=args.force,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
