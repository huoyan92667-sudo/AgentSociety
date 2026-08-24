from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yelp_agent.data.businesses import BUSINESS_SCHEMA
from yelp_agent.recommendation_v2.business_catalog import build_dining_catalog
from yelp_agent.recommendation_v2.business_catalog.builder import DiningCatalogError


def _business(
    business_id: str,
    categories: list[str],
) -> dict[str, object]:
    """建立和真实静态商家文件完全相同的一行。"""

    return {
        "business_id": business_id,
        "name": f"Business {business_id}",
        "address": "1 Test Street",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19103",
        "latitude": 39.95,
        "longitude": -75.16,
        "categories": categories,
        "attributes_json": json.dumps({"RestaurantsPriceRange2": "2"}),
    }


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=BUSINESS_SCHEMA), path)


def test_builds_strict_restaurant_subset_without_pruning_rare_categories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "businesses.parquet"
    output = tmp_path / "dining"
    _write_source(
        source,
        [
            _business("szechuan", ["Restaurants", "Chinese", "Szechuan"]),
            _business("pizza", ["Restaurants", "Pizza"]),
            _business("coffee", ["Food", "Coffee & Tea"]),
            _business("salon", ["Beauty & Spas", "Hair Salons"]),
        ],
    )

    result = build_dining_catalog(source, output)

    assert result.status == "written"
    assert result.manifest.source_business_count == 4
    assert result.manifest.dining_business_count == 2
    assert result.manifest.source_category_count == 8
    assert result.manifest.dining_category_count == 4
    businesses = pq.read_table(output / "businesses.parquet").to_pylist()
    assert [item["business_id"] for item in businesses] == ["pizza", "szechuan"]
    assert businesses[1]["categories"] == ["Restaurants", "Chinese", "Szechuan"]
    categories = pq.read_table(output / "categories.parquet").to_pylist()
    by_name = {item["category"]: item for item in categories}
    assert by_name["Restaurants"]["business_count"] == 2
    assert by_name["Restaurants"]["is_scope_marker"] is True
    assert by_name["Szechuan"]["business_count"] == 1
    assert by_name["Szechuan"]["is_scope_marker"] is False


def test_reuses_complete_matching_artifact(tmp_path: Path) -> None:
    source = tmp_path / "businesses.parquet"
    output = tmp_path / "dining"
    _write_source(source, [_business("one", ["Restaurants", "Cantonese"])])

    first = build_dining_catalog(source, output)
    second = build_dining_catalog(source, output)

    assert first.status == "written"
    assert second.status == "skipped"
    assert second.manifest == first.manifest


def test_rejects_source_without_restaurants(tmp_path: Path) -> None:
    source = tmp_path / "businesses.parquet"
    _write_source(source, [_business("coffee", ["Food", "Coffee & Tea"])])

    with pytest.raises(DiningCatalogError, match="no Restaurants"):
        build_dining_catalog(source, tmp_path / "dining")
