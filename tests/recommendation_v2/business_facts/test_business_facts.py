from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.data.businesses import BUSINESS_SCHEMA
from yelp_agent.recommendation_v2.business_facts import (
    BASE_FACT_FEATURE_COLUMNS,
    BUSINESS_FACT_SCHEMA,
    BusinessFactCatalog,
    build_business_facts,
)


def _sources(root: Path) -> tuple[Path, Path]:
    dining_path = root / "dining.parquet"
    raw_path = root / "business.jsonl"
    dining_rows = [
        {
            "business_id": "b1",
            "name": "川味馆",
            "address": "1 Main St",
            "city": "Philadelphia",
            "state": "PA",
            "postal_code": "19107",
            "latitude": 39.95,
            "longitude": -75.16,
            "categories": ["Restaurants", "Chinese", "Szechuan"],
            "attributes_json": "{}",
        },
        {
            "business_id": "b2",
            "name": "未知属性餐厅",
            "address": "2 Main St",
            "city": "Philadelphia",
            "state": "PA",
            "postal_code": "19107",
            "latitude": 39.96,
            "longitude": -75.17,
            "categories": ["Restaurants", "Italian"],
            "attributes_json": "{}",
        },
    ]
    pq.write_table(pa.Table.from_pylist(dining_rows, schema=BUSINESS_SCHEMA), dining_path)
    raw_rows = [
        {
            "business_id": "b1",
            "name": "川味馆",
            "address": "1 Main St",
            "city": "Philadelphia",
            "state": "PA",
            "postal_code": "19107",
            "latitude": 39.95,
            "longitude": -75.16,
            "stars": 4.5,
            "review_count": 123,
            "is_open": 1,
            "attributes": {
                "RestaurantsPriceRange2": "2",
                "RestaurantsReservations": "True",
                "RestaurantsDelivery": "None",
                "RestaurantsTakeOut": "False",
                "BusinessParking": (
                    "{'garage': False, 'street': True, 'validated': None, "
                    "'lot': False, 'valet': False}"
                ),
            },
            "categories": "Restaurants, Chinese, Szechuan",
            "hours": None,
        },
        {
            "business_id": "b2",
            "name": "未知属性餐厅",
            "address": "2 Main St",
            "city": "Philadelphia",
            "state": "PA",
            "postal_code": "19107",
            "latitude": 39.96,
            "longitude": -75.17,
            "stars": 3.5,
            "review_count": 50,
            "is_open": 1,
            "attributes": None,
            "categories": "Restaurants, Italian",
            "hours": None,
        },
    ]
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    return dining_path, raw_path


def test_builds_raw_categories_prices_ratings_and_unknown_values(tmp_path: Path) -> None:
    dining_path, raw_path = _sources(tmp_path)
    output = tmp_path / "facts"

    result = build_business_facts(dining_path, raw_path, output)
    catalog = BusinessFactCatalog.from_files(
        output / "business_facts.parquet",
        output / "price_bands.json",
    )
    known = catalog.get("b1")
    unknown = catalog.get("b2")

    assert result.manifest.business_count == 2
    # 原始类别原样保留，不生成第二份“可筛选类别”。
    assert known.categories == ["Restaurants", "Chinese", "Szechuan"]
    assert known.rating == 4.5
    assert known.review_count == 123
    assert known.price_level == 2
    assert (known.price_lower_usd, known.price_upper_usd) == (11, 30)
    assert known.accepts_reservations is True
    assert known.delivery is None
    assert known.takeout is False
    assert known.parking_street is True
    assert known.parking_available is True
    assert unknown.price_level is None
    assert unknown.accepts_reservations is None
    assert unknown.parking_available is None
    assert catalog.price_band(4).upper_inclusive is None


def test_reuses_a_complete_business_fact_build(tmp_path: Path) -> None:
    dining_path, raw_path = _sources(tmp_path)
    output = tmp_path / "facts"

    first = build_business_facts(dining_path, raw_path, output)
    second = build_business_facts(dining_path, raw_path, output)

    assert first.status == "written"
    assert second.status == "skipped"
    assert second.manifest == first.manifest


def test_every_base_feature_points_to_real_fact_columns() -> None:
    """防止统一要求声明了商家特征，基础事实映射却指向不存在的列。"""

    fact_columns = set(BUSINESS_FACT_SCHEMA.names)
    mapped_columns = {
        column
        for columns in BASE_FACT_FEATURE_COLUMNS.values()
        for column in columns
    }

    assert mapped_columns <= fact_columns
    assert set(BASE_FACT_FEATURE_COLUMNS) == {
        "categories",
        "coordinates",
        "price_level",
        "business_id",
        "rating",
        "review_count",
        "accepts_reservations",
        "delivery",
        "takeout",
        "outdoor_seating",
        "good_for_kids",
        "good_for_groups",
        "wheelchair_accessible",
        "dogs_allowed",
        "parking_available",
    }
