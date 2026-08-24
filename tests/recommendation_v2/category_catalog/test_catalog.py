from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.recommendation_v2.business_catalog import DINING_CATEGORY_SCHEMA
from yelp_agent.recommendation_v2.category_catalog import (
    FixedCategoryCatalog,
    build_fixed_category_catalog,
)


def _inventory(path: Path) -> None:
    rows = [
        {
            "category": "Restaurants",
            "business_count": 10,
            "business_share": 1.0,
            "is_scope_marker": True,
        },
        {
            "category": "Chinese",
            "business_count": 5,
            "business_share": 0.5,
            "is_scope_marker": False,
        },
        {
            "category": "Szechuan",
            "business_count": 2,
            "business_share": 0.2,
            "is_scope_marker": False,
        },
        {
            "category": "Bars",
            "business_count": 2,
            "business_share": 0.2,
            "is_scope_marker": False,
        },
        {
            "category": "Wine Bars",
            "business_count": 1,
            "business_share": 0.1,
            "is_scope_marker": False,
        },
        {
            "category": "Hair Salons",
            "business_count": 1,
            "business_share": 0.1,
            "is_scope_marker": False,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=DINING_CATEGORY_SCHEMA), path)


def test_builds_closed_model_choices_and_real_parent_hierarchy(tmp_path: Path) -> None:
    source = tmp_path / "categories.parquet"
    output = tmp_path / "fixed"
    _inventory(source)

    result = build_fixed_category_catalog(source, output)
    catalog = FixedCategoryCatalog.from_file(output / "catalog.json")

    assert result.manifest.source_category_count == 6
    assert result.manifest.selectable_category_count == 4
    assert catalog.is_selectable("Chinese") is True
    assert catalog.is_selectable("Szechuan") is True
    assert catalog.is_selectable("Restaurants") is False
    assert catalog.is_selectable("Hair Salons") is False
    assert catalog.expand_for_filter("Chinese") == ("Chinese", "Szechuan")
    assert catalog.expand_for_filter("Bars") == ("Bars", "Wine Bars")
    assert catalog.model_options() == (
        {"category": "Chinese", "parent_category": None},
        {"category": "Szechuan", "parent_category": "Chinese"},
        {"category": "Bars", "parent_category": None},
        {"category": "Wine Bars", "parent_category": "Bars"},
    )


def test_reuses_complete_fixed_catalog(tmp_path: Path) -> None:
    source = tmp_path / "categories.parquet"
    output = tmp_path / "fixed"
    _inventory(source)

    first = build_fixed_category_catalog(source, output)
    second = build_fixed_category_catalog(source, output)

    assert first.status == "written"
    assert second.status == "skipped"
    assert second.manifest == first.manifest
