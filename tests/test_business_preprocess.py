import json
from pathlib import Path

import pandas as pd
import pytest

from yelp_agent.config import load_config
from yelp_agent.data.businesses import (
    BusinessPreprocessError,
    preprocess_businesses,
    write_business_preprocess_report,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def business(
    business_id: str,
    *,
    city: str = "Philadelphia",
    is_open: int = 1,
    review_count: int = 20,
    categories: str | None = "Restaurants, Italian",
) -> dict:
    return {
        "business_id": business_id,
        "name": f"Business {business_id}",
        "address": "1 Market St",
        "city": city,
        "state": "PA",
        "postal_code": "19103",
        "latitude": 39.95,
        "longitude": -75.16,
        "stars": 4.5,
        "review_count": review_count,
        "is_open": is_open,
        "attributes": {"WiFi": "free", "OutdoorSeating": True},
        "categories": categories,
    }


def test_streams_and_filters_static_philadelphia_businesses(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "business.json"
    output_path = tmp_path / "processed" / "businesses.parquet"
    rows = [
        business("restaurant"),
        business("coffee", categories="Food, Coffee & Tea"),
        business("outside", city="Tampa"),
        business("closed", is_open=0),
        business("small", review_count=19),
        business("uncategorised", categories=None),
        business("unrelated", categories="Home Services, Plumbing"),
    ]
    write_jsonl(input_path, rows)
    config = load_config().data.model_copy(update={"review_chunk_size": 2})

    result = preprocess_businesses(input_path, output_path, config)

    frame = pd.read_parquet(output_path)
    assert result.status == "written"
    assert result.source_rows == 7
    assert result.selected_rows == 2
    assert set(frame["business_id"]) == {"restaurant", "coffee"}
    assert list(frame.columns) == [
        "business_id",
        "name",
        "address",
        "city",
        "state",
        "postal_code",
        "latitude",
        "longitude",
        "categories",
        "attributes_json",
    ]
    assert "stars" not in frame
    assert "review_count" not in frame
    restaurant = frame.set_index("business_id").loc["restaurant"]
    assert list(restaurant["categories"]) == ["Restaurants", "Italian"]
    assert json.loads(restaurant["attributes_json"])["WiFi"] == "free"
    assert result.rejection_counts == {
        "wrong_city": 1,
        "closed": 1,
        "insufficient_reviews": 1,
        "missing_categories": 1,
        "outside_allowed_categories": 1,
    }


def test_invalid_json_reports_line_and_leaves_no_partial_output(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "business.json"
    output_path = tmp_path / "processed" / "businesses.parquet"
    input_path.write_text(
        json.dumps(business("valid")) + "\n{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        BusinessPreprocessError,
        match="invalid JSON at line 2",
    ):
        preprocess_businesses(input_path, output_path, load_config().data)

    assert not output_path.exists()
    assert not output_path.with_name("businesses.parquet.partial").exists()


def test_duplicate_eligible_business_id_aborts_atomic_output(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "business.json"
    output_path = tmp_path / "processed" / "businesses.parquet"
    write_jsonl(input_path, [business("duplicate"), business("duplicate")])

    with pytest.raises(
        BusinessPreprocessError,
        match="duplicate eligible business_id at line 2",
    ):
        preprocess_businesses(input_path, output_path, load_config().data)

    assert not output_path.exists()
    assert not output_path.with_name("businesses.parquet.partial").exists()


def test_existing_valid_parquet_is_reused_without_rescanning_jsonl(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "business.json"
    output_path = tmp_path / "processed" / "businesses.parquet"
    write_jsonl(input_path, [business("first")])
    config = load_config().data
    preprocess_businesses(input_path, output_path, config)
    mtime_before = output_path.stat().st_mtime_ns
    write_jsonl(input_path, [business("first"), business("second")])

    result = preprocess_businesses(input_path, output_path, config)

    assert result.status == "skipped"
    assert result.source_rows is None
    assert result.selected_rows == 1
    assert output_path.stat().st_mtime_ns == mtime_before


def test_business_preprocess_report_can_be_saved_as_json(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "business.json"
    output_path = tmp_path / "processed" / "businesses.parquet"
    report_path = tmp_path / "runs" / "business_preprocess_report.json"
    write_jsonl(input_path, [business("first")])

    result = preprocess_businesses(input_path, output_path, load_config().data)
    write_business_preprocess_report(result, report_path)

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "written"
    assert saved["selected_rows"] == 1
