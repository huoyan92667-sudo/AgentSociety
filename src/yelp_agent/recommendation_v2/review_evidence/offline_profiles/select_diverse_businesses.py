"""选择菜系覆盖更广、同时具备较好评论基础的500家餐饮商家。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_PROJECT = Path(r"C:\Users\29072\PycharmProjects\AgentSociety")
CUISINE_TARGET = 400
OTHER_DINING_TARGET = 100
RATING_PRIOR = 3.8
RATING_PRIOR_WEIGHT = 50
MINIMUM_RATING = 3.5


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_output_dir() -> Path:
    return (
        _project_root()
        / "src"
        / "yelp_agent"
        / "recommendation_v2"
        / "data"
        / "review_evidence"
        / "v1"
        / "offline_profiles"
        / "restaurants_diverse_500_v1"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, default=DEFAULT_SOURCE_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--cuisine-count", type=int, default=CUISINE_TARGET)
    parser.add_argument("--other-dining-count", type=int, default=OTHER_DINING_TARGET)
    parser.add_argument("--minimum-rating", type=float, default=MINIMUM_RATING)
    args = parser.parse_args(argv)
    if args.cuisine_count < 1 or args.other_dining_count < 1:
        parser.error("both selection counts must be positive")
    if not 1 <= args.minimum_rating <= 5:
        parser.error("minimum rating must be between 1 and 5")
    return args


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _load_categories(path: Path) -> dict[str, dict[str, Any]]:
    import duckdb

    connection = duckdb.connect(database=":memory:")
    try:
        rows = connection.execute(
            """
            SELECT category, selectable, category_kind, parent, parent_is_category
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchall()
    finally:
        connection.close()
    return {
        str(row[0]): {
            "category": str(row[0]),
            "selectable": bool(row[1]),
            "category_kind": str(row[2]),
            "parent": str(row[3]),
            "parent_is_category": bool(row[4]),
        }
        for row in rows
    }


def _load_businesses(*, facts_path: Path, reviews_path: Path) -> list[dict[str, Any]]:
    import duckdb

    connection = duckdb.connect(database=":memory:")
    try:
        rows = connection.execute(
            """
            WITH review_counts AS (
                SELECT business_id, count(*) AS actual_review_count
                FROM read_parquet(?)
                GROUP BY business_id
            )
            SELECT
                facts.business_id,
                facts.name,
                facts.address,
                facts.city,
                facts.state,
                facts.postal_code,
                facts.latitude,
                facts.longitude,
                facts.categories,
                facts.rating,
                facts.review_count,
                counts.actual_review_count,
                facts.price_level
            FROM read_parquet(?) AS facts
            JOIN review_counts AS counts USING (business_id)
            """,
            [str(reviews_path), str(facts_path)],
        ).fetchall()
    finally:
        connection.close()
    result: list[dict[str, Any]] = []
    for row in rows:
        actual_reviews = int(row[11])
        rating = float(row[9])
        # 少量评论的满分商家不能自动压过评论充分的成熟商家。
        adjusted_rating = (
            actual_reviews * rating + RATING_PRIOR_WEIGHT * RATING_PRIOR
        ) / (actual_reviews + RATING_PRIOR_WEIGHT)
        result.append(
            {
                "business_id": str(row[0]),
                "name": str(row[1]),
                "address": str(row[2]),
                "city": str(row[3]),
                "state": str(row[4]),
                "postal_code": str(row[5]),
                "latitude": float(row[6]),
                "longitude": float(row[7]),
                "categories": [str(value) for value in row[8]],
                "rating": rating,
                "catalog_review_count": int(row[10]),
                "actual_review_count": actual_reviews,
                "price_level": None if row[12] is None else int(row[12]),
                "adjusted_rating": adjusted_rating,
            }
        )
    return result


def _quality_key(business: dict[str, Any]) -> tuple[float, int, float, str]:
    return (
        -float(business["adjusted_rating"]),
        -int(business["actual_review_count"]),
        -float(business["rating"]),
        str(business["business_id"]),
    )


def _category_depth(category: str, catalog: dict[str, dict[str, Any]]) -> int:
    depth = 0
    current = catalog[category]
    visited = {category}
    while current["parent_is_category"]:
        parent = current["parent"]
        if parent in visited or parent not in catalog:
            break
        visited.add(parent)
        depth += 1
        current = catalog[parent]
    return depth


def _balanced_select(
    *,
    candidates: list[dict[str, Any]],
    category_names: list[str],
    memberships: dict[str, set[str]],
    target: int,
    phase: str,
) -> list[dict[str, Any]]:
    """先保证每类至少一家，再按类别现有数量的平方根平衡补齐。"""

    by_id = {item["business_id"]: item for item in candidates}
    pools: dict[str, list[str]] = {}
    for category in category_names:
        ids = [
            item["business_id"]
            for item in candidates
            if category in memberships[item["business_id"]]
        ]
        ids.sort(key=lambda business_id: _quality_key(by_id[business_id]))
        if ids:
            pools[category] = ids
    if len(candidates) < target:
        raise ValueError(
            f"{phase} has only {len(candidates)} candidates for {target} slots"
        )

    selected: dict[str, dict[str, Any]] = {}
    coverage = Counter()

    def add(business_id: str, trigger: str, reason: str) -> None:
        if business_id in selected:
            return
        row = {
            **by_id[business_id],
            "selection_phase": phase,
            "selection_category": trigger,
            "selection_reason": reason,
        }
        selected[business_id] = row
        coverage.update(memberships[business_id])

    # 稀有类别先处理；某家同时覆盖多个类别时，不重复占名额。
    for category in sorted(pools, key=lambda value: (len(pools[value]), value)):
        if coverage[category] > 0:
            continue
        business_id = next(
            (item for item in pools[category] if item not in selected), None
        )
        if business_id is not None:
            add(business_id, category, "category_first_coverage")

    while len(selected) < target:
        available: list[tuple[float, str, str]] = []
        for category, pool in pools.items():
            business_id = next((item for item in pool if item not in selected), None)
            if business_id is None:
                continue
            # 目标占比与该类别商家数量的平方根成正比，既照顾主流菜系，
            # 又避免它们按照原始数量完全淹没小菜系。
            balance = coverage[category] / math.sqrt(len(pool))
            available.append((balance, category, business_id))
        if not available:
            remaining = sorted(
                (item for item in candidates if item["business_id"] not in selected),
                key=_quality_key,
            )
            for item in remaining[: target - len(selected)]:
                add(item["business_id"], "quality_fallback", "quality_fill")
            break
        _, category, business_id = min(available)
        add(business_id, category, "balanced_category_fill")
    return list(selected.values())


def _primary_category(
    business: dict[str, Any],
    *,
    category_kind: str,
    catalog: dict[str, dict[str, Any]],
    availability: Counter[str],
) -> str | None:
    values = [
        category
        for category in business["categories"]
        if category in catalog
        and catalog[category]["selectable"]
        and catalog[category]["category_kind"] == category_kind
    ]
    if not values:
        return None
    return min(
        values,
        key=lambda category: (
            -_category_depth(category, catalog),
            availability[category],
            category,
        ),
    )


def _write_csv(path: Path, businesses: Iterable[dict[str, Any]]) -> None:
    fields = [
        "selection_index",
        "name",
        "business_id",
        "city",
        "state",
        "rating",
        "actual_review_count",
        "adjusted_rating",
        "selection_phase",
        "primary_cuisine",
        "selection_category",
        "cuisine_labels",
        "categories",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in businesses:
            writer.writerow(
                {
                    **{field: item.get(field) for field in fields},
                    "cuisine_labels": " | ".join(item["cuisine_labels"]),
                    "categories": " | ".join(item["categories"]),
                }
            )


def _write_markdown(
    path: Path, businesses: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    lines = [
        "# 菜系多样化500家商家清单",
        "",
        "## 选择结果",
        "",
        f"- 商家总数：{len(businesses)}",
        f"- 有菜系标签的商家：{summary['cuisine_business_count']}",
        f"- 其他重要餐饮类型：{summary['other_dining_business_count']}",
        f"- 覆盖菜系标签：{summary['covered_cuisine_count']} / {summary['available_cuisine_count']}",
        f"- 覆盖全部可选餐饮标签：{summary['covered_selectable_category_count']} / {summary['available_selectable_category_count']}",
        "",
        "## 完整清单",
        "",
        "| 序号 | 商家 | 城市 | 评分 | 真实评论 | 主要菜系/类型 | 其他菜系 |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for item in businesses:
        label = item["primary_cuisine"] or item["selection_category"]
        cuisines = "、".join(item["cuisine_labels"])
        name = item["name"].replace("|", "/")
        lines.append(
            f"| {item['selection_index']} | {name} | {item['city']} | "
            f"{item['rating']:.1f} | {item['actual_review_count']} | {label} | {cuisines} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select(args: argparse.Namespace) -> Path:
    root = _project_root()
    facts_path = (
        root
        / "src"
        / "yelp_agent"
        / "recommendation_v2"
        / "data"
        / "business_facts"
        / "v1"
        / "business_facts.parquet"
    )
    category_path = (
        root
        / "src"
        / "yelp_agent"
        / "recommendation_v2"
        / "data"
        / "category_catalog"
        / "v1"
        / "categories.parquet"
    )
    reviews_path = args.source_project / "data" / "processed" / "reviews.parquet"
    catalog = _load_categories(category_path)
    businesses = _load_businesses(facts_path=facts_path, reviews_path=reviews_path)
    eligible_businesses = [
        item for item in businesses if item["rating"] >= args.minimum_rating
    ]
    cuisine_names = sorted(
        category
        for category, row in catalog.items()
        if row["selectable"] and row["category_kind"] == "cuisine"
    )
    other_names = sorted(
        category
        for category, row in catalog.items()
        if row["selectable"] and row["category_kind"] != "cuisine"
    )
    memberships: dict[str, set[str]] = {}
    for business in eligible_businesses:
        memberships[business["business_id"]] = {
            category
            for category in business["categories"]
            if category in catalog and catalog[category]["selectable"]
        }
    cuisine_candidates = [
        item
        for item in eligible_businesses
        if memberships[item["business_id"]].intersection(cuisine_names)
    ]
    other_candidates = [
        item
        for item in eligible_businesses
        if not memberships[item["business_id"]].intersection(cuisine_names)
    ]
    cuisine_selected = _balanced_select(
        candidates=cuisine_candidates,
        category_names=cuisine_names,
        memberships=memberships,
        target=args.cuisine_count,
        phase="cuisine_diversity",
    )
    other_present = [
        category
        for category in other_names
        if any(
            category in memberships[item["business_id"]] for item in other_candidates
        )
    ]
    other_selected = _balanced_select(
        candidates=other_candidates,
        category_names=other_present,
        memberships=memberships,
        target=args.other_dining_count,
        phase="other_dining_diversity",
    )
    selected = cuisine_selected + other_selected
    cuisine_availability = Counter(
        category
        for item in eligible_businesses
        for category in memberships[item["business_id"]]
        if category in cuisine_names
    )
    for index, item in enumerate(selected, 1):
        item["selection_index"] = index
        item["cuisine_labels"] = sorted(
            memberships[item["business_id"]].intersection(cuisine_names)
        )
        item["primary_cuisine"] = _primary_category(
            item,
            category_kind="cuisine",
            catalog=catalog,
            availability=cuisine_availability,
        )

    covered_cuisines = sorted(
        {category for item in selected for category in item["cuisine_labels"]}
    )
    covered_selectable = sorted(
        {category for item in selected for category in memberships[item["business_id"]]}
    )
    primary_counts = Counter(
        item["primary_cuisine"] or item["selection_category"] for item in selected
    )
    summary = {
        "business_count": len(selected),
        "cuisine_business_count": len(cuisine_selected),
        "other_dining_business_count": len(other_selected),
        "available_cuisine_count": len(cuisine_names),
        "covered_cuisine_count": len(covered_cuisines),
        "covered_cuisines": covered_cuisines,
        "available_selectable_category_count": sum(
            row["selectable"] for row in catalog.values()
        ),
        "covered_selectable_category_count": len(covered_selectable),
        "covered_selectable_categories": covered_selectable,
        "primary_category_counts": dict(sorted(primary_counts.items())),
        "rating_counts": dict(
            sorted(Counter(str(item["rating"]) for item in selected).items())
        ),
        "actual_review_count": {
            "minimum": min(item["actual_review_count"] for item in selected),
            "maximum": max(item["actual_review_count"] for item in selected),
            "total": sum(item["actual_review_count"] for item in selected),
        },
        "city_counts": dict(Counter(item["city"] for item in selected).most_common()),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_policy": {
            "total": args.cuisine_count + args.other_dining_count,
            "cuisine_slots": args.cuisine_count,
            "other_dining_slots": args.other_dining_count,
            "category_balance": "先覆盖每类，再按该类可用商家数的平方根分配后续名额",
            "quality_order": (
                "评论量为50的3.8分作为先验，计算修正评分；同类内按修正评分和真实评论数选择"
            ),
            "minimum_actual_reviews": min(
                item["actual_review_count"] for item in eligible_businesses
            ),
            "minimum_rating": args.minimum_rating,
        },
        "summary": summary,
        "businesses": selected,
    }
    json_path = output_dir / "selection.json"
    _write_json(json_path, selection)
    _write_json(output_dir / "selection_summary.json", summary)
    _write_csv(output_dir / "selection.csv", selected)
    _write_markdown(output_dir / "selection.md", selected, summary)
    print(f"[SELECT] businesses={len(selected)}")
    print(f"[SELECT] cuisine_coverage={len(covered_cuisines)}/{len(cuisine_names)}")
    print(
        f"[SELECT] selectable_coverage={len(covered_selectable)}/"
        f"{summary['available_selectable_category_count']}"
    )
    print(f"[SELECT] output={output_dir}")
    return json_path


def main(argv: Sequence[str] | None = None) -> int:
    select(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
