from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.recommendation_v2.business_facts import (
    BUSINESS_FACT_SCHEMA,
    PRICE_BAND_DOCUMENT,
    BusinessFact,
    BusinessFactCatalog,
    WeeklyHours,
)
from yelp_agent.recommendation_v2.category_catalog.catalog import FixedCategoryCatalog
from yelp_agent.recommendation_v2.category_catalog.schema import (
    CategoryGroup,
    FixedCategoryDocument,
    FixedCategoryRow,
)
from yelp_agent.recommendation_v2.schema import (
    GeoPoint,
    HardConstraint,
    RequirementBasis,
    UnifiedRecommendationState,
    merchant_feature_for,
    requirement_unit_for,
)
from yelp_agent.recommendation_v2.tools import (
    GeographicDistanceTool,
    StructuredHardFilterTool,
)


def _fact(
    business_id: str,
    category: str,
    *,
    latitude: float,
    price_level: int | None,
    accepts_reservations: bool | None,
    weekly_hours: WeeklyHours | None = None,
) -> BusinessFact:
    band = PRICE_BAND_DOCUMENT.bands[price_level - 1] if price_level else None
    return BusinessFact(
        business_id=business_id,
        name=f"Restaurant {business_id}",
        address="1 Main St",
        city="Philadelphia",
        state="PA",
        postal_code="19107",
        latitude=latitude,
        longitude=-75.16,
        categories=["Restaurants", "Chinese", category],
        price_level=price_level,
        price_lower_usd=None if band is None else band.lower_inclusive,
        price_upper_usd=None if band is None else band.upper_inclusive,
        rating=4.5,
        review_count=100,
        weekly_hours=weekly_hours,
        accepts_reservations=accepts_reservations,
    )


def _catalogs(tmp_path):
    facts = [
        _fact(
            "b1", "Szechuan", latitude=39.95, price_level=2,
            accepts_reservations=True,
            weekly_hours=WeeklyHours(tuesday="17:0-22:0"),
        ),
        _fact("b2", "Cantonese", latitude=39.951, price_level=2, accepts_reservations=False),
        _fact(
            "b3", "Szechuan", latitude=40.0, price_level=3,
            accepts_reservations=True,
            weekly_hours=WeeklyHours(tuesday="20:0-23:0"),
        ),
        _fact(
            "b4", "Szechuan", latitude=39.952, price_level=None,
            accepts_reservations=True,
            weekly_hours=WeeklyHours(tuesday="10:0-20:0"),
        ),
    ]
    fact_path = tmp_path / "facts.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [item.model_dump(mode="python") for item in facts],
            schema=BUSINESS_FACT_SCHEMA,
        ),
        fact_path,
    )
    businesses = BusinessFactCatalog(facts, PRICE_BAND_DOCUMENT, fact_path)
    categories = FixedCategoryCatalog(
        FixedCategoryDocument(
            groups=[CategoryGroup(group_id="cuisine", label="Cuisine")],
            categories=[
                FixedCategoryRow(
                    category="Chinese",
                    business_count=4,
                    business_share=1.0,
                    selectable=True,
                    category_kind="cuisine",
                    parent="cuisine",
                    parent_is_category=False,
                ),
                FixedCategoryRow(
                    category="Szechuan",
                    business_count=3,
                    business_share=0.75,
                    selectable=True,
                    category_kind="cuisine",
                    parent="Chinese",
                    parent_is_category=True,
                ),
                FixedCategoryRow(
                    category="Cantonese",
                    business_count=1,
                    business_share=0.25,
                    selectable=True,
                    category_kind="cuisine",
                    parent="Chinese",
                    parent_is_category=True,
                ),
            ],
        )
    )
    return businesses, categories


def _constraint(key: str, field: str, operator: str, value: object) -> HardConstraint:
    return HardConstraint.model_validate(
        {
            "key": key,
            "field": field,
            "operator": operator,
            "value": value,
            "unit": requirement_unit_for(field),
            "merchant_feature": merchant_feature_for(field),
            "controlling_source": "current_query",
            "sources": [
                RequirementBasis(
                    source="current_query",
                    text=f"test {field}",
                    turn_index=1,
                )
            ],
        }
    )


def test_geography_then_parameterized_hard_filter(tmp_path) -> None:
    businesses, categories = _catalogs(tmp_path)
    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=1,
        latest_query_text="Chinese within two kilometres, price level two, reservations",
        user_location=GeoPoint(latitude=39.95, longitude=-75.16),
        hard_constraints=[
            _constraint("category.any", "category", "any_of", ["Chinese"]),
            _constraint("distance.max", "distance_km", "less_than_or_equal", 2.0),
            _constraint("price.max", "price_level", "less_than_or_equal", 2),
            _constraint("reservation.yes", "accepts_reservations", "equals", True),
        ],
    )
    assert state.search_center is not None

    geography = GeographicDistanceTool(businesses).execute(state.search_center)
    result = StructuredHardFilterTool(businesses, categories).execute(
        state,
        geography=geography,
    )

    assert geography.source_business_count == 4
    assert geography.distance_by_business_id()["b1"] == 0
    assert 0.1 < geography.distance_by_business_id()["b2"] < 0.12
    assert result.source_business_count == 4
    assert result.candidate_business_ids == ["b1"]
    assert [item.after_count for item in result.steps] == [4, 3, 2, 1]
    assert result.steps[2].unknown_excluded_count == 1
    # 用户值只作为参数传入，不能被拼进程序生成的查询语句。
    assert "Chinese" not in result.generated_sql
    assert "?" in result.generated_sql


def test_open_at_is_a_real_hard_filter_and_unknown_hours_are_excluded(
    tmp_path,
) -> None:
    businesses, categories = _catalogs(tmp_path)
    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=1,
        latest_query_text="Tuesday at nine",
        hard_constraints=[
            _constraint(
                "open.exact",
                "open_at",
                "equals",
                "2026-08-25T21:00:00",
            )
        ],
    )

    result = StructuredHardFilterTool(businesses, categories).execute(state)

    assert result.candidate_business_ids == ["b1", "b3"]
    assert result.steps[0].unknown_excluded_count == 1


def test_default_supported_business_scope_is_applied_before_hard_constraints(
    tmp_path,
) -> None:
    businesses, categories = _catalogs(tmp_path)
    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=1,
        latest_query_text="find a restaurant",
    )
    tool = StructuredHardFilterTool(
        businesses,
        categories,
        default_candidate_business_ids=["b1", "b3"],
    )

    result = tool.execute(state)

    assert result.source_business_count == 2
    assert result.candidate_business_ids == ["b1", "b3"]
    assert "candidate_scope" in result.generated_sql
