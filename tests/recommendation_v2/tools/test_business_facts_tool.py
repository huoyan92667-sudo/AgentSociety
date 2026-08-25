from pathlib import Path

from yelp_agent.recommendation_v2.business_facts import (
    PRICE_BAND_DOCUMENT,
    BusinessFact,
    BusinessFactCatalog,
    WeeklyHours,
)
from yelp_agent.recommendation_v2.tools import BusinessFactsQuery, BusinessFactsTool


def test_business_fact_tool_returns_new_facts_including_hours(tmp_path: Path) -> None:
    business = BusinessFact(
        business_id="b1",
        name="川味馆",
        address="1 Main St",
        city="Philadelphia",
        state="PA",
        postal_code="19107",
        latitude=39.95,
        longitude=-75.16,
        categories=["Restaurants", "Szechuan"],
        rating=4.5,
        review_count=100,
        weekly_hours=WeeklyHours(tuesday="17:0-22:0"),
    )
    catalog = BusinessFactCatalog(
        [business],
        PRICE_BAND_DOCUMENT,
        tmp_path / "facts.parquet",
    )

    result = BusinessFactsTool(catalog).execute(
        BusinessFactsQuery(business_ids=["b1", "missing"])
    )

    assert result.status == "partial"
    assert result.businesses[0].weekly_hours is not None
    assert result.businesses[0].weekly_hours.tuesday == "17:0-22:0"
    assert result.missing_business_ids == ["missing"]
