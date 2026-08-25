from datetime import datetime
from zoneinfo import ZoneInfo

from yelp_agent.recommendation_v2.business_facts import (
    BusinessFact,
    WeeklyHours,
    is_open_at,
)


def _business(hours: WeeklyHours | None) -> BusinessFact:
    return BusinessFact(
        business_id="b1",
        name="Test",
        address="1 Main St",
        city="Philadelphia",
        state="PA",
        postal_code="19107",
        latitude=39.95,
        longitude=-75.16,
        categories=["Restaurants"],
        rating=4,
        review_count=10,
        weekly_hours=hours,
    )


def test_hours_handle_regular_and_overnight_intervals() -> None:
    business = _business(
        WeeklyHours(tuesday="17:0-22:0", friday="20:0-2:0")
    )

    timezone = ZoneInfo("America/New_York")
    assert is_open_at(business, datetime(2026, 8, 25, 21, tzinfo=timezone)) is True
    assert is_open_at(business, datetime(2026, 8, 25, 22, tzinfo=timezone)) is False
    assert is_open_at(business, datetime(2026, 8, 29, 1, tzinfo=timezone)) is True


def test_missing_and_ambiguous_hours_are_not_claimed_open() -> None:
    timezone = ZoneInfo("America/New_York")
    assert (
        is_open_at(_business(None), datetime(2026, 8, 25, 21, tzinfo=timezone))
        is None
    )
    assert (
        is_open_at(
            _business(WeeklyHours(tuesday="0:0-0:0")),
            datetime(2026, 8, 25, 12, tzinfo=timezone),
        )
        is False
    )
