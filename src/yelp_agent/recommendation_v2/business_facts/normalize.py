"""把 Yelp 原始商家属性转换成明确的真、假或未知。"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from .schema import PRICE_BANDS_BY_LEVEL, BusinessFact, WeeklyHours

DIRECT_BOOLEAN_ATTRIBUTES: dict[str, str] = {
    "accepts_reservations": "RestaurantsReservations",
    "delivery": "RestaurantsDelivery",
    "takeout": "RestaurantsTakeOut",
    "outdoor_seating": "OutdoorSeating",
    "good_for_kids": "GoodForKids",
    "good_for_groups": "RestaurantsGoodForGroups",
    "wheelchair_accessible": "WheelchairAccessible",
    "dogs_allowed": "DogsAllowed",
}

PARKING_KEYS: tuple[str, ...] = (
    "garage",
    "street",
    "validated",
    "lot",
    "valet",
)

_HOUR_DAY_KEYS: dict[str, str] = {
    "Monday": "monday",
    "Tuesday": "tuesday",
    "Wednesday": "wednesday",
    "Thursday": "thursday",
    "Friday": "friday",
    "Saturday": "saturday",
    "Sunday": "sunday",
}


class BusinessFactNormalizationError(ValueError):
    """原始商家中关键身份、评分或类别不合法时抛出。"""


def _tri_state_bool(value: object) -> bool | None:
    """只接受明确的真假；缺失、None 和无法识别的内容都保持未知。"""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().casefold()
    if normalized.startswith(("u'", 'u"')):
        normalized = normalized[1:]
    normalized = normalized.strip("'\"")
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _parking_values(value: object) -> dict[str, bool | None]:
    """拆开五种停车方式；原始停车对象无法解析时全部保持未知。"""

    parsed: object = value
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.casefold() == "none":
            parsed = None
        else:
            try:
                parsed = ast.literal_eval(normalized)
            except (SyntaxError, ValueError):
                parsed = None
    if not isinstance(parsed, Mapping):
        return {key: None for key in PARKING_KEYS}
    return {key: _tri_state_bool(parsed.get(key)) for key in PARKING_KEYS}


def _categories(value: object) -> list[str]:
    if isinstance(value, str):
        result = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
    else:
        result = []
    if not result:
        raise BusinessFactNormalizationError("restaurant has no categories")
    return list(dict.fromkeys(result))


def _price_level(attributes: Mapping[str, Any]) -> int | None:
    value = attributes.get("RestaurantsPriceRange2")
    if value is None or isinstance(value, bool):
        return None
    normalized = str(value).strip().strip("'\"")
    if normalized not in {"1", "2", "3", "4"}:
        return None
    return int(normalized)


def _weekly_hours(value: object) -> WeeklyHours | None:
    """只整理 Yelp 明确给出的星期与时间，不猜缺失日期。"""

    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, str | None] = {}
    for source_day, target_day in _HOUR_DAY_KEYS.items():
        raw = value.get(source_day)
        text = str(raw).strip() if raw is not None else ""
        normalized[target_day] = text or None
    if not any(normalized.values()):
        return None
    return WeeklyHours(**normalized)


def normalize_business_fact(raw: Mapping[str, Any]) -> BusinessFact:
    """生成一行可直接过滤和排序的商家基础事实。"""

    attributes_value = raw.get("attributes")
    attributes: Mapping[str, Any] = (
        attributes_value if isinstance(attributes_value, Mapping) else {}
    )
    price_level = _price_level(attributes)
    price_band = PRICE_BANDS_BY_LEVEL.get(price_level)
    parking = _parking_values(attributes.get("BusinessParking"))
    parking_items = [parking[key] for key in PARKING_KEYS]
    if any(value is True for value in parking_items):
        parking_available = True
    elif all(value is False for value in parking_items):
        parking_available = False
    else:
        parking_available = None

    try:
        rating = float(raw["stars"])
        review_count = int(raw["review_count"])
        latitude = float(raw["latitude"])
        longitude = float(raw["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessFactNormalizationError(
            "restaurant is missing coordinates, rating, or review count"
        ) from exc

    direct_booleans = {
        output_name: _tri_state_bool(attributes.get(source_name))
        for output_name, source_name in DIRECT_BOOLEAN_ATTRIBUTES.items()
    }
    return BusinessFact(
        business_id=str(raw.get("business_id", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        address=str(raw.get("address", "") or "").strip(),
        city=str(raw.get("city", "")).strip(),
        state=str(raw.get("state", "")).strip(),
        postal_code=str(raw.get("postal_code", "") or "").strip(),
        latitude=latitude,
        longitude=longitude,
        categories=_categories(raw.get("categories")),
        price_level=price_level,
        price_lower_usd=(
            price_band.lower_inclusive if price_band is not None else None
        ),
        price_upper_usd=(
            price_band.upper_inclusive if price_band is not None else None
        ),
        rating=rating,
        review_count=review_count,
        weekly_hours=_weekly_hours(raw.get("hours")),
        parking_available=parking_available,
        parking_garage=parking["garage"],
        parking_street=parking["street"],
        parking_validated=parking["validated"],
        parking_lot=parking["lot"],
        parking_valet=parking["valet"],
        **direct_booleans,
    )
