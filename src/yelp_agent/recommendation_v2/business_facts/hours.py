"""把 Yelp 每周营业时间解释成某个费城本地时刻是否营业。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .schema import BusinessFact, WeeklyHours

CATALOG_TIME_ZONE = "America/New_York"
_DAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def catalog_local_time(value: datetime) -> datetime:
    """把请求时间统一转换成当前费城餐厅目录使用的本地时间。"""

    timezone = ZoneInfo(CATALOG_TIME_ZONE)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone)
    return value.astimezone(timezone)


def parse_visit_time(value: str) -> datetime:
    """解析统一结构里的到店时间；无时区时按费城本地时间处理。"""

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("open_at must be an ISO date and time") from exc
    return catalog_local_time(parsed)


def is_open_at(business: BusinessFact, visit_time: datetime) -> bool | None:
    """有营业时间时返回真假；原始数据缺失时明确返回未知。"""

    if business.weekly_hours is None:
        return None
    local = catalog_local_time(visit_time)
    day_index = local.weekday()
    minute = local.hour * 60 + local.minute
    current = _hours_for_day(business.weekly_hours, day_index)
    previous = _hours_for_day(business.weekly_hours, (day_index - 1) % 7)

    if current is not None and _contains_current_day(current, minute):
        return True
    return previous is not None and _contains_previous_overnight(previous, minute)


def _hours_for_day(hours: WeeklyHours, day_index: int) -> str | None:
    return getattr(hours, _DAYS[day_index])


def _parse_clock(value: str) -> int:
    pieces = value.split(":")
    if len(pieces) != 2:
        raise ValueError("Yelp hour must use hour:minute")
    hour, minute = (int(item) for item in pieces)
    if not 0 <= hour <= 24 or not 0 <= minute <= 59:
        raise ValueError("Yelp hour is outside the valid clock range")
    if hour == 24 and minute != 0:
        raise ValueError("24 is only valid as 24:00")
    return hour * 60 + minute


def _interval(value: str) -> tuple[int, int]:
    pieces = value.split("-")
    if len(pieces) != 2:
        raise ValueError("Yelp hours must contain one start-end interval")
    return _parse_clock(pieces[0]), _parse_clock(pieces[1])


def _contains_current_day(value: str, minute: int) -> bool:
    start, end = _interval(value)
    # Yelp 数据中的 0:0-0:0 同时可能被外部系统解释为休息或全天营业。
    # 硬过滤采用保守做法：没有可靠依据时不把它当成正在营业。
    if start == end:
        return False
    if end > start:
        return start <= minute < end
    return minute >= start


def _contains_previous_overnight(value: str, minute: int) -> bool:
    start, end = _interval(value)
    return end < start and minute < end
