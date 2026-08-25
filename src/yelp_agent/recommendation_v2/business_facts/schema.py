"""统一商家基础事实、价格档位和生成记录的数据结构。"""

from __future__ import annotations

from typing import Literal, Self

import pyarrow as pa
from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel

BUSINESS_FACT_SCHEMA = pa.schema(
    [
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("address", pa.string(), nullable=False),
        pa.field("city", pa.string(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
        pa.field("postal_code", pa.string(), nullable=False),
        pa.field("latitude", pa.float64(), nullable=False),
        pa.field("longitude", pa.float64(), nullable=False),
        # 保留 Yelp 原始类别，不再复制一份“可筛选类别”。
        pa.field(
            "categories",
            pa.list_(pa.field("item", pa.string(), nullable=False)),
            nullable=False,
        ),
        pa.field("price_level", pa.int8(), nullable=True),
        pa.field("price_lower_usd", pa.int16(), nullable=True),
        pa.field("price_upper_usd", pa.int16(), nullable=True),
        pa.field("rating", pa.float64(), nullable=False),
        pa.field("review_count", pa.int64(), nullable=False),
        # Yelp 原始数据按星期保存一段营业时间。保留原始字符串，具体某个
        # 时刻是否营业由硬过滤工具统一解释，避免生成事实时偷偷改写含义。
        pa.field(
            "weekly_hours",
            pa.struct(
                [
                    pa.field("monday", pa.string(), nullable=True),
                    pa.field("tuesday", pa.string(), nullable=True),
                    pa.field("wednesday", pa.string(), nullable=True),
                    pa.field("thursday", pa.string(), nullable=True),
                    pa.field("friday", pa.string(), nullable=True),
                    pa.field("saturday", pa.string(), nullable=True),
                    pa.field("sunday", pa.string(), nullable=True),
                ]
            ),
            nullable=True,
        ),
        pa.field("accepts_reservations", pa.bool_(), nullable=True),
        pa.field("delivery", pa.bool_(), nullable=True),
        pa.field("takeout", pa.bool_(), nullable=True),
        pa.field("outdoor_seating", pa.bool_(), nullable=True),
        pa.field("good_for_kids", pa.bool_(), nullable=True),
        pa.field("good_for_groups", pa.bool_(), nullable=True),
        pa.field("wheelchair_accessible", pa.bool_(), nullable=True),
        pa.field("dogs_allowed", pa.bool_(), nullable=True),
        pa.field("parking_available", pa.bool_(), nullable=True),
        pa.field("parking_garage", pa.bool_(), nullable=True),
        pa.field("parking_street", pa.bool_(), nullable=True),
        pa.field("parking_validated", pa.bool_(), nullable=True),
        pa.field("parking_lot", pa.bool_(), nullable=True),
        pa.field("parking_valet", pa.bool_(), nullable=True),
    ]
)

BOOLEAN_FACT_FIELDS: tuple[str, ...] = (
    "accepts_reservations",
    "delivery",
    "takeout",
    "outdoor_seating",
    "good_for_kids",
    "good_for_groups",
    "wheelchair_accessible",
    "dogs_allowed",
    "parking_available",
    "parking_garage",
    "parking_street",
    "parking_validated",
    "parking_lot",
    "parking_valet",
)

# 统一要求中的商家特征在基础事实里只从这里找列。后续硬过滤工具不需要
# 再了解 Yelp 原始属性名称，也不会各写一套真假值解析逻辑。
BASE_FACT_FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "categories": ("categories",),
    "coordinates": ("latitude", "longitude"),
    "price_level": ("price_level",),
    "business_id": ("business_id",),
    "rating": ("rating",),
    "review_count": ("review_count",),
    "weekly_hours": ("weekly_hours",),
    "accepts_reservations": ("accepts_reservations",),
    "delivery": ("delivery",),
    "takeout": ("takeout",),
    "outdoor_seating": ("outdoor_seating",),
    "good_for_kids": ("good_for_kids",),
    "good_for_groups": ("good_for_groups",),
    "wheelchair_accessible": ("wheelchair_accessible",),
    "dogs_allowed": ("dogs_allowed",),
    "parking_available": ("parking_available",),
}


class WeeklyHours(StrictModel):
    """Yelp 数据中一家商户每周每天记录的一段营业时间。"""

    monday: str | None = None
    tuesday: str | None = None
    wednesday: str | None = None
    thursday: str | None = None
    friday: str | None = None
    saturday: str | None = None
    sunday: str | None = None

    @model_validator(mode="after")
    def validate_has_value(self) -> Self:
        if not any(getattr(self, day) for day in type(self).model_fields):
            raise ValueError("weekly hours must contain at least one day")
        return self


class PriceBand(StrictModel):
    """项目用于把 Yelp 四档价格转换成可比较金额区间的固定规则。"""

    price_level: Literal[1, 2, 3, 4]
    symbol: Literal["$", "$$", "$$$", "$$$$"]
    currency: Literal["USD"] = "USD"
    lower_inclusive: int = Field(ge=0)
    upper_inclusive: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if (
            self.upper_inclusive is not None
            and self.upper_inclusive < self.lower_inclusive
        ):
            raise ValueError("price upper bound cannot be below lower bound")
        if (self.price_level == 4) != (self.upper_inclusive is None):
            raise ValueError("only level 4 has an open upper bound")
        return self


class PriceBandDocument(StrictModel):
    """独立保存价格换算规则，供问题理解和过滤共同读取。"""

    schema_version: Literal[1] = 1
    policy_version: Literal["1.0.0"] = "1.0.0"
    meaning: Literal["approximate_cost_per_person"] = "approximate_cost_per_person"
    bands: list[PriceBand] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_complete_levels(self) -> Self:
        if [item.price_level for item in self.bands] != [1, 2, 3, 4]:
            raise ValueError("price bands must contain levels 1 through 4 in order")
        return self


class BusinessFact(StrictModel):
    """一行商家事实；空值明确表示原始商家数据没有给出答案。"""

    business_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: str
    city: str = Field(min_length=1)
    state: str = Field(min_length=1)
    postal_code: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    categories: list[str] = Field(min_length=1)
    price_level: Literal[1, 2, 3, 4] | None = None
    price_lower_usd: int | None = Field(default=None, ge=0)
    price_upper_usd: int | None = Field(default=None, ge=0)
    rating: float = Field(ge=1, le=5)
    review_count: int = Field(ge=0)
    weekly_hours: WeeklyHours | None = None
    accepts_reservations: bool | None = None
    delivery: bool | None = None
    takeout: bool | None = None
    outdoor_seating: bool | None = None
    good_for_kids: bool | None = None
    good_for_groups: bool | None = None
    wheelchair_accessible: bool | None = None
    dogs_allowed: bool | None = None
    parking_available: bool | None = None
    parking_garage: bool | None = None
    parking_street: bool | None = None
    parking_validated: bool | None = None
    parking_lot: bool | None = None
    parking_valet: bool | None = None

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, values: list[str]) -> list[str]:
        if any(not item or item != item.strip() for item in values):
            raise ValueError("categories must be nonempty and trimmed")
        if len(values) != len(set(values)):
            raise ValueError("categories must be unique")
        return values

    @model_validator(mode="after")
    def validate_derived_values(self) -> Self:
        if self.price_level is None:
            if self.price_lower_usd is not None or self.price_upper_usd is not None:
                raise ValueError("unknown price level cannot have a price range")
        else:
            expected = PRICE_BANDS_BY_LEVEL[self.price_level]
            if (
                self.price_lower_usd != expected.lower_inclusive
                or self.price_upper_usd != expected.upper_inclusive
            ):
                raise ValueError("price range must match the fixed price level")

        parking_values = (
            self.parking_garage,
            self.parking_street,
            self.parking_validated,
            self.parking_lot,
            self.parking_valet,
        )
        if any(value is True for value in parking_values):
            expected_parking = True
        elif all(value is False for value in parking_values):
            expected_parking = False
        else:
            expected_parking = None
        if self.parking_available is not expected_parking:
            raise ValueError("parking_available must summarize the five parking facts")
        return self


class BusinessFactManifest(StrictModel):
    """记录事实来源、字段覆盖情况和生成文件校验值。"""

    schema_version: Literal[1] = 1
    fact_version: Literal["1.1.0"] = "1.1.0"
    dining_source_path: str = Field(min_length=1)
    raw_business_source_path: str = Field(min_length=1)
    source_sha256: dict[str, str]
    business_count: int = Field(ge=1)
    known_value_counts: dict[str, int]
    output_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if set(self.source_sha256) != {"dining_businesses", "raw_businesses"}:
            raise ValueError("manifest must hash both business sources")
        if set(self.output_sha256) != {"business_facts", "price_bands"}:
            raise ValueError("manifest must hash facts and price bands")
        expected_counts = {"price_level", "weekly_hours", *BOOLEAN_FACT_FIELDS}
        if set(self.known_value_counts) != expected_counts:
            raise ValueError("manifest known-value fields are incomplete")
        if any(not 0 <= count <= self.business_count for count in self.known_value_counts.values()):
            raise ValueError("known-value counts must fit the business count")
        for hashes in (self.source_sha256, self.output_sha256):
            if any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes.values()
            ):
                raise ValueError("manifest hashes must be lowercase sha256 values")
        return self


class BusinessFactBuildResult(StrictModel):
    """一次基础事实生成的结果。"""

    status: Literal["written", "skipped"]
    output_root: str = Field(min_length=1)
    manifest: BusinessFactManifest


PRICE_BAND_DOCUMENT = PriceBandDocument(
    bands=[
        PriceBand(price_level=1, symbol="$", lower_inclusive=0, upper_inclusive=10),
        PriceBand(price_level=2, symbol="$$", lower_inclusive=11, upper_inclusive=30),
        PriceBand(price_level=3, symbol="$$$", lower_inclusive=31, upper_inclusive=60),
        PriceBand(price_level=4, symbol="$$$$", lower_inclusive=61),
    ]
)
PRICE_BANDS_BY_LEVEL = {item.price_level: item for item in PRICE_BAND_DOCUMENT.bands}
