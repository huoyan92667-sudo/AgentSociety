"""Unified contracts for the new recommendation flow.

This package deliberately does not import the old query, session-memory, or
profile contracts.  Adapters will translate those records into this one shape.
"""

from __future__ import annotations

from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel

type SourceKind = Literal[
    "current_query",
    "session",
    "scene",
    "user_profile",
    "system_default",
]
type SceneKind = Literal[
    "casual",
    "date",
    "business",
    "friends",
    "family",
    "solo",
    "custom",
]
type BaselineSceneKind = Literal[
    "casual",
    "date",
    "business",
    "friends",
    "family",
    "solo",
]
type SearchCenterKind = Literal[
    "current_location",
    "named_place",
    "postal_code",
    "business",
]
type AspectField = Literal[
    "food_quality",
    "service",
    "price_value",
    "quiet_environment",
    "crowded",
    "queue_time",
    "portion_size",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "spiciness",
    "cleanliness",
]
# 第一批硬筛选只允许数据库中能够直接核对的结构化字段。
type HardConstraintField = Literal[
    "category",
    "distance_km",
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
    "open_at",
]
# 软偏好既可以使用结构化字段，也可以使用评论归纳出的商家特征。
type RequirementField = Literal[
    "category",
    "distance_km",
    "price_level",
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
    "food_quality",
    "service",
    "price_value",
    "quiet_environment",
    "crowded",
    "queue_time",
    "portion_size",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "spiciness",
    "cleanliness",
]
type MerchantFeature = Literal[
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
    "weekly_hours",
    "food_quality",
    "service",
    "price_value",
    "quiet_environment",
    "crowded",
    "queue_time",
    "portion_size",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "spiciness",
    "cleanliness",
]
type RequirementUnit = Literal[
    "category",
    "kilometer",
    "price_level",
    "business_id",
    "rating",
    "count",
    "boolean",
    "match_score",
    "datetime",
]
type HardOperator = Literal[
    "equals",
    "any_of",
    "all_of",
    "none_of",
    "less_than",
    "less_than_or_equal",
    "greater_than",
    "greater_than_or_equal",
]
type PreferenceDirection = Literal[
    "higher",
    "lower",
    "closer_to",
    "match",
    "avoid",
]
type PreferenceStrength = Literal[25, 50, 75, 100]
type PreferenceMemoryStatus = Literal[
    "active",
    "supporting",
    "suppressed",
    "shadowed",
]
type RequirementValue = str | int | float | bool | list[str]
type DerivationKind = Literal[
    "reference_business_distance",
    "budget_to_price_level",
]
type OpenRequirementBehavior = Literal["must_have", "prefer", "avoid"]


ASPECT_FIELDS: tuple[AspectField, ...] = (
    "food_quality",
    "service",
    "price_value",
    "quiet_environment",
    "crowded",
    "queue_time",
    "portion_size",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "spiciness",
    "cleanliness",
)

_SOURCE_PRIORITY: dict[SourceKind, int] = {
    "current_query": 4,
    "session": 3,
    "user_profile": 2,
    "scene": 1,
    "system_default": 0,
}
_FIELD_FEATURE: dict[RequirementField, MerchantFeature] = {
    "category": "categories",
    "distance_km": "coordinates",
    "price_level": "price_level",
    "rating": "rating",
    "review_count": "review_count",
    "accepts_reservations": "accepts_reservations",
    "delivery": "delivery",
    "takeout": "takeout",
    "outdoor_seating": "outdoor_seating",
    "good_for_kids": "good_for_kids",
    "good_for_groups": "good_for_groups",
    "wheelchair_accessible": "wheelchair_accessible",
    "dogs_allowed": "dogs_allowed",
    "parking_available": "parking_available",
    **{aspect: aspect for aspect in ASPECT_FIELDS},
}
_HARD_ONLY_FIELD_FEATURE: dict[
    Literal["business_id", "open_at"], MerchantFeature
] = {
    "business_id": "business_id",
    "open_at": "weekly_hours",
}
_FIELD_UNIT: dict[RequirementField, RequirementUnit] = {
    "category": "category",
    "distance_km": "kilometer",
    "price_level": "price_level",
    "rating": "rating",
    "review_count": "count",
    "accepts_reservations": "boolean",
    "delivery": "boolean",
    "takeout": "boolean",
    "outdoor_seating": "boolean",
    "good_for_kids": "boolean",
    "good_for_groups": "boolean",
    "wheelchair_accessible": "boolean",
    "dogs_allowed": "boolean",
    "parking_available": "boolean",
    **{aspect: "match_score" for aspect in ASPECT_FIELDS},
}
_HARD_ONLY_FIELD_UNIT: dict[
    Literal["business_id", "open_at"], RequirementUnit
] = {
    "business_id": "business_id",
    "open_at": "datetime",
}
_COLLECTION_FIELDS = {"category", "business_id"}
_COLLECTION_OPERATORS = {"any_of", "all_of", "none_of"}
_BOOLEAN_FIELDS = {
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
_NUMERIC_OPERATORS = {
    "equals",
    "less_than",
    "less_than_or_equal",
    "greater_than",
    "greater_than_or_equal",
}


def merchant_feature_for(
    field: HardConstraintField | RequirementField,
) -> MerchantFeature:
    """返回某个要求字段在商家数据中对应的真实字段。

    大模型只需要说明用户要求了什么，不需要记住数据库字段名。所有模块都
    通过这里补齐商家字段，避免不同文件各写一份映射后逐渐不一致。
    """

    if field in _HARD_ONLY_FIELD_FEATURE:
        return _HARD_ONLY_FIELD_FEATURE[field]
    return _FIELD_FEATURE[field]


def requirement_unit_for(
    field: HardConstraintField | RequirementField,
) -> RequirementUnit:
    """返回某个要求字段的固定单位，供完整统一状态自动补齐。"""

    if field in _HARD_ONLY_FIELD_UNIT:
        return _HARD_ONLY_FIELD_UNIT[field]
    return _FIELD_UNIT[field]


def _is_number(value: object) -> bool:
    """判断条件值是不是可用于大小比较的数值，并排除真假值。"""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_sources(
    sources: list[RequirementBasis],
    controlling_source: SourceKind,
) -> None:
    """检查一个要求的主要来源是否遵守既定的来源优先顺序。"""

    source_kinds = {item.source for item in sources}
    if controlling_source not in source_kinds:
        raise ValueError("controlling_source must appear in sources")
    highest = max(source_kinds, key=_SOURCE_PRIORITY.__getitem__)
    if controlling_source != highest:
        raise ValueError("controlling_source must be the highest-priority source")
    keys = {
        (item.source, item.turn_index, item.text, item.profile_last_confirmed)
        for item in sources
    }
    if len(keys) != len(sources):
        raise ValueError("requirement sources must be unique")


def _validate_structured_constraint(
    *,
    field: HardConstraintField,
    operator: HardOperator,
    value: RequirementValue,
    unit: RequirementUnit,
    merchant_feature: MerchantFeature,
    derivation: ConstraintDerivation | None,
) -> None:
    """统一检查用户硬条件和场景默认筛选条件的数据是否合法。"""

    expected_feature = (
        _HARD_ONLY_FIELD_FEATURE[field]
        if field in _HARD_ONLY_FIELD_FEATURE
        else _FIELD_FEATURE[field]
    )
    expected_unit = (
        _HARD_ONLY_FIELD_UNIT[field]
        if field in _HARD_ONLY_FIELD_UNIT
        else _FIELD_UNIT[field]
    )
    if merchant_feature != expected_feature:
        raise ValueError("merchant_feature does not match the requirement field")
    if unit != expected_unit:
        raise ValueError("unit does not match the requirement field")

    if field in _COLLECTION_FIELDS:
        if operator not in _COLLECTION_OPERATORS:
            raise ValueError(
                "category and business filters use any_of/all_of/none_of"
            )
        if not isinstance(value, list) or not value:
            raise ValueError("collection filters require a nonempty string list")
        if any(not item or item != item.strip() for item in value):
            raise ValueError("collection filter values must be nonempty and trimmed")
        if len(value) != len(set(value)):
            raise ValueError("collection filter values must be unique")
    elif field == "open_at":
        if operator != "equals" or not isinstance(value, str):
            raise ValueError("open_at requires equals and an ISO date-time string")
        if "T" not in value and " " not in value:
            raise ValueError("open_at must include both date and time")
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("open_at must be an ISO date-time string") from exc
    elif field in _BOOLEAN_FIELDS:
        if operator != "equals" or not isinstance(value, bool):
            raise ValueError("boolean filters require equals and a true/false value")
    elif operator not in _NUMERIC_OPERATORS or not _is_number(value):
        raise ValueError("numeric filters require a numeric comparison")

    if field == "distance_km" and float(value) < 0:
        raise ValueError("distance cannot be negative")
    if field == "price_level" and (
        not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 4
    ):
        raise ValueError("price level must be an integer from 1 to 4")
    if field == "rating" and not 1 <= float(value) <= 5:
        raise ValueError("rating must be between 1 and 5")
    if field == "review_count" and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError("review count must be a nonnegative integer")
    if derivation is not None:
        if derivation.kind == "reference_business_distance" and field != "distance_km":
            raise ValueError(
                "referenced distance can only derive a distance constraint"
            )
        if derivation.kind == "budget_to_price_level" and field != "price_level":
            raise ValueError("budget mapping can only derive a price-level constraint")


class RequirementBasis(StrictModel):
    """记录一条要求为什么存在，不保存大模型自报的置信度。"""

    source: SourceKind
    text: str = Field(min_length=1, max_length=500)
    turn_index: int | None = Field(default=None, ge=1)
    preference_strength: PreferenceStrength | None = None
    profile_score: float | None = Field(default=None, ge=-1, le=1)
    profile_evidence_count: int | None = Field(default=None, ge=1)
    profile_last_confirmed: datetime | None = None

    @model_validator(mode="after")
    def validate_source_metadata(self) -> Self:
        if self.source in {"current_query", "session"} and self.turn_index is None:
            raise ValueError("query and session sources require turn_index")
        if self.source == "user_profile":
            if (
                self.profile_evidence_count is None
                or self.profile_last_confirmed is None
            ):
                raise ValueError(
                    "user-profile sources require count and last-confirmed time"
                )
        elif (
            self.profile_score is not None
            or self.profile_evidence_count is not None
            or self.profile_last_confirmed is not None
        ):
            raise ValueError(
                "only user-profile sources may carry profile evidence metadata"
            )
        return self


class ConstraintDerivation(StrictModel):
    """A real value derived from conversation context instead of guessed by a model."""

    kind: DerivationKind
    reference_business_id: str | None = Field(default=None, min_length=1)
    source_value: str | int | float = Field(union_mode="left_to_right")
    source_unit: str = Field(min_length=1, max_length=50)
    mapping_policy: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_derivation(self) -> Self:
        if self.kind == "reference_business_distance":
            if self.reference_business_id is None:
                raise ValueError("distance derivation requires a referenced business")
            if not _is_number(self.source_value) or self.source_unit != "kilometer":
                raise ValueError("referenced distance must be a kilometer number")
            if self.mapping_policy is not None:
                raise ValueError("distance derivation does not use a mapping policy")
        else:
            if self.reference_business_id is not None:
                raise ValueError("budget mapping cannot reference a business")
            if not _is_number(self.source_value) or self.mapping_policy is None:
                raise ValueError("budget mapping requires a number and mapping policy")
        return self


class HardConstraint(StrictModel):
    """所有候选商家都必须满足的结构化条件。"""

    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,100}$")
    field: HardConstraintField
    operator: HardOperator
    value: RequirementValue
    unit: RequirementUnit
    merchant_feature: MerchantFeature
    unknown_action: Literal["exclude"] = "exclude"
    controlling_source: Literal["current_query", "session"]
    sources: list[RequirementBasis] = Field(min_length=1, max_length=20)
    derivation: ConstraintDerivation | None = None

    @model_validator(mode="after")
    def validate_constraint(self) -> Self:
        """保证硬条件来自对话，并且能用真实商家字段直接检查。"""

        _validate_sources(self.sources, self.controlling_source)
        if any(
            item.source not in {"current_query", "session"} for item in self.sources
        ):
            raise ValueError("scene and user profile cannot create hard constraints")
        _validate_structured_constraint(
            field=self.field,
            operator=self.operator,
            value=self.value,
            unit=self.unit,
            merchant_feature=self.merchant_feature,
            derivation=self.derivation,
        )
        return self


class DefaultConstraint(StrictModel):
    """第一次推荐时参与过滤、但允许大模型根据新问题修改的默认条件。"""

    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,100}$")
    field: HardConstraintField
    operator: HardOperator
    value: RequirementValue
    unit: RequirementUnit
    merchant_feature: MerchantFeature
    unknown_action: Literal["exclude"] = "exclude"
    controlling_source: SourceKind
    sources: list[RequirementBasis] = Field(min_length=1, max_length=20)
    derivation: ConstraintDerivation | None = None

    @model_validator(mode="after")
    def validate_constraint(self) -> Self:
        """检查默认值的来源和对应商家字段，防止默认值凭空失去依据。"""

        _validate_sources(self.sources, self.controlling_source)
        _validate_structured_constraint(
            field=self.field,
            operator=self.operator,
            value=self.value,
            unit=self.unit,
            merchant_feature=self.merchant_feature,
            derivation=self.derivation,
        )
        return self


class SoftPreference(StrictModel):
    """用于分层排序的一条偏好，包含偏好强度和明确先后顺序。"""

    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,100}$")
    field: RequirementField
    direction: PreferenceDirection
    target_value: RequirementValue | None = None
    preference_strength: PreferenceStrength
    priority: int = Field(ge=1, le=100)
    merchant_feature: MerchantFeature
    controlling_source: SourceKind
    sources: list[RequirementBasis] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_preference(self) -> Self:
        _validate_sources(self.sources, self.controlling_source)
        if self.merchant_feature != _FIELD_FEATURE[self.field]:
            raise ValueError("merchant_feature does not match the preference field")
        if any(item.preference_strength is None for item in self.sources):
            raise ValueError(
                "every soft-preference source requires preference_strength"
            )
        controlling_preference_strength = max(
            item.preference_strength
            for item in self.sources
            if item.source == self.controlling_source
            and item.preference_strength is not None
        )
        if self.preference_strength != controlling_preference_strength:
            raise ValueError(
                "effective preference_strength must come from the controlling source"
            )

        needs_target = self.direction in {"closer_to", "match", "avoid"}
        if needs_target != (self.target_value is not None):
            raise ValueError(
                "only closer_to, match, and avoid preferences use a target"
            )
        if self.field == "category":
            if self.direction not in {"match", "avoid"}:
                raise ValueError("category preferences match or avoid values")
            if (
                not isinstance(self.target_value, list)
                or not self.target_value
                or any(
                    not item or item != item.strip()
                    for item in self.target_value
                )
            ):
                raise ValueError(
                    "category preferences require a nonempty category list"
                )
        if self.field in _BOOLEAN_FIELDS and (
            self.direction not in {"match", "avoid"}
            or not isinstance(self.target_value, bool)
        ):
            raise ValueError(
                "boolean preferences match or avoid a true/false value"
            )
        if self.field == "distance_km" and self.direction != "lower":
            raise ValueError("distance preference must rank nearer merchants first")
        if self.field in ASPECT_FIELDS and self.direction not in {"higher", "lower"}:
            raise ValueError("aspect preferences rank the aspect higher or lower")
        if self.field in {"rating", "review_count"} and self.direction not in {
            "higher",
            "lower",
        }:
            raise ValueError("rating and review-count preferences rank higher or lower")
        if (
            self.field == "price_level"
            and self.direction == "closer_to"
            and (
                not isinstance(self.target_value, int)
                or not 1 <= self.target_value <= 4
            )
        ):
            raise ValueError("preferred price level must be an integer from 1 to 4")
        return self


class PreferenceMemoryItem(StrictModel):
    """完整保存一条来源偏好及其本轮裁决结果，供后续轮次恢复。"""

    candidate_id: str = Field(min_length=1, max_length=200)
    source: SourceKind
    preference: SoftPreference
    status: PreferenceMemoryStatus
    reason: str = Field(min_length=1, max_length=300)
    controller_candidate_id: str | None = Field(default=None, min_length=1)
    hard_constraint_keys: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_memory_item(self) -> Self:
        """记忆项必须保持单一来源，非生效项必须指出控制方。"""

        if self.preference.controlling_source != self.source:
            raise ValueError("memory preference must be controlled by its source")
        if any(item.source != self.source for item in self.preference.sources):
            raise ValueError("memory preference must contain one atomic source")
        if self.status == "active":
            if self.controller_candidate_id is not None or self.hard_constraint_keys:
                raise ValueError("active memory item has no external controller")
        elif self.hard_constraint_keys:
            if self.controller_candidate_id is not None:
                raise ValueError("hard-controlled memory item has no soft controller")
        elif self.controller_candidate_id is None:
            raise ValueError("inactive memory item requires a controller candidate")
        return self


class SceneBaseline(StrictModel):
    """一个场景的完整初始值，只包含可修改默认条件和软偏好。"""

    scene: BaselineSceneKind
    scene_label: str = Field(min_length=1, max_length=50)
    default_constraints: list[DefaultConstraint] = Field(
        default_factory=list,
        max_length=10,
    )
    soft_preferences: list[SoftPreference] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_baseline(self) -> Self:
        """保证场景基准不会混入用户硬条件或强度为100的假装明确要求。"""

        requirements = [*self.default_constraints, *self.soft_preferences]
        keys = [item.key for item in requirements]
        if len(keys) != len(set(keys)):
            raise ValueError("scene baseline requirement keys must be unique")
        priorities = sorted(item.priority for item in self.soft_preferences)
        if priorities != list(range(1, len(priorities) + 1)):
            raise ValueError("scene preference priorities must be contiguous from 1")
        if any(item.controlling_source != "scene" for item in requirements):
            raise ValueError("scene baseline requirements must be controlled by scene")
        if any(
            source.source != "scene" for item in requirements for source in item.sources
        ):
            raise ValueError("scene baseline requirements may only use scene sources")
        if any(item.preference_strength == 100 for item in self.soft_preferences):
            raise ValueError("scene baseline cannot use explicit-user strength 100")
        return self


class OpenRequirement(StrictModel):
    """保留尚未结构化的用户原话，后续由大模型选择是否调用工具处理。"""

    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,100}$")
    text: str = Field(min_length=1, max_length=500)
    behavior: OpenRequirementBehavior
    priority: int | None = Field(default=None, ge=1, le=100)
    controlling_source: Literal["current_query", "session"]
    sources: list[RequirementBasis] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_open_requirement(self) -> Self:
        _validate_sources(self.sources, self.controlling_source)
        if any(
            item.source not in {"current_query", "session"} for item in self.sources
        ):
            raise ValueError("open requirements must come from explicit conversation")
        if (self.behavior == "must_have") == (self.priority is not None):
            raise ValueError(
                "must-have items have no rank; prefer/avoid items require one"
            )
        return self


class SceneSelection(StrictModel):
    kind: SceneKind
    custom_label: str | None = Field(default=None, min_length=1, max_length=100)
    basis: RequirementBasis

    @model_validator(mode="after")
    def validate_scene(self) -> Self:
        if (self.kind == "custom") != (self.custom_label is not None):
            raise ValueError("only a custom scene uses custom_label")
        if self.basis.source not in {"current_query", "session"}:
            raise ValueError("the active scene must come from conversation")
        return self


class GeoPoint(StrictModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    def distance_km_to(self, other: GeoPoint) -> float:
        """根据两组经纬度计算地表距离，供过滤和排序共同使用。"""

        earth_radius_km = 6371.0088
        latitude_delta = radians(other.latitude - self.latitude)
        longitude_delta = radians(other.longitude - self.longitude)
        start_latitude = radians(self.latitude)
        end_latitude = radians(other.latitude)
        haversine_value = (
            sin(latitude_delta / 2) ** 2
            + cos(start_latitude)
            * cos(end_latitude)
            * sin(longitude_delta / 2) ** 2
        )
        central_angle = 2 * asin(min(1.0, sqrt(haversine_value)))
        return earth_radius_km * central_angle


class SearchCenter(StrictModel):
    """本轮所有距离条件共同使用的计算起点。"""

    kind: SearchCenterKind
    label: str = Field(min_length=1, max_length=200)
    location: GeoPoint
    reference_business_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        """只有以某家商家为中心时才保存对应的商家编号。"""

        if (self.kind == "business") != (self.reference_business_id is not None):
            raise ValueError(
                "only a business search center uses reference_business_id"
            )
        return self


class BusinessReference(StrictModel):
    """Facts retained for expressions such as 'the third one is too far'."""

    presented_turn_index: int = Field(ge=1)
    position: int = Field(ge=1, le=100)
    business_id: str = Field(min_length=1, max_length=200)
    business_name: str = Field(min_length=1, max_length=500)
    location: GeoPoint | None = None
    distance_km: float | None = Field(default=None, ge=0)
    price_level: int | None = Field(default=None, ge=1, le=4)
    categories: list[str] = Field(default_factory=list, max_length=100)
    aspect_scores: dict[AspectField, float] = Field(default_factory=dict)

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, values: list[str]) -> list[str]:
        if any(not item or item != item.strip() for item in values):
            raise ValueError("business categories must be nonempty and trimmed")
        if len(values) != len(set(values)):
            raise ValueError("business categories must be unique")
        return values

    @field_validator("aspect_scores")
    @classmethod
    def validate_aspect_scores(
        cls,
        values: dict[AspectField, float],
    ) -> dict[AspectField, float]:
        if any(not 0 <= score <= 100 for score in values.values()):
            raise ValueError("business aspect scores must be between 0 and 100")
        return values


class UnifiedRecommendationState(StrictModel):
    """一轮对话处理完成后，用户当前所有仍然生效的要求。"""

    schema_version: Literal["2.1.0"] = "2.1.0"
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    turn_index: int = Field(ge=1)
    latest_query_text: str = Field(min_length=1, max_length=2000)
    user_location: GeoPoint | None = None
    search_center: SearchCenter | None = None
    scene: SceneSelection | None = None
    hard_constraints: list[HardConstraint] = Field(default_factory=list, max_length=50)
    default_constraints: list[DefaultConstraint] = Field(
        default_factory=list,
        max_length=20,
    )
    soft_preferences: list[SoftPreference] = Field(default_factory=list, max_length=50)
    preference_memory: list[PreferenceMemoryItem] = Field(
        default_factory=list,
        max_length=200,
    )
    open_requirements: list[OpenRequirement] = Field(
        default_factory=list, max_length=30
    )
    referenced_businesses: list[BusinessReference] = Field(
        default_factory=list,
        max_length=200,
    )

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.search_center is None and self.user_location is not None:
            self.search_center = SearchCenter(
                kind="current_location",
                label="当前位置",
                location=self.user_location.model_copy(deep=True),
            )
        elif (
            self.search_center is not None
            and self.search_center.kind == "current_location"
        ):
            if self.user_location is None:
                raise ValueError(
                    "current-location search center requires user_location"
                )
            if self.search_center.location != self.user_location:
                raise ValueError(
                    "current-location search center must equal user_location"
                )

        requirements = [
            *self.hard_constraints,
            *self.default_constraints,
            *self.soft_preferences,
            *self.open_requirements,
        ]
        distance_requirements = [
            item
            for item in [
                *self.hard_constraints,
                *self.default_constraints,
                *self.soft_preferences,
            ]
            if item.field == "distance_km"
        ]
        if distance_requirements and self.search_center is None:
            raise ValueError("distance requirements need a search center")
        keys = [item.key for item in requirements]
        if len(keys) != len(set(keys)):
            raise ValueError("active requirement keys must be globally unique")

        priorities = sorted(
            [item.priority for item in self.soft_preferences]
            + [
                item.priority
                for item in self.open_requirements
                if item.priority is not None
            ]
        )
        if priorities != list(range(1, len(priorities) + 1)):
            raise ValueError("ranking priorities must be unique and contiguous from 1")

        reference_keys = [
            (item.presented_turn_index, item.position)
            for item in self.referenced_businesses
        ]
        if len(reference_keys) != len(set(reference_keys)):
            raise ValueError(
                "each presented turn and position may identify only one business"
            )

        memory_ids = [item.candidate_id for item in self.preference_memory]
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("preference memory candidate IDs must be unique")
        known_memory_ids = set(memory_ids)
        if any(
            item.controller_candidate_id not in known_memory_ids
            for item in self.preference_memory
            if item.controller_candidate_id is not None
        ):
            raise ValueError("memory controller must identify a stored candidate")

        all_bases = [basis for item in requirements for basis in item.sources]
        all_bases.extend(
            basis
            for memory in self.preference_memory
            for basis in memory.preference.sources
        )
        if self.scene is not None:
            all_bases.append(self.scene.basis)
        for basis in all_bases:
            if basis.source == "current_query" and basis.turn_index != self.turn_index:
                raise ValueError("current-query sources must use the current turn")
            if basis.source == "session" and (
                basis.turn_index is None or basis.turn_index >= self.turn_index
            ):
                raise ValueError("session sources must come from an earlier turn")
        return self


