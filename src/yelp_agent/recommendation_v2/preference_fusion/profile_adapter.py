"""把旧的长期画像逐条转换成统一软偏好，不在这里解决冲突。"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal, NamedTuple, Self, cast

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.profiles.schema import PreferenceSignal, UserProfileV1
from yelp_agent.recommendation_v2.schema import (
    AspectField,
    MerchantFeature,
    PreferenceDirection,
    PreferenceStrength,
    RequirementBasis,
    RequirementField,
    RequirementValue,
    SoftPreference,
)


class AspectDirectionPolicy(NamedTuple):
    """同一评论特征在正向记录和负向记录下分别代表的排序方向。"""

    positive: Literal["higher", "lower"]
    negative: Literal["higher", "lower"]


# 这张表集中说明全部现有评论特征的真实排序含义，禁止在转换循环里散落特例。
ASPECT_DIRECTION_POLICY: dict[AspectField, AspectDirectionPolicy] = {
    "food_quality": AspectDirectionPolicy("higher", "higher"),
    "service": AspectDirectionPolicy("higher", "higher"),
    "price_value": AspectDirectionPolicy("higher", "higher"),
    "quiet_environment": AspectDirectionPolicy("higher", "higher"),
    "crowded": AspectDirectionPolicy("lower", "lower"),
    "queue_time": AspectDirectionPolicy("lower", "lower"),
    "portion_size": AspectDirectionPolicy("higher", "higher"),
    "parking": AspectDirectionPolicy("higher", "higher"),
    "pet_friendly": AspectDirectionPolicy("higher", "higher"),
    "family_friendly": AspectDirectionPolicy("higher", "higher"),
    "date_suitable": AspectDirectionPolicy("higher", "higher"),
    "group_suitable": AspectDirectionPolicy("higher", "higher"),
    # 辣度是真正双向的量：正向记录偏辣，负向记录偏不辣。
    "spiciness": AspectDirectionPolicy("higher", "lower"),
    "cleanliness": AspectDirectionPolicy("higher", "higher"),
}


class IgnoredProfileSignal(StrictModel):
    """记录目前无法映射的画像信号，避免转换时静默丢失。"""

    kind: str = Field(min_length=1)
    value: str = Field(min_length=1)
    reason: Literal["unsupported_aspect", "unresolved_location"]
    profile_score: float = Field(ge=-1, le=1)
    profile_evidence_count: int = Field(ge=1)
    profile_last_confirmed: datetime


class ProfilePreferenceSet(StrictModel):
    """长期画像转换结果；这里只允许出现用户画像来源的软偏好。"""

    schema_version: Literal["2.0.0"] = "2.0.0"
    profile_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_id: str = Field(min_length=1)
    soft_preferences: list[SoftPreference] = Field(
        default_factory=list,
        max_length=100,
    )
    ignored_signals: list[IgnoredProfileSignal] = Field(
        default_factory=list,
        max_length=50,
    )

    @model_validator(mode="after")
    def validate_preferences(self) -> Self:
        """检查来源、唯一编号和画像内部顺序。"""

        if any(
            item.controlling_source != "user_profile"
            for item in self.soft_preferences
        ):
            raise ValueError("profile adapter may only return profile preferences")
        if any(
            basis.source != "user_profile"
            for item in self.soft_preferences
            for basis in item.sources
        ):
            raise ValueError("profile preferences may only use profile evidence")
        keys = [item.key for item in self.soft_preferences]
        if len(keys) != len(set(keys)):
            raise ValueError("profile preference IDs must be unique")
        priorities = sorted(item.priority for item in self.soft_preferences)
        if priorities != list(range(1, len(priorities) + 1)):
            raise ValueError("profile priorities must be contiguous from 1")
        return self


def _strength(score: float) -> PreferenceStrength:
    """只按原始画像分数分成弱、中、强三档，不做时间衰减。"""

    magnitude = abs(score)
    if magnitude > 2 / 3:
        return 75
    if magnitude > 1 / 3:
        return 50
    return 25


def _merchant_feature(field: RequirementField) -> MerchantFeature:
    """把统一字段对应到后续真正可读取的商家特征。"""

    aliases: dict[str, str] = {
        "category": "categories",
        "distance_km": "coordinates",
    }
    return cast(MerchantFeature, aliases.get(field, field))


def _stable_suffix(
    profile_id: str,
    signal: PreferenceSignal,
    origin: str,
    direction: PreferenceDirection,
) -> str:
    """生成稳定且与冲突判断无关的唯一编号后缀。"""

    raw = f"{profile_id}\0{origin}\0{signal.kind}\0{signal.value}\0{direction}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _preference(
    *,
    profile: UserProfileV1,
    signal: PreferenceSignal,
    origin: str,
    field: RequirementField,
    direction: PreferenceDirection,
    target_value: RequirementValue | None,
    explanation: str,
) -> SoftPreference:
    """把一条原始画像信号完整翻译成一条独立软偏好。"""

    strength = _strength(signal.score)
    suffix = _stable_suffix(
        profile.profile_id,
        signal,
        origin,
        direction,
    )
    return SoftPreference(
        key=f"profile.{field}.{suffix}",
        field=field,
        direction=direction,
        target_value=target_value,
        preference_strength=strength,
        priority=1,
        merchant_feature=_merchant_feature(field),
        controlling_source="user_profile",
        sources=[
            RequirementBasis(
                source="user_profile",
                text=explanation,
                preference_strength=strength,
                profile_score=signal.score,
                profile_evidence_count=signal.evidence_count,
                profile_last_confirmed=signal.last_confirmed,
            )
        ],
    )


def _aspect_preference(
    *,
    profile: UserProfileV1,
    signal: PreferenceSignal,
    negative_history: bool,
) -> SoftPreference | IgnoredProfileSignal:
    """按照集中规则翻译评论特征，不根据所在列表机械决定方向。"""

    if signal.value not in ASPECT_DIRECTION_POLICY:
        return IgnoredProfileSignal(
            kind=signal.kind,
            value=signal.value,
            reason="unsupported_aspect",
            profile_score=signal.score,
            profile_evidence_count=signal.evidence_count,
            profile_last_confirmed=signal.last_confirmed,
        )
    field = cast(AspectField, signal.value)
    policy = ASPECT_DIRECTION_POLICY[field]
    direction = policy.negative if negative_history else policy.positive
    history_label = "负向评论记录" if negative_history else "正向评论记录"
    return _preference(
        profile=profile,
        signal=signal,
        origin=("aspect_dislikes" if negative_history else "aspect_preferences"),
        field=field,
        direction=direction,
        target_value=None,
        explanation=f"长期画像：{history_label}反复涉及“{field}”",
    )


def _rank(preferences: list[SoftPreference]) -> list[SoftPreference]:
    """只在画像内部按强度、证据次数、最近确认时间和稳定编号排序。"""

    ordered = sorted(preferences, key=lambda item: item.key)
    ordered.sort(
        key=lambda item: item.sources[0].profile_last_confirmed,
        reverse=True,
    )
    ordered.sort(
        key=lambda item: item.sources[0].profile_evidence_count or 0,
        reverse=True,
    )
    ordered.sort(key=lambda item: item.preference_strength, reverse=True)
    return [
        item.model_copy(update={"priority": priority}, deep=True)
        for priority, item in enumerate(ordered, start=1)
    ]


def adapt_user_profile(profile: UserProfileV1) -> ProfilePreferenceSet:
    """逐条转换长期画像；不聚合目标、不产生硬条件、不裁决冲突。"""

    preferences: list[SoftPreference] = []
    ignored: list[IgnoredProfileSignal] = []

    for signal in profile.category_preferences:
        preferences.append(
            _preference(
                profile=profile,
                signal=signal,
                origin="category_preferences",
                field="category",
                direction="match",
                target_value=[signal.value],
                explanation=f"长期画像：更常选择“{signal.value}”",
            )
        )
    for signal in profile.category_dislikes:
        preferences.append(
            _preference(
                profile=profile,
                signal=signal,
                origin="category_dislikes",
                field="category",
                direction="avoid",
                target_value=[signal.value],
                explanation=f"长期画像：对“{signal.value}”评价较差",
            )
        )
    for negative_history, signals in (
        (False, profile.aspect_preferences),
        (True, profile.aspect_dislikes),
    ):
        for signal in signals:
            converted = _aspect_preference(
                profile=profile,
                signal=signal,
                negative_history=negative_history,
            )
            if isinstance(converted, IgnoredProfileSignal):
                ignored.append(converted)
            else:
                preferences.append(converted)
    if profile.price_preference is not None:
        signal = profile.price_preference
        preferences.append(
            _preference(
                profile=profile,
                signal=signal,
                origin="price_preference",
                field="price_level",
                direction="closer_to",
                target_value=int(signal.value),
                explanation=f"长期画像：更常选择{signal.value}档价格",
            )
        )
    for signal in profile.frequent_areas:
        # 区域字符串不能直接参与距离排序；这里只完整保留原信号，之后由
        # 大模型结合可用工具决定如何解析，不能把邮编或地名假装成商家字段。
        ignored.append(
            IgnoredProfileSignal(
                kind=signal.kind,
                value=signal.value,
                reason="unresolved_location",
                profile_score=signal.score,
                profile_evidence_count=signal.evidence_count,
                profile_last_confirmed=signal.last_confirmed,
            )
        )

    return ProfilePreferenceSet(
        profile_id=profile.profile_id,
        user_id=profile.user_id,
        soft_preferences=_rank(preferences),
        ignored_signals=ignored,
    )
