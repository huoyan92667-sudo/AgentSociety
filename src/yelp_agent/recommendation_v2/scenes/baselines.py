"""六个固定场景的默认筛选范围和默认软偏好。"""

from __future__ import annotations

from typing import cast

from yelp_agent.recommendation_v2.schema import (
    BaselineSceneKind,
    DefaultConstraint,
    MerchantFeature,
    PreferenceDirection,
    PreferenceStrength,
    RequirementBasis,
    RequirementField,
    RequirementValue,
    SceneBaseline,
    SoftPreference,
)

SCENE_ORDER: tuple[BaselineSceneKind, ...] = (
    "casual",
    "date",
    "business",
    "friends",
    "family",
    "solo",
)


def _merchant_feature_for(field: RequirementField) -> MerchantFeature:
    """把统一偏好字段转换成实际用于比较的商家特征字段。"""

    special_features: dict[str, str] = {
        "category": "categories",
        "distance_km": "coordinates",
    }
    return cast(MerchantFeature, special_features.get(field, field))


def _scene_basis(
    scene_label: str,
    reason: str,
    *,
    preference_strength: PreferenceStrength | None = None,
) -> RequirementBasis:
    """生成一条场景来源说明，让默认值以后能够被识别和替换。"""

    return RequirementBasis(
        source="scene",
        text=f"{scene_label}场景默认：{reason}",
        preference_strength=preference_strength,
    )


def _default_distance(scene_label: str, distance_km: float) -> DefaultConstraint:
    """生成场景的初始搜索距离；它不是用户明确说出的硬条件。"""

    return DefaultConstraint(
        key="distance.max",
        field="distance_km",
        operator="less_than_or_equal",
        value=distance_km,
        unit="kilometer",
        merchant_feature="coordinates",
        controlling_source="scene",
        sources=[
            _scene_basis(
                scene_label,
                f"第一次推荐先搜索{distance_km:g}公里以内",
            )
        ],
    )


def _preference(
    *,
    scene_label: str,
    key: str,
    field: RequirementField,
    direction: PreferenceDirection,
    preference_strength: PreferenceStrength,
    priority: int,
    reason: str,
    target_value: RequirementValue | None = None,
) -> SoftPreference:
    """生成一条场景软偏好，顺序只表示该场景内部的先后关系。"""

    return SoftPreference(
        key=key,
        field=field,
        direction=direction,
        target_value=target_value,
        preference_strength=preference_strength,
        priority=priority,
        merchant_feature=_merchant_feature_for(field),
        controlling_source="scene",
        sources=[
            _scene_basis(
                scene_label,
                reason,
                preference_strength=preference_strength,
            )
        ],
    )


def _build_scene_baselines() -> dict[BaselineSceneKind, SceneBaseline]:
    """集中建立六个场景，避免场景默认值散落到查询和排序代码中。"""

    casual = "随便吃"
    date = "约会"
    business = "商务"
    friends = "朋友聚餐"
    family = "家庭聚餐"
    solo = "一个人吃"

    return {
        "casual": SceneBaseline(
            scene="casual",
            scene_label=casual,
            default_constraints=[_default_distance(casual, 5)],
            soft_preferences=[
                _preference(
                    scene_label=casual,
                    key="food_quality.prefer",
                    field="food_quality",
                    direction="higher",
                    preference_strength=75,
                    priority=1,
                    reason="优先保证食物质量",
                ),
                _preference(
                    scene_label=casual,
                    key="price_value.prefer",
                    field="price_value",
                    direction="higher",
                    preference_strength=50,
                    priority=2,
                    reason="通常会考虑性价比",
                ),
                _preference(
                    scene_label=casual,
                    key="distance.near",
                    field="distance_km",
                    direction="lower",
                    preference_strength=50,
                    priority=3,
                    reason="同等情况下优先更近的商家",
                ),
                _preference(
                    scene_label=casual,
                    key="price_level.prefer",
                    field="price_level",
                    direction="closer_to",
                    target_value=2,
                    preference_strength=50,
                    priority=4,
                    reason="价格默认更接近二档",
                ),
                _preference(
                    scene_label=casual,
                    key="queue_time.avoid",
                    field="queue_time",
                    direction="lower",
                    preference_strength=25,
                    priority=5,
                    reason="辅助考虑少排队",
                ),
            ],
        ),
        "date": SceneBaseline(
            scene="date",
            scene_label=date,
            default_constraints=[_default_distance(date, 8)],
            soft_preferences=[
                _preference(
                    scene_label=date,
                    key="quiet_environment.prefer",
                    field="quiet_environment",
                    direction="higher",
                    preference_strength=75,
                    priority=1,
                    reason="优先保证交流环境安静",
                ),
                _preference(
                    scene_label=date,
                    key="date_suitable.prefer",
                    field="date_suitable",
                    direction="higher",
                    preference_strength=75,
                    priority=2,
                    reason="优先适合约会的商家",
                ),
                _preference(
                    scene_label=date,
                    key="service.prefer",
                    field="service",
                    direction="higher",
                    preference_strength=50,
                    priority=3,
                    reason="通常会考虑服务体验",
                ),
                _preference(
                    scene_label=date,
                    key="price_level.prefer",
                    field="price_level",
                    direction="closer_to",
                    target_value=3,
                    preference_strength=50,
                    priority=4,
                    reason="价格默认更接近三档",
                ),
                _preference(
                    scene_label=date,
                    key="cleanliness.prefer",
                    field="cleanliness",
                    direction="higher",
                    preference_strength=25,
                    priority=5,
                    reason="辅助考虑环境干净",
                ),
            ],
        ),
        "business": SceneBaseline(
            scene="business",
            scene_label=business,
            default_constraints=[_default_distance(business, 6)],
            soft_preferences=[
                _preference(
                    scene_label=business,
                    key="quiet_environment.prefer",
                    field="quiet_environment",
                    direction="higher",
                    preference_strength=75,
                    priority=1,
                    reason="优先保证谈话环境安静",
                ),
                _preference(
                    scene_label=business,
                    key="service.prefer",
                    field="service",
                    direction="higher",
                    preference_strength=75,
                    priority=2,
                    reason="优先保证服务稳定",
                ),
                _preference(
                    scene_label=business,
                    key="group_suitable.prefer",
                    field="group_suitable",
                    direction="higher",
                    preference_strength=50,
                    priority=3,
                    reason="通常需要适合多人交谈",
                ),
                _preference(
                    scene_label=business,
                    key="cleanliness.prefer",
                    field="cleanliness",
                    direction="higher",
                    preference_strength=50,
                    priority=4,
                    reason="通常会考虑环境干净",
                ),
                _preference(
                    scene_label=business,
                    key="price_level.prefer",
                    field="price_level",
                    direction="closer_to",
                    target_value=3,
                    preference_strength=50,
                    priority=5,
                    reason="价格默认更接近三档",
                ),
            ],
        ),
        "friends": SceneBaseline(
            scene="friends",
            scene_label=friends,
            default_constraints=[_default_distance(friends, 8)],
            soft_preferences=[
                _preference(
                    scene_label=friends,
                    key="group_suitable.prefer",
                    field="group_suitable",
                    direction="higher",
                    preference_strength=75,
                    priority=1,
                    reason="优先适合多人聚餐",
                ),
                _preference(
                    scene_label=friends,
                    key="food_quality.prefer",
                    field="food_quality",
                    direction="higher",
                    preference_strength=75,
                    priority=2,
                    reason="优先保证食物质量",
                ),
                _preference(
                    scene_label=friends,
                    key="price_value.prefer",
                    field="price_value",
                    direction="higher",
                    preference_strength=50,
                    priority=3,
                    reason="通常会考虑多人消费的性价比",
                ),
                _preference(
                    scene_label=friends,
                    key="price_level.prefer",
                    field="price_level",
                    direction="closer_to",
                    target_value=2,
                    preference_strength=50,
                    priority=4,
                    reason="价格默认更接近二档",
                ),
                _preference(
                    scene_label=friends,
                    key="queue_time.avoid",
                    field="queue_time",
                    direction="lower",
                    preference_strength=25,
                    priority=5,
                    reason="辅助考虑少排队",
                ),
            ],
        ),
        "family": SceneBaseline(
            scene="family",
            scene_label=family,
            default_constraints=[_default_distance(family, 6)],
            soft_preferences=[
                _preference(
                    scene_label=family,
                    key="family_friendly.prefer",
                    field="family_friendly",
                    direction="higher",
                    preference_strength=75,
                    priority=1,
                    reason="优先适合家庭用餐",
                ),
                _preference(
                    scene_label=family,
                    key="cleanliness.prefer",
                    field="cleanliness",
                    direction="higher",
                    preference_strength=75,
                    priority=2,
                    reason="优先保证环境干净",
                ),
                _preference(
                    scene_label=family,
                    key="parking.prefer",
                    field="parking",
                    direction="higher",
                    preference_strength=50,
                    priority=3,
                    reason="通常会考虑停车方便",
                ),
                _preference(
                    scene_label=family,
                    key="price_level.prefer",
                    field="price_level",
                    direction="closer_to",
                    target_value=2,
                    preference_strength=50,
                    priority=4,
                    reason="价格默认更接近二档",
                ),
                _preference(
                    scene_label=family,
                    key="queue_time.avoid",
                    field="queue_time",
                    direction="lower",
                    preference_strength=25,
                    priority=5,
                    reason="辅助考虑少排队",
                ),
            ],
        ),
        "solo": SceneBaseline(
            scene="solo",
            scene_label=solo,
            default_constraints=[_default_distance(solo, 3)],
            soft_preferences=[
                _preference(
                    scene_label=solo,
                    key="distance.near",
                    field="distance_km",
                    direction="lower",
                    preference_strength=75,
                    priority=1,
                    reason="优先更近的商家",
                ),
                _preference(
                    scene_label=solo,
                    key="queue_time.avoid",
                    field="queue_time",
                    direction="lower",
                    preference_strength=50,
                    priority=2,
                    reason="通常会考虑少排队",
                ),
                _preference(
                    scene_label=solo,
                    key="price_value.prefer",
                    field="price_value",
                    direction="higher",
                    preference_strength=50,
                    priority=3,
                    reason="通常会考虑性价比",
                ),
                _preference(
                    scene_label=solo,
                    key="price_level.prefer",
                    field="price_level",
                    direction="closer_to",
                    target_value=2,
                    preference_strength=50,
                    priority=4,
                    reason="价格默认更接近二档",
                ),
                _preference(
                    scene_label=solo,
                    key="food_quality.prefer",
                    field="food_quality",
                    direction="higher",
                    preference_strength=25,
                    priority=5,
                    reason="辅助考虑食物质量",
                ),
            ],
        ),
    }


_SCENE_BASELINES = _build_scene_baselines()


def get_scene_baseline(scene: BaselineSceneKind) -> SceneBaseline:
    """读取一个场景基准，并返回独立副本，防止调用方改坏固定基准。"""

    return _SCENE_BASELINES[scene].model_copy(deep=True)
