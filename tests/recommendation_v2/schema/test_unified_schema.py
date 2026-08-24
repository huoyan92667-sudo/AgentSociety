from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from yelp_agent.recommendation_v2 import (
    BusinessReference,
    GeoPoint,
    HardConstraint,
    OpenRequirement,
    RequirementBasis,
    SceneSelection,
    SearchCenter,
    SoftPreference,
    UnifiedRecommendationState,
)


def current_basis(
    text: str,
    *,
    preference_strength: int | None = None,
) -> RequirementBasis:
    """构造来自当前问题的测试依据。"""

    return RequirementBasis(
        source="current_query",
        text=text,
        turn_index=2,
        preference_strength=preference_strength,
    )


def session_basis(
    text: str,
    *,
    preference_strength: int | None = None,
) -> RequirementBasis:
    """构造来自上一轮对话的测试依据。"""

    return RequirementBasis(
        source="session",
        text=text,
        turn_index=1,
        preference_strength=preference_strength,
    )


def test_complete_state_round_trips_without_rebuilding_old_requirements() -> None:
    category = HardConstraint(
        key="category.include",
        field="category",
        operator="any_of",
        value=["日式餐厅", "寿司", "拉面"],
        unit="category",
        merchant_feature="categories",
        controlling_source="session",
        sources=[session_basis("想吃日料")],
    )
    quiet = SoftPreference(
        key="quiet.prefer",
        field="quiet_environment",
        direction="higher",
        preference_strength=100,
        priority=1,
        merchant_feature="quiet_environment",
        controlling_source="current_query",
        sources=[
            current_basis("安静最重要", preference_strength=100),
            RequirementBasis(
                source="scene",
                text="约会场景补充安静偏好",
                preference_strength=50,
            ),
            RequirementBasis(
                source="user_profile",
                text="长期评论中持续偏好安静环境",
                preference_strength=75,
                profile_evidence_count=8,
                profile_last_confirmed=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            ),
        ],
    )
    distance = SoftPreference(
        key="distance.near",
        field="distance_km",
        direction="lower",
        preference_strength=75,
        priority=2,
        merchant_feature="coordinates",
        controlling_source="current_query",
        sources=[current_basis("距离其次", preference_strength=75)],
    )
    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=2,
        turn_index=2,
        latest_query_text="安静最重要，距离其次",
        user_location=GeoPoint(latitude=39.9526, longitude=-75.1652),
        scene=SceneSelection(
            kind="date",
            basis=session_basis("今晚和对象吃饭"),
        ),
        hard_constraints=[category],
        soft_preferences=[quiet, distance],
        open_requirements=[
            OpenRequirement(
                key="private_room.required",
                text="必须有包间",
                behavior="must_have",
                controlling_source="session",
                sources=[session_basis("必须有包间")],
            )
        ],
        referenced_businesses=[
            BusinessReference(
                presented_turn_index=1,
                position=3,
                business_id="business-3",
                business_name="第三家餐厅",
                distance_km=3.7,
                price_level=2,
                categories=["日式餐厅", "寿司"],
                aspect_scores={"quiet_environment": 72.0},
            )
        ],
    )

    restored = UnifiedRecommendationState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert restored.hard_constraints[0].value == ["日式餐厅", "寿司", "拉面"]
    assert restored.soft_preferences[0].controlling_source == "current_query"


def test_explicit_three_kilometers_uses_the_same_field_without_derivation() -> None:
    constraint = HardConstraint(
        key="distance.max",
        field="distance_km",
        operator="less_than_or_equal",
        value=3,
        unit="kilometer",
        merchant_feature="coordinates",
        controlling_source="current_query",
        sources=[current_basis("最大距离3公里以内")],
    )

    assert constraint.field == "distance_km"
    assert constraint.derivation is None


def test_scene_and_profile_cannot_create_hard_constraints() -> None:
    with pytest.raises(ValidationError, match="cannot create hard constraints"):
        HardConstraint(
            key="category.include",
            field="category",
            operator="any_of",
            value=["日式餐厅"],
            unit="category",
            merchant_feature="categories",
            controlling_source="current_query",
            sources=[
                current_basis("必须吃日料"),
                RequirementBasis(
                    source="scene",
                    text="约会场景偏好日料",
                ),
            ],
        )


def test_review_aspect_cannot_be_used_as_a_hard_constraint() -> None:
    """安静等评论归纳特征在第一批只能进入软偏好。"""

    with pytest.raises(ValidationError):
        HardConstraint(
            key="quiet.minimum",
            field="quiet_environment",
            operator="greater_than_or_equal",
            value=65,
            unit="match_score",
            merchant_feature="quiet_environment",
            controlling_source="current_query",
            sources=[current_basis("必须安静")],
        )


def test_current_query_must_control_a_supported_profile_preference() -> None:
    with pytest.raises(ValidationError, match="highest-priority source"):
        SoftPreference(
            key="quiet.prefer",
            field="quiet_environment",
            direction="higher",
            preference_strength=75,
            priority=1,
            merchant_feature="quiet_environment",
            controlling_source="user_profile",
            sources=[
                current_basis("安静最重要", preference_strength=100),
                RequirementBasis(
                    source="user_profile",
                    text="长期偏好安静",
                    preference_strength=75,
                    profile_evidence_count=5,
                    profile_last_confirmed=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
                ),
            ],
        )


def test_ranking_priorities_must_be_unique_and_contiguous() -> None:
    first = SoftPreference(
        key="quiet.prefer",
        field="quiet_environment",
        direction="higher",
        preference_strength=100,
        priority=1,
        merchant_feature="quiet_environment",
        controlling_source="current_query",
        sources=[current_basis("安静最重要", preference_strength=100)],
    )
    third = SoftPreference(
        key="distance.near",
        field="distance_km",
        direction="lower",
        preference_strength=75,
        priority=3,
        merchant_feature="coordinates",
        controlling_source="current_query",
        sources=[current_basis("距离其次", preference_strength=75)],
    )

    with pytest.raises(ValidationError, match="contiguous"):
        UnifiedRecommendationState(
            user_id="user-1",
            session_id="session-1",
            revision=1,
            turn_index=2,
            latest_query_text="安静最重要，距离其次",
            user_location=GeoPoint(latitude=39.9526, longitude=-75.1652),
            soft_preferences=[first, third],
        )


def test_category_filter_must_use_real_merchant_categories() -> None:
    with pytest.raises(ValidationError, match="merchant_feature"):
        HardConstraint(
            key="category.include",
            field="category",
            operator="any_of",
            value=["日式餐厅"],
            unit="category",
            merchant_feature="coordinates",
            controlling_source="current_query",
            sources=[current_basis("想吃日料")],
        )


def test_current_location_becomes_the_default_search_center() -> None:
    """用户没有指定区域时，距离计算默认从当前定位开始。"""

    location = GeoPoint(latitude=39.9526, longitude=-75.1652)
    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=2,
        latest_query_text="附近找一家",
        user_location=location,
    )

    assert state.search_center == SearchCenter(
        kind="current_location",
        label="当前位置",
        location=location,
    )


def test_named_place_can_replace_the_default_search_center() -> None:
    """用户明确地点时，使用地点坐标而不是当前位置。"""

    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=2,
        latest_query_text="找唐人街附近的川菜",
        user_location=GeoPoint(latitude=39.9526, longitude=-75.1652),
        search_center=SearchCenter(
            kind="named_place",
            label="费城唐人街",
            location=GeoPoint(latitude=39.9559, longitude=-75.1566),
        ),
    )

    assert state.search_center is not None
    assert state.search_center.kind == "named_place"
    assert state.search_center.label == "费城唐人街"


def test_geo_points_calculate_distance_in_kilometers() -> None:
    """同一套经纬度距离供后续硬过滤和软排序共同使用。"""

    start = GeoPoint(latitude=39.9526, longitude=-75.1652)
    end = GeoPoint(latitude=39.9559, longitude=-75.1566)

    assert start.distance_km_to(end) == pytest.approx(0.82, abs=0.05)
    assert start.distance_km_to(start) == pytest.approx(0.0)


def test_distance_requirement_needs_a_search_center() -> None:
    distance = SoftPreference(
        key="distance.near",
        field="distance_km",
        direction="lower",
        preference_strength=75,
        priority=1,
        merchant_feature="coordinates",
        controlling_source="current_query",
        sources=[current_basis("近一点", preference_strength=75)],
    )

    with pytest.raises(ValidationError, match="search center"):
        UnifiedRecommendationState(
            user_id="user-1",
            session_id="session-1",
            revision=1,
            turn_index=2,
            latest_query_text="近一点",
            soft_preferences=[distance],
        )


@pytest.mark.parametrize(
    ("field", "value", "unit", "merchant_feature"),
    [
        ("rating", 4.0, "rating", "rating"),
        ("review_count", 100, "count", "review_count"),
        ("accepts_reservations", True, "boolean", "accepts_reservations"),
        ("delivery", True, "boolean", "delivery"),
        ("takeout", True, "boolean", "takeout"),
        ("outdoor_seating", True, "boolean", "outdoor_seating"),
        ("good_for_kids", True, "boolean", "good_for_kids"),
        ("good_for_groups", True, "boolean", "good_for_groups"),
        (
            "wheelchair_accessible",
            True,
            "boolean",
            "wheelchair_accessible",
        ),
        ("dogs_allowed", True, "boolean", "dogs_allowed"),
        ("parking_available", True, "boolean", "parking_available"),
    ],
)
def test_real_structured_merchant_fields_can_be_hard_constraints(
    field: str,
    value: float | bool,
    unit: str,
    merchant_feature: str,
) -> None:
    operator = (
        "greater_than_or_equal"
        if field in {"rating", "review_count"}
        else "equals"
    )

    constraint = HardConstraint(
        key=f"{field}.required",
        field=field,
        operator=operator,
        value=value,
        unit=unit,
        merchant_feature=merchant_feature,
        controlling_source="current_query",
        sources=[current_basis("用户明确要求")],
    )

    assert constraint.unknown_action == "exclude"


def test_real_structured_merchant_fields_can_also_be_soft_preferences() -> None:
    rating = SoftPreference(
        key="rating.prefer",
        field="rating",
        direction="higher",
        preference_strength=50,
        priority=1,
        merchant_feature="rating",
        controlling_source="current_query",
        sources=[current_basis("评分高一点", preference_strength=50)],
    )
    reservations = SoftPreference(
        key="accepts_reservations.prefer",
        field="accepts_reservations",
        direction="match",
        target_value=True,
        preference_strength=25,
        priority=2,
        merchant_feature="accepts_reservations",
        controlling_source="current_query",
        sources=[current_basis("最好能预约", preference_strength=25)],
    )

    assert rating.direction == "higher"
    assert reservations.target_value is True


def test_area_is_no_longer_a_requirement_field() -> None:
    with pytest.raises(ValidationError):
        HardConstraint(
            key="area.include",
            field="area",
            operator="any_of",
            value=["唐人街"],
            unit="area",
            merchant_feature="area",
            controlling_source="current_query",
            sources=[current_basis("必须在唐人街")],
        )
