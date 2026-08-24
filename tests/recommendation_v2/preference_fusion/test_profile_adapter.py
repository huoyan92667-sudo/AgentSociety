from datetime import UTC, datetime, timedelta

import pytest

from yelp_agent.profiles.schema import (
    PreferenceSignal,
    ProfileEvidenceSummary,
    UserProfileV1,
)
from yelp_agent.recommendation_v2.preference_fusion.profile_adapter import (
    ASPECT_DIRECTION_POLICY,
    adapt_user_profile,
)
from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS

NOW = datetime(2026, 8, 19, tzinfo=UTC)


def _signal(
    *,
    kind: str,
    value: str,
    score: float,
    evidence_count: int = 3,
    last_confirmed: datetime = NOW,
) -> PreferenceSignal:
    """建立一条带完整时间和证据数据的画像信号。"""

    source = {
        "category": "rating_category",
        "aspect": "review_aspect",
        "price": "business_price",
        "area": "business_area",
    }[kind]
    return PreferenceSignal.model_validate(
        {
            "kind": kind,
            "value": value,
            "score": score,
            "confidence": 0.9,
            "evidence_count": evidence_count,
            "effective_evidence": float(evidence_count),
            "first_seen": last_confirmed - timedelta(days=400),
            "last_confirmed": last_confirmed,
            "source": source,
        }
    )


def _profile(**overrides: object) -> UserProfileV1:
    """建立可按测试需要替换各类画像信号的完整画像。"""

    payload: dict[str, object] = {
        "profile_id": "a" * 64,
        "user_id": "user-1",
        "cutoff_time": NOW,
        "history_length": 10,
        "average_rating": 4.0,
        "rating_distribution": {"1": 0, "2": 1, "3": 2, "4": 4, "5": 3},
        "category_preferences": [],
        "category_dislikes": [],
        "aspect_preferences": [],
        "aspect_dislikes": [],
        "price_preference": None,
        "frequent_areas": [],
        "location_center": None,
        "reliability": 0.8,
        "evidence_summary": ProfileEvidenceSummary(
            category_evidence_count=0,
            aspect_evidence_count=0,
            price_evidence_count=0,
            area_evidence_count=0,
            first_interaction=NOW - timedelta(days=400),
            last_interaction=NOW,
        ),
        "profile_version": "1.0.0",
    }
    payload.update(overrides)
    return UserProfileV1.model_validate(payload)


def test_all_existing_aspects_have_one_explicit_direction_policy() -> None:
    """新增评论特征时如果忘记定义方向，这个测试会立即失败。"""

    assert set(ASPECT_DIRECTION_POLICY) == set(ASPECT_FIELDS)


def test_each_category_stays_atomic_and_areas_wait_for_coordinates() -> None:
    """菜系逐条保留；区域先等地点工具解析，不能拿字符串直接排序。"""

    profile = _profile(
        category_preferences=[
            _signal(kind="category", value="Japanese", score=0.9),
            _signal(kind="category", value="Sichuan", score=0.5),
        ],
        category_dislikes=[
            _signal(kind="category", value="Fast Food", score=-0.8),
        ],
        frequent_areas=[
            _signal(kind="area", value="Flushing", score=0.7),
            _signal(kind="area", value="Manhattan", score=0.4),
        ],
    )

    result = adapt_user_profile(profile)

    categories = [
        item for item in result.soft_preferences if item.field == "category"
    ]
    assert {
        (item.direction, tuple(item.target_value)) for item in categories
    } == {
        ("match", ("Japanese",)),
        ("avoid", ("Fast Food",)),
        ("match", ("Sichuan",)),
    }
    assert {
        (item.value, item.reason) for item in result.ignored_signals
    } == {
        ("Flushing", "unresolved_location"),
        ("Manhattan", "unresolved_location"),
    }
    assert len({item.key for item in result.soft_preferences}) == len(
        result.soft_preferences
    )


def test_price_and_profile_provenance_are_preserved_without_decay() -> None:
    """价格、原始分数、次数和时间都保留，旧记录不会在转换时擅自降级。"""

    confirmed = NOW - timedelta(days=800)
    profile = _profile(
        price_preference=_signal(
            kind="price",
            value="3",
            score=0.9,
            evidence_count=1,
            last_confirmed=confirmed,
        )
    )

    result = adapt_user_profile(profile)
    price = result.soft_preferences[0]
    basis = price.sources[0]

    assert price.field == "price_level"
    assert price.direction == "closer_to"
    assert price.target_value == 3
    assert price.preference_strength == 75
    assert basis.profile_score == 0.9
    assert basis.profile_evidence_count == 1
    assert basis.profile_last_confirmed == confirmed


@pytest.mark.parametrize(
    ("field", "negative_history", "expected_direction"),
    [
        ("service", False, "higher"),
        ("service", True, "higher"),
        ("crowded", False, "lower"),
        ("crowded", True, "lower"),
        ("queue_time", False, "lower"),
        ("quiet_environment", True, "higher"),
        ("spiciness", False, "higher"),
        ("spiciness", True, "lower"),
    ],
)
def test_aspect_direction_uses_semantic_policy(
    field: str,
    negative_history: bool,
    expected_direction: str,
) -> None:
    """评论特征方向由集中语义表决定，而不是由列表名称统一决定。"""

    signal = _signal(
        kind="aspect",
        value=field,
        score=-0.8 if negative_history else 0.8,
    )
    profile = _profile(
        aspect_dislikes=[signal] if negative_history else [],
        aspect_preferences=[] if negative_history else [signal],
    )

    result = adapt_user_profile(profile)

    assert result.soft_preferences[0].direction == expected_direction


def test_unsupported_aspect_is_reported_instead_of_silently_dropped() -> None:
    """旧画像出现统一结构尚不认识的评论特征时，要留下明确记录。"""

    profile = _profile(
        aspect_preferences=[
            _signal(kind="aspect", value="live_music", score=0.8),
        ]
    )

    result = adapt_user_profile(profile)

    assert result.soft_preferences == []
    assert result.ignored_signals[0].value == "live_music"
    assert result.ignored_signals[0].reason == "unsupported_aspect"


def test_conflicting_profile_aspect_records_do_not_crash_adapter() -> None:
    """画像里同一辣度同时有正负记录时都应忠实保留，留给融合裁决。"""

    profile = _profile(
        aspect_preferences=[
            _signal(kind="aspect", value="spiciness", score=0.8),
        ],
        aspect_dislikes=[
            _signal(kind="aspect", value="spiciness", score=-0.7),
        ],
    )

    result = adapt_user_profile(profile)

    assert [item.direction for item in result.soft_preferences] == [
        "higher",
        "lower",
    ]
    assert len({item.key for item in result.soft_preferences}) == 2


def test_profile_priority_is_internal_contiguous_and_evidence_aware() -> None:
    """先按强度，再按次数、时间排序，并从1开始连续编号。"""

    profile = _profile(
        category_preferences=[
            _signal(kind="category", value="A", score=0.9, evidence_count=2),
            _signal(kind="category", value="B", score=0.9, evidence_count=8),
            _signal(kind="category", value="C", score=0.5, evidence_count=20),
        ]
    )

    result = adapt_user_profile(profile)

    assert [item.target_value for item in result.soft_preferences] == [
        ["B"],
        ["A"],
        ["C"],
    ]
    assert [item.priority for item in result.soft_preferences] == [1, 2, 3]
    assert all(
        item.controlling_source == "user_profile"
        for item in result.soft_preferences
    )
