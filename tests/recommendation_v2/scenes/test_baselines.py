import pytest

from yelp_agent.recommendation_v2 import SCENE_ORDER, get_scene_baseline

EXPECTED_LABELS = {
    "casual": "随便吃",
    "date": "约会",
    "business": "商务",
    "friends": "朋友聚餐",
    "family": "家庭聚餐",
    "solo": "一个人吃",
}
EXPECTED_DISTANCE_KM = {
    "casual": 5,
    "date": 8,
    "business": 6,
    "friends": 8,
    "family": 6,
    "solo": 3,
}
EXPECTED_PREFERENCE_FIELDS = {
    "casual": [
        "food_quality",
        "price_value",
        "distance_km",
        "price_level",
        "queue_time",
    ],
    "date": [
        "quiet_environment",
        "date_suitable",
        "service",
        "price_level",
        "cleanliness",
    ],
    "business": [
        "quiet_environment",
        "service",
        "group_suitable",
        "cleanliness",
        "price_level",
    ],
    "friends": [
        "group_suitable",
        "food_quality",
        "price_value",
        "price_level",
        "queue_time",
    ],
    "family": [
        "family_friendly",
        "cleanliness",
        "parking",
        "price_level",
        "queue_time",
    ],
    "solo": [
        "distance_km",
        "queue_time",
        "price_value",
        "price_level",
        "food_quality",
    ],
}


def test_exactly_six_scene_baselines_are_available() -> None:
    """场景基准只能是已经约定的六种，不能悄悄增加其他场景。"""

    assert SCENE_ORDER == (
        "casual",
        "date",
        "business",
        "friends",
        "family",
        "solo",
    )
    assert {scene: get_scene_baseline(scene).scene_label for scene in SCENE_ORDER} == (
        EXPECTED_LABELS
    )


@pytest.mark.parametrize("scene", SCENE_ORDER)
def test_each_scene_has_one_revisable_default_distance(scene: str) -> None:
    """场景只提供初始距离，不把价格、菜系等猜测成默认硬条件。"""

    baseline = get_scene_baseline(scene)

    assert len(baseline.default_constraints) == 1
    distance = baseline.default_constraints[0]
    assert distance.key == "distance.max"
    assert distance.field == "distance_km"
    assert distance.operator == "less_than_or_equal"
    assert distance.value == EXPECTED_DISTANCE_KM[scene]
    assert distance.controlling_source == "scene"
    assert {item.source for item in distance.sources} == {"scene"}


@pytest.mark.parametrize("scene", SCENE_ORDER)
def test_scene_soft_preferences_have_fixed_order_and_moderate_strength(
    scene: str,
) -> None:
    """场景软偏好有明确顺序，但不能使用代表用户明确强调的强度100。"""

    baseline = get_scene_baseline(scene)

    assert [item.field for item in baseline.soft_preferences] == (
        EXPECTED_PREFERENCE_FIELDS[scene]
    )
    assert [item.priority for item in baseline.soft_preferences] == [1, 2, 3, 4, 5]
    assert all(item.controlling_source == "scene" for item in baseline.soft_preferences)
    assert all(
        item.preference_strength in {25, 50, 75} for item in baseline.soft_preferences
    )
    assert all(
        source.source == "scene"
        for item in baseline.soft_preferences
        for source in item.sources
    )


@pytest.mark.parametrize(
    ("scene", "expected_price_level"),
    [
        ("casual", 2),
        ("date", 3),
        ("business", 3),
        ("friends", 2),
        ("family", 2),
        ("solo", 2),
    ],
)
def test_scene_price_is_a_soft_target_not_a_default_filter(
    scene: str,
    expected_price_level: int,
) -> None:
    """场景只能让某个价格档位更靠前，不能因此删掉其他价格档位。"""

    baseline = get_scene_baseline(scene)
    price = next(
        item for item in baseline.soft_preferences if item.field == "price_level"
    )

    assert price.direction == "closer_to"
    assert price.target_value == expected_price_level
    assert all(item.field != "price_level" for item in baseline.default_constraints)


def test_reading_a_scene_returns_a_fresh_copy() -> None:
    """调用方修改读取结果后，固定场景基准本身不能被污染。"""

    first = get_scene_baseline("date")
    first.soft_preferences.pop()
    first.default_constraints[0].value = 99

    second = get_scene_baseline("date")

    assert len(second.soft_preferences) == 5
    assert second.default_constraints[0].value == 8
