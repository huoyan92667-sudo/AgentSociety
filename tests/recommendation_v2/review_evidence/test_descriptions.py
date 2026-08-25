import json

from yelp_agent.agent.llm import LLMCallResult
from yelp_agent.recommendation_v2.review_evidence.descriptions import (
    PreferenceDescriptionBuilder,
)
from yelp_agent.recommendation_v2.scenes import get_scene_baseline
from yelp_agent.recommendation_v2.schema import OpenRequirement, RequirementBasis


class _Generator:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, messages: list[object]) -> LLMCallResult:
        self.call_count += 1
        user_payload = json.loads(messages[-1].content)  # type: ignore[attr-defined]
        items = user_payload["items"]
        return LLMCallResult(
            status="success",
            content=json.dumps(
                {
                    "items": [
                        {
                            "requirement_id": item["requirement_id"],
                            "positive_descriptions": [
                                f"positive meaning one for {item['requirement_id']}",
                                f"positive meaning two for {item['requirement_id']}",
                            ],
                            "negative_descriptions": [
                                f"negative meaning one for {item['requirement_id']}",
                                f"negative meaning two for {item['requirement_id']}",
                            ],
                        }
                        for item in items
                    ]
                }
            ),
            model="fake",
            latency_ms=1,
            attempt_count=1,
        )


def _open(key: str, text: str, priority: int) -> OpenRequirement:
    return OpenRequirement(
        key=key,
        text=text,
        behavior="prefer",
        priority=priority,
        controlling_source="current_query",
        sources=[
            RequirementBasis(
                source="current_query",
                text=text,
                turn_index=1,
                preference_strength=100,
            )
        ],
    )


def test_fixed_aspect_uses_fixed_descriptions_without_model_call() -> None:
    generator = _Generator()
    preference = get_scene_baseline("date").soft_preferences[0]

    result = PreferenceDescriptionBuilder(generator).build([preference], [])

    assert result.failure_reason is None
    assert result.descriptions[0].kind == "fixed_aspect"
    assert generator.call_count == 0


def test_all_long_tail_requirements_share_one_model_call() -> None:
    generator = _Generator()

    result = PreferenceDescriptionBuilder(generator).build(
        [],
        [
            _open("open.window", "最好坐窗边", 1),
            _open("open.music", "现场音乐别太吵", 2),
        ],
    )

    assert result.failure_reason is None
    assert len(result.descriptions) == 2
    assert all(item.kind == "long_tail" for item in result.descriptions)
    assert generator.call_count == 1


def test_online_fixed_aspect_is_rewritten_with_current_query() -> None:
    generator = _Generator()
    preference = get_scene_baseline("date").soft_preferences[0]

    result = PreferenceDescriptionBuilder(generator).build(
        [preference],
        [],
        query_text="我想吃牛排",
    )

    assert result.failure_reason is None
    assert result.descriptions[0].kind == "fixed_aspect"
    assert generator.call_count == 1


def test_current_long_tail_requirement_is_returned_before_weaker_fixed_aspect() -> None:
    """当前问题的第一偏好必须先于画像或场景评论，供后续挑证据使用。"""

    generator = _Generator()
    scene_preference = get_scene_baseline("date").soft_preferences[0].model_copy(
        update={"priority": 2},
        deep=True,
    )

    result = PreferenceDescriptionBuilder(generator).build(
        [scene_preference],
        [_open("open.authentic", "要地道的川菜", 1)],
        query_text="今晚去吃地道川菜",
    )

    assert [item.requirement_id for item in result.descriptions] == [
        "open.authentic",
        scene_preference.key,
    ]
