import json

from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.recommendation_v2 import SIX_SCENE_QUERIES
from yelp_agent.recommendation_v2.query_compiler import (
    QueryCompiler,
    QueryCompilerRequest,
)


class FakeGenerator:
    """返回指定 JSON，并保存提示内容供测试检查。"""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.messages: list[LLMMessage] = []

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        """模拟一次成功的大模型 JSON 返回。"""

        self.messages = messages
        return LLMCallResult(
            status="success",
            content=json.dumps(self._payload, ensure_ascii=False),
            model="fake-model",
            latency_ms=1,
            attempt_count=1,
        )


def date_payload() -> dict[str, object]:
    """构造一份完整的约会场景模型输出。"""

    return {
        "scene": {"kind": "date", "evidence_span": "和对象约会"},
        "hard_constraints": [
            {
                "field": "category",
                "operator": "any_of",
                "value": ["日料"],
                "evidence_span": "想吃日料",
            },
            {
                "field": "distance_km",
                "operator": "less_than_or_equal",
                "value": 5,
                "evidence_span": "距离5公里以内",
            },
            {
                "field": "price_level",
                "operator": "less_than_or_equal",
                "value": 3,
                "evidence_span": "价格三档以内",
            },
        ],
        "soft_preferences": [
            {
                "field": "quiet_environment",
                "direction": "higher",
                "target_value": None,
                "preference_strength": 100,
                "priority": 1,
                "evidence_span": "安静最重要",
            },
            {
                "field": "date_suitable",
                "direction": "higher",
                "target_value": None,
                "preference_strength": 75,
                "priority": 2,
                "evidence_span": "适合约会其次",
            },
            {
                "field": "service",
                "direction": "higher",
                "target_value": None,
                "preference_strength": 50,
                "priority": 3,
                "evidence_span": "服务第三",
            },
        ],
    }


def test_compiler_only_extracts_current_requirements() -> None:
    """编译结果不能混入场景默认值，也不能产生任何商家推荐。"""

    generator = FakeGenerator(date_payload())
    compiler = QueryCompiler(generator)
    query = (
        "今晚和对象约会，想吃日料，距离5公里以内，价格三档以内，"
        "安静最重要，适合约会其次，服务第三。"
    )

    attempt = compiler.compile(QueryCompilerRequest(query_text=query))

    assert attempt.status == "success"
    assert attempt.compiled_query is not None
    assert attempt.compiled_query.scene.kind == "date"
    assert [item.field for item in attempt.compiled_query.hard_constraints] == [
        "category",
        "distance_km",
        "price_level",
    ]
    assert [item.field for item in attempt.compiled_query.soft_preferences] == [
        "quiet_environment",
        "date_suitable",
        "service",
    ]
    dumped = attempt.compiled_query.model_dump()
    assert "default_constraints" not in dumped
    assert "businesses" not in dumped
    assert "绝对不要推荐商家" in generator.messages[0].content


def test_review_aspect_in_hard_constraints_is_rejected() -> None:
    """大模型把安静误放进硬条件时，结构检查必须拒绝整份结果。"""

    payload = date_payload()
    payload["hard_constraints"] = [
        {
            "field": "quiet_environment",
            "operator": "greater_than_or_equal",
            "value": 70,
            "evidence_span": "安静最重要",
        }
    ]
    compiler = QueryCompiler(FakeGenerator(payload))

    attempt = compiler.compile(
        QueryCompilerRequest(
            query_text="今晚和对象约会，安静最重要，适合约会其次，服务第三。"
        )
    )

    assert attempt.status == "invalid_output"
    assert attempt.compiled_query is None


def test_query_without_scene_keeps_scene_empty() -> None:
    """用户没有表达用餐场景时，不能擅自套用“随便吃”。"""

    generator = FakeGenerator(
        {
            "scene": None,
            "hard_constraints": [],
            "soft_preferences": [
                {
                    "field": "distance_km",
                    "direction": "lower",
                    "target_value": None,
                    "preference_strength": 75,
                    "priority": 1,
                    "evidence_span": "近一点",
                }
            ],
        }
    )

    attempt = QueryCompiler(generator).compile(
        QueryCompilerRequest(query_text="找一家近一点的餐厅")
    )

    assert attempt.status == "success"
    assert attempt.compiled_query is not None
    assert attempt.compiled_query.scene is None


def test_rewritten_evidence_not_present_in_query_is_rejected() -> None:
    """大模型不能把自己的概括冒充用户原话。"""

    payload = date_payload()
    payload["scene"] = {"kind": "date", "evidence_span": "浪漫约会"}
    compiler = QueryCompiler(FakeGenerator(payload))
    query = (
        "今晚和对象约会，想吃日料，距离5公里以内，价格三档以内，"
        "安静最重要，适合约会其次，服务第三。"
    )

    attempt = compiler.compile(QueryCompilerRequest(query_text=query))

    assert attempt.status == "invalid_output"
    assert attempt.failure_reason == "evidence_span_not_in_query"


def test_nearer_distance_must_use_lower_direction() -> None:
    """“距离近”表示数值越小越好，不是接近一个没有给出的目标值。"""

    payload = date_payload()
    payload["soft_preferences"] = [
        {
            "field": "distance_km",
            "direction": "closer_to",
            "target_value": None,
            "preference_strength": 100,
            "priority": 1,
            "evidence_span": "距离近第一",
        }
    ]
    compiler = QueryCompiler(FakeGenerator(payload))

    attempt = compiler.compile(
        QueryCompilerRequest(
            query_text=(
                "今晚和对象约会，想吃日料，距离5公里以内，价格三档以内，距离近第一。"
            )
        )
    )

    assert attempt.status == "invalid_output"


def test_six_examples_cover_each_supported_scene_once() -> None:
    """实际调用大模型的六条问题必须一一覆盖六个场景。"""

    assert [item.scene for item in SIX_SCENE_QUERIES] == [
        "casual",
        "date",
        "business",
        "friends",
        "family",
        "solo",
    ]
    assert len({item.query_text for item in SIX_SCENE_QUERIES}) == 6
