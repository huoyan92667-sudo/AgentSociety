import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.recommendation_v2.preference_fusion import (
    CompactHardRequirement,
    CompactOpenRequirement,
    CompactReviewSearchPlan,
    CompactSceneSelection,
    CompactSearchCenter,
    CompactSoftRequirement,
    ConversationHistoryTurn,
    HistoryBusinessFactTool,
    HistoryFactQuery,
    PreferenceFusion,
    PreferenceFusionProposal,
    PreferenceFusionRequest,
    PreferenceFusionToolCall,
    ProfilePreferenceSet,
)
from yelp_agent.recommendation_v2.scenes import get_scene_baseline
from yelp_agent.recommendation_v2.schema import (
    BusinessReference,
    GeoPoint,
    SoftPreference,
    UnifiedRecommendationState,
)

PROFILE_TIME = datetime(2026, 8, 1, tzinfo=UTC)
USER_LOCATION = GeoPoint(latitude=39.9526, longitude=-75.1652)


class FakeGenerator:
    """按顺序返回工具调用或最终需求，并保存每次实际输入。"""

    def __init__(
        self,
        *responses: PreferenceFusionProposal | PreferenceFusionToolCall | str,
        failure: str | None = None,
    ) -> None:
        self._responses = list(responses)
        self._failure = failure
        self.calls: list[list[LLMMessage]] = []

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        """模拟成功、工具循环、供应商失败或非法 JSON。"""

        self.calls.append(list(messages))
        if self._failure is not None:
            return LLMCallResult(
                status="failure",
                failure_reason=self._failure,
                model="fake-model",
                latency_ms=1,
                attempt_count=1,
            )
        response = self._responses.pop(0)
        content = response if isinstance(response, str) else response.model_dump_json()
        return LLMCallResult(
            status="success",
            content=content,
            model="fake-model",
            latency_ms=1,
            attempt_count=1,
            input_tokens=10,
            output_tokens=5,
        )


def _profile_soft(
    *,
    key: str,
    field: str,
    direction: str,
    target_value: object = None,
    strength: int = 75,
    priority: int = 1,
) -> SoftPreference:
    """建立一条真实的长期画像软偏好。"""

    feature = "categories" if field == "category" else field
    return SoftPreference.model_validate(
        {
            "key": key,
            "field": field,
            "direction": direction,
            "target_value": target_value,
            "preference_strength": strength,
            "priority": priority,
            "merchant_feature": feature,
            "controlling_source": "user_profile",
            "sources": [
                {
                    "source": "user_profile",
                    "text": "长期画像记录",
                    "preference_strength": strength,
                    "profile_score": 0.8,
                    "profile_evidence_count": 5,
                    "profile_last_confirmed": PROFILE_TIME,
                }
            ],
        }
    )


def _profile(*preferences: SoftPreference) -> ProfilePreferenceSet:
    """建立一份画像输入。"""

    return ProfilePreferenceSet(
        profile_id="a" * 64,
        user_id="user-1",
        soft_preferences=list(preferences),
    )


def _previous_state() -> UnifiedRecommendationState:
    """建立最小上一轮状态，用于检查模型收到的内容。"""

    return UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=1,
        latest_query_text="上一轮想找餐厅",
        user_location=USER_LOCATION,
    )


def test_model_receives_only_dialogue_context_and_hides_program_managed_sources() -> None:
    """模型只看对话；画像、场景基准和商家事实都由程序管理。"""

    history = ConversationHistoryTurn(
        turn_index=1,
        user_message="推荐几家",
        assistant_message="给你三家",
        presented_businesses=[
            BusinessReference(
                presented_turn_index=1,
                position=3,
                business_id="business-c",
                business_name="第三家",
                distance_km=3.0,
                price_level=4,
            )
        ],
    )
    profile_parking = _profile_soft(
        key="profile.parking",
        field="parking",
        direction="higher",
    )
    proposal = PreferenceFusionProposal(
        scene=CompactSceneSelection(
            kind="date",
            evidence_text="约会",
            evidence_turn_index=2,
        )
    )
    generator = FakeGenerator(proposal)

    attempt = PreferenceFusion(generator).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=2,
            query_text="这次约会",
            previous_state=_previous_state(),
            conversation_history=[history],
            scene_baseline=get_scene_baseline("date"),
            profile_preferences=_profile(profile_parking),
        )
    )

    assert attempt.status == "success"
    assert len(generator.calls) == 1
    payload = json.loads(generator.calls[0][1].content)
    assert payload["previous_state"]["revision"] == 1
    visible_business = payload["conversation_history"][0]["presented_businesses"][0]
    assert visible_business == {
        "presented_turn_index": 1,
        "position": 3,
        "business_id": "business-c",
        "business_name": "第三家",
    }
    assert "scene_baseline" not in payload
    assert "profile_preferences" not in payload
    assert payload["available_tool"]["name"] == "lookup_history_business"


def test_model_receives_only_retrieved_category_candidates_and_compact_contract() -> None:
    """类别先在本地召回，不能把179个类别和完整模型定义重复发给大模型。"""

    generator = FakeGenerator(PreferenceFusionProposal())
    attempt = PreferenceFusion(generator).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=1,
            query_text="我今天晚上7点想吃牛排",
        )
    )

    assert attempt.status == "success"
    payload = json.loads(generator.calls[0][1].content)
    candidates = payload["category_candidates"]
    assert candidates[0]["category"] == "Steakhouses"
    assert len(candidates) <= 5
    assert "allowed_dining_categories" not in payload
    assert "final_output_schema" not in payload
    assert "output_contract" in payload
    assert len(generator.calls[0][1].content) < 8_000


def test_one_fusion_call_also_prepares_long_tail_review_search_descriptions() -> None:
    """开放要求的正反检索说法必须随融合结果一起生成，不再另调模型。"""

    proposal = PreferenceFusionProposal(
        open_requirements=[
            CompactOpenRequirement(
                text="地道川菜",
                behavior="prefer",
                priority=1,
                evidence_text="要地道川菜",
                evidence_turn_index=1,
            )
        ],
        review_search_plans=[
            CompactReviewSearchPlan(
                kind="long_tail",
                requirement_text="地道川菜",
                behavior="prefer",
                positive_descriptions=[
                    "authentic Sichuan flavor",
                    "proper numbing spicy balance",
                ],
                negative_descriptions=[
                    "watered down Sichuan flavor",
                    "inauthentic generic Chinese food",
                ],
            )
        ],
    )
    generator = FakeGenerator(proposal)

    attempt = PreferenceFusion(generator).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=1,
            query_text="要地道川菜",
        )
    )

    assert attempt.status == "success"
    assert len(generator.calls) == 1
    assert len(attempt.review_search_descriptions) == 1
    description = attempt.review_search_descriptions[0]
    assert description.requirement_text == "地道川菜"
    assert description.positive_descriptions[0] == "authentic Sichuan flavor"


def test_compact_output_does_not_ask_model_for_fixed_technical_fields() -> None:
    """编号、单位、商家字段和来源不属于大模型输出。"""

    hard_properties = CompactHardRequirement.model_json_schema()["properties"]
    soft_properties = CompactSoftRequirement.model_json_schema()["properties"]

    assert "key" not in hard_properties
    assert "unit" not in hard_properties
    assert "merchant_feature" not in hard_properties
    assert "controlling_source" not in hard_properties
    assert "key" not in soft_properties
    assert "merchant_feature" not in soft_properties
    assert "controlling_source" not in soft_properties
    assert "preference_strength" not in soft_properties


def test_named_place_creates_search_center_and_distance_hard_constraint() -> None:
    """地点由模型给出近似坐标，程序必须固定补成可执行的范围硬条件。"""

    proposal = PreferenceFusionProposal(
        search_center=CompactSearchCenter(
            label="费城唐人街",
            latitude=39.9537,
            longitude=-75.1579,
            radius_km=1.5,
            evidence_text="费城唐人街",
            evidence_turn_index=1,
        ),
        hard_constraints=[
            CompactHardRequirement(
                field="category",
                operator="any_of",
                value=["Szechuan"],
                evidence_text="吃川菜",
                evidence_turn_index=1,
            )
        ],
    )

    attempt = PreferenceFusion(FakeGenerator(proposal)).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=1,
            query_text="去费城唐人街吃川菜",
        )
    )

    assert attempt.status == "success"
    assert attempt.state is not None
    assert attempt.state.search_center is not None
    assert attempt.state.search_center.label == "费城唐人街"
    distance = next(
        item for item in attempt.state.hard_constraints if item.field == "distance_km"
    )
    assert distance.operator == "less_than_or_equal"
    assert distance.value == 1.5
    assert distance.sources[0].text == "费城唐人街"


def test_dialogue_soft_preferences_require_a_clear_total_order() -> None:
    """安静和距离不能都写成第一名，否则后面无法做先安静再比距离。"""

    with pytest.raises(ValidationError):
        PreferenceFusionProposal(
            soft_preferences=[
                CompactSoftRequirement(
                    field="quiet_environment",
                    direction="higher",
                    priority=1,
                    evidence_text="安静最重要",
                    evidence_turn_index=1,
                ),
                CompactSoftRequirement(
                    field="distance_km",
                    direction="lower",
                    priority=1,
                    evidence_text="然后近一点",
                    evidence_turn_index=1,
                ),
            ]
        )


def test_no_scene_means_no_scene_defaults_or_preferences() -> None:
    """没有场景就是空，不偷偷套用任何场景。"""

    attempt = PreferenceFusion(
        FakeGenerator(PreferenceFusionProposal(scene=None))
    ).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=1,
            query_text="帮我找个餐厅",
        )
    )

    assert attempt.status == "success"
    assert attempt.state is not None
    assert attempt.state.scene is None
    assert attempt.state.default_constraints == []
    assert attempt.state.soft_preferences == []


def test_code_enforces_query_session_profile_scene_source_order() -> None:
    """大模型整理语义，来源冲突仍由固定顺序裁决。"""

    profile_quiet = _profile_soft(
        key="profile.quiet",
        field="quiet_environment",
        direction="lower",
    )
    proposal = PreferenceFusionProposal(
        scene=CompactSceneSelection(
            kind="date",
            evidence_text="之前说是约会",
            evidence_turn_index=1,
        ),
        soft_preferences=[
            CompactSoftRequirement(
                field="spiciness",
                direction="higher",
                priority=1,
                evidence_text="辣一点",
                evidence_turn_index=2,
            ),
            CompactSoftRequirement(
                field="spiciness",
                direction="lower",
                priority=2,
                evidence_text="清淡一点",
                evidence_turn_index=1,
            ),
        ],
    )

    attempt = PreferenceFusion(FakeGenerator(proposal)).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=2,
            query_text="辣一点",
            previous_state=_previous_state(),
            conversation_history=[
                ConversationHistoryTurn(
                    turn_index=1,
                    user_message="之前说是约会，清淡一点",
                )
            ],
            scene_baseline=get_scene_baseline("date"),
            profile_preferences=_profile(profile_quiet),
        )
    )

    assert attempt.status == "success"
    assert attempt.state is not None
    spicy = next(
        item for item in attempt.state.soft_preferences if item.field == "spiciness"
    )
    quiet = next(
        item
        for item in attempt.state.soft_preferences
        if item.field == "quiet_environment"
    )
    assert spicy.direction == "higher"
    assert spicy.controlling_source == "current_query"
    assert quiet.direction == "lower"
    assert quiet.controlling_source == "user_profile"


def test_suppressed_profile_is_saved_and_recovers_after_dialogue_preference_removed() -> None:
    """被本轮川菜压住的日料画像不会丢，下一轮可以恢复。"""

    japanese = _profile_soft(
        key="profile.category.japanese",
        field="category",
        direction="match",
        target_value=["Japanese"],
    )
    first = PreferenceFusion(
        FakeGenerator(
            PreferenceFusionProposal(
                hard_constraints=[
                    CompactHardRequirement(
                        field="category",
                        operator="any_of",
                        value=["Szechuan"],
                        evidence_text="想吃川菜",
                        evidence_turn_index=1,
                    )
                ]
            )
        )
    ).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=1,
            query_text="想吃川菜",
            profile_preferences=_profile(japanese),
        )
    )
    assert first.status == "success"
    assert first.state is not None
    memory = next(
        item for item in first.state.preference_memory if item.source == "user_profile"
    )
    assert memory.status == "suppressed"

    second = PreferenceFusion(
        FakeGenerator(PreferenceFusionProposal())
    ).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=2,
            query_text="川菜算了，按我平常喜欢的来",
            previous_state=first.state,
        )
    )

    assert second.status == "success"
    assert second.state is not None
    assert second.state.soft_preferences[0].target_value == ["Japanese"]
    assert second.state.soft_preferences[0].controlling_source == "user_profile"


def test_model_chooses_history_tool_then_builds_distance_constraint() -> None:
    """程序不匹配“第三家太远”，而是执行模型主动选择的通用查询。"""

    business = BusinessReference(
        presented_turn_index=1,
        position=3,
        business_id="business-c",
        business_name="第三家",
        distance_km=3.0,
    )
    query = HistoryFactQuery(position=3, fields=["distance_km"])
    expected = HistoryBusinessFactTool().execute(query, [business])
    assert expected.fact is not None
    tool_call = PreferenceFusionToolCall(
        action="lookup_history_business",
        arguments=query,
    )
    proposal = PreferenceFusionProposal(
        hard_constraints=[
            CompactHardRequirement(
                field="distance_km",
                operator="less_than",
                value=3.0,
                evidence_text="第三家太远了，别推荐这么远的",
                evidence_turn_index=2,
                supporting_fact_ids=[expected.fact.fact_id],
            )
        ]
    )
    generator = FakeGenerator(tool_call, proposal)

    attempt = PreferenceFusion(generator).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=2,
            query_text="第三家太远了，别推荐这么远的",
            conversation_history=[
                ConversationHistoryTurn(
                    turn_index=1,
                    user_message="推荐几家",
                    presented_businesses=[business],
                )
            ],
            user_location=USER_LOCATION,
        )
    )

    assert attempt.status == "success"
    assert attempt.model_call_count == 2
    assert attempt.input_tokens == 20
    assert attempt.output_tokens == 10
    assert len(attempt.tool_observations) == 1
    assert attempt.tool_observations[0].fact is not None
    assert len(generator.calls[1]) == 4
    assert attempt.state is not None
    constraint = attempt.state.hard_constraints[0]
    assert constraint.field == "distance_km"
    assert constraint.operator == "less_than"
    assert constraint.value == 3.0
    assert constraint.unit == "kilometer"
    assert constraint.merchant_feature == "coordinates"
    assert constraint.derivation is not None
    assert constraint.derivation.reference_business_id == "business-c"


def test_same_history_tool_handles_price_without_a_special_sentence_rule() -> None:
    """同一个工具也能支持“第二个太贵”，证明没有硬编码距离句式。"""

    business = BusinessReference(
        presented_turn_index=4,
        position=2,
        business_id="business-b",
        business_name="第二家",
        price_level=4,
    )
    query = HistoryFactQuery(position=2, fields=["price_level"])
    expected = HistoryBusinessFactTool().execute(query, [business])
    assert expected.fact is not None
    generator = FakeGenerator(
        PreferenceFusionToolCall(
            action="lookup_history_business",
            arguments=query,
        ),
        PreferenceFusionProposal(
            hard_constraints=[
                CompactHardRequirement(
                    field="price_level",
                    operator="less_than",
                    value=4,
                    evidence_text="第二个太贵了",
                    evidence_turn_index=5,
                    supporting_fact_ids=[expected.fact.fact_id],
                )
            ]
        ),
    )

    attempt = PreferenceFusion(generator).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=5,
            query_text="第二个太贵了",
            conversation_history=[
                ConversationHistoryTurn(
                    turn_index=4,
                    user_message="继续推荐",
                    presented_businesses=[business],
                )
            ],
        )
    )

    assert attempt.status == "success"
    assert attempt.state is not None
    assert attempt.state.hard_constraints[0].field == "price_level"
    assert attempt.state.hard_constraints[0].value == 4


def test_model_cannot_reference_a_fact_that_no_tool_returned() -> None:
    """模型即使编造看似正确的事实编号，也不能进入最终状态。"""

    proposal = PreferenceFusionProposal(
        hard_constraints=[
            CompactHardRequirement(
                field="distance_km",
                operator="less_than",
                value=3,
                evidence_text="别超过三公里",
                evidence_turn_index=1,
                supporting_fact_ids=["history_business.0000000000000000"],
            )
        ]
    )
    attempt = PreferenceFusion(FakeGenerator(proposal)).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=1,
            query_text="别超过三公里",
            user_location=USER_LOCATION,
        )
    )

    assert attempt.status == "invalid_output"
    assert attempt.state is None
    assert attempt.failure_reason == "requirement references a fact not returned by a tool"


def test_missing_history_fact_can_be_kept_as_open_requirement_without_guessing() -> None:
    """工具查不到时，大模型可以保留原要求，但不能凭空造数值。"""

    generator = FakeGenerator(
        PreferenceFusionToolCall(
            action="lookup_history_business",
            arguments=HistoryFactQuery(position=3, fields=["distance_km"]),
        ),
        PreferenceFusionProposal(
            open_requirements=[
                CompactOpenRequirement(
                    text="不要像第三家那么远",
                    behavior="must_have",
                    evidence_text="第三家太远",
                    evidence_turn_index=1,
                )
            ]
        ),
    )

    attempt = PreferenceFusion(generator).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=1,
            query_text="第三家太远",
        )
    )

    assert attempt.status == "success"
    assert attempt.tool_observations[0].status == "not_found"
    assert attempt.state is not None
    assert attempt.state.hard_constraints == []
    assert attempt.state.open_requirements[0].text == "不要像第三家那么远"


def test_explicit_distance_removes_weaker_scene_default_distance() -> None:
    """用户明确距离出现后，同字段场景默认值不再生效。"""

    proposal = PreferenceFusionProposal(
        scene=CompactSceneSelection(
            kind="date",
            evidence_text="约会",
            evidence_turn_index=1,
        ),
        hard_constraints=[
            CompactHardRequirement(
                field="distance_km",
                operator="less_than_or_equal",
                value=10,
                evidence_text="距离10公里以内",
                evidence_turn_index=1,
            )
        ],
    )
    attempt = PreferenceFusion(FakeGenerator(proposal)).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=1,
            query_text="约会，距离10公里以内",
            scene_baseline=get_scene_baseline("date"),
            user_location=USER_LOCATION,
        )
    )

    assert attempt.status == "success"
    assert attempt.state is not None
    assert len(attempt.state.hard_constraints) == 1
    assert attempt.state.default_constraints == []


def test_session_evidence_must_exist_in_recorded_user_history() -> None:
    """模型不能声称用户在不存在的历史里说过某个偏好。"""

    proposal = PreferenceFusionProposal(
        soft_preferences=[
            CompactSoftRequirement(
                field="quiet_environment",
                direction="higher",
                priority=1,
                evidence_text="上一轮要求安静",
                evidence_turn_index=1,
            )
        ]
    )
    attempt = PreferenceFusion(FakeGenerator(proposal)).fuse(
        PreferenceFusionRequest(
            user_id="user-1",
            session_id="session-1",
            turn_index=2,
            query_text="继续推荐",
        )
    )

    assert attempt.status == "invalid_output"
    assert attempt.state is None
    assert attempt.failure_reason == "session evidence must appear in recorded user history"


def test_provider_and_invalid_json_fail_without_partial_state() -> None:
    """模型调用或结构失败时不产生半份新状态。"""

    request = PreferenceFusionRequest(
        user_id="user-1",
        session_id="session-1",
        turn_index=1,
        query_text="找个餐厅",
    )

    provider_failure = PreferenceFusion(
        FakeGenerator(failure="timeout")
    ).fuse(request)
    invalid_json = PreferenceFusion(FakeGenerator("not-json")).fuse(request)

    assert provider_failure.status == "provider_failure"
    assert provider_failure.state is None
    assert invalid_json.status == "invalid_output"
    assert invalid_json.state is None
