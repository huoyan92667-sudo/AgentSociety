"""运行一个真实的四来源融合示例，并把输入、中间结果和最终状态全部打印出来。"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any, cast

from yelp_agent.recommendation_v2.preference_fusion.fusion import (
    ConversationHistoryTurn,
    PreferenceFusionProposal,
    PreferenceFusionRequest,
)
from yelp_agent.recommendation_v2.preference_fusion.profile_adapter import (
    ProfilePreferenceSet,
)
from yelp_agent.recommendation_v2.preference_fusion.runtime import (
    build_preference_fusion,
)
from yelp_agent.recommendation_v2.scenes import get_scene_baseline
from yelp_agent.recommendation_v2.schema import (
    GeoPoint,
    PreferenceMemoryItem,
    SoftPreference,
    UnifiedRecommendationState,
)


def _dialogue_preference(
    *,
    key: str,
    field: str,
    direction: str,
    target_value: object,
    strength: int,
    priority: int,
    evidence: str,
) -> SoftPreference:
    """建立上一轮已经确认过的一条要求。"""

    feature = {"distance_km": "coordinates"}.get(field, field)
    return SoftPreference.model_validate(
        {
            "key": key,
            "field": field,
            "direction": direction,
            "target_value": target_value,
            "preference_strength": strength,
            "priority": priority,
            "merchant_feature": feature,
            "controlling_source": "current_query",
            "sources": [
                {
                    "source": "current_query",
                    "text": evidence,
                    "turn_index": 1,
                    "preference_strength": strength,
                }
            ],
        }
    )


def _profile_preference(
    *,
    key: str,
    field: str,
    direction: str,
    target_value: object,
    strength: int,
    priority: int,
    text: str,
) -> SoftPreference:
    """建立一条带真实历史次数和最近确认时间的长期偏好。"""

    feature = {"category": "categories"}.get(field, field)
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
                    "text": text,
                    "preference_strength": strength,
                    "profile_score": 0.8,
                    "profile_evidence_count": 8,
                    "profile_last_confirmed": datetime(
                        2026,
                        8,
                        10,
                        tzinfo=UTC,
                    ),
                }
            ],
        }
    )


def build_demo_request() -> PreferenceFusionRequest:
    """构造本轮、历史、画像和场景之间存在真实冲突的测试输入。"""

    previous_preferences = [
        _dialogue_preference(
            key="distance.near",
            field="distance_km",
            direction="lower",
            target_value=None,
            strength=75,
            priority=1,
            evidence="距离近一点优先",
        ),
        _dialogue_preference(
            key="price_level.previous",
            field="price_level",
            direction="closer_to",
            target_value=2,
            strength=50,
            priority=2,
            evidence="价格二档左右",
        ),
    ]
    previous_state = UnifiedRecommendationState(
        user_id="demo-user",
        session_id="demo-session",
        revision=1,
        turn_index=1,
        latest_query_text="距离近一点优先，价格二档左右。",
        user_location=GeoPoint(latitude=39.9526, longitude=-75.1652),
        soft_preferences=previous_preferences,
        preference_memory=[
            PreferenceMemoryItem(
                candidate_id=f"turn1:{item.key}",
                source="current_query",
                preference=item,
                status="active",
                reason="上一轮明确提出",
            )
            for item in previous_preferences
        ],
    )
    profile = ProfilePreferenceSet(
        profile_id="a" * 64,
        user_id="demo-user",
        soft_preferences=[
            _profile_preference(
                key="profile.category.japanese",
                field="category",
                direction="match",
                target_value=["Japanese"],
                strength=75,
                priority=1,
                text="长期画像：更常选择日料",
            ),
            _profile_preference(
                key="profile.spiciness.lower",
                field="spiciness",
                direction="lower",
                target_value=None,
                strength=75,
                priority=2,
                text="长期画像：平时偏清淡",
            ),
            _profile_preference(
                key="profile.quiet.higher",
                field="quiet_environment",
                direction="higher",
                target_value=None,
                strength=50,
                priority=3,
                text="长期画像：更喜欢安静环境",
            ),
        ],
    )
    return PreferenceFusionRequest(
        user_id="demo-user",
        session_id="demo-session",
        turn_index=2,
        query_text=(
            "今天和朋友聚餐，想吃川菜，辣一点，安静最重要，距离5公里以内。"
        ),
        previous_state=previous_state,
        conversation_history=[
            ConversationHistoryTurn(
                turn_index=1,
                user_message="距离近一点优先，价格二档左右。",
            )
        ],
        scene_baseline=get_scene_baseline("friends"),
        profile_preferences=profile,
        user_location=GeoPoint(latitude=39.9526, longitude=-75.1652),
    )


def _group_model_sources(
    request: PreferenceFusionRequest,
    proposal: PreferenceFusionProposal,
) -> dict[str, Any]:
    """按证据轮次展示大模型整理出的本轮与历史会话要求。"""

    dialogue = [
        *proposal.hard_constraints,
        *proposal.soft_preferences,
        *proposal.open_requirements,
    ]
    return {
        "current_query": [
            item.model_dump(mode="json")
            for item in dialogue
            if item.evidence_turn_index == request.turn_index
        ],
        "session": [
            item.model_dump(mode="json")
            for item in dialogue
            if item.evidence_turn_index < request.turn_index
        ],
        "user_profile": "由程序直接从画像输入补齐，不要求大模型抄写",
        "scene": None if proposal.scene is None else proposal.scene.model_dump(mode="json"),
    }


def _compact_preference(preference: SoftPreference) -> dict[str, Any]:
    """保留肉眼判断融合结果真正需要的偏好字段。"""

    return {
        "key": preference.key,
        "field": preference.field,
        "direction": preference.direction,
        "target_value": preference.target_value,
        "preference_strength": preference.preference_strength,
        "priority": preference.priority,
        "controlling_source": preference.controlling_source,
        "sources": [
            {
                "source": item.source,
                "text": item.text,
                "turn_index": item.turn_index,
                "preference_strength": item.preference_strength,
            }
            for item in preference.sources
        ],
    }


def _compact_result(
    request: PreferenceFusionRequest,
    proposal: PreferenceFusionProposal | None,
    raw_json: str | None,
    attempt_status: str,
    failure_reason: str | None,
    model: str | None,
    final_state: UnifiedRecommendationState | None,
) -> dict[str, Any]:
    """生成适合讨论的短版 JSON，不改变实际调用和融合过程。"""

    previous = cast(UnifiedRecommendationState, request.previous_state)
    grouped: dict[str, list[dict[str, Any]]] | None = None
    if proposal is not None:
        grouped = _group_model_sources(request, proposal)
    final: dict[str, Any] | None = None
    if final_state is not None:
        final = {
            "scene": None
            if final_state.scene is None
            else final_state.scene.model_dump(mode="json"),
            "hard_constraints": [
                {
                    "key": item.key,
                    "field": item.field,
                    "operator": item.operator,
                    "value": item.value,
                    "controlling_source": item.controlling_source,
                }
                for item in final_state.hard_constraints
            ],
            "default_constraints": [
                item.model_dump(mode="json")
                for item in final_state.default_constraints
            ],
            "soft_preferences": [
                _compact_preference(item) for item in final_state.soft_preferences
            ],
            "preference_memory": [
                {
                    "candidate_id": item.candidate_id,
                    "source": item.source,
                    "field": item.preference.field,
                    "direction": item.preference.direction,
                    "target_value": item.preference.target_value,
                    "status": item.status,
                    "reason": item.reason,
                    "controller_candidate_id": item.controller_candidate_id,
                }
                for item in final_state.preference_memory
            ],
        }
    return {
        "四份原始输入": {
            "current_query": {"query_text": request.query_text},
            "session": [
                _compact_preference(item) for item in previous.soft_preferences
            ],
            "user_profile": [
                _compact_preference(item)
                for item in cast(
                    ProfilePreferenceSet,
                    request.profile_preferences,
                ).soft_preferences
            ],
            "scene": {
                "scene": request.scene_baseline.scene,
                "default_constraints": [
                    item.model_dump(mode="json")
                    for item in request.scene_baseline.default_constraints
                ],
                "soft_preferences": [
                    _compact_preference(item)
                    for item in request.scene_baseline.soft_preferences
                ],
            },
        },
        "大模型拆出的四路结构": grouped,
        # 这里故意保留供应商返回的原始字符串，不用解析后的默认值冒充原文。
        "大模型原始输出": raw_json,
        "处理结果": {
            "status": attempt_status,
            "failure_reason": failure_reason,
            "model": model,
        },
        "最终JSON": final,
    }


def main() -> None:
    """调用项目当前配置的大模型，并打印完整可复查结果。"""

    request = build_demo_request()
    attempt = build_preference_fusion().fuse(request)
    raw_output: object = attempt.raw_json
    grouped: object = None
    parsed_proposal: PreferenceFusionProposal | None = None
    if attempt.raw_json is not None:
        try:
            parsed = PreferenceFusionProposal.model_validate_json(attempt.raw_json)
            parsed_proposal = parsed
            raw_output = parsed.model_dump(mode="json")
            grouped = _group_model_sources(request, parsed)
        except ValueError:
            pass

    if "--compact" in sys.argv:
        print(
            json.dumps(
                _compact_result(
                    request,
                    parsed_proposal,
                    attempt.raw_json,
                    attempt.status,
                    attempt.failure_reason,
                    attempt.model,
                    attempt.state,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    result = {
        "四份原始输入": {
            "本轮用户原话": request.query_text,
            "上一轮完整状态": cast(
                UnifiedRecommendationState,
                request.previous_state,
            ).model_dump(mode="json"),
            "场景初始值": request.scene_baseline.model_dump(mode="json")
            if request.scene_baseline is not None
            else None,
            "长期画像": request.profile_preferences.model_dump(mode="json")
            if request.profile_preferences is not None
            else None,
        },
        "大模型拆出的四路结构": grouped,
        "大模型原始返回": raw_output,
        "处理结果": {
            "status": attempt.status,
            "failure_reason": attempt.failure_reason,
            "model": attempt.model,
            "latency_ms": attempt.latency_ms,
            "input_tokens": attempt.input_tokens,
            "output_tokens": attempt.output_tokens,
        },
        "最终完整状态": (
            None if attempt.state is None else attempt.state.model_dump(mode="json")
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
