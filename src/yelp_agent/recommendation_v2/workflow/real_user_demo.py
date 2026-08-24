"""用项目真实用户画像和真实模型运行最小推荐入口。"""

from __future__ import annotations

import json
from pathlib import Path

from yelp_agent.profiles.schema import UserProfileV1
from yelp_agent.recommendation_v2.preference_fusion import ProfilePreferenceSet
from yelp_agent.recommendation_v2.workflow.recommendation import (
    RecommendationInput,
    build_recommendation_workflow,
)

REAL_USER_ID = "ET8n-r7glWYqZhuR6GcdNw"
DEMO_QUERY = "今天和朋友聚餐，想吃川菜，辣一点，安静最重要，距离5公里以内。"


def _raw_profile_summary(profile: UserProfileV1) -> dict[str, object]:
    """展示真实画像的来源、规模和最有代表性的内容，避免打印几千行。"""

    data = profile.model_dump(mode="json")
    return {
        "profile_id": data["profile_id"],
        "user_id": data["user_id"],
        "cutoff_time": data["cutoff_time"],
        "history_length": data["history_length"],
        "average_rating": data["average_rating"],
        "reliability": data["reliability"],
        "location_center": data["location_center"],
        "category_preference_count": len(data["category_preferences"]),
        "category_preference_examples": data["category_preferences"][:5],
        "category_dislike_count": len(data["category_dislikes"]),
        "aspect_preferences": data["aspect_preferences"],
        "aspect_dislikes": data["aspect_dislikes"],
        "price_preference": data["price_preference"],
        "frequent_areas": data["frequent_areas"],
    }


def _adapted_profile_summary(profile: ProfilePreferenceSet) -> dict[str, object]:
    """展示画像转换模块确实产生了独立偏好，而不是压成一条。"""

    data = profile.model_dump(mode="json")
    preferences = data["soft_preferences"]
    return {
        "profile_id": data["profile_id"],
        "soft_preference_count": len(preferences),
        "first_twelve_preferences": [
            {
                "field": item["field"],
                "direction": item["direction"],
                "target_value": item["target_value"],
                "preference_strength": item["preference_strength"],
                "priority": item["priority"],
                "evidence": item["sources"][0],
            }
            for item in preferences[:12]
        ],
        "ignored_signals": data["ignored_signals"],
    }


def _final_state_summary(state_data: dict[str, object]) -> dict[str, object]:
    """分别展示直接用于过滤、排序和冲突追踪的最终内容。"""

    soft_preferences = state_data["soft_preferences"]
    preference_memory = state_data["preference_memory"]
    dialogue_soft = [
        item
        for item in soft_preferences
        if item["controlling_source"] in {"current_query", "session"}
    ]
    category_memory = [
        {
            "candidate_id": item["candidate_id"],
            "source": item["source"],
            "target_value": item["preference"]["target_value"],
            "status": item["status"],
            "reason": item["reason"],
        }
        for item in preference_memory
        if item["preference"]["field"] == "category"
    ]
    return {
        "scene": state_data["scene"],
        "user_location": state_data["user_location"],
        "search_center": state_data["search_center"],
        "hard_constraints_for_filtering": state_data["hard_constraints"],
        "scene_default_constraints": state_data["default_constraints"],
        "dialogue_soft_preferences_for_ranking": dialogue_soft,
        "all_active_soft_preference_count": len(soft_preferences),
        "first_twelve_active_soft_preferences": soft_preferences[:12],
        "category_conflict_results": category_memory,
        "open_requirements": state_data["open_requirements"],
    }


def main() -> None:
    """运行一次真实请求并依次打印输入、画像转换、模型输出和最终状态。"""

    project_root = Path(__file__).resolve().parents[4]
    with build_recommendation_workflow(project_root) as workflow:
        result = workflow.process(
            RecommendationInput(
                user_id=REAL_USER_ID,
                session_id="real-user-demo-session",
                query_text=DEMO_QUERY,
            )
        )
    raw_profile = result.raw_profile
    adapted_profile = result.adapted_profile
    payload = {
        "入口输入": {
            "user_id": REAL_USER_ID,
            "session_id": "real-user-demo-session",
            "query": DEMO_QUERY,
        },
        "真实原始长期画像": (
            None if raw_profile is None else _raw_profile_summary(raw_profile)
        ),
        "画像转换后的统一结构": (
            None
            if adapted_profile is None
            else _adapted_profile_summary(adapted_profile)
        ),
        "大模型原始输出": result.fusion.raw_json,
        "处理结果": {
            "status": result.fusion.status,
            "failure_reason": result.fusion.failure_reason,
            "model": result.fusion.model,
            "model_call_count": result.fusion.model_call_count,
        },
        "程序补齐后的最终关键状态": (
            None
            if result.fusion.state is None
            else _final_state_summary(result.fusion.state.model_dump(mode="json"))
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
