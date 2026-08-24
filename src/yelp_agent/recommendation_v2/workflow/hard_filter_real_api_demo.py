"""用真实用户画像、真实大模型和真实餐厅事实演示结构化硬过滤。"""

from __future__ import annotations

import json
from pathlib import Path

from yelp_agent.recommendation_v2.workflow.recommendation import (
    RecommendationInput,
    build_recommendation_workflow,
)

REAL_USER_ID = "ET8n-r7glWYqZhuR6GcdNw"
DEMO_QUERY = (
    "今天想吃川菜，距离必须在5公里以内，人均价格最多二档，评分至少4分。"
    "硬条件满足以后，优先环境安静，其次距离近。"
)


def main() -> None:
    """逐段打印模型理解、程序补齐、地点计算和真实过滤结果。"""

    project_root = Path(__file__).resolve().parents[4]
    with build_recommendation_workflow(project_root) as workflow:
        result = workflow.process(
            RecommendationInput(
                user_id=REAL_USER_ID,
                session_id="hard-filter-real-api-demo",
                query_text=DEMO_QUERY,
            )
        )

    state = result.fusion.state
    hard_filter = result.hard_filter
    geography = result.geography
    payload = {
        "original_query": DEMO_QUERY,
        "model_raw_output": result.fusion.raw_json,
        "fusion_status": result.fusion.status,
        "failure_reason": result.fusion.failure_reason,
        "model": result.fusion.model,
        "model_call_count": result.fusion.model_call_count,
        "program_completed_state": (
            None
            if state is None
            else {
                "search_center": state.search_center.model_dump(mode="json")
                if state.search_center is not None
                else None,
                "hard_constraints": [
                    item.model_dump(mode="json") for item in state.hard_constraints
                ],
                "default_constraints": [
                    item.model_dump(mode="json")
                    for item in state.default_constraints
                ],
                "soft_preferences": [
                    item.model_dump(mode="json") for item in state.soft_preferences
                ],
            }
        ),
        "geography": (
            None
            if geography is None
            else {
                "search_center": geography.search_center.model_dump(mode="json"),
                "calculated_business_count": geography.source_business_count,
            }
        ),
        "hard_filter": (
            None
            if hard_filter is None
            else {
                "source_business_count": hard_filter.source_business_count,
                "candidate_count": hard_filter.candidate_count,
                "steps": [item.model_dump(mode="json") for item in hard_filter.steps],
                "generated_sql": hard_filter.generated_sql,
                "sql_parameters": hard_filter.sql_parameters,
                "first_ten_candidates": [
                    {
                        "business_id": item.business.business_id,
                        "name": item.business.name,
                        "categories": item.business.categories,
                        "distance_km": item.distance_km,
                        "price_level": item.business.price_level,
                        "rating": item.business.rating,
                        "review_count": item.business.review_count,
                    }
                    for item in hard_filter.candidates[:10]
                ],
            }
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
