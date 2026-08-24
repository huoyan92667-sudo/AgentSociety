"""用真实用户、真实牛排餐厅和 DeepSeek 展示证据逐层排序。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from yelp_agent.recommendation_v2.workflow.recommendation import (
    RecommendationInput,
    build_recommendation_workflow,
)

REAL_USER_ID = "gpXLAdgNBglNB_DuQ4JFXA"
QUERY_TEXT = "这次想吃牛排"


def _evidence_rows(items) -> list[dict[str, object]]:
    """把最终附回的完整原评论整理成便于人工核对的输出。"""

    return [
        {
            "review_id": item.review_id,
            "review_time": item.review_time,
            "stars": item.stars,
            "evidence_weight": item.evidence_weight,
            "judgment_reason": item.judgment_reason,
            "review_text": item.review_text,
        }
        for item in items
    ]


def main() -> None:
    """执行一次完整真实流程并打印每个中间结果。"""

    # Windows默认控制台不能表示部分Yelp评论字符，演示输出固定使用UTF-8。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    project_root = Path(__file__).resolve().parents[4]
    with build_recommendation_workflow(project_root) as workflow:
        result = workflow.process(
            RecommendationInput(
                user_id=REAL_USER_ID,
                session_id="real-steak-soft-ranking-demo",
                query_text=QUERY_TEXT,
            )
        )

    hard_filter = result.hard_filter
    soft_ranking = result.soft_ranking
    names = (
        {}
        if hard_filter is None
        else {
            item.business.business_id: item.business.name
            for item in hard_filter.candidates
        }
    )
    passes = []
    if soft_ranking is not None:
        for ranking_pass in soft_ranking.passes:
            passes.append(
                {
                    "priority": ranking_pass.preference.priority,
                    "field": ranking_pass.preference.field,
                    "direction": ranking_pass.preference.direction,
                    "method": ranking_pass.method,
                    "order_before": [
                        {"business_id": item, "name": names[item]}
                        for item in ranking_pass.order_before
                    ],
                    "order_after": [
                        {"business_id": item, "name": names[item]}
                        for item in ranking_pass.order_after
                    ],
                    "assessments": [
                        {
                            "business_id": item.business_id,
                            "name": names[item.business_id],
                            "level": item.level,
                            "satisfaction_score": item.satisfaction_score,
                            "preference_weight": item.preference_weight,
                            "positive_weight": item.positive_weight,
                            "negative_weight": item.negative_weight,
                            "conditional_weight": item.conditional_weight,
                            "accepted_evidence_count": item.accepted_evidence_count,
                            "reason": item.reason,
                            "positive_evidence": _evidence_rows(item.positive_evidence),
                            "negative_evidence": _evidence_rows(item.negative_evidence),
                            "conditional_evidence": _evidence_rows(
                                item.conditional_evidence
                            ),
                        }
                        for item in ranking_pass.assessments
                    ],
                }
            )
    output = {
        "real_user_id": REAL_USER_ID,
        "query": QUERY_TEXT,
        "fusion_status": result.fusion.status,
        "fusion_raw_model_json": result.fusion.raw_json,
        "unified_state": (
            None
            if result.fusion.state is None
            else result.fusion.state.model_dump(mode="json")
        ),
        "hard_filter": (
            None
            if hard_filter is None
            else {
                "source_count": hard_filter.source_business_count,
                "candidate_count": hard_filter.candidate_count,
                "steps": [item.model_dump(mode="json") for item in hard_filter.steps],
            }
        ),
        "baseline": (
            None
            if soft_ranking is None or soft_ranking.baseline is None
            else {
                "source": soft_ranking.baseline.source,
                "fallback_reason": soft_ranking.baseline.fallback_reason,
                "top_10": [
                    {
                        **item.model_dump(mode="json"),
                        "name": names[item.business_id],
                    }
                    for item in soft_ranking.baseline.ranked_businesses[:10]
                ],
            }
        ),
        "soft_ranking_status": (None if soft_ranking is None else soft_ranking.status),
        "model": None if soft_ranking is None else soft_ranking.model,
        "model_call_count": (
            None if soft_ranking is None else soft_ranking.model_call_count
        ),
        "input_tokens": None if soft_ranking is None else soft_ranking.input_tokens,
        "output_tokens": (None if soft_ranking is None else soft_ranking.output_tokens),
        "passes": passes,
        "final_top_10": (
            []
            if soft_ranking is None
            else [
                {
                    "final_rank": item.final_rank,
                    "baseline_rank": item.baseline_rank,
                    "business_id": item.business.business_id,
                    "name": item.business.name,
                    "rating": item.business.rating,
                    "review_count": item.business.review_count,
                    "distance_km": item.distance_km,
                    "combined_preference_score": item.combined_preference_score,
                    "preference_scores": item.preference_scores,
                    "preference_levels": item.preference_levels,
                }
                for item in soft_ranking.ranking
            ]
        ),
        "raw_evidence_model_json": (
            [] if soft_ranking is None else soft_ranking.raw_model_outputs
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
