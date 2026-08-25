"""真实执行“我想吃牛排”，保存四路融合、硬筛和评论证据 Top5。"""

from __future__ import annotations

import json
from pathlib import Path

from yelp_agent.recommendation_v2.workflow import (
    RecommendationInput,
    build_recommendation_workflow,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "review_evidence"
    / "v1"
    / "runs"
    / "steak_query_result.json"
)
_REAL_USER_ID = "gpXLAdgNBglNB_DuQ4JFXA"


def main() -> None:
    """调用真实 DeepSeek 做问题理解和四路融合，再执行本地硬筛与证据排序。"""

    with build_recommendation_workflow(_PROJECT_ROOT) as workflow:
        result = workflow.process(
            RecommendationInput(
                user_id=_REAL_USER_ID,
                session_id="review-evidence-steak-demo",
                query_text="我想吃牛排",
            )
        )

    payload = result.model_dump(mode="json")
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ranking = result.review_evidence_ranking
    if ranking is None:
        raise RuntimeError("review evidence ranking did not run")
    print(f"result_file={_OUTPUT.resolve()}")
    print(f"fusion_status={result.fusion.status}")
    print(
        "hard_filter_count="
        f"{result.hard_filter.candidate_count if result.hard_filter else 'not_run'}"
    )
    print(f"review_ranking_status={ranking.status}")
    for item in ranking.ranking:
        print(
            f"TOP{item.final_rank} {item.business.name} "
            f"final={item.final_score:.4f} preference={item.preference_score:.4f} "
            f"rating={item.business.rating:.1f} distance={item.distance_km}"
        )
        for assessment in item.preference_evidence:
            positive = assessment.positive_evidence[:1]
            negative = assessment.negative_evidence[:1]
            if positive:
                print(
                    f"  + {assessment.requirement_id}: "
                    f"{positive[0].matched_segment_text[:240]}"
                )
            if negative:
                print(
                    f"  - {assessment.requirement_id}: "
                    f"{negative[0].matched_segment_text[:240]}"
                )


if __name__ == "__main__":
    main()
