"""真实执行费城唐人街约会川菜查询，并保存最终自然推荐和证据。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
    / "chinatown_szechuan_result.json"
)
_REAL_USER_ID = "gpXLAdgNBglNB_DuQ4JFXA"
_QUERY = "我今天晚上9点想和我女朋友去费城唐人街吃川菜，要地道的川菜"


def main() -> None:
    """调用真实模型完成需求理解、检索说法生成和Top5自然总结。"""

    request_time = datetime(
        2026,
        8,
        25,
        12,
        0,
        tzinfo=ZoneInfo("America/New_York"),
    )
    with build_recommendation_workflow(_PROJECT_ROOT) as workflow:
        result = workflow.process(
            RecommendationInput(
                user_id=_REAL_USER_ID,
                session_id="chinatown-szechuan-answer-demo",
                query_text=_QUERY,
                request_time=request_time,
            )
        )

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"result_file={_OUTPUT.resolve()}")
    print(f"fusion_status={result.fusion.status}")
    if result.fusion.state is not None:
        print(
            "search_center="
            + json.dumps(
                result.fusion.state.search_center.model_dump(mode="json")
                if result.fusion.state.search_center is not None
                else None,
                ensure_ascii=False,
            )
        )
        print(
            "hard_constraints="
            + json.dumps(
                [
                    item.model_dump(mode="json")
                    for item in result.fusion.state.hard_constraints
                ],
                ensure_ascii=False,
            )
        )
    if result.hard_filter is not None:
        print(f"hard_filter_count={result.hard_filter.candidate_count}")
        for step in result.hard_filter.steps:
            print(
                f"  {step.field}: {step.before_count}->{step.after_count} "
                f"unknown={step.unknown_excluded_count}"
            )
    if result.review_evidence_ranking is not None:
        print(f"ranking_status={result.review_evidence_ranking.status}")
        for item in result.review_evidence_ranking.ranking:
            print(
                f"TOP{item.final_rank} {item.business.name} "
                f"final={item.final_score:.4f} distance={item.distance_km}"
            )
    if result.answer is not None:
        print(f"answer_status={result.answer.status}")
        print(result.answer.text or result.answer.failure_reason)


if __name__ == "__main__":
    main()
