"""真实执行费城唐人街约会川菜查询，并保存最终自然推荐和证据。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from yelp_agent.recommendation_v2.workflow import (
    RecommendationInput,
    RecommendationTurnResult,
    build_recommendation_workflow,
)

from .synthesizer import build_recommendation_answer_synthesizer

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


def _run_full_workflow() -> RecommendationTurnResult:
    """调用真实模型完成需求理解、评论检索、排序和最终总结。"""

    request_time = datetime(
        2026,
        8,
        25,
        12,
        0,
        tzinfo=ZoneInfo("America/New_York"),
    )
    with build_recommendation_workflow(_PROJECT_ROOT) as workflow:
        return workflow.process(
            RecommendationInput(
                user_id=_REAL_USER_ID,
                session_id="chinatown-szechuan-answer-demo",
                query_text=_QUERY,
                request_time=request_time,
            )
        )


def _rerun_answer_only() -> RecommendationTurnResult:
    """复用已经保存的排序和证据，只重新测试最终自然回答。"""

    saved = RecommendationTurnResult.model_validate_json(
        _OUTPUT.read_text(encoding="utf-8")
    )
    if saved.fusion.state is None or saved.review_evidence_ranking is None:
        raise RuntimeError("saved result has no state or evidence ranking")
    answer = build_recommendation_answer_synthesizer().synthesize(
        query_text=_QUERY,
        state=saved.fusion.state,
        ranking=saved.review_evidence_ranking,
    )
    return saved.model_copy(update={"answer": answer}, deep=True)


def _save_and_print(result: RecommendationTurnResult) -> None:
    """统一保存完整结果，并向终端打印便于人工核对的紧凑摘要。"""

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


def main(argv: list[str] | None = None) -> None:
    """默认跑完整流程；--answer-only只复用现有Top5重新总结。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-only", action="store_true")
    args = parser.parse_args(argv)
    result = _rerun_answer_only() if args.answer_only else _run_full_workflow()
    _save_and_print(result)


if __name__ == "__main__":
    main()
