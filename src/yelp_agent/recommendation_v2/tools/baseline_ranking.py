"""把硬筛选结果适配给旧版个性化排序模型。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from yelp_agent.agent_tools import OnlineHybridV2RankingService
from yelp_agent.business_profiles import BusinessKnowledgeStore
from yelp_agent.config import load_business_profile_config, load_config
from yelp_agent.learning_to_rank import FrozenLambdaMARTRanker
from yelp_agent.learning_to_rank.features import HybridV1Weights
from yelp_agent.profiles.store import UserProfileStore
from yelp_agent.recommendation_v2.soft_ranking.schema import (
    BaselineRankedBusiness,
    BaselineRankingResult,
)

from .hard_filter import FilteredBusiness, StructuredHardFilterResult


class RatingBaselineRankingTool:
    """不用旧画像模型，直接按评分、评论数和距离产生稳定基础顺序。"""

    name = "build_rating_baseline_ranking"

    def execute(
        self,
        *,
        hard_filter: StructuredHardFilterResult,
        **_: object,
    ) -> BaselineRankingResult:
        """评分高者优先；同分再看评论数，最后用距离和编号稳定排序。"""

        ordered = sorted(
            hard_filter.candidates,
            key=lambda item: (
                -item.business.rating,
                -item.business.review_count,
                math.inf if item.distance_km is None else item.distance_km,
                item.business.business_id,
            ),
        )
        return BaselineRankingResult(
            source="rating",
            ranked_businesses=[
                BaselineRankedBusiness(
                    business_id=item.business.business_id,
                    baseline_rank=rank,
                )
                for rank, item in enumerate(ordered, 1)
            ],
        )


class OldRankingService(Protocol):
    """旧排序模型已经稳定下来的在线调用形状。"""

    def rank(
        self,
        *,
        request_id: str,
        user_id: str,
        cutoff_time: datetime,
        candidates: Sequence[Mapping[str, object]],
    ) -> Sequence[Mapping[str, object]]: ...


class LegacyBaselineRankingTool:
    """先用旧排序产生基础顺序；失败时保留一个可重复的事实顺序。"""

    name = "build_legacy_baseline_ranking"

    def __init__(self, service: OldRankingService) -> None:
        self._service = service

    def execute(
        self,
        *,
        request_id: str,
        user_id: str,
        cutoff_time: datetime,
        hard_filter: StructuredHardFilterResult,
    ) -> BaselineRankingResult:
        """把真实商家事实转成旧模型需要的候选特征并生成完整顺序。"""

        candidates = hard_filter.candidates
        if not candidates:
            return BaselineRankingResult(
                source="legacy_hybrid_v2",
                ranked_businesses=[],
            )
        feature_rows = self._candidate_features(candidates)
        try:
            ranked = list(
                self._service.rank(
                    request_id=request_id,
                    user_id=user_id,
                    cutoff_time=cutoff_time,
                    candidates=feature_rows,
                )
            )
            expected = [item.business.business_id for item in candidates]
            actual = [str(item.get("business_id") or "") for item in ranked]
            if len(actual) != len(expected) or set(actual) != set(expected):
                raise ValueError(
                    "old ranking did not return a complete candidate order"
                )
            return BaselineRankingResult(
                source="legacy_hybrid_v2",
                ranked_businesses=[
                    BaselineRankedBusiness(
                        business_id=business_id,
                        baseline_rank=index,
                        model_rank=int(row["model_rank"]),
                        old_hybrid_rank=int(row["hybrid_v1_rank"]),
                        model_score=float(row["model_score"]),
                        old_hybrid_score=float(row["hybrid_v1_score"]),
                        blend_score=float(row["blend_score"]),
                    )
                    for index, (business_id, row) in enumerate(
                        zip(actual, ranked, strict=True),
                        start=1,
                    )
                ],
            )
        # 旧模型的读取、特征或模型产物任一层失败，都不能让新版推荐中断。
        except Exception as exc:  # noqa: BLE001
            return self._fallback(candidates, reason=str(exc))

    @staticmethod
    def _candidate_features(
        candidates: list[FilteredBusiness],
    ) -> list[dict[str, object]]:
        """补齐旧模型需要、但硬筛选结果本身没有保存的基础路线特征。"""

        ids = [item.business.business_id for item in candidates]
        quality_scores = {
            item.business.business_id: (item.business.rating - 1.0) / 4.0
            for item in candidates
        }
        location_scores = {
            item.business.business_id: (
                0.0 if item.distance_km is None else math.exp(-item.distance_km / 5.0)
            )
            for item in candidates
        }

        def rank_map(scores: dict[str, float]) -> dict[str, int]:
            ordered = sorted(ids, key=lambda value: (-scores[value], value))
            return {business_id: rank for rank, business_id in enumerate(ordered, 1)}

        quality_ranks = rank_map(quality_scores)
        location_ranks = rank_map(location_scores)
        rows: list[dict[str, object]] = []
        for item in candidates:
            business_id = item.business.business_id
            quality_rank = quality_ranks[business_id]
            location_rank = (
                None if item.distance_km is None else location_ranks[business_id]
            )
            route_count = 2 + int(location_rank is not None)
            fusion_score = (
                1.0 / (60 + quality_rank)
                + 1.0 / (60 + 1)  # 硬筛后类别都匹配，类别路线视为并列第一。
                + (0.0 if location_rank is None else 1.0 / (60 + location_rank))
            )
            rows.append(
                {
                    "rank": quality_rank,
                    "business_id": business_id,
                    "fusion_score": fusion_score,
                    "route_count": route_count,
                    "quality_rank": quality_rank,
                    "quality_score": quality_scores[business_id],
                    "category_rank": 1,
                    "category_score": 1.0,
                    "text_rank": None,
                    "text_score": 0.0,
                    "location_rank": location_rank,
                    "location_score": location_scores[business_id],
                    "distance_km": item.distance_km,
                    "item_knn_positive_score": 0.0,
                    "item_knn_negative_evidence": 0.0,
                    "item_knn_positive_support_count": 0,
                    "item_knn_negative_support_count": 0,
                    "item_knn_positive_neighbor_count": 0,
                    "item_knn_negative_neighbor_count": 0,
                    "item_knn_missing": True,
                }
            )
        return rows

    @staticmethod
    def _fallback(
        candidates: list[FilteredBusiness],
        *,
        reason: str,
    ) -> BaselineRankingResult:
        """旧模型不可用时按评分、评论数、距离得到稳定基础顺序。"""

        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.business.rating,
                -item.business.review_count,
                math.inf if item.distance_km is None else item.distance_km,
                item.business.business_id,
            ),
        )
        return BaselineRankingResult(
            source="fact_fallback",
            fallback_reason=reason[:500] or "legacy ranking failed",
            ranked_businesses=[
                BaselineRankedBusiness(
                    business_id=item.business.business_id,
                    baseline_rank=rank,
                )
                for rank, item in enumerate(ordered, 1)
            ],
        )


def build_legacy_baseline_ranking_tool(
    project_root: str | Path,
    profile_store: UserProfileStore,
) -> LegacyBaselineRankingTool:
    """从项目冻结产物建立旧排序适配工具，不启动整个旧 Agent。"""

    root = Path(project_root)
    app_config = load_config(root / "configs")
    business_config = load_business_profile_config(root / "configs")
    frozen_weights = json.loads(
        (root / "runs" / "hybrid" / "hybrid_weights.json").read_text(encoding="utf-8")
    )
    service = OnlineHybridV2RankingService(
        user_profiles=profile_store,
        business_profiles=BusinessKnowledgeStore.from_artifacts(
            root / "data" / "features" / "business_profiles" / "v1",
            config=business_config,
        ),
        ranker=FrozenLambdaMARTRanker.from_artifacts(
            root / "runs" / "hybrid_v2_b" / "frozen"
        ),
        weights=HybridV1Weights.model_validate(frozen_weights["selected_weights"]),
        broad_categories=set(app_config.data.broad_categories),
        business_profile_config=business_config,
    )
    return LegacyBaselineRankingTool(service)
