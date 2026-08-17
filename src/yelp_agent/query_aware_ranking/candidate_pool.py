"""Protected union of the frozen History and Query Top-500 channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from yelp_agent.query_retrieval import QueryRetrievalResult
from yelp_agent.retrieval import RetrievalCandidate, RetrievalResult

from .schema import CandidatePoolItem, HardConstraintExclusion


@dataclass(frozen=True, slots=True)
class ProtectedCandidatePoolResult:
    items: tuple[CandidatePoolItem, ...]
    history_candidates: dict[str, RetrievalCandidate]
    exclusions: tuple[HardConstraintExclusion, ...]
    union_candidate_count: int
    overlap_count: int


class ProtectedCandidatePool:
    """Keep both final retrieval channels until the learned coarse ranker runs."""

    def __init__(self, *, candidate_limit: int) -> None:
        if candidate_limit < 500:
            raise ValueError("protected union limit must be at least 500")
        self._candidate_limit = candidate_limit

    def build(
        self,
        history: RetrievalResult,
        query: QueryRetrievalResult,
        *,
        additional_exclusions: set[str] | frozenset[str] = frozenset(),
    ) -> ProtectedCandidatePoolResult:
        history_by_id = {item.business_id: item for item in history.candidates}
        query_by_id = {item.business_id: item for item in query.candidates}
        union_ids = set(history_by_id).union(query_by_id)
        if len(union_ids) > self._candidate_limit:
            raise ValueError(
                f"protected union contains {len(union_ids)} candidates, "
                f"above configured limit {self._candidate_limit}"
            )
        excluded_by_id = {
            item.business_id: HardConstraintExclusion(
                business_id=item.business_id,
                reason_codes=list(item.reason_codes),
            )
            for item in query.excluded
            if item.business_id in union_ids
        }
        for business_id in sorted(additional_exclusions.intersection(union_ids)):
            existing = excluded_by_id.get(business_id)
            reason_codes = (
                [] if existing is None else list(existing.reason_codes)
            )
            if "SESSION_REJECTED" not in reason_codes:
                reason_codes.append("SESSION_REJECTED")
            excluded_by_id[business_id] = HardConstraintExclusion(
                business_id=business_id,
                reason_codes=reason_codes,
            )
        eligible_ids = sorted(union_ids.difference(excluded_by_id))
        items = []
        for business_id in eligible_ids:
            history_item = history_by_id.get(business_id)
            query_item = query_by_id.get(business_id)
            channels = []
            if history_item is not None:
                channels.append("history")
            if query_item is not None:
                channels.append("query")
            items.append(
                CandidatePoolItem(
                    business_id=business_id,
                    source_channels=channels,
                    history_rank=(None if history_item is None else history_item.rank),
                    history_fusion_score=(
                        None if history_item is None else history_item.fusion_score
                    ),
                    query_rank=None if query_item is None else query_item.rank,
                    query_fusion_score=(
                        None if query_item is None else query_item.fusion_score
                    ),
                )
            )
        if not items:
            raise ValueError("hard constraints removed the complete protected union")
        return ProtectedCandidatePoolResult(
            items=tuple(items),
            history_candidates=history_by_id,
            exclusions=tuple(
                excluded_by_id[business_id] for business_id in sorted(excluded_by_id)
            ),
            union_candidate_count=len(union_ids),
            overlap_count=len(set(history_by_id).intersection(query_by_id)),
        )


def hybrid_candidate_rows(
    pool: ProtectedCandidatePoolResult,
    *,
    history_candidate_count: int,
) -> list[dict[str, object]]:
    """Adapt Query-only candidates to the frozen history-feature contract."""

    missing_rank = history_candidate_count + 1
    rows: list[dict[str, object]] = []
    for item in pool.items:
        history = pool.history_candidates.get(item.business_id)
        if history is not None:
            rows.append(asdict(history))
            continue
        rows.append(
            {
                "business_id": item.business_id,
                "rank": missing_rank,
                "fusion_score": 0.0,
                "route_count": 0,
                "quality_rank": None,
                "quality_score": None,
                "category_rank": None,
                "category_score": None,
                "text_rank": None,
                "text_score": None,
                "location_rank": None,
                "location_score": None,
                "distance_km": None,
                "item_knn_rank": None,
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
