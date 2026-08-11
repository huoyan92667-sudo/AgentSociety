"""Aggressive replacement and bounded conservative fusion policies."""

from __future__ import annotations

from collections.abc import Sequence

from .config import SemanticRankingPolicy
from .schema import CandidateSemanticScore, RankingIntent, SemanticRankingMode


def _percentile(rank: int, size: int) -> float:
    return 1.0 if size == 1 else (size - rank) / (size - 1)


def _bounded_order(
    rows: Sequence[CandidateSemanticScore],
    *,
    maximum_upward_move: int,
    maximum_downward_move: int,
) -> list[str]:
    remaining = {item.business_id: item for item in rows}
    result: list[str] = []
    for position in range(1, len(rows) + 1):
        forced = [
            item
            for item in remaining.values()
            if item.base_rank + maximum_downward_move <= position
        ]
        available = [
            item
            for item in remaining.values()
            if item.base_rank <= position + maximum_upward_move
        ]
        choices = forced or available
        if not choices:
            choices = list(remaining.values())
        selected = min(
            choices,
            key=lambda item: (-item.fused_score, item.business_id),
        )
        result.append(selected.business_id)
        remaining.pop(selected.business_id)
    return result


def _protect_top_k(
    ranking: list[str],
    rows_by_id: dict[str, CandidateSemanticScore],
    policy: SemanticRankingPolicy,
) -> tuple[list[str], dict[str, list[str]]]:
    if policy.protected_top_k == 0:
        return ranking, {}
    result = list(ranking)
    reasons: dict[str, list[str]] = {}
    originally_protected = {
        item.business_id
        for item in rows_by_id.values()
        if item.base_rank <= policy.protected_top_k
    }
    for protected_id in sorted(
        originally_protected,
        key=lambda business_id: rows_by_id[business_id].base_rank,
    ):
        current = result.index(protected_id)
        if current < policy.protected_top_k:
            continue
        challengers = [
            business_id
            for business_id in result[: policy.protected_top_k]
            if business_id not in originally_protected
        ]
        if not challengers:
            continue
        challenger = min(
            challengers,
            key=lambda business_id: (
                rows_by_id[business_id].semantic_score,
                business_id,
            ),
        )
        challenger_row = rows_by_id[challenger]
        protected_row = rows_by_id[protected_id]
        sufficient_margin = (
            challenger_row.semantic_score - protected_row.semantic_score
            >= policy.displacement_margin
        )
        sufficient_evidence = (
            challenger_row.evidence_coverage >= policy.evidence_floor
            and any(match.status == "matched" for match in challenger_row.matches)
        )
        if sufficient_margin and sufficient_evidence:
            continue
        challenger_index = result.index(challenger)
        result[challenger_index], result[current] = (
            result[current],
            result[challenger_index],
        )
        reasons.setdefault(protected_id, []).append("TOP_K_BASE_RANK_PROTECTED")
        reasons.setdefault(challenger, []).append("TOP_K_DISPLACEMENT_REJECTED")
    return result, reasons


def apply_ranking_policy(
    *,
    base_ranking: Sequence[str],
    rows: Sequence[CandidateSemanticScore],
    intent: RankingIntent,
    mode: SemanticRankingMode,
    policy: SemanticRankingPolicy,
) -> tuple[list[str], list[CandidateSemanticScore], float, str | None]:
    base = list(base_ranking)
    if intent.rankable_condition_count == 0:
        return base, list(rows), 0.0, "NO_RANKABLE_SEMANTIC_CONDITIONS"
    if intent.mean_confidence < policy.minimum_intent_confidence:
        return base, list(rows), 0.0, "INTENT_CONFIDENCE_BELOW_THRESHOLD"
    size = len(rows)
    rows_by_id = {item.business_id: item for item in rows}
    semantic_order = [
        item.business_id
        for item in sorted(rows, key=lambda item: (item.semantic_rank, item.business_id))
    ]
    if mode == "aggressive":
        final_prefix = semantic_order
        effective_alpha = 1.0
        reasons: dict[str, list[str]] = {}
        fused_by_id = {
            item.business_id: item.semantic_score for item in rows
        }
    else:
        average_coverage = sum(item.evidence_coverage for item in rows) / size
        coverage_factor = policy.evidence_floor + (
            1.0 - policy.evidence_floor
        ) * average_coverage
        effective_alpha = min(
            policy.maximum_fusion_alpha,
            policy.maximum_fusion_alpha
            * intent.mean_confidence
            * coverage_factor,
        )
        fused_by_id = {
            item.business_id: (
                (1.0 - effective_alpha) * _percentile(item.base_rank, size)
                + effective_alpha * _percentile(item.semantic_rank, size)
            )
            for item in rows
        }
        scored_rows = [
            item.model_copy(update={"fused_score": fused_by_id[item.business_id]})
            for item in rows
        ]
        final_prefix = _bounded_order(
            scored_rows,
            maximum_upward_move=policy.maximum_upward_move,
            maximum_downward_move=policy.maximum_downward_move,
        )
        final_prefix, reasons = _protect_top_k(
            final_prefix,
            {item.business_id: item for item in scored_rows},
            policy,
        )
    tail = base[size:]
    final = [*final_prefix, *tail]
    final_rank = {business_id: rank for rank, business_id in enumerate(final, start=1)}
    updated = [
        item.model_copy(
            update={
                "fused_score": fused_by_id[item.business_id],
                "final_rank": final_rank[item.business_id],
                "rank_movement": item.base_rank - final_rank[item.business_id],
                "protection_reason_codes": reasons.get(item.business_id, []),
            }
        )
        for item in rows
    ]
    return final, updated, effective_alpha, None
