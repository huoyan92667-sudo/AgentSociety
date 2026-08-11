"""Deterministic cross-review aggregation and cautious answer policy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import hashlib
import json

from .config import EvidenceAggregationPolicy
from .normalizer import normalize_evidence
from .schema import (
    AspectEvidenceAssessment,
    BusinessEvidenceAssessment,
    ConditionEvidenceGroup,
    EvidenceAggregationRequest,
    EvidenceAssessment,
    EvidenceAtom,
    EvidenceConfidenceLevel,
)


_CONFIDENCE_ORDER: dict[EvidenceConfidenceLevel, int] = {
    "insufficient": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


class EvidenceAggregator:
    """Deep module: one request in, one cutoff-safe assessment out."""

    def __init__(self, policy: EvidenceAggregationPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> EvidenceAggregationPolicy:
        return self._policy

    def aggregate(self, request: EvidenceAggregationRequest) -> EvidenceAssessment:
        atoms = normalize_evidence(request, self._policy)
        grouped: defaultdict[tuple[str, str], list[EvidenceAtom]] = defaultdict(list)
        for atom in atoms:
            grouped[(atom.business_id, atom.aspect)].append(atom)
        businesses: list[BusinessEvidenceAssessment] = []
        for business_id in request.search_result.business_ids:
            aspects = [
                self._assess_aspect(
                    business_id,
                    aspect,
                    grouped.get((business_id, aspect), []),
                    explicit_uncertainty=request.explicit_uncertainty_request,
                )
                for aspect in request.requested_aspects
            ]
            if not aspects:
                inferred = sorted(
                    {
                        aspect
                        for found_business, aspect in grouped
                        if found_business == business_id
                    }
                )
                aspects = [
                    self._assess_aspect(
                        business_id,
                        aspect,
                        grouped[(business_id, aspect)],
                        explicit_uncertainty=request.explicit_uncertainty_request,
                    )
                    for aspect in inferred
                ]
            mode = (
                "grounded"
                if aspects and all(item.response_mode == "grounded" for item in aspects)
                else "uncertain"
            )
            latest = max(
                (item.latest_evidence_time for item in aspects if item.latest_evidence_time),
                default=None,
            )
            level = min(
                (item.confidence_level for item in aspects),
                key=lambda value: _CONFIDENCE_ORDER[value],
                default="insufficient",
            )
            businesses.append(
                BusinessEvidenceAssessment(
                    business_id=business_id,
                    aspects=aspects,
                    overall_response_mode=mode,
                    has_conflict=any(item.has_conflict for item in aspects),
                    confidence_level=level,
                    latest_evidence_time=latest,
                )
            )
        payload = {
            "query_text": request.query_text,
            "task_type": request.task_type,
            "requested_aspects": request.requested_aspects,
            "desired_polarity_by_aspect": request.desired_polarity_by_aspect,
            "explicit_uncertainty_request": request.explicit_uncertainty_request,
            "search_request_sha256": request.search_result.request_sha256,
            "policy_version": self._policy.policy_version,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return EvidenceAssessment(
            request_sha256=digest,
            policy_version=self._policy.policy_version,
            task_type=request.task_type,
            business_ids=list(request.search_result.business_ids),
            cutoff_time=request.search_result.cutoff_time,
            businesses=businesses,
            recommend_official_verification=(
                request.task_type == "official_policy_question"
            ),
        )

    def _assess_aspect(
        self,
        business_id: str,
        aspect: str,
        atoms: list[EvidenceAtom],
        *,
        explicit_uncertainty: bool,
    ) -> AspectEvidenceAssessment:
        support = [item for item in atoms if item.stance == "supports"]
        contradict = [item for item in atoms if item.stance == "contradicts"]
        neutral = [item for item in atoms if item.stance == "neutral"]
        support_mass = sum(item.weight for item in support)
        contradiction_mass = sum(item.weight for item in contradict)
        directional_mass = support_mass + contradiction_mass
        minority_share = (
            min(support_mass, contradiction_mass) / directional_mass
            if directional_mass > 0
            else 0.0
        )
        has_conflict = (
            len(support) >= self._policy.conflict_minimum_count_per_side
            and len(contradict) >= self._policy.conflict_minimum_count_per_side
            and minority_share >= self._policy.conflict_minority_mass_share
        )
        if directional_mass <= 0:
            consensus = "insufficient"
            consistency = 0.0
        else:
            support_share = support_mass / directional_mass
            contradiction_share = contradiction_mass / directional_mass
            consistency = max(support_share, contradiction_share)
            if has_conflict:
                consensus = "mixed"
            elif support_share >= self._policy.consensus_mass_share:
                consensus = "supports"
            elif contradiction_share >= self._policy.consensus_mass_share:
                consensus = "contradicts"
            else:
                consensus = "mixed"
        count = len(atoms)
        unique_users = len({item.user_id for item in atoms})
        relevance = _mean(item.relevance_score for item in atoms)
        recency = _mean(item.recency_score for item in atoms)
        extraction = _mean(item.extraction_confidence for item in atoms)
        diversity = min(1.0, unique_users / self._policy.diversity_target_users)
        sample_strength = (
            count / (count + self._policy.sample_prior) if count else 0.0
        )
        confidence = float(
            relevance
            * recency
            * consistency
            * diversity
            * sample_strength
            * extraction
        )
        level = self._confidence_level(confidence, directional_mass > 0)
        enough = (
            len(support) + len(contradict)
            >= self._policy.grounded_minimum_evidence
            and len({item.user_id for item in (*support, *contradict)})
            >= self._policy.grounded_minimum_unique_users
        )
        response_mode = (
            "grounded"
            if enough
            and consensus in {"supports", "contradicts"}
            and not has_conflict
            and not (explicit_uncertainty and level in {"insufficient", "low"})
            else "uncertain"
        )
        citations = [
            item.review_id
            for item in sorted(
                (*support, *contradict),
                key=lambda item: (-item.weight, item.hit_rank, item.review_id),
            )[: self._policy.maximum_citations_per_aspect]
        ]
        return AspectEvidenceAssessment(
            business_id=business_id,
            aspect=aspect,  # type: ignore[arg-type]
            consensus=consensus,  # type: ignore[arg-type]
            confidence_score=confidence,
            confidence_level=level,
            response_mode=response_mode,
            requires_caveat=(
                response_mode == "uncertain" or level in {"insufficient", "low"}
            ),
            has_conflict=has_conflict,
            evidence_count=count,
            directional_evidence_count=len(support) + len(contradict),
            unique_user_count=unique_users,
            support_count=len(support),
            contradiction_count=len(contradict),
            neutral_count=len(neutral),
            support_mass=support_mass,
            contradiction_mass=contradiction_mass,
            mean_relevance=relevance,
            mean_recency=recency,
            consistency=consistency,
            source_diversity=diversity,
            sample_strength=sample_strength,
            mean_extraction_confidence=extraction,
            latest_evidence_time=max(
                (item.review_time for item in atoms), default=None
            ),
            condition_groups=_condition_groups(atoms),
            citation_review_ids=list(dict.fromkeys(citations)),
            atoms=atoms,
        )

    def _confidence_level(
        self,
        confidence: float,
        has_directional_evidence: bool,
    ) -> EvidenceConfidenceLevel:
        if not has_directional_evidence:
            return "insufficient"
        if confidence >= self._policy.high_confidence_threshold:
            return "high"
        if confidence >= self._policy.medium_confidence_threshold:
            return "medium"
        return "low"


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _condition_groups(atoms: list[EvidenceAtom]) -> list[ConditionEvidenceGroup]:
    grouped: defaultdict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in atoms:
        for tag in atom.condition_tags:
            grouped[tag].append(atom)
    result: list[ConditionEvidenceGroup] = []
    for tag in sorted(grouped):
        values = grouped[tag]
        result.append(
            ConditionEvidenceGroup(
                condition_tag=tag,
                evidence_count=len(values),
                support_count=sum(item.stance == "supports" for item in values),
                contradiction_count=sum(
                    item.stance == "contradicts" for item in values
                ),
                review_ids=list(dict.fromkeys(item.review_id for item in values)),
            )
        )
    return result
