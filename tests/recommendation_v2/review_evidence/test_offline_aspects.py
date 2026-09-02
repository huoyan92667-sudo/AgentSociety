from datetime import UTC, datetime

import pytest

from yelp_agent.recommendation_v2.business_aspect_profiles import (
    BusinessAspectEvidence,
    BusinessAspectScore,
)
from yelp_agent.recommendation_v2.review_evidence.offline_aspects import (
    OfflineAspectEvidenceResolver,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
)
from yelp_agent.recommendation_v2.schema import (
    RequirementBasis,
    SoftPreference,
    merchant_feature_for,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)


class _Catalog:
    """只提供本测试需要的一家商户离线画像。"""

    def __init__(self, *, usable: bool = True) -> None:
        self.usable = usable

    def scores(self, business_ids, aspect_ids):
        assert list(business_ids) == ["b1"]
        assert list(aspect_ids) == ["quiet_environment"]
        return (
            BusinessAspectScore(
                business_id="b1",
                aspect_id="quiet_environment",
                degree=0.9,
                degree_0_to_100=90,
                degree_level_code="4",
                degree_level_name_zh="非常高",
                degree_level_meaning="非常安静",
                evidence_sufficiency=0.5 if self.usable else 0.2,
                evidence_sufficiency_level="一般" if self.usable else "不足",
                controversy=0.2,
                controversy_level="低",
                business_total_review_count=100,
                retrieved_candidate_count=20,
                model_related_review_count=8,
                unique_evidence_user_count=8,
                strong_evidence_count=6,
                unique_strong_user_count=6,
                usable_for_ranking=self.usable,
                ranking_degree=0.9 if self.usable else None,
                unusable_reasons=[] if self.usable else ["证据充分程度不足"],
                effective_sample_size=5,
                evidence_weight_sum=3,
                high_retrieval_limit_reached=False,
                low_retrieval_limit_reached=False,
            ),
        )

    def evidence(self, business_ids, aspect_ids, *, limit_per_group):
        assert limit_per_group == 5
        return (
            _evidence("high", "high_degree", 4, 5),
            _evidence("low", "low_degree", 0, 1),
        )


def _evidence(
    review_id: str,
    group: str,
    strength: int,
    stars: int,
) -> BusinessAspectEvidence:
    return BusinessAspectEvidence.model_validate(
        {
            "business_id": "b1",
            "aspect_id": "quiet_environment",
            "evidence_group": group,
            "evidence_rank": 1,
            "review_id": review_id,
            "user_id": f"u-{review_id}",
            "review_time": "2026-08-01T00:00:00+00:00",
            "stars": stars,
            "useful": 2,
            "text": f"complete {review_id} review",
            "relevance": 3,
            "strength": strength,
            "evidence_weight": 0.8,
        }
    )


def _requirement(direction: str) -> PreferenceSearchDescription:
    preference = SoftPreference(
        key=f"quiet_environment.{direction}",
        field="quiet_environment",
        direction=direction,  # type: ignore[arg-type]
        preference_strength=100,
        priority=1,
        merchant_feature=merchant_feature_for("quiet_environment"),
        controlling_source="current_query",
        sources=[
            RequirementBasis(
                source="current_query",
                text="test quiet direction",
                turn_index=1,
                preference_strength=100,
            )
        ],
    )
    return PreferenceSearchDescription(
        requirement_id=preference.key,
        requirement_text="安静" if direction == "higher" else "热闹",
        kind="fixed_aspect",
        priority=1,
        preference_strength=100,
        positive_descriptions=["positive overall", "positive detail"],
        negative_descriptions=["negative overall", "negative detail"],
        preference=preference,
    )


@pytest.mark.parametrize(
    ("direction", "expected_score", "positive_id", "negative_id"),
    [
        ("higher", 0.7, "high", "low"),
        ("lower", 0.3, "low", "high"),
    ],
)
def test_offline_degree_is_flipped_by_user_direction_and_shrunk_by_sufficiency(
    direction: str,
    expected_score: float,
    positive_id: str,
    negative_id: str,
) -> None:
    resolver = OfflineAspectEvidenceResolver(_Catalog())  # type: ignore[arg-type]
    requirement = _requirement(direction)

    result = resolver.assess_many(
        [requirement],
        ["b1"],
        reference_time=NOW,
    )[requirement.requirement_id]["b1"]

    assert result.evidence_source == "offline_business_profile"
    assert result.evidence_score == pytest.approx(expected_score)
    assert result.objective_degree == 0.9
    assert result.positive_evidence[0].review_id == positive_id
    assert result.negative_evidence[0].review_id == negative_id
    assert result.positive_evidence[0].positive_similarity is None
    assert result.positive_evidence[0].model_strength in {0, 4}


def test_unusable_offline_degree_is_neutral_instead_of_using_extreme_raw_value() -> None:
    resolver = OfflineAspectEvidenceResolver(
        _Catalog(usable=False)  # type: ignore[arg-type]
    )
    requirement = _requirement("higher")

    result = resolver.assess_many(
        [requirement],
        ["b1"],
        reference_time=NOW,
    )[requirement.requirement_id]["b1"]

    assert result.objective_degree == 0.9
    assert result.direction_adjusted_degree is None
    assert result.evidence_score == 0.5
    assert result.satisfaction_level == "证据不足"
    assert result.unusable_reasons == ["证据充分程度不足"]
