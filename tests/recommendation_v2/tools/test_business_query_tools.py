from __future__ import annotations

from yelp_agent.recommendation_v2.business_aspect_profiles import (
    load_business_aspect_profile_catalog,
)
from yelp_agent.recommendation_v2.business_facts import load_business_fact_catalog
from yelp_agent.recommendation_v2.tools import (
    BusinessAspectEvidenceQuery,
    BusinessAspectEvidenceTool,
    BusinessNameSearchQuery,
    BusinessNameSearchTool,
)


def test_business_name_search_resolves_unique_exact_name() -> None:
    tool = BusinessNameSearchTool(load_business_fact_catalog())

    result = tool.execute(BusinessNameSearchQuery(name="spice 28"))

    assert result.status == "resolved"
    assert len(result.matches) == 1
    assert result.matches[0].name == "Spice 28"
    assert result.matches[0].match_kind == "exact"


def test_business_name_search_keeps_same_name_candidates_for_model_disambiguation() -> None:
    tool = BusinessNameSearchTool(load_business_fact_catalog())

    result = tool.execute(BusinessNameSearchQuery(name="Wawa"))

    assert result.status == "ambiguous"
    assert len(result.matches) == 5
    assert all(item.name == "Wawa" for item in result.matches)
    assert len({item.business_id for item in result.matches}) == 5


def test_fixed_aspect_tool_returns_score_sufficiency_controversy_and_reviews() -> None:
    catalog = load_business_aspect_profile_catalog()
    business = catalog.supported_businesses()[0]
    tool = BusinessAspectEvidenceTool(catalog)

    result = tool.execute(
        BusinessAspectEvidenceQuery(
            business_ids=[business.business_id],
            aspect_ids=["service"],
            evidence_limit_per_group=2,
        )
    )

    assert result.status == "found"
    assert len(result.assessments) == 1
    assessment = result.assessments[0]
    assert assessment.business_id == business.business_id
    assert assessment.aspect_id == "service"
    assert assessment.direction.higher_value_means
    assert assessment.score.evidence_sufficiency_level
    assert assessment.score.controversy_level
    assert len(assessment.high_degree_evidence) <= 2
    assert len(assessment.low_degree_evidence) <= 2


def test_fixed_aspect_tool_tells_caller_when_business_has_no_offline_profile() -> None:
    facts = load_business_fact_catalog()
    profiles = load_business_aspect_profile_catalog()
    unsupported = next(
        item for item in facts.all() if not profiles.contains(item.business_id)
    )

    result = BusinessAspectEvidenceTool(profiles).execute(
        BusinessAspectEvidenceQuery(
            business_ids=[unsupported.business_id],
            aspect_ids=["quiet_environment"],
        )
    )

    assert result.status == "not_found"
    assert result.unsupported_business_ids == [unsupported.business_id]
