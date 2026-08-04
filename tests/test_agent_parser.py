from __future__ import annotations

from collections.abc import Callable
import json

import pytest

from yelp_agent.agent.parser import (
    AgentResponseError,
    merge_reranked_top_k,
    parse_rerank_response,
)


def test_valid_json_is_parsed_and_only_replaces_hybrid_top_eight() -> None:
    hybrid_ranking = [f"business-{index:02d}" for index in range(20)]
    reranked = list(reversed(hybrid_ranking[:8]))
    content = json.dumps(
        {
            "ranking": reranked,
            "reason": "Closer matches for the user's preferences.",
        }
    )

    parsed = parse_rerank_response(
        content,
        expected_business_ids=hybrid_ranking[:8],
    )
    final_ranking = merge_reranked_top_k(parsed, hybrid_ranking)

    assert parsed.ranking == reranked
    assert parsed.reason == "Closer matches for the user's preferences."
    assert final_ranking[:8] == reranked
    assert final_ranking[8:] == hybrid_ranking[8:]
    assert len(final_ranking) == 20
    assert len(set(final_ranking)) == 20


@pytest.mark.parametrize(
    "content, expected_reason",
    [
        ("", "empty_response"),
        ("Here is the ranking", "non_json"),
        ('```json\n{"ranking": []}\n```', "non_json"),
    ],
)
def test_empty_text_and_non_json_wrappers_are_rejected(
    content: str,
    expected_reason: str,
) -> None:
    expected_ids = [f"business-{index:02d}" for index in range(8)]

    with pytest.raises(AgentResponseError) as captured:
        parse_rerank_response(
            content,
            expected_business_ids=expected_ids,
        )

    assert captured.value.reason == expected_reason


@pytest.mark.parametrize(
    "payload_factory, expected_reason",
    [
        (lambda ids: [], "schema_validation"),
        (lambda ids: {"reason": "missing ranking"}, "schema_validation"),
        (
            lambda ids: {"ranking": ids, "unexpected": True},
            "schema_validation",
        ),
        (
            lambda ids: {"ranking": [*ids[:7], ids[0]]},
            "duplicate_id",
        ),
        (lambda ids: {"ranking": ids[:7]}, "missing_id"),
        (
            lambda ids: {"ranking": [*ids[:7], "outside-candidate"]},
            "unknown_id",
        ),
    ],
)
def test_invalid_schema_or_candidate_permutation_has_stable_reason(
    payload_factory: Callable[[list[str]], object],
    expected_reason: str,
) -> None:
    expected_ids = [f"business-{index:02d}" for index in range(8)]
    payload = payload_factory(expected_ids)

    with pytest.raises(AgentResponseError) as captured:
        parse_rerank_response(
            json.dumps(payload),
            expected_business_ids=expected_ids,
        )

    assert captured.value.reason == expected_reason
