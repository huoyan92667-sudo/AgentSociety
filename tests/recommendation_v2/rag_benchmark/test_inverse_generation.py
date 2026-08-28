from __future__ import annotations

from yelp_agent.recommendation_v2.rag_benchmark.inverse_generation import (
    _GenerationBatch,
    _SeedRecord,
    _deduplicate_cases,
    _find_evidence_segment,
    _validate_and_materialize,
)
from yelp_agent.recommendation_v2.rag_benchmark.inverse_schema import (
    InverseGenerationProposal,
)


def _batch() -> _GenerationBatch:
    return _GenerationBatch(
        batch_id="fixed_quiet_environment",
        dataset_kind="fixed_aspect",
        aspect="quiet_environment",
        target_direction="higher",
        candidates=(
            _SeedRecord("r1", "b1", "It was calm and quiet inside.", "a" * 64),
            _SeedRecord("r2", "b2", "The music was painfully loud.", "b" * 64),
        ),
        expected_count=2,
        expected_positive=1,
        expected_negative=1,
        prompt="test",
    )


def _proposal() -> InverseGenerationProposal:
    return InverseGenerationProposal.model_validate(
        {
            "drafts": [
                {
                    "seed_review_id": "r1",
                    "query_text": "想找一家能够安静聊天的餐厅",
                    "requirement_text": "环境安静",
                    "expected_direction": "positive",
                    "evidence_span": "calm and quiet",
                },
                {
                    "seed_review_id": "r2",
                    "query_text": "希望吃饭时不用扯着嗓子说话",
                    "requirement_text": "环境不要吵闹",
                    "expected_direction": "negative",
                    "evidence_span": "painfully loud",
                },
            ]
        }
    )


def test_materialization_requires_exact_evidence_and_indexed_segment() -> None:
    cases = _validate_and_materialize(
        _batch(),
        _proposal(),
        {
            "r1": [("1" * 64, "It was calm and quiet inside.")],
            "r2": [("2" * 64, "The music was painfully loud.")],
        },
        {"b1": "Quiet Place", "b2": "Loud Place"},
        teacher_model="fake",
    )

    assert [item.expected_direction for item in cases] == ["positive", "negative"]
    assert cases[0].seed_segment_id == "1" * 64
    assert cases[1].evidence_span == "painfully loud"


def test_evidence_segment_chooses_shortest_context_containing_exact_span() -> None:
    selected = _find_evidence_segment(
        [("a", "long context with exact words inside"), ("b", "exact words")],
        "exact words",
    )

    assert selected == ("b", "exact words")


def test_deduplication_uses_review_id_but_allows_same_natural_query() -> None:
    cases = _validate_and_materialize(
        _batch(),
        _proposal(),
        {
            "r1": [("1" * 64, "It was calm and quiet inside.")],
            "r2": [("2" * 64, "The music was painfully loud.")],
        },
        {},
        teacher_model="fake",
    )
    duplicate_query = cases[1].model_copy(update={"query_text": cases[0].query_text})

    result = _deduplicate_cases([cases[0], duplicate_query])

    assert len(result) == 2
