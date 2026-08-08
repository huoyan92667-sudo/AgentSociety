"""Leakage-resistant synthetic request benchmark and parser evaluation."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.query.parser import RecommendationRequestParser
from yelp_agent.query.schema import (
    ConditionField,
    ConditionOperator,
    ConditionValue,
    EnforcementMode,
    MissingField,
    QueryParseInput,
    RequestIntent,
    RequirementImportance,
)


class ExpectedRequestCondition(StrictModel):
    field: ConditionField
    operator: ConditionOperator
    value: ConditionValue
    importance: RequirementImportance
    enforcement: EnforcementMode


class QueryBenchmarkCase(StrictModel):
    """One business-independent semantic frame rendered as user text."""

    case_id: str = Field(min_length=1)
    split: Literal["development", "validation"]
    frame_family: str = Field(min_length=1)
    language: Literal["zh-CN", "en-US"]
    query_text: str = Field(min_length=1, max_length=2000)
    expected_intent: RequestIntent
    expected_conditions: list[ExpectedRequestCondition]
    expected_party_size: int | None = Field(default=None, ge=1, le=100)
    expected_missing_fields: list[MissingField]
    generator_kind: Literal["human_seed", "openai_compatible"]
    generator_model: str | None = None
    generator_prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    uses_specific_business: Literal[False]
    uses_future_review: Literal[False]
    user_latitude: float | None = Field(default=None, ge=-90, le=90)
    user_longitude: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def validate_generation_and_location(self) -> QueryBenchmarkCase:
        if (self.user_latitude is None) != (self.user_longitude is None):
            raise ValueError("benchmark coordinates must be both present or absent")
        generated = self.generator_kind == "openai_compatible"
        if generated != bool(self.generator_model and self.generator_prompt_sha256):
            raise ValueError(
                "model-generated cases require model and prompt hash metadata"
            )
        keys = [_expected_key(condition) for condition in self.expected_conditions]
        if len(set(keys)) != len(keys):
            raise ValueError("expected conditions must be unique")
        if len(set(self.expected_missing_fields)) != len(self.expected_missing_fields):
            raise ValueError("expected missing fields must be unique")
        return self

    def parse_input(self) -> QueryParseInput:
        return QueryParseInput(
            user_id=f"benchmark-user:{self.case_id}",
            session_id=f"benchmark-session:{self.case_id}",
            cutoff_time=datetime(2024, 1, 1),
            query_text=self.query_text,
            user_latitude=self.user_latitude,
            user_longitude=self.user_longitude,
        )


class QueryBenchmarkFailure(StrictModel):
    case_id: str
    split: Literal["development", "validation"]
    expected_conditions: list[str]
    actual_conditions: list[str]
    expected_missing_fields: list[MissingField]
    actual_missing_fields: list[MissingField]
    intent_match: bool
    party_size_match: bool


class QueryParserBenchmarkReport(StrictModel):
    parser_version: str
    case_count: int = Field(ge=1)
    split_counts: dict[str, int]
    exact_match_rate: float = Field(ge=0, le=1)
    intent_accuracy: float = Field(ge=0, le=1)
    party_size_accuracy: float = Field(ge=0, le=1)
    missing_fields_exact_match_rate: float = Field(ge=0, le=1)
    condition_precision: float = Field(ge=0, le=1)
    condition_recall: float = Field(ge=0, le=1)
    condition_f1: float = Field(ge=0, le=1)
    failures: list[QueryBenchmarkFailure]
    target_business_fields_available: Literal[False] = False
    future_review_fields_available: Literal[False] = False


def _value_token(value: ConditionValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _expected_key(condition: ExpectedRequestCondition) -> str:
    return ":".join(
        (
            condition.field,
            condition.operator,
            _value_token(condition.value),
            condition.importance,
            condition.enforcement,
        )
    )


def _actual_key(condition: object) -> str:
    return ":".join(
        (
            str(condition.field),
            str(condition.operator),
            _value_token(condition.value),
            str(condition.importance),
            str(condition.enforcement),
        )
    )


def load_query_benchmark(path: str | Path) -> tuple[QueryBenchmarkCase, ...]:
    """Load stable JSONL while forbidding split leakage by frame family."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Query benchmark does not exist: {source}")
    cases: list[QueryBenchmarkCase] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(QueryBenchmarkCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid query benchmark row at line {line_number}"
                ) from exc
    if not cases:
        raise ValueError("Query benchmark cannot be empty")
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Query benchmark case IDs must be unique")
    family_splits: dict[str, set[str]] = {}
    for case in cases:
        family_splits.setdefault(case.frame_family, set()).add(case.split)
    leaked = sorted(
        family for family, splits in family_splits.items() if len(splits) > 1
    )
    if leaked:
        raise ValueError(f"Frame families cross benchmark splits: {leaked[:3]}")
    return tuple(sorted(cases, key=lambda case: case.case_id))


def evaluate_request_parser(
    parser: RecommendationRequestParser,
    cases: Sequence[QueryBenchmarkCase],
) -> QueryParserBenchmarkReport:
    """Score observable structured meaning, never internal parser steps."""

    if not cases:
        raise ValueError("benchmark cases cannot be empty")
    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact_matches = 0
    intent_matches = 0
    party_matches = 0
    missing_matches = 0
    failures: list[QueryBenchmarkFailure] = []
    split_counts: Counter[str] = Counter()
    for case in cases:
        split_counts[case.split] += 1
        request = parser.parse(case.parse_input())
        expected = {_expected_key(item) for item in case.expected_conditions}
        actual = {_actual_key(item) for item in request.conditions}
        true_positive += len(expected.intersection(actual))
        false_positive += len(actual.difference(expected))
        false_negative += len(expected.difference(actual))
        intent_match = request.intent == case.expected_intent
        party_match = request.party_size == case.expected_party_size
        missing_match = set(request.missing_fields) == set(case.expected_missing_fields)
        intent_matches += int(intent_match)
        party_matches += int(party_match)
        missing_matches += int(missing_match)
        exact = expected == actual and intent_match and party_match and missing_match
        exact_matches += int(exact)
        if not exact:
            failures.append(
                QueryBenchmarkFailure(
                    case_id=case.case_id,
                    split=case.split,
                    expected_conditions=sorted(expected),
                    actual_conditions=sorted(actual),
                    expected_missing_fields=case.expected_missing_fields,
                    actual_missing_fields=request.missing_fields,
                    intent_match=intent_match,
                    party_size_match=party_match,
                )
            )
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 1.0
    recall = true_positive / recall_denominator if recall_denominator else 1.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    count = len(cases)
    return QueryParserBenchmarkReport(
        parser_version=parser.version,
        case_count=count,
        split_counts=dict(sorted(split_counts.items())),
        exact_match_rate=exact_matches / count,
        intent_accuracy=intent_matches / count,
        party_size_accuracy=party_matches / count,
        missing_fields_exact_match_rate=missing_matches / count,
        condition_precision=precision,
        condition_recall=recall,
        condition_f1=f1,
        failures=failures,
    )


def write_query_parser_benchmark_report(
    report: QueryParserBenchmarkReport,
    output_path: str | Path,
) -> Path:
    """Atomically publish one deterministic, secret-free benchmark report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_text(
            report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return output
