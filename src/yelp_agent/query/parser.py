"""Deterministic request parsing with a seam for future semantic adapters."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from yelp_agent.models import LocationCenter
from yelp_agent.query.schema import (
    ConditionField,
    ConditionOperator,
    ConditionSource,
    ConditionValue,
    EnforcementMode,
    MissingField,
    QueryParseInput,
    RecommendationRequest,
    RequestCondition,
    RequirementImportance,
    UnknownPolicy,
    stable_request_id,
)


@dataclass(frozen=True, slots=True)
class ExtractedRequestSignal:
    """Model-neutral observation awaiting deterministic policy assignment."""

    field: ConditionField
    operator: ConditionOperator
    value: ConditionValue
    importance: RequirementImportance
    evidence_span: str
    evidence_start: int
    evidence_end: int
    confidence: float
    source: ConditionSource


class RequestSignalExtractor(Protocol):
    """Internal seam implemented by rules now and semantic models later."""

    version: str

    def extract(self, value: QueryParseInput) -> Sequence[ExtractedRequestSignal]: ...


_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "Steakhouses": ("牛排", "牛排馆", "steakhouse", "steak"),
    "Bars": ("酒吧", "bar", "bars"),
    "Japanese": ("日料", "日本料理", "japanese"),
    "Chinese": ("中餐", "中国菜", "chinese"),
    "Italian": ("意大利菜", "意餐", "italian"),
    "Coffee & Tea": ("咖啡", "coffee", "tea"),
    "Fast Food": ("快餐", "fast food"),
}
_MANDATORY_MARKERS = (
    "必须",
    "一定要",
    "只想",
    "只要",
    "只能",
    "不能",
    "不要",
    "不吃",
    "only",
    "must",
)
_PREFERRED_MARKERS = (
    "最好",
    "尽量",
    "优先",
    "希望",
    "可以的话",
    "prefer",
    "preferably",
    "ideally",
)
_EXCLUSION_MARKERS = ("不要", "别", "排除", "不能是", "not ", "no ", "without ")

_ASPECT_PATTERNS: tuple[
    tuple[ConditionField, ConditionOperator, tuple[re.Pattern[str], ...]], ...
] = (
    (
        "quiet_environment",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|必须|一定要)?\s*(?:安静(?:一点)?|不吵)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:prefer|ideally|must)?\s*(?:quiet|not noisy)", re.IGNORECASE
            ),
        ),
    ),
    (
        "date_suitable",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|必须|一定要)?\s*(?:适合)?(?:约会|求婚)",
                re.IGNORECASE,
            ),
            re.compile(r"(?:romantic|date night|proposal)", re.IGNORECASE),
        ),
    ),
    (
        "parking",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|必须|一定要)?\s*(?:停车方便|有停车场|有车位)",
                re.IGNORECASE,
            ),
            re.compile(r"(?:easy parking|parking available|parking)", re.IGNORECASE),
        ),
    ),
    (
        "pet_friendly",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|必须|一定要)?\s*(?:宠物友好|能带(?:狗|宠物))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:pet[- ]friendly|allows? dogs?|bring (?:a )?dog)", re.IGNORECASE
            ),
        ),
    ),
    (
        "family_friendly",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|必须|一定要)?\s*(?:亲子|适合孩子|适合小孩)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:family[- ]friendly|good for (?:kids|children))", re.IGNORECASE
            ),
        ),
    ),
    (
        "group_suitable",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|必须|一定要)?\s*(?:适合)?(?:聚会|团建|多人聚餐)",
                re.IGNORECASE,
            ),
            re.compile(r"(?:good for groups|group dinner)", re.IGNORECASE),
        ),
    ),
    (
        "crowded",
        "avoid",
        (
            re.compile(r"(?:不要|别|尽量别)?\s*(?:太)?(?:拥挤|挤)", re.IGNORECASE),
            re.compile(r"(?:not crowded|avoid crowds?)", re.IGNORECASE),
        ),
    ),
    (
        "queue_time",
        "avoid",
        (
            re.compile(r"(?:不要|别|尽量别)?\s*(?:排队|等位)(?:太久)?", re.IGNORECASE),
            re.compile(r"(?:no long wait|avoid (?:a )?queue)", re.IGNORECASE),
        ),
    ),
    (
        "spiciness",
        "avoid",
        (
            re.compile(r"(?:不要|不能吃|不吃|别太)\s*(?:太)?辣", re.IGNORECASE),
            re.compile(r"(?:not spicy|avoid spicy food)", re.IGNORECASE),
        ),
    ),
    (
        "cleanliness",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|必须|一定要)?\s*(?:干净|卫生)", re.IGNORECASE
            ),
            re.compile(r"(?:clean|hygienic)", re.IGNORECASE),
        ),
    ),
    (
        "food_quality",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|必须|一定要)?\s*(?:好吃|味道好)", re.IGNORECASE
            ),
            re.compile(r"(?:great food|tasty|delicious)", re.IGNORECASE),
        ),
    ),
    (
        "service",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|必须|一定要)?\s*(?:服务好|服务周到)", re.IGNORECASE
            ),
            re.compile(r"(?:good service|great service)", re.IGNORECASE),
        ),
    ),
    (
        "price_value",
        "prefer",
        (
            re.compile(
                r"(?:最好|尽量|优先|希望)?\s*(?:性价比高|实惠|划算)", re.IGNORECASE
            ),
            re.compile(r"(?:good value|affordable)", re.IGNORECASE),
        ),
    ),
)


def _nearby_prefix(text: str, start: int, width: int = 12) -> str:
    prefix = text[max(0, start - width) : start].casefold()
    return re.split(r"[，,。.!?！？；;]", prefix)[-1]


def _has_marker(prefix: str, markers: tuple[str, ...]) -> bool:
    return any(marker in prefix for marker in markers)


def _importance(prefix: str) -> RequirementImportance:
    if _has_marker(prefix, _PREFERRED_MARKERS):
        return "preferred"
    if _has_marker(prefix, _MANDATORY_MARKERS):
        return "mandatory"
    return "strong"


def _find_alias(text: str, alias: str) -> re.Match[str] | None:
    escaped = re.escape(alias.casefold())
    pattern = (
        rf"(?<!\w){escaped}(?!\w)"
        if all(ord(character) < 128 for character in alias)
        else escaped
    )
    return re.search(pattern, text)


class RuleBasedRequestSignalExtractor:
    """High-precision bilingual baseline for explicit, auditable signals."""

    version = "rule-based-v1.0.0"

    def extract(self, value: QueryParseInput) -> tuple[ExtractedRequestSignal, ...]:
        text = value.query_text
        lowered = text.casefold()
        signals: list[ExtractedRequestSignal] = []
        occupied_categories: set[str] = set()
        for category, aliases in _CATEGORY_ALIASES.items():
            matches = [
                match
                for alias in aliases
                if (match := _find_alias(lowered, alias)) is not None
            ]
            if not matches:
                continue
            match = min(matches, key=lambda item: item.start())
            prefix = _nearby_prefix(lowered, match.start())
            excluded = _has_marker(prefix, _EXCLUSION_MARKERS)
            importance = "mandatory" if excluded else _importance(prefix)
            signals.append(
                ExtractedRequestSignal(
                    field="category",
                    operator="excludes" if excluded else "includes",
                    value=category,
                    importance=importance,
                    evidence_span=text[match.start() : match.end()],
                    evidence_start=match.start(),
                    evidence_end=match.end(),
                    confidence=1.0,
                    source="rule",
                )
            )
            occupied_categories.add(category)

        distance_patterns = (
            re.compile(
                r"(?:必须|一定要|只要)?在?\s*(\d+(?:\.\d+)?)\s*(?:公里|km)\s*(?:以内|内)",
                re.IGNORECASE,
            ),
            re.compile(r"within\s+(\d+(?:\.\d+)?)\s*(?:km|kilometers?)", re.IGNORECASE),
        )
        for pattern in distance_patterns:
            match = pattern.search(text)
            if match is None:
                continue
            signals.append(
                ExtractedRequestSignal(
                    field="distance_km",
                    operator="less_than_or_equal",
                    value=float(match.group(1)),
                    importance="mandatory",
                    evidence_span=match.group(0),
                    evidence_start=match.start(),
                    evidence_end=match.end(),
                    confidence=1.0,
                    source="rule",
                )
            )
            break

        budget_patterns = (
            re.compile(
                r"人均\s*(?:不超过|不要超过|不能超过|不高于|最多|控制在)?\s*(\d+(?:\.\d+)?)\s*(?:元|块)?",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:per person|per-person)\s*(?:under|below|at most)?"
                r"\s*\$?(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
        )
        for pattern in budget_patterns:
            match = pattern.search(text)
            if match is None:
                continue
            signals.append(
                ExtractedRequestSignal(
                    field="budget_per_person",
                    operator="less_than_or_equal",
                    value=float(match.group(1)),
                    importance="mandatory",
                    evidence_span=match.group(0),
                    evidence_start=match.start(),
                    evidence_end=match.end(),
                    confidence=0.98,
                    source="rule",
                )
            )
            break

        for field, operator, patterns in _ASPECT_PATTERNS:
            for pattern in patterns:
                match = pattern.search(text)
                if match is None:
                    continue
                span = match.group(0).strip()
                start = match.start() + (
                    len(match.group(0)) - len(match.group(0).lstrip())
                )
                signals.append(
                    ExtractedRequestSignal(
                        field=field,
                        operator=operator,
                        value=True,
                        importance=_importance(
                            (_nearby_prefix(lowered, start) + span).casefold()
                        ),
                        evidence_span=span,
                        evidence_start=start,
                        evidence_end=start + len(span),
                        confidence=0.95,
                        source="rule",
                    )
                )
                break
        return tuple(signals)


def condition_from_signal(
    signal: ExtractedRequestSignal,
    *,
    has_user_location: bool,
) -> tuple[RequestCondition, MissingField | None]:
    if signal.field == "category":
        filterable = signal.operator == "excludes" or signal.importance == "mandatory"
        enforcement: EnforcementMode = "filter" if filterable else "rank"
        unknown_policy: UnknownPolicy = "exclude" if filterable else "not_applicable"
        missing = None
    elif signal.field == "distance_km":
        if signal.importance == "mandatory":
            enforcement = "filter" if has_user_location else "clarify"
            unknown_policy = "exclude" if has_user_location else "ask"
            missing = None if has_user_location else "user_location"
        else:
            enforcement = "rank"
            unknown_policy = "allow_with_warning"
            missing = None
    elif signal.field == "budget_per_person":
        # Yelp exposes coarse price tiers, not a reliable per-person amount.
        enforcement = "clarify"
        unknown_policy = "ask"
        missing = "budget_precision"
    elif signal.importance == "mandatory":
        enforcement = "evidence"
        unknown_policy = "ask"
        missing = None
    else:
        enforcement = "rank"
        unknown_policy = "allow_with_warning"
        missing = None
    return (
        RequestCondition(
            field=signal.field,
            operator=signal.operator,
            value=signal.value,
            importance=signal.importance,
            enforcement=enforcement,
            explicit=True,
            confidence=signal.confidence,
            evidence_span=signal.evidence_span,
            evidence_start=signal.evidence_start,
            evidence_end=signal.evidence_end,
            source=signal.source,
            unknown_policy=unknown_policy,
        ),
        missing,
    )


class RecommendationRequestParser:
    """Turn one user turn into the only request contract used downstream."""

    def __init__(
        self,
        primary_extractor: RequestSignalExtractor,
        semantic_extractor: RequestSignalExtractor | None = None,
    ) -> None:
        self._primary_extractor = primary_extractor
        self._semantic_extractor = semantic_extractor
        self.version = primary_extractor.version + (
            "" if semantic_extractor is None else f"+{semantic_extractor.version}"
        )

    def parse(self, value: QueryParseInput) -> RecommendationRequest:
        signals = list(self._primary_extractor.extract(value))
        warnings: list[str] = []
        if self._semantic_extractor is not None:
            try:
                signals.extend(self._semantic_extractor.extract(value))
            # A provider adapter is optional; its failure must preserve the
            # deterministic baseline regardless of provider exception type.
            except Exception as exc:
                warnings.append(
                    "SEMANTIC_EXTRACTOR_FAILED:"
                    f"{self._semantic_extractor.version}:{type(exc).__name__}"
                )
        conditions: list[RequestCondition] = []
        missing: list[MissingField] = []
        seen: set[tuple[object, ...]] = set()
        for signal in signals:
            key = (signal.field, signal.operator, str(signal.value))
            if key in seen:
                continue
            seen.add(key)
            condition, missing_field = condition_from_signal(
                signal,
                has_user_location=value.user_latitude is not None,
            )
            conditions.append(condition)
            if missing_field is not None and missing_field not in missing:
                missing.append(missing_field)
        conditions.sort(
            key=lambda item: (
                item.evidence_start,
                item.field,
                item.operator,
                str(item.value),
            )
        )
        party_size = _extract_party_size(value.query_text)
        has_desired_category = any(
            item.field == "category" and item.operator == "includes"
            for item in conditions
        )
        if not has_desired_category and "desired_category" not in missing:
            missing.append("desired_category")
        return RecommendationRequest(
            request_id=stable_request_id(value, parser_version=self.version),
            user_id=value.user_id,
            session_id=value.session_id,
            cutoff_time=value.cutoff_time,
            query_text=value.query_text,
            intent="recommendation_request",
            conditions=conditions,
            party_size=party_size,
            location_center=(
                None
                if value.user_latitude is None
                else LocationCenter(
                    latitude=value.user_latitude,
                    longitude=value.user_longitude,
                )
            ),
            missing_fields=missing,
            referenced_business_ids=value.referenced_business_ids,
            parse_warnings=warnings,
            parser_version=self.version,
        )


_CHINESE_NUMBERS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _extract_party_size(text: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*(?:个人|人位|人)", text)
    if match is not None:
        return int(match.group(1))
    match = re.search(r"([一两二三四五六七八九十])\s*(?:个人|人位|人)", text)
    if match is not None:
        return _CHINESE_NUMBERS[match.group(1)]
    match = re.search(
        r"(?:for\s+)?(\d{1,2})\s+(?:people|persons?)", text, re.IGNORECASE
    )
    if match is not None:
        return int(match.group(1))
    english_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(
        r"(?:for\s+)?(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:people|persons?)",
        text,
        re.IGNORECASE,
    )
    return None if match is None else english_numbers[match.group(1).casefold()]


def build_rule_based_request_parser() -> RecommendationRequestParser:
    """Build the deterministic baseline without loading any external model."""

    return RecommendationRequestParser(RuleBasedRequestSignalExtractor())
