"""Plan canonical Query meanings from evidence visible before each cutoff."""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.query.benchmark import ExpectedRequestCondition
from yelp_agent.reviews.schema import AspectName

from .schema import QueryRecommendationFrame
from .selection import BehaviorAnchor

type FrameFamily = Literal[
    "category_only",
    "category_distance",
    "category_price",
    "category_single_aspect",
    "category_two_aspects",
    "category_distance_aspect",
    "occasion_context",
]

_FAMILIES: tuple[FrameFamily, ...] = (
    "category_only",
    "category_distance",
    "category_price",
    "category_single_aspect",
    "category_two_aspects",
    "category_distance_aspect",
    "occasion_context",
)
_ALLOCATION_ORDER: tuple[FrameFamily, ...] = (
    "occasion_context",
    "category_two_aspects",
    "category_distance_aspect",
    "category_single_aspect",
    "category_price",
    "category_distance",
    "category_only",
)
_OCCASION_ASPECTS: tuple[AspectName, ...] = (
    "date_suitable",
    "group_suitable",
    "family_friendly",
)
# Yelp's comma-separated category snapshot contains the taxonomy label
# ``Books, Mags, Music & Video`` as three fragments after normalization.
# Those fragments are not useful standalone user requests, so the benchmark
# prefers another genuine category from the same target business.
_NON_STANDALONE_CATEGORY_FRAGMENTS = frozenset(
    {"Books", "Mags", "Music & Video"}
)


def _development_counts() -> dict[FrameFamily, int]:
    return {
        "category_only": 64,
        "category_distance": 64,
        "category_price": 48,
        "category_single_aspect": 80,
        "category_two_aspects": 64,
        "category_distance_aspect": 48,
        "occasion_context": 32,
    }


def _validation_counts() -> dict[FrameFamily, int]:
    return {
        "category_only": 16,
        "category_distance": 16,
        "category_price": 12,
        "category_single_aspect": 20,
        "category_two_aspects": 16,
        "category_distance_aspect": 12,
        "occasion_context": 8,
    }


class QueryFramePlanConfig(StrictModel):
    seed: int = 42
    development_family_counts: dict[FrameFamily, int] = Field(
        default_factory=_development_counts
    )
    validation_family_counts: dict[FrameFamily, int] = Field(
        default_factory=_validation_counts
    )
    development_language_counts: dict[Literal["zh-CN", "en-US"], int] = Field(
        default_factory=lambda: {"zh-CN": 240, "en-US": 160}
    )
    validation_language_counts: dict[Literal["zh-CN", "en-US"], int] = Field(
        default_factory=lambda: {"zh-CN": 60, "en-US": 40}
    )

    @model_validator(mode="after")
    def validate_counts(self) -> QueryFramePlanConfig:
        for split, families, languages in (
            (
                "development",
                self.development_family_counts,
                self.development_language_counts,
            ),
            (
                "validation",
                self.validation_family_counts,
                self.validation_language_counts,
            ),
        ):
            if any(value < 0 for value in [*families.values(), *languages.values()]):
                raise ValueError(f"{split} plan counts must be nonnegative")
            if set(families).difference(_FAMILIES):
                raise ValueError(f"{split} plan contains an unknown frame family")
            if sum(families.values()) != sum(languages.values()):
                raise ValueError(f"{split} frame and language counts must agree")
        return self


class AnchorQueryKnowledge(StrictModel):
    """Only target properties that were supportable at the anchor cutoff."""

    business_id: str = Field(min_length=1)
    fine_categories: tuple[str, ...]
    is_food_business: bool
    price_level: int | None = Field(default=None, ge=1, le=4)
    user_latitude: float | None = Field(default=None, ge=-90, le=90)
    user_longitude: float | None = Field(default=None, ge=-180, le=180)
    target_distance_km: float | None = Field(default=None, ge=0)
    positive_aspects: tuple[AspectName, ...]
    aspect_source_scope: Literal["selected_user_interactions"]

    @model_validator(mode="after")
    def validate_knowledge(self) -> AnchorQueryKnowledge:
        if not self.fine_categories:
            raise ValueError("target must have a fine-grained category")
        coordinates = (self.user_latitude, self.user_longitude)
        if (coordinates[0] is None) != (coordinates[1] is None):
            raise ValueError("history centroid coordinates must appear together")
        if (self.target_distance_km is None) != (coordinates[0] is None):
            raise ValueError("target distance requires a history centroid")
        if len(self.positive_aspects) != len(set(self.positive_aspects)):
            raise ValueError("positive aspects must be unique")
        return self


class AnchorKnowledgeReader(Protocol):
    def describe(self, anchor: BehaviorAnchor) -> AnchorQueryKnowledge: ...


class PlannedQueryRecommendationCase(StrictModel):
    anchor: BehaviorAnchor
    frame: QueryRecommendationFrame
    language: Literal["zh-CN", "en-US"]
    user_latitude: float | None = Field(default=None, ge=-90, le=90)
    user_longitude: float | None = Field(default=None, ge=-180, le=180)


def _condition(
    field: str,
    operator: str,
    value: str | int | float | bool,
    importance: str,
    enforcement: str,
) -> ExpectedRequestCondition:
    return ExpectedRequestCondition.model_validate(
        {
            "field": field,
            "operator": operator,
            "value": value,
            "importance": importance,
            "enforcement": enforcement,
        }
    )


def _stable_key(seed: int, family: str, anchor: BehaviorAnchor) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}\0{family}\0{anchor.source_task_id}".encode()
    ).hexdigest()
    return digest, anchor.source_task_id


def _distance_limit(distance: float | None) -> float | None:
    if distance is None:
        return None
    for limit in (2.0, 5.0, 10.0):
        if distance <= limit:
            return limit
    return None


def _eligible_for(family: FrameFamily, knowledge: AnchorQueryKnowledge) -> bool:
    if not _renderable_categories(knowledge):
        return False
    if family in {"category_distance", "category_distance_aspect"} and (
        _distance_limit(knowledge.target_distance_km) is None
    ):
        return False
    if family == "category_price" and knowledge.price_level is None:
        return False
    if family in {"category_single_aspect", "category_distance_aspect"} and not (
        knowledge.positive_aspects
    ):
        return False
    if family == "category_two_aspects" and len(knowledge.positive_aspects) < 2:
        return False
    if family == "occasion_context" and not set(knowledge.positive_aspects).intersection(
        _OCCASION_ASPECTS
    ):
        return False
    return True


def _renderable_categories(knowledge: AnchorQueryKnowledge) -> list[str]:
    return [
        value
        for value in knowledge.fine_categories
        if value not in _NON_STANDALONE_CATEGORY_FRAGMENTS
    ]


def _pick_category(knowledge: AnchorQueryKnowledge) -> str:
    renderable = _renderable_categories(knowledge)
    if not renderable:
        raise ValueError("target has no standalone category suitable for a Query")
    return sorted(renderable, key=lambda value: (len(value), value))[0]


def _pick_aspects(
    knowledge: AnchorQueryKnowledge,
    count: int,
) -> tuple[AspectName, ...]:
    return tuple(sorted(_compatible_aspects(knowledge))[:count])


def _compatible_aspects(
    knowledge: AnchorQueryKnowledge,
) -> tuple[AspectName, ...]:
    """Reject only domain-impossible Aspect/category combinations."""

    return tuple(
        aspect
        for aspect in knowledge.positive_aspects
        if aspect != "food_quality" or knowledge.is_food_business
    )


def _semantically_eligible_for(
    family: FrameFamily,
    knowledge: AnchorQueryKnowledge,
) -> bool:
    if not _eligible_for(family, knowledge):
        return False
    aspects = _compatible_aspects(knowledge)
    if family in {"category_single_aspect", "category_distance_aspect"}:
        return bool(aspects)
    if family == "category_two_aspects":
        return len(aspects) >= 2
    if family == "occasion_context":
        return bool(set(aspects).intersection(_OCCASION_ASPECTS))
    return True


def _repair_semantic_assignments(
    chosen: list[
        tuple[BehaviorAnchor, AnchorQueryKnowledge, FrameFamily, str]
    ],
    seed: int,
) -> list[tuple[BehaviorAnchor, AnchorQueryKnowledge, FrameFamily, str]]:
    """Swap families while preserving exact anchors, quotas, and determinism."""

    repaired = list(chosen)
    invalid_indexes = [
        index
        for index, (_, knowledge, family, _) in enumerate(repaired)
        if not _semantically_eligible_for(family, knowledge)
    ]
    for invalid_index in invalid_indexes:
        anchor, knowledge, family, case_id = repaired[invalid_index]
        donors = []
        for donor_index, (
            donor_anchor,
            donor_knowledge,
            donor_family,
            donor_case_id,
        ) in enumerate(repaired):
            if donor_index == invalid_index:
                continue
            if not _semantically_eligible_for(family, donor_knowledge):
                continue
            if not _semantically_eligible_for(donor_family, knowledge):
                continue
            donors.append(
                (
                    donor_index,
                    donor_anchor,
                    donor_knowledge,
                    donor_family,
                    donor_case_id,
                )
            )
        if not donors:
            raise ValueError(
                f"cannot repair incompatible frame assignment: {anchor.source_task_id}"
            )
        donors.sort(
            key=lambda item: _stable_key(
                seed,
                f"semantic-swap:{family}:{item[3]}",
                item[1],
            )
        )
        (
            donor_index,
            donor_anchor,
            donor_knowledge,
            donor_family,
            donor_case_id,
        ) = donors[0]
        repaired[invalid_index] = (anchor, knowledge, donor_family, case_id)
        repaired[donor_index] = (
            donor_anchor,
            donor_knowledge,
            family,
            donor_case_id,
        )
    if any(
        not _semantically_eligible_for(family, knowledge)
        for _, knowledge, family, _ in repaired
    ):
        raise AssertionError("semantic assignment repair did not reach a valid plan")
    return repaired


def _frame(
    anchor: BehaviorAnchor,
    knowledge: AnchorQueryKnowledge,
    family: FrameFamily,
    seed: int,
    *,
    case_id: str | None = None,
) -> QueryRecommendationFrame:
    category = _pick_category(knowledge)
    conditions = [_condition("category", "includes", category, "strong", "rank")]
    supports = ["static_category"]
    party_size = None
    location_source = "none"
    if family in {"category_distance", "category_distance_aspect"}:
        limit = _distance_limit(knowledge.target_distance_km)
        assert limit is not None
        conditions.append(
            _condition(
                "distance_km",
                "less_than_or_equal",
                limit,
                "mandatory",
                "filter",
            )
        )
        supports.append("history_location")
        location_source = "history_centroid"
    if family == "category_price":
        assert knowledge.price_level is not None
        conditions.append(
            _condition(
                "price_level",
                "less_than_or_equal",
                knowledge.price_level,
                "strong",
                "rank",
            )
        )
        supports.append("static_price")
    aspect_count = 0
    if family in {"category_single_aspect", "category_distance_aspect"}:
        aspect_count = 1
    elif family == "category_two_aspects":
        aspect_count = 2
    aspects = _pick_aspects(knowledge, aspect_count)
    if family == "occasion_context":
        compatible_aspects = _compatible_aspects(knowledge)
        aspects = (
            next(
                aspect
                for aspect in _OCCASION_ASPECTS
                if aspect in compatible_aspects
            ),
        )
        party_size = {
            "date_suitable": 2,
            "group_suitable": 6,
            "family_friendly": 4,
        }[aspects[0]]
    for aspect in aspects:
        conditions.append(_condition(aspect, "prefer", True, "strong", "rank"))
    if aspects:
        supports.append("cutoff_aspect")
    stable_case_id = case_id or hashlib.sha256(
        f"{seed}\0{anchor.source_task_id}\0{family}".encode()
    ).hexdigest()
    return QueryRecommendationFrame(
        case_id=stable_case_id,
        frame_family=family,
        conditions=conditions,
        party_size=party_size,
        location_source=location_source,
        target_support_sources=supports,
    )


def _plan_split(
    anchors: list[BehaviorAnchor],
    reader: AnchorKnowledgeReader,
    family_counts: dict[FrameFamily, int],
    language_counts: dict[str, int],
    seed: int,
) -> list[PlannedQueryRecommendationCase]:
    knowledge = {item.source_task_id: reader.describe(item) for item in anchors}
    remaining = {item.source_task_id: item for item in anchors}
    chosen: list[
        tuple[BehaviorAnchor, AnchorQueryKnowledge, FrameFamily, str]
    ] = []
    for family in _ALLOCATION_ORDER:
        count = family_counts.get(family, 0)
        eligible = [
            item
            for item in remaining.values()
            if _eligible_for(family, knowledge[item.source_task_id])
        ]
        eligible.sort(key=lambda item: _stable_key(seed, family, item))
        if len(eligible) < count:
            raise ValueError(
                f"not enough anchors for {family}: {len(eligible)}/{count}"
            )
        for anchor in eligible[:count]:
            item_knowledge = knowledge[anchor.source_task_id]
            chosen.append(
                (
                    anchor,
                    item_knowledge,
                    family,
                    hashlib.sha256(
                        f"{seed}\0{anchor.source_task_id}\0{family}".encode()
                    ).hexdigest(),
                )
            )
            del remaining[anchor.source_task_id]
    if len(chosen) != sum(family_counts.values()):
        raise AssertionError("frame allocation did not satisfy configured counts")
    chosen = _repair_semantic_assignments(chosen, seed)
    ordered = sorted(
        chosen,
        key=lambda value: _stable_key(seed, "language", value[0]),
    )
    languages = [
        language
        for language in ("zh-CN", "en-US")
        for _ in range(language_counts.get(language, 0))
    ]
    return [
        PlannedQueryRecommendationCase(
            anchor=anchor,
            frame=frame,
            language=language,
            user_latitude=(
                item_knowledge.user_latitude
                if frame.location_source == "history_centroid"
                else None
            ),
            user_longitude=(
                item_knowledge.user_longitude
                if frame.location_source == "history_centroid"
                else None
            ),
        )
        for (anchor, item_knowledge, family, case_id), language in zip(
            ordered, languages, strict=True
        )
        for frame in (
            _frame(
                anchor,
                item_knowledge,
                family,
                seed,
                case_id=case_id,
            ),
        )
    ]


def plan_query_recommendation_frames(
    anchors: tuple[BehaviorAnchor, ...],
    reader: AnchorKnowledgeReader,
    config: QueryFramePlanConfig,
) -> tuple[PlannedQueryRecommendationCase, ...]:
    """Allocate exact frame/language quotas without inventing target support."""

    development = [item for item in anchors if item.benchmark_split == "development"]
    validation = [item for item in anchors if item.benchmark_split == "validation"]
    result = _plan_split(
        development,
        reader,
        config.development_family_counts,
        config.development_language_counts,
        config.seed,
    )
    result.extend(
        _plan_split(
            validation,
            reader,
            config.validation_family_counts,
            config.validation_language_counts,
            config.seed,
        )
    )
    return tuple(sorted(result, key=lambda item: item.frame.case_id))
