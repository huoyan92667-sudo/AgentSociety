"""Deep Step 20 module: one catalog and config in, one complete benchmark out."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from yelp_agent.config import AgentBenchmarkConfig
from yelp_agent.features.location import haversine_km
from yelp_agent.query.benchmark import ExpectedRequestCondition, QueryBenchmarkCase

from .rewriting import DeterministicScenarioRewriter, ScenarioRewriter
from .schema import (
    EvidenceLabel,
    InformationGap,
    ScenarioCategory,
    ScenarioGroundTruth,
    ScenarioLanguage,
    ScenarioSplit,
    ScenarioTaskType,
    ScriptedUserTurn,
    VisibleAgentScenario,
)
from .sources import (
    BenchmarkCatalog,
    BusinessRecord,
    ReviewEvidenceRecord,
    UserContextRecord,
)


_BROAD_CATEGORIES = {
    "Restaurants",
    "Food",
    "Nightlife",
    "Shopping",
    "Bars",
}
_ASPECT_ZH = {
    "quiet_environment": "安静程度",
    "crowded": "拥挤程度",
    "queue_time": "排队时间",
    "parking": "停车便利性",
    "pet_friendly": "宠物友好程度",
    "family_friendly": "家庭友好程度",
    "date_suitable": "约会氛围",
    "group_suitable": "聚餐适合程度",
    "spiciness": "辣度",
    "cleanliness": "卫生情况",
    "food_quality": "食物质量",
    "service": "服务",
    "price_value": "性价比",
}


@dataclass(frozen=True, slots=True)
class AgentBenchmarkBundle:
    visible_scenarios: tuple[VisibleAgentScenario, ...]
    ground_truth: tuple[ScenarioGroundTruth, ...]
    evidence_labels: tuple[EvidenceLabel, ...]


def _condition(
    field: str,
    operator: str,
    value: str | int | float | bool,
    *,
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


def _scenario_id(
    *,
    split: ScenarioSplit,
    category: ScenarioCategory,
    ordinal: int,
    context: UserContextRecord,
    business_ids: Iterable[str],
) -> str:
    payload = json.dumps(
        {
            "split": split,
            "category": category,
            "ordinal": ordinal,
            "user_id": context.user_id,
            "cutoff_time": context.cutoff_time.isoformat(),
            "business_ids": list(business_ids),
            "version": "1.0.0",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _language(
    ordinal: int,
    *,
    split: ScenarioSplit,
    category_count: int,
) -> ScenarioLanguage:
    development_count = category_count * 4 // 5
    global_ordinal = ordinal + (
        development_count if split == "validation" else 0
    )
    return "en-US" if global_ordinal % 5 in {3, 4} else "zh-CN"


def _price_level(business: BusinessRecord) -> int | None:
    raw = business.attributes.get("RestaurantsPriceRange2")
    try:
        value = int(str(raw).strip().strip("'\""))
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 4 else None


def _fine_categories(business: BusinessRecord) -> tuple[str, ...]:
    values = tuple(
        category for category in business.categories if category not in _BROAD_CATEGORIES
    )
    return values or business.categories


class _BuildState:
    def __init__(
        self,
        catalog: BenchmarkCatalog,
        config: AgentBenchmarkConfig,
        rewriter: ScenarioRewriter,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.rewriter = rewriter
        self.context_cursor: dict[ScenarioSplit, int] = {
            "development": 0,
            "validation": 0,
        }
        self.used_query_case_ids: set[str] = set()
        self.visible: list[VisibleAgentScenario] = []
        self.truth: list[ScenarioGroundTruth] = []
        self.evidence: list[EvidenceLabel] = []

    def context(self, split: ScenarioSplit) -> UserContextRecord:
        cursor = self.context_cursor[split]
        contexts = self.catalog.user_contexts[split]
        if cursor >= len(contexts):
            raise ValueError(f"exhausted {split} user contexts")
        self.context_cursor[split] += 1
        return contexts[cursor]

    def business_pool(
        self,
        split: ScenarioSplit,
        cutoff: datetime,
    ) -> list[BusinessRecord]:
        values = [
            business
            for business in self.catalog.businesses[split]
            if business.first_review_time < cutoff
        ]
        if len(values) < self.config.candidate_scope_size:
            raise ValueError(f"not enough pre-cutoff businesses for {split}")
        return values

    def scope(
        self,
        split: ScenarioSplit,
        cutoff: datetime,
        *,
        ordinal: int,
        preferred_category: str | None = None,
    ) -> tuple[BusinessRecord, ...]:
        pool = self.business_pool(split, cutoff)
        offset = (ordinal * 37) % len(pool)
        rotated = pool[offset:] + pool[:offset]
        preferred = (
            [item for item in rotated if preferred_category in item.categories]
            if preferred_category is not None
            else []
        )
        ordered = preferred + [item for item in rotated if item not in preferred]
        return tuple(ordered[: self.config.candidate_scope_size])

    def query_seed(
        self,
        *,
        split: ScenarioSplit,
        language: ScenarioLanguage,
        mode: str,
    ) -> QueryBenchmarkCase:
        for case in self.catalog.query_cases:
            if (
                case.case_id in self.used_query_case_ids
                or case.split != split
                or case.language != language
            ):
                continue
            if mode == "hard":
                valid = not case.expected_missing_fields and any(
                    condition.enforcement == "filter"
                    and condition.field == "category"
                    and condition.operator == "includes"
                    for condition in case.expected_conditions
                )
            else:
                fields = set(case.expected_missing_fields)
                valid = bool(fields.intersection({"user_location", "budget_precision"}))
            if valid:
                self.used_query_case_ids.add(case.case_id)
                return case
        raise ValueError(f"no unused {split}/{language} query seed for {mode}")

    def events_before(
        self,
        business_id: str,
        cutoff: datetime,
        *,
        aspect: str | None = None,
    ) -> list[ReviewEvidenceRecord]:
        return [
            event
            for event in self.catalog.evidence_by_business.get(business_id, ())
            if event.review_time < cutoff and (aspect is None or event.aspect == aspect)
        ]

    def evidence_business(
        self,
        split: ScenarioSplit,
        cutoff: datetime,
        *,
        ordinal: int,
        mode: str = "any",
    ) -> tuple[BusinessRecord, str, list[ReviewEvidenceRecord]]:
        pool = self.business_pool(split, cutoff)
        offset = (ordinal * 53) % len(pool)
        rotated = pool[offset:] + pool[:offset]
        for business in rotated:
            by_aspect: defaultdict[str, list[ReviewEvidenceRecord]] = defaultdict(list)
            for event in self.events_before(business.business_id, cutoff):
                by_aspect[event.aspect].append(event)
            for aspect in sorted(by_aspect):
                events = by_aspect[aspect]
                directional = {event.sentiment for event in events}.intersection(
                    {"positive", "negative"}
                )
                if mode == "conflict" and directional != {"positive", "negative"}:
                    continue
                if mode == "sparse" and len(events) != 1:
                    continue
                return business, aspect, events
        if mode != "any":
            return self.evidence_business(
                split,
                cutoff,
                ordinal=ordinal,
                mode="any",
            )
        raise ValueError(f"no pre-cutoff evidence business for {split}")

    def evidence_pair(
        self,
        split: ScenarioSplit,
        cutoff: datetime,
        *,
        ordinal: int,
    ) -> tuple[
        BusinessRecord,
        BusinessRecord,
        str,
        list[ReviewEvidenceRecord],
        list[ReviewEvidenceRecord],
    ]:
        grouped: defaultdict[str, list[tuple[BusinessRecord, list[ReviewEvidenceRecord]]]] = (
            defaultdict(list)
        )
        for business in self.business_pool(split, cutoff):
            by_aspect: defaultdict[str, list[ReviewEvidenceRecord]] = defaultdict(list)
            for event in self.events_before(business.business_id, cutoff):
                by_aspect[event.aspect].append(event)
            for aspect, events in by_aspect.items():
                grouped[aspect].append((business, events))
        choices = [
            (aspect, values)
            for aspect, values in sorted(grouped.items())
            if len(values) >= 2
        ]
        if not choices:
            raise ValueError(f"no comparable evidence pair for {split}")
        aspect, values = choices[ordinal % len(choices)]
        first = values[(ordinal * 2) % len(values)]
        second = values[(ordinal * 2 + 1) % len(values)]
        if first[0].business_id == second[0].business_id:
            second = values[(ordinal * 2 + 2) % len(values)]
        return first[0], second[0], aspect, first[1], second[1]

    def labels(
        self,
        *,
        scenario_id: str,
        scope: tuple[str, ...],
        selected: dict[str, tuple[str, list[ReviewEvidenceRecord]]],
        cutoff: datetime,
        add_decoy: bool = True,
    ) -> list[EvidenceLabel]:
        labels: list[EvidenceLabel] = []
        for business_id, (aspect, events) in selected.items():
            for event in events[: self.config.maximum_evidence_labels_per_business]:
                stance = (
                    "supports"
                    if event.sentiment == "positive"
                    else "contradicts"
                    if event.sentiment == "negative"
                    else "neutral"
                )
                labels.append(
                    EvidenceLabel(
                        scenario_id=scenario_id,
                        business_id=business_id,
                        source_type="review",
                        review_id=event.review_id,
                        aspect=aspect,
                        relevance="relevant",
                        stance=stance,
                        event_time=event.review_time,
                        confidence=event.confidence,
                        source_text_sha256=event.source_text_sha256,
                    )
                )
        if add_decoy:
            for business in self.catalog.businesses[
                "validation" if scope and scope[0] in {
                    item.business_id for item in self.catalog.businesses["validation"]
                } else "development"
            ]:
                if business.business_id in scope:
                    continue
                events = self.events_before(business.business_id, cutoff)
                if events:
                    event = events[0]
                    labels.append(
                        EvidenceLabel(
                            scenario_id=scenario_id,
                            business_id=business.business_id,
                            source_type="review",
                            review_id=event.review_id,
                            aspect=event.aspect,
                            relevance="out_of_scope",
                            stance="neutral",
                            event_time=event.review_time,
                            confidence=event.confidence,
                            source_text_sha256=event.source_text_sha256,
                        )
                    )
                    break
        return labels

    def add(
        self,
        *,
        context: UserContextRecord,
        split: ScenarioSplit,
        category: ScenarioCategory,
        ordinal: int,
        language: ScenarioLanguage,
        query: str,
        frame_family: str,
        task_type: ScenarioTaskType,
        conditions: list[ExpectedRequestCondition],
        gaps: list[InformationGap],
        allowed: list[str],
        required: list[str],
        forbidden: list[str],
        business_scope: list[str],
        acceptable: list[str],
        uncertainty: str,
        references: list[str] | None = None,
        source_query_case_id: str | None = None,
        source_generator_kind: str | None = None,
        source_generator_model: str | None = None,
        source_generator_prompt_sha256: str | None = None,
        scripted_turns: list[ScriptedUserTurn] | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        evidence: list[EvidenceLabel] | None = None,
    ) -> str:
        scenario_id = _scenario_id(
            split=split,
            category=category,
            ordinal=ordinal,
            context=context,
            business_ids=business_scope,
        )
        rewritten = self.rewriter.rewrite(
            scenario_id=scenario_id,
            language=language,
            visible_query=query,
        )
        generator_kind = rewritten.generator_kind
        generator_model = rewritten.generator_model
        generator_prompt_sha256 = rewritten.prompt_sha256
        if rewritten.generator_kind == "deterministic" and source_generator_kind:
            generator_kind = source_generator_kind
            generator_model = source_generator_model
            generator_prompt_sha256 = source_generator_prompt_sha256
        self.visible.append(
            VisibleAgentScenario(
                scenario_id=scenario_id,
                split=split,
                language=language,
                user_id=context.user_id,
                session_id=f"agent-benchmark:{scenario_id[:16]}",
                cutoff_time=context.cutoff_time,
                query_text=rewritten.text,
                user_latitude=latitude,
                user_longitude=longitude,
                referenced_business_ids=references or [],
                generator_kind=generator_kind,
                generator_model=generator_model,
                generator_prompt_sha256=generator_prompt_sha256,
            )
        )
        self.truth.append(
            ScenarioGroundTruth(
                scenario_id=scenario_id,
                scenario_category=category,
                frame_family=f"{split}:{frame_family}",
                source_query_case_id=source_query_case_id,
                source_task_id=context.task_id,
                source_profile_id=context.profile_id,
                task_type=task_type,
                expected_conditions=conditions,
                expected_information_gaps=gaps,
                allowed_actions=allowed,
                required_actions=required,
                forbidden_actions=forbidden,
                business_scope=business_scope,
                acceptable_business_ids=acceptable,
                scripted_user_turns=scripted_turns or [],
                uncertainty_policy=uncertainty,
                current_request_overrides_profile=category == "profile_conflict",
            )
        )
        if evidence:
            self.evidence.extend(
                label.model_copy(update={"scenario_id": scenario_id})
                for label in evidence
            )
        return scenario_id


def _split_counts(total: int) -> dict[ScenarioSplit, int]:
    return {"development": total * 4 // 5, "validation": total // 5}


def _matching_category(
    conditions: Iterable[ExpectedRequestCondition],
) -> str | None:
    for condition in conditions:
        if (
            condition.field == "category"
            and condition.operator == "includes"
            and condition.enforcement == "filter"
        ):
            return str(condition.value)
    return None


def _eligible(
    scope: Iterable[BusinessRecord],
    conditions: Iterable[ExpectedRequestCondition],
    *,
    latitude: float | None,
    longitude: float | None,
) -> list[str]:
    result: list[str] = []
    for business in scope:
        valid = True
        for condition in conditions:
            if condition.enforcement != "filter":
                continue
            if condition.field == "category":
                present = str(condition.value) in business.categories
                if condition.operator == "includes" and not present:
                    valid = False
                if condition.operator == "excludes" and present:
                    valid = False
            elif condition.field == "price_level":
                price = _price_level(business)
                if price is None:
                    valid = False
                elif condition.operator == "less_than_or_equal" and price > float(
                    condition.value
                ):
                    valid = False
            elif condition.field == "distance_km":
                if (
                    latitude is None
                    or longitude is None
                    or business.latitude is None
                    or business.longitude is None
                ):
                    valid = False
                elif haversine_km(
                    latitude,
                    longitude,
                    business.latitude,
                    business.longitude,
                ) > float(condition.value):
                    valid = False
        if valid:
            result.append(business.business_id)
    return result


def _build_hard_constraints(state: _BuildState) -> None:
    category: ScenarioCategory = "hard_constraint"
    for split, count in _split_counts(state.config.category_counts[category]).items():
        for ordinal in range(count):
            context = state.context(split)
            language = _language(
                ordinal,
                split=split,
                category_count=state.config.category_counts[category],
            )
            try:
                seed = state.query_seed(split=split, language=language, mode="hard")
            except ValueError:
                seed = None
            if seed is None:
                anchor = state.scope(
                    split,
                    context.cutoff_time,
                    ordinal=ordinal + 400,
                )[0]
                desired = _fine_categories(anchor)[0]
                conditions = [
                    _condition(
                        "category",
                        "includes",
                        desired,
                        importance="mandatory",
                        enforcement="filter",
                    )
                ]
                query = (
                    f"我们{2 + ordinal % 7}个人在{anchor.postal_code or '费城'}附近，必须找{desired}，其他类别不要推荐。"
                    if language == "zh-CN"
                    else f"For {2 + ordinal % 7} people near {anchor.postal_code or 'Philadelphia'}, it must be {desired}; exclude other categories."
                )
                source_case_id = None
                latitude, longitude = context.latitude, context.longitude
                frame_family = f"hard-generated:{ordinal % 15}"
            else:
                desired = _matching_category(seed.expected_conditions)
                conditions = list(seed.expected_conditions)
                query = seed.query_text
                source_case_id = seed.case_id
                latitude = seed.user_latitude
                longitude = seed.user_longitude
                if (
                    latitude is None
                    and "user_location" not in seed.expected_missing_fields
                ):
                    latitude, longitude = context.latitude, context.longitude
                frame_family = f"hard:{seed.frame_family}"
            scope = state.scope(
                split,
                context.cutoff_time,
                ordinal=ordinal,
                preferred_category=desired,
            )
            acceptable = _eligible(
                scope,
                conditions,
                latitude=latitude,
                longitude=longitude,
            )
            if desired and not acceptable:
                raise ValueError(f"hard-constraint scene has no valid {desired} business")
            state.add(
                context=context,
                split=split,
                category=category,
                ordinal=ordinal,
                language=language,
                query=query,
                frame_family=frame_family,
                source_query_case_id=source_case_id,
                source_generator_kind=(
                    None if seed is None else seed.generator_kind
                ),
                source_generator_model=(
                    None if seed is None else seed.generator_model
                ),
                source_generator_prompt_sha256=(
                    None if seed is None else seed.generator_prompt_sha256
                ),
                task_type="recommendation_request",
                conditions=conditions,
                gaps=[],
                allowed=[
                    "retrieve_candidates",
                    "apply_hard_constraints",
                    "rank_candidates",
                    "return_recommendation",
                ],
                required=["apply_hard_constraints", "return_recommendation"],
                forbidden=["ask_clarification", "retrieve_business_reviews"],
                business_scope=[business.business_id for business in scope],
                acceptable=acceptable,
                uncertainty="proceed",
                latitude=latitude,
                longitude=longitude,
            )


def _build_profile_conflicts(state: _BuildState) -> None:
    category: ScenarioCategory = "profile_conflict"
    for split, count in _split_counts(state.config.category_counts[category]).items():
        for ordinal in range(count):
            context = state.context(split)
            language = _language(
                ordinal,
                split=split,
                category_count=state.config.category_counts[category],
            )
            pool = state.business_pool(split, context.cutoff_time)
            liked = context.preferred_categories[0]
            candidates = [
                business
                for business in pool
                if liked not in business.categories
                and any(value not in _BROAD_CATEGORIES for value in business.categories)
            ]
            primary = candidates[(ordinal * 29) % len(candidates)]
            requested = _fine_categories(primary)[0]
            scope = state.scope(
                split,
                context.cutoff_time,
                ordinal=ordinal + 700,
                preferred_category=requested,
            )
            condition = _condition(
                "category",
                "includes",
                requested,
                importance="mandatory",
                enforcement="filter",
            )
            area = primary.postal_code or "Philadelphia"
            query = (
                f"我平时常吃{liked}，但这次在{area}附近只想找{requested}，请按这次要求来。"
                if language == "zh-CN"
                else f"I often choose {liked}, but near {area} today I only want {requested}. Follow today's request."
            )
            state.add(
                context=context,
                split=split,
                category=category,
                ordinal=ordinal,
                language=language,
                query=query,
                frame_family=f"profile-conflict:{ordinal % 10}",
                task_type="recommendation_request",
                conditions=[condition],
                gaps=[],
                allowed=[
                    "retrieve_candidates",
                    "apply_hard_constraints",
                    "rank_candidates",
                    "return_recommendation",
                ],
                required=["apply_hard_constraints", "rank_candidates"],
                forbidden=["ask_clarification"],
                business_scope=[business.business_id for business in scope],
                acceptable=_eligible(
                    scope,
                    [condition],
                    latitude=context.latitude,
                    longitude=context.longitude,
                ),
                uncertainty="proceed",
                latitude=context.latitude,
                longitude=context.longitude,
            )


def _scripted_clarification(
    *,
    language: ScenarioLanguage,
    gaps: list[InformationGap],
) -> list[ScriptedUserTurn]:
    answers: list[str] = []
    updates: dict[str, str | int | float | bool] = {}
    if "missing_location" in gaps:
        answers.append("我在费城市政厅附近" if language == "zh-CN" else "I am near Philadelphia City Hall")
        updates["user_location"] = "Philadelphia City Hall"
    if "missing_budget" in gaps:
        answers.append("人均不超过40美元" if language == "zh-CN" else "No more than $40 per person")
        updates["budget_per_person"] = 40
    if "missing_party_size" in gaps:
        answers.append("一共8个人" if language == "zh-CN" else "There will be 8 people")
        updates["party_size"] = 8
    if "ambiguous_reference" in gaps:
        answers.append("我指的是上一轮第一家" if language == "zh-CN" else "I mean the first place from the previous result")
        updates["reference_resolution"] = "first_previous_candidate"
    if "constraint_conflict" in gaps:
        answers.append("保留想要酒吧，取消排除酒吧" if language == "zh-CN" else "Keep Bars and remove the exclusion")
        updates["conflict_resolution"] = "keep_includes"
    return [
        ScriptedUserTurn(
            turn_index=2,
            trigger_action="ask_clarification",
            query_text="；".join(answers) if language == "zh-CN" else "; ".join(answers),
            expected_task_type="recommendation_request",
            expected_information_gaps=[],
            state_updates=updates,
            expected_allowed_actions=[
                "retrieve_candidates",
                "apply_hard_constraints",
                "rank_candidates",
                "return_recommendation",
            ],
        )
    ]


def _build_information_gaps(state: _BuildState) -> None:
    category: ScenarioCategory = "information_gap"
    for split, count in _split_counts(state.config.category_counts[category]).items():
        seed_count = count * 4 // 7
        for ordinal in range(count):
            context = state.context(split)
            language = _language(
                ordinal,
                split=split,
                category_count=state.config.category_counts[category],
            )
            source_case_id: str | None = None
            conditions: list[ExpectedRequestCondition] = []
            if ordinal < seed_count:
                seed = state.query_seed(split=split, language=language, mode="missing")
                source_case_id = seed.case_id
                query = seed.query_text
                conditions = list(seed.expected_conditions)
                gaps: list[InformationGap] = []
                if "user_location" in seed.expected_missing_fields:
                    gaps.append("missing_location")
                if "budget_precision" in seed.expected_missing_fields:
                    gaps.append("missing_budget")
                frame = f"query-gap:{seed.frame_family}"
                task_type: ScenarioTaskType = "recommendation_request"
            else:
                mode = (ordinal - seed_count) % 3
                if mode == 0:
                    query = (
                        f"想在{context.preferred_categories[0]}里找一家适合聚餐的，但我还没说人数。"
                        if language == "zh-CN"
                        else f"Find a {context.preferred_categories[0]} place for a group; I have not said how many people."
                    )
                    gaps = ["missing_party_size"]
                    frame = "missing-party-size"
                    task_type = "recommendation_request"
                elif mode == 1:
                    query = (
                        (
                            f"当前没有候选清单。我们{2 + ordinal % 9}个人原本想吃{context.preferred_categories[0]}，我说的第一家停车方便吗？"
                            if split == "validation"
                            else f"我们{2 + ordinal % 9}个人原本想吃{context.preferred_categories[0]}，第一家停车方便吗？但当前没有候选列表。"
                        )
                        if language == "zh-CN"
                        else (
                            f"No candidate list is available. For {2 + ordinal % 9} people seeking {context.preferred_categories[0]}, does the place I called first have convenient parking?"
                            if split == "validation"
                            else f"For {2 + ordinal % 9} people looking for {context.preferred_categories[0]}, does the first place have convenient parking? There is no candidate list in this conversation."
                        )
                    )
                    gaps = ["ambiguous_reference"]
                    frame = "ambiguous-reference"
                    task_type = "business_detail_question"
                else:
                    area = context.preferred_categories[0]
                    query = (
                        (
                            f"先别推荐：我平时常吃{area}，这次却同时要求必须是酒吧和排除全部酒吧，请帮我消除矛盾。"
                            if split == "validation"
                            else f"我原本常吃{area}，但这次要求必须是酒吧，同时又必须排除所有酒吧，请先确认冲突。"
                        )
                        if language == "zh-CN"
                        else (
                            f"Do not recommend yet: I usually choose {area}, but I simultaneously require a bar and exclude every bar. Resolve that contradiction."
                            if split == "validation"
                            else f"I usually choose {area}, but this time it must be a bar and must exclude all bars; clarify the conflict first."
                        )
                    )
                    conditions = [
                        _condition(
                            "category",
                            "includes",
                            "Bars",
                            importance="mandatory",
                            enforcement="filter",
                        ),
                        _condition(
                            "category",
                            "excludes",
                            "Bars",
                            importance="mandatory",
                            enforcement="filter",
                        ),
                    ]
                    gaps = ["constraint_conflict"]
                    frame = "constraint-conflict"
                    task_type = "recommendation_request"
            state.add(
                context=context,
                split=split,
                category=category,
                ordinal=ordinal,
                language=language,
                query=query,
                frame_family=frame,
                source_query_case_id=source_case_id,
                source_generator_kind=(
                    None if source_case_id is None else seed.generator_kind
                ),
                source_generator_model=(
                    None if source_case_id is None else seed.generator_model
                ),
                source_generator_prompt_sha256=(
                    None
                    if source_case_id is None
                    else seed.generator_prompt_sha256
                ),
                task_type=task_type,
                conditions=conditions,
                gaps=gaps,
                allowed=["ask_clarification", "safe_fallback"],
                required=["ask_clarification"],
                forbidden=["return_recommendation", "retrieve_business_reviews"],
                business_scope=[],
                acceptable=[],
                uncertainty="clarify_before_action",
                scripted_turns=_scripted_clarification(language=language, gaps=gaps),
            )


def _review_labels(
    state: _BuildState,
    *,
    split: ScenarioSplit,
    scenario_id: str,
    scope: tuple[str, ...],
    selected: dict[str, tuple[str, list[ReviewEvidenceRecord]]],
    cutoff: datetime,
) -> list[EvidenceLabel]:
    return state.labels(
        scenario_id=scenario_id,
        scope=scope,
        selected=selected,
        cutoff=cutoff,
    )


def _build_business_details(state: _BuildState) -> None:
    category: ScenarioCategory = "business_detail"
    for split, count in _split_counts(state.config.category_counts[category]).items():
        for ordinal in range(count):
            context = state.context(split)
            language = _language(
                ordinal,
                split=split,
                category_count=state.config.category_counts[category],
            )
            mode = ordinal % 8
            provisional_labels: list[EvidenceLabel] = []
            if mode < 3:
                business = state.scope(
                    split,
                    context.cutoff_time,
                    ordinal=ordinal + 1300,
                )[0]
                query = (
                    f"{business.name}属于什么类型的商家？"
                    if language == "zh-CN"
                    else f"What type of business is {business.name}?"
                )
                task_type: ScenarioTaskType = "business_detail_question"
                uncertainty = "proceed"
                allowed = ["get_business_details", "return_grounded_answer"]
                required = allowed
                provisional_labels = [
                    EvidenceLabel(
                        scenario_id="0" * 64,
                        business_id=business.business_id,
                        source_type="business_attribute",
                        source_field="categories",
                        aspect=None,
                        relevance="relevant",
                        stance="supports",
                        confidence=1.0,
                    )
                ]
            elif mode < 6:
                business, aspect, events = state.evidence_business(
                    split,
                    context.cutoff_time,
                    ordinal=ordinal,
                )
                label = _ASPECT_ZH.get(aspect, aspect)
                query = (
                    f"评论里大家觉得{business.name}的{label}怎么样？"
                    if language == "zh-CN"
                    else f"What do reviews say about {business.name}'s {aspect.replace('_', ' ')}?"
                )
                task_type = "review_experience_question"
                uncertainty = "proceed" if len(events) >= 2 else "answer_with_caveat"
                allowed = [
                    "get_business_details",
                    "retrieve_business_reviews",
                    "return_grounded_answer",
                    "return_uncertain_answer",
                ]
                required = ["retrieve_business_reviews"]
            else:
                business = state.scope(
                    split,
                    context.cutoff_time,
                    ordinal=ordinal + 1700,
                )[0]
                aspect = "pet_friendly"
                events = []
                query = (
                    f"{business.name}现在官方允许带宠物吗？"
                    if language == "zh-CN"
                    else f"Does {business.name} officially allow pets right now?"
                )
                task_type = "official_policy_question"
                uncertainty = "require_official_verification"
                allowed = [
                    "get_business_details",
                    "check_official_source",
                    "return_uncertain_answer",
                ]
                required = ["check_official_source", "return_uncertain_answer"]
            scenario_id = state.add(
                context=context,
                split=split,
                category=category,
                ordinal=ordinal,
                language=language,
                query=query,
                frame_family=f"business-detail:{mode}",
                task_type=task_type,
                conditions=[],
                gaps=[],
                allowed=allowed,
                required=required,
                forbidden=["retrieve_candidates", "rank_candidates"],
                business_scope=[business.business_id],
                acceptable=[],
                uncertainty=uncertainty,
                references=[business.business_id],
                evidence=provisional_labels,
            )
            if mode in {3, 4, 5}:
                state.evidence.extend(
                    _review_labels(
                        state,
                        split=split,
                        scenario_id=scenario_id,
                        scope=(business.business_id,),
                        selected={business.business_id: (aspect, events)},
                        cutoff=context.cutoff_time,
                    )
                )


def _build_comparisons(state: _BuildState) -> None:
    category: ScenarioCategory = "candidate_comparison"
    for split, count in _split_counts(state.config.category_counts[category]).items():
        for ordinal in range(count):
            context = state.context(split)
            language = _language(
                ordinal,
                split=split,
                category_count=state.config.category_counts[category],
            )
            first, second, aspect, first_events, second_events = state.evidence_pair(
                split,
                context.cutoff_time,
                ordinal=ordinal,
            )
            query = (
                f"{first.name}和{second.name}哪个在{_ASPECT_ZH.get(aspect, aspect)}方面评价更好？"
                if language == "zh-CN"
                else f"Which is reviewed better for {aspect.replace('_', ' ')}: {first.name} or {second.name}?"
            )
            scenario_id = state.add(
                context=context,
                split=split,
                category=category,
                ordinal=ordinal,
                language=language,
                query=query,
                frame_family=f"candidate-comparison:{aspect}",
                task_type="candidate_comparison",
                conditions=[],
                gaps=[],
                allowed=[
                    "get_business_details",
                    "retrieve_business_reviews",
                    "compare_candidates",
                    "return_grounded_answer",
                    "return_uncertain_answer",
                ],
                required=["retrieve_business_reviews", "compare_candidates"],
                forbidden=["retrieve_candidates", "rank_candidates"],
                business_scope=[first.business_id, second.business_id],
                acceptable=[],
                uncertainty="proceed",
                references=[first.business_id, second.business_id],
            )
            state.evidence.extend(
                _review_labels(
                    state,
                    split=split,
                    scenario_id=scenario_id,
                    scope=(first.business_id, second.business_id),
                    selected={
                        first.business_id: (aspect, first_events),
                        second.business_id: (aspect, second_events),
                    },
                    cutoff=context.cutoff_time,
                )
            )


def _build_multi_turn(state: _BuildState) -> None:
    category: ScenarioCategory = "multi_turn_feedback"
    for split, count in _split_counts(state.config.category_counts[category]).items():
        for ordinal in range(count):
            context = state.context(split)
            language = _language(
                ordinal,
                split=split,
                category_count=state.config.category_counts[category],
            )
            scope = state.scope(
                split,
                context.cutoff_time,
                ordinal=ordinal + 2300,
            )
            primary = scope[0]
            requested = _fine_categories(primary)[0]
            party_size = 2 + ordinal % 8
            query = (
                f"我们{party_size}个人在{primary.postal_code or '费城'}附近，先推荐一家{requested}。"
                if language == "zh-CN"
                else f"There are {party_size} of us near {primary.postal_code or 'Philadelphia'}; first recommend a {requested} place."
            )
            category_condition = _condition(
                "category",
                "includes",
                requested,
                importance="strong",
                enforcement="rank",
            )
            turns = [
                ScriptedUserTurn(
                    turn_index=2,
                    trigger_action="return_recommendation",
                    query_text="太贵了，换便宜一点的。" if language == "zh-CN" else "That is too expensive; choose something cheaper.",
                    expected_task_type="feedback_refinement",
                    added_conditions=[
                        _condition(
                            "price_level",
                            "less_than_or_equal",
                            2,
                            importance="strong",
                            enforcement="rank",
                        )
                    ],
                    state_updates={"maximum_price_level": 2},
                    rejected_business_ids=[primary.business_id],
                    expected_allowed_actions=[
                        "apply_feedback",
                        "rank_candidates",
                        "return_recommendation",
                    ],
                ),
                ScriptedUserTurn(
                    turn_index=3,
                    trigger_action="return_recommendation",
                    query_text="再近一点。" if language == "zh-CN" else "Make it closer as well.",
                    expected_task_type="feedback_refinement",
                    added_conditions=[
                        _condition(
                            "distance_km",
                            "less_than_or_equal",
                            3.0,
                            importance="strong",
                            enforcement="rank",
                        )
                    ],
                    state_updates={"maximum_distance_km": 3.0},
                    expected_allowed_actions=[
                        "apply_feedback",
                        "rank_candidates",
                        "return_recommendation",
                    ],
                ),
                ScriptedUserTurn(
                    turn_index=4,
                    trigger_action="return_recommendation",
                    query_text="而且不要连锁店。" if language == "zh-CN" else "And exclude chain restaurants.",
                    expected_task_type="feedback_refinement",
                    state_updates={"exclude_chain_businesses": True},
                    expected_allowed_actions=[
                        "apply_feedback",
                        "apply_hard_constraints",
                        "rank_candidates",
                        "return_recommendation",
                    ],
                ),
            ]
            state.add(
                context=context,
                split=split,
                category=category,
                ordinal=ordinal,
                language=language,
                query=query,
                frame_family=f"multi-turn:{ordinal % 12}",
                task_type="recommendation_request",
                conditions=[category_condition],
                gaps=[],
                allowed=[
                    "retrieve_candidates",
                    "rank_candidates",
                    "return_recommendation",
                ],
                required=["rank_candidates", "return_recommendation"],
                forbidden=["retrieve_business_reviews"],
                business_scope=[business.business_id for business in scope],
                acceptable=_eligible(
                    scope,
                    [category_condition],
                    latitude=context.latitude,
                    longitude=context.longitude,
                ),
                uncertainty="proceed",
                scripted_turns=turns,
                latitude=context.latitude,
                longitude=context.longitude,
            )


def _build_uncertainty(state: _BuildState) -> None:
    category: ScenarioCategory = "evidence_uncertainty"
    for split, count in _split_counts(state.config.category_counts[category]).items():
        for ordinal in range(count):
            context = state.context(split)
            language = _language(
                ordinal,
                split=split,
                category_count=state.config.category_counts[category],
            )
            mode = ordinal % 3
            if mode == 2:
                business = state.scope(
                    split,
                    context.cutoff_time,
                    ordinal=ordinal + 3100,
                )[0]
                query = (
                    f"{business.name}现在官方允许带宠物吗？只凭 Yelp 历史信息能确认吗？"
                    if language == "zh-CN"
                    else f"Does {business.name} officially allow pets now, and can historical Yelp data confirm it?"
                )
                task_type: ScenarioTaskType = "official_policy_question"
                uncertainty = "require_official_verification"
                allowed = [
                    "get_business_details",
                    "check_official_source",
                    "return_uncertain_answer",
                ]
                required = ["check_official_source", "return_uncertain_answer"]
                selected: dict[str, tuple[str, list[ReviewEvidenceRecord]]] = {}
                aspect = "pet_friendly"
            else:
                business, aspect, events = state.evidence_business(
                    split,
                    context.cutoff_time,
                    ordinal=ordinal,
                    mode="conflict" if mode == 0 else "sparse",
                )
                if mode == 0:
                    query = (
                        f"{business.name}关于{_ASPECT_ZH.get(aspect, aspect)}的新旧评论说法冲突，我该相信哪边？"
                        if language == "zh-CN"
                        else f"Reviews conflict over {business.name}'s {aspect.replace('_', ' ')}; which side is better supported?"
                    )
                    uncertainty = "report_conflict"
                else:
                    query = (
                        f"{business.name}关于{_ASPECT_ZH.get(aspect, aspect)}只有很少评论，能确定吗？"
                        if language == "zh-CN"
                        else f"There are very few reviews about {business.name}'s {aspect.replace('_', ' ')}; is the evidence sufficient?"
                    )
                    uncertainty = "answer_with_caveat"
                task_type = "review_experience_question"
                allowed = [
                    "get_business_details",
                    "retrieve_business_reviews",
                    "return_uncertain_answer",
                ]
                required = ["retrieve_business_reviews", "return_uncertain_answer"]
                selected = {business.business_id: (aspect, events)}
            scenario_id = state.add(
                context=context,
                split=split,
                category=category,
                ordinal=ordinal,
                language=language,
                query=query,
                frame_family=f"evidence-uncertainty:{mode}:{aspect}",
                task_type=task_type,
                conditions=[],
                gaps=[],
                allowed=allowed,
                required=required,
                forbidden=["retrieve_candidates", "rank_candidates"],
                business_scope=[business.business_id],
                acceptable=[],
                uncertainty=uncertainty,
                references=[business.business_id],
            )
            if selected:
                state.evidence.extend(
                    _review_labels(
                        state,
                        split=split,
                        scenario_id=scenario_id,
                        scope=(business.business_id,),
                        selected=selected,
                        cutoff=context.cutoff_time,
                    )
                )


def build_agent_benchmark_bundle(
    catalog: BenchmarkCatalog,
    config: AgentBenchmarkConfig,
    *,
    rewriter: ScenarioRewriter | None = None,
) -> AgentBenchmarkBundle:
    """Build all seven scene families behind one stable interface."""

    state = _BuildState(
        catalog,
        config,
        rewriter or DeterministicScenarioRewriter(),
    )
    _build_hard_constraints(state)
    _build_profile_conflicts(state)
    _build_information_gaps(state)
    _build_business_details(state)
    _build_comparisons(state)
    _build_multi_turn(state)
    _build_uncertainty(state)
    return AgentBenchmarkBundle(
        visible_scenarios=tuple(state.visible),
        ground_truth=tuple(state.truth),
        evidence_labels=tuple(state.evidence),
    )
