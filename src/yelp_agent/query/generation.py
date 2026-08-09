"""Build and render a leakage-safe, model-generated query benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.query.benchmark import (
    ExpectedRequestCondition,
    QueryBenchmarkCase,
)
from yelp_agent.query.schema import MissingField


type QueryDifficulty = Literal[
    "simple",
    "multi_constraint",
    "negation_priority",
    "clarification",
]


class QueryFrameSpec(StrictModel):
    """One canonical meaning which the provider may phrase but not change."""

    frame_id: str = Field(min_length=1)
    split: Literal["development", "validation"]
    difficulty: QueryDifficulty
    scenario: str = Field(min_length=1)
    expected_conditions: list[ExpectedRequestCondition]
    expected_party_size: int | None = Field(default=None, ge=1, le=100)
    expected_missing_fields: list[MissingField]
    user_latitude: float | None = Field(default=None, ge=-90, le=90)
    user_longitude: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def validate_location(self) -> QueryFrameSpec:
        if (self.user_latitude is None) != (self.user_longitude is None):
            raise ValueError("frame coordinates must be both present or absent")
        return self


class GeneratedParaphrase(StrictModel):
    language: Literal["zh-CN", "en-US"]
    style: Literal[
        "direct_zh",
        "colloquial_zh",
        "implicit_zh",
        "natural_en",
        "conversational_en",
    ]
    text: str = Field(min_length=3, max_length=1000)


class GeneratedFrame(StrictModel):
    frame_id: str = Field(min_length=1)
    paraphrases: list[GeneratedParaphrase]

    @model_validator(mode="after")
    def validate_paraphrases(self) -> GeneratedFrame:
        required_styles = {
            "direct_zh",
            "colloquial_zh",
            "implicit_zh",
            "natural_en",
            "conversational_en",
        }
        styles = [item.style for item in self.paraphrases]
        if len(styles) != 5 or set(styles) != required_styles:
            raise ValueError("each frame must contain the five required styles")
        language_counts = Counter(item.language for item in self.paraphrases)
        if language_counts != {"zh-CN": 3, "en-US": 2}:
            raise ValueError("each frame must contain three Chinese and two English queries")
        texts = [item.text.strip().casefold() for item in self.paraphrases]
        if len(set(texts)) != len(texts):
            raise ValueError("paraphrases within a frame must be unique")
        return self


class GeneratedQueryBatch(StrictModel):
    frames: list[GeneratedFrame]


class QueryBenchmarkGenerationSummary(StrictModel):
    schema_version: Literal[1] = 1
    generator_kind: Literal["openai_compatible"] = "openai_compatible"
    generator_model: str
    case_count: int = Field(ge=1)
    semantic_frame_count: int = Field(ge=1)
    cases_per_frame: Literal[5] = 5
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    api_batch_count: int = Field(ge=1)
    api_attempt_count: int = Field(ge=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    total_latency_ms: float = Field(ge=0)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contains_specific_business: Literal[False] = False
    contains_future_review: Literal[False] = False


_CATEGORIES = (
    "Steakhouses",
    "Japanese",
    "Chinese",
    "Italian",
    "Mexican",
    "Thai",
    "Coffee & Tea",
    "Fast Food",
    "Bars",
    "Seafood",
)

_PREFER_ASPECTS = (
    "quiet_environment",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "cleanliness",
    "food_quality",
    "service",
    "price_value",
)

_AVOID_ASPECTS = ("crowded", "queue_time", "spiciness")

_SCENARIOS = (
    "日常用餐 / everyday meal",
    "情侣约会 / date night",
    "家庭聚餐 / family dinner",
    "朋友聚会 / friends gathering",
    "商务用餐 / business meal",
    "一个人快速用餐 / quick solo meal",
    "多人团建 / team dinner",
    "带孩子吃饭 / dining with children",
    "带宠物外出 / dining with a pet",
    "庆祝纪念日 / anniversary celebration",
)


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


def _category(
    value: str,
    *,
    importance: Literal["mandatory", "strong", "preferred"] = "strong",
    excluded: bool = False,
) -> ExpectedRequestCondition:
    mandatory = excluded or importance == "mandatory"
    return _condition(
        "category",
        "excludes" if excluded else "includes",
        value,
        "mandatory" if excluded else importance,
        "filter" if mandatory else "rank",
    )


def _aspect(
    field: str,
    *,
    importance: Literal["mandatory", "strong", "preferred"],
    avoid: bool = False,
) -> ExpectedRequestCondition:
    return _condition(
        field,
        "avoid" if avoid else "prefer",
        True,
        importance,
        "evidence" if importance == "mandatory" else "rank",
    )


def build_query_frame_specs() -> tuple[QueryFrameSpec, ...]:
    """Create 100 meanings: 20 simple, 40 compound, 20 negative, 20 unclear."""

    frames: list[QueryFrameSpec] = []

    for index in range(20):
        conditions = [_category(_CATEGORIES[index % len(_CATEGORIES)])]
        if index >= 10:
            conditions.append(
                _aspect(
                    _PREFER_ASPECTS[index % len(_PREFER_ASPECTS)],
                    importance="preferred",
                )
            )
        frames.append(
            QueryFrameSpec(
                frame_id=f"simple-{index + 1:03d}",
                split="development" if index < 16 else "validation",
                difficulty="simple",
                scenario=_SCENARIOS[index % len(_SCENARIOS)],
                expected_conditions=conditions,
                expected_missing_fields=[],
            )
        )

    for index in range(40):
        block = index // 10
        importance: Literal["mandatory", "strong", "preferred"] = (
            "mandatory"
            if (index + block) % 7 == 0
            else "strong"
            if (index + block) % 2 == 0
            else "preferred"
        )
        conditions = [
            _category(
                _CATEGORIES[(index + 2) % len(_CATEGORIES)],
                importance="mandatory" if index % 9 == 0 else "strong",
            ),
            _aspect(
                _PREFER_ASPECTS[(index + block * 2) % len(_PREFER_ASPECTS)],
                importance=importance,
            ),
            _aspect(
                _PREFER_ASPECTS[(index + 3 + block * 3) % len(_PREFER_ASPECTS)],
                importance="preferred",
            ),
        ]
        if index % 4 == 0:
            conditions.append(
                _aspect(
                    _AVOID_ASPECTS[index % len(_AVOID_ASPECTS)],
                    importance="strong",
                    avoid=True,
                )
            )
        latitude = 39.9526 if index % 10 == 0 else None
        longitude = -75.1652 if index % 10 == 0 else None
        if latitude is not None:
            conditions.append(
                _condition(
                    "distance_km",
                    "less_than_or_equal",
                    float((index % 4) + 2),
                    "mandatory",
                    "filter",
                )
            )
        frames.append(
            QueryFrameSpec(
                frame_id=f"multi-{index + 1:03d}",
                split="development" if index < 32 else "validation",
                difficulty="multi_constraint",
                scenario=_SCENARIOS[(index + block * 3 + 1) % len(_SCENARIOS)],
                expected_conditions=conditions,
                expected_party_size=(index % 7) + 2 if index % 2 == 0 else None,
                expected_missing_fields=[],
                user_latitude=latitude,
                user_longitude=longitude,
            )
        )

    for index in range(20):
        included = _CATEGORIES[(index + 4) % len(_CATEGORIES)]
        excluded = _CATEGORIES[(index + 7) % len(_CATEGORIES)]
        frames.append(
            QueryFrameSpec(
                frame_id=f"negation-{index + 1:03d}",
                split="development" if index < 16 else "validation",
                difficulty="negation_priority",
                scenario=_SCENARIOS[(index + 2) % len(_SCENARIOS)],
                expected_conditions=[
                    _category(
                        included,
                        importance="mandatory" if index % 3 == 0 else "strong",
                    ),
                    _category(excluded, excluded=True),
                    _aspect(
                        _AVOID_ASPECTS[index % len(_AVOID_ASPECTS)],
                        importance="mandatory" if index % 4 == 0 else "strong",
                        avoid=True,
                    ),
                    _aspect(
                        _PREFER_ASPECTS[(index + 5) % len(_PREFER_ASPECTS)],
                        importance="preferred",
                    ),
                ],
                expected_party_size=(index % 5) + 2 if index % 2 else None,
                expected_missing_fields=[],
            )
        )

    budgets = (30.0, 45.0, 60.0, 80.0, 100.0)
    distances = (2.0, 3.0, 5.0, 8.0)
    for index in range(20):
        mode = index % 4
        conditions: list[ExpectedRequestCondition] = []
        missing: list[MissingField] = []
        if mode in {0, 1}:
            conditions.append(_category(_CATEGORIES[(index + 6) % len(_CATEGORIES)]))
        if mode in {0, 3}:
            conditions.append(
                _condition(
                    "distance_km",
                    "less_than_or_equal",
                    distances[index % len(distances)],
                    "mandatory",
                    "clarify",
                )
            )
            missing.append("user_location")
        if mode in {1, 3}:
            conditions.append(
                _condition(
                    "budget_per_person",
                    "less_than_or_equal",
                    budgets[index % len(budgets)],
                    "mandatory",
                    "clarify",
                )
            )
            missing.append("budget_precision")
        if mode in {2, 3}:
            conditions.append(
                _aspect(
                    _PREFER_ASPECTS[(index + 7) % len(_PREFER_ASPECTS)],
                    importance="strong",
                )
            )
            missing.append("desired_category")
        frames.append(
            QueryFrameSpec(
                frame_id=f"clarify-{index + 1:03d}",
                split="development" if index < 16 else "validation",
                difficulty="clarification",
                scenario=_SCENARIOS[(index + 3) % len(_SCENARIOS)],
                expected_conditions=conditions,
                expected_party_size=(index % 6) + 2 if index % 3 == 0 else None,
                expected_missing_fields=sorted(set(missing)),
            )
        )

    if len(frames) != 100 or len({frame.frame_id for frame in frames}) != 100:
        raise AssertionError("query frame builder must produce 100 unique frames")
    semantic_signatures = {
        json.dumps(
            {
                "conditions": [
                    item.model_dump(mode="json") for item in frame.expected_conditions
                ],
                "party_size": frame.expected_party_size,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for frame in frames
    }
    if len(semantic_signatures) != 100:
        raise AssertionError("query frame builder must produce 100 distinct meanings")
    return tuple(frames)


_SYSTEM_PROMPT = """You create a bilingual benchmark for a Yelp recommendation request parser.
Return one valid JSON object only. Never use Markdown fences.
The caller supplies canonical semantic frames. You may only paraphrase them; never add,
remove, weaken, strengthen, contradict, or invent a condition. Never name a specific
business. Never mention reviews, future behavior, labels, field names, or coordinates.
For every frame produce exactly five natural user queries:
1) direct_zh in zh-CN, 2) colloquial_zh in zh-CN, 3) implicit_zh in zh-CN,
4) natural_en in en-US, 5) conversational_en in en-US.
Chinese versions must be genuinely different, not punctuation changes. English versions
must also differ in sentence structure. Preserve exact numbers and party size. Express
mandatory as a non-negotiable requirement, strong as a clear desire, and preferred as
"最好/优先/ideally/preferably". Express excludes and avoid unambiguously. If the frame
lacks a category, do not invent one. A budget value is US dollars: use 美元 in Chinese
and $ or dollars in English. Do not ask the user for missing information; simply phrase
the supplied request. Output schema:
{"frames":[{"frame_id":"...","paraphrases":[{"language":"zh-CN","style":"direct_zh","text":"..."},{"language":"zh-CN","style":"colloquial_zh","text":"..."},{"language":"zh-CN","style":"implicit_zh","text":"..."},{"language":"en-US","style":"natural_en","text":"..."},{"language":"en-US","style":"conversational_en","text":"..."}]}]}
"""


def build_generation_messages_payload(
    frames: Sequence[QueryFrameSpec],
) -> tuple[str, str, str]:
    """Return system prompt, user payload, and an auditable combined prompt hash."""

    if not frames:
        raise ValueError("generation batch cannot be empty")
    serialized = [
        {
            "frame_id": frame.frame_id,
            "conditions": [
                {
                    "field": item.field,
                    "operator": item.operator,
                    "value": item.value,
                    "importance": item.importance,
                }
                for item in frame.expected_conditions
            ],
            "party_size": frame.expected_party_size,
        }
        for frame in frames
    ]
    user_prompt = (
        "Paraphrase every supplied frame exactly once. Canonical frames:\n"
        + json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))
    )
    prompt_hash = hashlib.sha256(
        (_SYSTEM_PROMPT + "\n" + user_prompt).encode("utf-8")
    ).hexdigest()
    return _SYSTEM_PROMPT, user_prompt, prompt_hash


def parse_generated_batch(
    content: str,
    expected_frames: Sequence[QueryFrameSpec],
) -> GeneratedQueryBatch:
    """Validate provider JSON against the exact requested frame IDs."""

    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    batch = GeneratedQueryBatch.model_validate_json(stripped)
    expected_ids = [frame.frame_id for frame in expected_frames]
    actual_ids = [frame.frame_id for frame in batch.frames]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError("provider response must contain every requested frame exactly once")
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("provider response contains duplicate frame IDs")
    return batch


def expand_generated_batch(
    generated: GeneratedQueryBatch,
    frames: Sequence[QueryFrameSpec],
    *,
    model: str,
    prompt_sha256: str,
) -> tuple[QueryBenchmarkCase, ...]:
    """Attach locally owned gold labels to provider-written paraphrases."""

    by_id = {item.frame_id: item for item in generated.frames}
    cases: list[QueryBenchmarkCase] = []
    for frame in frames:
        generated_frame = by_id[frame.frame_id]
        for paraphrase_index, paraphrase in enumerate(
            generated_frame.paraphrases,
            start=1,
        ):
            cases.append(
                QueryBenchmarkCase(
                    case_id=f"{frame.split[:3]}-{frame.frame_id}-p{paraphrase_index}",
                    split=frame.split,
                    frame_family=frame.frame_id,
                    language=paraphrase.language,
                    query_text=paraphrase.text.strip(),
                    expected_intent="recommendation_request",
                    expected_conditions=frame.expected_conditions,
                    expected_party_size=frame.expected_party_size,
                    expected_missing_fields=frame.expected_missing_fields,
                    generator_kind="openai_compatible",
                    generator_model=model,
                    generator_prompt_sha256=prompt_sha256,
                    uses_specific_business=False,
                    uses_future_review=False,
                    user_latitude=frame.user_latitude,
                    user_longitude=frame.user_longitude,
                )
            )
    return tuple(cases)


def benchmark_payload(cases: Sequence[QueryBenchmarkCase]) -> bytes:
    """Serialize the frozen dataset in stable case-id order."""

    ordered = sorted(cases, key=lambda item: item.case_id)
    return "".join(item.model_dump_json() + "\n" for item in ordered).encode("utf-8")
