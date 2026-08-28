"""14 种评论特征的固定含义和语义粗筛示例。"""

from __future__ import annotations

from pydantic import Field, model_validator

from yelp_agent.config import ReviewAspectVocabulary
from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS, AspectField


class AspectRecallDefinition(StrictModel):
    """一种评论特征的旧词语和两端语义示例。"""

    aspect: AspectField
    meaning_zh: str = Field(min_length=1)
    keyword_terms: list[str] = Field(min_length=1)
    semantic_anchors: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_unique_values(self) -> AspectRecallDefinition:
        for values in (self.keyword_terms, self.semantic_anchors):
            normalized = [value.casefold() for value in values]
            if len(normalized) != len(set(normalized)):
                raise ValueError("recall terms and anchors must be unique")
        return self


class PreferenceSemanticAnchors(StrictModel):
    """一条软偏好的两条满足说法和两条相反说法。"""

    satisfying: list[str] = Field(min_length=2, max_length=2)
    contradicting: list[str] = Field(min_length=2, max_length=2)


_MEANINGS: dict[AspectField, str] = {
    "food_quality": "菜品质量从很差到很好",
    "service": "服务从很差到很好",
    "price_value": "性价比从很低到很高",
    "quiet_environment": "环境从非常吵到非常安静",
    "crowded": "客流从不拥挤到非常拥挤",
    "queue_time": "等位从无需等待到等待很久",
    "portion_size": "分量从很少到很多",
    "parking": "停车从很困难到很方便",
    "pet_friendly": "从不允许宠物到非常宠物友好",
    "family_friendly": "从不适合家庭儿童到非常适合",
    "date_suitable": "从不适合约会到非常适合约会",
    "group_suitable": "从不适合多人聚餐到非常适合",
    "spiciness": "辣度从完全不辣到非常辣",
    "cleanliness": "环境从很脏到非常干净",
}


# 两端都要覆盖。这里只用于寻找相关评论，不代表正面、负面或最终分数。
_SEMANTIC_ANCHORS: dict[AspectField, tuple[str, ...]] = {
    "food_quality": (
        "The food is delicious, fresh, flavorful, and cooked well.",
        "The dishes taste excellent and the ingredients are high quality.",
        "The food is bland, stale, badly cooked, or tastes terrible.",
        "The dishes are undercooked, overcooked, flavorless, or poor quality.",
    ),
    "service": (
        "The staff are friendly, attentive, helpful, and provide prompt service.",
        "Employees respond quickly and take good care of customers.",
        "The staff are rude, inattentive, slow, or ignore customers.",
        "Customers receive poor service or have difficulty getting help.",
    ),
    "price_value": (
        "The meal is affordable, reasonably priced, and worth the money.",
        "The portions and quality provide good value for the price.",
        "The restaurant is overpriced, expensive, or not worth the money.",
        "The price is too high for the quality or quantity received.",
    ),
    "quiet_environment": (
        "The restaurant is quiet, peaceful, calm, and suitable for conversation.",
        "Customers can talk comfortably without raising their voices.",
        "The restaurant is loud and noisy, making conversation difficult.",
        "Customers cannot hear each other or need to shout over the noise.",
    ),
    "crowded": (
        "The restaurant is uncrowded, spacious, and has plenty of room.",
        "There are few people and customers do not feel cramped.",
        "The restaurant is crowded, packed, cramped, or extremely busy.",
        "There are too many people and very little space between customers.",
    ),
    "queue_time": (
        "Customers are seated immediately or wait only a short time.",
        "There is no line and no meaningful wait for a table.",
        "Customers wait a long time in a line or queue before being seated.",
        "Getting a table takes a very long time or feels like forever.",
    ),
    "portion_size": (
        "The portions are large, generous, and provide plenty of food.",
        "Serving sizes are substantial and enough to satisfy customers.",
        "The portions are small, tiny, skimpy, or do not provide enough food.",
        "Serving sizes are much smaller than customers expect.",
    ),
    "parking": (
        "Parking is easy, convenient, free, or plentiful near the restaurant.",
        "The restaurant has a useful parking lot or many available spaces.",
        "Parking is difficult, limited, expensive, full, or unavailable.",
        "Customers struggle to find a place to park near the restaurant.",
    ),
    "pet_friendly": (
        "Dogs and other pets are welcome and treated well at the restaurant.",
        "The restaurant provides a pet-friendly space or amenities for dogs.",
        "Dogs or pets are not allowed and the restaurant is not pet friendly.",
        "Customers are refused service or seating because they brought a pet.",
    ),
    "family_friendly": (
        "The restaurant is suitable for families, children, and young kids.",
        "Children are welcome and the restaurant offers helpful family amenities.",
        "The restaurant is unsuitable for children or is not family friendly.",
        "Children are unwelcome, restricted, or have a poor experience here.",
    ),
    "date_suitable": (
        "The restaurant is romantic and well suited to a date or anniversary.",
        "The atmosphere supports an intimate meal for two people.",
        "The restaurant is not romantic and is a poor place for a date.",
        "The atmosphere makes an intimate date uncomfortable or unsuitable.",
    ),
    "group_suitable": (
        "The restaurant can comfortably accommodate large groups and parties.",
        "Group dinners are handled well with enough space and suitable seating.",
        "The restaurant cannot accommodate groups or is too small for a party.",
        "Large groups are split up, refused, or have difficulty dining together.",
    ),
    "spiciness": (
        "The food has little heat, is mild, or is not spicy.",
        "The spice level is low and gentle for customers who avoid heat.",
        "The food is spicy, very hot, or has an extremely intense heat level.",
        "The dish is painfully spicy or much hotter than customers expect.",
    ),
    "cleanliness": (
        "The restaurant, tables, and bathrooms are clean and well maintained.",
        "The dining area is spotless, hygienic, and carefully maintained.",
        "The restaurant is dirty, filthy, sticky, or unsanitary.",
        "Tables, bathrooms, or dining areas are poorly cleaned and maintained.",
    ),
}

# 这三个维度的前两条描述的是“更低”：不拥挤、少等位、少辣。
# 其余维度的前两条都描述“更高”：更好、更安静、更方便等。
_FIRST_PAIR_DIRECTION: dict[AspectField, str] = {
    **{aspect: "higher" for aspect in ASPECT_FIELDS},
    "crowded": "lower",
    "queue_time": "lower",
    "spiciness": "lower",
}


def aspect_meaning(aspect: AspectField) -> str:
    """返回固定特征的中文量尺含义，供评测和展示复用。"""

    return _MEANINGS[aspect]


def preference_semantic_anchors(
    aspect: AspectField,
    direction: str,
) -> PreferenceSemanticAnchors:
    """按用户想要的方向，返回满足和违反要求的固定检索说法。"""

    if direction not in {"higher", "lower"}:
        raise ValueError("review aspects only support higher or lower")
    anchors = _SEMANTIC_ANCHORS[aspect]
    first = list(anchors[:2])
    second = list(anchors[2:])
    if direction == _FIRST_PAIR_DIRECTION[aspect]:
        satisfying, contradicting = first, second
    else:
        satisfying, contradicting = second, first
    return PreferenceSemanticAnchors(
        satisfying=satisfying,
        contradicting=contradicting,
    )


def build_aspect_recall_definitions(
    vocabulary: ReviewAspectVocabulary,
) -> tuple[AspectRecallDefinition, ...]:
    """把旧正负词表合并成纯召回词表，并配上固定语义说明。"""

    definitions: list[AspectRecallDefinition] = []
    for aspect in ASPECT_FIELDS:
        old_terms = vocabulary.aspects[aspect]
        terms = list(dict.fromkeys((*old_terms.positive, *old_terms.negative)))
        definitions.append(
            AspectRecallDefinition(
                aspect=aspect,
                meaning_zh=_MEANINGS[aspect],
                keyword_terms=terms,
                semantic_anchors=list(_SEMANTIC_ANCHORS[aspect]),
            )
        )
    return tuple(definitions)
