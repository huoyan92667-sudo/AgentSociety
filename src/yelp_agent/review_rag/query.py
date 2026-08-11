"""Visible-query Aspect hints used when the Step 18 parser is conservative."""

from __future__ import annotations

from yelp_agent.reviews.schema import AspectName


_ALIASES: dict[AspectName, tuple[str, ...]] = {
    "food_quality": ("food quality", "food", "dish", "taste", "食物质量", "菜品", "味道"),
    "service": ("service", "staff", "服务", "店员"),
    "price_value": ("price value", "value for money", "worth the price", "性价比", "值不值"),
    "quiet_environment": ("quiet", "noisy", "noise", "loud", "安静", "噪音", "吵"),
    "crowded": ("crowded", "crowd", "packed", "拥挤", "人多"),
    "queue_time": ("queue", "wait time", "long wait", "排队", "等位", "等待时间"),
    "portion_size": ("portion", "serving size", "份量", "分量"),
    "parking": ("parking", "park", "停车"),
    "pet_friendly": ("pet friendly", "dog friendly", "pet", "dog", "宠物", "带狗"),
    "family_friendly": ("family friendly", "kid friendly", "children", "家庭", "亲子", "儿童"),
    "date_suitable": ("date night", "romantic", "date suitable", "约会", "浪漫"),
    "group_suitable": ("group suitable", "large group", "group dinner", "聚餐", "团体", "多人"),
    "spiciness": ("spiciness", "spicy", "spice level", "辣度", "辣"),
    "cleanliness": ("cleanliness", "clean", "hygiene", "卫生", "干净", "清洁"),
}


def infer_review_aspects(query_text: str) -> list[AspectName]:
    """Return deterministic Aspect hints from user-visible wording only."""

    text = query_text.casefold()
    return [
        aspect
        for aspect, aliases in _ALIASES.items()
        if any(alias.casefold() in text for alias in aliases)
    ]
