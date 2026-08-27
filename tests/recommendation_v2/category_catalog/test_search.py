from yelp_agent.recommendation_v2.category_catalog import (
    CategoryCandidateSearch,
    load_fixed_category_catalog,
)


def test_steak_query_recalls_small_real_category_candidate_set() -> None:
    search = CategoryCandidateSearch(load_fixed_category_catalog())

    candidates = search.search("我今天晚上7点想吃牛排", limit=5)

    assert candidates
    assert candidates[0].category == "Steakhouses"
    assert len(candidates) <= 5
    assert all(
        load_fixed_category_catalog().is_selectable(item.category)
        for item in candidates
    )


def test_non_category_follow_up_does_not_inject_random_categories() -> None:
    search = CategoryCandidateSearch(load_fixed_category_catalog())

    assert search.search("第二家的停车情况怎么样？", limit=5) == []


def test_negated_category_is_recalled_but_not_interpreted_as_positive() -> None:
    """检索只找候选，不替大模型判断“不要牛排”的方向。"""

    search = CategoryCandidateSearch(load_fixed_category_catalog())

    candidates = search.search("今天不想吃牛排，换点别的", limit=5)

    assert candidates[0].category == "Steakhouses"
