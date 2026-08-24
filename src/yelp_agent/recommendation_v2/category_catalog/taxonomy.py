"""第一版固定餐饮类别层级，只使用餐厅数据中真实存在的类别名。"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import CategoryGroup, CategoryKind

GROUPS = (
    CategoryGroup(group_id="dining", label="Dining"),
    CategoryGroup(
        group_id="cuisine",
        label="Cuisine",
        parent_group_id="dining",
    ),
    CategoryGroup(
        group_id="north_american_cuisine",
        label="North American Cuisine",
        parent_group_id="cuisine",
    ),
    CategoryGroup(
        group_id="east_asian_cuisine",
        label="East Asian Cuisine",
        parent_group_id="cuisine",
    ),
    CategoryGroup(
        group_id="southeast_asian_cuisine",
        label="Southeast Asian Cuisine",
        parent_group_id="cuisine",
    ),
    CategoryGroup(
        group_id="south_asian_cuisine",
        label="South Asian Cuisine",
        parent_group_id="cuisine",
    ),
    CategoryGroup(
        group_id="middle_eastern_cuisine",
        label="Middle Eastern and Mediterranean Cuisine",
        parent_group_id="cuisine",
    ),
    CategoryGroup(
        group_id="european_cuisine",
        label="European Cuisine",
        parent_group_id="cuisine",
    ),
    CategoryGroup(
        group_id="latin_caribbean_cuisine",
        label="Latin American and Caribbean Cuisine",
        parent_group_id="cuisine",
    ),
    CategoryGroup(
        group_id="african_cuisine",
        label="African Cuisine",
        parent_group_id="cuisine",
    ),
    CategoryGroup(
        group_id="other_cuisine",
        label="Other Regional Cuisine",
        parent_group_id="cuisine",
    ),
    CategoryGroup(group_id="dish", label="Dish", parent_group_id="dining"),
    CategoryGroup(
        group_id="handheld_dish",
        label="Handheld and Bread-based Dish",
        parent_group_id="dish",
    ),
    CategoryGroup(
        group_id="noodle_rice_dish",
        label="Noodle and Rice Dish",
        parent_group_id="dish",
    ),
    CategoryGroup(
        group_id="meat_grill_dish",
        label="Meat and Grill Dish",
        parent_group_id="dish",
    ),
    CategoryGroup(
        group_id="seafood_dish",
        label="Seafood Dish",
        parent_group_id="dish",
    ),
    CategoryGroup(
        group_id="light_dish",
        label="Light Meal",
        parent_group_id="dish",
    ),
    CategoryGroup(
        group_id="dessert_bakery",
        label="Dessert and Bakery",
        parent_group_id="dining",
    ),
    CategoryGroup(
        group_id="beverage",
        label="Beverage",
        parent_group_id="dining",
    ),
    CategoryGroup(
        group_id="venue",
        label="Restaurant Format",
        parent_group_id="dining",
    ),
    CategoryGroup(
        group_id="bar_venue",
        label="Bar and Drinking Venue",
        parent_group_id="venue",
    ),
    CategoryGroup(
        group_id="dietary",
        label="Dietary Requirement",
        parent_group_id="dining",
    ),
    CategoryGroup(
        group_id="generic",
        label="Generic Dataset Marker",
        parent_group_id="dining",
    ),
    CategoryGroup(
        group_id="excluded",
        label="Non-dining Companion Category",
        parent_group_id="dining",
    ),
)


# 每个允许大模型选择的真实类别只出现一次。没有列在这里的319表内类别会被
# 明确标为“非餐饮伴随类别”，不会因为新奇或低频就自动进入模型选择范围。
CATEGORY_GROUP_MEMBERS: dict[str, tuple[str, ...]] = {
    "north_american_cuisine": (
        "American (New)",
        "American (Traditional)",
        "Cajun/Creole",
        "Comfort Food",
        "Hawaiian",
        "Soul Food",
        "Southern",
    ),
    "east_asian_cuisine": (
        "Asian Fusion",
        "Cantonese",
        "Chinese",
        "Conveyor Belt Sushi",
        "Dim Sum",
        "Hainan",
        "Hong Kong Style Cafe",
        "Hot Pot",
        "Izakaya",
        "Japanese",
        "Japanese Curry",
        "Korean",
        "Mongolian",
        "Pan Asian",
        "Ramen",
        "Shanghainese",
        "Sushi Bars",
        "Szechuan",
        "Taiwanese",
        "Teppanyaki",
    ),
    "southeast_asian_cuisine": (
        "Burmese",
        "Cambodian",
        "Filipino",
        "Indonesian",
        "Laotian",
        "Malaysian",
        "Singaporean",
        "Thai",
        "Vietnamese",
    ),
    "south_asian_cuisine": (
        "Afghan",
        "Bangladeshi",
        "Himalayan/Nepalese",
        "Indian",
        "Pakistani",
    ),
    "middle_eastern_cuisine": (
        "Arabic",
        "Falafel",
        "Greek",
        "Israeli",
        "Kebab",
        "Lebanese",
        "Mediterranean",
        "Middle Eastern",
        "Moroccan",
        "Persian/Iranian",
        "Turkish",
    ),
    "european_cuisine": (
        "Armenian",
        "Basque",
        "Belgian",
        "British",
        "French",
        "Georgian",
        "German",
        "Hungarian",
        "Iberian",
        "Irish",
        "Italian",
        "Modern European",
        "Polish",
        "Portuguese",
        "Russian",
        "Sardinian",
        "Scandinavian",
        "Sicilian",
        "Spanish",
        "Ukrainian",
    ),
    "latin_caribbean_cuisine": (
        "Argentine",
        "Brazilian",
        "Caribbean",
        "Colombian",
        "Cuban",
        "Dominican",
        "Empanadas",
        "Honduran",
        "Latin American",
        "Mexican",
        "New Mexican Cuisine",
        "Peruvian",
        "Puerto Rican",
        "Salvadoran",
        "Tacos",
        "Tex-Mex",
        "Trinidadian",
        "Venezuelan",
    ),
    "african_cuisine": (
        "African",
        "Ethiopian",
        "Senegalese",
    ),
    "other_cuisine": (
        "Australian",
        "Uzbek",
    ),
    "handheld_dish": (
        "Bagels",
        "Burgers",
        "Cheesesteaks",
        "Chicken Shop",
        "Chicken Wings",
        "Delicatessen",
        "Delis",
        "Hot Dogs",
        "Pizza",
        "Poutineries",
        "Sandwiches",
        "Wraps",
    ),
    "noodle_rice_dish": (
        "Noodles",
        "Pasta Shops",
    ),
    "meat_grill_dish": (
        "Barbeque",
        "Smokehouse",
        "Steakhouses",
    ),
    "seafood_dish": (
        "Fish & Chips",
        "Poke",
        "Seafood",
    ),
    "light_dish": (
        "Acai Bowls",
        "Salad",
        "Soup",
        "Tapas/Small Plates",
    ),
    "dessert_bakery": (
        "Bakeries",
        "Candy Stores",
        "Chocolatiers & Shops",
        "Creperies",
        "Cupcakes",
        "Custom Cakes",
        "Desserts",
        "Donuts",
        "Gelato",
        "Ice Cream & Frozen Yogurt",
        "Patisserie/Cake Shop",
        "Pretzels",
        "Shaved Ice",
        "Waffles",
    ),
    "beverage": (
        "Bubble Tea",
        "Coffee & Tea",
        "Coffee Roasteries",
        "Juice Bars & Smoothies",
        "Tea Rooms",
    ),
    "venue": (
        "Breakfast & Brunch",
        "Brasseries",
        "Buffets",
        "Cafes",
        "Cafeteria",
        "Diners",
        "Dinner Theater",
        "Eatertainment",
        "Fast Food",
        "Food Court",
        "Food Stands",
        "Food Trucks",
        "Gastropubs",
        "Internet Cafes",
        "Pop-Up Restaurants",
        "Street Vendors",
        "Themed Cafes",
    ),
    "bar_venue": (
        "Bars",
        "Beer Bar",
        "Beer Gardens",
        "Breweries",
        "Brewpubs",
        "Cideries",
        "Cocktail Bars",
        "Dive Bars",
        "Gay Bars",
        "Hookah Bars",
        "Irish Pub",
        "Lounges",
        "Pubs",
        "Sports Bars",
        "Tapas Bars",
        "Whiskey Bars",
        "Wine Bars",
        "Wineries",
    ),
    "dietary": (
        "Gluten-Free",
        "Halal",
        "Kosher",
        "Live/Raw Food",
        "Vegan",
        "Vegetarian",
    ),
}

GENERIC_CATEGORIES = frozenset(
    {
        "Ethnic Food",
        "Food",
        "Local Flavor",
        "Nightlife",
        "Restaurants",
        "Specialty Food",
    }
)

# 更细类别尽量挂到真实上级类别，使 Chinese -> Szechuan 和
# Japanese -> Sushi Bars 这类关系可以直接用于后续数据库条件扩展。
PARENT_CATEGORY_OVERRIDES: dict[str, str] = {
    "Beer Bar": "Bars",
    "Cocktail Bars": "Bars",
    "Dive Bars": "Bars",
    "Gay Bars": "Bars",
    "Hookah Bars": "Bars",
    "Lounges": "Bars",
    "Pubs": "Bars",
    "Sports Bars": "Bars",
    "Whiskey Bars": "Bars",
    "Wine Bars": "Bars",
    "Cantonese": "Chinese",
    "Dim Sum": "Chinese",
    "Hainan": "Chinese",
    "Hong Kong Style Cafe": "Chinese",
    "Hot Pot": "Chinese",
    "Shanghainese": "Chinese",
    "Szechuan": "Chinese",
    "Taiwanese": "Chinese",
    "Conveyor Belt Sushi": "Japanese",
    "Izakaya": "Japanese",
    "Japanese Curry": "Japanese",
    "Ramen": "Japanese",
    "Sushi Bars": "Japanese",
    "Teppanyaki": "Japanese",
    "New Mexican Cuisine": "Mexican",
    "Tacos": "Mexican",
    "Tex-Mex": "Mexican",
    "Empanadas": "Latin American",
    "Pasta Shops": "Italian",
    "Sicilian": "Italian",
    "Tapas Bars": "Spanish",
    "Tapas/Small Plates": "Spanish",
    "Irish Pub": "Irish",
    "Falafel": "Middle Eastern",
    "Kebab": "Middle Eastern",
    "Poke": "Hawaiian",
}


@dataclass(frozen=True, slots=True)
class CategoryClassification:
    selectable: bool
    category_kind: CategoryKind
    parent: str
    parent_is_category: bool


def _kind_for_group(group_id: str) -> CategoryKind:
    if group_id.endswith("cuisine"):
        return "cuisine"
    if group_id in {
        "handheld_dish",
        "noodle_rice_dish",
        "meat_grill_dish",
        "seafood_dish",
        "light_dish",
        "dessert_bakery",
    }:
        return "dish"
    if group_id == "dietary":
        return "dietary"
    return "venue"


def _member_to_group() -> dict[str, str]:
    result: dict[str, str] = {}
    for group_id, categories in CATEGORY_GROUP_MEMBERS.items():
        for category in categories:
            if category in result:
                raise ValueError(f"category appears in multiple groups: {category}")
            result[category] = group_id
    return result


_MEMBER_TO_GROUP = _member_to_group()


def classify_category(category: str) -> CategoryClassification:
    """给一条真实类别返回固定结论；未列入白名单的一律不提供给模型。"""

    if category in GENERIC_CATEGORIES:
        return CategoryClassification(
            selectable=False,
            category_kind="generic",
            parent="generic",
            parent_is_category=False,
        )
    group_id = _MEMBER_TO_GROUP.get(category)
    if group_id is None:
        return CategoryClassification(
            selectable=False,
            category_kind="excluded",
            parent="excluded",
            parent_is_category=False,
        )
    parent_category = PARENT_CATEGORY_OVERRIDES.get(category)
    return CategoryClassification(
        selectable=True,
        category_kind=_kind_for_group(group_id),
        parent=parent_category or group_id,
        parent_is_category=parent_category is not None,
    )
