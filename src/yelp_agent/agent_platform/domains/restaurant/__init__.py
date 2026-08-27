"""把新版餐饮推荐能力接入通用 Agent。"""

from .tools import RestaurantToolSet, build_restaurant_tools

__all__ = ["RestaurantToolSet", "build_restaurant_tools"]
