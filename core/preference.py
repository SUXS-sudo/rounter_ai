"""Shared preference matching helpers."""

from __future__ import annotations

from typing import Any


STRONG_PREFERENCES = ["火锅", "小吃", "咖啡", "奶茶", "夜景", "室内"]
STRONG_FOOD_PREFERENCES = {"火锅", "小吃", "咖啡", "奶茶"}


SOFT_PREFERENCES = {"书店", "安静", "文化"}


def matches_preference(poi: dict[str, Any], preference: str) -> bool:
    """Return whether a POI satisfies a route-planning preference."""

    sub_category = str(poi.get("sub_category") or "").lower()
    category = str(poi.get("category") or "").lower()
    tags = {str(tag) for tag in _as_list(poi.get("tags"))}
    features = _to_dict(poi.get("features", {}))

    if preference == "火锅":
        return sub_category == "hotpot" or "hotpot" in sub_category or "火锅" in tags
    if preference == "小吃":
        return sub_category in {"snack", "street_food"} or "snack" in sub_category or "小吃" in tags
    if preference == "咖啡":
        return sub_category in {"coffee", "cafe"} or "coffee" in sub_category or category == "cafe" or "咖啡" in tags
    if preference == "奶茶":
        return (
            sub_category in {"milk_tea", "bubble_tea", "teahouse", "dessert"}
            or any(token in sub_category for token in ("milk_tea", "bubble_tea", "tea", "dessert"))
            or bool(tags.intersection({"奶茶", "茶饮", "果茶", "甜品饮品", "下午茶"}))
            or any(token in _poi_text(poi).lower() for token in ("奶茶", "茶饮", "果茶", "bubble_tea", "milk_tea"))
        )
    if preference == "书店":
        return sub_category in {"bookstore", "library"} or any(token in _poi_text(poi).lower() for token in ("书店", "bookstore", "阅读空间", "图书馆", "文创书店"))
    if preference in {"拍照", "出片"}:
        return _feature(features, "photo") >= 0.65 or bool(tags.intersection({"拍照", "出片"}))
    if preference == "夜景":
        return _feature(features, "night_view") >= 0.65 or "夜景" in tags
    if preference == "室内":
        return _feature(features, "indoor") >= 0.65 or "室内" in tags
    if preference == "安静":
        return _feature(features, "quiet") >= 0.65 or "安静" in tags
    if preference == "文化":
        return category == "culture" or sub_category in {"museum", "gallery", "theater", "heritage", "bookstore", "library"} or any(token in _poi_text(poi).lower() for token in ("文化", "文艺", "展览", "博物馆", "画廊", "剧院", "书店", "bookstore"))
    if preference == "少排队":
        return _feature(features, "queue_risk", 0.5) <= 0.45
    return str(preference) in _poi_text(poi)


def get_strong_preferences(intent: dict[str, Any]) -> list[str]:
    """Return strong preferences from an intent in stable order."""

    preferences = [str(item) for item in _as_list(intent.get("preferences"))]
    return [preference for preference in STRONG_PREFERENCES if preference in preferences]


def get_must_satisfy_preferences(intent: dict[str, Any]) -> list[str]:
    """Return dynamically promoted must-satisfy preferences from an intent."""

    return [str(item) for item in _as_list(intent.get("must_satisfy_preferences")) if str(item).strip()]


def get_covered_strong_preferences(route_pois: list[dict[str, Any]], intent: dict[str, Any]) -> list[str]:
    """Return strong preferences covered by at least one POI in the route."""

    covered: list[str] = []
    for preference in get_strong_preferences(intent):
        if any(matches_preference(poi, preference) for poi in route_pois):
            covered.append(preference)
    return covered


def candidate_has_preference(candidate_pois: list[dict[str, Any]], preference: str) -> bool:
    """Return whether the candidate pool contains at least one matching POI."""

    return any(matches_preference(poi, preference) for poi in candidate_pois)


def avoid_matches_poi(poi: dict[str, Any], avoid_items: Any) -> bool:
    """Return whether a POI violates an avoid constraint."""

    for item in _as_list(avoid_items):
        if matches_preference(poi, str(item)):
            return True
        if str(item) and str(item) in _poi_text(poi):
            return True
    return False


def route_contains_food(route_pois: list[dict[str, Any]]) -> bool:
    """Return whether a route contains any food or drink POI."""

    return any(
        str(poi.get("category") or "") in {"food", "cafe"}
        or matches_preference(poi, "火锅")
        or matches_preference(poi, "小吃")
        or matches_preference(poi, "咖啡")
        or matches_preference(poi, "奶茶")
        for poi in route_pois
    )


def preference_match_score(poi: dict[str, Any], preference: str) -> float:
    """Return a soft 0-1 score for a preference."""

    if matches_preference(poi, preference):
        return 1.0

    features = _to_dict(poi.get("features", {}))
    if preference in {"火锅", "小吃", "咖啡", "奶茶"}:
        return min(0.55, _feature(features, "taste") * 0.65)
    if preference == "书店":
        return max(_feature(features, "indoor"), _feature(features, "quiet"))
    if preference in {"拍照", "出片"}:
        return _feature(features, "photo")
    if preference == "夜景":
        return _feature(features, "night_view")
    if preference == "室内":
        return _feature(features, "indoor")
    if preference == "安静":
        return _feature(features, "quiet")
    if preference == "文化":
        return max(_feature(features, "quiet"), 0.8 if str(poi.get("category") or "") == "culture" else 0.0)
    if preference == "少排队":
        return 1 - _feature(features, "queue_risk", 0.5)
    if preference == "少走路":
        return 0.5
    return 0.0


def _feature(features: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(features.get(name, default))))
    except (TypeError, ValueError):
        return default


def _poi_text(poi: dict[str, Any]) -> str:
    return " ".join(
        str(value)
        for value in (
            poi.get("name", ""),
            poi.get("category", ""),
            poi.get("sub_category", ""),
            poi.get("address", ""),
            " ".join(str(tag) for tag in _as_list(poi.get("tags"))),
        )
    )


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]
