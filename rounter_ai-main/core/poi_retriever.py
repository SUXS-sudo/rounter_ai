"""Candidate POI retrieval for the local route planner."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from utils.geo import haversine_distance

from core.poi_artifact_store import load_poi_store
from core.preference import get_must_satisfy_preferences, get_strong_preferences, matches_preference
from core.zone_catalog import get_zone_aliases, get_zone_metadata
from models.config import settings
from models.schemas import POI

ZONE_METADATA = get_zone_metadata()


DEFAULT_CITY = "成都"

# 商圈名称别名映射：用户可能用简称提到商圈
ZONE_ALIASES = get_zone_aliases()

PREFERENCE_KEYWORDS = {
    "火锅": ("火锅", "hotpot", "skewer_hotpot"),
    "小吃": ("小吃", "snack", "street_food", "food_street", "chengdu_snack"),
    "咖啡": ("咖啡", "coffee", "cafe"),
    "奶茶": ("奶茶", "茶饮", "果茶", "milk_tea", "bubble_tea", "teahouse", "dessert", "甜品饮品"),
    "书店": ("书店", "bookstore", "阅读空间", "文创书店", "图书馆", "library"),
    "拍照": ("拍照", "出片", "打卡", "地标", "建筑", "landmark"),
    "夜景": ("夜景", "夜游", "夜生活", "night"),
    "室内": ("室内", "雨天友好", "商场", "影院", "书店", "indoor", "mall", "cinema", "bookstore"),
    "安静": ("安静", "清静", "quiet", "适合坐坐", "适合聊天"),
    "文化": ("文化", "文艺", "博物馆", "画廊", "剧院", "展览", "museum", "gallery", "theater", "heritage"),
    "少排队": ("少排队", "不排队", "不用排队"),
}

_load_pois_cache: dict[str, list[POI]] = {}
_scope_cache: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = {}
_nearby_zones_cache: dict[tuple[str, str], set[str]] = {}


def load_pois(path: Path = settings.pois_file) -> list[POI]:
    """Load POIs from configured artifacts and validate them with the local schema."""

    cache_key = str(path)
    if cache_key in _load_pois_cache:
        return _load_pois_cache[cache_key]

    store = load_poi_store(path)
    items = [item for plist in store.load_all_by_city().values() for item in plist]

    result = [POI.model_validate(item) for item in items]
    _load_pois_cache[cache_key] = result
    return result


def find_by_tags(tags: list[str]) -> list[POI]:
    """Find POIs that contain any of the requested tags."""

    wanted = set(tags)
    return [poi for poi in load_pois() if wanted.intersection(poi.tags)]


def retrieve_candidate_pois(
    intent: dict[str, Any],
    user_profile: dict[str, Any] | Any | None,
    pois: list[dict[str, Any] | Any],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Retrieve relevant, diverse POI candidates for an intent.

    Args:
        intent: Parsed user intent. Expected keys include ``city``, ``budget``
            and ``preferences``.
        user_profile: Optional user profile as a dict or Pydantic model.
        pois: POI records as dicts or Pydantic models.
        limit: Maximum number of candidates to return.

    Returns:
        Up to ``limit`` POI dictionaries, filtered by city (and optionally by
        zone), ranked by preference match, budget fit, quality and user profile
        affinity, then diversified by category.
    """

    if limit <= 0:
        return []

    intent_data = _to_dict(intent)
    profile_data = _to_dict(user_profile) if user_profile is not None else {}
    city = str(intent_data.get("city") or DEFAULT_CITY)
    budget = _resolve_budget(intent_data, profile_data)
    preferences = [str(item) for item in intent_data.get("preferences", [])]
    required_preferences = get_must_satisfy_preferences(intent_data) or get_strong_preferences(intent_data)

    zone_filter = _resolve_zone_filter(intent_data)
    target_district = _resolve_zone_district(city, zone_filter)
    nearby_zones = _resolve_nearby_zones(city, zone_filter)
    scope_data = _get_scope_data(pois, city)

    city_pois = scope_data["city"]
    zone_pois = scope_data["zone"].get(zone_filter, []) if zone_filter else []
    nearby_zone_pois = _merge_unique_pois(zone_pois, *(scope_data["zone"].get(zone_name, []) for zone_name in nearby_zones)) if nearby_zones else zone_pois
    district_pois = scope_data["district"].get(target_district, []) if target_district else []

    candidate_scopes: list[list[dict[str, Any]]] = []
    if zone_filter and len(zone_pois) >= 5:
        candidate_scopes.append(zone_pois)
    if nearby_zone_pois:
        candidate_scopes.append(nearby_zone_pois)
    if district_pois:
        candidate_scopes.append(district_pois)
    candidate_scopes.append(city_pois)

    city_matched_pois = city_pois
    for scope in candidate_scopes:
        if not scope:
            continue
        if not required_preferences or all(any(matches_preference(poi, preference) for poi in scope) for preference in required_preferences):
            city_matched_pois = scope
            break

    coarse_pool = _coarse_filter_candidates(city_matched_pois, preferences, profile_data, budget, limit)
    scored_candidates: list[tuple[float, dict[str, Any]]] = []
    for poi in coarse_pool:
        preference_score = _preference_match_score(poi, preferences)
        if preferences and preference_score <= 0 and not _profile_matches_poi(poi, profile_data):
            continue

        score = _candidate_score(poi, preferences, budget, profile_data)
        scored_candidates.append((score, poi))

    if not scored_candidates and preferences:
        fallback_pool = coarse_pool or city_matched_pois[: max(limit * 4, 80)]
        scored_candidates = [
            (_candidate_score(poi, [], budget, profile_data), poi)
            for poi in fallback_pool
        ]

    ranked = [poi for _, poi in sorted(scored_candidates, key=lambda item: item[0], reverse=True)]
    ranked = _ensure_required_preference_candidates(ranked, city_matched_pois, required_preferences)
    ranked = ranked[: max(limit * 6, 120)]
    diversified = _diversify_by_category(ranked, limit)
    return _ensure_constraint_quotas(diversified, ranked, intent_data, limit)


def _get_scope_data(pois: list[dict[str, Any] | Any], city: str) -> dict[str, Any]:
    cache_key = (id(pois), city)
    cached = _scope_cache.get(cache_key)
    if cached is not None:
        return cached

    city_pois: list[dict[str, Any]] = []
    by_zone: dict[str, list[dict[str, Any]]] = {}
    by_district: dict[str, list[dict[str, Any]]] = {}

    for poi in pois:
        prepared = _prepare_poi(poi, city)
        poi_city = str(prepared.get("city") or city)
        if city and poi_city != city:
            continue
        city_pois.append(prepared)
        zone = str(prepared.get("zone") or "")
        district = str(prepared.get("district") or "")
        if zone:
            by_zone.setdefault(zone, []).append(prepared)
        if district:
            by_district.setdefault(district, []).append(prepared)

    result = {"city": city_pois, "zone": by_zone, "district": by_district}
    _scope_cache[cache_key] = result
    return result



def _prepare_poi(poi: dict[str, Any] | Any, city: str) -> dict[str, Any]:
    poi_data = _to_dict(poi)
    if poi_data.get("_prepared_for_retrieval"):
        return poi_data

    tags = [str(tag) for tag in _as_list(poi_data.get("tags"))]
    search_text = " ".join(
        str(value)
        for value in (
            poi_data.get("name", ""),
            poi_data.get("category", ""),
            poi_data.get("sub_category", ""),
            poi_data.get("address", ""),
            " ".join(tags),
        )
    )
    features = _to_dict(poi_data.get("features", {}))
    poi_data["city"] = str(poi_data.get("city") or city)
    poi_data["_tags_set"] = set(tags)
    poi_data["_search_text_lower"] = search_text.lower()
    poi_data["_features_dict"] = features
    poi_data["_prepared_for_retrieval"] = True
    return poi_data



def _merge_unique_pois(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for group in groups:
        for poi in group:
            poi_id = str(poi.get("id") or id(poi))
            if poi_id in seen_ids:
                continue
            seen_ids.add(poi_id)
            merged.append(poi)
    return merged



def _coarse_filter_candidates(
    pois: list[dict[str, Any]],
    preferences: list[str],
    user_profile: dict[str, Any],
    budget: float,
    limit: int,
) -> list[dict[str, Any]]:
    if len(pois) <= max(limit * 6, 180):
        return pois

    preferred_tags = {str(tag) for tag in _as_list(user_profile.get("preferred_tags"))}
    favorite_categories = {str(item) for item in _as_list(user_profile.get("favorite_categories"))}
    budget_cap = budget * 2.4 if budget > 0 else 0
    target_size = max(limit * 8, 240)
    coarse_scored: list[tuple[float, dict[str, Any]]] = []

    for poi in pois:
        features = poi.get("_features_dict", {})
        price = _num(poi.get("price"))
        if budget_cap and price > budget_cap:
            continue

        score = _clamp(_num(poi.get("rating")) / 5) * 0.35
        score += _preference_hint_score(poi, preferences) * 0.45
        score += (1 - _feature(poi, "queue_risk", 0.5)) * 0.08

        tags = poi.get("_tags_set", set())
        if preferred_tags and tags.intersection(preferred_tags):
            score += 0.08
        if favorite_categories and str(poi.get("category") or "") in favorite_categories:
            score += 0.06
        if "室内" in preferences:
            score += _feature_from_dict(features, "indoor", 0.5) * 0.16

        coarse_scored.append((score, poi))

    if not coarse_scored:
        return pois[:target_size]

    coarse_scored.sort(key=lambda item: item[0], reverse=True)
    return [poi for _, poi in coarse_scored[:target_size]]



def _preference_hint_score(poi: dict[str, Any], preferences: list[str]) -> float:
    if not preferences:
        return 0.5
    text = poi.get("_search_text_lower", "")
    category = str(poi.get("category") or "")
    features = poi.get("_features_dict", {})

    score = 0.0
    for preference in preferences:
        keywords = tuple(keyword.lower() for keyword in PREFERENCE_KEYWORDS.get(preference, (preference,)))
        if any(keyword in text for keyword in keywords):
            score += 1.0
            continue
        if preference == "拍照":
            score += _feature_from_dict(features, "photo", 0.5)
        elif preference == "夜景":
            score += _feature_from_dict(features, "night_view", 0.5)
        elif preference == "室内":
            score += _feature_from_dict(features, "indoor", 0.5)
        elif preference == "安静":
            score += _feature_from_dict(features, "quiet", 0.5)
        elif preference == "少排队":
            score += 1 - _feature_from_dict(features, "queue_risk", 0.5)
        elif preference == "咖啡" and category == "cafe":
            score += 0.85
        elif preference == "文化" and category == "culture":
            score += 0.9
        elif preference == "少走路":
            score += 0.65
    return _clamp(score / len(preferences))



def _candidate_score(
    poi: dict[str, Any],
    preferences: list[str],
    budget: int | float,
    user_profile: dict[str, Any],
) -> float:
    quality_score = _clamp(_num(poi.get("rating")) / 5)
    preference_score = _preference_match_score(poi, preferences) if preferences else 0.55
    budget_score = _budget_fit_score(_num(poi.get("price")), budget)
    queue_score = 1 - _feature(poi, "queue_risk", 0.5)
    personalization_score = _profile_affinity_score(poi, user_profile)

    weights = {
        "quality": 0.25,
        "preference": 0.38,
        "budget": 0.16,
        "queue": 0.08,
        "personalization": 0.13,
    }

    if budget <= 200:
        weights["budget"] += 0.12
        weights["preference"] -= 0.06
        weights["quality"] -= 0.03
        weights["personalization"] -= 0.03
    elif budget <= 300:
        weights["budget"] += 0.06
        weights["preference"] -= 0.03
        weights["quality"] -= 0.03

    if "少排队" in preferences:
        weights["queue"] += 0.12
        weights["preference"] += 0.04
        weights["quality"] -= 0.06
        weights["personalization"] -= 0.10

    if "少走路" in preferences:
        weights["preference"] += 0.14
        weights["queue"] += 0.04
        weights["quality"] -= 0.06
        weights["personalization"] -= 0.08

    if "室内" in preferences:
        weights["preference"] += 0.10
        weights["queue"] += 0.03
        weights["quality"] -= 0.05
        weights["personalization"] -= 0.08

    weights = _normalize_weights(weights)
    return (
        quality_score * weights["quality"]
        + preference_score * weights["preference"]
        + budget_score * weights["budget"]
        + queue_score * weights["queue"]
        + personalization_score * weights["personalization"]
    )


def _preference_match_score(poi: dict[str, Any], preferences: list[str]) -> float:
    if not preferences:
        return 0.5

    scores = [_single_preference_score(poi, preference) for preference in preferences]
    return _clamp(sum(scores) / len(scores))


def _single_preference_score(poi: dict[str, Any], preference: str) -> float:
    feature_scores = {
        "火锅": max(_keyword_score(poi, "火锅"), _keyword_score(poi, "hotpot"), _feature(poi, "taste", 0.5) * 0.8),
        "小吃": max(_keyword_score(poi, "小吃"), _keyword_score(poi, "snack"), _feature(poi, "taste", 0.5) * 0.75),
        "咖啡": max(_keyword_score(poi, "咖啡"), _keyword_score(poi, "coffee"), 0.85 if poi.get("category") == "cafe" else 0),
        "奶茶": max(_keyword_score(poi, "奶茶"), _keyword_score(poi, "茶饮"), _keyword_score(poi, "milk_tea"), _keyword_score(poi, "bubble_tea"), _keyword_score(poi, "甜品饮品"), _keyword_score(poi, "果茶"), _feature(poi, "cost_performance", 0.5) * 0.7),
        "书店": max(_keyword_score(poi, "书店"), _keyword_score(poi, "bookstore"), _keyword_score(poi, "图书馆"), _feature(poi, "quiet", 0.5) * 0.8),
        "拍照": max(_feature(poi, "photo", 0.5), _keyword_score(poi, "拍照"), _keyword_score(poi, "出片")),
        "夜景": max(_feature(poi, "night_view", 0.5), _keyword_score(poi, "夜景"), _keyword_score(poi, "夜游")),
        "室内": max(_feature(poi, "indoor", 0.5), _keyword_score(poi, "室内"), _keyword_score(poi, "雨天友好")),
        "安静": max(_feature(poi, "quiet", 0.5), _keyword_score(poi, "安静"), _keyword_score(poi, "quiet")),
        "文化": max(0.9 if poi.get("category") == "culture" else 0, _keyword_score(poi, "文化"), _keyword_score(poi, "文艺"), _keyword_score(poi, "博物馆"), _keyword_score(poi, "画廊")),
        "少排队": 1 - _feature(poi, "queue_risk", 0.5),
        "少走路": 0.65,
    }

    if preference in feature_scores:
        return _clamp(feature_scores[preference])

    keywords = PREFERENCE_KEYWORDS.get(preference, (preference,))
    return max((_keyword_score(poi, keyword) for keyword in keywords), default=0.0)


def _ensure_required_preference_candidates(
    ranked: list[dict[str, Any]],
    all_scope_pois: list[dict[str, Any]],
    required_preferences: list[str],
) -> list[dict[str, Any]]:
    if not required_preferences:
        return ranked

    selected = list(ranked)
    selected_ids = {str(poi.get('id') or id(poi)) for poi in selected}
    for preference in required_preferences:
        if any(matches_preference(poi, preference) for poi in selected[:20]):
            continue
        fallback = next((poi for poi in all_scope_pois if matches_preference(poi, preference)), None)
        if fallback is None:
            continue
        fallback_id = str(fallback.get('id') or id(fallback))
        if fallback_id not in selected_ids:
            selected.insert(0, fallback)
            selected_ids.add(fallback_id)
    return selected



def _profile_affinity_score(poi: dict[str, Any], user_profile: dict[str, Any]) -> float:
    if not user_profile:
        return 0.5

    score = 0.5
    tags = set(_as_list(poi.get("tags")))
    preferred_tags = set(_as_list(user_profile.get("preferred_tags")))
    disliked_tags = set(_as_list(user_profile.get("disliked_tags")))
    favorite_categories = set(_as_list(user_profile.get("favorite_categories")))

    score += min(0.24, 0.06 * len(tags.intersection(preferred_tags)))
    score -= min(0.30, 0.10 * len(tags.intersection(disliked_tags)))

    if poi.get("category") in favorite_categories:
        score += 0.12

    feature_weights = _to_dict(user_profile.get("feature_weights", {}))
    if feature_weights:
        weighted_total = 0.0
        weight_sum = 0.0
        for feature_name, weight in feature_weights.items():
            numeric_weight = _num(weight)
            weighted_total += _feature(poi, feature_name, 0.5) * numeric_weight
            weight_sum += numeric_weight
        if weight_sum > 0:
            score = score * 0.65 + (weighted_total / weight_sum) * 0.35

    return _clamp(score)


def _profile_matches_poi(poi: dict[str, Any], user_profile: dict[str, Any]) -> bool:
    if not user_profile:
        return False

    tags = set(_as_list(poi.get("tags")))
    preferred_tags = set(_as_list(user_profile.get("preferred_tags")))
    favorite_categories = set(_as_list(user_profile.get("favorite_categories")))
    return bool(tags.intersection(preferred_tags) or poi.get("category") in favorite_categories)


def _diversify_by_category(ranked_pois: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not ranked_pois:
        return []

    categories = {str(poi.get("category") or "unknown") for poi in ranked_pois}
    soft_cap = max(2, limit // max(1, min(4, len(categories))))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    category_counts: Counter[str] = Counter()

    for poi in ranked_pois:
        category = str(poi.get("category") or "unknown")
        poi_id = str(poi.get("id") or id(poi))
        if category_counts[category] >= soft_cap:
            continue
        selected.append(poi)
        selected_ids.add(poi_id)
        category_counts[category] += 1
        if len(selected) >= limit:
            return selected

    for poi in ranked_pois:
        poi_id = str(poi.get("id") or id(poi))
        if poi_id in selected_ids:
            continue
        selected.append(poi)
        selected_ids.add(poi_id)
        if len(selected) >= limit:
            break

    return selected


FOOD_CATEGORIES = {"food", "cafe"}
TEA_SUBCATEGORIES = {"milk_tea", "bubble_tea", "teahouse", "dessert", "cafe"}
TEA_KEYWORDS = ("奶茶", "茶饮", "果茶", "甜品饮品", "下午茶", "milk_tea", "bubble_tea", "teahouse", "dessert")


def _ensure_constraint_quotas(
    diversified: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    intent: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    selected = list(diversified[:limit])
    strong_preferences = get_strong_preferences(intent)
    quotas: list[tuple[callable, int]] = [(_is_food_or_drink_poi, max(2, limit // 8))]

    if any(preference == "奶茶" for preference in strong_preferences):
        quotas.append((_is_tea_poi, max(2, limit // 10)))
    if "室内" in strong_preferences:
        quotas.append((lambda poi: matches_preference(poi, "室内"), max(6, limit // 4)))
        quotas.append((_is_food_or_drink_poi, max(4, limit // 6)))

    for preference in strong_preferences:
        if preference in {"火锅", "小吃", "咖啡", "奶茶"}:
            quotas.append((lambda poi, pref=preference: matches_preference(poi, pref), 2))
        elif preference == "夜景":
            quotas.append((lambda poi: matches_preference(poi, "夜景"), max(3, limit // 10)))

    return _apply_quota_replacements(selected, ranked, quotas, limit)


def _is_tea_poi(poi: dict[str, Any]) -> bool:
    sub_category = str(poi.get("sub_category") or "")
    if sub_category in TEA_SUBCATEGORIES or any(token in sub_category for token in TEA_SUBCATEGORIES):
        return True
    return any(keyword.lower() in _poi_text(poi).lower() for keyword in TEA_KEYWORDS)


def _is_food_or_drink_poi(poi: dict[str, Any]) -> bool:
    return str(poi.get("category")) in FOOD_CATEGORIES or _is_tea_poi(poi)


def _apply_quota_replacements(
    selected: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    quotas: list[tuple[callable, int]],
    limit: int,
) -> list[dict[str, Any]]:
    selected_ids = {str(poi.get("id")) for poi in selected}

    def count_matches(predicate: callable) -> int:
        return sum(1 for poi in selected if predicate(poi))

    protected_predicates = [predicate for predicate, minimum in quotas if minimum > 0]

    for predicate, minimum in quotas:
        missing = minimum - count_matches(predicate)
        if missing <= 0:
            continue
        for poi in ranked:
            if missing <= 0:
                break
            poi_id = str(poi.get("id"))
            if poi_id in selected_ids or not predicate(poi):
                continue

            replace_index = next(
                (
                    index for index in range(len(selected) - 1, -1, -1)
                    if not any(protected(selected[index]) for protected in protected_predicates)
                ),
                None,
            )

            if replace_index is None:
                if len(selected) < limit:
                    selected.append(poi)
                else:
                    break
            else:
                removed_id = str(selected[replace_index].get("id"))
                selected_ids.discard(removed_id)
                selected[replace_index] = poi

            selected_ids.add(poi_id)
            missing -= 1

    return selected[:limit]


def _resolve_nearby_zones(city: str, zone_name: str) -> set[str]:
    if not zone_name:
        return set()
    cache_key = (city, zone_name)
    if cache_key in _nearby_zones_cache:
        return _nearby_zones_cache[cache_key]
    city_zones = ZONE_METADATA.get(city, {})
    current = city_zones.get(zone_name)
    if not current:
        return set()
    current_lng, current_lat = current["center"]
    distances: list[tuple[float, str]] = []
    for other_zone, metadata in city_zones.items():
        if other_zone == zone_name:
            continue
        lng, lat = metadata["center"]
        distances.append((haversine_distance(current_lat, current_lng, lat, lng), other_zone))
    distances.sort(key=lambda item: item[0])
    result = {zone for _, zone in distances[:3]}
    _nearby_zones_cache[cache_key] = result
    return result


def _resolve_zone_district(city: str, zone_name: str) -> str:
    if not zone_name:
        return ""
    return str(ZONE_METADATA.get(city, {}).get(zone_name, {}).get("district") or "")


def _resolve_budget(intent: dict[str, Any], user_profile: dict[str, Any]) -> float:
    if intent.get("budget") is not None:
        return max(0, _num(intent["budget"]))
    if user_profile.get("budget_per_day") is not None:
        return max(0, _num(user_profile["budget_per_day"]))
    return 300.0


def _resolve_zone_filter(intent: dict[str, Any]) -> str:
    """从用户意图中提取商圈过滤关键词。

    优先使用意图中已解析的 ``zone`` 字段；若为空则回退到从
    ``start_location`` 和 ``query`` 中匹配已知商圈名称。
    """

    # 优先使用 intent_parser 已解析的 zone
    zone = str(intent.get("zone") or "").strip()
    if zone:
        return zone

    start_location = str(intent.get("start_location") or "")
    query = str(intent.get("query") or "")
    text = f"{start_location} {query}"

    for alias, full_name in ZONE_ALIASES.items():
        if alias in text:
            return full_name

    return ""


def _budget_fit_score(price: float, budget: float) -> float:
    if budget <= 0:
        return 0.5
    if price <= budget:
        return _clamp(1 - (price / budget) * 0.35)
    return _clamp((budget / max(price, 1)) * 0.6)


def _keyword_score(poi: dict[str, Any], keyword: str) -> float:
    text = poi.get("_search_text_lower") or _poi_text(poi).lower()
    keyword_lower = keyword.lower()
    if keyword_lower in text:
        return 1.0
    for mapped_keyword in PREFERENCE_KEYWORDS.get(keyword, ()):
        if mapped_keyword.lower() in text:
            return 1.0
    return 0.0


def _poi_text(poi: dict[str, Any]) -> str:
    cached_text = poi.get("_search_text_lower")
    if cached_text:
        return cached_text
    tags = " ".join(str(tag) for tag in _as_list(poi.get("tags")))
    return " ".join(
        str(value)
        for value in (
            poi.get("name", ""),
            poi.get("category", ""),
            poi.get("sub_category", ""),
            poi.get("address", ""),
            tags,
        )
    )


def _feature_from_dict(features: dict[str, Any], name: str, default: float = 0.0) -> float:
    return _clamp(_num(features.get(name, default), default))



def _feature(poi: dict[str, Any], name: str, default: float = 0.0) -> float:
    features = poi.get("_features_dict")
    if not isinstance(features, dict):
        features = _to_dict(poi.get("features", {}))
    return _feature_from_dict(features, name, default)


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.02, value) for key, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {key: 1 / len(weights) for key in weights}
    return {key: value / total for key, value in cleaned.items()}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
