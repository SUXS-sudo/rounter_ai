"""POI and route scoring logic for the local route planner."""

from __future__ import annotations

from collections import Counter
from typing import Any

from core.preference import (
    STRONG_FOOD_PREFERENCES,
    avoid_matches_poi,
    get_covered_strong_preferences,
    get_strong_preferences,
    matches_preference,
    preference_match_score,
)
from utils.geo import estimate_travel_minutes, haversine_distance


FOOD_CATEGORIES = {"food", "cafe"}
DEFAULT_BUDGET = 300.0


def score_poi(
    poi: dict[str, Any] | Any,
    intent: dict[str, Any] | Any | None = None,
    user_profile: dict[str, Any] | Any | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a single POI against intent, profile and route context.

    Args:
        poi: POI as a dictionary or Pydantic model.
        intent: Parsed user intent containing ``preferences`` and ``budget``.
        user_profile: Optional user profile containing preference tags, disliked
            tags, favorite categories and feature weights.
        context: Optional route context, for example existing route categories or
            travel distance from the previous POI.

    Returns:
        A dictionary with ``score``, ``score_detail`` and ``reason_codes``. The
        original POI dictionary is included under ``poi`` for downstream use.
    """

    poi_data = _to_dict(poi)
    intent_data = _to_dict(intent)
    profile_data = _to_dict(user_profile)
    context_data = context or {}

    if not profile_data and _looks_like_user_profile(intent_data):
        profile_data = intent_data
        intent_data = {}

    preferences = [str(item) for item in intent_data.get("preferences", [])]
    avoid_items = [str(item) for item in _as_list(intent_data.get("avoid"))]
    hard_avoid_items = [item for item in avoid_items if item in STRONG_FOOD_PREFERENCES]
    budget = _resolve_budget(intent_data, profile_data)

    quality_score = _clamp(_num(poi_data.get("rating")) / 5)
    preference_score = _preference_score(poi_data, preferences, context_data)
    budget_score = _budget_score(_num(poi_data.get("price")), budget)
    queue_score = 1 - _feature(poi_data, "queue_risk", 0.5)
    personalization_score = _personalization_score(poi_data, profile_data)
    diversity_score = _diversity_score(poi_data, context_data)
    explicit_strong_matches = [
        preference
        for preference in get_strong_preferences(intent_data)
        if matches_preference(poi_data, preference)
    ]
    explicit_strong_bonus = min(0.22, 0.12 * len(explicit_strong_matches))
    avoid_penalty = 0.75 if avoid_matches_poi(poi_data, hard_avoid_items) else 0.0

    weights = _poi_weights(preferences, budget)
    final_score = (
        quality_score * weights["quality"]
        + preference_score * weights["preference"]
        + budget_score * weights["budget"]
        + queue_score * weights["queue"]
        + personalization_score * weights["personalization"]
        + diversity_score * weights["diversity"]
        + explicit_strong_bonus
        - avoid_penalty
    )

    score_detail = {
        "quality_score": round(quality_score, 4),
        "preference_score": round(preference_score, 4),
        "budget_score": round(budget_score, 4),
        "queue_score": round(queue_score, 4),
        "personalization_score": round(personalization_score, 4),
        "diversity_score": round(diversity_score, 4),
        "price": _num(poi_data.get("price")),
        "budget": budget,
        "queue_risk": _feature(poi_data, "queue_risk", 0.5),
        "explicit_strong_matches": explicit_strong_matches,
        "explicit_strong_bonus": round(explicit_strong_bonus, 4),
        "avoid_penalty": round(avoid_penalty, 4),
        "weights": {key: round(value, 4) for key, value in weights.items()},
    }

    return {
        "id": poi_data.get("id"),
        "name": poi_data.get("name"),
        "category": poi_data.get("category"),
        "score": round(_clamp(final_score), 4),
        "score_detail": score_detail,
        "reason_codes": _poi_reason_codes(poi_data, score_detail, preferences),
        "poi": poi_data,
    }


def score_route(
    route: list[dict[str, Any] | Any] | dict[str, Any],
    intent: dict[str, Any] | Any,
    user_profile: dict[str, Any] | Any | None,
) -> dict[str, Any]:
    """Score a whole route using POI quality, budget, travel and fit signals.

    Args:
        route: A list of POIs, a list of route stops containing ``poi``, or a
            route dictionary with a ``stops`` field.
        intent: Parsed user intent.
        user_profile: Optional user profile.

    Returns:
        A dictionary containing final route ``score``, ``score_detail`` and
        ``reason_codes``.
    """

    intent_data = _to_dict(intent)
    profile_data = _to_dict(user_profile)
    pois = _extract_route_pois(route)
    if not pois:
        return {
            "score": 0.0,
            "score_detail": {"poi_count": 0},
            "reason_codes": ["empty_route"],
        }

    preferences = [str(item) for item in intent_data.get("preferences", [])]
    budget = _resolve_budget(intent_data, profile_data)
    poi_scores = [score_poi(poi, intent_data, profile_data) for poi in pois]

    average_poi_score = sum(item["score"] for item in poi_scores) / len(poi_scores)
    estimated_cost = sum(_num(poi.get("price")) for poi in pois)
    budget_score = _route_budget_score(estimated_cost, budget)
    total_travel_minutes = _route_travel_minutes(route, pois, intent_data)
    compact_score = _compact_score(total_travel_minutes, "少走路" in preferences)
    average_queue_risk = sum(_feature(poi, "queue_risk", 0.5) for poi in pois) / len(pois)
    queue_score = 1 - average_queue_risk
    category_diversity_score = _category_diversity_score(pois)
    food_score = 1.0 if any(_is_food_poi(poi) for poi in pois) else 0.25
    preference_match_score = sum(_preference_score(poi, preferences, {}) for poi in pois) / len(pois)
    temporal_alignment_score = _temporal_alignment_score(route, pois)
    strong_preferences = get_strong_preferences(intent_data)
    covered_strong_preferences = get_covered_strong_preferences(pois, intent_data)
    preference_coverage_score = (
        len(covered_strong_preferences) / len(strong_preferences)
        if strong_preferences
        else 1.0
    )
    missing_food_preference_penalty = (
        0.25
        if any(preference in STRONG_FOOD_PREFERENCES for preference in strong_preferences)
        and not any(preference in covered_strong_preferences for preference in STRONG_FOOD_PREFERENCES)
        else 0.0
    )
    photo_score = sum(_feature(poi, "photo", 0.5) for poi in pois) / len(pois)
    indoor_score = sum(_feature(poi, "indoor", 0.5) for poi in pois) / len(pois)

    weights = _route_weights(preferences, budget)
    final_score = (
        average_poi_score * weights["poi_average"]
        + budget_score * weights["budget"]
        + compact_score * weights["compact"]
        + queue_score * weights["queue"]
        + category_diversity_score * weights["diversity"]
        + food_score * weights["food"]
        + preference_match_score * weights["preference"]
        + preference_coverage_score * weights["preference_coverage"]
        + photo_score * weights["photo"]
        + indoor_score * weights["indoor"]
        + temporal_alignment_score * weights["temporal"]
        - missing_food_preference_penalty
    )

    score_detail = {
        "poi_average_score": round(average_poi_score, 4),
        "budget_score": round(budget_score, 4),
        "compact_score": round(compact_score, 4),
        "queue_score": round(queue_score, 4),
        "category_diversity_score": round(category_diversity_score, 4),
        "food_score": round(food_score, 4),
        "preference_match_score": round(preference_match_score, 4),
        "preference_coverage_score": round(preference_coverage_score, 4),
        "covered_strong_preferences": covered_strong_preferences,
        "missing_food_preference_penalty": round(missing_food_preference_penalty, 4),
        "photo_score": round(photo_score, 4),
        "indoor_score": round(indoor_score, 4),
        "temporal_alignment_score": round(temporal_alignment_score, 4),
        "estimated_cost": round(estimated_cost, 2),
        "budget": budget,
        "budget_band_lower": max(0.0, budget - 50),
        "budget_band_upper": budget + 20,
        "budget_band_delta": round(_budget_band_delta(estimated_cost, budget), 2),
        "total_travel_minutes": total_travel_minutes,
        "average_queue_risk": round(average_queue_risk, 4),
        "category_count": len({poi.get("category") for poi in pois}),
        "poi_count": len(pois),
        "weights": {key: round(value, 4) for key, value in weights.items()},
        "poi_scores": poi_scores,
    }

    return {
        "score": round(_clamp(final_score), 4),
        "score_detail": score_detail,
        "reason_codes": _route_reason_codes(score_detail, preferences),
    }


def _poi_weights(preferences: list[str], budget: float) -> dict[str, float]:
    weights = {
        "quality": 0.22,
        "preference": 0.40,
        "budget": 0.16,
        "queue": 0.12,
        "personalization": 0.07,
        "diversity": 0.03,
    }

    if any(preference in STRONG_FOOD_PREFERENCES for preference in preferences):
        weights["preference"] += 0.12
        weights["personalization"] -= 0.03
        weights["quality"] -= 0.04
        weights["diversity"] -= 0.03
    if "少排队" in preferences:
        weights["queue"] += 0.18
        weights["quality"] -= 0.05
        weights["personalization"] -= 0.04
        weights["diversity"] -= 0.05
    if "拍照" in preferences:
        weights["preference"] += 0.10
        weights["quality"] -= 0.03
        weights["budget"] -= 0.03
        weights["diversity"] -= 0.02
    if "室内" in preferences:
        weights["preference"] += 0.10
        weights["quality"] -= 0.03
        weights["budget"] -= 0.03
    if budget <= 300:
        weights["budget"] += 0.10
        weights["quality"] -= 0.04
        weights["personalization"] -= 0.04
        weights["diversity"] -= 0.02

    return _normalize_weights(weights)


def _route_weights(preferences: list[str], budget: float) -> dict[str, float]:
    weights = {
        "poi_average": 0.28,
        "budget": 0.14,
        "compact": 0.12,
        "queue": 0.10,
        "diversity": 0.10,
        "food": 0.05,
        "preference": 0.14,
        "preference_coverage": 0.12,
        "photo": 0.04,
        "indoor": 0.03,
        "temporal": 0.08,
    }

    if any(preference in STRONG_FOOD_PREFERENCES for preference in preferences):
        weights["preference_coverage"] += 0.20
        weights["preference"] += 0.06
        weights["poi_average"] -= 0.08
        weights["diversity"] -= 0.05
        weights["food"] -= 0.03
    if "少排队" in preferences:
        weights["queue"] += 0.18
        weights["poi_average"] -= 0.05
        weights["diversity"] -= 0.04
        weights["food"] -= 0.03
        weights["photo"] -= 0.03
        weights["indoor"] -= 0.03
    if "拍照" in preferences:
        weights["photo"] += 0.14
        weights["preference"] += 0.04
        weights["budget"] -= 0.04
        weights["compact"] -= 0.04
        weights["food"] -= 0.03
        weights["diversity"] -= 0.03
    if budget <= 300:
        weights["budget"] += 0.14
        weights["poi_average"] -= 0.05
        weights["photo"] -= 0.03
        weights["diversity"] -= 0.03
        weights["food"] -= 0.03
    if "少走路" in preferences:
        weights["compact"] += 0.22
        weights["queue"] += 0.03
        weights["photo"] -= 0.05
        weights["food"] -= 0.03
        weights["diversity"] -= 0.06
        weights["poi_average"] -= 0.05
    if "室内" in preferences:
        weights["indoor"] += 0.15
        weights["photo"] -= 0.03
        weights["food"] -= 0.03
        weights["compact"] -= 0.03
        weights["diversity"] -= 0.03
        weights["poi_average"] -= 0.03

    return _normalize_weights(weights)


def _preference_score(
    poi: dict[str, Any],
    preferences: list[str],
    context: dict[str, Any],
) -> float:
    if not preferences:
        return 0.55

    scores = [_single_preference_score(poi, preference, context) for preference in preferences]
    return _clamp(sum(scores) / len(scores))


def _single_preference_score(
    poi: dict[str, Any],
    preference: str,
    context: dict[str, Any],
) -> float:
    if preference == "少走路":
        travel_minutes = _num(context.get("travel_minutes"), -1)
        distance_km = _num(context.get("distance_from_previous_km"), -1)
        if travel_minutes >= 0:
            return _clamp(1 - travel_minutes / 45)
        if distance_km >= 0:
            return _clamp(1 - distance_km / 3)
        return 0.65
    return preference_match_score(poi, preference)


def _budget_score(price: float, budget: float) -> float:
    if budget <= 0:
        return 0.5
    soft_target = max(20.0, budget * 0.35)
    lower_bound = max(0.0, soft_target - 20)
    upper_bound = soft_target + 15
    if lower_bound <= price <= upper_bound:
        return 1.0
    if price < lower_bound:
        return _clamp(0.85 - (lower_bound - price) / max(40.0, budget) * 0.5)
    return _clamp(0.88 - (price - upper_bound) / max(40.0, budget) * 0.8)


def _temporal_alignment_score(
    route: list[dict[str, Any] | Any] | dict[str, Any],
    pois: list[dict[str, Any]],
) -> float:
    stops = _extract_route_stops(route)
    if not stops:
        return 0.5

    scores: list[float] = []
    for stop, poi in zip(stops, pois):
        stop_data = _to_dict(stop)
        arrival_time = stop_data.get("arrival_time") or stop_data.get("arrive_time")
        if not arrival_time:
            scores.append(0.6)
            continue
        try:
            minutes = _parse_hhmm(arrival_time)
        except ValueError:
            scores.append(0.6)
            continue
        scores.append(_poi_temporal_score(poi, minutes))
    return _clamp(sum(scores) / len(scores)) if scores else 0.5


def _poi_temporal_score(poi: dict[str, Any], arrival_minutes: int) -> float:
    text = _poi_text(poi).lower()
    hour = arrival_minutes // 60

    if _is_food_poi(poi):
        if 11 <= hour <= 13 or 17 <= hour <= 20:
            return 1.0
        if 9 <= hour <= 15 or 16 <= hour <= 21:
            return 0.72
        return 0.38

    if any(keyword in text for keyword in ("park", "公园", "植物园", "动物园", "自然", "landmark", "古镇", "historic", "tower")):
        if 9 <= hour <= 17:
            return 1.0
        if 8 <= hour <= 18:
            return 0.72
        return 0.3

    if any(keyword in text for keyword in ("night_view", "夜景", "bar", "酒吧", "livehouse")):
        if 18 <= hour <= 23:
            return 1.0
        if 16 <= hour <= 23:
            return 0.7
        return 0.25

    if any(keyword in text for keyword in ("museum", "gallery", "bookstore", "library", "culture", "theater", "heritage", "exhibition")):
        if 10 <= hour <= 18:
            return 0.95
        if 9 <= hour <= 20:
            return 0.7
        return 0.4

    return 0.6


def _budget_band_delta(estimated_cost: float, budget: float) -> float:
    lower_bound = max(0.0, budget - 50)
    upper_bound = budget + 20
    if lower_bound <= estimated_cost <= upper_bound:
        return 0.0
    if estimated_cost < lower_bound:
        return estimated_cost - lower_bound
    return estimated_cost - upper_bound


def _route_budget_score(estimated_cost: float, budget: float) -> float:
    if budget <= 0:
        return 0.5
    lower_bound = max(0.0, budget - 50)
    upper_bound = budget + 20
    if lower_bound <= estimated_cost <= upper_bound:
        return 1.0
    if estimated_cost < lower_bound:
        gap = lower_bound - estimated_cost
        return _clamp(0.95 - gap / max(40.0, budget) * 0.9)
    gap = estimated_cost - upper_bound
    return _clamp(0.92 - gap / max(30.0, budget) * 1.2)


def _personalization_score(poi: dict[str, Any], user_profile: dict[str, Any]) -> float:
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
            score = score * 0.6 + (weighted_total / weight_sum) * 0.4

    profile_budget = _num(user_profile.get("budget_per_day"), 0)
    if profile_budget > 0 and _num(poi.get("price")) <= profile_budget:
        score += 0.04

    return _clamp(score)


def _diversity_score(poi: dict[str, Any], context: dict[str, Any]) -> float:
    route_categories = _as_list(context.get("route_categories"))
    if not route_categories:
        return 0.8
    category = poi.get("category")
    if category in route_categories:
        return 0.55
    return 0.9


def _category_diversity_score(pois: list[dict[str, Any]]) -> float:
    if not pois:
        return 0.0
    counts = Counter(str(poi.get("category") or "unknown") for poi in pois)
    unique_ratio = len(counts) / len(pois)
    dominance_penalty = max(counts.values()) / len(pois)
    return _clamp(unique_ratio * 0.75 + (1 - dominance_penalty) * 0.25)


def _compact_score(total_travel_minutes: int, prefer_compact: bool) -> float:
    if total_travel_minutes <= 0:
        return 1.0
    threshold = 50 if prefer_compact else 80
    return _clamp(1 - total_travel_minutes / (threshold * 2))


def _route_travel_minutes(
    route: list[dict[str, Any] | Any] | dict[str, Any],
    pois: list[dict[str, Any]],
    intent: dict[str, Any],
) -> int:
    stops = _extract_route_stops(route)
    explicit_minutes = 0
    has_explicit_minutes = False
    for stop in stops:
        if isinstance(stop, dict) and "travel_minutes_from_previous" in stop:
            explicit_minutes += max(0, int(_num(stop.get("travel_minutes_from_previous"))))
            has_explicit_minutes = True
    if has_explicit_minutes:
        return explicit_minutes

    mode = str(intent.get("travel_mode") or "walk")
    total = 0
    for previous, current in zip(pois, pois[1:]):
        if not _has_coordinates(previous) or not _has_coordinates(current):
            continue
        distance_km = haversine_distance(
            _num(previous.get("lat")),
            _num(previous.get("lng")),
            _num(current.get("lat")),
            _num(current.get("lng")),
        )
        total += estimate_travel_minutes(distance_km, mode)
    return total


def _extract_route_stops(route: list[dict[str, Any] | Any] | dict[str, Any]) -> list[Any]:
    if isinstance(route, dict):
        return list(route.get("stops") or route.get("pois") or [])
    return list(route)


def _extract_route_pois(route: list[dict[str, Any] | Any] | dict[str, Any]) -> list[dict[str, Any]]:
    pois: list[dict[str, Any]] = []
    for item in _extract_route_stops(route):
        item_data = _to_dict(item)
        if "poi" in item_data:
            pois.append(_to_dict(item_data["poi"]))
        else:
            pois.append(item_data)
    return pois


def _poi_reason_codes(
    poi: dict[str, Any],
    score_detail: dict[str, Any],
    preferences: list[str],
) -> list[str]:
    reason_codes: list[str] = []
    if score_detail["quality_score"] >= 0.88:
        reason_codes.append("high_quality")
    if score_detail["preference_score"] >= 0.7:
        reason_codes.append("matches_preferences")
    if score_detail["budget_score"] >= 0.75:
        reason_codes.append("budget_friendly")
    elif score_detail["price"] > score_detail["budget"]:
        reason_codes.append("over_budget")
    if score_detail["queue_score"] >= 0.65:
        reason_codes.append("low_queue_risk")
    elif score_detail["queue_score"] <= 0.35:
        reason_codes.append("high_queue_risk")
    if score_detail["personalization_score"] >= 0.65:
        reason_codes.append("profile_fit")
    if "拍照" in preferences and _feature(poi, "photo", 0.5) >= 0.75:
        reason_codes.append("photo_friendly")
    if "夜景" in preferences and _feature(poi, "night_view", 0.5) >= 0.75:
        reason_codes.append("night_view_friendly")
    if "室内" in preferences and _feature(poi, "indoor", 0.5) >= 0.75:
        reason_codes.append("indoor_friendly")
    return reason_codes or ["neutral_match"]


def _route_reason_codes(score_detail: dict[str, Any], preferences: list[str]) -> list[str]:
    reason_codes: list[str] = []
    if score_detail["poi_average_score"] >= 0.75:
        reason_codes.append("strong_poi_quality")
    if score_detail["budget_band_lower"] <= score_detail["estimated_cost"] <= score_detail["budget_band_upper"]:
        reason_codes.append("within_budget_band")
    elif score_detail["estimated_cost"] < score_detail["budget_band_lower"]:
        reason_codes.append("below_budget_band")
    else:
        reason_codes.append("above_budget_band")
    if score_detail["compact_score"] >= 0.7:
        reason_codes.append("compact_route")
    elif score_detail["total_travel_minutes"] >= 90:
        reason_codes.append("long_travel_time")
    if score_detail["queue_score"] >= 0.65:
        reason_codes.append("low_queue_risk")
    elif score_detail["queue_score"] <= 0.4:
        reason_codes.append("high_queue_risk")
    if score_detail["category_diversity_score"] >= 0.55:
        reason_codes.append("category_diverse")
    if score_detail["food_score"] >= 1.0:
        reason_codes.append("contains_food")
    else:
        reason_codes.append("missing_food")
    if score_detail["preference_match_score"] >= 0.65:
        reason_codes.append("matches_user_preferences")
    if score_detail.get("preference_coverage_score", 1) >= 1:
        reason_codes.append("covers_strong_preferences")
    elif score_detail.get("preference_coverage_score", 1) < 1:
        reason_codes.append("misses_strong_preferences")
    if "拍照" in preferences and score_detail["photo_score"] >= 0.7:
        reason_codes.append("photo_friendly_route")
    if "室内" in preferences and score_detail["indoor_score"] >= 0.7:
        reason_codes.append("indoor_friendly_route")
    return reason_codes


def _is_food_poi(poi: dict[str, Any]) -> bool:
    if poi.get("category") in FOOD_CATEGORIES:
        return True
    text = _poi_text(poi)
    return any(keyword in text for keyword in ("火锅", "小吃", "咖啡", "餐饮", "food", "hotpot", "coffee"))


def _resolve_budget(intent: dict[str, Any], user_profile: dict[str, Any]) -> float:
    if intent.get("budget") is not None:
        return max(0, _num(intent["budget"], DEFAULT_BUDGET))
    if user_profile.get("budget_per_day") is not None:
        return max(0, _num(user_profile["budget_per_day"], DEFAULT_BUDGET))
    return DEFAULT_BUDGET


def _keyword_score(poi: dict[str, Any], keywords: tuple[str, ...]) -> float:
    text = _poi_text(poi).lower()
    return 1.0 if any(keyword.lower() in text for keyword in keywords) else 0.0


def _poi_text(poi: dict[str, Any]) -> str:
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


def _feature(poi: dict[str, Any], name: str, default: float = 0.0) -> float:
    features = _to_dict(poi.get("features", {}))
    return _clamp(_num(features.get(name, default), default))


def _has_coordinates(poi: dict[str, Any]) -> bool:
    return poi.get("lat") is not None and poi.get("lng") is not None


def _looks_like_user_profile(value: dict[str, Any]) -> bool:
    return bool({"preferred_tags", "favorite_categories", "feature_weights"}.intersection(value))


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


def _parse_hhmm(value: str) -> int:
    hour_text, minute_text = str(value).split(":", 1)
    return int(hour_text) * 60 + int(minute_text)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.02, value) for key, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {key: 1 / len(cleaned) for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
