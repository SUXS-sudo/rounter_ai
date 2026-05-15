"""LLM-powered route generation with Beam Search fallback.

Uses MiMo API for intelligent route planning, keeping the original Beam Search
algorithm as a reliable fallback when the API is unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from core.mimo_client import chat_json
from core.preference import (
    STRONG_FOOD_PREFERENCES,
    candidate_has_preference,
    get_covered_strong_preferences,
    get_must_satisfy_preferences,
    get_strong_preferences,
    matches_preference,
    route_contains_food,
)
from core.scorer import score_poi, score_route
from utils.geo import auto_travel, haversine_distance
from utils.time_utils import DAY_MINUTES, format_hhmm, minutes_between, parse_hhmm

logger = logging.getLogger(__name__)

FOOD_CATEGORIES = {"food", "cafe"}
DEFAULT_START_TIME = "09:00"
DEFAULT_END_TIME = "21:00"
DEFAULT_BUDGET = 300.0
DEFAULT_START_LOCATION = {
    "label": "春熙路",
    "lat": 30.65708,
    "lng": 104.08096,
}

ROUTE_PLANNER_SYSTEM_PROMPT = """你是一个专业的城市出行路线规划助手。用户会提供出发信息和候选地点列表，你需要规划出3条不同的路线方案。

要求：
1. 每条路线包含3-5个地点，至少包含1个餐饮/咖啡类地点
2. 考虑地点的营业时间、预算约束、地理位置（减少绕路）
3. 3条路线的侧重点不同：综合最优、少排队优先、低预算优先
4. 为每个地点估算到达时间、停留时间、交通方式和费用
5. 交通费估算：0.5公里以内步行（免费），1-3公里公交/地铁（2-5元），3公里以上打车（起步价+里程费）

请严格返回以下 JSON 格式，不要添加任何其他文字：
{
  "routes": [
    {
      "title": "路线名称",
      "route_label": "综合最优路线/少排队优先路线/低预算优先路线",
      "pois": [
        {
          "poi_id": "POI的id",
          "name": "地点名称",
          "arrival_time": "HH:MM",
          "leave_time": "HH:MM",
          "stay_minutes": 停留分钟数,
          "travel_from_previous_minutes": 从上一站交通分钟数,
          "travel_mode": "步行/打车/地铁",
          "travel_cost": 交通费用数字,
          "estimated_queue_minutes": 预估排队分钟数,
          "reason": "推荐理由"
        }
      ]
    }
  ]
}"""


def generate_routes(
    start_location: dict[str, Any],
    candidate_pois: list[dict[str, Any]],
    intent: dict[str, Any],
    user_profile: dict[str, Any],
    top_k: int = 3,
    beam_size: int = 8,
    max_steps: int = 5,
) -> list[dict[str, Any]]:
    """Generate top route plans with local search first and LLM reranking.

    Args:
        start_location: Starting point dictionary with ``lat`` and ``lng``.
        candidate_pois: Candidate POIs as dictionaries.
        intent: Parsed user intent.
        user_profile: User profile dictionary.
        top_k: Number of final routes to return.
        beam_size: Beam size for fallback algorithm.
        max_steps: Maximum POIs per route.

    Returns:
        A list of route dictionaries with POI details, scores, and warnings.
    """

    if top_k <= 0 or not candidate_pois:
        return []

    intent_data = _to_dict(intent)
    profile_data = _to_dict(user_profile)
    candidates = [_to_dict(poi) for poi in candidate_pois]
    start = _normalize_start_location(start_location)

    local_routes = _beam_search_generate(start, candidates, intent_data, profile_data, max(top_k, 5), beam_size, max_steps)
    if not local_routes and len(candidates) > 24:
        local_routes = _beam_search_generate(start, candidates[:24], intent_data, profile_data, max(top_k, 5), beam_size, max_steps)
    if not local_routes:
        try:
            llm_routes = _llm_generate_routes(start, candidates, intent_data, profile_data, top_k)
            return _finalize_routes(llm_routes, intent_data, candidates, top_k)
        except Exception as exc:
            logger.warning("LLM route planning failed and Beam Search produced nothing: %s", exc)
            return []

    return _finalize_routes(local_routes, intent_data, candidates, top_k)


def _llm_generate_routes(
    start: dict[str, Any],
    candidates: list[dict[str, Any]],
    intent: dict[str, Any],
    user_profile: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Use MiMo API to generate route plans."""

    poi_summaries = [_summarize_poi_for_llm(poi) for poi in candidates[:12]]
    preferences_text = "、".join(intent.get("preferences", [])) or "无特殊偏好"
    avoid_text = "、".join(intent.get("avoid", [])) or "无"

    user_prompt = f"""出发地点：{start.get('label', '春熙路')}（纬度{start.get('lat')}, 经度{start.get('lng')}）
出发时间：{intent.get('start_time', '09:00')}
结束时间：{intent.get('end_time', '21:00')}
预算：{intent.get('budget', 300)}元
偏好：{preferences_text}
避开：{avoid_text}
出行方式：{intent.get('travel_mode', 'walking')}
人数：{intent.get('people_count', 1)}人
场景：{intent.get('scenario', 'general')}

候选地点列表（JSON）：
{json.dumps(poi_summaries, ensure_ascii=False)}

请规划{top_k}条路线方案。"""

    result = chat_json(
        system_prompt=ROUTE_PLANNER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=1536,
    )

    raw_routes = result.get("routes", [])
    if not raw_routes:
        return []

    enriched_routes = []
    poi_map = {str(poi.get("id")): poi for poi in candidates}
    start_minutes = parse_hhmm(str(intent.get("start_time", DEFAULT_START_TIME)))
    budget = _resolve_budget(intent, user_profile)

    for idx, raw_route in enumerate(raw_routes[:top_k]):
        try:
            route = _enrich_llm_route(raw_route, poi_map, start, start_minutes, budget, intent, user_profile, idx)
            if route and route.get("pois"):
                enriched_routes.append(route)
        except Exception as exc:
            logger.warning("Failed to enrich route %d: %s", idx, exc)
            continue

    return enriched_routes


def _llm_rerank_routes(
    start: dict[str, Any],
    intent: dict[str, Any],
    user_profile: dict[str, Any],
    routes: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Ask the LLM to rerank locally generated routes."""

    route_summaries = []
    for index, route in enumerate(routes[:8], start=1):
        pois = route.get("pois", [])
        route_summaries.append({
            "index": index,
            "title": route.get("title", f"路线{index}"),
            "total_score": route.get("total_score", 0),
            "total_budget": route.get("total_budget", 0),
            "total_travel_minutes": route.get("total_travel_minutes", 0),
            "warnings": route.get("warnings", [])[:2],
            "pois": [
                {
                    "poi_id": step.get("poi_id"),
                    "name": step.get("name"),
                    "category": step.get("category"),
                    "arrival_time": step.get("arrival_time"),
                }
                for step in pois[:5]
            ],
        })

    user_prompt = f"""起点：{start.get('label', '春熙路')}
时间：{intent.get('start_time', '09:00')} - {intent.get('end_time', '21:00')}
预算：{intent.get('budget', 300)}元
偏好：{json.dumps(intent.get('preferences', []), ensure_ascii=False)}
避开：{json.dumps(intent.get('avoid', []), ensure_ascii=False)}

候选路线：
{json.dumps(route_summaries, ensure_ascii=False)}

请只基于这些候选路线选出最优的{top_k}条，必要时调整顺序和标题，并返回 JSON：
{{"routes": [{{"index": 1, "title": "...", "route_label": "...", "reason": "..."}}]}}"""

    result = chat_json(
        system_prompt="你是一个路线重排助手，只能基于给定候选路线进行排序，不要重新生成新路线。",
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=768,
    )

    ranking = result.get("routes", [])
    if not isinstance(ranking, list) or not ranking:
        return []

    indexed_routes = {idx + 1: dict(route) for idx, route in enumerate(routes[:10])}
    selected: list[dict[str, Any]] = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        route = indexed_routes.get(idx)
        if route is None:
            continue
        route["title"] = item.get("title") or route.get("title")
        if item.get("route_label"):
            route["route_label"] = item.get("route_label")
        if item.get("reason"):
            route.setdefault("warnings", [])
            route["warnings"].append(str(item.get("reason")))
        selected.append(route)
        if len(selected) >= top_k:
            break

    return selected


def _enrich_llm_route(
    raw_route: dict[str, Any],
    poi_map: dict[str, dict[str, Any]],
    start: dict[str, Any],
    start_minutes: int,
    budget: float,
    intent: dict[str, Any],
    user_profile: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    """Validate and enrich an LLM-generated route with calculated fields."""

    raw_pois = raw_route.get("pois", [])
    if not raw_pois or len(raw_pois) < 2:
        return None

    steps = []
    current_lat = start.get("lat", DEFAULT_START_LOCATION["lat"])
    current_lng = start.get("lng", DEFAULT_START_LOCATION["lng"])
    current_time = start_minutes
    total_budget = 0.0
    total_travel_cost = 0.0
    total_travel_minutes = 0
    total_queue_minutes = 0
    total_stay_minutes = 0

    for step_data in raw_pois:
        poi_id = str(step_data.get("poi_id", ""))
        poi = poi_map.get(poi_id)
        if poi is None:
            # Try matching by name
            for pid, p in poi_map.items():
                if p.get("name") == step_data.get("name"):
                    poi = p
                    poi_id = pid
                    break
        if poi is None:
            continue

        # Calculate actual travel time from coordinates
        distance_km = haversine_distance(current_lat, current_lng, _num(poi.get("lat")), _num(poi.get("lng")))
        travel = auto_travel(distance_km)
        travel_minutes = travel["minutes"]
        travel_cost = travel["cost"]

        # Use LLM's time estimates but validate with calculated travel
        arrival_time_str = step_data.get("arrival_time", format_hhmm(current_time + travel_minutes))
        arrival_minutes = parse_hhmm(arrival_time_str) if arrival_time_str else current_time + travel_minutes

        stay_minutes = max(1, int(_num(step_data.get("stay_minutes"), poi.get("avg_stay_minutes", 60))))
        queue_minutes = int(_num(step_data.get("estimated_queue_minutes"), 0))
        leave_minutes = arrival_minutes + stay_minutes + queue_minutes

        price = _num(poi.get("price"))
        stop_travel_cost = _num(step_data.get("travel_cost"), travel_cost)

        steps.append({
            "poi": poi,
            "poi_id": poi_id,
            "name": poi.get("name", ""),
            "arrival_time": format_hhmm(arrival_minutes),
            "leave_time": format_hhmm(leave_minutes),
            "stay_minutes": stay_minutes,
            "travel_from_previous_minutes": travel_minutes,
            "travel_mode": step_data.get("travel_mode", travel["mode_cn"]),
            "travel_cost": stop_travel_cost,
            "estimated_queue_minutes": queue_minutes,
            "reason": step_data.get("reason", "LLM推荐"),
            "_arrival_minutes": arrival_minutes,
            "_leave_minutes": leave_minutes,
            "_distance_from_previous_km": round(distance_km, 3),
        })

        current_lat = _num(poi.get("lat"))
        current_lng = _num(poi.get("lng"))
        current_time = leave_minutes
        total_budget += price + stop_travel_cost
        total_travel_cost += stop_travel_cost
        total_travel_minutes += travel_minutes
        total_queue_minutes += queue_minutes
        total_stay_minutes += stay_minutes

    if len(steps) < 2:
        return None

    # Score the route using existing scorer
    route_stops = [{"poi": step["poi"], "travel_minutes_from_previous": step["travel_from_previous_minutes"]} for step in steps]
    scorer_result = score_route({"stops": route_stops}, intent, user_profile)
    total_duration_minutes = max(0, current_time - start_minutes)

    # Build warnings
    warnings = []
    if budget > 0 and total_budget > budget:
        over = int(round(total_budget - budget))
        warnings.append(f"预算超出约{over}元。")
    if total_queue_minutes >= 60:
        warnings.append(f"预计总排队时间约{total_queue_minutes}分钟。")
    if total_travel_minutes >= 90:
        warnings.append(f"预计交通时间约{total_travel_minutes}分钟。")

    # Determine route title
    default_titles = ["综合最优路线", "少排队优先路线", "低预算优先路线"]
    title = raw_route.get("route_label") or raw_route.get("title") or (default_titles[index] if index < len(default_titles) else f"路线{index + 1}")

    return {
        "route_id": f"route_{index:03d}",
        "title": title,
        "total_score": round(scorer_result["score"], 4),
        "total_budget": int(round(total_budget)),
        "total_travel_cost": round(total_travel_cost, 1),
        "total_travel_minutes": total_travel_minutes,
        "total_duration_minutes": total_duration_minutes,
        "pois": [_public_step(step) for step in steps],
        "score_detail": scorer_result.get("score_detail", {}),
        "reason_codes": scorer_result.get("reason_codes", []),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Beam Search fallback (original algorithm)
# ---------------------------------------------------------------------------


@dataclass
class RouteState:
    """Internal state used by Beam Search."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    visited_ids: set[str] = field(default_factory=set)
    current_lat: float = DEFAULT_START_LOCATION["lat"]
    current_lng: float = DEFAULT_START_LOCATION["lng"]
    current_time: int = 0
    total_budget: float = 0.0
    total_travel_cost: float = 0.0
    total_travel_minutes: int = 0
    total_queue_minutes: int = 0
    total_wait_minutes: int = 0
    total_stay_minutes: int = 0
    warnings: list[str] = field(default_factory=list)
    beam_score: float = 0.0


def _beam_search_generate(
    start: dict[str, Any],
    candidates: list[dict[str, Any]],
    intent_data: dict[str, Any],
    profile_data: dict[str, Any],
    top_k: int,
    beam_size: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Original Beam Search route generation algorithm."""

    max_steps = max(3, min(5, max_steps))
    beam_size = max(1, beam_size)
    if get_strong_preferences(intent_data):
        beam_size = max(beam_size, 18)

    start_time = str(intent_data.get("start_time") or DEFAULT_START_TIME)
    end_time = str(intent_data.get("end_time") or DEFAULT_END_TIME)
    start_minutes = parse_hhmm(start_time)
    end_minutes = start_minutes + minutes_between(start_time, end_time)
    if end_minutes <= start_minutes:
        end_minutes += DAY_MINUTES

    initial_state = RouteState(
        current_lat=_num(start.get("lat"), DEFAULT_START_LOCATION["lat"]),
        current_lng=_num(start.get("lng"), DEFAULT_START_LOCATION["lng"]),
        current_time=start_minutes,
        beam_score=0.0,
    )

    beam: list[RouteState] = [initial_state]
    complete_states: list[RouteState] = []

    for _ in range(max_steps):
        expanded_states: list[RouteState] = []
        for state in beam:
            expanded_states.extend(
                _expand_state(state, candidates, intent_data, profile_data, end_minutes)
            )
        if not expanded_states:
            break
        expanded_states.sort(key=lambda item: item.beam_score, reverse=True)
        beam = expanded_states[:beam_size]
        for state in beam:
            if 3 <= len(state.steps) <= max_steps and _has_food_poi(state.steps):
                complete_states.append(state)

    unique_states = _dedupe_states(complete_states)
    if not unique_states:
        return []

    budget = _resolve_budget(intent_data, profile_data)
    final_routes = [
        _build_route(state, f"route_{index:03d}", "候选路线", intent_data, profile_data, start_minutes, budget)
        for index, state in enumerate(unique_states, start=1)
    ]
    required_preferences = _required_preferences_present_in_candidates(intent_data, candidates)
    final_routes = _enforce_strong_preference_coverage(final_routes, required_preferences)
    selected_routes = select_diverse_routes(final_routes, top_k)
    for index, route in enumerate(selected_routes, start=1):
        route["route_id"] = f"route_{index:03d}"
    return selected_routes


def optimize_route(pois: list[Any], max_pois: int = 6) -> list[Any]:
    """Backward-compatible helper that returns high-rating POIs first."""
    return sorted(pois, key=lambda poi: _num(_to_dict(poi).get("rating")), reverse=True)[:max_pois]


def _expand_state(
    state: RouteState,
    candidates: list[dict[str, Any]],
    intent: dict[str, Any],
    user_profile: dict[str, Any],
    end_minutes: int,
) -> list[RouteState]:
    expanded: list[RouteState] = []
    prefer_low_queue = "少排队" in _preferences(intent)

    for poi in candidates:
        poi_id = str(poi.get("id") or poi.get("name") or id(poi))
        if poi_id in state.visited_ids:
            continue
        if not _has_coordinates(poi):
            continue

        distance_km = haversine_distance(state.current_lat, state.current_lng, _num(poi.get("lat")), _num(poi.get("lng")))
        travel = auto_travel(distance_km)
        travel_minutes = travel["minutes"]
        travel_cost = travel["cost"]
        arrival_minutes = state.current_time + travel_minutes
        queue_minutes = _estimated_queue_minutes(poi, prefer_low_queue)
        stay_minutes = max(1, int(_num(poi.get("avg_stay_minutes"), 60)))
        visit_minutes = stay_minutes + queue_minutes
        schedule = _schedule_visit(poi, arrival_minutes, visit_minutes)

        if not schedule["ok"]:
            continue
        if schedule["leave_minutes"] > end_minutes:
            continue

        step = {
            "poi": poi,
            "poi_id": poi_id,
            "name": poi.get("name", ""),
            "arrival_time": format_hhmm(arrival_minutes),
            "leave_time": format_hhmm(schedule["leave_minutes"]),
            "stay_minutes": stay_minutes,
            "travel_from_previous_minutes": travel_minutes,
            "travel_mode": travel["mode_cn"],
            "travel_cost": travel_cost,
            "estimated_queue_minutes": queue_minutes,
            "reason": _poi_reason(poi, intent, distance_km, schedule["wait_minutes"], queue_minutes),
            "_arrival_minutes": arrival_minutes,
            "_visit_start_minutes": schedule["visit_start_minutes"],
            "_leave_minutes": schedule["leave_minutes"],
            "_wait_minutes": schedule["wait_minutes"],
            "_distance_from_previous_km": round(distance_km, 3),
        }

        new_steps = state.steps + [step]
        new_warnings = list(state.warnings)
        total_budget = state.total_budget + _num(poi.get("price")) + travel_cost
        total_travel_cost = state.total_travel_cost + travel_cost
        budget = _resolve_budget(intent, user_profile)
        lower_bound, upper_bound = _budget_band(budget)
        if budget > 0:
            if total_budget > upper_bound and not any("预算" in w for w in new_warnings):
                new_warnings.append(f"预计总预算{int(round(total_budget))}元，高于目标区间上界{int(round(upper_bound))}元，已在评分中扣分。")
            elif len(new_steps) >= 3 and total_budget < lower_bound and len(new_steps) == max(3, min(5, len(new_steps))) and not any("预算" in w for w in new_warnings):
                new_warnings.append(f"预计总预算{int(round(total_budget))}元，低于目标区间下界{int(round(lower_bound))}元，路线可能偏保守。")

        new_state = RouteState(
            steps=new_steps,
            visited_ids=set(state.visited_ids) | {poi_id},
            current_lat=_num(poi.get("lat")),
            current_lng=_num(poi.get("lng")),
            current_time=schedule["leave_minutes"],
            total_budget=total_budget,
            total_travel_cost=total_travel_cost,
            total_travel_minutes=state.total_travel_minutes + travel_minutes,
            total_queue_minutes=state.total_queue_minutes + queue_minutes,
            total_wait_minutes=state.total_wait_minutes + schedule["wait_minutes"],
            total_stay_minutes=state.total_stay_minutes + stay_minutes,
            warnings=new_warnings,
        )
        new_state.beam_score = _partial_beam_score(new_state, intent, user_profile)
        expanded.append(new_state)

    return expanded


def _build_route(
    state: RouteState,
    route_id: str,
    title: str,
    intent: dict[str, Any],
    user_profile: dict[str, Any],
    start_minutes: int,
    budget: float,
) -> dict[str, Any]:
    route_stops = [{"poi": step["poi"], "travel_minutes_from_previous": step["travel_from_previous_minutes"]} for step in state.steps]
    scorer_result = score_route({"stops": route_stops}, intent, user_profile)
    route_adjustment = _route_adjustment_score(state, intent, budget)
    route_pois = [step["poi"] for step in state.steps]
    must_satisfy_preferences = get_must_satisfy_preferences(intent)
    strong_preferences = get_strong_preferences(intent)
    covered_strong_preferences = get_covered_strong_preferences(route_pois, intent)
    covered_must_satisfy_preferences = [p for p in must_satisfy_preferences if any(matches_preference(poi, p) for poi in route_pois)]
    coverage_source = must_satisfy_preferences or strong_preferences
    covered_source = covered_must_satisfy_preferences if must_satisfy_preferences else covered_strong_preferences
    preference_coverage_score = len(covered_source) / len(coverage_source) if coverage_source else 1.0
    missing_food_preference = (
        any(p in STRONG_FOOD_PREFERENCES for p in coverage_source)
        and not any(p in covered_source for p in STRONG_FOOD_PREFERENCES)
    )
    missing_food_penalty = 0.35 if missing_food_preference else 0.0
    total_score = _clamp(
        scorer_result["score"] * 0.58 + route_adjustment["score"] * 0.22 + preference_coverage_score * 0.20 - missing_food_penalty
    )
    total_duration_minutes = max(0, state.current_time - start_minutes)
    warnings = _route_warnings(state, budget)

    score_detail = {
        **scorer_result.get("score_detail", {}),
        "beam_score": round(state.beam_score, 4),
        "route_adjustment_score": round(route_adjustment["score"], 4),
        "preference_coverage_score": round(preference_coverage_score, 4),
        "must_satisfy_preferences": must_satisfy_preferences,
        "covered_must_satisfy_preferences": covered_must_satisfy_preferences,
        "covered_strong_preferences": covered_strong_preferences,
        "missing_food_preference": missing_food_preference,
        "budget_penalty": round(route_adjustment["budget_penalty"], 4),
        "queue_penalty": round(route_adjustment["queue_penalty"], 4),
        "time_efficiency_score": round(route_adjustment["time_efficiency_score"], 4),
        "indoor_adjustment_score": round(route_adjustment["indoor_adjustment_score"], 4),
        "total_wait_minutes": state.total_wait_minutes,
    }

    return {
        "route_id": route_id,
        "title": title,
        "total_score": round(total_score, 4),
        "total_budget": int(round(state.total_budget)),
        "total_travel_cost": round(state.total_travel_cost, 1),
        "total_travel_minutes": state.total_travel_minutes,
        "total_duration_minutes": total_duration_minutes,
        "pois": [_public_step(step) for step in state.steps],
        "score_detail": score_detail,
        "reason_codes": scorer_result.get("reason_codes", []),
        "warnings": warnings,
    }


def _finalize_routes(
    routes: list[dict[str, Any]],
    intent: dict[str, Any],
    candidate_pois: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    required_preferences = _required_preferences_present_in_candidates(intent, candidate_pois)
    if not _constraints_feasible(required_preferences, candidate_pois):
        return _prefer_budget_band_routes(routes, intent)[:top_k]
    routes = _enforce_strong_preference_coverage(routes, required_preferences)
    return _prefer_budget_band_routes(routes, intent)[:top_k]


def _required_preferences_present_in_candidates(intent: dict[str, Any], candidate_pois: list[dict[str, Any]]) -> list[str]:
    must_satisfy = get_must_satisfy_preferences(intent)
    if must_satisfy:
        return [p for p in must_satisfy if candidate_has_preference(candidate_pois, p)]
    return [p for p in get_strong_preferences(intent) if candidate_has_preference(candidate_pois, p)]


def _constraints_feasible(required_preferences: list[str], candidate_pois: list[dict[str, Any]]) -> bool:
    if not required_preferences:
        return True
    if any(not candidate_has_preference(candidate_pois, preference) for preference in required_preferences):
        return False
    if any(preference == "室内" for preference in required_preferences):
        return any(_is_food_poi(poi) and matches_preference(poi, "室内") for poi in candidate_pois)
    return True


def _prefer_budget_band_routes(routes: list[dict[str, Any]], intent: dict[str, Any]) -> list[dict[str, Any]]:
    budget = _resolve_budget(intent, {})
    lower_bound, upper_bound = _budget_band(budget)
    in_band = [route for route in routes if lower_bound <= float(route.get("total_budget", 0)) <= upper_bound]
    out_of_band = [route for route in routes if not (lower_bound <= float(route.get("total_budget", 0)) <= upper_bound)]
    in_band = sorted(in_band, key=lambda route: float(route.get("total_score", 0)), reverse=True)
    out_of_band = sorted(out_of_band, key=lambda route: (_budget_band_distance(float(route.get("total_budget", 0)), budget), -float(route.get("total_score", 0))))
    return in_band + out_of_band


def _enforce_strong_preference_coverage(routes: list[dict[str, Any]], required_preferences: list[str]) -> list[dict[str, Any]]:
    if not required_preferences:
        return routes
    covered_routes = [
        r for r in routes
        if all(p in r.get("score_detail", {}).get("covered_strong_preferences", []) for p in required_preferences)
    ]
    if covered_routes:
        return covered_routes
    relaxed_routes: list[dict[str, Any]] = []
    for route in routes:
        route = dict(route)
        warnings = list(route.get("warnings", []))
        covered = set(route.get("score_detail", {}).get("covered_strong_preferences", []))
        missing = [p for p in required_preferences if p not in covered]
        if missing:
            route["total_score"] = round(_clamp(float(route.get("total_score", 0)) - 0.30), 4)
            warnings.append(f"候选中存在{','.join(missing)}点位，但该路线未覆盖，已在评分中大幅扣分。")
        route["warnings"] = list(dict.fromkeys(warnings))
        relaxed_routes.append(route)
    return relaxed_routes


def select_diverse_routes(routes: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    """Select high-scoring routes while limiting POI-set similarity."""
    selected: list[dict[str, Any]] = []
    objectives = [
        ("综合最优路线", lambda r: r["total_score"]),
        ("少排队优先路线", lambda r: (
            r["score_detail"].get("queue_score", 0) * 0.55
            + r["total_score"] * 0.35
            + (1 - _clamp(r["score_detail"].get("average_queue_risk", 0.5))) * 0.10
        )),
        ("低预算优先路线", lambda r: (
            r["score_detail"].get("budget_score", 0) * 0.55
            + (1 - _clamp(r["total_budget"] / max(1, r["score_detail"].get("budget", DEFAULT_BUDGET)))) * 0.25
            + r["total_score"] * 0.20
        )),
    ]
    for title, key_func in objectives[:max(0, min(top_k, len(objectives)))]:
        route = _best_diverse_route(routes, key_func, selected, max_similarity=0.7)
        if route is None:
            continue
        route = dict(route)
        route["title"] = title
        selected.append(route)
    if len(selected) < top_k:
        fallback_routes = sorted(routes, key=lambda item: item["total_score"], reverse=True)
        for route in fallback_routes:
            if any(_route_signature(route) == _route_signature(e) for e in selected):
                continue
            route = dict(route)
            route["title"] = f"备选路线{len(selected) + 1}"
            selected.append(route)
            if len(selected) >= top_k:
                break
    return selected[:top_k]


def _best_diverse_route(routes: list[dict[str, Any]], key_func: Any, selected: list[dict[str, Any]], max_similarity: float) -> dict[str, Any] | None:
    ranked = sorted(routes, key=key_func, reverse=True)
    for route in ranked:
        signature = _route_signature(route)
        if any(signature == _route_signature(e) for e in selected):
            continue
        if all(_jaccard_similarity(signature, _route_signature(e)) <= max_similarity for e in selected):
            return route
    for route in ranked:
        signature = _route_signature(route)
        if not any(signature == _route_signature(e) for e in selected):
            return route
    return None


def _jaccard_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set.intersection(right_set)) / len(left_set.union(right_set))


def _partial_beam_score(state: RouteState, intent: dict[str, Any], user_profile: dict[str, Any]) -> float:
    if not state.steps:
        return 0.0
    poi_scores = [
        score_poi(step["poi"], intent, user_profile, context={
            "travel_minutes": step["travel_from_previous_minutes"],
            "distance_from_previous_km": step["_distance_from_previous_km"],
            "route_categories": [prev["poi"].get("category") for prev in state.steps[:-1]],
        })["score"]
        for step in state.steps
    ]
    avg_poi_score = sum(poi_scores) / len(poi_scores)
    queue_weight = 1.35 if "少排队" in _preferences(intent) else 1.0
    queue_penalty = _clamp((state.total_queue_minutes / max(1, len(state.steps))) / 40 * 0.28 * queue_weight)
    travel_limit = 95 if "少走路" in _preferences(intent) else 160
    travel_penalty = _clamp(state.total_travel_minutes / travel_limit * 0.22)
    food_bonus = 0.08 if _has_food_poi(state.steps) else 0.0
    length_bonus = min(0.10, len(state.steps) * 0.02)
    route_pois = [step["poi"] for step in state.steps]
    strong_preferences = get_strong_preferences(intent)
    covered = get_covered_strong_preferences(route_pois, intent)
    coverage_bonus = 0.16 * (len(covered) / len(strong_preferences)) if strong_preferences else 0.0
    missing_food_penalty = (
        0.20
        if any(p in STRONG_FOOD_PREFERENCES for p in strong_preferences) and not any(p in covered for p in STRONG_FOOD_PREFERENCES)
        else 0.0
    )
    return _clamp(avg_poi_score - queue_penalty - travel_penalty + food_bonus + length_bonus + coverage_bonus - missing_food_penalty)


def _route_adjustment_score(state: RouteState, intent: dict[str, Any], budget: float) -> dict[str, float]:
    preferences = _preferences(intent)
    prefer_low_queue = "少排队" in preferences
    prefer_compact = "少走路" in preferences
    prefer_indoor = "室内" in preferences
    queue_weight = 1.45 if prefer_low_queue else 1.0
    lower_bound, upper_bound = _budget_band(budget)
    budget_penalty = 0.0
    if budget > 0:
        if state.total_budget < lower_bound:
            budget_penalty = _clamp((lower_bound - state.total_budget) / max(50.0, budget))
        elif state.total_budget > upper_bound:
            budget_penalty = _clamp((state.total_budget - upper_bound) / max(40.0, budget))
    queue_penalty = _clamp((state.total_queue_minutes / max(1, len(state.steps))) / 40 * queue_weight)
    travel_threshold = 70 if prefer_compact else 120
    time_efficiency_score = _clamp(1 - state.total_travel_minutes / travel_threshold)
    wait_score = _clamp(1 - state.total_wait_minutes / 90)
    completion_score = 1.0 if 3 <= len(state.steps) <= 5 and _has_food_poi(state.steps) else 0.4
    indoor_score = (
        sum(_feature(step["poi"], "indoor", 0.5) for step in state.steps) / len(state.steps) if state.steps else 0.5
    )
    score = _clamp(
        time_efficiency_score * (0.38 if prefer_compact else 0.30)
        + wait_score * 0.15
        + (1 - queue_penalty) * (0.32 if prefer_low_queue else 0.25)
        + (1 - budget_penalty) * 0.15
        + completion_score * 0.15
        + (indoor_score * 0.26 if prefer_indoor else 0)
    )
    return {"score": score, "budget_penalty": budget_penalty, "queue_penalty": queue_penalty, "time_efficiency_score": time_efficiency_score, "indoor_adjustment_score": indoor_score}


def _schedule_visit(poi: dict[str, Any], arrival_minutes: int, visit_minutes: int) -> dict[str, Any]:
    open_minutes = parse_hhmm(str(poi.get("open_time", "00:00")))
    close_minutes = parse_hhmm(str(poi.get("close_time", "23:59")))
    preferred_start, preferred_end = _preferred_visit_window(poi)
    if open_minutes == close_minutes:
        adjusted_start = _align_to_preferred_window(arrival_minutes, preferred_start, preferred_end)
        return {"ok": True, "visit_start_minutes": adjusted_start, "leave_minutes": adjusted_start + visit_minutes, "wait_minutes": max(0, adjusted_start - arrival_minutes), "reason": "全天营业"}
    for window_start, window_end in _candidate_business_windows(open_minutes, close_minutes, arrival_minutes):
        if arrival_minutes > window_end:
            continue
        visit_start = max(arrival_minutes, window_start)
        visit_start = _align_to_preferred_window(visit_start, preferred_start, preferred_end)
        if visit_start > window_end:
            continue
        leave_minutes = visit_start + visit_minutes
        if leave_minutes <= window_end:
            return {"ok": True, "visit_start_minutes": visit_start, "leave_minutes": leave_minutes, "wait_minutes": max(0, visit_start - arrival_minutes), "reason": "营业时间内可访问"}
        if arrival_minutes <= window_end:
            return {"ok": False, "visit_start_minutes": visit_start, "leave_minutes": leave_minutes, "wait_minutes": max(0, visit_start - arrival_minutes), "reason": f"预计{format_hhmm(leave_minutes)}离开，超过闭店时间{format_hhmm(window_end)}"}
    return {"ok": False, "visit_start_minutes": arrival_minutes, "leave_minutes": arrival_minutes + visit_minutes, "wait_minutes": 0, "reason": "没有匹配到可访问的营业时间窗口"}


def _preferred_visit_window(poi: dict[str, Any]) -> tuple[int | None, int | None]:
    text = _poi_text(poi).lower()
    if _is_food_poi(poi):
        return 11 * 60, 20 * 60
    if any(keyword in text for keyword in ("park", "公园", "植物园", "动物园", "自然", "landmark", "古镇", "historic", "tower")):
        return 9 * 60, 17 * 60
    if any(keyword in text for keyword in ("night_view", "夜景", "bar", "酒吧", "livehouse")):
        return 18 * 60, 23 * 60
    if any(keyword in text for keyword in ("museum", "gallery", "bookstore", "library", "culture", "theater", "heritage", "exhibition")):
        return 10 * 60, 18 * 60
    return None, None


def _align_to_preferred_window(minutes: int, preferred_start: int | None, preferred_end: int | None) -> int:
    if preferred_start is None or preferred_end is None:
        return minutes
    if minutes < preferred_start:
        return preferred_start
    if minutes > preferred_end:
        return minutes
    return minutes


def _candidate_business_windows(open_minutes: int, close_minutes: int, arrival_minutes: int) -> list[tuple[int, int]]:
    day = arrival_minutes // DAY_MINUTES
    windows: list[tuple[int, int]] = []
    for day_offset in range(-1, 3):
        base = (day + day_offset) * DAY_MINUTES
        window_start = base + open_minutes
        window_end = base + close_minutes
        if close_minutes < open_minutes:
            window_end += DAY_MINUTES
        windows.append((window_start, window_end))
    return sorted(windows, key=lambda item: item[0])


def _estimated_queue_minutes(poi: dict[str, Any], prefer_low_queue: bool) -> int:
    queue_risk = _feature(poi, "queue_risk", 0.5)
    multiplier = 1.35 if prefer_low_queue else 1.0
    return int(round(queue_risk * 40 * multiplier))


def _poi_reason(poi: dict[str, Any], intent: dict[str, Any], distance_km: float, wait_minutes: int, queue_minutes: int) -> str:
    preferences = set(_preferences(intent))
    reasons: list[str] = []
    if "拍照" in preferences and _feature(poi, "photo", 0.5) >= 0.7:
        reasons.append("适合拍照")
    if "夜景" in preferences and _feature(poi, "night_view", 0.5) >= 0.7:
        reasons.append("夜景表现好")
    if "室内" in preferences and _feature(poi, "indoor", 0.5) >= 0.7:
        reasons.append("适合室内安排")
    if "少排队" in preferences and _feature(poi, "queue_risk", 0.5) <= 0.45:
        reasons.append("排队风险较低")
    if distance_km <= 0.8:
        reasons.append("距离上一站近")
    if _is_food_poi(poi):
        reasons.append("可补充餐饮")
    if wait_minutes > 0:
        reasons.append(f"需等待{wait_minutes}分钟开门")
    if queue_minutes > 0 and "少排队" not in preferences:
        reasons.append(f"预计排队{queue_minutes}分钟")
    return "，".join(reasons[:3]) or "综合评分较高，适合作为路线节点"


def _route_warnings(state: RouteState, budget: float) -> list[str]:
    warnings = list(dict.fromkeys(state.warnings))
    if budget > 0:
        lower_bound, upper_bound = _budget_band(budget)
        if state.total_budget < lower_bound:
            under = int(round(lower_bound - state.total_budget))
            if not any("预算" in w for w in warnings):
                warnings.append(f"总花费低于目标区间下界约{under}元，路线可能偏保守。")
        elif state.total_budget > upper_bound:
            over = int(round(state.total_budget - upper_bound))
            if not any("预算" in w for w in warnings):
                warnings.append(f"总花费高于目标区间上界约{over}元，路线偏贵。")
    if state.total_queue_minutes >= 60:
        warnings.append(f"预计总排队时间约{state.total_queue_minutes}分钟。")
    if state.total_travel_minutes >= 90:
        warnings.append(f"预计交通时间约{state.total_travel_minutes}分钟，路线可能偏分散。")
    return list(dict.fromkeys(warnings))


def _summarize_poi_for_llm(poi: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "id": str(poi.get("id", "")),
        "name": poi.get("name", ""),
        "category": poi.get("category", ""),
        "sub_category": poi.get("sub_category", ""),
        "rating": poi.get("rating", 0),
        "price": poi.get("price", 0),
        "open_time": poi.get("open_time", "00:00"),
        "close_time": poi.get("close_time", "23:59"),
        "avg_stay_minutes": poi.get("avg_stay_minutes", 60),
    }
    features = poi.get("features", {})
    if isinstance(features, dict):
        summary["queue_risk"] = round(features.get("queue_risk", 0.5), 2)
        summary["photo_score"] = round(features.get("photo", 0.5), 2)
        summary["indoor_score"] = round(features.get("indoor", 0.5), 2)
    return summary


def _public_step(step: dict[str, Any]) -> dict[str, Any]:
    poi = step.get("poi", {})
    price = poi.get("price", 0)
    travel_cost = step.get("travel_cost", 0)
    return {
        "poi_id": step["poi_id"],
        "name": step["name"],
        "lat": poi.get("lat"),
        "lng": poi.get("lng"),
        "category": poi.get("category", ""),
        "price": price,
        "travel_cost": travel_cost,
        "stop_cost": round(price + travel_cost, 1),
        "rating": poi.get("rating", 0),
        "arrival_time": step["arrival_time"],
        "leave_time": step["leave_time"],
        "stay_minutes": step["stay_minutes"],
        "travel_from_previous_minutes": step["travel_from_previous_minutes"],
        "travel_mode": step.get("travel_mode", "步行"),
        "estimated_queue_minutes": step["estimated_queue_minutes"],
        "reason": step["reason"],
    }


def _dedupe_states(states: list[RouteState]) -> list[RouteState]:
    best_by_signature: dict[tuple[str, ...], RouteState] = {}
    for state in states:
        signature = tuple(step["poi_id"] for step in state.steps)
        current = best_by_signature.get(signature)
        if current is None or state.beam_score > current.beam_score:
            best_by_signature[signature] = state
    return sorted(best_by_signature.values(), key=lambda item: item.beam_score, reverse=True)


def _route_signature(route: dict[str, Any]) -> tuple[str, ...]:
    return tuple(step["poi_id"] for step in route.get("pois", []))


def _normalize_start_location(start_location: dict[str, Any] | None) -> dict[str, Any]:
    start = _to_dict(start_location)
    if start.get("lat") is not None and start.get("lng") is not None:
        return start
    label = str(start.get("label") or start.get("name") or start.get("start_location") or "")
    if "春熙路" in label or not label:
        return dict(DEFAULT_START_LOCATION)
    return {**DEFAULT_START_LOCATION, "label": label}


def _has_food_poi(steps: list[dict[str, Any]]) -> bool:
    return route_contains_food([step["poi"] for step in steps])


def _is_food_poi(poi: dict[str, Any]) -> bool:
    if poi.get("category") in FOOD_CATEGORIES:
        return True
    text = _poi_text(poi)
    return any(k in text for k in ("火锅", "小吃", "咖啡", "奶茶", "茶饮", "果茶", "茶馆", "甜品", "餐饮", "food", "hotpot", "coffee", "milk_tea", "bubble_tea", "teahouse", "dessert"))


def _budget_band(budget: float) -> tuple[float, float]:
    if budget <= 0:
        return 0.0, 0.0
    return max(0.0, budget - 50), budget + 20


def _budget_band_distance(total_budget: float, budget: float) -> float:
    lower_bound, upper_bound = _budget_band(budget)
    if lower_bound <= total_budget <= upper_bound:
        return 0.0
    if total_budget < lower_bound:
        return lower_bound - total_budget
    return total_budget - upper_bound


def _resolve_budget(intent: dict[str, Any], user_profile: dict[str, Any]) -> float:
    if intent.get("budget") is not None:
        return max(0.0, _num(intent["budget"], DEFAULT_BUDGET))
    if user_profile.get("budget_per_day") is not None:
        return max(0.0, _num(user_profile["budget_per_day"], DEFAULT_BUDGET))
    return DEFAULT_BUDGET


def _preferences(intent: dict[str, Any]) -> list[str]:
    value = intent.get("preferences") or []
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def _has_coordinates(poi: dict[str, Any]) -> bool:
    return poi.get("lat") is not None and poi.get("lng") is not None


def _feature(poi: dict[str, Any], name: str, default: float = 0.0) -> float:
    features = _to_dict(poi.get("features", {}))
    return _clamp(_num(features.get(name, default), default))


def _poi_text(poi: dict[str, Any]) -> str:
    tags = " ".join(str(tag) for tag in _as_list(poi.get("tags")))
    return " ".join(str(v) for v in (poi.get("name", ""), poi.get("category", ""), poi.get("sub_category", ""), poi.get("address", ""), tags))


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


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
