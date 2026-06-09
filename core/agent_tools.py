"""LangChain tools for the multi-agent route-planning system.

Tools are grouped by the sub-agent that owns them:
- Intent tools  → IntentAgent
- Planning tools → PlanningAgent
- Explanation tools → ExplanationAgent
- Utility tools → available to Supervisor
"""

from __future__ import annotations

import json
import logging
from typing import Any


class _SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)

from langchain_core.tools import tool

from core.explanation import generate_explanation
from core.intent_parser import parse_user_intent
from core.poi_artifact_store import load_poi_store
from core.poi_retriever import retrieve_candidate_pois
from core.replanner import replan_route, understand_replan_intent
from core.route_optimizer import generate_routes
from models.config import settings
from models.schemas import POI, UserProfile
from utils.geo import haversine_distance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared data loading (process-level cache)
# ---------------------------------------------------------------------------

_raw_cache: dict[str, Any] = {}
_city_cache: dict[str, Any] = {}
_validated_profiles_cache: list[dict[str, Any]] | None = None

START_LOCATIONS: dict[str, dict[str, Any]] = {
    "春熙路": {"label": "春熙路", "lat": 30.65708, "lng": 104.08096},
    "太古里": {"label": "太古里", "lat": 30.65398, "lng": 104.08394},
    "宽窄巷子": {"label": "宽窄巷子", "lat": 30.66994, "lng": 104.05958},
    "九眼桥": {"label": "九眼桥", "lat": 30.64057, "lng": 104.09194},
    "三里屯": {"label": "三里屯", "lat": 39.9330, "lng": 116.4550},
    "新天地": {"label": "新天地", "lat": 31.2160, "lng": 121.4750},
    "北京路": {"label": "北京路", "lat": 23.1280, "lng": 113.2690},
    "海岸城": {"label": "海岸城", "lat": 22.5170, "lng": 113.9420},
    "西湖": {"label": "西湖", "lat": 30.2590, "lng": 120.1480},
    "楚河汉街": {"label": "楚河汉街", "lat": 30.5550, "lng": 114.3450},
    "大雁塔": {"label": "大雁塔", "lat": 34.2200, "lng": 108.9620},
    "解放碑": {"label": "解放碑", "lat": 29.5560, "lng": 106.5720},
    "新街口": {"label": "新街口", "lat": 32.0420, "lng": 118.7870},
    "观前街": {"label": "观前街", "lat": 31.3100, "lng": 120.6200},
    "曾厝垵": {"label": "曾厝垵", "lat": 24.4400, "lng": 118.0980},
    "翠湖": {"label": "翠湖", "lat": 25.0500, "lng": 102.7000},
    "星海广场": {"label": "星海广场", "lat": 38.8800, "lng": 121.5700},
    "大东海": {"label": "大东海", "lat": 18.2200, "lng": 109.5150},
    "大研古城": {"label": "大研古城", "lat": 26.8720, "lng": 100.2270},
}
DEFAULT_START_LOCATION = START_LOCATIONS["春熙路"]


def _get_raw_data() -> dict[str, Any]:
    if not _raw_cache:
        _raw_cache["poi_store"] = load_poi_store(settings.pois_file)
        _raw_cache["user_profiles"] = _load_user_profiles()
    return _raw_cache


def _load_user_profiles() -> list[dict[str, Any]]:
    from pathlib import Path
    path = Path(settings.user_profiles_file)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_validated_profiles() -> list[dict[str, Any]]:
    global _validated_profiles_cache
    if _validated_profiles_cache is not None:
        return _validated_profiles_cache
    raw = _get_raw_data()
    _validated_profiles_cache = [
        UserProfile.model_validate(p).model_dump() for p in raw["user_profiles"]
    ]
    return _validated_profiles_cache


def _load_city_data(city: str, scope_key: str = "__all__") -> dict[str, Any]:
    cache_key = f"{city or '__all__'}::{scope_key}"
    if cache_key in _city_cache:
        return _city_cache[cache_key]

    raw = _get_raw_data()
    poi_store = raw["poi_store"]

    if city:
        scoped_pois = poi_store.load_scope(city, scope_key)
        if scoped_pois is None:
            scoped_pois = _filter_pois_by_scope(poi_store.load_city(city), city, scope_key)
    else:
        all_by_city = poi_store.load_all_by_city()
        city_pois = [p for plist in all_by_city.values() for p in plist]
        scoped_pois = _filter_pois_by_scope(city_pois, city, scope_key)

    validated_pois = [POI.model_validate(p).model_dump() for p in scoped_pois]
    validated_profiles = _get_validated_profiles()

    result = {"pois": validated_pois, "user_profiles": validated_profiles}
    _city_cache[cache_key] = result
    return result


def _filter_pois_by_scope(city_pois: list[dict[str, Any]], city: str, scope_key: str) -> list[dict[str, Any]]:
    if scope_key == "__all__":
        return city_pois
    if scope_key.startswith("zone:"):
        zone = scope_key.split(":", 1)[1]
        return [poi for poi in city_pois if str(poi.get("zone") or "") == zone]
    if scope_key.startswith("nearby:"):
        zones = set(scope_key.split(":", 1)[1].split("|"))
        return [poi for poi in city_pois if str(poi.get("zone") or "") in zones]
    if scope_key.startswith("district:"):
        district = scope_key.split(":", 1)[1]
        return [poi for poi in city_pois if str(poi.get("district") or "") == district]
    return city_pois


def _normalize_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal route format to the public RoutePlan schema."""
    normalized = []
    for route in routes:
        stops = []
        for poi_entry in route.get("pois", []):
            stops.append({
                "poi": {
                    "id": poi_entry.get("poi_id", ""),
                    "name": poi_entry.get("name", ""),
                    "category": poi_entry.get("category", ""),
                    "sub_category": "",
                    "lat": poi_entry.get("lat"),
                    "lng": poi_entry.get("lng"),
                    "address": "",
                    "rating": poi_entry.get("rating", 0),
                    "price": poi_entry.get("price", 0),
                    "tags": [],
                    "features": {},
                },
                "arrive_time": poi_entry.get("arrival_time"),
                "leave_time": poi_entry.get("leave_time"),
                "travel_minutes_from_previous": poi_entry.get("travel_from_previous_minutes", 0),
                "score": 0,
                "reason": poi_entry.get("reason"),
            })
        normalized.append({
            "request": {},
            "stops": stops,
            "total_travel_minutes": route.get("total_travel_minutes", 0),
            "total_stay_minutes": route.get("total_duration_minutes", 0),
            "estimated_cost": route.get("total_budget", 0),
            "title": route.get("title", ""),
            "warnings": route.get("warnings", []),
        })
    return normalized


def _resolve_start_location(start_location: Any) -> dict[str, Any]:
    if isinstance(start_location, dict) and start_location.get("lat") is not None:
        return dict(start_location)
    label = str(start_location or DEFAULT_START_LOCATION["label"])
    for name, location in START_LOCATIONS.items():
        if name in label:
            return dict(location)
    return {**dict(DEFAULT_START_LOCATION), "label": label}


def _resolve_scope_key(intent: dict[str, Any], level: str = "zone") -> str:
    from core.zone_catalog import get_zone_aliases, get_zone_metadata
    zone_metadata = get_zone_metadata()
    zone_aliases = get_zone_aliases()

    city = str(intent.get("city") or "")
    zone = str(intent.get("zone") or "").strip()
    if not zone:
        start_location = str(intent.get("start_location") or "")
        query = str(intent.get("query") or "")
        text = f"{start_location} {query}"
        for alias, full_name in zone_aliases.items():
            if alias in text:
                zone = full_name
                break
    if not zone:
        city_zones = zone_metadata.get(city, {})
        for zn in city_zones:
            if zn in str(intent.get("start_location") or "") or zn in str(intent.get("query") or ""):
                zone = zn
                break
    if not zone:
        return "__all__"
    if level == "zone":
        return f"zone:{zone}"
    if level == "nearby":
        nearby = _resolve_nearby_zones(city, zone, zone_metadata)
        return f"nearby:{'|'.join(sorted(nearby))}" if nearby else f"zone:{zone}"
    if level == "district":
        district = str(zone_metadata.get(city, {}).get(zone, {}).get("district") or "")
        return f"district:{district}" if district else "__all__"
    return "__all__"


def _resolve_nearby_zones(city: str, zone_name: str, zone_metadata: dict) -> set[str]:
    if not zone_name:
        return set()
    city_zones = zone_metadata.get(city, {})
    current = city_zones.get(zone_name)
    if not current:
        return set()
    current_lng, current_lat = current["center"]
    distances = []
    for other_zone, metadata in city_zones.items():
        if other_zone == zone_name:
            continue
        lng, lat = metadata["center"]
        distances.append((haversine_distance(current_lat, current_lng, lat, lng), other_zone))
    distances.sort(key=lambda x: x[0])
    return {zone_name, *(z for _, z in distances[:3])}


# ---------------------------------------------------------------------------
# Intent Agent Tools (意图解析)
# ---------------------------------------------------------------------------

@tool
def parse_intent(user_query: str) -> str:
    """解析用户的中文出行需求，提取城市、时间、预算、偏好等结构化信息。

    输入是用户的自然语言描述，例如："下午从春熙路出发，想吃火锅、拍照，不想排队，预算300，晚上9点前结束"。
    返回 JSON 格式的结构化意图。
    """
    intent = parse_user_intent(user_query)
    return json.dumps(intent, ensure_ascii=False)


@tool
def get_user_profile(user_id: str) -> str:
    """根据用户ID获取用户画像信息，包括偏好标签、预算、喜好分类等。

    可用的用户ID有：u001(文艺慢逛型), u002(亲子轻松型), u003(拍照打卡型), u004(夜游朋友局), u005(本地美食优先), u006(文化家庭游)。
    """
    profiles = _get_validated_profiles()
    for profile in profiles:
        if profile.get("id") == user_id:
            return json.dumps(profile, ensure_ascii=False)
    return json.dumps({"error": f"未找到用户画像: {user_id}", "available": [p["id"] for p in profiles]}, ensure_ascii=False)


@tool
def list_supported_cities(query: str = "") -> str:
    """查询系统支持的城市列表和热门商圈。

    可传入城市名查询该城市的热门商圈，不传则返回所有支持的城市。
    """
    from core.intent_parser import CITY_DEFAULT_START_LOCATIONS
    cities = list(CITY_DEFAULT_START_LOCATIONS.keys())
    if query and query in CITY_DEFAULT_START_LOCATIONS:
        return json.dumps({
            "city": query,
            "default_start": CITY_DEFAULT_START_LOCATIONS[query],
        }, ensure_ascii=False)
    return json.dumps({"supported_cities": cities, "total": len(cities)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Planning Agent Tools (路线规划)
# ---------------------------------------------------------------------------

@tool
def retrieve_pois(intent_json: str) -> str:
    """根据用户意图检索候选POI(兴趣点)列表。

    输入是 parse_intent 返回的 JSON 字符串。会自动根据城市和商圈筛选候选地点，
    并按偏好匹配度、预算适配度、质量评分等维度排序，返回最多40个候选POI。
    """
    try:
        intent = json.loads(intent_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "无法解析意图JSON"}, ensure_ascii=False)

    city = str(intent.get("city") or "").strip()
    scope_levels = ["zone", "nearby", "district", "all"]
    candidate_pois = []
    for level in scope_levels:
        scope_key = _resolve_scope_key(intent, level=level)
        data = _load_city_data(city, scope_key=scope_key)
        candidate_pois = retrieve_candidate_pois(intent, {}, data["pois"], limit=40)
        if len(candidate_pois) >= 12 or scope_key == "__all__":
            break

    result = {
        "candidate_count": len(candidate_pois),
        "candidates": [
            {"id": p.get("id"), "name": p.get("name"), "category": p.get("category"),
             "rating": p.get("rating"), "price": p.get("price"), "zone": p.get("zone")}
            for p in candidate_pois[:20]
        ],
    }
    return json.dumps(result, ensure_ascii=False, cls=_SafeEncoder)


@tool
def plan_routes(intent_json: str, user_id: str = "u001") -> str:
    """根据用户意图和用户画像规划3条差异化路线方案。

    输入是 parse_intent 返回的 JSON 字符串和可选的用户ID。
    会自动完成：加载数据 → 检索候选POI → 生成路线 → 评分排序。
    返回包含3条路线的 JSON，每条路线包含地点列表、时间安排、费用明细。
    """
    try:
        intent = json.loads(intent_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "无法解析意图JSON"}, ensure_ascii=False)

    city = str(intent.get("city") or "").strip()
    start_location = _resolve_start_location(intent.get("start_location"))

    scope_levels = ["zone", "nearby", "district", "all"]
    routes = []
    candidate_pois = []
    user_profile = {}
    data = None
    for level in scope_levels:
        scope_key = _resolve_scope_key(intent, level=level)
        data = _load_city_data(city, scope_key=scope_key)
        profiles = data["user_profiles"]
        user_profile = next((p for p in profiles if p.get("id") == user_id), profiles[0] if profiles else {})
        candidate_pois = retrieve_candidate_pois(intent, user_profile, data["pois"], limit=40)
        if len(candidate_pois) >= 12 or scope_key == "__all__":
            break

    raw_routes = generate_routes(
        start_location=start_location,
        candidate_pois=candidate_pois,
        intent=intent,
        user_profile=user_profile,
        top_k=3,
        beam_size=8,
        max_steps=5,
    )

    explanation = generate_explanation(raw_routes, intent)
    routes = _normalize_routes(raw_routes)

    result = {
        "route_count": len(routes),
        "routes": routes,
        "explanation": explanation,
        "intent": intent,
        "meta": {
            "user_id": user_id,
            "poi_count": len(data["pois"]) if data else 0,
            "candidate_count": len(candidate_pois),
        },
    }
    return json.dumps(result, ensure_ascii=False, cls=_SafeEncoder)


@tool
def replan_routes(intent_json: str, feedback: str, user_id: str = "u001") -> str:
    """根据用户反馈重新规划路线。

    输入是之前的意图JSON、用户反馈文本（如"便宜点"、"少走路"、"换成小吃"等）和用户ID。
    会理解反馈内容，更新意图参数，重新生成路线。
    """
    try:
        previous_intent = json.loads(intent_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "无法解析意图JSON"}, ensure_ascii=False)

    city = str(previous_intent.get("city") or "").strip()
    updated_intent, changes = understand_replan_intent(previous_intent, feedback)

    scope_levels = ["zone", "nearby", "district", "all"]
    result_data = None
    data = None
    for level in scope_levels:
        scope_key = _resolve_scope_key(updated_intent, level=level)
        data = _load_city_data(city, scope_key=scope_key)
        profiles = data["user_profiles"]
        user_profile = next((p for p in profiles if p.get("id") == user_id), profiles[0] if profiles else {})
        result_data = replan_route(
            previous_intent=previous_intent,
            user_feedback=feedback,
            user_profile=user_profile,
            pois=data["pois"],
        )
        if result_data.get("candidate_count", 0) >= 12 or scope_key == "__all__":
            break

    raw_routes = result_data.get("routes", [])
    explanation = generate_explanation(raw_routes, result_data.get("updated_intent", updated_intent))
    routes = _normalize_routes(raw_routes)

    return json.dumps({
        "intent": result_data.get("updated_intent", updated_intent),
        "routes": routes,
        "explanation": explanation,
        "changes": result_data.get("changes", []),
        "warnings": result_data.get("warnings", []),
    }, ensure_ascii=False, cls=_SafeEncoder)


# ---------------------------------------------------------------------------
# Explanation Agent Tools (解释生成)
# ---------------------------------------------------------------------------

@tool
def explain_routes(routes_json: str, intent_json: str) -> str:
    """为已生成的路线方案生成友好的中文解释。

    输入是路线方案JSON和用户意图JSON。
    返回自然语言解释，包含路线亮点、行程安排、预算分析、风险提示等。
    """
    try:
        routes_data = json.loads(routes_json)
        intent = json.loads(intent_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "无法解析JSON"}, ensure_ascii=False)

    routes = routes_data if isinstance(routes_data, list) else routes_data.get("routes", [])
    explanation = generate_explanation(routes, intent)
    return json.dumps({"explanation": explanation}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool groups for each sub-agent
# ---------------------------------------------------------------------------

INTENT_TOOLS = [parse_intent, get_user_profile, list_supported_cities]
PLANNING_TOOLS = [retrieve_pois, plan_routes, replan_routes]
EXPLANATION_TOOLS = [explain_routes]

ALL_TOOLS = INTENT_TOOLS + PLANNING_TOOLS + EXPLANATION_TOOLS
