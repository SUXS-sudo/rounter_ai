"""FastAPI entrypoint for the local intelligent route planner."""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from utils.geo import haversine_distance

from core.explanation import generate_explanation, generate_explanation_stream
from core.intent_parser import parse_user_intent
from core.poi_artifact_store import PoiStore, load_poi_store
from core.poi_retriever import retrieve_candidate_pois
from core.replanner import replan_route, replan_route_for_intent, understand_replan_intent
from core.route_optimizer import generate_routes
from core.zone_catalog import get_zone_aliases, get_zone_metadata
from models.config import settings
from models.schemas import POI, UserProfile


BASE_DIR = Path(__file__).resolve().parent


app = FastAPI(title=settings.app_name, version=settings.app_version)


def _sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event."""

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_raw_cache: dict[str, Any] = {}
_validated_profiles_cache: list[dict[str, Any]] | None = None


def _get_raw_data() -> dict[str, Any]:
    if not _raw_cache:
        _raw_cache["poi_store"] = load_poi_store(settings.pois_file)
        _raw_cache["pois"] = _raw_cache["poi_store"]
        _raw_cache["user_profiles"] = load_json_list(settings.user_profiles_file, "user_profiles")
    return _raw_cache


START_LOCATIONS = {
    "春熙路": {"label": "春熙路", "lat": 30.65708, "lng": 104.08096},
    "太古里": {"label": "太古里", "lat": 30.65398, "lng": 104.08394},
    "宽窄巷子": {"label": "宽窄巷子", "lat": 30.66994, "lng": 104.05958},
    "九眼桥": {"label": "九眼桥", "lat": 30.64057, "lng": 104.09194},
    "大慈寺": {"label": "大慈寺", "lat": 30.65461, "lng": 104.08511},
    "安顺廊桥": {"label": "安顺廊桥", "lat": 30.64202, "lng": 104.08856},
    "望江楼公园": {"label": "望江楼公园", "lat": 30.63582, "lng": 104.09597},
    "三里屯": {"label": "三里屯", "lat": 39.9330, "lng": 116.4550},
    "中关村": {"label": "中关村", "lat": 39.9820, "lng": 116.3160},
    "王府井": {"label": "王府井", "lat": 39.9140, "lng": 116.4140},
    "新天地": {"label": "新天地", "lat": 31.2160, "lng": 121.4750},
    "陆家嘴": {"label": "陆家嘴", "lat": 31.2390, "lng": 121.4990},
    "南京路": {"label": "南京路", "lat": 31.2350, "lng": 121.4750},
    "北京路": {"label": "北京路", "lat": 23.1280, "lng": 113.2690},
    "天河城": {"label": "天河城", "lat": 23.1350, "lng": 113.3280},
    "珠江新城": {"label": "珠江新城", "lat": 23.1180, "lng": 113.3200},
    "海岸城": {"label": "海岸城", "lat": 22.5170, "lng": 113.9420},
    "华强北": {"label": "华强北", "lat": 22.5460, "lng": 114.0880},
    "福田CBD": {"label": "福田CBD", "lat": 22.5350, "lng": 114.0550},
    "西湖": {"label": "西湖", "lat": 30.2590, "lng": 120.1480},
    "湖滨": {"label": "湖滨", "lat": 30.2500, "lng": 120.1600},
    "武林广场": {"label": "武林广场", "lat": 30.2750, "lng": 120.1650},
    "楚河汉街": {"label": "楚河汉街", "lat": 30.5550, "lng": 114.3450},
    "江汉路": {"label": "江汉路", "lat": 30.5830, "lng": 114.2850},
    "大雁塔": {"label": "大雁塔", "lat": 34.2200, "lng": 108.9620},
    "钟楼": {"label": "钟楼", "lat": 34.2650, "lng": 108.9400},
    "解放碑": {"label": "解放碑", "lat": 29.5560, "lng": 106.5720},
    "观音桥": {"label": "观音桥", "lat": 29.5750, "lng": 106.5480},
    "新街口": {"label": "新街口", "lat": 32.0420, "lng": 118.7870},
    "夫子庙": {"label": "夫子庙", "lat": 32.0180, "lng": 118.7900},
    "观前街": {"label": "观前街", "lat": 31.3100, "lng": 120.6200},
    "金鸡湖": {"label": "金鸡湖", "lat": 31.3100, "lng": 120.6800},
    "曾厝垵": {"label": "曾厝垵", "lat": 24.4400, "lng": 118.0980},
    "中山路": {"label": "中山路", "lat": 24.4500, "lng": 118.0750},
    "翠湖": {"label": "翠湖", "lat": 25.0500, "lng": 102.7000},
    "星海广场": {"label": "星海广场", "lat": 38.8800, "lng": 121.5700},
    "大东海": {"label": "大东海", "lat": 18.2200, "lng": 109.5150},
    "大研古城": {"label": "大研古城", "lat": 26.8720, "lng": 100.2270},
}
DEFAULT_START_LOCATION = START_LOCATIONS["春熙路"]


class PlanRequest(BaseModel):
    """Request model for creating a route plan from a natural-language query."""

    user_id: str = Field(..., examples=["u001"])
    query: str = Field(
        ...,
        min_length=1,
        examples=["我周六2.下午从春熙路出发，可以步行，打车和坐地铁，0.5公里以内步行，大于1公里可以选择打车和坐地铁，，想吃火锅、拍照，不想排队，预算600，晚上9点前结束"],
    )


class ReplanRequest(BaseModel):
    """Request model for replanning from previous intent and user feedback."""

    user_id: str = Field(..., examples=["u001"])
    previous_intent: dict[str, Any]
    feedback: str = Field(..., min_length=1, examples=["从早上8.出发，想吃火锅、拍照，不想排队，预算600，晚上9点前结束"])


class LocalData(BaseModel):
    """Validated local data bundle used by the planning pipeline."""

    pois: list[dict[str, Any]]
    user_profiles: list[dict[str, Any]]


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Return API errors in a stable JSON envelope."""

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected failures and expose a concise error to clients."""

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/start")
def start() -> dict[str, str]:
    """Start check endpoint."""

    return {"status": "ok"}


@app.get("/profiles")
def list_profiles() -> list[dict[str, Any]]:
    """Return all available user profiles."""

    raw = _get_raw_data()
    return raw["user_profiles"]


@app.get("/pois")
def list_pois() -> dict[str, Any]:
    """Return city-level POI summary."""

    raw = _get_raw_data()
    pois_dict: dict[str, list[dict[str, Any]]] = raw["pois"]
    city_counts = {city: len(pois) for city, pois in pois_dict.items()}
    total = sum(city_counts.values())
    return {"total": total, "cities": city_counts}


def get_enriched_pois(city: str = "") -> list[dict[str, Any]]:
    """Return the enriched POI list for testing / CLI use."""

    data = _load_city_data(city)
    return data.pois


@app.post("/plan")
def plan_route(request: PlanRequest) -> dict[str, Any]:
    """Create route plans from a user query."""

    t0 = time.time()
    intent = parse_user_intent(request.query)
    city = str(intent.get("city") or "").strip()

    scope_levels = ["zone", "nearby", "district", "all"]
    data = None
    candidate_pois: list[dict[str, Any]] = []
    user_profile: dict[str, Any] | None = None
    start_location = resolve_start_location(intent.get("start_location"))
    for level in scope_levels:
        scope_key = _resolve_scope_key(intent, level=level)
        data = _load_city_data(city, scope_key=scope_key)
        user_profile = get_user_profile(data.user_profiles, request.user_id)
        candidate_pois = retrieve_candidate_pois(intent, user_profile, data.pois, limit=40)
        if len(candidate_pois) >= 12 or scope_key == "__all__":
            break
    routes = generate_routes(
        start_location=start_location,
        candidate_pois=candidate_pois,
        intent=intent,
        user_profile=user_profile,
        top_k=3,
        beam_size=8,
        max_steps=5,
    )
    explanation = generate_explanation(routes, intent)
    elapsed = round(time.time() - t0, 2)

    return {
        "intent": intent,
        "routes": routes,
        "explanation": explanation,
        "elapsed": elapsed,
        "meta": {
            "user_id": request.user_id,
            "poi_count": len(data.pois),
            "user_profile_count": len(data.user_profiles),
            "candidate_count": len(candidate_pois),
        },
    }


@app.post("/plan/stream")
def plan_route_stream(request: PlanRequest) -> StreamingResponse:
    """Create route plans with streaming SSE output."""

    def event_stream() -> Generator[str, None, None]:
        t0 = time.time()

        yield _sse_event("progress", {"status": "正在解析意图..."})
        intent = parse_user_intent(request.query)
        city = str(intent.get("city") or "").strip()
        yield _sse_event("intent", intent)

        yield _sse_event("progress", {"status": "正在加载数据..."})
        scope_levels = ["zone", "nearby", "district", "all"]
        data = None
        candidate_pois: list[dict[str, Any]] = []
        user_profile: dict[str, Any] | None = None
        start_location = resolve_start_location(intent.get("start_location"))

        yield _sse_event("progress", {"status": "正在检索候选地点..."})
        for level in scope_levels:
            scope_key = _resolve_scope_key(intent, level=level)
            data = _load_city_data(city, scope_key=scope_key)
            user_profile = get_user_profile(data.user_profiles, request.user_id)
            candidate_pois = retrieve_candidate_pois(intent, user_profile, data.pois, limit=40)
            if len(candidate_pois) >= 12 or scope_key == "__all__":
                break

        yield _sse_event("progress", {"status": "正在规划路线..."})
        routes = generate_routes(
            start_location=start_location,
            candidate_pois=candidate_pois,
            intent=intent,
            user_profile=user_profile,
            top_k=3,
            beam_size=8,
            max_steps=5,
        )
        yield _sse_event("routes", routes)

        yield _sse_event("progress", {"status": "正在生成路线解释..."})
        for chunk in generate_explanation_stream(routes, intent):
            yield _sse_event("explanation_chunk", {"text": chunk})

        elapsed = round(time.time() - t0, 2)
        yield _sse_event("done", {
            "elapsed": elapsed,
            "meta": {
                "user_id": request.user_id,
                "poi_count": len(data.pois),
                "user_profile_count": len(data.user_profiles),
                "candidate_count": len(candidate_pois),
            },
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/replan")
def replan(request: ReplanRequest) -> dict[str, Any]:
    """Regenerate route plans from prior intent and user feedback."""

    t0 = time.time()
    updated_intent, changes = understand_replan_intent(request.previous_intent, request.feedback)
    city = str(updated_intent.get("city") or "").strip()

    scope_levels = ["zone", "nearby", "district", "all"]
    data = None
    user_profile: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    for level in scope_levels:
        scope_key = _resolve_scope_key(updated_intent, level=level)
        data = _load_city_data(city, scope_key=scope_key)
        user_profile = get_user_profile(data.user_profiles, request.user_id)
        result = replan_route_for_intent(
            updated_intent=updated_intent,
            user_profile=user_profile,
            pois=data.pois,
            changes=changes,
        )
        if result.get("candidate_count", 0) >= 12 or scope_key == "__all__":
            break

    updated_intent = result["updated_intent"]
    routes = result["routes"]
    explanation = generate_explanation(routes, updated_intent)
    elapsed = round(time.time() - t0, 2)

    return {
        "intent": updated_intent,
        "routes": routes,
        "explanation": explanation,
        "elapsed": elapsed,
        "changes": result.get("changes", []),
        "warnings": result.get("warnings", []),
        "meta": {
            "user_id": request.user_id,
            "poi_count": len(data.pois),
            "user_profile_count": len(data.user_profiles),
            "candidate_count": result.get("candidate_count", 0),
        },
    }


@app.post("/replan/stream")
def replan_stream(request: ReplanRequest) -> StreamingResponse:
    """Regenerate route plans with streaming SSE output."""

    def event_stream() -> Generator[str, None, None]:
        t0 = time.time()

        yield _sse_event("progress", {"status": "正在理解反馈..."})
        updated_intent, changes = understand_replan_intent(request.previous_intent, request.feedback)
        city = str(updated_intent.get("city") or "").strip()
        yield _sse_event("intent", updated_intent)

        yield _sse_event("progress", {"status": "正在加载数据..."})
        scope_levels = ["zone", "nearby", "district", "all"]
        data = None
        user_profile: dict[str, Any] | None = None
        result: dict[str, Any] | None = None

        yield _sse_event("progress", {"status": "正在重新规划路线..."})
        for level in scope_levels:
            scope_key = _resolve_scope_key(updated_intent, level=level)
            data = _load_city_data(city, scope_key=scope_key)
            user_profile = get_user_profile(data.user_profiles, request.user_id)
            result = replan_route_for_intent(
                updated_intent=updated_intent,
                user_profile=user_profile,
                pois=data.pois,
                changes=changes,
            )
            if result.get("candidate_count", 0) >= 12 or scope_key == "__all__":
                break

        updated_intent = result["updated_intent"]
        routes = result["routes"]

        yield _sse_event("routes", routes)

        if result.get("changes"):
            yield _sse_event("changes", result["changes"])
        if result.get("warnings"):
            yield _sse_event("warnings", result["warnings"])

        yield _sse_event("progress", {"status": "正在生成路线解释..."})
        for chunk in generate_explanation_stream(routes, updated_intent):
            yield _sse_event("explanation_chunk", {"text": chunk})

        elapsed = round(time.time() - t0, 2)
        yield _sse_event("done", {
            "elapsed": elapsed,
            "meta": {
                "user_id": request.user_id,
                "poi_count": len(data.pois),
                "user_profile_count": len(data.user_profiles),
                "candidate_count": result.get("candidate_count", 0),
            },
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


_city_cache: dict[str, LocalData] = {}
_validated_city_pois_cache: dict[str, list[dict[str, Any]]] = {}

ZONE_METADATA = get_zone_metadata()
ZONE_ALIASES = get_zone_aliases()


def _get_validated_profiles() -> list[dict[str, Any]]:
    global _validated_profiles_cache
    if _validated_profiles_cache is not None:
        return _validated_profiles_cache

    raw = _get_raw_data()
    try:
        _validated_profiles_cache = [UserProfile.model_validate(p).model_dump() for p in raw["user_profiles"]]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local data validation failed: {exc}") from exc
    return _validated_profiles_cache


def _get_validated_city_pois(city: str, poi_store: PoiStore, scope_key: str = "__all__") -> list[dict[str, Any]]:
    cache_key = f"{city or '__all__'}::{scope_key}"
    if cache_key in _validated_city_pois_cache:
        return _validated_city_pois_cache[cache_key]

    if city:
        scoped_pois = poi_store.load_scope(city, scope_key)
        if scoped_pois is None:
            scoped_pois = _filter_pois_by_scope(poi_store.load_city(city), city, scope_key)
    else:
        all_by_city = poi_store.load_all_by_city()
        city_pois = [p for plist in all_by_city.values() for p in plist]
        scoped_pois = _filter_pois_by_scope(city_pois, city, scope_key)

    try:
        validated_pois = [POI.model_validate(p).model_dump() for p in scoped_pois]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local data validation failed: {exc}") from exc

    _validated_city_pois_cache[cache_key] = validated_pois
    return validated_pois


def _load_city_data(city: str, scope_key: str = "__all__") -> LocalData:
    """Load and validate POI data with process-level hot caching and zone-first scopes."""

    cache_key = f"{city or '__all__'}::{scope_key}"
    if cache_key in _city_cache:
        return _city_cache[cache_key]

    raw = _get_raw_data()
    poi_store: PoiStore = raw["poi_store"]

    validated_pois = _get_validated_city_pois(city, poi_store, scope_key=scope_key)
    validated_profiles = _get_validated_profiles()

    result = LocalData(
        pois=validated_pois,
        user_profiles=validated_profiles,
    )
    _city_cache[cache_key] = result
    return result


def load_json_list(path: Path, display_name: str) -> list[dict[str, Any]]:
    """Load a JSON array file and return a list of dictionaries."""

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Local data file not found: {display_name}") from exc
    except JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Local data file is invalid JSON: {display_name}") from exc

    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail=f"Local data file must contain a JSON array: {display_name}")
    if not all(isinstance(item, dict) for item in payload):
        raise HTTPException(status_code=500, detail=f"Local data file must contain objects only: {display_name}")
    return payload


def load_json_dict(path: Path, display_name: str) -> dict[str, list[dict[str, Any]]]:
    """Load a JSON file keyed by city, returning a dict of lists."""

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Local data file not found: {display_name}") from exc
    except JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Local data file is invalid JSON: {display_name}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"Local data file must contain a JSON object: {display_name}")
    return payload


def _resolve_scope_key(intent: dict[str, Any], level: str = "zone") -> str:
    city = str(intent.get("city") or "")
    zone = str(intent.get("zone") or "").strip()
    if not zone:
        start_location = str(intent.get("start_location") or "")
        query = str(intent.get("query") or "")
        text = f"{start_location} {query}"
        for alias, full_name in ZONE_ALIASES.items():
            if alias in text:
                zone = full_name
                break
    if not zone:
        direct_zone = _match_known_zone_name(city, intent)
        if direct_zone:
            zone = direct_zone
    if not zone:
        return "__all__"
    if level == "zone":
        return f"zone:{zone}"
    if level == "nearby":
        nearby = sorted(_resolve_nearby_zones(city, zone))
        return f"nearby:{'|'.join(nearby)}" if nearby else f"zone:{zone}"
    if level == "district":
        district = _resolve_zone_district(city, zone)
        return f"district:{district}" if district else "__all__"
    return "__all__"


def _match_known_zone_name(city: str, intent: dict[str, Any]) -> str:
    city_zones = ZONE_METADATA.get(city, {})
    if not city_zones:
        return ""
    candidates = [
        str(intent.get("zone") or "").strip(),
        str(intent.get("start_location") or "").strip(),
        str(intent.get("query") or "").strip(),
    ]
    for text in candidates:
        if not text:
            continue
        for zone_name in city_zones:
            if zone_name in text:
                return zone_name
            if zone_name.endswith("商圈") and zone_name[:-2] in text:
                return zone_name
    return ""



def _resolve_nearby_zones(city: str, zone_name: str) -> set[str]:
    if not zone_name:
        return set()
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
    return {zone_name, *(zone for _, zone in distances[:3])}


def _resolve_zone_district(city: str, zone_name: str) -> str:
    return str(ZONE_METADATA.get(city, {}).get(zone_name, {}).get("district") or "")


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


def get_user_profile(user_profiles: list[dict[str, Any]], user_id: str) -> dict[str, Any]:
    """Find a user profile by ID or return a clear 404 error."""

    for profile in user_profiles:
        if profile.get("id") == user_id:
            return profile
    raise HTTPException(status_code=404, detail=f"User profile not found: {user_id}")


def resolve_start_location(start_location: Any) -> dict[str, Any]:
    """Resolve an intent start location into coordinates."""

    if isinstance(start_location, dict) and start_location.get("lat") is not None and start_location.get("lng") is not None:
        return dict(start_location)

    label = str(start_location or DEFAULT_START_LOCATION["label"])
    for name, location in START_LOCATIONS.items():
        if name in label:
            return dict(location)
    return {**dict(DEFAULT_START_LOCATION), "label": label}
