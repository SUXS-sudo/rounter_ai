from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from app import PlanRequest, ReplanRequest, _load_city_data, _resolve_scope_key, get_enriched_pois, plan_route, replan
from core.poi_artifact_store import load_poi_store


def _poi_by_id() -> dict[str, dict]:
    return {poi["id"]: poi for poi in get_enriched_pois()}


def _route_ids(route: dict) -> list[str]:
    return [step["poi_id"] for step in route["pois"]]


def _poi_text(poi: dict) -> str:
    return " ".join(
        str(value)
        for value in (
            poi.get("name", ""),
            poi.get("category", ""),
            poi.get("sub_category", ""),
            " ".join(str(tag) for tag in poi.get("tags", [])),
        )
    ).lower()


def _route_has_hotpot(route: dict, pois: dict[str, dict]) -> bool:
    return any(
        "hotpot" in _poi_text(pois[poi_id]) or "火锅" in _poi_text(pois[poi_id])
        for poi_id in _route_ids(route)
    )


def _route_has_snack(route: dict, pois: dict[str, dict]) -> bool:
    return any(
        "snack" in _poi_text(pois[poi_id]) or "小吃" in _poi_text(pois[poi_id])
        for poi_id in _route_ids(route)
    )


def _avg_feature(route: dict, pois: dict[str, dict], feature: str) -> float:
    values = [pois[poi_id]["features"][feature] for poi_id in _route_ids(route)]
    return sum(values) / len(values)


def _plan(query: str) -> dict:
    return plan_route(PlanRequest(user_id="u001", query=query))


def test_store_zones_are_all_recognizable() -> None:
    root = Path(__file__).resolve().parents[1]
    sqlite_store = load_poi_store(root / 'data' / 'poi_data_500k.db')

    failures = []
    for city in sqlite_store.city_counts():
        city_pois = sqlite_store.load_city(city)
        zones = sorted({str(poi.get('zone') or '') for poi in city_pois if str(poi.get('zone') or '').strip()})
        for zone in zones:
            short = zone[:-2] if zone.endswith('商圈') else zone
            intent = {
                'city': city,
                'start_location': short,
                'query': f'在{city}{short}附近逛逛',
                'zone': '',
            }
            scope_key = _resolve_scope_key(intent, level='zone')
            if scope_key != f'zone:{zone}':
                failures.append((city, zone, scope_key))
    assert not failures



def test_sqlite_store_matches_monolithic_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    monolithic = load_poi_store(root / 'data' / 'poi_data_500k.json')
    sqlite_store = load_poi_store(root / 'data' / 'poi_data_500k.db')

    assert sqlite_store.city_counts() == monolithic.city_counts()
    sample_city = next(iter(sqlite_store.city_counts()))
    assert len(sqlite_store.load_city(sample_city)) == len(monolithic.load_city(sample_city))



def test_xiamen_zengcuoan_uses_zone_scope() -> None:
    intent = {
        'city': '厦门',
        'start_location': '曾厝垵',
        'query': '在厦门曾厝垵附近逛逛，想拍照喝奶茶',
        'zone': '',
    }
    scope_key = _resolve_scope_key(intent, level='zone')
    data = _load_city_data('厦门', scope_key=scope_key)

    assert scope_key == 'zone:曾厝垵商圈'
    assert data.pois
    assert all(str(poi.get('zone') or '') == '曾厝垵商圈' for poi in data.pois)



def test_hotpot_preference_route_contains_hotpot() -> None:
    pois = _poi_by_id()
    result = _plan("我周六下午从春熙路出发，想吃火锅、拍照，不想排队，预算300，晚上9点前结束")

    assert "火锅" in result["intent"]["preferences"]
    assert result["routes"]
    assert _route_has_hotpot(result["routes"][0], pois)


def test_replan_no_hotpot_switch_to_snack() -> None:
    pois = _poi_by_id()
    original = _plan("下午从春熙路出发，想吃火锅、拍照，不想排队，预算300，晚上9点前结束")
    result = replan(
        ReplanRequest(
            user_id="u001",
            previous_intent=original["intent"],
            feedback="不要火锅了，换成小吃",
        )
    )

    assert "火锅" not in result["intent"]["preferences"]
    assert "火锅" in result["intent"]["avoid"]
    assert "小吃" in result["intent"]["preferences"]
    assert result["routes"]
    assert not any(_route_has_hotpot(route, pois) for route in result["routes"])
    assert any(_route_has_snack(route, pois) for route in result["routes"]) or result["warnings"]


def test_rainy_indoor_route_has_higher_indoor_score() -> None:
    pois = _poi_by_id()
    rainy = _plan("今天下雨，我想下午在春熙路附近玩，不想排队，预算300，尽量安排室内")
    normal = _plan("今天下午在春熙路附近玩，不想排队，预算300")

    assert "室内" in rainy["intent"]["preferences"]
    assert _avg_feature(rainy["routes"][0], pois, "indoor") > _avg_feature(normal["routes"][0], pois, "indoor")


def test_low_walk_route_is_more_compact_than_general_food_route() -> None:
    compact = _plan("下午从春熙路出发，想吃点好吃的，少走路，别太累，预算300")
    normal = _plan("下午从春熙路出发，想吃点好吃的，预算300")

    assert "少走路" in compact["intent"]["preferences"]
    compact_detail = compact["routes"][0]["score_detail"]
    normal_detail = normal["routes"][0]["score_detail"]
    assert (
        compact["routes"][0]["total_travel_minutes"] <= normal["routes"][0]["total_travel_minutes"]
        or compact_detail["compact_score"] >= normal_detail["compact_score"]
    )


def test_diverse_routes_are_not_identical() -> None:
    result = _plan("下午从春熙路出发，想喝咖啡拍照，预算300")
    route_sets = [tuple(_route_ids(route)) for route in result["routes"]]

    assert len(route_sets) >= 3
    assert len(set(route_sets)) == len(route_sets)


def test_bookstore_intent_is_parsed_and_reflected_in_routes() -> None:
    pois = _poi_by_id()
    result = _plan("在上海，周末下午从新天地出发，想喝咖啡、逛书店、拍照，节奏轻松一点，预算300")

    assert result["intent"]["city"] == "上海"
    assert result["intent"]["start_location"] == "新天地"
    assert "咖啡" in result["intent"]["preferences"]
    assert "拍照" in result["intent"]["preferences"]
    assert "书店" in result["intent"]["preferences"]
    assert "书店" in result["intent"]["must_satisfy_preferences"]
    assert result["routes"]
    assert any(
        any(any(token in _poi_text(pois[poi_id]) for token in ("bookstore", "书店", "图书馆", "library", "阅读空间")) for poi_id in _route_ids(route))
        for route in result["routes"]
    ) or result.get("warnings")


def test_weak_bookstore_phrase_is_not_promoted_to_must_satisfy() -> None:
    result = _plan("在上海，从新天地出发，顺便看看书店也行，预算300")

    assert "书店" in result["intent"]["preferences"]
    assert "书店" not in result["intent"].get("must_satisfy_preferences", [])


def test_replan_quiet_feedback_maps_to_quiet_preference() -> None:
    original = _plan("在厦门，下午从曾厝垵出发，想拍照、喝奶茶、慢慢逛，预算220")
    replanned = replan(
        ReplanRequest(
            user_id="u001",
            previous_intent=original["intent"],
            feedback="换个更安静一点、适合坐坐的地方",
        )
    )

    assert "安静" in replanned["intent"]["preferences"]
    assert replanned["routes"] or replanned["warnings"]



def test_replan_dating_feedback_maps_to_dating_scenario() -> None:
    original = _plan("在厦门，下午从曾厝垵出发，想拍照、喝奶茶、慢慢逛，预算220")
    replanned = replan(
        ReplanRequest(
            user_id="u001",
            previous_intent=original["intent"],
            feedback="帮我改成更适合约会，别太赶",
        )
    )

    assert replanned["intent"]["scenario"] == "dating"
    assert replanned["intent"]["people_count"] >= 2
    assert "少走路" in replanned["intent"]["preferences"]



def test_replan_local_feedback_maps_to_less_queueing() -> None:
    original = _plan("在厦门，下午从曾厝垵出发，想拍照、喝奶茶、慢慢逛，预算220")
    replanned = replan(
        ReplanRequest(
            user_id="u001",
            previous_intent=original["intent"],
            feedback="不要游客太多的地方，想更本地一点",
        )
    )

    assert "少排队" in replanned["intent"]["preferences"]
    assert "排队" in replanned["intent"]["avoid"]



def test_low_budget_replan_reduces_budget_or_warns() -> None:
    original = _plan("下午从春熙路出发，想吃火锅、拍照，预算300，晚上9点前结束")
    replanned = replan(
        ReplanRequest(
            user_id="u001",
            previous_intent=original["intent"],
            feedback="太贵了，控制在150以内",
        )
    )

    assert replanned["intent"]["budget"] == 150
    assert (
        100 <= replanned["routes"][0]["total_budget"] <= 170
        or replanned["routes"][0]["warnings"]
        or replanned["warnings"]
    )
    assert (
        replanned["routes"][0]["total_budget"] <= original["routes"][0]["total_budget"]
        or replanned["routes"][0]["warnings"]
        or replanned["warnings"]
    )
