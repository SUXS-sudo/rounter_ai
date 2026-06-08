"""Command-line interface for testing the route planner."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import (
    PlanRequest,
    ReplanRequest,
    _load_city_data,
    _resolve_scope_key,
    get_user_profile,
    list_pois,
    plan_route,
    replan,
    resolve_start_location,
)
from core.explanation import generate_explanation_stream
from core.intent_parser import parse_user_intent
from core.poi_retriever import retrieve_candidate_pois
from core.replanner import replan_route as replan_route_core
from core.route_optimizer import generate_routes


INTENT_CACHE = Path(__file__).resolve().parent / ".last_intent.json"
PROFILES_FILE = Path(__file__).resolve().parent / "data" / "user_profiles.json"


def print_json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_plan(query: str, user_id: str = "u001", stream: bool = False) -> None:
    if stream:
        _cmd_plan_stream(query, user_id)
    else:
        result = plan_route(PlanRequest(user_id=user_id, query=query))
        with INTENT_CACHE.open("w", encoding="utf-8") as f:
            json.dump({"intent": result["intent"], "user_id": user_id}, f, ensure_ascii=False)
        print_json(result)


def _cmd_plan_stream(query: str, user_id: str) -> None:
    t0 = time.time()

    print("⏳ 正在解析意图...", flush=True)
    intent = parse_user_intent(query)
    city = str(intent.get("city") or "").strip()
    print(f"✅ 意图解析完成: 城市={intent.get('city')}, 偏好={intent.get('preferences')}", flush=True)

    print("⏳ 正在加载数据...", flush=True)
    scope_levels = ["zone", "nearby", "district", "all"]
    data = None
    candidate_pois: list[dict] = []
    user_profile = None
    start_location = resolve_start_location(intent.get("start_location"))
    final_scope_key = "__all__"
    print("⏳ 正在检索候选地点...", flush=True)
    for level in scope_levels:
        scope_key = _resolve_scope_key(intent, level=level)
        data = _load_city_data(city, scope_key=scope_key)
        user_profile = get_user_profile(data.user_profiles, user_id)
        candidate_pois = retrieve_candidate_pois(intent, user_profile, data.pois, limit=40)
        final_scope_key = scope_key
        if len(candidate_pois) >= 12 or scope_key == "__all__":
            break
    print(f"✅ 数据加载完成: scope={final_scope_key}, {len(data.pois)}个地点", flush=True)
    print(f"✅ 检索到 {len(candidate_pois)} 个候选地点", flush=True)

    print("⏳ 正在规划路线...", flush=True)
    routes = generate_routes(
        start_location=start_location,
        candidate_pois=candidate_pois,
        intent=intent,
        user_profile=user_profile,
        top_k=3,
        beam_size=8,
        max_steps=5,
    )
    print(f"✅ 生成了 {len(routes)} 条路线", flush=True)

    with INTENT_CACHE.open("w", encoding="utf-8") as f:
        json.dump({"intent": intent, "user_id": user_id}, f, ensure_ascii=False)

    print("\n📝 路线解释（流式输出）:", flush=True)
    print("-" * 50, flush=True)
    for chunk in generate_explanation_stream(routes, intent):
        print(chunk, end="", flush=True)
    print("\n" + "-" * 50, flush=True)

    elapsed = round(time.time() - t0, 2)
    print(f"\n⏱️  总耗时: {elapsed}秒", flush=True)


def cmd_replan(previous_intent: dict, feedback: str, user_id: str = "u001", stream: bool = False) -> None:
    if stream:
        _cmd_replan_stream(previous_intent, feedback, user_id)
    else:
        result = replan(ReplanRequest(user_id=user_id, previous_intent=previous_intent, feedback=feedback))
        with INTENT_CACHE.open("w", encoding="utf-8") as f:
            json.dump({"intent": result["intent"], "user_id": user_id}, f, ensure_ascii=False)
        print_json(result)


def _cmd_replan_stream(previous_intent: dict, feedback: str, user_id: str) -> None:
    t0 = time.time()

    print("⏳ 正在理解反馈...", flush=True)
    city = str(previous_intent.get("city") or "").strip()

    print("⏳ 正在加载数据...", flush=True)
    scope_key = _resolve_scope_key(previous_intent, level="zone")
    data = _load_city_data(city, scope_key=scope_key)
    user_profile = get_user_profile(data.user_profiles, user_id)

    print(f"✅ 数据加载完成: scope={scope_key}, {len(data.pois)}个地点", flush=True)
    print("⏳ 正在重新规划路线...", flush=True)
    result = replan_route_core(
        previous_intent=previous_intent,
        user_feedback=feedback,
        user_profile=user_profile,
        pois=data.pois,
    )
    updated_intent = result["updated_intent"]
    routes = result["routes"]
    print(f"✅ 生成了 {len(routes)} 条路线", flush=True)

    if result.get("changes"):
        print("📝 变更:", flush=True)
        for change in result["changes"]:
            print(f"  - {change}", flush=True)

    if result.get("warnings"):
        print("⚠️  警告:", flush=True)
        for warning in result["warnings"]:
            print(f"  - {warning}", flush=True)

    with INTENT_CACHE.open("w", encoding="utf-8") as f:
        json.dump({"intent": updated_intent, "user_id": user_id}, f, ensure_ascii=False)

    print("\n📝 路线解释（流式输出）:", flush=True)
    print("-" * 50, flush=True)
    for chunk in generate_explanation_stream(routes, updated_intent):
        print(chunk, end="", flush=True)
    print("\n" + "-" * 50, flush=True)

    elapsed = round(time.time() - t0, 2)
    print(f"\n⏱️  总耗时: {elapsed}秒", flush=True)


def cmd_pois() -> None:
    print_json(list_pois())


def cmd_profiles() -> None:
    if not PROFILES_FILE.exists():
        print("错误：user_profiles.json 不存在", file=sys.stderr)
        sys.exit(1)
    with PROFILES_FILE.open(encoding="utf-8") as f:
        profiles = json.load(f)
    for p in profiles:
        print(f"  {p['id']}  {p['name']:　<8}  预算{p['budget_per_day']}元/天  偏好: {', '.join(p['preferred_tags'][:4])}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="智能路线规划系统 - 命令行工具")
    sub = parser.add_subparsers(dest="command")

    p_plan = sub.add_parser("plan", help="根据自然语言描述生成路线")
    p_plan.add_argument("query", help="用户查询，例如: 下午从春熙路出发，想吃火锅，预算300")
    p_plan.add_argument("--user", default="u001", help="用户ID (默认: u001)")
    p_plan.add_argument("--no-stream", action="store_true", help="禁用流式输出，一次性返回完整结果")

    p_replan = sub.add_parser("replan", help="根据反馈重新规划路线（自动使用上次plan的结果）")
    p_replan.add_argument("feedback", help="用户反馈，例如: 太贵了，控制在150以内")
    p_replan.add_argument("--user", default="u001", help="用户ID (默认: u001)")
    p_replan.add_argument("--no-stream", action="store_true", help="禁用流式输出，一次性返回完整结果")

    sub.add_parser("pois", help="查看POI数据概览")
    sub.add_parser("profiles", help="查看可用用户画像")

    args = parser.parse_args()

    if args.command == "plan":
        cmd_plan(args.query, args.user, stream=not args.no_stream)
    elif args.command == "replan":
        if not INTENT_CACHE.exists():
            print("错误：没有上次的规划结果，请先运行 plan 命令。", file=sys.stderr)
            sys.exit(1)
        with INTENT_CACHE.open(encoding="utf-8") as f:
            cache = json.load(f)
        intent = cache["intent"] if "intent" in cache else cache
        user_id = cache.get("user_id", args.user)
        cmd_replan(intent, args.feedback, user_id, stream=not args.no_stream)
    elif args.command == "pois":
        cmd_pois()
    elif args.command == "profiles":
        cmd_profiles()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
