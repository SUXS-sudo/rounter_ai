"""LLM-powered Chinese explanation generation for route results.

Uses MiMo API to generate natural, conversational route explanations.
Falls back to template-based generation when the API is unavailable.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from core.mimo_client import chat, chat_stream

logger = logging.getLogger(__name__)

EXPLANATION_SYSTEM_PROMPT = """你是一个贴心的出行路线推荐助手。用户已经获得了路线规划结果，你需要用自然、口语化的中文为用户解释这些路线。

你的解释应该包含：
1. 简要总结用户的需求（城市、时间、预算、偏好）
2. 每条路线的亮点和特色
3. 具体行程安排（时间、地点、交通方式、费用）
4. 预算分析
5. 风险提示（排队、超预算、时间紧张等）
6. 调整建议

要求：
- 语气亲切自然，像朋友推荐一样
- 使用 emoji 增加可读性
- 突出每条路线的差异化优势
- 如果有风险，给出具体的应对建议
- 整体结构清晰，方便快速浏览"""

PREFERENCE_LABELS = {
    "火锅": "想吃火锅",
    "小吃": "想吃小吃",
    "咖啡": "想喝咖啡",
    "奶茶": "想喝奶茶",
    "书店": "想逛书店",
    "拍照": "重视拍照出片",
    "夜景": "重视夜景",
    "安静": "想找安静一点的地方",
    "文化": "偏好文化 / 文艺氛围",
    "少排队": "希望少排队",
    "少走路": "希望少走路、别太累",
    "室内": "需要室内或雨天友好",
}

REASON_CODE_LABELS = {
    "strong_poi_quality": "整体POI质量较高",
    "within_budget": "预算可控",
    "over_budget": "预算略有超出",
    "compact_route": "路线比较紧凑",
    "long_travel_time": "交通时间偏长",
    "low_queue_risk": "排队风险较低",
    "high_queue_risk": "排队风险偏高",
    "category_diverse": "品类较丰富",
    "contains_food": "包含餐饮补给",
    "missing_food": "餐饮补给不足",
    "matches_user_preferences": "匹配用户偏好",
    "covers_strong_preferences": "覆盖了关键偏好",
    "misses_strong_preferences": "部分关键偏好未覆盖",
    "photo_friendly_route": "适合拍照",
    "indoor_friendly_route": "适合室内安排",
    "high_quality": "品质较高",
    "matches_preferences": "匹配偏好",
    "budget_friendly": "性价比好",
    "low_queue_risk": "排队少",
    "high_queue_risk": "排队较多",
    "profile_fit": "适合你的偏好",
    "photo_friendly": "适合拍照",
    "night_view_friendly": "夜景好",
    "indoor_friendly": "室内体验好",
    "neutral_match": "综合体验均衡",
}


def generate_explanation(routes: list[dict[str, Any]], intent: dict[str, Any] | Any) -> str:
    """Generate a Chinese explanation for route results using MiMo LLM.

    Args:
        routes: Route dictionaries returned by ``generate_routes``.
        intent: Parsed user intent dictionary.

    Returns:
        A natural Chinese explanation string.
    """

    intent_data = _to_dict(intent)
    try:
        explanation = _llm_explain(routes, intent_data)
    except Exception as exc:
        logger.warning("LLM explanation failed, falling back to template: %s", exc)
        explanation = _template_explain(routes, intent_data)
    return explanation + "\n\n【完整路线明细】\n" + _full_route_details(routes)


def generate_explanation_stream(routes: list[dict[str, Any]], intent: dict[str, Any] | Any) -> Generator[str, None, None]:
    """Generate a streaming Chinese explanation for route results.

    Tries LLM streaming first, falls back to yielding the full template explanation at once.

    Args:
        routes: Route dictionaries returned by ``generate_routes``.
        intent: Parsed user intent dictionary.

    Yields:
        Text chunks of the explanation.
    """

    intent_data = _to_dict(intent)

    try:
        yield from _llm_explain_stream(routes, intent_data)
    except Exception as exc:
        logger.warning("LLM streaming explanation failed, falling back to template: %s", exc)
        yield _template_explain(routes, intent_data)
    yield "\n\n【完整路线明细】\n" + _full_route_details(routes)


def _llm_explain(routes: list[dict[str, Any]], intent: dict[str, Any]) -> str:
    """Use MiMo API to generate natural language explanation."""

    user_prompt = _build_explain_prompt(routes, intent)

    return chat(
        system_prompt=EXPLANATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=1024,
    )


def _llm_explain_stream(routes: list[dict[str, Any]], intent: dict[str, Any]) -> Generator[str, None, None]:
    """Use MiMo API to stream natural language explanation."""

    user_prompt = _build_explain_prompt(routes, intent)

    yield from chat_stream(
        system_prompt=EXPLANATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=1024,
    )


def _build_explain_prompt(routes: list[dict[str, Any]], intent: dict[str, Any]) -> str:
    """Build the user prompt for LLM explanation generation."""

    route_summaries = []
    for idx, route in enumerate(routes, 1):
        summary = {
            "title": route.get("title", f"路线{idx}"),
            "total_score": route.get("total_score", 0),
            "total_budget": route.get("total_budget", 0),
            "total_travel_cost": route.get("total_travel_cost", 0),
            "total_duration_minutes": route.get("total_duration_minutes", 0),
            "total_travel_minutes": route.get("total_travel_minutes", 0),
            "stops": [],
            "reason_codes": route.get("reason_codes", []),
            "warnings": route.get("warnings", []),
        }
        for stop in route.get("pois", []):
            summary["stops"].append({
                "name": stop.get("name", ""),
                "category": stop.get("category", ""),
                "arrival_time": stop.get("arrival_time", ""),
                "leave_time": stop.get("leave_time", ""),
                "stay_minutes": stop.get("stay_minutes", 0),
                "travel_mode": stop.get("travel_mode", ""),
                "travel_cost": stop.get("travel_cost", 0),
                "price": stop.get("price", 0),
                "queue_minutes": stop.get("estimated_queue_minutes", 0),
                "reason": stop.get("reason", ""),
            })
        route_summaries.append(summary)

    preferences_text = "、".join(intent.get("preferences", [])) or "无特殊偏好"
    avoid_text = "、".join(intent.get("avoid", [])) or "无"

    return f"""用户需求：
- 城市：{intent.get('city', '成都')}
- 起点：{intent.get('start_location', '春熙路')}
- 时间：{intent.get('start_time', '未指定')} 至 {intent.get('end_time', '21:00')}
- 预算：{intent.get('budget', 300)}元
- 偏好：{preferences_text}
- 避开：{avoid_text}

路线规划结果（JSON）：
{__import__('json').dumps(route_summaries, ensure_ascii=False, indent=1)}

请为用户生成友好的路线解释。"""


def _full_route_details(routes: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, route in enumerate(routes, start=1):
        title = route.get("title") or f"路线{index}"
        lines.append(f"### 路线{index}：{title}")
        lines.append(
            f"评分：{route.get('total_score', 0)} | 预算：{route.get('total_budget', 0)}元 | "
            f"总耗时：{route.get('total_duration_minutes', 0)}分钟 | 交通：{route.get('total_travel_minutes', 0)}分钟"
        )
        for stop_index, step in enumerate(route.get("pois", []), start=1):
            lines.append(
                f"{stop_index}. {step.get('arrival_time', '')} - {step.get('name', '')}"
                f"（{step.get('category', '')}，停留{step.get('stay_minutes', 0)}分钟，"
                f"{step.get('travel_mode', '步行')}{step.get('travel_from_previous_minutes', 0)}分钟）"
            )
        warnings = route.get("warnings", [])
        if warnings:
            lines.append("提示：" + "；".join(str(item) for item in warnings))
        lines.append("")
    return "\n".join(lines).strip()


def explain_route(plan: Any) -> str:
    """Backward-compatible wrapper for older ``RoutePlan`` style objects."""

    if isinstance(plan, dict):
        routes = [plan] if "pois" in plan else plan.get("routes", [])
        intent = plan.get("intent") or plan.get("request") or {}
        return generate_explanation(routes, intent)

    stops = getattr(plan, "stops", [])
    names = "、".join(getattr(getattr(stop, "poi", None), "name", "") for stop in stops)
    return f"本路线包含{len(stops)}个点位：{names}。"


# ---------------------------------------------------------------------------
# Template-based fallback (original implementation)
# ---------------------------------------------------------------------------


def _template_explain(routes: list[dict[str, Any]], intent: dict[str, Any]) -> str:
    """Template-based explanation generation as fallback."""

    lines: list[str] = []
    lines.append("已根据你的需求生成路线建议。")
    lines.append("")
    lines.append("【识别到的需求】")
    lines.append(_describe_intent(intent))
    lines.append("")

    if not routes:
        lines.append("当前没有生成可行路线。可能原因是营业时间、结束时间、预算或偏好约束过紧。")
        lines.append("可调整建议：放宽结束时间、提高预算、减少必选偏好，或允许更长交通时间。")
        return "\n".join(lines)

    lines.append("【路线推荐】")
    for index, route in enumerate(routes, start=1):
        lines.extend(_describe_route(index, route))
        lines.append("")

    lines.append("【可调整建议】")
    lines.extend(_adjustment_suggestions(routes, intent))
    return "\n".join(lines).strip()


def _describe_intent(intent: dict[str, Any]) -> str:
    preferences = [PREFERENCE_LABELS.get(item, str(item)) for item in intent.get("preferences", [])]
    avoid = intent.get("avoid", [])
    parts = [
        f"城市：{intent.get('city', '成都')}",
        f"起点：{intent.get('start_location', '春熙路')}",
        f"时间：{intent.get('start_time') or '未指定'} 至 {intent.get('end_time') or '21:00'}",
        f"预算：{intent.get('budget', 300)}元以内",
    ]
    if preferences:
        parts.append("偏好：" + "、".join(preferences))
    else:
        parts.append("偏好：未指定，优先按综合体验排序")
    if avoid:
        parts.append("避开：" + "、".join(str(item) for item in avoid))
    return "；".join(parts) + "。"


def _describe_route(index: int, route: dict[str, Any]) -> list[str]:
    title = route.get("title") or f"路线{index}"
    total_score = route.get("total_score", 0)
    total_budget = route.get("total_budget", 0)
    total_duration = route.get("total_duration_minutes", 0)
    total_travel_cost = route.get("total_travel_cost", 0)
    score_detail = route.get("score_detail", {})

    emoji = ["\U0001f947", "\U0001f948", "\U0001f949"]
    tag = emoji[index - 1] if index <= 3 else f"  {index}."

    lines = [
        f"{tag} {title}（综合评分 {total_score}）",
        f"   预算：约{total_budget}元（含交通费{total_travel_cost}元） | 总耗时：约{total_duration}分钟",
        "   行程：",
    ]

    for i, step in enumerate(route.get("pois", []), 1):
        queue_text = ""
        queue_minutes = int(step.get("estimated_queue_minutes") or 0)
        if queue_minutes > 0:
            queue_text = f"，排队约{queue_minutes}分钟"
        travel_mode = step.get("travel_mode", "步行")
        travel_cost = step.get("travel_cost", 0)
        price = step.get("price", 0)
        travel_text = f"{travel_mode}{step.get('travel_from_previous_minutes')}分钟"
        if travel_cost > 0:
            travel_text += f"（{travel_cost}元）"
        cost_text = f"，人均{price}元" if price > 0 else ""
        lines.append(
            f"      {i}. {step.get('arrival_time')} -> {step.get('name')} "
            f"（{travel_text}，停留{step.get('stay_minutes')}分钟{cost_text}{queue_text}）"
        )

    reasons = _route_reasons(route)
    lines.append(f"   推荐：{reasons}")
    risk_text = _route_risks(route, score_detail)
    lines.append(f"   提示：{risk_text}")
    return lines


def _route_reasons(route: dict[str, Any]) -> str:
    reason_labels = [
        REASON_CODE_LABELS.get(code, code)
        for code in route.get("reason_codes", [])
        if code not in {"over_budget", "long_travel_time", "high_queue_risk", "missing_food"}
    ]
    poi_reasons = [step.get("reason", "") for step in route.get("pois", []) if step.get("reason")]
    combined: list[str] = []
    for item in reason_labels + poi_reasons:
        if item and item not in combined:
            combined.append(item)
    if not combined:
        return "路线在预算、距离和兴趣匹配之间较为均衡。"
    return "；".join(combined[:5]) + "。"


def _route_risks(route: dict[str, Any], score_detail: dict[str, Any]) -> str:
    risks = list(route.get("warnings", []))
    average_queue_risk = float(score_detail.get("average_queue_risk", 0))
    if average_queue_risk >= 0.6:
        risks.append("部分点位排队风险偏高，建议避开饭点或热门时段。")
    budget_lower = score_detail.get("budget_band_lower")
    budget_upper = score_detail.get("budget_band_upper")
    estimated_cost = route.get("total_budget", score_detail.get("estimated_cost", 0))
    if budget_lower is not None and estimated_cost < budget_lower:
        risks.append("总花费明显低于目标预算区间，路线可能偏保守。")
    if budget_upper is not None and estimated_cost > budget_upper:
        risks.append("总花费高于目标预算区间，整体花费会偏贵。")
    if route.get("total_travel_minutes", 0) >= 80:
        risks.append("交通时间偏长，实际体验可能会被路程拉散。")
    must_satisfy = route.get("score_detail", {}).get("must_satisfy_preferences", [])
    covered = route.get("score_detail", {}).get("covered_must_satisfy_preferences", [])
    missing_must = [item for item in must_satisfy if item not in covered]
    if missing_must:
        risks.append(f"当前方案未完全满足主诉求：{','.join(str(item) for item in missing_must)}。")
    if not risks:
        return "暂无明显风险，按当前时间和预算约束可正常执行。"
    return " ".join(str(item) for item in risks)


def _adjustment_suggestions(routes: list[dict[str, Any]], intent: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []
    best_route = max(routes, key=lambda item: item.get("total_score", 0))
    preferences = set(intent.get("preferences", []))
    if any(route.get("warnings") for route in routes):
        suggestions.append("- 如果想更稳，可以提高预算或减少高客单价餐饮点。")
    if "少排队" not in preferences and _avg_queue_risk(best_route) >= 0.5:
        suggestions.append('- 如果临时不想等位，可以补充"少排队"，系统会更偏向低排队风险点位。')
    if "少走路" not in preferences and best_route.get("total_travel_minutes", 0) >= 60:
        suggestions.append('- 如果担心体力，可以补充"少走路"或缩短路线点位数量。')
    if "室内" not in preferences:
        suggestions.append('- 如果遇到下雨，可以补充"雨天"或"室内"，系统会优先选择商场、书店、茶馆等点位。')
    if not suggestions:
        suggestions.append('- 当前路线约束比较清晰，可以按综合最优路线执行；若临时变化，可反馈"便宜点""少走路""不想排队"等重新规划。')
    return suggestions


def _avg_queue_risk(route: dict[str, Any]) -> float:
    score_detail = route.get("score_detail", {})
    if score_detail.get("average_queue_risk") is not None:
        return float(score_detail["average_queue_risk"])
    queue_minutes = [int(step.get("estimated_queue_minutes") or 0) for step in route.get("pois", [])]
    if not queue_minutes:
        return 0.0
    return min(1.0, sum(queue_minutes) / len(queue_minutes) / 40)


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
