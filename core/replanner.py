"""LLM-powered route replanning based on user feedback.

Uses MiMo API to understand natural language feedback and update route intent.
Falls back to rule-based feedback processing when the API is unavailable.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

from core.mimo_client import chat_json
from core.preference import get_strong_preferences, matches_preference
from core.poi_retriever import retrieve_candidate_pois
from core.route_optimizer import _beam_search_generate, _finalize_routes, generate_routes
from utils.time_utils import add_minutes, parse_hhmm

logger = logging.getLogger(__name__)

DEFAULT_START_LOCATION = {
    "label": "春熙路",
    "lat": 30.65708,
    "lng": 104.08096,
}

LOCATION_COORDS = {
    "春熙路": {"label": "春熙路", "lat": 30.65708, "lng": 104.08096},
    "太古里": {"label": "太古里", "lat": 30.65398, "lng": 104.08394},
    "宽窄巷子": {"label": "宽窄巷子", "lat": 30.66994, "lng": 104.05958},
    "九眼桥": {"label": "九眼桥", "lat": 30.64057, "lng": 104.09194},
    "大慈寺": {"label": "大慈寺", "lat": 30.65461, "lng": 104.08511},
    "安顺廊桥": {"label": "安顺廊桥", "lat": 30.64202, "lng": 104.08856},
    "望江楼公园": {"label": "望江楼公园", "lat": 30.63582, "lng": 104.09597},
}

REPLAN_SYSTEM_PROMPT = """你是一个路线规划助手的反馈理解模块。用户对之前的路线方案提出了修改意见，你需要理解反馈并输出修改后的意图参数。

请严格返回以下 JSON 格式，不要添加任何其他文字：
{
  "city": "城市名称",
  "start_location": "出发地点",
  "end_location": "终点地点或null",
  "start_time": "出发时间 HH:MM",
  "end_time": "结束时间 HH:MM",
  "budget": 数字,
  "preferences": ["偏好列表"],
  "avoid": ["避开内容"],
  "travel_mode": "出行方式",
  "people_count": 数字,
  "scenario": "场景类型",
  "changes": ["本次修改说明列表"]
}

规则：
- 在原有意图基础上，只修改用户反馈中提到的部分
- 保留用户没有提到的原有参数
- 从反馈中提取新的偏好、预算、时间等变化
- 在changes数组中说明做了哪些修改"""


def understand_replan_intent(
    previous_intent: dict[str, Any] | Any,
    user_feedback: str,
) -> tuple[dict[str, Any], list[str]]:
    """Understand feedback and return the updated intent plus change summary."""

    intent = _normalize_intent(previous_intent)
    feedback = (user_feedback or "").strip()

    updated_intent = deepcopy(intent)
    rule_changes = _apply_feedback_rules(updated_intent, feedback)
    if _has_meaningful_rule_changes(rule_changes):
        return updated_intent, rule_changes

    try:
        llm_intent, llm_changes = _llm_understand_feedback(intent, feedback)
        normalized_llm_intent = deepcopy(llm_intent)
        normalized_rule_changes = _apply_feedback_rules(normalized_llm_intent, feedback)
        changes = list(llm_changes)
        for change in normalized_rule_changes:
            if change not in changes:
                changes.append(change)
        return normalized_llm_intent, changes
    except Exception as exc:
        logger.warning("LLM feedback understanding failed, falling back to rules: %s", exc)
        return updated_intent, rule_changes



def replan_route(
    previous_intent: dict[str, Any] | Any,
    user_feedback: str,
    user_profile: dict[str, Any] | Any,
    pois: list[dict[str, Any] | Any],
) -> dict[str, Any]:
    """Update route constraints from feedback and regenerate route candidates."""

    updated_intent, changes = understand_replan_intent(previous_intent, user_feedback)
    return replan_route_for_intent(updated_intent, user_profile, pois, changes=changes)



def _generate_routes_with_local_recovery(
    start_location: dict[str, Any],
    candidates: list[dict[str, Any]],
    intent: dict[str, Any],
    user_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    recovery_notes: list[str] = []
    recovery_attempts = [
        intent,
        _soften_intent(intent, remove_preferences=['安静']),
        _soften_intent(intent, remove_preferences=['安静', '拍照']),
        _soften_intent(intent, remove_preferences=['安静', '拍照', '少排队']),
        _soften_intent(intent, remove_preferences=['安静', '拍照', '少排队', '少走路']),
    ]

    seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for attempt_index, attempt_intent in enumerate(recovery_attempts):
        signature = (
            tuple(str(item) for item in attempt_intent.get('preferences', [])),
            tuple(str(item) for item in attempt_intent.get('avoid', [])),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        routes = _generate_routes_locally(
            start_location=start_location,
            candidate_pois=candidates,
            intent=attempt_intent,
            user_profile=user_profile,
            top_k=3,
            beam_size=8,
            max_steps=5,
        )
        if routes:
            if attempt_index > 0:
                recovery_notes.append('已自动放宽部分软偏好后生成可行路线。')
            return routes, recovery_notes

    routes = generate_routes(
        start_location=start_location,
        candidate_pois=candidates,
        intent=intent,
        user_profile=user_profile,
        top_k=3,
        beam_size=8,
        max_steps=5,
    )
    return routes, recovery_notes



def _generate_routes_locally(
    start_location: dict[str, Any],
    candidate_pois: list[dict[str, Any]],
    intent: dict[str, Any],
    user_profile: dict[str, Any],
    top_k: int = 3,
    beam_size: int = 8,
    max_steps: int = 5,
) -> list[dict[str, Any]]:
    if top_k <= 0 or not candidate_pois:
        return []

    local_routes = _beam_search_generate(
        start_location,
        candidate_pois,
        intent,
        user_profile,
        max(top_k, 5),
        beam_size,
        max_steps,
    )
    if not local_routes and len(candidate_pois) > 24:
        local_routes = _beam_search_generate(
            start_location,
            candidate_pois[:24],
            intent,
            user_profile,
            max(top_k, 5),
            beam_size,
            max_steps,
        )
    if not local_routes:
        return []
    return _finalize_routes(local_routes, intent, candidate_pois, top_k)



def _soften_intent(intent: dict[str, Any], remove_preferences: list[str]) -> dict[str, Any]:
    softened = deepcopy(intent)
    preferences = [str(item) for item in softened.get('preferences', []) if str(item) not in remove_preferences]
    avoid = [str(item) for item in softened.get('avoid', [])]
    if '少排队' in remove_preferences:
        avoid = [item for item in avoid if item != '排队']
    softened['preferences'] = preferences
    softened['avoid'] = avoid
    return softened



def replan_route_for_intent(
    updated_intent: dict[str, Any] | Any,
    user_profile: dict[str, Any] | Any,
    pois: list[dict[str, Any] | Any],
    changes: list[str] | None = None,
) -> dict[str, Any]:
    """Generate replanned routes for an already-updated intent."""

    profile = _to_dict(user_profile)
    normalized_intent = _normalize_intent(updated_intent)
    poi_dicts = [_to_dict(poi) for poi in pois]

    start_location = _resolve_start_location(normalized_intent.get("start_location"))
    available_pois = _filter_pois_by_avoid(poi_dicts, normalized_intent.get("avoid", []))
    candidates = retrieve_candidate_pois(normalized_intent, profile, available_pois, limit=40)
    routes, recovery_notes = _generate_routes_with_local_recovery(
        start_location=start_location,
        candidates=candidates,
        intent=normalized_intent,
        user_profile=profile,
    )

    warnings: list[str] = []
    if recovery_notes:
        warnings.extend(recovery_notes)
    if not candidates:
        warnings.append("根据新的约束没有召回到候选POI，建议放宽偏好或预算。")
    elif not routes:
        warnings.append("已重新召回候选POI，但营业时间、预算或结束时间约束过紧，暂未生成可行路线。")
    else:
        for preference in get_strong_preferences(normalized_intent):
            candidate_has_match = any(matches_preference(poi, preference) for poi in candidates)
            route_has_match = any(
                matches_preference(_poi_by_id(poi_dicts, step.get("poi_id")), preference)
                for route in routes
                for step in route.get("pois", [])
            )
            if candidate_has_match and not route_has_match:
                warnings.append(f"候选中存在{preference}点位，但当前可行路线未覆盖，可放宽时间或减少其他偏好。")

    return {
        "updated_intent": normalized_intent,
        "routes": routes,
        "candidate_count": len(candidates),
        "changes": changes or [],
        "warnings": warnings,
    }


def _llm_understand_feedback(
    intent: dict[str, Any],
    feedback: str,
) -> tuple[dict[str, Any], list[str]]:
    """Use MiMo API to understand user feedback and update intent."""

    import json

    intent_summary = json.dumps(intent, ensure_ascii=False)

    user_prompt = f"""当前路线意图参数：
{intent_summary}

用户反馈：{feedback}

请根据反馈修改意图参数，返回修改后的完整JSON。"""

    result = chat_json(
        system_prompt=REPLAN_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=1024,
    )

    changes = result.pop("changes", [])
    if not isinstance(changes, list):
        changes = [str(changes)]

    # Merge with original intent (LLM result takes precedence)
    updated_intent = deepcopy(intent)
    for key in ("city", "start_location", "end_location", "start_time", "end_time",
                "budget", "preferences", "avoid", "travel_mode", "people_count", "scenario"):
        if key in result and result[key] is not None:
            updated_intent[key] = result[key]

    # Ensure lists
    for key in ("preferences", "avoid"):
        if not isinstance(updated_intent.get(key), list):
            updated_intent[key] = []

    # Re-apply deterministic feedback rules so explicit hard constraints always stick.
    rule_adjusted_intent = deepcopy(updated_intent)
    rule_changes = _apply_feedback_rules(rule_adjusted_intent, feedback)
    updated_intent = rule_adjusted_intent
    for change in rule_changes:
        if change not in changes:
            changes.append(change)

    return updated_intent, changes


# ---------------------------------------------------------------------------
# Rule-based fallback (original implementation)
# ---------------------------------------------------------------------------


def _apply_feedback_rules(intent: dict[str, Any], feedback: str) -> list[str]:
    """Rule-based feedback processing as fallback."""

    changes: list[str] = []

    explicit_budget = _extract_budget(feedback)
    if explicit_budget is not None:
        intent["budget"] = explicit_budget
        changes.append(f"预算已调整为{explicit_budget}元以内")
    elif _contains_any(feedback, ("太贵了", "便宜点", "便宜一点", "省钱点", "预算低点")):
        current_budget = _safe_int(intent.get("budget"), 300)
        new_budget = max(80, int(round(current_budget * 0.75)))
        intent["budget"] = new_budget
        changes.append(f'已根据"更便宜"的反馈把预算下调到约{new_budget}元')

    if _contains_any(feedback, ("少走路", "别太累", "不想太累", "别太赶", "轻松一点", "放慢一点", "太折腾了", "节奏放慢")):
        _add_preference(intent, "少走路")
        changes.append("已加入少走路偏好")

    if _contains_any(feedback, ("安静一点", "适合坐坐", "能坐很久", "坐很久", "坐久一点", "安静些")):
        _add_preference(intent, "安静")
        changes.append("已加入安静/适合久坐偏好")

    if _contains_any(feedback, ("下雨了", "下雨", "雨天")):
        _add_preference(intent, "室内")
        changes.append("已加入室内/雨天友好偏好")

    if _contains_any(feedback, ("看海", "海边", "海景")):
        _add_preference(intent, "拍照")
        changes.append("已加入海边/看海相关偏好，并优先考虑适合拍照的点位")

    if "不要火锅" in feedback:
        _remove_preference(intent, "火锅")
        _add_avoid(intent, "火锅")
        changes.append("已移除火锅偏好，并加入避开火锅")

    if "换成小吃" in feedback:
        _remove_preference(intent, "火锅")
        _add_avoid(intent, "火锅")
        _add_preference(intent, "小吃")
        changes.append("已将餐饮偏好从火锅调整为小吃")

    if _contains_any(feedback, ("想多拍照", "多拍照", "更出片", "多出片")):
        _add_preference(intent, "拍照")
        changes.append("已加强拍照/出片偏好")

    if _contains_any(feedback, ("不想排队", "少排队", "别排队", "不要游客太多", "游客太多", "别太热门", "别太网红", "不要这么网红", "更本地一点")):
        _add_preference(intent, "少排队")
        _add_avoid(intent, "排队")
        changes.append("已加入少排队/避开热门点位偏好")

    if _contains_any(feedback, ("热闹点", "热闹一点", "朋友加入了", "朋友临时加入了")):
        intent["people_count"] = max(2, _safe_int(intent.get("people_count"), 1))
        changes.append("已按多人/热闹场景调整人数")

    if _contains_any(feedback, ("约会", "更适合约会")):
        intent["scenario"] = "dating"
        intent["people_count"] = max(2, _safe_int(intent.get("people_count"), 1))
        _add_preference(intent, "安静")
        changes.append("已切换到约会场景，并增强安静偏好")

    if _contains_any(feedback, ("吃点东西", "先吃点东西", "找点吃的", "吃个东西")):
        _add_preference(intent, "小吃")
        changes.append("已加入餐饮补给偏好")

    if _contains_any(feedback, ("咖啡店", "咖啡馆", "喝咖啡")):
        _add_preference(intent, "咖啡")
        changes.append("已加入咖啡偏好")

    if "晚点出发" in feedback:
        old_time = str(intent.get("start_time") or "09:00")
        intent["start_time"] = _shift_time(old_time, 60)
        changes.append(f"开始时间已从{old_time}调整为{intent['start_time']}")

    if "早点结束" in feedback:
        old_time = str(intent.get("end_time") or "21:00")
        intent["end_time"] = _shift_time(old_time, -60)
        changes.append(f"结束时间已从{old_time}调整为{intent['end_time']}")

    if not changes:
        changes.append("未识别到明确约束变化，已基于原始意图重新规划")

    _dedupe_intent_lists(intent)
    return changes



def _has_meaningful_rule_changes(changes: list[str]) -> bool:
    return any(change != "未识别到明确约束变化，已基于原始意图重新规划" for change in changes)


def _extract_budget(text: str) -> int | None:
    patterns = (
        r"控制在\s*(\d{2,5})\s*(?:元|块)?\s*(?:以内|以下|内)?",
        r"(?:预算|人均)\s*(\d{2,5})",
        r"(\d{2,5})\s*(?:元|块)?\s*(?:以内|以下|内)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _resolve_start_location(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("lat") is not None and value.get("lng") is not None:
        return dict(value)
    label = str(value or DEFAULT_START_LOCATION["label"])
    for location_name, location in LOCATION_COORDS.items():
        if location_name in label:
            return dict(location)
    return dict(DEFAULT_START_LOCATION)


def _normalize_intent(previous_intent: dict[str, Any] | Any) -> dict[str, Any]:
    intent = deepcopy(_to_dict(previous_intent))
    intent.setdefault("city", "成都")
    intent.setdefault("start_location", "春熙路")
    intent.setdefault("end_location", None)
    intent.setdefault("start_time", "09:00")
    intent.setdefault("end_time", "21:00")
    intent.setdefault("budget", 300)
    intent.setdefault("preferences", [])
    intent.setdefault("avoid", [])
    intent.setdefault("travel_mode", "walking")
    intent.setdefault("people_count", 1)
    intent.setdefault("scenario", "general")
    return intent


def _add_preference(intent: dict[str, Any], preference: str) -> None:
    preferences = intent.setdefault("preferences", [])
    if not isinstance(preferences, list):
        preferences = [preferences]
        intent["preferences"] = preferences
    preferences.append(preference)


def _remove_preference(intent: dict[str, Any], preference: str) -> None:
    preferences = intent.get("preferences", [])
    if not isinstance(preferences, list):
        preferences = [preferences]
    intent["preferences"] = [item for item in preferences if item != preference]


def _add_avoid(intent: dict[str, Any], avoid_item: str) -> None:
    avoid = intent.setdefault("avoid", [])
    if not isinstance(avoid, list):
        avoid = [avoid]
        intent["avoid"] = avoid
    avoid.append(avoid_item)


def _dedupe_intent_lists(intent: dict[str, Any]) -> None:
    for key in ("preferences", "avoid"):
        values = intent.get(key, [])
        if not isinstance(values, list):
            values = [values]
        deduped: list[Any] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        intent[key] = deduped


def _filter_pois_by_avoid(pois: list[dict[str, Any]], avoid_items: Any) -> list[dict[str, Any]]:
    if not isinstance(avoid_items, list):
        avoid_items = [avoid_items]
    hard_keywords: list[str] = []
    for item in avoid_items:
        if item == "火锅":
            hard_keywords.extend(["火锅", "hotpot", "skewer_hotpot"])
    if not hard_keywords:
        return pois
    return [poi for poi in pois if not _contains_any(_poi_text(poi), tuple(hard_keywords))]


def _poi_by_id(pois: list[dict[str, Any]], poi_id: Any) -> dict[str, Any]:
    for poi in pois:
        if str(poi.get("id")) == str(poi_id):
            return poi
    return {}


def _poi_text(poi: dict[str, Any]) -> str:
    tags = poi.get("tags") or []
    if not isinstance(tags, list):
        tags = [tags]
    return " ".join(
        str(value)
        for value in (
            poi.get("name", ""),
            poi.get("category", ""),
            poi.get("sub_category", ""),
            poi.get("address", ""),
            " ".join(str(tag) for tag in tags),
        )
    )


def _shift_time(time_value: str, minutes: int) -> str:
    try:
        parse_hhmm(time_value)
    except ValueError:
        time_value = "09:00" if minutes > 0 else "21:00"
    return add_minutes(time_value, minutes)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
