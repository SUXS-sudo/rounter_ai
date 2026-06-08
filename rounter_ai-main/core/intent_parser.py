"""LLM-powered user intent parser with rule-based fallback.

Uses MiMo API to understand natural language route-planning requests.
Falls back to regex-based parsing if the API is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from core.mimo_client import chat_json
from core.poi_retriever import ZONE_ALIASES

logger = logging.getLogger(__name__)

DEFAULT_CITY = "成都"
DEFAULT_END_TIME = "21:00"
DEFAULT_BUDGET = 300
DEFAULT_TRAVEL_MODE = "walking"
DEFAULT_PEOPLE_COUNT = 1
DEFAULT_SCENARIO = "general"

CITY_DEFAULT_START_LOCATIONS = {
    "北京": "三里屯",
    "上海": "新天地",
    "广州": "北京路",
    "深圳": "海岸城",
    "成都": "春熙路",
    "杭州": "西湖",
    "武汉": "楚河汉街",
    "西安": "大雁塔",
    "重庆": "解放碑",
    "南京": "新街口",
    "天津": "滨江道",
    "苏州": "观前街",
    "长沙": "五一广场",
    "青岛": "台东",
    "郑州": "二七广场",
    "厦门": "曾厝垵",
    "昆明": "翠湖",
    "大连": "星海广场",
    "三亚": "大东海",
    "丽江": "大研古城",
}
DEFAULT_START_LOCATION = CITY_DEFAULT_START_LOCATIONS[DEFAULT_CITY]

SUPPORTED_CITIES = (
    "北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安", "重庆", "南京",
    "天津", "苏州", "长沙", "青岛", "郑州", "厦门", "昆明", "大连", "三亚", "丽江",
)

KNOWN_LOCATIONS = (
    "春熙路", "太古里", "宽窄巷子", "九眼桥", "大慈寺", "IFS", "安顺廊桥", "望江楼公园",
    "三里屯", "中关村", "王府井", "西单", "国贸",
    "新天地", "陆家嘴", "南京路", "徐家汇", "静安寺",
    "北京路", "天河城", "珠江新城", "上下九",
    "海岸城", "华强北", "东门", "福田CBD",
    "西湖", "湖滨", "武林广场", "钱江新城",
    "楚河汉街", "江汉路", "光谷",
    "大雁塔", "钟楼", "小寨", "回民街",
    "解放碑", "观音桥", "洪崖洞",
    "新街口", "夫子庙",
    "观前街", "金鸡湖",
    "曾厝垵", "中山路", "鼓浪屿",
    "翠湖", "南屏街",
    "星海广场", "青泥洼桥",
    "大东海", "三亚湾", "亚龙湾",
    "大研古城", "束河古镇",
)

PREFERENCE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("火锅", ("火锅",)),
    ("小吃", ("小吃",)),
    ("咖啡", ("咖啡",)),
    ("奶茶", ("奶茶", "茶饮", "果茶")),
    ("书店", ("书店", "逛书店", "阅读空间", "文创书店")),
    ("拍照", ("拍照", "出片")),
    ("夜景", ("夜景",)),
    ("安静", ("安静", "清静", "安安静静", "适合坐坐")),
    ("文化", ("看展", "展览", "博物馆", "画廊", "文艺", "文化")),
    ("少排队", ("少排队", "不想排队", "别排队")),
    ("少走路", ("少走路", "别太累", "不想太累", "节奏轻松", "轻松一点")),
    ("室内", ("下雨", "雨天", "室内")),
)

CHINESE_NUMBER_MAP = {
    "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

INTENT_SYSTEM_PROMPT = """你是一个路线规划助手的意图解析模块。用户会输入中文的出行需求，你需要提取结构化信息。

请严格返回以下 JSON 格式，不要添加任何其他文字：
{
  "city": "城市名称",
  "start_location": "出发地点",
  "end_location": "终点地点或null",
  "zone": "商圈名称或空字符串",
  "start_time": "出发时间 HH:MM 格式或null",
  "end_time": "结束时间 HH:MM 格式",
  "budget": 数字（预算金额，单位元）,
  "preferences": ["偏好关键词列表"],
  "avoid": ["需要避开的内容"],
  "travel_mode": "出行方式",
  "people_count": 数字（人数）,
  "scenario": "场景类型"
}

规则：
- city：提取城市名，默认为"成都"
- zone：如果用户提到了商圈（如春熙路、太古里、宽窄巷子、万象城、三里屯、南京路等），提取商圈全称（如"春熙路商圈"），否则为空字符串
- start_time：如果提到"下午"则为"14:00"，"上午"则为"09:00"，具体时间按用户说的
- end_time：如果提到"晚上9点前"则为"21:00"，按用户说的提取
- budget：提取数字，默认300
- preferences：提取偏好如"火锅"、"拍照"、"夜景"、"少排队"、"少走路"、"室内"、"小吃"、"咖啡"
- avoid：根据偏好推导，少排队→避开排队，少走路→避长远路，室内→避开露天
- travel_mode：walking/taxi/subway/bike，默认walking
- people_count：提取人数，默认1
- scenario：family/date/friends/solo/rainy_day/general

示例：
输入："下午从春熙路出发，想吃火锅、拍照，不想排队，预算300，晚上9点前结束"
输出：{"city":"成都","start_location":"春熙路","end_location":null,"zone":"春熙路商圈","start_time":"14:00","end_time":"21:00","budget":300,"preferences":["火锅","拍照","少排队"],"avoid":["排队"],"travel_mode":"walking","people_count":1,"scenario":"general"}

输入："带孩子去宽窄巷子附近玩，预算200，下午开始"
输出：{"city":"成都","start_location":"宽窄巷子","end_location":null,"zone":"宽窄巷子商圈","start_time":"14:00","end_time":"21:00","budget":200,"preferences":[],"avoid":[],"travel_mode":"walking","people_count":3,"scenario":"family"}

输入："晚上和朋友喝酒看夜景，从九眼桥出发"
输出：{"city":"成都","start_location":"九眼桥","end_location":null,"zone":"九眼桥商圈","start_time":"19:00","end_time":"23:00","budget":300,"preferences":["夜景"],"avoid":[],"travel_mode":"walking","people_count":2,"scenario":"friends"}"""


SEMANTIC_COMPLETION_SYSTEM_PROMPT = """你是一个路线规划助手的语义补全模块。规则解析已经提取了城市、时间、预算、起点等基础字段，你只需要补充规则难以直接识别的偏好和语义标签。

请严格返回 JSON，不要添加其他文字：
{
  "preferences_to_add": ["偏好词"],
  "must_satisfy_preferences_to_add": ["必须优先满足的偏好词"],
  "avoid_to_add": ["避开词"],
  "style_tags": ["风格标签"],
  "semantic_hints": ["语义提示"]
}

可补充的偏好词优先使用这些 canonical 词：
["火锅", "小吃", "咖啡", "奶茶", "书店", "拍照", "夜景", "室内", "安静", "文化", "少排队", "少走路"]

规则：
- 不要重复规则解析已经明确识别出的偏好
- 只补自然语言里隐含但有价值的偏好
- 如果用户表达非常强，例如“就想 / 一定要 / 必须 / 主要想 / 优先”，则可把对应偏好放进 must_satisfy_preferences_to_add
- 例如：
  - 逛书店 -> 书店
  - 文艺一点 -> 文化 / 书店 / 安静
  - 想坐坐 -> 安静 / 咖啡 / 书店
  - 节奏轻松一点 -> 少走路
  - 就想吃火锅 -> must_satisfy_preferences_to_add = ["火锅"]
"""


def parse_user_intent(user_query: str) -> dict[str, Any]:
    """Parse a Chinese route-planning request into a normalized intent dict.

    Uses rule-based parsing for deterministic fields and LLM semantic completion
    for preferences that are hard to extract with keywords alone.
    """

    query = (user_query or "").strip()
    if not query:
        return _default_intent()

    base_intent = _rule_based_parse(query)
    if not _should_run_semantic_completion(query, base_intent):
        return _fill_defaults(base_intent)
    try:
        completed = _llm_complete_intent_semantics(query, base_intent)
    except Exception as exc:
        logger.warning("Semantic completion failed, keeping rule-based intent: %s", exc)
        completed = base_intent
    return _fill_defaults(completed)


def _llm_parse(query: str) -> dict[str, Any]:
    """Use MiMo API to parse user intent."""

    result = chat_json(
        system_prompt=INTENT_SYSTEM_PROMPT,
        user_prompt=query,
        temperature=0.1,
        max_tokens=1024,
    )
    return result


def _llm_complete_intent_semantics(query: str, base_intent: dict[str, Any]) -> dict[str, Any]:
    result = chat_json(
        system_prompt=SEMANTIC_COMPLETION_SYSTEM_PROMPT,
        user_prompt=(
            f"原始用户输入：{query}\n"
            f"规则解析结果：{json.dumps(base_intent, ensure_ascii=False)}\n"
            "请只补充明确能从原句中看出来、且规则层没有提取出来的偏好。不要为了让结果更丰富而额外补偏好。"
        ),
        temperature=0.1,
        max_tokens=512,
    )
    return _merge_semantic_completion(base_intent, result)


def _should_run_semantic_completion(query: str, base_intent: dict[str, Any]) -> bool:
    if len(base_intent.get("preferences", [])) <= 1:
        return True
    semantic_triggers = ("书店", "逛书店", "文艺", "安静", "坐坐", "文化", "看展", "展览", "阅读", "节奏轻松")
    return any(trigger in query for trigger in semantic_triggers)


def _merge_semantic_completion(base_intent: dict[str, Any], semantic_result: dict[str, Any]) -> dict[str, Any]:
    intent = dict(base_intent)
    preferences = [str(item) for item in intent.get("preferences", [])]
    must_satisfy = [str(item) for item in intent.get("must_satisfy_preferences", [])]
    avoid = [str(item) for item in intent.get("avoid", [])]

    for item in semantic_result.get("preferences_to_add", []):
        value = str(item).strip()
        if value and value not in preferences:
            preferences.append(value)
    for item in semantic_result.get("must_satisfy_preferences_to_add", []):
        value = str(item).strip()
        if value and value not in must_satisfy:
            must_satisfy.append(value)
        if value and value not in preferences:
            preferences.append(value)
    for item in semantic_result.get("avoid_to_add", []):
        value = str(item).strip()
        if value and value not in avoid:
            avoid.append(value)

    intent["preferences"] = preferences
    intent["must_satisfy_preferences"] = must_satisfy
    intent["avoid"] = avoid
    if semantic_result.get("style_tags"):
        intent["style_tags"] = [str(item) for item in semantic_result.get("style_tags", []) if str(item).strip()]
    if semantic_result.get("semantic_hints"):
        intent["semantic_hints"] = [str(item) for item in semantic_result.get("semantic_hints", []) if str(item).strip()]
    return intent


def _fill_defaults(intent: dict[str, Any]) -> dict[str, Any]:
    """Fill missing fields with defaults."""

    defaults = _default_intent()
    for key, default_value in defaults.items():
        if key not in intent or intent[key] is None:
            intent[key] = default_value

    # Ensure preferences, must_satisfy_preferences and avoid are lists
    if not isinstance(intent.get("preferences"), list):
        intent["preferences"] = []
    if not isinstance(intent.get("must_satisfy_preferences"), list):
        intent["must_satisfy_preferences"] = []
    if not isinstance(intent.get("avoid"), list):
        intent["avoid"] = []

    intent["preferences"] = list(dict.fromkeys(str(item) for item in intent["preferences"] if str(item).strip()))
    intent["must_satisfy_preferences"] = list(dict.fromkeys(str(item) for item in intent["must_satisfy_preferences"] if str(item).strip()))
    for item in intent["must_satisfy_preferences"]:
        if item not in intent["preferences"]:
            intent["preferences"].append(item)
    intent["avoid"] = list(dict.fromkeys(str(item) for item in intent["avoid"] if str(item).strip()))

    derived_avoid = _derive_avoid(intent["preferences"])
    intent["avoid"] = list(dict.fromkeys(intent["avoid"] + derived_avoid))

    return intent


def _default_intent() -> dict[str, Any]:
    """Return an intent dict with all default values."""

    return {
        "city": DEFAULT_CITY,
        "start_location": DEFAULT_START_LOCATION,
        "end_location": None,
        "zone": "",
        "start_time": None,
        "end_time": DEFAULT_END_TIME,
        "budget": DEFAULT_BUDGET,
        "preferences": [],
        "avoid": [],
        "travel_mode": DEFAULT_TRAVEL_MODE,
        "people_count": DEFAULT_PEOPLE_COUNT,
        "scenario": DEFAULT_SCENARIO,
    }


def _derive_avoid(preferences: list[str]) -> list[str]:
    """Derive avoid list from preferences."""

    avoid: list[str] = []
    preference_set = set(preferences)
    if "少排队" in preference_set:
        avoid.append("排队")
    if "少走路" in preference_set:
        avoid.append("长距离步行")
    if "室内" in preference_set:
        avoid.append("露天")
    return avoid


# ---------------------------------------------------------------------------
# Rule-based fallback (kept for reliability when API is unavailable)
# ---------------------------------------------------------------------------


def _rule_based_parse(user_query: str) -> dict[str, Any]:
    """Fallback rule-based parser using regex matching."""

    query = (user_query or "").strip()
    preferences = _extract_preferences(query)

    city = _extract_city(query)
    must_satisfy_preferences = _extract_must_satisfy_preferences(query, preferences)
    return {
        "city": city,
        "start_location": _extract_start_location(query, city),
        "end_location": _extract_end_location(query),
        "zone": _extract_zone(query),
        "start_time": _extract_start_time(query),
        "end_time": _extract_end_time(query) or DEFAULT_END_TIME,
        "budget": _extract_budget(query) or DEFAULT_BUDGET,
        "preferences": preferences,
        "must_satisfy_preferences": must_satisfy_preferences,
        "avoid": _extract_avoid(preferences),
        "travel_mode": _extract_travel_mode(query),
        "people_count": _extract_people_count(query),
        "scenario": _extract_scenario(query),
    }


def parse_intent(user_query: str) -> dict[str, Any]:
    """Backward-compatible alias for callers using the earlier function name."""

    return parse_user_intent(user_query)


def _extract_city(query: str) -> str:
    for city in SUPPORTED_CITIES:
        if city in query:
            return city
    return DEFAULT_CITY


def _extract_preferences(query: str) -> list[str]:
    preferences: list[str] = []
    for preference, keywords in PREFERENCE_RULES:
        if _contains_any(query, keywords):
            preferences.append(preference)
    return preferences


def _extract_must_satisfy_preferences(query: str, preferences: list[str]) -> list[str]:
    must_satisfy: list[str] = []
    strong_prefixes = ("就想", "一定要", "必须", "优先", "主要想")
    for preference in preferences:
        if any(f"{prefix}{preference}" in query for prefix in strong_prefixes):
            must_satisfy.append(preference)
            continue
        if preference in {"火锅", "小吃", "咖啡", "奶茶", "书店"}:
            patterns = [f"想吃{preference}", f"想喝{preference}", f"想去{preference}", f"想逛{preference}"]
            if preference == "书店":
                patterns.extend(["逛书店", "去书店", "就想去书店", "主要想逛书店"])
            if any(pattern in query for pattern in patterns):
                must_satisfy.append(preference)
    return list(dict.fromkeys(must_satisfy))


def _extract_avoid(preferences: Iterable[str]) -> list[str]:
    avoid: list[str] = []
    preference_set = set(preferences)
    if "少排队" in preference_set:
        avoid.append("排队")
    if "少走路" in preference_set:
        avoid.append("长距离步行")
    if "室内" in preference_set:
        avoid.append("露天")
    return avoid


def _extract_budget(query: str) -> int | None:
    patterns = (
        r"(?:预算|人均)\s*(\d{2,5})",
        r"(\d{2,5})\s*(?:元|块)?\s*(?:以内|以下|内)",
    )
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return int(match.group(1))
    return None


def _extract_start_time(query: str) -> str | None:
    if "下午" in query:
        return "14:00"
    if "上午" in query:
        return "09:00"
    return None


def _extract_end_time(query: str) -> str | None:
    colon_time_match = re.search(r"([01]?\d|2[0-3])[:：]([0-5]\d)\s*前", query)
    if colon_time_match:
        hour = int(colon_time_match.group(1))
        minute = int(colon_time_match.group(2))
        return f"{hour:02d}:{minute:02d}"

    point_time_match = re.search(r"(晚上|夜里|晚间)?\s*(\d{1,2})\s*点\s*前", query)
    if point_time_match:
        evening_hint = point_time_match.group(1)
        hour = int(point_time_match.group(2))
        if evening_hint and 1 <= hour <= 11:
            hour += 12
        if 0 <= hour <= 23:
            return f"{hour:02d}:00"

    return None


def _extract_zone(query: str) -> str:
    """Extract business zone name from query using ZONE_ALIASES mapping."""
    for alias, full_name in ZONE_ALIASES.items():
        if alias in query:
            return full_name
    return ""


def _extract_start_location(query: str, city: str) -> str:
    for location in KNOWN_LOCATIONS:
        patterns = (
            rf"从\s*{re.escape(location)}",
            rf"{re.escape(location)}\s*出发",
            rf"起点\s*(?:是|为|在)?\s*{re.escape(location)}",
        )
        if any(re.search(pattern, query) for pattern in patterns):
            return location
    return CITY_DEFAULT_START_LOCATIONS.get(city, DEFAULT_START_LOCATION)


def _extract_end_location(query: str) -> str | None:
    for location in KNOWN_LOCATIONS:
        patterns = (
            rf"到\s*{re.escape(location)}",
            rf"去\s*{re.escape(location)}",
            rf"终点\s*(?:是|为|在)?\s*{re.escape(location)}",
        )
        if any(re.search(pattern, query) for pattern in patterns):
            return location
    return None


def _extract_travel_mode(query: str) -> str:
    if _contains_any(query, ("打车", "出租车", "网约车")):
        return "taxi"
    if "地铁" in query:
        return "subway"
    if _contains_any(query, ("骑行", "骑车", "单车")):
        return "bike"
    if _contains_any(query, ("步行", "走路")):
        return "walking"
    return DEFAULT_TRAVEL_MODE


def _extract_people_count(query: str) -> int:
    if "一家三口" in query:
        return 3

    digit_match = re.search(r"(\d{1,2})\s*(?:个)?(?:人|位)", query)
    if digit_match:
        return max(1, int(digit_match.group(1)))

    chinese_match = re.search(r"([一二两俩三四五六七八九十])\s*(?:个)?(?:人|位)", query)
    if chinese_match:
        return CHINESE_NUMBER_MAP[chinese_match.group(1)]

    if _contains_any(query, ("情侣", "约会")):
        return 2
    if _contains_any(query, ("亲子", "带娃", "孩子", "家庭")):
        return 3

    return DEFAULT_PEOPLE_COUNT


def _extract_scenario(query: str) -> str:
    if _contains_any(query, ("亲子", "带娃", "孩子", "家庭", "一家")):
        return "family"
    if _contains_any(query, ("情侣", "约会")):
        return "date"
    if _contains_any(query, ("朋友", "同事", "团建", "聚餐")):
        return "friends"
    if _contains_any(query, ("一个人", "独自", "solo")):
        return "solo"
    if _contains_any(query, ("下雨", "雨天")):
        return "rainy_day"
    return DEFAULT_SCENARIO


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)
