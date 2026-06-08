"""Keyword-based UGC analyzer for POI review text.

This module keeps the first version fully local and deterministic. It converts
plain review strings into normalized feature scores that can later be blended
with structured POI features and user preferences.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

try:
    from models.schemas import Review
except ImportError:  # pragma: no cover - keeps the module usable in isolation.
    Review = Any  # type: ignore


FEATURE_NAMES = (
    "taste",
    "photo",
    "queue_risk",
    "cost_performance",
    "quiet",
    "indoor",
    "family_friendly",
    "night_view",
)

KEYWORD_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "taste": {
        "positive": ("好吃", "味道好", "口味好", "香", "锅底", "新鲜", "稳定", "地道", "本地味道"),
        "negative": ("不好吃", "踩雷", "一般般", "寡淡", "油腻", "失望"),
    },
    "photo": {
        "positive": ("拍照", "出片", "好拍", "打卡", "地标", "建筑", "氛围", "构图"),
        "negative": ("不好拍", "杂乱", "光线差", "遮挡"),
    },
    "queue_risk": {
        "positive": ("排队", "等位", "人多", "拥挤", "爆满", "热门", "高峰", "要等"),
        "negative": ("少排队", "不用排队", "不排队", "人少", "空位", "清净", "工作日体验好"),
    },
    "cost_performance": {
        "positive": ("性价比", "划算", "不贵", "价格友好", "值", "免费", "便宜", "实惠"),
        "negative": ("偏贵", "太贵", "很贵", "价格高", "不值", "性价比低", "坑"),
    },
    "quiet": {
        "positive": ("安静", "清静", "慢", "舒服", "放松", "闹中取静", "休息"),
        "negative": ("吵", "嘈杂", "喧闹", "热闹", "音量大", "人声鼎沸"),
    },
    "indoor": {
        "positive": ("室内", "雨天", "下雨", "躲雨", "商场", "影院", "书店", "茶馆", "不晒"),
        "negative": ("露天", "户外", "晒", "淋雨", "天气影响"),
    },
    "family_friendly": {
        "positive": ("亲子", "孩子", "带娃", "老人", "家庭", "家人", "一家", "小朋友"),
        "negative": ("不适合亲子", "深夜", "酒吧", "音量大", "抽烟"),
    },
    "night_view": {
        "positive": ("夜景", "晚上", "夜晚", "灯光", "夜游", "江景", "夜生活", "深夜"),
        "negative": ("白天更好", "晚上一般", "灯光暗", "太暗"),
    },
}

BASE_SCORES = {
    "taste": 0.5,
    "photo": 0.5,
    "queue_risk": 0.35,
    "cost_performance": 0.5,
    "quiet": 0.5,
    "indoor": 0.5,
    "family_friendly": 0.5,
    "night_view": 0.5,
}


def analyze_reviews(reviews: list[str]) -> dict[str, float | str]:
    """Analyze review texts and return normalized UGC feature scores.

    Args:
        reviews: A list of plain review strings. Empty strings are ignored.

    Returns:
        A dictionary with eight 0-to-1 feature scores: ``taste``, ``photo``,
        ``queue_risk``, ``cost_performance``, ``quiet``, ``indoor``,
        ``family_friendly`` and ``night_view``, plus a Chinese ``ugc_summary``.
    """

    texts = _normalize_texts(reviews)
    if not texts:
        result = {feature: 0.5 for feature in FEATURE_NAMES}
        result["ugc_summary"] = "暂无可分析的UGC文本，已返回中性默认分。"
        return result

    scores: dict[str, float] = {}
    hit_details: dict[str, tuple[int, int]] = {}
    for feature in FEATURE_NAMES:
        positive_hits, negative_hits = _count_rule_hits(
            texts,
            KEYWORD_RULES[feature]["positive"],
            KEYWORD_RULES[feature]["negative"],
        )
        hit_details[feature] = (positive_hits, negative_hits)
        scores[feature] = _score_feature(feature, positive_hits, negative_hits, len(texts))

    return {
        **scores,
        "ugc_summary": _build_summary(scores, hit_details, len(texts)),
    }


def enrich_pois_with_ugc(pois: list[dict], reviews: list[dict]) -> list[dict]:
    """Blend review-derived UGC features into POI features.

    UGC weight scales with review count:
    - >= 10 reviews: UGC weight 0.50 (high confidence)
    - 5-9 reviews:  UGC weight 0.35 (moderate confidence)
    - 1-4 reviews:  UGC weight 0.15 (low confidence)
    - 0 reviews:    no blending, keep original features

    Args:
        pois: POI dictionaries. The original objects are not mutated.
        reviews: Review dictionaries containing ``poi_id`` and ``text``.

    Returns:
        New POI dictionaries with the same original fields, enriched
        ``features`` and an added ``ugc_summary`` field.
    """

    reviews_by_poi: dict[str, list[str]] = {}
    for review in reviews:
        poi_id = str(review.get("poi_id") or "")
        if not poi_id:
            continue
        reviews_by_poi.setdefault(poi_id, []).append(str(review.get("text") or ""))

    enriched_pois: list[dict] = []
    for poi in pois:
        enriched = {**poi}
        features = dict(enriched.get("features") or {})
        poi_id = str(enriched.get("id"))
        poi_reviews = reviews_by_poi.get(poi_id, [])
        ugc_result = analyze_reviews(poi_reviews)

        review_count = len(poi_reviews)
        ugc_weight = _ugc_weight(review_count)
        original_weight = 1.0 - ugc_weight

        for feature in FEATURE_NAMES:
            original = _as_float(features.get(feature), 0.5)
            ugc_score = _as_float(ugc_result.get(feature), 0.5)
            if feature == "queue_risk":
                features[feature] = round(max(original, ugc_score), 3)
            else:
                features[feature] = round(_clamp(original * original_weight + ugc_score * ugc_weight), 3)

        enriched["features"] = features
        enriched["ugc_summary"] = str(ugc_result.get("ugc_summary", "暂无UGC摘要。"))
        enriched["ugc_review_count"] = review_count
        enriched_pois.append(enriched)

    return enriched_pois


def _ugc_weight(review_count: int) -> float:
    """根据评论数量返回 UGC 混合权重。"""
    if review_count >= 10:
        return 0.50
    if review_count >= 5:
        return 0.35
    if review_count >= 1:
        return 0.15
    return 0.0


def summarize_reviews(reviews: list[Review]) -> dict[str, object]:
    """Summarize structured review objects and include UGC feature analysis.

    Args:
        reviews: Review model instances with ``rating``, ``tags`` and ``text``.

    Returns:
        A compact aggregate summary containing average rating, top tags, review
        count and the same keyword-based feature scores from ``analyze_reviews``.
    """

    tags = Counter(tag for review in reviews for tag in getattr(review, "tags", []))
    ratings = [float(review.rating) for review in reviews if getattr(review, "rating", None) is not None]
    average_rating = sum(ratings) / len(ratings) if ratings else 0
    texts = [str(getattr(review, "text", "")) for review in reviews]

    return {
        "average_rating": round(average_rating, 2),
        "top_tags": [tag for tag, _ in tags.most_common(8)],
        "review_count": len(reviews),
        "ugc_features": analyze_reviews(texts),
    }


def _normalize_texts(reviews: Iterable[str]) -> list[str]:
    return [str(review).strip() for review in reviews if str(review).strip()]


def _count_rule_hits(
    texts: list[str],
    positive_keywords: Iterable[str],
    negative_keywords: Iterable[str],
) -> tuple[int, int]:
    positive_hits = 0
    negative_hits = 0
    for text in texts:
        masked_text = text
        for keyword in negative_keywords:
            if keyword in masked_text:
                negative_hits += 1
                masked_text = masked_text.replace(keyword, " " * len(keyword))
        for keyword in positive_keywords:
            if keyword in masked_text:
                positive_hits += 1
    return positive_hits, negative_hits


def _score_feature(feature: str, positive_hits: int, negative_hits: int, review_count: int) -> float:
    base_score = BASE_SCORES[feature]
    evidence_scale = max(1, review_count)
    delta = (positive_hits - negative_hits) / evidence_scale * 0.22
    return round(_clamp(base_score + delta), 3)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _build_summary(
    scores: dict[str, float],
    hit_details: dict[str, tuple[int, int]],
    review_count: int,
) -> str:
    summary_parts: list[str] = [f"共分析{review_count}条UGC文本"]

    if scores["taste"] >= 0.62:
        summary_parts.append("口味反馈较好")
    elif scores["taste"] <= 0.38:
        summary_parts.append("口味评价偏弱")

    if scores["photo"] >= 0.62:
        summary_parts.append("拍照出片信号明显")

    if scores["queue_risk"] >= 0.62:
        summary_parts.append("排队或拥挤风险偏高")
    elif scores["queue_risk"] <= 0.32:
        summary_parts.append("排队风险较低")

    if scores["cost_performance"] >= 0.62:
        summary_parts.append("性价比反馈较好")
    elif scores["cost_performance"] <= 0.38:
        summary_parts.append("价格或性价比存在负面反馈")

    if scores["quiet"] >= 0.62:
        summary_parts.append("环境偏安静")
    elif scores["quiet"] <= 0.38:
        summary_parts.append("环境偏热闹或嘈杂")

    if scores["indoor"] >= 0.62:
        summary_parts.append("适合室内或雨天安排")

    if scores["family_friendly"] >= 0.62:
        summary_parts.append("亲子或家庭友好")
    elif scores["family_friendly"] <= 0.38:
        summary_parts.append("亲子友好度偏低")

    if scores["night_view"] >= 0.62:
        summary_parts.append("夜景或夜间体验突出")

    if len(summary_parts) == 1:
        strongest_feature = max(FEATURE_NAMES, key=lambda feature: hit_details[feature][0])
        feature_labels = {
            "taste": "口味",
            "photo": "拍照",
            "queue_risk": "排队风险",
            "cost_performance": "性价比",
            "quiet": "安静度",
            "indoor": "室内友好度",
            "family_friendly": "亲子友好度",
            "night_view": "夜景",
        }
        summary_parts.append(f"整体信号较均衡，当前最明显的维度是{feature_labels[strongest_feature]}")

    return "，".join(summary_parts) + "。"
