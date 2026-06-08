"""Geographic helper functions for route planning."""

from __future__ import annotations

from math import asin, ceil, cos, radians, sin, sqrt
from typing import Any


EARTH_RADIUS_KM = 6371.0088

MODE_SPEED_KMPH = {
    "walk": 5.0,
    "walking": 5.0,
    "bike": 12.0,
    "bicycle": 12.0,
    "taxi": 25.0,
    "car": 25.0,
    "transit": 18.0,
    "subway": 18.0,
    "bus": 18.0,
}


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate the great-circle distance between two WGS84 points in km.

    Args:
        lat1: Latitude of the first point.
        lng1: Longitude of the first point.
        lat2: Latitude of the second point.
        lng2: Longitude of the second point.

    Returns:
        Distance in kilometers as a floating-point value.
    """

    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def haversine_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Backward-compatible alias for ``haversine_distance``."""

    return haversine_distance(lat1, lng1, lat2, lng2)


def estimate_travel_minutes(distance_km: float, mode: str = "walk") -> int:
    """Estimate travel duration from distance and travel mode.

    Args:
        distance_km: Distance in kilometers. Must be non-negative.
        mode: One of ``walk``, ``bike``, ``taxi`` or ``transit``. Common aliases
            such as ``walking``, ``subway`` and ``bus`` are also accepted.

    Returns:
        Estimated travel duration in whole minutes. Non-zero trips are rounded
        up to at least one minute.

    Raises:
        ValueError: If ``distance_km`` is negative or ``mode`` is unsupported.
    """

    if distance_km < 0:
        raise ValueError("distance_km must be non-negative")

    normalized_mode = (mode or "walk").strip().lower()
    speed_kmph = MODE_SPEED_KMPH.get(normalized_mode)
    if speed_kmph is None:
        supported_modes = ", ".join(sorted(MODE_SPEED_KMPH))
        raise ValueError(f"Unsupported travel mode: {mode}. Supported modes: {supported_modes}")

    if distance_km == 0:
        return 0

    minutes = distance_km / speed_kmph * 60
    return max(1, ceil(minutes))


WALK_MAX_KM = 0.5
TRANSIT_COST_PER_KM = 0.5
TAXI_COST_PER_KM = 3.0
TAXI_BASE_FARE = 8.0


def auto_travel(distance_km: float) -> dict[str, Any]:
    """Select the best transport mode by distance and return time + cost info.

    Rules:
        - ≤ 0.5 km: walk (free)
        - > 0.5 km and ≤ 5 km: transit (0.5 ¥/km)
        - > 5 km: taxi (3 ¥/km, base fare 8 ¥)
    """

    if distance_km <= WALK_MAX_KM:
        mode = "walk"
        cost = 0.0
    elif distance_km <= 5.0:
        mode = "transit"
        cost = round(distance_km * TRANSIT_COST_PER_KM, 1)
    else:
        mode = "taxi"
        cost = round(TAXI_BASE_FARE + distance_km * TAXI_COST_PER_KM, 1)

    minutes = estimate_travel_minutes(distance_km, mode)
    return {
        "mode": mode,
        "mode_cn": {"walk": "步行", "transit": "地铁/公交", "taxi": "打车"}[mode],
        "minutes": minutes,
        "cost": cost,
        "distance_km": round(distance_km, 3),
    }
