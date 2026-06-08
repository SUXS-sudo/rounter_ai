"""Time helper functions based on minutes from midnight."""

from __future__ import annotations

import re
from typing import Any


DAY_MINUTES = 24 * 60
TIME_PATTERN = re.compile(r"^(\d{1,2}):([0-5]\d)$")


def parse_hhmm(value: str) -> int:
    """Parse ``HH:MM`` into minutes from midnight.

    Args:
        value: A time string such as ``"09:30"`` or ``"21:00"``.

    Returns:
        Integer minutes since 00:00, ranging from 0 to 1439.

    Raises:
        ValueError: If the input is not a valid 24-hour ``HH:MM`` string.
    """

    if not isinstance(value, str):
        raise ValueError("time value must be a string in HH:MM format")

    match = TIME_PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"invalid HH:MM time: {value}")

    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23:
        raise ValueError(f"invalid HH:MM time: {value}")

    return hour * 60 + minute


def format_hhmm(minutes_from_midnight: int) -> str:
    """Format minutes from midnight as ``HH:MM``.

    Values outside one day wrap around with modulo 24 hours, so ``1500`` becomes
    ``"01:00"``.
    """

    minutes = int(minutes_from_midnight) % DAY_MINUTES
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def add_minutes(time_str: str, minutes: int) -> str:
    """Add minutes to a ``HH:MM`` string and return a wrapped ``HH:MM`` string."""

    return format_hhmm(parse_hhmm(time_str) + minutes)


def minutes_between(start: str, end: str) -> int:
    """Return forward minutes from ``start`` to ``end``.

    If ``end`` is earlier than ``start``, the calculation assumes the interval
    crosses midnight.
    """

    start_minutes = parse_hhmm(start)
    end_minutes = parse_hhmm(end)
    delta = end_minutes - start_minutes
    if delta < 0:
        delta += DAY_MINUTES
    return delta


def is_open_at(poi: dict[str, Any], arrival_time: str) -> bool:
    """Check whether a POI is open at the given arrival time.

    Args:
        poi: Dictionary containing ``open_time`` and ``close_time``.
        arrival_time: Arrival time in ``HH:MM``.

    Returns:
        ``True`` when the arrival time falls inside the opening window. Overnight
        windows such as ``18:00`` to ``02:00`` are supported.
    """

    open_minutes, close_minutes, arrival_minutes = _parse_visit_times(poi, arrival_time)

    if open_minutes == close_minutes:
        return True

    if _is_overnight_window(open_minutes, close_minutes):
        return arrival_minutes >= open_minutes or arrival_minutes <= close_minutes

    return open_minutes <= arrival_minutes <= close_minutes


def can_visit_poi(poi: dict[str, Any], arrival_time: str, stay_minutes: int) -> tuple[bool, str]:
    """Decide whether a POI can be visited at the given arrival time.

    If the user arrives before opening time, waiting until opening is allowed.
    The visit is rejected when the final leave time exceeds closing time.

    Args:
        poi: Dictionary containing at least ``open_time`` and ``close_time``.
        arrival_time: Arrival time in ``HH:MM``.
        stay_minutes: Planned stay duration in minutes.

    Returns:
        ``(True, reason)`` if the visit fits the business window, otherwise
        ``(False, clear_failure_reason)``.
    """

    if stay_minutes <= 0:
        return False, "停留时间必须大于0分钟。"

    try:
        open_minutes, close_minutes, arrival_minutes = _parse_visit_times(poi, arrival_time)
    except (KeyError, ValueError) as exc:
        return False, f"时间数据无效：{exc}。"

    if open_minutes == close_minutes:
        leave_minutes = arrival_minutes + stay_minutes
        return True, f"可访问，预计{_format_absolute_minutes(leave_minutes)}离开。"

    window_start, window_end, adjusted_arrival = _resolve_relevant_window(
        open_minutes,
        close_minutes,
        arrival_minutes,
    )

    if adjusted_arrival > window_end:
        return (
            False,
            f"到达时间{arrival_time}已晚于闭店时间{_format_absolute_minutes(window_end)}，无法访问。",
        )

    visit_start = max(adjusted_arrival, window_start)
    wait_minutes = visit_start - adjusted_arrival
    leave_minutes = visit_start + stay_minutes

    if leave_minutes > window_end:
        wait_text = ""
        if wait_minutes > 0:
            wait_text = f"可等待{wait_minutes}分钟至{_format_absolute_minutes(visit_start)}开门，但"
        return (
            False,
            (
                f"到达时间{arrival_time}{'早于开门时间，' if wait_minutes > 0 else ''}"
                f"{wait_text}停留{stay_minutes}分钟会在{_format_absolute_minutes(leave_minutes)}离开，"
                f"超过闭店时间{_format_absolute_minutes(window_end)}。"
            ),
        )

    if wait_minutes > 0:
        return (
            True,
            (
                f"到达时间{arrival_time}早于开门时间{_format_absolute_minutes(window_start)}，"
                f"可等待{wait_minutes}分钟，预计{_format_absolute_minutes(leave_minutes)}离开。"
            ),
        )

    return True, f"可访问，预计{_format_absolute_minutes(leave_minutes)}离开。"


def is_time_between(target: str, start: str, end: str) -> bool:
    """Backward-compatible helper for checking a time inside a time window."""

    target_minutes = parse_hhmm(target)
    start_minutes = parse_hhmm(start)
    end_minutes = parse_hhmm(end)
    if start_minutes == end_minutes:
        return True
    if _is_overnight_window(start_minutes, end_minutes):
        return target_minutes >= start_minutes or target_minutes <= end_minutes
    return start_minutes <= target_minutes <= end_minutes


def _parse_visit_times(poi: dict[str, Any], arrival_time: str) -> tuple[int, int, int]:
    try:
        open_time = poi["open_time"]
        close_time = poi["close_time"]
    except KeyError as exc:
        raise KeyError("POI缺少open_time或close_time") from exc

    return parse_hhmm(str(open_time)), parse_hhmm(str(close_time)), parse_hhmm(arrival_time)


def _is_overnight_window(open_minutes: int, close_minutes: int) -> bool:
    return close_minutes < open_minutes


def _resolve_relevant_window(
    open_minutes: int,
    close_minutes: int,
    arrival_minutes: int,
) -> tuple[int, int, int]:
    if not _is_overnight_window(open_minutes, close_minutes):
        return open_minutes, close_minutes, arrival_minutes

    if arrival_minutes <= close_minutes:
        return open_minutes - DAY_MINUTES, close_minutes, arrival_minutes

    return open_minutes, close_minutes + DAY_MINUTES, arrival_minutes


def _format_absolute_minutes(minutes: int) -> str:
    day_offset, minute_of_day = divmod(minutes, DAY_MINUTES)
    time_text = format_hhmm(minute_of_day)
    if day_offset == 0:
        return time_text
    if day_offset == 1:
        return f"次日{time_text}"
    if day_offset == -1:
        return f"前一日{time_text}"
    if day_offset > 1:
        return f"{day_offset}日后{time_text}"
    return f"{abs(day_offset)}日前{time_text}"
