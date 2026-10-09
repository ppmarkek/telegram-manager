"""Работа со временем: всё хранится в unix-времени (UTC), показывается в часовом поясе владельца."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

WEEKDAYS_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
WEEKDAYS_FULL = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
MONTHS = ["янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]

LOCAL_FORMAT = "%Y-%m-%dT%H:%M"


def now_ts() -> int:
    return int(time.time())


def local_dt(ts: int, tz: str) -> datetime:
    return datetime.fromtimestamp(ts, ZoneInfo(tz))


def describe_now(tz: str, ts: int | None = None) -> str:
    """Строка о текущем моменте для ИИ: «2026-10-09 15:30, пятница (Europe/Moscow)»."""
    dt = local_dt(now_ts() if ts is None else ts, tz)
    return f"{dt:%Y-%m-%d %H:%M}, {WEEKDAYS_FULL[dt.weekday()]} (часовой пояс {tz})"


def parse_local(value: str, tz: str) -> int:
    """'YYYY-MM-DDTHH:MM' в часовом поясе владельца -> unix-время."""
    value = value.strip().replace(" ", "T")
    dt = datetime.strptime(value[:16], LOCAL_FORMAT)
    return int(dt.replace(tzinfo=ZoneInfo(tz)).timestamp())


def to_local_str(ts: int, tz: str) -> str:
    return local_dt(ts, tz).strftime(LOCAL_FORMAT)


def human(ts: int, tz: str, now: int | None = None) -> str:
    """«сегодня в 15:00», «завтра в 10:00», «пт, 17 окт в 18:00»."""
    dt = local_dt(ts, tz)
    today = local_dt(now_ts() if now is None else now, tz).date()
    hm = f"{dt:%H:%M}"
    delta = (dt.date() - today).days
    if delta == 0:
        return f"сегодня в {hm}"
    if delta == 1:
        return f"завтра в {hm}"
    if delta == -1:
        return f"вчера в {hm}"
    year = f" {dt.year}" if dt.year != today.year else ""
    return f"{WEEKDAYS_SHORT[dt.weekday()]}, {dt.day} {MONTHS[dt.month - 1]}{year} в {hm}"


def tomorrow_at(tz: str, hour: int = 9, now: int | None = None) -> int:
    dt = local_dt(now_ts() if now is None else now, tz)
    target = (dt + timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0)
    return int(target.timestamp())


def local_date_str(tz: str, now: int | None = None) -> str:
    return local_dt(now_ts() if now is None else now, tz).strftime("%Y-%m-%d")


def local_hm(tz: str, now: int | None = None) -> str:
    return local_dt(now_ts() if now is None else now, tz).strftime("%H:%M")
