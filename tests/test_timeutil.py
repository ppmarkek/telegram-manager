from datetime import datetime
from zoneinfo import ZoneInfo

from tgmanager.timeutil import describe_now, human, parse_local, to_local_str, tomorrow_at

TZ = "Europe/Moscow"


def ts(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=ZoneInfo(TZ)).timestamp())


def test_parse_local_roundtrip():
    value = parse_local("2026-10-09T15:30", TZ)
    assert value == ts("2026-10-09T15:30")
    assert to_local_str(value, TZ) == "2026-10-09T15:30"


def test_parse_local_accepts_space_and_seconds():
    assert parse_local("2026-10-09 15:30:00", TZ) == ts("2026-10-09T15:30")


def test_human_relative_days():
    now = ts("2026-10-09T12:00")
    assert human(ts("2026-10-09T15:00"), TZ, now) == "сегодня в 15:00"
    assert human(ts("2026-10-10T09:05"), TZ, now) == "завтра в 09:05"
    assert human(ts("2026-10-08T20:00"), TZ, now) == "вчера в 20:00"
    assert human(ts("2026-10-17T18:00"), TZ, now) == "сб, 17 окт в 18:00"
    assert human(ts("2027-01-05T10:00"), TZ, now) == "вт, 5 янв 2027 в 10:00"


def test_tomorrow_at():
    now = ts("2026-10-09T23:30")
    assert tomorrow_at(TZ, 9, now) == ts("2026-10-10T09:00")


def test_describe_now():
    assert describe_now(TZ, ts("2026-10-09T15:30")) == (
        "2026-10-09 15:30, пятница (часовой пояс Europe/Moscow)"
    )
