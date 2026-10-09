"""Настройки из переменных окружения / файла .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

DEFAULT_MODEL = "claude-opus-5-5"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    api_id: int
    api_hash: str
    bot_token: str
    data_dir: Path
    timezone: str
    model: str
    digest_time: str | None  # "09:00" или None — без утренней сводки
    scan_interval_min: int
    groups_mode: str  # mentions | all | off
    excluded_chats: frozenset[str]
    folder_context_messages: int
    max_chats_for_sort: int

    @property
    def user_session(self) -> Path:
        return self.data_dir / "user"

    @property
    def bot_session(self) -> Path:
        return self.data_dir / "bot"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tgmanager.db"

    def is_excluded(self, chat_id: int, username: str | None) -> bool:
        if str(chat_id) in self.excluded_chats:
            return True
        return bool(username) and username.lower() in self.excluded_chats


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Не задана переменная {name} (см. .env.example)")
    return value


def _required_int(name: str) -> int:
    raw = _required(name)
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} должно быть числом, а не {raw!r}") from None


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} должно быть числом, а не {raw!r}") from None


def parse_digest_time(raw: str) -> str | None:
    raw = raw.strip().lower()
    if raw in ("", "off", "no", "нет", "выкл"):
        return None
    try:
        hours, minutes = raw.split(":")
        h, m = int(hours), int(minutes)
    except ValueError:
        raise ConfigError(f"Время сводки должно быть в формате ЧЧ:ММ, а не {raw!r}") from None
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ConfigError(f"Некорректное время сводки: {raw!r}")
    return f"{h:02d}:{m:02d}"


def validate_timezone(name: str) -> str:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ConfigError(f"Неизвестный часовой пояс: {name!r} (пример: Europe/Moscow)") from None
    return name


def parse_excluded(raw: str) -> frozenset[str]:
    items = (part.strip().lower().lstrip("@") for part in raw.split(","))
    return frozenset(item for item in items if item)


def load_settings(require_bot: bool = True) -> Settings:
    load_dotenv()

    groups_mode = os.getenv("GROUPS_MODE", "mentions").strip().lower()
    if groups_mode not in ("mentions", "all", "off"):
        raise ConfigError("GROUPS_MODE должен быть mentions, all или off")

    data_dir = Path(os.getenv("DATA_DIR", "data")).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        api_id=_required_int("TG_API_ID"),
        api_hash=_required("TG_API_HASH"),
        bot_token=_required("BOT_TOKEN") if require_bot else os.getenv("BOT_TOKEN", ""),
        data_dir=data_dir,
        timezone=validate_timezone(os.getenv("TIMEZONE", "Europe/Moscow").strip()),
        model=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        digest_time=parse_digest_time(os.getenv("DIGEST_TIME", "09:00")),
        scan_interval_min=max(1, _int("SCAN_INTERVAL_MIN", 10)),
        groups_mode=groups_mode,
        excluded_chats=parse_excluded(os.getenv("EXCLUDED_CHATS", "")),
        folder_context_messages=max(1, _int("FOLDER_CONTEXT_MESSAGES", 3)),
        max_chats_for_sort=max(10, _int("MAX_CHATS_FOR_SORT", 400)),
    )
