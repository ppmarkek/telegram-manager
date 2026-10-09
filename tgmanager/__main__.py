"""Точка входа.

    python -m tgmanager login   — один раз войти в аккаунт Telegram
    python -m tgmanager check   — проверить, что ИИ (agy или Claude) отвечает
    python -m tgmanager         — запустить помощника
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel
from telethon import TelegramClient, utils

from .config import ConfigError, Settings, load_settings


async def login(settings: Settings) -> None:
    client = TelegramClient(str(settings.user_session), settings.api_id, settings.api_hash)
    print("Вход в твой аккаунт Telegram. Код придёт в приложение Telegram.")
    await client.start()  # спросит номер телефона, код и пароль 2FA, если он есть
    me = await client.get_me()
    print(f"\nГотово! Вошли как {utils.get_display_name(me)}.")
    print("Файл сессии сохранён в", settings.user_session.with_suffix(".session"))
    print("Никому его не передавай: он даёт полный доступ к аккаунту.")
    print("\nТеперь запусти помощника: python -m tgmanager")
    await client.disconnect()


class _Ping(BaseModel):
    answer: int


async def check(settings: Settings) -> bool:
    from .agy import AgyAI, find_agy, risky_permission_mode
    from .app import make_ai
    from .storage import Storage

    storage = Storage(settings.db_path)
    model = storage.get(f"model:{settings.ai_backend}") or settings.model
    storage.close()
    ai = make_ai(settings, model)
    print(f"ИИ: {ai.backend_title}; модель: {model or 'по умолчанию'}")

    ok = True

    async def step(title: str, job: Awaitable[Any]) -> Any:
        nonlocal ok
        try:
            result = await job
        except Exception as exc:
            ok = False
            print(f"  ✗ {title}: {exc}")
            return None
        if isinstance(result, str):
            detail = result.strip()[:200]
        elif isinstance(result, list):
            detail = f"{len(result)} шт."
        elif isinstance(result, BaseModel):
            detail = result.model_dump_json()
        else:
            detail = ""
        print(f"  ✓ {title}" + (f": {detail}" if detail else ""))
        return result

    if isinstance(ai, AgyAI):
        print(f"  путь к agy: {find_agy(settings.agy_bin) or 'не найден'}")
        print(f"  рабочая папка agy: {settings.agy_workdir}")
        if risky_permission_mode():
            print("  ! в настройках agy включено toolPermission: always-proceed — лучше request-review")
        await step("версия agy", ai.version())

    models = await step("список моделей", ai.list_models())
    if models:
        for item in models:
            print(f"      {item.id:<40} {item.name}")

    await step("обычный ответ", ai._text("Ты проверяешь связь.", "Ответь одним словом: работает", "low"))
    ping = await step(
        "ответ в JSON",
        ai._structured("Ты калькулятор.", "Сколько будет 2+2? Число — в поле answer.", _Ping, "low", 1000),
    )
    if ping is not None and getattr(ping, "answer", None) != 4:
        print("  ! модель ответила странно, но формат правильный")
    if isinstance(ai, AgyAI):
        result = await step(
            "длинный запрос через stdin",
            ai.runner.run("Ответь одним словом: работает", model=ai.model, force_stream=True),
        )
        if result is not None:
            print(f"      ответ: {result.response.strip()[:100]}")

    print("\nВсё работает." if ok else "\nЕсть проблемы — см. строки с ✗.")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(prog="tgmanager", description="ИИ-помощник для Telegram")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "login", "check"])
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("telethon").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    try:
        if args.command == "login":
            asyncio.run(login(load_settings(require_bot=False)))
        elif args.command == "check":
            ok = asyncio.run(check(load_settings(require_telegram=False, require_bot=False)))
            sys.exit(0 if ok else 1)
        else:
            from .app import App

            asyncio.run(App(load_settings()).run())
    except ConfigError as exc:
        print(f"Ошибка настройки: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
