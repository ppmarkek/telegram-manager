"""Точка входа: `python -m tgmanager login` один раз, затем `python -m tgmanager`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

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


def main() -> None:
    parser = argparse.ArgumentParser(prog="tgmanager", description="ИИ-помощник для Telegram")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "login"])
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
