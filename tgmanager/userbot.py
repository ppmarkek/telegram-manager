"""Действия от имени аккаунта владельца (MTProto через Telethon).

Здесь только чтение чатов и управление папками. Писать кому-то или удалять
что-либо этот модуль не умеет — так задумано.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from telethon import TelegramClient, utils
from telethon.errors import RPCError
from telethon.tl import functions, types

from .config import Settings
from .folders import ChatInfo, ResolvedFolder, allocate_ids

log = logging.getLogger(__name__)

SNIPPET_LEN = 120
TELEGRAM_SERVICE_ID = 777000

DEFAULT_LIMITS = {
    "dialog_filters_limit_default": 10,
    "dialog_filters_limit_premium": 30,
    "dialog_filters_chats_limit_default": 100,
    "dialog_filters_chats_limit_premium": 200,
}

Progress = Callable[[int], Awaitable[None]]


def chat_kind(entity: Any) -> str:
    if isinstance(entity, types.User):
        return "бот" if entity.bot else "личный"
    if isinstance(entity, types.Channel) and entity.broadcast:
        return "канал"
    return "группа"


def media_label(message: Any) -> str:
    if message.photo:
        return "[фото]"
    if message.voice:
        return "[голосовое]"
    if message.video_note:
        return "[кружок]"
    if message.video:
        return "[видео]"
    if message.sticker:
        return "[стикер]"
    if message.document:
        return "[файл]"
    if message.poll:
        return "[опрос]"
    return "[сообщение]"


def message_snippet(message: Any, max_len: int = SNIPPET_LEN) -> str:
    text = " ".join((message.message or "").split())
    if not text:
        return media_label(message)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def sender_name(message: Any, me_id: int) -> str:
    if message.out or message.sender_id == me_id:
        return "Я"
    sender = message.sender
    return utils.get_display_name(sender) if sender else "?"


def _is_skipped(settings: Settings, entity: Any, me_id: int, bot_id: int) -> bool:
    peer_id = utils.get_peer_id(entity)
    if peer_id in (me_id, bot_id, TELEGRAM_SERVICE_ID):
        return True
    return settings.is_excluded(peer_id, getattr(entity, "username", None))


async def collect_chats(
    client: TelegramClient,
    settings: Settings,
    me_id: int,
    bot_id: int,
    progress: Progress | None = None,
) -> list[ChatInfo]:
    """Список неархивных чатов с последними сообщениями — материал для раскладки по папкам."""
    chats: list[ChatInfo] = []
    async for dialog in client.iter_dialogs(archived=False):
        if len(chats) >= settings.max_chats_for_sort:
            break
        entity = dialog.entity
        if _is_skipped(settings, entity, me_id, bot_id):
            continue
        snippets: list[str] = []
        if dialog.message is not None:
            snippets.append(message_snippet(dialog.message))
        if settings.folder_context_messages > 1:
            try:
                history = await client.get_messages(entity, limit=settings.folder_context_messages)
                snippets = [message_snippet(m) for m in reversed(history)]
            except RPCError as exc:
                log.debug("Не удалось прочитать историю %s: %s", dialog.name, exc)
            await asyncio.sleep(0.2)  # бережно к лимитам Telegram
        chats.append(
            ChatInfo(
                idx=len(chats) + 1,
                peer_id=dialog.id,
                title=dialog.name or "Без названия",
                kind=chat_kind(entity),
                username=getattr(entity, "username", None),
                snippets=snippets,
                input_peer=dialog.input_entity,
            )
        )
        if progress and len(chats) % 25 == 0:
            await progress(len(chats))
    return chats


def _json_to_py(value: Any) -> Any:
    if isinstance(value, types.JsonObject):
        return {item.key: _json_to_py(item.value) for item in value.value}
    if isinstance(value, types.JsonArray):
        return [_json_to_py(item) for item in value.value]
    if isinstance(value, types.JsonNull):
        return None
    return getattr(value, "value", None)


async def folder_limits(client: TelegramClient, premium: bool) -> tuple[int, int]:
    """(сколько можно папок, сколько чатов в одной папке) для этого аккаунта."""
    config = dict(DEFAULT_LIMITS)
    try:
        result = await client(functions.help.GetAppConfigRequest(hash=0))
        remote = _json_to_py(getattr(result, "config", None))
        if isinstance(remote, dict):
            for key in config:
                if isinstance(remote.get(key), (int, float)):
                    config[key] = int(remote[key])
    except RPCError as exc:
        log.warning("Не удалось получить лимиты папок, беру стандартные: %s", exc)
    suffix = "premium" if premium else "default"
    return config[f"dialog_filters_limit_{suffix}"], config[f"dialog_filters_chats_limit_{suffix}"]


@dataclass
class FolderState:
    filters: list[Any]  # DialogFilter и DialogFilterChatlist (без «Все чаты»)
    managed: dict[int, str]  # папки, созданные ботом и не изменённые пользователем
    user_folders: list[Any]  # обычные папки пользователя
    chatlists: list[Any]  # общие папки по ссылке — их никогда не трогаем
    max_folders: int
    max_chats: int

    def free_slots(self, replace_existing: bool) -> int:
        occupied = len(self.chatlists) + (0 if replace_existing else len(self.user_folders))
        return max(0, self.max_folders - occupied)


def filter_title(f: Any) -> str:
    title = f.title
    return title.text if isinstance(title, types.TextWithEntities) else str(title)


async def folder_state(client: TelegramClient, stored_managed: dict[int, str], premium: bool) -> FolderState:
    result = await client(functions.messages.GetDialogFiltersRequest())
    raw = getattr(result, "filters", result)
    filters = [f for f in raw if isinstance(f, (types.DialogFilter, types.DialogFilterChatlist))]
    managed: dict[int, str] = {}
    user_folders: list[Any] = []
    chatlists: list[Any] = []
    for f in filters:
        if isinstance(f, types.DialogFilterChatlist):
            chatlists.append(f)
        elif stored_managed.get(f.id) == filter_title(f):
            managed[f.id] = filter_title(f)
        else:
            # Если пользователь переименовал «нашу» папку — считаем её уже его папкой.
            user_folders.append(f)
    max_folders, max_chats = await folder_limits(client, premium)
    return FolderState(filters, managed, user_folders, chatlists, max_folders, max_chats)


async def apply_folders(
    client: TelegramClient,
    state: FolderState,
    folders: list[ResolvedFolder],
    replace_existing: bool,
) -> tuple[dict[int, str], list[str]]:
    """Создаёт/обновляет папки бота. Возвращает (новые папки бота, ошибки)."""
    errors: list[str] = []
    if replace_existing:
        for f in state.user_folders:
            try:
                await client(functions.messages.UpdateDialogFilterRequest(id=f.id))
            except RPCError as exc:
                errors.append(f"Не удалось удалить папку «{filter_title(f)}»: {exc}")

    taken = {f.id for f in state.filters}
    ids = allocate_ids(len(folders), list(state.managed), taken - set(state.managed))
    created: dict[int, str] = {}
    for folder_id, folder in zip(ids, folders, strict=True):
        dialog_filter = types.DialogFilter(
            id=folder_id,
            title=types.TextWithEntities(text=folder.title, entities=[]),
            pinned_peers=[],
            include_peers=[c.input_peer for c in folder.chats],
            exclude_peers=[],
            emoticon=folder.emoji,
        )
        try:
            await client(functions.messages.UpdateDialogFilterRequest(id=folder_id, filter=dialog_filter))
            created[folder_id] = folder.title
        except RPCError as exc:
            errors.append(f"Папка «{folder.title}»: {exc}")
        await asyncio.sleep(0.3)

    # Старые папки бота, которые больше не нужны.
    for folder_id in set(state.managed) - set(ids):
        try:
            await client(functions.messages.UpdateDialogFilterRequest(id=folder_id))
        except RPCError as exc:
            errors.append(f"Не удалось удалить старую папку: {exc}")
    return created, errors


@dataclass
class UnreadChat:
    title: str
    kind: str
    unread: int
    mentions: int
    messages: list[str]


@dataclass
class UnreadSummary:
    chats: list[UnreadChat]
    skipped_channels: int
    skipped_muted: int


def _is_muted(dialog: Any) -> bool:
    settings = getattr(dialog.dialog, "notify_settings", None)
    until = getattr(settings, "mute_until", None)
    if until is None:
        return False
    if isinstance(until, datetime):
        return until > datetime.now(timezone.utc)
    return int(until) > int(datetime.now(timezone.utc).timestamp())


async def collect_unread(
    client: TelegramClient,
    settings: Settings,
    me_id: int,
    bot_id: int,
    max_chats: int = 30,
    per_chat: int = 8,
) -> UnreadSummary:
    """Непрочитанное для сводки. Сообщения не помечаются прочитанными."""
    summary = UnreadSummary(chats=[], skipped_channels=0, skipped_muted=0)
    async for dialog in client.iter_dialogs(archived=False, limit=300):
        if not dialog.unread_count and not dialog.unread_mentions_count:
            continue
        entity = dialog.entity
        if _is_skipped(settings, entity, me_id, bot_id):
            continue
        kind = chat_kind(entity)
        if kind == "канал":
            summary.skipped_channels += 1
            continue
        if _is_muted(dialog) and not dialog.unread_mentions_count:
            summary.skipped_muted += 1
            continue
        if len(summary.chats) >= max_chats:
            break
        limit = max(1, min(dialog.unread_count or 1, per_chat))
        messages = await client.get_messages(entity, limit=limit)
        summary.chats.append(
            UnreadChat(
                title=dialog.name or "Без названия",
                kind=kind,
                unread=dialog.unread_count,
                mentions=dialog.unread_mentions_count,
                messages=[f"{sender_name(m, me_id)}: {message_snippet(m, 300)}" for m in reversed(messages)],
            )
        )
        await asyncio.sleep(0.2)
    return summary


def format_unread(summary: UnreadSummary) -> str:
    blocks = []
    for chat in summary.chats:
        head = f"Чат «{chat.title}» ({chat.kind}), непрочитанных: {chat.unread}"
        if chat.mentions:
            head += f", упоминаний: {chat.mentions}"
        blocks.append(head + "\n" + "\n".join(f"  {m}" for m in chat.messages))
    extra = []
    if summary.skipped_channels:
        extra.append(f"Ещё каналов с непрочитанным: {summary.skipped_channels}")
    if summary.skipped_muted:
        extra.append(f"Ещё заглушённых чатов с непрочитанным: {summary.skipped_muted}")
    return "\n\n".join(blocks + extra)


async def search_messages(
    client: TelegramClient, settings: Settings, query: str, me_id: int, bot_id: int, limit: int = 15
) -> list[str]:
    found: list[str] = []
    async for message in client.iter_messages(None, search=query, limit=limit * 2):
        chat = await message.get_chat()
        if chat is None or _is_skipped(settings, chat, me_id, bot_id):
            continue
        await message.get_sender()
        date = message.date.strftime("%Y-%m-%d %H:%M") if message.date else "?"
        found.append(
            f"{date} UTC | «{utils.get_display_name(chat)}» | "
            f"{sender_name(message, me_id)}: {message_snippet(message, 400)}"
        )
        if len(found) >= limit:
            break
    return found
