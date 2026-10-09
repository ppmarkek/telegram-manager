"""Связывает всё вместе: чат с ботом — единственный интерфейс, аккаунт — источник данных."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import anthropic
from telethon import Button, TelegramClient, events, utils
from telethon.errors import RPCError
from telethon.tl import functions, types

from . import userbot
from .agy import AgyAI, AgyRunner, find_agy, risky_permission_mode
from .ai import AI, AIError, ClaudeAI, ModelInfo
from .config import ConfigError, Settings, parse_digest_time, validate_timezone
from .folders import (
    ResolvedPlan,
    format_chats_for_ai,
    format_plan_for_ai,
    preview_html,
    resolve_plan,
)
from .storage import InboxItem, Storage, Task
from .timeutil import (
    describe_now,
    human,
    local_dt,
    local_hm,
    now_ts,
    parse_local,
    to_local_str,
    tomorrow_at,
)

log = logging.getLogger(__name__)

BTN_FOLDERS = "📁 Разложить чаты"
BTN_TASKS = "📝 Задачи"
BTN_DIGEST = "☀️ Сводка"
BTN_MODEL = "🧠 Модель"
BTN_HELP = "❓ Помощь"

MAIN_KEYBOARD = [
    [Button.text(BTN_FOLDERS, resize=True), Button.text(BTN_TASKS, resize=True)],
    [Button.text(BTN_DIGEST, resize=True), Button.text(BTN_MODEL, resize=True)],
    [Button.text(BTN_HELP, resize=True)],
]

HELP_TEXT = """\
<b>Привет! Я твой помощник в Telegram.</b>

Всё работает прямо в этом чате:

📁 <b>Разложить чаты</b> — предложу папки по смыслу. Ничего не меняю, пока не нажмёшь «Применить».
📝 <b>Задачи</b> — список дел и напоминаний.
☀️ <b>Сводка</b> — что важного в непрочитанном и что на сегодня.
🧠 <b>Модель</b> — выбрать, какой моделью я думаю. О новых моделях сообщу сам.

Можно писать обычным текстом:
• <i>напомни завтра в 10 позвонить маме</i>
• <i>перенеси задачу 3 на пятницу</i>
• <i>найди, где Саша скидывал адрес</i>
• <i>объедини папки учёба и курсы</i>
• <i>присылай сводку в 8:30</i>
• <i>переключись на gemini pro</i>

Ещё я сам читаю новые сообщения в личках (и в группах, где тебя упомянули) \
и предлагаю задачи: «ты обещал скинуть отчёт до пятницы». Сохраняю только то, что ты подтвердишь."""

REMINDER_LOOP_SEC = 20
DIGEST_LOOP_SEC = 30
DIGEST_WINDOW_MIN = 60  # если бот был выключен в момент сводки, догоняем в течение часа
INBOX_BATCH = 300
MESSAGE_LIMIT = 4000
MODELS_CACHE_SEC = 600
FIRST_UPDATE_CHECK_SEC = 60


@dataclass
class PendingPlan:
    id: str
    resolved: ResolvedPlan
    replace_existing: bool


def split_text(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


def parse_callback(data: bytes) -> tuple[str, str, str]:
    parts = data.decode(errors="ignore").split(":", 2)
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def format_inbox_for_ai(items: list[InboxItem], tz: str) -> str:
    lines = []
    for number, item in enumerate(items, start=1):
        when = local_dt(item.date, tz).strftime("%d.%m %H:%M")
        lines.append(f"[{number}] {when} | чат «{item.chat_title}» | {item.sender}: {item.text}")
    return "\n".join(lines)


def minutes(hm: str) -> int:
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


def make_ai(settings: Settings, model: str | None) -> AI:
    if settings.ai_backend == "claude":
        return ClaudeAI(model)
    runner = AgyRunner(
        settings.agy_bin, settings.agy_workdir, settings.agy_timeout_min, settings.agy_sandbox
    )
    return AgyAI(model, runner)


def resolve_model(models: list[ModelInfo], query: str) -> tuple[ModelInfo | None, list[ModelInfo]]:
    """Ищет модель по id или названию: точное совпадение, потом единственное частичное."""
    q = " ".join(query.lower().split())
    for model in models:
        if q in (model.id.lower(), model.name.lower()):
            return model, []
    words = q.replace("-", " ").split()
    candidates = [
        m for m in models if all(w in f"{m.id} {m.name}".lower().replace("-", " ") for w in words)
    ]
    if len(candidates) == 1:
        return candidates[0], []
    return None, candidates


def model_button_data(model: ModelInfo, index: int) -> bytes:
    """Callback-данные кнопки модели: id модели, а если он не влезает в 64 байта — номер в списке."""
    data = f"m:id:{model.id}".encode()
    return data if len(data) <= 64 else f"m:ix:{index}".encode()


class App:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = Storage(settings.db_path)
        stored_model = self.storage.get(f"model:{settings.ai_backend}")
        self.ai = make_ai(settings, stored_model or settings.model)
        self.models_cache: list[ModelInfo] = []
        self.models_cached_at = 0.0
        self.user = TelegramClient(str(settings.user_session), settings.api_id, settings.api_hash)
        self.bot = TelegramClient(str(settings.bot_session), settings.api_id, settings.api_hash)
        self.bot.parse_mode = "html"
        self.owner_id = 0
        self.bot_id = 0
        self.premium = False
        self.pending_plan: PendingPlan | None = None
        self.folder_lock = asyncio.Lock()
        self.history: deque[str] = deque(maxlen=8)

    # --- настройки, которые можно менять из чата ---

    @property
    def tz(self) -> str:
        return self.storage.get("timezone") or self.settings.timezone

    def digest_time(self) -> str | None:
        stored = self.storage.get("digest_time")
        if stored is None:
            return self.settings.digest_time
        return None if stored == "off" else stored

    # --- запуск ---

    async def run(self) -> None:
        warnings = self.startup_warnings()
        for warning in warnings:
            log.warning(warning)
        await self.user.connect()
        if not await self.user.is_user_authorized():
            raise ConfigError("Аккаунт не подключён. Сначала выполни: python -m tgmanager login")
        me = await self.user.get_me()
        self.owner_id = me.id
        self.premium = bool(getattr(me, "premium", False))

        await self.bot.start(bot_token=self.settings.bot_token)
        bot_me = await self.bot.get_me()
        self.bot_id = bot_me.id
        await self._set_commands()

        self.bot.add_event_handler(
            self.on_bot_message, events.NewMessage(incoming=True, func=lambda e: e.is_private)
        )
        self.bot.add_event_handler(self.on_callback, events.CallbackQuery())
        self.user.add_event_handler(self.on_user_message, events.NewMessage())

        log.info(
            "Запущено: аккаунт %s, бот @%s", utils.get_display_name(me), bot_me.username
        )
        greeting = "👋 Я на связи. Нажми «❓ Помощь», чтобы узнать, что я умею."
        if warnings:
            greeting += "\n\n" + "\n\n".join(f"⚠️ {w}" for w in warnings)
        try:
            await self.bot.send_message(self.owner_id, html.escape(greeting), buttons=MAIN_KEYBOARD)
        except (RPCError, ValueError):
            log.warning("Открой @%s в Telegram и нажми Start — тогда я смогу тебе писать.", bot_me.username)

        jobs = [
            self.user.disconnected,
            self.bot.disconnected,
            self._loop(self.check_reminders, REMINDER_LOOP_SEC),
            self._loop(self.check_digest, DIGEST_LOOP_SEC),
            self._loop(self.scan_inbox, self.settings.scan_interval_min * 60),
        ]
        if self.settings.update_hours:
            jobs.append(
                self._loop(self.check_updates, self.settings.update_hours * 3600, FIRST_UPDATE_CHECK_SEC)
            )
        await asyncio.gather(*jobs)

    def startup_warnings(self) -> list[str]:
        warnings = []
        if self.ai.backend == "agy":
            if not find_agy(self.settings.agy_bin):
                warnings.append(
                    "Не нашёл agy. Установи Antigravity CLI и войди в аккаунт, иначе ИИ работать не будет."
                )
            if risky_permission_mode():
                warnings.append(
                    "В настройках agy включено toolPermission: always-proceed — агент выполняет команды "
                    "без спроса. Раз я передаю ему тексты чужих сообщений, лучше вернуть request-review "
                    "в ~/.gemini/antigravity-cli/settings.json."
                )
        elif not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            warnings.append("ANTHROPIC_API_KEY не задан в .env — ИИ работать не будет.")
        return warnings

    async def _set_commands(self) -> None:
        commands = [
            types.BotCommand("tasks", "Мои задачи"),
            types.BotCommand("folders", "Разложить чаты по папкам"),
            types.BotCommand("digest", "Сводка непрочитанного"),
            types.BotCommand("model", "Выбрать модель ИИ"),
            types.BotCommand("update", "Проверить обновления и новые модели"),
            types.BotCommand("help", "Что я умею"),
        ]
        try:
            await self.bot(
                functions.bots.SetBotCommandsRequest(
                    scope=types.BotCommandScopeDefault(), lang_code="", commands=commands
                )
            )
        except RPCError as exc:
            log.warning("Не удалось установить команды бота: %s", exc)

    @staticmethod
    async def _loop(job: Any, interval: int, first_delay: int | None = None) -> None:
        delay = interval if first_delay is None else first_delay
        while True:
            await asyncio.sleep(delay)
            delay = interval
            try:
                await job()
            except Exception:
                log.exception("Ошибка в фоновой задаче %s", getattr(job, "__name__", job))

    # --- отправка владельцу ---

    async def send_html(self, text: str, buttons: Any = None) -> Any:
        chunks = split_text(text)
        message = None
        for i, chunk in enumerate(chunks):
            message = await self.bot.send_message(
                self.owner_id, chunk, buttons=buttons if i == len(chunks) - 1 else None
            )
        return message

    async def send_text(self, text: str, buttons: Any = None) -> Any:
        return await self.send_html(html.escape(text), buttons=buttons)

    # --- сообщения в чате с ботом ---

    async def on_bot_message(self, event: events.NewMessage.Event) -> None:
        if event.sender_id != self.owner_id:
            await event.respond("Это личный помощник, он работает только для своего владельца.")
            return
        text = (event.raw_text or "").strip()
        command = text.split()[0].split("@")[0].lower() if text.startswith("/") else ""
        try:
            if command in ("/start", "/help") or text == BTN_HELP:
                await self.send_html(HELP_TEXT, buttons=MAIN_KEYBOARD)
            elif command == "/folders" or text == BTN_FOLDERS:
                await self.propose_folders()
            elif command == "/tasks" or text == BTN_TASKS:
                await self.send_tasks()
            elif command == "/digest" or text == BTN_DIGEST:
                await self.send_digest()
            elif command == "/model" or text == BTN_MODEL:
                await self.send_model_picker()
            elif command == "/update":
                await self.send_text("🔄 Проверяю обновления…")
                await self.check_updates(manual=True)
            elif text:
                if event.message.fwd_from:
                    text = "Владелец переслал сообщение:\n" + text
                await self.chat(text)
        except AIError as exc:
            await self.send_text(f"⚠️ {exc}")
        except anthropic.APIConnectionError:
            await self.send_text("⚠️ Не могу связаться с ИИ. Проверь интернет и попробуй ещё раз.")
        except anthropic.AuthenticationError:
            await self.send_text("⚠️ ИИ не принял ключ. Проверь ANTHROPIC_API_KEY в .env.")
        except anthropic.APIStatusError as exc:
            log.exception("Ошибка API Claude")
            await self.send_text(f"⚠️ ИИ временно недоступен (код {exc.status_code}). Попробуй позже.")
        except Exception:
            log.exception("Ошибка обработки сообщения")
            await self.send_text("⚠️ Что-то пошло не так. Подробности в логе.")

    async def chat(self, text: str) -> None:
        context = [f"Сейчас: {describe_now(self.tz)}", "", "Активные задачи:", self.tasks_for_ai() or "нет"]
        if self.pending_plan:
            context += ["", "Владельцу показан план папок, он ещё не применён."]
        if self.history:
            context += ["", "Недавний разговор:", *self.history]
        context += ["", "Новое сообщение владельца:", text]

        async with self.bot.action(self.owner_id, "typing"):
            reply = await self.ai.assistant("\n".join(context), self.execute_tool)
        self.history.append(f"Владелец: {text[:500]}")
        self.history.append(f"Помощник: {reply[:500]}")
        await self.send_text(reply)

    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        tz = self.tz
        if name == "add_task":
            title = str(args.get("title") or "").strip()
            if not title:
                return "Ошибка: пустое название задачи."
            due = self._parse_due(args.get("due"))
            task = self.storage.add_task(title, due)
            when = human(due, tz) if due else "без срока"
            return f"Создана задача #{task.id}: {title} ({when})."
        if name == "list_tasks":
            return self.tasks_for_ai() or "Активных задач нет."
        if name == "update_task":
            return self._update_task(args)
        if name == "sort_chats":
            wishes = args.get("wishes") or None
            return await self.propose_folders(wishes, bool(args.get("replace_existing")))
        if name == "send_digest":
            await self.send_digest()
            return "Сводка отправлена владельцу."
        if name == "search_messages":
            query = str(args.get("query") or "").strip()
            if not query:
                return "Ошибка: пустой запрос."
            found = await userbot.search_messages(
                self.user, self.settings, query, self.owner_id, self.bot_id
            )
            return "\n".join(found) if found else "Ничего не найдено."
        if name == "update_settings":
            return await self._update_settings(args)
        return f"Неизвестный инструмент {name}."

    def _parse_due(self, value: Any) -> int | None:
        if not value:
            return None
        try:
            return parse_local(str(value), self.tz)
        except ValueError:
            raise ValueError(f"не понимаю время {value!r}, нужен формат YYYY-MM-DDTHH:MM") from None

    def _update_task(self, args: dict[str, Any]) -> str:
        task = self.storage.get_task(int(args.get("id", 0)))
        if task is None or task.status not in ("active", "suggested"):
            return "Ошибка: нет активной задачи с таким номером."
        action = args.get("action")
        if action == "done":
            self.storage.set_status(task.id, "done")
            return f"Задача #{task.id} отмечена выполненной."
        if action == "delete":
            self.storage.set_status(task.id, "dismissed")
            return f"Задача #{task.id} удалена."
        if action == "reschedule":
            due = self._parse_due(args.get("due"))
            self.storage.reschedule(task.id, due)
            return f"Задача #{task.id} перенесена: {human(due, self.tz) if due else 'без срока'}."
        if action == "rename":
            title = str(args.get("title") or "").strip()
            if not title:
                return "Ошибка: пустое название."
            self.storage.rename(task.id, title)
            return f"Задача #{task.id} переименована."
        return "Ошибка: неизвестное действие."

    async def _update_settings(self, args: dict[str, Any]) -> str:
        done = []
        if args.get("model"):
            models = await self.get_models()
            found, candidates = resolve_model(models, str(args["model"]))
            if found is None:
                options = candidates or models
                names = ", ".join(f"{m.name} ({m.id})" for m in options[:15])
                return f"Ошибка: не понял, какую модель выбрать. Варианты: {names}"
            self.set_model(found.id)
            done.append(f"модель {found.name}")
        try:
            if args.get("timezone"):
                tz = validate_timezone(str(args["timezone"]).strip())
                self.storage.set("timezone", tz)
                done.append(f"часовой пояс {tz}")
            if args.get("digest_time"):
                value = parse_digest_time(str(args["digest_time"]))
                self.storage.set("digest_time", value or "off")
                done.append(f"сводка {'в ' + value if value else 'выключена'}")
        except ConfigError as exc:
            return f"Ошибка: {exc}"
        return "Сохранено: " + ", ".join(done) if done else "Ничего не изменено."

    # --- задачи ---

    def tasks_for_ai(self) -> str:
        tz = self.tz
        lines = []
        for task in self.storage.open_tasks():
            due = f"напомнить {to_local_str(task.due_at, tz)}" if task.due_at else "без срока"
            lines.append(f"#{task.id} {task.title} ({due})")
        return "\n".join(lines)

    def tasks_message(self) -> tuple[str, Any]:
        tasks = self.storage.open_tasks()
        if not tasks:
            return "📝 Задач нет. Напиши, например: «напомни завтра в 10 позвонить маме».", None
        tz = self.tz
        now = now_ts()
        lines = ["<b>📝 Твои задачи</b>", ""]
        for task in tasks:
            if task.due_at:
                mark = "🔴" if task.due_at < now else "⏰"
                when = f"{mark} {human(task.due_at, tz, now)}"
            else:
                when = "без срока"
            lines.append(f"<b>#{task.id}</b> {html.escape(task.title)} — {when}")
        lines += ["", "Нажми на номер, чтобы отметить выполненной."]
        shown = tasks[:12]
        buttons = [
            [Button.inline(f"✅ #{t.id}", f"t:ldone:{t.id}".encode()) for t in shown[i : i + 4]]
            for i in range(0, len(shown), 4)
        ]
        return "\n".join(lines), buttons

    async def send_tasks(self) -> None:
        text, buttons = self.tasks_message()
        await self.send_html(text, buttons=buttons)

    def _source_html(self, task: Task) -> str:
        if not task.source_chat:
            return ""
        snippet = (task.source_text or "")[:200]
        return f"\n<i>Чат «{html.escape(task.source_chat)}»: {html.escape(snippet)}</i>"

    async def check_reminders(self) -> None:
        tz = self.tz
        for task in self.storage.due_tasks(now_ts()):
            text = f"⏰ <b>Напоминание</b>\n{html.escape(task.title)}" + self._source_html(task)
            buttons = [
                [
                    Button.inline("✅ Готово", f"t:done:{task.id}".encode()),
                    Button.inline("⏰ +1 час", f"t:hour:{task.id}".encode()),
                ],
                [Button.inline("🌅 Завтра в 9:00", f"t:tomorrow:{task.id}".encode())],
            ]
            await self.send_html(text, buttons=buttons)
            self.storage.mark_notified(task.id)
            log.info("Напоминание #%s отправлено (%s)", task.id, human(task.due_at or 0, tz))

    async def scan_inbox(self) -> None:
        items = self.storage.peek_inbox(INBOX_BATCH)
        if not items:
            return
        tz = self.tz
        found = await self.ai.extract_tasks(
            format_inbox_for_ai(items, tz), describe_now(tz), self.tasks_for_ai()
        )
        for candidate in found.tasks:
            if not 1 <= candidate.message <= len(items) or not candidate.title.strip():
                continue
            source = items[candidate.message - 1]
            try:
                due = parse_local(candidate.due, tz) if candidate.due else None
            except ValueError:
                due = None
            task = self.storage.add_task(
                candidate.title.strip(),
                due,
                status="suggested",
                source_chat=source.chat_title,
                source_text=source.text[:300],
            )
            when = human(due, tz) if due else "без срока"
            text = (
                f"🆕 <b>Похоже, есть задача</b>\n{html.escape(task.title)}\n⏰ {when}"
                + self._source_html(task)
            )
            buttons = [
                [
                    Button.inline("✅ Сохранить", f"s:ok:{task.id}".encode()),
                    Button.inline("✖️ Не надо", f"s:no:{task.id}".encode()),
                ]
            ]
            await self.send_html(text, buttons=buttons)
        self.storage.delete_inbox([item.id for item in items])
        log.info("Разобрано сообщений: %d, найдено задач: %d", len(items), len(found.tasks))

    # --- сводка ---

    async def send_digest(self) -> None:
        tz = self.tz
        async with self.bot.action(self.owner_id, "typing"):
            unread = await userbot.collect_unread(self.user, self.settings, self.owner_id, self.bot_id)
            tasks = self.tasks_for_ai()
            if not unread.chats and not tasks:
                note = ""
                if unread.skipped_channels or unread.skipped_muted:
                    note = " (есть непрочитанное только в каналах и заглушённых чатах)"
                await self.send_text(f"📭 Важного непрочитанного нет, задач тоже{note}.")
                return
            text = await self.ai.digest(userbot.format_unread(unread), tasks, describe_now(tz))
        await self.send_text("☀️ Сводка\n\n" + text)

    async def check_digest(self) -> None:
        digest_time = self.digest_time()
        if not digest_time:
            return
        tz = self.tz
        now_min = minutes(local_hm(tz))
        start = minutes(digest_time)
        today = local_dt(now_ts(), tz).strftime("%Y-%m-%d")
        if start <= now_min < start + DIGEST_WINDOW_MIN and self.storage.get("last_digest") != today:
            self.storage.set("last_digest", today)
            await self.send_digest()

    # --- папки ---

    async def propose_folders(self, wishes: str | None = None, replace_existing: bool = False) -> str:
        if self.folder_lock.locked():
            return "Уже работаю над папками, подожди немного."
        async with self.folder_lock:
            status = await self.send_text("🔍 Собираю список чатов…")
            try:
                return await self._propose_folders(status, wishes, replace_existing)
            except AIError as exc:
                await status.edit(html.escape(f"⚠️ {exc}"))
                return f"Ошибка: {exc}"
            except anthropic.APIError as exc:
                log.exception("Ошибка API Claude при раскладке")
                await status.edit("⚠️ ИИ сейчас недоступен, попробуй позже.")
                return f"Ошибка API: {exc}"
            except Exception as exc:
                log.exception("Ошибка при раскладке по папкам")
                await status.edit(html.escape(f"⚠️ Не получилось: {exc}"))
                return f"Ошибка: {exc}"

    async def _propose_folders(self, status: Any, wishes: str | None, replace_existing: bool) -> str:
        async def progress(count: int) -> None:
            try:
                await status.edit(f"🔍 Собираю список чатов… {count}")
            except RPCError:
                pass

        chats = await userbot.collect_chats(
            self.user, self.settings, self.owner_id, self.bot_id, progress
        )
        if not chats:
            await status.edit("Не нашёл чатов для раскладки.")
            return "Чатов не найдено."

        state = await userbot.folder_state(self.user, self.storage.managed_folders(), self.premium)
        slots = state.free_slots(replace_existing)
        if slots == 0:
            names = ", ".join(userbot.filter_title(f) for f in state.user_folders)
            await status.edit(
                html.escape(
                    f"У тебя уже {state.max_folders} папок из {state.max_folders} возможных ({names}). "
                    "Удали лишние в настройках Telegram или напиши мне «можешь заменить мои папки»."
                )
            )
            return "Нет свободных мест для папок."

        idx_by_peer = {c.peer_id: c.idx for c in chats}
        current = None
        if self.pending_plan:
            current = format_plan_for_ai(self.pending_plan.resolved, idx_by_peer)
        elif state.managed:
            current = self._managed_plan_for_ai(state, idx_by_peer)

        await status.edit("🤖 Думаю, как лучше разложить…")
        user_titles = [] if replace_existing else [userbot.filter_title(f) for f in state.user_folders]
        plan = await self.ai.plan_folders(
            format_chats_for_ai(chats), slots, user_titles, current, wishes
        )
        resolved = resolve_plan(plan, chats, slots, state.max_chats)
        if not resolved.folders:
            await status.edit("ИИ не смог предложить папки. Попробуй ещё раз или опиши, какие папки нужны.")
            return "План не получился."

        plan_id = secrets.token_hex(4)
        self.pending_plan = PendingPlan(plan_id, resolved, replace_existing)
        text = preview_html(resolved)
        if len(text) > MESSAGE_LIMIT:
            text = preview_html(resolved, shown_per_folder=2)[:MESSAGE_LIMIT]
        if replace_existing and state.user_folders:
            names = ", ".join(userbot.filter_title(f) for f in state.user_folders)
            text += f"\n\n⚠️ Твои текущие папки будут удалены: {html.escape(names)}. Сами чаты останутся."
        buttons = [
            [Button.inline("✅ Применить", f"f:apply:{plan_id}".encode())],
            [Button.inline("✖️ Отмена", f"f:cancel:{plan_id}".encode())],
        ]
        await status.edit(text[: MESSAGE_LIMIT + 200], buttons=buttons)
        return f"План показан владельцу: папок {len(resolved.folders)}. Ждём нажатия «Применить»."

    @staticmethod
    def _managed_plan_for_ai(state: userbot.FolderState, idx_by_peer: dict[int, int]) -> str:
        lines = []
        for f in state.filters:
            if f.id not in state.managed:
                continue
            ids = [idx_by_peer.get(utils.get_peer_id(p)) for p in f.include_peers]
            ids_text = ", ".join(str(i) for i in ids if i)
            lines.append(f"{f.emoticon or ''} {state.managed[f.id]}: {ids_text}".strip())
        return "\n".join(lines)

    async def apply_plan(self, event: events.CallbackQuery.Event, plan_id: str) -> None:
        plan = self.pending_plan
        if plan is None or plan.id != plan_id:
            await event.answer("Этот план уже неактуален.", alert=True)
            return
        if self.folder_lock.locked():
            await event.answer("Подожди, я ещё работаю над папками.", alert=True)
            return
        await event.answer("Применяю…")
        async with self.folder_lock:
            state = await userbot.folder_state(self.user, self.storage.managed_folders(), self.premium)
            created, errors = await userbot.apply_folders(
                self.user, state, plan.resolved.folders, plan.replace_existing
            )
        self.storage.set_managed_folders(created)
        self.pending_plan = None
        text = f"✅ Готово! Папок создано или обновлено: {len(created)}."
        if errors:
            text += "\n\n⚠️ Не всё получилось:\n" + "\n".join(errors[:10])
        await event.edit(html.escape(text), buttons=None)

    # --- модели и обновления ---

    def _model_key(self) -> str:
        return f"model:{self.ai.backend}"

    def chosen_model(self) -> str | None:
        """Модель, выбранная в чате (None — по умолчанию)."""
        return self.storage.get(self._model_key()) or None

    def set_model(self, model_id: str | None) -> None:
        self.storage.set(self._model_key(), model_id or "")
        self.ai.model = model_id or self.settings.model

    async def get_models(self, refresh: bool = False) -> list[ModelInfo]:
        fresh = time.monotonic() - self.models_cached_at < MODELS_CACHE_SEC
        if refresh or not self.models_cache or not fresh:
            self.models_cache = await self.ai.list_models()
            self.models_cached_at = time.monotonic()
        return self.models_cache

    def model_title(self, model_id: str | None) -> str:
        if not model_id:
            return "по умолчанию"
        for model in self.models_cache:
            if model.id == model_id:
                return model.name
        return model_id

    async def model_picker(self, refresh: bool = False) -> tuple[str, Any]:
        models = await self.get_models(refresh)
        chosen = self.chosen_model()
        default = self.settings.model
        if self.ai.backend == "agy":
            default_label = f"По умолчанию ({self.model_title(default)})" if default else "По умолчанию agy"
        else:
            default_label = f"По умолчанию ({self.model_title(default or self.ai.model)})"
        lines = [
            f"🧠 <b>Модель ИИ</b> — {html.escape(self.ai.backend_title)}",
            f"Сейчас: <b>{html.escape(self.model_title(chosen) if chosen else default_label)}</b>",
            "",
            "Выбери модель. Лимиты у каждой модели свои: если одна упрётся в лимит, переключись на другую.",
        ]
        buttons = [
            [Button.inline(("✅ " if m.id == chosen else "") + m.name[:56], model_button_data(m, i))]
            for i, m in enumerate(models[:40])
        ]
        buttons.append(
            [
                Button.inline(("✅ " if not chosen else "") + default_label[:40], b"m:default:0"),
                Button.inline("🔄 Обновить список", b"m:refresh:0"),
            ]
        )
        return "\n".join(lines), buttons

    async def send_model_picker(self) -> None:
        async with self.bot.action(self.owner_id, "typing"):
            text, buttons = await self.model_picker()
        await self.send_html(text, buttons=buttons)

    async def _model_button(self, event: Any, action: str, value: str) -> None:
        if action == "ix":
            index = int(value) if value.isdigit() else -1
            if not 0 <= index < len(self.models_cache):
                await event.answer("Список устарел, открой «🧠 Модель» заново.", alert=True)
                return
            action, value = "id", self.models_cache[index].id
        if action == "id":
            self.set_model(value)
            name = self.model_title(value)
            await event.answer(f"Модель: {name}")
            await event.edit(f"🧠 Теперь думаю моделью <b>{html.escape(name)}</b>.", buttons=None)
        elif action == "default":
            self.set_model(None)
            await event.answer("Модель по умолчанию")
            await event.edit("🧠 Вернул модель по умолчанию.", buttons=None)
        elif action == "refresh":
            await event.answer("Обновляю список…")
            text, buttons = await self.model_picker(refresh=True)
            await event.edit(text, buttons=buttons)

    async def check_updates(self, manual: bool = False) -> None:
        """Обновляет agy и сравнивает список моделей с прошлым. О новых моделях пишет сам."""
        ai = self.ai
        before = await ai.version()
        update_error = None
        try:
            await ai.update()
        except AIError as exc:
            update_error = str(exc)
            log.warning("Не удалось обновить %s: %s", ai.backend_title, exc)
        after = await ai.version() if ai.backend == "agy" else before
        models = await self.get_models(refresh=True)
        ids = [m.id for m in models]

        key = f"known_models:{ai.backend}"
        known_raw = self.storage.get(key)
        self.storage.set(key, json.dumps(ids))
        known = set(json.loads(known_raw)) if known_raw else None
        new = [m for m in models if known is not None and m.id not in known]

        lines: list[str] = []
        if before and after and before != after:
            lines.append(f"🔄 agy обновился: {before} → {after}")
        if new:
            lines.append("🆕 <b>Появились новые модели:</b>")
            lines += [f"• {html.escape(m.name)} <code>{html.escape(m.id)}</code>" for m in new]
            lines.append("Нажми, чтобы переключиться:")
        chosen = self.chosen_model()
        if chosen and chosen not in ids:
            self.set_model(None)
            lines.append(f"⚠️ Модель {html.escape(chosen)} больше недоступна — вернул модель по умолчанию.")

        if manual:
            if not lines:
                version = f", версия agy {after}" if after else ""
                lines.append(f"✅ Всё актуально{version}. Моделей доступно: {len(models)}.")
            if update_error:
                lines.append(f"⚠️ Обновить не получилось: {html.escape(update_error)}")
        if lines:
            buttons = [
                [Button.inline(m.name[:60], model_button_data(m, models.index(m)))] for m in new[:10]
            ]
            await self.send_html("\n".join(lines), buttons=buttons or None)

    # --- кнопки ---

    async def on_callback(self, event: events.CallbackQuery.Event) -> None:
        if event.sender_id != self.owner_id:
            await event.answer("Нет доступа.")
            return
        kind, action, ident = parse_callback(event.data)
        try:
            if kind == "f":
                if action == "apply":
                    await self.apply_plan(event, ident)
                else:
                    if self.pending_plan and self.pending_plan.id == ident:
                        self.pending_plan = None
                    await event.edit("Отменено. Папки не менял.", buttons=None)
            elif kind == "m":
                await self._model_button(event, action, ident)
            elif kind in ("t", "s") and ident.isdigit():
                await self._task_button(event, kind, action, int(ident))
            else:
                await event.answer()
        except Exception:
            log.exception("Ошибка обработки кнопки")
            await event.answer("Что-то пошло не так.", alert=True)

    async def _task_button(self, event: Any, kind: str, action: str, task_id: int) -> None:
        task = self.storage.get_task(task_id)
        if task is None or task.status not in ("active", "suggested"):
            await event.answer("Эта задача уже неактуальна.")
            return
        tz = self.tz
        title = html.escape(task.title)
        if kind == "s":
            if action == "ok":
                self.storage.set_status(task.id, "active")
                when = human(task.due_at, tz) if task.due_at else "без срока"
                await event.edit(f"✅ Сохранил задачу #{task.id}: {title} ({when})", buttons=None)
            else:
                self.storage.set_status(task.id, "dismissed")
                await event.edit(f"✖️ <s>{title}</s>", buttons=None)
            return
        if action == "ldone":  # кнопка в списке задач — перерисовываем список
            self.storage.set_status(task.id, "done")
            await event.answer(f"#{task.id} выполнена!")
            text, buttons = self.tasks_message()
            await event.edit(text, buttons=buttons)
        elif action == "done":  # кнопка под напоминанием
            self.storage.set_status(task.id, "done")
            await event.edit(f"✅ <s>{title}</s>", buttons=None)
        elif action == "hour":
            self.storage.reschedule(task.id, now_ts() + 3600)
            await event.edit(f"⏰ {title}\nНапомню через час.", buttons=None)
        elif action == "tomorrow":
            due = tomorrow_at(tz, 9)
            self.storage.reschedule(task.id, due)
            await event.edit(f"⏰ {title}\nНапомню {human(due, tz)}.", buttons=None)

    # --- новые сообщения в аккаунте: материал для поиска задач ---

    async def on_user_message(self, event: events.NewMessage.Event) -> None:
        try:
            message = event.message
            text = (message.message or "").strip()
            if not text:
                return
            chat_id = event.chat_id
            if chat_id in (self.owner_id, self.bot_id, userbot.TELEGRAM_SERVICE_ID):
                return
            if event.is_channel and not event.is_group:
                return  # каналы-рассылки
            if event.is_group:
                mode = self.settings.groups_mode
                if mode == "off" or (mode == "mentions" and not (message.out or message.mentioned)):
                    return
            chat = await event.get_chat()
            if self.settings.is_excluded(chat_id, getattr(chat, "username", None)):
                return
            if event.is_private and getattr(chat, "bot", False):
                return
            if message.out:
                sender = "Я"
            else:
                sender = utils.get_display_name(await event.get_sender()) or "?"
            self.storage.add_inbox(
                chat_id,
                utils.get_display_name(chat) or "Без названия",
                sender,
                bool(message.out),
                text[:1000],
                int(message.date.timestamp()),
            )
        except Exception:
            log.exception("Не удалось сохранить входящее сообщение")
