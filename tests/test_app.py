"""Сценарии бота без Telegram и без ИИ: подменяем отправку сообщений и ответы модели."""

import asyncio
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tgmanager import app as app_module
from tgmanager.ai import FoundTask, FoundTasks, ModelInfo
from tgmanager.app import App, parse_callback, resolve_model, split_text
from tgmanager.config import Settings

TZ = "Europe/Moscow"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        api_id=1,
        api_hash="hash",
        bot_token="token",
        data_dir=tmp_path,
        timezone=TZ,
        ai_backend="agy",
        model=None,
        agy_bin="agy",
        agy_workdir=tmp_path / "agy-workspace",
        agy_timeout_min=1,
        agy_sandbox=True,
        update_hours=12,
        digest_time="09:00",
        scan_interval_min=10,
        groups_mode="mentions",
        excluded_chats=frozenset(),
        folder_context_messages=3,
        max_chats_for_sort=400,
    )


class FakeAI:
    def __init__(self, found=None):
        self.found = found or FoundTasks(tasks=[])
        self.calls = []

    async def extract_tasks(self, messages_text, now_text, open_tasks):
        self.calls.append(messages_text)
        return self.found


class FakeEvent:
    def __init__(self, data: bytes, sender_id: int = 42):
        self.data = data
        self.sender_id = sender_id
        self.answers = []
        self.edits = []

    async def answer(self, text=None, alert=False):
        self.answers.append(text)

    async def edit(self, text, buttons=None):
        self.edits.append((text, buttons))


@pytest.fixture
def bot_app(tmp_path):
    app = App(make_settings(tmp_path))
    app.owner_id = 42
    app.sent = []

    async def fake_send_html(text, buttons=None):
        app.sent.append((text, buttons))

    app.send_html = fake_send_html
    yield app
    app.storage.close()


def data(button) -> bytes:
    return button.type.data


def ts(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=ZoneInfo(TZ)).timestamp())


def test_helpers():
    assert parse_callback(b"t:done:12") == ("t", "done", "12")
    assert parse_callback(b"junk") == ("", "", "")
    chunks = split_text("а" * 3000 + "\n" + "б" * 3000, limit=4000)
    assert [len(c) for c in chunks] == [3000, 3000]


def test_tools_manage_tasks(bot_app):
    run = asyncio.run
    reply = run(bot_app.execute_tool("add_task", {"title": "Позвонить маме", "due": "2026-10-10T10:00"}))
    assert reply.startswith("Создана задача #1: Позвонить маме")
    task = bot_app.storage.get_task(1)
    assert task.due_at == ts("2026-10-10T10:00")

    run(bot_app.execute_tool("add_task", {"title": "Купить хлеб", "due": None}))
    assert "#2 Купить хлеб (без срока)" in run(bot_app.execute_tool("list_tasks", {}))

    run(bot_app.execute_tool("update_task", {"id": 1, "action": "reschedule", "due": "2026-10-11T09:00", "title": None}))
    assert bot_app.storage.get_task(1).due_at == ts("2026-10-11T09:00")

    run(bot_app.execute_tool("update_task", {"id": 2, "action": "done", "due": None, "title": None}))
    assert bot_app.storage.get_task(2).status == "done"
    assert "нет активной задачи" in run(
        bot_app.execute_tool("update_task", {"id": 2, "action": "done", "due": None, "title": None})
    )

    with pytest.raises(ValueError):
        run(bot_app.execute_tool("add_task", {"title": "x", "due": "в пятницу"}))


def test_tool_settings(bot_app):
    run = asyncio.run
    out = run(bot_app.execute_tool("update_settings", {"timezone": "Europe/Vilnius", "digest_time": "8:30"}))
    assert out == "Сохранено: часовой пояс Europe/Vilnius, сводка в 08:30"
    assert bot_app.tz == "Europe/Vilnius"
    assert bot_app.digest_time() == "08:30"
    run(bot_app.execute_tool("update_settings", {"timezone": None, "digest_time": "off"}))
    assert bot_app.digest_time() is None
    assert "Ошибка" in run(bot_app.execute_tool("update_settings", {"timezone": "Mars/Base", "digest_time": None}))


def test_scan_inbox_suggests_tasks(bot_app):
    bot_app.storage.add_inbox(7, "Иван", "Иван", False, "Скинь отчёт до пятницы", ts("2026-10-09T12:00"))
    bot_app.storage.add_inbox(7, "Иван", "Я", True, "Ок, скину", ts("2026-10-09T12:01"))
    bot_app.ai = FakeAI(
        FoundTasks(
            tasks=[
                FoundTask(message=2, title="Скинуть Ивану отчёт", due="2026-10-10T10:00"),
                FoundTask(message=99, title="Мусор", due=None),  # несуществующий номер
            ]
        )
    )
    asyncio.run(bot_app.scan_inbox())

    assert "[2] 09.10 12:01 | чат «Иван» | Я: Ок, скину" in bot_app.ai.calls[0]
    assert bot_app.storage.peek_inbox(10) == []
    task = bot_app.storage.get_task(1)
    assert task.status == "suggested" and task.source_chat == "Иван"
    assert bot_app.storage.get_task(2) is None
    text, buttons = bot_app.sent[0]
    assert "Скинуть Ивану отчёт" in text
    assert data(buttons[0][0]) == b"s:ok:1"

    event = FakeEvent(b"s:ok:1")
    asyncio.run(bot_app.on_callback(event))
    assert bot_app.storage.get_task(1).status == "active"
    assert event.edits[0][0].startswith("✅ Сохранил задачу #1")


def test_reminders_and_buttons(bot_app, monkeypatch):
    task = bot_app.storage.add_task("Позвонить маме", ts("2026-10-09T10:00"))
    monkeypatch.setattr(app_module, "now_ts", lambda: ts("2026-10-09T10:00"))
    asyncio.run(bot_app.check_reminders())
    asyncio.run(bot_app.check_reminders())  # повторно не напоминаем
    assert len(bot_app.sent) == 1
    assert [data(b) for row in bot_app.sent[0][1] for b in row] == [
        b"t:done:1", b"t:hour:1", b"t:tomorrow:1",
    ]

    asyncio.run(bot_app.on_callback(FakeEvent(b"t:hour:1")))
    assert bot_app.storage.get_task(task.id).due_at == ts("2026-10-09T11:00")
    assert bot_app.storage.get_task(task.id).notified is False

    asyncio.run(bot_app.on_callback(FakeEvent(b"t:done:1")))
    assert bot_app.storage.get_task(task.id).status == "done"

    stranger = FakeEvent(b"t:done:1", sender_id=7)
    asyncio.run(bot_app.on_callback(stranger))
    assert stranger.answers == ["Нет доступа."]


def test_task_list_button_redraws_list(bot_app):
    bot_app.storage.add_task("Первая", None)
    bot_app.storage.add_task("Вторая", None)
    event = FakeEvent(b"t:ldone:1")
    asyncio.run(bot_app.on_callback(event))
    text, buttons = event.edits[0]
    assert "Первая" not in text and "Вторая" in text
    assert data(buttons[0][0]) == b"t:ldone:2"


def test_digest_window(bot_app, monkeypatch):
    sent = []

    async def fake_digest():
        sent.append(True)

    bot_app.send_digest = fake_digest
    for moment, expected in [
        ("2026-10-09T08:59", 0),
        ("2026-10-09T09:10", 1),
        ("2026-10-09T09:20", 1),  # уже отправили сегодня
        ("2026-10-10T12:00", 1),  # бот был выключен утром — не шлём в обед
        ("2026-10-11T09:00", 2),
    ]:
        monkeypatch.setattr(app_module, "now_ts", lambda m=moment: ts(m))
        monkeypatch.setattr(app_module, "local_hm", lambda tz, m=moment: m[11:16])
        asyncio.run(bot_app.check_digest())
        assert len(sent) == expected, moment


class FakeModelsAI:
    backend = "agy"
    backend_title = "agy"

    def __init__(self):
        self.model = None
        self.models = [ModelInfo("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)")]
        self.versions = ["1.1.27", "1.1.27"]
        self.updated = 0

    async def version(self):
        return self.versions.pop(0) if len(self.versions) > 1 else self.versions[0]

    async def update(self):
        self.updated += 1

    async def list_models(self):
        return list(self.models)


def test_resolve_model():
    models = [
        ModelInfo("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
        ModelInfo("gemini-3.8-flash-low", "Gemini 3.8 Flash (Low)"),
        ModelInfo("gemini-3.1-pro-high", "Gemini 3.1 Pro (High)"),
    ]
    assert resolve_model(models, "gemini-3.1-pro-high")[0].id == "gemini-3.1-pro-high"
    assert resolve_model(models, "Gemini 3.8 Flash (Low)")[0].id == "gemini-3.8-flash-low"
    assert resolve_model(models, "gemini pro")[0].id == "gemini-3.1-pro-high"
    match, candidates = resolve_model(models, "flash")
    assert match is None and len(candidates) == 2


def test_model_choice_from_button_and_tool(bot_app):
    bot_app.ai = FakeModelsAI()
    bot_app.ai.models.append(ModelInfo("gemini-3.1-pro-high", "Gemini 3.1 Pro (High)"))

    text, buttons = asyncio.run(bot_app.model_picker())
    assert "Сейчас: <b>По умолчанию agy</b>" in text
    assert data(buttons[1][0]) == b"m:id:gemini-3.1-pro-high"

    event = FakeEvent(b"m:id:gemini-3.1-pro-high")
    asyncio.run(bot_app.on_callback(event))
    assert bot_app.ai.model == "gemini-3.1-pro-high"
    assert bot_app.storage.get("model:agy") == "gemini-3.1-pro-high"
    assert "Gemini 3.1 Pro (High)" in event.edits[0][0]

    out = asyncio.run(bot_app.execute_tool("update_settings", {"timezone": None, "digest_time": None, "model": "flash"}))
    assert out == "Сохранено: модель Gemini 3.8 Flash (High)"
    assert bot_app.ai.model == "gemini-3.8-flash-high"
    assert "Ошибка" in asyncio.run(
        bot_app.execute_tool("update_settings", {"timezone": None, "digest_time": None, "model": "gpt-9"})
    )

    asyncio.run(bot_app.on_callback(FakeEvent(b"m:default:0")))
    assert bot_app.ai.model is None and bot_app.chosen_model() is None


def test_model_persists_between_restarts(tmp_path):
    first = App(make_settings(tmp_path))
    first.set_model("gemini-3.1-pro-high")
    first.storage.close()
    second = App(make_settings(tmp_path))
    assert second.ai.model == "gemini-3.1-pro-high"
    second.storage.close()


def test_check_updates_notifies_about_new_models(bot_app):
    ai = FakeModelsAI()
    bot_app.ai = ai

    asyncio.run(bot_app.check_updates())  # первый раз только запоминаем список
    assert bot_app.sent == [] and ai.updated == 1

    ai.models.append(ModelInfo("gemini-4.0-pro-high", "Gemini 4.0 Pro (High)"))
    ai.versions = ["1.1.27", "1.2.0"]
    asyncio.run(bot_app.check_updates())
    text, buttons = bot_app.sent[-1]
    assert "agy обновился: 1.1.27 → 1.2.0" in text
    assert "Gemini 4.0 Pro (High)" in text
    assert data(buttons[0][0]) == b"m:id:gemini-4.0-pro-high"

    bot_app.sent.clear()
    asyncio.run(bot_app.check_updates())  # ничего нового — молчим
    assert bot_app.sent == []

    bot_app.set_model("gemini-4.0-pro-high")
    ai.models.pop()  # выбранная модель исчезла
    asyncio.run(bot_app.check_updates(manual=True))
    assert "больше недоступна" in bot_app.sent[-1][0]
    assert bot_app.chosen_model() is None

    bot_app.sent.clear()
    asyncio.run(bot_app.check_updates(manual=True))
    assert "Всё актуально, версия agy 1.2.0" in bot_app.sent[-1][0]


def test_long_model_id_uses_index(bot_app):
    long_id = "vendor/" + "x" * 70 + "-1"
    bot_app.ai = FakeModelsAI()
    bot_app.ai.models = [ModelInfo(long_id, "Длинная модель")]
    text, buttons = asyncio.run(bot_app.model_picker())
    assert data(buttons[0][0]) == b"m:ix:0"
    asyncio.run(bot_app.on_callback(FakeEvent(b"m:ix:0")))
    assert bot_app.chosen_model() == long_id
    stale = FakeEvent(b"m:ix:5")
    asyncio.run(bot_app.on_callback(stale))
    assert "устарел" in stale.answers[0]
