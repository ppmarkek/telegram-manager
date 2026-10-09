"""agy-бэкенд: разбор вывода, схемы и запуск настоящего процесса (поддельного agy)."""

import asyncio
import json
import stat
import sys
from pathlib import Path

import pytest

from tgmanager.agy import (
    AGENT_STEP_SCHEMA,
    AgyAI,
    AgyRunner,
    agy_schema,
    explain_failure,
    extract_json,
    parse_json_output,
    parse_models,
    parse_stream_output,
    risky_permission_mode,
)
from tgmanager.ai import AIError, FolderPlan, FoundTasks

MODELS = """\
gemini-3.8-flash-high Gemini 3.8 Flash (High)
gemini-3.8-flash-low Gemini 3.8 Flash (Low)
gemini-3.1-pro-high Gemini 3.1 Pro (High)
claude-sonnet-4-6     Claude Sonnet 4.6
gpt-oss-120b"""


@pytest.fixture
def fake_agy(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    (state / "models.txt").write_text(MODELS, encoding="utf-8")
    (state / "version.txt").write_text("1.1.27")
    script = tmp_path / "agy"
    script.write_text(
        f"#!{sys.executable}\n" + (Path(__file__).parent / "fake_agy.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_AGY_DIR", str(state))

    def respond(*envelopes):
        (state / "responses.json").write_text(json.dumps(list(envelopes), ensure_ascii=False), encoding="utf-8")

    def calls():
        path = state / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    runner = AgyRunner(str(script), tmp_path / "workspace", timeout_min=1, sandbox=True)
    return runner, respond, calls


def test_parse_models():
    models = parse_models("Available models:\nSLUG  NAME\n" + MODELS + "\n\n")
    assert [m.id for m in models] == [
        "gemini-3.8-flash-high",
        "gemini-3.8-flash-low",
        "gemini-3.1-pro-high",
        "claude-sonnet-4-6",
        "gpt-oss-120b",
    ]
    assert models[0].name == "Gemini 3.8 Flash (High)"
    assert models[3].name == "Claude Sonnet 4.6"
    assert models[4].name == "gpt-oss-120b"
    assert parse_models("\x1b[32m* gemini-3.8-flash-high\x1b[0m  Gemini 3.8 Flash (High)")[0].id == "gemini-3.8-flash-high"


def test_parse_outputs():
    envelope = {"status": "SUCCESS", "response": "hi", "structured_output": {"a": 1}}
    result = parse_json_output("some log line\n" + json.dumps(envelope))
    assert (result.status, result.response, result.structured) == ("SUCCESS", "hi", {"a": 1})
    assert parse_json_output("not json") is None

    stream = "\n".join(
        [
            json.dumps({"event": "init"}),
            "garbage",
            json.dumps({"event": "result", "result": {"status": "ERROR", "error": {"message": "quota"}}}),
        ]
    )
    result = parse_stream_output(stream)
    assert result.status == "ERROR" and "quota" in result.error


def test_agy_schema_is_simple():
    schema = agy_schema(FolderPlan)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and '"title": "' not in text.replace('"title": {', "")
    folder = schema["properties"]["folders"]["items"]
    assert folder["properties"]["title"] == {"type": "string"}
    assert folder["required"] == ["title", "emoji", "chats"]

    task = agy_schema(FoundTasks)["properties"]["tasks"]["items"]
    assert task["properties"]["due"] == {"type": "string"}
    assert task["required"] == ["message", "title"]  # due стал необязательным, null убран
    # А модель всё равно принимает ответ без due.
    parsed = FoundTasks.model_validate({"tasks": [{"message": 1, "title": "x"}]})
    assert parsed.tasks[0].due is None


def test_extract_json_and_errors():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Вот ответ: {"a": 2} — готово') == {"a": 2}
    assert extract_json({"a": 3}) == {"a": 3}
    with pytest.raises(AIError):
        extract_json("никакого json")
    assert "не авторизован" in explain_failure("Error: authentication required", 1)
    assert "лимиты" in explain_failure("RESOURCE_EXHAUSTED: quota exceeded", 1)
    assert "модель" in explain_failure("unknown model: foo", 1)
    assert "код 3" in explain_failure("boom", 3)


def test_risky_permission_mode(tmp_path):
    path = tmp_path / "settings.json"
    assert risky_permission_mode(path) is None
    path.write_text('{"toolPermission": "request-review"}')
    assert risky_permission_mode(path) is None
    path.write_text('{"toolPermission": "always-proceed"}')
    assert risky_permission_mode(path) == "always-proceed"


def test_runner_short_prompt_uses_args(fake_agy):
    runner, respond, calls = fake_agy
    respond({"status": "SUCCESS", "response": "", "structured_output": {"answer": 4}})
    result = asyncio.run(runner.run("Сколько 2+2?", model="gemini-3.1-pro-high", schema={"type": "object"}))
    assert result.structured == {"answer": 4}
    call = calls()[0]
    args = call["args"]
    assert args[:2] == ["-p", "Сколько 2+2?"]
    assert args[args.index("--output-format") + 1] == "json"
    assert args[args.index("--model") + 1] == "gemini-3.1-pro-high"
    assert json.loads(args[args.index("--json-schema") + 1]) == {"type": "object"}
    assert args[args.index("--print-timeout") + 1] == "1m"
    assert "--sandbox" in args
    assert call["stdin"] == ""
    assert Path(call["cwd"]).name == "workspace"


def test_runner_long_prompt_uses_stdin(fake_agy):
    runner, respond, calls = fake_agy
    respond({"status": "SUCCESS", "response": "готово"})
    prompt = "чат " * 40_000
    result = asyncio.run(runner.run(prompt))
    assert result.response == "готово"
    call = calls()[0]
    assert call["args"][:2] == ["-p", ""]
    assert call["args"][call["args"].index("--input-format") + 1] == "stream-json"
    assert call["args"][call["args"].index("--output-format") + 1] == "stream-json"
    assert "--model" not in call["args"]
    sent = json.loads(call["stdin"])
    assert sent == {"event": "user", "message": {"content": prompt}}


def test_runner_errors(fake_agy):
    runner, respond, _ = fake_agy
    respond({"status": "ERROR", "error": "Quota exceeded for model", "_exit": 1})
    with pytest.raises(AIError, match="лимиты"):
        asyncio.run(runner.run("привет"))

    missing = AgyRunner("/nonexistent/agy-binary", runner.workdir, 1, False)
    with pytest.raises(AIError, match="Не нашёл agy"):
        asyncio.run(missing.run("привет"))


def test_models_version_update(fake_agy):
    runner, _, calls = fake_agy
    models = asyncio.run(runner.models())
    assert len(models) == 5
    assert asyncio.run(runner.version()) == "1.1.27"
    asyncio.run(runner.update())
    assert asyncio.run(runner.version()) == "1.2.0"
    assert [c["args"][0] for c in calls()] == ["models", "--version", "update", "--version"]


def test_agy_structured_and_text(fake_agy):
    runner, respond, calls = fake_agy
    ai = AgyAI("gemini-3.8-flash-high", runner)
    respond(
        # structured_output может не прийти — тогда берём JSON из текста ответа
        {"status": "SUCCESS", "response": '```json\n{"tasks": [{"message": 1, "title": "Скинуть отчёт"}]}\n```'},
        {"status": "SUCCESS", "response": "☀️ Всё спокойно"},
    )
    found = asyncio.run(ai.extract_tasks("[1] ...", "сейчас", ""))
    assert found.tasks[0].title == "Скинуть отчёт" and found.tasks[0].due is None
    assert asyncio.run(ai.digest("", "", "сейчас")) == "☀️ Всё спокойно"

    first = calls()[0]["args"]
    assert "не используй никакие инструменты" in first[1]
    schema = json.loads(first[first.index("--json-schema") + 1])
    assert schema["properties"]["tasks"]["items"]["required"] == ["message", "title"]
    assert "--json-schema" not in calls()[1]["args"]


def test_agy_assistant_runs_actions(fake_agy):
    runner, respond, calls = fake_agy
    ai = AgyAI(None, runner)
    respond(
        {"status": "SUCCESS", "structured_output": {"actions": [{"tool": "list_tasks"}], "reply": ""}},
        {
            "status": "SUCCESS",
            "structured_output": {
                "actions": [
                    {"tool": "list_tasks"},  # повтор — не выполняем второй раз
                    {"tool": "update_task", "id": 3, "action": "done"},
                ],
                "reply": "Отметил задачу #3 выполненной.",
            },
        },
    )
    executed = []

    async def execute(name, args):
        executed.append((name, {k: v for k, v in args.items() if v is not None}))
        return "#3 Позвонить маме (без срока)" if name == "list_tasks" else "Задача #3 отмечена выполненной."

    reply = asyncio.run(ai.assistant("отметь звонок маме", execute))
    assert reply == "Отметил задачу #3 выполненной."
    assert executed == [("list_tasks", {}), ("update_task", {"id": 3, "action": "done"})]
    second_prompt = calls()[1]["args"][1]
    assert "- list_tasks: #3 Позвонить маме" in second_prompt
    schema = json.loads(calls()[0]["args"][calls()[0]["args"].index("--json-schema") + 1])
    assert schema == AGENT_STEP_SCHEMA


def test_agy_assistant_plain_reply(fake_agy):
    runner, respond, calls = fake_agy
    ai = AgyAI(None, runner)
    respond({"status": "SUCCESS", "structured_output": {"actions": [], "reply": "Привет!"}})

    async def execute(name, args):
        raise AssertionError("не должно вызываться")

    assert asyncio.run(ai.assistant("привет", execute)) == "Привет!"
    assert len(calls()) == 1
