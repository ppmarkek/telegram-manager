"""Проверяем, какие запросы уходят в Claude API и как разбираются ответы (без сети)."""

import asyncio
import json

import anthropic
import httpx2
import pytest

from tgmanager.ai import ASSISTANT_TOOLS, AIError, ClaudeAI, FALLBACK_BETA


def message(content, stop_reason="end_turn"):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def make_ai(responses, model="claude-opus-5-5"):
    requests = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        requests.append({"headers": request.headers, "body": body, "url": str(request.url)})
        return httpx2.Response(200, json=responses.pop(0))

    client = anthropic.AsyncAnthropic(
        api_key="test",
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return ClaudeAI(model, client), requests


def test_extract_tasks_request_and_parse():
    payload = {"tasks": [{"message": 1, "title": "Скинуть отчёт", "due": "2026-10-10T10:00"}]}
    ai, requests = make_ai([message([{"type": "text", "text": json.dumps(payload)}])])
    result = asyncio.run(ai.extract_tasks("[1] ... Скинь отчёт", "сейчас", ""))

    assert result.tasks[0].title == "Скинуть отчёт"
    assert result.tasks[0].due == "2026-10-10T10:00"
    body = requests[0]["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in requests[0]["headers"]["anthropic-beta"]
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in body
    assert "Уже существующие задачи:\nнет" in body["messages"][0]["content"]


def test_no_fallback_for_haiku():
    ai, requests = make_ai([message([{"type": "text", "text": "Сводка"}])], model="claude-haiku-5-5")
    assert asyncio.run(ai.digest("чаты", "", "сейчас")) == "Сводка"
    assert "fallbacks" not in requests[0]["body"]


def test_refusal_raises():
    ai, _ = make_ai([message([], stop_reason="refusal")])
    with pytest.raises(AIError):
        asyncio.run(ai.digest("чаты", "", "сейчас"))


def test_assistant_runs_tools_until_done():
    responses = [
        message(
            [
                {"type": "text", "text": "Создаю."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "add_task",
                    "input": {"title": "Позвонить маме", "due": "2026-10-10T10:00"},
                },
            ],
            stop_reason="tool_use",
        ),
        message([{"type": "text", "text": "Напомню завтра в 10:00."}]),
    ]
    ai, requests = make_ai(responses)
    calls = []

    async def execute(name, args):
        calls.append((name, args))
        return "Создана задача #1"

    reply = asyncio.run(ai.assistant("напомни завтра в 10 позвонить маме", execute))
    assert reply == "Напомню завтра в 10:00."
    assert calls == [("add_task", {"title": "Позвонить маме", "due": "2026-10-10T10:00"})]

    second = requests[1]["body"]
    # История внутри хода только дополняется: user -> assistant(tool_use) -> user(tool_result).
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "Создана задача #1",
    }
    assert second["tools"] == requests[0]["body"]["tools"]


def test_tool_error_is_reported_to_model():
    responses = [
        message(
            [{"type": "tool_use", "id": "toolu_1", "name": "list_tasks", "input": {}}],
            stop_reason="tool_use",
        ),
        message([{"type": "text", "text": "Не вышло."}]),
    ]
    ai, requests = make_ai(responses)

    async def execute(name, args):
        raise ValueError("сломалось")

    assert asyncio.run(ai.assistant("покажи задачи", execute)) == "Не вышло."
    result = requests[1]["body"]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert "сломалось" in result["content"]


def test_tool_schemas_are_strict():
    for tool in ASSISTANT_TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])


def test_list_models_uses_models_api():
    page = {
        "data": [
            {"type": "model", "id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "created_at": "2026-09-01T00:00:00Z"},
            {"type": "model", "id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5", "created_at": "2026-09-01T00:00:00Z"},
        ],
        "has_more": False,
        "first_id": "claude-opus-5-5",
        "last_id": "claude-haiku-5-5",
    }
    ai, requests = make_ai([page])
    models = asyncio.run(ai.list_models())
    assert "/v1/models" in requests[0]["url"] and "lifecycle" in requests[0]["url"]
    assert [(m.id, m.name) for m in models] == [
        ("claude-opus-5-5", "Claude Opus 5.5"),
        ("claude-haiku-5-5", "Claude Haiku 5.5"),
    ]


def test_claude_default_model_when_none():
    ai, requests = make_ai([message([{"type": "text", "text": "ok"}])], model=None)
    asyncio.run(ai.digest("", "", "сейчас"))
    assert requests[0]["body"]["model"] == "claude-opus-5-5"
