"""ИИ через Antigravity CLI (`agy`) — модели и лимиты из твоего Google-аккаунта.

agy вызывается в безголовом режиме (`agy -p ... --output-format json`), ответы
в нужном формате получаем через `--json-schema`. Каждый вызов — отдельный процесс
в пустой рабочей папке вне проекта, чтобы агенту agy нечего было читать или менять.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .ai import (
    AI,
    ASSISTANT_SYSTEM,
    AIError,
    ModelInfo,
    ToolExecutor,
    run_tool,
)

log = logging.getLogger(__name__)

# Длинные запросы (сотни чатов) не влезают в аргумент командной строки —
# их отправляем через stdin в формате stream-json.
ARG_PROMPT_LIMIT = 20_000 if sys.platform == "win32" else 100_000

AGY_SETTINGS = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"

NO_TOOLS_NOTE = (
    "Важно: не используй никакие инструменты (файлы, терминал, интернет) — "
    "всё нужное уже есть в этом сообщении."
)

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
VERSION_RE = re.compile(r"\d+\.\d+(?:\.\d+)?")


@dataclass
class AgyResult:
    status: str
    response: str
    structured: Any
    error: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgyResult:
        error = data.get("error")
        if error is not None and not isinstance(error, str):
            error = json.dumps(error, ensure_ascii=False)
        return cls(
            status=str(data.get("status", "")).upper(),
            response=str(data.get("response") or ""),
            structured=data.get("structured_output"),
            error=error,
        )


def parse_json_output(stdout: str) -> AgyResult | None:
    """Разбор `--output-format json`: один JSON-объект с полем status."""
    text = stdout.strip()
    candidates = [text] + [ln for ln in reversed(text.splitlines()) if ln.strip().startswith("{")]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict) and "status" in data:
            return AgyResult.from_dict(data)
    return None


def parse_stream_output(stdout: str) -> AgyResult | None:
    """Разбор `--output-format stream-json`: ищем финальное событие result."""
    for line in stdout.splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("event") == "result" and isinstance(data.get("result"), dict):
            return AgyResult.from_dict(data["result"])
    return None


def parse_models(text: str) -> list[ModelInfo]:
    """`agy models` печатает строки вида «gemini-3.8-flash-high Gemini 3.8 Flash (High)»."""
    models: list[ModelInfo] = []
    seen: set[str] = set()
    for raw in ANSI_RE.sub("", text).splitlines():
        line = raw.strip().strip("│|*•>✓✔ ").strip()
        if not line:
            continue
        parts = line.split(None, 1)
        slug = parts[0].strip("│|,")
        # Заголовки и пояснения отсекаем: у моделей в id есть и дефис, и цифра.
        if not SLUG_RE.match(slug) or "-" not in slug or not re.search(r"\d", slug) or slug in seen:
            continue
        name = parts[1].strip(" │|-—:") if len(parts) > 1 else ""
        models.append(ModelInfo(id=slug, name=name or slug))
        seen.add(slug)
    return models


def extract_json(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    try:
        return json.loads(body)
    except ValueError:
        pass
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(body[start : end + 1])
        except ValueError:
            pass
    raise AIError("Модель ответила не в формате JSON. Попробуй ещё раз или выбери другую модель (🧠 Модель).")


def _is_nullable(node: Any) -> bool:
    return isinstance(node, dict) and any(
        isinstance(option, dict) and option.get("type") == "null" for option in node.get("anyOf", [])
    )


def agy_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Упрощённая JSON-схема: без $ref, служебных полей и null-вариантов
    (nullable-поля становятся необязательными) — так её поймёт любая модель в agy."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        if _is_nullable(node):
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            base = walk(options[0]) if len(options) == 1 else {"anyOf": walk(options)}
            if "description" in node:
                base = {**base, "description": node["description"]}
            return base
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("title", "default"):
                continue
            if key == "properties":
                out[key] = {name: walk(sub) for name, sub in value.items()}
            else:
                out[key] = walk(value)
        if "properties" in node and "required" in node:
            out["required"] = [
                name for name in node["required"] if not _is_nullable(node["properties"].get(name))
            ]
        return out

    return walk(raw)


def explain_failure(text: str, code: int | None) -> str:
    low = text.lower()
    if "auth" in low and any(k in low for k in ("required", "login", "sign in", "unauthenticated")):
        return "agy не авторизован. Один раз запусти `agy` в терминале и войди в Google-аккаунт."
    if any(k in low for k in ("quota", "resource_exhausted", "rate limit", "429", "exhausted")):
        return "Закончились лимиты agy для этой модели. Выбери другую модель (🧠 Модель) или подожди."
    if "model" in low and any(k in low for k in ("unknown", "not found", "invalid", "unsupported", "not available")):
        return "agy не знает выбранную модель. Выбери модель заново (🧠 Модель)."
    tail = " ".join(text.split())[-300:]
    suffix = f": {tail}" if tail else "."
    return f"agy завершился с ошибкой (код {code}){suffix}"


def risky_permission_mode(path: Path = AGY_SETTINGS) -> str | None:
    """Если в agy включено «всё разрешать без вопросов», чужое сообщение в чате
    теоретически сможет уговорить агента agy выполнить команду. Предупреждаем."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    mode = data.get("toolPermission") if isinstance(data, dict) else None
    return mode if mode == "always-proceed" else None


def find_agy(binary: str) -> str | None:
    found = shutil.which(binary)
    if found:
        return found
    candidates = [Path.home() / ".local" / "bin" / "agy"]
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "agy" / "bin" / "agy.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


class AgyRunner:
    def __init__(self, binary: str, workdir: Path, timeout_min: int, sandbox: bool, parallel: int = 2):
        self.binary = binary
        self.workdir = workdir
        self.timeout_min = timeout_min
        self.sandbox = sandbox
        self._semaphore = asyncio.Semaphore(parallel)

    def path(self) -> str:
        found = find_agy(self.binary)
        if not found:
            raise AIError(
                "Не нашёл agy. Установи Antigravity CLI "
                "(curl -fsSL https://antigravity.google/cli/install.sh | bash) или укажи путь в AGY_BIN."
            )
        return found

    async def _exec(self, args: list[str], stdin: bytes | None, timeout: float) -> tuple[int, str, str]:
        self.workdir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "NO_COLOR": "1"}
        async with self._semaphore:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.workdir),
                    env=env,
                )
            except OSError as exc:
                raise AIError(f"Не удалось запустить agy: {exc}") from exc
            try:
                out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise AIError("agy слишком долго не отвечал. Попробуй ещё раз или выбери модель побыстрее.") from None
        return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")

    async def run(
        self,
        prompt: str,
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        force_stream: bool = False,
    ) -> AgyResult:
        use_stream = force_stream or len(prompt.encode()) > ARG_PROMPT_LIMIT
        args = [self.path(), "-p"]
        stdin = None
        if use_stream:
            # В режиме stream-json текст из флага игнорируется, запрос идёт через stdin.
            args += ["", "--input-format", "stream-json", "--output-format", "stream-json"]
            line = json.dumps({"event": "user", "message": {"content": prompt}}, ensure_ascii=False)
            stdin = (line + "\n").encode()
        else:
            args += [prompt, "--output-format", "json"]
        if model:
            args += ["--model", model]
        if schema is not None:
            args += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
        args += ["--print-timeout", f"{self.timeout_min}m"]
        if self.sandbox:
            args.append("--sandbox")

        code, out, err = await self._exec(args, stdin, timeout=self.timeout_min * 60 + 60)
        result = parse_stream_output(out) if use_stream else parse_json_output(out)
        if result is None:
            raise AIError(explain_failure(err or out, code))
        if result.status != "SUCCESS":
            raise AIError(explain_failure(result.error or err or result.status, code))
        return result

    async def models(self) -> list[ModelInfo]:
        code, out, err = await self._exec([self.path(), "models"], None, timeout=120)
        if code != 0:
            raise AIError(explain_failure(err or out, code))
        models = parse_models(out) or parse_models(err)
        if not models:
            raise AIError("agy не вернул список моделей. Проверь, что ты вошёл в аккаунт: запусти `agy`.")
        return models

    async def version(self) -> str | None:
        try:
            _, out, err = await self._exec([self.path(), "--version"], None, timeout=60)
        except AIError:
            return None
        match = VERSION_RE.search(out) or VERSION_RE.search(err)
        return match.group(0) if match else None

    async def update(self) -> str:
        code, out, err = await self._exec([self.path(), "update"], None, timeout=600)
        text = (out + "\n" + err).strip()
        if code != 0:
            raise AIError(explain_failure(text, code))
        return text


# --- помощник с действиями: agy не знает наших инструментов, поэтому просим план в JSON ---

TOOL_NAMES = [
    "add_task",
    "list_tasks",
    "update_task",
    "sort_chats",
    "send_digest",
    "search_messages",
    "update_settings",
]
NEEDS_RESULT = {"list_tasks", "search_messages"}
AGY_ASSISTANT_STEPS = 4
MAX_ACTIONS_PER_STEP = 5

AGY_ASSISTANT_PROTOCOL = """\
Как отвечать: верни JSON {"actions": [...], "reply": "..."}.
- actions — действия, которые нужно выполнить. Каждое: {"tool": "<имя>", ...параметры}. Если ничего делать не нужно — [].
- reply — ответ владельцу.
- Если для ответа нужен результат действия (list_tasks, search_messages) — верни только эти действия \
и пустой reply: я выполню их и пришлю результат.
- Действия, которые уже выполнены (список ниже, если он есть), не повторяй — просто напиши итоговый reply.

Действия и параметры:
- add_task: title, due (YYYY-MM-DDTHH:MM; не указывай, если срока нет)
- list_tasks: без параметров
- update_task: id, action (done | delete | reschedule | rename), due (для reschedule), title (для rename)
- sort_chats: wishes (пожелания к папкам), replace_existing (true только если владелец явно разрешил заменить его папки)
- send_digest: без параметров
- search_messages: query
- update_settings: timezone, digest_time (ЧЧ:ММ или off), model (название или id модели)"""

AGENT_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": TOOL_NAMES},
                    "title": {"type": "string"},
                    "due": {"type": "string"},
                    "id": {"type": "integer"},
                    "action": {"type": "string", "enum": ["done", "delete", "reschedule", "rename"]},
                    "wishes": {"type": "string"},
                    "replace_existing": {"type": "boolean"},
                    "query": {"type": "string"},
                    "timezone": {"type": "string"},
                    "digest_time": {"type": "string"},
                    "model": {"type": "string"},
                },
                "required": ["tool"],
            },
        },
        "reply": {"type": "string"},
    },
    "required": ["actions", "reply"],
}


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    tool: str
    title: str | None = None
    due: str | None = None
    id: int | None = None
    action: str | None = None
    wishes: str | None = None
    replace_existing: bool | None = None
    query: str | None = None
    timezone: str | None = None
    digest_time: str | None = None
    model: str | None = None


class AgentStep(BaseModel):
    actions: list[AgentAction] = []
    reply: str = ""


class AgyAI(AI):
    backend = "agy"
    backend_title = "Antigravity CLI (agy)"

    def __init__(self, model: str | None, runner: AgyRunner):
        super().__init__(model)
        self.runner = runner

    async def _ask_json(self, prompt: str, model_cls: type[BaseModel], schema: dict[str, Any]) -> Any:
        full = (
            f"{prompt}\n\n{NO_TOOLS_NOTE}\n"
            "Ответь только JSON-объектом по этой схеме, без пояснений:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        result = await self.runner.run(full, model=self.model, schema=schema)
        data = result.structured if result.structured is not None else result.response
        try:
            return model_cls.model_validate(extract_json(data))
        except ValidationError as exc:
            log.warning("agy вернул неожиданный JSON: %s", exc)
            raise AIError(
                "Модель вернула ответ в неожиданном формате. Попробуй ещё раз или выбери другую модель (🧠 Модель)."
            ) from exc

    async def _structured(
        self, system: str, user: str, schema: type[BaseModel], effort: str, max_tokens: int
    ) -> Any:
        return await self._ask_json(f"{system}\n\n{user}", schema, agy_schema(schema))

    async def _text(self, system: str, user: str, effort: str) -> str:
        result = await self.runner.run(f"{system}\n\n{user}\n\n{NO_TOOLS_NOTE}", model=self.model)
        text = result.response.strip()
        if not text:
            raise AIError("Модель вернула пустой ответ.")
        return text

    async def assistant(self, user_text: str, execute: ToolExecutor) -> str:
        done: list[str] = []
        executed: set[str] = set()
        for _ in range(AGY_ASSISTANT_STEPS):
            parts = [ASSISTANT_SYSTEM, AGY_ASSISTANT_PROTOCOL, user_text]
            if done:
                parts.append("Уже выполненные действия и их результаты:\n" + "\n".join(done))
            step: AgentStep = await self._ask_json("\n\n".join(parts), AgentStep, AGENT_STEP_SCHEMA)

            fresh = []
            for action in step.actions[:MAX_ACTIONS_PER_STEP]:
                key = action.model_dump_json()
                if action.tool in TOOL_NAMES and key not in executed:
                    fresh.append(action)
                    executed.add(key)
            if not fresh:
                return step.reply.strip() or "Готово."

            failed = need_result = False
            for action in fresh:
                args = action.model_dump(exclude={"tool"})
                output, is_error = await run_tool(execute, action.tool, args)
                failed = failed or is_error or output.startswith("Ошибка")
                need_result = need_result or action.tool in NEEDS_RESULT
                done.append(f"- {action.tool}: {output}")
            if step.reply.strip() and not failed and not need_result:
                return step.reply.strip()
        return "Не получилось закончить за разумное число шагов — попробуй переформулировать."

    async def list_models(self) -> list[ModelInfo]:
        return await self.runner.models()

    async def version(self) -> str | None:
        return await self.runner.version()

    async def update(self) -> str | None:
        return await self.runner.update()
