"""Поддельный `agy` для тестов: пишет вызовы в calls.jsonl и отвечает из responses.json."""

import json
import os
import sys
from pathlib import Path

state = Path(os.environ["FAKE_AGY_DIR"])
args = sys.argv[1:]
stdin = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()
with open(state / "calls.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"args": args, "stdin": stdin, "cwd": os.getcwd()}, ensure_ascii=False) + "\n")

if args[:1] == ["models"]:
    print("\x1b[1mAvailable models\x1b[0m")
    print((state / "models.txt").read_text(encoding="utf-8"))
    sys.exit(0)
if args[:1] == ["--version"]:
    print("agy " + (state / "version.txt").read_text().strip())
    sys.exit(0)
if args[:1] == ["update"]:
    (state / "version.txt").write_text("1.2.0")
    print("Updated to 1.2.0")
    sys.exit(0)

responses_file = state / "responses.json"
responses = json.loads(responses_file.read_text(encoding="utf-8")) if responses_file.exists() else []
envelope = responses.pop(0) if responses else {"status": "SUCCESS", "response": "ok"}
responses_file.write_text(json.dumps(responses, ensure_ascii=False), encoding="utf-8")
envelope.setdefault("conversation_id", "conv-1")
code = envelope.pop("_exit", 0)

if "--input-format" in args:
    print(json.dumps({"event": "init", "cwd": os.getcwd(), "permission_mode": "request-review"}))
    print(json.dumps({"event": "step_update", "step_type": "agent_response", "state": "DONE"}))
    print(json.dumps({"event": "result", "result": envelope}, ensure_ascii=False))
else:
    print(json.dumps(envelope, ensure_ascii=False))
sys.exit(code)
