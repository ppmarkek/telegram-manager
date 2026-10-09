"""SQLite-хранилище: задачи, входящие сообщения для разбора, папки бота и настройки."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .timeutil import now_ts

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    due_at INTEGER,                -- unix-время; NULL = без срока
    status TEXT NOT NULL,          -- suggested | active | done | dismissed
    notified INTEGER NOT NULL DEFAULT 0,
    source_chat TEXT,
    source_text TEXT,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS inbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    chat_title TEXT NOT NULL,
    sender TEXT NOT NULL,
    outgoing INTEGER NOT NULL,
    text TEXT NOT NULL,
    date INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS managed_folders (
    folder_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Если ИИ долго недоступен, входящие не должны копиться бесконечно.
INBOX_MAX_ROWS = 2000


@dataclass(frozen=True)
class Task:
    id: int
    title: str
    due_at: int | None
    status: str
    notified: bool
    source_chat: str | None
    source_text: str | None
    created_at: int


@dataclass(frozen=True)
class InboxItem:
    id: int
    chat_id: int
    chat_title: str
    sender: str
    outgoing: bool
    text: str
    date: int


def _task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        title=row["title"],
        due_at=row["due_at"],
        status=row["status"],
        notified=bool(row["notified"]),
        source_chat=row["source_chat"],
        source_text=row["source_text"],
        created_at=row["created_at"],
    )


class Storage:
    def __init__(self, path: Path | str):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # --- задачи ---

    def add_task(
        self,
        title: str,
        due_at: int | None,
        status: str = "active",
        source_chat: str | None = None,
        source_text: str | None = None,
    ) -> Task:
        cur = self.db.execute(
            "INSERT INTO tasks (title, due_at, status, source_chat, source_text, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (title, due_at, status, source_chat, source_text, now_ts()),
        )
        self.db.commit()
        task = self.get_task(cur.lastrowid)
        assert task is not None
        return task

    def get_task(self, task_id: int) -> Task | None:
        row = self.db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _task(row) if row else None

    def set_status(self, task_id: int, status: str) -> None:
        self.db.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        self.db.commit()

    def reschedule(self, task_id: int, due_at: int | None) -> None:
        self.db.execute(
            "UPDATE tasks SET due_at = ?, notified = 0 WHERE id = ?", (due_at, task_id)
        )
        self.db.commit()

    def rename(self, task_id: int, title: str) -> None:
        self.db.execute("UPDATE tasks SET title = ? WHERE id = ?", (title, task_id))
        self.db.commit()

    def open_tasks(self) -> list[Task]:
        """Активные задачи: сначала со сроком (по времени), потом без срока."""
        rows = self.db.execute(
            "SELECT * FROM tasks WHERE status = 'active'"
            " ORDER BY due_at IS NULL, due_at, id"
        ).fetchall()
        return [_task(r) for r in rows]

    def due_tasks(self, now: int) -> list[Task]:
        rows = self.db.execute(
            "SELECT * FROM tasks WHERE status = 'active' AND notified = 0"
            " AND due_at IS NOT NULL AND due_at <= ? ORDER BY due_at",
            (now,),
        ).fetchall()
        return [_task(r) for r in rows]

    def mark_notified(self, task_id: int) -> None:
        self.db.execute("UPDATE tasks SET notified = 1 WHERE id = ?", (task_id,))
        self.db.commit()

    # --- входящие сообщения для поиска задач ---

    def add_inbox(
        self, chat_id: int, chat_title: str, sender: str, outgoing: bool, text: str, date: int
    ) -> None:
        self.db.execute(
            "INSERT INTO inbox (chat_id, chat_title, sender, outgoing, text, date)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, chat_title, sender, int(outgoing), text, date),
        )
        self.db.execute(
            "DELETE FROM inbox WHERE id NOT IN"
            " (SELECT id FROM inbox ORDER BY id DESC LIMIT ?)",
            (INBOX_MAX_ROWS,),
        )
        self.db.commit()

    def peek_inbox(self, limit: int) -> list[InboxItem]:
        rows = self.db.execute("SELECT * FROM inbox ORDER BY id LIMIT ?", (limit,)).fetchall()
        return [
            InboxItem(
                id=r["id"],
                chat_id=r["chat_id"],
                chat_title=r["chat_title"],
                sender=r["sender"],
                outgoing=bool(r["outgoing"]),
                text=r["text"],
                date=r["date"],
            )
            for r in rows
        ]

    def delete_inbox(self, ids: list[int]) -> None:
        self.db.executemany("DELETE FROM inbox WHERE id = ?", [(i,) for i in ids])
        self.db.commit()

    # --- папки, созданные ботом ---

    def managed_folders(self) -> dict[int, str]:
        rows = self.db.execute("SELECT folder_id, title FROM managed_folders").fetchall()
        return {r["folder_id"]: r["title"] for r in rows}

    def set_managed_folders(self, folders: dict[int, str]) -> None:
        self.db.execute("DELETE FROM managed_folders")
        self.db.executemany(
            "INSERT INTO managed_folders (folder_id, title) VALUES (?, ?)", list(folders.items())
        )
        self.db.commit()

    # --- настройки ---

    def get(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.db.commit()
