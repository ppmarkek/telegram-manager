"""Логика папок без обращения к Telegram: подготовка данных для ИИ, проверка плана, превью."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any

from .ai import FolderPlan

# Telegram-клиенты не дают задать название папки длиннее 12 символов.
TITLE_MAX_LEN = 12
# id 0 и 1 зарезервированы («Все чаты» и архив).
FIRST_FOLDER_ID = 2
LAST_FOLDER_ID = 255


@dataclass
class ChatInfo:
    idx: int  # порядковый номер для ИИ (короче и надёжнее, чем peer id)
    peer_id: int
    title: str
    kind: str  # личный | бот | группа | канал
    username: str | None
    snippets: list[str]
    input_peer: Any = field(default=None, repr=False)


@dataclass
class ResolvedFolder:
    title: str
    emoji: str | None
    chats: list[ChatInfo]


@dataclass
class ResolvedPlan:
    folders: list[ResolvedFolder]
    unassigned: int
    comment: str
    warnings: list[str]


def format_chats_for_ai(chats: list[ChatInfo]) -> str:
    lines = []
    for chat in chats:
        name = chat.title + (f" (@{chat.username})" if chat.username else "")
        lines.append(f"[{chat.idx}] {chat.kind}: {name}")
        for snippet in chat.snippets:
            lines.append(f"    > {snippet}")
    return "\n".join(lines)


def format_plan_for_ai(plan: ResolvedPlan, idx_by_peer: dict[int, int]) -> str:
    """План текстом для ИИ. Номера берём из свежего списка чатов: порядок мог измениться."""
    lines = []
    for folder in plan.folders:
        ids = [idx_by_peer.get(c.peer_id) for c in folder.chats]
        ids_text = ", ".join(str(i) for i in ids if i)
        lines.append(f"{folder.emoji or ''} {folder.title}: {ids_text}".strip())
    return "\n".join(lines)


def clean_title(title: str) -> str:
    title = " ".join(title.split())
    return title[:TITLE_MAX_LEN].strip() or "Папка"


def resolve_plan(
    plan: FolderPlan, chats: list[ChatInfo], max_folders: int, max_chats_per_folder: int
) -> ResolvedPlan:
    """Превращает ответ ИИ в проверенный план с учётом лимитов Telegram."""
    by_idx = {c.idx: c for c in chats}
    warnings: list[str] = []
    used: set[int] = set()
    seen_titles: set[str] = set()
    folders: list[ResolvedFolder] = []

    for planned in plan.folders:
        title = clean_title(planned.title)
        if title.lower() in seen_titles:
            continue
        members: list[ChatInfo] = []
        for idx in planned.chats:
            chat = by_idx.get(idx)
            # Каждый чат — максимум в одной папке, несуществующие номера пропускаем.
            if chat is None or idx in used:
                continue
            members.append(chat)
            used.add(idx)
        if not members:
            continue
        if len(members) > max_chats_per_folder:
            warnings.append(
                f"В папке «{title}» {len(members)} чатов, а Telegram разрешает "
                f"{max_chats_per_folder}: лишние не попадут в папку."
            )
            for extra in members[max_chats_per_folder:]:
                used.discard(extra.idx)
            members = members[:max_chats_per_folder]
        emoji = (planned.emoji or "").strip() or None
        folders.append(ResolvedFolder(title=title, emoji=emoji, chats=members))
        seen_titles.add(title.lower())

    if len(folders) > max_folders:
        dropped = folders[max_folders:]
        warnings.append(
            "Не хватило мест для папок: "
            + ", ".join(f"«{f.title}»" for f in dropped)
            + " не будут созданы."
        )
        for folder in dropped:
            for chat in folder.chats:
                used.discard(chat.idx)
        folders = folders[:max_folders]

    return ResolvedPlan(
        folders=folders,
        unassigned=len(chats) - len(used),
        comment=plan.comment.strip(),
        warnings=warnings,
    )


def allocate_ids(count: int, reusable: list[int], taken: set[int]) -> list[int]:
    """Выбирает id для папок: сначала переиспользует свои, потом берёт свободные."""
    ids = sorted(reusable)[:count]
    candidate = FIRST_FOLDER_ID
    blocked = set(taken) | set(ids)
    while len(ids) < count:
        if candidate > LAST_FOLDER_ID:
            raise ValueError("Не осталось свободных id для папок")
        if candidate not in blocked:
            ids.append(candidate)
            blocked.add(candidate)
        candidate += 1
    return ids


def preview_html(plan: ResolvedPlan, shown_per_folder: int = 6) -> str:
    esc = html.escape
    parts = ["<b>📁 Предлагаю такие папки:</b>", ""]
    for folder in plan.folders:
        names = [esc(c.title) for c in folder.chats[:shown_per_folder]]
        rest = len(folder.chats) - len(names)
        listing = ", ".join(names) + (f" и ещё {rest}" if rest > 0 else "")
        icon = esc(folder.emoji) + " " if folder.emoji else ""
        parts.append(f"{icon}<b>{esc(folder.title)}</b> — {len(folder.chats)}")
        parts.append(f"<i>{listing}</i>")
        parts.append("")
    if plan.unassigned:
        parts.append(f"Без папки останется чатов: {plan.unassigned} (они будут в «Все чаты»).")
    if plan.comment:
        parts.append("")
        parts.append(f"💬 {esc(plan.comment)}")
    for warning in plan.warnings:
        parts.append(f"⚠️ {esc(warning)}")
    parts.append("")
    parts.append(
        "Чтобы что-то поменять, просто напиши, например: "
        "<i>«объедини учёбу и курсы, а каналы про крипту вынеси отдельно»</i>."
    )
    return "\n".join(parts)
