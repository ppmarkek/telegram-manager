from tgmanager.ai import FolderPlan, PlannedFolder
from tgmanager.folders import (
    ChatInfo,
    allocate_ids,
    format_chats_for_ai,
    format_plan_for_ai,
    preview_html,
    resolve_plan,
)


def chats(n: int) -> list[ChatInfo]:
    return [
        ChatInfo(idx=i, peer_id=1000 + i, title=f"Чат {i}", kind="личный", username=None, snippets=[])
        for i in range(1, n + 1)
    ]


def test_resolve_plan_cleans_ai_output():
    plan = FolderPlan(
        folders=[
            PlannedFolder(title="Очень длинное название папки", emoji="💼", chats=[1, 2, 2, 99]),
            PlannedFolder(title="Семья", emoji="🏠", chats=[2, 3]),  # 2 уже занят
            PlannedFolder(title="Пустая", emoji="", chats=[42]),
            PlannedFolder(title="семья", emoji="", chats=[4]),  # дубль названия
        ],
        comment="  Разложил по смыслу  ",
    )
    resolved = resolve_plan(plan, chats(5), max_folders=10, max_chats_per_folder=100)
    assert [f.title for f in resolved.folders] == ["Очень длинно", "Семья"]
    assert [[c.idx for c in f.chats] for f in resolved.folders] == [[1, 2], [3]]
    assert resolved.folders[1].emoji == "🏠"
    assert resolved.unassigned == 2
    assert resolved.comment == "Разложил по смыслу"
    assert resolved.warnings == []


def test_resolve_plan_respects_limits():
    plan = FolderPlan(
        folders=[
            PlannedFolder(title="A", emoji="", chats=[1, 2, 3]),
            PlannedFolder(title="B", emoji="", chats=[4]),
            PlannedFolder(title="C", emoji="", chats=[5]),
        ],
        comment="",
    )
    resolved = resolve_plan(plan, chats(5), max_folders=2, max_chats_per_folder=2)
    assert [f.title for f in resolved.folders] == ["A", "B"]
    assert [len(f.chats) for f in resolved.folders] == [2, 1]
    assert resolved.unassigned == 2  # чат 3 не влез, папка C не создана
    assert len(resolved.warnings) == 2


def test_allocate_ids_reuses_own_and_skips_taken():
    assert allocate_ids(3, reusable=[7], taken={2, 3}) == [7, 4, 5]
    assert allocate_ids(1, reusable=[9, 8], taken=set()) == [8]
    assert allocate_ids(0, reusable=[5], taken=set()) == []


def test_format_for_ai_and_preview():
    items = chats(3)
    items[0].username = "ivan"
    items[0].snippets = ["Привет", "[фото]"]
    text = format_chats_for_ai(items)
    assert "[1] личный: Чат 1 (@ivan)" in text
    assert "    > [фото]" in text

    plan = FolderPlan(folders=[PlannedFolder(title="Друзья <3", emoji="🎉", chats=[1, 3])], comment="ok")
    resolved = resolve_plan(plan, items, 10, 100)
    # Номера пересчитываются по свежему списку чатов.
    assert format_plan_for_ai(resolved, {1001: 5, 1003: 1}) == "🎉 Друзья <3: 5, 1"
    html_text = preview_html(resolved)
    assert "<b>Друзья &lt;3</b> — 2" in html_text
    assert "Без папки останется чатов: 1" in html_text
