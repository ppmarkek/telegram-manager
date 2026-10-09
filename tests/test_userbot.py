"""Работа с папками через подменённый клиент Telegram: какие запросы уходят в API."""

import asyncio

from telethon.tl import functions, types

from tgmanager import userbot
from tgmanager.folders import ChatInfo, ResolvedFolder


def title(text):
    return types.TextWithEntities(text=text, entities=[])


def dialog_filter(fid, name, peers=()):
    return types.DialogFilter(
        id=fid, title=title(name), pinned_peers=[], include_peers=list(peers), exclude_peers=[]
    )


class FakeClient:
    def __init__(self, filters, app_config=None):
        self.filters = filters
        self.app_config = app_config
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        if isinstance(request, functions.messages.GetDialogFiltersRequest):
            return types.messages.DialogFilters(filters=[types.DialogFilterDefault(), *self.filters])
        if isinstance(request, functions.help.GetAppConfigRequest):
            return types.help.AppConfig(hash=1, config=self.app_config or types.JsonObject(value=[]))
        return True


def config(**values):
    return types.JsonObject(
        value=[types.JsonObjectValue(key=k, value=types.JsonNumber(value=v)) for k, v in values.items()]
    )


def test_folder_state_and_limits():
    filters = [
        dialog_filter(2, "Мои"),
        dialog_filter(3, "Работа"),  # папка бота
        dialog_filter(4, "Переименов"),  # была папкой бота, но пользователь её переименовал
        types.DialogFilterChatlist(id=5, title=title("Общая"), pinned_peers=[], include_peers=[]),
    ]
    client = FakeClient(filters, config(dialog_filters_limit_default=12, dialog_filters_chats_limit_default=150))
    state = asyncio.run(userbot.folder_state(client, {3: "Работа", 4: "Семья"}, premium=False))
    assert state.managed == {3: "Работа"}
    assert [f.id for f in state.user_folders] == [2, 4]
    assert [f.id for f in state.chatlists] == [5]
    assert (state.max_folders, state.max_chats) == (12, 150)
    assert state.free_slots(replace_existing=False) == 12 - 3
    assert state.free_slots(replace_existing=True) == 12 - 1


def test_default_limits_for_premium():
    client = FakeClient([])
    assert asyncio.run(userbot.folder_limits(client, premium=True)) == (30, 200)


def test_apply_folders_reuses_ids_and_deletes_leftovers():
    filters = [dialog_filter(2, "Мои"), dialog_filter(3, "Работа"), dialog_filter(6, "Старая")]
    client = FakeClient(filters)
    state = asyncio.run(userbot.folder_state(client, {3: "Работа", 6: "Старая"}, premium=False))
    client.requests.clear()

    peer = types.InputPeerUser(user_id=10, access_hash=1)
    chat = ChatInfo(idx=1, peer_id=10, title="Иван", kind="личный", username=None, snippets=[], input_peer=peer)
    folders = [
        ResolvedFolder(title="Семья", emoji="🏠", chats=[chat]),
        ResolvedFolder(title="Друзья", emoji=None, chats=[chat]),
    ]
    created, errors = asyncio.run(userbot.apply_folders(client, state, folders, replace_existing=False))

    assert errors == []
    assert created == {3: "Семья", 6: "Друзья"}
    updates = [r for r in client.requests if isinstance(r, functions.messages.UpdateDialogFilterRequest)]
    assert [(r.id, r.filter.title.text, r.filter.emoticon) for r in updates] == [
        (3, "Семья", "🏠"),
        (6, "Друзья", None),
    ]
    assert updates[0].filter.include_peers == [peer]
    # Папку пользователя «Мои» (id 2) не трогали.
    assert all(r.id != 2 for r in updates)


def test_apply_folders_replace_existing_deletes_user_folders():
    client = FakeClient([dialog_filter(2, "Мои")])
    state = asyncio.run(userbot.folder_state(client, {}, premium=False))
    client.requests.clear()
    chat = ChatInfo(idx=1, peer_id=10, title="Иван", kind="личный", username=None, snippets=[],
                    input_peer=types.InputPeerUser(user_id=10, access_hash=1))
    created, _ = asyncio.run(
        userbot.apply_folders(client, state, [ResolvedFolder("Семья", None, [chat])], replace_existing=True)
    )
    requests = [(r.id, r.filter is None) for r in client.requests]
    assert requests == [(2, True), (3, False)]  # сначала удалили «Мои», потом создали новую с id 3
    assert created == {3: "Семья"}
