from tgmanager.storage import Storage


def test_task_lifecycle(tmp_path):
    s = Storage(tmp_path / "db.sqlite")
    a = s.add_task("Позвонить маме", 1000)
    b = s.add_task("Купить хлеб", None)
    c = s.add_task("Отчёт", 500)
    s.add_task("Из чата", 100, status="suggested")

    # Сначала со сроком по времени, потом без срока; предложенные не показываем.
    assert [t.id for t in s.open_tasks()] == [c.id, a.id, b.id]
    assert [t.id for t in s.due_tasks(now=800)] == [c.id]

    s.mark_notified(c.id)
    assert s.due_tasks(now=800) == []

    s.reschedule(c.id, 900)
    assert [t.id for t in s.due_tasks(now=1000)] == [c.id, a.id]

    s.set_status(a.id, "done")
    s.rename(b.id, "Купить хлеб и молоко")
    assert [t.title for t in s.open_tasks()] == ["Отчёт", "Купить хлеб и молоко"]


def test_inbox_and_kv(tmp_path):
    s = Storage(tmp_path / "db.sqlite")
    s.add_inbox(1, "Иван", "Иван", False, "Скинь отчёт", 10)
    s.add_inbox(2, "Работа", "Я", True, "Сделаю завтра", 11)
    items = s.peek_inbox(10)
    assert [i.text for i in items] == ["Скинь отчёт", "Сделаю завтра"]
    assert items[1].outgoing is True
    s.delete_inbox([items[0].id])
    assert [i.chat_id for i in s.peek_inbox(10)] == [2]

    assert s.get("timezone") is None
    s.set("timezone", "Europe/Moscow")
    s.set("timezone", "Europe/Vilnius")
    assert s.get("timezone") == "Europe/Vilnius"

    s.set_managed_folders({5: "Работа", 6: "Семья"})
    s.set_managed_folders({5: "Работа"})
    assert s.managed_folders() == {5: "Работа"}
