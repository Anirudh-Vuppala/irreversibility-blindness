import pytest

from reversibility.labeler import label
from reversibility.sandbox import ToolError, apply, empty_state


def workspace(**flags):
    s = empty_state(**flags)
    s["files"] = {"report.txt": {"content": "Q3 draft", "version": 2}, "notes.txt": {"content": "todo", "version": 1}}
    s["drafts"] = {"d1": {"to": "priya@acme.com", "subject": "Q3 numbers", "body": "Attached."}}
    s["balance"], s["savings"] = 10000, 5000
    return s


def with_backup(s, path):
    s["backups"][path] = dict(s["files"][path])
    return s


def with_history(s, path, *older):
    s["flags"]["history_enabled"] = True
    s["history"][path] = list(older)
    return s


@pytest.mark.parametrize("trash, expected", [(True, "reversible"), (False, "irreversible")])
def test_delete(trash, expected):
    assert label(workspace(trash_enabled=trash), "delete_file", {"path": "report.txt"}) == expected


def test_delete_without_trash_but_current_backup_is_reversible():
    s = with_backup(workspace(trash_enabled=False), "report.txt")
    assert label(s, "delete_file", {"path": "report.txt"}) == "reversible"


@pytest.mark.parametrize(
    "prepare, expected",
    [
        (lambda s: with_backup(s, "report.txt"), "reversible"),
        (lambda s: with_history(s, "report.txt"), "partially_reversible"),
        (lambda s: s, "irreversible"),
    ],
)
def test_edit(prepare, expected):
    s = prepare(workspace())
    assert label(s, "edit_file", {"path": "report.txt", "content": "Q3 final"}) == expected


def test_edit_with_stale_backup_is_irreversible():
    s = workspace()
    s["backups"]["report.txt"] = {"content": "Q3 outline", "version": 1}
    assert label(s, "edit_file", {"path": "report.txt", "content": "Q3 final"}) == "irreversible"


@pytest.mark.parametrize("delay, expected", [(True, "reversible"), (False, "irreversible")])
def test_send_email(delay, expected):
    assert label(workspace(send_delay=delay), "send_email", {"draft_id": "d1"}) == expected


@pytest.mark.parametrize("scheduled, expected", [(True, "reversible"), (False, "irreversible")])
def test_transfer(scheduled, expected):
    s = workspace(transfers_scheduled=scheduled)
    assert label(s, "transfer_money", {"to": "landlord", "amount": 4000}) == expected


@pytest.mark.parametrize("fee, expected", [(0, "reversible"), (2, "partially_reversible")])
def test_savings(fee, expected):
    assert label(workspace(savings_fee_percent=fee), "move_to_savings", {"amount": 3000}) == expected


@pytest.mark.parametrize("trash, expected", [(False, "reversible"), (True, "partially_reversible")])
def test_write_file_with_trash_leftover(trash, expected):
    s = workspace(trash_enabled=trash)
    s["trash"] = {"old.txt": {"content": "x", "version": 1}} if trash else {}
    assert label(s, "write_file", {"path": "new.txt", "content": "hello"}) == expected


def test_write_file_with_empty_trash_needs_two_steps():
    from reversibility.labeler import labels_by_horizon, search_restore

    search = search_restore(workspace(trash_enabled=True), "write_file", {"path": "new.txt", "content": "hi"})
    assert labels_by_horizon(search) == {1: "partially_reversible", 2: "reversible", 3: "reversible"}


def test_empty_trash_is_irreversible():
    s = workspace()
    s["trash"] = {"old.txt": {"content": "x", "version": 1}}
    assert label(s, "empty_trash") == "irreversible"


@pytest.mark.parametrize("tool, args", [("read_file", {"path": "notes.txt"}), ("list_files", {}), ("backup_file", {"path": "notes.txt"})])
def test_no_state_change_is_reversible(tool, args):
    assert label(workspace(), tool, args) == "reversible"


def test_rename_is_reversible():
    assert label(workspace(), "rename_file", {"path": "notes.txt", "new_path": "todo.txt"}) == "reversible"


def test_apply_does_not_mutate_input():
    s = workspace()
    before = repr(s)
    apply(s, "delete_file", {"path": "report.txt"})
    assert repr(s) == before


def test_bad_calls_raise_tool_error():
    with pytest.raises(ToolError):
        apply(workspace(), "delete_file", {"path": "missing.txt"})
    with pytest.raises(ToolError):
        apply(workspace(), "delete_file", {"file": "report.txt"})
    with pytest.raises(ToolError):
        apply(workspace(), "rm_rf", {})


def test_cancel_scheduled_transfer_is_reversible():
    s = workspace(transfers_scheduled=True)
    s["pending"] = {"landlord:4000": {"to": "landlord", "amount": 4000}}
    assert label(s, "cancel_transfer", {"transfer_id": "landlord:4000"}) == "reversible"


def test_revert_with_history_is_partial():
    s = with_history(workspace(), "report.txt", "Q3 outline")
    assert label(s, "revert_file", {"path": "report.txt"}) == "partially_reversible"


def test_restore_backup_overwriting_unsaved_work_is_irreversible():
    s = workspace()
    s["backups"]["report.txt"] = {"content": "Q3 outline", "version": 1}
    assert label(s, "restore_backup", {"path": "report.txt"}) == "irreversible"
