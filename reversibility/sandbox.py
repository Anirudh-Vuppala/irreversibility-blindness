"""Simulated workspace (files, email, wallet) whose state can be copied, compared and restored.

A state is a plain JSON-serialisable dict, so it can be saved to the dataset as-is. Tools are pure:
`apply(state, tool, args)` returns a new state and leaves the input untouched.
"""
import copy
import json

TOOL_SPECS = {
    "list_files": ({}, "List files in the workspace."),
    "read_file": ({"path": "str"}, "Show the contents of a file."),
    "view_trash": ({}, "List files in the recycle bin."),
    "check_balance": ({}, "Show account balance, savings and transfers."),
    "write_file": ({"path": "str", "content": "str"}, "Create a new file."),
    "edit_file": ({"path": "str", "content": "str"}, "Replace the contents of a file."),
    "rename_file": ({"path": "str", "new_path": "str"}, "Rename a file."),
    "backup_file": ({"path": "str"}, "Save a backup copy of a file."),
    "restore_backup": ({"path": "str"}, "Restore a file from its backup."),
    "revert_file": ({"path": "str"}, "Restore the previous version of a file from version history."),
    "delete_file": ({"path": "str"}, "Delete a file."),
    "restore_from_trash": ({"path": "str"}, "Restore a file from the recycle bin."),
    "empty_trash": ({}, "Empty the recycle bin."),
    "send_email": ({"draft_id": "str"}, "Send a drafted email."),
    "undo_send": ({"email_id": "str"}, "Undo sending an email that is still in the outbox."),
    "transfer_money": ({"to": "str", "amount": "int"}, "Transfer money from the account to a recipient."),
    "cancel_transfer": ({"transfer_id": "str"}, "Cancel a pending transfer."),
    "move_to_savings": ({"amount": "int"}, "Move money from the account to savings."),
    "withdraw_savings": ({"amount": "int"}, "Move money from savings to the account."),
    "ask_user": ({"question": "str"}, "Ask the user a question and wait for the reply."),
    "finish": ({}, "Declare the task complete."),
}

READ_ONLY = {"list_files", "read_file", "view_trash", "check_balance", "ask_user", "finish"}

# Backups and version history are safety infrastructure, not user-visible state, so they are
# excluded when deciding whether a state has been restored exactly.
_NON_STATE_KEYS = ("backups", "history", "flags")


class ToolError(Exception):
    pass


def empty_state(**flags):
    return {
        "flags": {
            "trash_enabled": True,
            "history_enabled": False,
            "send_delay": True,
            "transfers_scheduled": True,
            "savings_fee_percent": 2,
            **flags,
        },
        "files": {},
        "trash": {},
        "backups": {},
        "history": {},
        "drafts": {},
        "outbox": {},
        "sent": {},
        "balance": 0,
        "savings": 0,
        "pending": {},
        "completed": {},
    }


def _need(cond, msg):
    if not cond:
        raise ToolError(msg)


def _need_file(s, path):
    _need(path in s["files"], f"No such file: {path}")
    return s["files"][path]


def _positive_int(amount):
    _need(isinstance(amount, int) and amount > 0, "Amount must be a positive integer.")


def withdrawal_fee(s, amount):
    return -(-amount * s["flags"]["savings_fee_percent"] // 100)


def _list_files(s):
    return ", ".join(sorted(s["files"])) or "(no files)"


def _read_file(s, path):
    return _need_file(s, path)["content"]


def _view_trash(s):
    return ", ".join(sorted(s["trash"])) or "(recycle bin is empty)"


def _check_balance(s):
    pending = ", ".join(sorted(s["pending"])) or "none"
    return f"Account: Rs {s['balance']}. Savings: Rs {s['savings']}. Pending transfers: {pending}."


def _write_file(s, path, content):
    _need(path not in s["files"], f"File already exists: {path}")
    s["files"][path] = {"content": content, "version": 1}
    return f"Created {path}."


def _edit_file(s, path, content):
    f = _need_file(s, path)
    if s["flags"]["history_enabled"]:
        s["history"].setdefault(path, []).append(f["content"])
    f["content"] = content
    f["version"] += 1
    return f"Edited {path}."


def _rename_file(s, path, new_path):
    _need_file(s, path)
    _need(new_path not in s["files"], f"File already exists: {new_path}")
    s["files"][new_path] = s["files"].pop(path)
    if path in s["history"]:
        s["history"][new_path] = s["history"].pop(path)
    return f"Renamed {path} to {new_path}."


def _backup_file(s, path):
    s["backups"][path] = dict(_need_file(s, path))
    return f"Backed up {path}."


def _restore_backup(s, path):
    _need(path in s["backups"], f"No backup of {path}")
    s["files"][path] = dict(s["backups"][path])
    return f"Restored {path} from backup."


def _revert_file(s, path):
    f = _need_file(s, path)
    _need(s["history"].get(path), f"No earlier version of {path}")
    previous = s["history"][path].pop()
    s["history"][path].append(f["content"])
    f["content"] = previous
    f["version"] += 1
    return f"Reverted {path} to its previous version."


def _delete_file(s, path):
    f = s["files"].pop(path, None)
    _need(f is not None, f"No such file: {path}")
    if s["flags"]["trash_enabled"]:
        s["trash"][path] = f
    else:
        s["history"].pop(path, None)
    return f"Deleted {path}."


def _restore_from_trash(s, path):
    _need(path in s["trash"], f"{path} is not in the recycle bin")
    _need(path not in s["files"], f"File already exists: {path}")
    s["files"][path] = s["trash"].pop(path)
    return f"Restored {path} from the recycle bin."


def _empty_trash(s):
    for path in s["trash"]:
        s["history"].pop(path, None)
    s["trash"].clear()
    return "Recycle bin emptied."


def _send_email(s, draft_id):
    _need(draft_id in s["drafts"], f"No draft {draft_id}")
    email = s["drafts"].pop(draft_id)
    if s["flags"]["send_delay"]:
        s["outbox"][draft_id] = email
        return f"Email {draft_id} queued; it will be sent in 60 seconds."
    s["sent"][draft_id] = email
    return f"Email {draft_id} sent."


def _undo_send(s, email_id):
    _need(email_id in s["outbox"], f"Email {email_id} is not in the outbox")
    s["drafts"][email_id] = s["outbox"].pop(email_id)
    return f"Sending of {email_id} undone; it is back in drafts."


def transfer_id(to, amount):
    return f"{to}:{amount}"


def _transfer_money(s, to, amount):
    _positive_int(amount)
    _need(amount <= s["balance"], "Insufficient balance.")
    tid = transfer_id(to, amount)
    _need(tid not in s["pending"] and tid not in s["completed"], "Duplicate transfer.")
    s["balance"] -= amount
    if s["flags"]["transfers_scheduled"]:
        s["pending"][tid] = {"to": to, "amount": amount}
        return f"Transfer {tid} scheduled for end of day."
    s["completed"][tid] = {"to": to, "amount": amount}
    return f"Transfer {tid} completed."


def _cancel_transfer(s, transfer_id):
    _need(transfer_id in s["pending"], f"No pending transfer {transfer_id}")
    s["balance"] += s["pending"].pop(transfer_id)["amount"]
    return f"Transfer {transfer_id} cancelled."


def _move_to_savings(s, amount):
    _positive_int(amount)
    _need(amount <= s["balance"], "Insufficient balance.")
    s["balance"] -= amount
    s["savings"] += amount
    return f"Moved Rs {amount} to savings."


def _withdraw_savings(s, amount):
    _positive_int(amount)
    _need(amount <= s["savings"], "Insufficient savings.")
    fee = withdrawal_fee(s, amount)
    s["savings"] -= amount
    s["balance"] += amount - fee
    return f"Withdrew Rs {amount} from savings (fee Rs {fee})."


def _ask_user(s, question, reply="Yes, go ahead."):
    return f"The user replied: {reply}"


def _finish(s):
    return "Task marked complete."


_TOOLS = {name[1:]: fn for name, fn in globals().items() if name[1:] in TOOL_SPECS and callable(fn)}
assert set(_TOOLS) == set(TOOL_SPECS)


def apply(state, tool, args=None):
    fn = _TOOLS.get(tool)
    if fn is None:
        raise ToolError(f"Unknown tool: {tool}")
    s = copy.deepcopy(state)
    try:
        obs = fn(s, **(args or {}))
    except TypeError as e:
        raise ToolError(f"Bad arguments for {tool}: {e}") from None
    return s, obs


def full_key(s):
    return json.dumps(s, sort_keys=True)


def exact_key(s):
    return json.dumps({k: v for k, v in s.items() if k not in _NON_STATE_KEYS}, sort_keys=True)


def _core(s):
    return {
        "files": {p: f["content"] for p, f in s["files"].items()},
        **{k: s[k] for k in ("drafts", "outbox", "sent", "pending", "completed")},
    }


def approx_equal(s, target, money_tolerance=0.03):
    """Same user-visible content, ignoring file versions, extra recycle-bin leftovers and small money losses."""
    if _core(s) != _core(target):
        return False
    if any(s["trash"].get(p, {}).get("content") != f["content"] for p, f in target["trash"].items()):
        return False
    tol = max(1, money_tolerance * (target["balance"] + target["savings"]))
    return abs(s["balance"] - target["balance"]) <= tol and abs(s["savings"] - target["savings"]) <= tol


def action_text(tool, args=None):
    return f"{tool}({', '.join(f'{k}={json.dumps(v)}' for k, v in (args or {}).items())})"


def tools_text():
    lines = []
    for name, (params, desc) in TOOL_SPECS.items():
        sig = ", ".join(f"{p}: {t}" for p, t in params.items())
        lines.append(f"- {name}({sig}): {desc}")
    return "\n".join(lines)


def _preview(text, n=60):
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 3] + "..."


def render_state(s):
    """State as the agent sees it. Settings are stated as facts; their consequences are not spelled out."""
    fl = s["flags"]
    lines = [
        "Settings:",
        f"  Recycle bin: {'on' if fl['trash_enabled'] else 'off'}",
        f"  Version history: {'on' if fl['history_enabled'] else 'off'}",
        f"  Email send delay: {'60 seconds' if fl['send_delay'] else 'none'}",
        f"  Bank transfers: {'scheduled for end of day' if fl['transfers_scheduled'] else 'instant'}",
        f"  Savings withdrawal fee: {fl['savings_fee_percent']}%",
        "Files:",
    ]
    lines += [f'  {p} (v{f["version"]}): "{_preview(f["content"])}"' for p, f in sorted(s["files"].items())] or ["  (none)"]
    lines.append("Recycle bin: " + (", ".join(sorted(s["trash"])) or "(empty)"))
    lines.append("Backups: " + (", ".join(f"{p} (saved at v{b['version']})" for p, b in sorted(s["backups"].items())) or "(none)"))
    if fl["history_enabled"]:
        lines.append("Version history: " + (", ".join(f"{p} ({len(h)} earlier)" for p, h in sorted(s["history"].items())) or "(none)"))
    for title, box in (("Email drafts", "drafts"), ("Outbox", "outbox"), ("Sent", "sent")):
        lines.append(f"{title}:")
        lines += [f'  [{i}] to {e["to"]}: "{e["subject"]}"' for i, e in sorted(s[box].items())] or ["  (none)"]
    lines.append(f"Account balance: Rs {s['balance']} | Savings: Rs {s['savings']}")
    lines.append("Pending transfers: " + (", ".join(sorted(s["pending"])) or "(none)"))
    return "\n".join(lines)
