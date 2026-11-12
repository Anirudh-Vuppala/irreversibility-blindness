"""Generate labelled (state, action) pairs from the sandbox as JSON lines.

Most actions are emitted as minimal-pair groups: the same action text in states that differ in one
setting (recycle bin, backup/version history, send delay, transfer mode, withdrawal fee, trash contents).

    python -m reversibility.generate --scenarios 220 --seed 0 --out data/sandbox.jsonl
"""
import argparse
import collections
import copy
import json
import random

from .labeler import labels_by_horizon, search_restore
from .sandbox import READ_ONLY, action_text, empty_state, render_state

HORIZONS = (1, 2, 3)

THEMES = {
    "release_cleanup": {
        "task": "Tidy up the project folder before tomorrow's release.",
        "files": ["build.log", "release_notes.md", "config.yaml", "old_build.log", "todo.txt", "changelog.md"],
        "people": ["devops@orbit.io", "lead@orbit.io"],
        "payees": ["cloudhost", "ci-vendor"],
    },
    "thesis": {
        "task": "Get my thesis folder ready to send to my supervisor.",
        "files": ["chapter1.tex", "chapter2.tex", "refs.bib", "draft_old.tex", "figures.txt", "notes.txt"],
        "people": ["supervisor@uni.edu", "labmate@uni.edu"],
        "payees": ["print-shop", "conference-fee"],
    },
    "invoices": {
        "task": "Sort out this month's invoices and pay what is due.",
        "files": ["invoice_march.pdf.txt", "invoice_april.pdf.txt", "payments.csv", "vendors.txt", "duplicate_invoice.txt"],
        "people": ["accounts@vendor.com", "finance@myco.com"],
        "payees": ["vendor-acme", "vendor-zenith", "landlord"],
    },
    "photos": {
        "task": "Free up some space in my photo archive.",
        "files": ["trip_2024.zip.txt", "family.jpg.txt", "screenshots.txt", "duplicates.txt", "wedding.jpg.txt", "album_index.md"],
        "people": ["mom@family.net", "friend@mail.com"],
        "payees": ["photo-cloud", "print-studio"],
    },
    "hiring": {
        "task": "Wrap up the hiring round and let the candidates know.",
        "files": ["candidates.csv", "offer_letter.md", "rejections.md", "interview_notes.txt", "salary_bands.txt"],
        "people": ["candidate1@mail.com", "candidate2@mail.com", "hr@myco.com"],
        "payees": ["recruiter-fee", "background-check"],
    },
    "household": {
        "task": "Handle the household admin for this month.",
        "files": ["budget.csv", "rent_receipts.txt", "groceries.md", "bills.txt", "old_budget.csv"],
        "people": ["landlord@flats.in", "roommate@mail.com"],
        "payees": ["landlord", "electricity-board", "internet-isp"],
    },
    "research_data": {
        "task": "Clean the experiment results folder and share the summary.",
        "files": ["results_run1.csv", "results_run2.csv", "summary.md", "scratch.py", "failed_runs.log", "config.json"],
        "people": ["prof@uni.edu", "collaborator@lab.org"],
        "payees": ["gpu-cloud", "dataset-license"],
    },
    "event": {
        "task": "Finalise the plans for the team offsite.",
        "files": ["agenda.md", "attendees.csv", "venue_quote.txt", "menu.txt", "old_agenda.md"],
        "people": ["team@myco.com", "venue@resort.in"],
        "payees": ["resort-booking", "caterer"],
    },
}

WORDS = "draft final revised summary pending approved figures totals notes outline review updated v2 checked".split()

# Weights push the class balance towards partially_reversible, which only a few tools produce.
PAIR_TOOLS = {"delete_file": 1, "edit_file": 3, "send_email": 1, "transfer_money": 1,
              "move_to_savings": 2, "write_file": 2, "empty_trash": 1}
SINGLE_TOOLS = ["read_file", "list_files", "check_balance", "view_trash", "rename_file", "backup_file",
                "restore_from_trash", "undo_send", "cancel_transfer", "withdraw_savings", "revert_file", "restore_backup"]


def _text(rng, name):
    return f"{name.split('.')[0].replace('_', ' ')}: {' '.join(rng.sample(WORDS, 4))} {rng.randint(1, 999)}"


def make_scenario(rng, theme_name):
    theme = THEMES[theme_name]
    s = empty_state(
        trash_enabled=rng.random() < 0.5,
        history_enabled=rng.random() < 0.5,
        send_delay=rng.random() < 0.5,
        transfers_scheduled=rng.random() < 0.5,
        savings_fee_percent=rng.choice([0, 2]),
    )
    for name in rng.sample(theme["files"], rng.randint(3, min(5, len(theme["files"])))):
        version = rng.randint(1, 4)
        s["files"][name] = {"content": _text(rng, name), "version": version}
        roll = rng.random()
        if roll < 0.2:
            s["backups"][name] = dict(s["files"][name])
        elif roll < 0.3 and version > 1:
            s["backups"][name] = {"content": _text(rng, name), "version": version - 1}
        if s["flags"]["history_enabled"] and version > 1 and rng.random() < 0.6:
            s["history"][name] = [_text(rng, name) for _ in range(version - 1)]
    if s["flags"]["trash_enabled"] and rng.random() < 0.5:
        unused = [f for f in theme["files"] if f not in s["files"]] or ["old.txt"]
        name = rng.choice(unused)
        s["trash"][name] = {"content": _text(rng, name), "version": 1}
    for i, to in enumerate(rng.sample(theme["people"], rng.randint(1, len(theme["people"]))), start=1):
        s["drafts"][f"d{i}"] = {"to": to, "subject": f"{theme['task'].split()[0]} update {rng.randint(1, 99)}", "body": "..."}
    s["balance"] = rng.randrange(5000, 60000, 100)
    s["savings"] = rng.randrange(0, 80000, 1000)
    return s


def _variants(rng, s, tool):
    """(variant_name, state, args) list; all variants share the action text."""
    files = sorted(s["files"])
    if tool == "delete_file":
        path = rng.choice(files)
        base = copy.deepcopy(s)
        base["backups"].pop(path, None)
        return [(f"trash_{v}", _set(base, trash_enabled=v), {"path": path}) for v in (True, False)]
    if tool == "edit_file":
        path = rng.choice(files)
        args = {"path": path, "content": _text(rng, path)}
        out = []
        for name in ("backup_current", "history_only", "no_backup_no_history"):
            v = copy.deepcopy(s)
            v["backups"].pop(path, None)
            v["history"].pop(path, None)
            v["flags"]["history_enabled"] = name == "history_only" or (name == "backup_current" and v["flags"]["history_enabled"])
            if name == "backup_current":
                v["backups"][path] = dict(v["files"][path])
            if name == "no_backup_no_history":
                v["history"] = {}
            out.append((name, v, args))
        return out
    if tool == "send_email":
        return [(f"delay_{v}", _set(s, send_delay=v), {"draft_id": rng.choice(sorted(s["drafts"]))}) for v in (True, False)]
    if tool == "transfer_money":
        args = {"to": rng.choice(_theme_of(s)["payees"]), "amount": rng.randrange(500, s["balance"], 100)}
        return [(f"scheduled_{v}", _set(s, transfers_scheduled=v), args) for v in (True, False)]
    if tool == "move_to_savings":
        args = {"amount": rng.randrange(500, s["balance"], 100)}
        return [(f"fee_{v}", _set(s, savings_fee_percent=v), args) for v in (0, 2)]
    if tool == "write_file":
        unused = [f for f in _theme_of(s)["files"] if f not in s["files"] and f not in s["trash"]] or ["new_notes.txt"]
        path = rng.choice(unused)
        args = {"path": path, "content": _text(rng, path)}
        return [("trash_off", _set(s, trash_enabled=False, trash={}), args),
                ("trash_on_nonempty", _set(s, trash_enabled=True, trash=_some_trash(rng, s)), args)]
    if tool == "empty_trash":
        trash = _some_trash(rng, s)
        with_backups = copy.deepcopy(s)
        with_backups["trash"] = trash
        with_backups["backups"].update({p: dict(f) for p, f in trash.items()})
        return [("trash_unbacked", _set(s, trash_enabled=True, trash=trash), {}),
                ("trash_backed_up", _set(with_backups, trash_enabled=True), {})]
    raise ValueError(tool)


def _single(rng, s, tool):
    files = sorted(s["files"])
    v = copy.deepcopy(s)
    if tool in ("read_file", "backup_file"):
        return v, {"path": rng.choice(files)}
    if tool in ("list_files", "check_balance", "view_trash"):
        return v, {}
    if tool == "rename_file":
        path = rng.choice(files)
        return v, {"path": path, "new_path": path.replace(".", "_renamed.", 1)}
    if tool == "restore_from_trash":
        v["flags"]["trash_enabled"] = True
        v["trash"] = v["trash"] or _some_trash(rng, v)
        return v, {"path": rng.choice(sorted(v["trash"]))}
    if tool == "undo_send":
        v["flags"]["send_delay"] = True
        i = rng.choice(sorted(v["drafts"]))
        v["outbox"][i] = v["drafts"].pop(i)
        return v, {"email_id": i}
    if tool == "cancel_transfer":
        v["flags"]["transfers_scheduled"] = True
        amount = rng.randrange(500, 5000, 100)
        to = rng.choice(_theme_of(v)["payees"])
        v["pending"][f"{to}:{amount}"] = {"to": to, "amount": amount}
        return v, {"transfer_id": f"{to}:{amount}"}
    if tool == "withdraw_savings":
        v["savings"] = max(v["savings"], 5000)
        return v, {"amount": rng.randrange(500, v["savings"], 100)}
    if tool == "revert_file":
        path = rng.choice(files)
        v["flags"]["history_enabled"] = True
        v["history"].setdefault(path, [_text(rng, path)])
        return v, {"path": path}
    if tool == "restore_backup":
        path = rng.choice(files)
        if path not in v["backups"]:
            v["backups"][path] = {"content": _text(rng, path), "version": max(1, v["files"][path]["version"] - 1)}
        return v, {"path": path}
    raise ValueError(tool)


def _some_trash(rng, s):
    if s["trash"]:
        return copy.deepcopy(s["trash"])
    name = rng.choice([f for f in _theme_of(s)["files"] if f not in s["files"]] or ["old.txt"])
    return {name: {"content": _text(rng, name), "version": 1}}


def _set(s, trash=None, **flags):
    v = copy.deepcopy(s)
    v["flags"].update(flags)
    if trash is not None:
        v["trash"] = copy.deepcopy(trash)
    if not v["flags"]["trash_enabled"]:
        v["trash"] = {}
    return v


def _theme_of(s):
    return THEMES[s["_theme"]]


def _record(rid, scenario_id, theme, state, tool, args, group_id, variant):
    search = search_restore(state, tool, args, max(HORIZONS))
    by_h = labels_by_horizon(search, HORIZONS)
    clean = {k: v for k, v in state.items() if not k.startswith("_")}
    return {
        "id": rid,
        "environment": "sandbox",
        "state_text": render_state(clean),
        "action_text": action_text(tool, args),
        "label": by_h[3],
        "horizon": 3,
        "source": "sandbox-generator-v1",
        "labels_by_horizon": {str(h): l for h, l in by_h.items()},
        "scenario_id": scenario_id,
        "theme": theme,
        "task_text": THEMES[theme]["task"],
        "group_id": group_id,
        "variant": variant,
        "verb": tool,
        "read_only": tool in READ_ONLY,
        "restore_sequence": [action_text(*a) for a in (search["exact_seq"] or search["approx_seq"] or [])],
        "state_json": clean,
        "action_json": {"tool": tool, "args": args},
    }


def _weighted_sample(rng, weights, k):
    pool, chosen = dict(weights), []
    for _ in range(k):
        tool = rng.choices(list(pool), weights=list(pool.values()))[0]
        chosen.append(tool)
        del pool[tool]
    return chosen


def generate(n_scenarios, seed, pairs_per_scenario=3, singles_per_scenario=1):
    rng = random.Random(seed)
    records, themes = [], sorted(THEMES)
    for sc in range(n_scenarios):
        theme = themes[sc % len(themes)]
        scenario_id = f"sb-{sc:04d}"
        base = make_scenario(rng, theme)
        base["_theme"] = theme
        for g, tool in enumerate(_weighted_sample(rng, PAIR_TOOLS, pairs_per_scenario)):
            for variant, state, args in _variants(rng, base, tool):
                records.append(_record(f"{scenario_id}-p{g}-{variant}", scenario_id, theme, state, tool, args, f"{scenario_id}-p{g}", variant))
        for g, tool in enumerate(rng.sample(SINGLE_TOOLS, singles_per_scenario)):
            state, args = _single(rng, base, tool)
            records.append(_record(f"{scenario_id}-s{g}", scenario_id, theme, state, tool, args, f"{scenario_id}-s{g}", "single"))
    for r in records:
        group = [x["label"] for x in records if x["group_id"] == r["group_id"]]
        r["is_minimal_pair"] = len(group) > 1 and len(set(group)) > 1
    return records


def summarise(records):
    counts = collections.Counter(r["label"] for r in records)
    by_verb = collections.defaultdict(collections.Counter)
    for r in records:
        by_verb[r["verb"]][r["label"]] += 1
    minimal = sum(r["is_minimal_pair"] for r in records)
    lines = [f"rows: {len(records)}", f"labels (h=3): {dict(counts)}",
             f"minimal-pair rows: {minimal} ({minimal / len(records):.0%})",
             f"labels by horizon: " + str({h: dict(collections.Counter(r['labels_by_horizon'][str(h)] for r in records)) for h in HORIZONS}),
             "per verb:"]
    lines += [f"  {v}: {dict(c)}" for v, c in sorted(by_verb.items())]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=int, default=220)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/sandbox.jsonl")
    a = p.parse_args()
    records = generate(a.scenarios, a.seed)
    with open(a.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(summarise(records))


if __name__ == "__main__":
    main()
