"""Sample (state, action) items for the human label-quality check and write blind rater sheets.

Per environment: a few matched pairs (same action, different state, different automatic labels), then
label-stratified single rows, at most one per pair/group. Exact duplicates are removed first. The rater
sheets carry no label; the automatic labels go to answer_key.csv only.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

from experiments.analyze_sw import dedup_rows
from experiments.model_runtime import REPO_ROOT, read_jsonl
from experiments.rq2_stated import LABELS, pair_key

SHEET_COLUMNS = ("item_id", "environment", "state_text", "action_text", "label_rater")
KEY_COLUMNS = ("item_id", "environment", "row_id", "pair_ref", "auto_label")
SOURCES = {"sandbox": REPO_ROOT / "data" / "sandbox.jsonl", "scienceworld": REPO_ROOT / "data" / "scienceworld.jsonl"}


def unit_key(row):
    return pair_key(row) or row["id"]


def quotas(n):
    return {label: n // 3 + (i < n % 3) for i, label in enumerate(LABELS)}


def sample_environment(rows, n, n_pairs, rng):
    """Return [(row, pair_ref or None)]: n_pairs matched pairs plus label-stratified singles."""
    rows, _ = dedup_rows(rows)
    units = defaultdict(list)
    for row in rows:
        units[unit_key(row)].append(row)
    pair_units = sorted(k for k, members in units.items() if len({r["label"] for r in members}) > 1)
    chosen = []
    for key in rng.sample(pair_units, n_pairs):
        for label in rng.sample(sorted({r["label"] for r in units[key]}), 2):
            chosen.append((rng.choice([r for r in units[key] if r["label"] == label]), key))
    used = {key for _, key in chosen}
    picked = Counter(r["label"] for r, _ in chosen)
    need = {label: max(0, q - picked[label]) for label, q in quotas(n).items()}
    free = sorted(k for k in units if k not in used)
    rng.shuffle(free)
    singles = []
    for label in sorted(LABELS, key=lambda l: sum(any(r["label"] == l for r in units[k]) for k in free)):
        for key in free:
            if need[label] == 0:
                break
            options = [r for r in units[key] if r["label"] == label]
            if key not in used and options:
                singles.append(rng.choice(options))
                used.add(key)
                need[label] -= 1
    for key in free:
        if len(chosen) + len(singles) >= n:
            break
        if key not in used:
            singles.append(rng.choice(units[key]))
            used.add(key)
    return chosen + [(row, None) for row in singles]


def build_items(sources, per_environment, n_pairs, seed):
    rng = random.Random(seed)
    items = []
    for environment, path in sources.items():
        rows, errors = read_jsonl(path)
        if errors:
            raise SystemExit(f"{path} has {len(errors)} malformed rows")
        rows = [r for r in rows if r.get("environment") == environment]
        refs = {}
        for row, key in sample_environment(rows, per_environment, n_pairs, rng):
            ref = f"{environment}-pair-{refs.setdefault(key, len(refs) + 1)}" if key else ""
            items.append({"environment": environment, "row": row, "pair_ref": ref})
    rng.shuffle(items)
    for i, item in enumerate(items, 1):
        item["item_id"] = f"item-{i:03d}"
    return items


def write_csv(path, columns, records):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "labeling"))
    parser.add_argument("--per-environment", type=int, default=50)
    parser.add_argument("--pairs", type=int, default=5, help="matched pairs per environment")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sandbox", default=str(SOURCES["sandbox"]))
    parser.add_argument("--scienceworld", default=str(SOURCES["scienceworld"]))
    parser.add_argument("--force", action="store_true", help="overwrite existing sheets")
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    sheets = [out / "sheet_rater1.csv", out / "sheet_rater2.csv"]
    if any(p.exists() for p in sheets) and not args.force:
        raise SystemExit("rater sheets already exist (they may hold labels); pass --force to overwrite")
    sources = {"sandbox": Path(args.sandbox), "scienceworld": Path(args.scienceworld)}
    items = build_items(sources, args.per_environment, args.pairs, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    blank = [{"item_id": i["item_id"], "environment": i["environment"], "state_text": i["row"]["state_text"],
              "action_text": i["row"]["action_text"], "label_rater": ""} for i in items]
    for sheet in sheets:
        write_csv(sheet, SHEET_COLUMNS, blank)
    write_csv(out / "answer_key.csv", KEY_COLUMNS,
              [{"item_id": i["item_id"], "environment": i["environment"], "row_id": i["row"]["id"],
                "pair_ref": i["pair_ref"], "auto_label": i["row"]["label"]} for i in items])
    manifest = {
        "seed": args.seed, "per_environment": args.per_environment, "pairs_per_environment": args.pairs,
        "sources": {e: {"path": str(p.resolve().relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT)
                        else str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for e, p in sources.items()},
        "items": len(items), "items_in_pairs": sum(bool(i["pair_ref"]) for i in items),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} items to {out} (answer key: answer_key.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
