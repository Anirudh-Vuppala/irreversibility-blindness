"""Human label agreement: Cohen's kappa between raters and between each rater and the automatic label.

Reads labeling/sheet_rater1.csv, labeling/sheet_rater2.csv (optional), labeling/sheet_rater1_retest.csv
(optional; intra-rater kappa on its items) and labeling/answer_key.csv.
Kappa CIs are percentile bootstraps over items, resampling matched pairs as one unit.
Writes results/analysis/human_agreement.json.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.model_runtime import REPO_ROOT
from experiments.rq2_stated import LABELS

ALIASES = {"r": "reversible", "p": "partially_reversible", "partial": "partially_reversible",
           "partially": "partially_reversible", "i": "irreversible"}
SCOPES = ("overall", "sandbox", "scienceworld")


def normalize(value):
    text = (value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    return ALIASES.get(text, text) or None


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_sheet(path, key_ids, subset=False):
    rows = read_csv(path)
    ids = [r["item_id"] for r in rows]
    if (len(set(ids)) != len(ids) or not set(ids) <= set(key_ids)) if subset else sorted(ids) != sorted(key_ids):
        raise SystemExit(f"{path}: item_ids do not match the answer key")
    labels = {r["item_id"]: normalize(r.get("label_rater")) for r in rows}
    bad = sorted(i for i, label in labels.items() if label is not None and label not in LABELS)
    if bad:
        raise SystemExit(f"{path}: unrecognised labels for {', '.join(bad)} (use {', '.join(LABELS)})")
    return labels


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    observed = float(np.mean(a == b))
    expected = float(sum(np.mean(a == k) * np.mean(b == k) for k in LABELS))
    return None if expected == 1 else (observed - expected) / (1 - expected)


def bootstrap_kappa(a, b, clusters, resamples, seed):
    rng = np.random.default_rng(seed)
    units = sorted(set(clusters))
    members = {u: np.flatnonzero(np.asarray(clusters) == u) for u in units}
    a, b, values = np.asarray(a), np.asarray(b), []
    for _ in range(resamples):
        index = np.concatenate([members[u] for u in rng.choice(units, size=len(units), replace=True)])
        value = kappa(a[index], b[index])
        if value is not None:
            values.append(value)
    ci = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None
    return ci, resamples - len(values)


def compare(first, second, key, scope, resamples, seed):
    ids = sorted(i for i in key if (scope == "overall" or key[i]["environment"] == scope)
                 and first.get(i) and second.get(i))
    result = {"n_items": len(ids)}
    if not ids:
        return result
    a, b = [first[i] for i in ids], [second[i] for i in ids]
    ci, dropped = bootstrap_kappa(a, b, [key[i]["pair_ref"] or i for i in ids], resamples, seed)
    matrix = [[sum(x == r and y == c for x, y in zip(a, b)) for c in LABELS] for r in LABELS]
    return {**result, "observed_agreement": float(np.mean(np.array(a) == np.array(b))), "kappa": kappa(a, b),
            "kappa_ci95": ci, "bootstrap_resamples_dropped": dropped,
            "confusion_matrix": {"rows": "first", "columns": "second", "labels": list(LABELS), "counts": matrix}}


def _optional_sheet(path, key_ids, subset=False):
    if not path or not Path(path).exists():
        return None
    labels = read_sheet(path, key_ids, subset)
    return labels if any(labels.values()) else None


def agreement(sheet1, sheet2, key_path, resamples=1000, seed=17, retest=None):
    key = {r["item_id"]: r for r in read_csv(key_path)}
    auto = {i: normalize(r["auto_label"]) for i, r in key.items()}
    raters = {"rater1": read_sheet(sheet1, list(key))}
    second = _optional_sheet(sheet2, list(key))
    if second:
        raters["rater2"] = second
    retest_labels = _optional_sheet(retest, list(key), subset=True)
    sources = {**raters, "auto": auto, "rater1_retest": retest_labels}
    pairs = ([("rater1", "rater1_retest")] * bool(retest_labels) + [("rater1", "rater2")] * ("rater2" in raters)
             + [(r, "auto") for r in raters])
    return {
        "note": "Cohen's kappa, three classes; 95% percentile bootstrap over items with matched pairs resampled "
                "together; comparisons use items labelled by both sides",
        "bootstrap_resamples": resamples, "seed": seed, "raters": sorted(raters),
        "rater2_missing": "rater2" not in raters, "retest_missing": not retest_labels,
        "retest_items_labelled": sum(v is not None for v in (retest_labels or {}).values()),
        "items_labelled": {r: sum(v is not None for v in labels.values()) for r, labels in raters.items()},
        "items_total": len(key),
        "comparisons": {f"{x}_vs_{y}": {s: compare(sources[x], sources[y], key, s, resamples, seed) for s in SCOPES}
                        for x, y in pairs},
    }


def main(argv=None):
    labeling = REPO_ROOT / "labeling"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater1", default=str(labeling / "sheet_rater1.csv"))
    parser.add_argument("--rater2", default=str(labeling / "sheet_rater2.csv"))
    parser.add_argument("--retest", default=str(labeling / "sheet_rater1_retest.csv"))
    parser.add_argument("--key", default=str(labeling / "answer_key.csv"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "analysis" / "human_agreement.json"))
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    result = agreement(args.rater1, args.rater2, args.key, args.bootstrap_resamples, retest=args.retest)
    if not result["items_labelled"]["rater1"]:
        raise SystemExit(f"{args.rater1} has no labels yet")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, scopes in result["comparisons"].items():
        for scope, r in scopes.items():
            if r["n_items"]:
                ci = r["kappa_ci95"]
                kappa_text = "undefined" if r["kappa"] is None else f"{r['kappa']:.3f}"
                ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
                print(f"{name:24s} {scope:12s} n={r['n_items']:3d} kappa={kappa_text} {ci_text}")
    if result["rater2_missing"]:
        print("rater2 sheet missing or empty: no rater1 vs rater2 or rater2 vs auto")
    if result["retest_missing"]:
        print("retest sheet missing or empty: no intra-rater kappa")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
