"""ScienceWorld analyses (docs/analysis_plan.md, ScienceWorld section).

Probes with controls on deduplicated ScienceWorld rows, cross-environment transfer in both directions,
and the ScienceWorld horizon ablation. Reuses the sandbox analysis code; writes results/analysis/<model>/sw/.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from experiments.analyze import (
    POSITIONS, SEED, TARGETS, _fit, _predict, auroc, class_name, class_scores, group_bootstrap_auroc,
    group_bootstrap_ci, load_layer, load_rows, log, make_splits, metrics, plot_layers, run_auroc, run_probes,
    select_c, summarize_probes, write_json,
)
from experiments.model_runtime import REPO_ROOT, read_jsonl
from experiments.probes import map_label

SW_SUBSETS = ("all",)
SW_GROUP_KEY = "group_id"
GROUP_KEYS = {"sandbox": "scenario_id", "scienceworld": SW_GROUP_KEY}


def dedup_rows(rows):
    """Drop exact duplicates (same state_text, action_text, label), keeping the first in file order."""
    seen, kept, removed = set(), [], []
    for row in rows:
        key = (row["state_text"], row["action_text"], row["label"])
        (removed if key in seen else kept).append(row)
        seen.add(key)
    return kept, removed


def scienceworld_rows(model_dir, data_path):
    data, errors = read_jsonl(data_path)
    if errors:
        raise SystemExit(f"{data_path} has {len(errors)} malformed rows")
    data = [r for r in data if r.get("environment") == "scienceworld"]
    kept, removed = dedup_rows(data)
    keep_ids = {str(r["id"]) for r in kept}
    index = load_rows(model_dir / "sw")
    rows = [r for r in index if str(r["id"]) in keep_ids]
    report = {
        "input_rows": len(data), "duplicates_removed": len(removed), "removed_ids": [r["id"] for r in removed],
        "rows_after_dedup": len(kept), "rows_with_hidden_states": len(rows),
        "rows_missing_hidden_states": sorted(keep_ids - {str(r["id"]) for r in index}),
        "class_counts": dict(Counter(r["label"] for r in rows)), "groups": len({r[SW_GROUP_KEY] for r in rows}),
    }
    return rows, report


def layer_set(rows, stride):
    n_layers = int(rows[0]["shape"][0])
    return sorted(set(range(0, n_layers, stride)) | {n_layers - 1})


def horizon_label(row, horizon, target):
    return map_label((row.get("labels_by_horizon") or {}).get(str(horizon)), target)


def labelled(rows, target, horizon=3):
    keep = [i for i, r in enumerate(rows) if horizon_label(r, horizon, target) is not None]
    return keep, np.array([horizon_label(rows[i], horizon, target) for i in keep], dtype=int)


def _counts(labels, target):
    return {class_name(target, k): int(v) for k, v in sorted(Counter(labels.tolist()).items())}


# ---------- (c) cross-environment transfer ----------

def _inner_cv_layer(model_dir, rows, layer, position, labels, groups, target):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=int(os.environ.get("ANALYZE_THREADS", "2"))):
        c, scores = select_c(load_layer(model_dir, rows, layer, position), labels, groups, target)
    return layer, c, scores[c]


def _scored(gold, pred, scores, groups, target, resamples):
    ci, dropped = group_bootstrap_auroc([(gold, scores, groups)], target, resamples)
    return {**metrics(gold, pred, target), **group_bootstrap_ci([(gold, pred, groups)], target, resamples),
            "auroc": auroc(gold, scores, target), "auroc_ci95": ci, "auroc_bootstrap_resamples_dropped": dropped}


def transfer_fit(x_source, y_source, x_target, y_target, g_target, c, target, resamples, seed=SEED):
    """Fit once on all source rows and score once on all target rows, with majority and shuffled-label controls."""
    fits = {
        "probe": _fit(x_source, y_source, c),
        "shuffled_label": _fit(x_source, np.random.default_rng(seed).permutation(y_source), c),
        "majority": (None, int(np.bincount(y_source).argmax())),
    }
    return {name: _scored(y_target, _predict(f, x_target), class_scores(f, x_target, target), g_target, target, resamples)
            for name, f in fits.items()}


def vocabulary_overlap(source_texts, target_texts):
    vocab = lambda texts: set(CountVectorizer().fit(texts).vocabulary_)
    source, target = vocab(source_texts), vocab(target_texts)
    return {"target_action_words": len(target), "shared_words": sorted(source & target),
            "fraction_of_target_words_seen_in_source": len(source & target) / len(target)}


def cross_environment(envs, layers, jobs, resamples):
    from joblib import Parallel, delayed

    result = {"note": "layer and C chosen on source rows only (inner grouped CV); target scored once",
              "lexical_baseline": "not run: a bag-of-words model on action_text cannot transfer across these vocabularies",
              "directions": {}}
    for source, target_env in (("sandbox", "scienceworld"), ("scienceworld", "sandbox")):
        (s_dir, s_rows), (t_dir, t_rows) = envs[source], envs[target_env]
        direction = {"vocabulary_overlap": vocabulary_overlap([r["action_text"] for r in s_rows],
                                                              [r["action_text"] for r in t_rows])}
        for target in TARGETS:
            s_keep, s_y = labelled(s_rows, target)
            t_keep, t_y = labelled(t_rows, target)
            s_sel, t_sel = [s_rows[i] for i in s_keep], [t_rows[i] for i in t_keep]
            s_groups = np.array([r[GROUP_KEYS[source]] for r in s_sel])
            t_groups = np.array([r[GROUP_KEYS[target_env]] for r in t_sel])
            setting = {"source_class_counts": _counts(s_y, target), "target_class_counts": _counts(t_y, target)}
            if len(np.unique(s_y)) < 2 or len(t_y) == 0:
                setting["not_estimable"] = "source has a single class or target has no rows"
                direction[target] = setting
                continue
            for position in POSITIONS:
                inner = list(Parallel(n_jobs=jobs)(
                    delayed(_inner_cv_layer)(s_dir, s_sel, layer, position, s_y, s_groups, target) for layer in layers
                ))
                layer, c, score = max(inner, key=lambda t: (t[2], -t[0]))
                scores = transfer_fit(load_layer(s_dir, s_sel, layer, position), s_y,
                                      load_layer(t_dir, t_sel, layer, position), t_y, t_groups, c, target, resamples)
                setting[position] = {"layer": layer, "C": c, "source_inner_cv_macro_f1": score,
                                     "inner_cv_by_layer": {str(l): s for l, _, s in inner}, **scores}
                log(f"transfer {source}->{target_env} {target} {position}: L{layer} "
                    f"acc {scores['probe']['accuracy']:.3f} bal {scores['probe']['balanced_accuracy']:.3f}")
            direction[target] = setting
        result["directions"][f"{source}->{target_env}"] = direction
    return result


# ---------- (d) horizon ablation ----------

def horizon_ablation(model_dir, rows, layers, resamples):
    features = {(p, l): load_layer(model_dir, rows, l, p) for p in POSITIONS for l in layers}
    result = {"note": "probe only; hidden states do not depend on the horizon", "horizons": {}}
    for horizon in (1, 2, 3):
        by_target = {}
        for target in TARGETS:
            keep, labels = labelled(rows, target, horizon)
            groups = np.array([rows[i][SW_GROUP_KEY] for i in keep])
            entry = {"class_counts": _counts(labels, target), "n_rows": len(keep)}
            splits = make_splits(groups) if len(keep) else []
            if not splits or any(len(np.unique(labels[train])) < 2 for train, _ in splits):
                entry["not_estimable"] = "a training split has a single class"
                by_target[target] = entry
                continue
            probe_split, majority_split, chosen = [], [], []
            for train, test in splits:
                best = None
                for key, x in features.items():
                    x = x[keep]
                    c, scores = select_c(x[train], labels[train], groups[train], target)
                    if best is None or (scores[c], -key[1]) > best[0]:
                        best = ((scores[c], -key[1]), key, c)
                _, (position, layer), c = best
                x = features[(position, layer)][keep]
                pred = _predict(_fit(x[train], labels[train], c), x[test])
                probe_split.append((labels[test], pred, groups[test]))
                majority = int(np.bincount(labels[train]).argmax())
                majority_split.append((labels[test], np.full(len(test), majority), groups[test]))
                chosen.append([position, layer, c])
            for name, per_split in (("probe", probe_split), ("majority", majority_split)):
                values = [metrics(g, p, target) for g, p, _ in per_split]
                entry[name] = {**{f"{m}_mean": float(np.mean([v[m] for v in values])) for m in values[0]},
                               **{f"{m}_std": float(np.std([v[m] for v in values], ddof=1)) for m in values[0]},
                               **group_bootstrap_ci(per_split, target, resamples)}
            entry["train_selected_position_layer_C_per_split"] = chosen
            by_target[target] = entry
            log(f"horizon {horizon} {target}: probe acc {entry['probe']['accuracy_mean']:.3f}, "
                f"majority {entry['majority']['accuracy_mean']:.3f}")
        result["horizons"][str(horizon)] = by_target
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="results/<model>; ScienceWorld arrays are read from its sw/")
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "scienceworld.jsonl"))
    parser.add_argument("--out-dir")
    parser.add_argument("--part", choices=("probes", "transfer", "horizon", "all"), default="all")
    parser.add_argument("--layer-stride", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else REPO_ROOT / "results" / "analysis" / model_dir.name / "sw"
    rows, dedup = scienceworld_rows(model_dir, Path(args.data))
    write_json(out_dir / "sw_rows.json", dedup)
    layers = layer_set(rows, args.layer_stride)
    log(f"{model_dir.name} ScienceWorld: {len(rows)} rows after removing {dedup['duplicates_removed']} duplicates, "
        f"{dedup['groups']} groups, layers {layers}")
    started = time.perf_counter()
    if args.part in ("probes", "all"):
        sw_dir = model_dir / "sw"
        run_probes(sw_dir, out_dir, rows, layers, args.jobs, SW_GROUP_KEY, SW_SUBSETS)
        run_auroc(sw_dir, out_dir, rows, layers, args.bootstrap_resamples, SW_GROUP_KEY, SW_SUBSETS)
        summary_rows, selected = summarize_probes(out_dir, layers, args.bootstrap_resamples, SW_SUBSETS)
        plot_layers(out_dir, summary_rows, selected, f"{model_dir.name} ScienceWorld")
    if args.part in ("transfer", "all"):
        envs = {"sandbox": (model_dir, load_rows(model_dir)), "scienceworld": (model_dir / "sw", rows)}
        write_json(out_dir / "sw_transfer.json", cross_environment(envs, layers, args.jobs, args.bootstrap_resamples))
    if args.part in ("horizon", "all"):
        write_json(out_dir / "sw_horizon.json", horizon_ablation(model_dir / "sw", rows, layers, args.bootstrap_resamples))
    log(f"ScienceWorld analysis done in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
