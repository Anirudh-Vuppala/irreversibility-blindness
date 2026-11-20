"""Post-hoc, exploratory analyses that were not in the original plan (docs/analysis_plan.md, last section).

(a) bag-of-words and TF-IDF baselines on the full extraction prompt text, on the main analysis splits;
(b) leave-one-tool-out probe at the train-selected position and layer;
(c) RQ2 directional metric over reversible/irreversible pairs.
Reads only existing files and writes results/analysis/<model>/posthoc_*.json.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from experiments.analyze import (
    RQ2_V2, SEED, TARGETS, SUBSETS, _ci, _fit, _converged, _predict, class_name, group_bootstrap_ci,
    load_layer, load_rows, log, make_splits, metrics, select_c, setting_rows, write_json,
)
from experiments.model_runtime import REPO_ROOT, read_jsonl
from experiments.rq2_stated import select_stated_pairs

LABEL = "post-hoc, not in the original plan"
ORDINAL = {"reversible": 0, "partially_reversible": 1, "irreversible": 2}
VECTORIZERS = {
    "fulltext_bow": lambda: CountVectorizer(ngram_range=(1, 2), binary=True),
    "fulltext_tfidf": lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
}


def full_prompt_texts(model_dir):
    """Extraction prompt as the model saw it, minus the fixed system line: task, state and candidate action."""
    prompts, _ = read_jsonl(model_dir / "extraction" / "prompts.jsonl")
    return {str(p["id"]): "\n".join(m["content"] for m in p["messages"] if m["role"] != "system") for p in prompts}


def _split_summary(per_split, target, resamples):
    scores = [metrics(np.array(g), np.array(p), target) for g, p, _ in per_split]
    out = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = [s[metric] for s in scores]
        out[f"{metric}_mean"] = float(np.mean(values))
        out[f"{metric}_std"] = float(np.std(values, ddof=1))
    return {**out, **group_bootstrap_ci(per_split, target, resamples), "n_test": [len(g) for g, _, _ in per_split]}


def main_summary_rows(out_dir):
    with (out_dir / "probes_summary.csv").open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _existing(summary_rows, subset, target, control, selected):
    for row in summary_rows:
        if (row["subset"], row["target"], row["control"]) == (subset, target, control) and \
                row["layer"].startswith("train_selected") == selected:
            return {k: row[k] for k in ("layer", "position", "accuracy_mean", "macro_f1_mean", "accuracy_ci95", "n_test")}
    return None


def fulltext_baselines(model_dir, out_dir, rows, resamples):
    texts_by_id = full_prompt_texts(model_dir)
    summary_rows = main_summary_rows(out_dir)
    result = {"label": LABEL, "text": "user and assistant messages of the extraction prompt (task, state, action)",
              "splits": "3 GroupShuffleSplit by scenario_id, 20% test, seed 17 (same as the main analysis)",
              "C_selection": "inner GroupKFold(3) macro-F1 on training rows", "settings": {}}
    for subset in SUBSETS:
        for target in TARGETS:
            keep, labels = setting_rows(rows, subset, target)
            groups = np.array([rows[i]["scenario_id"] for i in keep])
            texts = np.array([texts_by_id[str(rows[i]["id"])] for i in keep])
            splits = make_splits(groups)
            setting = {}
            for name, factory in VECTORIZERS.items():
                def featurize(train_texts, test_texts, factory=factory):
                    vectorizer = factory()
                    return vectorizer.fit_transform(train_texts), vectorizer.transform(test_texts)

                per_split, chosen = [], []
                for train, test in splits:
                    c, _ = select_c(texts[train], labels[train], groups[train], target, featurize=featurize)
                    x_train, x_test = featurize(texts[train], texts[test])
                    pred = _predict(_fit(x_train, labels[train], c, scale=False), x_test)
                    per_split.append((labels[test], pred, groups[test]))
                    chosen.append(c)
                setting[name] = {**_split_summary(per_split, target, resamples), "C": chosen}
            existing_probe = _existing(summary_rows, subset, target, "probe", True)
            setting["existing_probe_train_selected"] = existing_probe
            setting["existing_action_text_bow"] = _existing(summary_rows, subset, target, "lexical_bow", False)
            setting["existing_majority"] = _existing(summary_rows, subset, target, "majority", False)
            setting["n_test_matches_main_analysis"] = (
                json.loads(existing_probe["n_test"]) == setting["fulltext_bow"]["n_test"] if existing_probe else None
            )
            result["settings"][f"{subset}/{target}"] = setting
            log(f"fulltext {subset}/{target}: bow {setting['fulltext_bow']['accuracy_mean']:.3f} "
                f"tfidf {setting['fulltext_tfidf']['accuracy_mean']:.3f}")
    return result


def most_common_choice(positions, layers):
    """Modal (position, layer) over splits; without a unique mode, the modal position and modal layer separately."""
    pairs = Counter(zip(positions, layers)).most_common()
    if len(pairs) == 1 or pairs[0][1] > pairs[1][1]:
        return pairs[0][0], "modal (position, layer) over splits"
    return (Counter(positions).most_common(1)[0][0], Counter(layers).most_common(1)[0][0]), \
        "no modal (position, layer); modal position and modal layer taken separately"


def _held_out_tool(x, labels, groups, tools, tool, target):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=int(os.environ.get("ANALYZE_THREADS", "2"))):
        test, train = tools == tool, tools != tool
        c, _ = select_c(x[train], labels[train], groups[train], target)
        fitted = _fit(x[train], labels[train], c)
        pred = _predict(fitted, x[test])
    gold = labels[test]
    train_majority = int(np.bincount(labels[train]).argmax())
    counts = Counter(gold.tolist())
    return {
        "tool": tool, "n_test": int(test.sum()), "C": c, "converged": _converged(fitted),
        "label_counts": {class_name(target, k): v for k, v in sorted(counts.items())},
        "single_label_tool": len(counts) == 1,
        "probe_accuracy": float(np.mean(pred == gold)),
        "train_majority_class": class_name(target, train_majority),
        "train_majority_accuracy": float(np.mean(gold == train_majority)),
        "within_tool_majority_share": max(counts.values()) / len(gold),
        "predicted_counts": {class_name(target, k): v for k, v in sorted(Counter(pred.tolist()).items())},
    }


def leave_one_tool_out(model_dir, out_dir, rows, jobs):
    from joblib import Parallel, delayed

    summary_rows = main_summary_rows(out_dir)
    result = {"label": LABEL, "subset": "all", "tool_field": "verb",
              "C_selection": "inner GroupKFold(3) by scenario_id on the training tools only", "targets": {}}
    for target in TARGETS:
        selected = _existing(summary_rows, "all", target, "probe", True)
        positions = json.loads(selected["position"])
        layers = json.loads(selected["layer"].split(":", 1)[1])
        (position, layer), rule = most_common_choice(positions, layers)
        keep, labels = setting_rows(rows, "all", target)
        x = load_layer(model_dir, [rows[i] for i in keep], layer, position)
        groups = np.array([rows[i]["scenario_id"] for i in keep])
        tools = np.array([rows[i]["verb"] for i in keep])
        per_tool = list(Parallel(n_jobs=jobs)(
            delayed(_held_out_tool)(x, labels, groups, tools, tool, target) for tool in sorted(set(tools))
        ))
        mixed = [t for t in per_tool if not t["single_label_tool"]]
        n = sum(t["n_test"] for t in per_tool)
        result["targets"][target] = {
            "position": position, "layer": layer, "selection_rule": rule,
            "main_analysis_choice_per_split": list(zip(positions, layers)),
            "per_tool": per_tool,
            "mean_probe_accuracy_over_tools": float(np.mean([t["probe_accuracy"] for t in per_tool])),
            "mean_train_majority_accuracy_over_tools": float(np.mean([t["train_majority_accuracy"] for t in per_tool])),
            "mean_probe_accuracy_over_mixed_label_tools": float(np.mean([t["probe_accuracy"] for t in mixed])),
            "mean_train_majority_accuracy_over_mixed_label_tools": float(np.mean([t["train_majority_accuracy"] for t in mixed])),
            "mean_within_tool_majority_share_over_mixed_label_tools": float(np.mean([t["within_tool_majority_share"] for t in mixed])),
            "pooled_probe_accuracy": sum(t["probe_accuracy"] * t["n_test"] for t in per_tool) / n,
            "single_label_tools": [t["tool"] for t in per_tool if t["single_label_tool"]],
            "n_tools": len(per_tool), "n_rows": n,
        }
        log(f"leave-one-tool-out {target} ({position}, L{layer}): mean {result['targets'][target]['mean_probe_accuracy_over_tools']:.3f}")
    return result


def pair_direction(rev_label, irr_label):
    """1 if the irreversible version is stated as less reversible than the reversible one, 0.5 on a tie, else 0."""
    if rev_label is None or irr_label is None:
        return None
    a, b = ORDINAL[irr_label], ORDINAL[rev_label]
    return 1.0 if a > b else 0.5 if a == b else 0.0


def direction_summary(scores, resamples=1000, seed=SEED):
    values = np.asarray(scores, float)
    if len(values) == 0:
        return {"n_pairs": 0}
    idx = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    return {"n_pairs": len(values), "bootstrap_resamples": resamples, "chance": 0.5,
            "directional_score": float(values.mean()), "directional_score_ci95": _ci(values[idx].mean(axis=1)),
            "tied_fraction": float(np.mean(values == 0.5)),
            "higher_fraction": float(np.mean(values == 1.0)), "lower_fraction": float(np.mean(values == 0.0))}


def rq2_directional(model_dir, data_rows, resamples):
    directory = model_dir / RQ2_V2
    summary = json.loads((directory / "summary.json").read_text())
    responses, _ = read_jsonl(directory / "responses.jsonl")
    answers = {str(r["id"]): r["parsed_label"] for r in responses if r.get("parsed_label")}
    groups = {}
    for row in select_stated_pairs(data_rows, summary["selected"], 3, "three-class", 17):
        groups.setdefault(str(row["group_id"]), []).append(row)
    per_pair, unscored = {}, []
    for gid, members in sorted(groups.items()):
        gold = {r["id"]: (r.get("labels_by_horizon") or {}).get("3", r.get("label")) for r in members}
        rev = next((i for i, g in gold.items() if g == "reversible"), None)
        irr = next((i for i, g in gold.items() if g == "irreversible"), None)
        if rev is None or irr is None:
            continue
        score = pair_direction(answers.get(rev), answers.get(irr))
        if score is None:
            unscored.append(gid)
        else:
            per_pair[gid] = {"reversible_id": rev, "irreversible_id": irr, "stated_reversible_version": answers[rev],
                             "stated_irreversible_version": answers[irr], "score": score}
    tasks, _ = read_jsonl(model_dir / "rq3" / "h3" / "tasks.jsonl")
    rq3_ids = sorted({str(t["pair_id"]) for t in tasks})
    return {
        "label": LABEL, "prompt_version": summary.get("prompt_version"), "source": str(directory),
        "scoring": "stated labels reversible=0, partially_reversible=1, irreversible=2; per pair 1 if "
                   "stated(irreversible version) > stated(reversible version), 0.5 on a tie, 0 if lower",
        "all_rq2_pairs": {**direction_summary([p["score"] for p in per_pair.values()], resamples),
                          "pairs_without_both_parsed": unscored},
        "rq3_pairs": {**direction_summary([per_pair[p]["score"] for p in rq3_ids if p in per_pair], resamples),
                      "rq3_pairs_without_both_parsed": [p for p in rq3_ids if p not in per_pair]},
        "stated_label_pairs": dict(Counter(f"{p['stated_reversible_version']} -> {p['stated_irreversible_version']}"
                                           for p in per_pair.values())),
        "per_pair": per_pair,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--part", choices=("fulltext", "tools", "directional", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else REPO_ROOT / "results" / "analysis" / model_dir.name
    rows = load_rows(model_dir)
    started = time.perf_counter()
    if args.part in ("directional", "all"):
        data_rows, _ = read_jsonl(REPO_ROOT / "data" / "sandbox.jsonl")
        write_json(out_dir / "posthoc_rq2_directional.json", rq2_directional(model_dir, data_rows, args.bootstrap_resamples))
    if args.part in ("fulltext", "all"):
        write_json(out_dir / "posthoc_fulltext_baselines.json", fulltext_baselines(model_dir, out_dir, rows, args.bootstrap_resamples))
    if args.part in ("tools", "all"):
        write_json(out_dir / "posthoc_leave_one_tool_out.json", leave_one_tool_out(model_dir, out_dir, rows, args.jobs))
    log(f"post-hoc done in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
