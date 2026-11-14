import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from experiments.extract_hidden import model_slug, read_jsonl
from experiments.extract_hidden import stratified_limit


REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {"reversible": 0, "partially_reversible": 1, "irreversible": 2}
CLASS_NAMES = {0: "reversible", 1: "partially_reversible", 2: "irreversible"}


def select_rows(rows, limit, label_key="label"):
    return stratified_limit(rows, limit, label_key=label_key)


def map_label(label, mode):
    normalized = str(label).casefold().replace("-", "_").replace(" ", "_")
    if mode == "binary":
        if normalized == "reversible":
            return 0
        if normalized == "irreversible":
            return 1
        return None
    return LABEL_MAP.get(normalized)


def _horizon_label(row, horizon):
    by_horizon = row.get("labels_by_horizon") or {}
    key = str(horizon)
    if key in by_horizon:
        return by_horizon[key]
    if row.get("horizon") == horizon:
        return row.get("label")
    return None


def align_probabilities(probabilities, model_classes, class_ids):
    aligned = np.zeros((len(probabilities), len(class_ids)), dtype=float)
    class_to_column = {int(label): index for index, label in enumerate(model_classes)}
    for column, label in enumerate(class_ids):
        if label in class_to_column:
            aligned[:, column] = probabilities[:, class_to_column[label]]
    return aligned


def fit_classifier(features, train_labels, test_features, class_ids, seed):
    labels = np.asarray(train_labels, dtype=int)
    if len(np.unique(labels)) < 2:
        estimator = DummyClassifier(strategy="most_frequent")
    else:
        estimator = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
            random_state=seed,
        )
    estimator.fit(features, labels)
    prediction = estimator.predict(test_features).astype(int)
    raw_probabilities = estimator.predict_proba(test_features)
    probabilities = align_probabilities(raw_probabilities, estimator.classes_, class_ids)
    return prediction, probabilities


def _auc(y_true, probabilities, class_ids):
    if set(map(int, y_true)) != set(map(int, class_ids)):
        return None
    try:
        if len(class_ids) == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(
            y_true, probabilities, labels=class_ids, multi_class="ovr", average="macro"
        ))
    except ValueError:
        return None


def _sample_metrics(y_true, prediction, class_ids):
    y_true = np.asarray(y_true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    recalls = []
    f1s = []
    for label in class_ids:
        true_mask = y_true == label
        pred_mask = prediction == label
        support = int(true_mask.sum())
        if support:
            recalls.append(float(np.sum(true_mask & pred_mask) / support))
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return (
        float(np.mean(y_true == prediction)),
        float(np.mean(recalls)) if recalls else None,
        float(np.mean(f1s)) if f1s else None,
    )


def bootstrap_intervals(y_true, prediction, probabilities, class_ids, resamples, seed):
    y_true = np.asarray(y_true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    count = len(y_true)
    if count == 0:
        return {"balanced_accuracy": None, "macro_f1": None, "auroc": None, "auroc_valid_resamples": 0}

    rng = np.random.default_rng(seed)
    values = {"accuracy": [], "balanced_accuracy": [], "macro_f1": [], "auroc": []}
    for _ in range(resamples):
        indices = rng.integers(0, count, size=count)
        sample_y = y_true[indices]
        sample_pred = prediction[indices]
        accuracy, balanced, macro = _sample_metrics(sample_y, sample_pred, class_ids)
        if accuracy is not None:
            values["accuracy"].append(accuracy)
        if balanced is not None:
            values["balanced_accuracy"].append(balanced)
        if macro is not None:
            values["macro_f1"].append(macro)
        auc = _auc(sample_y, probabilities[indices], class_ids)
        if auc is not None:
            values["auroc"].append(auc)

    intervals = {}
    for metric, scores in values.items():
        intervals[metric] = (
            [float(x) for x in np.quantile(scores, [0.025, 0.975])]
            if scores else None
        )
    intervals["auroc_valid_resamples"] = len(values["auroc"])
    return intervals


def evaluate(y_true, prediction, probabilities, class_ids, bootstrap_resamples, seed):
    y_true = np.asarray(y_true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    auc = _auc(y_true, probabilities, class_ids)
    intervals = bootstrap_intervals(
        y_true, prediction, probabilities, class_ids, bootstrap_resamples, seed
    )
    accuracy, balanced_accuracy, macro_f1 = _sample_metrics(y_true, prediction, class_ids)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "auroc": auc,
        "balanced_accuracy_ci95": intervals["balanced_accuracy"],
        "macro_f1_ci95": intervals["macro_f1"],
        "auroc_ci95": intervals["auroc"],
        "auroc_valid_resamples": intervals["auroc_valid_resamples"],
        "bootstrap_resamples": bootstrap_resamples,
    }


def selectivity_interval(y_true, task_prediction, control_true, control_prediction, resamples, seed):
    y_true = np.asarray(y_true, dtype=int)
    task_prediction = np.asarray(task_prediction, dtype=int)
    control_true = np.asarray(control_true, dtype=int)
    control_prediction = np.asarray(control_prediction, dtype=int)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        task_accuracy = float(np.mean(y_true[indices] == task_prediction[indices]))
        control_accuracy = float(np.mean(control_true[indices] == control_prediction[indices]))
        differences.append(task_accuracy - control_accuracy)
    return [float(x) for x in np.quantile(differences, [0.025, 0.975])] if differences else None


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _verb(row):
    if row.get("verb"):
        return str(row["verb"]).casefold()
    action = str(row.get("action_text", "")).casefold().strip()
    return "pick up" if action.startswith("pick up ") else action.split(maxsplit=1)[0]


def _load_examples(index_path, extraction_dir, limit, horizon, minimal_pairs_only, failures_path):
    index_rows, parse_errors = read_jsonl(index_path)
    if parse_errors:
        for item in parse_errors:
            _append(failures_path, {"stage": "index-parse", **item})
        raise ValueError(f"Index contains {len(parse_errors)} malformed JSONL rows")
    eligible = [
        {**row, "_requested_horizon_label": _horizon_label(row, horizon)}
        for row in index_rows
        if (not minimal_pairs_only or row.get("is_minimal_pair", False))
        and _horizon_label(row, horizon) is not None
    ]
    selected = select_rows(eligible, limit, label_key="_requested_horizon_label")
    examples = []
    failures = 0
    horizon_missing = sum(
        1 for row in index_rows if _horizon_label(row, horizon) is None
    )
    for row in selected:
        label = _horizon_label(row, horizon)
        try:
            array_path = extraction_dir / row["array_path"]
            array = np.load(array_path, allow_pickle=False)
            if array.ndim != 3 or array.shape[1] != 2:
                raise ValueError(f"Expected [layers, 2, hidden], got {array.shape}")
            if not row.get("scenario_id"):
                raise ValueError("Missing scenario_id for grouped split")
            if not row.get("action_text"):
                raise ValueError("Missing action_text for lexical baseline")
            examples.append({**row, "label": label, "array": array})
        except Exception as exc:
            failures += 1
            _append(failures_path, {
                "stage": "load-example",
                "id": row.get("id"),
                "array_path": str(row.get("array_path")),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    return examples, len(selected), failures, horizon_missing


def _record_metrics(base, y_true, prediction, probabilities, class_ids, resamples, seed):
    result = {**base}
    result.update(evaluate(y_true, prediction, probabilities, class_ids, resamples, seed))
    result["n_test"] = int(len(y_true))
    result["status"] = "ok"
    return result


def _fit_and_score(x_train, y_train, x_test, y_test, class_ids, seed, base, resamples):
    prediction, probabilities = fit_classifier(x_train, y_train, x_test, class_ids, seed)
    return _record_metrics(base, y_test, prediction, probabilities, class_ids, resamples, seed), prediction, probabilities


def _aggregate(records, path):
    groups = defaultdict(list)
    for row in records:
        if row.get("status") == "ok":
            key = tuple(row.get(k) for k in ("run_id", "label_mode", "horizon", "position", "layer", "control"))
            groups[key].append(row)
    with path.open("a", encoding="utf-8") as stream:
        for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
            summary = {
                "run_id": key[0],
                "label_mode": key[1],
                "horizon": key[2],
                "position": key[3],
                "layer": key[4],
                "control": key[5],
                "split_count": len(rows),
            }
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "selectivity"):
                values = [row[metric] for row in rows if row.get(metric) is not None]
                summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
                summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
                summary[f"{metric}_valid_splits"] = len(values)
            stream.write(json.dumps(summary, sort_keys=True) + "\n")
        stream.flush()


def run_probes(examples, model_name, mode, positions, horizon, seed, bootstrap_resamples, scores_path, summary_path, predictions_path, failures_path):
    failures = 0
    if len(examples) < 2:
        raise ValueError("At least two labeled rows are needed")
    layer_counts = {int(example["array"].shape[0]) for example in examples}
    hidden_sizes = {int(example["array"].shape[2]) for example in examples}
    if len(layer_counts) != 1 or len(hidden_sizes) != 1:
        raise ValueError("Extraction arrays do not share layer and hidden dimensions")

    if mode == "binary":
        class_ids = [0, 1]
    else:
        class_ids = [0, 1, 2]
    unknown_labels = sorted({
        str(row.get("label"))
        for row in examples
        if str(row.get("label", "")).casefold().replace("-", "_").replace(" ", "_") not in LABEL_MAP
    })
    if unknown_labels:
        raise ValueError(f"Unsupported label values: {unknown_labels}")
    labels = np.array([map_label(row["label"], mode) for row in examples], dtype=object)
    keep = np.array([value is not None for value in labels], dtype=bool)
    excluded_count = int((~keep).sum())
    examples = [row for row, include in zip(examples, keep) if include]
    labels = np.asarray([int(value) for value in labels[keep]], dtype=int)
    if len(examples) < 2:
        raise ValueError(f"Too few labels for {mode}; excluded {excluded_count} partial/unknown labels")

    groups = np.array([str(row["scenario_id"]) for row in examples])
    texts = np.array([str(row["action_text"]) for row in examples])
    verbs = np.array([_verb(row) for row in examples])
    split_maker = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=seed)
    split_iter = list(split_maker.split(np.zeros(len(examples)), labels, groups))
    if len(set(groups)) < 2:
        raise ValueError("At least two scenario groups are needed for scenario-level splits")

    position_indices = {"last": 0, "mean": 1}
    results = []
    per_split_probe = {}
    rng = np.random.default_rng(seed)
    verb_values = sorted(set(verbs))
    verb_to_random = {verb: int(rng.choice(class_ids)) for verb in verb_values}
    random_verb_labels = np.array([verb_to_random[verb] for verb in verbs], dtype=int)
    run_tag = hashlib.sha256(
        f"{model_name}|{mode}|{horizon}|{seed}|{len(examples)}|{','.join(row['id'] for row in examples)}".encode()
    ).hexdigest()[:16]
    arrays = np.stack([row["array"] for row in examples], axis=0)
    layer_count = arrays.shape[1]

    for split_number, (train_idx, test_idx) in enumerate(split_iter, start=1):
        y_train, y_test = labels[train_idx], labels[test_idx]
        split_base = {
            "run_id": run_tag,
            "model": model_name,
            "label_mode": mode,
            "horizon": horizon,
            "split": split_number,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_scenarios": int(len(set(groups[train_idx]))),
            "n_test_scenarios": int(len(set(groups[test_idx]))),
            "excluded_partial_or_unknown": excluded_count,
        }
        try:
            majority = int(np.bincount(y_train, minlength=len(class_ids)).argmax())
            majority_pred = np.full(len(test_idx), majority, dtype=int)
            majority_prob = np.zeros((len(test_idx), len(class_ids)), dtype=float)
            majority_prob[:, majority] = 1.0
            majority_row = _record_metrics(
                {**split_base, "position": "n/a", "layer": -1, "control": "majority"},
                y_test, majority_pred, majority_prob, class_ids, bootstrap_resamples,
                seed + split_number,
            )
            results.append(majority_row)
            _append(scores_path, majority_row)

            vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True)
            bow_train = vectorizer.fit_transform(texts[train_idx])
            bow_test = vectorizer.transform(texts[test_idx])
            bow_pred, bow_prob = fit_classifier(
                bow_train, y_train, bow_test, class_ids, seed + split_number
            )
            bow_row = _record_metrics(
                {**split_base, "position": "n/a", "layer": -1, "control": "action-text-bow"},
                y_test, bow_pred, bow_prob, class_ids, bootstrap_resamples,
                seed + split_number + 1,
            )
            results.append(bow_row)
            _append(scores_path, bow_row)
        except Exception as exc:
            failures += 1
            _append(failures_path, {
                **split_base,
                "stage": "baseline",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

        for position in positions:
            for layer in range(layer_count):
                x_train = arrays[train_idx, layer, position_indices[position], :].astype(np.float32)
                x_test = arrays[test_idx, layer, position_indices[position], :].astype(np.float32)
                base = {**split_base, "position": position, "layer": layer}
                try:
                    probe_row, probe_pred, probe_prob = _fit_and_score(
                        x_train, y_train, x_test, y_test, class_ids,
                        seed + split_number + layer, {**base, "control": "probe"},
                        bootstrap_resamples,
                    )
                    results.append(probe_row)
                    _append(scores_path, probe_row)
                    per_split_probe[(split_number, position, layer)] = (probe_row, probe_pred)
                    for row_index, predicted, probabilities in zip(test_idx, probe_pred, probe_prob):
                        example = examples[int(row_index)]
                        predicted = int(predicted)
                        _append(predictions_path, {
                            "id": str(example["id"]),
                            "scenario_id": example["scenario_id"],
                            "group_id": example.get("group_id"),
                            "label_mode": mode,
                            "horizon": horizon,
                            "split": split_number,
                            "position": position,
                            "layer": layer,
                            "control": "probe",
                            "gold_label": example["label"],
                            "predicted_class": predicted,
                            "predicted_label": CLASS_NAMES[predicted],
                            "probabilities": [float(value) for value in probabilities],
                        })

                    shuffled = y_train.copy()
                    np.random.default_rng(seed + split_number * 1000 + layer).shuffle(shuffled)
                    shuffled_pred, shuffled_prob = fit_classifier(
                        x_train, shuffled, x_test, class_ids, seed + layer
                    )
                    shuffled_row = _record_metrics(
                        {**base, "control": "shuffled-label"}, y_test,
                        shuffled_pred, shuffled_prob, class_ids,
                        bootstrap_resamples, seed + split_number + layer + 2,
                    )
                    results.append(shuffled_row)
                    _append(scores_path, shuffled_row)

                    control_train = random_verb_labels[train_idx]
                    control_test = random_verb_labels[test_idx]
                    control_pred, control_prob = fit_classifier(
                        x_train, control_train, x_test, class_ids, seed + layer + 3
                    )
                    control_row = _record_metrics(
                        {**base, "control": "verb-random-label"}, control_test,
                        control_pred, control_prob, class_ids,
                        bootstrap_resamples, seed + split_number + layer + 4,
                    )
                    task_accuracy = probe_row["accuracy"]
                    control_accuracy = control_row["accuracy"]
                    control_row["task_accuracy"] = task_accuracy
                    control_row["control_accuracy"] = control_accuracy
                    control_row["selectivity"] = task_accuracy - control_accuracy
                    control_row["selectivity_ci95"] = selectivity_interval(
                        y_test, probe_pred, control_test, control_pred,
                        bootstrap_resamples, seed + split_number + layer + 5,
                    )
                    results.append(control_row)
                    _append(scores_path, control_row)
                except Exception as exc:
                    failures += 1
                    _append(failures_path, {
                        **base,
                        "stage": "probe-fit",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })

    if results:
        _aggregate(results, summary_path)
    return {
        "run_id": run_tag,
        "label_mode": mode,
        "rows": len(examples),
        "scenarios": len(set(groups)),
        "classes": {str(label): int(np.sum(labels == label)) for label in class_ids},
        "excluded_partial_or_unknown": excluded_count,
        "splits": len(split_iter),
        "result_rows": len(results),
        "failures": failures,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train linear probes and controls from extracted states")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--extraction-dir")
    parser.add_argument("--index")
    parser.add_argument("--output-dir")
    parser.add_argument("--task", choices=("binary", "three-class", "both"), default="both")
    parser.add_argument("--horizon", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--position", choices=("last", "mean", "both"), default="both")
    parser.add_argument("--minimal-pairs-only", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.bootstrap_resamples < 1:
        parser.error("limit and bootstrap-resamples must be positive")

    extraction_dir = Path(args.extraction_dir) if args.extraction_dir else REPO_ROOT / "results" / "extractions" / model_slug(args.model)
    if not extraction_dir.is_absolute():
        extraction_dir = REPO_ROOT / extraction_dir
    index_path = Path(args.index) if args.index else extraction_dir / "index.jsonl"
    if not index_path.is_absolute():
        index_path = REPO_ROOT / index_path
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / "probes" / model_slug(args.model)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failures.jsonl"
    scores_path = output_dir / "scores.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.jsonl"
    index_rows, parse_errors = read_jsonl(index_path)
    if parse_errors:
        for item in parse_errors:
            _append(failures_path, {"stage": "index-parse", **item})
        raise SystemExit(f"Index has {len(parse_errors)} malformed JSONL rows; see {failures_path}")

    examples, selected_count, load_failures, horizon_missing = _load_examples(
        index_path, extraction_dir, args.limit, args.horizon,
        args.minimal_pairs_only, failures_path,
    )
    if not examples:
        raise SystemExit("No usable labeled hidden-state examples; see failures.jsonl")

    modes = ("binary", "three-class") if args.task == "both" else (args.task,)
    positions = ("last", "mean") if args.position == "both" else (args.position,)
    started = time.perf_counter()
    summaries = []
    failures = load_failures
    for mode in modes:
        try:
            summary = run_probes(
                examples, args.model, mode, positions, args.horizon, args.seed,
                args.bootstrap_resamples, scores_path, summary_path, predictions_path,
                failures_path,
            )
            summaries.append(summary)
            failures += summary["failures"]
        except Exception as exc:
            failures += 1
            _append(failures_path, {
                "stage": "probe-run",
                "label_mode": mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
    summary_record = {
        "model": args.model,
        "limit": args.limit,
        "selected_index_rows": selected_count,
        "usable_examples": len(examples),
        "rows_missing_requested_horizon": horizon_missing,
        "minimal_pairs_only": args.minimal_pairs_only,
        "horizon": args.horizon,
        "positions": list(positions),
        "requested_splits": 3,
        "bootstrap_resamples": args.bootstrap_resamples,
        "mode_summaries": summaries,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "scores": str(scores_path),
        "predictions": str(predictions_path),
        "aggregate_summary": str(summary_path),
        "failure_log": str(failures_path),
    }
    _append(output_dir / "runs.jsonl", summary_record)
    print(json.dumps(summary_record, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
