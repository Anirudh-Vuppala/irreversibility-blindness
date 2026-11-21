"""Final report tables and figures from existing result files only (no model, no GPU, no refits).

Writes results/analysis/final/ (figures, CSVs, ablations.tex, headline_numbers.json) and copies the figures,
CSVs and the LaTeX table into docs/figures/. Every quoted number is recorded in headline_numbers.json with the
file it came from. The one new summary is the token-position ablation: the train-selected layer within each
position, computed from the saved per-layer probe files with the same selection rule and bootstrap.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.analyze import (
    POSITIONS, TARGETS, _summary_row, load_probe_files, select_per_split,
)
from experiments.model_runtime import REPO_ROOT

MODELS = {"Qwen_Qwen2.5-7B-Instruct": "Qwen2.5-7B", "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
CONDITIONS = {"neutral": "neutral", "explicit-carefulness": "careful"}
HEADLINE_CAUTION = "spec_check_ask_backup"
COLORS = {"probe": "#2a78d6", "shuffled_label": "#eb6834", "control_task": "#1baf7a", "lexical_bow": "#8a5cc2"}
MODEL_COLORS = ("#2a78d6", "#eb6834")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# ---------- reading ----------

class Sources:
    """Reads result files relative to the repo and records every quoted number with its source."""

    def __init__(self, root):
        self.root = Path(root)
        self.numbers = {}

    def path(self, relative):
        return self.root / relative

    def json(self, relative):
        return json.loads(self.path(relative).read_text())

    def csv(self, relative):
        with open(self.path(relative), newline="") as handle:
            return list(csv.DictReader(handle))

    def put(self, name, value, source, field, note=None):
        self.numbers[name] = {"value": value, "source": str(source), "field": field, **({"note": note} if note else {})}
        return value


def num(value):
    return None if value in ("", None) else float(value)


def ci(value):
    if value in ("", None):
        return None
    return json.loads(value) if isinstance(value, str) else list(value)


def modal(values):
    return Counter(values).most_common(1)[0][0]


def selected_rows(summary, subset, target):
    """Train-selected rows (one per control) and the modal selected position."""
    rows = {r["control"]: r for r in summary if r["subset"] == subset and r["target"] == target
            and r["layer"].startswith("train_selected")}
    position = modal(json.loads(rows["probe"]["position"]))
    return rows, position


def baseline(summary, subset, target, control):
    return next(r for r in summary if r["subset"] == subset and r["target"] == target and r["control"] == control)


def per_layer(summary, subset, target, control, position):
    rows = [r for r in summary if r["subset"] == subset and r["target"] == target and r["control"] == control
            and r["position"] == position and r["layer"].isdigit()]
    return sorted(rows, key=lambda r: int(r["layer"]))


def rule_of_three(value, interval, n):
    """A CI that is all zeros on n pairs is reported as 0 of n with the rule-of-three upper bound 3/n."""
    if value == 0 and interval is not None and list(interval) == [0, 0] and n:
        return {"text": f"0 of {n} pairs", "upper_bound": 3 / n}
    return None


# ---------- plotting helpers ----------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                         "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _grid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _errors(values, intervals):
    low = [v - i[0] if i else 0 for v, i in zip(values, intervals)]
    high = [i[1] - v if i else 0 for v, i in zip(values, intervals)]
    return np.array([low, high])


# ---------- 1. balanced accuracy vs layer ----------

def layer_panels(src, model, environment):
    base = f"results/analysis/{model}" + ("/sw" if environment == "scienceworld" else "")
    return base, src.csv(f"{base}/probes_summary.csv")


def figure_layers(src, out, target):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for row, model in enumerate(MODELS):
        for col, environment in enumerate(("sandbox", "scienceworld")):
            ax = axes[row, col]
            base, summary = layer_panels(src, model, environment)
            source = f"{base}/probes_summary.csv"
            chosen, position = selected_rows(summary, "all", target)
            other = next(p for p in POSITIONS if p != position)
            for control, style in (("probe", "-"), ("shuffled_label", "-"), ("control_task", "-")):
                rows = per_layer(summary, "all", target, control, position)
                x = [int(r["layer"]) for r in rows]
                y = [num(r["balanced_accuracy_mean"]) for r in rows]
                band = [ci(r["balanced_accuracy_ci95"]) for r in rows]
                label = {"probe": "probe, train-selected position", "shuffled_label": "shuffled-label probe",
                         "control_task": "control task (own labels)"}[control]
                ax.plot(x, y, style, color=COLORS[control], linewidth=2 if control == "probe" else 1.2, label=label)
                ax.fill_between(x, [b[0] for b in band], [b[1] for b in band], color=COLORS[control], alpha=0.15,
                                linewidth=0)
                if control == "probe":
                    ax.plot([x[0]], [y[0]], "o", color=COLORS["probe"], markersize=6, label="layer 0")
            rows = per_layer(summary, "all", target, "probe", other)
            ax.plot([int(r["layer"]) for r in rows], [num(r["balanced_accuracy_mean"]) for r in rows], ":",
                    color=COLORS["probe"], linewidth=1, label="probe, other position")
            for control, style, color in (("majority", "--", INK_2), ("lexical_bow", "-.", COLORS["lexical_bow"])):
                b = baseline(summary, "all", target, control)
                value, band = num(b["balanced_accuracy_mean"]), ci(b["balanced_accuracy_ci95"])
                ax.axhline(value, linestyle=style, color=color, linewidth=1.2,
                           label={"majority": "majority class", "lexical_bow": "bag-of-words (action text)"}[control])
                ax.axhspan(band[0], band[1], color=color, alpha=0.08, linewidth=0)
            layers = json.loads(chosen["probe"]["layer"].split(":", 1)[1])
            value = num(chosen["probe"]["balanced_accuracy_mean"])
            src.put(f"{model}/{environment}/{target}/probe_train_selected/balanced_accuracy", value, source,
                    f"control=probe, subset=all, target={target}, layer={chosen['probe']['layer']}")
            ax.set_title(f"{MODELS[model]}, {environment}: {position} token selected\ntrain-selected layers {layers}, "
                         f"balanced accuracy {value:.3f}", fontsize=9)
            ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("balanced accuracy (test)")
            _grid(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Probe balanced accuracy vs layer ({target}, all rows; bands: 95% group-bootstrap CI)")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    path = out / f"fig1_probe_balanced_accuracy_vs_layer_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------- 2. stated vs probed vs behaviour ----------

def stated_probed_behaviour(src):
    records = []
    for model in MODELS:
        base = f"results/analysis/{model}"
        summary = src.csv(f"{base}/probes_summary.csv")
        for target in TARGETS:
            chosen, _ = selected_rows(summary, "all", target)
            probe = chosen["probe"]
            for metric, column in (("accuracy", "accuracy"), ("balanced accuracy", "balanced_accuracy"),
                                   ("AUROC", "auroc")):
                value = num(probe[f"{column}_mean"])
                interval = ci(probe[f"{column}_ci95"])
                name = f"{model}/sandbox/{target}/probe_train_selected/{column}"
                src.put(name, value, f"{base}/probes_summary.csv",
                        f"control=probe, subset=all, target={target}, layer={probe['layer']}, column={column}_mean")
                majority = baseline(summary, "all", target, "majority")
                records.append({"model": MODELS[model], "panel": "probe (sandbox, train-selected layer)",
                                "measure": f"{target} {metric}", "value": value,
                                "ci_low": interval[0] if interval else None, "ci_high": interval[1] if interval else None,
                                "n": probe["n_test"], "reference": num(majority[f"{column}_mean"]),
                                "reference_name": "majority class", "source": f"{base}/probes_summary.csv"})
        rq2 = src.json(f"{base}/rq2_summary.json")
        accuracy = src.put(f"{model}/sandbox/rq2/stated_accuracy_three_class", rq2["stated_answer_accuracy"],
                           f"{base}/rq2_summary.json", "stated_answer_accuracy")
        records.append({"model": MODELS[model], "panel": "stated (RQ2, prompt v2)", "measure": "three-class accuracy",
                        "value": accuracy, "ci_low": None, "ci_high": None, "n": rq2["n_scored"],
                        "reference": None, "reference_name": "", "source": f"{base}/rq2_summary.json"})
        head = src.json(f"{base}/headline_h3_plus_extra.json")
        k_all = head["headline_merged"]["K_all_rq2_pairs"]
        src.put(f"{model}/sandbox/rq2/K_all_pairs", k_all["K"], f"{base}/headline_h3_plus_extra.json",
                "headline_merged.K_all_rq2_pairs.K")
        src.put(f"{model}/sandbox/rq2/K_all_pairs_ci95", k_all["K_ci95"], f"{base}/headline_h3_plus_extra.json",
                "headline_merged.K_all_rq2_pairs.K_ci95")
        src.put(f"{model}/sandbox/rq2/K_all_pairs_n", k_all["n_pairs"], f"{base}/headline_h3_plus_extra.json",
                "headline_merged.K_all_rq2_pairs.n_pairs")
        records.append({"model": MODELS[model], "panel": "stated (RQ2, prompt v2)", "measure": "K (all RQ2 pairs)",
                        "value": k_all["K"], "ci_low": k_all["K_ci95"][0], "ci_high": k_all["K_ci95"][1],
                        "n": k_all["n_pairs"], "reference": None, "reference_name": "",
                        "source": f"{base}/headline_h3_plus_extra.json"})
        caution = head["rq3_merged"]["neutral"]["caution"][HEADLINE_CAUTION]
        combo = head["headline_merged"]["combinations"][f"neutral/{HEADLINE_CAUTION}"]
        field = f"rq3_merged.neutral.caution.{HEADLINE_CAUTION}"
        for measure, key in (("caution on irreversible", "irreversible_caution_rate"),
                             ("caution on reversible", "reversible_over_caution_rate")):
            value = src.put(f"{model}/sandbox/rq3_merged/neutral/{key}", caution[key],
                            f"{base}/headline_h3_plus_extra.json", f"{field}.{key}")
            records.append({"model": MODELS[model], "panel": "behaviour (RQ3, 60 pairs, neutral)", "measure": measure,
                            "value": value, "ci_low": None, "ci_high": None,
                            "n": caution["n_complete_reversible_irreversible_pairs"], "reference": None,
                            "reference_name": "", "source": f"{base}/headline_h3_plus_extra.json"})
        records.append({"model": MODELS[model], "panel": "behaviour (RQ3, 60 pairs, neutral)", "measure": "A (paired)",
                        "value": combo["A"], "ci_low": combo["A_ci95"][0], "ci_high": combo["A_ci95"][1],
                        "n": combo["n_pairs"], "reference": None, "reference_name": "",
                        "source": f"{base}/headline_h3_plus_extra.json"})
    return records


def figure_stated_probed_behaviour(records, out):
    plt = _plt()
    panels = list(dict.fromkeys(r["panel"] for r in records))
    widths = [len({r["measure"] for r in records if r["panel"] == p}) + 1 for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 4.4), gridspec_kw={"width_ratios": widths})
    for ax, panel in zip(axes, panels):
        rows = [r for r in records if r["panel"] == panel]
        measures = list(dict.fromkeys(r["measure"] for r in rows))
        x = np.arange(len(measures))
        for i, model in enumerate(MODELS.values()):
            values = [next(r for r in rows if r["model"] == model and r["measure"] == m) for m in measures]
            heights = [v["value"] for v in values]
            intervals = [[v["ci_low"], v["ci_high"]] if v["ci_low"] is not None else None for v in values]
            offset = (i - 0.5) * 0.38
            ax.bar(x + offset, heights, 0.36, color=MODEL_COLORS[i], label=model, edgecolor="white", linewidth=1)
            for xi, v, height, interval in zip(x + offset, values, heights, intervals):
                degenerate = rule_of_three(v["value"], interval, v["n"] if isinstance(v["n"], int) else None)
                if degenerate:
                    interval = [0, degenerate["upper_bound"]]
                    text = f"0 of {v['n']}\n≤ {degenerate['upper_bound']:.3f}"
                else:
                    text = f"{v['value']:.2f}"
                if interval:
                    ax.errorbar([xi], [height], yerr=_errors([height], [interval]), fmt="none", ecolor=INK,
                                capsize=2, linewidth=0.8)
                top = interval[1] if interval else height
                ax.text(xi, top + 0.02, text, ha="center", fontsize=6.5, color=INK)
                if v["reference"] is not None and i == 0:
                    ax.plot([xi - 0.18, xi + 0.56], [v["reference"]] * 2, color=INK_2, linestyle="--", linewidth=1)
        ax.set_xticks(x, [m.replace(" ", "\n", 1) for m in measures], fontsize=7.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(panel, fontsize=9)
        _grid(ax)
    axes[0].set_ylabel("value (each panel has its own metric)")
    axes[0].plot([], [], color=INK_2, linestyle="--", label="majority class")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Probed vs stated vs behaviour (sandbox). Error bars: 95% bootstrap CI where saved; "
                 "RQ3 caution: spec definition")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path = out / "fig2_probed_stated_behaviour.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------- 3. blindness gap ----------

def gap_records(src):
    records = []
    for model in MODELS:
        source = f"results/analysis/{model}/headline_h3_plus_extra.json"
        head = src.json(source)
        for condition, short in CONDITIONS.items():
            combo = head["headline_merged"]["combinations"][f"{condition}/{HEADLINE_CAUTION}"]
            for metric in ("K", "A", "Gap"):
                field = f"headline_merged.combinations.{condition}/{HEADLINE_CAUTION}"
                value = src.put(f"{model}/gap60/{short}/{metric}", combo[metric], source, f"{field}.{metric}")
                interval = src.put(f"{model}/gap60/{short}/{metric}_ci95", combo[f"{metric}_ci95"], source,
                                   f"{field}.{metric}_ci95")
                n = src.put(f"{model}/gap60/{short}/n_pairs", combo["n_pairs"], source, f"{field}.n_pairs")
                degenerate = rule_of_three(value, interval, n)
                if degenerate:
                    src.put(f"{model}/gap60/{short}/{metric}_rule_of_three_upper", degenerate["upper_bound"], source,
                            f"3 / {field}.n_pairs", "derived: rule of three, CI is all zeros")
                records.append({"model": MODELS[model], "condition": short, "metric": metric, "value": value,
                                "ci_low": interval[0], "ci_high": interval[1], "n_pairs": n,
                                "degenerate": degenerate["text"] if degenerate else "",
                                "rule_of_three_upper": degenerate["upper_bound"] if degenerate else None,
                                "excluded_pairs": len(combo["excluded_pairs"]), "source": source})
    return records


def figure_gap(records, out):
    plt = _plt()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 4.8), sharey=True)
    metrics = ("K", "A", "Gap")
    for ax, model in zip(axes, MODELS.values()):
        for i, condition in enumerate(CONDITIONS.values()):
            rows = [next(r for r in records if r["model"] == model and r["condition"] == condition and r["metric"] == m)
                    for m in metrics]
            x = np.arange(len(metrics)) + (i - 0.5) * 0.38
            heights = [r["value"] for r in rows]
            intervals = [[r["ci_low"], r["rule_of_three_upper"]] if r["degenerate"] else [r["ci_low"], r["ci_high"]]
                         for r in rows]
            ax.bar(x, heights, 0.36, color=MODEL_COLORS[i], edgecolor="white", linewidth=1)
            ax.errorbar(x, heights, yerr=_errors(heights, intervals), fmt="none", ecolor=INK, capsize=2, linewidth=0.8)
            for xi, r, interval in zip(x, rows, intervals):
                text = (r["degenerate"].replace(" pairs", "\npairs") + f"\n≤ {r['rule_of_three_upper']:.3f}"
                        if r["degenerate"] else f"{r['value']:.3f}\n[{r['ci_low']:.3f},\n{r['ci_high']:.3f}]")
                ax.text(xi, max(interval[1], r["value"]) + 0.004, text, ha="center", va="bottom", fontsize=6.5,
                        color=INK)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_xticks(np.arange(len(metrics)), metrics)
        _grid(ax)
    axes[0].set_ylabel("fraction of pairs")
    top = max(max(r["ci_high"], r["rule_of_three_upper"] or 0) for r in records)
    axes[0].set_ylim(min(0, min(r["ci_low"] for r in records)) - 0.01, top + 0.045)
    for ax, model in zip(axes, MODELS.values()):
        n = {c: next(r["n_pairs"] for r in records if r["model"] == model and r["condition"] == c)
             for c in CONDITIONS.values()}
        ax.set_title(f"{model}: complete pairs n = {n['neutral']} (neutral), {n['careful']} (careful)", fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[i]) for i in range(len(CONDITIONS))]
    fig.legend(handles, [f"{c} prompt" for c in CONDITIONS.values()], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Blindness gap K, A, Gap = K - A on the merged 60-pair RQ3 batch (spec caution definition)\n"
                 "Error bars: 95% paired bootstrap CI; an all-zero CI is shown as 0 of n pairs with the "
                 "rule-of-three upper bound 3/n", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    path = out / "fig3_blindness_gap_60_pairs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------- 4. ablations ----------

ABLATION_ORDER = ("token position", "label granularity", "rows", "prompt (RQ3, 60 pairs)",
                  "cross-environment transfer", "ScienceWorld horizon", "leave-one-tool-out",
                  "full-prompt text baseline")


def _row(ablation, setting, model, metric, value, interval, n, source, post_hoc=False, note=""):
    return {"ablation": ablation, "setting": setting, "model": MODELS[model], "metric": metric, "value": value,
            "ci_low": interval[0] if interval else None, "ci_high": interval[1] if interval else None, "n": n,
            "post_hoc": post_hoc, "note": note, "source": source}


def position_ablation(src, model, resamples):
    """Train-selected layer within one token position (same rule as the main analysis), from saved probe files."""
    out_dir = src.path(f"results/analysis/{model}")
    summary = src.csv(f"results/analysis/{model}/probes_summary.csv")
    layers = sorted({int(r["layer"]) for r in summary if r["layer"].isdigit()})
    rows = []
    for target in TARGETS:
        loaded = load_probe_files(out_dir, "all", target, layers)
        for position in POSITIONS:
            within = {k: v for k, v in loaded.items() if k[0] == position}
            choice = select_per_split(within)
            per_split = [within[choice[i]]["splits"][i] for i in range(3)]
            row = _summary_row({}, [s["scores"]["probe"] for s in per_split],
                               [(s["gold"], s["predictions"]["probe"], s["test_groups"]) for s in per_split],
                               target, resamples)
            value, interval = row["balanced_accuracy_mean"], json.loads(row["balanced_accuracy_ci95"])
            source = f"results/analysis/{model}/probes/all__{target}__{position}__L*.json"
            src.put(f"{model}/ablation/position/{target}/{position}/balanced_accuracy", value, source,
                    f"train-selected within position, layers {[c[1] for c in choice]}",
                    "derived summary of saved probe files (not in probes_summary.csv)")
            rows.append(_row("token position", f"{position} token, {target}, all rows", model, "balanced accuracy",
                             value, interval, sum(s["n_test"] for s in per_split), source,
                             note=f"train-selected layers {[c[1] for c in choice]}"))
    return rows


def ablation_rows(src, resamples):
    rows = []
    for model in MODELS:
        base = f"results/analysis/{model}"
        summary = src.csv(f"{base}/probes_summary.csv")
        rows += position_ablation(src, model, resamples)
        for subset, name in (("all", "all rows"), ("minimal", "minimal pairs only")):
            for target in TARGETS:
                chosen, _ = selected_rows(summary, subset, target)
                probe = chosen["probe"]
                value = src.put(f"{model}/sandbox/{subset}/{target}/probe_train_selected/balanced_accuracy",
                                num(probe["balanced_accuracy_mean"]), f"{base}/probes_summary.csv",
                                f"control=probe, subset={subset}, target={target}, layer={probe['layer']}")
                n = sum(json.loads(probe["n_test"]))
                for ablation in ("rows", "label granularity"):
                    if ablation == "label granularity" and subset != "all":
                        continue
                    setting = f"{name}, {target}" if ablation == "rows" else target
                    rows.append(_row(ablation, setting, model, "balanced accuracy", value,
                                     ci(probe["balanced_accuracy_ci95"]), n, f"{base}/probes_summary.csv"))
        head = src.json(f"{base}/headline_h3_plus_extra.json")
        for condition, short in CONDITIONS.items():
            caution = head["rq3_merged"][condition]["caution"][HEADLINE_CAUTION]
            combo = head["headline_merged"]["combinations"][f"{condition}/{HEADLINE_CAUTION}"]
            source = f"{base}/headline_h3_plus_extra.json"
            n = caution["n_complete_reversible_irreversible_pairs"]
            for metric, key in (("caution rate, irreversible", "irreversible_caution_rate"),
                                ("caution rate, reversible", "reversible_over_caution_rate")):
                value = src.put(f"{model}/sandbox/rq3_merged/{short}/{key}", caution[key], source,
                                f"rq3_merged.{condition}.caution.{HEADLINE_CAUTION}.{key}")
                rows.append(_row("prompt (RQ3, 60 pairs)", f"{short} prompt", model, metric, value, None, n, source))
            rows.append(_row("prompt (RQ3, 60 pairs)", f"{short} prompt", model, "A", combo["A"], combo["A_ci95"],
                             combo["n_pairs"], source))
        transfer_source = f"{base}/sw/sw_transfer.json"
        transfer = src.json(transfer_source)["directions"]
        for direction, settings in transfer.items():
            for target in TARGETS:
                for position in POSITIONS:
                    entry = settings[target].get(position)
                    if not entry:
                        continue
                    for control in ("probe", "shuffled_label", "majority"):
                        result = entry[control]
                        name = f"{model}/transfer/{direction}/{target}/{position}/{control}/balanced_accuracy"
                        src.put(name, result["balanced_accuracy"], transfer_source,
                                f"directions.{direction}.{target}.{position}.{control}.balanced_accuracy")
                        rows.append(_row("cross-environment transfer", f"{direction}, {target}, {position} token, "
                                         f"{control.replace('_', ' ')}", model, "balanced accuracy",
                                         result["balanced_accuracy"], result["balanced_accuracy_ci95"],
                                         sum(settings[target]["target_class_counts"].values()),
                                         transfer_source, note=f"layer {entry['layer']}, C {entry['C']}"))
        loto_source = f"{base}/posthoc_leave_one_tool_out.json"
        loto = src.json(loto_source)["targets"]
        for target in TARGETS:
            for key, setting in (("mean_probe_accuracy_over_tools", "probe, mean over all tools"),
                                 ("mean_train_majority_accuracy_over_tools", "train majority, mean over all tools"),
                                 ("mean_probe_accuracy_over_mixed_label_tools", "probe, mean over mixed-label tools"),
                                 ("mean_train_majority_accuracy_over_mixed_label_tools",
                                  "train majority, mean over mixed-label tools")):
                value = src.put(f"{model}/posthoc/leave_one_tool_out/{target}/{key}", loto[target][key], loto_source,
                                f"targets.{target}.{key}", "post-hoc")
                rows.append(_row("leave-one-tool-out", f"{target}, {setting}", model, "accuracy", value, None,
                                 loto[target]["n_tools"], loto_source, post_hoc=True,
                                 note=f"{loto[target]['position']} token, layer {loto[target]['layer']}"))
        full_source = f"{base}/posthoc_fulltext_baselines.json"
        full = src.json(full_source)["settings"]
        for setting_key, entry in full.items():
            subset, target = setting_key.split("/")
            for name in ("fulltext_bow", "fulltext_tfidf"):
                value = src.put(f"{model}/posthoc/{name}/{setting_key}/balanced_accuracy",
                                entry[name]["balanced_accuracy_mean"], full_source,
                                f"settings.{setting_key}.{name}.balanced_accuracy_mean", "post-hoc")
                rows.append(_row("full-prompt text baseline", f"{name.split('_')[1]}, {subset} rows, {target}", model,
                                 "balanced accuracy", value, None, sum(entry[name]["n_test"]), full_source,
                                 post_hoc=True, note="no CI saved for balanced accuracy"))
        horizon_source = f"{base}/sw/sw_horizon.json"
        horizons = src.json(horizon_source)["horizons"]
        for horizon, by_target in sorted(horizons.items()):
            for target in TARGETS:
                entry = by_target[target]
                if "not_estimable" in entry:
                    rows.append(_row("ScienceWorld horizon", f"h = {horizon}, {target}", model, "balanced accuracy",
                                     None, None, entry["n_rows"], horizon_source, note=entry["not_estimable"]))
                    continue
                for control in ("probe", "majority"):
                    value = src.put(f"{model}/scienceworld/horizon{horizon}/{target}/{control}/balanced_accuracy",
                                    entry[control]["balanced_accuracy_mean"], horizon_source,
                                    f"horizons.{horizon}.{target}.{control}.balanced_accuracy_mean")
                    rows.append(_row("ScienceWorld horizon", f"h = {horizon}, {target}, {control}", model,
                                     "balanced accuracy", value, entry[control]["balanced_accuracy_ci95"],
                                     entry["n_rows"], horizon_source))
    return sorted(rows, key=lambda r: ABLATION_ORDER.index(r["ablation"]))


def _cell(row):
    if row is None or row["value"] is None:
        return "--"
    degenerate = rule_of_three(row["value"], [row["ci_low"], row["ci_high"]], row["n"] if isinstance(row["n"], int)
                               else None)
    if degenerate:
        return f"{degenerate['text']} ($\\le$ {degenerate['upper_bound']:.3f})"
    text = f"{row['value']:.3f}"
    if row["ci_low"] is not None:
        text += f" [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
    return text


def latex_table(rows):
    def esc(text):
        return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace(
            "%", r"\%").replace("->", r"$\rightarrow$")
    keys = list(dict.fromkeys((r["ablation"], r["setting"], r["metric"], r["post_hoc"]) for r in rows))
    by = {(r["ablation"], r["setting"], r["metric"], r["model"]): r for r in rows}
    lines = [r"\begin{tabular}{lllll}", r"\hline",
             "Ablation & Setting & Metric & " + " & ".join(MODELS.values()) + r" \\", r"\hline"]
    previous = None
    for ablation, setting, metric, post_hoc in keys:
        label = esc(ablation) + (r"$^\dagger$" if post_hoc else "") if ablation != previous else ""
        if previous is not None and ablation != previous:
            lines.append(r"\hline")
        previous = ablation
        cells = [_cell(by.get((ablation, setting, metric, m))) for m in MODELS.values()]
        lines.append(f"{label} & {esc(setting)} & {esc(metric)} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\multicolumn{5}{l}{$^\dagger$ post-hoc, not in the original plan. "
              r"Brackets: 95\% bootstrap CI where saved.} \\", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------- 5. other headline numbers ----------

def other_numbers(src):
    for model in MODELS:
        base = f"results/analysis/{model}"
        summary = src.csv(f"{base}/probes_summary.csv")
        for target in TARGETS:
            for control in ("majority", "lexical_bow"):
                b = baseline(summary, "all", target, control)
                for column in ("accuracy", "balanced_accuracy"):
                    src.put(f"{model}/sandbox/{target}/{control}/{column}", num(b[f"{column}_mean"]),
                            f"{base}/probes_summary.csv", f"control={control}, subset=all, target={target}")
            chosen, _ = selected_rows(summary, "all", target)
            src.put(f"{model}/sandbox/{target}/probe_train_selected/accuracy_ci95", ci(chosen["probe"]["accuracy_ci95"]),
                    f"{base}/probes_summary.csv", f"control=probe, subset=all, target={target}, train_selected")
            src.put(f"{model}/sandbox/{target}/probe_train_selected/layers", chosen["probe"]["layer"],
                    f"{base}/probes_summary.csv", "layer")
            sw_summary = src.csv(f"{base}/sw/probes_summary.csv")
            sw_chosen, _ = selected_rows(sw_summary, "all", target)
            for column in ("accuracy", "auroc"):
                src.put(f"{model}/scienceworld/{target}/probe_train_selected/{column}",
                        num(sw_chosen["probe"][f"{column}_mean"]), f"{base}/sw/probes_summary.csv",
                        f"control=probe, subset=all, target={target}, layer={sw_chosen['probe']['layer']}")
            for control in ("majority", "lexical_bow"):
                b = baseline(sw_summary, "all", target, control)
                src.put(f"{model}/scienceworld/{target}/{control}/balanced_accuracy", num(b["balanced_accuracy_mean"]),
                        f"{base}/sw/probes_summary.csv", f"control={control}, subset=all, target={target}")
        first = src.json(f"{base}/headline.json")["combinations"][f"neutral/{HEADLINE_CAUTION}"]
        for metric in ("K", "A", "Gap", "n_pairs"):
            src.put(f"{model}/gap30_first_batch/neutral/{metric}", first[metric], f"{base}/headline.json",
                    f"combinations.neutral/{HEADLINE_CAUTION}.{metric}")
        directional = src.json(f"{base}/posthoc_rq2_directional.json")["all_rq2_pairs"]
        for key in ("directional_score", "directional_score_ci95", "tied_fraction"):
            src.put(f"{model}/posthoc/rq2_directional/{key}", directional[key], f"{base}/posthoc_rq2_directional.json",
                    f"all_rq2_pairs.{key}", "post-hoc")
        sw_rq2_source = f"results/{model}/sw/rq2/three-class-h3-neutral-v2/summary.json"
        if src.path(sw_rq2_source).exists():
            sw_rq2 = src.json(sw_rq2_source)
            for key in ("stated_answer_accuracy", "K_correct_stated_distinction", "n_reversible_irreversible_pairs",
                        "n_scored", "new_parse_or_generation_failures"):
                src.put(f"{model}/scienceworld/rq2/{key}", sw_rq2.get(key), sw_rq2_source, key)
    rows_source = "results/analysis/Qwen_Qwen2.5-7B-Instruct/sw/sw_rows.json"
    sw_rows = src.json(rows_source)
    for key in ("input_rows", "duplicates_removed", "rows_after_dedup", "groups", "class_counts"):
        src.put(f"scienceworld/{key}", sw_rows[key], rows_source, key)
    agreement_source = "results/analysis/human_agreement.json"
    if src.path(agreement_source).exists():
        agreement = src.json(agreement_source)
        src.put("human/raters", agreement["raters"], agreement_source, "raters")
        src.put("human/retest_missing", agreement.get("retest_missing"), agreement_source, "retest_missing")
        for name, scopes in agreement["comparisons"].items():
            for scope, result in scopes.items():
                if not result.get("n_items"):
                    continue
                for key in ("kappa", "kappa_ci95", "observed_agreement", "n_items"):
                    src.put(f"human/{name}/{scope}/{key}", result[key], agreement_source,
                            f"comparisons.{name}.{scope}.{key}")
                restricted = result.get("post_hoc_reversible_vs_irreversible")
                if restricted and restricted.get("n_items"):
                    for key in ("kappa", "kappa_ci95", "percent_agreement", "n_items"):
                        src.put(f"human/{name}/{scope}/reversible_vs_irreversible/{key}", restricted[key],
                                agreement_source, f"comparisons.{name}.{scope}.post_hoc_reversible_vs_irreversible.{key}",
                                "post-hoc")


# ---------- consistency checks ----------

def consistency_checks(src):
    """Cross-file agreements a reader would expect; a failure is reported, never corrected."""
    checks = []
    for model in MODELS:
        base = f"results/analysis/{model}"
        head = src.json(f"{base}/headline_h3_plus_extra.json")
        for condition in CONDITIONS:
            combo = head["headline_merged"]["combinations"][f"{condition}/{HEADLINE_CAUTION}"]
            caution = head["rq3_merged"][condition]["caution"][HEADLINE_CAUTION]
            checks.append({"check": f"{MODELS[model]} {condition}: A in rq3_merged caution equals A in headline_merged",
                           "ok": abs(caution["A_paired_reversibility_sensitivity"] - combo["A"]) < 1e-9
                           or caution["n_complete_reversible_irreversible_pairs"] != combo["n_pairs"],
                           "values": [caution["A_paired_reversibility_sensitivity"], combo["A"]],
                           "n": [caution["n_complete_reversible_irreversible_pairs"], combo["n_pairs"]]})
        rq2 = src.json(f"{base}/rq2_summary.json")
        k_all = head["headline_merged"]["K_all_rq2_pairs"]["K"]
        checks.append({"check": f"{MODELS[model]}: K in rq2_summary equals K_all_rq2_pairs in the headline",
                       "ok": abs(rq2["K_correct_stated_distinction"] - k_all) < 1e-9,
                       "values": [rq2["K_correct_stated_distinction"], k_all]})
        for environment, path in (("sandbox", f"{base}/probes_summary.csv"), ("scienceworld", f"{base}/sw/probes_summary.csv")):
            for target in TARGETS:
                majority = baseline(src.csv(path), "all", target, "majority")
                interval = ci(majority["balanced_accuracy_ci95"])
                value = num(majority["balanced_accuracy_mean"])
                checks.append({"check": f"{MODELS[model]} {environment} {target}: majority balanced-accuracy CI is "
                               "the constant chance level", "ok": abs(interval[1] - interval[0]) < 1e-9,
                               "values": [value, interval]})
    return checks


# ---------- main ----------

def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--out-dir")
    parser.add_argument("--docs-dir")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    src = Sources(args.root)
    out = Path(args.out_dir) if args.out_dir else src.path("results/analysis/final")
    docs = Path(args.docs_dir) if args.docs_dir else src.path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    produced = [figure_layers(src, out, target) for target in TARGETS]
    records = stated_probed_behaviour(src)
    write_csv(out / "table_probed_stated_behaviour.csv", records)
    produced += [out / "table_probed_stated_behaviour.csv", figure_stated_probed_behaviour(records, out)]
    gaps = gap_records(src)
    write_csv(out / "table_blindness_gap_60_pairs.csv", gaps)
    produced += [out / "table_blindness_gap_60_pairs.csv", figure_gap(gaps, out)]
    ablations = ablation_rows(src, args.bootstrap_resamples)
    write_csv(out / "table_ablations.csv", ablations)
    (out / "table_ablations.tex").write_text(latex_table(ablations))
    produced += [out / "table_ablations.csv", out / "table_ablations.tex"]
    other_numbers(src)
    checks = consistency_checks(src)
    (out / "headline_numbers.json").write_text(json.dumps(
        {"note": "every number the report quotes, with its source file (paths relative to the repo root)",
         "numbers": dict(sorted(src.numbers.items())), "consistency_checks": checks},
        indent=2, default=float) + "\n")
    for path in produced:
        shutil.copy2(path, docs / path.name)
    for check in checks:
        if not check["ok"]:
            print(f"CHECK FAILED: {check['check']}: {check['values']}")
    print(f"wrote {len(produced)} files to {out} and {docs}; {len(src.numbers)} numbers in headline_numbers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
