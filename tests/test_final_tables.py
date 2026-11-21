import hashlib
import json
import re

import pytest

from experiments.make_final_tables import MODELS, latex_table, main, rule_of_three
from experiments.model_runtime import REPO_ROOT


def test_rule_of_three_only_for_all_zero_intervals():
    assert rule_of_three(0.0, [0.0, 0.0], 56) == {"text": "0 of 56 pairs", "upper_bound": 3 / 56}
    assert rule_of_three(0.0, [0.0, 0.05], 56) is None
    assert rule_of_three(0.02, [0.0, 0.0], 56) is None
    assert rule_of_three(0.0, None, 56) is None


def test_latex_table_marks_post_hoc_escapes_and_shows_degenerate_cis():
    base = {"metric": "balanced_accuracy", "n": 40, "note": "", "source": "x"}
    rows = [
        {**base, "ablation": "prompt", "setting": "neutral_prompt", "model": "Qwen2.5-7B", "value": 0.0,
         "ci_low": 0.0, "ci_high": 0.0, "post_hoc": False},
        {**base, "ablation": "prompt", "setting": "neutral_prompt", "model": "Llama-3.1-8B", "value": 0.5,
         "ci_low": 0.4, "ci_high": 0.6, "post_hoc": False},
        {**base, "ablation": "leave-one-tool-out", "setting": "a->b", "model": "Qwen2.5-7B", "value": 0.25,
         "ci_low": None, "ci_high": None, "post_hoc": True},
    ]
    tex = latex_table(rows)
    assert tex.startswith(r"\begin{tabular}{lllll}") and tex.rstrip().endswith(r"\end{tabular}")
    assert r"neutral\_prompt" in tex and r"balanced\_accuracy" in tex and r"$\rightarrow$" in tex
    assert r"0 of 40 pairs ($\le$ 0.075)" in tex and "0.500 [0.400, 0.600]" in tex
    assert r"leave-one-tool-out$^\dagger$" in tex and r"prompt$^\dagger$" not in tex
    assert re.search(r"0\.250 & -- \\\\", tex)


def _fingerprint(root):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.skipif(not all((REPO_ROOT / "results" / "analysis" / m / "headline_h3_plus_extra.json").exists()
                            for m in MODELS), reason="needs the saved analysis results")
def test_final_tables_on_saved_results_change_nothing_and_cite_sources(tmp_path):
    inputs = [REPO_ROOT / "results" / "analysis" / m for m in MODELS]
    before = {str(d): _fingerprint(d) for d in inputs}
    assert main(["--out-dir", str(tmp_path / "final"), "--docs-dir", str(tmp_path / "docs")]) == 0
    assert {str(d): _fingerprint(d) for d in inputs} == before
    produced = {p.name for p in (tmp_path / "final").iterdir()}
    assert {"headline_numbers.json", "table_ablations.csv", "table_ablations.tex", "fig3_blindness_gap_60_pairs.png",
            "fig2_probed_stated_behaviour.png", "fig1_probe_balanced_accuracy_vs_layer_binary.png"} <= produced
    assert produced - {"headline_numbers.json"} == {p.name for p in (tmp_path / "docs").iterdir()}
    numbers = json.loads((tmp_path / "final" / "headline_numbers.json").read_text())["numbers"]
    for entry in numbers.values():
        source = entry["source"]
        assert list(REPO_ROOT.glob(source)) if "*" in source else (REPO_ROOT / source).exists()
    for model in MODELS:
        head = json.loads((REPO_ROOT / "results" / "analysis" / model / "headline_h3_plus_extra.json").read_text())
        combo = head["headline_merged"]["combinations"]["neutral/spec_check_ask_backup"]
        for metric in ("K", "A", "Gap", "n_pairs"):
            assert numbers[f"{model}/gap60/neutral/{metric}"]["value"] == combo[metric]


def test_requirements_pin_every_dependency_exactly():
    lines = [l.strip() for l in (REPO_ROOT / "requirements.txt").read_text().splitlines() if l.strip()]
    assert lines and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.]*", l) for l in lines)
    names = {l.split("==")[0].lower() for l in lines}
    assert {"numpy", "scikit-learn", "joblib", "threadpoolctl", "matplotlib", "scipy", "torch", "transformers"} <= names
