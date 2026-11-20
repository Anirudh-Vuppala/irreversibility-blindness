import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.analyze_sw import dedup_rows, main as analyze_sw_main
from experiments.extract_hidden import make_messages
from experiments.model_runtime import REPO_ROOT, read_jsonl
from experiments.rq2_stated import (
    SW_PROMPT_VERSION, candidate_pairs, parse_stated_label, row_prompt_version, score_answers, stated_messages,
)

sys.path.insert(0, str(REPO_ROOT / "kaggle"))
import run_full  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "scienceworld_rows.jsonl"


def _fixture():
    return read_jsonl(FIXTURE)[0]


def _saved(pattern):
    return sorted(REPO_ROOT.glob(pattern))


def test_scienceworld_prompt_has_no_sandbox_wording():
    row = _fixture()[0]
    system, user = (m["content"] for m in stated_messages(row))
    text = (system + user).casefold()
    assert "science-lab simulator" in text and "text commands" in text
    assert "could this action be undone within the next 3 actions?" in text
    for word in ("tool", "email", "delete", "workspace", "transfer"):
        assert word not in text
    assert "reversible, partially_reversible, irreversible" in user
    assert row_prompt_version(row, "rq2-stated-v2") == SW_PROMPT_VERSION
    assert parse_stated_label('{"label": "irreversible", "evidence": "x"}', "three-class")[0] == "irreversible"
    with pytest.raises(ValueError):
        row_prompt_version(row, "rq2-stated-v1")


@pytest.mark.parametrize("prompts", _saved("results/*/rq2/three-class-h3-neutral-v2/prompts.jsonl"))
def test_sandbox_rq2_prompts_are_byte_identical_to_saved_runs(prompts):
    rows = {r["id"]: r for r in read_jsonl(REPO_ROOT / "data" / "sandbox.jsonl")[0]}
    saved = read_jsonl(prompts)[0]
    assert saved
    for record in saved:
        regenerated = stated_messages(rows[record["id"]], "three-class", "neutral", 3, "rq2-stated-v2")
        dump = lambda messages: json.dumps(messages, ensure_ascii=False, sort_keys=True)
        assert dump(regenerated) == dump(record["messages"])
        assert record["prompt_version"] == "rq2-stated-v2"


@pytest.mark.parametrize("prompts", _saved("results/*/extraction/prompts.jsonl"))
def test_sandbox_extraction_prompts_are_unchanged(prompts):
    rows = {r["id"]: r for r in read_jsonl(REPO_ROOT / "data" / "sandbox.jsonl")[0]}
    for record in read_jsonl(prompts)[0]:
        assert make_messages(rows[record["id"]]) == record["messages"]


def test_scienceworld_extraction_wrapper():
    row = _fixture()[0]
    system, user, assistant = make_messages(row)
    assert "science-lab simulator" in system["content"] and "Current observation:" in user["content"]
    assert assistant == {"role": "assistant", "content": "activate stove"}


def test_scienceworld_rows_pair_by_pair_id_and_k_is_unchanged():
    rows = _fixture()
    pairs = {c["pair_id"]: {r["id"] for r in c["rows"]} for c in candidate_pairs(rows, 3, "three-class")}
    assert pairs == {"sw-pair-0": {"sw-fixture-0", "sw-fixture-1"}, "sw-pair-1": {"sw-fixture-2", "sw-fixture-3"}}
    answers = {"sw-fixture-0": "reversible", "sw-fixture-1": "irreversible",
               "sw-fixture-2": "reversible", "sw-fixture-3": "reversible", "sw-fixture-4": "reversible"}
    scored = score_answers(rows[:5], answers, {}, 3, "three-class")
    assert scored["n_reversible_irreversible_pairs"] == 2 and scored["K_correct_stated_distinction"] == 0.5


def test_dedup_removes_exact_duplicates_only():
    kept, removed = dedup_rows(_fixture())
    assert [r["id"] for r in removed] == ["sw-fixture-5"] and len(kept) == 5


def test_runner_scienceworld_runs_extraction_and_rq2_only(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_full, "run_module", lambda paths, step, module, args: calls.append((step, args)) or 0)
    code = run_full.main(["--model", "m", "--env", "scienceworld", "--input", str(FIXTURE),
                          "--results-root", str(tmp_path / "r"), "--zip-dir", str(tmp_path / "z")])
    out = capsys.readouterr().out
    assert "steps: extract, rq2, zip" in out and "RQ3" not in out
    assert [step for step, _ in calls] == ["extract", "rq2"]
    extract_args, rq2_args = calls[0][1], calls[1][1]
    assert extract_args[extract_args.index("--output-dir") + 1] == str((tmp_path / "r" / "m" / "sw" / "extraction").resolve())
    assert "/m/sw/rq2/" in rq2_args[rq2_args.index("--output-dir") + 1] and rq2_args[rq2_args.index("--limit") + 1] == "4"
    assert (tmp_path / "z" / "m_sw_results.zip").exists() and code == 1
    with pytest.raises(SystemExit):
        run_full.main(["--model", "m", "--env", "scienceworld", "--only", "rq3", "--input", str(FIXTURE)])


def test_runner_scienceworld_without_pairs_skips_rq2(tmp_path, monkeypatch, capsys):
    unpaired = tmp_path / "sw.jsonl"
    unpaired.write_text("".join(json.dumps({**r, "pair_id": None}) + "\n" for r in _fixture()))
    monkeypatch.setattr(run_full, "run_module", lambda paths, step, module, args: 0)
    assert run_full.main(["--model", "m", "--env", "scienceworld", "--input", str(unpaired), "--only", "rq2",
                          "--results-root", str(tmp_path / "r"), "--zip-dir", str(tmp_path / "z")]) == 1
    assert "rq2: no eligible reversible/irreversible pairs" in capsys.readouterr().out


def _fake_extraction(directory, environment, n_groups, rows_per_group, group_key, rng):
    labels = ("reversible", "partially_reversible", "irreversible")
    index, data = [], []
    for g in range(n_groups):
        for k in range(rows_per_group):
            label = labels[(g + k) % 3]
            row_id = f"{environment}-{g}-{k}"
            array = rng.normal(size=(3, 2, 6)).astype(np.float16)
            array[1:, :, 0] += 3 * labels.index(label)
            path = directory / "extraction" / "arrays" / f"{row_id}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array)
            meta = {"id": row_id, "environment": environment, "label": label, "group_id": f"{environment}-g{g}",
                    "scenario_id": f"{environment}-g{g}", "labels_by_horizon": {"1": "irreversible" if k else label,
                    "2": label, "3": label}, "is_minimal_pair": True, "action_text": f"act{k} obj{g}"}
            index.append({**meta, "array_path": f"arrays/{row_id}.npy", "shape": [3, 2, 6]})
            data.append({**meta, "state_text": f"state {g}", "horizon": 3, "source": "test"})
    data.append({**data[0], "id": f"{environment}-duplicate"})
    (directory / "extraction" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    return data


def test_analyze_sw_end_to_end_on_synthetic_arrays(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    model_dir = tmp_path / "results" / "M"
    _fake_extraction(model_dir, "sandbox", 10, 6, "scenario_id", rng)
    sw = _fake_extraction(model_dir / "sw", "scienceworld", 10, 6, "group_id", rng)
    (tmp_path / "sw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sw))
    out = tmp_path / "out"
    assert analyze_sw_main(["--model-dir", str(model_dir), "--data", str(tmp_path / "sw.jsonl"), "--out-dir", str(out),
                            "--jobs", "1", "--layer-stride", "1", "--bootstrap-resamples", "20"]) == 0
    assert json.loads((out / "sw_rows.json").read_text())["duplicates_removed"] == 1
    transfer = json.loads((out / "sw_transfer.json").read_text())["directions"]
    probe = transfer["sandbox->scienceworld"]["three-class"]["mean"]["probe"]
    assert probe["accuracy"] >= 0.7 and transfer["scienceworld->sandbox"]["binary"]["last"]["layer"] in (1, 2)
    horizon = json.loads((out / "sw_horizon.json").read_text())["horizons"]
    h3 = horizon["3"]["three-class"]
    assert set(horizon) == {"1", "2", "3"} and h3["probe"]["accuracy_mean"] > h3["majority"]["accuracy_mean"] + 0.2
    assert (out / "probes_summary.csv").exists() and (out / "figures" / "probe_accuracy_vs_layer.png").exists()
