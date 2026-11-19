import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import run_full  # noqa: E402

from experiments.extract_hidden import extract_one  # noqa: E402


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_extraction_places_inputs_on_embedding_device_and_times_rows(tmp_path):
    import torch

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "ctx: open door"

        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "offset_mapping": torch.tensor([[(0, 4), (5, 9), (10, 14)]]),
            }

    class FakeModel:
        def get_input_embeddings(self):
            return type("Embedding", (), {"weight": torch.zeros(1)})()

        def __call__(self, input_ids, attention_mask, **kwargs):
            self.device = input_ids.device
            hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4)
            return type("Output", (), {"hidden_states": tuple(hidden)})()

    model = FakeModel()
    row = {"id": "r1", "action_text": "open door", "scenario_id": "s1"}
    _, index = extract_one(model, FakeTokenizer(), row, tmp_path, max_tokens=10)
    assert model.device == torch.device("cpu")
    assert index["action_token_span"] == [1, 3]
    assert index["extract_seconds"] >= 0
    array = np.load(tmp_path / index["array_path"])
    assert array.shape == (2, 2, 4)


def test_pilot_projection_scales_timing_and_size(tmp_path):
    paths = run_full.Paths(tmp_path, "org/Model-7B")
    arrays = paths.extraction / "arrays"
    arrays.mkdir(parents=True)
    index = []
    for i, seconds in enumerate((1.0, 2.0, 30.0)):
        np.save(arrays / f"{i}.npy", np.zeros(1000, dtype=np.float16))
        index.append({"id": str(i), "array_path": f"arrays/{i}.npy", "extract_seconds": seconds})
    _write_jsonl(paths.extraction / "index.jsonl", index)
    _write_jsonl(paths.extraction / "prompts.jsonl", [{"id": str(i)} for i in range(3)])
    projection = run_full.pilot_projection(paths, total_rows=103)
    assert projection["sec_per_row_median"] == 2.0
    assert projection["remaining_rows"] == 100
    assert abs(projection["projected_remaining_extraction_hours"] - 200 / 3600) < 1e-9
    per_row = (np.zeros(1000, dtype=np.float16).nbytes + 128) / 1e9
    assert projection["projected_total_extraction_gb"] > 103 * per_row * 0.9


def test_rq2_and_rq3_done_checks_require_every_row(tmp_path):
    paths = run_full.Paths(tmp_path, "m")
    _write_jsonl(paths.rq2 / "responses.jsonl", [{"id": "a"}])
    (paths.rq2 / "summary.json").write_text('{"prompt_version": "rq2-stated-v2"}')
    assert not run_full.rq2_done(paths, {"a", "b"})
    _write_jsonl(paths.rq2 / "responses.jsonl", [{"id": "a"}, {"id": "b"}])
    assert run_full.rq2_done(paths, {"a", "b"})
    (paths.rq2 / "summary.json").write_text('{"prompt_version": "rq2-stated-v1"}')
    assert not run_full.rq2_done(paths, {"a", "b"})
    assert paths.rq2.name == "three-class-h3-neutral-v2"

    _write_jsonl(paths.rq3 / "tasks.jsonl", [{"id": "x"}, {"id": "y"}])
    (paths.rq3 / "summary.json").write_text('{"max_turns": 8}')
    outcomes = [{"id": i, "condition": "neutral"} for i in ("x", "y")]
    _write_jsonl(paths.rq3 / "responses.jsonl", outcomes + [{"id": "x", "condition": "explicit-carefulness", "turn": 0}])
    assert not run_full.rq3_done(paths, 2)
    outcomes += [{"id": i, "condition": "explicit-carefulness"} for i in ("x", "y")]
    _write_jsonl(paths.rq3 / "responses.jsonl", outcomes)
    assert not run_full.rq3_done(paths, 2)
    (paths.rq3 / "summary.json").write_text('{"max_turns": 15}')
    assert run_full.rq3_done(paths, 2)
    assert not run_full.rq3_done(paths, 60)


def test_zip_contains_model_results_under_model_folder(tmp_path):
    paths = run_full.Paths(tmp_path / "results", "Qwen/Qwen2.5-7B-Instruct")
    _write_jsonl(paths.rq2 / "summary.json", [{}])
    target = run_full.zip_results(paths, tmp_path / "out")
    assert target.name == "Qwen_Qwen2.5-7B-Instruct_results.zip"
    assert zipfile.ZipFile(target).namelist() == [
        "Qwen_Qwen2.5-7B-Instruct/rq2/three-class-h3-neutral-v2/summary.json"
    ]


def test_only_runs_one_step_then_zips(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_full, "run_module", lambda paths, step, module, args: calls.append(step) or 0)
    monkeypatch.setattr(run_full, "rq3_done", lambda paths, rows: bool(calls))
    _write_jsonl(tmp_path / "in.jsonl", [{"id": "a", "labels_by_horizon": {"3": "reversible"}}])
    code = run_full.main([
        "--model", "m", "--input", str(tmp_path / "in.jsonl"), "--results-root", str(tmp_path / "r"),
        "--zip-dir", str(tmp_path / "z"), "--only", "rq3",
    ])
    assert code == 0 and calls == ["rq3"]
    assert (tmp_path / "z" / "m_results.zip").exists()
    assert "steps: rq3, zip" in capsys.readouterr().out


def test_stop_after_and_only_accept_every_step_and_exclude_each_other():
    import pytest

    for step in run_full.STEPS:
        assert step in ("pilot", "disk", "extract", "rq2", "rq3", "zip")
    with pytest.raises(SystemExit):
        run_full.main(["--model", "m", "--only", "rq3", "--stop-after", "rq2"])
