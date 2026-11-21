import csv
import json
import random

import pytest
from sklearn.metrics import cohen_kappa_score

from experiments.label_agreement import agreement, kappa, main as agreement_main, normalize
from experiments.label_sample import KEY_COLUMNS, SHEET_COLUMNS, SOURCES, build_items, main as sample_main
from experiments.rq2_stated import LABELS


def _write(path, columns, records):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def _synthetic(tmp_path, n=40, rater2=True, noise=0.2, seed=0):
    rng = random.Random(seed)
    key = [{"item_id": f"item-{i:03d}", "environment": ("sandbox", "scienceworld")[i % 2], "row_id": f"r{i}",
            "pair_ref": f"p{i // 4}" if i < 8 else "", "auto_label": LABELS[i % 3]} for i in range(n)]
    _write(tmp_path / "key.csv", KEY_COLUMNS, key)

    def sheet(name, flip):
        records = [{"item_id": k["item_id"], "environment": k["environment"], "state_text": "s", "action_text": "a",
                    "label_rater": rng.choice(LABELS) if rng.random() < flip else k["auto_label"]} for k in key]
        _write(tmp_path / name, SHEET_COLUMNS, records)
        return records

    s1 = sheet("r1.csv", noise)
    s2 = sheet("r2.csv", noise) if rater2 else None
    return key, s1, s2


def test_kappa_matches_sklearn_and_handles_constant_labels():
    rng = random.Random(1)
    a = [rng.choice(LABELS) for _ in range(60)]
    b = [x if rng.random() < 0.7 else rng.choice(LABELS) for x in a]
    assert kappa(a, b) == pytest.approx(cohen_kappa_score(a, b, labels=list(LABELS)))
    assert kappa(["reversible"] * 3, ["reversible"] * 3) is None
    assert normalize(" Partially reversible ") == "partially_reversible" and normalize("I") == "irreversible"


def test_agreement_both_raters(tmp_path):
    key, s1, s2 = _synthetic(tmp_path)
    result = agreement(tmp_path / "r1.csv", tmp_path / "r2.csv", tmp_path / "key.csv", resamples=200)
    assert set(result["comparisons"]) == {"rater1_vs_rater2", "rater1_vs_auto", "rater2_vs_auto"}
    overall = result["comparisons"]["rater1_vs_auto"]["overall"]
    expected = cohen_kappa_score([k["auto_label"] for k in key], [r["label_rater"] for r in s1], labels=list(LABELS))
    assert overall["kappa"] == pytest.approx(expected, abs=1e-12)
    assert overall["kappa_ci95"][0] <= overall["kappa"] <= overall["kappa_ci95"][1]
    counts = overall["confusion_matrix"]["counts"]
    assert sum(map(sum, counts)) == 40 and sum(counts[i][i] for i in range(3)) == round(40 * overall["observed_agreement"])
    assert result["comparisons"]["rater1_vs_rater2"]["sandbox"]["n_items"] == 20


def test_missing_or_empty_second_rater_reports_rater1_vs_auto_only(tmp_path):
    _synthetic(tmp_path, rater2=False)
    result = agreement(tmp_path / "r1.csv", tmp_path / "absent.csv", tmp_path / "key.csv", resamples=50)
    assert result["rater2_missing"] and list(result["comparisons"]) == ["rater1_vs_auto"]
    _write(tmp_path / "empty.csv", SHEET_COLUMNS, [{**r, "label_rater": ""} for r in csv.DictReader(open(tmp_path / "r1.csv"))])
    out = tmp_path / "out.json"
    assert agreement_main(["--rater1", str(tmp_path / "r1.csv"), "--rater2", str(tmp_path / "empty.csv"), "--key",
                           str(tmp_path / "key.csv"), "--out", str(out), "--bootstrap-resamples", "50"]) == 0
    assert list(json.loads(out.read_text())["comparisons"]) == ["rater1_vs_auto"]


def test_partial_and_invalid_sheets(tmp_path):
    _synthetic(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "r1.csv")))
    rows[0]["label_rater"] = ""
    _write(tmp_path / "partial.csv", SHEET_COLUMNS, rows)
    result = agreement(tmp_path / "partial.csv", None, tmp_path / "key.csv", resamples=20)
    assert result["items_labelled"]["rater1"] == 39 and result["comparisons"]["rater1_vs_auto"]["overall"]["n_items"] == 39
    rows[0]["label_rater"] = "maybe"
    _write(tmp_path / "bad.csv", SHEET_COLUMNS, rows)
    with pytest.raises(SystemExit, match="item-000"):
        agreement(tmp_path / "bad.csv", None, tmp_path / "key.csv")


def test_sample_is_blind_balanced_paired_and_deterministic(tmp_path):
    assert sample_main(["--out-dir", str(tmp_path)]) == 0
    with pytest.raises(SystemExit):
        sample_main(["--out-dir", str(tmp_path)])
    for name in ("sheet_rater1.csv", "sheet_rater2.csv"):
        with open(tmp_path / name, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            sheet = list(reader)
        assert tuple(reader.fieldnames) == SHEET_COLUMNS and all(r["label_rater"] == "" for r in sheet)
    items = build_items(SOURCES, 50, 5, 2026)
    assert [i["item_id"] for i in items] == [r["item_id"] for r in sheet]
    for environment in SOURCES:
        chosen = [i for i in items if i["environment"] == environment]
        labels = [i["row"]["label"] for i in chosen]
        assert len(chosen) == 50 and all(15 <= labels.count(label) <= 18 for label in LABELS)
        pairs = {}
        for i in chosen:
            if i["pair_ref"]:
                pairs.setdefault(i["pair_ref"], []).append(i["row"])
        assert len(pairs) == 5
        for a, b in pairs.values():
            assert a["action_text"] == b["action_text"] and a["label"] != b["label"]
    assert len({(i["row"]["state_text"], i["row"]["action_text"]) for i in items}) == 100
    assert [i["row"]["id"] for i in build_items(SOURCES, 50, 5, 2026)] == [i["row"]["id"] for i in items]


def test_retest_sheet_is_blind_subset_in_new_order(tmp_path):
    assert sample_main(["--out-dir", str(tmp_path)]) == 0
    read = lambda name: list(csv.DictReader(open(tmp_path / name, newline="", encoding="utf-8")))
    full, retest = read("sheet_rater1.csv"), read("sheet_rater1_retest.csv")
    position = {r["item_id"]: i for i, r in enumerate(full)}
    assert len(retest) == 30 and len({r["item_id"] for r in retest}) == 30
    assert tuple(retest[0]) == SHEET_COLUMNS and all(r["label_rater"] == "" for r in retest)
    assert all({**r, "label_rater": ""} == full[position[r["item_id"]]] for r in retest)
    assert [position[r["item_id"]] for r in retest] != sorted(position[r["item_id"]] for r in retest)
    (tmp_path / "sheet_rater1_retest.csv").unlink()
    assert sample_main(["--out-dir", str(tmp_path), "--retest-only"]) == 0
    assert read("sheet_rater1_retest.csv") == retest
    with pytest.raises(SystemExit):
        sample_main(["--out-dir", str(tmp_path), "--retest-only"])


def test_intra_rater_kappa_on_retest_items_without_rater2(tmp_path):
    key, s1, _ = _synthetic(tmp_path, rater2=False)
    rng = random.Random(3)
    retest = [{**r, "label_rater": r["label_rater"] if rng.random() < 0.8 else rng.choice(LABELS)}
              for r in rng.sample(s1, 30)]
    _write(tmp_path / "retest.csv", SHEET_COLUMNS, retest)
    result = agreement(tmp_path / "r1.csv", None, tmp_path / "key.csv", resamples=200, retest=tmp_path / "retest.csv")
    assert list(result["comparisons"]) == ["rater1_vs_rater1_retest", "rater1_vs_auto"] and result["rater2_missing"]
    intra = result["comparisons"]["rater1_vs_rater1_retest"]["overall"]
    first = {r["item_id"]: r["label_rater"] for r in s1}
    expected = cohen_kappa_score([first[r["item_id"]] for r in retest], [r["label_rater"] for r in retest],
                                 labels=list(LABELS))
    assert intra["n_items"] == 30 and intra["kappa"] == pytest.approx(expected, abs=1e-12)
    assert intra["kappa_ci95"][0] <= intra["kappa"] <= intra["kappa_ci95"][1]
    _write(tmp_path / "foreign.csv", SHEET_COLUMNS, [{**retest[0], "item_id": "item-999"}])
    with pytest.raises(SystemExit):
        agreement(tmp_path / "r1.csv", None, tmp_path / "key.csv", retest=tmp_path / "foreign.csv")
    no_retest = agreement(tmp_path / "r1.csv", None, tmp_path / "key.csv", resamples=20, retest=tmp_path / "none.csv")
    assert no_retest["retest_missing"] and list(no_retest["comparisons"]) == ["rater1_vs_auto"]
