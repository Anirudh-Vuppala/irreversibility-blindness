import numpy as np
import pytest

from experiments import rq3_agent
from experiments.analyze import (
    ask_attempt_before_target,
    group_bootstrap_ci,
    mean_bootstrap,
    pair_a,
    pair_k,
    paired_bootstrap,
    rederive_episode,
)
from experiments.rq2_stated import score_answers


def _pair(group, rev_id, irr_id):
    return [
        {"id": rev_id, "group_id": group, "labels_by_horizon": {"3": "reversible"}},
        {"id": irr_id, "group_id": group, "labels_by_horizon": {"3": "irreversible"}},
    ]


@pytest.mark.parametrize("answers, expected", [
    ({"r": "reversible", "i": "irreversible"}, True),
    ({"r": "partially_reversible", "i": "irreversible"}, False),
    ({"r": "irreversible", "i": "irreversible"}, False),
    ({"r": "reversible"}, None),
])
def test_pair_k_matches_score_answers(answers, expected):
    rows = _pair("g", "r", "i")
    assert pair_k(rows, answers) is expected
    k = score_answers(rows, answers, {}, 3, "three-class")["K_correct_stated_distinction"]
    assert k == (None if expected is None else float(expected))


def test_pair_a_matches_rq3_caution_metrics():
    tasks = [
        {"id": "r1", "group_id": "g1", "label": "reversible"}, {"id": "i1", "group_id": "g1", "label": "irreversible"},
        {"id": "r2", "group_id": "g2", "label": "reversible"}, {"id": "i2", "group_id": "g2", "label": "irreversible"},
        {"id": "r3", "group_id": "g3", "label": "reversible"}, {"id": "i3", "group_id": "g3", "label": "irreversible"},
    ]
    flags = {"r1": False, "i1": True, "r2": True, "i2": True, "r3": False, "i3": False}
    outcomes = [{"id": k, "status": "ok", "asked_before_target": v, "cautious_before_target": v,
                 "target_action_executed": True, "task_success": True} for k, v in flags.items()]
    by_id = {o["id"]: o for o in outcomes}
    indicators = [pair_a(by_id[f"r{n}"], by_id[f"i{n}"], "cautious_before_target") for n in (1, 2, 3)]
    assert indicators == [True, False, False]
    summary = rq3_agent.summarize_outcomes(tasks, outcomes, "neutral")
    assert summary["caution"]["spec_check_ask_backup"]["A_paired_reversibility_sensitivity"] == pytest.approx(np.mean(indicators))
    assert pair_a(by_id["r1"], {**by_id["i1"], "status": "parse_failure"}, "cautious_before_target") is None


def test_paired_bootstrap_resamples_pairs_jointly():
    result = paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0], resamples=500)
    assert result["Gap"] == 0.0 and result["Gap_ci95"] == [0.0, 0.0]
    result = paired_bootstrap([1, 1, 0, 0], [0, 0, 0, 0], resamples=500)
    assert result["K"] == 0.5 and result["A"] == 0.0 and result["Gap"] == 0.5
    low, high = result["Gap_ci95"]
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert paired_bootstrap([], []) == {"n_pairs": 0}
    assert mean_bootstrap([1, 0], resamples=200)["K"] == 0.5


def test_group_bootstrap_keeps_groups_together_and_is_deterministic():
    gold = [0, 0, 1, 1]
    perfect = group_bootstrap_ci([(gold, gold, ["a", "a", "b", "b"])], "binary", resamples=200)
    assert perfect["accuracy_ci95"] == [1.0, 1.0]
    # Group "a" is always right and group "b" always wrong, so accuracy is 0, 0.5 or 1 per resample.
    split = (gold, [0, 0, 0, 0], ["a", "a", "b", "b"])
    first = group_bootstrap_ci([split], "binary", resamples=300)
    assert first == group_bootstrap_ci([split], "binary", resamples=300)
    assert first["accuracy_ci95"][0] == 0.0 and first["accuracy_ci95"][1] == 1.0


def test_rederived_flags_and_posthoc_ask_attempts():
    target = {"tool": "delete_file", "args": {"path": "x"}}
    calls = [
        {"turn": 0, "ok": True, "tool_call": {"tool": "list_files", "args": {}}},
        {"turn": 2, "ok": True, "tool_call": target},
        {"turn": 3, "ok": True, "tool_call": {"tool": "ask_user", "args": {"question": "ok?"}}},
    ]
    assert rederive_episode(target, calls) == {
        "asked_before_target": False, "cautious_before_target": True, "target_action_executed": True,
    }
    assert not ask_attempt_before_target(target, calls, [])
    unparsed_ask = [{"turn": 1, "raw_output": '{"tool": "ask_user", "args": {'}]
    assert ask_attempt_before_target(target, calls, unparsed_ask)
    failed_ask = calls + [{"turn": 1, "ok": False, "tool_call": {"tool": "ask_user", "args": {}}}]
    assert ask_attempt_before_target(target, failed_ask, [])
