import json
import pytest

from scienceworld_slice.core import (
    build_parent_index,
    canonicalize_tree,
    expand_with_parents,
    focus_candidates,
    labels_from_depths,
    matched_state_relation,
    restoration_candidates,
    state_key,
)
from scienceworld_slice.generate import mark_minimal_pairs, round_robin
from scienceworld_slice.generate import _action_is_available


def test_canonicalize_removes_move_counters_and_rounds_temperatures():
    tree = {
        "moveCount": 8,
        "uuid": "1",
        "propMoveable": {"isMovable": True},
        "propMaterial": {"temperatureC": 20.126, "stateOfMatter": "liquid"},
    }

    result = canonicalize_tree(tree)

    assert "moveCount" not in result
    assert result["propMoveable"] == {"isMovable": True}
    assert result["propMaterial"]["temperatureC"] == 20.1
    assert "temperatureC" not in canonicalize_tree(tree, drop_temperatures=True)["propMaterial"]


def test_state_key_ignores_move_count_and_small_temperature_noise():
    first = {"moveCount": 2, "propMaterial": {"temperatureC": 20.121}}
    second = {"moveCount": 9, "propMaterial": {"temperatureC": 20.124}}
    assert state_key(first) == state_key(second)


def test_matched_wait_relation_uses_only_agreed_canonicalization():
    control = {
        "moveCount": 1,
        "contents": {"water": {"temperatureC": 10.04, "amount": 2}},
    }
    exact = {
        "moveCount": 9,
        "contents": {"water": {"temperatureC": 10.03, "amount": 2}},
    }
    temperature_only = {
        "contents": {"water": {"temperatureC": 10.2, "amount": 2}},
    }
    unrelated_drift = {
        "contents": {"water": {"temperatureC": 10.04, "amount": 3}},
    }

    assert matched_state_relation(exact, control, "door closed", "door closed") == "exact"
    assert matched_state_relation(
        temperature_only, control, "door closed", "door closed"
    ) == "temperature_only"
    assert matched_state_relation(unrelated_drift, control, "door closed", "door closed") is None
    assert matched_state_relation(exact, control, "door open", "door closed") is None


def test_object_parent_index_expands_scope_to_original_container():
    tree = {
        "uuid": "1",
        "contents": {"sink-2": {"uuid": "2", "name": "sink", "contents": {}}},
    }
    parents = build_parent_index(tree)
    assert parents["2"] == "1"
    assert expand_with_parents({2}, parents) == {"1", "2"}


def test_focus_candidates_require_focus_verb_and_context_object():
    actions = ["open door", "move cup to sink", "read recipe", "close other door"]
    templates = [
        {"action": "open door", "obj_ids": [10]},
        {"action": "move cup to sink", "obj_ids": [20, 21]},
        {"action": "read recipe", "obj_ids": [10]},
        {"action": "close other door", "obj_ids": [11]},
    ]
    assert focus_candidates(actions, templates, {10}) == [
        ("open door", templates[0])
    ]


def test_restoration_candidates_restrict_verbs_and_object_ids():
    actions = ["open door", "close door", "close window", "move door to room"]
    templates = [
        {"action": "open door", "obj_ids": [10]},
        {"action": "close door", "obj_ids": [10]},
        {"action": "close window", "obj_ids": [12]},
        {"action": "move door to room", "obj_ids": [10, 13]},
    ]
    assert restoration_candidates(actions, templates, {10, 11}, "open door") == [
        "close door", "open door"
    ]


def test_action_availability_uses_pre_step_list_membership_not_truthiness():
    class CandidateEnv:
        def __init__(self, candidates):
            self.candidates = candidates

        def get_valid_action_object_combinations(self):
            return self.candidates

    env = CandidateEnv(["mix door to kitchen"])
    assert _action_is_available(env, "mix door to kitchen") is True
    assert _action_is_available(env, "close cupboard") is False

    with pytest.raises(RuntimeError, match="must return a list of strings"):
        _action_is_available(CandidateEnv(True), "mix door to kitchen")


def test_horizon_labels_use_first_exact_or_partial_restoration():
    assert labels_from_depths(2, 1, 3) == {
        "1": "partially_reversible",
        "2": "reversible",
        "3": "reversible",
    }
    assert labels_from_depths(None, None, 2) == {
        "1": "irreversible",
        "2": "irreversible",
        "3": None,
    }


def test_round_robin_proposals_balance_tasks():
    proposals = [
        {"id": f"a{i}", "task_name": "a"} for i in range(4)
    ] + [{"id": f"b{i}", "task_name": "b"} for i in range(4)]
    assert [row["task_name"] for row in round_robin(proposals, ["a", "b"], 4)] == [
        "a", "b", "a", "b"
    ]


def test_minimal_pair_flag_requires_same_group_and_label_difference(tmp_path):
    path = tmp_path / "slice.jsonl"
    rows = [
        {"id": "a", "group_id": "g", "variant": "p0", "labels_by_horizon": {"1": "reversible"}},
        {"id": "b", "group_id": "g", "variant": "p1", "labels_by_horizon": {"1": "irreversible"}},
        {"id": "c", "group_id": "single", "variant": "p0", "labels_by_horizon": {"1": "reversible"}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert mark_minimal_pairs(path) == 3
    output = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["is_minimal_pair"] for row in output] == [True, True, False]
