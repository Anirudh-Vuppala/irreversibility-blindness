import copy
import json
from pathlib import Path

import pytest

from scienceworld_slice.build_slice import cap_classes, mark_pairs, probe_ok
from scienceworld_slice.projection import (
    door_states,
    failed_observation,
    index_tree,
    labels_from_depths,
    match,
    project,
    root_ids_for,
)

REPO = Path(__file__).resolve().parents[1]
SANDBOX_KEYS = {"id", "environment", "state_text", "action_text", "label", "horizon", "source"}
LABELS = {"reversible", "partially_reversible", "irreversible"}


def node(uuid, name, contents=(), temp=None, state=None, active=None, is_open=None, movable=True):
    return {
        "uuid": uuid,
        "name": name,
        "isDeleted": False,
        "propMaterial": None if temp is None else {
            "temperatureC": temp,
            "stateOfMatter": state or "solid",
            "substanceName": name if state == "liquid" else None,
        },
        "propContainer": {"isContainer": "true", "isOpen": bool(is_open), "isClosable": is_open is not None},
        "propDevice": None if active is None else {"isActivable": True, "isActivated": active},
        "propMoveable": {"isMovable": movable},
        "contents": {f"{child['name']}-{child['uuid']}": child for child in contents},
    }


def world(pot_temp=10.0, water_temp=10.0, stove_on=False, fountain_temp=10.0, extra_in_pot=()):
    water = node("5", "water", temp=water_temp, state="liquid")
    terminal = node("9", "terminal 1")
    pot = node("4", "metal pot", [water, terminal, *extra_in_pot], temp=pot_temp)
    stove = node("3", "stove", [pot], active=stove_on, movable=False)
    bowl = node("13", "bowl", temp=10.0)
    inventory = node("7", "inventory")
    agent = node("6", "agent", [inventory])
    fountain = node("8", "fountain", [node("10", "water", temp=fountain_temp, state="liquid")], movable=False)
    kitchen = node("2", "kitchen", [stove, bowl, agent], movable=False)
    outside = node("11", "outside", [fountain], movable=False)
    return node("1", "universe", [kitchen, outside], movable=False)


LOOK = "This room is called the kitchen.\nYou also see:\n\tA door to the hallway (that is open)\n"


def projected(tree, look=LOOK, action_ids=("3",)):
    return project(tree, look, root_ids_for(action_ids, index_tree(tree)))


def test_drifting_temperature_does_not_change_the_label():
    target = projected(world(pot_temp=10.0, water_temp=10.0))
    drifted = projected(world(pot_temp=10.6, water_temp=9.7))
    assert match(drifted, target) == "exact"
    # same search outcome -> same labels, regardless of the 0.6 C drift
    assert labels_from_depths(1, None) == {"1": "reversible", "2": "reversible", "3": "reversible"}


def test_unrelated_world_drift_is_outside_the_projection():
    target = projected(world(fountain_temp=10.0))
    other = projected(world(fountain_temp=40.0))
    assert match(other, target) == "exact"


def test_residual_heat_is_an_approximate_match():
    target = projected(world(pot_temp=10.0))
    heated = projected(world(pot_temp=33.0, water_temp=11.2))
    assert match(heated, target) == "approximate"


def test_discrete_change_is_no_match():
    target = projected(world())
    assert match(projected(world(stove_on=True)), target) is None
    salty = node("12", "sodium chloride", temp=10.0)
    assert match(projected(world(extra_in_pot=(salty,))), target) is None
    closed = LOOK.replace("(that is open)", "(that is closed)")
    assert match(projected(world(), look=closed), target) is None


def test_terminals_and_door_parsing():
    pot = projected(world())["roots"]["3"]["contents"][0]
    assert pot["name"] == "metal pot"
    assert [child["name"] for child in pot["contents"]] == ["water"]
    assert door_states(LOOK) == ["hallway:open"]


def test_labels_by_horizon_from_one_search():
    assert labels_from_depths(2, None) == {"1": "irreversible", "2": "reversible", "3": "reversible"}
    assert labels_from_depths(3, 1) == {
        "1": "partially_reversible", "2": "partially_reversible", "3": "reversible",
    }
    assert labels_from_depths(None, 0) == {
        "1": "partially_reversible", "2": "partially_reversible", "3": "partially_reversible",
    }
    assert labels_from_depths(None, None) == {"1": "irreversible", "2": "irreversible", "3": "irreversible"}


def test_failure_observations_are_no_ops():
    assert failed_observation("That container is empty, so there are no items to mix.")
    assert failed_observation("Mixing the contents of the flower pot 6 does not appear to produce anything new.")
    assert failed_observation("No known action matches that input.")
    assert not failed_observation("The stove is now activated.")
    assert not failed_observation("You pour the contents of the metal pot into the stove.")


def test_probe_filter_rejects_fixtures_and_self_containment():
    tree = world()
    index = index_tree(tree)
    assert probe_ok("pour", ["4", "13"], index)         # pot with water into the bowl
    assert not probe_ok("pour", ["4", "3"], index)      # into its current container
    assert not probe_ok("move", ["3", "4"], index)      # stove is a fixture
    assert not probe_ok("pour", ["4", "5"], index)      # into its own contents
    assert not probe_ok("mix", ["3"], index)            # stove holds one item


def _row(identifier, label, key):
    return {"id": identifier, "label": label, "pair_key": key, "is_minimal_pair": False, "pair_id": None}


def test_pairs_and_class_cap():
    rows = [
        _row("a", "reversible", "k1"), _row("b", "irreversible", "k1"),
        _row("c", "reversible", "k2"), _row("d", "reversible", "k3"), _row("e", "reversible", "k4"),
        _row("f", "partially_reversible", "k5"),
    ]
    rows = mark_pairs(rows)
    assert [row["is_minimal_pair"] for row in rows] == [True, True, False, False, False, False]
    assert rows[0]["pair_id"] == rows[1]["pair_id"] == "k1"
    capped = cap_classes(copy.deepcopy(rows), 0.5)
    counts = {label: sum(row["label"] == label for row in capped) for label in LABELS}
    assert counts["reversible"] <= 0.5 * len(capped)
    assert {"a", "b"} <= {row["id"] for row in capped}


@pytest.mark.skipif(not (REPO / "data" / "scienceworld.jsonl").exists(), reason="slice not generated")
def test_released_rows_follow_the_sandbox_schema():
    rows = [json.loads(line) for line in (REPO / "data" / "scienceworld.jsonl").open() if line.strip()]
    assert rows
    for row in rows:
        assert SANDBOX_KEYS <= set(row)
        assert row["environment"] == "scienceworld"
        assert row["horizon"] == 3 and row["label"] in LABELS
        assert set(row["labels_by_horizon"]) == {"1", "2", "3"}
        assert row["labels_by_horizon"]["3"] == row["label"]
        assert row["group_id"] == f"sw-{row['task_name']}-v{row['variation']:03d}"
        assert (row["pair_id"] is not None) == row["is_minimal_pair"]
        assert "In your inventory" in row["state_text"] and "state_json" not in row
        assert len(json.dumps(row)) < 8000
        # horizon monotonicity: a label can only get more reversible with a longer horizon
        order = {"irreversible": 0, "partially_reversible": 1, "reversible": 2}
        values = [order[row["labels_by_horizon"][h]] for h in ("1", "2", "3")]
        assert values == sorted(values)
    assert len({row["id"] for row in rows}) == len(rows)
