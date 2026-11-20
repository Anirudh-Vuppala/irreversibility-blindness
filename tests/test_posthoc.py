import pytest

from experiments.posthoc import direction_summary, most_common_choice, pair_direction


@pytest.mark.parametrize("rev, irr, expected", [
    ("reversible", "irreversible", 1.0),
    ("reversible", "partially_reversible", 1.0),
    ("partially_reversible", "partially_reversible", 0.5),
    ("irreversible", "irreversible", 0.5),
    ("irreversible", "reversible", 0.0),
    ("partially_reversible", "reversible", 0.0),
    (None, "irreversible", None),
])
def test_pair_direction(rev, irr, expected):
    assert pair_direction(rev, irr) == expected


def test_direction_summary_fractions():
    out = direction_summary([1.0, 0.5, 0.5, 0.0], resamples=200)
    assert out["directional_score"] == 0.5
    assert (out["higher_fraction"], out["tied_fraction"], out["lower_fraction"]) == (0.25, 0.5, 0.25)
    assert out["directional_score_ci95"][0] <= 0.5 <= out["directional_score_ci95"][1]


def test_most_common_choice():
    assert most_common_choice(["mean"] * 3, [20, 20, 21])[0] == ("mean", 20)
    choice, rule = most_common_choice(["mean", "mean", "last"], [11, 13, 13])
    assert choice == ("mean", 13) and rule.startswith("no modal")
