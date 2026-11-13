import numpy as np
import pytest

from experiments.extract_hidden import (
    action_char_span,
    model_slug,
    stratified_limit,
    token_span_from_offsets,
)
from experiments.probes import (
    _auc,
    bootstrap_intervals,
    fit_classifier,
    map_label,
    select_rows,
)


def test_action_span_and_token_offsets_are_retained():
    rendered = "prefix: open door"
    char_span = action_char_span(rendered, "open door")
    assert char_span == (8, 17)
    assert token_span_from_offsets([(0, 7), (8, 12), (13, 17)], char_span) == (1, 3)


def test_action_token_span_errors_are_not_silently_ignored():
    with pytest.raises(ValueError, match="not present"):
        action_char_span("prefix only", "missing action")
    with pytest.raises(ValueError, match="No tokenizer offsets"):
        token_span_from_offsets([(0, 2), (4, 6)], (2, 4))


def test_model_slug_and_scenario_stratified_limit():
    assert model_slug("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen_Qwen2.5-0.5B-Instruct"
    rows = [
        {"id": "a0", "scenario_id": "a"},
        {"id": "a1", "scenario_id": "a"},
        {"id": "b0", "scenario_id": "b"},
        {"id": "c0", "scenario_id": "c"},
    ]
    assert [row["id"] for row in stratified_limit(rows, 3)] == ["a0", "b0", "c0"]
    assert [row["id"] for row in select_rows(rows, 3)] == ["a0", "b0", "c0"]


def test_binary_label_mapping_excludes_partial_and_three_class_keeps_it():
    assert map_label("reversible", "binary") == 0
    assert map_label("irreversible", "binary") == 1
    assert map_label("partially_reversible", "binary") is None
    assert map_label("partially_reversible", "three-class") == 1


def test_probe_classifier_and_bootstrap_metrics():
    x_train = np.array([[0.0], [0.2], [0.8], [1.0]])
    y_train = np.array([0, 0, 1, 1])
    x_test = np.array([[0.1], [0.9]])
    y_test = np.array([0, 1])
    prediction, probabilities = fit_classifier(x_train, y_train, x_test, [0, 1], 7)
    assert prediction.tolist() == [0, 1]
    assert probabilities.shape == (2, 2)
    intervals = bootstrap_intervals(y_test, prediction, probabilities, [0, 1], 20, 7)
    assert intervals["balanced_accuracy"] == [1.0, 1.0]
    assert intervals["macro_f1"] == [0.5, 1.0]
    assert intervals["auroc_valid_resamples"] <= 20


def test_auc_is_undefined_when_a_fold_lacks_a_class():
    assert _auc([0, 1, 1], np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]]), [0, 1, 2]) is None
