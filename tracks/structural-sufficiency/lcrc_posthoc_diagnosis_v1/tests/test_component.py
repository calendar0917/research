import numpy as np

import ph_common as C


def test_component_equal_weighting():
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    comps = ["a", "a", "b", "c"]
    # a -> 1.5, b -> 3, c -> 4
    assert abs(C.component_equal_mae(errors, comps) - (1.5 + 3.0 + 4.0) / 3) < 1e-12


def test_component_equal_rmse():
    errors = np.array([3.0, 4.0, 0.0, 2.0])
    comps = ["a", "a", "b", "c"]
    # a -> sqrt(12.5), b -> 0, c -> 2
    want = (np.sqrt(12.5) + 0.0 + 2.0) / 3
    assert abs(C.component_equal_rmse(errors, comps) - want) < 1e-12


def test_bootstrap_selected_component_multiplicity():
    # Mirrors the local-edit bootstrap: selected duplicates have full weight.
    comp_means = {"a": 1.5, "b": 3.0, "c": 4.0}
    selected = ["a", "a", "b"]
    got = np.mean([comp_means[c] for c in selected])
    assert abs(got - (1.5 + 1.5 + 3.0) / 3) < 1e-12


def test_component_names_are_strings():
    errors = np.array([1.0, 2.0])
    comps = [1, 1]
    assert abs(C.component_equal_mae(errors, comps) - 1.5) < 1e-12
