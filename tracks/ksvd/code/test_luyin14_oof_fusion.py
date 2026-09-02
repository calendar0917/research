from __future__ import annotations

import numpy as np

from .run_luyin14_oof_fusion import (
    _apply_gate_fusion,
    _apply_scalar_fusion,
    _centered_log_probabilities,
    _fit_gate_fusion,
    _fit_scalar_fusion,
    _fusion_outputs,
)


def test_centered_logits_are_shift_invariant() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    logits = _centered_log_probabilities(probabilities)
    assert np.allclose(logits.mean(axis=1), 0.0)


def test_scalar_fusion_ignores_adversarial_structure() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    base = np.asarray(
        [[.9, .1], [.8, .2], [.2, .8], [.1, .9], [.7, .3], [.3, .7]]
    )
    structure = base[:, ::-1]
    alpha = _fit_scalar_fusion(base, structure, labels)
    fused = _apply_scalar_fusion(base, structure, alpha)
    assert alpha <= 0.0
    assert np.mean(np.argmax(fused, axis=1) == labels) == 1.0


def test_gate_returns_valid_low_structure_weights_when_structure_is_bad() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    base = np.asarray(
        [[.9, .1], [.8, .2], [.2, .8], [.1, .9], [.7, .3], [.3, .7]]
    )
    structure = np.full_like(base, 0.5)
    parameters = _fit_gate_fusion(base, structure, labels)
    probabilities, weights = _apply_gate_fusion(base, structure, parameters)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(weights) < 0.5


def test_all_fusion_outputs_are_probabilities() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    base = np.asarray(
        [[.8, .2], [.7, .3], [.3, .7], [.2, .8], [.65, .35], [.4, .6]]
    )
    structure = np.asarray(
        [[.7, .3], [.6, .4], [.2, .8], [.3, .7], [.75, .25], [.35, .65]]
    )
    outputs, metadata = _fusion_outputs(
        base, structure, labels, base, structure, seed=0
    )
    assert set(outputs) == {
        "FIXED_AVG",
        "LOGIT_ADD",
        "OOF_SCALAR",
        "OOF_GATE",
        "OOF_STACK",
    }
    assert -1.0 <= metadata["scalar_alpha"] <= 1.0
    for probabilities in outputs.values():
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0.0)

