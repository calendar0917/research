"""Unit checks for the zinc_graph_dictionary_readout_v1 round.

Lightweight (no ZINC data, no checkpoint): the frozen vocabulary of the round,
the vendored prototype-dictionary / certified-MAE head on synthetic arrays, and
the PyTorch readout scaffold against the NumPy folded prediction.

The real frozen-backbone export, the cached train-only fit and the one paired
official-valid screen are exercised by the control-plane runner, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ksvd_research.runners import get_runner
from tracks.ksvd.experiments.luyin16 import zinc_graph_dictionary_readout_v1 as stages
from tracks.ksvd.experiments.luyin16.graph_dictionary_readout.v1_20261002 import (
    prototype_dictionary as pd,
)
from tracks.ksvd.experiments.luyin16.graph_dictionary_readout.v1_20261002 import (
    run_cached_head as rch,
)
from tracks.ksvd.experiments.luyin16.graph_dictionary_readout.v1_20261002 import (
    torch_readout_scaffold as trs,
)


def test_runner_is_registered():
    runner = get_runner("zinc_graph_dictionary_readout_v1")
    assert runner.default_study == "zinc-context-gap"
    info = runner.info()
    assert info["name"] == "zinc_graph_dictionary_readout_v1"
    assert Path(info["default_config"]).is_file()


def test_frozen_vocabulary():
    assert stages.EXPECTED_FULL_PARAMETERS == 408651
    assert stages.EXPECTED_VALID_MAE == 0.1191540920053958
    assert pd.N_PROTOTYPES == 256
    assert pd.SEED == 20261002
    assert pd.LAMBDA_GRID == (1e-5, 1e-4, 1e-3)
    assert sum(pd.FULL_BLOCK_WIDTHS) == 814


def test_official_test_blocker():
    stages.official_test_blocker({"official_test_loaded": False})
    with pytest.raises(RuntimeError):
        stages.official_test_blocker({"official_test_loaded": True})


def test_prototype_fit_and_folded_equivalence(tmp_path):
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(300, sum(pd.FULL_BLOCK_WIDTHS)))
    raw[:, 288] = 0.0  # C6-like constant slot
    target = (
        0.4 * np.tanh(raw[:, 0]) + 0.3 * raw[:, 300] - 0.2 * raw[:, 600]
    )
    dictionary = pd.PrototypeDictionary.fit(raw, n_prototypes=pd.N_PROTOTYPES)
    assert dictionary.centers.shape[0] == 256
    design = dictionary.design(raw)
    coef, report = pd.fit_mae(design, target, 1e-4, gap_tolerance=1e-6)
    assert report["status"] == "CONVERGED"
    assert report["certificate"]["gap"] <= 1e-6

    median = float(np.median(target))
    folded = pd.folded_predict(dictionary, raw, coef, median)
    direct = median + design @ coef
    assert np.max(np.abs(folded - direct)) < 1e-9

    # Torch scaffold must reproduce the NumPy folded prediction.
    model_path = tmp_path / "model.npz"
    rch.save_model(model_path, dictionary, coef, median, "0" * 64, 1e-4)
    readout = trs.GraphPrototypeReadout(model_path)
    with torch.no_grad():
        torch_pred = readout(torch.from_numpy(raw.astype(np.float32))).numpy()
    assert np.max(np.abs(torch_pred - folded)) < 1e-5

    # Buffers only; the scaffold holds no trainable parameters.
    assert not list(readout.parameters())
    for name in ("mean", "scale", "weights", "centers", "values", "offset"):
        assert name in dict(readout.named_buffers())

    # A state_dict round-trips a prediction exactly.
    rebuilt = trs.GraphPrototypeReadout(model_path)
    rebuilt.load_state_dict(readout.state_dict())
    with torch.no_grad():
        replay = rebuilt(torch.from_numpy(raw.astype(np.float32))).numpy()
    assert np.array_equal(torch_pred, replay)


def test_paired_screen_gate_logic():
    def gate(original, new, bins):
        gain = original - new
        return (
            gain >= 0.006
            and new <= 0.113
            and sum(x > 0 for x in bins) >= 4
        )

    assert gate(0.119, 0.110, [0.1, 0.1, 0.1, -0.1, 0.1])
    assert not gate(0.119, 0.1135, [0.1, 0.1, 0.1, 0.1, 0.1])
    assert not gate(0.119, 0.110, [0.1, -0.1, -0.1, -0.1, 0.1])
    assert not gate(0.248, 0.248, [-0.1, -0.1, -0.1, -0.1, -0.1])