"""Focused CPU tests for E2E-DictEnv-Joint709-Absolute-v1.

These tests are data-free and checkpoint-free: they cover the frozen geometry,
the absolute-performance bands, the real tied-IHT reconstruction diagnostic,
the route-2 joint-dictionary coordinate path, the refactored explicit
``results_dir`` / ``model_factory`` / ``joint_values`` wiring, and the
control-plane runner contract.

The synthetic-batch / synthetic-model helpers are shared with the sibling
increment-v2 test module so both rounds test the same construction.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_joint709_absolute_v1 as core
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_increment_v2 as inc
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_joint709_absolute_v1 as stages
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_rolecorr_increment_v2 as incrun
from tracks.ksvd.src.ksvd_research.runners import (
    zinc_e2e_dictenv_joint709_absolute_v1 as runner,
)
from tracks.ksvd.tests import test_e2e_dictenv_rolecorr_increment_v2 as helpers


# ---------------------------------------------------------------------------
# frozen geometry / bands
# ---------------------------------------------------------------------------


def test_frozen_geometry():
    assert core.JOINT_DIM == 65 + 108 + 536 == 709
    assert core.JOINT_ATOMS == 48
    assert core.JOINT_SPARSITY == 12
    assert core.IHT_STEPS == 10  # iterations, NOT the sparsity level
    assert core.COORD_DIM == 49
    assert (core.JOINT_SLICE.start, core.JOINT_SLICE.stop) == (1, 49)
    assert core.CANDIDATE == inc.ARM_JOINT_SPARSE
    assert core.DICT_EPOCHS == 10 and core.DICT_SEED == 20260924
    assert list(core.JOINT_BLOCK_SLICES) == ["struct", "sem", "corr"]
    assert core.JOINT_BLOCK_SLICES["struct"] == slice(0, 65)
    assert core.JOINT_BLOCK_SLICES["sem"] == slice(65, 173)
    assert core.JOINT_BLOCK_SLICES["corr"] == slice(173, 709)


def test_absolute_band_boundaries():
    assert core.absolute_band(0.1000)["band"] == core.BAND_STRONG
    assert core.absolute_band(0.1150)["band"] == core.BAND_STRONG
    assert core.absolute_band(0.1151)["band"] == core.BAND_PROMISING
    assert core.absolute_band(0.1200)["band"] == core.BAND_PROMISING
    assert core.absolute_band(0.1201)["band"] == core.BAND_BORDERLINE
    assert core.absolute_band(0.1233)["band"] == core.BAND_BORDERLINE
    assert core.absolute_band(0.1234)["band"] == core.BAND_STOP
    payload = core.absolute_band(0.1233)
    assert payload["statistical_significance_gate"] is False
    assert payload["official_test_loaded"] is False
    assert "borderline" in payload["recommendation"]
    rows = core.band_metrics_rows("abs", payload)
    assert rows["abs_band"] == core.BAND_BORDERLINE
    assert rows["abs_soup_mae"] == pytest.approx(0.1233)
    with pytest.raises(ValueError):
        core.absolute_band(float("nan"))


# ---------------------------------------------------------------------------
# tied-IHT chunking and the real reconstruction diagnostic
# ---------------------------------------------------------------------------


def _synthetic_joint(rows: int = 128, seed: int = 3):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((core.JOINT_DIM, core.JOINT_ATOMS)).astype(np.float32)
    codes = rng.standard_normal((rows, core.JOINT_ATOMS)).astype(np.float32)
    mask = rng.random((rows, core.JOINT_ATOMS)) < (core.JOINT_SPARSITY / core.JOINT_ATOMS)
    codes = codes * mask
    return D, codes


def test_tied_iht_chunked_matches_reference_and_l0():
    D, codes = _synthetic_joint(rows=64)
    rng = np.random.default_rng(11)
    X = (codes @ D.T + 0.01 * rng.standard_normal((64, core.JOINT_DIM))).astype(np.float32)
    chunked = core.tied_iht_codes_chunked(D, X, s=core.JOINT_SPARSITY, steps=core.IHT_STEPS, chunk=16)
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

    Dbar = v0.normalized_dictionary(torch.as_tensor(D, dtype=torch.float32))
    reference = v0.tied_iht_codes(
        Dbar, torch.as_tensor(X), s=core.JOINT_SPARSITY, steps=core.IHT_STEPS
    ).numpy()
    assert np.array_equal(chunked, reference)
    assert int((chunked != 0).sum(axis=1).max()) <= core.JOINT_SPARSITY


def test_joint_reconstruction_diagnostic_sane():
    D, codes = _synthetic_joint(rows=256)
    X = (codes @ D.T).astype(np.float32)
    diag = core.joint_reconstruction_diagnostic(D, X, omp_rows=32)
    assert diag["n_rows"] == 256 and diag["dim"] == core.JOINT_DIM
    assert diag["iht_steps"] == core.IHT_STEPS and diag["s"] == core.JOINT_SPARSITY
    assert diag["exact_l0_max"] <= core.JOINT_SPARSITY
    assert np.isfinite(diag["relative_error_tied_iht"]) and diag["relative_error_tied_iht"] < 1.0
    assert set(diag["blocks"]) == {"struct", "sem", "corr"}
    assert diag["omp_reference"]["n_rows"] == 32
    assert np.isfinite(diag["omp_reference"]["relative_error_omp"])
    assert diag["official_test_loaded"] is False
    with pytest.raises(RuntimeError):
        core.joint_reconstruction_diagnostic(D, X[:, :100])


# ---------------------------------------------------------------------------
# route-2 joint-dictionary coordinate path
# ---------------------------------------------------------------------------


def _route2_model(seed: int = 0):
    models, dictionaries = helpers._build_all_arms(seed=seed)
    return models[inc.ARM_JOINT_SPARSE], dictionaries


def test_route2_coordinate_shape_sparsity_and_freezing():
    model, dictionaries = _route2_model()
    batch = helpers._synthetic_batch()
    with torch.no_grad():
        coord = model.code(batch.dict_phi, batch.joint_vec)
    assert tuple(coord.shape) == (batch.dict_phi.shape[0], core.COORD_DIM)
    assert int((coord[:, core.JOINT_SLICE] != 0).sum(dim=1).max()) <= core.JOINT_SPARSITY
    assert tuple(model.D_block.shape) == (core.JOINT_DIM, core.JOINT_ATOMS)
    assert not model.D.requires_grad and not model.D_block.requires_grad
    purity = inc.zero_block_purity(model, batch.dict_phi, batch.joint_vec)
    assert purity["span"] == [1, 49]
    assert purity["other_columns_bit_identical"] and purity["block_columns_zero"]
    assert inc.zero_base_purity(model, batch.dict_phi, None)["applicable"] is False


def test_route2_coordinate_is_pure_function_and_relabel_equivariant():
    model, _ = _route2_model()
    batch = helpers._synthetic_batch()
    with torch.no_grad():
        coord = model.code(batch.dict_phi, batch.joint_vec)
        perturbed = batch.joint_vec + 0.5 * torch.randn_like(batch.joint_vec)
        moved = model.code(batch.dict_phi, perturbed)
        assert not torch.equal(coord[:, core.JOINT_SLICE], moved[:, core.JOINT_SLICE])
        assert torch.equal(coord[:, :1], moved[:, :1])
        permutation = torch.tensor([3, 0, 4, 1, 2])
        permuted = model.code(batch.dict_phi[permutation], batch.joint_vec[permutation])
        inverse = torch.argsort(permutation)
        # BLAS reductions can round differently for a permuted row layout, so the
        # invariance is asserted up to a tight tolerance plus an identical active
        # support (no atom-selection flips)
        assert float((permuted[inverse] - coord).abs().max()) <= 1e-5
        assert torch.equal(
            (permuted[inverse][:, core.JOINT_SLICE] != 0),
            (coord[:, core.JOINT_SLICE] != 0),
        )


def test_route2_prediction_uses_the_joint_block():
    model, _ = _route2_model()
    batch = helpers._synthetic_batch()
    model.eval()
    with torch.no_grad():
        intact = model(batch, mask=helpers.cm_mask())
        model.inference_zero_block = True
        try:
            zeroed = model(batch, mask=helpers.cm_mask())
        finally:
            model.inference_zero_block = False
    assert torch.isfinite(intact).all() and tuple(intact.shape) == (1,)
    # the shared binding rows 1..49 carry the parent weights, so the untrained
    # model is already sensitive to the joint coordinate
    assert float((intact - zeroed).abs().max()) > 1e-6


# ---------------------------------------------------------------------------
# refactored explicit wiring (results_dir / model_factory / joint_values)
# ---------------------------------------------------------------------------


def test_attach_joint_values_respects_node_offsets(monkeypatch):
    from torch_geometric.data import Data

    node_sizes = [4, 2, 3]
    monkeypatch.setattr(
        incrun.rcrun,
        "_env_blob",
        lambda split: {"node_sizes": torch.as_tensor(node_sizes, dtype=torch.long)},
    )
    values = np.arange(sum(node_sizes) * core.JOINT_DIM, dtype=np.float32).reshape(
        sum(node_sizes), core.JOINT_DIM
    )
    items = [Data() for _ in node_sizes]
    for item in items:
        item.num_nodes = 0
    incrun._attach_values(items, "train", values, name="joint_vec")
    offset = 0
    for item, size in zip(items, node_sizes):
        assert tuple(item.joint_vec.shape) == (size, core.JOINT_DIM)
        assert np.array_equal(item.joint_vec.numpy(), values[offset : offset + size])
        offset += size
    # a prefix subset must consume exactly the matching prefix rows
    prefix = [Data(), Data()]
    incrun._attach_arm_block(prefix, "train", core.CANDIDATE, joint_values=values)
    assert np.array_equal(prefix[1].joint_vec.numpy(), values[node_sizes[0] : node_sizes[0] + node_sizes[1]])
    # permuted-object path stays distribution-compatible
    permuted = [Data(), Data(), Data()]
    incrun._attach_arm_block(
        permuted, "train", core.CANDIDATE, joint_values=values, permute_seed=101
    )
    stacked = np.concatenate([item.joint_vec.numpy() for item in permuted], axis=0)
    assert stacked.shape == values.shape
    assert sorted(stacked[:, 0].tolist()) == sorted(values[:, 0].tolist())


def test_refactored_signatures_and_factory_contract():
    for name in (
        "stage_joint_scaler",
        "build_joint_cache",
        "load_joint",
        "stage_joint_objects",
        "load_joint_dictionary",
        "load_joint_pca",
        "build_arm_route2",
        "arm_kwargs_route2",
        "run_route2",
    ):
        signature = inspect.signature(getattr(incrun, name))
        assert "results_dir" in signature.parameters, name
    assert "model_factory" in inspect.signature(incrun.train_arm_device).parameters
    assert "model_factory" in inspect.signature(incrun._soup_model).parameters
    assert "joint_values" in inspect.signature(incrun._attach_arm_block).parameters
    # the round's own factory refuses any other arm
    with pytest.raises(KeyError):
        stages.model_factory(inc.ARM_CORR_ADD)


def test_stage_and_runner_contract():
    assert stages.PREPARE_STAGES[0] == "verify_frozen"
    assert "joint_objects" in stages.PREPARE_STAGES
    assert stages.SCREEN_STAGES == ("correctness", "smoke", "train", "interventions", "analysis")
    assert stages.SMOKE_EPOCHS <= 4
    assert stages.TRAIN_EPOCHS == 320
    assert stages.SEED == 0
    for name in ("summary.json", "REPORT.md", "DECISION.md", "run_JOINT-SPARSE.json"):
        assert name in runner.ARTIFACTS
    assert runner.DEFAULT_CONFIG.exists()
    assert runner._config_stage({"model": {"stage": "prepare"}}) == "prepare"
    assert runner._config_stage({"model": {"stage": "screen"}}) == "screen"
    assert runner._config_stage({}) == "screen"
    with pytest.raises(ValueError):
        runner._config_stage({"model": {"stage": "route2"}})
    assert stages.RESULTS_DIR.name == "e2e_dictenv_joint709_absolute_v1"
    assert stages.RESULTS_DIR != incrun.RESULTS_DIR
