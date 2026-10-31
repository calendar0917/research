"""Zero-extra equivalence, weight migration, gradients and training interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import v3_common as C
import v3_models as M


def _batch(fx, n=32):
    view = C.D.build_fold_view(
        C.FOLD,
        cache=C.load_fit_cache()["cache"],
        y_fit=np.asarray(C.load_fit_labels()["y"], dtype=np.float64),
        folds=C.F.load_folds(),
    )
    rows = list(range(min(n, int(view.dev_cache["n_molecules"]))))
    batch = C.D.collate(view.dev_cache, rows)
    extra = np.asarray(fx.dev_raw[rows], dtype=np.float64)
    return batch, extra


def _aug_batch(batch, extra):
    out = dict(batch)
    out["global"] = torch.cat(
        [batch["global"], torch.as_tensor(np.asarray(extra), dtype=torch.float32)], dim=1
    )
    return out


@pytest.mark.parametrize("arm,variant", [("A_Chem12", "chem12"), ("A_Chem13", "chem13"), ("C_Chem12", "chem12")])
def test_zero_standardised_extra_forward_is_equivalent(arm, variant, request):
    fx = request.getfixturevalue("fold_extras_12" if variant == "chem12" else "fold_extras_13")
    model, meta = M.build_chem_model(arm, fx.spec, fx.norm, C.SEED, "cpu", dropout=0.0)
    original = meta["shared_with_original"]
    model.eval()
    original.eval()
    batch, _extra = _batch(fx)
    zero = np.tile(np.asarray(fx.audit["extra_norm"]["mean"], dtype=np.float64), (batch["n_graphs"], 1))
    z = (zero - np.asarray(fx.audit["extra_norm"]["mean"])) / np.asarray(fx.audit["extra_norm"]["scale"])
    assert np.allclose(z, 0.0)
    with torch.no_grad():
        y0 = original(batch).detach().cpu().numpy().reshape(-1)
        y1 = model(_aug_batch(batch, zero)).detach().cpu().numpy().reshape(-1)
    assert np.allclose(y0, y1, rtol=0.0, atol=1e-6)
    assert meta["migration"]["all_shared_bitwise_equal"] is True
    assert meta["migration"]["rho_first_layer"]["new_columns_initialised_to_zero"] is True
    n_extra = fx.train_raw.shape[1]
    assert torch.count_nonzero(model.rho.lins[0].weight[:, 42 : 42 + n_extra]) == 0


@pytest.mark.parametrize("arm,variant,expected", [("A_Chem12", "chem12", 768), ("A_Chem13", "chem13", 832), ("C_Chem12", "chem12", 768)])
def test_parameter_delta_is_explicit_and_small(arm, variant, expected, request):
    fx = request.getfixturevalue("fold_extras_12" if variant == "chem12" else "fold_extras_13")
    _model, meta = M.build_chem_model(arm, fx.spec, fx.norm, C.SEED, "cpu", dropout=0.0)
    assert meta["parameter_delta"] == expected
    assert meta["parameters_augmented"] == meta["parameters_original"] + expected
    assert meta["parameter_delta"] / meta["parameters_original"] < 0.004


@pytest.mark.parametrize("arm,variant", [("A_Chem12", "chem12"), ("A_Chem13", "chem13"), ("C_Chem12", "chem12")])
def test_new_rho_columns_receive_finite_nonzero_gradient(arm, variant, request):
    fx = request.getfixturevalue("fold_extras_12" if variant == "chem12" else "fold_extras_13")
    model, _meta = M.build_chem_model(arm, fx.spec, fx.norm, C.SEED, "cpu", dropout=0.0)
    model.train()
    batch, extra = _batch(fx)
    model.zero_grad(set_to_none=True)
    pred = model(_aug_batch(batch, extra))
    (pred * pred).mean().backward()
    grad = model.rho.lins[0].weight.grad
    assert grad is not None
    new = grad[:, 42 : 42 + fx.train_raw.shape[1]].detach().cpu().numpy()
    assert np.isfinite(new).all()
    assert np.count_nonzero(new) > 0


def test_shared_encoder_state_is_bitwise_identical(request):
    fx = request.getfixturevalue("fold_extras_12")
    for arm, prefixes in (
        ("A_Chem12", ("atom_encoder.", "bond_encoder.", "local.")),
        ("C_Chem12", ("input_proj.", "convs.", "norms.")),
    ):
        model, meta = M.build_chem_model(arm, fx.spec, fx.norm, C.SEED, "cpu", dropout=0.0)
        base = meta["shared_with_original"]
        b = base.state_dict()
        a = model.state_dict()
        for key, value in b.items():
            if key.startswith(prefixes):
                assert key in a
                assert tuple(a[key].shape) == tuple(value.shape)
                assert torch.equal(a[key], value), key


def test_chem_train_job_uses_v2_loop_in_a_temporary_directory(request):
    fx = request.getfixturevalue("fold_extras_12")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        job = M.ChemTrainJob(
            key="test_A_Chem12",
            arm="A_Chem12",
            fold=C.FOLD,
            recipe=C.RECIPE,
            seed=C.SEED,
            device="cpu",
            out_root=tmp,
            smoke_train_limit=64,
            smoke_epochs=1,
            global_extra_train=fx.train_raw,
            global_extra_dev=fx.dev_raw,
            global_extra_audit={"names": list(C.CHEM12_NAMES), "n_extra": 12, "labels_used": False},
            spec_override=fx.spec,
            norm_override=fx.norm,
            v3_audit={"test": True},
        )
        payload = M.train_chem_job(job)
        run_dir = tmp / job.key
        assert (run_dir / "train.json").exists()
        assert (run_dir / "soup.pt").exists()
        assert (run_dir / "predictions.npz").exists()
        assert payload["protocol_id"] == C.PROTOCOL_ID
        assert payload["v2_loop"]["function"] == "v2_training.train_job reused verbatim"
        assert payload["v3"]["new_parameters"] == 768
        assert payload["v3"]["training_loop_changed"] is False
        assert np.isfinite(payload["metrics"]["dev_rmse_raw"])
        assert not (C.RUNS / job.key).exists()


def test_job_validation_refuses_unregistered_variations(request):
    fx = request.getfixturevalue("fold_extras_12")
    base = dict(
        key="bad",
        arm="A_Chem12",
        fold=C.FOLD,
        recipe=C.RECIPE,
        seed=C.SEED,
        device="cpu",
        global_extra_train=fx.train_raw,
        global_extra_dev=fx.dev_raw,
        spec_override=fx.spec,
        norm_override=fx.norm,
    )
    for change in (
        {"fold": 0},
        {"seed": 1},
        {"recipe": "R0"},
        {"arm": "A"},
        {"variant": "ssl"},
    ):
        with pytest.raises(ValueError):
            M.ChemTrainJob(**{**base, **change}).validate()
