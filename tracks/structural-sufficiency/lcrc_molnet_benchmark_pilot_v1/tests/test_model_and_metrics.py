"""Frozen-model and metric tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import mb_common as C
import mb_data as D
import mb_model as M
import mb_train as T

import lcrc_data as PREP_DATA


EXPECTED_PARAMETERS = 247_106


@pytest.fixture(scope="module")
def built(small_view):
    model, meta = M.build_lcrc(small_view.spec, small_view.train_norm, 0, torch.device("cpu"))
    return model, meta


def test_parameter_count_is_frozen(built):
    _model, meta = built
    assert meta["parameters"] == EXPECTED_PARAMETERS
    assert meta["config"]["width"] == 64
    assert meta["input_widths"]["atom_feat"] == 51
    assert meta["input_widths"]["bond_feat"] == 13
    assert meta["input_widths"]["global"] == 42
    assert meta["input_widths"]["local_mlp_input"] == 529


def test_no_message_passing_modules(built):
    model, _meta = built
    audit = M.module_name_audit(model)
    assert audit["passed"], audit["flagged_modules"]
    assert "local" in audit["module_names"]


def test_single_vs_batch_consistency(built, small_view):
    model, _meta = built
    model.eval()
    batch = PREP_DATA.collate(small_view.train_cache, [0, 1, 2, 3])
    single = PREP_DATA.collate(small_view.train_cache, [0])
    with torch.no_grad():
        batched = model(batch).reshape(-1)
        alone = model(single).reshape(-1)
    assert float(batched[0]) == pytest.approx(float(alone[0]), abs=1e-5)


def test_locality_is_radius_two(built, small_view):
    model, _meta = built
    report = M.locality_audit(model, small_view.train_cache, small_view.spec)
    assert report["passed"], report
    assert report["max_abs_h_delta_beyond_radius_2"] == 0.0
    assert report["perturbed_atom_responded"]


def test_forward_backward_all_parameters(built, small_view):
    model, _meta = built
    model.train()
    batch = PREP_DATA.collate(small_view.train_cache, [0, 1, 2])
    pred = model(batch).reshape(-1)
    loss = torch.nn.functional.mse_loss(pred, torch.zeros_like(pred))
    loss.backward()
    n_total = sum(1 for _ in model.parameters())
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    assert n_grad == n_total
    assert math.isfinite(float(loss))


def test_dropout_hook_fires_once_per_forward(built, small_view):
    model, meta = built
    before = meta["dropout_hook"].calls
    model(PREP_DATA.collate(small_view.train_cache, [0]))
    assert meta["dropout_hook"].calls - before == 1


def test_regression_metrics_match_numpy():
    pred = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([1.5, 1.5, 2.5])
    m = T.regression_metrics(pred, y)
    assert m["rmse"] == pytest.approx(math.sqrt(((pred - y) ** 2).mean()))
    assert m["mae"] == pytest.approx(np.abs(pred - y).mean())
    ss_tot = ((y - y.mean()) ** 2).sum()
    assert m["r2"] == pytest.approx(1 - ((pred - y) ** 2).sum() / ss_tot)


def test_classification_metrics_on_separable_logits():
    y = np.asarray([0.0, 0.0, 1.0, 1.0])
    logits = np.asarray([-3.0, -1.0, 1.0, 3.0])
    m = T.classification_metrics(logits, y)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["accuracy_at_0.5"] == pytest.approx(1.0)
    assert m["sigmoid_probability"] is True
    assert m["threshold_used_for_selection"] is None


def test_classification_metrics_reject_single_class():
    with pytest.raises(ValueError, match="undefined"):
        T.classification_metrics(np.asarray([0.1, 0.2]), np.asarray([1.0, 1.0]))


def test_denormalisation_roundtrip(small_view):
    y = small_view.y_valid[:5]
    norm = small_view.target_norm
    scaled = np.asarray((y - norm["mean"]) / norm["scale"])
    back = scaled * float(norm["scale"]) + float(norm["mean"])
    assert np.allclose(back, y)


def test_best_epoch_tracker_regression_keeps_earlier_on_tie():
    tracker = T.BestEpochTracker(task="regression")
    assert tracker.consider(1, {"rmse": 1.0, "mae": 0.5})
    assert not tracker.consider(2, {"rmse": 1.0, "mae": 0.5})
    assert tracker.best_epoch == 1
    assert tracker.consider(3, {"rmse": 0.9, "mae": 0.4})
    assert tracker.best_epoch == 3


def test_best_epoch_tracker_classification_tie_breaks():
    tracker = T.BestEpochTracker(task="classification")
    assert tracker.consider(1, {"roc_auc": 0.8, "bce": 0.5})
    assert not tracker.consider(2, {"roc_auc": 0.8, "bce": 0.6})
    assert tracker.consider(3, {"roc_auc": 0.8, "bce": 0.4})
    assert tracker.best_epoch == 3
    assert tracker.consider(4, {"roc_auc": 0.81, "bce": 0.9})
    assert tracker.best_epoch == 4


def test_formal_seed_guard(tmp_path):
    with pytest.raises(ValueError, match="training seed"):
        T.train_run(
            dataset="FreeSolv",
            seed=1,
            device=torch.device("cpu"),
            out_dir=tmp_path / "run",
            epochs=1,
            smoke=False,
        )
