"""Model-identity and metric tests (frozen LCRC-A, three task families)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import mb2_common as C
import mb2_data as D
import mb2_metrics as MET
import mb2_model as M

import lcrc_data as PREP_DATA


# ---------------------------------------------------------------------------
# model identity
# ---------------------------------------------------------------------------


def test_single_task_parameter_count_is_frozen(freesolv_view):
    model, meta = M.build_lcrc(freesolv_view.spec, freesolv_view.train_norm, 0, torch.device("cpu"))
    assert meta["parameters"] == 247_106 == M.SINGLE_TASK_PARAMETERS
    assert meta["n_outputs"] == 1
    assert meta["input_widths"]["atom_feat"] == 51
    assert meta["input_widths"]["bond_feat"] == 13
    assert meta["input_widths"]["global"] == 42
    assert meta["input_widths"]["local_mlp_input"] == 529


def test_tox21_head_parameter_count(tox21_view):
    model, meta = M.build_lcrc(tox21_view.spec, tox21_view.train_norm, 0, torch.device("cpu"), n_tasks=12)
    assert meta["parameters"] == 248_536
    assert meta["parameters"] != 247_106
    assert meta["n_outputs"] == 12
    assert meta["parameters_breakdown"]["f"] == 4_940
    assert meta["parameters_breakdown"]["rho"] == 11_788
    out = model(D.to_device(PREP_DATA.collate(tox21_view.train_cache, [0, 1, 2]), torch.device("cpu")))
    assert tuple(out.shape) == (3, 12)


def test_multitask_keeps_the_shared_encoder_identical(freesolv_view):
    single, _ = M.build_lcrc(freesolv_view.spec, freesolv_view.train_norm, 0, torch.device("cpu"), n_tasks=1)
    multi, _ = M.build_lcrc(freesolv_view.spec, freesolv_view.train_norm, 0, torch.device("cpu"), n_tasks=12)
    assert M.shared_encoder_state_hash(single) == M.shared_encoder_state_hash(multi)
    # ... and the heads are the only difference
    single_keys = {k for k in single.state_dict() if not k.startswith(("f.", "rho."))}
    multi_keys = {k for k in multi.state_dict() if not k.startswith(("f.", "rho."))}
    assert single_keys == multi_keys


def test_no_message_passing_or_attention_modules(freesolv_view):
    model, _ = M.build_lcrc(freesolv_view.spec, freesolv_view.train_norm, 0, torch.device("cpu"))
    audit = M.module_name_audit(model)
    assert audit["passed"], audit["flagged_modules"]
    assert not any("conv" in n for n in audit["module_names"])


def test_dropout_position_is_the_local_output(freesolv_view):
    model, meta = M.build_lcrc(freesolv_view.spec, freesolv_view.train_norm, 0, torch.device("cpu"), dropout=0.5)
    model.train()
    before = meta["dropout_hook"].calls
    model(D.to_device(PREP_DATA.collate(freesolv_view.train_cache, [0, 1]), torch.device("cpu")))
    assert meta["dropout_hook"].calls - before == 1
    # the frozen dropout module lives on the local encoder output only
    assert isinstance(dict(model.named_modules())["_pilot_dropout_head"], torch.nn.Dropout)


def test_locality_is_radius_two(freesolv_view):
    model, _ = M.build_lcrc(freesolv_view.spec, freesolv_view.train_norm, 0, torch.device("cpu"))
    report = M.locality_audit(model, freesolv_view.train_cache, freesolv_view.spec)
    assert report["passed"], report
    assert report["max_abs_h_delta_beyond_radius_2"] == 0.0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_regression_metrics_known_answer():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 6.0])
    out = MET.regression_metrics(pred, y)
    assert out["rmse"] == pytest.approx(np.sqrt(4.0 / 4.0))
    assert out["mae"] == pytest.approx(0.5)


def test_binary_metrics_known_answer():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    logits = np.array([0.1, 0.4, 0.35, 0.8])
    out = MET.binary_metrics(logits, y)
    assert out["roc_auc"] == pytest.approx(0.75)
    assert out["sigmoid_probability"] is True
    assert out["threshold_used_for_selection"] is None


def test_multitask_metrics_ignore_missing_cells():
    y = np.array([[1.0, np.nan], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    logits = np.array([[2.0, 5.0], [-2.0, 2.0], [1.0, -1.0], [-1.0, 3.0]])
    out = MET.multitask_metrics(logits, y, ["a", "b"])
    assert out["per_task"]["a"]["n_observed"] == 4
    assert out["per_task"]["b"]["n_observed"] == 3
    assert out["n_definable_tasks"] == 2
    assert out["macro_roc_auc"] == pytest.approx(1.0)
    # a zero-fill would have produced a different, larger error
    refilled = np.nan_to_num(y)
    assert MET.masked_bce(logits, y) != pytest.approx(MET.masked_bce(logits, refilled))


def test_multitask_undefined_endpoint_is_reported_not_scored():
    y = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    logits = np.zeros_like(y)
    out = MET.multitask_metrics(logits, y, ["ok", "single_class"])
    assert out["n_definable_tasks"] == 1
    assert out["undefined_tasks"]["single_class"] == "no negatives"
    assert out["per_task"]["single_class"]["roc_auc"] is None
    assert out["macro_roc_auc"] == pytest.approx(out["per_task"]["ok"]["roc_auc"])


def test_masked_bce_matches_hand_computation():
    y = np.array([[1.0, np.nan], [0.0, 1.0]])
    logits = np.array([[0.7, -0.2], [-0.4, 1.3]])
    observed = [0.7, -0.4, 1.3]  # targets 1, 0, 1
    targets = [1.0, 0.0, 1.0]
    manual = np.mean([np.logaddexp(0.0, z) - t * z for z, t in zip(observed, targets)])
    assert MET.masked_bce(logits, y) == pytest.approx(manual)


def test_task_metrics_dispatch_and_primary_names():
    assert MET.primary_name("regression") == "rmse"
    assert MET.primary_name("binary") == "roc_auc"
    assert MET.primary_name("multitask") == "macro_roc_auc"
    assert MET.better("regression", 0.5, 0.6) is True
    assert MET.better("binary", 0.8, 0.9) is False
    assert MET.better("binary", 0.9, None) is True


def test_ogb_evaluator_matches_sklearn():
    report = MET.ogb_evaluator_roc_auc(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert report["ogb_rocauc"] == pytest.approx(0.75)
    assert report["consistent"] is True
    assert report["y_true_shape"] == [4, 1]


# ---------------------------------------------------------------------------
# probe guard
# ---------------------------------------------------------------------------


def test_probe_overrides_are_exactly_the_frozen_four():
    import mb2_train as T

    assert set(C.PROBE_CANDIDATES) == {"fs_dropout00", "fs_dropout20", "bace_dropout20", "bace_wd1e2"}
    recipe, record = T.resolve_recipe("FreeSolv", "fs_dropout00")
    assert recipe["dropout"] == 0.0
    assert record["effective_differences_vs_R3"] == {"dropout": {"R3": 0.1, "probe": 0.0}}
    recipe, record = T.resolve_recipe("BACE", "bace_wd1e2")
    assert recipe["weight_decay"] == 1e-2
    assert set(record["effective_differences_vs_R3"]) == {"weight_decay"}


def test_probe_guard_rejects_wrong_dataset_and_unknown_names():
    import mb2_train as T

    with pytest.raises(SystemExit):
        T.resolve_recipe("BACE", "fs_dropout00")
    with pytest.raises(SystemExit):
        T.resolve_recipe("FreeSolv", "fs_dropout05")
    with pytest.raises(ValueError):
        C.recipe_for({"peak_lr": 1e-3})
