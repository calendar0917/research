"""Two-epoch CPU training smoke, artifact integrity and metric recomputation."""

from __future__ import annotations

import numpy as np
import torch

import x1_common as C
import x1_data as XD
import x1_train as T


def _train(arm, cfg, split, labels, run_dir):
    return T.train_arm(
        arm=arm,
        cfg=cfg,
        split=split,
        labels=labels,
        device=torch.device("cpu"),
        run_dir=run_dir,
        limit_train=64,
        limit_valid=32,
        max_epochs=2,
        soup_last=2,
        pool="cpu-test",
        smoke=True,
    )


def test_two_arm_cpu_training_smoke(tmp_path, cfg, frozen_data):
    split, labels, _views = frozen_data
    payloads = {}
    for arm in C.ARMS:
        payloads[arm] = _train(arm, cfg, split, labels, tmp_path / arm)
    for arm, payload in payloads.items():
        assert payload["arm"] == arm
        assert payload["recipe"]["epochs_run"] == 2
        assert payload["recipe"]["soup_members"] == [1, 2]
        assert payload["recipe"]["optimizer_steps"] == 2 * 1
        assert payload["recipe"]["early_stopping"] is False
        assert payload["recipe"]["dropout"] == 0.10
        assert payload["model_audit"]["dropout_hook_calls"] >= 2
        assert np.isfinite(payload["metrics"]["train"]["rmse"])
        assert np.isfinite(payload["metrics"]["valid"]["rmse"])
        assert payload["data"]["test_labels_read"] is False
        checkpoint_hash = C.sha256_file(tmp_path / arm / "checkpoint.pt")
        assert checkpoint_hash == payload["checkpoint_sha256"]
        valid = np.load(tmp_path / arm / "valid_predictions.npz")
        assert np.array_equal(valid["rows"], np.asarray(split["rows"]["valid"][:32], dtype=np.int64))
        recomputed = float(np.sqrt(np.mean((valid["pred"] - valid["y_true"]) ** 2)))
        assert abs(recomputed - payload["metrics"]["valid"]["rmse"]) <= 1.0e-12
        train = np.load(tmp_path / arm / "train_predictions.npz")
        assert np.array_equal(train["rows"], np.asarray(split["rows"]["train"][:64], dtype=np.int64))


def test_context_run_records_path_usage(tmp_path, cfg, frozen_data):
    split, labels, _views = frozen_data
    payload = _train("context", cfg, split, labels, tmp_path / "context")
    usage = payload["diagnostics"]["context_usage"]
    assert payload["model_audit"]["parameters_new"] == C.CONTEXT_NEW_PARAMS_REFERENCE
    assert usage["valid420"]["n_hetero_atoms"] > 0
    assert usage["valid420"]["change_l2_mean_over_hetero_atoms"] >= 0.0
    assert usage["valid420"]["f_delta_abs_mean_over_hetero_atoms"] >= 0.0
    assert usage["valid420"]["f_delta_abs_max_over_hetero_atoms"] >= 0.0
    assert usage["gradient_probe"]["all_new_tensors_received_nonzero_gradient"]
    assert payload["diagnostics"]["rho_after_soup"]["passed"]
    assert payload["diagnostics"]["dropout"]["passed"]
    ablation = usage["w_out_zero_ablation_valid"]
    assert np.isfinite(ablation["rmse"])


def test_base_run_has_no_context_diagnostics(tmp_path, cfg, frozen_data):
    split, labels, _views = frozen_data
    payload = _train("base", cfg, split, labels, tmp_path / "base")
    assert payload["diagnostics"] == {}
    assert payload["model_audit"]["parameters_new"] == 0
