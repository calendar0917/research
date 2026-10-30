from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import v2_common as C
import v2_training as T


def _job(tmp_path: Path, arm: str, **kwargs) -> T.TrainJob:
    base = dict(
        key=f"smoke_{arm}",
        arm=arm,
        fold=0,
        recipe="R0",
        seed=0,
        device="cpu",
        out_root=tmp_path,
        smoke_train_limit=64,
        smoke_epochs=2,
    )
    base.update(kwargs)
    return T.TrainJob(**base)


def test_cpu_smoke_for_all_arms_uses_temp_dir(tmp_path):
    for arm in ("A", "E", "C"):
        payload = T.train_job(_job(tmp_path, arm))
        assert payload["key"] == f"smoke_{arm}"
        assert payload["arm"] == arm
        assert payload["fold"] == 0
        assert payload["recipe"] == "R0"
        assert payload["metrics"]["dev_is_train"] is False
        assert np.isfinite(payload["metrics"]["train_rmse_raw"])
        assert np.isfinite(payload["metrics"]["dev_rmse_raw"])
        assert payload["anomalies"]["nonfinite_events"] == 0
        assert payload["monitor_labels_read"] is False
        assert payload["reserved_labels_read"] is False
        assert payload["official_valid_test_read"] is False
        out = tmp_path / f"smoke_{arm}"
        assert (out / "soup.pt").exists()
        assert (out / "train.json").exists()
        assert (out / "predictions.npz").exists()
        data = np.load(out / "predictions.npz")
        assert data["pred"].shape == (784,)
        assert len(data["rows"]) == 784
        # The prediction file stores cache positions in fold order; those are
        # the same positions used to index the Fit label vector.
        import v2_folds as F

        expected_positions = F.fold_positions(F.load_folds(), 0)
        assert [int(x) for x in data["rows"].tolist()] == expected_positions
        fit = C.load_fit_labels()
        y_all = np.asarray(fit["y"], dtype=np.float64)
        assert np.array_equal(y_all[np.asarray(expected_positions)], y_all[np.asarray(data["rows"], dtype=np.int64)])


def test_training_is_deterministic_under_same_seed(tmp_path):
    p1 = T.train_job(_job(tmp_path / "a", "A", key="a"))
    p2 = T.train_job(_job(tmp_path / "b", "A", key="b"))
    assert p1["model"]["soup_state_hash"] == p2["model"]["soup_state_hash"]
    assert p1["metrics"]["dev_rmse_raw"] == p2["metrics"]["dev_rmse_raw"]
    assert p1["metrics"]["train_rmse_raw"] == p2["metrics"]["train_rmse_raw"]


def test_different_seed_changes_initialisation_and_result(tmp_path):
    p0 = T.train_job(_job(tmp_path / "s0", "A", key="s0", seed=0))
    p1 = T.train_job(_job(tmp_path / "s1", "A", key="s1", seed=1))
    assert p0["model"]["init_state_hash"] != p1["model"]["init_state_hash"]
    assert p0["model"]["soup_state_hash"] != p1["model"]["soup_state_hash"]


def test_formal_recipe_cannot_be_overridden():
    with pytest.raises(ValueError):
        T.TrainJob(key="x", arm="A", fold=0, recipe="R9", seed=0, device="cpu").validate()
    with pytest.raises(ValueError):
        T.TrainJob(key="x", arm="A", fold=0, recipe="R0", seed=3, device="cpu").validate()
    with pytest.raises(ValueError):
        T.TrainJob(key="x", arm="A", fold=3, recipe="R0", seed=0, device="cpu").validate()
    with pytest.raises(ValueError):
        T.TrainJob(key="x", arm="A", fold=0, recipe="R0", seed=0, variant="ssl", device="cpu").validate()
    with pytest.raises(ValueError):
        T.TrainJob(key="x", arm="E", fold=0, recipe="R0", seed=0, variant="ssl", ssl_checkpoint="x", device="cpu").validate()


def test_ssl_loader_loads_only_atom_bond_local(tmp_path, spec, norm):
    model = T.M.build_model("A", spec, norm, 0, "cpu")[0]
    f_rho_before = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("f.", "rho."))}
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # A corrupted f/rho value must be ignored, not loaded.
    for key in state:
        if key.startswith(("f.", "rho.")):
            state[key] = state[key] + 1.0
    ckpt = tmp_path / "ssl.pt"
    torch.save({"module_state": state}, ckpt)
    info = T.load_ssl_encoder_state(model, ckpt)
    assert info["f_rho_from_scratch"] is True
    assert all(k.startswith(("atom_encoder.", "bond_encoder.", "local.")) for k in info["loaded_keys"])
    for key, value in f_rho_before.items():
        assert torch.equal(value, model.state_dict()[key])
    # A shape mismatch in an allowed key is rejected.
    bad = dict(state)
    bad["local.mlp.lins.0.weight"] = torch.zeros(1, 1)
    torch.save({"module_state": bad}, ckpt)
    with pytest.raises(ValueError):
        T.load_ssl_encoder_state(model, ckpt)
