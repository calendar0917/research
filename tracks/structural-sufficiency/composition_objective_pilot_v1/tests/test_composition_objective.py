"""Engineering tests for the composition-objective pilot.

These tests are deliberately small: they pin the model/loss semantics and the
two pair-delta definitions without duplicating the full audit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import cpo_common as C  # noqa: E402
import cpo_train as T  # noqa: E402


RUNS = {
    "A": ("parent", "FACTOR", C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0"),
    "B": ("csoi", "CSOI_ADD", C.CSOI_ROUND / "results" / "CSOI_ADD_LIPO_R0"),
    "C": ("csoi", "CSOI_GLOBAL", C.CSOI_ROUND / "results" / "CSOI_GLOBAL_LIPO_R0"),
}


@pytest.fixture(scope="module")
def loaded():
    cfg = C.load_config()
    pack = C.load_pack(cfg)
    return cfg, pack


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_checkpoint_reproduces_stored_valid_predictions(loaded, name):
    cfg, pack = loaded
    family, arm, path = RUNS[name]
    model, ckpt = C.load_checkpoint(family, cfg, pack, arm, path)
    adapter = C.CutAdapter(model, family, pack)
    adapter.prepare()
    rows = np.asarray(pack["valid_idx"][:8], np.int64)
    res = adapter.cuts(rows)
    pred = res["f_bar"].double().cpu().numpy() * ckpt["scaler"]["std"] + ckpt["scaler"]["mean"]
    old = C.load_predictions(path)["valid_pred"][:8]
    assert np.max(np.abs(pred - old)) < 1e-4


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_loss_decomposition_identity(loaded, name):
    cfg, pack = loaded
    family, arm, path = RUNS[name]
    model, ckpt = C.load_checkpoint(family, cfg, pack, arm, path)
    adapter = C.CutAdapter(model, family, pack)
    adapter.prepare()
    rows = np.asarray(pack["valid_idx"][:24], np.int64)
    y_std = (np.asarray(pack["labels"], np.float64) - ckpt["scaler"]["mean"]) / ckpt["scaler"]["std"]
    res = C.evaluate_cut_rows(adapter, pack, rows, y_std, ckpt["scaler"], batch=8)
    assert res["identity_max_abs"] < 1e-10
    per = res["per_molecule"]
    manual = np.max(np.abs(per["L_cut"] - per["L_graph"] - per["Var_cut"]))
    assert manual < 1e-10
    assert np.allclose(per["L_cut"], per["L_graph"] + per["Var_cut"], atol=1e-10, rtol=0)


def test_graph_loss_expression_is_manual_mean(loaded):
    import torch

    cfg, pack = loaded
    family, arm, path = RUNS["C"]
    model, ckpt = C.load_checkpoint(family, cfg, pack, arm, path)
    rows = np.asarray(pack["valid_idx"][:4], np.int64)
    y = torch.as_tensor(((np.asarray(pack["labels"][rows], np.float64) - ckpt["scaler"]["mean"])
                         / ckpt["scaler"]["std"]), dtype=torch.float32)
    y_all = torch.as_tensor(((np.asarray(pack["labels"], np.float64) - ckpt["scaler"]["mean"])
                             / ckpt["scaler"]["std"]), dtype=torch.float32)
    with torch.no_grad():
        res = model.molecule_forward(torch.as_tensor(rows))
        f_bar = res["f_bar"]
        loss = torch.mean((f_bar[:4] - y_all[rows]) ** 2)
        manual = torch.mean((f_bar[:4] - y) ** 2)
    assert torch.allclose(loss, manual)


def test_pair_delta_definitions_are_distinct(loaded):
    cfg, pack = loaded
    family, arm, path = RUNS["A"]
    model, ckpt = C.load_checkpoint(family, cfg, pack, arm, path)
    adapter = C.CutAdapter(model, family, pack)
    adapter.prepare()
    labels = np.asarray(pack["labels"], np.float64)
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    split = np.asarray(pack["molecule_split"], np.uint8)
    mask = (split[a] == 0) & (split[b] == 0)
    idx = np.nonzero(mask)[0][:128]
    direct = adapter.direct_pair_delta(idx) * float(ckpt["scaler"]["std"])
    preds = C.load_predictions(path)
    pred = C.predictions_by_molecule(pack, preds)
    whole = pred[b[idx]] - pred[a[idx]]
    assert direct.size == whole.size == idx.size
    # They are related but not the same code path / quantity.
    assert np.mean(np.abs(direct - whole)) > 0.0
    assert C.pair_delta_stats(labels[b[idx]] - labels[a[idx]], direct)["n"] == idx.size
    assert C.pair_delta_stats(labels[b[idx]] - labels[a[idx]], whole)["n"] == idx.size


def test_run_data_exposes_only_train_valid(loaded):
    cfg, pack = loaded
    data = T.RunData(cfg, "lipo", log=lambda *_: None, train_subset=8, valid_subset=4)
    assert set(data.splits) == {"train", "valid"}
    assert data.splits["train"].size == 8
    assert data.splits["valid"].size == 4
    # no test row id is exposed through the split view
    assert not np.any(np.isin(data.row_ids[data.splits["train"]],
                              data.row_ids[np.asarray(pack["test_idx"], np.int64)]))


@pytest.mark.parametrize("family,arm", [("parent", "FACTOR"), ("csoi", "CSOI_ADD"),
                                        ("csoi", "CSOI_GLOBAL")])
def test_cpu_trainer_smoke(loaded, tmp_path, family, arm):
    cfg, pack = loaded
    run_id = str(tmp_path / f"smoke_{family}_{arm}")
    rc = T.main(["--run-id", run_id, "--family", family, "--arm", arm,
                 "--epochs", "1", "--device", "cpu", "--seed", "0",
                 "--train-subset", "16", "--valid-subset", "8", "--max-steps", "2"])
    assert rc == 0
    rdir = Path(run_id)
    assert (rdir / "curves.csv").is_file()
    assert (rdir / "state_best.pt").is_file()
    assert (rdir / "predictions_best.npz").is_file()
    manifest = C.read_json(rdir / "manifest.json")
    assert manifest["main_loss"] == "graph_level_mean_squared_error"
    assert manifest["test_policy"].startswith("no test")
    with np.load(rdir / "predictions_best.npz", allow_pickle=False) as npz:
        assert not any(k.startswith("test") for k in npz.files)