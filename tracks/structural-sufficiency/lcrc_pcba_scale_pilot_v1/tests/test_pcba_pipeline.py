"""CPU-only unit tests for the PCBA scale pilot (no OGB download, no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import json

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import pc_common as C  # noqa: E402
import pc_data as DATA  # noqa: E402
import pc_metrics as MET  # noqa: E402


def test_frozen_inheritance_is_asserted():
    from mb2_data import frozen_spec

    info = C.assert_frozen_inheritance()
    assert info["model_config_equal"]
    spec = frozen_spec()
    assert spec.atom_feat_dim == 51 and spec.global_dim == 42 and spec.n_scalars == 9
    assert spec.n_nbr_codes == 80 and spec.n_path_codes == 400
    assert C.RECIPE["batch_size"] == 128 and C.RECIPE["peak_lr"] == 3e-4
    assert C.RECIPE["max_epochs"] == 40 and C.RECIPE["min_epochs"] == 12 and C.RECIPE["patience"] == 10


def test_masked_bce_excludes_nan_and_all_nan_is_zero():
    import torch

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(12, 6)).astype(np.float32) * 2
    y = (rng.random((12, 6)) < 0.5).astype(np.float32)
    y[0, :] = np.nan
    y[2, 3] = np.nan
    t_logits = torch.as_tensor(logits)
    t_y = torch.as_tensor(y)
    got = float(MET.masked_bce_loss(t_logits, t_y))
    want = MET.masked_bce_numpy(logits, y)
    assert abs(got - want) <= 1e-6
    # perturbing *logits* at unobserved positions must not move the loss
    logits2 = logits.copy()
    logits2[~np.isfinite(y)] = 1e3
    assert float(MET.masked_bce_loss(torch.as_tensor(logits2), t_y)) == got
    # NaN must not be treated as class 0
    as_zero = MET.masked_bce_numpy(logits, np.where(np.isfinite(y), y, 0.0))
    assert abs(got - as_zero) > 1e-9
    all_nan = torch.full((4, 3), float("nan"))
    assert float(MET.masked_bce_loss(torch.zeros((4, 3)), all_nan)) == 0.0


def test_masked_bce_matches_torch_on_observed_only():
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(1)
    logits = rng.normal(size=(8, 5)).astype(np.float32)
    y = (rng.random((8, 5)) < 0.5).astype(np.float32)
    y[1, 0] = np.nan
    mask = np.isfinite(y)
    expected = float(
        F.binary_cross_entropy_with_logits(
            torch.as_tensor(logits)[torch.as_tensor(mask)], torch.as_tensor(y)[torch.as_tensor(mask)]
        )
    )
    assert abs(float(MET.masked_bce_loss(torch.as_tensor(logits), torch.as_tensor(y))) - expected) <= 1e-7


def test_official_ap_matches_independent_recomputation_with_missing_labels():
    rng = np.random.default_rng(2)
    n, tasks = 400, 128
    y = (rng.random((n, tasks)) < 0.3).astype(np.float64)
    score = rng.normal(size=(n, tasks))
    # mask ~40% at random and make two tasks single-class
    y[rng.random((n, tasks)) < 0.4] = np.nan
    y[:, 0] = np.nan
    y[:, 1] = np.nan
    y[:5, 1] = 1.0
    agree = MET.ap_agreement(y, score)
    assert agree["abs_delta"] <= 1e-9, agree
    assert agree["n_tasks_scored"] == 126
    assert agree["n_tasks_with_observed_labels"] < tasks
    # score values at masked positions must not affect the AP
    score2 = score.copy()
    score2[~np.isfinite(y)] = 1e6
    agree2 = MET.ap_agreement(y, score2)
    assert abs(agree2["official_ap"] - agree["official_ap"]) <= 1e-12


def test_task_label_counts():
    y = np.array(
        [
            [1.0, np.nan, 0.0],
            [0.0, 1.0, np.nan],
            [np.nan, 1.0, 1.0],
        ]
    )
    counts = MET.per_task_ap(y, np.array([[0.9, 0.1, 0.2], [0.1, 0.9, 0.3], [0.2, 0.8, 0.7]]))
    assert counts["observed"] == [2, 2, 2]
    assert counts["positive"] == [1, 2, 1]
    assert counts["negative"] == [1, 0, 1]
    assert counts["n_tasks_scored"] == 2


def test_selection_tie_prefers_earlier_epoch():
    import pc_train as T

    tracker = T.Tracker()
    assert tracker.consider(1, 0.30)
    assert tracker.consider(2, 0.40)
    assert not tracker.consider(3, 0.40)
    assert tracker.best_epoch == 2
    assert tracker.tie_check()["consistent"]


def _tiny_cache():
    pytest.importorskip("rdkit")
    from feature_spec import load_spec
    from mb2_data import frozen_spec
    import lcrc_data as PREP_DATA
    import mb2_rdkit as R

    spec = frozen_spec()
    smiles = ["CCO", "c1ccccc1", "Nc1ccc(O)cc1", "CC(=O)Oc1ccccc1C(=O)O"]
    mols = []
    for i, smi in enumerate(smiles):
        raw, _mode = R.smiles_to_raw_row_relaxed(smi, i)
        graph = R.raw_row_to_graph(raw, i, spec)
        mols.append(PREP_DATA.build_molecule(graph, spec, validate=True))
    cache = PREP_DATA.concat_molecules(mols, spec, split="tiny")
    cache["gids"] = np.arange(len(smiles), dtype=np.int64)
    return spec, cache


def test_directory_cache_round_trip(tmp_path):
    """load_cache must memory-map a directory cache without changing values."""
    meta = {
        "n_molecules": 2,
        "split": "all",
        "fields": ["atom_ptr", "atom_type", "global"],
        "feature_spec_hash": "x",
        "labels_present": False,
    }
    np.save(tmp_path / "atom_ptr.npy", np.array([0, 2, 3], dtype=np.int64))
    np.save(tmp_path / "atom_type.npy", np.array([6, 7, 8], dtype=np.int64))
    np.save(tmp_path / "global.npy", np.zeros((2, 4), dtype=np.float32))
    np.save(tmp_path / "gids.npy", np.array([0, 1], dtype=np.int64))
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "molecule_ids.jsonl").write_text('{"row": 0}\n{"row": 1}\n')
    (tmp_path / "smiles.jsonl").write_text('"C"\n"CC"\n')
    cache = DATA.load_cache(tmp_path)
    assert int(cache["n_molecules"]) == 2
    assert isinstance(cache["atom_type"], np.ndarray)
    assert cache["atom_type"].base is not None or isinstance(cache["atom_type"], np.memmap)
    assert np.array_equal(np.asarray(cache["atom_type"]), np.array([6, 7, 8]))
    assert [m["row"] for m in cache["molecule_ids"]] == [0, 1]
    assert [s for s in cache["smiles"]] == ["C", "CC"]
    # a mutated copy must not write through to disk
    view = np.asarray(cache["atom_type"])
    assert not np.shares_memory(view.copy(), view)


def test_cache_label_free_and_s2_identity():
    import pc_data as DATA

    spec, cache = _tiny_cache()
    free = DATA.cache_label_free(cache)
    assert free["ok"] and not free["label_like_keys"]
    ident = DATA.s2_count_identity(cache)
    assert ident["ok"], ident
    assert ident["s1_total_equals_2_bonds"]
    assert ident["s2_total_observed"] > 0
    audit = DATA.cache_structure_audit(cache, spec)
    assert audit["ok"], audit["value_ranges"]


def test_c1_output_shape_and_finite_gradients_128_tasks():
    import torch
    import bp_data as D
    import pc_model as BM

    spec, cache = _tiny_cache()
    rows = np.arange(int(cache["n_molecules"]), dtype=np.int64)
    norm = D.fit_norm_rows(cache, spec, rows, source="unit_test")
    device = torch.device("cpu")
    model, info = BM.build_c1(spec, norm, C.N_TASKS, 0, device)
    assert info["n_outputs"] == C.N_TASKS
    assert "b_mean" not in info["local_layout"]
    batch = BM.collate(cache, rows.tolist())
    y = np.zeros((len(rows), C.N_TASKS), dtype=np.float32)
    y[:, 3] = 1.0
    y[0, :] = np.nan
    logits = model(batch)
    assert tuple(logits.shape) == (len(rows), C.N_TASKS)
    loss = MET.masked_bce_loss(logits, torch.as_tensor(y))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_atom_renumbering_invariance():
    import torch
    import bp_data as D
    import pc_model as BM
    from pc_smoke import _permute_single

    spec, cache = _tiny_cache()
    rows = np.arange(int(cache["n_molecules"]), dtype=np.int64)
    norm = D.fit_norm_rows(cache, spec, rows, source="unit_test")
    device = torch.device("cpu")
    model, _ = BM.build_c1(spec, norm, C.N_TASKS, 0, device)
    model.eval()
    batch = BM.collate(cache, [2])
    n_atoms = int(batch["atom_feat"].shape[0])
    perm = np.random.default_rng(0).permutation(n_atoms)
    permuted = _permute_single(batch, perm)
    with torch.no_grad():
        a = model(batch).numpy()
        b = model(permuted).numpy()
    assert np.max(np.abs(a - b)) <= 1e-5


def test_single_vs_batch_inference():
    import torch
    import bp_data as D
    import pc_model as BM

    spec, cache = _tiny_cache()
    rows = np.arange(int(cache["n_molecules"]), dtype=np.int64)
    norm = D.fit_norm_rows(cache, spec, rows, source="unit_test")
    device = torch.device("cpu")
    model, _ = BM.build_c1(spec, norm, C.N_TASKS, 0, device)
    model.eval()
    batched = BM.predict_logits(model, cache, rows, device, batch_size=8)
    singles = np.stack([BM.predict_logits(model, cache, rows[i : i + 1], device)[0] for i in range(len(rows))])
    assert np.max(np.abs(batched - singles)) <= 1e-5


def test_cache_structure_rejects_non_monotonic_pointer():
    import pc_data as DATA

    spec, cache = _tiny_cache()
    broken = dict(cache)
    broken["path_ptr"] = np.asarray(cache["path_ptr"]).copy()
    broken["path_ptr"][1] = -1
    with pytest.raises(Exception):
        DATA.cache_structure_audit(broken, spec)


def test_s2_identity_detects_shifted_path_counts():
    import pc_data as DATA

    _spec, cache = _tiny_cache()
    broken = dict(cache)
    broken["path_ptr"] = np.asarray(cache["path_ptr"]).copy()
    broken["path_ptr"][1:] += 1
    ident = DATA.s2_count_identity(broken)
    assert not ident["ok"]
    assert ident["n_mismatched_atoms"] > 0
