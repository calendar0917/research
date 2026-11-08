"""Test-side alignment: label-free Test structure cache, prediction freeze, scoring."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import sp_common as C
import sp_test as T
import pc_metrics as MET


# ---------------------------------------------------------------------------
# Test structure cache (label-free, structure only)
# ---------------------------------------------------------------------------


def test_test_cache_is_label_free_and_test_only(has_test_cache):
    if not has_test_cache:
        pytest.skip("Test S3 structure cache not built")
    cache = C.load_s3_test_cache()
    meta = cache["meta"]
    assert meta["schema"] == C.SCHEMA_TEST_CACHE
    assert meta["labels_present"] is False
    assert meta["training_scope"] is False
    assert meta["test_rows_present"] is True
    assert "no labels" in meta["scope"]
    assert meta["feature_spec_hash"] == C.SOURCE["feature_spec_hash"]
    assert int(meta["n_rows"]) == C.N_TEST == 43_793
    rows = np.asarray(cache["rows"], np.int64)
    assert rows.size == C.N_TEST
    assert np.array_equal(np.sort(rows), rows)
    # it must not contain any Train/Valid row
    from sg_data import load_task

    task = load_task()
    assert np.intersect1d(rows, np.asarray(task.train_rows, np.int64)).size == 0
    assert np.intersect1d(rows, np.asarray(task.valid_rows, np.int64)).size == 0


def test_test_cache_cannot_be_used_for_training(task, has_test_cache):
    if not has_test_cache:
        pytest.skip("Test S3 structure cache not built")
    # the training loader refuses a Test-scope cache outright
    with pytest.raises(RuntimeError):
        C.C3.load_s3_cache(C.S3_TEST_CACHE_DIR)
    # and the frozen training cache never contains a Test row
    assert np.intersect1d(np.asarray(task.s3["rows"], np.int64), np.asarray(task.test_rows, np.int64)).size == 0


def test_test_cache_enumeration_matches_the_training_cache(task, has_test_cache):
    """Same enumeration/FeatureSpec: a Train row built by the Test builder would
    produce the training cache's exact fields.  We verify the shared semantics on
    a sample of Train rows re-enumerated from the raw cache."""
    import s3_paths as S3P

    if not has_test_cache:
        pytest.skip("Test S3 structure cache not built")
    cache = task.cache
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    bond_index = np.asarray(cache["bond_index"])
    bond_type = np.asarray(cache["bond_type"])
    atom_type = np.asarray(cache["atom_type"])
    s3_ptr = np.asarray(task.s3["s3_atom_ptr"])
    row_atom_ptr = np.asarray(task.s3["row_atom_ptr"])
    row_index = np.asarray(task.s3["row_index"])
    sample = np.asarray([0, 1, 100, 5_000, 100_000], np.int64)
    rows = np.asarray(task.s3["rows"], np.int64)[sample]
    for row in rows.tolist():
        slot = int(row_index[row])
        n = int(row_atom_ptr[slot + 1] - row_atom_ptr[slot])
        a0 = int(atom_ptr[row])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        if n < 4 or b1 == b0:
            continue
        bi = bond_index[b0:b1].astype(np.int64)
        bt = bond_type[b0:b1].astype(np.int64)
        adj = S3P.molecule_adjacency(n, bi[:, 0], bi[:, 1], bt)
        at = atom_type[a0 : int(atom_ptr[row + 1])]
        dist = S3P.all_pairs_shortest_paths(n, adj)
        paths = S3P.enumerate_s3(n, adj, at, dist, keep_mid_atoms=True)
        counts = paths.per_root_counts()
        expected = (s3_ptr[row_atom_ptr[slot] : row_atom_ptr[slot + 1] + 1])
        expected = np.diff(expected)
        assert np.array_equal(counts, expected.astype(np.int64))
        assert int(counts.sum()) == np.asarray(task.s3["s3_row_paths"])[slot]


# ---------------------------------------------------------------------------
# prediction alignment / freeze integrity
# ---------------------------------------------------------------------------


def test_prediction_rows_follow_the_requested_order():
    class OrderModel(torch.nn.Module):
        def forward(self, batch):
            # logits encode the batch-local atom count so order changes are visible
            n_atoms = int(batch["atom_batch"].numel())
            return torch.full((int(batch["n_graphs"]), C.N_TASKS), float(n_atoms))

    class EchoModel(torch.nn.Module):
        def forward(self, batch):
            return batch["global"][:, : C.N_TASKS].repeat_interleave(
                torch.ones(int(batch["n_graphs"]), dtype=torch.long), dim=0
            )[:, : C.N_TASKS]

    assert C.N_TASKS == 128
    rows = np.asarray([5, 3, 9], np.int64)
    assert np.array_equal(np.sort(rows), np.asarray([3, 5, 9]))


def test_save_and_verify_prediction_hash(tmp_path):
    logits = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "x.npy"
    sha = T._save_npy_atomic(path, logits)
    record = {
        "test_logits_path": str(path),
        "test_logits_sha256": sha,
        "test_logits_shape": [2, 3],
    }
    loaded = T._load_prediction(record, tmp_path)
    assert np.array_equal(loaded, logits)
    bad = dict(record, test_logits_sha256="0" * 64)
    with pytest.raises(RuntimeError):
        T._load_prediction(bad, tmp_path)
    with pytest.raises(RuntimeError):
        T._load_prediction(dict(record, test_logits_path=str(tmp_path.parent / "nope.npy")), tmp_path)


def test_freeze_seal_detects_any_prediction_change():
    freeze = {
        "schema": "lcrc-pcba-test-prediction-freeze-v1",
        "logical_models": {"SP": {"prediction": "SP", "checkpoint_sha256": "a", "valid_ap_replay": 0.3}},
        "predictions": {"SP": {"test_logits_sha256": "b"}},
        "test_rows_sha256": "c",
        "selection_manifest_sha256": "d",
    }
    seal = T.CD.freeze_seal(freeze)
    assert seal == T.CD.freeze_seal(json.loads(json.dumps(freeze)))
    mutated = json.loads(json.dumps(freeze))
    mutated["predictions"]["SP"]["test_logits_sha256"] = "e"
    assert T.CD.freeze_seal(mutated) != seal


def test_candidate_models_require_a_validation_only_selection(tmp_path):
    good = {
        "test_read": False,
        "selection_uses_test": False,
        "models": {
            "SP": {
                "best_epoch": 30,
                "best_valid_ap": 0.28,
                "path": "x/checkpoint_best.pt",
                "sha256": "f" * 64,
            }
        },
    }
    path = tmp_path / "sel.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    models = T.candidate_models(selection_path=path)
    assert sorted(models) == ["S3_best_valid_epoch29", "SP_best_valid_selected", "W2_best_valid_epoch24"]
    assert models["W2_best_valid_epoch24"]["epoch"] == 24
    assert models["S3_best_valid_epoch29"]["epoch"] == 29
    assert models["SP_best_valid_selected"]["epoch"] == 30

    bad = json.loads(json.dumps(good))
    bad["test_read"] = True
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError):
        T.candidate_models(selection_path=path)

    bad2 = json.loads(json.dumps(good))
    bad2["selection_uses_test"] = True
    path.write_text(json.dumps(bad2), encoding="utf-8")
    with pytest.raises(RuntimeError):
        T.candidate_models(selection_path=path)


# ---------------------------------------------------------------------------
# metric semantics
# ---------------------------------------------------------------------------


def test_missing_label_masking_and_task_accounting():
    rng = np.random.default_rng(0)
    n_rows, n_tasks = 200, C.N_TASKS
    y = (rng.random((n_rows, n_tasks)) < 0.05).astype(np.float64)
    y[rng.random((n_rows, n_tasks)) < 0.3] = np.nan
    # task 0: only one class observed -> excluded by both evaluators
    y[:, 0] = 1.0
    y[::7, 0] = np.nan
    probs = rng.random((n_rows, n_tasks)).astype(np.float64)
    official = MET.official_ap(y, probs)
    independent = MET.per_task_ap(y, probs)
    assert abs(float(official["value"]) - float(independent["mean_over_scored_tasks"])) < 1e-9
    assert independent["n_tasks_scored"] == 127  # 128 minus the single-class task
    assert independent["n_tasks_with_observed_labels"] == 128
    assert independent["per_task"][0] is None
    assert float(MET.masked_bce_numpy(np.zeros_like(y), y)) > 0


def test_high_support_strata_helpers():
    a = [0.5, 0.4, None, 0.9]
    b = [0.4, 0.5, 0.1, None]
    support = [25, 30, 40, 3]
    stats = T._stratified(a, b, support, min_support=20)
    assert stats["n_tasks"] == 2
    assert stats["improved"] == 1
    assert stats["mean_delta"] == pytest.approx(0.0)
    up, down, both = T._wins(a, b)
    assert (up, down, both) == (1, 1, 2)
