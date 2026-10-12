"""Tests for cscl-bjoc-full10k-pretraining-transfer-v1.

Fast, data-free tests for the driver's frozen semantics; result-dependent
checks skip when the git-ignored result artifacts are absent.  Official valid
and official test are never referenced or loaded here.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import run_bjoc_full10k_pretraining_transfer_v1 as drv  # noqa: E402
import run_bjoc_masked_pretraining_v1 as mpr  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bjoc_full10k_pretraining_transfer_v1.yaml"
RESULTS = ROOT / "results" / "bjoc_full10k_pretraining_transfer_v1"


def _tiny_molecule(atom_types, edges, mol_id=0):
    x = np.asarray(atom_types, dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in edges:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(mol_id, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))


# ---------------------------------------------------------------------------
# frozen constants / config
# ---------------------------------------------------------------------------


def test_frozen_constants():
    assert drv.N_TRAIN == 10000 and drv.N_VALID == 1000
    assert drv.STEPS_PER_EPOCH == 79 and drv.UPDATES == 18960
    assert drv.TRAIN_SHUFFLE_OFFSET == 101
    assert mp.PRETRAIN_UPDATES == 7560 and mp.ATOM_MASK_TOKEN == 27 and mp.BOND_MASK_TOKEN == 0
    assert drv.ENCODER_LR_Q == pytest.approx(3e-4)
    assert drv.Q_CKA_THRESHOLD == 0.75 and drv.Q_RANK_DROP_THRESHOLD == 0.25


def test_config_frozen_block_matches_driver():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frozen = cfg["frozen"]
    assert frozen["n_train"] == drv.N_TRAIN and frozen["n_valid"] == drv.N_VALID
    assert frozen["pretrain_updates"] == mp.PRETRAIN_UPDATES
    assert frozen["updates"] == drv.UPDATES
    assert frozen["train_shuffle_offset"] == drv.TRAIN_SHUFFLE_OFFSET
    assert float(frozen["encoder_lr_q"]) == pytest.approx(drv.ENCODER_LR_Q)
    assert float(frozen["q_cka_threshold"]) == pytest.approx(drv.Q_CKA_THRESHOLD)


# ---------------------------------------------------------------------------
# batch order == the exact historical full-10k recipe
# ---------------------------------------------------------------------------


def test_batch_order_hash_matches_frozen_masked_pretraining_driver():
    for seed in (0, 1):
        assert drv.batch_order_hash(seed, 10000, 128, 240) == mpr.batch_order_hash(seed, 10000, 128, 240)


def test_shuffle_batches_are_full_permutations_and_deterministic():
    gen1 = torch.Generator().manual_seed(7)
    gen2 = torch.Generator().manual_seed(7)
    b1 = drv._shuffle_batches(10000, 128, gen1)
    b2 = drv._shuffle_batches(10000, 128, gen2)
    assert b1 == b2
    flat = [i for batch in b1 for i in batch]
    assert sorted(flat) == list(range(10000))
    assert len(b1) == 79


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------


def test_row_metrics_and_paired_bootstrap():
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    pred = np.array([0.0, 2.0, 1.0, 6.0], dtype=np.float64)
    m = drv.row_metrics(pred, y)
    assert m["mae"] == pytest.approx((0 + 1 + 1 + 3) / 4)
    assert m["rmse"] == pytest.approx(np.sqrt((0 + 1 + 1 + 9) / 4))
    assert m["max_abs"] == pytest.approx(3.0)
    delta = np.array([-0.1, -0.2, 0.3, -0.4])
    boot1 = drv.paired_bootstrap(delta, n=200, seed=0)
    boot2 = drv.paired_bootstrap(delta, n=200, seed=0)
    assert boot1 == boot2
    assert boot1["mean"] == pytest.approx(delta.mean())
    assert boot1["frac_better"] == pytest.approx(0.75)


def test_top_k_removed_uses_worst_of_either_arm():
    delta = np.array([0.0, 0.0, 1.0])
    e_new = np.array([1.0, 1.0, 5.0])
    e_ref = np.array([1.0, 1.0, 1.0])
    out = drv.top_k_removed(delta, e_new, e_ref, k=1)
    assert out["delta_top1_removed"] == pytest.approx(0.0)
    assert out["worst_rows_top1"] == [2]


def test_effective_rank_and_linear_cka():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 10))
    assert drv._linear_cka(x, x) == pytest.approx(1.0, abs=1e-9)
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    rotated = x @ q
    assert drv._linear_cka(x, rotated) == pytest.approx(1.0, abs=1e-9)
    rank1 = np.outer(rng.normal(size=100), rng.normal(size=4))
    assert drv._effective_rank(rank1) == pytest.approx(1.0, abs=1e-6)


def test_last_soup_is_parameter_mean():
    states = [{"p": torch.full((2,), float(i))} for i in range(5)]
    members, soup = drv._last_soup(states, 5)
    assert members == [1, 2, 3, 4, 5]
    assert torch.allclose(soup["p"], torch.tensor([2.0, 2.0]))


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def test_model_init_is_seed_deterministic_and_parameter_count():
    norm = {"global_mean": np.zeros(sro.GLOBAL_DIM), "global_scale": np.ones(sro.GLOBAL_DIM)}
    m1 = drv._make_downstream_model(0, norm, "cpu")
    m2 = drv._make_downstream_model(0, norm, "cpu")
    m3 = drv._make_downstream_model(1, norm, "cpu")
    assert src.state_dict_hash(m1) == src.state_dict_hash(m2)
    assert src.state_dict_hash(m1) != src.state_dict_hash(m3)
    assert sum(p.numel() for p in m1.parameters()) == 413505


def test_encoder_group_hash_tracks_only_encoders():
    norm = {"global_mean": np.zeros(sro.GLOBAL_DIM), "global_scale": np.ones(sro.GLOBAL_DIM)}
    m = drv._make_downstream_model(0, norm, "cpu")
    h0 = drv._encoder_group_hash(m)
    with torch.no_grad():
        next(m.reader.parameters()).add_(1.0)
    assert drv._encoder_group_hash(m) == h0
    with torch.no_grad():
        next(m.atom_encoder.parameters()).add_(1.0)
    assert drv._encoder_group_hash(m) != h0


# ---------------------------------------------------------------------------
# discipline: the pretraining function must not read labels
# ---------------------------------------------------------------------------


def test_pretrain_function_has_no_label_or_cache_access():
    scan = mpr._scan_source_for_y_reads(Path(drv.__file__), "pretrain_full10k")
    assert scan["found"] is False, scan


def test_pretrain_function_uses_only_structure_features():
    tree = ast.parse(Path(drv.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "pretrain_full10k")
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "y" not in attrs
    assert "edge_index" in attrs and "edge_attr" in attrs and "x" in attrs


# ---------------------------------------------------------------------------
# contrast arithmetic on the extreme-row decomposition
# ---------------------------------------------------------------------------


def test_contrast_extreme_row_conservation():
    y = np.array([0.0, 0.0, -20.0, 1.0])
    preds = {"P0": np.array([0.1, 0.1, -10.0, 1.1]), "P1": np.array([0.2, 0.1, -10.0, 1.1]),
             "B0": np.array([0.0, 0.1, 0.0, 1.0]), "B1": np.array([0.0, 0.1, 0.0, 1.0])}
    original = drv.EXTREME_VALID_ROW
    drv.EXTREME_VALID_ROW = 2
    try:
        out = drv._contrast(["P0", "P1"], ["B0", "B1"], y, preds)
    finally:
        drv.EXTREME_VALID_ROW = original
    assert out["extreme_row"]["delta_row"] == pytest.approx(10.0 - 20.0)
    expected_without = np.mean([0.15 - 0.0, 0.1 - 0.1, 0.1 - 0.0])
    assert out["extreme_row"]["delta_without_row"] == pytest.approx(expected_without)


def test_compare_objects_detects_exact_and_perturbed():
    obj = _tiny_molecule([0, 1, 2], [(0, 1, 1), (1, 2, 2)])
    other = _tiny_molecule([0, 1, 2], [(0, 1, 1), (1, 2, 2)])
    cmp_exact = drv._compare_objects([obj], [other])
    assert cmp_exact["exact"] is True
    other.global_feat = other.global_feat + 1.0
    cmp_diff = drv._compare_objects([obj], [other])
    assert cmp_diff["exact"] is False


# ---------------------------------------------------------------------------
# result-dependent checks (skip when artifacts are absent)
# ---------------------------------------------------------------------------


def _load(name: str):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage_a_audits_if_present():
    if not (RESULTS / "consistency_audit.json").exists():
        pytest.skip("no Phase-A artifacts")
    assert _load("consistency_audit.json")["all_pass"] is True
    assert _load("correctness_gates.json")["all_pass"] is True
    dep = _load("masked_feature_dependency_audit.json")
    assert dep["verdict"] == "MASK_TOKEN_SLOTS_FREE"
    assert dep["mask_token_scan_train10k"]["n_rows_scanned"] == 10000


def test_reuse_audit_if_present():
    if not (RESULTS / "reuse_audit.json").exists():
        pytest.skip("no reuse audit")
    audit = _load("reuse_audit.json")
    if audit["decision"] == "REUSE_HISTORICAL_FULL10K_B":
        assert all(audit["conditions"].values())
        for seed in ("0", "1"):
            assert audit["per_seed"][seed]["valid_replay"]["max_abs_pred_diff"] < 1e-4
    else:
        assert audit["decision"] == "RETRAIN_B"


def test_prediction_freeze_predates_valid_read_if_present():
    freeze_path = RESULTS / "prediction_freeze.json"
    lock_path = RESULTS / "valid_read.lock"
    if not freeze_path.exists():
        pytest.skip("no prediction freeze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze["frozen_before_valid_read"] is True
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        assert lock["prediction_freeze_sha256"] == drv.file_sha256(freeze_path)


def test_scientific_decision_vocabulary_if_present():
    if not (RESULTS / "scientific_decision.json").exists():
        pytest.skip("no decision")
    decision = _load("scientific_decision.json")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert decision["branch"] in cfg["decision_vocabulary"]
    assert decision["official_test_loaded"] is False
