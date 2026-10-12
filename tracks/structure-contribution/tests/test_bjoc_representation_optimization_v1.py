"""Tests for cscl-bjoc-representation-optimization-v1.

Split into two halves:

* **data-free** tests (typed paths, ring blocks, models, capacity audit,
  schedule, supervision contract) — safe in a fresh clone, CPU only;
* **cache-dependent** tests (frozen splits, standardisers, extended objects)
  which skip when the round's generated cache is absent.  They never read
  official valid/test.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_representation_optimization_v1 as bro  # noqa: E402
import bjoc_typed_path_ring_features_v1 as tpr  # noqa: E402
import run_bjoc_representation_optimization_v1 as drv  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "bjoc_representation_optimization_v1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def mk_graph(n_atoms: int, edges, atom_types=None):
    """(n_atoms, [(u, v, bond_type)]) -> PyG-style (x, edge_index, edge_attr)."""
    x = (
        np.zeros((n_atoms, 1), dtype=np.int64)
        if atom_types is None
        else np.asarray(atom_types, dtype=np.int64).reshape(n_atoms, 1)
    )
    ei, ea = [], []
    for u, v, t in edges:
        ei.extend([(u, v), (v, u)])
        ea.extend([t, t])
    return x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64)


def ring(n: int, start: int = 0, t: int = 1):
    return [(start + i, start + (i + 1) % n, t) for i in range(n)]


def build(n_atoms, edges, atom_types=None):
    return tpr.build_extended_objects(-1, *mk_graph(n_atoms, edges, atom_types))


def model_for(arm: str, seed: int = 0):
    return bro.build_model(
        arm,
        seed=seed,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )


def batch_for(objs, mode: str = "zero"):
    stats = {
        "atom_mean": np.zeros(tpr.NEW_ATOM_DIM),
        "atom_scale": np.ones(tpr.NEW_ATOM_DIM),
        "bond_mean": np.zeros(tpr.NEW_BOND_DIM),
        "bond_scale": np.ones(tpr.NEW_BOND_DIM),
    }
    return drv._batch_from_objects(objs, mode, stats)


# ---------------------------------------------------------------------------
# 1. typed paths
# ---------------------------------------------------------------------------


def test_path_dims_match_layout():
    assert tpr.PATH_ATOM_DIM == 16 + 64 + 3 + 4 + 28 + 2
    assert tpr.NEW_ATOM_DIM == tpr.PATH_ATOM_DIM + tpr.RING_ATOM_DIM
    lay = tpr.atom_layout()
    assert lay["ring_block_bondtype_frac"].stop == tpr.NEW_ATOM_DIM
    blay = tpr.bond_layout()
    assert blay["is_fused_edge"].stop == tpr.NEW_BOND_DIM


def test_path_counts_on_path_graph():
    """A 4-atom chain 0-1-2-3 all single bonds rooted at 0."""
    obj = build(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
    lay = tpr.atom_layout()
    p2 = obj.atom_new[:, lay["path2_seq"]]
    p3 = obj.atom_new[:, lay["path3_seq"]]
    # root 0: one length-2 path (0-1-2) with sequence (single, single) = 1*4+1 = 5
    assert float(p2[0, 5]) == pytest.approx(np.log1p(1))
    # root 0: one length-3 path (0-1-2-3) sequence (1,1,1) = 21
    assert float(p3[0, 1 * 16 + 1 * 4 + 1]) == pytest.approx(np.log1p(1))
    # root 1: two length-2 paths? no: only the direction away from root counts
    totals = obj.atom_new[:, lay["path_totals"]]
    assert float(totals[1, 0]) == pytest.approx(np.log1p(1))  # one length-2 path from atom 1
    assert float(totals[0, 0]) == pytest.approx(np.log1p(1))  # one length-2 path from atom 0


def test_path_ring_count_flags():
    """The cyclohexane length-2 paths all use two ring bonds."""
    obj = build(6, ring(6))
    lay = tpr.atom_layout()
    ring_counts = obj.atom_new[:, lay["path2_ring_count"]]
    assert ring_counts[:, 2].min() == pytest.approx(1.0)  # every path has 2 ring bonds
    assert ring_counts[:, :2].max() == 0.0


def test_typed_paths_distinguish_bond_sequence_from_root():
    """The OLD row for an atom can be identical while its typed 3-paths differ.

    ``0-1-2-3-4`` with the double bond on ``(3,4)`` vs on ``(2,3)``: atom 0 sees the
    same own bonds, the same neighbour type, the same radius-2 *atom types* and the
    same length-2 sequence (1,1), yet the length-3 bond-type sequence is (1,1,1)
    vs (1,1,2).
    """
    a = build(5, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 2)])
    b = build(5, [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1)])
    lay = tpr.atom_layout()
    assert np.allclose(a.base.atom_feat[0], b.base.atom_feat[0])
    assert np.allclose(a.atom_new[0, lay["path2_seq"]], b.atom_new[0, lay["path2_seq"]])
    pa = a.atom_new[0, lay["path3_seq"]]
    pb = b.atom_new[0, lay["path3_seq"]]
    assert float(pa[1 * 16 + 1 * 4 + 1]) == pytest.approx(np.log1p(1))  # (1,1,1)
    assert float(pb[1 * 16 + 1 * 4 + 2]) == pytest.approx(np.log1p(1))  # (1,1,2)
    assert not np.allclose(pa, pb)


def test_typed_paths_are_blind_to_which_branch_carries_the_bond_type():
    """Documented limitation of a rooted path histogram (mechanism caveat).

    Swapping which of two branches carries the double bond leaves the *multiset*
    of length-2 sequences at the branch point unchanged, so a purely local path
    histogram cannot resolve this kind of junction content.
    """
    a = build(5, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (2, 4, 2)])
    b = build(5, [(0, 1, 1), (1, 2, 1), (2, 3, 2), (2, 4, 1)])
    lay = tpr.atom_layout()
    assert np.allclose(a.atom_new[1, lay["path2_seq"]], b.atom_new[1, lay["path2_seq"]])


# ---------------------------------------------------------------------------
# 2. ring blocks
# ---------------------------------------------------------------------------


def test_ring_block_cyclohexane_is_monocyclic():
    obj = build(6, ring(6))
    lay = tpr.atom_layout()
    blay = tpr.bond_layout()
    assert obj.atom_new[0, lay["ring_n_cyclic_blocks"]][1] == pytest.approx(1.0)
    assert float(obj.atom_new[0, lay["ring_in_monocyclic_block"]]) == pytest.approx(1.0)
    assert float(obj.atom_new[0, lay["ring_in_polycyclic_block"]]) == 0.0
    assert float(obj.atom_new[0, lay["ring_max_block_cycle_rank"]]) == pytest.approx(np.log1p(1))
    assert float(obj.atom_new[0, lay["ring_max_block_n_atoms"]]) == pytest.approx(np.log1p(6))
    assert obj.bond_new[:, blay["block_monocyclic"]].min() == pytest.approx(1.0)
    assert obj.bond_new[:, blay["is_fused_edge"]].max() == 0.0
    assert obj.bond_new[:, blay["is_bridge"]].max() == 0.0


def test_ring_block_fused_bicyclic():
    """Two six-rings sharing one edge -> rank 2 block, every edge on >= 2 cycles."""
    edges = ring(6, 0) + [(0, 6, 1), (6, 7, 1), (7, 8, 1), (8, 9, 1), (9, 1, 1)]
    obj = build(10, edges)
    lay = tpr.atom_layout()
    blay = tpr.bond_layout()
    assert float(obj.atom_new[0, lay["ring_in_polycyclic_block"]]) == pytest.approx(1.0)
    assert float(obj.atom_new[0, lay["ring_is_spiro_center"]]) == 0.0
    assert float(obj.atom_new[0, lay["ring_max_block_cycle_rank"]]) == pytest.approx(np.log1p(2))
    assert obj.bond_new[:, blay["is_fused_edge"]].min() == pytest.approx(1.0)
    assert obj.bond_new[:, blay["block_polycyclic"]].min() == pytest.approx(1.0)


def test_ring_block_spiro_center():
    edges = ring(5, 0) + [(0, 5, 1), (5, 6, 1), (6, 7, 1), (7, 8, 1), (8, 0, 1)]
    obj = build(9, edges)
    lay = tpr.atom_layout()
    assert float(obj.atom_new[0, lay["ring_is_spiro_center"]]) == pytest.approx(1.0)
    assert obj.atom_new[0, lay["ring_n_cyclic_blocks"]][2] == pytest.approx(1.0)
    assert float(obj.atom_new[0, lay["ring_in_monocyclic_block"]]) == pytest.approx(1.0)
    assert float(obj.atom_new[0, lay["ring_max_block_attachments"]]) == pytest.approx(np.log1p(1))


def test_ring_block_bridge_edge_is_not_ring():
    obj = build(8, ring(6) + [(0, 6, 1), (6, 7, 1)])
    blay = tpr.bond_layout()
    bridge = obj.bond_new[:, blay["is_bridge"]]
    assert bridge.sum() == pytest.approx(2.0)  # bonds (0,6) and (6,7)
    assert obj.bond_new[:, blay["block_cycle_rank"]][bridge > 0.5].max() == 0.0


def test_ring_block_never_reads_labels():
    src_text = Path(tpr.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src_text)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "y" not in names
    assert "decomp" not in names


# ---------------------------------------------------------------------------
# 3. standardisation / padding
# ---------------------------------------------------------------------------


def test_zero_mode_is_exactly_zero_and_real_mode_standardised():
    objs = [build(6, ring(6)), build(10, ring(6, 0) + [(0, 6, 1), (6, 7, 1), (7, 8, 1), (8, 9, 1), (9, 1, 1)])]
    stats = tpr.standardise_stats(objs, [0, 1])
    z_atoms, z_bonds = tpr.apply_standardisation(objs, stats, mode="zero")
    assert float(np.abs(z_atoms).max()) == 0.0 and float(np.abs(z_bonds).max()) == 0.0
    r_atoms, _ = tpr.apply_standardisation(objs, stats, mode="real")
    # standardised channels over the fitting set have ~zero mean
    assert abs(float(r_atoms.mean())) < 0.05


def test_standardisation_scale_is_fit_only():
    objs = [build(6, ring(6)), build(10, ring(6, 0) + [(0, 6, 1), (6, 7, 1), (7, 8, 1), (8, 9, 1), (9, 1, 1)])]
    s1 = tpr.standardise_stats(objs, [0])
    s2 = tpr.standardise_stats(objs, [0, 1])
    assert not np.allclose(s1["atom"]["mean"], s2["atom"]["mean"])


# ---------------------------------------------------------------------------
# 4. models / capacity
# ---------------------------------------------------------------------------


def test_arms_have_identical_parameter_counts():
    counts = {arm: int(sum(p.numel() for p in model_for(arm).parameters())) for arm in ("B", "T", "J")}
    assert counts["B"] == counts["T"] == counts["J"]
    assert counts["B"] == 432705
    assert bro.ORIGINAL_BJOC_PARAMETERS == 413505


def test_B_T_identical_initialisation_J_differs():
    hb = src.state_dict_hash(model_for("B"))
    ht = src.state_dict_hash(model_for("T"))
    hj = src.state_dict_hash(model_for("J"))
    assert hb == ht
    assert hb != hj


def test_no_forbidden_modules_in_any_arm():
    for arm in ("B", "T", "J"):
        assert bro.forbidden_module_scan(model_for(arm)) == []


def test_junctionless_has_no_junction_parameters_or_state():
    model = model_for("J")
    assert not any("junc" in k for k in model.state_dict())
    assert not any("junction" in n for n, _ in model.named_modules())


def test_junctionless_forward_does_not_read_junction_content():
    """Only the fixed junction COUNT may enter J; junc_feat/junc_index must not."""
    obj = build(6, ring(6))
    batch = batch_for([obj], "zero")
    model = model_for("J").eval()
    with torch.no_grad():
        p1 = model(batch)
        batch2 = dict(batch)
        batch2["junc_feat"] = torch.rand_like(batch["junc_feat"]) * 100.0
        batch2["junc_index"] = torch.zeros_like(batch["junc_index"])
        p2 = model(batch2)
    assert float((p1 - p2).abs().max()) == 0.0


def test_T_with_zero_channels_equals_B_exactly():
    obj = build(10, ring(6, 0) + [(0, 6, 1), (6, 7, 1), (7, 8, 1), (8, 9, 1), (9, 1, 1)])
    batch = batch_for([obj], "zero")
    b = model_for("B").eval()
    t = model_for("T").eval()
    with torch.no_grad():
        assert float((b(batch) - t(batch)).abs().max()) == 0.0


def test_T_with_real_channels_differs_from_B():
    obj = build(10, ring(6, 0) + [(0, 6, 1), (6, 7, 1), (7, 8, 1), (8, 9, 1), (9, 1, 1)])
    real = batch_for([obj], "real")
    zero = batch_for([obj], "zero")
    t = model_for("T").eval()
    with torch.no_grad():
        assert float((t(real) - t(zero)).abs().max()) > 0.0


def test_relabelling_invariance_of_models():
    obj = build(10, ring(6, 0) + [(0, 6, 1), (6, 7, 1), (7, 8, 1), (8, 9, 1), (9, 1, 1)])
    perm_obj = drv._relabel_object(obj, seed=3)
    assert perm_obj.base.n_atoms == obj.base.n_atoms
    for arm, mode in (("B", "zero"), ("T", "real"), ("J", "zero")):
        model = model_for(arm).eval()
        with torch.no_grad():
            p1 = model(batch_for([obj], mode))
            p2 = model(batch_for([perm_obj], mode))
        assert float((p1 - p2).abs().max()) < 1e-5, arm


def test_relabel_helper_preserves_graph_structure():
    """Regression for the inverse-permutation bug found by the correctness gate."""
    obj = build(7, ring(6) + [(0, 6, 1)])
    perm_obj = drv._relabel_object(obj, seed=11)
    old_edges = sorted((int(u), int(v), int(t)) for (u, v), t in zip(obj.base.bond_index, obj.base.bond_type))
    new_edges = sorted((int(u), int(v), int(t)) for (u, v), t in zip(perm_obj.base.bond_index, perm_obj.base.bond_type))
    assert len(old_edges) == len(new_edges)
    old_types = sorted(int(np.argmax(obj.base.atom_feat[i, :28])) for i in range(obj.base.n_atoms))
    new_types = sorted(int(np.argmax(perm_obj.base.atom_feat[i, :28])) for i in range(perm_obj.base.n_atoms))
    assert old_types == new_types


def test_batch_grouping_invariance():
    objs = [build(6, ring(6)), build(9, ring(5, 0) + [(0, 5, 1), (5, 6, 1), (6, 7, 1), (7, 8, 1), (8, 0, 1)])]
    model = model_for("J").eval()
    single = batch_for([objs[0]], "zero")
    pair = drv._batch_from_objects(objs, "zero", {
        "atom_mean": np.zeros(tpr.NEW_ATOM_DIM), "atom_scale": np.ones(tpr.NEW_ATOM_DIM),
        "bond_mean": np.zeros(tpr.NEW_BOND_DIM), "bond_scale": np.ones(tpr.NEW_BOND_DIM),
    })
    with torch.no_grad():
        assert abs(float(model(single)[0] - model(pair)[0])) < 1e-5


def test_junctionless_gradients_reach_every_block():
    obj = build(6, ring(6))
    model = model_for("J")
    report = bro.junctionless_gradient_audit(model, batch_for([obj], "zero"))
    assert report["all_grads_live"]
    assert not report["has_junction_module"]
    assert report["has_junction_state_key"] is False


# ---------------------------------------------------------------------------
# 5. schedules / statistics / discipline
# ---------------------------------------------------------------------------


def test_lr_schedule_is_the_frozen_stabilisation_candidate():
    assert drv.lr_for_epoch(1, False) == 1e-3
    assert drv.lr_for_epoch(240, False) == 1e-3
    assert drv.lr_for_epoch(160, True) == 1e-3
    assert drv.lr_for_epoch(161, True) == 3e-4
    assert drv.lr_for_epoch(200, True) == 3e-4
    assert drv.lr_for_epoch(201, True) == 1e-4
    assert drv.lr_for_epoch(240, True) == 1e-4


def test_feature_mode_mapping():
    assert drv.feature_mode_for("T") == "real"
    assert drv.feature_mode_for("B") == "zero"
    assert drv.feature_mode_for("J") == "zero"
    assert drv.feature_mode_for("S") == "zero"


def test_paired_bootstrap_identifies_a_clear_shift():
    rng = np.random.default_rng(0)
    delta = rng.normal(-0.02, 0.05, size=500)
    out = drv.paired_bootstrap(delta, n=500)
    assert out["ci95"][1] < 0
    assert out["frac_candidate_better"] > 0.6


def test_paired_bootstrap_covers_zero_for_symmetric_noise():
    rng = np.random.default_rng(1)
    out = drv.paired_bootstrap(rng.normal(0.0, 0.1, size=500), n=500)
    assert out["ci95"][0] < 0 < out["ci95"][1]


def test_split_builder_is_label_blind_and_grouped():
    smiles = ["A"] * 2 + ["B", "C", "D", "E"] * 3000
    split = drv.build_split(smiles, "unit-test-salt")
    assert len(split["fit"]) + len(split["monitor"]) + len(split["locked"]) == len(smiles)
    for a, b in (("fit", "monitor"), ("fit", "locked"), ("monitor", "locked")):
        assert not (set(split[a]) & set(split[b]))
    dup_rows = {i for i, s in enumerate(smiles) if s == "A"}
    side = {name for name, key in (("fit", "fit"), ("monitor", "monitor"), ("locked", "locked")) if dup_rows <= set(split[key])}
    assert len(side) == 1, "duplicate SMILES must stay on one side"


def test_two_salts_produce_different_splits():
    smiles = [f"C{i}" for i in range(9000)]
    a = drv.build_split(smiles, drv.SPLIT_SALTS["A"])
    b = drv.build_split(smiles, drv.SPLIT_SALTS["B"])
    assert set(a["fit"]) != set(b["fit"])


def test_metrics_and_subgroup_helper_do_not_touch_official_splits():
    text = inspect.getsource(drv)
    assert "load_official_" + "valid(" not in text
    assert "load_official_" + "test(" not in text
    assert "load_official_train" in text  # only the official TRAIN split may be read


def test_driver_declares_no_banned_target_terms():
    """The driver may name the banned tokens only inside its own gate scanner."""
    text = inspect.getsource(drv)
    for token in ("decomp[", "g_loss", "ell_target", "c_target", "k_target"):
        hits = [line.strip() for line in text.splitlines() if token in line]
        assert hits, token
        assert all("banned" in line for line in hits), (token, hits)


def test_gate_thresholds_are_frozen_in_the_driver():
    assert drv.GATE_MONITOR_DELTA == -0.006
    assert drv.GATE_LOCKED_DELTA == -0.003


# ---------------------------------------------------------------------------
# 6. cache-dependent (skip when the round cache is absent)
# ---------------------------------------------------------------------------


pytestmark_cache = pytest.mark.skipif(not (RESULTS / "split_a.json").exists(), reason="round cache absent")


@pytestmark_cache
def test_frozen_splits_match_rebuild():
    for key in ("A", "B"):
        rows = drv.read_split_rows(key)
        assert len(rows["fit"]) == 8000
        assert len(rows["monitor"]) == 1000
        assert len(rows["locked"]) == 1000


@pytestmark_cache
def test_split_files_record_no_label_read():
    for key in ("A", "B"):
        payload = json.loads((RESULTS / f"split_{key.lower()}.json").read_text())
        assert payload["locked_labels_read"] is False
        assert payload["overlap"]["union"] == 10000


@pytestmark_cache
def test_extended_objects_are_non_degenerate():
    objs = torch.load(drv.OBJECTS_CACHE, map_location="cpu", weights_only=False)
    atom = np.concatenate([o.atom_new for o in objs[:200]], 0)
    assert float(np.abs(atom).max()) > 0.0
    assert atom.shape[1] == tpr.NEW_ATOM_DIM
    del objs


@pytestmark_cache
def test_standardiser_files_are_fit_only():
    for key in ("A", "B"):
        payload = json.loads((RESULTS / f"standardiser_{key.lower()}.json").read_text())
        assert payload["fit_only"] is True
        assert payload["n_fit_rows"] == 8000
        assert len(payload["atom_mean"]) == tpr.NEW_ATOM_DIM
        assert len(payload["bond_mean"]) == tpr.NEW_BOND_DIM
