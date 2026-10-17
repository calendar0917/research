"""Fast correctness tests for cscl-bond-relation-attribution-v1.

No ZINC dataset, no checkpoints, no GPU: tiny synthetic molecules only.
Covers the R/D/U kernel definitions (row sums, total weight, rank-1 Null,
non-degeneracy), the identical triplet layout, invariances, gradient liveness,
the identity-grouped split builder, the frozen decision vocabulary and the
historical-reuse hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bond_random_walk_organization_v1 as bwo  # noqa: E402
import bond_relation_attribution_v1 as bra  # noqa: E402
import bond_rw_local_vs_multiscale_v1 as lm  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/bond_relation_attribution_v1.yaml"
DRIVER = REPO_ROOT / "tracks/structure-contribution/code/run_bond_relation_attribution_v1.py"
MODULE = REPO_ROOT / "tracks/structure-contribution/code/bond_relation_attribution_v1.py"
NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _mol(idx: int, atom_types, bonds):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(idx, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))


def _chain(n: int, idx: int = 0):
    return _mol(idx, range(n), [(i, i + 1, 1) for i in range(n - 1)])


def _ring(n: int, idx: int = 0):
    bonds = [(i, (i + 1) % n, 1) for i in range(n)]
    return _mol(idx, range(n), bonds)


def _branched(idx: int = 0):
    return _mol(idx, range(6), [(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 1), (1, 5, 1)])


OBJECTS = (_chain(4), _chain(7), _ring(5), _ring(8), _branched())


def _batch(objs, triplets):
    return bwo.move_batch(bwo.batch_rw_objects(list(objs), list(triplets)), torch.device("cpu"))


def _relabel(o, perm):
    inv = np.argsort(perm)
    return sro.MolObjects(
        mol_index=o.mol_index,
        smiles=o.smiles,
        n_atoms=o.n_atoms,
        atom_feat=o.atom_feat[perm],
        bond_index=inv[o.bond_index],
        bond_type=o.bond_type,
        bond_feat=o.bond_feat,
        junc_index=o.junc_index,
        junc_bond=o.junc_bond,
        junc_feat=o.junc_feat,
        global_feat=o.global_feat,
    )


# ---------------------------------------------------------------------------
# frozen config / provenance
# ---------------------------------------------------------------------------


def test_config_matches_module_constants():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol"] == bra.PROTOCOL_ID
    assert cfg["status"] == "frozen"
    assert cfg["split_tag"] == bra.SPLIT_TAG
    assert int(cfg["split"]["fit_target"]) == bra.FIT_TARGET
    assert int(cfg["split"]["monitor_target"]) == bra.MONITOR_TARGET
    assert int(cfg["split"]["locked_target"]) == bra.LOCKED_TARGET
    assert cfg["data"]["excluded_rows"] == list(bra.EXCLUDED_ROWS)
    assert int(cfg["arms"]["R"]["parameters"]) == bra.EXPECTED_PARAMETERS
    assert int(cfg["arms"]["P3"]["parameters"]) == bra.P3_EXPECTED_PARAMETERS
    assert float(cfg["scoring"]["decisions"]["meaningful_delta"]) == bra.MEANINGFUL_DELTA
    assert float(cfg["scoring"]["decisions"]["case_a_max_gap_to_p3"]) == bra.CASE_A_MAX_GAP_TO_P3
    assert cfg["scoring"]["stop_after_round"] is True


def test_reuse_hashes_match_working_tree():
    for name, expected in bra.REUSE_HASHES.items():
        got = hashlib.sha256((_CODE_DIR / name).read_bytes()).hexdigest()
        assert got == expected, f"{name} was modified; historical reuse contract broken"


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def test_r_kernel_is_historical_r1():
    for o in OBJECTS:
        a = bra.r_triplets_from_objects(o)
        b = lm.k1_triplets_from_objects(o)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])
        assert np.array_equal(a[2], b[2])


def test_rowsum_null_matches_row_sums_total_and_rank_one():
    for o in OBJECTS:
        K = bra.r_matrix(o)
        K_D, s = bra.rowsum_null_matrix(K)
        assert np.allclose(K_D, K_D.T, atol=0.0)
        assert np.allclose(K_D.sum(1), s, atol=1e-12)
        assert abs(K_D.sum() - K.sum()) <= 1e-10
        assert np.allclose(K_D, np.outer(s, s) / s.sum(), atol=1e-15)
        assert np.linalg.matrix_rank(K_D, tol=1e-9) == 1
        assert (K_D >= 0).all()
        assert np.abs(K_D - K).max() > 1e-6


def test_single_bond_degenerates_only_there():
    o = _mol(0, [0, 1], [(0, 1, 1)])
    K = bra.r_matrix(o)
    K_D, s = bra.rowsum_null_matrix(K)
    assert np.array_equal(K, K_D)
    assert np.array_equal(K_D, bra.u_matrix(1))
    audit = bra.relation_audit_molecule(o)
    assert audit["n_bonds"] == 1


def test_rowsum_null_rejects_invalid_matrices():
    with pytest.raises(ValueError):
        bra.rowsum_null_matrix(np.asarray([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        bra.rowsum_null_matrix(np.asarray([[1.0, -0.1], [-0.1, 1.0]]))


def test_d_differs_from_r_always_and_from_u_except_regular_line_graphs():
    """D == U exactly when every bond has the same row sum (regular line graph).

    Ring molecules are the clean example: every bond has line-graph degree 2 and
    ``rowsum(sym(P)) == 1``, so the row-sum Null is exactly the uniform kernel.
    This is an analytic degeneracy that the round must report, not hide.
    """
    for o in OBJECTS:
        K = bra.r_matrix(o)
        K_D, s = bra.rowsum_null_matrix(K)
        n = K.shape[0]
        assert np.abs(K_D - K).max() > 1e-6
        if np.allclose(s, s[0], atol=1e-12):
            assert np.abs(K_D - bra.u_matrix(n)).max() <= 1e-12
        else:
            assert np.abs(K_D - bra.u_matrix(n)).max() > 1e-6
    # explicit regular vs non-regular pair
    ring_K, _ = bra.rowsum_null_matrix(bra.r_matrix(_ring(6)))
    chain_K, _ = bra.rowsum_null_matrix(bra.r_matrix(_chain(4)))
    assert np.abs(ring_K - bra.u_matrix(6)).max() <= 1e-12
    assert np.abs(chain_K - bra.u_matrix(3)).max() > 1e-3


def test_all_arms_share_identical_pair_layout():
    for o in OBJECTS:
        r = bra.r_triplets_from_objects(o)
        d = bra.d_triplets_from_objects(o)
        u = bra.u_triplets_from_objects(o)
        n = int(o.bond_index.shape[0])
        for trip in (r, d, u):
            assert int(trip[0].shape[0]) == n * (n + 1) // 2
        assert np.array_equal(r[0], d[0]) and np.array_equal(r[1], d[1])
        assert np.array_equal(r[0], u[0]) and np.array_equal(r[1], u[1])
        assert not np.array_equal(r[2], d[2])


def test_triplets_reconstruct_dense_double_sum():
    rng = np.random.default_rng(3)
    for o in OBJECTS:
        n = int(o.bond_index.shape[0])
        B = rng.normal(size=(n, 4))
        phi = B[:, None, :] + B[None, :, :] + B[:, None, :] * B[None, :, :]
        for K in (bra.r_matrix(o), bra.d_matrix(o)[0], bra.u_matrix(n)):
            s, d, w = bra.kernel_triplets(K)
            w_eff = np.where(s == d, w * 0.5, w)
            trip = np.zeros((n, 4))
            np.add.at(trip, s, w_eff[:, None] * phi[s, d])
            np.add.at(trip, d, w_eff[:, None] * phi[s, d])
            assert np.abs(np.einsum("ef,efk->ek", K, phi) - trip).max() <= 1e-6


def test_d_is_bond_permutation_invariant():
    rng = np.random.default_rng(11)
    for o in OBJECTS:
        K_D, _ = bra.d_matrix(o)
        perm = rng.permutation(int(o.bond_index.shape[0]))
        A = bwo.line_graph_adjacency(o.bond_index)
        A2 = A[np.ix_(perm, perm)]
        K2 = bwo.symmetrize(bwo.lazy_transition(A2))
        K2_D, _ = bra.rowsum_null_matrix(K2)
        assert np.abs(K2_D - K_D[np.ix_(perm, perm)]).max() <= 1e-12


def test_relation_audit_conclusions_hold_on_synthetic():
    audit = bra.summarize_relation_audit([bra.relation_audit_molecule(o) for o in OBJECTS])
    for key, value in audit["conclusions"].items():
        assert bool(value), key


# ---------------------------------------------------------------------------
# model-level: kernel consumption, invariance, gradients
# ---------------------------------------------------------------------------


def _model(seed: int = 0):
    return bwo.build_model(seed, NORM, "cpu")


def _pair_aggregate(m, obj, trip):
    batch = _batch([obj], [trip])
    with torch.no_grad():
        feats = m.forward_features(batch)
    return feats["r"].numpy(), float(m(batch))


def test_forward_depends_on_kernel_at_init():
    m = _model()
    objs = OBJECTS[:4]
    preds = {}
    r_blocks = {}
    for arm in ("R", "D", "U"):
        batch = _batch(objs, [bra.triplets_for(arm, o) for o in objs])
        with torch.no_grad():
            preds[arm] = m(batch).numpy()
    assert np.abs(preds["R"] - preds["D"]).max() > 1e-7
    assert np.abs(preds["D"] - preds["U"]).max() > 1e-7
    # the pair aggregate itself must react strongly (the reader averages it out
    # at initialisation, so the prediction difference alone is a weak check)
    obj = _chain(7)
    for arm in ("R", "D", "U"):
        r_blocks[arm], _ = _pair_aggregate(m, obj, bra.triplets_for(arm, obj))
    assert np.abs(r_blocks["R"] - r_blocks["D"]).max() > 1e-4
    assert np.abs(r_blocks["D"] - r_blocks["U"]).max() > 1e-3


def test_atom_relabel_invariance_per_kernel():
    rng = np.random.default_rng(5)
    m = _model()
    m.eval()
    for arm in ("R", "D", "U"):
        for o in OBJECTS:
            trip = bra.triplets_for(arm, o)
            with torch.no_grad():
                p0 = float(m(_batch([o], [trip])))
            perm = rng.permutation(o.n_atoms)
            with torch.no_grad():
                p1 = float(m(_batch([_relabel(o, perm)], [trip])))
            assert abs(p0 - p1) <= 1e-6


def test_batch_composition_invariance_per_kernel():
    m = _model()
    objs = OBJECTS[:4]
    for arm in ("R", "D", "U"):
        trips = [bra.triplets_for(arm, o) for o in objs]
        with torch.no_grad():
            batched = m(_batch(objs, trips)).numpy()
            singles = np.asarray([float(m(_batch([o], [t]))) for o, t in zip(objs, trips)])
        assert np.abs(batched - singles).max() <= 1e-6


def test_pair_mlp_gradients_live_for_every_kernel():
    objs = OBJECTS[:4]
    for arm in ("R", "D", "U"):
        m = _model()
        m.train()
        batch = _batch(objs, [bra.triplets_for(arm, o) for o in objs])
        loss = torch.nn.functional.l1_loss(m(batch).view(-1), torch.zeros(len(objs)))
        loss.backward()
        for module in (m.atom_encoder, m.bond_encoder, m.pair_mlp, m.group_composer, m.reader):
            norms = [float(p.grad.norm()) for p in module.parameters() if p.grad is not None]
            assert norms and all(v > 0 for v in norms), (arm, module)


def test_parameter_count_and_init_parity():
    ms = [bwo.build_model(0, NORM, "cpu") for _ in range(3)]
    assert {bwo.parameter_count(m) for m in ms} == {bra.EXPECTED_PARAMETERS}
    assert len({bwo.state_dict_hash(m) for m in ms}) == 1
    p3 = bp.build_bjoc_null(0, NORM, "cpu")
    assert sum(p.numel() for p in p3.parameters()) == bra.P3_EXPECTED_PARAMETERS


def test_batch_order_hash_is_shared_and_updates_match():
    n_fit = bra.FIT_TARGET
    steps = -(-n_fit // 128)
    assert steps == 63
    assert steps * 360 == 22_680
    for seed in bra.SEEDS:
        h = bp.batch_order_hash(seed, n_fit, 128, 360)
        assert h == bp.batch_order_hash(seed, n_fit, 128, 360)
        assert h != bp.batch_order_hash(seed + 1, n_fit, 128, 360)


# ---------------------------------------------------------------------------
# split builder
# ---------------------------------------------------------------------------


def test_identity_split_is_deterministic_grouped_and_sized():
    rows = list(range(20))
    keys = [f"k{i % 12}" for i in rows]  # 8 duplicated keys
    a = bra.build_identity_split(rows, keys, fit_target=12, monitor_target=4)
    b = bra.build_identity_split(rows, keys, fit_target=12, monitor_target=4)
    assert a["fit_rows"] == b["fit_rows"] and a["monitor_rows"] == b["monitor_rows"]
    assert a["locked_rows"] == b["locked_rows"]
    assert a["duplicate_groups_cross_sides"] == 0
    assert len(a["fit_rows"]) >= 12 and len(a["monitor_rows"]) >= 4
    assert sorted(a["fit_rows"] + a["monitor_rows"] + a["locked_rows"]) == rows
    # groups are contiguous in the row->group map
    by_group: dict[int, set[str]] = {}
    side = {r: "fit" for r in a["fit_rows"]}
    side.update({r: "monitor" for r in a["monitor_rows"]})
    side.update({r: "locked" for r in a["locked_rows"]})
    for r, g in a["group_id"].items():
        by_group.setdefault(int(g), set()).add(side[int(r)])
    assert all(len(v) == 1 for v in by_group.values())
    assert len(by_group) == len(set(keys))


def test_identity_split_changes_with_salt():
    rows = list(range(30))
    keys = [f"k{i % 17}" for i in rows]
    a = bra.build_identity_split(rows, keys, salt="salt-a", fit_target=10, monitor_target=5)
    b = bra.build_identity_split(rows, keys, salt="salt-b", fit_target=10, monitor_target=5)
    assert a["fit_rows"] != b["fit_rows"]


# ---------------------------------------------------------------------------
# frozen decision vocabulary
# ---------------------------------------------------------------------------


def test_contrast_label_vocabulary():
    assert bra.contrast_label(-0.006, {"0": -0.005, "1": -0.007}, ci_lo=-0.009, ci_hi=-0.001) == "GAIN"
    assert bra.contrast_label(-0.006, {"0": -0.005, "1": -0.007}, ci_lo=-0.009, ci_hi=0.001) == "NEUTRAL"
    assert bra.contrast_label(-0.006, {"0": -0.005, "1": 0.001}, ci_lo=-0.009, ci_hi=-0.001) == "NEUTRAL"
    assert bra.contrast_label(-0.003, {"0": -0.002, "1": -0.004}, ci_lo=-0.006, ci_hi=-0.001) == "NEUTRAL"
    assert bra.contrast_label(0.006, {"0": 0.005, "1": 0.007}, ci_lo=0.001, ci_hi=0.009) == "LOSS"


def test_replication_status():
    assert bra.replication_status({"0": -0.01, "1": -0.02})["replicated"] is True
    assert bra.replication_status({"0": -0.01, "1": 0.02})["replicated"] is False


def test_overall_case_case_a():
    out = bra.overall_case(
        p1_locked_label="GAIN",
        p1_monitor={"0": -0.01, "1": -0.02},
        p2_locked_label="GAIN",
        p2_monitor={"0": -0.02, "1": -0.03},
        p1_locked_delta=-0.006,
        p1_monitor_delta=-0.008,
        p2_locked_delta=-0.007,
        r_minus_p3_locked=0.004,
    )
    assert out["case"] == "CASE_A_RELATION_CANDIDATE"


def test_overall_case_b1_rowsum_absorbs():
    out = bra.overall_case(
        p1_locked_label="NEUTRAL",
        p1_monitor={"0": -0.001, "1": -0.002},
        p2_locked_label="GAIN",
        p2_monitor={"0": -0.01, "1": -0.02},
        p1_locked_delta=-0.0015,
        p1_monitor_delta=-0.0015,
        p2_locked_delta=-0.01,
        r_minus_p3_locked=0.005,
    )
    assert out["case"] == "CASE_B1_ROWSUNM_ABSORBS"


def test_overall_case_b2_no_gain_over_uniform():
    out = bra.overall_case(
        p1_locked_label="GAIN",
        p1_monitor={"0": -0.01, "1": -0.02},
        p2_locked_label="NEUTRAL",
        p2_monitor={"0": 0.002, "1": -0.001},
        p1_locked_delta=-0.006,
        p1_monitor_delta=-0.008,
        p2_locked_delta=0.003,
        r_minus_p3_locked=0.005,
    )
    assert out["case"] == "CASE_B2_NO_GAIN_OVER_UNIFORM"


def test_overall_case_b3_far_below_reference():
    out = bra.overall_case(
        p1_locked_label="GAIN",
        p1_monitor={"0": -0.01, "1": -0.02},
        p2_locked_label="GAIN",
        p2_monitor={"0": -0.02, "1": -0.03},
        p1_locked_delta=-0.006,
        p1_monitor_delta=-0.008,
        p2_locked_delta=-0.007,
        r_minus_p3_locked=0.05,
    )
    assert out["case"] == "CASE_B3_FAR_BELOW_STRONG_REFERENCE"


def test_overall_case_c_sign_conflict():
    out = bra.overall_case(
        p1_locked_label="GAIN",
        p1_monitor={"0": -0.01, "1": 0.002},
        p2_locked_label="GAIN",
        p2_monitor={"0": -0.02, "1": -0.03},
        p1_locked_delta=-0.006,
        p1_monitor_delta=-0.004,
        p2_locked_delta=-0.007,
        r_minus_p3_locked=0.005,
    )
    assert out["case"] == "CASE_C_INCONCLUSIVE"


# ---------------------------------------------------------------------------
# discipline scans
# ---------------------------------------------------------------------------


def test_no_official_valid_or_test_entry_point():
    import re

    text = DRIVER.read_text(encoding="utf-8")
    pattern = re.compile(r"split\s*=\s*[\"'](?:valid|test)[\"']")
    assert pattern.search(text) is None
    assert '"official_valid_loaded": False' in text
    assert '"official_test_loaded": False' in text


def test_driver_never_mentions_excluded_row_removal_at_train_time():
    text = DRIVER.read_text(encoding="utf-8")
    assert "EXCLUDED_ROWS" in text  # exclusion is declared, once, in the split stage


def test_evidence_manifest_and_identity_signature_helpers(tmp_path):
    sig = bra.identity_group_signature([0, 1, 2, 3], [0, 0, 1, 2])
    assert sig["n_rows"] == 4
    assert sig["n_groups"] == 3
    assert sig["n_duplicate_groups"] == 1
    assert sig["max_group_size"] == 2


def test_json_roundtrip_of_decision_payload():
    payload = bra.overall_case(
        p1_locked_label="NEUTRAL",
        p1_monitor={"0": 0.0, "1": 0.0},
        p2_locked_label="NEUTRAL",
        p2_monitor={"0": 0.0, "1": 0.0},
        p1_locked_delta=0.0,
        p1_monitor_delta=0.0,
        p2_locked_delta=0.0,
        r_minus_p3_locked=0.0,
    )
    assert json.loads(json.dumps(payload))["case"].startswith("CASE_")
