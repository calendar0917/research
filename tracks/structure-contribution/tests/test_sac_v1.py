"""Fast correctness tests for cscl-bjoc-sac-v1 (BJOC-SAC).

Synthetic molecules only: no ZINC dataset, no checkpoints, no GPU.
Covers the R1/R2/R3 definitions and disjointness, ring-block semantics
(bridge / spiro / fused / disconnected), the content-null identity and its
content-permutation invariance, the relation-assignment witness, the absence of
per-bond aggregation, permutation invariance, parameter/init matching and the
frozen decision vocabulary.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import p3_bond_relation_bridge_v1 as p3b  # noqa: E402
import run_sac_v1 as drv  # noqa: E402
import sac_v1 as sac  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/sac_v1.yaml"
NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _zone(objs):
    return sac.build_zone_relations([sac.build_mol_relations(o) for o in objs])


def _batch(objs):
    return sac.move_batch(sro.batch_objects(objs), torch.device("cpu"))


def _pairs(zr, mol, s):
    a = zr.src[s][zr.starts[mol, s] : zr.ends[mol, s]]
    b = zr.dst[s][zr.starts[mol, s] : zr.ends[mol, s]]
    return set(zip(a.tolist(), b.tolist()))


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------


def test_relations_chain():
    obj = drv.chain(7)
    assert sac.relation_counts(obj).tolist() == [5, 4, 0]


def test_relations_ring():
    obj = drv.ring(6)
    assert sac.relation_counts(obj).tolist() == [6, 6, 3]


def test_relations_fused():
    obj = drv.fused()  # naphthalene skeleton, 11 bonds, one ring block
    m = sac.relation_counts(obj)
    block, n_blocks, n_bridges = sac.bond_ring_blocks(obj)
    assert n_blocks == 1 and n_bridges == 0
    assert (block >= 0).all()
    assert m[2] > 0
    assert int(m.sum()) == sac.choose2(11)


def test_relations_spiro_excludes_cross_ring_pairs():
    obj = drv.spiro()  # two 4-rings sharing one atom -> two blocks
    block, n_blocks, n_bridges = sac.bond_ring_blocks(obj)
    assert n_blocks == 2 and n_bridges == 0
    n = int(obj.bond_index.shape[0])
    dist = sac.line_distances(p3b.line_adjacency(obj))
    all_pairs = sac.complete_pair_list(n)
    cross_far = [
        (int(a), int(b))
        for a, b in zip(all_pairs["src"], all_pairs["dst"])
        if block[int(a)] != block[int(b)] and dist[int(a), int(b)] >= 3
    ]
    assert cross_far, "expected cross-ring pairs at line distance >= 3"
    zr = _zone([obj])
    r3 = _pairs(zr, 0, 2)
    assert not (r3 & set(cross_far))


def test_relations_bridge_excluded_from_r3():
    obj = drv.ring(5)
    # attach a bridge: atom 5 hangs off atom 0
    obj = drv._mol(0, range(6), [(i, (i + 1) % 5, 1) for i in range(5)] + [(0, 5, 1)])
    block, n_blocks, n_bridges = sac.bond_ring_blocks(obj)
    assert n_blocks == 1 and n_bridges == 1
    bridge_bond = int(obj.bond_index.shape[0]) - 1
    r3 = _pairs(_zone([obj]), 0, 2)
    assert all(bridge_bond not in pair for pair in r3)


def test_relations_disconnected_have_no_r3():
    obj = drv.disconnected()
    zr = _zone([obj])
    assert len(zr.src[2]) == 0
    dist = sac.line_distances(p3b.line_adjacency(obj))
    assert (dist >= sac.INF_DIST).any()


def test_relations_are_disjoint_and_cover_only_valid_pairs():
    objs = [drv.chain(8), drv.ring(6), drv.spiro(), drv.fused(), drv.disconnected()]
    zr = _zone(objs)
    for mol, obj in enumerate(objs):
        n = int(obj.bond_index.shape[0])
        sets = [_pairs(zr, mol, s) for s in range(sac.N_RELATIONS)]
        assert sets[0].isdisjoint(sets[1])
        assert sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])
        for s in sets:
            for a, b in s:
                assert 0 <= a < b < n


# ---------------------------------------------------------------------------
# tensors
# ---------------------------------------------------------------------------


def test_graph_tensors_match_explicit_double_loop():
    obj = drv.fused()
    zr = _zone([obj])
    model = sac.build_sac(0, NORM, "cpu")
    feats = model.base.forward_features(_batch([obj]))
    q = [torch.tanh(lin(feats["c"])).detach() for lin in model.project]
    tensors, counts = sac.graph_tensors(q, zr.batch([0], "R", "cpu"))
    assert counts.shape == (1, 3)
    for s in range(sac.N_RELATIONS):
        expected = np.zeros(sac.PROJECT_DIM, dtype=np.float64)
        qn = q[s].numpy().astype(np.float64)
        for a, b in _pairs(zr, 0, s):
            expected += qn[a] * qn[b]
        expected /= math.sqrt(max(1, len(_pairs(zr, 0, s))))
        assert np.abs(expected - tensors[s][0].numpy()).max() <= 2e-5


def test_content_null_identity_and_zero_cases():
    objs = [drv._mol(0, [0, 1], [(0, 1, 1)]), drv.chain(3), drv.fused()]
    zr = _zone(objs)
    model = sac.build_sac(0, NORM, "cpu")
    feats = model.base.forward_features(_batch(objs))
    q = [torch.tanh(lin(feats["c"])).detach() for lin in model.project]
    tensors, _ = sac.graph_tensors(q, zr.batch([0, 1, 2], "U", "cpu"))
    for mol in range(len(objs)):
        n = int(zr.n_bonds[mol])
        for s in range(sac.N_RELATIONS):
            m = int(zr.m[mol, s])
            start = int(zr.n_bonds[:mol].sum())
            qn = q[s][start : start + n].numpy().astype(np.float64)
            if n < 2 or m == 0:
                expected = np.zeros(sac.PROJECT_DIM)
            else:
                total = np.zeros(sac.PROJECT_DIM)
                for a in range(n):
                    for b in range(a + 1, n):
                        total += qn[a] * qn[b]
                expected = (m / (n * (n - 1) / 2)) * total / math.sqrt(m)
            assert np.abs(expected - tensors[s][mol].numpy()).max() <= 2e-5
    # single bond: every tensor is strictly zero under both modes
    solo = _zone([objs[0]])
    feats_solo = model.base.forward_features(_batch([objs[0]]))
    q_solo = [torch.tanh(lin(feats_solo["c"])).detach() for lin in model.project]
    for mode in ("R", "U"):
        tensors_solo, _ = sac.graph_tensors(q_solo, solo.batch([0], mode, "cpu"))
        assert all(float(t.abs().max()) == 0.0 for t in tensors_solo)


def test_relation_assignment_witness():
    """Same bond-content multiset, different assignment: R moves, U cannot."""
    obj = drv.fused()
    zr = _zone([obj])
    model = sac.build_sac(0, NORM, "cpu")
    feats = model.base.forward_features(_batch([obj]))
    q = [torch.tanh(lin(feats["c"])).detach() for lin in model.project]
    n = int(obj.bond_index.shape[0])
    base_R, _ = sac.graph_tensors(q, zr.batch([0], "R", "cpu"))
    base_U, _ = sac.graph_tensors(q, zr.batch([0], "U", "cpu"))
    best_r, worst_u = 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cand = np.arange(n)
            cand[i], cand[j] = cand[j], cand[i]
            if drv._relation_automorphism(cand, zr, 0):
                continue
            qp = [x[torch.as_tensor(cand, dtype=torch.long)] for x in q]
            tR, _ = sac.graph_tensors(qp, zr.batch([0], "R", "cpu"))
            tU, _ = sac.graph_tensors(qp, zr.batch([0], "U", "cpu"))
            best_r = max(best_r, max(float((a - b).abs().max()) for a, b in zip(base_R, tR)))
            worst_u = max(worst_u, max(float((a - b).abs().max()) for a, b in zip(base_U, tU)))
    assert best_r >= 1e-4
    assert worst_u <= 1e-6
    assert best_r >= 100.0 * max(worst_u, 1e-12)


def test_no_per_bond_aggregation_and_tensor_locality():
    objs = [drv.chain(5), drv.fused()]
    zr = _zone(objs)
    model = sac.build_sac(0, NORM, "cpu")
    batch = _batch(objs)
    base_feats = model.base.forward_features(batch)
    sac_feats = model.forward_features(batch, zr.batch([0, 1], "R", "cpu"))
    # the composed per-bond vector and the summed z block are untouched
    assert torch.equal(base_feats["c"], sac_feats["c"])
    assert torch.equal(base_feats["z"], sac_feats["z"])
    # q_s(e) is a pure function of that bond's own c_e
    for s, lin in enumerate(model.project):
        assert torch.equal(sac_feats["q"][s], torch.tanh(lin(sac_feats["c"])))
    assert sac_feats["reader_in"].shape[1] == int(model.config.reader_in) + sac.PROJECT_OUT + 3


# ---------------------------------------------------------------------------
# model / init / invariance
# ---------------------------------------------------------------------------


def test_parameter_budget_and_shared_trunk():
    cfg = yaml.safe_load(CONFIG.read_text())
    p3 = sac.build_p3(0, NORM, "cpu")
    model = sac.build_sac(0, NORM, "cpu")
    assert int(sum(p.numel() for p in p3.parameters())) == cfg["base_parameters"] == 413_505
    assert model.n_parameters == cfg["sac_parameters"] == 439_689
    assert model.new_parameters == 26_184
    assert model.n_parameters <= cfg["param_cap"]
    assert sac.shared_state_hash(model) == sac.shared_state_hash(p3)
    assert sac.full_state_hash(sac.build_sac(0, NORM, "cpu")) == sac.full_state_hash(model)
    # the P3 content readout path is preserved bit-for-bit inside the new reader
    old_dim = int(p3.config.reader_in)
    assert torch.equal(model.reader.lins[0].weight[:, :old_dim], p3.reader.lins[0].weight)
    assert torch.equal(model.reader.lins[0].bias, p3.reader.lins[0].bias)
    for a, b in zip(model.reader.lins[1:], p3.reader.lins[1:]):
        assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)


def test_permutation_invariance():
    obj = drv.ring(8, alternating=True)
    rng = np.random.default_rng(7)
    perm = rng.permutation(obj.n_atoms)
    relabelled = drv._relabel(obj, perm)
    model = sac.build_sac(0, NORM, "cpu")
    pred_a = sac.predict_model(model, [obj], _zone([obj]), "R", "cpu")
    pred_b = sac.predict_model(model, [relabelled], _zone([relabelled]), "R", "cpu")
    assert abs(float(pred_a[0]) - float(pred_b[0])) <= 1e-5
    # batch composition invariance
    objs = [drv.chain(4), obj, drv.fused()]
    pa = sac.predict_model(model, objs, _zone(objs), "R", "cpu")
    sub = sac.predict_model(model, [obj], _zone([obj]), "R", "cpu")
    assert abs(float(pa[1]) - float(sub[0])) <= 1e-5


def test_gradient_liveness():
    objs = [drv.chain(5), drv.ring(6), drv.fused()]
    zr = _zone(objs)
    model = sac.build_sac(0, NORM, "cpu")
    y = torch.tensor([-1.0, 0.5, 2.0])
    for mode in ("R", "U"):
        probe = sac.gradient_probe(model, _batch(objs), zr.batch([0, 1, 2], mode, "cpu"), y)
        for key in ("atom_encoder", "bond_encoder", "junction_encoder", "group_composer", "reader", "project"):
            assert probe[f"grad_norm_{key}"] > 0.0, key
        assert all(probe[f"grad_norm_T_R{i}"] > 0.0 for i in (1, 2, 3))


# ---------------------------------------------------------------------------
# frozen vocabulary / config
# ---------------------------------------------------------------------------


def test_config_is_consistent_with_module_constants():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["protocol"] == sac.PROTOCOL_ID
    assert cfg["salt"] == sac.SPLIT_SALT
    assert tuple(cfg["seeds"]) == sac.SEEDS
    assert tuple(cfg["arms"]) == sac.ARMS
    assert int(cfg["project_dim"]) == sac.PROJECT_DIM
    assert int(cfg["sac_parameters"]) == sac.EXPECTED_SAC_PARAMETERS
    assert float(cfg["meaningful_delta_r_vs_p3"]) == sac.MEANINGFUL_DELTA_R_VS_P3
    assert float(cfg["meaningful_delta_r_vs_u"]) == sac.MEANINGFUL_DELTA_R_VS_U
    assert cfg["reader_gate"] == "none"


def test_contrast_label_vocabulary():
    gain = sac.contrast_label(-0.005, {"0": -0.006, "1": -0.004}, -0.008, -0.002, -0.001, 0.004)
    assert gain == "GAIN"
    neutral = sac.contrast_label(-0.001, {"0": -0.002, "1": 0.0005}, -0.003, 0.001, 0.000, 0.004)
    assert neutral == "NEUTRAL"
    loss = sac.contrast_label(0.005, {"0": 0.004, "1": 0.006}, 0.002, 0.008, 0.001, 0.004)
    assert loss == "LOSS"
    # a gain whose ordinary slice worsens is not a GAIN
    assert sac.contrast_label(-0.005, {"0": -0.006, "1": -0.004}, -0.008, -0.002, 0.001, 0.004) == "NEUTRAL"


def test_sac_case_rules():
    kw = dict(
        label_r_p3="GAIN",
        label_r_u="GAIN",
        label_u_p3="GAIN",
        replicated_r_p3=True,
        replicated_r_u=True,
        replicated_u_p3=True,
        top10_r_p3=-0.001,
        participation=True,
    )
    assert sac.sac_case(**kw)["case"] == "CASE_A_STRUCTURAL_RELATION_GAIN"
    assert sac.sac_case(**{**kw, "label_r_u": "NEUTRAL"})["case"] == "CASE_C_GENERIC_SECOND_ORDER"
    assert (
        sac.sac_case(**{**kw, "label_r_p3": "NEUTRAL", "label_u_p3": "NEUTRAL", "replicated_u_p3": False})["case"]
        == "CASE_B_RELATION_USED_NO_HEADLINE_GAIN"
    )
    assert sac.sac_case(**{**kw, "participation": False})["case"] == "CASE_E_R_GAIN_NOT_ATTRIBUTED"
    assert (
        sac.sac_case(**{**kw, "label_r_p3": "NEUTRAL", "label_r_u": "NEUTRAL"})["case"] == "CASE_D_NO_GAIN"
    )


def test_scoring_path_on_synthetic_predictions() -> None:
    """Exercise the score-holdout maths (arm stats + contrasts + case rule) on label-free fakes."""
    import run_sac_v1 as drv

    rng = np.random.default_rng(7)
    n = 96
    y = rng.normal(0.5, 0.9, n)
    preds = {arm: {s: y + rng.normal(0.0, 0.25, n) for s in drv.SEEDS} for arm in drv.ARMS}
    preds["SACR"] = {s: y + rng.normal(0.0, 0.10, n) for s in drv.SEEDS}

    stats = {arm: drv._arm_stats(y, preds[arm]) for arm in drv.ARMS}
    for arm in drv.ARMS:
        assert set(stats[arm]) >= {"mae_seed_mean", "mae_per_seed", "ordinary_mae", "tail_mae", "top10_sse_share"}
    assert stats["SACR"]["mae_seed_mean"] < stats["P3"]["mae_seed_mean"]

    contrasts = {
        name: drv._contrast(y, preds[a], preds[b], 0.004) for name, a, b in drv.sac.CONTRASTS
    }
    for name, c in contrasts.items():
        assert set(c) >= {"delta_seed_mean", "per_seed_delta", "bootstrap", "label", "top10_removed_delta"}
        assert set(c["per_seed_delta"]) == {str(s) for s in drv.SEEDS}
        assert c["bootstrap"]["ci_lo"] <= c["bootstrap"]["ci_hi"]
        assert c["label"] in {"GAIN", "NEUTRAL", "HARM"}
        for seed_info in c["per_seed_bootstrap"].values():
            assert seed_info["ci_lo"] <= seed_info["ci_hi"]
    assert contrasts["SACR_minus_P3"]["delta_seed_mean"] < 0.0

    rep = {
        name: {
            "seed_mean_delta": float(np.mean(list(c["per_seed_delta"].values()))),
            "per_seed_delta": dict(c["per_seed_delta"]),
            "replicated": bool(all(v < 0.0 for v in c["per_seed_delta"].values())),
        }
        for name, c in contrasts.items()
    }
    primary = contrasts
    case = drv.sac.sac_case(
        label_r_p3=primary["SACR_minus_P3"]["label"],
        label_r_u=primary["SACR_minus_SACU"]["label"],
        label_u_p3=primary["SACU_minus_P3"]["label"],
        replicated_r_p3=rep["SACR_minus_P3"]["replicated"],
        replicated_r_u=rep["SACR_minus_SACU"]["replicated"],
        replicated_u_p3=rep["SACU_minus_P3"]["replicated"],
        top10_r_p3=primary["SACR_minus_P3"]["top10_removed_delta"],
        participation=True,
    )
    assert case["case"].startswith("CASE_")
    assert isinstance(case["reasons"], list) and case["reasons"]

    # deterministic: the same inputs give the same bootstrap interval
    again = drv._contrast(y, preds["SACR"], preds["P3"], 0.004)
    assert again["bootstrap"] == contrasts["SACR_minus_P3"]["bootstrap"]
