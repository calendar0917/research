"""Fast correctness tests for cscl-bjoc-architecture-breakthrough-v1.

No ZINC dataset and no GPU: synthetic molecules only, plus committed tables
(canonical SMILES fingerprints).  Covers model identity/parity, conservation,
invariance, the ownership counterexample, pair construction, the frozen
training-loop equivalence with the historical loop, gate arithmetic, the
label-access guard and the no-official-test discipline.
"""

from __future__ import annotations

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

import bjoc_arch_breakthrough_v1 as ab  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import run_bjoc_architecture_breakthrough_v1 as drv  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _cfg():
    return yaml.safe_load(drv.CONFIG_PATH.read_text(encoding="utf-8"))


def _mol(idx, atom_types, bonds):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    edge_index = np.asarray(ei, dtype=np.int64).T
    return sro.build_mol_objects(idx, x, edge_index, np.asarray(ea, dtype=np.int64))


def _small_objects():
    return [
        _mol(0, [0, 1, 2, 3], [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 0)]),
        _mol(1, [4, 5, 6], [(0, 1, 0), (1, 2, 2)]),
        _mol(2, [7, 7, 8, 9, 10], [(0, 1, 0), (1, 2, 0), (2, 3, 1), (3, 4, 0), (4, 1, 0)]),
        _mol(3, [0, 0, 0, 0, 0, 0], [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0)]),
    ]


# ---------------------------------------------------------------------------
# frozen config / registry agreement
# ---------------------------------------------------------------------------


def test_config_and_registry_agree():
    cfg = _cfg()
    assert cfg["protocol"] == ab.PROTOCOL_ID == drv.PROTOCOL_ID
    assert cfg["official_test"] == "never_read"
    assert cfg["splits"]["salts"]["C"] == ab.SPLIT_SALTS["C"] == drv.ab.SPLIT_SALTS["C"]
    assert cfg["splits"]["salts"]["D"] == ab.SPLIT_SALTS["D"]
    assert cfg["frozen"]["expected_parameters"] == bp.EXPECTED_PARAMETERS == 413505
    assert cfg["frozen"]["glrc_parameters"] == ab.GLRC_PARAMETERS == 413539
    assert cfg["frozen"]["dlp_parameters"] == ab.DLP_PARAMETERS == 441153
    assert cfg["stage_a_glrc"]["architecture"]["new_parameters"] == 34
    assert cfg["stage_b_dlp"]["architecture"]["new_parameters"] == 27648
    assert cfg["stage_b_dlp"]["architecture"]["projection"].startswith("h_i = P a_i")
    assert cfg["bans"][0] == "official test access in any form"
    assert cfg["budget"]["gpu_hours_max"] == 6.0
    assert bp.RECIPES["R3"] == {
        "epochs": 360,
        "schedule": "cosine",
        "lr": 1.0e-3,
        "lr_min": 1.0e-5,
        "weight_decay": 1.0e-5,
    }


def test_gate_constants_frozen():
    cfg = _cfg()
    assert cfg["stage_a_glrc"]["monitor_gate"]["delta_seed_mean_le"] == ab.GLRC_DELTA_THRESHOLD == -0.004
    assert cfg["stage_a_glrc"]["monitor_gate"]["rmse_tolerance"] == ab.GATE_RMSE_TOL == 0.005
    assert cfg["stage_a_glrc"]["monitor_gate"]["max_single_gain_fraction"] == ab.GATE_MAX_SINGLE_GAIN_FRACTION == 0.25
    assert cfg["stage_a_glrc"]["architecture"]["change"].count("W_null_b + lambda(G)") == 1
    assert "three 32-dim vectors appended" in cfg["stage_b_dlp"]["architecture"]["reader"]


def test_split_fingerprints_match_committed_table():
    from cscl_features import load_canonical_smiles

    cfg = _cfg()
    smiles = [str(s) for s in load_canonical_smiles()]
    assert len(smiles) == 10000
    for tag in ("C", "D"):
        split = bp.build_split(smiles, ab.SPLIT_SALTS[tag])
        expected = cfg["splits"]["fingerprints"][tag]
        assert len(split["fit"]) == expected["n_fit"]
        assert len(split["monitor"]) == expected["n_monitor"]
        assert len(split["locked"]) == expected["n_locked"]
        assert bp.rows_sha256([int(i) for i in split["fit"]]) == expected["fit_sha256_rows"]
        assert bp.rows_sha256([int(i) for i in split["monitor"]]) == expected["monitor_sha256_rows"]
        assert bp.rows_sha256([int(i) for i in split["locked"]]) == expected["locked_sha256_rows"]
        assert split["duplicate_groups_cross_sides"] == 0


# ---------------------------------------------------------------------------
# label guard
# ---------------------------------------------------------------------------


def test_label_guard_positive_control_and_real_passthrough():
    assert ab.label_guard_selfcheck() == {
        "structure_passthrough": True,
        "guarded_y_blocked": True,
        "unguarded_y_allowed": True,
    }

    class _Mol:
        x = np.zeros((3, 1), dtype=np.int64)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0,), dtype=np.int64)
        y = np.asarray([2.0])

    guard = ab.LabelGuard([_Mol(), _Mol(), _Mol()], forbidden_rows=[1, 2])
    for row in (1, 2):
        d = guard[row]
        assert int(d.x.shape[0]) == 3
        with pytest.raises(ab.ForbiddenLabelAccess):
            _ = d.y
        with pytest.raises(ab.ForbiddenLabelAccess):
            _ = d["y"]
    assert guard.access_attempts == [1, 1, 2, 2]
    assert float(guard[0].y[0]) == 2.0
    assert guard.access_attempts == [1, 1, 2, 2]


def test_objects_build_is_label_free_and_alignment_is_fit_only():
    text = (Path(_CODE_DIR) / "run_bjoc_architecture_breakthrough_v1.py").read_text(encoding="utf-8")
    assert "ab.LabelGuard(ds, forbidden_rows)" in text
    assert 'split["monitor_rows"] + split["locked_rows"]' in text
    assert 'check_rows = list(split["fit_rows"][:200])' in text
    assert 'alignment_rows_scope": "Fit rows only"' in text
    # the only raw-dataset .y access in cmd_objects is the Fit-only alignment check
    body = text.split("def cmd_objects", 1)[1].split("\ndef ", 1)[0]
    assert body.count("ds[i].y") == 1
    assert body.count("enc[i].y") == 1


# ---------------------------------------------------------------------------
# GLRC
# ---------------------------------------------------------------------------


def test_glrc_parameter_budget_and_init_parity():
    cfg = _cfg()
    base, model, info = ab.build_glrc_from_init(0, NORM, "cpu", None)
    assert sum(p.numel() for p in model.parameters()) == ab.GLRC_PARAMETERS
    assert sum(p.numel() for p in base.parameters()) == bp.EXPECTED_PARAMETERS
    assert info["new_parameters"] == 34
    assert ab.base_parameter_hash(base) == ab.base_parameter_hash(model, exclude_prefixes=("lambda_gate.",))
    assert src.state_dict_hash(base) == cfg["frozen"]["p_init_full_sha256"]["0"]
    for seed in (0, 1):
        m = ab.build_glrc_from_init(seed, NORM, "cpu", None)[1]
        b = bp.build_bjoc_null(seed, NORM, "cpu")
        assert ab.base_parameter_hash(b) == ab.base_parameter_hash(m, exclude_prefixes=("lambda_gate.",))
        assert torch.equal(m.lambda_gate.weight, torch.zeros_like(m.lambda_gate.weight))
        assert torch.equal(m.lambda_gate.bias, torch.zeros_like(m.lambda_gate.bias))


def test_glrc_identity_at_init():
    base, model, _ = ab.build_glrc_from_init(0, NORM, "cpu", None)
    batch = sro.batch_objects(_small_objects())
    with torch.no_grad():
        yb = base(batch)
        yg = model(batch)
        lam = model.lambda_of(batch, next(model.parameters()).dtype)
    assert float((yb - yg).abs().max()) < 1.0e-6
    assert float(lam.abs().max()) == 0.0


def test_glrc_conservation_per_molecule():
    _, model, _ = ab.build_glrc_from_init(0, NORM, "cpu", None)
    objs = _small_objects()
    batch = sro.batch_objects(objs)
    feats = model.forward_features(batch, lambda_override=0.5)
    m = batch["bond_batch"]
    for g in range(len(objs)):
        mm = m == g
        assert float(feats["dW"][mm].sum(0).abs().max()) < 1.0e-5
        assert float(feats["dV"][mm].sum(0).abs().max()) < 1.0e-5


def test_glrc_invariance_and_batch_membership():
    _, model, _ = ab.build_glrc_from_init(0, NORM, "cpu", None)
    objs = _small_objects()
    batch = sro.batch_objects(objs)
    with torch.no_grad():
        y0 = model(batch, lambda_override=0.5).numpy()
        y_single = np.asarray([model(sro.batch_objects([o]), lambda_override=0.5).item() for o in objs])
    assert np.abs(y0 - y_single).max() < 1e-5

    # atom relabel through the raw builder
    topo = ([0, 1, 2, 3], [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 0)])
    perm = np.asarray([2, 0, 3, 1])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    x = np.asarray(topo[0], dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in topo[1]:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    ei = np.asarray(ei, dtype=np.int64).T
    p1 = sro.build_mol_objects(-1, x, ei, np.asarray(ea, dtype=np.int64))
    p2 = sro.build_mol_objects(-1, x[perm], inv[ei], np.asarray(ea, dtype=np.int64))
    p3 = sro.build_mol_objects(-1, x, ei[::-1], np.asarray(ea, dtype=np.int64))
    with torch.no_grad():
        a = model(sro.batch_objects([p1]), lambda_override=0.5).item()
        b = model(sro.batch_objects([p2]), lambda_override=0.5).item()
        c = model(sro.batch_objects([p3]), lambda_override=0.5).item()
    assert abs(a - b) < 1e-5
    assert abs(a - c) < 1e-5

    # junction row permutation
    rng = np.random.default_rng(0)
    o = objs[0]
    order = rng.permutation(o.junc_index.shape[0])
    perm_obj = sro.MolObjects(
        mol_index=o.mol_index,
        smiles=o.smiles,
        n_atoms=o.n_atoms,
        atom_feat=o.atom_feat,
        bond_index=o.bond_index,
        bond_type=o.bond_type,
        bond_feat=o.bond_feat,
        junc_index=o.junc_index[order],
        junc_bond=o.junc_bond[order],
        junc_feat=o.junc_feat[order],
        global_feat=o.global_feat,
    )
    with torch.no_grad():
        a0 = model(sro.batch_objects([o]), lambda_override=0.5).item()
        a1 = model(sro.batch_objects([perm_obj]), lambda_override=0.5).item()
    assert abs(a0 - a1) < 1e-5


def test_glrc_ownership_counterexample():
    _, model, _ = ab.build_glrc_from_init(0, NORM, "cpu", None)
    base = bp.build_bjoc_null(0, NORM, "cpu")
    inst_a, inst_b = src.counterexample_object_pair()
    assert src.object_marginal_signature(inst_a) == src.object_marginal_signature(inst_b)
    assert src.per_bond_junction_tokens(inst_a) != src.per_bond_junction_tokens(inst_b)
    batch = src.batch_object_instances([inst_a, inst_b])
    with torch.no_grad():
        b_pred = base(batch)
        g0 = model(batch, lambda_override=0.0)
        g1 = model(batch, lambda_override=1.0)
        feats = model.forward_features(batch, lambda_override=1.0)
    assert float(abs(b_pred[0] - b_pred[1])) < 1e-5
    assert float(abs(g0[0] - g0[1])) < 1e-5
    assert float(abs(g1[0] - g1[1])) > 0.0
    # the local residual differs although the null statistics are identical
    assert float((feats["dW"][:4] - feats["dW"][4:]).abs().max()) > 1e-6
    assert float((feats["dW"][:4] - feats["dW"][4:]).abs().max()) > 0.0


def test_glrc_gate_gradient_and_activation():
    _, model, _ = ab.build_glrc_from_init(0, NORM, "cpu", None)
    objs = _small_objects()
    batch = sro.batch_objects(objs)
    y = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    for _ in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(model(batch).view(-1), y)
        loss.backward()
        assert model.lambda_gate.weight.grad is not None
        assert float(model.lambda_gate.weight.grad.abs().max()) > 0.0
        opt.step()
    with torch.no_grad():
        active = model(batch)
        zero = model(batch, lambda_override=0.0)
    assert float((active - zero).abs().max()) > 0.0
    assert float(model.lambda_gate.bias.detach().abs().max()) > 0.0


# ---------------------------------------------------------------------------
# DLP
# ---------------------------------------------------------------------------


def test_dlp_pair_construction_and_buckets():
    objs = _small_objects()
    cache = ab.build_pair_cache(objs)
    assert set(cache) == {o.mol_index for o in objs}
    summary = ab.pair_cache_summary(cache)
    assert summary["n_molecules"] == len(objs)
    for o in objs:
        rec = ab.pair_arrays_for_object(o)
        adj = [[] for _ in range(int(o.n_atoms))]
        for (u, v), bt in zip(np.asarray(o.bond_index), np.asarray(o.bond_type)):
            adj[int(u)].append((int(v), int(bt)))
            adj[int(v)].append((int(u), int(bt)))
        dist = sro._all_pairs_distances(int(o.n_atoms), adj)
        for i, j, b in zip(rec["i"], rec["j"], rec["bucket"]):
            assert int(i) < int(j)
            d = int(dist[int(i), int(j)])
            assert d >= 3
            assert int(b) == (0 if d == 3 else (1 if d == 4 else 2))


def test_dlp_parameter_budget_identity_and_gradient_order():
    base, model, info = ab.build_dlp_from_init(0, NORM, "cpu", None)
    assert sum(p.numel() for p in model.parameters()) == ab.DLP_PARAMETERS
    assert info["new_parameters"] == 27648
    assert ab.base_parameter_hash(base, exclude_prefixes=("reader.lins.0.weight",)) == ab.base_parameter_hash(
        model, exclude_prefixes=("pair_proj.", "reader.lins.0.weight")
    )
    objs = _small_objects()
    pairs = ab.build_pair_cache(objs)
    batch = ab.build_dlp_batch(objs, pairs)
    with torch.no_grad():
        yb = base(batch)
        yd = model(batch)
        base_reader_in = int(src.BJOCConfig().reader_in)
        new_cols = model.reader.lins[0].weight[:, base_reader_in:]
        old_cols = model.reader.lins[0].weight[:, :base_reader_in]
    assert float((yb - yd).abs().max()) < 1e-6
    assert float(new_cols.abs().max()) == 0.0
    assert torch.equal(old_cols, base.reader.lins[0].weight)

    y = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    grads = []
    for _ in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(model(batch).view(-1), y)
        loss.backward()
        grads.append(
            (
                float((model.reader.lins[0].weight.grad[:, int(src.BJOCConfig().reader_in):] != 0).float().mean()),
                float((model.pair_proj.weight.grad != 0).float().mean()),
            )
        )
        opt.step()
    assert grads[0][0] > 0.0
    assert grads[0][1] == 0.0  # zero reader columns block the projection at step 0
    assert grads[1][1] > 0.0  # projection becomes trainable from step 1 on


def test_dlp_information_witness():
    w = ab.pair_feature_partition_witness()
    assert w["first_second_moments_equal"]
    assert w["different_pair_features"]
    assert w["max_abs_pair_feature_diff"] > 1e-6


def test_dlp_invariance():
    _, model, _ = ab.build_dlp_from_init(0, NORM, "cpu", None)
    objs = _small_objects()
    pairs = ab.build_pair_cache(objs)
    batch = ab.build_dlp_batch(objs, pairs)
    with torch.no_grad():
        y0 = model(batch).numpy()
        y_single = np.asarray([model(ab.build_dlp_batch([o], pairs)).item() for o in objs])
    assert np.abs(y0 - y_single).max() < 1e-5
    flipped = {k: {"i": v["i"][::-1].copy(), "j": v["j"][::-1].copy(), "bucket": v["bucket"][::-1].copy()} for k, v in pairs.items()}
    with torch.no_grad():
        y1 = model(ab.build_dlp_batch(objs, flipped)).numpy()
        feats = model.forward_features(batch)
    assert np.abs(y0 - y1).max() < 1e-5
    assert torch.isfinite(feats["dlp"]).all()
    assert feats["dlp"].shape[0] == len(objs)


def test_dlp_recoverability_helper():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8))
    target = x @ rng.normal(size=(8, 3)) + 0.05 * rng.normal(size=(200, 3))
    rec = ab.linear_recoverability(x, target, seed=0)
    assert rec["test_r2_median"] > 0.8  # linearly recoverable by construction
    target2 = rng.normal(size=(200, 3))
    rec2 = ab.linear_recoverability(x, target2, seed=0)
    assert rec2["test_r2_median"] < 0.8


# ---------------------------------------------------------------------------
# frozen training loop equivalence
# ---------------------------------------------------------------------------


def test_training_loop_matches_frozen_loop():
    objs = _small_objects()
    y = np.asarray([0.1, -0.2, 0.3, 0.4])
    p1 = bp.train_supervised(
        arm="P3",
        seed=0,
        recipe_key="R0",
        init="scratch",
        train_objs=objs,
        rows=[0, 1, 2, 3],
        y_fit=y,
        norm=NORM,
        device=torch.device("cpu"),
        model_kind="base",
        smoke_epochs=2,
    )
    p2 = ab.train_supervised_arch(
        arm="P3",
        seed=0,
        recipe_key="R0",
        init="scratch",
        train_objs=objs,
        rows=[0, 1, 2, 3],
        y_fit=y,
        norm=NORM,
        device=torch.device("cpu"),
        model_kind="base",
        smoke_epochs=2,
    )
    assert p1["batch_order"]["full_hash"] == p2["batch_order"]["full_hash"]
    assert abs(p1["soup_fit_mae"] - p2["soup_fit_mae"]) < 1e-12
    for key in p1["soup_state"]:
        assert torch.equal(p1["soup_state"][key], p2["soup_state"][key]), key


def test_arch_loop_smoke_for_candidates():
    objs = _small_objects()
    pairs = ab.build_pair_cache(objs)
    y = np.asarray([0.1, -0.2, 0.3, 0.4])
    p = ab.train_supervised_arch(
        arm="GLRC",
        seed=0,
        recipe_key="R0",
        init="scratch",
        train_objs=objs,
        rows=[0, 1, 2, 3],
        y_fit=y,
        norm=NORM,
        device=torch.device("cpu"),
        model_kind="glrc",
        smoke_epochs=2,
    )
    assert p["parameter_audit"]["actual_parameters"] == ab.GLRC_PARAMETERS
    assert p["batch_order"]["full_matches"]
    p = ab.train_supervised_arch(
        arm="DLP",
        seed=0,
        recipe_key="R0",
        init="scratch",
        train_objs=objs,
        rows=[0, 1, 2, 3],
        y_fit=y,
        norm=NORM,
        device=torch.device("cpu"),
        model_kind="dlp",
        pairs_by_index=pairs,
        smoke_epochs=2,
    )
    assert p["parameter_audit"]["actual_parameters"] == ab.DLP_PARAMETERS


# ---------------------------------------------------------------------------
# contrast / gate arithmetic
# ---------------------------------------------------------------------------


def test_compute_contrast_and_monitor_gate():
    y = np.linspace(-4.0, 4.0, 200)
    ref = {0: y + 0.10, 1: y + 0.12}
    new = {0: y + 0.10 - 0.005, 1: y + 0.12 - 0.006}
    c = ab.compute_contrast(new, ref, y)
    assert abs(c["delta_seed_mean"] + 0.0055) < 1e-9
    assert c["per_seed_delta"]["0"] < 0 and c["per_seed_delta"]["1"] < 0
    gate = ab.monitor_gate(c, two_seed=True)
    assert gate["passed"]
    # missing two-seed condition fails
    c2 = ab.compute_contrast({0: y + 0.10 - 0.005, 1: y + 0.12 + 0.001}, ref, y)
    assert not ab.monitor_gate(c2, two_seed=True)["passed"]
    # negative direction wrong
    c3 = ab.compute_contrast({0: y + 0.2, 1: y + 0.2}, ref, y)
    assert not ab.monitor_gate(c3, two_seed=True)["passed"]


def test_single_row_dominance_fraction():
    y = np.zeros(100)
    ref = {0: np.ones(100)}  # every reference prediction has unit error
    new = np.ones(100)
    new[0] = 0.001  # one molecule carries the entire improvement
    c = ab.compute_contrast({0: new}, ref, y)
    assert c["max_single_gain_fraction"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# driver discipline
# ---------------------------------------------------------------------------


def test_driver_has_no_official_test_path():
    text = (Path(_CODE_DIR) / "run_bjoc_architecture_breakthrough_v1.py").read_text(encoding="utf-8")
    for token in ("encoded_test", "load_official_test", "ZINC(root", 'split="test"', "test_read"):
        assert token not in text, token
    assert "official_test" in text
    assert text.count('"official_test_loaded": False') >= 3


def test_pretraining_module_never_reads_labels():
    path = Path(_CODE_DIR) / "bjoc_arch_breakthrough_v1.py"
    text = path.read_text(encoding="utf-8")
    assert "load_y_rows" not in text.split("def pair_arrays_for_object", 1)[0]
    assert 'split["monitor_rows"] + split["locked_rows"]' in (Path(_CODE_DIR) / "run_bjoc_architecture_breakthrough_v1.py").read_text(encoding="utf-8")


def test_frozen_input_hashes_declared():
    cfg = _cfg()["frozen"]
    assert len(cfg["canonical_smiles_sha256"]) == 64
    assert len(cfg["encoded_train_sha256"]) == 64
    assert cfg["expected_parameters"] == 413505
    assert set(cfg["p_init_full_sha256"]) == {"0", "1"}
