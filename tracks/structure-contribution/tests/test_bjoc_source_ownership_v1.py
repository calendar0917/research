"""Fast correctness tests for cscl-bjoc-source-ownership-v1.

No ZINC dataset, no checkpoints, no GPU: tiny synthetic molecules only.
Covers Phase-1 model mathematics (permutation invariance, ownership
isolation, global-pathway separation, arm parity), the sham-arm invariants,
the deterministic train loop, Phase-0 primitives and the label discipline.
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

import bjoc_source_ownership_v1 as own  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/bjoc_source_ownership_v1.yaml"
NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _mol(idx, atom_types, bonds):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(idx, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))


def _tiny_world(n=6):
    specs = [
        ([0, 1, 2, 3], [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 1)]),          # 4-ring
        ([4, 5, 6], [(0, 1, 1), (1, 2, 3)]),                                   # path
        ([7, 7, 8, 9, 10], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 1, 1)]),
        ([1, 2, 3, 4, 5], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 0, 1)]),
        ([2, 2, 3], [(0, 1, 2), (1, 2, 1)]),
        ([0, 1, 2, 3, 4], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 0, 2)]),
    ]
    return [_mol(i, specs[i % len(specs)][0], specs[i % len(specs)][1]) for i in range(n)]


def _build(mode, seed=0, ssl_state=None):
    base, model, info = own.build_ownership_from_init(seed, NORM, "cpu", mode, ssl_state)
    return base, model, info


def _batch(objs, seed=0):
    keys = own.molecule_keys(objs)
    return own.batch_objects_with_keys(objs, keys)


def _permute_bonds(batch, perm):
    out = own.clone_batch(batch)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    for key in ("bond_feat", "bond_batch", "bond_index"):
        out[key] = batch[key][perm]
    out["junc_bond"] = inv[batch["junc_bond"]]
    return out


def _permute_junctions(batch, perm):
    out = own.clone_batch(batch)
    for key in ("junc_feat", "junc_batch", "junc_index", "junc_bond"):
        out[key] = batch[key][perm]
    return out


# ---------------------------------------------------------------------------
# arm parity
# ---------------------------------------------------------------------------


def test_arm_registry_and_parameter_parity():
    assert own.MODES == ("N", "O", "S")
    objs = _tiny_world(4)
    hashes = {}
    params = {}
    for mode in own.MODES:
        _, model, info = _build(mode)
        hashes[mode] = own.model_state_hash(model)
        params[mode] = int(sum(p.numel() for p in model.parameters()))
        assert params[mode] == own.OWNERSHIP_EXPECTED_PARAMETERS
        assert info["added_parameters_vs_p3"] == own.OWNERSHIP_EXPECTED_PARAMETERS - bp.EXPECTED_PARAMETERS
    assert len(set(hashes.values())) == 1, hashes
    assert len(set(params.values())) == 1


def test_init_parity_with_p3_is_black_box_exact():
    """N degenerates to P3 at init; the added global block contributes exactly zero.

    O/S legitimately change the *grouping* function (that is the experiment),
    so their init parity claim is only that the new pathway itself is inert.
    """
    objs = _tiny_world(4)
    batch = _batch(objs)
    base, model_n, _ = _build(own.MODE_N)
    with torch.no_grad():
        p_base = base(batch)
        p_n = model_n(batch)
    assert float((p_base - p_n).abs().max()) < 1e-6
    zeros_p = torch.zeros((int(batch["n_graphs"]), own.GLOBAL_CONTENT_OUT))
    for mode in own.MODES:
        _, model, _ = _build(mode)
        with torch.no_grad():
            p0 = model(batch)
            p1 = model(batch, p_override=zeros_p)
        assert torch.equal(p0, p1), mode


def test_n_and_o_delegate_to_frozen_groupings():
    objs = _tiny_world(3)
    batch = _batch(objs)
    _, model, _ = _build(own.MODE_N)
    with torch.no_grad():
        w = model.encode_junctions(model.encode_atoms(batch), model.encode_bonds(model.encode_atoms(batch), batch), batch)
        Wn, Vn, nn = model.group_to_bonds_ownership(w, batch, own.MODE_N)
        Wn_ref, Vn_ref, nn_ref = src.BJOC.group_to_bonds(model, w, batch, src.NULL)
        Wo, Vo, no = model.group_to_bonds_ownership(w, batch, own.MODE_O)
        Wo_ref, Vo_ref, no_ref = src.BJOC.group_to_bonds(model, w, batch, src.ORGANIZED)
    assert torch.equal(Wn, Wn_ref) and torch.equal(Vn, Vn_ref) and torch.equal(nn, nn_ref)
    assert torch.equal(Wo, Wo_ref) and torch.equal(Vo, Vo_ref) and torch.equal(no, no_ref)


# ---------------------------------------------------------------------------
# permutation invariance
# ---------------------------------------------------------------------------


def test_atom_relabel_invariance_all_modes():
    objs = _tiny_world(3)
    rng = np.random.default_rng(11)
    specs = []
    for obj in objs:
        n = obj.n_atoms
        perm = rng.permutation(n)
        types = [int(np.argmax(obj.atom_feat[i, : sro.ATOM_CATEGORIES])) for i in range(n)]
        bonds = [(int(u), int(v), int(obj.bond_type[b])) for b, (u, v) in enumerate(obj.bond_index)]
        rel_types = [0] * n
        for i in range(n):
            rel_types[int(perm[i])] = types[i]
        rel_bonds = [(int(perm[u]), int(perm[v]), bt) for u, v, bt in bonds]
        specs.append(_mol(obj.mol_index, rel_types, rel_bonds))
    assert [o.smiles for o in objs] == [o.smiles for o in specs]
    for mode in own.MODES:
        _, model, _ = _build(mode)
        with torch.no_grad():
            p0 = model(_batch(objs))
            p1 = model(_batch(specs))
        assert float((p0 - p1).abs().max()) < 1e-5, (mode, float((p0 - p1).abs().max()))


def test_bond_order_and_junction_permutation_invariance():
    objs = _tiny_world(3)
    batch = _batch(objs)
    rng = np.random.default_rng(3)
    bond_perm = torch.as_tensor(rng.permutation(batch["bond_feat"].shape[0]), dtype=torch.long)
    junc_perm = torch.as_tensor(rng.permutation(batch["junc_feat"].shape[0]), dtype=torch.long)
    for mode in own.MODES:
        _, model, _ = _build(mode)
        with torch.no_grad():
            p0 = model(batch)
            p_bond = model(_permute_bonds(batch, bond_perm))
            p_junc = model(_permute_junctions(batch, junc_perm))
        assert float((p0 - p_bond).abs().max()) < 1e-5, (mode, "bond")
        assert float((p0 - p_junc).abs().max()) < 1e-5, (mode, "junc")


def test_single_vs_batch_parity():
    objs = _tiny_world(3)
    for mode in own.MODES:
        _, model, _ = _build(mode)
        with torch.no_grad():
            p_batch = model(_batch(objs))
            p_single = torch.cat([model(_batch([o])) for o in objs])
        assert float((p_batch - p_single).abs().max()) < 1e-6


def test_save_load_replay(tmp_path):
    objs = _tiny_world(3)
    batch = _batch(objs)
    _, model, _ = _build(own.MODE_O)
    with torch.no_grad():
        p0 = model(batch)
    path = tmp_path / "n.pt"
    torch.save({k: v for k, v in model.state_dict().items()}, path)
    _, other, _ = _build(own.MODE_O)
    other.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    with torch.no_grad():
        p1 = other(batch)
    assert torch.equal(p0, p1)


# ---------------------------------------------------------------------------
# ownership isolation
# ---------------------------------------------------------------------------


def test_o_ownership_edit_changes_only_incident_bonds():
    obj = _mol(0, [0, 1, 2, 3, 0, 1], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 1)])
    batch = _batch([obj])
    _, model, _ = _build(own.MODE_O)
    # rewrite junction 0's incidence to a different bond pair of the same molecule
    edit = own.clone_batch(batch)
    old = batch["junc_bond"][0].clone()
    new = torch.as_tensor([1, 3], dtype=torch.long)
    touched = sorted(set(old.tolist()) | set(new.tolist()))
    edit["junc_bond"][0] = new
    with torch.no_grad():
        c0 = model.forward_features(batch)["c"]
        c1 = model.forward_features(edit)["c"]
    for e in range(c0.shape[0]):
        if e in touched:
            continue
        assert torch.equal(c0[e], c1[e]), f"bond {e} changed outside the edited incidence"
    assert any(not torch.equal(c0[e], c1[e]) for e in touched)


def test_n_is_ownership_blind_but_o_is_not():
    """Cyclic rotation of a C4 grouping ignores which junction touches which bond.

    Held at the *grouping* level with fixed ``w``: the Junction encoder itself
    legitimately consumes the true incidence (that is part of ``w``), so the
    ownership-blindness claim of N is about the Junction->Bond aggregation.
    """
    obj = _mol(0, [0, 1, 2, 3], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)])
    batch = _batch([obj])
    rotated = own.clone_batch(batch)
    rotated["junc_bond"] = torch.roll(batch["junc_bond"], shifts=1, dims=0)

    def counts(b):
        n_e = torch.zeros(b["bond_feat"].shape[0], dtype=torch.float32)
        n_e.index_add_(0, b["junc_bond"].reshape(-1), torch.ones(b["junc_bond"].numel()))
        return n_e

    assert torch.equal(counts(batch), counts(rotated))
    for mode, changed in ((own.MODE_N, False), (own.MODE_O, True)):
        _, model, _ = _build(mode)
        with torch.no_grad():
            a = model.encode_atoms(batch)
            b = model.encode_bonds(a, batch)
            w = model.encode_junctions(a, b, batch)
            W0, V0, n0 = model.group_to_bonds_ownership(w, batch, mode)
            W1, V1, n1 = model.group_to_bonds_ownership(w, rotated, mode)
        assert torch.equal(n0, n1)
        if changed:
            assert not torch.equal(W0, W1), mode
        else:
            assert torch.equal(W0, W1), mode


def test_n_mixes_junction_content_globally_but_o_does_not():
    objs = _tiny_world(1)
    batch = _batch(objs)
    _, model_n, _ = _build(own.MODE_N)
    _, model_o, _ = _build(own.MODE_O)
    edited = own.clone_batch(batch)
    edited["junc_feat"] = batch["junc_feat"].clone()
    edited["junc_feat"][0, 0] += 1.0  # perturb one junction encoding input
    for mode in (own.MODE_N, own.MODE_O):
        model = model_n if mode == own.MODE_N else model_o
        with torch.no_grad():
            f0 = model.forward_features(batch)
            f1 = model.forward_features(edited)
        diff_bonds = (f0["c"] - f1["c"]).abs().sum(1) > 0
        if mode == own.MODE_N:
            assert bool(diff_bonds.all()), "N must mix every junction into every bond"
        else:
            incident = set(batch["junc_bond"][0].tolist())
            assert set(int(i) for i in torch.nonzero(diff_bonds).reshape(-1)) <= incident


# ---------------------------------------------------------------------------
# global content pathway separation
# ---------------------------------------------------------------------------


def test_global_block_never_touches_local_pathway():
    objs = _tiny_world(3)
    batch = _batch(objs)
    for mode in own.MODES:
        _, model, _ = _build(mode)
        with torch.no_grad():
            f0 = model.forward_features(batch)
            rnd = torch.randn_like(f0["p_global"])
            f1 = model.forward_features(batch, p_override=rnd)
        assert torch.equal(f0["c"], f1["c"])
        assert torch.equal(f0["z"], f1["z"])
        assert torch.equal(f0["W"], f1["W"])
        # only the appended global block of reader_in changes
        diff = (f0["reader_in"] - f1["reader_in"]).abs() > 0
        assert bool(diff[:, :321].sum() == 0)
        assert bool(diff[:, 321:].sum() > 0)


def test_global_content_parameters_do_not_affect_c():
    objs = _tiny_world(3)
    batch = _batch(objs)
    _, model, _ = _build(own.MODE_O)
    with torch.no_grad():
        c0 = model.forward_features(batch)["c"]
        for p in model.global_content.parameters():
            p.add_(torch.randn_like(p))
        c1 = model.forward_features(batch)["c"]
    assert torch.equal(c0, c1)


def test_zero_local_flag_zeroes_only_z():
    objs = _tiny_world(3)
    batch = _batch(objs)
    _, model, _ = _build(own.MODE_O)
    with torch.no_grad():
        f0 = model.forward_features(batch)
        f1 = model.forward_features(batch, zero_local=True)
    assert torch.equal(f1["z"], torch.zeros_like(f1["z"]))
    assert torch.equal(f0["c"], f1["c"])


# ---------------------------------------------------------------------------
# sham arm invariants
# ---------------------------------------------------------------------------


def test_sham_preserves_counts_and_format():
    objs = _tiny_world(5)
    batch = _batch(objs)
    _, model_s, _ = _build(own.MODE_S)
    _, model_o, _ = _build(own.MODE_O)
    with torch.no_grad():
        a = model_s.encode_atoms(batch)
        b = model_s.encode_bonds(a, batch)
        w = model_s.encode_junctions(a, b, batch)
        Ws, Vs, ns = model_s.group_to_bonds_ownership(w, batch, own.MODE_S)
        Wo, Vo, no = model_o.group_to_bonds_ownership(w, batch, own.MODE_O)
    assert torch.equal(ns, no)
    # total slot count equals 2 * n_junctions
    assert float(ns.sum()) == 2.0 * float(batch["junc_feat"].shape[0])
    # deterministic replay
    with torch.no_grad():
        Ws2, _, _ = model_s.group_to_bonds_ownership(w, batch, own.MODE_S)
    assert torch.equal(Ws, Ws2)
    # some bond must differ from the true ownership aggregate
    assert not torch.equal(Ws, Wo)
    # zero-count bonds receive nothing
    assert float(Ws[ns == 0].abs().sum()) == 0.0


def test_sham_keys_are_collision_free_on_test_world():
    objs = _tiny_world(8)
    batch = _batch(objs)
    _, model, _ = _build(own.MODE_S)
    with torch.no_grad():
        a = model.encode_atoms(batch)
        b = model.encode_bonds(a, batch)
        w = model.encode_junctions(a, b, batch)
        jkey = model._junction_key(batch["junc_feat"], w)
        n_e = torch.zeros(batch["bond_feat"].shape[0], dtype=torch.float32)
        n_e.index_add_(0, batch["junc_bond"].reshape(-1), torch.ones(batch["junc_bond"].numel()))
        bkey = model._bond_key(
            batch["bond_feat"],
            batch["atom_feat"][batch["bond_index"][:, 0]],
            batch["atom_feat"][batch["bond_index"][:, 1]],
            n_e,
        )
    desc_j = torch.cat([batch["junc_feat"], w], dim=1)
    desc_b = model.bond_descriptor(batch, n_e)
    for g in range(int(batch["n_graphs"])):
        jm = batch["junc_batch"] == g
        keys = jkey[jm]
        if keys.numel() >= 2:
            order = torch.argsort(keys)
            for u, v in zip(order[:-1].tolist(), order[1:].tolist()):
                if torch.equal(keys[u], keys[v]):
                    assert torch.equal(desc_j[jm][u], desc_j[jm][v]), "lossless-key collision"
    for u in range(bkey.numel()):
        for v in range(u + 1, bkey.numel()):
            if torch.equal(bkey[u], bkey[v]):
                assert torch.equal(desc_b[u], desc_b[v]), "lossless bond-key collision"


# ---------------------------------------------------------------------------
# training loop determinism
# ---------------------------------------------------------------------------


def test_train_loop_deterministic_smoke_and_batch_order():
    objs = _tiny_world(4)
    keys = own.molecule_keys(objs)
    y = np.asarray([0.5, -1.0, 2.0, -0.25], dtype=np.float64)
    _, donor, _ = _build(own.MODE_N, seed=0)
    ssl_state = {
        "atom_encoder": donor.atom_encoder.state_dict(),
        "bond_encoder": donor.bond_encoder.state_dict(),
        "junction_encoder": donor.junction_encoder.state_dict(),
    }
    out = {}
    for tag in ("a", "b"):
        payload = own.train_supervised_ownership(
            arm="N",
            seed=0,
            recipe_key="R3",
            train_objs=objs,
            rows=list(range(len(objs))),
            y_fit=y,
            norm=NORM,
            device=torch.device("cpu"),
            ssl_state=ssl_state,
            smoke_epochs=2,
            log_fn=lambda m: None,
        )
        out[tag] = payload
    assert out["a"]["batch_order"]["full_matches"]
    assert out["a"]["batch_order"]["full_hash"] == out["b"]["batch_order"]["full_hash"]
    ha = bp.state_group_sha256(out["a"]["soup_state"])
    hb = bp.state_group_sha256(out["b"]["soup_state"])
    assert ha == hb
    assert np.isfinite(out["a"]["soup_fit_mae"])
    assert out["a"]["parameter_audit"]["parameter_exact"]


# ---------------------------------------------------------------------------
# Phase-0 primitives
# ---------------------------------------------------------------------------


def test_signature_is_deterministic_and_ring_aware():
    cyclic = _mol(0, [0, 0, 0, 0], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)])
    chain = _mol(1, [0, 0, 0, 0], [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
    s_cyc = own.junction_signature(cyclic, 0)
    s_chain = own.junction_signature(chain, 0)
    assert s_cyc[3] == 1 and s_cyc[4] == 2
    assert s_chain[3] == 0 and s_chain[4] == 0
    assert s_cyc != s_chain
    counts = own.signature_corpus_counts([cyclic, chain])
    assert counts[s_cyc] == 4
    assert counts[s_chain] >= 1


def test_rankings_and_rarity_deterministic():
    objs = _tiny_world(3)
    counts = own.signature_corpus_counts(objs)
    total = sum(counts.values())
    keys = own.molecule_keys(objs)
    _, model, _ = _build(own.MODE_N)
    batch = own.batch_objects_with_keys(objs, keys)
    with torch.no_grad():
        w = model.encode_junctions(model.encode_atoms(batch), model.encode_bonds(model.encode_atoms(batch), batch), batch)
    dev = own.dev_ranking(w)
    assert dev.shape[0] == w.shape[0]
    r0 = own.rarity_ranking(objs[0], counts, total)
    r1 = own.rarity_ranking(objs[0], counts, total)
    assert np.array_equal(r0, r1)
    assert np.all(np.isfinite(r0))


def test_matched_control_respects_structure_and_dev():
    obj = _tiny_world(1)[0]
    counts = own.signature_corpus_counts([obj])
    total = sum(counts.values())
    _, model, _ = _build(own.MODE_N)
    batch = own.batch_objects_with_keys([obj], own.molecule_keys([obj]))
    with torch.no_grad():
        w = model.encode_junctions(model.encode_atoms(batch), model.encode_bonds(model.encode_atoms(batch), batch), batch)
    dev = own.dev_ranking(w).numpy()
    sig = own.junction_signature(obj, 0)
    deg = own.center_degree(obj, 0)
    ctrl = own.matched_controls(obj, 0, dev, sig, deg, sig[3], {0}, 4, np.random.default_rng(0))
    for j in ctrl:
        assert own.junction_signature(obj, j) == sig
        assert own.center_degree(obj, j) == deg


def test_forbidden_module_scan_clean():
    _, model, _ = _build(own.MODE_O)
    assert own.forbidden_module_scan(model) == []


# ---------------------------------------------------------------------------
# discipline
# ---------------------------------------------------------------------------


def test_config_freeze_words_and_budget():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == own.PROTOCOL_ID
    assert cfg["training"]["epochs"] == 360
    assert cfg["training"]["gpu_budget_cap_hours"] == 1.5
    assert cfg["phase0"]["top_k"] == 4
    assert cfg["split"]["locked_status"].startswith("NEVER READ")
    assert cfg["split"]["official_test_status"].lower().startswith("never loaded")
    assert set(cfg["training"]["seed0_arms"]) == {"N", "O", "S"}


def test_no_official_test_or_locked_loader_tokens():
    src_text = (REPO_ROOT / "tracks/structure-contribution/code/bjoc_source_ownership_v1.py").read_text(encoding="utf-8")
    assert "objects_test" not in src_text
    assert "official_test" not in src_text or "official_test_loaded" in src_text


def test_runner_exposes_frozen_commands_and_imports_clean():
    import run_bjoc_source_ownership_v1 as runner

    parser = runner.build_parser()
    commands = {a.dest: a.choices for a in parser._actions if a.dest == "cmd"}[ "cmd"]
    for name in ("verify-inputs", "freeze", "phase0", "cpu-gates", "p3-replay", "arm", "predict", "score-monitor", "mechanism", "valid-secondary", "finalize"):
        assert name in commands
    assert runner.PHASE0_ALPHA == 1.0
    assert runner.BOOTSTRAP_N == 5000 and runner.TOP_K_REMOVED == 10


def test_runner_prediction_paths_are_frozen_and_locked_never_read():
    import run_bjoc_source_ownership_v1 as runner

    assert runner.monitor_pred_path("O", 0).name == "monitor_O_D_s0.npz"
    assert runner.soup_path("N", 1).name == "soup_N_D_s1.pt"
    text = (REPO_ROOT / "tracks/structure-contribution/code/run_bjoc_source_ownership_v1.py").read_text(encoding="utf-8")
    assert "objects_test" not in text
    assert "locked_y" not in text
    assert "monitor_read.lock" in text


def test_sham_vectorized_grouping_matches_per_graph_reference():
    """The vectorised S grouping must reproduce the naive per-graph draw exactly."""
    _, model, _ = _build(own.MODE_S)
    objs = _tiny_world(8)
    batch = _batch(objs)
    with torch.no_grad():
        a = model.encode_atoms(batch)
        b = model.encode_bonds(a, batch)
        w = model.encode_junctions(a, b, batch)
        W1, V1, n1 = model.group_to_bonds_ownership(w, batch, own.MODE_S)
        junc_bond, bond_batch, junc_batch = batch["junc_bond"], batch["bond_batch"], batch["junc_batch"]
        n_bonds = int(batch["bond_feat"].shape[0])
        n_e = torch.zeros(n_bonds)
        n_e.index_add_(0, junc_bond.reshape(-1), torch.ones(junc_bond.numel()))
        jkey = model._junction_key(batch["junc_feat"], w)
        idx = batch["bond_index"]
        au = batch["atom_feat"][idx[:, 0]]
        av = batch["atom_feat"][idx[:, 1]]
        bkey = model._bond_key(batch["bond_feat"], au, av, n_e).view(torch.int64)
        W = torch.zeros_like(W1)
        V = torch.zeros_like(V1)
        for g in range(int(batch["n_graphs"])):
            jm = junc_batch == g
            bm = bond_batch == g
            nj = int(jm.sum())
            if nj == 0:
                continue
            bg = torch.nonzero(bm).reshape(-1)
            ne = n_e[bg].to(torch.int64)
            if int(ne.sum()) == 0:
                continue
            wcan = w[jm][torch.argsort(jkey[jm], stable=True)]
            e_idx = torch.repeat_interleave(bg, ne)
            k_idx = torch.cat([torch.arange(int(m)) for m in ne.tolist()])
            h = (batch["mol_key"][g] * own.MIX_MUL_A) ^ (bkey[e_idx] * own.MIX_MUL_B) ^ (k_idx * 2654435761)
            h = torch.bitwise_and(h, own.MIX_MASK)
            r = torch.remainder(h, nj).to(torch.int64)
            drawn = wcan[r]
            W.index_add_(0, e_idx, drawn)
            V.index_add_(0, e_idx, drawn * drawn)
    assert torch.equal(W1, W)
    assert torch.equal(V1, V)
    assert torch.equal(n1, n_e)
