"""CPU tests for CSS-v0 (descriptors, fit-only sharing, model parity, loop)."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import css_common as C
import css_data as D
import css_descriptors as DD
import css_fit as F
import css_model as M
import css_train as TR
import gwic_model as GM
import ses_data as SD
import ses_structure as SS


# ---------------------------------------------------------------------------
# fixed split
# ---------------------------------------------------------------------------


def test_fit_hash_split():
    z = np.load(C.RAW_CACHE)
    train = np.sort(np.asarray(z["train_idx"], np.int64))
    valid = np.sort(np.asarray(z["valid_idx"], np.int64))
    test = np.sort(np.asarray(z["test_idx"], np.int64))
    assert (train.size, valid.size, test.size) == (3360, 420, 420)
    assert np.intersect1d(train, valid).size == 0
    assert np.intersect1d(train, test).size == 0
    assert np.intersect1d(valid, test).size == 0
    assert np.union1d(np.union1d(train, valid), test).size == 4200
    fit, held = D.fit_rows_for(50, train)
    fit2, held2 = D.fit_rows_for(50, train)
    assert np.array_equal(fit, fit2) and np.array_equal(held, held2)
    assert fit.size == held.size == 1680
    assert np.intersect1d(fit, held).size == 0
    assert np.array_equal(np.union1d(fit, held), train)
    assert np.isin(fit, train).all()
    # hash selection is label-free: it only reads row ids
    assert D.selection_hash(int(train[0])) == D.selection_hash(int(train[0]))


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------


def _chain():
    atom_feat = np.zeros((4, 9), np.int64)
    atom_feat[:, 0] = [0, 1, 2, 3]
    bond_feat = np.zeros((3, 3), np.int64)
    bond_feat[:, 0] = [0, 1, 0]
    bu = np.asarray([0, 1, 2], np.int64)
    bv = np.asarray([1, 2, 3], np.int64)
    return atom_feat, bond_feat, bu, bv


def test_descriptor_hand_counts_and_normalisation():
    atom_feat, bond_feat, bu, bv = _chain()
    keys, counts, ptr = DD.molecule_descriptors(atom_feat, bond_feat, bu, bv)
    n_roots = 4
    totals = DD.block_totals(keys, counts, ptr, n_roots)
    v = 1
    lo, hi = int(ptr[v]), int(ptr[v + 1])
    block = DD.block_of(keys[lo:hi])
    counts_v = counts[lo:hi]
    block_sums = np.asarray([counts_v[block == b].sum() for b in range(DD.N_BLOCKS)])
    # ROOT 9 fields; S1 two neighbours; S2 one path 1-2-3; EGO_ATOM 9 fields x 4
    # R2 atoms; EGO_BOND 3 fields x 3 induced bonds
    assert block_sums.tolist() == [9, 2, 1, 36, 9]
    assert totals[v].tolist() == block_sums.tolist()

    values = DD.normalised_values(keys, counts, ptr, n_roots)
    row = np.zeros(len(DD.BLOCK_NAMES))
    for b in range(DD.N_BLOCKS):
        row[b] = (values[lo:hi][block == b] ** 2).sum()
    assert np.allclose(row, 1.0 / DD.N_BLOCKS, atol=1e-12)

    # real multiplicity is preserved: duplicate the first neighbour environment
    atom_feat2, bond_feat2, bu2, bv2 = _chain()
    # add a second atom of type 0 bonded with bond type 0 to atom 1
    atom_feat2 = np.vstack([atom_feat2, np.zeros((1, 9), np.int64)])
    bond_feat2 = np.vstack([bond_feat2, np.zeros((1, 3), np.int64)])
    bu2 = np.append(bu2, 1)
    bv2 = np.append(bv2, 4)
    keys2, counts2, ptr2 = DD.molecule_descriptors(atom_feat2, bond_feat2, bu2, bv2)
    lo2, hi2 = int(ptr2[1]), int(ptr2[2])
    s1 = keys2[lo2:hi2][DD.block_of(keys2[lo2:hi2]) == DD.S1]
    s1_counts = counts2[lo2:hi2][DD.block_of(keys2[lo2:hi2]) == DD.S1]
    assert len(s1) == 2 and sorted(s1_counts.tolist()) == [1, 2]


def test_empty_block_stays_zero():
    atom_feat = np.zeros((1, 9), np.int64)
    keys, counts, ptr = DD.molecule_descriptors(atom_feat, np.zeros((0, 3), np.int64),
                                               np.zeros(0, np.int64), np.zeros(0, np.int64))
    totals = DD.block_totals(keys, counts, ptr, 1)
    assert totals[0].tolist()[1] == 0 and totals[0].tolist()[2] == 0 and totals[0].tolist()[4] == 0
    values = DD.normalised_values(keys, counts, ptr, 1)
    block = DD.block_of(keys)
    assert not np.isin(block, [DD.S1, DD.S2, DD.EGO_BOND]).any()
    assert abs((values ** 2).sum() - 2.0 / 5.0) < 1e-12


def test_descriptor_relabel_invariance():
    z = np.load(C.RAW_CACHE)
    rng = np.random.default_rng(3)
    row = 7
    a0, a1 = int(z["node_ptr"][row]), int(z["node_ptr"][row + 1])
    b0, b1 = int(z["bond_ptr"][row]), int(z["bond_ptr"][row + 1])
    n = a1 - a0
    perm = rng.permutation(n)
    pos = np.empty(n, np.int64)
    pos[perm] = np.arange(n)
    bu = np.asarray(z["bond_u"][b0:b1], np.int64)
    bv = np.asarray(z["bond_v"][b0:b1], np.int64)
    nu = np.minimum(pos[bu], pos[bv])
    nv = np.maximum(pos[bu], pos[bv])
    border = rng.permutation(bu.size)
    k1, c1, p1 = DD.molecule_descriptors(z["atom_feat"][a0:a1], z["bond_feat"][b0:b1], bu, bv)
    k2, c2, p2 = DD.molecule_descriptors(z["atom_feat"][a0:a1][perm],
                                         z["bond_feat"][b0:b1][border],
                                         nu[border], nv[border])
    for old in range(n):
        new = int(pos[old])
        a = k1[p1[old]:p1[old + 1]]
        b = k2[p2[new]:p2[new + 1]]
        assert np.array_equal(np.sort(a), np.sort(b))
        assert np.array_equal(np.sort(c1[p1[old]:p1[old + 1]]),
                              np.sort(c2[p2[new]:p2[new + 1]]))


# ---------------------------------------------------------------------------
# references / q / branch
# ---------------------------------------------------------------------------


def _toy_z(n=300, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    z, _ = F.l2_normalize_rows(rng.normal(size=(n, dim)))
    return z


def test_reference_sampling_and_q():
    z = _toy_z()
    pos, info = F.sample_references(z[:250], k=64, seed=0)
    assert pos.size == 64 and len(np.unique(pos)) == 64
    assert info["distinct_projection_vectors"] == 64
    assert np.unique(z[:250][pos], axis=0).shape[0] == 64
    tau_info = F.temperature(z[:250], z[:250][pos])
    assert tau_info["tau"] > 0
    q, nearest = F.reference_weights(z, z[:250][pos], tau_info["tau"])
    assert np.isfinite(q).all()
    assert (q >= 0).all()
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-6)
    assert q.dtype == np.float32
    # the reference point itself concentrates mass on itself
    self_q = q[int(pos[0])]
    assert int(np.argmax(self_q)) == 0
    assert self_q[0] == self_q.max()
    # the temperature is the Fit median of the 8th-nearest squared distance
    d2 = F.squared_distances(z[:250], z[:250][pos])
    d8 = np.partition(d2, 7, axis=1)[:, 7]
    assert abs(tau_info["tau"] - np.median(d8)) < 1e-12


def test_branch_normalisation_and_plan():
    z = _toy_z()
    q, _ = F.reference_weights(z, z[:64], 0.5)
    norm_z = F.branch_normalisation(z)
    tz = F.normalise_branch(z, norm_z)
    assert abs(float(tz.mean())) < 1e-6
    assert abs(float(np.sqrt(np.mean(tz ** 2))) - 1.0) < 1e-5
    norm_q = F.branch_normalisation(q)
    tq = F.normalise_branch(q, norm_q)
    assert abs(float(np.sqrt(np.mean(tq ** 2))) - 1.0) < 1e-5
    plan = F.make_batch_plan(10, seed=0, epochs=3)
    assert plan.shape == (3, 10)
    for epoch in plan:
        assert sorted(epoch.tolist()) == list(range(10))
    plan2 = F.make_batch_plan(10, seed=0, epochs=3)
    assert np.array_equal(plan, plan2)


# ---------------------------------------------------------------------------
# model parity
# ---------------------------------------------------------------------------


def _small_structs(n_mol=3):
    z = np.load(C.RAW_CACHE)
    return {row: SS.canonical_full_struct(z, row) for row in range(n_mol)}


def _batch(rows, structs, table, ptr):
    batch = __import__("ses_tensor").collate([structs[int(r)] for r in rows], "cpu")
    batch["branch"] = M.branch_gather(np.asarray(rows), structs, table, ptr, "cpu")
    return batch


def test_model_equal_params_init_and_zero_branch_equivalence():
    z = np.load(C.RAW_CACHE)
    atom_dims, bond_dims = SD.atom_bond_dims()
    cfg = C.load_config()
    norm = {"atom_scalar_mean": np.zeros(5), "atom_scalar_scale": np.ones(5),
            "graph_mean": np.zeros(8), "graph_scale": np.ones(8)}
    m1, i1 = M.build_css_model(atom_dims, bond_dims, 0, norm, cfg)
    m2, i2 = M.build_css_model(atom_dims, bond_dims, 0, norm, cfg)
    assert i1["parameters"] == i2["parameters"]
    assert i1["init_hashes"] == i2["init_hashes"]
    base = GM.build_model(atom_dims, bond_dims, 0, norm, use_words=False, config=cfg)[1]
    assert base["parameters"]["total"] == C.FROZEN_PREDICTOR_PARAMS
    assert i1["parameters"]["total"] == C.FROZEN_PREDICTOR_PARAMS + i1["branch_parameters"]
    assert i1["branch_parameters"] == 32 * 192 + 64 * 32 + 32

    # frozen-equivalent trunk with a zero branch reproduces the frozen model
    frozen, _ = GM.build_model(atom_dims, bond_dims, 0, norm, use_words=False, config=cfg)
    sd_css = m1.state_dict()
    sd_frozen = frozen.state_dict()
    new = dict(sd_frozen)
    for key, value in sd_css.items():
        if key.startswith("backbone_mlp.") or key.startswith("branch_proj"):
            continue
        new[key] = value
    first = m1.backbone_mlp[0]
    new["backbone_mlp.0.weight"] = first.weight.detach()[:, :m1.backbone_in_dim].clone()
    new["backbone_mlp.0.bias"] = first.bias.detach().clone()
    for key in ("backbone_mlp.2.weight", "backbone_mlp.2.bias"):
        new[key] = sd_css[key]
    frozen.load_state_dict(new, strict=True)
    with torch.no_grad():
        m1.branch_proj.weight.zero_()
        m1.branch_proj.bias.zero_()
    m1.eval()
    frozen.eval()

    structs = _small_structs(3)
    ptr = TR.row_atom_ptr(structs, 3)
    table = np.zeros((int(ptr[-1]), 64), np.float32)
    rows = [0, 1, 2]
    batch = _batch(rows, structs, table, ptr)
    with torch.no_grad():
        a = m1(batch, arm="SENTENCE_BASE")["yhat_std"]
        b = frozen(batch, arm="SENTENCE_BASE")["yhat_std"]
    assert float((a - b).abs().max()) < 1e-6

    # single-molecule and batch predictions agree
    m1.eval()
    with torch.no_grad():
        single = [float(m1(_batch([r], structs, table, ptr), arm="SENTENCE_BASE")["yhat_std"][0])
                  for r in rows]
    assert np.max(np.abs(np.asarray(single) - a.numpy())) < 1e-6

    # the branch has a finite non-zero gradient and the trunk updates
    rng = np.random.default_rng(0)
    table = rng.normal(size=(int(ptr[-1]), 64)).astype(np.float32)
    m1.train()
    opt = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    batch = _batch(rows, structs, table, ptr)
    out = m1(batch, arm="SENTENCE_BASE")
    loss = torch.mean(out["yhat_std"] ** 2)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(m1.branch_proj.weight.grad).all()
    assert float(m1.branch_proj.weight.grad.norm()) > 0
    assert float(m1.atom_emb[0].weight.grad.norm()) > 0
    before = m1.atom_emb[0].weight.detach().clone()
    opt.step()
    assert float((m1.atom_emb[0].weight.detach() - before).abs().max()) > 0


# ---------------------------------------------------------------------------
# tiny end-to-end training loop
# ---------------------------------------------------------------------------


def _toy_prep(tmp_path: Path, n_mol: int = 14, fit_n: int = 8) -> Path:
    z = np.load(C.RAW_CACHE)
    rows = list(range(n_mol))
    desc = DD.build_all(z, rows)
    n_roots = int(desc["root_row"].size)
    fit_rows = np.asarray(sorted(rows)[:fit_n], np.int64)
    valid_rows = np.asarray([rows[fit_n]], np.int64)
    test_rows = np.asarray([rows[fit_n + 1]], np.int64)
    hold_rows = np.asarray(rows[fit_n + 2:], np.int64)
    fit_mask = np.isin(np.asarray(desc["root_row"], np.int64), fit_rows)
    frozen, _ = F.fit_projection(desc, fit_rows, fit_mask, n_roots)
    projected = frozen.project(desc, n_roots)
    zz, q = projected["z"], projected["q"]
    ref_pos = np.asarray(frozen.info["ref_positions"], np.int64)
    z_ref = frozen.z_ref
    tau = frozen.info["temperature"]
    nz = frozen.norm_direct
    nq = frozen.norm_share
    structs = {row: SS.canonical_full_struct(z, row) for row in rows}
    counts = np.asarray([structs[r].n_atoms for r in rows], np.int64)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    bd, bs = F.canonical_branch_tables(zz, q, nz, nq, structs, ptr, len(rows))
    out = tmp_path / "prep"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "structs.pkl").open("wb") as handle:
        pickle.dump({"format": "css-v0-full-structs-v1", "structs": structs}, handle)
    np.savez(out / "fit_100.npz", branch_DIRECT=bd, branch_SHARE=bs)
    plan = F.make_batch_plan(fit_rows.size, seed=0, epochs=2)
    np.save(out / "plan_100.npy", plan)
    labels = np.asarray(z["labels"], np.float64)
    scaler = {"mean": float(labels[fit_rows].mean()),
              "std": float(labels[fit_rows].std()), "n": int(fit_rows.size)}
    fitstats = {
        "format": "css-v0-fitstats-v1", "size": 100, "seed": 0,
        "target_scaler": scaler,
        "graph_mean": np.zeros(8), "graph_scale": np.ones(8),
        "atom_scalar_mean": np.zeros(5), "atom_scalar_scale": np.ones(5),
        "atom_ok": [np.ones(d, bool) for d in SD.atom_bond_dims()[0]],
        "bond_ok": [np.ones(d, bool) for d in SD.atom_bond_dims()[1]],
        "fit_rows": fit_rows, "heldout_rows": np.zeros(0, np.int64),
        "valid_rows": valid_rows, "test_rows": test_rows,
    }
    with (out / "fitstats_100.pkl").open("wb") as handle:
        pickle.dump(fitstats, handle)
    manifest = {
        "format": "css-v0-prep-v1",
        "topology": {"sources": {str(C.RAW_CACHE.relative_to(C.REPO_ROOT)): C.RAW_CACHE_SHA256}},
        "n_roots": n_roots,
        "descriptor_sha256": {"keys": "toy"},
        "code_sha256": "toy",
        "official_rows": {"train": "toy", "valid": "toy", "test": "toy"},
        "sizes": {"100": {"branch_table_sha256": {
            "DIRECT": C.array_sha256(bd), "SHARE": C.array_sha256(bs)}}},
    }
    (out / "prep_manifest.json").write_text(json.dumps(manifest))
    assert hold_rows.size >= 0
    return out


def test_train_loop_two_arms(tmp_path):
    prep = _toy_prep(tmp_path)
    manifests = {}
    for arm in ("DIRECT", "SHARE"):
        args = TR_argparse(arm=arm, size=100, epochs=2, prep_dir=str(prep),
                           out=str(tmp_path / arm), run_id=f"toy_{arm}")
        manifests[arm] = TR.train(args)
    a, b = manifests["DIRECT"], manifests["SHARE"]
    assert a["params"] == b["params"]
    assert a["init_hashes"] == b["init_hashes"]
    assert a["prep"]["batch_plan_sha256"] == b["prep"]["batch_plan_sha256"]
    assert a["prep"]["branch_table_sha256"] != b["prep"]["branch_table_sha256"]
    assert a["epochs_run"] == b["epochs_run"] == 2
    for m in manifests.values():
        assert m["gradient_probe_step1"]["branch_proj"] > 0
        assert m["gradient_probe_step1"]["atom_emb"] > 0
        assert np.isfinite(m["metrics"]["valid_rmse"])
        assert (Path(m["checkpoint"]).is_file())
        assert Path(m["predictions"]["valid"]).is_file()


def TR_argparse(**kwargs):
    import argparse

    return argparse.Namespace(seed=0, device="cpu", smoke=True, **kwargs)
