#!/usr/bin/env python
"""Engineering checks that can change a conclusion.  No general test framework.

Covers PROTOCOL.md section 11: data identity, strict-catalogue definition and
counts, FGCR parameter/init/batch-plan identity and historical replay, exact
brute-force agreement in float64, symmetries, belief-call accounting, the target
identity, aux-path purity (BatchNorm buffers, training RNG, autograd), pair
endpoint provenance, and checkpoint reload.

    uv run python .../code/checks.py --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import sr_common as C
import sr_context as SX
import sr_data as SD
import sr_pairs as SP
import sr_fgcr as SF
import sr_resp as SR
import sr_train as ST

DEVICE = "cpu"
OUT = C.ANALYSIS_DIR / "checks.json"


def _lse(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def brute_force(u, edges, psi):
    n = int(u.shape[0])
    states = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)
    energy = np.zeros(((1 << n), u.shape[1]), dtype=np.float64)
    for v in range(n):
        energy += u[v][:, states[:, v]].T
    for e, (a, b) in enumerate(edges):
        energy += psi[e][:, states[:, a], states[:, b]].T
    logz = _lse(energy, axis=0)
    h_full = logz - _lse(u, axis=-1).sum(axis=0)
    post = np.stack([
        _lse(energy[states[:, v] == 1], axis=0) - _lse(energy[states[:, v] == 0], axis=0)
        for v in range(n)])
    eta_full = post - (u[:, :, 1] - u[:, :, 0])
    log_p = u - _lse(u, axis=-1)[:, :, None]
    h_ind = np.zeros((1, u.shape[1]))
    eta_star = np.zeros((n, u.shape[1]))
    for e, (a, b) in enumerate(edges):
        el = log_p[a][:, :, None] + log_p[b][:, None, :] + psi[e]
        h_ind[0] += _lse(el.reshape(u.shape[1], 4), axis=1)
        m_vu = _lse(log_p[b][:, None, :] + psi[e], axis=2)
        m_uv = _lse(log_p[a][:, :, None] + psi[e], axis=1)
        eta_star[a] += m_vu[:, 1] - m_vu[:, 0]
        eta_star[b] += m_uv[:, 1] - m_uv[:, 0]
    return {"H_full": h_full, "eta_full": eta_full, "H_ind": h_ind[0], "eta_star": eta_star}


def exact_modes(components, smax=4):
    """Run both response modes through the real plan path in float64."""
    graphs = [(int(n), list(edges)) for n, edges, _, _ in components]
    plan = SF.I.build_corpus_plan(graphs, smax=int(smax))
    atom_counts = [int(n) for n, _, _, _ in components]
    bond_counts = [len(e) for _, e, _, _ in components]
    bp = plan.select(list(range(len(components))), atom_counts, bond_counts)
    u_all, psi_all, bu, bv, bm, am = [], [], [], [], [], []
    off = 0
    for g, (n, edges, u, psi) in enumerate(components):
        u_all.append(u)
        psi_all.append(psi)
        bu.extend([a + off for a, b in edges])
        bv.extend([b + off for a, b in edges])
        bm.extend([g] * len(edges))
        am.extend([g] * int(n))
        off += int(n)
    u_t = torch.as_tensor(np.concatenate(u_all), dtype=torch.float64)
    psi_t = torch.as_tensor(np.concatenate(psi_all), dtype=torch.float64)
    resp = SF.I.exact_responses(u_t, psi_t, bp)
    unary = torch.logsumexp(u_t, dim=-1)
    usum = u_t.new_zeros((len(components), int(u_t.shape[1])))
    usum.index_add_(0, torch.as_tensor(am, dtype=torch.long), unary)
    h_full = (resp["logz"] - usum).numpy()
    eta_full = ((resp["logw"][..., 1] - resp["logw"][..., 0])
                - (u_t[..., 1] - u_t[..., 0])).numpy()
    h_ind, eta_star = SF.local_references(
        u_t, psi_t, torch.as_tensor(bu, dtype=torch.long),
        torch.as_tensor(bv, dtype=torch.long), torch.as_tensor(bm, dtype=torch.long),
        len(components), int(u_t.shape[1]))
    post = (resp["logw"][..., 1] - resp["logw"][..., 0])
    prob_sum = (torch.sigmoid(post)).numpy()
    return {"H_full": h_full, "eta_full": eta_full,
            "H_ind": h_ind.numpy(), "eta_star": eta_star.numpy(),
            "logz": resp["logz"].numpy(), "posterior_prob": prob_sum}


def synthetic_cases():
    rng = np.random.default_rng(20270115)
    channels = 4

    def sym_psi(m):
        raw = rng.normal(size=(m, channels, 3)) * 0.8
        psi = np.empty((m, channels, 2, 2))
        psi[:, :, 0, 0] = raw[:, :, 0]
        psi[:, :, 0, 1] = raw[:, :, 1]
        psi[:, :, 1, 0] = raw[:, :, 1]
        psi[:, :, 1, 1] = raw[:, :, 2]
        return psi

    def make(n, edges, zero=False):
        u = rng.normal(size=(n, channels, 2)) * 1.4
        psi = np.zeros((len(edges), channels, 2, 2)) if zero else sym_psi(len(edges))
        return (n, list(edges), u, psi)

    return {
        "single_edge": [make(2, [(0, 1)])],
        "two_disjoint_single_edges": [make(2, [(0, 1)]), make(2, [(0, 1)])],
        "chain4": [make(4, [(0, 1), (1, 2), (2, 3)])],
        "star4": [make(4, [(0, 1), (0, 2), (0, 3)])],
        "cycle4": [make(4, [(0, 1), (1, 2), (2, 3), (3, 0)])],
        "multicomponent_with_isolated_atom": [
            make(4, [(0, 1), (1, 2), (2, 0)]), make(2, [(0, 1)]), make(1, [])],
        "zero_psi_cycle4": [make(4, [(0, 1), (1, 2), (2, 3), (3, 0)], zero=True)],
    }


def main() -> int:
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    DEVICE = args.device
    checks: dict[str, dict] = {}

    def record(name, passed, **details):
        checks[name] = {"pass": bool(passed), **C.jsonable(details)}

    # ---------------- data identity -----------------------------------
    ctx = SX.build_lipo(include_valid=True)
    b = ctx.bundle
    record("data_lipo_sha256", b.cache_sha256 == C.EXPECTED_SHA["lipo_cache"],
           sha256=b.cache_sha256)
    record("data_test_block_dropped",
           ctx.audit["test_arrays_dropped_before_use"] and ctx.audit["test_block_present_in_file"],
           statement=ctx.audit["test_statement"])
    record("data_split_sizes",
           ctx.split_rows["train"].size == 3360 and ctx.split_rows["valid"].size == 420,
           train=int(ctx.split_rows["train"].size), valid=int(ctx.split_rows["valid"].size))
    record("data_train_valid_disjoint",
           np.intersect1d(ctx.split_rows["train"], ctx.split_rows["valid"]).size == 0)
    record("data_fit_dev_partition",
           ctx.d0["n_fit"] == 2856 and ctx.d0["n_dev"] == 504
           and np.intersect1d(ctx.d0["fit_rows"], ctx.d0["dev_rows"]).size == 0
           and np.array_equal(np.sort(np.concatenate([ctx.d0["fit_rows"], ctx.d0["dev_rows"]])),
                              np.sort(ctx.split_rows["train"])),
           fit=int(ctx.d0["n_fit"]), dev=int(ctx.d0["n_dev"]))
    scaler_fit = C.target_scaler(b.labels[ctx.d0["fit_rows"]])
    record("data_scaler_train_only",
           abs(ctx.scaler["mean"] - float(b.labels[ctx.split_rows["train"]].astype(np.float64).mean())) < 1e-12
           and abs(scaler_fit["mean"] - float(b.labels[ctx.d0["fit_rows"]].astype(np.float64).mean())) < 1e-12,
           official_train_scaler=ctx.scaler, fit_scaler=scaler_fit)

    # ---------------- strict catalogue --------------------------------
    meta = ctx.pairs.meta
    expected = dict(n_pairs_2133=2133, n_components_2133=391, n_fit_fit_pairs=1619,
                    n_fit_fit_molecules=1146, n_fit_fit_components=340,
                    n_fit_dev_pairs=238, n_fit_dev_components=44,
                    n_dev_dev_pairs=276, n_dev_dev_molecules=202,
                    n_dev_dev_components=62, dev_dev_transform_seen_in_fit=123,
                    dev_dev_supported_other_component=119,
                    dev_dev_supported_other_component_components=51,
                    dev_dev_supported_other_component_transforms=65,
                    dev_dev_supported_two_components=97,
                    dev_dev_supported_two_components_components=45,
                    dev_dev_supported_two_components_transforms=46)
    mismatch = {k: {"expected": v, "actual": meta.get(k)} for k, v in expected.items()
                if meta.get(k) != v}
    record("pairs_frozen_counts_match", not mismatch, mismatch=mismatch, meta=meta)
    record("pairs_orientation_all_canonical", meta["orientation_all_plus_one"],
           n_sign_minus_one=meta["n_sign_minus_one"])

    # ---------------- FGCR identity -----------------------------------
    init_state = SX.fgcr_init_state(ctx)
    hashes = {}
    models = {}
    for arm, (h, e) in (("A_IS", ("IND", "STAR")), ("A_IJ", ("IND", "FULL")),
                        ("A_FS", ("FULL", "STAR")), ("A_FJ", ("FULL", "FULL"))):
        model, info = SF.build_variant(b.atom_dims, b.bond_dims, h_mode=h, eta_mode=e,
                                       init_state=init_state, device=DEVICE)
        model.eval()
        models[arm] = model
        hashes[arm] = info["init_hash"]
        record(f"params_{arm}", info["n_parameters"] == 100926, n=info["n_parameters"])
    record("fgcr_init_identical_across_arms", len(set(hashes.values())) == 1, hashes=hashes)

    valid_rows = np.asarray(ctx.split_rows["valid"], dtype=np.int64)
    full_model = models["A_FJ"]
    ckpt = torch.load(C.FGCR_CKPT, map_location="cpu", weights_only=False)
    hist, _ = SF.build_variant(b.atom_dims, b.bond_dims, h_mode="FULL", eta_mode="FULL",
                               init_state=ckpt["state_dict"], device=DEVICE)
    hist.eval()
    pred = SF.predict_std(hist, b, ctx.plan, ctx.row_to_plan, valid_rows, batch_size=256,
                          device=DEVICE)
    raw = pred * ctx.scaler["std"] + ctx.scaler["mean"]
    hist_rmse = C.rmse(raw, b.labels[valid_rows].astype(np.float64))
    record("fgcr_historical_replay_valid_rmse",
           abs(hist_rmse - 0.7128937193209034) <= 1e-4,
           rmse=hist_rmse, reference=0.7128937193209034,
           tolerance=1e-4, note="1e-5 is the cross-device target; 1e-4 allows CPU replay")

    # ---------------- exact brute force -------------------------------
    bf_report = {}
    for name, comps in synthetic_cases().items():
        out = exact_modes(comps)
        bf_h, bf_e, bf_hi, bf_es = [], [], [], []
        for n, edges, u, psi in comps:
            r = brute_force(u, edges, psi)
            bf_h.append(r["H_full"])
            bf_e.append(r["eta_full"])
            bf_hi.append(r["H_ind"])
            bf_es.append(r["eta_star"])
        bf_h = np.stack(bf_h)
        bf_hi = np.stack(bf_hi)
        bf_e = np.concatenate(bf_e)
        bf_es = np.concatenate(bf_es)
        bf_report[name] = {
            "max_abs_H_full": float(np.abs(out["H_full"] - bf_h).max()),
            "max_abs_H_ind": float(np.abs(out["H_ind"] - bf_hi).max()),
            "max_abs_eta_full": float(np.abs(out["eta_full"] - bf_e).max()),
            "max_abs_eta_star": float(np.abs(out["eta_star"] - bf_es).max()),
            "max_abs_H_increment": float(np.abs(out["H_full"] - out["H_ind"]).max()),
            "max_abs_eta_increment": float(np.abs(out["eta_full"] - out["eta_star"]).max()),
            "max_abs_posterior_prob_minus_half": float(np.abs(out["posterior_prob"] - 0.5).max()),
        }
    worst = max(v[k] for v in bf_report.values()
                for k in ("max_abs_H_full", "max_abs_H_ind", "max_abs_eta_full", "max_abs_eta_star"))
    record("exact_bruteforce_float64", worst <= 1e-9, worst=worst, cases= bf_report)
    record("zero_coupling_gives_zero_responses",
           bf_report["zero_psi_cycle4"]["max_abs_H_increment"] <= 1e-12
           and bf_report["zero_psi_cycle4"]["max_abs_eta_increment"] <= 1e-12,
           case=bf_report["zero_psi_cycle4"])
    record("single_edge_and_disjoint_edges_full_equals_local",
           bf_report["single_edge"]["max_abs_H_increment"] <= 1e-9
           and bf_report["single_edge"]["max_abs_eta_increment"] <= 1e-9
           and bf_report["two_disjoint_single_edges"]["max_abs_H_increment"] <= 1e-9
           and bf_report["two_disjoint_single_edges"]["max_abs_eta_increment"] <= 1e-9)
    record("adjacent_edge_graphs_need_not_coincide",
           bf_report["chain4"]["max_abs_H_increment"] > 1e-6
           and bf_report["cycle4"]["max_abs_eta_increment"] > 1e-6,
           chain4_H_increment=bf_report["chain4"]["max_abs_H_increment"],
           cycle4_eta_increment=bf_report["cycle4"]["max_abs_eta_increment"])

    # permutation / edge direction / edge order on the real model path
    inv = _invariance_checks(ctx, hist)
    record("real_model_symmetries", inv["max_abs_all"] <= 1e-4, **inv)

    # ---------------- belief accounting -------------------------------
    rows64 = np.asarray(ctx.split_rows["train"][:64], dtype=np.int64)
    counts = {}
    for arm, model in models.items():
        model.belief_calls = 0
        batch = SF.collate_rows(b, rows64.tolist(), ctx.plan, ctx.row_to_plan, DEVICE)
        with torch.no_grad():
            model(batch)
        counts[arm] = model.belief_calls
    record("belief_calls_once_per_forward",
           counts == {"A_IS": 0, "A_IJ": 1, "A_FS": 1, "A_FJ": 1}, counts=counts)

    # ---------------- gradients --------------------------------------
    model = models["A_FJ"]
    model.train()
    batch = SF.collate_rows(b, rows64.tolist(), ctx.plan, ctx.row_to_plan, DEVICE)
    target = torch.as_tensor(b.labels_std[rows64], dtype=torch.float32, device=DEVICE)
    loss = torch.mean((model(batch)["yhat_std"] - target) ** 2)
    loss.backward()
    grads = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
    record("gradients_finite_and_nonzero",
           all(math.isfinite(v) for v in grads.values())
           and min(grads.get(n, 0.0) for n in ("atom_mlp.0.weight", "bond_mlp.0.weight",
                                               "readout_mlp.0.weight", "beta")) > 0.0,
           min_grad=min(grads.values()), max_grad=max(grads.values()),
           n_with_grad=len(grads))
    model.zero_grad(set_to_none=True)

    # ---------------- target identity ---------------------------------
    ident = _target_identity(ctx, models["A_FS"], rows64)
    record("target_identity_and_swap_invariance", ident["pass"], **ident["details"])

    # ---------------- aux path purity (BatchNorm / RNG / autograd) ----
    purity = _aux_purity(ctx)
    record("aux_eval_path_purity", purity["pass"], **purity["details"])

    # ---------------- pair endpoint provenance ------------------------
    fit_set = set(int(v) for v in ctx.d0["fit_rows"].tolist())
    dev_set = set(int(v) for v in ctx.d0["dev_rows"].tolist())
    sched = SP.pair_schedule(ctx.pairs, 50)
    used = set()
    for step in sched:
        for p in step:
            r = ctx.pairs.rows[int(p)]
            used.add(r["row_a"])
            used.add(r["row_b"])
    record("pair_schedule_fit_only",
           used <= fit_set and not (used & dev_set),
           n_endpoint_rows=len(used), n_dev_rows_touched=len(used & dev_set),
           schedule_sha256=C.sha256_array(sched))
    record("pair_schedule_prefix_shared",
           np.array_equal(SP.pair_schedule(ctx.pairs, 20), sched[:20]))

    # ---------------- checkpoint reload ------------------------------
    reload = _reload_check(models["A_IS"], init_state, ctx, valid_rows)
    record("checkpoint_reload_reproduces", reload["pass"], **reload["details"])

    payload = {"round_id": C.ROUND_ID, "generated_at": C.utc_stamp(),
               "device": DEVICE, "environment": C.environment_snapshot(),
               "checks": checks,
               "all_required_pass": all(v["pass"] for v in checks.values())}
    C.write_json(args.out, payload)
    failed = [k for k, v in checks.items() if not v["pass"]]
    print(json.dumps({"n_checks": len(checks), "failed": failed}, indent=1))
    return 0 if not failed else 1


def _invariance_checks(ctx, model):
    b = ctx.bundle
    row = int(ctx.split_rows["valid"][0])
    a0, a1 = int(b.node_ptr[row]), int(b.node_ptr[row + 1])
    b0, b1 = int(b.bond_ptr[row]), int(b.bond_ptr[row + 1])
    n = a1 - a0
    m = b1 - b0
    rng = np.random.default_rng(9000 + row)
    perm = rng.permutation(n).astype(np.int64)
    order = rng.permutation(m).astype(np.int64)
    bu = perm[b.bond_u[b0:b1][order]]
    bv = perm[b.bond_v[b0:b1][order]]
    bf = b.bond_feat[b0:b1][order]
    flip = rng.random(m) < 0.5
    nu = np.where(flip, bv, bu)
    nv = np.where(flip, bu, bv)
    atom_new = np.empty_like(b.atom_feat[a0:a1])
    atom_new[perm] = b.atom_feat[a0:a1]
    graphs = [(n, list(zip(nu.tolist(), nv.tolist())))]
    plan = SF.I.build_corpus_plan(graphs, smax=4)
    batch_new = {
        "atom_feat": torch.as_tensor(atom_new, dtype=torch.long, device=DEVICE),
        "bond_feat": torch.as_tensor(bf, dtype=torch.long, device=DEVICE),
        "atom_mol": torch.zeros(n, dtype=torch.long, device=DEVICE),
        "bond_mol": torch.zeros(m, dtype=torch.long, device=DEVICE),
        "bond_u": torch.as_tensor(nu, dtype=torch.long, device=DEVICE),
        "bond_v": torch.as_tensor(nv, dtype=torch.long, device=DEVICE),
        "n_atoms": n, "n_bonds": m, "n_molecules": 1,
        "graph_n_atoms": torch.as_tensor([n], dtype=torch.float32, device=DEVICE),
        "graph_n_bonds": torch.as_tensor([m], dtype=torch.float32, device=DEVICE),
        "plan": plan.select([0], [n], [m]).to(DEVICE),
    }
    batch_orig = SF.collate_rows(b, [row], ctx.plan, ctx.row_to_plan, DEVICE)
    with torch.no_grad():
        o1 = model(batch_orig, return_aux=True)
        o2 = model(batch_new, return_aux=True)
    eta_back = o2["eta"].double().cpu().numpy()[perm]
    _, star2 = SF.local_references(o2["u"], o2["log_psi"], batch_new["bond_u"],
                                   batch_new["bond_v"], batch_new["bond_mol"], 1, o2["eta"].shape[1])
    _, star1 = SF.local_references(o1["u"], o1["log_psi"], batch_orig["bond_u"],
                                   batch_orig["bond_v"], batch_orig["bond_mol"], 1, o1["eta"].shape[1])
    loo_back = star2.double().cpu().numpy()[perm]
    return {
        "max_abs_yhat_std": float(np.abs(o2["yhat_std"].double().cpu().numpy()
                                          - o1["yhat_std"].double().cpu().numpy()).max()),
        "max_abs_H": float(np.abs(o2["H"].double().cpu().numpy()
                                  - o1["H"].double().cpu().numpy()).max()),
        "max_abs_eta": float(np.abs(eta_back - o1["eta"].double().cpu().numpy()).max()),
        "max_abs_eta_star": float(np.abs(loo_back - star1.double().cpu().numpy()).max()),
        "max_abs_all": float(max(
            np.abs(o2["yhat_std"].double().cpu().numpy() - o1["yhat_std"].double().cpu().numpy()).max(),
            np.abs(o2["H"].double().cpu().numpy() - o1["H"].double().cpu().numpy()).max(),
            np.abs(eta_back - o1["eta"].double().cpu().numpy()).max(),
            np.abs(loo_back - star1.double().cpu().numpy()).max())),
    }


def _target_identity(ctx, model, rows):
    """L_DELTA - L_ENDPOINT must equal -(lambda/(1+lambda)) * E[e_A e_B]."""
    model.eval()
    b = ctx.bundle
    pair_a = np.asarray([r["row_a"] for r in ctx.pairs.rows], dtype=np.int64)
    pair_b = np.asarray([r["row_b"] for r in ctx.pairs.rows], dtype=np.int64)
    rng = np.random.default_rng(7)
    pairs = ctx.pairs.fit_fit_index[rng.integers(0, ctx.pairs.fit_fit_index.size, size=64)]
    ends = np.stack([pair_a[pairs], pair_b[pairs]], axis=1).reshape(-1)
    uniq, inv = np.unique(ends, return_inverse=True)
    with torch.no_grad():
        pred = SF.predict_std(model, b, ctx.plan, ctx.row_to_plan, uniq, device=DEVICE)
    y_std = b.labels_std[ends].astype(np.float64)
    e = (pred[inv] - y_std).reshape(pairs.size, 2)
    le = 0.5 * float(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2))
    ld = 0.5 * float(np.mean((e[:, 1] - e[:, 0]) ** 2))
    lam = C.LAMBDA
    lhs = lam * (ld - le) / (1.0 + lam)
    rhs = -lam / (1.0 + lam) * float(np.mean(e[:, 0] * e[:, 1]))
    # swapping endpoints must leave both auxiliary terms unchanged
    e_swap = e[:, ::-1]
    le_swap = 0.5 * float(np.mean(e_swap[:, 0] ** 2 + e_swap[:, 1] ** 2))
    ld_swap = 0.5 * float(np.mean((e_swap[:, 1] - e_swap[:, 0]) ** 2))
    return {
        "pass": abs(lhs - rhs) <= 1e-10 and abs(le - le_swap) <= 1e-12 and abs(ld - ld_swap) <= 1e-12,
        "details": {"LE": le, "LD": ld, "lhs": lhs, "rhs": rhs,
                    "abs_diff": abs(lhs - rhs),
                    "swap_LE_diff": abs(le - le_swap),
                    "swap_LD_diff": abs(ld - ld_swap)},
    }


def _aux_purity(ctx):
    """GINE (BatchNorm) aux path must not touch buffers, RNG or autograd."""
    b = ctx.bundle
    import sr_gine as SG

    state = SG.make_init_state(b.gine_categories, seed=0)
    model = SG.build_model(b.gine_categories, state).to(DEVICE)
    model.train()
    rows = np.asarray(ctx.d0["fit_rows"][:64], dtype=np.int64)
    target = torch.as_tensor(b.labels_std[rows], dtype=torch.float32, device=DEVICE)
    batch = {k: v.to(DEVICE) for k, v in SG.pack(b, rows.tolist()).items()}
    loss0 = torch.mean((model(batch) - target) ** 2)
    buffers_before = {n: v.detach().clone() for n, v in model.named_buffers()}
    torch_rng_before = torch.random.get_rng_state().clone()
    np_rng_before = np.random.get_state()[1].copy()
    model.train(False)
    aux = model(batch)
    model.train(True)
    buffers_after = {n: v for n, v in model.named_buffers()}
    rng_after = torch.random.get_rng_state()
    np_rng_after = np.random.get_state()[1]
    buffers_same = all(torch.equal(buffers_before[n], buffers_after[n]) for n in buffers_before)
    torch_rng_same = torch.equal(torch_rng_before, rng_after)
    np_rng_same = np.array_equal(np_rng_before, np_rng_after)

    # base gradient must be unaffected by whether the aux path was run
    def base_grad(run_aux: bool):
        model.zero_grad(set_to_none=True)
        model.train()
        torch.manual_seed(1234)
        loss = torch.mean((model(batch) - target) ** 2)
        if run_aux:
            model.train(False)
            _ = model(batch)
            model.train(True)
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    g_no = base_grad(False)
    g_yes = base_grad(True)
    grad_same = all(torch.equal(g_no[n], g_yes[n]) for n in g_no)
    aux_grad = torch.autograd.grad(aux.sum(), list(model.parameters())[0], retain_graph=False,
                                   allow_unused=True)[0]
    details = {
        "bn_buffers_unchanged": bool(buffers_same),
        "torch_rng_unchanged": bool(torch_rng_same),
        "numpy_rng_unchanged": bool(np_rng_same),
        "base_grad_independent_of_aux_call": bool(grad_same),
        "aux_path_has_autograd": bool(aux.requires_grad and aux_grad is not None
                                      and float(aux_grad.abs().sum()) > 0),
        "base_loss_finite": bool(torch.isfinite(loss0)),
        "n_buffers": len(buffers_before),
    }
    return {"pass": all([buffers_same, torch_rng_same, np_rng_same, grad_same,
                         details["aux_path_has_autograd"]]), "details": details}


def _reload_check(model, init_state, ctx, rows):
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    m1, _ = SF.build_variant(ctx.bundle.atom_dims, ctx.bundle.bond_dims,
                             h_mode="IND", eta_mode="STAR", init_state=state,
                             device=DEVICE)
    p1 = SF.predict_std(m1, ctx.bundle, ctx.plan, ctx.row_to_plan, rows, device=DEVICE)
    p0 = SF.predict_std(model, ctx.bundle, ctx.plan, ctx.row_to_plan, rows, device=DEVICE)
    d = float(np.abs(p1 - p0).max())
    return {"pass": d <= 1e-6, "details": {"max_abs_pred_diff": d}}


if __name__ == "__main__":
    raise SystemExit(main())
