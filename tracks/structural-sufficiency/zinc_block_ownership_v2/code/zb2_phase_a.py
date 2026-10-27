"""Phase A — CPU diagnostics for ``zinc_block_ownership_v2`` (no GPU).

Five blocks, exactly as pre-registered in ``PLAN.md`` §1:

A1  cross-coverage of support tier x ordinary/tail x size-combo x sig-combo novelty,
    plus where row 172 sits inside those subsets;
A2  per-group contribution of the global N-R delta, every drop variant (not just
    the favourable ones), and leave-one-out bounds;
A3  ``n_multi`` stratification -- ``n_multi == 1`` is the R/S input-equivalence
    negative control (with one multi block, S's multi mean equals that block);
A4  S-arm data checks at v1's level: conservation, S == R when ``n_multi <= 1``,
    invariance, init identity, gradient flow, and prediction consistency under an
    arm switch with frozen weights;
A5  fixed-weight content interventions R->S, R->N (on the R checkpoint) and
    N->R, N->S (on the N checkpoint).  These show what the trained model *reads*;
    they are not evidence about generalization.

Everything reads official train/valid only.  The official test is never touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zb2_common as Z  # noqa: E402

BC = Z.BC
ZB = Z.ZB
ZM = Z.ZM
ZK = __import__("zbo_checks")
V1RUNS = Z.V1ROUND / "runs"


def _delta_of(run_ref: dict, run_other: dict) -> np.ndarray:
    """Per-row AE(other) - AE(ref): positive => the reference arm is better."""
    return run_other["ae"]["best"] - run_ref["ae"]["best"]


# ---------------------------------------------------------------------------
# A1  cross-coverage
# ---------------------------------------------------------------------------


def a1_coverage(ctx: dict) -> dict:
    y = ctx["y"]
    tier = np.asarray(ctx["tiers"], dtype=object)
    size_novel = np.asarray(ctx["size_novel"])
    sig_novel = np.asarray(ctx["sig_novel"])
    n_multi = ctx["n_multi"]
    groups = ctx["groups"]

    cells = []
    for t in ("all_supported", "low_support", "any_unseen"):
        for side, side_mask in (("ordinary", y > -3.0), ("tail", y <= -3.0)):
            for sn in (False, True):
                for gn in (False, True):
                    m = (tier == t) & side_mask & (size_novel == sn) & (sig_novel == gn)
                    if not m.any():
                        continue
                    cells.append({
                        "tier": t, "side": side,
                        "size_novel": bool(sn), "sig_novel": bool(gn),
                        "n": int(m.sum()),
                        "share_of_valid": float(m.mean()),
                        "n_multi_mean": float(n_multi[m].mean()),
                        "n_multi_median": float(np.median(n_multi[m])),
                        "y_mean": float(y[m].mean()),
                    })
    # tier x side marginals so the cells can be read against their parents
    marginals = {}
    for t in ("all_supported", "low_support", "any_unseen"):
        for side, side_mask in (("ordinary", y > -3.0), ("tail", y <= -3.0)):
            marginals[f"{t}|{side}"] = int(((tier == t) & side_mask).sum())
    nov = {
        "size_novel": int(size_novel.sum()),
        "sig_novel": int(sig_novel.sum()),
        "size_or_sig_novel": int((size_novel | sig_novel).sum()),
        "size_and_sig_novel": int((size_novel & sig_novel).sum()),
        "sig_novel_and_any_unseen": int((sig_novel & (tier == "any_unseen")).sum()),
        "sig_novel_and_low_support": int((sig_novel & (tier == "low_support")).sum()),
        "size_novel_and_tail": int((size_novel & (y <= -3.0)).sum()),
    }
    # where is row 172?
    r = 172
    row = {
        "row": r,
        "y": float(y[r]),
        "tier": str(tier[r]),
        "size_novel": bool(size_novel[r]),
        "sig_novel": bool(sig_novel[r]),
        "n_multi": int(n_multi[r]),
        "n_blocks": int(ctx["vaf"]["n_blocks"][r]),
        "max_block_edges": int(ctx["vaf"]["max_block_edges"][r]),
        "in_fixed_worst10": bool(ctx["fixed_mask"][r]),
        "in_any_novel_subset": bool(size_novel[r] or sig_novel[r]),
        "groups": sorted([k for k, m in groups.items() if bool(m[r])]),
    }
    return {
        "cells": cells,
        "marginals": marginals,
        "novelty_counts": nov,
        "row172": row,
        "tier_counts": {t: int((tier == t).sum())
                        for t in ("all_supported", "low_support", "any_unseen")},
        "note": ("cells are a complete partition of the 1000 valid rows across "
                 "tier x side x size_novel x sig_novel; empty cells are omitted"),
    }


# ---------------------------------------------------------------------------
# A2  contributions and every drop variant
# ---------------------------------------------------------------------------


def a2_contributions(ctx: dict, delta: np.ndarray, ae_ref: np.ndarray) -> dict:
    y = ctx["y"]
    n = len(delta)
    total = float(delta.mean())

    groups = {}
    for k, m in ctx["groups"].items():
        m = np.asarray(m, dtype=bool)
        groups[k] = {
            "n": int(m.sum()),
            "share": float(m.mean()),
            "delta_mae": float(delta[m].mean()),
            "contribution": float(delta[m].mean() * m.mean()),
            "ci95": Z.bootstrap_ci(delta[m]),
        }
    # partitions that must sum back to the total
    partition_checks = {}
    for name, keys in {
        "ordinary_tail": ["ordinary_y_gt_-3", "tail_y_le_-3"],
        "support_tier": ["support_all_supported", "support_low_support",
                         "support_any_unseen"],
        "n_multi": ["n_multi_0", "n_multi_1", "n_multi_2", "n_multi_3plus"],
        "blocks": [k for k in groups if k.startswith("n_blocks_")],
    }.items():
        keys = [k for k in keys if k in groups]
        s = float(sum(groups[k]["contribution"] for k in keys))
        partition_checks[name] = {
            "keys": keys, "sum_contribution": s, "total": total,
            "abs_gap": float(abs(s - total)),
        }

    fixed = np.asarray(ctx["fixed_mask"], dtype=bool)
    row172 = np.zeros(n, dtype=bool)
    row172[172] = True
    absd = np.abs(delta)
    order_abs = np.argsort(-absd, kind="stable")
    ae_order = np.argsort(-ae_ref, kind="stable")

    def _mask_drop(idx):
        m = np.ones(n, dtype=bool)
        m[np.asarray(idx, dtype=np.int64)] = False
        return m

    variants = {}
    for name, mask in {
        "full": np.ones(n, dtype=bool),
        "drop_row172": ~row172,
        "drop_fixed_worst10": ~fixed,
        "drop_row172_and_fixed_worst10": ~(fixed | row172),
        "drop_top1_abs_delta": _mask_drop(order_abs[:1]),
        "drop_top5_abs_delta": _mask_drop(order_abs[:5]),
        "drop_top10_abs_delta": _mask_drop(order_abs[:10]),
        "drop_top10_ae_ref": _mask_drop(ae_order[:10]),
        "drop_top1_abs_delta_and_row172": _mask_drop(
            sorted(set([int(order_abs[0]), 172]))),
    }.items():
        d = delta[mask]
        variants[name] = {
            "n": int(mask.sum()), "delta_mae": float(d.mean()),
            "ci95": Z.bootstrap_ci(d),
            "fraction_of_full": (float(d.mean() / total) if total != 0 else None),
        }

    # leave-one-out bounds: how far can one row move the headline?
    sums = delta.sum()
    loo = (sums - delta) / (n - 1)
    imin = int(np.argmin(loo))
    imax = int(np.argmax(loo))
    # concentration: share of the *net* delta contributed by the top-k rows
    pos = np.sort(delta)[::-1]
    concentration = {}
    for k in (1, 5, 10, 20, 50):
        concentration[f"top{k}_share_of_net"] = (
            float(pos[:k].sum() / sums) if sums != 0 else None)
    return {
        "delta_total": total,
        "groups": groups,
        "partition_checks": partition_checks,
        "drop_variants": variants,
        "leave_one_out": {
            "min_delta_dropping_one_row": float(loo[imin]),
            "argmin_row": imin,
            "max_delta_dropping_one_row": float(loo[imax]),
            "argmax_row": imax,
            "spread": float(loo[imax] - loo[imin]),
        },
        "concentration": concentration,
        "top1_abs_delta_row": int(order_abs[0]),
        "top1_abs_delta_value": float(delta[order_abs[0]]),
        "n_rows_delta_gt_0.01": int((delta > 0.01).sum()),
        "n_rows_delta_lt_-0.01": int((delta < -0.01).sum()),
        "row172_delta": float(delta[172]),
        "row172_share_of_net": (float(delta[172] / sums) if sums != 0 else None),
        "y_present": bool(len(y) == n),
    }


# ---------------------------------------------------------------------------
# A3  n_multi stratification and the R/S data-equivalence premise
# ---------------------------------------------------------------------------


def a3_n_multi(ctx: dict, delta: np.ndarray, runs: dict) -> dict:
    n_multi = ctx["n_multi"]
    bins = {"n_multi_1": n_multi == 1, "n_multi_2": n_multi == 2,
            "n_multi_ge_3": n_multi >= 3}
    out = {}
    for k, m in bins.items():
        out[k] = {"n": int(m.sum()), "share": float(m.mean()),
                  "delta_mae": float(delta[m].mean()),
                  "contribution": float(delta[m].mean() * m.mean()),
                  "ci95": Z.bootstrap_ci(delta[m])}
    # data-level: how often does S's block content equal R's?
    norm = BC.load_norm()
    model = runs["R"]["_model"]
    cache, rel, blocks = ctx["vcache"], BC.load_rel("valid"), ctx["vblocks"]
    rows = np.arange(len(ctx["y"]), dtype=np.int64)
    eq_rows = []
    max_abs_per_row = np.zeros(len(rows), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            batch = ZB.collate(cache, rel, blocks, chunk)
            _, st = model(batch, return_stats=True)
            c = st["block_c"]
            r1, r2 = ZB.block_content(c, batch, "R")
            s1, s2 = ZB.block_content(c, batch, "S")
            d = torch.maximum((r1 - s1).abs().max(dim=1).values,
                              (r2 - s2).abs().max(dim=1).values)
            # per-molecule max over its blocks
            b = batch["block_batch"]
            agg = torch.zeros(len(chunk)).index_reduce_(
                0, b, d, "amax", include_self=True)
            max_abs_per_row[start:start + len(chunk)] = agg.numpy()
    exact = max_abs_per_row == 0.0
    for k, m in bins.items():
        out[k]["s_equals_r_exact_fraction"] = float(exact[m].mean()) if m.any() else None
    out["n_multi_le_1_s_equals_r_exact_fraction"] = float(exact[n_multi <= 1].mean())
    out["n_multi_le_1_n"] = int((n_multi <= 1).sum())
    out["n_multi_ge_2_s_equals_r_exact_fraction"] = float(exact[n_multi >= 2].mean())
    out["max_abs_block_content_diff_overall"] = float(max_abs_per_row.max())
    out["s_equals_r_exact_fraction_all"] = float(exact.mean())
    out["premise"] = ("n_multi == 1 (and n_multi == 0) molecules have identical "
                     "R and S block content, so they are the input-equivalence "
                     "negative control for the R-vs-S comparison")
    return out


# ---------------------------------------------------------------------------
# A4  S-arm data checks at v1's level
# ---------------------------------------------------------------------------


def a4_s_checks(ctx: dict, runs: dict) -> dict:
    out = {}
    norm = BC.load_norm()
    cache, rel, blocks = ctx["vcache"], BC.load_rel("valid"), ctx["vblocks"]
    rows = np.arange(len(ctx["y"]), dtype=np.int64)

    # -- S1 conservation: per-molecule sums are arm-invariant ---------------
    model = runs["R"]["_model"]
    worst = 0.0
    worst2 = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            batch = ZB.collate(cache, rel, blocks, chunk)
            _, st = model(batch, return_stats=True)
            c = st["block_c"]
            per_mol = c.new_zeros((len(chunk), c.shape[1])).index_add_(
                0, batch["bond_batch"], c)
            per_mol2 = c.new_zeros((len(chunk), c.shape[1])).index_add_(
                0, batch["bond_batch"], c * c)
            for arm in ("R", "N", "S"):
                f, s = ZB.block_content(c, batch, arm)
                sf = f.new_zeros((len(chunk), f.shape[1])).index_add_(
                    0, batch["block_batch"], f)
                ss = s.new_zeros((len(chunk), s.shape[1])).index_add_(
                    0, batch["block_batch"], s)
                worst = max(worst, float((sf - per_mol).abs().max()))
                worst2 = max(worst2, float((ss - per_mol2).abs().max()))
    out["S1_conservation"] = {
        "max_abs_first_moment_error": worst,
        "max_abs_second_moment_error": worst2,
        "pass": bool(worst < 1e-3 and worst2 < 1e-2),   # fp32, magnitudes ~1e2
        "note": "relative error is ~1e-15 in float64 (v1 C3); this is the fp32 path",
    }

    # -- S2 invariance for the S content on multi-block molecules -----------
    n_multi = ctx["n_multi"]
    cand = np.where(n_multi >= 2)[0][:8]
    specs = ZK.specs_from_rows(cache, cand)
    _, _, _, base_batch = ZK.build_mini(specs)
    norm_model, _ = ZM.build_model(norm, "S", 0, "cpu")
    gen = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        norm_model.reader_ext.weight.normal_(0.0, 0.05, generator=gen)
    norm_model.eval()

    def h_of(var_specs, arm="S"):
        _, _, _, b = ZK.build_mini(var_specs)
        with torch.no_grad():
            _, st = norm_model(b, return_stats=True)
        return st["z_new"], st["block_c"], b

    with torch.no_grad():
        base_h, base_c, base_b = h_of(specs)
        base_cf, base_cs = ZB.block_content(base_c, base_b, "S")
    rng = np.random.default_rng(7)
    relab = []
    for s in specs:
        k = len(s["atom_feat"])
        p = rng.permutation(k)
        inv = np.empty(k, dtype=np.int64)
        inv[p] = np.arange(k)
        item = dict(s)
        item["atom_feat"] = s["atom_feat"][p]
        item["bond_index"] = inv[s["bond_index"]]
        relab.append(item)
    rev = []
    for s in specs:
        item = dict(s)
        item["bond_index"] = s["bond_index"][:, ::-1].copy()
        item["bond_feat"] = s["bond_feat"].copy()
        rev.append(item)
    rng2 = np.random.default_rng(11)
    eord = []
    for s in specs:
        perm = rng2.permutation(len(s["bond_index"]))
        item = dict(s)
        item["bond_index"] = s["bond_index"][perm].copy()
        item["bond_feat"] = s["bond_feat"][perm].copy()
        eord.append(item)

    diffs = {}
    for name, var in (("atom_relabel", relab), ("edge_reverse", rev),
                      ("edge_order", eord)):
        h, c, b = h_of(var)
        diffs[name] = float((h - base_h).abs().max())
    perm = list(range(len(specs)))[::-1]
    h_perm, _, _ = h_of([specs[i] for i in perm])
    back = np.argsort(np.asarray(perm))
    diffs["batch_reorder"] = float((h_perm[torch.as_tensor(back)] - base_h).abs().max())
    out["S2_invariance_S_content"] = {
        **{f"{k}_max_h_diff": v for k, v in diffs.items()},
        "n_molecules": int(len(specs)),
        "n_multi_min": int(n_multi[cand].min()),
        "pass": bool(max(diffs.values()) <= 1e-5),
    }

    # -- S3 init identity across arms --------------------------------------
    init = {}
    for seed in (0, 1):
        hashes = {}
        for arm in ("R", "N", "S"):
            m, info = ZM.build_model(norm, arm, seed, "cpu")
            hashes[arm] = {"initial_state_sha256": info["initial_state_sha256"],
                           "parameters": info["parameters"]}
        init[f"seed{seed}"] = {
            "arms": hashes,
            "all_equal": len({v["initial_state_sha256"] for v in hashes.values()}) == 1,
            "parameters_equal": len({v["parameters"] for v in hashes.values()}) == 1,
        }
    out["S3_init_identity"] = init

    # -- S4 gradient flow for the S arm ------------------------------------
    m_s = ZM.build_model(norm, "S", 0, "cpu")[0]
    m_s.train()
    multi_rows = np.where(n_multi >= 2)[0][:64]
    b = ZB.collate(cache, rel, blocks, multi_rows)
    y = torch.as_tensor(np.asarray(ctx["y"])[multi_rows], dtype=torch.float32)
    opt = torch.optim.AdamW(m_s.parameters(), lr=float(BC.RECIPE["lr"]))

    def _grads() -> dict:
        g = {}
        for name in ("block_norm", "block_mlp", "reader_ext"):
            mod = getattr(m_s, name)
            g[name] = float(np.sqrt(sum(float(q.grad.pow(2).sum())
                                        for q in mod.parameters()
                                        if q.grad is not None)))
        return g

    # step 0: reader_ext is zero-initialised, so dL/d(block encoder) is exactly 0
    # by construction; this is the expected zero-init-last-layer fact, not a bug.
    pred = m_s(b)
    loss0 = float(torch.abs(pred - y).mean())
    loss0_t = torch.abs(pred - y).mean()
    loss0_t.backward()
    g0 = _grads()
    opt.step()
    opt.zero_grad(set_to_none=True)
    # step 1: after one update the block encoder must receive gradient
    pred = m_s(b)
    loss1 = float(torch.abs(pred - y).mean())
    torch.abs(pred - y).mean().backward()
    g1 = _grads()
    out["S4_gradient"] = {
        "loss_step0": loss0, "loss_step1": loss1,
        "grad_norms_step0": g0, "grad_norms_step1": g1,
        "expected_step0_block_grad_zero": bool(g0["block_norm"] == 0.0
                                              and g0["block_mlp"] == 0.0),
        "pass": bool(g1["block_norm"] > 1e-8 and g1["block_mlp"] > 1e-8
                     and g1["reader_ext"] > 1e-8),
        "note": ("v1's mechanism gate is 'block grad at step 1 > 1e-8'; step 0 is "
                 "zero by the zero-initialised reader_ext and is expected to be 0"),
    }

    # -- S5 prediction consistency under an arm switch (frozen R weights) --
    r_model = runs["R"]["_model"]
    p_R = BC.predict(r_model, cache, rel, blocks, rows, device="cpu", arm="R")
    p_S = BC.predict(r_model, cache, rel, blocks, rows, device="cpu", arm="S")
    p_N = BC.predict(r_model, cache, rel, blocks, rows, device="cpu", arm="N")
    eq = n_multi <= 1
    out["S5_prediction_consistency"] = {
        "model": "R_s0 (frozen weights)",
        "n_rows_multi_le_1": int(eq.sum()),
        "max_abs_pred_diff_R_vs_S_on_multi_le_1": float(np.max(np.abs(p_R - p_S)[eq])),
        "max_abs_pred_diff_R_vs_N_on_multi_le_1": float(np.max(np.abs(p_R - p_N)[eq])),
        "max_abs_pred_diff_R_vs_S_on_multi_ge_2": float(np.max(np.abs(p_R - p_S)[~eq])),
        "pass": bool(np.max(np.abs(p_R - p_S)[eq]) < 1e-6),
        "note": ("on n_multi<=1 the S content is identical to R, so a frozen model "
                 "must produce identical predictions; N differs there too because "
                 "N also replaces singleton content"),
    }
    return out


# ---------------------------------------------------------------------------
# A5  fixed-weight content interventions
# ---------------------------------------------------------------------------


def a5_probe(ctx: dict, runs: dict) -> dict:
    out = {}
    for tag, run in runs.items():
        if tag == "S":
            continue
        model_arm = run["arm"]
        pairs = ([("R", "S"), ("R", "N")] if model_arm == "R"
                 else [("N", "R"), ("N", "S")])
        out[tag] = {}
        for a, b in pairs:
            pr = Z.probe_contents(run, ctx, a, b)
            keys = ("n_multi_1", "n_multi_2", "n_multi_3plus",
                    "support_low_support", "support_any_unseen",
                    "ordinary_y_gt_-3", "tail_y_le_-3", "fixed_worst10")
            pr["delta_by_group_selected"] = {
                k: pr["delta_by_group"][k] for k in keys if k in pr["delta_by_group"]}
            out[tag][f"{a}->{b}"] = pr
    out["claim_limit"] = (
        "fixed-weight content substitution measures what the trained network "
        "reads. It is not a causal estimate of generalization: the weights were "
        "fit on one content construction, so an off-distribution substitution "
        "conflates 'reads this' with 'was trained for this'.")
    return out


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Z.ANALYSIS / "phase_a.json"))
    p.add_argument("--utc", default=None)
    args = p.parse_args(argv)

    Z.log("[zb2 phase A] verifying v1 frozen inputs")
    integ = Z.require_v1_integrity()
    Z.log(f"[zb2 phase A] v1 config/gates/code/caches unchanged: {integ['ok']}")

    ctx = Z.valid_context()
    Z.log(f"[zb2 phase A] valid context ready: {len(ctx['y'])} rows, "
          f"{len(ctx['groups'])} groups")
    runs = {
        "R": Z.load_run(V1RUNS / "R_s0", ctx),
        "N": Z.load_run(V1RUNS / "N_s0", ctx),
    }
    Z.log(f"[zb2 phase A] runs: R {runs['R']['best_valid']:.6f} / "
          f"N {runs['N']['best_valid']:.6f}")
    delta = _delta_of(runs["R"], runs["N"])

    payload = {
        "round": Z.ROUND_ID, "phase": "A", "written": args.utc or Z.now(),
        "official_test_loaded": False,
        "v1_integrity": integ,
        "inputs": {
            "runs": {k: {"dir": v["dir"], "best_valid": v["best_valid"],
                         "W20": v["W20"], "T100": v["T100"],
                         "final_valid": v["final_valid"],
                         "checkpoint_sha256": v["checkpoint_sha256"],
                         "pred_cpu_recompute_max_abs_diff":
                             v["pred_cpu_recompute_max_abs_diff"]}
                     for k, v in runs.items()},
            "delta_definition": "ae(N) - ae(R) on best checkpoints; positive => R better",
            "delta_total": float(delta.mean()),
            "delta_ci95": Z.bootstrap_ci(delta),
            "n_valid": int(len(delta)),
        },
        "A1_coverage": a1_coverage(ctx),
        "A2_contributions": a2_contributions(ctx, delta, runs["R"]["ae"]["best"]),
        "A3_n_multi": a3_n_multi(ctx, delta, runs),
        "A4_s_checks": a4_s_checks(ctx, runs),
        "A5_probe": a5_probe(ctx, runs),
    }
    # headline consistency: the saved predictions must reproduce the logged MAE
    for k, v in runs.items():
        payload["inputs"]["runs"][k]["best_valid_recomputed_from_saved"] = float(
            np.mean(v["ae"]["best"]))
    Z.write_json(args.out, Z.jsonable(payload))

    a4 = payload["A4_s_checks"]
    Z.log(f"[zb2 phase A] S checks: S1={a4['S1_conservation']['pass']} "
          f"S2={a4['S2_invariance_S_content']['pass']} "
          f"S3={a4['S3_init_identity']['seed0']['all_equal'] and a4['S3_init_identity']['seed1']['all_equal']} "
          f"S4={a4['S4_gradient']['pass']} S5={a4['S5_prediction_consistency']['pass']}")
    Z.log(f"[zb2 phase A] -> {BC.rel_or_abs(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
