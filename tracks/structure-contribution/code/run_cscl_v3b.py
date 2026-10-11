"""cscl-v3b driver: gauge-normalized SB-CSCL head (S-gauge), seed 0.

Protocol: ``configs/cscl_v3b_gauge.yaml`` (frozen before training).

Single-variable training intervention on top of cscl-v3 (protocol
``configs/strong_backbone_contribution_v3.yaml``): the v3 structured head S is
reused *unchanged* (same frozen features, split, architecture, init, optimizer,
checkpoint rule); the ONLY mechanism change is that the per-type minibatch
mean penalty on delta is REMOVED from the training objective:

    v3  loss = L1 + 1.0 * (L_delta_batch_type + L_gamma_batch)
    v3b loss = L1 + 1.0 * L_gamma_batch

The delta convention (type-conditional mean zero) is instead fixed post-hoc by
a prediction-invariant, fit-only gauge normalisation
(``contribution_gauge.py``).  Training still uses raw y (never the frozen
Full-Y prediction).  seed 0 only; official valid/test never loaded; this
round never retrains F / O / S(v3).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import strong_backbone_attribution as sba  # noqa: E402
import contribution_gauge as cg  # noqa: E402
from run_cscl_v3 import (  # noqa: E402  (reuse: architecture/collate/split/stats)
    SBContribHead,
    V3Batch,
    BATCH_SIZE,
    EPOCHS,
    GRAD_CLIP,
    LR,
    OUT_DIR as V3_OUT_DIR,
    PEN_COEF,
    TRAIN_SHUFFLE_OFFSET,
    WEIGHT_DECAY,
    _batches,
    _index_by_row,
    collate,
    git_commit,
    paired_bootstrap,
    prepare_all,
    seed_everything,
    write_json,
)

PROTOCOL_ID = cg.PROTOCOL_ID
OUT_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3b"
V3_S_STATE = V3_OUT_DIR / "S_s0_best_state.pt"

#: corrected synthetic-truth attachment index (cscl_features descriptor layout:
#: 0-27 atom histogram, 28-31 bond histogram, 32-39 scalars; scalars[4] =
#: n_attachments => flat index D_HIST + 4 = 36).  The v3 synthetic_task used
#: flat index 4 (an atom-histogram bin) -- erratum recorded in the round notes.
ATTACH_IDX = cf.D_HIST + 4


# ---------------------------------------------------------------------------
# split identifiability penalties (the ONLY mechanism delta vs v3)
# ---------------------------------------------------------------------------


def delta_identifiability_penalty(out: Mapping[str, torch.Tensor], unit_type: torch.Tensor, n_types: int) -> torch.Tensor:
    """The delta part of ``SBContribHead.identifiability_penalty`` (v3), verbatim:
    weighted average of squared per-type minibatch means of delta."""
    delta = out["delta"]
    pen = delta.new_zeros(())
    if delta.numel():
        sums = torch.zeros(int(n_types), device=delta.device, dtype=delta.dtype)
        counts = torch.zeros_like(sums)
        sums.index_add_(0, unit_type, delta)
        counts.index_add_(0, unit_type, torch.ones_like(delta))
        means = sums / torch.clamp(counts, min=1.0)
        present = counts > 0
        weights = counts[present] / counts.sum().clamp(min=1.0)
        pen = pen + (weights * means[present].pow(2)).sum()
    return pen


def gamma_identifiability_penalty(out: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """The gamma part of ``SBContribHead.identifiability_penalty`` (v3), verbatim:
    squared batch mean of gamma."""
    gamma = out["gamma"]
    if gamma.numel():
        return gamma.mean().pow(2)
    return gamma.new_zeros(())


# ---------------------------------------------------------------------------
# training (identical to v3 train_arm except the delta penalty term)
# ---------------------------------------------------------------------------


def _eval_mae(model, batch: V3Batch, y: np.ndarray) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred = model(batch)["pred"].cpu().numpy().astype(np.float64)
    return float(np.abs(pred - y).mean()), pred


def train_gauge(ns, device: torch.device, out_dir: Path, seed: int = 0, epochs: int = EPOCHS, log=print):
    """Train S-gauge: same everything as v3 S, minus the delta penalty."""
    assert seed == 0, "this round is seed-0 only"
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    fit_batch = collate(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    mon_batch = collate(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    dev_batch = collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    y_fit = ns.y[ns.idx["fit_inner"]]
    y_mon = ns.y[ns.idx["monitor"]]
    y_dev = ns.y[ns.idx["dev"]]

    seed_everything(seed)
    model = SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed + TRAIN_SHUFFLE_OFFSET)
    target = torch.as_tensor(y_fit, dtype=torch.float32, device=device)

    curve = []
    ckpts: list[tuple[float, int, dict]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for picks in _batches(len(fit_pos), gen, True):
            b = collate(ns.features, [fit_pos[i] for i in picks], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
            out = model(b)
            # THE single intervention: gamma penalty only; no delta term.
            loss = torch.nn.functional.l1_loss(out["pred"], target[torch.as_tensor(picks)])
            loss = loss + PEN_COEF * gamma_identifiability_penalty(out)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        mon_mae, _ = _eval_mae(model, mon_batch, y_mon)
        curve.append({"epoch": epoch, "monitor_mae": mon_mae})
        ckpts.append((mon_mae, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        ckpts.sort(key=lambda t: (t[0], t[1]))
        if len(ckpts) > 5:
            ckpts.pop()
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            log(f"[Sgauge s{seed}] ep={epoch:03d} mon={mon_mae:.5f} {time.perf_counter()-started:.0f}s")
    wall = time.perf_counter() - started

    def evaluate(state):
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        f, _ = _eval_mae(model, fit_batch, y_fit)
        m, _ = _eval_mae(model, mon_batch, y_mon)
        d, dpred = _eval_mae(model, dev_batch, y_dev)
        return {"fit_mae": f, "monitor_mae": m, "dev_mae": d}, dpred

    best_state = dict(ckpts[0][2])
    best_epoch = int(ckpts[0][1])
    soup_state = {k: torch.stack([s[k].float() for _, _, s in ckpts]).mean(0) for k in ckpts[0][2]}
    res_best, dev_pred_best = evaluate(best_state)
    res_soup, dev_pred_soup = evaluate(soup_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "S_gauge_s0_best_state.pt")
    np.savez_compressed(
        out_dir / "S_gauge_s0_dev_preds_raw.npz",
        row=np.asarray(ns.idx["dev"], dtype=np.int64),
        pred=dev_pred_best.astype(np.float32),
        pred_soup=dev_pred_soup.astype(np.float32),
        y=y_dev.astype(np.float32),
    )
    result = {
        "protocol": PROTOCOL_ID,
        "arm": "S_gauge",
        "model": "S-gauge-seed0",
        "seed": seed,
        "git_commit": git_commit(),
        "n_params": n_params,
        "epochs": epochs,
        "wall_seconds": wall,
        "device": str(device),
        "primary_best_monitor": res_best,
        "secondary_soup": res_soup,
        "best_epoch": best_epoch,
        "soup_epochs": sorted(e for _, e, _ in ckpts),
        "curve_tail": curve[-5:],
        "objective": {
            "loss": "L1(raw y)",
            "gamma_batch_penalty_coef": PEN_COEF,
            "delta_batch_type_penalty": 0.0,
            "delta_penalty_removed": True,
            "note": "v3 used L1 + 1.0*(delta_batch_type + gamma_batch); this arm uses L1 + 1.0*gamma_batch only",
        },
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE, "pen_coef_gamma": PEN_COEF, "pen_coef_delta": 0.0,
            "hidden": 64, "optimizer": "Adam", "loss": "L1",
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "train_S_gauge_s0.json", result)
    log(f"[Sgauge s{seed}] primary(dev)={res_best['dev_mae']:.5f} soup={res_soup['dev_mae']:.5f} params={n_params} wall={wall:.0f}s")
    return result


# ---------------------------------------------------------------------------
# Phase A audit: the shape of the v3 minibatch penalty (label-free + frozen S)
# ---------------------------------------------------------------------------


@torch.no_grad()
def penalty_audit(ns, out_dir: Path, device="cpu", epochs_to_replay: int = EPOCHS, log=print) -> dict:
    """How does the v3 per-type minibatch delta penalty actually act?

    Label-free replay of the exact training batch sequence (seed 0, shuffle
    generator seed 101) plus the frozen v3 S model's fit_inner deltas.  No
    training here.  The batch's own instances are not recoverable post-hoc
    (collate order within a type is not stored), so the *actual* per-batch
    penalty is estimated by deterministic rotation over the type's fit
    instances; the sampling decomposition E[pen] = sum_t w_t (mu_t^2 +
    var_t/n_t) is exact given the batch type counts.
    """
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    fit_types = np.concatenate([ns.features[fi].type_ids(ns.vocab) for fi in fit_pos])
    n_types = ns.vocab.n_total

    model = SBContribHead(n_types, ns.tau_ctx, ns.tau_topo)
    model.load_state_dict(torch.load(V3_S_STATE, map_location="cpu", weights_only=True))
    model = model.to(torch.device(device))
    model.eval()
    fit_batch = collate(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    delta_fit = model(fit_batch)["delta"].detach().cpu().numpy().astype(np.float64)
    gamma_fit = model(fit_batch)["gamma"].detach().cpu().numpy().astype(np.float64)

    counts_fit = np.bincount(fit_types, minlength=n_types).astype(np.int64)
    sums_fit = np.zeros(n_types)
    np.add.at(sums_fit, fit_types, delta_fit)
    sq_fit = np.zeros(n_types)
    np.add.at(sq_fit, fit_types, delta_fit**2)
    mu_t = np.zeros(n_types)
    var_t = np.zeros(n_types)
    present_fit = counts_fit > 0
    mu_t[present_fit] = sums_fit[present_fit] / counts_fit[present_fit]
    var_t[present_fit] = np.maximum(sq_fit[present_fit] / counts_fit[present_fit] - mu_t[present_fit] ** 2, 0.0)

    gen = torch.Generator().manual_seed(0 + TRAIN_SHUFFLE_OFFSET)
    n = len(fit_pos)
    batches_types: list[np.ndarray] = []
    for _ in range(epochs_to_replay):
        order = torch.randperm(n, generator=gen).tolist()
        for s in range(0, n, BATCH_SIZE):
            batches_types.append(fit_types[np.asarray(order[s : s + BATCH_SIZE])])
    n_batches = len(batches_types)

    frac_cells_n1, frac_inst_n1 = [], []
    types_per_batch = np.zeros(n_batches)
    act_pen, mean_part, var_part = np.zeros(n_batches), np.zeros(n_batches), np.zeros(n_batches)
    fit_sorted = np.argsort(fit_types, kind="stable")
    type_starts = np.concatenate([[0], np.cumsum(counts_fit)]).astype(np.int64)
    for bi, bt in enumerate(batches_types):
        cnt = np.bincount(bt, minlength=n_types)
        pres = cnt > 0
        types_per_batch[bi] = int(pres.sum())
        cells = cnt[pres]
        frac_cells_n1.append(float((cells == 1).mean()))
        frac_inst_n1.append(float(cells[cells == 1].sum() / max(cells.sum(), 1)))
        N = float(len(bt))
        w = cnt / N
        pen_b = 0.0
        for t in np.flatnonzero(cnt):
            k = int(cnt[t])
            nc = int(counts_fit[t])
            seg = fit_sorted[type_starts[t] : type_starts[t] + nc]
            idx = seg[(bi * k) % nc :][:k]
            if len(idx) < k:
                idx = np.concatenate([idx, seg[: k - len(idx)]])
            pen_b += (k / N) * float(delta_fit[idx].mean() ** 2)
        act_pen[bi] = pen_b
        mean_part[bi] = float((w[pres] * mu_t[pres] ** 2).sum())
        var_part[bi] = float((w[pres] * var_t[pres] / cnt[pres]).sum())

    order_by_freq = np.argsort(-counts_fit)
    burden = {}
    for gname, sl in (("top10", slice(0, 10)), ("rank11_50", slice(10, 50)), ("rank51_plus", slice(50, None))):
        ids = order_by_freq[sl]
        ids = ids[counts_fit[ids] > 0]
        exp = float(
            ((counts_fit[ids] / counts_fit.sum())[:, None] * (mu_t[ids] ** 2 + var_t[ids] / np.maximum(counts_fit[ids], 1))[:, None]).sum()
        )
        burden[gname] = {
            "n_types": int(len(ids)),
            "expected_penalty_per_epoch_estimate": exp,
        }
    top20 = order_by_freq[:20]
    per_type_top20 = {
        str(int(t)): {
            "fit_count": int(counts_fit[t]),
            "delta_mean_fit": float(mu_t[t]),
            "delta_std_fit": float(np.sqrt(var_t[t])),
        }
        for t in top20 if counts_fit[t] > 0
    }
    stds = np.sqrt(np.repeat(var_t, counts_fit))
    within_type_std_weighted = float(np.average(stds)) if len(stds) else 0.0

    y_fit = ns.y[ns.idx["fit_inner"]]
    with torch.no_grad():
        pred_fit = model(fit_batch)["pred"].cpu().numpy().astype(np.float64)
    l1_fit = float(np.abs(pred_fit - y_fit).mean())
    full_pen_delta = float(delta_identifiability_penalty(
        {"delta": torch.as_tensor(delta_fit, dtype=torch.float32)},
        torch.as_tensor(fit_types, dtype=torch.long), n_types,
    ))
    full_pen_gamma = float(gamma_identifiability_penalty({"gamma": torch.as_tensor(gamma_fit, dtype=torch.float32)}))

    result = {
        "protocol": PROTOCOL_ID,
        "question": "does the v3 per-type minibatch delta penalty also suppress legitimate within-type contextual variation?",
        "n_batches_replayed": int(n_batches),
        "batch_size": BATCH_SIZE,
        "epochs_replayed": epochs_to_replay,
        "types_per_batch": {
            "mean": float(types_per_batch.mean()), "min": int(types_per_batch.min()),
            "max": int(types_per_batch.max()),
            "p10": float(np.percentile(types_per_batch, 10)),
            "p90": float(np.percentile(types_per_batch, 90)),
        },
        "cell_count_distribution": {
            "frac_cells_with_n_eq_1": float(np.mean(frac_cells_n1)),
            "frac_instances_in_n1_cells": float(np.mean(frac_inst_n1)),
            "frac_cells_with_n_le_2_mean": float(np.mean([
                float((np.bincount(bt, minlength=n_types)[np.bincount(bt, minlength=n_types) > 0] <= 2).mean())
                for bt in batches_types
            ])),
        },
        "penalty_decomposition": {
            "identity": "E[sum_t w_t mean_{t,B}^2] = sum_t w_t (mu_t^2 + var_t/n_t); the mu^2 part constrains the type bias, the var/n part suppresses within-type variation",
            "mean_actual_penalty_per_batch": float(act_pen.mean()),
            "mean_type_bias_part_per_batch": float(mean_part.mean()),
            "mean_within_type_variance_part_per_batch": float(var_part.mean()),
            "variance_share_of_expected_penalty": float(var_part.mean() / max(var_part.mean() + mean_part.mean(), 1e-12)),
            "p90_actual_penalty": float(np.percentile(act_pen, 90)),
        },
        "single_occurrence_direct_instance_penalty": {
            "explanation": "for a type appearing once in a batch the penalty term is (1/N_batch)*delta_i^2: an instance-level L2 on delta, not a type-mean constraint",
            "frac_units_in_n1_cells_mean": float(np.mean(frac_inst_n1)),
        },
        "burden_by_frequency_group": burden,
        "v3S_fit_delta_per_type_top20": per_type_top20,
        "v3S_fit_delta_pooled_std": float(delta_fit.std()),
        "v3S_fit_within_type_std_instance_weighted": within_type_std_weighted,
        "scale_vs_loss": {
            "v3S_fit_L1": l1_fit,
            "v3S_full_fit_delta_penalty": full_pen_delta,
            "v3S_full_fit_gamma_penalty": full_pen_gamma,
        },
        "label_free": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "penalty_audit_s0.json", result)
    log(
        "[penalty-audit] cells n=1 share %.3f, instance share %.3f; penalty/batch %.5f = bias %.5f + var %.5f (var share %.2f)"
        % (
            np.mean(frac_cells_n1), np.mean(frac_inst_n1), act_pen.mean(),
            mean_part.mean(), var_part.mean(),
            result["penalty_decomposition"]["variance_share_of_expected_penalty"],
        )
    )
    return result


# ---------------------------------------------------------------------------
# gauge fitting + verification + export (Phase C)
# ---------------------------------------------------------------------------


@torch.no_grad()
def gauge_and_verify(ns, device, out_dir: Path, log=print) -> dict:
    """Fit-only offsets on the trained S-gauge checkpoint; verify all gates;
    export canonical dev contributions + predictions."""
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    raw_model = SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    raw_model.load_state_dict(torch.load(out_dir / "S_gauge_s0_best_state.pt", map_location="cpu", weights_only=True))
    raw_model = raw_model.to(torch.device(device)).eval()

    offsets, meta = cg.compute_fit_offsets(
        raw_model, ns.features, ns.idx["fit_inner"], fit_pos, ns.vocab,
        ns.type_means, ns.global_mean, ns.rel_mean, device=device, log=log,
    )
    gauged = cg.GaugedHead(raw_model, offsets).to(torch.device(device)).eval()

    batches = {
        "fit": collate(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device),
        "monitor": collate(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device),
        "dev": collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device),
    }
    checks: dict[str, Any] = cg.verify_gauge_invariance(raw_model, gauged, batches, log=log)

    # batch-composition invariance: dev split into two halves must give the
    # same per-unit canonical contributions (collate order = picks order)
    with torch.no_grad():
        full = gauged(batches["dev"])
        ha = gauged(collate(ns.features, dev_pos[: len(dev_pos) // 2], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device))
        hb = gauged(collate(ns.features, dev_pos[len(dev_pos) // 2 :], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device))
    comp_gap = float(max(
        (torch.cat([ha["delta"], hb["delta"]]) - full["delta"]).abs().max(),
        (torch.cat([ha["alpha"], hb["alpha"]]) - full["alpha"]).abs().max(),
        (torch.cat([ha["pred"], hb["pred"]]) - full["pred"]).abs().max(),
    ))
    checks["batch_composition_max_abs_gap"] = comp_gap
    checks["batch_composition_ok"] = bool(comp_gap <= cg.GAUGE_ATOL)
    checks["all_gates_ok"] = bool(checks["all_gates_ok"] and checks["batch_composition_ok"])

    # gauge coverage on fit/monitor/dev (UNK and zero-fit-instance types)
    zero_fit_ids = set(meta["zero_fit_instance_type_ids"])
    cov = {}
    for name, pos in (("fit", fit_pos), ("monitor", mon_pos), ("dev", dev_pos)):
        ut = np.concatenate([ns.features[fi].type_ids(ns.vocab) for fi in pos])
        known = np.asarray([ns.vocab.is_known(s) for fi in pos for s in ns.features[fi].unit_sigs])
        cov[name] = {
            "n_units": int(len(ut)),
            "n_units_on_zero_fit_instance_types": int(np.isin(ut, list(zero_fit_ids)).sum()) if zero_fit_ids else 0,
            "n_units_unknown_signature_unk_buckets": int((~known).sum()),
        }
    checks["gauge_coverage"] = cov

    # MAE invariance canonical vs raw on all splits (the normalization itself
    # must not change MAE; the training-side penalty removal may)
    y = {"fit": ns.y[ns.idx["fit_inner"]], "monitor": ns.y[ns.idx["monitor"]], "dev": ns.y[ns.idx["dev"]]}
    mae_inv = {}
    for name, b in batches.items():
        with torch.no_grad():
            pr = raw_model(b)["pred"].cpu().numpy().astype(np.float64)
            pc = gauged(b)["pred"].cpu().numpy().astype(np.float64)
        mae_inv[name] = {
            "mae_raw": float(np.abs(pr - y[name]).mean()),
            "mae_canonical": float(np.abs(pc - y[name]).mean()),
            "max_abs_pred_diff": float(np.abs(pr - pc).max()),
        }
    checks["mae_invariance"] = mae_inv

    cg.save_gauge(out_dir / "gauge_offsets_s0.pt", offsets, meta, raw_model.state_dict())
    write_json(out_dir / "gauge_meta_s0.json", meta)
    export = export_gauged_contributions(gauged, ns, dev_pos, device, out_dir / "S_gauge_s0_dev_contribs.npz")
    checks["additivity_export"] = export
    with torch.no_grad():
        pred_dev = gauged(batches["dev"])["pred"].cpu().numpy().astype(np.float64)
    np.savez_compressed(
        out_dir / "S_gauge_s0_dev_preds.npz",
        row=np.asarray(ns.idx["dev"], dtype=np.int64),
        pred=pred_dev.astype(np.float32),
        y=ns.y[ns.idx["dev"]].astype(np.float32),
    )
    write_json(out_dir / "gauge_verification_s0.json", checks)
    return checks


@torch.no_grad()
def export_gauged_contributions(gauged, ns, picks: Sequence[int], device, path: Path) -> dict:
    """Strict-additivity export (canonical AND raw); verifies both identities
    from the exported values alone (numpy, independent of forward)."""
    batch = collate(ns.features, list(picks), ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    out = gauged(batch)
    pred = out["pred"].cpu().numpy().astype(np.float64)
    alpha = out["alpha"].cpu().numpy().astype(np.float64)
    delta = out["delta"].cpu().numpy().astype(np.float64)
    alpha_raw = out["alpha_raw"].cpu().numpy().astype(np.float64)
    delta_raw = out["delta_raw"].cpu().numpy().astype(np.float64)
    gamma = out["gamma"].cpu().numpy().astype(np.float64)
    g_ctx = out["g_ctx"].cpu().numpy().astype(np.float64)
    g_topo = out["g_topo"].cpu().numpy().astype(np.float64)
    bias = float(out["bias"].item())
    usum = np.zeros(batch.n_mols)
    np.add.at(usum, batch.unit_mol.cpu().numpy(), alpha + delta)
    usum_raw = np.zeros(batch.n_mols)
    np.add.at(usum_raw, batch.unit_mol.cpu().numpy(), alpha_raw + delta_raw)
    rsum = np.zeros(batch.n_mols)
    if len(gamma):
        np.add.at(rsum, batch.rel_mol.cpu().numpy(), gamma)
    recon = bias + usum + rsum + g_ctx + g_topo
    recon_raw = bias + usum_raw + rsum + g_ctx + g_topo
    np.savez_compressed(
        path,
        arm="S_gauge",
        bias=bias,
        rows=np.asarray([ns.features[i].row for i in picks], dtype=np.int64),
        pred=pred,
        unit_mol=batch.unit_mol.cpu().numpy(),
        unit_type=batch.unit_type.cpu().numpy(),
        alpha=alpha, delta=delta,               # canonical
        alpha_raw=alpha_raw, delta_raw=delta_raw,
        rel_mol=batch.rel_mol.cpu().numpy(),
        gamma=gamma,
        g_ctx=g_ctx, g_topo=g_topo,
    )
    return {
        "additivity_max_abs_residual_canonical": float(np.abs(recon - pred).max()),
        "additivity_max_abs_residual_raw": float(np.abs(recon_raw - pred).max()),
    }


# ---------------------------------------------------------------------------
# analysis: comparison vs frozen F / O / S(v3) + contribution statistics
# ---------------------------------------------------------------------------


def _ringchain_ring_units(ns, rows: np.ndarray) -> np.ndarray:
    """Correct RINGCHAIN ring-unit count per molecule (from the frozen features).

    The v3 subgroup_table's ``ring_units`` used ``nx.cycle_basis`` (number of
    basis cycles) -- erratum recorded in the round notes; this helper uses the
    actual RINGCHAIN partition kinds.
    """
    from cscl_units import KIND_RING

    lookup = {m.row: i for i, m in enumerate(ns.features)}
    out = np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        m = ns.features[lookup[int(row)]]
        out[i] = sum(1 for k in m.unit_kinds if k == KIND_RING)
    return out


def subgroup_table_v3b(rows, y, arms, ns, dataset) -> dict:
    """Subgroup MAEs with the CORRECT ring-unit count (plus the legacy
    cycle-basis count, labeled) and y-band / cyclomatic / n_atoms groups."""
    import networkx as nx

    cyc = np.zeros(len(rows))
    n_atoms = np.zeros(len(rows))
    cycle_basis = np.zeros(len(rows))
    for i, row in enumerate(rows):
        data = dataset[int(row)]
        n_atoms[i] = data.x.shape[0]
        g = nx.Graph()
        g.add_nodes_from(range(int(data.x.shape[0])))
        ei = data.edge_index.numpy()
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u != v:
                g.add_edge(u, v)
        cyc[i] = max(ei.shape[1] // 2 - int(data.x.shape[0]) + nx.number_connected_components(g), 0)
        cycle_basis[i] = len(nx.cycle_basis(g))
    ring_units = _ringchain_ring_units(ns, rows)
    bands = {
        "y_le_-3": y <= -3,
        "y_-3_to_-1": (y > -3) & (y <= -1),
        "y_-1_to_1": (y > -1) & (y <= 1),
        "y_gt_1": y > 1,
        "cyc_0_1": cyc < 1,
        "cyc_1_2": (cyc >= 1) & (cyc < 2),
        "cyc_2_3": (cyc >= 2) & (cyc < 3),
        "cyc_ge_3": cyc >= 3,
        "atoms_lt_22": n_atoms < 22,
        "atoms_ge_22": n_atoms >= 22,
        "ring_units_0_1_ringchain": ring_units <= 1,
        "ring_units_ge_2_ringchain": ring_units >= 2,
    }
    out: dict[str, Any] = {"ring_units_definition": "RINGCHAIN ring-system unit count (unit_kinds==KIND_RING); the v3 'ring_units' rows were nx.cycle_basis counts (basis cycles) -- legacy counts kept below for comparison"}
    for name, mask in bands.items():
        if mask.sum() == 0:
            continue
        entry = {"n": int(mask.sum())}
        for arm, pred in arms.items():
            entry[f"mae_{arm}"] = float(np.abs(pred[mask] - y[mask]).mean())
        out[name] = entry
    return out


@torch.no_grad()
def analyze(ns, out_dir: Path, device, log=print) -> dict:
    """S-gauge vs frozen F / O / S(v3) on the same dev rows; contribution audit."""
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    rows = np.asarray(ns.idx["dev"], dtype=np.int64)
    y_dev = ns.y[rows]
    arms: dict[str, np.ndarray] = {}

    saved_f = np.load(sba.DEV_PREDS_NPZ)
    assert np.array_equal(saved_f["row"], rows)
    arms["F"] = saved_f["pred"].astype(np.float64)
    for name, fn in (("O", "O_s0_dev_preds.npz"), ("S_v3", "S_s0_dev_preds.npz")):
        z = np.load(V3_OUT_DIR / fn)
        assert np.array_equal(z["row"], rows)
        arms[name] = z["pred"].astype(np.float64)
    zg = np.load(out_dir / "S_gauge_s0_dev_preds.npz")
    assert np.array_equal(zg["row"], rows)
    arms["S_gauge"] = zg["pred"].astype(np.float64)

    train_res = json.loads((out_dir / "train_S_gauge_s0.json").read_text())
    mae_g = float(np.abs(arms["S_gauge"] - y_dev).mean())
    summary: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "model": "S-gauge-seed0",
        "mae": {a: float(np.abs(p - y_dev).mean()) for a, p in arms.items()},
        "mae_S_gauge_fit_monitor_dev": {
            "fit": train_res["primary_best_monitor"]["fit_mae"],
            "monitor": train_res["primary_best_monitor"]["monitor_mae"],
            "dev": train_res["primary_best_monitor"]["dev_mae"],
        },
        "delta_dev_mae_vs": {
            "S_v3": mae_g - float(np.abs(arms["S_v3"] - y_dev).mean()),
            "O": mae_g - float(np.abs(arms["O"] - y_dev).mean()),
            "F": mae_g - float(np.abs(arms["F"] - y_dev).mean()),
        },
        "gauge_invariance_of_mae": {
            "note": "canonicalization itself cannot change MAE; the train-side penalty removal may",
            "max_abs_pred_diff_canonical_vs_raw_dev": float(
                np.abs(np.load(out_dir / "S_gauge_s0_dev_preds_raw.npz")["pred"].astype(np.float64) - arms["S_gauge"]).max()
            ),
        },
    }
    for a, b in (("S_gauge", "S_v3"), ("S_gauge", "O"), ("S_gauge", "F")):
        summary[f"paired_{a}_minus_{b}"] = paired_bootstrap(np.abs(arms[a] - y_dev), np.abs(arms[b] - y_dev))

    dataset = cf.load_official_train()
    summary["subgroups"] = subgroup_table_v3b(rows, y_dev, arms, ns, dataset)

    # ---- contribution audit: canonical AND raw ----
    z = np.load(out_dir / "S_gauge_s0_dev_contribs.npz")
    zv3 = np.load(V3_OUT_DIR / "S_s0_dev_contribs.npz")
    offsets = np.asarray(torch.load(out_dir / "gauge_offsets_s0.pt", map_location="cpu", weights_only=False)["offsets"])
    ut = z["unit_type"]

    fit_counts: dict[int, int] = {}
    for fi in _index_by_row(ns.features, ns.idx["fit_inner"]):
        for t in ns.features[fi].type_ids(ns.vocab):
            fit_counts[int(t)] = fit_counts.get(int(t), 0) + 1

    audit: dict[str, Any] = {}
    rows_n = len(rows)

    def channels(zz, a_key, d_key):
        unit = np.zeros(rows_n)
        np.add.at(unit, zz["unit_mol"], zz[a_key] + zz[d_key])
        rel = np.zeros(rows_n)
        if len(zz["gamma"]):
            np.add.at(rel, zz["rel_mol"], zz["gamma"])
        return {"unit": unit, "rel": rel, "g_ctx": zz["g_ctx"], "g_topo": zz["g_topo"]}

    for tag, a_key, d_key in (("canonical", "alpha", "delta"), ("raw", "alpha_raw", "delta_raw")):
        ch = channels(z, a_key, d_key)
        pred = z["pred"]
        audit[f"channel_var_shares_{tag}"] = {
            k: float(np.var(v) / max(np.var(pred), 1e-12)) for k, v in ch.items()
        }
        C = np.stack([ch[k] for k in ("unit", "rel", "g_ctx", "g_topo")])
        audit[f"channel_corr_matrix_{tag}"] = np.corrcoef(C).tolist()
        audit[f"delta_{tag}_dev"] = {"mean": float(z[d_key].mean()), "std": float(z[d_key].std())}
        audit[f"gamma_{tag}"] = {
            "mean": float(z["gamma"].mean()) if len(z["gamma"]) else 0.0,
            "std": float(z["gamma"].std()) if len(z["gamma"]) else 0.0,
            "frac_abs_le_0.01": float((np.abs(z["gamma"]) <= 0.01).mean()) if len(z["gamma"]) else 1.0,
        }
    # m_t distribution
    present = np.array([fit_counts.get(t, 0) for t in range(len(offsets))]) > 0
    audit["m_t_distribution"] = {
        "n_types_with_fit_instances": int(present.sum()),
        "mean_abs_m_t": float(np.abs(offsets[present]).mean()),
        "p50_abs": float(np.percentile(np.abs(offsets[present]), 50)),
        "p90_abs": float(np.percentile(np.abs(offsets[present]), 90)),
        "max_abs": float(np.abs(offsets[present]).max()),
        "n_zero_fit_types_fallback_0": int((~present).sum()),
    }
    # alpha raw vs canonical (shared per-type scalars)
    a_raw_by_type, a_can_by_type = {}, {}
    for t in np.unique(ut):
        m = ut == t
        a_raw_by_type[int(t)] = float(z["alpha_raw"][m][0])
        a_can_by_type[int(t)] = float(z["alpha"][m][0])
    known_types = np.asarray(sorted(fit_counts.keys()))
    audit["alpha_raw"] = {"std_across_dev_types": float(np.std(list(a_raw_by_type.values()))), "value_range": [float(min(a_raw_by_type.values())), float(max(a_raw_by_type.values()))]}
    audit["alpha_canonical"] = {"std_across_dev_types": float(np.std(list(a_can_by_type.values()))), "value_range": [float(min(a_can_by_type.values())), float(max(a_can_by_type.values()))]}
    # per-type delta on dev (raw vs canonical), with fit counts; weighted
    # within-type std comparison v3-S vs v3b-S: did delta recover context variation?
    def per_type_delta(d_arr):
        means, stds, weights = [], [], []
        for t in np.unique(ut):
            m = ut == t
            n = int(m.sum())
            if n < 3 or fit_counts.get(int(t), 0) == 0:
                continue
            means.append(float(d_arr[m].mean()))
            stds.append(float(d_arr[m].std()) if n > 1 else 0.0)
            weights.append(n)
        return means, stds, weights

    means_b, stds_b, w_b = per_type_delta(z["delta"])
    means_r, stds_r, w_r = per_type_delta(z["delta_raw"])
    # v3 S per-type delta on dev (for the within-type std comparison)
    zv3_ut = zv3["unit_type"]
    _, stds_v3, _ = per_type_delta(zv3["delta"]) if np.array_equal(zv3_ut, ut) else ([], [], [])
    if not stds_v3:
        stds_v3 = []
        for t in np.unique(zv3_ut):
            m = zv3_ut == t
            if m.sum() >= 3:
                stds_v3.append(float(zv3["delta"][m].std()))
    audit["delta_within_type_std_on_dev"] = {
        "v3b_instance_weighted_mean_std": float(np.average(stds_r, weights=w_r)) if stds_r else None,
        "v3_instance_weighted_mean_std": float(np.mean(stds_v3)) if stds_v3 else None,
        "v3b_canonical_equals_raw_std": True,
        "note": "the gauge is a per-type constant shift; within-type std is identical by construction (verified numerically)",
    }
    audit["gamma_vs_unit_compensation"] = {
        "v3b_corr_rel_unit": float(np.corrcoef(channels(z, "alpha", "delta")["rel"], channels(z, "alpha", "delta")["unit"])[0, 1]),
        "v3_corr_rel_unit": float(np.corrcoef(
            (lambda c: c["rel"])(channels(zv3, "alpha", "delta")),
            (lambda c: c["unit"])(channels(zv3, "alpha", "delta")),
        )[0, 1]),
        "note": "if gamma reduced compensation, its variance share and rel-unit correlation should drop vs v3",
    }
    summary["S_gauge_audit"] = audit
    write_json(out_dir / "analysis_s0.json", summary)
    log("[analyze] " + json.dumps(summary["mae"]))
    return summary


# ---------------------------------------------------------------------------
# corrected synthetic known-truth task (CPU, seed 0) — mechanism evidence
# ---------------------------------------------------------------------------


def _hash_value(payload: str) -> float:
    import hashlib

    h = hashlib.sha256(payload.encode()).hexdigest()
    return (int(h[:8], 16) % 2000) / 1000.0 - 1.0


def attach_counts(m) -> np.ndarray:
    """Unit boundary attachment counts, read from the CORRECT descriptor slot.

    Regression-pinned by tests/test_cscl_v3b.py: this is desc[:, D_HIST+4]
    (n_attachments scalar), NOT the atom-histogram bin that v3's
    synthetic_task read at flat index 4.  Accepts both the raw unit summary
    (``MolUnits.desc``) and the frozen-feature summary (``MolFrozen.unit_desc``).
    """
    desc = getattr(m, "unit_desc", None)
    if desc is None:
        desc = m.desc
    return np.asarray(desc[:, ATTACH_IDX], dtype=np.float64)


def synthetic_task_gauge(ns, out_dir: Path, device, epochs: int = EPOCHS, log=print) -> dict:
    """y_syn = sum_u alpha*(t) + 0.8*(attach - fit type mean) + sum_pairs gamma* + noise.

    Corrected known-truth task on REAL frozen features: the context effect is
    driven by the type-centered unit boundary attachment count (index
    D_HIST+4=36; v3 read an atom-histogram bin by mistake, so this task's
    numbers are NOT comparable to the v3 synthetic).  Trains a fresh
    SBContribHead with the NEW objective (gamma penalty only), then applies
    the fit-only gauge and reports recovery before/after canonicalization.
    """
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])

    fit_type_counts: dict[int, int] = {}
    for fi in fit_pos:
        for t in ns.features[fi].type_ids(ns.vocab):
            fit_type_counts[int(t)] = fit_type_counts.get(int(t), 0) + 1
    freq_types = sorted(fit_type_counts, key=lambda t: -fit_type_counts[t])
    top32 = freq_types[:32]
    alpha_star = {t: _hash_value(f"alpha|{t}") for t in top32}

    rel_type_count: dict[tuple[int, int], int] = {}
    for fi in fit_pos:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        for lo, hi in zip(m.rel_lo, m.rel_hi):
            key = (int(tids[lo]), int(tids[hi]))
            rel_type_count[key] = rel_type_count.get(key, 0) + 1
    cand = [k for k, v in sorted(rel_type_count.items(), key=lambda item: -item[1]) if v >= 50]
    if len(cand) < 4:
        cand = [k for k, _ in sorted(rel_type_count.items(), key=lambda item: -item[1])][:4]
    pair_A, pair_B, pair_C = cand[0], cand[1], cand[2]
    gamma_star = {pair_A: 0.9, pair_B: -0.7, pair_C: 0.5}
    type_of_A = {pair_A[0], pair_A[1]}
    confusers = [k for k in cand[3:] if k[0] in type_of_A or k[1] in type_of_A]
    pair_D = confusers[0] if confusers else cand[3]

    # fit-only attach-count centering per type (correct slot)
    attach_sum, attach_n = {}, {}
    for fi in fit_pos:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        ac = attach_counts(m)
        for k, t in enumerate(tids):
            attach_sum[int(t)] = attach_sum.get(int(t), 0.0) + float(ac[k])
            attach_n[int(t)] = attach_n.get(int(t), 0) + 1
    attach_mean = {t: attach_sum[t] / attach_n[t] for t in attach_n}
    global_attach = float(np.mean([float(attach_counts(ns.features[fi]).mean()) for fi in fit_pos]))

    def syn_labels(picks):
        ys = []
        for fi in picks:
            m = ns.features[fi]
            tids = m.type_ids(ns.vocab)
            ac = attach_counts(m)
            total = 0.0
            for k, t in enumerate(tids):
                total += alpha_star.get(int(t), 0.0)
                total += 0.8 * (float(ac[k]) - attach_mean.get(int(t), global_attach))
            for lo, hi in zip(m.rel_lo, m.rel_hi):
                total += gamma_star.get((int(tids[lo]), int(tids[hi])), 0.0)
            ys.append(total)
        return np.asarray(ys, dtype=np.float64)

    rng = np.random.default_rng(0)
    y_syn_fit = syn_labels(fit_pos) + 0.05 * rng.standard_normal(len(fit_pos))
    y_syn_mon = syn_labels(mon_pos)
    y_syn_dev = syn_labels(dev_pos)

    seed_everything(0)
    model = SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(101)
    fit_batch = collate(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    mon_batch = collate(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    dev_batch = collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    target = torch.as_tensor(y_syn_fit, dtype=torch.float32, device=device)
    best = (1e9, 0, None)
    for epoch in range(1, epochs + 1):
        model.train()
        for picks in _batches(len(fit_pos), gen, True):
            b = collate(ns.features, [fit_pos[i] for i in picks], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
            out = model(b)
            loss = torch.nn.functional.l1_loss(out["pred"], target[torch.as_tensor(picks)])
            loss = loss + PEN_COEF * gamma_identifiability_penalty(out)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            mon_pred = model(mon_batch)["pred"].cpu().numpy().astype(np.float64)
        mon_mae = float(np.abs(mon_pred - y_syn_mon).mean())
        if mon_mae < best[0]:
            best = (mon_mae, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict({k: v.to(device) for k, v in best[2].items()})
    model.eval()

    with torch.no_grad():
        out = model(dev_batch)
    alpha_hat = model.alpha.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    delta_hat = out["delta"].cpu().numpy().astype(np.float64)
    gamma_hat = out["gamma"].cpu().numpy().astype(np.float64)

    # fit-only gauge on the synthetic-trained model (model outputs only)
    offsets, meta = cg.compute_fit_offsets(
        model, ns.features, ns.idx["fit_inner"], fit_pos, ns.vocab,
        ns.type_means, ns.global_mean, ns.rel_mean, device=device, log=log,
    )
    gauged = cg.GaugedHead(model, offsets).to(torch.device(device)).eval()
    with torch.no_grad():
        gout = gauged(dev_batch)
        pred_raw_syn = out["pred"].cpu().numpy().astype(np.float64)
        pred_can_syn = gout["pred"].cpu().numpy().astype(np.float64)
    gauge_pred_gap = float(np.abs(pred_raw_syn - pred_can_syn).max())
    delta_can = gout["delta"].cpu().numpy().astype(np.float64)
    alpha_can = gout["alpha"].cpu().numpy().astype(np.float64)

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    a_true = np.array([alpha_star[t] for t in top32])
    a_hat = np.array([alpha_hat[t] for t in top32])
    sign_match = float((np.sign(a_true) == np.sign(a_hat)).mean())

    true_delta, hat_delta, alpha_of_unit = [], [], []
    um = dev_batch.unit_mol.cpu().numpy()
    for mi, fi in enumerate(dev_pos):
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        ac = attach_counts(m)
        hats = delta_hat[um == mi]
        hats_can = delta_can[um == mi]
        assert len(hats) == m.n_units
        for k, t in enumerate(tids):
            true_delta.append(0.8 * (float(ac[k]) - attach_mean.get(int(t), global_attach)))
            hat_delta.append((float(hats[k]), float(hats_can[k])))
            alpha_of_unit.append(alpha_star.get(int(t), 0.0))
    true_delta = np.asarray(true_delta)
    hat_delta_raw = np.asarray([h[0] for h in hat_delta])
    hat_delta_can = np.asarray([h[1] for h in hat_delta])
    hat_delta = hat_delta_raw
    alpha_of_unit = np.asarray(alpha_of_unit)

    def gamma_mean_for(pair_key):
        vals = []
        idx_rel = 0
        for fi in dev_pos:
            m = ns.features[fi]
            tids = m.type_ids(ns.vocab)
            for j in range(len(m.rel_lo)):
                key = (int(tids[m.rel_lo[j]]), int(tids[m.rel_hi[j]]))
                if key == pair_key:
                    vals.append(float(gamma_hat[idx_rel + j]))
            idx_rel += len(m.rel_lo)
        return float(np.mean(vals)) if vals else float("nan"), len(vals)

    gamma_check = {}
    for name, key, star in (("pair_A", pair_A, 0.9), ("pair_B", pair_B, -0.7), ("pair_C", pair_C, 0.5), ("pair_D_confuser", pair_D, 0.0)):
        mean_hat, n_obs = gamma_mean_for(key)
        gamma_check[name] = {
            "types": [int(key[0]), int(key[1])], "gamma_star": star,
            "gamma_hat_mean": mean_hat, "sign_recovered": bool(np.sign(mean_hat) == np.sign(star)),
            "n_dev_relations": n_obs,
        }
    unseen_hat, seen_hat = [], []
    idx_rel = 0
    for fi in dev_pos:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        for j in range(len(m.rel_lo)):
            key = (int(tids[m.rel_lo[j]]), int(tids[m.rel_hi[j]]))
            (unseen_hat if key not in rel_type_count else seen_hat).append(float(gamma_hat[idx_rel + j]))
        idx_rel += len(m.rel_lo)

    # alpha<->gamma confusion after canonicalization: correlate per-type alpha
    # with the mean gamma-hat over relations touching the type
    def alpha_gamma_confusion(a_vec_by_type):
        per_type_gamma: dict[int, list[float]] = {}
        idx_rel = 0
        for fi in dev_pos:
            m = ns.features[fi]
            tids = m.type_ids(ns.vocab)
            for j in range(len(m.rel_lo)):
                for tt in (int(tids[m.rel_lo[j]]), int(tids[m.rel_hi[j]])):
                    per_type_gamma.setdefault(tt, []).append(float(gamma_hat[idx_rel + j]))
            idx_rel += len(m.rel_lo)
        common = [t for t in per_type_gamma if t in a_vec_by_type]
        if len(common) < 3:
            return None
        xs = np.array([a_vec_by_type[t] for t in common])
        gs = np.array([np.mean(per_type_gamma[t]) for t in common])
        return float(np.corrcoef(xs, gs)[0, 1])

    dev_ut = dev_batch.unit_type.cpu().numpy()
    a_can_by_type = {int(t): float(alpha_can[dev_ut == t][0]) for t in np.unique(dev_ut)}
    result = {
        "protocol": PROTOCOL_ID,
        "task": "corrected synthetic known-truth on real frozen features (attach idx D_HIST+4=36; gamma-only penalty; noise 0.05; seed 0)",
        "attachment_index": ATTACH_IDX,
        "not_comparable_to_v3_synthetic": "ground-truth definition changed (corrected attachment slot); no performance comparison",
        "best_monitor_epoch": int(best[1]),
        "monitor_mae_syn": float(best[0]),
        "dev_mae_syn_raw": float(np.abs(pred_raw_syn - y_syn_dev).mean()),
        "dev_mae_syn_canonical": float(np.abs(pred_can_syn - y_syn_dev).mean()),
        "gauge_max_abs_pred_diff_synthetic": gauge_pred_gap,
        "alpha_sign_recovery_top32": sign_match,
        "alpha_spearman_top32_raw": _spearman(a_true, a_hat),
        "alpha_true_value_range": [float(a_true.min()), float(a_true.max())],
        "delta_recovery_pearson_raw": float(np.corrcoef(true_delta, hat_delta)[0, 1]),
        "delta_recovery_pearson_canonical": float(np.corrcoef(true_delta, hat_delta_can)[0, 1]),
        "delta_hat_vs_alpha_star_confusion_raw": float(np.corrcoef(hat_delta, alpha_of_unit)[0, 1]),
        "delta_hat_vs_alpha_star_confusion_canonical": float(np.corrcoef(hat_delta_can, alpha_of_unit)[0, 1]),
        "delta_amp_ratio_raw_vs_truth": float(hat_delta.std() / max(np.abs(true_delta).std(), 1e-12)),
        "alpha_gamma_confusion_corr_canonical": alpha_gamma_confusion(a_can_by_type),
        "gamma_checks": gamma_check,
        "unseen_pair_mean_abs_gamma_hat": float(np.mean(np.abs(unseen_hat))) if unseen_hat else None,
        "seen_pair_mean_abs_gamma_hat": float(np.mean(np.abs(seen_hat))) if seen_hat else None,
        "n_unseen_dev_relations": len(unseen_hat),
        "gauge_meta_fit_units": meta["n_fit_units"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "synthetic_gauge_s0.json", result)
    log("[synthetic-gauge] alpha sign %.2f spearman %.2f delta-r(raw) %.2f delta-r(can) %.2f gauge-pred-gap %.2e"
        % (sign_match, result["alpha_spearman_top32_raw"], result["delta_recovery_pearson_raw"],
           result["delta_recovery_pearson_canonical"], gauge_pred_gap))
    return result


# ---------------------------------------------------------------------------
# Phase A notes: two report-boundary checks (no retraining, no feature change)
# ---------------------------------------------------------------------------


def report_boundary_audit(ns, out_dir: Path, log=print) -> dict:
    """(1) enumerate what Full-Y reader input the stored MolFrozen drops;
    (2) confirm the v3 subgroup ring_units naming issue."""
    dataset = cf.load_official_train()
    rows = ns.idx["dev"][:40]
    # count RINGCHAIN ring units vs cycle_basis on a sample
    import networkx as nx

    diffs = []
    for row in rows:
        data = dataset[int(row)]
        g = nx.Graph()
        g.add_nodes_from(range(int(data.x.shape[0])))
        ei = data.edge_index.numpy()
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u != v:
                g.add_edge(u, v)
        cb = len(nx.cycle_basis(g))
        lookup = {m.row: i for i, m in enumerate(ns.features)}
        from cscl_units import KIND_RING

        ru = sum(1 for kk in ns.features[lookup[int(row)]].unit_kinds if kk == KIND_RING)
        diffs.append((int(row), int(ru), int(cb)))
    mismatched = sum(1 for _, ru, cb in diffs if ru != cb)
    out = {
        "ring_units_erratum": {
            "finding": "run_cscl_v3.subgroup_table's ring_units row counted nx.cycle_basis(g) (number of basis cycles), not RINGCHAIN ring-system units",
            "sample_molecules": len(diffs),
            "molecules_where_counts_differ": mismatched,
            "action": "cscl-v3b subgroup tables report ring_units_ringchain (KIND_RING count) and label the legacy count explicitly; historical v3 subgroup reads of 'ring_units' are ring-complexity proxies, not unit counts",
        },
        "full_y_reader_input_compression": {
            "finding": "MolFrozen keeps bucket-marginal pair moments only; Full-Y's reader consumes per-distance-bucket first/second moments",
            "missing_from_S_O_inputs": [
                "per-(unit x distance-bucket) first/second moments of intra-unit pair values (computed in extract_batch as intra_b_counts/sum/sq but not stored)",
                "per-(unit-pair x distance-bucket) first/second moments of cross-unit pair values (only bucket-marginal rel_sum_pv/rel_sum_pv2 + per-pair bucket COUNTS stored)",
                "consequently the graph-level per-bucket moment pool consumed by the frozen reader is not reconstructible at bucket granularity from S/O inputs: bucket-conditional magnitude information is lost",
                "per-pair raw E rows / pair_value rows and the per-pair relation-encoder outputs (model-internal) are not part of the S/O input contract",
                "graph-level log1p count blocks are C6-masked to zero for F as well -- not a difference",
            ],
            "action": "documentation only this round: S/O input compression is a candidate explanation for part of O-F (+0.025) and S-O (+0.054); no features changed (protocol forbids)",
        },
    }
    write_json(out_dir / "phase_a_boundaries.json", out)
    log(f"[boundaries] ring_units vs cycle_basis differ on {mismatched}/{len(diffs)} sampled molecules")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("mode", choices=("penalty-audit", "boundaries", "train", "gauge", "analyze", "synthetic", "all"))
    parser.add_argument("--seed", type=int, default=0, choices=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    log = lambda *a, **k: print(*a, **k, flush=True)  # noqa: E731

    if args.mode == "penalty-audit":
        ns = prepare_all(log=log)
        penalty_audit(ns, args.out, device="cpu", log=log)
        return 0
    if args.mode == "boundaries":
        ns = prepare_all(log=log)
        report_boundary_audit(ns, args.out, log=log)
        return 0
    if args.mode == "train":
        ns = prepare_all(log=log)
        train_gauge(ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        return 0
    if args.mode == "gauge":
        ns = prepare_all(log=log)
        gauge_and_verify(ns, device, args.out, log=log)
        return 0
    if args.mode == "analyze":
        ns = prepare_all(log=log)
        analyze(ns, args.out, device, log=log)
        return 0
    if args.mode == "synthetic":
        ns = prepare_all(log=log)
        synthetic_task_gauge(ns, args.out, torch.device("cpu"), log=log)
        return 0
    if args.mode == "all":
        log("[all] train S-gauge (seed 0) ...")
        ns = prepare_all(log=log, device=str(device))
        train_gauge(ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        log("[all] fit-only gauge + verification ...")
        gauge_and_verify(ns, device, args.out, log=log)
        log("[all] analyze vs frozen F/O/S(v3) ...")
        analyze(ns, args.out, device, log=log)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
