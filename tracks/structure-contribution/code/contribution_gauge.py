"""Fit-only gauge normalization for the SB-CSCL structured head (cscl-v3b).

Research track: ``structure-contribution``, round ``cscl-v3b``.  Protocol:
``configs/cscl_v3b_gauge.yaml`` (frozen before the S-gauge training run).

Mathematical object
-------------------
The structured head's prediction only depends on ``alpha(t) + delta(i)``, so
the decomposition into the shared type contribution and the contextual
correction has one constant per-type shift freedom (a gauge freedom):

    m_t = mean of delta_raw over ALL fit_inner unit instances of type t
    alpha_canonical(t) = alpha_raw(t) + m_t
    delta_canonical(i) = delta_raw(i) - m_{t(i)}

This module implements that gauge fixing as a *prediction-invariant,
fit-only, post-training* transformation:

* ``m_t`` is computed from fit_inner units and frozen model outputs only
  (no monitor/dev labels, no monitor/dev structural statistics);
* types with zero fit instances (expected: only the GLOBAL_UNK bucket)
  get the deterministic fallback ``m_t = 0``; the count is reported;
* offsets are stored as a frozen buffer shipped with the checkpoint; the
  raw trained parameters are never modified;
* :class:`GaugedHead` evaluates ``alpha_embedding[type] + offset[type]`` and
  ``delta_mlp(local_input) - offset[type]`` so canonical contributions are
  produced without touching trained weights.

What this does NOT do (research_core sect. 5 honesty): it fixes only the
alpha/delta constant shift freedom.  It does not resolve alpha/delta vs gamma
transfer, relation/global statistical dependence, overlapping-context
attribution ambiguity, chemical causality, or unseen-type stability.

This module never reads official valid/test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import strong_backbone_attribution as sba

PROTOCOL_ID = "cscl-v3b-gauge-normalization"
GAUGE_ATOL = 1e-5          # pred/term identity tolerance (float32 forward)
FIT_MEAN_ATOL = 1e-6       # per-type fit mean of canonical delta (float64)
# The gauge shift is EXACT in real arithmetic (a per-type constant moves from
# delta to alpha); the tolerance below only covers float32 output rounding of
# the stored contributions (observed ~6e-8 on real heads).
STD_ATOL = 1e-6            # within-type std / pairwise-diff preservation


# ---------------------------------------------------------------------------
# fit-only offsets
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_fit_offsets(
    model,
    features: Sequence[sba.MolFrozen],
    fit_rows: Sequence[int],
    picks: Sequence[int],
    vocab,
    type_means,
    global_mean,
    rel_mean,
    device="cpu",
    log=print,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``m_t`` per type id from ALL fit_inner unit instances (float64).

    ``picks`` are the feature-list indices of the fit_inner rows (order of
    ``run_cscl_v3._index_by_row``).  The model must be in eval mode; its
    parameters are not modified.  Returns ``(offsets [vocab_size] float64,
    meta)``.
    """
    from run_cscl_v3 import collate  # reuse the exact v3 collate (same inputs)

    was_training = model.training
    model.eval()
    offsets = np.zeros(vocab.n_total, dtype=np.float64)
    counts = np.zeros(vocab.n_total, dtype=np.int64)
    batch = collate(features, list(picks), vocab, type_means, global_mean, rel_mean).to(device)
    delta = model(batch)["delta"].detach().cpu().numpy().astype(np.float64)
    unit_type = batch.unit_type.cpu().numpy()
    sums = np.zeros(vocab.n_total, dtype=np.float64)
    np.add.at(sums, unit_type, delta)
    np.add.at(counts, unit_type, 1)
    present = counts > 0
    offsets[present] = sums[present] / counts[present]
    # deterministic fallback for zero-fit-instance type buckets: m_t = 0
    zero_ids = np.flatnonzero(~present)
    meta = {
        "n_fit_units": int(counts.sum()),
        "n_types_total": int(vocab.n_total),
        "n_types_with_fit_instances": int(present.sum()),
        "n_types_zero_fit_instances": int((~present).sum()),
        "zero_fit_instance_type_ids": zero_ids.tolist(),
        "fallback_value": 0.0,
        "fallback_rule": "m_t = 0.0 for type buckets with zero fit_inner instances",
        "fit_unit_type_hash": _hash_array(unit_type),
        "fit_rows_hash": _hash_array(np.asarray(sorted(int(r) for r in fit_rows), dtype=np.int64)),
        "counts_hash": _hash_array(counts),
        "offsets_hash": _hash_array(offsets),
        "min_fit_count": int(counts.min()),
        "max_fit_count": int(counts.max()),
    }
    log(
        f"[gauge] m_t computed on {meta['n_fit_units']} fit units; "
        f"{meta['n_types_with_fit_instances']}/{meta['n_types_total']} types have fit instances; "
        f"fallback ids {meta['zero_fit_instance_type_ids']}"
    )
    if was_training:
        model.train()
    return offsets, meta


def _hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# gauged head (frozen wrapper; raw parameters untouched)
# ---------------------------------------------------------------------------


class GaugedHead(nn.Module):
    """Wraps a trained (frozen) structured head and applies the gauge offsets.

    alpha_canonical = alpha_embedding[type] + offset[type]
    delta_canonical = delta_mlp(local_input) - offset[type]

    The unit terms are delegated to ``head.unit_terms(batch)`` -> (alpha, delta),
    so the SAME gauged wrapper serves any head whose prediction is
    ``bias + sum_u(alpha_u + delta_u) + sum_r gamma + g_ctx + g_topo``
    (cscl-v3/v3b ``SBContribHead`` and the cscl-v3c type-conditioned head;
    for v3c ``unit_terms`` returns the type-conditioned delta, which does not
    change the gauge algebra: the offset is still a per-type constant).

    ``offset`` is a frozen buffer (not a parameter); gamma / global banks /
    bias are passed through untouched, so the prediction is invariant up to
    float32 rounding and every exported contribution remains strictly
    additive to the same prediction.
    """

    def __init__(self, head: nn.Module, offsets: np.ndarray):
        super().__init__()
        self.head = head
        offset = torch.as_tensor(np.asarray(offsets, dtype=np.float32).copy())
        self.register_buffer("offset", offset)

    def forward(self, batch):
        h = self.head
        alpha_raw, delta_raw = h.unit_terms(batch)
        off = self.offset[batch.unit_type]
        alpha_canon = alpha_raw + off
        delta_canon = delta_raw - off
        gamma = h.relation_terms(batch)
        g_ctx, ctx_pf = h._bank(batch.graph_feat[:, : sba.CTX_STRUCTURE], h.tau_ctx, h.w_ctx)
        g_topo, topo_pf = h._bank(batch.graph_feat[:, sba.CTX_STRUCTURE :], h.tau_topo, h.w_topo)
        unit_sum = torch.zeros(batch.n_mols, device=alpha_canon.device).index_add(
            0, batch.unit_mol, alpha_canon + delta_canon
        )
        rel_sum = torch.zeros(batch.n_mols, device=alpha_canon.device).index_add(0, batch.rel_mol, gamma)
        pred = h.bias + unit_sum + rel_sum + g_ctx + g_topo
        return {
            "pred": pred,
            "alpha": alpha_canon,          # canonical (default export = canonical)
            "delta": delta_canon,
            "alpha_raw": alpha_raw,
            "delta_raw": delta_raw,
            "gamma": gamma,
            "g_ctx": g_ctx,
            "g_topo": g_topo,
            "g_ctx_per_feature": ctx_pf,
            "g_topo_per_feature": topo_pf,
            "bias": h.bias.detach(),
        }


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def save_gauge(path: Path, offsets: np.ndarray, meta: Mapping[str, Any], head_state: Mapping[str, torch.Tensor], protocol_id: str = PROTOCOL_ID) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": protocol_id,
            "offsets": np.asarray(offsets, dtype=np.float64),
            "meta": dict(meta),
            "head_state": head_state,
        },
        path,
    )


def load_gauge(path: Path, vocab_size: int, tau_ctx: np.ndarray, tau_topo: np.ndarray, device="cpu", head_builder=None):
    """Rebuild (GaugedHead, meta) from a gauge checkpoint; strict state load.

    ``head_builder(vocab_size, tau_ctx, tau_topo) -> head`` defaults to the
    cscl-v3/v3b ``SBContribHead``; cscl-v3c passes its type-conditioned head.
    """
    from run_cscl_v3 import SBContribHead

    blob = torch.load(path, map_location="cpu", weights_only=False)
    head = (head_builder or SBContribHead)(vocab_size, tau_ctx, tau_topo)
    head.load_state_dict(blob["head_state"], strict=True)
    offsets = np.asarray(blob["offsets"], dtype=np.float64)
    assert offsets.shape == (vocab_size,)
    gauged = GaugedHead(head, offsets).to(torch.device(device))
    gauged.eval()
    return gauged, blob["meta"]


# ---------------------------------------------------------------------------
# verification (the frozen gates from the protocol yaml)
# ---------------------------------------------------------------------------


def verify_gauge_invariance(
    raw_model,
    gauged: GaugedHead,
    batches: Mapping[str, Any],
    fit_unit_type: np.ndarray | None = None,
    log=print,
) -> dict[str, Any]:
    """Check every frozen gate; returns a metrics dict (all booleans must hold).

    ``batches``: name -> V3Batch.  ``fit_unit_type``: flat type ids of the fit
    batch (for the per-type zero-mean check); when None it is read from
    ``batches["fit"].unit_type``.
    """
    out: dict[str, Any] = {"atol_pred": GAUGE_ATOL, "atol_fit_mean": FIT_MEAN_ATOL, "atol_std": STD_ATOL}
    per_split_ok = True
    for name, b in batches.items():
        raw_model.eval()
        gauged.eval()
        with torch.no_grad():
            r = raw_model(b)
            g = gauged(b)
        d_pred = float((g["pred"] - r["pred"]).abs().max())
        # alpha+delta identity per unit (canonical vs raw)
        d_term = float(
            ((g["alpha"] + g["delta"]) - (r["alpha"] + r["delta"])).abs().max()
        )
        # gamma / global / bias untouched
        d_gamma = float((g["gamma"] - r["gamma"]).abs().max()) if r["gamma"].numel() else 0.0
        d_gctx = float((g["g_ctx"] - r["g_ctx"]).abs().max())
        d_gtopo = float((g["g_topo"] - r["g_topo"]).abs().max())
        d_bias = float((g["bias"] - r["bias"]).abs().max())
        # within-type std preservation (canonical vs raw delta)
        ut = b.unit_type.cpu().numpy()
        dr = r["delta"].cpu().numpy().astype(np.float64)
        dc = g["delta"].cpu().numpy().astype(np.float64)
        std_gap = 0.0
        for t in np.unique(ut):
            m = ut == t
            if m.sum() > 1:
                std_gap = max(std_gap, abs(dr[m].std() - dc[m].std()))
        # pairwise same-type delta differences preserved (exact linear identity,
        # checked numerically on a sample)
        rng = np.random.default_rng(0)
        pair_gap = 0.0
        for t in np.unique(ut):
            idx = np.flatnonzero(ut == t)
            if len(idx) >= 2:
                sel = rng.choice(idx, size=min(len(idx), 8), replace=False)
                a, b_ = np.meshgrid(sel, sel)
                keep = a.ravel() != b_.ravel()
                diff_raw = dr[a.ravel()[keep]] - dr[b_.ravel()[keep]]
                diff_can = dc[a.ravel()[keep]] - dc[b_.ravel()[keep]]
                pair_gap = max(pair_gap, float(np.abs(diff_raw - diff_can).max()))
        entry = {
            "max_abs_pred_diff": d_pred,
            "max_abs_alpha_delta_identity_diff": d_term,
            "max_abs_gamma_diff": d_gamma,
            "max_abs_g_ctx_diff": d_gctx,
            "max_abs_g_topo_diff": d_gtopo,
            "max_abs_bias_diff": d_bias,
            "max_abs_within_type_std_gap": std_gap,
            "max_abs_same_type_pairwise_diff_gap": pair_gap,
            "ok": bool(
                d_pred <= GAUGE_ATOL and d_term <= GAUGE_ATOL and d_gamma == 0.0
                and d_gctx == 0.0 and d_gtopo == 0.0 and d_bias == 0.0
                and std_gap <= STD_ATOL and pair_gap <= STD_ATOL
            ),
        }
        out[name] = entry
        per_split_ok &= entry["ok"]
        log(f"[gauge-verify:{name}] pred|Δ|={d_pred:.2e} term|Δ|={d_term:.2e} ok={entry['ok']}")

    # per-type fit mean of canonical delta == 0 (float64)
    fb = batches["fit"]
    ut = (fit_unit_type if fit_unit_type is not None else fb.unit_type.cpu().numpy())
    with torch.no_grad():
        dc = gauged(fb)["delta"].cpu().numpy().astype(np.float64)
    offsets = gauged.offset.cpu().numpy().astype(np.float64)
    sums = np.zeros(len(offsets), dtype=np.float64)
    counts = np.zeros(len(offsets), dtype=np.int64)
    np.add.at(sums, ut, dc)
    np.add.at(counts, ut, 1)
    present = counts > 0
    means = np.zeros_like(sums)
    means[present] = sums[present] / counts[present]
    out["fit_canonical_delta_per_type_mean_max_abs"] = float(np.abs(means[present]).max())
    out["fit_canonical_delta_per_type_mean_ok"] = bool(
        np.abs(means[present]).max() <= FIT_MEAN_ATOL
    )
    # alpha must remain a shared per-type scalar: canonical alpha equals
    # alpha_raw + offset (identity by construction; verify numerically)
    with torch.no_grad():
        g_fit = gauged(fb)
        r_fit = raw_model(fb)
    alpha_id = float(
        ((g_fit["alpha"] - g_fit["alpha_raw"]) - gauged.offset[fb.unit_type]).abs().max()
    )
    out["alpha_offset_identity_max_abs"] = alpha_id
    out["alpha_is_shared_type_scalar"] = bool(alpha_id <= GAUGE_ATOL)
    out["all_gates_ok"] = bool(per_split_ok and out["fit_canonical_delta_per_type_mean_ok"])
    log(f"[gauge-verify] fit per-type |mean| max = {out['fit_canonical_delta_per_type_mean_max_abs']:.2e}; all_ok={out['all_gates_ok']}")
    return out
