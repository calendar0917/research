"""Phase-D conditional probes on the frozen Full representation.

Frozen protocol (written before the fit was inspected):
  baseline  B : [1, H2]                     (H2 = 39-D second reader hidden)
  probe  P1  : [1, H2, Z/scales]            Z  = per-graph [sum(z), sum(z^2)]
                                                of the 446-D pre-fusion interface
  probe  P2  : [1, H2, NJ/scale]            NJ = per-graph sum_i outer(coord_i, q_i)
                                                coord 33-D structural code, q atom onehot 28
  control C1 : [1, H2, P/scale]             P  = fixed random projection of H2 to the
                                                same width as P1 (regularization control)
  control C2 : [1, H2, P/scale]             same for P2
Solver: reused upstream-prototype MAE+L2 ADMM (lambda 1e-5, gap 1e-6).  Train-fit only.
No backbone training, no official test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tracks.ksvd.experiments.luyin16.upstream_portfolio_v1_reference.prototype_dictionary import fit_mae
from tracks.ksvd.experiments.luyin16.zinc_upstream_portfolio_v1 import (
    CONDITIONAL_GAP,
    CONDITIONAL_LAMBDA,
)

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "tracks/ksvd/results/zinc_full_bottleneck_audit_v1"
FEAT = OUT / "probe_features.npz"
GATE_GAIN = 0.003
TIME_BUDGET = 600


def _scale(train_block):
    s = float(np.sqrt(np.mean(train_block ** 2)))
    return s if (np.isfinite(s) and s > 1e-12) else 1.0


def fit(Dtr, ytr, Dva, yva, tag):
    coef, status = fit_mae(Dtr, ytr, CONDITIONAL_LAMBDA, max_iterations=12000,
                           gap_tolerance=CONDITIONAL_GAP, time_budget_s=TIME_BUDGET)
    pred = Dva @ coef
    mae = float(np.mean(np.abs(pred - yva)))
    return coef, status, pred, mae


def group_delta(cand, base, y):
    d = np.abs(base - y) - np.abs(cand - y)  # positive = candidate better
    ids = np.arange(len(y))
    out = {"groups_gain": [], "groups_improved": 0}
    for g in range(5):
        m = (ids % 5) == g
        v = float(d[m].mean())
        out["groups_gain"].append(v)
        if v > 0:
            out["groups_improved"] += 1
    i172 = int(np.argmax(np.abs(base - y)))
    out["ex172_gain"] = float((d.sum() - d[i172]) / (len(d) - 1))
    out["id172_gain"] = float(d[i172])
    out["ex172_index"] = i172
    return out


def main():
    d = np.load(FEAT)
    H2t, H2v = d["H2t"].astype(np.float64), d["H2v"].astype(np.float64)
    Zt, Zv = d["Zt"].astype(np.float64), d["Zv"].astype(np.float64)
    NJt, NJv = d["NJt"].astype(np.float64), d["NJv"].astype(np.float64)
    yt, yv = d["yt"].astype(np.float64), d["yv"].astype(np.float64)

    one_t = np.ones((len(yt), 1))
    one_v = np.ones((len(yv), 1))

    # baseline
    Dtr = np.column_stack([one_t, H2t]); Dva = np.column_stack([one_v, H2v])
    _, bst, bpred, bmae = fit(Dtr, yt, Dva, yv, "baseline")
    results = {"baseline": {"valid_mae": bmae, "status": bst["status"]}}

    # candidate P1: pre-fusion interface, per-half train RMS
    half = Zt.shape[1] // 2
    s1, s2 = _scale(Zt[:, :half]), _scale(Zt[:, half:])
    Zts = np.concatenate([Zt[:, :half] / s1, Zt[:, half:] / s2], 1)
    Zvs = np.concatenate([Zv[:, :half] / s1, Zv[:, half:] / s2], 1)
    Dtr = np.column_stack([one_t, H2t, Zts]); Dva = np.column_stack([one_v, H2v, Zvs])
    _, st, pred, mae = fit(Dtr, yt, Dva, yv, "P1")
    results["P1_prefix_interface"] = {
        "valid_mae": mae, "gain_vs_baseline": bmae - mae, "status": st["status"],
        "width": int(Zts.shape[1]), "scales": [s1, s2], **group_delta(pred, bpred, yv),
    }

    # candidate P2: node structural-code x atom-type joint
    s = _scale(NJt)
    NJts, NJvs = NJt / s, NJv / s
    Dtr = np.column_stack([one_t, H2t, NJts]); Dva = np.column_stack([one_v, H2v, NJvs])
    _, st, pred, mae = fit(Dtr, yt, Dva, yv, "P2")
    results["P2_node_joint"] = {
        "valid_mae": mae, "gain_vs_baseline": bmae - mae, "status": st["status"],
        "width": int(NJts.shape[1]), "scale": s, **group_delta(pred, bpred, yv),
    }

    # same-width controls: fixed random projection of H2
    for name, width in (("C1_control_h2proj_892", Zts.shape[1]), ("C2_control_h2proj_924", NJts.shape[1])):
        rng = np.random.default_rng(20261002)
        P = rng.normal(size=(H2t.shape[1], width)).astype(np.float64) / np.sqrt(H2t.shape[1])
        Pt, Pv = H2t @ P, H2v @ P
        sp = _scale(Pt)
        Dtr = np.column_stack([one_t, H2t, Pt / sp]); Dva = np.column_stack([one_v, H2v, Pv / sp])
        _, st, pred, mae = fit(Dtr, yt, Dva, yv, name)
        results[name] = {
            "valid_mae": mae, "gain_vs_baseline": bmae - mae, "status": st["status"],
            "width": int(width), "scale": sp,
        }

    results["gate"] = {"gain_min": GATE_GAIN, "calibrated": True}
    results["official_test_loaded"] = False
    (OUT / "phase_d_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()