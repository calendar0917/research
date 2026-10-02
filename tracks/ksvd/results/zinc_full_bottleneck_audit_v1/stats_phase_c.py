"""Phase C error-budget analysis from the frozen handoff arrays (read-only).

Uses the exported valid/train reader inputs R, frozen predictions p_base and
targets y.  No backbone forward, no training, official test never loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
HAND = REPO / "tracks/ksvd/results/zinc_dictionary_real_data_handoff"
OUT = REPO / "tracks/ksvd/results/zinc_full_bottleneck_audit_v1"

D = 144
P = 48
# reader-input block boundaries for the Full scale spec
UNARY = (0, 2 * D + 1)
PAIR = (2 * D + 1, 2 * D + 1 + 5 * (2 * P + 1))
GLOBAL = (PAIR[1], PAIR[1] + 32)
TOPO = (GLOBAL[1], GLOBAL[1] + 8)


def load(split: str):
    z = np.load(HAND / f"{split}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def budget(y, pred, bias):
    r = y - (pred + bias)
    a = np.abs(r)
    n = a.size
    order = np.argsort(-a)
    out = {
        "n": int(n),
        "mae": float(a.mean()),
        "median_abs": float(np.median(a)),
        "mean_signed": float(r.mean()),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max_abs": float(a.max()),
        "top1pct_share": float(a[order[: max(1, int(round(0.01 * n)))]].sum() / a.sum()),
        "top5pct_share": float(a[order[: max(1, int(round(0.05 * n)))]].sum() / a.sum()),
        "top10pct_share": float(a[order[: max(1, int(round(0.10 * n)))]].sum() / a.sum()),
        "central90_mae_pooled": float(a[order[int(round(0.10 * n)):]].mean()),
        "mae_excluding_max": float((a.sum() - a.max()) / (n - 1)),
    }
    return out, r, a


def grouped(y, r, a, edges, name):
    g = np.digitize(y, edges)
    res = {}
    k = len(edges) + 1
    for i in range(k):
        m = g == i
        if m.sum() == 0:
            continue
        res[f"{name}_{i}"] = {
            "n": int(m.sum()),
            "share_rows": float(m.mean()),
            "share_abs_sum": float(a[m].sum() / a.sum()),
            "mae": float(a[m].mean()),
            "mean_signed": float(r[m].mean()),
            "y_min": float(y[m].min()),
            "y_max": float(y[m].max()),
        }
    return res


def main():
    tr = load("train")
    va = load("valid")
    yt, pt = tr["y"], tr["p_base"]
    yv, pv = va["y"], va["p_base"]
    bias = float(np.median(yt - pt))

    rep = {
        "bias_train_median": bias,
        "raw": {
            "train_mae": float(np.abs(yt - pt).mean()),
            "valid_mae": float(np.abs(yv - pv).mean()),
        },
        "calibrated": {
            "train_mae": float(np.abs(yt - (pt + bias)).mean()),
            "valid_mae": float(np.abs(yv - (pv + bias)).mean()),
        },
        "target_distribution": {
            "train_y": {k: float(v) for k, v in zip(
                ["min", "p1", "p5", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"],
                np.quantile(yt, [0, .01, .05, .10, .25, .5, .75, .90, .95, .99, 1.0]))},
            "valid_y": {k: float(v) for k, v in zip(
                ["min", "p1", "p5", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"],
                np.quantile(yv, [0, .01, .05, .10, .25, .5, .75, .90, .95, .99, 1.0]))},
        },
        "valid_gap_calibrated": {},
        "train_gap_calibrated": {},
    }
    rep["train_budget"], rt, at = budget(yt, pt, bias)
    rep["valid_budget"], rv, av = budget(yv, pv, bias)
    rep["valid_gap_calibrated"] = {}
    rep["train_gap_calibrated"] = {}

    # target quantile grouping with TRAIN-fitted edges
    q = np.quantile(yt, [0.1, 0.25, 0.5, 0.75, 0.9])
    rep["valid_by_train_target_quantile"] = grouped(yv, rv, av, q, "q")
    rep["train_by_train_target_quantile"] = grouped(yt, rt, at, q, "q")

    # node-count quartiles from node_ptr (train-fit edges)
    nt = np.diff(tr["node_ptr"]).astype(np.float64)
    nv = np.diff(va["node_ptr"]).astype(np.float64)
    nq = np.quantile(nt, [0.25, 0.5, 0.75])
    rep["valid_by_nodecount_quartile"] = grouped(nv, rv, av, nq, "n")
    rep["train_by_nodecount_quartile"] = grouped(nt, rt, at, nq, "n")

    # per-graph graph-level features from R (unary sum block and topology/global)
    Rt = tr["R"]; Rv = va["R"]
    size_t = np.expm1(np.log1p(nt))  # nt
    # chemistry: whole-molecule atom-type histogram is removed from R by C6.
    # rarity via topology/global structure features
    topo_t = Rt[:, TOPO[0]:TOPO[1]]
    topo_v = Rv[:, TOPO[0]:TOPO[1]]
    mu, sd = topo_t.mean(0), topo_t.std(0) + 1e-9
    zt = (topo_t - mu) / sd
    zv = (topo_v - mu) / sd
    # also use R (model representation) as a second fixed metric
    muR, sdR = Rt.mean(0), Rt.std(0) + 1e-9
    zRt = (Rt - muR) / sdR
    zRv = (Rv - muR) / sdR

    def nn_dist(query, ref, chunk=200):
        out = np.empty(query.shape[0])
        for i in range(0, query.shape[0], chunk):
            qq = query[i:i + chunk]
            d2 = ((qq[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
            out[i:i + chunk] = np.sqrt(d2.min(1))
        return out

    dist_topo = nn_dist(zv, zt)
    dist_R = nn_dist(zRv, zRt)
    rep["coverage"] = {
        "topo_nn_dist_valid_mean": float(dist_topo.mean()),
        "topo_nn_dist_valid_median": float(np.median(dist_topo)),
        "R_nn_dist_valid_mean": float(dist_R.mean()),
        "corr_topo_nn_dist_absres": float(np.corrcoef(dist_topo, av)[0, 1]),
        "corr_R_nn_dist_absres": float(np.corrcoef(dist_R, av)[0, 1]),
        "corr_nodecount_absres": float(np.corrcoef(nv, av)[0, 1]),
        "corr_target_absres": float(np.corrcoef(yv, av)[0, 1]),
        "high_err_top10pct_topo_nn_mean": float(dist_topo[np.argsort(-av)[:100]].mean()),
        "rest_topo_nn_mean": float(dist_topo[np.argsort(-av)[100:]].mean()),
        "high_err_top10pct_R_nn_mean": float(dist_R[np.argsort(-av)[:100]].mean()),
        "rest_R_nn_mean": float(dist_R[np.argsort(-av)[100:]].mean()),
    }
    # extreme-tail only: y below train p5
    p5 = float(np.quantile(yt, 0.05))
    m_tail = yv < p5
    rep["coverage"]["tail_below_train_p5"] = {
        "n_valid": int(m_tail.sum()),
        "mae": float(av[m_tail].mean()),
        "abs_sum_share": float(av[m_tail].sum() / av.sum()),
        "topo_nn_mean": float(dist_topo[m_tail].mean()),
        "topo_nn_median": float(np.median(dist_topo[m_tail])),
        "R_nn_mean": float(dist_R[m_tail].mean()),
        "nodecount_median": float(np.median(nv[m_tail])),
    }
    m_rest = ~m_tail
    rep["coverage"]["rest_above_train_p5"] = {
        "n_valid": int(m_rest.sum()),
        "mae": float(av[m_rest].mean()),
        "topo_nn_mean": float(dist_topo[m_rest].mean()),
        "R_nn_mean": float(dist_R[m_rest].mean()),
    }

    # id%5 groups
    rep["id5_groups_valid"] = {}
    for g in range(5):
        m = (va["ids"] % 5) == g
        rep["id5_groups_valid"][str(g)] = {"n": int(m.sum()), "mae": float(av[m].mean()),
                                            "mean_signed": float(rv[m].mean())}

    # largest errors
    order = np.argsort(-av)[:15]
    rep["valid_top15"] = [
        {"idx": int(i), "id": int(va["ids"][i]), "y": float(yv[i]), "p_cal": float(pv[i] + bias),
         "abs_cal": float(av[i]), "nodecount": int(nv[i]),
         "nn_topo": float(dist_topo[i]), "nn_R": float(dist_R[i])}
        for i in order
    ]
    (OUT / "phase_c_stats.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()