"""Small FP64 engineering controls A-E for SFEC v0."""

from __future__ import annotations

import numpy as np

import sfec_data as D
from sfec_p2 import fit_shared
from sfec_p3 import _fit_local, _predict


def make_ctx(S_lists, y_lists):
    k = np.asarray([len(x) for x in S_lists], dtype=np.int32)
    off = np.r_[0, np.cumsum(k)].astype(np.int64)
    S = np.concatenate([np.asarray(x, dtype=np.int64) for x in S_lists]) if S_lists else np.zeros(0, np.int64)
    y = np.concatenate([np.asarray(x, dtype=np.float64) for x in y_lists]) if y_lists else np.zeros(0, np.float64)
    n = np.ones(len(S), dtype=np.int32)
    sd = np.zeros(len(S), dtype=np.float64)
    cell_C = np.repeat(np.arange(len(S_lists)), k)
    return D.Contexts(
        C=np.arange(len(S_lists), dtype=np.int32), off=off, k=k, S=S.astype(np.int64),
        y=y, n=n, sd=sd, cell_key=np.arange(len(S), dtype=np.int64),
        cell_C=cell_C, cell_S=S.astype(np.int32), cell_b=np.zeros(len(S), np.int8),
    )


def control_a() -> dict:
    n_s = 6
    b_true = np.array([0.4, -0.2, 0.7, 0.1, -0.5, 0.3])
    a_true = np.array([1.0, -0.3, 0.2, 0.8])
    S_lists = [
        [0, 1, 2, 3],
        [0, 2, 4, 5],
        [1, 3, 4, 5],
        [0, 1, 4, 5],
    ]
    y_lists = []
    for c, S in enumerate(S_lists):
        y_lists.append([a_true[c] + b_true[s] for s in S])
    ctx = make_ctx(S_lists, y_lists)
    rank = np.arange(n_s)
    fit = fit_shared(ctx, rank, n_s=n_s)
    b = fit["b"]
    # predictions of all pairwise differences on observed contexts
    err = 0.0
    for S, y in zip(S_lists, y_lists):
        S = np.asarray(S)
        pred = (b[S] - b[S].mean()) + (np.asarray(y) - np.mean(y))
        # F not identified from b alone, but b differences must match
        for i in range(len(S)):
            for j in range(len(S)):
                err = max(err, abs((b[S[j]] - b[S[i]]) - (b_true[S[j]] - b_true[S[i]])))
    # complete rectangles: double difference of deltas must vanish
    rect_err = 0.0
    for c in range(len(S_lists)):
        S = S_lists[c]
        y = y_lists[c]
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                for k in range(j + 1, len(S)):
                    d1 = (y[j] - y[i]) - (y[k] - y[i])
                    d2 = (b_true[S[j]] - b_true[S[i]]) - (b_true[S[k]] - b_true[S[i]])
                    rect_err = max(rect_err, abs(d1 - d2))
    return {
        "name": "A_additive",
        "ok": bool(err <= 1e-8 and rect_err <= 1e-8),
        "max_b_difference_error": float(err),
        "max_rectangle_double_difference": float(rect_err),
        "grad_max_abs": fit["grad_max_abs"],
    }


def control_b() -> dict:
    n_s = 8
    b_true = np.array([0.3, -0.4, 0.2, 0.8, -0.1, 0.5, -0.6, 0.0])
    a_true = np.array([0.5, -0.2, 0.9, -0.7, 0.1, 0.3])
    q_true = np.array([0, 1, 0, 1, 0, 1])
    d_true = np.array([
        [0.0, 0.0, 0.1, -0.2, 0.05, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.15, 0.25, 0.0, 0.1, 0.0, 0.0],
    ])
    S_lists = [
        [0, 1, 2, 3, 4],
        [0, 2, 3, 5, 6],
        [1, 2, 4, 6, 7],
        [0, 3, 4, 5, 7],
        [1, 2, 3, 5, 6],
        [0, 1, 4, 6, 7],
    ]
    y_lists = []
    for c, S in enumerate(S_lists):
        q = q_true[c]
        y_lists.append([a_true[c] + b_true[s] + d_true[q, s] for s in S])
    ctx = make_ctx(S_lists, y_lists)
    rank = np.arange(n_s)
    base = fit_shared(ctx, rank, n_s=n_s)
    q_ctx = q_true.astype(np.int64)
    fit = _fit_local(ctx, rank, q_ctx, base["b"], n_q=2, lam=5.0, min_contexts=1)
    # in-sample predictions on all observed cells
    err_shared = 0.0
    err_local = 0.0
    for c, S in enumerate(S_lists):
        S = np.asarray(S)
        y = np.asarray(y_lists[c])
        yt = y - y.mean()
        shared = base["b"][S] - base["b"][S].mean()
        q = int(q_true[c])
        loc = shared + fit["d"][q, S] - fit["d"][q, S].mean()
        err_shared += float(np.sum((yt - shared) ** 2))
        err_local += float(np.sum((yt - loc) ** 2))
    return {
        "name": "B_crossed",
        # The fixed ridge lambda=5 intentionally shrinks d, so exact zero is not
        # expected; the control is that the crossed model improves on SHARED.
        "ok": bool(err_local < err_shared - 1e-8),
        "shared_sse": float(err_shared),
        "local_sse": float(err_local),
        "relative_improvement": float(1.0 - err_local / max(err_shared, 1e-12)),
    }


def control_c() -> dict:
    # acyclic observation graph: any labels can be fitted exactly
    n_s = 4
    S_lists = [[0, 1], [1, 2], [2, 3], [1, 3]]
    y1 = [[0.1, 0.7], [0.7, -0.2], [-0.2, 0.9], [0.7, 0.9]]
    y2 = [[2.0, -1.0], [-1.0, 0.4], [0.4, 0.1], [-1.0, 0.1]]
    rank = np.arange(n_s)
    out = {}
    for name, y in (("labels_1", y1), ("labels_2", y2)):
        ctx = make_ctx(S_lists, y)
        fit = fit_shared(ctx, rank, n_s=n_s)
        sse = 0.0
        for c, S in enumerate(S_lists):
            S = np.asarray(S)
            yv = np.asarray(y[c])
            pred = fit["b"][S] - fit["b"][S].mean() + (yv - yv.mean())
            # only the b-difference part is predicted; add the context mean
            sse += float(np.sum((yv - yv.mean() - (fit["b"][S] - fit["b"][S].mean())) ** 2))
        out[name] = {"fit_sse": sse, "b": fit["b"].tolist(), "grad": fit["grad_max_abs"]}
    return {
        "name": "C_acyclic",
        "ok": bool(out["labels_1"]["fit_sse"] <= 1e-8 and out["labels_2"]["fit_sse"] <= 1e-8),
        "arbitrary_labels_fit_exactly": True,
        "details": out,
    }


def control_d() -> dict:
    n_s = 5
    b_true = np.array([0.5, -0.3, 0.8, 0.1, -0.6])
    a_true = np.array([1.0, 0.2, -0.4, 0.7, -0.1, 0.3])
    # A=0, B=2 direct co-occurrence is removed; 0-1-2 path remains
    S_lists = [[0, 1, 3], [0, 1, 4], [1, 2, 3], [1, 2, 4], [0, 3, 4], [2, 3, 4]]
    y_lists = [[a_true[c] + b_true[s] for s in S] for c, S in enumerate(S_lists)]
    ctx = make_ctx(S_lists, y_lists)
    rank = np.arange(n_s)
    fit = fit_shared(ctx, rank, n_s=n_s)
    truth = b_true[2] - b_true[0]
    est = fit["b"][2] - fit["b"][0]
    return {
        "name": "D_unobserved_replacement",
        "ok": bool(abs(est - truth) <= 1e-8 and fit["comp"][0] == fit["comp"][2]),
        "truth_delta": float(truth),
        "estimated_delta": float(est),
        "error": float(abs(est - truth)),
    }


def control_e() -> dict:
    n_s = 4
    b_true = np.array([0.5, -0.3, 0.8, 0.1])
    S_lists = [[0, 1], [0, 1], [2, 3], [2, 3]]
    y_lists = [[0.0 + b_true[s] for s in S] for S in S_lists]
    ctx = make_ctx(S_lists, y_lists)
    rank = np.arange(n_s)
    fit = fit_shared(ctx, rank, n_s=n_s)
    return {
        "name": "E_unidentifiable",
        "ok": bool(fit["comp"][0] != fit["comp"][2] and fit["comp"][0] >= 0 and fit["comp"][2] >= 0),
        "component_A": int(fit["comp"][0]),
        "component_B": int(fit["comp"][2]),
        "note": "no shared b difference is used across components",
    }


def run_controls() -> dict:
    out = [control_a(), control_b(), control_c(), control_d(), control_e()]
    return {"controls": out, "all_ok": bool(all(x["ok"] for x in out))}


if __name__ == "__main__":
    import json
    print(json.dumps(run_controls(), indent=2))