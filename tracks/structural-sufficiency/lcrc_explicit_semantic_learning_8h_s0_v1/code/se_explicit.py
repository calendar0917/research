"""Low-cost performance coordinates on Lipo and ESOL (CPU only, no GPU).

Builds a bag-level explicit-statistics matrix X (scaled global42 + nbr_code80
counts + path_cat400 counts + end_code80 counts + closure2 counts +
(path_cat,end_code) 32000 counts = 32604 dims, sparse) and fits

* Ridge(alpha=10, solver='lsqr') on Train-column-scaled sparse inputs, raw y;
* XGBRegressor with the frozen fixed hyper-parameters (no search/early stop).

Both are fitted on the official Train only and scored on full Train/Valid/Test.
The sparse zero semantics are the XGB default (missing -> a learned default
direction), which is reported explicitly rather than treated as a data gap.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp

import se_common as C

import lp_common as LPC
import lp_data as LPD

XGB_PARAMS = dict(
    n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=10.0,
    objective="reg:squarederror", tree_method="hist", max_bin=128,
    random_state=0, n_jobs=8,
)
RIDGE_ALPHA = 10.0


def build_matrix(view: LPD.View, split: str) -> sp.csr_matrix:
    cache = view.caches[split]
    n = int(cache["n_molecules"])
    compact_ptr = np.asarray(cache["compact_ptr"]) if "compact_ptr" in cache else None
    glob = np.asarray(cache["global"])            # (n, 42)
    path_ptr = np.asarray(cache["path_ptr"])
    path_cat = np.asarray(cache["path_cat"])
    end63 = np.asarray(cache["path_end63"])
    closure = np.asarray(cache["path_closure"]) if "path_closure" in cache else None
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    nbr_code = np.asarray(cache["nbr_code"])

    GLOB, PATH, END, NBR, CLOS, JOINT = 42, 400, 80, 80, 2, 32000
    rows, cols, vals = [], [], []
    base = 0
    off = {"g": 0, "path": GLOB, "end": GLOB + PATH, "nbr": GLOB + PATH + END,
           "clos": GLOB + PATH + END + NBR, "joint": GLOB + PATH + END + NBR + CLOS}
    for m in range(n):
        # global (standardised with Train structure scaler is already done)
        for j in range(GLOB):
            if glob[m, j] != 0:
                rows.append(m); cols.append(off["g"] + j); vals.append(float(glob[m, j]))
        pc = path_cat[path_ptr[m]:path_ptr[m + 1]]
        ec = end63[path_ptr[m]:path_ptr[m + 1]]
        cc = closure[path_ptr[m]:path_ptr[m + 1]] if closure is not None else np.zeros_like(pc)
        # counts
        from collections import Counter
        cpc = Counter(int(x) for x in pc)
        cec = Counter(int(x) for x in ec)
        ccc = Counter(int(x) for x in cc)
        cjoint = Counter((int(a), int(b)) for a, b in zip(pc, ec))
        for k, v in cpc.items():
            rows.append(m); cols.append(off["path"] + k); vals.append(float(v))
        for k, v in cec.items():
            rows.append(m); cols.append(off["end"] + k); vals.append(float(v))
        for k, v in ccc.items():
            rows.append(m); cols.append(off["clos"] + k); vals.append(float(v))
        for (a, b), v in cjoint.items():
            rows.append(m); cols.append(off["joint"] + a * END + b); vals.append(float(v))
        nc = nbr_code[nbr_ptr[m]:nbr_ptr[m + 1]]
        for k, v in Counter(int(x) for x in nc).items():
            rows.append(m); cols.append(off["nbr"] + k); vals.append(float(v))
    total = GLOB + PATH + END + NBR + CLOS + JOINT
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n, total), dtype=np.float64)
    return X


def _fit_scaler(X_train: sp.csr_matrix) -> np.ndarray:
    """Per-column scale from Train; sparse inputs are not centred."""
    mean = np.asarray(X_train.mean(axis=0)).reshape(-1)
    mx = np.asarray(X_train.max(axis=0).todense()).reshape(-1)
    scale = np.where(mx > 0, mx, 1.0)
    return scale


def _metrics(pred, y):
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    return {"rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "mae": float(np.mean(np.abs(pred - y))),
            "n": int(y.size)}


def run(view: LPD.View, *, label: str, xgb: bool = True) -> dict[str, Any]:
    X = {s: build_matrix(view, s) for s in LPC.SPLITS}
    scale = _fit_scaler(X["train"])
    Xs = {s: X[s] @ sp.diags(1.0 / scale) for s in LPC.SPLITS}
    out: dict[str, Any] = {"label": label, "n_features": int(X["train"].shape[1])}

    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
    ridge.fit(Xs["train"], view.y["train"])
    out["ridge"] = {s: _metrics(ridge.predict(Xs[s]), view.y[s]) for s in LPC.SPLITS}
    out["ridge_window_ridge"] = "fit on Train only; sparse, not centred; raw y"

    if xgb:
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X["train"], view.y["train"])
            out["xgb"] = {s: _metrics(model.predict(X[s]), view.y[s]) for s in LPC.SPLITS}
            out["xgb_zero_semantics"] = ("sparse zeros are the XGB 'missing' value and take a learned "
                                         "default direction, not a data gap")
            out["xgb_params"] = {k: v for k, v in XGB_PARAMS.items()}
        except Exception as exc:  # noqa: BLE001
            out["xgb"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def run_all() -> dict[str, Any]:
    out: dict[str, Any] = {}
    lipo = LPD.build_view()
    out["lipo"] = run(lipo, label="ogbg-mollipo")
    try:
        import se_data as SED

        esol = SED.build_esol_view()
        out["esol"] = run(esol, label="ogbg-molesol")
    except Exception as exc:  # noqa: BLE001
        out["esol"] = {"error": f"{type(exc).__name__}: {exc}"}
    C.write_json(C.ANALYSIS / "explicit_baseline.json", out)
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(run_all(), indent=1)[:2000])
