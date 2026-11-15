"""Corrected CPU explicit-statistics coordinate (Lipo + ESOL).

Same feature definition, column order, scaler and hyper-parameters as the
predecessor ``se_explicit.py``; the **only** change is the per-molecule
occurrence slice (``atom_ptr -> path_ptr/nbr_ptr`` instead of ``path_ptr[m]``).

Matrix (32 604 columns, sparse), exactly the old order::

    global42 | path400 | end80 | nbr80 | closure2 | joint32000

* Ridge: ``alpha=10, solver='lsqr'`` on Train-column-scaled sparse inputs (the
  actual scaler is the Train column positive maximum, ``max>0 ? max : 1``, no
  centring), raw ``y``.  This is the realised algorithm, described accurately.
* XGBoost: frozen fixed hyper-parameters, no search, no early stopping.  Sparse
  zeros are the XGBoost *missing* value and take a learned default direction;
  that is reported as the semantics, not as a data gap.

Every row's path/nbr/end/closure/joint counts are cross-checked against the
independent single-molecule ``build_molecule`` oracle before fitting.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp

import rp_common as RP
import rp_reader as RR

GLOB, PATH, END, NBR, CLOS, JOINT = 42, 400, 80, 80, 2, 32000
TOTAL = GLOB + PATH + END + NBR + CLOS + JOINT
OFF = {
    "g": 0,
    "path": GLOB,
    "end": GLOB + PATH,
    "nbr": GLOB + PATH + END,
    "clos": GLOB + PATH + END + NBR,
    "joint": GLOB + PATH + END + NBR + CLOS,
}
RIDGE_ALPHA = 10.0
XGB_PARAMS = dict(
    n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=10.0,
    objective="reg:squarederror", tree_method="hist", max_bin=128,
    random_state=0, n_jobs=8,
)


def build_matrix(view, split: str) -> sp.csr_matrix:
    reader = RR.MoleculeReader(view.caches[split], name=split)
    glob = np.asarray(view.caches[split]["global"])
    n = reader.n_molecules
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for m in range(n):
        for j in range(GLOB):
            v = float(glob[m, j])
            if v != 0.0:
                rows.append(m)
                cols.append(OFF["g"] + j)
                vals.append(v)
        r = reader.ranges(m)
        pc = reader.path_cat[r["path0"]:r["path1"]].tolist()
        ec = reader.path_end63[r["path0"]:r["path1"]].tolist()
        cc = reader.path_closure[r["path0"]:r["path1"]].tolist()
        nc = reader.nbr_code[r["s1_0"]:r["s1_1"]].tolist()
        for k, c in Counter(pc).items():
            rows.append(m); cols.append(OFF["path"] + k); vals.append(float(c))
        for k, c in Counter(ec).items():
            rows.append(m); cols.append(OFF["end"] + k); vals.append(float(c))
        for k, c in Counter(cc).items():
            rows.append(m); cols.append(OFF["clos"] + k); vals.append(float(c))
        for (a, b), c in Counter(zip(pc, ec)).items():
            rows.append(m); cols.append(OFF["joint"] + a * END + b); vals.append(float(c))
        for k, c in Counter(nc).items():
            rows.append(m); cols.append(OFF["nbr"] + k); vals.append(float(c))
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n, TOTAL), dtype=np.float64)
    return X


def count_oracle(view, split: str) -> dict[str, Any]:
    """Per-row path/nbr/end/closure/joint occurrence counts from the reader vs the matrix."""
    reader = RR.MoleculeReader(view.caches[split], name=split)
    X = build_matrix(view, split)
    bad = 0
    for m in range(reader.n_molecules):
        row = X.getrow(m).toarray().reshape(-1)
        n_path = int(row[OFF["path"]:OFF["path"] + PATH].sum())
        n_end = int(row[OFF["end"]:OFF["end"] + END].sum())
        n_nbr = int(row[OFF["nbr"]:OFF["nbr"] + NBR].sum())
        n_clos = int(row[OFF["clos"]:OFF["clos"] + CLOS].sum())
        n_joint = int(row[OFF["joint"]:OFF["joint"] + JOINT].sum())
        r = reader.ranges(m)
        exp = {
            "path": r["path1"] - r["path0"],
            "end": r["path1"] - r["path0"],
            "nbr": r["s1_1"] - r["s1_0"],
            "clos": r["path1"] - r["path0"],
            "joint": r["path1"] - r["path0"],
        }
        got = {"path": n_path, "end": n_end, "nbr": n_nbr, "clos": n_clos, "joint": n_joint}
        if got != exp:
            bad += 1
            if bad <= 3:
                RP.log(f"count mismatch {split} mol {m}: got {got} expected {exp}")
    return {"split": split, "n_rows_checked": reader.n_molecules, "n_mismatch": bad, "ok": bad == 0}


def _fit_scaler(X_train: sp.csr_matrix) -> np.ndarray:
    mx = np.asarray(X_train.max(axis=0).todense()).reshape(-1)
    return np.where(mx > 0, mx, 1.0)


def _metrics(pred, y) -> dict[str, float]:
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    return {
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "n": int(y.size),
    }


def run_dataset(view, *, label: str, xgb: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    X = {s: build_matrix(view, s) for s in RP.SPLITS}
    oracle = {s: count_oracle(view, s) for s in RP.SPLITS}
    scale = _fit_scaler(X["train"])
    Xs = {s: X[s] @ sp.diags(1.0 / scale) for s in RP.SPLITS}
    out: dict[str, Any] = {
        "label": label,
        "n_features": int(X["train"].shape[1]),
        "row_count_oracle": oracle,
        "build_seconds": float(time.perf_counter() - t0),
        "sparsity": {s: float(1.0 - X[s].nnz / (X[s].shape[0] * X[s].shape[1])) for s in RP.SPLITS},
        "scaler": "Train column positive maximum (max>0 -> max else 1.0); not centred",
        "global_features": "raw lipo_v1 global42 values (NOT pre-standardised)",
    }
    from sklearn.linear_model import Ridge

    t1 = time.perf_counter()
    ridge = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
    ridge.fit(Xs["train"], view.y["train"])
    out["ridge"] = {s: _metrics(ridge.predict(Xs[s]), view.y[s]) for s in RP.SPLITS}
    out["ridge_seconds"] = float(time.perf_counter() - t1)
    out["ridge_predictions"] = {s: ridge.predict(Xs[s]) for s in RP.SPLITS}
    if xgb:
        try:
            from xgboost import XGBRegressor

            t2 = time.perf_counter()
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X["train"], view.y["train"])
            out["xgb"] = {s: _metrics(model.predict(X[s]), view.y[s]) for s in RP.SPLITS}
            out["xgb_seconds"] = float(time.perf_counter() - t2)
            out["xgb_zero_semantics"] = (
                "sparse zeros are the XGBoost 'missing' value and take a learned default "
                "direction, not a data gap"
            )
            out["xgb_params"] = dict(XGB_PARAMS)
            out["xgb_predictions"] = {s: model.predict(X[s]) for s in RP.SPLITS}
        except Exception as exc:  # noqa: BLE001
            out["xgb"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def run_all() -> dict[str, Any]:
    import lp_data as LPD
    import se_data as SED

    lipo = LPD.build_view()
    out: dict[str, Any] = {"lipo": run_dataset(lipo, label="ogbg-mollipo")}
    try:
        esol = SED.build_esol_view()
        out["esol"] = run_dataset(esol, label="ogbg-molesol")
    except Exception as exc:  # noqa: BLE001
        out["esol"] = {"error": f"{type(exc).__name__}: {exc}"}
    out["schema"] = "lcrc-semantic-analysis-repair-explicit-baseline-v1"
    out["protocol_id"] = RP.PROTOCOL_ID
    out["created_utc"] = RP.utcnow()
    out["old_status"] = "INVALID_FOR_COMPARISON (predecessor matrix read path_ptr/nbr_ptr with a molecule index)"
    # save predictions separately from the json (arrays are not json)
    pred_payload: dict[str, Any] = {}
    for ds in ("lipo", "esol"):
        block = out.get(ds, {})
        for model in ("ridge", "xgb"):
            preds = block.get(model + "_predictions", {})
            for s, arr in preds.items():
                pred_payload[f"{ds}_{model}_{s}"] = np.asarray(arr, dtype=np.float64)
            block.pop(model + "_predictions", None)
    np.savez_compressed(RP.DATA / "explicit_predictions_corrected.npz", **pred_payload)
    RP.write_json(RP.ANALYSIS / "explicit_baseline_corrected.json", out)
    return out


__all__ = [name for name in dir() if not name.startswith("_")]
