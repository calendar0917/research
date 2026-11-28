"""Phase C runner (ISRD-v1): prediction sufficiency of the explicit structure input.

    COUNT128   log1p(n_atoms), log1p(n_bonds) + the R128 structures' continuous
               scaled counts, same LR recipe / parameter count as R128
    COUNT-ALL  the same two size terms + all 754 C and 3 652 F continuous counts,
               same LR objective (Train-only standardisation) via the sparse
               equivalent parametrisation
    XGB-ALL    identical input, frozen historical XGBoost recipe (500x5, lr .03,
               subsample .8, colsample .8, lambda 5, hist, seed 0)

Also writes the COUNT-ALL local contribution analysis and the saved booster.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp

import isrd1_common as K
import isrd1_rules as R

C_GRID = (0.03, 0.3)
XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 1,
    "tree_method": "hist",
    "seed": K.SEED,
    "nthread": 8,
}
N_ROUNDS = 500


def standardise(train_vals: np.ndarray, other_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mu = train_vals.mean(axis=0)
    sd = train_vals.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (train_vals - mu) / sd, (other_vals - mu) / sd, {"mean": mu, "std": sd}


def sparse_standardisation_stats(X: sp.csr_matrix, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = X[rows]
    mu = np.asarray(sub.mean(axis=0)).ravel()
    mu2 = np.asarray(sub.multiply(sub).mean(axis=0)).ravel()
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def fit_linear_arm(name: str, X: sp.csr_matrix, y: np.ndarray, sw: np.ndarray,
                   data: K.Data, sparse: bool) -> dict[str, Any]:
    train = data.rows["train"]
    mu, sd = sparse_standardisation_stats(X, train)
    fits = []
    for C in C_GRID:
        if sparse:
            fit = K.fit_logistic_sparse_standardised(X, y, sw, C, mu, sd, verbose=True)
        else:
            Xd = X.toarray()
            Z = (Xd - mu) / sd
            fit = dict(K.fit_logistic(Z[train], y[train], sw[train], C))
            fit["logit"] = fit["intercept"] + Z @ fit["coef"]
            del Xd, Z
        m = K.all_metrics(fit["logit"], data.y, data.rows)
        fits.append({"C": C, "fit": fit, "metrics": m})
        print(f"[C] {name} C={C} train={m['train']['roc_auc']:.4f} "
              f"valid={m['valid']['roc_auc']:.4f} test={m['test']['roc_auc']:.4f} "
              f"grad={fit['grad_inf']:.2e} it={fit['n_iter']}", flush=True)
    best = max(fits, key=lambda f: (f["metrics"]["valid"]["roc_auc"], -f["C"]))
    mu_, sd_ = mu, sd
    b_raw = float(best["fit"]["intercept"] - np.sum(best["fit"]["coef"] * mu_ / sd_))
    w_raw = best["fit"]["coef"] / sd_
    z_raw = b_raw + np.asarray(X @ w_raw).ravel()
    closure = float(np.max(np.abs(z_raw - best["fit"]["logit"])))
    return {"arm": name, "selected_C": best["C"], "logit": best["fit"]["logit"],
            "metrics": best["metrics"], "intercept_standardised": float(best["fit"]["intercept"]),
            "coef_standardised": best["fit"]["coef"], "intercept_raw": b_raw,
            "coef_raw": w_raw, "mean": mu_, "sd": sd_,
            "closure_raw_units": closure,
            "C_grid_valid_auc": {str(f["C"]): f["metrics"]["valid"]["roc_auc"] for f in fits},
            "n_parameters": int(best["fit"]["coef"].size + 1),
            "train_valid_gap_auc": best["metrics"]["train"]["roc_auc"] - best["metrics"]["valid"]["roc_auc"],
            "converged": bool(best["fit"]["converged"]), "grad_inf": float(best["fit"]["grad_inf"])}


def dense_design(data: K.Data, rows: np.ndarray) -> np.ndarray:
    """float32 ``[log1p(n_atoms), log1p(n_bonds) | C | F]`` for ``rows`` (zeros stay zero)."""
    n = rows.size
    out = np.zeros((n, 2 + K.N_STRUCT), dtype=np.float32)
    size = data.size_matrix()[rows]
    out[:, 0] = size[:, 0]
    out[:, 1] = size[:, 1]
    out[:, 2:2 + K.N_C] = data.C[rows].toarray()
    out[:, 2 + K.N_C:] = data.F[rows].toarray()
    return out


def run_xgb_all(data: K.Data) -> dict[str, Any]:
    import xgboost as xgb

    train = data.rows["train"]
    ytr = data.y[train].astype(np.float32)
    t0 = time.perf_counter()
    Xtr = dense_design(data, train)
    dtrain = xgb.DMatrix(Xtr, label=ytr, nthread=8)
    del Xtr
    booster = xgb.train(dict(XGB_PARAMS), dtrain, num_boost_round=N_ROUNDS)
    del dtrain
    fit_seconds = time.perf_counter() - t0
    print(f"[C] XGB-ALL fitted in {fit_seconds:.1f}s", flush=True)
    logit = np.zeros(data.y.size)
    for split in K.SPLITS:
        Xs = dense_design(data, data.rows[split])
        logit[data.rows[split]] = booster.predict(xgb.DMatrix(Xs, nthread=8), output_margin=True)
        del Xs
    K.MODELS.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(K.MODELS / "XGB-ALL.json"))
    m = K.all_metrics(logit, data.y, data.rows)
    imp = booster.get_score(importance_type="total_gain")
    gains = {int(k[1:]): float(v) for k, v in imp.items()}
    total = sum(gains.values())
    struct_cols = {k: v for k, v in gains.items() if k >= 2}
    out = {"arm": "XGB-ALL", "logit": logit, "metrics": m, "fit_seconds": fit_seconds,
           "params": XGB_PARAMS, "n_rounds": N_ROUNDS, "design_width": 2 + K.N_STRUCT,
           "booster_path": K.rel(K.MODELS / "XGB-ALL.json"),
           "booster_sha256": K.sha256_file(K.MODELS / "XGB-ALL.json"),
           "n_features_used": len(gains), "total_gain": total,
           "structural_gain_share": float(sum(struct_cols.values()) / total) if total else None,
           "n_structural_features_used": len(struct_cols),
           "top_structural": sorted(struct_cols.items(), key=lambda kv: -kv[1])[:20],
           "boundary_note": "feature 0/1 are log1p(n_atoms)/log1p(n_bonds); "
                            "2+i maps to C[i], 2+754+i maps to F[i]"}
    print(f"[C] XGB-ALL train={m['train']['roc_auc']:.4f} valid={m['valid']['roc_auc']:.4f} "
          f"test={m['test']['roc_auc']:.4f}", flush=True)
    return out


def local_contributions(arm: dict, X: sp.csr_matrix, data: K.Data,
                        chunk: int = 2000) -> dict[str, Any]:
    """Per-molecule COUNT-ALL structure contributions and top-k concentration."""
    w_raw = np.asarray(arm["coef_raw"], dtype=np.float64)
    w_struct = w_raw[2:]
    b_raw = float(arm["intercept_raw"])
    struct = X[:, 2:]                                # [C|F] raw scaled counts
    mu, sd = arm["mean"][2:], arm["sd"][2:]
    y = data.y
    n = y.size
    n_active = np.zeros(n, dtype=np.int32)
    abs_total = np.zeros(n)
    topk = {k: np.zeros(n) for k in (5, 10, 20)}
    remaining_abs = np.zeros(n)
    recon = np.zeros(n)
    top_ids = np.zeros((n, 20), dtype=np.int32)
    struct_abs = np.zeros(K.N_STRUCT)
    struct_sum = np.zeros(K.N_STRUCT)
    struct_active = np.zeros(K.N_STRUCT, dtype=np.int64)
    rows_all = np.arange(n)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        sl = slice(lo, hi)
        x = struct[lo:hi].toarray().astype(np.float64)
        contrib = (x - mu) / sd * w_struct
        abs_c = np.abs(contrib)
        abs_total[sl] = abs_c.sum(axis=1)
        n_active[sl] = (x > 0).sum(axis=1)
        idx = np.argsort(-abs_c, axis=1)
        for k in topk:
            topk[k][sl] = np.take_along_axis(abs_c, idx[:, :k], axis=1).sum(axis=1)
            if k == 20:
                top_ids[sl] = idx[:, :20].astype(np.int32)
        remaining_abs[sl] = abs_total[sl] - topk[20][sl]
        size_term = arm["coef_raw"][0] * np.log1p(data.n_atoms[sl]) + \
            arm["coef_raw"][1] * np.log1p(data.n_bonds[sl])
        z = b_raw + size_term + x @ w_struct
        recon[sl] = z - arm["logit"][lo:hi]
        struct_abs += abs_c.sum(axis=0)
        struct_sum += contrib.sum(axis=0)
        struct_active += (x > 0).sum(axis=0)
        del x, contrib, abs_c, idx

    shares = {f"top{k}_share": (topk[k] / np.maximum(abs_total, 1e-12)) for k in topk}
    reg = K.registry()
    names = [f"C[{i}]" for i in range(K.N_C)] + [f"F[{i}]" for i in range(K.N_F)]
    order = np.argsort(-struct_abs)[:40]
    top_struct = []
    for i in order:
        gran = "C" if i < K.N_C else "F"
        col = int(i if gran == "C" else i - K.N_C)
        meta = reg[gran].iloc[col]
        top_struct.append({"structure": names[i], "granularity": gran, "column": col,
                           "canonical_key": str(meta["canonical_key"]),
                           "family": str(meta["family"]),
                           "coef_raw": float(w_struct[i]),
                           "support_all": int(struct_active[i]),
                           "abs_contribution_total": float(struct_abs[i]),
                           "signed_contribution_total": float(struct_sum[i])})
    # FULL/COARSE overlap: how often an active FULL type's COARSE parent is also active
    parent = full_parent_columns()
    Cpres = data.C.copy()
    Cpres.data[:] = 1.0
    coo = data.F.tocoo()
    pcol = parent[coo.col]
    has_parent = pcol >= 0
    vals = np.asarray(Cpres[coo.row[has_parent], pcol[has_parent]]).ravel()
    parent_present = (vals > 0)
    overlap = float(parent_present.sum() / max(1, int(has_parent.sum())))
    multiplicity = {split: {"mean_active_structures": float(n_active[data.rows[split]].mean()),
                            "median_active_structures": float(np.median(n_active[data.rows[split]])),
                            "p95_active_structures": float(np.percentile(n_active[data.rows[split]], 95))}
                    for split in K.SPLITS}
    return {"per_molecule": {"n_active_structures": n_active, "abs_total": abs_total,
                             **{f"{k}_abs_sum": topk[k] for k in topk},
                             **{k: v for k, v in shares.items()},
                             "remaining_abs": remaining_abs,
                             "reconstruction_error": recon, "top20_ids": top_ids},
            "summary": {
                "reconstruction_max_abs_error": float(np.max(np.abs(recon))),
                "share_top5_mean": {s: float(shares["top5_share"][data.rows[s]].mean()) for s in K.SPLITS},
                "share_top10_mean": {s: float(shares["top10_share"][data.rows[s]].mean()) for s in K.SPLITS},
                "share_top20_mean": {s: float(shares["top20_share"][data.rows[s]].mean()) for s in K.SPLITS},
                "remaining_abs_mean": {s: float(remaining_abs[data.rows[s]].mean()) for s in K.SPLITS},
                "abs_total_mean": {s: float(abs_total[data.rows[s]].mean()) for s in K.SPLITS},
                **{f"active_{k}": v for k, v in multiplicity.items()},
                "frac_active_full_with_active_coarse_parent": overlap,
                "n_active_structures_total": int(n_active.sum()),
                "n_nonzero_structure_terms": int((struct_abs > 0).sum()),
            },
            "top_structures": top_struct,
            "names": names}


_PARENT: np.ndarray | None = None


def full_parent_columns() -> np.ndarray:
    """COARSE column of each FULL column's parent (``-1`` if outside the vocabulary)."""
    global _PARENT
    if _PARENT is not None:
        return _PARENT
    reg = K.registry()
    with np.load(K.INSTANCES, allow_pickle=True) as z:
        full_parent = np.asarray(z["full_parent"], dtype=np.int64)
    c_dtid = reg["C"]["discovered_type_id"].to_numpy().astype(np.int64)
    c_col_of = {int(d): i for i, d in enumerate(c_dtid)}
    f_dtid = reg["F"]["discovered_type_id"].to_numpy().astype(np.int64)
    parent = np.array([c_col_of.get(int(full_parent[d]), -1) for d in f_dtid], dtype=np.int64)
    _PARENT = parent
    return parent


def main() -> None:
    t0 = time.perf_counter()
    data = K.Data()
    bc = K.read_json(K.ANALYSIS / "phaseBC.json")
    pa = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    singles = {c["rule_id"]: c for c in pa["singles"]}
    train = data.rows["train"]

    # ---------------- COUNT128: identical structures to R128 --------------
    r128_ids = bc["phaseB"]["selected_rule_ids"][:128]
    r128_rules = [singles[i] for i in r128_ids]
    structs = [(c["granularity"], int(c["column"])) for c in r128_rules]
    assert len(set(structs)) == len(structs), "R128 structure identities are not unique"
    counts = np.column_stack([data.col(g, c).astype(np.float64) for g, c in structs])
    size = data.size_matrix()
    X128 = np.column_stack([size, counts])
    sw = np.zeros(data.y.size)
    sw[train] = K.class_balanced_weights(data.y[train].astype(np.float64))
    count128 = fit_linear_arm("COUNT128", sp.csr_matrix(X128), data.y, sw, data, sparse=False)
    del counts, X128
    print(f"[C] COUNT128 valid={count128['metrics']['valid']['roc_auc']:.4f} "
          f"test={count128['metrics']['test']['roc_auc']:.4f}", flush=True)

    # ---------------- COUNT-ALL: all C + F continuous counts --------------
    stat = data.all_structure_matrix()
    Xall = sp.hstack([sp.csr_matrix(size), stat], format="csr")
    print(f"[C] COUNT-ALL design {Xall.shape} nnz={Xall.nnz}", flush=True)
    count_all = fit_linear_arm("COUNT-ALL", Xall, data.y, sw, data, sparse=True)
    print(f"[C] COUNT-ALL valid={count_all['metrics']['valid']['roc_auc']:.4f} "
          f"test={count_all['metrics']['test']['roc_auc']:.4f} closure="
          f"{count_all['closure_raw_units']:.2e}", flush=True)

    # ---------------- XGB-ALL ---------------------------------------------
    xgb_all = run_xgb_all(data)

    # ---------------- predictions + contributions -------------------------
    for arm in (count128, count_all, xgb_all):
        safe = arm["arm"]
        np.savez_compressed(K.PREDICTIONS / f"{safe}.npz",
                            logit=np.asarray(arm["logit"], dtype=np.float64), y=data.y,
                            split_code=data.split_code(), graph_id=np.arange(data.y.size),
                            scaffold=data.scaffold)
    contrib = local_contributions(count_all, Xall, data)
    per = contrib["per_molecule"]
    np.savez_compressed(K.ANALYSIS / "count_all_contributions.npz",
                        **{k: v for k, v in per.items()},
                        coef_raw=count_all["coef_raw"], intercept_raw=count_all["intercept_raw"],
                        structure_names=np.array(contrib["names"]))

    payload = {
        "round_id": K.ROUND_ID, "phase": "C: prediction sufficiency",
        "seconds": float(time.perf_counter() - t0),
        "count128": {k: v for k, v in count128.items() if k not in ("logit",)},
        "count128_structures": [list(s) for s in structs],
        "count128_rule_ids": r128_ids,
        "count_all": {k: v for k, v in count_all.items() if k not in ("logit",)},
        "xgb_all": {k: v for k, v in xgb_all.items() if k not in ("logit",)},
        "count_all_local": contrib["summary"],
        "count_all_top_structures": contrib["top_structures"],
        "input_identity": {
            "count128_structures": len(structs),
            "count_all_width": int(Xall.shape[1]),
            "count_all_structure_columns": int(K.N_STRUCT),
            "xgb_all_width": int(xgb_all["design_width"]),
        },
        "git": K.git_state(),
    }
    K.write_json(K.ANALYSIS / "phaseC.json", payload)
    print(f"[C] done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
