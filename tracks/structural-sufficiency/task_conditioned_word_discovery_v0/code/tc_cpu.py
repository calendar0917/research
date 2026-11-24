"""CPU discovery, frequency control, ridge arms and the known-signal control.

The supervised part is deliberately small and fully specified: fixed scaffold
three-fold inside the current Fit, ridge alpha 10, training-part standardisation,
multi-output residualisation of every candidate on X0, and a min-|corr| score
over folds whose conditional correlation has the same strict sign.

Nothing here reads a Dev/Valid/Test label for vocabulary construction.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg
from scipy import sparse

import tc_common as C
import tc_vocab as V


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------
def scaffold_folds(split: str, rows_fit: Sequence[int], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    scaf = C.scaffold_map(split)
    groups: dict[str, list[int]] = defaultdict(list)
    for row in np.asarray(rows_fit, np.int64).tolist():
        groups[scaf[str(int(row))]].append(int(row))
    keys = sorted(groups)
    order = np.random.default_rng(int(seed)).permutation(len(keys))
    sizes = np.zeros(3, np.int64)
    fold_of_key: dict[str, int] = {}
    for i in order.tolist():
        key = keys[int(i)]
        fold = int(np.argmin(sizes))
        fold_of_key[key] = fold
        sizes[fold] += len(groups[key])
    fold_of_row = np.asarray([fold_of_key[scaf[str(int(r))]] for r in rows_fit], np.int64)
    info = {
        "seed": int(seed), "n_folds": 3,
        "fold_molecule_counts": sizes.tolist(),
        "fold_scaffold_counts": [int(sum(1 for k in keys if fold_of_key[k] == f))
                                 for f in range(3)],
        "n_scaffolds": int(len(keys)),
        "fold_of_row_sha256": C.array_sha256(fold_of_row),
        "scaffold_groups_sha256": hashlib.sha256(
            "\n".join(f"{k}\t{fold_of_key[k]}" for k in keys).encode()).hexdigest(),
    }
    return fold_of_row, info


# ---------------------------------------------------------------------------
# ridge
# ---------------------------------------------------------------------------
def _standardize_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, dtype=np.float64)
    sd = x.std(axis=0, dtype=np.float64)
    const = sd <= 1e-12
    sd = np.where(const, 1.0, sd)
    return (x - mu) / sd, mu, sd


def _ridge_beta(z: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Beta for centred standardised z and centred y (no intercept penalty)."""
    zc = z - z.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    n, d = zc.shape
    if d <= n:
        a = zc.T @ zc + float(alpha) * np.eye(d)
        return linalg.solve(a, zc.T @ yc, assume_a="pos")
    k = zc @ zc.T + float(alpha) * np.eye(n)
    coef = linalg.solve(k, yc, assume_a="pos")
    return zc.T @ coef


def ridge_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"rmse": float(C.rmse(y, pred)), "mae": float(C.mae(y, pred))}


# ---------------------------------------------------------------------------
# selection core
# ---------------------------------------------------------------------------
def _corr(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation of every column of ``a`` with vector ``b``."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean()
    va = (ac * ac).sum(axis=0)
    vb = float((bc * bc).sum())
    denom = np.sqrt(np.maximum(va * vb, 0.0))
    out = np.full(a.shape[1], np.nan)
    ok = denom > 1e-24
    out[ok] = (ac[:, ok] * bc[:, None]).sum(axis=0) / denom[ok]
    return out, va


def select_candidates(x0_fit: np.ndarray, x_fit: sparse.csc_matrix, y_fit: np.ndarray,
                      fold_of_row: np.ndarray, meta: Sequence[Mapping[str, Any]],
                      alpha: float, chunk: int = 1024) -> dict[str, Any]:
    """Run the frozen three-fold conditional-correlation selection."""
    n, p = x_fit.shape
    folds = sorted(set(int(f) for f in np.unique(fold_of_row)))
    if folds != [0, 1, 2]:
        raise ValueError(f"expected folds 0,1,2, got {folds}")
    cond = np.full((3, p), np.nan)
    raw = np.full((3, p), np.nan)
    xres_var = np.zeros((3, p))
    rvar = np.zeros(3)
    y_beta: list[np.ndarray] = []
    y_mu: list[float] = []
    z0_mu: list[np.ndarray] = []
    for j in folds:
        tr = fold_of_row != j
        ztr, mu, sd = _standardize_train(x0_fit[tr].astype(np.float64))
        beta = _ridge_beta(ztr, y_fit[tr].reshape(-1, 1), alpha).ravel()
        y_beta.append(beta)
        y_mu.append(float(y_fit[tr].mean()))
        z0_mu.append(ztr.mean(axis=0))
    const_folds = np.zeros((3, p), bool)
    for j in folds:
        tr = fold_of_row != j
        te = ~tr
        ztr, mu, sd = _standardize_train(x0_fit[tr].astype(np.float64))
        zte = (x0_fit[te].astype(np.float64) - mu) / sd
        ztr_c = ztr - ztr.mean(axis=0, keepdims=True)
        zte_c = zte - ztr.mean(axis=0, keepdims=True)
        pred_y = zte @ y_beta[j] + (y_mu[j] - z0_mu[j] @ y_beta[j])
        r = np.asarray(y_fit[te], np.float64) - pred_y
        rvar[j] = float(r.var())
        d = ztr_c.shape[1]
        if d <= ztr_c.shape[0]:
            factor = linalg.cho_factor(ztr_c.T @ ztr_c + alpha * np.eye(d))
            solve = lambda rhs: linalg.cho_solve(factor, rhs)  # noqa: E731
        else:
            factor = linalg.cho_factor(ztr_c @ ztr_c.T + alpha * np.eye(ztr_c.shape[0]))
            solve = lambda rhs: ztr_c.T @ linalg.cho_solve(factor, rhs)  # noqa: E731
        for lo in range(0, p, chunk):
            hi = min(p, lo + chunk)
            xtr = x_fit[tr, lo:hi].toarray().astype(np.float64)
            xte = x_fit[te, lo:hi].toarray().astype(np.float64)
            ymu = xtr.mean(axis=0)
            ysd = xtr.std(axis=0)
            const = ysd <= 1e-12
            const_folds[j, lo:hi] = const
            ysd = np.where(const, 1.0, ysd)
            ytr = (xtr - ymu) / ysd
            yte = (xte - ymu) / ysd
            beta_w = solve(ztr_c.T @ (ytr - ytr.mean(axis=0, keepdims=True)))
            pred = zte_c @ beta_w + ytr.mean(axis=0)
            xres = yte - pred
            xres_var[j, lo:hi] = xres.var(axis=0)
            cond[j, lo:hi], _ = _corr(xres, r)
            raw[j, lo:hi], _ = _corr(xte, np.asarray(y_fit[te], np.float64))
    # candidates whose conditional correlation is defined and has one strict sign
    defined = np.isfinite(cond).all(axis=0)
    pos = np.all(cond > 0, axis=0) & defined
    neg = np.all(cond < 0, axis=0) & defined
    consistent = (pos | neg) & ~const_folds.any(axis=0)
    score = np.where(consistent, np.min(np.abs(np.where(np.isfinite(cond), cond, np.nan)),
                                        axis=0), np.nan)
    direction = np.where(pos, 1, np.where(neg, -1, 0)).astype(np.int8)
    return {
        "cond": cond, "raw": raw, "xres_var": xres_var, "rvar": rvar,
        "constant_fold": const_folds, "defined": defined,
        "consistent": consistent, "direction": direction, "score": score,
        "fold_molecule_counts": [int((fold_of_row == j).sum()) for j in folds],
        "n_defined_all_folds": int(defined.sum()),
        "n_direction_consistent": int(consistent.sum()),
    }


def _candidate_rows(meta: Sequence[Mapping[str, Any]], sel: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, t in enumerate(meta):
        rows.append({
            "column": int(i), "type_id": int(t["type_id"]), "size": int(t["size"]),
            "gran": str(t["gran"]), "key": str(t["key"]), "df": int(t["df"]),
            "scaffold_df": int(t["scaffold_df"]), "occurrence_fit": int(t["occurrence_fit"]),
            "equiv_group": str(t["equiv_group"]),
            "equiv_members": [int(x) for x in t["equiv_members"]],
            "equiv_member_keys": list(t["equiv_member_keys"]),
            "cond_corr_folds": [C.safe_float(x) for x in sel["cond"][:, i]],
            "raw_corr_folds": [C.safe_float(x) for x in sel["raw"][:, i]],
            "x_res_var_folds": [float(x) for x in sel["xres_var"][:, i]],
            "constant_train_folds": [bool(x) for x in sel["constant_fold"][:, i]],
            "score": C.safe_float(sel["score"][i]),
            "direction": int(sel["direction"][i]),
            "direction_consistent": bool(sel["consistent"][i]),
        })
    return rows


def df_quartile_buckets(df: np.ndarray, keys: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    p = len(df)
    order = sorted(range(p), key=lambda i: (int(df[i]), str(keys[i])))
    buckets = np.zeros(p, np.int64)
    parts = np.array_split(np.asarray(order, np.int64), 4)
    for b, part in enumerate(parts):
        buckets[part] = b
    boundaries = []
    for b, part in enumerate(parts):
        vals = [int(df[i]) for i in part.tolist()]
        boundaries.append({"bucket": b, "n": int(len(part)),
                           "df_min": int(min(vals)), "df_max": int(max(vals)),
                           "df_median": float(np.median(vals))})
    return buckets, {"rule": "sorted (df asc, key asc) -> four contiguous equal-count groups",
                     "buckets": boundaries}


def frequency_vocabulary(meta: Sequence[Mapping[str, Any]], buckets: np.ndarray,
                         task_cols: Sequence[int]) -> tuple[list[int], dict[str, Any]]:
    cell = Counter()
    for i in task_cols:
        t = meta[i]
        cell[(int(t["size"]), str(t["gran"]), int(buckets[i]))] += 1
    by_cell: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for i, t in enumerate(meta):
        by_cell[(int(t["size"]), str(t["gran"]), int(buckets[i]))].append(i)
    chosen: list[int] = []
    detail = []
    for key in sorted(cell):
        need = int(cell[key])
        cand = sorted(by_cell[key], key=lambda i: (-int(meta[i]["df"]), str(meta[i]["key"])))
        if len(cand) < need:
            raise ValueError(f"frequency cell {key} has {len(cand)} candidates, needs {need}")
        take = cand[:need]
        chosen.extend(take)
        detail.append({"size": key[0], "gran": key[1], "bucket": key[2],
                       "requested": need, "available": len(cand),
                       "selected_columns": [int(x) for x in take]})
    chosen = sorted(set(int(x) for x in chosen))
    return chosen, {"cells": detail, "n_selected": int(len(chosen))}


def _hash_cols(x: sparse.csc_matrix, cols: Sequence[int]) -> str:
    sub = x[:, list(cols)].tocsc()
    return hashlib.sha256(sub.data.tobytes() + sub.indices.tobytes()
                          + sub.indptr.tobytes()).hexdigest()


def run_selection(split: str, y_fit: np.ndarray, label: str,
                  meta: Sequence[Mapping[str, Any]], x0_fit: np.ndarray,
                  x_fit: sparse.csc_matrix, fold_of_row: np.ndarray,
                  alpha: float, K: int) -> dict[str, Any]:
    sel = select_candidates(x0_fit, x_fit, y_fit, fold_of_row, meta, alpha)
    rows = _candidate_rows(meta, sel)
    df = np.asarray([int(t["df"]) for t in meta], np.int64)
    keys = [str(t["key"]) for t in meta]
    buckets, bucket_info = df_quartile_buckets(df, keys)
    for i, row in enumerate(rows):
        row["df_bucket"] = int(buckets[i])
    eligible = [i for i, r in enumerate(rows) if r["direction_consistent"]]
    eligible.sort(key=lambda i: (-float(rows[i]["score"]), str(rows[i]["key"])))
    task_cols = eligible[:int(K)]
    task_set = set(int(x) for x in task_cols)
    freq_cols, freq_info = frequency_vocabulary(meta, buckets, task_cols)
    same = set(freq_cols) == task_set
    return {
        "label": label, "split": split,
        "selection": {k: v for k, v in sel.items() if k not in ("cond", "raw", "xres_var",
                                                                "constant_fold")},
        "rows": rows, "buckets": bucket_info,
        "task_columns": [int(x) for x in task_cols],
        "task_keys": [str(meta[i]["key"]) for i in task_cols],
        "freq_columns": [int(x) for x in freq_cols],
        "freq_keys": [str(meta[i]["key"]) for i in freq_cols],
        "freq_detail": freq_info,
        "intersection": sorted(set(int(x) for x in task_cols) & set(int(x) for x in freq_cols)),
        "n_task": int(len(task_cols)), "n_freq": int(len(freq_cols)),
        "identical_task_freq": bool(same),
        "task_x_hash": _hash_cols(x_fit, task_cols),
        "freq_x_hash": _hash_cols(x_fit, freq_cols),
        "task_cond_corr_sha256": hashlib.sha256(
            sel["cond"][:, task_cols].astype(np.float64).tobytes()).hexdigest() if task_cols else "",
        "freq_cond_corr_sha256": hashlib.sha256(
            sel["cond"][:, freq_cols].astype(np.float64).tobytes()).hexdigest() if freq_cols else "",
    }


# ---------------------------------------------------------------------------
# ridge arms
# ---------------------------------------------------------------------------
def run_ridge_arms(split: str, payload: Mapping[str, Any], x_fit: sparse.csc_matrix,
                   x_eval: sparse.csc_matrix, x0_fit: np.ndarray, x0_eval: np.ndarray,
                   y_fit: np.ndarray, y_eval: np.ndarray, selection: Mapping[str, Any],
                   alpha: float) -> dict[str, Any]:
    arms: dict[str, np.ndarray] = {
        "LOW_RIDGE": x0_fit,
        "ALL_RIDGE": sparse.hstack([sparse.csr_matrix(x0_fit), x_fit], format="csc"),
        "TASK_RIDGE": sparse.hstack([sparse.csr_matrix(x0_fit),
                                     x_fit[:, selection["task_columns"]]], format="csc"),
        "FREQ_RIDGE": sparse.hstack([sparse.csc_matrix(x0_fit),
                                     x_fit[:, selection["freq_columns"]]], format="csc"),
    }
    eval_arms = {
        "LOW_RIDGE": x0_eval,
        "ALL_RIDGE": sparse.hstack([sparse.csr_matrix(x0_eval), x_eval], format="csc"),
        "TASK_RIDGE": sparse.hstack([sparse.csr_matrix(x0_eval),
                                     x_eval[:, selection["task_columns"]]], format="csc"),
        "FREQ_RIDGE": sparse.hstack([sparse.csr_matrix(x0_eval),
                                     x_eval[:, selection["freq_columns"]]], format="csc"),
    }
    out: dict[str, Any] = {}
    for arm in ("LOW_RIDGE", "ALL_RIDGE", "TASK_RIDGE", "FREQ_RIDGE"):
        xf = np.asarray(arms[arm].todense(), np.float64) if sparse.issparse(arms[arm]) \
            else np.asarray(arms[arm], np.float64)
        xe = np.asarray(eval_arms[arm].todense(), np.float64) if sparse.issparse(eval_arms[arm]) \
            else np.asarray(eval_arms[arm], np.float64)
        z, mu, sd = _standardize_train(xf)
        beta = _ridge_beta(z, y_fit.reshape(-1, 1), alpha).ravel()
        zmu = z.mean(axis=0)
        ymu = float(y_fit.mean())
        pf = ((xf - mu) / sd) @ beta + (ymu - zmu @ beta)
        pe = ((xe - mu) / sd) @ beta + (ymu - zmu @ beta)
        out[arm] = {
            "n_features": int(xf.shape[1]),
            "n_constant_features": int(np.count_nonzero(sd <= 1e-12)),
            "fit": ridge_metrics(y_fit, pf),
            "eval": ridge_metrics(y_eval, pe),
            "pred_fit": pf, "pred_eval": pe,
        }
    return out


# ---------------------------------------------------------------------------
# positive control
# ---------------------------------------------------------------------------
def planted_control(split: str, payload: Mapping[str, Any], x_fit: sparse.csc_matrix,
                    x_eval: sparse.csc_matrix, x0_fit: np.ndarray, x0_eval: np.ndarray,
                    y_eval_real: np.ndarray, fold_of_row: np.ndarray,
                    selection: Mapping[str, Any], alpha: float, K: int) -> dict[str, Any]:
    rows = selection["rows"]
    meta = payload["candidates"]
    eligible = [r for r in rows if min(r["x_res_var_folds"]) > 1e-12
                and int(r["size"]) in (3, 4)]
    if not eligible:
        return {"status": "NO_PLANTABLE_CANDIDATE", "passed": False}
    pick = max(eligible, key=lambda r: (min(r["x_res_var_folds"]), -int(r["column"])))
    col = int(pick["column"])
    xf = x_fit[:, col].toarray().ravel().astype(np.float64)
    xe = x_eval[:, col].toarray().ravel().astype(np.float64)
    mu = float(xf.mean())
    sd = float(xf.std())
    if not (sd > 1e-12):
        return {"status": "ZERO_PLANTED_SCALE", "passed": False, "planted": pick}
    y_fit_syn = (xf - mu) / sd
    y_eval_syn = (xe - mu) / sd
    syn = run_selection(split, y_fit_syn, "synthetic_planted", meta, x0_fit,
                        x_fit, fold_of_row, alpha, K)
    planted_keys = set(str(k) for k in pick["equiv_member_keys"])
    planted_in = bool(planted_keys & set(syn["task_keys"]))
    syn_arms = run_ridge_arms(split, payload, x_fit, x_eval, x0_fit, x0_eval,
                              y_fit_syn, y_eval_syn, syn, alpha)
    pick_row = next(r for r in syn["rows"] if int(r["column"]) == col)
    return {
        "status": "EXECUTED",
        "passed": planted_in,
        "planted": {
            "column": col, "type_id": int(pick["type_id"]), "size": int(pick["size"]),
            "gran": str(pick["gran"]), "key": str(pick["key"]),
            "equiv_member_keys": list(pick["equiv_member_keys"]),
            "min_x_res_var": float(min(pick["x_res_var_folds"])),
            "x_res_var_folds": list(pick["x_res_var_folds"]),
            "synthetic_cond_corr_folds": pick_row["cond_corr_folds"],
            "synthetic_score": pick_row["score"],
            "synthetic_direction_consistent": bool(pick_row["direction_consistent"]),
        },
        "planted_in_synthetic_task_vocabulary": planted_in,
        "synthetic_task_size": int(syn["n_task"]),
        "synthetic_selection": {k: v for k, v in syn.items() if k != "rows"},
        "synthetic_ridge": {k: {"n_features": v["n_features"],
                                "fit": v["fit"], "eval": v["eval"]}
                            for k, v in syn_arms.items()},
        "synthetic_y_fit_sha256": C.array_sha256(y_fit_syn),
        "synthetic_y_eval_sha256": C.array_sha256(y_eval_syn),
        "synthetic_labels_used_in_real_training": False,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run_split(split: str) -> dict[str, Any]:
    cfg = C.load_config()
    alpha = float(cfg["selection"]["ridge_alpha"])
    K = int(cfg["selection"]["K"])
    bundle = C.split_bundle(split)
    payload, x_fit, x_eval, x0_fit, x0_eval = V.load_candidates(split)
    if str(payload["split_meta"]["identity_sha256"]) != str(bundle.split["identity_sha256"]):
        raise ValueError("candidate cache and split identity disagree")
    y_fit = np.asarray(bundle.labels[bundle.fit_rows], np.float64)
    y_eval = np.asarray(bundle.labels[bundle.eval_rows], np.float64)
    fold_of_row, fold_info = scaffold_folds(split, bundle.fit_rows,
                                            int(cfg["folds"]["seed"]))
    selection = run_selection(split, y_fit, "real_lipo", payload["candidates"],
                              x0_fit, x_fit, fold_of_row, alpha, K)
    if selection["identical_task_freq"]:
        ridge = run_ridge_arms(split, payload, x_fit, x_eval, x0_fit, x0_eval,
                               y_fit, y_eval, selection, alpha)
        ridge["FREQ_RIDGE"] = {"skipped": "identical to TASK_RIDGE",
                               "n_features": ridge["TASK_RIDGE"]["n_features"],
                               "fit": ridge["TASK_RIDGE"]["fit"],
                               "eval": ridge["TASK_RIDGE"]["eval"],
                               "pred_fit": ridge["TASK_RIDGE"]["pred_fit"],
                               "pred_eval": ridge["TASK_RIDGE"]["pred_eval"]}
    else:
        ridge = run_ridge_arms(split, payload, x_fit, x_eval, x0_fit, x0_eval,
                               y_fit, y_eval, selection, alpha)
    control = planted_control(split, payload, x_fit, x_eval, x0_fit, x0_eval,
                              y_eval, fold_of_row, selection, alpha, K)

    out = C.ANALYSIS_DIR
    out.mkdir(parents=True, exist_ok=True)
    # selection rows are large: keep one JSON with all per-candidate scores
    scores = {
        "protocol_id": cfg["protocol_id"], "version": cfg["version"], "split": split,
        "split_meta": C.split_meta(bundle), "folds": fold_info,
        "label": "real_lipo",
        "selection_rule": cfg["selection"],
        "buckets": selection["buckets"],
        "task_columns": selection["task_columns"],
        "freq_columns": selection["freq_columns"],
        "rows": selection["rows"],
        "task_x_hash": selection["task_x_hash"],
        "freq_x_hash": selection["freq_x_hash"],
        "task_cond_corr_sha256": selection["task_cond_corr_sha256"],
        "freq_cond_corr_sha256": selection["freq_cond_corr_sha256"],
        "n_eligible_direction_consistent": int(selection["selection"]["n_direction_consistent"]),
        "n_candidates": int(len(payload["candidates"])),
        "candidate_cache_sha256": C.sha256_file(out / f"candidate_columns_{split}.npz"),
        "candidate_json_sha256": C.sha256_file(out / f"candidates_{split}.json"),
        "frozen_utc": C.now_utc(),
    }
    C.write_json(out / f"selection_scores_{split}.json", scores)
    C.write_csv(out / f"selection_scores_{split}.csv", selection["rows"])

    # selected word records (Task and Freq) with decoded structure
    atom_dims, bond_dims = C.atom_bond_dims()
    selected = []
    for arm, cols in (("D_TASK", selection["task_columns"]),
                      ("D_FREQ", selection["freq_columns"])):
        for rank, col in enumerate(cols):
            t = payload["candidates"][int(col)]
            entry = {
                "arm": arm, "rank": int(rank), "column": int(col),
                "type_id": int(t["type_id"]), "size": int(t["size"]),
                "gran": str(t["gran"]), "key": str(t["key"]),
                "signature": [int(x) for x in t["signature"]],
                "df": int(t["df"]), "scaffold_df": int(t["scaffold_df"]),
                "occurrence_fit": int(t["occurrence_fit"]),
                "equiv_group": str(t["equiv_group"]),
                "equiv_members": [int(x) for x in t["equiv_members"]],
                "equiv_member_keys": list(t["equiv_member_keys"]),
                "score": next(r["score"] for r in selection["rows"] if r["column"] == col),
                "cond_corr_folds": next(r["cond_corr_folds"] for r in selection["rows"]
                                        if r["column"] == col),
                "raw_corr_folds": next(r["raw_corr_folds"] for r in selection["rows"]
                                       if r["column"] == col),
                "structure": C.decode_word(t, atom_dims, bond_dims),
            }
            selected.append(entry)
    C.write_json(out / f"selected_words_{split}.json", selected)
    C.write_csv(out / f"selected_words_{split}.csv", selected)
    C.write_json(out / f"vocab_{split}_task.json",
                 [e for e in selected if e["arm"] == "D_TASK"])
    C.write_json(out / f"vocab_{split}_freq.json",
                 [e for e in selected if e["arm"] == "D_FREQ"])

    # ridge arms
    ridge_rows = []
    preds: dict[str, np.ndarray] = {}
    for arm, res in ridge.items():
        ridge_rows.append({
            "split": split, "arm": arm, "n_features": res.get("n_features"),
            "fit_rmse": res["fit"]["rmse"], "fit_mae": res["fit"]["mae"],
            "dev_rmse": res["eval"]["rmse"], "dev_mae": res["eval"]["mae"],
            "skipped": res.get("skipped", ""),
        })
        preds[f"{arm}_fit"] = res["pred_fit"]
        preds[f"{arm}_dev"] = res["pred_eval"]
    C.write_csv(out / f"ridge_arms_{split}.csv", ridge_rows)
    np.savez_compressed(out / f"ridge_predictions_{split}.npz", **preds,
                        fit_rows=bundle.fit_rows, eval_rows=bundle.eval_rows)
    ridge_payload = {
        "protocol_id": cfg["protocol_id"], "split": split,
        "split_meta": C.split_meta(bundle),
        "alpha": alpha, "arms": ridge_rows,
        "task_size": int(selection["n_task"]),
        "freq_size": int(selection["n_freq"]),
        "intersection_size": int(len(selection["intersection"])),
        "identical_task_freq": bool(selection["identical_task_freq"]),
        "frozen_utc": C.now_utc(),
    }
    C.write_json(out / f"ridge_arms_{split}.json", ridge_payload)

    control_payload = {
        "protocol_id": cfg["protocol_id"], "split": split,
        "split_meta": C.split_meta(bundle),
        "control": control, "frozen_utc": C.now_utc(),
    }
    C.write_json(out / f"positive_control_{split}.json", control_payload)
    return {
        "split": split, "fold_info": fold_info,
        "n_candidates": int(len(payload["candidates"])),
        "n_task": int(selection["n_task"]), "n_freq": int(selection["n_freq"]),
        "intersection": int(len(selection["intersection"])),
        "ridge": ridge_rows, "control": {k: control.get(k) for k in
                                         ("status", "passed", "planted_in_synthetic_task_vocabulary")},
        "selection_scores_sha256": C.sha256_file(out / f"selection_scores_{split}.json"),
        "selected_words_sha256": C.sha256_file(out / f"selected_words_{split}.json"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("freeze", "run"))
    ap.add_argument("--split", choices=("D0", "D1"), required=True)
    args = ap.parse_args()
    if args.command == "freeze":
        payload = V.freeze_split(args.split)
        print({k: payload[k] for k in ("split", "n_types_total", "n_candidates_qualified",
                                       "n_candidates_dedup", "n_x0_columns")})
    else:
        result = run_split(args.split)
        print(C.jsonable(result))


if __name__ == "__main__":
    main()