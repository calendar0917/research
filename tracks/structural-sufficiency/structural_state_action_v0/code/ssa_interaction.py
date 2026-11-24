"""Section 3: does LCRC predict the *environment interaction*, not just the mean?

For one canonical replacement A -> B observed in two different exact contexts
C1, C2 the real and modelled double differences are

``D_true  = delta_true(C2) - delta_true(C1)``,
``D_model = delta_model(C2) - delta_model(C1)``,

with ``delta_true(C) = y(C+B) - y(C+A)`` from the parent cell aggregation and
``delta_model`` one of ZERO / PAIR_Q / LOCAL / LCRC.  ZERO is the
environment-independent fixed replacement response; PAIR_Q and LOCAL are the
frozen SFEC lookup / ridge estimates; LCRC is ``pred(B) - pred(A)`` from the
frozen W3 checkpoint.

Rectangles are not independent samples: metrics are reported observation-equal
*and* as a pair-type macro (one value per canonical replacement type, then
averaged), with pair-type bootstrap intervals.  A molecule-disjoint rectangle
subset (no original molecule identity is shared by two rectangles of the same
pair type) is the primary uncertainty check.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import ssa_common as C

MODELS = ("ZERO", "PAIR_Q", "LOCAL", "LCRC")


def _dir_acc(d_true: np.ndarray, d_pred: np.ndarray, thr: float) -> tuple[float, int]:
    m = np.abs(d_true) >= float(thr)
    if not m.any():
        return float("nan"), 0
    return float(np.mean(np.sign(d_pred[m]) == np.sign(d_true[m]))), int(m.sum())


def dd_metrics(d_true: np.ndarray, d_pred: np.ndarray) -> dict[str, float]:
    d_true = np.asarray(d_true, np.float64)
    d_pred = np.asarray(d_pred, np.float64)
    n = int(d_true.size)
    if n == 0:
        return {"n": 0, "mae_eV": float("nan"), "rmse_eV": float("nan"),
                "corr": float("nan"), "slope": float("nan"),
                "dir_acc": float("nan"), "dir_n": 0,
                "std_d_true": float("nan"), "std_d_pred": float("nan"),
                "mean_d_true": float("nan"), "mean_d_pred": float("nan")}
    err = d_pred - d_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    st, sp = float(np.std(d_true)), float(np.std(d_pred))
    corr = float(np.corrcoef(d_true, d_pred)[0, 1]) if st > 0 and sp > 0 else float("nan")
    slope = float(np.polyfit(d_true, d_pred, 1)[0]) if st > 0 else float("nan")
    acc, nn = _dir_acc(d_true, d_pred, C.DIRECTION_THRESHOLD_EV)
    return {"n": n, "mae_eV": mae, "rmse_eV": rmse, "corr": corr, "slope": slope,
            "dir_acc": acc, "dir_n": nn, "std_d_true": st, "std_d_pred": sp,
            "mean_d_true": float(np.mean(d_true)), "mean_d_pred": float(np.mean(d_pred))}


def pair_type_macro(values: np.ndarray, pair_index: np.ndarray, n_pairs: int) -> np.ndarray:
    """Per-pair-type mean, returned as an array of length ``n_pairs``."""
    counts = np.bincount(pair_index, minlength=n_pairs).astype(np.float64)
    sums = np.bincount(pair_index, weights=np.asarray(values, np.float64), minlength=n_pairs)
    out = np.full(n_pairs, np.nan)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out


def macro_metrics(d_true: np.ndarray, d_pred: np.ndarray, pair_index: np.ndarray,
                  n_pairs: int, reps: int = C.BOOTSTRAP_REPS, seed: int = 0) -> dict[str, Any]:
    """Pair-type macro metrics with a bootstrap over pair types."""
    ae = np.abs(np.asarray(d_pred) - np.asarray(d_true))
    se = (np.asarray(d_pred) - np.asarray(d_true)) ** 2
    abs_true = np.abs(np.asarray(d_true))
    hit = (np.sign(d_pred) == np.sign(d_true)) & (abs_true >= C.DIRECTION_THRESHOLD_EV)
    got = abs_true >= C.DIRECTION_THRESHOLD_EV
    n_pairs = int(n_pairs)
    mae = pair_type_macro(ae, pair_index, n_pairs)
    rmse = np.sqrt(pair_type_macro(se, pair_index, n_pairs))
    hit_c = np.bincount(pair_index, weights=hit.astype(np.float64), minlength=n_pairs)
    got_c = np.bincount(pair_index, weights=got.astype(np.float64), minlength=n_pairs)
    valid = np.flatnonzero(~np.isnan(mae))
    if valid.size == 0:
        return {"n_pairs": 0, "mae_eV": float("nan"), "rmse_eV": float("nan"),
                "dir_acc": float("nan"), "dir_n": 0, "ci_mae": [float("nan"), float("nan")],
                "per_pair_mae": mae, "per_pair_rmse": rmse}
    dir_pair = np.where(got_c > 0, hit_c / np.maximum(got_c, 1.0), np.nan)
    dir_valid = np.flatnonzero(~np.isnan(dir_pair))
    rng = np.random.default_rng(int(seed))
    boot = rng.integers(0, valid.size, size=(int(reps), valid.size))
    mae_boot = mae[valid][boot].mean(axis=1)
    return {
        "n_pairs": int(valid.size),
        "mae_eV": float(np.mean(mae[valid])),
        "rmse_eV": float(np.mean(rmse[valid])),
        "dir_acc": float(np.mean(dir_pair[dir_valid])) if dir_valid.size else float("nan"),
        "dir_n": int(got_c[dir_valid].sum()) if dir_valid.size else 0,
        "ci_mae": [float(np.quantile(mae_boot, 0.025)), float(np.quantile(mae_boot, 0.975))],
        "per_pair_mae": mae,
        "per_pair_rmse": rmse,
        "per_pair_dir": dir_pair,
    }


def macro_contrast(a: np.ndarray, b: np.ndarray, pair_index: np.ndarray, n_pairs: int,
                   reps: int = C.BOOTSTRAP_REPS, seed: int = 0) -> dict[str, Any]:
    """Paired pair-type macro difference ``mean(a) - mean(b)`` with CI."""
    va = pair_type_macro(a, pair_index, n_pairs)
    vb = pair_type_macro(b, pair_index, n_pairs)
    m = np.isfinite(va) & np.isfinite(vb)
    if not m.any():
        return {"point": float("nan"), "ci": [float("nan"), float("nan")], "n_pairs": 0}
    diff = va[m] - vb[m]
    rng = np.random.default_rng(int(seed))
    boot = rng.integers(0, diff.size, size=(int(reps), diff.size))
    means = diff[boot].mean(axis=1)
    return {"point": float(diff.mean()),
            "ci": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "n_pairs": int(diff.size)}


def context_equal(values: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """Weight every distinct context of a rectangle set equally."""
    u, inv = np.unique(ctx, return_inverse=True)
    cnt = np.bincount(inv, minlength=u.size).astype(np.float64)
    s = np.bincount(inv, weights=np.asarray(values, np.float64), minlength=u.size)
    return s / np.maximum(cnt, 1.0)


def molecule_disjoint(rect: Mapping[str, np.ndarray], mol_cols: Mapping[str, np.ndarray]) -> np.ndarray:
    """Greedy molecule-disjoint rectangles, independently inside each pair type.

    Rectangles are visited in fixed hash order inside a pair type; a rectangle
    is kept only if none of its four original molecule identities was already
    used by a kept rectangle of the same pair type.
    """
    key = rect["pair_key"]
    order = np.lexsort((rect["hash"], key))
    keep = np.zeros(len(key), bool)
    used: dict[int, set[int]] = {}
    for i in order.tolist():
        k = int(key[i])
        s = used.setdefault(k, set())
        mols = [int(mol_cols[c][i]) for c in ("m1", "m2", "m3", "m4")]
        if s.intersection(mols):
            continue
        keep[i] = True
        s.update(mols)
    return keep


def analyse(obs: Mapping[str, Any], sel: Mapping[str, Any]) -> dict[str, Any]:
    rect = sel["rect"]
    pair_types = sel["pair_types"]
    n_pairs = len(pair_types)
    pair_index = np.searchsorted(pair_types, rect["pair_key"])

    mol_cols = {
        "m1": obs["cell_mol"][rect["pos1_i"]],
        "m2": obs["cell_mol"][rect["pos1_j"]],
        "m3": obs["cell_mol"][rect["pos2_i"]],
        "m4": obs["cell_mol"][rect["pos2_j"]],
    }
    disjoint = molecule_disjoint(rect, mol_cols)
    rect_mols = np.stack([mol_cols[c] for c in ("m1", "m2", "m3", "m4")], axis=1)

    d_true = rect["delta_true_2"] - rect["delta_true_1"]
    d_pred = {
        "ZERO": np.zeros_like(d_true),
        "PAIR_Q": rect["delta_pairq_2"] - rect["delta_pairq_1"],
        "LOCAL": rect["delta_local_2"] - rect["delta_local_1"],
        "LCRC": rect["delta_lcrc_2"] - rect["delta_lcrc_1"],
    }

    subsets: dict[str, np.ndarray] = {
        "all": np.ones(len(d_true), bool),
        "same_q": np.asarray(rect["same_q"], bool),
        "different_q": ~np.asarray(rect["same_q"], bool),
        "both_contexts_unseen_in_1m_prefix": np.asarray(rect["unseen_both"], bool),
        "clean_cells": np.asarray(rect["clean"], bool),
        "molecule_disjoint": disjoint,
    }

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, mask in subsets.items():
        n = int(mask.sum())
        if n == 0:
            summary[name] = {"n_rect": 0, "note": "no rectangle in this subset"}
            continue
        dt = d_true[mask]
        idx = pair_index[mask]
        present = np.unique(idx)
        n_pt = int(present.size)
        pts = rect["pair_key"][mask]
        rec: dict[str, Any] = {
            "n_rect": n,
            "n_pair_types": n_pt,
            "n_contexts": int(np.unique(np.r_[rect["ctx1"][mask], rect["ctx2"][mask]]).size),
            "n_unique_molecules": int(np.unique(rect_mols[mask]).size),
            "n_rectangles_possible_in_subset_pair_types": int(np.asarray(sel["n_possible_per_pair"])[present].sum()),
            "n_rect_selected_in_subset_pair_types": int(np.isin(pair_index, present).sum()),
            "models": {},
        }
        for model in MODELS:
            dp = d_pred[model][mask]
            obs_m = dd_metrics(dt, dp)
            mac = macro_metrics(dt, dp, idx, n_pairs, seed=11)
            rec["models"][model] = {"observation": obs_m, "pair_type_macro": {
                "n_pairs": mac["n_pairs"], "mae_eV": mac["mae_eV"], "rmse_eV": mac["rmse_eV"],
                "dir_acc": mac["dir_acc"], "dir_n": mac["dir_n"], "ci_mae": mac["ci_mae"],
            }}
            rows.append({
                "subset": name, "weighting": "observation", "model": model,
                "n_rect": n, "n_pair_types": n_pt,
                "n_contexts": rec["n_contexts"], "n_unique_molecules": rec["n_unique_molecules"],
                "mae_eV": obs_m["mae_eV"], "rmse_eV": obs_m["rmse_eV"],
                "corr": obs_m["corr"], "slope": obs_m["slope"],
                "dir_acc": obs_m["dir_acc"], "dir_n": obs_m["dir_n"],
                "std_d_true": obs_m["std_d_true"], "std_d_pred": obs_m["std_d_pred"],
                "mean_d_true": obs_m["mean_d_true"], "mean_d_pred": obs_m["mean_d_pred"],
                "mae_ci_lo": "", "mae_ci_hi": "",
            })
            rows.append({
                "subset": name, "weighting": "pair_type_macro", "model": model,
                "n_rect": n, "n_pair_types": mac["n_pairs"],
                "n_contexts": rec["n_contexts"], "n_unique_molecules": rec["n_unique_molecules"],
                "mae_eV": mac["mae_eV"], "rmse_eV": mac["rmse_eV"],
                "corr": "", "slope": "",
                "dir_acc": mac["dir_acc"], "dir_n": mac["dir_n"],
                "std_d_true": obs_m["std_d_true"], "std_d_pred": obs_m["std_d_pred"],
                "mean_d_true": obs_m["mean_d_true"], "mean_d_pred": obs_m["mean_d_pred"],
                "mae_ci_lo": mac["ci_mae"][0], "mae_ci_hi": mac["ci_mae"][1],
            })
        # paired contrasts on the pair-type macro scale
        rec["contrasts"] = {}
        for other in ("ZERO", "PAIR_Q", "LOCAL"):
            c = macro_contrast(np.abs(d_pred["LCRC"][mask] - dt),
                               np.abs(d_pred[other][mask] - dt), idx, n_pairs, seed=12)
            rec["contrasts"][f"LCRC_minus_{other}"] = c
            rows.append({
                "subset": name, "weighting": "pair_type_macro",
                "model": f"contrast:LCRC-{other}",
                "n_rect": n, "n_pair_types": c["n_pairs"],
                "n_contexts": rec["n_contexts"], "n_unique_molecules": rec["n_unique_molecules"],
                "mae_eV": c["point"], "rmse_eV": "", "corr": "", "slope": "",
                "dir_acc": "", "dir_n": "", "std_d_true": "", "std_d_pred": "",
                "mean_d_true": "", "mean_d_pred": "",
                "mae_ci_lo": c["ci"][0], "mae_ci_hi": c["ci"][1],
            })
        summary[name] = rec

    # context-equal sensitivity on the full set (each context weighted once)
    ce = {}
    for model in MODELS:
        ce[model] = {
            "mae_eV": float(np.mean(context_equal(np.abs(d_pred[model] - d_true), rect["ctx1"]))),
            "note": "per-context mean absolute double-difference error, then averaged",
        }

    return {
        "rows": rows,
        "summary": summary,
        "context_equal": ce,
        "disjoint_mask": disjoint,
        "d_true": d_true,
        "d_pred": d_pred,
        "pair_index": pair_index,
        "n_pairs": n_pairs,
        "molecule_cols": mol_cols,
    }


__all__ = ["MODELS", "analyse", "dd_metrics", "macro_metrics", "molecule_disjoint"]
