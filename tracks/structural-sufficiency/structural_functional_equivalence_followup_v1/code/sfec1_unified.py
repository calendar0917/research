"""Phase B: unified direct versus shared conditional comparison on one set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec1_common as C
import sfec1_data as D

MODELS = ["ZERO", "PAIR", "PAIR_MEAN", "SHARED", "PAIR_Q", "PAIR_Q_MEAN",
          "PAIR_Q_NULL", "LOCAL", "LOCAL_NULL"]
CONDITIONAL = {"PAIR_Q", "PAIR_Q_MEAN", "PAIR_Q_NULL", "LOCAL", "LOCAL_NULL"}
COMPARISONS = [
    ("PAIR_Q", "PAIR"),
    ("PAIR_Q_MEAN", "PAIR_MEAN"),
    ("PAIR_Q", "PAIR_Q_NULL"),
    ("LOCAL", "SHARED"),
    ("LOCAL", "LOCAL_NULL"),
    ("LOCAL", "PAIR_Q"),
    ("LOCAL", "PAIR_Q_MEAN"),
]
PAIR_MACRO_MODELS = ["PAIR", "PAIR_MEAN", "SHARED", "PAIR_Q", "PAIR_Q_MEAN",
                     "PAIR_Q_NULL", "LOCAL", "LOCAL_NULL"]


def load_prepared() -> dict[str, np.ndarray]:
    z = np.load(C.DATA_DIR / "prepared_pcqm.npz")
    return {k: z[k] for k in z.files}


def _context_weighted_rmse(row_ctx: np.ndarray, sq: np.ndarray, n_ctx: int) -> float:
    cnt = np.bincount(row_ctx, minlength=n_ctx).astype(np.float64)
    num = np.bincount(row_ctx, weights=sq, minlength=n_ctx)
    ok = cnt > 0
    return float(np.sqrt(np.mean(num[ok] / cnt[ok]))) if np.any(ok) else float("nan")


def _pair_weighted_rmse(pair_idx: np.ndarray, sq: np.ndarray) -> float:
    sums = np.bincount(pair_idx, weights=sq)
    cnt = np.bincount(pair_idx)
    ok = cnt > 0
    return float(np.sqrt(np.mean(sums[ok] / cnt[ok]))) if np.any(ok) else float("nan")


def _weighted_points(
    delta: np.ndarray, preds: dict[str, np.ndarray], row_ctx: np.ndarray,
    pair_idx: np.ndarray, n_ctx: int,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, pred in preds.items():
        err = np.asarray(pred, np.float64) - delta
        ae = np.abs(err)
        sq = err ** 2
        base = D.basic_metric(delta, pred)
        out[name] = {
            "observation": {
                "mae": base["mae"], "rmse": base["rmse"],
                "bias": base["bias"], "sign_accuracy": base["sign_accuracy"],
            },
            "context": {
                "mae": D.context_equal_mae(row_ctx, ae, n_ctx),
                "rmse": _context_weighted_rmse(row_ctx, sq, n_ctx),
                "sign_accuracy": base["sign_accuracy"],
            },
            "pair_type": {
                "mae": D.pair_macro_mae(pair_idx, ae),
                "rmse": _pair_weighted_rmse(pair_idx, sq),
                "sign_accuracy": base["sign_accuracy"],
            },
        }
    return out


def _compact_pair_index(pair_index: np.ndarray) -> np.ndarray:
    u, inv = np.unique(pair_index, return_inverse=True)
    return inv.astype(np.int64)


def run_subset(
    p: dict[str, np.ndarray], mask: np.ndarray, label: str, cfg: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    delta = p["eval_delta"][mask]
    row_ctx, n_ctx = D.compact_ctx(p["eval_ctx"][mask])
    pair_idx = _compact_pair_index(p["eval_pair_index"][mask])
    preds = {name: p[f"eval_preds_{name}"][mask] for name in MODELS}
    enabled = {}
    for name in CONDITIONAL:
        key = f"eval_enabled_{name}"
        if key not in p and name == "PAIR_Q_MEAN":
            key = "eval_enabled_PAIR_Q"
        enabled[name] = p[key][mask]
    points = _weighted_points(delta, preds, row_ctx, pair_idx, n_ctx)
    reps = int(cfg["statistics"]["bootstrap_replicates"])
    seed = int(cfg["statistics"]["bootstrap_seed"])
    boot_obs = D.bootstrap_obs_context(
        delta, preds, row_ctx, n_ctx, context_equal=False, reps=reps,
        seed=seed + 101, comparisons=COMPARISONS,
    )
    boot_ctx = D.bootstrap_obs_context(
        delta, preds, row_ctx, n_ctx, context_equal=True, reps=reps,
        seed=seed + 102, comparisons=COMPARISONS,
    )
    pm_preds = {name: preds[name] for name in PAIR_MACRO_MODELS}
    boot_pair = D.bootstrap_pair_macro(
        delta, pm_preds, row_ctx, pair_idx, n_ctx, reps=reps,
        seed=seed + 103, comparisons=[c for c in COMPARISONS if c[0] in pm_preds and c[1] in pm_preds],
    )
    result: dict[str, Any] = {
        "subset": label,
        "n_obs": int(len(delta)),
        "n_contexts": int(n_ctx),
        "n_pair_types": int(len(np.unique(pair_idx))),
        "points": points,
        "coverage": {name: float(np.mean(enabled[name])) for name in CONDITIONAL},
        "bootstrap": {
            "observation": boot_obs,
            "context": boot_ctx,
            "pair_type": boot_pair,
        },
    }
    # tidy CSV rows
    rows = []
    base_common = {
        "subset": label, "n_obs": int(len(delta)), "n_contexts": int(n_ctx),
        "n_pair_types": int(len(np.unique(pair_idx))),
    }
    for name in MODELS:
        for weighting, key in (("observation", "observation"), ("context_equal", "context"),
                               ("pair_type_equal", "pair_type")):
            b = boot_obs["models"].get(name) if weighting == "observation" else (
                boot_ctx["models"].get(name) if weighting == "context_equal" else boot_pair["models"].get(name)
            )
            rows.append({
                **base_common, "row_kind": "model", "weighting": weighting,
                "model": name, "comparison": "",
                "mae": points[name][key]["mae"],
                "rmse": points[name][key]["rmse"],
                "sign_accuracy": points[name][key]["sign_accuracy"],
                "coverage": (float(np.mean(enabled[name])) if name in CONDITIONAL else ""),
                "ci_lo": b["ci"][0] if b is not None else "",
                "ci_hi": b["ci"][1] if b is not None else "",
                "relative_gain_vs_first": "",
                "note": "conditional correction enabled fraction" if name in CONDITIONAL else "",
            })
    for a, bname in COMPARISONS:
        for weighting, boot in (("observation", boot_obs), ("context_equal", boot_ctx),
                                ("pair_type_equal", boot_pair)):
            key = f"{a}_minus_{bname}"
            if key not in boot["diffs"]:
                continue
            d = boot["diffs"][key]
            ma = points[a]["observation" if weighting == "observation" else
                         ("context" if weighting == "context_equal" else "pair_type")]["mae"]
            mb = points[bname]["observation" if weighting == "observation" else
                               ("context" if weighting == "context_equal" else "pair_type")]["mae"]
            rows.append({
                **base_common, "row_kind": "paired_diff", "weighting": weighting,
                "model": "", "comparison": key,
                "mae": d["point"],
                "rmse": "",
                "sign_accuracy": "",
                "coverage": "",
                "ci_lo": d["ci"][0], "ci_hi": d["ci"][1],
                "relative_gain_vs_first": 1.0 - mb / max(ma, 1e-12),
                "note": "positive means the second model has lower MAE",
            })
    # per-pair results
    by_pair = []
    key_all = p["eval_key"][mask]
    for pi in np.unique(pair_idx):
        sel = pair_idx == pi
        kk = int(key_all[sel][0])
        rec = {
            "subset": label, "pair_key": kk,
            "A_rank": int(kk // 64), "B_rank": int(kk % 64),
            "n_obs": int(sel.sum()),
            "n_contexts": int(len(np.unique(p["eval_ctx"][mask][sel]))),
        }
        for name in MODELS:
            rec[f"mae_{name}"] = float(np.mean(np.abs(preds[name][sel] - delta[sel])))
        rec["relative_gain_LOCAL_vs_SHARED"] = (
            1.0 - rec["mae_LOCAL"] / max(rec["mae_SHARED"], 1e-12)
        )
        rec["relative_gain_PAIR_Q_vs_PAIR"] = (
            1.0 - rec["mae_PAIR_Q"] / max(rec["mae_PAIR"], 1e-12)
        )
        by_pair.append(rec)
    return result, rows, by_pair


def main() -> int:
    cfg = C.load_config()
    p = load_prepared()
    masks = {
        "main_pair_shared_support": p["eval_main"].astype(bool),
        "all_conditional_models_enabled": p["eval_allcond"].astype(bool),
        "main_clean_subset": p["eval_main"].astype(bool) & p["eval_clean"].astype(bool),
    }
    all_rows: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for label, mask in masks.items():
        if not np.any(mask):
            continue
        result, rows, by_pair = run_subset(p, mask, label, cfg)
        details[label] = result
        all_rows.extend(rows)
        all_pairs.extend(by_pair)
        print(label, result["n_obs"], result["n_contexts"], result["n_pair_types"])
    C.write_csv(C.ANALYSIS_DIR / "conditional_transfer.csv", all_rows)
    C.write_csv(C.ANALYSIS_DIR / "conditional_transfer_by_pair.csv", all_pairs)
    C.write_json(C.ANALYSIS_DIR / "conditional_transfer.json", details)
    print(json.dumps({k: {"n_obs": v["n_obs"], "n_contexts": v["n_contexts"]}
                      for k, v in details.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
