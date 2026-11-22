"""Main table, paired bootstrap contrasts and summary for the pilot.

Only Train/Valid prediction arrays are read.  Official Test arrays present in
legacy npz files are never indexed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cpo_common as C

OLD = {
    "A_TRC_FACTOR_CUTRESP": {
        "role": "reused parent", "family": "parent", "model": "FACTOR",
        "run_id": "LIPO_FACTOR_CUTRESP_R0",
        "run_dir": C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0",
    },
    "B_CSOI_ADD": {
        "role": "reused CSOI", "family": "csoi", "model": "CSOI_ADD",
        "run_id": "CSOI_ADD_LIPO_R0",
        "run_dir": C.CSOI_ROUND / "results" / "CSOI_ADD_LIPO_R0",
    },
    "C_CSOI_GLOBAL": {
        "role": "reused CSOI", "family": "csoi", "model": "CSOI_GLOBAL",
        "run_id": "CSOI_GLOBAL_LIPO_R0",
        "run_dir": C.CSOI_ROUND / "results" / "CSOI_GLOBAL_LIPO_R0",
    },
    "E_GINE_RAW_REF": {
        "role": "external reference", "family": "gine", "model": "GINE_RAW",
        "run_id": "GINE_RAW_LIPO_R0",
        "run_dir": C.PARENT_ROUND / "results" / "GINE_RAW_LIPO_R0",
    },
}
NEW = {
    "F_GRAPH": {"role": "trained", "family": "parent", "model": "FACTOR",
                "run_id": "F_GRAPH_LIPO_R0"},
    "B_GRAPH": {"role": "trained", "family": "csoi", "model": "CSOI_ADD",
                "run_id": "B_GRAPH_LIPO_R0"},
    "C_GRAPH": {"role": "trained", "family": "csoi", "model": "CSOI_GLOBAL",
                "run_id": "C_GRAPH_LIPO_R0"},
}
CONTRASTS = [
    ("A_TRC_FACTOR_CUTRESP", "F_GRAPH"),
    ("B_CSOI_ADD", "B_GRAPH"),
    ("C_CSOI_GLOBAL", "C_GRAPH"),
    ("B_GRAPH", "C_GRAPH"),
    ("F_GRAPH", "C_GRAPH"),
    ("C_GRAPH", "E_GINE_RAW_REF"),
]


def _pred_mol(pack: Mapping[str, Any], preds: Mapping[str, np.ndarray]) -> np.ndarray:
    if "train_row_id" in preds:
        return C.predictions_by_molecule(pack, preds)
    train = np.asarray(pack["train_idx"], np.int64)
    valid = np.asarray(pack["valid_idx"], np.int64)
    labels = np.asarray(pack["labels"], np.float64)
    out = np.full(int(pack["n_molecules"]), np.nan, np.float64)
    for rows, split in ((train, "train"), (valid, "valid")):
        p = np.asarray(preds[f"{split}_pred"], np.float64)
        y = np.asarray(preds[f"{split}_label"], np.float64)
        if p.size != rows.size or not np.allclose(y, labels[rows]):
            raise ValueError(f"{split} prediction order does not match the official split")
        out[rows] = p
    return out


def _curve_row(run_dir: Path, epoch: int | None) -> dict[str, Any]:
    path = Path(run_dir) / "curves.csv"
    if not path.is_file():
        return {}
    rows = C.read_csv(path)
    if not rows:
        return {}
    if epoch is None:
        return rows[-1]
    for row in rows:
        if int(float(row.get("epoch", -1))) == int(epoch):
            return row
    return rows[-1]


def load_arm(name: str, entry: Mapping[str, Any], pack: Mapping[str, Any],
             log=print) -> dict[str, Any] | None:
    run_dir = C.run_dir(entry["run_id"]) if "run_dir" not in entry else Path(entry["run_dir"])
    try:
        preds = C.load_predictions(run_dir, kind="best")
    except FileNotFoundError:
        log(f"  [skip] {name}: no predictions in {run_dir}")
        return None
    pred = _pred_mol(pack, preds)
    labels = np.asarray(pack["labels"], np.float64)
    train = np.asarray(pack["train_idx"], np.int64)
    valid = np.asarray(pack["valid_idx"], np.int64)
    if np.any(~np.isfinite(pred[train])) or np.any(~np.isfinite(pred[valid])):
        raise ValueError(f"{name}: missing Train/Valid predictions")
    man_path = Path(run_dir) / "manifest.json"
    man = C.read_json(man_path) if man_path.is_file() else {}
    best_epoch = (man.get("best") or {}).get("epoch")
    row = _curve_row(run_dir, best_epoch)
    out: dict[str, Any] = {
        "name": name, "run_id": entry["run_id"], "role": entry["role"],
        "family": entry["family"], "model": entry["model"],
        "run_dir": str(run_dir),
        "n_train": int(train.size), "n_valid": int(valid.size),
        "train_pred": pred[train], "valid_pred": pred[valid],
        "train_label": labels[train], "valid_label": labels[valid],
        "train_rmse": C.rmse(pred[train], labels[train]),
        "train_mae": C.mae(pred[train], labels[train]),
        "valid_rmse": C.rmse(pred[valid], labels[valid]),
        "valid_mae": C.mae(pred[valid], labels[valid]),
        "valid_pearson": C.pearson(pred[valid], labels[valid]),
        "valid_r2": float(1.0 - np.sum((pred[valid] - labels[valid]) ** 2)
                          / np.sum((labels[valid] - labels[valid].mean()) ** 2)),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "params": (man.get("params") or {}).get("total"),
        "gpu_hours": man.get("gpu_hours"),
        "wall_seconds": man.get("wall_seconds"),
        "manifest": man,
        "curve_row": row,
    }
    out["train_graph_mse_std"] = float(row.get("train_graph_mse_std", float("nan")))
    out["valid_graph_mse_std"] = float(row.get("valid_graph_mse_std", float("nan")))
    out["train_cut_var_std"] = float(row.get("train_cut_var_std", float("nan")))
    out["valid_cut_var_std"] = float(row.get("valid_cut_var_std", float("nan")))
    # old CSOI curves record cut RMSE, not cut MSE
    tcm = row.get("train_cut_mse_std")
    vcm = row.get("valid_cut_mse_std")
    if tcm is None:
        tcm = float(row.get("train_cut_rmse_std", float("nan"))) ** 2
    if vcm is None:
        vcm = float(row.get("valid_cut_rmse_std", float("nan"))) ** 2
    out["train_cut_mse_std"] = float(tcm)
    out["valid_cut_mse_std"] = float(vcm)
    out["train_cancel_share"] = float(row.get("train_cancel_share", float("nan")))
    out["valid_cancel_share"] = float(row.get("valid_cancel_share", float("nan")))
    return out


def contrast(a: Mapping[str, Any], b: Mapping[str, Any], reps: int, seed: int) -> dict[str, Any]:
    ya, pa = a["valid_label"], a["valid_pred"]
    yb, pb = b["valid_label"], b["valid_pred"]
    if not np.allclose(ya, yb):
        raise ValueError("Valid label order mismatch")
    res = C.paired_bootstrap(pa, pb, ya, n_boot=reps, seed=seed)
    res.update({
        "arm_a": a["name"], "arm_b": b["name"],
        "rmse_a": a["valid_rmse"], "rmse_b": b["valid_rmse"],
        "delta_rmse": a["valid_rmse"] - b["valid_rmse"],
        "delta_mae": a["valid_mae"] - b["valid_mae"],
        "delta_train_rmse": a["train_rmse"] - b["train_rmse"],
        "valid_rmse_a": a["valid_rmse"], "valid_rmse_b": b["valid_rmse"],
    })
    res["ci_low"] = res.pop("lo")
    res["ci_high"] = res.pop("hi")
    lo, hi, point = res["ci_low"], res["ci_high"], res["delta_rmse"]
    res["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
    res["direction"] = "b_better" if point > 0 else ("a_better" if point < 0 else "tie")
    res["point_ge_0.02_improvement_for_b"] = bool(point >= 0.02)
    if not res["ci_excludes_zero"]:
        res["reading"] = "not_resolved"
    elif point >= 0.02:
        res["reading"] = "exploratory_positive"
    elif point <= -0.02:
        res["reading"] = "negative"
    else:
        res["reading"] = "resolved_below_0.02"
    return res


def pair_summary() -> dict[str, Any]:
    path = C.analysis_dir() / "pair_delta_comparison.json"
    if not path.is_file():
        return {"available": False}
    data = C.read_json(path)
    out: dict[str, Any] = {"available": True, "by_arm": {}}
    for name, rec in data.get("arms", {}).items():
        entry: dict[str, Any] = {}
        for scope in ("train_in_sample_pairs", "train_valid_cross_pairs", "valid_only_pairs"):
            s = rec["scopes"].get(scope, {})
            if not s.get("n"):
                continue
            w = s["whole_molecule_delta"]
            d = s.get("direct_cut_delta")
            entry[scope] = {
                "n": s["n"], "n_clusters": s.get("n_clusters"),
                "whole_molecule_delta_mae": w["delta_mae"],
                "whole_molecule_delta_rmse": w["delta_rmse"],
                "whole_molecule_direction_accuracy": w["direction_accuracy"],
                "whole_molecule_cluster_ci_mae": w.get("cluster_bootstrap", {}).get("delta_mae"),
            }
            if d:
                entry[scope].update({
                    "direct_cut_delta_mae": d["delta_mae"],
                    "direct_cut_delta_rmse": d["delta_rmse"],
                    "direct_cut_direction_accuracy": d["direction_accuracy"],
                    "direct_cut_cluster_ci_mae": d.get("cluster_bootstrap", {}).get("delta_mae"),
                })
        out["by_arm"][name] = entry
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--reps", type=int, default=int(C.load_config()["decision"]["bootstrap_reps"]))
    ap.add_argument("--seed", type=int, default=int(C.load_config()["decision"]["bootstrap_seed"]))
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = C.load_pack(cfg, args.dataset)
    log = print
    arms: dict[str, dict[str, Any]] = {}
    for name, entry in OLD.items():
        rec = load_arm(name, entry, pack, log=log)
        if rec is not None:
            arms[name] = rec
    for name, entry in NEW.items():
        rec = load_arm(name, entry, pack, log=log)
        if rec is not None:
            arms[name] = rec

    # attach best-checkpoint cut diagnostics from the audit for reused arms
    audit_path = C.analysis_dir() / "loss_decomposition.json"
    if audit_path.is_file():
        audit = C.read_json(audit_path)
        for name, rec in arms.items():
            a = audit["arms"].get(name)
            if a is not None:
                for split in ("train", "valid"):
                    s = a["splits"][split]
                    for key in ("graph_mse_std", "cut_var_std", "cut_mse_std",
                                "cancel_share", "var_share_of_lcut"):
                        rec[f"{split}_{key}"] = s.get(key)
                rec["best_epoch"] = a.get("best_epoch", rec["best_epoch"])
                rec["params"] = rec["params"] or a.get("params")

    table: list[dict[str, Any]] = []
    for name, rec in arms.items():
        table.append({
            "arm": name, "role": rec["role"], "family": rec["family"],
            "model": rec["model"], "run_id": rec["run_id"],
            "n_train": rec["n_train"], "n_valid": rec["n_valid"],
            "train_rmse": rec["train_rmse"], "train_mae": rec["train_mae"],
            "valid_rmse": rec["valid_rmse"], "valid_mae": rec["valid_mae"],
            "valid_pearson": rec["valid_pearson"], "valid_r2": rec["valid_r2"],
            "best_epoch": rec["best_epoch"], "params": rec["params"],
            "gpu_hours": rec["gpu_hours"],
            "train_graph_mse_std": rec.get("train_graph_mse_std"),
            "valid_graph_mse_std": rec.get("valid_graph_mse_std"),
            "train_cut_var_std": rec.get("train_cut_var_std"),
            "valid_cut_var_std": rec.get("valid_cut_var_std"),
            "train_cut_mse_std": rec.get("train_cut_mse_std"),
            "valid_cut_mse_std": rec.get("valid_cut_mse_std"),
            "train_cancel_share": rec.get("train_cancel_share"),
            "valid_cancel_share": rec.get("valid_cancel_share"),
        })
    table.sort(key=lambda r: list(arms).index(r["arm"]))
    C.write_csv(C.results_root() / "main_table.csv", table)

    contrasts: dict[str, Any] = {}
    for a_name, b_name in CONTRASTS:
        if a_name not in arms or b_name not in arms:
            continue
        contrasts[f"{a_name}->{b_name}"] = contrast(arms[a_name], arms[b_name],
                                                    args.reps, args.seed)

    gap = {}
    if "C_GRAPH" in arms and "E_GINE_RAW_REF" in arms:
        gap["C_GRAPH_minus_GINE_valid_rmse"] = arms["C_GRAPH"]["valid_rmse"] - arms["E_GINE_RAW_REF"]["valid_rmse"]
        if "C_CSOI_GLOBAL" in arms:
            gap["C_CSOI_GLOBAL_minus_GINE_valid_rmse"] = arms["C_CSOI_GLOBAL"]["valid_rmse"] - arms["E_GINE_RAW_REF"]["valid_rmse"]
            gap["gap_narrowing"] = gap["C_CSOI_GLOBAL_minus_GINE_valid_rmse"] - gap["C_GRAPH_minus_GINE_valid_rmse"]
    train_valid = {}
    for key, c in contrasts.items():
        train_valid[key] = {
            "delta_train_rmse": c["delta_train_rmse"],
            "delta_valid_rmse": c["delta_rmse"],
            "train_improved_for_b": c["delta_train_rmse"] > 0,
            "valid_improved_for_b": c["delta_rmse"] > 0,
            "only_train": c["delta_train_rmse"] > 0 and c["delta_rmse"] <= 0,
        }
    cut_var = {}
    for name, rec in arms.items():
        cut_var[name] = {
            "train_cut_var_std": rec.get("train_cut_var_std"),
            "valid_cut_var_std": rec.get("valid_cut_var_std"),
            "train_cancel_share": rec.get("train_cancel_share"),
            "valid_cancel_share": rec.get("valid_cancel_share"),
            "train_graph_mse_std": rec.get("train_graph_mse_std"),
            "valid_graph_mse_std": rec.get("valid_graph_mse_std"),
        }

    summary = {
        "protocol": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "commit": C.git_commit(),
        "cache_sha256": pack["cache_sha256"],
        "cache_content_sha256": pack.get("content_sha256"),
        "cache_path_used": pack["cache_path_used"],
        "test_policy": "no official Test label read; model selection by Valid only",
        "bootstrap": {"reps": args.reps, "seed": args.seed,
                      "unit": "Valid molecule",
                      "train_seed_uncertainty_included": False},
        "signal_definition": {
            "exploratory_signal_rmse": float(cfg["decision"]["exploratory_signal_rmse"]),
            "note": "single seed 0; ~0.02 is an exploratory signal, not a formal confirmation threshold",
        },
        "arms": {
            name: {k: v for k, v in rec.items()
                   if k not in ("train_pred", "valid_pred", "train_label", "valid_label",
                                "manifest", "curve_row")}
            for name, rec in arms.items()
        },
        "main_table": table,
        "contrasts": contrasts,
        "train_valid_decomposition": train_valid,
        "cut_diagnostics": cut_var,
        "pair_delta_summary": pair_summary(),
        "gine_gap": gap,
        "execution": {
            "runs": {
                name: {
                    "run_id": rec["run_id"],
                    "commit": rec["manifest"].get("signature", {}).get("commit"),
                    "environment": rec["manifest"].get("environment"),
                    "gpu_hours": rec["gpu_hours"],
                    "wall_seconds": rec["wall_seconds"],
                    "main_loss": rec["manifest"].get("main_loss"),
                    "test_policy": rec["manifest"].get("test_policy"),
                }
                for name, rec in arms.items() if rec["role"] == "trained"
            },
            "gpu_hours_total_new_training": sum(
                float(rec["gpu_hours"] or 0.0) for rec in arms.values() if rec["role"] == "trained"),
            "budget_gpu_hours_max": float(cfg["budget"]["gpu_hours_max"]),
        },
    }
    C.write_json(C.results_root() / "summary.json", summary)
    print(f"wrote {C.results_root() / 'main_table.csv'}")
    print(f"wrote {C.results_root() / 'summary.json'}")
    for row in table:
        print(f"{row['arm']:24s} train={row['train_rmse']:.4f} valid={row['valid_rmse']:.4f} "
              f"mae={row['valid_mae']:.4f} var={row['valid_cut_var_std']}")
    for key, c in contrasts.items():
        print(f"{key:42s} d={c['delta_rmse']:+.4f} CI=[{c['ci_low']:+.4f},{c['ci_high']:+.4f}] "
              f"{c['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())