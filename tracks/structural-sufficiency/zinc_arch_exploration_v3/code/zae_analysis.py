"""Valid-side analysis for zinc-arch-exploration-v3.

Reads the frozen best-valid checkpoints' stored predictions for the round's
formal seed-0 arms plus the reused P3B/L/C predictions, and produces:

* the main table (params, best valid, best epoch, eval-mode train@best, final,
  updates, SSL, status, GPU-h);
* paired bootstrap contrasts (same row resamples for every contrast);
* the fixed subgroup table (radius-2 signature unseen / covered, ring>=4,
  tail y <= -3, P3B worst-100) using the frozen audit masks;
* train/valid/LR curves;
* a descriptive decision payload.

No official test data is touched.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SUF = _EXP.parent
_V1 = _SUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import zae_models as ZM  # noqa: E402
import zae_train as ZT  # noqa: E402

ANALYSIS = _EXP / "analysis"
RUNS = _EXP / "runs"
DCC = _SUF / "zinc_dual_channel_composer_v1"
AUDIT = _SUF / "zinc_generalization_audit_v1"
P3B_RUN = _SUF / "zinc_recipe_trunk_v2" / "runs" / "xBssl_P3N_s0"

ARMS = ("A_s0", "B_s0", "A_SSL_s0", "A_attn_s0")
REFS = ("P3B_s0", "L_s0", "C_s0")
BOOT = 10_000
BOOT_SEED = 20261022


def load_npz_pred(path: Path, expected_rows: int) -> np.ndarray:
    z = np.load(path)
    rows = np.asarray(z["row"], dtype=np.int64)
    pred = np.asarray(z["pred"], dtype=np.float64)
    if rows.shape[0] != expected_rows or not np.array_equal(rows, np.arange(expected_rows)):
        raise RuntimeError(f"{path}: unexpected row coverage")
    return pred


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    ae = np.abs(pred - y)
    return {
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "max_ae": float(ae.max()),
        "mean_signed": float((pred - y).mean()),
        "top100_share": float(np.sort(ae)[-100:].sum() / ae.sum()),
    }


def paired_bootstrap(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray, *, seed: int) -> dict:
    ae_a = np.abs(pred_a - y)
    ae_b = np.abs(pred_b - y)
    delta = ae_a - ae_b
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(BOOT, dtype=np.float64)
    for i in range(BOOT):
        idx = rng.integers(0, n, n)
        boots[i] = delta[idx].mean()
    return {
        "mean_delta_mae": float(delta.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n": int(n),
        "note": "delta = AE(first) - AE(second); negative means first is better",
    }


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    valid_cache = __import__("ad_data").load_cache("valid")
    y = np.asarray(valid_cache["labels"], dtype=np.float64)
    n_valid = len(y)

    preds: dict[str, np.ndarray] = {}
    infos: dict[str, dict] = {}
    preds["P3B_s0"] = load_npz_pred(P3B_RUN / "predictions" / "valid_best.npz", n_valid)
    infos["P3B_s0"] = C.read_json(P3B_RUN / "train.json")
    for arm in ("L", "C"):
        run = DCC / "runs" / f"{arm}_s0"
        preds[f"{arm}_s0"] = load_npz_pred(run / "predictions" / "valid_best.npz", n_valid)
        infos[f"{arm}_s0"] = C.read_json(run / "train.json")
    missing = []
    truncated = []
    for arm in ARMS:
        run = RUNS / arm
        if (run / "train.json").exists():
            infos[arm] = C.read_json(run / "train.json")
            arr = load_npz_pred(run / "predictions" / "valid_best.npz", n_valid)
            stored = float(infos[arm]["fixed_objects"]["best"]["valid_mae"])
            if abs(float(np.abs(arr - y).mean()) - stored) > 1e-6:
                raise RuntimeError(f"{arm}: stored best valid does not match its predictions")
            preds[arm] = arr
        elif (run / "train_partial.json").exists():
            # budget-stopped run: curve and best checkpoint recovered from the log
            info = C.read_json(run / "train_partial.json")
            info["fixed_objects"] = {"best": {"valid_mae": info["best_valid"]["valid_mae"],
                                              "epoch": info["best_valid"]["epoch"],
                                              "train_mae_eval_mode": float("nan")},
                                     "final": None}
            info["best_valid"] = {"valid_mae": info["best_valid"]["valid_mae"],
                                  "epoch": info["best_valid"]["epoch"]}
            info["recipe"] = {"epochs": info["epochs_declared"], "epochs_run": info["epochs_run"],
                              "batch_size": info["batch_size"], "updates": info["updates"],
                              "updates_declared": info["epochs_declared"] * info["declared_steps_per_epoch"]}
            info["wall_seconds"] = info["wall_seconds_logged"]
            info["gpu_hours"] = info["gpu_hours_logged"]
            model, _ = ZM.build_model(info["candidate"], D.load_norm(), 0, torch.device("cpu"),
                                      ssl=False)
            info["parameter_count"] = C.parameter_count(model)
            del model
            infos[arm] = info
            truncated.append(arm)
            missing.append(arm)  # no per-molecule valid predictions were saved
        else:
            missing.append(arm)

    table = []
    for name, info in infos.items():
        partial = info.get("status") == "CANCELLED_BY_BUDGET"
        rec = {
            "model": name,
            "params": int(info.get("parameter_count") or 0),
            "valid_mae_best": float(info["fixed_objects"]["best"]["valid_mae"]),
            "best_epoch": int(info["best_valid"]["epoch"]),
            "train_mae_eval_at_best": float(info["fixed_objects"]["best"]["train_mae_eval_mode"]),
            "valid_mae_final": (float("nan") if partial
                                else float(info["fixed_objects"]["final"]["valid_mae"])),
            "train_mae_eval_at_final": (float("nan") if partial
                                        else float(info["fixed_objects"]["final"]["train_mae_eval_mode"])),
            "updates": int(info["recipe"].get("updates", 0)),
            "updates_declared": (int(info["recipe"]["updates_declared"]) if partial
                                 else int(info["recipe"].get("updates", 0))),
            "epochs": int(info["recipe"].get("epochs_run", info["recipe"].get("epochs", 0))),
            "epochs_declared": int(info["recipe"].get("epochs", 0)),
            "batch_size": int(info["recipe"].get("batch_size", 0)),
            "ssl": (
                "P3 masked SSL encoders + native A-SSL pretrain"
                if name == "A_SSL_s0"
                else "P3 masked SSL encoders"
                if name in ("P3B_s0", "L_s0", "C_s0", "A_s0", "B_s0", "A_attn_s0")
                else "none"
            ),
            "status": "CANCELLED_BY_BUDGET" if partial else "completed",
            "wall_seconds": float(info.get("wall_seconds", float("nan"))),
            "gpu_hours_run_level": info.get("gpu_hours"),
            "cancel_reason": info.get("cancel_reason") if partial else None,
            "official_test_loaded": bool(info.get("data", {}).get("official_test_loaded", False)),
        }
        if name == "A_SSL_s0":
            rec["ssl_pretrain"] = info.get("extra", {}).get("ssl_pretrain")
        table.append(rec)
    order = {k: i for i, k in enumerate(("P3B_s0", "L_s0", "C_s0", "A_s0", "B_s0", "A_SSL_s0"))}
    table.sort(key=lambda r: order.get(r["model"], 99))
    C.write_json(ANALYSIS / "main_table.json", {"rows": table, "missing_runs": missing})

    contrasts = {}
    base_pred = {k: preds[k] for k in preds}
    for first in ("A_s0", "B_s0", "A_SSL_s0"):
        if first not in preds:
            continue
        for second in ("P3B_s0", "L_s0"):
            contrasts[f"{first} - {second}"] = paired_bootstrap(
                base_pred[first], base_pred[second], y, seed=BOOT_SEED
            )
        for second in ("A_s0", "B_s0"):
            if second != first and second in preds:
                contrasts[f"{first} - {second}"] = paired_bootstrap(
                    base_pred[first], base_pred[second], y, seed=BOOT_SEED
                )
    C.write_json(ANALYSIS / "paired_contrasts.json", contrasts)

    # horizon-robust comparison: best valid inside a fixed learning-rate band
    band = {}
    curves = {
        "A_s0": RUNS / "A_s0" / "curve.csv",
        "B_s0": RUNS / "B_s0" / "curve.csv",
        "A_SSL_s0": RUNS / "A_SSL_s0" / "curve.csv",
        "A_attn_s0": RUNS / "A_attn_s0" / "curve.csv",
        "P3B_s0": P3B_RUN / "curve.csv",
        "L_s0": DCC / "runs" / "L_s0" / "curve.csv",
        "C_s0": DCC / "runs" / "C_s0" / "curve.csv",
    }
    for name, path in curves.items():
        if not path.exists():
            continue
        rows_c = list(csv.DictReader(open(path, encoding="utf-8")))
        sel = [(int(r["epoch"]), float(r["valid_mae"])) for r in rows_c
               if 1e-4 < float(r["lr"]) <= 3e-4]
        if not sel:
            continue
        best = min(sel, key=lambda t: t[1])
        band[name] = {"band": "(1e-4, 3e-4]", "n_epochs_in_band": len(sel),
                      "best_valid_in_band": best[1], "epoch": best[0],
                      "best_valid_overall": min(float(r["valid_mae"]) for r in rows_c),
                      "epochs_logged": len(rows_c)}
    C.write_json(ANALYSIS / "lr_band.json", band)

    # fixed subgroups from the frozen audit table
    rows_audit = list(csv.DictReader(open(AUDIT / "analysis" / "per_molecule.csv", encoding="utf-8")))
    if len(rows_audit) != n_valid:
        raise RuntimeError("audit per_molecule.csv is not the valid split")
    unseen = np.asarray([float(r["r2_unseen_atom_frac"]) > 0 for r in rows_audit])
    rings4 = np.asarray([int(r["n_rings"]) >= 4 for r in rows_audit])
    tail = y <= -3.0
    worst100 = set(np.argsort(np.abs(preds["P3B_s0"] - y))[-100:].tolist())
    masks = {
        "signature_unseen": unseen,
        "signature_covered": ~unseen,
        "ordinary_y_gt_-3": ~tail,
        "tail_y_le_-3": tail,
        "rings_ge4": rings4,
        "rings_ge4_and_unseen": rings4 & unseen,
        "P3B_worst100": np.asarray([i in worst100 for i in range(n_valid)]),
    }
    subgroup = {}
    names = [k for k in ("P3B_s0", "L_s0", "C_s0", "A_s0", "B_s0", "A_SSL_s0") if k in preds]
    for g, mask in masks.items():
        entry = {"n": int(mask.sum())}
        for k in names:
            entry[k] = float(np.abs(preds[k] - y)[mask].mean())
        for k in ("A_s0", "B_s0", "A_SSL_s0"):
            if k in preds:
                entry[f"{k}_minus_P3B"] = entry[k] - entry["P3B_s0"]
                entry[f"{k}_minus_L"] = entry[k] - entry["L_s0"]
        subgroup[g] = entry
    subgroup["unseen_vs_covered_delta_of_deltas"] = {
        f"{k}_minus_P3B": (subgroup["signature_unseen"][f"{k}_minus_P3B"]
                           - subgroup["signature_covered"][f"{k}_minus_P3B"])
        for k in ("A_s0", "B_s0", "A_SSL_s0") if k in preds
    }
    C.write_json(ANALYSIS / "subgroups.json", subgroup)

    # descriptive observations used by the report (no decisions taken here)
    obs: dict[str, Any] = {"pairs": {}, "size_bins": {}, "bias": {}}
    names_all = [k for k in ("P3B_s0", "L_s0", "C_s0", "A_s0", "B_s0", "A_SSL_s0", "A_attn_s0")
                 if k in preds]
    for a in ("A_s0", "B_s0", "A_SSL_s0", "A_attn_s0"):
        if a not in preds:
            continue
        for b in ("P3B_s0", "L_s0"):
            ea, eb = preds[a] - y, preds[b] - y
            delta = np.abs(ea) - np.abs(eb)
            obs["pairs"][f"{a}_vs_{b}"] = {
                "corr_signed_error": float(np.corrcoef(ea, eb)[0, 1]),
                "corr_abs_error": float(np.corrcoef(np.abs(ea), np.abs(eb))[0, 1]),
                "win_rate_first_better": float((delta < 0).mean()),
                "mean_delta_when_better": float(delta[delta < 0].mean()) if (delta < 0).any() else 0.0,
                "mean_delta_when_worse": float(delta[delta > 0].mean()) if (delta > 0).any() else 0.0,
            }
    n_atoms = np.asarray([int(r["n_atoms"]) for r in rows_audit])
    edges = np.quantile(n_atoms, [0.25, 0.5, 0.75])
    bin_id = np.digitize(n_atoms, edges)
    for q in range(4):
        m = bin_id == q
        if not m.any():
            continue
        obs["size_bins"][f"q{q+1}_n_atoms_{int(n_atoms[m].min())}_{int(n_atoms[m].max())}"] = {
            k: float(np.abs(preds[k] - y)[m].mean()) for k in names_all}
    for k in names_all:
        obs["bias"][k] = {
            "mean_signed_error_all": float((preds[k] - y).mean()),
            "mean_signed_error_tail": float((preds[k] - y)[tail].mean()),
            "mean_signed_error_ordinary": float((preds[k] - y)[~tail].mean()),
            "pred_std": float(preds[k].std()),
        }
    C.write_json(ANALYSIS / "observations.json", obs)

    # per-molecule table for later inspection
    per_path = ANALYSIS / "per_molecule.csv"
    with open(per_path, "w", newline="", encoding="utf-8") as fh:
        fields = ["row_id", "y_raw", "n_atoms", "n_rings", "r2_unseen_atom_frac"] + \
                 [f"pred_{k}" for k in names] + [f"ae_{k}" for k in names]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows_audit):
            out = {"row_id": i, "y_raw": float(y[i]), "n_atoms": r["n_atoms"],
                   "n_rings": r["n_rings"], "r2_unseen_atom_frac": r["r2_unseen_atom_frac"]}
            for k in names:
                out[f"pred_{k}"] = f"{preds[k][i]:.9g}"
                out[f"ae_{k}"] = f"{abs(preds[k][i] - y[i]):.9g}"
            w.writerow(out)

    # curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for arm in ("A_s0", "B_s0", "A_SSL_s0"):
            run = RUNS / arm
            if not (run / "curve.csv").exists():
                continue
            curve = list(csv.DictReader(open(run / "curve.csv", encoding="utf-8")))
            ep = np.asarray([int(e["epoch"]) for e in curve])
            axes[0].plot(ep, [float(e["online_train_mae"]) for e in curve], label=arm)
            axes[1].plot(ep, [float(e["valid_mae"]) for e in curve], label=arm)
            axes[2].plot(ep, [float(e["lr"]) for e in curve], label=arm)
        axes[0].set_title("online train MAE")
        axes[1].set_title("valid MAE")
        axes[1].axhline(float(infos["P3B_s0"]["fixed_objects"]["best"]["valid_mae"]),
                        color="k", ls="--", lw=1, label="P3B best valid")
        axes[1].axhline(float(infos["L_s0"]["fixed_objects"]["best"]["valid_mae"]),
                        color="gray", ls=":", lw=1, label="L best valid")
        axes[2].set_title("learning rate")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ANALYSIS / "curves.png", dpi=130)
        C.log(f"wrote {ANALYSIS / 'curves.png'}")
    except Exception as exc:  # plotting is not a gate
        C.log(f"[warn] curve plot failed: {exc}")

    # markdown main table
    lines = [
        "| model | params | best valid | best ep | eval-train@best | final valid | updates | SSL | status | GPU-h |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for r in table:
        final = "n/a (truncated)" if r["status"] != "completed" else f"{r['valid_mae_final']:.9f}"
        lines.append(
            f"| {r['model']} | {r['params']:,} | {r['valid_mae_best']:.9f} | {r['best_epoch']} | "
            f"{r['train_mae_eval_at_best']:.6f} | {final} | {r['updates']:,} | "
            f"{r['ssl']} | {r['status']} | {r['gpu_hours_run_level']:.4f} |"
        )
    (ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.log("\n".join(lines))
    C.log(json.dumps(contrasts, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
