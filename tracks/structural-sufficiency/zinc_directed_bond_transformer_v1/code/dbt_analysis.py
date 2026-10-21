"""Curves and the single main table for the DBT round.

Historical rows (P3B_s0, L_s0, A_s0, A_SSL_s0, A_attn_s0) are read from the
machine-written results of the earlier rounds; nothing is retyped by hand and
nothing is retrained.  The new row comes from this round's ``train.json`` and
``analysis/test_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

ANALYSIS = _EXP / "analysis"
PREV = _SSUF / "zinc_arch_exploration_v3" / "analysis"
HISTORY_ROWS = ("P3B_s0", "L_s0", "A_s0", "A_SSL_s0", "A_attn_s0")


def history() -> list[dict[str, Any]]:
    main = C.read_json(PREV / "main_table.json")["rows"]
    test = C.read_json(PREV / "test_reporting.json")["models"]
    by_name = {r["model"]: r for r in main}
    out = []
    for name in HISTORY_ROWS:
        r = dict(by_name[name])
        t = test[name]
        out.append({
            "model": name,
            "params": int(r["params"]),
            "valid_mae": float(r["valid_mae_best"]),
            "best_epoch": int(r["best_epoch"]),
            "epochs_declared": int(r["epochs_declared"]),
            "updates": int(r["updates"]),
            "train_mae_at_best": float(r["train_mae_eval_at_best"]),
            "test_mae": float(t["mae"]),
            "test_rmse": float(t["rmse"]),
            "gpu_hours": float(r["gpu_hours_run_level"]),
            "status": str(r["status"]),
            "source": "reused from zinc_arch_exploration_v3/analysis",
        })
    return out


def plot(curve: list[Mapping[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = np.asarray([c["epoch"] for c in curve])
    tr = np.asarray([c["online_train_mae"] for c in curve])
    va = np.asarray([c["valid_mae"] for c in curve])
    lr = np.asarray([c["lr"] for c in curve])
    best = int(np.argmin(va))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, tr, lw=1.0, label="online train MAE")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch")
    ax[0].set_title("train (online, pre-update)")
    ax[0].legend()
    ax[1].plot(ep, va, lw=1.0, color="tab:orange")
    ax[1].axvline(ep[best], color="k", ls="--", lw=0.8)
    ax[1].plot([ep[best]], [va[best]], "k.", ms=8)
    ax[1].set_ylim(max(0.0, va.min() - 0.02), min(0.2, va.max()))
    ax[1].set_xlabel("epoch")
    ax[1].set_title(f"valid MAE (best {va[best]:.6f} @ {ep[best]})")
    ax[2].plot(ep, lr, lw=1.0, color="tab:green")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("epoch")
    ax[2].set_title("learning rate (per update)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


#: frozen official-test predictions of the two primary references (the same
#: 1,000 rows); used only for a descriptive paired bootstrap.
REF_PREDS = {
    "P3B_s0": _SSUF / "zinc_dual_channel_composer_v1/analysis/test_preds/P3B_s0_test.npz",
    "L_s0": _SSUF / "zinc_dual_channel_composer_v1/analysis/test_preds/L_s0_test.npz",
}
REF_VALID = {
    "P3B_s0": _SSUF / "zinc_recipe_trunk_v2/runs/xBssl_P3N_s0/predictions/valid_best.npz",
    "L_s0": _SSUF / "zinc_dual_channel_composer_v1/runs/L_s0/predictions/valid_best.npz",
}


def paired_test_contrasts(run: Path) -> dict[str, Any]:
    path = run / "predictions" / "test_best.npz"
    if not path.exists():
        return {"available": False}
    mine = np.load(path)
    pred = np.asarray(mine["pred"], np.float64)
    y = np.asarray(mine["y"], np.float64)
    out: dict[str, Any] = {"available": True, "n": int(len(y)), "rows": {}}
    for name, ref_path in REF_PREDS.items():
        if not ref_path.exists():
            continue
        ref = np.load(ref_path)
        if not np.array_equal(np.asarray(ref["row"]), np.asarray(mine["row"])):
            out["rows"][name] = {"error": "row mismatch"}
            continue
        rp = np.asarray(ref["pred"], np.float64)
        ry = np.asarray(ref["y"], np.float64)
        ae_mine, ae_ref = np.abs(pred - y), np.abs(rp - ry)
        delta = ae_mine - ae_ref
        rng = np.random.default_rng(20261022)
        idx = rng.integers(0, len(y), size=(10000, len(y)))
        boot = delta[idx].mean(1)
        out["rows"][name] = {
            "mae_mine": float(ae_mine.mean()),
            "mae_ref": float(ae_ref.mean()),
            "mean_delta": float(delta.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "my_better_fraction": float((delta < 0).mean()),
            "resamples": 10000,
            "seed": 20261022,
        }
    return out


def paired_valid_contrasts(run: Path) -> dict[str, Any]:
    path = run / "predictions" / "valid_best.npz"
    if not path.exists():
        return {"available": False}
    import dbt_data as BD

    mine = np.load(path)
    y = np.asarray(BD.load_object_cache("valid")["labels"], np.float64)
    pred = np.asarray(mine["pred"], np.float64)
    ae = np.abs(pred - y)
    worst = int(ae.argmax())
    out: dict[str, Any] = {
        "available": True, "n": int(len(y)),
        "mae": float(ae.mean()),
        "worst_ae": float(ae[worst]), "worst_row": worst,
        "worst_share_of_mae": float(ae[worst] / ae.sum()),
        "mae_excluding_worst": float(np.delete(ae, worst).mean()),
        "n_ae_gt_1": int((ae > 1).sum()),
        "rows": {},
    }
    for name, ref_path in REF_VALID.items():
        if not ref_path.exists():
            continue
        ref = np.load(ref_path)
        rp = np.asarray(ref["pred"], np.float64)
        ae_ref = np.abs(rp - y)
        delta = ae - ae_ref
        rng = np.random.default_rng(20261022)
        idx = rng.integers(0, len(y), size=(10000, len(y)))
        boot = delta[idx].mean(1)
        out["rows"][name] = {
            "mae_ref": float(ae_ref.mean()),
            "mean_delta": float(delta.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "my_better_fraction": float((delta < 0).mean()),
            "ref_ae_at_worst_row": float(ae_ref[worst]),
            "mean_delta_excluding_worst": float(np.delete(delta, worst).mean()),
            "resamples": 10000, "seed": 20261022,
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(_EXP / "runs" / "dbt_s0"))
    args = p.parse_args(argv)
    run = Path(args.run)
    info = C.read_json(run / "train.json")
    curve = info["curve"]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    plot(curve, ANALYSIS / "curves.png")
    C.write_curve_csv(ANALYSIS / "curve.csv", curve)

    test = None
    if (ANALYSIS / "test_report.json").exists():
        test = C.read_json(ANALYSIS / "test_report.json")

    new_row = {
        "model": "DBT_s0",
        "params": int(info["parameter_audit"]["parameters"]),
        "valid_mae": float(info["fixed_objects"]["best"]["valid_mae"]),
        "best_epoch": int(info["fixed_objects"]["best"]["epoch"]),
        "epochs_declared": int(info["recipe"]["epochs"]),
        "updates": int(info["recipe"]["actual_global_steps"]),
        "train_mae_at_best": float(info["fixed_objects"]["best"]["train_mae_eval_mode"]),
        "valid_mae_final": float(info["fixed_objects"]["final"]["valid_mae"]),
        "train_mae_final": float(info["fixed_objects"]["final"]["train_mae_eval_mode"]),
        "test_mae": None if test is None else float(test["test"]["mae"]),
        "test_rmse": None if test is None else float(test["test"]["rmse"]),
        "gpu_hours": float(info["gpu_hours"] or 0.0),
        "wall_seconds": float(info["wall_seconds"]),
        "status": "completed" if info["recipe"]["actual_global_steps"] == info["recipe"]["updates"]
        else "partial",
        "source": "this round",
    }
    rows = history() + [new_row]

    # curve facts, no decision attached
    va = np.asarray([c["valid_mae"] for c in curve])
    ep = np.asarray([c["epoch"] for c in curve])
    lr = np.asarray([c["lr"] for c in curve])
    best = int(np.argmin(va))
    band = lr > 1.0e-4
    band_best = int(np.argmin(np.where(band, va, np.inf)))
    observations = {
        "best_epoch": int(ep[best]),
        "best_valid": float(va[best]),
        "lr_at_best": float(lr[best]),
        "valid_at_epoch_100": float(va[min(99, len(va) - 1)]),
        "valid_at_epoch_300": float(va[min(299, len(va) - 1)]),
        "valid_final": float(va[-1]),
        "valid_min_after_best": float(va[best:].min()),
        "best_in_lr_band_1e-4": {"epoch": int(ep[band_best]), "valid": float(va[band_best])},
        "epochs_run": int(ep[-1]),
        "mean_epoch_seconds": float(np.mean([c["seconds"] for c in curve])),
    }
    contrasts = paired_test_contrasts(run)
    valid_contrasts = paired_valid_contrasts(run)
    C.write_json(ANALYSIS / "test_contrasts.json", contrasts)
    C.write_json(ANALYSIS / "valid_contrasts.json", valid_contrasts)
    C.write_json(ANALYSIS / "observations.json", observations)
    C.write_json(ANALYSIS / "main_table.json", {"rows": rows, "observations": observations})

    def fmt(v, nd=9):
        return "n/a" if v is None else f"{v:.{nd}f}"

    lines = [
        "# DBT — main table (valid = development/selection, test = descriptive)",
        "",
        "| model | params | best valid | best ep | train@best | test MAE | updates | status | GPU-h |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['params']:,} | {fmt(r['valid_mae'])} | {r['best_epoch']} | "
            f"{fmt(r['train_mae_at_best'], 6)} | {fmt(r['test_mae'])} | {r['updates']:,} | "
            f"{r['status']} | {r['gpu_hours']:.2f} |"
        )
    lines += [
        "",
        "Historical rows are reused unchanged from "
        "`zinc_arch_exploration_v3/analysis/{main_table,test_reporting}.json`; they are not "
        "retrained and their test numbers are not recomputed.  The new row's test MAE is one "
        "descriptive read of its frozen best-valid checkpoint.",
        "",
        f"Best epoch {observations['best_epoch']}, lr at best {observations['lr_at_best']:.3e}, "
        f"valid in the `lr>1e-4` band {observations['best_in_lr_band_1e-4']['valid']:.9f} "
        f"@{observations['best_in_lr_band_1e-4']['epoch']}, "
        f"mean epoch {observations['mean_epoch_seconds']:.2f}s.",
        "",
    ]
    if test is not None:
        lines += [
            f"Test − valid for DBT: {test['test']['mae'] - test['valid']['mae']:+.6f} "
            f"(test RMSE {test['test']['rmse']:.4f}, median AE {test['test']['median_ae']:.4f}, "
            f"p95 AE {test['test']['p95_ae']:.4f}).",
            "",
        ]
    if valid_contrasts.get("available"):
        lines += [
            f"Valid worst molecule: row {valid_contrasts['worst_row']} with AE "
            f"{valid_contrasts['worst_ae']:.4f} = "
            f"{100*valid_contrasts['worst_share_of_mae']:.1f} % of the valid MAE "
            f"(valid MAE without it {valid_contrasts['mae_excluding_worst']:.9f}; "
            f"{valid_contrasts['n_ae_gt_1']} molecules have AE > 1).",
            "",
        ]
        if valid_contrasts["rows"]:
            lines += ["Descriptive paired valid bootstrap (same resamples):", "",
                      "| ref | MAE ref | mean Δ | 95 % CI | DBT better on | ref AE at worst row | mean Δ without worst |",
                      "|---|---:|---:|---|---:|---:|---:|"]
            for name, r in valid_contrasts["rows"].items():
                lines.append(
                    f"| {name} | {r['mae_ref']:.9f} | {r['mean_delta']:+.6f} | "
                    f"[{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] | "
                    f"{r['my_better_fraction']*100:.1f} % | {r['ref_ae_at_worst_row']:.4f} | "
                    f"{r['mean_delta_excluding_worst']:+.6f} |")
            lines += ["", "Descriptive only; the worst molecule is not used for selection or as a headline.", ""]
    if contrasts.get("available"):
        lines += ["Descriptive paired test bootstrap (10,000 resamples, Δ = AE(DBT) − AE(ref)):", "",
                  "| ref | MAE ref | MAE DBT | mean Δ | 95 % CI | DBT better on |",
                  "|---|---:|---:|---:|---|---:|"]
        for name, r in contrasts["rows"].items():
            if "error" in r:
                continue
            lines.append(
                f"| {name} | {r['mae_ref']:.9f} | {r['mae_mine']:.9f} | {r['mean_delta']:+.6f} | "
                f"[{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] | {r['my_better_fraction']*100:.1f} % |"
            )
        lines += ["", "Single seed; the intervals are descriptive, not confirmatory.", ""]
    (ANALYSIS / "main_table.md").write_text("\n".join(lines), encoding="utf-8")
    C.log("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
