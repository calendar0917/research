"""Post-hoc summary for ``zinc_local_factorization_v1`` (descriptive only).

Reads the two finished arms (``analysis/LF_D``, ``analysis/LF_F`` and their
``runs/``) plus the frozen reference prediction CSVs, and writes:

    analysis/summary.json      machine-readable main table + contrasts + groups
    analysis/main_table.md     the same table in Markdown
    analysis/groups_valid.json analysis/groups_test.json
    analysis/preds_test.csv    row-level test predictions (D, F and references)
    analysis/curves.png        valid/train curves (if matplotlib is available)

Nothing here selects a checkpoint, an epoch, a rank or a group threshold: the
best-valid checkpoints were frozen by the runner before the single test read.
The paired bootstrap over molecules describes molecular sampling uncertainty
for this one trained pair; it is not a cross-seed repeatability estimate.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_REF = _SS / "zinc_restricted_adaptation_v1" / "analysis"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import lf_core as LC  # noqa: E402

ARMS = {"D": "LF-Dense", "F": "LF-Factorized"}
GROUPS = (
    ("covered_ordinary", 0, 1),
    ("covered_tail", 0, 0),
    ("unseen_ordinary", 1, 1),
    ("unseen_tail", 1, 0),
)
REF_NAMES = ("P3B_s0", "L_s0", "DBT_s0")
ANALYSIS = LC.ANALYSIS


def arm_dirs(arm: str) -> dict[str, Path]:
    return {"run": LC.RUNS / f"LF_{arm}_s0", "analysis": ANALYSIS / f"LF_{arm}"}


def read_reference(split: str) -> dict:
    path = _REF / f"reference_preds_{split}.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    row = np.asarray([int(r["row"]) for r in raw], dtype=np.int64)
    y = np.asarray([float(r["y_raw"]) for r in raw], dtype=np.float64)
    preds = {n: np.asarray([float(r[f"pred_{n}"]) for r in raw], dtype=np.float64)
             for n in REF_NAMES}
    return {"rows": row, "y": y, "refs": preds,
            "unseen": np.asarray([int(r["signature_unseen"]) for r in raw], dtype=np.int64),
            "ordinary": np.asarray([int(r["ordinary_y_gt_minus3"]) for r in raw], dtype=np.int64),
            "csv": str(path.relative_to(C.REPO))}


def aux(pred: np.ndarray, y: np.ndarray) -> dict:
    err = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    return {"mae": float(err.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "p95_ae": float(np.quantile(err, 0.95)), "max_ae": float(err.max()),
            "max_ae_row": int(np.argmax(err)), "n_ae_gt_0.5": int((err > 0.5).sum()),
            "n_ae_gt_1.0": int((err > 1.0).sum())}


def paired_bootstrap(ae_a: np.ndarray, ae_b: np.ndarray, *, reps: int = 10000,
                     seed: int = 20261022) -> dict:
    delta = np.asarray(ae_a, dtype=np.float64) - np.asarray(ae_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(reps, dtype=np.float64)
    done = 0
    while done < reps:
        m = min(2000, reps - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done:done + m] = delta[idx].mean(axis=1)
        done += m
    return {"mean_delta_ae": float(delta.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "fraction_a_better": float((delta < 0).mean()), "reps": int(reps),
            "seed": int(seed),
            "delta_definition": "AE(first arm) - AE(second arm); negative = first arm better",
            "scope": "molecular sampling uncertainty for this trained pair only; "
                     "not a cross-seed repeatability estimate"}


def group_rows(preds: dict[str, np.ndarray], ref: dict) -> tuple[list[dict], dict]:
    y = ref["y"]
    ae = {k: np.abs(np.asarray(v, dtype=np.float64) - y) for k, v in preds.items()}
    rows = []
    for gname, unseen_flag, ordinary_flag in GROUPS:
        mask = (ref["unseen"] == unseen_flag) & (ref["ordinary"] == ordinary_flag)
        item = {"group": gname, "n": int(mask.sum()),
                "mae": {k: float(v[mask].mean()) for k, v in ae.items()}}
        if "F" in ae and "D" in ae:
            item["F_minus_D"] = float(item["mae"]["F"] - item["mae"]["D"])
        rows.append(item)
    return rows, {"overall_mae": {k: float(v.mean()) for k, v in ae.items()},
                  "row_172_kept": True, "analysis_only": True}


def curve_plot(curves: dict[str, list[dict]], path: Path) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return {"plotted": False, "reason": f"{type(exc).__name__}: {exc}"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, curve in curves.items():
        ep = [r["epoch"] for r in curve]
        axes[0].plot(ep, [r["valid_mae"] for r in curve], label=ARMS[arm], lw=1.2)
        axes[1].plot(ep, [r["online_train_mae"] for r in curve], label=ARMS[arm], lw=1.2)
    axes[0].set_title("full valid MAE (every epoch)")
    axes[1].set_title("online train MAE (pre-update)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"plotted": True, "path": str(path.relative_to(C.REPO))}


def main() -> int:
    cfg = LC.load_config()
    checks = C.read_json(ANALYSIS / "cpu_checks.json") if (ANALYSIS / "cpu_checks.json").exists() \
        else {"n_pass": None, "n_fail": None}
    out: dict = {"protocol": LC.PROTOCOL, "config_sha256": cfg["_config_sha256"],
                 "cpu_checks": {"n_pass": checks.get("n_pass"), "n_fail": checks.get("n_fail")},
                 "arms": {}, "references": {}, "groups": {}}

    curves = {}
    preds_valid, preds_test = {}, {}
    for arm in ("D", "F"):
        d = arm_dirs(arm)
        if not (d["run"] / "train.json").exists():
            C.log(f"[report] arm {arm}: no train.json yet; skipping")
            continue
        train = C.read_json(d["run"] / "train.json")
        freeze = C.read_json(d["analysis"] / "freeze.json")
        test = C.read_json(d["analysis"] / "test_report.json")
        curve = train["curve"]
        curves[arm] = curve
        pv = np.load(d["run"] / "predictions" / "valid_best.npz")
        pt = np.load(d["run"] / "predictions" / "test_best.npz")
        preds_valid[arm] = np.asarray(pv["pred"], dtype=np.float64)
        preds_test[arm] = np.asarray(pt["pred"], dtype=np.float64)
        tail = [r["valid_mae"] for r in curve[-100:]]
        out["arms"][arm] = {
            "name": ARMS[arm], "candidate": train["candidate"],
            "params": int(train["parameter_count"]),
            "init": {"ssl": train["gate"]["arms"][arm]["ssl_sha256"],
                     "init_state_sha256": freeze["init_state_sha256"],
                     "public_state_sha256": freeze["public_state_sha256"],
                     "w_s_zero": train["gate"]["arms"][arm]["w_s_zero"]},
            "epochs": int(train["recipe"]["epochs"]),
            "updates": int(train["recipe"]["updates"]),
            "best_valid": float(train["best_valid"]["valid_mae"]),
            "best_epoch": int(train["best_valid"]["epoch"]),
            "eval_train_at_best": float(freeze["best_train_mae_eval_mode"]),
            "final_valid": float(freeze["final_valid_mae"]),
            "final_train_eval": float(freeze["final_train_mae_eval_mode"]),
            "final_online_train_mae": float(curve[-1]["online_train_mae"]),
            "test": test["test"],
            "valid_tail_100": train["valid_tail_100"],
            "intervention": train["intervention"],
            "wall_seconds": float(train["wall_seconds"]),
            "gpu_hours": train["gpu_hours"],
            "steps_per_epoch": int(train["recipe"]["steps_per_epoch"]),
            "steady_epoch_seconds": float(train["timing"]["steady_epoch_seconds"]),
            "best_sha256": freeze["best_sha256"],
            "freeze": freeze, "plan": train["plan"],
            "epoch0_valid_diagnostic": float(train["epoch0_valid_diagnostic"]),
        }

    # declared reference rows (reused, not retrained in this round)
    for name, r in cfg["reference"].items():
        if not isinstance(r, dict) or "parameters" not in r:
            continue
        out["references"][name] = {
            "params": r.get("parameters"), "best_valid": r.get("valid_mae"),
            "test_mae": r.get("test_mae"), "status": "reused (not retrained this round)",
        }
    # enrich with wall/GPU and extra fields from the saved run/analysis records
    for name, run, ana in (
            ("A_joint_BF_s0", _SS / "zinc_structural_object_tokens_v1" / "runs" / "A_joint_BF_s0",
             _SS / "zinc_structural_object_tokens_v1" / "analysis" / "A_joint_BF"),
            ("P3_Control_s0_2026_10_22", _SS / "zinc_p3_sam_v1" / "runs" / "C_control_s0",
             _SS / "zinc_p3_sam_v1" / "analysis" / "P3_Control"),
            ("P3_SAM_s0_2026_10_22", _SS / "zinc_p3_sam_v1" / "runs" / "S_sam_s0",
             _SS / "zinc_p3_sam_v1" / "analysis" / "P3_SAM")):
        rec = out["references"].setdefault(name, {})
        tj = run / "train.json"
        if tj.exists():
            t = C.read_json(tj)
            rec.update({"params": t.get("parameter_count", rec.get("params")),
                        "best_valid": t.get("best_valid", {}).get("valid_mae", rec.get("best_valid")),
                        "best_epoch": t.get("best_valid", {}).get("epoch"),
                        "gpu_hours": t.get("gpu_hours"),
                        "wall_seconds": t.get("wall_seconds"),
                        "run": str(run.relative_to(C.REPO))})
        tr = ana / "test_report.json"
        if tr.exists():
            t = C.read_json(tr)
            rec.setdefault("test_mae", t["test"]["mae"])
            rec["test_rmse"] = t["test"].get("rmse")
        fr = ana / "freeze.json"
        if fr.exists():
            f = C.read_json(fr)
            rec.setdefault("final_valid", f.get("final_valid_mae"))
            rec.setdefault("test_mae", None)
            rec["best_sha256"] = f.get("best_sha256")
    summary = _SS / "zinc_structural_object_tokens_v1" / "analysis" / "summary.json"
    if summary.exists():
        a = C.read_json(summary).get("arms", {}).get("A_joint_BF", {})
        ref = out["references"].get("A_joint_BF_s0", {})
        for key, src in (("eval_train_at_best", "train_eval_at_best"),
                         ("final_valid", "final_valid"), ("test_mae", "test_mae")):
            if a.get(src) is not None:
                ref[key] = a[src]

    # per-epoch curve comparison (same epochs, same order stream)
    if {"D", "F"} <= set(curves):
        dv = {r["epoch"]: r["valid_mae"] for r in curves["D"]}
        fv = {r["epoch"]: r["valid_mae"] for r in curves["F"]}
        shared = sorted(set(dv) & set(fv))
        delta = np.asarray([fv[e] - dv[e] for e in shared])
        tail_mask = np.asarray([e > shared[-1] - 100 for e in shared])
        out["curve_comparison"] = {
            "shared_epochs": len(shared),
            "F_better_epochs": int((delta < 0).sum()),
            "F_worse_epochs": int((delta > 0).sum()),
            "F_minus_D_valid_median": float(np.median(delta)),
            "F_minus_D_valid_mean": float(delta.mean()),
            "F_minus_D_valid_tail100_median": float(np.median(delta[tail_mask])),
            "F_minus_D_valid_tail100_mean": float(delta[tail_mask].mean()),
            "F_better_tail100_epochs": int((delta[tail_mask] < 0).sum()),
            "F_minus_D_online_train_last": float(
                curves["F"][-1]["online_train_mae"] - curves["D"][-1]["online_train_mae"]),
            "order_hashes_identical": all(
                a["order_sha256"] == b["order_sha256"] for a, b in zip(curves["D"], curves["F"])),
        }

    # contrasts (same-round only for F-D)
    contrasts = {}
    for split, preds in (("valid", preds_valid), ("test", preds_test)):
        if not ({"D", "F"} <= set(preds)):
            continue
        y = read_reference(split)["y"]
        ae = {k: np.abs(v - y) for k, v in preds.items()}
        contrasts[split] = {
            "F_minus_D_mae": float(ae["F"].mean() - ae["D"].mean()),
            "paired_bootstrap_F_minus_D": paired_bootstrap(ae["F"], ae["D"]),
            "max_abs_prediction_difference": float(np.abs(preds["F"] - preds["D"]).max()),
            "correlation": float(np.corrcoef(preds["D"], preds["F"])[0, 1]),
        }
    out["contrasts"] = contrasts

    # groups (descriptive)
    for split, preds in (("valid", preds_valid), ("test", preds_test)):
        if not preds:
            continue
        ref = read_reference(split)
        combined = dict(preds)
        combined.update({n: ref["refs"][n] for n in REF_NAMES})
        rows, meta = group_rows(combined, ref)
        out["groups"][split] = {"rows": rows, "meta": meta,
                               "reference_csv": ref["csv"]}
        C.write_json(ANALYSIS / f"groups_{split}.json", out["groups"][split])

    # row-level test predictions
    if "test" in out["groups"]:
        ref = read_reference("test")
        path = ANALYSIS / "preds_test.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "y_raw", "pred_D", "pred_F", *[f"pred_{n}" for n in REF_NAMES],
                        "signature_unseen", "ordinary_y_gt_minus3"])
            d = preds_test.get("D", np.full(1000, np.nan))
            f = preds_test.get("F", np.full(1000, np.nan))
            for i in range(1000):
                w.writerow([i, f"{ref['y'][i]:.9f}", f"{d[i]:.9f}", f"{f[i]:.9f}",
                            *[f"{ref['refs'][n][i]:.9f}" for n in REF_NAMES],
                            int(ref["unseen"][i]), int(ref["ordinary"][i])])
        out["preds_test_csv"] = str(path.relative_to(C.REPO))

    out["curve_plot"] = curve_plot(curves, ANALYSIS / "curves.png") if curves else {"plotted": False}
    C.write_json(ANALYSIS / "summary.json", out)

    # markdown main table
    lines = ["| model | params | init | E / updates | best valid | ep | train@best | final valid |"
             " test MAE | test RMSE | p95 AE | max AE | GPU-h | wall s |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, a in out["arms"].items():
        lines.append(
            f"| **{arm} {a['name']}** | {a['params']:,} | SSL-start P3 | {a['epochs']} / "
            f"{a['updates']:,} | {a['best_valid']:.9f} | {a['best_epoch']} | "
            f"{a['eval_train_at_best']:.6f} | {a['final_valid']:.9f} | "
            f"{a['test']['mae']:.9f} | {a['test']['rmse']:.6f} | {a['test']['p95_ae']:.6f} | "
            f"{a['test']['max_ae']:.6f} (row {a['test']['max_ae_row']}) | "
            f"{a['gpu_hours']:.3f} | {a['wall_seconds']:.0f} |")
    for name, r in out["references"].items():
        lines.append(f"| ref {name} | {r.get('params')} | reused | - | {r.get('best_valid')} | "
                     f"{r.get('best_epoch')} | {r.get('eval_train_at_best')} | "
                     f"{r.get('final_valid')} | {r.get('test_mae')} | - | - | - | "
                     f"{r.get('gpu_hours')} | {r.get('wall_seconds')} |")
    (ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.log("\n".join(lines))
    C.log(f"[report] wrote {ANALYSIS / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
