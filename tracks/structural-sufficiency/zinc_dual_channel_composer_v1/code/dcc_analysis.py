"""Valid-only analysis for the dual-channel composer (P3B / L / C).

All inputs are the fixed official-valid row order (0..999) and already saved
best-checkpoint predictions.  No official test path is referenced.  Paired
bootstraps are descriptive fixed-checkpoint intervals on an exposed
development split, not cross-seed confirmation.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SUFF = _HERE.parents[1]
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_SUFF / "zinc_arch_discovery_v1" / "code"), str(_SUFF / "zinc_recipe_trunk_v2" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
AUDIT = _SUFF / "zinc_generalization_audit_v1"
P3B_RUN = _SUFF / "zinc_recipe_trunk_v2" / "runs" / "xBssl_P3N_s0"
RUNS = {"L": EXP / "runs" / "L_s0", "C": EXP / "runs" / "C_s0"}
MODELS = ("P3B_s0", "L_s0", "C_s0")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz_pred(path: Path, expected_rows: np.ndarray) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    pred = np.asarray(z["pred"], dtype=np.float64).reshape(-1)
    row = np.asarray(z["row"], dtype=np.int64).reshape(-1)
    if not np.array_equal(row, expected_rows):
        raise RuntimeError(f"{path}: row ids are not the fixed valid order")
    if not np.isfinite(pred).all():
        raise RuntimeError(f"{path}: non-finite predictions")
    return pred


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    e = pred - y
    ae = np.abs(e)
    order = np.sort(ae)[::-1]
    total = float(ae.sum())
    return {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "max_ae": float(ae.max()),
        "mean_signed_error": float(e.mean()),
        "positive_error_fraction": float(np.mean(e > 0)),
        "top1_abs_error_share": float(order[:1].sum() / total),
        "top10_abs_error_share": float(order[:10].sum() / total),
        "top50_abs_error_share": float(order[:50].sum() / total),
        "top100_abs_error_share": float(order[:100].sum() / total),
    }


def bootstrap_delta(delta: np.ndarray, idx: np.ndarray) -> dict:
    means = delta[idx].mean(axis=1)
    return {
        "mean_delta_mae": float(delta.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def group_row(mask: np.ndarray, y: np.ndarray, pred: dict) -> dict:
    ae = {k: np.abs(pred[k] - y) for k in MODELS}
    out = {"n": int(mask.sum())}
    for k in MODELS:
        out[f"mae_{k}"] = float(ae[k][mask].mean())
    out["delta_L_minus_P3B"] = float((ae["L_s0"] - ae["P3B_s0"])[mask].mean())
    out["delta_C_minus_P3B"] = float((ae["C_s0"] - ae["P3B_s0"])[mask].mean())
    out["delta_L_minus_C"] = float((ae["L_s0"] - ae["C_s0"])[mask].mean())
    return out


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(AUDIT / "analysis" / "per_molecule.csv", encoding="utf-8")))
    n = len(rows)
    row_ids = np.asarray([int(r["row_id"]) for r in rows], dtype=np.int64)
    if not np.array_equal(row_ids, np.arange(n)):
        raise RuntimeError("per_molecule.csv is not in fixed row-id order")
    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    expected_rows = np.arange(n, dtype=np.int64)

    p3b_info = read_json(P3B_RUN / "train.json")
    p3b_pred = load_npz_pred(P3B_RUN / "predictions" / "valid_best.npz", expected_rows)
    pred = {"P3B_s0": p3b_pred}
    infos = {"P3B_s0": p3b_info}
    for arm in ("L", "C"):
        info = read_json(RUNS[arm] / "train.json")
        infos[f"{arm}_s0"] = info
        pred[f"{arm}_s0"] = load_npz_pred(RUNS[arm] / "predictions" / "valid_best.npz", expected_rows)
        recomputed = float(np.abs(pred[f"{arm}_s0"] - y).mean())
        stored = float(info["fixed_objects"]["best"]["valid_mae"])
        if abs(recomputed - stored) > 1e-5:
            raise RuntimeError(f"{arm}: recomputed MAE {recomputed} != stored {stored}")

    # Per-molecule table.
    per_path = ANALYSIS / "per_molecule_dual.csv"
    fields = [
        "row_id", "y_raw", "n_atoms", "n_rings", "cycle_rank", "max_ring_size",
        "r2_unseen_atom_frac", "j_unseen_mass",
        "pred_P3B_s0", "pred_L_s0", "pred_C_s0",
        "signed_error_P3B_s0", "signed_error_L_s0", "signed_error_C_s0",
        "abs_error_P3B_s0", "abs_error_L_s0", "abs_error_C_s0",
        "P3B_worst100",
    ]
    ae = {k: np.abs(pred[k] - y) for k in MODELS}
    worst100 = set(np.argsort(ae["P3B_s0"])[-100:].tolist())
    with open(per_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            out = {
                "row_id": i,
                "y_raw": r["y_raw"],
                "n_atoms": r["n_atoms"],
                "n_rings": r["n_rings"],
                "cycle_rank": r["cycle_rank"],
                "max_ring_size": r["max_ring_size"],
                "r2_unseen_atom_frac": r["r2_unseen_atom_frac"],
                "j_unseen_mass": r["j_unseen_mass"],
                "P3B_worst100": int(i in worst100),
            }
            for k in MODELS:
                out[f"pred_{k}"] = f"{pred[k][i]:.9g}"
                out[f"signed_error_{k}"] = f"{pred[k][i] - y[i]:.9g}"
                out[f"abs_error_{k}"] = f"{ae[k][i]:.9g}"
            w.writerow(out)

    # Main metrics and fixed table.
    model_metrics = {k: metrics(pred[k], y) for k in MODELS}
    C.write_json(ANALYSIS / "model_metrics.json", model_metrics)

    def curve_point(info, epoch):
        for e in info["curve"]:
            if int(e["epoch"]) == int(epoch):
                return e
        raise KeyError(epoch)

    main_rows = []
    ledger = read_json(ANALYSIS / "budget_ledger.json") if (ANALYSIS / "budget_ledger.json").exists() else {}
    for k in MODELS:
        info = infos[k]
        best_epoch = int(info["fixed_objects"]["best"]["epoch"])
        final_epoch = int(info["fixed_objects"]["final"]["epoch"])
        best_curve = curve_point(info, best_epoch)
        final_curve = curve_point(info, final_epoch)
        gpu_hours = None
        gpu_source = None
        if k == "P3B_s0":
            gpu_hours = float(info.get("gpu_hours") or info["wall_seconds"] / 3600.0)
            gpu_source = "historical run-level wall"
        else:
            arm_jobs = ledger.get("runs", {}).get(k, {})
            if "job_occupancy_hours" in arm_jobs:
                gpu_hours = float(arm_jobs["job_occupancy_hours"])
                gpu_source = "rr job occupancy"
            else:
                gpu_hours = float(info.get("gpu_hours") or info["wall_seconds"] / 3600.0)
                gpu_source = "run-level wall"
        main_rows.append({
            "model": k,
            "parameters": int(info["parameter_count"]),
            "best_valid": float(info["fixed_objects"]["best"]["valid_mae"]),
            "best_epoch": best_epoch,
            "final_valid": float(info["fixed_objects"]["final"]["valid_mae"]),
            "best_train_eval_mode": float(info["fixed_objects"]["best"]["train_mae_eval_mode"]),
            "final_train_eval_mode": float(info["fixed_objects"]["final"]["train_mae_eval_mode"]),
            "best_online_train": float(best_curve["online_train_mae"]),
            "final_online_train": float(final_curve["online_train_mae"]),
            "gpu_hours": gpu_hours,
            "gpu_hours_source": gpu_source,
        })
    C.write_json(ANALYSIS / "main_table.json", {"rows": main_rows, "valid_role": "exposed development split; official test only reporting-only after the valid decision was frozen"})

    # Paired bootstrap with one fixed set of row resamples for all contrasts.
    rng = np.random.default_rng(20261021)
    boot_idx = rng.integers(0, n, size=(10000, n))
    contrasts = {}
    for a, b in (("L_s0", "P3B_s0"), ("C_s0", "P3B_s0"), ("L_s0", "C_s0")):
        contrasts[f"{a}_minus_{b}"] = bootstrap_delta(ae[a] - ae[b], boot_idx)
    C.write_json(ANALYSIS / "paired_contrasts.json", {
        "resamples": 10000,
        "rng_seed": 20261021,
        "same_row_resamples_for_all_contrasts": True,
        "delta_definition": "abs_error(first) - abs_error(second); negative means the first model is better",
        "interval_role": "descriptive fixed-checkpoint interval on the exposed official-valid 1000; not cross-seed confirmation",
        "contrasts": contrasts,
    })

    # Fixed subgroups (row masks fixed before seeing L/C results).
    unseen = np.asarray([float(r["r2_unseen_atom_frac"]) > 0 for r in rows])
    ordinary = y > -3
    rings4 = np.asarray([int(r["n_rings"]) >= 4 for r in rows])
    worst100_mask = np.zeros(n, dtype=bool)
    worst100_mask[list(worst100)] = True
    groups = {
        "signature_unseen_251": unseen,
        "signature_covered_749": ~unseen,
        "ordinary_y_gt_minus3": ordinary,
        "tail_y_le_minus3": ~ordinary,
        "rings_ge4": rings4,
        "rings_ge4_and_signature_unseen": rings4 & unseen,
        "fixed_P3B_worst100": worst100_mask,
    }
    subgroup = {name: group_row(mask, y, pred) for name, mask in groups.items()}
    subgroup["unseen_minus_covered_delta_of_deltas"] = {
        "L_minus_P3B": subgroup["signature_unseen_251"]["delta_L_minus_P3B"] - subgroup["signature_covered_749"]["delta_L_minus_P3B"],
        "C_minus_P3B": subgroup["signature_unseen_251"]["delta_C_minus_P3B"] - subgroup["signature_covered_749"]["delta_C_minus_P3B"],
        "L_minus_C": subgroup["signature_unseen_251"]["delta_L_minus_C"] - subgroup["signature_covered_749"]["delta_L_minus_C"],
        "definition": "mean paired delta inside unseen minus mean paired delta inside covered; negative means the first model's advantage is larger in the unseen group",
    }
    C.write_json(ANALYSIS / "subgroups.json", {
        "groups": subgroup,
        "note": "Fixed P3B worst-100 absolute MAE is the hard-case monitor; top-100 share is descriptive only and is not a gate.",
    })

    # Decision.
    d_l_p3b = contrasts["L_s0_minus_P3B_s0"]["mean_delta_mae"]
    d_c_p3b = contrasts["C_s0_minus_P3B_s0"]["mean_delta_mae"]
    d_l_c = contrasts["L_s0_minus_C_s0"]["mean_delta_mae"]
    perf_l = d_l_p3b <= -0.003
    perf_c = d_c_p3b <= -0.003
    ownership = d_l_c <= -0.002
    if perf_l and ownership:
        status = "A_DUAL_CHANNEL_LOCAL_OWNERSHIP_GAIN"
    elif perf_c or perf_l:
        status = "B_DUAL_CHANNEL_PERFORMANCE_SIGNAL_NO_OWNERSHIP"
    elif ownership:
        status = "C_OWNERSHIP_ONLY_NO_PERFORMANCE"
    else:
        status = "D_DUAL_CHANNEL_NO_GAIN"

    ordinary_l = subgroup["ordinary_y_gt_minus3"]["delta_L_minus_P3B"]
    ordinary_c = subgroup["ordinary_y_gt_minus3"]["delta_C_minus_P3B"]
    worst100_l = subgroup["fixed_P3B_worst100"]["delta_L_minus_P3B"]
    worst100_c = subgroup["fixed_P3B_worst100"]["delta_C_minus_P3B"]
    tradeoffs = []
    for arm, ordinary_delta, worst_delta in (("L", ordinary_l, worst100_l), ("C", ordinary_c, worst100_c)):
        if ordinary_delta > 0.002 or worst_delta > 0.0:
            tradeoffs.append({
                "arm": arm,
                "ordinary_y_gt_minus3_delta_vs_P3B": ordinary_delta,
                "fixed_P3B_worst100_abs_mae_delta_vs_P3B": worst_delta,
            })

    def strength(delta, ownership=False):
        if ownership and delta <= -0.002:
            return "ownership_signal_ge_0.002_descriptive"
        if delta <= -0.003:
            return "performance_signal_ge_0.003"
        if delta < 0.0:
            return "weak_positive_lt_0.003"
        if delta == 0.0:
            return "zero"
        return "negative"

    decision = {
        "protocol_id": "zinc-dual-channel-composer-v1",
        "status": status,
        "official_test_loaded": True,
        "official_test_loaded_before_valid_decision": False,
        "valid_is_exposed_development_split": True,
        "single_seed": 0,
        "budget_note": "See analysis/budget_ledger.json. Actual job occupancy was 0.7703 A100-hours (timing 0.0189 + formal 0.7514); the pre-run conservative projection would have stopped, and the user then explicitly instructed to ignore the GPU budget and run for results.",
        "test_access": {
            "official_test_loaded": True,
            "test_requested_by_user_after_run": True,
            "authorized_after_valid_decision": True,
            "test_read": True,
            "test_read_count": 3,
            "prereg": "TEST_PREREG.md",
            "amendment": "TEST_AMENDMENT.md",
            "report": "TEST_REPORT.md",
            "test_used_for_selection": False,
            "reason": "The user explicitly authorized a reporting-only test read after the valid-only decision was frozen. The test split is spent and not pristine; test numbers are descriptive and did not change any model, epoch, recipe, or decision.",
        },
        "main_contrasts": {
            "L_minus_P3B": {"mean_delta_mae": d_l_p3b, "ci95": contrasts["L_s0_minus_P3B_s0"]["ci95"], "signal": strength(d_l_p3b)},
            "C_minus_P3B": {"mean_delta_mae": d_c_p3b, "ci95": contrasts["C_s0_minus_P3B_s0"]["ci95"], "signal": strength(d_c_p3b)},
            "L_minus_C": {"mean_delta_mae": d_l_c, "ci95": contrasts["L_s0_minus_C_s0"]["ci95"], "signal": strength(d_l_c, ownership=True)},
        },
        "performance_signal": {
            "L_vs_P3B": bool(perf_l),
            "C_vs_P3B": bool(perf_c),
            "threshold": -0.003,
            "note": "threshold is a development signal, not statistical significance or SOTA",
        },
        "ownership_signal": {
            "L_minus_C_le_minus_0.002": bool(ownership),
            "threshold": -0.002,
        },
        "safety": {
            "ordinary_y_gt_minus3_delta_L_vs_P3B": ordinary_l,
            "ordinary_y_gt_minus3_delta_C_vs_P3B": ordinary_c,
            "fixed_P3B_worst100_delta_L_vs_P3B": worst100_l,
            "fixed_P3B_worst100_delta_C_vs_P3B": worst100_c,
            "tradeoffs": tradeoffs,
            "top100_share_is_descriptive_only": True,
        },
        "unseen_vs_covered_delta_of_deltas": subgroup["unseen_minus_covered_delta_of_deltas"],
        "interpretation_constraints": [
            "Single seed on an exposed valid split; no cross-seed confirmation.",
            "A gain does not prove Null pooling is the whole generalization cause.",
            "A null/negative result does not imply unseen environments cannot generalize or that all local composition is useless.",
            "Do not route by subgroup or select epochs by subgroup.",
        ],
        "next_round_cost_if_confirming": "Not recommended for this negative-performance result. If ownership alone were to be confirmed, P3B seed1 + L seed1 + C seed1 are all required; using this round's occupancy about 0.30 + 0.35 + 0.40 = 1.05 A100-hours plus timing/margin.",
    }
    C.write_json(ANALYSIS / "decision.json", decision)

    # Curves.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        colors = {"P3B_s0": "#718096", "L_s0": "#2b6cb0", "C_s0": "#dd6b20"}
        for k, run in (("P3B_s0", P3B_RUN), ("L_s0", RUNS["L"]), ("C_s0", RUNS["C"])):
            csv_rows = list(csv.DictReader(open(run / "curve.csv", encoding="utf-8")))
            ep = np.asarray([int(r["epoch"]) for r in csv_rows])
            valid = np.asarray([float(r["valid_mae"]) for r in csv_rows])
            online = np.asarray([float(r["online_train_mae"]) for r in csv_rows])
            axes[0].plot(ep, valid, color=colors[k], lw=1.3, label=k)
            axes[1].plot(ep, online, color=colors[k], lw=1.0, label=k)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("valid MAE (raw y)")
        axes[0].set_title("Full valid every epoch")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("online train MAE")
        axes[1].set_title("Within-epoch pre-update train MAE")
        for ax in axes:
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(ANALYSIS / "curves.png", dpi=180)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is not a result gate
        print(f"[warn] curve plot skipped: {exc}")

    # Artifact hashes and model manifest.
    artifacts = {}
    for k, run in (("P3B_s0", P3B_RUN), ("L_s0", RUNS["L"]), ("C_s0", RUNS["C"])):
        artifacts[k] = {
            "run_dir": str(run.relative_to(_REPO)),
            "best_pt_sha256": C.sha256_file(run / "best.pt"),
            "final_pt_sha256": C.sha256_file(run / "final.pt"),
            "valid_best_npz_sha256": C.sha256_file(run / "predictions" / "valid_best.npz"),
            "curve_csv_sha256": C.sha256_file(run / "curve.csv"),
        }
        if k != "P3B_s0":
            art = read_json(run / "artifacts.json")
            artifacts[k]["initial_state_sha256"] = art.get("initial_state_sha256")
            artifacts[k]["train_json_sha256"] = art["sha256"]["train_json"]
    C.write_json(ANALYSIS / "artifact_hashes.json", artifacts)

    # Compact markdown for the report.
    lines = [
        "| model | params | best valid | best ep | final valid | best train (eval) | final train (eval) | best online | final online | GPU-h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in main_rows:
        lines.append(
            f"| {r['model']} | {r['parameters']:,} | {r['best_valid']:.9f} | {r['best_epoch']} | "
            f"{r['final_valid']:.9f} | {r['best_train_eval_mode']:.6f} | {r['final_train_eval_mode']:.6f} | "
            f"{r['best_online_train']:.6f} | {r['final_online_train']:.6f} | {r['gpu_hours']:.4f} |"
        )
    (ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.log("\n".join(lines))
    C.log(json.dumps({"status": status, "L-P3B": d_l_p3b, "C-P3B": d_c_p3b, "L-C": d_l_c}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
