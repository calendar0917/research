"""Aggregate P0/P1 results, apply the frozen stop rule, and write machine-readable facts.

This script does not read Test data.  It only consumes the local run artifacts
created by the formal P1 jobs (Validation-only) and the P0 JSON files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

import pc_common as C

RUN_NAMES = {
    "p1a": "p1a_480_s0",
    "A0": "a0_s0",
    "A1": "a1_s0",
    "A2": "a2_s0",
    "A3": "a3_s0",
}
CONFIRM_NAMES = {
    (arm, seed): f"{arm.lower()}_s{seed}" for arm in ("A0", "A2", "A3") for seed in (1, 2)
}
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20_261_104


def read_run(name: str) -> dict[str, Any] | None:
    path = C.RUNS / name / "run.json"
    if not path.is_file():
        return None
    run = C.read_json(path)
    run["_dir"] = str(C.RUNS / name)
    curve_path = C.RUNS / name / "curve.csv"
    if curve_path.is_file():
        with open(curve_path, "r", encoding="utf-8") as fh:
            run["_curve"] = list(csv.DictReader(fh))
    else:
        run["_curve"] = []
    run["_predictions_path"] = str(C.RUNS / name / "predictions.npz")
    return run


def macro(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def load_pred(name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(C.RUNS / name / "predictions.npz") as z:
        return (
            np.asarray(z["y_valid"], dtype=np.float64),
            np.asarray(z["valid_probs"], dtype=np.float64),
        )


def paired_bootstrap(
    name_a: str,
    name_b: str,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired Validation-row bootstrap for macro AP(a) - macro AP(b)."""
    y, pa = load_pred(name_a)
    y2, pb = load_pred(name_b)
    if not np.array_equal(y, y2):
        raise RuntimeError(f"Validation labels differ for {name_a} / {name_b}")
    n = y.shape[0]
    point = macro(y, pa) - macro(y, pb)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aps_a = []
        aps_b = []
        for j in range(y.shape[1]):
            yj = y[idx, j]
            if yj.sum() == 0 or yj.sum() == yj.size:
                continue
            aps_a.append(average_precision_score(yj, pa[idx, j]))
            aps_b.append(average_precision_score(yj, pb[idx, j]))
        diffs[b] = float(np.mean(aps_a) - np.mean(aps_b)) if aps_a else np.nan
    valid = diffs[np.isfinite(diffs)]
    return {
        "comparison": f"{name_a} - {name_b}",
        "n_valid_rows": int(n),
        "n_bootstrap": int(n_boot),
        "bootstrap_seed": int(seed),
        "point_macro_ap_diff": float(point),
        "point_macro_ap_diff_pp": 100.0 * float(point),
        "ci95_low": float(np.percentile(valid, 2.5)),
        "ci95_high": float(np.percentile(valid, 97.5)),
        "ci95_low_pp": 100.0 * float(np.percentile(valid, 2.5)),
        "ci95_high_pp": 100.0 * float(np.percentile(valid, 97.5)),
        "n_finite_bootstrap": int(valid.size),
    }


def run_summary(run: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "name": name,
        "status": run["status"],
        "arm": run["arm"],
        "label": run.get("label"),
        "seed": int(run["seed"]),
        "counts_kind": run.get("counts_kind"),
        "best_epoch": int(run["best_epoch"]),
        "last_epoch": int(run["last_epoch"]),
        "epochs_configured": int(run["epochs_configured"]),
        "epochs_run": int(run["epochs_run"]),
        "valid_macro_ap": float(run["valid_macro_ap"]),
        "valid_macro_ap_pct": float(run["valid_macro_ap_pct"]),
        "train_macro_ap": float(run["train_macro_ap"]),
        "train_macro_ap_pct": float(run["train_macro_ap_pct"]),
        "reload_delta": float(run["reload_delta"]),
        "last_epoch_valid_macro_ap_pct": 100.0 * float(run["last_epoch_valid_macro_ap"]),
        "last_epoch_train_loss": float(run["last_epoch_train_loss"]),
        "per_label_valid_ap": {k: float(v) for k, v in run["per_label_valid_ap"].items()},
        "per_label_valid_ap_pct": {k: 100.0 * float(v) for k, v in run["per_label_valid_ap"].items()},
        "per_label_train_ap_pct": {k: 100.0 * float(v) for k, v in run["per_label_train_ap"].items()},
        "parameters": int(run["model_info"]["total_parameters"]),
        "base_parameters": int(run["model_info"]["base_parameters"]),
        "branch_parameters": int(run["model_info"]["branch_parameters"]),
        "gpu_hours": float(run["gpu_hours"]),
        "seconds": float(run["seconds"]),
        "gpu_peak_allocated_bytes": int(run.get("gpu_peak_allocated_bytes", 0)),
        "test_evaluated": bool(run["test_evaluated"]),
        "guard_attempts": list(run.get("guard", {}).get("attempts", [])),
        "init_gate": run.get("init_gate"),
        "first_step_gate": run.get("first_step_gate"),
        "branch_post_gate": run.get("branch_post_gate"),
        "first_epoch_rows_sha256": run.get("first_epoch_rows_sha256"),
        "label_hashes": run.get("label_hashes"),
        "feature_spec_hash": run.get("feature_spec_hash"),
        "norm_fingerprint": run.get("norm_fingerprint"),
        "cache_identity": run.get("cache_identity"),
        "curve": run.get("_curve", []),
        "predictions_npz": C.track_rel(Path(run["_predictions_path"])),
        "run_json": C.track_rel(C.RUNS / name / "run.json"),
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "environment": run.get("environment"),
    }


def p0_summary() -> dict[str, Any]:
    ecfp = C.read_json(C.ANALYSIS / "ecfp_validation.json")
    lgbm = C.read_json(C.ANALYSIS / "p0_lgbm.json")
    mlp = C.read_json(C.ANALYSIS / "p0_mlp.json")
    manifest = C.read_json(C.ANALYSIS / "p0_features_manifest.json")
    return {
        "features": {
            "train": {
                k: manifest["splits"]["train"][k]
                for k in (
                    "n_graphs",
                    "n_paths_total",
                    "n_paths_per_graph",
                    "global",
                    "full_binary",
                    "null_audit",
                )
            },
            "valid": {
                k: manifest["splits"]["valid"][k]
                for k in (
                    "n_graphs",
                    "n_paths_total",
                    "n_paths_per_graph",
                    "global",
                    "full_binary",
                    "null_audit",
                )
            },
        },
        "ecfp": {
            "valid_macro_ap_pct": float(ecfp["valid_macro_ap_pct"]),
            "historical_macro_ap_pct": float(ecfp["historical_macro_ap_pct"]),
            "delta_vs_historical_pp": float(ecfp["delta_vs_historical_pp"]),
            "reproduction_ok": bool(ecfp["reproduction_ok"]),
            "per_label": ecfp["per_label"],
            "fingerprint": ecfp["fingerprint"],
        },
        "lgbm": {
            "valid_macro_ap_pct": {
                k: float(v["valid_macro_ap_pct"]) for k, v in lgbm["results"].items()
            },
            "train_macro_ap_pct": {
                k: float(v["train_macro_ap_pct"]) for k, v in lgbm["results"].items()
            },
            "per_label": {
                k: [
                    {
                        "label": row["label"],
                        "valid_ap_pct": float(row["valid_ap_pct"]),
                        "train_ap_pct": float(row["train_ap_pct"]),
                    }
                    for row in v["per_label"]
                ]
                for k, v in lgbm["results"].items()
            },
        },
        "mlp": {
            "valid_macro_ap_pct": float(mlp["valid_macro_ap_pct"]),
            "train_macro_ap_pct": float(mlp["train_macro_ap_pct"]),
            "best_epoch": int(mlp["best_epoch"]),
            "per_label_valid_ap_pct": {
                k: 100.0 * float(v) for k, v in mlp["per_label_valid_ap"].items()
            },
            "reload_delta": float(mlp["reload_delta"]),
        },
        "labels": list(C.LABELS),
    }


def markdown_table(rows: list[list[Any]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main() -> int:
    p0 = p0_summary()
    p1a = read_run(RUN_NAMES["p1a"])
    runs = {arm: read_run(name) for arm, name in RUN_NAMES.items() if arm != "p1a"}
    missing = [name for name in RUN_NAMES.values() if read_run(name) is None]
    if missing:
        raise SystemExit(f"missing formal runs: {missing}")
    summaries = {arm: run_summary(run, RUN_NAMES[arm]) for arm, run in runs.items()}
    p1a_summary = run_summary(p1a, RUN_NAMES["p1a"])
    assert p1a_summary is not None
    if p1a_summary["status"] != "complete_no_test" or p1a_summary["test_evaluated"]:
        raise SystemExit("P1a: bad status / test flag")
    if p1a_summary["guard_attempts"]:
        raise SystemExit(f"P1a: test guard attempts {p1a_summary['guard_attempts']}")
    for arm in ("A0", "A1", "A2", "A3"):
        assert summaries[arm] is not None
        if summaries[arm]["status"] != "complete_no_test" or summaries[arm]["test_evaluated"]:
            raise SystemExit(f"{arm}: bad status / test flag")
        if summaries[arm]["guard_attempts"]:
            raise SystemExit(f"{arm}: test guard attempts {summaries[arm]['guard_attempts']}")

    ap = {arm: summaries[arm]["valid_macro_ap_pct"] for arm in ("A0", "A1", "A2", "A3")}
    gain_full_base = ap["A2"] - ap["A0"]
    gain_full_null = ap["A2"] - ap["A3"]
    gain_full_type = ap["A2"] - ap["A1"]
    labels = list(C.LABELS)
    full_ge_null = [
        name
        for name in labels
        if summaries["A2"]["per_label_valid_ap_pct"][name]
        >= summaries["A3"]["per_label_valid_ap_pct"][name]
    ]
    full_gt_null = [
        name
        for name in labels
        if summaries["A2"]["per_label_valid_ap_pct"][name]
        > summaries["A3"]["per_label_valid_ap_pct"][name]
    ]
    trigger = (
        gain_full_base >= 2.0
        and gain_full_null >= 1.5
        and len(full_ge_null) >= 6
    )
    decision = {
        "gain_full_base_pp": float(gain_full_base),
        "gain_full_null_pp": float(gain_full_null),
        "gain_full_type_pp": float(gain_full_type),
        "full_ge_null_label_count": int(len(full_ge_null)),
        "full_ge_null_labels": full_ge_null,
        "full_gt_null_labels": full_gt_null,
        "threshold_gain_full_base_pp": 2.0,
        "threshold_gain_full_null_pp": 1.5,
        "threshold_label_count": 6,
        "confirmation_triggered": bool(trigger),
        "decision": "CONFIRM_SEEDS_1_2" if trigger else "STOP_AFTER_SEED_0",
        "rule": "Gain_FULL_BASE >= 2.0 pp AND Gain_FULL_NULL >= 1.5 pp AND AP(A2)>=AP(A3) on >=6/10 labels",
    }

    bootstrap_seed0 = {
        "full_base": paired_bootstrap(RUN_NAMES["A2"], RUN_NAMES["A0"]),
        "full_null": paired_bootstrap(RUN_NAMES["A2"], RUN_NAMES["A3"]),
    }

    confirmation: dict[str, Any] | None = None
    if trigger:
        conf_runs = {key: read_run(name) for key, name in CONFIRM_NAMES.items()}
        missing_conf = [name for key, name in CONFIRM_NAMES.items() if conf_runs[key] is None]
        if missing_conf:
            raise SystemExit(f"confirmation triggered but missing runs: {missing_conf}")
        conf_runs = {k: v for k, v in conf_runs.items() if v is not None}
        conf_summary = {
            f"{arm}_s{seed}": run_summary(run, CONFIRM_NAMES[(arm, seed)])
            for (arm, seed), run in conf_runs.items()
        }
        per_seed = []
        for seed in (0, 1, 2):
            rows = []
            for arm in ("A0", "A2", "A3"):
                key = f"{arm}_s{seed}"
                rows.append(conf_summary[key] if key in conf_summary else summaries[arm])
            aps = {row["arm"]: row["valid_macro_ap_pct"] for row in rows}
            per_seed.append(
                {
                    "seed": seed,
                    "A0": aps["A0"],
                    "A2": aps["A2"],
                    "A3": aps["A3"],
                    "gain_full_base_pp": aps["A2"] - aps["A0"],
                    "gain_full_null_pp": aps["A2"] - aps["A3"],
                    "gain_full_type_pp": aps["A2"] - aps.get("A1", float("nan")),
                }
            )
        mean_gain_base = float(np.mean([row["gain_full_base_pp"] for row in per_seed]))
        mean_gain_null = float(np.mean([row["gain_full_null_pp"] for row in per_seed]))
        per_label = {}
        for name in labels:
            per_label[name] = {}
            for seed in (0, 1, 2):
                a2 = (
                    summaries["A2"]["per_label_valid_ap_pct"][name]
                    if seed == 0
                    else conf_summary[f"A2_s{seed}"]["per_label_valid_ap_pct"][name]
                )
                a0 = (
                    summaries["A0"]["per_label_valid_ap_pct"][name]
                    if seed == 0
                    else conf_summary[f"A0_s{seed}"]["per_label_valid_ap_pct"][name]
                )
                a3 = (
                    summaries["A3"]["per_label_valid_ap_pct"][name]
                    if seed == 0
                    else conf_summary[f"A3_s{seed}"]["per_label_valid_ap_pct"][name]
                )
                per_label[name][f"seed{seed}"] = {
                    "A0": a0,
                    "A2": a2,
                    "A3": a3,
                    "gain_full_base_pp": a2 - a0,
                    "gain_full_null_pp": a2 - a3,
                }
            per_label[name]["mean"] = {
                k: float(np.mean([per_label[name][f"seed{s}"][k] for s in (0, 1, 2)]))
                for k in ("A0", "A2", "A3", "gain_full_base_pp", "gain_full_null_pp")
            }
        bootstrap = {
            "seed0_full_base": paired_bootstrap(RUN_NAMES["A2"], RUN_NAMES["A0"]),
            "seed0_full_null": paired_bootstrap(RUN_NAMES["A2"], RUN_NAMES["A3"]),
        }
        for seed in (1, 2):
            bootstrap[f"seed{seed}_full_base"] = paired_bootstrap(
                CONFIRM_NAMES[("A2", seed)], CONFIRM_NAMES[("A0", seed)]
            )
            bootstrap[f"seed{seed}_full_null"] = paired_bootstrap(
                CONFIRM_NAMES[("A2", seed)], CONFIRM_NAMES[("A3", seed)]
            )
        confirmation = {
            "per_seed": per_seed,
            "mean_gain_full_base_pp": mean_gain_base,
            "mean_gain_full_null_pp": mean_gain_null,
            "per_label": per_label,
            "bootstrap": bootstrap,
            "runs": conf_summary,
        }

    # GPU ledger: only runs that actually used CUDA count against the A100 budget.
    gpu_runs = []
    for name, run in [(RUN_NAMES["p1a"], p1a)] + [(n, runs[a]) for a, n in RUN_NAMES.items() if a != "p1a"]:
        if run is None:
            continue
        if bool(run.get("environment", {}).get("cuda_available")):
            gpu_runs.append({"name": name, "gpu_hours": float(run.get("gpu_hours", 0.0))})
    if confirmation:
        for key, run in confirmation["runs"].items():
            if run and bool(run.get("environment", {}).get("cuda_available")):
                gpu_runs.append({"name": key, "gpu_hours": float(run.get("gpu_hours", 0.0))})
    smoke_path = C.ANALYSIS / "gpu_smoke.json"
    if smoke_path.is_file():
        smoke = C.read_json(smoke_path)
        gpu_runs.append({"name": "smoke_A2_gpu", "gpu_hours": float(smoke["gpu_hours"])})
    gpu_total = float(sum(row["gpu_hours"] for row in gpu_runs))

    p1a_vs_240 = {
        "historical_240ep_seed0_valid_macro_ap_pct": 60.62,
        "historical_240ep_seed0_best_epoch": 231,
        "observed_480ep_valid_macro_ap_pct": p1a_summary["valid_macro_ap_pct"],
        "observed_480ep_best_epoch": p1a_summary["best_epoch"],
        "delta_pp": p1a_summary["valid_macro_ap_pct"] - 60.62,
        "historical_source": "tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/REPORT.md",
        "caveat": "cosine horizon differs (480 vs 240 epochs); not a pure epoch-count causal contrast",
    }

    summary = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "p0": p0,
        "p1a": p1a_summary,
        "p1a_vs_historical_240": p1a_vs_240,
        "p1b_seed0": summaries,
        "bootstrap_seed0": bootstrap_seed0,
        "decision": decision,
        "confirmation": confirmation,
        "gpu_ledger": {
            "runs": gpu_runs,
            "total_gpu_hours": gpu_total,
            "hard_cap_gpu_hours": 10.0,
            "within_cap": bool(gpu_total <= 10.0),
        },
        "test_access": {
            "test_evaluated": False,
            "guard_attempts": {
                name: run.get("guard", {}).get("attempts", [])
                for name, run in [(RUN_NAMES["p1a"], p1a)] + list(runs.items())
            },
        },
    }
    C.write_json(C.ANALYSIS / "p1_summary.json", summary)
    C.write_json(C.ANALYSIS / "results_machine.json", summary)

    # Markdown tables (facts only; REPORT.md contains interpretation).
    lines = ["# P1/P0 tables (generated by `code/pc_aggregate.py`)", ""]
    lines.append("## P0 LightGBM")
    rows = []
    for kind in ("type", "full", "full_binary"):
        rows.append([kind, f"{p0['lgbm']['valid_macro_ap_pct'][kind]:.2f}", f"{p0['lgbm']['train_macro_ap_pct'][kind]:.2f}"])
    lines.append(markdown_table(rows, ["features", "Validation macro AP (%)", "Train macro AP (%)"]))
    lines.append("")
    lines.append(f"ECFP4: Validation macro AP **{p0['ecfp']['valid_macro_ap_pct']:.4f}%** "
                 f"(historical {p0['ecfp']['historical_macro_ap_pct']:.2f}%, delta "
                 f"{p0['ecfp']['delta_vs_historical_pp']:+.4f} pp; ok={p0['ecfp']['reproduction_ok']}).")
    lines.append(f"FULL-S2 MLP: Validation macro AP **{p0['mlp']['valid_macro_ap_pct']:.4f}%**, "
                 f"Train **{p0['mlp']['train_macro_ap_pct']:.4f}%**, best epoch {p0['mlp']['best_epoch']}.")
    lines.append("")
    lines.append("## P1b seed 0")
    rows = [
        [arm, f"{summaries[arm]['valid_macro_ap_pct']:.4f}", f"{summaries[arm]['train_macro_ap_pct']:.4f}",
         summaries[arm]["best_epoch"], summaries[arm]["last_epoch"], f"{summaries[arm]['gpu_hours']:.4f}"]
        for arm in ("A0", "A1", "A2", "A3")
    ]
    lines.append(markdown_table(rows, ["arm", "Val macro AP (%)", "Train AP (%)", "best ep", "last ep", "GPU h"]))
    lines.append("")
    lines.append(f"- Gain_FULL_BASE = **{gain_full_base:+.4f} pp**")
    lines.append(f"- Gain_FULL_NULL = **{gain_full_null:+.4f} pp**")
    lines.append(f"- Gain_FULL_TYPE = **{gain_full_type:+.4f} pp**")
    lines.append(f"- FULL >= NULL on **{len(full_ge_null)}/10** labels (strict > on {len(full_gt_null)}/10)")
    lines.append(f"- Confirmation triggered: **{trigger}** → `{decision['decision']}`")
    lines.append("")
    lines.append("Paired Validation bootstrap (95% percentile, 10,000 resamples):")
    for key, b in bootstrap_seed0.items():
        lines.append(f"- seed0 {key}: {b['point_macro_ap_diff_pp']:+.3f} pp "
                     f"[{b['ci95_low_pp']:+.3f}, {b['ci95_high_pp']:+.3f}]")
    lines.append("")
    lines.append("## P1b seed 0 per-label Validation AP (%)")
    rows = []
    for name in labels:
        rows.append([
            name,
            f"{summaries['A0']['per_label_valid_ap_pct'][name]:.2f}",
            f"{summaries['A1']['per_label_valid_ap_pct'][name]:.2f}",
            f"{summaries['A2']['per_label_valid_ap_pct'][name]:.2f}",
            f"{summaries['A3']['per_label_valid_ap_pct'][name]:.2f}",
            f"{summaries['A2']['per_label_valid_ap_pct'][name] - summaries['A3']['per_label_valid_ap_pct'][name]:+.2f}",
        ])
    lines.append(markdown_table(rows, ["label", "A0", "A1", "A2", "A3", "A2-A3"]))
    lines.append("")
    lines.append("## P1a 480 epochs vs historical 240 epochs")
    rows = [[
        "480ep seed0",
        f"{p1a_summary['valid_macro_ap_pct']:.4f}",
        p1a_summary["best_epoch"],
        p1a_summary["last_epoch"],
        f"{p1a_summary['train_macro_ap_pct']:.4f}",
    ], [
        "historical 240ep seed0",
        f"{p1a_vs_240['historical_240ep_seed0_valid_macro_ap_pct']:.2f}",
        p1a_vs_240["historical_240ep_seed0_best_epoch"],
        240,
        "n/a",
    ]]
    lines.append(markdown_table(rows, ["run", "Val macro AP (%)", "best ep", "last ep", "Train AP (%)"]))
    lines.append("")
    lines.append(f"Caveat: {p1a_vs_240['caveat']}.")
    if confirmation:
        lines.append("")
        lines.append("## Confirmation seeds 1-2")
        rows = [
            [row["seed"], f"{row['A0']:.4f}", f"{row['A2']:.4f}", f"{row['A3']:.4f}",
             f"{row['gain_full_base_pp']:+.4f}", f"{row['gain_full_null_pp']:+.4f}"]
            for row in confirmation["per_seed"]
        ]
        lines.append(markdown_table(rows, ["seed", "A0", "A2", "A3", "A2-A0", "A2-A3"]))
        lines.append("")
        lines.append(f"Mean Gain_FULL_BASE = **{confirmation['mean_gain_full_base_pp']:+.4f} pp**; "
                     f"Mean Gain_FULL_NULL = **{confirmation['mean_gain_full_null_pp']:+.4f} pp**.")
        lines.append("")
        lines.append("Paired Validation bootstrap (95% percentile, 10,000 resamples):")
        for key, b in confirmation["bootstrap"].items():
            lines.append(f"- {key}: {b['point_macro_ap_diff_pp']:+.3f} pp "
                         f"[{b['ci95_low_pp']:+.3f}, {b['ci95_high_pp']:+.3f}]")
    lines.append("")
    lines.append(f"GPU total: **{gpu_total:.4f} A100-hours** (cap 10.0).")
    (C.ANALYSIS / "REPORT_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.log(f"aggregate written: Gain_FULL_BASE={gain_full_base:+.3f} pp, "
          f"Gain_FULL_NULL={gain_full_null:+.3f} pp, trigger={trigger}, GPU={gpu_total:.3f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
