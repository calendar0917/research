"""Assemble the main table, Valid paired bootstrap and summary.json.

No model is trained here.  External reference records (TRC / GINE) are read
only; their predictions are aligned to the same official Valid molecules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D

ARMS = ("ACRA_ADD", "ACRA_COND", "ACRA_REL")
RUNS = {a: f"{a}_S0" for a in ARMS}
REFS = {
    "TRC_FACTOR_CUTRESP": {
        "dir": C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0",
        "kind": "external_reference_read_only",
        "note": "per-cut whole-molecule response model from the parent round",
    },
    "GINE_RAW_REF": {
        "dir": C.PARENT_ROUND / "results" / "GINE_RAW_LIPO_R0",
        "kind": "external_reference_read_only",
        "note": "external GNN reference prediction record",
    },
}


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {"rmse": C.rmse(pred, truth), "mae": C.mae(pred, truth)}


def load_arm(run_id: str) -> dict[str, Any]:
    rdir = C.run_dir(run_id)
    manifest = C.read_json(rdir / "manifest.json")
    with np.load(rdir / "predictions_best_train.npz") as tr, \
            np.load(rdir / "predictions_best.npz") as va:
        train = {k: np.asarray(tr[k], np.float64) for k in tr.files}
        valid = {k: np.asarray(va[k], np.float64) for k in va.files}
    curves = C.read_csv(rdir / "curves.csv")
    return {"manifest": manifest, "train": train, "valid": valid, "curves": curves,
            "run_dir": str(rdir)}


def load_reference(path: Path) -> dict[str, Any]:
    with np.load(path / "predictions_best.npz") as handle:
        return {k: np.asarray(handle[k], np.float64) for k in handle.files}


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    raw = D.load_raw(cfg)
    splits = D.official_splits(raw)
    labels = np.asarray(raw["labels"], np.float64)

    arms: dict[str, Any] = {}
    for arm in ARMS:
        rdir = C.run_dir(RUNS[arm])
        if not (rdir / "manifest.json").is_file():
            log(f"  {arm}: missing run {rdir}")
            return {"error": f"missing run {RUNS[arm]}"}
        arms[arm] = load_arm(RUNS[arm])
    valid_row_ids = arms["ACRA_ADD"]["valid"]["row_id"].astype(np.int64)
    truth = arms["ACRA_ADD"]["valid"]["label"]
    if not np.array_equal(valid_row_ids, splits["valid"]):
        raise ValueError("Valid prediction rows are not the official Valid split order")

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        rec = arms[arm]
        m = rec["manifest"]
        v = rec["valid"]
        best_row = next(r for r in rec["curves"]
                        if int(r["epoch"]) == int(m["best"]["epoch"]))
        rec["best_row"] = best_row
        rows.append({
            "model": arm,
            "kind": "acra_arm",
            "seed": m["seed"],
            "params": m["params"]["total"],
            "best_epoch": m["best"]["epoch"],
            "epochs_run": m["epochs"],
            "train_rmse": best_row["train_rmse"],
            "train_mae": best_row["train_mae"],
            "valid_rmse": m["best"]["valid_rmse"],
            "valid_mae": m["best"]["valid_mae"],
            "generalization_gap_rmse": m["best"]["valid_rmse"] - best_row["train_rmse"],
            "gate": (None if not np.isfinite(float(m["final"].get("gate", float("nan"))))
                     else float(m["final"]["gate"])),
            "wall_seconds": m["wall_seconds"],
            "gpu_hours": m["gpu_hours"],
            "run_id": RUNS[arm],
            "test_labels_read": False,
        })
        arms[arm]["valid_metrics"] = _metrics(v["pred"], truth)

    refs: dict[str, Any] = {}
    for name, spec in REFS.items():
        path = Path(spec["dir"])
        if not (path / "predictions_best.npz").is_file():
            log(f"  reference {name} missing at {path}")
            continue
        pred = load_reference(path)
        v_pred, v_lab = pred["valid_pred"], pred["valid_label"]
        if not np.allclose(v_lab, labels[splits["valid"]], atol=1e-8):
            raise ValueError(f"{name} Valid labels do not match the official split")
        rec = {
            "model": name,
            "kind": spec["kind"],
            "seed": 0,
            "params": None,
            "best_epoch": None,
            "epochs_run": None,
            "train_rmse": C.rmse(pred["train_pred"], pred["train_label"]),
            "train_mae": C.mae(pred["train_pred"], pred["train_label"]),
            "valid_rmse": C.rmse(v_pred, v_lab),
            "valid_mae": C.mae(v_pred, v_lab),
            "generalization_gap_rmse": C.rmse(v_pred, v_lab)
            - C.rmse(pred["train_pred"], pred["train_label"]),
            "gate": None,
            "wall_seconds": None,
            "gpu_hours": None,
            "run_id": str(path.relative_to(C.REPO_ROOT)),
            "test_labels_read": False,
            "note": spec["note"],
        }
        refs[name] = {"row": rec, "valid_pred": v_pred}
        rows.append(rec)

    ridge_path = C.analysis_dir() / "reference_morgan_ridge.json"
    if ridge_path.is_file():
        ridge = C.read_json(ridge_path)
        rows.append({
            "model": "MORGAN2_COUNT_RIDGE",
            "kind": "non_gnn_reference",
            "seed": 0,
            "params": None,
            "best_epoch": None,
            "epochs_run": None,
            "train_rmse": ridge["train_rmse"],
            "train_mae": ridge["train_mae"],
            "valid_rmse": ridge["valid_rmse"],
            "valid_mae": ridge["valid_mae"],
            "generalization_gap_rmse": ridge["valid_rmse"] - ridge["train_rmse"],
            "gate": None,
            "wall_seconds": ridge["wall_seconds"],
            "gpu_hours": 0.0,
            "run_id": "analysis/reference_morgan_ridge_pred.npz",
            "test_labels_read": False,
            "note": "Morgan radius=2 count fingerprint, Ridge alpha=10, Train only",
        })

    comparison_preds = {name: refs[name]["valid_pred"] for name in refs}
    ridge_pred_path = C.analysis_dir() / "reference_morgan_ridge_pred.npz"
    if ridge_pred_path.is_file():
        with np.load(ridge_pred_path) as handle:
            if not np.array_equal(np.asarray(handle["valid_row_id"], np.int64),
                                  splits["valid"]):
                raise ValueError("Ridge reference rows do not match the official Valid split")
            comparison_preds["MORGAN2_COUNT_RIDGE"] = np.asarray(handle["valid_pred"],
                                                                np.float64)

    C.write_csv(C.analysis_dir() / "main_table.csv", rows)
    C.write_csv(C.results_root() / "main_table.csv", rows)
    log(f"wrote main_table.csv ({len(rows)} rows)")

    # Valid paired bootstrap (molecules resampled as a unit)
    comparisons: dict[str, Any] = {}
    pairs = (("ACRA_COND", "ACRA_ADD"), ("ACRA_REL", "ACRA_COND"),
             ("ACRA_REL", "ACRA_ADD"), ("ACRA_ADD", "GINE_RAW_REF"),
             ("ACRA_REL", "GINE_RAW_REF"), ("ACRA_ADD", "TRC_FACTOR_CUTRESP"),
             ("ACRA_REL", "TRC_FACTOR_CUTRESP"),
             ("ACRA_ADD", "MORGAN2_COUNT_RIDGE"),
             ("ACRA_REL", "MORGAN2_COUNT_RIDGE"))
    for candidate, control in pairs:
        if candidate.startswith("ACRA"):
            pc = arms[candidate]["valid"]["pred"]
        else:
            pc = comparison_preds[candidate]
        if control.startswith("ACRA"):
            po = arms[control]["valid"]["pred"]
        else:
            po = comparison_preds[control]
        key = f"{candidate}_vs_{control}"
        comparisons[key] = C.paired_bootstrap(po, pc, truth, n_boot=2000, seed=0)
        comparisons[key]["definition"] = "RMSE(control) - RMSE(candidate); positive => candidate better"
    log("wrote Valid paired bootstrap comparisons")

    # training curve plot (Valid RMSE per epoch)
    plot_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for arm in ARMS:
            curves = arms[arm]["curves"]
            ep = [int(r["epoch"]) for r in curves]
            val = [float(r["valid_rmse"]) for r in curves]
            ax.plot(ep, val, label=arm, linewidth=1.4)
        for name, spec in REFS.items():
            if name in refs:
                value = refs[name]["row"]["valid_rmse"]
                ax.axhline(value, linestyle="--", linewidth=0.9,
                           label=f"{name} {value:.3f}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("Valid RMSE (label units)")
        ax.set_title("ACRA-v1 seed 0: Valid whole-molecule RMSE")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plot_path = C.analysis_dir() / "training_curves.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"  plot skipped: {exc!r}")

    diagnostics_path = C.analysis_dir() / "contribution_diagnostics.json"
    diagnostics = C.read_json(diagnostics_path) if diagnostics_path.is_file() else {}
    cases_path = C.analysis_dir() / "cases.json"
    cases = C.read_json(cases_path) if cases_path.is_file() else {}
    support_path = C.analysis_dir() / "structural_support.json"
    support = C.read_json(support_path) if support_path.is_file() else {}
    pairs_path = C.analysis_dir() / "pair_delta.json"
    pair_delta = C.read_json(pairs_path) if pairs_path.is_file() else {}
    checks_path = C.analysis_dir() / "checks.json"
    checks = C.read_json(checks_path) if checks_path.is_file() else {}
    relational_path = C.analysis_dir() / "relational_capacity.json"
    relational = C.read_json(relational_path) if relational_path.is_file() else {}

    gpu_hours = sum(float(arms[a]["manifest"]["gpu_hours"]) for a in ARMS)
    summary = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "commit": C.git_commit(),
        "cache_sha256": arms["ACRA_ADD"]["manifest"]["cache_sha256"],
        "raw_cache_sha256": arms["ACRA_ADD"]["manifest"]["raw_cache_sha256"],
        "seed": 0,
        "horizon_epochs": arms["ACRA_ADD"]["manifest"]["epochs_planned"],
        "horizon_source": arms["ACRA_ADD"]["manifest"]["horizon_source"],
        "test_labels_read": False,
        "checks_all_passed": bool(checks.get("all_passed")),
        "arms": {
            arm: {
                "run_id": RUNS[arm],
                "params": arms[arm]["manifest"]["params"]["total"],
                "best_epoch": arms[arm]["manifest"]["best"]["epoch"],
                "valid_rmse": arms[arm]["manifest"]["best"]["valid_rmse"],
                "valid_mae": arms[arm]["manifest"]["best"]["valid_mae"],
                "train_rmse_final": arms[arm]["manifest"]["final"]["train_rmse"],
                "train_rmse_best_epoch": arms[arm]["best_row"]["train_rmse"],
                "generalization_gap_rmse": arms[arm]["manifest"]["best"]["valid_rmse"]
                - arms[arm]["best_row"]["train_rmse"],
                "gate_final": arms[arm]["manifest"]["final"].get("gate"),
                "interaction": arm != "ACRA_ADD",
                "relation": arm == "ACRA_REL",
                "gpu_hours": arms[arm]["manifest"]["gpu_hours"],
                "wall_seconds": arms[arm]["manifest"]["wall_seconds"],
                "max_vram_bytes": arms[arm]["manifest"]["max_vram_bytes"],
                "curve_csv": str(Path(arms[arm]["run_dir"]).relative_to(C.REPO_ROOT)
                                 / "curves.csv"),
                "checkpoint": str(Path(arms[arm]["run_dir"]).relative_to(C.REPO_ROOT)
                                  / "state_best.pt"),
            }
            for arm in ARMS
        },
        "external_references": {name: refs[name]["row"] for name in refs},
        "paired_bootstrap_valid": comparisons,
        "budget": {"gpu_hours_used": gpu_hours, "gpu_hours_max": 4,
                   "gpus_used_in_parallel": 2, "runs": 3},
        "structural_support_headline": {
            "valid_S_seen_rate": support.get("overall", {}).get("S", {}).get("valid_seen_rate"),
            "valid_C_seen_rate": support.get("overall", {}).get("C", {}).get("valid_seen_rate"),
            "valid_combo_seen_rate": support.get("overall", {}).get("combo", {}).get(
                "valid_seen_rate"),
            "bridge_combo_seen_rate": support.get("strata", {}).get("bridge", {}).get(
                "combo", {}).get("valid_seen_rate"),
            "non_bridge_combo_seen_rate": support.get("strata", {}).get("non_bridge", {}).get(
                "combo", {}).get("valid_seen_rate"),
            "valid_molecules_all_events_seen": support.get("valid_molecules", {}).get(
                "molecules_with_all_events_seen"),
            "n_valid_molecules": support.get("valid_molecules", {}).get("n"),
        },
        "mechanism_headline": {
            arm: {
                "valid_cancellation_index": diagnostics.get("arms", {}).get(arm, {}).get(
                    "valid", {}).get("cancellation_index"),
                "valid_mean_abs_e_atom": diagnostics.get("arms", {}).get(arm, {}).get(
                    "valid", {}).get("e_atom", {}).get("mean_abs"),
                "valid_mean_abs_e_bond": diagnostics.get("arms", {}).get(arm, {}).get(
                    "valid", {}).get("e_bond", {}).get("mean_abs"),
                "valid_mean_abs_e_rel": diagnostics.get("arms", {}).get(arm, {}).get(
                    "valid", {}).get("e_rel", {}).get("mean_abs"),
                "interaction_ablation": diagnostics.get("arms", {}).get(arm, {}).get(
                    "valid", {}).get("interaction", {}).get("ablation"),
                "relation_ablation": diagnostics.get("arms", {}).get(arm, {}).get(
                    "valid", {}).get("relation", {}).get("ablation"),
                "valid_novelty_vs_error": diagnostics.get("arms", {}).get(arm, {}).get(
                    "valid_novelty_vs_error"),
            }
            for arm in ARMS
        },
        "pair_delta_headline": {
            arm: {
                scope: {k: pair_delta.get("scopes", {}).get(RUNS[arm], {})
                        .get(scope, {}).get(k)
                        for k in ("n", "delta_mae", "delta_rmse", "direction_accuracy",
                                  "pearson_pred_true", "delta_mae_lo", "delta_mae_hi")}
                for scope in ("train_train", "train_valid", "valid_valid")
            }
            for arm in ARMS
        },
        "pair_delta_arm_comparisons_train_valid": pair_delta.get(
            "arm_comparisons_train_valid"),
        "relational_capacity_headline": {
            "real_groups_with_identical_local_multiset": relational.get(
                "real_molecules", {}).get("n_groups_with_identical_local_multiset"),
            "real_groups_with_different_relation_profile": relational.get(
                "real_molecules", {}).get("n_groups_with_different_relation_profile"),
            "real_eta2_e_atom": relational.get("real_molecules", {}).get(
                "eta_squared_group_explains", {}).get("e_atom"),
            "real_eta2_e_bond": relational.get("real_molecules", {}).get(
                "eta_squared_group_explains", {}).get("e_bond"),
            "real_eta2_e_rel": relational.get("real_molecules", {}).get(
                "eta_squared_group_explains", {}).get("e_rel"),
            "real_within_group_max_range": relational.get("real_molecules", {}).get(
                "within_group_term_range"),
            "atlas_graphs": relational.get("synthetic_graphs", {}).get(
                "atlas", {}).get("n_graphs_enumerated"),
            "atlas_groups_with_different_relations": relational.get(
                "synthetic_graphs", {}).get("atlas", {}).get(
                "n_groups_with_different_relation_profile"),
            "constructed_witness": relational.get("synthetic_graphs", {}).get(
                "constructed_witness", {}).get("pair_decomposition"),
            "constructed_witness_profiles_differ": relational.get(
                "synthetic_graphs", {}).get("constructed_witness", {}).get(
                "shared_ring_profiles_differ"),
        },
        "cases_headline": {
            key: {k: cases.get(key, {}).get(k) for k in
                  ("n_candidate_better", "n_candidate_worse",
                   "mean_squared_error_reduction", "mean_squared_error_reduction_ci")}
            for key in ("improvement_over_control", "degradation_vs_control",
                        "relation_improvement", "relation_degradation")
        },
        "training_curve_plot": str(plot_path.relative_to(C.REPO_ROOT)) if plot_path else None,
    }
    C.write_json(C.analysis_dir() / "summary.json", summary)
    # also keep the required results/summary.json copy
    C.write_json(C.results_root() / "summary.json", summary)
    log(f"wrote summary.json (gpu_hours={gpu_hours:.4f}, checks={summary['checks_all_passed']})")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    _args = ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())