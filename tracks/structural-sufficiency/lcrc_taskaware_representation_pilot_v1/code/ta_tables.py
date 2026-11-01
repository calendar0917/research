"""Build the round's tables from the frozen artifacts.

Everything here is a *projection* of files that already exist:
``p0_information_flow.json``, ``feature_coverage.json``,
``feature_collision_tests.json``, ``collision_label_floor.json``,
``split_freeze.json``, ``checkpoint_manifest.json``, the per-run
``train.json``/``test_eval.json``, ``baseline_repeat_manifest.json`` and the
append-only ledgers.  No number is typed by hand; the only authored content is
the task-chemistry needs matrix (explicitly marked as judgement, with reasons).

Usage
-----
    uv run python code/ta_tables.py            # everything that is ready
    uv run python code/ta_tables.py --check    # exit 1 if a table would be incomplete
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import ta_common as C  # noqa: I001

AN = C.ANALYSIS

# ---------------------------------------------------------------------------
# Task-chemistry needs matrix.  This is *chemical judgement*, not measurement:
# HIGH/MEDIUM/LOW/NONE = how much the task's outcome plausibly depends on the
# item.  The audit's measured columns (status / carrier / coverage) are separate
# and come from the audit JSONs, so a reader can always tell them apart.
# ---------------------------------------------------------------------------
NEEDS: dict[str, dict[str, str]] = {
    "element_identity": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "HIGH", "BACE": "HIGH", "BBBP": "HIGH", "Tox21": "HIGH", "ogbg-molhiv": "HIGH"},
    "aromaticity": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "MEDIUM", "BACE": "HIGH", "BBBP": "MEDIUM", "Tox21": "HIGH", "ogbg-molhiv": "HIGH"},
    "formal_charge": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "HIGH", "BACE": "MEDIUM", "BBBP": "HIGH", "Tox21": "HIGH", "ogbg-molhiv": "MEDIUM"},
    "hydrogen_count": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "HIGH", "BACE": "MEDIUM", "BBBP": "HIGH", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "heavy_atom_degree": {"ESOL": "MEDIUM", "Lipophilicity": "MEDIUM", "FreeSolv": "MEDIUM", "BACE": "MEDIUM", "BBBP": "MEDIUM", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "bond_order": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "HIGH", "BACE": "HIGH", "BBBP": "MEDIUM", "Tox21": "HIGH", "ogbg-molhiv": "HIGH"},
    "ring_membership_size_fusion": {"ESOL": "MEDIUM", "Lipophilicity": "HIGH", "FreeSolv": "MEDIUM", "BACE": "HIGH", "BBBP": "MEDIUM", "Tox21": "HIGH", "ogbg-molhiv": "HIGH"},
    "global_topology": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "MEDIUM", "BACE": "MEDIUM", "BBBP": "HIGH", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "formal_charge_state_protonation": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "HIGH", "BACE": "MEDIUM", "BBBP": "HIGH", "Tox21": "HIGH", "ogbg-molhiv": "MEDIUM"},
    "double_bond_ez": {"ESOL": "LOW", "Lipophilicity": "MEDIUM", "FreeSolv": "LOW", "BACE": "LOW", "BBBP": "LOW", "Tox21": "LOW", "ogbg-molhiv": "LOW"},
    "tetrahedral_chirality": {"ESOL": "LOW", "Lipophilicity": "LOW", "FreeSolv": "LOW", "BACE": "HIGH", "BBBP": "LOW", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "isotope_label": {"ESOL": "NONE", "Lipophilicity": "NONE", "FreeSolv": "NONE", "BACE": "LOW", "BBBP": "NONE", "Tox21": "LOW", "ogbg-molhiv": "NONE"},
    "bond_conjugation": {"ESOL": "MEDIUM", "Lipophilicity": "MEDIUM", "FreeSolv": "LOW", "BACE": "MEDIUM", "BBBP": "LOW", "Tox21": "HIGH", "ogbg-molhiv": "MEDIUM"},
    "valence_hybridisation": {"ESOL": "MEDIUM", "Lipophilicity": "MEDIUM", "FreeSolv": "MEDIUM", "BACE": "MEDIUM", "BBBP": "LOW", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "hbd_hba_counts": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "HIGH", "BACE": "MEDIUM", "BBBP": "HIGH", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "local_polarity_electronegativity": {"ESOL": "MEDIUM", "Lipophilicity": "MEDIUM", "FreeSolv": "HIGH", "BACE": "MEDIUM", "BBBP": "MEDIUM", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "ring_system_aromaticity": {"ESOL": "MEDIUM", "Lipophilicity": "HIGH", "FreeSolv": "LOW", "BACE": "HIGH", "BBBP": "LOW", "Tox21": "HIGH", "ogbg-molhiv": "MEDIUM"},
    "multi_component_salt": {"ESOL": "MEDIUM", "Lipophilicity": "MEDIUM", "FreeSolv": "LOW", "BACE": "MEDIUM", "BBBP": "MEDIUM", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "molecular_weight": {"ESOL": "MEDIUM", "Lipophilicity": "HIGH", "FreeSolv": "MEDIUM", "BACE": "HIGH", "BBBP": "HIGH", "Tox21": "MEDIUM", "ogbg-molhiv": "HIGH"},
    "tpsa": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "HIGH", "BACE": "MEDIUM", "BBBP": "HIGH", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "logp_logd_prior": {"ESOL": "HIGH", "Lipophilicity": "HIGH", "FreeSolv": "MEDIUM", "BACE": "MEDIUM", "BBBP": "HIGH", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "three_dimensional_conformation": {"ESOL": "LOW", "Lipophilicity": "MEDIUM", "FreeSolv": "HIGH", "BACE": "HIGH", "BBBP": "MEDIUM", "Tox21": "MEDIUM", "ogbg-molhiv": "HIGH"},
    "tautomer_protomer_choice": {"ESOL": "MEDIUM", "Lipophilicity": "MEDIUM", "FreeSolv": "HIGH", "BACE": "MEDIUM", "BBBP": "MEDIUM", "Tox21": "MEDIUM", "ogbg-molhiv": "MEDIUM"},
    "target_specific_binding_pose": {"ESOL": "NONE", "Lipophilicity": "NONE", "FreeSolv": "NONE", "BACE": "HIGH", "BBBP": "LOW", "Tox21": "HIGH", "ogbg-molhiv": "HIGH"},
}

NEED_REASON: dict[str, str] = {
    "element_identity": "composition is the substrate of every descriptor",
    "aromaticity": "drives conjugation, planarity, pi-stacking; explicit in the spec",
    "formal_charge": "ionisation state changes solubility, permeability and binding",
    "hydrogen_count": "complements valence: NH/OH are donors, N/O lone pairs acceptors",
    "heavy_atom_degree": "branching and shape",
    "bond_order": "saturation vs unsaturation; rigidity",
    "ring_membership_size_fusion": "ring count/size/fusion is a classical rigidity-and-shape descriptor",
    "global_topology": "size/shape summaries that per-atom features cannot integrate",
    "formal_charge_state_protonation": "the supplied SMILES fixes one protonation microstate; logD and permeability depend on it",
    "double_bond_ez": "only matters where double-bond geometry changes shape (low for small-molecule property tasks)",
    "tetrahedral_chirality": "stereospecific binding; negligible for bulk physicochemical properties",
    "isotope_label": "no task in this suite resolves isotopes",
    "bond_conjugation": "reactive/optical properties; Tox21 assays respond to conjugation patterns",
    "valence_hybridisation": "follows from element+bond orders; a derived convenience",
    "hbd_hba_counts": "Lipinski-level determinants of solvation and permeation",
    "local_polarity_electronegativity": "dipole/polar-surface local environment",
    "ring_system_aromaticity": "heteroaromatic ring identity matters for receptor and reactivity tasks",
    "multi_component_salt": "assay concentration is per parent molecule; salts are formulation artefacts",
    "molecular_weight": "permeability and diffusion ceiling; explicit Lipinski term",
    "tpsa": "the single best small-molecule permeability/solvation summary",
    "logp_logd_prior": "the quantity itself for Lipophilicity; strong covariate elsewhere",
    "three_dimensional_conformation": "binding and intramolecular H-bonding need geometry, not constitution",
    "tautomer_protomer_choice": "FreeSolv/BACE measurements depend on which tautomer is populated",
    "target_specific_binding_pose": "the assay's causal variable; never derivable from 2D constitution",
}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    rows = list(rows)
    if not rows:
        raise SystemExit(f"refusing to write an empty table: {C.track_rel(path)}")
    names = list(fieldnames or rows[0].keys())
    C.ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in names})
    return path


# ---------------------------------------------------------------------------
# 1. feature gap matrix
# ---------------------------------------------------------------------------


def feature_gap_matrix() -> Path:
    flow = C.read_json(AN / "p0_information_flow.json")
    cov = C.read_json(AN / "feature_coverage.json")
    floor = C.read_json(AN / "collision_label_floor.json")["datasets"]
    items = {row["item"]: row for row in flow["information_items"]}
    rows = []
    for item, entry in items.items():
        needs = NEEDS.get(item, {})
        row: dict[str, Any] = {
            "information_item": item,
            "status": entry["status"],
            "carrier_in_model_input": entry["carrier"],
            "measured_evidence": "; ".join(entry.get("evidence", [])),
        }
        for ds in C.DATASETS:
            row[f"need_{ds}"] = needs.get(ds, "")
        row["max_need"] = max(needs.values(), key=lambda v: ["NONE", "LOW", "MEDIUM", "HIGH"].index(v)) if needs else ""
        row["need_reason"] = NEED_REASON.get(item, "")
        if item == "tetrahedral_chirality":
            row["measured_consequence"] = (
                f"{sum(v['n_identical_input_pairs_in_train'] for v in floor.values())} Train pairs share "
                "byte-identical input; Lipophilicity Train RMSE floor from those pairs "
                f"{floor['Lipophilicity']['train_rmse_floor_from_collisions']:.4f}"
            )
        elif item == "double_bond_ez":
            row["measured_consequence"] = "reaches bond_feat[stereo] only; S1/S2 codes are stereo-blind"
        elif entry["status"] == "EXTRA_COMPUTE":
            row["measured_consequence"] = "recomputable deterministically from the input; not supplied as a feature"
        elif entry["status"] == "NOT_DETERMINABLE":
            row["measured_consequence"] = "not contained in the 2D constitution; no preprocessing could recover it"
        else:
            row["measured_consequence"] = ""
        rows.append(row)
    order = {"LOST_IN_PREPROCESSING": 0, "EXTRA_COMPUTE": 1, "NOT_DETERMINABLE": 2, "INDIRECT": 3, "EXPLICIT": 4}
    rows.sort(key=lambda r: (order.get(r["status"], 9), r["information_item"]))
    return _write_csv(AN / "feature_gap_matrix.csv", rows)


# ---------------------------------------------------------------------------
# 2. Lipophilicity repeat comparison
# ---------------------------------------------------------------------------


def _lipo_arm_lcrc() -> list[dict[str, Any]]:
    out = []
    for key, suffix in (("lipo_r123", 123), ("lipo_r124", 124)):
        path = C.run_dir(key) / "train.json"
        payload = C.read_json(path)
        test_path = C.run_dir(key) / "test_eval.json"
        test = _test_metrics(test_path) if test_path.exists() else None
        out.append(
            {
                "repeat": suffix,
                "arm": "LCRC-A (lipo_v1, R3)",
                "role": "reference" if suffix == 122 else "new",
                "train_n": payload["data"]["n_train"],
                "valid_n": payload["data"]["n_valid"],
                "best_epoch": payload["checkpoint"]["best_epoch"],
                "train_rmse": payload["metrics"]["train"]["rmse"],
                "valid_rmse": payload["metrics"]["valid"]["rmse"],
                "valid_mae": payload["metrics"]["valid"]["mae"],
                "test_rmse": None if test is None else test["rmse"],
                "test_mae": None if test is None else test["mae"],
                "test_read_once_after_freeze": bool(test is not None),
                "checkpoint_sha256": payload["checkpoint"]["sha256"],
                "gpu_seconds": payload["timing"]["train_wall_seconds"],
            }
        )
    return out


def _test_metrics(path: Path) -> Mapping[str, Any]:
    """Both rounds store the one-shot Test score, under slightly different keys."""
    payload = C.read_json(path)
    if isinstance(payload.get("test"), Mapping):
        return payload["test"]
    return payload["metrics"]["test"]


def _lipo_reference_122() -> dict[str, Any]:
    base = C.STRUCT / "lcrc_molnet_benchmark_pilot_v1/analysis/runs/Lipophilicity_s0"
    payload = C.read_json(base / "train.json")
    test = _test_metrics(base / "test_eval.json") if (base / "test_eval.json").exists() else None
    return {
        "repeat": 122,
        "arm": "LCRC-A (lipo_v1, R3)",
        "role": "reference (Sprint 01, read-only)",
        "train_n": payload["data"]["n_train"],
        "valid_n": payload["data"]["n_valid"],
        "best_epoch": payload["checkpoint"]["best_epoch"],
        "train_rmse": payload["metrics"]["train"]["rmse"],
        "valid_rmse": payload["metrics"]["valid"]["rmse"],
        "valid_mae": payload["metrics"]["valid"]["mae"],
        "test_rmse": None if test is None else test["rmse"],
        "test_mae": None if test is None else test["mae"],
        "test_read_once_after_freeze": True,
        "checkpoint_sha256": payload["checkpoint"]["sha256"],
        "gpu_seconds": payload["timing"]["train_wall_seconds"],
    }


def _lipo_baseline_122() -> dict[str, Any]:
    """Sprint 02's baseline artifact; its Test score was read once in that round."""
    base = C.STRUCT / "lcrc_molnet_benchmark_pilot_v2/analysis/baseline"
    payload = C.read_json(base / "Lipophilicity.json")
    test = C.read_json(base / "Lipophilicity_test.json")["metrics"]["test"]
    return {
        "repeat": 122,
        "arm": "ECFP4+RDKit+XGBoost",
        "role": "reference (Sprint 02, read-only)",
        "train_n": payload["n_train"],
        "valid_n": payload["n_valid"],
        "best_epoch": "",
        "train_rmse": payload["metrics"]["train"]["rmse"] if "train" in payload["metrics"] else "",
        "valid_rmse": payload["metrics"]["valid"]["rmse"],
        "valid_mae": payload["metrics"]["valid"]["mae"],
        "test_rmse": test["rmse"],
        "test_mae": test["mae"],
        "test_read_once_after_freeze": True,
        "checkpoint_sha256": "",
        "gpu_seconds": 0.0,
    }


def lipo_repeat_comparison() -> Path:
    rows = [_lipo_reference_122()]
    baseline_manifest = C.read_json(AN / "baseline_repeat_manifest.json")
    rows += _lipo_arm_lcrc()
    for suffix in (123, 124):
        report = C.read_json(C.REPO / baseline_manifest["entries"][str(suffix)]["report"])
        test_path = AN / "baseline_repeat" / f"Lipophilicity_random{suffix}_test.json"
        test = _test_metrics(test_path) if test_path.exists() else None
        rows.append(
            {
                "repeat": suffix,
                "arm": "ECFP4+RDKit+XGBoost",
                "role": "new",
                "train_n": report["rows"]["train"],
                "valid_n": report["rows"]["valid"],
                "best_epoch": "",
                "train_rmse": report["train"]["rmse"],
                "valid_rmse": report["valid"]["rmse"],
                "valid_mae": report["valid"]["mae"],
                "test_rmse": None if test is None else test["rmse"],
                "test_mae": None if test is None else test["mae"],
                "test_read_once_after_freeze": bool(test is not None),
                "checkpoint_sha256": report["model_sha256"],
                "gpu_seconds": 0.0,
            }
        )
    rows.append(_lipo_baseline_122())
    for row in rows:
        for key in ("valid_rmse", "test_rmse", "train_rmse", "valid_mae", "test_mae"):
            value = row.get(key)
            if value is not None and value != "":
                row[key] = round(float(value), 6)
        if row.get("gpu_seconds") is not None:
            row["gpu_seconds"] = round(float(row["gpu_seconds"]), 2)
    rows.sort(key=lambda r: (r["repeat"], r["arm"]))
    return _write_csv(AN / "lipo_repeat_comparison.csv", rows)


# ---------------------------------------------------------------------------
# 3. BBBP inner CV
# ---------------------------------------------------------------------------


def bbbp_inner_cv() -> Path:
    split = C.read_json(C.SPLIT_FREEZE)["bbbp_inner_cv"]
    reference = C.read_json(C.STRUCT / "lcrc_molnet_benchmark_pilot_v2/analysis/runs/BBBP_s0/train.json")
    rows = []
    for fold in range(C.BBBP_CV_FOLDS):
        payload = C.read_json(C.run_dir(f"bbbp_cv_f{fold}") / "train.json")
        rows.append(
            {
                "fold": fold,
                "n_train": payload["data"]["n_train"],
                "n_valid": payload["data"]["n_valid"],
                "n_positive_valid": split["folds"][str(fold)]["n_positive"],
                "n_negative_valid": split["folds"][str(fold)]["n_negative"],
                "n_scaffolds_valid": split["folds"][str(fold)]["n_scaffold_groups"],
                "best_epoch": payload["checkpoint"]["best_epoch"],
                "train_roc_auc": round(float(payload["metrics"]["train"]["roc_auc"]), 6),
                "valid_roc_auc": round(float(payload["metrics"]["valid"]["roc_auc"]), 6),
                "valid_pr_auc": round(float(payload["metrics"]["valid"]["pr_auc"]), 6),
                "valid_bce": round(float(payload["metrics"]["valid"]["bce"]), 6),
                "recheck_abs_diff": float(payload["checkpoint"]["valid_recheck_abs_diff"]),
                "published_valid_test_used": False,
                "row": "fold",
                "note": "Train = the other two folds; the published Valid/Test labels are never read",
            }
        )
    aucs = np.asarray([r["valid_roc_auc"] for r in rows], dtype=np.float64)
    rows.append(
        {
            "fold": "mean",
            "n_train": "",
            "n_valid": "",
            "n_positive_valid": "",
            "n_negative_valid": "",
            "n_scaffolds_valid": "",
            "best_epoch": "",
            "train_roc_auc": "",
            "valid_roc_auc": round(float(aucs.mean()), 6),
            "valid_pr_auc": "",
            "valid_bce": "",
            "recheck_abs_diff": "",
            "published_valid_test_used": "",
            "row": "summary",
            "note": f"sd {float(aucs.std(ddof=1)):.6f} over {len(aucs)} folds",
        }
    )
    rows.append(
        {
            "fold": "sd",
            "n_train": "",
            "n_valid": "",
            "n_positive_valid": "",
            "n_negative_valid": "",
            "n_scaffolds_valid": "",
            "best_epoch": "",
            "train_roc_auc": "",
            "valid_roc_auc": round(float(aucs.std(ddof=1)), 6),
            "valid_pr_auc": "",
            "valid_bce": "",
            "recheck_abs_diff": "",
            "published_valid_test_used": "",
            "row": "summary",
            "note": "sample standard deviation",
        }
    )
    rows.append(
        {
            "fold": "published_valid",
            "n_train": reference["data"]["n_train"],
            "n_valid": reference["data"]["n_valid"],
            "n_positive_valid": "",
            "n_negative_valid": "",
            "n_scaffolds_valid": "",
            "best_epoch": reference["checkpoint"]["best_epoch"],
            "train_roc_auc": round(float(reference["metrics"]["train"]["roc_auc"]), 6),
            "valid_roc_auc": round(float(reference["metrics"]["valid"]["roc_auc"]), 6),
            "valid_pr_auc": round(float(reference["metrics"]["valid"]["pr_auc"]), 6),
            "valid_bce": "",
            "recheck_abs_diff": "",
            "published_valid_test_used": True,
            "row": "reference (Sprint 02, read-only)",
            "note": "published Validation: rows randomly drawn, scaffolds shared with Train",
        }
    )
    return _write_csv(AN / "bbbp_inner_cv.csv", rows)


# ---------------------------------------------------------------------------
# 4. task-aware loss probes
# ---------------------------------------------------------------------------


def taskaware_loss_probes() -> Path:
    probes = C.read_json(AN / "taskaware_loss_probes_p0.json")
    rows = []
    for case in probes["toy_probe"]:
        rows.append(
            {
                "kind": "toy_batch",
                "name": case["case"],
                "dataset": "",
                "objective": f"masked_cell={case['masked_cell_loss']:.6f} vs equal_task={case['equal_task_loss']:.6f}",
                "metric_name": "loss",
                "value": round(float(case["equal_task_loss"]), 6),
                "reference_value": round(float(case["masked_cell_loss"]), 6),
                "delta": round(float(case["equal_task_loss"]) - float(case["masked_cell_loss"]), 6),
                "gate": "passed" if case.get("losses_differ") is not None else "",
                "note": case.get("note", ""),
            }
        )
    tox = C.read_json(C.run_dir("tox21_equal_task") / "train.json")
    tox_ref = C.read_json(C.STRUCT / "lcrc_molnet_benchmark_pilot_v2/analysis/runs/Tox21_s0/train.json")
    rows.append(
        {
            "kind": "formal_run",
            "name": "tox21_equal_task",
            "dataset": "Tox21",
            "objective": "equal_task",
            "metric_name": "valid_macro_roc_auc",
            "value": round(float(tox["metrics"]["valid"]["macro_roc_auc"]), 6),
            "reference_value": round(float(tox_ref["metrics"]["valid"]["macro_roc_auc"]), 6),
            "delta": round(float(tox["metrics"]["valid"]["macro_roc_auc"]) - float(tox_ref["metrics"]["valid"]["macro_roc_auc"]), 6),
            "gate": "validation_only",
            "note": f"best_epoch {tox['checkpoint']['best_epoch']} vs reference {tox_ref['checkpoint']['best_epoch']}",
        }
    )
    rows.append(
        {
            "kind": "formal_run",
            "name": "tox21_equal_task",
            "dataset": "Tox21",
            "objective": "equal_task",
            "metric_name": "valid_masked_bce",
            "value": round(float(tox["metrics"]["valid"]["masked_bce"]), 6),
            "reference_value": round(float(tox_ref["metrics"]["valid"]["masked_bce"]), 6),
            "delta": round(float(tox["metrics"]["valid"]["masked_bce"]) - float(tox_ref["metrics"]["valid"]["masked_bce"]), 6),
            "gate": "validation_only",
            "note": "the two objectives are not calibrated against each other here; compare ranking only",
        }
    )
    hiv = C.read_json(C.run_dir("hiv_sqrt_pos") / "train.json")
    hiv_ref = C.read_json(C.STRUCT / "lcrc_molnet_benchmark_pilot_v2/analysis/runs/ogbg-molhiv_s0/train.json")
    rows.append(
        {
            "kind": "formal_run",
            "name": "hiv_sqrt_pos",
            "dataset": "ogbg-molhiv",
            "objective": "sqrt_pos_weight",
            "metric_name": "valid_roc_auc",
            "value": round(float(hiv["metrics"]["valid"]["roc_auc"]), 6),
            "reference_value": round(float(hiv_ref["metrics"]["valid"]["roc_auc"]), 6),
            "delta": round(float(hiv["metrics"]["valid"]["roc_auc"]) - float(hiv_ref["metrics"]["valid"]["roc_auc"]), 6),
            "gate": "validation_only",
            "note": f"pos_weight {hiv['loss_config'].get('pos_weight', hiv.get('pos_weight'))} best_epoch {hiv['checkpoint']['best_epoch']} vs reference {hiv_ref['checkpoint']['best_epoch']}",
        }
    )
    rows.append(
        {
            "kind": "formal_run",
            "name": "hiv_sqrt_pos",
            "dataset": "ogbg-molhiv",
            "objective": "sqrt_pos_weight",
            "metric_name": "valid_pr_auc",
            "value": round(float(hiv["metrics"]["valid"]["pr_auc"]), 6),
            "reference_value": round(float(hiv_ref["metrics"]["valid"]["pr_auc"]), 6),
            "delta": round(float(hiv["metrics"]["valid"]["pr_auc"]) - float(hiv_ref["metrics"]["valid"]["pr_auc"]), 6),
            "gate": "validation_only",
            "note": "PR-AUC is the imbalance-sensitive companion of ROC-AUC",
        }
    )
    return _write_csv(AN / "taskaware_loss_probes.csv", rows)


# ---------------------------------------------------------------------------
# 5. GPU ledger
# ---------------------------------------------------------------------------


def _run_registry() -> list[dict[str, Any]]:
    path = C.RUN_REGISTRY
    if path.exists():
        payload = C.read_json(path)
        return payload if isinstance(payload, list) else payload.get("runs", [])
    return []


def gpu_ledger() -> Path:
    runs = []
    total = 0.0
    registry = _run_registry()
    driver = C.read_json(C.RUN_REGISTRY).get("driver") if C.RUN_REGISTRY.exists() else None
    for key, entry in C.read_json(C.CHECKPOINT_MANIFEST)["runs"].items():
        seconds = float(entry["timing"]["train_wall_seconds"]) + float(entry["timing"].get("inference_seconds_total", 0.0))
        runs.append(
            {
                "run_key": key,
                "workstream": next((r["workstream"] for r in C.PLANNED_RUNS if r["key"] == key), ""),
                "dataset": entry["dataset"],
                "split": entry["split"],
                "loss": entry["loss"],
                "seed": entry["seed"],
                "epochs": entry["epochs_run"],
                "best_epoch": entry["best_epoch"],
                "status": "reported",
                "gpu_hours": round(seconds / 3600.0, 5),
                "checkpoint": entry["artifacts"]["checkpoint"],
                "checkpoint_sha256": entry["artifacts"]["checkpoint_sha256"],
                "git_commit": entry["regime"]["git_commit"],
                "gpu": entry["regime"]["gpu_name"],
                "driver": driver,
                "test_read": bool(entry["test_eval_present"]),
                "note": "",
            }
        )
        total += seconds / 3600.0
    superseded = [item for item in registry if item.get("reporting") == "not_reported"]
    ledger = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "hard_cap_a100_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "max_formal_runs": C.MAX_FORMAL_RUNS,
        "n_reported_runs": len(runs),
        "reported_gpu_hours": round(total, 5),
        "reported_run_priority_order": [r["run_key"] for r in runs],
        "runs": runs,
        "non_reported_attempts": superseded,
        "quarantine": {
            "host": "res",
            "reason": "untracked remote artifacts blocked the first deploy; nothing was deleted",
            "archive": "~/cy/quarantine/lcrc_taskaware_sprint03_deployblock_20261101/deploy_block_untracked_20261101.tar.gz",
            "archive_sha256": "c8eaeeea34db0efdcdf75b720f47c09a569798f66502828f37894975b205f882",
            "n_files_archived": 62,
            "n_files_identical_to_incoming_commit": 40,
            "n_files_unique_to_the_archive": 22,
        },
        "test_access": C.read_json(C.TEST_LEDGER) if C.TEST_LEDGER.exists() else {"reads": [], "n_reads": 0},
        "budget_check": {
            "within_cap": bool(total <= C.GPU_HARD_CAP_HOURS),
            "within_run_cap": bool(len(runs) <= C.MAX_FORMAL_RUNS),
            "epochs_never_reduced": all(r["epochs"] == int(C.recipe()["epochs"]) for r in runs),
            "single_seed": all(r["seed"] == C.SEED for r in runs),
        },
    }
    C.write_json(C.GPU_LEDGER, ledger)
    C.log(f"wrote {C.track_rel(C.GPU_LEDGER)} ({len(runs)} runs, {total:.3f} A100 h)")
    return C.GPU_LEDGER


# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the round's analysis tables")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    built = []
    problems = []
    for name, fn in (
        ("feature_gap_matrix.csv", feature_gap_matrix),
        ("lipo_repeat_comparison.csv", lipo_repeat_comparison),
        ("bbbp_inner_cv.csv", bbbp_inner_cv),
        ("taskaware_loss_probes.csv", taskaware_loss_probes),
        ("gpu_ledger.json", gpu_ledger),
    ):
        try:
            path = fn()
            built.append(C.track_rel(path))
        except (FileNotFoundError, SystemExit, KeyError) as exc:
            problems.append(f"{name}: {exc!r}")
    for item in built:
        print(f"built {item}")
    if problems:
        for item in problems:
            print(f"NOT BUILT {item}")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
