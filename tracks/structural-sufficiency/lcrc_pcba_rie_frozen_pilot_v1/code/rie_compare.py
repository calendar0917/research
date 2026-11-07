"""Epoch-8 comparison, per-task diagnostics and the frozen decision.

Inputs are only artifacts produced by this round:

* ``analysis/feature_manifest.json``   frozen A replay + feature cache identity,
* ``runs/heads/{real,shuffle}/run.json``  per-epoch official Validation AP,
* ``analysis/paired_initialization.json`` pairing proof,
* ``runs/smoke/smoke.json``, ``analysis/motif_coverage.json``,
  ``analysis/shuffle_control_audit.json`` for the resource/coverage blocks.

Outputs:

* ``analysis/epoch8_comparison.json``  AP_A / AP_REAL / AP_SHUFFLE and deltas,
* ``analysis/per_task_deltas.csv``     127 scored task APs and all deltas,
* ``analysis/gpu_ledger.json``         GPU time, peaks, feature-construction cost,
* ``analysis/decision.json``           the frozen decision label.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import rie_common as C  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as PC  # noqa: E402

FEATURES = C.RUNS / "features"
HEADS = C.RUNS / "heads"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _finite(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return arr


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "sum": 0.0}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "sum": float(finite.sum()),
    }


def _stratum_summary(
    deltas: Mapping[str, np.ndarray], mask: np.ndarray, label: str
) -> dict[str, Any]:
    out: dict[str, Any] = {"stratum": label, "n_tasks": int(mask.sum())}
    for name, values in deltas.items():
        selected = values[mask]
        finite = selected[np.isfinite(selected)]
        out[name] = {
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "n_finite": int(finite.size),
        }
    return out


def _topk_share(deltas: np.ndarray, k: int) -> dict[str, Any]:
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    total = float(positive.sum())
    if total <= 0:
        return {"k": int(k), "share": None, "positive_mass": total}
    ordered = np.sort(positive)[::-1]
    top = float(ordered[:k].sum())
    return {
        "k": int(k),
        "share": float(top / total),
        "positive_mass": total,
        "top_value": float(ordered[:k].min()) if ordered[:k].size == k else float(ordered[-1]),
    }


def classify_decision(
    *,
    d_base: float,
    d_ctrl: float,
    d_ge20: float,
    improved_ge20: int,
    preconditions_ok: bool,
    intervention_ok: bool,
) -> tuple[str, list[str], dict[str, bool]]:
    """The frozen decision rule (pure function, unit-tested)."""
    rules = C.DECISION
    promising = rules["promising"]
    checks = {
        "D_base_ge_0.005": bool(d_base >= promising["d_base_min"]),
        "D_ctrl_ge_0.003": bool(d_ctrl >= promising["d_ctrl_min"]),
        "D_ge20_ge_0.003": bool(np.isfinite(d_ge20) and d_ge20 >= promising["d_ge20_min"]),
        "improved_ge20_tasks_ge_60": bool(improved_ge20 >= promising["improved_ge20_min"]),
    }
    if not preconditions_ok:
        return rules["invalid"]["incomplete"], ["precondition_failed"], checks
    if not intervention_ok:
        return rules["invalid"]["intervention"], ["insufficient_relation_intervention"], checks
    if all(checks.values()):
        return promising["label"], [], checks
    if d_base >= rules["weak"]["d_base_min"]:
        return rules["weak"]["label"], [k for k, v in checks.items() if not v], checks
    return rules["no_signal"]["label"], [k for k, v in checks.items() if not v], checks


def _weight_diagnostics(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    weight = payload["state_dict"]["task_weight.weight"].numpy().astype(np.float64)
    shared = payload["state_dict"]["shared_relation_projection.weight"].numpy().astype(np.float64)
    per_task_l2 = np.linalg.norm(weight, axis=1)
    return {
        "path": C.track_rel(path),
        "arm": payload.get("arm"),
        "task_weight_shape": list(weight.shape),
        "shared_projection_shape": list(shared.shape),
        "task_weight_l1": float(np.abs(weight).sum()),
        "task_weight_l2": float(np.linalg.norm(weight)),
        "task_weight_mean_abs": float(np.abs(weight).mean()),
        "task_weight_nonzero_fraction": float((weight != 0).mean()),
        "per_task_l2_min": float(per_task_l2.min()),
        "per_task_l2_median": float(np.median(per_task_l2)),
        "per_task_l2_max": float(per_task_l2.max()),
        "per_task_l2_mean": float(per_task_l2.mean()),
        "shared_projection_l2": float(np.linalg.norm(shared)),
        "per_task_l2": [float(v) for v in per_task_l2],
        "initial_state_sha256": payload.get("initial_state_sha256"),
        "final_state_sha256": payload.get("final_state_sha256"),
    }


# ---------------------------------------------------------------------------
# main comparison
# ---------------------------------------------------------------------------


def build_comparison() -> dict[str, Any]:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    manifest = C.read_json(C.ANALYSIS / "feature_manifest.json")
    coverage = C.read_json(C.ANALYSIS / "motif_coverage.json")
    shuffle_audit = C.read_json(C.ANALYSIS / "shuffle_control_audit.json")
    preflight = C.read_json(C.ANALYSIS / "paired_initialization.json")
    smoke = C.read_json(C.RUNS / "smoke" / "smoke.json")
    replay = C.read_json(C.ANALYSIS / "checkpoint_replay.json")
    real = C.read_json(HEADS / "real" / "run.json")
    shuffle = C.read_json(HEADS / "shuffle" / "run.json")
    paired = C.read_json(HEADS / "paired_run.json")

    epoch = str(C.MAIN_EPOCH)
    labels = np.load(FEATURES / "valid_labels.npy")
    base_logits = np.asarray(np.load(FEATURES / "valid_base_logits.npy", mmap_mode="r"))
    base_probs = PC.sigmoid(base_logits)
    base_task = MET.per_task_ap(labels, base_probs)
    base_official = MET.official_ap(labels, base_probs)
    ap_a = float(base_official["value"])

    real_task = _finite(real["evaluations"][epoch]["valid"]["per_task_ap"])
    shuffle_task = _finite(shuffle["evaluations"][epoch]["valid"]["per_task_ap"])
    base_task_arr = _finite(base_task["per_task"])
    positives = np.asarray(base_task["positive"], dtype=np.int64)
    observed = np.asarray(base_task["observed"], dtype=np.int64)

    ap_real = float(real["evaluations"][epoch]["valid"]["ap"])
    ap_shuffle = float(shuffle["evaluations"][epoch]["valid"]["ap"])
    d_base = ap_real - ap_a
    d_ctrl = ap_real - ap_shuffle

    scored = np.isfinite(real_task) & np.isfinite(shuffle_task) & np.isfinite(base_task_arr)
    delta_real_a = np.where(scored, real_task - base_task_arr, np.nan)
    delta_shuffle_a = np.where(scored, shuffle_task - base_task_arr, np.nan)
    delta_real_shuffle = np.where(scored, real_task - shuffle_task, np.nan)

    strata = {
        "ge20": positives >= int(C.DECISION["ge20_threshold"]),
        "5_19": (positives >= 5) & (positives < int(C.DECISION["ge20_threshold"])),
        "lt5": positives < 5,
    }
    strata_summary = {
        name: _stratum_summary(
            {
                "delta_REAL_minus_A": delta_real_a,
                "delta_SHUFFLE_minus_A": delta_shuffle_a,
                "delta_REAL_minus_SHUFFLE": delta_real_shuffle,
            },
            mask & scored,
            name,
        )
        for name, mask in strata.items()
    }
    ge20 = strata["ge20"] & scored
    d_ge20 = float(np.nanmean(delta_real_a[ge20])) if int(ge20.sum()) else float("nan")
    improved_ge20 = int(np.nansum(delta_real_a[ge20] > 0))
    improved_all = int(np.nansum(delta_real_a[scored] > 0))
    improved_vs_shuffle = int(np.nansum(delta_real_shuffle[scored] > 0))
    n_scored = int(scored.sum())

    arm_base = {
        "A": {
            "official_valid_ap": ap_a,
            "official_valid_ap_from_run_registry": C.AP_A,
            "valid_ap_abs_delta_to_anchor": abs(ap_a - C.AP_A),
            "macro_ap_independent": float(base_task["mean_over_scored_tasks"]),
            "valid_bce": float(MET.masked_bce_numpy(base_logits, labels)),
            "n_tasks_scored": int(base_task["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(base_task["n_tasks_with_observed_labels"]),
            "train_macro_ap": float(replay["train_macro_ap_replayed"]),
            "parameters": int(C.SOURCE["n_parameters"]),
        },
        "REAL": {
            "official_valid_ap": ap_real,
            "valid_bce": float(real["evaluations"][epoch]["valid"]["bce"]),
            "train_macro_ap": float(real["evaluations"][epoch]["train"]["macro_ap"]),
            "train_valid_gap": float(
                real["evaluations"][epoch]["train"]["macro_ap"] - ap_real
            ),
            "n_tasks_scored": int(real["evaluations"][epoch]["valid"]["n_tasks_scored"]),
            "parameters": int(C.SOURCE["n_parameters"]) + int(C.HEAD_PARAMETERS),
            "trainable_parameters": int(C.HEAD_PARAMETERS),
        },
        "SHUFFLE": {
            "official_valid_ap": ap_shuffle,
            "valid_bce": float(shuffle["evaluations"][epoch]["valid"]["bce"]),
            "train_macro_ap": float(shuffle["evaluations"][epoch]["train"]["macro_ap"]),
            "train_valid_gap": float(
                shuffle["evaluations"][epoch]["train"]["macro_ap"] - ap_shuffle
            ),
            "n_tasks_scored": int(shuffle["evaluations"][epoch]["valid"]["n_tasks_scored"]),
            "parameters": int(C.SOURCE["n_parameters"]) + int(C.HEAD_PARAMETERS),
            "trainable_parameters": int(C.HEAD_PARAMETERS),
        },
    }

    curve = {
        arm: {
            epoch_key: {
                "valid_ap": arm_report["evaluations"][epoch_key]["valid"]["ap"],
                "valid_bce": arm_report["evaluations"][epoch_key]["valid"]["bce"],
                "train_macro_ap": arm_report["evaluations"][epoch_key]["train"]["macro_ap"],
            }
            for epoch_key in arm_report["evaluations"]
        }
        for arm, arm_report in (("REAL", real), ("SHUFFLE", shuffle))
    }

    diag_real = _weight_diagnostics(HEADS / "real" / "head.pt")
    diag_shuffle = _weight_diagnostics(HEADS / "shuffle" / "head.pt")
    per_task_weight_real = diag_real["per_task_l2"]
    per_task_weight_shuffle = diag_shuffle["per_task_l2"]
    weight_diag = {"REAL": diag_real, "SHUFFLE": diag_shuffle}
    for _block in weight_diag.values():
        _block.pop("per_task_l2", None)

    head_flops = 2 * (
        C.PAIR_FEATURE_DIM * 16 + C.N_BINS * 16 * C.N_TASKS
    )  # MACs -> FLOPs, per molecule (3 bins)
    probe = coverage["timing_probe"]

    comparison = {
        "schema": "lcrc-pcba-rie-epoch8-comparison-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "main_epoch": int(C.MAIN_EPOCH),
        "arms": arm_base,
        "deltas": {
            "D_base_REAL_minus_A": d_base,
            "D_ctrl_REAL_minus_SHUFFLE": d_ctrl,
            "D_SHUFFLE_minus_A": ap_shuffle - ap_a,
            "D_ge20_REAL_minus_A": d_ge20 if np.isfinite(d_ge20) else None,
            "n_tasks_scored": n_scored,
            "n_tasks_improved_REAL_vs_A": improved_all,
            "n_tasks_improved_REAL_vs_SHUFFLE": improved_vs_shuffle,
            "n_tasks_improved_SHUFFLE_vs_A": int(np.nansum(delta_shuffle_a[scored] > 0)),
            "ge20_tasks": int(ge20.sum()),
            "ge20_tasks_improved_REAL_vs_A": improved_ge20,
            "ge20_tasks_improved_REAL_vs_SHUFFLE": int(np.nansum(delta_real_shuffle[ge20] > 0)),
        },
        "strata_by_valid_positives": strata_summary,
        "topk_positive_delta_contribution": {
            "REAL_minus_A": {k: _topk_share(delta_real_a, k) for k in (5, 10)},
            "REAL_minus_SHUFFLE": {k: _topk_share(delta_real_shuffle, k) for k in (5, 10)},
        },
        "per_task_delta_summary": {
            "REAL_minus_A": _summary(delta_real_a),
            "SHUFFLE_minus_A": _summary(delta_shuffle_a),
            "REAL_minus_SHUFFLE": _summary(delta_real_shuffle),
        },
        "evaluation_curve": curve,
        "weights": weight_diag,
        "pairing": {
            "initial_state_sha256": paired["initial_state_sha256"],
            "initial_state_identical": paired["paired_identical"]["initial_state"],
            "order_plan_identical": paired["paired_identical"]["order_plan"],
            "recipe_identical": paired["paired_identical"]["recipe"],
            "optimizer_identical": paired["paired_identical"]["optimizer"],
            "relation_inputs_differ": paired["paired_identical"]["relation_inputs_differ_between_arms"],
            "plan_sha256": paired["plan"]["plan_sha256"],
            "trainable_parameters_each_arm": int(C.HEAD_PARAMETERS),
            "preflight_all_passed": bool(preflight["all_passed"]),
        },
        "coverage": {
            split: {
                key: coverage[split][key]
                for key in (
                    "rows",
                    "fraction_with_eligible_pairs",
                    "fraction_with_multiple_distance_bins",
                    "fraction_REAL_ne_SHUFFLE_structure_level",
                    "molecules_with_any_eligible_pair",
                    "pairs_per_molecule_mean",
                    "motifs_per_molecule_mean",
                    "motif_family_counts",
                    "pair_bin_counts",
                    "parse_failed_molecules",
                )
            }
            for split in ("train", "valid")
        },
        "shuffle_control": {
            "valid_fraction_REAL_ne_SHUFFLE": shuffle_audit["splits"]["valid"][
                "fraction_REAL_ne_SHUFFLE_tensor"
            ],
            "train_fraction_REAL_ne_SHUFFLE": shuffle_audit["splits"]["train"][
                "fraction_REAL_ne_SHUFFLE_tensor"
            ],
            "valid_fraction_multiple_bins": shuffle_audit["splits"]["valid"][
                "fraction_with_multiple_distance_bins"
            ],
            "preservation_passed": bool(shuffle_audit["preservation_passed"]),
            "intervention_passed": bool(shuffle_audit["intervention_passed"]),
        },
        "diagnostics": {
            "head_flops_per_molecule": float(head_flops),
            "head_macs_per_molecule": float(head_flops / 2.0),
            "head_parameter_fraction_of_A": float(C.HEAD_PARAMETERS / (C.SOURCE["n_parameters"] + C.HEAD_PARAMETERS)),
            "cpu_structure_seconds_per_molecule_serial": float(probe["serial_seconds_per_molecule"]),
            "cpu_structure_seconds_per_molecule_parallel": float(probe["parallel_seconds_per_molecule"]),
            "cpu_structure_peak_rss_mb": float(probe["peak_rss_total_mb"]),
            "train_seconds_per_epoch_REAL": float(real["timings_seconds"]["train"] / C.EPOCHS),
            "train_seconds_per_epoch_SHUFFLE": float(shuffle["timings_seconds"]["train"] / C.EPOCHS),
            "extraction_forward_molecules_per_second": float(
                (C.N_TRAIN + C.N_VALID) / max(float(manifest["forward_seconds_total"]), 1e-9)
            ),
            "extraction_end_to_end_molecules_per_second": float(
                (C.N_TRAIN + C.N_VALID) / max(float(manifest["gpu_seconds"]), 1e-9)
            ),
            "table": {},
        },
        "test_evaluated": False,
    }
    comparison["diagnostics"]["table"] = {
        "relation_head_parameters": int(C.HEAD_PARAMETERS),
        "A_parameters": int(C.SOURCE["n_parameters"]),
        "relation_feature_columns": int(C.RELATION_DIM_FLAT),
        "task_weight_mean_abs_REAL": weight_diag["REAL"]["task_weight_mean_abs"],
        "task_weight_nonzero_fraction_REAL": weight_diag["REAL"]["task_weight_nonzero_fraction"],
    }

    # ---- per-task CSV ------------------------------------------------------
    rows = []
    for task_idx in range(C.N_TASKS):
        rows.append(
            {
                "task": task_idx,
                "valid_observed": int(observed[task_idx]),
                "valid_positive": int(positives[task_idx]),
                "scored": bool(scored[task_idx]),
                "ap_A": base_task_arr[task_idx],
                "ap_REAL_epoch8": real_task[task_idx],
                "ap_SHUFFLE_epoch8": shuffle_task[task_idx],
                "delta_REAL_minus_A": delta_real_a[task_idx],
                "delta_SHUFFLE_minus_A": delta_shuffle_a[task_idx],
                "delta_REAL_minus_SHUFFLE": delta_real_shuffle[task_idx],
                "real_task_weight_l2": per_task_weight_real[task_idx],
                "shuffle_task_weight_l2": per_task_weight_shuffle[task_idx],
                "stratum": "ge20"
                if positives[task_idx] >= int(C.DECISION["ge20_threshold"])
                else ("5_19" if positives[task_idx] >= 5 else "lt5"),
            }
        )
    csv_path = C.ANALYSIS / "per_task_deltas.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: ("" if isinstance(v, float) and not np.isfinite(v) else (f"{v:.12f}" if isinstance(v, float) else v))
                    for k, v in row.items()
                }
            )
    comparison["per_task_csv"] = C.track_rel(csv_path)
    comparison["per_task_csv_sha256"] = C.sha256_file(csv_path)
    C.write_json(C.ANALYSIS / "epoch8_comparison.json", comparison)

    # ---- GPU ledger --------------------------------------------------------
    gpu_ledger = {
        "schema": "lcrc-pcba-rie-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": manifest["device"],
        "entries": {
            "extraction": {
                "gpu_hours": float(manifest["gpu_hours"]),
                "gpu_peak": manifest["gpu_peak"],
                "forward_seconds": float(manifest["forward_seconds_total"]),
                "molecules": int(C.N_TRAIN + C.N_VALID),
            },
            "smoke": {
                "gpu_hours": float(smoke["seconds"] / 3600.0),
                "gpu_peak": smoke.get("gpu_peak", {}),
                "rows": int(len(smoke["rows"])),
            },
            "head_training_REAL": {
                "gpu_hours": float(real["gpu_hours"]),
                "gpu_peak": real["gpu_peak"],
                "total_seconds": float(real["total_seconds"]),
                "train_seconds": float(real["timings_seconds"]["train"]),
            },
            "head_training_SHUFFLE": {
                "gpu_hours": float(shuffle["gpu_hours"]),
                "gpu_peak": shuffle["gpu_peak"],
                "total_seconds": float(shuffle["total_seconds"]),
                "train_seconds": float(shuffle["timings_seconds"]["train"]),
            },
            "paired_validation_evaluation": {
                "gpu_hours": float(paired["gpu_hours"]),
                "note": "REAL + SHUFFLE formally trained back to back in one job",
            },
        },
        "cpu_entries": {
            "structure_build": {
                "seconds_full": float(
                    coverage["structures_fingerprint"]["train"]["build_seconds"]
                    + coverage["structures_fingerprint"]["valid"]["build_seconds"]
                ),
                "timing_probe": {
                    "serial_seconds": probe["serial_seconds"],
                    "parallel_seconds": probe["parallel_seconds"],
                    "extrapolation": probe["extrapolation"],
                    "peak_rss_total_mb": probe["peak_rss_total_mb"],
                },
            },
            "audit": {"seconds": None, "note": "measured in the audit job log; CPU only"},
        },
        "budget": {
            "gpu_hours_target": float(C.BUDGET["gpu_hours_target"]),
            "gpu_hours_hard": float(C.BUDGET["gpu_hours_hard"]),
        },
        "totals": {
            "gpu_hours_sum_of_stages": float(
                manifest["gpu_hours"] + smoke["seconds"] / 3600.0 + real["gpu_hours"] + shuffle["gpu_hours"]
            ),
            "gpu_hours_note": (
                "the paired training stages ran sequentially in one GPU1 job; the sum of the "
                "measured stage wall times is the conservative upper bound"
            ),
        },
        "test_evaluated": False,
    }
    gpu_ledger["totals"]["within_target"] = bool(
        gpu_ledger["totals"]["gpu_hours_sum_of_stages"] <= float(C.BUDGET["gpu_hours_target"])
    )
    gpu_ledger["totals"]["within_hard_cap"] = bool(
        gpu_ledger["totals"]["gpu_hours_sum_of_stages"] <= float(C.BUDGET["gpu_hours_hard"])
    )
    C.write_json(C.ANALYSIS / "gpu_ledger.json", gpu_ledger)

    # ---- decision ----------------------------------------------------------
    rules = C.DECISION
    promising = rules["promising"]
    preconditions = {
        "source_checkpoint_reproduced": bool(replay["valid_ap_reproduced"]),
        "feature_gates_passed": bool(manifest["all_passed"]),
        "invariance_audit_passed": bool(
            C.read_json(C.ANALYSIS / "relation_invariance_audit.json")["all_passed"]
        ),
        "shuffle_audit_passed": bool(shuffle_audit["all_passed"]),
        "paired_initialization_passed": bool(preflight["all_passed"]),
        "both_arms_completed_8_epochs": bool(
            real["global_steps"] == shuffle["global_steps"] == C.EPOCHS * C.STEPS_PER_EPOCH
        ),
        "all_eval_epochs_present": bool(
            set(real["evaluations"]) == set(shuffle["evaluations"]) == {str(e) for e in C.EVAL_EPOCHS}
        ),
        "no_test_access": bool(
            not manifest["test_isolation"]["test_predictions_saved"]
            and manifest["test_isolation"]["test_finite_labels"] == 0
        ),
        "within_gpu_budget": bool(gpu_ledger["totals"]["within_hard_cap"]),
    }
    label, reasons, promising_checks = classify_decision(
        d_base=d_base,
        d_ctrl=d_ctrl,
        d_ge20=d_ge20,
        improved_ge20=improved_ge20,
        preconditions_ok=all(preconditions.values()),
        intervention_ok=bool(shuffle_audit["intervention_passed"]),
    )
    if not all(preconditions.values()):
        reasons = [k for k, v in preconditions.items() if not v]

    interpretation = {
        "proposition_1_REAL_beats_A": bool(d_base > 0),
        "proposition_2_REAL_beats_SHUFFLE": bool(d_ctrl > 0),
        "proposition_3_advantage_on_well_supported_assays": bool(
            np.isfinite(d_ge20) and d_ge20 > 0 and improved_ge20 >= promising["improved_ge20_min"]
        ),
        "readout_useful_but_relation_not_implicated": bool(d_base >= 0.002 and d_ctrl < 0.001),
        "statement": (
            "Three separate propositions are reported: (1) REAL > A, (2) REAL > SHUFFLE, "
            "(3) the REAL-over-A advantage also holds on assays with >= 20 Validation positives. "
            "Only a consistent pattern justifies a full LCRC-RIE joint-training round; this round "
            "never starts one. A large D_base with a small D_ctrl means the tiny supervised readout "
            "may be useful but there is no evidence that the gain comes from true motif-distance "
            "relations."
        ),
    }
    decision = {
        "schema": "lcrc-pcba-rie-decision-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "label": label,
        "reasons": reasons,
        "metrics": {
            "AP_A": ap_a,
            "AP_REAL": ap_real,
            "AP_SHUFFLE": ap_shuffle,
            "D_base": d_base,
            "D_ctrl": d_ctrl,
            "D_SHUFFLE_minus_A": ap_shuffle - ap_a,
            "D_ge20": d_ge20 if np.isfinite(d_ge20) else None,
            "improved_ge20_tasks": improved_ge20,
            "ge20_tasks": int(ge20.sum()),
            "n_tasks_scored": n_scored,
            "n_tasks_improved_REAL_vs_A": improved_all,
            "n_tasks_improved_REAL_vs_SHUFFLE": improved_vs_shuffle,
        },
        "thresholds": {
            "promising": promising,
            "weak": rules["weak"],
            "min_intervention_fraction": rules["min_intervention_fraction"],
        },
        "promising_checks": {k: bool(v) for k, v in promising_checks.items()},
        "preconditions": {k: bool(v) for k, v in preconditions.items()},
        "interpretation": interpretation,
        "next_round_authorised": False,
        "guidance": (
            "This label is a resource decision, not a significance test. Even PROMISING_RELATION_SIGNAL "
            "only means a full joint-training round is worth considering; it is NOT started here."
        ),
        "test_evaluated": False,
        "artifacts": {
            "comparison": C.track_rel(C.ANALYSIS / "epoch8_comparison.json"),
            "per_task_deltas": C.track_rel(csv_path),
            "gpu_ledger": C.track_rel(C.ANALYSIS / "gpu_ledger.json"),
            "feature_manifest": C.track_rel(C.ANALYSIS / "feature_manifest.json"),
            "checkpoint_replay": C.track_rel(C.ANALYSIS / "checkpoint_replay.json"),
            "motif_coverage": C.track_rel(C.ANALYSIS / "motif_coverage.json"),
            "relation_invariance_audit": C.track_rel(C.ANALYSIS / "relation_invariance_audit.json"),
            "shuffle_control_audit": C.track_rel(C.ANALYSIS / "shuffle_control_audit.json"),
            "paired_initialization": C.track_rel(C.ANALYSIS / "paired_initialization.json"),
        },
    }
    C.write_json(C.ANALYSIS / "decision.json", decision)
    C.log(
        f"[compare] AP_A {ap_a:.10f} AP_REAL {ap_real:.10f} AP_SHUFFLE {ap_shuffle:.10f} "
        f"D_base {d_base:+.6f} D_ctrl {d_ctrl:+.6f} D_ge20 {d_ge20:+.6f} -> {label}"
    )
    return {"comparison": comparison, "decision": decision, "gpu_ledger": gpu_ledger}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        record = build_comparison()
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
