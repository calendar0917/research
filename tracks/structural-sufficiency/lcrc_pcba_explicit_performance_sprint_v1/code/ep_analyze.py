"""Build the committed analysis deliverables from the pulled run records.

Reads the formal run directories (``run.json`` + epoch-40 per-task reports),
the pulled preflight and feature-build audits, and an ``rr`` status capture.
Writes the eight pre-registered analysis artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence



def _configure() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure()

import ep_common as C  # noqa: E402


def _load(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _fmt(x: Any, nd: int = 9) -> Any:
    if x is None or x == "":
        return ""
    return round(float(x), nd)


def _status_of(run: Mapping[str, Any] | None, arm: str) -> str:
    if run is None:
        return "SKIPPED_BY_RULE" if arm == "COMBO" else "INCOMPLETE"
    status = str(run.get("status"))
    if status == "COMPLETED":
        return "COMPLETED"
    if status == "FAILED":
        return "FAILED"
    return "INCOMPLETE"


def _arm_row(
    arm: str,
    run: Mapping[str, Any] | None,
    a0_ap: float | None,
    a0_source: str,
    cancelled: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "arm": arm,
        "status": _status_of(run, arm),
        "reference": a0_source,
        "modules": "",
        "valid_ap10": "",
        "valid_ap20": "",
        "valid_ap30": "",
        "valid_ap40": "",
        "train_ap40": "",
        "train_bce40": "",
        "valid_bce40": "",
        "delta_vs_A0": "",
        "label": "",
        "tasks_improved": "",
        "tasks_degraded": "",
        "tasks_tied": "",
        "train_minus_valid_ap": "",
        "parameters": "",
        "extra_parameters": "",
        "gpu_peak_allocated_bytes": "",
        "gpu_hours": "",
        "step_ms_p50": "",
        "checkpoint_sha256": "",
    }
    if cancelled is not None and arm == "A0":
        row["status"] = str(cancelled.get("status", "CANCELLED_BY_PRINCIPAL"))
        row["label"] = "CANCELLED"
        row["gpu_hours"] = round(float(cancelled.get("gpu_hours", 0.0)), 6)
        row["parameters"] = int(C.A0_PARAMETERS)
        for e, v in (cancelled.get("valid_ap_by_epoch") or {}).items():
            row[f"valid_ap{e}"] = _fmt(v)
        return row
    if run is None:
        return row
    reports = run.get("fixed_epoch_reports") or {}
    row["modules"] = ",".join(run.get("modules") or [])
    for e in (10, 20, 30, 40):
        rep = reports.get(str(e))
        row[f"valid_ap{e}"] = "" if rep is None else _fmt(rep.get("ap"))
    rep40 = reports.get("40")
    if rep40 is not None:
        row["valid_bce40"] = _fmt(rep40.get("valid_bce"))
    te = run.get("train_eval_final") or {}
    if te:
        row["train_ap40"] = _fmt(te.get("ap"))
        row["train_bce40"] = _fmt(te.get("bce"))
    if arm == "A0":
        row["label"] = "REFERENCE"
    if a0_ap is not None and rep40 is not None:
        delta = float(rep40["ap"]) - float(a0_ap)
        row["delta_vs_A0"] = round(delta, 9)
        if arm == "A0":
            pass
        elif delta >= C.DECISION["screen_positive_delta_min"]:
            row["label"] = "SCREEN_POSITIVE"
        elif delta >= C.DECISION["weak_signal_delta_min"]:
            row["label"] = "WEAK_SIGNAL"
        elif delta < 0.0:
            row["label"] = "REGRESSION"
        else:
            row["label"] = "NO_CLEAR_GAIN"
    if te and rep40 is not None:
        row["train_minus_valid_ap"] = round(float(te["ap"]) - float(rep40["ap"]), 9)
    row["parameters"] = int(run.get("parameters", 0))
    row["extra_parameters"] = int(run.get("extra_parameters", 0))
    row["gpu_peak_allocated_bytes"] = int(run.get("gpu_peak_allocated_bytes", 0))
    row["gpu_hours"] = round(float(run.get("gpu_hours", 0.0)), 6)
    curve = run.get("_curve") or []
    row["step_ms_p50"] = "" if not curve else round(float(curve[-1].get("step_ms_p50", 0.0)), 4)
    row["checkpoint_sha256"] = run.get("checkpoint_epoch40_sha256") or ""
    return row


def _load_curve(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "train_curve.csv"
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def analyze(
    runs_dir: Path,
    *,
    preflight_path: Path,
    build_path: Path,
    branch_build_path: Path | None,
    out_dir: Path,
    rr_status: Mapping[str, Any] | None,
    smoke_path: Path | None = None,
    historical_a0_path: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = list(C.ARM_ORDER) + ["COMBO"]
    runs: dict[str, dict[str, Any] | None] = {}
    for arm in arms:
        path = runs_dir / arm / "run.json"
        if not path.is_file():
            runs[arm] = None
            continue
        run = _load(path)
        run["_curve"] = _load_curve(runs_dir / arm)
        runs[arm] = run
    preflight = _load(preflight_path) if preflight_path.is_file() else {}
    build = _load(build_path) if build_path.is_file() else {}
    if branch_build_path is not None and branch_build_path.is_file():
        branch_build = _load(branch_build_path)
        if "branch" in branch_build:
            build = dict(build)
            build["branch"] = branch_build["branch"]
            build["branch_build_source"] = C.track_rel(branch_build_path)

    a0_run = runs.get("A0")
    cancelled_a0 = None
    if a0_run is None and (runs_dir / "A0" / "CANCELLED.json").is_file():
        cancelled_a0 = _load(runs_dir / "A0" / "CANCELLED.json")
    same_round_a0 = bool(a0_run and a0_run.get("fixed_epoch_reports", {}).get("40"))
    hist_run = None
    hist_task_rows: list[dict[str, Any]] = []
    if not same_round_a0 and historical_a0_path is not None:
        if (historical_a0_path / "run.json").is_file():
            hist_run = _load(historical_a0_path / "run.json")
            hist_ap = float(hist_run["fixed_epoch_reports"]["40"]["ap"])
            if abs(hist_ap - float(C.HISTORICAL_A0["valid_ap40"])) > 1e-9:
                raise RuntimeError(
                    f"historical A0 run AP {hist_ap} != frozen reference {C.HISTORICAL_A0['valid_ap40']}"
                )
        task_csv = historical_a0_path / "per_task_ap.csv"
        if task_csv.is_file():
            with open(task_csv, newline="", encoding="utf-8") as fh:
                hist_task_rows = [r for r in csv.DictReader(fh) if str(r.get("epoch")) == "40"]
    if same_round_a0:
        a0_ap = float(a0_run["fixed_epoch_reports"]["40"]["ap"])
        a0_source = "same_round_A0"
    else:
        a0_ap = float(C.HISTORICAL_A0["valid_ap40"])
        a0_source = "historical_phaseC_seed0_A"

    # ---- epoch-40 comparison ------------------------------------------------
    comp_rows = [
        _arm_row(arm, runs[arm], a0_ap, a0_source, cancelled=cancelled_a0) for arm in arms
    ]
    # per-task deltas vs the reference A0
    a0_tasks = (a0_run or {}).get("fixed_epoch_reports", {}).get("40", {}).get("per_task_ap") or []
    if not a0_tasks and hist_task_rows:
        a0_tasks = [
            None if not r["ap"] else float(r["ap"]) for r in hist_task_rows
        ]
    for row in comp_rows:
        run = runs[row["arm"]]
        rep = (run or {}).get("fixed_epoch_reports", {}).get("40", {})
        tasks = rep.get("per_task_ap") or []
        if not tasks or not a0_tasks:
            continue
        improved = degraded = tied = 0
        for a, b in zip(tasks, a0_tasks):
            if a is None or b is None:
                continue
            d = float(a) - float(b)
            if d > 1e-12:
                improved += 1
            elif d < -1e-12:
                degraded += 1
            else:
                tied += 1
        row["tasks_improved"] = improved
        row["tasks_degraded"] = degraded
        row["tasks_tied"] = tied
    fields = list(comp_rows[0].keys())
    _write_csv(out_dir / "epoch40_comparison.csv", comp_rows, fields)

    # ---- per-task comparison ------------------------------------------------
    valid_pos = list(
        (a0_run or {}).get("fixed_epoch_reports", {}).get("40", {}).get("positive") or []
    )
    valid_obs = list(
        (a0_run or {}).get("fixed_epoch_reports", {}).get("40", {}).get("observed") or []
    )
    if not valid_pos and hist_task_rows:
        valid_pos = [int(r["positive"]) for r in hist_task_rows]
        valid_obs = [int(r["observed"]) for r in hist_task_rows]
    rows = []
    for t in range(C.N_TASKS):
        row: dict[str, Any] = {
            "task": t,
            "valid_observed": valid_obs[t] if t < len(valid_obs) else "",
            "valid_positive": valid_pos[t] if t < len(valid_pos) else "",
            "stratum": (
                "ge20" if t < len(valid_pos) and valid_pos[t] >= 20
                else "s5_19" if t < len(valid_pos) and valid_pos[t] >= 5
                else "lt5"
            ),
            "scorable": bool(t < len(a0_tasks) and a0_tasks[t] is not None),
        }
        for arm in arms:
            if arm == "A0":
                ap = a0_tasks[t] if t < len(a0_tasks) else None
            else:
                run = runs[arm]
                rep = (run or {}).get("fixed_epoch_reports", {}).get("40", {})
                tasks = rep.get("per_task_ap") or []
                ap = tasks[t] if t < len(tasks) else None
            row[f"ap_{arm}"] = "" if ap is None else round(float(ap), 9)
            if a0_ap is not None and ap is not None and t < len(a0_tasks) and a0_tasks[t] is not None:
                row[f"delta_{arm}"] = round(float(ap) - float(a0_tasks[t]), 9)
            else:
                row[f"delta_{arm}"] = ""
        rows.append(row)
    task_fields = ["task", "valid_observed", "valid_positive", "stratum", "scorable"]
    for arm in arms:
        task_fields.append(f"ap_{arm}")
        if arm != "A0":
            task_fields.append(f"delta_{arm}")
    _write_csv(out_dir / "per_task_comparison.csv", rows, task_fields)

    # ---- decision -----------------------------------------------------------
    deltas = {}
    for arm in C.ARM_ORDER[1:]:
        if runs.get(arm) and runs[arm].get("fixed_epoch_reports", {}).get("40"):
            deltas[arm] = float(runs[arm]["fixed_epoch_reports"]["40"]["ap"]) - float(a0_ap)
    eligible = sorted(
        [a for a, d in deltas.items() if d >= C.DECISION["combo_eligible_delta_min"]],
        key=lambda a: (-deltas[a], C.DECISION["tie_order"].index(a)),
    )
    combo_row = runs.get("COMBO")
    decision: dict[str, Any] = {
        "schema": "lcrc-pcba-explicit-performance-decision-v1",
        "round_id": C.ROUND_ID,
        "main_metric": C.DECISION["main_metric"],
        "a0_valid_ap40": a0_ap,
        "a0_reference": {
            "same_round": bool(same_round_a0),
            "source": a0_source,
            "historical_a0_valid_ap40": C.HISTORICAL_A0["valid_ap40"],
            "historical_a0_run": "lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A",
            "note": (
                "A0 was cancelled by principal instruction; deltas are versus the "
                "historical phase-C A run and therefore do not control same-round "
                "implementation drift"
            )
            if not same_round_a0
            else "same-round A0 control",
        },
        "historical_a0_valid_ap40": C.HISTORICAL_A0["valid_ap40"],
        "deltas_vs_a0": {k: round(v, 9) for k, v in deltas.items()},
        "eligible_delta_min": C.DECISION["combo_eligible_delta_min"],
        "strong_delta_min": C.DECISION["combo_strong_delta_min"],
        "eligible": eligible,
        "labels": {r["arm"]: r["label"] for r in comp_rows},
        "statuses": {r["arm"]: r["status"] for r in comp_rows},
        "combo": {
            "run_status": _status_of(combo_row, "COMBO"),
            "modules": [] if combo_row is None else list(combo_row.get("modules") or []),
        },
        "test_evaluated": False,
    }
    missing_arms = [
        a for a in C.ARM_ORDER[1:]
        if not (runs.get(a) or {}).get("fixed_epoch_reports", {}).get("40")
    ]
    if missing_arms:
        decision["combo"]["decision"] = "PENDING"
        decision["combo"]["reason"] = f"primary arms without an epoch-40 report: {missing_arms}"
    elif len(eligible) < C.DECISION["combo_min_eligible"]:
        decision["combo"]["decision"] = "SKIPPED_BY_RULE"
        decision["combo"]["reason"] = (
            f"only {len(eligible)} arm(s) with delta >= {C.DECISION['combo_eligible_delta_min']}; "
            f"need at least {C.DECISION['combo_min_eligible']}"
        )
    elif max(deltas.values()) < C.DECISION["combo_strong_delta_min"]:
        decision["combo"]["decision"] = "SKIPPED_BY_RULE"
        decision["combo"]["reason"] = (
            f"largest delta {max(deltas.values()):.6f} < {C.DECISION['combo_strong_delta_min']}"
        )
    else:
        chosen = eligible[:2]
        decision["combo"]["decision"] = "RUN_COMBO"
        decision["combo"]["chosen_modules"] = chosen
        decision["combo"]["reason"] = (
            f"two largest deltas {chosen} with max delta {max(deltas.values()):.6f} "
            f">= {C.DECISION['combo_strong_delta_min']}"
        )
    if combo_row is None:
        decision["labels"]["COMBO"] = decision["combo"]["decision"]
    C.write_json(out_dir / "decision.json", decision)

    # ---- architecture audit -------------------------------------------------
    arch: dict[str, Any] = {
        "schema": "lcrc-pcba-explicit-performance-architecture-v1",
        "round_id": C.ROUND_ID,
        "base": {
            "model": "frozen LCRC-C1 + original 4-head graph attention readout",
            "base_parameters": C.A0_PARAMETERS,
            "historical_a0_valid_ap40": C.HISTORICAL_A0["valid_ap40"],
            "historical_a0_parameters": C.HISTORICAL_A0["parameters"],
            "base_init_fingerprint": C.HISTORICAL_A0["base_init_fingerprint"],
        },
        "forbidden": [
            "learnable inter-atom message passing",
            "hidden-state update along bond_index",
            "inter-node QKV self-attention",
        ],
        "arms": {},
        "test_evaluated": False,
    }
    for arm in arms:
        run = runs.get(arm)
        modules = tuple(
            (run or {}).get("modules")
            or (() if arm in ("A0", "COMBO") else C.arm_modules(arm))
        )
        pf = (preflight.get("per_arm") or {}).get(arm, {})
        entry: dict[str, Any] = {
            "modules": list(modules),
            "module_names": [C.MODULES[m]["name"] for m in modules],
            "new_parameters_by_definition": sum(int(C.MODULES[m]["new_parameters"]) for m in modules),
            "expected_total_parameters": C.expected_parameters(modules),
            "status": (
                cancelled_a0["status"]
                if (arm == "A0" and cancelled_a0 is not None)
                else _status_of(run, arm)
            ),
            "same_round_control": bool(same_round_a0) if arm == "A0" else None,
        }
        if run is not None:
            entry["model_audit"] = run.get("model_audit")
            entry["parameters"] = int(run.get("parameters", 0))
            entry["extra_parameters"] = int(run.get("extra_parameters", 0))
            entry["feature_hashes"] = (run.get("features") or {}).get("hashes")
        if pf:
            entry["preflight_checks"] = {
                k: v.get("passed") for k, v in (pf.get("checks") or {}).items()
            }
            entry["preflight_all_passed"] = bool(pf.get("all_passed"))
            entry["flops"] = pf.get("flops")
        arch["arms"][arm] = entry
    C.write_json(out_dir / "architecture_audit.json", arch)

    # ---- feature audit ------------------------------------------------------
    feature_audit: dict[str, Any] = {
        "schema": "lcrc-pcba-explicit-performance-features-v1",
        "round_id": C.ROUND_ID,
        "test_evaluated": False,
        "a2": {},
        "a3": {},
    }
    if build:
        feature_audit["a2_a3_same_row_set"] = bool(
            (build.get("physchem") or {}).get("rows_attempted_sha256")
            and (build.get("physchem") or {}).get("rows_attempted_sha256")
            == (build.get("branch") or {}).get("rows_attempted_sha256")
        )
        a2 = (build.get("physchem") or {})
        if a2:
            feature_audit["a2"] = {
                "algorithm": a2.get("algorithm"),
                "rdkit_version": a2.get("rdkit_version"),
                "names": a2.get("names"),
                "rows_attempted": a2.get("rows_attempted"),
                "rows_attempted_sha256": a2.get("rows_attempted_sha256"),
                "built_rows": a2.get("built_rows"),
                "failed_rows": a2.get("failed_rows"),
                "coverage_attempted": a2.get("coverage_attempted"),
                "coverage_train": a2.get("coverage_train"),
                "coverage_valid": a2.get("coverage_valid"),
                "failures": a2.get("failures"),
                "missing_policy": a2.get("missing_policy"),
                "train_only_norm": a2.get("train_only_norm"),
                "finite": a2.get("finite"),
                "test_rows_built": a2.get("test_rows_built"),
                "cache_sha256": (a2.get("files") or {}).get("cache_sha256"),
                "source": a2.get("source"),
            }
        a3 = (build.get("branch") or {})
        if a3:
            feature_audit["a3"] = {
                "algorithm": a3.get("algorithm"),
                "symbol": a3.get("symbol"),
                "hash": a3.get("hash"),
                "n_buckets": a3.get("n_buckets"),
                "embedding_dim": a3.get("embedding_dim"),
                "rows_attempted": a3.get("rows_attempted"),
                "rows_attempted_sha256": a3.get("rows_attempted_sha256"),
                "n_tokens": a3.get("n_tokens"),
                "tokens_per_molecule_mean": a3.get("tokens_per_molecule_mean"),
                "max_tokens_per_atom": a3.get("max_tokens_per_atom"),
                "n_zero_degree_atoms_built": a3.get("n_zero_degree_atoms_built"),
                "collision_stats": a3.get("collision_stats"),
                "test_rows_built": a3.get("test_rows_built"),
                "cache_sha256": (a3.get("files") or {}).get("cache_sha256"),
                "build_source": build.get("branch_build_source"),
                "source": a3.get("source"),
            }
    pf_checks = {}
    for arm in ("A2", "A3"):
        pf = (preflight.get("per_arm") or {}).get(arm, {})
        key = "a2_physchem" if arm == "A2" else "a3_branch_tokens"
        if key in (pf.get("checks") or {}):
            pf_checks[arm] = pf["checks"][key]
    feature_audit["preflight_alignment"] = pf_checks
    C.write_json(out_dir / "feature_audit.json", feature_audit)

    # ---- preflight ----------------------------------------------------------
    if preflight:
        C.write_json(out_dir / "preflight.json", preflight)

    # ---- run registry -------------------------------------------------------
    registry: dict[str, Any] = {
        "schema": "lcrc-pcba-explicit-performance-registry-v1",
        "round_id": C.ROUND_ID,
        "host": "res (a100-2)",
        "runs": [],
        "test_evaluated": False,
    }
    for arm in arms:
        run = runs.get(arm)
        entry = {
            "arm": arm,
            "status": _status_of(run, arm),
            "arm_label": C.arm_label(arm, C.arm_modules(arm)) if arm != "COMBO" else "COMBO",
        }
        if arm == "A0" and cancelled_a0 is not None:
            entry.update({"status": cancelled_a0["status"], "run_id": cancelled_a0["run_id"],
                          "gpu_hours": cancelled_a0["gpu_hours"],
                          "epochs_completed": cancelled_a0["epochs_completed"],
                          "same_round_control": False,
                          "note": cancelled_a0["reason"]})
        if rr_status:
            statuses = (rr_status.get("runs") or {}).get(arm)
            if statuses:
                entry.update(statuses)
        if run is not None:
            entry.update(
                {
                    "commit": run.get("git_commit"),
                    "device": (run.get("environment") or {}).get("cuda_device_name"),
                    "driver_version": (
                        entry.get("driver_version")
                        or (run.get("environment") or {}).get("driver_version")
                    ),
                    "cuda_visible_devices": entry.get("cuda_visible_devices"),
                    "torch": (run.get("environment") or {}).get("torch"),
                    "torch_cuda": (run.get("environment") or {}).get("torch_cuda"),
                    "gpu_hours": round(float(run.get("gpu_hours", 0.0)), 6),
                    "started_utc": run.get("started_utc"),
                    "finished_utc": run.get("finished_utc"),
                    "stop_reason": run.get("stop_reason"),
                    "parameters": int(run.get("parameters", 0)),
                    "extra_parameters": int(run.get("extra_parameters", 0)),
                    "batch_plan_sha256": (run.get("batch_plan") or {}).get("global_plan_sha256"),
                    "checkpoint_epoch40_sha256": run.get("checkpoint_epoch40_sha256"),
                    "feature_hashes": (run.get("features") or {}).get("hashes"),
                    "eval_epochs_completed": run.get("eval_epochs_completed"),
                    "test_evaluated": False,
                    "run_dir": C.track_rel(runs_dir / arm),
                }
            )
        registry["runs"].append(entry)
    C.write_json(out_dir / "run_registry.json", registry)

    # ---- GPU ledger ---------------------------------------------------------
    prep_gpu = 0.0
    preflight_gpu = float(preflight.get("gpu_hours") or 0.0) if preflight else 0.0
    formal = {
        arm: round(float((runs.get(arm) or {}).get("gpu_hours") or 0.0), 6) for arm in arms
    }
    smoke_run = _load(smoke_path) if smoke_path is not None and smoke_path.is_file() else None
    smoke_gpu = float((smoke_run or {}).get("gpu_hours") or 0.0)
    # The first GPU preflight attempt died on a device-comparison bug before any
    # training step; its cost is bounded by the 13 s of launcher wall time.
    failed_attempt_gpu = 0.001
    cancelled_gpu = float((cancelled_a0 or {}).get("gpu_hours") or 0.0)
    total = (
        preflight_gpu + sum(formal.values()) + prep_gpu + smoke_gpu
        + failed_attempt_gpu + cancelled_gpu
    )
    ledger = {
        "schema": "lcrc-pcba-explicit-performance-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "budget": C.DECISION["budget"],
        "preflight": {
            "status": preflight.get("status"),
            "gpu_hours": round(preflight_gpu, 6),
            "per_arm_all_passed": preflight.get("all_passed"),
        },
        "cpu_prep": {
            "note": "feature cache build runs on CPU; counted as 0 A100 GPU-hours",
            "gpu_hours": 0.0,
        },
        "formal_runs": formal,
        "non_formal": {
            "smoke_1_epoch_a0": {
                "gpu_hours": round(smoke_gpu, 6),
                "status": None if smoke_run is None else smoke_run.get("status"),
                "note": "1-epoch A0 end-to-end trainer smoke test (not a result)",
            },
            "cancelled_A0_attempt": {
                "gpu_hours": round(cancelled_gpu, 6),
                "epochs_completed": None if cancelled_a0 is None else cancelled_a0.get("epochs_completed"),
                "note": "A0 rerun cancelled by principal instruction; historical A0 used as reference",
            },
            "preflight_failed_attempt": {
                "gpu_hours": failed_attempt_gpu,
                "note": "device-comparison bug before any training step; bound from 13 s launcher wall time",
            },
        },
        "gpu_hours_total": round(total, 6),
        "remaining_to_target": round(C.DECISION["budget"]["gpu_target_hours"] - total, 6),
        "remaining_to_hard": round(C.DECISION["budget"]["gpu_hard_hours"] - total, 6),
        "within_target": bool(total <= C.DECISION["budget"]["gpu_target_hours"]),
        "within_hard": bool(total <= C.DECISION["budget"]["gpu_hard_hours"]),
        "note": "formal run gpu_hours are process wall seconds / 3600 from run.json",
    }
    C.write_json(out_dir / "gpu_ledger.json", ledger)

    summary = {
        "a0_ap40": a0_ap,
        "rows": comp_rows,
        "decision": decision,
        "gpu_hours_total": total,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--preflight", default="")
    parser.add_argument("--build", default="")
    parser.add_argument("--build-branch", default="")
    parser.add_argument("--rr-status", default="")
    parser.add_argument("--smoke", default="")
    parser.add_argument("--historical-a0", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rr_status = _load(Path(args.rr_status)) if args.rr_status else None
    summary = analyze(
        Path(args.runs_dir),
        preflight_path=Path(args.preflight) if args.preflight else Path(""),
        build_path=Path(args.build) if args.build else Path(""),
        branch_build_path=Path(args.build_branch) if args.build_branch else None,
        out_dir=Path(args.out),
        rr_status=rr_status,
        smoke_path=Path(args.smoke) if args.smoke else None,
        historical_a0_path=Path(args.historical_a0) if args.historical_a0 else None,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, default=str))
    for row in summary["rows"]:
        print(
            f"{row['arm']:6s} {row['status']:18s} AP40={row['valid_ap40']} "
            f"delta={row['delta_vs_A0']} {row['label']} params={row['parameters']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
