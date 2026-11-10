"""Frozen best-Validation mechanism diagnostics and the round decision.

Everything here runs on **official Validation only**:

* ``real``               -- the trained model as frozen;
* ``delta_zero``         -- the identical W2 readout with the RHIC residual
                            removed (``h = h0``), i.e. the untrained-RHIC
                            ablation at the *trained* W2 weights;
* ``branch_assignment_shuffle`` -- same path tokens, same root membership, same
                            per-root branch sizes, but the assignment of paths
                            to branches inside every root is permuted (seed 0).

``delta_zero`` and the shuffle are out-of-distribution interventions on a
jointly trained network: they measure how much the trained predictions depend
on the learned branch structure, not an independent causal generalisation gain.

The script then merges the frozen Test scores (already sealed and scored) with
the Validation-only mechanism gate and writes ``analysis/decision.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import rhic_common as C  # noqa: E402
import rhic_model as M  # noqa: E402
import rhic_test as T  # noqa: E402

P = C.P


def _mean(values: Sequence[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _set_mean(per_task: Sequence[Any], ids: Sequence[int]) -> float | None:
    return _mean([per_task[i] for i in ids])


def _wins(a: Sequence[Any], b: Sequence[Any], ids: Sequence[int]) -> tuple[int, int, int]:
    return T._wins(a, b, ids)


def _probs(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def _report(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probs = _probs(logits).astype(np.float32)
    rep = C.MET.validation_report(y, probs)
    rep["valid_bce"] = float(C.MET.masked_bce_numpy(logits, y))
    return rep


def _aggregate_usage(model: M.RHICModel, task: Any, rows: np.ndarray, device: torch.device, *, batch_size: int = 128) -> dict[str, Any]:
    """Full-Validation aggregate of the residual / group statistics (eval mode)."""
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    acc = {
        "n_atoms": 0,
        "sum_h0_sq": 0.0,
        "sum_delta_sq": 0.0,
        "sum_h_sq": 0.0,
        "sum_branch_mu_sq": 0.0,
        "sum_branch_q_sq": 0.0,
        "sum_branch_value_sq": 0.0,
        "sum_mean_branch_sq": 0.0,
        "sum_cross_branch_sq": 0.0,
        "sum_z_sq": 0.0,
        "sum_v_sq": 0.0,
        "branch_active": 0,
        "branch_total": 0,
        "branches_multi_path": 0,
        "q_nonzero": 0,
        "cross_nonzero": 0,
        "roots": 0,
        "roots_ge2_branches": 0,
        "roots_with_multi_path_branch": 0,
        "n_paths": 0,
        "n_branches": 0,
        "gate_sum": 0.0,
        "gate_sq": 0.0,
        "gate_n": 0,
        "delta_max": 0.0,
    }
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = M.to_device(P.collate_with_cpc(task, task.cpc, chunk.tolist()), device)
            _logits, aux = model(batch, return_aux=True)
            r = aux["rhic"]
            h0, delta, h = aux["h0"], aux["delta"], aux["h"]
            acc["n_atoms"] += int(h0.shape[0])
            acc["sum_h0_sq"] += float(h0.double().pow(2).sum().item())
            acc["sum_delta_sq"] += float(delta.double().pow(2).sum().item())
            acc["sum_h_sq"] += float(h.double().pow(2).sum().item())
            acc["delta_max"] = max(acc["delta_max"], float(delta.abs().max().item()))
            for key, tensor in (
                ("sum_branch_mu_sq", r["branch_mu"]),
                ("sum_branch_q_sq", r["branch_q"]),
                ("sum_branch_value_sq", r["branch_value"]),
                ("sum_mean_branch_sq", r["mean_branch"]),
                ("sum_cross_branch_sq", r["cross_branch"]),
                ("sum_z_sq", r["z"]),
                ("sum_v_sq", r["v"]),
            ):
                acc[key] += float(tensor.double().pow(2).sum().item())
            counts = r["branch_counts"]
            active = r["branch_active"]
            acc["branch_active"] += int(active.sum().item())
            acc["branch_total"] += int(counts.numel())
            acc["branches_multi_path"] += int((counts[active] >= 2).sum().item())
            acc["q_nonzero"] += int((r["branch_q"].abs().sum(dim=1) > 0).sum().item())
            acc["cross_nonzero"] += int((r["cross_branch"].abs().sum(dim=1) > 0).sum().item())
            d = r["root_branch_count"]
            pc = r["root_path_count"]
            roots = d > 0
            acc["roots"] += int(roots.sum().item())
            acc["roots_ge2_branches"] += int((d >= 2).sum().item())
            multi = torch.zeros_like(d, dtype=torch.bool)
            if int(r["active_ids"].numel()):
                per_root = torch.zeros_like(d)
                per_root.index_add_(
                    0,
                    r["branch_root"],
                    (r["branch_n"] >= 2).to(d.dtype),
                )
                multi = per_root >= 1
            acc["roots_with_multi_path_branch"] += int(multi.sum().item())
            acc["n_paths"] += int(r["n_paths"])
            acc["n_branches"] += int(r["n_branches"])
            gate = r["gate"].reshape(-1)
            acc["gate_sum"] += float(gate.double().sum().item())
            acc["gate_sq"] += float(gate.double().pow(2).sum().item())
            acc["gate_n"] += int(gate.numel())

    def rms(sq: float, n: int) -> float:
        return float(np.sqrt(sq / n)) if n else 0.0

    n = acc["n_atoms"] or 1
    branch_n = acc["branch_active"] or 1
    gate_n = acc["gate_n"] or 1
    out = {
        "rows": int(len(rows)),
        "n_atoms": int(acc["n_atoms"]),
        "n_paths": int(acc["n_paths"]),
        "n_branches": int(acc["n_branches"]),
        "rms_h0": rms(acc["sum_h0_sq"], n),
        "rms_delta": rms(acc["sum_delta_sq"], n),
        "rms_h": rms(acc["sum_h_sq"], n),
        "delta_abs_max": float(acc["delta_max"]),
        "rms_branch_mu": rms(acc["sum_branch_mu_sq"], branch_n),
        "rms_branch_q": rms(acc["sum_branch_q_sq"], branch_n),
        "rms_branch_value": rms(acc["sum_branch_value_sq"], branch_n),
        "rms_mean_branch": rms(acc["sum_mean_branch_sq"], n),
        "rms_cross_branch": rms(acc["sum_cross_branch_sq"], n),
        "rms_z": rms(acc["sum_z_sq"], int(acc["n_paths"]) or 1),
        "rms_v": rms(acc["sum_v_sq"], int(acc["n_paths"]) or 1),
        "branch_active_fraction": float(acc["branch_active"] / max(acc["branch_total"], 1)),
        "multi_path_branch_fraction": float(acc["branches_multi_path"] / branch_n),
        "q_nonzero_fraction": float(acc["q_nonzero"] / branch_n),
        "cross_nonzero_fraction": float(acc["cross_nonzero"] / n),
        "root_fraction_ge2_branches": float(acc["roots_ge2_branches"] / max(acc["roots"], 1)),
        "root_fraction_multi_path_branch": float(acc["roots_with_multi_path_branch"] / max(acc["roots"], 1)),
        "gate_mean": float(acc["gate_sum"] / gate_n),
        "gate_std": float(np.sqrt(max(acc["gate_sq"] / gate_n - (acc["gate_sum"] / gate_n) ** 2, 0.0))),
    }
    out["R"] = float(out["rms_delta"] / out["rms_h0"]) if out["rms_h0"] else 0.0
    return out


def run_diagnostics(
    *,
    run_path: Path,
    out_path: Path,
    usage_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    run = C.read_json(run_path)
    if run.get("status") != "COMPLETED" or run.get("test_evaluated"):
        raise RuntimeError("run must be COMPLETED and Test-free")
    if int(run["epochs_run"]) != int(C.TOTAL_EPOCHS):
        raise RuntimeError("diagnostics require the completed 40-epoch run")
    best_path = T._abs_path(run["best_checkpoint"])
    if C.sha256_file(best_path) != run["best_checkpoint_sha256"]:
        raise RuntimeError("best checkpoint hash changed")
    task = C.SD.load_task()
    task.cpc = C.ensure_cpc_cache(task)
    if np.isfinite(np.asarray(task.labels[task.test_rows])).any():
        raise RuntimeError("official Test labels are visible; diagnostics must be Validation-only")
    model, payload, sha = T._load_checkpoint(best_path, task, device)
    if int(payload["epoch"]) != int(run["best_epoch"]):
        raise RuntimeError("checkpoint epoch != run best_epoch")
    y = np.asarray(task.labels[task.valid_rows], dtype=np.float64)
    valid_rows = np.asarray(task.valid_rows, dtype=np.int64)

    real = M.predict_logits(model, task, valid_rows, device, batch_size=256, mode="real")
    zero = M.predict_logits(model, task, valid_rows, device, batch_size=256, mode="delta_zero")
    shuffle = M.predict_logits(
        model, task, valid_rows, device, batch_size=256, mode="branch_assignment_shuffle", shuffle_seed=0
    )
    rep_real = _report(y, real)
    rep_zero = _report(y, zero)
    rep_shuffle = _report(y, shuffle)
    usage = _aggregate_usage(model, task, valid_rows, device)

    def compare(a: np.ndarray, b: np.ndarray, rep_a: Mapping[str, Any], rep_b: Mapping[str, Any]) -> dict[str, Any]:
        da = np.asarray(rep_a["per_task_ap"], dtype=object)
        db = np.asarray(rep_b["per_task_ap"], dtype=object)
        up, down, both = T._wins(list(da), list(db))
        return {
            "valid_ap": float(rep_a["ap"]),
            "valid_ap_reference": float(rep_b["ap"]),
            "valid_ap_delta": float(rep_a["ap"] - rep_b["ap"]),
            "valid_bce": float(rep_a["valid_bce"]),
            "valid_bce_reference": float(rep_b["valid_bce"]),
            "mean_abs_logit_delta": float(np.mean(np.abs(a - b))),
            "max_abs_logit_delta": float(np.max(np.abs(a - b))),
            "mean_abs_prob_delta": float(np.mean(np.abs(_probs(a) - _probs(b)))),
            "logit_pearson": float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1]),
            "improved_tasks": int(up),
            "degraded_tasks": int(down),
            "compared_tasks": int(both),
            "changed_entries_fraction": float(np.mean(np.abs(a - b) > 1e-7)),
        }

    vs_zero = compare(real, zero, rep_real, rep_zero)
    vs_shuffle = compare(real, shuffle, rep_real, rep_shuffle)

    checks = {
        "trained_residual_nonzero": bool(usage["delta_abs_max"] > 0.0 and usage["R"] > 0.005),
        "residual_scale_reasonable": bool(0.005 <= usage["R"] <= 2.0),
        "branch_groups_active": bool(usage["branch_active_fraction"] > 0.5),
        "multi_path_branches_exist": bool(usage["multi_path_branch_fraction"] > 0.1),
        "branch_q_nonzero": bool(usage["q_nonzero_fraction"] > 0.1),
        "cross_branch_nonzero": bool(usage["cross_nonzero_fraction"] > 0.1),
        "delta_changes_validation_predictions": bool(vs_zero["mean_abs_logit_delta"] > 1e-6),
        "grouping_changes_validation_predictions": bool(vs_shuffle["mean_abs_logit_delta"] > 1e-6),
    }
    mechanism_valid = bool(all(checks.values()))
    payload_out = {
        "schema": C.SCHEMA_MECHANISM,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "split": "official Validation only",
        "test_rows_read": False,
        "checkpoint": C.track_rel(best_path),
        "checkpoint_sha256": sha,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "parameters": int(payload["parameters"]),
        "interventions": dict(C.MECHANISM),
        "protocol": {
            "real": "the frozen model as trained",
            "delta_zero": "identical W2 readout with h = h0 (RHIC residual removed)",
            "branch_assignment_shuffle": "within-root permutation of path -> branch assignment, seed 0",
            "note": C.MECHANISM["note"],
        },
        "valid": {
            "real": {k: rep_real[k] for k in ("ap", "valid_bce", "n_tasks_scored", "n_tasks_with_observed_labels", "per_task_ap")},
            "delta_zero": {k: rep_zero[k] for k in ("ap", "valid_bce", "n_tasks_scored", "n_tasks_with_observed_labels")},
            "branch_assignment_shuffle": {
                k: rep_shuffle[k] for k in ("ap", "valid_bce", "n_tasks_scored", "n_tasks_with_observed_labels")
            },
        },
        "real_vs_delta_zero": vs_zero,
        "real_vs_branch_shuffle": vs_shuffle,
        "usage_full_validation": usage,
        "checks": checks,
        "mechanism_valid": mechanism_valid,
        "logits_sha256": {
            "real": T._np_sha(real),
            "delta_zero": T._np_sha(zero),
            "branch_assignment_shuffle": T._np_sha(shuffle),
        },
        "valid_ap_replay_vs_run": float(rep_real["ap"] - float(run["best_valid_ap"])),
    }
    if abs(payload_out["valid_ap_replay_vs_run"]) > 1e-4:
        raise RuntimeError("Validation AP does not replay the run record")
    C.write_json(out_path, payload_out)
    C.write_json(
        usage_path,
        {
            "schema": "lcrc-pcba-rhic-usage-v1",
            "round_id": C.ROUND_ID,
            "checkpoint": C.track_rel(best_path),
            "checkpoint_sha256": sha,
            "split": "official Validation only",
            "usage": usage,
            "note": "full-Validation aggregate, not the per-epoch 32-row diagnostic batch",
        },
    )
    C.log(
        f"[mechanism] valid real {rep_real['ap']:.9f} zero {rep_zero['ap']:.9f} "
        f"shuffle {rep_shuffle['ap']:.9f} R {usage['R']:.4f} valid={mechanism_valid}"
    )
    return payload_out


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def run_decision(
    *,
    run_path: Path,
    mechanism_path: Path,
    scores_path: Path,
    out_path: Path,
    per_task_path: Path,
) -> dict[str, Any]:
    run = C.read_json(run_path)
    mechanism = C.read_json(mechanism_path)
    scores = C.read_json(scores_path)
    results = scores["results"]
    rhic_best = results["LCRC_RHIC_best_valid"]
    rhic_40 = results["LCRC_RHIC_epoch40"]
    cpc_best = results["LCRC_CPC_best_valid"]
    high_ids = list(C.HIGH_SUPPORT_TASKS)
    low_ids = list(C.LOW_SUPPORT_TASKS)

    valid_best = float(run["best_valid_ap"])
    valid_40 = float(run["fixed_epoch_reports"]["40"]["ap"])
    valid_per_task = run["fixed_epoch_reports"][str(run["best_epoch"])]["per_task_ap"]
    reference_runs = {
        "W2": C.REPO / C.W2_SEED0["run_json"],
        "RC-REAL": C.REPO / "tracks/structural-sufficiency/lcrc_pcba_rc_v1/runs/REAL/run.json",
        "CPC": C.CPC["test_scores"].parent.parent / "runs/LCRC_CPC/run.json",
        "W2-DYNAMIC": C.REPO / "tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/runs/W2_DYNAMIC/run.json",
    }
    reference_valid = {}
    for name, path in reference_runs.items():
        ref_run = C.read_json(path)
        reference_valid[name] = ref_run["fixed_epoch_reports"][str(ref_run["best_epoch"])]["per_task_ap"]
    cpc_valid_per_task = reference_valid["CPC"]

    high_valid_rhic = _set_mean(valid_per_task, high_ids)
    high_valid_cpc = _set_mean(cpc_valid_per_task, high_ids)
    low_valid_rhic = _set_mean(valid_per_task, low_ids)
    low_valid_cpc = _set_mean(cpc_valid_per_task, low_ids)
    high_test_rhic = _set_mean(rhic_best["per_task_ap"], high_ids)
    high_test_cpc = _set_mean(cpc_best["per_task_ap"], high_ids)
    low_test_rhic = _set_mean(rhic_best["per_task_ap"], low_ids)
    low_test_cpc = _set_mean(cpc_best["per_task_ap"], low_ids)
    up_high, down_high, both_high = _wins(rhic_best["per_task_ap"], cpc_best["per_task_ap"], high_ids)
    up_high_valid, down_high_valid, both_high_valid = _wins(valid_per_task, cpc_valid_per_task, high_ids)
    if both_high != 106 or both_high_valid != 106:
        raise RuntimeError("high-support comparison does not cover 106 tasks")

    high_delta_test = float(high_test_rhic - high_test_cpc)
    high_delta_valid = float(high_valid_rhic - high_valid_cpc)
    mechanism_valid = bool(mechanism["mechanism_valid"])
    label = T.choose_label(
        valid_best=valid_best,
        test_best=float(rhic_best["official_ap"]),
        high_delta_vs_cpc=high_delta_test,
        improved_high_vs_cpc=int(up_high),
        mechanism_valid=mechanism_valid,
    )

    # per-task table with strata and support
    strata = {
        t: ("support_lt5" if int(rhic_best["observed"][t]) < 5 else "support_5_19" if int(rhic_best["observed"][t]) < 20 else "support_ge20")
        for t in range(C.N_TASKS)
    }
    rows = []
    for t in range(C.N_TASKS):
        rows.append(
            {
                "task": int(t),
                "high_support": bool(t in C.HIGH_SUPPORT_SET),
                "low_support": bool(t in C.LOW_SUPPORT_SET),
                "stratum": strata[t],
                "test_observed": int(rhic_best["observed"][t]),
                "test_positive": int(rhic_best["positive"][t]),
                "valid_positive": int(run["task_counts"]["valid"]["positive"][t]),
                "valid_ap_rhic": valid_per_task[t],
                "valid_ap_cpc": cpc_valid_per_task[t],
                "test_ap_rhic": rhic_best["per_task_ap"][t],
                "test_ap_rhic_40": rhic_40["per_task_ap"][t],
                "test_ap_cpc": cpc_best["per_task_ap"][t],
                "test_delta_vs_cpc": None
                if rhic_best["per_task_ap"][t] is None or cpc_best["per_task_ap"][t] is None
                else float(rhic_best["per_task_ap"][t] - cpc_best["per_task_ap"][t]),
            }
        )
    C.write_json(per_task_path, {"schema": "lcrc-pcba-rhic-per-task-v1", "tasks": rows})

    by_stratum = {}
    for name in ("support_lt5", "support_5_19", "support_ge20"):
        ids = [r["task"] for r in rows if r["stratum"] == name]
        by_stratum[name] = {
            "tasks": len(ids),
            "valid_ap_rhic": _set_mean(valid_per_task, ids),
            "valid_ap_cpc": _set_mean(cpc_valid_per_task, ids),
            "test_ap_rhic": _set_mean(rhic_best["per_task_ap"], ids),
            "test_ap_cpc": _set_mean(cpc_best["per_task_ap"], ids),
        }

    decision = {
        "schema": C.SCHEMA_DECISION,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "model": C.MODEL_NAME,
        "label": label,
        "mechanism_valid": mechanism_valid,
        "mechanism_invalid_reasons": [k for k, v in mechanism["checks"].items() if not v],
        "rules": dict(C.DECISION_RULES),
        "observed": {
            "parameters": int(run["parameters"]),
            "parameter_delta_vs_w2": int(run["parameters"] - C.W2_SEED0["parameters"]),
            "parameter_delta_vs_cpc": int(run["parameters"] - C.CPC["parameters"]),
            "best_epoch": int(run["best_epoch"]),
            "valid_best_ap": valid_best,
            "valid_ap_at_40": valid_40,
            "test_best_ap": float(rhic_best["official_ap"]),
            "test_ap_at_40": float(rhic_40["official_ap"]),
            "train_ap_40": None if not run.get("train_eval_final") else float(run["train_eval_final"]["ap"]),
            "train_bce_40": None if not run.get("train_eval_final") else float(run["train_eval_final"]["bce"]),
            "valid_bce_40": float(run["fixed_epoch_reports"]["40"]["valid_bce"]),
            "gpu_hours": float(run["gpu_hours"]),
            "gpu_hours_vs_w2": float(run["gpu_hours"] - C.W2_SEED0["gpu_hours"]),
            "gpu_hours_vs_cpc": float(run["gpu_hours"] - C.CPC["gpu_hours"]),
            "macro_test_tasks_scored": int(rhic_best["n_tasks_scored"]),
            "test_ap_official_vs_independent_abs_delta": float(rhic_best["abs_delta_official_vs_independent"]),
            "high_support_tasks": 106,
            "low_support_tasks": 22,
            "high_support_test_ap_rhic": high_test_rhic,
            "high_support_test_ap_cpc": high_test_cpc,
            "high_support_test_ap_w2": _set_mean(results["W2_best_valid"]["per_task_ap"], high_ids),
            "high_support_test_ap_dynamic": _set_mean(results["W2_DYNAMIC_best_valid"]["per_task_ap"], high_ids),
            "high_support_test_delta_vs_cpc": high_delta_test,
            "high_support_test_delta_vs_w2": float(
                high_test_rhic - _set_mean(results["W2_best_valid"]["per_task_ap"], high_ids)
            ),
            "high_support_test_delta_vs_dynamic": float(
                high_test_rhic - _set_mean(results["W2_DYNAMIC_best_valid"]["per_task_ap"], high_ids)
            ),
            "high_support_valid_ap_rhic": high_valid_rhic,
            "high_support_valid_ap_cpc": high_valid_cpc,
            "high_support_valid_delta_vs_cpc": high_delta_valid,
            "high_support_valid_ap_w2": _set_mean(reference_valid["W2"], high_ids),
            "high_support_valid_ap_rc": _set_mean(reference_valid["RC-REAL"], high_ids),
            "high_support_valid_ap_dynamic": _set_mean(reference_valid["W2-DYNAMIC"], high_ids),
            "high_support_test_ap_rc": _set_mean(results["RC_REAL_best_valid"]["per_task_ap"], high_ids),
            "improved_high_support_tasks_test_vs_cpc": int(up_high),
            "degraded_high_support_tasks_test_vs_cpc": int(down_high),
            "improved_high_support_tasks_valid_vs_cpc": int(up_high_valid),
            "degraded_high_support_tasks_valid_vs_cpc": int(down_high_valid),
            "low_support_test_ap_rhic": low_test_rhic,
            "low_support_test_ap_cpc": low_test_cpc,
            "low_support_valid_ap_rhic": low_valid_rhic,
            "low_support_valid_ap_cpc": low_valid_cpc,
            "test_best_minus_cpc": float(rhic_best["official_ap"] - cpc_best["official_ap"]),
            "test_best_minus_w2": float(rhic_best["official_ap"] - results["W2_best_valid"]["official_ap"]),
            "test_best_minus_rc": float(rhic_best["official_ap"] - results["RC_REAL_best_valid"]["official_ap"]),
            "test_best_minus_dynamic": float(
                rhic_best["official_ap"] - results["W2_DYNAMIC_best_valid"]["official_ap"]
            ),
            "valid_best_minus_cpc": float(valid_best - C.CPC["best_valid_ap"]),
            "valid_best_minus_w2": float(valid_best - C.W2_SEED0["best_valid_ap"]),
            "valid_best_minus_dynamic": float(valid_best - C.DYNAMIC["best_valid_ap"]),
            "mechanism": {
                "valid_ap_real": float(mechanism["valid"]["real"]["ap"]),
                "valid_ap_delta_zero": float(mechanism["valid"]["delta_zero"]["ap"]),
                "valid_ap_branch_shuffle": float(mechanism["valid"]["branch_assignment_shuffle"]["ap"]),
                "real_minus_delta_zero": float(mechanism["real_vs_delta_zero"]["valid_ap_delta"]),
                "real_minus_branch_shuffle": float(mechanism["real_vs_branch_shuffle"]["valid_ap_delta"]),
                "mean_abs_logit_delta_vs_zero": float(mechanism["real_vs_delta_zero"]["mean_abs_logit_delta"]),
                "mean_abs_logit_delta_vs_shuffle": float(
                    mechanism["real_vs_branch_shuffle"]["mean_abs_logit_delta"]
                ),
                "R": float(mechanism["usage_full_validation"]["R"]),
                "q_nonzero_fraction": float(mechanism["usage_full_validation"]["q_nonzero_fraction"]),
                "cross_nonzero_fraction": float(
                    mechanism["usage_full_validation"]["cross_nonzero_fraction"]
                ),
            },
        },
        "by_stratum": by_stratum,
        "per_task_artifact": C.track_rel(per_task_path),
        "test_scores": C.track_rel(scores_path),
        "test_scores_sha256": C.sha256_file(scores_path),
        "mechanism_artifact": C.track_rel(mechanism_path),
        "note": (
            "Decision uses official Validation for selection/mechanism and the exposed official "
            "Test split only as the pre-registered engineering comparison. Test was never a "
            "selection signal."
        ),
    }
    C.write_json(out_path, decision)
    C.log(
        f"[decision] {label} | valid {valid_best:.6f} test {rhic_best['official_ap']:.6f} "
        f"high-support d(Test) {high_delta_test:+.6f} ({up_high}/106 up) mechanism={mechanism_valid}"
    )
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mechanism", "decision", "all"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", default=str(C.RUNS / "LCRC_RHIC/run.json"))
    parser.add_argument("--mechanism", default=str(C.ANALYSIS / "mechanism.json"))
    parser.add_argument("--usage", default=str(C.ANALYSIS / "usage_full_valid.json"))
    parser.add_argument("--scores", default=str(C.ANALYSIS / "test_scores.json"))
    parser.add_argument("--decision", default=str(C.ANALYSIS / "decision.json"))
    parser.add_argument("--per-task", default=str(C.ANALYSIS / "per_task.json"))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.mode in ("mechanism", "all"):
        run_diagnostics(
            run_path=Path(args.run),
            out_path=Path(args.mechanism),
            usage_path=Path(args.usage),
            device=device,
        )
    if args.mode in ("decision", "all"):
        run_decision(
            run_path=Path(args.run),
            mechanism_path=Path(args.mechanism),
            scores_path=Path(args.scores),
            out_path=Path(args.decision),
            per_task_path=Path(args.per_task),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
