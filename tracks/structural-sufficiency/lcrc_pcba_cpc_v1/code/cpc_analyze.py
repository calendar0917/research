"""Validation-only mechanism diagnostics and the pre-registered decision summary.

``mechanism`` runs after the formal 40-epoch model is frozen for checkpoint
selection and before any Test label is opened.  It evaluates only the official
Validation split: the real CPC path, ``delta`` forced to zero, and the
middle-endpoint pairing shuffled within each root while root/path-count/bond
multisets are held fixed.  It also computes whole-Validation branch usage.

``decision`` runs after the single post-freeze Test scoring pass.  It writes the
per-task/support-stratified analysis and the one pre-registered label.  It does
not alter the model, checkpoint, threshold or training plan.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import cpc_common as C  # noqa: E402
import cpc_model as M  # noqa: E402
import cpc_paths as P  # noqa: E402

BATCH = 256


def _abs_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else C.REPO / path


def load_run() -> dict[str, Any]:
    return C.read_json(C.RUNS / "LCRC_CPC" / "run.json")


def load_best(run: Mapping[str, Any], task: Any, device: torch.device):
    path = _abs_path(run["best_checkpoint"])
    if C.sha256_file(path) != run["best_checkpoint_sha256"]:
        raise RuntimeError("best checkpoint hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model, _audit = M.build_cpc_model(
        task.spec, task.norm, C.N_TASKS, seed=int(payload["seed"]), device=device
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload, path


def _probs(logits: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).astype(np.float32)


def report(logits: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    rep = C.MET.validation_report(y, _probs(logits))
    rep["valid_bce"] = float(C.MET.masked_bce_numpy(logits, y))
    return rep


def logits_change(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = b - a
    return {
        "rms_reference": float(np.sqrt((a * a).mean())),
        "rms_difference": float(np.sqrt((diff * diff).mean())),
        "relative_rms_change": float(np.sqrt((diff * diff).mean()) / np.sqrt((a * a).mean())),
        "mean_abs_change": float(np.abs(diff).mean()),
        "max_abs_change": float(np.abs(diff).max()),
        "fraction_entries_changed_gt_1e_6": float((np.abs(diff) > 1e-6).mean()),
    }


def full_validation_usage(model: M.CPCModel, task: Any, device: torch.device) -> dict[str, Any]:
    sums = {"h0": 0.0, "delta": 0.0, "z": 0.0, "context": 0.0}
    counts_elems = 0
    gate_parts: list[np.ndarray] = []
    path_total = 0
    zero_path_total = 0
    atom_total = 0
    finite_bad = 0
    with torch.inference_mode():
        for start in range(0, len(task.valid_rows), BATCH):
            rows = task.valid_rows[start : start + BATCH]
            batch = P.collate_with_cpc(task, task.cpc, rows.tolist())
            batch = M.to_device(batch, device)
            logits, aux = model(batch, return_aux=True)
            h0 = aux["h0"]
            delta = aux["delta"]
            cpc = aux["cpc"]
            n = int(h0.numel())
            counts_elems += n
            sums["h0"] += float(h0.detach().float().pow(2).sum().item())
            sums["delta"] += float(delta.detach().float().pow(2).sum().item())
            if cpc.get("z") is not None:
                sums["z"] += float(cpc["z"].detach().float().pow(2).sum().item())
            if cpc.get("context") is not None:
                sums["context"] += float(cpc["context"].detach().float().pow(2).sum().item())
            if cpc.get("gate") is not None and cpc["gate"].numel():
                gate_parts.append(cpc["gate"].detach().float().reshape(-1).cpu().numpy())
            path_total += int(cpc.get("n_paths", 0))
            atom_total += int(h0.shape[0])
            zero_path_total += int(cpc.get("zero_path_atoms", 0))
            finite_bad += int((~torch.isfinite(aux["h_cpc"])).sum().item())
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("non-finite Validation logits")
    gate = np.concatenate(gate_parts) if gate_parts else np.zeros(0, dtype=np.float32)
    out = {
        "rms_h0": float(np.sqrt(sums["h0"] / counts_elems)),
        "rms_delta": float(np.sqrt(sums["delta"] / counts_elems)),
        "R": float(np.sqrt(sums["delta"] / sums["h0"])),
        "rms_z": float(np.sqrt(sums["z"] / counts_elems)),
        "rms_context": float(np.sqrt(sums["context"] / counts_elems)),
        "n_atoms": int(atom_total),
        "n_paths": int(path_total),
        "paths_per_atom_mean": float(path_total / atom_total),
        "zero_path_atoms": int(zero_path_total),
        "zero_path_fraction": float(zero_path_total / atom_total),
        "nonfinite_h_cpc": int(finite_bad),
        "gate_mean": float(gate.mean()) if gate.size else None,
        "gate_std": float(gate.std()) if gate.size else None,
        "gate_min": float(gate.min()) if gate.size else None,
        "gate_max": float(gate.max()) if gate.size else None,
        "gate_quantiles": {
            f"p{q}": float(np.percentile(gate, q)) for q in (1, 5, 25, 50, 75, 95, 99)
        }
        if gate.size
        else {},
        "gate_fraction_gt_0_5": float((gate > 0.5).mean()) if gate.size else None,
        "gate_fraction_lt_0_1": float((gate < 0.1).mean()) if gate.size else None,
    }
    return out


def run_mechanism(*, out_path: Path, device: torch.device) -> int:
    started = time.perf_counter()
    task = P.load_task_with_cpc()
    run = load_run()
    if run.get("status") != "COMPLETED":
        raise RuntimeError("formal run is not COMPLETED")
    if run.get("test_evaluated"):
        raise RuntimeError("mechanism analysis must run before Test evaluation")
    model, payload, checkpoint_path = load_best(run, task, device)
    y = np.asarray(task.labels[task.valid_rows])
    seed = int(C.MECHANISM["shuffle_seed"])

    t0 = time.perf_counter()
    real = M.predict_logits(model, task, task.valid_rows, device, batch_size=BATCH, mode="real")
    zero = M.predict_logits(model, task, task.valid_rows, device, batch_size=BATCH, mode="delta_zero")
    shuffled = M.predict_logits(
        model, task, task.valid_rows, device, batch_size=BATCH, mode="pairing_shuffle", shuffle_seed=seed
    )
    seconds = float(time.perf_counter() - t0)
    real_rep = report(real, y)
    zero_rep = report(zero, y)
    shuffled_rep = report(shuffled, y)

    # The shuffled permutation must stay inside roots and molecules.  Verify a
    # real batch independently of predict_logits.
    batch = P.collate_with_cpc(task, task.cpc, [int(x) for x in task.valid_rows[:8]])
    pa, mid, end, b1, b2, _perm = P.shuffle_pairing_within_root(
        batch["cpc_path_atom"], batch["cpc_middle"], batch["cpc_end"],
        batch["cpc_bond_ij"], batch["cpc_bond_jk"], seed=seed,
    )
    shuffle_batch = dict(batch)
    shuffle_batch.update(
        {"cpc_path_atom": pa, "cpc_middle": mid, "cpc_end": end, "cpc_bond_ij": b1, "cpc_bond_jk": b2}
    )
    iso = P.check_batch_isolation(shuffle_batch)

    usage = full_validation_usage(model, task, device)
    prev_epoch = int(payload.get("epoch", -1))
    branch = C.read_json(C.track_rel(run["branch_usage"]))
    hist = {int(x["epoch"]): x for x in branch["history"]}

    out = {
        "schema": "lcrc-pcba-cpc-mechanism-v1",
        "round_id": C.ROUND_ID,
        "model": C.MODEL_NAME,
        "seed": int(payload["seed"]),
        "checkpoint": C.track_rel(checkpoint_path),
        "checkpoint_sha256": C.sha256_file(checkpoint_path),
        "checkpoint_epoch": prev_epoch,
        "checkpoint_global_step": int(payload["global_step"]),
        "parameters": int(payload["parameters"]),
        "test_labels_read": False,
        "test_rows_indexed": False,
        "validation_rows": int(len(task.valid_rows)),
        "interventions": {
            "real": real_rep,
            "delta_zero": zero_rep,
            "pairing_shuffle": shuffled_rep,
        },
        "changes": {
            "real_vs_delta_zero": logits_change(real, zero),
            "real_vs_pairing_shuffle": logits_change(real, shuffled),
        },
        "pairing_shuffle_isolation": iso,
        "pairing_shuffle_seed": seed,
        "full_validation_usage": usage,
        "best_epoch_branch_usage": hist.get(int(run["best_epoch"])),
        "branch_usage_summary": {
            "best_epoch": int(run["best_epoch"]),
            "best_R": hist.get(int(run["best_epoch"]), {}).get("usage", {}).get("R"),
            "final_epoch_R": hist.get(int(C.TOTAL_EPOCHS), {}).get("usage", {}).get("R"),
        },
        "interpretation_guard": (
            "delta_zero and pairing_shuffle are out-of-distribution interventions on a jointly "
            "trained network. AP drops measure reliance on the trained CPC path, not the independent "
            "causal gain of CPC over W2."
        ),
        "seconds": float(time.perf_counter() - started),
        "evaluation_seconds": seconds,
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "created_utc": C.utcnow(),
    }
    C.write_json(out_path, out)
    C.log(
        f"[mechanism] valid real {real_rep['ap']:.9f} zero {zero_rep['ap']:.9f} "
        f"shuffle {shuffled_rep['ap']:.9f} R {usage['R']:.5f}"
    )
    return 0


def _per_task(run: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    return run["fixed_epoch_reports"][str(int(epoch))]


def _mean(values: Sequence[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _diff(a: Sequence[Any], b: Sequence[Any]) -> tuple[int, int, int]:
    up = down = both = 0
    for x, y in zip(a, b):
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        both += 1
        if float(x) > float(y):
            up += 1
        elif float(x) < float(y):
            down += 1
    return up, down, both


def run_decision(*, scores_path: Path, out_path: Path) -> int:
    run = load_run()
    scores = C.read_json(scores_path)
    if scores.get("schema") != C.SCHEMA_TEST_RESULTS:
        raise RuntimeError("Test scores have the wrong schema")
    freeze = C.read_json(_abs_path(scores["freeze_manifest"]))
    if freeze.get("seal_sha256") != scores["freeze_seal_sha256"]:
        raise RuntimeError("Test score does not reference the verified freeze")
    rec = scores["recovery"]
    rules = C.DECISION_RULES

    cpc_best = scores["results"]["LCRC_CPC_best_valid"]
    cpc_40 = scores["results"]["LCRC_CPC_epoch40"]
    w2_best = scores["results"]["W2_best_valid"]
    w2_40 = scores["results"]["W2_epoch40"]
    dyn_best = scores["results"]["W2_DYNAMIC_best_valid"]
    dyn_40 = scores["results"]["W2_DYNAMIC_epoch40"]

    # Common high-support set: >=20 positives in BOTH official Test and
    # Validation, exactly as pre-registered.
    valid_best_report = _per_task(run, int(run["best_epoch"]))
    high = [
        t
        for t in range(C.N_TASKS)
        if int(cpc_best["positive"][t]) >= C.SUPPORT_MIN
        and int(valid_best_report["positive"][t]) >= C.SUPPORT_MIN
    ]
    if len(high) != int(C.RECOVERY["high_support_tasks"]):
        raise RuntimeError(f"common high-support set has {len(high)} tasks, expected 106")
    low = [t for t in range(C.N_TASKS) if t not in high]

    def strat(ids: Sequence[int], a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
        vals_a = [a["per_task_ap"][i] for i in ids]
        vals_b = [b["per_task_ap"][i] for i in ids]
        up, down, both = _diff(vals_a, vals_b)
        return {
            "tasks": len(ids),
            "mean_ap_a": _mean(vals_a),
            "mean_ap_b": _mean(vals_b),
            "delta": None if _mean(vals_a) is None else float(_mean(vals_a) - _mean(vals_b)),
            "improved_a": int(up),
            "degraded_a": int(down),
            "comparable": int(both),
        }

    def comparisons(ids: Sequence[int]) -> dict[str, Any]:
        return {
            "cpc_best_minus_w2_best": strat(ids, cpc_best, w2_best),
            "cpc_best_minus_dynamic_best": strat(ids, cpc_best, dyn_best),
            "cpc_40_minus_w2_40": strat(ids, cpc_40, w2_40),
            "cpc_40_minus_dynamic_40": strat(ids, cpc_40, dyn_40),
        }

    rows = []
    for t in range(C.N_TASKS):
        rows.append(
            {
                "task": int(t),
                "stratum": "high_support" if t in high else "low_support",
                "valid_positive_best_epoch": int(valid_best_report["positive"][t]),
                "test_positive": int(cpc_best["positive"][t]),
                "cpc_best_valid_ap": _per_task(run, int(run["best_epoch"]))["per_task_ap"][t],
                "cpc_best_test_ap": cpc_best["per_task_ap"][t],
                "cpc_40_test_ap": cpc_40["per_task_ap"][t],
                "w2_best_test_ap": w2_best["per_task_ap"][t],
                "w2_40_test_ap": w2_40["per_task_ap"][t],
                "dynamic_best_test_ap": dyn_best["per_task_ap"][t],
                "dynamic_40_test_ap": dyn_40["per_task_ap"][t],
                "cpc_minus_w2_best": None
                if cpc_best["per_task_ap"][t] is None or w2_best["per_task_ap"][t] is None
                else float(cpc_best["per_task_ap"][t] - w2_best["per_task_ap"][t]),
                "cpc_minus_dynamic_best": None
                if cpc_best["per_task_ap"][t] is None or dyn_best["per_task_ap"][t] is None
                else float(cpc_best["per_task_ap"][t] - dyn_best["per_task_ap"][t]),
            }
        )

    train_ap = None if not run.get("train_eval_final") else float(run["train_eval_final"]["ap"])
    valid_best = float(rec["valid_best_ap"])
    valid_40 = float(rec["valid_ap_at_40"])
    test_best = float(rec["test_best_ap"])
    test_40 = float(rec["test_ap_at_40"])

    mechanism_ok = True
    reasons: list[str] = []
    if not C.read_json(C.ANALYSIS / "preflight.json").get("all_ok"):
        mechanism_ok = False
        reasons.append("preflight audit failed")
    if run.get("parameters") != C.EXPECTED_PARAMETERS["total"]:
        mechanism_ok = False
        reasons.append("parameter audit mismatch")
    if rec["high_support_tasks"] != int(C.RECOVERY["high_support_tasks"]):
        mechanism_ok = False
        reasons.append("high-support set mismatch")
    if not freeze.get("unique_predictions"):
        mechanism_ok = False
        reasons.append("freeze has no unique prediction")
    if scores["test_labels"].get("read_after_freeze") is not True:
        mechanism_ok = False
        reasons.append("Test labels were not read through the post-freeze gate")

    if not mechanism_ok:
        label = rules["invalid"]["label"]
    elif (
        rec["recovery_valid_best"] >= rules["matches"]["recovery_valid_min"]
        and rec["recovery_test_best"] >= rules["matches"]["recovery_test_min"]
        and rec["improved_high_support_tasks_vs_w2"] >= rules["matches"]["improved_high_support_min"]
    ):
        label = rules["matches"]["label"]
    elif (
        rec["recovery_valid_best"] >= rules["most"]["recovery_valid_min"]
        and rec["recovery_test_best"] >= rules["most"]["recovery_test_min"]
        and rec["improved_high_support_tasks_vs_w2"] >= rules["most"]["improved_high_support_min"]
    ):
        label = rules["most"]["label"]
    elif (
        test_best > float(w2_best["official_ap"])
        and valid_best > float(C.W2_SEED0["best_valid_ap"])
        and rec["improved_high_support_tasks_vs_w2"] >= rules["partial"]["improved_high_support_min"]
        and (rec["high_support_delta_vs_w2"] or 0.0) > 0.0
    ):
        label = rules["partial"]["label"]
    else:
        label = rules["none"]["label"]

    decision = {
        "schema": "lcrc-pcba-cpc-decision-v1",
        "round_id": C.ROUND_ID,
        "model": C.MODEL_NAME,
        "label": label,
        "mechanism_valid": bool(mechanism_ok),
        "mechanism_invalid_reasons": reasons,
        "rules": rules,
        "recovery": rec,
        "observed": {
            "valid_best_ap": valid_best,
            "valid_ap_at_40": valid_40,
            "test_best_ap": test_best,
            "test_ap_at_40": test_40,
            "w2_best_valid_ap": float(C.W2_SEED0["best_valid_ap"]),
            "w2_test_best_ap": float(w2_best["official_ap"]),
            "dynamic_best_valid_ap": float(C.DYNAMIC["best_valid_ap"]),
            "dynamic_test_best_ap": float(dyn_best["official_ap"]),
            "r_valid": float(rec["recovery_valid_best"]),
            "r_test": float(rec["recovery_test_best"]),
            "r_valid_40": float(rec["recovery_valid_40"]),
            "r_test_40": float(rec["recovery_test_40"]),
            "target_valid_ap": float(C.RECOVERY["target_valid_ap"]),
            "target_test_ap": float(C.RECOVERY["target_test_ap"]),
            "best_epoch": int(run["best_epoch"]),
            "valid_best_minus_40": float(valid_best - valid_40),
            "test_best_minus_40": float(test_best - test_40),
            "train_ap_40": train_ap,
            "train_valid_gap_40": None if train_ap is None else float(train_ap - valid_40),
            "parameters": int(run["parameters"]),
            "parameter_delta_vs_w2": int(run["parameters"]) - int(C.W2_SEED0["parameters"]),
            "parameter_delta_vs_dynamic": int(run["parameters"]) - int(C.DYNAMIC["parameters"]),
            "gpu_hours": float(run["gpu_hours"]),
            "gpu_hours_vs_w2": float(run["gpu_hours"] - C.W2_SEED0["gpu_hours"]),
            "gpu_hours_vs_dynamic": float(run["gpu_hours"] - C.DYNAMIC["gpu_hours"]),
            "validation_to_test_drop": float(valid_best - test_best),
            "dynamic_validation_to_test_drop": float(C.DYNAMIC["best_valid_ap"] - dyn_best["official_ap"]),
            "w2_validation_to_test_drop": float(C.W2_SEED0["best_valid_ap"] - w2_best["official_ap"]),
        },
        "strata": {
            "high_support": comparisons(high),
            "low_support": comparisons(low),
            "all_tasks": comparisons(list(range(C.N_TASKS))),
        },
        "high_support_task_ids": [int(t) for t in high],
        "low_support_task_ids": [int(t) for t in low],
        "per_task": rows,
        "per_task_artifact": C.track_rel(out_path.with_name("per_task_analysis.json")),
        "test_scores": C.track_rel(scores_path),
        "test_scores_sha256": C.sha256_file(scores_path),
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "note": (
            "No statistical-significance claim. Test is an exposed engineering comparison set. "
            "The label applies the frozen decision rules to the measured numbers."
        ),
    }
    C.write_json(
        out_path.with_name("per_task_analysis.json"),
        {"tasks": rows, "high_support": high, "low_support": low},
    )
    C.write_json(out_path, decision)
    C.log(
        f"[decision] {label} R_valid {rec['recovery_valid_best']:.4f} "
        f"R_test {rec['recovery_test_best']:.4f} high {rec['improved_high_support_tasks_vs_w2']}/106"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mechanism", "decision", "all"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mechanism", default=str(C.ANALYSIS / "mechanism.json"))
    parser.add_argument("--scores", default=str(C.ANALYSIS / "test_scores.json"))
    parser.add_argument("--decision", default=str(C.ANALYSIS / "decision.json"))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.mode in ("mechanism", "all"):
        run_mechanism(out_path=Path(args.mechanism), device=device)
    if args.mode in ("decision", "all"):
        run_decision(scores_path=Path(args.scores), out_path=Path(args.decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
