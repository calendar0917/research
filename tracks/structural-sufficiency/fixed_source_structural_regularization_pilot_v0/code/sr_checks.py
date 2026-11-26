"""FSRP v0 necessary checks + timing preflight.

Checks (only what this round needs; the inherited rounds' caches are not
re-audited):

C1  both arms load the same base state / optimizer / RNG / LR / order plan
C2  eval-mode initial logits and Valid AP agree between the arms
C3  STRUCT_DROP at q=0 reproduces FIXED1 forward and gradients
C4  both directions and both blocks share the same physical-bond mask
C5  realised drop fraction ~10%, mask mapping is exact (no cross-molecule slip)
C6  eval mode never draws a mask; eval arithmetic equals inherited FIXED1
C7  both arms have 1,128,468 parameters
C8  complete resume state restored; Test is not accessed

``--phase timing`` additionally measures the step/eval cost and projects
whether both arms fit in the remaining wall budget.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import sr_common as R
import sr_model as SM

C = R.C
DM = R.DM
DT = R.DT

import pc_metrics as MET  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CHECKS: list[dict[str, Any]] = []


def record(cid: str, name: str, passed: bool, detail: Mapping[str, Any]) -> None:
    CHECKS.append({"id": cid, "name": name, "passed": bool(passed), "detail": dict(detail)})
    C.log(f"[{cid}] {'PASS' if passed else 'FAIL'} {name}")


def _build(arm: str, task: Any, device: torch.device) -> torch.nn.Module:
    if arm == "BASE_CONT":
        model, _audit = DM.build_ds_model(
            task.spec, task.norm, C.N_TASKS, arm="FIXED1", seed=R.SEED, device=device
        )
    else:
        model = SM.StructuralDropModel(
            task.spec, task.norm, C.N_TASKS, dict(C.BC.config_for_scale(2)),
            struct_drop_q=R.Q, mask_seed=R.MASK_SEED,
        )
        model.to(device)
    return model


def _load_base_into(model: torch.nn.Module, payload: Mapping[str, Any]) -> None:
    model.load_state_dict(payload["state_dict"], strict=True)


def _batch(task: Any, rows: np.ndarray, device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in C.PC.collate(task.cache, rows.tolist()).items()}


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item()) if a.numel() else 0.0


def _grad_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])


def check_c1_c8(task: Any, payload: Mapping[str, Any], base_audit: Mapping[str, Any], device: torch.device, plan: list[dict[str, Any]]) -> tuple[Any, Any]:
    # C1: both arms from the same state
    rows = []
    models = {}
    for arm in R.ARMS:
        model = _build(arm, task, device)
        optimizer, opt_audit = DT.build_optimizer(model)
        resume = DT.restore(payload, model, optimizer, arm="FIXED1", seed=R.SEED)
        lr_restore = float(optimizer.param_groups[0]["lr"])
        lr_first_step = R.current_lr_at(optimizer, R.EXPECTED_BASE_STEP)
        fp = {
            "arm": arm,
            "state_fingerprint": R.model_state_fingerprint(model),
            "optimizer_fingerprint": R.optimizer_fingerprint(optimizer),
            "rng_state_hashes": C.rng_state_hashes(C.capture_rng_state()),
            "lr_at_restore": lr_restore,
            "lr_first_continued_step": lr_first_step,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "resume": resume,
        }
        rows.append(fp)
        models[arm] = (model, optimizer)
    same = {
        "state": len({r["state_fingerprint"] for r in rows}) == 1,
        "optimizer": len({r["optimizer_fingerprint"] for r in rows}) == 1,
        "rng": len({str(r["rng_state_hashes"]["combined_sha256"]) for r in rows}) == 1,
        "lr_restore": len({float(r["lr_at_restore"]) for r in rows}) == 1,
        "lr_first_step": len({float(r["lr_first_continued_step"]) for r in rows}) == 1,
        "parameters": len({r["parameters"] for r in rows}) == 1,
    }
    record(
        "C1", "both arms start from the same weights / optimizer / RNG / LR",
        all(same.values())
        and abs(rows[0]["lr_at_restore"] - R.EXPECTED_LR_AT_RESTORE) <= 1e-15
        and abs(rows[0]["lr_first_continued_step"] - R.EXPECTED_LR_FIRST_STEP) <= 1e-15,
        {
            "same": same,
            "arms": rows,
            "expected_lr_at_restore": R.EXPECTED_LR_AT_RESTORE,
            "expected_lr_first_continued_step": R.EXPECTED_LR_FIRST_STEP,
            "base_plan": base_audit["batch_plan"],
            "recomputed_plan": C.plan_meta(plan, R.START_EPOCH, R.ORDER_SEED),
            "mask_rng_note": "STRUCT_DROP adds an independent mask generator; the model RNG is untouched",
        },
    )

    # C8: complete resume state + Test isolation
    required = ["state_dict", "optimizer_state_dict", "rng_state", "rng_state_hashes", "batch_plan", "scheduler_state"]
    test_finite = int(np.isfinite(np.asarray(task.labels[task.test_rows])).sum())
    c8_details = {
        "checkpoint": base_audit,
        "required_resume_fields_present": {k: (k in payload) for k in required},
        "optimizer_state_nonempty": bool(payload["optimizer_state_dict"].get("state")),
        "scheduler_global_step": int(payload["scheduler_state"]["global_step"]),
        "batch_plan_consumed_epochs": int(payload["batch_plan"]["consumed_epochs"]),
        "test_rows": int(len(task.test_rows)),
        "finite_test_labels_visible_in_task": test_finite,
        "test_labels_loaded_by_this_round": False,
        "test_scored_by_this_round": False,
    }
    record(
        "C8", "complete resumable FIXED1 state restored; Test not accessed",
        all(c8_details["required_resume_fields_present"].values())
        and c8_details["optimizer_state_nonempty"]
        and test_finite == 0
        and int(payload["batch_plan"]["consumed_epochs"]) == R.START_EPOCH,
        c8_details,
    )
    return models


def check_c2(models: Mapping[str, tuple[Any, Any]], task: Any, device: torch.device) -> dict[str, Any]:
    out = {}
    for arm, (model, _opt) in models.items():
        rep = C.evaluate_split(model, task, device, "valid", per_task=True, keep_logits=True, reason=f"C2_{arm}")
        out[arm] = rep
    a, b = out["BASE_CONT"], out["STRUCT_DROP"]
    ap_diff = float(b["ap"] - a["ap"])
    bce_diff = float(b["bce"] - a["bce"])
    logit_diff = float(np.max(np.abs(b["logits"] - a["logits"])))
    per_task = np.asarray(
        [(np.nan if x is None or y is None else float(y) - float(x)) for x, y in zip(a["per_task_ap"], b["per_task_ap"])],
        dtype=np.float64,
    )
    passed = abs(ap_diff) <= 2e-5 and abs(bce_diff) <= 1e-7 and logit_diff <= 1e-4
    record(
        "C2", "eval-mode initial logits / Valid AP agree between the two arms",
        passed,
        {
            "valid_ap": {"BASE_CONT": a["ap"], "STRUCT_DROP": b["ap"]},
            "valid_bce": {"BASE_CONT": a["bce"], "STRUCT_DROP": b["bce"]},
            "delta_ap": ap_diff,
            "delta_bce": bce_diff,
            "max_abs_logit_diff": logit_diff,
            "logits_sha256": {"BASE_CONT": a["logits_sha256"], "STRUCT_DROP": b["logits_sha256"]},
            "n_tasks_with_nonzero_per_task_delta": int(np.nansum(per_task != 0)),
            "max_abs_per_task_ap_delta": float(np.nanmax(np.abs(per_task))) if np.isfinite(per_task).any() else None,
            "recorded_base_valid_ap": R.EXPECTED_BASE_VALID_AP,
            "abs_delta_vs_recorded": abs(float(a["ap"]) - R.EXPECTED_BASE_VALID_AP),
            "tolerance_note": "CUDA scatter uses atomic adds; re-scoring tolerance 1e-4 logits / 2e-5 AP",
        },
    )
    return out


def check_c3(payload: Mapping[str, Any], task: Any, device: torch.device, plan: list[dict[str, Any]]) -> None:
    rows = plan[R.START_EPOCH]["batches"][0]  # epoch-26 first batch
    batch = _batch(task, rows, device)
    y = task.labels[rows]
    y_t = torch.as_tensor(np.ascontiguousarray(y, dtype=np.float32), device=device)
    base = _build("BASE_CONT", task, device)
    drop = _build("STRUCT_DROP", task, device)
    drop.struct_drop_q = 0.0
    _load_base_into(base, payload)
    _load_base_into(drop, payload)
    results = {}
    for name, model in (("BASE_CONT", base), ("STRUCT_DROP_q0", drop)):
        C.seed_everything(20261126)
        model.train()
        model.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = MET.masked_bce_loss(logits, y_t)
        loss.backward()
        results[name] = {
            "logits": logits.detach().clone(),
            "loss": float(loss.detach()),
            "grad": _grad_vector(model).clone(),
        }
    a, b = results["BASE_CONT"], results["STRUCT_DROP_q0"]
    logit_diff = _max_abs_diff(a["logits"], b["logits"])
    loss_diff = abs(a["loss"] - b["loss"])
    grad_diff = _max_abs_diff(a["grad"], b["grad"])
    record(
        "C3", "STRUCT_DROP at q=0 reproduces FIXED1 forward and gradients",
        logit_diff == 0.0 and loss_diff == 0.0 and grad_diff == 0.0,
        {
            "batch_rows": int(len(rows)),
            "max_abs_logit_diff": logit_diff,
            "abs_loss_diff": loss_diff,
            "max_abs_grad_diff": grad_diff,
            "grad_tensor_elements": int(a["grad"].numel()),
            "mask_draws_at_q0": int(drop.mask_draws),
            "note": "q=0 masked path executes keep=1 exactly; the base path has no mask tensor",
        },
    )


def check_c4_c5(task: Any, payload: Mapping[str, Any], device: torch.device, plan: list[dict[str, Any]]) -> None:
    model = _build("STRUCT_DROP", task, device)
    _load_base_into(model, payload)
    model.train()
    rows = plan[R.START_EPOCH]["batches"][0]
    batch = _batch(task, rows, device)
    _ = model(batch)
    ev = model.last_mask
    idx = batch["bond_index"]
    n_phys = int(idx.shape[0])
    src, dst, _em = DM.directed_bonds(model, batch)
    expected_src = torch.cat([idx[:, 0], idx[:, 1]])
    expected_dst = torch.cat([idx[:, 1], idx[:, 0]])
    mapping_exact = bool(torch.equal(src, expected_src) and torch.equal(dst, expected_dst))
    same_mol = bool(torch.equal(batch["atom_batch"][src], batch["atom_batch"][dst]))
    c4_pass = (
        ev is not None
        and ev["same_mask_tensor_both_blocks"]
        and ev["directions_share_physical_mask"]
        and int(ev["n_physical_bonds"]) == n_phys
        and int(ev["n_directed_edges"]) == 2 * n_phys
        and mapping_exact
        and same_mol
    )
    record(
        "C4", "two directions and both blocks share the same physical-bond mask",
        c4_pass,
        {
            "n_physical_bonds": n_phys,
            "n_directed_edges": 2 * n_phys,
            "block_masks": ev["block_masks"] if ev else None,
            "same_mask_tensor_both_blocks": None if ev is None else ev["same_mask_tensor_both_blocks"],
            "directions_share_physical_mask": None if ev is None else ev["directions_share_physical_mask"],
            "directed_bonds_order_exact": mapping_exact,
            "every_directed_edge_inside_one_molecule": same_mol,
        },
    )

    # C5: realised fraction over many training batches + exact scatter reference
    n_batches = 40
    dropped = 0
    total = 0
    per_batch = []
    for b in range(n_batches):
        rows_b = plan[R.START_EPOCH]["batches"][b]
        m = model(_batch(task, rows_b, device))
        ev_b = model.last_mask
        n_b = int(ev_b["n_physical_bonds"])
        d_b = int(round(ev_b["dropped_fraction_physical"] * n_b))
        dropped += d_b
        total += n_b
        per_batch.append(d_b / n_b if n_b else 0.0)
    frac = dropped / total if total else 0.0
    se = math.sqrt(R.Q * (1.0 - R.Q) / total) if total else 0.0
    z = abs(frac - R.Q) / se if se else 0.0
    # exact mapping reference on a tiny deterministic example
    z_t = torch.randn(6, 4, device=device)
    small_src = torch.tensor([0, 1, 2, 3, 4, 5], device=device)
    small_dst = torch.tensor([1, 2, 3, 4, 5, 0], device=device)
    small_edge = torch.zeros(6, 4, device=device)
    keep = torch.tensor([1.0, 0.0, 2.0, 0.0, 1.0, 2.0], device=device)
    got = SM.masked_source_message(z_t, small_src, small_dst, small_edge, keep)
    ref = torch.zeros_like(z_t)
    for e in range(6):
        ref[small_dst[e]] += torch.nn.functional.silu(z_t[small_src[e]]) * keep[e]
    exact = float((got - ref).abs().max().item())
    record(
        "C5", "realised drop fraction ~10% and exact bond-level mapping",
        z <= 3.0 and exact <= 1e-6,
        {
            "batches": n_batches,
            "physical_bonds": int(total),
            "dropped": int(dropped),
            "realised_fraction": frac,
            "q": R.Q,
            "binomial_se": se,
            "abs_deviation_in_se": z,
            "mean_per_batch_fraction": float(np.mean(per_batch)),
            "min_per_batch_fraction": float(np.min(per_batch)),
            "max_per_batch_fraction": float(np.max(per_batch)),
            "synthetic_scatter_reference_max_abs_diff": exact,
        },
    )


def check_c6(task: Any, payload: Mapping[str, Any], device: torch.device, plan: list[dict[str, Any]]) -> None:
    drop = _build("STRUCT_DROP", task, device)
    plain = _build("BASE_CONT", task, device)
    _load_base_into(drop, payload)
    _load_base_into(plain, payload)
    drop.train()
    rows = plan[R.START_EPOCH]["batches"][0]
    batch = _batch(task, rows, device)
    _ = drop(batch)
    draws_after_train = int(drop.mask_draws)
    drop.eval()
    with torch.inference_mode():
        logits_drop = drop(batch)
        logits_plain = plain.eval()(batch)
    diff = _max_abs_diff(logits_drop, logits_plain)
    record(
        "C6", "eval mode draws no mask and equals inherited FIXED1 arithmetic",
        int(drop.mask_draws) == draws_after_train and diff <= 1e-6,
        {
            "mask_draws_after_training_forward": draws_after_train,
            "mask_draws_after_eval_forward": int(drop.mask_draws),
            "eval_forward_calls": int(drop.eval_forward_calls),
            "max_abs_logit_diff_vs_plain_FIXED1": diff,
            "tolerance_note": "identical eval code path (delegated); CUDA atomics tolerance 1e-6",
        },
    )


def check_c7(models: Mapping[str, tuple[Any, Any]]) -> None:
    counts = {}
    audits = {}
    for arm, (model, _opt) in models.items():
        counts[arm] = int(sum(p.numel() for p in model.parameters()))
        audits[arm] = C.parameter_audit(model)
    record(
        "C7", "both arms have 1,128,468 parameters",
        all(v == R.PARAMS_TOTAL for v in counts.values()) and all(a["all_match"] for a in audits.values()),
        {"counts": counts, "expected": R.PARAMS_TOTAL, "audit_all_match": {k: v["all_match"] for k, v in audits.items()}},
    )


def phase_timing(task: Any, payload: Mapping[str, Any], device: torch.device, plan: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    model = _build("STRUCT_DROP", task, device)
    optimizer, _ = DT.build_optimizer(model)
    DT.restore(payload, model, optimizer, arm="FIXED1", seed=R.SEED)
    model.train()
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    batches = list(plan[R.START_EPOCH]["batches"])
    global_steps = R.EXPECTED_BASE_STEP
    timings = []
    n_steps = int(args.timing_steps)
    warm = 3
    for i in range(warm + n_steps):
        rows = batches[i % len(batches)]
        batch = stager.batch(C.PC.collate(task.cache, rows.tolist()))
        y = stager.labels(task.labels[rows])
        t0 = time.perf_counter()
        DT.train_step(model, optimizer, batch, y, global_step=global_steps, clip=clip, audit_new=False)
        global_steps += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if i >= warm:
            timings.append(float(time.perf_counter() - t0))
    stager.release()
    step_s = float(np.mean(timings))
    step_p90 = float(np.percentile(timings, 90))
    steps = int(len(batches))
    epoch_train_s = steps * step_s
    # eval measurement
    model.eval()
    t0 = time.perf_counter()
    rep = C.evaluate_split(model, task, device, "valid", per_task=False, reason="timing_valid")
    valid_s = float(time.perf_counter() - t0)
    n_train = int(len(task.train_rows))
    probe_rows = min(2048, n_train)
    # score a probe subset without touching the full Train split
    t0 = time.perf_counter()
    probe = np.asarray(task.train_rows[:probe_rows], dtype=np.int64)
    logits = C.PC.predict_logits(model, task.cache, probe, device, batch_size=512)
    probe_s = float(time.perf_counter() - t0)
    train_eval_s = probe_s * (n_train / max(probe_rows, 1))
    per_epoch = epoch_train_s + valid_s * 1.05 + 1.5
    per_arm = (
        35.0  # load task/model/plan
        + len(R.TRAIN_EPOCHS) * per_epoch
        + (train_eval_s + valid_s)  # epoch-25 baseline Train + Valid
        + 2 * train_eval_s  # epochs 29 and 33
        + (train_eval_s + valid_s)  # best checkpoint Train + Valid
    )
    parallel = bool(args.two_gpus)
    wall = per_arm if parallel else 2.0 * per_arm
    remaining = float(args.remaining_seconds)
    fits = bool(wall + 45.0 <= remaining)
    out = {
        "schema": R.SCHEMA_PREFLIGHT,
        "round_id": R.ROUND_ID,
        "device": str(device),
        "timing_steps": n_steps,
        "warmup_steps": warm,
        "step_seconds_mean": step_s,
        "step_seconds_p90": step_p90,
        "steps_per_epoch": steps,
        "projected_epoch_train_seconds": epoch_train_s,
        "measured_full_valid_seconds": valid_s,
        "valid_ap_probe": float(rep["ap"]),
        "train_eval_projection_seconds": train_eval_s,
        "train_eval_probe_rows": probe_rows,
        "projected_seconds_per_epoch": per_epoch,
        "projected_seconds_per_arm": per_arm,
        "two_gpus_parallel": parallel,
        "projected_wall_seconds": wall,
        "remaining_wall_seconds": remaining,
        "fits_two_arms": fits,
        "overhead_reserve_seconds": 45.0,
        "note": "train-eval time is extrapolated from a 2048-row probe; the formal run measures it exactly",
    }
    R.write_json(Path(args.timing_out), out)
    record(
        "T1", "timing preflight: both arms fit the remaining wall budget",
        fits,
        out,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("checks", "timing", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(R.ANALYSIS / "checks.json"))
    parser.add_argument("--timing-out", default=str(R.ANALYSIS / "preflight_timing.json"))
    parser.add_argument("--base-checkpoint", default="")
    parser.add_argument("--timing-steps", type=int, default=25)
    parser.add_argument("--remaining-seconds", type=float, default=0.0)
    parser.add_argument("--two-gpus", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    started = time.perf_counter()
    task = R.load_task_without_test()
    payload, base_audit, base_path = R.load_base_checkpoint(args.base_checkpoint)
    plan, plan_audit = R.plan_and_identity(task)
    R.verify_plan_prefix(payload, plan, R.START_EPOCH)
    models = None
    if args.phase in ("checks", "all"):
        models = check_c1_c8(task, payload, base_audit, device, plan)
        check_c2(models, task, device)
        check_c3(payload, task, device, plan)
        check_c4_c5(task, payload, device, plan)
        check_c6(task, payload, device, plan)
        check_c7(models)
    timing = None
    if args.phase in ("timing", "all"):
        timing = phase_timing(task, payload, device, plan, args)
    elapsed = float(time.perf_counter() - started)
    if args.phase in ("checks", "all"):
        R.write_json(
            Path(args.out),
            {
                "schema": R.SCHEMA_CHECKS,
                "round_id": R.ROUND_ID,
                "base_checkpoint": base_audit,
                "plan_identity": plan_audit,
                "checks": CHECKS,
                "all_passed": all(c["passed"] for c in CHECKS),
                "checks_seconds": elapsed,
                "utc": R.utcnow(),
                "test_scored": False,
                "test_labels_loaded": False,
            },
        )
    C.log(f"phase={args.phase} elapsed {elapsed:.1f}s all_passed={all(c['passed'] for c in CHECKS)}")
    if args.phase in ("checks", "all") and not all(c["passed"] for c in CHECKS):
        return 4
    if timing is not None and not timing["fits_two_arms"]:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
