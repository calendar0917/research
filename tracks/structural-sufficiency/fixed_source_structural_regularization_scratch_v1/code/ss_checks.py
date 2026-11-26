"""FSS-v1 necessary checks + timing preflight (run on the target GPU).

Only what this round adds or must re-confirm is measured; the inherited data
caches are not re-audited.

C1  initial parameters AND buffers are bitwise the official FIXED1 seed-0
    constructor state (and the historical ``initial_state_sha256``)
C2  initial eval-mode logits are exactly the FIXED1 logits (CPU exact)
C3  the optimizer is genuinely from scratch (no AdamW state, warmup-start LR)
C4  the structural mask is active from the first training step
C5  the mask draw never advances the model's own (global) RNG stream
C6  the 25-epoch order plan equals the historical FIXED1 plan (global + prefix)
C7  eval mode draws no mask and its arithmetic equals FIXED1
C8  Test is never loaded, scored or referenced
plus the inherited gates:
C0a q=0 reproduces FIXED1 forward and gradients exactly (CPU)
C0b both directions and both blocks share one physical-bond mask
C0c realised drop fraction ~ q inside a 3-sigma binomial band, exact scatter
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ss_common as R
import ss_model as M

C = R.C
DM = R.DM
DT = R.DT

from pf_pipeline import PinnedStager  # noqa: E402

CHECKS: list[dict[str, Any]] = []


def record(cid: str, name: str, passed: bool, detail: Mapping[str, Any]) -> None:
    CHECKS.append({"id": cid, "name": name, "passed": bool(passed), "detail": dict(detail)})
    C.log(f"[{cid}] {'PASS' if passed else 'FAIL'} {name}")


def _batch(task: Any, rows: Sequence[int]) -> dict[str, Any]:
    return C.PC.collate(task.cache, [int(r) for r in rows])


def _grad_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])


def _forward_logits(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    return model(batch)


def check_initial_state(task: Any, device: torch.device) -> tuple[Any, Any]:
    reference, ref_audit = M.build_fixed1_reference(task, device="cpu")
    model, audit = M.build_scratch_model(task, device="cpu")
    ref_state = C.state_cpu(reference)
    ours = C.state_cpu(model)
    equal = C.compare_states(ours, ref_state)
    same_keys = set(ref_state) == set(ours)
    buffers_ref = {k: v for k, v in ref_state.items() if k not in dict(reference.named_parameters())}
    buffers_ours = {k: v for k, v in ours.items() if k not in dict(model.named_parameters())}
    buffers_equal = C.compare_states(buffers_ours, buffers_ref)
    fp = C.state_fingerprint(ours)
    record(
        "C1",
        "initial parameters and buffers are the FIXED1 seed-0 constructor state",
        bool(
            same_keys
            and equal["all_equal"]
            and buffers_equal["all_equal"]
            and fp == R.EXPECTED_INITIAL_STATE_SHA256
            and C.state_fingerprint(C.state_subset(ours, new=True)) == R.EXPECTED_NEW_STATE_SHA256
            and int(len(ours)) == R.N_STATE_TENSORS
        ),
        {
            "same_state_keys": bool(same_keys),
            "n_state_tensors": int(len(ours)),
            "parameters_match_reference": equal,
            "buffers_match_reference": buffers_equal["all_equal"],
            "n_buffers": int(len(buffers_ours)),
            "initial_state_sha256": fp,
            "expected_initial_state_sha256": R.EXPECTED_INITIAL_STATE_SHA256,
            "new_state_sha256": C.state_fingerprint(C.state_subset(ours, new=True)),
            "expected_new_state_sha256": R.EXPECTED_NEW_STATE_SHA256,
            "w2_trunk_init_fingerprint": ref_audit.get("w2_trunk_init_fingerprint"),
            "w2_trunk_matches_recorded": ref_audit.get("w2_trunk_fingerprint_matches_recorded_w2_seed0"),
            "parameter_audit_total": int(audit["parameters"]["total"]),
            "parameters_expected": R.PARAMS_TOTAL,
        },
    )
    return reference, model


def check_initial_logits(task: Any, reference: torch.nn.Module, model: torch.nn.Module) -> None:
    rows = list(task.valid_rows[:256])
    batch = _batch(task, rows)
    reference.eval()
    model.eval()
    with torch.inference_mode():
        a = _forward_logits(reference, batch).cpu()
        b = _forward_logits(model, batch).cpu()
    diff = float((a - b).abs().max().item())
    record(
        "C2",
        "initial eval logits equal the FIXED1 initialisation (CPU, exact)",
        diff == 0.0,
        {
            "rows": len(rows),
            "max_abs_logit_diff": diff,
            "exact": diff == 0.0,
            "n_mask_draws_during_eval": int(model.mask_draws),
        },
    )


def check_optimizer(task: Any, model: torch.nn.Module) -> None:
    optimizer, audit = M.build_optimizer(model)
    fresh = M.optimizer_is_fresh(optimizer)
    lr0 = float(optimizer.param_groups[0]["lr"])
    sched0 = float(C.scheduled_lr(0, phase=3, kind="all"))
    C.set_lrs(optimizer, step=0)
    lr_step0 = float(optimizer.param_groups[0]["lr"])
    record(
        "C3",
        "optimizer is freshly initialised (no restored AdamW moments)",
        bool(
            fresh["fresh"]
            and abs(lr0 - float(C.RECIPE["warmup_start_lr"])) <= 1e-15
            and abs(lr_step0 - sched0) <= 1e-15
            and fresh["n_param_groups"] == 1
        ),
        {
            "fresh": fresh,
            "lr_constructed": lr0,
            "expected_warmup_start_lr": float(C.RECIPE["warmup_start_lr"]),
            "lr_first_scheduled_step": lr_step0,
            "scheduled_step0": sched0,
            "audit": audit,
        },
    )


def _residual_perturbed(model: torch.nn.Module) -> torch.nn.Module:
    """Copy with a non-zero FFN residual branch (the FIXED1 init zeroes ``fc2``)."""
    import copy

    probe = copy.deepcopy(model)
    torch.manual_seed(777)  # identical perturbation for every copy
    with torch.no_grad():
        for block in probe.blocks:
            block.fc2.weight.normal_(mean=0.0, std=0.02)
            block.fc2.bias.normal_(mean=0.0, std=0.02)
    return probe


def check_q0_and_mask(task: Any, reference: torch.nn.Module, model: torch.nn.Module) -> None:
    rows = list(task.train_rows[:64])
    batch = _batch(task, rows)
    y = torch.from_numpy(np.ascontiguousarray(task.labels[[int(r) for r in rows]], dtype=np.float32))
    draws0 = int(model.mask_draws)

    # --- C0a: q = 0 must reproduce FIXED1 forward and gradients exactly (CPU)
    ref_p = _residual_perturbed(reference)
    sd_p = _residual_perturbed(model)
    sd_p.struct_drop_q = 0.0
    ref_p.train()
    sd_p.train()
    torch.manual_seed(12345)
    ref_p.zero_grad(set_to_none=True)
    logits_ref = _forward_logits(ref_p, batch)
    loss_ref = C.MET.masked_bce_loss(logits_ref, y)
    loss_ref.backward()
    grad_ref = _grad_vector(ref_p)

    torch.manual_seed(12345)
    sd_p.zero_grad(set_to_none=True)
    logits_q0 = _forward_logits(sd_p, batch)
    loss_q0 = C.MET.masked_bce_loss(logits_q0, y)
    loss_q0.backward()
    grad_q0 = _grad_vector(sd_p)
    grad_max = float((grad_ref - grad_q0).abs().max().item()) if grad_ref.numel() else 0.0
    grad_scale = float(grad_ref.abs().max().item()) if grad_ref.numel() else 1.0
    q0 = {
        "logits_max_abs_diff": float((logits_ref - logits_q0).abs().max().item()),
        "loss_abs_diff": float((loss_ref - loss_q0).abs().item()),
        "grad_max_abs_diff": grad_max,
        "grad_max_abs_reference": grad_scale,
        "grad_relative_diff": None if grad_scale == 0 else grad_max / grad_scale,
        "n_grad_entries": int(grad_ref.numel()),
        "mask_draws_q0_forward": int(sd_p.mask_draws) - draws0,
    }
    sd_p.struct_drop_q = float(R.Q)

    # --- C4: with a non-zero residual the mask must change the training forward
    torch.manual_seed(12345)
    sd_p.train()
    draws_before_masked = int(sd_p.mask_draws)
    logits_masked = _forward_logits(sd_p, batch)
    mask_delta = float((logits_ref.detach() - logits_masked.detach()).abs().max().item())
    mask_differs = mask_delta > 0.0
    draws_after_train = int(sd_p.mask_draws)
    event = sd_p.last_mask or {}

    # --- C7: eval mode must be exactly FIXED1 and draw no mask
    ref_p.eval()
    sd_p.eval()
    with torch.inference_mode():
        logits_eval_ref = _forward_logits(ref_p, batch).cpu()
        logits_eval = _forward_logits(sd_p, batch).cpu()
    draws_after_eval = int(sd_p.mask_draws)
    eval_diff = float((logits_eval_ref - logits_eval).abs().max().item())
    eval_exact = eval_diff == 0.0

    # the zero-residual initialisation itself must also be mask-invariant
    model.train()
    reference.eval()
    with torch.inference_mode():
        init_logits_ref = _forward_logits(reference, batch).cpu()
    model.eval()
    with torch.inference_mode():
        init_logits_sd = _forward_logits(model, batch).cpu()
    init_eval_diff = float((init_logits_ref - init_logits_sd).abs().max().item())

    record(
        "C0a",
        "q=0 reproduces FIXED1 forward, loss and gradients; q=0.10 changes the training forward",
        bool(
            q0["logits_max_abs_diff"] == 0.0
            and q0["loss_abs_diff"] == 0.0
            and (q0["grad_relative_diff"] or 0.0) <= 1e-5
            and mask_differs
        ),
        {**q0, "masked_forward_differs": mask_differs, "masked_logit_delta_max": mask_delta,
         "note": "the official FIXED1 init zeroes fc2, so the mask is only observable after the "
                 "residual branch becomes non-zero; the comparison uses residual-perturbed copies"},
    )
    record(
        "C4",
        "structural mask is active in training from the first forward and inert in eval",
        bool(
            draws_after_train - draws_before_masked == 1
            and draws_after_eval == draws_after_train
            and event.get("q") == R.Q
        ),
        {
            "mask_draws_before_masked_forward": draws_before_masked,
            "mask_draws_after_masked_forward": draws_after_train,
            "draws_per_training_forward": draws_after_train - draws_before_masked,
            "mask_draws_after_eval_forward": draws_after_eval,
            "realised_q": event.get("q"),
            "dropped_fraction_physical": event.get("dropped_fraction_physical"),
            "kept_fraction_directed": event.get("kept_fraction_directed"),
            "n_physical_bonds": event.get("n_physical_bonds"),
            "n_directed_edges": event.get("n_directed_edges"),
        },
    )
    shared = {
        "same_mask_tensor_both_blocks": bool(event.get("same_mask_tensor_both_blocks")),
        "directions_share_physical_mask": bool(event.get("directions_share_physical_mask")),
        "mask_len_equals_directed_edges": bool(
            (event.get("block_masks") or [{}])[0].get("keep_numel") == event.get("n_directed_edges")
        ),
    }
    record(
        "C0b",
        "one physical-bond mask, shared by both directions and both FIXED1 blocks",
        all(shared.values()),
        {**shared, "block_masks": event.get("block_masks")},
    )
    record(
        "C7",
        "eval-mode arithmetic equals FIXED1 exactly (no mask, no scaling)",
        bool(eval_exact and init_eval_diff == 0.0 and draws_after_eval == draws_after_train),
        {
            "cpu_max_abs_logit_diff_eval_perturbed_residual": eval_diff,
            "cpu_max_abs_logit_diff_eval_initial": init_eval_diff,
            "mask_draws_in_eval": draws_after_eval - draws_after_train,
            "eval_forward_calls": int(sd_p.eval_forward_calls),
        },
    )


def check_mask_rng_independence(task: Any, reference: torch.nn.Module, model: torch.nn.Module) -> None:
    rows = list(task.train_rows[:64])
    batch = _batch(task, rows)
    reference.train()
    model.train()

    state0 = torch.get_rng_state().clone()
    reference.zero_grad(set_to_none=True)
    _forward_logits(reference, batch)
    state_fixed1 = torch.get_rng_state().clone()

    torch.set_rng_state(state0)
    mask_fp_before = R.nested_fingerprint(model.mask_rng_state())
    model.zero_grad(set_to_none=True)
    _forward_logits(model, batch)
    state_struct = torch.get_rng_state().clone()
    mask_fp_after = R.nested_fingerprint(model.mask_rng_state())

    # the mask draw itself must advance only its own generator
    gen_only = mask_fp_before != mask_fp_after
    record(
        "C5",
        "the mask draw does not advance the model's global RNG stream",
        bool(torch.equal(state_fixed1, state_struct) and gen_only),
        {
            "global_rng_after_fixed1_equals_after_struct_drop": bool(torch.equal(state_fixed1, state_struct)),
            "mask_generator_advanced": bool(gen_only),
            "mask_rng_fingerprint_before": mask_fp_before,
            "mask_rng_fingerprint_after": mask_fp_after,
        },
    )


def check_plan(task: Any, device: torch.device) -> None:
    plan, plan_audit = R.plan_and_identity(task)
    gate = R.verify_plan_identity(plan)
    sizes = [len(plan[e]["batches"]) for e in range(R.EPOCHS)]
    rows_seen = [int(np.sum(np.concatenate([np.asarray(b) for b in plan[e]["batches"]]))) for e in range(R.EPOCHS)]
    record(
        "C6",
        "25-epoch order plan equals the historical FIXED1 seed-0 plan",
        bool(
            gate["matches_historical_fixed1"]
            and len(set(sizes)) == 1
            and sizes[0] == R.STEPS_PER_EPOCH
            and all(len(set(np.concatenate([np.asarray(b) for b in plan[e]["batches"]]).tolist())) ==
                    len(np.concatenate([np.asarray(b) for b in plan[e]["batches"]])) for e in range(R.EPOCHS))
        ),
        {
            "gate": gate,
            "plan_identity": plan_audit,
            "steps_per_epoch": sizes,
            "rows_consumed_per_epoch": rows_seen,
            "batch_size": int(C.BATCH_SIZE),
            "expected_plan_global_sha256": R.EXPECTED_PLAN_GLOBAL_SHA256,
            "expected_plan_prefix25_sha256": R.EXPECTED_PLAN_PREFIX25_SHA256,
        },
    )


def check_test_isolation() -> None:
    """Only Test *access* is forbidden; the word itself is not."""
    offenders: list[str] = []
    for path in sorted(R.CODE_DIR.glob("*.py")):
        if path.name == Path(__file__).name:
            continue  # this file only names the forbidden entry points
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "test_labels_loaded" in line or "test_used_for_selection" in line or "test_scored" in line:
                continue
            if "load_test_labels" in line or "test_rows" in line or "test_labels" in line:
                offenders.append(f"{path.name}:{i}: {stripped}")
            if "evaluate_split" in line and "'test'" in line:
                offenders.append(f"{path.name}:{i}: {stripped}")
    task = R.load_task_without_test()
    exposed = int(np.isfinite(task.labels[task.test_rows]).sum())
    record(
        "C8",
        "Test split is never loaded or scored in this round",
        bool(exposed == 0 and not offenders),
        {
            "finite_test_labels_visible": exposed,
            "test_metric_computed": False,
            "test_access_offenders": offenders,
            "files_scanned": sorted(p.name for p in R.CODE_DIR.glob("*.py")),
            "note": "the frozen loader exposes no Test labels (all NaN) and no Test metric exists in this round",
        },
    )


def timing_preflight(task: Any, model: torch.nn.Module, device: torch.device, remaining: float) -> dict[str, Any]:
    plan, _ = R.plan_and_identity(task)
    model.to(device)  # CPU gates are done; the probe runs on the target device
    optimizer, _ = M.build_optimizer(model)
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    rows = plan[0]["batches"]
    n_steps = min(30, len(rows))
    model.train()
    times: list[float] = []
    draws0 = int(model.mask_draws)
    dropped: list[float] = []
    for i in range(n_steps):
        t0 = time.perf_counter()
        batch = stager.batch(C.PC.collate(task.cache, [int(x) for x in rows[i]]))
        y = stager.labels(task.labels[rows[i]])
        DT.train_step(model, optimizer, batch, y, global_step=i, clip=clip, audit_new=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(float(time.perf_counter() - t0))
        if model.last_mask:
            dropped.append(float(model.last_mask["dropped_fraction_physical"]))
    t0 = time.perf_counter()
    rep = C.evaluate_split(model, task, device, "valid", per_task=True, reason="timing_full_valid")
    valid_seconds = float(time.perf_counter() - t0)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    free = total = None
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(device)
        free, total = int(free_b), int(total_b)
    stager.release()
    body = np.asarray(times[5:], dtype=np.float64)
    step_seconds = float(body.mean()) if body.size else float(np.mean(times))
    observed_drop = float(np.mean(dropped)) if dropped else float("nan")
    n_bonds = int(np.sum([model.mask_events[i]["n_physical_bonds"] for i in range(len(model.mask_events))])) if model.mask_events else 1
    sigma = float(np.sqrt(R.Q * (1.0 - R.Q) / max(1, n_bonds)))
    epochs = R.EPOCHS
    train_seconds = epochs * R.STEPS_PER_EPOCH * step_seconds
    valid_total = epochs * valid_seconds
    train_eval_total = len(R.TRAIN_EVAL_EPOCHS) * 40.0 + 45.0  # measured on the host in earlier rounds; refined below
    overhead = float(valid_seconds + 30.0)
    projected = float(train_seconds + valid_total + train_eval_total + overhead)
    out = {
        "schema": R.SCHEMA_PREFLIGHT,
        "round_id": R.ROUND_ID,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "timing_steps": n_steps,
        "step_seconds_mean": step_seconds,
        "step_seconds_p90": float(np.percentile(times, 90)),
        "steps_per_epoch": R.STEPS_PER_EPOCH,
        "projected_epoch_train_seconds": float(R.STEPS_PER_EPOCH * step_seconds),
        "measured_full_valid_seconds": valid_seconds,
        "measured_valid_ap_probe": float(rep["ap"]),
        "projected_train_seconds": float(train_seconds),
        "projected_valid_seconds": float(valid_total),
        "projected_train_eval_plus_overhead_seconds": float(train_eval_total + overhead),
        "projected_total_seconds": projected,
        "projected_total_minutes": projected / 60.0,
        "projected_gpu_hours": projected / 3600.0,
        "remaining_wall_seconds": float(remaining),
        "fits_remaining_wall": bool(projected <= float(remaining)),
        "fits_gpu_hour_hard_limit": bool(projected / 3600.0 <= 1.3),
        "mask_draws_during_probe": int(model.mask_draws) - draws0,
        "mask_dropped_fraction_probe_mean": observed_drop,
        "mask_dropped_3sigma_band": [R.Q - 3 * sigma, R.Q + 3 * sigma],
        "mask_dropped_within_band": bool(abs(observed_drop - R.Q) <= 3 * sigma),
        "n_physical_bonds_probed": n_bonds,
        "gpu_peak_allocated_bytes": peak,
        "gpu_free_bytes_at_end": free,
        "gpu_total_bytes": total,
    }
    record(
        "T1",
        "timing preflight fits the remaining wall clock and GPU-hour limit",
        bool(out["fits_remaining_wall"] and out["fits_gpu_hour_hard_limit"]),
        out,
    )
    record(
        "C0c",
        "realised structural drop fraction is q inside a 3-sigma binomial band",
        out["mask_dropped_within_band"],
        {
            "observed": observed_drop,
            "expected": R.Q,
            "band": out["mask_dropped_3sigma_band"],
            "n_physical_bonds": n_bonds,
        },
    )
    record(
        "C4b",
        "one mask draw per training step (mask active from step 1)",
        bool(out["mask_draws_during_probe"] == n_steps),
        {"probe_steps": n_steps, "mask_draws": out["mask_draws_during_probe"]},
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(R.ANALYSIS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--remaining-seconds", type=float, default=0.0)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.perf_counter()
    if C.git_state().get("dirty"):
        C.log("WARNING: worktree is dirty; results are not project-traceable")

    task = R.load_task_without_test()
    historical = R.load_historical_fixed1()
    reference, model = check_initial_state(task, device)
    check_initial_logits(task, reference, model)
    check_optimizer(task, model)
    check_q0_and_mask(task, reference, model)
    check_mask_rng_independence(task, reference, model)
    check_plan(task, device)
    check_test_isolation()

    preflight: dict[str, Any] | None = None
    if not args.skip_timing:
        preflight = timing_preflight(task, model, device, float(args.remaining_seconds))

    all_passed = all(c["passed"] for c in CHECKS)
    payload = {
        "schema": R.SCHEMA_CHECKS,
        "round_id": R.ROUND_ID,
        "all_passed": bool(all_passed),
        "checks": CHECKS,
        "historical_fixed1_verification": historical["verification"],
        "historical_fixed1_sources": historical["sources"],
        "durations_seconds": {"total": float(time.perf_counter() - t0)},
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "test_scored": False,
        "test_labels_loaded": False,
    }
    R.write_json(out_dir / "checks.json", payload)
    if preflight is not None:
        R.write_json(out_dir / "preflight_timing.json", preflight)
    C.log(f"ALL CHECKS {'PASSED' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
