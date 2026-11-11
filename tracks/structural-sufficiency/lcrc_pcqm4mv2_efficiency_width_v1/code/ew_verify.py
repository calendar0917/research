"""Correctness gate for the optimised training path (GPU preflight).

Every claim the optimised loop makes is checked against the frozen historical
code path on the *real* 1M cache:

1. inherited task identity (1M rows, 73,545 Validation rows, 7,813 steps/epoch,
   seed 0, no Train/Validation overlap),
2. analytic == built parameter count and width audit for W1 / W2 / W3, and
   ``ew_model.build_model(scale=2)`` bitwise == ``pq_model.build_model``,
3. E1 collation equals the frozen collation field-by-field (value, dtype,
   shape) on random real batches,
4. forward equality on the identical batch dict for W1 / W2 / W3,
5. **200 real training steps** frozen vs optimised (and frozen vs the adopted
   E1 path) on the real cache: per-step loss and gradient norm, final weights
   and AdamW state, RNG state, and fixed-sample predictions,
6. checkpoint save -> restore -> continue: the restored run reproduces the
   uninterrupted run bitwise,
7. atom renumbering invariance at every width,
8. official MAE == independent NumPy MAE on the full Validation split,
9. no Validation row in any training batch of sampled epochs,
10. a short end-to-end run of the formal loop for W1 and W3 (2 epochs x 6
    batches) with real checkpoints, plus a real resume from its last checkpoint,
11. the historical W2 artifacts are untouched (checkpoint SHA256 == ledger).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ew_common as C  # noqa: E402
import ew_collate as EC  # noqa: E402
import ew_data as DATA  # noqa: E402
import ew_model as M  # noqa: E402
import ew_pipeline as PIPE  # noqa: E402
import ew_train as T  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

import torch  # noqa: E402


def _clone_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[dict, dict]:
    return (
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        {
            "state": {
                k: {kk: (vv.detach().clone() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                for k, v in optimizer.state_dict()["state"].items()
            },
            "param_groups": [dict(g) for g in optimizer.state_dict()["param_groups"]],
        },
    )


def _state_equal(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> tuple[bool, float]:
    if sorted(a) != sorted(b):
        return False, float("inf")
    same = True
    delta = 0.0
    for key in a:
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            same = False
            continue
        if not torch.equal(a[key], b[key]):
            same = False
        if a[key].is_floating_point() and a[key].numel():
            delta = max(delta, float((a[key].to(torch.float64) - b[key].to(torch.float64)).abs().max()))
    return same, delta


def _optimizer_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if sorted(a["state"]) != sorted(b["state"]):
        return False
    for key in a["state"]:
        for name in a["state"][key]:
            va, vb = a["state"][key][name], b["state"][key][name]
            if torch.is_tensor(va) or torch.is_tensor(vb):
                if not torch.equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


def _rows_for(task: Mapping[str, Any], steps: int, *, seed: int, batch_size: int = C.BATCH_SIZE) -> np.ndarray:
    return np.random.default_rng(seed).permutation(C.TRAIN_ROWS)[: int(steps) * int(batch_size)]


def _run_frozen_window(
    task: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: np.ndarray,
    device: torch.device,
    *,
    start_step: int,
    collate_impl: str = "frozen",
) -> dict[str, Any]:
    """Historical loop body: pinned staging, per-step sync (collate selectable)."""
    stager = PinnedStager(device, depth=2)
    losses: list[float] = []
    grad_norms: list[float] = []
    step_ms: list[float] = []
    clip = float(C.RECIPE["gradient_clip"])
    for i in range(0, rows.size, C.BATCH_SIZE):
        chunk = rows[i : i + C.BATCH_SIZE]
        t0 = time.perf_counter()
        batch = stager.batch(EC.collate_for(task["train_cache"], chunk.tolist(), implementation=str(collate_impl)))
        y = stager.labels(task["train_y_z"][chunk])
        out = C.PQ_TRAIN.train_step(
            model,
            optimizer,
            batch,
            y,
            global_step=int(start_step) + i // C.BATCH_SIZE,
            steps_epoch=int(task["steps_epoch"]),
            steps_total=int(task["steps_total"]),
            clip=clip,
        )
        losses.append(float(out["loss"]))
        grad_norms.append(float(out["grad_norm"]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_ms.append((time.perf_counter() - t0) * 1e3)
    stager.release()
    return {"losses": losses, "grad_norms": grad_norms, "step_ms": step_ms}


def _run_optimized_window(
    task: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: np.ndarray,
    device: torch.device,
    *,
    start_step: int,
    implementation: str = "optimized",
) -> dict[str, Any]:
    impl = dict(T.IMPLEMENTATIONS[implementation])
    stager = PinnedStager(device, depth=2)
    collate_fn = lambda cache, r: EC.collate_for(cache, r, implementation=str(impl["collate"]))  # noqa: E731
    prefetcher = (
        PIPE.Prefetcher(
            task["train_cache"],
            task["train_y_z"],
            collator=collate_fn,
            batch_size=int(C.BATCH_SIZE),
            depth=int(impl["prefetch"]),
        )
        if int(impl["prefetch"]) > 0
        else None
    )
    deferred = T.DeferredLog(check_interval=int(impl["check_interval"]), clip=float(C.RECIPE["gradient_clip"]))
    clip = float(C.RECIPE["gradient_clip"])
    block_start = time.perf_counter()
    first_step = int(start_step)
    iterator = (
        prefetcher.epoch(rows)
        if prefetcher is not None
        else ((rows[i : i + C.BATCH_SIZE], None, None) for i in range(0, rows.size, C.BATCH_SIZE))
    )
    for chunk, raw, y_pref in iterator:
        if raw is None:
            raw = collate_fn(task["train_cache"], chunk.tolist())
            y_np = np.ascontiguousarray(task["train_y_z"][chunk], dtype=np.float32)
        else:
            y_np = y_pref
        batch = stager.batch(raw)
        y = stager.labels(y_np)
        out = T.train_step_deferred(
            model,
            optimizer,
            batch,
            y,
            global_step=int(start_step) + deferred.steps,
            steps_epoch=int(task["steps_epoch"]),
            steps_total=int(task["steps_total"]),
            clip=clip,
        )
        deferred.add(out)
        if deferred.ready():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            deferred.flush(block_seconds=float(time.perf_counter() - block_start), first_step=first_step)
            block_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if deferred.pending():
        deferred.flush(block_seconds=float(time.perf_counter() - block_start), first_step=first_step)
    if prefetcher is not None:
        prefetcher.close()
    stager.release()
    return {
        "losses": list(deferred.losses),
        "grad_norms": list(deferred.grad_norms),
        "step_ms": list(deferred.block_step_ms),
        "deferred": {
            "steps": deferred.steps,
            "flushes": deferred.flushes,
            "clip_events": deferred.clip_events,
        },
    }


def _fixed_sample_predictions(
    model: torch.nn.Module, task: Mapping[str, Any], device: torch.device, *, n: int = 2048, seed: int = 777
) -> np.ndarray:
    rows = np.sort(np.random.default_rng(seed).choice(73_545, size=int(n), replace=False))
    z, _info = C.PQ_TRAIN.predict_rows(model, task["valid_cache"], rows, device, batch_size=256)
    return z


def _equivalence_gate(
    task: Mapping[str, Any],
    scale: int,
    device: torch.device,
    *,
    steps: int,
    label: str,
    checkpoint_gate: bool,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    rows = _rows_for(task, steps, seed=20260000 + scale)
    # two identical models from the same seed-0 initialisation
    C.seed_everything(0)
    model_frozen, audit = M.build_model(task["spec"], task["norm"], scale=scale, seed=0, device=device)
    optimizer_frozen, _ = M.make_optimizer(model_frozen)
    state0, opt0 = _clone_state(model_frozen, optimizer_frozen)
    C.seed_everything(0)
    rng0 = C.capture_rng_state()
    model_opt, _ = M.build_model(task["spec"], task["norm"], scale=scale, seed=0, device=device)
    optimizer_opt, _ = M.make_optimizer(model_opt)
    model_opt.load_state_dict(state0, strict=True)
    optimizer_opt.load_state_dict(opt0)

    C.log(f"[verify] {label}: {steps} frozen steps ...")
    M.restore_rng_state(rng0)
    frozen = _run_frozen_window(task, model_frozen, optimizer_frozen, rows, device, start_step=0)
    rng_after_frozen = C.rng_state_hashes(C.capture_rng_state())
    C.log(f"[verify] {label}: {steps} optimised steps ...")
    M.restore_rng_state(rng0)
    optimized = _run_optimized_window(task, model_opt, optimizer_opt, rows, device, start_step=0)

    # the adopted implementation (E1 = vectorised collate only) through the
    # exact same frozen step / staging / sync path
    C.log(f"[verify] {label}: {steps} E1 (vectorised collate) steps ...")
    C.seed_everything(0)
    model_e1, _ = M.build_model(task["spec"], task["norm"], scale=scale, seed=0, device=device)
    optimizer_e1, _ = M.make_optimizer(model_e1)
    model_e1.load_state_dict(state0, strict=True)
    optimizer_e1.load_state_dict(opt0)
    M.restore_rng_state(rng0)
    e1 = _run_frozen_window(task, model_e1, optimizer_e1, rows, device, start_step=0, collate_impl="fast")

    loss_delta = max(
        (abs(a - b) for a, b in zip(frozen["losses"], optimized["losses"])), default=float("inf")
    )
    gnorm_delta = max(
        (abs(a - b) for a, b in zip(frozen["grad_norms"], optimized["grad_norms"])), default=float("inf")
    )
    weights_equal, weight_delta = _state_equal(model_frozen.state_dict(), model_opt.state_dict())
    opt_equal = _optimizer_equal(optimizer_frozen.state_dict(), optimizer_opt.state_dict())
    pred_delta = 0.0
    model_frozen.eval()
    model_opt.eval()
    with torch.inference_mode():
        p_f = _fixed_sample_predictions(model_frozen, task, device)
        p_o = _fixed_sample_predictions(model_opt, task, device)
    pred_delta = float(np.max(np.abs(p_f - p_o))) if p_f.size else 0.0
    model_frozen.train()
    model_opt.train()
    rng_optimized = C.rng_state_hashes(C.capture_rng_state())

    result: dict[str, Any] = {
        "label": label,
        "scale": int(scale),
        "steps": int(steps),
        "parameters": int(audit["total_parameters"]),
        "n_losses_frozen": len(frozen["losses"]),
        "n_losses_optimized": len(optimized["losses"]),
        "loss_max_abs_delta": float(loss_delta),
        "grad_norm_max_abs_delta": float(gnorm_delta),
        "weights_bitwise_equal": bool(weights_equal),
        "weights_max_abs_delta": float(weight_delta),
        "optimizer_state_bitwise_equal": bool(opt_equal),
        "fixed_sample_predictions_max_abs_delta": float(pred_delta),
        "frozen_step_ms_mean": float(np.mean(frozen["step_ms"])),
        "optimized_step_wall_ms_mean": float(np.mean(optimized["step_ms"])),
        "optimized_deferred": optimized["deferred"],
        "rng_state_after_frozen": rng_after_frozen,
        "rng_state_after_optimized": rng_optimized,
        "rng_states_equal": bool(rng_after_frozen == rng_optimized),
    }
    gates = {
        "losses_exact": bool(loss_delta == 0.0 and len(frozen["losses"]) == len(optimized["losses"])),
        "grad_norms_exact": bool(gnorm_delta == 0.0 and len(frozen["grad_norms"]) == len(optimized["grad_norms"])),
        "weights_bitwise_equal": bool(weights_equal),
        "optimizer_state_bitwise_equal": bool(opt_equal),
        "fixed_sample_predictions_equal": bool(pred_delta <= C.GATES["fixed_sample_prediction_abs_tol"]),
        "rng_states_equal": bool(rng_after_frozen == rng_optimized),
    }

    if checkpoint_gate:
        # save at the end of the optimised window, restore into a fresh model,
        # continue 10 steps; an uninterrupted clone does the same 10 steps.
        run = "PCQM_W3_1M" if scale == 3 else ("PCQM_W1_1M" if scale == 1 else "PCQM_W3_1M")
        task_for_ckpt = dict(task)
        task_for_ckpt["scale"] = int(scale)
        task_for_ckpt["model"] = f"W{scale}"
        task_for_ckpt["model_name"] = C.MODEL_NAMES[scale]
        payload = T.checkpoint_payload(
            task=task_for_ckpt,
            run=run,
            epoch=1,
            global_step=int(steps),
            model=model_opt,
            optimizer=optimizer_opt,
            rng_state=C.capture_rng_state(),
            model_audit=audit,
            valid_report=None,
            kind="verify_gate",
            implementation={"name": "optimized", "gate": True},
        )
        ckpt_dir = checkpoint_dir / f"gate_W{scale}"
        ckpt_path = ckpt_dir / "checkpoint_gate.pt"
        sha = T._atomic_save(payload, ckpt_path)
        cont_rows = _rows_for(task, 10, seed=20261000 + scale)
        state_before_cont, opt_before_cont = _clone_state(model_opt, optimizer_opt)
        rng_before_cont = C.capture_rng_state()
        cont_a = _run_optimized_window(task, model_opt, optimizer_opt, cont_rows, device, start_step=int(steps))
        state_a, opt_a = _clone_state(model_opt, optimizer_opt)

        C.seed_everything(0)
        model_restored, _ = M.build_model(task["spec"], task["norm"], scale=scale, seed=0, device=device)
        optimizer_restored, _ = M.make_optimizer(model_restored)
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        restore_audit = T.restore(loaded, model_restored, optimizer_restored, run=run, task=task_for_ckpt)
        cont_b = _run_optimized_window(
            task, model_restored, optimizer_restored, cont_rows, device, start_step=int(steps)
        )
        state_b, opt_b = _clone_state(model_restored, optimizer_restored)
        cont_loss_delta = max(
            (abs(a - b) for a, b in zip(cont_a["losses"], cont_b["losses"])), default=float("inf")
        )
        cont_same, cont_delta = _state_equal(state_a, state_b)
        result["checkpoint"] = {
            "path": C.track_rel(ckpt_path),
            "sha256": sha,
            "restore_audit": restore_audit,
            "continued_losses_max_abs_delta": float(cont_loss_delta),
            "continued_weights_bitwise_equal": bool(cont_same),
            "continued_weights_max_abs_delta": float(cont_delta),
            "continued_optimizer_bitwise_equal": bool(_optimizer_equal(opt_a, opt_b)),
            "continued_losses_a": cont_a["losses"][:3],
            "continued_losses_b": cont_b["losses"][:3],
        }
        gates["checkpoint_restore_continues_exactly"] = bool(
            cont_loss_delta == 0.0 and cont_same and _optimizer_equal(opt_a, opt_b)
        )
        # state_before_cont / rng_before_cont are captured for the record
        result["checkpoint"]["rng_before_continuation"] = C.rng_state_hashes(rng_before_cont)
        del state_before_cont, opt_before_cont

    e1_loss_delta = max((abs(a - b) for a, b in zip(frozen["losses"], e1["losses"])), default=float("inf"))
    e1_gnorm_delta = max((abs(a - b) for a, b in zip(frozen["grad_norms"], e1["grad_norms"])), default=float("inf"))
    e1_weights_equal, e1_weight_delta = _state_equal(model_frozen.state_dict(), model_e1.state_dict())
    e1_opt_equal = _optimizer_equal(optimizer_frozen.state_dict(), optimizer_e1.state_dict())
    model_e1.eval()
    with torch.inference_mode():
        p_e1 = _fixed_sample_predictions(model_e1, task, device)
    e1_pred_delta = float(np.max(np.abs(p_f - p_e1))) if p_f.size and p_e1.size else float("inf")
    result["e1"] = {
        "loss_max_abs_delta": float(e1_loss_delta),
        "grad_norm_max_abs_delta": float(e1_gnorm_delta),
        "weights_bitwise_equal": bool(e1_weights_equal),
        "weights_max_abs_delta": float(e1_weight_delta),
        "optimizer_state_bitwise_equal": bool(e1_opt_equal),
        "fixed_sample_predictions_max_abs_delta": float(e1_pred_delta),
        "step_ms_mean": float(np.mean(e1["step_ms"])),
    }
    gates["E1_losses_exact"] = bool(e1_loss_delta == 0.0 and len(e1["losses"]) == len(frozen["losses"]))
    gates["E1_grad_norms_exact"] = bool(e1_gnorm_delta == 0.0)
    gates["E1_weights_bitwise_equal"] = bool(e1_weights_equal)
    gates["E1_optimizer_state_bitwise_equal"] = bool(e1_opt_equal)
    gates["E1_predictions_equal"] = bool(e1_pred_delta <= C.GATES["fixed_sample_prediction_abs_tol"])
    result["gates"] = gates
    result["all_gates_pass"] = bool(all(gates.values()))
    # clean up the large objects before the next scale
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", default=str(C.ANALYSIS / "verify.json"))
    parser.add_argument("--skip-end-to-end", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    t_start = time.perf_counter()
    C.apply_deterministic_settings()
    task = DATA.load_task("PCQM_W3_1M", verify_hashes=True)
    out_path = Path(args.out)
    work_dir = out_path.parent / "verify_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    gates: dict[str, Any] = {}
    details: dict[str, Any] = {}

    # 1. inherited task identity -------------------------------------------
    details["task_identity"] = {
        "checks": task["inherited_task_checks"],
        "inherited": task["inherited_task_identity"],
        "source_round": task["source_round"],
        "cache_paths": task["cache_paths"],
        "data_isolation": task["data_isolation"],
    }
    gates["task_identity"] = bool(all(task["inherited_task_checks"].values()))

    # 2. parameters / widths / frozen W2 builder ---------------------------
    C.log("[verify] building W1 / W2 / W3 and checking parameter counts ...")
    built: dict[int, Any] = {}
    param_details: dict[str, Any] = {"analytic": C.param_table()}
    for scale in (1, 2, 3):
        C.seed_everything(0)
        model, audit = M.build_model(task["spec"], task["norm"], scale=scale, seed=0, device="cpu")
        built[scale] = (model, audit)
        param_details[f"W{scale}"] = {
            "parameters": int(audit["total_parameters"]),
            "expected": int(C.EXPECTED_PARAMETERS[scale]),
            "base": int(audit["base_parameters"]),
            "readout": int(audit["readout_parameters"]),
            "width_audit_all_match": bool(audit["width_audit"]["all_match"]),
            "state_fingerprint": audit["state_fingerprint"],
        }
    eq = M.historical_w2_equivalence(task["spec"], task["norm"], seed=0)
    param_details["historical_w2_builder_equivalence"] = eq
    details["parameters"] = param_details
    gates["parameter_counts_match_preregistered"] = all(
        int(built[s][1]["total_parameters"]) == int(C.EXPECTED_PARAMETERS[s]) for s in (1, 2, 3)
    )
    gates["width_audits_all_match"] = all(bool(built[s][1]["width_audit"]["all_match"]) for s in (1, 2, 3))
    gates["frozen_w2_builder_bitwise_equivalent"] = bool(eq["bitwise_identical"])

    # 3. E1 collation equality --------------------------------------------
    C.log("[verify] collate equality ...")
    cache = task["train_cache"]
    rng = np.random.default_rng(2026)
    collate_checks = []
    all_collate_equal = True
    for n_batch, batch_size in enumerate((1, 2, 5, 128, 128, 128)):
        rows = rng.permutation(C.TRAIN_ROWS)[:batch_size]
        a = C.PQ_DATA.collate(cache, rows.tolist())
        b = EC.collate_v2(cache, rows.tolist())
        cmp = EC.compare_batches(a, b)
        collate_checks.append({"batch_size": int(batch_size), "all_equal": cmp["all_equal"], "fields": len(cmp["fields"])})
        all_collate_equal = all_collate_equal and bool(cmp["all_equal"])
    details["collate"] = {
        "checks": collate_checks,
        "fields_per_batch": len(a),
        "gate": "value, dtype and shape equal for every field",
    }
    gates["collate_v2_equivalent"] = bool(all_collate_equal)

    # 4. forward equality on the identical batch, every width ---------------
    C.log("[verify] forward equality at W1/W2/W3 ...")
    rows = rng.permutation(C.TRAIN_ROWS)[: C.BATCH_SIZE]
    frozen_batch = C.PQ_DATA.collate(cache, rows.tolist())
    fast_batch = EC.collate_v2(cache, rows.tolist())
    forward_details: dict[str, Any] = {}
    forward_ok = True
    for scale in (1, 2, 3):
        model = built[scale][0]
        model.eval()
        with torch.inference_mode():
            z_frozen = M.predict_z(model, frozen_batch).to(torch.float64).numpy()
            z_fast = M.predict_z(model, fast_batch).to(torch.float64).numpy()
        delta = float(np.max(np.abs(z_frozen - z_fast))) if z_frozen.size else 0.0
        forward_details[f"W{scale}"] = {
            "max_abs_delta": delta,
            "bitwise": bool(np.array_equal(z_frozen, z_fast)),
            "n": int(z_frozen.size),
        }
        forward_ok = forward_ok and bool(np.array_equal(z_frozen, z_fast))
        model.train()
    details["forward"] = forward_details
    gates["forward_identical_on_identical_inputs"] = bool(forward_ok)

    # 7. atom renumbering invariance --------------------------------------
    C.log("[verify] atom-renumbering invariance ...")
    perm_details: dict[str, Any] = {}
    perm_ok = True
    for scale in (1, 2, 3):
        model = built[scale][0]
        model.eval()
        permuted, _perm = M.permute_atoms(frozen_batch, seed=20261111)
        with torch.inference_mode():
            z0 = M.predict_z(model, frozen_batch).to(torch.float64).numpy()
            z1 = M.predict_z(model, permuted).to(torch.float64).numpy()
        delta = float(np.max(np.abs(z0 - z1))) if z0.size else 0.0
        perm_details[f"W{scale}"] = {"max_abs_delta": delta, "n": int(z0.size)}
        perm_ok = perm_ok and delta <= C.GATES["atom_renumbering_abs_tol"]
        model.train()
    details["atom_renumbering"] = perm_details
    gates["atom_renumbering_invariant"] = bool(perm_ok)

    # free the CPU models before the GPU windows
    built.clear()
    del frozen_batch, fast_batch
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 5/6. 200-step equivalence + checkpoint continuation ------------------
    eq_w3 = _equivalence_gate(
        task,
        3,
        device,
        steps=int(args.steps),
        label=f"W3-{args.steps}steps",
        checkpoint_gate=True,
        checkpoint_dir=work_dir,
    )
    eq_w2 = _equivalence_gate(
        task,
        2,
        device,
        steps=int(args.steps),
        label=f"W2-{args.steps}steps",
        checkpoint_gate=False,
        checkpoint_dir=work_dir,
    )
    details["step_equivalence"] = {"W3": eq_w3, "W2": eq_w2}
    gates["step_equivalence_W3"] = bool(eq_w3["all_gates_pass"])
    gates["step_equivalence_W2"] = bool(eq_w2["all_gates_pass"])

    # 8. official MAE == numpy MAE on the full Validation split ------------
    C.log("[verify] full Validation evaluation (official vs NumPy) ...")
    C.seed_everything(0)
    model_valid, _ = M.build_model(task["spec"], task["norm"], scale=3, seed=0, device=device)
    valid_report, _pred = C.PQ_TRAIN.evaluate_split(model_valid, task, device, which="valid", batch_size=512)
    monitor_report, _ = C.PQ_TRAIN.evaluate_split(model_valid, task, device, which="monitor", batch_size=512)
    details["evaluation"] = {
        "valid_mae_eV": float(valid_report["mae_eV"]),
        "valid_mae_eV_numpy": float(valid_report["mae_eV_numpy"]),
        "valid_delta_eV": float(valid_report["crosscheck_delta_eV"]),
        "valid_n": int(valid_report["n"]),
        "monitor_mae_eV": float(monitor_report["mae_eV"]),
        "monitor_n": int(monitor_report["n"]),
    }
    gates["official_mae_equals_numpy"] = bool(
        float(valid_report["crosscheck_delta_eV"]) <= C.GATES["official_vs_numpy_mae_abs_tol"]
    )
    gates["valid_split_is_73545"] = int(valid_report["n"]) == 73_545
    gates["initial_mae_is_finite_and_reasonable"] = bool(
        np.isfinite(valid_report["mae_eV"]) and valid_report["mae_eV"] < 2.0
    )

    # 9. no Validation row in sampled training batches ---------------------
    C.log("[verify] Train/Validation isolation over sampled epochs ...")
    valid_gids = np.asarray(task["valid_cache"]["gids"], dtype=np.int64)
    train_gids_all = np.asarray(cache["gids"], dtype=np.int64)
    order_rng = np.random.default_rng(int(task["orders"]["order_seed"]))
    leaked = 0
    checked = 0
    for epoch in range(1, 4):
        order = order_rng.permutation(C.TRAIN_ROWS)
        for start in np.linspace(0, C.TRAIN_ROWS - C.BATCH_SIZE, num=100, dtype=np.int64):
            batch_rows = order[int(start) : int(start) + C.BATCH_SIZE]
            checked += int(batch_rows.size)
            gids = train_gids_all[batch_rows]
            leaked += int(np.intersect1d(gids, valid_gids, assume_unique=False).size)
    details["isolation"] = {
        "epochs_sampled": 3,
        "batches_sampled": 300,
        "molecules_checked": int(checked),
        "validation_gids_in_sampled_batches": int(leaked),
        "train_valid_overlap_in_task": int(task["data_isolation"]["train_valid_overlap"]),
        "hidden_split_rows_touched": int(task["data_isolation"]["hidden_split_rows_touched"]),
    }
    gates["no_validation_row_in_training_batches"] = bool(leaked == 0)
    gates["hidden_rows_untouched"] = int(task["data_isolation"]["hidden_split_rows_touched"]) == 0

    # 10. short end-to-end run of the formal loop for W1 / W3 ---------------
    if not args.skip_end_to_end:
        C.log("[verify] short end-to-end formal-loop run (2 epochs x 6 batches) ...")
        e2e: dict[str, Any] = {}
        for run in ("PCQM_W1_1M", "PCQM_W3_1M"):
            run_task = DATA.load_task(run, verify_hashes=False)
            out_dir = work_dir / f"e2e_{run}"
            payload = T.train(
                run_task,
                run=run,
                epochs=2,
                out_dir=out_dir,
                device=device,
                implementation="optimized",
                soft_seconds=1800.0,
                hard_seconds=2400.0,
                allow_partial_epochs=True,
                limit_batches=6,
            )
            resume = torch.load(out_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            resumed = T.train(
                run_task,
                run=run,
                epochs=3,
                out_dir=out_dir,
                device=device,
                implementation="optimized",
                resume=resume,
                soft_seconds=1800.0,
                hard_seconds=2400.0,
                allow_partial_epochs=True,
                limit_batches=6,
            )
            e2e[run] = {
                "status": payload["status"],
                "epochs_run": int(payload["epochs_run"]),
                "parameters": int(payload["parameters"]),
                "best_epoch": payload["best_epoch"],
                "valid_mae_eV": payload["fixed_epoch_reports"]["2"]["mae_eV"],
                "curve": payload["curve"],
                "resume_status": resumed["status"],
                "resume_epochs_run": int(resumed["epochs_run"]),
                "resume_audit": resumed["resume_audit"],
                "optimized_mol_per_second": payload["throughput"]["molecules_per_second_train"],
            }
        details["end_to_end"] = e2e
        gates["end_to_end_W1_completes"] = bool(
            e2e["PCQM_W1_1M"]["status"] == "COMPLETED" and e2e["PCQM_W1_1M"]["epochs_run"] == 2
        )
        gates["end_to_end_W3_completes"] = bool(
            e2e["PCQM_W3_1M"]["status"] == "COMPLETED" and e2e["PCQM_W3_1M"]["epochs_run"] == 2
        )
        gates["end_to_end_resume_completes"] = bool(
            e2e["PCQM_W1_1M"]["resume_status"] == "COMPLETED"
            and e2e["PCQM_W3_1M"]["resume_status"] == "COMPLETED"
        )

    # 11. historical artifacts untouched -----------------------------------
    hist = C.HISTORICAL
    best_path = C.OLD_ROUND / "runs/PCQM_W2_1M/checkpoint_best_valid.pt"
    hist_sha = C.sha256_file(best_path) if best_path.exists() else None
    details["historical_control"] = {
        "checkpoint": C.track_rel(best_path),
        "sha256": hist_sha,
        "expected_sha256": hist["checkpoint_best_sha256"],
        "curve_unchanged_path": C.track_rel(C.OLD_RUN_CURVE),
        "code_commit": hist["code_commit"],
    }
    gates["historical_w2_checkpoint_untouched"] = bool(hist_sha == hist["checkpoint_best_sha256"])

    # 12. recipe identity --------------------------------------------------
    gates["recipe_is_preregistered"] = bool(
        C.BATCH_SIZE == 128
        and C.TOTAL_EPOCHS == 40
        and C.STEPS_PER_EPOCH == 7813
        and C.TOTAL_STEPS == 312_520
        and float(C.RECIPE["learning_rate"]) == 3e-4
        and float(C.RECIPE["end_lr"]) == 1e-5
        and tuple(C.RECIPE["adam_betas"]) == (0.9, 0.999)
    )

    payload = {
        "schema": C.SCHEMA_VERIFY,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "device": str(device),
        "host_facts": C.host_facts(),
        "gpu_occupancy": C.gpu_occupancy(),
        "equivalent_steps": int(args.steps),
        "gates": gates,
        "details": details,
        "all_gates_pass": bool(all(bool(v) for v in gates.values())),
        "failed_gates": [k for k, v in gates.items() if not bool(v)],
        "wall_seconds": float(time.perf_counter() - t_start),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "peak_rss_mb": C.peak_rss_mb(),
        "cpu_seconds": C.cpu_seconds(),
        "environment": C.environment(),
    }
    payload["status"] = "OK" if payload["all_gates_pass"] else "FAILED"
    C.write_json(out_path, C.json_safe(payload))
    C.log(
        f"[verify] {payload['status']} gates={sum(bool(v) for v in gates.values())}/{len(gates)} "
        f"failed={payload['failed_gates']} wall={payload['wall_seconds']:.1f}s -> {C.track_rel(out_path)}"
    )
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
