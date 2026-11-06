"""Stage-0 audit, epoch-40 replay and paired L/R initialization checks.

``audit`` is CPU-only: checkpoint identity, AdamW restoration, RNG metadata,
the new epoch-41..80 order plan, the LR schedules, and a paired CPU first-step
probe.  ``gpu`` performs the official epoch-40 Validation AP replay, a paired
GPU first-step probe for L/R and a short throughput/memory smoke.

No official Test label is read here.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import torch  # noqa: E402

import cc_common as C  # noqa: E402
import cc_data as D  # noqa: E402
import cc_restore as R  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402


def _tensor_bytes_fingerprint(tensors: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in sorted(tensors):
        tensor = tensors[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _grad_fingerprint(model: torch.nn.Module) -> str:
    return _tensor_bytes_fingerprint(
        {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    )


def _abs(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else C.REPO / p


# ---------------------------------------------------------------------------
# CPU source audit
# ---------------------------------------------------------------------------


def audit_source_checkpoint(model_id: str, task: Any) -> dict[str, Any]:
    spec = C.SOURCE[model_id]
    path = Path(spec["path"])
    if not path.is_file():
        raise RuntimeError(f"source checkpoint missing: {path}")
    sha = C.sha256_file(path)
    if sha != spec["sha256"]:
        raise RuntimeError(f"{model_id} source sha256 {sha} != frozen {spec['sha256']}")
    payload = R.load_checkpoint(path)
    payload["_path"] = C.track_rel(path)

    checks = {
        "file_exists": True,
        "sha256": sha == spec["sha256"],
        "seed": int(payload.get("seed", -1)) == C.SEED,
        "epoch": int(payload.get("epoch", -1)) == C.EPOCH40,
        "global_step": int(payload.get("global_step", -1)) == C.STEP40,
        "dataset": payload.get("dataset") == C.DATASET,
        "n_outputs": int(payload.get("n_outputs", -1)) == C.N_TASKS,
        "optimizer_is_adamw": payload.get("optimizer") == "AdamW",
        "status_completed": payload.get("status_at_save") == "COMPLETED",
        "test_evaluated_false": payload.get("test_evaluated") is False,
        "has_optimizer_state": isinstance(payload.get("optimizer_state_dict"), dict),
        "has_rng_state": isinstance(payload.get("rng_state"), dict),
        "feature_spec_hash": payload.get("feature_spec_hash") == C.FEATURE_SPEC_HASH,
        "norm_fingerprint": payload.get("norm_fingerprint") == C.NORM_FINGERPRINT,
        "split_partition_sha256": payload.get("split_partition_sha256") == C.SPLIT_PARTITION_SHA256,
        "label_artifact_sha256": payload.get("label_artifact_sha256") == C.LABEL_ARTIFACT_SHA256,
        "cache_sha256_manifest": payload.get("cache_sha256_manifest") == C.CACHE_SHA256,
        "valid_ap_matches_history": abs(float(payload.get("valid_ap", float("nan"))) - spec["valid_ap"]) <= 1e-12,
    }
    if model_id in ("A", "G"):
        checks["arm"] = payload.get("arm") == spec["arm"]
        checks["schema"] = payload.get("schema") == "lcrc-pcba-no-mp-checkpoint-v1"
        checks["round"] = payload.get("round_id") == spec["round"]
    else:
        checks["arm"] = payload.get("arm") == spec["arm"]
        checks["schema"] = payload.get("schema") == "lcrc-pcba-structural-gain-checkpoint-v1"
        checks["round"] = payload.get("round_id") == spec["round"]
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"{model_id} source metadata checks failed: {failed}")

    model, model_audit = R.build_model(model_id, task, device="cpu")
    result = model.load_state_dict(dict(payload["state_dict"]), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{model_id} state_dict keys mismatch: {result}")
    state = payload["state_dict"]
    got = model.state_dict()
    if set(got) != set(state):
        raise RuntimeError(f"{model_id} state_dict key sets differ")
    max_abs = 0.0
    worst = None
    for key in sorted(state):
        saved = state[key].detach().cpu()
        here = got[key].detach().cpu()
        if tuple(saved.shape) != tuple(here.shape) or saved.dtype != here.dtype:
            raise RuntimeError(f"{model_id} tensor {key} shape/dtype mismatch")
        if not torch.equal(saved, here):
            diff = float((saved.float() - here.float()).abs().max().item())
            if diff > max_abs:
                max_abs, worst = diff, key
    if max_abs > 0.0:
        raise RuntimeError(f"{model_id} state_dict not bitwise equal: worst={worst} {max_abs}")
    if int(len(state)) != int(spec["state_tensors"]):
        raise RuntimeError(f"{model_id} has {len(state)} state tensors != {spec['state_tensors']}")
    if int(sum(p.numel() for p in model.parameters())) != int(spec["parameters"]):
        raise RuntimeError(f"{model_id} parameter count differs from the frozen architecture")

    optimizer = R.make_optimizer(model)
    saved_opt = payload["optimizer_state_dict"]
    if len(saved_opt.get("param_groups", [])) != 1:
        raise RuntimeError(f"{model_id} source optimizer is not one parameter group")
    optimizer.load_state_dict(copy.deepcopy(dict(saved_opt)))
    loaded_opt = R.SM.optimizer_state_cpu(optimizer)
    eq = R.SM.compare_optimizer_states(loaded_opt, saved_opt)
    if not eq.get("equal", False):
        raise RuntimeError(f"{model_id} restored AdamW state differs: {eq}")
    params = list(model.parameters())
    expected_ids = list(range(len(params)))
    groups = loaded_opt["param_groups"]
    if list(groups[0].get("params", [])) != expected_ids:
        raise RuntimeError(f"{model_id} optimizer parameter ids are not contiguous")
    if set(loaded_opt["state"]) != set(expected_ids):
        raise RuntimeError(f"{model_id} optimizer state does not cover all parameters")
    moments = []
    for pid, param in enumerate(params):
        values = loaded_opt["state"][pid]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if key not in values:
                raise RuntimeError(f"{model_id} AdamW state[{pid}] missing {key}")
        for key in ("exp_avg", "exp_avg_sq"):
            if tuple(values[key].shape) != tuple(param.shape):
                raise RuntimeError(f"{model_id} AdamW {key}[{pid}] shape mismatch")
            if values[key].dtype != torch.float32:
                raise RuntimeError(f"{model_id} AdamW {key}[{pid}] is not FP32")
        step_value = (
            float(values["step"].detach().cpu().item())
            if torch.is_tensor(values["step"])
            else float(values["step"])
        )
        if step_value != float(C.STEP40):
            raise RuntimeError(f"{model_id} AdamW step[{pid}]={step_value} != {C.STEP40}")
        moments.append({"param": int(pid), "shape": list(param.shape), "step": step_value})
    hyper = {
        "betas": tuple(groups[0].get("betas", ())),
        "eps": float(groups[0].get("eps", -1.0)),
        "weight_decay": float(groups[0].get("weight_decay", -1.0)),
    }
    if hyper["betas"] != tuple(C.COMMON_RECIPE["adam_betas"]):
        raise RuntimeError(f"{model_id} AdamW betas {hyper['betas']} != frozen")
    if hyper["eps"] != float(C.COMMON_RECIPE["adam_eps"]):
        raise RuntimeError(f"{model_id} AdamW eps != frozen")
    if hyper["weight_decay"] != float(C.COMMON_RECIPE["weight_decay"]):
        raise RuntimeError(f"{model_id} AdamW weight_decay != frozen")

    rng = payload["rng_state"]
    for key in ("python_random", "numpy", "torch_cpu", "torch_cuda"):
        if key not in rng:
            raise RuntimeError(f"{model_id} RNG state missing {key}")
    if len(rng["torch_cuda"]) != 1:
        raise RuntimeError(f"{model_id} RNG has {len(rng['torch_cuda'])} CUDA states, expected 1")
    rng_hashes = C.rng_state_hashes(rng)
    if payload.get("rng_state_hashes", {}).get("combined_sha256") != rng_hashes["combined_sha256"]:
        raise RuntimeError(f"{model_id} stored RNG hashes do not match the RNG payload")

    sched = payload.get("scheduler_state") or {}
    if int(sched.get("global_step", -1)) != C.STEP40:
        raise RuntimeError(f"{model_id} scheduler global_step mismatch")
    lr_now = [float(x) for x in sched.get("lr_now", [])]
    if not lr_now or abs(lr_now[0] - C.LR_FLOOR) > 1e-6:
        raise RuntimeError(f"{model_id} scheduler lr_now {lr_now} is not the 1e-5 endpoint")

    run_json = C.read_json(spec["run_json"])
    run_status = run_json.get("status")
    run_ap = float((run_json.get("fixed_epoch_reports") or {}).get("40", {}).get("ap", float("nan")))
    if run_status != "COMPLETED":
        raise RuntimeError(f"{model_id} historical run status {run_status} != COMPLETED")
    if abs(run_ap - spec["valid_ap"]) > 1e-12:
        raise RuntimeError(f"{model_id} historical epoch-40 AP {run_ap} != frozen {spec['valid_ap']}")

    return {
        "model": model_id,
        "source_round": spec["round"],
        "source_commit": spec["source_commit"],
        "source_run_status": run_status,
        "path": C.track_rel(path),
        "sha256": sha,
        "metadata_checks": {k: bool(v) for k, v in checks.items()},
        "state_dict_tensors": int(len(state)),
        "state_dict_keys": sorted(state),
        "state_dict_bitwise_equal": True,
        "state_dict_fingerprint": C.state_fingerprint(state),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "model_audit": model_audit,
        "adamw_param_groups": 1,
        "adamw_param_ids": expected_ids,
        "adamw_step_values": sorted({row["step"] for row in moments}),
        "adamw_moments": moments,
        "adamw_hyperparameters": hyper,
        "adamw_state_fingerprint": R.SM.optimizer_state_fingerprint(loaded_opt),
        "scheduler_state": sched,
        "rng_keys": sorted(rng),
        "rng_cuda_states": int(len(rng["torch_cuda"])),
        "rng_state_hashes": rng_hashes,
        "historical_epoch40_valid_ap": float(spec["valid_ap"]),
        "historical_epoch40_valid_ap_in_run_json": run_ap,
        "historical_epoch40_ap_abs_delta": abs(run_ap - spec["valid_ap"]),
        "test_evaluated": False,
    }


# ---------------------------------------------------------------------------
# CPU paired first-step probe
# ---------------------------------------------------------------------------


def _cpu_batch(task: Any, rows: np.ndarray) -> tuple[dict[str, Any], torch.Tensor]:
    batch = BM.collate(task.cache, np.asarray(rows).tolist())
    y = torch.as_tensor(task.labels[rows], dtype=torch.float32)
    return batch, y


def _cpu_probe(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: Any,
    rows: np.ndarray,
    rng_state: Mapping[str, Any],
    recipe: str,
    global_step: int,
    *,
    take_step: bool,
) -> dict[str, Any]:
    R.restore_rng_state(rng_state)
    model.train()
    lr = C.scheduled_lr(recipe, global_step)
    R.set_lr(optimizer, lr)
    batch, y = _cpu_batch(task, rows)
    logits = model(batch)
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite CPU preflight logits")
    loss = MET.masked_bce_loss(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite CPU preflight loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.COMMON_RECIPE["gradient_clip"])))
    if not np.isfinite(gnorm):
        raise RuntimeError("non-finite CPU preflight gradient norm")
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    out = {
        "global_step": int(global_step),
        "lr": float(lr),
        "loss": float(loss.detach().cpu().item()),
        "grad_norm": float(gnorm),
        "logits_sha256": C.sha256_bytes(
            np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float32).tobytes()
        ),
        "logits": logits.detach().cpu().clone(),
        "grad_state": grads,
        "state_fingerprint_before_step": C.state_fingerprint(model.state_dict()),
        "optimizer_fingerprint_before_step": R.optimizer_fingerprint(optimizer),
        "state_before_step": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_before_step": R.SM.optimizer_state_cpu(optimizer),
    }
    if take_step:
        optimizer.step()
        out["state_fingerprint_after_step"] = C.state_fingerprint(model.state_dict())
        out["optimizer_fingerprint_after_step"] = R.optimizer_fingerprint(optimizer)
        out["state_after_step"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        out["optimizer_after_step"] = R.SM.optimizer_state_cpu(optimizer)
    return out


def paired_cpu_audit(task: Any, plan: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)
    if rows.size != C.BATCH_SIZE:
        raise RuntimeError(f"first plan batch has {rows.size} rows != {C.BATCH_SIZE}")
    out: dict[str, Any] = {
        "schema": "lcrc-pcba-continuation-paired-init-v1",
        "round_id": C.ROUND_ID,
        "batch_rows": int(rows.size),
        "batch_rows_sha256": C.sha256_int64(rows),
        "batch_epoch": int(plan[0]["epoch"]),
        "first_step_global_step": int(C.STEP40),
        "models": {},
    }
    for model_id in C.MODEL_ORDER:
        spec = C.SOURCE[model_id]
        payload = R.load_checkpoint(spec["path"])
        payload["_path"] = C.track_rel(spec["path"])
        model_l, opt_l, audit_l = R.restore_model_and_optimizer(model_id, payload, task, device="cpu")
        model_r, opt_r, audit_r = R.restore_model_and_optimizer(model_id, payload, task, device="cpu")
        initial_equal = C.state_fingerprint(model_l.state_dict()) == C.state_fingerprint(model_r.state_dict())
        opt_equal = R.optimizer_fingerprint(opt_l) == R.optimizer_fingerprint(opt_r)
        if not initial_equal or not opt_equal:
            raise RuntimeError(f"{model_id} paired L/R restore is not identical")
        # Reproducibility: the same model/recipe restored twice gives the same first step.
        rep_a = _cpu_probe(model_l, opt_l, task, rows, payload["rng_state"], "L", C.STEP40, take_step=False)
        rep_b = _cpu_probe(model_l, opt_l, task, rows, payload["rng_state"], "L", C.STEP40, take_step=False)
        rep_logits = R.SM.compare_states({"x": rep_a["logits"]}, {"x": rep_b["logits"]})
        rep_grads = R.SM.compare_states(rep_a["grad_state"], rep_b["grad_state"])
        reproducible = (
            abs(rep_a["loss"] - rep_b["loss"]) <= 1e-9
            and abs(rep_a["grad_norm"] - rep_b["grad_norm"]) <= 1e-9
            and rep_logits["max_abs_diff"] <= 1e-7
            and rep_grads["max_abs_diff"] <= 1e-7
        )
        if not reproducible:
            raise RuntimeError(f"{model_id} first-step CPU probe is not reproducible: {rep_logits} {rep_grads}")
        # L and R start from the same LR at step 109520; both must take the same step.
        l_probe = _cpu_probe(model_l, opt_l, task, rows, payload["rng_state"], "L", C.STEP40, take_step=True)
        r_probe = _cpu_probe(model_r, opt_r, task, rows, payload["rng_state"], "R", C.STEP40, take_step=True)
        after_cmp = R.SM.compare_states(l_probe["state_after_step"], r_probe["state_after_step"])
        opt_after_cmp = R.SM.compare_optimizer_states(
            l_probe["optimizer_after_step"], r_probe["optimizer_after_step"]
        )
        after_equal = bool(after_cmp["equal"])
        opt_after_equal = bool(opt_after_cmp["equal"])
        if not after_equal and after_cmp.get("max_abs_diff", 0.0) > 1e-7:
            raise RuntimeError(f"{model_id} L/R first CPU step differs: {after_cmp}")
        if not opt_after_equal and opt_after_cmp.get("max_abs_diff", 0.0) > 1e-7:
            raise RuntimeError(f"{model_id} L/R first CPU optimizer step differs: {opt_after_cmp}")
        lr_grad_cmp = R.SM.compare_states(l_probe["grad_state"], r_probe["grad_state"])
        out["models"][model_id] = {
            "source_sha256": spec["sha256"],
            "initial_model_state_fingerprint": l_probe["state_fingerprint_before_step"],
            "initial_optimizer_fingerprint": l_probe["optimizer_fingerprint_before_step"],
            "paired_initial_state_equal": bool(initial_equal),
            "paired_initial_optimizer_equal": bool(opt_equal),
            "first_step_lr_L": l_probe["lr"],
            "first_step_lr_R": r_probe["lr"],
            "first_step_loss_L": l_probe["loss"],
            "first_step_loss_R": r_probe["loss"],
            "first_step_loss_abs_delta": abs(l_probe["loss"] - r_probe["loss"]),
            "first_step_grad_norm_L": l_probe["grad_norm"],
            "first_step_grad_norm_R": r_probe["grad_norm"],
            "first_step_logits_sha256_equal": l_probe["logits_sha256"] == r_probe["logits_sha256"],
            "first_step_grad_max_abs_diff": float(lr_grad_cmp.get("max_abs_diff", 0.0)),
            "first_step_grad_bitwise_equal": bool(lr_grad_cmp.get("equal", False)),
            "first_step_reproducible_second_restore": True,
            "after_one_step_same_lr_model_equal": bool(after_equal),
            "after_one_step_same_lr_model_max_abs_diff": float(after_cmp.get("max_abs_diff", 0.0)),
            "after_one_step_same_lr_optimizer_equal": bool(opt_after_equal),
            "after_one_step_same_lr_optimizer_max_abs_diff": float(opt_after_cmp.get("max_abs_diff", 0.0)),
            "state_fingerprint_after_same_lr_step": l_probe["state_fingerprint_after_step"],
            "optimizer_fingerprint_after_same_lr_step": l_probe["optimizer_fingerprint_after_step"],
            "lr_schedule_differs_after_first_step": C.scheduled_lr("L", C.STEP40 + 1)
            != C.scheduled_lr("R", C.STEP40 + 1),
            "lr_L_step109521": C.scheduled_lr("L", C.STEP40 + 1),
            "lr_R_step109521": C.scheduled_lr("R", C.STEP40 + 1),
            "restore_audit": audit_l,
        }
    out["all_models_passed"] = True
    out["generated_utc"] = C.utcnow()
    out["git_commit"] = C.git_commit()
    return out


# ---------------------------------------------------------------------------
# GPU replay / paired probe / smoke
# ---------------------------------------------------------------------------


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _gpu_probe(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: Any,
    rows: np.ndarray,
    device: torch.device,
    rng_state: Mapping[str, Any],
    recipe: str,
    global_step: int,
    *,
    take_step: bool,
) -> dict[str, Any]:
    R.restore_rng_state(rng_state)
    model.train()
    lr = C.scheduled_lr(recipe, global_step)
    R.set_lr(optimizer, lr)
    batch = _to_device(BM.collate(task.cache, np.asarray(rows).tolist()), device)
    y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
    logits = model(batch)
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite GPU preflight logits")
    loss = MET.masked_bce_loss(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite GPU preflight loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.COMMON_RECIPE["gradient_clip"])))
    if not np.isfinite(gnorm):
        raise RuntimeError("non-finite GPU preflight gradient norm")
    grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    out = {
        "global_step": int(global_step),
        "lr": float(lr),
        "loss": float(loss.detach().cpu().item()),
        "grad_norm": float(gnorm),
        "logits_sha256": C.sha256_bytes(
            np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float32).tobytes()
        ),
        "grad_state": grads,
        "state_fingerprint_before_step": C.state_fingerprint(model.state_dict()),
        "optimizer_fingerprint_before_step": R.optimizer_fingerprint(optimizer),
        "state_before_step": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    if take_step:
        optimizer.step()
        out["state_fingerprint_after_step"] = C.state_fingerprint(model.state_dict())
        out["optimizer_fingerprint_after_step"] = R.optimizer_fingerprint(optimizer)
        out["state_after_step"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        out["optimizer_after_step"] = R.SM.optimizer_state_cpu(optimizer)
    return out


def gpu_preflight(task: Any, plan: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    t_all = time.perf_counter()
    rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)
    out: dict[str, Any] = {
        "schema": "lcrc-pcba-continuation-preflight-gpu-v1",
        "round_id": C.ROUND_ID,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "batch_rows_sha256": C.sha256_int64(rows),
        "models": {},
        "all_passed": True,
    }
    for model_id in C.MODEL_ORDER:
        spec = C.SOURCE[model_id]
        payload = R.load_checkpoint(spec["path"])
        payload["_path"] = C.track_rel(spec["path"])
        model, optimizer, restore_audit = R.restore_model_and_optimizer(
            model_id, payload, task, device=device
        )
        rng_hashes = R.restore_rng_state(payload["rng_state"])
        t_eval = time.perf_counter()
        report = D.evaluate_validation(model, task, device, reason=f"stage0_epoch40_replay_{model_id}")
        eval_seconds = float(time.perf_counter() - t_eval)
        replay_ap = float(report["ap"])
        replay_delta = abs(replay_ap - spec["valid_ap"])
        replay_ok = replay_delta <= 1e-4

        # Paired L/R first-step probe.  Rebuild two independent restores.
        model_l, opt_l, audit_l = R.restore_model_and_optimizer(model_id, payload, task, device=device)
        model_r, opt_r, audit_r = R.restore_model_and_optimizer(model_id, payload, task, device=device)
        initial_model_equal = C.state_fingerprint(model_l.state_dict()) == C.state_fingerprint(model_r.state_dict())
        initial_opt_equal = R.optimizer_fingerprint(opt_l) == R.optimizer_fingerprint(opt_r)
        l_probe = _gpu_probe(model_l, opt_l, task, rows, device, payload["rng_state"], "L", C.STEP40, take_step=True)
        r_probe = _gpu_probe(model_r, opt_r, task, rows, device, payload["rng_state"], "R", C.STEP40, take_step=True)
        loss_delta = abs(l_probe["loss"] - r_probe["loss"])
        grad_delta = abs(l_probe["grad_norm"] - r_probe["grad_norm"])
        logits_equal = l_probe["logits_sha256"] == r_probe["logits_sha256"]
        lr_grad_cmp = R.SM.compare_states(l_probe["grad_state"], r_probe["grad_state"])
        after_cmp = R.SM.compare_states(l_probe["state_after_step"], r_probe["state_after_step"])
        opt_after_cmp = R.SM.compare_optimizer_states(
            l_probe["optimizer_after_step"], r_probe["optimizer_after_step"]
        )
        paired_ok = (
            initial_model_equal
            and initial_opt_equal
            and loss_delta <= 1e-4
            and grad_delta <= 1e-2
            and float(lr_grad_cmp.get("max_abs_diff", 0.0)) <= 1e-2
            and l_probe["lr"] == r_probe["lr"]
        )

        # Throughput / memory smoke on the first 20 actual plan batches.
        smoke_model, smoke_opt, _ = R.restore_model_and_optimizer(model_id, payload, task, device=device)
        R.restore_rng_state(payload["rng_state"])
        smoke_model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        times: list[float] = []
        smoke_success = True
        for i, batch_rows in enumerate(plan[0]["batches"][:20]):
            step = C.STEP40 + i
            R.set_lr(smoke_opt, C.scheduled_lr("L", step))
            t0 = time.perf_counter()
            batch = _to_device(BM.collate(task.cache, np.asarray(batch_rows).tolist()), device)
            y = torch.as_tensor(task.labels[batch_rows], dtype=torch.float32, device=device)
            logits = smoke_model(batch)
            loss = MET.masked_bce_loss(logits, y)
            if not (torch.isfinite(logits).all() and torch.isfinite(loss)):
                smoke_success = False
                break
            smoke_opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(smoke_model.parameters(), float(C.COMMON_RECIPE["gradient_clip"])))
            if not np.isfinite(gnorm):
                smoke_success = False
                break
            smoke_opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(float(time.perf_counter() - t0))
        peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        step_mean = float(np.mean(times)) if times else float("nan")
        projected_stage1_gpu_hours = float(step_mean * C.STEPS_PER_EPOCH * 20 * 6 / 3600.0)
        projected_stage2_gpu_hours = float(step_mean * C.STEPS_PER_EPOCH * 20 * 3 / 3600.0)
        model_ok = bool(replay_ok and paired_ok and smoke_success)
        out["all_passed"] = bool(out["all_passed"] and model_ok)
        out["models"][model_id] = {
            "source_sha256": spec["sha256"],
            "restore_audit": restore_audit,
            "epoch40_replay": {
                "official_valid_ap": replay_ap,
                "historical_valid_ap": float(spec["valid_ap"]),
                "abs_delta": replay_delta,
                "passed_1e_4": bool(replay_ok),
                "valid_bce": float(report["valid_bce"]),
                "n_tasks_scored": int(report["n_tasks_scored"]),
                "valid_logits_sha256": report.get("valid_logits_sha256"),
                "per_task_ap": report["per_task_ap"],
                "observed": report["observed"],
                "positive": report["positive"],
                "negative": report["negative"],
                "seconds": eval_seconds,
            },
            "paired_first_step": {
                "initial_model_equal": bool(initial_model_equal),
                "initial_optimizer_equal": bool(initial_opt_equal),
                "rng_state_restored_sha256": rng_hashes["combined_sha256"],
                "lr_L": l_probe["lr"],
                "lr_R": r_probe["lr"],
                "loss_L": l_probe["loss"],
                "loss_R": r_probe["loss"],
                "loss_abs_delta": loss_delta,
                "grad_norm_L": l_probe["grad_norm"],
                "grad_norm_R": r_probe["grad_norm"],
                "grad_norm_abs_delta": grad_delta,
                "logits_sha256_equal": bool(logits_equal),
                "grad_max_abs_diff": float(lr_grad_cmp.get("max_abs_diff", 0.0)),
                "grad_bitwise_equal": bool(lr_grad_cmp.get("equal", False)),
                "after_step_model_max_abs_diff": float(after_cmp.get("max_abs_diff", 0.0)),
                "after_step_optimizer_max_abs_diff": float(opt_after_cmp.get("max_abs_diff", 0.0)),
                "state_fingerprint_after_same_lr_step_L": l_probe["state_fingerprint_after_step"],
                "state_fingerprint_after_same_lr_step_R": r_probe["state_fingerprint_after_step"],
                "optimizer_fingerprint_after_same_lr_step_L": l_probe["optimizer_fingerprint_after_step"],
                "optimizer_fingerprint_after_same_lr_step_R": r_probe["optimizer_fingerprint_after_step"],
                "after_step_model_bitwise_equal": l_probe["state_fingerprint_after_step"]
                == r_probe["state_fingerprint_after_step"],
                "after_step_optimizer_bitwise_equal": l_probe["optimizer_fingerprint_after_step"]
                == r_probe["optimizer_fingerprint_after_step"],
                "passed": bool(paired_ok),
            },
            "smoke": {
                "steps": len(times),
                "step_ms_mean": step_mean * 1e3,
                "steps_per_second": 1.0 / step_mean if step_mean > 0 else float("nan"),
                "gpu_peak_allocated_bytes": peak_alloc,
                "gpu_peak_reserved_bytes": peak_reserved,
                "finite_loss_grad_logits": bool(smoke_success),
                "projected_stage1_gpu_hours_6_runs": projected_stage1_gpu_hours,
                "projected_stage2_gpu_hours_3_runs": projected_stage2_gpu_hours,
                "projected_total_gpu_hours_9_runs": projected_stage1_gpu_hours
                + projected_stage2_gpu_hours,
            },
            "passed": model_ok,
        }
        C.log(
            f"[preflight] {model_id} replay {replay_ap:.9f} (delta {replay_delta:.2e}) "
            f"paired={paired_ok} smoke={smoke_success} step_ms={step_mean*1e3:.2f}"
        )
    out["generated_utc"] = C.utcnow()
    out["git_commit"] = C.git_commit()
    out["environment"] = C.environment()
    out["total_seconds"] = float(time.perf_counter() - t_all)
    out["total_gpu_hours"] = out["total_seconds"] / 3600.0
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_audit() -> int:
    task = D.load_and_gate()
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    plan = D.make_plan_41_80(task, seed=C.SEED)
    plan_summary = D.plan_summary(plan, task, seed=C.SEED)
    plan_path = C.ANALYSIS / "epoch41_80_plan.json"
    plan_sha = D.write_plan(plan_path, plan_summary)
    sources = {m: audit_source_checkpoint(m, task) for m in C.MODEL_ORDER}
    source_payload = {
        "schema": "lcrc-pcba-continuation-source-audit-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "models": sources,
        "all_passed": True,
        "source_checks": {
            "checkpoint_exists": True,
            "sha256_matches_frozen": True,
            "state_dict_bitwise": True,
            "adamw_state_restored": True,
            "adamw_step": C.STEP40,
            "scheduler_global_step": C.STEP40,
            "rng_state_present": True,
            "status_completed": True,
            "data_identity_matches": True,
        },
    }
    C.write_json(C.ANALYSIS / "source_checkpoint_audit.json", source_payload)
    C.write_json(C.ANALYSIS / "lr_schedule_audit.json", C.schedule_audit())

    paired = paired_cpu_audit(task, plan)
    paired["plan"] = {
        "path": C.track_rel(plan_path),
        "sha256": plan_sha,
        "global_plan_sha256": plan_summary["global_plan_sha256"],
        "isolation_passed": bool(plan_summary["isolation"]["passed"]),
        "plan_seed": plan_summary["plan_seed"],
    }
    C.write_json(C.ANALYSIS / "paired_initialization_audit.json", paired)
    C.log("audit complete")
    return 0


def run_gpu(device_name: str) -> int:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    task = D.load_and_gate()
    plan = D.make_plan_41_80(task, seed=C.SEED)
    preflight = gpu_preflight(task, plan, device)
    C.write_json(C.ANALYSIS / "preflight_gpu.json", preflight)

    # Merge the GPU evidence into the paired-initialization audit file.
    paired_path = C.ANALYSIS / "paired_initialization_audit.json"
    paired = C.read_json(paired_path) if paired_path.is_file() else {}
    paired["gpu"] = preflight
    paired["gpu_all_passed"] = bool(preflight["all_passed"])
    C.write_json(paired_path, paired)
    return 0 if preflight["all_passed"] else 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("audit")
    gpu_p = sub.add_parser("gpu")
    gpu_p.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return run_audit()
    if args.mode == "gpu":
        return run_gpu(args.device)
    raise KeyError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
