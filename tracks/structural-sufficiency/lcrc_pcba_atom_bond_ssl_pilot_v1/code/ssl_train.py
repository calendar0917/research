"""The one supervised 40-epoch fine-tune of ``A_SSL_INIT_S0``.

Everything except the two encoder tensors is the historical seed-0 scratch A:
same builder, same zero-residual readout init, same non-encoder tensors, same
40-epoch Train order plan (``06f43729…``), same AdamW recipe, same masked
observed-entry BCE, same fixed evaluation epochs, same official OGB metric.

The SSL initialization is an *initialization*: no SSL module, head, optimizer
state or extra parameter enters the downstream model.  Parameter count must
stay 287,644 and the forward path is the untouched original A.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
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

import torch  # noqa: E402

import ssl_common as C  # noqa: E402
import ssl_model as SM  # noqa: E402
import nmp_checkpoint as CK  # noqa: E402
import nmp_data as D  # noqa: E402
import nmp_model as NM  # noqa: E402
import nmp_train as NT  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402
from pf_pipeline import PinnedStager, transfer_pageable  # noqa: E402

CURVE_FIELDS = list(NT.CURVE_FIELDS) + [
    "encoder_l2_from_scratch_init",
    "encoder_l2_from_ssl_init",
]
PER_TASK_FIELDS = list(NT.PER_TASK_FIELDS)


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in CURVE_FIELDS})
    return C.sha256_file(path)


def _write_per_task(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_TASK_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in PER_TASK_FIELDS})
    return C.sha256_file(path)


# ---------------------------------------------------------------------------
# construction of the historical seed-0 scratch A
# ---------------------------------------------------------------------------


def build_a0_scratch(
    spec: Any, norm: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, torch.Tensor]]:
    """Historical seed-0 scratch A (the model the SSL init re-uses)."""
    ref, ref_info, base_init = NM.build_reference(spec, norm, C.N_TASKS, C.SSL_SEED)
    model, audit = NM.build_arm(
        "A",
        spec,
        norm,
        C.N_TASKS,
        seed=C.SSL_SEED,
        device=device,
        base_init_state=base_init,
        zero_residual=True,
    )
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    base_fp = C.state_fingerprint(
        {k: v for k, v in state.items() if not k.startswith(("blocks.", "readout."))}
    )
    if base_fp != C.HISTORICAL_PARITY["base_init_fingerprint"]:
        raise RuntimeError(
            f"scratch A base initialisation {base_fp} != historical "
            f"{C.HISTORICAL_PARITY['base_init_fingerprint']}"
        )
    if int(audit["total_parameter_count"]) != C.EXPECTED_PARAMETERS:
        raise RuntimeError(f"scratch A has {audit['total_parameter_count']} parameters")
    readout_fp = C.state_fingerprint(
        {k: v for k, v in state.items() if k.startswith("readout.")}
    )
    info = {
        "model_audit": audit,
        "base_init_fingerprint": base_fp,
        "readout_init_fingerprint": readout_fp,
        "reference_info": ref_info,
        "zero_residual_at_init": bool(audit["zero_residual_at_init"]),
    }
    return model, info, state


def transfer_ssl_init(
    model: torch.nn.Module,
    scratch_state: Mapping[str, torch.Tensor],
    ssl_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Overwrite exactly ``atom_encoder.*`` and ``bond_encoder.*``."""
    encoder_state = ssl_payload.get("encoder_state")
    if not isinstance(encoder_state, dict) or not encoder_state:
        raise RuntimeError("SSL checkpoint carries no encoder_state")
    declared = str(ssl_payload.get("encoder_fingerprint", ""))
    if declared != C.state_fingerprint(encoder_state):
        raise RuntimeError("SSL checkpoint encoder fingerprint does not match its tensor payload")
    shape_before = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    keys_before = list(model.state_dict())
    params_before = C.parameter_count(model)
    scratch_enc = SM.encoder_state(scratch_state)
    install_audit = SM.install_encoders(model, encoder_state)
    if list(model.state_dict()) != keys_before:
        raise RuntimeError("transfer changed the state-dict keys")
    if {k: tuple(v.shape) for k, v in model.state_dict().items()} != shape_before:
        raise RuntimeError("transfer changed a tensor shape")
    if C.parameter_count(model) != params_before or params_before != C.EXPECTED_PARAMETERS:
        raise RuntimeError("transfer changed the parameter count")

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    non_encoder_scratch = {
        k: v for k, v in scratch_state.items() if not k.startswith(C.TRANSFERRED_PREFIXES)
    }
    non_encoder_now = {
        k: v for k, v in state.items() if not k.startswith(C.TRANSFERRED_PREFIXES)
    }
    if set(non_encoder_now) != set(non_encoder_scratch):
        raise RuntimeError("non-encoder tensor set changed")
    equal = {k: bool(torch.equal(non_encoder_scratch[k], non_encoder_now[k])) for k in non_encoder_scratch}
    if not all(equal.values()):
        bad = [k for k, ok in equal.items() if not ok]
        raise RuntimeError(f"non-transferred tensors differ from the historical seed-0 init: {bad}")
    enc_now = SM.encoder_state(state)
    if set(enc_now) != set(encoder_state):
        raise RuntimeError("encoder tensor set mismatch after install")
    if any(not torch.equal(enc_now[k], encoder_state[k]) for k in encoder_state):
        raise RuntimeError("encoder tensors were not installed exactly")
    still_scratch = sorted(k for k in enc_now if torch.equal(enc_now[k], scratch_enc[k]))
    if len(still_scratch) == len(enc_now):
        raise RuntimeError("no encoder tensor changed: the SSL init was not transferred")

    return {
        "install": install_audit,
        "encoder_keys": sorted(encoder_state),
        "encoder_tensor_sha256": {k: C.tensor_sha256(v) for k, v in enc_now.items()},
        "encoder_tensor_sha256_scratch": {k: C.tensor_sha256(v) for k, v in scratch_enc.items()},
        "encoder_fingerprint": C.state_fingerprint(enc_now),
        "encoder_fingerprint_scratch": C.state_fingerprint(scratch_enc),
        "encoder_change_vs_scratch": SM.parameter_change(scratch_enc, enc_now),
        "encoder_tensors_still_equal_to_scratch": still_scratch,
        "n_encoder_tensors": len(enc_now),
        "n_encoder_tensors_changed": len(enc_now) - len(still_scratch),
        "non_transferred_tensors_checked": len(equal),
        "non_transferred_tensors_all_equal_to_seed0_scratch": True,
        "non_transferred_tensor_sha256": {k: C.tensor_sha256(v) for k, v in non_encoder_now.items()},
        "state_dict_keys_unchanged": True,
        "state_dict_key_count": len(keys_before),
        "parameter_count": int(params_before),
        "ssl_optimizer_state_transferred": False,
        "pretext_heads_present": [k for k in keys_before if k.startswith(("atom_head.", "bond_head."))],
        "shape_probe": None,
    }


def measure_flops(model: torch.nn.Module, task: Any, device: torch.device) -> dict[str, Any]:
    """Forward FLOPs of one real 128-molecule official-Validation batch."""
    from torch.utils.flop_counter import FlopCounterMode

    rows = np.asarray(task.valid_rows[:128], dtype=np.int64)
    moved = transfer_pageable(BM.collate(task.cache, rows.tolist()), device)
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            logits = model(moved)
    flops = int(counter.get_total_flops())
    if tuple(logits.shape) != (len(rows), C.N_TASKS):
        raise RuntimeError(f"FLOP probe logits {tuple(logits.shape)}")
    model.train()
    return {
        "batch_molecules": int(len(rows)),
        "batch_atoms": int(moved["atom_feat"].shape[0]),
        "batch_bonds": int(moved["bond_feat"].shape[0]),
        "forward_flops": flops,
        "forward_flops_per_molecule": float(flops / len(rows)),
        "counter": "torch.utils.flop_counter.FlopCounterMode (forward only, FP32)",
        "batch_rows_sha256": C.sha256_int64(rows),
    }


# ---------------------------------------------------------------------------
# the single supervised run
# ---------------------------------------------------------------------------


def train(
    task: Any,
    *,
    out_dir: Path,
    device: torch.device,
    ssl_checkpoint: Path,
    soft_seconds: float,
    hard_seconds: float,
    smoke_epochs: int = 0,
    smoke_eval_epochs: Sequence[int] = (),
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke_epochs:
        # a smoke run must be impossible to mistake for the formal 40-epoch run
        if not (out_dir.parent.name == "runs" and out_dir.name.startswith("smoke")):
            raise RuntimeError("smoke runs must live under runs/smoke_*")
    ssl_payload = torch.load(Path(ssl_checkpoint), map_location="cpu", weights_only=False)
    if int(ssl_payload.get("update", -1)) != int(C.PRETRAIN["final_update_for_export"]):
        raise RuntimeError(
            f"SSL checkpoint update {ssl_payload.get('update')} != "
            f"{C.PRETRAIN['final_update_for_export']} (export must be update 10000)"
        )
    gate = dict(ssl_payload.get("pretext_gate") or {})
    if gate.get("label") != "PRETEXT_LEARNED":
        raise RuntimeError(f"pretext gate is {gate.get('label')}; refusing the 40-epoch budget")

    model, build_audit, scratch_state = build_a0_scratch(task.spec, task.norm, device)
    transfer_audit = transfer_ssl_init(model, scratch_state, ssl_payload)
    probe_rows = np.asarray(task.valid_rows[:8], dtype=np.int64)
    with torch.no_grad():
        probe_batch = transfer_pageable(BM.collate(task.cache, probe_rows.tolist()), device)
        probe_logits = model(probe_batch)
    if tuple(probe_logits.shape) != (8, C.N_TASKS):
        raise RuntimeError(f"post-transfer forward returned {tuple(probe_logits.shape)}")
    transfer_audit["shape_probe"] = {
        "batch": 8,
        "rows_sha256": C.sha256_int64(probe_rows),
        "logits_shape": list(probe_logits.shape),
        "finite": bool(torch.isfinite(probe_logits).all()),
    }
    flops = measure_flops(model, task, device)
    optimizer, opt_audit = NM.make_optimizer_scratch(model)
    plan = D.make_plan40(task, seed=int(C.SSL_SEED))
    plan_audit = D.plan_summary(plan, task)
    if plan[0]["global_plan_sha256"] != C.HISTORICAL_PARITY["global_plan_sha256"]:
        raise RuntimeError("40-epoch order plan differs from the historical seed-0 plan")
    prefix = D.plan_prefix_check(plan, C.TOTAL_EPOCHS)
    if prefix["prefix_sha256"] != C.HISTORICAL_PARITY["consumed40_prefix_sha256"]:
        raise RuntimeError("consumed-40 order prefix differs from the historical plan")
    seed_info = C.seed_everything(int(C.SSL_SEED))
    rng_start = C.rng_state_hashes(C.capture_rng_state())
    C.log(
        f"[ssl-train] transferred {transfer_audit['install']['transferred_tensors']} tensors "
        f"({transfer_audit['install']['transferred_elements']} elements), params "
        f"{transfer_audit['parameter_count']}, plan {plan[0]['global_plan_sha256'][:12]}"
    )

    eval_epochs = tuple(
        int(x) for x in (smoke_eval_epochs or C.SUPERVISED["eval_epochs"])
    )
    final_epoch = int(smoke_epochs) if smoke_epochs else int(C.TOTAL_EPOCHS)
    if smoke_epochs and final_epoch not in eval_epochs:
        raise RuntimeError("a smoke run must evaluate at its last epoch")
    clip = float(C.SUPERVISED["gradient_clip"])
    stager = PinnedStager(device, depth=2)
    watchdog = NT.Watchdog(hard_seconds)
    tracker = NT.Tracker()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    counts = D.task_counts(task)
    train_counts = counts["train"]
    strata = [
        "support_lt5"
        if train_counts["positive"][t] < 5
        else "support_5_19"
        if train_counts["positive"][t] < 20
        else "support_ge20"
        for t in range(C.N_TASKS)
    ]
    fixed_reports: dict[int, dict[str, Any]] = {}
    eval_completed: list[int] = []
    train_eval_summary: dict[str, Any] | None = None
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    snapshots: dict[int, str] = {}
    non_finite_events = 0
    clip_events_total = 0
    global_steps_run = 0
    epochs_run = 0
    stop_reason = "RUNNING"
    ssl_enc = SM.encoder_state(ssl_payload["encoder_state"])
    scratch_enc = SM.encoder_state(scratch_state)
    model.train()

    for epoch in range(1, final_epoch + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        if time.perf_counter() - t_start > float(soft_seconds):
            stop_reason = "SOFT_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        batches = plan[epoch - 1]["batches"]
        losses: list[float] = []
        step_times: list[float] = []
        grad_norms: list[float] = []
        clip_events_epoch = 0
        aborted = False
        for step, rows in enumerate(batches, start=1):
            if watchdog.stop.is_set():
                aborted = True
                break
            NM.set_lrs(optimizer, step=int(global_steps_run))
            t0 = time.perf_counter()
            batch = stager.batch(BM.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            logits = model(batch)
            if tuple(logits.shape) != (len(rows), C.N_TASKS):
                raise RuntimeError(f"model output {tuple(logits.shape)} != {(len(rows), C.N_TASKS)}")
            if not torch.isfinite(logits).all():
                non_finite_events += 1
                raise RuntimeError(f"non-finite logits at epoch {epoch} step {step}")
            loss = MET.masked_bce_loss(logits, y)
            if not torch.isfinite(loss):
                non_finite_events += 1
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
            if not math.isfinite(gnorm):
                non_finite_events += 1
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} step {step}")
            if gnorm > clip:
                clip_events_epoch += 1
            grad_norms.append(gnorm)
            optimizer.step()
            global_steps_run += 1
            losses.append(float(loss.detach().cpu().item()))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[ssl-train] hard wall limit during epoch {epoch}; partial epoch discarded")
            break
        train_seconds = float(time.perf_counter() - t_epoch)
        steps_arr = np.asarray(step_times, dtype=np.float64)
        report = None
        strict = False
        valid_seconds = 0.0
        if epoch in eval_epochs:
            model.eval()
            t_eval = time.perf_counter()
            report = D.evaluate_validation(model, task, device, reason=f"epoch_{epoch}_official_validation")
            valid_seconds = float(time.perf_counter() - t_eval)
            strict = tracker.consider(epoch, float(report["ap"]))
            fixed_reports[int(epoch)] = {
                "ap": float(report["ap"]),
                "valid_bce": float(report["valid_bce"]),
                "n_tasks_scored": int(report["n_tasks_scored"]),
                "n_tasks_with_observed_labels": int(report["n_tasks_with_observed_labels"]),
                "per_task_ap": report["per_task_ap"],
                "positive": report["positive"],
                "observed": report["observed"],
                "valid_logits_sha256": report.get("valid_logits_sha256"),
                "reason": report.get("reason", ""),
            }
            eval_completed.append(int(epoch))
            for t, ap in enumerate(report["per_task_ap"]):
                per_task_rows.append(
                    {
                        "epoch": int(epoch),
                        "task": int(t),
                        "ap": "" if ap is None else float(ap),
                        "observed": int(report["observed"][t]),
                        "positive": int(report["positive"][t]),
                        "negative": int(report["negative"][t]),
                        "train_observed": int(train_counts["observed"][t]),
                        "train_positive": int(train_counts["positive"][t]),
                        "stratum": strata[t],
                    }
                )
            if epoch == final_epoch:
                train_report = D.evaluate_train(model, task, device)
                C.write_json(
                    out_dir / f"train_eval_epoch{epoch}.json",
                    {
                        "epoch": int(epoch),
                        "ap": float(train_report["ap"]),
                        "bce": float(train_report["train_bce"]),
                        "per_task_ap": train_report["per_task_ap"],
                        "observed": train_report["observed"],
                        "positive": train_report["positive"],
                        "negative": train_report["negative"],
                        "n_tasks_scored": int(train_report["n_tasks_scored"]),
                        "evaluated_rows": int(train_report["evaluated_rows"]),
                        "seconds": float(train_report["seconds"]),
                        "ap_source": train_report["ap_source"],
                        "test_evaluated": False,
                    },
                )
                train_eval_summary = {
                    "epoch": int(epoch),
                    "ap": float(train_report["ap"]),
                    "bce": float(train_report["train_bce"]),
                    "n_tasks_scored": int(train_report["n_tasks_scored"]),
                    "evaluated_rows": int(train_report["evaluated_rows"]),
                    "per_task_ap": train_report["per_task_ap"],
                    "seconds": float(train_report["seconds"]),
                }
            model.train()
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        rng_epoch_hashes.append({"epoch": int(epoch), **rng_hashes})
        epochs_run = int(epoch)
        delta_scratch = SM.parameter_change(scratch_enc, SM.encoder_state(model))
        delta_ssl = SM.parameter_change(ssl_enc, SM.encoder_state(model))
        row = {
            "epoch": int(epoch),
            "evaluated": bool(report is not None),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "valid_ap": None if report is None else float(report["ap"]),
            "valid_bce": None if report is None else float(report["valid_bce"]),
            "n_tasks_with_observed_labels": None
            if report is None
            else int(report["n_tasks_with_observed_labels"]),
            "n_tasks_scored": None if report is None else int(report["n_tasks_scored"]),
            "train_seconds": train_seconds,
            "valid_seconds": valid_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": float(time.perf_counter() - t_start),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "step_ms_mean": float(steps_arr.mean() * 1e3) if steps_arr.size else float("nan"),
            "step_ms_p50": float(np.percentile(steps_arr, 50) * 1e3) if steps_arr.size else float("nan"),
            "step_ms_p90": float(np.percentile(steps_arr, 90) * 1e3) if steps_arr.size else float("nan"),
            "steps_per_second": float(len(step_times) / train_seconds) if train_seconds > 0 else float("nan"),
            "train_molecules_per_second": float(len(task.train_rows) / train_seconds)
            if train_seconds > 0
            else float("nan"),
            "best_epoch_so_far": None if tracker.best_epoch is None else int(tracker.best_epoch),
            "best_valid_ap_so_far": None if tracker.best_epoch is None else float(tracker.best_metric),
            "is_best": bool(strict),
            "strict_improve": bool(strict),
            "grad_norm_max_epoch": float(max(grad_norms)) if grad_norms else None,
            "grad_norm_mean_epoch": float(np.mean(grad_norms)) if grad_norms else None,
            "clip_events_epoch": int(clip_events_epoch),
            "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
            "encoder_l2_from_scratch_init": float(delta_scratch["l2_delta"]),
            "encoder_l2_from_ssl_init": float(delta_ssl["l2_delta"]),
            "notes": "" if report is not None else "validation not scheduled at this epoch",
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        _write_curve(curve_path, curve_rows)
        _write_per_task(per_task_path, per_task_rows)
        valid_txt = "" if report is None else f"{float(report['ap']):.6f}"
        C.log(
            f"[ssl-train] epoch {epoch}/{final_epoch} loss {row['train_loss']:.6f} "
            f"validAP {valid_txt} LR {row['learning_rate']:.3e} train {train_seconds:.1f}s "
            f"step_p50 {row['step_ms_p50']:.2f}ms{' *best*' if strict else ''}"
        )
        if epoch % 10 == 0 or epoch == final_epoch:
            payload = CK.checkpoint_payload(
                phase="D",
                arm="A",
                seed=int(C.SSL_SEED),
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=C.capture_rng_state(),
                plan=NT._plan_meta(plan, int(epoch), int(C.SSL_SEED)),
                model_audit=build_audit["model_audit"],
                valid_ap=None if report is None else float(report["ap"]),
                valid_report=report,
                kind=f"ssl_init_epoch{epoch}",
                extra={
                    "model_name": C.MODEL_NAME,
                    "optimizer_state_audit": opt_audit,
                    "encoder_transfer_audit": {
                        k: v
                        for k, v in transfer_audit.items()
                        if k not in ("non_transferred_tensor_sha256", "encoder_tensor_sha256")
                    },
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "status_at_save": "IN_PROGRESS" if epoch != final_epoch else "COMPLETED",
                    "flops": flops,
                },
            )
            snapshots[int(epoch)] = CK.save_checkpoint(payload, out_dir / f"checkpoint_epoch{epoch}.pt")
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()
    completed = (
        epochs_run >= final_epoch
        and set(eval_completed) == set(eval_epochs)
        and non_finite_events == 0
    )
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED_40_EPOCHS"
    elif non_finite_events:
        stop_reason = "NON_FINITE_EVENT"
    elif stop_reason == "RUNNING":
        stop_reason = "MISSING_EPOCH_OR_EVAL"
    if smoke_epochs:
        status = "SMOKE" if completed else "INCOMPLETE"
        stop_reason = "SMOKE_ONLY" if completed else stop_reason

    final_path = out_dir / f"checkpoint_epoch{final_epoch}.pt"
    total_seconds = float(time.perf_counter() - t_start)
    curve_sha = _write_curve(curve_path, curve_rows)
    per_task_sha = _write_per_task(per_task_path, per_task_rows)
    ap40 = float(fixed_reports[final_epoch]["ap"]) if final_epoch in fixed_reports else None
    if smoke_epochs:
        # smoke metrics are diagnostics only and must never enter a decision
        ap40 = None
    delta = None if ap40 is None else float(ap40 - C.AP40_A0)
    gap_train_valid = None
    if train_eval_summary is not None and ap40 is not None:
        gap_train_valid = float(train_eval_summary["ap"] - ap40)
    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "stage": "supervised",
        "model_name": C.MODEL_NAME,
        "arm": "A",
        "seed": int(C.SSL_SEED),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(transfer_audit["parameter_count"]),
        "new_parameters": int(C.EXPECTED_NEW_PARAMETERS),
        "base_parameters": int(C.EXPECTED_BASE_PARAMETERS),
        "smoke": bool(smoke_epochs),
        "smoke_epochs": int(smoke_epochs),
        "epochs_run": int(epochs_run),
        "final_epoch": int(final_epoch),
        "global_steps_run": int(global_steps_run),
        "eval_epochs_requested": list(eval_epochs),
        "eval_epochs_completed": sorted(eval_completed),
        "fixed_epoch_reports": {str(k): v for k, v in fixed_reports.items()},
        "validation_ap40": ap40,
        "historical_a0_ap40": C.AP40_A0,
        "delta_ap40_vs_historical_a0": delta,
        "decision_label": None if delta is None else C.decision_label(delta),
        "train_eval_final": train_eval_summary,
        "train_valid_ap_gap": gap_train_valid,
        "transfer_audit": transfer_audit,
        "build_audit": build_audit,
        "optimizer_state_audit": opt_audit,
        "flops": flops,
        "task_counts": counts,
        "task_strata": {str(t): strata[t] for t in range(C.N_TASKS)},
        "batch_plan": NT._plan_meta(plan, max(int(epochs_run), 0), int(C.SSL_SEED)),
        "plan_prefix": prefix,
        "training_order_summary": plan_audit["summary"],
        "training_order_isolation": plan_audit["isolation"],
        "ssl_checkpoint": C.track_rel(ssl_checkpoint),
        "ssl_checkpoint_sha256": C.sha256_file(ssl_checkpoint),
        "ssl_checkpoint_encoder_fingerprint": str(ssl_payload.get("encoder_fingerprint")),
        "ssl_updates": int(ssl_payload.get("update", -1)),
        "scores": tracker.scores,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "per_task_ap_path": C.track_rel(per_task_path),
        "per_task_ap_sha256": per_task_sha,
        "final_checkpoint": C.track_rel(final_path) if final_path.exists() else None,
        "final_checkpoint_sha256": snapshots.get(final_epoch, ""),
        "checkpoint_snapshots": {str(k): v for k, v in snapshots.items()},
        "rng_state_hashes_start": rng_start,
        "rng_state_hashes_end": C.rng_state_hashes(C.capture_rng_state()),
        "rng_epoch_hashes": rng_epoch_hashes,
        "seed_state": seed_info,
        "recipe": dict(C.SUPERVISED),
        "model_dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "non_finite_events": int(non_finite_events),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "data_load_seconds": float(task.load_seconds),
        "train_seconds": float(sum(float(r.get("train_seconds") or 0.0) for r in curve_rows)),
        "valid_seconds": float(sum(float(r.get("valid_seconds") or 0.0) for r in curve_rows)),
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"[ssl-train] done status={status} AP40={ap40} delta={delta} "
        f"params={run['parameters']} gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ssl-checkpoint", required=True)
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    parser.add_argument("--smoke-epochs", type=int, default=0)
    parser.add_argument("--smoke-eval-epochs", default="")
    args = parser.parse_args(argv)
    smoke_eval = [int(x) for x in args.smoke_eval_epochs.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "stage": "supervised",
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        return 2
    try:
        task = D.load_task()
        run = train(
            task,
            out_dir=out_dir,
            device=device,
            ssl_checkpoint=Path(args.ssl_checkpoint),
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
            smoke_epochs=int(args.smoke_epochs),
            smoke_eval_epochs=tuple(smoke_eval),
        )
        return 0 if run["status"] in ("COMPLETED", "SMOKE") else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "stage": "supervised",
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        C.log(f"[ssl-train] FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
