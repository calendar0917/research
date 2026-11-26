"""FSF follow-up v1: resume RSSC at epoch 12 and continue to epoch 25.

One process performs, in order:

1. checkpoint identity + epoch-12 replay + optimizer/scheduler/plan/data/model/
   loss/RNG/Test checks (read-only);
2. a 20-step real-batch timing probe and a one-step continuation fingerprint;
3. a budget projection; formal training starts only if the projection says
   epoch 25 fits in the remaining wall clock and inside the GPU-hour cap;
4. epochs 13..25 with full Valid every epoch, full Train at 16/20/25, light
   diagnostics, and best/last checkpoints;
5. best-checkpoint Valid+Train evaluation and run artefacts.

The update rule is the unchanged v0 ``frs_train.train_step``; the horizon is a
prefix of the frozen 40-epoch cosine schedule.  No Test split is ever read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import fsf_common as F


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def _f(x: Any) -> float | None:
    if x in (None, ""):
        return None
    return float(x)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k)
                if isinstance(v, (np.floating, np.integer, np.bool_)):
                    v = v.item()
                out[k] = "" if v is None else v
            writer.writerow(out)
    return F.sha256_file(path)


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _guard_eval(model: Any, task: Any, device: Any, split: str, **kwargs: Any) -> dict[str, Any]:
    if str(split) == "test":
        raise RuntimeError("TEST_FORBIDDEN: this round never reads or scores the Test split")
    return F.C.evaluate_split(model, task, device, split, **kwargs)


def _build_and_restore(task: Any, device: torch.device, ckpt: Mapping[str, Any]):
    model, audit = F.FM.build_frs_model(
        task.spec, task.norm, F.C.N_TASKS, arm=F.SOURCE_ARM, seed=F.SEED, device=device
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    optimizer, opt_audit = F.FT.build_optimizer(model)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, audit, opt_audit


def _first_step_record(model, optimizer, task, plan, device, stager, global_step, clip):
    rows = plan[F.RESUME_EPOCH]["batches"][0]
    batch = stager.batch(F.C.PC.collate(task.cache, rows.tolist()))
    y = stager.labels(task.labels[rows])
    out = F.FT.train_step(model, optimizer, batch, y, global_step=int(global_step), clip=float(clip))
    return out, _state_cpu(model)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def run_checks(
    task: Any,
    plan: list[dict[str, Any]],
    plan_audit: Mapping[str, Any],
    ckpt: Mapping[str, Any],
    ckpt_path: Path,
    *,
    device: torch.device,
    resume_epoch: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Mapping[str, Any]) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": dict(detail)})

    # C1 checkpoint identity -------------------------------------------------
    if ckpt_path == F.SOURCE_LAST:
        identity = F.verify_source_checkpoint(F.SOURCE_LAST, "last")
        best_identity = F.verify_source_checkpoint(F.SOURCE_BEST, "best")
        add("C1_source_checkpoint_identity", True,
            {"last": identity, "best": best_identity,
             "checkpoint_sha256": F.sha256_file(ckpt_path),
             "n_state_tensors": len(ckpt["state_dict"]),
             "state_fingerprint": F.state_fingerprint(
                 {k: v.detach().cpu() for k, v in ckpt["state_dict"].items()}),
             "checkpoint_fields": {k: ckpt.get(k) for k in (
                 "schema", "round_id", "protocol_id", "arm", "seed", "horizon", "epoch",
                 "global_step", "parameters", "valid_ap", "checkpoint_kind", "test_scored")}})
    else:
        add("C1_resumed_checkpoint_identity", True,
            {"path": F.track_rel(ckpt_path), "sha256": F.sha256_file(ckpt_path),
             "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step"),
             "state_fingerprint": F.state_fingerprint(
                 {k: v.detach().cpu() for k, v in ckpt["state_dict"].items()})})

    # C2 epoch-12 replay -----------------------------------------------------
    model, optimizer, model_audit, opt_audit = _build_and_restore(task, device, ckpt)
    model.eval()
    with torch.inference_mode():
        replay = _guard_eval(model, task, device, "valid", per_task=True,
                             reason=f"resume_replay_epoch_{resume_epoch}")
    recorded_ap = float(ckpt["valid_ap"])
    recorded_bce = None
    recorded_logits_sha = None
    recorded_per_task: dict[int, dict[str, Any]] = {}
    if ckpt_path == F.SOURCE_LAST:
        reports = F.read_json(F.SOURCE_REPORTS)
        if int(reports["epoch"]) != int(resume_epoch):
            raise RuntimeError("source best_epoch_reports epoch differs from checkpoint")
        recorded_bce = float(reports["valid"]["bce"])
        recorded_logits_sha = reports["valid"].get("logits_sha256")
        recorded_per_task = F.source_per_task(int(resume_epoch))
    else:
        rows = [r for r in _read_csv(F.RUN_DIR / "train_curve.csv") if int(r["epoch"]) == int(resume_epoch)]
        if rows:
            recorded_bce = _f(rows[-1].get("valid_bce"))
            recorded_logits_sha = None
        recorded_per_task = {}
        for r in _read_csv(F.RUN_DIR / "per_task_valid.csv"):
            if r["split"] == "valid" and int(r["epoch"]) == int(resume_epoch):
                recorded_per_task[int(r["task"])] = {
                    "ap": (None if r["ap"] == "" else float(r["ap"])),
                    "observed": int(r["observed"]), "positive": int(r["positive"]),
                    "negative": int(r["negative"])}
    per_task_diff = []
    if recorded_per_task:
        for t, rec in recorded_per_task.items():
            got = replay["per_task_ap"][t]
            if got is None or rec["ap"] is None:
                continue
            per_task_diff.append(abs(float(got) - float(rec["ap"])))
    logits_sha = replay.get("logits_sha256")
    replay_passed = (
        abs(float(replay["ap"]) - recorded_ap) <= 1e-8
        and (recorded_bce is None or abs(float(replay["bce"]) - recorded_bce) <= 1e-8)
        and (not per_task_diff or max(per_task_diff) <= 1e-7)
    )
    add("C2_epoch12_checkpoint_replay", replay_passed, {
        "epoch": int(resume_epoch),
        "replay_valid_ap": float(replay["ap"]), "recorded_valid_ap": recorded_ap,
        "delta_ap": float(replay["ap"]) - recorded_ap,
        "replay_valid_bce": float(replay["bce"]), "recorded_valid_bce": recorded_bce,
        "delta_bce": (None if recorded_bce is None else float(replay["bce"]) - recorded_bce),
        "replay_logits_sha256": logits_sha, "recorded_logits_sha256": recorded_logits_sha,
        "logits_sha_matches_recorded": (None if recorded_logits_sha is None else logits_sha == recorded_logits_sha),
        "logits_sha_note": "logits SHA is informational: CUDA index_add_ can differ at the last bit between processes; AP/BCE/per-task are the pass criteria",
        "n_tasks_replayed": len(per_task_diff),
        "max_abs_per_task_ap_diff": (max(per_task_diff) if per_task_diff else None),
        "replay_seconds": float(replay["seconds"]),
        "replay_n_tasks_scored": int(replay["n_tasks_scored"]),
    })

    # C3 optimizer ------------------------------------------------------------
    opt_state = ckpt["optimizer_state_dict"]
    group = opt_state["param_groups"][0]
    params = list(model.parameters())
    steps = {int(float(v["step"])) for v in opt_state["state"].values()}
    shape_ok = all(
        tuple(opt_state["state"][i]["exp_avg"].shape) == tuple(params[i].shape)
        and tuple(opt_state["state"][i]["exp_avg_sq"].shape) == tuple(params[i].shape)
        for i in range(len(params)) if i in opt_state["state"]
    )
    finite_ok = all(
        torch.isfinite(v).all().item()
        for st in opt_state["state"].values() for k, v in st.items() if torch.is_tensor(v) and v.is_floating_point()
    )
    loaded_equal = F.optimizer_state_equal(
        {k: v.cpu() for k, v in optimizer.state_dict()["state"].items()},
        opt_state["state"],
    )
    recipe_ok = (
        tuple(group["betas"]) == tuple(F.C.RECIPE["adam_betas"])
        and abs(float(group["eps"]) - float(F.C.RECIPE["adam_eps"])) == 0.0
        and abs(float(group["weight_decay"]) - float(F.C.RECIPE["weight_decay"])) == 0.0
        and len(opt_state["param_groups"]) == 1
    )
    add("C3_optimizer_state", all([
        len(opt_state["state"]) == len(params), steps == {F.RESUME_GLOBAL_STEP},
        shape_ok, finite_ok, loaded_equal["equal"], recipe_ok,
        len(group["params"]) == len(params) and group["params"] == list(range(len(params))),
    ]), {
        "n_parameters": len(params), "n_optimizer_states": len(opt_state["state"]),
        "optimizer_steps": sorted(steps), "expected_step": F.RESUME_GLOBAL_STEP,
        "moment_shapes_match": shape_ok, "moments_finite": finite_ok,
        "loaded_state_bitwise_equal": loaded_equal,
        "param_group": {k: group[k] for k in ("lr", "betas", "eps", "weight_decay")},
        "recipe": {k: F.C.RECIPE[k] for k in ("adam_betas", "adam_eps", "weight_decay")},
        "optimizer_audit": opt_audit,
    })

    # C4 scheduler / next LR / schedule continuity ---------------------------
    sched = ckpt["scheduler_state"]
    lr_now = float(group["lr"])
    next_lr = float(F.C.scheduled_lr(F.RESUME_GLOBAL_STEP, phase=3, kind="all"))
    hist_lr = {e: float(F.historical_lr(e)) for e in range(resume_epoch + 1, F.TARGET_HORIZON + 1)}
    hist_lr_calc = {e: float(F.C.scheduled_lr(e * F.STEPS_PER_EPOCH - 1, phase=3, kind="all"))
                    for e in hist_lr}
    hist_lr_max_diff = max(abs(hist_lr[e] - hist_lr_calc[e]) for e in hist_lr)
    add("C4_scheduler_and_next_lr", all([
        int(sched["global_step"]) == F.RESUME_GLOBAL_STEP,
        int(sched["total_steps_40_epochs"]) == F.TOTAL_STEPS_40,
        int(sched["steps_per_epoch"]) == F.STEPS_PER_EPOCH,
        abs(float(sched["lr_now"][0]) - lr_now) <= 1e-15,
        next_lr < lr_now,
        abs(next_lr - 0.00024670457946132966) <= 1e-15,
        hist_lr_max_diff <= 1e-15,
    ]), {
        "global_step": int(sched["global_step"]), "expected_global_step": F.RESUME_GLOBAL_STEP,
        "total_steps_40_epochs": int(sched["total_steps_40_epochs"]),
        "steps_per_epoch": int(sched["steps_per_epoch"]),
        "lr_at_epoch12_end_step": lr_now,
        "next_update_global_step": F.RESUME_GLOBAL_STEP,
        "next_update_lr": next_lr,
        "expected_next_lr": 0.00024670457946132966,
        "schedule_not_restarted": bool(next_lr < lr_now),
        "historical_fixed1_epoch_end_lr_max_abs_diff": float(hist_lr_max_diff),
        "historical_fixed1_epoch_end_lr": hist_lr,
    })

    # C5 order plan + next batch row ids -------------------------------------
    detail = F.historical_plan_detail()
    plan_rows = []
    for e in range(resume_epoch + 1, F.TARGET_HORIZON + 1):
        p = plan[e - 1]
        h = detail[e - 1]
        first_rows = p["batches"][0][: F.BATCH_SIZE].tolist()
        plan_rows.append({
            "epoch": e,
            "global_plan_sha256": p.get("global_plan_sha256") or plan_audit["global_plan_sha256"],
            "row_order_sha256": p["row_order_sha256"],
            "historical_row_order_sha256": h["row_order_sha256"],
            "batch_plan_sha256": p.get("batch_plan_sha256"),
            "historical_batch_plan_sha256": h["batch_plan_sha256"],
            "first_batch_rows_equal": first_rows == list(h["first_batch_rows"]),
            "n_batches": int(len(p["batches"])),
            "historical_n_batches": int(h["n_batches"]),
            "last_batch_size": int(len(p["batches"][-1])),
            "historical_last_batch_size": int(h["last_batch_size"]),
            "row_order_equal": p["row_order_sha256"] == h["row_order_sha256"],
        })
    consumed_prefix = F.C.plan_meta(plan, F.RESUME_EPOCH, F.SEED)["prefix_sha256"]
    source_prefix = ckpt["batch_plan"]["prefix_sha256"]
    iso = plan_audit["plan_isolation"]
    iso_target = [c for c in iso["checks"] if resume_epoch + 1 <= int(c["epoch"]) <= F.TARGET_HORIZON]
    iso_ok = all(
        c["is_train_permutation"] and int(c["valid_rows_in_epoch"]) == 0 and int(c["test_rows_in_epoch"]) == 0
        for c in iso_target
    )
    add("C5_order_plan_and_next_batch", all([
        plan_audit["global_plan_sha256"] == "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
        consumed_prefix == source_prefix,
        all(r["row_order_equal"] and r["first_batch_rows_equal"] and r["batch_plan_sha256"] == r["historical_batch_plan_sha256"]
            and r["n_batches"] == r["historical_n_batches"] and r["last_batch_size"] == r["historical_last_batch_size"]
            for r in plan_rows),
        iso["passed"] and iso_ok,
    ]), {
        "global_plan_sha256": plan_audit["global_plan_sha256"],
        "checkpoint_batch_plan": ckpt["batch_plan"],
        "recomputed_consumed_prefix_sha256": consumed_prefix,
        "source_prefix_equals_checkpoint": consumed_prefix == source_prefix,
        "epochs": plan_rows,
        "plan_isolation_passed": bool(iso["passed"]),
        "epochs_13_25_train_only": bool(iso_ok),
        "first_epoch13_batch_rows": plan[resume_epoch]["batches"][0][: F.BATCH_SIZE].tolist(),
        "historical_epoch13_first_batch_rows": detail[resume_epoch]["first_batch_rows"],
        "note": "historical per-epoch row/batch hashes are from the real FIXED1 seed-0 DSCR training_order.json",
    })

    # C6 data identity --------------------------------------------------------
    v0_checks = F.read_json(F.SOURCE_CHECKS)
    b0 = next(c for c in v0_checks["checks"] if c["check"] == "B0_data_identity")["detail"]
    hist_identity = b0["identity"]["historical"]
    current = {k: F.C.SOURCE.get(k) for k in hist_identity}
    data_checks = {k: bool(current[k] == hist_identity[k]) for k in hist_identity}
    counts = {
        "n_train_rows": int(len(task.train_rows)), "n_valid_rows": int(len(task.valid_rows)),
        "n_test_rows": int(len(task.test_rows)), "n_tasks": int(F.C.N_TASKS),
        "labels_shape": list(task.labels.shape),
        "test_labels_all_nan": bool(np.isnan(task.labels[task.test_rows]).all()),
    }
    count_ok = (
        counts["n_train_rows"] == 350343 and counts["n_valid_rows"] == 43793
        and counts["n_test_rows"] == 43793 and counts["n_tasks"] == 128
        and counts["test_labels_all_nan"]
    )
    add("C6_data_identity", all(data_checks.values()) and count_ok, {
        "current": current, "historical": hist_identity, "checks": data_checks,
        "counts": counts, "count_checks": {
            "train": counts["n_train_rows"] == 350343, "valid": counts["n_valid_rows"] == 43793,
            "test": counts["n_test_rows"] == 43793, "tasks": counts["n_tasks"] == 128,
            "test_labels_all_nan": counts["test_labels_all_nan"]},
        "historical_source": "fixed_source_role_separation_pilot_v0/analysis/checks.json B0",
    })

    # C7 model / loss / recipe / compute files --------------------------------
    param_ok = bool(model_audit["parameters"]["all_match"])
    rank_ok = int(model_audit["rank"]) == 32
    dropout_ok = abs(float(model.drop.p) - 0.10) <= 1e-12
    widths_ok = model_audit["widths"]["width"] == 128
    recipe_ok2 = (
        tuple(F.C.RECIPE["adam_betas"]) == (0.9, 0.999)
        and abs(float(F.C.RECIPE["adam_eps"]) - 1e-8) <= 1e-20
        and abs(float(F.C.RECIPE["weight_decay"]) - 1e-3) <= 1e-15
        and abs(float(F.C.RECIPE["gradient_clip"]) - 5.0) == 0.0
        and int(F.C.RECIPE["batch_size"]) == 128
        and int(F.C.TOTAL_STEPS_40) == 109520
    )
    git_check = F.compute_files_unchanged(ckpt["git_commit"])
    add("C7_model_loss_recipe_unchanged", all([
        param_ok, rank_ok, dropout_ok, widths_ok, recipe_ok2, git_check["all_equal"],
    ]), {
        "parameter_audit": model_audit["parameters"],
        "rank": int(model_audit["rank"]), "dropout": float(model.drop.p),
        "widths": model_audit["widths"], "recipe_ok": recipe_ok2,
        "loss_function": {"name": "masked_bce_loss",
                          "module": F.MET.masked_bce_loss.__module__,
                          "file": F.MET.masked_bce_loss.__code__.co_filename},
        "git_compute_files_unchanged": git_check,
        "checkpoint_recipe": ckpt.get("recipe"),
        "checkpoint_git_commit": ckpt["git_commit"],
    })

    # C8 parameter + RNG restoration -----------------------------------------
    current_fp = F.state_fingerprint({k: v.detach().cpu() for k, v in model.state_dict().items()})
    source_fp = F.state_fingerprint({k: v.detach().cpu() for k, v in ckpt["state_dict"].items()})
    rng_now = F.C.rng_state_hashes(F.C.capture_rng_state())
    rng_saved = F.C.rng_state_hashes(ckpt["rng_state"])
    add("C8_parameters_and_rng_restored", all([
        current_fp == source_fp, rng_now == rng_saved,
        model_audit["fingerprints"]["base_state_sha256"] != current_fp,
        model_audit["fingerprints"]["initial_state_sha256"] != current_fp,
    ]), {
        "state_fingerprint_after_load": current_fp,
        "state_fingerprint_in_checkpoint": source_fp,
        "rng_after_restore": rng_now, "rng_in_checkpoint": rng_saved,
        "rng_bitwise_equal": bool(rng_now == rng_saved),
        "build_initial_state_sha256": model_audit["fingerprints"]["initial_state_sha256"],
        "not_reinitialised": bool(current_fp != model_audit["fingerprints"]["initial_state_sha256"]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    })

    # C9 no Test --------------------------------------------------------------
    plan_has_test = any(
        int(c["test_rows_in_epoch"]) != 0 or int(c["valid_rows_in_epoch"]) != 0
        for c in iso["checks"] if int(c["epoch"]) > resume_epoch
    )
    add("C9_test_not_read", all([
        counts["test_labels_all_nan"], not plan_has_test,
        ckpt.get("test_scored") is False,
        bool(ckpt.get("test_used_for_selection") is False),
    ]), {
        "test_labels_all_nan": counts["test_labels_all_nan"],
        "plan_has_test_or_valid_rows_after_resume": bool(plan_has_test),
        "checkpoint_test_scored": ckpt.get("test_scored"),
        "checkpoint_test_used_for_selection": ckpt.get("test_used_for_selection"),
        "test_evaluated_this_round": False,
        "guard": "every evaluation call is wrapped; split=='test' raises",
    })

    passed = sum(1 for c in checks if c["passed"])
    return {
        "schema": F.SCHEMA_CHECKS,
        "round_id": F.ROUND_ID,
        "created_utc": F.utcnow(),
        "source_checkpoint": {"path": F.track_rel(ckpt_path), "sha256": F.sha256_file(ckpt_path),
                              "epoch": int(ckpt["epoch"]), "global_step": int(ckpt["global_step"])},
        "n_checks": len(checks), "n_passed": passed,
        "all_passed": bool(passed == len(checks)),
        "checks": checks,
        "replay_seconds": float(replay["seconds"]),
        "model_audit": model_audit,
        "optimizer_audit": opt_audit,
        "git_commit": F.C.git_commit(),
        "environment": F.C.environment(),
        "host_facts": F.C.host_facts(),
    }


# ---------------------------------------------------------------------------
# timing probe
# ---------------------------------------------------------------------------


def timing_probe(task, plan, ckpt, device, *, steps: int) -> dict[str, Any]:
    model, optimizer, _audit, _opt = _build_and_restore(task, device, ckpt)
    rng = F.restore_rng_state(ckpt["rng_state"])
    stager = F.FT.PinnedStager(device, depth=2)
    model.train()
    clip = float(F.C.RECIPE["gradient_clip"])
    batches = plan[F.RESUME_EPOCH]["batches"]
    times: list[float] = []
    first: dict[str, Any] | None = None
    state_after_first = None
    for i in range(int(steps)):
        rows = batches[i]
        t0 = time.perf_counter()
        batch = stager.batch(F.C.PC.collate(task.cache, rows.tolist()))
        y = stager.labels(task.labels[rows])
        out = F.FT.train_step(model, optimizer, batch, y,
                              global_step=F.RESUME_GLOBAL_STEP + i, clip=clip)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(float(time.perf_counter() - t0))
        if i == 0:
            first = {
                "global_step": F.RESUME_GLOBAL_STEP,
                "loss": float(out["loss"]), "lr": float(out["lr"]),
                "grad_norm": float(out["grad_norm"]),
                "rng_before": rng,
                "rng_after": F.C.rng_state_hashes(F.C.capture_rng_state()),
                "state_fingerprint_after": F.state_fingerprint(_state_cpu(model)),
                "lowrank_grads": {k: v for k, v in out["lowrank_grads"].items()},
            }
            state_after_first = _state_cpu(model)
    stager.release()
    arr = np.asarray(times, dtype=np.float64)
    warm = int(min(3, max(0, arr.size - 1)))
    steady = arr[warm:] if arr.size > warm else arr
    step_med = float(np.median(steady))
    step_mean = float(steady.mean()) if steady.size else float(arr.mean())
    torch.save(state_after_first, F.ANALYSIS / "next_step_state_1step.pt")
    return {
        "schema": F.SCHEMA_TIMING,
        "round_id": F.ROUND_ID,
        "created_utc": F.utcnow(),
        "measured_steps": int(steps),
        "step_ms_median": step_med * 1e3,
        "step_ms_mean": step_mean * 1e3,
        "step_ms_raw_mean": float(arr.mean() * 1e3),
        "warmup_steps_excluded": warm,
        "train_epoch_seconds_projected": float(step_med * F.STEPS_PER_EPOCH),
        "next_step": first,
        "next_step_state_path": F.track_rel(F.ANALYSIS / "next_step_state_1step.pt"),
        "next_step_state_sha256": F.sha256_file(F.ANALYSIS / "next_step_state_1step.pt"),
        "deterministic_seed": F.SEED,
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_followup(
    task,
    plan,
    plan_audit,
    ckpt,
    ckpt_path,
    device,
    *,
    resume_epoch: int,
    target_epoch: int,
    wall_seconds: float,
    finalize_reserve: float,
    timing: Mapping[str, Any],
    checks: Mapping[str, Any],
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = F.utcnow()
    resume_train_seconds = 0.0
    # time already spent in checks/probe belongs to this round
    elapsed_before_train = float(checks.get("seconds") or 0.0) + float(timing.get("seconds") or 0.0)

    F.RUN_DIR.mkdir(parents=True, exist_ok=True)
    best_copy = F.copy_source_best_to_run_dir()

    model, optimizer, model_audit, opt_audit = _build_and_restore(task, device, ckpt)
    rng_restored = F.restore_rng_state(ckpt["rng_state"])
    if F.state_fingerprint({k: v.detach().cpu() for k, v in model.state_dict().items()}) != \
            F.state_fingerprint({k: v.detach().cpu() for k, v in ckpt["state_dict"].items()}):
        raise RuntimeError("model state changed between checks and training start")

    # history / best
    if ckpt_path == F.SOURCE_LAST:
        source_run = F.read_json(F.SOURCE_RUN_JSON)
        best_epoch = int(source_run["best_epoch"])
        best_valid_ap = float(source_run["best_valid_ap"])
        prior_curve = F.source_curve(resume_epoch)
        prior_per_task: list[dict[str, Any]] = []
        prior_usage: list[dict[str, Any]] = []
    else:
        state_path = F.RUN_DIR / "resume_state.json"
        state = F.read_json(state_path) if state_path.is_file() else {}
        best_epoch = int(state.get("best_epoch", resume_epoch))
        best_valid_ap = float(state.get("best_valid_ap", ckpt["valid_ap"]))
        prior_curve = _read_csv(F.RUN_DIR / "train_curve.csv")
        prior_per_task = _read_csv(F.RUN_DIR / "per_task_valid.csv")
        usage_path = F.RUN_DIR / "structural_usage.json"
        prior_usage = (F.read_json(usage_path).get("history", []) if usage_path.is_file() else [])

    curve_fields = ["source"] + list(F.FT.CURVE_FIELDS)
    curve_rows = list(prior_curve)
    per_task_rows = list(prior_per_task)
    usage_history = list(prior_usage)
    curve_path = F.RUN_DIR / "train_curve.csv"
    per_task_path = F.RUN_DIR / "per_task_valid.csv"
    usage_path = F.RUN_DIR / "structural_usage.json"
    best_path = F.RUN_DIR / "checkpoint_best.pt"
    last_path = F.RUN_DIR / "checkpoint_last.pt"

    clip = float(F.C.RECIPE["gradient_clip"])
    counts = F.C.SD.task_counts(task)
    diag_rows = [int(x) for x in task.valid_rows[: int(F.C.DIAG_VALID_ROWS)]]
    diag_batch = F.C.PC.collate(task.cache, diag_rows)
    diag_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in diag_batch.items()}
    stager = F.FT.PinnedStager(device, depth=2)

    remaining = max(300.0, float(wall_seconds) - finalize_reserve - elapsed_before_train)
    watchdog = F.FT.Watchdog(remaining)

    probe_reference = None
    probe_state = None
    probe_path = F.ANALYSIS / "next_step_state_1step.pt"
    if probe_path.is_file():
        probe_state = torch.load(probe_path, map_location="cpu", weights_only=False)
    probe_json = F.read_json(F.ANALYSIS / "timing.json")
    probe_reference = probe_json.get("next_step")

    def eval_all(epoch: int, train_eval: bool) -> tuple[dict[str, Any], float]:
        before = F.C.rng_state_hashes(F.C.capture_rng_state())["combined_sha256"]
        model.eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            valid_rep = _guard_eval(model, task, device, "valid", per_task=True,
                                    reason=f"epoch_{epoch}_full_valid")
            train_rep = (_guard_eval(model, task, device, "train", per_task=False,
                                     reason=f"epoch_{epoch}_full_train") if train_eval else None)
        seconds = float(time.perf_counter() - t0)
        after = F.C.rng_state_hashes(F.C.capture_rng_state())["combined_sha256"]
        if before != after:
            raise RuntimeError("evaluation moved the training RNG state")
        model.train()
        return {"valid": valid_rep, "train": train_rep}, seconds

    epochs_run = int(ckpt["epoch"])
    global_step = int(ckpt["global_step"])
    if epochs_run != resume_epoch or global_step != epochs_run * F.STEPS_PER_EPOCH:
        raise RuntimeError(f"unexpected resume coordinates {epochs_run}/{global_step}")
    stop_reason = "RUNNING"
    first_step_check: dict[str, Any] | None = None
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0

    def update_peak() -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    for epoch in range(int(ckpt["epoch"]) + 1, target_epoch + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        batches = plan[epoch - 1]["batches"]
        losses: list[float] = []
        step_times: list[float] = []
        grad_norms: list[float] = []
        clip_events = 0
        lowrank_grad_acc = {f"lowrank.{b}.{p}.weight": [] for b in (0, 1) for p in ("A", "B")}
        lowrank_total: list[float] = []
        inherited_total: list[float] = []
        w2_total: list[float] = []
        aborted = False
        for i, rows in enumerate(batches):
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch = stager.batch(F.C.PC.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            global_step_i = (epoch - 1) * F.STEPS_PER_EPOCH + i
            out = F.FT.train_step(model, optimizer, batch, y, global_step=int(global_step_i), clip=clip)
            losses.append(float(out["loss"]))
            grad_norms.append(float(out["grad_norm"]))
            clip_events += int(out["clipped"])
            lowrank_total.append(float(out["grad_norm_lowrank_total"]))
            inherited_total.append(float(out["grad_norm_inherited_new_total"]))
            w2_total.append(float(out["grad_norm_w2_total"]))
            for name, value in out["lowrank_grads"].items():
                if value is not None:
                    lowrank_grad_acc[name].append(float(value))
            if first_step_check is None:
                state_now = _state_cpu(model)
                max_diff = None
                equal = None
                if probe_state is not None:
                    diffs = [float((state_now[k].float() - probe_state[k].float()).abs().max().item())
                             for k in state_now if k in probe_state]
                    max_diff = max(diffs) if diffs else None
                    equal = bool(max_diff is not None and max_diff <= 1e-5)
                first_step_check = {
                    "global_step": int(global_step_i),
                    "loss": float(out["loss"]), "lr": float(out["lr"]),
                    "grad_norm": float(out["grad_norm"]),
                    "probe_loss": None if probe_reference is None else probe_reference.get("loss"),
                    "probe_lr": None if probe_reference is None else probe_reference.get("lr"),
                    "probe_grad_norm": None if probe_reference is None else probe_reference.get("grad_norm"),
                    "loss_abs_diff": (None if probe_reference is None else abs(float(out["loss"]) - float(probe_reference["loss"]))),
                    "lr_abs_diff": (None if probe_reference is None else abs(float(out["lr"]) - float(probe_reference["lr"]))),
                    "grad_norm_abs_diff": (None if probe_reference is None else abs(float(out["grad_norm"]) - float(probe_reference["grad_norm"]))),
                    "state_vs_probe_max_abs_diff": max_diff,
                    "state_matches_probe": equal,
                    "rng_after": F.C.rng_state_hashes(F.C.capture_rng_state()),
                }
                if equal is False:
                    raise RuntimeError(f"first resumed step does not match the independent probe: max_diff={max_diff}")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            F.log(f"[RSSC] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        train_eval = epoch in F.TRAIN_EVAL_EPOCHS
        reports, eval_seconds = eval_all(epoch, train_eval=train_eval)
        valid_rep = reports["valid"]
        valid_ap = float(valid_rep["ap"])
        for t, ap in enumerate(valid_rep["per_task_ap"]):
            per_task_rows.append({
                "arm": "RSSC", "epoch": int(epoch), "split": "valid", "task": int(t),
                "ap": "" if ap is None else float(ap),
                "observed": int(valid_rep["observed"][t]),
                "positive": int(valid_rep["positive"][t]),
                "negative": int(valid_rep["negative"][t]),
            })
        strict = best_epoch is None or valid_ap > float(best_valid_ap)
        if strict:
            best_epoch, best_valid_ap = int(epoch), valid_ap
            payload = F.FT.checkpoint_payload(
                arm="RSSC", seed=F.SEED, horizon=target_epoch, epoch=int(epoch),
                global_step=int(epoch * F.STEPS_PER_EPOCH), model=model, optimizer=optimizer,
                rng_state=F.C.capture_rng_state(),
                plan_meta=F.C.plan_meta(plan, int(epoch), F.SEED),
                model_audit=model_audit, valid_ap=best_valid_ap, kind="best_valid_ap",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "resume_audit": {
                        "source_checkpoint": F.track_rel(ckpt_path),
                        "source_checkpoint_sha256": F.sha256_file(ckpt_path),
                        "source_epoch": int(resume_epoch),
                        "source_global_step": int(F.RESUME_GLOBAL_STEP),
                        "rng_restored": rng_restored,
                        "resumed_by": F.ROUND_ID,
                    },
                },
            )
            payload["round_id"] = F.ROUND_ID
            payload["protocol_id"] = F.PROTOCOL_ID
            payload["checkpoint_origin"] = "fixed-source-role-separation-followup-v1/resumed-from-v0-epoch12"
            F.FT._atomic_save(payload, best_path)

        # light diagnostics
        usage = F.FT.diagnostic_usage(model, diag_batch)
        scales = F.FSM._block_scales(model, diag_batch)
        model.train()
        dstat = {str(i): F.FSM._d_stat(model.lowrank[i], model.blocks[i].fc1) for i in (0, 1)}
        usage_history.append({
            "epoch": int(epoch), "global_step": int(epoch * F.STEPS_PER_EPOCH),
            **usage,
            "light": {"D_stat": dstat,
                      "scales": {k: scales[k] for k in ("0", "1")}},
        })
        update_peak()
        epochs_run = int(epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        cumulative = float(time.perf_counter() - t_start)

        def _mean(values: Sequence[float]) -> float | None:
            return float(np.mean(values)) if len(values) else None

        row: dict[str, Any] = {
            "source": "RSSC_followup_v1_epochs_13_25",
            "arm": "RSSC", "epoch": int(epoch), "global_step": int(epoch * F.STEPS_PER_EPOCH),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_bce_step_mean": float(np.mean(losses)) if losses else float("nan"),
            "train_evaluated": bool(train_eval),
            "train_bce": (float(reports["train"]["bce"]) if reports["train"] else None),
            "train_ap": (float(reports["train"]["ap"]) if reports["train"] else None),
            "valid_bce": float(valid_rep["bce"]), "valid_ap": valid_ap,
            "n_tasks_scored_valid": int(valid_rep["n_tasks_scored"]),
            "epoch_train_seconds": train_seconds,
            "valid_seconds": float(valid_rep["seconds"]),
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
            "step_ms_p50": float(np.percentile(arr, 50) * 1e3) if arr.size else float("nan"),
            "gap_norm_max": float(max(grad_norms)) if grad_norms else None,
            "gap_clip_events": int(clip_events),
            "best_epoch_so_far": int(best_epoch),
            "best_valid_ap_so_far": float(best_valid_ap),
            "is_best": bool(strict),
        }
        row.update({k: usage.get(k) for k in (
            "R1", "R2", "Rtotal", "rms_h1_minus_h0", "rms_h2_minus_h1",
            "rms_z1", "rms_m1", "rms_z_target", "rms_m2",
            "rms_fc1_term_0", "rms_correction_0", "rms_fc1_term_1", "rms_correction_1",
            "lowrank_weight_norm_0", "lowrank_weight_norm_1")})
        row["grad_norm_lowrank_0_A"] = _mean(lowrank_grad_acc["lowrank.0.A.weight"])
        row["grad_norm_lowrank_0_B"] = _mean(lowrank_grad_acc["lowrank.0.B.weight"])
        row["grad_norm_lowrank_1_A"] = _mean(lowrank_grad_acc["lowrank.1.A.weight"])
        row["grad_norm_lowrank_1_B"] = _mean(lowrank_grad_acc["lowrank.1.B.weight"])
        row["grad_norm_lowrank_total"] = _mean(lowrank_total)
        row["grad_norm_inherited_new_total"] = _mean(inherited_total)
        row["grad_norm_w2_total"] = _mean(w2_total)
        curve_rows.append(row)
        _write_csv(curve_path, curve_fields, curve_rows)
        _write_csv(per_task_path, ["arm", "epoch", "split", "task", "ap", "observed", "positive", "negative"],
                   per_task_rows)
        F.write_json(usage_path, {
            "schema": "fsf-structural-usage-v1", "round_id": F.ROUND_ID, "arm": "RSSC",
            "diagnostic_batch": "first official Validation rows, eval mode",
            "history": usage_history,
        })
        F.write_json(F.RUN_DIR / "resume_state.json", {
            "schema": "fsf-resume-state-v1", "round_id": F.ROUND_ID,
            "best_epoch": int(best_epoch), "best_valid_ap": float(best_valid_ap),
            "epochs_run": int(epochs_run), "global_steps_run": int(global_step),
            "cumulative_seconds": cumulative, "updated_utc": F.utcnow(),
        })
        F.log(f"[RSSC] ep {epoch}/{target_epoch} bce {row['train_bce_step_mean']:.6f} "
              f"validAP {valid_ap:.6f} LR {row['learning_rate']:.3e} "
              f"train {train_seconds:.1f}s valid {row['valid_seconds']:.1f}s "
              f"{'*best*' if strict else ''}")
        payload_last = F.FT.checkpoint_payload(
            arm="RSSC", seed=F.SEED, horizon=target_epoch, epoch=int(epoch),
            global_step=int(epoch * F.STEPS_PER_EPOCH), model=model, optimizer=optimizer,
            rng_state=F.C.capture_rng_state(),
            plan_meta=F.C.plan_meta(plan, int(epoch), F.SEED),
            model_audit=model_audit, valid_ap=valid_ap, kind="last_epoch",
            extra={"resume_audit": {
                "source_checkpoint": F.track_rel(ckpt_path),
                "source_checkpoint_sha256": F.sha256_file(ckpt_path),
                "source_epoch": int(resume_epoch),
                "resumed_by": F.ROUND_ID}},
        )
        payload_last["round_id"] = F.ROUND_ID
        payload_last["protocol_id"] = F.PROTOCOL_ID
        payload_last["checkpoint_origin"] = "fixed-source-role-separation-followup-v1/resumed-from-v0-epoch12"
        F.FT._atomic_save(payload_last, last_path)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()
    completed = epochs_run >= target_epoch and best_epoch is not None
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED"

    # best-checkpoint evaluation (Valid per task + full Train) -----------------
    best_eval: dict[str, Any] | None = None
    best_checkpoint_sha = F.sha256_file(best_path) if best_path.is_file() else None
    if best_path.is_file():
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            breps = {
                "valid": _guard_eval(model, task, device, "valid", per_task=True,
                                     reason="best_checkpoint_valid"),
                "train": _guard_eval(model, task, device, "train", per_task=False,
                                     reason="best_checkpoint_train"),
            }
        best_eval = {
            "epoch": int(payload["epoch"]), "seconds": float(time.perf_counter() - t0),
            "valid": {k: breps["valid"][k] for k in (
                "ap", "bce", "per_task_ap", "observed", "positive", "negative",
                "n_tasks_scored", "logits_sha256", "probs_sha256")},
            "train": {k: breps["train"][k] for k in ("ap", "bce", "n_tasks_scored", "logits_sha256")},
            "selection_metric": "valid.ap", "test_scored": False,
        }
        best_eval["valid_ap_minus_recorded"] = float(best_eval["valid"]["ap"]) - float(best_valid_ap)
        if abs(best_eval["valid_ap_minus_recorded"]) > 1e-4:
            raise RuntimeError(f"best-checkpoint Valid AP {best_eval['valid']['ap']} != recorded {best_valid_ap}")
        F.write_json(F.RUN_DIR / "best_checkpoint_eval.json", best_eval)

    # epoch-12 light baseline from the source checkpoint + v0 usage -----------
    source_usage = F.read_json(F.SOURCE_USAGE)["history"]
    src12 = next((r for r in source_usage if int(r["epoch"]) == resume_epoch), None)
    src_ckpt = F.load_checkpoint(F.SOURCE_LAST)
    baseline = {}
    if src12 is not None:
        for i in (0, 1):
            W = src_ckpt["state_dict"][f"blocks.{i}.fc1.weight"]
            D = src_ckpt["state_dict"][f"lowrank.{i}.A.weight"] @ src_ckpt["state_dict"][f"lowrank.{i}.B.weight"]
            baseline[str(i)] = {
                "epoch": int(resume_epoch),
                "D_frobenius": float(D.norm().item()), "W_frobenius": float(W.norm().item()),
                "R_ratio": float(D.norm().item() / W.norm().item()),
                "rms_fc1_term": src12.get(f"rms_fc1_term_{i}"),
                "rms_lowrank_correction": src12.get(f"rms_correction_{i}"),
                "correction_to_fc1_ratio": (
                    float(src12[f"rms_correction_{i}"]) / float(src12[f"rms_fc1_term_{i}"])
                    if src12.get(f"rms_fc1_term_{i}") else None),
            }
    mechanism_light = {
        "schema": "fsf-mechanism-light-v1", "round_id": F.ROUND_ID,
        "note": "light only: D/W and correction/fc1 RMS; no frozen deletion intervention",
        "epoch12_baseline": baseline,
        "history": usage_history,
        "best_epoch": int(best_epoch), "best_valid_ap": float(best_valid_ap),
    }
    F.write_json(F.RUN_DIR / "mechanism_light.json", mechanism_light)

    total_seconds = float(time.perf_counter() - t_start) + elapsed_before_train
    final_row = curve_rows[-1] if curve_rows else {}
    valid_rows = [r for r in curve_rows if _f(r.get("valid_ap")) is not None]
    last3 = valid_rows[-3:]
    last5 = valid_rows[-5:]
    endpoint = next((r for r in curve_rows if int(r["epoch"]) == target_epoch), None)
    run = {
        "schema": F.SCHEMA_RUN, "round_id": F.ROUND_ID, "protocol_id": F.PROTOCOL_ID,
        "arm": "RSSC", "base_computation": "FIXED1",
        "role_branch": "low-rank correction uses z - m (role-separated input)",
        "seed": F.SEED, "horizon": int(target_epoch), "status": status,
        "stop_reason": stop_reason, "dataset": F.C.DATASET, "n_outputs": int(F.C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "parameter_audit": model_audit["parameters"], "rank": 32,
        "epochs_run": int(epochs_run), "global_steps_run": int(epochs_run * F.STEPS_PER_EPOCH),
        "resumed_from_epoch": int(resume_epoch),
        "resumed_from_global_step": int(F.RESUME_GLOBAL_STEP),
        "eval_protocol": {"valid_every_epoch": True, "train_eval_epochs": list(F.TRAIN_EVAL_EPOCHS),
                          "train_eval_best_checkpoint_at_end": True, "test_evaluated": False},
        "task_counts": counts, "model_audit": model_audit, "optimizer_state_audit": opt_audit,
        "plan_identity": plan_audit, "batch_plan": F.C.plan_meta(plan, int(epochs_run), F.SEED),
        "best_epoch": int(best_epoch), "best_valid_ap": float(best_valid_ap),
        "best_valid_bce": (float(best_eval["valid"]["bce"]) if best_eval else None),
        "horizon_endpoint": {
            "epoch": int(epochs_run),
            "valid_ap": (None if endpoint is None else _f(endpoint.get("valid_ap"))),
            "valid_bce": (None if endpoint is None else _f(endpoint.get("valid_bce"))),
            "train_ap": (None if endpoint is None else _f(endpoint.get("train_ap"))),
            "train_bce": (None if endpoint is None else _f(endpoint.get("train_bce"))),
        },
        "last3_mean_valid_ap": float(np.mean([_f(r["valid_ap"]) for r in last3])) if last3 else None,
        "last3_epochs": [int(r["epoch"]) for r in last3],
        "last5_mean_valid_ap": float(np.mean([_f(r["valid_ap"]) for r in last5])) if last5 else None,
        "last5_epochs": [int(r["epoch"]) for r in last5],
        "best_checkpoint_eval": best_eval,
        "best_checkpoint": F.track_rel(best_path), "best_checkpoint_sha256": best_checkpoint_sha,
        "last_checkpoint": F.track_rel(last_path),
        "last_checkpoint_sha256": (F.sha256_file(last_path) if last_path.is_file() else None),
        "curve": F.track_rel(curve_path), "curve_sha256": F.sha256_file(curve_path),
        "per_task_valid": F.track_rel(per_task_path), "per_task_valid_sha256": F.sha256_file(per_task_path),
        "structural_usage": F.track_rel(usage_path), "mechanism_light": F.track_rel(F.RUN_DIR / "mechanism_light.json"),
        "recipe": F.C.recipe("FIXED1", F.SEED, int(target_epoch)),
        "model_dropout": float(model.drop.p), "precision": "fp32",
        "grad_clip": clip,
        "gpu_seconds": total_seconds, "gpu_hours": total_seconds / 3600.0,
        "gpu_seconds_this_run_including_checks": total_seconds,
        "train_seconds": float(sum(_f(r.get("epoch_train_seconds")) or 0.0
                                   for r in curve_rows if r.get("source") == "RSSC_followup_v1_epochs_13_25")),
        "environment": F.C.environment(), "host_facts": F.C.host_facts(),
        "git_commit": F.C.git_commit(), "git_state": F.C.git_state(),
        "rng_restored": rng_restored, "rng_state_hashes_end": F.C.rng_state_hashes(F.C.capture_rng_state()),
        "first_resumed_step": first_step_check,
        "started_utc": started_utc, "finished_utc": F.utcnow(),
        "wall_seconds_this_process": float(time.perf_counter() - t_start),
        "wall_seconds_this_round_including_checks": float(time.perf_counter() - t_start) + elapsed_before_train,
        "test_role": "NOT_RUN / FORBIDDEN_THIS_ROUND", "test_scored": False,
        "test_used_for_selection": False,
        "source": {
            "round": "fixed-source-role-separation-pilot-v0",
            "checkpoint": F.track_rel(ckpt_path), "checkpoint_sha256": F.sha256_file(ckpt_path),
            "checkpoint_epoch": int(resume_epoch),
            "checkpoint_global_step": int(F.RESUME_GLOBAL_STEP),
            "best_checkpoint_copy": best_copy,
        },
        "budget": {"wall_seconds_cap": float(wall_seconds), "finalize_reserve": float(finalize_reserve),
                   "checks_seconds": float(checks.get("seconds") or 0.0),
                   "probe_seconds": float(timing.get("seconds") or 0.0)},
    }
    F.write_json(F.RUN_DIR / "run.json", run)
    F.log(f"done RSSC status={status} epochs={epochs_run}/{target_epoch} best_epoch={best_epoch} "
          f"best_valid={best_valid_ap} gpu_hours={run['gpu_hours']:.4f}")
    return run


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--wall-seconds", type=float, default=2100.0)
    parser.add_argument("--finalize-reserve", type=float, default=240.0)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--skip-checks", action="store_true",
                        help="reuse a previously written, all-passed resume_checks.json and timing.json")
    parser.add_argument("--min-projection-margin", type=float, default=0.0)
    args = parser.parse_args(argv)

    t_all = time.perf_counter()
    F.ANALYSIS.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        F.write_json(F.ANALYSIS / "run_status.json", {
            "status": "FAILED", "reason": "CUDA_UNAVAILABLE", "utc": F.utcnow()})
        return 2

    task = F.C.SD.load_task()
    plan, plan_audit = F.C.plan_for_seed(task, F.SEED)

    # resume source: initial v0 epoch-12 last, or a later checkpoint of this round
    ckpt_path = F.SOURCE_LAST
    resume_epoch = F.RESUME_EPOCH
    if (F.RUN_DIR / "checkpoint_last.pt").is_file() and (F.RUN_DIR / "resume_state.json").is_file():
        candidate = F.load_checkpoint(F.RUN_DIR / "checkpoint_last.pt")
        if int(candidate.get("epoch", 0)) > F.RESUME_EPOCH:
            ckpt_path = F.RUN_DIR / "checkpoint_last.pt"
            resume_epoch = int(candidate["epoch"])
            F.log(f"retry: resuming from this round's epoch-{resume_epoch} last checkpoint")
    ckpt = F.load_checkpoint(ckpt_path)
    if int(ckpt.get("epoch")) != resume_epoch:
        raise RuntimeError(f"checkpoint epoch {ckpt.get('epoch')} != resume epoch {resume_epoch}")

    F.log(f"checks: source {F.track_rel(ckpt_path)} epoch {resume_epoch} step {ckpt.get('global_step')}")
    if args.skip_checks:
        checks_path = F.ANALYSIS / "resume_checks.json"
        timing_path = F.ANALYSIS / "timing.json"
        if not checks_path.is_file() or not timing_path.is_file():
            raise RuntimeError("--skip-checks requested but resume_checks.json/timing.json are missing")
        checks = F.read_json(checks_path)
        timing = F.read_json(timing_path)
        checks["seconds"] = 0.0
        timing["seconds"] = 0.0
        if not checks.get("all_passed"):
            raise RuntimeError("--skip-checks requested but the stored checks did not all pass")
        F.log("checks: reused stored all-passed resume_checks.json and timing.json")
    else:
        t_checks = time.perf_counter()
        checks = run_checks(task, plan, plan_audit, ckpt, ckpt_path, device=device,
                            resume_epoch=resume_epoch)
        checks["seconds"] = float(time.perf_counter() - t_checks)
        F.write_json(F.ANALYSIS / "resume_checks.json", checks)
        if not checks["all_passed"]:
            F.write_json(F.ANALYSIS / "run_status.json", {
                "status": "RESUME_CHECK_FAILED", "failed": [c["check"] for c in checks["checks"] if not c["passed"]],
                "utc": F.utcnow()})
            F.log("RESUME_CHECK_FAILED: " + ", ".join(c["check"] for c in checks["checks"] if not c["passed"]))
            return 3
        F.log(f"checks: {checks['n_passed']}/{checks['n_checks']} passed")

        t_probe = time.perf_counter()
        timing = timing_probe(task, plan, ckpt, device, steps=int(args.steps))
        timing["seconds"] = float(time.perf_counter() - t_probe)
        F.write_json(F.ANALYSIS / "timing.json", timing)

    # budget projection
    remaining_epochs = F.TARGET_HORIZON - resume_epoch
    valid_seconds = float(checks["replay_seconds"])
    train_eval_estimate = max(45.0, 8.0 * valid_seconds)
    n_train_evals = len([e for e in F.TRAIN_EVAL_EPOCHS if e > resume_epoch]) + 1
    per_epoch = float(timing["step_ms_median"]) / 1e3 * F.STEPS_PER_EPOCH + valid_seconds + 20.0
    projected_remaining = remaining_epochs * per_epoch + n_train_evals * train_eval_estimate + 60.0
    elapsed = float(time.perf_counter() - t_all)
    remaining_wall = float(args.wall_seconds) - elapsed - float(args.finalize_reserve)
    projected_gpu_seconds = elapsed + projected_remaining
    go = (projected_remaining + float(args.min_projection_margin) <= remaining_wall
          and projected_gpu_seconds / 3600.0 <= 0.95)
    projection = {
        "remaining_epochs": int(remaining_epochs),
        "step_ms_median": float(timing["step_ms_median"]),
        "per_epoch_train_seconds": float(timing["step_ms_median"]) / 1e3 * F.STEPS_PER_EPOCH,
        "valid_eval_seconds_measured": valid_seconds,
        "train_eval_seconds_estimate": train_eval_estimate,
        "n_train_evals": int(n_train_evals),
        "projected_remaining_seconds": float(projected_remaining),
        "projected_remaining_minutes": float(projected_remaining / 60.0),
        "elapsed_seconds_so_far": float(elapsed),
        "wall_cap_seconds": float(args.wall_seconds),
        "finalize_reserve_seconds": float(args.finalize_reserve),
        "remaining_wall_seconds": float(remaining_wall),
        "projected_total_gpu_hours": float(projected_gpu_seconds / 3600.0),
        "gpu_hour_cap": 0.95,
        "decision": "GO" if go else "NO_GO",
        "rule": "start only if projected_remaining <= remaining wall and projected GPU-hours <= 0.95",
    }
    F.write_json(F.ANALYSIS / "budget_projection.json", projection)
    F.log(f"timing: step {timing['step_ms_median']:.1f} ms, epoch train "
          f"{projection['per_epoch_train_seconds']:.1f}s, projected remaining "
          f"{projection['projected_remaining_minutes']:.1f} min, wall left {remaining_wall/60:.1f} min -> "
          f"{projection['decision']}")
    if args.checks_only or not go:
        F.write_json(F.ANALYSIS / "run_status.json", {
            "status": "NOT_STARTED_BUDGET" if not go else "CHECKS_ONLY",
            "projection": projection, "utc": F.utcnow()})
        return 0 if args.checks_only else 3

    # formal training (fresh restore after the probe)
    ckpt = F.load_checkpoint(ckpt_path)
    run = train_followup(task, plan, plan_audit, ckpt, ckpt_path, device,
                         resume_epoch=resume_epoch, target_epoch=F.TARGET_HORIZON,
                         wall_seconds=float(args.wall_seconds),
                         finalize_reserve=float(args.finalize_reserve),
                         timing=timing, checks=checks)
    return 0 if run["status"] == "COMPLETED" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        F.ANALYSIS.mkdir(parents=True, exist_ok=True)
        F.write_json(F.ANALYSIS / "run.failed.json", {
            "schema": F.SCHEMA_RUN, "round_id": F.ROUND_ID, "status": "FAILED",
            "reason": type(exc).__name__, "detail": str(exc),
            "traceback": traceback.format_exc(),
            "environment": F.C.environment(), "utc": F.utcnow()})
        F.log(f"FAILED: {exc}")
        traceback.print_exc()
        raise SystemExit(1)
