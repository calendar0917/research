"""Formal from-scratch training / every-epoch three-split evaluation.

One arm, one seed (0), 40 epochs, 2,738 updates per epoch:

* frozen recipe (inherited ``sg_common.SCRATCH_RECIPE`` + ``scheduled_lr``),
* frozen seed-0 batch order (``sg_data.make_plan40``; hash-checked),
* masked BCE over observed entries only,
* dedicated dropout generator (seed ``230000 + 0``), saved and restored,
* full Train / Validation / Test evaluation after every completed epoch
  (``test_policy=always``; the official Test split is a DEVELOPMENT_TEST set and
  never selects a checkpoint),
* best-Validation checkpoint (ties keep the earlier epoch), epoch-40 checkpoint,
  and a resumable last checkpoint after every epoch.

Train gradients only ever see Train rows: targets are read through
``DevTask.training_targets``, which refuses official Test rows.
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import css_common as C
import css_data as D
import css_model as M
from pf_pipeline import PinnedStager

#: how many batches the collation thread may build ahead of the GPU
PREFETCH_DEPTH = 3

CURVE_FIELDS = [
    "epoch",
    "global_step",
    "learning_rate",
    "train_loss_online",
    "train_eval_bce",
    "train_ap",
    "valid_ap",
    "valid_bce",
    "test_ap",
    "test_bce",
    "n_tasks_scored",
    "train_seconds",
    "eval_train_seconds",
    "eval_valid_seconds",
    "eval_test_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "step_ms_mean",
    "step_ms_p50",
    "step_ms_p90",
    "collate_ms_p50",
    "collate_ms_mean",
    "pipeline_wait_ms_mean",
    "prefetch_depth",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "grad_norm_max",
    "grad_norm_mean",
    "gradient_clip_events",
    "gate_mean",
    "gate_std",
    "query_norm",
    "query_norm_task_spread",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
    "train_logits_sha256",
    "valid_logits_sha256",
    "test_logits_sha256",
]


# ---------------------------------------------------------------------------
# dropout / step
# ---------------------------------------------------------------------------


def make_dropout_generator(device: torch.device) -> torch.Generator:
    gen = torch.Generator(device="cuda" if device.type == "cuda" else "cpu")
    gen.manual_seed(int(C.DROPOUT_GENERATOR_SEED))
    return gen


def dropout_mask(
    n_atoms: int,
    n_tasks: int,
    dim: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    p: float,
) -> torch.Tensor:
    """One ``[n_atoms, n_tasks, dim]`` inverted-dropout mask per Train batch."""
    if n_atoms == 0:
        return torch.zeros(0, n_tasks, dim, device=device)
    u = torch.rand((n_atoms, n_tasks, dim), generator=generator, device=device)
    keep = (u >= float(p)).to(torch.float32)
    return keep / (1.0 - float(p))


MODULE_PREFIXES = {
    "atom": ("atom.",),
    "bond": ("bond.",),
    "s1_event": ("s1_event.",),
    "s2_event": ("s2_event.",),
    "keys": ("key1.", "key2."),
    "values": ("value1.", "value2."),
    "root_static": ("root_static.",),
    "root_stat": ("root_stat.",),
    "layer_norm": ("ln.",),
    "graph_mlp": ("graph.",),
    "task_output": ("out.",),
    "query": ("q0", "qt", "Wc."),
}


def module_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    out = {name: 0.0 for name in MODULE_PREFIXES}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        for label, prefixes in MODULE_PREFIXES.items():
            if any(name == pre.rstrip(".") or name.startswith(pre) for pre in prefixes):
                out[label] += float(p.grad.detach().pow(2).sum().item())
                break
    return {k: math.sqrt(v) for k, v in out.items()}


def train_step(
    model: M.CSSModel,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    y: torch.Tensor,
    *,
    global_step: int,
    clip: float,
    generator: torch.Generator,
    diagnostics: M.ForwardDiagnostics | None = None,
) -> dict[str, Any]:
    lr = float(C.scheduled_lr(int(global_step), phase=3, kind="all"))
    for group in optimizer.param_groups:
        group["lr"] = lr
    n_atoms = int(batch["n_atoms"])
    mask = dropout_mask(
        n_atoms,
        model.n_tasks,
        64,
        generator=generator,
        device=batch["atom_feat"].device,
        p=model.dropout_p,
    )
    logits = model(batch, dropout_mask=mask, diagnostics=diagnostics)
    if tuple(logits.shape) != (int(y.shape[0]), C.N_TASKS):
        raise RuntimeError(f"logits {tuple(logits.shape)} != {(int(y.shape[0]), C.N_TASKS)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"non-finite logits at step {global_step}")
    loss = C.MET.masked_bce_loss(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {global_step}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = module_grad_norms(model)
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip)))
    if not math.isfinite(gnorm):
        raise RuntimeError(f"non-finite gradient norm at step {global_step}")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "grad_norm": gnorm,
        "lr": lr,
        "clipped": bool(gnorm > float(clip)),
        "module_grad_norms": grads,
        "n_observed": int(torch.isfinite(y).sum()),
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def collate_one(dev: D.DevTask, rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any], np.ndarray, float]:
    """CPU side of one step: static structure objects + Train targets.

    Pure function of ``rows`` (no RNG, no model state), so building it one batch
    ahead in a worker thread cannot change what the model sees.
    """
    t0 = time.perf_counter()
    batch = D.collate_batch(dev, rows)
    y = np.asarray(dev.training_targets(rows), dtype=np.float32)
    return rows, batch, y, (time.perf_counter() - t0)


def _prefetched(
    dev: D.DevTask, batches: Sequence[np.ndarray], depth: int
) -> Any:
    """Yield ``collate_one`` results in the exact plan order, ``depth`` ahead.

    ``depth = 0`` is the fully synchronous path (collate inside the step).
    Otherwise one worker thread builds the next batches while the GPU works.
    The tensors, their order, their dtypes and their values are identical - only
    the overlap changes.  Yields ``(rows, batch, targets, collate_s, wait_s)``.
    """
    if depth <= 0:
        for rows in batches:
            r, b, y, t = collate_one(dev, rows)
            yield r, b, y, t, 0.0
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="css-collate") as pool:
        it = iter(batches)
        pending: deque = deque()
        for rows in it:
            pending.append(pool.submit(collate_one, dev, rows))
            if len(pending) >= int(depth):
                break
        while pending:
            future = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(collate_one, dev, nxt))
            t0 = time.perf_counter()
            rows_, batch_, y_, t_collate = future.result()
            yield rows_, batch_, y_, t_collate, time.perf_counter() - t0


def _array_sha(array: np.ndarray, dtype=np.float32) -> str:
    return C.sha256_bytes(np.ascontiguousarray(array, dtype=dtype).tobytes())


def evaluate_split(
    model: M.CSSModel,
    dev: D.DevTask,
    split: str,
    device: torch.device,
    *,
    batch_size: int = 128,
    diagnostics: M.ForwardDiagnostics | None = None,
    row_limit: int | None = None,
) -> dict[str, Any]:
    """Full-split official AP + masked BCE + per-task AP.

    ``row_limit`` exists only for local smoke runs (``train`` with
    ``eval_limit``); every formal / preflight call leaves it at ``None`` and
    therefore evaluates the complete split.
    """
    t0 = time.perf_counter()
    rows = dev.rows[split]
    if row_limit is not None:
        rows = rows[: int(row_limit)]
    logits = M.predict_logits(
        model,
        lambda r: D.collate_batch(dev, r),
        rows,
        device,
        batch_size=batch_size,
        diagnostics=diagnostics,
    )
    y = np.asarray(dev.labels_dev[rows], dtype=np.float32)
    if logits.shape[0] != rows.size:
        raise RuntimeError(f"{split} prediction rows {logits.shape[0]} != {rows.size}")
    probs = M.sigmoid(logits)
    report = C.MET.validation_report(y, probs)
    bce = float(C.MET.masked_bce_numpy(logits, y))
    return {
        "split": split,
        "rows": int(rows.size),
        "evaluated_rows": int(logits.shape[0]),
        "ap": float(report["ap"]),
        "ap_source": report["ap_source"],
        "bce": bce,
        "per_task_ap": report["per_task_ap"],
        "observed": report["observed"],
        "positive": report["positive"],
        "negative": report["negative"],
        "n_tasks_scored": int(report["n_tasks_scored"]),
        "n_tasks_with_observed_labels": int(report["n_tasks_with_observed_labels"]),
        "observed_entries": int(np.isfinite(y).sum()),
        "seconds": float(time.perf_counter() - t0),
        "logits_sha256": _array_sha(logits),
        "logits": logits,
        "labels": y,
        "row_ids": np.asarray(rows, dtype=np.int64),
    }


def save_predictions(path: Path, report: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        logits=np.asarray(report["logits"], dtype=np.float32),
        labels=np.asarray(report["labels"], dtype=np.float32),
        row_ids=np.asarray(report["row_ids"], dtype=np.int64),
        observed=np.isfinite(np.asarray(report["labels"], dtype=np.float32)),
        split=np.asarray(report["split"]),
    )
    os.replace(tmp, path)
    return C.sha256_file(path)


def _strip(report: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k not in ("logits", "labels", "row_ids", "per_task_ap")}


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def _atomic_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def build_optimizer(model: torch.nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(C.RECIPE["warmup_start_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    audit = {
        "optimizer": "AdamW",
        "n_param_groups": len(opt.param_groups),
        "n_parameters": int(sum(p.numel() for p in params)),
        "betas": list(C.RECIPE["adam_betas"]),
        "eps": float(C.RECIPE["adam_eps"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "scheduler": "1-epoch linear warmup 3e-5->3e-4 then per-step cosine 3e-4->1e-5 over 109,520 updates",
        "single_group_same_lr": True,
    }
    if len(opt.param_groups) != 1:
        raise RuntimeError("this round uses exactly one parameter group")
    return opt, audit


def checkpoint_payload(
    *,
    arm: str,
    epoch: int,
    global_step: int,
    model: M.CSSModel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    model_audit: Mapping[str, Any],
    valid_ap: float | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "arm": str(arm),
        "arm_name": C.ARMS[arm]["name"],
        "seed": int(C.SEED),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": str(kind),
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]),
        "state_dict": C.state_dict_cpu(model.state_dict()),
        "optimizer_state_dict": C.state_dict_cpu(optimizer.state_dict()),
        "dropout_generator_state": generator.get_state().clone(),
        "dropout_generator_seed": int(C.DROPOUT_GENERATOR_SEED),
        "batch_plan": {
            "global_plan_sha256": C.PLAN["global_plan_sha256"],
            "consumed40_prefix_sha256": C.PLAN["consumed40_prefix_sha256"],
            "order_seed": int(C.PLAN["order_seed"]),
            "batch_size": int(C.BATCH_SIZE),
            "steps_per_epoch": int(C.STEPS_PER_EPOCH),
            "consumed_epochs": int(epoch),
        },
        "model_audit": dict(model_audit),
        "feature_spec_hash": C.FEATURE_SPEC_HASH,
        "norm_fingerprint": C.NORM_FINGERPRINT,
        "split_partition_sha256": C.SPLIT_PARTITION_SHA256,
        "label_artifact_sha256": C.LABEL_ARTIFACT_SHA256,
        "cache_sha256": C.CACHE_SHA256,
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "recipe": C.recipe_for_arm(arm),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "test_policy": C.TEST_ROLE,
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def restore(
    payload: Mapping[str, Any],
    model: M.CSSModel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> dict[str, Any]:
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if payload.get("arm") != model.arm:
        raise RuntimeError(f"checkpoint arm {payload.get('arm')} != {model.arm}")
    if int(payload.get("parameters", -1)) != model.n_parameters():
        raise RuntimeError("checkpoint parameter count mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    generator.set_state(payload["dropout_generator_state"])
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in optimizer.state_dict()["state"].values()
        }
    )
    if int(payload["global_step"]) > 0 and steps != [float(payload["global_step"])]:
        raise RuntimeError(f"restored AdamW steps {steps} != payload {payload['global_step']}")
    return {
        "resumed_from_epoch": int(payload["epoch"]),
        "resumed_global_step": int(payload["global_step"]),
        "optimizer_state_restored": True,
        "dropout_generator_restored": True,
    }


# ---------------------------------------------------------------------------
# the trajectory
# ---------------------------------------------------------------------------


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        ""
                        if row.get(k) is None
                        else (row[k].item() if isinstance(row[k], (np.floating, np.integer)) else row[k])
                    )
                    for k in fields
                }
            )
    return C.sha256_file(path)


def train(
    dev: D.DevTask,
    *,
    arm: str,
    out_dir: Path,
    device: torch.device,
    epochs: int = C.TOTAL_EPOCHS,
    task_chunk: int = C.TASK_CHUNK_DEFAULT,
    resume: bool = True,
    train_timeout_seconds: float | None = None,
    log_every: int = 200,
    steps_per_epoch: int | None = None,
    eval_limit: int | None = None,
    prefetch_depth: int = PREFETCH_DEPTH,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    plan = D.make_plan40(dev)
    if epochs > C.TOTAL_EPOCHS:
        raise ValueError("the frozen recipe has exactly 40 epochs")

    C.seed_everything(int(C.SEED))
    model, model_audit = M.build_model(arm, dev.task.norm, task_chunk=task_chunk)
    model.to(device)
    optimizer, optimizer_audit = build_optimizer(model)
    generator = make_dropout_generator(device)
    transfer = PinnedStager(device)

    curve: list[dict[str, Any]] = []
    best = {"epoch": 0, "valid_ap": -math.inf}
    start_epoch = 1
    global_step = 0
    resumed = None
    last_path = out_dir / "checkpoint_last.pt"
    if resume and last_path.is_file():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        resumed = restore(payload, model, optimizer, generator)
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        curve_path = out_dir / "train_curve.csv"
        if curve_path.is_file():
            with open(curve_path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    parsed = {}
                    for k, v in row.items():
                        if v == "":
                            parsed[k] = None
                        else:
                            try:
                                parsed[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                            except ValueError:
                                parsed[k] = v
                    curve.append(parsed)
        best = {
            "epoch": int(payload.get("best_epoch", payload["epoch"])),
            "valid_ap": float(payload.get("valid_ap") or -math.inf),
        }
        C.log(f"[{arm}] resumed from epoch {payload['epoch']} step {global_step}")

    def _finalize(epoch: int, row: Mapping[str, Any]) -> None:
        _write_csv(out_dir / "train_curve.csv", CURVE_FIELDS, curve)

    for epoch in range(start_epoch, epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        batches = plan[epoch - 1]["batches"]
        if steps_per_epoch is not None:
            batches = batches[: int(steps_per_epoch)]
        step_ms: list[float] = []
        collate_ms: list[float] = []
        wait_ms: list[float] = []
        losses: list[float] = []
        grad_norms: list[float] = []
        clip_events = 0
        gate_diag = M.ForwardDiagnostics()
        t_train = time.perf_counter()
        for i, (rows, batch, y_np, t_collate, t_wait) in enumerate(
            _prefetched(dev, batches, depth=prefetch_depth)
        ):
            t_step = time.perf_counter()
            collate_ms.append(t_collate * 1000.0)
            wait_ms.append(t_wait * 1000.0)
            batch = transfer.batch(batch)
            y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
            diag = gate_diag if i % 32 == 0 else None
            out = train_step(
                model,
                optimizer,
                batch,
                y,
                global_step=global_step,
                clip=float(C.RECIPE["gradient_clip"]),
                generator=generator,
                diagnostics=diag,
            )
            global_step += 1
            step_ms.append((time.perf_counter() - t_step) * 1000.0)
            losses.append(out["loss"])
            grad_norms.append(out["grad_norm"])
            clip_events += int(out["clipped"])
            if log_every and (i + 1) % int(log_every) == 0:
                C.log(
                    f"[{arm}] epoch {epoch} step {i + 1}/{len(batches)} "
                    f"loss={np.mean(losses[-log_every:]):.5f} "
                    f"lr={out['lr']:.3e} step_ms={np.mean(step_ms[-log_every:]):.1f}"
                )
        train_seconds = time.perf_counter() - t_train

        model.eval()
        reports = {}
        for split in ("train", "valid", "test"):
            rep = evaluate_split(model, dev, split, device, row_limit=eval_limit)
            reports[split] = rep
            C.log(
                f"[{arm}] epoch {epoch} {split}: ap={rep['ap']:.6f} bce={rep['bce']:.6f} "
                f"rows={rep['evaluated_rows']} ({rep['seconds']:.1f}s)"
            )

        valid_ap = float(reports["valid"]["ap"])
        is_best = valid_ap > best["valid_ap"]
        if is_best:
            best = {"epoch": int(epoch), "valid_ap": valid_ap}
            payload = checkpoint_payload(
                arm=arm,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                generator=generator,
                model_audit=model_audit,
                valid_ap=valid_ap,
                kind="best_valid",
                extra={"best_epoch": int(epoch), "best_valid_ap": valid_ap},
            )
            _atomic_save(payload, out_dir / "checkpoint_best.pt")
            for split in ("train", "valid", "test"):
                save_predictions(out_dir / f"predictions_best_{split}.npz", reports[split])
            C.atomic_write_json(
                out_dir / "best_eval.json",
                {
                    "epoch": int(epoch),
                    "valid_ap": valid_ap,
                    "train": _strip(reports["train"]),
                    "valid": _strip(reports["valid"]),
                    "test": _strip(reports["test"]),
                    "test_role": C.TEST_ROLE,
                },
            )

        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "learning_rate": float(C.scheduled_lr(global_step, phase=3, kind="all")),
            "train_loss_online": float(np.mean(losses)),
            "train_eval_bce": float(reports["train"]["bce"]),
            "train_ap": float(reports["train"]["ap"]),
            "valid_ap": valid_ap,
            "valid_bce": float(reports["valid"]["bce"]),
            "test_ap": float(reports["test"]["ap"]),
            "test_bce": float(reports["test"]["bce"]),
            "n_tasks_scored": int(reports["valid"]["n_tasks_scored"]),
            "train_seconds": float(train_seconds),
            "eval_train_seconds": float(reports["train"]["seconds"]),
            "eval_valid_seconds": float(reports["valid"]["seconds"]),
            "eval_test_seconds": float(reports["test"]["seconds"]),
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": float(time.perf_counter() - t_start),
            "step_ms_mean": float(np.mean(step_ms)),
            "step_ms_p50": float(np.percentile(step_ms, 50)),
            "step_ms_p90": float(np.percentile(step_ms, 90)),
            "collate_ms_p50": float(np.percentile(collate_ms, 50)) if collate_ms else 0.0,
            "collate_ms_mean": float(np.mean(collate_ms)) if collate_ms else 0.0,
            "pipeline_wait_ms_mean": float(np.mean(wait_ms)) if wait_ms else 0.0,
            "prefetch_depth": int(prefetch_depth),
            "gpu_peak_allocated_bytes": C.gpu_peak(device)["peak_allocated_bytes"],
            "gpu_peak_reserved_bytes": C.gpu_peak(device)["peak_reserved_bytes"],
            "grad_norm_max": float(np.max(grad_norms)),
            "grad_norm_mean": float(np.mean(grad_norms)),
            "gradient_clip_events": int(clip_events),
            "gate_mean": gate_diag.summary()["gate_mean"] if gate_diag.gate_terms else None,
            "gate_std": gate_diag.summary()["gate_std"] if gate_diag.gate_terms else None,
            "query_norm": (
                float(model.q0.detach().norm()) if not hasattr(model, "qt") else float(model.qt.detach().norm())
            ),
            "query_norm_task_spread": gate_diag.summary()["query_norm_task_spread"]
            if gate_diag.gate_terms
            else None,
            "best_epoch_so_far": int(best["epoch"]),
            "best_valid_ap_so_far": float(best["valid_ap"]),
            "is_best": bool(is_best),
            "train_logits_sha256": reports["train"]["logits_sha256"],
            "valid_logits_sha256": reports["valid"]["logits_sha256"],
            "test_logits_sha256": reports["test"]["logits_sha256"],
        }
        curve.append(row)
        _finalize(epoch, row)
        _write_per_task(out_dir / "per_task_ap.csv", reports)

        last_payload = checkpoint_payload(
            arm=arm,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            generator=generator,
            model_audit=model_audit,
            valid_ap=valid_ap,
            kind="last",
            extra={
                "best_epoch": int(best["epoch"]),
                "best_valid_ap": float(best["valid_ap"]),
                "curve_sha256": _write_csv(out_dir / "train_curve.csv", CURVE_FIELDS, curve),
            },
        )
        last_payload["per_task_ap"] = {
            split: {
                "per_task_ap": reports[split]["per_task_ap"],
                "positive": reports[split]["positive"],
                "observed": reports[split]["observed"],
            }
            for split in ("train", "valid", "test")
        }
        last_payload["epoch_metrics"] = {
            split: _strip(reports[split]) for split in ("train", "valid", "test")
        }
        _atomic_save(last_payload, last_path)

        if epoch == epochs:
            final_payload = checkpoint_payload(
                arm=arm,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                generator=generator,
                model_audit=model_audit,
                valid_ap=valid_ap,
                kind="epoch40",
                extra={"best_epoch": int(best["epoch"]), "best_valid_ap": float(best["valid_ap"])},
            )
            _atomic_save(final_payload, out_dir / "checkpoint_epoch40.pt")
            for split in ("train", "valid", "test"):
                save_predictions(out_dir / f"predictions_epoch40_{split}.npz", reports[split])
            C.atomic_write_json(
                out_dir / "epoch40_eval.json",
                {
                    "epoch": int(epoch),
                    "train": _strip(reports["train"]),
                    "valid": _strip(reports["valid"]),
                    "test": _strip(reports["test"]),
                    "test_role": C.TEST_ROLE,
                },
            )
        if train_timeout_seconds is not None:
            if time.perf_counter() - t_start > float(train_timeout_seconds):
                C.log(f"[{arm}] stopping after epoch {epoch}: wall guard reached")
                break

    summary = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "arm_name": C.ARMS[arm]["name"],
        "seed": int(C.SEED),
        "epochs_run": int(len(curve)),
        "final_epoch": int(curve[-1]["epoch"]) if curve else 0,
        "all_epochs_completed": bool(len(curve) == epochs),
        "best_epoch": int(best["epoch"]),
        "best_valid_ap": float(best["valid_ap"]),
        "parameters": int(model_audit["parameters"]),
        "parameter_breakdown": model_audit["parameter_breakdown"],
        "model_audit": model_audit,
        "optimizer_audit": optimizer_audit,
        "plan": dict(C.PLAN),
        "epoch0_eval": "shared initial-function evaluation recorded in analysis/preflight.json",
        "resumed": resumed,
        "wall_seconds": float(time.perf_counter() - t_start),
        "steps_completed": int(global_step),
        "device": str(device),
        "host": C.host_facts(),
        "recipe": C.recipe_for_arm(arm),
        "identities": {
            "feature_spec_hash": C.FEATURE_SPEC_HASH,
            "norm_fingerprint": C.NORM_FINGERPRINT,
            "split_partition_sha256": C.SPLIT_PARTITION_SHA256,
            "cache_sha256": C.CACHE_SHA256,
            "label_artifact_sha256": C.LABEL_ARTIFACT_SHA256,
        },
        "label_view": dev.label_view_audit(),
        "test_role": C.TEST_ROLE,
        "curve": curve,
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "environment": C.environment(),
        "created_utc": C.utcnow(),
    }
    C.atomic_write_json(out_dir / "run.json", summary)
    return summary


def _write_per_task(path: Path, reports: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "split",
                "task",
                "ap",
                "observed",
                "positive",
                "negative",
            ]
        )
        for split in ("train", "valid", "test"):
            rep = reports[split]
            for t in range(C.N_TASKS):
                writer.writerow(
                    [
                        split,
                        t,
                        "" if rep["per_task_ap"][t] is None else rep["per_task_ap"][t],
                        rep["observed"][t],
                        rep["positive"][t],
                        rep["negative"][t],
                    ]
                )
    return C.sha256_file(path)


__all__ = [
    "CURVE_FIELDS",
    "build_optimizer",
    "checkpoint_payload",
    "dropout_mask",
    "evaluate_split",
    "make_dropout_generator",
    "module_grad_norms",
    "restore",
    "save_predictions",
    "train",
    "train_step",
]
