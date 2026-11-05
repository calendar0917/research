"""Phase B: two tiny residual heads on the frozen cached features.

``new_logits = frozen_logits + head(standardized_rho_in)``

Both arms share the recipe, the initialisation seed, the batch size, the epoch
count and the **identical** training-row order; only the head architecture
differs.  The frozen LCRC-C1 checkpoint is re-loaded and fingerprinted, is never
placed in an optimizer, and is verified unchanged after both arms finish.

The epoch-8 value is the primary comparison.  Best-Validation AP is recorded as
an auxiliary quantity only.
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

import fr_common as C  # noqa: E402
import fr_metrics as FM  # noqa: E402
import fr_model as MODEL  # noqa: E402
import lr_data as DATA  # noqa: E402
import pc_model as BM  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "train_bce_full",
    "train_ap",
    "valid_ap",
    "n_valid_tasks_scored",
    "delta_vs_ap0",
    "train_minus_valid_ap",
    "loss_step_mean",
    "loss_entries",
    "grad_norm_max",
    "train_seconds",
    "eval_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "head_fingerprint_sha256",
    "logits_bitwise_equal_frozen",
    "notes",
]

PER_TASK_FIELDS = ["epoch", "task", "ap", "observed", "positive", "negative"]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})
    return C.sha256_file(path)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


# ---------------------------------------------------------------------------
# cache loading
# ---------------------------------------------------------------------------


def load_cache(device: torch.device) -> dict[str, Any]:
    """Verify and load the Phase-A cache (hashes checked before use)."""
    manifest_path = C.FEATURE_CACHE / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"feature cache manifest missing: {manifest_path}")
    manifest = C.read_json(manifest_path)
    audit_path = C.ANALYSIS / "feature_extraction_audit.json"
    if not audit_path.exists():
        raise RuntimeError("phase-A audit missing; extraction must run first")
    audit = C.read_json(audit_path)
    if not audit.get("all_passed") or not manifest.get("all_passed"):
        raise RuntimeError("phase A did not pass its gates; refusing to train heads")
    expected_rho_dim = int(manifest["rho_in_dim"])
    if expected_rho_dim != int(C.RHO_IN_DIM_EXPECTED):
        raise RuntimeError(f"cache rho_in dim {expected_rho_dim} != frozen {C.RHO_IN_DIM_EXPECTED}")

    out: dict[str, Any] = {
        "manifest": manifest,
        "manifest_sha256": C.sha256_file(manifest_path),
        "audit_sha256": C.sha256_file(audit_path),
        "files": {},
        "arrays": {},
    }
    for key, record in manifest["arrays"].items():
        path = C.REPO / record["path"]
        if not path.exists():
            raise RuntimeError(f"cached array missing: {path}")
        file_sha = C.sha256_file(path)
        arr = np.load(path)
        content_sha = C.sha256_array(arr)
        entry = {
            "path": record["path"],
            "file_sha256": file_sha,
            "expected_file_sha256": record["file_sha256"],
            "content_sha256": content_sha,
            "expected_content_sha256": record["content_sha256"],
            "file_hash_ok": bool(file_sha == record["file_sha256"]),
            "content_hash_ok": bool(content_sha == record["content_sha256"]),
            "shape": record["shape"],
            "dtype": record["dtype"],
            "bytes": record["bytes"],
        }
        entry["hash_ok"] = bool(entry["file_hash_ok"] and entry["content_hash_ok"])
        out["files"][key] = entry
        if not entry["hash_ok"]:
            raise RuntimeError(f"cached array hash mismatch for {key}: {entry}")
        if list(arr.shape) != list(record["shape"]) or str(arr.dtype) != str(record["dtype"]):
            raise RuntimeError(f"cached array {key} shape/dtype drifted from the manifest")
        out["arrays"][key] = arr

    a = out["arrays"]
    if list(a["train_rho_in"].shape) != [C.N_TRAIN, expected_rho_dim]:
        raise RuntimeError(f"train rho_in shape {a['train_rho_in'].shape} unexpected")
    if list(a["valid_rho_in"].shape) != [C.N_VALID, expected_rho_dim]:
        raise RuntimeError(f"valid rho_in shape {a['valid_rho_in'].shape} unexpected")
    for name in ("train_logits", "train_labels"):
        if list(a[name].shape) != [C.N_TRAIN, C.N_TASKS]:
            raise RuntimeError(f"{name} shape {a[name].shape} unexpected")
    for name in ("valid_logits", "valid_labels"):
        if list(a[name].shape) != [C.N_VALID, C.N_TASKS]:
            raise RuntimeError(f"{name} shape {a[name].shape} unexpected")
    for split in ("train", "valid"):
        mask = np.asarray(a[f"{split}_mask"], dtype=bool)
        if mask.shape != a[f"{split}_labels"].shape:
            raise RuntimeError(f"{split} mask shape mismatch")
        if not np.array_equal(mask, np.isfinite(a[f"{split}_labels"])):
            raise RuntimeError(f"{split} mask is not exactly isfinite(labels)")
        values = a[f"{split}_labels"][mask]
        if not np.all((values == 0.0) | (values == 1.0)):
            raise RuntimeError(f"{split} observed labels are not binary 0/1")

    out["gpu"] = {
        key: torch.from_numpy(np.ascontiguousarray(a[key])).to(device)
        for key in (
            "train_rho_in",
            "valid_rho_in",
            "train_logits",
            "valid_logits",
            "train_labels",
            "valid_labels",
        )
    }
    out["gpu"]["train_mask"] = torch.from_numpy(np.ascontiguousarray(a["train_mask"])).to(device)
    out["gpu"]["valid_mask"] = torch.from_numpy(np.ascontiguousarray(a["valid_mask"])).to(device)
    return out


def standardize_cache(gpu: Mapping[str, torch.Tensor], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen Train-only mean/scale and verify the Phase-A fingerprint."""
    mean = torch.from_numpy(np.load(C.REPO / manifest["arrays"]["standardizer_mean"]["path"]))
    scale = torch.from_numpy(np.load(C.REPO / manifest["arrays"]["standardizer_scale"]["path"]))
    expected = manifest["cache_content_fingerprints"]
    ref = manifest["standardizer"]
    audit = {
        "mean_sha256": C.sha256_array(mean.numpy()),
        "scale_sha256": C.sha256_array(scale.numpy()),
        "mean_sha256_matches_manifest": C.sha256_array(mean.numpy()) == expected["standardizer_mean_sha256"],
        "scale_sha256_matches_manifest": C.sha256_array(scale.numpy()) == expected["standardizer_scale_sha256"],
        "n_zero_variance_columns": int(ref["n_zero_variance_columns"]),
        "zero_variance_columns": ref["zero_variance_columns"],
        "zero_variance_scale_is_one": bool(
            all(float(scale[int(item["column"])]) == 1.0 for item in ref["zero_variance_columns"])
        ),
        "fit_split": ref["fit_split"],
        "ddof": ref["ddof"],
    }
    FM.standardize_(gpu["train_rho_in"], mean, scale)
    FM.standardize_(gpu["valid_rho_in"], mean, scale)
    audit["train_standardized_sha256"] = C.sha256_array(gpu["train_rho_in"].cpu().numpy())
    audit["valid_standardized_sha256"] = C.sha256_array(gpu["valid_rho_in"].cpu().numpy())
    audit["train_standardized_sha256_matches_phase_a"] = bool(
        audit["train_standardized_sha256"] == expected["train_rho_in_standardized_sha256"]
    )
    audit["valid_standardized_sha256_matches_phase_a"] = bool(
        audit["valid_standardized_sha256"] == expected["valid_rho_in_standardized_sha256"]
    )
    audit["train_stats"] = FM.column_stats(gpu["train_rho_in"])
    audit["valid_stats"] = FM.column_stats(gpu["valid_rho_in"])
    return audit


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _predict_logits(
    head: torch.nn.Module,
    x: torch.Tensor,
    frozen: torch.Tensor,
    *,
    chunk: int = 65_536,
) -> torch.Tensor:
    out = torch.empty_like(frozen)
    head.eval()
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(chunk)):
            stop = min(start + int(chunk), int(x.shape[0]))
            out[start:stop].copy_(frozen[start:stop] + head(x[start:stop]))
    return out


def evaluate(
    head: torch.nn.Module,
    cache: Mapping[str, Any],
    *,
    epoch: int,
    compute_train_official: bool,
) -> dict[str, Any]:
    """Train + Validation metrics for one evaluation epoch."""
    gpu = cache["gpu"]
    t0 = time.perf_counter()
    train_logits = _predict_logits(head, gpu["train_rho_in"], gpu["train_logits"])
    valid_logits = _predict_logits(head, gpu["valid_rho_in"], gpu["valid_logits"])
    if not bool(torch.isfinite(train_logits).all()) or not bool(torch.isfinite(valid_logits).all()):
        raise RuntimeError(f"non-finite logits at epoch {epoch}")
    bitwise_frozen = bool(
        torch.equal(train_logits, gpu["train_logits"]) and torch.equal(valid_logits, gpu["valid_logits"])
    )

    train_logits_np = train_logits.cpu().numpy()
    valid_logits_np = valid_logits.cpu().numpy()
    del train_logits, valid_logits
    train_labels = cache["arrays"]["train_labels"]
    valid_labels = cache["arrays"]["valid_labels"]
    train_probs = BM.sigmoid(train_logits_np)
    valid_probs = BM.sigmoid(valid_logits_np)

    train_table = FM.task_table(train_labels, train_probs)
    valid_table = FM.task_table(valid_labels, valid_probs)
    official_valid = FM.official_ap(valid_labels, valid_probs)
    train_macro = FM.macro_ap_from_tasks(train_table["per_task"])
    train_official = (
        float(FM.official_ap(train_labels, train_probs)["value"]) if compute_train_official else None
    )
    train_bce = FM.masked_bce_numpy(train_logits_np, train_labels)
    valid_bce = FM.masked_bce_numpy(valid_logits_np, valid_labels)
    seconds = float(time.perf_counter() - t0)
    return {
        "epoch": int(epoch),
        "train_bce_full": float(train_bce),
        "valid_bce_full": float(valid_bce),
        "train_ap": float(train_macro),
        "train_official_ap": train_official,
        "valid_ap": float(official_valid["value"]),
        "valid_official_source": official_valid["source"],
        "valid_macro_independent": float(FM.macro_ap_from_tasks(valid_table["per_task"])),
        "n_valid_tasks_scored": int(valid_table["n_tasks_scored"]),
        "n_train_tasks_scored": int(train_table["n_tasks_scored"]),
        "valid_per_task": [None if v is None else float(v) for v in valid_table["per_task"]],
        "train_per_task": [None if v is None else float(v) for v in train_table["per_task"]],
        "valid_observed": [int(x) for x in valid_table["observed"]],
        "valid_positive": [int(x) for x in valid_table["positive"]],
        "valid_negative": [int(x) for x in valid_table["negative"]],
        "logits_bitwise_equal_frozen": bitwise_frozen,
        "train_minus_valid_ap": float(train_macro - float(official_valid["value"])),
        "eval_seconds": seconds,
    }


# ---------------------------------------------------------------------------
# training one arm
# ---------------------------------------------------------------------------


def train_arm(
    arm: str,
    *,
    cache: Mapping[str, Any],
    plan: Mapping[str, Any],
    out_dir: Path,
    device: torch.device,
    base_model_audit: Mapping[str, Any],
    soft_seconds: float,
    assigned_pool: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = cache["gpu"]
    d = int(gpu["train_rho_in"].shape[1])
    head = FM.build_head(arm, d).to(device)
    init_fp = FM.head_parameter_fingerprint(head)
    sample = min(4096, int(gpu["train_rho_in"].shape[0]))
    init_audit = FM.zero_init_audit(head, gpu["train_rho_in"][:sample], gpu["train_logits"][:sample])
    if not init_audit["residual_is_exactly_zero"] or not init_audit["logits_bitwise_equal_frozen"]:
        raise RuntimeError(f"arm {arm}: zero-init identity gate failed: {init_audit}")

    params = list(head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.HEAD_RECIPE["learning_rate"]),
        betas=tuple(float(x) for x in C.HEAD_RECIPE["adam_betas"]),
        eps=float(C.HEAD_RECIPE["adam_eps"]),
        weight_decay=float(C.HEAD_RECIPE["weight_decay"]),
    )
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not optimizer_param_ids.issubset({id(p) for p in params}):
        raise RuntimeError(f"arm {arm}: optimizer contains a non-head parameter")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t_arm = time.perf_counter()
    deadline = t_arm + float(soft_seconds)
    curve: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    metric_history: dict[int, dict[str, Any]] = {}
    state_history: dict[int, dict[str, torch.Tensor]] = {}
    status = "COMPLETED"
    stop_reason = f"COMPLETED_{C.EPOCHS}_EPOCHS"
    loss_all: list[float] = []
    grad_norm_max = 0.0
    non_finite_events = 0
    evaluated: list[int] = []
    train_seconds_total = 0.0
    step_loss_mean: float | None = None

    def record(epoch: int, metrics: Mapping[str, Any], train_seconds: float, notes: str) -> None:
        for t in range(C.N_TASKS):
            ap = metrics["valid_per_task"][t]
            per_task_rows.append(
                {
                    "epoch": int(epoch),
                    "task": int(t),
                    "ap": None if ap is None else float(ap),
                    "observed": int(metrics["valid_observed"][t]),
                    "positive": int(metrics["valid_positive"][t]),
                    "negative": int(metrics["valid_negative"][t]),
                }
            )
        peak = C.gpu_peak(device)
        curve.append(
            {
                "epoch": int(epoch),
                "evaluated": True,
                "train_bce_full": float(metrics["train_bce_full"]),
                "train_ap": float(metrics["train_ap"]),
                "valid_ap": float(metrics["valid_ap"]),
                "n_valid_tasks_scored": int(metrics["n_valid_tasks_scored"]),
                "delta_vs_ap0": float(metrics["valid_ap"]) - float(C.AP0),
                "train_minus_valid_ap": float(metrics["train_minus_valid_ap"]),
                "loss_step_mean": None if step_loss_mean is None else float(step_loss_mean),
                "loss_entries": None,
                "grad_norm_max": float(grad_norm_max),
                "train_seconds": float(train_seconds),
                "eval_seconds": float(metrics["eval_seconds"]),
                "epoch_seconds": float(train_seconds) + float(metrics["eval_seconds"]),
                "cumulative_seconds": float(time.perf_counter() - t_arm),
                "gpu_peak_allocated_bytes": int(peak["peak_allocated_bytes"]),
                "gpu_peak_reserved_bytes": int(peak["peak_reserved_bytes"]),
                "head_fingerprint_sha256": FM.head_parameter_fingerprint(head)["combined_sha256"],
                "logits_bitwise_equal_frozen": bool(metrics["logits_bitwise_equal_frozen"]),
                "notes": notes,
            }
        )
        metric_history[int(epoch)] = dict(metrics)
        state_history[int(epoch)] = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        evaluated.append(int(epoch))

    # ---- epoch 0: exact identity, no optimiser step ----
    head.eval()
    metrics0 = evaluate(head, cache, epoch=0, compute_train_official=True)
    record(0, metrics0, 0.0, "epoch 0 = zero-initialised residual, before any update")
    if not metrics0["logits_bitwise_equal_frozen"]:
        raise RuntimeError("arm epoch 0 does not reproduce the frozen logits bitwise")
    C.log(
        f"[{arm}] epoch 0 validAP {metrics0['valid_ap']:.8f} trainAP {metrics0['train_ap']:.8f} "
        f"identical_to_frozen={metrics0['logits_bitwise_equal_frozen']}"
    )

    # ---- epochs 1..8 ----
    for epoch in range(1, C.EPOCHS + 1):
        if time.perf_counter() > deadline:
            status = "INCOMPLETE"
            stop_reason = "SOFT_WALL_LIMIT"
            break
        head.train()
        t_epoch = time.perf_counter()
        perm = plan["perms"][epoch - 1]
        step_losses: list[float] = []
        epoch_entries = 0
        for start in range(0, int(perm.size), C.BATCH_SIZE):
            idx = torch.from_numpy(np.ascontiguousarray(perm[start : start + C.BATCH_SIZE])).to(device)
            x = gpu["train_rho_in"].index_select(0, idx)
            y = gpu["train_labels"].index_select(0, idx)
            base = gpu["train_logits"].index_select(0, idx)
            logits = base + head(x)
            mask = torch.isfinite(y)
            loss = FM.masked_bce_torch(logits, y)
            if not torch.isfinite(loss):
                non_finite_events += 1
                raise RuntimeError(f"non-finite loss at epoch {epoch} rows {start}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                total = torch.zeros((), dtype=torch.float64, device=device)
                for param in params:
                    if param.grad is not None:
                        total += param.grad.detach().double().pow(2).sum()
            grad_norm_max = max(grad_norm_max, float(total.sqrt().item()))
            optimizer.step()
            step_losses.append(float(loss.detach().item()))
            epoch_entries += int(mask.sum().item())
        train_seconds = float(time.perf_counter() - t_epoch)
        train_seconds_total += train_seconds
        loss_all.extend(step_losses)
        step_loss_mean = float(np.mean(step_losses)) if step_losses else None

        if epoch in C.EVAL_EPOCHS:
            head.eval()
            metrics = evaluate(head, cache, epoch=epoch, compute_train_official=epoch == C.EPOCHS)
            record(epoch, metrics, train_seconds, f"epoch {epoch} official validation evaluation")
            curve[-1]["loss_entries"] = int(epoch_entries)
            C.log(
                f"[{arm}] epoch {epoch}/{C.EPOCHS} trainBCE {curve[-1]['train_bce_full']:.6f} "
                f"trainAP {curve[-1]['train_ap']:.6f} validAP {curve[-1]['valid_ap']:.6f} "
                f"delta {curve[-1]['delta_vs_ap0']:+.6f} train {train_seconds:.2f}s"
            )
            del metrics
        else:
            C.log(f"[{arm}] epoch {epoch}/{C.EPOCHS} trainBCE(step-mean) {step_loss_mean:.6f} (no eval)")
        if time.perf_counter() > deadline and epoch < C.EPOCHS:
            status = "INCOMPLETE"
            stop_reason = "SOFT_WALL_LIMIT"
            break

    total_seconds = float(time.perf_counter() - t_arm)
    peak = C.gpu_peak(device)
    final_fp = FM.head_parameter_fingerprint(head)
    curve_sha = _write_csv(out_dir / "train_curve.csv", curve, CURVE_FIELDS)
    per_task_sha = _write_csv(out_dir / "per_task_ap.csv", per_task_rows, PER_TASK_FIELDS)

    if status == "COMPLETED" and sorted(evaluated) != sorted(C.EVAL_EPOCHS):
        status = "INCOMPLETE"
        stop_reason = "MISSING_EVAL_EPOCH"
    if status == "COMPLETED" and len(loss_all) != C.EPOCHS * C.STEPS_PER_EPOCH:
        status = "INCOMPLETE"
        stop_reason = "MISSING_TRAINING_STEPS"

    epoch8 = metric_history.get(C.MAIN_EPOCH)
    best_epoch = None
    best_ap = -math.inf
    for row in curve:
        if float(row["valid_ap"]) > best_ap:
            best_ap = float(row["valid_ap"])
            best_epoch = int(row["epoch"])

    def checkpoint_payload(kind: str, epoch_used: int, valid_ap: float | None, states: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return {
            "schema": "lcrc-pcba-frozen-readout-head-checkpoint-v1",
            "round_id": C.ROUND_ID,
            "protocol_id": C.PROTOCOL_ID,
            "arm": arm,
            "arm_name": C.ARMS[arm]["name"],
            "architecture": C.ARMS[arm]["architecture"],
            "checkpoint_kind": kind,
            "epoch": int(epoch_used),
            "d": int(d),
            "n_tasks": int(C.N_TASKS),
            "trainable_parameters": int(sum(p.numel() for p in head.parameters())),
            "learning_rate": float(C.HEAD_RECIPE["learning_rate"]),
            "weight_decay": float(C.HEAD_RECIPE["weight_decay"]),
            "recipe": C.HEAD_RECIPE,
            "arm_spec": dict(C.ARMS[arm]),
            "state_dict": {k: v.detach().cpu().clone() for k, v in states.items()},
            "init_audit": init_audit,
            "init_fingerprint": init_fp,
            "final_fingerprint": final_fp,
            "valid_ap": None if valid_ap is None else float(valid_ap),
            "metric": "Validation AP (OGB official Evaluator)",
            "epoch8": None
            if epoch8 is None
            else {
                "valid_ap": float(epoch8["valid_ap"]),
                "train_ap": float(epoch8["train_ap"]),
                "train_bce_full": float(epoch8["train_bce_full"]),
                "delta_vs_ap0": float(epoch8["valid_ap"]) - float(C.AP0),
            },
            "row_plan": {
                "seed": int(plan["seed"]),
                "batch_size": int(plan["batch_size"]),
                "steps_per_epoch": int(plan["steps_per_epoch"]),
                "epochs": int(plan["epochs"]),
                "row_order_sha256": list(plan["row_order_sha256"]),
                "batch_plan_sha256": list(plan["batch_plan_sha256"]),
                "global_plan_sha256": plan["global_plan_sha256"],
            },
            "base_model": dict(base_model_audit),
            "cache": {
                "manifest": C.track_rel(C.FEATURE_CACHE / "manifest.json"),
                "manifest_sha256": cache["manifest_sha256"],
                "phase_a_audit_sha256": cache["audit_sha256"],
                "train_rho_in_content_sha256": cache["manifest"]["arrays"]["train_rho_in"]["content_sha256"],
                "train_logits_content_sha256": cache["manifest"]["arrays"]["train_logits"]["content_sha256"],
            },
            "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
            "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
            "environment": C.environment(),
            "git_commit": C.git_commit(),
            "status": status,
            "stop_reason": stop_reason,
            "test_evaluated": False,
            "created_utc": C.utcnow(),
        }

    final_epoch = max(evaluated) if evaluated else 0
    final_path = out_dir / "checkpoint_final.pt"
    final_sha = _atomic_torch_save(
        checkpoint_payload("final_fixed_epoch", int(final_epoch), None, state_history[int(final_epoch)]),
        final_path,
    )
    best_path = out_dir / "checkpoint_best.pt"
    best_sha = _atomic_torch_save(
        checkpoint_payload(
            "auxiliary_best_valid_ap", int(best_epoch or 0), float(best_ap), state_history[int(best_epoch or 0)]
        ),
        best_path,
    )
    selection = {
        "arm": arm,
        "primary_epoch": int(C.MAIN_EPOCH),
        "primary_valid_ap": None if epoch8 is None else float(epoch8["valid_ap"]),
        "auxiliary_best_epoch": int(best_epoch or 0),
        "auxiliary_best_valid_ap": float(best_ap),
        "auxiliary_best_is_primary_epoch": bool(best_epoch is not None and int(best_epoch) == int(C.MAIN_EPOCH)),
        "auxiliary_best_checkpoint": C.track_rel(best_path),
        "auxiliary_best_checkpoint_sha256": best_sha,
        "rule": "fixed epoch 8 is the primary comparison; the best-Validation checkpoint is auxiliary",
    }
    C.write_json(out_dir / "selection.json", selection)

    summary = {
        "schema": "lcrc-pcba-frozen-readout-head-arm-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "arm_spec": dict(C.ARMS[arm]),
        "status": status,
        "stop_reason": stop_reason,
        "assigned_pool": assigned_pool,
        "device": str(device),
        "d": int(d),
        "trainable_parameters": int(sum(p.numel() for p in head.parameters())),
        "recipe": C.HEAD_RECIPE,
        "init_fingerprint": init_fp,
        "init_audit": init_audit,
        "final_fingerprint": final_fp,
        "training": {
            "epochs_completed": int(max(evaluated) if evaluated else 0),
            "evaluated_epochs": sorted(evaluated),
            "n_steps": len(loss_all),
            "steps_per_epoch": int(plan["steps_per_epoch"]),
            "row_plan": {
                "seed": int(plan["seed"]),
                "global_plan_sha256": plan["global_plan_sha256"],
                "row_order_sha256": list(plan["row_order_sha256"]),
                "batch_plan_sha256": list(plan["batch_plan_sha256"]),
            },
            "loss_step_first10": [float(x) for x in loss_all[:10]],
            "loss_step_last10": [float(x) for x in loss_all[-10:]],
            "loss_sha256": C.sha256_array(np.asarray(loss_all, dtype=np.float64)),
            "grad_norm_max": float(grad_norm_max),
            "gradient_clipping": "none (not part of the frozen head recipe)",
            "non_finite_events": int(non_finite_events),
        },
        "epoch_metrics": {
            str(k): {
                "train_bce_full": float(v["train_bce_full"]),
                "valid_bce_full": float(v["valid_bce_full"]),
                "train_ap": float(v["train_ap"]),
                "train_official_ap": v["train_official_ap"],
                "valid_ap": float(v["valid_ap"]),
                "valid_macro_independent": float(v["valid_macro_independent"]),
                "n_valid_tasks_scored": int(v["n_valid_tasks_scored"]),
                "delta_vs_ap0": float(v["valid_ap"]) - float(C.AP0),
                "train_minus_valid_ap": float(v["train_minus_valid_ap"]),
                "logits_bitwise_equal_frozen": bool(v["logits_bitwise_equal_frozen"]),
                "valid_per_task": v["valid_per_task"],
            }
            for k, v in sorted(metric_history.items())
        },
        "primary": selection,
        "total_seconds": total_seconds,
        "train_seconds": float(train_seconds_total),
        "eval_seconds": float(sum(r["eval_seconds"] for r in curve)),
        "gpu_hours": float(total_seconds / 3600.0),
        "gpu_peak_allocated_bytes": int(peak["peak_allocated_bytes"]),
        "gpu_peak_reserved_bytes": int(peak["peak_reserved_bytes"]),
        "curve": C.track_rel(out_dir / "train_curve.csv"),
        "curve_sha256": curve_sha,
        "per_task_ap": C.track_rel(out_dir / "per_task_ap.csv"),
        "per_task_ap_sha256": per_task_sha,
        "checkpoint_final": C.track_rel(final_path),
        "checkpoint_final_sha256": final_sha,
        "checkpoint_best_auxiliary": C.track_rel(best_path),
        "checkpoint_best_auxiliary_sha256": best_sha,
        "base_model": dict(base_model_audit),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "test_evaluated": False,
        "finished_utc": C.utcnow(),
    }
    C.write_json(out_dir / "arm.json", summary)
    return summary


# ---------------------------------------------------------------------------
# full phase B
# ---------------------------------------------------------------------------


def run(
    *,
    device: torch.device,
    out_root: Path = C.HEADS_RUNS,
    soft_seconds: float = float(C.BUDGET["heads_soft_seconds"]),
    assigned_pool: str = "res-gpu1",
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    inherited = C.assert_inherited_constants()
    task = DATA.load_task()
    cache = load_cache(device)
    standardizer_audit = standardize_cache(cache["gpu"], cache["manifest"])
    if not (
        standardizer_audit["train_standardized_sha256_matches_phase_a"]
        and standardizer_audit["valid_standardized_sha256_matches_phase_a"]
    ):
        raise RuntimeError(f"standardised cache does not reproduce phase A: {standardizer_audit}")

    frozen_logits_sha = {
        "train": C.sha256_array(cache["gpu"]["train_logits"].cpu().numpy()),
        "valid": C.sha256_array(cache["gpu"]["valid_logits"].cpu().numpy()),
    }

    # The frozen encoder is re-loaded and fingerprinted for the "unchanged" gate.
    base_model, base_audit = MODEL.load_frozen_model(
        task, device=torch.device("cpu"), label="frozen_reference"
    )
    base_fp_before = base_audit["parameter_fingerprint"]

    plan = FM.row_plan(int(cache["gpu"]["train_rho_in"].shape[0]))
    isolation = FM.plan_isolation(plan, task.train_rows, task.valid_rows, task.test_rows)
    if not isolation["passed"]:
        raise RuntimeError(f"row plan isolation gate failed: {isolation}")
    if int(plan["steps_per_epoch"]) != C.STEPS_PER_EPOCH:
        raise RuntimeError(f"steps per epoch {plan['steps_per_epoch']} != frozen {C.STEPS_PER_EPOCH}")

    gpu_occupancy = C.gpu_occupancy()
    arms: dict[str, Any] = {}
    for arm in C.ARM_ORDER:
        remaining = float(soft_seconds) - float(time.perf_counter() - t_start)
        if remaining <= 60.0:
            C.log(f"[{arm}] budget exhausted before start")
            arms[arm] = {
                "arm": arm,
                "status": "INCOMPLETE",
                "stop_reason": "BUDGET_EXHAUSTED_BEFORE_START",
            }
            continue
        arms[arm] = train_arm(
            arm,
            cache=cache,
            plan=plan,
            out_dir=out_root / arm,
            device=device,
            base_model_audit=base_audit,
            soft_seconds=remaining,
            assigned_pool=assigned_pool,
        )
        arms[arm]["gpu_occupancy_start"] = gpu_occupancy
        C.write_json(out_root / arm / "arm.json", arms[arm])

    base_fp_after = MODEL.parameter_digests(base_model)["parameter_fingerprint"]
    base_grads = any(p.grad is not None for p in base_model.parameters())
    del base_model
    frozen_logits_after = {
        "train": C.sha256_array(cache["gpu"]["train_logits"].cpu().numpy()),
        "valid": C.sha256_array(cache["gpu"]["valid_logits"].cpu().numpy()),
    }
    plan_hashes = {
        arm: arms.get(arm, {}).get("training", {}).get("row_plan", {}).get("global_plan_sha256")
        for arm in C.ARM_ORDER
    }
    gates = {
        "phase_a_passed": True,
        "cache_hashes_verified": bool(all(v["hash_ok"] for v in cache["files"].values())),
        "standardizer_reproduces_phase_a": True,
        "row_plan_isolated_from_valid_and_test": bool(isolation["passed"]),
        "shared_row_plan_identical": bool(
            len([h for h in plan_hashes.values() if h == plan["global_plan_sha256"]]) == len(C.ARM_ORDER)
        ),
        "epoch0_identity_both_arms": bool(
            all(arms.get(arm, {}).get("epoch_metrics", {}).get("0", {}).get("logits_bitwise_equal_frozen") for arm in C.ARM_ORDER)
        ),
        "both_arms_completed": bool(all(arms.get(a, {}).get("status") == "COMPLETED" for a in C.ARM_ORDER)),
        "base_model_unchanged": bool(base_fp_before == base_fp_after == base_audit["parameter_fingerprint"]),
        "base_model_has_no_gradients": bool(not base_grads),
        "base_model_requires_grad_false": bool(not base_audit["requires_grad_any"]),
        "frozen_logits_unchanged": bool(frozen_logits_sha == frozen_logits_after),
        "eval_epochs_complete": bool(
            all(
                arms.get(a, {}).get("training", {}).get("evaluated_epochs") == sorted(C.EVAL_EPOCHS)
                for a in C.ARM_ORDER
            )
        ),
        "primary_epoch_8_present": bool(
            all(arms.get(a, {}).get("epoch_metrics", {}).get(str(C.MAIN_EPOCH)) is not None for a in C.ARM_ORDER)
        ),
        "no_non_finite_events": bool(
            all(arms.get(a, {}).get("training", {}).get("non_finite_events", 1) == 0 for a in C.ARM_ORDER)
        ),
        "within_soft_deadline": bool(float(time.perf_counter() - t_start) <= float(soft_seconds)),
        "test_evaluated_false": True,
    }
    gates = {k: bool(v) for k, v in gates.items()}
    total_seconds = float(time.perf_counter() - t_start)
    run_payload = {
        "schema": "lcrc-pcba-frozen-readout-heads-run-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "phase": "B_residual_heads",
        "all_passed": bool(all(gates.values())),
        "gates": gates,
        "failed_gates": [k for k, v in gates.items() if not v],
        "inherited": inherited,
        "device": str(device),
        "assigned_pool": assigned_pool,
        "environment": C.environment(),
        "gpu_occupancy_start": gpu_occupancy,
        "gpu_occupancy_end": C.gpu_occupancy(),
        "cache": {
            "manifest": C.track_rel(C.FEATURE_CACHE / "manifest.json"),
            "manifest_sha256": cache["manifest_sha256"],
            "audit_sha256": cache["audit_sha256"],
            "files": cache["files"],
            "standardizer": standardizer_audit,
            "frozen_logits_sha256_before_training": frozen_logits_sha,
            "frozen_logits_sha256_after_training": frozen_logits_after,
        },
        "base_model": base_audit,
        "base_model_fingerprint_before": base_fp_before,
        "base_model_fingerprint_after": base_fp_after,
        "base_model_gradients_present": bool(base_grads),
        "row_plan": {
            "seed": int(plan["seed"]),
            "batch_size": int(plan["batch_size"]),
            "steps_per_epoch": int(plan["steps_per_epoch"]),
            "epochs": int(plan["epochs"]),
            "global_plan_sha256": plan["global_plan_sha256"],
            "row_order_sha256": list(plan["row_order_sha256"]),
            "batch_plan_sha256": list(plan["batch_plan_sha256"]),
            "isolation": isolation,
            "same_plan_for_both_arms": True,
        },
        "arms": {
            arm: {
                key: arms[arm].get(key)
                for key in (
                    "status",
                    "stop_reason",
                    "trainable_parameters",
                    "init_audit",
                    "init_fingerprint",
                    "final_fingerprint",
                    "training",
                    "primary",
                    "epoch_metrics",
                    "total_seconds",
                    "gpu_hours",
                    "gpu_peak_allocated_bytes",
                    "gpu_peak_reserved_bytes",
                    "checkpoint_final",
                    "checkpoint_final_sha256",
                    "checkpoint_best_auxiliary_sha256",
                    "curve_sha256",
                    "per_task_ap_sha256",
                )
            }
            for arm in arms
        },
        "total_seconds": total_seconds,
        "gpu_hours": float(total_seconds / 3600.0),
        "budget": C.BUDGET,
        "decision_rule": C.DECISION,
        "test_evaluated": False,
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
    }
    C.write_json(out_root / "run.json", run_payload)
    C.log(
        f"phase B all_passed={run_payload['all_passed']} "
        + " ".join(
            f"{a}:{arms.get(a, {}).get('epoch_metrics', {}).get('8', {}).get('valid_ap')}"
            for a in C.ARM_ORDER
        )
        + f" gpu_hours={run_payload['gpu_hours']:.6f}"
    )
    return {"all_passed": run_payload["all_passed"], "run": run_payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", default=str(C.HEADS_RUNS))
    parser.add_argument("--assigned-pool", default="res-gpu1")
    parser.add_argument("--soft-seconds", type=float, default=float(C.BUDGET["heads_soft_seconds"]))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            Path(args.out_root) / "run.json",
            {
                "schema": "lcrc-pcba-frozen-readout-heads-run-v1",
                "round_id": C.ROUND_ID,
                "phase": "B_residual_heads",
                "all_passed": False,
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        return 2
    try:
        result = run(
            device=device,
            out_root=Path(args.out_root),
            soft_seconds=float(args.soft_seconds),
            assigned_pool=str(args.assigned_pool),
        )
        return 0 if result["all_passed"] else 3
    except Exception as exc:  # noqa: BLE001 - the failure is the deliverable
        C.write_json(
            Path(args.out_root) / "run.json",
            {
                "schema": "lcrc-pcba-frozen-readout-heads-run-v1",
                "round_id": C.ROUND_ID,
                "phase": "B_residual_heads",
                "all_passed": False,
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        C.log(f"phase B FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
