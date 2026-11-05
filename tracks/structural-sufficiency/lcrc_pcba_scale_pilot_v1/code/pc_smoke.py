"""PCBA correctness + throughput smoke (no hyper-parameter selection).

The smoke answers three questions only:

1. does the inherited C1 pipeline produce the right shapes and finite numbers on
   the real 128-task cache,
2. are the mandatory numerical identities (masked loss, official AP, reload,
   batch/single, atom renumbering) satisfied,
3. how long does one *full* epoch take, and does the 40-epoch ceiling fit the
   4 A100 GPU-hour budget.

Nothing here chooses a model, a recipe or a data rule from an AP value.
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F  # noqa: E402

import pc_common as C  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402
import pc_train as TRAIN  # noqa: E402


def _permute_single(batch: Mapping[str, Any], perm: np.ndarray) -> dict[str, Any]:
    """Permute the atoms of a 1-graph collated batch (``perm``: new -> old)."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    out = dict(batch)
    dev = batch["atom_feat"].device
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=dev)
    for key in ("atom_feat", "atom_scalar", "atom_type"):
        out[key] = batch[key][perm_t]
    for key in ("bond_index", "nbr_atom", "path_atom", "path_end"):
        src = batch[key].detach().cpu().numpy()
        out[key] = torch.as_tensor(inv[src], dtype=batch[key].dtype, device=dev)
    return out


def check_masked_loss(device: torch.device) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(16, 8)).astype(np.float32) * 2.0
    y = (rng.random((16, 8)) < 0.5).astype(np.float32)
    y[0, :] = np.nan
    y[3, 2] = np.nan
    y[7, 5] = np.nan
    got = float(
        MET.masked_bce_loss(
            torch.as_tensor(logits, device=device), torch.as_tensor(y, device=device)
        ).item()
    )
    want = MET.masked_bce_numpy(logits, y)
    all_nan = np.full((4, 3), np.nan, dtype=np.float32)
    zero = float(
        MET.masked_bce_loss(
            torch.as_tensor(rng.normal(size=(4, 3)), device=device),
            torch.as_tensor(all_nan, device=device),
        ).item()
    )
    # A NaN-as-negative bug would give a different (lower) loss; a NaN-as-zero
    # bug would produce NaN.  Compare against the explicit float64 reference.
    zero_filled = MET.masked_bce_numpy(logits, np.where(np.isfinite(y), y, 0.0))
    return {
        "torch_loss": got,
        "numpy_reference": want,
        "abs_delta": abs(got - want),
        "nan_as_zero_counterfactual": zero_filled,
        "delta_vs_nan_as_zero": abs(got - zero_filled),
        "all_nan_batch_loss": zero,
        "n_observed_entries": int(np.isfinite(y).sum()),
        "n_total_entries": int(y.size),
        "ok": bool(
            np.isfinite(got)
            and abs(got - want) <= 1e-6
            and abs(got - zero_filled) > 1e-9
            and zero == 0.0
        ),
    }


def check_gradients(
    model: torch.nn.Module, task: TRAIN.Task, device: torch.device, rows: np.ndarray
) -> dict[str, Any]:
    model.train()
    batch = BM.to_device(BM.collate(task.cache, rows.tolist()), device)
    y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
    logits = model(batch)
    loss = MET.masked_bce_loss(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    worst = 0.0
    n_nonfinite = 0
    total = 0
    for _name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        total += int(g.numel())
        n_nonfinite += int((~torch.isfinite(g)).sum().item())
        worst = max(worst, float(g.abs().max().item()))
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "loss": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "n_grad_elements": total,
        "n_nonfinite_grad": int(n_nonfinite),
        "max_abs_grad": worst,
        "ok": bool(torch.isfinite(loss).item() and n_nonfinite == 0 and total > 0),
    }


def check_reload(
    model: torch.nn.Module, task: TRAIN.Task, device: torch.device, rows: np.ndarray, seed: int, out_dir: Path
) -> dict[str, Any]:
    before = BM.predict_logits(model, task.cache, rows, device, batch_size=128)
    path = out_dir / "smoke_checkpoint.pt"
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    TRAIN._atomic_torch_save({"state_dict": state, "seed": int(seed), "n_outputs": C.N_TASKS}, path)
    fresh, _info = BM.build_c1(task.spec, task.norm, C.N_TASKS, int(seed), device)
    fresh.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    after = BM.predict_logits(fresh, task.cache, rows, device, batch_size=128)
    delta = float(np.max(np.abs(before - after)))
    sha = C.sha256_file(path)
    path.unlink()
    return {
        "n_rows": int(len(rows)),
        "max_abs_logit_delta": delta,
        "checkpoint_sha256": sha,
        "reloaded_state_matches": delta <= C.RELOAD_TOL,
        "ok": delta <= C.RELOAD_TOL,
    }


def check_single_vs_batch(
    model: torch.nn.Module, task: TRAIN.Task, device: torch.device, rows: np.ndarray
) -> dict[str, Any]:
    model.eval()
    batch = BM.to_device(BM.collate(task.cache, rows.tolist()), device)
    with torch.no_grad():
        batched = model(batch).cpu().numpy()
        singles = []
        for r in rows.tolist():
            one = BM.to_device(BM.collate(task.cache, [r]), device)
            singles.append(model(one).cpu().numpy()[0])
    singles = np.stack(singles, axis=0)
    delta = float(np.max(np.abs(batched - singles)))
    return {
        "n_molecules": int(len(rows)),
        "max_abs_logit_delta": delta,
        "ok": delta <= C.INFERENCE_TOL,
    }


def check_atom_renumbering(
    model: torch.nn.Module, task: TRAIN.Task, device: torch.device, row: int, seed: int
) -> dict[str, Any]:
    model.eval()
    batch = BM.to_device(BM.collate(task.cache, [int(row)]), device)
    n_atoms = int(batch["atom_feat"].shape[0])
    rng = np.random.default_rng(int(seed) + 7)
    perm = rng.permutation(n_atoms)
    permuted = _permute_single(batch, perm)
    with torch.no_grad():
        a = model(batch).cpu().numpy()
        b = model(permuted).cpu().numpy()
    delta = float(np.max(np.abs(a - b)))
    return {
        "cache_row": int(row),
        "n_atoms": n_atoms,
        "max_abs_logit_delta": delta,
        "ok": delta <= C.INFERENCE_TOL,
    }


def check_ap(
    model: torch.nn.Module, task: TRAIN.Task, device: torch.device, rows: np.ndarray
) -> dict[str, Any]:
    logits = BM.predict_logits(model, task.cache, rows, device, batch_size=256)
    probs = BM.sigmoid(logits)
    y = task.labels[rows]
    agreement = MET.ap_agreement(y, probs)
    agreement["n_rows"] = int(len(rows))
    agreement["n_nan_entries"] = int((~np.isfinite(y)).sum())
    agreement["n_observed_entries"] = int(np.isfinite(y).sum())
    agreement["ok"] = bool(agreement["abs_delta"] <= C.AP_AGREEMENT_TOL)
    return agreement


def smoke_epochs(
    task: TRAIN.Task,
    model: torch.nn.Module,
    device: torch.device,
    *,
    seed: int,
    batch_size: int,
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    epochs: int,
    formal_steps_per_epoch: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Two short epochs on a fixed row subset; timing only, no selection."""
    recipe = dict(C.RECIPE)
    horizon = int(recipe["scheduler_horizon_epochs"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(recipe["peak_lr"]),
        betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(formal_steps_per_epoch * horizon, 1), eta_min=float(recipe["min_lr"])
    )
    generator = torch.Generator().manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    rows_log: list[dict[str, Any]] = []
    step_seconds: list[float] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        t_epoch = time.perf_counter()
        losses = []
        for rows in TRAIN._batches(train_rows, batch_size, generator):
            t0 = time.perf_counter()
            batch = BM.to_device(BM.collate(task.cache, rows.tolist()), device)
            y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
            logits = model(batch)
            loss = MET.masked_bce_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError("non-finite loss/gradient in the smoke loop")
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - t0)
            losses.append(float(loss.detach().cpu().item()))
        train_seconds = time.perf_counter() - t_epoch
        t_eval = time.perf_counter()
        logits_valid = BM.predict_logits(model, task.cache, valid_rows, device, batch_size=256)
        report = MET.validation_report(task.labels[valid_rows], BM.sigmoid(logits_valid))
        valid_seconds = time.perf_counter() - t_eval
        rows_log.append(
            {
                "epoch": epoch,
                "n_train_rows": int(len(train_rows)),
                "steps": int(len(losses)),
                "train_loss": float(np.mean(losses)),
                "valid_ap_subset": float(report["ap"]),
                "n_tasks_scored": int(report["n_tasks_scored"]),
                "train_seconds": float(train_seconds),
                "valid_seconds": float(valid_seconds),
                "seconds_per_step": float(train_seconds / max(len(losses), 1)),
            }
        )
        C.log(
            f"[smoke] epoch {epoch} loss {rows_log[-1]['train_loss']:.5f} "
            f"validAP_subset {rows_log[-1]['valid_ap_subset']:.4f} "
            f"{rows_log[-1]['seconds_per_step'] * 1000:.1f} ms/step"
        )
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    sec_per_step = float(np.median(step_seconds)) if step_seconds else float("nan")
    valid_per_row = float(np.mean([r["valid_seconds"] / max(len(valid_rows), 1) for r in rows_log]))
    projected_train = sec_per_step * formal_steps_per_epoch
    projected_valid = valid_per_row * C.SPLIT_SIZES["valid"]
    projected_epoch = projected_train + projected_valid
    projected_40 = projected_epoch * int(C.RECIPE["max_epochs"]) / 3600.0
    return {
        "epochs": rows_log,
        "median_seconds_per_step": sec_per_step,
        "train_rows": int(len(train_rows)),
        "valid_rows": int(len(valid_rows)),
        "formal_steps_per_epoch": int(formal_steps_per_epoch),
        "projected_train_seconds_per_epoch": projected_train,
        "projected_valid_seconds_per_epoch": projected_valid,
        "projected_seconds_per_epoch": projected_epoch,
        "projected_40_epoch_hours": projected_40,
        "gpu_peak_allocated_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
        "n_steps_timed": len(step_seconds),
        "first_step_seconds": float(step_seconds[0]) if step_seconds else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", default=str(C.CACHE / "pcba_v1"))
    parser.add_argument("--labels", default=str(C.ANALYSIS / "pcba_labels_trainvalid.npy"))
    parser.add_argument("--spec", default=str(C.ANALYSIS / "pcba_spec.json"))
    parser.add_argument("--split", default=str(C.ANALYSIS / "pcba_split.json"))
    parser.add_argument("--train-rows", type=int, default=int(C.BUDGET["smoke_train_rows"]))
    parser.add_argument("--valid-rows", type=int, default=int(C.BUDGET["smoke_valid_rows"]))
    parser.add_argument("--epochs", type=int, default=int(C.BUDGET["smoke_epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(C.RECIPE["batch_size"]))
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--out", default=str(C.ANALYSIS / "smoke.json"))
    parser.add_argument("--work-dir", default=str(C.RUNS / "pcba_smoke"))
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    payload: dict[str, Any] = {
        "schema": "lcrc-pcba-smoke-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "purpose": "correctness, throughput, memory, numerical stability; no hyper-parameter selection",
        "git_commit": C.git_commit(),
        "device": str(device),
        "environment": C.environment(),
    }
    t_start = time.perf_counter()
    try:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        C.assert_frozen_inheritance()
        C.seed_everything(int(args.seed))
        task = TRAIN.load_task(Path(args.cache), Path(args.labels), Path(args.spec), Path(args.split))
        payload["data_load_seconds"] = float(task.load_seconds)
        payload["cache"] = {
            "path": C.track_rel(task.cache_path),
            "sha256": C.sha256_tree(task.cache_path)
            if Path(task.cache_path).is_dir()
            else C.sha256_file(task.cache_path),
            "bytes": int(sum(f.stat().st_size for f in Path(task.cache_path).rglob("*") if f.is_file()))
            if Path(task.cache_path).is_dir()
            else int(Path(task.cache_path).stat().st_size),
            "format": "npy-directory-v1" if Path(task.cache_path).is_dir() else "pickle",
            "n_molecules": int(task.cache["n_molecules"]),
            "n_atoms_total": int(task.cache["meta"]["n_atoms_total"]),
            "n_s2_paths": int(task.cache["meta"]["n_s2_paths"]),
            "feature_spec_hash": task.spec.hash(),
            "labels_present": bool(task.cache["meta"].get("labels_present")),
            "label_like_keys": sorted(
                str(k) for k in task.cache.keys() if str(k).lower() in C.LABEL_KEYS
            ),
        }
        model, model_info = BM.build_c1(task.spec, task.norm, C.N_TASKS, int(args.seed), device)
        payload["model"] = {
            "variant": model_info["variant"],
            "parameters": int(model_info["parameters"]),
            "n_outputs": int(model_info["n_outputs"]),
            "local_input_dim": int(model_info["local_input_dim"]),
            "local_layout": model_info["local_layout"],
            "has_b_mean": "b_mean" in model_info["local_layout"],
            "init_state_hash": model_info["init_state_hash"],
        }
        formal_steps = int(math.ceil(C.SPLIT_SIZES["train"] / int(args.batch_size)))
        payload["formal_steps_per_epoch"] = formal_steps

        train_rows = np.asarray(task.split["train"][: int(args.train_rows)], dtype=np.int64)
        valid_rows = np.asarray(task.split["valid"][: int(args.valid_rows)], dtype=np.int64)
        payload["smoke_rows"] = {"train": int(len(train_rows)), "valid": int(len(valid_rows))}

        # --- shape / dropout / forward gate on a full batch
        probe_rows = train_rows[: int(args.batch_size)]
        batch = BM.to_device(BM.collate(task.cache, probe_rows.tolist()), device)
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        payload["shape_gate"] = {
            "batch_size": int(len(probe_rows)),
            "logits_shape": list(logits.shape),
            "expected_shape": [int(len(probe_rows)), C.N_TASKS],
            "finite_logits": bool(torch.isfinite(logits).all().item()),
            "output_dim_128": bool(logits.shape[1] == C.N_TASKS),
            "ok": bool(logits.shape == (len(probe_rows), C.N_TASKS) and torch.isfinite(logits).all().item()),
        }
        payload["masked_loss_gate"] = check_masked_loss(device)
        payload["gradient_gate"] = check_gradients(model, task, device, probe_rows)
        payload["reload_gate"] = check_reload(
            model, task, device, valid_rows[:256], int(args.seed), work_dir
        )
        payload["single_vs_batch_gate"] = check_single_vs_batch(model, task, device, valid_rows[:8])
        payload["atom_renumbering_gate"] = check_atom_renumbering(
            model, task, device, int(valid_rows[0]), int(args.seed)
        )
        payload["ap_gate"] = check_ap(model, task, device, valid_rows)
        payload["label_masking_gate"] = {
            "test_rows_all_nan": bool(np.isnan(task.labels[task.test_rows]).all()),
            "n_test_rows": int(task.test_rows.size),
            "test_rows_indexed_by_training_code": False,
            "ok": bool(np.isnan(task.labels[task.test_rows]).all()),
        }
        payload["timing"] = smoke_epochs(
            task,
            model,
            device,
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            train_rows=train_rows,
            valid_rows=valid_rows,
            epochs=int(args.epochs),
            formal_steps_per_epoch=formal_steps,
            out_dir=work_dir,
        )
        payload["smoke_seconds"] = float(time.perf_counter() - t_start)
        projected = float(payload["timing"]["projected_40_epoch_hours"])
        payload["budget"] = {
            **C.BUDGET,
            "projected_40_epoch_hours": projected,
            "projected_total_with_smoke_hours": projected + payload["smoke_seconds"] / 3600.0,
            "within_total_budget": bool(
                projected + payload["smoke_seconds"] / 3600.0 <= float(C.BUDGET["gpu_hours_total"])
            ),
        }
    except Exception as exc:  # noqa: BLE001 - recorded, gate fails
        payload["error"] = {"type": type(exc).__name__, "detail": str(exc), "traceback": traceback.format_exc()}
        payload["status"] = "FAILED"
        payload["gates_ok"] = False
        C.write_json(out_path, payload)
        C.log(f"SMOKE FAILED: {exc}")
        return 1
    gate_names = (
        "shape_gate",
        "masked_loss_gate",
        "gradient_gate",
        "reload_gate",
        "single_vs_batch_gate",
        "atom_renumbering_gate",
        "ap_gate",
        "label_masking_gate",
    )
    payload["gates"] = {name: bool(payload[name]["ok"]) for name in gate_names}
    payload["gates_ok"] = bool(all(payload["gates"].values()))
    payload["status"] = "OK" if payload["gates_ok"] else "FAILED"
    C.write_json(out_path, payload)
    C.log(
        f"smoke {payload['status']} gates={payload['gates']} "
        f"projected_40ep={payload['budget']['projected_40_epoch_hours']:.2f}h"
    )
    return 0 if payload["gates_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
