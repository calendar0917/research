"""SSRL v0 continued training: one arm per process, frozen W3 computation.

The three arms (ABS / PAIR / RECT) share

* the exact W3 checkpoint and therefore bitwise-identical initial parameters,
* the exact same rectangle + plain-molecule plan and per-epoch batch order,
* the same optimiser recipe and fresh AdamW state.

They differ only in the auxiliary term of the training loss.  No parameter is
added, no architecture branch is introduced, no teacher or pseudo-label exists,
and no Test label is indexed.

Examples::

    CUDA_VISIBLE_DEVICES=0 python ssrl_train.py --arm ABS  --epochs 5
    CUDA_VISIBLE_DEVICES=1 python ssrl_train.py --arm PAIR --epochs 5
    python ssrl_train.py --arm RECT --epochs 5 --device cpu --max-steps 2 --valid-limit 1024
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ssrl_common as C


def _torch():
    import torch

    return torch


def _load_modules():
    import ew_collate as EC
    import ew_model as EM
    import pq_data as PQD
    import pq_train as PQT

    return EC, EM, PQD, PQT


def _log(msg: str) -> None:
    print(f"[ssrl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_plan(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def build_model(device: str, source_sha: str):
    torch = _torch()
    EC, EM, PQD, PQT = _load_modules()
    spec = C.load_spec()
    norm = C.load_feature_norm()
    label_norm = C.load_label_norm()
    model, audit = EM.build_model(spec, norm, scale=3, seed=0, device=device)
    if int(audit["total_parameters"]) != 2_038_167:
        raise RuntimeError(f"W3 parameter count {audit['total_parameters']} != 2,038,167")
    payload = torch.load(C.LCW_CHECKPOINT, map_location="cpu", weights_only=False)
    if str(payload.get("schema")) != "lcrc-pcqm4mv2-efficiency-width-checkpoint-v1":
        raise RuntimeError("unexpected checkpoint schema")
    if int(payload.get("epoch", -1)) != 39:
        raise RuntimeError(f"checkpoint epoch {payload.get('epoch')} != 39")
    if abs(float(payload.get("valid_mae_eV", -1)) - 0.12780916721022834) > 1e-12:
        raise RuntimeError("checkpoint Valid MAE is not the frozen 0.127809167")
    if C.sha256_file(C.LCW_CHECKPOINT) != source_sha:
        raise RuntimeError("checkpoint sha256 changed")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint state mismatch: missing={missing} unexpected={unexpected}")
    model.to(device)
    return {
        "model": model,
        "audit": audit,
        "payload": payload,
        "spec": spec,
        "norm": norm,
        "label_norm": label_norm,
        "EC": EC,
        "EM": EM,
        "PQD": PQD,
        "PQT": PQT,
        "device": device,
    }


def state_sha(model) -> str:
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return C.sha256_arrays(arrays)


def make_optimizer(model, cfg: Mapping[str, Any]):
    torch = _torch()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg["optim"]["lr_start"]),
        betas=tuple(float(x) for x in cfg["optim"]["adam_betas"]),
        eps=float(cfg["optim"]["adam_eps"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )
    audit = {
        "optimizer": "AdamW",
        "fresh_state": True,
        "n_parameters": int(sum(p.numel() for p in params)),
        "n_tensors": int(len(params)),
        "lr_start": float(cfg["optim"]["lr_start"]),
        "lr_end": float(cfg["optim"]["lr_end"]),
        "schedule": "per-step linear over the frozen horizon",
        "betas": [float(x) for x in cfg["optim"]["adam_betas"]],
        "eps": float(cfg["optim"]["adam_eps"]),
        "weight_decay": float(cfg["optim"]["weight_decay"]),
        "clip": float(cfg["optim"]["gradient_clip"]),
    }
    return opt, audit


def set_lr(opt, *, step: int, total_steps: int, start: float, end: float) -> float:
    frac = 0.0 if int(total_steps) <= 1 else float(step) / float(int(total_steps) - 1)
    lr = float(start) + (float(end) - float(start)) * frac
    for group in opt.param_groups:
        group["lr"] = lr
    return lr


def forward_unique(model, EC, cache, rows: np.ndarray, device: str, EM):
    """Forward each unique row once; returns predictions for the input order."""
    torch = _torch()
    rows = np.asarray(rows, np.int64)
    unique, inverse = np.unique(rows, return_inverse=True)
    batch = EC.collate_v2(cache, unique.tolist())
    batch = {k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    pred_unique = EM.predict_z(model, batch)
    inv = torch.from_numpy(inverse.astype(np.int64)).to(device)
    return pred_unique[inv], int(unique.size), int(rows.size)


def loss_terms(pred, target, n_rect: int):
    """The three frozen loss terms, exposed for the manual small-sample check.

    ``pred`` and ``target`` are the 128 graph positions in the endpoint order
    ``[C1+A, C1+B, C2+A, C2+B]`` per rectangle, followed by the plain graphs.
    """
    torch = _torch()
    n_rect = int(n_rect)
    n_graphs = int(pred.numel())
    pr_rect = pred[: 4 * n_rect].reshape(n_rect, 4)
    pr_plain = pred[4 * n_rect:]
    yr = target[: 4 * n_rect].reshape(n_rect, 4)
    yp = target[4 * n_rect:]
    er = pr_rect - yr
    ep = pr_plain - yp
    loss_abs = (er.abs().sum() + ep.abs().sum()) / float(n_graphs)
    d1 = er[:, 1] - er[:, 0]
    d2 = er[:, 3] - er[:, 2]
    aux_pair = ((d1.abs() + d2.abs()) / 4.0).mean() if n_rect else pred.new_zeros(())
    aux_rect = ((d2 - d1).abs() / 4.0).mean() if n_rect else pred.new_zeros(())
    return {
        "loss_abs": loss_abs,
        "aux_pair": aux_pair,
        "aux_rect": aux_rect,
        "n_rect": n_rect,
        "n_graphs": n_graphs,
        "n_plain": int(n_graphs - 4 * n_rect),
    }


def train_batch(model, EC, cache, plan, rect_idx, plain_idx, device, EM):
    torch = _torch()
    rr = plan["rect_cache_row"][rect_idx]          # (B,4)
    pr = plan["plain_cache_row"][plain_idx]        # (4B,)
    rows = np.concatenate([rr.ravel(), pr])
    target = np.concatenate([plan["rect_target_z"][rect_idx].ravel(),
                             plan["plain_target_z"][plain_idx]]).astype(np.float32)
    pred, n_unique, n_graphs = forward_unique(model, EC, cache, rows, device, EM)
    y = torch.from_numpy(target).to(device)
    terms = loss_terms(pred, y, int(rect_idx.size))
    terms.update({
        "n_graphs": int(n_graphs),
        "n_unique": int(n_unique),
    })
    return terms


def arm_loss(cfg: Mapping[str, Any], arm: str, terms: Mapping[str, Any]):
    if arm == "ABS":
        return terms["loss_abs"]
    if arm == "PAIR":
        return terms["loss_abs"] + float(cfg["loss"]["coefficients"]["pair"]) * terms["aux_pair"]
    if arm == "RECT":
        return terms["loss_abs"] + float(cfg["loss"]["coefficients"]["rect"]) * terms["aux_rect"]
    raise KeyError(arm)


def predict_rows(model, EC, cache, rows: np.ndarray, device: str, EM,
                 *, batch_size: int = 512) -> tuple[np.ndarray, dict[str, Any]]:
    torch = _torch()
    rows = np.asarray(rows, np.int64)
    out = np.empty(rows.size, np.float32)
    model.eval()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, rows.size, int(batch_size)):
            chunk = rows[i:i + int(batch_size)]
            batch = EC.collate_v2(cache, chunk.tolist())
            batch = {k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            pred = EM.predict_z(model, batch)
            out[i:i + chunk.size] = pred.detach().float().cpu().numpy()
    return out, {
        "seconds": float(time.perf_counter() - t0),
        "n": int(rows.size),
        "batch_size": int(batch_size),
        "pred_z_sha256": C.sha256_bytes(out.tobytes()),
        "pred_z_mean": float(out.mean()),
        "pred_z_std": float(out.std()),
    }


def evaluate_valid(model, EC, task, plan, device, EM, label_norm, evaluator,
                   limit: int | None = None) -> tuple[dict[str, Any], np.ndarray]:
    cache = task["valid_cache"]
    rows = np.arange(int(cache["n_molecules"]), dtype=np.int64)
    y_all = np.asarray(task["valid_y_eV"], np.float64)
    if limit is not None and int(limit) < rows.size:
        rows = rows[: int(limit)]
        y_all = y_all[: int(limit)]
    pred_z, info = predict_rows(model, EC, cache, rows, device, EM, batch_size=512)
    pred_eV = C.from_z(pred_z, label_norm)
    y = y_all
    mae_np = float(np.mean(np.abs(pred_eV - y)))
    official = None
    if evaluator is not None:
        official = float(evaluator.eval({"y_pred": pred_eV, "y_true": y})["mae"])
    payload = {
        "n": int(rows.size),
        "valid_rows": rows,
        "mae_eV": mae_np if official is None else official,
        "mae_eV_numpy": mae_np,
        "crosscheck_delta_eV": abs((mae_np if official is None else official) - mae_np),
        "pred_mean_eV": float(pred_eV.mean()),
        "pred_std_eV": float(pred_eV.std()),
        "pred_min_eV": float(pred_eV.min()),
        "pred_max_eV": float(pred_eV.max()),
        "target_mean_eV": float(y.mean()),
        "target_std_eV": float(y.std()),
        "pred_z_sha256": info["pred_z_sha256"],
        "inference_seconds": info["seconds"],
        "limit": None if limit is None else int(limit),
    }
    return payload, pred_z


def evaluate_train_subset(model, EC, task, plan, device, EM, label_norm,
                          rows: np.ndarray) -> dict[str, Any]:
    cache = task["train_cache"]
    pred_z, info = predict_rows(model, EC, cache, rows, device, EM, batch_size=512)
    pred_eV = C.from_z(pred_z, label_norm)
    y = np.asarray(task["train_y_eV"], np.float64)[rows]
    return {
        "n_unique_molecules": int(rows.size),
        "mae_eV": float(np.mean(np.abs(pred_eV - y))),
        "l1_z": float(np.mean(np.abs(pred_z - C.to_z(y, label_norm)))),
        "pred_mean_eV": float(pred_eV.mean()),
        "pred_std_eV": float(pred_eV.std()),
        "inference_seconds": info["seconds"],
        "pred_z_sha256": info["pred_z_sha256"],
        "pred_z": pred_z,
        "pred_eV": pred_eV,
    }


def save_checkpoint(path: Path, model, *, arm: str, epoch: int, global_step: int,
                    valid_mae: float, best_epoch: int, best_valid_mae: float,
                    final_valid_mae: float, plan_sha: str, config_sha: str,
                    source_sha: str, initial_sha: str, cfg: Mapping[str, Any],
                    training_seconds: float, gpu_hours: float) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ssrl-v0-checkpoint",
        "round": cfg["round"],
        "arm": arm,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "valid_mae_eV": float(valid_mae),
        "best_epoch": int(best_epoch),
        "best_valid_mae_eV": float(best_valid_mae),
        "final_valid_mae_eV": float(final_valid_mae),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "n_outputs": 1,
        "source_checkpoint": C.track_rel(C.LCW_CHECKPOINT),
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_epoch": 39,
        "source_checkpoint_valid_mae_eV": 0.12780916721022834,
        "initial_state_sha256": initial_sha,
        "plan_sha256": plan_sha,
        "config_sha256": config_sha,
        "optimizer": "AdamW (fresh)",
        "training_seconds": float(training_seconds),
        "gpu_hours_process": float(gpu_hours),
        "test_evaluated": False,
        "test_policy": "official Test labels are hidden and never indexed",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["ABS", "PAIR", "RECT"])
    parser.add_argument("--config", default=str(C.CONFIG_PATH))
    parser.add_argument("--plan", default=str(C.DATA_DIR / "train_plan.npz"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many formal steps (preflight/smoke)")
    parser.add_argument("--valid-limit", type=int, default=None,
                        help="evaluate only the first N official Valid rows (smoke only)")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args(argv)

    cfg = C.read_json(args.config)
    config_sha = C.sha256_file(args.config)
    arm = str(args.arm)
    epochs = int(args.epochs if args.epochs is not None else cfg["horizon_epochs"])
    if epochs not in (3, 5):
        raise RuntimeError(f"frozen horizon must be 3 or 5, got {epochs}")
    out_dir = Path(args.out_dir) if args.out_dir else (C.RUNS_DIR / f"PCQM_SSRL_{arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    max_seconds = float(args.max_seconds if args.max_seconds is not None else 880.0)

    C.set_torch_threads(int(args.torch_threads))
    torch = _torch()
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(int(cfg["seed"]))
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cuda":
        device = "cuda:0" if torch.cuda.device_count() == 1 else "cuda"

    plan = load_plan(Path(args.plan))
    manifest_path = C.DATA_DIR / "train_plan_manifest.json"
    manifest = C.read_json(manifest_path)
    plan_sha = C.sha256_file(args.plan)
    if plan_sha != manifest["plan_sha256"]:
        raise RuntimeError("train plan sha256 does not match its manifest")
    if int(plan["rect_cache_row"].shape[0]) != int(manifest["rectangles"]):
        raise RuntimeError("plan rectangle count does not match its manifest")
    if int(plan["plain_cache_row"].size) != int(manifest["plain_molecules"]):
        raise RuntimeError("plan plain count does not match its manifest")
    if int(plan["rect_order"].shape[0]) < epochs:
        raise RuntimeError("plan does not contain enough pre-computed epochs")

    _log(f"loading frozen W3 and caches (arm {arm}, epochs {epochs}, device {device})")
    built = build_model(device, str(cfg["checkpoint"]["sha256"]))
    model = built["model"]
    EC, EM = built["EC"], built["EM"]
    initial_sha = state_sha(model)
    opt, opt_audit = make_optimizer(model, cfg)

    # Caches and labels via the frozen transfer-pilot task loader.  Hash-tree
    # verification is disabled here because every selected row is checked
    # against the frozen 1M gid vector below; the check script repeats the
    # stronger identity checks independently.
    task = built["PQD"].load_task("PCQM_W2_1M", verify_hashes=False)
    if not np.array_equal(np.asarray(task["train_cache"]["gids"], np.int64),
                          np.load(C.TRAIN_GIDS).astype(np.int64)):
        raise RuntimeError("train cache gids are not the frozen 1M plan")
    rect_rows = np.asarray(plan["rect_cache_row"], np.int64)
    plain_rows = np.asarray(plan["plain_cache_row"], np.int64)
    if rect_rows.min() < 0 or rect_rows.max() >= int(task["train_cache"]["n_molecules"]):
        raise RuntimeError("rectangle cache row outside the 1M cache")
    if plain_rows.min() < 0 or plain_rows.max() >= int(task["train_cache"]["n_molecules"]):
        raise RuntimeError("plain cache row outside the 1M cache")
    train_y = np.asarray(task["train_y_eV"], np.float64)
    if not np.array_equal(np.asarray(plan["rect_target_eV"], np.float64),
                          train_y[rect_rows]):
        raise RuntimeError("rectangle target labels do not match the frozen labels")
    if not np.array_equal(np.asarray(plan["plain_target_eV"], np.float64),
                          train_y[plain_rows]):
        raise RuntimeError("plain target labels do not match the frozen labels")
    if not np.array_equal(np.asarray(plan["rect_target_z"], np.float64),
                          C.to_z(plan["rect_target_eV"], built["label_norm"])):
        raise RuntimeError("rectangle z labels do not match the frozen label normalisation")
    if not np.array_equal(np.asarray(plan["plain_target_z"], np.float64),
                          C.to_z(plan["plain_target_eV"], built["label_norm"])):
        raise RuntimeError("plain z labels do not match the frozen label normalisation")
    train_unique_rows = np.unique(np.concatenate([rect_rows.ravel(), plain_rows]))
    if train_unique_rows.size != int(manifest["unique_rectangle_endpoint_identities"]) + int(manifest["plain_molecules"]):
        # The manifest counts endpoint identities and plain molecules separately;
        # they are disjoint, so the unique row count must be their sum.
        raise RuntimeError(
            f"unique Train subset {train_unique_rows.size} != "
            f"{manifest['unique_rectangle_endpoint_identities']} + {manifest['plain_molecules']}"
        )

    steps_per_epoch = int(math.ceil(int(plan["rect_cache_row"].shape[0]) / 16))
    total_steps = int(epochs * steps_per_epoch)
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))
    t_start = time.perf_counter()
    curve: list[dict[str, Any]] = []
    epoch_preds: list[np.ndarray] = []
    epoch_train_eV: list[np.ndarray] = []

    def current_seconds() -> float:
        return float(time.perf_counter() - t_start)

    def check_budget() -> None:
        if current_seconds() > max_seconds:
            raise TimeoutError(f"wall budget {max_seconds:.1f}s exceeded")

    def evaluate(epoch: int, global_step: int, train_loss: float | None,
                 train_aux: float | None) -> dict[str, Any]:
        model.eval()
        valid, pred_z = evaluate_valid(model, EC, task, plan, device, EM,
                                       built["label_norm"], task["evaluator"],
                                       limit=args.valid_limit)
        train_sub = evaluate_train_subset(model, EC, task, plan, device, EM,
                                          built["label_norm"], train_unique_rows)
        peak_alloc = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        peak_reserved = int(torch.cuda.max_memory_reserved()) if device.startswith("cuda") else 0
        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": None if train_loss is None else float(train_loss),
            "train_aux": None if train_aux is None else float(train_aux),
            "valid_mae_eV": float(valid["mae_eV"]),
            "valid_mae_eV_numpy": float(valid["mae_eV_numpy"]),
            "valid_crosscheck_delta_eV": float(valid["crosscheck_delta_eV"]),
            "valid_n": int(valid["n"]),
            "valid_pred_mean_eV": float(valid["pred_mean_eV"]),
            "valid_pred_std_eV": float(valid["pred_std_eV"]),
            "valid_seconds": float(valid["inference_seconds"]),
            "train_subset_n": int(train_sub["n_unique_molecules"]),
            "train_subset_mae_eV": float(train_sub["mae_eV"]),
            "train_subset_l1_z": float(train_sub["l1_z"]),
            "train_subset_seconds": float(train_sub["inference_seconds"]),
            "cuda_peak_allocated_bytes": peak_alloc,
            "cuda_peak_reserved_bytes": peak_reserved,
            "elapsed_seconds": current_seconds(),
        }
        epoch_preds.append(np.asarray(pred_z, np.float32))
        epoch_train_eV.append(np.asarray(train_sub["pred_eV"], np.float32))
        _log(f"arm {arm} epoch {epoch}: valid MAE {row['valid_mae_eV']:.9f} eV "
             f"(n={row['valid_n']}), train-subset {row['train_subset_mae_eV']:.9f} eV, "
             f"elapsed {row['elapsed_seconds']:.1f}s")
        return row

    _log(f"epoch 0 evaluation (original W3 checkpoint), plan {plan_sha[:16]}...")
    row0 = evaluate(0, 0, None, None)
    curve.append(row0)
    best_epoch = 0
    best_step = 0
    best_mae = float(row0["valid_mae_eV"])
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_mae = best_mae
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_epoch = 0
    final_step = 0

    global_step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        train_loss_sum = 0.0
        aux_sum = 0.0
        n_steps = 0
        n_graphs = 0
        n_unique = 0
        step_ms: list[float] = []
        rect_order = np.asarray(plan["rect_order"][epoch - 1], np.int64)
        plain_order = np.asarray(plan["plain_order"][epoch - 1], np.int64)
        for s in range(steps_per_epoch):
            if global_step >= total_steps:
                break
            check_budget()
            r_lo = s * 16
            p_lo = s * 64
            rect_idx = rect_order[r_lo:r_lo + 16]
            plain_idx = plain_order[p_lo:p_lo + 64]
            if rect_idx.size == 0:
                break
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_step = time.perf_counter()
            terms = train_batch(model, EC, task["train_cache"], plan, rect_idx,
                                plain_idx, device, EM)
            loss = arm_loss(cfg, arm, terms)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step}")
            lr = set_lr(opt, step=global_step, total_steps=total_steps,
                        start=float(cfg["optim"]["lr_start"]),
                        end=float(cfg["optim"]["lr_end"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["optim"]["gradient_clip"])))
            if not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite gradient norm at step {global_step}")
            opt.step()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = float(time.perf_counter() - t_step)
            train_loss_sum += float(loss.detach())
            aux = float((terms["aux_pair"] if arm == "PAIR" else
                         terms["aux_rect"] if arm == "RECT" else terms["loss_abs"] * 0).detach())
            aux_sum += aux
            n_steps += 1
            n_graphs += int(terms["n_graphs"])
            n_unique += int(terms["n_unique"])
            step_ms.append(dt * 1000.0)
            global_step += 1
            if global_step % 50 == 0:
                _log(f"arm {arm} step {global_step}/{total_steps} "
                     f"loss {train_loss_sum / n_steps:.6f} lr {lr:.3e} "
                     f"{dt * 1000:.1f} ms")
            if global_step >= total_steps:
                break
        if n_steps == 0:
            break
        epoch_train_seconds = float(time.perf_counter() - epoch_t0)
        model.eval()
        row = evaluate(epoch, global_step, train_loss_sum / n_steps, aux_sum / n_steps)
        row.update({
            "epoch_train_seconds": epoch_train_seconds,
            "steps": int(n_steps),
            "graphs_seen": int(n_graphs),
            "unique_graphs_seen": int(n_unique),
            "mean_unique_per_batch": float(n_unique / max(n_steps, 1)),
            "step_ms_mean": float(np.mean(step_ms)),
            "step_ms_p50": float(np.percentile(step_ms, 50)),
            "step_ms_p90": float(np.percentile(step_ms, 90)),
            "learning_rate": float(lr),
        })
        curve.append(row)
        final_mae = float(row["valid_mae_eV"])
        if final_mae < best_mae - 1e-12:
            best_mae = final_mae
            best_epoch = int(epoch)
            best_step = int(global_step)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        final_epoch = int(epoch)
        final_step = int(global_step)
        model.train()
        if global_step >= total_steps:
            break

    seconds = current_seconds()
    # Save the final checkpoint, then the best checkpoint (which may be epoch 0
    # = the original W3 weights).
    model.load_state_dict(final_state, strict=True)
    save_checkpoint(out_dir / "checkpoint_final.pt", model, arm=arm,
                    epoch=final_epoch, global_step=final_step, valid_mae=final_mae,
                    best_epoch=best_epoch, best_valid_mae=best_mae, final_valid_mae=final_mae,
                    plan_sha=plan_sha, config_sha=config_sha,
                    source_sha=str(cfg["checkpoint"]["sha256"]),
                    initial_sha=initial_sha, cfg=cfg, training_seconds=seconds,
                    gpu_hours=seconds / 3600.0)
    model.load_state_dict(best_state, strict=True)
    save_checkpoint(out_dir / "checkpoint_best_valid.pt", model, arm=arm,
                    epoch=best_epoch, global_step=best_step, valid_mae=best_mae,
                    best_epoch=best_epoch, best_valid_mae=best_mae, final_valid_mae=final_mae,
                    plan_sha=plan_sha, config_sha=config_sha,
                    source_sha=str(cfg["checkpoint"]["sha256"]),
                    initial_sha=initial_sha, cfg=cfg, training_seconds=seconds,
                    gpu_hours=seconds / 3600.0)

    epochs_eval = np.asarray([int(r["epoch"]) for r in curve], np.int64)
    valid_pred_z = np.stack(epoch_preds, axis=0).astype(np.float32)
    valid_pred_eV = C.from_z(valid_pred_z.astype(np.float64), built["label_norm"]).astype(np.float32)
    if len(epoch_train_eV) == len(curve):
        train_pred_eV = np.stack(epoch_train_eV, axis=0).astype(np.float32)
    else:  # defensive: should not happen
        train_pred_eV = np.zeros((len(curve), train_unique_rows.size), np.float32)
    valid_rows_used = np.arange(int(task["valid_cache"]["n_molecules"]), dtype=np.int64)
    if args.valid_limit is not None and int(args.valid_limit) < valid_rows_used.size:
        valid_rows_used = valid_rows_used[: int(args.valid_limit)]
    np.savez_compressed(
        out_dir / "predictions_epochs.npz",
        epochs=epochs_eval,
        valid_rows=valid_rows_used,
        valid_gids=np.asarray(task["valid_cache"]["gids"], np.int64)[valid_rows_used],
        valid_target_eV=np.asarray(task["valid_y_eV"], np.float64)[valid_rows_used],
        valid_pred_z=valid_pred_z,
        valid_pred_eV=valid_pred_eV,
        train_subset_rows=train_unique_rows.astype(np.int64),
        train_subset_target_eV=np.asarray(task["train_y_eV"], np.float64)[train_unique_rows],
        train_subset_pred_eV=train_pred_eV,
        best_epoch=np.asarray(best_epoch, np.int64),
        final_epoch=np.asarray(final_epoch, np.int64),
    )

    env = {
        "hostname": os.uname().nodename,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "cuda_capability": (list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None),
        "numpy": str(np.__version__),
        "device": device,
        "torch_threads": int(torch.get_num_threads()),
    }
    run = {
        "schema": "ssrl-v0-run",
        "round": cfg["round"],
        "arm": arm,
        "status": "OK",
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "config": C.track_rel(args.config),
        "config_sha256": config_sha,
        "plan": C.track_rel(args.plan),
        "plan_sha256": plan_sha,
        "batch_plan_sha256": manifest["batch_plan_sha256"],
        "rectangles": int(plan["rect_cache_row"].shape[0]),
        "plain_molecules": int(plan["plain_cache_row"].size),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs": int(epochs),
        "total_steps": int(total_steps),
        "best_epoch": int(best_epoch),
        "best_global_step": int(best_step),
        "best_valid_mae_eV": float(best_mae),
        "final_epoch": int(final_epoch),
        "final_global_step": int(final_step),
        "final_valid_mae_eV": float(final_mae),
        "epoch0_valid_mae_eV": float(row0["valid_mae_eV"]),
        "initial_state_sha256": initial_sha,
        "source_checkpoint": C.track_rel(C.LCW_CHECKPOINT),
        "source_checkpoint_sha256": str(cfg["checkpoint"]["sha256"]),
        "source_checkpoint_epoch": 39,
        "source_checkpoint_valid_mae_eV": 0.12780916721022834,
        "optimizer": opt_audit,
        "loss": {
            "name": arm,
            "formula": cfg["loss"][arm],
            "coefficients": cfg["loss"]["coefficients"],
            "units": cfg["loss"]["units"],
        },
        "seconds": float(seconds),
        "gpu_hours_process": float(seconds / 3600.0),
        "max_seconds": float(max_seconds),
        "curve": curve,
        "environment": env,
        "test_evaluated": False,
        "test_policy": "official Test labels are hidden and never indexed",
        "notes": [
            "fresh AdamW state; this is not a seamless resume of the original W3 trajectory",
            "the model computation, features, normalisation and parameter count are unchanged",
            "only the training auxiliary term differs between arms",
        ],
    }
    C.write_json(out_dir / "run.json", run)
    C.write_csv(out_dir / "train_curve.csv", curve,
                columns=["epoch", "global_step", "learning_rate", "train_loss",
                         "train_aux", "valid_mae_eV", "valid_mae_eV_numpy",
                         "valid_crosscheck_delta_eV", "valid_n", "valid_pred_mean_eV",
                         "valid_pred_std_eV", "valid_seconds", "train_subset_n",
                         "train_subset_mae_eV", "train_subset_l1_z", "train_subset_seconds",
                         "epoch_train_seconds", "steps", "graphs_seen", "unique_graphs_seen",
                         "mean_unique_per_batch", "step_ms_mean", "step_ms_p50",
                         "step_ms_p90", "cuda_peak_allocated_bytes",
                         "cuda_peak_reserved_bytes", "elapsed_seconds"])
    if args.preflight:
        C.write_json(out_dir / "preflight.json", {
            "schema": "ssrl-v0-preflight",
            "arm": arm,
            "epochs_measured": int(len(curve) - 1),
            "steps_measured": int(total_steps),
            "epoch0_valid_mae_eV": float(row0["valid_mae_eV"]),
            "last_valid_mae_eV": float(curve[-1]["valid_mae_eV"]),
            "seconds": float(seconds),
            "gpu_hours_process": float(seconds / 3600.0),
            "step_ms_mean": float(np.mean([r["step_ms_mean"] for r in curve[1:] if "step_ms_mean" in r])) if len(curve) > 1 else None,
            "valid_seconds": float(curve[-1]["valid_seconds"]),
            "note": "timing only; intermediate predictions do not change the horizon",
        })
    _log(f"arm {arm} done: best epoch {best_epoch} valid {best_mae:.9f} eV; "
         f"final epoch {final_epoch} valid {final_mae:.9f} eV; "
         f"{seconds:.1f}s process ({seconds / 3600.0:.4f} GPU-h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
