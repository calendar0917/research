"""Formal trainer for LCRC-HC3-v1 (B / H / F).

Frozen recipe (identical to the historical DCC-L / LCRC-A rounds):

    AdamW, peak LR 1e-3, min LR 1e-5, per-update cosine after a 500-update
    linear warm-up, weight decay 1e-5, batch 128, raw-y L1, clip 5, no dropout,
    exactly 47,400 optimizer updates.  Full Monitor/valid every epoch; the
    checkpoint with the strictly smallest monitored MAE is kept (ties keep the
    earlier epoch).  Data order is ``randperm`` from ``Generator(seed+101)``,
    reshuffled once per epoch and identical across arms.

Fit8k stages train only on the frozen Fit8k rows and select on Monitor; the
train10k stage trains on all official train rows and selects on official valid.
Official test is never constructed.

Resume granularity is one epoch: a saved ``last_state.pt`` contains the model,
optimizer, RNG generator state, current epoch, global step, curve and best
record.  Re-running after an interruption re-enters the same epoch with the
same permutation, so an uninterrupted and a resumed run are identical.
"""

from __future__ import annotations

import argparse
import math
import platform
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import hc_common as C
import hc_data as D
import hc_models as M

FITS = ("fit8k", "train10k")


def run_dir(kind: str, seed: int, fit: str, *, timing: bool = False,
            tag: str | None = None) -> Path:
    if tag:
        return C.RUNS / tag
    name = f"{kind}_s{int(seed)}_{fit}"
    if timing:
        name += "_timing"
    return C.RUNS / name


def _grad_norms(model: torch.nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = ".".join(name.split(".")[:2])
        out[key] = out.get(key, 0.0) + float(param.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(out.items())}


def _param_audit(model, optimizer) -> dict:
    total = C.parameter_count(model)
    opt = int(sum(p.numel() for group in optimizer.param_groups for p in group["params"]))
    requires = sum(1 for p in model.parameters() if p.requires_grad)
    nparams = sum(1 for p in model.parameters())
    return {
        "parameters": total,
        "optimizer_parameters": opt,
        "parameters_in_optimizer": bool(total == opt),
        "parameter_tensors": nparams,
        "requires_grad_tensors": requires,
        "all_parameters_require_grad": bool(requires == nparams),
    }


def _window_diagnostics(curve: Sequence[Mapping[str, Any]]) -> dict:
    vals = [float(c["monitor_mae"]) for c in curve]
    if not vals:
        return {"W20": None, "T100": None, "W20_epoch": None, "T100_epochs": 0}
    w20, w20_ep = None, None
    if len(vals) >= 20:
        for end in range(20, len(vals) + 1):
            med = float(np.median(vals[end - 20:end]))
            if w20 is None or med < w20:
                w20, w20_ep = med, end
    tail = vals[-min(100, len(vals)):]
    return {"W20": w20, "T100": float(np.median(tail)), "W20_epoch": w20_ep,
            "T100_epochs": int(len(tail))}


def _save_checkpoint(path: Path, model, *, kind, seed, fit, epoch, step,
                     monitor_mae, selection) -> str:
    payload = {
        "state_dict": C.cpu_state(model),
        "kind": kind,
        "arm": kind,
        "seed": int(seed),
        "fit": fit,
        "epoch": int(epoch),
        "step": int(step),
        "monitor_mae": float(monitor_mae),
        "selection": selection,
    }
    C.atomic_torch_save(payload, path)
    return C.sha256_file(path)


def _save_last(path: Path, model, optimizer, gen, *, epoch, global_step, best,
               curve, kind, seed, fit) -> None:
    C.atomic_torch_save({
        "state_dict": C.cpu_state(model),
        "optimizer": optimizer.state_dict(),
        "gen_state": gen.get_state(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best": dict(best),
        "curve": list(curve),
        "kind": kind,
        "seed": int(seed),
        "fit": fit,
    }, path)


def _monitor_setup(fit: str):
    split = C.split_rows()
    C.verify_split_hashes(split["rows"])
    if fit == "fit8k":
        train_rows = split["rows"]["fit8k"]
        mon_rows = split["rows"]["monitor"]
        train_cache = C.load_cache("train")
        mon_cache = train_cache
        norm = C.fit8k_norm(train_cache, train_rows)
        mon_split = "monitor"
    elif fit == "train10k":
        train_cache = C.load_cache("train")
        train_rows = np.arange(int(train_cache["n_molecules"]), dtype=np.int64)
        mon_cache = C.load_cache("valid")
        mon_rows = np.arange(int(mon_cache["n_molecules"]), dtype=np.int64)
        norm = C.load_full_norm()
        mon_split = "valid"
    else:
        raise ValueError(fit)
    return {
        "split": split,
        "train_cache": train_cache,
        "mon_cache": mon_cache,
        "train_rows": train_rows,
        "mon_rows": mon_rows,
        "norm": norm,
        "monitor_split": mon_split,
    }


def train(kind: str, seed: int, fit: str, device: str, *, total_updates: int | None = None,
          timing_epochs: int | None = None, resume: bool = False, force: bool = False,
          tag: str | None = None, stop_after_steps: int | None = None) -> dict:
    if kind not in M.KINDS:
        raise ValueError(kind)
    if fit not in FITS:
        raise ValueError(fit)
    timing = timing_epochs is not None
    out = run_dir(kind, seed, fit, timing=timing, tag=tag)
    C.ensure_dir(out)
    if (out / "train.json").exists() and not force:
        C.log(f"[{kind} s{seed} {fit}] already complete: {out / 'train.json'}")
        return C.read_json(out / "train.json")

    setup = _monitor_setup(fit)
    split = setup["split"]
    train_cache = setup["train_cache"]
    norm = setup["norm"]
    train_rows = setup["train_rows"]
    mon_rows = setup["mon_rows"]
    mon_cache = setup["mon_cache"]
    mon_split = setup["monitor_split"]

    # The Monitor set may live in the train cache (fit8k) or the valid cache.
    train_ds = D.SplitData("train", cache=train_cache, device=device)
    mon_ds = train_ds if mon_cache is train_cache else D.SplitData("valid", cache=mon_cache,
                                                                 device=device)
    y_train = train_ds.targets(train_rows)
    y_mon = mon_ds.targets(mon_rows)

    model, model_info = M.build_model(kind, norm, seed, device)
    params = C.parameter_count(model)
    initial_hash = C.state_hash(model)
    recipe = dict(C.RECIPE)
    total_updates = int(recipe["total_updates"] if total_updates is None else total_updates)
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    n_epochs = math.ceil(total_updates / steps_per_epoch)
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    if timing:
        n_epochs = int(timing_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                                  weight_decay=recipe["weight_decay"])
    audit = _param_audit(model, optimizer)
    if not audit["parameters_in_optimizer"]:
        raise RuntimeError(f"optimizer does not cover all parameters: {audit}")

    gen = torch.Generator().manual_seed(int(seed) + 101)
    curve: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    global_step = 0
    start_epoch = 1
    last_lr = float(recipe["lr"])
    grad_log: list[dict] = []
    first_batch_rows = None
    resumed_from = None
    last_path = out / "last_state.pt"

    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume requested but {last_path} is missing")
        state = C.atomic_torch_load(last_path)
        C.load_state(model, state["state_dict"], device)
        optimizer.load_state_dict(state["optimizer"])
        gen.set_state(state["gen_state"].cpu() if torch.is_tensor(state["gen_state"])
                      else state["gen_state"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best = dict(state["best"])
        curve = list(state["curve"])
        last_lr = float(curve[-1]["lr"]) if curve else float(recipe["lr"])
        resumed_from = {
            "last_state_sha256": C.sha256_file(last_path),
            "epoch": int(state["epoch"]),
            "global_step": global_step,
            "best": best,
        }
        C.log(f"[{kind} s{seed} {fit}] resumed at epoch {start_epoch} step {global_step}")
    else:
        C.atomic_torch_save({"state_dict": C.cpu_state(model), "kind": kind,
                             "seed": int(seed), "fit": fit},
                            out / "initial_state.pt")

    started_at = C.now()
    t_start = time.perf_counter()
    epoch_seconds: list[float] = []

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= total_updates:
                break
            idx = order[start:start + batch_size]
            rows = train_rows[idx]
            if first_batch_rows is None:
                first_batch_rows = [int(x) for x in rows]
            batch = train_ds.batch(rows)
            target = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if global_step in (0, 1, 2, 3, 99):
                grad_log.append({"step": int(global_step), "norms": _grad_norms(model)})
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, total_updates, warmup,
                                              float(recipe["lr"]), float(recipe["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)

        mon_pred = D.predict(model, mon_ds, mon_rows, batch_size=int(recipe["eval_batch_size"]))
        mon_mae = D.mae(mon_pred, y_mon)
        sec = time.perf_counter() - t_ep
        epoch_seconds.append(sec)
        entry = {
            "epoch": int(epoch),
            "step": int(global_step),
            "online_train_mae": float(online),
            "monitor_mae": float(mon_mae),
            "lr": float(last_lr),
            "seconds": float(sec),
        }
        curve.append(entry)
        if mon_mae < best["monitor_mae"]:
            best = {"monitor_mae": float(mon_mae), "epoch": int(epoch), "step": int(global_step)}
            _save_checkpoint(out / "best.pt", model, kind=kind, seed=seed, fit=fit,
                             epoch=epoch, step=global_step, monitor_mae=mon_mae,
                             selection=mon_split)
        _save_last(last_path, model, optimizer, gen, epoch=epoch, global_step=global_step,
                   best=best, curve=curve, kind=kind, seed=seed, fit=fit)
        if epoch % 10 == 0 or global_step >= total_updates or timing:
            C.log(f"[{kind} s{seed} {fit}] ep={epoch} step={global_step} "
                  f"mon={mon_mae:.6f} online={online:.6f} best={best['monitor_mae']:.6f}"
                  f"@{best['epoch']} lr={last_lr:.2e} {sec:.2f}s")
        if global_step >= total_updates:
            break
        if stop_after_steps is not None and global_step >= int(stop_after_steps):
            break

    if stop_after_steps is not None and global_step < total_updates:
        return {
            "round": C.ROUND_ID,
            "kind": kind,
            "seed": int(seed),
            "fit": fit,
            "stopped_early": True,
            "global_step": int(global_step),
            "epoch": int(curve[-1]["epoch"]) if curve else 0,
            "total_updates": int(total_updates),
            "last_state_sha256": C.sha256_file(last_path),
        }

    wall = time.perf_counter() - t_start
    if timing:
        sec_per_epoch = float(np.mean(epoch_seconds)) if epoch_seconds else float("nan")
        full_epochs = math.ceil(int(recipe["total_updates"]) / steps_per_epoch)
        info = {
            "round": C.ROUND_ID,
            "kind": kind,
            "seed": int(seed),
            "fit": fit,
            "timing_run": True,
            "timing_epochs": int(timing_epochs),
            "updates_run": int(global_step),
            "steps_per_epoch": int(steps_per_epoch),
            "seconds": epoch_seconds,
            "mean_seconds_per_epoch": sec_per_epoch,
            "gpu_hours_measured": float(wall / 3600.0),
            "projected_full_epochs": int(full_epochs),
            "projected_full_updates": int(recipe["total_updates"]),
            "projected_gpu_hours": float(sec_per_epoch * full_epochs / 3600.0)
            if math.isfinite(sec_per_epoch) else None,
            "parameters": params,
            "parameter_breakdown": model_info.get("parameters_breakdown"),
            "device": str(device),
            "host": socket.gethostname(),
            "environment": C.environment(),
            "git_commit": C.git_commit(),
            "started_at": started_at,
            "finished_at": C.now(),
            "curve": curve,
        }
        C.write_json(out / "timing.json", info)
        C.log(f"[timing {kind} s{seed} {fit}] {sec_per_epoch:.2f}s/epoch "
              f"-> {info['projected_gpu_hours']:.3f} GPU*h projected")
        return info

    final_state_path = out / "final.pt"
    final_mon_mae = float(curve[-1]["monitor_mae"])
    final_sha = _save_checkpoint(final_state_path, model, kind=kind, seed=seed, fit=fit,
                                 epoch=int(curve[-1]["epoch"]), step=global_step,
                                 monitor_mae=final_mon_mae, selection="final")

    # Replay best/final in eval mode and persist per-molecule predictions.
    best_payload = C.atomic_torch_load(out / "best.pt")
    reads = {}
    for tag_name, payload in (("best", best_payload), ("final", C.atomic_torch_load(final_state_path))):
        C.load_state(model, payload["state_dict"], device)
        pred_mon = D.predict(model, mon_ds, mon_rows,
                             batch_size=int(recipe["eval_batch_size"]))
        pred_train = D.predict(model, train_ds, train_rows,
                               batch_size=int(recipe["eval_batch_size"]))
        reads[tag_name] = {
            "monitor_mae": D.mae(pred_mon, y_mon),
            "eval_train_mae": D.mae(pred_train, y_train),
            "state_hash": C.state_hash(model),
            "epoch": int(payload["epoch"]),
            "step": int(payload["step"]),
        }
        C.atomic_torch_save({
            "pred": np.asarray(pred_mon, dtype=np.float32),
            "rows": np.asarray(mon_rows, dtype=np.int64),
            "kind": kind, "seed": int(seed), "fit": fit,
            "checkpoint": tag_name, "monitor_split": mon_split,
        }, out / f"predictions_{tag_name}.pt")
        C.atomic_torch_save({
            "pred": np.asarray(pred_train, dtype=np.float32),
            "rows": np.asarray(train_rows, dtype=np.int64),
            "kind": kind, "seed": int(seed), "fit": fit,
            "checkpoint": tag_name,
        }, out / f"predictions_train_{tag_name}.pt")

    if abs(reads["best"]["monitor_mae"] - best["monitor_mae"]) > 1e-6:
        raise RuntimeError(f"best checkpoint replay mismatch: {reads['best']['monitor_mae']} "
                           f"vs {best['monitor_mae']}")
    if abs(reads["final"]["monitor_mae"] - final_mon_mae) > 1e-6:
        raise RuntimeError(f"final checkpoint replay mismatch: {reads['final']['monitor_mae']} "
                           f"vs {final_mon_mae}")

    data_info = {
        "source": "official PyG ZINC subset=True train/valid; official test never opened",
        "n_train": int(len(train_rows)),
        "n_monitor": int(len(mon_rows)),
        "monitor_split": mon_split,
        "official_valid_loaded": bool(fit == "train10k"),
        "official_test_loaded": False,
        "train_cache_file_sha256": C.sha256_file(C.cache_path("train")),
        "monitor_cache_file_sha256": C.sha256_file(C.cache_path("train" if fit == "fit8k" else "valid")),
        "train_label_sha256": C.sha256_array(y_train),
        "monitor_label_sha256": C.sha256_array(y_mon),
        "split_hashes": split["hashes"],
        "feature_version": C.LC.FEATURE_VERSION,
        "structure_version": C.STRUCTURE_VERSION,
        "structure_cache_sha256": {
            "train": C.sha256_file(C.CACHE / "hc3_train.pt"),
            **({"valid": C.sha256_file(C.CACHE / "hc3_valid.pt")} if fit == "train10k" else {}),
        },
    }
    diag = _window_diagnostics(curve)
    info = {
        "round": C.ROUND_ID,
        "kind": kind,
        "arm": kind,
        "seed": int(seed),
        "fit": fit,
        "monitor_split": mon_split,
        "timing_run": False,
        "status": "completed",
        "resumed_from": resumed_from,
        "parameter_audit": audit,
        "parameters": params,
        "parameter_breakdown": model_info.get("parameters_breakdown"),
        "model_info": model_info,
        "initial_state_sha256": initial_hash,
        "recipe": recipe,
        "total_updates": int(total_updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "warmup_steps": int(warmup),
        "data_order_seed": int(seed) + 101,
        "first_batch_rows": first_batch_rows,
        "gradient_log": grad_log,
        "norm": {
            "source": norm["source"],
            "fingerprint": C.norm_fingerprint(norm),
            "fit_split": norm.get("fit_split"),
        },
        "data": data_info,
        "best": best,
        "final_monitor_mae": final_mon_mae,
        "reads": reads,
        "W20": diag["W20"],
        "W20_epoch": diag["W20_epoch"],
        "T100": diag["T100"],
        "T100_epochs": diag["T100_epochs"],
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "mean_epoch_seconds": float(np.mean(epoch_seconds)) if epoch_seconds else None,
        "device": str(device),
        "host": socket.gethostname(),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "started_at": started_at,
        "finished_at": C.now(),
        "checkpoint_sha256": {
            "best": C.sha256_file(out / "best.pt"),
            "final": final_sha,
            "last_state": C.sha256_file(last_path),
        },
        "curve": curve,
    }
    C.write_json(out / "train.json", info)
    C.log(f"[{kind} s{seed} {fit}] DONE best={best['monitor_mae']:.6f}@{best['epoch']} "
          f"final={final_mon_mae:.6f} gpu_h={info['gpu_hours']:.3f}")
    return info


def timing(kinds: Sequence[str], seed: int, fit: str, device: str,
           epochs: int = 2) -> dict:
    out = {}
    for kind in kinds:
        info = train(kind, seed, fit, device, timing_epochs=epochs,
                     tag=run_dir(kind, seed, fit, timing=True).name, force=True)
        out[kind] = info
    payload = {"round": C.ROUND_ID, "fit": fit, "seed": int(seed),
               "timing_epochs": int(epochs), "runs": out,
               "total_projected_gpu_hours": float(sum(
                   v.get("projected_gpu_hours") or 0.0 for v in out.values())),
               "git_commit": C.git_commit(), "created": C.now()}
    C.write_json(C.ANALYSIS / f"timing_{fit}_s{seed}.json", payload)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=M.KINDS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fit", choices=FITS, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--timing", type=int, default=None,
                   help="run N epochs and report projected cost (not a formal run)")
    args = p.parse_args()
    if args.timing is not None:
        info = train(args.kind, args.seed, args.fit, args.device,
                     timing_epochs=args.timing, force=args.force)
    else:
        info = train(args.kind, args.seed, args.fit, args.device,
                     total_updates=args.total_updates, resume=args.resume, force=args.force)
    print({k: info.get(k) for k in ("kind", "seed", "fit", "parameters",
                                    "best", "final_monitor_mae", "gpu_hours")})


if __name__ == "__main__":
    main()
