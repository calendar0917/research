"""Train one non-local architecture arm on the frozen ZINC protocol.

Same R0 recipe as LCRC/DCC-L: AdamW, per-step cosine after 500 warmup updates,
lr 1e-3 -> 1e-5, wd 1e-5, clip 5, raw-y L1, batch 128, 600 epochs (47,400
updates), full official-valid every epoch, strict-argmin checkpoint.

Supports exact pause/resume: model, optimizer, epoch, global_step, batch-order
generator and the best-state are saved periodically and reloaded on --resume.
Official test is never constructed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import nl_bridge
import nl_common as N
import nl_data as ND
import nl_models as NM

RECIPE = {
    "optimizer": "adamw",
    "lr": 1.0e-3,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-5,
    "warmup_steps": 500,
    "epochs": 600,
    "batch_size": 128,
    "valid_batch_size": 256,
    "grad_clip": 5.0,
    "loss": "l1",
    "checkpoint_selection": "strict argmin official-valid MAE; ties -> earlier epoch",
}


def run_name(head: str, mode: str, seed: int, tag: str | None) -> str:
    prefix = f"{tag}_" if tag else ""
    return f"{prefix}{head}-{mode}_s{int(seed)}"


def _param_audit(model, optimizer):
    total = N.L.C.parameter_count(model)
    opt = N.L.C.optimizer_parameter_count(optimizer)
    return {
        "parameters": total,
        "optimizer_parameters": opt,
        "parameters_in_optimizer": bool(total == opt),
        "all_parameters_require_grad": all(p.requires_grad for p in model.parameters()),
    }


def _save(path: Path, model, *, info: dict, epoch: int, valid_mae: float, extra=None):
    payload = {
        "state_dict": N.L.C.cpu_state(model),
        "head": info["head"],
        "mode": info["mode"],
        "lambda": info["lambda"],
        "seed": info["seed"],
        "epoch": int(epoch),
        "valid_mae": float(valid_mae),
        "config": model.cfg,
    }
    if extra:
        payload.update(extra)
    N.L.atomic_save(payload, path)
    return N.L.C.sha256_file(path)


def _head_state(model, info):
    mod = model.sce if model.sce is not None else model.gef
    return {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}


def _grad_stats(model, info):
    out = {}
    mod = model.sce if model.sce is not None else model.gef
    for name, p in mod.named_parameters():
        g = p.grad
        out[name] = {
            "grad_norm": None if g is None else float(g.norm()),
            "nonzero": None if g is None else bool(torch.count_nonzero(g).item() > 0),
            "finite": None if g is None else bool(torch.isfinite(g).all().item()),
            "param_norm": float(p.detach().norm()),
        }
    for name in ("f.lins.0.weight", "rho.lins.0.weight"):
        p = dict(model.trunk.named_parameters()).get(name)
        if p is not None and p.grad is not None:
            out[f"trunk:{name}"] = {"grad_norm": float(p.grad.norm())}
    return out


def train(head: str, mode: str, seed: int, device: str, *, epochs: int | None = None,
          tag: str | None = None, lam: float = 0.5, force: bool = False,
          out_dir: Path | None = None, resume: bool = False, timing: bool = False,
          ckpt_every: int = 25) -> dict:
    N.require_cuda(device)
    name = run_name(head, mode, seed, tag)
    out = Path(out_dir) if out_dir is not None else N.RUNS / name
    N.ensure_dir(out)
    train_json = out / "train.json"
    if train_json.exists() and not force and not resume:
        N.L.log(f"[{name}] already complete: {train_json}")
        return N.L.read_json(train_json)

    norm = N.load_norm()
    train_cache = N.train_cache()
    valid_cache = N.valid_cache()
    for c in (train_cache, valid_cache):
        if c["meta"]["feature_version"] != N.L.FEATURE_VERSION:
            raise RuntimeError("cache feature version mismatch")
    bridges = {"train": nl_bridge.load("train"), "valid": nl_bridge.load("valid")} \
        if head == "sce" else {"train": None, "valid": None}
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)

    model, info = NM.build_model(head, norm, seed, device, None, mode=mode, lam=lam)
    recipe = dict(RECIPE)
    n_epochs = int(recipe["epochs"] if epochs is None else epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                                  weight_decay=recipe["weight_decay"])
    audit = _param_audit(model, optimizer)
    if not audit["parameters_in_optimizer"]:
        raise RuntimeError(f"optimizer does not cover all parameters: {audit}")

    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(len(rows_train) / batch_size)
    total_updates = steps_per_epoch * n_epochs
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    gen = torch.Generator().manual_seed(int(seed) + 101)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1}
    best_state = None
    best_sha = None
    global_step = 0
    start_epoch = 1
    wall_prev = 0.0
    rng_cpu = None
    step_logs: list[dict] = []
    probe_rows = np.arange(0, len(rows_valid), max(1, len(rows_valid) // 256))[:256]

    state_path = out / "state.pt"
    if resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in st["state_dict"].items()})
        optimizer.load_state_dict(st["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = float(st["last_lr"])
        gen.set_state(st["gen_state"])
        global_step = int(st["global_step"])
        start_epoch = int(st["epoch"]) + 1
        best = dict(st["best"])
        best_state = st.get("best_state")
        best_sha = st.get("best_sha")
        curve = list(st["curve"])
        step_logs = list(st.get("step_logs", []))
        wall_prev = float(st.get("wall_seconds", 0.0))
        rng_cpu = st.get("torch_rng")
        if rng_cpu is not None:
            torch.set_rng_state(rng_cpu)
        N.L.log(f"[{name}] resumed at epoch {start_epoch - 1}, step {global_step}")
    elif resume:
        N.L.log(f"[{name}] no state.pt; starting fresh")

    started = time.perf_counter()
    last_lr = float(recipe["lr"])
    first_batch_rows = None

    def save_state(epoch_done: int):
        N.L.atomic_save(
            {
                "state_dict": N.L.C.cpu_state(model),
                "optimizer": optimizer.state_dict(),
                "gen_state": gen.get_state(),
                "torch_rng": torch.get_rng_state(),
                "epoch": int(epoch_done),
                "global_step": int(global_step),
                "best": best,
                "best_state": best_state,
                "best_sha": best_sha,
                "curve": curve,
                "step_logs": step_logs,
                "last_lr": float(last_lr),
                "wall_seconds": float(wall_prev + time.perf_counter() - started),
            },
            state_path,
        )

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = rows_train[order[start:start + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = rows.copy()
            batch = ND.to_device(ND.collate(train_cache, bridges["train"], rows,
                                            with_bridge=(head == "sce")), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            global_step += 1
            if global_step <= 5:
                step_logs.append({
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "grads": _grad_stats(model, info),
                })
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            last_lr = N.L.C.cosine_with_warmup_lr(
                global_step, total_updates, warmup, float(recipe["lr"]), float(recipe["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)
        _, valid_mae = ND.evaluate(model, valid_cache, bridges["valid"], rows_valid,
                                   device=device, batch_size=int(recipe["valid_batch_size"]),
                                   with_bridge=(head == "sce"))
        sec = time.perf_counter() - t_ep
        entry = {"epoch": int(epoch), "online_train_mae": float(online),
                 "valid_mae": float(valid_mae), "lr": float(last_lr),
                 "seconds": float(sec), "steps": int(steps_per_epoch)}
        if epoch % 25 == 0 or epoch == 1:
            _, _, aux = ND.evaluate(model, valid_cache, bridges["valid"], probe_rows,
                                    device=device, batch_size=int(recipe["valid_batch_size"]),
                                    return_aux=True, with_bridge=(head == "sce"))
            entry["probe"] = ND.aux_stats(aux, len(probe_rows))
        curve.append(entry)
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            best_state = N.L.C.cpu_state(model)
            best_sha = _save(out / "best.pt", model, info=info, epoch=epoch, valid_mae=valid_mae)
        N.L.log(f"[{name}] ep={epoch:03d} valid={valid_mae:.6f} online={online:.6f} "
                f"lr={last_lr:.3e} best={best['valid_mae']:.6f}@{best['epoch']} {sec:.2f}s")
        if epoch % ckpt_every == 0 and epoch < n_epochs:
            save_state(epoch)
        if timing and epoch >= n_epochs:
            break

    wall = wall_prev + time.perf_counter() - started
    final_sha = _save(out / "final.pt", model, info=info, epoch=n_epochs,
                      valid_mae=curve[-1]["valid_mae"])
    N.L.C.atomic_torch_save(
        {"state_dict": N.L.C.cpu_state(model), "optimizer": optimizer.state_dict(),
         "gen_state": gen.get_state(), "epoch": n_epochs, "global_step": global_step,
         "best": best, "curve": curve, "head": head, "mode": mode, "seed": int(seed),
         "best_state": best_state, "best_sha": best_sha, "step_logs": step_logs},
        out / "final_state.pt",
    )

    # Frozen-checkpoint eval-mode measurements.
    best_model = NM.load_model(head, norm, best_state, device, None, mode=mode, lam=lam)
    best_pred_valid, best_valid = ND.evaluate(best_model, valid_cache, bridges["valid"], rows_valid,
                                              device=device, batch_size=int(recipe["valid_batch_size"]),
                                              with_bridge=(head == "sce"))
    _, best_train = ND.evaluate(best_model, train_cache, bridges["train"], rows_train,
                                device=device, batch_size=int(recipe["valid_batch_size"]),
                                with_bridge=(head == "sce"))
    final_pred_valid, final_valid = ND.evaluate(model, valid_cache, bridges["valid"], rows_valid,
                                                device=device, batch_size=int(recipe["valid_batch_size"]),
                                                with_bridge=(head == "sce"))
    _, final_train = ND.evaluate(model, train_cache, bridges["train"], rows_train,
                                 device=device, batch_size=int(recipe["valid_batch_size"]),
                                 with_bridge=(head == "sce"))
    if abs(best_valid - best["valid_mae"]) > 1e-6:
        raise RuntimeError(f"best replay mismatch: {best_valid} vs {best['valid_mae']}")
    if abs(final_valid - curve[-1]["valid_mae"]) > 1e-6:
        raise RuntimeError(f"final replay mismatch: {final_valid} vs {curve[-1]['valid_mae']}")

    pred_dir = N.ensure_dir(out / "predictions")
    np.savez(pred_dir / "valid_best.npz", row=rows_valid, pred=best_pred_valid)
    np.savez(pred_dir / "valid_final.npz", row=rows_valid, pred=final_pred_valid)
    N.L.C.write_curve_csv(out / "curve.csv", curve)
    vals = [c["valid_mae"] for c in curve]
    w20 = min(float(np.median(vals[i:i + 20])) for i in range(len(vals) - 19)) if len(vals) >= 20 else None
    payload = {
        "round": N.ROUND_ID,
        "run": name,
        "head": head,
        "mode": mode,
        "lambda": lam,
        "seed": int(seed),
        "status": "completed",
        "timing_run": bool(timing),
        "recipe": recipe,
        "model": info,
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(rows_valid)),
            "official_test_loaded": False,
            "train_cache_sha256": N.L.C.sha256_file(N.D.cache_path("train")),
            "valid_cache_sha256": N.L.C.sha256_file(N.D.cache_path("valid")),
            "norm_fingerprint": N.D.norm_fingerprint(norm),
        },
        "parameters": info["parameters"],
        "best": best,
        "best_valid_mae_eval": float(best_valid),
        "final_valid_mae_eval": float(final_valid),
        "best_train_mae_eval": float(best_train),
        "final_train_mae_eval": float(final_train),
        "W20": w20,
        "T100": float(np.median(vals[-min(100, len(vals)):])),
        "curve": curve,
        "step_logs": step_logs,
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "git_head": N.L.C.git_commit(),
        "environment": N.L.env_summary(),
        "best_checkpoint_sha256": best_sha,
        "final_checkpoint_sha256": final_sha,
    }
    N.L.write_json(train_json, payload)
    N.L.log(f"[{name}] DONE best={best_valid:.6f}@{best['epoch']} "
            f"T100={payload['T100']:.6f} {wall / 3600:.3f} GPU-h")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["sce", "gef"], required=True)
    ap.add_argument("--mode", default="real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    train(args.head, args.mode, args.seed, args.device, epochs=args.epochs, tag=args.tag,
          lam=args.lam, force=args.force, out_dir=Path(args.out) if args.out else None,
          resume=args.resume, timing=args.timing)


if __name__ == "__main__":
    main()
