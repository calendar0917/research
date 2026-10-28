"""Training / valid evaluation for LCRC-v1.

The recipe is frozen to the historical R0 rules used by DCC-L:

    AdamW, peak LR 1e-3, min LR 1e-5, per-step cosine after a 500-update
    linear warm-up, weight decay 1e-5, batch 128, 600 epochs (79 steps/epoch,
    47,400 updates), raw-y L1, gradient clip 5, no dropout, no EMA, no soup.
    Full official valid every epoch; the single checkpoint with the strictly
    smallest valid MAE is selected (ties keep the earlier epoch).

Data order is ``torch.randperm(10000, generator=Generator().manual_seed(seed+101))``
reshuffled once per epoch, identical for A/R/N.  The last partial batch is kept.
Official test is never constructed.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import lc_common as L
import lc_data as D
import lc_models as M

RECIPE = {
    "optimizer": "adamw",
    "lr": 1.0e-3,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-5,
    "warmup_steps": 500,
    "epochs": 600,
    "batch_size": 128,
    "valid_batch_size": 256,
    "dropout": 0.0,
    "grad_clip": 5.0,
    "loss": "l1",
}


def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return L.RUNS / f"lcrc{arm}_{prefix}s{int(seed)}"


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    rows: Sequence[int],
    *,
    device,
    batch_size: int = 256,
    return_aux: bool = False,
):
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    preds, aux_parts = [], {}
    with L.C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = _to_device(D.collate(cache, chunk), device)
            if return_aux:
                pred, aux = model(batch, return_aux=True)
                for k, v in aux.items():
                    if torch.is_tensor(v):
                        aux_parts.setdefault(k, []).append(v.detach().float().cpu())
            else:
                pred = model(batch)
            preds.append(pred.detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(preds, 0) if preds else np.zeros(0, dtype=np.float32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    mae = float(np.mean(np.abs(pred.astype(np.float64) - y))) if len(y) else float("nan")
    if not return_aux:
        return pred, mae
    aux = {k: (torch.cat(v, 0).numpy() if v else np.zeros(0)) for k, v in aux_parts.items()}
    return pred, mae, aux


def _param_audit(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    total = L.C.parameter_count(model)
    opt = L.C.optimizer_parameter_count(optimizer)
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
    vals = [float(c["valid_mae"]) for c in curve]
    if not vals:
        return {"W20": None, "T100": None}
    w20 = None
    if len(vals) >= 20:
        w20 = min(float(np.median(vals[i:i + 20])) for i in range(len(vals) - 19))
    return {"W20": w20, "T100": float(np.median(vals[-min(100, len(vals)):]))}


def _save_checkpoint(path: Path, model, *, arm, seed, epoch, valid_mae, extra=None) -> str:
    payload = {
        "state_dict": L.C.cpu_state(model),
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "valid_mae": float(valid_mae),
        "config": model.cfg,
    }
    if extra:
        payload.update(extra)
    L.atomic_save(payload, path)
    return L.C.sha256_file(path)


def train(arm: str, seed: int, device: str, *, epochs: int | None = None,
          tag: str | None = None, force: bool = False, out_dir: Path | None = None,
          timing: bool = False) -> dict:
    L.require_cuda(device)
    out = Path(out_dir) if out_dir is not None else run_dir(arm, seed, tag)
    L.ensure_dir(out)
    train_json = out / "train.json"
    if train_json.exists() and not force:
        L.log(f"[{arm} s{seed}] already complete: {train_json}")
        return L.read_json(train_json)

    norm = D.load_norm()
    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    if train_cache["meta"]["feature_version"] != L.FEATURE_VERSION:
        raise RuntimeError("train cache feature version mismatch")
    if valid_cache["meta"]["feature_version"] != L.FEATURE_VERSION:
        raise RuntimeError("valid cache feature version mismatch")
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)

    cfg = dict(M.DEFAULT_CONFIG)
    model, model_info = M.build_model(arm, norm, seed, device, cfg)
    recipe = dict(RECIPE)
    n_epochs = int(recipe["epochs"] if epochs is None else epochs)
    if recipe["optimizer"] != "adamw":
        raise RuntimeError("frozen recipe is AdamW")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
    )
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
    started = time.perf_counter()
    global_step = 0
    last_lr = float(recipe["lr"])
    first_batch_rows = None
    first_batch_grad = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = rows_train[order[start:start + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = rows.copy()
            batch = _to_device(D.collate(train_cache, rows), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if first_batch_grad is None:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                first_batch_grad = {
                    "grad_tensors": len(grads),
                    "nonzero_grad_tensors": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
                    "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "rel_head_grad_nonzero": (
                        None if model.rel is None
                        else bool(torch.count_nonzero(model.rel.head.weight.grad).item() > 0)
                    ),
                }
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            global_step += 1
            last_lr = L.C.cosine_with_warmup_lr(
                global_step, total_updates, warmup, float(recipe["lr"]), float(recipe["min_lr"])
            )
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)
        _, valid_mae = evaluate(model, valid_cache, rows_valid, device=device,
                                batch_size=int(recipe["valid_batch_size"]))
        sec = time.perf_counter() - t_ep
        entry = {
            "epoch": int(epoch),
            "online_train_mae": float(online),
            "valid_mae": float(valid_mae),
            "lr": float(last_lr),
            "seconds": float(sec),
            "steps": int(steps_per_epoch),
        }
        curve.append(entry)
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            best_state = L.C.cpu_state(model)
            best_sha = _save_checkpoint(out / "best.pt", model, arm=arm, seed=seed,
                                        epoch=epoch, valid_mae=valid_mae)
        L.log(f"[{arm} s{seed}] ep={epoch:03d} valid={valid_mae:.6f} online={online:.6f} "
              f"lr={last_lr:.3e} best={best['valid_mae']:.6f}@{best['epoch']} {sec:.2f}s")
        if timing and epoch >= n_epochs:
            break

    wall = time.perf_counter() - started
    final_sha = _save_checkpoint(out / "final.pt", model, arm=arm, seed=seed,
                                 epoch=n_epochs, valid_mae=curve[-1]["valid_mae"])
    L.C.atomic_torch_save(
        {
            "state_dict": L.C.cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "gen_state": gen.get_state(),
            "epoch": n_epochs,
            "global_step": global_step,
            "best": best,
            "curve": curve,
            "arm": arm,
            "seed": int(seed),
        },
        out / "final_state.pt",
    )

    # Fixed-checkpoint eval-mode measurements.
    best_model = M.load_model_from_state(arm, norm, best_state, device, cfg)
    best_pred_valid, best_valid = evaluate(best_model, valid_cache, rows_valid, device=device,
                                           batch_size=int(recipe["valid_batch_size"]))
    _, best_train = evaluate(best_model, train_cache, rows_train, device=device,
                             batch_size=int(recipe["valid_batch_size"]))
    final_pred_valid, final_valid = evaluate(model, valid_cache, rows_valid, device=device,
                                             batch_size=int(recipe["valid_batch_size"]))
    _, final_train = evaluate(model, train_cache, rows_train, device=device,
                              batch_size=int(recipe["valid_batch_size"]))
    if abs(best_valid - best["valid_mae"]) > 1e-6:
        raise RuntimeError(f"best checkpoint replay mismatch: {best_valid} vs {best['valid_mae']}")
    if abs(final_valid - curve[-1]["valid_mae"]) > 1e-6:
        raise RuntimeError(f"final checkpoint replay mismatch: {final_valid} vs {curve[-1]['valid_mae']}")

    pred_dir = out / "predictions"
    L.ensure_dir(pred_dir)
    np.savez(pred_dir / "valid_best.npz", row=rows_valid, pred=best_pred_valid)
    np.savez(pred_dir / "valid_final.npz", row=rows_valid, pred=final_pred_valid)
    L.C.write_curve_csv(out / "curve.csv", curve)
    diag = _window_diagnostics(curve)
    gpu_hours = wall / 3600.0
    payload = {
        "round": L.ROUND_ID,
        "candidate": f"LCRC-{arm}",
        "arm": arm,
        "seed": int(seed),
        "status": "completed",
        "timing_run": bool(timing),
        "git_commit": L.C.git_commit(),
        "environment": L.env_summary(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(rows_valid)),
            "official_test_loaded": False,
            "train_cache_file_sha256": L.C.sha256_file(D.cache_path("train")),
            "valid_cache_file_sha256": L.C.sha256_file(D.cache_path("valid")),
            "train_cache_content_sha256": D.content_fingerprint(train_cache),
            "valid_cache_content_sha256": D.content_fingerprint(valid_cache),
            "train_label_sha256": L.C.sha256_array(y_train),
            "valid_label_sha256": L.C.sha256_array(valid_cache["labels"]),
            "feature_version": L.FEATURE_VERSION,
        },
        "model_info": model_info,
        "recipe": recipe,
        "warmup_steps": int(warmup),
        "steps_per_epoch": int(steps_per_epoch),
        "total_updates": int(total_updates),
        "epochs": int(n_epochs),
        "param_audit": audit,
        "gradient_audit_first_batch": first_batch_grad,
        "first_batch_rows": None if first_batch_rows is None else first_batch_rows.tolist(),
        "data_order_seed": int(seed) + 101,
        "norm": {"source": norm["source"], "fingerprint": D.norm_fingerprint(norm)},
        "best": best,
        "final_valid_mae": float(final_valid),
        "best_valid_mae_eval": float(best_valid),
        "best_train_mae_eval": float(best_train),
        "final_train_mae_eval": float(final_train),
        "W20": diag["W20"],
        "T100": diag["T100"],
        "wall_seconds": float(wall),
        "gpu_hours_estimate": float(gpu_hours),
        "checkpoint_sha256": {"best": best_sha, "final": final_sha},
        "curve": curve,
    }
    L.write_json(train_json, payload)
    L.log(f"[{arm} s{seed}] done best={best['valid_mae']:.6f}@{best['epoch']} "
          f"final={final_valid:.6f} wall={wall:.1f}s")
    return payload


def timing(arms: Sequence[str], seed: int, device: str, epochs: int = 2) -> dict:
    L.require_cuda(device)
    out = {}
    for arm in arms:
        cfg = dict(M.DEFAULT_CONFIG)
        norm = D.load_norm()
        train_cache = D.load_cache("train")
        valid_cache = D.load_cache("valid")
        model, info = M.build_model(arm, norm, seed, device, cfg)
        recipe = dict(RECIPE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
        batch_size = int(recipe["batch_size"])
        steps = math.ceil(len(train_cache["labels"]) / batch_size)
        gen = torch.Generator().manual_seed(int(seed) + 101)
        rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)
        epoch_seconds = []
        total_updates = steps * int(RECIPE["epochs"])
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            order = torch.randperm(len(train_cache["labels"]), generator=gen).numpy()
            for start in range(0, len(order), batch_size):
                rows = order[start:start + batch_size]
                batch = _to_device(D.collate(train_cache, rows), device)
                target = torch.as_tensor(train_cache["labels"][rows], dtype=torch.float32, device=device)
                pred = model(batch)
                loss = F.l1_loss(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
                optimizer.step()
            _, valid_mae = evaluate(model, valid_cache, rows_valid, device=device, batch_size=256)
            epoch_seconds.append(time.perf_counter() - t0)
            L.log(f"[timing {arm}] epoch {epoch} {epoch_seconds[-1]:.2f}s valid={valid_mae:.6f}")
        steady = float(np.mean(epoch_seconds[1:])) if len(epoch_seconds) > 1 else float(epoch_seconds[0])
        projected = (600 * steady * 1.15 + 60.0) / 3600.0
        out[arm] = {
            "seed": int(seed),
            "parameters": info["parameters"],
            "epoch_seconds": epoch_seconds,
            "steady_seconds_per_epoch": steady,
            "steps_per_epoch": int(steps),
            "total_updates_600ep": int(total_updates),
            "projected_gpu_hours": projected,
        }
    path = L.ANALYSIS / "timing.json"
    L.write_json(path, {"recipe": RECIPE, "runs": out, "created": L.now()})
    L.log(f"timing written to {L.track_rel(path)}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LCRC-v1 prep/train/timing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prep")
    p_prep.add_argument("--splits", default="train,valid")
    p_prep.add_argument("--force", action="store_true")

    p_train = sub.add_parser("train")
    p_train.add_argument("--arm", required=True, choices=["A", "R", "N"])
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--device", default="cuda:0")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--tag", default=None)
    p_train.add_argument("--out", default=None)
    p_train.add_argument("--force", action="store_true")

    p_time = sub.add_parser("timing")
    p_time.add_argument("--arms", default="A,R")
    p_time.add_argument("--seed", type=int, default=0)
    p_time.add_argument("--device", default="cuda:0")
    p_time.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args(argv)
    if args.cmd == "prep":
        res = D.prep(args.splits.split(","), force=bool(args.force))
        L.log(str(res))
        return 0
    if args.cmd == "train":
        res = train(args.arm, args.seed, args.device, epochs=args.epochs,
                    tag=args.tag, force=bool(args.force),
                    out_dir=None if args.out is None else Path(args.out))
        L.log(f"best valid {res['best']['valid_mae']:.6f} @ {res['best']['epoch']}")
        return 0
    if args.cmd == "timing":
        res = timing(args.arms.split(","), args.seed, args.device, args.epochs)
        for arm, v in res.items():
            L.log(f"{arm}: {v['steady_seconds_per_epoch']:.2f}s/epoch, "
                  f"projected {v['projected_gpu_hours']:.3f} GPU-h")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
