"""Training / valid evaluation for ECLC-v1.

Frozen recipe (identical to LCRC-A / ZPR-v1):

    AdamW, LR 1e-3 -> 1e-5 with 500-update warm-up and per-step cosine,
    weight decay 1e-5, batch 128, 600 epochs = 79 steps/epoch = 47,400
    updates, raw-y L1, clip 5, dropout 0, no EMA/soup/seed averaging.
    Full official valid every epoch; strict minimum valid MAE single
    checkpoint (ties -> earlier epoch).

ECLC adds a fixed pair loss with coefficient 0.2 and 8 edit pairs per update.
The pair stream is generated only from the seed and is identical for REAL and
NULL.  The official test split is never constructed.
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

import eclc_common as E
import eclc_models as M2

D = E.D
L = E.L

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
    "loss": "l1_plus_pair_l1",
    "pair_loss_coef": E.PAIR_LOSS_COEF,
    "pairs_per_step": E.PAIRS_PER_STEP,
}


def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return E.RUNS / f"eclc{arm}_{prefix}s{int(seed)}"


def load_pair_arrays(path: Path | str = E.ANALYSIS / "edit_pairs.json") -> dict[str, Any]:
    payload = E.read_json(path)
    pairs = payload["pairs"]
    return {
        "payload": payload,
        "path": str(path),
        "sha256": E.sha256_file(path),
        "i": np.asarray([int(p["i"]) for p in pairs], dtype=np.int64),
        "j": np.asarray([int(p["j"]) for p in pairs], dtype=np.int64),
        "n": int(len(pairs)),
    }


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, cache: Mapping[str, Any], rows: Sequence[int], *,
             device, batch_size: int = 256, return_aux: bool = False):
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
    L.C.atomic_torch_save(payload, path)
    return L.C.sha256_file(path)


def _grad_summary(model, pair_loss: torch.Tensor | None = None) -> dict:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    out = {
        "grad_tensors": len(grads),
        "nonzero_grad_tensors": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
        "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
    }
    w = model.base.lins[0].weight.grad
    if w is not None:
        out["base_first_weight_grad_norm"] = float(torch.linalg.vector_norm(w).item())
        out["base_r2_column_grad_max_abs"] = float(w[:, E.BASE_R1_TOTAL:].abs().max().item())
        out["base_r1_column_grad_max_abs"] = float(w[:, :E.BASE_R1_TOTAL].abs().max().item())
    out["base_last_weight_grad_norm"] = float(
        torch.linalg.vector_norm(model.base.lins[1].weight.grad).item()
        if model.base.lins[1].weight.grad is not None else 0.0
    )
    out["f_context_last_weight_grad_norm"] = float(
        torch.linalg.vector_norm(model.backbone.f.lins[-1].weight.grad).item()
        if model.backbone.f.lins[-1].weight.grad is not None else 0.0
    )
    out["rho_last_weight_grad_norm"] = float(
        torch.linalg.vector_norm(model.backbone.rho.lins[-1].weight.grad).item()
        if model.backbone.rho.lins[-1].weight.grad is not None else 0.0
    )
    return out


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

    norm = E.load_norm()
    train_cache = E.load_train_cache()
    valid_cache = E.load_valid_cache()
    if train_cache["meta"]["feature_version"] != L.FEATURE_VERSION:
        raise RuntimeError("train cache feature version mismatch")
    if valid_cache["meta"]["feature_version"] != L.FEATURE_VERSION:
        raise RuntimeError("valid cache feature version mismatch")
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)
    pair_data = load_pair_arrays()
    if pair_data["i"].min() < 0 or pair_data["i"].max() >= len(y_train) \
            or pair_data["j"].min() < 0 or pair_data["j"].max() >= len(y_train):
        raise RuntimeError("pair list contains non-train rows")
    if len(valid_cache["labels"]) != 1000:
        raise RuntimeError("valid cache is not the official 1000 rows")

    cfg_backbone = dict(M2.M.DEFAULT_CONFIG)
    model, model_info = M2.build_model(arm, norm, seed, device, cfg_backbone)
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
    pairs_per_step = int(recipe["pairs_per_step"])
    steps_per_epoch = math.ceil(len(rows_train) / batch_size)
    total_updates = steps_per_epoch * n_epochs
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    gen = torch.Generator().manual_seed(int(seed) + 101)
    pair_gen = torch.Generator().manual_seed(int(seed) + 202)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1}
    best_state = None
    best_sha = None
    started = time.perf_counter()
    global_step = 0
    last_lr = float(recipe["lr"])
    first_batch_rows = None
    first_batch_grad = None
    first_pair_ids = None
    first_pair_targets = None
    first_pair_pred_diff = None
    first_pair_loss = None
    first_losses = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        pair_err_sum, pair_n = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = rows_train[order[start:start + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = rows.copy()
            batch = _to_device(D.collate(train_cache, rows), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss_mol = F.l1_loss(pred.view(-1), target.view(-1))

            pair_ids = torch.randint(0, pair_data["n"], (pairs_per_step,), generator=pair_gen)
            pi = pair_data["i"][pair_ids.numpy()]
            pj = pair_data["j"][pair_ids.numpy()]
            pair_rows = np.stack([pi, pj], axis=1).reshape(-1)
            pair_batch = _to_device(D.collate(train_cache, pair_rows), device)
            pair_pred = model(pair_batch).view(pairs_per_step, 2)
            pair_target = torch.as_tensor(
                y_train[pi] - y_train[pj], dtype=torch.float32, device=device
            )
            pair_pred_diff = pair_pred[:, 0] - pair_pred[:, 1]
            loss_pair = F.l1_loss(pair_pred_diff, pair_target)
            loss = loss_mol + float(recipe["pair_loss_coef"]) * loss_pair

            if first_pair_ids is None:
                first_pair_ids = pair_ids.numpy().copy()
                first_pair_targets = pair_target.detach().cpu().numpy().copy()
                first_pair_pred_diff = pair_pred_diff.detach().cpu().numpy().copy()
                first_pair_loss = float(loss_pair.item())
                first_losses = {"loss_mol": float(loss_mol.item()),
                                "loss_pair": float(loss_pair.item()),
                                "loss_total": float(loss.item())}

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if first_batch_grad is None:
                first_batch_grad = _grad_summary(model, loss_pair)
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
            pair_err_sum += float((pair_pred_diff.detach() - pair_target).abs().sum())
            pair_n += pairs_per_step
        online = total_err / max(n_mol, 1)
        pair_online = pair_err_sum / max(pair_n, 1)
        _, valid_mae = evaluate(model, valid_cache, rows_valid, device=device,
                                batch_size=int(recipe["valid_batch_size"]))
        sec = time.perf_counter() - t_ep
        entry = {
            "epoch": int(epoch),
            "online_train_mae": float(online),
            "online_pair_mae": float(pair_online),
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
        L.log(f"[{arm} s{seed}] ep={epoch:03d} valid={valid_mae:.6f} "
              f"online={online:.6f} pair={pair_online:.6f} lr={last_lr:.3e} "
              f"best={best['valid_mae']:.6f}@{best['epoch']} {sec:.2f}s")
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
            "pair_gen_state": pair_gen.get_state(),
            "epoch": n_epochs,
            "global_step": global_step,
            "best": best,
            "curve": curve,
            "arm": arm,
            "seed": int(seed),
        },
        out / "final_state.pt",
    )

    best_model = M2.load_model_from_state(arm, norm, best_state, device, cfg_backbone)
    best_pred_valid, best_valid = evaluate(best_model, valid_cache, rows_valid,
                                           device=device,
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
        "round": E.ROUND_ID,
        "candidate": f"ECLC-{arm}",
        "arm": arm,
        "seed": int(seed),
        "status": "completed",
        "timing_run": bool(timing),
        "git_commit": E.git_commit(),
        "environment": L.env_summary(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(rows_valid)),
            "official_test_loaded": False,
            "train_cache_file_sha256": E.sha256_file(E.LCRC_TRAIN_CACHE),
            "valid_cache_file_sha256": E.sha256_file(E.LCRC_VALID_CACHE),
            "train_cache_content_sha256": D.content_fingerprint(train_cache),
            "valid_cache_content_sha256": D.content_fingerprint(valid_cache),
            "train_label_sha256": L.C.sha256_array(y_train),
            "valid_label_sha256": L.C.sha256_array(valid_cache["labels"]),
            "feature_version": L.FEATURE_VERSION,
            "pair_list": pair_data["path"],
            "pair_list_sha256": pair_data["sha256"],
            "n_pairs": int(pair_data["n"]),
        },
        "model_info": model_info,
        "recipe": recipe,
        "warmup_steps": int(warmup),
        "steps_per_epoch": int(steps_per_epoch),
        "total_updates": int(total_updates),
        "epochs": int(n_epochs),
        "param_audit": audit,
        "gradient_audit_first_batch": first_batch_grad,
        "pair_audit_first_batch": {
            "pair_ids": None if first_pair_ids is None else first_pair_ids.tolist(),
            "targets": None if first_pair_targets is None else first_pair_targets.tolist(),
            "pred_diffs": None if first_pair_pred_diff is None else first_pair_pred_diff.tolist(),
            "loss_pair": first_pair_loss,
            "losses": first_losses,
        },
        "first_batch_rows": None if first_batch_rows is None else first_batch_rows.tolist(),
        "data_order_seed": int(seed) + 101,
        "pair_order_seed": int(seed) + 202,
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
    E.write_json(train_json, payload)
    L.log(f"[{arm} s{seed}] done best={best['valid_mae']:.6f}@{best['epoch']} "
          f"final={final_valid:.6f} wall={wall:.1f}s")
    return payload


def timing(arms: Sequence[str], seed: int, device: str, epochs: int = 2) -> dict:
    """2-epoch smoke: measures steady epoch time with the pair path active."""
    L.require_cuda(device)
    out = {}
    norm = E.load_norm()
    train_cache = E.load_train_cache()
    valid_cache = E.load_valid_cache()
    pair_data = load_pair_arrays()
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)
    for arm in arms:
        cfg_backbone = dict(M2.M.DEFAULT_CONFIG)
        model, info = M2.build_model(arm, norm, seed, device, cfg_backbone)
        recipe = dict(RECIPE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                                      weight_decay=recipe["weight_decay"])
        batch_size = int(recipe["batch_size"])
        steps = math.ceil(len(train_cache["labels"]) / batch_size)
        gen = torch.Generator().manual_seed(int(seed) + 101)
        pair_gen = torch.Generator().manual_seed(int(seed) + 202)
        epoch_seconds = []
        total_updates = steps * int(RECIPE["epochs"])
        first_grad = None
        first_pair = None
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.perf_counter()
            order = torch.randperm(len(train_cache["labels"]), generator=gen).numpy()
            for start in range(0, len(order), batch_size):
                rows = np.asarray(order[start:start + batch_size], dtype=np.int64)
                batch = _to_device(D.collate(train_cache, rows), device)
                target = torch.as_tensor(train_cache["labels"][rows], dtype=torch.float32,
                                         device=device)
                pred = model(batch)
                loss_mol = F.l1_loss(pred.view(-1), target.view(-1))
                pair_ids = torch.randint(0, pair_data["n"], (int(recipe["pairs_per_step"]),),
                                         generator=pair_gen)
                pi = pair_data["i"][pair_ids.numpy()]
                pj = pair_data["j"][pair_ids.numpy()]
                pair_rows = np.stack([pi, pj], axis=1).reshape(-1)
                pair_batch = _to_device(D.collate(train_cache, pair_rows), device)
                pair_pred = model(pair_batch).view(int(recipe["pairs_per_step"]), 2)
                pair_target = torch.as_tensor(train_cache["labels"][pi] - train_cache["labels"][pj],
                                              dtype=torch.float32, device=device)
                loss_pair = F.l1_loss(pair_pred[:, 0] - pair_pred[:, 1], pair_target)
                loss = loss_mol + float(recipe["pair_loss_coef"]) * loss_pair
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if first_grad is None:
                    first_grad = _grad_summary(model, loss_pair)
                    first_pair = {
                        "loss_mol": float(loss_mol.item()),
                        "loss_pair": float(loss_pair.item()),
                        "pair_ids": pair_ids.numpy().tolist(),
                    }
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
                optimizer.step()
            _, valid_mae = evaluate(model, valid_cache, rows_valid, device=device,
                                    batch_size=int(recipe["valid_batch_size"]))
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
            "first_batch_gradient_audit": first_grad,
            "first_pair_audit": first_pair,
            "initial_state_sha256": info["initial_state_sha256"],
        }
    path = E.ANALYSIS / "timing.json"
    E.write_json(path, {"recipe": RECIPE, "runs": out, "created": E.now()})
    L.log(f"timing written to {E.track_rel(path)}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--arm", choices=list(E.ARMS), required=True)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device", type=str, default="cuda:0")
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--tag", type=str, default=None)
    tr.add_argument("--out-dir", type=str, default=None)
    tr.add_argument("--force", action="store_true")
    tm = sub.add_parser("timing")
    tm.add_argument("--arms", type=str, default="NULL")
    tm.add_argument("--seed", type=int, default=0)
    tm.add_argument("--device", type=str, default="cuda:0")
    tm.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        out = train(args.arm, args.seed, args.device, epochs=args.epochs,
                    tag=args.tag,
                    out_dir=Path(args.out_dir) if args.out_dir else None,
                    force=args.force)
        print(out["best"], out["final_valid_mae"])
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        out = timing(arms, args.seed, args.device, epochs=args.epochs)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
