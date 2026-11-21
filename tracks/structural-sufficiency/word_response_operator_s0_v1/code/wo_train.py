"""Frozen training recipe and evaluation for the word-response-operator s0.

One seed (0), one recipe, fixed epoch horizon.  Everything that touches a label
uses Fit only; the official Valid split is used only for checkpoint selection.
The H holdout labels are never loaded here.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping, Sequence

import numpy as np

import wo_common as C
import wo_data as D
import wo_model as M


def load_templates() -> dict[str, np.ndarray]:
    record = C.read_json(C.analysis_dir() / "structure_record.json")
    payload = C.load_npz(C.ROUND_DIR / record["template_path"])
    return {k.replace("orbit/", "").replace("tmpl/", ""): v for k, v in payload.items()}


def load_all() -> dict[str, Any]:
    record = C.read_json(C.analysis_dir() / "structure_record.json")
    split = C.read_json(C.analysis_dir() / "split_freeze.json")
    data = C.data_dir()
    fit = D.load_split("fit", data / "wo_struct_fit.npz")
    valid = D.load_split("valid", data / "wo_struct_valid.npz")
    h = D.load_split("h", data / "wo_struct_h.npz")
    unlock = data / "wo_h_labels.npz"
    if unlock.is_file():
        h.labels = np.load(unlock)["labels"].astype(np.float32)
    # Fit-only standardisation of the count vector; zero-variance columns dropped.
    z = fit.z
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    keep = std > 1e-8
    z_fit = (z - mean) / np.where(keep, std, 1.0)
    z_fit = z_fit[:, keep]
    for split_data in (fit, valid, h):
        zz = (split_data.z - mean) / np.where(keep, std, 1.0)
        split_data.set_normalizer(zz[:, keep], record["n0"])
        split_data.target_mean = float(record["scaler"]["mean"])
        split_data.target_std = float(record["scaler"]["std"])
    return {"record": record, "split": split, "fit": fit, "valid": valid, "h": h,
            "templates": load_templates(), "z_keep": keep, "z_mean": mean, "z_std": std}


def build_model(cfg: Mapping[str, Any], data: Mapping[str, Any], arm: str,
                hidden: int | None = None, seed: int | None = None) -> M.OperatorNet:
    record = data["record"]
    templates = data["templates"]
    model = M.OperatorNet(
        cfg, record["dims"], record["n_templates"],
        int(record["n_site_orbits"]), int(record["n_pair_orbits"]),
        int(data["fit"].z_std.shape[1]), arm, hidden=hidden,
        seed=int(cfg["recipe"]["seed"] if seed is None else seed))
    return model


def scaled_penalty(model: M.OperatorNet, templates: Mapping[str, np.ndarray]) -> Any:
    import torch

    if not model.use_word:
        return torch.zeros((), device=model.beta.device)
    s_type = torch.as_tensor(templates["s_type"], dtype=torch.float32, device=model.theta.device)
    site_s = s_type[torch.as_tensor(templates["site_type"], dtype=torch.long,
                                   device=model.theta.device)]
    pair_s = s_type[torch.as_tensor(templates["pair_type"], dtype=torch.long,
                                   device=model.theta.device)]
    term = (site_s[:, None] ** 2 * model.psi ** 2).mean() + \
           (pair_s[:, None] ** 2 * model.theta ** 2).mean()
    return term


def loss_terms(model: M.OperatorNet, out, target, templates, data) -> tuple[Any, dict[str, float]]:
    import torch

    n_fit = float(len(data["fit"].row_ids))
    mse = torch.nn.functional.mse_loss(out["yhat"], target)
    beta_pen = (model.beta ** 2).sum() / n_fit
    word_pen = scaled_penalty(model, templates)
    alpha_pen = (model.alpha ** 2).sum()
    loss = mse + beta_pen + 0.001 * word_pen + 1e-4 * alpha_pen
    return loss, {"mse": float(mse.detach()), "beta_pen": float(beta_pen.detach()),
                  "word_pen": float(word_pen.detach()), "alpha_pen": float(alpha_pen.detach())}


def batches(n: int, batch: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + batch, n), dtype=np.int64)
            for start in range(0, n, batch)]


def evaluate(model: M.OperatorNet, split_data: D.SplitData, device, batch: int = 256,
             with_F: bool = False) -> dict[str, Any]:
    import torch

    model.eval()
    preds, rows, targets, Fs = [], [], [], []
    with torch.no_grad():
        for idx in batches(split_data.n_molecules, batch):
            packed = split_data.pack(idx, device=device)
            out = model(packed)
            preds.append(out["yhat"].detach().cpu().numpy())
            if with_F:
                Fs.append(out["F"].detach().cpu().numpy())
            rows.append(split_data.row_ids[idx])
            targets.append(split_data.labels[idx])
    pred = np.concatenate(preds)
    out = {
        "pred_std": pred,
        "row_ids": np.concatenate(rows),
        "target_raw": np.concatenate(targets),
        "pred_raw": pred * split_data.target_std + split_data.target_mean,
    }
    if with_F:
        out["F"] = np.concatenate(Fs)
    return out


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def lr_at(step: int, peak: float, warmup_steps: int, total_steps: int, min_lr: float) -> float:
    if step < warmup_steps:
        frac = step / max(1, warmup_steps)
        return peak * frac
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + np.cos(np.pi * progress))


def train_arm(data: Mapping[str, Any], arm: str, epochs: int, device: str,
              hidden: int | None = None, score_valid: bool = True,
              tag: str | None = None, dry_steps: int | None = None,
              deadline: float | None = None) -> dict[str, Any]:
    import torch
    from pathlib import Path

    cfg = C.load_config()
    recipe = cfg["recipe"]
    C.set_determinism()
    fit = data["fit"]
    model = build_model(cfg, data, arm, hidden=hidden).to(device)
    init_hash = _state_hash(model)
    n_fit = fit.n_molecules
    batch = int(recipe["batch"])
    order_rng = np.random.default_rng(int(recipe["batch_order_seed"]))
    perms = [order_rng.permutation(n_fit) for _ in range(max(1, epochs))]
    steps_per_epoch = int(np.ceil(n_fit / batch))
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    total_steps = int(epochs) * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                                  weight_decay=float(recipe["weight_decay"]))
    target_all = (fit.labels - fit.target_mean) / fit.target_std
    curves: list[dict[str, Any]] = []
    best = {"valid_rmse_raw": float("inf"), "epoch": None}
    best_state = None
    final_state = None
    best_pred: dict[str, Any] = {}
    global_step = 0
    t0 = time.perf_counter()
    grad_stats: dict[str, Any] = {}
    stop = False
    for epoch in range(1, int(epochs) + 1):
        model.train()
        perm = perms[epoch - 1]
        for start in range(0, n_fit, batch):
            idx = perm[start:start + batch]
            packed = fit.pack(idx, device=device)
            target = torch.as_tensor(target_all[idx], device=device)
            optimizer.param_groups[0]["lr"] = lr_at(global_step, float(recipe["peak_lr"]),
                                                    warmup_steps, total_steps,
                                                    float(recipe["min_lr"]))
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            out = model(packed)
            loss, parts = loss_terms(model, out, target, data["templates"], data)
            loss.backward()
            if epoch == 1 and start == 0:
                grad_stats = _grad_stats(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["clip"]))
            optimizer.step()
            if dry_steps is not None and global_step >= dry_steps:
                stop = True
                break
        row = {"epoch": epoch, "seconds": time.perf_counter() - t0,
               "lr": float(optimizer.param_groups[0]["lr"]),
               "train_loss": float(loss.detach()),
               **{f"loss_{k}": v for k, v in parts.items()},
               "cholesky_jitter": int(model.cholesky_jitter)}
        if score_valid:
            train_eval = evaluate(model, fit, device)
            valid_eval = evaluate(model, data["valid"], device)
            row["train_rmse_raw"] = C.rmse(train_eval["pred_raw"], train_eval["target_raw"])
            row["valid_rmse_raw"] = C.rmse(valid_eval["pred_raw"], valid_eval["target_raw"])
            row["train_mae_raw"] = C.mae(train_eval["pred_raw"], train_eval["target_raw"])
            row["valid_mae_raw"] = C.mae(valid_eval["pred_raw"], valid_eval["target_raw"])
            row["seconds"] = time.perf_counter() - t0
            if row["valid_rmse_raw"] < best["valid_rmse_raw"]:
                best = {"valid_rmse_raw": float(row["valid_rmse_raw"]), "epoch": int(epoch)}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_pred = {"fit": train_eval, "valid": valid_eval}
        curves.append(row)
        if stop:
            break
        if deadline is not None and time.time() > deadline:
            break
    seconds = time.perf_counter() - t0
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    result = {
        "arm": arm,
        "epochs": int(epochs),
        "hidden": int(model.hidden),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "init_state_hash": init_hash,
        "cholesky_jitter": int(model.cholesky_jitter),
        "curves": curves,
        "seconds": seconds,
        "grad_stats": grad_stats,
        "steps_per_epoch": steps_per_epoch,
        "seed": int(cfg["recipe"]["seed"]),
        "truncated_by_deadline": bool(deadline is not None and time.time() > deadline),
        "state_best": best_state if best_state is not None else final_state,
        "state_final": final_state,
    }
    if score_valid and best_state is not None:
        result["best"] = best
        result["predictions_best"] = {"fit": best_pred["fit"], "valid": best_pred["valid"]}
        result["final"] = {
            "epoch": int(epochs),
            "train_rmse_raw": curves[-1].get("train_rmse_raw"),
            "valid_rmse_raw": curves[-1].get("valid_rmse_raw"),
            "train_mae_raw": curves[-1].get("train_mae_raw"),
            "valid_mae_raw": curves[-1].get("valid_mae_raw"),
        }
    return result


def _grad_stats(model) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = name.split(".")[0]
        out[key] = out.get(key, 0.0) + float(param.grad.abs().sum())
    return out


def save_run(result: Mapping[str, Any], path) -> None:
    import torch
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items()
               if k not in ("state_best", "state_final", "predictions_best")}
    torch.save({"state": result["state_best"], "config": C.load_config()}, path / "best.pt")
    torch.save({"state": result["state_final"], "config": C.load_config()}, path / "final.pt")
    C.write_json(path / "run.json", C.jsonable(payload))
    preds = result.get("predictions_best")
    if preds:
        np.savez_compressed(path / "predictions_best.npz", **{
            f"{split}_{key}": np.asarray(value)
            for split, payload in preds.items()
            for key, value in payload.items() if key in ("row_ids", "pred_raw", "target_raw",
                                                         "pred_std")})
    curves = result.get("curves") or []
    C.write_json(path / "curves.json", C.jsonable(curves))
