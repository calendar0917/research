"""Training / evaluation loop for zinc-arch-exploration-v3.

This is a purpose-built loop (not ``ad_train.train``) because the pair-arm batch
layout is different from every earlier family.  The semantics it keeps from the
shared v1/v2 harness are identical:

* raw-``y`` L1 on the official ZINC train split, batch shuffle seeded ``seed+101``;
* full official-valid MAE after every epoch, selection by strict ``<`` (earlier
  epoch wins a tie), ``best.pt`` + ``final.pt`` + ``curve.csv`` + ``train.json``;
* an eval-mode full-train MAE measurement for the selected best checkpoint;
* optimizer coverage audit, parameter cap, finite-loss/finite-gradient audit.

Recipes are not inherited mechanically: the new trunks get a declared, longer
schedule (see ``DESIGN.md``), chosen after the measured per-epoch timing.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_V1 = _EXP.parent / "zinc_arch_discovery_v1" / "code"
import sys

for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import zae_data as ZD  # noqa: E402

ROUND_ID = "zinc-arch-exploration-v3"
RUNS = _EXP / "runs"
ANALYSIS = _EXP / "analysis"
PROTOCOL_ID = "zinc-arch-exploration-v3"

#: declared formal recipe (identical for A, B and the A-SSL supervised phase)
RECIPE: dict[str, Any] = dict(
    optimizer="adamw",
    lr=1.0e-3,
    min_lr=1.0e-5,
    weight_decay=1.0e-4,
    warmup_steps=1000,
    epochs=1200,
    batch_size=64,
    valid_batch_size=128,
    grad_clip=5.0,
    dropout=0.0,
    loss="raw-y L1",
)

#: declared masked-SSL recipe for A-SSL
SSL_RECIPE: dict[str, Any] = dict(
    optimizer="adamw",
    lr=1.0e-3,
    min_lr=1.0e-5,
    weight_decay=1.0e-4,
    warmup_steps=300,
    epochs=120,
    batch_size=64,
    grad_clip=5.0,
    atom_loss_weight=1.0,
    bond_loss_weight=1.0,
)


def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}{arm}_s{int(seed)}"


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _param_audit(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    total = C.parameter_count(model)
    opt = C.optimizer_parameter_count(optimizer)
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


@torch.no_grad()
def evaluate(model, cache, pair_cache, rows, *, device, batch_size=128) -> tuple[np.ndarray, float]:
    model.eval()
    out = []
    with C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = _to_device(ZD.collate_pair(cache, pair_cache, chunk, valid=True), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    return pred, C.mae(pred, y)


def _make_optimizer(model, recipe):
    name = str(recipe["optimizer"]).lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(recipe["lr"]),
                                 weight_decay=float(recipe["weight_decay"]))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]),
                                weight_decay=float(recipe["weight_decay"]))
    raise ValueError(name)


def train_supervised(arm: str, seed: int, device, *, builder: Callable, model_info: dict,
                     out_dir: Path, recipe: Mapping[str, Any] | None = None,
                     epochs: int | None = None, force: bool = False,
                     init_state: Mapping[str, torch.Tensor] | None = None,
                     extra_info: Mapping[str, Any] | None = None) -> dict:
    recipe = dict(RECIPE if recipe is None else recipe)
    n_epochs = int(recipe["epochs"] if epochs is None else epochs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "train.json").exists() and not force:
        C.log(f"[{arm} s{seed}] already complete; use --force to rerun")
        return C.read_json(out_dir / "train.json")

    norm = D.load_norm()
    train_cache, train_pair = D.load_cache("train"), ZD.load_pair_split("train")
    valid_cache, valid_pair = D.load_cache("valid"), ZD.load_pair_split("valid")
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)

    model, info = builder(arm, norm, seed, device)
    if init_state is not None:
        C.load_state(model, init_state, device)
        info["supervised_init"] = "A-SSL pretrained state"
        info["initial_state_sha256_after_ssl"] = C.state_hash(model)
    if model_info:
        info.update(model_info)
    params = C.parameter_count(model)
    if params > 500_000:
        raise RuntimeError(f"{arm}: {params} > 500k")
    optimizer = _make_optimizer(model, recipe)
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
    gradient_audit: dict[str, Any] = {}
    started = time.perf_counter()
    last_lr = float(recipe["lr"])
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            batch = _to_device(ZD.collate_pair(train_cache, train_pair, rows), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 1 and start == 0:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                gradient_audit = {
                    "grad_tensors": len(grads),
                    "nonzero_grad_tensors": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
                    "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "first_batch_rows": int(len(rows)),
                }
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(
                global_step, total_updates, warmup, float(recipe["lr"]), float(recipe["min_lr"])
            )
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)
        _, valid_mae = evaluate(model, valid_cache, valid_pair, rows_valid,
                                device=device, batch_size=int(recipe["valid_batch_size"]))
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
            best_state = C.cpu_state(model)
            C.atomic_torch_save(
                {"state_dict": best_state, "epoch": int(epoch), "valid_mae": float(valid_mae),
                 "candidate": arm, "seed": int(seed)},
                out_dir / "best.pt",
            )
        C.log(
            f"[{arm} s{seed}] ep={epoch:04d} valid={valid_mae:.6f} online={online:.6f} "
            f"lr={last_lr:.3e} best={best['valid_mae']:.6f}@{best['epoch']} {sec:.2f}s"
        )

    wall = time.perf_counter() - started
    C.atomic_torch_save(
        {
            "state_dict": C.cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "gen_state": gen.get_state(),
            "epoch": n_epochs,
            "global_step": global_step,
            "best": best,
            "curve": curve,
            "candidate": arm,
            "seed": int(seed),
        },
        out_dir / "final.pt",
    )

    best_model, _ = builder(arm, norm, seed, device)
    C.load_state(best_model, best_state, device)
    best_pred_valid, best_valid = evaluate(best_model, valid_cache, valid_pair, rows_valid,
                                          device=device, batch_size=int(recipe["valid_batch_size"]))
    _, best_train = evaluate(best_model, train_cache, train_pair, rows_train,
                             device=device, batch_size=128)
    _, final_valid = evaluate(model, valid_cache, valid_pair, rows_valid,
                              device=device, batch_size=int(recipe["valid_batch_size"]))
    _, final_train = evaluate(model, train_cache, train_pair, rows_train, device=device, batch_size=128)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / "valid_best.npz", row=rows_valid, pred=best_pred_valid)

    payload = {
        "round": ROUND_ID,
        "protocol_id": PROTOCOL_ID,
        "candidate": arm,
        "seed": int(seed),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(rows_valid)),
            "official_test_loaded": False,
            "train_label_sha256": C.sha256_array(y_train),
            "valid_label_sha256": C.sha256_array(np.asarray(valid_cache["labels"])),
            "train_cache_meta": {k: train_cache["meta"].get(k) for k in
                                 ("feature_version", "object_source", "labels_sha256")},
            "pair_cache_meta": train_pair["meta"],
            "pair_cache_sha256": {"train": C.sha256_file(ZD.pair_cache_path("train")),
                                  "valid": C.sha256_file(ZD.pair_cache_path("valid"))},
        },
        "model_info": info,
        "parameter_count": int(params),
        "parameter_audit": audit,
        "recipe": {**recipe, "epochs": int(n_epochs), "steps_per_epoch": int(steps_per_epoch),
                   "updates": int(total_updates), "warmup_steps_effective": int(warmup),
                   "actual_global_steps": int(global_step), "valid_every_epoch": True},
        "gradient_audit": gradient_audit,
        "best_valid": best,
        "fixed_objects": {
            "best": {"valid_mae": float(best_valid), "train_mae_eval_mode": float(best_train),
                     "epoch": int(best["epoch"])},
            "final": {"valid_mae": float(final_valid), "train_mae_eval_mode": float(final_train),
                      "epoch": int(n_epochs)},
        },
        "curve": curve,
        "wall_seconds": float(wall),
        "gpu_hours": None if str(device).startswith("cpu") else float(wall) / 3600.0,
        "smoke_override": bool(epochs is not None),
        "device": str(device),
        "notes": [
            "online_train_mae is a within-epoch pre-update running mean",
            "train_mae_eval_mode is a full-10k eval-mode measurement of the fixed checkpoint",
            "selection is argmin(valid_mae); ties keep the earlier epoch",
        ],
    }
    if extra_info:
        payload["extra"] = dict(extra_info)
    C.write_json(out_dir / "train.json", payload)
    C.write_curve_csv(out_dir / "curve.csv", curve)
    C.log(f"[{arm} s{seed}] done best={best_valid:.6f}@{best['epoch']} "
          f"final={final_valid:.6f} wall={wall:.1f}s params={params}")
    return payload
