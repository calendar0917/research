"""Short CUDA smoke for the TU adaptation (never touches Test).

For each of the three models: build on the device, two forward/backward
optimiser steps on two Train batches, verify finite non-zero gradients, save a
checkpoint, rebuild and reload it, and compare the logits.  The smoke uses the
prepared Train/Valid caches only; no Test structure or label is loaded.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import tu_common as C  # noqa: E402  (inserts the frozen prep code path)

import lcrc_data as PREP_DATA  # noqa: E402
import tu_data as D  # noqa: E402
import tu_model as M  # noqa: E402


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def smoke_model(dataset: str, model_name: str, device: torch.device, out_dir: Path) -> dict[str, Any]:
    prepared = D.load_prepared(dataset, splits=("train", "valid"))
    spec = prepared["spec"]
    norm = prepared["norm"]
    n_classes = int(prepared["n_classes"])
    train_cache = prepared["caches"]["train"]
    valid_cache = prepared["caches"]["valid"]
    C.seed_everything(C.TRAINING_SEED)
    model, info = M.build_model(prepared, model_name, C.TRAINING_SEED, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(C.RECIPE["peak_lr"]),
        betas=tuple(float(x) for x in C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    y_train = np.asarray(prepared["labels"]["train"], dtype=np.int64)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    for step in range(2):
        rows = list(range(step * 64, min((step + 1) * 64, int(train_cache["n_molecules"]))))
        batch = _to_device(PREP_DATA.collate(train_cache, rows), device)
        y = torch.as_tensor(y_train[rows], dtype=torch.int64, device=device)
        logits = model(batch)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.RECIPE["gradient_clip"])))
        if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
            raise RuntimeError(f"{dataset}/{model_name}: non-finite smoke loss/gradient")
        losses.append(float(loss.item()))
        optimizer.step()

    grads = {}
    dead = []
    for name, param in model.named_parameters():
        gn = 0.0 if param.grad is None else float(param.grad.detach().norm().item())
        grads[name] = gn
        if not (math.isfinite(gn) and gn > 0.0):
            dead.append(name)
    if dead:
        raise RuntimeError(f"{dataset}/{model_name}: dead gradient tensors {dead}")

    rows = list(range(min(16, int(valid_cache["n_molecules"]))))
    batch = _to_device(PREP_DATA.collate(valid_cache, rows), device)
    model.eval()
    with torch.no_grad():
        logits_before = model(batch).detach().cpu().clone()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"smoke_{model_name}.pt"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "model": model_name}, ckpt)
    reloaded = (
        M.TULCRC(spec, norm, n_classes, model_name)
        if model_name in ("FULL", "C1")
        else M.GINPlusGlobal(spec, norm, n_classes)
    )
    reloaded.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    reloaded.to(device)
    reloaded.eval()
    with torch.no_grad():
        logits_after = reloaded(batch).detach().cpu()
    reload_delta = float((logits_before - logits_after).abs().max().item())
    if reload_delta != 0.0:
        raise RuntimeError(f"{dataset}/{model_name}: save/reload changed logits by {reload_delta}")
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "model": model_name,
        "parameters": int(M.parameter_count(model)),
        "losses": losses,
        "grad_tensors": len(grads),
        "dead_gradient_tensors": dead,
        "grad_norm_max": float(max(grads.values())),
        "reload_max_abs_logit_delta": reload_delta,
        "checkpoint": C.track_rel(ckpt),
        "checkpoint_sha256": C.sha256_file(ckpt),
        "peak_cuda_allocated_bytes": peak_alloc,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "init_state_hash": info["init_state_hash"],
        "input_widths": info["input_widths"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="IMDB-BINARY", choices=sorted(C.DATASETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=str(C.RUNS / "_smoke"))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    t0 = time.perf_counter()
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "dataset": args.dataset,
        "device": str(device),
        "test_loaded": False,
        "valid_loaded": True,
        "environment": C.environment(),
        "determinism": C.seed_everything(C.TRAINING_SEED),
        "models": {},
        "created_utc": C.utcnow(),
    }
    out_dir = Path(args.out)
    for model_name in C.MODEL_NAMES:
        payload["models"][model_name] = smoke_model(args.dataset, model_name, device, out_dir)
    payload["all_passed"] = bool(
        all(m["reload_max_abs_logit_delta"] == 0.0 and not m["dead_gradient_tensors"] for m in payload["models"].values())
    )
    payload["seconds"] = float(time.perf_counter() - t0)
    C.write_json(C.ANALYSIS / "smoke_cuda.json", payload)
    print(C.stable_json({"all_passed": payload["all_passed"], "device": str(device), "seconds": payload["seconds"]}))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
