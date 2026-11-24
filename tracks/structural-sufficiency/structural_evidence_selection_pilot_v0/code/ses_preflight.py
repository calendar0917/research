"""Throughput preflight and horizon decision for SES-v0.

Measures the real three-arm cost on the actual candidate artifact:

* real candidate collation + R2 encoding + backward/AdamW step;
* the full Train/Valid/Test eval-mode sweep that every epoch performs;
* the DENSE arm on the complete original graph.

The models built here are thrown away.  Formal runs are separate processes and
reinitialise from the same seed.  The chosen horizon is written to
``analysis/preflight.json`` and must be one of 120/90/60.
"""

from __future__ import annotations

import math
import pickle
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import ses_common as C
import ses_data as D
import ses_structure as S
import ses_tensor as T
import ses_prepare
import ses_train as TR

ARM_ORDER = ("DENSE", "RANDOM", "SELECT")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _train_step(model: Any, selector: Any, arm: str, rows: np.ndarray,
                prepared: dict[str, Any], bundle: D.DataBundle,
                device: torch.device, step: int, total_steps: int,
                warmup_steps: int, recipe: dict[str, Any], opt: Any,
                batch_mol: int) -> float:
    model.train()
    if selector is not None:
        selector.train()
    part = np.asarray(rows, np.int64)[:batch_mol]
    y_std = bundle.y_std(part)
    lr = C.lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                 float(recipe["min_lr"]))
    for group in opt.param_groups:
        group["lr"] = lr
    if arm == "DENSE":
        structs = [prepared["full_structs"][int(r)] for r in part.tolist()]
        batch = T.collate(structs, device)
        loss = TR._loss_dense(model, batch, y_std, device)
    else:
        structs, mol_index, _ = TR._chunk_structs(part, prepared)
        loss, _ = TR._candidate_loss(model, selector, arm, structs, mol_index,
                                     y_std, part.size, device)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    return float(loss.detach().cpu())


def _measure_arm(arm: str, bundle: D.DataBundle, prepared: dict[str, Any],
                 cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    recipe = dict(cfg["recipe"])
    atom_dims, bond_dims = D.atom_bond_dims()
    model, info = C.build_predictor(atom_dims, bond_dims, bundle, cfg, int(cfg["seed"]))
    model.zero_unseen(bundle.atom_ok, bundle.bond_ok, atom_dims, bond_dims)
    init_hash = C.array_sha256(*[
        p.detach().cpu().numpy() for _, p in sorted(model.state_dict().items())])
    model = model.to(device)
    selector = (TR.build_selector(model.readout_in_dim, int(cfg["seed"])).to(device)
                if arm == "SELECT" else None)
    params = list(model.parameters()) + (list(selector.parameters()) if selector else [])
    opt = torch.optim.AdamW(params, lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    batch_mol = int(recipe["batch_molecules"])
    steps_per_epoch = int(math.ceil(bundle.train_rows.size / batch_mol))
    total_steps = int(cfg["horizon"]["target_epochs"]) * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    rows = np.asarray(bundle.train_rows, np.int64)

    # warmup (not measured)
    for _ in range(2):
        _train_step(model, selector, arm, rows, prepared, bundle, device, 0,
                    total_steps, warmup_steps, recipe, opt, batch_mol)
    _sync(device)
    t0 = time.time()
    n_steps = 5
    losses = []
    for k in range(n_steps):
        losses.append(_train_step(model, selector, arm, rows, prepared, bundle,
                                  device, 2 + k, total_steps, warmup_steps,
                                  recipe, opt, batch_mol))
    _sync(device)
    step_seconds = (time.time() - t0) / n_steps

    _sync(device)
    t0 = time.time()
    train_res = TR.eval_arm(model, selector, arm, bundle.train_rows, prepared, bundle,
                            device, int(recipe["eval_batch_molecules"]))
    valid_res = TR.eval_arm(model, selector, arm, bundle.valid_rows, prepared, bundle,
                            device, int(recipe["eval_batch_molecules"]))
    test_res = TR.eval_arm(model, selector, arm, bundle.test_rows, prepared, bundle,
                           device, int(recipe["eval_batch_molecules"]))
    _sync(device)
    eval_seconds = time.time() - t0
    eval_metrics = {}
    if arm == "DENSE":
        eval_metrics = {"train": TR.metrics_dense(train_res),
                        "valid": TR.metrics_dense(valid_res),
                        "test": TR.metrics_dense(test_res)}
    else:
        eval_metrics = {"train": TR.metrics_candidates(train_res, arm),
                        "valid": TR.metrics_candidates(valid_res, arm),
                        "test": TR.metrics_candidates(test_res, arm)}
    epoch_seconds = steps_per_epoch * step_seconds + eval_seconds
    result = {
        "arm": arm, "device": str(device),
        "predictor_parameters": int(sum(p.numel() for p in model.parameters())),
        "selector_parameters": TR.selector_parameters(selector),
        "init_state_sha256": init_hash,
        "steps_per_epoch": int(steps_per_epoch),
        "measured_train_step_seconds": float(step_seconds),
        "measured_eval_seconds": float(eval_seconds),
        "estimated_epoch_seconds": float(epoch_seconds),
        "losses": losses,
        "eval_metrics_smoke": eval_metrics,
        "estimates": {str(h): float(epoch_seconds) * int(h)
                      for h in cfg["horizon"]["allowed"]},
    }
    del model, selector, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main(prepared_path: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    if not C.load_config()["candidate"]:
        raise RuntimeError("config missing candidate block")
    bundle = D.load_bundle()
    path = Path(prepared_path) if prepared_path else C.PREPARED_DIR / "ses_prepared.pkl"
    prepare_info = None
    if not path.is_file():
        prepare_info = ses_prepare.main(str(path))
    prepared = TR.load_prepared(path, bundle)
    cfg = C.load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    measurements = {}
    for arm in ARM_ORDER:
        measurements[arm] = _measure_arm(arm, bundle, prepared, cfg, device)

    target_seconds = float(cfg["horizon"]["target_wall_minutes"][1]) * 60.0
    hard_seconds = float(cfg["horizon"]["hard_wall_minutes"]) * 60.0
    gpu_budget_seconds = float(cfg["horizon"]["gpu_budget_hours"]) * 3600.0
    decision = None
    for horizon in sorted([int(x) for x in cfg["horizon"]["allowed"]], reverse=True):
        est = {a: measurements[a]["estimates"][str(horizon)] for a in ARM_ORDER}
        # SELECT and RANDOM start on GPU0/GPU1; DENSE starts on the first free GPU.
        wall = max(est["SELECT"], est["RANDOM"],
                   min(est["SELECT"], est["RANDOM"]) + est["DENSE"])
        gpu_hours = sum(est.values()) / 3600.0
        decision = {
            "horizon": horizon, "estimated_seconds": est,
            "estimated_wall_seconds": wall, "estimated_gpu_hours": gpu_hours,
            "within_target_wall": wall <= target_seconds,
            "within_hard_wall": wall <= hard_seconds,
            "within_gpu_budget": gpu_hours <= gpu_budget_seconds,
        }
        if wall <= target_seconds and wall <= hard_seconds and gpu_hours <= gpu_budget_seconds:
            break
    feasible = bool(decision["within_target_wall"] and decision["within_hard_wall"]
                    and decision["within_gpu_budget"])
    payload = {
        "host": socket.gethostname(), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "prepared_path": str(path), "prepared_build_seconds": prepared["stats"]["build_seconds"],
        "prepare_created_within_preflight": prepare_info is not None,
        "measurements": measurements,
        "target_wall_seconds": target_seconds, "hard_wall_seconds": hard_seconds,
        "gpu_budget_hours": cfg["horizon"]["gpu_budget_hours"],
        "decision": decision, "feasible": feasible,
        "stop_if_not_feasible": not feasible,
        "preflight_seconds": time.time() - t0,
        "note": "measurement models are discarded; formal runs are separate processes",
    }
    C.write_json(C.ANALYSIS_DIR / "preflight.json", payload)
    print(f"[preflight] horizon={decision['horizon']} feasible={feasible} "
          f"wall={decision['estimated_wall_seconds'] / 60:.2f} min "
          f"gpu_h={decision['estimated_gpu_hours']:.3f}", flush=True)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", default=None)
    args = parser.parse_args()
    main(args.prepared)
