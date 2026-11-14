"""Implementation smoke test: at most 8 short jobs, at most 8 updates each.

Nothing here is a result: the smoke only checks that the plumbing runs end to end
on the same code path as a formal run (model, gated operator, frozen alpha
buffer, batch plan, per-epoch and grid evaluation, checkpoint retention, dose
diagnostics).  No hyper-parameter is changed based on it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import sc_common as C
import sc_model as M
import sc_support as S
import sc_train as T

import lb_data as LD

SMOKE_ARMS = [
    (C.PRIMARY_DATASET, "100", arm) for arm in C.ARMS
] + [("BACE", "100", C.ARM_BASE), ("BACE", "100", C.ARM_SELECT)]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one(dataset: str, fit: str, arm: str, device: torch.device) -> dict[str, Any]:
    key = None if arm == C.ARM_BASE else C.read_json(C.ANALYSIS / "key_selection.json")["chosen_key"]
    out = C.RUNS / f"SMOKE_{'LIP' if dataset == C.PRIMARY_DATASET else 'BACE'}_F{fit}_{arm}"
    payload = T.train_arm(dataset=dataset, fit=fit, arm=arm, device=device, out_dir=out,
                          key=key, smoke=True)
    return {
        "job": out.name, "arm": arm, "dataset": dataset, "fit": fit,
        "status": payload["status"], "parameters": payload["model_info"]["parameters"],
        "init_state_hash": payload["model_info"]["init_state_hash"],
        "gated": payload["model_info"]["gated"],
        "updates": payload["cost"]["updates_run"],
        "wall_seconds": payload["cost"]["wall_seconds"],
        "selection_updates": {r: v["update"] for r, v in payload["selection"]["rules"].items()},
        "grid_points": int(np.load(out / "grid_predictions.npz")["grid_updates"].size),
        "support_key": payload.get("support", {}).get("key") if payload.get("support") else None,
        "dose_final": (payload.get("dose") or {}).get("final", {}).get("molecules", {}),
        "curve_rows": int(sum(1 for _ in open(out / "curve.csv", encoding="utf-8"))) - 1,
    }


def grad_path_check(device: torch.device) -> dict[str, Any]:
    """One gradient path through the gate: every parameter gets a finite gradient."""
    spec = LD.frozen_spec()
    snorm = C.shared_norms(C.PRIMARY_DATASET)[0]
    cache = LD.load_cache(C.PRIMARY_DATASET, "train")
    support = S.load_support(C.PRIMARY_DATASET, "100", C.KEY_FINE)
    rows = sorted(C.fit_positions(C.PRIMARY_DATASET, "100"))[:64]
    runner = T.GateRunner(
        M.build_arm_model(spec, snorm, C.ARM_SELECT).to(device), device,
        {"train": cache, "valid": LD.load_cache(C.PRIMARY_DATASET, "valid"),
         "test": LD.load_cache(C.PRIMARY_DATASET, "test")},
        {s: (np.asarray(support[f"alpha_{C.ARM_SELECT}_{s}"]) if s == "train" else None)
         for s in C.SPLITS},
    )
    model = runner.model
    M.load_shared_init(model, spec, snorm)
    model.train()
    batch = runner.batch("train", rows)
    y = model(batch).reshape(-1)
    loss = (y ** 2).mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    return {"loss": float(loss.detach().cpu()), "n_parameters": len(list(model.parameters())),
            "params_without_finite_grad": missing, "all_trainable": not missing}


def main(argv: Sequence[str] | None = None) -> int:
    t0 = time.perf_counter()
    device = _device()
    C.log(f"[smoke] device={device}")
    gp = grad_path_check(device)
    C.log(f"[smoke] gradient path: {gp['n_parameters']} tensors, missing={gp['params_without_finite_grad']}")
    jobs = []
    for dataset, fit, arm in SMOKE_ARMS[: C.MAX_SMOKE_JOBS]:
        jobs.append(run_one(dataset, fit, arm, device))
    hashes = {(j["dataset"], j["arm"]): j["init_state_hash"] for j in jobs}
    base_hash = hashes.get((C.PRIMARY_DATASET, C.ARM_BASE))
    shared = all(h == base_hash for (d, a), h in hashes.items() if d == C.PRIMARY_DATASET)
    bace_shared = all(h == hashes[("BACE", C.ARM_BASE)] for (d, a), h in hashes.items() if d == "BACE")
    payload = {
        "round_id": C.ROUND_ID, "created_utc": C.utcnow(),
        "device": str(device), "n_jobs": len(jobs), "max_updates": C.SMOKE_MAX_UPDATES,
        "jobs": jobs, "gradient_path": gp,
        "all_arms_share_init_hash": {"Lipophilicity": shared, "BACE": bace_shared},
        "all_parameter_counts_equal_228674": all(j["parameters"] == C.C1_PARAMS for j in jobs),
        "wall_seconds": time.perf_counter() - t0,
        "note": "implementation only; no score is interpreted here",
    }
    C.write_json(C.ANALYSIS / "smoke.json", payload)
    C.log(f"[smoke] {len(jobs)} jobs done in {payload['wall_seconds']:.1f}s, "
          f"shared init={shared}/{bace_shared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
