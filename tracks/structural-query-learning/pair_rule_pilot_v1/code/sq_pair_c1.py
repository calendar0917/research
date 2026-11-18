"""C1 (LCRC-C1 / NO_BOND_MEAN) retrained with the frozen ``Rf`` recipe.

The upstream model and trajectory loop are used read-only; this module only
builds the canonical C1 state, verifies the parameter count, runs a small GPU
smoke, then hands the fresh canonical model to ``gr_train.run_trajectory``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

import sq_pair_common as C

import gr_common as GC
import gr_data as GD
import gr_models as GM
import gr_train as GT


def task() -> dict[str, Any]:
    return {
        "key": f"{C.ROUND_ID}:C1",
        "stage": GC.STAGE1,
        "dataset": C.DATASET,
        "model": "L",
        "recipe": C.C1_RECIPE,
        "epochs": int(C.EPOCHS),
    }


def build_c1(b) -> tuple[Any, dict[str, Any]]:
    model_set = GM.ModelSet(b)
    model = model_set.lcrc()
    params = GM.parameter_report(model)
    if int(params["n_parameters"]) != int(C.C1_EXPECTED_PARAMS):
        raise ValueError(
            f"C1 parameter count {params['n_parameters']} != frozen {C.C1_EXPECTED_PARAMS}"
        )
    init_audit = {
        "kind": "canonical_c1",
        "state_hash_before_training": GM.model_state_hash(model),
        "n_parameters": int(params["n_parameters"]),
        "canonical_init_seed": GC.INIT_SEED_CANONICAL,
    }
    return model, init_audit


def smoke_c1(b, device: str, *, n_rows: int = 16) -> dict[str, Any]:
    import torch

    model, init_audit = build_c1(b)
    model = model.to(device)
    model.train()
    order, _row_ids, targets = GD.eval_order(b, "train")
    idx = order[: min(int(n_rows), order.size)]
    batch = GD.collate(b, "train", idx)
    batch = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()}
    y_std = (targets - float(b.scaler["mean"])) / float(b.scaler["std"])
    batch["y"] = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    out = model(batch).view(-1)
    loss = torch.nn.functional.mse_loss(out, batch["y"].view(-1))
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(GC.GRAD_CLIP)))
    optimizer.step()
    return {
        "stage": "c1_smoke",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_rows": int(idx.size),
        "n_parameters": int(init_audit["n_parameters"]),
        "init": init_audit,
        "loss": float(loss.detach().cpu()),
        "grad_norm_before_clip": grad_norm,
        "finite_output": bool(torch.isfinite(out).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_grad": bool(np.isfinite(grad_norm)),
        "pass": bool(
            torch.isfinite(out).all() and torch.isfinite(loss) and bool(np.isfinite(grad_norm))
        ),
    }


def run_c1(*, out_dir: Path, device: str, smoke: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = C.ensure_dir(out_dir)
    b = GD.bundle(C.DATASET, verify=True)
    smoke_payload = None
    if smoke:
        smoke_payload = smoke_c1(b, device)
        C.write_json(out.parent / "c1_gpu_smoke.json", smoke_payload)
        if not smoke_payload["pass"]:
            raise RuntimeError(f"C1 GPU smoke failed: {smoke_payload}")
    # formal model is always rebuilt from the canonical C1 state; the smoke
    # update above can never leak into it
    model, init_audit = build_c1(b)
    summary = GT.run_trajectory(
        task=task(),
        b=b,
        model=model,
        init_audit=init_audit,
        out_dir=out,
        device=device,
        epochs=int(C.EPOCHS),
        recipe=C.C1_RECIPE,
    )
    summary["round_wall_seconds"] = time.perf_counter() - t0
    summary["smoke"] = smoke_payload
    summary["round"] = C.ROUND_ID
    summary["run_provenance"] = C.run_provenance()
    C.write_json(out / "round_status.json", summary)
    return summary
