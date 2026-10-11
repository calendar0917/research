"""Diagnostics only: how reproducible is a seed-0 F7200 epoch on the GPU?

Motivation
----------
The historical F7200 arm (``results/feasibility_v2_gpu/train_Full-Y-fit7200-seed0.json``)
ran the same recipe on the same host/GPU as the budget-control re-run, yet its
recorded epoch-1 train MAE (0.906464) does not match the re-run (0.872567).
This probe isolates the cause: it runs independent, freshly re-seeded replicas
of one training epoch *inside one process* on the same device, so environment,
code path and data are held fixed.

It answers two narrow questions:

1. Does a re-seeded replica reproduce the first *forward* pass bitwise (the
   dropout mask stream and the kernel reductions)?
2. Do two re-seeded replicas land on the same epoch-1 MAE / post-step state?

Prints one JSON object; writes no artifacts; never reads labels of the
official validation or test split.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_full_y_budget_control_v1 as drv  # noqa: E402
import run_prediction_feasibility_v2 as rpv2  # noqa: E402
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm  # noqa: E402


def first_batch_fingerprint(
    blob: dict, train_rows, fit_rows, y_all: np.ndarray, device: torch.device
) -> dict:
    """Forward pass on the first training batch only, then one optimizer step."""
    drv.seed_everything(0)
    model = drv.build_model(blob, 0).to(device)
    init_hash = drv.state_hash({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    fit_data = [train_rows[i] for i in fit_rows]
    y_fit = torch.as_tensor(y_all[fit_rows], dtype=torch.float32)
    gen = torch.Generator().manual_seed(0 + drv.TRAIN_SHUFFLE_OFFSET)
    first = next(iter(drv._batches(len(fit_data), drv.BATCH_SIZE, gen, True)))
    batch = drv._batch(fit_data, first, y_fit, device)
    model.train()
    prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
    task = nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1))
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = task + drv.H1_LAMBDA * rec
    fingerprint = {
        "init_state_sha256": init_hash,
        "first_batch_indices_sum": int(np.sum(first)),
        "loss_sha256": drv.state_hash({"loss": loss.detach().cpu()}),
        "pred_sha256": drv.state_hash({"pred": prediction.detach().cpu()}),
        "task": float(task.detach()),
        "rec": float(rec.detach()),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=drv.LR, weight_decay=drv.WEIGHT_DECAY)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), drv.GRAD_CLIP)
    optimizer.step()
    fingerprint["post_step_state_sha256"] = drv.state_hash(
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    )
    return fingerprint


def replica_epoch(blob: dict, train_rows, fit_rows, y_all: np.ndarray, device: torch.device) -> dict:
    _, meta = drv.train_one_arm(
        arm="probe", blob=blob, train_rows=train_rows, fit_rows=fit_rows,
        y_all=y_all, device=device, epochs=1, log_every=0,
    )
    return {
        "init_state_sha256": meta["init_state_sha256"],
        "epoch1_train_mae": meta["curve"][0]["train_task_mae"],
        "epoch1_rec": meta["curve"][0]["train_rec"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replicas", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)

    idx = drv.load_split_indices()
    fit_rows = np.asarray(idx["fit_inner"], dtype=np.int64)
    train = drv.load_encoded_train()
    y_all = np.array([float(d.y.reshape(-1)[0]) for d in train], dtype=np.float64)
    prep7200 = drv.load_blob(drv.PREP7200)
    blob10 = drv.load_blob(drv.HIST_PREP)
    rpv2.apply_prep_fit7200(train, prep7200, blob10)

    out = {
        "probe": "seed0-determinism-v1",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "n_fit_rows": int(fit_rows.size),
        "replicas": [replica_epoch(prep7200, train, fit_rows, y_all, device)
                     for _ in range(int(args.replicas))],
        "first_batch": [first_batch_fingerprint(prep7200, train, fit_rows, y_all, device)
                        for _ in range(int(args.replicas))],
        "historical_F7200_epoch1_train_mae": 0.906464,
        "budget_control_F7200_epoch1_train_mae": 0.872567,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
