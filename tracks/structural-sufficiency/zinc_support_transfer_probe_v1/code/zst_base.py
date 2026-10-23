"""Base model for zinc-support-transfer-probe-v1: train on ``F`` only.

One process = one supervised seed. The model, features and recipe are the
frozen P3 platform; the only change is the supervised row set (``F`` from the
frozen split manifest) and therefore the number of updates per epoch. The
checkpoint is the epoch-600 endpoint: no valid, C_dev or Q based selection.

The job also writes the genuinely out-of-sample per-molecule predictions for
every non-``F`` role, so the head work is pure CPU afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import zst_common as Z  # noqa: E402
import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402

RECIPE = dict(T1.RECIPE_VARIANTS["B"])
BATCH = int(RECIPE["batch_size"])
EPOCHS = int(RECIPE["epochs"])
WARMUP = int(RECIPE["warmup_steps"])
LR = float(RECIPE["lr"])
MIN_LR = float(RECIPE["min_lr"])
WD = float(RECIPE["weight_decay"])
CLIP = float(RECIPE["grad_clip"])


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train(seed: int, device, out_dir: Path, manifest_path: Path,
          epochs: int = EPOCHS, predict: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.load(open(manifest_path, encoding="utf-8"))
    rows_f = np.asarray(man["roles"]["F"], dtype=np.int64)
    cache = D.load_cache("train")
    y = np.asarray(cache["labels"], dtype=np.float64)

    model, info = Z.build_base_model(seed, device)
    if info["parameters"] > 500_000:
        raise RuntimeError("parameter cap exceeded")
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD,
                              betas=(0.9, 0.999), eps=1e-8)
    steps = math.ceil(len(rows_f) / BATCH)
    total = steps * int(epochs)
    warmup = min(WARMUP, max(1, total // 20))
    gen = torch.Generator().manual_seed(int(seed) + 101)
    C.log(f"[base s{seed}] |F|={len(rows_f)} steps/epoch={steps} updates={total} "
          f"init={info['init_state_sha256'][:12]}")
    curve = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(len(rows_f), generator=gen).numpy()
        t_ep = time.perf_counter()
        err, n = 0.0, 0
        for start in range(0, len(order), BATCH):
            local = order[start:start + BATCH]
            rows = rows_f[local]
            batch = _to_device(T1.collate(cache, rows, "P3N", valid=False), device)
            target = torch.as_tensor(y[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            step = (epoch - 1) * steps + (start // BATCH) + 1
            lr = C.cosine_with_warmup_lr(step, total, warmup, LR, MIN_LR)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.step()
            err += float((pred.detach().view(-1) - target).abs().sum())
            n += len(rows)
        sec = time.perf_counter() - t_ep
        curve.append({"epoch": epoch, "online_train_mae": err / max(n, 1),
                      "lr": lr, "seconds": sec, "steps": steps})
        if epoch == 1 or epoch % 50 == 0:
            C.log(f"[base s{seed}] ep={epoch:03d} online={err / max(n, 1):.6f} "
                  f"lr={lr:.2e} {sec:.2f}s")
    wall = time.perf_counter() - started

    # final endpoint: no selection
    ckpt_path = out_dir / "final.pt"
    C.atomic_torch_save(
        {"state_dict": C.cpu_state(model), "epoch": int(epochs), "seed": int(seed),
         "fit_role": "F", "n_fit": int(len(rows_f)), "steps_per_epoch": int(steps),
         "updates": int(total), "model_info": info, "curve": curve,
         "train_wall_seconds": wall},
        ckpt_path,
    )
    in_sample = Z.predict_rows(model, cache, rows_f, device)
    record = {
        "seed": int(seed),
        "fit_role": "F",
        "n_fit": int(len(rows_f)),
        "steps_per_epoch": int(steps),
        "updates": int(total),
        "epochs": int(epochs),
        "parameters": int(info["parameters"]),
        "ssl_encoder_sha256": info["sha256"],
        "init_state_sha256": info["init_state_sha256"],
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": C.sha256_file(ckpt_path),
        "train_wall_seconds": wall,
        "gpu_hours": wall / 3600.0,
        "epoch1_seconds": curve[0]["seconds"],
        "epoch_last_seconds": curve[-1]["seconds"],
        "in_sample_train_mae": float(C.mae(in_sample, y[rows_f])),
        "git_commit": C.git_commit(),
        "device": str(device),
        "torch": torch.__version__,
        "curve": curve,
    }
    C.write_json(out_dir / "train.json", record)
    C.log(f"[base s{seed}] final in-sample MAE {record['in_sample_train_mae']:.6f} "
          f"wall {wall / 60:.1f} min")

    # out-of-sample predictions for every non-F role
    if not predict:
        return record
    roles = {k: np.asarray(v, dtype=np.int64) for k, v in man["roles"].items()}
    union = np.unique(np.concatenate([roles[k] for k in Z.ROLES if k != "F"]))
    pred = Z.predict_rows(model, cache, union, device)
    keys = Z.load_keys()
    _, mol2 = Z.molecule_key_sets(keys)
    sup_fit = Z.key_support(rows_f, mol2, int(keys["n_key2"]))
    Z.PREDS.mkdir(parents=True, exist_ok=True)
    out = Z.PREDS / f"base_F_s{seed}.npz"
    np.savez_compressed(out, rows=union, pred=pred, y=y[union],
                        fit_rows=rows_f, key_support_fit=sup_fit,
                        base_ckpt_sha256=np.asarray(C.sha256_file(ckpt_path)))
    C.log(f"[base s{seed}] predictions -> {out} ({len(union)} rows)")
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=str(Z.CACHE / "split_manifest.json"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no-predict", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (Z.RUNS / f"base_F_s{args.seed}")
    train(args.seed, torch.device(args.device), out, Path(args.manifest),
          epochs=args.epochs, predict=not args.no_predict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
