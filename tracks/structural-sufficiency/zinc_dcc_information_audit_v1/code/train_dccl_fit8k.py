#!/usr/bin/env python
"""Native DCC-L reference training on Fit8k, exactly 47,400 updates.

Frozen contract: AdamW 1e-3 -> 1e-5 cosine (500-step warm-up), wd 1e-5,
batch 128, raw-y L1, clip 5, dropout 0.  No valid / ProbeFit / AuditEval
selection: the saved checkpoint is the state after update 47,400.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_REPO = _TRACK.parents[2]
_V1 = _REPO / "tracks" / "structural-sufficiency" / "zinc_arch_discovery_v1" / "code"
_DCC = _REPO / "tracks" / "structural-sufficiency" / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DCC), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_models as DM  # noqa: E402

UPDATES = 47400
BATCH = 128
LR = 1e-3
MIN_LR = 1e-5
WD = 1e-5
WARMUP = 500
CLIP = 5.0


def _rows_sha(rows) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


def _param_audit(model, opt):
    total = sum(p.numel() for p in model.parameters())
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    opt = sum(p.numel() for p in model.parameters() if id(p) in opt_ids)
    return {"parameters": int(total), "optimizer_parameters": int(opt),
            "parameters_in_optimizer": bool(total == opt),
            "all_parameters_require_grad": all(p.requires_grad for p in model.parameters())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ssl", required=True, help="Fit8k SSL encoder checkpoint for this seed")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--updates", type=int, default=UPDATES)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    out = Path(args.out_dir) if args.out_dir else _TRACK / "runs" / f"DCC_L_fit8k_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    ssl = Path(args.ssl).resolve()
    if not ssl.exists():
        raise FileNotFoundError(ssl)
    # The DCC builder selects the SSL file by seed; point both slots at the
    # declared per-seed Fit8k encoder.
    C.P3_SSL_S0 = ssl
    C.P3_SSL_S1 = ssl

    split = json.loads((_TRACK / "configs" / "split_fit8k_probe_audit.json").read_text())
    fit_rows = np.asarray(split["rows"]["fit"], dtype=np.int64)
    assert len(fit_rows) == 8000
    cache = D.load_cache("train")
    norm = D.load_norm()
    # Fit8k-only global-33 standardization.  DCC-L/P3N does not consume the
    # root/block scalar normalization fields, so they are left at the frozen
    # cache values and flagged in the record.
    gfit = np.asarray(cache["global_feat"], dtype=np.float64)[fit_rows]
    norm = dict(norm)
    norm["global_mean"] = gfit.mean(0).astype(np.float32)
    norm["global_scale"] = np.maximum(gfit.std(0).astype(np.float32), 1e-6)
    y = np.asarray(cache["labels"], dtype=np.float64)
    # no valid/test cache is loaded anywhere in this process
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model, model_info = DM.build_model("DCC_L", norm, int(args.seed), device,
                                       organized_local=True)
    params = sum(p.numel() for p in model.parameters())
    if params != 494193:
        raise RuntimeError(f"unexpected parameter count {params}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    audit = _param_audit(model, opt)
    if not audit["parameters_in_optimizer"]:
        raise RuntimeError("optimizer does not cover all parameters")
    gen = torch.Generator().manual_seed(int(args.seed) + 101)
    total_updates = int(args.updates)
    curve = []
    global_step = 0
    epoch = 0
    last_lr = LR
    started = time.perf_counter()
    gradient_audit = {}
    while global_step < total_updates:
        epoch += 1
        model.train()
        order = torch.randperm(len(fit_rows), generator=gen).numpy()
        t_ep = time.perf_counter()
        err, n_mol = 0.0, 0
        for start in range(0, len(order), int(args.batch_size)):
            if global_step >= total_updates:
                break
            rows = fit_rows[order[start:start + int(args.batch_size)]]
            batch = T1._to_device(T1.collate(cache, rows, "P3N", valid=False), device)
            target = torch.as_tensor(y[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if global_step == 0:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                gradient_audit = {
                    "grad_tensors": len(grads),
                    "nonzero_grad_tensors": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
                    "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "first_batch_rows": int(len(rows)),
                }
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            last_lr = C.cosine_with_warmup_lr(global_step + 1, total_updates, WARMUP, LR, MIN_LR)
            for g in opt.param_groups:
                g["lr"] = last_lr
            opt.step()
            global_step += 1
            err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        curve.append({"epoch": epoch, "global_step": global_step,
                      "online_train_mae": err / max(n_mol, 1),
                      "lr": last_lr, "seconds": time.perf_counter() - t_ep})
        if epoch % int(args.log_every) == 0 or global_step >= total_updates:
            print(f"[DCC_L fit8k s{args.seed}] ep={epoch} step={global_step}/{total_updates} "
                  f"online={curve[-1]['online_train_mae']:.6f} lr={last_lr:.3e}", flush=True)

    # Fixed final object: no selection.  Evaluate only the Fit8k training rows
    # for a train-fit reference; ProbeFit/AuditEval remain untouched.
    state = C.cpu_state(model)
    torch.save({
        "state_dict": state,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best": None,
        "curve": curve,
        "candidate": "DCC_L",
        "seed": int(args.seed),
        "selection": "final update only; no valid/probe/audit selection",
    }, out / "final.pt")
    model.eval()
    fit_mae = T1.evaluate(model, cache, fit_rows, candidate="P3N", device=device,
                          batch_size=256, collate_fn=None)[1]
    wall = time.perf_counter() - started
    payload = {
        "protocol": "zinc-dcc-information-audit-v1",
        "candidate": "DCC_L",
        "seed": int(args.seed),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "git_commit": C.git_commit(),
        "data": {
            "source": "official ZINC train, Fit8k subset only",
            "n_train": int(len(fit_rows)),
            "fit_rows_sha256": _rows_sha(fit_rows),
            "labels_read": "Fit8k train labels only",
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
        "model_info": model_info,
        "normalization": {
            "global33": "Fit8k rows only",
            "global_mean_sha256": C.sha256_array(norm["global_mean"]),
            "global_scale_sha256": C.sha256_array(norm["global_scale"]),
            "root_block_scalars": "not consumed by DCC-L/P3N collate; frozen cache values only",
        },
        "parameter_count": int(params),
        "parameter_audit": audit,
        "recipe": {
            "optimizer": "AdamW", "lr": LR, "min_lr": MIN_LR, "weight_decay": WD,
            "batch_size": int(args.batch_size), "loss": "raw-y L1",
            "warmup_steps": WARMUP, "grad_clip": CLIP, "dropout": 0.0,
            "updates": int(total_updates), "actual_global_steps": int(global_step),
            "epochs_entered": int(epoch), "steps_per_epoch": math.ceil(len(fit_rows) / int(args.batch_size)),
            "checkpoint": "final update only",
        },
        "gradient_audit": gradient_audit,
        "train_fit_mae": float(fit_mae),
        "curve": curve,
        "wall_seconds": float(wall),
        "gpu_hours": None if str(device).startswith("cpu") else float(wall) / 3600.0,
        "output": str(out / "final.pt"),
        "output_sha256": C.sha256_file(out / "final.pt"),
    }
    (out / "train.json").write_text(json.dumps(_jsonable(payload), indent=2))
    print(json.dumps({"seed": args.seed, "updates": global_step, "train_fit_mae": float(fit_mae),
                      "wall_seconds": wall, "gpu_hours": payload["gpu_hours"],
                      "output_sha256": payload["output_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
