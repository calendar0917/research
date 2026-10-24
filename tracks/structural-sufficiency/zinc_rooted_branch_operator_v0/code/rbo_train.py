"""Train RBO (seed 0) on F_minus = B + D_minus, select on M, then score once.

Same supervision placement as the frozen P3-minus reference of
``zinc_composition_transfer_v1`` (7,549 labelled train molecules: 7,098 B rows
plus the 451 D_minus donors), same recipe B (AdamW 1e-3 -> 1e-5, 500-step
warm-up, per-step cosine, wd 1e-5, clip 5, batch 128, raw-y L1, FP32), same
batch construction (``zct_support.fit_batches``).

Selection is best-M only.  ``Q_target`` / ``Q_background`` labels are read once,
after the best checkpoint is written and hashed (``--score-only`` path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_CT = _SSUF / "zinc_composition_transfer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_CT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import rbo_model as R  # noqa: E402
import zct_support as ZS  # noqa: E402

TASK = _TASK
CACHE = TASK / "cache"
RUNS = TASK / "runs"

BATCH = 128
EPOCHS = 600
WARMUP = 500
LR = 1e-3
MIN_LR = 1e-5
WD = 1e-5
CLIP = 5.0
BETAS = (0.9, 0.999)
EPS = 1e-8


def collate_fn(cache, rows, candidate="RBO", valid=False):
    return R.collate(rows, cache, PREP[0], "cpu")


PREP = [None]


def load_prep(device):
    d = torch.load(CACHE / "rbo_train.pt", map_location="cpu", weights_only=False)
    PREP[0] = d
    return d


def evaluate(model, rows, cache, device, batch_size=256):
    return T1.evaluate(model, cache, np.asarray(rows), candidate="RBO", device=device,
                       batch_size=batch_size, collate_fn=collate_fn)


def state_digest(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(RUNS / "rbo_minus_s0"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sup = dict(np.load(_SSUF / "zinc_composition_transfer_v1" / "cache" / "support_sets.npz"))
    cache = D.load_cache("train")
    prep = load_prep(device)
    y = np.asarray(cache["labels"], dtype=np.float64)
    norm = D.load_norm()
    monitor = np.asarray(sup["monitor"], dtype=np.int64)
    q_target = np.asarray(sup["q_target"], dtype=np.int64)
    q_background = np.asarray(sup["q_background"], dtype=np.int64)

    if args.score_only:
        ck = torch.load(out / "best_monitor.pt", map_location="cpu", weights_only=False)
        model, info = R.build_model(norm, args.seed, device)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        preds = {}
        for name, rows in (("monitor", monitor), ("q_target", q_target),
                           ("q_background", q_background)):
            pred, mae = evaluate(model, rows, cache, device)
            preds[name] = {"rows": rows.tolist(), "pred": pred.tolist(), "mae": float(mae)}
        (out / "scores.json").write_text(json.dumps({
            "checkpoint": str(out / "best_monitor.pt"),
            "checkpoint_epoch": int(ck["epoch"]),
            "checkpoint_sha256": C.sha256_file(out / "best_monitor.pt"),
            "scores": preds,
        }, indent=1))
        print(json.dumps({k: v["mae"] for k, v in preds.items()}, indent=1))
        return 0

    base_rows = np.asarray(sup["b_rows"], dtype=np.int64)
    donor_rows = np.asarray(sup["d_minus"], dtype=np.int64)
    fit_rows = np.asarray(sup["fit_minus"], dtype=np.int64)
    steps = math.ceil(len(fit_rows) / BATCH)
    total = steps * args.epochs
    model, info = R.build_model(norm, args.seed, device)
    init_hash = state_digest(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    run_meta = {
        "model": "RBO", "fit": "minus", "seed": int(args.seed),
        "parameters": info["parameters"], "optimizer_parameters": int(n_opt),
        "ssl": info["ssl"], "init_state_sha256": init_hash,
        "rows": int(len(fit_rows)), "base_rows": int(len(base_rows)),
        "donor_rows": int(len(donor_rows)), "steps_per_epoch": int(steps),
        "epochs": int(args.epochs), "updates": int(total),
        "recipe": {"optimizer": "AdamW", "lr": LR, "min_lr": MIN_LR, "wd": WD,
                   "betas": BETAS, "eps": EPS, "warmup": WARMUP, "clip": CLIP,
                   "batch": BATCH, "loss": "L1 raw-y", "fp32": True, "dropout": 0.0},
        "data": {"fit": "zinc_composition_transfer_v1 fit_minus",
                 "monitor": "M (1000 train-only rows, target-key free)",
                 "support_npz_sha256": C.sha256_file(
                     _SSUF / "zinc_composition_transfer_v1" / "cache" / "support_sets.npz")},
        "environment": {"device": str(device), "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "git_commit": C.git_commit() if hasattr(C, "git_commit") else None},
    }
    C.log(run_meta)
    curve = []
    best = {"mae": float("inf"), "epoch": -1}
    started = time.perf_counter()
    rng_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = ZS.fit_batches(base_rows, donor_rows, epoch, args.seed)
        err, cnt, rate = 0.0, 0, LR
        for k_batch, (b_rows, d_rows) in enumerate(batches):
            rows = np.concatenate([b_rows, d_rows])
            batch = R.collate(rows, cache, prep, device)
            target = torch.as_tensor(y[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            rate = C.cosine_with_warmup_lr((epoch - 1) * steps + k_batch + 1, total,
                                           WARMUP, LR, MIN_LR)
            for group in opt.param_groups:
                group["lr"] = rate
            opt.step()
            err += float((pred.detach().view(-1) - target).abs().sum())
            cnt += len(rows)
        mon_pred, mon_mae = evaluate(model, monitor, cache, device)
        rec = {"epoch": epoch, "online_train_mae": err / cnt, "monitor_mae": float(mon_mae),
               "lr": rate, "steps": steps, "elapsed_s": time.perf_counter() - started}
        curve.append(rec)
        if mon_mae < best["mae"]:
            best = {"mae": float(mon_mae), "epoch": int(epoch)}
            C.atomic_torch_save({"state_dict": C.cpu_state(model), "epoch": epoch,
                                 "monitor_mae": float(mon_mae), "model": "RBO",
                                 "seed": int(args.seed)}, out / "best_monitor.pt")
        if epoch % 10 == 0 or epoch == 1 or args.smoke:
            C.log(f"[RBO s{args.seed}] ep={epoch:03d} monitor={mon_mae:.6f} "
                  f"train={err / cnt:.6f} lr={rate:.2e} t={rec['elapsed_s']:.0f}s")
        if args.smoke and epoch >= 3:
            break
    C.atomic_torch_save({"state_dict": C.cpu_state(model), "epoch": epoch,
                         "monitor_mae": float(curve[-1]["monitor_mae"]), "model": "RBO",
                         "seed": int(args.seed)}, out / "final.pt")
    run_meta.update({
        "best_monitor": best, "curve": curve,
        "best_sha256": C.sha256_file(out / "best_monitor.pt"),
        "final_sha256": C.sha256_file(out / "final.pt"),
        "wall_seconds": time.perf_counter() - started,
    })
    C.write_json(out / "train.json", run_meta)
    print(json.dumps({"best_monitor": best,
                      "last_monitor": curve[-1]["monitor_mae"],
                      "wall_s": run_meta["wall_seconds"],
                      "parameters": info["parameters"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
