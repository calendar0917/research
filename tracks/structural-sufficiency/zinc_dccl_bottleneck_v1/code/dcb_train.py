"""Phase B: native DCC-L trained on Fit4k vs Fit8k at a matched optimizer-update
budget (T = 47,400 = the champion's 600 epochs x 79 updates at 10k).

Everything about the model, the SSL initialisation, the recipe and the update
schedule is the historical one; only three things change:

* the training rows are the frozen nested BJOC split rows (Fit4k / Fit8k),
* the standardisation statistics are computed from the same Fit rows only,
* checkpoint selection uses the frozen Monitor split (official valid is
  evaluated only every ``--eval-every`` epochs and is descriptive).

The Locked-dev split is read once, at the end, for the two frozen checkpoints
(best-monitor and final).  Official test is never constructed.
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
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
_REPO = _SUFF.parents[1]
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_DCC = _SUFF / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_runner as DR  # noqa: E402

ROUND_ID = "zinc-dccl-bottleneck-v1"
BJOC = _REPO / "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1"
RUNS = _ROUND / "runs"
ANALYSIS = _ROUND / "analysis"

RECIPE = dict(optimizer="adamw", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-5,
              warmup_steps=500, dropout=0.0, grad_clip=5.0, batch_size=128,
              valid_batch_size=256, loss="l1")
TOTAL_UPDATES = 47400  # 600 epochs x 79 updates at 10k = the champion horizon


def split_rows() -> dict:
    s = json.loads((BJOC / "split_a.json").read_text())
    f4 = json.loads((BJOC / "fit4k_subset.json").read_text())
    return {
        "fit4k": np.asarray(f4["rows"], dtype=np.int64),
        "fit8k": np.asarray(s["fit_rows"], dtype=np.int64),
        "monitor": np.asarray(s["monitor_rows"], dtype=np.int64),
        "locked": np.asarray(s["locked_rows"], dtype=np.int64),
        "hashes": {"fit4k": f4["fit4k_sha256_rows"], "fit8k": s["fit8k_sha256_rows"],
                   "monitor": s["monitor_sha256_rows"], "locked": s["locked_sha256_rows"]},
    }


def fit_only_norm(train: dict, rows: np.ndarray) -> dict:
    """Standardisation from the Fit rows only (never valid / monitor / locked)."""
    g = np.asarray(train["global_feat"], dtype=np.float64)[rows]
    gmean = g.mean(0).astype(np.float32)
    gscale = np.maximum(g.std(0).astype(np.float32), 1e-6)

    ap = np.asarray(train["atom_ptr"])
    atom_rows = np.concatenate([np.arange(ap[i], ap[i + 1]) for i in rows])
    rs = np.asarray(train["root_scalar"], dtype=np.float64)[atom_rows]
    rmean = rs.mean(0).astype(np.float32)
    rscale = np.maximum(rs.std(0).astype(np.float32), 1e-6)

    bp = np.asarray(train["block_ptr"])
    block_rows = np.concatenate([np.arange(bp[i], bp[i + 1]) for i in rows])
    bs = np.asarray(train["block_scalar"], dtype=np.float64)[block_rows]
    bmean = bs.mean(0).astype(np.float32)
    bscale = np.maximum(bs.std(0).astype(np.float32), 1e-6)

    return {
        "source": f"computed_from_fit_rows_n={len(rows)}",
        "global_mean": gmean, "global_scale": gscale,
        "root_scalar_mean": rmean, "root_scalar_scale": rscale,
        "block_scalar_mean": bmean, "block_scalar_scale": bscale,
        "fit_split": "bjoc-internal-fit", "valid_or_test_used": False,
        "n_fit_rows": int(len(rows)),
    }


def train(fit: str, seed: int, device, *, total_updates: int = TOTAL_UPDATES,
          eval_every: int = 20, force: bool = False) -> dict:
    rows = split_rows()
    train_rows = rows[fit]
    monitor_rows = rows["monitor"]
    out = RUNS / f"{fit}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        return json.loads((out / "train.json").read_text())

    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    norm = fit_only_norm(train_cache, train_rows)

    builder = DR.make_builder("L")
    model, model_info = builder("P3N", norm, seed, device)
    params = C.parameter_count(model)
    if params != 494193:
        raise RuntimeError(f"parameter count {params} != 494193")

    batch_size = int(RECIPE["batch_size"])
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    n_epochs = math.ceil(total_updates / steps_per_epoch)
    warmup = min(int(RECIPE["warmup_steps"]), max(1, total_updates // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"],
                                  weight_decay=RECIPE["weight_decay"])
    gen = torch.Generator().manual_seed(int(seed) + 101)
    collate_ = DR.collate_p3n

    curve: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    official_valid_curve = []
    global_step = 0
    started = time.perf_counter()
    last_lr = float(RECIPE["lr"])
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= total_updates:
                break
            batch_rows = train_rows[order[start:start + batch_size]]
            batch = T1._to_device(collate_(train_cache, batch_rows, "P3N", valid=False),
                                  device)
            target = torch.as_tensor(y_train[batch_rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"])
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, total_updates, warmup,
                                              float(RECIPE["lr"]), float(RECIPE["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)
        _, monitor_mae = T1.evaluate(model, train_cache, monitor_rows, candidate="P3N",
                                     device=device, batch_size=int(RECIPE["valid_batch_size"]),
                                     collate_fn=collate_)
        entry = {"epoch": int(epoch), "step": int(global_step),
                 "online_train_mae": float(online), "monitor_mae": float(monitor_mae),
                 "lr": float(last_lr), "seconds": float(time.perf_counter() - t_ep)}
        if epoch % eval_every == 0 or global_step >= total_updates:
            _, vmae = T1.evaluate(model, valid_cache, np.arange(valid_cache["n_molecules"]),
                                  candidate="P3N", device=device,
                                  batch_size=int(RECIPE["valid_batch_size"]), collate_fn=collate_)
            entry["official_valid_mae"] = float(vmae)
            official_valid_curve.append({"epoch": int(epoch), "step": int(global_step),
                                         "official_valid_mae": float(vmae)})
        curve.append(entry)
        if monitor_mae < best["monitor_mae"]:
            best = {"monitor_mae": float(monitor_mae), "epoch": int(epoch), "step": int(global_step)}
            best_state = C.cpu_state(model)
            C.atomic_torch_save({"state_dict": best_state, "epoch": int(epoch),
                                 "step": int(global_step), "monitor_mae": float(monitor_mae),
                                 "candidate": "P3N", "seed": int(seed), "fit": fit,
                                 "selection": "monitor"}, out / "best_monitor.pt")
        if epoch % 10 == 0 or global_step >= total_updates:
            C.log(f"[{ROUND_ID} {fit} s{seed}] ep={epoch} step={global_step} "
                  f"monitor={monitor_mae:.6f} online={online:.6f} best={best['monitor_mae']:.6f}"
                  f"@{best['epoch']} lr={last_lr:.2e} {entry['seconds']:.2f}s")
        if global_step >= total_updates:
            break

    final_state = C.cpu_state(model)
    C.atomic_torch_save({"state_dict": final_state, "epoch": int(len(curve)),
                         "step": int(global_step), "candidate": "P3N", "seed": int(seed),
                         "fit": fit, "selection": "final"}, out / "final.pt")
    wall = time.perf_counter() - started

    # ---- one frozen read of the locked split, for the two declared checkpoints
    lock = rows["locked"]
    reads = {}
    for tag, state in (("best_monitor", best_state), ("final", final_state)):
        C.load_state(model, state, device)
        pred_lock, mae_lock = T1.evaluate(model, train_cache, lock, candidate="P3N",
                                         device=device,
                                         batch_size=int(RECIPE["valid_batch_size"]),
                                         collate_fn=collate_)
        pred_mon, mae_mon = T1.evaluate(model, train_cache, monitor_rows, candidate="P3N",
                                       device=device,
                                       batch_size=int(RECIPE["valid_batch_size"]),
                                       collate_fn=collate_)
        pred_val, mae_val = T1.evaluate(model, valid_cache, np.arange(valid_cache["n_molecules"]),
                                        candidate="P3N", device=device,
                                        batch_size=int(RECIPE["valid_batch_size"]),
                                        collate_fn=collate_)
        reads[tag] = {"locked_mae": float(mae_lock), "monitor_mae": float(mae_mon),
                      "official_valid_mae": float(mae_val),
                      "locked_pred": pred_lock.tolist(),
                      "monitor_pred": pred_mon.tolist(),
                      "official_valid_pred": pred_val.tolist()}

    tail = curve[-max(1, len(curve) // 10):]
    info = {
        "round": ROUND_ID,
        "fit": fit,
        "seed": int(seed),
        "parameters": params,
        "model_info": {k: v for k, v in model_info.items() if k != "state_dict"},
        "norm": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()},
        "recipe": dict(RECIPE),
        "total_updates": int(total_updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "n_train_rows": int(len(train_rows)),
        "split_hashes": rows["hashes"],
        "best_monitor": best,
        "tail10_median_monitor_mae": float(np.median([e["monitor_mae"] for e in tail])),
        "tail10_median_online_train_mae": float(np.median([e["online_train_mae"] for e in tail])),
        "eval_reads": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_pred")}
                       for k, v in reads.items()},
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "official_test_loaded": False,
        "curve": curve,
        "official_valid_curve": official_valid_curve,
    }
    C.write_json(out / "train.json", info)
    for tag, v in reads.items():
        C.atomic_torch_save(
            {"locked_pred": np.asarray(v["locked_pred"]),
             "monitor_pred": np.asarray(v["monitor_pred"]),
             "official_valid_pred": np.asarray(v["official_valid_pred"]),
             "locked_rows": rows["locked"], "monitor_rows": rows["monitor"],
             "fit": fit, "seed": int(seed), "checkpoint": tag},
            out / f"predictions_{tag}.pt")
    C.log(f"[{ROUND_ID} {fit} s{seed}] DONE bestMonitor={best['monitor_mae']:.6f}"
          f"@{best['epoch']} locked(best)={reads['best_monitor']['locked_mae']:.6f}"
          f" locked(final)={reads['final']['locked_mae']:.6f}"
          f" valid(best)={reads['best_monitor']['official_valid_mae']:.6f}"
          f" gpu_h={info['gpu_hours']:.3f}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", required=True, choices=["fit4k", "fit8k"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-updates", type=int, default=TOTAL_UPDATES)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.smoke:
        # CPU/GPU smoke: tiny budget, same code path, separate output dir.
        global RUNS
        RUNS = _ROUND / "runs_smoke"
        info = train(args.fit, args.seed, device, total_updates=64, eval_every=4, force=True)
        print(json.dumps({k: info[k] for k in ("fit", "total_updates", "epochs_run",
                                               "best_monitor", "eval_reads", "gpu_hours")},
                         indent=2))
        return 0
    train(args.fit, args.seed, device, total_updates=args.total_updates,
          eval_every=args.eval_every, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
