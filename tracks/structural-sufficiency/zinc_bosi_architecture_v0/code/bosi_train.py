"""BOSI v0 formal trainer — one arm, one seed, the frozen DCC-L Fit8k recipe.

Identical to ``zinc_dccl_bottleneck_v1/code/dcb_train.py`` in every declared
recipe element: Fit8k rows, Fit-only standardisation, AdamW 1e-3 -> 1e-5
per-step cosine, 500-step warmup, wd 1e-5, clip 5, raw-y L1, batch 128,
47,400 updates, batch order from ``torch.Generator().manual_seed(seed + 101)``,
best checkpoint = strict minimum Monitor MAE.

Differences, all of them declared: the model body (BOSI), and the fact that this
script never loads the official-valid cache and never reads Locked.  Locked is a
separate, freeze-gated step (``bosi_locked.py``).

Usage:
    python .../bosi_train.py --arm R --seed 0 --device cuda:0
    python .../bosi_train.py --arm B --seed 0 --device cuda:0
    python .../bosi_train.py --arm R --seed 0 --device cpu --smoke
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bosi_common as BC  # noqa: E402  (also puts ad_common on the path)
import ad_common as C  # noqa: E402
import bosi_data as BD  # noqa: E402
import bosi_models as BM  # noqa: E402


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def load_relation_cache() -> dict:
    path = BC.CACHE / "bosi_rel_train.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing relation cache {path}; run bosi_prep.py")
    return torch.load(path, map_location="cpu", weights_only=False)


def grad_norms(model) -> dict:
    out = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = name.split(".")[0]
        out[key] = out.get(key, 0.0) + float(param.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(out.items())}


def train(arm: str, seed: int, device, *, total_updates: int = BC.RECIPE["total_updates"],
          eval_every: int = 1, smoke: bool = False, force: bool = False) -> dict:
    rows = BC.split_rows()
    train_rows = rows["fit8k"]
    monitor_rows = rows["monitor"]
    runs_dir = BC.RUNS_SMOKE if smoke else BC.RUNS
    out = runs_dir / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        return json.loads((out / "train.json").read_text())

    cache = BC.load_train_cache()
    rel = load_relation_cache()
    y_train = np.asarray(cache["labels"], dtype=np.float64)
    y_monitor = np.asarray(cache["labels"], dtype=np.float64)[monitor_rows]
    norm = BC.fit_only_norm(cache, train_rows)

    # arm-neutral construction: the same RNG stream and therefore the same
    # initial state for R and B at a given seed.
    model, model_info = BM.build_model(norm, "R", seed, device)
    model.arm = arm
    model_info["arm"] = arm
    params = C.parameter_count(model)
    if params != BC.EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameters {params} != {BC.EXPECTED_PARAMETERS}")

    batch_size = int(BC.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    n_epochs = math.ceil(total_updates / steps_per_epoch)
    warmup = min(int(BC.RECIPE["warmup_steps"]), max(1, total_updates // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=BC.RECIPE["lr"],
                                  weight_decay=BC.RECIPE["weight_decay"])
    gen = torch.Generator().manual_seed(int(seed) + 101)

    curve: list[dict] = []
    grad_log: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    global_step = 0
    started = time.perf_counter()
    last_lr = float(BC.RECIPE["lr"])
    first_batch_rows = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= total_updates:
                break
            batch_rows = train_rows[order[start:start + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = batch_rows.tolist()
            batch = _to_device(BD.collate(cache, rel, batch_rows), device)
            target = torch.as_tensor(y_train[batch_rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if global_step in (0, 1, 99, 999):
                grad_log.append({"step": global_step, "norms": grad_norms(model)})
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), BC.RECIPE["grad_clip"])
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, total_updates, warmup,
                                              float(BC.RECIPE["lr"]),
                                              float(BC.RECIPE["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)

        monitor_pred = BC.predict(model, cache, rel, monitor_rows, device=device,
                                  batch_size=int(BC.RECIPE["eval_batch_size"]))
        monitor_mae = BC.mae(monitor_pred, y_monitor)
        entry = {"epoch": int(epoch), "step": int(global_step),
                 "online_train_mae": float(online), "monitor_mae": float(monitor_mae),
                 "lr": float(last_lr), "seconds": float(time.perf_counter() - t_ep)}
        curve.append(entry)
        if monitor_mae < best["monitor_mae"]:
            best = {"monitor_mae": float(monitor_mae), "epoch": int(epoch),
                    "step": int(global_step)}
            best_state = C.cpu_state(model)
            C.atomic_torch_save({"state_dict": best_state, "epoch": int(epoch),
                                 "step": int(global_step), "monitor_mae": float(monitor_mae),
                                 "arm": arm, "seed": int(seed), "fit": "fit8k",
                                 "selection": "monitor"}, out / "best_monitor.pt")
        if epoch % 10 == 0 or global_step >= total_updates:
            C.log(f"[{BC.ROUND_ID} {arm} s{seed}] ep={epoch} step={global_step} "
                  f"monitor={monitor_mae:.6f} online={online:.6f} "
                  f"best={best['monitor_mae']:.6f}@{best['epoch']} "
                  f"lr={last_lr:.2e} {entry['seconds']:.2f}s")
        if global_step >= total_updates:
            break

    final_state = C.cpu_state(model)
    C.atomic_torch_save({"state_dict": final_state, "epoch": int(len(curve)),
                         "step": int(global_step), "arm": arm, "seed": int(seed),
                         "fit": "fit8k", "selection": "final"}, out / "final.pt")
    wall = time.perf_counter() - started

    reads = {}
    for tag, state in (("best_monitor", best_state), ("final", final_state)):
        C.load_state(model, state, device)
        pred_mon = BC.predict(model, cache, rel, monitor_rows, device=device,
                              batch_size=int(BC.RECIPE["eval_batch_size"]))
        reads[tag] = {"monitor_mae": BC.mae(pred_mon, y_monitor),
                      "monitor_pred": pred_mon.tolist()}
        C.atomic_torch_save({"monitor_pred": np.asarray(pred_mon),
                             "monitor_rows": monitor_rows,
                             "arm": arm, "seed": int(seed), "checkpoint": tag},
                            out / f"predictions_monitor_{tag}.pt")

    tail = curve[-max(1, len(curve) // 10):]
    info = {
        "round": BC.ROUND_ID,
        "fit": "fit8k",
        "seed": int(seed),
        "arm": arm,
        "parameters": params,
        "model_info": model_info,
        "norm": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()},
        "recipe": dict(BC.RECIPE),
        "total_updates": int(total_updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "n_train_rows": int(len(train_rows)),
        "split_hashes": rows["hashes"],
        "best_monitor": best,
        "monitor_reads": {k: {"monitor_mae": v["monitor_mae"]} for k, v in reads.items()},
        "tail10_median_monitor_mae": float(np.median([e["monitor_mae"] for e in tail])),
        "tail10_median_online_train_mae": float(np.median([e["online_train_mae"] for e in tail])),
        "first_batch_rows": first_batch_rows,
        "grad_norms": grad_log,
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_read": False,
        "smoke": bool(smoke),
        "curve": curve,
    }
    C.write_json(out / "train.json", info)
    C.log(f"[{BC.ROUND_ID} {arm} s{seed}] DONE bestMonitor={best['monitor_mae']:.6f}"
          f"@{best['epoch']} finalMonitor={reads['final']['monitor_mae']:.6f} "
          f"gpu_h={info['gpu_hours']:.3f}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=list(BM.ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.smoke:
        total = 63 if args.total_updates is None else int(args.total_updates)
        info = train(args.arm, args.seed, device, total_updates=total, smoke=True,
                     force=True)
        print(json.dumps({k: info[k] for k in ("arm", "total_updates", "epochs_run",
                                               "best_monitor", "monitor_reads",
                                               "grad_norms", "gpu_hours")}, indent=2))
        return 0
    total = BC.RECIPE["total_updates"] if args.total_updates is None else int(args.total_updates)
    train(args.arm, args.seed, device, total_updates=total, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
