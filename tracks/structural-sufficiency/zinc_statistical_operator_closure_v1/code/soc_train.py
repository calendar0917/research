"""Train SOC-v1 arms on the frozen BJOC screen (Fit8k/Monitor) or full 10k.

Modes
-----
screen : Fit8k rows, checkpoint selection on the frozen Monitor 1k split,
         47,400 optimizer updates (the champion horizon).  Official valid is
         evaluated every 20 epochs for description only.  Locked and test are
         never loaded.
full   : all 10,000 official-train rows, checkpoint selection on the official
         valid 1k split, same 47,400-update horizon.  Test is never loaded here;
         it is read only by ``soc_score.py`` after a frozen gate.

The data pipeline, normalization, recipe, SSL state and metric are the frozen
DCC-L ones.  The only change is the model returned by the arm builder.
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
for _p in (str(_HERE), str(_SUFF / "zinc_arch_discovery_v1" / "code"),
           str(_SUFF / "zinc_dual_channel_composer_v1" / "code"),
           str(_SUFF / "zinc_dccl_bottleneck_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_runner as DR  # noqa: E402
import soc_common as SC  # noqa: E402

ROUND_ID = SC.ROUND_ID


def train(arm: str, seed: int, device, *, mode: str = "screen", fit: str = "fit8k",
          total_updates: int = SC.TOTAL_UPDATES, eval_every: int = 20,
          out_root: Path | None = None, force: bool = False,
          tag: str | None = None) -> dict:
    if mode not in ("screen", "full"):
        raise ValueError(mode)
    cfg = SC.load_config()
    rows = SC.split_rows()
    if mode == "screen":
        train_rows = rows[fit]
    else:
        train_rows = rows["all"]
    monitor_rows = rows["monitor"]
    valid_rows = np.arange(1000, dtype=np.int64)

    name = tag or f"{mode}_{fit if mode == 'screen' else '10k'}_{arm}_s{seed}"
    runs_root = Path(out_root) if out_root is not None else SC.RUNS
    out = runs_root / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        return json.loads((out / "train.json").read_text())

    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    norm = SC.fit_norm(train_cache, train_rows)

    builder = SC.make_builder(arm, cfg)
    model, model_info = builder("P3N", norm, seed, device)
    params = C.parameter_count(model)
    expected = int(cfg["expected_params"][arm])
    if params != expected:
        raise RuntimeError(f"parameter count {params} != expected {expected} for arm {arm}")

    batch_size = int(SC.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    n_epochs = math.ceil(total_updates / steps_per_epoch)
    warmup = min(int(SC.RECIPE["warmup_steps"]), max(1, total_updates // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=SC.RECIPE["lr"],
                                  weight_decay=SC.RECIPE["weight_decay"])
    gen = torch.Generator().manual_seed(int(seed) + 101)
    collate_ = DR.collate_p3n

    curve: list[dict] = []
    best = {"mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    valid_curve = []
    global_step = 0
    started = time.perf_counter()
    last_lr = float(SC.RECIPE["lr"])

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= total_updates:
                break
            batch_rows = train_rows[order[start:start + batch_size]]
            batch = T1._to_device(collate_(train_cache, batch_rows, "P3N", valid=False), device)
            target = torch.as_tensor(y_train[batch_rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), SC.RECIPE["grad_clip"])
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, total_updates, warmup,
                                              float(SC.RECIPE["lr"]), float(SC.RECIPE["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)
        _, monitor_mae = SC.evaluate(model, train_cache, monitor_rows, device=device)
        entry = {"epoch": int(epoch), "step": int(global_step),
                 "online_train_mae": float(online), "monitor_mae": float(monitor_mae),
                 "lr": float(last_lr), "seconds": float(time.perf_counter() - t_ep)}
        if epoch % eval_every == 0 or global_step >= total_updates:
            _, vmae = SC.evaluate(model, valid_cache, valid_rows, device=device)
            entry["official_valid_mae"] = float(vmae)
            valid_curve.append({"epoch": int(epoch), "step": int(global_step),
                                "official_valid_mae": float(vmae)})
        curve.append(entry)

        if mode == "screen":
            key_mae = float(monitor_mae)
            selection = "monitor"
            best_name = "best_monitor.pt"
        else:
            key_mae = float(entry.get("official_valid_mae", float("inf")))
            selection = "official_valid"
            best_name = "best_valid.pt"
        # In full mode, only epochs with a valid evaluation can become the best.
        if key_mae < best["mae"]:
            best = {"mae": key_mae, "epoch": int(epoch), "step": int(global_step),
                    "selection": selection}
            best_state = C.cpu_state(model)
            C.atomic_torch_save({"state_dict": best_state, "epoch": int(epoch),
                                 "step": int(global_step), "mae": key_mae,
                                 "selection": selection, "arm": arm, "seed": int(seed),
                                 "fit": fit if mode == "screen" else "10k",
                                 "mode": mode}, out / best_name)
        if epoch % 10 == 0 or global_step >= total_updates:
            SC.log(f"[{ROUND_ID} {name}] ep={epoch} step={global_step} "
                   f"monitor={monitor_mae:.6f} online={online:.6f} best={best['mae']:.6f}"
                   f"@{best['epoch']} lr={last_lr:.2e} {entry['seconds']:.2f}s")
        if global_step >= total_updates:
            break

    final_state = C.cpu_state(model)
    C.atomic_torch_save({"state_dict": final_state, "epoch": int(len(curve)),
                         "step": int(global_step), "arm": arm, "seed": int(seed),
                         "fit": fit if mode == "screen" else "10k", "mode": mode,
                         "selection": "final"}, out / "final.pt")
    wall = time.perf_counter() - started

    reads = {}
    for tag_, state in (("best", best_state), ("final", final_state)):
        C.load_state(model, state, device)
        pm, mm = SC.evaluate(model, train_cache, monitor_rows, device=device)
        pv, mv = SC.evaluate(model, valid_cache, valid_rows, device=device)
        reads[tag_] = {"monitor_mae": float(mm), "official_valid_mae": float(mv),
                       "monitor_pred": pm.tolist(), "official_valid_pred": pv.tolist()}
        C.atomic_torch_save({"monitor_pred": pm, "official_valid_pred": pv,
                             "monitor_rows": monitor_rows,
                             "official_valid_rows": valid_rows,
                             "arm": arm, "seed": int(seed), "checkpoint": tag_},
                            out / f"predictions_{tag_}.pt")

    tail = curve[-max(1, len(curve) // 10):]
    info = {
        "round": ROUND_ID,
        "name": name,
        "mode": mode,
        "arm": arm,
        "seed": int(seed),
        "fit": fit if mode == "screen" else "10k",
        "n_train_rows": int(len(train_rows)),
        "parameters": int(params),
        "model_info": {k: v for k, v in model_info.items() if k != "state_dict"},
        "norm": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()},
        "recipe": dict(SC.RECIPE),
        "total_updates": int(total_updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "split_hashes": {"train_rows": SC.sha256_rows(train_rows),
                         "monitor_rows": SC.sha256_rows(monitor_rows),
                         "valid_rows": SC.sha256_rows(valid_rows)},
        "best": best,
        "tail10_median_monitor_mae": float(np.median([e["monitor_mae"] for e in tail])),
        "tail10_median_online_train_mae": float(np.median([e["online_train_mae"] for e in tail])),
        "reads": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_pred")}
                  for k, v in reads.items()},
        "wall_seconds": float(wall),
        "gpu_hours": SC.gpu_hours(wall),
        "environment": SC.environment(),
        "official_valid_loaded": True,
        "official_test_loaded": False,
        "locked_loaded": False,
        "curve": curve,
        "official_valid_curve": valid_curve,
    }
    SC.write_json(out / "train.json", info)
    SC.log(f"[{ROUND_ID} {name}] DONE best={best['mae']:.6f}@{best['epoch']} "
           f"monitor(best)={reads['best']['monitor_mae']:.6f} "
           f"valid(best)={reads['best']['official_valid_mae']:.6f} "
           f"gpu_h={info['gpu_hours']:.3f}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["R", "T", "CTRL"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="screen", choices=["screen", "full"])
    p.add_argument("--fit", default="fit8k", choices=["fit4k", "fit8k"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-updates", type=int, default=SC.TOTAL_UPDATES)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.smoke:
        info = train(args.arm, args.seed, device, mode="screen", fit=args.fit,
                     total_updates=64, eval_every=4, force=True,
                     out_root=Path("/tmp/soc_smoke"), tag=f"smoke_{args.arm}_s{args.seed}")
        print(json.dumps({k: info[k] for k in ("name", "total_updates", "epochs_run",
                                               "best", "reads", "gpu_hours")}, indent=2))
        return 0
    train(args.arm, args.seed, device, mode=args.mode, fit=args.fit,
          total_updates=args.total_updates, eval_every=args.eval_every,
          out_root=Path(args.out_root) if args.out_root else None,
          tag=args.tag, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
