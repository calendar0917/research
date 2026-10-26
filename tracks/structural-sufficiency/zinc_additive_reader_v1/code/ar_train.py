"""Train the additive (A) or dense (D) reader on frozen Fit8k.

Identical supervised convention to ``zinc_dccl_bottleneck_v1/code/dcb_train.py``:
AdamW, lr 1e-3 -> 1e-5 per-step cosine, 500-step warm-up, wd 1e-5, raw-y L1,
batch 128, clip 5, FP32, dropout 0, 47,400 updates, per-epoch Monitor
evaluation, strict-minimum Monitor best checkpoint.  The only cache that can be
opened here is the official-train cache (Fit8k / Monitor / Locked rows).
Official valid and official test are never loaded.  Locked is not evaluated
during training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ar_common as C  # noqa: E402
import ar_models as M  # noqa: E402

for _p in (str(C.V1), str(C.DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_train as T1  # noqa: E402
import ad_common as AD  # noqa: E402
import dcc_runner as DR  # noqa: E402


def _to_device(batch, device):
    return T1._to_device(batch, device)


@torch.no_grad()
def evaluate_rows(model, cache, rows, device, batch_size=256) -> tuple[np.ndarray, float]:
    return T1.evaluate(model, cache, np.asarray(rows, dtype=np.int64), candidate="P3N",
                       device=device, batch_size=int(batch_size),
                       collate_fn=DR.collate_p3n)


def run(arm: str, seed: int, device, *, total_updates: int, smoke: bool = False,
        force: bool = False) -> dict:
    cfg = C.load_config()
    recipe = cfg["recipe"]
    rows = C.split_rows()
    train_rows = rows["fit8k"]
    monitor_rows = rows["monitor"]
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")

    cache = C.load_train_cache()
    y_train = np.asarray(cache["labels"], dtype=np.float64)
    norm = C.fit_only_norm(cache, train_rows)
    n_hash = C.norm_hash(norm)

    run_root = C.RUNS_SMOKE if smoke else C.RUNS
    tag = f"{arm}_s{int(seed)}"
    out = run_root / tag
    out.mkdir(parents=True, exist_ok=True)
    done_file = out / ("smoke.json" if smoke else "train.json")
    if done_file.exists() and not force:
        return C.read_json(done_file)

    model, model_info = M.build_arm(arm, norm, seed, device)
    params = C.parameter_count(model)
    expected = cfg["model"]["expected_parameters"]["additive_total" if arm == "A" else "dense_total"]
    if int(params) != int(expected):
        raise RuntimeError(f"parameter count {params} != frozen expectation {expected}")

    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    n_epochs = math.ceil(total_updates / steps_per_epoch)
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(recipe["lr"]),
                                  weight_decay=float(recipe["weight_decay"]))
    gen = torch.Generator().manual_seed(int(seed) + 101)
    collate_ = DR.collate_p3n

    curve: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    global_step = 0
    last_lr = float(recipe["lr"])
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= total_updates:
                break
            batch_rows = train_rows[order[start:start + batch_size]]
            batch = _to_device(collate_(cache, batch_rows, "P3N", valid=False), device)
            target = torch.as_tensor(y_train[batch_rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(
                global_step, total_updates, warmup,
                float(recipe["lr"]), float(recipe["min_lr"])
            )
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)
        _, monitor_mae = evaluate_rows(model, cache, monitor_rows, device,
                                       batch_size=int(recipe["valid_batch_size"]))
        entry = {
            "epoch": int(epoch), "step": int(global_step),
            "online_train_mae": float(online), "monitor_mae": float(monitor_mae),
            "lr": float(last_lr), "seconds": float(time.perf_counter() - t_ep),
        }
        curve.append(entry)
        if monitor_mae < best["monitor_mae"]:
            best = {"monitor_mae": float(monitor_mae), "epoch": int(epoch),
                    "step": int(global_step)}
            best_state = AD.cpu_state(model)
            C.atomic_torch_save(
                {"state_dict": best_state, "epoch": int(epoch), "step": int(global_step),
                 "monitor_mae": float(monitor_mae), "candidate": f"AdditiveReader_{arm}",
                 "arm": arm, "seed": int(seed), "fit": "fit8k",
                 "selection": "monitor", "config_sha256": C.sha256_file(C.CONFIG),
                 "git_commit": C.git_commit()},
                out / "best_monitor.pt",
            )
        if epoch % 10 == 0 or global_step >= total_updates:
            C.log(f"[{C.ROUND_ID} {tag}] ep={epoch} step={global_step} "
                  f"monitor={monitor_mae:.6f} online={online:.6f} "
                  f"best={best['monitor_mae']:.6f}@{best['epoch']} "
                  f"lr={last_lr:.2e} {entry['seconds']:.2f}s")
        if global_step >= total_updates:
            break

    if best_state is None:
        raise RuntimeError("no monitor checkpoint was produced")
    wall = time.perf_counter() - started
    final_state = AD.cpu_state(model)
    C.atomic_torch_save(
        {"state_dict": final_state, "epoch": int(len(curve)), "step": int(global_step),
         "candidate": f"AdditiveReader_{arm}", "arm": arm, "seed": int(seed),
         "fit": "fit8k", "selection": "final",
         "config_sha256": C.sha256_file(C.CONFIG), "git_commit": C.git_commit()},
        out / "final.pt",
    )

    # fixed-model fit / monitor measurements (no Locked, no official valid)
    reads = {}
    for ck_tag, state in (("best_monitor", best_state), ("final", final_state)):
        AD.load_state(model, state, device)
        monitor_pred, monitor_mae = evaluate_rows(model, cache, monitor_rows, device,
                                                  batch_size=int(recipe["valid_batch_size"]))
        fit_pred, fit_mae = evaluate_rows(model, cache, train_rows, device,
                                          batch_size=int(recipe["valid_batch_size"]))
        reads[ck_tag] = {
            "monitor_mae": float(monitor_mae),
            "train_fit_mae_eval_mode": float(fit_mae),
            "monitor_pred": monitor_pred,
            "fit_pred_hash": C.sha256_array(np.asarray(fit_pred, dtype=np.float32), "<f4"),
        }
        C.atomic_torch_save(
            {"monitor_pred": np.asarray(monitor_pred, dtype=np.float64),
             "monitor_rows": monitor_rows,
             "train_fit_mae_eval_mode": float(fit_mae),
             "fit": "fit8k", "seed": int(seed), "arm": arm, "checkpoint": ck_tag,
             "monitor_rows_sha256": rows["hashes"]["monitor"],
             "config_sha256": C.sha256_file(C.CONFIG),
             "git_commit": C.git_commit()},
            out / f"predictions_{ck_tag}.pt",
        )

    peak_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    epoch_seconds = [float(e["seconds"]) for e in curve]
    tail = curve[-max(1, len(curve) // 10):]
    info = {
        "round": C.ROUND_ID,
        "protocol_id": cfg["protocol_id"],
        "arm": arm,
        "candidate": f"AdditiveReader_{arm}",
        "fit": "fit8k",
        "seed": int(seed),
        "parameters": int(params),
        "model_info": model_info,
        "norm": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()},
        "norm_hash": n_hash,
        "recipe": dict(recipe),
        "total_updates_declared": int(cfg["recipe"]["total_updates"]),
        "total_updates_run": int(total_updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "n_train_rows": int(len(train_rows)),
        "n_monitor_rows": int(len(monitor_rows)),
        "split_hashes": rows["hashes"],
        "best_monitor": best,
        "tail10_median_monitor_mae": float(np.median([e["monitor_mae"] for e in tail])),
        "tail10_median_online_train_mae": float(np.median([e["online_train_mae"] for e in tail])),
        "eval_reads": {k: {kk: vv for kk, vv in v.items() if kk != "monitor_pred"}
                       for k, v in reads.items()},
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "peak_memory_mib": float(peak_mem),
        "epoch_seconds": epoch_seconds,
        "host": socket.gethostname(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "git_commit": C.git_commit(),
        "config_sha256": C.sha256_file(C.CONFIG),
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "locked_read": False,
        "monitor_used_for_selection": True,
        "curve": curve,
    }
    C.write_json(done_file, info)
    AD.write_curve_csv(out / "curve.csv", curve)
    C.log(f"[{C.ROUND_ID} {tag}] DONE bestMonitor={best['monitor_mae']:.6f}"
          f"@{best['epoch']} finalMonitor={reads['final']['monitor_mae']:.6f}"
          f" fit(best)={reads['best_monitor']['train_fit_mae_eval_mode']:.6f}"
          f" params={params} gpu_h={info['gpu_hours']:.4f}")

    if smoke:
        full_epochs = [s for e, s in zip(curve, epoch_seconds) if e["step"] % steps_per_epoch == 0]
        steady = float(np.median(full_epochs)) if full_epochs else float(np.median(epoch_seconds))
        projected_total = 753.0 * steady * 1.15 + 90.0
        smoke_info = {
            "arm": arm, "seed": int(seed), "updates": int(total_updates),
            "epochs": int(len(curve)), "epoch_seconds": epoch_seconds,
            "steady_full_epoch_seconds": steady,
            "projected_753_epoch_seconds": float(projected_total),
            "projected_gpu_hours_per_run": float(projected_total / 3600.0),
            "wall_seconds": float(wall),
            "gpu_hours": float(wall / 3600.0),
            "peak_memory_mib": float(peak_mem),
            "best_monitor_at_smoke_horizon": best,
            "grad_finite": True,
            "parameters": int(params),
            "device_name": info["device_name"],
            "host": info["host"],
            "git_commit": info["git_commit"],
            "config_sha256": info["config_sha256"],
            "official_test_loaded": False,
            "official_valid_loaded": False,
            "locked_read": False,
        }
        C.write_json(out / "smoke.json", smoke_info)
        C.log(json.dumps(smoke_info, indent=2))
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["A", "D"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = C.load_config()
    total = args.total_updates
    if total is None:
        total = 128 if args.smoke else int(cfg["recipe"]["total_updates"])
    run(args.arm, args.seed, torch.device(args.device), total_updates=int(total),
        smoke=bool(args.smoke), force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
