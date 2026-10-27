"""COC-v1 Fit8k trainer.

The supervised loop is the frozen ``zinc_dccl_bottleneck_v1/code/dcb_train.py``
loop: same Fit8k rows, same seed-matched SSL initialisation, same Fit-only
normalisation, same batch generator, same 47,400-update cosine schedule, same
raw-y L1 and same strict Monitor checkpoint selection.  The only difference is
the model builder (``coc_model.build_model``).

This script never constructs the official valid or test cache and never reads
Locked labels.  It writes only what is needed for the pre-registered Monitor
stage.
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

import coc_common as CC  # noqa: E402
import coc_model as CM  # noqa: E402
import ad_common as C  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_runner as DR  # noqa: E402

TOTAL_UPDATES = 47400
RECIPE = {
    "optimizer": "adamw", "lr": 1.0e-3, "min_lr": 1.0e-5,
    "weight_decay": 1.0e-5, "warmup_steps": 500, "dropout": 0.0,
    "grad_clip": 5.0, "batch_size": 128, "valid_batch_size": 256, "loss": "l1",
}


def _mean(xs):
    return float(np.mean(xs)) if xs else 0.0


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return float(math.sqrt(total))


def _reader_local_grad_norm(model) -> float:
    w = model.reader.lins[0].weight.grad
    return float(w[:, 321:513].detach().float().pow(2).sum().sqrt()) if w is not None else 0.0


def _first_batch_hash(rows: np.ndarray) -> tuple[str, list[int]]:
    arr = np.asarray(rows, dtype=np.int64)
    return CC.sha256_array(arr), [int(v) for v in arr]


def train(seed: int, device, *, total_updates: int = TOTAL_UPDATES,
          out_dir: Path | None = None, profile: bool = False,
          force: bool = False) -> dict:
    cfg = CC.load_cfg()
    split = CC.split_rows()
    fit_rows = split["fit"]
    monitor_rows = split["monitor"]
    cache = CC.load_train_cache()
    y_train = np.asarray(cache["labels"], dtype=np.float64)
    norm = CC.fit_only_norm(cache, fit_rows)

    out = Path(out_dir) if out_dir is not None else (CC.RUNS / f"COC_fit8k_s{int(seed)}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        return json.loads((out / "train.json").read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    model, model_info = CM.build_model("COC", norm, int(seed), device)
    params = int(sum(p.numel() for _, p in model.named_parameters()))
    if params != int(cfg["model"]["parameter_count_expected"]):
        raise RuntimeError(f"parameter count {params} != expected")
    if model_info["initial_state_sha256"] != cfg["frozen_l_reference"]["initial_state_sha256"][str(int(seed))]:
        raise RuntimeError(f"initial state hash mismatch: {model_info['initial_state_sha256']}")
    if model_info["ssl"]["sha256"] != cfg["frozen_l_reference"]["ssl_sha256"][str(int(seed))]:
        raise RuntimeError("SSL hash mismatch")

    batch_size = int(RECIPE["batch_size"])
    steps_per_epoch = math.ceil(len(fit_rows) / batch_size)
    n_epochs = math.ceil(int(total_updates) / steps_per_epoch)
    warmup = min(int(RECIPE["warmup_steps"]), max(1, int(total_updates) // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"],
                                  weight_decay=RECIPE["weight_decay"])
    gen = torch.Generator().manual_seed(int(seed) + 101)
    collate_ = DR.collate_p3n

    curve: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    global_step = 0
    gradient_audit: dict = {}
    first_batch_rows = None
    last_lr = float(RECIPE["lr"])
    started = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(fit_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= int(total_updates):
                break
            batch_rows = fit_rows[order[start:start + batch_size]]
            batch = T1._to_device(collate_(cache, batch_rows, "P3N", valid=False), device)
            target = torch.as_tensor(y_train[batch_rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 1 and start == 0:
                first_batch_rows = batch_rows.copy()
                grad_list = [p.grad for p in model.parameters() if p.grad is not None]
                gradient_audit = {
                    "step": int(global_step),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "grad_tensors": int(len(grad_list)),
                    "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grad_list)),
                    "local_composer_grad_norm_step1": _grad_norm(model.local_composer.parameters()),
                    "reader_local_column_grad_norm_step1": _reader_local_grad_norm(model),
                    "first_batch_rows": int(len(batch_rows)),
                    "first_batch_rows_sha256": _first_batch_hash(batch_rows)[0],
                }
            if epoch == 2 and start == 0:
                gradient_audit["local_composer_grad_norm_step2_after_first_optimizer_step"] = (
                    _grad_norm(model.local_composer.parameters())
                )
                gradient_audit["reader_local_column_grad_norm_step2"] = _reader_local_grad_norm(model)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(RECIPE["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, int(total_updates), warmup,
                                              float(RECIPE["lr"]), float(RECIPE["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)
        _, _, monitor_mae = CC.eval_rows(
            model, cache, monitor_rows, device=device,
            batch_size=int(RECIPE["valid_batch_size"]))
        entry = {
            "epoch": int(epoch), "step": int(global_step),
            "online_train_mae": float(online), "monitor_mae": float(monitor_mae),
            "lr": float(last_lr), "seconds": float(time.perf_counter() - t_ep),
            "steps": int(min(steps_per_epoch, int(total_updates) - (global_step - 1))),
        }
        curve.append(entry)
        if monitor_mae < best["monitor_mae"]:
            best = {"monitor_mae": float(monitor_mae), "epoch": int(epoch), "step": int(global_step)}
            best_state = C.cpu_state(model)
            C.atomic_torch_save({
                "state_dict": best_state, "epoch": int(epoch), "step": int(global_step),
                "monitor_mae": float(monitor_mae), "candidate": "COC",
                "seed": int(seed), "fit": "fit8k", "selection": "monitor",
                "initial_state_sha256": model_info["initial_state_sha256"],
            }, out / "best_monitor.pt")
        if epoch % 10 == 0 or global_step >= int(total_updates) or profile:
            C.log(f"[COC s{seed}] ep={epoch} step={global_step} "
                  f"monitor={monitor_mae:.6f} online={online:.6f} "
                  f"best={best['monitor_mae']:.6f}@{best['epoch']} lr={last_lr:.2e} "
                  f"{entry['seconds']:.2f}s")
        if global_step >= int(total_updates):
            break

    wall = time.perf_counter() - started
    final_state = C.cpu_state(model)
    C.atomic_torch_save({
        "state_dict": final_state,
        "optimizer": optimizer.state_dict(),
        "gen_state": gen.get_state(),
        "epoch": int(len(curve)), "global_step": int(global_step),
        "best": best, "curve": curve, "candidate": "COC", "seed": int(seed),
        "fit": "fit8k", "selection": "final",
        "initial_state_sha256": model_info["initial_state_sha256"],
    }, out / "final.pt")

    # Final Monitor reads for the two declared checkpoint semantics.
    best_model = CM.load_model_from_state(norm, best_state, device)
    best_pred, _, best_mon = CC.eval_rows(
        best_model, cache, monitor_rows, device=device,
        batch_size=int(RECIPE["valid_batch_size"]))
    final_pred, _, final_mon = CC.eval_rows(
        model, cache, monitor_rows, device=device,
        batch_size=int(RECIPE["valid_batch_size"]))
    best_fit_pred, _, best_fit = CC.eval_rows(
        best_model, cache, fit_rows, device=device,
        batch_size=int(RECIPE["valid_batch_size"]))
    final_fit_pred, _, final_fit = CC.eval_rows(
        model, cache, fit_rows, device=device,
        batch_size=int(RECIPE["valid_batch_size"]))
    C.atomic_torch_save({
        "best_monitor_pred": best_pred, "final_pred": final_pred,
        "monitor_rows": monitor_rows, "seed": int(seed), "fit": "fit8k",
        "checkpoint_semantics": {"best_monitor": "strict min monitor MAE",
                                 "final": "step 47400"},
    }, out / "predictions_monitor.pt")

    tail = curve[-max(1, len(curve) // 10):]
    info = {
        "round": cfg["round"],
        "protocol_id": cfg["protocol_id"],
        "fit": "fit8k",
        "seed": int(seed),
        "parameters": params,
        "model_info": model_info,
        "norm": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()},
        "recipe": dict(RECIPE),
        "total_updates_requested": int(total_updates),
        "total_updates_completed": int(global_step),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "warmup_steps_effective": int(warmup),
        "n_train_rows": int(len(fit_rows)),
        "split_hashes": split["hashes"],
        "split_sha256": split["split_sha256"],
        "initial_state_sha256": model_info["initial_state_sha256"],
        "first_batch_rows": first_batch_rows.tolist() if first_batch_rows is not None else None,
        "first_batch_rows_sha256": (first_batch_rows is not None and _first_batch_hash(first_batch_rows)[0]),
        "gradient_audit": gradient_audit,
        "best_monitor": best,
        "tail10_median_monitor_mae": float(np.median([e["monitor_mae"] for e in tail])),
        "tail10_median_online_train_mae": float(np.median([e["online_train_mae"] for e in tail])),
        "monitor_mae_best_read": float(best_mon),
        "monitor_mae_final_read": float(final_mon),
        "fit_train_mae_at_best": float(best_fit),
        "fit_train_mae_at_final": float(final_fit),
        "wall_seconds": float(wall),
        "startup_seconds": float(started - t0),
        "final_eval_seconds": float(time.perf_counter() - started - wall),
        "total_seconds": float(time.perf_counter() - t0),
        "gpu_hours_run_wall": float((time.perf_counter() - t0) / 3600.0),
        "profile": bool(profile),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_labels_read": False,
        "curve": curve,
        "loaded_paths": CC.loaded_paths_audit()["loaded"],
    }
    CC.write_json(out / "train.json", info)
    C.log(f"[COC s{seed}] DONE bestMonitor={best['monitor_mae']:.6f}@{best['epoch']} "
          f"monitor(best)={best_mon:.6f} monitor(final)={final_mon:.6f} "
          f"fit(best)={best_fit:.6f} fit(final)={final_fit:.6f} "
          f"wall={info['wall_seconds']:.1f}s")
    return info


def profile(seed: int, device) -> dict:
    cfg = CC.load_cfg()
    out = CC.RUNS / f"profile_s{int(seed)}"
    info = train(int(seed), device, total_updates=189, out_dir=out, profile=True, force=True)
    epochs = [float(e["seconds"]) for e in info["curve"]]
    mean_epoch = float(np.mean(epochs))
    steady = float(np.mean(epochs[1:])) if len(epochs) > 1 else mean_epoch
    projected_train = 753.0 * mean_epoch
    projected_total = (753.0 * mean_epoch * 1.15 + float(info["startup_seconds"])
                       + float(info["final_eval_seconds"]) + 30.0)
    payload = {
        "protocol_id": cfg["protocol_id"],
        "seed": int(seed),
        "profile_updates": int(info["total_updates_completed"]),
        "profile_epochs": int(info["epochs_run"]),
        "steps_per_epoch": int(info["steps_per_epoch"]),
        "epoch_seconds_including_full_monitor": epochs,
        "mean_epoch_seconds": mean_epoch,
        "steady_epoch_seconds": steady,
        "startup_seconds": float(info["startup_seconds"]),
        "final_eval_seconds": float(info["final_eval_seconds"]),
        "projected_formal_train_seconds": projected_train,
        "projected_formal_total_seconds_with_15pct_and_overheads": projected_total,
        "projected_formal_gpu_hours": float(projected_total / 3600.0),
        "stop_threshold_gpu_hours": float(cfg["profile"]["stop_if_projected_single_run_a100_h_gt"]),
        "pass": bool(projected_total / 3600.0 <= float(cfg["profile"]["stop_if_projected_single_run_a100_h_gt"])),
        "parameters": int(info["parameters"]),
        "initial_state_sha256": info["initial_state_sha256"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_labels_read": False,
        "note": "Projection is 753 * mean full-epoch seconds * 1.15 + measured startup/final-eval + 30 s reserve; job occupancy is recorded separately from rr status.",
    }
    CC.write_json(CC.ANALYSIS / f"profile_s{int(seed)}.json", payload)
    C.log(json.dumps(payload, indent=2))
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True, choices=[0, 1])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--total-updates", type=int, default=TOTAL_UPDATES)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.profile:
        profile(args.seed, device)
    else:
        train(args.seed, device, total_updates=args.total_updates, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
