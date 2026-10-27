"""Train one SCSB arm (L / S / P) on the frozen Fit8k split.

Phase 1 only.  Official valid and official test are never loaded here.  The
recipe, batch order, initialisation and 47,400-update horizon are the frozen
DCC-L Fit8k recipe, so the L arm can be reproduced exactly and the S/P arms
differ only by the per-bond alpha passed to the model.
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

import scb_common as S
import scb_model as SM
import ad_common as C
import ad_train as T1
import dcc_runner as DR

ROUND_ID = "zinc-support-conditioned-backoff-v1"
ARMS = ("L", "S", "P")


def prepare_alpha(cache, support, fit_rows, monitor_rows, arm, per_atom):
    alpha_fit = S.alpha_for_rows(cache, support, fit_rows, leave_one_out_for=fit_rows)
    alpha_mon = S.alpha_for_rows(cache, support, monitor_rows)
    if arm == "P":
        salt = int(S.load_cfg()["permutation"]["salt"])
        alpha_fit = S.permute_alpha_within_molecules(cache, alpha_fit, fit_rows, salt)
        alpha_mon = S.permute_alpha_within_molecules(cache, alpha_mon, monitor_rows, salt)
    elif arm == "L":
        alpha_fit, alpha_mon = None, None
    return alpha_fit, alpha_mon


def train(
    arm: str,
    seed: int,
    fit: str,
    device,
    *,
    total_updates: int = 47400,
    eval_every: int = 20,
    out_root: Path | None = None,
    force: bool = False,
    max_epochs: int | None = None,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if fit != "fit8k":
        raise ValueError("Phase 1 only supports fit8k; Phase 2 is unlocked separately")
    rows = S.split_rows()
    fit_rows = rows["fit"]
    monitor_rows = rows["monitor"]
    out = (out_root or S.RUNS) / f"{arm}_{fit}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        info = S.read_json(out / "train.json")
        print(json.dumps({"resumed": True, "out": str(out),
                          "best_monitor": info["best_monitor"]}, indent=2))
        return info

    t_start = time.perf_counter()
    cache = S.load_train_cache()
    y_train = np.asarray(cache["labels"], dtype=np.float64)
    _, per_atom = S.build_signatures(cache)
    support = S.build_support(cache, fit_rows, per_atom)
    alpha_fit, alpha_mon = prepare_alpha(cache, support, fit_rows, monitor_rows, arm, per_atom)
    alpha_sha = S.sha256_array(alpha_fit) if alpha_fit is not None else None
    alpha_mon_sha = S.sha256_array(alpha_mon) if alpha_mon is not None else None

    norm = S.DB.fit_only_norm(cache, fit_rows)
    model, model_info = SM.build_scb(seed, norm)
    model.to(device)
    params = SM.param_count(model)
    if params != 494193:
        raise RuntimeError(f"parameter count {params} != 494193")

    recipe = dict(S.DB.RECIPE)
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(len(fit_rows) / batch_size)
    n_epochs = math.ceil(total_updates / steps_per_epoch)
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(recipe["lr"]),
                                  weight_decay=float(recipe["weight_decay"]))
    gen = torch.Generator().manual_seed(int(seed) + 101)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    collate_ = DR.collate_p3n

    def batch_alpha(alpha, batch_rows):
        if alpha is None:
            return None
        gidx = np.concatenate([np.arange(bp[r], bp[r + 1]) for r in batch_rows])
        return torch.as_tensor(alpha[gidx], dtype=torch.float32, device=device)

    curve: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    global_step = 0
    last_lr = float(recipe["lr"])

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(fit_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= total_updates:
                break
            batch_rows = fit_rows[order[start:start + batch_size]]
            batch = T1._to_device(collate_(cache, batch_rows, "P3N", valid=False), device)
            target = torch.as_tensor(y_train[batch_rows], dtype=torch.float32, device=device)
            a = batch_alpha(alpha_fit, batch_rows)
            pred = model(batch) if a is None else model(batch, a)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, total_updates, warmup,
                                              float(recipe["lr"]), float(recipe["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)

        model.eval()
        _, monitor_mae = S.eval_with_alpha(model, cache, monitor_rows, alpha_mon, device,
                                           batch_size=int(recipe["valid_batch_size"]))
        entry = {"epoch": int(epoch), "step": int(global_step),
                 "online_train_mae": float(online), "monitor_mae": float(monitor_mae),
                 "lr": float(last_lr), "seconds": float(time.perf_counter() - t_ep)}
        curve.append(entry)
        if monitor_mae < best["monitor_mae"]:
            best = {"monitor_mae": float(monitor_mae), "epoch": int(epoch), "step": int(global_step)}
            best_state = C.cpu_state(model)
            C.atomic_torch_save(
                {"state_dict": best_state, "epoch": int(epoch), "step": int(global_step),
                 "monitor_mae": float(monitor_mae), "arm": arm, "seed": int(seed),
                 "fit": fit, "selection": "monitor", "alpha_sha256": alpha_sha,
                 "alpha_monitor_sha256": alpha_mon_sha},
                out / "best_monitor.pt",
            )
        if epoch % 10 == 0 or global_step >= total_updates or epoch == 1:
            C.log(f"[{ROUND_ID} {arm} {fit} s{seed}] ep={epoch} step={global_step} "
                  f"monitor={monitor_mae:.6f} online={online:.6f} "
                  f"best={best['monitor_mae']:.6f}@{best['epoch']} lr={last_lr:.2e} "
                  f"{entry['seconds']:.2f}s")
        if global_step >= total_updates or (max_epochs is not None and epoch >= int(max_epochs)):
            break

    final_state = C.cpu_state(model)
    C.atomic_torch_save(
        {"state_dict": final_state, "epoch": int(len(curve)), "step": int(global_step),
         "arm": arm, "seed": int(seed), "fit": fit, "selection": "final",
         "alpha_sha256": alpha_sha, "alpha_monitor_sha256": alpha_mon_sha},
        out / "final.pt",
    )
    wall = time.perf_counter() - t_start

    # Monitor predictions for the best checkpoint (diagnostic only; no Locked,
    # no official valid, no official test).
    C.load_state(model, best_state, device)
    model.eval()
    mon_pred, mon_mae = S.eval_with_alpha(model, cache, monitor_rows, alpha_mon, device,
                                          batch_size=int(recipe["valid_batch_size"]))
    C.atomic_torch_save(
        {"monitor_pred": mon_pred, "monitor_rows": monitor_rows, "monitor_mae": float(mon_mae),
         "arm": arm, "seed": int(seed), "fit": fit, "checkpoint": "best_monitor"},
        out / "predictions_monitor.pt",
    )

    if alpha_fit is not None:
        fit_bonds = np.concatenate([np.arange(bp[r], bp[r + 1]) for r in fit_rows])
        a_fit_stats = {"fit_mean": float(alpha_fit[fit_bonds].mean()),
                       "fit_frac_zero": float(np.mean(alpha_fit[fit_bonds] == 0.0)),
                       "fit_median": float(np.median(alpha_fit[fit_bonds]))}
    else:
        a_fit_stats = {"fit_mean": None, "fit_frac_zero": None, "fit_median": None}
    tail = curve[-max(1, len(curve) // 10):]
    info = {
        "round": ROUND_ID,
        "arm": arm,
        "fit": fit,
        "seed": int(seed),
        "parameters": params,
        "model_info": {k: v for k, v in model_info.items() if k != "state_dict"},
        "norm": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()},
        "recipe": recipe,
        "total_updates": int(total_updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "n_train_rows": int(len(fit_rows)),
        "split_hashes": rows["hashes"],
        "support_sha256": support["meta"]["df_sha256"],
        "alpha_sha256": alpha_sha,
        "alpha_monitor_sha256": alpha_mon_sha,
        "alpha_stats": {"arm": arm, "none": alpha_fit is None, **a_fit_stats},
        "best_monitor": best,
        "monitor_mae_at_best_recomputed": float(mon_mae),
        "tail10_median_monitor_mae": float(np.median([e["monitor_mae"] for e in tail])),
        "tail10_median_online_train_mae": float(np.median([e["online_train_mae"] for e in tail])),
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_read": False,
        "curve": curve,
    }
    C.write_json(out / "train.json", info)
    C.log(f"[{ROUND_ID} {arm} {fit} s{seed}] DONE bestMonitor={best['monitor_mae']:.6f}"
          f"@{best['epoch']} steps={global_step} gpu_h={info['gpu_hours']:.3f}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, required=True, choices=[0, 1])
    p.add_argument("--fit", default="fit8k", choices=["fit8k"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-updates", type=int, default=47400)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    out_root = Path(args.out_dir) if args.out_dir else (S.RUNS if not args.smoke else S.ROUND / "runs_smoke")
    if args.smoke:
        info = train(args.arm, args.seed, args.fit, torch.device(args.device),
                     total_updates=64, eval_every=4, out_root=out_root, force=True)
        print(json.dumps({k: info[k] for k in ("arm", "seed", "total_updates", "epochs_run",
                                               "best_monitor", "gpu_hours")}, indent=2))
        return 0
    train(args.arm, args.seed, args.fit, torch.device(args.device),
          total_updates=args.total_updates, eval_every=args.eval_every,
          out_root=out_root, force=args.force, max_epochs=args.max_epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
