"""Formal trainer for ``zinc_adaptive_arch_search_v1`` — one arm, one seed.

The official PyG ZINC-subset protocol, verbatim:

* official train 10,000 rows, official valid 1,000 rows, no test;
* batch 128 -> ``ceil(10000/128) = 79`` steps/epoch, 600 epochs = **47,400**
  optimizer updates (never shortened);
* AdamW, lr 1e-3 -> 1e-5 per-step cosine, 500-update warm-up, wd 1e-5,
  dropout 0, gradient clip 5, raw-y L1, fp32;
* batch order from ``torch.Generator().manual_seed(seed + 101)``, one
  ``randperm(10000)`` per epoch — the same stream for every arm at a seed;
* global-33 standardisation from ``zinc_arch_discovery_v1/analysis/norm.json``
  (computed on the official train 10k only);
* atom/bond encoders initialised from the frozen train-only masked-SSL
  checkpoint of the same seed; every other module from one shared seed stream;
* full official valid every epoch; best checkpoint = strict minimum valid MAE,
  an exact tie keeps the earlier epoch;
* final, W20, T100, eval-mode train@best and train@final all recorded.

Official valid is used only for model selection / development evaluation.  It
never enters a feature, a normalisation constant or a pretraining objective.

Usage:
    python .../aas_train.py --arm A1 --seed 0 --device cuda:0
    python .../aas_train.py --arm A0 --seed 0 --device cuda:0
    python .../aas_train.py --arm A1 --seed 0 --device cpu --smoke
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

import aas_common as BC  # noqa: E402
import ad_common as C  # noqa: E402
import bosi_data as BD  # noqa: E402
import aas_models as AM  # noqa: E402


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def module_grad_norms(model) -> dict:
    out: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = name.split(".")[0]
        out[key] = out.get(key, 0.0) + float(param.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(out.items())}


def train(arm: str, seed: int, device, *, total_updates: int | None = None,
          smoke: bool = False, force: bool = False, max_epochs: int | None = None) -> dict:
    rows_all = np.arange(10_000, dtype=np.int64)
    cache = BC.load_cache("train")
    if len(np.asarray(cache["atom_ptr"])) - 1 != 10_000:
        raise RuntimeError("official train cache is not 10,000 molecules")
    y_train = np.asarray(cache["labels"], dtype=np.float64)
    if len(y_train) != 10_000:
        raise RuntimeError("official train labels are not 10,000 rows")

    vcache = BC.load_cache("valid")
    y_valid = np.asarray(vcache["labels"], dtype=np.float64)
    valid_rows = np.arange(len(y_valid), dtype=np.int64)
    if len(valid_rows) != 1_000:
        raise RuntimeError("official valid cache is not 1,000 molecules")

    rel = BC.load_rel("train")
    vrel = BC.load_rel("valid")
    norm = BC.load_norm()

    runs_dir = BC.RUNS_SMOKE if smoke else BC.RUNS
    out = runs_dir / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        BC.log(f"[{BC.ROUND_ID} {arm} s{seed}] already complete; use --force to rerun")
        return BC.read_json(out / "train.json")

    target_updates = int(BC.RECIPE["total_updates"] if total_updates is None
                         else total_updates)
    if not smoke and target_updates != int(BC.RECIPE["total_updates"]):
        raise RuntimeError("formal runs must use exactly 47,400 updates")

    batch_size = int(BC.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(len(rows_all) / batch_size)
    if steps_per_epoch != int(BC.RECIPE["steps_per_epoch"]):
        raise RuntimeError(f"steps/epoch {steps_per_epoch} != "
                           f"{BC.RECIPE['steps_per_epoch']}")
    n_epochs = max_epochs if max_epochs is not None else int(BC.RECIPE["epochs"])
    warmup = min(int(BC.RECIPE["warmup_steps"]), max(1, target_updates // 20))

    torch.manual_seed(int(seed))
    model, model_info = AM.build_model(norm, arm, seed, device)
    params = int(sum(p.numel() for p in model.parameters()))
    init_state = C.cpu_state(model)
    init_info = {
        "initial_state_sha256": BC.state_hash(init_state),
        "initial_common_state_sha256": BC.state_hash(AM.common_state(init_state)),
        "initial_state_n_tensors": int(len(init_state)),
        "parameter_tensors": int(sum(1 for _ in model.parameters())),
    }
    torch_rng_state = torch.get_rng_state()

    optimizer = torch.optim.AdamW(model.parameters(), lr=BC.RECIPE["lr"],
                                  weight_decay=BC.RECIPE["weight_decay"])
    opt_params = int(sum(p.numel() for g in optimizer.param_groups for p in g["params"]))
    gen = torch.Generator().manual_seed(int(seed) + 101)
    gen_state0 = gen.get_state()

    curve: list[dict] = []
    grad_log: list[dict] = []
    checkpoint_hashes: dict[str, str] = {}
    best = {"valid_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    global_step = 0
    last_lr = float(BC.RECIPE["lr"])
    first_batch_rows = None
    started = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_all), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= target_updates:
                break
            batch_rows = rows_all[order[start:start + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = [int(r) for r in batch_rows]
            batch = _to_device(BD.collate(cache, rel, batch_rows), device)
            target = torch.as_tensor(y_train[batch_rows], dtype=torch.float32,
                                     device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if global_step in (0, 1, 2, 3, 99, 999):
                grad_log.append({"step": int(global_step),
                                 "norms": module_grad_norms(model)})
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(BC.RECIPE["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, target_updates, warmup,
                                              float(BC.RECIPE["lr"]),
                                              float(BC.RECIPE["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)

        valid_pred = BC.predict(model, vcache, vrel, valid_rows, device=device,
                               batch_size=int(BC.RECIPE["eval_batch_size"]))
        valid_mae = BC.mae(valid_pred, y_valid)
        entry = {"epoch": int(epoch), "step": int(global_step),
                 "online_train_mae": float(online), "valid_mae": float(valid_mae),
                 "lr": float(last_lr),
                 "seconds": float(time.perf_counter() - t_ep)}
        curve.append(entry)
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch),
                    "step": int(global_step)}
            best_state = C.cpu_state(model)
            C.atomic_torch_save({"state_dict": best_state, "epoch": int(epoch),
                                 "step": int(global_step), "valid_mae": float(valid_mae),
                                 "arm": arm, "seed": int(seed),
                                 "fit": "official_train_10k",
                                 "selection": "official_valid"}, out / "best.pt")
        if epoch % 10 == 0 or global_step >= target_updates:
            C.log(f"[{BC.ROUND_ID} {arm} s{seed}] ep={epoch} step={global_step} "
                  f"valid={valid_mae:.6f} online={online:.6f} "
                  f"best={best['valid_mae']:.6f}@{best['epoch']} "
                  f"lr={last_lr:.2e} {entry['seconds']:.2f}s")
        if global_step >= target_updates:
            break

    wall_train = time.perf_counter() - started
    final_state = C.cpu_state(model)
    C.atomic_torch_save({"state_dict": final_state, "epoch": int(len(curve)),
                         "step": int(global_step), "arm": arm, "seed": int(seed),
                         "fit": "official_train_10k", "selection": "final"},
                        out / "final.pt")

    # -- fixed-object reads (eval mode) ------------------------------------
    reads: dict[str, dict] = {}
    valid_preds: dict[str, np.ndarray] = {}
    for tag, state in (("best", best_state), ("final", final_state)):
        C.load_state(model, state, device)
        pv = BC.predict(model, vcache, vrel, valid_rows, device=device,
                        batch_size=int(BC.RECIPE["eval_batch_size"]))
        pt = BC.predict(model, cache, rel, rows_all, device=device,
                        batch_size=int(BC.RECIPE["eval_batch_size"]))
        reads[tag] = {
            "valid_mae": BC.mae(pv, y_valid),
            "train_mae_eval_mode": BC.mae(pt, y_train),
            "valid_pred_sha256": BC.sha256_array(pv),
        }
        valid_preds[tag] = pv
        np.savez(out / f"valid_{tag}.npz", row=valid_rows, pred=pv,
                 arm=arm, seed=int(seed), checkpoint=tag)

    wall = time.perf_counter() - started
    for name in ("best.pt", "final.pt"):
        checkpoint_hashes[name] = BC.sha256_file(out / name)
    for name in ("valid_best.npz", "valid_final.npz"):
        checkpoint_hashes[name] = BC.sha256_file(out / name)

    cm = BC.curve_metrics(curve)
    info = {
        "round": BC.ROUND_ID,
        "arm": arm,
        "seed": int(seed),
        "parameters": params,
        "parameter_expectation": BC.EXPECTED_PARAMETERS.get(arm),
        "parameter_audit": {"parameters": params,
                            "optimizer_parameters": opt_params,
                            "parameters_in_optimizer": params == opt_params,
                            "parameter_tensors": init_info["parameter_tensors"]},
        "model_info": model_info,
        "recipe": dict(BC.RECIPE),
        "total_updates": int(target_updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "n_train_rows": int(len(rows_all)),
        "n_valid_rows": int(len(valid_rows)),
        "train_rows_sha256": BC.rows_hash(rows_all),
        "valid_rows_sha256": BC.rows_hash(valid_rows),
        "train_label_sha256": BC.sha256_array(y_train, "<f8"),
        "valid_label_sha256": BC.sha256_array(y_valid, "<f8"),
        "data": {
            "train_cache": BC.rel_or_abs(BC.DATA_TRAIN),
            "train_cache_sha256": BC.sha256_file(BC.DATA_TRAIN),
            "valid_cache": BC.rel_or_abs(BC.DATA_VALID),
            "valid_cache_sha256": BC.sha256_file(BC.DATA_VALID),
            "relation_train_content_sha256": BD.relation_content_hash(rel),
            "relation_valid_content_sha256": BD.relation_content_hash(vrel),
            "norm_source": norm["source"],
            "norm_path": norm["path"],
            "norm_sha256": norm["sha256"],
            "norm_valid_or_test_used": bool(norm["valid_or_test_used"]),
            "ssl": model_info["ssl"],
            "official_valid_loaded": True,
            "official_valid_role": "per-epoch evaluation + best-checkpoint selection only",
            "official_test_loaded": False,
        },
        "initial": init_info,
        "rng": {"torch_seed": int(seed),
                "batch_order_generator": "torch.Generator().manual_seed(seed + 101)",
                "generator_state_sha256": BC.sha256_bytes(gen_state0),
                "torch_rng_state_sha256": BC.sha256_bytes(torch_rng_state)},
        "first_batch_rows": first_batch_rows,
        "first_batch_rows_sha256": BC.rows_hash(first_batch_rows),
        "best": {"valid_mae": best["valid_mae"], "epoch": best["epoch"],
                 "step": best["step"]},
        "reads": reads,
        "W20": cm["W20"],
        "W20_window_start_epoch": cm["W20_window_start_epoch"],
        "T100": cm["T100"],
        "valid_min_over_curve": cm["valid_min"],
        "valid_last_epoch": cm["valid_last"],
        "n_epochs_in_curve": cm["n_epochs"],
        "checkpoint_sha256": checkpoint_hashes,
        "grad_norms": grad_log,
        "wall_train_seconds": float(wall_train),
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "device": str(device),
        "smoke": bool(smoke),
        "curve": curve,
        "written": BC.now(),
    }
    BC.write_json(out / "train.json", info)
    C.write_curve_csv(out / "curve.csv", curve)
    BC.log(f"[{BC.ROUND_ID} {arm} s{seed}] DONE bestValid={best['valid_mae']:.6f}"
           f"@{best['epoch']} finalValid={reads['final']['valid_mae']:.6f} "
           f"train@best={reads['best']['train_mae_eval_mode']:.6f} "
           f"W20={cm['W20']:.6f} T100={cm['T100']:.6f} gpu_h={info['gpu_hours']:.3f}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=list(AM.ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.smoke:
        total = 79 * 2 if args.total_updates is None else int(args.total_updates)
        info = train(args.arm, args.seed, device, total_updates=total, smoke=True,
                     force=True, max_epochs=args.max_epochs)
        print(json.dumps({k: info[k] for k in ("arm", "parameters", "total_updates",
                                               "epochs_run", "best", "reads",
                                               "grad_norms", "gpu_hours")}, indent=2))
        return 0
    train(args.arm, args.seed, device, total_updates=args.total_updates,
          force=args.force, max_epochs=args.max_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
