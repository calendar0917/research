"""Formal trainer for both families (identical recipe, task §5).

    python -m ...  ->  xs_train.py --family cma --arm REAL --seed 0 --fit fit8k
                       [--device cuda:0] [--prepare-only] [--smoke]

Recipe is fixed and read from ``xs_common.RECIPE``: AdamW 1e-3 -> 1e-5
per-update cosine, 500-step warmup, wd 1e-5, clip 5, raw-y L1, batch 128,
47,400 updates, full Monitor every epoch, strict-minimum Monitor checkpoint.

No label-derived quantity reaches the model: the batch carries structure and
raw label-blind features only, and the target is used only inside the loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
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

import xs_common as C  # noqa: E402
import xs_data as D  # noqa: E402
import xs_models as SM  # noqa: E402

FAMILIES = ("cma", "spc")


def _family_modules(family: str):
    if family == "cma":
        import cma_model as M
        import cma_structure as S
    elif family == "spc":
        import spc_model as M
        import spc_structure as S
    else:
        raise ValueError(family)
    return M, S


def grad_norms(model) -> dict:
    out: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = ".".join(name.split(".")[:2])
        out[key] = out.get(key, 0.0) + float(param.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(out.items())}


class SplitData:
    """One split: base tensors + family structure cache."""

    def __init__(self, cache: dict, family: str, name: str, device, *, struct=None):
        self.cache = cache
        self.base = D.BaseData(cache, device=device)
        _, S = _family_modules(family)
        self.struct = S.load_split(name, cache=cache) if struct is None else struct
        self.collate_fn = S.collate

    def batch(self, rows, norm):
        base = self.base.collate(rows, norm=norm)
        out = dict(base)
        out.update(self.collate_fn(self.struct, base, np.asarray(rows, dtype=np.int64)))
        for key in ("atom_ptr", "bond_ptr", "atom_off", "bond_off", "rows"):
            out.pop(key, None)
        C.assert_forward_inputs(out)
        return out

    def targets(self, rows):
        return self.base.targets(rows)


def predict(model, dataset: SplitData, rows: np.ndarray, norm, *, device,
            batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = np.asarray(rows[start:start + batch_size], dtype=np.int64)
            batch = dataset.batch(chunk, norm)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)


def mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(y, np.float64))))


def _out_dir(family: str, arm: str, seed: int, fit: str, smoke: bool) -> Path:
    tag = f"{family}-{arm.lower()}_s{seed}_{fit}"
    if smoke:
        tag += "_smoke"
    return C.RUNS / tag


def train(family: str, arm: str, seed: int, *, fit: str = "fit8k", monitor: str = "monitor",
          device: str = "cpu", total_updates: int | None = None, smoke: bool = False,
          force: bool = False, tag: str | None = None) -> dict:
    M, _S = _family_modules(family)
    total_updates = int(C.RECIPE["total_updates"] if total_updates is None else total_updates)
    out_dir = C.RUNS / (tag if tag else f"{family}-{arm.lower()}_s{seed}_{fit}")
    C.ensure_dir(out_dir)
    if (out_dir / "train.json").exists() and not force:
        return C.read_json(out_dir / "train.json")

    started_at = C.now()
    t_start = time.perf_counter()

    split = C.split_rows()
    C.verify_split_hashes(split["rows"])
    if monitor not in ("monitor", "valid"):
        raise ValueError(monitor)
    split_rows = split["rows"]

    if fit == "fit8k":
        train_rows = split_rows["fit8k"]
        train_cache = C.load_cache("train")
        train_ds = SplitData(train_cache, family, "train", device)
        norm = C.fit_only_norm(train_cache, train_rows)
        monitor_ds = train_ds
        mon_rows = split_rows[monitor] if monitor == "monitor" else None
        if mon_rows is None:
            raise ValueError("internal monitor rows required for fit8k stage")
    elif fit == "train10k":
        train_cache = C.load_cache("train")
        train_rows = np.arange(int(train_cache["n_molecules"]), dtype=np.int64)
        train_ds = SplitData(train_cache, family, "train", device)
        norm = C.fit_only_norm(train_cache, train_rows)
        if monitor != "valid":
            raise ValueError("train10k confirmation must be selected on official valid")
        valid_cache = C.load_cache("valid")
        monitor_ds = SplitData(valid_cache, family, "valid", device)
        mon_rows = np.arange(int(valid_cache["n_molecules"]), dtype=np.int64)
    else:
        raise ValueError(fit)

    y_train = train_ds.targets(train_rows)
    y_mon = monitor_ds.targets(mon_rows)

    model, model_info = M.build_model(norm, int(seed), arm=arm, device=device, ssl_seed=int(seed))
    params = C.parameter_count(model)
    if params > C.PARAM_CAP:
        raise RuntimeError(f"{params} deployment parameters exceed cap {C.PARAM_CAP}")
    initial_hash = C.state_hash(model)

    batch_size = int(C.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    n_epochs = math.ceil(total_updates / steps_per_epoch)
    warmup = min(int(C.RECIPE["warmup_steps"]), max(1, total_updates // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.RECIPE["lr"],
                                  weight_decay=C.RECIPE["weight_decay"])
    gen = torch.Generator().manual_seed(int(seed) + 101)

    curve: list[dict] = []
    grad_log: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    global_step = 0
    last_lr = float(C.RECIPE["lr"])
    first_batch_rows = None
    nonfinite = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= total_updates:
                break
            b_rows = train_rows[order[start:start + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = [int(x) for x in b_rows]
            batch = train_ds.batch(b_rows, norm)
            target = torch.as_tensor(y_train[order[start:start + batch_size]],
                                     dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if global_step in (0, 1, 2, 99, 999):
                grad_log.append({"step": global_step, "norms": grad_norms(model)})
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), C.RECIPE["grad_clip"])
            if not math.isfinite(float(gn)):
                nonfinite += 1
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, total_updates, warmup,
                                              float(C.RECIPE["lr"]), float(C.RECIPE["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(b_rows)
        online = total_err / max(n_mol, 1)

        mon_pred = predict(model, monitor_ds, mon_rows, norm, device=device,
                           batch_size=int(C.RECIPE["eval_batch_size"]))
        mon_mae = mae(mon_pred, y_mon)
        entry = {"epoch": int(epoch), "step": int(global_step),
                 "online_train_mae": float(online), "monitor_mae": float(mon_mae),
                 "lr": float(last_lr), "seconds": float(time.perf_counter() - t_ep)}
        curve.append(entry)
        if mon_mae < best["monitor_mae"]:
            best = {"monitor_mae": float(mon_mae), "epoch": int(epoch), "step": int(global_step)}
            best_state = C.cpu_state(model)
            C.atomic_torch_save({"state_dict": best_state, "epoch": int(epoch),
                                 "step": int(global_step), "monitor_mae": float(mon_mae),
                                 "arm": arm, "seed": int(seed), "family": family, "fit": fit,
                                 "selection": monitor}, out_dir / "best_monitor.pt")
        if epoch % 10 == 0 or global_step >= total_updates:
            C.log(f"[{family}-{arm} s{seed} {fit}] ep={epoch} step={global_step} "
                  f"mon={mon_mae:.6f} online={online:.6f} best={best['monitor_mae']:.6f}"
                  f"@{best['epoch']} lr={last_lr:.2e} {entry['seconds']:.2f}s")
        if global_step >= total_updates:
            break

    final_state = C.cpu_state(model)
    C.atomic_torch_save({"state_dict": final_state, "epoch": int(len(curve)),
                         "step": int(global_step), "arm": arm, "seed": int(seed),
                         "family": family, "fit": fit, "selection": "final"},
                        out_dir / "final.pt")
    wall = time.perf_counter() - t_start

    reads = {}
    for tag_name, state in (("best_monitor", best_state), ("final", final_state)):
        C.load_state(model, state, device)
        pred_mon = predict(model, monitor_ds, mon_rows, norm, device=device,
                           batch_size=int(C.RECIPE["eval_batch_size"]))
        pred_train = predict(model, train_ds, train_rows, norm, device=device,
                             batch_size=int(C.RECIPE["eval_batch_size"]))
        reads[tag_name] = {
            "monitor_mae": mae(pred_mon, y_mon),
            "eval_train_mae": mae(pred_train, y_train),
            "state_hash": C.state_hash(model),
        }
        C.atomic_torch_save({"pred": np.asarray(pred_mon, dtype=np.float32),
                             "rows": np.asarray(mon_rows, dtype=np.int64),
                             "arm": arm, "seed": int(seed), "family": family,
                             "fit": fit, "checkpoint": tag_name},
                            out_dir / f"predictions_{tag_name}.pt")
        C.atomic_torch_save({"pred": np.asarray(pred_train, dtype=np.float32),
                             "rows": np.asarray(train_rows, dtype=np.int64),
                             "arm": arm, "seed": int(seed), "family": family,
                             "fit": fit, "checkpoint": tag_name},
                            out_dir / f"predictions_train_{tag_name}.pt")

    tail = curve[-max(1, min(100, len(curve))):]
    rolling = _rolling_min_median([e["monitor_mae"] for e in curve], 20)
    info = {
        "round": C.ROUND_ID,
        "family": family,
        "arm": arm,
        "fit": fit,
        "monitor_split": monitor,
        "seed": int(seed),
        "parameters": params,
        "parameter_breakdown": SM.parameter_breakdown(model),
        "model_info": model_info,
        "initial_state_sha256": initial_hash,
        "norm": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()},
        "recipe": dict(C.RECIPE),
        "total_updates": int(total_updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "n_train_rows": int(len(train_rows)),
        "n_monitor_rows": int(len(mon_rows)),
        "split_hashes": split["hashes"],
        "best_monitor": best,
        "rolling20_min_median_monitor_mae": float(rolling["min"]) if rolling else None,
        "rolling20_argmin_epoch": int(rolling["epoch"]) if rolling else None,
        "tail100_median_monitor_mae": float(np.median([e["monitor_mae"] for e in tail]))
        if len(curve) >= 100 else None,
        "final_monitor_mae": float(curve[-1]["monitor_mae"]),
        "reads": reads,
        "first_batch_rows": first_batch_rows,
        "grad_norms": grad_log,
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "device": str(device),
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "started_at": started_at,
        "finished_at": C.now(),
        "checkpoint_sha256": {
            "best_monitor": C.sha256_file(out_dir / "best_monitor.pt"),
            "final": C.sha256_file(out_dir / "final.pt"),
        },
        "official_valid_loaded": bool(fit == "train10k"),
        "official_test_loaded": False,
        "locked_read": False,
        "smoke": bool(smoke),
        "curve": curve,
    }
    C.write_json(out_dir / "train.json", info)
    C.log(f"[{family}-{arm} s{seed} {fit}] DONE bestMon={best['monitor_mae']:.6f}"
          f"@{best['epoch']} gpu_h={info['gpu_hours']:.3f}")
    return info


def _rolling_min_median(values: list[float], window: int) -> dict | None:
    if len(values) < window:
        return None
    best = None
    for end in range(window, len(values) + 1):
        med = float(np.median(values[end - window:end]))
        if best is None or med < best["min"]:
            best = {"min": med, "epoch": end}
    return best


def prepare(family: str) -> dict:
    """Build both structure caches (CPU, deterministic)."""
    _, S = _family_modules(family)
    out = {}
    t0 = time.perf_counter()
    train_cache = C.load_cache("train")
    obj = S.load_split("train", cache=train_cache, build=True)
    out["train"] = dict(S.coverage_stats(obj, train_cache) if family == "cma"
                        else S.coverage_stats(obj))
    valid_cache = C.load_cache("valid")
    objv = S.load_split("valid", cache=valid_cache, build=True)
    out["valid"] = dict(S.coverage_stats(objv, valid_cache) if family == "cma"
                        else S.coverage_stats(objv))
    out["seconds"] = float(time.perf_counter() - t0)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=list(FAMILIES))
    p.add_argument("--arm", required=True, choices=["REAL", "MEAN"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit", default="fit8k", choices=["fit8k", "train10k"])
    p.add_argument("--monitor", default="monitor", choices=["monitor", "valid"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args(argv)

    if args.prepare_only:
        print(json.dumps(prepare(args.family), indent=2))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device} but CUDA is unavailable")
    total = args.total_updates
    if args.smoke and total is None:
        total = 126  # 2 epochs at 8k
    info = train(args.family, args.arm, args.seed, fit=args.fit, monitor=args.monitor,
                 device=str(device), total_updates=total, smoke=args.smoke,
                 force=args.force, tag=args.tag)
    print(json.dumps({k: info[k] for k in ("family", "arm", "fit", "total_updates",
                                           "epochs_run", "best_monitor", "reads",
                                           "gpu_hours", "checkpoint_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
