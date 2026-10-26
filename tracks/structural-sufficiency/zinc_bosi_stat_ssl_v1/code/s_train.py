"""Supervised trainer for BOSI-SSL and for the compute-matched control C.

The recipe is the frozen BOSI v0 / partner-content one, byte-for-byte:

    Fit8k, Fit-only standardisation, batch 128, AdamW 1e-3 -> 1e-5 per-step
    cosine (warm-up 500, wd 1e-5, clip 5), raw-y L1, 47,400 updates, batch order
    ``torch.Generator(seed + 101)``, best checkpoint = strict minimum Monitor MAE.

Two declared differences, and only these:

* ``--init ssl`` replaces ``half_encoder`` / ``side`` / ``compose`` by the
  frozen pretrained export of this seed.  ``atom_encoder`` / ``bond_encoder``
  (the historical per-seed SSL checkpoint) and ``reader`` (the historical seed
  initialisation) are bit-identical to BOSI-B; gate S3 checks it.
* ``--init historical --extra-updates N`` is the compute control C: the first
  47,400 updates run the exact frozen schedule and batch order, then ``N``
  further updates continue the same batch-order stream at the terminal LR
  (1e-5).  ``N`` is frozen in advance from the measured pretraining wall time.

Locked is never read here.
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

import s_common as SC  # noqa: E402
import s_pretext as SP  # noqa: E402
import ad_common as C  # noqa: E402

BD = SC.BD
M_START = 2 * SC.BM.WIDTH + SC.BM.RANK
M_STOP = M_START + BD.N_REL * 3 * SC.BM.RANK


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def grad_norms(model) -> dict:
    out = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = name.split(".")[0]
        out[key] = out.get(key, 0.0) + float(param.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(out.items())}


def module_hash(module) -> str:
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def load_ssl_export(model, seed: int, run_dir: Path) -> dict:
    """Load the frozen pretrained half/side/compose of this seed."""
    ckpt = run_dir / SC.pretrain_state_name(seed)
    meta_path = run_dir / "pretrain.json"
    if not ckpt.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing pretraining artefacts in {run_dir}")
    meta = json.loads(meta_path.read_text())
    if meta["mode"] != "full" or int(meta["updates"]) != SC.PRETRAIN_UPDATES:
        raise RuntimeError("the SSL initialisation must come from the full pretraining run")
    if SC.sha256_file(ckpt) != meta["state_sha256"]:
        raise RuntimeError("pretraining state file changed since it was written")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    before = {n: module_hash(getattr(model, n)) for n in
              ("atom_encoder", "bond_encoder", "half_encoder", "side", "compose", "reader")}
    for name in ("half_encoder", "side", "compose"):
        getattr(model, name).load_state_dict({k: v for k, v in blob["export"][name].items()})
    after = {n: module_hash(getattr(model, n)) for n in before}
    changed = sorted(n for n in before if before[n] != after[n])
    if changed != ["compose", "half_encoder", "side"]:
        raise RuntimeError(f"SSL load changed the wrong modules: {changed}")
    return {
        "source": SC.rel_or_abs(ckpt),
        "state_sha256": meta["state_sha256"],
        "export_sha256": meta["export_sha256"],
        "updates": int(meta["updates"]),
        "pretrain_wall_seconds": float(meta["wall_seconds"]),
        "pretrain_gpu_hours": float(meta["gpu_hours"]),
        "modules_changed": changed,
        "module_sha256_before": before,
        "module_sha256_after": after,
    }


def train(tag: str, seed: int, device, *, init: str = "historical",
          total_updates: int = SC.BC.RECIPE["total_updates"],
          extra_updates: int = 0, force: bool = False) -> dict:
    rows = SC.split_rows()
    train_rows = rows["fit8k"]
    monitor_rows = rows["monitor"]
    out = SC.RUNS / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        return json.loads((out / "train.json").read_text())

    cache = SC.load_train_cache()
    rel = SC.load_relation_cache()
    y_train = np.asarray(cache["labels"], dtype=np.float64)
    y_monitor = np.asarray(cache["labels"], dtype=np.float64)[monitor_rows]
    norm = SC.fit_only_norm(cache)

    model, model_info = SP.build_backbone(norm, seed, device)
    if model_info["initial_state_sha256"] != SC.HISTORICAL_INIT_SHA256[int(seed)]:
        raise RuntimeError("initial state is not the historical BOSI-B initial state")
    ssl_info = {"init": init, "loaded": False}
    if init == "ssl":
        ssl_info = {"init": init, "loaded": True,
                    **load_ssl_export(model, seed, SC.RUNS / f"S_pre{int(seed)}_full")}
    params = C.parameter_count(model)
    if params != SC.BC.EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameters {params} != {SC.BC.EXPECTED_PARAMETERS}")

    n_total = int(total_updates) + int(extra_updates)
    batch_size = int(SC.BC.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    n_epochs = math.ceil(n_total / steps_per_epoch)
    warmup = min(int(SC.BC.RECIPE["warmup_steps"]), max(1, int(total_updates) // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=SC.BC.RECIPE["lr"],
                                  weight_decay=SC.BC.RECIPE["weight_decay"])
    gen = torch.Generator().manual_seed(int(seed) + 101)

    curve: list[dict] = []
    grad_log: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    global_step = 0
    started = time.perf_counter()
    last_lr = float(SC.BC.RECIPE["lr"])
    first_batch_rows = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= n_total:
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
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), SC.BC.RECIPE["grad_clip"])
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            if global_step <= int(total_updates):
                last_lr = C.cosine_with_warmup_lr(global_step, int(total_updates), warmup,
                                                  float(SC.BC.RECIPE["lr"]),
                                                  float(SC.BC.RECIPE["min_lr"]))
            else:
                last_lr = float(SC.BC.RECIPE["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)

        monitor_pred = SC.BC.predict(model, cache, rel, monitor_rows, device=device,
                                     batch_size=int(SC.BC.RECIPE["eval_batch_size"]))
        monitor_mae = SC.BC.mae(monitor_pred, y_monitor)
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
                                 "arm": "B", "seed": int(seed), "fit": "fit8k",
                                 "init": init, "selection": "monitor"},
                                out / "best_monitor.pt")
        if epoch % 10 == 0 or global_step >= n_total:
            SC.log(f"[{SC.ROUND_ID} {tag}] ep={epoch} step={global_step} "
                   f"monitor={monitor_mae:.6f} online={online:.6f} "
                   f"best={best['monitor_mae']:.6f}@{best['epoch']} "
                   f"lr={last_lr:.2e} {entry['seconds']:.2f}s")
        if global_step >= n_total:
            break

    final_state = C.cpu_state(model)
    C.atomic_torch_save({"state_dict": final_state, "epoch": int(len(curve)),
                         "step": int(global_step), "arm": "B", "seed": int(seed),
                         "fit": "fit8k", "init": init, "selection": "final"},
                        out / "final.pt")
    wall = time.perf_counter() - started

    reads = {}
    for ckpt_tag, state in (("best_monitor", best_state), ("final", final_state)):
        C.load_state(model, state, device)
        pred_mon = SC.BC.predict(model, cache, rel, monitor_rows, device=device,
                                 batch_size=int(SC.BC.RECIPE["eval_batch_size"]))
        reads[ckpt_tag] = {"monitor_mae": SC.BC.mae(pred_mon, y_monitor)}
        C.atomic_torch_save({"monitor_pred": np.asarray(pred_mon),
                             "monitor_rows": monitor_rows,
                             "arm": "B", "seed": int(seed), "tag": tag,
                             "checkpoint": ckpt_tag},
                            out / f"predictions_monitor_{ckpt_tag}.pt")

    tail = curve[-max(1, len(curve) // 10):]
    info = {
        "round": SC.ROUND_ID,
        "kind": "supervised",
        "tag": tag,
        "fit": "fit8k",
        "seed": int(seed),
        "arm": "B",
        "init": ssl_info,
        "parameters": params,
        "model_info": model_info,
        "recipe": dict(SC.BC.RECIPE),
        "total_updates": int(total_updates),
        "extra_updates": int(extra_updates),
        "updates_run": int(global_step),
        "lr_rule": ("per-step cosine over 47,400 then constant min_lr"
                    if extra_updates else "per-step cosine over 47,400"),
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
        "module_sha256_at_start": {n: module_hash(getattr(model, n)) for n in
                                   ("atom_encoder", "bond_encoder", "half_encoder",
                                    "side", "compose", "reader")},
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_read": False,
        "curve": curve,
    }
    SC.write_json(out / "train.json", info)
    SC.log(f"[{SC.ROUND_ID} {tag}] DONE bestMonitor={best['monitor_mae']:.6f}"
           f"@{best['epoch']} finalMonitor={reads['final']['monitor_mae']:.6f} "
           f"gpu_h={info['gpu_hours']:.3f}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--init", choices=("historical", "ssl"), default="historical")
    p.add_argument("--extra-updates", type=int, default=0)
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    total = (SC.BC.RECIPE["total_updates"] if args.total_updates is None
             else int(args.total_updates))
    info = train(args.tag, args.seed, torch.device(args.device), init=args.init,
                 total_updates=total, extra_updates=args.extra_updates, force=args.force)
    print(json.dumps({k: info[k] for k in ("tag", "seed", "init", "updates_run",
                                           "best_monitor", "monitor_reads",
                                           "gpu_hours")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
