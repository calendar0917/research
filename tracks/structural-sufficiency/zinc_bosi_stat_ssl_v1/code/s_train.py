"""Supervised trainer for BOSI-SSL and for the compute-matched control C.

The recipe is the frozen BOSI v0 / partner-content one, byte-for-byte:

    Fit8k, Fit-only standardisation, batch 128, AdamW 1e-3 -> 1e-5 per-step
    cosine (warm-up 500, wd 1e-5, clip 5, raw-y L1), 47,400 updates, batch order
    ``torch.Generator(seed + 101)``, best checkpoint = strict minimum Monitor MAE.

Two declared arms, and only these:

``--init ssl``
    ``half_encoder`` / ``side`` / ``compose`` are replaced by the frozen
    pretrained export of this seed.  ``atom_encoder`` / ``bond_encoder`` (the
    historical per-seed SSL checkpoints) and ``reader`` are bit-identical to
    BOSI-B; the loader verifies that exactly the three modules changed.

``--init historical --extra-updates N``  (control C)
    The historical BOSI-B checkpoint that the previous rounds actually deploy
    (``best_monitor.pt``, Monitor-selected) is loaded, hash-verified against the
    v0 freeze, and receives ``N`` further supervised updates: the batch-order
    stream continues from the next epoch of ``Generator(seed + 101)`` and the
    learning rate is the frozen schedule's terminal value ``1e-5``.  ``N`` is
    frozen in advance from the measured pretraining wall time
    (``analysis/control_budget.json``).  Checkpoint selection stays "strict
    minimum Monitor MAE", over ``{the loaded historical state} ∪ {epoch ends}``.

BOSI-B itself is never retrained: ``--init historical --extra-updates 0`` is
refused unless ``--allow-full-rerun`` is passed explicitly.
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


MODULES = ("atom_encoder", "bond_encoder", "half_encoder", "side", "compose", "reader")


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
    before = {n: module_hash(getattr(model, n)) for n in MODULES}
    for name in ("half_encoder", "side", "compose"):
        getattr(model, name).load_state_dict({k: v for k, v in blob["export"][name].items()})
    after = {n: module_hash(getattr(model, n)) for n in MODULES}
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


def load_historical_start(model, seed: int, device) -> dict:
    """Control C: the deployed historical BOSI-B checkpoint of this seed."""
    path = SC.historical_b_checkpoint(seed, "best_monitor")
    expect = SC.historical_b_run(seed)["checkpoints"]["best_monitor"]["sha256"]
    if SC.sha256_file(path) != expect:
        raise RuntimeError(f"historical BOSI-B checkpoint changed: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    C.load_state(model, blob["state_dict"], device)
    step, epoch = int(blob["step"]), int(blob["epoch"])
    spe = math.ceil(len(SC.split_rows()["fit8k"]) / int(SC.BC.RECIPE["batch_size"]))
    if step != epoch * spe:
        raise RuntimeError(f"historical checkpoint is not at an epoch boundary: "
                           f"step={step} epoch={epoch} steps_per_epoch={spe}")
    return {
        "source": SC.rel_or_abs(path),
        "checkpoint_sha256": expect,
        "step": step,
        "epoch": epoch,
        "resume_epoch": epoch + 1,
        "monitor_mae": float(blob["monitor_mae"]),
        "steps_per_epoch": int(spe),
    }


def train(tag: str, seed: int, device, *, init: str = "ssl",
          total_updates: int = SC.BC.RECIPE["total_updates"],
          extra_updates: int = 0, force: bool = False,
          allow_full_rerun: bool = False) -> dict:
    warm = init == "historical" and int(extra_updates) > 0
    if init == "historical" and int(extra_updates) == 0 and not allow_full_rerun:
        raise SystemExit("refusing to retrain BOSI-B from scratch this round "
                         "(pass --allow-full-rerun only for an explicit rerun)")
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

    init_info = {"init": init, "loaded": False}
    start = None
    if init == "ssl":
        init_info = {"init": init, "loaded": True,
                     **load_ssl_export(model, seed, SC.RUNS / f"S_pre{int(seed)}_full")}
    elif warm:
        start = load_historical_start(model, seed, device)
        init_info = {"init": "historical+extra", "loaded": True, **start}
    params = C.parameter_count(model)
    if params != SC.BC.EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameters {params} != {SC.BC.EXPECTED_PARAMETERS}")
    module_sha_at_start = {n: module_hash(getattr(model, n)) for n in MODULES}

    n_target = int(extra_updates) if warm else int(total_updates)
    batch_size = int(SC.BC.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    warmup = min(int(SC.BC.RECIPE["warmup_steps"]), max(1, int(total_updates) // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=SC.BC.RECIPE["lr"],
                                  weight_decay=SC.BC.RECIPE["weight_decay"])
    gen = torch.Generator().manual_seed(int(seed) + 101)
    base_step = 0
    first_epoch = 1
    if warm:
        base_step = int(start["step"])
        first_epoch = int(start["resume_epoch"])
        for _ in range(first_epoch - 1):
            gen.randperm(len(train_rows))   # replay the frozen batch-order stream

    curve: list[dict] = []
    grad_log: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    best_origin = "trained"
    started = time.perf_counter()
    last_lr = float(SC.BC.RECIPE["min_lr"] if warm else SC.BC.RECIPE["lr"])
    first_batch_rows = None
    done = 0

    if warm:
        # the loaded historical checkpoint is a candidate under the same
        # strict-minimum Monitor rule
        best.update({"monitor_mae": float(start["monitor_mae"]),
                     "epoch": int(start["epoch"]), "step": int(start["step"])})
        best_state = C.cpu_state(model)
        best_origin = "historical"
        SC.log(f"[{SC.ROUND_ID} {tag}] warm start from B_s{seed} step {base_step} "
               f"monitor={start['monitor_mae']:.9f}; {n_target} extra updates")

    epoch = first_epoch
    while done < n_target:
        model.train()
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for s0 in range(0, len(order), batch_size):
            if done >= n_target:
                break
            batch_rows = train_rows[order[s0:s0 + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = batch_rows.tolist()
            batch = _to_device(BD.collate(cache, rel, batch_rows), device)
            target = torch.as_tensor(y_train[batch_rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if done in (0, 1, 99, 999):
                grad_log.append({"step": int(base_step + done), "norms": grad_norms(model)})
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), SC.BC.RECIPE["grad_clip"])
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {base_step + done}")
            done += 1
            if warm:
                last_lr = float(SC.BC.RECIPE["min_lr"])
            else:
                last_lr = C.cosine_with_warmup_lr(done, int(total_updates), warmup,
                                                  float(SC.BC.RECIPE["lr"]),
                                                  float(SC.BC.RECIPE["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)

        monitor_pred = SC.BC.predict(model, cache, rel, monitor_rows, device=device,
                                     batch_size=int(SC.BC.RECIPE["eval_batch_size"]))
        monitor_mae = SC.BC.mae(monitor_pred, y_monitor)
        entry = {"epoch": int(epoch), "step": int(base_step + done),
                 "online_train_mae": float(online), "monitor_mae": float(monitor_mae),
                 "lr": float(last_lr), "seconds": float(time.perf_counter() - t_ep)}
        curve.append(entry)
        if monitor_mae < best["monitor_mae"]:
            best = {"monitor_mae": float(monitor_mae), "epoch": int(epoch),
                    "step": int(base_step + done)}
            best_state = C.cpu_state(model)
            best_origin = "extra"
            C.atomic_torch_save({"state_dict": best_state, "epoch": int(epoch),
                                 "step": int(base_step + done),
                                 "monitor_mae": float(monitor_mae),
                                 "arm": "B", "seed": int(seed), "fit": "fit8k",
                                 "init": init, "selection": "monitor"},
                                out / "best_monitor.pt")
        if epoch % 10 == 0 or done >= n_target:
            SC.log(f"[{SC.ROUND_ID} {tag}] ep={epoch} step={base_step + done} "
                   f"monitor={monitor_mae:.6f} online={online:.6f} "
                   f"best={best['monitor_mae']:.6f}@{best['epoch']} "
                   f"lr={last_lr:.2e} {entry['seconds']:.2f}s")
        epoch += 1

    if best_state is None:
        raise RuntimeError("no candidate checkpoint selected")
    if not (out / "best_monitor.pt").exists():
        # the historical warm start stayed the best checkpoint
        C.atomic_torch_save({"state_dict": best_state, "epoch": int(best["epoch"]),
                             "step": int(best["step"]),
                             "monitor_mae": float(best["monitor_mae"]),
                             "arm": "B", "seed": int(seed), "fit": "fit8k",
                             "init": init, "selection": "monitor",
                             "origin": "historical"}, out / "best_monitor.pt")
    final_state = C.cpu_state(model)
    C.atomic_torch_save({"state_dict": final_state, "epoch": int(epoch - 1),
                         "step": int(base_step + done), "arm": "B", "seed": int(seed),
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
        "init": init_info,
        "warm_start": warm,
        "parameters": params,
        "model_info": model_info,
        "recipe": dict(SC.BC.RECIPE),
        "total_updates": int(base_step + done),
        "extra_updates": int(extra_updates),
        "updates_run": int(done),
        "updates_in_schedule": int(total_updates),
        "lr_rule": ("constant terminal lr 1e-5 after the frozen cosine"
                    if warm else "per-step cosine over 47,400"),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "first_epoch": int(first_epoch),
        "n_train_rows": int(len(train_rows)),
        "split_hashes": rows["hashes"],
        "best_monitor": best,
        "best_origin": best_origin,
        "monitor_reads": {k: {"monitor_mae": v["monitor_mae"]} for k, v in reads.items()},
        "tail10_median_monitor_mae": float(np.median([e["monitor_mae"] for e in tail])),
        "tail10_median_online_train_mae": float(np.median([e["online_train_mae"] for e in tail])),
        "first_batch_rows": first_batch_rows,
        "grad_norms": grad_log,
        "module_sha256_at_start": module_sha_at_start,
        "module_sha256_at_end": {n: module_hash(getattr(model, n)) for n in MODULES},
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "seconds_per_update": float(wall / max(done, 1)),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_read": False,
        "curve": curve,
    }
    SC.write_json(out / "train.json", info)
    SC.log(f"[{SC.ROUND_ID} {tag}] DONE bestMonitor={best['monitor_mae']:.6f}"
           f"@{best['epoch']}({best_origin}) finalMonitor={reads['final']['monitor_mae']:.6f} "
           f"gpu_h={info['gpu_hours']:.3f}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--init", choices=("historical", "ssl"), default="ssl")
    p.add_argument("--extra-updates", type=int, default=0)
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-full-rerun", action="store_true")
    args = p.parse_args(argv)
    total = (SC.BC.RECIPE["total_updates"] if args.total_updates is None
             else int(args.total_updates))
    info = train(args.tag, args.seed, torch.device(args.device), init=args.init,
                 total_updates=total, extra_updates=args.extra_updates,
                 force=args.force, allow_full_rerun=args.allow_full_rerun)
    print(json.dumps({k: info[k] for k in ("tag", "seed", "init", "updates_run",
                                           "best_monitor", "monitor_reads",
                                           "gpu_hours")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
