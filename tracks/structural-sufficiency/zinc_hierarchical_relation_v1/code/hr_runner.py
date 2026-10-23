"""Runner for zinc_hierarchical_relation_v1.

Stages
------
prep     build the label-blind train/valid cache, fit train-only statistics
smoke    short GPU timing / memory / gate run (not a formal result)
profile  forward+backward timing and FLOP count on a real batch
train    formal A or B training (seed 0, frozen recipe)
score    valid metrics, P3B comparison, frozen gate

Official test is never loaded anywhere in this file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import hr_common as C
import hr_data as D
import hr_models as M


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def environment() -> dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "git_commit": git_commit(),
    }
    if torch.cuda.is_available():
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["driver_version"] = os.environ.get("NVIDIA_DRIVER_VERSION", "")
    return info


def load_y(split: str) -> np.ndarray:
    from torch_geometric.datasets import ZINC

    import cscl_features as cf

    ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split=split)
    y = np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(len(ds))], dtype=np.float64)
    return y


def label_hash(y: np.ndarray) -> str:
    return C.sha256_array(y.astype(np.float32))


def load_ssl_path() -> Path:
    cfg = C.load_config()
    path = C.REPO / cfg["ssl"]["path"]
    if not path.exists():
        raise SystemExit(f"missing SSL checkpoint: {path}")
    got = C.sha256_file(path)
    if got != cfg["ssl"]["sha256"]:
        raise SystemExit(f"SSL checkpoint hash mismatch: {got} != {cfg['ssl']['sha256']}")
    return path


def cache_paths(split: str) -> tuple[Path, Path]:
    return C.CACHE_DIR / f"hr_{split}.pt", C.CACHE_DIR / f"hr_stats.npz"


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> int:
    cfg = C.load_config()
    patch_radius = int(cfg["data"]["patch_radius"])
    t0 = time.time()
    ssl_path = load_ssl_path()
    train_entries = D.build_split("train", patch_radius=patch_radius, limit=args.limit_train)
    stats = D.fit_support_and_stats(train_entries)
    D.apply_stats(train_entries, stats)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = C.CACHE_DIR / "hr_stats.npz"
    D.save_stats(stats, stats_path)
    train_path = C.CACHE_DIR / "hr_train.pt"
    train_sha = D.save_cache(train_entries, train_path)
    val_entries = D.build_split("val", patch_radius=patch_radius, limit=args.limit_valid)
    D.apply_stats(val_entries, stats)
    val_path = C.CACHE_DIR / "hr_val.pt"
    val_sha = D.save_cache(val_entries, val_path)

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "ssl": {"path": str(ssl_path.relative_to(C.REPO)), "sha256": C.sha256_file(ssl_path)},
        "patch_radius": patch_radius,
        "splits": {
            "train": {"n": len(train_entries), "cache": str(train_path.relative_to(C.REPO)), "sha256": train_sha},
            "valid": {"n": len(val_entries), "cache": str(val_path.relative_to(C.REPO)), "sha256": val_sha},
            "official_test_loaded": False,
        },
        "stats": {
            "path": str(stats_path.relative_to(C.REPO)),
            "sha256": C.sha256_file(stats_path),
            "support": {
                "n_train_bonds": stats["support"]["n_bonds"],
                "n_unique_keys": stats["support"]["n_unique"],
                "logc_mean": stats["support"]["logc_mean"],
                "logc_scale": stats["support"]["logc_scale"],
            },
            "patch": {
                "mean": np.asarray(stats["patch"]["mean"]).tolist(),
                "scale": np.asarray(stats["patch"]["scale"]).tolist(),
            },
            "global": {
                "mean": np.asarray(stats["global"]["mean"]).tolist(),
                "scale": np.asarray(stats["global"]["scale"]).tolist(),
            },
            "fit_split": "train only",
        },
        "labels_read_during_prep": False,
        "prep_seconds": time.time() - t0,
        "train_counts": {
            "n_atoms": int(sum(e["n_atoms"] for e in train_entries)),
            "n_bonds": int(sum(e["n_bonds"] for e in train_entries)),
            "n_junctions": int(sum(e["n_junc"] for e in train_entries)),
            "n_patch_nodes": int(sum(len(e["pnode_atom"]) for e in train_entries)),
            "n_patch_edges": int(sum(len(e["pedge_x"]) for e in train_entries)),
            "n_relation_pairs": int(sum(len(e["r_u"]) for e in train_entries)),
        },
    }
    manifest_path = Path(args.manifest) if args.manifest else C.ANALYSIS_DIR / "prep.json"
    if not manifest_path.is_absolute():
        manifest_path = C.REPO / manifest_path
    C.write_json(manifest_path, payload)
    print(json.dumps({k: payload[k] for k in ("prep_seconds", "train_counts")}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _lr_at(step: int, total: int, peak: float, low: float, warmup: int) -> float:
    step = int(step)
    if step < warmup:
        return peak * float(step + 1) / float(warmup)
    progress = float(step + 1 - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return low + 0.5 * (peak - low) * (1.0 + float(np.cos(np.pi * progress)))


def _eval_predict(model: M.HRNet, entries: Sequence[Mapping[str, Any]], batch_size: int, device, relation_mode=None):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(entries), batch_size):
            batch = D.move(D.collate(entries[i : i + batch_size]), device)
            preds.append(model(batch, relation_mode=relation_mode).detach().float().cpu().numpy())
    model.train()
    if not preds:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(preds, axis=0)


def _checkpoint_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _hash_state(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        v = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def cmd_train(args: argparse.Namespace) -> int:
    cfg = C.load_config()
    recipe = dict(cfg["recipe"])
    epochs = int(args.epochs if args.epochs is not None else recipe["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else recipe["batch_size"])
    valid_batch = int(recipe["valid_batch_size"])
    device = torch.device(args.device)
    arm = args.arm
    if arm not in ("A", "B"):
        raise SystemExit("formal arms are A and B only")
    out_dir = Path(args.out) if args.out else C.RUNS_DIR / f"HR_{arm}_s0"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_entries = D.load_cache(C.CACHE_DIR / "hr_train.pt")
    val_entries = D.load_cache(C.CACHE_DIR / "hr_val.pt")
    if args.limit_train:
        train_entries = train_entries[: int(args.limit_train)]
    if args.limit_valid:
        val_entries = val_entries[: int(args.limit_valid)]
    y_train = load_y("train")
    y_valid = load_y("val")[: len(val_entries)]
    if args.limit_train:
        y_train = y_train[: len(train_entries)]

    ssl_path = load_ssl_path()
    model, model_info = M.build_model(arm=arm, ssl_path=ssl_path, seed=int(recipe["seed"]), cfg=cfg["model"], device=device)
    shared_hash = M.shared_state_hash(model)
    if model_info["parameters"] > 500_000:
        raise SystemExit(f"parameter budget exceeded: {model_info['parameters']}")

    steps_per_epoch = int(np.ceil(len(train_entries) / batch_size))
    total_steps = steps_per_epoch * epochs
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["lr"]),
        weight_decay=float(recipe["weight_decay"]),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    gen = torch.Generator()
    gen.manual_seed(int(recipe["shuffle_seed"]))
    order_hashes: list[str] = []
    curve: list[dict[str, Any]] = []
    best_valid = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_train_online: float | None = None
    final_state: dict[str, torch.Tensor] | None = None
    start_wall = time.time()
    epoch_seconds: list[float] = []
    n_seen = 0
    step = 0

    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        perm = torch.randperm(len(train_entries), generator=gen).numpy().astype(np.int64)
        order_hashes.append(hashlib.sha256(perm.tobytes()).hexdigest())
        online_sum = 0.0
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            lr = _lr_at(step, total_steps, float(recipe["lr"]), float(recipe["min_lr"]), int(recipe["warmup_steps"]))
            for group in opt.param_groups:
                group["lr"] = lr
            batch = D.move(D.collate([train_entries[j] for j in idx]), device)
            target = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = (pred - target).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            opt.step()
            online_sum += float(loss.detach()) * len(idx)
            n_seen += len(idx)
            step += 1
        online_train = online_sum / max(len(perm), 1)
        pred_valid = _eval_predict(model, val_entries, valid_batch, device)
        valid_mae = C.mae(pred_valid, y_valid)
        if valid_mae < best_valid:
            best_valid = valid_mae
            best_epoch = epoch
            best_state = _checkpoint_state(model)
            best_train_online = online_train
        epoch_seconds.append(time.time() - t_epoch)
        curve.append(
            {
                "epoch": epoch,
                "online_train_mae": online_train,
                "valid_mae": valid_mae,
                "lr": lr,
                "seconds": epoch_seconds[-1],
                "steps": steps_per_epoch,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[{arm}] ep {epoch:3d}/{epochs} lr {lr:.2e} train {online_train:.5f} "
                f"valid {valid_mae:.6f} best {best_valid:.6f}@{best_epoch} ({epoch_seconds[-1]:.2f}s)",
                flush=True,
            )
        with open(out_dir / "curve.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(curve[0]))
            w.writeheader()
            w.writerows(curve)

    final_state = _checkpoint_state(model)
    # eval-mode train MAE at best / final (full train rows only)
    if args.skip_train_eval:
        train_mae_best = None
        train_mae_final = None
    else:
        model.load_state_dict(best_state)
        pred_best = _eval_predict(model, train_entries, valid_batch, device)
        train_mae_best = C.mae(pred_best, y_train)
        model.load_state_dict(final_state)
        pred_final = _eval_predict(model, train_entries, valid_batch, device)
        train_mae_final = C.mae(pred_final, y_train)
    # restore best for the saved prediction
    model.load_state_dict(best_state)
    pred_valid_best = _eval_predict(model, val_entries, valid_batch, device)
    if len(val_entries) == C.N_VALID:
        (out_dir / "predictions").mkdir(exist_ok=True)
        np.savez(
            out_dir / "predictions/valid_best.npz",
            row=np.arange(C.N_VALID, dtype=np.int64),
            pred=pred_valid_best.astype(np.float32),
        )

    torch.save({"state": best_state, "arm": arm, "epoch": best_epoch, "valid_mae": best_valid}, out_dir / "best.pt")
    torch.save({"state": final_state, "arm": arm, "epoch": epochs, "valid_mae": curve[-1]["valid_mae"]}, out_dir / "final.pt")
    best_sha = _hash_state(best_state)
    final_sha = _hash_state(final_state)
    wall = time.time() - start_wall
    gpu_hours = wall / 3600.0
    train_info = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "seed": int(recipe["seed"]),
        "git_commit": git_commit(),
        "environment": environment(),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "ssl": model_info["ssl"],
        "model_info": {
            "parameters": model_info["parameters"],
            "parameter_tensors": model_info["parameter_tensors"],
            "parameter_breakdown": M.parameter_breakdown(model),
            "shared_state_hash": shared_hash,
            "arm": arm,
        },
        "data": {
            "train_rows": len(train_entries),
            "valid_rows": len(val_entries),
            "official_test_loaded": False,
            "train_label_sha256": label_hash(y_train),
            "valid_label_sha256": label_hash(y_valid),
            "valid_labels_used_for": "per-epoch monitoring and best-valid checkpoint selection only",
        },
        "recipe": {
            "optimizer": recipe["optimizer"],
            "lr": recipe["lr"],
            "min_lr": recipe["min_lr"],
            "weight_decay": recipe["weight_decay"],
            "warmup_steps": recipe["warmup_steps"],
            "epochs": epochs,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "updates": total_steps,
            "grad_clip": recipe["grad_clip"],
            "loss": recipe["loss"],
            "shuffle_seed": recipe["shuffle_seed"],
            "selection": recipe["selection"],
        },
        "best_valid": {
            "valid_mae": best_valid,
            "epoch": best_epoch,
            "online_train_mae": best_train_online,
            "train_mae_eval_mode": train_mae_best,
            "checkpoint_sha256": best_sha,
        },
        "final": {
            "valid_mae": curve[-1]["valid_mae"],
            "epoch": epochs,
            "train_mae_eval_mode": train_mae_final,
            "checkpoint_sha256": final_sha,
        },
        "order_hashes": {
            "first": order_hashes[0] if order_hashes else "",
            "last": order_hashes[-1] if order_hashes else "",
            "n_epochs": len(order_hashes),
        },
        "timing": {
            "wall_seconds": wall,
            "gpu_hours": gpu_hours,
            "epoch_seconds_mean": float(np.mean(epoch_seconds)) if epoch_seconds else 0.0,
            "epoch_seconds_median": float(np.median(epoch_seconds)) if epoch_seconds else 0.0,
            "peak_cuda_memory_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0,
        },
        "curve": curve,
    }
    C.write_json(out_dir / "train.json", train_info)
    print(json.dumps({k: train_info[k] for k in ("best_valid", "final", "timing")}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).expanduser().resolve()
    train_info = C.read_json(run_dir / "train.json")
    z = np.load(run_dir / "predictions/valid_best.npz", allow_pickle=False)
    pred = z["pred"].astype(np.float64)
    rows = z["row"].astype(np.int64)
    if len(pred) != C.N_VALID or not np.array_equal(rows, np.arange(C.N_VALID)):
        raise SystemExit("run predictions are not the full 1000-row valid split")
    groups = C.load_valid_groups()
    y = groups["y"]
    p3b = C.load_p3b_valid()
    if not np.allclose(p3b["pred"], groups["p3b_pred"], rtol=0.0, atol=1e-6):
        raise SystemExit(
            f"two frozen P3B valid arrays disagree: max abs diff "
            f"{float(np.max(np.abs(p3b['pred'] - groups['p3b_pred'])))}"
        )
    ae_c = np.abs(pred - y)
    ae_r = np.abs(p3b["pred"] - y)
    rec_mae = float(ae_c.mean())
    if abs(rec_mae - float(train_info["best_valid"]["valid_mae"])) > 5e-6:
        raise SystemExit(f"recomputed valid MAE {rec_mae} != recorded {train_info['best_valid']['valid_mae']}")

    gate_cfg = C.GATE
    fixed_top10 = groups["p3b_top10"].tolist()
    drop = C.topk_removed_gain(ae_c, ae_r, k=int(gate_cfg["drop_top_k"]), fixed_rows=fixed_top10)
    boot = C.paired_bootstrap(ae_c, ae_r)
    groups_table = C.group_table(
        ae_c,
        ae_r,
        {
            "all": np.ones(len(y), dtype=bool),
            "radius2_unseen": groups["radius2_unseen"],
            "radius2_covered": ~groups["radius2_unseen"],
            "ordinary": ~groups["tail"],
            "tail": groups["tail"],
            "ring_rich": groups["ring_rich"],
            "ring_rich_and_unseen": groups["ring_rich"] & groups["radius2_unseen"],
            "p3b_worst100": np.isin(np.arange(len(y)), groups["p3b_worst100"]),
        },
    )
    train_mae_best = train_info["best_valid"]["train_mae_eval_mode"]
    checks = {
        "best_valid_mae_le_0.073": bool(rec_mae <= float(gate_cfg["best_valid_mae_max"])),
        "drop_top10_gain_ge_0.002": bool(drop["gain_drop_topk"] >= float(gate_cfg["drop_top10_min_gain"])),
        "train_mae_at_best_le_0.026": bool(train_mae_best is not None and train_mae_best <= float(gate_cfg["train_mae_at_best_max"])),
    }
    eligible = all(checks.values())
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": train_info["arm"],
        "run_dir": str(run_dir.relative_to(C.REPO)) if run_dir.is_relative_to(C.REPO) else str(run_dir),
        "git_commit": train_info["git_commit"],
        "config_sha256": train_info["config_sha256"],
        "parameters": train_info["model_info"]["parameters"],
        "best_valid": train_info["best_valid"],
        "final_valid": train_info["final"],
        "timing": train_info["timing"],
        "metrics": C.metric_block(pred, y),
        "reference": {
            "name": "P3B_s0",
            "mae": float(ae_r.mean()),
            "prediction_sha256": p3b["sha256"],
            "train_mae_at_best": C.P3B_TRAIN_MAE_AT_BEST,
        },
        "paired_bootstrap": boot,
        "drop_top10": drop,
        "groups": groups_table,
        "gate": {"checks": checks, "eligible": bool(eligible), "config": gate_cfg},
        "official_test_loaded": False,
    }
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{train_info['arm']}_s0"
    C.write_json(C.ANALYSIS_DIR / f"score_{tag}.json", payload)
    # per-molecule table
    pd_path = C.ANALYSIS_DIR / f"per_molecule_{tag}.csv"
    with open(pd_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "y", "pred", "ae", "p3b_pred", "p3b_ae", "delta_ae", "radius2_unseen", "tail", "ring_rich"])
        for i in range(len(y)):
            w.writerow([i, y[i], pred[i], ae_c[i], p3b["pred"][i], ae_r[i], ae_c[i] - ae_r[i],
                        int(groups["radius2_unseen"][i]), int(groups["tail"][i]), int(groups["ring_rich"][i])])
    print(json.dumps({"arm": train_info["arm"], "valid_mae": rec_mae, "reference_mae": float(ae_r.mean()),
                      "gain": float(ae_r.mean() - rec_mae), "drop_top10": drop,
                      "checks": checks, "eligible": bool(eligible)}, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    paths = [C.ANALYSIS_DIR / f"score_{t}.json" for t in args.tags]
    data = [C.read_json(p) for p in paths]
    out = {"protocol_id": C.PROTOCOL_ID, "scores": data}
    if len(data) == 2:
        a, b = data
        def _run_pred(rec):
            p = Path(rec["run_dir"])
            if not p.is_absolute():
                p = C.REPO / p
            return np.load(p / "predictions/valid_best.npz", allow_pickle=False)["pred"].astype(np.float64)
        pa = _run_pred(a)
        pb = _run_pred(b)
        groups = C.load_valid_groups()
        y = groups["y"]
        out["A_minus_B"] = C.paired_bootstrap(np.abs(pa - y), np.abs(pb - y))
        out["prediction_correlation"] = float(np.corrcoef(pa, pb)[0, 1])
        out["error_correlation"] = float(np.corrcoef(np.abs(pa - y), np.abs(pb - y))[0, 1])
    C.write_json(C.ANALYSIS_DIR / "comparison.json", out)
    print(json.dumps(out.get("A_minus_B", {}), indent=2))
    return 0


# ---------------------------------------------------------------------------
# smoke / profile
# ---------------------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> int:
    """Timing + FLOPs for one real batch, no weight updates kept."""
    cfg = C.load_config()
    device = torch.device(args.device)
    entries = D.load_cache(C.CACHE_DIR / "hr_train.pt")[: int(args.n_molecules)]
    y = load_y("train")[: len(entries)]
    ssl_path = load_ssl_path()
    out: dict[str, Any] = {"device": str(device), "n_molecules": len(entries)}
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    for arm in ("A", "B"):
        model, info = M.build_model(arm=arm, ssl_path=ssl_path, seed=int(cfg["recipe"]["seed"]), cfg=cfg["model"], device=device)
        batch = D.move(D.collate(entries), device)
        target = torch.as_tensor(y, dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # warmup
        for _ in range(2):
            pred = model(batch)
            loss = (pred - target).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        n_rep = int(args.repeat)
        for _ in range(n_rep):
            pred = model(batch)
            loss = (pred - target).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / n_rep
        flops = None
        try:
            from torch.utils.flop_counter import FlopCounterMode

            with FlopCounterMode(display=False) as counter:
                model(batch)
            flops = int(counter.get_total_flops())
        except Exception as exc:  # pragma: no cover - diagnostic only
            flops = None
            out[f"flop_error_{arm}"] = repr(exc)
        out[arm] = {
            "parameters": info["parameters"],
            "seconds_per_batch": dt,
            "forward_backward_flops": flops,
            "peak_cuda_memory_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0,
            "batch_tokens": int(batch["n_tokens_total"]),
            "batch_pairs": int(batch["r_src"].numel()),
            "batch_patch_nodes": int(batch["pnode_atom"].numel()),
        }
        del model, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    C.write_json(C.ANALYSIS_DIR / f"profile_{args.tag}.json", out)
    print(json.dumps(out, indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """One short real training run per arm on a fixed subset (budgeted)."""
    cfg = C.load_config()
    device = torch.device(args.device)
    ssl_path = load_ssl_path()
    train_entries = D.load_cache(C.CACHE_DIR / "hr_train.pt")[: int(args.limit_train)]
    val_entries = D.load_cache(C.CACHE_DIR / "hr_val.pt")[: int(args.limit_valid)]
    y_train = load_y("train")[: len(train_entries)]
    y_val = load_y("val")[: len(val_entries)]
    out: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "limit_train": len(train_entries),
                           "limit_valid": len(val_entries), "device": str(device)}
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    for arm in args.arms:
        model, info = M.build_model(arm=arm, ssl_path=ssl_path, seed=int(cfg["recipe"]["seed"]), cfg=cfg["model"], device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["recipe"]["lr"]), weight_decay=float(cfg["recipe"]["weight_decay"]))
        batch_size = int(cfg["recipe"]["batch_size"])
        steps = int(args.steps)
        gen = torch.Generator(); gen.manual_seed(int(cfg["recipe"]["shuffle_seed"]))
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train_entries), generator=gen).numpy()
        losses = []
        pos = 0
        for step in range(steps):
            if pos + batch_size > len(perm):
                perm = torch.randperm(len(train_entries), generator=gen).numpy(); pos = 0
            idx = perm[pos : pos + batch_size]; pos += batch_size
            batch = D.move(D.collate([train_entries[j] for j in idx]), device)
            target = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = (pred - target).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["recipe"]["grad_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        pred = _eval_predict(model, val_entries, int(cfg["recipe"]["valid_batch_size"]), device)
        out[arm] = {
            "parameters": info["parameters"],
            "steps": steps,
            "seconds": dt,
            "seconds_per_step": dt / max(steps, 1),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "valid_mae": C.mae(pred, y_val),
            "peak_cuda_memory_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    C.write_json(C.ANALYSIS_DIR / f"smoke_{args.tag}.json", out)
    print(json.dumps(out, indent=2))
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prep")
    q.add_argument("--limit-train", type=int, default=None)
    q.add_argument("--limit-valid", type=int, default=None)
    q.add_argument("--manifest", default=None,
                   help="manifest path (default analysis/prep.json; use cache/prep.json to keep the remote tree clean)")
    q.set_defaults(func=cmd_prep)

    q = sub.add_parser("train")
    q.add_argument("--arm", choices=("A", "B"), required=True)
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--epochs", type=int, default=None)
    q.add_argument("--batch-size", type=int, default=None)
    q.add_argument("--limit-train", type=int, default=None)
    q.add_argument("--limit-valid", type=int, default=None)
    q.add_argument("--out", default=None)
    q.add_argument("--skip-train-eval", action="store_true")
    q.set_defaults(func=cmd_train)

    q = sub.add_parser("score")
    q.add_argument("--run", required=True)
    q.add_argument("--tag", default=None)
    q.set_defaults(func=cmd_score)

    q = sub.add_parser("compare")
    q.add_argument("--tags", nargs=2, required=True)
    q.set_defaults(func=cmd_compare)

    q = sub.add_parser("profile")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--n-molecules", type=int, default=128)
    q.add_argument("--repeat", type=int, default=5)
    q.add_argument("--tag", default="gpu0")
    q.set_defaults(func=cmd_profile)

    q = sub.add_parser("smoke")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--arms", nargs="+", default=["A", "B"])
    q.add_argument("--limit-train", type=int, default=1024)
    q.add_argument("--limit-valid", type=int, default=128)
    q.add_argument("--steps", type=int, default=20)
    q.add_argument("--tag", default="gpu")
    q.set_defaults(func=cmd_smoke)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
