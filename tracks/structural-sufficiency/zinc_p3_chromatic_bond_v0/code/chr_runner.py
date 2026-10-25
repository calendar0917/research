"""Execution entry points for zinc-p3-chromatic-bond-v0.

Commands
--------
prep    build the label-free train/valid relation cache
smoke    one GPU/CPU forward+backward sanity run (no formal result)
timing  three complete full-train/full-valid epochs for one arm
train   the frozen 600-epoch formal run for one arm

Official test is never referenced.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SS = _EXP.parent
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import chr_data as R  # noqa: E402
import chr_models as M  # noqa: E402

EXP = _EXP
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
CONFIG = EXP / "configs" / "chr_v0.json"
MEAN_LINE_DEGREE = 1.308120486872196
RECIPE = "B"


def _cfg() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def run_dir(arm: str, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}{arm}_s0"


def _builder(arm: str):
    def builder(candidate, norm, seed, device):
        return M.build_model(arm, norm, seed, device,
                             train_mean_log1p_line_degree=MEAN_LINE_DEGREE)
    return builder


def _check_data() -> None:
    missing = [p for p in (D.cache_path("train"), D.cache_path("valid")) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing train/valid P3 cache: " + ", ".join(str(p) for p in missing))
    for split in ("train", "valid"):
        if not R.meta_path(split).exists() or not R.rel_path(split).exists():
            raise FileNotFoundError(f"missing relation cache for {split}; run `prep` first")


def prep() -> dict[str, Any]:
    out = {}
    for split in ("train", "valid"):
        meta = R.build_split(split, force=False)
        out[split] = {
            "n_molecules": meta["n_molecules"],
            "total_pairs": meta["total_pairs"],
            "total_bonds": meta["total_bonds"],
            "ring_pair_fraction": meta["stats"]["ring_pair_fraction"],
            "cycle_length_histogram": meta["stats"]["cycle_length_histogram"],
            "relation_cache": meta["relation_cache"],
            "relation_cache_sha256": meta["relation_cache_sha256"],
        }
        print(json.dumps({split: out[split]}, indent=2))
    C.write_json(ANALYSIS / "prep.json", {
        "round_id": _cfg()["round_id"],
        "splits": out,
        "official_test_loaded": False,
        "utc": C.utcnow(),
    })
    return out


def _device(name: str) -> torch.device:
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return dev


def smoke(device_name: str = "cuda:0") -> dict[str, Any]:
    _check_data()
    cfg = _cfg()
    norm = D.load_norm()
    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    collate = R.make_train_eval_collate()
    rows = np.arange(32, dtype=np.int64)
    batch = T1._to_device(collate(train_cache, rows, "P3N", valid=False), torch.device(device_name))
    target = torch.as_tensor(np.asarray(train_cache["labels"], dtype=np.float32)[rows],
                             device=batch["n_bonds"].device)
    out = {}
    for arm in ("C", "S"):
        model, info = M.build_model(arm, norm, 0, device_name,
                                    train_mean_log1p_line_degree=MEAN_LINE_DEGREE)
        model.train()
        logits = model(batch)
        loss = F.l1_loss(logits, target)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        out[arm] = {
            "parameters": info["parameters"],
            "initial_state_sha256": info["initial_state_sha256"],
            "loss": float(loss.detach()),
            "pred_finite": bool(torch.isfinite(logits).all()),
            "grad_tensors": len(grads),
            "grad_finite": bool(all(torch.isfinite(g).all() for g in grads)),
            "grad_nonzero": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
        }
        print(json.dumps({arm: out[arm]}, indent=2))
    out["device"] = device_name
    out["official_test_loaded"] = False
    out["utc"] = C.utcnow()
    C.write_json(ANALYSIS / "smoke.json", out)
    return out


def _write_artifacts(out: Path, info: dict[str, Any]) -> dict[str, Any]:
    best = out / "best.pt"
    final = out / "final.pt"
    pred = out / "predictions" / "valid_best.npz"
    art = {
        "run_dir": str(out.relative_to(EXP)),
        "candidate": info.get("candidate"),
        "arm": info.get("model_info", {}).get("arm"),
        "seed": info.get("seed"),
        "recipe_variant": info.get("recipe_variant"),
        "parameter_count": info.get("parameter_count"),
        "parameters_breakdown": info.get("model_info", {}).get("parameters_breakdown"),
        "initial_state_sha256": info.get("model_info", {}).get("initial_state_sha256"),
        "dcc_d0_state_sha256": info.get("model_info", {}).get("dcc_d0_state_sha256"),
        "ssl": info.get("model_info", {}).get("ssl"),
        "best_valid": info.get("best_valid"),
        "fixed_objects": info.get("fixed_objects"),
        "gpu_hours_run_level": (info.get("gpu_hours") if info.get("gpu_hours") is not None else float(info["wall_seconds"]) / 3600.0),
        "wall_seconds": info.get("wall_seconds"),
        "sha256": {
            "best_pt": _sha(best),
            "final_pt": _sha(final),
            "valid_best_npz": _sha(pred),
            "train_json": _sha(out / "train.json"),
            "curve_csv": _sha(out / "curve.csv"),
        },
        "official_test_loaded": False,
    }
    C.write_json(out / "artifacts.json", art)
    return art


def timing(arm: str, device_name: str = "cuda:0", *, epochs: int = 3) -> dict[str, Any]:
    _check_data()
    out = run_dir(arm, tag="timing")
    t0 = time.perf_counter()
    info = T1.train(
        f"CHR_{arm}", 0, device_name, epochs=int(epochs), tag="timing", force=True,
        recipe_variant=RECIPE, model_builder=_builder(arm), out_dir=out,
        collate_fn=R.make_train_eval_collate(),
    )
    total_seconds = time.perf_counter() - t0
    curve = info["curve"]
    epoch_seconds = [float(e["seconds"]) for e in curve]
    mean_epoch = float(np.mean(epoch_seconds))
    steady = float(np.mean(epoch_seconds[1:])) if len(epoch_seconds) > 1 else mean_epoch
    final_eval_seconds = float(info["wall_seconds"]) - float(np.sum(epoch_seconds))
    startup_seconds = float(total_seconds) - float(info["wall_seconds"])
    projected_train = 600.0 * mean_epoch
    projected_total = 1.20 * projected_train + startup_seconds + final_eval_seconds + 60.0
    payload = {
        "round_id": _cfg()["round_id"],
        "arm": arm,
        "seed": 0,
        "device": device_name,
        "epochs_measured": int(len(epoch_seconds)),
        "steps_per_epoch": int(info["recipe"]["steps_per_epoch"]),
        "updates_per_epoch": int(info["recipe"]["steps_per_epoch"]),
        "epoch_seconds_including_full_valid": epoch_seconds,
        "mean_epoch_seconds": mean_epoch,
        "steady_epoch_seconds": steady,
        "sum_epoch_seconds": float(np.sum(epoch_seconds)),
        "startup_seconds": startup_seconds,
        "final_eval_seconds": final_eval_seconds,
        "projected_600ep_train_seconds": projected_train,
        "projected_600ep_total_seconds_with_20pct": projected_total,
        "projected_600ep_gpu_hours": projected_total / 3600.0,
        "run_wall_seconds": float(total_seconds),
        "train_json_wall_seconds": float(info["wall_seconds"]),
        "gpu_hours_run_level": float(info["gpu_hours"]) if info["gpu_hours"] is not None else float(info["wall_seconds"]) / 3600.0,
        "parameters": int(info["parameter_count"]),
        "initial_state_sha256": info["model_info"]["initial_state_sha256"],
        "gradient_audit": info["gradient_audit"],
        "official_test_loaded": False,
        "note": "Projection = 1.20 * 600 * mean full-epoch seconds + measured startup/final-eval + 60 s reserve.",
    }
    C.write_json(ANALYSIS / f"timing_{arm}.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def diagnose(arm: str, device_name: str = "cpu", *, n_rows: int = 256) -> dict[str, Any]:
    """Fixed-weight mechanism diagnostics from the best checkpoint (no training)."""
    _check_data()
    cfg = _cfg()
    run = run_dir(arm)
    best = run / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"missing {best}; pull the run first")
    norm = D.load_norm()
    valid_cache = D.load_cache("valid")
    collate = R.make_collate("valid")
    model, info = M.build_model(arm, norm, 0, device_name,
                                train_mean_log1p_line_degree=MEAN_LINE_DEGREE)
    state = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    rows = np.arange(min(int(n_rows), int(len(valid_cache["labels"]))), dtype=np.int64)
    labels = np.asarray(valid_cache["labels"], dtype=np.float64)[rows]
    # Train/eval identity probe (no dropout, no BN by design).
    probe_rows = rows[:min(64, len(rows))]
    probe = T1._to_device(collate(valid_cache, probe_rows, "P3N", valid=True), torch.device(device_name))
    with torch.no_grad():
        model.train()
        y_train_mode = model(probe).detach().cpu()
        model.eval()
        y_eval_mode = model(probe).detach().cpu()
    train_eval_delta = float((y_train_mode - y_eval_mode).abs().max())
    preds = []
    acc: dict[str, float] = {}
    weight = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            rr = rows[start:start + 64]
            batch = T1._to_device(collate(valid_cache, rr, "P3N", valid=True), torch.device(device_name))
            y, stats = model(batch, collect=True)
            preds.append(y.detach().cpu().numpy().reshape(-1))
            w = float(sum(s["n_pairs"] for s in stats["layers"]))
            weight += w
            for i, st in enumerate(stats["layers"]):
                for k, v in st.items():
                    if isinstance(v, (int, float, np.floating, np.integer)) and k != "n_pairs":
                        acc.setdefault(f"layer{i}.{k}", 0.0)
                        acc[f"layer{i}.{k}"] += float(v) * w
    pred = np.concatenate(preds) if preds else np.zeros(0)
    pred_off = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            rr = rows[start:start + 64]
            batch = T1._to_device(collate(valid_cache, rr, "P3N", valid=True), torch.device(device_name))
            batch["rel"] = batch["rel"].clone()
            batch["rel"][..., 21] = 0.0
            y = model(batch)
            pred_off.append(y.detach().cpu().numpy().reshape(-1))
    pred_off = np.concatenate(pred_off) if pred_off else np.zeros(0)
    ae = np.abs(pred - labels)
    ae_off = np.abs(pred_off - labels)
    layer_stats = {}
    for k, v in acc.items():
        layer, name = k.split(".", 1)
        layer_stats.setdefault(layer, {})[name] = float(v / max(weight, 1.0))
    payload = {
        "round_id": cfg["round_id"],
        "arm": arm,
        "checkpoint": str(best),
        "checkpoint_sha256": C.sha256_file(best),
        "epoch": int(state.get("epoch", -1)),
        "stored_valid_mae": float(state.get("valid_mae", float("nan"))),
        "rows": [int(x) for x in rows],
        "n_rows": int(len(rows)),
        "device": device_name,
        "base_mae": float(ae.mean()) if len(ae) else None,
        "ring_off_mae": float(ae_off.mean()) if len(ae_off) else None,
        "ring_off_mae_delta": float((ae_off - ae).mean()) if len(ae) else None,
        "ring_off_mean_abs_pred_delta": float(np.abs(pred_off - pred).mean()) if len(pred) else None,
        "ring_off_max_abs_pred_delta": float(np.abs(pred_off - pred).max()) if len(pred) else None,
        "rel_channel_weight_rms": {
            f"layer{i}": float(layer.rel_channel.weight.detach().pow(2).mean().sqrt())
            for i, layer in enumerate(model.layers)
        },
        "layer_stats": layer_stats,
        "train_eval_max_abs_delta": train_eval_delta,
        "official_test_loaded": False,
        "utc": C.utcnow(),
    }
    C.write_json(run / "diagnostics.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def train(arm: str, device_name: str = "cuda:0", *, epochs: int | None = None,
          force: bool = False) -> dict[str, Any]:
    _check_data()
    out = run_dir(arm)
    info = T1.train(
        f"CHR_{arm}", 0, device_name, epochs=epochs, tag=None, force=force,
        recipe_variant=RECIPE, model_builder=_builder(arm), out_dir=out,
        collate_fn=R.make_train_eval_collate(),
    )
    art = _write_artifacts(out, info)
    C.log(
        f"[CHR-{arm}] best={info['best_valid']['valid_mae']:.9f}"
        f"@{info['best_valid']['epoch']} final={info['fixed_objects']['final']['valid_mae']:.9f} "
        f"params={info['parameter_count']} wall={info['wall_seconds']:.1f}s "
        f"init={info['model_info']['initial_state_sha256'][:16]}"
    )
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "smoke", "timing", "train", "diagnose"])
    p.add_argument("--arm", choices=["C", "S"], default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.command == "prep":
        prep()
    elif args.command == "smoke":
        smoke(args.device)
    elif args.command == "timing":
        if args.arm is None:
            raise SystemExit("timing needs --arm")
        timing(args.arm, args.device, epochs=args.epochs or 3)
    elif args.command == "train":
        if args.arm is None:
            raise SystemExit("train needs --arm")
        train(args.arm, args.device, epochs=args.epochs, force=args.force)
    elif args.command == "diagnose":
        if args.arm is None:
            raise SystemExit("diagnose needs --arm")
        diagnose(args.arm, args.device, n_rows=args.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
