"""Prep / GPU timing / horizon planning for DBT-PA.

``prep`` builds (or reuses) the frozen local-stack features for train and valid,
checks them against a live frozen-stack forward on a few graphs, and writes the
cache hashes.  ``timing`` runs a short full-recipe schedule (including the
complete valid forward) and writes the measured epoch times.  ``plan`` turns
those measurements into the formal epoch count E; it never looks at valid MAE.

Nothing here reads the official test split.
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

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import dbt_data as BD  # noqa: E402
import dbt_pa_cache as PC  # noqa: E402
import dbt_pa_models as PM  # noqa: E402
import dbt_pa_train as BT  # noqa: E402

ANALYSIS = _EXP / "analysis"
DBT_ANALYSIS = _SSUF / "zinc_directed_bond_transformer_v1" / "analysis"


def _scales() -> dict:
    path = ANALYSIS / "graph_scales.json"
    if path.exists():
        return C.read_json(path)
    src = DBT_ANALYSIS / "graph_scales.json"
    if not src.exists():
        raise FileNotFoundError(f"missing graph scales {src}")
    data = C.read_json(src)
    C.write_json(path, data)
    return data


def prep(device, *, force: bool = False) -> dict:
    model, info = PM.build_model_pa(D.load_norm(), 0, device,
                                    graph_scales=_scales(), dropout=float(BT.RECIPE["dropout"]),
                                    adapter_rank=int(BT.RECIPE["adapter_rank"]))
    ssl_sha = str(info["ssl"]["sha256"])
    enc_sha = str(info["encoder_sha256"])
    out: dict[str, Any] = {
        "device": str(device),
        "parameters": info["parameters"],
        "trainable_parameters": info["trainable_parameters"],
        "frozen_parameters": info["frozen_parameters"],
        "ssl_sha256": ssl_sha,
        "encoder_sha256": enc_sha,
        "ssl_source": info["ssl"]["source"],
        "ssl_tensor_match": info["ssl_tensor_match"],
        "caches": {},
    }
    for split, rows in (
        ("train", [0, 1, 17, 123, 999, 5000, 9998, 9999]),
        ("valid", [0, 172, 500, 999]),
    ):
        obj = BD.load_object_cache(split)
        struct = PC.structural_fingerprint(obj)
        feat = PC.load_feature_split(split, model, ssl_sha256=ssl_sha,
                                     encoder_sha256=enc_sha,
                                     structure_sha256=struct, require=False)
        source = "reuse"
        if feat is None or force:
            feat, path = PC.build_feature_split(split, model, device,
                                                ssl_sha256=ssl_sha,
                                                encoder_sha256=enc_sha,
                                                batch_size=128)
            source = "build"
        else:
            path = Path(feat["_path"])
            # Reload after the identity check so _path/_sha256 are populated.
            feat = PC.load_feature_split(split, model, ssl_sha256=ssl_sha,
                                         encoder_sha256=enc_sha,
                                         structure_sha256=struct, require=True)
        check = PC.verify_feature_split(model, split, feat, rows, device)
        out["caches"][split] = {
            "source": source,
            "path": str(path),
            "sha256": C.sha256_file(path),
            "meta": {k: v for k, v in feat["meta"].items()
                     if k in ("feature_version", "split", "ssl_sha256",
                              "encoder_sha256", "structure_sha256",
                              "n_atoms_total", "n_bonds_total",
                              "n_junctions_total", "build_seconds", "dtype")},
            "verification": check,
        }
    out["all_cache_checks_pass"] = all(v["verification"]["all_pass"]
                                       for v in out["caches"].values())
    C.write_json(ANALYSIS / "prep.json", out)
    C.log(json.dumps({k: v for k, v in out.items() if k != "caches"}, indent=2))
    for split, row in out["caches"].items():
        C.log(f"[pa prep] {split} {row['source']} {row['sha256'][:12]} "
              f"max_diff={row['verification']['a']['max_abs_diff']:.3e}/"
              f"{row['verification']['b']['max_abs_diff']:.3e}/"
              f"{row['verification']['w']['max_abs_diff']:.3e}")
    if not out["all_cache_checks_pass"]:
        raise RuntimeError("cache verification failed")
    return out


def timing(device, epochs: int = 4) -> dict:
    t0 = time.perf_counter()
    info = BT.train(0, device, epochs=int(epochs), tag="timing", force=True,
                    timing_only=True)
    wall = time.perf_counter() - t0
    secs = [float(row["seconds"]) for row in info["curve"]]
    steady = secs[1:] if len(secs) > 1 else secs
    steady_mean = float(np.mean(steady))
    out = {
        "device": str(device),
        "epochs_measured": int(epochs),
        "epoch_seconds": secs,
        "first_epoch_seconds": secs[0],
        "steady_epoch_seconds": steady_mean,
        "steady_epoch_seconds_median": float(np.median(steady)),
        "safe_epoch_seconds": float(1.15 * steady_mean),
        "steps_per_epoch": int(BT.STEPS_PER_EPOCH),
        "valid_batch_size": int(BT.RECIPE["valid_batch_size"]),
        "train_batch_size": int(BT.RECIPE["batch_size"]),
        "timing_wall_seconds": float(wall),
        "timing_train_loop_seconds": float(info["wall_seconds"]),
        "timing_gpu_seconds": info["gpu_seconds"],
        "peak_gpu_mem_mb": info["peak_gpu_mem_mb"],
        "parameters": info["parameter_audit"]["parameters_total_inference"],
        "trainable_parameters": info["parameter_audit"]["parameters_trainable"],
        "cache_build_seconds": {
            split: info["feature_cache"][split]["build_seconds"]
            for split in ("train", "valid")
        },
        "valid_mae_timing_only": info["best_valid"]["valid_mae"],
        "valid_mae_is_not_used_for_horizon": True,
        "environment": info["environment"],
    }
    C.write_json(ANALYSIS / "timing.json", out)
    C.log(json.dumps(out, indent=2))
    return out


def plan(*, spent_seconds: float, budget_seconds: float = 3600.0,
         reserve_seconds: float = 180.0) -> dict:
    timing_data = C.read_json(ANALYSIS / "timing.json")
    safe = float(timing_data["safe_epoch_seconds"])
    remaining = float(budget_seconds) - float(spent_seconds) - float(reserve_seconds)
    if safe <= 0:
        raise RuntimeError("non-positive safe epoch estimate")
    raw = math.floor(remaining / safe)
    epochs = int(min(int(BT.RECIPE["max_epochs_cap"]), raw) // 10 * 10)
    out = {
        "budget_seconds": float(budget_seconds),
        "gpu_seconds_spent_before_formal": float(spent_seconds),
        "final_scoring_reserve_seconds": float(reserve_seconds),
        "remaining_seconds": float(remaining),
        "steady_epoch_seconds": float(timing_data["steady_epoch_seconds"]),
        "safety_factor": 1.15,
        "safe_epoch_seconds": safe,
        "raw_epoch_cap": int(raw),
        "epochs_cap": int(BT.RECIPE["max_epochs_cap"]),
        "E": int(epochs),
        "steps_per_epoch": int(BT.STEPS_PER_EPOCH),
        "total_updates": int(epochs * BT.STEPS_PER_EPOCH),
        "projected_formal_seconds": float(epochs * safe),
        "projected_total_seconds": float(spent_seconds + epochs * safe + reserve_seconds),
        "decision_rule": "E = min(600, floor(remaining / (1.15 * steady_epoch))) floored to 10",
        "valid_mae_not_used": True,
    }
    if epochs < 200:
        out["status"] = "BLOCKED_TOO_SHORT"
        C.write_json(ANALYSIS / "timing_plan.json", out)
        raise RuntimeError(f"E={epochs} < 200; formal run not launched")
    out["status"] = "OK"
    C.write_json(ANALYSIS / "timing_plan.json", out)
    C.log(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("timing")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--epochs", type=int, default=4)
    s = sub.add_parser("plan")
    s.add_argument("--spent-seconds", type=float, required=True)
    s.add_argument("--budget-seconds", type=float, default=3600.0)
    s.add_argument("--reserve-seconds", type=float, default=180.0)
    args = p.parse_args(argv)
    if args.cmd == "prep":
        prep(torch.device(args.device), force=args.force)
        return 0
    if args.cmd == "timing":
        timing(torch.device(args.device), args.epochs)
        return 0
    if args.cmd == "plan":
        plan(spent_seconds=args.spent_seconds, budget_seconds=args.budget_seconds,
             reserve_seconds=args.reserve_seconds)
        return 0
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
