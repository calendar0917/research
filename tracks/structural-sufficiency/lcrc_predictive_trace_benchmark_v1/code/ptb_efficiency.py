"""Efficiency table (same environment, same batch, same data structure, paired A0/C1).

Measures the quantities named in ``PROTOCOL.md`` §4:

* trainable parameter count and parameter memory,
* structural cache build time,
* preprocessing data size / cache footprint,
* per-epoch training time (from the measured cost run, same environment),
* cumulative GPU time before the best checkpoint is selected,
* fixed-batch inference throughput,
* CUDA peak allocated,
* long-tail graph cost.

Every row carries a scope label (``MEASURED_THIS_ROUND``,
``MEASURED_UPSTREAM_SAME_ENV``, ``ESTIMATED_FROM_BEST_EPOCH``).  The two variants
share the data pipeline, so cache build time and footprint are measured once per
dataset and reported with ``shared=True``.

Structural inference is measured on the **Validation** cache only: the Test split
is never touched here, so this script cannot consume the one-shot Test read.
"""

from __future__ import annotations

import argparse
import platform
import time
from typing import Any, Sequence

import numpy as np
import torch

import ptb_common as C
import ptb_data as D
import ptb_model as M
import ptb_train_mol as TM


def _cache_mb(cache) -> float:
    nbytes = 0
    for value in cache.values():
        if isinstance(value, np.ndarray):
            nbytes += int(value.nbytes)
    return nbytes / 2**20


def _cache_build_seconds(dataset: str) -> dict[str, Any]:
    """Time to construct the frozen train+valid view (warm filesystem cache)."""
    t0 = time.perf_counter()
    D.molecule_view(dataset)
    return {
        "seconds": time.perf_counter() - t0,
        "scope": "MEASURED_THIS_ROUND_WARM_FS_CACHE",
        "note": "includes loading the frozen split and the on-disk structure caches; a cold build also writes them",
    }


def _params(spec, norm, model_name: str) -> dict[str, Any]:
    model, _info = M.build_model(spec, norm, model_name, 0, "cpu")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "param_mb_fp32": total * 4 / 2**20,
    }


@torch.no_grad()
def _inference(spec, norm, model_name: str, cache, device: str, batch_size: int = 128) -> dict[str, Any]:
    model, _info = M.build_model(spec, norm, model_name, 0, device)
    model.eval()
    TM.predict_matrix(model, cache, device, batch_size=batch_size)  # warm-up
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = TM.predict_matrix(model, cache, device, batch_size=batch_size)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n = int(out.shape[0])
    return {
        "n_graphs": n,
        "batch_size": int(batch_size),
        "wall_seconds": elapsed,
        "graphs_per_second": n / elapsed if elapsed > 0 else float("nan"),
        "ms_per_graph": 1000.0 * elapsed / n if n else float("nan"),
        "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 2**20)
        if device.startswith("cuda")
        else None,
        "split": "valid",
    }


@torch.no_grad()
def _long_tail(spec, norm, model_name: str, cache, device: str, batch_size: int = 128) -> dict[str, Any]:
    """Per-graph inference cost as a function of graph size (atom-count deciles)."""
    model, _info = M.build_model(spec, norm, model_name, 0, device)
    model.eval()
    atom_ptr = np.asarray(cache["atom_ptr"])
    n_atoms = atom_ptr[1:] - atom_ptr[:-1]
    order = np.argsort(n_atoms, kind="stable")
    buckets = np.array_split(order, 10)
    rows = []
    for bi, idx in enumerate(buckets):
        if idx.size == 0:
            continue
        sub = TM._subset_cache(cache, idx.tolist(), spec)
        TM.predict_matrix(model, sub, device, batch_size=batch_size)  # warm
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        TM.predict_matrix(model, sub, device, batch_size=batch_size)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append(
            {
                "bucket": bi,
                "n_graphs": int(idx.size),
                "atom_min": int(n_atoms[idx].min()),
                "atom_max": int(n_atoms[idx].max()),
                "atom_mean": float(n_atoms[idx].mean()),
                "ms_per_graph": 1000.0 * dt / idx.size,
            }
        )
    costs = [r["ms_per_graph"] for r in rows if r["ms_per_graph"] > 0]
    return {
        "status": "OK" if costs else "UNAVAILABLE",
        "buckets": rows,
        "max_over_min_cost_ratio": float(max(costs) / min(costs)) if costs else None,
        "note": "ratio between the largest-atom and smallest-atom decile at the same batch size",
    }


def _cumulative_gpu_before_best(run_dir) -> dict[str, Any] | None:
    train_json = run_dir / "train.json"
    select_json = run_dir / "selection.json"
    if not train_json.exists():
        return None
    payload = C.read_json(train_json)
    total = float(payload.get("train_wall_seconds", float("nan")))
    epochs = int(payload.get("epochs", 0) or 0)
    best = C.read_json(select_json).get("best_epoch") if select_json.exists() else None
    if best is None or epochs <= 0 or not np.isfinite(total):
        return {"scope": "UNAVAILABLE", "reason": "train.json / selection.json incomplete"}
    return {
        "scope": "ESTIMATED_FROM_BEST_EPOCH",
        "best_epoch": int(best),
        "epochs": epochs,
        "train_wall_seconds_total": total,
        "seconds_before_best": total * (int(best) / float(epochs)),
        "note": "linear estimate from the completed run; per-epoch wall times are not stored individually",
    }


def run(datasets: Sequence[str], device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "lcrc-ptb-efficiency-v1",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cost_measurement": None,
        "rows": [],
        "cache": {},
        "cumulative_gpu_before_best": {},
    }
    cost_path = C.ANALYSIS / "cost_measurement.json"
    if cost_path.exists():
        payload["cost_measurement"] = C.track_rel(cost_path)
        payload["cost_measurement_sha256"] = C.sha256_file(cost_path)
        payload["epoch_seconds"] = {
            f"{row['dataset']}|{row['model']}": float(row["mean_epoch_seconds"])
            for row in C.read_json(cost_path)["rows"]
        }
    for dataset in datasets:
        view = D.molecule_view(dataset)
        spec, norm = view.spec, view.train_norm
        payload["cache"][dataset] = {
            "n_train": int(view.train_cache["n_molecules"]),
            "n_valid": int(view.valid_cache["n_molecules"]),
            "train_cache_mb": _cache_mb(view.train_cache),
            "valid_cache_mb": _cache_mb(view.valid_cache),
            "cache_build": _cache_build_seconds(dataset),
            "shared_by_variants": True,
        }
        for model_name in C.MODELS:
            payload["rows"].append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    **_params(spec, norm, model_name),
                    "inference": _inference(spec, norm, model_name, view.valid_cache, device),
                    "long_tail": _long_tail(spec, norm, model_name, view.valid_cache, device),
                    "epoch_seconds": payload.get("epoch_seconds", {}).get(f"{dataset}|{model_name}"),
                    "scope": "MEASURED_THIS_ROUND",
                }
            )
        for model_name in C.MODELS:
            info = _cumulative_gpu_before_best(C.mol_run_dir(dataset, model_name, 0))
            if info is not None:
                payload["cumulative_gpu_before_best"][f"{dataset}|{model_name}|s0"] = info
    C.write_json(C.ANALYSIS / "efficiency_measurements.json", payload)
    C.write_csv(C.ANALYSIS / "efficiency_table.csv", _table_rows(payload))
    C.log(f"efficiency measured for {len(payload['rows'])} (dataset, variant) rows")
    return payload


def _table_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in payload["rows"]:
        inf = row["inference"]
        cache = payload["cache"].get(row["dataset"], {})
        out.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "params_total": row["params_total"],
                "params_trainable": row["params_trainable"],
                "param_mb_fp32": round(row["param_mb_fp32"], 6),
                "epoch_seconds": None if row["epoch_seconds"] is None else round(row["epoch_seconds"], 6),
                "graphs_per_second": round(inf["graphs_per_second"], 4),
                "ms_per_graph": round(inf["ms_per_graph"], 6),
                "cuda_peak_allocated_mb": None
                if inf["cuda_peak_allocated_mb"] is None
                else round(inf["cuda_peak_allocated_mb"], 3),
                "train_cache_mb": round(cache.get("train_cache_mb", float("nan")), 3),
                "cache_build_seconds": round(cache.get("cache_build", {}).get("seconds", float("nan")), 4),
                "long_tail_max_over_min": None
                if row["long_tail"].get("max_over_min_cost_ratio") is None
                else round(row["long_tail"]["max_over_min_cost_ratio"], 3),
                "inference_split": inf["split"],
                "scope": row["scope"],
            }
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(C.MOLECULE_ORDER))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    run(args.datasets, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
