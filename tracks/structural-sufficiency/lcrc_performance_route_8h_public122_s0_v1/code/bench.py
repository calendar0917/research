"""Practicality bench: inference latency and cost coordinates (PROTOCOL §7 cost).

Run once on an idle pool (it measures wall latency, so it must not share a GPU
with a training job).  Everything it reports is a *practical* coordinate: the two
models consume different inputs and recipes, so the numbers are not a controlled
comparison of architectures.
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

import numpy as np
import torch

import lcpr_common as C


def _timeit(fn, *, warmup: int = 10, repeats: int = 5, device=None) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    return {"median_seconds": statistics.median(times),
            "min_seconds": min(times), "max_seconds": max(times), "repeats": repeats}


def run(dataset: str = "Lipophilicity", device_name: str = "cuda") -> dict[str, Any]:
    import lcpr_data as D
    import lcpr_dmpnn as DMN
    import lcpr_model as M
    import lcpr_train as T

    dev = torch.device(device_name)
    spec = D.frozen_spec()
    norm = T._train_structure_norm(D.load_cache(dataset, "train"), spec)
    model = M.build_model(spec, norm, "c1").to(dev).eval()
    cache = D.load_cache(dataset, "test")
    n = int(cache["n_molecules"])
    batches = [D.to_device(D.collate(cache, list(range(i, min(i + 128, n)))), dev)
               for i in range(0, n, 128)]

    out: dict[str, Any] = {"dataset": dataset, "device": str(dev),
                           "n_test_rows": n, "batch_size": 128,
                           "parameters": {"c1": int(sum(p.numel() for p in model.parameters()))}}

    def _run_c1():
        with torch.no_grad():
            for b in batches:
                model(b)

    with torch.no_grad():
        out["c1_inference"] = _timeit(_run_c1, device=dev)

    # D-MPNN
    try:
        feat_path = DMN.featurize_path(dataset)
        if not feat_path.exists():
            DMN.build_features(dataset)
        feat = DMN.load_feat(feat_path)
        dmp = DMN.DMPNNReference(d_h=int(C.DMPNN_FAMILY["hidden"])).to(dev).eval()
        out["parameters"]["dmpnn"] = int(sum(p.numel() for p in dmp.parameters()))
        rows = list(range(n))
        dmp_batches = [DMN._batch(feat, rows[i: i + 1024], dev) for i in range(0, len(rows), 1024)]

        def _run_dmp():
            with torch.no_grad():
                for b in dmp_batches:
                    dmp(b)

        out["dmpnn_inference"] = _timeit(_run_dmp, device=dev)
        out["dmpnn_note"] = "batched at 1024 molecules, the padding-free setting used in training"
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        out["dmpnn_inference"] = {"error": f"{type(exc).__name__}: {exc}"}

    out["cache_bytes"] = {s: int(D.cache_path(dataset, s).stat().st_size)
                          for s in C.SPLITS if D.cache_path(dataset, s).exists()}
    out["cache_paths"] = {s: C.track_rel(D.cache_path(dataset, s)) for s in C.SPLITS}
    out["created_utc"] = C.utcnow()
    C.write_json(C.ANALYSIS / "bench.json", out)
    return out


if __name__ == "__main__":
    import sys

    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    print(json.dumps(run(device_name=dev), indent=2))
