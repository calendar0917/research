"""Mechanism probe for the host-to-device cost measured in ``profile.json``.

``profile.json`` shows the step is host-latency bound and that the single
largest phase is ``to_device`` (~26 ms of host time per step for ~2 MB), while
the profiler attributes only 0.16 ms/step to ``Memcpy HtoD``.  This probe
isolates the mechanism on the *same* batch, the *same* tensors and the same
device, by timing four transfer strategies and their host/stream components:

``pageable_per_tensor``   what ``pc_model.to_device`` does today
``pinned_per_tensor``     ``.pin_memory()`` then ``.to(device, non_blocking=True)``
``pinned_bulk``           one pinned staging buffer, one async copy, views
``pageable_bulk``         one pageable buffer copy (lower bound for pinning value)

It also measures the cost of a bare ``cudaStreamSynchronize`` and of
``pin_memory`` itself so the numbers can be decomposed rather than guessed.

No model, no loss, no optimizer: this probe answers "why is the copy slow",
nothing else.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pf_common as PF

import torch  # noqa: E402

import pf_collate as PC_C  # noqa: E402


def _stats(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64) * 1e3
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "min_ms": float(a.min()),
    }


def _bytes_of(batch) -> int:
    return int(sum(t.numel() * t.element_size() for t in batch.values() if torch.is_tensor(t)))


def probe(cache, device: torch.device, batches, n: int = 60) -> dict[str, Any]:
    out: dict[str, Any] = {"n_reps": int(n), "device": str(device)}
    keep_device: list[dict[str, torch.Tensor]] = []
    keep_pinned: list[dict[str, Any]] = []
    timings: dict[str, list[float]] = {
        "pageable_per_tensor": [],
        "pinned_per_tensor": [],
        "pageable_bulk": [],
        "pinned_bulk": [],
        "sync_only": [],
        "pin_memory_only": [],
        "bulk_single_pageable": [],
    }
    sizes = []
    for i, rows in enumerate(batches[:n]):
        batch = PC_C.collate_reference(cache, rows.tolist())
        sizes.append(_bytes_of(batch))
        # warm the destination allocations once
        if i == 0:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    v.to(device)

        torch.cuda.synchronize()
        t = time.perf_counter()
        got = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        torch.cuda.synchronize()
        timings["pageable_per_tensor"].append(time.perf_counter() - t)
        del got

        torch.cuda.synchronize()
        t = time.perf_counter()
        torch.cuda.synchronize()
        timings["sync_only"].append(time.perf_counter() - t)

        t = time.perf_counter()
        pinned = {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in batch.items()}
        timings["pin_memory_only"].append(time.perf_counter() - t)

        torch.cuda.synchronize()
        t = time.perf_counter()
        got_p = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in pinned.items()}
        torch.cuda.synchronize()
        timings["pinned_per_tensor"].append(time.perf_counter() - t)
        # keep one batch of pinned memory alive (the safe-belongs-to-caller pattern)
        keep_pinned.append(pinned)
        keep_pinned = keep_pinned[-1:]
        del got_p

        # bulk: concatenate every tensor's bytes into one flat int64 view, copy once
        order = [k for k in batch if torch.is_tensor(batch[k])]
        # bit-exact flat staging area: one uint8 view per field
        flat_src = torch.cat([batch[k].reshape(-1).view(torch.uint8) for k in order])
        torch.cuda.synchronize()
        t = time.perf_counter()
        _ = flat_src.to(device)  # pageable bulk
        torch.cuda.synchronize()
        timings["pageable_bulk"].append(time.perf_counter() - t)

        flat_pin = flat_src.pin_memory()
        torch.cuda.synchronize()
        t = time.perf_counter()
        flat_dev = flat_pin.to(device, non_blocking=True)
        torch.cuda.synchronize()
        timings["pinned_bulk"].append(time.perf_counter() - t)
        del flat_dev, flat_pin, flat_src

        # single 2 MB pageable tensor (isolates size vs count)
        big = torch.empty(sizes[-1] // 4, dtype=torch.float32)
        torch.cuda.synchronize()
        t = time.perf_counter()
        _ = big.to(device)
        torch.cuda.synchronize()
        timings["bulk_single_pageable"].append(time.perf_counter() - t)
        del big

    out["bytes_per_batch"] = {
        "p50": int(np.percentile(sizes, 50)),
        "mean": float(np.mean(sizes)),
        "n_tensors": int(len([k for k in batch if torch.is_tensor(batch[k])])),
    }
    out["timings"] = {k: _stats(v) for k, v in timings.items()}

    # bandwidth view of the same numbers
    mean_bytes = float(np.mean(sizes))
    out["effective_bandwidth_gb_s"] = {
        k: float(mean_bytes / (v["mean_ms"] * 1e-3) / 1e9)
        for k, v in out["timings"].items()
        if k in ("pageable_per_tensor", "pinned_per_tensor", "pageable_bulk", "pinned_bulk", "bulk_single_pageable")
        and v["mean_ms"] > 0
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reps", type=int, default=60)
    parser.add_argument("--out", default=str(PF.ANALYSIS / "h2d_probe.json"))
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this probe only makes sense on a CUDA device")
    inputs = PF.load_inputs()
    batches = PF.fixed_batches(inputs.train_rows, args.reps)
    result = {
        "schema": "lcrc-pcba-h2d-probe-v1",
        "round_id": PF.ROUND_ID,
        "created_utc": PF.utcnow(),
        "git_commit": PF.git_commit(),
        "environment": PF.environment(),
        "protocol": {"reps": int(args.reps), "batch_size": PF.BATCH_SIZE, "seed": PF.SEED},
        "measurements": probe(inputs.cache, device, batches, args.reps),
        "notes": [
            "every variant is followed by torch.cuda.synchronize() so the number is the "
            "wall time to have the data actually resident on the device",
            "pin_memory_only isolates the host cost of staging into pinned memory",
            "pinned_bulk/pageable_bulk concatenate all fields into one flat buffer first, "
            "so they include the concatenation cost",
        ],
    }
    PF.write_json(args.out, result)
    PF.log(f"wrote {PF.track_rel(args.out)}")
    for k, v in result["measurements"]["timings"].items():
        PF.log(f"{k:22s} p50={v['p50_ms']:8.3f} ms  mean={v['mean_ms']:8.3f} ms")
    PF.log(f"bytes/batch p50={result['measurements']['bytes_per_batch']['p50']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
