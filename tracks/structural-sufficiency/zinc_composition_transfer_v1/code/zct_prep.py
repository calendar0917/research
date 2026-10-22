"""Deterministic CPU preparation for the composition-transfer round.

Builds everything that must be frozen *before* the first formal GPU run:

1. radius-1/2 keys (``zct_keys``);
2. every R/S source-query partition of every training epoch, for seeds 0 and 1
   (``zct_split``), including the intervention-strength summary;
3. the fixed 32 train-only probe episodes used for the kappa calibration and for
   the per-epoch process diagnostics;
4. the §7 supervision-support package (``zct_support``);
5. the strength gate artifact against the pre-declared thresholds, including the
   64-proposal (pre-fix) measurement.

No label, no valid/test access beyond the label-free frozen caches.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
for _p in (str(_HERE), str(_SSUF / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import zct_keys as K  # noqa: E402
import zct_split as SP  # noqa: E402

CACHE = _TASK / "cache"
ANALYSIS = _TASK / "analysis"


def build_probe_episodes(seed: int = 0, n_batches: int = 32) -> dict:
    """Fixed train-only episodes: rows plus the frozen R and S halves."""
    import torch

    import ad_data as D

    part = SP.load_partitions(seed)
    n_mol = int(D.load_cache("train")["n_molecules"])
    gen = torch.Generator().manual_seed(int(seed) + 101)
    order = torch.randperm(n_mol, generator=gen).numpy()
    if SP.order_hash(order) != part["summary"]["epoch_order_sha256"][0]:
        raise RuntimeError("epoch-1 order does not match the frozen partition file")
    rows_out, q_r, q_s = [], [], []
    for idx in range(n_batches):
        n = int(part["batch_rows"][idx])
        rows = np.asarray(order[idx * 128: idx * 128 + n], dtype=np.int64)
        rows_out.append(rows)
        steps = int(part["summary"]["steps_per_epoch"])
        q_r.append(SP.mask_for(part, 1, idx, "R", n, steps))
        q_s.append(SP.mask_for(part, 1, idx, "S", n, steps))
    payload = {
        "seed": int(seed),
        "epoch": 1,
        "n_batches": int(n_batches),
        "probe_rows": np.stack(rows_out),
        "probe_q": np.stack(q_r),
        "probe_q_structural": np.stack(q_s),
    }
    np.savez_compressed(CACHE / "probe_episodes.npz", **payload)
    meta = {
        "seed": int(seed), "epoch": 1, "n_batches": int(n_batches),
        "source": "first 32 batches of the frozen seed-0 epoch-1 partition file",
        "partitions_sha256": part["summary"]["npz_sha256"],
        "npz_sha256": C.sha256_file(CACHE / "probe_episodes.npz"),
        "note": "train-only; labels are read by the probe/diagnostics, never for model selection",
    }
    C.write_json(CACHE / "probe_episodes.json", meta)
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--skip-support", action="store_true")
    args = p.parse_args(argv)
    t0 = time.perf_counter()
    out = {"steps": []}

    def step(name):
        dt = time.perf_counter() - t0
        out["steps"].append({"step": name, "seconds": round(dt, 1)})
        C.log(f"[prep] {name} done at {dt:.1f}s")

    K.build_all()
    step("keys")

    for seed in [int(s) for s in args.seeds.split(",") if s]:
        summary = SP.build_all(seed, epochs=args.epochs, max_swaps=SP.MAX_SWAPS)
        out[f"partitions_s{seed}"] = summary
        step(f"partitions_s{seed}")

    out["probe"] = build_probe_episodes(0)
    step("probe_episodes")

    if not args.skip_support:
        import zct_support as SU

        out["support"] = SU.build_support()
        step("support")

    # split strength gate
    pre = SP.calibrate(0, n_batches=256, max_swaps=64)
    post = SP.calibrate(0, n_batches=256, max_swaps=SP.MAX_SWAPS)
    gate = {
        "thresholds": {
            "delta_d_sym_min": 0.02,
            "delta_d_elig_min": 0.10,
            "r1_unseen_increase_max": SP.R1_TOL,
        },
        "pre_fix_64_proposals": pre,
        "declared_256_proposals": post,
        "decision": "PASS" if all(post["passes"].values()) else "INSUFFICIENT_INTERVENTION",
        "note": ("the 64 -> 256 proposal increase is the single pre-authorised construction fix; "
                 "both numbers were measured on official-train batches only, before any valid read"),
    }
    C.write_json(ANALYSIS / "split_strength.json", gate)
    out["split_strength"] = gate["decision"]
    step("split_strength")

    out["total_seconds"] = round(time.perf_counter() - t0, 1)
    out["git_commit"] = C.git_commit()
    out["environment"] = C.environment()
    C.write_json(CACHE / "prep.json", out)
    C.log({"total_seconds": out["total_seconds"], "split_strength": gate["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
