"""Frozen ordinary-molecule batch order (R3 shuffle generator, seed 917000).

Only the ordinary batch sequence exists in this round: there is no auxiliary
pair stream, no pair loss and no token shuffle.  The sequence must be *identical*
to the ordinary stream of the ordinary-C1 baseline so that the only difference
between arms is the correction operator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import common as C

PERM_NPZ = C.CACHE / "batch_order.npz"


def build_batch_perms(n_rows: int, epochs: int, seed: int = C.SHUFFLE_SEED) -> np.ndarray:
    gen = torch.Generator().manual_seed(int(seed))
    out = np.empty((int(epochs), int(n_rows)), dtype=np.int64)
    for epoch in range(int(epochs)):
        out[epoch] = torch.randperm(int(n_rows), generator=gen).numpy().astype(np.int64)
    return out


def build_schedule(*, write: bool = True) -> dict[str, Any]:
    perms = build_batch_perms(int(C.RECIPE["global_batch_size"]), int(C.RECIPE["epochs"]))
    manifest = {
        "schema": "lcrc-lipo-s3-locality-batch-order-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "label_blind": True,
        "shuffle_generator_seed": int(C.SHUFFLE_SEED),
        "epochs": int(C.RECIPE["epochs"]),
        "n_rows": int(C.RECIPE["global_batch_size"]),
        "steps_per_epoch": int(C.RECIPE["steps_per_epoch"]),
        "batch_size": int(C.RECIPE["batch_size"]),
        "drop_last": False,
        "shape": list(perms.shape),
        "sha256": C.array_sha256(perms),
        "epoch1_first10": [int(x) for x in perms[0, :10]],
        "baseline_manifest_match": None,
    }
    baseline_manifest = C.BASELINE_ROUND / "analysis" / "schedule_manifest.json"
    if baseline_manifest.exists():
        base = C.read_json(baseline_manifest)
        expected = str(base["batch_order"]["sha256"])
        manifest["baseline_batch_order_sha256"] = expected
        manifest["baseline_manifest_match"] = bool(manifest["sha256"] == expected)
        if not manifest["baseline_manifest_match"]:
            raise ValueError(
                f"batch order sha256 {manifest['sha256']} != baseline {expected}; the ordinary stream changed"
            )
    if write:
        C.CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PERM_NPZ, batch_perms=perms)
        C.write_json(C.ANALYSIS / "batch_order_manifest.json", manifest)
        C.log(f"[batch-order] {perms.shape} sha256={manifest['sha256'][:16]} baseline_match={manifest['baseline_manifest_match']}")
    return {"batch_perms": perms, "manifest": manifest}


def load_schedule() -> dict[str, Any]:
    path = PERM_NPZ
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run code/prepare.py --cache-only first")
    with np.load(path) as z:
        perms = z["batch_perms"].copy()
    manifest = C.read_json(C.ANALYSIS / "batch_order_manifest.json")
    got = C.array_sha256(perms)
    if got != manifest["sha256"]:
        raise ValueError(f"{path}: batch order sha256 {got} != frozen {manifest['sha256']}")
    return {"batch_perms": perms, "manifest": manifest}


def ensure_schedule() -> bool:
    """Rebuild the ignored cache only when absent, then verify the frozen hash."""
    if PERM_NPZ.exists():
        load_schedule()
        return False
    built = build_schedule(write=False)
    manifest = built["manifest"]
    committed = C.ANALYSIS / "batch_order_manifest.json"
    if committed.exists():
        expected = C.read_json(committed)
        bad = {
            k: (manifest[k], expected[k])
            for k in ("sha256", "epoch1_first10", "n_rows", "epochs")
            if manifest[k] != expected[k]
        }
        if bad:
            raise ValueError(f"rebuilt batch order differs from the committed manifest: {bad}")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PERM_NPZ, batch_perms=built["batch_perms"])
    C.write_json(committed, manifest)
    C.log("[batch-order] rebuilt ignored cache from the committed manifest")
    return True


__all__ = ["PERM_NPZ", "build_batch_perms", "build_schedule", "load_schedule", "ensure_schedule"]
