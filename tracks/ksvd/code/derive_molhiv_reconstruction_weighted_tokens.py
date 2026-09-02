"""Weight fixed KSVD codes by their OMP explained-energy confidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    t0 = time.time()

    source = np.load(args.token_cache)
    D = np.asarray(source["dictionary_ksvd"], dtype=np.float64)
    X = np.asarray(source["tokens_ksvd"], dtype=np.float64)
    reconstruction = X @ D.T
    explained_energy = np.einsum("ij,ij->i", reconstruction, reconstruction)
    confidence = np.clip(explained_energy, 0.0, 1.0)
    tokens = (X * confidence[:, None]).astype(np.float32)

    offsets = source["offsets"]
    for raw_i in source["test_indices"]:
        i = int(raw_i)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if np.any(tokens[lo:hi] != 0):
            raise AssertionError("weighted cache contains official-test codes")

    train_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in source["train_indices"]
    ])
    q = np.quantile(confidence[train_rows], [0, .01, .1, .25, .5, .75, .9, .99, 1])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        offsets=offsets,
        original_indices=source["original_indices"],
        train_indices=source["train_indices"],
        valid_indices=source["valid_indices"],
        test_indices=source["test_indices"],
        dictionary_ksvd_recon_weighted=D.astype(np.float32),
        tokens_ksvd_recon_weighted=tokens,
    )
    meta = {
        "protocol_id": "molhiv-ksvd-reconstruction-confidence-weighted-node-tokens-v1",
        "test_policy": "derived from a cache whose official-test rows are exactly zero",
        "config": vars(args),
        "formula": "x_weighted = clip(||D x||_2^2, 0, 1) * x for unit-normalized input patches",
        "train_confidence_quantiles": {
            str(k): float(v) for k, v in zip([0,.01,.1,.25,.5,.75,.9,.99,1], q)
        },
        "mean_l2_delta_on_train_nodes": float(
            np.linalg.norm(tokens[train_rows].astype(np.float64) - X[train_rows], axis=1).mean()
        ),
        "official_test_nonzero_values": 0,
        "elapsed_sec": time.time() - t0,
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
