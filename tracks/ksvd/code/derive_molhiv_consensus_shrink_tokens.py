"""Derive conservative consensus-shrunk KSVD tokens without re-vectorizing graphs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _topk_rows(X: np.ndarray, k: int) -> np.ndarray:
    if not 0 < k <= X.shape[1]:
        raise ValueError("invalid top-k")
    if k == X.shape[1]:
        return X.copy()
    keep = np.argpartition(np.abs(X), -k, axis=1)[:, -k:]
    out = np.zeros_like(X)
    rows = np.arange(X.shape[0])[:, None]
    out[rows, keep] = X[rows, keep]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-cache", required=True)
    ap.add_argument("--consensus-cache", required=True)
    ap.add_argument("--anchor-weight", type=float, default=0.75)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not 0.0 <= args.anchor_weight <= 1.0:
        raise ValueError("anchor_weight must be in [0,1]")

    t0 = time.time()
    anchor = np.load(args.anchor_cache)
    consensus = np.load(args.consensus_cache)
    invariant_keys = ["offsets", "original_indices", "train_indices", "valid_indices", "test_indices"]
    for key in invariant_keys:
        if not np.array_equal(anchor[key], consensus[key]):
            raise ValueError(f"cache invariant mismatch: {key}")
    D0 = np.asarray(anchor["dictionary_ksvd"], dtype=np.float32)
    aligned0 = np.asarray(consensus["dictionary_ksvd_consensus_aligned_member0"], dtype=np.float32)
    if not np.array_equal(D0, aligned0):
        raise ValueError("consensus anchor is not the frozen global KSVD dictionary")

    X0 = np.asarray(anchor["tokens_ksvd"], dtype=np.float32)
    Xc = np.asarray(consensus["tokens_ksvd_consensus_codes"], dtype=np.float32)
    if X0.shape != Xc.shape:
        raise ValueError("token shape mismatch")
    mixed = args.anchor_weight * X0 + (1.0 - args.anchor_weight) * Xc
    tokens = _topk_rows(mixed, args.sparsity).astype(np.float32)

    offsets = anchor["offsets"]
    for raw_i in anchor["test_indices"]:
        i = int(raw_i)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if np.any(tokens[lo:hi] != 0):
            raise AssertionError("derived cache contains official-test codes")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        offsets=offsets,
        original_indices=anchor["original_indices"],
        train_indices=anchor["train_indices"],
        valid_indices=anchor["valid_indices"],
        test_indices=anchor["test_indices"],
        dictionary_ksvd_consensus_shrink=D0,
        tokens_ksvd_consensus_shrink=tokens,
    )
    changed_rows = np.any(tokens != X0, axis=1)
    encoded_rows = np.any(X0 != 0, axis=1)
    meta = {
        "protocol_id": "molhiv-conservative-consensus-shrunk-ksvd-node-tokens-v1",
        "test_policy": "derived only from caches whose official-test rows are exactly zero",
        "config": vars(args),
        "formula": "topT(anchor_weight * anchor_code + (1-anchor_weight) * aligned_consensus_code)",
        "token_dim": int(tokens.shape[1]),
        "changed_encoded_row_fraction": float(changed_rows[encoded_rows].mean()),
        "mean_l2_delta_on_encoded_rows": float(
            np.linalg.norm(tokens[encoded_rows] - X0[encoded_rows], axis=1).mean()
        ),
        "official_test_nonzero_values": 0,
        "elapsed_sec": time.time() - t0,
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
