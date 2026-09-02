"""Encode official-test node tokens with an already frozen train-only dictionary.

This utility is intentionally separate from dictionary learning.  It may only be
used after a model/configuration is frozen: the source dictionary and all
train/valid token rows are copied byte-for-byte, and only requested official-test
rows are sparse-coded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_molhiv import load_molhiv
from .ksvd import _omp
from .molhiv_node_tokens import centered_ego_vector


def _csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _sha256(arr: np.ndarray) -> str:
    view = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha256(view).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-cache", required=True)
    ap.add_argument("--families", default="ksvd")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--frozen-config-id",
        required=True,
        help="human-readable identifier proving this is a post-freeze encoding step",
    )
    args = ap.parse_args()

    t0 = time.time()
    source = Path(args.source_cache)
    source_meta_path = source.with_suffix(".json")
    if not source_meta_path.exists():
        raise FileNotFoundError(f"missing source metadata: {source_meta_path}")
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    cfg = source_meta.get("config", {})
    max_graphs_raw = int(cfg.get("max_graphs", 0))
    max_graphs = None if max_graphs_raw <= 0 else max_graphs_raw
    radius = int(cfg["radius"])
    max_nodes = int(cfg["max_nodes"])
    sparsity = int(cfg["sparsity"])
    families = _csv(args.families)
    if not families:
        raise ValueError("at least one family is required")

    bundle = load_molhiv(
        max_graphs=max_graphs,
        seed=int(cfg.get("data_seed", 0)),
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV chemical features were not loaded")
    te = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(source) as cache:
        archive: dict[str, np.ndarray] = {
            key: np.asarray(cache[key]).copy() for key in cache.files
        }

    offsets = np.asarray(archive["offsets"], dtype=np.int64)
    original_indices = np.asarray(archive["original_indices"], dtype=np.int64)
    expected_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    if not np.array_equal(original_indices, expected_indices):
        raise ValueError("source cache original_indices do not match loaded data")
    if not np.array_equal(np.asarray(archive["test_indices"], dtype=np.int64), te):
        raise ValueError("source cache official-test split mismatch")
    if offsets.shape != (len(bundle.graphs) + 1,):
        raise ValueError("source cache offsets shape mismatch")

    before_hashes: dict[str, str] = {}
    for family in families:
        token_key = f"tokens_{family}"
        dict_key = f"dictionary_{family}"
        if token_key not in archive or dict_key not in archive:
            raise KeyError(f"source cache lacks {token_key}/{dict_key}")
        tokens = np.asarray(archive[token_key], dtype=np.float32)
        if tokens.shape[0] != int(offsets[-1]):
            raise ValueError(f"{token_key} node row mismatch")
        before_hashes[token_key] = _sha256(tokens)
        for i in te:
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            if np.any(tokens[lo:hi] != 0):
                raise ValueError(f"source {token_key} already contains official-test codes")

    encoded_nodes = 0
    for count, raw_i in enumerate(te, 1):
        i = int(raw_i)
        g = bundle.graphs[i]
        for u in g.nodes:
            vec = centered_ego_vector(
                g,
                int(u),
                bundle.node_feats[i],
                bundle.edge_feats[i],
                radius=radius,
                max_nodes=max_nodes,
            )
            y = np.asarray(vec, dtype=np.float64)
            y /= max(float(np.linalg.norm(y)), 1e-12)
            row = int(offsets[i] + u)
            for family in families:
                D = np.asarray(archive[f"dictionary_{family}"], dtype=np.float64)
                archive[f"tokens_{family}"][row] = _omp(D, y, sparsity).astype(np.float32)
            encoded_nodes += 1
        if count % 500 == 0 or count == len(te):
            print(
                f"encoded official-test graphs {count}/{len(te)}; nodes={encoded_nodes}",
                flush=True,
            )

    for family in families:
        tokens = np.asarray(archive[f"tokens_{family}"], dtype=np.float32)
        nonzero_graphs = 0
        for i in te:
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            nonzero_graphs += int(np.any(tokens[lo:hi] != 0))
        if nonzero_graphs != len(te):
            raise AssertionError(
                f"{family}: only {nonzero_graphs}/{len(te)} test graphs received codes"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **archive)
    result: dict[str, Any] = {
        "protocol_id": "molhiv-localized-node-token-frozen-test-encoding-v1",
        "frozen_config_id": args.frozen_config_id,
        "source_cache": str(source),
        "output": str(output),
        "families_encoded": families,
        "dictionary_learning_policy": "no dictionary was learned or updated; source train-only dictionary copied exactly",
        "row_policy": "train/valid rows copied; only official-test rows were newly sparse-coded",
        "source_config": cfg,
        "n_official_test_graphs": int(len(te)),
        "n_official_test_nodes_encoded": int(encoded_nodes),
        "dictionary_hashes": {
            family: _sha256(np.asarray(archive[f"dictionary_{family}"]))
            for family in families
        },
        "source_token_hashes_before_test_encoding": before_hashes,
        "elapsed_sec": time.time() - t0,
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"wrote {output} and {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
