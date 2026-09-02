"""Encode MolHIV atom-centered raw patch descriptors into a frozen metric.

This utility reconstructs the exact permutation-invariant radius-r descriptor
used to fit a node-token cache, applies that cache's train-only PCA/whitening
transform, and stores unit-normalized local vectors.  Only official-train
molecules are encoded by default; terminal evaluation may explicitly encode frozen official-valid/test rows after the transform and dictionary are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv
from code.molhiv_node_tokens import centered_ego_vector, graph_node_offsets


def _sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument(
        "--encode-splits", choices=("train", "train_valid", "train_test", "all"), default="train",
        help=(
            "splits to transform with the already frozen train-fit metric; "
            "train_valid/train_test are split-isolated frozen evaluations; all is audit-only"
        ),
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.radius < 0 or args.max_nodes <= 0:
        raise ValueError("invalid patch configuration")
    start = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise AssertionError("molecular features were not loaded")
    offsets = graph_node_offsets(bundle.graphs)
    original_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)

    token_sidecar = Path(args.token_cache).with_suffix(".json")
    if not token_sidecar.exists():
        raise FileNotFoundError(f"missing token-cache audit sidecar: {token_sidecar}")
    token_meta = json.loads(token_sidecar.read_text(encoding="utf-8"))
    official_train_sha = _sha256(official_train)
    if token_meta.get("dictionary_fit_indices_sha256") != official_train_sha:
        raise ValueError("source metric/dictionary was not fit on exact official train")
    if bool(token_meta.get("encoded_official_valid", True)):
        raise ValueError("source token cache unexpectedly encoded official valid")

    with np.load(args.token_cache, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    for key in (
        "offsets", "original_indices", "train_indices", "valid_indices",
        "test_indices", "patch_transform_mean", "patch_transform_basis",
        "patch_transform_scale", "dictionary_ksvd",
    ):
        if key not in arrays:
            raise ValueError(f"token cache lacks required array {key}")
    checks = (
        (arrays["offsets"], offsets, "offsets"),
        (arrays["original_indices"], original_indices, "original_indices"),
        (arrays["train_indices"], official_train, "train_indices"),
        (arrays["valid_indices"], official_valid, "valid_indices"),
        (arrays["test_indices"], official_test, "test_indices"),
    )
    for actual, expected, name in checks:
        if not np.array_equal(np.asarray(actual, dtype=np.int64), expected):
            raise ValueError(f"misaligned token-cache {name}")

    mean = np.asarray(arrays["patch_transform_mean"], dtype=np.float64)
    basis = np.asarray(arrays["patch_transform_basis"], dtype=np.float64)
    scale = np.asarray(arrays["patch_transform_scale"], dtype=np.float64)
    if basis.shape[0] != mean.size or basis.shape[1] != scale.size:
        raise ValueError("invalid PCA transform shapes")
    dictionary = np.asarray(arrays["dictionary_ksvd"], dtype=np.float64)
    if dictionary.shape[0] != scale.size:
        raise ValueError("dictionary and transformed latent dimension mismatch")

    if args.encode_splits == "train":
        encode_indices = official_train
    elif args.encode_splits == "train_valid":
        encode_indices = np.concatenate([official_train, official_valid])
    elif args.encode_splits == "train_test":
        encode_indices = np.concatenate([official_train, official_test])
    else:
        encode_indices = np.concatenate([official_train, official_valid, official_test])

    latents = np.zeros((int(offsets[-1]), scale.size), dtype=np.float32)
    zero_norm = 0
    encoded = 0
    for count, raw_i in enumerate(encode_indices, 1):
        i = int(raw_i)
        graph = bundle.graphs[i]
        for raw_u in graph.nodes:
            u = int(raw_u)
            y = np.asarray(
                centered_ego_vector(
                    graph,
                    u,
                    bundle.node_feats[i],
                    bundle.edge_feats[i],
                    radius=args.radius,
                    max_nodes=args.max_nodes,
                ),
                dtype=np.float64,
            )
            y /= max(float(np.linalg.norm(y)), 1e-12)
            if y.size != mean.size:
                raise ValueError(
                    f"descriptor dimension {y.size} does not match transform {mean.size}"
                )
            z = (basis.T @ (y - mean)) * scale
            norm = float(np.linalg.norm(z))
            if norm <= 1e-12:
                zero_norm += 1
            else:
                z /= norm
            latents[int(offsets[i]) + u] = z.astype(np.float32)
            encoded += 1
        if count % 500 == 0 or count == len(encode_indices):
            print(
                f"encoded frozen-metric graphs {count}/{len(encode_indices)}; "
                f"nodes={encoded}",
                flush=True,
            )

    encoded_set = set(np.asarray(encode_indices, dtype=np.int64).tolist())
    omitted_indices = np.asarray(
        [i for i in range(len(bundle.graphs)) if i not in encoded_set], dtype=np.int64
    )
    if len(omitted_indices):
        omitted_rows = np.concatenate([
            np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
            for i in omitted_indices
        ])
        if np.any(latents[omitted_rows] != 0):
            raise AssertionError("omitted split patch latents are nonzero")
    train_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in official_train
    ])
    norms = np.linalg.norm(latents[train_rows], axis=1)
    if np.mean(norms > 0.99) < 0.999:
        raise AssertionError("too many missing/unnormalized official-train latents")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        offsets=offsets,
        original_indices=original_indices,
        train_indices=official_train,
        valid_indices=official_valid,
        test_indices=official_test,
        fit_indices=official_train,
        latents=latents,
    )
    report: dict[str, Any] = {
        "protocol_id": "molhiv-frozen-raw-patch-metric-latents-v1",
        "date": "2026-07-28",
        "representation": (
            "unit-normalized permutation-invariant radius-2 descriptor, then "
            "fold-fit PCA/whitening projection and unit normalization"
        ),
        "supervised_gnn_layers": 0,
        "unsupervised_gnn_layers": 0,
        "official_valid_encoded": bool(args.encode_splits in ("train_valid", "all")),
        "official_test_encoded": bool(args.encode_splits in ("train_test", "all")),
        "fit_policy": "PCA/whitening and KSVD source cache fitted on official-train only",
        "fit_indices_sha256": official_train_sha,
        "source_dictionary_fit_indices_sha256": token_meta["dictionary_fit_indices_sha256"],
        "config": vars(args),
        "n_graphs": int(len(bundle.graphs)),
        "n_official_train_graphs": int(len(official_train)),
        "n_encoded_nodes": int(encoded),
        "latent_dim": int(latents.shape[1]),
        "zero_norm_nodes": int(zero_norm),
        "mean_train_norm": float(norms.mean()),
        "offsets_sha256": _sha256(offsets),
        "latents_sha256": _sha256(latents),
        "elapsed_sec": time.time() - start,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
