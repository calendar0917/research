"""Convert sparse SSL-latent codes into dictionary-reconstructed context tokens.

For each family, x (D-dimensional OMP code) becomes D x in the shared SSL
latent coordinate system. This preserves dictionary geometry that atom-ID
transport discards while keeping KSVD as the explicit sparse bottleneck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with np.load(args.token_cache, allow_pickle=False) as source:
        arrays = {k: np.asarray(source[k]) for k in source.files}

    families = sorted(k.removeprefix("dictionary_") for k in arrays if k.startswith("dictionary_"))
    archive = {
        k: arrays[k]
        for k in ["offsets", "original_indices", "train_indices", "valid_indices", "test_indices"]
    }
    stats = {}
    offsets = arrays["offsets"]
    train_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in arrays["train_indices"]
    ])
    for family in families:
        D = arrays[f"dictionary_{family}"].astype(np.float64)
        X = arrays[f"tokens_{family}"].astype(np.float64)
        reconstruction = (X @ D.T).astype(np.float32)
        # Retain the original dictionary for provenance. Inject fusion only reads
        # tokens, so the code/atom dimensionality need not match reconstructed dim.
        archive[f"dictionary_{family}"] = arrays[f"dictionary_{family}"]
        archive[f"tokens_{family}"] = reconstruction
        norms = np.linalg.norm(reconstruction[train_rows].astype(np.float64), axis=1)
        stats[family] = {
            "token_dimension": int(reconstruction.shape[1]),
            "train_reconstruction_norm_mean": float(norms.mean()),
            "train_reconstruction_norm_quantiles": {
                str(q): float(v) for q, v in zip(
                    [0, .01, .1, .5, .9, .99, 1],
                    np.quantile(norms, [0, .01, .1, .5, .9, .99, 1]),
                )
            },
        }
        for i in np.concatenate([arrays["valid_indices"], arrays["test_indices"]]):
            lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
            if np.any(reconstruction[lo:hi] != 0):
                raise AssertionError(f"{family} reconstructed official-valid/test rows")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **archive)
    meta = {
        "protocol_id": "molhiv-fold-only-ssl-sparse-dictionary-reconstruction-tokens-v1",
        "source": args.token_cache,
        "formula": "tokens_family = sparse_codes_family @ dictionary_family.T",
        "labels_used": False,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "official_valid_test_tokens_encoded": False,
        "families": families,
        "stats": stats,
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
