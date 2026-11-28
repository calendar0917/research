"""The frozen historical input block ``H`` (2258 columns) — rebuilt, never retrained.

``H`` = historical ECFP4 (radius 2, 2048 bits, bit-OR) + 217 RDKit 2-D descriptors,
with the historical Train-only preprocessing (per-column median imputation, drop
columns whose Train std == 0).  The rebuild reuses ``run_phaseD_xgb.historical_block``
from the predecessor round verbatim, so the Phase-A/X0 identity question is decided
by the predecessor's own verified ``EXACT_MATCH`` check plus the hashes recorded
here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import sga_common as K

K.add_prev_code_path()

import run_phaseD_xgb as PD  # noqa: E402  (predecessor round)


def load_H() -> tuple[np.ndarray, dict[str, Any]]:
    """Dense ``H`` (41127, 2258) float32 plus its provenance checks."""
    import sys

    path = K.MSDS_CODE
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    X, checks = PD.historical_block()
    if not checks["preprocessing_reproduced"]:
        raise SystemExit("historical preprocessing could not be reproduced; refusing to continue")
    for name in ("features_sha256", "median_sha256", "keep_sha256", "n_used_columns"):
        checks.pop(name, None)
    payload = {
        **checks,
        "block": "H = historical ECFP4 binary 2048 + RDKit 2-D descriptors (Train-only median/std preprocessing)",
        "shape": list(X.shape),
        "nnz_fraction": float((X != 0).mean()),
        "n_ecfp_columns": 2048,
        "n_descriptor_columns": int(X.shape[1]) - 2048,
    }
    return X, payload


def column_roles(checks: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Column index roles inside ``H``: the ECFP bit block and the descriptor block.

    The historical preprocessing drops only zero-variance *descriptor* columns (all
    2048 ECFP bit columns have Train variance > 0), so ``H[:, :2048]`` are the bits
    and ``H[:, 2048:]`` are the surviving descriptors.
    """
    n_ecfp = 2048
    with np.load(K.HIST_FEATURES, allow_pickle=False) as z:
        descriptor_names = np.asarray([str(s) for s in z["descriptor_names"]])
        X = np.array(z["features"], dtype=np.float32, copy=False)
    import json

    manifest = json.loads(K.HIST_MANIFEST.read_text())
    rows = {k: np.asarray(v, dtype=np.int64) for k, v in manifest["datasets"]["ogbg-molhiv"]["rows"].items()
            if k in ("train", "valid", "test")}
    Xr = X[rows["train"]]
    std = np.nanstd(np.where(np.isnan(Xr), np.nan, Xr), axis=0)
    keep = np.flatnonzero(std > 0)
    n_bit_kept = int((keep < n_ecfp).sum())
    return {
        "n_ecfp_kept": np.int64(n_bit_kept),
        "n_descriptor_kept": np.int64(keep.size - n_bit_kept),
        "dropped_columns": np.asarray(np.setdiff1d(np.arange(X.shape[1]), keep), dtype=np.int64),
        "descriptor_names": descriptor_names[keep[keep >= n_ecfp] - n_ecfp],
        "all_bit_columns_kept": np.bool_(n_bit_kept == n_ecfp),
    }


def pack_H(X: np.ndarray) -> dict[str, np.ndarray]:
    """Pack ``H`` for the committed remote bundle: bits packed, descriptors float32."""
    n_ecfp = 2048
    bits = X[:, :n_ecfp].astype(np.uint8)
    if not np.array_equal(bits, (bits > 0).astype(np.uint8)):
        raise SystemExit("the first 2048 columns of H are not binary")
    return {
        "H_bits_packed": np.packbits(bits, axis=1),
        "H_desc": np.ascontiguousarray(X[:, n_ecfp:], dtype=np.float32),
        "H_n_bits": np.int64(n_ecfp),
        "H_shape": np.asarray(X.shape, dtype=np.int64),
    }


def unpack_H(bundle) -> np.ndarray:
    n_bits = int(bundle["H_n_bits"])
    shape = tuple(int(v) for v in bundle["H_shape"])
    bits = np.unpackbits(bundle["H_bits_packed"], axis=1)[:, :n_bits].astype(np.float32)
    desc = np.asarray(bundle["H_desc"], dtype=np.float32)
    out = np.empty(shape, dtype=np.float32)
    out[:, :n_bits] = bits
    out[:, n_bits:] = desc
    return out
