"""E2E-DictEnv-Joint709-Absolute-v1 (core).

Round ``e2e_dictenv_joint709_absolute_v1`` (study ``zinc-context-gap``).
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_joint709_absolute_v1_preregistration.md``.

This round tests **one** candidate, the frozen joint 709-D environment
dictionary already *defined* (but never trained) by route 2 of
``e2e_dictenv_rolecorr_increment_v2``:

* coordinate ``z = [ c~ (1) ; IHT_10(colnorm(D_joint48), x) (48) ]`` — width 49;
* joint input ``x`` (709) = structural residual (65) + ``Sem108`` (108) +
  RoleCorr correspondence (536), balanced by the frozen train-only scaler;
* ``D_joint`` is a frozen K-SVD K48/s12 dictionary fit on all official-train
  rows (detached, 10 epochs, seed 20260924);
* everything else is the frozen Sem108 parent (C6 mask, bindings, fusion,
  relation features, pooling, backend, reader, optimiser, 320-epoch budget,
  Top-5 soup).

The round screens the **absolute** performance of this candidate.  There is no
control arm in this round, so no increment / sparse-vs-dense statement may be
derived from it.  The official ZINC **test** split is never instantiated.

Everything in this module is a pure definition (no file I/O, no training), so
it is directly unit-testable without data or checkpoints.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_increment_v2 as inc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb

PROTOCOL_VERSION = "e2e_dictenv_joint709_absolute_v1"
ROUND = "e2e_dictenv_joint709_absolute_v1"

#: the single candidate of this round
CANDIDATE = inc.ARM_JOINT_SPARSE  # "JOINT-SPARSE"

# frozen geometry (re-exported for the runner/tests)
PHI_DIM = int(inc.PHI_DIM)                 # 65
SEM_DIM = int(inc.SEM_DIM)                 # 108
CORR_DIM = int(inc.CORR_DIM)               # 536
JOINT_DIM = int(inc.JOINT_DIM)             # 709
JOINT_ATOMS = int(inc.JOINT_ATOMS)         # 48
JOINT_SPARSITY = int(inc.JOINT_SPARSITY)   # 12
IHT_STEPS = int(inc.IHT_STEPS)             # 10 (iteration count, NOT s)
COMMON_DIM = int(inc.COMMON_DIM)           # 1
COORD_DIM = int(inc.COORD_DIM)             # 49
JOINT_SLICE = inc.JOINT_SLICE              # 1..49
JOINT_BLOCK_SLICES = dict(inc.JOINT_BLOCK_SLICES)
DICT_EPOCHS = int(inc.DICT_EPOCHS)         # 10
DICT_SEED = int(inc.DICT_SEED)             # 20260924

#: absolute-performance resource-allocation bands (soup valid MAE, lower better)
ABS_STRONG = 0.115
ABS_PROMISING = 0.120
ABS_BORDERLINE = 0.1233

BAND_STRONG = "strong_prospect"
BAND_PROMISING = "promising"
BAND_BORDERLINE = "borderline"
BAND_STOP = "stop"

BAND_RECOMMENDATION = {
    BAND_STRONG: "strong prospect: recommend entering a separate confirmation round",
    BAND_PROMISING: "promising: worth one further validation round (not automatic)",
    BAND_BORDERLINE: "borderline: no further experiments for now",
    BAND_STOP: "stop: the candidate did not enter a better absolute interval",
}

#: soup row-shuffle probe seeds for the usage diagnostic (two fixed seeds)
BLOCK_SHUFFLE_SEEDS: tuple[int, ...] = (11, 22)


def absolute_band(soup_valid_mae: float) -> dict[str, Any]:
    """Map one soup valid MAE onto the frozen absolute-performance bands."""
    mae = float(soup_valid_mae)
    if not np.isfinite(mae):
        raise ValueError("soup valid MAE must be finite")
    if mae <= ABS_STRONG:
        band = BAND_STRONG
    elif mae <= ABS_PROMISING:
        band = BAND_PROMISING
    elif mae <= ABS_BORDERLINE:
        band = BAND_BORDERLINE
    else:
        band = BAND_STOP
    return {
        "band": band,
        "soup_valid_mae": mae,
        "thresholds": {
            "strong_max": ABS_STRONG,
            "promising_max": ABS_PROMISING,
            "borderline_max": ABS_BORDERLINE,
        },
        "recommendation": BAND_RECOMMENDATION[band],
        "statistical_significance_gate": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# true joint reconstruction diagnostic (tied-IHT, the actual model path)
# ---------------------------------------------------------------------------


def _relative_error(X: np.ndarray, rec: np.ndarray, eps: float = 1e-12) -> float:
    X = np.asarray(X, dtype=np.float64)
    rec = np.asarray(rec, dtype=np.float64)
    num = float(np.mean(np.sum((X - rec) ** 2, axis=1)))
    den = float(np.mean(np.sum(X * X, axis=1)))
    return num / (den + eps)


def tied_iht_codes_chunked(
    dictionary: np.ndarray,
    X: np.ndarray,
    *,
    s: int = JOINT_SPARSITY,
    steps: int = IHT_STEPS,
    chunk: int = 65536,
) -> np.ndarray:
    """Chunked ``v0.tied_iht_codes`` over ``colnorm(dictionary)`` in float32.

    Rows are independent, so chunking is bit-identical to a single call except
    for the (deterministic) power-iteration on the dictionary, which is applied
    to the same normalized tensor.
    """
    D = np.asarray(dictionary, dtype=np.float32)
    Dbar = v0.normalized_dictionary(torch.as_tensor(D, dtype=torch.float32))
    out: list[np.ndarray] = []
    X = np.asarray(X, dtype=np.float32)
    for start in range(0, int(X.shape[0]), int(chunk)):
        part = torch.as_tensor(
            np.ascontiguousarray(X[start : start + int(chunk)]), dtype=torch.float32
        )
        with torch.no_grad():
            codes = v0.tied_iht_codes(Dbar, part, s=int(s), steps=int(steps))
        out.append(codes.numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, D.shape[1]), np.float32)


def joint_reconstruction_diagnostic(
    dictionary: np.ndarray,
    X: np.ndarray,
    *,
    s: int = JOINT_SPARSITY,
    steps: int = IHT_STEPS,
    omp_rows: int | None = None,
    chunk: int = 65536,
) -> dict[str, Any]:
    """Real ``X vs D_joint alpha`` error of the frozen dictionary.

    ``alpha`` is produced by exactly the tied-IHT(top-``s``, ``steps``
    iterations) that the model uses, so the diagnostic measures the actual
    coding path (the old route-2 ``reconstruct`` zero placeholder must never be
    used as a reconstruction quality number).  An OMP(top-``s``) reference on a
    deterministic row prefix is reported alongside; the reference is exhaustive
    rather than tied, hence an upper bound on achievable quality.
    """
    D = np.asarray(dictionary, dtype=np.float32)
    X = np.asarray(X, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != int(X.shape[1]):
        raise RuntimeError(f"dictionary {D.shape} incompatible with input {X.shape}")
    Dbar64 = v0.normalized_dictionary(torch.as_tensor(D, dtype=torch.float32)).numpy().astype(
        np.float64
    )
    codes = tied_iht_codes_chunked(D, X, s=s, steps=steps, chunk=chunk)
    reconstruction = codes.astype(np.float64) @ Dbar64.T
    l0 = np.abs(codes) > 0.0
    payload: dict[str, Any] = {
        "n_rows": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "s": int(s),
        "iht_steps": int(steps),
        "relative_error_tied_iht": _relative_error(X, reconstruction),
        "squared_error_per_row": float(np.mean(np.sum((X - reconstruction) ** 2, axis=1))),
        "exact_l0_mean": float(l0.sum(axis=1).mean()),
        "exact_l0_max": int(l0.sum(axis=1).max()) if l0.shape[0] else 0,
        "active_atoms": int((l0.any(axis=0)).sum()),
        "official_test_loaded": False,
    }
    blocks: dict[str, Any] = {}
    for name, span in JOINT_BLOCK_SLICES.items():
        blocks[name] = {
            "relative_error_tied_iht": _relative_error(X[:, span], reconstruction[:, span]),
            "mean_row_squared_norm": float(np.mean(np.sum(X[:, span] ** 2, axis=1))),
        }
    payload["blocks"] = blocks
    if omp_rows is not None and int(omp_rows) > 0:
        n_omp = min(int(omp_rows), int(X.shape[0]))
        omp_codes = sdb.omp_codes(D, np.ascontiguousarray(X[:n_omp]), s=int(s))
        omp_rec = omp_codes.astype(np.float64) @ Dbar64.T
        payload["omp_reference"] = {
            "n_rows": int(n_omp),
            "relative_error_omp": _relative_error(X[:n_omp], omp_rec),
            "exact_l0_mean": float((np.abs(omp_codes) > 0.0).sum(axis=1).mean()),
        }
    return payload


def band_metrics_rows(prefix: str, band: Mapping[str, Any]) -> dict[str, Any]:
    """Flat metric rows for the control-plane metrics payload."""
    return {
        f"{prefix}_band": str(band["band"]),
        f"{prefix}_soup_mae": float(band["soup_valid_mae"]),
        f"{prefix}_recommendation": str(band["recommendation"]),
    }


__all__ = [
    "PROTOCOL_VERSION",
    "ROUND",
    "CANDIDATE",
    "PHI_DIM",
    "SEM_DIM",
    "CORR_DIM",
    "JOINT_DIM",
    "JOINT_ATOMS",
    "JOINT_SPARSITY",
    "IHT_STEPS",
    "COMMON_DIM",
    "COORD_DIM",
    "JOINT_SLICE",
    "JOINT_BLOCK_SLICES",
    "DICT_EPOCHS",
    "DICT_SEED",
    "ABS_STRONG",
    "ABS_PROMISING",
    "ABS_BORDERLINE",
    "BAND_STRONG",
    "BAND_PROMISING",
    "BAND_BORDERLINE",
    "BAND_STOP",
    "BAND_RECOMMENDATION",
    "BLOCK_SHUFFLE_SEEDS",
    "absolute_band",
    "tied_iht_codes_chunked",
    "joint_reconstruction_diagnostic",
    "band_metrics_rows",
]
