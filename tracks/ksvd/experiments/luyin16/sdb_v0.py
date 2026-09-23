"""SDB-v0 — Sparse Structural Dictionary Binding: core primitives.

Pre-registration: ``tracks/ksvd/notes/sdb_v0_preregistration.md`` (frozen).

This module holds only the *definitions* of the round, so every algebraic
property demanded by the pre-registration can be unit-tested without loading
data or training anything:

* detached K-SVD / OMP sparse coding on the frozen pure-topology coordinate
  ``phi_v in R^65`` (reusing the correctness-tested TCCD-v0 primitives),
* the random-normalised and rank-32 affine-PCA structural references,
* the dictionary-chemistry binding tensor
  ``C_D = sum_v (alpha_v - alphabar)(q_v - qbar)^T  in R^{32 x 28}``,
* the label-free structural and binding reconstruction errors
  ``E_phi`` / ``E_bind``,
* dictionary-health statistics,
* a generic frozen-base + linear assignment-residual trainer and a matched
  learned dense-32-coordinate control.

The runner is ``zinc_sdb_v0.py``; focused CPU tests are
``tracks/ksvd/tests/test_sdb_v0.py``.  **Official ZINC test is never loaded.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

# ---------------------------------------------------------------------------
# frozen configuration (pre-registered; no sweep)
# ---------------------------------------------------------------------------

K_ATOMS = 32
SPARSITY = 8
PHI_DIM = int(r2.PHI_DIM)  # 65
ATOM_CATEGORIES = int(r2.ATOM_CATEGORIES)  # 28
DICT_SEED = 20260924
EPS = 1.0e-12

PROTOCOL_VERSION = "sdb_v0"


# ---------------------------------------------------------------------------
# dictionary fitting / coding
# ---------------------------------------------------------------------------


def normalize_columns(D: np.ndarray) -> np.ndarray:
    return T.normalize_columns(np.asarray(D, dtype=np.float64))


def random_normalized_dictionary(
    features: int = PHI_DIM, atoms: int = K_ATOMS, seed: int = DICT_SEED
) -> np.ndarray:
    return normalize_columns(
        T.random_normalized_dictionary(int(features), int(atoms), int(seed))
    )


def omp_codes(D: np.ndarray, X: np.ndarray, s: int = SPARSITY) -> np.ndarray:
    return T.omp_codes(np.asarray(D, dtype=np.float64), np.asarray(X, dtype=np.float64), s=int(s))


def fit_ksvd(
    X: np.ndarray,
    atoms: int = K_ATOMS,
    s: int = SPARSITY,
    epochs: int = 10,
    seed: int = DICT_SEED,
    *,
    D0: np.ndarray | None = None,
    log: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detached K-SVD fit of a ``K``-atom dictionary on ``X [N, F]``."""
    X = np.asarray(X, dtype=np.float64)
    start = random_normalized_dictionary(X.shape[1], int(atoms), int(seed)) if D0 is None else np.asarray(D0, dtype=np.float64)
    return T.ksvd_fit(
        X,
        start,
        s=int(s),
        epochs=int(epochs),
        seed=int(seed),
        log=(log if log is not None else (lambda *_: None)),
    )


@dataclass
class PCARank:
    mean: np.ndarray  # [F]
    components: np.ndarray  # [rank, F]

    def dense_codes(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean[None, :]) @ self.components.T

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        return self.mean[None, :] + np.asarray(codes, dtype=np.float64) @ self.components


def fit_pca_rank(X: np.ndarray, rank: int = K_ATOMS) -> PCARank:
    """Rank-``rank`` affine PCA; the best dense linear reconstruction reference."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    centered = X - mean[None, :]
    # right singular vectors of the centred matrix
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    r = min(int(rank), vt.shape[0])
    return PCARank(mean=mean, components=np.ascontiguousarray(vt[:r]))


# ---------------------------------------------------------------------------
# binding tensors
# ---------------------------------------------------------------------------


def binding_tensor(features: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Sum-centred ``sum_v (f_v - fbar)(q_v - qbar)^T`` for one molecule."""
    c_matrix, _p = r2.center_stats(
        np.asarray(features, dtype=np.float64), np.asarray(q, dtype=np.float64)
    )
    return c_matrix


def binding_tensor_from_codes(alpha: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``C_D`` for one molecule: ``sum_v (alpha_v - alphabar)(q_v - qbar)^T``."""
    return binding_tensor(alpha, q)


# ---------------------------------------------------------------------------
# reconstruction / binding errors
# ---------------------------------------------------------------------------


def structural_relative_error(
    X: np.ndarray,
    reconstruction: np.ndarray,
    mean: np.ndarray | None = None,
) -> float:
    """Aggregate ``E_phi = mean||phi - rec||^2 / mean||phi||^2``.

    ``reconstruction`` is the already-materialised ``[N, F]`` estimate of ``X``
    (``codes @ D.T`` for K-SVD, ``mean + codes @ components`` for the affine
    PCA reference).  ``mean`` (affine offset) is subtracted from the target
    only, so the denominator is always ``mean||phi||^2``.
    """
    X = np.asarray(X, dtype=np.float64)
    target = X if mean is None else X - np.asarray(mean, dtype=np.float64)[None, :]
    residual = target - np.asarray(reconstruction, dtype=np.float64)
    num = float(np.mean(np.sum(residual * residual, axis=1)))
    den = float(np.mean(np.sum(X * X, axis=1)))
    return num / (den + EPS)


def binding_relative_error(
    recon_op: np.ndarray,
    molecules: Sequence[Any],
    codes: Sequence[np.ndarray],
) -> dict[str, float]:
    """Aggregate ``E_bind = sum||C_phi - R C_code||_F^2 / sum||C_phi||_F^2``.

    ``recon_op`` is the ``[F, rank]`` operator that maps the code-space binding
    back to structural-coordinate space: ``D`` for K-SVD (``C_phi ~ D C_D``)
    and ``V^T`` for the affine-PCA reference (``C_phi ~ V^T C_a``).
    ``codes`` are the matching per-molecule ``[n_v, rank]`` assignments.
    """
    recon_op = np.asarray(recon_op, dtype=np.float64)
    num = 0.0
    den = 0.0
    ratios: list[float] = []
    for alpha, molecule in zip(codes, molecules):
        phi = np.asarray(molecule.phi, dtype=np.float64)
        q = r2.one_hot_q(np.asarray(molecule.atom_idx, dtype=np.int64))
        c_phi = binding_tensor(phi, q)
        c_code = binding_tensor(alpha, q)
        residual = c_phi - recon_op @ c_code
        n2 = float(np.sum(residual * residual))
        d2 = float(np.sum(c_phi * c_phi))
        num += n2
        den += d2
        if d2 > 0.0:
            ratios.append(n2 / (d2 + EPS))
    return {
        "aggregate": float(num / (den + EPS)),
        "per_molecule_median": float(np.median(ratios)) if ratios else float("nan"),
        "n_molecules": int(len(molecules)),
    }


# ---------------------------------------------------------------------------
# dictionary health
# ---------------------------------------------------------------------------


def dictionary_health(alpha: np.ndarray, atoms: int = K_ATOMS) -> dict[str, float]:
    """Coefficient-mass / support statistics of a code matrix ``[N, K]``."""
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 2:
        raise ValueError("alpha must be [N, K]")
    n, k = alpha.shape
    support = np.abs(alpha) > 0.0
    used = int((support.sum(axis=0) > 0).sum())
    support_freq = support.mean(axis=0)
    abs_alpha = np.abs(alpha)
    row_mass = abs_alpha.sum(axis=1, keepdims=True)
    row_mass[row_mass <= 0.0] = 1.0
    mass = abs_alpha / row_mass
    top1 = float(np.mean(np.sort(mass, axis=1)[:, -1]))
    top8 = float(np.mean(np.sort(mass, axis=1)[:, -min(8, k):].sum(axis=1)))
    entropy = float(np.mean(-(mass * np.log(mass + 1e-30)).sum(axis=1)))
    max_entropy = float(math.log(k)) if k > 1 else 0.0
    nonzero_rows = int((support.sum(axis=1) > 0).sum())
    return {
        "n_codes": int(n),
        "atoms": int(k),
        "used_atoms": used,
        "dead_atoms": int(k - used),
        "rows_with_support": nonzero_rows,
        "row_coverage": float(nonzero_rows / max(n, 1)),
        "support_entropy_mean": entropy,
        "support_entropy_normalized": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        "top1_mass_mean": top1,
        "top8_mass_mean": top8,
        "atom_support_frequency_min": float(support_freq[support_freq > 0].min()) if used else 0.0,
        "atom_support_frequency_max": float(support_freq.max()),
        "atom_support_frequency_mean": float(support_freq.mean()),
    }


def exact_sparsity(alpha: np.ndarray) -> tuple[int, bool]:
    """Return ``(max_l0, all_within_s)`` for the frozen ``s = SPARSITY``."""
    alpha = np.asarray(alpha)
    l0 = np.count_nonzero(alpha, axis=1)
    if l0.size == 0:
        return 0, True
    max_l0 = int(l0.max())
    return max_l0, bool(max_l0 <= SPARSITY)


# ---------------------------------------------------------------------------
# codes for a molecule collection
# ---------------------------------------------------------------------------


def codes_for_molecules(D: np.ndarray, molecules: Sequence[Any], s: int = SPARSITY) -> list[np.ndarray]:
    """Per-molecule ``alpha [n_v, K]`` using OMP over the frozen dictionary."""
    D = np.asarray(D, dtype=np.float64)
    out: list[np.ndarray] = []
    for molecule in molecules:
        phi = np.asarray(molecule.phi, dtype=np.float64)
        out.append(omp_codes(D, phi, s=s).astype(np.float64))
    return out


def binding_matrices_for_molecules(
    codes: Sequence[np.ndarray], molecules: Sequence[Any]
) -> np.ndarray:
    """Stack per-molecule ``C_D`` into ``[n_molecules, K, 28]``."""
    rows: list[np.ndarray] = []
    for alpha, molecule in zip(codes, molecules):
        q = r2.one_hot_q(np.asarray(molecule.atom_idx, dtype=np.int64))
        rows.append(binding_tensor(alpha, q))
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# frozen-base + linear assignment-residual trainer
# ---------------------------------------------------------------------------


def train_linear_residual(
    statistic_train: np.ndarray,
    base_train: np.ndarray,
    y_train: np.ndarray,
    statistic_valid: np.ndarray,
    base_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    protocol: Mapping[str, Any],
    *,
    width: int,
    categories: int = ATOM_CATEGORIES,
) -> dict[str, Any]:
    """Train a bias-free, zero-init ``[width, categories]`` weight on a frozen base.

    ``statistic_*`` are already RMS-normalised ``[N, width, categories]``
    tensors; the prediction is ``base + <W, statistic>``.
    """
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    weight = torch.zeros(int(width), int(categories), dtype=torch.float64)
    weight.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [weight],
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    s_train = torch.as_tensor(np.asarray(statistic_train), dtype=torch.float64)
    base_train_t = torch.as_tensor(np.asarray(base_train), dtype=torch.float64)
    residual_train = torch.as_tensor(np.asarray(y_train) - np.asarray(base_train), dtype=torch.float64)
    s_valid = torch.as_tensor(np.asarray(statistic_valid), dtype=torch.float64)
    base_valid_t = torch.as_tensor(np.asarray(base_valid), dtype=torch.float64)
    y_valid_t = torch.as_tensor(np.asarray(y_valid), dtype=torch.float64)

    n = int(s_train.shape[0])
    batch_size = int(protocol["batch_size"])
    max_epochs = int(protocol["max_epochs"])
    patience = int(protocol["patience"])
    clip = float(protocol["gradient_clip_norm"])
    generator = torch.Generator().manual_seed(int(seed) + int(protocol.get("train_shuffle_seed_offset", 0)))

    def _valid_mae(current: "torch.Tensor") -> float:
        prediction = base_valid_t + (s_valid * current).sum(dim=(1, 2))
        return float((prediction - y_valid_t).abs().mean())

    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    top5: list[tuple[float, int, "torch.Tensor"]] = []
    curve: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            prediction = (s_train[index] * weight).sum(dim=(1, 2))
            loss = torch.nn.functional.l1_loss(prediction, residual_train[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([weight], clip)
            optimizer.step()
        valid_mae = _valid_mae(weight)
        curve.append({"epoch": float(epoch), "valid_mae": float(valid_mae)})
        state = weight.detach().clone()
        top5.append((float(valid_mae), int(epoch), state))
        top5.sort(key=lambda item: (item[0], item[1]))
        top5 = top5[:5]
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = state
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    soup_state = torch.stack([item[2] for item in top5], dim=0).mean(dim=0)
    return {
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curve)),
        "top5_epochs": [int(item[1]) for item in top5],
        "top5_soup_valid_mae": float(_valid_mae(soup_state)),
        "W_norm": float(best_state.norm()),
        "W_soup_norm": float(soup_state.norm()),
        "W_soup": soup_state,
        "trainable_params": int(weight.numel()),
        "curve": curve,
    }


def fit_rms_scaler(statistic: np.ndarray, floor: float = r2.SCALER_FLOOR) -> tuple[np.ndarray, np.ndarray]:
    """Train-only per-coordinate RMS scaler + mask (no mean subtraction)."""
    statistic = np.asarray(statistic, dtype=np.float64)
    rms = np.sqrt((statistic ** 2).mean(axis=0))
    mask = (rms > float(floor)).astype(np.float64)
    scale = np.where(mask > 0.0, np.sqrt((statistic ** 2).mean(axis=0) + r2.SCALER_EPS), 1.0)
    return scale.astype(np.float64), mask.astype(np.float64)


def apply_rms_scaler(statistic: np.ndarray, scale: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (np.asarray(statistic, dtype=np.float64) / scale[None, :, :]) * mask[None, :, :]


# ---------------------------------------------------------------------------
# assignment shuffle (evaluation-only)
# ---------------------------------------------------------------------------


def shuffle_chemistry_within_molecules(
    molecules: Sequence[Any], seed: int
) -> list[np.ndarray]:
    """Permute ``atom_idx`` within each molecule; returns shuffled idx arrays."""
    rng = np.random.default_rng(int(seed))
    out: list[np.ndarray] = []
    for molecule in molecules:
        idx = np.asarray(molecule.atom_idx, dtype=np.int64).copy()
        if idx.shape[0] > 1:
            order = rng.permutation(idx.shape[0])
            idx = idx[order]
        out.append(idx)
    return out


def shuffle_codes_against_chemistry(
    codes: Sequence[np.ndarray], molecules: Sequence[Any], seed: int
) -> list[np.ndarray]:
    """Permute the *code rows* within each molecule, keeping the code multiset.

    This keeps the dictionary-code multiset and the chemistry multiset and only
    breaks the ``alpha_v <-> q_v`` alignment (pre-registered Stage 2 mechanism
    probe).
    """
    rng = np.random.default_rng(int(seed))
    out: list[np.ndarray] = []
    for alpha, molecule in zip(codes, molecules):
        n = int(np.asarray(molecule.atom_idx).shape[0])
        if n > 1:
            order = rng.permutation(n)
            out.append(np.asarray(alpha, dtype=np.float64)[order])
        else:
            out.append(np.asarray(alpha, dtype=np.float64).copy())
    return out


__all__ = [
    "K_ATOMS",
    "SPARSITY",
    "PHI_DIM",
    "ATOM_CATEGORIES",
    "DICT_SEED",
    "PROTOCOL_VERSION",
    "normalize_columns",
    "random_normalized_dictionary",
    "omp_codes",
    "fit_ksvd",
    "PCARank",
    "fit_pca_rank",
    "binding_tensor",
    "binding_tensor_from_codes",
    "structural_relative_error",
    "binding_relative_error",
    "dictionary_health",
    "exact_sparsity",
    "codes_for_molecules",
    "binding_matrices_for_molecules",
    "train_linear_residual",
    "fit_rms_scaler",
    "apply_rms_scaler",
    "shuffle_chemistry_within_molecules",
    "shuffle_codes_against_chemistry",
]
