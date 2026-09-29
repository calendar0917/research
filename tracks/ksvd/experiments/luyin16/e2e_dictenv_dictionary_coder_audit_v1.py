"""E2E-DictEnv-Dictionary-Coder-Audit-v1 — dictionary / coder disentanglement.

Round ``e2e_dictenv_dictionary_coder_audit_v1`` (Workstream Z, ZINC).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_dictionary_coder_audit_v1_preregistration.md``.

Question.  The closed ``e2e_dictenv_clean_mechanism_v1`` round established that
the FINAL-CLEAN architecture (C6 + paired node binding + paired edge binding +
sparse tied-IHT + full relation) is safe, but that *sparse dictionary
specificity is not established*: the matched DenseTied control is not worse
(``G_dict = -0.002936``).  The sparse code is strongly concentrated
(top-5 atoms carry 4.4-5.0 of the 8 activations per row).

This module audits the *representation* with **no predictor training**.  It
implements, reusing the frozen implementations rather than duplicating them:

* same-``D`` coder variants — IHT-10 / IHT-30 / IHT-100 (``e2e_dictenv_v0``)
  and exact OMP(s=8) (``tccd_v0``) plus the frozen DenseTied coordinate
  (``phi @ Dbar``, exactly the P1 control definition);
* reconstruction, code-concentration, coefficient-geometry, support-agreement
  and coefficient-agreement metrics;
* a fixed, named, topology-only structural descriptor vector recovered from the
  phi65 provenance (``fsar_r2_ar0`` / ``fsar_v2``): the same tensor the
  dictionary actually receives;
* mean-direction / PCA common-direction diagnostics;
* the train-only-standardised atom structural profiles (SMD), specialisation
  scores and train->valid profile stability;
* cross-seed dictionary matching under permutation + sign symmetry;
* train-only linear sparse<->dense recoverability (R^2, CKA, cosine);
* the frozen IHT-30 training gate.

CPU only.  ``official_test_loaded`` is ``False`` in every payload.  The module
imports nothing that reads the official test split.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

PROTOCOL_VERSION = "e2e_dictenv_dictionary_coder_audit_v1"

#: frozen dictionary geometry of the FINAL-CLEAN architecture.
K_ATOMS = int(v0.K_ATOMS)          # 32
SPARSITY = int(v0.SPARSITY)        # 8
PHI_DIM = int(v0.PHI_DIM)          # 65
EPS = float(v0.EPS)

#: frozen same-D coder variants (exact step counts from the pre-registration).
IHT_STEPS: dict[str, int] = {"iht10": 10, "iht30": 30, "iht100": 100}
CODERS: tuple[str, ...] = ("iht10", "iht30", "iht100", "omp")
DENSE_CODER = "dense_tied"
SPLITS: tuple[str, ...] = ("train", "valid")
SEEDS: tuple[int, ...] = (0, 1, 2)

#: frozen top-response fraction for the stricter per-atom population.
TOP_RESPONSE_FRACTION = 0.10

# ---------------------------------------------------------------------------
# frozen IHT-30 training gate (pre-registration section 8; never tuned later)
# ---------------------------------------------------------------------------

GATE_RECON_FACTOR = 0.50          # IHT30 must be <= 0.5x the IHT10 Frobenius error
GATE_RECON_MIN_ABS = 1.0e-4       # and the absolute drop must be >= 1e-4
GATE_SUPPORT_JACCARD_GAIN = 0.05  # |mean Jaccard(IHT30, OMP) - (IHT10, OMP)|
GATE_EFF_ATOMS_GAIN = 1.0         # N_eff increase
GATE_TOP5_SHARE_DROP = 0.10       # top-5 activation share decrease
GATE_MAX_RATE_DROP = 0.10         # max activation rate decrease
GATE_TOP1_L1_SHARE_DROP = 0.05    # mean per-row top-1/L1 share decrease
GATE_CODE_COSINE_GAIN = 0.05      # mean code cosine to OMP increase
GATE_SEEDS_REQUIRED = 2           # of 3 dictionaries, direction-consistent

#: cross-seed vocabulary verdict thresholds.
STABLE_MEAN_COS = 0.90
STABLE_N90 = 24
STABLE_PROFILE_COS = 0.80
PARTIAL_MEAN_COS = 0.70
PARTIAL_N90 = 16
PARTIAL_PROFILE_COS = 0.50


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    """Every payload this module writes must carry the false blocker."""
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


def cpu_only_guard(device: Any) -> None:
    """Analysis and training in this round are CPU-only."""
    import torch

    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise RuntimeError(f"dictionary-coder audit is CPU-only, got device={resolved}")


# ---------------------------------------------------------------------------
# dictionaries and coders (all variants share one frozen D)
# ---------------------------------------------------------------------------


def effective_dictionary(D: np.ndarray) -> np.ndarray:
    """``Dbar`` — column-normalised dictionary, exactly the model's operator."""
    return np.asarray(T.normalize_columns(np.asarray(D, dtype=np.float64)), dtype=np.float64)


def effective_dictionary_torch(D: np.ndarray) -> "Any":  # torch.Tensor
    import torch

    return torch.as_tensor(effective_dictionary(D), dtype=torch.float32)


def iht_codes(Dbar: Any, X: Any, steps: int) -> Any:
    """IHT codes of the frozen shared coder (never re-implemented here)."""
    return v0.tied_iht_codes(Dbar, X, s=SPARSITY, steps=int(steps))


def omp_codes(Dbar: np.ndarray, X: np.ndarray, s: int = SPARSITY) -> np.ndarray:
    """Exact OMP(s) reference solver over the column-normalised ``Dbar``."""
    return T.omp_codes(np.asarray(Dbar, dtype=np.float64), np.asarray(X, dtype=np.float64), s=int(s))


def dense_codes(X: np.ndarray, Dbar: np.ndarray) -> np.ndarray:
    """The frozen DenseTied coordinate of P1: ``z = phi @ Dbar``."""
    return np.asarray(X, dtype=np.float64) @ np.asarray(Dbar, dtype=np.float64)


def codes_for(
    coder: str, X: np.ndarray, Dbar: np.ndarray, *, Dbar_torch: Any | None = None
) -> np.ndarray:
    """Compute one coder's codes on ``X`` with the single frozen ``Dbar``."""
    if coder in IHT_STEPS:
        import torch

        if Dbar_torch is None:
            Dbar_torch = effective_dictionary_torch(Dbar)
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        with torch.no_grad():
            return iht_codes(Dbar_torch, X_t, IHT_STEPS[coder]).detach().cpu().numpy().astype(np.float64)
    if coder == "omp":
        return omp_codes(Dbar, X).astype(np.float64)
    if coder == DENSE_CODER:
        return dense_codes(X, Dbar)
    raise KeyError(f"unknown coder {coder!r}")


# ---------------------------------------------------------------------------
# reconstruction metrics
# ---------------------------------------------------------------------------


def row_squared_relative_error(X: np.ndarray, A: np.ndarray, Dbar: np.ndarray) -> np.ndarray:
    """Per-row squared relative reconstruction error (the training definition).

    ``sum((x - a Dbar^T)^2) / (sum(x^2) + EPS)``.
    """
    residual = np.asarray(X, dtype=np.float64) - np.asarray(A, dtype=np.float64) @ np.asarray(Dbar, dtype=np.float64).T
    numerator = np.einsum("ij,ij->i", residual, residual)
    denominator = np.einsum("ij,ij->i", X, X) + EPS
    return numerator / denominator


def row_relative_l2(X: np.ndarray, A: np.ndarray, Dbar: np.ndarray) -> np.ndarray:
    residual = np.asarray(X, dtype=np.float64) - np.asarray(A, dtype=np.float64) @ np.asarray(Dbar, dtype=np.float64).T
    return np.linalg.norm(residual, axis=1) / np.sqrt(
        np.einsum("ij,ij->i", X, X) + EPS
    )


def reconstruction_metrics(X: np.ndarray, A: np.ndarray, Dbar: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(X, dtype=np.float64) - np.asarray(A, dtype=np.float64) @ np.asarray(Dbar, dtype=np.float64).T
    squared = row_squared_relative_error(X, A, Dbar)
    relative = row_relative_l2(X, A, Dbar)
    return {
        "recon_frobenius": float(np.linalg.norm(residual) / (np.linalg.norm(X) + EPS)),
        "recon_mean_row_squared": float(np.mean(squared)),
        "row_l2": dist_summary(relative),
    }


# ---------------------------------------------------------------------------
# generic distribution / concentration statistics
# ---------------------------------------------------------------------------


def dist_summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {key: float("nan") for key in ("mean", "median", "p10", "p90", "p95", "min", "max")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def gini(values: Sequence[float] | np.ndarray) -> float:
    """Gini coefficient of a non-negative usage vector."""
    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if array.size == 0 or array.sum() <= 0:
        return 0.0
    n = array.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * array) / (n * array.sum())) - (n + 1.0) / n)


def activation_frequency(codes: np.ndarray) -> np.ndarray:
    """``f_j = #{v : j in supp(alpha_v)} / N`` for all K atoms."""
    support = np.abs(np.asarray(codes, dtype=np.float64)) > 0.0
    if support.shape[0] == 0:
        return np.zeros(support.shape[1], dtype=np.float64)
    return support.mean(axis=0)


def usage_concentration(frequencies: np.ndarray) -> dict[str, float]:
    """Activation-frequency concentration metrics of one code matrix."""
    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    n_atoms = int(frequencies.size)
    active = int((frequencies > 0).sum())
    total = float(frequencies.sum())
    probabilities = frequencies / total if total > 0 else np.zeros_like(frequencies)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum()) if nonzero.size else 0.0
    shares = np.sort(frequencies)[::-1]
    return {
        "active_atoms": active,
        "active_atom_fraction": float(active / max(n_atoms, 1)),
        "effective_atoms": float(math.exp(entropy)),
        "usage_entropy": entropy,
        "top1_share": float(shares[0]) if shares.size else float("nan"),
        "top3_share": float(shares[:3].sum()) if shares.size else float("nan"),
        "top5_share": float(shares[:5].sum()) if shares.size else float("nan"),
        "top8_share": float(shares[:8].sum()) if shares.size else float("nan"),
        "max_activation_rate": float(frequencies.max()) if frequencies.size else float("nan"),
        "gini": gini(frequencies),
    }


def coefficient_geometry(codes: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-row coefficient geometry distributions."""
    absolute = np.abs(np.asarray(codes, dtype=np.float64))
    if absolute.shape[0] == 0:
        empty = dist_summary([])
        return {key: dict(empty) for key in ("l1", "l2", "max_abs", "top1_over_l1", "top3_over_l1")}
    l1 = absolute.sum(axis=1)
    l2 = np.linalg.norm(absolute, axis=1)
    maximum = absolute.max(axis=1)
    sorted_abs = np.sort(absolute, axis=1)[:, ::-1]
    top3 = sorted_abs[:, : min(3, absolute.shape[1])].sum(axis=1)
    denominator = l1 + EPS
    return {
        "l1": dist_summary(l1),
        "l2": dist_summary(l2),
        "max_abs": dist_summary(maximum),
        "top1_over_l1": dist_summary(maximum / denominator),
        "top3_over_l1": dist_summary(top3 / denominator),
    }


# ---------------------------------------------------------------------------
# support / coefficient agreement against the OMP reference
# ---------------------------------------------------------------------------


def support_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Per-row Jaccard index of two supports."""
    left_support = np.abs(np.asarray(left, dtype=np.float64)) > 0.0
    right_support = np.abs(np.asarray(right, dtype=np.float64)) > 0.0
    if left_support.shape != right_support.shape:
        raise ValueError("support shapes differ")
    intersection = np.logical_and(left_support, right_support).sum(axis=1).astype(np.float64)
    union = np.logical_or(left_support, right_support).sum(axis=1).astype(np.float64)
    return intersection / np.maximum(union, 1.0)


def support_agreement(codes: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    jaccard = support_jaccard(codes, reference)
    left_support = np.abs(np.asarray(codes, dtype=np.float64)) > 0.0
    right_support = np.abs(np.asarray(reference, dtype=np.float64)) > 0.0
    intersection = np.logical_and(left_support, right_support).sum(axis=1).astype(np.float64)
    union = np.logical_or(left_support, right_support).sum(axis=1).astype(np.float64)
    return {
        "jaccard": dist_summary(jaccard),
        "exact_support_match_rate": float(np.mean(intersection == union)) if jaccard.size else float("nan"),
        "top_s_intersection": dist_summary(intersection),
    }


def rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / (denominator + EPS)


def rowwise_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    return rowwise_cosine(left_centered, right_centered)


def coefficient_agreement(codes: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    codes = np.asarray(codes, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = rowwise_pearson(codes, reference)
    finite = pearson[np.isfinite(pearson)]
    return {
        "cosine": dist_summary(rowwise_cosine(codes, reference)),
        "pearson": dist_summary(finite),
        "normalized_l2_difference": dist_summary(
            np.linalg.norm(codes - reference, axis=1) / (np.linalg.norm(reference, axis=1) + EPS)
        ),
    }


# ---------------------------------------------------------------------------
# fixed topology-only structural descriptors from the phi65 provenance
# ---------------------------------------------------------------------------
#
# phi layout (``fsar_r2_ar0.phi_for_center``):
#   [0:11]   root basis  = root indicator | root shell one-hot | log1p(degree) |
#                          log1p(neighbour-by-shell) | log1p(rooted walks A^k)
#   [11:22]  patch mean of the root basis
#   [22:33]  patch std of the root basis
#   [33:48]  patch mean of the 15-D induced-edge basis
#   [48:63]  patch std of the 15-D induced-edge basis
#   [63]     log1p(patch node count)   [64] log1p(patch edge count)
#
# edge basis = shellpair one-hot (6) | log1p(deg_i+deg_j) | log1p|deg_i-deg_j| |
#              log1p(common neighbours) | log1p(nbr shell sum)(3) | log1p(|nbr shell diff|)(3)

STRUCTURAL_DESCRIPTORS: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("patch_nodes", lambda phi: np.expm1(phi[:, 63])),
    ("patch_edges", lambda phi: np.expm1(phi[:, 64])),
    ("root_induced_degree", lambda phi: np.expm1(phi[:, 4])),
    ("root_neighbour_shell1", lambda phi: np.expm1(phi[:, 5])),
    ("root_neighbour_shell2", lambda phi: np.expm1(phi[:, 6])),
    ("root_walk1", lambda phi: np.expm1(phi[:, 8])),
    ("root_walk2", lambda phi: np.expm1(phi[:, 9])),
    ("root_walk3", lambda phi: np.expm1(phi[:, 10])),
    ("shell_pop1", lambda phi: np.expm1(phi[:, 63]) * phi[:, 13]),
    ("shell_pop2", lambda phi: np.expm1(phi[:, 63]) * phi[:, 14]),
    ("shell_frac1", lambda phi: phi[:, 13]),
    ("shell_frac2", lambda phi: phi[:, 14]),
    ("mean_log1p_induced_degree", lambda phi: phi[:, 15]),
    ("std_log1p_induced_degree", lambda phi: phi[:, 26]),
    ("mean_log1p_neighbour_shell1", lambda phi: phi[:, 16]),
    ("mean_log1p_neighbour_shell2", lambda phi: phi[:, 17]),
    ("mean_log1p_walk1", lambda phi: phi[:, 19]),
    ("mean_log1p_walk2", lambda phi: phi[:, 20]),
    ("mean_log1p_walk3", lambda phi: phi[:, 21]),
    ("std_log1p_walk2", lambda phi: phi[:, 31]),
    ("std_log1p_walk3", lambda phi: phi[:, 32]),
    ("edge_frac_shellpair_00", lambda phi: phi[:, 33]),
    ("edge_frac_shellpair_01", lambda phi: phi[:, 34]),
    ("edge_frac_shellpair_02", lambda phi: phi[:, 35]),
    ("edge_frac_shellpair_11", lambda phi: phi[:, 36]),
    ("edge_frac_shellpair_12", lambda phi: phi[:, 37]),
    ("edge_frac_shellpair_22", lambda phi: phi[:, 38]),
    ("mean_edge_log1p_degree_sum", lambda phi: phi[:, 39]),
    ("mean_edge_log1p_degree_absdiff", lambda phi: phi[:, 40]),
    ("mean_edge_log1p_common_neighbours", lambda phi: phi[:, 41]),
    ("std_edge_log1p_degree_sum", lambda phi: phi[:, 54]),
    ("std_edge_log1p_common_neighbours", lambda phi: phi[:, 56]),
)

STRUCTURAL_DESCRIPTOR_NAMES: tuple[str, ...] = tuple(name for name, _ in STRUCTURAL_DESCRIPTORS)


def structural_descriptors(phi: np.ndarray) -> np.ndarray:
    """``[N, p]`` fixed named structural descriptor matrix from raw phi65."""
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != PHI_DIM:
        raise ValueError(f"phi must be [N, {PHI_DIM}], got {phi.shape}")
    columns = [function(phi) for _, function in STRUCTURAL_DESCRIPTORS]
    return np.stack(columns, axis=1)


# ---------------------------------------------------------------------------
# common-direction diagnostics
# ---------------------------------------------------------------------------


def mean_direction(X: np.ndarray) -> np.ndarray:
    """Train mean of the *dictionary input space* (raw phi65; no re-scaling)."""
    return np.asarray(X, dtype=np.float64).mean(axis=0)


def atom_cosine_to_vector(Dbar: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Absolute cosine of every atom against a direction (sign symmetry)."""
    Dbar = np.asarray(Dbar, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(Dbar, axis=0) * (np.linalg.norm(vector) + EPS)
    return np.abs(Dbar.T @ vector) / (denominator + EPS)


def pca_basis(X_train: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train-only PCA (mean-centred SVD).

    Returns ``(components [k, F], explained_variance_ratio [k], mean [F])``.
    The valid split never participates.
    """
    X = np.asarray(X_train, dtype=np.float64)
    mean = X.mean(axis=0)
    centered = X - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    ratio = variance / (variance.sum() + EPS)
    k = int(min(n_components, vt.shape[0]))
    return vt[:k].copy(), ratio[:k].copy(), mean


# ---------------------------------------------------------------------------
# atom structural profiles (SMD), specialisation and train->valid stability
# ---------------------------------------------------------------------------


def _profile_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _top_features(smd: np.ndarray, names: Sequence[str], k: int = 5) -> tuple[list[str], list[str]]:
    absolute = np.argsort(-np.abs(smd))[:k]
    signed = np.argsort(smd)[::-1][:k]
    return [str(names[i]) for i in absolute], [str(names[i]) for i in signed]


def atom_specialization_rows(
    *,
    seed: int,
    coder: str,
    Z_train: np.ndarray,
    Z_valid: np.ndarray,
    A_train: np.ndarray,
    A_valid: np.ndarray,
    top_fraction: float = TOP_RESPONSE_FRACTION,
) -> dict[str, Any]:
    """Per-atom SMD profiles, specialisation scores and train->valid stability.

    Normalisation (mean / std) is fit on train only and frozen for valid.  The
    stricter ``top-response`` population is the top ``top_fraction`` of the
    non-zero ``|alpha_vj|`` of that atom — no target is ever read.
    """
    Z_train = np.asarray(Z_train, dtype=np.float64)
    Z_valid = np.asarray(Z_valid, dtype=np.float64)
    A_train = np.asarray(A_train, dtype=np.float64)
    A_valid = np.asarray(A_valid, dtype=np.float64)
    names = list(STRUCTURAL_DESCRIPTOR_NAMES)
    mean = Z_train.mean(axis=0)
    std = Z_train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    rows: list[dict[str, Any]] = []
    profiles: dict[str, list[list[float]]] = {"train_active": [], "valid_active": []}
    for atom in range(A_train.shape[1]):
        active_train = np.abs(A_train[:, atom]) > 0.0
        active_valid = np.abs(A_valid[:, atom]) > 0.0
        n_active_train = int(active_train.sum())
        n_active_valid = int(active_valid.sum())
        row: dict[str, Any] = {
            "seed": int(seed),
            "coder": str(coder),
            "atom": int(atom),
            "activation_rate_train": float(n_active_train / max(A_train.shape[0], 1)),
            "activation_rate_valid": float(n_active_valid / max(A_valid.shape[0], 1)),
            "n_active_train": n_active_train,
            "n_active_valid": n_active_valid,
        }
        smd_train = np.full(len(names), np.nan)
        smd_valid = np.full(len(names), np.nan)
        if n_active_train >= 2:
            smd_train = (Z_train[active_train].mean(axis=0) - mean) / std
            row["specialization_score"] = float(np.sqrt(np.mean(smd_train**2)))
            top_abs, top_signed = _top_features(smd_train, names)
            row["top_structural_features"] = top_abs
            row["top_signed_structural_features"] = top_signed
        else:
            row["specialization_score"] = float("nan")
            row["top_structural_features"] = []
            row["top_signed_structural_features"] = []
        if n_active_valid >= 2:
            smd_valid = (Z_valid[active_valid].mean(axis=0) - mean) / std
            row["specialization_score_valid"] = float(np.sqrt(np.mean(smd_valid**2)))
        else:
            row["specialization_score_valid"] = float("nan")
        row["train_valid_profile_cosine"] = (
            _profile_cosine(smd_train, smd_valid)
            if n_active_train >= 2 and n_active_valid >= 2
            else float("nan")
        )
        # stricter top-response population (no target involved)
        train_values = np.abs(A_train[active_train, atom]) if n_active_train else np.zeros(0)
        valid_values = np.abs(A_valid[active_valid, atom]) if n_active_valid else np.zeros(0)
        row["top_response_specialization_score"] = float("nan")
        row["top_response_profile_cosine"] = float("nan")
        if train_values.size >= 2 and valid_values.size >= 2:
            train_cut = np.quantile(train_values, 1.0 - float(top_fraction))
            valid_cut = np.quantile(valid_values, 1.0 - float(top_fraction))
            train_subset = active_train.copy()
            train_subset[active_train] = train_values >= train_cut
            valid_subset = active_valid.copy()
            valid_subset[active_valid] = valid_values >= valid_cut
            if train_subset.sum() >= 2:
                smd_train_top = (Z_train[train_subset].mean(axis=0) - mean) / std
                row["top_response_specialization_score"] = float(np.sqrt(np.mean(smd_train_top**2)))
                if valid_subset.sum() >= 2:
                    smd_valid_top = (Z_valid[valid_subset].mean(axis=0) - mean) / std
                    row["top_response_profile_cosine"] = _profile_cosine(smd_train_top, smd_valid_top)
        profiles["train_active"].append(smd_train.tolist())
        profiles["valid_active"].append(smd_valid.tolist())
        rows.append(row)
    usage = np.asarray([row["activation_rate_train"] for row in rows], dtype=np.float64)
    scores = np.asarray([row["specialization_score"] for row in rows], dtype=np.float64)
    finite = np.isfinite(scores)
    weight = usage[finite] / max(usage[finite].sum(), EPS)
    return {
        "rows": rows,
        "profiles": profiles,
        "median_specialization": float(np.nanmedian(scores)) if finite.any() else float("nan"),
        "usage_weighted_specialization": float(np.sum(weight * scores[finite])) if finite.any() else float("nan"),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# cross-seed dictionary stability
# ---------------------------------------------------------------------------


def match_dictionaries(Dbar_a: np.ndarray, Dbar_b: np.ndarray) -> dict[str, Any]:
    """Hungarian matching of atoms under permutation + sign symmetry.

    Cost is ``-|cos|``; equal column counts are required.
    """
    left = np.asarray(Dbar_a, dtype=np.float64)
    right = np.asarray(Dbar_b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"dictionary shapes differ: {left.shape} vs {right.shape}")
    cosine = np.abs(left.T @ right)
    rows, columns = linear_sum_assignment(-cosine)
    matched = cosine[rows, columns]
    stats = dist_summary(matched)
    return {
        "pairs": [
            {"atom_a": int(i), "atom_b": int(j), "abs_cosine": float(cosine[i, j])}
            for i, j in zip(rows, columns)
        ],
        "matched_abs_cosine": stats,
        "n_above_090": int((matched >= 0.90).sum()),
        "n_above_095": int((matched >= 0.95).sum()),
        "official_test_loaded": False,
    }


def vocabulary_verdict(pair_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Frozen cross-seed vocabulary classification (pre-registration section 6)."""
    if not pair_results:
        return {"verdict": "NO_DATA", "pairs": 0}
    mean_cos = float(np.mean([float(pair["mean_matched_abs_cosine"]) for pair in pair_results]))
    n90 = float(np.mean([float(pair["n_above_090"]) for pair in pair_results]))
    profile_cosines = [value for pair in pair_results for value in pair["matched_profile_cosines"] if np.isfinite(value)]
    median_profile = float(np.median(profile_cosines)) if profile_cosines else float("nan")
    if mean_cos >= STABLE_MEAN_COS and n90 >= STABLE_N90 and median_profile >= STABLE_PROFILE_COS:
        verdict = "STABLE_VOCABULARY"
    elif (mean_cos >= PARTIAL_MEAN_COS or n90 >= PARTIAL_N90) and median_profile >= PARTIAL_PROFILE_COS:
        verdict = "PARTIAL_VOCABULARY"
    else:
        verdict = "UNSTABLE_BASIS"
    return {
        "verdict": verdict,
        "pairs": len(pair_results),
        "mean_matched_abs_cosine": mean_cos,
        "mean_n_above_090": n90,
        "median_matched_profile_cosine": median_profile,
        "thresholds": {
            "stable_mean_cos": STABLE_MEAN_COS,
            "stable_n90": STABLE_N90,
            "stable_profile_cos": STABLE_PROFILE_COS,
            "partial_mean_cos": PARTIAL_MEAN_COS,
            "partial_n90": PARTIAL_N90,
            "partial_profile_cos": PARTIAL_PROFILE_COS,
        },
    }


# ---------------------------------------------------------------------------
# sparse <-> dense linear recoverability
# ---------------------------------------------------------------------------


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(left.T @ right) ** 2)
    denominator = float(np.linalg.norm(left.T @ left) * np.linalg.norm(right.T @ right))
    return numerator / (denominator + EPS)


def recoverability_metrics(
    target_valid: np.ndarray,
    prediction: np.ndarray,
    target_train_mean: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(target_valid, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(target_train_mean, dtype=np.float64).reshape(1, -1)
    residual = target - prediction
    ss_res = float(np.einsum("ij,ij->i", residual, residual).sum())
    ss_tot = float(np.einsum("ij,ij->i", target - reference, target - reference).sum())
    per_dim_res = np.einsum("ij,ij->i", residual, residual)
    per_dim_tot = np.einsum("ij,ij->i", target - reference, target - reference)
    per_dim_r2 = 1.0 - per_dim_res / np.maximum(per_dim_tot, EPS)
    return {
        "valid_r2": float(1.0 - ss_res / (ss_tot + EPS)),
        "per_dimension_r2": dist_summary(per_dim_r2),
        "valid_normalized_error": float(np.linalg.norm(residual) / (np.linalg.norm(target) + EPS)),
        "valid_mean_cosine": dist_summary(rowwise_cosine(target, prediction)),
        "linear_cka": float(linear_cka(target, prediction)),
    }


def fit_linear_recoverability(
    source_train: np.ndarray,
    target_train: np.ndarray,
    source_valid: np.ndarray,
    target_valid: np.ndarray,
) -> dict[str, Any]:
    """Ordinary least squares (train only), evaluated on valid.

    A single fixed design: ``[features, 1] @ W`` via ``np.linalg.lstsq``; no
    ridge and no hyper-parameter selection.
    """
    source_train = np.asarray(source_train, dtype=np.float64)
    target_train = np.asarray(target_train, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=np.float64)
    target_valid = np.asarray(target_valid, dtype=np.float64)
    design_train = np.concatenate([source_train, np.ones((source_train.shape[0], 1))], axis=1)
    design_valid = np.concatenate([source_valid, np.ones((source_valid.shape[0], 1))], axis=1)
    coefficients, _residuals, rank, singular = np.linalg.lstsq(design_train, target_train, rcond=None)
    prediction = design_valid @ coefficients
    metrics = recoverability_metrics(target_valid, prediction, target_train.mean(axis=0))
    metrics["n_train"] = int(source_train.shape[0])
    metrics["n_valid"] = int(source_valid.shape[0])
    metrics["design_rank"] = int(rank)
    metrics["design_singular_max"] = float(singular.max()) if singular.size else float("nan")
    metrics["design_singular_min"] = float(singular.min()) if singular.size else float("nan")
    metrics["official_test_loaded"] = False
    return metrics


# ---------------------------------------------------------------------------
# the IHT-30 training gate
# ---------------------------------------------------------------------------


def iht30_gate(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Frozen gate: may FINAL-CLEAN + IHT-30 be trained?

    ``per_seed[seed]`` must carry the valid-split summary rows computed by the
    runner: ``frobenius``, ``support_jaccard_mean``, ``effective_atoms``,
    ``top5_share``, ``max_activation_rate``, ``top1_over_l1_mean``,
    ``code_cosine_mean`` for ``iht10`` and ``iht30``.
    """
    evidence: dict[str, Any] = {}
    supported: list[int] = []
    for seed, entry in sorted(per_seed.items()):
        ten = entry["iht10"]
        thirty = entry["iht30"]
        recon_factor = float(thirty["frobenius"] / max(float(ten["frobenius"]), EPS))
        recon_drop = float(ten["frobenius"]) - float(thirty["frobenius"])
        cond1 = bool(recon_factor <= GATE_RECON_FACTOR and recon_drop >= GATE_RECON_MIN_ABS)
        support_gain = float(thirty["support_jaccard_mean"]) - float(ten["support_jaccard_mean"])
        cond2a = bool(support_gain >= GATE_SUPPORT_JACCARD_GAIN)
        concentration = {
            "effective_atoms_delta": float(thirty["effective_atoms"]) - float(ten["effective_atoms"]),
            "top5_share_drop": float(ten["top5_share"]) - float(thirty["top5_share"]),
            "max_activation_rate_drop": float(ten["max_activation_rate"]) - float(thirty["max_activation_rate"]),
        }
        cond2b = bool(
            concentration["effective_atoms_delta"] >= GATE_EFF_ATOMS_GAIN
            or concentration["top5_share_drop"] >= GATE_TOP5_SHARE_DROP
            or concentration["max_activation_rate_drop"] >= GATE_MAX_RATE_DROP
        )
        geometry = {
            "top1_over_l1_drop": float(ten["top1_over_l1_mean"]) - float(thirty["top1_over_l1_mean"]),
            "code_cosine_gain": float(thirty["code_cosine_mean"]) - float(ten["code_cosine_mean"]),
        }
        cond2c = bool(
            geometry["top1_over_l1_drop"] >= GATE_TOP1_L1_SHARE_DROP
            or geometry["code_cosine_gain"] >= GATE_CODE_COSINE_GAIN
        )
        cond2 = bool(cond2a or cond2b or cond2c)
        seed_supported = bool(cond1 and cond2)
        if seed_supported:
            supported.append(int(seed))
        evidence[str(seed)] = {
            "recon_factor": recon_factor,
            "recon_drop": recon_drop,
            "condition_1_reconstruction": cond1,
            "support_jaccard_gain": support_gain,
            "condition_2a_support": cond2a,
            "concentration": concentration,
            "condition_2b_concentration": cond2b,
            "geometry": geometry,
            "condition_2c_geometry": cond2c,
            "condition_2": cond2,
            "seed_supported": seed_supported,
        }
    fired = len(supported) >= GATE_SEEDS_REQUIRED
    return {
        "gate": "IHT30_TRAIN",
        "fired": bool(fired),
        "supported_seeds": supported,
        "n_supported": len(supported),
        "seeds_required": GATE_SEEDS_REQUIRED,
        "evidence": evidence,
        "thresholds": {
            "recon_factor": GATE_RECON_FACTOR,
            "recon_min_abs": GATE_RECON_MIN_ABS,
            "support_jaccard_gain": GATE_SUPPORT_JACCARD_GAIN,
            "eff_atoms_gain": GATE_EFF_ATOMS_GAIN,
            "top5_share_drop": GATE_TOP5_SHARE_DROP,
            "max_rate_drop": GATE_MAX_RATE_DROP,
            "top1_l1_share_drop": GATE_TOP1_L1_SHARE_DROP,
            "code_cosine_gain": GATE_CODE_COSINE_GAIN,
            "seeds_required": GATE_SEEDS_REQUIRED,
        },
        "official_test_loaded": False,
    }


def coder_issue_classification(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Frozen descriptive classification of what IHT-10 gets wrong."""
    labels = {"A1_CODER_SUPPORT_ARTIFACT": 0, "A2_CODER_AMPLITUDE_BOTTLENECK": 0, "A3_INTRINSIC_CONCENTRATION": 0}
    per_seed_labels: dict[str, list[str]] = {}
    for seed, entry in sorted(per_seed.items()):
        ten = entry["iht10"]
        thirty = entry["iht30"]
        omp = entry["omp"]
        support_improves = float(thirty["support_jaccard_mean"]) - float(ten["support_jaccard_mean"]) >= GATE_SUPPORT_JACCARD_GAIN
        concentration_improves = (
            float(thirty["effective_atoms"]) - float(ten["effective_atoms"]) >= GATE_EFF_ATOMS_GAIN
            or float(ten["top5_share"]) - float(thirty["top5_share"]) >= GATE_TOP5_SHARE_DROP
        )
        recon_factor = float(thirty["frobenius"] / max(float(ten["frobenius"]), EPS))
        recon_improves = recon_factor <= GATE_RECON_FACTOR
        omp_concentrated = (
            float(omp["effective_atoms"]) <= float(ten["effective_atoms"]) + GATE_EFF_ATOMS_GAIN
            and float(omp["top5_share"]) >= float(ten["top5_share"]) - GATE_TOP5_SHARE_DROP
        )
        current: list[str] = []
        if support_improves and concentration_improves:
            current.append("A1_CODER_SUPPORT_ARTIFACT")
        if recon_improves and not support_improves and not concentration_improves:
            current.append("A2_CODER_AMPLITUDE_BOTTLENECK")
        if omp_concentrated:
            current.append("A3_INTRINSIC_CONCENTRATION")
        if not current:
            current.append("NONE")
        per_seed_labels[str(seed)] = current
        for label in current:
            if label in labels:
                labels[label] += 1
    return {
        "per_seed": per_seed_labels,
        "counts": labels,
        "fired": [label for label, count in labels.items() if count >= GATE_SEEDS_REQUIRED],
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# dominant-atom relation to topology variables (Spearman)
# ---------------------------------------------------------------------------


def dominant_atom_analysis(
    *,
    Dbar: np.ndarray,
    X_train: np.ndarray,
    Z_train: np.ndarray,
    alpha_train: np.ndarray,
    frequencies: np.ndarray,
    pca_components: np.ndarray,
    pca_ratio: np.ndarray,
    mean_vector: np.ndarray,
) -> dict[str, Any]:
    """Dominant-atom identity, mean/PCA cosines and topology correlations."""
    dominant = int(np.argmax(frequencies))
    magnitude = np.abs(np.asarray(alpha_train, dtype=np.float64)[:, dominant])
    correlations: list[dict[str, Any]] = []
    for index, name in enumerate(STRUCTURAL_DESCRIPTOR_NAMES):
        rho, pvalue = spearmanr(magnitude, np.asarray(Z_train, dtype=np.float64)[:, index])
        correlations.append({"feature": name, "spearman": float(rho), "pvalue": float(pvalue)})
    return {
        "dominant_atom": dominant,
        "dominant_atom_activation_rate": float(frequencies[dominant]),
        "atom_cosine_to_mean_direction": atom_cosine_to_vector(Dbar, mean_vector).tolist(),
        "dominant_atom_cosine_to_mean_direction": float(
            atom_cosine_to_vector(Dbar, mean_vector)[dominant]
        ),
        "atom_cosine_to_pcs": [
            atom_cosine_to_vector(Dbar, pca_components[index]).tolist()
            for index in range(int(pca_components.shape[0]))
        ],
        "dominant_atom_cosine_to_pcs": [
            float(atom_cosine_to_vector(Dbar, pca_components[index])[dominant])
            for index in range(int(pca_components.shape[0]))
        ],
        "pca_explained_variance_ratio": [float(value) for value in pca_ratio],
        "pca_top3_cumulative": float(np.sum(pca_ratio)),
        "dominant_atom_spearman": correlations,
        "official_test_loaded": False,
    }


__all__ = [
    "PROTOCOL_VERSION",
    "K_ATOMS",
    "SPARSITY",
    "PHI_DIM",
    "EPS",
    "IHT_STEPS",
    "CODERS",
    "DENSE_CODER",
    "SPLITS",
    "SEEDS",
    "TOP_RESPONSE_FRACTION",
    "GATE_RECON_FACTOR",
    "GATE_RECON_MIN_ABS",
    "GATE_SUPPORT_JACCARD_GAIN",
    "GATE_EFF_ATOMS_GAIN",
    "GATE_TOP5_SHARE_DROP",
    "GATE_MAX_RATE_DROP",
    "GATE_TOP1_L1_SHARE_DROP",
    "GATE_CODE_COSINE_GAIN",
    "GATE_SEEDS_REQUIRED",
    "official_test_blocker",
    "cpu_only_guard",
    "effective_dictionary",
    "effective_dictionary_torch",
    "iht_codes",
    "omp_codes",
    "dense_codes",
    "codes_for",
    "row_squared_relative_error",
    "row_relative_l2",
    "reconstruction_metrics",
    "dist_summary",
    "gini",
    "activation_frequency",
    "usage_concentration",
    "coefficient_geometry",
    "support_jaccard",
    "support_agreement",
    "rowwise_cosine",
    "rowwise_pearson",
    "coefficient_agreement",
    "STRUCTURAL_DESCRIPTORS",
    "STRUCTURAL_DESCRIPTOR_NAMES",
    "structural_descriptors",
    "mean_direction",
    "atom_cosine_to_vector",
    "pca_basis",
    "atom_specialization_rows",
    "match_dictionaries",
    "vocabulary_verdict",
    "linear_cka",
    "recoverability_metrics",
    "fit_linear_recoverability",
    "iht30_gate",
    "coder_issue_classification",
    "dominant_atom_analysis",
]
