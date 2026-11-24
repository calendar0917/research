"""Fit-only descriptor projection, reference selection and branch inputs.

Everything here is fitted on the Fit molecules of one training size only:

1. sparse feature vocabulary (union of Fit root descriptors);
2. ``TruncatedSVD(n_components=64, n_iter=5, random_state=0)`` on the Fit matrix;
3. L2 normalisation of every projected root descriptor ``z_v``;
4. K = 64 real reference roots by fixed seed-0 k-means++ D^2 sampling on Fit z;
5. ``tau = max(1e-6, median_Fit(d_8th^2))`` and ``q_v = softmax(-d^2 / tau)``;
6. the fixed branch normalisation ``(t - mean_Fit(t)) / max(global_RMS_Fit, 1e-6)``
   and the shared per-epoch batch plan.

``FrozenProjection`` is the complete frozen map from a descriptor set to
``(z_v, q_v)``; the training-time checks reuse it to prove that every fitted
statistic comes from Fit only and to test relabelling invariance.

Nothing is re-fitted from Valid or Test.  Unseen descriptor keys are ignored
and their fraction is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD

import css_common as C
import css_descriptors as D

SVD_COMPONENTS = 64
SVD_N_ITER = 5
SVD_RANDOM_STATE = 0
N_REFERENCE = 64
EPS = 1e-6
PLAN_EPOCHS = 120


def l2_normalize_rows(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(values, np.float64)
    norms = np.linalg.norm(values, axis=1)
    zero = norms <= 1e-12
    out = np.zeros_like(values)
    non = ~zero
    out[non] = values[non] / norms[non, None]
    return out.astype(np.float32), int(zero.sum())


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    out = (a * a).sum(axis=1)[:, None] - 2.0 * a @ b.T + (b * b).sum(axis=1)[None, :]
    return np.maximum(out, 0.0)


def sample_references(z_fit: np.ndarray, k: int = N_REFERENCE,
                      seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """k-means++ D^2 sampling without Lloyd updates; selected points are real."""
    z_fit = np.asarray(z_fit, np.float64)
    n = int(z_fit.shape[0])
    if n < k:
        raise ValueError(f"only {n} Fit roots for {k} references")
    rng = np.random.default_rng(int(seed))
    chosen = [int(rng.integers(0, n))]
    closest = squared_distances(z_fit, z_fit[chosen])[:, 0]
    d2_trace = [float(n / 1.0)]  # first pick is uniform, not D^2
    for _ in range(1, k):
        total = float(closest.sum())
        if total <= 1e-12:
            raise ValueError(f"fewer than {k} distinct projected Fit vectors")
        prob = closest / total
        pick = int(rng.choice(n, p=prob))
        if closest[pick] <= 0.0:
            raise ValueError("duplicate projected reference sampled")
        chosen.append(pick)
        d2 = squared_distances(z_fit, z_fit[[pick]])[:, 0]
        closest = np.minimum(closest, d2)
        d2_trace.append(total)
    z_ref = z_fit[np.asarray(chosen)]
    if np.unique(z_ref, axis=0).shape[0] != k:
        raise ValueError("reference projections are not distinct")
    info = {
        "n_reference": int(k),
        "seed": int(seed),
        "algorithm": "kmeans++ D^2 sampling, no Lloyd update, fixed seed",
        "first_pick": "uniform",
        "d2_step_sums": d2_trace,
        "distinct_projection_vectors": int(np.unique(z_ref, axis=0).shape[0]),
    }
    return np.asarray(chosen, np.int64), info


def temperature(z_fit: np.ndarray, z_ref: np.ndarray, kth: int = 8) -> dict[str, Any]:
    d2 = squared_distances(z_fit, z_ref)
    index = min(int(kth), d2.shape[1]) - 1
    d8 = np.partition(d2, index, axis=1)[:, index]
    tau = max(EPS, float(np.median(d8)))
    return {
        "tau": tau,
        "d8_squared_median": float(np.median(d8)),
        "d8_squared_mean": float(d8.mean()),
        "d8_squared_min": float(d8.min()),
        "d8_squared_max": float(d8.max()),
        "d8_squared_p10": float(np.percentile(d8, 10)),
        "d8_squared_p90": float(np.percentile(d8, 90)),
        "nearest_squared_mean": float(d2.min(axis=1).mean()),
        "nearest_squared_median": float(np.median(d2.min(axis=1))),
        "nearest_squared_max": float(d2.min(axis=1).max()),
        "kth_neighbour": int(index + 1),
    }


def reference_weights(z: np.ndarray, z_ref: np.ndarray, tau: float,
                      chunk: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    n = int(z.shape[0])
    q = np.zeros((n, int(z_ref.shape[0])), np.float32)
    nearest = np.zeros(n, np.float64)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        d2 = squared_distances(z[start:stop], z_ref)
        nearest[start:stop] = d2.min(axis=1)
        logits = -d2 / float(tau)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        q[start:stop] = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    return q, nearest


def entropy_stats(q: np.ndarray) -> dict[str, Any]:
    q = np.asarray(q, np.float64)
    q = np.clip(q, 1e-300, 1.0)
    entropy = -(q * np.log(q)).sum(axis=1)
    effective = np.exp(entropy)
    return {
        "mean_entropy": float(entropy.mean()),
        "max_entropy": float(entropy.max()),
        "min_entropy": float(entropy.min()),
        "mean_effective_number": float(effective.mean()),
        "max_effective_number": float(effective.max()),
        "mean_max_weight": float(q.max(axis=1).mean()),
        "max_max_weight": float(q.max(axis=1).max()),
        "mean_column_weight": q.mean(axis=0),
        "spread_from_mean_l2": float(np.linalg.norm(q - q.mean(axis=0, keepdims=True),
                                                    axis=1).mean()),
        "row_sum_min": float(q.sum(axis=1).min()),
        "row_sum_max": float(q.sum(axis=1).max()),
        "min_weight": float(q.min()),
        "nonfinite": int((~np.isfinite(q)).sum()),
    }


def branch_normalisation(t_fit: np.ndarray) -> dict[str, Any]:
    t_fit = np.asarray(t_fit, np.float64)
    mean = t_fit.mean(axis=0)
    centred = t_fit - mean
    rms = float(np.sqrt(np.mean(centred * centred)))
    scale = max(rms, EPS)
    return {"mean": mean, "rms": rms, "scale": scale,
            "centred_mean_abs": float(np.abs(centred).mean())}


def normalise_branch(t: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    t = np.asarray(t, np.float64)
    return ((t - np.asarray(norm["mean"], np.float64))
            / float(norm["scale"])).astype(np.float32)


def make_batch_plan(n_fit: int, seed: int = 0, epochs: int = PLAN_EPOCHS) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 101)
    plan = np.zeros((int(epochs), int(n_fit)), np.int32)
    for epoch in range(int(epochs)):
        plan[epoch] = rng.permutation(int(n_fit)).astype(np.int32)
    return plan


@dataclass
class FrozenProjection:
    """The complete fitted map descriptor -> (z_v, q_v)."""

    vocab: np.ndarray
    components: np.ndarray
    z_ref: np.ndarray
    tau: float
    norm_direct: Mapping[str, Any]
    norm_share: Mapping[str, Any]
    info: Mapping[str, Any]

    def project(self, descriptors: Mapping[str, np.ndarray], n_roots: int,
                chunk: int = 40000) -> dict[str, Any]:
        matrix, stats = D.sparse_matrix(descriptors["keys"], descriptors["counts"],
                                        descriptors["root_ptr"], n_roots, self.vocab)
        projected = np.zeros((int(n_roots), int(self.components.shape[0])), np.float64)
        for start in range(0, int(n_roots), chunk):
            stop = min(int(n_roots), start + chunk)
            projected[start:stop] = matrix[start:stop] @ self.components.T
        z, zero_rows = l2_normalize_rows(projected)
        if z.shape[1] < SVD_COMPONENTS:
            pad = np.zeros((z.shape[0], SVD_COMPONENTS - z.shape[1]), np.float32)
            z = np.concatenate([z, pad], axis=1)
        q, nearest = reference_weights(z, self.z_ref, self.tau)
        return {"z": z, "q": q, "nearest": nearest, "matrix_stats": stats,
                "zero_rows": zero_rows, "matrix": matrix}

    def branch_tables(self, z: np.ndarray, q: np.ndarray):
        return (normalise_branch(z, self.norm_direct),
                normalise_branch(q, self.norm_share))


def fit_projection(descriptors: Mapping[str, np.ndarray], fit_rows: Sequence[int],
                   fit_root_mask: np.ndarray, n_roots: int, seed: int = 0,
                   verbose: bool = False) -> tuple[FrozenProjection, dict[str, Any]]:
    keys = descriptors["keys"]
    counts = descriptors["counts"]
    root_ptr = descriptors["root_ptr"]
    vocab = D.build_vocabulary(keys, root_ptr, descriptors["root_row"], fit_rows)
    matrix, stats = D.sparse_matrix(keys, counts, root_ptr, n_roots, vocab)
    n_features = int(matrix.shape[1])
    if n_features < 2:
        raise ValueError(f"degenerate descriptor vocabulary: {n_features} features")
    n_components = int(min(SVD_COMPONENTS, n_features))
    fit_index = np.flatnonzero(fit_root_mask)
    fit_matrix = matrix[fit_index]
    svd = TruncatedSVD(n_components=n_components, n_iter=SVD_N_ITER,
                       random_state=SVD_RANDOM_STATE, algorithm="randomized")
    svd.fit(fit_matrix)
    components = np.asarray(svd.components_, np.float64)
    projected = matrix @ components.T
    z, zero_rows = l2_normalize_rows(projected)
    if n_components < SVD_COMPONENTS:
        pad = np.zeros((z.shape[0], SVD_COMPONENTS - n_components), np.float32)
        z = np.concatenate([z, pad], axis=1)
    z_fit = z[fit_index]
    ref_pos, ref_info = sample_references(z_fit, k=N_REFERENCE, seed=seed)
    z_ref = z_fit[ref_pos]
    tau_info = temperature(z_fit, z_ref)
    q, nearest = reference_weights(z, z_ref, tau_info["tau"])
    q_fit = q[fit_index]
    norm_direct = branch_normalisation(z_fit)
    norm_share = branch_normalisation(q_fit)
    singular = np.asarray(svd.singular_values_, np.float64)
    ratio = np.asarray(svd.explained_variance_ratio_, np.float64)
    info = {
        "n_features": n_features,
        "n_svd_components": n_components,
        "padded_components": int(SVD_COMPONENTS - n_components),
        "effective_components": int((ratio > 1e-12).sum()),
        "singular_values": singular,
        "explained_variance_ratio": ratio,
        "explained_variance_total": float(ratio.sum()),
        "zero_projected_rows": int(zero_rows),
        "matrix_stats": stats,
        "vocab_sha256": C.array_sha256(vocab),
        "components_sha256": C.array_sha256(components),
        "fit_matrix_nnz": int(fit_matrix.nnz),
        "density_fit": float(fit_matrix.nnz / max(1, fit_index.size * n_features)),
        "fit_matrix_sha256": C.array_sha256(fit_matrix.data, fit_matrix.indices,
                                            fit_matrix.indptr),
        "ref_positions": ref_pos,
        "reference_selection": ref_info,
        "temperature": tau_info,
        "q_stats": entropy_stats(q_fit),
        "z_fit_sha256": C.array_sha256(z_fit),
        "q_fit_sha256": C.array_sha256(q_fit),
    }
    if verbose:
        print(f"projection: D={n_features} shown={n_components} "
              f"evr={info['explained_variance_total']:.4f} zero_rows={zero_rows} "
              f"tau={tau_info['tau']:.5f}", flush=True)
    frozen = FrozenProjection(vocab=vocab, components=components, z_ref=z_ref,
                              tau=float(tau_info["tau"]), norm_direct=norm_direct,
                              norm_share=norm_share, info=info)
    return frozen, info


def canonical_branch_tables(z: np.ndarray, q: np.ndarray,
                            norm_direct: Mapping[str, Any], norm_share: Mapping[str, Any],
                            structs: Mapping[int, Any], row_atom_ptr: np.ndarray,
                            n_mol: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom branch inputs in the canonical atom order of each molecule."""
    t_direct = normalise_branch(z, norm_direct)
    t_share = normalise_branch(q, norm_share)
    out_direct = np.zeros((int(row_atom_ptr[-1]), SVD_COMPONENTS), np.float32)
    out_share = np.zeros((int(row_atom_ptr[-1]), SVD_COMPONENTS), np.float32)
    for row in range(n_mol):
        st = structs[row]
        order = np.asarray(st.support_atoms, np.int64)  # canonical rank -> original id
        lo = int(row_atom_ptr[row])
        if order.size != st.n_atoms:
            raise ValueError(f"row {row}: canonical order width mismatch")
        source = lo + order
        out_direct[lo:lo + st.n_atoms] = t_direct[source]
        out_share[lo:lo + st.n_atoms] = t_share[source]
    return out_direct, out_share
