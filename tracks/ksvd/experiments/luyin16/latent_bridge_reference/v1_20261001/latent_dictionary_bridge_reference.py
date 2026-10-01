"""NumPy reference for a proposed dictionary bridge, not a ZINC experiment.

The target integration point is SEM108Model.environments() -> E in R^48.
The bridge replaces E for both unary pooling and static pair construction.
No graph indices are accepted: there can be no neighbour/state write-back.
No data, checkpoint, PyTorch, or performance claim is included here.
"""
from __future__ import annotations

import numpy as np


def frame_initialization(dim: int = 48, seed: int = 0, dtype=np.float64):
    """Unit-column tight frame [I,Q]; distinct orthogonal bases, not duplicates."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    q *= np.where(np.diag(r) < 0, -1.0, 1.0)[None, :]
    dictionary = np.concatenate([np.eye(dim), q], axis=1).astype(dtype)
    values = dictionary.T.copy()
    return dictionary, values


def normalized_dictionary(dictionary):
    norms = np.linalg.norm(dictionary, axis=0, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero dictionary column")
    return dictionary / norms


def soft_threshold(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def elastic_objective(x, alpha, dictionary, lambda1, lambda2):
    residual = x - alpha @ dictionary.T
    return (
        0.5 * np.sum(residual * residual, axis=1)
        + lambda1 * np.sum(np.abs(alpha), axis=1)
        + 0.5 * lambda2 * np.sum(alpha * alpha, axis=1)
    )


def ista_codes(x, dictionary, *, lambda1=0.05, lambda2=0.01, steps=16,
               return_history=False):
    """Finite unrolled ISTA, not a claim of an exactly solved elastic net.

    Production uses normalized D, zero initial coefficients and a conservative
    step from the largest eigenvalue of the small d x d Gram matrix.
    In a PyTorch translation, the step estimate is a detached solver setting.
    """
    if lambda1 < 0 or lambda2 < 0 or steps <= 0:
        raise ValueError("invalid solver setting")
    dbar = normalized_dictionary(dictionary)
    spectral = float(np.linalg.eigvalsh(dbar @ dbar.T)[-1])
    step = 1.0 / (1.05 * spectral + lambda2)
    alpha = np.zeros((len(x), dbar.shape[1]), dtype=x.dtype)
    history = [elastic_objective(x, alpha, dbar, lambda1, lambda2)]
    for _ in range(steps):
        gradient = (alpha @ dbar.T - x) @ dbar + lambda2 * alpha
        alpha = soft_threshold(alpha - step * gradient, step * lambda1)
        if return_history:
            history.append(elastic_objective(x, alpha, dbar, lambda1, lambda2))
    return (alpha, np.stack(history)) if return_history else alpha


def bridge(h, dictionary, values, *, lambda1=0.05, lambda2=0.01, steps=16):
    """h[d] -> coefficients[2d] -> decoded E[d], with no raw residual bypass.

    Rowwise RMS removes a moving global scale; it is restored after decoding.
    Values start at D.T but are an independent task-trained parameter matrix.
    """
    scale = np.sqrt(np.mean(h * h, axis=1, keepdims=True) + 1e-12)
    normalized = h / scale
    alpha = ista_codes(normalized, dictionary, lambda1=lambda1,
                       lambda2=lambda2, steps=steps)
    decoded = (alpha @ values) * scale
    return decoded, {"alpha": alpha, "normalized": normalized, "scale": scale}


def symmetric_pair(a, b):
    return np.concatenate([a + b, np.abs(a - b), a * b], axis=-1)
