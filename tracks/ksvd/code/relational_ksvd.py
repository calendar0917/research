"""Relation-regularized sparse dictionary learning.

The deployed encoder remains content-only OMP.  Typed patch relations are used
only while fitting the train-fold dictionary: sparse coefficient values are
optimized to reconstruct patch content and predict train-fold relations, then
the dictionary is updated by least squares.  Held-out patches are encoded from
their content alone, so relation prediction cannot read its own target.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ksvd import _omp, ksvd


def sparse_codes(D: np.ndarray, Y: np.ndarray, sparsity: int) -> np.ndarray:
    """Content-only OMP codes, shaped ``(n_atoms, n_patches)``."""
    return np.stack([_omp(D, Y[:, i], sparsity) for i in range(Y.shape[1])], axis=1)


def relational_ksvd(
    Y: np.ndarray,
    pair_index: np.ndarray,
    pair_labels: np.ndarray,
    *,
    n_atoms: int = 12,
    sparsity: int = 3,
    relation_weight: float = 0.1,
    outer_iter: int = 3,
    code_steps: int = 30,
    code_lr: float = 0.03,
    ridge: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a shared dictionary with a typed relation regularizer.

    ``pair_index`` is ``(n_pairs, 2)`` over columns of ``Y`` and
    ``pair_labels`` is ``(n_pairs, n_relation_types)``.  Each outer iteration:

    1. content-only OMP chooses a sparse support;
    2. supported coefficient values and bilinear relation decoders are refined;
    3. the dictionary is updated from the relation-refined coefficients.

    Relation decoders are training auxiliaries only and are not returned as a
    deployed predictor.
    """
    import torch
    import torch.nn.functional as F

    Y = np.asarray(Y, dtype=np.float64)
    pairs = np.asarray(pair_index, dtype=np.int64)
    labels = np.asarray(pair_labels, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] < 2:
        raise ValueError("Y must contain at least two patch columns")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_index must have shape (n_pairs, 2)")
    if labels.ndim != 2 or labels.shape[0] != pairs.shape[0]:
        raise ValueError("pair_labels must have shape (n_pairs, n_relation_types)")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= Y.shape[1]):
        raise ValueError("pair_index is outside the columns of Y")
    if relation_weight < 0 or outer_iter < 1 or code_steps < 1:
        raise ValueError("invalid relational optimization configuration")

    # Matched ordinary K-SVD initialization avoids attributing a lucky random
    # start to the relation objective.
    D, _, init_info = ksvd(
        Y,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=1,
        n_iter=4,
        seed=seed,
    )
    n_atoms = D.shape[1]
    n_rel = labels.shape[1]
    torch.manual_seed(seed)
    relation_mats = torch.nn.Parameter(
        0.01 * torch.randn(n_rel, n_atoms, n_atoms, dtype=torch.float64)
    )
    history: list[dict[str, float]] = []

    y_t = torch.tensor(Y, dtype=torch.float64)
    pair_t = torch.tensor(pairs, dtype=torch.long)
    label_t = torch.tensor(labels, dtype=torch.float64)
    # Fixed class weights per relation type; relations are sparse.
    positive = label_t.sum(dim=0)
    negative = label_t.shape[0] - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 30.0)

    for outer in range(outer_iter):
        X0 = sparse_codes(D, Y, sparsity)
        support = np.abs(X0) > 1e-12
        x_t = torch.nn.Parameter(torch.tensor(X0, dtype=torch.float64))
        support_t = torch.tensor(support, dtype=torch.float64)
        d_t = torch.tensor(D, dtype=torch.float64)
        optimizer = torch.optim.Adam([x_t, relation_mats], lr=code_lr)

        final_content = final_relation = 0.0
        for _ in range(code_steps):
            reconstruction = d_t @ x_t
            content_loss = ((reconstruction - y_t) ** 2).sum(dim=0).mean()
            codes = x_t.T
            left = codes[pair_t[:, 0]]
            right = codes[pair_t[:, 1]]
            # score[pair, relation] = x_i^T M_relation x_j
            logits = torch.einsum("pk,rkl,pl->pr", left, relation_mats, right)
            relation_loss = F.binary_cross_entropy_with_logits(
                logits, label_t, pos_weight=pos_weight
            )
            loss = content_loss + relation_weight * relation_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x_t.mul_(support_t)
            final_content = float(content_loss.detach())
            final_relation = float(relation_loss.detach())

        X = x_t.detach().cpu().numpy()
        gram = X @ X.T + ridge * np.eye(n_atoms)
        D = (Y @ X.T) @ np.linalg.pinv(gram)
        for atom in range(n_atoms):
            norm = float(np.linalg.norm(D[:, atom]))
            if norm <= 1e-12:
                # Keep the previous atom direction when an optimized support
                # becomes degenerate.
                D[:, atom] = d_t[:, atom].cpu().numpy()
                norm = float(np.linalg.norm(D[:, atom]))
            D[:, atom] /= max(norm, 1e-12)
        deploy_X = sparse_codes(D, Y, sparsity)
        deploy_recon = float(
            np.linalg.norm(Y - D @ deploy_X) / (np.linalg.norm(Y) + 1e-12)
        )
        history.append(
            {
                "outer": float(outer),
                "content_loss": final_content,
                "relation_loss": final_relation,
                "deploy_reconstruction_relative_error": deploy_recon,
            }
        )

    return D, {
        "initial_ksvd": init_info,
        "relation_weight": float(relation_weight),
        "outer_iter": int(outer_iter),
        "code_steps": int(code_steps),
        "n_pairs": int(pairs.shape[0]),
        "n_relation_types": int(n_rel),
        "history": history,
    }
