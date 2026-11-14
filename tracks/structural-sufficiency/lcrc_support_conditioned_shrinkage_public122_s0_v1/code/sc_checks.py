"""Identity / intervention checks that do not need a trained model.

Three families live here, all of them *forward-only*:

``residual_linear_map``
    Verifies the algebra the protocol claims: the residual table
    ``R_{v,b}(c,e) = N_{v,b}(c,e) - n_c n_e / n_b`` maps **linearly** onto
    ``J_full - J_marg`` (bias cancels exactly).  This is *not* a claim that R is a
    covariance or a calibrated uncertainty.

``permute_atoms``
    Re-numbers the atoms of a batch (and the atom-indexed fields) so a test can
    prove that nothing depends on atom order / ids: the same molecule must give
    the same per-root alpha, the same J block and the same prediction.

``joint_endpoints``
    ``a = 1`` -> frozen FULL operator, ``a = 0`` -> the ``marginal_bt2`` operator,
    with the *other* local blocks unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import sc_common as C


# ---------------------------------------------------------------------------
# the R -> (J_full - J_marg) linear identity
# ---------------------------------------------------------------------------


@torch.no_grad()
def residual_linear_map(local, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Per-root ``sum_b W( sum_{c,e} R_{v,b}(c,e) U_c (.) V_e )`` in float64.

    Compared against the operator difference ``J_full - J_marg`` of the very same
    module.  Both are computed on the small occupied ``c x e`` support; no dense
    ``n_path_codes x n_nbr_codes`` matrix is ever allocated.
    """
    path_atom = batch["path_atom"].cpu().numpy()
    cat = batch["path_cat"].cpu().numpy()
    end = batch["path_end63"].cpu().numpy()
    n_atoms = int(batch["atom_feat"].shape[0])
    rank = int(local.rank)

    U = local.path_u.weight.detach().double().cpu().numpy()
    V = local.path_v.weight.detach().double().cpu().numpy()
    W = local.joint_proj.weight.detach().double().cpu().numpy()

    out = np.zeros((n_atoms, rank), dtype=np.float64)
    for v in range(n_atoms):
        sel = path_atom == v
        if not np.any(sel):
            continue
        cv, ev = cat[sel].astype(np.int64), end[sel].astype(np.int64)
        bond = cv % C.N_BOND_TYPES
        acc = np.zeros(rank, dtype=np.float64)
        for g in np.unique(bond):
            m = bond == g
            cg, eg = cv[m], ev[m]
            n_b = int(cg.size)
            uc, ue = np.unique(cg), np.unique(eg)
            ci = np.searchsorted(uc, cg)
            ei = np.searchsorted(ue, eg)
            N = np.zeros((uc.size, ue.size), dtype=np.float64)
            np.add.at(N, (ci, ei), 1.0)
            n_c = N.sum(axis=1)
            n_e = N.sum(axis=0)
            R = N - np.outer(n_c, n_e) / float(n_b)      # full small Cartesian product
            # M = sum_{(c,e)} R(c,e) * (U_c (.) V_e)   -- element-wise per channel
            prod = (U[uc][:, None, :] * V[ue][None, :, :])   # (nc, ne, rank)
            M = (R[:, :, None] * prod).sum(axis=(0, 1))
            acc += W @ M                       # linear map, bias excluded
        out[v] = acc
    probe = batch["atom_feat"].new_zeros((n_atoms, rank))
    j_full, j_marg = local.joint_parts(batch, probe)
    diff = (j_full - j_marg).double().cpu().numpy().reshape(n_atoms, rank)
    err = float(np.abs(diff - out).max())
    scale = float(max(np.abs(diff).max(), 1e-12))
    return {"max_abs_error": err, "relative_error": err / scale, "rank": rank,
            "n_atoms": n_atoms, "n_roots_with_effect": int(
                (np.linalg.norm(diff, axis=1) > 1e-9).sum())}


# ---------------------------------------------------------------------------
# atom re-numbering
# ---------------------------------------------------------------------------


def permute_atoms(batch: Mapping[str, Any], order: Sequence[int]) -> dict[str, Any]:
    """Return a batch whose atoms of a **single-molecule** batch are re-numbered.

    ``order`` is the new atom order (a permutation of ``0..n_atoms-1``); every
    atom-indexed field is remapped, path/bond/nbr *rows* keep their content.
    """
    n_atoms = int(batch["atom_feat"].shape[0])
    order = np.asarray(list(order), dtype=np.int64)
    if order.size != n_atoms or sorted(order.tolist()) != list(range(n_atoms)):
        raise ValueError("order must be a permutation of the atom rows")
    inv = np.empty(n_atoms, dtype=np.int64)
    inv[order] = np.arange(n_atoms, dtype=np.int64)

    out = dict(batch)
    for key in ("atom_feat", "atom_type", "atom_scalar", "atom_batch", "root_alpha"):
        if key in batch and torch.is_tensor(batch[key]) and batch[key].shape[0] == n_atoms:
            out[key] = batch[key][torch.as_tensor(order)]
    if "bond_index" in batch:
        bi = batch["bond_index"].cpu().numpy()
        if bi.size:
            out["bond_index"] = torch.as_tensor(inv[bi], dtype=batch["bond_index"].dtype)
    for key in ("nbr_atom", "path_atom", "path_end"):
        if key in batch:
            arr = batch[key].cpu().numpy()
            if arr.size:
                out[key] = torch.as_tensor(inv[arr], dtype=batch[key].dtype)
    return out


# ---------------------------------------------------------------------------
# endpoint fixtures
# ---------------------------------------------------------------------------


@torch.no_grad()
def joint_endpoints(model, batch: Mapping[str, Any]) -> dict[str, Any]:
    """``a=1`` vs FULL and ``a=0`` vs the marginal (B5) operator, plus block identity."""
    was = model.training
    model.eval()
    try:
        n_atoms = int(batch["atom_feat"].shape[0])
        b1 = dict(batch, root_alpha=torch.ones(n_atoms, dtype=torch.float32))
        b0 = dict(batch, root_alpha=torch.zeros(n_atoms, dtype=torch.float32))
        y1 = model(b1).detach()
        y0 = model(b0).detach()
        out: dict[str, Any] = {
            "y_a1": y1.cpu().numpy(), "y_a0": y0.cpu().numpy(),
            "n_atoms": n_atoms,
        }
        a_atoms, _b, scal = model._encode(batch)
        j_full, j_marg = model.local.joint_parts(batch, a_atoms)
        out["joint_full"] = j_full.detach().cpu().numpy()
        out["joint_marg"] = j_marg.detach().cpu().numpy()
        return out
    finally:
        model.train(was)


__all__ = ["residual_linear_map", "permute_atoms", "joint_endpoints"]
