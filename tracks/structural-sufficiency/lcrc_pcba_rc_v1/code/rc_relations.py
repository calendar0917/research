"""The differentiable relation-combination statistics (no message passing).

Everything here is a *graph-level* statistic of the per-atom role distribution
``R = softmax(Linear(h), -1)``.  Nothing is written back to an atom state, no
bond adjacency is propagated a second time, and there is exactly one
atom-encode per forward (enforced in ``rc_model`` by forward hooks).

Definitions implemented verbatim (see PROTOCOL.md):

    A_d[i,j] = A_d[j,i] = 1   iff atoms i != j of the same connected component
                              have shortest-bond-path distance in bucket d
    N_d      = sum(A_d)                       (= 2 x unordered pairs in bucket d)
    w_d      = N_d / max(N_all, 1)
    T_d      = 144 * (R^T A_d R) / max(N_d, 1)          (0 when N_d = 0)
    T_all    = 144 * (R^T A_all R) / max(N_all, 1)
    F_d      = sqrt(w_d) * T_d                          (REAL)
    F_d      = sqrt(w_d) * T_all if N_d > 0 else 0      (BLIND)
    x        = concat(triu_78(F_0..F_5), w_6, mean_a(R)_12, log1p(n_atoms))

Symmetry is used to halve the work: with ``M_d = A_d^upper @ R`` (only the
``i < j`` pairs), ``R^T A_d R = Z_d + Z_d^T`` where ``Z_d = R^T M_d``.  The
per-molecule reduction over atoms is a single ``[B, A] @ [A, 1020]`` matmul
(the "[B, A]" operand is the constant graph-indicator matrix of the batch), so
no Python loop ever runs over atom pairs.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

import rc_common as C

N_ROLES = C.N_ROLES
N_BUCKETS = C.N_BUCKETS
TRIU = C.TRIU


# ---------------------------------------------------------------------------
# exact inverse of the row-major strict upper-triangle numbering
# ---------------------------------------------------------------------------


def triangle_index(p: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(i, j)`` with ``i < j`` from the flat index ``p`` of an ``n``-row triangle.

    Row ``i`` of the strict upper triangle holds ``n-1-i`` entries, starting at
    ``O(i) = i*n - i*(i+1)/2``.  The closed form is evaluated in float64 and
    then corrected with the exact integer ``O(i)``, so the result is exact for
    every ``n <= 2**31`` (the float64 square root is only a search hint).
    """
    if p.numel() == 0:
        return p.to(torch.int64), p.to(torch.int64)
    pf = p.to(torch.float64)
    nf = n.to(torch.float64)
    m = 2.0 * nf - 1.0
    disc = (m * m - 8.0 * pf).clamp_min(0.0)
    i = torch.floor((m - torch.sqrt(disc)) * 0.5).to(torch.int64).clamp_(min=0)
    oi = i * n - (i * (i + 1)) // 2
    # one exact correction step in each direction is always sufficient
    i = torch.where(p < oi, i - 1, i).clamp_(min=0)
    oi = i * n - (i * (i + 1)) // 2
    row_len = n - 1 - i
    i = torch.where(p >= oi + row_len, i + 1, i)
    oi = i * n - (i * (i + 1)) // 2
    j = i + 1 + (p - oi)
    return i, j


def brute_triangle_index(p: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation (explicit per-molecule row scan, CPU only)."""
    ii = torch.zeros_like(p)
    jj = torch.zeros_like(p)
    for k in range(p.numel()):
        nn = int(n[k].item())
        pp = int(p[k].item())
        i = 0
        while pp >= nn - 1 - i:
            pp -= nn - 1 - i
            i += 1
        ii[k] = i
        jj[k] = i + 1 + pp
    return ii, jj


# ---------------------------------------------------------------------------
# batch geometry: which atoms does every stored pair connect
# ---------------------------------------------------------------------------


def pair_geometry(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Expand the packed per-molecule pair list into global atom index pairs."""
    bucket = batch["pair_bucket"].to(torch.int64)
    pair_counts = batch["pair_counts"].to(torch.int64)
    n_graphs = int(batch["n_graphs"])
    device = bucket.device
    total = int(bucket.numel())
    if int(pair_counts.sum().item()) != total:
        raise RuntimeError(f"pair_counts sum {int(pair_counts.sum())} != packed pairs {total}")
    atom_counts = batch["atom_counts"].to(torch.int64)
    atom_off = batch["atom_off"].to(torch.int64)
    pair_graph = torch.repeat_interleave(
        torch.arange(n_graphs, device=device, dtype=torch.int64), pair_counts
    )
    if total:
        offsets = torch.cumsum(pair_counts, 0) - pair_counts
        p_local = torch.arange(total, device=device, dtype=torch.int64) - offsets[pair_graph]
        n_local = atom_counts[pair_graph]
        i, j = triangle_index(p_local, n_local)
        base = atom_off[pair_graph]
        u = base + i
        v = base + j
    else:
        u = torch.zeros(0, device=device, dtype=torch.int64)
        v = torch.zeros(0, device=device, dtype=torch.int64)
    return {
        "bucket": bucket,
        "pair_graph": pair_graph,
        "u": u,
        "v": v,
        "valid": bucket < N_BUCKETS,
    }


# ---------------------------------------------------------------------------
# the statistics
# ---------------------------------------------------------------------------


def _triu_indices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = torch.triu_indices(N_ROLES, N_ROLES, offset=0, device=device)
    return rows, cols


def _to_full(mat: torch.Tensor) -> torch.Tensor:
    """Symmetrise ``Z + Z^T`` from the upper-triangle pair sum (already summed)."""
    return mat + mat.transpose(-1, -2)


def relation_input(
    R: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    pair_graph: torch.Tensor,
    bucket: torch.Tensor,
    atom_batch: torch.Tensor,
    atom_counts: torch.Tensor,
    n_graphs: int,
    *,
    kind: str,
    return_matrices: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build the 487-dim relation input ``x`` for a batch of molecules.

    ``u < v`` are global atom ids of unordered pairs of the *same* molecule,
    ``bucket`` is the per-pair bucket code (``>= N_BUCKETS`` = invalid/unreachable).
    """
    if kind not in ("real", "blind"):
        raise KeyError(kind)
    device = R.device
    dtype = R.dtype
    n_atoms_total = int(R.shape[0])
    n_graphs = int(n_graphs)
    valid = bucket < N_BUCKETS

    # ---- per-atom, per-bucket pair sums M_d = A_d^upper @ R ---------------
    M = []
    ordered_counts = torch.zeros((n_graphs, N_BUCKETS), device=device, dtype=dtype)
    for d in range(N_BUCKETS):
        sel = valid & (bucket == d)
        md = R.new_zeros((n_atoms_total, N_ROLES))
        if bool(sel.any()):
            md.index_add_(0, u[sel], R[v[sel]])
        M.append(md)
    if int(pair_graph.numel()):
        flat = pair_graph[valid] * N_BUCKETS + bucket[valid]
        counts = torch.bincount(flat, minlength=n_graphs * N_BUCKETS).to(dtype)
        ordered_counts = 2.0 * counts.reshape(n_graphs, N_BUCKETS)

    all_counts = ordered_counts.sum(dim=1, keepdim=True)
    w = ordered_counts / all_counts.clamp_min(1.0)

    # ---- one [B, A] @ [A, 1020] matmul does every per-molecule reduction --
    rows, cols = _triu_indices(device)

    def _outer(mat: torch.Tensor) -> torch.Tensor:
        return (R.unsqueeze(2) * mat.unsqueeze(1)).reshape(n_atoms_total, N_ROLES * N_ROLES)

    prod = torch.cat([_outer(m) for m in M], dim=1)  # [A, 6*144]
    # A_all = sum_d A_d, so the full-graph matrix is one extra reduction of the
    # same per-atom sums (exactly R^T A_all R).
    m_all = torch.stack(M, dim=0).sum(dim=0)
    prod = torch.cat([prod, _outer(m_all), R], dim=1)  # [A, 7*144 + 12]
    indicator = torch.zeros(
        (n_atoms_total, n_graphs), device=device, dtype=dtype
    ).index_put_(
        (torch.arange(n_atoms_total, device=device), atom_batch.to(torch.int64)),
        torch.ones(n_atoms_total, device=device, dtype=dtype),
    )
    reduced = indicator.transpose(0, 1) @ prod  # [B, 7*144 + 12]

    Z = reduced[:, : N_BUCKETS * N_ROLES * N_ROLES].reshape(n_graphs, N_BUCKETS, N_ROLES, N_ROLES)
    Z_all = reduced[:, N_BUCKETS * N_ROLES * N_ROLES : N_BUCKETS * N_ROLES * N_ROLES + N_ROLES * N_ROLES].reshape(
        n_graphs, N_ROLES, N_ROLES
    )
    role_sum = reduced[:, -N_ROLES:]
    role_mean = role_sum / atom_counts.to(dtype).clamp_min(1.0).unsqueeze(1)

    scale = 144.0
    T = scale * _to_full(Z) / ordered_counts.clamp_min(1.0).reshape(n_graphs, N_BUCKETS, 1, 1)
    T_all = scale * _to_full(Z_all) / all_counts.clamp_min(1.0).reshape(n_graphs, 1, 1)

    sqrt_w = torch.sqrt(w.clamp_min(1e-12)).reshape(n_graphs, N_BUCKETS, 1, 1)
    if kind == "real":
        F = sqrt_w * T
    else:
        present = (ordered_counts > 0).reshape(n_graphs, N_BUCKETS, 1, 1).to(dtype)
        F = sqrt_w * T_all.unsqueeze(1) * present

    features = [
        F[:, d][:, rows, cols] for d in range(N_BUCKETS)
    ]  # each [B, 78]
    x = torch.cat(
        features
        + [
            w,
            role_mean,
            torch.log1p(atom_counts.to(dtype)).unsqueeze(1),
        ],
        dim=1,
    )
    if x.shape[1] != C.X_DIM:
        raise RuntimeError(f"relation input width {x.shape[1]} != {C.X_DIM}")
    aux: dict[str, Any] = {
        "x": x,
        "w": w,
        "ordered_counts": ordered_counts,
        "role_mean": role_mean,
        "T_all": T_all,
        "role_entropy": float(
            (-(R.clamp_min(1e-12).log() * R).sum(dim=1)).mean().item()
        )
        if R.numel()
        else 0.0,
        "role_max_mean": float(R.max(dim=1).values.mean().item()) if R.numel() else 0.0,
        "n_pairs_valid": int(valid.sum().item()),
        "n_pairs_total": int(bucket.numel()),
        "n_atoms_without_pairs": int(
            (torch.bincount(u, minlength=n_atoms_total) == 0).sum().item()
        )
        if u.numel()
        else n_atoms_total,
    }
    if return_matrices:
        aux["T"] = T
        aux["F"] = F
    return x, aux


def relation_input_from_roles(
    R: torch.Tensor,
    *,
    n_graphs: int,
    atom_counts: torch.Tensor,
    pairs: list[tuple[int, int, int]],
    kind: str,
    return_matrices: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Test helper: build ``x`` from explicit roles and explicit (u, v, bucket)."""
    device = R.device
    atom_counts = torch.as_tensor(atom_counts, dtype=torch.int64, device=device)
    atom_batch = torch.repeat_interleave(
        torch.arange(int(n_graphs), device=device, dtype=torch.int64), atom_counts
    )
    if pairs:
        u = torch.tensor([p[0] for p in pairs], dtype=torch.int64, device=device)
        v = torch.tensor([p[1] for p in pairs], dtype=torch.int64, device=device)
        b = torch.tensor([p[2] for p in pairs], dtype=torch.int64, device=device)
        graph = atom_batch[u]
    else:
        u = torch.zeros(0, dtype=torch.int64, device=device)
        v = torch.zeros(0, dtype=torch.int64, device=device)
        b = torch.zeros(0, dtype=torch.int64, device=device)
        graph = torch.zeros(0, dtype=torch.int64, device=device)
    return relation_input(
        R, u, v, graph, b, atom_batch, atom_counts, int(n_graphs), kind=kind, return_matrices=return_matrices
    )


__all__ = [
    "N_BUCKETS",
    "N_ROLES",
    "TRIU",
    "brute_triangle_index",
    "pair_geometry",
    "relation_input",
    "relation_input_from_roles",
    "triangle_index",
]
