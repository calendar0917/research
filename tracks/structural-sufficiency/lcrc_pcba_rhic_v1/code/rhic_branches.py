"""Branch construction and the two-level grouping statistics of LCRC-RHIC.

A **branch** is a directed physical bond ``(i, j)``: ``i`` is the root, ``j``
the first-hop middle atom.  ``(i, j)`` and ``(j, i)`` are different branches
because they belong to different roots.  Every physical bond ``e`` (stored as
``bond_index[e] = (u, v)``) therefore owns two directed branches

    branch_id = 2 * e      : u -> v      (root = u, middle = v)
    branch_id = 2 * e + 1  : v -> u      (root = v, middle = u)

and a path ``(i, j, k)`` is assigned to the branch of its first bond.  Nothing
here is cached: the branch id is derived at run time from ``cpc_bond_ij``, the
root and ``bond_index``.  Flipping the *storage* direction of a physical bond
only swaps the two labels ``2e`` and ``2e+1``; the grouping of paths is
unchanged, which is why the model output is invariant to it.

Statistics (frozen arithmetic, see ``RHIC`` in ``rhic_common``):

    n_b        = |P_b|                       number of paths in branch b
    S_b        = sum_p v_p                   [D]
    T_b        = sum_p v_p * v_p             [D]
    mu_b       = S_b / n_b
    q_b        = (S_b*S_b - T_b) / (n_b (n_b-1))          if n_b >= 2 else 0

The model evaluates ``q_b`` through the algebraically identical, numerically
stable form

    D_b        = sum_p (v_p - mu_b)^2
    q_b        = mu_b^2 - D_b / (n_b (n_b-1))             if n_b >= 2 else 0

which avoids forming ``S_b*S_b`` and subtracting the much larger ``T_b``.  The
identity is checked against the naive form in the audits.

The second level uses the same arithmetic on ``branch_value`` over the active
branches of one root.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# branch ids and the compact active-branch layout (torch)
# ---------------------------------------------------------------------------


def branch_ids(
    path_atom: torch.Tensor,
    middle: torch.Tensor,
    bond_ij: torch.Tensor,
    bond_index: torch.Tensor,
    *,
    verify: bool = False,
) -> torch.Tensor:
    """``2 * bond_ij + dir`` for every path, with ``dir`` given by the root side.

    ``bond_index`` must be the batch-local physical bond table (``u < v`` as the
    frozen collate writes it).  The returned ids are in ``[0, 2 * n_bonds)``.
    """
    n = int(path_atom.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=path_atom.device)
    pair = bond_index.index_select(0, bond_ij)
    u = pair[:, 0]
    v = pair[:, 1]
    forward = u == path_atom
    if verify:
        backward = v == path_atom
        if not bool((forward ^ backward).all()):
            raise RuntimeError("path root is neither endpoint of its first bond")
        mid_ok = torch.where(forward, v == middle, u == middle)
        if not bool(mid_ok.all()):
            raise RuntimeError("path middle atom is not the other endpoint of its first bond")
    return 2 * bond_ij + (~forward).to(torch.long)


def branch_group_layout(
    branch_id: torch.Tensor, n_branches: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Counts / active mask / compact index per path / active branch ids.

    ``compact[inv]`` indexes a dense ``[n_active, ...]`` table.  Inactive
    branches are never materialised: ``n_b = 0`` groups cannot contribute.
    """
    counts = torch.bincount(branch_id, minlength=int(n_branches))
    active = counts > 0
    compact = torch.cumsum(active.to(torch.long), 0) - 1
    inv = compact.index_select(0, branch_id) if branch_id.numel() else branch_id
    active_ids = active.nonzero(as_tuple=False).reshape(-1)
    return counts, active, inv, active_ids


def branch_endpoints(active_ids: torch.Tensor, bond_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(root, middle, physical_bond)`` of every active branch, from the bond table."""
    phys = active_ids >> 1
    direction = active_ids & 1
    pair = bond_index.index_select(0, phys)
    root = pair.gather(1, direction.unsqueeze(1)).reshape(-1)
    middle = pair.gather(1, (1 - direction).unsqueeze(1)).reshape(-1)
    return root, middle, phys


# ---------------------------------------------------------------------------
# the two-level statistics (torch, FP32, batched scatter)
# ---------------------------------------------------------------------------


def group_mean_dispersion(
    values: torch.Tensor, inv: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(sum, mean, dispersion)`` per group for ``values[inv]``.

    ``dispersion = sum (v - mean)^2`` is accumulated directly, so the pairwise
    mean below never subtracts two large nearly equal numbers.
    """
    n_groups = int(counts.numel())
    width = int(values.shape[1])
    sums = values.new_zeros((n_groups, width)).index_add_(0, inv, values)
    n = counts.to(values.dtype)
    mean = sums / n.clamp(min=1.0).unsqueeze(1)
    dev = values - mean.index_select(0, inv)
    dispersion = values.new_zeros((n_groups, width)).index_add_(0, inv, dev * dev)
    return sums, mean, dispersion


def pairwise_cooccurrence(mean: torch.Tensor, dispersion: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Per-channel mean of ``v_p * v_q`` over **distinct** pairs, zero for n < 2.

    ``q = mu^2 - D / (n (n-1))`` with ``D = sum_p (v_p - mu)^2`` is exactly
    ``(S^2 - T) / (n (n-1))`` because ``S^2 - T = n(n-1) mu^2 - D``; the
    self-products are therefore excluded by construction.
    """
    n = counts.to(mean.dtype)
    denom = (n * (n - 1.0)).clamp(min=1.0).unsqueeze(1)
    has_pair = (n >= 2.0).unsqueeze(1)
    q = mean * mean - dispersion / denom
    return torch.where(has_pair, q, torch.zeros_like(mean))


# ---------------------------------------------------------------------------
# independent NumPy reference (audits only; explicit loops, float64)
# ---------------------------------------------------------------------------


def reference_two_level_stats(
    values: np.ndarray,
    group_key: Sequence[Any],
    n_groups: int | None = None,
    *,
    group_of_row: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Explicit float64 reference for one level of grouping.

    ``values`` is ``[M, D]``, ``group_key[p]`` is the group key of row ``p``
    (for the branch level a ``(root, middle)`` tuple, for the root level the
    root id of each *branch*), and ``group_of_row`` optionally maps each row to
    a parent group (used at the root level when the rows are branches).

    ``T`` is formed explicitly as ``sum_p v_p^2`` and ``q`` as
    ``(S^2 - T) / (n (n-1))``: this is the naive formula the model must match.
    """
    v = np.asarray(values, dtype=np.float64)
    m, dim = v.shape
    keys = list(group_key)
    if len(keys) != m:
        raise ValueError("group_key length mismatch")
    uniq: list[Any] = []
    index: dict[Any, int] = {}
    inv = np.empty(m, dtype=np.int64)
    for p, key in enumerate(keys):
        if key not in index:
            index[key] = len(uniq)
            uniq.append(key)
        inv[p] = index[key]
    g = len(uniq) if n_groups is None else int(n_groups)
    S = np.zeros((g, dim), dtype=np.float64)
    T = np.zeros((g, dim), dtype=np.float64)
    cnt = np.zeros(g, dtype=np.int64)
    for p in range(m):
        S[inv[p]] += v[p]
        T[inv[p]] += v[p] * v[p]
        cnt[inv[p]] += 1
    mu = np.zeros_like(S)
    q = np.zeros_like(S)
    for i in range(g):
        if cnt[i] >= 1:
            mu[i] = S[i] / cnt[i]
        if cnt[i] >= 2:
            q[i] = (S[i] * S[i] - T[i]) / (cnt[i] * (cnt[i] - 1))
    return {
        "groups": uniq,
        "inv": inv,
        "counts": cnt,
        "S": S,
        "T": T,
        "mu": mu,
        "q": q,
    }


def reference_root_level(
    branch_values: np.ndarray, branch_root_of_group: Sequence[int], n_atoms: int
) -> dict[str, Any]:
    """Explicit float64 reference for the second (branch -> root) level."""
    return reference_two_level_stats(
        branch_values,
        [int(x) for x in branch_root_of_group],
        n_groups=int(n_atoms),
    )


# ---------------------------------------------------------------------------
# the diagnostic within-root branch-assignment shuffle (numpy, CPU)
# ---------------------------------------------------------------------------


def shuffle_branch_assignment(
    path_atom: torch.Tensor, branch_id: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute branch *membership* among the paths of the same root.

    Every root's multiset of branch ids is preserved (so ``n_b`` totals, the
    branch count ``d_i`` and the path count are unchanged) and the multiset of
    path tokens is untouched, but which path contributes to which branch is
    randomised.  Roots are independent: no path leaves its own root or
    molecule, and different roots never exchange content.  Diagnostic only.
    """
    n = int(path_atom.numel())
    if n == 0:
        return branch_id, torch.zeros(0, dtype=torch.long)
    atom = path_atom.detach().cpu().numpy().astype(np.int64)
    bid = branch_id.detach().cpu().numpy().astype(np.int64)
    order = np.argsort(atom, kind="stable")
    atoms_sorted = atom[order]
    bounds = np.flatnonzero(np.diff(atoms_sorted) != 0) + 1
    starts = np.concatenate([[0], bounds]).astype(np.int64)
    ends = np.concatenate([bounds, [n]]).astype(np.int64)
    rng = np.random.default_rng(int(seed))
    permuted_sorted = bid[order].copy()
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s > 1:
            permuted_sorted[s:e] = permuted_sorted[s:e][rng.permutation(e - s)]
    out = np.empty_like(bid)
    out[order] = permuted_sorted
    perm = np.empty(n, dtype=np.int64)
    perm[order] = np.arange(n, dtype=np.int64)
    return (
        torch.as_tensor(out, dtype=branch_id.dtype, device=branch_id.device),
        torch.as_tensor(perm, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# structural coverage from the frozen path cache (labels never read)
# ---------------------------------------------------------------------------


def per_atom_branch_structure(cpc: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Path/branch summaries of every cached atom, from the frozen CPC arrays.

    * ``path_count``  -- rooted two-bond paths of the atom (``deg(mid)-1`` summed)
    * ``branch_count``-- first-hop branches with at least one path
    * ``max_paths_in_branch`` -- largest ``n_b`` of the atom
    * ``multi_path_branches`` -- number of branches with ``n_b >= 2``

    The cache enumerates root-major with the middle atoms grouped, so a branch
    is a contiguous run of equal ``cpc_middle`` inside one atom's slice.
    """
    cpc_ptr = np.asarray(cpc["cpc_ptr"], dtype=np.int64)
    middle = np.asarray(cpc["cpc_middle"], dtype=np.int64)
    n_atoms = int(cpc_ptr.size - 1)
    total = int(cpc_ptr[-1])
    counts = np.diff(cpc_ptr).astype(np.int64)
    if total == 0:
        zeros = np.zeros(n_atoms, dtype=np.int64)
        return {
            "path_count": zeros,
            "branch_count": zeros.copy(),
            "max_paths_in_branch": zeros.copy(),
            "multi_path_branches": zeros.copy(),
        }
    change = np.empty(total, dtype=bool)
    change[0] = True
    change[1:] = middle[1:] != middle[:-1]
    # a new atom always starts a new branch, even when its middle repeats
    starts_nonzero = cpc_ptr[:-1][counts > 0]
    change[starts_nonzero] = True
    branch_start = np.flatnonzero(change).astype(np.int64)
    branch_len = np.diff(np.append(branch_start, total)).astype(np.int64)
    branch_atom = np.searchsorted(cpc_ptr[:-1], branch_start, side="right") - 1
    branch_count = np.zeros(n_atoms, dtype=np.int64)
    multi = np.zeros(n_atoms, dtype=np.int64)
    max_len = np.zeros(n_atoms, dtype=np.int64)
    if branch_start.size:
        # branch_atom is non-decreasing -> reduceat over first occurrences
        first = np.concatenate([[0], np.flatnonzero(np.diff(branch_atom) != 0) + 1]).astype(np.int64)
        owners = branch_atom[first]
        branch_count[owners] = np.diff(np.append(first, branch_start.size))
        max_len[owners] = np.maximum.reduceat(branch_len, first)
        multi[owners] = np.add.reduceat((branch_len >= 2).astype(np.int64), first)
    return {
        "path_count": counts,
        "branch_count": branch_count,
        "max_paths_in_branch": max_len,
        "multi_path_branches": multi,
    }


def coverage_summary(structure: Mapping[str, np.ndarray], atom_mask: np.ndarray) -> dict[str, Any]:
    """Aggregate the per-atom structure into the pre-registered coverage table."""
    mask = np.asarray(atom_mask, dtype=bool)
    pc = np.asarray(structure["path_count"], dtype=np.int64)[mask]
    bc = np.asarray(structure["branch_count"], dtype=np.int64)[mask]
    mb = np.asarray(structure["max_paths_in_branch"], dtype=np.int64)[mask]
    mp = np.asarray(structure["multi_path_branches"], dtype=np.int64)[mask]
    n = int(pc.size)

    def frac(x: np.ndarray) -> float:
        return float(x.mean()) if n else 0.0

    def q(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p)) if n else 0.0

    return {
        "atoms": n,
        "path_count": {
            "mean": float(pc.mean()) if n else 0.0,
            "p95": q(pc, 95),
            "max": int(pc.max()) if n else 0,
            "zero_fraction": frac(pc == 0),
        },
        "branch_count": {
            "mean": float(bc.mean()) if n else 0.0,
            "p95": q(bc, 95),
            "max": int(bc.max()) if n else 0,
            "zero_fraction": frac(bc == 0),
        },
        "root_with_at_least_two_paths": frac(pc >= 2),
        "root_with_at_least_two_branches": frac(bc >= 2),
        "root_with_multi_path_branch": frac(mp >= 1),
        "root_with_ge2_multi_path_branches": frac(mp >= 2),
        "root_with_at_least_two_paths_or_branches": frac((pc >= 2) | (bc >= 2)),
        "branches": int(bc.sum()),
        "branches_with_at_least_two_paths": int(mp.sum()),
        "branch_q_structural_opportunity": float(mp.sum() / bc.sum()) if int(bc.sum()) else 0.0,
        "root_q_structural_opportunity": frac(mp >= 1),
        "root_cross_structural_opportunity": frac(bc >= 2),
        "max_paths_in_branch": {
            "mean": float(mb.mean()) if n else 0.0,
            "p95": q(mb, 95),
            "max": int(mb.max()) if n else 0,
        },
    }


__all__ = [
    "branch_endpoints",
    "branch_group_layout",
    "branch_ids",
    "coverage_summary",
    "group_mean_dispersion",
    "pairwise_cooccurrence",
    "per_atom_branch_structure",
    "reference_root_level",
    "reference_two_level_stats",
    "shuffle_branch_assignment",
]
