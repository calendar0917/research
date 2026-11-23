"""Exact finite-state graph energy, root-conditioned response and Mobius inversion.

Fixed energy model (shared by every molecule, root and word type; no per-word
parameters):

    E_r(S, sigma) = sum_{i in S} u_i[r, sigma_i] + sum_{(i,j) in E(S)} J_ij[r, sigma_i, sigma_j]
    logZ_r(S)     = logsumexp_sigma E_r(S, sigma)
    logZ_r(S; sigma_v = t) : the same sum with the root state fixed to t

    Phi_r,t(S, v) = logZ_r(S; sigma_v=t) - logZ_r(S \\ {v})

    kappa(S, v)   = Phi(S, v) - sum_{T proper subset of S, v in T} kappa(T, v)

The support size is at most 4, so every log Z is an exact enumeration over at
most 16 state assignments (no sampling, no message passing, no approximation).
``S \\ {v}`` may be disconnected; the energy of that set is still well defined
and is evaluated by the same exact enumeration with the root's unary and
incident pairwise terms switched off, restricted to root-state 0 so that each
state of the remaining atoms is counted exactly once.

Largest possible |E| for a support is 4 * umax + 6 * Jmax.  The unary potential is
bounded by 2 and the pair potential by 2 (bond-only) or 4 (chemistry-conditioned,
``|J_base + delta| <= 2 + 2``), so ``exp`` can never overflow.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import torch

from rice_common import MAX_WORD_SIZE


def state_bits(device: Any = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    idx = torch.arange(1 << MAX_WORD_SIZE, device=device)
    return torch.stack([(idx >> s) & 1 for s in range(MAX_WORD_SIZE)], dim=1).to(dtype)


def couple_pairs(u: torch.Tensor, j_base: torch.Tensor,
                 bond_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Chemistry-conditioned symmetric binary pair potential.

    ``u`` [n_atoms, R, 2], ``j_base`` [n_bonds, R, 3] = (J00, J01/J10, J11),
    ``bond_index`` [n_bonds, 2] with the physical bond endpoints.

        h_i = tanh((u_i1 - u_i0) / 2)                     # [n_atoms, R]
        k_e = (J00 + J11 - 2 J01) / 4                     # [n_bonds, R]
        delta_e = k_e * h_endpoint0 * h_endpoint1         # [n_bonds, R]

        J_real = (J00 + delta, J01 - delta, J11 + delta)

    The correction is symmetric in the two endpoints and is an even function of
    the latent-state orientation (flipping both states of a channel flips ``h``
    twice).  Nothing is detached, so the correction carries gradients into both
    the unary and the bond MLP.  With ``|J_base| <= 2`` and ``|h| <= 1`` we have
    ``|k| <= 2``, ``|delta| <= 2`` and therefore ``|J_real| <= 4``.
    """
    if int(j_base.shape[0]) == 0:
        return j_base, j_base.new_zeros(j_base.shape[:-1])
    if int(bond_index.shape[0]) != int(j_base.shape[0]):
        raise ValueError("bond_index and pair potential disagree on the bond count")
    h = torch.tanh((u[..., 1] - u[..., 0]) / 2.0)          # [n_atoms, R]
    k = (j_base[..., 0] + j_base[..., 2] - 2.0 * j_base[..., 1]) / 4.0
    i = bond_index[:, 0].to(torch.long)
    j = bond_index[:, 1].to(torch.long)
    delta = k * h[i] * h[j]                                # [n_bonds, R]
    j_real = torch.stack(
        [j_base[..., 0] + delta,
         j_base[..., 1] - delta,
         j_base[..., 2] + delta],
        dim=-1,
    )
    return j_real, delta


def pair_matrices(j: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [..., 2, 2] symmetric pair matrix (analysis only)."""
    zero = j[..., 0]
    one = j[..., 1]
    two = j[..., 2]
    return torch.stack([torch.stack([zero, one], dim=-1),
                        torch.stack([one, two], dim=-1)], dim=-2)


def _energy(up: torch.Tensor, jp: torch.Tensor, slot_i: torch.Tensor,
            slot_j: torch.Tensor, term_ok: torch.Tensor,
            bits: torch.Tensor) -> torch.Tensor:
    """E[support, state, channel] for every state of the padded 4-slot support."""
    n_states = int(bits.shape[0])
    bf = bits.unsqueeze(0)                                   # [1,16,4]
    one_minus = 1.0 - bf
    unary = (up[..., 0].unsqueeze(1) * one_minus[:, :, :, None]
             + up[..., 1].unsqueeze(1) * bf[:, :, :, None])   # [C,16,4,R]
    energy = unary.sum(dim=2)
    bi = bits[:, slot_i].permute(1, 0, 2)                     # [C,16,6]
    bj = bits[:, slot_j].permute(1, 0, 2)
    w00 = (1.0 - bi) * (1.0 - bj)
    w11 = bi * bj
    w01 = bi + bj - 2.0 * bi * bj
    ok = term_ok.unsqueeze(1)
    for weight, term in ((w00, 0), (w01, 1), (w11, 2)):
        energy = energy + (jp[..., term].unsqueeze(1) * (weight * ok)[..., None]).sum(dim=2)
    if int(energy.shape[1]) != n_states:
        raise ValueError("state axis mismatch")
    return energy


def compute_phi(u: torch.Tensor, j: torch.Tensor, atoms: torch.Tensor,
                bonds: torch.Tensor, bslot: torch.Tensor,
                chunk: int = 4096) -> torch.Tensor:
    """``Phi`` for every stored support, flat channel layout ``r * 2 + t``.

    ``u`` [n_atoms, R, 2]; ``j`` [n_bonds, R, 3] = (J00, J01/J10, J11);
    ``atoms`` [n_words, 4] with the root in slot 0 (-1 padded);
    ``bonds`` [n_words, 6] global bond ids (-1 padded);
    ``bslot`` [n_words, 6, 2] slot pair of each internal bond (-1 padded).
    """
    n = int(atoms.shape[0])
    channels = int(u.shape[1])
    out = u.new_zeros((n, channels * 2))
    if n == 0:
        return out
    device = u.device
    bits = state_bits(device, u.dtype)
    n_states = int(bits.shape[0])
    state_ids = torch.arange(n_states, device=device)
    bit0_zero = (state_ids % 2 == 0).unsqueeze(0)              # [1,16]
    for start in range(0, n, int(chunk)):
        stop = min(n, start + int(chunk))
        a = atoms[start:stop]
        b = bonds[start:stop]
        s = bslot[start:stop]
        size = (a >= 0).sum(dim=1)
        if int(size.min()) < 1:
            raise ValueError("empty support")
        slot_i = s[:, :, 0].to(torch.long).clamp(min=0)
        slot_j = s[:, :, 1].to(torch.long).clamp(min=0)
        valid_term = (b >= 0) & (s[:, :, 0] >= 0)
        up = u[a.clamp(min=0)] * (a >= 0).unsqueeze(-1).unsqueeze(-1).to(u.dtype)
        jp = j[b.clamp(min=0)] * valid_term.unsqueeze(-1).unsqueeze(-1).to(u.dtype)
        state_ok = state_ids.unsqueeze(0) < (1 << size).unsqueeze(1).to(torch.long)
        energy = _energy(up, jp, slot_i, slot_j, valid_term.to(u.dtype), bits)

        mask0 = state_ok & bit0_zero
        mask1 = state_ok & ~bit0_zero
        logz0 = torch.logsumexp(energy.masked_fill(~mask0.unsqueeze(-1), float("-inf")), dim=1)
        logz1 = torch.logsumexp(energy.masked_fill(~mask1.unsqueeze(-1), float("-inf")), dim=1)

        # S \ {root}: root unary and every root-incident bond switched off,
        # restricted to root-state 0 (each remaining state counted once)
        keeps_root = ((slot_i != 0) & (slot_j != 0) & valid_term)
        up_rest = up.clone()
        up_rest[:, 0] = 0.0
        jp_rest = jp * keeps_root.unsqueeze(-1).unsqueeze(-1).to(u.dtype)
        energy_rest = _energy(up_rest, jp_rest, slot_i, slot_j,
                              keeps_root.to(u.dtype), bits)
        logz_rest = torch.logsumexp(energy_rest.masked_fill(~mask0.unsqueeze(-1), float("-inf")), dim=1)

        phi = torch.stack([logz0 - logz_rest, logz1 - logz_rest], dim=-1)
        out[start:stop] = phi.reshape(-1, channels * 2)
    return out


def mobius(u_flat: torch.Tensor,
           phi: Mapping[int, torch.Tensor],
           atoms: Mapping[int, torch.Tensor],
           sub2: Mapping[int, torch.Tensor],
           sub3_4: torch.Tensor) -> dict[int, torch.Tensor]:
    """Root-anchored Mobius inversion, sizes 1 -> 4 (increasing size)."""
    kappa: dict[int, torch.Tensor] = {1: u_flat}
    for size in (2, 3, 4):
        p = phi[size]
        if int(p.shape[0]) == 0:
            kappa[size] = p
            continue
        root = atoms[size][:, 0].to(torch.long)
        total = p - kappa[1][root]
        if size >= 3:
            total = total - _gather_sum(kappa[2], sub2[size])
        if size >= 4:
            total = total - _gather_sum(kappa[3], sub3_4)
        kappa[size] = total
    return kappa


def _gather_sum(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if int(idx.shape[0]) == 0:
        return values.new_zeros((0, values.shape[1]))
    ok = (idx >= 0).to(values.dtype)
    picked = values[idx.clamp(min=0)] * ok.unsqueeze(-1)
    return picked.sum(dim=1)


def seg_stats(values: torch.Tensor, seg: torch.Tensor, n: int,
              dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = values.new_zeros((int(n), int(dim)))
    if values.numel():
        out.index_add_(0, seg, values)
    sq = values.new_zeros((int(n), int(dim)))
    if values.numel():
        sq.index_add_(0, seg, values * values)
    count = values.new_zeros(int(n))
    if values.numel():
        count.index_add_(0, seg, torch.ones(values.shape[0], device=values.device,
                                           dtype=values.dtype))
    denom = count.clamp(min=1.0)
    mean = out / denom[:, None]
    var = (sq / denom[:, None] - mean * mean).clamp(min=0.0)
    std = torch.sqrt(var + 1e-12)
    std = torch.where((count > 1)[:, None], std, torch.zeros_like(std))
    return out, mean, std, count


# ---------------------------------------------------------------------------
# independent brute-force reference (float64, explicit python loops)
# ---------------------------------------------------------------------------

def _energy_np(u: Any, j: Any, support: Sequence[int], sigma: Sequence[int],
               bond_index: Mapping[tuple[int, int], int]) -> Any:
    import numpy as np

    e = np.zeros(u.shape[1], np.float64)
    for i in support:
        e = e + u[int(i), :, int(sigma[i])]
    for (a, b) in itertools.combinations(sorted(support), 2):
        bid = bond_index.get((int(a), int(b)))
        if bid is not None:
            sa = int(sigma[a])
            sb = int(sigma[b])
            term = 2 if (sa == 1 and sb == 1) else (0 if (sa == 0 and sb == 0) else 1)
            e = e + j[bid][:, term]
    return e


def _logsumexp_np(values: Any) -> Any:
    import numpy as np

    m = values.max(axis=0)
    return m + np.log(np.exp(values - m).sum(axis=0))


def brute_force_phi(u: Any, j: Any, n_atoms: int,
                    bond_list: Sequence[tuple[int, int, int]],
                    root: int, support: Sequence[int]) -> Any:
    import numpy as np

    bond_index = {(min(a, b), max(a, b)): bid for a, b, bid in bond_list}
    support = sorted(int(x) for x in support)
    others = [x for x in support if x != int(root)]
    all_vals = {0: [], 1: []}
    for combo in itertools.product((0, 1), repeat=len(others)):
        sigma = {int(root): 0}
        for a, c in zip(others, combo):
            sigma[a] = c
        for t in (0, 1):
            sigma[int(root)] = t
            all_vals[t].append(_energy_np(u, j, support, sigma, bond_index))
    logz_t = [_logsumexp_np(np.stack(all_vals[t])) if all_vals[t] else None
              for t in (0, 1)]
    rest = [x for x in support if x != int(root)]
    if rest:
        vals = []
        for combo in itertools.product((0, 1), repeat=len(rest)):
            sigma = {a: c for a, c in zip(rest, combo)}
            vals.append(_energy_np(u, j, rest, sigma, bond_index))
        logz_rest = _logsumexp_np(np.stack(vals))
    else:
        logz_rest = np.zeros(u.shape[1], np.float64)
    return np.stack([logz_t[0] - logz_rest, logz_t[1] - logz_rest], axis=-1)


def brute_force_kappa(u: Any, j: Any, n_atoms: int,
                      bond_list: Sequence[tuple[int, int, int]], root: int,
                      radius_atoms: Sequence[int]) -> dict[str, Any]:
    """Every subset (connected or not) of the radius-2 atom set, size 1..4."""
    phi: dict[frozenset, Any] = {}
    kappa: dict[frozenset, Any] = {}
    others = [int(x) for x in radius_atoms if int(x) != int(root)]
    for size in range(1, MAX_WORD_SIZE + 1):
        for combo in itertools.combinations(others, size - 1):
            key = frozenset((int(root),) + combo)
            phi[key] = brute_force_phi(u, j, n_atoms, bond_list, int(root), sorted(key))
    ordered = sorted(phi, key=lambda s: (len(s), sorted(s)))
    for key in ordered:
        total = phi[key].copy()
        for sub_size in range(1, len(key)):
            for combo in itertools.combinations(sorted(key - {int(root)}), sub_size - 1):
                sub = frozenset((int(root),) + combo)
                total = total - kappa[sub]
        kappa[key] = total
    return {"phi": phi, "kappa": kappa}


def reference_is_connected(support: Sequence[int],
                           bond_list: Sequence[tuple[int, int, int]]) -> bool:
    support = [int(x) for x in support]
    present = set(support)
    adj = {a: set() for a in support}
    for a, b, _ in bond_list:
        if int(a) in present and int(b) in present:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
    seen = {support[0]}
    stack = [support[0]]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == len(present)