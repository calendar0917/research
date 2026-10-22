"""Content encoders for ``zinc_shared_factor_v1`` (arms D, T, N).

All three arms are the *same* P3 carrier (``ad2_models.P3NullModel`` encoders,
Null grouping, ``group_composer``, ``reader``, ``global33``) with the *same*
198 -> 128 bias-free zero-initialised ``W_s`` port, the *same* ``bf_data``
S3/S4/F4 objects and the *same* per-root/per-family occurrence aggregation.

``D`` (dense control, imported verbatim)
    ``zinc_local_factorization_v1``'s ``ContentBranch("D")``:
    ``E_atom[28,8]`` + ``E_pair[4,4]`` + content MLP 98 -> 80 -> 32, plus the
    exact root-preserving permutation average (6/24 rows).

``T`` (shared atomic state, exact contraction)
    One state variable per real atom, ``R = 4`` states and ``C = 16`` channels:

        q_c(s|a)        = softmax_s A[a,c,s]                       A[28,C,R]
        K_b,c(s,t)      = exp(tanh(S_b,c(s,t)))                    S symmetric,
                          upper triangle incl. diagonal -> 3*C*R*(R+1)/2
        T_H,c(s)        = sum_{s_i, i != r} prod_i q_c(s_i|a_i)
                                         prod_{(i,j) in E} K_bij,c(s_i,s_j)

    The sum runs over ``R**(n-1)`` assignments (64 for 4-node, 256 for 5-node
    objects) in log space; it is an exact contraction on the small fragment,
    not belief propagation.  Every real edge of the complete typed induced
    adjacency participates (extra chords included).  A pair with type 0 ("no
    real edge") and every padding slot contribute the multiplicative identity
    1: they enter the log-sum as exactly 0, and no learnable "no edge"
    potential exists.  Padding nodes carry no state and are never summed.

``N`` (independent per-edge state copies)
    Same ``q``, ``K``, root state, 133-dim input and MLP, but

        N_H,c(s) = prod_{(r,j) in E} [sum_t q_c(t|a_j) K_b,c(s,t)]
                   * prod_{(i,j) in E, i,j != r}
                     [sum_{u,v} q_c(u|a_i) q_c(v|a_j) K_b,c(u,v)]

    i.e. every non-root occurrence in every edge gets its own state copy.  N
    keeps each edge's type and both endpoint atom types; it removes the
    dependence between different edges that share a non-root atom.  N is a
    separately trained arm with identical parameters / shapes / initial
    values -- never a post-hoc inference-time slicing of a trained T.

Content input (T and N, 133 dimensions, identical layout)::

    [ q_c(s|a_root)            16*4 = 64
      log(T_H,c(s) or N_H,c(s)) / max(m,1)   16*4 = 64
      family one-hot                         3
      n / 5,  m / 10                         2 ]

with ``m`` the number of real edges.  ``phi`` is
``Linear(133,64) -> SiLU -> Linear(64,32)`` (bias, no norm / dropout / skip).

Because ``T`` / ``N`` are computed from the whole typed fragment they are
automatically invariant to any reordering of the non-root nodes, so T/N do not
enumerate permutations at all; D keeps the historical exact average.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_LF = _SS / "zinc_local_factorization_v1" / "code"
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_V2 = _SS / "zinc_recipe_trunk_v2" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_BF), str(_LF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import bf_data as BFD  # noqa: E402
import lf_models as LM  # noqa: E402

ARMS = ("D", "T", "N")
MODES = ("T", "N")

C_CH = 16                       # channels C (fixed, not swept)
R_STATE = 4                     # states per atom (fixed, not swept)
TRI = R_STATE * (R_STATE + 1) // 2   # 10 = symmetric upper triangle incl. diagonal
K_BONDS = 3                     # bond types 1..3

MAX_NODES = BFD.MAX_NODES       # 5
N_PAIRS = BFD.N_PAIRS           # 10
PAIR_INDEX = BFD.PAIR_INDEX
N_FAMILIES = len(BFD.FAMILIES)  # 3
ATOM_CLASSES = BFD.ATOM_CATEGORIES  # 28
PHI_OUT = LM.PHI_OUT            # 32
S_DIM = LM.S_DIM                # 198
CONTENT_IN = 2 * C_CH * R_STATE + N_FAMILIES + 2  # 133

# declared initialisation seeds (never tuned; seed 0 keeps LF-D's exact values)
CONTENT_SEED = 101
PHI_SEED_D = 202
SF_SEED = 404
SEED_STRIDE = 500

PHI_PARAMETERS = (ATOM_CLASSES * C_CH * R_STATE + K_BONDS * C_CH * TRI
                  + (CONTENT_IN * 64 + 64) + (64 * 32 + 32))
# 1792 + 480 + 8576 + 2080 = 12,928


# ---------------------------------------------------------------------------
# symmetric potential parameterisation
# ---------------------------------------------------------------------------


def _tri_to_full() -> torch.Tensor:
    """``[TRI, R*R]`` 0/1 matrix: packed upper triangle -> symmetric matrix."""
    out = torch.zeros(TRI, R_STATE * R_STATE)
    k = 0
    for i in range(R_STATE):
        for j in range(i, R_STATE):
            out[k, i * R_STATE + j] = 1.0
            out[k, j * R_STATE + i] = 1.0
            k += 1
    if k != TRI:
        raise RuntimeError("upper-triangle packing is inconsistent")
    return out


_TRI_TO_FULL = _tri_to_full()

_ASSIGN: dict[int, torch.Tensor] = {}


def table_lengths(table) -> torch.Tensor:
    """Real node count per pattern (``PatternTable`` stores masks, not lengths)."""
    return table.node_mask.sum(-1).to(torch.long)


def assignments(length: int) -> torch.Tensor:
    """All ``R**(length-1)`` state assignments of the real non-root slots.

    Row ``k`` holds the state index of canonical slots ``1..length-1``.  This is
    a static enumeration of the *sum index*, not learned state and not a cache
    of any model output.
    """
    if length not in _ASSIGN:
        k = int(length) - 1
        if k < 1 or k > MAX_NODES - 1:
            raise ValueError(f"unsupported object length {length}")
        total = R_STATE ** k
        idx = torch.arange(total, dtype=torch.long)
        out = torch.zeros(total, k, dtype=torch.long)
        for t in range(k - 1, -1, -1):
            out[:, t] = idx % R_STATE
            idx = idx // R_STATE
        _ASSIGN[int(length)] = out
    return _ASSIGN[int(length)]


# ---------------------------------------------------------------------------
# atomic state module
# ---------------------------------------------------------------------------


class AtomicState(nn.Module):
    """``A[28,C,R]`` state logits and the ``3*C`` symmetric pair potentials."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(ATOM_CLASSES, C_CH, R_STATE))
        self.pot = nn.Parameter(torch.zeros(K_BONDS, C_CH, TRI))

    def q_log(self) -> torch.Tensor:
        """``log q_c(s|a)`` -> ``[28, C, R]``."""
        return F.log_softmax(self.logits, dim=-1)

    def log_k(self) -> torch.Tensor:
        """``log K_{b,c}(s,t) = tanh(S_b,c(s,t))`` -> ``[3, C, R, R]`` (symmetric)."""
        full = self.pot.reshape(K_BONDS * C_CH, TRI) @ _TRI_TO_FULL.to(
            device=self.pot.device, dtype=self.pot.dtype)
        return torch.tanh(full.reshape(K_BONDS, C_CH, R_STATE, R_STATE))

    def reinit(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(int(seed))
        self.reinit_with(gen)

    def reinit_with(self, gen: torch.Generator) -> None:
        LM._isolated_normal(self.logits, 0.5, gen)   # A ~ Normal(0, 0.5)
        LM._isolated_normal(self.pot, 0.5, gen)      # S upper triangle ~ Normal(0, 0.5)


# ---------------------------------------------------------------------------
# T / N branch
# ---------------------------------------------------------------------------


class AtomicBranch(nn.Module):
    """Content branch for T (exact shared state) or N (independent copies).

    ``permuted_phi`` keeps the name/semantics of the LF branch: it returns one
    32-dim vector per *pattern* ``[P, 32]`` recomputed from the current weights
    on every forward call (autograd kept, nothing cached across steps).  T/N are
    intrinsically permutation invariant, so the "average" is the identity.
    """

    def __init__(self, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = str(mode)
        self.state = AtomicState()
        self.l1 = nn.Linear(CONTENT_IN, 64)
        self.l2 = nn.Linear(64, PHI_OUT)

    # -- initialisation -------------------------------------------------
    def reinit(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(int(seed))
        self.state.reinit_with(gen)
        LM._isolated_linear(self.l1, gen)
        LM._isolated_linear(self.l2, gen)

    # -- per-length contraction ----------------------------------------
    def _group_content(self, table: LM.PatternTable, sel: torch.Tensor,
                       length: int) -> torch.Tensor:
        logq = self.state.q_log()                                  # [28,C,R]
        logk = self.state.log_k().reshape(K_BONDS, C_CH, R_STATE * R_STATE)
        dtype = logq.dtype
        device = logq.device
        C, R = C_CH, R_STATE
        PL = int(sel.numel())
        atom = table.atom_ids[sel]                                 # [PL,5] long
        pair = table.pair_ids[sel]                                 # [PL,10] long
        assign = assignments(length).to(device)                    # [K,L-1]
        K = int(assign.shape[0])
        edge = pair > 0                                            # real typed edge
        root_slots = [k for k, (i, _j) in enumerate(PAIR_INDEX) if i == 0 and _j < length]
        nr_slots = [k for k, (i, j) in enumerate(PAIR_INDEX) if i >= 1 and j < length]
        valid_slots = [k for k, (_i, j) in enumerate(PAIR_INDEX) if j < length]
        lk_c = logk.permute(1, 0, 2)                               # [C,3,R*R]

        def pair_pot(k):
            """``log K_{b_k,c}(.,.)`` for slot ``k`` -> ``[PL,C,R,R]``."""
            b = pair[:, k].clamp(min=1) - 1
            return (lk_c[:, b, :].expand(C, PL, R * R)
                    .reshape(C, PL, R, R).permute(1, 0, 2, 3))

        # ---- T: exact sum over the joint state of the non-root atoms ----
        base = logq.new_zeros((PL, C, K))          # sum_i log q + sum_nonroot log K
        q_sel = logq[atom[:, 1:length]]                            # [PL,L-1,C,R]
        for t in range(length - 1):
            base = base + q_sel[:, t][:, :, assign[:, t]]
        for k in nr_slots:
            i, j = PAIR_INDEX[k]
            used = edge[:, k]
            if not bool(used.any()):
                continue
            b = pair[:, k].clamp(min=1) - 1
            idx = assign[:, i - 1] * R + assign[:, j - 1]           # [K]
            term = (lk_c[:, b, :].expand(C, PL, R * R)
                    .gather(2, idx.view(1, 1, K).expand(C, PL, K)))
            base = base + term.permute(1, 0, 2) * used.view(PL, 1, 1).to(dtype)
        root = base.new_zeros((PL, C, R, K))       # sum_{root edges} log K(s, s_j)
        for k in root_slots:
            j = PAIR_INDEX[k][1]
            used = edge[:, k]
            if not bool(used.any()):
                continue
            # log K_{b,c}(s, s_j) for every root state s and assignment k
            term = (pair_pot(k)
                    .gather(3, assign[:, j - 1].view(1, 1, 1, K)
                            .expand(PL, C, R, K)))                   # [PL,C,R,K]
            root = root + term * used.view(PL, 1, 1, 1).to(dtype)
        log_t = torch.logsumexp(base.unsqueeze(2) + root, dim=-1)   # [PL,C,R]

        # ---- N: one independent state copy per edge occurrence ----------
        log_n = base.new_zeros((PL, C, R))
        for k in root_slots:
            j = PAIR_INDEX[k][1]
            used = edge[:, k]
            if not bool(used.any()):
                continue
            lqj = logq[atom[:, j]]                                  # [PL,C,R]
            term = torch.logsumexp(lqj.unsqueeze(-2) + pair_pot(k), dim=-1)  # [PL,C,R]
            log_n = log_n + term * used.view(PL, 1, 1).to(dtype)
        for k in nr_slots:
            i, j = PAIR_INDEX[k]
            used = edge[:, k]
            if not bool(used.any()):
                continue
            lqi, lqj = logq[atom[:, i]], logq[atom[:, j]]            # [PL,C,R]
            term = torch.logsumexp(lqi.unsqueeze(-1) + lqj.unsqueeze(-2) + pair_pot(k),
                                   dim=(-2, -1))                     # [PL,C]
            log_n = log_n + term.unsqueeze(-1) * used.view(PL, 1, 1).to(dtype)

        # only real edges of *this* object count: padded pair slots are ignored
        m = edge[:, valid_slots].sum(dim=1).to(dtype)                # [PL]
        denom = m.clamp(min=1.0).view(PL, 1, 1)
        log_val = log_t if self.mode == "T" else log_n
        q_root = torch.exp(logq[atom[:, 0]])                         # [PL,C,R]
        fam = F.one_hot(table.family[sel], num_classes=N_FAMILIES).to(dtype)
        lengths = table_lengths(table)
        scal = torch.stack([lengths[sel].to(dtype) / float(MAX_NODES),
                            m / float(N_PAIRS)], dim=-1)
        x = torch.cat([q_root.reshape(PL, C * R),
                       (log_val / denom).reshape(PL, C * R),
                       fam, scal], dim=-1)
        if x.shape[-1] != CONTENT_IN:
            raise RuntimeError(f"content width {x.shape[-1]} != {CONTENT_IN}")
        return x

    # -- pattern-level phi ---------------------------------------------
    def content_input(self, table: LM.PatternTable) -> torch.Tensor:
        lengths = table_lengths(table)
        P = int(table.n_patterns)
        pieces = []
        for length in sorted({int(v) for v in lengths.tolist()}):
            sel = (lengths == length).nonzero(as_tuple=False).reshape(-1)
            pieces.append((sel, self._group_content(table, sel, length)))
        if not pieces:
            raise RuntimeError("empty pattern table")
        out = torch.zeros(P, CONTENT_IN, dtype=pieces[0][1].dtype,
                          device=pieces[0][1].device)
        for sel, blk in pieces:
            out = out.index_copy(0, sel, blk)
        if int(out.shape[0]) != P:
            raise RuntimeError("content block assembly lost patterns")
        return out

    def permuted_phi(self, table: LM.PatternTable) -> torch.Tensor:
        return self.l2(F.silu(self.l1(self.content_input(table))))

    def forward(self, batch: Mapping[str, torch.Tensor],
                table: LM.PatternTable) -> torch.Tensor:
        # The occurrence aggregation is the audited LF implementation, reused
        # verbatim; only ``permuted_phi`` above differs.
        return LM.ContentBranch.forward(self, batch, table)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class SFModel(LM.LocalFactorModel):
    """P3 carrier + ``W_s`` port; the content branch is arm specific.

    ``D`` keeps ``lf_models.ContentBranch("D")`` untouched (same modules, same
    construction order, same declared seeds) so this round's D arm *is* the
    previous round's D training path.  ``T`` / ``N`` replace the branch by
    :class:`AtomicBranch`, which has no content embedding table at all.
    """

    def __init__(self, norm: Mapping[str, object], *, arm: str, dropout: float = 0.0):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        super().__init__(norm, arm="D", dropout=dropout)
        self.arm = arm
        if arm in MODES:
            self.branch = AtomicBranch(arm)

    @property
    def mode(self) -> str:
        return getattr(self.branch, "mode", "D")


def _reinit_dense(branch: LM.ContentBranch, seed: int) -> None:
    """LF-D's content init, with the seed offset applied for seed 1."""
    gen = torch.Generator().manual_seed(int(CONTENT_SEED) + SEED_STRIDE * int(seed))
    LM._isolated_normal(branch.atom_emb.weight, 1.0, gen)
    LM._isolated_normal(branch.pair_emb.weight, 1.0, gen)
    gen_phi = torch.Generator().manual_seed(int(PHI_SEED_D) + SEED_STRIDE * int(seed))
    LM._isolated_linear(branch.phi.l1, gen_phi)
    LM._isolated_linear(branch.phi.l2, gen_phi)


def build_model(arm: str, norm: Mapping[str, object], seed: int = 0, device="cpu"):
    """Seed-0 public P3 core + train-only SSL encoders + W_s=0 + arm branch."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    torch.manual_seed(int(seed))
    model = SFModel(norm, arm=arm)
    ssl = LM.load_ssl_encoders(model, device="cpu")
    if arm == "D":
        _reinit_dense(model.branch, seed)
    else:
        model.branch.reinit(SF_SEED + SEED_STRIDE * int(seed))
    model = model.to(device)
    total = C.parameter_count(model)
    info = {
        "arm": arm,
        "candidate": f"SF-{arm}",
        "seed": int(seed),
        "parameters": int(total),
        "parameters_expected": EXPECTED_PARAMETERS[arm],
        "parameters_breakdown": parameter_breakdown(model),
        "ssl_encoders": ssl,
        "content_init_seed": (CONTENT_SEED + SEED_STRIDE * int(seed)) if arm == "D"
        else (SF_SEED + SEED_STRIDE * int(seed)),
        "phi_init_seed": (PHI_SEED_D + SEED_STRIDE * int(seed)) if arm == "D" else None,
        "w_s_zero": bool(model.w_s.weight.abs().sum().item() == 0.0),
        "all_require_grad": bool(all(p.requires_grad for p in model.parameters())),
    }
    if total != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError(f"arm {arm}: {total} parameters != expected {EXPECTED_PARAMETERS[arm]}")
    return model, info


def parameter_breakdown(model: SFModel) -> dict[str, int]:
    groups: dict[str, int] = {}
    for name in ("encoders.atom_encoder", "encoders.bond_encoder", "encoders.junction_encoder",
                 "group_composer", "reader", "w_s"):
        groups[name] = int(sum(p.numel() for p in model.get_submodule(name).parameters()))
    if model.arm == "D":
        for name in ("branch.atom_emb", "branch.pair_emb", "branch.phi"):
            groups[name] = int(sum(p.numel() for p in model.get_submodule(name).parameters()))
    else:
        for name in ("branch.state", "branch.l1", "branch.l2"):
            groups[name] = int(sum(p.numel() for p in model.get_submodule(name).parameters()))
    groups["total"] = int(sum(p.numel() for p in model.parameters()))
    return groups


EXPECTED_PARAMETERS = {"D": 449_601, "T": 451_777, "N": 451_777}
PARAMETER_CAP = 500_000
SSL_SHA256 = "006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97"


def content_state(model: SFModel) -> dict[str, torch.Tensor]:
    """The learnable content parameters (D: embeddings + phi; T/N: state + MLP)."""
    return {k: v.detach() for k, v in model.state_dict().items() if k.startswith("branch.")}


def hash_tensors(state: Mapping[str, torch.Tensor]) -> str:
    return LM.hash_tensors(state)
