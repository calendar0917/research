"""Local-content encoders for ``zinc_local_factorization_v1`` (arms D and F).

Both arms are the *same* P3 carrier: ``P3NullModel``'s atom/bond/junction
encoders, the Null grouping, ``group_composer``, ``reader`` and ``global33``,
plus the 198 -> 128 bias-free ``W_s`` port (zero initialised) that adds the
per-root fragment statistics to the atom encoder's first layer::

    a_v = Linear128->96( SiLU( Linear108->128(raw108_v) + W_s s_v ) )

Only the local content function ``phi`` differs:

``D`` (dense)
    One plain MLP on the fully expanded, masked content vector (98 -> 80 -> 32).

``F`` (factorized)
    Shared atom/key projections ``A`` (8 -> 32) and ``B`` (4 -> 32) plus
    learned role tables ``R_atom[5,32]`` / ``R_pair[10,32]`` produce one 32-d
    field per real atom / pair slot; channel-wise first, second and third
    elementary-symmetric moments over the real fields form
    ``[z1, z2, z3]``, concatenated with the masks and the family one-hot and
    read by a 114 -> 64 -> 32 head.

Both arms average ``phi`` over the root-preserving permutations of the real
non-root slots *before* the per-root/per-family occurrence aggregation
(``mean`` and ``mean_sq``), so no arm can depend on which real node a canonical
content order happened to place in slot 1..4.  The permutation average is an
internal symmetry of one model, not an ensemble: every physical object still
contributes exactly one multiplicity unit.

The content itself (atom categories / typed induced adjacency / masks /
family) is the unchanged ``bf_data`` object from
``zinc_branching_fragment_v1``.  No whole-pattern embedding exists: a content
is encoded from its atom/pair categories only, so a never-seen content is
encoded by the same shared parameters.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_V2 = _SS / "zinc_recipe_trunk_v2" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_BF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad2_models as A2  # noqa: E402
import bf_data as BFD  # noqa: E402

ARMS = ("D", "F")
MAX_NODES = BFD.MAX_NODES          # 5
N_PAIRS = BFD.N_PAIRS              # 10
N_FAMILIES = len(BFD.FAMILIES)     # 3
ATOM_CLASSES = BFD.ATOM_CATEGORIES  # 28
D_ATOM = 8
D_PAIR = 4
PHI_OUT = 32
S_DIM = N_FAMILIES * (2 * PHI_OUT + 2)  # 198
PAIR_INDEX = BFD.PAIR_INDEX
PAIR_J = np.asarray([j for _, j in PAIR_INDEX], dtype=np.int64)
PERM_MAX = math.factorial(MAX_NODES - 1)  # 24

# shared content embedding seeds and per-arm phi seeds (declared, never varied)
CONTENT_SEED = 101
PHI_SEED = {"D": 202, "F": 303}

_PAIR_LOOKUP = np.full((MAX_NODES, MAX_NODES), -1, dtype=np.int64)
for _k, (_i, _j) in enumerate(PAIR_INDEX):
    _PAIR_LOOKUP[_i, _j] = _k
    _PAIR_LOOKUP[_j, _i] = _k


# ---------------------------------------------------------------------------
# static permutation table (content-keyed, split-local, structure only)
# ---------------------------------------------------------------------------


class PatternTable:
    """Static root-preserving permutation enumeration for one split's patterns.

    ``idx_node[p, k, i]`` is the source slot whose content fills target slot
    ``i`` in permutation ``k``; ``weight[p, k]`` is ``1 / K_p`` for the ``K_p``
    real permutations (6 for 4-node, 24 for 5-node) and exactly 0 on unused
    padding rows.  ``node_mask`` / ``pair_mask`` are invariant under these
    permutations (the root stays at slot 0 and padding stays after the real
    nodes), so they are stored once per pattern.

    The table holds *no* learned state: it is built from the audited static
    content arrays and never caches a ``phi`` output.
    """

    def __init__(self, *, split: str, atom_ids, pair_ids, idx_node, idx_pair,
                 weight, node_mask, pair_mask, family, n_perm_by_pattern):
        self.split = str(split)
        self.atom_ids = atom_ids
        self.pair_ids = pair_ids
        self.idx_node = idx_node
        self.idx_pair = idx_pair
        self.weight = weight
        self.node_mask = node_mask
        self.pair_mask = pair_mask
        self.family = family
        self.n_perm_by_pattern = n_perm_by_pattern

    @property
    def n_patterns(self) -> int:
        return int(self.atom_ids.shape[0])

    def to(self, device) -> "PatternTable":
        return PatternTable(
            split=self.split,
            atom_ids=self.atom_ids.to(device),
            pair_ids=self.pair_ids.to(device),
            idx_node=self.idx_node.to(device),
            idx_pair=self.idx_pair.to(device),
            weight=self.weight.to(device),
            node_mask=self.node_mask.to(device),
            pair_mask=self.pair_mask.to(device),
            family=self.family.to(device),
            n_perm_by_pattern=self.n_perm_by_pattern,
        )


def build_pattern_table(cache: Mapping[str, object], *, split: str | None = None) -> PatternTable:
    pat_atom = np.asarray(cache["pat_atom"], dtype=np.int64)
    pat_pair = np.asarray(cache["pat_pair"], dtype=np.int64)
    pat_len = np.asarray(cache["pat_len"], dtype=np.int64)
    pat_family = np.asarray(cache["pat_family"], dtype=np.int64)
    split = str(cache["split"] if split is None else split)
    P = int(len(pat_len))

    idx_node = np.tile(np.arange(MAX_NODES, dtype=np.int64), (P, PERM_MAX, 1))
    weight = np.zeros((P, PERM_MAX), dtype=np.float32)
    n_perm = np.zeros((P,), dtype=np.int64)
    for p in range(P):
        length = int(pat_len[p])
        if length < 2 or length > MAX_NODES:
            raise RuntimeError(f"pattern {p}: unsupported real length {length}")
        real = list(range(1, length))
        perms = [(0,) + tuple(q) + tuple(range(length, MAX_NODES))
                 for q in itertools.permutations(real)]
        K = len(perms)
        if K != math.factorial(length - 1):
            raise RuntimeError("permutation enumeration is not factorial")
        idx_node[p, :K] = np.asarray(perms, dtype=np.int64)
        weight[p, :K] = np.float32(1.0 / K)
        n_perm[p] = K
    idx_pair = np.zeros((P, PERM_MAX, N_PAIRS), dtype=np.int64)
    for k, (i, j) in enumerate(PAIR_INDEX):
        idx_pair[..., k] = _PAIR_LOOKUP[idx_node[..., i], idx_node[..., j]]  # type: ignore[index]
    if (idx_pair < 0).any():
        raise RuntimeError("pair index lookup failed")

    node_mask = (np.arange(MAX_NODES)[None, :] < pat_len[:, None]).astype(np.float32)
    pair_mask = (PAIR_J[None, :] < pat_len[:, None]).astype(np.float32)
    return PatternTable(
        split=split,
        atom_ids=torch.from_numpy(pat_atom),
        pair_ids=torch.from_numpy(pat_pair),
        idx_node=torch.from_numpy(idx_node),
        idx_pair=torch.from_numpy(idx_pair),
        weight=torch.from_numpy(weight),
        node_mask=torch.from_numpy(node_mask),
        pair_mask=torch.from_numpy(pair_mask),
        family=torch.from_numpy(pat_family),
        n_perm_by_pattern=torch.from_numpy(n_perm),
    )


# ---------------------------------------------------------------------------
# content encoders
# ---------------------------------------------------------------------------


def _isolated_linear(lin: nn.Linear, gen: torch.Generator) -> None:
    """``nn.Linear`` default init (kaiming-uniform a=sqrt(5) == U(+-1/sqrt(fan_in)))."""
    fan_in = int(lin.weight.shape[1])
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        lin.weight.copy_(torch.empty_like(lin.weight).uniform_(-bound, bound, generator=gen))
        lin.bias.copy_(torch.empty_like(lin.bias).uniform_(-bound, bound, generator=gen))


def _isolated_normal(param: torch.Tensor, std: float, gen: torch.Generator) -> None:
    with torch.no_grad():
        param.copy_(torch.empty_like(param).normal_(0.0, float(std), generator=gen))


class DensePhi(nn.Module):
    """D: ordinary content MLP 98 -> 80 -> 32 (bias, SiLU, no norm/dropout/skip)."""

    IN_DIM = MAX_NODES * D_ATOM + N_PAIRS * D_PAIR + MAX_NODES + N_PAIRS + N_FAMILIES  # 98

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(self.IN_DIM, 80)
        self.l2 = nn.Linear(80, PHI_OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(F.silu(self.l1(x)))

    def reinit(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(int(seed))
        _isolated_linear(self.l1, gen)
        _isolated_linear(self.l2, gen)


class FactorizedPhi(nn.Module):
    """F: shared key projections + role modulation + 1st/2nd/3rd moment pooling.

    Fields (one 32-d vector per real slot, exactly 0 on padding)::

        f_i   = node_mask_i * tanh(A(E_atom[a_i]) + R_atom[i])
        f_ij  = pair_mask_ij * tanh(B(E_pair[b_ij]) + R_pair[ij])

    with ``m`` real fields (10 for 4-node, 15 for 5-node content) and
    channel-wise elementary symmetric moments z1/z2/z3 that never count the
    same field twice and never divide by padding.
    """

    ATOM_IN = D_ATOM
    PAIR_IN = D_PAIR
    IN_DIM = 3 * PHI_OUT + MAX_NODES + N_PAIRS + N_FAMILIES  # 114

    def __init__(self):
        super().__init__()
        self.a_proj = nn.Linear(self.ATOM_IN, PHI_OUT)
        self.b_proj = nn.Linear(self.PAIR_IN, PHI_OUT)
        self.role_atom = nn.Parameter(torch.zeros(MAX_NODES, PHI_OUT))
        self.role_pair = nn.Parameter(torch.zeros(N_PAIRS, PHI_OUT))
        self.l1 = nn.Linear(self.IN_DIM, 64)
        self.l2 = nn.Linear(64, PHI_OUT)

    def reinit(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(int(seed))
        _isolated_linear(self.a_proj, gen)
        _isolated_linear(self.b_proj, gen)
        _isolated_normal(self.role_atom, 0.1, gen)
        _isolated_normal(self.role_pair, 0.1, gen)
        _isolated_linear(self.l1, gen)
        _isolated_linear(self.l2, gen)

    @staticmethod
    def moments(fields: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Channel-wise z1/z2/z3 over the ``m`` real fields of each row.

        ``fields`` is ``[..., T, C]`` with padding fields exactly 0 and ``m``
        the number of real fields (``[...]``).  z_k is 0 where ``m < k``.
        """
        p1 = fields.sum(dim=-2)
        p2 = (fields * fields).sum(dim=-2)
        p3 = (fields * fields * fields).sum(dim=-2)
        mm = m.unsqueeze(-1).to(fields.dtype)
        z1 = p1 / mm.clamp(min=1.0)
        z2 = (p1 * p1 - p2) / (mm * (mm - 1.0)).clamp(min=1.0)
        z2 = torch.where((m >= 2).unsqueeze(-1), z2, torch.zeros_like(z2))
        z3 = (p1 * p1 * p1 - 3.0 * p1 * p2 + 2.0 * p3) / (
            mm * (mm - 1.0) * (mm - 2.0)
        ).clamp(min=1.0)
        z3 = torch.where((m >= 3).unsqueeze(-1), z3, torch.zeros_like(z3))
        return z1, z2, z3

    def forward(self, atom_e: torch.Tensor, pair_e: torch.Tensor,
                node_mask: torch.Tensor, pair_mask: torch.Tensor,
                family_onehot: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        fa = torch.tanh(self.a_proj(atom_e) + self.role_atom) * node_mask.unsqueeze(-1)
        fb = torch.tanh(self.b_proj(pair_e) + self.role_pair) * pair_mask.unsqueeze(-1)
        fields = torch.cat([fa, fb], dim=-2)
        z1, z2, z3 = self.moments(fields, m)
        u = torch.cat([z1, z2, z3, node_mask, pair_mask, family_onehot], dim=-1)
        return self.l2(F.silu(self.l1(u)))


class ContentBranch(nn.Module):
    """Shared content embeddings + the arm's content function ``phi``."""

    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.atom_emb = nn.Embedding(ATOM_CLASSES, D_ATOM)
        self.pair_emb = nn.Embedding(4, D_PAIR)  # 0 = no edge, 1..3 = raw bond type
        self.phi = DensePhi() if arm == "D" else FactorizedPhi()

    # -- initialisation -------------------------------------------------
    def reinit_shared_embeddings(self) -> None:
        gen = torch.Generator().manual_seed(CONTENT_SEED)
        _isolated_normal(self.atom_emb.weight, 1.0, gen)
        _isolated_normal(self.pair_emb.weight, 1.0, gen)

    def reinit_phi(self) -> None:
        self.phi.reinit(PHI_SEED[self.arm])  # type: ignore[attr-defined]

    # -- forward --------------------------------------------------------
    def permuted_phi(self, table: PatternTable) -> torch.Tensor:
        """``phi_bar`` for every pattern of the split, ``[P, 32]``.

        Recomputed from the current weights on every forward (autograd kept);
        nothing here is cached across optimizer steps.
        """
        P = table.n_patterns
        atoms = table.atom_ids.unsqueeze(1).expand(P, PERM_MAX, MAX_NODES)
        atoms = atoms.gather(2, table.idx_node)                     # [P, 24, 5]
        pairs = table.pair_ids.unsqueeze(1).expand(P, PERM_MAX, N_PAIRS)
        pairs = pairs.gather(2, table.idx_pair)                     # [P, 24, 10]

        atom_e = self.atom_emb(atoms)                                # [P,24,5,8]
        pair_e = self.pair_emb(pairs)                                # [P,24,10,4]
        node_mask = table.node_mask.unsqueeze(1).expand(P, PERM_MAX, MAX_NODES)
        pair_mask = table.pair_mask.unsqueeze(1).expand(P, PERM_MAX, N_PAIRS)
        atom_e = atom_e * node_mask.unsqueeze(-1)
        pair_e = pair_e * pair_mask.unsqueeze(-1)
        fam = F.one_hot(table.family, num_classes=N_FAMILIES).to(atom_e.dtype)
        fam = fam.unsqueeze(1).expand(P, PERM_MAX, N_FAMILIES)

        if self.arm == "D":
            x = torch.cat(
                [atom_e.flatten(-2), pair_e.flatten(-2), node_mask, pair_mask, fam], dim=-1
            )
            out = self.phi(x)                                        # [P,24,32]
        else:
            m = node_mask.sum(-1) + pair_mask.sum(-1)                # [P,24]
            out = self.phi(atom_e, pair_e, node_mask, pair_mask, fam, m)
        weight = table.weight.unsqueeze(-1)
        return (out * weight).sum(dim=1)                             # [P,32]

    def forward(self, batch: Mapping[str, torch.Tensor], table: PatternTable) -> torch.Tensor:
        phi_bar = self.permuted_phi(table)
        n_atoms = int(batch["n_atoms_total"])
        dtype = phi_bar.dtype
        if phi_bar.shape[0] != table.n_patterns:
            raise RuntimeError("pattern table / encoder mismatch")
        occ_phi = phi_bar[batch["occ_pat"].long()]
        occ_weight = batch["occ_count"].to(dtype).unsqueeze(1)
        occ_count = batch["occ_count"].to(dtype)
        flat = batch["occ_root"].long() * N_FAMILIES + batch["occ_family"].long()
        sums = phi_bar.new_zeros((n_atoms * N_FAMILIES, PHI_OUT))
        sqs = phi_bar.new_zeros((n_atoms * N_FAMILIES, PHI_OUT))
        counts = phi_bar.new_zeros((n_atoms * N_FAMILIES,))
        sums.index_add_(0, flat, occ_phi * occ_weight)
        # mean_sq is taken over the *permutation-averaged* phi_bar, exactly like
        # the mean: within-object permutation variance never enters.
        sqs.index_add_(0, flat, (occ_phi * occ_phi) * occ_weight)
        counts.index_add_(0, flat, occ_count)
        n = counts.clamp(min=1.0)
        mean = sums / n.unsqueeze(-1)
        mean_sq = sqs / n.unsqueeze(-1)
        stats = torch.cat(
            [mean, mean_sq, torch.log1p(counts).unsqueeze(-1), (counts > 0).to(dtype).unsqueeze(-1)],
            dim=-1,
        )
        return stats.reshape(n_atoms, S_DIM)


# ---------------------------------------------------------------------------
# model: unchanged P3 carrier + the content port
# ---------------------------------------------------------------------------


class LocalFactorModel(A2.P3NullModel):
    """P3 Null graph with ``s_v`` added to the atom encoder's first layer."""

    def __init__(self, norm: Mapping[str, object], *, arm: str, dropout: float = 0.0):
        super().__init__(norm, dropout=dropout)
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.w_s = nn.Linear(S_DIM, 128, bias=False)
        nn.init.zeros_(self.w_s.weight)
        self.branch = ContentBranch(arm)
        self.table: PatternTable | None = None

    def set_table(self, table: PatternTable) -> None:
        self.table = table

    def _table_for(self, batch: Mapping[str, torch.Tensor]) -> PatternTable:
        table = self.table
        if table is None:
            raise RuntimeError("pattern table not set before forward")
        split = batch.get("split")
        if split is not None and str(split) != table.split:
            raise RuntimeError(f"pattern table for {table.split!r} used on {split!r} batch")
        return table

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        s = self.branch(batch, self._table_for(batch))
        first = self.encoders.atom_encoder.lins[0]
        a = F.silu(first(batch["atom_feat"]) + self.w_s(s))
        a = self.encoders.atom_encoder.lins[1](a)

        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bond_feat = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        b = self.encoders.bond_encoder(bond_feat)

        ji = batch["junc_index"]
        jb = batch["junc_bond"]
        ai, aj, ak = a[ji[:, 0]], a[ji[:, 1]], a[ji[:, 2]]
        b1, b2 = b[jb[:, 0]], b[jb[:, 1]]
        junc_feat = torch.cat(
            [ai + ak, (ai - ak).abs(), ai * ak, aj, b1 + b2, (b1 - b2).abs(), batch["junc_feat"]],
            dim=1,
        )
        w = self.encoders.junction_encoder(junc_feat)

        W, V, n_b = self.group_to_bonds(w, batch)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1
        )
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = c.new_zeros((n_graphs, c.shape[1])).index_add_(0, batch["bond_batch"], c)
        a_sum = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"].to(a.dtype) - self.global_mean) / self.global_scale
        return self.reader(torch.cat([z, a_sum, a_sq, g], dim=1)).view(-1)


# ---------------------------------------------------------------------------
# construction / provenance
# ---------------------------------------------------------------------------

SSL_SHA256 = "006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97"
EXPECTED_PARAMETERS = {"D": 449_601, "F": 449_457}
PARAMETER_CAP = 500_000


def load_ssl_encoders(model: LocalFactorModel, device="cpu") -> dict:
    """Load only the seed-0 train-only masked-SSL encoders (sha256 checked)."""
    path = C.P3_SSL_S0
    if not path.exists():
        raise FileNotFoundError(f"missing train-only SSL encoder {path}; refusing a random fallback")
    sha = C.sha256_file(path)
    if sha != SSL_SHA256:
        raise RuntimeError(f"SSL encoder sha256 {sha} != {SSL_SHA256}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if sorted(state.keys()) != ["atom_encoder", "bond_encoder", "junction_encoder"]:
        raise RuntimeError(f"unexpected SSL encoder keys: {sorted(state.keys())}")
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        getattr(model.encoders, name).load_state_dict(
            {k: v.to(device) for k, v in state[name].items()}, strict=True
        )
    return {
        "source": str(path.relative_to(C.REPO)),
        "sha256": sha,
        "loaded": sorted(state.keys()),
        "note": "atom/bond/junction encoders only; composer and reader keep the seed-0 init",
    }


def build_model(arm: str, norm: Mapping[str, object], seed: int = 0, device="cpu"):
    """Public seed-0 P3 core + SSL encoders + W_s=0 + arm content branch.

    Construction order is fixed and identical for both arms:

    1. ``torch.manual_seed(seed)`` then the P3 core in the historical order;
    2. only the train-only SSL atom/bond/junction encoders are loaded;
    3. ``W_s`` (zero) and the content branch are appended;
    4. the shared content embedding tables are overwritten from
       ``CONTENT_SEED`` (one table, copied to both arms);
    5. the arm's ``phi`` is overwritten from its own ``PHI_SEED``.

    Steps 4-5 use isolated ``torch.Generator`` objects, so they cannot perturb
    the public construction stream or the data-order RNG.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    torch.manual_seed(int(seed))
    model = LocalFactorModel(norm, arm=arm)
    ssl = load_ssl_encoders(model, device="cpu")
    model.branch.reinit_shared_embeddings()
    model.branch.reinit_phi()
    model = model.to(device)
    total = C.parameter_count(model)
    if total > PARAMETER_CAP:
        raise RuntimeError(f"arm {arm}: {total} parameters > {PARAMETER_CAP}")
    info = {
        "arm": arm,
        "candidate": "LF-D" if arm == "D" else "LF-F",
        "seed": int(seed),
        "parameters": int(total),
        "parameters_expected": EXPECTED_PARAMETERS[arm],
        "parameters_breakdown": parameter_breakdown(model),
        "ssl_encoders": ssl,
        "content_embedding_seed": CONTENT_SEED,
        "phi_seed": PHI_SEED[arm],
        "w_s_zero": bool(model.w_s.weight.abs().sum().item() == 0.0),
        "w_s_zero_init": True,
        "all_require_grad": bool(all(p.requires_grad for p in model.parameters())),
        "init_order": [
            f"torch.manual_seed({int(seed)})",
            "P3Encoders + group_composer + reader (historical order)",
            "load train-only SSL atom/bond/junction encoders",
            "append w_s (zeros) + content branch",
            f"shared content embeddings from CONTENT_SEED={CONTENT_SEED} (isolated generator)",
            f"arm phi from PHI_SEED[{arm}]={PHI_SEED[arm]} (isolated generator)",
        ],
    }
    if total != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError(f"arm {arm}: {total} parameters != expected {EXPECTED_PARAMETERS[arm]}")
    return model, info


def parameter_breakdown(model: LocalFactorModel) -> dict[str, int]:
    groups = {}
    for name in ("encoders.atom_encoder", "encoders.bond_encoder", "encoders.junction_encoder",
                 "group_composer", "reader", "w_s", "branch.atom_emb", "branch.pair_emb",
                 "branch.phi"):
        groups[name] = int(sum(p.numel() for p in model.get_submodule(name).parameters()))
    groups["total"] = int(sum(p.numel() for p in model.parameters()))
    return groups


def public_state(model: LocalFactorModel) -> dict[str, torch.Tensor]:
    """Everything except the arm-specific ``w_s`` and content branch."""
    return {k: v for k, v in model.state_dict().items()
            if not (k.startswith("w_s.") or k.startswith("branch."))}


def shared_embedding_state(model: LocalFactorModel) -> dict[str, torch.Tensor]:
    return {
        "branch.atom_emb.weight": model.branch.atom_emb.weight.detach(),
        "branch.pair_emb.weight": model.branch.pair_emb.weight.detach(),
    }


def hash_tensors(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(np.ascontiguousarray(state[key].detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def p3_core_state(model: LocalFactorModel) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in model.state_dict().items():
        if key.startswith("encoders."):
            out[key[len("encoders."):]] = value
        elif key.startswith(("group_composer.", "reader.")):
            out[key] = value
        elif key in ("global_mean", "global_scale"):
            out[key] = value
    return out
