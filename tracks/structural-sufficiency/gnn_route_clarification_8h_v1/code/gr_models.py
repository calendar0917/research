"""The three stage-1 architectures, the additive readout family A and the
frozen-representation diagnostic head D.

Design provenance (frozen in the task brief, not re-designed here)
-------------------------------------------------------------------
``L``   the original LCRC-C1 local encoder + readout, re-used unchanged through
        ``cr_models.ArmA`` (:class:`LCRCModel` is a thin adapter only).
``G0``  the repository's existing 5-layer GINE (``sq_gine``), node input = the
        nine OGB atom categorical columns summed embeddings + degree + min-ring
        embeddings, edge input = three bond categorical embeddings, dense head
        ``[sum h, mean h, log1p n] -> 128 -> 1``.
``Gx``  the *input bridge*: node input ``concat(atom_feat51, normalised
        atom_scalar9)`` through a 60->128 Linear+SiLU, edge input
        ``bond_feat13`` through a 13->128 Linear (no extra activation); the five
        GINE/BN layers and the residual topology are identical to ``G0`` and are
        initialised from the ``G0`` canonical state; dense head
        ``[sum h, mean h, normalised global42, log1p n] -> 128 -> 1``.

``A``   the one new readout family of this round (task brief section 5)::

            t_v = dropout(SiLU(W h_v + b_h))
            s_v = w_s . t_v            (no bias)
            m_v = w_m . t_v            (no bias)
            y   = b + sum_v s_v + (1/n) sum_v m_v + sum_j u_j(z_j)
            u_j(z) = w2_j . [SiLU(w1_j z + b1_j) - SiLU(b1_j)]   (16 hidden, no output bias)

        There is no other residual / bypass / attention / graph MLP.  The
        per-atom native ledger is ``a_v = s_v + m_v / n``; ``u_j`` is reported
        per column and never redistributed over atoms.

``D``   the same head fitted on a frozen final representation (stage 3).

Parameter budget (analytic, never a width search)
--------------------------------------------------
``P_A(H) = (128 + 3) H + 48 M + 1``; ``H`` minimises ``|P_A(H) - P_dense(T)|``
over ``H in 1..1024``, ties to the smaller ``H``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gr_common as C
import gr_data as D

HIDDEN = 128
N_LAYERS = 5
G0_DENSE_PARAMS = C.G0_READOUT_PARAMS
GX_DENSE_PARAMS = C.GX_READOUT_PARAMS
UNIV_WIDTH = C.UNIVARIATE_WIDTH
HIDDEN_MAX = 1024


# ---------------------------------------------------------------------------
# budget arithmetic
# ---------------------------------------------------------------------------


def additive_params(hidden: int, n_univariate: int) -> int:
    """``(D+3)*H + 48*M + 1`` with D = 128."""
    return (HIDDEN + 3) * int(hidden) + 48 * int(n_univariate) + 1


def choose_hidden(dense_readout_params: int, n_univariate: int) -> dict[str, Any]:
    best = None
    for hidden in range(1, HIDDEN_MAX + 1):
        params = additive_params(hidden, n_univariate)
        delta = abs(params - int(dense_readout_params))
        if best is None or delta < best[1]:
            best = (hidden, delta, params)
    hidden, delta, params = best
    return {
        "hidden": int(hidden),
        "n_univariate": int(n_univariate),
        "nominal_readout_params": int(params),
        "dense_readout_params": int(dense_readout_params),
        "abs_delta": int(delta),
        "relative_delta": float(delta / max(1, int(dense_readout_params))),
        "formula": "(128+3)*H + 48*M + 1 = 131*H + 48*M + 1",
        "search": "analytic argmin over H in 1..1024, ties to the smaller H",
    }


# ---------------------------------------------------------------------------
# GINE trunk (G0 / Gx)
# ---------------------------------------------------------------------------


def _torch_modules():
    from torch_geometric.nn import GINEConv

    return GINEConv


class GineTrunk(nn.Module):
    """Five GINE/BN/SiLU/dropout/residual layers plus the input encoders."""

    def __init__(self, kind: str, spec, norm: Mapping[str, Any], n_categories: Mapping[str, int]) -> None:
        super().__init__()
        if kind not in ("G0", "Gx"):
            raise KeyError(kind)
        self.kind = kind
        GINEConv = _torch_modules()
        h = HIDDEN
        if kind == "G0":
            self.atom_embeddings = nn.ModuleList([nn.Embedding(int(n_categories[f"atom_{j}"]), h) for j in range(9)])
            self.degree_embedding = nn.Embedding(int(n_categories["degree"]), h)
            self.ring_embedding = nn.Embedding(int(n_categories["ring"]), h)
            self.bond_embeddings = nn.ModuleList([nn.Embedding(int(n_categories[f"bond_{j}"]), h) for j in range(3)])
        else:
            self.node_proj = nn.Linear(int(spec.atom_feat_dim) + int(spec.n_scalars), h)
            self.edge_proj = nn.Linear(int(spec.bond_feat_dim), h)
            self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
            self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))
            self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
            self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, h)),
                    train_eps=True,
                )
                for _ in range(N_LAYERS)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(N_LAYERS)])
        self.drop = nn.Dropout(C.DROPOUT)

    # -- input encoders ----------------------------------------------------
    def encode_nodes(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.kind == "G0":
            x_cat = batch["x_cat"]
            out = self.atom_embeddings[0](x_cat[:, 0])
            for j in range(1, 9):
                out = out + self.atom_embeddings[j](x_cat[:, j])
            return out + self.degree_embedding(batch["deg"]) + self.ring_embedding(batch["ring"])
        scalar = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        return F.silu(self.node_proj(torch.cat([batch["atom_feat"], scalar], dim=1)))

    def encode_edges(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.kind == "G0":
            edge_cat = batch["edge_cat"]
            out = self.bond_embeddings[0](edge_cat[:, 0])
            for j in range(1, 3):
                out = out + self.bond_embeddings[j](edge_cat[:, j])
            return out
        return self.edge_proj(batch["bond_feat"])

    def z_global(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (batch["global"] - self.global_mean) / self.global_scale

    # -- message passing ---------------------------------------------------
    def layers(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *,
        node_presence: torch.Tensor | None = None,
        edge_presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact GINE update, optionally gated by the continuous-extension masks."""
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            src, dst = edge_index[0], edge_index[1]
            msg = (h[src] + edge_attr).relu()
            if edge_presence is not None:
                msg = msg * edge_presence.view(-1, 1)
            agg = h.new_zeros((h.shape[0], h.shape[1])).index_add_(0, dst, msg)
            update = conv.nn((1.0 + conv.eps) * h + agg)
            update = F.silu(norm(update))
            update = self.drop(update)
            h = update + residual
            if node_presence is not None:
                h = h * node_presence.view(-1, 1)
        return h

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = self.encode_nodes(batch)
        e = self.encode_edges(batch)
        return self.layers(x, batch["edge_index"], e)


# ---------------------------------------------------------------------------
# dense graph head (G0 / Gx) and additive head (A / D)
# ---------------------------------------------------------------------------


def _sum_pool(values: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = values.new_zeros((n_graphs, values.shape[1])) if values.dim() > 1 else values.new_zeros(n_graphs)
    return out.index_add_(0, atom_batch, values)


def _mean_pool(values: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int, counts: torch.Tensor) -> torch.Tensor:
    denom = counts.clamp(min=1.0).to(values.dtype).unsqueeze(1)
    return values.new_zeros((n_graphs, values.shape[1])).index_add_(0, atom_batch, values) / denom


class DenseHead(nn.Module):
    """``[sum h, mean h, (global z), log1p n] -> 128 -> SiLU -> dropout -> 1``."""

    def __init__(self, kind: str, global_dim: int = 42) -> None:
        super().__init__()
        extra = int(global_dim) if kind == "Gx" else 0
        in_dim = 2 * HIDDEN + extra + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.SiLU(), nn.Dropout(C.DROPOUT), nn.Linear(HIDDEN, 1)
        )

    def forward_parts(
        self,
        h: torch.Tensor,
        atom_batch: torch.Tensor,
        gine_n_atoms: torch.Tensor,
        *,
        global_z: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        n_graphs = int(gine_n_atoms.numel())
        summed = _sum_pool(h, atom_batch, n_graphs)
        meaned = _mean_pool(h, atom_batch, n_graphs, gine_n_atoms)
        logn = torch.log1p(gine_n_atoms.view(-1, 1).to(h.dtype))
        blocks = [summed, meaned]
        if global_z is not None:
            blocks.append(global_z)
        blocks.append(logn)
        pooled = torch.cat(blocks, dim=1)
        return {"pred_std": self.net(pooled).view(-1), "pooled": pooled, "terminology": ["sum", "mean", "log1p_n"]}


class AdditiveHead(nn.Module):
    """The single new readout family A (also used by the frozen head D)."""

    def __init__(
        self,
        hidden: int,
        *,
        kind: str,
        n_active: int,
        active_indices: Sequence[int],
        size_mean: float,
        size_scale: float,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.hidden = int(hidden)
        self.n_active = int(n_active)
        self.active_indices = tuple(int(i) for i in active_indices)
        self.size_mean = float(size_mean)
        self.size_scale = float(size_scale)
        self.w_hidden = nn.Parameter(torch.empty(self.hidden, HIDDEN))
        self.b_hidden = nn.Parameter(torch.empty(self.hidden))
        self.w_s = nn.Parameter(torch.empty(self.hidden))
        self.w_m = nn.Parameter(torch.empty(self.hidden))
        self.bias = nn.Parameter(torch.zeros(1))
        self.u_w1 = nn.Parameter(torch.empty(self.n_active, UNIV_WIDTH))
        self.u_b1 = nn.Parameter(torch.empty(self.n_active, UNIV_WIDTH))
        self.u_w2 = nn.Parameter(torch.empty(self.n_active, UNIV_WIDTH))
        # default init exactly as nn.Linear would do it (one isolated RNG stream)
        nn.init.kaiming_uniform_(self.w_hidden, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(HIDDEN)
        nn.init.uniform_(self.b_hidden, -bound, bound)
        # the four bias-free output vectors follow nn.Linear's default bound
        nn.init.uniform_(self.w_s, -bound, bound)
        nn.init.uniform_(self.w_m, -bound, bound)
        nn.init.kaiming_uniform_(self.u_w1, a=math.sqrt(5))
        ub = 1.0 / math.sqrt(UNIV_WIDTH)
        nn.init.uniform_(self.u_b1, -ub, ub)
        nn.init.uniform_(self.u_w2, -ub, ub)

    # -- the size / global channel ----------------------------------------
    def size_z(self, n_atoms: torch.Tensor) -> torch.Tensor:
        n = n_atoms.view(-1).to(torch.float64)
        z = (torch.log1p(n) - self.size_mean) / self.size_scale
        return z.view(-1, 1).to(torch.float32)

    def univariate(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(sum_j u_j(z_j) [B], per-column u_j [B, M])."""
        zz = z.unsqueeze(-1)                                   # B, M, 1
        pre = torch.nn.functional.silu(self.u_w1.unsqueeze(0) * zz + self.u_b1.unsqueeze(0))
        ref = torch.nn.functional.silu(self.u_b1).unsqueeze(0)
        cols = ((pre - ref) * self.u_w2.unsqueeze(0)).sum(dim=-1)   # B, M
        return cols.sum(dim=1), cols

    def forward_parts(
        self,
        h: torch.Tensor,
        atom_batch: torch.Tensor,
        gine_n_atoms: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, Any]:
        n_graphs = int(gine_n_atoms.numel())
        t = torch.nn.functional.silu(h @ self.w_hidden.t() + self.b_hidden)
        t = F.dropout(t, p=C.DROPOUT, training=self.training)
        s_v = t @ self.w_s
        m_v = t @ self.w_m
        n = gine_n_atoms.view(-1).to(h.dtype).clamp(min=1.0)
        atom_graph = atom_batch
        sum_s = _sum_pool(s_v, atom_graph, n_graphs)
        sum_m = _sum_pool(m_v, atom_graph, n_graphs)
        mean_m = sum_m / n
        u_sum, u_cols = self.univariate(z)
        pred = self.bias.view(-1) + sum_s + mean_m + u_sum
        a_v = s_v + m_v / n[atom_graph]
        return {
            "pred_std": pred,
            "hidden": t,
            "s_v": s_v,
            "m_v": m_v,
            "a_v": a_v,
            "sum_s": sum_s,
            "mean_m": mean_m,
            "u_sum": u_sum,
            "u_terms": u_cols,
            "intercept": self.bias.view(-1).expand(n_graphs),
            "n_atoms": gine_n_atoms.view(-1),
            "z": z,
            "term_names": ["intercept", "sum_s", "mean_m", "u_sum"],
        }


class AdditiveGraphModel(nn.Module):
    """GINE trunk (G0/Gx) + the additive readout A."""

    def __init__(self, trunk: GineTrunk, head: AdditiveHead) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = head

    @property
    def kind(self) -> str:
        return self.trunk.kind

    def z_matrix(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.kind == "G0":
            return self.head.size_z(batch["gine_n_atoms"])
        return self.trunk.z_global(batch)[:, self.head.active_indices]

    def parts(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        h = self.trunk(batch)
        z = self.z_matrix(batch)
        parts = self.head.forward_parts(h, batch["gine_batch"], batch["gine_n_atoms"], z)
        parts["h"] = h
        parts["global_z"] = self.trunk.z_global(batch) if self.kind == "Gx" else None
        return parts

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.parts(batch)["pred_std"]


class DenseGraphModel(nn.Module):
    """GINE trunk (G0/Gx) + the original dense graph head."""

    def __init__(self, trunk: GineTrunk, head: DenseHead) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = head

    @property
    def kind(self) -> str:
        return self.trunk.kind

    def parts(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        h = self.trunk(batch)
        gz = self.trunk.z_global(batch) if self.kind == "Gx" else None
        parts = self.head.forward_parts(h, batch["gine_batch"], batch["gine_n_atoms"], global_z=gz)
        parts["h"] = h
        parts["global_z"] = gz
        return parts

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.parts(batch)["pred_std"]


class LCRCModel(nn.Module):
    """Thin adapter around ``cr_models.ArmA`` (the unchanged original C1)."""

    kind = "L"

    def __init__(self, arm) -> None:
        super().__init__()
        self.arm = arm
        self.trunk = arm.trunk

    def parts(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return self.arm.parts(batch)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.arm.parts(batch)["pred_std"]


# ---------------------------------------------------------------------------
# construction + canonical initial states
# ---------------------------------------------------------------------------


def build_trunk(kind: str, b) -> GineTrunk:
    return GineTrunk(kind, b.spec, b.norm, D.gine_cache(b.dataset).category_counts())


class ModelSet:
    """Canonical initial states for one task (built once, cached on disk).

    All initial states are CPU tensors; the canonical streams are isolated and
    never depend on an experiment's training RNG.
    """

    def __init__(self, b) -> None:
        self.b = b
        self.dataset = b.dataset
        mods = D.modules()
        self.mods = mods
        import cr_models as M

        self.cr_models = M
        self.c1_state = M.canonical_c1_state(b.spec, b.norm)
        self.g0_state = self._build_g0_state()
        self.gx_state = self._build_gx_state()

    def _build_g0_state(self) -> dict[str, torch.Tensor]:
        C.set_global_seeds(C.INIT_SEED_CANONICAL)
        model = DenseGraphModel(build_trunk("G0", self.b), DenseHead("G0", self.b.spec.global_dim))
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _build_gx_state(self) -> dict[str, torch.Tensor]:
        C.set_global_seeds(C.INIT_SEED_GX)
        model = DenseGraphModel(build_trunk("Gx", self.b), DenseHead("Gx", self.b.spec.global_dim))
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for key, value in self.g0_state.items():
            if key.startswith("trunk.convs.") or key.startswith("trunk.norms."):
                if key not in state or state[key].shape != value.shape:
                    raise RuntimeError(f"Gx trunk state cannot inherit {key}")
                state[key] = value.detach().clone()
        return state

    # -- builders ----------------------------------------------------------
    def dense(self, kind: str) -> DenseGraphModel:
        model = DenseGraphModel(build_trunk(kind, self.b), DenseHead(kind, self.b.spec.global_dim))
        state = self.g0_state if kind == "G0" else self.gx_state
        model.load_state_dict(state, strict=True)
        return model

    def additive(self, kind: str, *, active_indices: Sequence[int], size_scaler: tuple[float, float]) -> AdditiveGraphModel:
        """T trunk initialised from its canonical state; head from isolated seed 0."""
        trunk = build_trunk(kind, self.b)
        head = self._new_head(
            kind,
            n_active=(1 if kind == "G0" else len(active_indices)),
            active_indices=active_indices,
            size_scaler=size_scaler,
        )
        model = AdditiveGraphModel(trunk, head)
        canonical = self.g0_state if kind == "G0" else self.gx_state
        trunk_state = {
            k[len("trunk.") :]: v
            for k, v in canonical.items()
            if k.startswith("trunk.")
            and (k.startswith("trunk.convs.") or k.startswith("trunk.norms.") or _is_trunk_input(k, kind) or k.startswith("trunk.scalar_") or k.startswith("trunk.global_"))
        }
        missing = set(trunk.state_dict()) - set(trunk_state)
        if missing:
            raise RuntimeError(f"canonical {kind} trunk state is missing {sorted(missing)}")
        trunk.load_state_dict(trunk_state, strict=True)
        return model

    def _new_head(self, kind: str, *, n_active: int, active_indices: Sequence[int], size_scaler):
        C.set_global_seeds(C.INIT_SEED_HEAD)
        dense_params = G0_DENSE_PARAMS if kind == "G0" else GX_DENSE_PARAMS
        budget = choose_hidden(dense_params, n_active)
        head = AdditiveHead(
            budget["hidden"],
            kind=kind,
            n_active=n_active,
            active_indices=active_indices,
            size_mean=size_scaler[0],
            size_scale=size_scaler[1],
        )
        head.budget = budget
        return head

    def lcrc(self) -> LCRCModel:
        arm = self.cr_models.ArmA(self.b.spec, self.b.norm)
        arm.load_canonical(self.c1_state)
        return LCRCModel(arm)


def _is_trunk_input(key: str, kind: str) -> bool:
    if key.startswith("trunk.convs.") or key.startswith("trunk.norms."):
        return True
    if kind == "G0":
        return key.startswith("trunk.atom_embeddings.") or key.startswith("trunk.degree_embedding.") or key.startswith("trunk.ring_embedding.") or key.startswith("trunk.bond_embeddings.")
    return key.startswith("trunk.node_proj.") or key.startswith("trunk.edge_proj.")


def parameter_report(model: nn.Module) -> dict[str, Any]:
    total = int(sum(p.numel() for p in model.parameters()))
    buffers = int(sum(v.numel() for v in model.buffers() if torch.is_tensor(v)))
    names = {name: int(p.numel()) for name, p in model.named_parameters()}
    return {
        "n_parameters": total,
        "n_parameter_tensors": len(names),
        "n_buffer_elements": buffers,
        "by_module": names,
    }


def model_state_hash(model: nn.Module) -> str:
    return C.sha256_state(model.state_dict())
