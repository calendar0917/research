"""The Bond-Anchored S2 Joint Composer (BAJC) on top of the frozen C1 function.

C1 is preserved *exactly* (nothing deleted, replaced or simplified)::

    a_v = AtomEncoder(atom_feat_v)                                   51 -> 128 -> 96
    b_e = BondEncoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bf_e])          301 -> 160 -> 96
    S1_v  = sum_{e incident v} b_e
    S1J_v = sum_{(v,u)} nbr_emb[(type(u), bond_uv)]
    S2_v  = sum_{v->w->u, u != v} [ path_emb(cat) + end_emb(end63)
                                    + joint_proj(path_u(cat) * path_v(end63))
                                    + close_emb(closure) + path_gate(cat) * a_u ]
    h_v   = Dropout(LayerNorm(LocalMLP([a_v, S1_v, S1J_v, S2_v_typed, S2_v_content, scalars_v])))
    y_base(G) = sum_v f(h_v) + rho([z_g, mean_v h, mean_v h^2, log1p n])          (= C1(G))

New residual (switchable, identically zero at initialisation)::

    r_p        = the frozen 200-d per-occurrence typed-S2 record of occurrence p
    u(r)       = SiLU(Linear(200,32)(r))          v(b) = SiLU(Linear(96,16)(b))
    x(p,q,e)   = concat[u(r_p), u(r_q), u(r_p)*u(r_q), v(b_e), closure4]        (113)
    psi        = Linear(64,32) . SiLU . Linear(113,64)
    z_P        = 0.5 * ( psi(x(p,q,e)) + psi(x(p_rev,q_rev,e)) )
    c_P        = Linear(32,1,bias=False)(z_P) / sqrt(max(1, n_paths3(G)))
    delta_comp = sum_{P in paths3(G)} c_P
    y_new(G)   = y_base(G) + delta_comp

The nonlinearity runs **before** the per-path aggregation and the two directional
evaluations are averaged, so ``z_P`` is exactly invariant to atom relabelling
(which can flip the canonical direction).  ``Linear(32,1).weight`` is
initialised to exactly zero, hence ``delta_comp == 0`` at initialisation and
``y_new == y_base`` bit-for-bit at that state.

The MARGINAL control keeps the identical event list, anchor bond, first path,
``closure4`` and ``1/sqrt(n)`` normaliser, and replaces only the second-path
representation by the molecule-level mean of the raw 200-d evidence::

    r_q_null = mean_{e in paths3(G)} r_second(e)
    x_null   = concat[u(r_p), u(r_q_null), u(r_p)*u(r_q_null), v(b_e), closure4]

``u`` is then applied to that mean (``u(mean r)``, not ``mean u(r)``), exactly as
specified.  Both variants share every tensor shape, every parameter and the same
initialisation stream.

Physical-bond evidence gating (task book C4) reuses the already validated
``CONDITIONAL_S1_S2_EVIDENCE_MASK`` semantics of
``lcrc_auto_subgraph_faithfulness_pilot_v1/as_gate.py``: S1 evidence is weighted
by the bond gate, each S2 occurrence by the product of its two bond gates, and
each composition event by the product of its three bond gates.  ``atom_feat``,
``atom_scalar`` and ``global`` are never gated.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bc_common as C

WIDTH = C.WIDTH
S2_EVIDENCE_DIM = C.S2_EVIDENCE_DIM
BOND_EVIDENCE_DIM = C.BOND_EVIDENCE_DIM
PAIR_HIDDEN = C.PAIR_HIDDEN
BOND_HIDDEN = C.BOND_HIDDEN
PAIR_INPUT_DIM = C.PAIR_INPUT_DIM
PSI_HIDDEN = C.PSI_HIDDEN
PSI_OUT = C.PSI_OUT

#: Composition events are evaluated in chunks of at most this many rows (exact sum).
EVENT_CHUNK = 262_144


def _cm():
    return C.modules()["cr_models"]


# ---------------------------------------------------------------------------
# the frozen C1 trunk + readout (bitwise ``cr_models.ArmA``)
# ---------------------------------------------------------------------------


def build_c1(spec, norm) -> nn.Module:
    """A fresh ``cr_models.ArmA`` at the canonical C1 initial state."""
    CM = _cm()
    model = CM.ArmA(spec, norm)
    return model


def c1_canonical_state(spec, norm) -> dict[str, torch.Tensor]:
    """The canonical seed-0 C1 initial state (read-only reuse)."""
    CM = _cm()
    return CM.canonical_c1_state(spec, norm)


# ---------------------------------------------------------------------------
# the composition branch
# ---------------------------------------------------------------------------


class Composer(nn.Module):
    """Shared projection + nonlinear joint MLP + zero-initialised scalar readout."""

    def __init__(self) -> None:
        super().__init__()
        self.r_proj = nn.Linear(S2_EVIDENCE_DIM, PAIR_HIDDEN)
        self.b_proj = nn.Linear(BOND_EVIDENCE_DIM, BOND_HIDDEN)
        self.psi_in = nn.Linear(PAIR_INPUT_DIM, PSI_HIDDEN)
        self.psi_out = nn.Linear(PSI_HIDDEN, PSI_OUT)
        self.readout = nn.Linear(PSI_OUT, 1, bias=False)
        nn.init.zeros_(self.readout.weight)
        #: isolated dropout stream: the frozen C1 dropout stream is never touched
        self._generator: torch.Generator | None = None

    # -- isolated dropout --------------------------------------------------
    def set_dropout_seed(self, seed: int) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        self._generator = gen

    def _shared_dropout(self, hidden: torch.Tensor) -> torch.Tensor:
        """One Bernoulli(keep) mask, applied to both directional evaluations."""
        if not self.training or C.BRANCH_DROPOUT <= 0.0:
            return hidden
        if self._generator is None:
            self.set_dropout_seed(C.BRANCH_DROPOUT_SEED)
        keep = 1.0 - float(C.BRANCH_DROPOUT)
        uniform = torch.rand(hidden.shape, generator=self._generator, dtype=torch.float32)
        mask = (uniform < keep).to(hidden.dtype) / keep
        return hidden * mask.to(hidden.device)

    # -- projections -------------------------------------------------------
    def project(self, evidence: torch.Tensor, bond_emb: torch.Tensor):
        u = F.silu(self.r_proj(evidence))
        v = F.silu(self.b_proj(bond_emb))
        return u, v

    def psi(self, pair_input: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.psi_in(pair_input))
        hidden = self._shared_dropout(hidden)
        return self.psi_out(hidden)

    # -- parameter report --------------------------------------------------
    def parameter_report(self) -> dict[str, Any]:
        named = {name: int(p.numel()) for name, p in self.named_parameters()}
        return {
            "total": int(sum(named.values())),
            "breakdown": dict(sorted(named.items())),
        }


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class BAJC(nn.Module):
    """Frozen C1 + the Bond-Anchored S2 Joint Composer residual."""

    def __init__(self, c1: nn.Module, variant: str) -> None:
        super().__init__()
        if variant not in C.VARIANTS:
            raise KeyError(variant)
        self.variant = variant
        self.c1 = c1                     # cr_models.ArmA (trunk.local + f + rho)
        self.composer = Composer()
        self.composer.set_dropout_seed(C.BRANCH_DROPOUT_SEED)

    # -- structure ---------------------------------------------------------
    @property
    def trunk(self):
        return self.c1.trunk

    @property
    def local(self):
        return self.c1.trunk.local

    def branch_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.composer.parameters() if p.requires_grad]

    def parameter_report(self) -> dict[str, Any]:
        c1 = int(sum(p.numel() for p in self.c1.parameters()))
        branch = int(sum(p.numel() for p in self.composer.parameters()))
        return {
            "variant": self.variant,
            "c1": c1,
            "c1_expected": C.C1_PARAMETERS,
            "c1_matches_frozen": c1 == C.C1_PARAMETERS,
            "branch": branch,
            "branch_fraction_of_c1": float(branch / max(1, c1)),
            "branch_within_budget": bool(branch <= C.BRANCH_MAX_FRACTION * c1),
            "total": c1 + branch,
            "branch_breakdown": self.composer.parameter_report(),
        }

    # -- frozen 200-d per-occurrence typed-S2 evidence ---------------------
    def s2_evidence(self, batch: Mapping[str, torch.Tensor], a: torch.Tensor) -> torch.Tensor:
        local = self.local
        if batch["path_cat"].numel():
            typed = torch.cat(
                [
                    local.path_emb(batch["path_cat"]),
                    local.end_emb(batch["path_end63"]),
                    local.joint_proj(local.path_u(batch["path_cat"]) * local.path_v(batch["path_end63"])),
                    local.close_emb(batch["path_closure"]),
                ],
                dim=1,
            )
            content = a[batch["path_end"]] * local.path_gate(batch["path_cat"])
        else:
            typed = a.new_zeros((0, C.S2_TYPED_DIM))
            content = a.new_zeros((0, a.shape[1]))
        evidence = torch.cat([typed, content], dim=1)
        if evidence.shape[1] != S2_EVIDENCE_DIM:
            raise RuntimeError(f"S2 evidence width {evidence.shape[1]} != frozen {S2_EVIDENCE_DIM}")
        return evidence

    # -- the frozen local input (with optional physical-bond evidence gate) --
    def local_input(
        self,
        batch: Mapping[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        scal: torch.Tensor,
        evidence: torch.Tensor | None,
        gate: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.local
        n_atoms = int(a.shape[0])
        width = int(a.shape[1])
        idx = batch["bond_index"]
        if gate is not None:
            bond_gate, nbr_gate, path_weight = gate
            if b.numel():
                b_eff = b * bond_gate.unsqueeze(1)
            else:
                b_eff = b
        else:
            bond_gate = nbr_gate = path_weight = None
            b_eff = b
        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b_eff).index_add_(0, v, b_eff)
        else:
            b_sum = a.new_zeros((n_atoms, width))
        if batch["nbr_code"].numel():
            nbr_emb = local.nbr_emb(batch["nbr_code"])
            if nbr_gate is not None:
                nbr_emb = nbr_emb * nbr_gate.unsqueeze(1)
            nbr = a.new_zeros((n_atoms, local.rank)).index_add_(0, batch["nbr_atom"], nbr_emb)
        else:
            nbr = a.new_zeros((n_atoms, local.rank))

        # typed S2 block (type | end | joint | closure) then the content block,
        # in exactly the frozen ``LocalContent`` order.
        if evidence is not None and evidence.shape[0] == int(batch["path_cat"].numel()):
            typed_width = C.S2_TYPED_DIM
            typed = evidence[:, :typed_width]
            content = evidence[:, typed_width:]
        else:
            typed = None
            content = None
        if typed is None or not batch["path_cat"].numel():
            path_agg = a.new_zeros((n_atoms, C.S2_TYPED_DIM))
            content_agg = a.new_zeros((n_atoms, width))
        else:
            w = path_weight.unsqueeze(1) if path_weight is not None else 1.0
            path_agg = a.new_zeros((n_atoms, typed.shape[1])).index_add_(
                0, batch["path_atom"], typed * w
            )
            content_agg = a.new_zeros((n_atoms, width)).index_add_(
                0, batch["path_atom"], content * w
            )
        x = torch.cat([a, b_sum, nbr, path_agg, content_agg, scal[:, list(local.scalar_cols)]], dim=1)
        if x.shape[1] != len(local.local_cols):
            raise RuntimeError(f"local width {x.shape[1]} != frozen C1 column count {len(local.local_cols)}")
        return x, path_agg

    # -- composition residual ---------------------------------------------
    def _mean_second_evidence(
        self,
        evidence: torch.Tensor,
        occ_index: torch.Tensor,
        graph: torch.Tensor,
        counts: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """Molecule-level mean of the raw 200-d evidence of the second path."""
        n = counts.to(torch.float64).clamp(min=1.0)
        gathered = evidence[occ_index] if occ_index.numel() else evidence.new_zeros((0, evidence.shape[1]))
        summed = evidence.new_zeros((n_graphs, evidence.shape[1])).index_add_(0, graph, gathered)
        return summed / n.unsqueeze(1).to(evidence.dtype)

    def compose(
        self,
        evidence: torch.Tensor,
        bond_emb: torch.Tensor,
        p3: Mapping[str, torch.Tensor],
        *,
        n_graphs: int,
        gate_product: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-event composer contributions and their exact graph-level sum."""
        occ = p3["p3_occ"]                     # (n_ev, 4) batch-level occurrence ids
        graph = p3["p3_graph"]
        counts = p3["p3_counts"].to(torch.float64)
        anchor_bond = p3["p3_anchor_bond"]
        closure = p3["p3_closure4"].view(-1, 1)
        n_ev = int(occ.shape[0])
        u, v = self.composer.project(evidence, bond_emb)
        if self.variant == "MARGINAL" and n_ev:
            mean_q = self._mean_second_evidence(evidence, occ[:, 1], graph, counts, n_graphs)
            mean_q_rev = self._mean_second_evidence(evidence, occ[:, 3], graph, counts, n_graphs)
            u_q = F.silu(self.composer.r_proj(mean_q))[graph]
            u_q_rev = F.silu(self.composer.r_proj(mean_q_rev))[graph]
        else:
            u_q = u[occ[:, 1]] if n_ev else u.new_zeros((0, PAIR_HIDDEN))
            u_q_rev = u[occ[:, 3]] if n_ev else u.new_zeros((0, PAIR_HIDDEN))
        scale = 1.0 / torch.sqrt(counts.clamp(min=1.0)).to(evidence.dtype)
        per_graph_scale = scale[graph] if n_ev else evidence.new_zeros(0)

        chunks: list[torch.Tensor] = []
        for start in range(0, n_ev, EVENT_CHUNK):
            stop = min(start + EVENT_CHUNK, n_ev)
            sl = slice(start, stop)
            u_p = u[occ[sl, 0]]
            u_q_fwd = u_q[sl]
            u_p_rev = u[occ[sl, 2]]
            u_q_rev_sl = u_q_rev[sl]
            v_b = v[anchor_bond[sl]]
            cl = closure[sl]
            x_fwd = torch.cat([u_p, u_q_fwd, u_p * u_q_fwd, v_b, cl], dim=1)
            x_rev = torch.cat([u_p_rev, u_q_rev_sl, u_p_rev * u_q_rev_sl, v_b, cl], dim=1)
            z = 0.5 * (self.composer.psi(x_fwd) + self.composer.psi(x_rev))
            c = self.composer.readout(z).view(-1) * per_graph_scale[sl]
            if gate_product is not None:
                c = c * gate_product[sl]
            chunks.append(c)
        contrib = torch.cat(chunks) if chunks else evidence.new_zeros(0)
        delta = evidence.new_zeros(n_graphs).index_add_(0, graph, contrib) if n_ev else evidence.new_zeros(n_graphs)
        return {"delta_comp": delta, "c_P": contrib, "n_paths3": p3["p3_counts"]}

    # -- full forward ------------------------------------------------------
    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        p3: Mapping[str, torch.Tensor],
        *,
        branch: bool = True,
        gate: Any | None = None,
        gate_product: torch.Tensor | None = None,
        c_chunk: int = 1_000_000,
        return_parts: bool = False,
    ):
        c1 = self.c1
        trunk = c1.trunk
        local = trunk.local
        n_graphs = int(batch["n_graphs"])
        a = trunk.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        else:
            bond_input = a.new_zeros((0, 3 * a.shape[1] + batch["bond_feat"].shape[1]))
        b = trunk.bond_encoder(bond_input)
        scal = (batch["atom_scalar"] - trunk.scalar_mean) / trunk.scalar_scale
        evidence = self.s2_evidence(batch, a)

        x, path_agg = self.local_input(batch, a, b, scal, evidence, gate)
        h = trunk.drop(local.norm(local.mlp(x)))

        atom_batch = batch["atom_batch"]
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, WIDTH)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, WIDTH)).index_add_(0, atom_batch, h * h) / denom
        contrib_atom = c1.f(h).view(-1)
        local_sum = contrib_atom.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        g = (batch["global"] - trunk.global_mean) / trunk.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = c1.rho(rho_in).view(-1)
        base = local_sum + rho

        parts: dict[str, Any] = {
            "pred_base_std": base,
            "local_sum": local_sum,
            "rho": rho,
            "h": h,
            "count": count,
            "evidence": evidence,
            "path_agg": path_agg,
        }
        if branch:
            comp = self.compose(
                evidence, b, p3, n_graphs=n_graphs, gate_product=gate_product
            )
            parts.update(comp)
            parts["pred_std"] = base + comp["delta_comp"]
        else:
            parts["delta_comp"] = base.new_zeros(n_graphs)
            parts["c_P"] = base.new_zeros(0)
            parts["pred_std"] = base
        if return_parts:
            return parts
        return parts["pred_std"]

    # -- institution F: freeze the teacher ---------------------------------
    def freeze_c1(self) -> dict[str, Any]:
        n_frozen = 0
        for param in self.c1.parameters():
            param.requires_grad_(False)
            n_frozen += int(param.numel())
        self.c1.eval()
        return {"frozen_parameters": n_frozen, "frozen_modules": ["c1"]}


# ---------------------------------------------------------------------------
# constraint helpers
# ---------------------------------------------------------------------------


def param_counts(model: BAJC) -> dict[str, int]:
    return {
        "c1": int(sum(p.numel() for p in model.c1.parameters())),
        "branch": int(sum(p.numel() for p in model.composer.parameters())),
    }


def branch_state_hash(model: BAJC) -> str:
    return C.sha256_state({k: v.detach().clone() for k, v in model.composer.state_dict().items()})


def assert_equal_branch_counts(joint: BAJC, marginal: BAJC) -> dict[str, Any]:
    a = joint.parameter_report()
    b = marginal.parameter_report()
    return {
        "joint_branch": a["branch"],
        "marginal_branch": b["branch"],
        "equal": bool(a["branch"] == b["branch"]),
        "joint_shapes": sorted(
            (name, tuple(p.shape)) for name, p in joint.composer.named_parameters()
        ),
        "marginal_shapes": sorted(
            (name, tuple(p.shape)) for name, p in marginal.composer.named_parameters()
        ),
    }


__all__ = [
    "BAJC",
    "Composer",
    "build_c1",
    "c1_canonical_state",
    "param_counts",
    "branch_state_hash",
    "assert_equal_branch_counts",
    "EVENT_CHUNK",
]
