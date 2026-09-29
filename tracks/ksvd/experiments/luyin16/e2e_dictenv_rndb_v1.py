"""E2E-DictEnv-RNDB-v1 — Rolewise Nonlinear Dictionary Binding (core module).

Round ``e2e_dictenv_rndb_v1`` (Workstream Z, ZINC).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_rndb_v1_preregistration.md``.
Prior-artifact audit: ``tracks/ksvd/notes/e2e_dictenv_rndb_v1_prior_artifact_audit.md``.

Question.  The frozen CSSD-q1 binding forms the per-role structure x chemistry
products ``p_{v,k}`` and aggregates them **linearly** (``p_v = sum_k p_{v,k}``)
before any nonlinearity.  RNDB keeps the exact same products but inserts one
small **shared** nonlinear operator between the individual residual dictionary
role responses and their aggregation:

    u_v = p_v^common + sum_{k in K_dict} [ p_{v,k} + psi_A(p_{v,k}) ]
        = p_v + sum_{k in K_dict} psi_A(p_{v,k})

with the common coordinate untouched and, identically for edges,

    u^E_uv = p^E,common_uv + sum_{k in K_dict} [ p^E_{uv,k} + psi_E(p^E_{uv,k}) ].

The module reuses ``CSSDModel`` (and hence ``CleanMechModel`` / ``AuditModel``)
instead of copying the pipeline: only the occurrence-level binding is replaced.

CPU only.  ``official_test_loaded = False`` in every payload.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2

PROTOCOL_VERSION = "e2e_dictenv_rndb_v1"

#: the only new modules.  node 96 -> 32 -> 96, edge 48 -> 16 -> 48.
PSI_NODE_HIDDEN = 32
PSI_EDGE_HIDDEN = 16

#: frozen second-layer initialisation standard deviation (pre-registration 4.4).
PSI_INIT_STD = 0.01

#: numerical tolerances of the correctness gates (pre-registration section 7).
DECOMPOSITION_TOL = 1.0e-6
PSI_OFF_TOL = 1.0e-6

#: the round's single arm.
RNDB_TAG = "RNDB"
RNDB_SPEC = cm.CleanMechSpec(RNDB_TAG, "C6", coding="sparse")
RNDB_MASK = cm.C6_MASK


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


def cpu_only_guard(device: Any) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise RuntimeError(f"RNDB round is CPU-only, got device={resolved}")


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class RNDBModel(cssd.CSSDModel):
    """CSSD-q1 with the shared rolewise nonlinear dictionary binding.

    With ``psi_A``/``psi_E`` disabled (``psi_node_on = psi_edge_on = False``)
    the forward is the parent CSSD-q1 forward.  The parent parameters keep the
    exact ``build_cssd_model`` RNG stream: ``psi`` is initialised after the
    parent ``__init__`` and never consumes parent RNG draws.
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        psi_node_hidden: int = PSI_NODE_HIDDEN,
        psi_edge_hidden: int = PSI_EDGE_HIDDEN,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        if self.node_binding != "paired" or self.edge_binding != "paired":
            raise ValueError("RNDB requires the paired node and edge bindings")
        d_a = int(self.W_A_S.shape[1])
        d_e = int(self.W_E_S.shape[1])
        self.psi_node_hidden = int(psi_node_hidden)
        self.psi_edge_hidden = int(psi_edge_hidden)
        self.psi_A = nn.Sequential(
            nn.Linear(d_a, self.psi_node_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(self.psi_node_hidden, d_a, bias=False),
        )
        self.psi_E = nn.Sequential(
            nn.Linear(d_e, self.psi_edge_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(self.psi_edge_hidden, d_e, bias=False),
        )
        # W1: standard Kaiming-uniform (nn.Linear default).  W2: Normal(0, 0.01).
        nn.init.kaiming_uniform_(self.psi_A[0].weight, a=math.sqrt(5.0))
        nn.init.kaiming_uniform_(self.psi_E[0].weight, a=math.sqrt(5.0))
        with torch.no_grad():
            self.psi_A[2].weight.normal_(0.0, PSI_INIT_STD)
            self.psi_E[2].weight.normal_(0.0, PSI_INIT_STD)
        # frozen inference-time intervention switches (never trained).
        self.psi_node_on = True
        self.psi_edge_on = True
        self.dict_zero = False

    # -- frozen role bookkeeping -------------------------------------------------

    @property
    def n_common(self) -> int:
        return int(self.common_dim)

    @property
    def n_dict(self) -> int:
        return int(self.W_A_S.shape[0]) - int(self.common_dim)

    # -- node occurrence ---------------------------------------------------------

    def _psi_sum(self, psi: nn.Sequential, pre: torch.Tensor) -> torch.Tensor:
        """``sum_k psi(p_k) = W2 @ sum_k SiLU(W1 p_k)`` (W2 is linear)."""
        pooled = F.silu(pre).sum(dim=1)
        return pooled @ psi[2].weight.t()

    def _node_psi_basis(self) -> torch.Tensor:
        """``[ATOM_CATEGORIES, K, H]`` with ``basis[c,k,h] = sum_j W_A_C[c,j] W_A_S[n+k,j] W1_A[h,j]``.

        The chemistry projection is one-hot, so this category-conditioned basis
        is exact and removes the ``[n_occ, K, H, D_A]`` contraction.
        """
        w_dict = self.W_A_S[self.n_common :]
        return torch.einsum("cj,kj,hj->ckh", self.W_A_C, w_dict, self.psi_A[0].weight)

    def _edge_psi_basis(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Three ``[BOND_CATEGORIES, K, H]`` bases for the ``+``, ``|.|``, ``*`` blocks."""
        d_struct = int(self.W_E_S.shape[0]) // 3
        n = self.n_common
        w = self.W_E_S
        w1 = self.psi_E[0].weight
        c = self.W_E_C
        plus = torch.einsum("hj,cj,kj->ckh", w1, c, w[:d_struct][n:])
        delta = torch.einsum("hj,cj,kj->ckh", w1, c, w[d_struct : 2 * d_struct][n:])
        prod = torch.einsum("hj,cj,kj->ckh", w1, c, w[2 * d_struct :][n:])
        return plus, delta, prod

    # -- node occurrence ---------------------------------------------------------

    def rndb_node_occurrence(
        self, coord: torch.Tensor, data: Any, *, occ_coord_node: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``u_v`` (RNDB) for every node occurrence, ``[n_occ, D_A]``."""
        d = float(p2.D_A)
        n = self.n_common
        if occ_coord_node is None:
            z = coord
            cat = data.dict_atom
            gather = data.env_occ_node.to(coord.device)
            out = self._node_occurrence_value(z, cat, gather, d)
            return out
        occ = occ_coord_node.to(coord.device)
        z = coord[occ]
        cat = data.dict_atom[data.env_occ_node.to(coord.device)]
        chem = F.one_hot(cat, num_classes=int(p2.ATOM_CATEGORIES)).to(z.dtype) @ self.W_A_C
        if self.dict_zero:
            return (z[:, :n] @ self.W_A_S[:n]) * chem / math.sqrt(d)
        u = (z @ self.W_A_S) * chem / math.sqrt(d)
        if not self.psi_node_on:
            return u
        a = self._node_psi_basis()[cat]
        pre = z[:, n:].unsqueeze(-1) * a / math.sqrt(d)
        return u + self._psi_sum(self.psi_A, pre)

    def _node_occurrence_value(
        self, z: torch.Tensor, cat: torch.Tensor, gather: torch.Tensor, d: float
    ) -> torch.Tensor:
        """Per-node value gathered to occurrences (fast path; no shuffle)."""
        n = self.n_common
        chem = F.one_hot(cat, num_classes=int(p2.ATOM_CATEGORIES)).to(z.dtype) @ self.W_A_C
        if self.dict_zero:
            return ((z[:, :n] @ self.W_A_S[:n]) * chem / math.sqrt(d))[gather]
        u = (z @ self.W_A_S) * chem / math.sqrt(d)
        if not self.psi_node_on:
            return u[gather]
        a = self._node_psi_basis()[cat]
        pre = z[:, n:].unsqueeze(-1) * a / math.sqrt(d)
        return (u + self._psi_sum(self.psi_A, pre))[gather]

    # -- edge occurrence ---------------------------------------------------------

    def _edge_chemistry(self, data: Any, dtype: torch.dtype) -> torch.Tensor:
        b = F.one_hot(data.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(dtype)
        return b

    def rndb_edge_occurrence(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``u^E_uv`` (RNDB) for every bond occurrence, ``[n_bond, D_E]``."""
        bu = data.env_bond_u if bond_u is None else bond_u
        bv = data.env_bond_v if bond_v is None else bond_v
        bu = bu.to(coord.device)
        bv = bv.to(coord.device)
        d_struct = int(coord.shape[1])
        w = self.W_E_S
        w_plus, w_delta, w_prod = w[:d_struct], w[d_struct : 2 * d_struct], w[2 * d_struct :]
        b = self._edge_chemistry(data, coord.dtype)
        chem = b @ self.W_E_C
        d_e = float(w.shape[1])
        cu, cv = coord[bu], coord[bv]
        if self.dict_zero:
            n = self.n_common
            cu_c, cv_c = cu[:, :n], cv[:, :n]
            g_common = torch.cat([cu_c + cv_c, torch.abs(cu_c - cv_c), cu_c * cv_c], dim=1)
            w_common = torch.cat([w_plus[:n], w_delta[:n], w_prod[:n]], dim=0)
            return (g_common @ w_common) * chem / math.sqrt(d_e)
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        ue = (g @ w) * chem / math.sqrt(d_e)
        if not self.psi_edge_on:
            return ue
        n = self.n_common
        cu_d, cv_d = cu[:, n:], cv[:, n:]
        g_plus_d = cu_d + cv_d
        g_delta_d = torch.abs(cu_d - cv_d)
        g_prod_d = cu_d * cv_d
        basis_plus, basis_delta, basis_prod = self._edge_psi_basis()
        cat = data.env_bond_type.to(coord.device)
        pre = (
            g_plus_d.unsqueeze(-1) * basis_plus[cat]
            + g_delta_d.unsqueeze(-1) * basis_delta[cat]
            + g_prod_d.unsqueeze(-1) * basis_prod[cat]
        ) / math.sqrt(d_e)
        return ue + self._psi_sum(self.psi_E, pre)

    # -- environment (frozen parent structure; only the occurrence call differs) --

    def environments_masked(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: audit.AuditMask,
        fill: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        n = int(coord.shape[0])
        anchor = data.anchor.to(coord.dtype)
        if int(anchor.shape[1]) != audit.ANCHOR_DIM_EXPECTED:
            raise RuntimeError(f"anchor width {int(anchor.shape[1])} != {audit.ANCHOR_DIM_EXPECTED}")
        if mask.anchor_zero_groups:
            anchor = audit._replace_grouped_columns(
                anchor, audit.ANCHOR_GROUPS, mask.anchor_zero_groups, fill, "anchor:"
            )

        occ_coord_node = None
        if mask.use_node_shuffle:
            occ_coord_node = getattr(data, "env_occ_coord_node", None)
            if occ_coord_node is None:
                raise RuntimeError("node-shuffle intervention requires data.env_occ_coord_node")
        u = self.rndb_node_occurrence(coord, data, occ_coord_node=occ_coord_node)
        if mask.node_binding_zero:
            u = torch.zeros_like(u)
        flat = torch.zeros((n * int(p2.N_SHELLS), int(u.shape[1])), device=u.device, dtype=u.dtype)
        flat.index_add_(
            0,
            data.env_occ_root.to(coord.device) * int(p2.N_SHELLS) + data.env_occ_shell.to(coord.device),
            u,
        )
        node_slots = flat.view(n, int(p2.N_SHELLS), int(u.shape[1]))

        bond_u = data.env_bond_u
        bond_v = data.env_bond_v
        if mask.use_edge_shuffle:
            bond_u = getattr(data, "env_bond_u_shuffled", None)
            bond_v = getattr(data, "env_bond_v_shuffled", None)
            if bond_u is None or bond_v is None:
                raise RuntimeError("edge-shuffle intervention requires shuffled endpoint fields")
        ue = self.rndb_edge_occurrence(coord, data, bond_u=bond_u, bond_v=bond_v)
        if mask.edge_binding_zero:
            ue = torch.zeros_like(ue)
        flat_e = torch.zeros((n * int(p2.SHELLPAIR_CLASSES), int(ue.shape[1])), device=ue.device, dtype=ue.dtype)
        flat_e.index_add_(
            0,
            data.env_bond_root.to(coord.device) * int(p2.SHELLPAIR_CLASSES)
            + data.env_bond_shellpair.to(coord.device),
            ue,
        )
        edge_slots = flat_e.view(n, int(p2.SHELLPAIR_CLASSES), int(ue.shape[1]))

        if self.env_mlp is not None:
            z = torch.cat([anchor, node_slots.reshape(n, -1), edge_slots.reshape(n, -1)], dim=1)
            return self.env_mlp(z)
        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        anchor_out = self.anchor_encoder(anchor)
        fused = torch.cat([anchor_out, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1)
        return self.fusion(fused)

    # -- psi diagnostics ---------------------------------------------------------

    def psi_parameters(self) -> dict[str, float]:
        return {
            "psi_A_W1_norm": float(self.psi_A[0].weight.detach().norm()),
            "psi_A_W2_norm": float(self.psi_A[2].weight.detach().norm()),
            "psi_E_W1_norm": float(self.psi_E[0].weight.detach().norm()),
            "psi_E_W2_norm": float(self.psi_E[2].weight.detach().norm()),
        }


def build_rndb_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    spec: cm.CleanMechSpec | None = None,
) -> RNDBModel:
    """Build RNDB with the exact CSSD-q1 parent RNG stream for the parent params."""
    spec = RNDB_SPEC if spec is None else spec
    torch.manual_seed(int(seed))
    return RNDBModel(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        node_binding=spec.node_binding,
        edge_binding=spec.edge_binding,
        coding=spec.coding,
    )


# ---------------------------------------------------------------------------
# correctness diagnostics: exact per-coordinate decomposition
# ---------------------------------------------------------------------------


def node_role_contributions(
    model: RNDBModel,
    coord: torch.Tensor,
    occ_node: torch.Tensor,
    atom: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(p_all, parent)`` for node occurrences.

    ``p_all[i, k] = z_{v,k} (W_A_S[k] odot c_v)`` with ``c_v = q_v W_A_C / sqrt(D_A)``;
    ``parent[i] = (z_v W_A_S) odot c_v``.  ``p_all.sum(1)`` must equal ``parent``.
    """
    z = coord[occ_node].to(dtype)
    qc = F.one_hot(atom[occ_node], num_classes=int(p2.ATOM_CATEGORIES)).to(dtype)
    c = qc @ model.W_A_C.to(dtype) / math.sqrt(float(p2.D_A))
    p_all = z.unsqueeze(-1) * model.W_A_S.to(dtype).unsqueeze(0) * c.unsqueeze(1)
    parent = (z @ model.W_A_S.to(dtype)) * c
    return p_all, parent


def edge_role_contributions(
    model: RNDBModel,
    coord: torch.Tensor,
    bond_u: torch.Tensor,
    bond_v: torch.Tensor,
    bond_type: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(p_all, parent)`` for bond occurrences (analogous node decomposition)."""
    d_struct = int(coord.shape[1])
    w = model.W_E_S.to(dtype)
    w_plus, w_delta, w_prod = w[:d_struct], w[d_struct : 2 * d_struct], w[2 * d_struct :]
    cu, cv = coord[bond_u].to(dtype), coord[bond_v].to(dtype)
    b = F.one_hot(bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(dtype)
    c = b @ model.W_E_C.to(dtype) / math.sqrt(float(w.shape[1]))
    r_all = (
        (cu + cv).unsqueeze(-1) * w_plus.unsqueeze(0)
        + torch.abs(cu - cv).unsqueeze(-1) * w_delta.unsqueeze(0)
        + (cu * cv).unsqueeze(-1) * w_prod.unsqueeze(0)
    )
    p_all = r_all * c.unsqueeze(1)
    g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
    parent = (g @ w) * c
    return p_all, parent


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def parameter_audit(model: RNDBModel, parent_params: int) -> dict[str, Any]:
    """Exact parameter accounting (pre-registration section 6)."""
    new_node = sum(p.numel() for p in model.psi_A.parameters())
    new_edge = sum(p.numel() for p in model.psi_E.parameters())
    total = int(sum(p.numel() for p in model.parameters()))
    delta = int(total - parent_params)
    return {
        "parent_params": int(parent_params),
        "rndb_params": int(total),
        "psi_A_params": int(new_node),
        "psi_E_params": int(new_edge),
        "new_params": int(new_node + new_edge),
        "delta": int(delta),
        "relative_delta": float(delta / max(int(parent_params), 1)),
        "new_params_budget": 8000,
        "new_params_ok": bool((new_node + new_edge) <= 8000),
        "ratio_ok": bool(delta <= 0.10 * int(parent_params)),
        "psi_A_hidden": int(model.psi_node_hidden),
        "psi_E_hidden": int(model.psi_edge_hidden),
        "official_test_loaded": False,
    }


def _distribution_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().reshape(-1)
    if int(values.numel()) == 0:
        return {key: float("nan") for key in ("mean", "median", "p90", "max", "std")}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
        "max": float(values.max()),
        "std": float(values.std(unbiased=False)),
    }


def _response_block(p_dict: torch.Tensor, psi_out: torch.Tensor) -> dict[str, Any]:
    """Response magnitudes / ratios / effective rank of one role tensor pair."""
    base_norm = p_dict.detach().double().norm(dim=-1).reshape(-1)
    psi_norm = psi_out.detach().double().norm(dim=-1).reshape(-1)
    active = base_norm > 0.0
    ratio = psi_norm[active] / base_norm[active]
    flat = psi_out.detach().double().reshape(-1, psi_out.shape[-1])
    if int(flat.shape[0]) > 20000:
        generator = torch.Generator().manual_seed(0)
        index = torch.randperm(int(flat.shape[0]), generator=generator)[:20000]
        flat = flat[index]
    singular = torch.linalg.svdvals(flat)
    participation = float((singular.sum() ** 2) / (singular**2).sum()) if int(singular.numel()) else float("nan")
    return {
        "n_roles": int(base_norm.numel()),
        "n_active_roles": int(active.sum()),
        "base_norm": _distribution_stats(base_norm[active]),
        "psi_norm": _distribution_stats(psi_norm[active]),
        "ratio": _distribution_stats(ratio),
        "psi_output_variance": float(psi_out.detach().double().var(unbiased=False)),
        "psi_output_effective_rank": participation,
        "psi_output_top_singular_fraction": float((singular[0] / singular.sum())) if int(singular.numel()) else float("nan"),
    }


def response_stats(model: RNDBModel, data: Any) -> dict[str, Any]:
    """Node / edge rolewise base and psi response diagnostics (frozen inference)."""
    model.eval()
    with torch.no_grad():
        coord = model.code(data.dict_phi)
        n = model.n_common
        d_a = math.sqrt(float(p2.D_A))
        d_e = math.sqrt(float(model.W_E_S.shape[1]))
        # node roles
        cat = data.dict_atom
        chem = F.one_hot(cat, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype) @ model.W_A_C
        p_dict = (
            coord[:, n:].unsqueeze(-1)
            * (model.W_A_S[n:].unsqueeze(0) * chem.unsqueeze(1))
            / d_a
        )
        a = model._node_psi_basis()[cat]
        pre = coord[:, n:].unsqueeze(-1) * a / d_a
        psi_out = F.silu(pre) @ model.psi_A[2].weight.t()
        node = _response_block(p_dict, psi_out)
        # edge roles
        cu, cv = coord[data.env_bond_u], coord[data.env_bond_v]
        b = F.one_hot(data.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(coord.dtype)
        chem_e = b @ model.W_E_C
        d_struct = int(coord.shape[1])
        w = model.W_E_S
        g_plus = cu + cv
        g_delta = torch.abs(cu - cv)
        g_prod = cu * cv
        r_dict = (
            g_plus.unsqueeze(-1) * w[:d_struct].unsqueeze(0)
            + g_delta.unsqueeze(-1) * w[d_struct : 2 * d_struct].unsqueeze(0)
            + g_prod.unsqueeze(-1) * w[2 * d_struct :].unsqueeze(0)
        )
        p_dict_e = r_dict[:, n:] * chem_e.unsqueeze(1) / d_e
        bp, bd, bx = model._edge_psi_basis()
        bond_cat = data.env_bond_type
        pre_e = (
            (g_plus[:, n:].unsqueeze(-1) * bp[bond_cat])
            + (g_delta[:, n:].unsqueeze(-1) * bd[bond_cat])
            + (g_prod[:, n:].unsqueeze(-1) * bx[bond_cat])
        ) / d_e
        psi_out_e = F.silu(pre_e) @ model.psi_E[2].weight.t()
        edge = _response_block(p_dict_e, psi_out_e)
    return {"node": node, "edge": edge, "official_test_loaded": False}


def psi_health(model: RNDBModel) -> dict[str, Any]:
    """Parameter-norm / near-zero-fraction audit of the two psi operators."""
    out: dict[str, Any] = {"official_test_loaded": False}
    for name, psi in (("psi_A", model.psi_A), ("psi_E", model.psi_E)):
        weight = torch.cat([layer.weight.detach().double().reshape(-1) for layer in psi if hasattr(layer, "weight")])
        out[name] = {
            "param_count": int(weight.numel()),
            "param_norm": float(weight.norm()),
            "abs_mean": float(weight.abs().mean()),
            "abs_max": float(weight.abs().max()),
            "frac_abs_lt_1e8": float((weight.abs() < 1e-8).double().mean()),
            "frac_abs_lt_1e12": float((weight.abs() < 1e-12).double().mean()),
            "W1_norm": float(psi[0].weight.detach().norm()),
            "W2_norm": float(psi[2].weight.detach().norm()),
        }
    return out


__all__ = [
    "PROTOCOL_VERSION",
    "PSI_NODE_HIDDEN",
    "PSI_EDGE_HIDDEN",
    "PSI_INIT_STD",
    "DECOMPOSITION_TOL",
    "PSI_OFF_TOL",
    "RNDB_TAG",
    "RNDB_SPEC",
    "RNDB_MASK",
    "official_test_blocker",
    "cpu_only_guard",
    "RNDBModel",
    "build_rndb_model",
    "node_role_contributions",
    "edge_role_contributions",
    "max_abs_diff",
    "parameter_audit",
    "response_stats",
    "psi_health",
]
