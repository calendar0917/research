"""Candidate models for the performance-route round.

Base graph = the frozen LCRC ``C1`` (NO_BOND_MEAN) computation exactly as in
``lcrc_lipo_model_dissection_v1/dissect_model.py``:

    a_v = AtomEncoder(atom_feat_v)
    b_e = BondEncoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bf])
    S1, S1joint, S2(type/end/joint/closure/content), scalars
    h   = LayerNorm(LocalMLP([...]))
    h   = Dropout_{0.10}(h)                 # one call, shared by f and rho
    y   = sum_v f(h_v) + rho([g, mean h, mean h^2, log1p n])

Every extension keeps the module names of ``DissectLCRC`` for the shared
parameters, so the C1 initial state (``remap_state("C1", FULL(seed=0))``) is
copied exactly, never re-drawn.  New modules are initialised with a *private*
RNG stream.  ``lcpr_train`` resets the model RNG before the first update.

Extensions (all frozen before the first run):

``M_*``  the dropout/second-moment training statistics of ``rho``
``S_*``  support-conditional contraction of the 32-d joint (J) pair block
``A_*``  train-only auxiliary heads on the same shared ``h``/mask
``K_*``  the historical chem8 atom channels + 8-d latent graph contribution
``*_CONTEXT``  diagnostic learned context (only if the 7.6 trigger fires)
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lcpr_common as C

import dissect_model as DM

SHRINK_KINDS = {"s_select", "s_uniform"}
AUX_KINDS = {"a_logp", "a_mw", "a_shuffle"}
CONTEXT_KINDS = {"mp_context", "pool_context"}
CHEM8_KINDS = {"k_combo", "k_chem8", "a0_combo"}
CONTRIB_KINDS = {"k_combo", "k_contrib", "a0_combo"}


def _kaiming_rows_(weight: torch.Tensor, start: int) -> None:
    """Private Kaiming init for the inserted rows of a widened Linear."""
    nn.init.kaiming_uniform_(weight[start:], a=math.sqrt(5))


class ShrinkLocalContent(DM.LocalContent):
    """``LocalContent`` with the joint (J) block optionally contracted per root.

    ``J_r = J_marg_r + a_r * (J_full_r - J_marg_r)`` where ``J_marg`` is the
    frozen B5 (root, bt2) marginal operator and ``J_full`` the frozen full
    operator.  ``a_r`` is either the per-root mean of ``c/(c+kappa)`` over the
    paths rooted at ``r`` (``select``) or the Train-global mean ``a_bar``
    (``uniform``).  ``a_r == 1`` restores FULL, ``a_r == 0`` restores B5.
    """

    def __init__(self, spec, variant: str, cfg, kind: str | None = None):
        super().__init__(spec, variant, cfg)
        self.shrink_kind = kind
        if kind is not None:
            # route the *joint* block through ``_joint_agg``; the typed/end/
            # closure blocks and the content block keep their FULL aggregation.
            self.s2_agg = "marginal_bt2"
        self.register_buffer("shrink_kappa", torch.tensor(0.0), persistent=True)
        self.register_buffer("shrink_abar", torch.tensor(1.0), persistent=True)

    def _content_agg(self, batch, a, agg):
        if self.shrink_kind is None:
            return super()._content_agg(batch, a, agg)
        return super()._content_agg(batch, a, "full")

    def _joint_agg(self, batch, a, agg):  # noqa: D401 - override
        if self.shrink_kind is None or agg == "full":
            return super()._joint_agg(batch, a, agg)
        j_full = super()._joint_agg(batch, a, "full")
        j_marg = super()._joint_agg(batch, a, "marginal_bt2")
        if self.shrink_kind == "force_full":     # frozen endpoint fixture
            return j_full
        if self.shrink_kind == "force_zero":     # frozen endpoint fixture
            return j_marg
        n_atoms = a.shape[0]
        if self.shrink_kind == "uniform":
            coeff = self.shrink_abar.to(a.dtype).expand(n_atoms)
        else:
            cnt = batch.get("path_key_count")
            key_atom = batch["path_atom"]
            if cnt is None or key_atom.numel() == 0:
                coeff = torch.ones(n_atoms, dtype=a.dtype, device=a.device)
            else:
                num = (cnt.to(a.dtype) / (cnt.to(a.dtype) + self.shrink_kappa.to(a.dtype)))
                num = num.clamp(0.0, 1.0)
                tot = a.new_zeros(n_atoms).index_add_(0, key_atom, num)
                den = a.new_zeros(n_atoms).index_add_(
                    0, key_atom, torch.ones_like(num, dtype=a.dtype)
                )
                coeff = torch.where(den > 0, tot / den.clamp(min=1.0), torch.ones_like(den))
        return j_marg + coeff.unsqueeze(1) * (j_full - j_marg)


class RouteLCRC(nn.Module):
    """C1 plus the frozen extension switches selected by ``kind``."""

    def __init__(
        self,
        spec,
        norm: Mapping[str, Any],
        kind: str,
        cfg: Mapping[str, Any] | None = None,
        *,
        variant: str = "C1",
        extra_atom_cols: int = 0,
        dropout: float | None = None,
    ):
        super().__init__()
        self.kind = str(kind)
        self.variant = str(variant)
        self.spec = spec
        self.cfg = dict(C.C1_CFG if cfg is None else {**C.C1_CFG, **dict(cfg)})
        self.extra_atom_cols = int(extra_atom_cols)
        self.shrink_kind = "select" if kind == "s_select" else ("uniform" if kind == "s_uniform" else None)
        self.aux_target = {"a_logp": "wc_logp", "a_mw": "molwt", "a_shuffle": "wc_logp_shuffled"}.get(kind)
        self.context_kind = kind if kind in CONTEXT_KINDS else None
        self.has_chem8 = kind in CHEM8_KINDS or extra_atom_cols == 8
        self.has_contrib8 = kind in CONTRIB_KINDS
        self.dropout_p = float(C.BASE_RECIPE["dropout"]) if dropout is None else float(dropout)

        v = DM.C.VARIANT_DEFS[self.variant]
        self.local_cols, self.rho_cols = DM.C.variant_cols(self.variant)
        width = int(self.cfg["width"])
        a_out = int(self.cfg["atom_out"])

        # --- creation order identical to DissectLCRC(variant) ---------------
        self.atom_encoder = DM.MLP([spec.atom_feat_dim + self.extra_atom_cols, int(self.cfg["atom_hidden"]), a_out])
        self.bond_encoder = DM.MLP([3 * a_out + spec.bond_feat_dim, int(self.cfg["bond_hidden"]), int(self.cfg["bond_out"])])
        self.local = ShrinkLocalContent(spec, self.variant, self.cfg, kind=self.shrink_kind)
        self.drop = nn.Dropout(self.dropout_p)
        self.f = DM.MLP([width, width, 1])
        self.rho = DM.MLP([len(self.rho_cols), int(self.cfg["rho_hidden"]), 1])

        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))

        # --- extensions: private RNG, created after the shared modules -------
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(C.TRAIN_SEED + 771_001)
            if self.has_contrib8:
                self.q = DM.MLP([width, width, 8])
                self.delta = DM.MLP([9, 32, 1])
                nn.init.zeros_(self.delta.lins[-1].weight)
                nn.init.zeros_(self.delta.lins[-1].bias)
            if self.aux_target is not None:
                self.q_aux = DM.MLP([width, 64, 1])
                self.rho_aux = DM.MLP([len(self.rho_cols), 64, 1])
            if self.context_kind is not None:
                self.W_msg = nn.Linear(width + spec.bond_feat_dim, 32)
                self.W_self = nn.Linear(width, 32, bias=False)
                self.W_out = nn.Linear(32, width)
                nn.init.zeros_(self.W_out.weight)
                nn.init.zeros_(self.W_out.bias)

    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def _encode(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        else:
            bond_input = torch.zeros((0, 3 * a.shape[1] + batch["bond_feat"].shape[1]),
                                     dtype=a.dtype, device=a.device)
        b = self.bond_encoder(bond_input)
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        return a, b, scal

    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_aux: bool = False,
        force_p_zero: bool = False,
    ):
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        a, b, scal = self._encode(batch)
        h_clean, local_aux = self.local(batch, a, b, scal)

        # optional learned context (diagnostic branch only)
        if self.context_kind is not None:
            c_v = self._context(batch, h_clean)
            h_clean = h_clean + self.W_out(F.silu(self.W_self(h_clean) + c_v))

        p = 0.0 if force_p_zero else self.dropout_p
        if self.training and p > 0:
            d = F.dropout(h_clean, p=p, training=True)
        else:
            d = h_clean

        width = h_clean.shape[1]
        count = h_clean.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h_clean.dtype)
        )
        denom = count.clamp(min=1.0).unsqueeze(1)

        def _pool(x):
            m1 = x.new_zeros((n_graphs, width)).index_add_(0, atom_batch, x) / denom
            m2 = x.new_zeros((n_graphs, width)).index_add_(0, atom_batch, x * x) / denom
            return m1, m2

        if self.kind == "m_cleanpool":
            mean_h, mean_h2 = _pool(h_clean)
        else:
            mean_h, mean_h2 = _pool(d)
            if self.kind == "m_unbiased" and self.training and p > 0:
                mean_h2 = mean_h2 * (1.0 - p)

        contrib_atom = self.f(d).view(-1)
        contrib = h_clean.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)

        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in).view(-1)

        delta = torch.zeros_like(contrib)
        q_graph = None
        if self.has_contrib8:
            q_atom = self.q(d)
            q_graph = h_clean.new_zeros((n_graphs, 8)).index_add_(0, atom_batch, q_atom)
            delta = self.delta(torch.cat([q_graph, torch.log1p(count).unsqueeze(1)], dim=1)).view(-1)

        y = contrib + rho + delta

        aux_pred = None
        if self.aux_target is not None:
            qa = self.q_aux(d).view(-1)
            qa = h_clean.new_zeros(n_graphs).index_add_(0, atom_batch, qa)
            aux_pred = qa + self.rho_aux(rho_in).view(-1)

        if not return_aux:
            return y

        out = {
            "y": y,
            "contrib": contrib,
            "contrib_atom": contrib_atom,
            "rho": rho,
            "delta": delta,
            "q_graph": q_graph,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "h": d,
            "h_clean": h_clean,
            "count": count,
            "rho_input": rho_in,
            "aux_pred": aux_pred,
            "a": a,
            "b": b,
            **local_aux,
        }
        return out

    # ------------------------------------------------------------------
    def _context(self, batch, h_clean):
        """Diagnostic context c_v (7.6).  One round, no recursion."""
        n_atoms = h_clean.shape[0]
        idx = batch["bond_index"]
        if idx.numel() == 0:
            return h_clean.new_zeros((n_atoms, 32))
        u, v = idx[:, 0], idx[:, 1]
        phi = F.silu(self.W_msg(torch.cat([h_clean[u], batch["bond_feat"]], dim=1)))
        if self.context_kind == "mp_context":
            tot = h_clean.new_zeros((n_atoms, 32)).index_add_(0, v, phi)
            den = h_clean.new_zeros(n_atoms).index_add_(0, v, torch.ones(u.shape[0], dtype=h_clean.dtype))
            return torch.where(den.unsqueeze(1) > 0, tot / den.clamp(min=1.0).unsqueeze(1),
                               torch.zeros_like(tot))
        graph_mean = h_clean.new_zeros((int(batch["n_graphs"]), 32)).index_add_(
            0, batch["bond_batch"], phi
        ) / h_clean.new_zeros(int(batch["n_graphs"])).index_add_(
            0, batch["bond_batch"], torch.ones(u.shape[0], dtype=h_clean.dtype)
        ).clamp(min=1.0).unsqueeze(1)
        return graph_mean[batch["atom_batch"]]


# ---------------------------------------------------------------------------
# construction / state copy
# ---------------------------------------------------------------------------


def cfg_for_kind(kind: str) -> dict[str, Any]:
    cfg = dict(C.C1_CFG)
    if kind == "wide_local":
        cfg["local_hidden"] = 384
    return cfg


def extra_atom_cols(kind: str) -> int:
    return 8 if kind in CHEM8_KINDS else 0


def build_model(
    spec,
    norm: Mapping[str, Any],
    kind: str,
    *,
    variant: str = "C1",
    dropout: float | None = None,
) -> RouteLCRC:
    return RouteLCRC(spec, norm, kind, cfg_for_kind(kind), variant=variant,
                     extra_atom_cols=extra_atom_cols(kind), dropout=dropout)


def c1_reference_state(spec, norm) -> dict[str, torch.Tensor]:
    """Frozen C1 initial state: FULL(seed 0) remapped, shared with every arm."""
    full = DM.build_full_state(spec, norm, C.TRAIN_SEED)
    return DM.remap_state(C.C1_VARIANT, full)


def c1_reference_state_full(spec, norm) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    full = DM.build_full_state(spec, norm, C.TRAIN_SEED)
    return full, DM.remap_state(C.C1_VARIANT, full)


def load_init_state(model: RouteLCRC, base_state: Mapping[str, torch.Tensor], kind: str) -> dict[str, Any]:
    """Copy the shared C1 state; apply the frozen WIDE_LOCAL init surgery."""
    own = model.state_dict()
    copied = 0
    surgery: dict[str, Any] = {}
    with torch.no_grad():
        for key, value in own.items():
            if key not in base_state:
                continue
            src = base_state[key]
            if tuple(src.shape) == tuple(value.shape):
                value.copy_(src.to(value.dtype))
                copied += 1
                continue
            if key == "atom_encoder.lins.0.weight" and src.shape[0] == value.shape[0]:
                value.zero_()
                value[:, : src.shape[1]].copy_(src)
                surgery[key] = "zero_padded_columns"
                copied += 1
                continue
            if key == "local.mlp.lins.0.bias" and kind == "wide_local":
                value.zero_()
                value[: src.shape[0]].copy_(src)
                surgery[key] = "copy_rows_new_rows_zero"
                copied += 1
                continue
            if key == "local.mlp.lins.0.weight" and kind == "wide_local":
                value.zero_()
                value[: src.shape[0]].copy_(src)
                _kaiming_rows_(value, src.shape[0])
                surgery[key] = "copy_rows_then_private_kaiming"
                copied += 1
                continue
            if key == "local.mlp.lins.1.weight" and kind == "wide_local":
                value.zero_()
                value[:, : src.shape[1]].copy_(src)
                surgery[key] = "copy_cols_new_cols_zero"
                copied += 1
                continue
            raise ValueError(f"cannot place base tensor {key}: {tuple(src.shape)} -> {tuple(value.shape)}")
    new_keys = sorted(set(own) - set(base_state))
    return {
        "copied": copied,
        "new_extension_tensors": new_keys,
        "surgery": surgery,
        "n_parameters": model.parameter_count(),
    }


def shared_state_equal(model: RouteLCRC, base_state: Mapping[str, torch.Tensor], kind: str) -> dict[str, Any]:
    """Every shared parameter equals the frozen C1 initial state after surgery."""
    own = model.state_dict()
    checks: dict[str, bool] = {}
    for key, value in own.items():
        if key not in base_state:
            continue
        src = base_state[key]
        if tuple(src.shape) == tuple(value.shape):
            checks[key] = bool(torch.equal(value.cpu(), src.cpu()))
        elif key == "atom_encoder.lins.0.weight":
            checks[key] = bool(torch.equal(value.cpu()[:, : src.shape[1]], src.cpu())) and bool(
                float(value.cpu()[:, src.shape[1]:].abs().max()) == 0.0
            )
        elif key == "local.mlp.lins.0.bias" and kind == "wide_local":
            checks[key] = bool(torch.equal(value.cpu()[: src.shape[0]], src.cpu()))
        elif key == "local.mlp.lins.0.weight" and kind == "wide_local":
            checks[key] = bool(torch.equal(value.cpu()[: src.shape[0]], src.cpu()))
        elif key == "local.mlp.lins.1.weight" and kind == "wide_local":
            checks[key] = bool(torch.equal(value.cpu()[:, : src.shape[1]], src.cpu())) and bool(
                float(value.cpu()[:, src.shape[1]:].abs().max()) == 0.0
            )
        else:
            checks[key] = False
    return {"all_equal": all(checks.values()), "checks": checks, "n_checked": len(checks)}


def init_extensions_private(model: RouteLCRC, base_state: Mapping[str, torch.Tensor]) -> list[str]:
    """Names of the extension tensors that were *not* taken from the base state."""
    return sorted(set(model.state_dict()) - set(base_state))


__all__ = [
    "RouteLCRC",
    "ShrinkLocalContent",
    "build_model",
    "cfg_for_kind",
    "extra_atom_cols",
    "c1_reference_state",
    "c1_reference_state_full",
    "load_init_state",
    "shared_state_equal",
    "init_extensions_private",
]
