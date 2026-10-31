"""Models for the LCRC composition-discovery round.

Every model keeps the original LCRC encoder, S1/S2 local content, per-atom
contribution MLP ``f`` and global readout ``rho``.  ``rho`` always reads the
original ``h``; only the input of ``f`` (or an explicit scalar contribution)
is corrected.

Families
--------
g2l_all / g2l_loo / g2l_local_cap
    The four ROLEBLIND layers with the global context replaced by all-atom,
    leave-one-out, or purely local context.
g2l_scalar
    ROLEBLIND-style gating with a scalar per-hetero-atom correction; ``h`` is
    never rewritten.
dist_true / dist_blind
    Per-atom D34/D5 context.  ``dist_blind`` keeps the same bucket masks and
    parameter set but replaces each bucket's content with the molecule's
    leave-one-out all-atom mean.
path_bind / path_unbound
    Decomposed S2 path-role embeddings bound to the true path-endpoint content
    (``mean(v*r)``) or to its two marginals (``mean(v)*mean(r)``).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import dc_common as C

import lcrc_model  # noqa: E402
import x1_models as X1M  # noqa: E402
import m2_models as M2M  # noqa: E402

G2L_MODES = {
    "g2l_all": "all",
    "g2l_loo": "loo",
    "g2l_local_cap": "local_cap",
}


# ---------------------------------------------------------------------------
# generic readout
# ---------------------------------------------------------------------------


def standard_readout(model: lcrc_model.LCRC, batch: Mapping[str, torch.Tensor], h: torch.Tensor, h_new: torch.Tensor, delta: torch.Tensor | None = None):
    n_graphs = int(batch["n_graphs"])
    atom_batch = batch["atom_batch"]
    width = h.shape[1]
    count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
    mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / count.clamp(min=1.0).unsqueeze(1)
    mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / count.clamp(min=1.0).unsqueeze(1)
    contrib_atom = model.f(h_new).view(-1)
    contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
    g = (batch["global"] - model.global_mean) / model.global_scale
    rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
    rho = model.rho(rho_in).view(-1)
    y = contrib + rho
    if delta is not None:
        y = y + delta
    aux = {
        "contrib": contrib,
        "rho": rho,
        "contrib_atom": contrib_atom,
        "f_original_atom": model.f(h).view(-1),
        "rho_input": rho_in,
        "mean_h": mean_h,
        "mean_h2": mean_h2,
        "h_original": h,
        "h_new": h_new,
        "count": count,
    }
    return y, aux


def _leave_one_out_mean(h: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    width = h.shape[1]
    ones = h.new_ones(atom_batch.shape[0])
    count = h.new_zeros(n_graphs).index_add_(0, atom_batch, ones)
    total = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h)
    numerator = total.index_select(0, atom_batch) - h
    denom = (count.index_select(0, atom_batch) - 1.0).clamp(min=1.0)
    out = numerator / denom.unsqueeze(1)
    singleton = (count.index_select(0, atom_batch) <= 1).unsqueeze(1)
    return torch.where(singleton, torch.zeros_like(out), out)


def _all_atom_mean(h: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    width = h.shape[1]
    count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
    total = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h)
    return total / count.clamp(min=1.0).unsqueeze(1)


# ---------------------------------------------------------------------------
# A family
# ---------------------------------------------------------------------------


def g2l_context_forward(module: X1M.ElementContextCorrection, h, atom_type, atom_batch, n_graphs, mode: str):
    width = h.shape[1]
    het, ch = module.role_masks(atom_type)
    het_idx = torch.nonzero(het, as_tuple=False).squeeze(1)
    a_atom = torch.tanh(module.W_atom(h))

    if mode == "all":
        m_all = _all_atom_mean(h, atom_batch, n_graphs)
        p_graph = torch.tanh(module.W_het(m_all))
        c_graph = torch.tanh(module.W_ch(m_all))
        p_atom = p_graph.index_select(0, atom_batch)
        c_atom = c_graph.index_select(0, atom_batch)
        change = module.W_out(a_atom * p_atom * c_atom)
        h_new = h + change
        ctx = {"m_all": m_all, "p_graph": p_graph, "c_graph": c_graph, "p_atom": p_atom, "c_atom": c_atom}
    elif mode in ("loo", "local_cap"):
        if mode == "loo":
            context_atom = _leave_one_out_mean(h, atom_batch, n_graphs)
        else:
            context_atom = h
        p_atom = torch.tanh(module.W_het(context_atom))
        c_atom = torch.tanh(module.W_ch(context_atom))
        change = module.W_out(a_atom * p_atom * c_atom)
        h_new = h.clone()
        if int(het_idx.numel()):
            h_new = h_new.index_add(0, het_idx, change.index_select(0, het_idx))
        ctx = {"p_atom": p_atom, "c_atom": c_atom}
        if mode == "loo":
            ctx["m_except"] = context_atom
    else:
        raise ValueError(mode)

    ctx.update(
        {
            "a_atom": a_atom,
            "change": change,
            "het_mask": het,
            "carbon_halogen_mask": ch,
            "update_scope": "all_atoms" if mode == "all" else "hetero_atoms_only",
        }
    )
    return h_new, ctx


class LCRCG2L(X1M.LCRCContextual):
    """ROLEBLIND modules; context aggregation selected by ``mode``."""

    def __init__(self, spec, norm, cfg=None, mode: str = "all"):
        super().__init__(spec, norm, cfg)
        self.mode = str(mode)

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        a, b, h = self._encode(batch)
        h_new, ctx = g2l_context_forward(
            self.context, h, batch["atom_type"], batch["atom_batch"], int(batch["n_graphs"]), self.mode
        )
        y, aux = standard_readout(self, batch, h, h_new)
        if not return_aux:
            return y
        aux.update({"a_encoded": a, "b_encoded": b, **ctx})
        return y, aux


class ScalarContextCorrection(nn.Module):
    def __init__(self, spec, width: int = C.WIDTH, dim: int = C.CORR_DIM):
        super().__init__()
        ids = X1M.XD.element_role_ids(spec) if hasattr(X1M, "XD") else None
        n_types = len(list(spec.atom_type_metadata))
        het = torch.zeros(n_types, dtype=torch.bool)
        if ids is None:
            from x1_data import element_role_ids

            ids = element_role_ids(spec)
        het[torch.as_tensor(ids["hetero"], dtype=torch.long)] = True
        self.register_buffer("_het_lut", het, persistent=False)
        self.W_atom = nn.Linear(width, dim)
        self.W_het = nn.Linear(width, dim)
        self.W_ch = nn.Linear(width, dim)
        self.w_out = nn.Linear(dim, 1)
        with torch.no_grad():
            self.w_out.weight.normal_(mean=0.0, std=C.W_OUT_INIT_STD)
            self.w_out.bias.zero_()

    def forward(self, h, atom_type, atom_batch, n_graphs):
        het = self._het_lut[atom_type]
        m_all = _all_atom_mean(h, atom_batch, n_graphs)
        a_atom = torch.tanh(self.W_atom(h))
        p_graph = torch.tanh(self.W_het(m_all))
        c_graph = torch.tanh(self.W_ch(m_all))
        p_atom = p_graph.index_select(0, atom_batch)
        c_atom = c_graph.index_select(0, atom_batch)
        delta_atom = self.w_out(a_atom * p_atom * c_atom).view(-1) * het.to(h.dtype)
        delta = h.new_zeros(n_graphs).index_add_(0, atom_batch, delta_atom)
        return delta, {
            "delta_atom": delta_atom,
            "m_all": m_all,
            "a_atom": a_atom,
            "p_graph": p_graph,
            "c_graph": c_graph,
            "p_atom": p_atom,
            "c_atom": c_atom,
            "het_mask": het,
        }


class LCRCG2LScalar(lcrc_model.LCRC):
    def __init__(self, spec, norm, cfg=None):
        super().__init__(spec, norm, cfg)
        self.context = ScalarContextCorrection(spec, C.WIDTH, C.CORR_DIM)

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        a, b, h = self._encode(batch)
        delta, ctx = self.context(h, batch["atom_type"], batch["atom_batch"], int(batch["n_graphs"]))
        y, aux = standard_readout(self, batch, h, h, delta=delta)
        if not return_aux:
            return y
        aux.update({"a_encoded": a, "b_encoded": b, "delta": delta, **ctx})
        return y, aux


# ---------------------------------------------------------------------------
# B family
# ---------------------------------------------------------------------------


class DistanceCorrection(nn.Module):
    def __init__(self, width: int = C.WIDTH, dim: int = C.CORR_DIM):
        super().__init__()
        self.W_atom = nn.Linear(width, dim)
        self.W_34 = nn.Linear(width, dim)
        self.W_5 = nn.Linear(width, dim)
        self.W_out = nn.Linear(dim, width)
        with torch.no_grad():
            self.W_out.weight.normal_(mean=0.0, std=C.W_OUT_INIT_STD)
            self.W_out.bias.zero_()

    def forward(self, h, batch, n_graphs, mode: str):
        n_atoms = h.shape[0]
        width = h.shape[1]
        d34_idx = batch["d34_idx"]
        d5_idx = batch["d5_idx"]
        c34_counts = batch["d34_counts"]
        c5_counts = batch["d5_counts"]
        has34 = c34_counts > 0
        has5 = c5_counts > 0

        m_except = _leave_one_out_mean(h, batch["atom_batch"], n_graphs)
        sum34 = h.new_zeros((n_atoms, width)).index_add_(0, d34_idx, h.index_select(0, d34_idx))
        sum5 = h.new_zeros((n_atoms, width)).index_add_(0, d5_idx, h.index_select(0, d5_idx))
        c34 = sum34 / c34_counts.clamp(min=1).unsqueeze(1)
        c5 = sum5 / c5_counts.clamp(min=1).unsqueeze(1)
        if mode == "blind":
            c34 = torch.where(has34.unsqueeze(1), m_except, torch.zeros_like(c34))
            c5 = torch.where(has5.unsqueeze(1), m_except, torch.zeros_like(c5))
        elif mode != "true":
            raise ValueError(mode)

        u = torch.tanh(self.W_atom(h))
        p34 = torch.tanh(self.W_34(c34)) * has34.to(h.dtype).unsqueeze(1)
        p5 = torch.tanh(self.W_5(c5)) * has5.to(h.dtype).unsqueeze(1)
        change = self.W_out(u * (p34 + p5))
        both_empty = (~has34) & (~has5)
        change = torch.where(both_empty.unsqueeze(1), torch.zeros_like(change), change)
        return change, {
            "c34": c34,
            "c5": c5,
            "m_except": m_except,
            "p34": p34,
            "p5": p5,
            "u": u,
            "has34": has34,
            "has5": has5,
            "both_empty": both_empty,
            "d34_counts": c34_counts,
            "d5_counts": c5_counts,
        }


class LCRCDistance(lcrc_model.LCRC):
    def __init__(self, spec, norm, cfg=None, mode: str = "true"):
        super().__init__(spec, norm, cfg)
        self.dist = DistanceCorrection(C.WIDTH, C.CORR_DIM)
        self.mode = str(mode)

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        a, b, h = self._encode(batch)
        change, ctx = self.dist(h, batch, int(batch["n_graphs"]), self.mode)
        h_new = h + change
        y, aux = standard_readout(self, batch, h, h_new)
        if not return_aux:
            return y
        aux.update({"a_encoded": a, "b_encoded": b, "change": change, **ctx})
        return y, aux


# ---------------------------------------------------------------------------
# C family
# ---------------------------------------------------------------------------


class PathBindingCorrection(nn.Module):
    def __init__(self, atom_out: int = C.ATOM_OUT, width: int = C.WIDTH, dim: int = C.CORR_DIM):
        super().__init__()
        self.E_mid = nn.Embedding(16, dim)
        self.E_b1 = nn.Embedding(5, dim)
        self.E_b2 = nn.Embedding(5, dim)
        self.E_closure = nn.Embedding(2, dim)
        self.W_end = nn.Linear(atom_out, dim)
        self.W_root = nn.Linear(width, dim)
        self.W_out = nn.Linear(dim, width)
        with torch.no_grad():
            self.W_out.weight.normal_(mean=0.0, std=C.W_OUT_INIT_STD)
            self.W_out.bias.zero_()

    def relation(self, path_cat, path_closure):
        mid = torch.div(path_cat, 25, rounding_mode="floor")
        b1 = torch.div(path_cat.remainder(25), 5, rounding_mode="floor")
        b2 = path_cat.remainder(5)
        return torch.tanh(self.E_mid(mid) + self.E_b1(b1) + self.E_b2(b2) + self.E_closure(path_closure))

    def forward(self, h, a, batch, mode: str):
        n_atoms = h.shape[0]
        dim = self.W_root.out_features
        path_atom = batch["path_atom"]
        n_paths = int(path_atom.shape[0])
        counts = h.new_zeros(n_atoms).index_add_(0, path_atom, torch.ones_like(path_atom, dtype=h.dtype)) if n_paths else h.new_zeros(n_atoms)
        if n_paths:
            r = self.relation(batch["path_cat"], batch["path_closure"])
            v = torch.tanh(self.W_end(a.index_select(0, batch["path_end"])))
            if mode == "bind":
                prod = v * r
                total = h.new_zeros((n_atoms, dim)).index_add_(0, path_atom, prod)
                bind = total / counts.clamp(min=1.0).unsqueeze(1)
                mean_v = None
                mean_r = None
            elif mode == "unbound":
                sum_v = h.new_zeros((n_atoms, dim)).index_add_(0, path_atom, v)
                sum_r = h.new_zeros((n_atoms, dim)).index_add_(0, path_atom, r)
                mean_v = sum_v / counts.clamp(min=1.0).unsqueeze(1)
                mean_r = sum_r / counts.clamp(min=1.0).unsqueeze(1)
                bind = mean_v * mean_r
            else:
                raise ValueError(mode)
        else:
            r = h.new_zeros((0, dim))
            v = h.new_zeros((0, dim))
            bind = h.new_zeros((n_atoms, dim))
            mean_v = None
            mean_r = None
        u = torch.tanh(self.W_root(h))
        change = self.W_out(u * bind)
        has_path = counts > 0
        change = torch.where(has_path.unsqueeze(1), change, torch.zeros_like(change))
        return change, {
            "r": r,
            "v": v,
            "mean_v": mean_v,
            "mean_r": mean_r,
            "bind": bind,
            "u": u,
            "path_counts": counts,
            "has_path": has_path,
        }


class LCRCPathBinding(lcrc_model.LCRC):
    def __init__(self, spec, norm, cfg=None, mode: str = "bind"):
        super().__init__(spec, norm, cfg)
        self.pathbind = PathBindingCorrection(C.ATOM_OUT, C.WIDTH, C.CORR_DIM)
        self.mode = str(mode)

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        a, b, h = self._encode(batch)
        change, ctx = self.pathbind(h, a, batch, self.mode)
        h_new = h + change
        y, aux = standard_readout(self, batch, h, h_new)
        if not return_aux:
            return y
        aux.update({"a_encoded": a, "b_encoded": b, "change": change, **ctx})
        return y, aux


# ---------------------------------------------------------------------------
# builders / audits
# ---------------------------------------------------------------------------


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def parameter_breakdown(module: nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, child in module.named_children():
        if name == "_v2_dropout_head":
            continue
        out[name] = parameter_count(child)
    out["total"] = parameter_count(module)
    return out


def _load_shared_from_base(model: nn.Module, base: nn.Module) -> dict[str, Any]:
    own = model.state_dict()
    shared = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
        if key in own and tuple(own[key].shape) == tuple(value.shape)
    }
    model.load_state_dict({**own, **shared}, strict=True)
    audit = X1M.shared_init_audit(base, model)
    if not audit["all_shared_bitwise_identical"]:
        raise RuntimeError(f"shared initialisation mismatch: {audit['mismatched']}")
    return audit


def _attach(model, dropout):
    hook = X1M.attach_dropout(model, float(dropout))
    model.eval()
    return hook


def _build_g2l(family: str, spec, norm, seed: int, device, dropout: float):
    context, base, ctx_meta = X1M.build_context(spec, norm, seed, "cpu", dropout)
    mode = G2L_MODES[family]
    model = LCRCG2L(spec, norm, lcrc_model.DEFAULT_CONFIG, mode=mode)
    model.load_state_dict(context.state_dict(), strict=True)
    hook = _attach(model, dropout)
    parity = M2M.init_parity(context, model)
    if not parity["all_entries_bitwise_identical"]:
        raise RuntimeError(f"{family} init differs from CONTEXT: {parity['mismatched']}")
    meta = {
        "family": family,
        "arm": family,
        "class": f"LCRCG2L(mode={mode})",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameters_base": int(ctx_meta["parameters_base"]),
        "parameters_new": parameter_count(model) - int(ctx_meta["parameters_base"]),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
        "base_init_state_hash": ctx_meta["base_init_state_hash"],
        "context_init_state_hash": ctx_meta["init_state_hash"],
        "init_parity_with_context": parity,
        "shared_init": X1M.shared_init_audit(base, model),
        "definition": f"ROLEBLIND four layers, context mode={mode}",
    }
    model.to(device)
    return model, meta


def _build_g2l_scalar(spec, norm, seed: int, device, dropout: float):
    base, base_meta = X1M.build_base(spec, norm, seed, "cpu", dropout)
    context, _base2, ctx_meta = X1M.build_context(spec, norm, seed, "cpu", dropout)
    torch.manual_seed(int(seed))
    model = LCRCG2LScalar(spec, norm, lcrc_model.DEFAULT_CONFIG)
    shared = _load_shared_from_base(model, base)
    with torch.no_grad():
        model.context.W_atom.load_state_dict(context.context.W_atom.state_dict())
        model.context.W_het.load_state_dict(context.context.W_het.state_dict())
        model.context.W_ch.load_state_dict(context.context.W_ch.state_dict())
    hook = _attach(model, dropout)
    meta = {
        "family": "g2l_scalar",
        "arm": "g2l_scalar",
        "class": "LCRCG2LScalar",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameters_base": int(base_meta["parameters"]),
        "parameters_new": parameter_count(model) - int(base_meta["parameters"]),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
        "base_init_state_hash": base_meta["init_state_hash"],
        "context_init_state_hash": ctx_meta["init_state_hash"],
        "shared_init": shared,
        "definition": "h unchanged; delta_i=w_out(tanh(W_atom h_i)*tanh(W_het m_all)*tanh(W_ch m_all)) on hetero atoms",
    }
    model.to(device)
    return model, meta


def _build_distance(family: str, spec, norm, seed: int, device, dropout: float):
    base, base_meta = X1M.build_base(spec, norm, seed, "cpu", dropout)
    if family == "dist_true":
        torch.manual_seed(int(seed))
        model = LCRCDistance(spec, norm, lcrc_model.DEFAULT_CONFIG, mode="true")
        shared = _load_shared_from_base(model, base)
    elif family == "dist_blind":
        true_model, true_meta = _build_distance("dist_true", spec, norm, seed, "cpu", dropout)
        torch.manual_seed(int(seed))
        model = LCRCDistance(spec, norm, lcrc_model.DEFAULT_CONFIG, mode="blind")
        model.load_state_dict(true_model.state_dict(), strict=True)
        shared = true_meta["shared_init"]
    else:
        raise ValueError(family)
    hook = _attach(model, dropout)
    meta = {
        "family": family,
        "arm": family,
        "class": f"LCRCDistance(mode={model.mode})",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameters_base": int(base_meta["parameters"]),
        "parameters_new": parameter_count(model) - int(base_meta["parameters"]),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
        "base_init_state_hash": base_meta["init_state_hash"],
        "shared_init": shared,
        "definition": "h_new_i=h_i+W_out(tanh(W_atom h_i)*(p34_i+p5_i)); p bucket = mask*tanh(W_bucket mean bucket content)",
    }
    model.to(device)
    return model, meta


def _build_path(family: str, spec, norm, seed: int, device, dropout: float):
    base, base_meta = X1M.build_base(spec, norm, seed, "cpu", dropout)
    if family == "path_bind":
        torch.manual_seed(int(seed))
        model = LCRCPathBinding(spec, norm, lcrc_model.DEFAULT_CONFIG, mode="bind")
        shared = _load_shared_from_base(model, base)
    elif family == "path_unbound":
        true_model, true_meta = _build_path("path_bind", spec, norm, seed, "cpu", dropout)
        torch.manual_seed(int(seed))
        model = LCRCPathBinding(spec, norm, lcrc_model.DEFAULT_CONFIG, mode="unbound")
        model.load_state_dict(true_model.state_dict(), strict=True)
        shared = true_meta["shared_init"]
    else:
        raise ValueError(family)
    hook = _attach(model, dropout)
    meta = {
        "family": family,
        "arm": family,
        "class": f"LCRCPathBinding(mode={model.mode})",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameters_base": int(base_meta["parameters"]),
        "parameters_new": parameter_count(model) - int(base_meta["parameters"]),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
        "base_init_state_hash": base_meta["init_state_hash"],
        "shared_init": shared,
        "definition": "bind_i=mean_p(v_p*r_p) for PATH-BIND; mean_p(v_p)*mean_p(r_p) for PATH-UNBOUND",
    }
    model.to(device)
    return model, meta


def build_family(family: str, spec, norm, seed: int, device="cpu", dropout: float = C.TRAIN_RECIPE["dropout"]):
    if family == "base":
        model, meta = X1M.build_base(spec, norm, seed, device, dropout)
        return model, meta
    if family == "context":
        model, _base, meta = X1M.build_context(spec, norm, seed, device, dropout)
        return model, meta
    if family == "roleblind":
        model, _context, meta = M2M.build_roleblind(spec, norm, seed, device, dropout)
        return model, meta
    if family in G2L_MODES:
        return _build_g2l(family, spec, norm, seed, device, dropout)
    if family == "g2l_scalar":
        return _build_g2l_scalar(spec, norm, seed, device, dropout)
    if family in ("dist_true", "dist_blind"):
        return _build_distance(family, spec, norm, seed, device, dropout)
    if family in ("path_bind", "path_unbound"):
        return _build_path(family, spec, norm, seed, device, dropout)
    raise ValueError(f"unknown family {family!r}")


def predict(model, cache, device="cpu", batch_size: int = 256):
    import dc_data

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, int(cache["n_molecules"]), int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), int(cache["n_molecules"]))))
            batch = to_device(dc_data.collate(cache, rows), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    import numpy as np

    return np.concatenate(out).astype(np.float64)


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_checkpoint(model, path, *, expected_sha256=None, expected_family=None, expected_seed=None):
    from pathlib import Path
    import dc_common as CC

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        CC.sha256_file(path) == expected_sha256 or (_ for _ in ()).throw(PermissionError("checkpoint sha mismatch"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_family is not None:
        stored = payload.get("family", payload.get("arm"))
        if stored is not None and str(stored) != str(expected_family):
            raise PermissionError(f"checkpoint family {stored} != {expected_family}")
    if expected_seed is not None and payload.get("seed") is not None and int(payload["seed"]) != int(expected_seed):
        raise PermissionError(f"checkpoint seed {payload['seed']} != {expected_seed}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return payload


def new_parameter_prefix(family: str) -> str:
    return {
        "context": "context.",
        "roleblind": "context.",
        "g2l_all": "context.",
        "g2l_loo": "context.",
        "g2l_local_cap": "context.",
        "g2l_scalar": "context.",
        "dist_true": "dist.",
        "dist_blind": "dist.",
        "path_bind": "pathbind.",
        "path_unbound": "pathbind.",
    }[family]


def new_parameter_names(model, family: str) -> list[str]:
    prefix = new_parameter_prefix(family)
    return [name for name, _ in model.named_parameters() if name.startswith(prefix)]


__all__ = [
    "standard_readout",
    "LCRCG2L",
    "LCRCG2LScalar",
    "LCRCDistance",
    "LCRCPathBinding",
    "build_family",
    "predict",
    "to_device",
    "model_device",
    "load_checkpoint",
    "parameter_count",
    "parameter_breakdown",
    "new_parameter_names",
    "g2l_context_forward",
]
