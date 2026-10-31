"""BASE and CONTEXT models plus their mechanical audits.

BASE
----
``lcrc_model.LCRC`` (``lipo_v1`` schema, width 64) imported unchanged from the
frozen preparation track, with the v2 ``DropoutAtLocal`` hook attached to
``model.local`` and actually executed (``p = 0.10``, train mode only).

CONTEXT
-------
``LCRCContextual`` keeps every BASE module and adds exactly four ``Linear``
layers.  For each molecule ``g`` in the batch:

    m_het[g] = mean_{i in hetero(g)} h_i          (mean of an empty set is 0)
    m_ch[g]  = mean_{i in carbon_halogen(g)} h_i  (mean of an empty set is 0)
    a_i = tanh(W_atom h_i)
    p_g = tanh(W_het m_het[g])
    c_g = tanh(W_ch  m_ch[g])
    change_i = W_out(a_i * p_g * c_i)

    h_new_i = h_i + change_i   if i is hetero
    h_new_i = h_i              otherwise

    y = sum_i f(h_new_i) + rho(global, mean_i h_i, mean_i h_i^2, log1p(n))

``f`` is the original 64 -> 64 -> 1 nonlinear MLP, ``rho`` reads the original
``h`` only, and no module predicts a separate molecule-level residual.  The
element classes are *fixed element classes*, not HBA/HBD or hydrophobicity
labels.

Shared initialisation: CONTEXT is built with the same seed and every state entry
that also exists in BASE is then overwritten by BASE's value, so all shared
parameters and buffers are bit-identical by construction (and re-verified).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import x1_common as C
import x1_data as XD

import lcrc_data  # noqa: E402
import lcrc_model  # noqa: E402
import v2_models  # noqa: E402


# ---------------------------------------------------------------------------
# dropout hook (imported unchanged from the frozen v2 round)
# ---------------------------------------------------------------------------


def attach_dropout(model: nn.Module, p: float = C.TRAIN_RECIPE["dropout"]):
    """Attach ``v2_models.DropoutAtLocal`` to ``model.local`` (arm A)."""
    return v2_models.DropoutAtLocal(model, "A", float(p))


# ---------------------------------------------------------------------------
# CONTEXT module
# ---------------------------------------------------------------------------


class ElementContextCorrection(nn.Module):
    """Per-molecule element-class context on the hetero-atom contribution input."""

    def __init__(self, spec, width: int = C.WIDTH, dim: int = C.CONTEXT_DIM):
        super().__init__()
        if width != C.WIDTH or dim != C.CONTEXT_DIM:
            raise ValueError(f"CONTEXT requires width={C.WIDTH}, dim={C.CONTEXT_DIM}")
        ids = XD.element_role_ids(spec)
        n_types = len(list(spec.atom_type_metadata))
        het = torch.zeros(n_types, dtype=torch.bool)
        ch = torch.zeros(n_types, dtype=torch.bool)
        het[torch.as_tensor(ids["hetero"], dtype=torch.long)] = True
        ch[torch.as_tensor(ids["carbon_halogen"], dtype=torch.long)] = True
        self.register_buffer("_het_lut", het, persistent=False)
        self.register_buffer("_ch_lut", ch, persistent=False)
        self.width = int(width)
        self.dim = int(dim)
        self.W_atom = nn.Linear(self.width, self.dim)
        self.W_het = nn.Linear(self.width, self.dim)
        self.W_ch = nn.Linear(self.width, self.dim)
        self.W_out = nn.Linear(self.dim, self.width)
        with torch.no_grad():
            self.W_out.weight.normal_(mean=0.0, std=C.W_OUT_INIT_STD)
            self.W_out.bias.zero_()

    def role_masks(self, atom_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._het_lut[atom_type], self._ch_lut[atom_type]

    def forward(
        self,
        h: torch.Tensor,
        atom_type: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        width = h.shape[1]
        het, ch = self.role_masks(atom_type)
        het_idx = torch.nonzero(het, as_tuple=False).squeeze(1)
        ch_idx = torch.nonzero(ch, as_tuple=False).squeeze(1)
        ones = h.new_ones(atom_batch.shape[0])
        count_het = h.new_zeros(n_graphs).index_add_(0, atom_batch[het_idx], ones[het_idx])
        count_ch = h.new_zeros(n_graphs).index_add_(0, atom_batch[ch_idx], ones[ch_idx])
        sum_het = h.new_zeros((n_graphs, width)).index_add_(
            0, atom_batch[het_idx], h.index_select(0, het_idx)
        )
        sum_ch = h.new_zeros((n_graphs, width)).index_add_(
            0, atom_batch[ch_idx], h.index_select(0, ch_idx)
        )
        # Empty-set mean is exactly zero: the numerator is the zero tensor and
        # the denominator is clamped to 1.
        m_het = sum_het / count_het.clamp(min=1.0).unsqueeze(1)
        m_ch = sum_ch / count_ch.clamp(min=1.0).unsqueeze(1)

        a_atom = torch.tanh(self.W_atom(h))
        p_graph = torch.tanh(self.W_het(m_het))
        c_graph = torch.tanh(self.W_ch(m_ch))
        p_atom = p_graph.index_select(0, atom_batch)
        c_atom = c_graph.index_select(0, atom_batch)
        change = self.W_out(a_atom * p_atom * c_atom)

        h_new = h.clone()
        if int(het_idx.numel()) > 0:
            h_new = h_new.index_add(0, het_idx, change.index_select(0, het_idx))
        aux = {
            "m_het": m_het,
            "m_ch": m_ch,
            "p_graph": p_graph,
            "c_graph": c_graph,
            "p_atom": p_atom,
            "c_atom": c_atom,
            "a_atom": a_atom,
            "change": change,
            "het_mask": het,
            "carbon_halogen_mask": ch,
            "count_het": count_het,
            "count_ch": count_ch,
        }
        return h_new, aux


class LCRCContextual(lcrc_model.LCRC):
    """Frozen LCRC-A with the element-context correction on hetero atoms only."""

    def __init__(self, spec, norm: Mapping[str, Any], cfg: Mapping[str, Any] | None = None):
        super().__init__(spec, norm, cfg)
        self.context = ElementContextCorrection(spec, C.WIDTH, C.CONTEXT_DIM)

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        a, b, h = self._encode(batch)
        h_new, ctx = self.context(h, batch["atom_type"], batch["atom_batch"], n_graphs)
        atom_batch = batch["atom_batch"]
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / count.clamp(min=1.0).unsqueeze(1)
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / count.clamp(min=1.0).unsqueeze(1)
        contrib_atom = self.f(h_new).view(-1)
        contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in).view(-1)
        y = contrib + rho
        if not return_aux:
            return y
        aux = {
            "contrib": contrib,
            "rho": rho,
            "contrib_atom": contrib_atom,
            "rho_input": rho_in,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "h_original": h,
            "h_new": h_new,
            "count": count,
            "a_encoded": a,
            "b_encoded": b,
            "f_original_atom": self.f(h).view(-1),
            **ctx,
        }
        return y, aux


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_base(
    spec,
    norm: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
    dropout: float = C.TRAIN_RECIPE["dropout"],
) -> tuple[lcrc_model.LCRC, dict[str, Any]]:
    torch.manual_seed(int(seed))
    model = lcrc_model.LCRC(spec, norm, lcrc_model.DEFAULT_CONFIG)
    hook = attach_dropout(model, dropout)
    model.to(device)
    model.eval()
    return model, {
        "arm": "base",
        "class": "lcrc_model.LCRC",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
    }


def build_context(
    spec,
    norm: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
    dropout: float = C.TRAIN_RECIPE["dropout"],
) -> tuple[LCRCContextual, lcrc_model.LCRC, dict[str, Any]]:
    base, base_info = build_base(spec, norm, seed, "cpu", dropout)
    torch.manual_seed(int(seed))
    model = LCRCContextual(spec, norm, lcrc_model.DEFAULT_CONFIG)
    own = model.state_dict()
    shared = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
        if key in own and tuple(own[key].shape) == tuple(value.shape)
    }
    model.load_state_dict({**own, **shared}, strict=True)
    audit = shared_init_audit(base, model)
    if not audit["all_shared_bitwise_identical"]:
        raise RuntimeError(f"CONTEXT shared initialisation mismatch: {audit['mismatched']}")
    hook = attach_dropout(model, dropout)
    model.to(device)
    model.eval()
    return model, base, {
        **base_info,
        "arm": "context",
        "class": "LCRCContextual(lcrc_model.LCRC + ElementContextCorrection)",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameters_base": parameter_count(base),
        "parameters_new": parameter_count(model) - parameter_count(base),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
        "base_init_state_hash": base_info["init_state_hash"],
        "shared_init": audit,
    }


def build_arm(arm: str, spec, norm: Mapping[str, Any], seed: int, device: Any = "cpu", dropout: float = C.TRAIN_RECIPE["dropout"]):
    if arm == "base":
        model, meta = build_base(spec, norm, seed, device, dropout)
        return model, meta
    if arm == "context":
        model, _base, meta = build_context(spec, norm, seed, device, dropout)
        return model, meta
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# parameter audits
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


def shared_init_audit(base: nn.Module, model: nn.Module) -> dict[str, Any]:
    base_state = base.state_dict()
    model_state = model.state_dict()
    shared_keys: list[str] = []
    mismatched: list[str] = []
    for key, value in base_state.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            shared_keys.append(key)
            if not torch.equal(value, model_state[key]):
                mismatched.append(key)
    return {
        "n_shared_entries": len(shared_keys),
        "all_shared_bitwise_identical": not mismatched,
        "mismatched": mismatched,
        "shared_keys": shared_keys,
        "model_only_keys": sorted(set(model_state) - set(base_state)),
        "base_only_keys": sorted(set(base_state) - set(model_state)),
    }


def new_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if name.startswith("context.")]


# ---------------------------------------------------------------------------
# batch helpers / invariance audits
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def permute_batch_atoms(batch: Mapping[str, Any], generator: torch.Generator) -> dict[str, Any]:
    """Renumber atoms inside every molecule of a collated batch.

    Bond endpoints, neighbour owners, path owners and path endpoints are
    remapped consistently; per-node/per-bond/path content is untouched.
    """
    atom_batch = batch["atom_batch"]
    n_atoms = int(atom_batch.shape[0])
    perm = torch.arange(n_atoms, dtype=torch.long)
    n_graphs = int(batch["n_graphs"])
    for g in range(n_graphs):
        idx = torch.nonzero(atom_batch == g, as_tuple=False).squeeze(1)
        local = torch.randperm(int(idx.numel()), generator=generator)
        perm[idx] = idx[local]
    # ``perm[i]`` is the old atom that becomes new atom ``i`` (gather order).
    # Index tensors therefore need the inverse map old -> new.
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n_atoms, dtype=torch.long)
    out = dict(batch)
    for key in ("atom_feat", "atom_type", "atom_scalar", "atom_batch"):
        out[key] = batch[key].index_select(0, perm)
    out["bond_index"] = inverse[batch["bond_index"]]
    out["nbr_atom"] = inverse[batch["nbr_atom"]]
    out["path_atom"] = inverse[batch["path_atom"]]
    out["path_end"] = inverse[batch["path_end"]]
    return out


def renumber_invariance_audit(model: nn.Module, cache: Mapping[str, Any], rows: Sequence[int], seed: int = 0) -> dict[str, Any]:
    model.eval()
    batch = lcrc_data.collate(cache, [int(r) for r in rows])
    generator = torch.Generator().manual_seed(int(seed) + 484_117)
    permuted = permute_batch_atoms(batch, generator)
    with torch.no_grad():
        y = model(batch).detach().float().cpu()
        y_perm = model(permuted).detach().float().cpu()
    diff = (y - y_perm).abs()
    return {
        "n_graphs": len(rows),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "tolerance": 1.0e-5,
        "passed": bool(float(diff.max()) <= 1.0e-5),
    }


def molecule_reorder_audit(model: nn.Module, cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    model.eval()
    rows = [int(r) for r in rows]
    order = list(reversed(rows))
    batch_a = lcrc_data.collate(cache, rows)
    batch_b = lcrc_data.collate(cache, order)
    with torch.no_grad():
        y_a = model(batch_a).detach().float().cpu().numpy().reshape(-1)
        y_b = model(batch_b).detach().float().cpu().numpy().reshape(-1)
    inverse = np.argsort(np.asarray(order))
    y_b_aligned = y_b[inverse]
    diff = np.abs(y_a - y_b_aligned)
    return {
        "n_graphs": len(rows),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "tolerance": 1.0e-6,
        "passed": bool(float(diff.max()) <= 1.0e-6),
    }


def context_no_mixing_audit(model: LCRCContextual, cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """Per-molecule context must be identical alone and inside a batch."""
    model.eval()
    rows = [int(r) for r in rows]
    batch_all = lcrc_data.collate(cache, rows)
    with torch.no_grad():
        y_all, aux_all = model(batch_all, return_aux=True)
        max_ctx = 0.0
        max_pred = 0.0
        per_graph = []
        for g, row in enumerate(rows):
            single = lcrc_data.collate(cache, [row])
            y_single, aux_single = model(single, return_aux=True)
            d_het = float((aux_all["m_het"][g] - aux_single["m_het"][0]).abs().max())
            d_ch = float((aux_all["m_ch"][g] - aux_single["m_ch"][0]).abs().max())
            d_pred = float((y_all[g] - y_single[0]).abs().max())
            max_ctx = max(max_ctx, d_het, d_ch)
            max_pred = max(max_pred, d_pred)
            per_graph.append({"row": row, "d_m_het": d_het, "d_m_ch": d_ch, "d_pred": d_pred})

    # Perturb molecule 0 only; molecule 1's context must not move.
    perturbed = dict(batch_all)
    atom_batch = batch_all["atom_batch"]
    g0 = torch.nonzero(atom_batch == 0, as_tuple=False).squeeze(1)
    feat = batch_all["atom_feat"].clone()
    feat[g0] = 0.0
    perturbed["atom_feat"] = feat
    with torch.no_grad():
        _, aux_pert = model(perturbed, return_aux=True)
    if len(rows) > 1:
        d_het_pert = float((aux_all["m_het"][1] - aux_pert["m_het"][1]).abs().max())
        d_ch_pert = float((aux_all["m_ch"][1] - aux_pert["m_ch"][1]).abs().max())
        with torch.no_grad():
            y_pert = model(perturbed)
        d_pred_pert = float((y_all[1] - y_pert[1]).abs().max())
    else:
        d_het_pert = d_ch_pert = d_pred_pert = 0.0
    return {
        "rows": rows,
        "max_context_diff_single_vs_batch": max_ctx,
        "max_prediction_diff_single_vs_batch": max_pred,
        "other_molecule_context_diff_after_perturbation": {"m_het": d_het_pert, "m_ch": d_ch_pert},
        "other_molecule_prediction_diff_after_perturbation": d_pred_pert,
        "tolerance": 1.0e-6,
        "passed": bool(max(max_ctx, max_pred, d_het_pert, d_ch_pert, d_pred_pert) <= 1.0e-6),
        "per_graph": per_graph,
    }


def role_mask_audit(spec, model: LCRCContextual, cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    tables = XD.role_tables(spec)
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    selected = []
    for row in rows:
        selected.append(atom_type[int(atom_ptr[int(row)]) : int(atom_ptr[int(row) + 1])])
    atoms = np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)
    expected_het = np.isin(atoms, np.asarray(tables["hetero_type_ids"], dtype=np.int64))
    expected_ch = np.isin(atoms, np.asarray(tables["carbon_halogen_type_ids"], dtype=np.int64))
    lut_het = model.context._het_lut.detach().cpu().numpy()[atoms]
    lut_ch = model.context._ch_lut.detach().cpu().numpy()[atoms]
    # Independent re-derivation from the metadata names (element strings).
    metadata = list(spec.atom_type_metadata)
    by_element_het = np.asarray(
        [str(metadata[int(t)].get("element")) in C.HETERO_ELEMENTS for t in atoms], dtype=bool
    )
    by_element_ch = np.asarray(
        [str(metadata[int(t)].get("element")) in C.CARBON_HALOGEN_ELEMENTS for t in atoms], dtype=bool
    )
    return {
        "role_tables": tables,
        "n_atoms_checked": int(atoms.size),
        "n_hetero_atoms_checked": int(expected_het.sum()),
        "n_carbon_halogen_atoms_checked": int(expected_ch.sum()),
        "hetero_lut_matches_expected": bool(np.array_equal(lut_het, expected_het)),
        "carbon_halogen_lut_matches_expected": bool(np.array_equal(lut_ch, expected_ch)),
        "hetero_matches_element_metadata": bool(np.array_equal(expected_het, by_element_het)),
        "carbon_halogen_matches_element_metadata": bool(np.array_equal(expected_ch, by_element_ch)),
        "unknown_element_type_ids": [
            i for i, meta in enumerate(metadata) if meta.get("element") is None
        ],
        "passed": bool(
            np.array_equal(lut_het, expected_het)
            and np.array_equal(lut_ch, expected_ch)
            and np.array_equal(expected_het, by_element_het)
            and np.array_equal(expected_ch, by_element_ch)
        ),
    }


# ---------------------------------------------------------------------------
# forward-equivalence audits
# ---------------------------------------------------------------------------


def base_matches_original_lcrc(spec, norm: Mapping[str, Any], seed: int, cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    base, _ = build_base(spec, norm, seed, "cpu")
    torch.manual_seed(int(seed))
    reference = lcrc_model.LCRC(spec, norm, lcrc_model.DEFAULT_CONFIG)
    state_equal = {k: bool(torch.equal(v, reference.state_dict()[k])) for k, v in base.state_dict().items() if k in reference.state_dict()}
    key_sets_equal = set(base.state_dict()) == set(reference.state_dict())
    base.eval()
    reference.eval()
    batch = lcrc_data.collate(cache, [int(r) for r in rows])
    with torch.no_grad():
        y_base = base(batch).detach().float().cpu().numpy().reshape(-1)
        y_ref = reference(batch).detach().float().cpu().numpy().reshape(-1)
    diff = np.abs(y_base - y_ref)
    return {
        "n_state_entries_compared": len(state_equal),
        "state_dict_keys_identical": bool(key_sets_equal),
        "state_dict_bitwise_identical": bool(all(state_equal.values())),
        "max_abs_forward_diff": float(diff.max()),
        "tolerance": 0.0,
        "passed": bool(key_sets_equal and all(state_equal.values()) and float(diff.max()) == 0.0),
    }


def zero_out_equivalence_audit(base: nn.Module, context: LCRCContextual, cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    base.eval()
    context.eval()
    batch = lcrc_data.collate(cache, [int(r) for r in rows])
    # Make the shared state exactly identical to BASE before comparing.
    own = context.state_dict()
    shared = {k: v for k, v in base.state_dict().items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    context.load_state_dict({**own, **shared}, strict=True)
    with torch.no_grad():
        y_base = base(batch).detach().float().cpu().numpy().reshape(-1)
        y_init = context(batch).detach().float().cpu().numpy().reshape(-1)
        w = context.context.W_out.weight.detach().clone()
        b = context.context.W_out.bias.detach().clone()
        context.context.W_out.weight.zero_()
        context.context.W_out.bias.zero_()
        y_zero = context(batch).detach().float().cpu().numpy().reshape(-1)
        # Restore.
        context.context.W_out.weight.copy_(w)
        context.context.W_out.bias.copy_(b)
    diff_zero = np.abs(y_base - y_zero)
    diff_init = np.abs(y_base - y_init)
    return {
        "n_graphs": len(rows),
        "max_abs_diff_base_vs_zeroed_context": float(diff_zero.max()),
        "tolerance": 1.0e-5,
        "zeroed_context_matches_base": bool(float(diff_zero.max()) <= 1.0e-5),
        "initial_w_out_is_not_degenerate": bool(float(diff_init.max()) > 0.0),
        "max_abs_diff_base_vs_untouched_context_at_init": float(diff_init.max()),
        "mean_abs_diff_base_vs_untouched_context_at_init": float(diff_init.mean()),
        "passed": bool(float(diff_zero.max()) <= 1.0e-5 and float(diff_init.max()) > 0.0),
    }


def rho_input_audit(model: LCRCContextual, cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """Capture the actual ``rho`` input and reconstruct it from ``h_original``."""
    model.eval()
    batch = lcrc_data.collate(cache, [int(r) for r in rows])
    captured_h: list[torch.Tensor] = []
    captured_rho: list[torch.Tensor] = []
    h_local = model.local.register_forward_hook(
        lambda module, inputs, output: captured_h.append(output.detach().clone())
    )
    rho_pre = model.rho.register_forward_pre_hook(
        lambda module, inputs: captured_rho.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            _y, aux = model(batch, return_aux=True)
    finally:
        h_local.remove()
        rho_pre.remove()
    h_original = captured_h[0]
    actual = captured_rho[0]
    n_graphs = int(batch["n_graphs"])
    atom_batch = batch["atom_batch"]
    width = h_original.shape[1]
    count = h_original.new_zeros(n_graphs).index_add_(
        0, atom_batch, torch.ones_like(atom_batch, dtype=h_original.dtype)
    )
    mean_h = h_original.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h_original) / count.clamp(min=1.0).unsqueeze(1)
    mean_h2 = h_original.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h_original * h_original) / count.clamp(min=1.0).unsqueeze(1)
    g = (batch["global"] - model.global_mean) / model.global_scale
    expected = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
    h_new = aux["h_new"]
    mean_h_new = h_new.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h_new) / count.clamp(min=1.0).unsqueeze(1)
    mean_h2_new = h_new.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h_new * h_new) / count.clamp(min=1.0).unsqueeze(1)
    # Discordance check: h_new-based statistics must differ, so the equality
    # above is a discriminating test rather than a vacuous one.
    change_scale = float((h_new - h_original).abs().max())
    differs_if_h_new = float((mean_h - mean_h_new).abs().max())
    return {
        "n_graphs": len(rows),
        "rho_input_matches_h_original": bool(torch.equal(actual, expected)),
        "max_abs_diff_vs_h_original_reconstruction": float((actual - expected).abs().max()),
        "max_abs_diff_vs_h_new_reconstruction": float((actual - torch.cat([g, mean_h_new, mean_h2_new, torch.log1p(count).unsqueeze(1)], dim=1)).abs().max()),
        "max_abs_h_new_minus_h_original": change_scale,
        "max_abs_mean_h_difference_if_h_new_used": differs_if_h_new,
        "discriminating": bool(change_scale > 0.0 and differs_if_h_new > 0.0),
        "passed": bool(torch.equal(actual, expected) and change_scale > 0.0 and differs_if_h_new > 0.0),
    }


# ---------------------------------------------------------------------------
# prediction / diagnostics
# ---------------------------------------------------------------------------


def predict(model: nn.Module, cache: Mapping[str, Any], device: Any = "cpu", batch_size: int = 256) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(cache["n_molecules"]), int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), int(cache["n_molecules"]))))
            batch = to_device(lcrc_data.collate(cache, rows), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def predict_with_context_disabled(model: LCRCContextual, cache: Mapping[str, Any], device: Any = "cpu", batch_size: int = 256) -> np.ndarray:
    """CONTEXT forward with ``W_out`` forced to zero (the model's own base path)."""
    w = model.context.W_out.weight.detach().clone()
    b = model.context.W_out.bias.detach().clone()
    try:
        with torch.no_grad():
            model.context.W_out.weight.zero_()
            model.context.W_out.bias.zero_()
        return predict(model, cache, device, batch_size)
    finally:
        with torch.no_grad():
            model.context.W_out.weight.copy_(w)
            model.context.W_out.bias.copy_(b)


def context_diagnostics(model: LCRCContextual, cache: Mapping[str, Any], device: Any = "cpu", batch_size: int = 64) -> dict[str, Any]:
    """Magnitude and usage statistics of the learned context correction."""
    model.eval()
    n = int(cache["n_molecules"])
    change_abs: list[np.ndarray] = []
    change_component_abs_all: list[np.ndarray] = []
    change_per_molecule: list[float] = []
    f_delta_abs_hetero: list[np.ndarray] = []
    f_delta_abs_all: list[np.ndarray] = []
    f_delta_sum_per_molecule: list[float] = []
    f_delta_abs_sum_per_molecule: list[float] = []
    n_hetero_atoms = 0
    n_atoms = 0
    n_molecules_with_hetero = 0
    graph_change_mean: list[float] = []
    graph_change_max: list[float] = []
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), n)))
            batch = to_device(lcrc_data.collate(cache, rows), device)
            _y, aux = model(batch, return_aux=True)
            het = aux["het_mask"]
            change = aux["change"].detach().float().cpu()
            change_l2 = change.norm(dim=1).numpy()
            change_component_abs = change.abs().mean(dim=1).numpy()
            f_delta = (aux["contrib_atom"] - aux["f_original_atom"]).detach().float().cpu()
            atom_batch = batch["atom_batch"].detach().cpu()
            het_cpu = het.detach().cpu()
            n_atoms += int(change.shape[0])
            n_hetero_atoms += int(het_cpu.sum())
            for g in range(len(rows)):
                mask = (atom_batch == g).numpy()
                hg = het_cpu[mask].numpy()
                cg = change_l2[mask]
                cg_comp = change_component_abs[mask]
                fg = f_delta[mask].numpy()
                if int(hg.sum()) > 0:
                    n_molecules_with_hetero += 1
                    change_abs.append(cg[hg])
                    change_component_abs_all.append(cg_comp[hg])
                    change_per_molecule.append(float(cg[hg].mean()))
                    graph_change_mean.append(float(cg[hg].mean()))
                    graph_change_max.append(float(cg[hg].max()))
                    f_delta_abs_hetero.append(fg[hg])
                f_delta_abs_all.append(np.abs(fg))
                f_delta_sum_per_molecule.append(float(fg.sum()))
                f_delta_abs_sum_per_molecule.append(float(np.abs(fg).sum()))
    change_all = np.concatenate(change_abs) if change_abs else np.zeros(0, dtype=np.float64)
    change_comp_all = np.concatenate(change_component_abs_all) if change_component_abs_all else np.zeros(0, dtype=np.float64)
    f_het = np.concatenate(f_delta_abs_hetero) if f_delta_abs_hetero else np.zeros(0, dtype=np.float64)
    f_all = np.concatenate(f_delta_abs_all) if f_delta_abs_all else np.zeros(0, dtype=np.float64)
    per_mol = np.asarray(change_per_molecule, dtype=np.float64)
    f_sum = np.asarray(f_delta_sum_per_molecule, dtype=np.float64)
    f_abs_sum = np.asarray(f_delta_abs_sum_per_molecule, dtype=np.float64)
    return {
        "n_molecules": n,
        "n_atoms": n_atoms,
        "n_hetero_atoms": n_hetero_atoms,
        "n_molecules_with_hetero": n_molecules_with_hetero,
        "fraction_molecules_with_hetero": n_molecules_with_hetero / max(n, 1),
        "n_atoms_with_nonzero_change": int(np.sum(change_all > 0.0)),
        "change_l2_mean_over_hetero_atoms": float(change_all.mean()) if change_all.size else 0.0,
        "change_l2_max_over_hetero_atoms": float(change_all.max()) if change_all.size else 0.0,
        "change_component_abs_mean_over_hetero_atoms": float(change_comp_all.mean()) if change_comp_all.size else 0.0,
        "change_l2_sum_over_all_atoms_divided_by_n_atoms": float(change_all.sum() / max(n_atoms, 1)),
        "change_l2_per_molecule_mean_of_means": float(per_mol.mean()) if per_mol.size else 0.0,
        "change_l2_per_molecule_median": float(np.median(per_mol)) if per_mol.size else 0.0,
        "change_l2_per_molecule_p95": float(np.quantile(per_mol, 0.95)) if per_mol.size else 0.0,
        "change_l2_per_molecule_max": float(per_mol.max()) if per_mol.size else 0.0,
        "graph_change_l2_max_max": float(np.max(graph_change_max)) if graph_change_max else 0.0,
        "f_delta_abs_mean_over_hetero_atoms": float(f_het.mean()) if f_het.size else 0.0,
        "f_delta_abs_max_over_hetero_atoms": float(f_het.max()) if f_het.size else 0.0,
        "f_delta_abs_mean_over_all_atoms": float(f_all.mean()) if f_all.size else 0.0,
        "f_delta_sum_per_molecule_mean": float(f_sum.mean()) if f_sum.size else 0.0,
        "f_delta_abs_sum_per_molecule_mean": float(f_abs_sum.mean()) if f_abs_sum.size else 0.0,
        "definition_change": (
            "change_i = h_new_i - h_original_i is a 64-d vector; the reported magnitude is "
            "||change_i||_2 per atom, averaged over atoms that receive the correction (hetero)"
        ),
        "definition_change_component_abs": "per hetero atom mean_j |change_ij|, then averaged",
        "definition_f_delta": "|f(h_new) - f(h_original)| per atom, then averaged",
    }


def gradient_probe(
    model: LCRCContextual,
    cache: Mapping[str, Any],
    rows: Sequence[int],
    *,
    steps: int = 2,
    lr: float = C.TRAIN_RECIPE["peak_lr"],
    device: Any = "cpu",
) -> dict[str, Any]:
    """Two real optimizer steps on real Train batches; report new-layer gradients."""
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=0.0)
    names = new_parameter_names(model)
    before = {name: dict(model.named_parameters())[name].detach().clone() for name in names}
    history: list[dict[str, Any]] = []
    n = int(cache["n_molecules"])
    for step in range(int(steps)):
        rows_step = [int(rows[(step * 16 + i) % len(rows)]) for i in range(min(16, len(rows)))]
        batch = to_device(lcrc_data.collate(cache, rows_step), device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        target = torch.zeros_like(pred)
        loss = F.mse_loss(pred, target)
        loss.backward()
        grads = {}
        for name, param in model.named_parameters():
            if name in names:
                g = param.grad
                grads[name] = {
                    "finite": bool(g is not None and torch.isfinite(g).all()),
                    "norm": float(g.detach().norm()) if g is not None else 0.0,
                    "max_abs": float(g.detach().abs().max()) if g is not None else 0.0,
                }
        history.append({"step": step + 1, "loss": float(loss.detach()), "grads": grads})
        optimizer.step()
    delta = {
        name: float((dict(model.named_parameters())[name].detach() - before[name]).abs().max())
        for name in names
    }
    after = {row["step"]: row["grads"] for row in history}
    last = after[max(after)]
    nonzero_after_2 = all(last[name]["norm"] > 0.0 for name in names if name.endswith(".weight") or name.endswith(".bias"))
    return {
        "steps": int(steps),
        "new_parameter_names": names,
        "history": history,
        "all_new_layers_finite_gradient": all(
            grads["finite"] for row in history for grads in row["grads"].values()
        ),
        "all_new_tensors_nonzero_gradient_at_last_probe_step": bool(nonzero_after_2),
        "nonzero_gradient_at_step_2": bool(
            all(after.get(2, after[max(after)])[name]["norm"] > 0.0 for name in names)
        ),
        "parameter_max_abs_update": delta,
        "all_new_tensors_updated": bool(all(v > 0.0 for v in delta.values())),
        "passed": bool(
            all(grads["finite"] for row in history for grads in row["grads"].values())
            and all(after.get(2, after[max(after)])[name]["norm"] > 0.0 for name in names)
            and all(v > 0.0 for v in delta.values())
        ),
    }


def dropout_activity_audit(model: nn.Module, cache: Mapping[str, Any], rows: Sequence[int], generator_seed: int = 0) -> dict[str, Any]:
    """The hook must be active in train mode and the identity in eval mode."""
    model.train()
    batch = lcrc_data.collate(cache, [int(r) for r in rows])
    torch.manual_seed(int(generator_seed) + 12_345)
    with torch.no_grad():
        y_train_1 = model(batch).detach().float().cpu().numpy().reshape(-1)
    torch.manual_seed(int(generator_seed) + 99_999)
    with torch.no_grad():
        y_train_2 = model(batch).detach().float().cpu().numpy().reshape(-1)
    model.eval()
    with torch.no_grad():
        y_eval_1 = model(batch).detach().float().cpu().numpy().reshape(-1)
        y_eval_2 = model(batch).detach().float().cpu().numpy().reshape(-1)
    hook = getattr(model, "_v2_dropout_head", None)
    return {
        "dropout_p": float(hook.p) if hook is not None else None,
        "hook_module_present": hook is not None,
        "train_mode_forwards_differ": bool(np.abs(y_train_1 - y_train_2).max() > 0.0),
        "eval_mode_forwards_identical": bool(np.abs(y_eval_1 - y_eval_2).max() == 0.0),
        "passed": bool(
            hook is not None
            and float(hook.p) == C.TRAIN_RECIPE["dropout"]
            and np.abs(y_train_1 - y_train_2).max() > 0.0
            and np.abs(y_eval_1 - y_eval_2).max() == 0.0
        ),
    }


__all__ = [
    "attach_dropout",
    "ElementContextCorrection",
    "LCRCContextual",
    "build_base",
    "build_context",
    "build_arm",
    "parameter_count",
    "parameter_breakdown",
    "shared_init_audit",
    "new_parameter_names",
    "to_device",
    "permute_batch_atoms",
    "renumber_invariance_audit",
    "molecule_reorder_audit",
    "context_no_mixing_audit",
    "role_mask_audit",
    "base_matches_original_lcrc",
    "zero_out_equivalence_audit",
    "rho_input_audit",
    "predict",
    "predict_with_context_disabled",
    "context_diagnostics",
    "gradient_probe",
    "dropout_activity_audit",
]
