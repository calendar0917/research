"""Model definitions and mechanical audits for the mechanism check v2.

BASE and CONTEXT are imported *unchanged* from the frozen v1 round
(``x1_models.build_base`` / ``x1_models.build_context``), including the exact
same ``lcrc_model.LCRC`` and ``ElementContextCorrection`` classes and the same
``v2_models.DropoutAtLocal`` hook (p = 0.10, executed in train mode).

ROLEBLIND keeps every CONTEXT module and parameter.  The only change is the
source of the two per-molecule context vectors:

    CONTEXT   m_het = mean_{i in hetero} h_i ;  m_ch = mean_{i in carbon_halogen} h_i
    ROLEBLIND m_all = mean_{i in all atoms} h_i ; m_het = m_ch = m_all

Everything else is unchanged: ``a_i = tanh(W_atom h_i)``,
``p_g = tanh(W_het m_het)``, ``c_g = tanh(W_ch m_ch)``,
``change_i = W_out(a_i * p_g * c_g)``, the correction is added only to hetero
atoms, ``f`` and ``rho`` are untouched and ``rho`` still reads the original
``h``.  ROLEBLIND has exactly the same state-dict keys, shapes and
parameter count as CONTEXT (255,458).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import m2_common as C
import m2_data as MD

import lcrc_model  # noqa: E402
import x1_models as X1M  # noqa: E402


# ---------------------------------------------------------------------------
# ROLEBLIND context path
# ---------------------------------------------------------------------------


def roleblind_context_forward(
    module: X1M.ElementContextCorrection,
    h: torch.Tensor,
    atom_type: torch.Tensor,
    atom_batch: torch.Tensor,
    n_graphs: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """All-atom context correction using CONTEXT's own four Linear layers.

    ``module`` is a normal ``ElementContextCorrection``; only the aggregation is
    replaced, so the parameter set is exactly CONTEXT's.
    """
    width = h.shape[1]
    het, ch = module.role_masks(atom_type)
    het_idx = torch.nonzero(het, as_tuple=False).squeeze(1)

    ones = h.new_ones(atom_batch.shape[0])
    count_all = h.new_zeros(n_graphs).index_add_(0, atom_batch, ones)
    sum_all = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h)
    m_all = sum_all / count_all.clamp(min=1.0).unsqueeze(1)

    # The only definitional change relative to CONTEXT.
    m_het = m_all
    m_ch = m_all

    a_atom = torch.tanh(module.W_atom(h))
    p_graph = torch.tanh(module.W_het(m_het))
    c_graph = torch.tanh(module.W_ch(m_ch))
    p_atom = p_graph.index_select(0, atom_batch)
    c_atom = c_graph.index_select(0, atom_batch)
    change = module.W_out(a_atom * p_atom * c_atom)

    h_new = h.clone()
    if int(het_idx.numel()) > 0:
        h_new = h_new.index_add(0, het_idx, change.index_select(0, het_idx))
    aux = {
        "m_all": m_all,
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
        "count_all": count_all,
        "count_het": None,
        "count_ch": None,
    }
    return h_new, aux


class LCRCContextualRoleBlind(X1M.LCRCContextual):
    """CONTEXT with all-atom context aggregation; identical modules and keys."""

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        a, b, h = self._encode(batch)
        h_new, ctx = roleblind_context_forward(
            self.context, h, batch["atom_type"], batch["atom_batch"], n_graphs
        )
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


def build_base(spec, norm: Mapping[str, Any], seed: int, device: Any = "cpu", dropout: float = C.TRAIN_RECIPE["dropout"]):
    return X1M.build_base(spec, norm, seed, device, dropout)


def build_context(spec, norm: Mapping[str, Any], seed: int, device: Any = "cpu", dropout: float = C.TRAIN_RECIPE["dropout"]):
    return X1M.build_context(spec, norm, seed, device, dropout)


def build_roleblind(spec, norm: Mapping[str, Any], seed: int, device: Any = "cpu", dropout: float = C.TRAIN_RECIPE["dropout"]):
    """Build CONTEXT at ``seed``, then copy its complete state into ROLEBLIND."""
    context, base, context_meta = X1M.build_context(spec, norm, seed, "cpu", dropout)
    torch.manual_seed(int(seed))
    model = LCRCContextualRoleBlind(spec, norm, lcrc_model.DEFAULT_CONFIG)
    model.load_state_dict(context.state_dict(), strict=True)
    parity = init_parity(context, model)
    if not parity["all_entries_bitwise_identical"]:
        raise RuntimeError(f"ROLEBLIND initialisation differs from CONTEXT: {parity['mismatched']}")
    hook = X1M.attach_dropout(model, dropout)
    model.to(device)
    model.eval()
    meta = {
        **context_meta,
        "family": "roleblind",
        "arm": "roleblind",
        "class": "LCRCContextualRoleBlind(x1_models.LCRCContextual, all-atom context)",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameters_base": parameter_count(base),
        "parameters_new": parameter_count(model) - parameter_count(base),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
        "context_init_state_hash": context_meta["init_state_hash"],
        "base_init_state_hash": context_meta["base_init_state_hash"],
        "init_parity_with_context": parity,
        "context_definition": (
            "hetero-only h_new = h + W_out(tanh(W_atom h) * tanh(W_het m_all) * tanh(W_ch m_all)); "
            "m_all = mean over all atoms of the molecule; y = sum f(h_new) + rho(original h statistics)"
        ),
    }
    return model, context, meta


def build_family(
    family: str,
    spec,
    norm: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
    dropout: float = C.TRAIN_RECIPE["dropout"],
):
    if family == "base":
        model, meta = build_base(spec, norm, seed, device, dropout)
        return model, meta
    if family == "context":
        model, _base, meta = build_context(spec, norm, seed, device, dropout)
        return model, meta
    if family == "roleblind":
        model, _context, meta = build_roleblind(spec, norm, seed, device, dropout)
        return model, meta
    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------------------
# parameter / init audits
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


def init_parity(reference: nn.Module, candidate: nn.Module) -> dict[str, Any]:
    """Every state entry (parameter or persistent buffer) must be bit-identical."""
    ref_state = reference.state_dict()
    cand_state = candidate.state_dict()
    mismatched: list[str] = []
    for key, value in ref_state.items():
        if key not in cand_state:
            mismatched.append(f"missing:{key}")
        elif tuple(cand_state[key].shape) != tuple(value.shape):
            mismatched.append(f"shape:{key}")
        elif not torch.equal(value, cand_state[key]):
            mismatched.append(f"value:{key}")
    for key in cand_state:
        if key not in ref_state:
            mismatched.append(f"extra:{key}")
    ref_params = dict(reference.named_parameters())
    cand_params = dict(candidate.named_parameters())
    param_mismatched: list[str] = []
    for name, value in ref_params.items():
        if name not in cand_params:
            param_mismatched.append(f"missing:{name}")
        elif tuple(cand_params[name].shape) != tuple(value.shape):
            param_mismatched.append(f"shape:{name}")
        elif not torch.equal(value, cand_params[name]):
            param_mismatched.append(f"value:{name}")
    for name in cand_params:
        if name not in ref_params:
            param_mismatched.append(f"extra:{name}")
    return {
        "n_state_entries_reference": len(ref_state),
        "n_state_entries_candidate": len(cand_state),
        "state_dict_keys_identical": set(ref_state) == set(cand_state),
        "all_entries_bitwise_identical": not mismatched,
        "mismatched": mismatched,
        "n_parameters_reference": len(ref_params),
        "n_parameters_candidate": len(cand_params),
        "parameter_names_identical": set(ref_params) == set(cand_params),
        "all_parameters_bitwise_identical": not param_mismatched,
        "parameter_mismatched": param_mismatched,
        "parameter_count_reference": parameter_count(reference),
        "parameter_count_candidate": parameter_count(candidate),
        "parameter_counts_equal": parameter_count(reference) == parameter_count(candidate),
    }


def shared_init_audit(reference: nn.Module, candidate: nn.Module) -> dict[str, Any]:
    return X1M.shared_init_audit(reference, candidate)


def new_parameter_names(model: nn.Module) -> list[str]:
    return X1M.new_parameter_names(model)


# ---------------------------------------------------------------------------
# device / batch helpers
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return X1M.to_device(batch, device)


def model_device(model: nn.Module) -> torch.device:
    return X1M.model_device(model)


# ---------------------------------------------------------------------------
# ROLEBLIND-specific mechanical audits
# ---------------------------------------------------------------------------


def roleblind_aggregation_audit(
    model: LCRCContextualRoleBlind,
    cache: Mapping[str, Any],
    rows: Sequence[int],
    device: Any = "cpu",
) -> dict[str, Any]:
    """Verify m_het = m_ch = exact per-molecule all-atom mean of h."""
    import lcrc_data

    model.eval()
    device = model_device(model) if device is None else device
    batch = to_device(lcrc_data.collate(cache, [int(r) for r in rows]), device)
    with torch.no_grad():
        _y, aux = model(batch, return_aux=True)
    h = aux["h_original"]
    atom_batch = batch["atom_batch"]
    n_graphs = int(batch["n_graphs"])
    het = aux["het_mask"]
    manual_all = []
    manual_het = []
    manual_count = []
    for g in range(n_graphs):
        mask = atom_batch == g
        hg = h[mask]
        manual_all.append(hg.mean(dim=0))
        manual_count.append(int(mask.sum()))
        hg_het = h[mask & het]
        if int(hg_het.shape[0]) > 0:
            manual_het.append(hg_het.mean(dim=0))
        else:
            manual_het.append(torch.zeros_like(hg[0]))
    manual_all_t = torch.stack(manual_all)
    manual_het_t = torch.stack(manual_het)
    manual_count_t = torch.as_tensor(manual_count, dtype=aux["count_all"].dtype, device=device)
    d_all_het = float((aux["m_het"] - manual_all_t).abs().max())
    d_all_ch = float((aux["m_ch"] - manual_all_t).abs().max())
    d_het_vs_allatom = float((manual_het_t - manual_all_t).abs().max())
    per_graph_het_vs_all = [
        float((manual_het_t[g] - manual_all_t[g]).abs().max()) for g in range(n_graphs)
    ]
    p_expected = torch.tanh(model.context.W_het(manual_all_t))
    c_expected = torch.tanh(model.context.W_ch(manual_all_t))
    d_p = float((aux["p_graph"] - p_expected).abs().max())
    d_c = float((aux["c_graph"] - c_expected).abs().max())
    non_het_delta = float((aux["h_new"] - h)[~het].abs().max()) if int((~het).sum()) > 0 else 0.0
    het_delta = float((aux["h_new"] - h)[het].abs().max()) if int(het.sum()) > 0 else 0.0
    passed = bool(
        d_all_het <= 1.0e-6
        and d_all_ch <= 1.0e-6
        and torch.equal(aux["m_het"], aux["m_ch"])
        and torch.equal(aux["m_het"], aux["m_all"])
        and torch.equal(aux["count_all"], manual_count_t)
        and d_p <= 1.0e-6
        and d_c <= 1.0e-6
        and non_het_delta == 0.0
        and d_het_vs_allatom > 0.0
    )
    return {
        "rows": [int(r) for r in rows],
        "n_graphs": n_graphs,
        "m_het_equals_all_atom_mean_max_abs_diff": d_all_het,
        "m_ch_equals_all_atom_mean_max_abs_diff": d_all_ch,
        "m_het_equals_m_ch_bitwise": bool(torch.equal(aux["m_het"], aux["m_ch"])),
        "m_het_equals_m_all_bitwise": bool(torch.equal(aux["m_het"], aux["m_all"])),
        "count_all_matches_manual": bool(torch.equal(aux["count_all"], manual_count_t)),
        "p_graph_equals_tanh_W_het_m_all_max_abs_diff": d_p,
        "c_graph_equals_tanh_W_ch_m_all_max_abs_diff": d_c,
        "all_atom_mean_vs_hetero_only_mean_max_abs_diff": d_het_vs_allatom,
        "per_graph_all_atom_mean_vs_hetero_only_mean_max_abs_diff": per_graph_het_vs_all,
        "n_graphs_where_all_atom_mean_differs_from_hetero_mean": int(
            sum(1 for d in per_graph_het_vs_all if d > 0.0)
        ),
        "non_hetero_atom_delta_max_abs": non_het_delta,
        "hetero_atom_delta_max_abs": het_delta,
        "tolerance": 1.0e-6,
        "passed": passed,
    }


def roleblind_batch_invariance_audit(
    model: LCRCContextualRoleBlind,
    cache: Mapping[str, Any],
    rows: Sequence[int],
    device: Any = "cpu",
    *,
    subset_size: int = 32,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Prediction must not depend on batch composition or graph order."""
    import lcrc_data

    model.eval()
    device = model_device(model) if device is None else device
    rows = [int(r) for r in rows]
    selected = rows[: min(int(subset_size), len(rows))]
    single: dict[int, float] = {}
    with torch.no_grad():
        for row in selected:
            batch = to_device(lcrc_data.collate(cache, [row]), device)
            single[row] = float(model(batch).detach().float().cpu().reshape(-1)[0])

    def _batch_pred(order: Sequence[int]) -> dict[int, float]:
        order = [int(r) for r in order]
        batch = to_device(lcrc_data.collate(cache, order), device)
        with torch.no_grad():
            pred = model(batch).detach().float().cpu().numpy().reshape(-1)
        return {row: float(value) for row, value in zip(order, pred)}

    full = _batch_pred(selected)
    reversed_order = _batch_pred(list(reversed(selected)))
    even = _batch_pred(selected[::2])
    odd = _batch_pred(selected[1::2])
    # Target rows inside a mixed batch that also contains unrelated molecules.
    anchor = selected[: min(8, len(selected))]
    extra = [int(r) for r in rows if int(r) not in set(anchor)][:16]
    mixed = _batch_pred(extra + anchor) if extra else _batch_pred(anchor)

    def _max_diff(a: Mapping[int, float], b: Mapping[int, float], keys: Sequence[int]) -> float:
        return max((abs(a[k] - b[k]) for k in keys), default=0.0)

    diffs = {
        "single_vs_full_batch": _max_diff(single, full, selected),
        "single_vs_reversed_batch": _max_diff(single, reversed_order, selected),
        "single_vs_even_subset": _max_diff(single, even, selected[::2]),
        "single_vs_odd_subset": _max_diff(single, odd, selected[1::2]),
        "single_vs_mixed_batch_with_unrelated_graphs": _max_diff(single, mixed, anchor),
        "full_batch_vs_reversed": _max_diff(full, reversed_order, selected),
    }
    passed = bool(max(diffs.values()) <= float(tolerance))
    return {
        "n_rows_single": len(selected),
        "batch_sizes": {
            "single": 1,
            "full": len(selected),
            "reversed": len(selected),
            "even": len(selected[::2]),
            "odd": len(selected[1::2]),
            "mixed": len(mixed),
        },
        "max_abs_prediction_differences": diffs,
        "tolerance": float(tolerance),
        "passed": passed,
    }


def roleblind_usage(
    model: LCRCContextualRoleBlind,
    cache: Mapping[str, Any],
    device: Any = "cpu",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Descriptive: how different the all-atom mean is from the hetero mean."""
    import lcrc_data

    model.eval()
    device = model_device(model) if device is None else device
    n = int(cache["n_molecules"])
    max_all_vs_het: list[float] = []
    mean_all_vs_het: list[float] = []
    n_atoms = 0
    n_het_atoms = 0
    n_graphs_all_ne_het = 0
    p_c_input_equal = True
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), n)))
            batch = to_device(lcrc_data.collate(cache, rows), device)
            _y, aux = model(batch, return_aux=True)
            h = aux["h_original"]
            het = aux["het_mask"]
            atom_batch = batch["atom_batch"]
            n_atoms += int(h.shape[0])
            n_het_atoms += int(het.sum())
            p_c_input_equal = p_c_input_equal and bool(torch.equal(aux["m_het"], aux["m_ch"]))
            for g in range(len(rows)):
                mask = atom_batch == g
                hg = h[mask]
                m_all = hg.mean(dim=0)
                hg_het = h[mask & het]
                m_het = hg_het.mean(dim=0) if int(hg_het.shape[0]) > 0 else torch.zeros_like(hg[0])
                d = float((m_all - m_het).abs().max())
                max_all_vs_het.append(d)
                mean_all_vs_het.append(float((m_all - m_het).abs().mean()))
                if d > 0.0:
                    n_graphs_all_ne_het += 1
    return {
        "n_molecules": n,
        "n_atoms": n_atoms,
        "n_hetero_atoms": n_het_atoms,
        "m_het_input_equals_m_ch_input_for_every_batch": p_c_input_equal,
        "n_molecules_where_all_atom_mean_differs_from_hetero_mean": n_graphs_all_ne_het,
        "max_all_atom_vs_hetero_mean_abs_diff": float(max(max_all_vs_het)) if max_all_vs_het else 0.0,
        "mean_over_molecules_of_mean_abs_all_vs_hetero": float(np.mean(mean_all_vs_het)) if mean_all_vs_het else 0.0,
        "definition": "compare m_all = mean(h over all atoms) with mean(h over hetero atoms) of the same molecule",
    }


# ---------------------------------------------------------------------------
# prediction / checkpoint helpers
# ---------------------------------------------------------------------------


def predict(model: nn.Module, cache: Mapping[str, Any], device: Any = "cpu", batch_size: int = 256) -> np.ndarray:
    import lcrc_data

    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(cache["n_molecules"]), int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), int(cache["n_molecules"]))))
            batch = to_device(lcrc_data.collate(cache, rows), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def load_checkpoint(
    model: nn.Module,
    path: Any,
    *,
    expected_sha256: str | None = None,
    expected_family: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    path = __import__("pathlib").Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        C.verify_sha256(path, expected_sha256, f"checkpoint {path.name}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_family is not None and payload.get("family") not in (None, expected_family) and payload.get("arm") not in (None, expected_family):
        # Historical v1 checkpoints store ``arm`` = base/context, not ``family``.
        if payload.get("arm") != expected_family:
            raise PermissionError(f"checkpoint family {payload.get('family') or payload.get('arm')} != {expected_family}")
    if expected_seed is not None:
        stored_seed = payload.get("seed")
        if stored_seed is not None and int(stored_seed) != int(expected_seed):
            raise PermissionError(f"checkpoint seed {stored_seed} != {expected_seed}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return payload


def freeze_weights_audit(
    model: nn.Module,
    cache: Mapping[str, Any],
    device: Any = "cpu",
) -> dict[str, Any]:
    """Inference must not change a single weight or create a gradient."""
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    requires_grad_before = {name: bool(p.requires_grad) for name, p in model.named_parameters()}
    model.eval()
    pred = predict(model, cache, device=device)
    after = model.state_dict()
    changed = [key for key, value in before.items() if not torch.equal(value, after[key].detach())]
    grads = [name for name, p in model.named_parameters() if p.grad is not None]
    requires_grad_after = {name: bool(p.requires_grad) for name, p in model.named_parameters()}
    return {
        "n_state_entries": len(before),
        "n_changed_state_entries": len(changed),
        "changed_state_entries": changed,
        "parameters_with_grad_after_inference": grads,
        "requires_grad_flags_unchanged": requires_grad_before == requires_grad_after,
        "model_training_after_inference": bool(model.training),
        "predictions_finite": bool(np.isfinite(pred).all()),
        "n_predictions": int(pred.size),
        "passed": bool(
            not changed
            and not grads
            and requires_grad_before == requires_grad_after
            and not model.training
            and np.isfinite(pred).all()
        ),
    }


def roleblind_readout_parity_audit(
    context: X1M.LCRCContextual,
    roleblind: LCRCContextualRoleBlind,
    cache: Mapping[str, Any],
    rows: Sequence[int],
    device: Any = "cpu",
) -> dict[str, Any]:
    """With identical weights and identical context output, readouts must agree.

    ``context.context.forward`` is temporarily replaced by
    ``roleblind_context_forward`` bound to the same module, so both models
    receive exactly the same ``h_new``; any difference can only come from the
    readout code path (``f``, ``rho``, ``contrib``).
    """
    import lcrc_data

    original = context.context.forward
    try:
        context.context.forward = lambda h, atom_type, atom_batch, n_graphs: roleblind_context_forward(
            context.context, h, atom_type, atom_batch, n_graphs
        )
        context.eval()
        roleblind.eval()
        batch = to_device(lcrc_data.collate(cache, [int(r) for r in rows]), device)
        with torch.no_grad():
            y_context = context(batch).detach().float().cpu().numpy().reshape(-1)
            y_roleblind = roleblind(batch).detach().float().cpu().numpy().reshape(-1)
    finally:
        context.context.forward = original
    diff = float(np.abs(y_context - y_roleblind).max())
    return {
        "n_graphs": len(rows),
        "max_abs_prediction_diff": diff,
        "tolerance": 1.0e-6,
        "passed": bool(diff <= 1.0e-6),
        "note": "same weights, same roleblind context output; only the readout path is compared",
    }


__all__ = [
    "roleblind_context_forward",
    "LCRCContextualRoleBlind",
    "build_base",
    "build_context",
    "build_roleblind",
    "build_family",
    "parameter_count",
    "parameter_breakdown",
    "init_parity",
    "shared_init_audit",
    "new_parameter_names",
    "to_device",
    "model_device",
    "roleblind_aggregation_audit",
    "roleblind_batch_invariance_audit",
    "roleblind_usage",
    "predict",
    "load_checkpoint",
    "freeze_weights_audit",
    "roleblind_readout_parity_audit",
]
