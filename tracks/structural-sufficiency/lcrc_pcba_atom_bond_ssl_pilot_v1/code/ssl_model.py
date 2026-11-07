"""The masked atom/bond pretext network and the encoder-only transfer.

Only two pre-existing modules are trained here:

``atom_encoder``  : ``MLP([51, 128, 96])``   (Linear -> SiLU -> Linear)
``bond_encoder``  : ``MLP([3*96 + 13, 160, 96])`` with the frozen undirected
                    input ``[a_u + a_v, |a_u - a_v|, a_u * a_v, bond_feat]``

Both are instantiated exactly as the historical seed-0 LCRC builder does and
filled with the historical seed-0 tensors, so pretraining starts from the very
initialisation the downstream scratch A would have used.

The pretext computation is deliberately **static**: one atom/bond encode at
each masked view, incidence sums/means over the known physical topology, and a
two-layer MLP head.  There is no iterative message passing, no atom-to-atom
QKV attention and no hidden-state round.

The two reconstruction heads are temporary.  Only ``atom_encoder.*`` and
``bond_encoder.*`` are ever exported, and the downstream model never needs a
pretext head to run.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import ssl_common as C
import nmp_model as NM
import tu_model as TU

# ---------------------------------------------------------------------------
# static encoding (the frozen LCRC atom/bond encoder pair)
# ---------------------------------------------------------------------------


def encode_static(
    atom_encoder: nn.Module,
    bond_encoder: nn.Module,
    atom_feat: torch.Tensor,
    bond_feat: torch.Tensor,
    bond_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One static atom encode + one static bond encode (no propagation)."""
    a = atom_encoder(atom_feat)
    if bond_index.numel():
        au, av = a[bond_index[:, 0]], a[bond_index[:, 1]]
        bond_input = torch.cat([au + av, (au - av).abs(), au * av, bond_feat], dim=1)
    else:
        bond_input = a.new_zeros((0, 3 * a.shape[1] + bond_feat.shape[1]))
    b = bond_encoder(bond_input)
    return a, b


def incidence_sums(
    a: torch.Tensor, b: torch.Tensor, bond_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-atom degree, sum of incident bond embeddings, sum of neighbour atoms."""
    n_atoms = a.shape[0]
    degree = a.new_zeros(n_atoms)
    sum_b = a.new_zeros((n_atoms, b.shape[1]))
    sum_a = a.new_zeros((n_atoms, a.shape[1]))
    if bond_index.numel():
        u, v = bond_index[:, 0], bond_index[:, 1]
        ones = a.new_ones(u.shape)
        degree = degree.index_add(0, u, ones).index_add(0, v, ones)
        sum_b = sum_b.index_add(0, u, b).index_add(0, v, b)
        sum_a = sum_a.index_add(0, u, a[v]).index_add(0, v, a[u])
    return degree, sum_b, sum_a


def atom_context(
    a: torch.Tensor,
    b: torch.Tensor,
    bond_index: torch.Tensor,
    atom_ids: torch.Tensor,
) -> torch.Tensor:
    """``[a_v, mean_b(v), mean_a(v)]`` for the selected atoms (288-d)."""
    degree, sum_b, sum_a = incidence_sums(a, b, bond_index)
    if atom_ids.numel() == 0:
        return a.new_zeros((0, a.shape[1] + b.shape[1] + a.shape[1]))
    d = degree[atom_ids].clamp(min=1.0).unsqueeze(1)
    return torch.cat([a[atom_ids], sum_b[atom_ids] / d, sum_a[atom_ids] / d], dim=1)


def bond_context(
    a: torch.Tensor,
    b: torch.Tensor,
    bond_index: torch.Tensor,
    bond_ids: torch.Tensor,
) -> torch.Tensor:
    """``[b_e, ctx_u + ctx_v, |ctx_u - ctx_v|]`` for the selected bonds (480-d)."""
    width = a.shape[1] + 2 * (2 * a.shape[1])
    if bond_ids.numel() == 0:
        return a.new_zeros((0, width))
    degree, sum_b, sum_a = incidence_sums(a, b, bond_index)
    u = bond_index[bond_ids, 0]
    v = bond_index[bond_ids, 1]
    b_e = b[bond_ids]

    def _side(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        deg = degree[x]
        excl_b = sum_b[x] - b_e
        excl_a = sum_a[x] - a[y]
        keep = (deg > 1.0).to(a.dtype).unsqueeze(1)
        denom = (deg - 1.0).clamp(min=1.0).unsqueeze(1)
        return excl_b / denom * keep, excl_a / denom * keep

    mb_u, ma_u = _side(u, v)
    mb_v, ma_v = _side(v, u)
    ctx_u = torch.cat([mb_u, ma_u], dim=1)
    ctx_v = torch.cat([mb_v, ma_v], dim=1)
    return torch.cat([b_e, ctx_u + ctx_v, (ctx_u - ctx_v).abs()], dim=1)


# ---------------------------------------------------------------------------
# pretext network
# ---------------------------------------------------------------------------


class PretextNet(nn.Module):
    """Two frozen encoders + two temporary reconstruction heads."""

    def __init__(self, atom_encoder: nn.Module, bond_encoder: nn.Module):
        super().__init__()
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder
        self.atom_head = nn.Sequential(
            nn.Linear(C.ATOM_HEAD_INPUT, C.HEAD_HIDDEN),
            nn.SiLU(),
            nn.Linear(C.HEAD_HIDDEN, C.N_ATOM_CLASSES),
        )
        self.bond_head = nn.Sequential(
            nn.Linear(C.BOND_HEAD_INPUT, C.HEAD_HIDDEN),
            nn.SiLU(),
            nn.Linear(C.HEAD_HIDDEN, C.N_BOND_CLASSES),
        )

    # -- views -------------------------------------------------------------
    def atom_view_outputs(self, view: Mapping[str, torch.Tensor], atom_ids: torch.Tensor):
        a, b = encode_static(
            self.atom_encoder, self.bond_encoder, view["atom_feat"], view["bond_feat"], view["bond_index"]
        )
        ctx = atom_context(a, b, view["bond_index"], atom_ids)
        return a, b, ctx

    def bond_view_outputs(self, view: Mapping[str, torch.Tensor], bond_ids: torch.Tensor):
        a, b = encode_static(
            self.atom_encoder, self.bond_encoder, view["atom_feat"], view["bond_feat"], view["bond_index"]
        )
        ctx = bond_context(a, b, view["bond_index"], bond_ids)
        return a, b, ctx

    def forward(
        self,
        atom_view: Mapping[str, torch.Tensor],
        bond_view: Mapping[str, torch.Tensor],
        atom_ids: torch.Tensor,
        bond_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        a_atom, b_atom, atom_ctx = self.atom_view_outputs(atom_view, atom_ids)
        a_bond, b_bond, bond_ctx = self.bond_view_outputs(bond_view, bond_ids)
        atom_logits = self.atom_head(atom_ctx)
        bond_logits = self.bond_head(bond_ctx)
        return {
            "atom_logits": atom_logits,
            "bond_logits": bond_logits,
            "atom_ctx": atom_ctx,
            "bond_ctx": bond_ctx,
            "a_atom_view": a_atom,
            "b_atom_view": b_atom,
            "a_bond_view": a_bond,
            "b_bond_view": b_bond,
        }

    def loss(
        self,
        atom_view: Mapping[str, torch.Tensor],
        bond_view: Mapping[str, torch.Tensor],
        atom_ids: torch.Tensor,
        bond_ids: torch.Tensor,
        atom_targets: torch.Tensor,
        bond_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self(atom_view, bond_view, atom_ids, bond_ids)
        if atom_targets.numel() == 0 and bond_targets.numel() == 0:
            raise ValueError("empty pretext batch: no masked atom or bond target")
        zero = out["a_atom_view"].new_zeros(())
        ce_atom = (
            F.cross_entropy(out["atom_logits"], atom_targets) if atom_targets.numel() else None
        )
        ce_bond = (
            F.cross_entropy(out["bond_logits"], bond_targets) if bond_targets.numel() else None
        )
        loss = zero
        if ce_atom is not None:
            loss = loss + ce_atom
        if ce_bond is not None:
            loss = loss + ce_bond
        stats: dict[str, torch.Tensor] = {"loss": loss}
        if ce_atom is not None:
            stats["ce_atom"] = ce_atom.detach()
            stats["atom_correct"] = (out["atom_logits"].argmax(1) == atom_targets).sum().detach()
            stats["atom_n"] = torch.as_tensor(atom_targets.numel(), device=atom_targets.device)
        if ce_bond is not None:
            stats["ce_bond"] = ce_bond.detach()
            stats["bond_correct"] = (out["bond_logits"].argmax(1) == bond_targets).sum().detach()
            stats["bond_n"] = torch.as_tensor(bond_targets.numel(), device=bond_targets.device)
        return loss, stats


# ---------------------------------------------------------------------------
# construction from the historical seed-0 initialisation
# ---------------------------------------------------------------------------


def seed0_scratch_state(spec: Any, norm: Mapping[str, Any], *, device: str = "cpu") -> dict[str, Any]:
    """Build the historical seed-0 scratch A and return its full state."""
    ref, ref_info, base_init = NM.build_reference(spec, norm, C.N_TASKS, C.SSL_SEED)
    model, audit = NM.build_arm(
        "A",
        spec,
        norm,
        C.N_TASKS,
        seed=C.SSL_SEED,
        device=device,
        base_init_state=base_init,
        zero_residual=True,
    )
    return {
        "model": model,
        "audit": audit,
        "reference": ref,
        "reference_info": ref_info,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }


def make_encoders(spec: Any, state: Mapping[str, torch.Tensor]) -> tuple[nn.Module, nn.Module]:
    """Fresh encoder modules with exactly the frozen historical geometry."""
    atom_encoder = TU.MLP([C.ATOM_FEAT_DIM, C.ATOM_HIDDEN, C.ATOM_OUT])
    bond_encoder = TU.MLP(
        [3 * C.ATOM_OUT + C.BOND_FEAT_DIM, C.BOND_HIDDEN, C.BOND_OUT]
    )
    atom_encoder.load_state_dict(
        {k[len("atom_encoder.") :]: v for k, v in state.items() if k.startswith("atom_encoder.")},
        strict=True,
    )
    bond_encoder.load_state_dict(
        {k[len("bond_encoder.") :]: v for k, v in state.items() if k.startswith("bond_encoder.")},
        strict=True,
    )
    return atom_encoder, bond_encoder


def init_heads(model: PretextNet, *, seed: int = C.HEAD_INIT_SEED) -> dict[str, Any]:
    """Private-generator head init; the encoders and the global RNG are untouched."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    before = torch.get_rng_state().clone()
    enc_before = encoder_fingerprint(model)
    n_linear = 0
    with torch.no_grad():
        for head in (model.atom_head, model.bond_head):
            for sub in head.modules():
                if isinstance(sub, nn.Linear):
                    bound = 1.0 / math.sqrt(float(sub.in_features))
                    sub.weight.uniform_(-bound, bound, generator=gen)
                    sub.bias.uniform_(-bound, bound, generator=gen)
                    n_linear += 1
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("pretext head initialisation changed the global torch RNG state")
    if encoder_fingerprint(model) != enc_before:
        raise RuntimeError("pretext head initialisation changed the encoders")
    return {
        "private_generator_seed": int(seed),
        "linear_layers_initialised": int(n_linear),
        "global_rng_unchanged": True,
        "encoder_state_unchanged": True,
    }


def build_pretext(spec: Any, norm: Mapping[str, Any]) -> tuple[PretextNet, dict[str, Any]]:
    """Pretext model whose encoders are the historical seed-0 LCRC-A encoders."""
    scratch = seed0_scratch_state(spec, norm)
    state = scratch["state"]
    atom_encoder, bond_encoder = make_encoders(spec, state)
    model = PretextNet(atom_encoder, bond_encoder)
    init_audit = init_heads(model)
    enc = encoder_fingerprint(model)
    ref = encoder_fingerprint_from_state(state)
    if enc != ref:
        raise RuntimeError("pretext encoders do not match the historical seed-0 state")
    audit = {
        "encoder_fingerprint": enc,
        "historical_seed0_encoder_fingerprint": ref,
        "encoder_state_matches_seed0_scratch": True,
        "encoder_parameters": int(sum(p.numel() for p in atom_encoder.parameters())
                                   + sum(p.numel() for p in bond_encoder.parameters())),
        "atom_encoder_parameters": int(sum(p.numel() for p in atom_encoder.parameters())),
        "bond_encoder_parameters": int(sum(p.numel() for p in bond_encoder.parameters())),
        "atom_head_input": int(model.atom_head[0].in_features),
        "bond_head_input": int(model.bond_head[0].in_features),
        "atom_head_hidden": int(model.atom_head[0].out_features),
        "bond_head_hidden": int(model.bond_head[0].out_features),
        "atom_head_classes": int(model.atom_head[2].out_features),
        "bond_head_classes": int(model.bond_head[2].out_features),
        "head_init": init_audit,
        "total_pretext_parameters": C.parameter_count(model),
        "downstream_parameters_of_scratch_A": int(scratch["audit"]["total_parameter_count"]),
        "dropout_modules": [n for n, m in model.named_modules() if isinstance(m, nn.Dropout)],
        "iterative_message_passing_modules": [],
        "attention_modules": [n for n, m in model.named_modules() if "attention" in type(m).__name__.lower()],
        "reads": ["atom_feat", "bond_feat", "bond_index"],
        "does_not_read": [
            "atom_type",
            "bond_type",
            "atom_scalar",
            "global",
            "nbr_code",
            "path_cat",
            "path_end",
            "path_closure",
            "n_components",
            "n_junctions",
            "n_ring_systems",
        ],
    }
    return model, audit


# ---------------------------------------------------------------------------
# encoder state helpers
# ---------------------------------------------------------------------------


def encoder_state(model_or_state: Any) -> dict[str, torch.Tensor]:
    """Only ``atom_encoder.*`` / ``bond_encoder.*``, CPU, detached."""
    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, nn.Module)
        else dict(model_or_state)
    )
    return {
        k: v.detach().cpu().clone()
        for k, v in state.items()
        if k.startswith(C.TRANSFERRED_PREFIXES)
    }


def encoder_fingerprint(model_or_state: Any) -> str:
    return C.state_fingerprint(encoder_state(model_or_state))


def encoder_fingerprint_from_state(state: Mapping[str, torch.Tensor]) -> str:
    return C.state_fingerprint(encoder_state(state))


def install_encoders(model: nn.Module, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Strictly overwrite only the two encoder module groups on ``model``."""
    incoming = encoder_state(state)
    expected = {
        k for k in model.state_dict() if k.startswith(C.TRANSFERRED_PREFIXES)
    }
    if set(incoming) != expected:
        raise RuntimeError(
            f"encoder key mismatch: extra={sorted(set(incoming) - expected)} "
            f"missing={sorted(expected - set(incoming))}"
        )
    non_encoder_before = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith(C.TRANSFERRED_PREFIXES)
    }
    with torch.no_grad():
        for key, value in incoming.items():
            target = model.state_dict()[key]
            if tuple(target.shape) != tuple(value.shape):
                raise RuntimeError(f"{key} shape {tuple(value.shape)} != {tuple(target.shape)}")
            target.copy_(value.to(target.device))
    non_encoder_after = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith(C.TRANSFERRED_PREFIXES)
    }
    changed = [
        k for k in non_encoder_after
        if not torch.equal(non_encoder_before[k], non_encoder_after[k])
    ]
    if changed:
        raise RuntimeError(f"transfer touched non-encoder tensors: {changed}")
    return {
        "transferred_keys": sorted(incoming),
        "transferred_tensors": len(incoming),
        "transferred_elements": int(sum(v.numel() for v in incoming.values())),
        "non_encoder_tensors_unchanged": True,
        "non_encoder_tensors_checked": len(non_encoder_after),
    }


def parameter_change(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Per-tensor L2 displacement of the two encoder groups."""
    keys = sorted(set(before) | set(after))
    rows = []
    for key in keys:
        a = before[key].double()
        b = after[key].double()
        delta = b - a
        rows.append(
            {
                "key": key,
                "numel": int(a.numel()),
                "l2_before": float(a.norm()),
                "l2_after": float(b.norm()),
                "l2_delta": float(delta.norm()),
                "max_abs_delta": float(delta.abs().max()),
                "relative_l2_delta": float(delta.norm() / a.norm().clamp_min(1e-30)),
            }
        )
    total_before = float(sum(r["l2_before"] ** 2 for r in rows) ** 0.5)
    total_after = float(sum(r["l2_after"] ** 2 for r in rows) ** 0.5)
    total_delta = float(sum(r["l2_delta"] ** 2 for r in rows) ** 0.5)
    return {
        "per_tensor": rows,
        "l2_before": total_before,
        "l2_after": total_after,
        "l2_delta": total_delta,
        "relative_l2_delta": total_delta / max(total_before, 1e-30),
        "any_tensor_exactly_equal": any(r["l2_delta"] == 0.0 for r in rows),
        "all_tensors_changed": all(r["l2_delta"] > 0.0 for r in rows),
    }


__all__ = [
    "PretextNet",
    "atom_context",
    "bond_context",
    "build_pretext",
    "encode_static",
    "encoder_fingerprint",
    "encoder_fingerprint_from_state",
    "encoder_state",
    "incidence_sums",
    "init_heads",
    "install_encoders",
    "make_encoders",
    "parameter_change",
    "seed0_scratch_state",
]
