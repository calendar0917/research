"""STRC — Shared-Type Residual Composition (cscl-bjoc-strc-predictive-feasibility-v1).

Protocol: ``configs/bjoc_strc_predictive_feasibility_v1.yaml``.

Single scientific question
--------------------------
Can a *tiny*, cross-molecule-shared, explicitly additive structural residual
channel improve held-out single-model raw-y MAE **on top of the already
successful masked-pretrained BJOC-Null (P)**, when it is trained jointly with
the full model under the identical recipe?

The only model of this round::

    y_hat(G) = f_theta(G) + s_a * sum_{i in atoms}     alpha[type_i]
                         + s_j * sum_{j in junctions} beta [context_j]

where

* ``f_theta`` is the complete pretrained BJOC-Null forward (unchanged);
* ``alpha``  is a shared scalar table over the 28 raw PyG atom-type codes,
  initialised at zero;
* ``beta``   is a shared scalar table over a frozen 128-context vocabulary
  plus one ``OTHER`` slot (129 entries), initialised at zero — the context of
  a junction is ``(endpoint_type_low, centre_type, endpoint_type_high)`` with
  the two endpoint types sorted, the centre kept explicit;
* ``s_a = 1 / mean_train(n_atoms)`` and
  ``s_j = 1 / max(mean_train(n_junctions), 1)`` are frozen, non-learned
  scaling constants computed from the official train-10k structures only.

The vocabulary is built **only** from the official train-10k structures:
count each junction-context once, sort by count descending, break ties by
lexicographic context order, keep the 128 most frequent contexts, map every
other context to ``OTHER``.  No label, valid or test information is involved.

This module deliberately contains no learned calibration of the base
prediction: there is no ``a(counts) * f + b(counts)`` path, no MLP, no extra
structural feature, no GNN/attention/iteration.  ``alpha``/``beta`` only ever
enter through raw type/context counts.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-strc-predictive-feasibility-v1"

#: raw PyG atom-type vocabulary (the frozen BJOC atom-type block width)
ATOM_VOCAB = sro.ATOM_CATEGORIES  # 28

#: frozen junction-context vocabulary size + the OTHER slot
CONTEXT_KEEP = 128
OTHER_SLOT = CONTEXT_KEEP
BETA_SIZE = CONTEXT_KEEP + 1  # 129

#: context key space: (lo, centre, hi) with lo <= hi over ATOM_VOCAB codes
CONTEXT_KEY_SPACE = ATOM_VOCAB * ATOM_VOCAB * ATOM_VOCAB  # 21952

#: expected parameter budget (frozen by the protocol)
BASE_PARAMETERS = 413505
NEW_PARAMETERS = ATOM_VOCAB + BETA_SIZE  # 157
TOTAL_PARAMETERS = BASE_PARAMETERS + NEW_PARAMETERS  # 413662


# ---------------------------------------------------------------------------
# context vocabulary
# ---------------------------------------------------------------------------


def atom_type_codes(atom_feat: np.ndarray) -> np.ndarray:
    """Recover the raw PyG atom-type code from the frozen one-hot block."""
    block = np.asarray(atom_feat)[:, :ATOM_VOCAB]
    return block.argmax(axis=1).astype(np.int64)


def junction_context(t_i: int, t_j: int, t_k: int) -> tuple[int, int, int]:
    """Frozen context key ``(endpoint_low, centre, endpoint_high)``."""
    lo, hi = (int(t_i), int(t_k)) if int(t_i) <= int(t_k) else (int(t_k), int(t_i))
    return (lo, int(t_j), hi)


def context_key_index(lo: int, centre: int, hi: int) -> int:
    return (int(lo) * ATOM_VOCAB + int(centre)) * ATOM_VOCAB + int(hi)


def molecule_contexts(obj: sro.MolObjects) -> list[tuple[int, int, int]]:
    """All junction contexts of one molecule (one entry per junction)."""
    types = atom_type_codes(obj.atom_feat)
    out: list[tuple[int, int, int]] = []
    for i, j, k in np.asarray(obj.junc_index, dtype=np.int64):
        out.append(junction_context(types[int(i)], types[int(j)], types[int(k)]))
    return out


def build_context_vocabulary(objects: Sequence[sro.MolObjects]) -> dict[str, Any]:
    """Train-only vocabulary construction (label-blind, deterministic).

    Returns the ordered context list (top :data:`CONTEXT_KEEP`), their counts,
    the OTHER-junction count and the frozen scales.  There is no RNG and no
    dependence on any label.
    """
    counts: dict[tuple[int, int, int], int] = {}
    n_atoms_total = 0
    n_junctions_total = 0
    for obj in objects:
        n_atoms_total += int(obj.n_atoms)
        n_junctions_total += int(obj.junc_index.shape[0])
        for ctx in molecule_contexts(obj):
            counts[ctx] = counts.get(ctx, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
    kept = ordered[:CONTEXT_KEEP]
    other_count = int(sum(c for _, c in ordered[CONTEXT_KEEP:]))
    mean_n_atoms = float(n_atoms_total) / max(len(objects), 1)
    mean_n_junctions = float(n_junctions_total) / max(len(objects), 1)
    s_a = 1.0 / mean_n_atoms
    s_j = 1.0 / max(mean_n_junctions, 1.0)
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "n_molecules": int(len(objects)),
        "n_atoms_total": int(n_atoms_total),
        "n_junctions_total": int(n_junctions_total),
        "distinct_contexts": int(len(counts)),
        "context_keep": CONTEXT_KEEP,
        "other_slot": OTHER_SLOT,
        "beta_size": BETA_SIZE,
        "vocab": [list(ctx) for ctx, _ in kept],
        "vocab_counts": [int(c) for _, c in kept],
        "other_junctions": other_count,
        "covered_junctions": int(n_junctions_total - other_count),
        "covered_fraction": float((n_junctions_total - other_count) / max(n_junctions_total, 1)),
        "mean_n_atoms": mean_n_atoms,
        "mean_n_junctions": mean_n_junctions,
        "s_a": s_a,
        "s_j": s_j,
        "counts_full": {f"{a},{b},{c}": int(v) for (a, b, c), v in ordered},
    }
    payload["vocab_sha256"] = vocabulary_sha256(payload)
    return payload


def vocabulary_sha256(payload: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for ctx in payload["vocab"]:
        h.update((",".join(str(int(x)) for x in ctx)).encode())
        h.update(b";")
    for c in payload["vocab_counts"]:
        h.update(str(int(c)).encode())
        h.update(b",")
    h.update(str(int(payload["other_junctions"])).encode())
    h.update(b"|")
    h.update(repr(float(payload["s_a"])).encode())
    h.update(b"|")
    h.update(repr(float(payload["s_j"])).encode())
    return h.hexdigest()


def vocab_lookup_table(payload: Mapping[str, Any]) -> np.ndarray:
    """``[21952]`` int64 lookup: context key -> slot, default :data:`OTHER_SLOT`."""
    table = np.full(CONTEXT_KEY_SPACE, OTHER_SLOT, dtype=np.int64)
    for slot, ctx in enumerate(payload["vocab"]):
        lo, centre, hi = (int(x) for x in ctx)
        if not (0 <= lo < ATOM_VOCAB and 0 <= centre < ATOM_VOCAB and 0 <= hi < ATOM_VOCAB):
            raise ValueError(f"context {ctx} outside the frozen atom vocabulary")
        table[context_key_index(lo, centre, hi)] = slot
    return table


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class STRCResidual(nn.Module):
    """The shared additive count channel (157 parameters, zero-initialised)."""

    def __init__(self, vocab: Mapping[str, Any]) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(ATOM_VOCAB, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(BETA_SIZE, dtype=torch.float32))
        lookup = vocab_lookup_table(vocab)
        # non-persistent: the lookup is rebuilt from the frozen vocabulary
        # artifact before loading any checkpoint, so it must not enter soup
        # averaging or the state dict
        self.register_buffer("ctx_lookup", torch.as_tensor(lookup, dtype=torch.long), persistent=False)
        self.s_a = float(vocab["s_a"])
        self.s_j = float(vocab["s_j"])
        self.vocab_sha256 = str(vocab["vocab_sha256"])

    def count_vectors(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-molecule raw count vectors ``[n_graphs, 28]`` and ``[n_graphs, 129]``."""
        feats = batch["atom_feat"]
        n_graphs = int(batch["n_graphs"])
        a_counts = torch.zeros((n_graphs, ATOM_VOCAB), dtype=feats.dtype, device=feats.device)
        a_counts.index_add_(0, batch["atom_batch"], feats[:, :ATOM_VOCAB])
        types = feats[:, :ATOM_VOCAB].argmax(dim=1)
        junc = batch["junc_index"]
        ti, tj, tk = types[junc[:, 0]], types[junc[:, 1]], types[junc[:, 2]]
        lo = torch.minimum(ti, tk)
        hi = torch.maximum(ti, tk)
        ctx = self.ctx_lookup[(lo * ATOM_VOCAB + tj) * ATOM_VOCAB + hi]
        c_counts = torch.zeros((n_graphs, BETA_SIZE), dtype=feats.dtype, device=feats.device)
        if ctx.numel():
            c_counts.index_add_(0, batch["junc_batch"], F.one_hot(ctx, BETA_SIZE).to(dtype=feats.dtype))
        return a_counts, c_counts

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        return_counts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_counts, c_counts = self.count_vectors(batch)
        resid = self.s_a * (a_counts @ self.alpha) + self.s_j * (c_counts @ self.beta)
        if return_counts:
            return resid, a_counts, c_counts
        return resid


class STRCBJOC(nn.Module):
    """Frozen BJOC base plus the shared-type residual channel.

    The base module is held by reference, so its parameters and initialisation
    are exactly the P initialisation; only ``residual.alpha`` / ``residual.beta``
    are new.  With alpha/beta = 0 the forward is the base forward bit-for-bit
    (the residual is multiplied, not re-summed through the reader).
    """

    def __init__(self, base: src.BJOC, residual: STRCResidual) -> None:
        super().__init__()
        self.base = base
        self.residual = residual

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base_pred = self.base(batch, organization_mode=organization_mode)
        resid, a_counts, c_counts = self.residual(batch, return_counts=True)
        pred = base_pred + resid
        if return_features:
            return pred, {"base_pred": base_pred, "residual": resid, "a_counts": a_counts, "c_counts": c_counts}
        return pred


# ---------------------------------------------------------------------------
# builders / audits
# ---------------------------------------------------------------------------


def wrap_strc(base: src.BJOC, vocab: Mapping[str, Any], device: torch.device | str = "cpu") -> STRCBJOC:
    """Wrap an already-built base model.  Never touches the base parameters."""
    model = STRCBJOC(base, STRCResidual(vocab))
    return model.to(device)


def strc_parameter_audit(model: STRCBJOC) -> dict[str, Any]:
    base_params = int(sum(p.numel() for p in model.base.parameters()))
    alpha = int(model.residual.alpha.numel())
    beta = int(model.residual.beta.numel())
    actual = int(sum(p.numel() for p in model.parameters()))
    return {
        "base_parameters": base_params,
        "alpha_parameters": alpha,
        "beta_parameters": beta,
        "new_parameters": alpha + beta,
        "actual_parameters": actual,
        "expected_base": BASE_PARAMETERS,
        "expected_new": NEW_PARAMETERS,
        "expected_total": TOTAL_PARAMETERS,
        "parameter_exact": actual == TOTAL_PARAMETERS and base_params == BASE_PARAMETERS,
    }


def strc_group_state_sha256(model: STRCBJOC) -> dict[str, str]:
    """Group hashes for the pre-registered initialisation audit."""
    out: dict[str, str] = {}
    out["base_full"] = src.state_dict_hash(model.base)
    combined = {}
    for prefix, group in (("atom", "atom_encoder"), ("bond", "bond_encoder"), ("junc", "junction_encoder")):
        for k, v in getattr(model.base, group).state_dict().items():
            combined[f"{prefix}:{k}"] = v
    out["base_encoder"] = _hash_tensors(combined)
    out["base_composer_reader"] = _hash_group_modules(model.base, ("group_composer", "reader"))
    out["residual"] = _hash_tensors(
        {
            "alpha": model.residual.alpha.detach().cpu(),
            "beta": model.residual.beta.detach().cpu(),
        }
    )
    return out


def _hash_tensors(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def _hash_group_modules(model: nn.Module, names: Sequence[str]) -> str:
    combined: dict[str, torch.Tensor] = {}
    for name in names:
        module = getattr(model, name)
        for k, v in module.state_dict().items():
            combined[f"{name}.{k}"] = v
    return _hash_tensors(combined)


def _encoder_group_hash(model: src.BJOC) -> str:
    """Reproduce the frozen full10k driver's encoder-group hash exactly."""
    combined: dict[str, torch.Tensor] = {}
    for prefix, group in (("atom", "atom_encoder"), ("bond", "bond_encoder"), ("junc", "junction_encoder")):
        for k, v in getattr(model, group).state_dict().items():
            combined[f"{prefix}:{k}"] = v
    return _hash_tensors(combined)


def strc_forward_with_parts(model: STRCBJOC, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Label-blind forward returning base, residual and both count vectors."""
    with torch.no_grad():
        _, parts = model(batch, return_features=True)
    return parts


def atom_table_maps(model: STRCBJOC) -> dict[str, Any]:
    return {
        "alpha": model.residual.alpha.detach().cpu().numpy().astype(np.float64).tolist(),
        "beta": model.residual.beta.detach().cpu().numpy().astype(np.float64).tolist(),
        "s_a": float(model.residual.s_a),
        "s_j": float(model.residual.s_j),
        "vocab_sha256": model.residual.vocab_sha256,
    }
