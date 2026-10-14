"""Junction mechanism-exploration models (v2).

Protocol: ``configs/bjoc_junction_mechanism_exploration_v2.yaml``.
Track: ``tracks/structure-contribution``.

This module implements the three pre-declared Part-II arms as functional
forward modes on top of the frozen BJOC backbone.  No module, width, loss or
training primitive is added; the only structural change is the widened Reader
input of arm D.

Arms (composer inputs; ``w`` = learned Junction embeddings):

* ``A0`` — W=0, V=0, q=0 (history; seed-0 reference)
* ``A1`` — W=mu_G, V=nu_G, q=0 (history; seed-0 reference, seed-1 control)
* ``A2`` — W=n_e*mu_G, V=n_e*nu_G, q=log1p(n_e) (historical BJOC-Null / P3)
* ``C``  — W=0, V=0, q=log1p(n_e)                (J_COUNT_ONLY)
* ``D``  — A0 composer + reader input ``[z, z_A, z_A2, g, mu_G, nu_G]``
           (J_DIRECT_READOUT; 192 zero-initialised new reader columns)
* ``E``  — W=n_e*mu_{r_e}, V=n_e*nu_{r_e}, q=log1p(n_e) with r_e the raw
           physical bond type (J_TYPE_CONDITIONED_POOL)

E preserves the per-molecule totals ``sum_e W_e = 2 sum_j w_j`` and
``sum_e V_e = 2 sum_j w_j^2`` exactly (up to float reduction order), so the
only change versus A2 is how the same moment mass is distributed over the
bonds, conditioned on a fixed, label-blind chemical category.

Discipline: label-blind except the supervised loop (raw Fit ``y`` only);
official valid/test are never referenced.
"""

from __future__ import annotations

import hashlib
import math
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-junction-mechanism-exploration-v2"

MODE_A0 = "A0"
MODE_A1 = "A1"
MODE_A2 = "A2"
MODE_C = "C"
MODE_D = "D"
MODE_E = "E"
MODES = (MODE_A0, MODE_A1, MODE_A2, MODE_C, MODE_D, MODE_E)
ARM_MODES = {"A0": MODE_A0, "A1": MODE_A1, "A2": MODE_A2, "C": MODE_C, "D": MODE_D, "E": MODE_E}
ARM_NAMES = {
    "A0": "NO_LEARNED_J_CONTENT",
    "A1": "GLOBAL_J_MOMENTS",
    "A2": "P3_NULL_REFERENCE",
    "C": "J_COUNT_ONLY",
    "D": "J_DIRECT_READOUT",
    "E": "J_TYPE_CONDITIONED_POOL",
}

#: composer input layout (frozen): [b (96), W (96), V (96), q (1), raw bond (7)]
_COMPOSE_B = 96
_COMPOSE_W_OFF = 96
_COMPOSE_V_OFF = 192
_COMPOSE_Q_IDX = 288
_COMPOSE_RAW_OFF = 289

#: reader blocks
_READER_OLD_DIM = 321
_READER_NEW_DIM = 2 * 96  # mu_G + nu_G
READER_DIM_D = _READER_OLD_DIM + _READER_NEW_DIM  # 513

_PARAMS_BASE = bp.EXPECTED_PARAMETERS  # 413505
_PARAMS_D = _PARAMS_BASE + _READER_NEW_DIM * 256  # 462657


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class DirectReadoutConfig(src.BJOCConfig):
    """Reader input widened by the two graph-level Junction moment blocks."""

    @property
    def reader_in(self) -> int:
        return self.compose_out + 2 * self.atom_out + sro.GLOBAL_DIM + 2 * self.junc_out


def config_for_mode(mode: str) -> src.BJOCConfig:
    return DirectReadoutConfig() if mode == MODE_D else src.BJOCConfig()


# ---------------------------------------------------------------------------
# grouping primitives (pure functions; no learned parameter)
# ---------------------------------------------------------------------------


def per_bond_counts(batch: Mapping[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    n_bonds = int(batch["bond_feat"].shape[0])
    jb = batch["junc_bond"]
    n_b = torch.zeros(n_bonds, device=batch["junc_bond"].device, dtype=dtype)
    if jb.numel() > 0:
        n_b.index_add_(0, jb.reshape(-1), torch.ones_like(jb.reshape(-1), dtype=dtype))
    return n_b


def graph_junction_moments(
    w: torch.Tensor, batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """mu_G, nu_G of shape [n_graphs, junc_out] (unconditional graph means)."""
    n_graphs = int(batch["n_graphs"])
    junc_batch = batch["junc_batch"]
    w_sum = torch.zeros((n_graphs, w.shape[1]), device=w.device, dtype=w.dtype)
    w2_sum = torch.zeros_like(w_sum)
    if w.shape[0] > 0:
        w_sum.index_add_(0, junc_batch, w)
        w2_sum.index_add_(0, junc_batch, w * w)
    counts = torch.bincount(junc_batch, minlength=n_graphs).to(w.dtype).clamp(min=1.0).unsqueeze(1)
    return w_sum / counts, w2_sum / counts


def null_grouping(
    w: torch.Tensor, batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A2/BJOC-Null: W_e = n_e*mu_G, V_e = n_e*nu_G."""
    mu, nu = graph_junction_moments(w, batch)
    n_b = per_bond_counts(batch, w.dtype)
    bond_batch = batch["bond_batch"]
    return n_b.unsqueeze(1) * mu[bond_batch], n_b.unsqueeze(1) * nu[bond_batch], n_b


def type_conditioned_grouping(
    w: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    n_types: int = sro.BOND_CATEGORIES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Arm E: condition the Junction moments on the raw physical bond type.

    Returns ``(W_e, V_e, n_e, mu_r, nu_r)`` where ``mu_r``/``nu_r`` have shape
    ``[n_graphs * n_types, junc_out]`` (row ``g*n_types + r``).

    ``r_e`` is recovered from the one-hot ``bond_feat[:, :n_types]``; it is a
    fixed, label-blind physical category, not a learned bucket.
    """
    n_bonds = int(batch["bond_feat"].shape[0])
    n_graphs = int(batch["n_graphs"])
    jb = batch["junc_bond"]
    if n_types <= 0:
        raise ValueError("n_types must be positive")
    bond_type = batch["bond_feat"][:, :n_types].argmax(dim=1)
    n_b = per_bond_counts(batch, w.dtype)

    junc_batch = batch["junc_batch"]
    if jb.numel() > 0:
        key0 = junc_batch * n_types + bond_type[jb[:, 0]]
        key1 = junc_batch * n_types + bond_type[jb[:, 1]]
        keys = torch.cat([key0, key1], 0)
    else:
        keys = torch.zeros(0, dtype=torch.long, device=jb.device)
    n_buckets = n_graphs * n_types
    w_pair = torch.cat([w, w], 0) if w.shape[0] > 0 else w
    sum_w = torch.zeros((n_buckets, w.shape[1]), device=w.device, dtype=w.dtype)
    sum_w2 = torch.zeros_like(sum_w)
    if keys.numel() > 0:
        sum_w.index_add_(0, keys, w_pair)
        sum_w2.index_add_(0, keys, w_pair * w_pair)
    count = torch.bincount(keys, minlength=n_buckets).to(w.dtype)
    present = (count > 0).unsqueeze(1)
    denom = count.clamp(min=1.0).unsqueeze(1)
    mu_r = torch.where(present, sum_w / denom, torch.zeros_like(sum_w))
    nu_r = torch.where(present, sum_w2 / denom, torch.zeros_like(sum_w2))
    bucket = batch["bond_batch"] * n_types + bond_type
    W = n_b.unsqueeze(1) * mu_r[bucket]
    V = n_b.unsqueeze(1) * nu_r[bucket]
    return W, V, n_b, mu_r, nu_r


def type_conditioned_grouping_reference(
    w: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    n_types: int = sro.BOND_CATEGORIES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent Python-loop reference used by the correctness gate.

    Iterates over every junction and every incidence in interpreted Python and
    accumulates the type-conditioned sums; intentionally slow and simple.
    """
    jb = batch["junc_bond"].detach().cpu().numpy()
    bond_type = batch["bond_feat"][:, :n_types].argmax(dim=1).detach().cpu().numpy()
    bond_batch = batch["bond_batch"].detach().cpu().numpy()
    junc_batch = batch["junc_batch"].detach().cpu().numpy()
    n_bonds = int(batch["bond_feat"].shape[0])
    n_graphs = int(batch["n_graphs"])
    w_np = w.detach().cpu().numpy().astype(np.float64)
    jdim = w_np.shape[1] if w_np.ndim == 2 else 0
    n_b = np.zeros(n_bonds, dtype=np.float64)
    sum_w = np.zeros((n_graphs, n_types, jdim), dtype=np.float64)
    sum_w2 = np.zeros_like(sum_w)
    count = np.zeros((n_graphs, n_types), dtype=np.float64)
    for j in range(jb.shape[0]):
        g = int(junc_batch[j])
        for side in (0, 1):
            b = int(jb[j, side])
            r = int(bond_type[b])
            n_b[b] += 1.0
            count[g, r] += 1.0
            sum_w[g, r] += w_np[j]
            sum_w2[g, r] += w_np[j] * w_np[j]
    mu = np.zeros_like(sum_w)
    nu = np.zeros_like(sum_w2)
    for g in range(n_graphs):
        for r in range(n_types):
            if count[g, r] > 0:
                mu[g, r] = sum_w[g, r] / count[g, r]
                nu[g, r] = sum_w2[g, r] / count[g, r]
    W = np.zeros((n_bonds, jdim), dtype=np.float64)
    V = np.zeros_like(W)
    for b in range(n_bonds):
        g = int(bond_batch[b])
        r = int(bond_type[b])
        W[b] = n_b[b] * mu[g, r]
        V[b] = n_b[b] * nu[g, r]
    return (
        torch.as_tensor(W, dtype=w.dtype),
        torch.as_tensor(V, dtype=w.dtype),
        torch.as_tensor(n_b, dtype=w.dtype),
    )


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class BJOCMechV2(src.BJOC):
    """One-shot BJOC backbone with a functional switch over the Junction path."""

    def __init__(
        self,
        config: src.BJOCConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        mode: str = MODE_A2,
    ) -> None:
        super().__init__(config, global_mean, global_scale, organization_mode=src.NULL)
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode

    def group_to_bonds_mech(
        self,
        w: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        mode: str,
    ) -> dict[str, torch.Tensor]:
        n_bonds = int(batch["bond_feat"].shape[0])
        jdim = int(w.shape[1])
        if mode == MODE_A2:
            W, V, n_b = null_grouping(w, batch)
            mu, nu = graph_junction_moments(w, batch)
            return {"W": W, "V": V, "n_b": n_b, "mu_G": mu, "nu_G": nu}
        if mode == MODE_E:
            W, V, n_b, mu_r, nu_r = type_conditioned_grouping(w, batch)
            mu, nu = graph_junction_moments(w, batch)
            return {"W": W, "V": V, "n_b": n_b, "mu_G": mu, "nu_G": nu, "mu_r": mu_r, "nu_r": nu_r}
        mu, nu = graph_junction_moments(w, batch)
        n_b = per_bond_counts(batch, w.dtype)
        zeros = torch.zeros((n_bonds, jdim), device=w.device, dtype=w.dtype)
        if mode == MODE_A0 or mode == MODE_D:
            W, V, q = zeros, zeros, torch.zeros_like(n_b)
        elif mode == MODE_A1:
            W = mu[batch["bond_batch"]]
            V = nu[batch["bond_batch"]]
            q = torch.zeros_like(n_b)
        elif mode == MODE_C:
            W, V, q = zeros, zeros, n_b
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return {"W": W, "V": V, "n_b": n_b, "mu_G": mu, "nu_G": nu, "q_raw": q}

    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if organization_mode is not None:
            return src.BJOC.forward_features(self, batch, organization_mode)
        grp = self.mode if mode is None else mode
        if grp not in MODES:
            raise ValueError(f"unknown mode {grp!r}")
        if grp == MODE_A2:
            feats = src.BJOC.forward_features(self, batch)
            mu, nu = graph_junction_moments(feats["w"], batch)
            feats["mu_G"] = mu
            feats["nu_G"] = nu
            return feats
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)
        grp_feats = self.group_to_bonds_mech(w, batch, grp)
        W, V, n_b = grp_feats["W"], grp_feats["V"], grp_feats["n_b"]
        q = n_b if grp in (MODE_C, MODE_E) else torch.zeros_like(n_b)
        compose_in = torch.cat([b, W, V, torch.log1p(q).unsqueeze(1), batch["bond_feat"]], dim=1)
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        if c.shape[0] > 0:
            z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sq = torch.zeros_like(a_sum)
        if a.shape[0] > 0:
            a_sum.index_add_(0, batch["atom_batch"], a)
            a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        if grp == MODE_D:
            reader_in = torch.cat([z, a_sum, a_sq, g, grp_feats["mu_G"], grp_feats["nu_G"]], dim=1)
        else:
            reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        return {
            "a": a,
            "b": b,
            "w": w,
            "W": W,
            "V": V,
            "n_b": n_b,
            "q": q,
            "c": c,
            "z": z,
            "reader_in": reader_in,
            **{k: v for k, v in grp_feats.items() if k not in ("W", "V", "n_b", "q_raw")},
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        return_features: bool = False,
        mode: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = self.forward_features(batch, organization_mode, mode=mode)
        pred = self.reader(feats["reader_in"]).view(-1)
        if return_features:
            return pred, feats
        return pred


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def expand_reader_first_layer(base_reader: src._MLP, new_reader: src._MLP) -> dict[str, Any]:
    """Copy the base reader into a wider reader, new trailing columns zero.

    The new Junction moment blocks are appended AFTER the frozen 321 columns,
    so every old column keeps its index and the initial function is exactly the
    base reader on the old input.
    """
    old_in = int(base_reader.lins[0].weight.shape[1])
    new_in = int(new_reader.lins[0].weight.shape[1])
    if new_in <= old_in:
        raise SystemExit("expanded reader is not wider than the base reader")
    with torch.no_grad():
        w_new = new_reader.lins[0].weight.detach()
        w_new.zero_()
        w_new[:, :old_in].copy_(base_reader.lins[0].weight.detach())
        new_reader.lins[0].bias.copy_(base_reader.lins[0].bias.detach())
        for i in range(1, len(base_reader.lins)):
            new_reader.lins[i].load_state_dict(base_reader.lins[i].state_dict())
    return {
        "old_columns": old_in,
        "new_columns": new_in - old_in,
        "new_columns_zero": bool(float(w_new[:, old_in:].abs().max()) == 0.0),
        "new_columns_max_abs": float(w_new[:, old_in:].abs().max()),
    }


def _copy_shared_modules(target: src.BJOC, base: src.BJOC) -> None:
    target.atom_encoder.load_state_dict(base.atom_encoder.state_dict())
    target.bond_encoder.load_state_dict(base.bond_encoder.state_dict())
    target.junction_encoder.load_state_dict(base.junction_encoder.state_dict())
    target.group_composer.load_state_dict(base.group_composer.state_dict())
    with torch.no_grad():
        target.global_mean.copy_(base.global_mean)
        target.global_scale.copy_(base.global_scale)


def shared_parameter_hash(model: src.BJOC, *, exclude_prefixes: Sequence[str] = ()) -> str:
    h = hashlib.sha256()
    skip = tuple(exclude_prefixes)
    for key, value in sorted(model.state_dict().items()):
        if any(key.startswith(p) for p in skip):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def build_mech_from_init(
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str,
    mode: str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[src.BJOC, BJOCMechV2, dict[str, Any]]:
    """Build (scratch base holder, mech model) from one shared init.

    The mech model receives the exact P3-D-s{seed} scratch parameter tensors;
    for mode D only the reader first-layer weight is widened (old columns
    copied, 192 new columns zero).  The SSL encoder state, when given, is
    loaded after the assignment.
    """
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}")
    base = bp.build_bjoc_null(seed, norm, device)
    scratch_full = src.state_dict_hash(base)
    scratch_group = mp.group_state_hash(base)
    scratch_encoder = bp.encoder_group_hash(base)
    cfg = config_for_mode(mode)
    model = BJOCMechV2(
        cfg,
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        mode=mode,
    )
    _copy_shared_modules(model, base)
    reader_expansion = None
    if mode == MODE_D:
        reader_expansion = expand_reader_first_layer(base.reader, model.reader)
    else:
        model.reader.load_state_dict(base.reader.state_dict())
    model.to(device)
    if mode != MODE_D:
        if src.state_dict_hash(model) != scratch_full:
            raise SystemExit("mech init does not match the shared scratch init")
        if mp.group_state_hash(model) != scratch_group:
            raise SystemExit("mech group state does not match the shared scratch init")
    model_shared_hash = shared_parameter_hash(
        model, exclude_prefixes=("reader.lins.0.weight",) if mode == MODE_D else ()
    )
    base_shared_hash = shared_parameter_hash(
        base, exclude_prefixes=("reader.lins.0.weight",) if mode == MODE_D else ()
    )
    if model_shared_hash != base_shared_hash:
        raise SystemExit("mech shared parameter hash does not match the base init")
    if ssl_state is not None:
        mp.load_encoder_state(model, ssl_state, device)
    encoder_used = bp.encoder_group_hash(model)
    audit = src.parameter_audit(cfg, model)
    expected = _PARAMS_D if mode == MODE_D else _PARAMS_BASE
    if not audit["parameter_exact"] or audit["actual_parameters"] != expected:
        raise SystemExit(f"mech parameter audit failed: {audit} expected {expected}")
    named = {name: int(p.numel()) for name, p in model.named_parameters()}
    info = {
        "mode": mode,
        "mode_name": next(name for arm_key, name in ARM_NAMES.items() if ARM_MODES[arm_key] == mode),
        "base_scratch_init_full_sha256": scratch_full,
        "base_scratch_init_group_state_sha256": scratch_group,
        "base_scratch_init_encoder_sha256": scratch_encoder,
        "model_shared_parameter_sha256": model_shared_hash,
        "base_shared_parameter_sha256": base_shared_hash,
        "encoder_used_sha256": encoder_used,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "expected_parameters": int(expected),
        "new_parameters_vs_base": int(expected - _PARAMS_BASE),
        "reader_expansion": reader_expansion,
        "named_parameters": named,
        "reader_input_dim": int(cfg.reader_in),
    }
    return base, model, info


# ---------------------------------------------------------------------------
# gradient / capacity audit
# ---------------------------------------------------------------------------


def _grad_stats(p: torch.Tensor) -> tuple[int, float | None, float | None]:
    if p.grad is None:
        return 0, None, None
    g = p.grad.detach()
    return int((g != 0).sum()), float(g.norm()), float(g.abs().max())


def gradient_audit_mech(
    model: BJOCMechV2,
    *,
    objs: Sequence[sro.MolObjects],
    y: np.ndarray,
    device: torch.device | str,
    mode: str | None = None,
    n_molecules: int = 128,
) -> dict[str, Any]:
    """One forward/backward pass: per-module gradients and active capacity."""
    dev = torch.device(device)
    chunk = list(objs[: int(n_molecules)])
    y_t = torch.as_tensor(np.asarray(y[: len(chunk)]), dtype=torch.float32, device=dev)
    model.train()
    batch = bp._mv(sro.batch_objects(chunk), dev)
    model.zero_grad(set_to_none=True)
    pred = model(batch, mode=mode)
    loss = F.l1_loss(pred.view(-1), y_t.view(-1))
    loss.backward()
    modules = {
        "atom_encoder": model.atom_encoder,
        "bond_encoder": model.bond_encoder,
        "junction_encoder": model.junction_encoder,
        "group_composer": model.group_composer,
        "reader": model.reader,
    }
    per_module: dict[str, Any] = {}
    active_total = 0
    allocated_total = 0
    active_entries = 0
    for name, module in modules.items():
        params = list(module.parameters())
        allocated = int(sum(p.numel() for p in params))
        active = 0
        entries = 0
        norms: list[float] = []
        maxabs: list[float] = []
        for p in params:
            e, n, m = _grad_stats(p)
            entries += e
            if e > 0:
                active += int(p.numel())
            if n is not None:
                norms.append(n)
            if m is not None:
                maxabs.append(m)
        per_module[name] = {
            "allocated_parameters": allocated,
            "parameters_with_nonzero_grad": active,
            "active_entries_nonzero_grad": entries,
            "grad_norm_total": float(sum(norms)) if norms else None,
            "grad_max_abs": float(max(maxabs)) if maxabs else None,
            "has_grad": bool(norms),
        }
        active_total += active
        active_entries += entries
        allocated_total += allocated
    w0 = model.group_composer.lins[0].weight
    if w0.grad is None:
        col_blocks = {k: None for k in ("b", "W", "V", "q", "raw")}
    else:
        g = w0.grad.detach()
        col_blocks = {
            "b": float(g[:, :_COMPOSE_W_OFF].abs().max()),
            "W": float(g[:, _COMPOSE_W_OFF:_COMPOSE_V_OFF].abs().max()),
            "V": float(g[:, _COMPOSE_V_OFF:_COMPOSE_Q_IDX].abs().max()),
            "q": float(g[:, _COMPOSE_Q_IDX:_COMPOSE_RAW_OFF].abs().max()),
            "raw": float(g[:, _COMPOSE_RAW_OFF:].abs().max()),
        }
    reader_first = model.reader.lins[0].weight
    reader_cols: dict[str, Any]
    if model.mode == MODE_D:
        if reader_first.grad is None:
            reader_cols = {"old_max_abs": None, "new_max_abs": None}
        else:
            rg = reader_first.grad.detach()
            reader_cols = {
                "old_max_abs": float(rg[:, :_READER_OLD_DIM].abs().max()),
                "new_max_abs": float(rg[:, _READER_OLD_DIM:].abs().max()),
            }
    else:
        reader_cols = {"old_max_abs": None, "new_max_abs": None}
    return {
        "mode": model.mode if mode is None else mode,
        "n_molecules": len(chunk),
        "loss_l1": float(loss.detach()),
        "modules": per_module,
        "composer_first_layer_column_grad_max_abs": col_blocks,
        "reader_first_layer_column_grad_max_abs": reader_cols,
        "allocated_parameters": allocated_total,
        "parameters_with_nonzero_grad": active_total,
        "active_entries_nonzero_grad": active_entries,
        "effective_active_entry_fraction": float(active_entries) / float(max(allocated_total, 1)),
        "note": (
            "active_entries counts scalar parameter entries with a present non-zero gradient in this "
            "one batch; it is an effective-capacity indicator, not a capacity-matching guarantee."
        ),
    }


def gradient_audit_after_optimizer_step(
    model: BJOCMechV2,
    *,
    objs: Sequence[sro.MolObjects],
    y: np.ndarray,
    device: torch.device | str,
    mode: str | None = None,
    lr: float = 1e-3,
    n_molecules: int = 128,
) -> dict[str, Any]:
    """Gradient audit at init and again after a real optimizer step.

    A zero-initialised reader gate (arm D) blocks the Junction-encoder
    gradient on the very first backward pass; the gate opens once the new
    columns have moved.  This helper is the CPU witness that the channel is
    alive after training has started, not merely at initialisation.
    """
    first = gradient_audit_mech(model, objs=objs, y=y, device=device, mode=mode, n_molecules=n_molecules)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    opt.step()
    second = gradient_audit_mech(model, objs=objs, y=y, device=device, mode=mode, n_molecules=n_molecules)
    return {"at_init": first, "after_one_step": second}


# ---------------------------------------------------------------------------
# supervised training loop (frozen bp body, model construction parameterised)
# ---------------------------------------------------------------------------


def train_supervised_mech(
    *,
    arm: str,
    seed: int,
    recipe_key: str,
    init: str,
    train_objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    device: torch.device,
    mode: str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    smoke_epochs: int | None = None,
    log_fn: Callable[[str], None] = log,
) -> dict[str, Any]:
    """One supervised trajectory; loop body identical to ``bp.train_supervised``."""
    if recipe_key not in bp.RECIPES:
        raise SystemExit(f"unknown recipe {recipe_key!r}")
    recipe = bp.RECIPES[recipe_key]
    if init not in ("scratch", "ssl"):
        raise SystemExit(f"unknown init {init!r}")
    if init == "ssl" and ssl_state is None:
        raise SystemExit("init=ssl requires a pretrained encoder state")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")

    _base, model, build_info = build_mech_from_init(
        seed, norm, device, mode, ssl_state if init == "ssl" else None
    )
    if init == "ssl" and build_info["encoder_used_sha256"] == build_info["base_scratch_init_encoder_sha256"]:
        raise SystemExit("init=ssl did not change the encoder state")
    init_full_sha = build_info["base_scratch_init_full_sha256"]
    init_group_sha = build_info["base_scratch_init_group_state_sha256"]
    init_encoder_sha = build_info["base_scratch_init_encoder_sha256"]
    encoder_used_sha = build_info["encoder_used_sha256"]
    audit = src.parameter_audit(config_for_mode(mode), model)
    expected = _PARAMS_D if mode == MODE_D else _PARAMS_BASE
    if not audit["parameter_exact"] or audit["actual_parameters"] != expected:
        raise SystemExit(f"parameter audit failed: {audit}")

    epochs = int(recipe["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(rows))
    steps_per_epoch = math.ceil(n / bp.BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = None
    if recipe["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"])
        )
    elif recipe["schedule"] != "constant":
        raise SystemExit(f"unknown schedule {recipe['schedule']!r}")

    gen = torch.Generator().manual_seed(int(seed) + bp.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=bp.SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    total_mol = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, bp.BATCH_SIZE, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = bp._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), bp.GRAD_CLIP))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > bp.GRAD_CLIP:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                expected_lr = bp.schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
                lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        train_mae = total / max(n_mol, 1)
        total_mol += n_mol
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")
    wall = time.perf_counter() - started

    members, soup = bp._last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = bp.predict_model(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "mode": mode,
        "mode_name": ARM_NAMES[arm],
        "seed": int(seed),
        "recipe_key": recipe_key,
        "recipe": dict(recipe),
        "init": init,
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {bp.SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_state": soup_state,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "schedule": {
            "type": recipe["schedule"],
            "per": "step",
            "warmup": "none",
            "lr0": float(recipe["lr"]),
            "lr_min": None if recipe["lr_min"] is None else float(recipe["lr_min"]),
            "t_max_updates": total_updates,
            "lr_first_step": curve[0]["lr_first"],
            "lr_last_step": curve[-1]["lr_last"],
            "cosine_formula_max_abs_deviation": lr_deviation_max,
        },
        "init_full_sha256": init_full_sha,
        "init_group_state_sha256": init_group_sha,
        "init_encoder_sha256": init_encoder_sha,
        "encoder_used_sha256": encoder_used_sha,
        "smoke": bool(smoke_epochs is not None),
        "full_recipe_epochs": int(recipe["epochs"]),
        "build_info": {k: v for k, v in build_info.items() if k != "named_parameters"},
        "build_info_named_parameters": build_info["named_parameters"],
        "parameter_audit": audit,
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "full_hash_independent_recomputation": bp.batch_order_hash(seed, n, bp.BATCH_SIZE, epochs),
            "full_matches": order_hash.hexdigest() == bp.batch_order_hash(seed, n, bp.BATCH_SIZE, epochs),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": grad_norm_max,
            "lr_schedule_deviation_max": lr_deviation_max,
        },
        "lr_curve": curve,
        "wall_seconds": wall,
        "molecules_seen": int(total_mol),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_used": "train Fit rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# synthetic witnesses (CPU-only)
# ---------------------------------------------------------------------------


def type_witness_batch() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Two incidence systems with identical A2 summaries, different E summaries.

    Both systems have 4 bonds (2 single-type r=0: b0,b1; 2 double-type r=1:
    b2,b3), 4 junctions with two distinct w vectors `X=[1,0]` (j0,j1) and
    `Y=[0,1]` (j2,j3), every bond touching exactly two junctions, and the same
    junction multiset, bond multiset and per-bond counts {2,2,2,2}:

    * A: the X junctions touch only type-0 bonds, the Y junctions only type-1
    * B: the X junctions touch only type-1 bonds, the Y junctions only type-0

    A2 cannot see the difference (same w multiset, same per-bond counts and
    identical graph moments); E can (the conditional means swap).
    """
    bond_feat = torch.zeros((4, sro.BOND_FEAT_DIM), dtype=torch.float32)
    bond_feat[0, 0] = 1.0
    bond_feat[1, 0] = 1.0
    bond_feat[2, 1] = 1.0
    bond_feat[3, 1] = 1.0
    junc_bond_a = torch.tensor([[0, 1], [0, 1], [2, 3], [2, 3]], dtype=torch.long)
    junc_bond_b = torch.tensor([[2, 3], [2, 3], [0, 1], [0, 1]], dtype=torch.long)
    common = {
        "n_graphs": 1,
        "bond_batch": torch.zeros(4, dtype=torch.long),
        "junc_batch": torch.zeros(4, dtype=torch.long),
        "bond_feat": bond_feat,
    }
    batch_a = {**common, "junc_bond": junc_bond_a}
    batch_b = {**common, "junc_bond": junc_bond_b}
    return batch_a, batch_b


# ---------------------------------------------------------------------------
# hash helpers
# ---------------------------------------------------------------------------


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()
