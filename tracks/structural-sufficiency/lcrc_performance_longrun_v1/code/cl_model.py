"""LCRC model arms for the performance long-run.

The frozen ``LCRC`` graph is untouched.  Four single-change arms are built by
zero-initialised extension at exactly the place the hypothesis names:

``chem8``
    widen the first ``atom_encoder`` linear from 51 to 59 columns; the eight new
    columns are zero, so step 0 is BASE.

``rhoshape``
    append ``max_v h_v`` and ``mean_v |h_v - mean h|`` (2 x 64 dims) to the
    ``rho`` input; the 128 new first-layer columns are zero.

``multicontrib``
    add ``Q: 64 -> 64 -> 8``, a graph sum of eight latent contributions, and
    ``delta = MLP([Q_graph, log1p(n_atoms)])`` with a zero final layer.

``combo_*``
    the corresponding two extensions in one model.

No attention, no message passing, no extra interaction layer, no SSL, no
prototype dictionary.  ``f`` and ``rho`` remain plain MLPs.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cl_common as C  # noqa: I001

import mb2_model as M
import lcrc_model as PREP_MODEL
from feature_spec import FeatureSpec

RHO_ARMS = {"rhoshape", "combo_chem_rho", "combo_rho_contrib"}
CONTRIB_ARMS = {"multicontrib", "combo_chem_contrib", "combo_rho_contrib"}
CHEM8_ARMS = {"chem8", "combo_chem_contrib", "combo_chem_rho"}


class LCRCVariant(PREP_MODEL.LCRC):
    """Base LCRC plus the pre-registered extensions selected by ``arm``."""

    def __init__(
        self,
        spec: FeatureSpec,
        norm: Mapping[str, Any],
        cfg: Mapping[str, Any] | None,
        *,
        arm: str,
        n_tasks: int = 1,
    ):
        if arm not in C.ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        super().__init__(spec, norm, cfg)
        self.arm = str(arm)
        self.n_tasks = int(n_tasks)
        if self.n_tasks < 1:
            raise ValueError("n_tasks must be >= 1")
        self.has_rho_shape = arm in RHO_ARMS
        self.has_contrib8 = arm in CONTRIB_ARMS
        self.has_chem8 = arm in CHEM8_ARMS

        width = int(self.cfg["width"])
        rho_in = int(spec.global_dim) + 2 * width + 1
        if self.has_rho_shape:
            rho_in += 2 * width

        if self.n_tasks != 1:
            # Same structural change as Sprint 02's LCRCMulti: only the two
            # output layers widen.  The encoder and local content are unchanged.
            self.f = PREP_MODEL.MLP([width, width, self.n_tasks])
            self.rho = PREP_MODEL.MLP([rho_in, int(self.cfg["rho_hidden"]), self.n_tasks])
        elif self.has_rho_shape:
            self.rho.lins[0] = _widen_linear(self.rho.lins[0], 2 * width)

        if self.has_contrib8:
            self.q = PREP_MODEL.MLP([width, width, 8])
            self.delta = PREP_MODEL.MLP([9, 32, self.n_tasks])
            # Exact BASE at step 0: delta = 0 for every graph and task.
            nn.init.zeros_(self.delta.lins[-1].weight)
            nn.init.zeros_(self.delta.lins[-1].bias)

    # -- forward ---------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        _a, _b, h = self._encode(batch)
        atom_batch = batch["atom_batch"]
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom

        max_h = None
        mad_h = None
        if self.has_rho_shape:
            # -inf padding is safe: every graph has at least one atom (asserted
            # by the cache validator), so the segment maximum is finite.
            max_h = h.new_full((n_graphs, width), -float("inf")).scatter_reduce_(
                0,
                atom_batch.unsqueeze(1).expand(-1, width),
                h,
                reduce="amax",
                include_self=True,
            )
            mad_h = h.new_zeros((n_graphs, width)).index_add_(
                0, atom_batch, (h - mean_h[atom_batch]).abs()
            ) / denom
        else:
            max_h = h.new_zeros((n_graphs, width))
            mad_h = h.new_zeros((n_graphs, width))

        contrib_atom = self.f(h)
        if contrib_atom.dim() == 1:
            contrib_atom = contrib_atom.reshape(-1, 1)
        contrib = h.new_zeros((n_graphs, self.n_tasks)).index_add_(0, atom_batch, contrib_atom)

        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_parts = [g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)]
        if self.has_rho_shape:
            rho_parts.extend([max_h, mad_h])
        rho_in = torch.cat(rho_parts, dim=1)
        rho = self.rho(rho_in)
        if rho.dim() == 1:
            rho = rho.reshape(-1, 1)

        delta = h.new_zeros((n_graphs, self.n_tasks))
        q_graph = h.new_zeros((n_graphs, 8))
        if self.has_contrib8:
            q_atom = self.q(h)
            q_graph = h.new_zeros((n_graphs, 8)).index_add_(0, atom_batch, q_atom)
            delta = self.delta(torch.cat([q_graph, torch.log1p(count).unsqueeze(1)], dim=1))

        y = contrib + rho + delta
        if self.n_tasks == 1:
            y = y.reshape(-1)

        if not return_aux:
            return y
        aux = {
            "contrib": contrib if self.n_tasks != 1 else contrib.reshape(-1),
            "rho": rho if self.n_tasks != 1 else rho.reshape(-1),
            "contrib_atom": contrib_atom if self.n_tasks != 1 else contrib_atom.reshape(-1),
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "max_h": max_h,
            "mad_h": mad_h,
            "delta": delta if self.n_tasks != 1 else delta.reshape(-1),
            "q_graph": q_graph,
            "h": h,
            "count": count,
            "a": _a,
            "b": _b,
        }
        return y, aux


def _widen_linear(lin: nn.Linear, extra_in: int) -> nn.Linear:
    """A zero-padded first layer: original columns copied, new columns zero."""
    new = nn.Linear(lin.in_features + int(extra_in), lin.out_features, bias=lin.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : lin.in_features].copy_(lin.weight)
        if lin.bias is not None and new.bias is not None:
            new.bias.copy_(lin.bias)
    return new


# ---------------------------------------------------------------------------
# construction / state copy
# ---------------------------------------------------------------------------


def build_base_model(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    seed: int,
    device,
    *,
    cfg: Mapping[str, Any] | None = None,
    dropout: float = float(C.RECIPE["dropout"]),
    n_tasks: int = 1,
):
    """Frozen BASE construction, optionally at a different capacity config."""
    torch.manual_seed(int(seed))
    if int(n_tasks) == 1:
        model: nn.Module = PREP_MODEL.LCRC(spec, norm, cfg)
    else:
        model = M.LCRCMulti(spec, norm, cfg, int(n_tasks))
    hook = M.DropoutAtLocal(model, float(dropout))
    model.to(device)
    meta = model_meta(model, spec, cfg, arm="base", n_tasks=int(n_tasks), dropout=float(dropout))
    meta["dropout_hook"] = hook
    return model, meta


def copy_base_state(
    variant: LCRCVariant,
    base_state: Mapping[str, torch.Tensor],
    *,
    arm: str,
    base_atom_width: int,
    spec_atom_width: int,
) -> dict[str, Any]:
    """Load every BASE tensor into the variant; widen only the marked columns."""
    var_state = variant.state_dict()
    copied = 0
    widened = 0
    skipped_new = 0
    for key, value in var_state.items():
        if key not in base_state:
            skipped_new += 1
            continue
        src = base_state[key]
        if tuple(src.shape) == tuple(value.shape):
            value.copy_(src)
            copied += 1
            continue
        if key == "atom_encoder.lins.0.weight" and src.shape[0] == value.shape[0]:
            if int(src.shape[1]) != int(base_atom_width) or int(value.shape[1]) != int(spec_atom_width):
                raise ValueError(
                    f"{key}: unexpected widening {tuple(src.shape)} -> {tuple(value.shape)} "
                    f"(declared widths {base_atom_width} -> {spec_atom_width})"
                )
            with torch.no_grad():
                value.zero_()
                value[:, : src.shape[1]].copy_(src)
            widened += 1
            continue
        if key == "rho.lins.0.weight" and arm in RHO_ARMS and src.shape[0] == value.shape[0]:
            with torch.no_grad():
                value.zero_()
                value[:, : src.shape[1]].copy_(src)
            widened += 1
            continue
        raise ValueError(f"cannot copy BASE tensor {key}: {tuple(src.shape)} -> {tuple(value.shape)}")
    if copied + widened + skipped_new != len(var_state):
        raise AssertionError("state copy accounting mismatch")
    return {
        "copied_identical": copied,
        "widened_zero_padded": widened,
        "new_extension_tensors": skipped_new,
        "base_atom_width": int(base_atom_width),
        "spec_atom_width": int(spec_atom_width),
    }


def build_variant_model(
    spec: FeatureSpec,
    base_spec: FeatureSpec,
    norm: Mapping[str, Any],
    seed: int,
    device,
    *,
    arm: str,
    cfg: Mapping[str, Any] | None = None,
    dropout: float = float(C.RECIPE["dropout"]),
    n_tasks: int = 1,
):
    """BASE-initialised variant: all shared tensors copied, extensions zero-init."""
    if arm == "base":
        raise ValueError("use build_base_model for the base arm")
    base_model, base_meta = build_base_model(
        base_spec, norm, seed, device, cfg=cfg, dropout=dropout, n_tasks=n_tasks
    )
    variant = LCRCVariant(spec, norm, cfg, arm=arm, n_tasks=int(n_tasks))
    copy_info = copy_base_state(
        variant,
        base_model.state_dict(),
        arm=arm,
        base_atom_width=base_spec.atom_feat_dim,
        spec_atom_width=spec.atom_feat_dim,
    )
    hook = M.DropoutAtLocal(variant, float(dropout))
    variant.to(device)
    meta = model_meta(variant, spec, cfg, arm=arm, n_tasks=int(n_tasks), dropout=float(dropout))
    meta["dropout_hook"] = hook
    meta["base_parameters"] = int(base_meta["parameters"])
    meta["extension_parameters"] = int(meta["parameters"] - base_meta["parameters"])
    meta["base_init_state_hash"] = base_meta["init_state_hash"]
    meta["base_copy"] = copy_info
    meta["extension_zero_init"] = verify_zero_init(variant, base_model, arm=arm, spec=spec)
    return variant, meta


def verify_zero_init(
    variant: LCRCVariant,
    base_model: nn.Module,
    *,
    arm: str,
    spec: FeatureSpec,
) -> dict[str, Any]:
    """Prove the extension cannot change step 0: widened columns and delta are zero."""
    base_state = base_model.state_dict()
    var_state = variant.state_dict()
    out: dict[str, Any] = {"arm": arm}
    if "atom_encoder.lins.0.weight" in var_state and "atom_encoder.lins.0.weight" in base_state:
        b = base_state["atom_encoder.lins.0.weight"]
        v = var_state["atom_encoder.lins.0.weight"]
        if v.shape != b.shape:
            extra = v[:, b.shape[1] :]
            out["atom_encoder_new_columns_max_abs"] = float(extra.abs().max()) if extra.numel() else 0.0
    if arm in RHO_ARMS:
        b = base_state["rho.lins.0.weight"]
        v = var_state["rho.lins.0.weight"]
        extra = v[:, b.shape[1] :]
        out["rho_new_columns_max_abs"] = float(extra.abs().max()) if extra.numel() else 0.0
    if arm in CONTRIB_ARMS:
        out["delta_last_weight_max_abs"] = float(variant.delta.lins[-1].weight.abs().max())
        out["delta_last_bias_max_abs"] = float(variant.delta.lins[-1].bias.abs().max())
    passed = True
    for key, value in out.items():
        if key.endswith("_max_abs") and float(value) != 0.0:
            passed = False
    out["passed"] = bool(passed)
    return out


def load_variant_model(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    device,
    *,
    arm: str,
    cfg: Mapping[str, Any] | None = None,
    dropout: float = float(C.RECIPE["dropout"]),
    n_tasks: int = 1,
):
    model = LCRCVariant(spec, norm, cfg, arm=arm, n_tasks=int(n_tasks))
    M.DropoutAtLocal(model, float(dropout))
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    model.to(device)
    model.eval()
    return model


def model_meta(
    model: nn.Module,
    spec: FeatureSpec,
    cfg: Mapping[str, Any] | None,
    *,
    arm: str,
    n_tasks: int,
    dropout: float,
) -> dict[str, Any]:
    width = int(model.cfg["width"])
    rho_in = int(spec.global_dim) + 2 * width + 1 + (2 * width if arm in RHO_ARMS else 0)
    breakdown: dict[str, int] = {}
    for name, module in model.named_children():
        if name.startswith("_pilot_dropout"):
            continue
        breakdown[name] = int(sum(p.numel() for p in module.parameters()))
    return {
        "name": C.ARM_LABEL[arm],
        "arm": arm,
        "schema": spec.name,
        "feature_spec_hash": spec.hash(),
        "n_outputs": int(n_tasks),
        "config": dict(model.cfg),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "parameters_breakdown": breakdown,
        "input_widths": {
            "atom_feat": int(spec.atom_feat_dim),
            "bond_feat": int(spec.bond_feat_dim),
            "global": int(spec.global_dim),
            "atom_scalars": int(spec.n_scalars),
            "n_nbr_codes": int(spec.n_nbr_codes),
            "n_path_codes": int(spec.n_path_codes),
            "rho_input": int(rho_in),
        },
        "init_state_hash": M.state_dict_fingerprint(model.state_dict()),
        "shared_encoder_state_hash": M.shared_encoder_state_hash(model),
        "dropout": float(dropout),
        "dropout_position": C.RECIPE["dropout_position"],
        "has_attention": False,
        "has_message_passing": False,
        "iterative_hidden_message_passing": False,
        "has_relation_head": False,
    }


def per_graph_stats_naive(h: np.ndarray, atom_batch: np.ndarray, n_graphs: int) -> dict[str, np.ndarray]:
    """Reference implementation used by the RHOSHAPE tests."""
    width = h.shape[1]
    mean = np.zeros((n_graphs, width), dtype=np.float64)
    mean2 = np.zeros_like(mean)
    mx = np.full((n_graphs, width), -np.inf, dtype=np.float64)
    mad = np.zeros_like(mean)
    count = np.zeros(n_graphs, dtype=np.float64)
    for g in range(n_graphs):
        rows = h[atom_batch == g].astype(np.float64)
        count[g] = rows.shape[0]
        mean[g] = rows.mean(0)
        mean2[g] = (rows * rows).mean(0)
        mx[g] = rows.max(0)
        mad[g] = np.abs(rows - mean[g]).mean(0)
    return {"count": count, "mean": mean, "mean2": mean2, "max": mx, "mad": mad}


__all__ = [
    "RHO_ARMS",
    "CONTRIB_ARMS",
    "CHEM8_ARMS",
    "LCRCVariant",
    "build_base_model",
    "build_variant_model",
    "copy_base_state",
    "verify_zero_init",
    "load_variant_model",
    "model_meta",
    "per_graph_stats_naive",
]
