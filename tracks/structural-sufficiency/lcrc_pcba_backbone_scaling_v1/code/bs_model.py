"""The scaled LCRC-A: one trunk encode + the frozen four-head attention readout.

Every module is the historical one, constructed with *explicit* widths:

* ``tu_model.TULCRC(variant="C1")`` -- atom encoder, bond encoder, the one-shot
  S1/S2 ``TULocalContent`` block, ``f`` and ``rho``.  ``cfg`` is passed
  explicitly, so no module can silently fall back to the historical
  ``width=64`` / ``scorer_hidden=32`` / ``pool_hidden=88`` constants.
* ``nmp_model.AttentionPoolReadout`` -- the historical LCRC-A readout, imported
  unchanged and given ``width``, ``scorer_hidden``, ``n_heads=4`` and
  ``pool_hidden`` explicitly.
* ``bp_model.C1Model``'s empty ``cont_mean`` / ``cont_scale`` buffers are
  reproduced so the state dict keys are exactly the historical ones.

The graph is exactly LCRC-A: ``a = atom_encoder(atom_feat)``,
``b = bond_encoder([au+av, |au-av|, au*av, bond_feat])``,
``h = dropout(local(a, b, scalars))``, ``logits = f(h) pooled + rho(global,
mean h, mean h^2, log1p n) + readout(h)``.  The post-``local`` path reads only
``h``, ``atom_batch``, the molecule count and ``global``; nothing is propagated
along ``bond_index`` a second time.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import bs_common as C
import tu_model as TU
import nmp_model as NM
from feature_spec import FeatureSpec


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class BSModel(TU.TULCRC):
    """Frozen LCRC-A graph at arbitrary width."""

    def __init__(
        self,
        spec: FeatureSpec,
        norm: Mapping[str, Any],
        n_out: int,
        cfg: Mapping[str, Any],
        *,
        cont_slice: tuple[int, int] | None = None,
        cont_mean: np.ndarray | None = None,
        cont_scale: np.ndarray | None = None,
    ):
        # ``TULCRC`` is constructed with the *explicit* cfg: every width
        # (atom/bond/local/stat/rho/width) comes from the caller.
        super().__init__(spec, norm, int(n_out), "C1", cfg=dict(cfg))
        self.cfg = dict(cfg)
        lo, hi = (0, 0) if cont_slice is None else (int(cont_slice[0]), int(cont_slice[1]))
        if hi < lo or lo < 0 or hi > spec.atom_feat_dim:
            raise ValueError("continuous atom slice outside atom_feat")
        self.cont_slice = (lo, hi)
        dim = hi - lo
        if dim:
            if cont_mean is None or cont_scale is None:
                raise ValueError("continuous atom attributes need Train-only mean/scale")
            mean = np.asarray(cont_mean, dtype=np.float32).reshape(-1)
            scale = np.asarray(cont_scale, dtype=np.float32).reshape(-1)
            if mean.shape != (dim,) or scale.shape != (dim,):
                raise ValueError("continuous atom normaliser shape")
        else:
            mean = np.zeros(0, dtype=np.float32)
            scale = np.ones(0, dtype=np.float32)
        # Same (empty for PCBA) buffers as ``bp_model.C1Model`` -> identical keys.
        self.register_buffer("cont_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("cont_scale", torch.as_tensor(scale, dtype=torch.float32))
        # Historical LCRC-A readout, widths passed explicitly.
        self.readout = NM.AttentionPoolReadout(
            width=int(self.cfg["width"]),
            hidden=int(self.cfg["scorer_hidden"]),
            n_heads=int(self.cfg["n_heads"]),
            pool_hidden=int(self.cfg["pool_hidden"]),
            n_out=int(n_out),
        )

    # -- encoding (identical to C1Model._encode) ---------------------------

    def _encode(self, batch: Mapping[str, torch.Tensor]):
        lo, hi = self.cont_slice
        if hi > lo:
            atom_feat = batch["atom_feat"]
            normalized = atom_feat.clone()
            normalized[:, lo:hi] = (normalized[:, lo:hi] - self.cont_mean) / self.cont_scale
            batch = dict(batch)
            batch["atom_feat"] = normalized
        return super()._encode(batch)

    # -- forward -----------------------------------------------------------

    def encode_h(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(a, b, scal, h)`` -- exactly one atom/bond encode and one ``local`` call."""
        a, b, scal = self._encode(batch)
        h = self.drop(self.local(batch, a, b, scal))
        return a, b, scal, h

    def post_local(
        self,
        h: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        return_aux: bool = False,
        return_attention: bool = False,
    ):
        """Everything after ``local``.  Needs no bond/path/nbr field at all."""
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        residual, attention, pooled = self.readout(
            h, atom_batch, n_graphs, return_attention=bool(return_attention)
        )
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib = h.new_zeros(n_graphs, self.n_classes).index_add_(0, atom_batch, self.f(h))
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in)
        logits = contrib + rho + residual
        if return_aux:
            return logits, {
                "h": h,
                "contrib": contrib,
                "rho": rho,
                "residual_logits": residual,
                "mean_h": mean_h,
                "mean_h2": mean_h2,
                "attention_weights": attention,
                "attention_pooled": pooled,
            }
        return logits

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        _a, _b, _scal, h = self.encode_h(batch)
        if return_aux:
            return self.post_local(h, batch, return_aux=True, return_attention=True)
        return self.post_local(h, batch)


# ---------------------------------------------------------------------------
# state construction: FULL -> C1 remap at arbitrary width
# ---------------------------------------------------------------------------


def scaled_full_state(
    spec: FeatureSpec, norm: Mapping[str, Any], n_out: int, cfg: Mapping[str, Any], seed: int
) -> dict[str, torch.Tensor]:
    """``torch.manual_seed(seed)`` then build the scale's FULL trunk on CPU.

    This is exactly ``tu_model.build_full_state`` generalised to ``cfg``.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = TU.TULCRC(spec, norm, int(n_out), "FULL", cfg=dict(cfg))
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def remap_full_to_c1(
    full: Mapping[str, torch.Tensor], spec: FeatureSpec, cfg: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Generalised ``tu_model.remap_state('C1', ...)``: drop the b_mean columns."""
    layout_full = TU.local_layout(spec, "FULL", cfg)
    layout_c1 = TU.local_layout(spec, "C1", cfg)
    removed = set(range(*layout_full["b_mean"]))
    keep_full = [i for i in range(layout_full["atom_scalar"][1]) if i not in removed]
    expected_keep: list[int] = []
    for name in layout_c1:
        lo, hi = layout_full[name]
        expected_keep.extend(range(lo, hi))
    if keep_full != expected_keep:
        raise RuntimeError("C1 column layout is not FULL minus b_mean")
    out = {k: v.detach().cpu().clone() for k, v in full.items()}
    out["local.mlp.lins.0.weight"] = full["local.mlp.lins.0.weight"][:, keep_full].detach().cpu().clone()
    expected_in = len(TU.local_columns(spec, "C1", cfg))
    if out["local.mlp.lins.0.weight"].shape[1] != expected_in:
        raise RuntimeError("remapped local MLP width mismatch")
    if out["rho.lins.0.weight"].shape[0] != int(cfg["rho_hidden"]):
        raise RuntimeError("remapped rho hidden width mismatch")
    if out["local.mlp.lins.1.weight"].shape[0] != int(cfg["width"]):
        raise RuntimeError("remapped local output width mismatch")
    return out


def initialize_readout(model: BSModel, *, seed: int, zero_residual: bool = True) -> dict[str, Any]:
    """Historical A readout init: private generator, last Linear zeroed."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + 17)
    before = torch.get_rng_state().clone()
    n_linear = 0
    with torch.no_grad():
        for sub in model.readout.modules():
            if isinstance(sub, nn.Linear):
                bound = 1.0 / math.sqrt(float(sub.in_features))
                sub.weight.uniform_(-bound, bound, generator=gen)
                if sub.bias is not None:
                    sub.bias.uniform_(-bound, bound, generator=gen)
                n_linear += 1
    if zero_residual:
        with torch.no_grad():
            model.readout.fc2.weight.zero_()
            model.readout.fc2.bias.zero_()
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("private readout initialisation changed the global torch RNG state")
    return {
        "private_generator_seed": int(seed) * 1_000_003 + 17,
        "linear_layers_initialised": n_linear,
        "readout_output_zeroed": bool(zero_residual),
        "global_rng_unchanged": True,
    }


def _base_state_keys(model: nn.Module) -> set[str]:
    return {k for k in model.state_dict() if not k.startswith("readout.")}


def build_model(
    scale_or_key: int | str,
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int = C.MODEL_SEED,
    device: torch.device | str = "cpu",
    verify_parameters: bool = True,
) -> tuple[BSModel, dict[str, Any]]:
    """Build one scaled LCRC-A with the frozen initialisation."""
    if isinstance(scale_or_key, str):
        if scale_or_key not in C.MODELS:
            raise KeyError(scale_or_key)
        scale = int(C.MODELS[scale_or_key]["scale"])
        label = scale_or_key
    else:
        scale = int(scale_or_key)
        label = f"scale{scale}"
    cfg = C.config_for_scale(scale)
    model = BSModel(spec, norm, int(n_out), cfg)
    full_state = scaled_full_state(spec, norm, int(n_out), cfg, int(seed))
    c1_state = remap_full_to_c1(full_state, spec, cfg)
    c1_state["cont_mean"] = torch.zeros(0, dtype=torch.float32)
    c1_state["cont_scale"] = torch.ones(0, dtype=torch.float32)
    missing, unexpected = model.load_state_dict(c1_state, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith("readout.")}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"trunk init load mismatch: missing={sorted(missing)} "
            f"expected_readout={sorted(expected_missing)} unexpected={sorted(unexpected)}"
        )
    base_fp_before = C.state_fingerprint({k: v for k, v in model.state_dict().items() if k in _base_state_keys(model)})
    init_audit = initialize_readout(model, seed=int(seed), zero_residual=True)
    base_fp_after = C.state_fingerprint({k: v for k, v in model.state_dict().items() if k in _base_state_keys(model)})
    if base_fp_after != base_fp_before:
        raise RuntimeError("readout initialisation changed the trunk state")
    model.to(torch.device(device))

    total = C.parameter_count(model)
    base_elements = int(sum(p.numel() for n, p in model.named_parameters() if not n.startswith("readout.")))
    readout_elements = int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("readout.")))
    audit: dict[str, Any] = {
        "model": label,
        "scale": scale,
        "widths": cfg,
        "seed": int(seed),
        "init_rule": C.INIT_SEED_RULE,
        "init": init_audit,
        "trunk_init_fingerprint": base_fp_before,
        "trunk_init_fingerprint_after_readout_init": base_fp_after,
        "full_trunk_init_fingerprint": C.state_fingerprint(full_state),
        "state_keys": sorted(model.state_dict()),
        "n_state_tensors": len(model.state_dict()),
        "parameter_tensors": len(list(model.parameters())),
        "base_parameters": base_elements,
        "readout_parameters": readout_elements,
        "total_parameters": total,
        "expected_parameters": int(C.EXPECTED_PARAMETERS[scale]),
        "ref_model_a0_parameters": int(C.A0["parameters"]),
        "width_audit": width_audit(model, cfg),
    }
    if verify_parameters:
        if total != int(C.EXPECTED_PARAMETERS[scale]):
            raise RuntimeError(f"scale {scale} parameters {total} != {C.EXPECTED_PARAMETERS[scale]}")
        if base_elements != int(C.EXPECTED_BASE_PARAMETERS[scale]):
            raise RuntimeError(f"scale {scale} trunk parameters {base_elements} != {C.EXPECTED_BASE_PARAMETERS[scale]}")
        if readout_elements != int(C.EXPECTED_READOUT_PARAMETERS[scale]):
            raise RuntimeError(
                f"scale {scale} readout parameters {readout_elements} != {C.EXPECTED_READOUT_PARAMETERS[scale]}"
            )
        audit["parameter_check"] = "exact"
    return model, audit


# ---------------------------------------------------------------------------
# width audit
# ---------------------------------------------------------------------------


def _lin(module: nn.Linear) -> dict[str, int]:
    return {"in": int(module.in_features), "out": int(module.out_features), "params": int(module.weight.numel() + (module.bias.numel() if module.bias is not None else 0))}


def width_audit(model: BSModel, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Actual tensor widths of every module, plus the protocol expectation."""
    local = model.local
    local_in = int(local.mlp.lins[0].weight.shape[1])
    expected_local_in = int(3 * cfg["atom_out"] + 4 * cfg["stat_rank"] + 8 + model.spec.n_scalars)
    observed: dict[str, Any] = {
        "atom_hidden": _lin(model.atom_encoder.lins[0]),
        "atom_out": _lin(model.atom_encoder.lins[1]),
        "bond_hidden": _lin(model.bond_encoder.lins[0]),
        "bond_out": _lin(model.bond_encoder.lins[1]),
        "local_hidden": _lin(local.mlp.lins[0]),
        "local_width": _lin(local.mlp.lins[1]),
        "local_norm": int(local.norm.normalized_shape[0]),
        "local_input": {
            "observed": local_in,
            "expected_from_widths": expected_local_in,
            "layout": {k: list(v) for k, v in TU.local_layout(model.spec, "C1", cfg).items()},
        },
        "stat_rank": {
            "nbr_emb": list(local.nbr_emb.weight.shape),
            "path_emb": list(local.path_emb.weight.shape),
            "end_emb": list(local.end_emb.weight.shape),
            "path_u": list(local.path_u.weight.shape),
            "path_v": list(local.path_v.weight.shape),
            "joint_proj": _lin(local.joint_proj),
            "close_emb": list(local.close_emb.weight.shape),
            "path_gate": list(local.path_gate.weight.shape),
        },
        "f": [_lin(model.f.lins[0]), _lin(model.f.lins[1])],
        "rho_hidden": _lin(model.rho.lins[0]),
        "rho_out": _lin(model.rho.lins[1]),
        "rho_input": int(model.rho.lins[0].weight.shape[1]),
        "scorer_hidden": [_lin(h.fc1) for h in model.readout.heads],
        "scorer_out": [_lin(h.fc2) for h in model.readout.heads],
        "n_heads": int(model.readout.n_heads),
        "pool_hidden": _lin(model.readout.fc1),
        "readout_out": _lin(model.readout.fc2),
    }
    expected = {
        "atom_hidden": int(cfg["atom_hidden"]),
        "atom_out": int(cfg["atom_out"]),
        "bond_hidden": int(cfg["bond_hidden"]),
        "bond_out": int(cfg["bond_out"]),
        "local_hidden": int(cfg["local_hidden"]),
        "local_width": int(cfg["width"]),
        "stat_rank": int(cfg["stat_rank"]),
        "rho_hidden": int(cfg["rho_hidden"]),
        "rho_input": int(model.spec.global_dim + 2 * int(cfg["width"]) + 1),
        "scorer_hidden": int(cfg["scorer_hidden"]),
        "pool_hidden": int(cfg["pool_hidden"]),
        "n_heads": int(cfg["n_heads"]),
        "readout_out": int(cfg["n_out"]),
        "local_input": expected_local_in,
    }
    checks = {
        "atom_hidden": observed["atom_hidden"]["out"] == expected["atom_hidden"],
        "atom_out": observed["atom_out"]["out"] == expected["atom_out"],
        "bond_hidden": observed["bond_hidden"]["out"] == expected["bond_hidden"],
        "bond_out": observed["bond_out"]["out"] == expected["bond_out"],
        "local_hidden": observed["local_hidden"]["out"] == expected["local_hidden"],
        "local_width": observed["local_width"]["out"] == expected["local_width"],
        "local_norm": observed["local_norm"] == expected["local_width"],
        "local_input": observed["local_input"]["observed"] == expected["local_input"],
        "stat_rank_nbr_emb": observed["stat_rank"]["nbr_emb"][1] == expected["stat_rank"],
        "stat_rank_path_emb": observed["stat_rank"]["path_emb"][1] == expected["stat_rank"],
        "stat_rank_end_emb": observed["stat_rank"]["end_emb"][1] == expected["stat_rank"],
        "stat_rank_path_u": observed["stat_rank"]["path_u"][1] == expected["stat_rank"],
        "stat_rank_path_v": observed["stat_rank"]["path_v"][1] == expected["stat_rank"],
        "stat_rank_joint_proj": observed["stat_rank"]["joint_proj"]["out"] == expected["stat_rank"],
        "f_width": observed["f"][0]["out"] == expected["local_width"],
        "f_out": observed["f"][1]["out"] == int(model.n_classes),
        "rho_hidden": observed["rho_hidden"]["out"] == expected["rho_hidden"],
        "rho_out": observed["rho_out"]["out"] == int(model.n_classes),
        "rho_input": observed["rho_input"] == expected["rho_input"],
        "scorer_hidden": all(x["out"] == expected["scorer_hidden"] for x in observed["scorer_hidden"]),
        "scorer_in": all(x["in"] == expected["local_width"] for x in observed["scorer_hidden"]),
        "n_heads": observed["n_heads"] == expected["n_heads"],
        "pool_hidden": observed["pool_hidden"]["out"] == expected["pool_hidden"],
        "pool_in": observed["pool_hidden"]["in"] == expected["n_heads"] * expected["local_width"],
        "readout_out": observed["readout_out"]["out"] == expected["readout_out"],
    }
    return {"observed": observed, "expected": expected, "checks": checks, "all_match": all(checks.values())}


# ---------------------------------------------------------------------------
# optimizer / LR / state helpers (inherited, unchanged)
# ---------------------------------------------------------------------------

state_cpu = NM.state_cpu
optimizer_state_cpu = NM.optimizer_state_cpu
restore_rng_state = NM.restore_rng_state


def make_optimizer(model: nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.RECIPE["warmup_start_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    audit = {
        "optimizer": "AdamW",
        "mode": "from_scratch",
        "n_param_groups": len(optimizer.param_groups),
        "n_parameters": int(sum(p.numel() for p in params)),
        "n_tensors": int(len(params)),
        "peak_lr": float(C.RECIPE["learning_rate"]),
        "warmup_start_lr": float(C.RECIPE["warmup_start_lr"]),
        "betas": list(C.RECIPE["adam_betas"]),
        "eps": float(C.RECIPE["adam_eps"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "scheduler": "per_step_cosine_40epoch_with_1epoch_linear_warmup",
    }
    return optimizer, audit


def set_lrs(optimizer: torch.optim.Optimizer, *, step: int) -> float:
    lr = float(C.scheduled_lr(int(step), phase=3, kind="all"))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# audit helpers: atom renumbering
# ---------------------------------------------------------------------------


def permute_atoms(
    batch: Mapping[str, Any], *, seed: int = 20261108
) -> tuple[dict[str, Any], torch.Tensor]:
    """Renumber the atoms inside every molecule; returns ``(new_batch, p)``.

    ``p[i]`` is the old atom id that becomes new atom id ``i``.  All index
    tensors are rebuilt so the molecule is the same object with a different
    internal order.
    """
    n_graphs = int(batch["n_graphs"])
    atom_batch = batch["atom_batch"]
    counts = torch.bincount(atom_batch, minlength=n_graphs).tolist()
    gen = torch.Generator().manual_seed(int(seed))
    parts: list[torch.Tensor] = []
    off = 0
    for g in range(n_graphs):
        n = int(counts[g])
        parts.append(torch.arange(off, off + n, dtype=torch.long)[torch.randperm(n, generator=gen)])
        off += n
    p = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)
    p_inv = torch.empty_like(p)
    p_inv[p] = torch.arange(p.numel(), dtype=torch.long)

    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.clone() if torch.is_tensor(value) else value
    for key in ("atom_feat", "atom_scalar", "atom_type"):
        if key in out:
            out[key] = out[key][p]
    if "bond_index" in out and out["bond_index"].numel():
        out["bond_index"] = p_inv[out["bond_index"]]
    for key in ("nbr_atom", "path_atom"):
        if key in out and out[key].numel():
            out[key] = p_inv[out[key]]
    if "path_end" in out and out["path_end"].numel():
        out["path_end"] = p_inv[out["path_end"]]
    return out, p


def count_module_calls(model: BSModel, batch: Mapping[str, Any]) -> dict[str, int]:
    """Count how often each encoding module runs in one forward."""
    counters = {"atom_encoder": 0, "bond_encoder": 0, "local": 0, "f": 0, "rho": 0, "readout": 0}
    handles = []
    for name, module in (
        ("atom_encoder", model.atom_encoder),
        ("bond_encoder", model.bond_encoder),
        ("local", model.local),
        ("f", model.f),
        ("rho", model.rho),
        ("readout", model.readout),
    ):
        def _hook(_m, _inp, _out, key=name):
            counters[key] += 1

        handles.append(module.register_forward_hook(_hook))
    try:
        model.eval()
        with torch.no_grad():
            model(batch)
    finally:
        for h in handles:
            h.remove()
    return counters


__all__ = [
    "BSModel",
    "build_model",
    "count_module_calls",
    "initialize_readout",
    "make_optimizer",
    "optimizer_state_cpu",
    "permute_atoms",
    "remap_full_to_c1",
    "restore_rng_state",
    "scaled_full_state",
    "set_lrs",
    "state_cpu",
    "width_audit",
]
