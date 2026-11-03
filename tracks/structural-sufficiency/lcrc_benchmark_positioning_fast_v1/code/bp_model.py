"""LCRC-C1 (NO_BOND_MEAN) construction and input-adapter normalisation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import bp_common as C
import tu_model as M
from feature_spec import FeatureSpec


class C1Model(M.TULCRC):
    """``TULCRC`` variant ``C1`` plus Train-only standardisation of continuous node attributes.

    The only structural deletion relative to LCRC-A is the ``b_mean`` block.
    The extra buffers normalise the raw continuous TU node-attribute columns
    (ENZYMES / PROTEINS); no parameter, module or graph computation is added.
    """

    def __init__(
        self,
        spec: FeatureSpec,
        norm: Mapping[str, Any],
        n_out: int,
        *,
        cont_slice: tuple[int, int] | None = None,
        cont_mean: np.ndarray | None = None,
        cont_scale: np.ndarray | None = None,
    ):
        super().__init__(spec, norm, int(n_out), "C1")
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
        self.register_buffer("cont_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("cont_scale", torch.as_tensor(scale, dtype=torch.float32))

    def _encode(self, batch: Mapping[str, torch.Tensor]):
        lo, hi = self.cont_slice
        if hi > lo:
            atom_feat = batch["atom_feat"]
            normalized = atom_feat.clone()
            normalized[:, lo:hi] = (normalized[:, lo:hi] - self.cont_mean) / self.cont_scale
            batch = dict(batch)
            batch["atom_feat"] = normalized
        return super()._encode(batch)


def build_c1(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    n_out: int,
    seed: int,
    device: torch.device | str,
    *,
    cont_slice: tuple[int, int] | None = None,
    cont_mean: np.ndarray | None = None,
    cont_scale: np.ndarray | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build LCRC-C1 with the frozen FULL -> C1 seed mapping."""
    full_state = M.build_full_state(spec, norm, int(n_out), int(seed))
    init_state = M.remap_state("C1", full_state, spec, int(n_out))
    if cont_slice is not None and int(cont_slice[1]) > int(cont_slice[0]):
        init_state["cont_mean"] = torch.as_tensor(np.asarray(cont_mean, dtype=np.float32))
        init_state["cont_scale"] = torch.as_tensor(np.asarray(cont_scale, dtype=np.float32))
    else:
        init_state["cont_mean"] = torch.zeros(0, dtype=torch.float32)
        init_state["cont_scale"] = torch.ones(0, dtype=torch.float32)
    model = C1Model(
        spec,
        norm,
        int(n_out),
        cont_slice=cont_slice,
        cont_mean=cont_mean,
        cont_scale=cont_scale,
    )
    model.load_state_dict(init_state, strict=True)
    model.to(torch.device(device))
    shared = M.shared_state_equal("C1", full_state, {k: v for k, v in init_state.items() if not k.startswith("cont_")}, spec)
    info = {
        "variant": "C1",
        "variant_definition": "LCRC-A minus the 96-d b_mean = b_sum/deg local-input block; no other change",
        "seed": int(seed),
        "n_outputs": int(n_out),
        "parameters": int(M.parameter_count(model)),
        "parameters_breakdown": M.module_breakdown(model),
        "local_layout": {k: list(v) for k, v in M.local_layout(spec, "C1").items()},
        "local_input_dim": int(len(M.local_columns(spec, "C1"))),
        "rho_input_dim": int(spec.global_dim + 2 * int(C.MODEL_CONFIG["width"]) + 1),
        "init_state_hash": M.state_fingerprint(init_state),
        "full_init_state_hash": M.state_fingerprint(full_state),
        "shared_with_full": shared,
        "cont_slice": [int(cont_slice[0]), int(cont_slice[1])] if cont_slice is not None else [0, 0],
        "cont_norm_in_state": bool(cont_slice is not None and int(cont_slice[1]) > int(cont_slice[0])),
        "input_widths": {
            "atom_feat": spec.atom_feat_dim,
            "bond_feat": spec.bond_feat_dim,
            "global": spec.global_dim,
            "scalars": spec.n_scalars,
            "n_nbr_codes": spec.n_nbr_codes,
            "n_path_codes": spec.n_path_codes,
        },
    }
    return model, info


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    return M.state_fingerprint(state)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


__all__ = ["C1Model", "build_c1", "state_fingerprint", "parameter_count"]
