"""COC-v1 model: DCC-L with a shared-parameter counterfactual local composer.

The only forward change is

    c_loc = local_composer(x_real) - local_composer(x_null)

with the same ``local_composer`` module.  Everything else is the production
DualChannel.  The class adds no submodules, so its initial state has exactly the
same keys and values as DCC-L for the same seed and normalization.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Mapping

import torch

_HERE = Path(__file__).resolve().parent
_SUFF = _HERE.parents[1]
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_V2 = _SUFF / "zinc_recipe_trunk_v2" / "code"
_DCC = _SUFF / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad2_models as A2  # noqa: E402
import dcc_models as DM  # noqa: E402

WIDTH = DM.WIDTH
GLOBAL_DIM = DM.GLOBAL_DIM


class COCDualChannel(DM.DualChannel):
    """DualChannel whose local channel is f(real) - f(mean-field), f shared."""

    def local_input(self, b: torch.Tensor, W: torch.Tensor, V: torch.Tensor,
                    n_b: torch.Tensor, bond_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([b, W, V, torch.log1p(n_b).unsqueeze(1), bond_feat], dim=1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        x_null = self.local_input(b, Wg, Vg, n_b, batch["bond_feat"])
        c_glob = self.group_composer(x_null)

        Wl, Vl, _ = self.local_grouping(w, batch, organized=bool(self.organized_local))
        x_real = self.local_input(b, Wl, Vl, n_b, batch["bond_feat"])
        # Same module, same parameters, full autograd on both calls.
        c_loc = self.local_composer(x_real) - self.local_composer(x_null)

        B = int(batch["n_graphs"])
        z_glob = c_glob.new_zeros((B, c_glob.shape[1])).index_add_(
            0, batch["bond_batch"], c_glob)
        zl = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(
            0, batch["bond_batch"], c_loc)
        zl2 = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(
            0, batch["bond_batch"], c_loc * c_loc)
        a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        z = torch.cat([z_glob, a_sum, a_sq, g, zl, zl2], dim=1)
        return self.reader(z).view(-1)


def _param_count(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for _, p in module.named_parameters()))


def build_initial_state(norm: Mapping[str, object], seed: int) -> tuple[COCDualChannel, dict]:
    """Mirror ``dcc_models.build_initial_state`` with the COC forward class."""
    torch.manual_seed(int(seed))
    p3 = A2.P3NullModel(norm)
    ssl_path = C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
    ssl_info = A2._load_encoders(p3, ssl_path, torch.device("cpu"))
    p3_state = C.cpu_state(p3)

    torch.manual_seed(int(seed) + 90731)
    dual = COCDualChannel(norm)
    dual.load_p3_state(p3_state)
    info = {
        "arm_neutral_builder": True,
        "architecture": "coc_counterfactual_ownership_composer",
        "seed": int(seed),
        "ssl": ssl_info,
        "p3_reference_state_sha256": C.state_hash(p3),
        "initial_state_sha256": C.state_hash(dual),
        "parameters": _param_count(dual),
        "parameters_breakdown": {
            "p3_base_total": _param_count(p3),
            "encoders": _param_count(dual.encoders),
            "group_composer": _param_count(dual.group_composer),
            "local_composer": _param_count(dual.local_composer),
            "reader": _param_count(dual.reader),
        },
    }
    return dual, info


def build_model(candidate: str, norm: Mapping[str, object], seed: int, device,
                *, organized_local: bool = True):
    shared, base_info = build_initial_state(norm, seed)
    model = copy.deepcopy(shared)
    model.organized_local = bool(organized_local)
    model.to(device)
    info = dict(base_info)
    info.update({
        "candidate": candidate,
        "organized_local": bool(organized_local),
        "parameters": _param_count(model),
        "parameters_total": int(len(list(model.named_parameters()))),
        "requires_grad_true": int(sum(1 for p in model.parameters() if p.requires_grad)),
    })
    return model, info


def load_model_from_state(norm: Mapping[str, object], state: Mapping[str, torch.Tensor],
                          device) -> COCDualChannel:
    model = COCDualChannel(norm)
    model.organized_local = True
    C.load_state(model, state, device)
    model.to(device)
    model.eval()
    return model
