"""DIRECT / SHARE predictor for CSS-v0.

The frozen R2/SENTENCE_BASE trunk ``GwicModel(use_words=False)`` is reused
unchanged, except that its backbone MLP now also receives a 32-dimensional
branch:

    h_v = BackboneNorm(BackboneMLP(concat(x_v, r_v)))
    r_v = Linear(64, 32)(normalize(t_v))

The branch is injected at the *first* backbone-MLP layer, so every original
atom/bond embedding, S1/S2 encoder, radius-2 summary, local MLP, nonlinear graph
readout and all original computation paths are preserved.  DIRECT and SHARE
instantiate the same modules, the same parameter count, the same deterministic
initial state and the same shared training batch plan; they differ only in
``t_v`` (``z_v`` vs ``q_v``).

No message passing, transformer, teacher, distillation or pretraining is used.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import css_common as C
import gwic_model as GM

BRANCH_IN = 64
BRANCH_OUT = 32


class _BranchLinear(nn.Linear):
    """First backbone-MLP layer: concatenates the precomputed branch input."""

    def __init__(self, base_in: int, branch_in: int, out_features: int) -> None:
        super().__init__(int(base_in) + int(branch_in), int(out_features))
        self.base_in = int(base_in)
        self.branch_in = int(branch_in)
        self._branch: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._branch is None:
            raise RuntimeError("branch input missing before backbone MLP")
        if self._branch.shape[0] != x.shape[0]:
            raise ValueError("branch/atom batch mismatch")
        return super().forward(torch.cat([x, self._branch], dim=1))


class CssModel(GM.GwicModel):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 branch_in: int = BRANCH_IN, branch_out: int = BRANCH_OUT,
                 **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        kwargs["use_words"] = False
        super().__init__(atom_dims, bond_dims, **kwargs)
        self.branch_in_dim = int(branch_in)
        self.branch_out_dim = int(branch_out)
        old_first = self.backbone_mlp[0]
        if not isinstance(old_first, nn.Linear):
            raise TypeError("unexpected backbone MLP layout")
        self.backbone_mlp[0] = _BranchLinear(self.backbone_in_dim, self.branch_out_dim,
                                             old_first.out_features)
        self.branch_proj = nn.Linear(self.branch_in_dim, self.branch_out_dim)

    # -- deterministic initialisation ---------------------------------------
    def init_parameters(self, seed: int, last_std: float = 1e-3) -> dict[str, str]:
        super().init_parameters(int(seed), last_std)
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 97)
        with torch.no_grad():
            bound = 1.0 / math.sqrt(float(self.branch_in_dim))
            self.branch_proj.weight.uniform_(-bound, bound, generator=gen)
            self.branch_proj.bias.uniform_(-bound, bound, generator=gen)
        return self.shared_hashes()

    def shared_hashes(self) -> dict[str, str]:
        out = super().shared_hashes()
        out["branch_proj"] = C.module_hashes({"branch_proj": self.branch_proj})["branch_proj"]
        out["backbone_first"] = C.module_hashes(
            {"backbone_first": self.backbone_mlp[0]})["backbone_first"]
        return out

    def parameter_ledger(self) -> dict[str, int]:
        out = super().parameter_ledger()
        out["branch_proj"] = int(sum(p.numel() for p in self.branch_proj.parameters()))
        out["total"] = int(out["total"] + out["branch_proj"])
        return out

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *,
                arm: str = "SENTENCE_BASE", use_geometry: bool | None = None,
                geometry_override: torch.Tensor | None = None,
                return_aux: bool = False) -> dict[str, Any]:
        if arm != "SENTENCE_BASE":
            raise ValueError("CssModel is a SENTENCE_BASE trunk with a branch")
        branch = batch.get("branch")
        if branch is None:
            raise KeyError("batch is missing the precomputed branch input")
        first = self.backbone_mlp[0]
        first._branch = self.branch_proj(branch)
        try:
            out = super().forward(batch, arm="SENTENCE_BASE", return_aux=return_aux)
        finally:
            first._branch = None
        return out


def build_css_model(atom_dims: Sequence[int], bond_dims: Sequence[int], seed: int,
                    norm: Mapping[str, np.ndarray], config: Mapping[str, Any]
                    ) -> tuple[CssModel, dict[str, Any]]:
    dims = dict(config.get("dims", {}))
    model = CssModel(
        atom_dims, bond_dims,
        atom_dim=int(dims.get("atom", 48)), bond_dim=int(dims.get("bond", 24)),
        h_dim=int(dims.get("sentence", 96)), word_dim=int(dims.get("word", 64)),
        q_dim=int(dims.get("q", 64)), rank=int(dims.get("path_rank", 24)),
        end_dim=int(dims.get("path_end", 16)), close_dim=int(dims.get("path_close", 8)),
        nbr_dim=int(dims.get("nbr", 16)),
        backbone_hidden=int(dims.get("backbone_hidden", 192)),
        word_hidden=int(dims.get("word_hidden", 160)),
        score_hidden=int(dims.get("score_hidden", 96)),
        q_hidden=int(dims.get("q_hidden", 128)),
        readout_hidden=int(dims.get("readout_hidden", 192)),
        readout_mid=int(dims.get("readout_mid", 96)),
        dropout=float(dims.get("dropout", 0.1)))
    model.register_buffer("atom_scalar_mean", torch.as_tensor(norm["atom_scalar_mean"], dtype=torch.float32))
    model.register_buffer("atom_scalar_scale", torch.as_tensor(norm["atom_scalar_scale"], dtype=torch.float32))
    model.register_buffer("graph_mean", torch.as_tensor(norm["graph_mean"], dtype=torch.float32))
    model.register_buffer("graph_scale", torch.as_tensor(norm["graph_scale"], dtype=torch.float32))
    init_hashes = model.init_parameters(int(seed), last_std=1e-3)
    ledger = model.parameter_ledger()
    info = {
        "seed": int(seed),
        "config": dims,
        "parameters": ledger,
        "frozen_trunk_parameters": int(C.FROZEN_PREDICTOR_PARAMS),
        "branch_parameters": int(ledger["branch_proj"] + 32 * 192),
        "init_hashes": init_hashes,
        "backbone_in_dim": int(model.backbone_in_dim),
        "backbone_branch_in_dim": int(model.backbone_in_dim + model.branch_out_dim),
        "readout_in_dim": int(model.readout_in_dim),
    }
    return model, info


def branch_gather(rows: Sequence[int], structs: Mapping[int, Any], table: np.ndarray,
                  row_atom_ptr: np.ndarray, device: Any = "cpu") -> torch.Tensor:
    """Gather the per-atom branch table (canonical atom order) for a batch.

    ``row_atom_ptr[row]`` is the global first-atom offset of molecule ``row``;
    the table is the concatenation of all molecules in ascending row order, with
    atoms stored in the canonical order used by ``ses_structure``.
    """
    pieces = []
    for row in rows:
        lo = int(row_atom_ptr[int(row)])
        pieces.append(np.arange(lo, lo + int(structs[int(row)].n_atoms), dtype=np.int64))
    index = np.concatenate(pieces) if pieces else np.zeros(0, np.int64)
    return torch.as_tensor(np.asarray(table, np.float32)[index], dtype=torch.float32,
                           device=device)
