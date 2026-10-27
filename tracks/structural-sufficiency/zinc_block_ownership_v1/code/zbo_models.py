"""BOSI-B plus a shared block-ownership channel.

The BOSI-B body is imported unchanged.  ``BlockBOSI`` calls
``bosi_models.BOSI.__init__(arm="B")`` first, so all six historical MLPs are
constructed in the historical order and consume the same RNG stream.  Only
after that it adds:

* a per-block ``LayerNorm(196)`` + shared ``196 -> 64 -> 32`` MLP;
* a 32-column reader extension, implemented as a zero-initialised
  ``Linear(256 <- 32, bias=False)`` added to the old first reader layer.

The split reader is algebraically identical to a 449-column first layer whose
32 new columns start at zero, but it makes the step-0 forward bit-identical to
the original BOSI-B (no BLAS reduction-order change from the wider matrix).

No learned junction token, no recursion, no attention, no block-cut-tree
distance/overlap, no old BlockModel features.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import bosi_models as BM
import zbo_blocks as ZB
import zbo_common as BC

WIDTH = BM.WIDTH
N_ATOM_FEAT = BM.N_ATOM_FEAT
BLOCK_IN = BC.BLOCK_IN
BLOCK_HIDDEN = BC.BLOCK_HIDDEN
BLOCK_OUT = BC.BLOCK_OUT
READER_OLD_IN = BC.READER_OLD_IN
ARMS = ("R", "N", "S")


class BlockBOSI(BM.BOSI):
    """BOSI-B with a 32-d pooled block channel.

    ``block_arm`` is the only switch and is non-parametric: it selects how the
    per-block first/second moments are constructed.  The underlying BOSI
    relation statistics always use the frozen B arm (``self.arm == "B"``).
    """

    def __init__(self, norm: Mapping[str, object], *, block_arm: str = "R",
                 dropout: float = 0.0):
        if block_arm not in ARMS:
            raise ValueError(f"unknown block arm {block_arm!r}")
        super().__init__(norm, arm="B", dropout=dropout)
        self.block_arm = block_arm
        self.block_norm = nn.LayerNorm(BLOCK_IN)
        self.block_mlp = nn.Sequential(
            nn.Linear(BLOCK_IN, BLOCK_HIDDEN),
            nn.SiLU(),
            nn.Linear(BLOCK_HIDDEN, BLOCK_OUT),
        )
        self.reader_ext = nn.Linear(BLOCK_OUT, 256, bias=False)
        nn.init.zeros_(self.reader_ext.weight)

    # -- reader with the zero-initialised block columns --------------------
    def _readout(self, z_old: torch.Tensor, z_new: torch.Tensor) -> torch.Tensor:
        h = self.reader.lins[0](z_old) + self.reader_ext(z_new)
        h = F.silu(h)
        if self.reader.dropout > 0.0:
            h = F.dropout(h, p=self.reader.dropout, training=self.training)
        h = self.reader.lins[1](h)
        h = F.silu(h)
        if self.reader.dropout > 0.0:
            h = F.dropout(h, p=self.reader.dropout, training=self.training)
        return self.reader.lins[2](h).view(-1)

    # -- forward -----------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, return_stats: bool = False):
        # Run the frozen BOSI-B forward unchanged to obtain c_e and z_old.
        _, st = super().forward(batch, return_stats=True)
        c = st["c"]
        first, second = ZB.block_content(c, batch, self.block_arm)
        u = ZB.block_input(first, second, batch)
        phi = self.block_mlp(self.block_norm(u))
        z_new = ZB.pooled_block_channel(phi, batch, BLOCK_OUT)
        y = self._readout(st["z"], z_new)
        if return_stats:
            out = dict(st)
            out.update({
                "z_old": st["z"],
                "z_new": z_new,
                "block_u": u,
                "block_phi": phi,
                "block_first": first,
                "block_second": second,
                "block_c": c,
                "block_arm": self.block_arm,
            })
            return y, out
        return y


def parameter_breakdown(model: BlockBOSI) -> dict:
    def n(m):
        return int(sum(p.numel() for p in m.parameters()))

    old = {
        "atom_encoder": n(model.atom_encoder),
        "bond_encoder": n(model.bond_encoder),
        "half_encoder": n(model.half_encoder),
        "side": n(model.side),
        "compose": n(model.compose),
        "reader": n(model.reader),
    }
    new = {
        "block_norm": n(model.block_norm),
        "block_mlp": n(model.block_mlp),
        "reader_ext": n(model.reader_ext),
    }
    old_total = sum(old.values())
    new_total = sum(new.values())
    return {
        **old,
        **new,
        "old_total": old_total,
        "new_total": new_total,
        "total": old_total + new_total,
    }


def load_ssl(model: BlockBOSI, path, device="cpu") -> dict:
    """Load only the train-only masked-SSL atom/bond encoders."""
    info = BM.load_ssl(model, path, device)
    info["junction_encoder_loaded"] = False
    return info


def build_model(norm: Mapping[str, object], block_arm: str, seed: int, device,
                *, dropout: float = 0.0) -> tuple[BlockBOSI, dict]:
    """One shared initial state per seed; ``block_arm`` never consumes RNG."""
    import ad_common as C

    torch.manual_seed(int(seed))
    model = BlockBOSI(norm, block_arm=block_arm, dropout=dropout)
    info = {"ssl": load_ssl(model, BC.ssl_path(seed), device)}
    model.to(device)
    info["parameters"] = C.parameter_count(model)
    info["parameters_breakdown"] = parameter_breakdown(model)
    info["initial_state_sha256"] = C.state_hash(model)
    info["block_arm"] = block_arm
    info["seed"] = int(seed)
    info["official_test_loaded"] = False
    if info["parameters"] != BC.EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameters {info['parameters']} != "
                           f"expected {BC.EXPECTED_PARAMETERS}")
    if info["parameters"] > BC.PARAM_CAP:
        raise RuntimeError(f"parameter cap exceeded: {info['parameters']}")
    return model, info


def reference_bosi(norm: Mapping[str, object], seed: int, device,
                   *, dropout: float = 0.0):
    """The original BOSI-B at the same seed, for the init-equivalence gate."""
    import bosi_models as BM2

    torch.manual_seed(int(seed))
    ref = BM2.BOSI(norm, arm="B", dropout=dropout)
    BM2.load_ssl(ref, BC.ssl_path(seed), device)
    ref.to(device)
    return ref
