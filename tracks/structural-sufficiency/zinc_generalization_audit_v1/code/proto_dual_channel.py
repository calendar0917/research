"""CPU size/identity prototype for the proposed dual-channel composer.

No training.  It checks (i) parameter count, (ii) exact P3 output at
initialisation when the new reader columns are zero, and (iii) that the same
skeleton can implement the matched Null-local control.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "zinc_recipe_trunk_v2" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "zinc_arch_discovery_v1" / "code"))
import audit_common as C  # noqa: E402
import ad2_models as A2  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ad_train as T  # noqa: E402

ANALYSIS = C.ANALYSIS


class DualChannel(A2.P3NullModel):
    """P3 Null + parallel local-ownership channel with separate reader moments."""

    def __init__(self, norm, *, local_hidden: int = 80, dropout: float = 0.0):
        super().__init__(norm, dropout=dropout)
        self.local_composer = M1.MLP(96 + 2 * 96 + 1 + D.BOND_FEAT_DIM, local_hidden, 96, 1, dropout)
        self.dual_reader = M1.MLP(96 + 2 * 96 + D.GLOBAL_DIM + 2 * 96, 256, 1, n_hidden=2, dropout=dropout)
        # Copy the P3 reader into the wider reader; the two new 96-d blocks
        # start at exactly zero.  This makes the initial forward bit-identical.
        old = self.reader.lins
        new = self.dual_reader.lins
        with torch.no_grad():
            new[0].weight.zero_()
            new[0].weight[:, : old[0].weight.shape[1]].copy_(old[0].weight)
            new[0].bias.copy_(old[0].bias)
            for i in range(1, len(old)):
                new[i].weight.copy_(old[i].weight)
                new[i].bias.copy_(old[i].bias)
        self.reader = self.dual_reader


    def load_p3(self, p3: A2.P3NullModel) -> None:
        """Load all P3 weights and widen the reader with zero new columns."""
        sd = {k: v for k, v in p3.state_dict().items() if not k.startswith("reader.")}
        self.load_state_dict(sd, strict=False)
        old = p3.reader.lins
        new = self.reader.lins
        with torch.no_grad():
            new[0].weight.zero_()
            new[0].weight[:, : old[0].weight.shape[1]].copy_(old[0].weight)
            new[0].bias.copy_(old[0].bias)
            for i in range(1, len(old)):
                new[i].weight.copy_(old[i].weight)
                new[i].bias.copy_(old[i].bias)

    def local_grouping(self, w, batch, *, organized: bool):
        if not organized:
            W, V, n_b = self.group_to_bonds(w, batch)
            return W, V, n_b
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        W = torch.zeros((int(batch["bond_feat"].shape[0]), w.shape[1]), device=w.device, dtype=w.dtype)
        V = torch.zeros_like(W)
        W.index_add_(0, jb[:, 0], w)
        W.index_add_(0, jb[:, 1], w)
        w2 = w * w
        V.index_add_(0, jb[:, 0], w2)
        V.index_add_(0, jb[:, 1], w2)
        return W, V, n_b

    def forward(self, batch, *, organized_local: bool = True):
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        c_glob = self.group_composer(torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], 1))
        Wl, Vl, _ = self.local_grouping(w, batch, organized=organized_local)
        c_loc = self.local_composer(torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], 1))
        B = int(batch["n_graphs"])
        z = c_glob.new_zeros((B, c_glob.shape[1])).index_add_(0, batch["bond_batch"], c_glob)
        zl = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc)
        zl2 = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc * c_loc)
        a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        return self.reader(torch.cat([z, a_sum, a_sq, g, zl, zl2], 1)).view(-1)


def main() -> int:
    cache = C.load_cache("valid")
    rows = np.arange(64, dtype=np.int64)
    batch = T.collate(cache, rows, "P3N", valid=True)
    norm = D.load_norm()
    p3 = A2.P3NullModel(norm).eval()
    dual = DualChannel(norm).eval()
    dual.load_p3(p3)
    with torch.no_grad():
        y0 = p3(batch)
        y1 = dual(batch, organized_local=True)
        y2 = dual(batch, organized_local=False)
    out = {
        "p3_params": sum(p.numel() for p in p3.parameters()),
        "dual_params": sum(p.numel() for p in dual.parameters()),
        "dual_headroom_under_500k": 500000 - sum(p.numel() for p in dual.parameters()),
        "local_composer_params": sum(p.numel() for p in dual.local_composer.parameters()),
        "reader_param_delta": sum(p.numel() for p in dual.reader.parameters()) - sum(p.numel() for p in p3.reader.parameters()),
        "max_abs_diff_initial_dual_vs_p3": float((y1 - y0).abs().max()),
        "max_abs_diff_initial_control_vs_p3": float((y2 - y0).abs().max()),
        "all_finite": bool(torch.isfinite(y1).all() and torch.isfinite(y2).all()),
        "note": "CPU prototype only; initial exactness follows from zero new reader columns. Training is not run here.",
    }
    C.write_json(ANALYSIS / "dual_channel_prototype.json", out)
    print(json.dumps(out, indent=2))
    return 0 if out["max_abs_diff_initial_dual_vs_p3"] < 1e-6 and out["max_abs_diff_initial_control_vs_p3"] < 1e-6 and out["dual_params"] <= 500000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
