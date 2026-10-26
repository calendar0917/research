"""BOSI-QP: target-conditioned pair composition (A1) and its matched null (A0).

Both arms are the frozen ``bosi_models.BOSI(arm="B")`` body — same atom / bond /
half-edge encoders, same 384-d partner-moment block ``M``, same relation counts,
same Side / Compose / Reader — plus **one** new static computation:

    A1  q_e   = x[v, e]                                     the target half-edge
    A0  q_e   = mean_{f != e} x[v, f]                       the partner-set mean
        t_ef  = [q_e, p_f, |q_e - p_f|, q_e * p_f]           p_f = x[v, f]
        w_ef  = Phi(t_ef)                                    Phi = 128 -> 64 -> 32
        P[v,e]= sum_{f != e} w_ef                            (empty partner set -> 0)
        s[v,e]= Side([a_v, b_e, x[v,e], M[v,e], n[v,e], closure,
                      log1p(deg), P[v,e]])                  617 -> 649 input

``Phi`` is shared across every pair, is applied once (no recursion, no
attention), and ``P`` is appended to the existing Side input rather than
replacing the proven partner-moment block.

The two arms share every parameter shape, every initialisation draw (the common
BOSI tensors are bit-identical to BOSI-B at the same seed) and the whole recipe.
They differ only in whether the *target* half-edge state enters each pair.

The 32 new Side input columns are zero-initialised, so at step 0 the forward is
exactly BOSI-B: ``P`` is arbitrary but multiplied by zero weights.  ``Phi``
therefore receives no gradient at step 0 and a non-zero one from step 1 on —
the CPU gates measure exactly that.

``mode="B"`` reproduces the frozen BOSI-B body with no extra parameters and is
used only by the parity gate.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

import bosi_models as BM

N_REL = BM.N_REL
RANK = BM.RANK
WIDTH = BM.WIDTH
PAIR_HIDDEN = 64
PAIR_DIM = 32
PAIR_IN = 4 * RANK                      # 128
SIDE_IN_BASE = BM.SIDE_IN               # 617
SIDE_IN = SIDE_IN_BASE + PAIR_DIM       # 649
ARMS = ("B", "A1", "A0")


class BOSIQueryPair(BM.BOSI):
    """BOSI-B plus one shared, target-conditioned pair encoder.

    Construction order: ``BOSI.__init__`` first (the six frozen MLPs, consuming
    exactly the BOSI-B RNG stream), then ``pair_phi``, then the widened
    ``side.lins[0]``.  The arm never consumes RNG of its own.
    """

    def __init__(self, norm: Mapping[str, object], *, mode: str = "A1",
                 dropout: float = 0.0):
        if mode not in ARMS:
            raise ValueError(f"unknown mode {mode!r}")
        super().__init__(norm, arm="B", dropout=dropout)
        # ``self.arm`` stays "B" so the inherited ``relation_stats`` keeps the
        # frozen BOSI-B moment block; ``self.mode`` selects the new branch.
        self.mode = mode
        self.pair_phi = BM.MLP(PAIR_IN, PAIR_HIDDEN, PAIR_DIM, 1, dropout)
        if mode in ("A1", "A0"):
            self._widen_side()
        # diagnostic-only switch used by the gates (never set during training):
        # replaces the target state by the partner-set mean in A1 as well.
        self._null_override = False

    def _widen_side(self) -> None:
        """Replace ``side.lins[0]`` by a 649-input copy with zero new columns."""
        old = self.side.lins[0]
        new = nn.Linear(SIDE_IN, old.out_features, bias=True,
                        device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :SIDE_IN_BASE].copy_(old.weight)
            new.bias.copy_(old.bias)
        self.side.lins[0] = new

    # -- the only added computation ----------------------------------------
    def pair_stats(self, batch: Mapping[str, torch.Tensor], x: torch.Tensor,
                   *, null: bool):
        """``P[v,e] = sum_{f != e} Phi([q_e, p_f, |q_e-p_f|, q_e*p_f])``."""
        tgt = batch["partner_tgt"]
        part = batch["partner_half"]
        p = x[part]
        H = int(x.shape[0])
        if null:
            n_tot = x.new_zeros((H, 1))
            n_tot.index_add_(0, tgt, torch.ones((tgt.numel(), 1), dtype=x.dtype,
                                                device=x.device))
            q_tot = x.new_zeros((H, RANK))
            q_tot.index_add_(0, tgt, p)
            q = (q_tot / n_tot.clamp(min=1.0))[tgt]
        else:
            q = x[tgt]
        feat = torch.cat([q, p, (q - p).abs(), q * p], dim=1)
        w = self.pair_phi(feat)
        P = x.new_zeros((H, PAIR_DIM))
        P.index_add_(0, tgt, w)
        return P

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, return_stats: bool = False):
        a, b = self.encode_tokens(batch)
        x = self.half_edges(batch, a, b)
        M, n = self.relation_stats(batch, x)
        half_deg = batch["half_deg"].view(-1, 1)
        parts = [a[batch["half_atom"]], b[batch["half_bond"]], x, M, n,
                 batch["closure"], torch.log1p(half_deg)]
        P = None
        if self.mode in ("A1", "A0"):
            null = (self.mode == "A0") or bool(self._null_override)
            P = self.pair_stats(batch, x, null=null)
            parts.append(P)
        s = self.side(torch.cat(parts, dim=1))
        hb = batch["half_of_bond"]
        su, sv = s[hb[:, 0]], s[hb[:, 1]]
        c = self.compose(torch.cat(
            [b, su + sv, (su - sv).abs(), su * sv, batch["bond_feat"]], dim=1))
        Bn = int(batch["n_graphs"])
        bb, ab = batch["bond_batch"], batch["atom_batch"]
        zc = c.new_zeros((Bn, WIDTH)).index_add_(0, bb, c)
        zc2 = c.new_zeros((Bn, WIDTH)).index_add_(0, bb, c * c)
        za = c.new_zeros((Bn, WIDTH)).index_add_(0, ab, a)
        za2 = c.new_zeros((Bn, WIDTH)).index_add_(0, ab, a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        z = torch.cat([zc, zc2, za, za2, g], dim=1)
        y = self.reader(z).view(-1)
        if return_stats:
            return y, {"M": M, "n": n, "P": P, "z": z, "c": c, "s": s, "x": x,
                       "a": a, "b": b}
        return y


def parameter_breakdown(model: nn.Module) -> dict:
    def n(m):
        return int(sum(p.numel() for p in m.parameters()))

    out = {
        "atom_encoder": n(model.atom_encoder),
        "bond_encoder": n(model.bond_encoder),
        "half_encoder": n(model.half_encoder),
        "side": n(model.side),
        "compose": n(model.compose),
        "reader": n(model.reader),
    }
    if hasattr(model, "pair_phi"):
        out["pair_phi"] = n(model.pair_phi)
    out["total"] = n(model)
    return out


def common_state(state: Mapping[str, torch.Tensor]) -> dict:
    """The BOSI-B parameter set only (drops the pair encoder and the widened
    Side input matrix, whose shape differs from BOSI-B by construction)."""
    drop = {"side.lins.0.weight"}
    return {k: v for k, v in state.items()
            if not k.startswith("pair_phi.") and k not in drop}


def build_model(norm: Mapping[str, object], arm: str, seed: int, device,
                *, dropout: float = 0.0):
    import aas_common as AC

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    torch.manual_seed(int(seed))
    if arm == "B":
        model = BM.BOSI(norm, arm="B", dropout=dropout)
    else:
        model = BOSIQueryPair(norm, mode=arm, dropout=dropout)
    info = {"ssl": BM.load_ssl(model, AC.BC.ssl_path(seed), device)}
    model.to(device)
    info["parameters"] = int(sum(p.numel() for p in model.parameters()))
    info["parameters_breakdown"] = parameter_breakdown(model)
    info["arm"] = arm
    info["seed"] = int(seed)
    if arm in AC.EXPECTED_PARAMETERS:
        info["expected_parameters"] = AC.EXPECTED_PARAMETERS[arm]
        if info["parameters"] != AC.EXPECTED_PARAMETERS[arm]:
            raise RuntimeError(f"{arm}: parameters {info['parameters']} != "
                               f"{AC.EXPECTED_PARAMETERS[arm]}")
    if info["parameters"] > AC.PARAM_CAP:
        raise RuntimeError(f"parameter cap exceeded: {info['parameters']}")
    return model, info
