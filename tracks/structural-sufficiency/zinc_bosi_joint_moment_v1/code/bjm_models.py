"""BOSI-B / BOSI-JM / BOSI-CM.

The B arm is the frozen BOSI v0 ``BOSI(arm="B")`` class, imported unchanged:
same half-edge encoder, same relation construction, same Side / Compose /
Reader modules, same ``forward``.

``BOSI-JM`` keeps every one of those modules and changes exactly one thing --
the 384-dimensional statistics block ``M`` handed to ``Side``:

    A        = sum_f [T1(x_f), T2(x_f), T3(x_f)]                     96
    z_f      = LayerNorm(x_f)                                        no affine
    u_jf     = tanh(U_j z_f),  w_jf = tanh(V_j z_f)              12 each
    J_j      = sum_f outer(u_jf, w_jf)                              12 x 12
    M_JM     = concat(A, vec(J_1), vec(J_2))                        384

``BOSI-CM`` is the strictly matched control.  It shares every parameter,
every initialisation draw, every input and the whole training scheme; the only
difference is that the same partner statistics are first summed and then
paired:

    C_j      = outer(sum_f u_jf, sum_f w_jf) / max(n, 1)             12 x 12

with ``n`` the number of other bonds at the centre atom, and ``C_j = 0``
exactly when ``n = 0``.  The identity

    J_j - C_j = sum_f (u_jf - mean_f u_j)(w_jf - mean_f w_j)^T

is the algebraic content of the contrast: ``JM`` keeps the *within-partner
pairing* of the two projections, ``CM`` destroys it and keeps only the
marginal pair of sums.  For ``n<=1`` the two are provably identical.

Nothing else changes: no trainable junction token, no iterative state update,
no attention, no extra supervision, no global correction branch, same reader
width/depth/parameters.  Both models are 475,425 parameters (+1,536 for the
four 32x12 bias-free projections over ``BOSI-B``'s 473,889).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import bosi_models as BM

N_REL = BM.N_REL
RANK = BM.RANK
WIDTH = BM.WIDTH

JM_RANK = 12
N_JOINT = 2
A_DIM = 3 * RANK                      # 96
BLOCK_DIM = JM_RANK * JM_RANK         # 144
M_DIM = A_DIM + N_JOINT * BLOCK_DIM   # 384
SIDE_IN = BM.SIDE_IN                  # 617 (unchanged)
ARMS = ("B", "JM", "CM")


class BOSIJointMoment(BM.BOSI):
    """BOSI with the partner statistics replaced by an explicit joint moment.

    Construction order is ``BOSI.__init__`` first (the six frozen MLPs, in the
    same order, consuming the same RNG stream as the B arm), then the four
    ``Linear(32, 12, bias=False)`` projections in the fixed order
    ``U_1, V_1, U_2, V_2``.  With one ``torch.manual_seed(seed)`` before
    construction this makes every common BOSI tensor bit-identical to the B
    arm at the same seed, and the whole state bit-identical between JM and CM.

    ``_diag_pairing`` is a diagnostic-only switch used by the CPU gates to
    break the partner pairing while keeping both marginal multisets: it feeds
    ``w_jf`` the ``z`` of a permuted partner inside the same half-edge group.
    It is ``None`` on every training / inference path, and ``G7`` verifies the
    default path reproduces the unpermuted forward bit for bit.
    """

    def __init__(self, norm: Mapping[str, object], *, mode: str = "JM",
                 dropout: float = 0.0):
        if mode not in ("JM", "CM"):
            raise ValueError(f"unknown joint-moment mode {mode!r}")
        super().__init__(norm, arm="B", dropout=dropout)
        self.mode = mode
        self.arm = mode
        self.joint_u1 = nn.Linear(RANK, JM_RANK, bias=False)
        self.joint_v1 = nn.Linear(RANK, JM_RANK, bias=False)
        self.joint_u2 = nn.Linear(RANK, JM_RANK, bias=False)
        self.joint_v2 = nn.Linear(RANK, JM_RANK, bias=False)
        self._diag_pairing = None

    # -- the only changed computation --------------------------------------
    def joint_projections(self, z: torch.Tensor):
        return [
            (torch.tanh(self.joint_u1(z)), torch.tanh(self.joint_v1(z))),
            (torch.tanh(self.joint_u2(z)), torch.tanh(self.joint_v2(z))),
        ]

    def joint_blocks(self, x: torch.Tensor, tgt: torch.Tensor, idx_u: torch.Tensor,
                     idx_w: torch.Tensor) -> dict:
        """``[J_1, J_2]`` (JM) or ``[C_1, C_2]`` (CM) plus the raw factors.

        ``x`` is ``[H, 32]`` (every half-edge of the batch), ``tgt[p]`` is the
        half-edge that partner row ``p`` belongs to.  ``LayerNorm`` has no
        affine parameters; ``eps`` is the PyTorch default ``1e-5``.
        """
        z = F.layer_norm(x, (RANK,))
        proj = self.joint_projections(z)
        H = int(x.shape[0])
        n_tot = x.new_zeros((H, 1))
        n_tot.index_add_(0, tgt, torch.ones((tgt.numel(), 1), dtype=x.dtype,
                                            device=x.device))
        blocks, us, ws = [], [], []
        for u_all, w_all in proj:
            u = u_all[idx_u]
            w = w_all[idx_w]
            us.append(u)
            ws.append(w)
            if self.mode == "JM":
                outer = (u.unsqueeze(2) * w.unsqueeze(1)).reshape(u.shape[0], BLOCK_DIM)
                blk = x.new_zeros((H, BLOCK_DIM))
                blk.index_add_(0, tgt, outer)
            else:
                su = x.new_zeros((H, JM_RANK))
                su.index_add_(0, tgt, u)
                sw = x.new_zeros((H, JM_RANK))
                sw.index_add_(0, tgt, w)
                blk = (su.unsqueeze(2) * sw.unsqueeze(1)).reshape(H, BLOCK_DIM)
                blk = blk / n_tot.clamp(min=1.0)
            blocks.append(blk)
        return {"blocks": blocks, "u": us, "w": ws, "n_tot": n_tot, "z": z}

    # -- relation statistics: only the M block changes ----------------------
    def relation_stats(self, batch: Mapping[str, torch.Tensor], x: torch.Tensor):
        vals = BM.chebyshev(x)                       # [H, 3*RANK]
        H = int(x.shape[0])
        tgt = batch["partner_tgt"]
        lag = batch["partner_half"]
        A = x.new_zeros((H, vals.shape[1]))
        A.index_add_(0, tgt, vals[lag])

        tgt_rel = tgt * N_REL + batch["partner_rel"]
        n = x.new_zeros((H * N_REL,))
        n.index_add_(0, tgt_rel, torch.ones_like(tgt_rel, dtype=x.dtype))
        n = n.view(H, N_REL)

        idx_w = lag if self._diag_pairing is None else self._diag_pairing
        jb = self.joint_blocks(x, tgt, lag, idx_w)
        M = torch.cat([A] + jb["blocks"], dim=1)
        return M, n


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


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
    for name in ("joint_u1", "joint_v1", "joint_u2", "joint_v2"):
        if hasattr(model, name):
            out[name] = n(getattr(model, name))
    out["joint_projections"] = sum(out.get(k, 0)
                                   for k in ("joint_u1", "joint_v1",
                                             "joint_u2", "joint_v2"))
    out["total"] = n(model)
    return out


def build_model(norm: Mapping[str, object], arm: str, seed: int, device,
                *, dropout: float = 0.0):
    """One shared initial state per seed; all arms draw from the same stream.

    ``torch.manual_seed(seed)`` -> construct (6 frozen MLPs, then the 4 joint
    projections for JM/CM) -> load the train-only SSL atom/bond encoders ->
    move to device.  The arm never consumes RNG of its own.
    """
    import bjm_common as BC

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    torch.manual_seed(int(seed))
    if arm == "B":
        model = BM.BOSI(norm, arm="B", dropout=dropout)
    else:
        model = BOSIJointMoment(norm, mode=arm, dropout=dropout)
    info = {"ssl": BM.load_ssl(model, BC.ssl_path(seed), device)}
    model.to(device)
    info["parameters"] = int(sum(p.numel() for p in model.parameters()))
    info["parameters_breakdown"] = parameter_breakdown(model)
    info["arm"] = arm
    info["seed"] = int(seed)
    info["expected_parameters"] = BC.EXPECTED_PARAMETERS[arm]
    if info["parameters"] != BC.EXPECTED_PARAMETERS[arm]:
        raise RuntimeError(f"{arm}: parameters {info['parameters']} != "
                           f"{BC.EXPECTED_PARAMETERS[arm]}")
    if info["parameters"] > BC.PARAM_CAP:
        raise RuntimeError(f"{arm}: parameter cap exceeded")
    return model, info


def common_state(state: Mapping[str, torch.Tensor]) -> dict:
    """The BOSI-B parameter set only (drops the JM/CM joint projections)."""
    drop = {"joint_u1.weight", "joint_v1.weight", "joint_u2.weight", "joint_v2.weight"}
    return {k: v for k, v in state.items() if k not in drop}
