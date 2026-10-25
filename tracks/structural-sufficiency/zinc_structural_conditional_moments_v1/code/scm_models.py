"""Structural conditional moments (SCM) written into the DCC-L local channel.

The frozen DCC-L forward is reused verbatim.  The only insertion point is a
forward hook on ``DualChannel.local_composer``: the hook receives the real
``c_loc`` that the composer produced, applies the SCM correction, and returns
``c_loc + delta``.  ``DualChannel.forward`` then forms ``sum(c_loc')`` and
``sum(c_loc'^2)`` from that returned tensor, so the correction enters the
*existing* reader columns rather than a new readout branch.

Math (per molecule, physical bonds e/f, line-graph distance d_L):

    mu_t[e,f] = 1[d_L(e,f) in band_t]                     (frozen, self excluded)
    n_t[e]    = sum_f mu_t[e,f]
    p_t[e,f]  = mu_t[e,f] / n_t[e]        (only defined for n_t > 0)
    p_G[f]    = 1 / N_bonds

    v_e       = V(h_e),  q_e = Q(h_e),  k_e = K(h_e),  h = c_loc,  rank 8
    score[e,f]= tanh(q_e . k_f / sqrt(rank))

    S[e,t]    = sum_f mu_t[e,f] v_f  -  n_t[e] mean_G(v)
    cov_t[e]  = E_{p_t}(score*v) - E_{p_t}(score) E_{p_t}(v)
    cov_G[e]  = E_{p_G}(score*v) - E_{p_G}(score) E_{p_G}(v)
    A[e,t]    = n_t[e] (cov_t[e] - cov_G[e])

    delta_e   = Wout([S[e,1], A[e,1], S[e,2], A[e,2]])
    c_loc'    = c_loc + delta

``S`` and ``A`` are divided by one *frozen label-free* constant per band (the
mean band size over the Fit10k raw graphs).  There is no per-query division by
``n_t``, so the relative band count survives.  Both arms use the same
constants.

S-only arm: the A blocks are exactly zero in the training forward and ``Q``/
``K`` are not evaluated.  ``Q``/``K`` remain registered parameters (same
structure, same init as the S+A arm) but receive no gradient there.
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

import scm_common as S

# The frozen historical modules (imported, never modified).
import ad_common as C  # noqa: E402
import ad2_models as A2  # noqa: E402
import dcc_models as MD  # noqa: E402

WIDTH = MD.WIDTH  # 96
LOCAL_DIM = MD.WIDTH


class SCMModule(nn.Module):
    """Q/K/V + per-band structural difference / conditional covariance."""

    def __init__(self, *, width: int = WIDTH, rank: int = S.RANK,
                 scales: tuple[float, float], dropout: float = 0.0):
        super().__init__()
        self.width = int(width)
        self.rank = int(rank)
        self.q = nn.Linear(width, rank)
        self.k = nn.Linear(width, rank)
        self.v = nn.Linear(width, rank)
        # 4 blocks x rank -> width, zero-initialised so the initial function is
        # exactly DCC-L.
        self.out = nn.Linear(4 * rank, width)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()
        self.register_buffer("scale", torch.as_tensor(list(scales), dtype=torch.float32))
        self.dropout = float(dropout)

    # -- core math, exposed for the CPU identity checks -----------------------
    def moments(self, c_loc: torch.Tensor, batch: Mapping[str, torch.Tensor], *,
                with_cov: bool = True):
        """Return (S1, A1, S2, A2) in the padded (B, bmax, rank) layout."""
        B = int(batch["n_graphs"])
        bmax = int(batch["bond_bmax"])
        dev, dtype = c_loc.device, c_loc.dtype
        bb = batch["bond_batch"]
        bl = batch["bond_local"]

        v = self.v(c_loc)
        vpad = v.new_zeros((B, bmax, self.rank)).index_put_((bb, bl), v)
        mask = batch["bond_mask"].to(dtype)                       # (B, bmax)
        cnt = mask.sum(-1).clamp(min=1.0)                         # (B,)
        mean_g = vpad.sum(1) / cnt.unsqueeze(1)                   # (B, rank)

        mu1 = batch["mu1"].to(dtype)                              # (B, bmax, bmax)
        mu2 = batch["mu2"].to(dtype)
        n1 = mu1.sum(-1)                                          # (B, bmax)
        n2 = mu2.sum(-1)
        n1c = n1.clamp(min=1.0)
        n2c = n2.clamp(min=1.0)
        sum_v1 = torch.bmm(mu1, vpad)
        sum_v2 = torch.bmm(mu2, vpad)
        s1 = (sum_v1 - n1.unsqueeze(-1) * mean_g.unsqueeze(1)) / self.scale[0]
        s2 = (sum_v2 - n2.unsqueeze(-1) * mean_g.unsqueeze(1)) / self.scale[1]
        if not with_cov:
            z = c_loc.new_zeros((B, bmax, self.rank))
            return s1, z, s2, z

        q = self.q(c_loc)
        k = self.k(c_loc)
        qpad = q.new_zeros((B, bmax, self.rank)).index_put_((bb, bl), q)
        kpad = k.new_zeros((B, bmax, self.rank)).index_put_((bb, bl), k)
        score = torch.tanh(torch.bmm(qpad, kpad.transpose(1, 2)) / math.sqrt(self.rank))

        # band terms: E_{p_t}(score*v) needs mu*score then bmm with v
        ms1 = mu1 * score
        ms2 = mu2 * score
        sv1 = torch.bmm(ms1, vpad)
        sv2 = torch.bmm(ms2, vpad)
        sc1 = ms1.sum(-1)
        sc2 = ms2.sum(-1)
        band_v1 = sum_v1 / n1c.unsqueeze(-1)
        band_v2 = sum_v2 / n2c.unsqueeze(-1)
        cov1 = sv1 / n1c.unsqueeze(-1) - (sc1 / n1c).unsqueeze(-1) * band_v1
        cov2 = sv2 / n2c.unsqueeze(-1) - (sc2 / n2c).unsqueeze(-1) * band_v2

        sv_g = torch.bmm(score, vpad) / cnt.view(B, 1, 1)
        sc_g = score.sum(-1) / cnt.view(B, 1)
        cov_g = sv_g - sc_g.unsqueeze(-1) * mean_g.unsqueeze(1)

        a1 = n1.unsqueeze(-1) * (cov1 - cov_g) / self.scale[0]
        a2 = n2.unsqueeze(-1) * (cov2 - cov_g) / self.scale[1]
        return s1, a1, s2, a2

    def forward(self, c_loc: torch.Tensor, batch: Mapping[str, torch.Tensor], *,
                with_cov: bool = True, ablate: str | None = None) -> torch.Tensor:
        s1, a1, s2, a2 = self.moments(c_loc, batch, with_cov=with_cov)
        if ablate is not None:
            z = torch.zeros_like
            if ablate == "S":
                a1, a2 = z(a1), z(a2)
            elif ablate == "A":
                s1, s2 = z(s1), z(s2)
            elif ablate == "S1":
                s1 = z(s1)
            elif ablate == "S2":
                s2 = z(s2)
            elif ablate == "A1":
                a1 = z(a1)
            elif ablate == "A2":
                a2 = z(a2)
            elif ablate != "none":
                raise ValueError(f"unknown ablate {ablate!r}")
        cat = torch.cat([s1, a1, s2, a2], dim=-1)
        if self.dropout > 0.0:
            cat = torch.nn.functional.dropout(cat, p=self.dropout, training=self.training)
        dpad = self.out(cat)                                      # (B, bmax, width)
        return dpad[batch["bond_batch"], batch["bond_local"]]     # (E, width)


class SCMChannel(nn.Module):
    """The frozen DCC-L model plus one SCM correction inside its local channel."""

    def __init__(self, dcc: MD.DualChannel, scm: SCMModule, *, use_cov: bool):
        super().__init__()
        if not bool(dcc.organized_local):
            raise ValueError("SCM is defined on the organized DCC-L (L) arm only")
        self.dcc = dcc
        self.scm = scm
        self.use_cov = bool(use_cov)
        self.arm = "S+A" if self.use_cov else "S-only"

    # -- forward -----------------------------------------------------------
    def _run(self, batch, *, capture=None, with_cov=None, ablate=None):
        use_cov = self.use_cov if with_cov is None else bool(with_cov)

        def hook(_module, _inputs, output):
            delta = self.scm(output, batch, with_cov=use_cov, ablate=ablate)
            if capture is not None:
                capture["c_loc"] = output.detach()
                capture["delta"] = delta.detach()
                capture["c_loc_prime"] = (output + delta).detach()
                s1, a1, s2, a2 = self.scm.moments(output, batch, with_cov=use_cov)
                capture["S1"], capture["A1"] = s1.detach(), a1.detach()
                capture["S2"], capture["A2"] = s2.detach(), a2.detach()
            return output + delta

        handle = self.dcc.local_composer.register_forward_hook(hook)
        try:
            pred = self.dcc(batch)
        finally:
            handle.remove()
        return pred

    def forward(self, batch):
        return self._run(batch)

    @torch.no_grad()
    def forward_with_capture(self, batch, *, with_cov=None, ablate=None):
        cap: dict = {}
        pred = self._run(batch, capture=cap, with_cov=with_cov, ablate=ablate)
        return pred, cap


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _state_sha(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_initial_state(norm: Mapping[str, object], seed: int, *, use_cov: bool):
    """Historical DCC-L seed state, then the SCM module from an isolated RNG.

    The backbone construction is *byte-for-byte* the historical sequence in
    ``dcc_models.build_initial_state``: ``manual_seed(seed)`` -> P3NullModel ->
    masked-SSL encoder load -> ``manual_seed(seed + 90731)`` -> DualChannel ->
    ``load_p3_state``.  The SCM module is created inside a forked RNG stream
    with its own seed, so the backbone tensors and the post-build ambient RNG
    state are identical to the historical builder.
    """
    scales = S.band_scales()
    torch.manual_seed(int(seed))
    p3 = A2.P3NullModel(norm)
    ssl_path = C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
    ssl_info = A2._load_encoders(p3, ssl_path, torch.device("cpu"))
    p3_state = C.cpu_state(p3)

    torch.manual_seed(int(seed) + 90731)
    dcc = MD.DualChannel(norm)
    dcc.load_p3_state(p3_state)
    backbone_sha = _state_sha(dcc)
    if int(seed) == 0 and backbone_sha != "f5c34c285c98b4d8a88361cb7dc1043b5abf07799f4f6121b25a59c904166e33":
        raise RuntimeError(f"backbone init drift: {backbone_sha}")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + S.SCM_SEED_OFFSET)
        scm = SCMModule(scales=scales)
    model = SCMChannel(dcc, scm, use_cov=use_cov)
    info = {
        "protocol_id": S.TRACK_ID,
        "seed": int(seed),
        "arm": model.arm,
        "use_cov": bool(use_cov),
        "ssl": ssl_info,
        "p3_reference_state_sha256": C.state_hash(p3),
        "backbone_dcc_l_state_sha256": backbone_sha,
        "scm_seed": int(seed) + S.SCM_SEED_OFFSET,
        "scm_state_sha256": _state_sha(scm),
        "initial_state_sha256": _state_sha(model),
        "bands": [list(b) for b in S.BANDS],
        "rank": S.RANK,
        "scales": list(scales),
        "parameters": _param_count(model),
        "parameters_breakdown": {
            "dcc_l_backbone": _param_count(dcc),
            "scm_qkv": sum(_param_count(getattr(scm, n)) for n in ("q", "k", "v")),
            "scm_out": _param_count(scm.out),
            "scm_total": _param_count(scm),
            "deployment_total": _param_count(model),
            "headroom_vs_500k": 500_000 - _param_count(model),
        },
    }
    return model, info


def build_model(candidate: str, norm: Mapping[str, object], seed: int, device, *,
                use_cov: bool):
    """Builder compatible with ``ad_train.train``; deterministic per (seed, arm)."""
    model, info = build_initial_state(norm, seed, use_cov=use_cov)
    model.to(device)
    info.update(
        {
            "candidate": candidate,
            "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
            "parameters_total": sum(1 for p in model.parameters()),
        }
    )
    return model, info


def make_builder(use_cov: bool):
    def builder(candidate, norm, seed, device):
        return build_model(candidate, norm, seed, device, use_cov=use_cov)

    return builder


def shared_pair_builders(norm: Mapping[str, object], seed: int):
    """One shared initial state -> two arms (identical all parameter tensors)."""
    shared, base_info = build_initial_state(norm, seed, use_cov=True)
    out = {}
    for use_cov, arm in ((False, "S-only"), (True, "S+A")):
        model = copy.deepcopy(shared)
        model.use_cov = bool(use_cov)
        model.arm = arm
        info = dict(base_info)
        info.update({"candidate": f"SCM_{arm}", "use_cov": bool(use_cov), "arm": arm,
                     "shared_initial_state": True})
        out[arm] = (model, info)
    return out, base_info
