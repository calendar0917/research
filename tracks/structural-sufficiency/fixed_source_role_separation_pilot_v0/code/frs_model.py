"""FSR v0 model: FIXED1 plus one independent low-rank branch per block.

The inherited FIXED1 computation is untouched::

    h0 = base.drop(base.local(batch, a, b, scal))

    z1 = LN1(h0)
    m1 = sum_j SiLU(z1[j] + e_ji)
    h1 = h0 + block1(z1, m1)

    z_target = LN2(h1)
    z_source = LN2(h0)
    m2 = sum_j SiLU(z_source[j] + e_ji)
    h2 = h1 + block2(z_target, m2)

    logits = base.post_local(h2, batch)

with ``block_l(z, m) = fc2(SiLU(fc1(z + m)))`` replaced by

    a = fc1(z + m) + A(B(x)),      x = z + m   (TIED)  or  z - m   (RSSC)
    out = fc2(SiLU(a))

``B = Linear(128, 32, bias=False)`` uses the inherited fan-in uniform init
(``U(-1/sqrt(128), 1/sqrt(128))``) from a private generator; ``A`` starts at
exactly zero.  Both blocks use their own ``(A, B)`` pair, rank is fixed at 32
and there is no activation, gating, attention, normalisation or extra head.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import ds_common as C
import ds_model as DSM
import frs_common as FC

LOWRANK_INIT_SEED = 20261126


class LowRankBranch(nn.Module):
    """``A(B(x))`` with ``B: 128 -> 32`` and ``A: 32 -> 256``, no biases."""

    def __init__(self, width: int = FC.WIDTH, rank: int = FC.RANK, hidden: int = FC.BLOCK_HIDDEN):
        super().__init__()
        self.B = nn.Linear(int(width), int(rank), bias=False)
        self.A = nn.Linear(int(rank), int(hidden), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.A(self.B(x))

    def effective_weight(self) -> torch.Tensor:
        """``D = A.weight @ B.weight`` -- same shape as ``fc1.weight``."""
        return self.A.weight @ self.B.weight


class FRSModel(DSM.DSModel):
    """FIXED1 with a role-separated (``z-m``) or tied (``z+m``) low-rank branch."""

    def __init__(
        self,
        spec: Any,
        norm: Mapping[str, Any],
        n_out: int,
        cfg: Mapping[str, Any],
        *,
        arm: str,
        rank: int = FC.RANK,
        n_layers: int = 2,
        block_hidden: int = FC.BLOCK_HIDDEN,
    ):
        if str(arm) not in FC.ARMS:
            raise KeyError(f"{arm} is not one of {FC.ARMS}")
        super().__init__(spec, norm, int(n_out), dict(cfg), arm="FIXED1", n_layers=int(n_layers),
                         block_hidden=int(block_hidden))
        # ``self.arm`` stays "FIXED1": the inherited arithmetic family is FIXED1.
        self.frs_arm = str(arm)
        self.rank = int(rank)
        width = int(self.cfg["width"])
        self.lowrank = nn.ModuleList(
            [LowRankBranch(width, int(rank), int(block_hidden)) for _ in range(int(n_layers))]
        )

    # -- the branch input ---------------------------------------------------

    def branch_input(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return z + m if self.frs_arm == "TIED" else z - m

    def block_out(self, index: int, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        blk = self.blocks[int(index)]
        lr = self.lowrank[int(index)]
        a = blk.fc1(z + m) + lr(self.branch_input(z, m))
        return blk.fc2(F.silu(a))

    # -- the FIXED1 stages with the new pre-nonlinearity term ---------------

    def stage1(
        self,
        h0: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_msg: torch.Tensor,
        *,
        arm: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blk = self.blocks[0]
        z1 = blk.norm(h0)
        m1 = DSM._agg_source(z1, src, dst, edge_msg)
        return h0 + self.block_out(0, z1, m1), m1

    def stage2(
        self,
        h1: torch.Tensor,
        h0: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_msg: torch.Tensor,
        *,
        arm: str | None = None,
        gamma: float | None = None,
        c_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # no gamma / no branch context: the FIXED1 read is unchanged
        _ = (gamma, c_override)
        blk = self.blocks[1]
        z_target = blk.norm(h1)
        z_source = blk.norm(h0)
        m2 = DSM._agg_source(z_source, src, dst, edge_msg)
        info: dict[str, Any] = {
            "arm": self.frs_arm,
            "gamma": float(C.GAMMA),
            "layer2_reads": "LN2(h0) of the real 1-hop neighbours + LN2(h1)[i] (FIXED1)",
            "branch_input": "z + m" if self.frs_arm == "TIED" else "z - m",
            "m2": m2,
        }
        return h1 + self.block_out(1, z_target, m2), m2, info

    # -- diagnostics ---------------------------------------------------------

    def effective_weights(self) -> dict[str, torch.Tensor]:
        return {f"blocks.{i}": self.lowrank[i].effective_weight() for i in range(len(self.lowrank))}

    def lowrank_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if FC.is_lowrank(n)]


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def init_lowrank(model: FRSModel, *, seed: int = LOWRANK_INIT_SEED) -> dict[str, Any]:
    """``B`` fan-in uniform, ``A`` exactly zero, private generator, no global RNG."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    before = torch.get_rng_state().clone()
    n_b = 0
    with torch.no_grad():
        for branch in model.lowrank:
            fan_in = int(branch.B.in_features)
            bound = 1.0 / math.sqrt(float(fan_in))
            branch.B.weight.uniform_(-bound, bound, generator=gen)
            branch.A.weight.zero_()
            n_b += 1
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("low-rank initialisation advanced the global torch RNG")
    return {
        "generator_seed": int(seed),
        "n_B_initialised": int(n_b),
        "B_bound": 1.0 / math.sqrt(float(FC.WIDTH)),
        "A_initialised_zero": True,
        "global_rng_unchanged": True,
    }


def build_frs_model(
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    arm: str,
    seed: int = 0,
    device: torch.device | str = "cpu",
    rank: int = FC.RANK,
    verify: bool = True,
    strict_w2_fingerprint: bool | None = None,
) -> tuple[FRSModel, dict[str, Any]]:
    """Build one arm from the official W2 constructor state of ``seed``."""
    if str(arm) not in FC.ARMS:
        raise KeyError(str(arm))
    cfg = dict(C.BC.config_for_scale(2))
    reference, ref_audit = DSM.BM.build_model(
        "W2", spec, norm, int(n_out), seed=int(seed), device="cpu", verify_parameters=True
    )
    rng_after_w2 = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = FRSModel(spec, norm, int(n_out), cfg, arm=str(arm), rank=int(rank))
    if not torch.equal(rng_after_w2, torch.get_rng_state()):
        raise RuntimeError("FRSModel construction advanced the global torch RNG")
    load = model.load_state_dict(reference.state_dict(), strict=False)
    expected_missing = {
        k for k in model.state_dict() if FC.is_inherited_new(k) or FC.is_lowrank(k)
    }
    if set(load.missing_keys) != expected_missing or load.unexpected_keys:
        raise RuntimeError(f"W2 state load mismatch: {sorted(load.missing_keys)}")
    base_before = C.state_fingerprint(FC.frs_trunk_subset(model.state_dict()))
    init_audit = C.initialize_dynamic_modules(model, seed=int(seed), zero_residual=True)
    base_after = C.state_fingerprint(FC.frs_trunk_subset(model.state_dict()))
    if base_before != base_after:
        raise RuntimeError("inherited new-module initialisation changed the W2 trunk")
    lowrank_audit = init_lowrank(model)
    base_after_lr = C.state_fingerprint(FC.frs_trunk_subset(model.state_dict()))
    if base_before != base_after_lr:
        raise RuntimeError("low-rank initialisation changed the W2 trunk")
    model.to(torch.device(device))
    audit = FC.parameter_audit(model)
    if verify and not audit["all_match"]:
        raise RuntimeError(f"parameter audit failed for {arm}: {audit['checks']}")

    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ref_state = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
    ours_base = {k: v for k, v in state_cpu.items() if not FC.is_frs_new(k)}
    base_equal = C.compare_states(ours_base, ref_state)
    if verify and not base_equal["all_equal"]:
        raise RuntimeError(f"W2 part is not bitwise identical to the W2 constructor: {base_equal}")
    fp = FC.initial_fingerprints(model)
    matches_hist = {
        "base_state_sha256": fp["base_state_sha256"] == FC.HIST_FIXED1_INIT["base_state_sha256"],
        "new_state_sha256": fp["new_state_sha256"] == FC.HIST_FIXED1_INIT["new_state_sha256"],
        "n_state_tensors_inherited": fp["n_state_tensors_inherited"] == FC.HIST_FIXED1_INIT["n_state_tensors"],
    }
    if verify and not all(matches_hist.values()):
        raise RuntimeError(f"inherited FIXED1 init fingerprints differ: {matches_hist}")
    out = {
        "round_id": FC.ROUND_ID,
        "arm": str(arm),
        "base_computation": "FIXED1",
        "rank": int(rank),
        "seed": int(seed),
        "widths": dict(cfg),
        "parameters": audit,
        "w2_reference_parameters": int(ref_audit["total_parameters"]),
        "w2_part_bitwise_equal_reference": base_equal,
        "fingerprints": fp,
        "matches_historical_fixed1_init": matches_hist,
        "init": init_audit,
        "lowrank_init": lowrank_audit,
        "gamma": float(C.GAMMA),
        "device": str(device),
    }
    strict = True if strict_w2_fingerprint is None else bool(strict_w2_fingerprint)
    if verify and strict:
        recorded = C.W2_SEED0["trunk_init_fingerprint"]
        if fp["base_state_sha256"] != recorded:
            raise RuntimeError(f"W2 trunk fingerprint {fp['base_state_sha256']} != recorded {recorded}")
    return model, out


def build_fixed1_reference(
    spec: Any, norm: Mapping[str, Any], n_out: int, *, seed: int = 0
) -> tuple[Any, dict[str, Any]]:
    """The inherited FIXED1 model via the historical constructor (read-only)."""
    return DSM.build_ds_model(spec, norm, n_out, arm="FIXED1", seed=int(seed), device="cpu")


def dtype_and_device(model: nn.Module) -> tuple[torch.dtype, torch.device]:
    p = next(model.parameters())
    return p.dtype, p.device


__all__ = [
    "FRSModel",
    "LowRankBranch",
    "build_fixed1_reference",
    "build_frs_model",
    "init_lowrank",
    "LOWRANK_INIT_SEED",
]
