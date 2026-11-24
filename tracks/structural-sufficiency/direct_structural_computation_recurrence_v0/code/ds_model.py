"""The five parameter-matched structural computations of DSCR v0.

Every arm is the frozen historical LCRC-W2 graph plus the *same* new modules::

    bond_proj : Linear(13, 128)                              (1,792 params)
    blocks    : [MessageFFN(128, 256), MessageFFN(128, 256)]  (2 x 66,176 params)

so every arm has exactly 1,128,468 parameters and shares the frozen W2 seed-0
base state bit for bit.  The arms differ *only* in which states the second
block reads:

===========  =====================================================================
``DYNAMIC``  ``h1 = h0 + FFN1(LN1(h0), sum_j SiLU(LN1(h0)[j] + e_ji))``
             ``h2 = h1 + FFN2(LN2(h1), sum_j SiLU(LN2(h1)[j] + e_ji))``   (reads h1[j])
``SELF``     both blocks read ``z[i]`` (the target's own state) on every real
             incoming bond; no other node's hidden state is read at all
``FIXED1``   layer 2 reads ``LN2(h0)[j]`` -- the *initial* neighbour state
``REAL2``    layer 2 reads ``LN2(h0)[j] + gamma * c[j]`` where ``c[j]`` is the
             real rooted branch context of the source node ``j`` computed from
             ``LN2(h0)`` only (direct 2-hop computation from ``h0``)
``BAG2``     identical to REAL2 except that the outer edge ``j -> i`` is paired
             with *every* branch context ``c[l]`` of the same target root ``i``
             (multiset cross product; the real correspondence is removed)
===========  =====================================================================

Nothing else changes: atom encoder, bond encoder, S1/S2 ``local`` content,
dropout, ``f``, ``rho``, the four-head attention readout and every graph
descriptor are the inherited W2 modules, and ``post_local(h2, batch)`` produces
the logits.  No extra head, no extra loss, no pooling on the new state.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import ds_common as C
import bs_model as BM
from sg_model import MessageFFN


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def directed_bonds(model: "DSModel", batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(src, dst, edge_msg)`` for both directions of every physical bond."""
    idx = batch["bond_index"]
    width = int(model.cfg["width"])
    if idx.numel() == 0:
        empty = idx.new_zeros((0,))
        return empty, empty, batch["bond_feat"].new_zeros((0, width))
    src = torch.cat([idx[:, 0], idx[:, 1]], dim=0)
    dst = torch.cat([idx[:, 1], idx[:, 0]], dim=0)
    bf = batch["bond_feat"]
    edge_msg = model.bond_proj(torch.cat([bf, bf], dim=0))
    return src, dst, edge_msg


def _apply_ffn(block: MessageFFN, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``FFN_l(z, m) = fc2(SiLU(fc1(z + m)))`` -- the inherited block arithmetic."""
    return block.fc2(F.silu(block.fc1(z + m)))


def _agg_source(z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_msg: torch.Tensor,
                extra: torch.Tensor | None = None) -> torch.Tensor:
    """``m[i] = sum_j SiLU(z[j] + e_ji [+ extra_j])`` over real directed bonds."""
    if src.numel() == 0:
        return torch.zeros_like(z)
    term = z.index_select(0, src) + edge_msg
    if extra is not None:
        term = term + extra
    return torch.zeros_like(z).index_add_(0, dst, F.silu(term))


def _agg_self(z: torch.Tensor, dst: torch.Tensor, edge_msg: torch.Tensor) -> torch.Tensor:
    """``m[i] = sum_j SiLU(z[i] + e_ji)`` -- the target's own state only."""
    if dst.numel() == 0:
        return torch.zeros_like(z)
    return torch.zeros_like(z).index_add_(0, dst, F.silu(z.index_select(0, dst) + edge_msg))


def incident_degree(n_atoms: int, dst: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Number of real bonds incident to each node (== number of directed in-edges)."""
    deg = like.new_zeros(n_atoms)
    if dst.numel():
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=like.dtype))
    return deg


def branch_context(
    z: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_msg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``c[j] = sum_{k in N(j)} SiLU(z[k] + e_kj) / max(deg[j], 1)`` from ``h0`` only."""
    deg = incident_degree(int(z.shape[0]), dst, z)
    if src.numel() == 0:
        return torch.zeros_like(z), deg
    summed = torch.zeros_like(z).index_add_(0, dst, F.silu(z.index_select(0, src) + edge_msg))
    return summed / deg.clamp(min=1.0).unsqueeze(1), deg


def bag2_pair_index(
    src: torch.Tensor, dst: torch.Tensor, n_atoms: int
) -> dict[str, torch.Tensor]:
    """Exact ``(j -> i, l)`` pair enumeration over each target root's real in-edges.

    For target ``i`` with real in-edge sources ``N(i) = (s_1..s_d)`` the returned
    flat arrays contain the ``d**2`` pairs ``(j = s_a, l = s_b)``.  Multiplicities
    are kept, no degree is truncated and no random permutation is used.
    """
    n_edges = int(src.numel())
    device = src.device
    if n_edges == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return {
            "outer_edge": z,
            "inner_source": z,
            "pair_dst": z,
            "counts": torch.zeros(n_atoms, dtype=torch.long, device=device),
            "n_pairs": torch.zeros((), dtype=torch.long, device=device),
        }
    perm = torch.argsort(dst)
    dst_sorted = dst.index_select(0, perm)
    counts = torch.bincount(dst_sorted, minlength=int(n_atoms))
    starts = torch.cumsum(counts, 0) - counts
    d_sq = counts * counts
    # block bookkeeping per pair
    pair_block_start = torch.cumsum(d_sq, 0) - d_sq
    block_start_of_pair = torch.repeat_interleave(starts, d_sq)
    d_of_pair = torch.repeat_interleave(counts, d_sq)
    g_of_pair = torch.repeat_interleave(pair_block_start, d_sq)
    n_pairs = int(d_sq.sum().item())
    flat = torch.arange(n_pairs, device=device, dtype=torch.long)
    inner_pos = block_start_of_pair + (flat - g_of_pair) % d_of_pair
    outer_pos = torch.repeat_interleave(torch.arange(n_edges, device=device), counts.index_select(0, dst_sorted))
    outer_edge = perm.index_select(0, outer_pos)
    inner_edge = perm.index_select(0, inner_pos)
    return {
        "outer_edge": outer_edge,
        "inner_source": src.index_select(0, inner_edge),
        "pair_dst": dst.index_select(0, outer_edge),
        "counts": counts,
        "n_pairs": torch.as_tensor(n_pairs, device=device, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class DSModel(BM.BSModel):
    """Frozen W2 + two parameter-matched structural computation blocks."""

    def __init__(
        self,
        spec: Any,
        norm: Mapping[str, Any],
        n_out: int,
        cfg: Mapping[str, Any],
        *,
        arm: str,
        n_layers: int = 2,
        block_hidden: int = 256,
        bag2_chunk: int = 1 << 16,
    ):
        super().__init__(spec, norm, int(n_out), dict(cfg))
        if arm not in C.DEEP_ARMS:
            raise KeyError(f"{arm} is not a deep arm")
        if int(n_layers) != 2:
            raise ValueError("this round fixes exactly two structural blocks")
        width = int(self.cfg["width"])
        if int(block_hidden) != width * 2:
            raise ValueError("block hidden width is 2 * width")
        self.arm = str(arm)
        self.bag2_chunk = int(bag2_chunk)
        self.bond_proj = nn.Linear(int(spec.bond_feat_dim), width)
        self.blocks = nn.ModuleList(
            [MessageFFN(width=width, hidden=int(block_hidden)) for _ in range(int(n_layers))]
        )

    # -- parameter grouping -------------------------------------------------

    def new_parameter_names(self) -> list[str]:
        return [n for n, _ in self.named_parameters() if C.is_new_parameter(n)]

    def new_modules(self) -> list[nn.Module]:
        return [*self.blocks, self.bond_proj]

    def encode_h0(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, scal = self._encode(batch)
        return a, b, scal, self.drop(self.local(batch, a, b, scal))

    # -- the five computations ---------------------------------------------

    def stage1(
        self,
        h0: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_msg: torch.Tensor,
        *,
        arm: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer 1 (identical for every deep arm).  Returns ``(h1, m1)``."""
        name = self.arm if arm is None else str(arm)
        blk1 = self.blocks[0]
        z1 = blk1.norm(h0)
        m1 = _agg_self(z1, dst, edge_msg) if name == "SELF" else _agg_source(z1, src, dst, edge_msg)
        return h0 + _apply_ffn(blk1, z1, m1), m1

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
        """Layer 2 -- the only thing that differs between the five arms."""
        name = self.arm if arm is None else str(arm)
        g = float(C.GAMMA if gamma is None else gamma)
        blk2 = self.blocks[1]
        info: dict[str, Any] = {"arm": name, "gamma": g}
        if name == "DYNAMIC":
            z2 = blk2.norm(h1)
            m2 = _agg_source(z2, src, dst, edge_msg)
            h2 = h1 + _apply_ffn(blk2, z2, m2)
            info["layer2_reads"] = "h1 (updated neighbour states)"
        elif name == "SELF":
            z2 = blk2.norm(h1)
            m2 = _agg_self(z2, dst, edge_msg)
            h2 = h1 + _apply_ffn(blk2, z2, m2)
            info["layer2_reads"] = "h1[i] (own state only, through every incoming bond)"
        elif name == "FIXED1":
            z_target = blk2.norm(h1)
            z_source = blk2.norm(h0)
            m2 = _agg_source(z_source, src, dst, edge_msg)
            h2 = h1 + _apply_ffn(blk2, z_target, m2)
            info["layer2_reads"] = "h0 of the real 1-hop neighbours + h1[i]"
        elif name in ("REAL2", "BAG2"):
            z_target = blk2.norm(h1)
            z = blk2.norm(h0)
            deg = incident_degree(int(z.shape[0]), dst, z)
            if c_override is None:
                c, deg = branch_context(z, src, dst, edge_msg)
            else:
                c = c_override
            info.update(
                {
                    "n_directed_edges": int(src.numel()),
                    "mean_degree": float(deg.mean().item()) if deg.numel() else 0.0,
                    "max_degree": int(deg.max().item()) if deg.numel() else 0,
                    "c": c,
                }
            )
            if name == "REAL2":
                if src.numel():
                    m2 = torch.zeros_like(z).index_add_(
                        0, dst, F.silu(z.index_select(0, src) + edge_msg + g * c.index_select(0, src))
                    )
                else:
                    m2 = torch.zeros_like(z)
                info["layer2_reads"] = "h0 of the real 2-hop rooted branch (real correspondence)"
                info["n_pairs"] = int(src.numel())
            else:
                pairs = bag2_pair_index(src, dst, int(z.shape[0]))
                n_pairs = int(pairs["n_pairs"].item())
                info["n_pairs"] = n_pairs
                acc = torch.zeros_like(z)
                outer = pairs["outer_edge"]
                inner_src = pairs["inner_source"]
                pair_dst = pairs["pair_dst"]
                chunk = max(1, int(self.bag2_chunk))
                for start in range(0, n_pairs, chunk):
                    stop = min(n_pairs, start + chunk)
                    term = (
                        z.index_select(0, src.index_select(0, outer[start:stop]))
                        + edge_msg.index_select(0, outer[start:stop])
                        + g * c.index_select(0, inner_src[start:stop])
                    )
                    acc.index_add_(0, pair_dst[start:stop], F.silu(term))
                m2 = acc / pairs["counts"].clamp(min=1.0).unsqueeze(1).to(z.dtype)
                info["layer2_reads"] = "h0 of the real 2-hop rooted branch (pairing removed)"
            h2 = h1 + _apply_ffn(blk2, z_target, m2)
        else:  # pragma: no cover - defensive
            raise KeyError(name)
        info["m2"] = m2
        return h2, m2, info

    def structural_layers(
        self,
        h0: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        gamma: float | None = None,
        arm: str | None = None,
        c_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return ``(h1, h2, info)`` for this arm's arithmetic."""
        name = self.arm if arm is None else str(arm)
        src, dst, edge_msg = directed_bonds(self, batch) if edges is None else edges
        h1, _m1 = self.stage1(h0, src, dst, edge_msg, arm=name)
        h2, _m2, info = self.stage2(
            h1, h0, src, dst, edge_msg, arm=name, gamma=gamma, c_override=c_override
        )
        return h1, h2, info

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_aux: bool = False,
        return_attention: bool = False,
        gamma: float | None = None,
        arm: str | None = None,
        return_states: bool = False,
    ):
        a, b, scal, h0 = self.encode_h0(batch)
        h1, h2, info = self.structural_layers(h0, batch, gamma=gamma, arm=arm)
        if not return_aux:
            if return_states:
                return self.post_local(h2, batch), {"h0": h0, "h1": h1, "h2": h2, "info": info}
            return self.post_local(h2, batch)
        logits, aux = self.post_local(h2, batch, return_aux=True, return_attention=return_attention)
        aux.update({"h0": h0, "h1": h1, "h2": h2, "a": a, "b": b, "scal": scal, "info": info})
        return logits, aux


# ---------------------------------------------------------------------------
# construction (identical rule for every deep arm)
# ---------------------------------------------------------------------------

#: Guard: a deep arm at seed 0 must carry the *recorded* W2 seed-0 trunk
#: fingerprint.  The CPU-only quick check mode runs with a stand-in normaliser
#: and disables the guard; every remote run keeps it on.
STRICT_W2_FINGERPRINT = True


def build_ds_model(
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    arm: str,
    seed: int = 0,
    device: torch.device | str = "cpu",
    zero_residual: bool = True,
    verify: bool = True,
    bag2_chunk: int = 1 << 16,
    strict_w2_fingerprint: bool | None = None,
) -> tuple[DSModel, dict[str, Any]]:
    """Build one arm from the official W2 constructor state of ``seed``.

    The W2 part comes *from* :func:`bs_model.build_model` (never re-derived); the
    new modules are constructed inside a forked RNG and then filled from the
    historical private generator (``seed * 1_000_003 + 17``), with both ``fc2``
    weights and biases zeroed so that epoch 0 reproduces the untrained W2 exactly.
    """
    cfg = dict(C.BC.config_for_scale(2))
    reference, ref_audit = BM.build_model("W2", spec, norm, int(n_out), seed=int(seed), device="cpu",
                                          verify_parameters=True)
    rng_after_w2 = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = DSModel(spec, norm, int(n_out), cfg, arm=arm, bag2_chunk=int(bag2_chunk))
    if not torch.equal(rng_after_w2, torch.get_rng_state()):
        raise RuntimeError("DSModel construction advanced the global torch RNG")
    load = model.load_state_dict(reference.state_dict(), strict=False)
    expected_missing = {k for k in model.state_dict() if C.is_new_parameter(k)}
    if set(load.missing_keys) != expected_missing or load.unexpected_keys:
        raise RuntimeError(f"W2 state load mismatch: {sorted(load.missing_keys)}")
    base_before = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    init_audit = C.initialize_dynamic_modules(model, seed=int(seed), zero_residual=bool(zero_residual))
    base_after = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    if base_before != base_after:
        raise RuntimeError("new-module initialisation changed the W2 part of the state")
    model.to(torch.device(device))
    audit = C.parameter_audit(model)
    if verify and not audit["all_match"]:
        raise RuntimeError(f"parameter audit failed for {arm}: {audit['checks']}")
    ref_state = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
    ours_base = C.state_subset({k: v.detach().cpu() for k, v in model.state_dict().items()}, new=False)
    base_equal = C.compare_states(ours_base, ref_state)
    if verify and not base_equal["all_equal"]:
        raise RuntimeError(f"W2 part is not bitwise identical to the W2 constructor: {base_equal}")
    out = {
        "arm": str(arm),
        "role": C.ARMS[arm]["role"],
        "access_class": C.ARMS[arm]["class"],
        "seed": int(seed),
        "widths": dict(cfg),
        "parameters": audit,
        "w2_reference_parameters": int(ref_audit["total_parameters"]),
        "w2_part_bitwise_equal_reference": base_equal,
        "w2_trunk_init_fingerprint": base_before,
        "w2_trunk_fingerprint_matches_recorded_w2_seed0": base_before
        == C.W2_SEED0["trunk_init_fingerprint"],
        "base_state_sha256": base_before,
        "new_state_sha256": C.state_fingerprint(C.state_subset(model.state_dict(), new=True)),
        "init": init_audit,
        "gamma": float(C.GAMMA),
        "n_state_tensors": len(model.state_dict()),
        "device": str(device),
    }
    strict = STRICT_W2_FINGERPRINT if strict_w2_fingerprint is None else bool(strict_w2_fingerprint)
    if verify and strict and int(seed) == 0 and not out["w2_trunk_fingerprint_matches_recorded_w2_seed0"]:
        raise RuntimeError(
            f"W2 trunk fingerprint {base_before} != recorded W2 seed-0 "
            f"{C.W2_SEED0['trunk_init_fingerprint']}"
        )
    return model, out


def build_w2(spec: Any, norm: Mapping[str, Any], n_out: int, *, seed: int, device: torch.device | str) -> tuple[nn.Module, dict[str, Any]]:
    """The plain W2 baseline (994,324 parameters) at the same seed."""
    model, audit = BM.build_model("W2", spec, norm, int(n_out), seed=int(seed), device="cpu")
    model.to(torch.device(device))
    return model, {
        "arm": "W2",
        "role": C.ARMS["W2"]["role"],
        "access_class": C.ARMS["W2"]["class"],
        "seed": int(seed),
        "widths": dict(audit["widths"]),
        "parameters": {
            "total": int(audit["total_parameters"]),
            "base": int(audit["base_parameters"]),
            "readout": int(audit["readout_parameters"]),
            "groups": {"w2_base": int(audit["base_parameters"]), "w2_readout": int(audit["readout_parameters"])},
        },
        "w2_trunk_init_fingerprint": audit["trunk_init_fingerprint"],
        "w2_trunk_fingerprint_matches_recorded_w2_seed0": audit["trunk_init_fingerprint"]
        == C.W2_SEED0["trunk_init_fingerprint"],
        "base_state_sha256": C.state_fingerprint(trunk_state(model)),
        "init": audit["init"],
        "n_state_tensors": len(model.state_dict()),
        "device": str(device),
    }


def trunk_state(model: nn.Module) -> dict[str, Any]:
    return {k: v for k, v in model.state_dict().items() if not k.startswith("readout.")}


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item()) if x.numel() else 0.0


def usage_metrics(model: DSModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    """``R1 / R2 / Rtotal`` and the new-module output scale (eval mode)."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            _a, _b, _scal, h0 = model.encode_h0(batch)
            h1, h2, info = model.structural_layers(h0, batch)
            d1, d2, dt = h1 - h0, h2 - h1, h2 - h0
            r0, r1 = rms(h0), rms(h1)
            out = {
                "rms_h0": r0,
                "rms_h1": r1,
                "rms_h2": rms(h2),
                "rms_h1_minus_h0": rms(d1),
                "rms_h2_minus_h1": rms(d2),
                "rms_h2_minus_h0": rms(dt),
                "R1": (rms(d1) / r0) if r0 else None,
                "R2": (rms(d2) / r1) if r1 else None,
                "Rtotal": (rms(dt) / r0) if r0 else None,
                "finite_fraction_h2": float(torch.isfinite(h2).float().mean().item()),
                "fc2_weight_norm": {
                    f"blocks.{i}": float(model.blocks[i].fc2.weight.detach().norm().item()) for i in (0, 1)
                },
                "fc2_bias_norm": {
                    f"blocks.{i}": float(model.blocks[i].fc2.bias.detach().norm().item()) for i in (0, 1)
                },
                "fc1_weight_norm": {
                    f"blocks.{i}": float(model.blocks[i].fc1.weight.detach().norm().item()) for i in (0, 1)
                },
                "bond_proj_weight_norm": float(model.bond_proj.weight.detach().norm().item()),
                "cos_h0_h1": float(
                    torch.nn.functional.cosine_similarity(h0.flatten(), h1.flatten(), dim=0).item()
                ),
                "n_atoms": int(h0.shape[0]),
                "layer2_reads": info.get("layer2_reads"),
                "n_directed_edges": int(info.get("n_directed_edges", 0) or 0),
                "n_pairs": int(info.get("n_pairs", 0) or 0),
                "mean_degree": info.get("mean_degree"),
                "max_degree": info.get("max_degree"),
            }
            logits = model.post_local(h2, batch)
            out["finite_logits"] = bool(torch.isfinite(logits).all().item())
            out["logits_abs_max"] = float(logits.abs().max().item())
    finally:
        if was_training:
            model.train()
    return out


def module_output_scales(model: DSModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Per-block output RMS and the message-term RMS (is a block suppressed?)."""
    was_training = model.training
    model.eval()
    rec: dict[str, Any] = {}
    handles = []
    try:
        def _hook(name):
            def __call__(_m, _inp, out):
                rec[f"{name}_out_rms"] = rms(out)

            return __call__

        for i, blk in enumerate(model.blocks):
            handles.append(blk.register_forward_hook(_hook(f"blocks.{i}")))
        with torch.inference_mode():
            _a, _b, _scal, h0 = model.encode_h0(batch)
            src, dst, edge_msg = directed_bonds(model, batch)
            h1, h2, _info = model.structural_layers(h0, batch, edges=(src, dst, edge_msg))
        rec["rms_h0"] = rms(h0)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    return rec


def randomize_new_modules(model: DSModel, *, seed: int = 20261125) -> dict[str, str]:
    """Deterministic *non-zero* weights for the dependency checks.

    The zero-initialised ``fc2`` would hide every structural dependence, so the
    information-dependency audit uses these weights instead.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for module in model.new_modules():
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    bound = 1.0 / math.sqrt(float(sub.in_features))
                    sub.weight.uniform_(-bound, bound, generator=gen)
                    if sub.bias is not None:
                        sub.bias.uniform_(-bound, bound, generator=gen)
                elif isinstance(sub, nn.LayerNorm):
                    nn.init.ones_(sub.weight)
                    nn.init.zeros_(sub.bias)
    return {"randomised_utc": C.utcnow(), "generator_seed": int(seed)}


def nonzero_fc2_state(model: DSModel) -> dict[str, float]:
    return {
        f"blocks.{i}.fc2.weight_absmax": float(model.blocks[i].fc2.weight.detach().abs().max().item())
        for i in (0, 1)
    }


__all__ = [
    "DSModel",
    "MessageFFN",
    "bag2_pair_index",
    "branch_context",
    "build_ds_model",
    "build_w2",
    "directed_bonds",
    "incident_degree",
    "module_output_scales",
    "nonzero_fc2_state",
    "randomize_new_modules",
    "rms",
    "trunk_state",
    "usage_metrics",
]
