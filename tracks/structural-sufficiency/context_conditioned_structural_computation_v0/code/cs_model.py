"""The CSC v0 architecture: one shared first stage and four second stages.

Inherited, unchanged (section 4.1 of ``PROTOCOL.md``)::

    a, b, scal = BSModel._encode(batch)
    h0 = Dropout(LocalContent(batch, a, b, scal))          # 64-d
    y  = BSModel.post_local(h2, batch)                     # f + rho + 4-head attention residual

New modules (identical for every arm except the Gate, which only C/D have)::

    bond_proj : Linear(bond_feat_dim -> 64)
    blocks    : [MessageFFN(64, 128), MessageFFN(64, 128)]
    gate      : Linear(129 -> 16) -> SiLU -> Linear(16 -> 5*128)     # arms C/D only

Stage 1 (identical for all four arms)::

    z1 = LN1(h0)
    m1_ji = SiLU(z1_j + e_ji)
    h1 = h0 + fc2_1(SiLU(fc1_1(z1 + Sum_j m1_ji)))

Stage 2::

    z_i = LN2(h1_i);  s_j = LN2(h0_j)                      # arm B: s_i = LN2(h0_i)
    m_ji = SiLU(s_j + e_ji);  M_i = Sum_j m_ji
    a_i = W_in z_i + b_in;    p_ji = W_in m_ji;  B_i = Sum_j p_ji
    P_i,r = Sum_{j: relation(ji)=r} p_ji                   # arm C: P_null_i,r = n_i,r/max(n_i,1) * B_i
    c_i = concat(z_i, M_i / max(n_i,1), log1p(n_i))
    delta_i,r = 0.5 * tanh(Gate(c_i)_r)
    Q_i = Sum_r delta_i,r * P_i,r
    h2_i = h1_i + fc2_2(SiLU(a_i + B_i) + Q_i)

``Q_i = 0`` reproduces the historical FIXED1 second stage exactly, and
``delta = 0`` reproduces it from the CSC arms as well.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cs_common as C

C.setup_paths()

import bs_common as BC  # noqa: E402
import bs_model as BM  # noqa: E402
from sg_model import MessageFFN  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item()) if x.numel() else 0.0


def directed_bonds(model: "CSCModel", batch: Mapping[str, Any]):
    """``(src, dst, edge_msg, relation)`` for both directions of every physical bond."""
    idx = batch["bond_index"]
    width = int(model.cfg["width"])
    if idx.numel() == 0:
        empty = idx.new_zeros((0,))
        return empty, empty, batch["bond_feat"].new_zeros((0, width)), empty
    src = torch.cat([idx[:, 0], idx[:, 1]], dim=0)
    dst = torch.cat([idx[:, 1], idx[:, 0]], dim=0)
    bf = batch["bond_feat"]
    edge_msg = model.bond_proj(torch.cat([bf, bf], dim=0))
    bt = batch["bond_type"]
    mapping = torch.as_tensor(
        [C.BOND_TYPE_TO_RELATION[i] for i in range(1, len(C.BOND_TYPE_TO_RELATION) + 1)],
        dtype=torch.long,
        device=bt.device,
    )
    rel = mapping[(bt - 1).clamp(0, mapping.numel() - 1)]
    rel = torch.cat([rel, rel], dim=0)
    return src, dst, edge_msg, rel


def relation_counts(dst: torch.Tensor, rel: torch.Tensor, n_atoms: int, dtype) -> torch.Tensor:
    """``n[N, R]`` real in-edge count per relation group."""
    n = torch.zeros((int(n_atoms), C.N_RELATIONS), dtype=dtype, device=dst.device)
    if dst.numel():
        key = dst * C.N_RELATIONS + rel
        n.view(-1).index_add_(0, key, torch.ones(key.numel(), dtype=dtype, device=dst.device))
    return n


def group_sum(values: torch.Tensor, dst: torch.Tensor, rel: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """``S[N, R, d] = sum_{j: relation(ji)=r} values[j]`` over real directed edges."""
    d = int(values.shape[1])
    out = torch.zeros((int(n_atoms), C.N_RELATIONS, d), dtype=values.dtype, device=values.device)
    if dst.numel():
        key = dst * C.N_RELATIONS + rel
        out.view(int(n_atoms) * C.N_RELATIONS, d).index_add_(0, key, values)
    return out


def gamma_ratio(a: torch.Tensor, b_total: torch.Tensor) -> torch.Tensor:
    """``[SiLU(a+B) - SiLU(a)] / B`` with a stable limit at ``B -> 0``.

    The near-zero branch uses the analytic derivative ``SiLU'(a)``; the division
    is evaluated in float64 so the ratio never suffers from low-precision
    cancellation of two nearly equal float32 numbers.
    """
    a64 = a.double()
    b64 = b_total.double()
    exact = (F.silu(a64 + b64) - F.silu(a64)) / torch.where(
        b64.abs() > C.GAMMA_EPS, b64, torch.ones_like(b64)
    )
    sig = torch.sigmoid(a64)
    deriv = sig * (1.0 + a64 * (1.0 - sig))
    small = b64.abs() <= C.GAMMA_EPS
    return torch.where(small, deriv.expand_as(exact), exact).to(a.dtype)


def silu_prime(a: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(a.double())
    return (sig * (1.0 + a.double() * (1.0 - sig))).to(a.dtype)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class CSCModel(BM.BSModel):
    """Frozen 1x LCRC-A trunk + the CSC structural computation."""

    def __init__(
        self,
        spec: Any,
        norm: Mapping[str, Any],
        n_out: int,
        cfg: Mapping[str, Any],
        *,
        arm: str,
    ):
        super().__init__(spec, norm, int(n_out), dict(cfg))
        if arm not in C.ARMS:
            raise KeyError(arm)
        self.arm = str(arm)
        width = int(self.cfg["width"])
        if width != C.WIDTH:
            raise ValueError(f"this round fixes d = {C.WIDTH}")
        hidden = int(C.BLOCK_HIDDEN)
        self.bond_proj = nn.Linear(int(spec.bond_feat_dim), width)
        self.blocks = nn.ModuleList([MessageFFN(width=width, hidden=hidden) for _ in range(2)])
        if C.ARMS[arm]["statistics"]:
            self.gate = nn.Sequential(
                nn.Linear(2 * width + 1, int(C.GATE_HIDDEN)),
                nn.SiLU(),
                nn.Linear(int(C.GATE_HIDDEN), C.N_RELATIONS * hidden),
            )
        else:
            self.gate = None

    # -- parameter grouping -------------------------------------------------

    def new_parameter_names(self) -> list[str]:
        return [n for n, _ in self.named_parameters() if C.is_new_parameter(n)]

    def new_modules(self) -> list[nn.Module]:
        out: list[nn.Module] = [*self.blocks, self.bond_proj]
        if self.gate is not None:
            out.append(self.gate)
        return out

    # -- stage 0 ------------------------------------------------------------

    def encode_h0(self, batch: Mapping[str, Any]):
        a, b, scal = self._encode(batch)
        return a, b, scal, self.drop(self.local(batch, a, b, scal))

    # -- stage 1 ------------------------------------------------------------

    def stage1(self, h0: torch.Tensor, src, dst, edge_msg) -> tuple[torch.Tensor, dict[str, Any]]:
        blk1 = self.blocks[0]
        z1 = blk1.norm(h0)
        if src.numel():
            m1 = torch.zeros_like(z1).index_add_(0, dst, F.silu(z1.index_select(0, src) + edge_msg))
        else:
            m1 = torch.zeros_like(z1)
        h1 = h0 + blk1.fc2(F.silu(blk1.fc1(z1 + m1)))
        return h1, {"z1": z1, "m1": m1}

    # -- stage 2 ------------------------------------------------------------

    def stage2(
        self,
        h1: torch.Tensor,
        h0: torch.Tensor,
        src,
        dst,
        edge_msg,
        rel,
        *,
        content: str | None = None,
        delta_mode: str = "dynamic",
        delta_bar: torch.Tensor | None = None,
        zero_correction: bool = False,
        return_parts: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Second stage.  ``content`` overrides the arm's grouping when given."""
        blk2 = self.blocks[1]
        n_atoms = int(h1.shape[0])
        z = blk2.norm(h1)
        s = blk2.norm(h0)
        if self.arm == "B":
            if dst.numel():
                m = F.silu(z.index_select(0, dst) + edge_msg)
            else:
                m = torch.zeros_like(z)
            src_state = None
        else:
            src_state = s
            if src.numel():
                m = F.silu(s.index_select(0, src) + edge_msg)
            else:
                m = torch.zeros_like(z)
        M = torch.zeros_like(z)
        if dst.numel():
            M.index_add_(0, dst, m)
        n_r = relation_counts(dst, rel, n_atoms, z.dtype)
        n = n_r.sum(dim=1)
        denom = n.clamp(min=1.0).unsqueeze(1)

        info: dict[str, Any] = {}
        return_correction = bool(self.gate is not None and not zero_correction)
        if return_correction:
            if content is None:
                content = "mixed" if self.arm == "C" else "real"
            if content not in ("real", "mixed"):
                raise KeyError(content)
        a = blk2.fc1(z)  # W_in z + b_in, the bias appears exactly once
        B = F.linear(M, blk2.fc1.weight, None)
        aB = F.linear(z + M, blk2.fc1.weight, blk2.fc1.bias)
        u = F.silu(aB)

        if return_correction:
            if content == "real":
                P = F.linear(group_sum(m, dst, rel, n_atoms), blk2.fc1.weight, None)
            else:
                P = (n_r / denom).unsqueeze(-1) * B.unsqueeze(1)
            c_state = torch.cat([z, M / denom, torch.log1p(n).unsqueeze(1)], dim=1)
            gate_out = self.gate(c_state).view(n_atoms, C.N_RELATIONS, -1)
            if delta_mode == "dynamic":
                delta = float(C.DELTA_SCALE) * torch.tanh(gate_out)
            elif delta_mode == "bar":
                if delta_bar is None:
                    raise ValueError("delta_mode='bar' needs the frozen Train relation means")
                delta = delta_bar.to(z.dtype).view(1, C.N_RELATIONS, -1).expand(n_atoms, C.N_RELATIONS, -1)
            elif delta_mode == "zero":
                delta = torch.zeros_like(gate_out)
            else:
                raise KeyError(delta_mode)
            Q = (delta * P).sum(dim=1)
        else:
            P = None
            delta = None
            Q = None
            gate_out = None
            c_state = None
        h2 = h1 + blk2.fc2(u + (Q if Q is not None else 0.0))

        if return_parts:
            info.update(self._decomposition_parts(blk2, z, a, B, u, m, dst, rel, n_atoms, h1, h2, delta, P, n))
        info.update(
            {
                "z": z,
                "s": s,
                "m": m,
                "M": M,
                "n_r": n_r,
                "n": n,
                "a": a,
                "aB": aB,
                "B": B,
                "u": u,
                "Q": Q,
                "P": P,
                "delta": delta,
                "gate_out": gate_out,
                "content": content if return_correction else None,
                "delta_mode": delta_mode if return_correction else None,
            }
        )
        return h2, info

    def _decomposition_parts(self, blk2, z, a, B, u, m, dst, rel, n_atoms, h1, h2, delta, P, n) -> dict[str, Any]:
        """Per-message vectors of the exact decomposition (section 11.1)."""
        w_out = blk2.fc2.weight  # (d, q) -- no bias
        gamma = gamma_ratio(a, B)  # (N, q)
        base_u = F.silu(a)
        h2_base = h1 + blk2.fc2(base_u)
        q = int(blk2.fc1.weight.shape[0])
        p = F.linear(m, blk2.fc1.weight, None)
        if delta is None or not dst.numel():
            delta_msg = z.new_zeros((int(dst.numel()), q))
        else:
            delta_msg = delta[dst, rel]  # (E, q) -- the message's own relation channel
        if dst.numel():
            g = gamma.index_select(0, dst)
            v = F.linear((g + delta_msg) * p, w_out)
            v_extra = F.linear(delta_msg * p, w_out)
            dh = torch.zeros_like(h1).index_add_(0, dst, v)
        else:
            v = p.new_zeros((0, int(w_out.shape[0])))
            v_extra = v
            dh = torch.zeros_like(h1)
        return {
            "gamma": gamma,
            "p": p,
            "v": v,
            "v_extra": v_extra,
            "dh": dh,
            "h2_base": h2_base,
            "delta_msg": delta_msg,
        }

    # -- full computation ---------------------------------------------------

    def structural(
        self,
        h0: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        edges=None,
        **stage2_kwargs,
    ):
        src, dst, edge_msg, rel = directed_bonds(self, batch) if edges is None else edges
        h1, info1 = self.stage1(h0, src, dst, edge_msg)
        h2, info2 = self.stage2(h1, h0, src, dst, edge_msg, rel, **stage2_kwargs)
        edges = {"src": src, "dst": dst, "edge_msg": edge_msg, "rel": rel}
        return h1, h2, {**info1, **info2, "edges": edges}

    def forward(self, batch: Mapping[str, Any], *, return_aux: bool = False, **stage2_kwargs):
        _a, _b, _scal, h0 = self.encode_h0(batch)
        h1, h2, info = self.structural(h0, batch, **stage2_kwargs)
        logits = self.post_local(h2, batch)
        if return_aux:
            return logits, {"h0": h0, "h1": h1, "h2": h2, "logits": logits, **info}
        return logits

    def forward_states(self, batch: Mapping[str, Any], **stage2_kwargs) -> dict[str, Any]:
        """Diagnostic forward with the full decomposition bookkeeping (no grad)."""
        stage2_kwargs.setdefault("return_parts", True)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _a, _b, _scal, h0 = self.encode_h0(batch)
                h1, h2, info = self.structural(h0, batch, **stage2_kwargs)
                logits = self.post_local(h2, batch)
        finally:
            if was_training:
                self.train()
        return {"h0": h0, "h1": h1, "h2": h2, "logits": logits, **info}


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

#: private initialisation generator of the new modules (DSCR rule, reused verbatim)
INIT_RULE = "private torch.Generator seed*1_000_003+17; Linear U(+-1/sqrt(fan_in)); LayerNorm (1,0); block fc2 and gate output zeroed"


def initialize_new_modules(model: CSCModel, *, seed: int, zero_residual: bool = True) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + 17)
    before = torch.get_rng_state().clone()
    n_linear = 0
    with torch.no_grad():
        for module in model.new_modules():
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    bound = 1.0 / math.sqrt(float(sub.in_features))
                    sub.weight.uniform_(-bound, bound, generator=gen)
                    if sub.bias is not None:
                        sub.bias.uniform_(-bound, bound, generator=gen)
                    n_linear += 1
                elif isinstance(sub, nn.LayerNorm):
                    nn.init.ones_(sub.weight)
                    nn.init.zeros_(sub.bias)
    if zero_residual:
        with torch.no_grad():
            for blk in model.blocks:
                blk.fc2.weight.zero_()
                blk.fc2.bias.zero_()
            if model.gate is not None:
                last = model.gate[-1]
                last.weight.zero_()
                last.bias.zero_()
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("new-module initialisation moved the global torch RNG state")
    return {
        "generator_seed": int(seed) * 1_000_003 + 17,
        "linear_layers_initialised": int(n_linear),
        "block_fc2_zeroed": bool(zero_residual),
        "gate_output_zeroed": bool(zero_residual and model.gate is not None),
        "global_rng_unchanged": True,
        "rule": INIT_RULE,
    }


def build_csc(
    spec: Any,
    norm: Mapping[str, Any],
    *,
    arm: str,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    verify: bool = True,
) -> tuple[CSCModel, dict[str, Any]]:
    """Build one arm from the *same* scale-1 LCRC-A initial state."""
    cfg = dict(BC.config_for_scale(C.CFG_KEY))
    reference, ref_audit = BM.build_model(
        C.CFG_KEY, spec, norm, C.N_OUT, seed=int(seed), device="cpu", verify_parameters=False
    )
    ref_state = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
    rng_after_reference = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = CSCModel(spec, norm, C.N_OUT, cfg, arm=arm)
    if not torch.equal(rng_after_reference, torch.get_rng_state()):
        raise RuntimeError("CSCModel construction advanced the global torch RNG")
    load = model.load_state_dict(ref_state, strict=False)
    expected_missing = {k for k in model.state_dict() if C.is_new_parameter(k)}
    if set(load.missing_keys) != expected_missing or load.unexpected_keys:
        raise RuntimeError(
            f"base state load mismatch: missing={sorted(load.missing_keys)} unexpected={sorted(load.unexpected_keys)}"
        )
    base_before = C.state_fingerprint({k: v for k, v in model.state_dict().items() if not C.is_new_parameter(k)})
    init_audit = initialize_new_modules(model, seed=int(seed), zero_residual=True)
    base_after = C.state_fingerprint({k: v for k, v in model.state_dict().items() if not C.is_new_parameter(k)})
    if base_before != base_after:
        raise RuntimeError("new-module initialisation changed the inherited LCRC part of the state")
    model.to(torch.device(device))

    new_names = model.new_parameter_names()
    audit = {
        "arm": str(arm),
        "arm_name": C.ARMS[arm]["name"],
        "role": C.ARMS[arm]["role"],
        "seed": int(seed),
        "widths": cfg,
        "d": C.WIDTH,
        "q": C.BLOCK_HIDDEN,
        "n_relations": C.N_RELATIONS,
        "gate_hidden": C.GATE_HIDDEN,
        "delta_scale": C.DELTA_SCALE,
        "n_parameters": C.parameter_count(model),
        "n_parameters_base": int(sum(p.numel() for n, p in model.named_parameters() if not C.is_new_parameter(n))),
        "n_parameters_new": int(sum(p.numel() for n, p in model.named_parameters() if C.is_new_parameter(n))),
        "new_parameter_names": new_names,
        "new_parameter_shapes": {n: list(p.shape) for n, p in model.named_parameters() if C.is_new_parameter(n)},
        "has_gate": bool(model.gate is not None),
        "init": init_audit,
        "base_state_sha256": base_before,
        "new_state_sha256": C.state_fingerprint({k: v for k, v in model.state_dict().items() if C.is_new_parameter(k)}),
        "full_state_sha256": C.state_fingerprint(model.state_dict()),
        "lcrca_reference_parameters": int(ref_audit["total_parameters"]),
        "lcrca_reference_scale": int(ref_audit["scale"]),
    }
    if verify:
        expected_new = _expected_new_parameters(has_gate=model.gate is not None)
        if audit["n_parameters_new"] != expected_new:
            raise RuntimeError(f"{arm}: new parameters {audit['n_parameters_new']} != {expected_new}")
        audit["new_parameter_check"] = "exact"
    return model, audit


def _expected_new_parameters(*, has_gate: bool) -> int:
    bond_proj = 13 * C.WIDTH + C.WIDTH
    block = 2 * C.WIDTH + (C.WIDTH * C.BLOCK_HIDDEN + C.BLOCK_HIDDEN) + (C.BLOCK_HIDDEN * C.WIDTH + C.WIDTH)
    gate = 0
    if has_gate:
        gate = (2 * C.WIDTH + 1) * C.GATE_HIDDEN + C.GATE_HIDDEN + C.GATE_HIDDEN * (C.N_RELATIONS * C.BLOCK_HIDDEN) + C.N_RELATIONS * C.BLOCK_HIDDEN
    return int(bond_proj + 2 * block + gate)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def usage_metrics(model: CSCModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            out = model.forward_states(batch)
            h0, h1, h2 = out["h0"], out["h1"], out["h2"]
            r0, r1 = rms(h0), rms(h1)
            res = {
                "rms_h0": r0,
                "rms_h1": r1,
                "rms_h2": rms(h2),
                "rms_h1_minus_h0": rms(h1 - h0),
                "rms_h2_minus_h1": rms(h2 - h1),
                "R1": (rms(h1 - h0) / r0) if r0 else None,
                "R2": (rms(h2 - h1) / r1) if r1 else None,
                "rms_M": rms(out["M"]),
                "mean_degree": float(out["n"].mean().item()) if out["n"].numel() else 0.0,
                "max_degree": int(out["n"].max().item()) if out["n"].numel() else 0,
                "n_directed_edges": int(out["edges"]["src"].numel()),
                "finite_h2": bool(torch.isfinite(h2).all().item()),
                "logits_abs_max": float(out["logits"].abs().max().item()),
            }
            return res
    finally:
        if was_training:
            model.train()


def gate_statistics(model: CSCModel, batch: Mapping[str, Any], *, delta_bar: torch.Tensor | None = None) -> dict[str, Any]:
    """Gate output / correction / projected state-change magnitudes."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            out = model.forward_states(batch)
            blk2 = model.blocks[1]
            res: dict[str, Any] = {
                "n_atoms": int(out["z"].shape[0]),
                "n_directed_edges": int(out["edges"]["src"].numel()),
            }
            if out["delta"] is None:
                res["has_gate"] = False
                res["rms_dh"] = rms(out["dh"])
                res["Q_rms"] = 0.0
                return res
            delta = out["delta"]
            n_r = out["n_r"]
            present = n_r > 0
            d_vals = delta[present]
            res.update(
                {
                    "has_gate": True,
                    "delta_abs_mean": float(d_vals.abs().mean().item()),
                    "delta_mean": float(d_vals.mean().item()),
                    "delta_std": float(d_vals.std(unbiased=False).item()),
                    "delta_rms": float(d_vals.pow(2).mean().sqrt().item()),
                    "delta_abs_max": float(d_vals.abs().max().item()),
                    "delta_sign_negative_fraction": float((d_vals < 0).float().mean().item()),
                    "delta_tanh_saturation": float((d_vals.abs() > 0.45).float().mean().item()),
                    "delta_max_possible": float(C.DELTA_SCALE),
                    "per_relation": {},
                }
            )
            for r, name in enumerate(C.RELATION_NAMES):
                sel = delta[:, r][present[:, r]]
                res["per_relation"][name] = {
                    "n_nodes_with_relation": int(sel.shape[0]),
                    "delta_abs_mean": float(sel.abs().mean().item()) if sel.numel() else None,
                    "delta_mean": float(sel.mean().item()) if sel.numel() else None,
                    "delta_std": float(sel.std(unbiased=False).item()) if sel.numel() else None,
                    "delta_rms": float(sel.pow(2).mean().sqrt().item()) if sel.numel() else None,
                }
            hetero = present.sum(dim=1) >= 2
            res["hetero_node_fraction"] = float(hetero.float().mean().item()) if hetero.numel() else 0.0
            res["n_hetero_nodes"] = int(hetero.sum().item())
            if hetero.any():
                res["hetero_delta_abs_mean"] = float(delta[hetero][present[hetero]].abs().mean().item())
            res["Q_rms"] = rms(out["Q"])
            res["Q_abs_mean"] = float(out["Q"].abs().mean().item())
            res["rms_dh"] = rms(out["dh"])
            res["rms_h2_minus_h1"] = rms(out["h2"] - out["h1"])
            # the state change fc2 projects out of Q (fc2 is affine, no bias needed here)
            res["rms_Q_projected"] = rms(F.linear(out["Q"], blk2.fc2.weight)) if out["Q"] is not None else 0.0
            res["Q_projected_over_h2_delta"] = (
                float(res["rms_Q_projected"] / res["rms_h2_minus_h1"]) if res["rms_h2_minus_h1"] else None
            )
            # actual difference between the real and the mixed grouping
            P = out["P"]
            P_null = (out["n_r"] / out["n"].clamp(min=1.0).unsqueeze(1)).unsqueeze(-1) * out["B"].unsqueeze(1)
            active = out["n_r"] > 0
            if active.any():
                gap = (P - P_null).abs().amax(dim=-1)
                res["grouping_gap_abs_mean"] = float(gap[active].mean().item())
                rel_gap = gap[active] / (P.abs().amax(dim=-1)[active] + 1e-12)
                res["grouping_relative_gap_mean"] = float(rel_gap.mean().item())
                res["grouping_gap_nonzero_fraction"] = float((gap[active] > 1e-6).float().mean().item())
            res["fraction_nodes_single_relation_type"] = float((present.sum(dim=1) == 1).float().mean().item()) if present.numel() else 0.0
            return res
    finally:
        if was_training:
            model.train()


def train_relation_delta_bar(model: CSCModel, bundle: Any, device, *, batch_size: int | None = None) -> dict[str, Any]:
    """Train-set mean Gate output per relation, weighted by real relation occurrences (I2)."""
    import cs_data as D

    if model.gate is None:
        raise RuntimeError("delta_bar is only defined for the Gate arms")
    order, _ids, _y = D.split_order(bundle, "train")
    batch_size = int(C.RECIPE["eval_batch_size"] if batch_size is None else batch_size)
    blk2 = model.blocks[1]
    sums = torch.zeros((C.N_RELATIONS, C.BLOCK_HIDDEN), dtype=torch.float64)
    counts = torch.zeros(C.N_RELATIONS, dtype=torch.float64)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, order.size, batch_size):
            chunk = order[start:start + batch_size]
            batch = D.make_batch(bundle, "train", chunk, device)
            _a, _b, _scal, h0 = model.encode_h0(batch)
            h1, _h2, info = model.structural(h0, batch, return_parts=False)
            delta = info["delta"].detach().double()
            n_r = info["n_r"].detach().double()
            sums += torch.einsum("nr,nrq->rq", n_r, delta)
            counts += n_r.sum(dim=0)
    if was_training:
        model.train()
    bar = sums / counts.clamp(min=1.0).unsqueeze(1)
    return {
        "delta_bar": bar.to(torch.float32),
        "counts": counts,
        "method": "sum over atoms of n_i,r * delta_i,r, divided by sum_i n_i,r (Train only)",
        "delta_bar_abs_mean_by_relation": {
            name: float(bar[r].abs().mean().item()) for r, name in enumerate(C.RELATION_NAMES)
        },
    }


def randomize_new_modules(model: CSCModel, *, seed: int = 20261126) -> dict[str, Any]:
    """Deterministic non-zero new-module weights for the dependency checks."""
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
    return {"generator_seed": int(seed), "rule": "Linear U(+-1/sqrt(fan_in)) on every new Linear, LayerNorm (1,0)"}


__all__ = [
    "CSCModel",
    "INIT_RULE",
    "build_csc",
    "directed_bonds",
    "gamma_ratio",
    "gate_statistics",
    "group_sum",
    "initialize_new_modules",
    "randomize_new_modules",
    "relation_counts",
    "rms",
    "silu_prime",
    "train_relation_delta_bar",
    "usage_metrics",
]
