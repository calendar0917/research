"""The six explicit-evidence arms plus BASE / hidden and unary controls.

All arms read the *same* immutable local structure representation

    h0 = base.drop(base.local(batch, base._encode(batch)))

computed exactly once per forward.  The trunk (atom encoder, bond encoder, the
S1/S2 ``LocalContent`` block and dropout) and its initialisation are inherited
from the frozen LCRC-A / ``BSModel`` graph of ``lcrc_pcba_backbone_scaling_v1``.
``BASE`` keeps the historical ``f``, ``rho`` and attention readout untouched;
every new arm deletes those three read heads so that no unused parameter or
hidden bypass exists.

Shared token functions (identical for every arm and every seed):

    u_i   = LN(Linear(2K, K)(SiLU(Linear(d, 2K)(h0_i))))
    v_ij  = LN(Linear(2K, K)(SiLU(Linear(3K+19, 2K)(x_ij))))
    x_ij  = [u_i+u_j, u_i*u_j, |u_i-u_j|, r_ij]

Evidence aggregation (``tau = 1``, stable segment log-sum-exp):

    A(s) = sqrt(m) * [pi_mean * Mean(s) + pi_pos * LME(s) - pi_neg * LME(-s)]
    A_add(s) = sum(s) / sqrt(m)

``pi = softmax(theta)`` starts at ``[0.5, 0.25, 0.25]`` and is learned per token
stream (per task for EVID, per channel for BANK / COMPOUND / UNARY).  With a
single token ``A(s) = s`` exactly.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bs_common as BSC
import bs_model as BSM
import esr_common as C


# ---------------------------------------------------------------------------
# numerically stable segment statistics
# ---------------------------------------------------------------------------


def segment_sum(x: torch.Tensor, seg: torch.Tensor, n_segments: int) -> torch.Tensor:
    out = x.new_zeros((int(n_segments), x.shape[1]))
    return out.index_add_(0, seg, x)


def segment_logmeanexp(x: torch.Tensor, seg: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """``log(mean_{i in segment} exp(x_i))`` with an exact max shift and m=0 -> 0."""
    n_seg = int(counts.numel())
    m = counts.clamp(min=1).to(x.dtype)
    wid = x.shape[1]
    seg2 = seg.unsqueeze(1).expand(-1, wid)
    maxv = x.new_full((n_seg, wid), float("-inf")).scatter_reduce_(0, seg2, x, reduce="amax", include_self=False)
    maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
    e = torch.exp(x - maxv[seg])
    total = x.new_zeros((n_seg, wid)).index_add_(0, seg, e)
    out = maxv + torch.log(total.clamp_min(1e-30)) - torch.log(m).unsqueeze(1)
    return torch.where(counts.unsqueeze(1) > 0, out, torch.zeros_like(out))


def segment_logmeanexp_negative(
    x: torch.Tensor, seg: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """``log(mean exp(-x))`` in the same stable form (uses the segment minimum)."""
    n_seg = int(counts.numel())
    m = counts.clamp(min=1).to(x.dtype)
    wid = x.shape[1]
    seg2 = seg.unsqueeze(1).expand(-1, wid)
    minv = x.new_full((n_seg, wid), float("inf")).scatter_reduce_(0, seg2, x, reduce="amin", include_self=False)
    minv = torch.where(torch.isfinite(minv), minv, torch.zeros_like(minv))
    e = torch.exp(minv[seg] - x)
    total = x.new_zeros((n_seg, wid)).index_add_(0, seg, e)
    out = -minv + torch.log(total.clamp_min(1e-30)) - torch.log(m).unsqueeze(1)
    return torch.where(counts.unsqueeze(1) > 0, out, torch.zeros_like(out))


def evidence_aggregate(
    s: torch.Tensor,
    seg: torch.Tensor,
    counts: torch.Tensor,
    theta: torch.Tensor,
    *,
    mode: str = "evid",
) -> torch.Tensor:
    """``A(s)`` for every segment; ``theta`` has shape ``[C, 3]``."""
    if mode == "add":
        return segment_sum(s, seg, int(counts.numel())) / torch.sqrt(counts.clamp(min=1).to(s.dtype)).unsqueeze(1)
    if mode != "evid":
        raise ValueError(mode)
    if s.shape[1] != theta.shape[0]:
        raise RuntimeError(f"aggregation score width {s.shape[1]} != theta channels {theta.shape[0]}")
    m = counts.to(s.dtype).unsqueeze(1)
    root = torch.sqrt(m)
    mean = segment_sum(s, seg, int(counts.numel())) / m.clamp(min=1.0)
    lme = segment_logmeanexp(s, seg, counts)
    lme_neg = segment_logmeanexp_negative(s, seg, counts)
    pi = torch.softmax(theta, dim=-1)
    out = pi[:, 0].unsqueeze(0) * mean + pi[:, 1].unsqueeze(0) * lme - pi[:, 2].unsqueeze(0) * lme_neg
    out = root * out
    return torch.where(counts.unsqueeze(1) > 0, out, torch.zeros_like(out))


def evidence_decompose(
    s: torch.Tensor, seg: torch.Tensor, counts: torch.Tensor, theta: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Mean / positive-gap / negative-gap channel statistics (FLEX + diagnostics)."""
    m = counts.to(s.dtype).unsqueeze(1)
    root = torch.sqrt(m)
    mean = segment_sum(s, seg, int(counts.numel())) / m.clamp(min=1.0)
    lme = segment_logmeanexp(s, seg, counts)
    lme_neg = segment_logmeanexp_negative(s, seg, counts)
    pi = torch.softmax(theta, dim=-1)
    zero = counts.unsqueeze(1) == 0
    keep = lambda t: torch.where(zero, torch.zeros_like(t), t)  # noqa: E731
    return {
        "mean": keep(root * mean),
        "pos": keep(root * pi[:, 1].unsqueeze(0) * (lme - mean)),
        "neg": keep(root * pi[:, 2].unsqueeze(0) * (lme_neg + mean)),
    }


# ---------------------------------------------------------------------------
# token functions
# ---------------------------------------------------------------------------


class TokenMLP(nn.Module):
    def __init__(self, d_in: int, K: int):
        super().__init__()
        self.fc1 = nn.Linear(int(d_in), 2 * int(K))
        self.fc2 = nn.Linear(2 * int(K), int(K))
        self.norm = nn.LayerNorm(int(K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(F.silu(self.fc1(x))))


class RelationMLP(nn.Module):
    def __init__(self, K: int, rel_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(3 * int(K) + int(rel_dim), 2 * int(K))
        self.fc2 = nn.Linear(2 * int(K), int(K))
        self.norm = nn.LayerNorm(int(K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(F.silu(self.fc1(x))))


class CompoundMLP(nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.fc1 = nn.Linear(2 * int(K) + 1, 2 * int(K))
        self.fc2 = nn.Linear(2 * int(K), int(K))

    def forward(self, u: torch.Tensor, c: torch.Tensor, log_deg: torch.Tensor) -> torch.Tensor:
        z = torch.cat([u, c, log_deg.unsqueeze(1)], dim=1)
        return self.fc2(F.silu(self.fc1(z)))


# ---------------------------------------------------------------------------
# the arm
# ---------------------------------------------------------------------------


class SRENNet(nn.Module):
    """One explicit-evidence prediction function on the frozen trunk."""

    def __init__(
        self,
        base: BSM.BSModel,
        *,
        variant: str,
        K: int,
        n_tasks: int,
        global_dim: int,
        hidden_width: int | None = None,
        zero_task_heads: bool = False,
    ):
        super().__init__()
        self.base = base
        self.variant = str(variant)
        self.K = int(K)
        self.n_tasks = int(n_tasks)
        self.global_dim = int(global_dim)
        width = int(base.cfg["width"])
        self.width = width
        # the historical heads are removed: they must not be a hidden bypass
        for name in ("f", "rho", "readout"):
            if hasattr(base, name):
                delattr(base, name)
        self.U = TokenMLP(width, self.K)
        self.V = RelationMLP(self.K, C.REL_DIM)
        self.global_linear = nn.Linear(self.global_dim + C.GLOBAL_EXTRA, self.n_tasks)
        self.register_buffer("r_self", self._self_relation(), persistent=False)
        self.theta_u: nn.Parameter | None = None
        self.theta_v: nn.Parameter | None = None
        self.theta_w: nn.Parameter | None = None
        self.q_u: nn.Linear | None = None
        self.q_v: nn.Linear | None = None
        self.q_w: nn.Linear | None = None
        self.compound: CompoundMLP | None = None
        self.hidden_fc1: nn.Linear | None = None
        self.hidden_fc2: nn.Linear | None = None
        self.flex: nn.ModuleDict | None = None
        if self.variant in (C.ADD, C.EVID):
            self.q_u = nn.Linear(self.K, self.n_tasks, bias=False)
            self.q_v = nn.Linear(self.K, self.n_tasks, bias=False)
            if self.variant == C.EVID:
                self.theta_u = nn.Parameter(torch.log(torch.tensor(C.PI_INIT)).repeat(self.n_tasks, 1))
                self.theta_v = nn.Parameter(torch.log(torch.tensor(C.PI_INIT)).repeat(self.n_tasks, 1))
        elif self.variant in (C.BANK, C.UNARY, C.HIDDEN):
            if self.variant != C.HIDDEN:  # HIDDEN replaces the task-linear combination
                self.q_u = nn.Linear(self.K, self.n_tasks, bias=False)
                self.q_v = nn.Linear(self.K, self.n_tasks, bias=False)
            self.theta_u = nn.Parameter(torch.log(torch.tensor(C.PI_INIT)).repeat(self.K, 1))
            self.theta_v = nn.Parameter(torch.log(torch.tensor(C.PI_INIT)).repeat(self.K, 1))
            if self.variant == C.HIDDEN:
                if hidden_width is None:
                    raise ValueError("HIDDEN needs the capacity-matched hidden width")
                self.hidden_fc1 = nn.Linear(2 * self.K, int(hidden_width))
                self.hidden_fc2 = nn.Linear(int(hidden_width), self.n_tasks)
        elif self.variant == C.COMPOUND:
            self.compound = CompoundMLP(self.K)
            self.q_w = nn.Linear(self.K, self.n_tasks, bias=False)
            self.theta_w = nn.Parameter(torch.log(torch.tensor(C.PI_INIT)).repeat(self.K, 1))
            self.compound_norm = nn.LayerNorm(self.K)
        elif self.variant == C.FLEX:
            self.q_u = nn.Linear(self.K, self.n_tasks, bias=False)
            self.q_v = nn.Linear(self.K, self.n_tasks, bias=False)
            self.theta_u = nn.Parameter(torch.log(torch.tensor(C.PI_INIT)).repeat(self.K, 1))
            self.theta_v = nn.Parameter(torch.log(torch.tensor(C.PI_INIT)).repeat(self.K, 1))
            self.flex = nn.ModuleDict(
                {
                    f"{stream}_{part}": nn.Linear(self.K, self.n_tasks, bias=False)
                    for stream in ("u", "v")
                    for part in ("mean", "pos", "neg")
                }
            )
        else:
            raise KeyError(self.variant)
        self.zero_task_heads = bool(zero_task_heads)
        self._initialise(variant_seed=0)

    # -- construction helpers --------------------------------------------
    def _self_relation(self) -> torch.Tensor:
        r = torch.zeros(C.REL_DIM, dtype=torch.float32)
        r[0] = 1.0  # SELF one-hot; all other relation features are zero
        return r

    def _init_generator(self, variant_seed: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(20261125 + int(variant_seed))

    @staticmethod
    def _init_linear_tree(module: nn.Module, gen: torch.Generator) -> None:
        with torch.no_grad():
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    bound = 1.0 / math.sqrt(float(sub.in_features))
                    sub.weight.uniform_(-bound, bound, generator=gen)
                    if sub.bias is not None:
                        sub.bias.uniform_(-bound, bound, generator=gen)
                elif isinstance(sub, nn.LayerNorm):
                    sub.weight.fill_(1.0)
                    if sub.bias is not None:
                        sub.bias.zero_()

    def _initialise(self, *, variant_seed: int = 0) -> None:
        """Shared modules first, in a fixed order, so every arm sees the same U/V/global."""
        gen = self._init_generator(variant_seed)
        self._init_linear_tree(self.U, gen)
        self._init_linear_tree(self.V, gen)
        self._init_linear_tree(self.global_linear, gen)
        heads: list[nn.Module] = []
        if self.variant in (C.ADD, C.EVID, C.BANK, C.UNARY, C.FLEX):
            heads.extend([self.q_u, self.q_v])
        if self.variant == C.COMPOUND:
            heads.extend([self.compound, self.compound_norm, self.q_w])
        if self.variant == C.HIDDEN:
            heads.extend([self.hidden_fc1, self.hidden_fc2])
        if self.variant == C.FLEX:
            heads.extend([self.flex[k] for k in sorted(self.flex.keys())])
        for head in heads:
            if head is None:
                continue
            self._init_linear_tree(head, gen)
        if self.zero_task_heads:
            with torch.no_grad():
                for head in heads:
                    if isinstance(head, nn.Linear):
                        head.weight.zero_()

    # -- parameter inventory ---------------------------------------------
    def state_fingerprint(self) -> str:
        return C.state_fingerprint(self.state_dict())

    def active_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def module_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, child in self.named_children():
            out[name] = int(sum(p.numel() for p in child.parameters()))
        out["total"] = self.active_parameters()
        return out

    # -- forward pieces ---------------------------------------------------
    def encode_tokens(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        a, b, scal = self.base._encode(batch)
        h0 = self.base.drop(self.base.local(batch, a, b, scal))
        return h0, self.U(h0)

    def relation_tokens(self, u: torch.Tensor, batch: Mapping[str, Any]) -> torch.Tensor:
        pu, pv = batch["pair_u"], batch["pair_v"]
        if pu.numel() == 0:
            return u.new_zeros((0, self.K))
        x = torch.cat(
            [u[pu] + u[pv], u[pu] * u[pv], (u[pu] - u[pv]).abs(), batch["pair_r"]], dim=1
        )
        return self.V(x)

    def self_tokens(self, u: torch.Tensor) -> torch.Tensor:
        n = u.shape[0]
        r = self.r_self.to(device=u.device, dtype=u.dtype).unsqueeze(0).expand(n, -1)
        x = torch.cat([u + u, u * u, torch.zeros_like(u), r], dim=1)
        return self.V(x)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        h0, u = self.encode_tokens(batch)
        atom_batch = batch["atom_batch"]
        atom_counts = batch["atom_counts"]
        n_graphs = int(batch["n_graphs"])
        c = self.global_linear(batch["global_ext"])
        if self.variant in (C.ADD, C.EVID):
            v = self.relation_tokens(u, batch)
            a_scores = self.q_u(u)
            b_scores = self.q_v(v)
            mode = "add" if self.variant == C.ADD else "evid"
            if mode == "add":
                zero_theta = a_scores.new_zeros((1, 3))
                local = evidence_aggregate(a_scores, atom_batch, atom_counts, zero_theta, mode="add")
                pair = evidence_aggregate(b_scores, batch["pair_batch"], batch["pair_counts"], zero_theta, mode="add")
            else:
                assert self.theta_u is not None and self.theta_v is not None
                local = evidence_aggregate(a_scores, atom_batch, atom_counts, self.theta_u)
                pair = evidence_aggregate(b_scores, batch["pair_batch"], batch["pair_counts"], self.theta_v)
            return c + local + pair
        if self.variant == C.BANK:
            assert self.theta_u is not None and self.theta_v is not None
            v = self.relation_tokens(u, batch)
            u_bank = evidence_aggregate(u, atom_batch, atom_counts, self.theta_u)
            v_bank = evidence_aggregate(v, batch["pair_batch"], batch["pair_counts"], self.theta_v)
            return c + self.q_u(u_bank) + self.q_v(v_bank)
        if self.variant == C.UNARY:
            assert self.theta_u is not None and self.theta_v is not None
            v = self.self_tokens(u)
            u_bank = evidence_aggregate(u, atom_batch, atom_counts, self.theta_u)
            v_bank = evidence_aggregate(v, atom_batch, atom_counts, self.theta_v)
            return c + self.q_u(u_bank) + self.q_v(v_bank)
        if self.variant == C.HIDDEN:
            assert self.theta_u is not None and self.theta_v is not None
            v = self.relation_tokens(u, batch)
            u_bank = evidence_aggregate(u, atom_batch, atom_counts, self.theta_u)
            v_bank = evidence_aggregate(v, batch["pair_batch"], batch["pair_counts"], self.theta_v)
            z = torch.cat([u_bank, v_bank], dim=1)
            assert self.hidden_fc1 is not None and self.hidden_fc2 is not None
            return c + self.hidden_fc2(F.silu(self.hidden_fc1(z)))
        if self.variant == C.COMPOUND:
            assert self.theta_w is not None and self.q_w is not None and self.compound is not None
            v = self.relation_tokens(u, batch)
            pu, pv = batch["pair_u"], batch["pair_v"]
            n_atoms = int(u.shape[0])
            deg = u.new_zeros(n_atoms)
            acc = u.new_zeros((n_atoms, self.K))
            if pu.numel():
                deg = deg.index_add_(0, pu, torch.ones_like(pu, dtype=u.dtype)).index_add_(
                    0, pv, torch.ones_like(pv, dtype=u.dtype)
                )
                acc = acc.index_add_(0, pu, v).index_add_(0, pv, v)
            c_i = acc / torch.sqrt(deg.clamp(min=1.0)).unsqueeze(1)
            w = self.compound_norm(u + self.compound(u, c_i, torch.log1p(deg)))
            w_bank = evidence_aggregate(w, atom_batch, atom_counts, self.theta_w)
            return c + self.q_w(w_bank)
        if self.variant == C.FLEX:
            assert self.theta_u is not None and self.theta_v is not None and self.flex is not None
            v = self.relation_tokens(u, batch)
            u_dec = evidence_decompose(u, atom_batch, atom_counts, self.theta_u)
            v_dec = evidence_decompose(v, batch["pair_batch"], batch["pair_counts"], self.theta_v)
            out = c
            for stream, dec in (("u", u_dec), ("v", v_dec)):
                for part in ("mean", "pos", "neg"):
                    out = out + self.flex[f"{stream}_{part}"](dec[part])
            return out
        raise KeyError(self.variant)

    def _theta_like(self, s: torch.Tensor) -> torch.Tensor:
        return s.new_full((s.shape[1], 3), 0.0)


# ---------------------------------------------------------------------------
# building / initialisation
# ---------------------------------------------------------------------------


def _build_base(scale: int, spec: Any, norm: Mapping[str, Any], n_tasks: int, seed: int) -> tuple[BSM.BSModel, dict[str, Any]]:
    model, audit = BSM.build_model(int(scale), spec, norm, int(n_tasks), seed=int(seed), device="cpu", verify_parameters=False)
    return model, audit


def choose_hidden_width(
    *,
    K: int,
    n_tasks: int,
    global_dim: int,
    bank_active: int,
) -> dict[str, Any]:
    """Deterministic H in {K/2, K, 2K} with the closest total active parameter count."""
    candidates = sorted({max(1, int(K // 2)), int(K), int(2 * K)})
    rows = []
    for h in candidates:
        hidden_readout = 2 * int(K) * h + h + h * int(n_tasks) + int(n_tasks)
        bank_readout = 2 * int(K) * int(n_tasks)
        rows.append({"H": int(h), "total_delta": int(hidden_readout - bank_readout)})
    best = min(rows, key=lambda r: (abs(int(r["total_delta"])), int(r["H"])))
    return {
        "candidates": rows,
        "selected_H": int(best["H"]),
        "bank_active_parameters": int(bank_active),
        "selected_total_delta": int(best["total_delta"]),
        "relative_delta": float(abs(best["total_delta"]) / max(bank_active, 1)),
        "tie_break": "smaller H",
        "rule": "minimise |active total parameters - BANK|; ties keep the smaller H",
    }


def build_model(
    *,
    arm: str,
    view: Any,
    seed: int = 0,
    device: torch.device | str = "cpu",
    hidden_width: int | None = None,
    zero_task_heads: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    dataset = view.dataset
    meta = C.DATASETS[dataset]
    base, base_audit = _build_base(int(meta["scale"]), view.spec, view.train_norm, int(meta["n_tasks"]), int(seed))
    info: dict[str, Any] = {
        "arm": arm,
        "dataset": dataset,
        "seed": int(seed),
        "scale": int(meta["scale"]),
        "K": int(meta["K"]),
        "n_tasks": int(meta["n_tasks"]),
        "widths": dict(base.cfg),
        "trunk_init_fingerprint": base_audit["trunk_init_fingerprint"],
        "full_trunk_init_fingerprint": base_audit["full_trunk_init_fingerprint"],
        "trunk_parameters": int(base_audit["base_parameters"]),
    }
    if arm == C.BASE:
        model: nn.Module = base
        info["parameters"] = int(base_audit["total_parameters"])
        info["module_breakdown"] = {name: int(sum(p.numel() for p in child.parameters())) for name, child in model.named_children()}
        info["readout_parameters"] = int(base_audit["readout_parameters"])
        info["new_parameters"] = 0
        info["hidden_width"] = None
        model.to(torch.device(device))
        info["state_fingerprint"] = C.state_fingerprint(model.state_dict())
        return model, info

    if hidden_width is None and arm == C.HIDDEN:
        ref_model, ref_info = build_model(arm=C.BANK, view=view, seed=seed, device="cpu")
        sel = choose_hidden_width(
            K=int(meta["K"]),
            n_tasks=int(meta["n_tasks"]),
            global_dim=int(view.spec.global_dim),
            bank_active=int(ref_info["parameters"]),
        )
        hidden_width = int(sel["selected_H"])
        info["hidden_width_selection"] = sel
        del ref_model
    model = SRENNet(
        base,
        variant=arm,
        K=int(meta["K"]),
        n_tasks=int(meta["n_tasks"]),
        global_dim=int(view.spec.global_dim),
        hidden_width=hidden_width,
        zero_task_heads=zero_task_heads,
    )
    info["parameters"] = model.active_parameters()
    info["module_breakdown"] = model.module_breakdown()
    info["new_parameters"] = int(model.active_parameters() - int(base_audit["base_parameters"]))
    info["hidden_width"] = hidden_width
    info["deleted_base_heads"] = ["f", "rho", "readout"]
    info["unused_base_head_parameters_remaining"] = 0
    info["shared_parameter_fingerprint"] = C.state_fingerprint(
        {k: v for k, v in model.state_dict().items() if k.startswith(("U", "V", "base."))}
    )
    model.to(torch.device(device))
    info["state_fingerprint"] = C.state_fingerprint(model.state_dict())
    return model, info


def count_module_calls(model: nn.Module, batch: Mapping[str, Any]) -> dict[str, int]:
    counters = {"encode": 0, "local": 0, "U": 0, "V": 0}
    handles = []
    targets = [("encode", getattr(getattr(model, "base", model), "atom_encoder")), ("local", getattr(model, "base", model).local)]
    if hasattr(model, "base"):
        targets += [("U", model.U), ("V", model.V)]
    for name, module in targets:
        def _hook(_m, _inp, _out, key=name):
            counters[key] += 1

        handles.append(module.register_forward_hook(_hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return counters


__all__ = [name for name in dir() if not name.startswith("_")]
