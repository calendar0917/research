"""W2 + Cross-atom (P) / Same-atom (S) Predictive Evidence.

Both arms are the *historical* W2 graph exactly (`bs_model.BSModel`, imported
read-only) plus one new module that reads the post-``local``, post-dropout atom
state ``h`` **once**:

    a, b, scal, h = model.encode_h(batch)          # 1 atom enc, 1 bond enc, 1 local, 1 dropout
    base_logits   = model.post_local(h, batch)     # historical f + rho + 4-head readout
    logits        = base_logits + evidence_residual(h, batch)

The evidence module never re-encodes an atom, never calls the original dropout
again and never propagates an updated atom state along a bond.  Its only
cross-atom operations are segment reductions (``index_add_``).

Definitions (for every molecule G with n atoms and every assay t):

    evidence_A = tanh(linear_A(h))            [n_atoms, 128]
    evidence_B = tanh(linear_B(h))            [n_atoms, 128]

    P_t(G) = ( (sum_i a_it)(sum_j b_jt) - sum_i a_it b_it ) / (n(n-1))   # i != j
    S_t(G) = ( sum_i a_it b_it ) / n                                     # i == j

    residual = alpha_A * mean_A + alpha_B * mean_B + alpha_pair * selected

``pair_cross`` is exactly zero for ``n == 1`` (masked, never divided by zero) and
the code reports how many such graphs it saw.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import torch
import torch.nn as nn

import cpe_common as C
import bs_model as BSM


# ---------------------------------------------------------------------------
# the pure second-order reduction (no parameters, no state)
# ---------------------------------------------------------------------------


def segment_count(atom_batch: torch.Tensor, n_graphs: int, dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros(int(n_graphs), dtype=dtype, device=atom_batch.device)
    return out.index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=dtype))


def pair_statistics(
    evidence_A: torch.Tensor,
    evidence_B: torch.Tensor,
    atom_batch: torch.Tensor,
    n_graphs: int,
) -> dict[str, Any]:
    """Per-molecule, per-task first- and second-order evidence statistics.

    ``evidence_A`` / ``evidence_B`` are ``[n_atoms, n_tasks]``.  Everything is a
    sum / mean over atoms of the *same* molecule; no atom pair is enumerated and
    nothing crosses a molecule boundary.
    """
    if evidence_A.shape != evidence_B.shape or evidence_A.dim() != 2:
        raise ValueError(f"evidence shapes {tuple(evidence_A.shape)} {tuple(evidence_B.shape)}")
    n_graphs = int(n_graphs)
    d = evidence_A.dtype
    n_atoms = int(evidence_A.shape[0])
    n_tasks = int(evidence_A.shape[1])

    sum_A = torch.zeros((n_graphs, n_tasks), dtype=d, device=evidence_A.device).index_add_(0, atom_batch, evidence_A)
    sum_B = torch.zeros((n_graphs, n_tasks), dtype=d, device=evidence_A.device).index_add_(0, atom_batch, evidence_B)
    sum_AB = torch.zeros((n_graphs, n_tasks), dtype=d, device=evidence_A.device).index_add_(0, atom_batch, evidence_A * evidence_B)
    n = segment_count(atom_batch, n_graphs, d)

    n_safe = n.clamp(min=1.0).unsqueeze(1)
    mean_A = sum_A / n_safe
    mean_B = sum_B / n_safe
    pair_self = sum_AB / n_safe

    denom = n * (n - 1.0)
    valid_pair = denom > 0
    denom_safe = torch.where(valid_pair, denom, torch.ones_like(denom)).unsqueeze(1)
    pair_cross = torch.where(
        valid_pair.unsqueeze(1),
        (sum_A * sum_B - sum_AB) / denom_safe,
        torch.zeros_like(sum_AB),
    )

    return {
        "n_atoms": n_atoms,
        "n_graphs": n_graphs,
        "n_tasks": n_tasks,
        "n": n,
        "sum_A": sum_A,
        "sum_B": sum_B,
        "sum_AB": sum_AB,
        "mean_A": mean_A,
        "mean_B": mean_B,
        "pair_cross": pair_cross,
        "pair_self": pair_self,
        "n_single_atom_graphs": int((n == 1).sum().item()),
        "n_empty_graphs": int((n == 0).sum().item()),
        "n_pair_defined_graphs": int(valid_pair.sum().item()),
    }


# ---------------------------------------------------------------------------
# the new module
# ---------------------------------------------------------------------------


class PredictiveEvidence(nn.Module):
    """Task-specific evidence detectors + three per-task coefficients.

    ``pair_mode`` is a plain attribute (``"cross"`` or ``"self"``), never a
    parameter, so the two arms have identical parameters and differ only in
    which already-computed statistic is selected.
    """

    def __init__(self, width: int = C.EVIDENCE_WIDTH, n_tasks: int = C.N_TASKS, *, pair_mode: str = "cross"):
        super().__init__()
        if pair_mode not in ("cross", "self"):
            raise KeyError(f"pair_mode {pair_mode!r}")
        self.width = int(width)
        self.n_tasks = int(n_tasks)
        self.pair_mode = str(pair_mode)
        self.linear_A = nn.Linear(self.width, self.n_tasks, bias=True)
        self.linear_B = nn.Linear(self.width, self.n_tasks, bias=True)
        self.alpha_A = nn.Parameter(torch.zeros(self.n_tasks))
        self.alpha_B = nn.Parameter(torch.zeros(self.n_tasks))
        self.alpha_pair = nn.Parameter(torch.zeros(self.n_tasks))

    # -- detectors ---------------------------------------------------------

    def detect(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tanh(self.linear_A(h)), torch.tanh(self.linear_B(h))

    # -- residual ----------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
        *,
        return_stats: bool = False,
    ):
        evidence_A, evidence_B = self.detect(h)
        stats = pair_statistics(evidence_A, evidence_B, atom_batch, int(n_graphs))
        selected = stats["pair_cross"] if self.pair_mode == "cross" else stats["pair_self"]
        main = self.alpha_A * stats["mean_A"] + self.alpha_B * stats["mean_B"]
        pair = self.alpha_pair * selected
        residual = main + pair
        if not return_stats:
            return residual
        stats_out = {
            "pair_mode": self.pair_mode,
            "n_atoms": stats["n_atoms"],
            "n_graphs": stats["n_graphs"],
            "n_single_atom_graphs": stats["n_single_atom_graphs"],
            "n_empty_graphs": stats["n_empty_graphs"],
            "n_pair_defined_graphs": stats["n_pair_defined_graphs"],
            "evidence_A": evidence_A,
            "evidence_B": evidence_B,
            "mean_A": stats["mean_A"],
            "mean_B": stats["mean_B"],
            "pair_cross": stats["pair_cross"],
            "pair_self": stats["pair_self"],
            "selected_pair_statistic": selected,
            "main_effect_residual": main,
            "pair_residual": pair,
            "residual": residual,
        }
        return residual, stats_out


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class CPEModel(BSM.BSModel):
    """Historical W2 graph + one predictive-evidence residual."""

    def __init__(self, spec, norm, n_out: int, cfg: Mapping[str, Any], *, pair_mode: str):
        super().__init__(spec, norm, n_out, cfg)
        if int(cfg["width"]) != C.EVIDENCE_WIDTH:
            raise RuntimeError(f"local width {cfg['width']} != evidence width {C.EVIDENCE_WIDTH}")
        if int(n_out) != C.N_TASKS:
            raise RuntimeError(f"n_out {n_out} != {C.N_TASKS}")
        self.pair_mode = str(pair_mode)
        self.evidence = PredictiveEvidence(int(cfg["width"]), int(n_out), pair_mode=str(pair_mode))

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        if return_aux:
            _a, _b, _scal, h = self.encode_h(batch)
            base_logits, aux = self.post_local(h, batch, return_aux=True, return_attention=True)
        else:
            _a, _b, _scal, h = self.encode_h(batch)
            base_logits = self.post_local(h, batch)
            aux = None
        residual, stats = self.evidence(
            h, batch["atom_batch"], int(batch["n_graphs"]), return_stats=True
        )
        logits = base_logits + residual
        if not return_aux:
            return logits
        out = dict(aux)
        out.update(
            {
                "base_logits": base_logits,
                "evidence_residual": residual,
                "evidence": stats,
                "logits": logits,
            }
        )
        return logits, out

    # -- diagnostics -------------------------------------------------------

    def evidence_diagnostics(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """One extra eval-mode forward on a fixed batch; no RNG consumed."""
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                _logits, aux = self.forward(batch, return_aux=True)
        finally:
            self.train(was_training)
        ev = aux["evidence"]
        base = aux["base_logits"].detach().float()
        main = ev["main_effect_residual"].detach().float()
        pair = ev["pair_residual"].detach().float()
        resid = ev["residual"].detach().float()
        cross = ev["pair_cross"].detach().float()
        self_ = ev["pair_self"].detach().float()
        alpha_pair = self.evidence.alpha_pair.detach().float()
        selected = cross if self.pair_mode == "cross" else self_
        return {
            "pair_mode": self.pair_mode,
            "n_atoms": int(ev["n_atoms"]),
            "n_graphs": int(ev["n_graphs"]),
            "n_single_atom_graphs": int(ev["n_single_atom_graphs"]),
            "n_empty_graphs": int(ev["n_empty_graphs"]),
            "alpha": {
                "A_l2": float(self.evidence.alpha_A.detach().float().norm().item()),
                "B_l2": float(self.evidence.alpha_B.detach().float().norm().item()),
                "pair_l2": float(alpha_pair.norm().item()),
                "A_abs_mean": float(self.evidence.alpha_A.detach().float().abs().mean().item()),
                "B_abs_mean": float(self.evidence.alpha_B.detach().float().abs().mean().item()),
                "pair_abs_mean": float(alpha_pair.abs().mean().item()),
                "pair_abs_max": float(alpha_pair.abs().max().item()),
                "pair_non_negligible_fraction": float(
                    (alpha_pair.abs() >= float(C.PAIR_COEFFICIENT_NEGLIGIBLE_THRESHOLD)).float().mean().item()
                ),
                "pair_non_negligible_count": int(
                    (alpha_pair.abs() >= float(C.PAIR_COEFFICIENT_NEGLIGIBLE_THRESHOLD)).sum().item()
                ),
                "threshold": float(C.PAIR_COEFFICIENT_NEGLIGIBLE_THRESHOLD),
            },
            "linear": {
                "A_l2": float(self.evidence.linear_A.weight.detach().float().norm().item()),
                "B_l2": float(self.evidence.linear_B.weight.detach().float().norm().item()),
            },
            "magnitudes": {
                "base_logits_abs_mean": float(base.abs().mean().item()),
                "main_effect_residual_abs_mean": float(main.abs().mean().item()),
                "pair_residual_abs_mean": float(pair.abs().mean().item()),
                "residual_abs_mean": float(resid.abs().mean().item()),
                "pair_residual_abs_max": float(pair.abs().max().item()),
                "residual_abs_max": float(resid.abs().max().item()),
                "pair_over_residual": float((pair.abs().sum() / resid.abs().sum().clamp(min=1e-12)).item()),
            },
            "statistics": {
                "pair_cross_abs_mean": float(cross.abs().mean().item()),
                "pair_cross_std": float(cross.std().item()) if cross.numel() > 1 else 0.0,
                "pair_self_abs_mean": float(self_.abs().mean().item()),
                "pair_self_std": float(self_.std().item()) if self_.numel() > 1 else 0.0,
                "selected_abs_mean": float(selected.abs().mean().item()),
                "selected_std": float(selected.std().item()) if selected.numel() > 1 else 0.0,
                "selected_abs_max": float(selected.abs().max().item()),
                "main_effect_term_abs_mean": float(
                    (self.evidence.alpha_A.detach().float() * ev["mean_A"].detach().float()).abs().mean().item()
                ),
                "evidence_A_abs_mean": float(ev["evidence_A"].detach().float().abs().mean().item()),
                "evidence_B_abs_mean": float(ev["evidence_B"].detach().float().abs().mean().item()),
                "evidence_A_tanh_saturated_fraction": float(
                    (ev["evidence_A"].detach().float().abs() > 0.999).float().mean().item()
                ),
                "evidence_B_tanh_saturated_fraction": float(
                    (ev["evidence_B"].detach().float().abs() > 0.999).float().mean().item()
                ),
            },
            "finite": bool(torch.isfinite(resid).all().item() and torch.isfinite(base).all().item()),
        }


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


def _standard_linear_init(linear: nn.Linear, generator: torch.Generator) -> None:
    """``nn.Linear.reset_parameters`` with an explicit private generator."""
    nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    if linear.bias is not None:
        bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
        linear.bias.uniform_(-bound, bound, generator=generator)


def initialize_new_modules(model: CPEModel, *, generator_seed: int = C.INIT_GENERATOR_SEED) -> dict[str, Any]:
    """Standard Linear init from a private CPU generator; alphas exactly zero."""
    before = torch.get_rng_state().clone()
    rng_before = C.rng_state_hashes(C.capture_rng_state())
    gen = torch.Generator(device="cpu").manual_seed(int(generator_seed))
    with torch.no_grad():
        _standard_linear_init(model.evidence.linear_A, gen)
        _standard_linear_init(model.evidence.linear_B, gen)
        model.evidence.alpha_A.zero_()
        model.evidence.alpha_B.zero_()
        model.evidence.alpha_pair.zero_()
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("new-module initialisation changed the global torch RNG state")
    if C.rng_state_hashes(C.capture_rng_state()) != rng_before:
        raise RuntimeError("new-module initialisation changed a tracked RNG state")
    return {
        "kind": "standard_torch_linear_uniform_from_private_generator",
        "private_generator_seed": int(generator_seed),
        "call_order": ["linear_A.weight", "linear_A.bias", "linear_B.weight", "linear_B.bias", "alpha_* = 0"],
        "linear_A_shape": list(model.evidence.linear_A.weight.shape),
        "linear_B_shape": list(model.evidence.linear_B.weight.shape),
        "alphas_zero": bool(
            torch.count_nonzero(model.evidence.alpha_A).item() == 0
            and torch.count_nonzero(model.evidence.alpha_B).item() == 0
            and torch.count_nonzero(model.evidence.alpha_pair).item() == 0
        ),
        "global_rng_unchanged": True,
    }


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over the sorted (key, dtype, shape, bytes) state."""
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("utf-8"))
        h.update(str(tuple(value.shape)).encode("utf-8"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def build_model(
    key: str,
    spec,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    base_model: BSM.BSModel | None = None,
    verify_parameters: bool = True,
) -> tuple[CPEModel, dict[str, Any]]:
    """Build P/S from the frozen untrained W2 seed-0 state."""
    if key not in C.MODELS:
        raise KeyError(key)
    if base_model is None:
        base_model, base_audit = BSM.build_model(
            C.BASE_MODEL_KEY, spec, norm, int(n_out), seed=int(seed), device="cpu"
        )
    else:
        base_audit = {}
    cfg = dict(base_model.cfg)
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    base_keys = set(base_state)
    global_rng_before = torch.get_rng_state().clone()
    rng_before = C.rng_state_hashes(C.capture_rng_state())

    with torch.random.fork_rng(devices=[]):
        model = CPEModel(spec, norm, int(n_out), cfg, pair_mode=C.PAIR_MODES[key])

    new_keys = sorted(k for k in model.state_dict() if k not in base_keys)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if set(missing) != set(new_keys) or unexpected:
        raise RuntimeError(
            f"{key}: base state load mismatch: missing={sorted(missing)} "
            f"expected_new={new_keys} unexpected={sorted(unexpected)}"
        )

    init_audit = initialize_new_modules(model, generator_seed=C.INIT_GENERATOR_SEED)

    state = model.state_dict()
    subset_diff: dict[str, float] = {}
    for k in sorted(base_keys):
        if tuple(state[k].shape) != tuple(base_state[k].shape) or state[k].dtype != base_state[k].dtype:
            raise RuntimeError(f"{key}: original W2 tensor shape/dtype changed: {k}")
        if not torch.equal(state[k].cpu(), base_state[k].cpu()):
            subset_diff[k] = float((state[k].float().cpu() - base_state[k].float().cpu()).abs().max().item())
    if subset_diff:
        raise RuntimeError(f"{key}: original W2 parameters are not bitwise equal: {subset_diff}")

    if not torch.equal(global_rng_before, torch.get_rng_state()):
        raise RuntimeError(f"{key}: build advanced the global torch RNG state")
    if C.rng_state_hashes(C.capture_rng_state()) != rng_before:
        raise RuntimeError(f"{key}: build advanced a tracked RNG state")

    model.to(torch.device(device))

    base_elements = int(sum(p.numel() for n, p in model.named_parameters() if n in base_keys))
    new_elements = int(sum(p.numel() for n, p in model.named_parameters() if n not in base_keys))
    total = int(C.parameter_count(model))
    inventory = {k: {"shape": list(state[k].shape), "numel": int(state[k].numel())} for k in new_keys}
    audit: dict[str, Any] = {
        "model": key,
        "model_name": C.MODELS[key]["name"],
        "role": C.MODELS[key]["role"],
        "pair_mode": C.PAIR_MODES[key],
        "seed": int(seed),
        "base_model": C.BASE_MODEL_KEY,
        "base_parameters": base_elements,
        "new_parameters": new_elements,
        "total_parameters": total,
        "expected_base_parameters": int(C.BASE_PARAMETERS),
        "expected_new_parameters": int(C.NEW_PARAMETERS),
        "expected_total_parameters": int(C.EXPECTED_TOTALS[key]),
        "widths": cfg,
        "init_rule": C.INIT_RULE,
        "new_module_init": init_audit,
        "new_state_keys": new_keys,
        "new_parameter_inventory": inventory,
        "original_w2_subset": {"n_tensors": len(base_keys), "all_bitwise_equal": True, "n_differing": 0},
        "base_trunk_init_fingerprint": base_audit.get("trunk_init_fingerprint"),
        "base_full_trunk_init_fingerprint": base_audit.get("full_trunk_init_fingerprint"),
        "base_state_sha256": state_sha256(base_state),
        "model_state_sha256": state_sha256(state),
        "state_keys": sorted(state),
        "n_state_tensors": len(state),
        "parameter_tensors": len(list(model.parameters())),
        "base_model_audit": base_audit,
        "global_rng_unchanged": True,
    }
    audit["parameter_check"] = "exact"
    if verify_parameters:
        if new_elements != int(C.NEW_PARAMETERS):
            raise RuntimeError(f"{key}: new parameters {new_elements} != {C.NEW_PARAMETERS}")
        if total != int(C.EXPECTED_TOTALS[key]):
            raise RuntimeError(f"{key}: total parameters {total} != {C.EXPECTED_TOTALS[key]}")
        if base_elements != int(C.BASE_PARAMETERS):
            raise RuntimeError(f"{key}: base parameters {base_elements} != {C.BASE_PARAMETERS}")
        if int(C.EXPECTED_TOTALS[key]) != 1_027_732:
            raise RuntimeError(f"{key}: expected total is not the frozen 1,027,732")
    return model, audit


# ---------------------------------------------------------------------------
# audit / diagnostic helpers
# ---------------------------------------------------------------------------


def count_module_calls(model: CPEModel, batch: Mapping[str, Any], *, training: bool = False) -> dict[str, int]:
    """Count how often each module runs in one forward (including dropout)."""
    counters = {
        "atom_encoder": 0,
        "bond_encoder": 0,
        "local": 0,
        "f": 0,
        "rho": 0,
        "readout": 0,
        "dropout": 0,
        "evidence": 0,
        "evidence_linear_A": 0,
        "evidence_linear_B": 0,
    }
    handles = []
    for name, module in (
        ("atom_encoder", model.atom_encoder),
        ("bond_encoder", model.bond_encoder),
        ("local", model.local),
        ("f", model.f),
        ("rho", model.rho),
        ("readout", model.readout),
        ("dropout", model.drop),
        ("evidence", model.evidence),
        ("evidence_linear_A", model.evidence.linear_A),
        ("evidence_linear_B", model.evidence.linear_B),
    ):
        def _hook(_m, _inp, _out, key=name):
            counters[key] += 1

        handles.append(module.register_forward_hook(_hook))
    was_training = model.training
    try:
        model.train(bool(training))
        with torch.no_grad():
            model(batch)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return counters


def evidence_parameter_names(model: CPEModel) -> list[str]:
    return [n for n in dict(model.named_parameters()) if n.startswith("evidence.")]


def logits_without_pair(model: CPEModel, batch: Mapping[str, Any]) -> torch.Tensor:
    """Diagnostic-only ablation: the same trained weights with ``alpha_pair = 0``."""
    saved = model.evidence.alpha_pair.detach().clone()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model.evidence.alpha_pair.zero_()
            out = model(batch).detach().float().cpu()
    finally:
        with torch.no_grad():
            model.evidence.alpha_pair.copy_(saved)
        model.train(was_training)
    return out


__all__ = [
    "CPEModel",
    "PredictiveEvidence",
    "build_model",
    "count_module_calls",
    "evidence_parameter_names",
    "initialize_new_modules",
    "logits_without_pair",
    "pair_statistics",
    "segment_count",
    "state_sha256",
]
