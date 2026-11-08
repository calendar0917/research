"""LCRC-RC-REAL / LCRC-RC-BLIND: W2 + a graph-level relation-combination readout.

Both arms are the *same* class with one switch: how the per-distance-bucket
role co-occurrence matrix is formed.

    REAL   F_d = sqrt(w_d) * T_d            (trained role<->distance binding)
    BLIND  F_d = sqrt(w_d) * T_all if N_d>0 else 0   (binding destroyed)

Hard constraints, audited rather than asserted:

* exactly one ``atom_encoder``, ``bond_encoder``, ``local`` and ``drop`` call
  per forward: ``h`` is produced once by the frozen W2 local block and is never
  updated afterwards (forward hooks, ``count_module_calls_typed``);
* the branch reads only ``h`` (through the role softmax) and the static
  distance buckets, and returns one vector per *molecule*.  No atom state is
  written, no adjacency is propagated, there is no attention over atoms;
* the combiner's output layer is zero-initialised, so at step 0 the RC logits
  are exactly the frozen W2 logits;
* every original W2 tensor is bitwise equal to the historical seed-0 initial
  state, and the new tensors come from a private generator that never advances
  the global torch RNG;
* REAL and BLIND have identical new tensor names, shapes and counts.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import torch
import torch.nn as nn

import rc_common as C  # noqa: E402  (sets up the inherited code dirs)
import bs_model as BSM  # noqa: E402
import rc_relations as REL  # noqa: E402


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------


class Combiner(nn.Module):
    """``Linear(96,32)(Dropout(SiLU(Linear(487,96)(LayerNorm(x)))))``."""

    def __init__(self, x_dim: int = C.X_DIM, hidden: int = C.HIDDEN, z_dim: int = C.Z_DIM,
                 dropout: float = C.BRANCH_DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(int(x_dim))
        self.fc1 = nn.Linear(int(x_dim), int(hidden))
        self.act = nn.SiLU()
        self.drop = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden), int(z_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


class RelationBranch(nn.Module):
    """Role projection + relation statistics + task composition."""

    def __init__(self, width: int, n_out: int, kind: str):
        super().__init__()
        if kind not in ("real", "blind"):
            raise KeyError(kind)
        self.kind = str(kind)
        self.role_proj = nn.Linear(int(width), C.N_ROLES)
        self.combiner = Combiner()
        self.task_embedding = nn.Parameter(torch.empty(int(n_out), C.Z_DIM))
        self.task_bias = nn.Parameter(torch.zeros(int(n_out)))
        self.n_out = int(n_out)

    def roles(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.role_proj(h), dim=-1)

    def relation_input(self, batch: Mapping[str, Any], R: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        geom = REL.pair_geometry(batch)
        return REL.relation_input(
            R,
            geom["u"],
            geom["v"],
            geom["pair_graph"],
            geom["bucket"],
            batch["atom_batch"].to(torch.int64),
            batch["atom_counts"].to(torch.int64),
            int(batch["n_graphs"]),
            kind=self.kind,
            return_matrices=True,
        )

    def forward(self, batch: Mapping[str, Any], h: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        R = self.roles(h)
        x, aux = self.relation_input(batch, R)
        z = self.combiner(x)
        residual = z @ self.task_embedding.t() + self.task_bias
        size = int(batch["n_graphs"])
        if tuple(residual.shape) != (size, self.n_out):
            raise RuntimeError(f"residual logits {tuple(residual.shape)} != {(size, self.n_out)}")
        aux = dict(aux)
        aux.update(
            {
                "R": R,
                "z": z,
                "residual_rms": float(residual.detach().pow(2).mean().sqrt().item()),
                "z_rms": float(z.detach().pow(2).mean().sqrt().item()),
            }
        )
        return residual, aux


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------


class RCModel(BSM.BSModel):
    """Frozen W2 trunk + one graph-level relation-combination residual."""

    def __init__(self, spec, norm, n_out: int, cfg: Mapping[str, Any], *, kind: str):
        super().__init__(spec, norm, n_out, cfg)
        self.kind = str(kind)
        self.width = int(self.cfg["width"])
        self.branch = RelationBranch(self.width, int(n_out), self.kind)

    def forward(
        self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False, residual_zero: bool = False
    ):
        _a, _b, _scal, h = self.encode_h(batch)  # one atom encode, one bond encode, one local, one drop
        base = self.post_local(h, batch)
        if residual_zero:
            if return_aux:
                return base, {"base_logits": base, "residual_logits": torch.zeros_like(base), "h": h}
            return base
        residual, aux = self.branch(batch, h)
        logits = base + residual
        if return_aux:
            out = dict(aux)
            out.update({"base_logits": base, "residual_logits": residual, "h": h})
            return logits, out
        return logits


# ---------------------------------------------------------------------------
# initialisation: historical W2 bitwise + private generator for the new tensors
# ---------------------------------------------------------------------------


def _standard_linear_init(linear: nn.Linear, generator: torch.Generator) -> None:
    nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    if linear.bias is not None:
        bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
        linear.bias.uniform_(-bound, bound, generator=generator)


def new_keys(model: nn.Module, base_keys: set[str]) -> list[str]:
    return sorted(k for k in model.state_dict() if k not in base_keys)


def initialize_new_modules(model: RCModel, key: str, *, seed: int) -> dict[str, Any]:
    """Initialise only the new tensors; never advances the global torch RNG."""
    if key not in C.MODELS:
        raise KeyError(key)
    before = torch.get_rng_state().clone()
    rng_before = C.rng_state_hashes(C.capture_rng_state())
    gen = torch.Generator(device="cpu").manual_seed(
        int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSETS[key])
    )
    branch = model.branch
    with torch.no_grad():
        _standard_linear_init(branch.role_proj, gen)
        _standard_linear_init(branch.combiner.fc1, gen)
        branch.combiner.fc2.weight.zero_()
        branch.combiner.fc2.bias.zero_()
        branch.task_embedding.normal_(0.0, 1.0 / math.sqrt(float(C.Z_DIM)), generator=gen)
        branch.task_bias.zero_()
        norm_weight_ok = bool(torch.all(branch.combiner.norm.weight == 1).item())
        norm_bias_ok = bool(torch.count_nonzero(branch.combiner.norm.bias).item() == 0)
    if not torch.equal(before, torch.get_rng_state()):
        raise RuntimeError(f"{key}: new-module initialisation changed the global torch RNG state")
    if C.rng_state_hashes(C.capture_rng_state()) != rng_before:
        raise RuntimeError(f"{key}: new-module initialisation changed a tracked RNG state")
    audit = {
        "kind": "private_generator",
        "private_generator_seed": int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSETS[key]),
        "role_proj": "kaiming_uniform(a=sqrt(5)) + fan_in bias",
        "combiner.fc1": "kaiming_uniform(a=sqrt(5)) + fan_in bias",
        "combiner.fc2": "weight and bias zeroed (residual logits == 0 at step 0)",
        "layer_norm": "default (weight=1, bias=0, no RNG)",
        "task_embedding": "normal(0, 1/sqrt(32))",
        "task_bias": "zeros",
        "fc2_is_zero": bool(torch.count_nonzero(branch.combiner.fc2.weight).item() == 0)
        and bool(torch.count_nonzero(branch.combiner.fc2.bias).item() == 0),
        "layernorm_default": bool(norm_weight_ok and norm_bias_ok),
        "global_rng_unchanged": True,
    }
    return audit


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("utf-8"))
        h.update(str(tuple(value.shape)).encode("utf-8"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def build_model(
    key: str,
    spec,
    norm: Mapping[str, Any],
    n_out: int = C.N_TASKS,
    *,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    base_model: BSM.BSModel | None = None,
    verify_parameters: bool = True,
) -> tuple[RCModel, dict[str, Any]]:
    """Build one RC arm from the frozen, *untrained* historical W2 seed-0 state."""
    if key not in C.MODELS:
        raise KeyError(key)
    if base_model is None:
        base_model, base_audit = BSM.build_model(
            C.BASE_MODEL_KEY, spec, norm, int(n_out), seed=int(seed), device="cpu"
        )
    else:  # pragma: no cover - reuse path for the two-arm audit
        base_audit = {}
    cfg = dict(base_model.cfg)
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    base_keys = set(base_state)
    global_rng_before = torch.get_rng_state().clone()

    with torch.random.fork_rng(devices=[]):
        model = RCModel(spec, norm, int(n_out), cfg, kind=str(C.MODELS[key]["kind"]))

    added = new_keys(model, base_keys)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if set(missing) != set(added) or unexpected:
        raise RuntimeError(
            f"{key}: base state load mismatch: missing={sorted(missing)} "
            f"expected_new={sorted(added)} unexpected={sorted(unexpected)}"
        )
    init_audit = initialize_new_modules(model, key, seed=int(seed))

    subset_diff: dict[str, float] = {}
    state = model.state_dict()
    for k in sorted(base_keys):
        if tuple(state[k].shape) != tuple(base_state[k].shape) or state[k].dtype != base_state[k].dtype:
            raise RuntimeError(f"{key}: original W2 tensor shape/dtype changed: {k}")
        if not torch.equal(state[k].cpu(), base_state[k].cpu()):
            subset_diff[k] = float((state[k].float().cpu() - base_state[k].float().cpu()).abs().max().item())
    if subset_diff:
        raise RuntimeError(f"{key}: original W2 parameters are not bitwise equal: {subset_diff}")
    if not torch.equal(global_rng_before, torch.get_rng_state()):
        raise RuntimeError(f"{key}: build advanced the global torch RNG state")

    model.to(torch.device(device))
    base_elements = int(sum(p.numel() for n, p in model.named_parameters() if n in base_keys))
    new_elements = int(sum(p.numel() for n, p in model.named_parameters() if n not in base_keys))
    total = int(C.parameter_count(model))
    inventory = {k: {"shape": list(state[k].shape), "numel": int(state[k].numel())} for k in added}
    audit: dict[str, Any] = {
        "model": key,
        "model_name": C.MODELS[key]["name"],
        "role": C.MODELS[key]["role"],
        "branch_kind": str(C.MODELS[key]["kind"]),
        "seed": int(seed),
        "base_model": C.BASE_MODEL_KEY,
        "base_parameters": base_elements,
        "new_parameters": new_elements,
        "total_parameters": total,
        "expected_base_parameters": int(C.BASE_PARAMETERS),
        "expected_new_parameters": int(C.BRANCH_ADDED),
        "expected_total_parameters": int(C.EXPECTED_PARAMETERS),
        "parameter_ceiling": int(C.PARAMETER_CEILING),
        "widths": cfg,
        "branch_definition": {
            "n_roles": C.N_ROLES,
            "n_buckets": C.N_BUCKETS,
            "x_dim": C.X_DIM,
            "hidden": C.HIDDEN,
            "z_dim": C.Z_DIM,
            "dropout": C.BRANCH_DROPOUT,
            "triu_entries": C.TRIU,
            "distance_buckets": [list(b) for b in C.BUCKETS],
            "aggregation": "one [B, A] @ [A, 7*144+12] matmul per graph batch",
            "iterative_message_passing": False,
            "atom_state_writes": 0,
        },
        "init_rule": C.INIT_RULE,
        "new_module_init": init_audit,
        "new_state_keys": added,
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
    if verify_parameters:
        if new_elements != int(C.BRANCH_ADDED):
            raise RuntimeError(f"{key}: new parameters {new_elements} != {C.BRANCH_ADDED}")
        if total != int(C.EXPECTED_PARAMETERS):
            raise RuntimeError(f"{key}: total parameters {total} != {C.EXPECTED_PARAMETERS}")
        if base_elements != int(C.BASE_PARAMETERS):
            raise RuntimeError(f"{key}: base parameters {base_elements} != {C.BASE_PARAMETERS}")
        if total > int(C.PARAMETER_CEILING):
            raise RuntimeError(f"{key}: {total} parameters exceed the {C.PARAMETER_CEILING} ceiling")
        frozen_norm = norm.get("norm_fingerprint") == C.SOURCE["norm_fingerprint"]
        if frozen_norm:
            if audit["base_state_sha256"] != C.W2["base_state_sha256_at_init"]:
                raise RuntimeError(
                    f"{key}: W2 init state {audit['base_state_sha256']} != historical "
                    f"{C.W2['base_state_sha256_at_init']}"
                )
            if audit["base_trunk_init_fingerprint"] != C.W2["trunk_init_fingerprint"]:
                raise RuntimeError(f"{key}: trunk init fingerprint is not the historical seed-0 one")
            audit["historical_init_check"] = "exact"
        else:
            audit["historical_init_check"] = "skipped_non_frozen_norm_fingerprint"
        audit["parameter_check"] = "exact"
    return model, audit


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def _stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        "rms": float(x.pow(2).mean().sqrt().item()) if x.numel() else 0.0,
        "mean_abs": float(x.abs().mean().item()) if x.numel() else 0.0,
        "max_abs": float(x.abs().max().item()) if x.numel() else 0.0,
        "l2": float(x.norm().item()) if x.numel() else 0.0,
        "numel": int(x.numel()),
    }


@torch.no_grad()
def branch_diagnostics(model: RCModel, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Residual strength / role statistics on a fixed batch (eval mode)."""
    was_training = model.training
    model.eval()
    try:
        logits, aux = model(batch, return_aux=True)
        residual = aux["residual_logits"]
        base = aux["base_logits"]
        rel = {
            k: (float(v) if isinstance(v, (int, float)) else None)
            for k, v in aux.items()
            if isinstance(v, (int, float))
        }
        w = aux["w"]
        counts = aux["ordered_counts"]
        T_all = aux["T_all"]
        return {
            "kind": model.kind,
            "residual": _stats(residual),
            "base_logits": _stats(base),
            "logits": _stats(logits),
            "h": _stats(aux["h"]),
            "residual_over_base": float(
                residual.pow(2).mean().sqrt().item() / max(base.pow(2).mean().sqrt().item(), 1e-12)
            ),
            "nonzero_fraction": float((residual != 0).float().mean().item()),
            "w_mean": [float(v) for v in w.mean(dim=0).tolist()],
            "w_min": [float(v) for v in w.min(dim=0).values.tolist()],
            "w_max": [float(v) for v in w.max(dim=0).values.tolist()],
            "graphs_with_empty_bucket": [int((counts[:, d] == 0).sum().item()) for d in range(C.N_BUCKETS)],
            "T_all_mean_abs": float(T_all.abs().mean().item()),
            "T_all_frobenius_mean": float(T_all.flatten(1).norm(dim=1).mean().item()),
            "z_rms": float(aux["z"].pow(2).mean().sqrt().item()),
            "relation_aux": rel,
        }
    finally:
        model.train(was_training)


@torch.no_grad()
def module_norms(model: RCModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if name.startswith("branch."):
            out[name] = {
                "l2": float(param.detach().float().norm().item()),
                "rms": float(param.detach().float().pow(2).mean().sqrt().item()) if param.numel() else 0.0,
                "numel": int(param.numel()),
            }
    return out


@torch.no_grad()
def branch_grad_norm(model: RCModel) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in model.named_parameters():
        if name.startswith("branch.") and param.grad is not None:
            out[name] = float(param.grad.detach().float().norm().item())
    return out


def count_module_calls_typed(model: RCModel, batch: Mapping[str, torch.Tensor]) -> dict[str, int]:
    names = ("atom_encoder", "bond_encoder", "local", "drop", "f", "rho", "readout", "branch")
    counters = {n: 0 for n in names}
    handles = []
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue

        def _hook(_m, _inp, _out, key=name):
            counters[key] += 1

        handles.append(module.register_forward_hook(_hook))
    try:
        model.eval()
        with torch.no_grad():
            model(batch)
    finally:
        for h in handles:
            h.remove()
    return counters


__all__ = [
    "Combiner",
    "RCModel",
    "RelationBranch",
    "branch_diagnostics",
    "branch_grad_norm",
    "build_model",
    "count_module_calls_typed",
    "initialize_new_modules",
    "module_norms",
    "new_keys",
    "state_sha256",
]
