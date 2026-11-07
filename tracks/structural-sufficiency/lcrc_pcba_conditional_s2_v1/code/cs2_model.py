"""W2 + Conditional S2 (C2) and W2 + LocalResidualFFN (F).

Both arms start from the *same untrained historical W2 seed-0 state*.  The
original W2 computation, parameters and initialisation are inherited read-only
from ``bs_model``; the new modules are appended and initialised so that at step
0 the model output is exactly the original W2 output.

C2 changes only the S2 content aggregation inside ``TULocalContent``:

    g_p = g_scalar(c_p) + (E_c(c_p) * E_e(e_p)) V
    content_agg = index_add(path_atom, a[path_end] * g_p)

with ``c = path_cat`` (400 codes), ``e = path_end63`` (80 codes), rank 16 and
``V: (16, 192)``.  No path is added, removed or re-enumerated, no endpoint atom
is changed, and no updated atom state is propagated along an edge.

F adds one pre-norm residual FFN on the local per-atom representation, before
the original dropout:

    h = dropout(h + fc2(silu(fc1(LayerNorm(h)))))

which is a capacity control with no access to bonds, paths or other atoms.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import cs2_common as C
import bs_model as BSM
import tu_model as TU


# ---------------------------------------------------------------------------
# C2: conditional S2 vector gate
# ---------------------------------------------------------------------------


class CS2LocalContent(TU.TULocalContent):
    """The historical C1 local content block with a rank-16 vector path gate."""

    def __init__(self, spec, variant: str, cfg: Mapping[str, Any], *, cond_rank: int = C.COND_RANK):
        super().__init__(spec, variant, cfg)
        self.cond_rank = int(cond_rank)
        # The S2 content multiply acts on the atom-encoder output a
        # (atom_out = 192), not on the local MLP output width (128).
        self.gate_width = int(self.a_out)
        if self.gate_width != int(C.WIDTHS["atom_out"]):
            raise RuntimeError(f"unexpected atom_out for the vector gate: {self.gate_width}")
        if int(spec.n_path_codes) != 400 or int(spec.n_nbr_codes) != 80:
            raise RuntimeError(
                f"frozen path vocabulary drift: n_path_codes={spec.n_path_codes}, "
                f"n_nbr_codes={spec.n_nbr_codes}"
            )
        #: E_c / E_e use a private generator (see ``initialize_new_modules``).
        self.path_cond_ec = nn.Embedding(int(spec.n_path_codes), self.cond_rank)
        self.path_cond_ee = nn.Embedding(int(spec.n_nbr_codes), self.cond_rank)
        #: V is exactly zero at initialisation; the historical scalar gate stays.
        self.path_cond_v = nn.Parameter(torch.zeros(self.cond_rank, self.gate_width))

    def vector_gate(self, path_cat: torch.Tensor, path_end63: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(g_scalar + delta, delta)`` for every typed path."""
        delta = (self.path_cond_ec(path_cat) * self.path_cond_ee(path_end63)) @ self.path_cond_v
        return self.path_gate(path_cat) + delta, delta

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        scal: torch.Tensor,
    ) -> torch.Tensor:
        # This body is the historical ``TULocalContent.forward`` (tu_model.py)
        # with exactly one change: the scalar ``path_gate`` becomes the
        # conditional vector gate.  Every other line is unchanged.
        n_atoms = a.shape[0]
        width = a.shape[1]
        idx = batch["bond_index"]
        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            degree = a.new_zeros(n_atoms)
            degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
            degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
            b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
        else:
            degree = a.new_zeros(n_atoms)
            b_sum = a.new_zeros((n_atoms, width))

        nbr = (
            a.new_zeros((n_atoms, self.rank)).index_add_(0, batch["nbr_atom"], self.nbr_emb(batch["nbr_code"]))
            if batch["nbr_code"].numel()
            else a.new_zeros((n_atoms, self.rank))
        )

        has_paths = bool(batch["path_cat"].numel())
        path_dim = 3 * self.rank + 8
        if has_paths:
            path_joint = self.joint_proj(self.path_u(batch["path_cat"]) * self.path_v(batch["path_end63"]))
            path_emb = torch.cat(
                [
                    self.path_emb(batch["path_cat"]),
                    self.end_emb(batch["path_end63"]),
                    path_joint,
                    self.close_emb(batch["path_closure"]),
                ],
                dim=1,
            )
            path_agg = a.new_zeros((n_atoms, path_emb.shape[1])).index_add_(0, batch["path_atom"], path_emb)
            # ---- the only changed lines: scalar gate -> conditional vector gate
            vector_gate, _delta = self.vector_gate(batch["path_cat"], batch["path_end63"])
            content_agg = a.new_zeros((n_atoms, width)).index_add_(
                0, batch["path_atom"], a[batch["path_end"]] * vector_gate
            )
        else:
            path_agg = a.new_zeros((n_atoms, path_dim))
            content_agg = a.new_zeros((n_atoms, width))

        parts: list[torch.Tensor] = [a, b_sum]
        if self.use_b_mean:
            parts.append(b_sum / degree.clamp(min=1.0).unsqueeze(1))
        parts.extend([nbr, path_agg, content_agg, scal])
        x = torch.cat(parts, dim=1)
        if x.shape[1] != len(self.local_cols):
            raise RuntimeError(f"local width {x.shape[1]} != layout width {len(self.local_cols)}")
        return self.norm(self.mlp(x))


class CS2Model(BSM.BSModel):
    """W2 trunk + the conditional S2 local content block."""

    def __init__(self, spec, norm, n_out: int, cfg: Mapping[str, Any], *, cond_rank: int = C.COND_RANK):
        super().__init__(spec, norm, n_out, cfg)
        self.local = CS2LocalContent(spec, "C1", dict(cfg), cond_rank=int(cond_rank))


# ---------------------------------------------------------------------------
# F: near-parameter-matched local FFN control
# ---------------------------------------------------------------------------


class LocalResidualFFN(nn.Module):
    """Pre-norm residual FFN: 128 -> 40 -> 128, no inter-atom access."""

    def __init__(self, width: int = C.WIDTHS["width"], hidden: int = C.FFN_HIDDEN):
        super().__init__()
        self.norm = nn.LayerNorm(int(width))
        self.fc1 = nn.Linear(int(width), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(width))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(F.silu(self.fc1(self.norm(h))))


class FModel(BSM.BSModel):
    """W2 trunk + one local residual FFN inserted before the original dropout."""

    def __init__(self, spec, norm, n_out: int, cfg: Mapping[str, Any]):
        super().__init__(spec, norm, n_out, cfg)
        self.ffn = LocalResidualFFN(int(cfg["width"]), int(C.FFN_HIDDEN))

    def encode_h(self, batch: Mapping[str, torch.Tensor]):
        a, b, scal = self._encode(batch)
        # exactly one dropout call, at the historical position (after the
        # local block, before every graph-level readout)
        h = self.drop(self.ffn(self.local(batch, a, b, scal)))
        return a, b, scal, h


# ---------------------------------------------------------------------------
# initialisation: original W2 copied bitwise, new modules private
# ---------------------------------------------------------------------------


def _new_keys(model: nn.Module, base_keys: set[str]) -> list[str]:
    return sorted(k for k in model.state_dict() if k not in base_keys)


def _standard_linear_init(linear: nn.Linear, generator: torch.Generator) -> None:
    """``nn.Linear.reset_parameters`` with an explicit private generator."""
    nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    if linear.bias is not None:
        bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
        linear.bias.uniform_(-bound, bound, generator=generator)


def initialize_new_modules(model: nn.Module, key: str, *, seed: int) -> dict[str, Any]:
    """Initialise only the new tensors; never advances the global torch RNG."""
    before = torch.get_rng_state().clone()
    rng_before_hash = C.rng_state_hashes(C.capture_rng_state())
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSETS[key]))
    with torch.no_grad():
        if key == "C2":
            std = 1.0 / math.sqrt(float(C.COND_RANK))
            model.local.path_cond_ec.weight.normal_(0.0, std, generator=gen)
            model.local.path_cond_ee.weight.normal_(0.0, std, generator=gen)
            model.local.path_cond_v.zero_()
            init = {
                "kind": "normal_0_inv_sqrt_rank",
                "std": std,
                "ec_shape": list(model.local.path_cond_ec.weight.shape),
                "ee_shape": list(model.local.path_cond_ee.weight.shape),
                "v_shape": list(model.local.path_cond_v.shape),
                "v_is_zero": bool(torch.count_nonzero(model.local.path_cond_v).item() == 0),
            }
        elif key == "F":
            _standard_linear_init(model.ffn.fc1, gen)
            model.ffn.fc2.weight.zero_()
            model.ffn.fc2.bias.zero_()
            init = {
                "kind": "standard_linear_fc1_zero_fc2",
                "fc1_shape": list(model.ffn.fc1.weight.shape),
                "fc2_zero": bool(torch.count_nonzero(model.ffn.fc2.weight).item() == 0),
                "norm_weight_ones": bool(torch.all(model.ffn.norm.weight == 1).item()),
                "norm_bias_zero": bool(torch.count_nonzero(model.ffn.norm.bias).item() == 0),
            }
        else:  # pragma: no cover - guarded by caller
            raise KeyError(key)
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError(f"{key}: new-module initialisation changed the global torch RNG state")
    rng_after_hash = C.rng_state_hashes(C.capture_rng_state())
    if rng_after_hash != rng_before_hash:
        raise RuntimeError(f"{key}: new-module initialisation changed a tracked RNG state")
    init.update(
        {
            "private_generator_seed": int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSETS[key]),
            "global_rng_unchanged": True,
        }
    )
    return init


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
) -> tuple[BSM.BSModel, dict[str, Any]]:
    """Build C2/F from the frozen untrained W2 seed-0 state."""
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

    with torch.random.fork_rng(devices=[]):
        if key == "C2":
            model: BSM.BSModel = CS2Model(spec, norm, int(n_out), cfg, cond_rank=int(C.COND_RANK))
        elif key == "F":
            model = FModel(spec, norm, int(n_out), cfg)
        else:  # pragma: no cover
            raise KeyError(key)

    new_keys = _new_keys(model, base_keys)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if set(missing) != set(new_keys) or unexpected:
        raise RuntimeError(
            f"{key}: base state load mismatch: missing={sorted(missing)} expected_new={new_keys} "
            f"unexpected={sorted(unexpected)}"
        )

    init_audit = initialize_new_modules(model, key, seed=int(seed))

    # every historical W2 tensor must survive bitwise
    subset_diff: dict[str, float] = {}
    state = model.state_dict()
    for k in sorted(base_keys):
        if tuple(state[k].shape) != tuple(base_state[k].shape) or state[k].dtype != base_state[k].dtype:
            raise RuntimeError(f"{key}: original W2 tensor shape/dtype changed: {k}")
        if not torch.equal(state[k].cpu(), base_state[k].cpu()):
            subset_diff[k] = float((state[k].float().cpu() - base_state[k].float().cpu()).abs().max().item())
    if subset_diff:
        raise RuntimeError(f"{key}: original W2 parameters are not bitwise equal: {subset_diff}")

    after = torch.get_rng_state()
    if not torch.equal(global_rng_before, after):
        raise RuntimeError(f"{key}: build advanced the global torch RNG state")

    model.to(torch.device(device))

    base_elements = int(sum(p.numel() for n, p in model.named_parameters() if n in base_keys))
    new_elements = int(sum(p.numel() for n, p in model.named_parameters() if n not in base_keys))
    total = int(C.parameter_count(model))
    inventory = {
        k: {"shape": list(state[k].shape), "numel": int(state[k].numel())}
        for k in new_keys
    }
    audit: dict[str, Any] = {
        "model": key,
        "model_name": C.MODELS[key]["name"],
        "role": C.MODELS[key]["role"],
        "seed": int(seed),
        "base_model": C.BASE_MODEL_KEY,
        "base_parameters": base_elements,
        "new_parameters": new_elements,
        "total_parameters": total,
        "expected_base_parameters": int(C.BASE_PARAMETERS),
        "expected_new_parameters": int(C.NEW_PARAMETERS[key]),
        "expected_total_parameters": int(C.EXPECTED_PARAMETERS[key]),
        "widths": cfg,
        "init_rule": C.INIT_RULE_C2 if key == "C2" else C.INIT_RULE_F,
        "new_module_init": init_audit,
        "new_state_keys": new_keys,
        "new_parameter_inventory": inventory,
        "original_w2_subset": {
            "n_tensors": len(base_keys),
            "all_bitwise_equal": True,
            "n_differing": 0,
        },
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
        expected_new = int(C.C2_ADDED if key == "C2" else C.FFN_ADDED)
        if new_elements != expected_new or new_elements != int(C.NEW_PARAMETERS[key]):
            raise RuntimeError(f"{key}: new parameters {new_elements} != {C.NEW_PARAMETERS[key]}")
        if total != int(C.EXPECTED_PARAMETERS[key]):
            raise RuntimeError(f"{key}: total parameters {total} != {C.EXPECTED_PARAMETERS[key]}")
        if base_elements != int(C.BASE_PARAMETERS):
            raise RuntimeError(f"{key}: base parameters {base_elements} != {C.BASE_PARAMETERS}")
    return model, audit


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over the sorted (key, dtype, shape, bytes) state."""
    import hashlib

    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("utf-8"))
        h.update(str(tuple(value.shape)).encode("utf-8"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# diagnostics: is the vector gate actually moving?
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


def new_module_norms(model: nn.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if name.startswith("local.path_cond_") or name.startswith("ffn."):
            out[name] = {
                "l2": float(param.detach().float().norm().item()),
                "rms": float(param.detach().float().pow(2).mean().sqrt().item()) if param.numel() else 0.0,
                "numel": int(param.numel()),
            }
    return out


def gate_diagnostics(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Path-gate activation on a fixed batch (eval mode; no RNG consumed)."""
    local = getattr(model, "local", None)
    if not hasattr(local, "path_cond_v"):
        return {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            path_cat = batch["path_cat"]
            path_end63 = batch["path_end63"]
            scalar = local.path_gate(path_cat)
            delta = (local.path_cond_ec(path_cat) * local.path_cond_ee(path_end63)) @ local.path_cond_v
            gate = scalar + delta
            per_path = delta.norm(dim=1)
            out = {
                "n_paths": int(path_cat.numel()),
                "n_atoms": int(batch["atom_feat"].shape[0]),
                "gate_shape": list(gate.shape),
                "gate_width": int(gate.shape[1]) if gate.dim() == 2 else 0,
                "scalar_gate": _stats(scalar),
                "delta_gate": _stats(delta),
                "vector_gate": _stats(gate),
                "delta_per_path_l2": _stats(per_path),
                "delta_nonzero_fraction": float((per_path > 1e-12).float().mean().item()) if per_path.numel() else 0.0,
                "delta_over_scalar": float(per_path.mean().item() / scalar.abs().mean().clamp_min(1e-30).item())
                if per_path.numel()
                else 0.0,
            }
    finally:
        model.train(was_training)
    return out


def ffn_diagnostics(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    ffn = getattr(model, "ffn", None)
    if ffn is None:
        return {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _a, _b, _scal = model._encode(batch)
            h = model.local(batch, _a, _b, _scal)
            branch = ffn.fc2(F.silu(ffn.fc1(ffn.norm(h))))
            out = {
                "n_atoms": int(h.shape[0]),
                "width": int(h.shape[1]),
                "hidden": int(ffn.fc1.out_features),
                "h": _stats(h),
                "branch": _stats(branch),
                "branch_over_h": float(branch.norm().item() / h.norm().clamp_min(1e-30).item()),
            }
    finally:
        model.train(was_training)
    return out


def module_diagnostics(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {"parameter_norms": new_module_norms(model)}
    gate = gate_diagnostics(model, batch)
    if gate:
        out["gate"] = gate
    ffn = ffn_diagnostics(model, batch)
    if ffn:
        out["ffn"] = ffn
    return out


# ---------------------------------------------------------------------------
# forward / audit helpers
# ---------------------------------------------------------------------------


def count_module_calls(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Count every encoding module, the dropout and the new module in one forward."""
    counters = {
        "atom_encoder": 0,
        "bond_encoder": 0,
        "local": 0,
        "dropout": 0,
        "f": 0,
        "rho": 0,
        "readout": 0,
        "new_module": 0,
        "path_cond_ec": 0,
        "path_cond_ee": 0,
    }
    targets = [
        ("atom_encoder", model.atom_encoder),
        ("bond_encoder", model.bond_encoder),
        ("local", model.local),
        ("dropout", model.drop),
        ("f", model.f),
        ("rho", model.rho),
        ("readout", model.readout),
    ]
    if hasattr(model, "ffn"):
        targets.append(("new_module", model.ffn))
    else:
        targets.append(("path_cond_ec", model.local.path_cond_ec))
        targets.append(("path_cond_ee", model.local.path_cond_ee))
    handles = []
    for name, module in targets:
        def _hook(_m, _inp, _out, k=name):
            counters[k] += 1

        handles.append(module.register_forward_hook(_hook))
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return counters


def post_local_without_structure(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """The post-``local`` path must be a function of (h, atom_batch, n_graphs, global)."""
    _a, _b, _scal, h = model.encode_h(batch)
    slim = {
        "n_graphs": batch["n_graphs"],
        "atom_batch": batch["atom_batch"],
        "global": batch["global"],
    }
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model.post_local(h, slim)
    finally:
        model.train(was_training)


def gate_parameters(model: nn.Module) -> dict[str, Any]:
    local = model.local
    return {
        "has_conditional_gate": bool(hasattr(local, "path_cond_v")),
        "path_gate_present": bool(hasattr(local, "path_gate")),
        "path_gate_shape": list(local.path_gate.weight.shape),
        "path_gate_initial_ones": bool(torch.all(local.path_gate.weight == 1).item()),
        "ec_shape": list(local.path_cond_ec.weight.shape) if hasattr(local, "path_cond_ec") else None,
        "ee_shape": list(local.path_cond_ee.weight.shape) if hasattr(local, "path_cond_ee") else None,
        "v_shape": list(local.path_cond_v.shape) if hasattr(local, "path_cond_v") else None,
    }


__all__ = [
    "CS2LocalContent",
    "CS2Model",
    "FModel",
    "LocalResidualFFN",
    "build_model",
    "count_module_calls",
    "ffn_diagnostics",
    "gate_diagnostics",
    "gate_parameters",
    "initialize_new_modules",
    "module_diagnostics",
    "new_module_norms",
    "post_local_without_structure",
    "state_sha256",
]
