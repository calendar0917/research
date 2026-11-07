"""LCRC-S3 and the near-parameter-matched FFN control, both on the frozen W2.

Two arms, same untrained historical W2 seed-0 state, same recipe, same batch
order:

``S3``  W2 + one-shot three-hop residual

    ``z_p = SiLU(W_e a_x + E_u[type(u)] + E_v[type(v)] + E_1[b1] + E_2[b2]
                 + E_3[b3] + E_d[dist(r,x)])``         (per simple path p=r-u-v-x)
    ``s_r = [mean_p z_p, sum_p z_p / sqrt(|S3(r)|), log1p(|S3(r)|)]``  (97 dims)
    ``h   = dropout(h_raw + Linear(97,128)(s_r))``

``F``   W2 + per-atom FFN residual (capacity control, no S3 access at all)

    ``h   = dropout(h_raw + fc2(SiLU(fc1(LayerNorm(h_raw)))))``

Hard constraints, audited rather than asserted:

* exactly one ``atom_encoder``, one ``bond_encoder``, one ``local`` call and one
  ``drop`` call per forward (forward hooks, see ``count_module_calls_typed``);
* nothing is propagated along a bond a second time: the S3 branch reads only the
  **static** atom embedding ``a`` and the once-enumerated paths, and it never
  reads ``h``, ``bond_index`` or any updated atom state;
* the S3 output layer is zero-initialised, so at step 0 both arms produce
  exactly the W2 logits;
* every original W2 tensor is bitwise equal to the historical seed-0 initial
  state, and the new tensors are drawn from a private generator that never
  advances the global torch RNG.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import bs_model as BSM
import s3_common as C


# ---------------------------------------------------------------------------
# S3 path encoder + rooted aggregation
# ---------------------------------------------------------------------------


class S3Branch(nn.Module):
    """Explicit one-shot three-hop residual (root-local, no propagation)."""

    def __init__(self, spec, width: int, *, atom_out: int, s3_dim: int = C.S3_DIM):
        super().__init__()
        self.s3_dim = int(s3_dim)
        self.width = int(width)
        self.atom_out = int(atom_out)
        self.n_atom_types = int(spec.n_atom_types)
        self.n_bond_types = int(spec.n_bond_types)
        if self.n_atom_types != 16 or self.n_bond_types != 5:
            raise RuntimeError(
                f"frozen S3 vocabulary drift: n_atom_types={self.n_atom_types} "
                f"n_bond_types={self.n_bond_types}"
            )
        self.we = nn.Linear(self.atom_out, self.s3_dim)
        self.e_mid1 = nn.Embedding(self.n_atom_types, self.s3_dim)
        self.e_mid2 = nn.Embedding(self.n_atom_types, self.s3_dim)
        self.e_bond1 = nn.Embedding(self.n_bond_types, self.s3_dim)
        self.e_bond2 = nn.Embedding(self.n_bond_types, self.s3_dim)
        self.e_bond3 = nn.Embedding(self.n_bond_types, self.s3_dim)
        self.e_dist = nn.Embedding(3, self.s3_dim)  # dist in {1,2,3} -> 0..2
        self.out = nn.Linear(2 * self.s3_dim + 1, self.width)

    # -- per-path encoding --------------------------------------------------

    def encode_paths(self, batch: Mapping[str, torch.Tensor], a: torch.Tensor) -> torch.Tensor:
        """``z_p`` for every S3 path of the batch (order = cache order)."""
        end = batch["s3_path_end"]
        if end.numel() == 0:
            return a.new_zeros((0, self.s3_dim))
        # ``a`` is the *static* atom encoder output of the atom encoder; the
        # endpoint is gathered, never recomputed from a propagated state.
        gx = self.we(a)[end]
        z = (
            gx
            + self.e_mid1(batch["s3_mid1_type"])
            + self.e_mid2(batch["s3_mid2_type"])
            + self.e_bond1(batch["s3_bond1"] - 1)
            + self.e_bond2(batch["s3_bond2"] - 1)
            + self.e_bond3(batch["s3_bond3"] - 1)
            + self.e_dist(batch["s3_dist"] - 1)
        )
        return F.silu(z)

    def root_inputs(
        self, z: torch.Tensor, path_atom: torch.Tensor, n_atoms: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``s_r = [mu_r, q_r, log1p(m_r)]`` and the per-root path count."""
        d = self.s3_dim
        counts = z.new_zeros(n_atoms).index_add_(0, path_atom, torch.ones_like(path_atom, dtype=z.dtype))
        total = z.new_zeros((n_atoms, d)).index_add_(0, path_atom, z)
        denom = counts.clamp(min=1.0)
        mu = total / denom.unsqueeze(1)
        q = total / denom.sqrt().unsqueeze(1)
        logc = torch.log1p(counts).unsqueeze(1)
        return torch.cat([mu, q, logc], dim=1), counts

    def forward(
        self, batch: Mapping[str, torch.Tensor], a: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        z = self.encode_paths(batch, a)
        s, counts = self.root_inputs(z, batch["s3_path_atom"], int(a.shape[0]))
        residual = self.out(s)
        aux = {
            "n_paths": int(z.shape[0]),
            "path_count_per_atom_mean": float(counts.mean().item()) if counts.numel() else 0.0,
            "atoms_without_paths": int((counts == 0).sum().item()) if counts.numel() else 0,
            "z_rms": float(z.pow(2).mean().sqrt().item()) if z.numel() else 0.0,
        }
        return residual, aux


# ---------------------------------------------------------------------------
# FFN capacity control
# ---------------------------------------------------------------------------


class FFNBranch(nn.Module):
    """Pre-norm per-atom residual FFN; sees only ``h_raw``."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(width))
        self.fc1 = nn.Linear(int(width), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(width))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(self.norm(h))))


# ---------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------


class S3Model(BSM.BSModel):
    """W2 trunk + the one-shot three-hop residual inserted before dropout."""

    def __init__(self, spec, norm, n_out: int, cfg: Mapping[str, Any]):
        super().__init__(spec, norm, n_out, cfg)
        self.s3_branch = S3Branch(
            spec, int(cfg["width"]), atom_out=int(cfg["atom_out"]), s3_dim=int(C.S3_DIM)
        )

    def encode_h(self, batch: Mapping[str, torch.Tensor], *, s3_zero: bool = False):
        a, b, scal = self._encode(batch)          # one atom encode, one bond encode
        h_raw = self.local(batch, a, b, scal)     # one local call
        if s3_zero:
            h = self.drop(h_raw)                  # exactly one dropout call
            return a, b, scal, h
        residual, _aux = self.s3_branch(batch, a)
        h = self.drop(h_raw + residual)           # exactly one dropout call
        return a, b, scal, h

    def forward(self, batch: Mapping[str, torch.Tensor], *, s3_zero: bool = False):
        _a, _b, _scal, h = self.encode_h(batch, s3_zero=bool(s3_zero))
        return self.post_local(h, batch)


class FModel(BSM.BSModel):
    """W2 trunk + one per-atom residual FFN inserted before dropout."""

    def __init__(self, spec, norm, n_out: int, cfg: Mapping[str, Any], *, hidden: int = C.FFN_HIDDEN):
        super().__init__(spec, norm, n_out, cfg)
        self.ffn = FFNBranch(int(cfg["width"]), int(hidden))

    def encode_h(self, batch: Mapping[str, torch.Tensor], *, s3_zero: bool = False):
        a, b, scal = self._encode(batch)
        h_raw = self.local(batch, a, b, scal)
        h = self.drop(h_raw + self.ffn(h_raw))
        return a, b, scal, h


# ---------------------------------------------------------------------------
# initialisation: historical W2 bitwise + private generator for the new tensors
# ---------------------------------------------------------------------------


def _new_keys(model: nn.Module, base_keys: set[str]) -> list[str]:
    return sorted(k for k in model.state_dict() if k not in base_keys)


def _standard_linear_init(linear: nn.Linear, generator: torch.Generator) -> None:
    nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    if linear.bias is not None:
        bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
        linear.bias.uniform_(-bound, bound, generator=generator)


def _embedding_init(embedding: nn.Embedding, generator: torch.Generator) -> None:
    std = 1.0 / math.sqrt(float(embedding.embedding_dim))
    embedding.weight.normal_(0.0, std, generator=generator)


def initialize_new_modules(model: nn.Module, key: str, *, seed: int) -> dict[str, Any]:
    """Initialise only the new tensors; never advances the global torch RNG."""
    before = torch.get_rng_state().clone()
    rng_before = C.rng_state_hashes(C.capture_rng_state())
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSETS[key]))
    with torch.no_grad():
        if key == "S3":
            branch = model.s3_branch
            _standard_linear_init(branch.we, gen)
            for emb in (branch.e_mid1, branch.e_mid2, branch.e_bond1, branch.e_bond2, branch.e_bond3, branch.e_dist):
                _embedding_init(emb, gen)
            branch.out.weight.zero_()
            branch.out.bias.zero_()
            init = {
                "kind": "kaiming_linear_we + normal_0_inv_sqrt_dim_embeddings + zero_out",
                "embedding_std": 1.0 / math.sqrt(float(branch.s3_dim)),
                "out_is_zero": bool(torch.count_nonzero(branch.out.weight).item() == 0)
                and bool(torch.count_nonzero(branch.out.bias).item() == 0),
                "shapes": {
                    "we": list(branch.we.weight.shape),
                    "e_mid1": list(branch.e_mid1.weight.shape),
                    "e_dist": list(branch.e_dist.weight.shape),
                    "out": list(branch.out.weight.shape),
                },
            }
        elif key == "F":
            branch = model.ffn
            _standard_linear_init(branch.fc1, gen)
            branch.fc2.weight.zero_()
            branch.fc2.bias.zero_()
            init = {
                "kind": "standard_linear_fc1 + zero_fc2 + default_layernorm",
                "fc2_is_zero": bool(torch.count_nonzero(branch.fc2.weight).item() == 0)
                and bool(torch.count_nonzero(branch.fc2.bias).item() == 0),
                "norm_weight_ones": bool(torch.all(branch.norm.weight == 1).item()),
                "norm_bias_zero": bool(torch.count_nonzero(branch.norm.bias).item() == 0),
                "shapes": {
                    "norm": list(branch.norm.weight.shape),
                    "fc1": list(branch.fc1.weight.shape),
                    "fc2": list(branch.fc2.weight.shape),
                },
            }
        else:  # pragma: no cover - guarded by the caller
            raise KeyError(key)
    if not torch.equal(before, torch.get_rng_state()):
        raise RuntimeError(f"{key}: new-module initialisation changed the global torch RNG state")
    if C.rng_state_hashes(C.capture_rng_state()) != rng_before:
        raise RuntimeError(f"{key}: new-module initialisation changed a tracked RNG state")
    init.update(
        {
            "private_generator_seed": int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSETS[key]),
            "global_rng_unchanged": True,
        }
    )
    return init


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
    n_out: int,
    *,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    base_model: BSM.BSModel | None = None,
    verify_parameters: bool = True,
) -> tuple[BSM.BSModel, dict[str, Any]]:
    """Build S3/F from the frozen untrained W2 seed-0 state."""
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
        if key == "S3":
            model: BSM.BSModel = S3Model(spec, norm, int(n_out), cfg)
        elif key == "F":
            model = FModel(spec, norm, int(n_out), cfg, hidden=int(C.FFN_HIDDEN))
        else:  # pragma: no cover
            raise KeyError(key)

    new_keys = _new_keys(model, base_keys)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if set(missing) != set(new_keys) or unexpected:
        raise RuntimeError(
            f"{key}: base state load mismatch: missing={sorted(missing)} "
            f"expected_new={sorted(new_keys)} unexpected={sorted(unexpected)}"
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
    inventory = {k: {"shape": list(state[k].shape), "numel": int(state[k].numel())} for k in new_keys}
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
        "expected_new_parameters": int(C.MODELS[key]["new_parameters"]),
        "expected_total_parameters": int(C.EXPECTED_PARAMETERS[key]),
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
        "parameter_gap": int(C.PARAMETER_GAP),
        "parameter_gap_fraction": float(C.PARAMETER_GAP_FRACTION),
    }
    if verify_parameters:
        if new_elements != int(C.MODELS[key]["new_parameters"]):
            raise RuntimeError(f"{key}: new parameters {new_elements} != {C.MODELS[key]['new_parameters']}")
        if total != int(C.EXPECTED_PARAMETERS[key]):
            raise RuntimeError(f"{key}: total parameters {total} != {C.EXPECTED_PARAMETERS[key]}")
        if base_elements != int(C.BASE_PARAMETERS):
            raise RuntimeError(f"{key}: base parameters {base_elements} != {C.BASE_PARAMETERS}")
        if C.PARAMETER_GAP_FRACTION > C.PARAMETER_GAP_MAX_FRACTION:
            raise RuntimeError(
                f"parameter match {C.PARAMETER_GAP_FRACTION:.4%} exceeds {C.PARAMETER_GAP_MAX_FRACTION:.2%}"
            )
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


def new_module_norms(model: nn.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if name.startswith("s3_branch.") or name.startswith("ffn."):
            out[name] = {
                "l2": float(param.detach().float().norm().item()),
                "rms": float(param.detach().float().pow(2).mean().sqrt().item()) if param.numel() else 0.0,
                "numel": int(param.numel()),
            }
    return out


@torch.no_grad()
def branch_diagnostics(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Residual strength on a fixed batch (eval mode, no RNG consumed)."""
    was_training = model.training
    model.eval()
    try:
        a, b, scal = model._encode(batch)
        h_raw = model.local(batch, a, b, scal)
        if hasattr(model, "s3_branch"):
            residual, aux = model.s3_branch(batch, a)
            out = {
                "kind": "s3",
                "residual": _stats(residual),
                "h_raw": _stats(h_raw),
                "residual_over_h_raw": float(
                    residual.pow(2).mean().sqrt().item() / max(h_raw.pow(2).mean().sqrt().item(), 1e-12)
                ),
                "nonzero_fraction": float((residual != 0).float().mean().item()),
                "atom_fraction_nonzero": float((residual.abs().sum(1) > 0).float().mean().item()),
                "n_paths": int(batch["s3_path_atom"].numel()),
                "paths_per_atom": float(
                    batch["s3_path_atom"].numel() / max(int(batch["atom_batch"].numel()), 1)
                ),
                **{f"enc_{k}": v for k, v in aux.items()},
            }
        else:
            residual = model.ffn(h_raw)
            out = {
                "kind": "ffn",
                "residual": _stats(residual),
                "h_raw": _stats(h_raw),
                "residual_over_h_raw": float(
                    residual.pow(2).mean().sqrt().item() / max(h_raw.pow(2).mean().sqrt().item(), 1e-12)
                ),
                "nonzero_fraction": float((residual != 0).float().mean().item()),
            }
        out["h"] = _stats(h_raw + residual)
        return out
    finally:
        model.train(was_training)


@torch.no_grad()
def branch_grad_norm(model: nn.Module) -> dict[str, float]:
    """Gradient norm of every new tensor (call after ``backward``)."""
    out: dict[str, float] = {}
    for name, param in model.named_parameters():
        if (name.startswith("s3_branch.") or name.startswith("ffn.")) and param.grad is not None:
            out[name] = float(param.grad.detach().float().norm().item())
    return out


# ---------------------------------------------------------------------------
# analytical cost accounting
# ---------------------------------------------------------------------------


def branch_flops(model: nn.Module, *, n_paths: int, n_atoms: int) -> dict[str, float]:
    """Analytical FLOPs (2 per MAC, 1 per add) of the new branch, forward.

    ``W_e`` is applied to *every path* (one atom-embedding row per path), so its
    cost is path-based; the per-atom cost is the root aggregation statistics and
    the output ``Linear(2*s3_dim+1, width)``.
    """
    if hasattr(model, "s3_branch"):
        d = int(model.s3_branch.s3_dim)
        atom_out = int(model.s3_branch.atom_out)
        width = int(model.s3_branch.width)
        per_path = 2.0 * atom_out * d + 6.0 * d + 1.0 * d + 4.0 * d + 2.0 * d  # W_e, 6 gathers, add, SiLU, scatter-adds
        per_atom = 2.0 * (2 * d + 1) * width + 6.0 * d  # out Linear, clamp/sqrt/log1p/concat
        return {
            "kind": "s3",
            "flops_paths": float(n_paths) * per_path,
            "flops_atoms": float(n_atoms) * per_atom,
            "flops_total": float(n_paths) * per_path + float(n_atoms) * per_atom,
            "flops_per_path": float(per_path),
            "flops_per_atom": float(per_atom),
            "flops_we_paths": float(n_paths) * 2.0 * atom_out * d,
            "flops_out_atoms": float(n_atoms) * 2.0 * (2 * d + 1) * width,
        }
    width = int(model.ffn.fc1.in_features)
    hidden = int(model.ffn.fc1.out_features)
    per_atom = 2.0 * (width * hidden + hidden + hidden * width + width) + 4.0 * width
    return {
        "kind": "ffn",
        "flops_atoms": float(n_atoms) * per_atom,
        "flops_total": float(n_atoms) * per_atom,
        "flops_per_atom": float(per_atom),
    }


# ---------------------------------------------------------------------------
# call counting (no-propagation audit)
# ---------------------------------------------------------------------------


def count_module_calls_typed(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Count every trunk/branch module call in one forward."""
    names = ("atom_encoder", "bond_encoder", "local", "f", "rho", "readout", "drop", "s3_branch", "ffn")
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
    "FFNBranch",
    "FModel",
    "S3Branch",
    "S3Model",
    "branch_diagnostics",
    "branch_flops",
    "branch_grad_norm",
    "build_model",
    "count_module_calls_typed",
    "initialize_new_modules",
    "new_module_norms",
    "state_sha256",
]
