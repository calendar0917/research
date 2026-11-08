"""LCRC-S3-SP: W2 + the unchanged S3 residual + a selective path residual.

Forward (``SPModel.forward``), one pass, nothing propagated twice::

    a, b, scal = self._encode(batch)                 # one atom, one bond encode
    h_raw      = self.local(batch, a, b, scal)       # one local call
    z          = self.s3_branch.encode_paths(batch, a)          # unchanged S3
    s, counts  = self.s3_branch.root_inputs(z, path_atom, N)    # unchanged S3
    r_original = self.s3_branch.out(s)                          # unchanged S3
    r_select   = self.selective(a, z, path_atom, s[:, :48], counts)
    h          = self.drop(h_raw + r_original + r_select)       # one dropout
    logits     = self.post_local(h, batch)                      # unchanged W2

with the selective branch, per root ``r`` and head ``h``::

    q        = Linear(192,48)(a)                    # static atom encoding only
    k        = Linear(48,48)(z)                     # S3 path embedding
    v        = z                                    # no extra projection
    score    = <q[r][:,h], k[p][:,h]> / sqrt(24)
    weight   = softmax over {p : root(p) = r}       # per root, per head
    attn_r   = sum_p weight * v[p]
    delta_r  = attn_r - mean_z_r
    r_select = Linear(48,128)(delta_r)              # zero-initialised
    r_select[root with counts == 0] = 0

``selective_delta`` is exactly zero when the attention is uniform, and exactly
zero for a root with a single S3 path, because ``attn = mu`` there.

There is no iterative message passing here: the query comes from the *static*
atom encoder output ``a``, the keys/values come from the once-enumerated S3
paths, and the result is one additive residual added before the single dropout.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

import bs_model as BSM
import s3_common as C3
import s3_model as M3
import sp_common as C


SQRT_HEAD_DIM = math.sqrt(float(C.SP_HEAD_DIM))

#: inherited state hashing (sorted keys + dtype + shape + bytes)
state_sha256 = M3.state_sha256


# ---------------------------------------------------------------------------
# grouped (per-root) softmax, no per-path Python loop
# ---------------------------------------------------------------------------


def segment_softmax(score: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Stable softmax of ``score[n_paths, H]`` inside each ``index`` group.

    Groups are independent: no weight can leak across roots or molecules.  The
    grouping max is taken with ``scatter_reduce_(reduce='amax')`` and the
    normaliser with ``index_add_``; both are single kernels over the whole path
    set, so the operator count does not grow with the number of paths.
    """
    if score.numel() == 0:
        return score.new_zeros(score.shape)
    n_segments = int(n_segments)
    n_heads = int(score.shape[1])
    idx = index.unsqueeze(1).expand(-1, n_heads)
    dtype = score.dtype
    fill = torch.finfo(dtype).min
    seg_max = score.new_full((n_segments, n_heads), fill)
    seg_max.scatter_reduce_(0, idx, score, reduce="amax", include_self=True)
    shifted = score - seg_max.index_select(0, index)
    w = torch.exp(shifted)
    denom = score.new_zeros((n_segments, n_heads)).index_add_(0, index, w)
    denom = denom.clamp_min(torch.finfo(dtype).tiny)
    return w / denom.index_select(0, index)


def attention_summed(weight: torch.Tensor, v: torch.Tensor, index: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """``[n_atoms, heads*head_dim]`` scatter of ``weight * v`` inside each root."""
    n_paths, n_heads, head_dim = v.shape
    flat = (weight.unsqueeze(-1) * v).reshape(n_paths, n_heads * head_dim)
    return v.new_zeros((int(n_atoms), n_heads * head_dim)).index_add_(0, index, flat)


# ---------------------------------------------------------------------------
# the selective branch
# ---------------------------------------------------------------------------


class SelectivePathBranch(nn.Module):
    """Root-conditioned content attention over the once-enumerated S3 paths."""

    def __init__(self, *, atom_out: int, s3_dim: int = C.S3_DIM, n_heads: int = C.SP_HEADS, width: int = 128):
        super().__init__()
        if int(s3_dim) % int(n_heads):
            raise ValueError("s3_dim must be divisible by the number of heads")
        self.atom_out = int(atom_out)
        self.s3_dim = int(s3_dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(s3_dim) // int(n_heads)
        self.width = int(width)
        self.q = nn.Linear(self.atom_out, self.s3_dim)
        self.k = nn.Linear(self.s3_dim, self.s3_dim)
        self.out = nn.Linear(self.s3_dim, self.width)

    def forward(
        self,
        *,
        a: torch.Tensor,
        z: torch.Tensor,
        root: torch.Tensor,
        mean_z: torch.Tensor,
        counts: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        n_atoms = int(a.shape[0])
        n_paths = int(z.shape[0])
        q = self.q(a).reshape(n_atoms, self.n_heads, self.head_dim)
        if n_paths:
            k = self.k(z).reshape(n_paths, self.n_heads, self.head_dim)
            v = z.reshape(n_paths, self.n_heads, self.head_dim)
            score = (q.index_select(0, root) * k).sum(dim=-1) / SQRT_HEAD_DIM
            weight = segment_softmax(score, root, n_atoms)
            attn = attention_summed(weight, v, root, n_atoms)
        else:  # degenerate: no S3 path in the whole batch
            score = z.new_zeros((0, self.n_heads))
            weight = score
            attn = z.new_zeros((n_atoms, self.s3_dim))
        delta = attn - mean_z
        residual = self.out(delta)
        # A root without paths has no mean/attention content at all; forcing the
        # residual to zero prevents the learned bias from faking one.
        has_paths = (counts > 0).unsqueeze(1)
        residual = residual * has_paths.to(residual.dtype)
        delta = delta * has_paths.to(delta.dtype)
        aux: dict[str, Any] = {
            "n_paths": n_paths,
            "selective_delta": delta.detach(),
            "selective_residual": residual.detach(),
        }
        if return_attention:
            aux["attention_weight"] = weight.detach()
            aux["attention_score"] = score.detach()
            aux["attention_attn"] = attn.detach()
        return residual, aux


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class SPModel(M3.S3Model):
    """W2 trunk + unchanged S3 residual + the selective path residual."""

    def __init__(self, spec, norm, n_out: int, cfg: Mapping[str, Any]):
        super().__init__(spec, norm, n_out, cfg)
        self.selective = SelectivePathBranch(
            atom_out=int(cfg["atom_out"]),
            s3_dim=int(C.S3_DIM),
            n_heads=int(C.SP_HEADS),
            width=int(cfg["width"]),
        )

    # -- pieces, computed once ------------------------------------------------

    def components(self, batch: Mapping[str, torch.Tensor], *, return_attention: bool = False) -> dict[str, Any]:
        a, b, scal = self._encode(batch)          # one atom encode, one bond encode
        h_raw = self.local(batch, a, b, scal)     # one local call
        path_atom = batch["s3_path_atom"]
        n_atoms = int(a.shape[0])
        z = self.s3_branch.encode_paths(batch, a)                     # unchanged
        s, counts = self.s3_branch.root_inputs(z, path_atom, n_atoms)  # unchanged
        r_original = self.s3_branch.out(s)                            # unchanged
        r_selective, aux = self.selective(
            a=a,
            z=z,
            root=path_atom,
            mean_z=s[:, : C.S3_DIM],
            counts=counts,
            return_attention=bool(return_attention),
        )
        out = {
            "a": a,
            "b": b,
            "scal": scal,
            "h_raw": h_raw,
            "z": z,
            "s": s,
            "counts": counts,
            "r_original": r_original,
            "r_selective": r_selective,
            "selective_aux": aux,
        }
        return out

    def encode_h(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        selective_zero: bool = False,
        s3_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        comp = self.components(batch)
        residual = comp["r_selective"]
        if selective_zero:
            residual = torch.zeros_like(residual)
        if s3_zero:
            # diagnostic only: drop the whole S3 contribution (original + selective)
            h = self.drop(comp["h_raw"])
        else:
            h = self.drop(comp["h_raw"] + comp["r_original"] + residual)
        return comp["a"], comp["b"], comp["scal"], h

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_aux: bool = False,
        return_attention: bool = False,
        selective_zero: bool = False,
        s3_zero: bool = False,
    ):
        comp = self.components(batch, return_attention=bool(return_attention))
        residual = comp["r_selective"]
        if selective_zero:
            residual = torch.zeros_like(residual)
        if s3_zero:
            h = self.drop(comp["h_raw"])
        else:
            h = self.drop(comp["h_raw"] + comp["r_original"] + residual)
        if not (return_aux or return_attention):
            return self.post_local(h, batch)
        logits, aux = self.post_local(h, batch, return_aux=True, return_attention=True)
        aux.update(
            {
                "h_raw": comp["h_raw"],
                "h": h,
                "r_original": comp["r_original"],
                "r_selective": residual,
                "selective_delta": comp["selective_aux"]["selective_delta"],
                "counts": comp["counts"],
                "n_paths": comp["selective_aux"]["n_paths"],
            }
        )
        if return_attention:
            for key in ("attention_weight", "attention_score", "attention_attn"):
                aux[key] = comp["selective_aux"][key]
        return logits, aux


# ---------------------------------------------------------------------------
# initialisation: historical W2 + S3 bitwise, private generator for q/k
# ---------------------------------------------------------------------------


def initialize_selective(branch: SelectivePathBranch, *, seed: int = C.SP_PRIVATE_SEED) -> dict[str, Any]:
    """Initialise ``q`` and ``k`` from a private CPU generator; zero ``out``."""
    before = torch.get_rng_state().clone()
    rng_before = C.rng_state_hashes(C.capture_rng_state())
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        M3._standard_linear_init(branch.q, gen)
        M3._standard_linear_init(branch.k, gen)
        branch.out.weight.zero_()
        branch.out.bias.zero_()
    if not torch.equal(before, torch.get_rng_state()):
        raise RuntimeError("selective initialisation changed the global torch RNG state")
    if C.rng_state_hashes(C.capture_rng_state()) != rng_before:
        raise RuntimeError("selective initialisation changed a tracked RNG state")
    return {
        "kind": "private_generator(kaiming_uniform_q + kaiming_uniform_k) + zero_out",
        "private_generator_seed": int(seed),
        "generator_order": ["q.weight", "q.bias", "k.weight", "k.bias"],
        "q_weight_shape": list(branch.q.weight.shape),
        "k_weight_shape": list(branch.k.weight.shape),
        "out_shape": list(branch.out.weight.shape),
        "out_is_zero": bool(torch.count_nonzero(branch.out.weight).item() == 0)
        and bool(torch.count_nonzero(branch.out.bias).item() == 0),
        "global_rng_unchanged": True,
        "q_weight_sha256": C.sha256_bytes(branch.q.weight.detach().cpu().numpy().tobytes()),
        "k_weight_sha256": C.sha256_bytes(branch.k.weight.detach().cpu().numpy().tobytes()),
    }


def build_model(
    spec,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    verify_parameters: bool = True,
) -> tuple[SPModel, dict[str, Any]]:
    """Build SP from the frozen, historical W2 and S3 seed-0 initial states."""
    key = "SP"
    base_model, base_audit = BSM.build_model(
        C.BASE_MODEL_KEY, spec, norm, int(n_out), seed=int(seed), device="cpu"
    )
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    # The historical S3 arm, built by the historical code path, byte-for-byte.
    s3_model, s3_audit = M3.build_model(
        "S3", spec, norm, int(n_out), seed=int(seed), device="cpu", base_model=base_model
    )
    s3_state = {k: v.detach().cpu().clone() for k, v in s3_model.state_dict().items()}
    s3_base_keys = set(base_state)
    s3_keys = set(s3_state)
    inherited_state = dict(s3_state)
    global_rng_before = torch.get_rng_state().clone()

    with torch.random.fork_rng(devices=[]):
        model = SPModel(spec, norm, int(n_out), dict(base_model.cfg))

    new_keys = sorted(k for k in model.state_dict() if k not in s3_keys)
    missing, unexpected = model.load_state_dict(inherited_state, strict=False)
    if set(missing) != set(new_keys) or unexpected:
        raise RuntimeError(
            f"SP base state load mismatch: missing={sorted(missing)} "
            f"expected_new={new_keys} unexpected={sorted(unexpected)}"
        )
    init_audit = initialize_selective(model.selective, seed=int(C.SP_PRIVATE_SEED))

    state = model.state_dict()
    subset_diff: dict[str, float] = {}
    for k in sorted(s3_keys):
        if tuple(state[k].shape) != tuple(inherited_state[k].shape) or state[k].dtype != inherited_state[k].dtype:
            raise RuntimeError(f"SP inherited tensor shape/dtype changed: {k}")
        if not torch.equal(state[k].cpu(), inherited_state[k].cpu()):
            subset_diff[k] = float((state[k].float().cpu() - inherited_state[k].float().cpu()).abs().max().item())
    if subset_diff:
        raise RuntimeError(f"SP inherited parameters are not bitwise equal: {subset_diff}")
    if not torch.equal(global_rng_before, torch.get_rng_state()):
        raise RuntimeError("SP build advanced the global torch RNG state")

    model.to(torch.device(device))
    base_elements = int(sum(p.numel() for n, p in model.named_parameters() if n in s3_base_keys))
    s3_elements = int(sum(p.numel() for n, p in model.named_parameters() if n in s3_keys))
    new_elements = int(sum(p.numel() for n, p in model.named_parameters() if n not in s3_keys))
    total = int(C.parameter_count(model))
    new_state_keys = [k for k in state if k not in s3_keys]
    inventory = {k: {"shape": list(state[k].shape), "numel": int(state[k].numel())} for k in new_state_keys}
    audit: dict[str, Any] = {
        "model": key,
        "model_name": C.MODELS[key]["name"],
        "role": C.MODELS[key]["role"],
        "seed": int(seed),
        "base_model": C.BASE_MODEL_KEY,
        "base_parameters": base_elements,
        "inherited_s3_parameters": s3_elements,
        "s3_only_parameters": int(s3_elements - base_elements),
        "new_parameters": new_elements,
        "total_parameters": total,
        "expected_base_parameters": int(C.BASE_PARAMETERS),
        "expected_s3_parameters": int(C.S3_PARAMETERS),
        "expected_new_parameters": int(C.MODELS[key]["new_parameters"]),
        "expected_total_parameters": int(C.EXPECTED_PARAMETERS[key]),
        "new_parameter_fraction_of_s3": float(new_elements) / float(C.S3_PARAMETERS),
        "widths": dict(base_model.cfg),
        "init_rule": C.INIT_RULE,
        "new_module_init": init_audit,
        "new_state_keys": new_state_keys,
        "new_parameter_inventory": inventory,
        "original_w2_subset": {"n_tensors": len(s3_base_keys), "all_bitwise_equal": True, "n_differing": 0},
        "inherited_w2_plus_s3_subset": {"n_tensors": len(s3_keys), "all_bitwise_equal": True, "n_differing": 0},
        "base_state_sha256": M3.state_sha256(base_state),
        "s3_inherited_state_sha256": M3.state_sha256(inherited_state),
        "s3_rebuilt_state_sha256": s3_audit.get("model_state_sha256"),
        "model_state_sha256": M3.state_sha256(state),
        "base_trunk_init_fingerprint": base_audit.get("trunk_init_fingerprint"),
        "state_keys": sorted(state),
        "n_state_tensors": len(state),
        "parameter_tensors": len(list(model.parameters())),
        "base_model_audit": base_audit,
        "s3_model_audit": {
            "total_parameters": s3_audit.get("total_parameters"),
            "new_parameters": s3_audit.get("new_parameters"),
            "model_state_sha256": s3_audit.get("model_state_sha256"),
        },
        "global_rng_unchanged": True,
        "parameter_arithmetic": {
            "base": C.BASE_PARAMETERS,
            "s3_added": C.S3_ADDED,
            "selective_added": C.SP_ADDED,
            "q": int(C.SP_BRANCH["q"]["parameters"]),
            "k": int(C.SP_BRANCH["k"]["parameters"]),
            "out": int(C.SP_BRANCH["out"]["parameters"]),
            "total": int(C.SP_PARAMETERS),
        },
    }
    if verify_parameters:
        if base_elements != C.BASE_PARAMETERS:
            raise RuntimeError(f"SP base parameters {base_elements} != {C.BASE_PARAMETERS}")
        if s3_elements != C.S3_PARAMETERS:
            raise RuntimeError(f"SP inherited S3 parameters {s3_elements} != {C.S3_PARAMETERS}")
        if new_elements != int(C.MODELS[key]["new_parameters"]):
            raise RuntimeError(f"SP new parameters {new_elements} != {C.MODELS[key]['new_parameters']}")
        if total != int(C.EXPECTED_PARAMETERS[key]):
            raise RuntimeError(f"SP total parameters {total} != {C.EXPECTED_PARAMETERS[key]}")
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


def attention_stats(
    weight: torch.Tensor, root: torch.Tensor, counts: torch.Tensor, *, n_atoms: int
) -> dict[str, Any]:
    """Diagnostic summary of the per-root attention (never a selection signal)."""
    out: dict[str, Any] = {"n_paths": int(root.numel())}
    if root.numel() == 0:
        out.update({"weight_sum_max_abs_error": 0.0, "uniform_like_fraction": 0.0})
        return out
    w = weight.detach().float()
    idx = root
    n_heads = int(w.shape[1])
    sums = w.new_zeros((int(n_atoms), n_heads)).index_add_(0, idx, w)
    with_paths = counts > 0
    used = sums[with_paths]
    out["weight_sum_max_abs_error"] = float((used - 1.0).abs().max().item()) if used.numel() else 0.0
    c = counts[with_paths].clamp(min=1.0).unsqueeze(1)
    # entropy of each root's weight vector (relative to its own maximum log c)
    ent = -(w.clamp_min(1e-30).log() * w)
    ent_root = ent.new_zeros((int(n_atoms), n_heads)).index_add_(0, idx, ent)
    ent_root = ent_root[with_paths] / c
    ent_max = c.log()
    ratio = torch.where(ent_max > 0, ent_root / ent_max.clamp_min(1e-12), torch.ones_like(ent_root))
    multi = (counts[with_paths] > 1)
    ratio_multi = ratio[multi] if bool(multi.any()) else ratio.new_zeros(0)
    # per-root maximum weight (scatter amax, then average over roots that have paths)
    fill = torch.finfo(w.dtype).min
    root_max = w.new_full((int(n_atoms), n_heads), fill)
    root_max.scatter_reduce_(0, idx.unsqueeze(1).expand(-1, n_heads), w, reduce="amax", include_self=True)
    root_max_used = root_max[with_paths]
    out.update(
        {
            "entropy_ratio_mean": float(ratio.mean().item()) if ratio.numel() else 1.0,
            "entropy_ratio_mean_multi_path_roots": float(ratio_multi.mean().item()) if ratio_multi.numel() else 1.0,
            "entropy_ratio_min": float(ratio.min().item()) if ratio.numel() else 1.0,
            "max_weight_mean": float(root_max_used.mean().item()) if root_max_used.numel() else 1.0,
            "max_weight_p99": float(torch.quantile(root_max_used.flatten().float(), 0.99).item())
            if root_max_used.numel()
            else 1.0,
            "near_one_hot_fraction": float((root_max_used > 0.99).float().mean().item())
            if root_max_used.numel()
            else 0.0,
            "near_uniform_fraction": float((root_max_used <= 1.01 / c.repeat(1, n_heads)).float().mean().item())
            if root_max_used.numel()
            else 0.0,
            "roots_multi_path": int(multi.sum().item()),
            "roots_single_path": int((counts[with_paths] == 1).sum().item()),
            "roots_without_paths": int((counts == 0).sum().item()),
            "n_weights_outside_unit_interval": int(((w < -1e-6) | (w > 1.0 + 1e-6)).sum().item()),
            "finite": bool(torch.isfinite(w).all().item()),
        }
    )
    return out


@torch.no_grad()
def branch_diagnostics(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Residual/attention diagnostic on a fixed batch (eval mode, no RNG used)."""
    was_training = model.training
    model.eval()
    try:
        comp = model.components(batch, return_attention=True)
        h_raw = comp["h_raw"]
        r_orig = comp["r_original"]
        r_sel = comp["r_selective"]
        aux = comp["selective_aux"]
        counts = comp["counts"]
        stats = attention_stats(aux["attention_weight"], batch["s3_path_atom"], counts, n_atoms=int(h_raw.shape[0]))
        out: dict[str, Any] = {
            "residual_original": _stats(r_orig),
            "residual_selective": _stats(r_sel),
            "selective_delta": _stats(aux["selective_delta"]),
            "h_raw": _stats(h_raw),
            "residual_over_h_raw": float(
                (r_orig + r_sel).pow(2).mean().sqrt().item() / max(h_raw.pow(2).mean().sqrt().item(), 1e-12)
            ),
            "selective_over_residual_total": float(
                r_sel.pow(2).mean().sqrt().item()
                / max((r_orig + r_sel).pow(2).mean().sqrt().item(), 1e-12)
            ),
            "selective_nonzero_fraction": float((r_sel != 0).float().mean().item()),
            "roots_with_paths": int((counts > 0).sum().item()),
            "roots_with_zero_selective": int(((r_sel.abs().sum(1) == 0) & (counts > 0)).sum().item()),
            "n_paths": int(batch["s3_path_atom"].numel()),
            "paths_per_atom": float(batch["s3_path_atom"].numel() / max(int(batch["atom_batch"].numel()), 1)),
            "z_rms": float(comp["z"].pow(2).mean().sqrt().item()) if comp["z"].numel() else 0.0,
            "attention": stats,
            "finite": bool(torch.isfinite(r_sel).all().item()),
        }
        return out
    finally:
        model.train(was_training)


@torch.no_grad()
def branch_grad_norm(model: nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in model.named_parameters():
        if (name.startswith("s3_branch.") or name.startswith("selective.")) and param.grad is not None:
            out[name] = float(param.grad.detach().float().norm().item())
    return out


def new_module_norms(model: nn.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if (name.startswith("s3_branch.") or name.startswith("selective.")) and param.numel():
            out[name] = {
                "l2": float(param.detach().float().norm().item()),
                "rms": float(param.detach().float().pow(2).mean().sqrt().item()),
                "numel": int(param.numel()),
            }
    return out


# ---------------------------------------------------------------------------
# analytical cost accounting
# ---------------------------------------------------------------------------


def selective_flops(*, n_paths: int, n_atoms: int) -> dict[str, float]:
    """Analytical forward FLOPs (2 per MAC) of the selective branch."""
    heads, head_dim = int(C.SP_HEADS), int(C.SP_HEAD_DIM)
    atom_out, s3_dim, width = int(C.WIDTHS["atom_out"]), int(C.S3_DIM), int(C.WIDTHS["width"])
    q_atoms = 2.0 * atom_out * s3_dim
    k_paths = 2.0 * s3_dim * s3_dim
    dot_paths = 2.0 * heads * head_dim + 4.0 * heads        # dot + scale + softmax bookkeeping
    scatter_paths = 4.0 * heads * head_dim + 4.0 * heads    # weight*v + index_add + max/sum
    out_atoms = 2.0 * s3_dim * width
    return {
        "flops_atoms_q": float(n_atoms) * q_atoms,
        "flops_paths_k": float(n_paths) * k_paths,
        "flops_paths_score_softmax_attn": float(n_paths) * (dot_paths + scatter_paths),
        "flops_atoms_out": float(n_atoms) * out_atoms,
        "flops_total": float(n_atoms) * (q_atoms + out_atoms) + float(n_paths) * (k_paths + dot_paths + scatter_paths),
        "flops_per_path": float(k_paths + dot_paths + scatter_paths),
        "flops_per_atom": float(q_atoms + out_atoms),
    }


# ---------------------------------------------------------------------------
# call counting (no-propagation audit)
# ---------------------------------------------------------------------------


def count_module_calls_typed(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Count every trunk/branch module call in one forward.

    ``s3_encode`` / ``s3_out`` are counted on the S3 branch's own Linear layers
    because the SP forward calls ``encode_paths`` / ``root_inputs`` / ``out``
    explicitly (it needs ``z`` and the rooted statistics for the selective
    branch); the S3 code itself is unchanged.
    """
    modules: dict[str, nn.Module] = {}
    for name in ("atom_encoder", "bond_encoder", "local", "f", "rho", "readout", "drop", "selective"):
        module = getattr(model, name, None)
        if module is not None:
            modules[name] = module
    s3_branch = getattr(model, "s3_branch", None)
    if s3_branch is not None:
        modules["s3_encode"] = s3_branch.we
        modules["s3_out"] = s3_branch.out
    counters = {n: 0 for n in modules}
    handles = []
    for name, module in modules.items():

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
    "SelectivePathBranch",
    "SPModel",
    "attention_stats",
    "attention_summed",
    "branch_diagnostics",
    "branch_grad_norm",
    "build_model",
    "count_module_calls_typed",
    "initialize_selective",
    "new_module_norms",
    "segment_softmax",
    "state_sha256",
    "selective_flops",
]
