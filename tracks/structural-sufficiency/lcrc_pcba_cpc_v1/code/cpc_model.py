"""LCRC-CPC: the frozen W2 plus one non-iterative CPC path-conditioning residual.

Forward (exactly once each)::

    a, b, scal = base._encode(batch)
    h0         = base.drop(base.local(batch, a, b, scal))
    delta      = CPC(h0, b, rooted two-bond paths)
    logits     = base.post_local(h0 + delta, batch)

``h0`` has shape ``[N_atoms,128]``, ``b`` has shape ``[N_bonds,192]`` (the W2
bond encoder output, verified by the constructor), and ``delta`` has shape
``[N_atoms,128]``.  CPC never reads ``h0 + delta`` and ``post_local`` is called
once, so there is no iterative hidden-state propagation.

The joint path feature is deliberately non-additive::

    x_p = [r_i, m_j, d_k, r_i * d_k, m_j * d_k, e_ij, e_jk]       (192)
    z_p = SiLU(Linear(96,64)(SiLU(Linear(192,96)(x_p))))          (64)
    g_p = sigmoid(Linear(96,1)([r_i, z_p]))                       (1)
    context_i = sum_{p rooted at i} g_p * z_p / max(count_i, 1)   (64)
    delta_i = Linear(65,128)([context_i, log1p(count_i)]) * [count_i > 0]

The product ``r_i * d_k`` is the root-endpoint interaction; it cannot be
replaced by a sum of vectors.  One root scatter produces the context, and an
atom with no paths receives exactly zero.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bs_model as BM
import cpc_common as C
import cpc_paths as P


# ---------------------------------------------------------------------------
# the CPC residual
# ---------------------------------------------------------------------------


class CPCModule(nn.Module):
    """One-shot rooted two-bond path conditioner."""

    def __init__(self, width: int = C.WIDTH, bond_width: int = C.BOND_WIDTH):
        super().__init__()
        self.width = int(width)
        self.bond_width = int(bond_width)
        self.norm = nn.LayerNorm(self.width)
        self.root_proj = nn.Linear(self.width, 32)
        self.middle_proj = nn.Linear(self.width, 32)
        self.end_proj = nn.Linear(self.width, 32)
        self.bond_proj = nn.Linear(self.bond_width, 16)
        self.path1 = nn.Linear(192, 96)
        self.path2 = nn.Linear(96, 64)
        self.gate = nn.Linear(96, 1)
        self.out = nn.Linear(65, self.width)

    def project(self, h0: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.norm(h0)
        r = F.silu(self.root_proj(u))
        m = F.silu(self.middle_proj(u))
        d = F.silu(self.end_proj(u))
        e = F.silu(self.bond_proj(b))
        return r, m, d, e

    def path_token(
        self,
        r: torch.Tensor,
        m: torch.Tensor,
        d: torch.Tensor,
        e: torch.Tensor,
        path_atom: torch.Tensor,
        middle: torch.Tensor,
        end: torch.Tensor,
        bond_ij: torch.Tensor,
        bond_jk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rr = r.index_select(0, path_atom)
        mm = m.index_select(0, middle)
        dd = d.index_select(0, end)
        if bond_ij.numel():
            e1 = e.index_select(0, bond_ij)
            e2 = e.index_select(0, bond_jk)
        else:
            e1 = e.new_zeros((0, e.shape[1]))
            e2 = e.new_zeros((0, e.shape[1]))
        x = torch.cat([rr, mm, dd, rr * dd, mm * dd, e1, e2], dim=1)
        z = F.silu(self.path2(F.silu(self.path1(x))))
        gate = torch.sigmoid(self.gate(torch.cat([rr, z], dim=1)))
        return z, gate

    def aggregate(
        self,
        z: torch.Tensor,
        gate: torch.Tensor,
        path_atom: torch.Tensor,
        n_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = z.new_zeros(n_atoms).index_add_(
            0, path_atom, torch.ones_like(path_atom, dtype=z.dtype)
        )
        total = z.new_zeros((n_atoms, z.shape[1])).index_add_(0, path_atom, gate * z)
        context = total / counts.clamp(min=1.0).unsqueeze(1)
        log_count = torch.log1p(counts).unsqueeze(1)
        return context, log_count, counts

    def forward(
        self,
        h0: torch.Tensor,
        b: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        mode: str = "real",
        shuffle_seed: int = 0,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode not in ("real", "delta_zero", "pairing_shuffle"):
            raise KeyError(mode)
        n_atoms = int(h0.shape[0])
        if mode == "delta_zero":
            delta = h0.new_zeros((n_atoms, self.width))
            aux = {
                "mode": mode,
                "delta": delta,
                "n_paths": int(batch["cpc_path_atom"].numel()),
                "zero_path_atoms": int((self._counts(batch["cpc_path_atom"], n_atoms) == 0).sum().item()),
            }
            return delta, aux

        path_atom = batch["cpc_path_atom"]
        middle = batch["cpc_middle"]
        end = batch["cpc_end"]
        bond_ij = batch["cpc_bond_ij"]
        bond_jk = batch["cpc_bond_jk"]
        if mode == "pairing_shuffle":
            path_atom, middle, end, bond_ij, bond_jk, perm = P.shuffle_pairing_within_root(
                path_atom, middle, end, bond_ij, bond_jk, seed=int(shuffle_seed)
            )
        else:
            perm = None

        r, m, d, e = self.project(h0, b)
        z, gate = self.path_token(r, m, d, e, path_atom, middle, end, bond_ij, bond_jk)
        context, log_count, counts = self.aggregate(z, gate, path_atom, n_atoms)
        delta = self.out(torch.cat([context, log_count], dim=1))
        delta = delta * (counts > 0).unsqueeze(1)
        aux: dict[str, Any] = {
            "mode": mode,
            "n_paths": int(path_atom.numel()),
            "path_atom": path_atom,
            "counts": counts,
            "z": z,
            "gate": gate,
            "context": context,
            "log_count": log_count,
            "delta": delta,
            "zero_path_atoms": int((counts == 0).sum().item()),
            "path_permutation": perm,
        }
        return delta, aux

    @staticmethod
    def _counts(path_atom: torch.Tensor, n_atoms: int) -> torch.Tensor:
        return path_atom.new_zeros(n_atoms, dtype=torch.float32).index_add_(
            0, path_atom, torch.ones_like(path_atom, dtype=torch.float32)
        )


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class CPCModel(BM.BSModel):
    """Frozen LCRC-W2 with one CPC residual between ``drop(local)`` and readout."""

    def __init__(self, spec: Any, norm: Mapping[str, Any], n_out: int, cfg: Mapping[str, Any]):
        super().__init__(spec, norm, int(n_out), dict(cfg))
        self.cpc = CPCModule(int(self.cfg["width"]), int(self.cfg["bond_out"]))

    def new_modules(self) -> list[nn.Module]:
        return [self.cpc]

    def new_parameter_names(self) -> list[str]:
        return [n for n, _ in self.named_parameters() if C.is_new_parameter(n)]

    def encode_h0(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, scal = self._encode(batch)
        h0 = self.drop(self.local(batch, a, b, scal))
        return a, b, scal, h0

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_aux: bool = False,
        return_attention: bool = False,
        mode: str = "real",
        shuffle_seed: int = 0,
    ):
        a, b, scal, h0 = self.encode_h0(batch)
        delta, cpc_aux = self.cpc(
            h0, b, batch, mode=mode, shuffle_seed=shuffle_seed, return_aux=True
        )
        h_cpc = h0 + delta
        if not return_aux:
            return self.post_local(h_cpc, batch)
        logits, aux = self.post_local(h_cpc, batch, return_aux=True, return_attention=return_attention)
        aux.update({"h0": h0, "h_cpc": h_cpc, "delta": delta, "a": a, "b": b, "scal": scal, "cpc": cpc_aux})
        return logits, aux


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


def _standard_linear_init(linear: nn.Linear, generator: torch.Generator) -> None:
    nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    if linear.bias is not None:
        bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
        linear.bias.uniform_(-bound, bound, generator=generator)


def initialize_cpc_modules(model: CPCModel, *, seed: int, zero_output: bool = True) -> dict[str, Any]:
    """Private-generator init; the global torch RNG is never advanced."""
    gen = torch.Generator(device="cpu").manual_seed(
        int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSET)
    )
    before = torch.get_rng_state().clone()
    n_linear = 0
    with torch.no_grad():
        for module in model.new_modules():
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    _standard_linear_init(sub, gen)
                    n_linear += 1
                elif isinstance(sub, nn.LayerNorm):
                    nn.init.ones_(sub.weight)
                    nn.init.zeros_(sub.bias)
    if zero_output:
        with torch.no_grad():
            model.cpc.out.weight.zero_()
            model.cpc.out.bias.zero_()
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("private CPC initialisation changed the global torch RNG state")
    return {
        "private_generator_seed": int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSET),
        "linear_layers_initialised": n_linear,
        "expected_linear_layers": 8,
        "layernorm_layers_initialised": 1,
        "output_zeroed": bool(zero_output),
        "global_rng_unchanged": True,
    }


def compare_states(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    if set(a) != set(b):
        return {
            "all_equal": False,
            "reason": "key mismatch",
            "missing_in_a": sorted(set(b) - set(a)),
            "missing_in_b": sorted(set(a) - set(b)),
            "n_tensors": 0,
        }
    diffs = {}
    for key in a:
        ta = torch.as_tensor(a[key])
        tb = torch.as_tensor(b[key])
        diffs[key] = {
            "equal": bool(ta.shape == tb.shape and torch.equal(ta, tb)),
            "max_abs_diff": float((ta.float() - tb.float()).abs().max().item()) if ta.numel() else 0.0,
        }
    return {
        "all_equal": all(v["equal"] for v in diffs.values()),
        "n_tensors": len(diffs),
        "n_unequal": sum(1 for v in diffs.values() if not v["equal"]),
        "per_tensor": diffs,
    }


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    per_tensor: dict[str, dict[str, Any]] = {}
    groups = {"w2_base": 0, "w2_readout": 0, "cpc": 0}
    subgroups = {
        "norm": 0,
        "role_projections": 0,
        "bond_projection": 0,
        "path_mlp_1": 0,
        "path_mlp_2": 0,
        "gate": 0,
        "output": 0,
    }
    for name, p in model.named_parameters():
        n = int(p.numel())
        new = C.is_new_parameter(name)
        per_tensor[name] = {"shape": list(p.shape), "numel": n, "new": new}
        if name.startswith("readout."):
            groups["w2_readout"] += n
        elif new:
            groups["cpc"] += n
            if name.startswith("cpc.norm."):
                subgroups["norm"] += n
            elif name.startswith(("cpc.root_proj.", "cpc.middle_proj.", "cpc.end_proj.")):
                subgroups["role_projections"] += n
            elif name.startswith("cpc.bond_proj."):
                subgroups["bond_projection"] += n
            elif name.startswith("cpc.path1."):
                subgroups["path_mlp_1"] += n
            elif name.startswith("cpc.path2."):
                subgroups["path_mlp_2"] += n
            elif name.startswith("cpc.gate."):
                subgroups["gate"] += n
            elif name.startswith("cpc.out."):
                subgroups["output"] += n
            else:
                raise RuntimeError(f"unclassified CPC tensor {name}")
        else:
            groups["w2_base"] += n
    total = sum(groups.values())
    expected = C.EXPECTED_PARAMETERS
    checks = {
        "w2_base": groups["w2_base"] == expected["w2_base"],
        "w2_readout": groups["w2_readout"] == expected["w2_readout"],
        "w2_total": groups["w2_base"] + groups["w2_readout"] == expected["w2_total"],
        "cpc": groups["cpc"] == expected["cpc_total"],
        "norm": subgroups["norm"] == expected["norm"],
        "role_projections": subgroups["role_projections"] == expected["role_projections"],
        "bond_projection": subgroups["bond_projection"] == expected["bond_projection"],
        "path_mlp_1": subgroups["path_mlp_1"] == expected["path_mlp_1"],
        "path_mlp_2": subgroups["path_mlp_2"] == expected["path_mlp_2"],
        "gate": subgroups["gate"] == expected["gate"],
        "output": subgroups["output"] == expected["output"],
        "total": total == expected["total"],
    }
    return {
        "groups": groups,
        "cpc_subgroups": subgroups,
        "total": int(total),
        "cpc_total": int(groups["cpc"]),
        "expected": dict(expected),
        "checks": checks,
        "all_match": bool(all(checks.values())),
        "per_tensor": per_tensor,
        "n_parameter_tensors": len(per_tensor),
        "n_state_tensors": len(model.state_dict()),
    }


def block_fingerprints(model: CPCModel) -> dict[str, str]:
    return {"cpc": C.state_fingerprint({k: v for k, v in model.cpc.state_dict().items()})}


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def build_cpc_model(
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    zero_output: bool = True,
    verify: bool = True,
) -> tuple[CPCModel, dict[str, Any]]:
    cfg = dict(C.config_for_scale(2))
    reference, ref_audit = BM.build_model(
        C.W2_KEY, spec, norm, int(n_out), seed=int(seed), device="cpu", verify_parameters=True
    )
    rng_after_w2 = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = CPCModel(spec, norm, int(n_out), cfg)
    if not torch.equal(rng_after_w2, torch.get_rng_state()):
        raise RuntimeError("LCRC-CPC construction advanced the global torch RNG")
    load = model.load_state_dict(reference.state_dict(), strict=False)
    expected_missing = {k for k in model.state_dict() if C.is_new_parameter(k)}
    if set(load.missing_keys) != expected_missing or load.unexpected_keys:
        raise RuntimeError(
            f"W2 state load mismatch: missing={sorted(load.missing_keys)} "
            f"expected_new={sorted(expected_missing)} unexpected={sorted(load.unexpected_keys)}"
        )
    base_before = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    init_audit = initialize_cpc_modules(model, seed=int(seed), zero_output=bool(zero_output))
    base_after = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    if base_before != base_after:
        raise RuntimeError("CPC initialisation changed the W2 part of the state")
    model.to(torch.device(device))

    audit = parameter_audit(model)
    if verify and not audit["all_match"]:
        raise RuntimeError(f"LCRC-CPC parameter audit failed: {audit['checks']}")
    ref_state = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
    ours_base = C.state_subset({k: v.detach().cpu() for k, v in model.state_dict().items()}, new=False)
    base_equal = compare_states(ours_base, ref_state)
    if verify and not base_equal["all_equal"]:
        raise RuntimeError(f"W2 part is not bitwise identical to the W2 seed-0 constructor: {base_equal}")
    out = {
        "model": C.MODEL_NAME,
        "base_model": C.W2_KEY,
        "seed": int(seed),
        "widths": dict(cfg),
        "cpc": dict(C.CPC),
        "parameters": audit,
        "w2_reference_parameters": int(ref_audit["total_parameters"]),
        "w2_reference_expected_parameters": int(ref_audit["expected_parameters"]),
        "w2_part_bitwise_equal_reference": base_equal,
        "w2_trunk_init_fingerprint": base_before,
        "expected_w2_trunk_init_fingerprint": C.W2_SEED0["trunk_init_fingerprint"],
        "w2_trunk_fingerprint_matches_recorded_w2": base_before
        == C.W2_SEED0["trunk_init_fingerprint"],
        "init": init_audit,
        "block_fingerprints": block_fingerprints(model),
        "state_keys": sorted(model.state_dict()),
        "n_state_tensors": len(model.state_dict()),
        "base_state_sha256": base_before,
        "cpc_state_sha256": C.state_fingerprint(C.state_subset(model.state_dict(), new=True)),
        "device": str(device),
    }
    if verify and not out["w2_trunk_fingerprint_matches_recorded_w2"]:
        raise RuntimeError(
            f"W2 trunk fingerprint {base_before} != recorded W2 seed-0 "
            f"{C.W2_SEED0['trunk_init_fingerprint']}"
        )
    return model, out


def untrained_w2(spec: Any, norm: Mapping[str, Any], n_out: int, *, seed: int = C.SEED, device: torch.device | str = "cpu"):
    return BM.build_model(C.W2_KEY, spec, norm, int(n_out), seed=int(seed), device=device)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item()) if x.numel() else 0.0


def _quantiles(x: torch.Tensor) -> dict[str, float]:
    if x.numel() == 0:
        return {f"p{q}": 0.0 for q in (1, 5, 25, 50, 75, 95, 99)}
    a = x.detach().float().reshape(-1)
    return {f"p{q}": float(torch.quantile(a, q / 100.0).item()) for q in (1, 5, 25, 50, 75, 95, 99)}


def usage_metrics(model: CPCModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            logits, aux = model(batch, return_aux=True)
            h0 = aux["h0"]
            delta = aux["delta"]
            cpc = aux["cpc"]
            z = cpc.get("z")
            gate = cpc.get("gate")
            context = cpc.get("context")
            counts = cpc.get("counts")
            out: dict[str, Any] = {
                "rms_h0": rms(h0),
                "rms_delta": rms(delta),
                "R": (rms(delta) / rms(h0)) if rms(h0) else None,
                "rms_h_cpc": rms(aux["h_cpc"]),
                "rms_z": rms(z) if z is not None else 0.0,
                "rms_context": rms(context) if context is not None else 0.0,
                "finite_fraction_h_cpc": float(torch.isfinite(aux["h_cpc"]).float().mean().item()),
                "finite_logits": bool(torch.isfinite(logits).all().item()),
                "logits_abs_max": float(logits.abs().max().item()),
                "n_atoms": int(h0.shape[0]),
                "n_paths": int(cpc.get("n_paths", 0)),
            }
            if gate is not None and gate.numel():
                out["gate"] = {
                    "mean": float(gate.mean().item()),
                    "std": float(gate.std(unbiased=False).item()),
                    "min": float(gate.min().item()),
                    "max": float(gate.max().item()),
                    **_quantiles(gate),
                }
            else:
                out["gate"] = {"mean": None, "std": None, "min": None, "max": None}
            if counts is not None and counts.numel():
                nonzero = counts[counts > 0]
                out["path_counts"] = {
                    "mean": float(counts.mean().item()),
                    "p95": float(torch.quantile(counts.float(), 0.95).item()),
                    "max": int(counts.max().item()),
                    "zero_path_atoms": int((counts == 0).sum().item()),
                    "zero_path_fraction": float((counts == 0).float().mean().item()),
                    "mean_nonzero": float(nonzero.mean().item()) if nonzero.numel() else 0.0,
                }
                if gate is not None and gate.numel() and cpc.get("path_atom") is not None:
                    path_atom = cpc["path_atom"]
                    sums = gate.new_zeros(int(counts.numel())).index_add_(0, path_atom, gate.reshape(-1))
                    sums2 = gate.new_zeros(int(counts.numel())).index_add_(0, path_atom, gate.reshape(-1) ** 2)
                    denom = counts.clamp(min=1.0)
                    var = (sums2 / denom - (sums / denom) ** 2).clamp(min=0.0)
                    multi = counts > 1
                    out["gate"]["per_root_std_mean"] = float(var[multi].sqrt().mean().item()) if multi.any() else 0.0
                    out["gate"]["per_root_std_p95"] = (
                        float(torch.quantile(var[multi].sqrt(), 0.95).item()) if multi.any() else 0.0
                    )
                    out["gate"]["roots_with_multiple_paths"] = int(multi.sum().item())
            out["weight_norms"] = {
                name: float(p.detach().norm().item())
                for name, p in model.cpc.named_parameters()
                if name in ("root_proj.weight", "middle_proj.weight", "end_proj.weight", "bond_proj.weight", "path1.weight", "path2.weight", "gate.weight", "out.weight")
            }
            out["output_bias_norm"] = float(model.cpc.out.bias.detach().norm().item())
    finally:
        if was_training:
            model.train()
    return out


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_logits(
    model: CPCModel,
    task: Any,
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
    mode: str = "real",
    shuffle_seed: int = 0,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    rows = np.asarray(rows, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            if hasattr(task, "cpc"):
                batch = P.collate_with_cpc(task, task.cpc, chunk.tolist())
            else:  # pragma: no cover - defensive
                raise RuntimeError("task has no CPC structure cache")
            batch = to_device(batch, device)
            logits = model(batch, mode=mode, shuffle_seed=shuffle_seed)
            if logits.shape[1] != C.N_TASKS:
                raise RuntimeError(f"model returned {logits.shape[1]} outputs, expected {C.N_TASKS}")
            out.append(logits.detach().cpu().numpy().astype(np.float32))
    logits = np.concatenate(out, axis=0) if out else np.zeros((0, C.N_TASKS), dtype=np.float32)
    if logits.shape[0] != len(rows):
        raise RuntimeError(f"prediction rows {logits.shape[0]} != {len(rows)}")
    return logits


__all__ = [
    "CPCModule",
    "CPCModel",
    "block_fingerprints",
    "build_cpc_model",
    "compare_states",
    "initialize_cpc_modules",
    "parameter_audit",
    "predict_logits",
    "rms",
    "to_device",
    "untrained_w2",
    "usage_metrics",
]
