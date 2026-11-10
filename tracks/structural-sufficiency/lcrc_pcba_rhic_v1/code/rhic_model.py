"""LCRC-RHIC: the frozen W2 plus one non-iterative two-level branch residual.

Forward (exactly once each)::

    a, b, scal = base._encode(batch)
    h0         = base.drop(base.local(batch, a, b, scal))
    delta      = RHIC(h0, b, path_index, bond_index)
    logits     = base.post_local(h0 + delta, batch)

``h0`` is ``[N_atoms,128]``, ``b`` is the W2 bond encoder output
``[N_bonds,192]`` and ``delta`` is ``[N_atoms,128]``.  RHIC never reads
``h0 + delta``; ``post_local`` runs once.  There is no iterative atom-state
message passing.

The path token is the audited CPC v1 token, unchanged::

    x_p = [r_i, m_j, d_k, r_i*d_k, m_j*d_k, e_ij, e_jk]              (192)
    z_p = SiLU(Linear(96,64)(SiLU(Linear(192,96)(x_p))))             (64)
    g_p = sigmoid(Linear(96,1)([r_i, z_p]))                          (1)
    v_p = g_p * z_p                                                  (64)

RHIC then replaces the *global* path mean of CPC by a two-level composition:

* **branch level** -- for every directed branch ``b = (i, j)`` with paths
  ``P_b``: ``mu_b = mean v_p``, ``q_b = mean_{p != q} v_p * v_q``,
  ``n_b = |P_b|``.  The branch is encoded from
  ``[r_i, m_j, e_ij, mu_b, q_b, log1p n_b]`` (209 -> 96 -> 64).
* **root level** -- for every root ``i`` with active branches ``B_i``:
  ``mean_branch = mean branch_value``, ``cross_branch = mean_{b != c} bv_b*bv_c``,
  ``d_i = |B_i|``.  The residual is
  ``Linear(192,128)(SiLU(Linear(162,192)([r_i, mean_branch, cross_branch,
  log1p path_count_i, log1p d_i])))``, multiplied by ``[d_i > 0]``.

Both pair terms are the per-channel mean over *distinct* instances; the
implementation uses ``mean^2 - sum((x-mean)^2)/(n(n-1))``, the algebraically
identical stable form of ``(S^2 - T)/(n(n-1))``.  ``n < 2`` contributes exactly
zero.  Branches with ``n_b = 0`` are never encoded and never counted.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rhic_common as C  # sets up the inherited code paths first
import bs_model as BM  # noqa: E402
import cpc_paths as P  # noqa: E402
import rhic_branches as B  # noqa: E402
from cpc_model import _standard_linear_init, compare_states  # noqa: E402  audited helpers


# ---------------------------------------------------------------------------
# the RHIC residual
# ---------------------------------------------------------------------------


class RHICModule(nn.Module):
    """One-shot root-conditioned hierarchical composition of rooted paths."""

    def __init__(self, width: int = C.WIDTH, bond_width: int = C.BOND_WIDTH):
        super().__init__()
        self.width = int(width)
        self.bond_width = int(bond_width)
        # --- CPC v1 path token (identical layers, identical arithmetic) -----
        self.norm = nn.LayerNorm(self.width)
        self.root_proj = nn.Linear(self.width, 32)
        self.middle_proj = nn.Linear(self.width, 32)
        self.end_proj = nn.Linear(self.width, 32)
        self.bond_proj = nn.Linear(self.bond_width, 16)
        self.path1 = nn.Linear(192, 96)
        self.path2 = nn.Linear(96, 64)
        self.gate = nn.Linear(96, 1)
        # --- branch level ---------------------------------------------------
        self.branch1 = nn.Linear(209, 96)
        self.branch2 = nn.Linear(96, 64)
        # --- root level -----------------------------------------------------
        self.root1 = nn.Linear(162, 192)
        self.root2 = nn.Linear(192, 128)
        #: structural pre-conditions are audited before training; keeping them
        #: out of the hot loop avoids a device->host sync per forward.
        self.validate_branch_ids = False
        self.validate_path_root_consistency = False

    # -- token encoder (CPC v1, verbatim) ---------------------------------

    def project(self, h0: torch.Tensor, b: torch.Tensor):
        u = self.norm(h0)
        r = F.silu(self.root_proj(u))
        m = F.silu(self.middle_proj(u))
        d = F.silu(self.end_proj(u))
        e = F.silu(self.bond_proj(b))
        return r, m, d, e

    def path_tokens(
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
        *,
        return_features: bool = False,
    ):
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
        if return_features:
            return z, gate, gate * z, x
        return z, gate, gate * z

    # -- two-level composition -------------------------------------------

    def compose(
        self,
        r: torch.Tensor,
        m: torch.Tensor,
        e: torch.Tensor,
        v: torch.Tensor,
        path_atom: torch.Tensor,
        bond_index: torch.Tensor,
        branch_id: torch.Tensor,
        n_atoms: int,
    ) -> dict[str, Any]:
        n_branches = 2 * int(bond_index.shape[0])
        counts, active, inv, active_ids = B.branch_group_layout(branch_id, n_branches)
        root_b, mid_b, phys_b = B.branch_endpoints(active_ids, bond_index)
        counts_b = counts.index_select(0, active_ids)
        _, mu, dispersion_b = B.group_mean_dispersion(v, inv, counts_b)
        q_b = B.pairwise_cooccurrence(mu, dispersion_b, counts_b)
        branch_input = torch.cat(
            [
                r.index_select(0, root_b),
                m.index_select(0, mid_b),
                e.index_select(0, phys_b),
                mu,
                q_b,
                torch.log1p(counts_b.to(v.dtype)).unsqueeze(1),
            ],
            dim=1,
        )
        branch_value = F.silu(self.branch2(F.silu(self.branch1(branch_input))))

        d_root = torch.bincount(root_b, minlength=int(n_atoms))
        path_count = torch.bincount(path_atom, minlength=int(n_atoms))
        _, mean_branch, dispersion_r = B.group_mean_dispersion(branch_value, root_b, d_root)
        cross_branch = B.pairwise_cooccurrence(mean_branch, dispersion_r, d_root)
        root_input = torch.cat(
            [
                r,
                mean_branch,
                cross_branch,
                torch.log1p(path_count.to(v.dtype)).unsqueeze(1),
                torch.log1p(d_root.to(v.dtype)).unsqueeze(1),
            ],
            dim=1,
        )
        delta = self.root2(F.silu(self.root1(root_input)))
        delta = delta * (d_root > 0).unsqueeze(1)
        return {
            "branch_id": branch_id,
            "branch_counts": counts,
            "branch_active": active,
            "branch_inv": inv,
            "active_ids": active_ids,
            "branch_root": root_b,
            "branch_middle": mid_b,
            "branch_bond": phys_b,
            "branch_n": counts_b,
            "branch_mu": mu,
            "branch_q": q_b,
            "branch_dispersion": dispersion_b,
            "branch_input": branch_input,
            "branch_value": branch_value,
            "root_branch_count": d_root,
            "root_path_count": path_count,
            "mean_branch": mean_branch,
            "cross_branch": cross_branch,
            "root_dispersion": dispersion_r,
            "root_input": root_input,
            "delta": delta,
        }

    # -- forward ----------------------------------------------------------

    def forward(
        self,
        h0: torch.Tensor,
        b: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        mode: str = "real",
        shuffle_seed: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode not in ("real", "delta_zero", "branch_assignment_shuffle"):
            raise KeyError(mode)
        n_atoms = int(h0.shape[0])
        path_atom = batch["cpc_path_atom"]
        middle = batch["cpc_middle"]
        end = batch["cpc_end"]
        bond_ij = batch["cpc_bond_ij"]
        bond_jk = batch["cpc_bond_jk"]
        if mode == "delta_zero":
            delta = h0.new_zeros((n_atoms, self.width))
            return delta, {
                "mode": mode,
                "delta": delta,
                "n_paths": int(path_atom.numel()),
                "n_branches": 0,
                "zero_branch_atoms": n_atoms,
                "zero_path_atoms": n_atoms,
            }

        branch_id = B.branch_ids(
            path_atom,
            middle,
            bond_ij,
            batch["bond_index"],
            verify=bool(self.validate_branch_ids),
        )
        perm = None
        if mode == "branch_assignment_shuffle":
            branch_id, perm = B.shuffle_branch_assignment(
                path_atom, branch_id, seed=int(shuffle_seed)
            )
        r, m, d, e = self.project(h0, b)
        z, gate, v = self.path_tokens(r, m, d, e, path_atom, middle, end, bond_ij, bond_jk)
        grouped = self.compose(r, m, e, v, path_atom, batch["bond_index"], branch_id, n_atoms)
        grouped.update(
            {
                "mode": mode,
                "r": r,
                "m": m,
                "d": d,
                "be": e,
                "z": z,
                "gate": gate,
                "v": v,
                "path_permutation": perm,
                "n_paths": int(path_atom.numel()),
                "n_branches": int(grouped["active_ids"].numel()),
                "zero_path_atoms": int((grouped["root_path_count"] == 0).sum().item()),
                "zero_branch_atoms": int((grouped["root_branch_count"] == 0).sum().item()),
            }
        )
        return grouped["delta"], grouped

    @staticmethod
    def _counts(path_atom: torch.Tensor, n_atoms: int) -> torch.Tensor:
        return torch.bincount(path_atom, minlength=int(n_atoms)).to(torch.float32)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class RHICModel(BM.BSModel):
    """Frozen LCRC-W2 with one RHIC residual between ``drop(local)`` and readout."""

    def __init__(self, spec: Any, norm: Mapping[str, Any], n_out: int, cfg: Mapping[str, Any]):
        super().__init__(spec, norm, int(n_out), dict(cfg))
        self.rhic = RHICModule(int(self.cfg["width"]), int(self.cfg["bond_out"]))

    def new_modules(self) -> list[nn.Module]:
        return [self.rhic]

    def new_parameter_names(self) -> list[str]:
        return [n for n, _ in self.named_parameters() if C.is_new_parameter(n)]

    def encode_h0(self, batch: Mapping[str, Any]):
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
        delta, rhic_aux = self.rhic(
            h0, b, batch, mode=mode, shuffle_seed=shuffle_seed
        )
        h = h0 + delta
        if not return_aux:
            return self.post_local(h, batch)
        logits, aux = self.post_local(
            h, batch, return_aux=True, return_attention=return_attention
        )
        aux.update(
            {
                "h0": h0,
                "h": h,
                "delta": delta,
                "a": a,
                "b": b,
                "scal": scal,
                "rhic": rhic_aux,
            }
        )
        return logits, aux


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


def initialize_rhic_modules(model: RHICModel, *, seed: int, zero_output: bool = True) -> dict[str, Any]:
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
            model.rhic.root2.weight.zero_()
            model.rhic.root2.bias.zero_()
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("private RHIC initialisation changed the global torch RNG state")
    return {
        "private_generator_seed": int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSET),
        "linear_layers_initialised": n_linear,
        "expected_linear_layers": 11,
        "layernorm_layers_initialised": 1,
        "final_root_linear_zeroed": bool(zero_output),
        "global_rng_unchanged": True,
    }


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    per_tensor: dict[str, dict[str, Any]] = {}
    groups = {"w2_base": 0, "w2_readout": 0, "rhic": 0}
    subgroups = {"token": 0, "branch1": 0, "branch2": 0, "root1": 0, "root2": 0}
    for name, p in model.named_parameters():
        n = int(p.numel())
        new = C.is_new_parameter(name)
        per_tensor[name] = {"shape": list(p.shape), "numel": n, "new": new}
        if name.startswith("readout."):
            groups["w2_readout"] += n
        elif new:
            groups["rhic"] += n
            if C.is_token_parameter(name):
                subgroups["token"] += n
            elif name.startswith("rhic.branch1."):
                subgroups["branch1"] += n
            elif name.startswith("rhic.branch2."):
                subgroups["branch2"] += n
            elif name.startswith("rhic.root1."):
                subgroups["root1"] += n
            elif name.startswith("rhic.root2."):
                subgroups["root2"] += n
            else:
                raise RuntimeError(f"unclassified RHIC tensor {name}")
        else:
            groups["w2_base"] += n
    total = sum(groups.values())
    expected = C.EXPECTED_PARAMETERS
    checks = {
        "w2_base": groups["w2_base"] == expected["w2_base"],
        "w2_readout": groups["w2_readout"] == expected["w2_readout"],
        "w2_total": groups["w2_base"] + groups["w2_readout"] == expected["w2_total"],
        "rhic": groups["rhic"] == expected["rhic_total"],
        "token": subgroups["token"] == expected["token_total"],
        "branch1": subgroups["branch1"] == expected["branch1"],
        "branch2": subgroups["branch2"] == expected["branch2"],
        "branch_total": subgroups["branch1"] + subgroups["branch2"] == expected["branch_total"],
        "root1": subgroups["root1"] == expected["root1"],
        "root2": subgroups["root2"] == expected["root2"],
        "root_total": subgroups["root1"] + subgroups["root2"] == expected["root_total"],
        "total": total == expected["total"],
    }
    return {
        "groups": groups,
        "rhic_subgroups": subgroups,
        "total": int(total),
        "rhic_total": int(groups["rhic"]),
        "expected": dict(expected),
        "checks": checks,
        "all_match": bool(all(checks.values())),
        "per_tensor": per_tensor,
        "n_parameter_tensors": len(per_tensor),
        "n_state_tensors": len(model.state_dict()),
    }


def block_fingerprints(model: RHICModel) -> dict[str, str]:
    state = model.rhic.state_dict()
    token = {k: v for k, v in state.items() if k.split(".")[0] not in ("branch1", "branch2", "root1", "root2")}
    branch = {k: v for k, v in state.items() if k.split(".")[0] in ("branch1", "branch2")}
    root = {k: v for k, v in state.items() if k.split(".")[0] in ("root1", "root2")}
    return {
        "rhic": C.state_fingerprint(state),
        "rhic_token": C.state_fingerprint(token),
        "rhic_branch": C.state_fingerprint(branch),
        "rhic_root": C.state_fingerprint(root),
    }


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def build_rhic_model(
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    zero_output: bool = True,
    verify: bool = True,
) -> tuple[RHICModel, dict[str, Any]]:
    cfg = dict(C.config_for_scale(2))
    reference, ref_audit = BM.build_model(
        "W2", spec, norm, int(n_out), seed=int(seed), device="cpu", verify_parameters=True
    )
    rng_after_w2 = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = RHICModel(spec, norm, int(n_out), cfg)
    if not torch.equal(rng_after_w2, torch.get_rng_state()):
        raise RuntimeError("LCRC-RHIC construction advanced the global torch RNG")
    load = model.load_state_dict(reference.state_dict(), strict=False)
    expected_missing = {k for k in model.state_dict() if C.is_new_parameter(k)}
    if set(load.missing_keys) != expected_missing or load.unexpected_keys:
        raise RuntimeError(
            f"W2 state load mismatch: missing={sorted(load.missing_keys)} "
            f"expected_new={sorted(expected_missing)} unexpected={sorted(load.unexpected_keys)}"
        )
    base_before = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    init_audit = initialize_rhic_modules(model, seed=int(seed), zero_output=bool(zero_output))
    base_after = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    if base_before != base_after:
        raise RuntimeError("RHIC initialisation changed the W2 part of the state")
    model.to(torch.device(device))

    audit = parameter_audit(model)
    if verify and not audit["all_match"]:
        raise RuntimeError(f"LCRC-RHIC parameter audit failed: {audit['checks']}")
    ref_state = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
    ours_base = C.state_subset({k: v.detach().cpu() for k, v in model.state_dict().items()}, new=False)
    base_equal = compare_states(ours_base, ref_state)
    if verify and not base_equal["all_equal"]:
        raise RuntimeError(f"W2 part is not bitwise identical to the W2 seed-0 constructor: {base_equal}")
    out = {
        "model": C.MODEL_NAME,
        "base_model": C.BASE_MODEL,
        "seed": int(seed),
        "widths": dict(cfg),
        "rhic": dict(C.RHIC),
        "parameters": audit,
        "w2_reference_parameters": int(ref_audit["total_parameters"]),
        "w2_reference_expected_parameters": int(ref_audit["expected_parameters"]),
        "w2_part_bitwise_equal_reference": base_equal,
        "w2_trunk_init_fingerprint": base_before,
        "expected_w2_trunk_init_fingerprint": C.W2_SEED0["trunk_init_fingerprint"],
        # the recorded fingerprint belongs to the historical seed-0 W2; for any
        # other seed the bitwise equality with the freshly built reference is the
        # binding check above.
        "w2_trunk_fingerprint_matches_recorded_w2": None
        if int(seed) != int(C.SEED)
        else base_before == C.W2_SEED0["trunk_init_fingerprint"],
        "init": init_audit,
        "block_fingerprints": block_fingerprints(model),
        "state_keys": sorted(model.state_dict()),
        "n_state_tensors": len(model.state_dict()),
        "base_state_sha256": base_before,
        "rhic_state_sha256": C.state_fingerprint(C.state_subset(model.state_dict(), new=True)),
        "device": str(device),
    }
    if verify and int(seed) == int(C.SEED) and not out["w2_trunk_fingerprint_matches_recorded_w2"]:
        raise RuntimeError(
            f"W2 trunk fingerprint {base_before} != recorded W2 seed-0 "
            f"{C.W2_SEED0['trunk_init_fingerprint']}"
        )
    return model, out


def untrained_w2(spec: Any, norm: Mapping[str, Any], n_out: int, *, seed: int = C.SEED, device: torch.device | str = "cpu"):
    return BM.build_model("W2", spec, norm, int(n_out), seed=int(seed), device=device)


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


def usage_metrics(model: RHICModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            logits, aux = model(batch, return_aux=True)
            h0 = aux["h0"]
            delta = aux["delta"]
            rhic = aux["rhic"]
            d = rhic["root_branch_count"]
            pc = rhic["root_path_count"]
            out: dict[str, Any] = {
                "rms_h0": rms(h0),
                "rms_delta": rms(delta),
                "R": (rms(delta) / rms(h0)) if rms(h0) else None,
                "rms_h": rms(aux["h"]),
                "rms_branch_mu": rms(rhic["branch_mu"]),
                "rms_branch_q": rms(rhic["branch_q"]),
                "rms_branch_value": rms(rhic["branch_value"]),
                "rms_mean_branch": rms(rhic["mean_branch"]),
                "rms_cross_branch": rms(rhic["cross_branch"]),
                "rms_z": rms(rhic["z"]),
                "rms_v": rms(rhic["v"]),
                "finite_fraction_h": float(torch.isfinite(aux["h"]).float().mean().item()),
                "finite_logits": bool(torch.isfinite(logits).all().item()),
                "logits_abs_max": float(logits.abs().max().item()),
                "n_atoms": int(h0.shape[0]),
                "n_paths": int(rhic["n_paths"]),
                "n_active_branches": int(rhic["n_branches"]),
                "n_directed_branches": 2 * int(batch["bond_index"].shape[0]),
            }
            out["branch_active_fraction"] = (
                float(out["n_active_branches"] / out["n_directed_branches"])
                if out["n_directed_branches"]
                else 0.0
            )
            gate = rhic["gate"]
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
            if pc.numel():
                nz = pc[pc > 0]
                out["path_counts"] = {
                    "mean": float(pc.float().mean().item()),
                    "p95": float(torch.quantile(pc.float(), 0.95).item()),
                    "max": int(pc.max().item()),
                    "zero_path_atoms": int((pc == 0).sum().item()),
                    "zero_path_fraction": float((pc == 0).float().mean().item()),
                    "mean_nonzero": float(nz.float().mean().item()) if nz.numel() else 0.0,
                }
                out["branch_counts"] = {
                    "mean": float(d.float().mean().item()),
                    "p95": float(torch.quantile(d.float(), 0.95).item()),
                    "max": int(d.max().item()),
                    "zero_branch_atoms": int((d == 0).sum().item()),
                    "zero_branch_fraction": float((d == 0).float().mean().item()),
                    "at_least_two_fraction": float((d >= 2).float().mean().item()),
                }
                # how often the two pair terms are structurally non-degenerate
                nb = rhic["branch_n"]
                out["multi_path_branch_fraction"] = (
                    float((nb >= 2).float().mean().item()) if nb.numel() else 0.0
                )
                out["q_nonzero_fraction"] = (
                    float((rhic["branch_q"].abs() > 0).float().mean().item()) if rhic["branch_q"].numel() else 0.0
                )
                out["cross_nonzero_fraction"] = (
                    float((rhic["cross_branch"].abs() > 0).float().mean().item()) if rhic["cross_branch"].numel() else 0.0
                )
            out["weight_norms"] = {
                name: float(p.detach().norm().item())
                for name, p in model.rhic.named_parameters()
                if name
                in (
                    "root_proj.weight",
                    "middle_proj.weight",
                    "end_proj.weight",
                    "bond_proj.weight",
                    "path1.weight",
                    "path2.weight",
                    "gate.weight",
                    "branch1.weight",
                    "branch2.weight",
                    "root1.weight",
                    "root2.weight",
                )
            }
            out["output_bias_norm"] = float(model.rhic.root2.bias.detach().norm().item())
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
    model: RHICModel,
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
            batch = P_collate(task, chunk.tolist())
            batch = to_device(batch, device)
            logits = model(batch, mode=mode, shuffle_seed=shuffle_seed)
            if logits.shape[1] != C.N_TASKS:
                raise RuntimeError(f"model returned {logits.shape[1]} outputs, expected {C.N_TASKS}")
            out.append(logits.detach().cpu().numpy().astype(np.float32))
    logits = np.concatenate(out, axis=0) if out else np.zeros((0, C.N_TASKS), dtype=np.float32)
    if logits.shape[0] != len(rows):
        raise RuntimeError(f"prediction rows {logits.shape[0]} != {len(rows)}")
    return logits


def P_collate(task: Any, rows: Sequence[int]) -> dict[str, Any]:
    if not hasattr(task, "cpc"):
        raise RuntimeError("task has no CPC structure cache")
    return P.collate_with_cpc(task, task.cpc, rows)


__all__ = [
    "P_collate",
    "RHICModel",
    "RHICModule",
    "block_fingerprints",
    "build_rhic_model",
    "compare_states",
    "initialize_rhic_modules",
    "parameter_audit",
    "predict_logits",
    "rms",
    "to_device",
    "untrained_w2",
    "usage_metrics",
]
