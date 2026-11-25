"""Arms A / B / C: the frozen LCRC graph with only the joint S2 channel changed.

Arm A  (ORIGINAL)  ``J(p,e) = joint_proj(path_u(p) * path_v(e))`` — untouched.
Arm B  (JOINT)     ``path_u``/``path_v``/``joint_proj`` are deleted and replaced
                   by one learnable matrix ``J_table`` of shape ``[|Omega|, r]``:
                   ``J(p,e) = J_table[index(p,e)]``.
Arm C  (ADDITIVE)  same Omega and output width, but ``J_table = U Theta`` with
                   ``Theta`` of shape ``[rank(X), r]``: the orthogonal marginal
                   projection of the joint table.  No residual term.

Initialisation is a *mathematical re-parameterisation of the same random draw*:
``J_table_0[i] = J_A(p_i, e_i)`` computed from arm A's own initial parameters for
that seed (including the original ``joint_proj`` bias), and
``Theta_0 = U^T J_table_0``.  No trained weight is loaded, nothing is distilled,
and every common parameter is bitwise identical between A, B and C.

Unseen ``(p, e)`` (never in Train) use one fixed Train-only fallback:
a least-squares marginal fit when both types are known and in the same support
component, otherwise the type-equal mean of the current table.  No new parameter.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sj_common as C
import sj_support as S

DM = C.DM
PM = C.PM

MODE_OMEGA = 0
MODE_MARGINAL_LS = 1
MODE_TYPE_MEAN = 2


# ---------------------------------------------------------------------------
# base (arm A) construction — the frozen historical paths, nothing else
# ---------------------------------------------------------------------------


def build_base(task_key: str, view: Any, seed: int, device) -> tuple[nn.Module, dict[str, Any]]:
    """Arm A at its frozen same-seed initial state (no trained weight loaded)."""
    t = C.TASKS[task_key]
    spec, norm = view.spec, view.train_norm
    if t["family"] == "ptb":
        model, info = PM.build_model(spec, norm, str(t["variant"]), int(seed), device, n_tasks=1)
        params = int(sum(p.numel() for p in model.parameters()))
        return model, {
            "family": "ptb",
            "variant": str(t["variant"]),
            "init_state_hash": info["init_state_hash"],
            "parameters": params,
            "model_info": {k: v for k, v in info.items() if not isinstance(v, dict)},
        }
    if t["family"] == "dissect":
        full = DM.build_full_state(spec, norm, int(seed))
        state = DM.remap_state(str(t["variant"]), full)
        model = DM.DissectLCRC(spec, norm, str(t["variant"]))
        model.load_state_dict(state, strict=True)
        model.to(device)
        return model, {
            "family": "dissect",
            "variant": str(t["variant"]),
            "init_state_hash": DM.state_fingerprint(state),
            "full_init_state_hash": DM.state_fingerprint(full),
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }
    raise KeyError(t["family"])


# ---------------------------------------------------------------------------
# arm A: the original joint function of a model
# ---------------------------------------------------------------------------


def original_joint(model: nn.Module, support: S.Support, *, dtype=torch.float64) -> torch.Tensor:
    """``J_A(p, e)`` for every Omega element, from arm A's current parameters."""
    local = model.local
    pu = local.path_u.weight.detach().to(dtype)
    pv = local.path_v.weight.detach().to(dtype)
    w = local.joint_proj.weight.detach().to(dtype)
    b = local.joint_proj.bias.detach().to(dtype)
    pi = torch.as_tensor(support.omega_p, dtype=torch.long, device=pu.device)
    ei = torch.as_tensor(support.omega_e, dtype=torch.long, device=pu.device)
    return F.linear(pu[pi] * pv[ei], w, b)


# ---------------------------------------------------------------------------
# arm B / C: support-indexed joint table
# ---------------------------------------------------------------------------


class SupportLocal(DM.LocalContent):
    """``LocalContent`` whose S2 joint channel is a support-indexed table."""

    def __init__(self, spec: Any, variant: str, cfg: Mapping[str, Any], support: S.Support, arm: str):
        super().__init__(spec, variant, cfg)
        if not self.s2_joint:
            raise RuntimeError("the joint channel must exist in the frozen variant")
        if arm not in (C.ARM_B, C.ARM_C):
            raise KeyError(arm)
        rank = int(self.rank)
        self.arm = str(arm)
        self.task_key = support.task_key
        self.rank_support = int(rank)
        self.n_p_types = int(support.n_p)
        self.n_e_types = int(support.n_e)
        self.n_nbr_codes = int(support.n_nbr_codes)
        # the three original joint modules are deleted, not re-used
        del self.path_u
        del self.path_v
        del self.joint_proj
        if arm == C.ARM_B:
            self.J_table = nn.Parameter(torch.zeros(int(support.n_omega), rank))
        else:
            self.Theta = nn.Parameter(torch.zeros(int(support.rank), rank))
        # fixed Train-only support tables (never trained, never in the state dict)
        self.register_buffer("support_index", torch.as_tensor(support.index_table, dtype=torch.long),
                             persistent=False)
        self.register_buffer("support_p_col", torch.as_tensor(support.p_col, dtype=torch.long), persistent=False)
        self.register_buffer("support_e_col", torch.as_tensor(support.e_col, dtype=torch.long), persistent=False)
        self.register_buffer("support_mode", torch.as_tensor(support.pair_mode, dtype=torch.long), persistent=False)
        self.register_buffer("support_U", torch.as_tensor(support.U, dtype=torch.float32), persistent=False)
        self.register_buffer("support_pinv", torch.as_tensor(support.pinvX, dtype=torch.float32), persistent=False)
        self.register_buffer("override_map", torch.zeros(0), persistent=False)
        self.n_path_codes = int(support.n_path_codes)

    # -- table ------------------------------------------------------------
    def joint_table(self) -> torch.Tensor:
        if self.arm == C.ARM_B:
            return self.J_table
        return self.support_U @ self.Theta

    def table_n_parameters(self) -> int:
        return int(self.J_table.numel() if self.arm == C.ARM_B else self.Theta.numel())

    # -- fallback / lookup ------------------------------------------------
    def complete_code_map(self) -> torch.Tensor:
        """Dense ``[n_path_codes * n_nbr_codes, r]`` code -> J map of the current table.

        Reproduces the forward rule exactly (same op order, same dtype), so an
        intervention table built from it is bitwise identical to the plain path.
        """
        j = self.joint_table()
        rank = j.shape[1]
        n_codes = int(self.n_path_codes) * int(self.n_nbr_codes)
        theta = self.support_pinv @ j  # [|P|+|E|, r]
        out = j.mean(dim=0).unsqueeze(0).expand(n_codes, rank).clone()
        mode = self.support_mode
        ls_rows = mode == MODE_MARGINAL_LS
        if bool(ls_rows.any()):
            codes = torch.nonzero(ls_rows, as_tuple=False).view(-1)
            pc = self.support_p_col[codes // self.n_nbr_codes]
            ec = self.support_e_col[codes % self.n_nbr_codes]
            out[codes] = theta[pc] + theta[self.n_p_types + ec]
        seen_rows = mode == MODE_OMEGA
        if bool(seen_rows.any()):
            codes = torch.nonzero(seen_rows, as_tuple=False).view(-1)
            out[codes] = j[self.support_index[codes]]
        return out

    def code_values(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.to(device=self.support_index.device, dtype=torch.long)
        j = self.joint_table()
        rank = j.shape[1]
        if int(self.override_map.numel()):
            return self.override_map[codes]
        idx = self.support_index[codes]
        out = j.new_zeros((int(codes.numel()), rank))
        seen = idx >= 0
        if bool(seen.any()):
            out[seen] = j[idx[seen]]
        unknown = torch.nonzero(~seen, as_tuple=False).view(-1)
        if int(unknown.numel()):
            modes = self.support_mode[codes[unknown]]
            ls = modes == MODE_MARGINAL_LS
            if bool(ls.any()):
                theta = self.support_pinv @ j
                rows = unknown[ls]
                cc = codes[rows]
                pc = self.support_p_col[cc // self.n_nbr_codes]
                ec = self.support_e_col[cc % self.n_nbr_codes]
                out[rows] = theta[pc] + theta[self.n_p_types + ec]
            rest = unknown[~ls]
            if int(rest.numel()):
                out[rest] = j.mean(dim=0)
        return out

    # -- LocalContent hooks ----------------------------------------------
    def _nonmarginal_part(self, kind: str, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if kind != "joint":
            return super()._nonmarginal_part(kind, batch)
        codes = batch["path_cat"].to(torch.long) * int(self.n_nbr_codes) + batch["path_end63"].to(torch.long)
        return self.code_values(codes)

    def _joint_agg(self, batch, a, agg):  # pragma: no cover - only used by marginal variants
        raise NotImplementedError("this round never switches the S2 aggregation mode")

    def set_override(self, table: torch.Tensor) -> None:
        self.override_map = table.detach().to(device=self.support_index.device, dtype=torch.float32)

    def clear_override(self) -> None:
        self.override_map = torch.zeros(0, device=self.support_index.device)


def attach_support(model: nn.Module, support: S.Support, arm: str) -> nn.Module:
    """Replace ``model.local`` by a ``SupportLocal`` (in place) and initialise the table."""
    local = model.local
    if not isinstance(local, DM.LocalContent):
        raise TypeError("unexpected local encoder")
    if isinstance(local, SupportLocal):
        raise RuntimeError("support already attached")
    target = next(model.parameters()).device
    new = SupportLocal(model.spec, model.variant, model.cfg, support, arm).to(target)
    src = {k: v for k, v in local.state_dict().items()}
    dst_keys = set(new.state_dict().keys())
    table_key = "J_table" if arm == C.ARM_B else "Theta"
    shared = {k: v for k, v in src.items() if k in dst_keys}
    missing = dst_keys - set(shared) - {table_key}
    if missing:
        raise RuntimeError(f"arm {arm}: no source for shared tensors {sorted(missing)}")
    with torch.no_grad():
        new.load_state_dict({**shared, table_key: new.state_dict()[table_key]}, strict=True)
    model.local = new
    mismatched = sorted(
        name for name, tensor in new.state_dict().items() if tensor.device != target
    )
    if mismatched:
        raise RuntimeError(f"arm {arm}: support module tensors left on another device: {mismatched}")
    return model


# ---------------------------------------------------------------------------
# arm factories
# ---------------------------------------------------------------------------


def make_arm(task_key: str, view: Any, support: S.Support, arm: str, seed: int, device) -> tuple[nn.Module, dict[str, Any]]:
    model, info = build_base(task_key, view, seed, device)
    info = dict(info)
    info["arm"] = arm
    info["parameters_base_A"] = int(sum(p.numel() for p in model.parameters()))
    if arm == C.ARM_A:
        info["joint_module"] = "path_u + path_v + joint_proj (frozen original)"
        info["parameters"] = info["parameters_base_A"]
        info["parameters_joint_module"] = joint_module_parameters(model)
        info["train_supported_joint_parameters"] = int(train_supported_parameters(model, support))
        return model, info
    # materialise A's joint function *before* the three modules are removed
    with torch.no_grad():
        j0 = original_joint(model, support, dtype=torch.float64).detach().cpu().numpy()
    attach_support(model, support, arm)
    local = model.local
    with torch.no_grad():
        j32 = torch.as_tensor(j0, dtype=torch.float32, device=local.support_index.device)
        if arm == C.ARM_B:
            local.J_table.copy_(j32)
        else:
            # Theta_0 = U^T J_table_0: the orthogonal marginal projection
            u = torch.as_tensor(support.U, dtype=torch.float64, device=j32.device)
            local.Theta.copy_((u.T @ j32.to(torch.float64)).to(torch.float32))
    table = extract_table(model, support)
    info.update(
        {
            "joint_module": (
                "J_table[|Omega|, r]" if arm == C.ARM_B else "Theta[rank(X), r] with J_table = U Theta"
            ),
            "parameters_joint_module": int(local.table_n_parameters()),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "train_supported_joint_parameters": int(train_supported_parameters(model, support)),
            "initial_table_rms": float(np.sqrt(np.mean(np.square(table)))),
            "initial_table_nonadditive_rms": float(
                np.sqrt(np.mean(np.square(nonadditive_part(support, table))))
            ),
        }
    )
    return model, info


# ---------------------------------------------------------------------------
# extraction / accounting
# ---------------------------------------------------------------------------


def extract_table(model: nn.Module, support: S.Support, *, dtype=torch.float64) -> np.ndarray:
    """Full Omega joint table of any arm (A recomputed, B/C from their parameter)."""
    local = model.local
    with torch.no_grad():
        if isinstance(local, SupportLocal):
            table = local.joint_table()
        else:
            table = original_joint(model, support, dtype=dtype)
    return table.detach().to(torch.float64).cpu().numpy()


def additive_part(support: S.Support, table: np.ndarray) -> np.ndarray:
    return support.P @ np.asarray(table, dtype=np.float64)


def nonadditive_part(support: S.Support, table: np.ndarray) -> np.ndarray:
    t = np.asarray(table, dtype=np.float64)
    return t - support.P @ t


def train_supported_parameters(model: nn.Module, support: S.Support) -> int:
    """Joint-channel parameters that actually have Train support."""
    local = model.local
    r = int(args_rank(model))
    if isinstance(local, SupportLocal):
        if local.arm == C.ARM_B:
            return int((support.index_table >= 0).sum()) * r
        return int(support.rank) * r
    used_p = int(support.p_types.size) * r
    used_e = int(support.e_types.size) * r
    proj = int(local.joint_proj.weight.numel() + local.joint_proj.bias.numel())
    return int(used_p + used_e + proj)


def args_rank(model: nn.Module) -> int:
    return int(model.local.rank)


def joint_module_parameters(model: nn.Module) -> int:
    local = model.local
    if isinstance(local, SupportLocal):
        return int(local.table_n_parameters())
    return int(
        local.path_u.weight.numel()
        + local.path_v.weight.numel()
        + local.joint_proj.weight.numel()
        + local.joint_proj.bias.numel()
    )


def joint_input_columns(model: nn.Module) -> list[int]:
    """Positions of the 32-d joint block inside the model's LocalMLP input."""
    cols = list(model.local.local_cols)
    wanted = set(C.JOINT_FULL_COLS)
    pos = [i for i, c in enumerate(cols) if c in wanted]
    if len(pos) != len(wanted):
        raise RuntimeError(f"joint block not found in the local column map (found {len(pos)} of {len(wanted)})")
    return pos


def joint_readout_matrix(model: nn.Module) -> np.ndarray:
    """``W_joint``: the LocalMLP first-layer columns that consume the joint block."""
    w = model.local.mlp.lins[0].weight.detach().to(torch.float64).cpu().numpy()
    return w[:, joint_input_columns(model)]


__all__ = [n for n in dir() if not n.startswith("_")]
