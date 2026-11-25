"""Support-indexed joint channels for the two reused backbones.

Only the S2 joint channel changes:

``C1_JOINT`` / ``W2_JOINT`` / ``FIXED1_JOINT``
    ``J(p,e) = J_table[index(p,e)]`` over the Train-only Omega.

``C1_ADD``
    ``J_table = U Theta`` with ``Theta`` of shape ``[rank(X), r]``.

Everything else is the frozen original module.  Initialisation is a
mathematical re-parameterisation of the same random draw: the table is
materialised from the original model's own untrained ``path_u``/``path_v``/
``joint_proj`` (including the original bias).  No trained weight is loaded and
no teacher is used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jt_common as C

_PATHS_READY = False


def prepare_imports() -> None:
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (str(C.SJ_CODE), str(C.READOUT_CODE), str(C.BS_CODE), str(C.DSCR_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


def modules() -> dict[str, Any]:
    prepare_imports()
    import sj_model
    import sj_support

    return {"sj_model": sj_model, "sj_support": sj_support}


ARM_TO_LOCAL = {C.ARM_LIPO_JOINT: "B", C.ARM_LIPO_ADD: "C", C.ARM_W2_JOINT: "B", C.ARM_FIXED1_JOINT: "B"}


# ---------------------------------------------------------------------------
# joint table arithmetic shared by both local-content classes
# ---------------------------------------------------------------------------


def materialize_joint(local: nn.Module, support: Any, *, dtype=torch.float64) -> torch.Tensor:
    """Original ``J(p,e)`` for every Omega element from the current local module."""
    pu = local.path_u.weight.detach().to(dtype)
    pv = local.path_v.weight.detach().to(dtype)
    w = local.joint_proj.weight.detach().to(dtype)
    b = local.joint_proj.bias.detach().to(dtype)
    pi = torch.as_tensor(support.omega_p, dtype=torch.long, device=pu.device)
    ei = torch.as_tensor(support.omega_e, dtype=torch.long, device=pu.device)
    return F.linear(pu[pi] * pv[ei], w, b)


def _initial_table(support: Any, j0: torch.Tensor, arm_local: str, device: torch.device) -> torch.Tensor:
    j32 = torch.as_tensor(j0, dtype=torch.float32, device=device)
    if arm_local == "B":
        return j32
    u = torch.as_tensor(support.U, dtype=torch.float64, device=device)
    theta = (u.T @ j32.to(torch.float64)).to(torch.float32)
    return theta


def _copy_shared(old: nn.Module, new: nn.Module, table_key: str) -> None:
    src = old.state_dict()
    dst_keys = set(new.state_dict().keys())
    shared = {k: v for k, v in src.items() if k in dst_keys}
    missing = dst_keys - set(shared) - {table_key}
    if missing:
        raise RuntimeError(f"no source for shared tensors {sorted(missing)}")
    new.load_state_dict({**shared, table_key: new.state_dict()[table_key]}, strict=True)


# ---------------------------------------------------------------------------
# TU local content (PCBA W2 / FIXED1)
# ---------------------------------------------------------------------------


class SupportTULocal(nn.Module):
    """``TULocalContent`` with the S2 joint channel replaced by a lookup table."""

    def __init__(self, spec: Any, variant: str, cfg: Mapping[str, Any], support: Any, arm_local: str):
        super().__init__()
        import tu_model as TU

        if arm_local not in ("B", "C"):
            raise KeyError(arm_local)
        base = TU.TULocalContent(spec, variant, cfg)
        # Move every shared module/tensor into this module without re-drawing
        # any random number (state transfer happens in ``attach_support``).
        self.variant = base.variant
        self.spec = base.spec
        self.rank = int(base.rank)
        self.a_out = int(base.a_out)
        self.use_b_mean = bool(base.use_b_mean)
        self.layout = base.layout
        self.local_cols = base.local_cols
        self.nbr_emb = base.nbr_emb
        self.path_emb = base.path_emb
        self.end_emb = base.end_emb
        self.close_emb = base.close_emb
        self.path_gate = base.path_gate
        self.mlp = base.mlp
        self.norm = base.norm
        self.arm_local = str(arm_local)
        self.n_path_codes = int(spec.n_path_codes)
        self.n_nbr_codes = int(spec.n_nbr_codes)
        self.task_key = str(support.task_key)
        self.n_p_types = int(support.n_p)
        self.n_e_types = int(support.n_e)
        if arm_local == "B":
            self.J_table = nn.Parameter(torch.zeros(int(support.n_omega), self.rank))
        else:
            self.Theta = nn.Parameter(torch.zeros(int(support.rank), self.rank))
        self.register_buffer("support_index", torch.as_tensor(support.index_table, dtype=torch.long), persistent=False)
        self.register_buffer("support_p_col", torch.as_tensor(support.p_col, dtype=torch.long), persistent=False)
        self.register_buffer("support_e_col", torch.as_tensor(support.e_col, dtype=torch.long), persistent=False)
        self.register_buffer("support_mode", torch.as_tensor(support.pair_mode, dtype=torch.long), persistent=False)
        self.register_buffer("support_U", torch.as_tensor(support.U, dtype=torch.float32), persistent=False)
        self.register_buffer("support_pinv", torch.as_tensor(support.pinvX, dtype=torch.float32), persistent=False)
        self.register_buffer("override_map", torch.zeros(0), persistent=False)

    # -- table ------------------------------------------------------------
    def joint_table(self) -> torch.Tensor:
        if self.arm_local == "B":
            return self.J_table
        return self.support_U @ self.Theta

    def table_n_parameters(self) -> int:
        return int(self.J_table.numel() if self.arm_local == "B" else self.Theta.numel())

    # -- lookup / fallback (identical rule to sj_model.SupportLocal) -------
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
            ls = modes == 1
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

    def complete_code_map(self) -> torch.Tensor:
        j = self.joint_table()
        rank = j.shape[1]
        n_codes = int(self.n_path_codes) * int(self.n_nbr_codes)
        theta = self.support_pinv @ j
        out = j.mean(dim=0).unsqueeze(0).expand(n_codes, rank).clone()
        mode = self.support_mode
        ls_rows = mode == 1
        if bool(ls_rows.any()):
            codes = torch.nonzero(ls_rows, as_tuple=False).view(-1)
            pc = self.support_p_col[codes // self.n_nbr_codes]
            ec = self.support_e_col[codes % self.n_nbr_codes]
            out[codes] = theta[pc] + theta[self.n_p_types + ec]
        seen_rows = mode == 0
        if bool(seen_rows.any()):
            codes = torch.nonzero(seen_rows, as_tuple=False).view(-1)
            out[codes] = j[self.support_index[codes]]
        return out

    def set_override(self, table: torch.Tensor) -> None:
        self.override_map = table.detach().to(device=self.support_index.device, dtype=torch.float32)

    def clear_override(self) -> None:
        self.override_map = torch.zeros(0, device=self.support_index.device)

    # -- exact copy of ``TULocalContent.forward`` with one changed line ----
    def forward(self, batch, a, b, scal):
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
            codes = batch["path_cat"].to(torch.long) * int(self.n_nbr_codes) + batch["path_end63"].to(torch.long)
            path_joint = self.code_values(codes)
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
            gate = self.path_gate(batch["path_cat"])
            content_agg = a.new_zeros((n_atoms, width)).index_add_(
                0, batch["path_atom"], a[batch["path_end"]] * gate
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


# ---------------------------------------------------------------------------
# attach / detach
# ---------------------------------------------------------------------------


def attach_support(owner: nn.Module, attr: str, support: Any, arm: str) -> nn.Module:
    """Replace ``owner.<attr>`` in place and materialise the initial table."""
    import dissect_model as DM

    arm_local = ARM_TO_LOCAL[arm]
    old = getattr(owner, attr)
    mods = modules()
    if isinstance(old, (mods["sj_model"].SupportLocal, SupportTULocal)):
        raise RuntimeError("support already attached")
    if not isinstance(old, (DM.LocalContent,)) and not hasattr(old, "joint_proj"):
        raise TypeError(f"unexpected local content type {type(old)!r}")
    target = next(old.parameters()).device
    j0 = materialize_joint(old, support, dtype=torch.float64).detach().cpu()

    if isinstance(old, DM.LocalContent):
        mods = modules()
        if not hasattr(owner, "spec") or not hasattr(owner, "cfg"):
            raise RuntimeError("LocalContent owner must expose spec and cfg")
        new = mods["sj_model"].SupportLocal(owner.spec, old.variant, owner.cfg, support, arm_local)
        table_key = "J_table" if arm_local == "B" else "Theta"
    else:
        cfg = getattr(owner, "cfg", None)
        if cfg is None:
            raise RuntimeError("TULocalContent owner has no cfg")
        new = SupportTULocal(owner.spec, old.variant, cfg, support, arm_local)
        table_key = "J_table" if arm_local == "B" else "Theta"

    new = new.to(target)
    with torch.no_grad():
        _copy_shared(old, new, table_key)
        table = _initial_table(support, j0, arm_local, target)
        if arm_local == "B":
            new.J_table.copy_(table)
        else:
            new.Theta.copy_(table)
    setattr(owner, attr, new)
    return new


# ---------------------------------------------------------------------------
# frozen model factories
# ---------------------------------------------------------------------------


def build_lipo_model(bundle: Any, support: Any, arm: str) -> tuple[nn.Module, dict[str, Any]]:
    if arm not in C.LIPO_ARMS:
        raise KeyError(arm)
    prepare_imports()
    import cr_common as CC
    import cr_models as CM

    spec = bundle.spec if hasattr(bundle, "spec") else bundle["spec"]
    norm = bundle.norm if hasattr(bundle, "norm") else bundle["norm"]
    canonical = CM.canonical_c1_state(spec, norm)
    CC.set_global_seeds(C.SEED)
    model = CM.ArmA(spec, norm)
    model.load_canonical(canonical)
    base_params = int(sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        j0 = materialize_joint(model.trunk.local, support, dtype=torch.float64).detach().cpu()
    local = attach_support(model.trunk, "local", support, arm)
    info = {
        "arm": arm,
        "family": "lipo",
        "base_arm": "C1",
        "base_parameters": base_params,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "joint_module_parameters": int(local.table_n_parameters()),
        "train_supported_joint_parameters": int(
            (support.n_omega if ARM_TO_LOCAL[arm] == "B" else support.rank) * int(local.rank)
        ),
        "initial_table_rms": float(np.sqrt(np.mean(np.square(j0.numpy())))),
        "canonical_state_hash": CC.sha256_state(canonical) if hasattr(CC, "sha256_state") else None,
    }
    return model, info


def build_pcba_model(task: Any, support: Any, arm: str) -> tuple[nn.Module, dict[str, Any]]:
    if arm not in C.PCBA_ARMS:
        raise KeyError(arm)
    prepare_imports()
    import bs_model as BM
    import ds_model as DM

    if arm == C.ARM_W2_JOINT:
        model, audit = BM.build_model(
            "W2", task.spec, task.norm, int(C.PCBA_N_TASKS), seed=C.SEED, device="cpu", verify_parameters=True
        )
        base_arm = "W2"
    else:
        model, audit = DM.build_ds_model(
            task.spec, task.norm, int(C.PCBA_N_TASKS), arm="FIXED1", seed=C.SEED, device="cpu"
        )
        base_arm = "FIXED1"
    base_params = int(sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        j0 = materialize_joint(model.local, support, dtype=torch.float64).detach().cpu()
    local = attach_support(model, "local", support, arm)
    info = {
        "arm": arm,
        "family": "pcba",
        "base_arm": base_arm,
        "base_parameters": base_params,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "joint_module_parameters": int(local.table_n_parameters()),
        "train_supported_joint_parameters": int(
            (support.n_omega if ARM_TO_LOCAL[arm] == "B" else support.rank) * int(local.rank)
        ),
        "initial_table_rms": float(np.sqrt(np.mean(np.square(j0.numpy())))),
        "base_audit": audit,
    }
    return model, info


# ---------------------------------------------------------------------------
# accounting / comparisons
# ---------------------------------------------------------------------------


def joint_module_names(model: nn.Module) -> set[str]:
    return {n for n, _ in model.named_parameters() if n.endswith("J_table") or n.endswith("Theta")}


def common_parameter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name, value in model.state_dict().items():
        if any(tag in name for tag in (".path_u.", ".path_v.", ".joint_proj.")):
            continue
        if name.endswith("J_table") or name.endswith("Theta"):
            continue
        out[name] = value.detach().cpu().clone()
    return out


def compare_parameter_states(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys = sorted(set(a) | set(b))
    diffs = {}
    equal = True
    for key in keys:
        if key not in a or key not in b:
            diffs[key] = "missing"
            equal = False
            continue
        same = torch.equal(a[key].float(), b[key].float())
        if not same:
            diffs[key] = float((a[key].float() - b[key].float()).abs().max())
            equal = False
    return {"all_equal": bool(equal), "n_keys": len(keys), "mismatches": diffs}


def table_of(model: nn.Module) -> np.ndarray:
    local = model.trunk.local if hasattr(model, "trunk") else model.local
    return local.joint_table().detach().to(torch.float64).cpu().numpy()


def joint_input_columns(model: nn.Module) -> list[int]:
    local = model.trunk.local if hasattr(model, "trunk") else model.local
    if hasattr(local, "layout"):
        lo, hi = local.layout["s2_joint"]
        return list(range(int(lo), int(hi)))
    # ``dissect_model.LocalContent``: ``local_cols`` are FULL-layout indices.
    import sj_common as SC

    cols = list(local.local_cols)
    wanted = set(SC.JOINT_FULL_COLS)
    pos = [i for i, c in enumerate(cols) if c in wanted]
    if len(pos) != len(wanted):
        raise RuntimeError(f"joint block not found in the local column map (found {len(pos)} of {len(wanted)})")
    return pos


def joint_readout_matrix(model: nn.Module) -> np.ndarray:
    local = model.trunk.local if hasattr(model, "trunk") else model.local
    w = local.mlp.lins[0].weight.detach().to(torch.float64).cpu().numpy()
    return w[:, joint_input_columns(model)]


__all__ = [name for name in dir() if not name.startswith("_")]
