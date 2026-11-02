"""The three fixed models: LCRC-FULL, LCRC-C1 (no b_mean) and GIN+global.

``FULL`` is the existing LCRC-A computational graph (one local encode, per-node
contribution ``f``, global correction ``rho``) with TU input widths.  ``C1`` is
*only* the deletion of the 96-d ``b_mean = b_sum/deg`` block: no other module,
column or hyper-parameter changes.  ``C1`` is initialised from the seed-0
``FULL`` state by field mapping, with the local-MLP input columns sliced
according to the **actual TU layout** (computed at runtime, never hard-coded).

``GIN+global`` is the fixed application reference: node linear projection,
five residual GIN layers (``train_eps``, initial epsilon 0), sum/mean pooling
concatenated with the same 28-d TU global vector and ``log1p(n)``, then a
157-64-C SiLU MLP.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tu_common as C
from feature_spec import FeatureSpec


class MLP(nn.Module):
    """Plain MLP with frozen LCRC-A ``lins`` naming (SiLU between layers)."""

    def __init__(self, dims: list[int]):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
        return x


# ---------------------------------------------------------------------------
# local-input layout (computed from the TU spec, not copied from lipo)
# ---------------------------------------------------------------------------


def local_parts(spec: FeatureSpec, variant: str, cfg: Mapping[str, Any] | None = None) -> list[tuple[str, int]]:
    c = dict(C.MODEL_CONFIG if cfg is None else {**C.MODEL_CONFIG, **dict(cfg)})
    a_out = int(c["atom_out"])
    rank = int(c["stat_rank"])
    if variant not in ("FULL", "C1"):
        raise KeyError(variant)
    parts: list[tuple[str, int]] = [("a", a_out), ("b_sum", a_out)]
    if variant == "FULL":
        parts.append(("b_mean", a_out))
    parts.extend(
        [
            ("s1_joint", rank),
            ("s2_type", rank),
            ("s2_end", rank),
            ("s2_joint", rank),
            ("s2_closure", 8),
            ("s2_content", a_out),
            ("atom_scalar", spec.n_scalars),
        ]
    )
    return parts


def local_layout(spec: FeatureSpec, variant: str, cfg: Mapping[str, Any] | None = None) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    off = 0
    for name, dim in local_parts(spec, variant, cfg):
        out[name] = (off, off + int(dim))
        off += int(dim)
    return out


def local_columns(spec: FeatureSpec, variant: str, cfg: Mapping[str, Any] | None = None) -> tuple[int, ...]:
    cols: list[int] = []
    for _name, (lo, hi) in local_layout(spec, variant, cfg).items():
        cols.extend(range(lo, hi))
    return tuple(cols)


def rho_layout(spec: FeatureSpec, width: int) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    off = 0
    for name, dim in (("g", spec.global_dim), ("mean_h", width), ("mean_h2", width), ("log1p_n", 1)):
        out[name] = (off, off + int(dim))
        off += int(dim)
    return out


# ---------------------------------------------------------------------------
# LCRC local content encoder with the b_mean branch flag
# ---------------------------------------------------------------------------


class TULocalContent(nn.Module):
    def __init__(self, spec: FeatureSpec, variant: str, cfg: Mapping[str, Any] | None = None):
        super().__init__()
        c = dict(C.MODEL_CONFIG if cfg is None else {**C.MODEL_CONFIG, **dict(cfg)})
        self.variant = variant
        self.spec = spec
        self.rank = int(c["stat_rank"])
        self.a_out = int(c["atom_out"])
        self.use_b_mean = variant == "FULL"
        self.layout = local_layout(spec, variant, c)
        self.local_cols = local_columns(spec, variant, c)
        self.nbr_emb = nn.Embedding(spec.n_nbr_codes, self.rank)
        self.path_emb = nn.Embedding(spec.n_path_codes, self.rank)
        self.end_emb = nn.Embedding(spec.n_nbr_codes, self.rank)
        self.path_u = nn.Embedding(spec.n_path_codes, self.rank)
        self.path_v = nn.Embedding(spec.n_nbr_codes, self.rank)
        self.joint_proj = nn.Linear(self.rank, self.rank)
        self.close_emb = nn.Embedding(2, 8)
        self.path_gate = nn.Embedding(spec.n_path_codes, 1)
        nn.init.ones_(self.path_gate.weight)
        self.mlp = MLP([len(self.local_cols), int(c["local_hidden"]), int(c["width"])])
        self.norm = nn.LayerNorm(int(c["width"]))

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        scal: torch.Tensor,
    ) -> torch.Tensor:
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
# LCRC model (FULL / C1)
# ---------------------------------------------------------------------------


class TULCRC(nn.Module):
    def __init__(
        self,
        spec: FeatureSpec,
        norm: Mapping[str, Any],
        n_classes: int,
        variant: str = "FULL",
        cfg: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if variant not in ("FULL", "C1"):
            raise KeyError(variant)
        self.spec = spec
        self.variant = variant
        self.n_classes = int(n_classes)
        self.cfg = dict(C.MODEL_CONFIG if cfg is None else {**C.MODEL_CONFIG, **dict(cfg)})
        a_out = int(self.cfg["atom_out"])
        width = int(self.cfg["width"])
        self.atom_encoder = MLP([spec.atom_feat_dim, int(self.cfg["atom_hidden"]), a_out])
        self.bond_encoder = MLP(
            [3 * a_out + spec.bond_feat_dim, int(self.cfg["bond_hidden"]), int(self.cfg["bond_out"])]
        )
        self.local = TULocalContent(spec, variant, self.cfg)
        self.drop = nn.Dropout(float(C.RECIPE["dropout"]))
        self.f = MLP([width, width, self.n_classes])
        self.rho = MLP([spec.global_dim + 2 * width + 1, int(self.cfg["rho_hidden"]), self.n_classes])
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))
        for name, expected in (
            ("global_mean", spec.global_dim),
            ("global_scale", spec.global_dim),
            ("scalar_mean", spec.n_scalars),
            ("scalar_scale", spec.n_scalars),
        ):
            if tuple(getattr(self, name).shape) != (expected,):
                raise ValueError(f"norm {name} has shape {tuple(getattr(self, name).shape)} != {(expected,)}")

    def _encode(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        else:
            bond_input = torch.zeros(
                (0, 3 * a.shape[1] + batch["bond_feat"].shape[1]), dtype=a.dtype, device=a.device
            )
        b = self.bond_encoder(bond_input)
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        return a, b, scal

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        a, b, scal = self._encode(batch)
        h = self.drop(self.local(batch, a, b, scal))  # exactly one dropout call, shared by both readouts
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib = h.new_zeros(n_graphs, self.n_classes).index_add_(0, atom_batch, self.f(h))
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in)
        logits = contrib + rho
        if return_aux:
            return logits, {"h": h, "contrib": contrib, "rho": rho, "mean_h": mean_h, "mean_h2": mean_h2}
        return logits


# ---------------------------------------------------------------------------
# state construction / remapping
# ---------------------------------------------------------------------------


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def module_breakdown(model: nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, child in model.named_children():
        out[name] = parameter_count(child)
    out["total"] = parameter_count(model)
    return out


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        arr = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        h.update(arr.tobytes())
    return h.hexdigest()


def build_full_state(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    n_classes: int,
    seed: int = C.TRAINING_SEED,
) -> dict[str, torch.Tensor]:
    """One complete seed-0 FULL initial state (CPU RNG only)."""
    torch.manual_seed(int(seed))
    model = TULCRC(spec, norm, int(n_classes), "FULL")
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def remap_state(
    variant: str,
    full: Mapping[str, torch.Tensor],
    spec: FeatureSpec,
    n_classes: int,
) -> dict[str, torch.Tensor]:
    """Copy the FULL state into ``variant`` using the actual TU column layout."""
    if variant == "FULL":
        return {k: v.detach().cpu().clone() for k, v in full.items()}
    layout_full = local_layout(spec, "FULL")
    layout_v = local_layout(spec, variant)
    removed = set(range(*layout_full["b_mean"]))
    keep_full = [i for i in range(layout_full["atom_scalar"][1]) if i not in removed]
    expected_keep: list[int] = []
    for name in layout_v:
        lo, hi = layout_full[name]
        expected_keep.extend(range(lo, hi))
    if keep_full != expected_keep:
        raise RuntimeError("variant column layout is not FULL minus b_mean")
    out: dict[str, torch.Tensor] = {}
    for key, value in full.items():
        out[key] = value.detach().cpu().clone()
    w0 = full["local.mlp.lins.0.weight"]
    out["local.mlp.lins.0.weight"] = w0[:, keep_full].detach().cpu().clone()
    # Guard: the remapped state must load into a variant model of the same shape.
    expected_in = len(local_columns(spec, variant))
    if out["local.mlp.lins.0.weight"].shape[1] != expected_in:
        raise RuntimeError("remapped local MLP width mismatch")
    if out["rho.lins.0.weight"].shape[0] != int(C.MODEL_CONFIG["rho_hidden"]):
        raise RuntimeError("unexpected rho hidden width")
    return out


def shared_state_equal(
    variant: str,
    full: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    spec: FeatureSpec,
) -> dict[str, Any]:
    """Prove every shared parameter/buffer is bitwise identical to FULL."""
    checks: dict[str, bool] = {}
    for key in full:
        if key == "local.mlp.lins.0.weight":
            continue
        checks[key] = bool(torch.equal(state[key].cpu(), full[key].cpu()))
    if variant == "FULL":
        checks["local.mlp.lins.0.weight"] = bool(
            torch.equal(state["local.mlp.lins.0.weight"].cpu(), full["local.mlp.lins.0.weight"].cpu())
        )
    else:
        layout_full = local_layout(spec, "FULL")
        removed = set(range(*layout_full["b_mean"]))
        keep = [i for i in range(layout_full["atom_scalar"][1]) if i not in removed]
        checks["local.mlp.lins.0.weight"] = bool(
            torch.equal(
                state["local.mlp.lins.0.weight"].cpu(),
                full["local.mlp.lins.0.weight"][:, keep].cpu(),
            )
        )
    return {"all_equal": all(checks.values()), "n_checked": len(checks), "checks": checks}


def build_model(
    prepared: Mapping[str, Any],
    model_name: str,
    seed: int = C.TRAINING_SEED,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Build one model with the frozen initialisation for a formal run."""
    spec = prepared["spec"]
    norm = prepared["norm"]
    n_classes = int(prepared["n_classes"])
    device = torch.device(device)
    info: dict[str, Any] = {"model": model_name, "seed": int(seed)}
    if model_name in ("FULL", "C1"):
        full_state = build_full_state(spec, norm, n_classes, int(seed))
        init_state = remap_state(model_name, full_state, spec, n_classes)
        model = TULCRC(spec, norm, n_classes, model_name)
        model.load_state_dict(init_state, strict=True)
        info.update(
            {
                "init_state_hash": state_fingerprint(init_state),
                "full_init_state_hash": state_fingerprint(full_state),
                "shared_with_full": shared_state_equal(model_name, full_state, init_state, spec),
                "local_layout": {k: list(v) for k, v in local_layout(spec, model_name).items()},
                "local_input_dim": len(local_columns(spec, model_name)),
                "rho_input_dim": spec.global_dim + 2 * int(C.MODEL_CONFIG["width"]) + 1,
            }
        )
    elif model_name == "GIN":
        torch.manual_seed(int(seed))
        model = GINPlusGlobal(spec, norm, n_classes)
        info["init_state_hash"] = state_fingerprint(model.state_dict())
    else:
        raise KeyError(model_name)
    model.to(device)
    info["parameters"] = parameter_count(model)
    info["parameters_breakdown"] = module_breakdown(model)
    info["state_hash"] = state_fingerprint(model.state_dict())
    info["input_widths"] = {
        "atom_feat": spec.atom_feat_dim,
        "bond_feat": spec.bond_feat_dim,
        "global": spec.global_dim,
        "scalars": spec.n_scalars,
        "n_nbr_codes": spec.n_nbr_codes,
        "n_path_codes": spec.n_path_codes,
    }
    return model, info


# ---------------------------------------------------------------------------
# fixed GIN + global reference
# ---------------------------------------------------------------------------


class GINLayer(nn.Module):
    def __init__(self, hidden: int, mlp_hidden: int, dropout: float):
        super().__init__()
        self.mlp = MLP([int(hidden), int(mlp_hidden), int(hidden)])
        self.eps = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(int(hidden))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # ``edge_index`` is an [M, 2] edge list (u, v) repeated in both directions.
        src, dst = edge_index[:, 0], edge_index[:, 1]
        agg = x.new_zeros(x.shape).index_add_(0, dst, x[src])
        out = self.mlp((1.0 + self.eps) * x + agg)
        out = self.drop(F.silu(self.norm(out)))
        return out + x


class GINPlusGlobal(nn.Module):
    def __init__(
        self,
        spec: FeatureSpec,
        norm: Mapping[str, Any],
        n_classes: int,
        cfg: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        c = dict(C.GIN_CONFIG if cfg is None else {**C.GIN_CONFIG, **dict(cfg)})
        self.spec = spec
        self.n_classes = int(n_classes)
        self.cfg = c
        hidden = int(c["hidden"])
        self.node_proj = nn.Linear(spec.atom_feat_dim, hidden)
        self.layers = nn.ModuleList(
            [GINLayer(hidden, int(c["mlp_hidden"]), float(c["dropout"])) for _ in range(int(c["n_layers"]))]
        )
        for layer in self.layers:
            nn.init.constant_(layer.eps, float(c["eps_init"]))
        readout_in = 2 * hidden + spec.global_dim + 1
        if readout_in != int(c["readout_in"]):
            raise RuntimeError(f"GIN readout width {readout_in} != frozen {c['readout_in']}")
        self.readout = MLP([readout_in, int(c["readout_hidden"]), self.n_classes])
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        x = self.node_proj(batch["atom_feat"])
        idx = batch["bond_index"]
        if idx.numel():
            edge_index = torch.cat([idx, idx[:, [1, 0]]], dim=0)
        else:
            edge_index = idx.new_zeros((2, 0))
        for layer in self.layers:
            x = layer(x, edge_index)
        count = x.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=x.dtype))
        sum_pool = x.new_zeros((n_graphs, x.shape[1])).index_add_(0, atom_batch, x)
        mean_pool = sum_pool / count.clamp(min=1.0).unsqueeze(1)
        g = (batch["global"] - self.global_mean) / self.global_scale
        z = torch.cat([sum_pool, mean_pool, g, torch.log1p(count).unsqueeze(1)], dim=1)
        return self.readout(z)


__all__ = [
    "MLP",
    "TULocalContent",
    "TULCRC",
    "GINLayer",
    "GINPlusGlobal",
    "local_parts",
    "local_layout",
    "local_columns",
    "rho_layout",
    "parameter_count",
    "module_breakdown",
    "state_fingerprint",
    "build_full_state",
    "remap_state",
    "shared_state_equal",
    "build_model",
]
