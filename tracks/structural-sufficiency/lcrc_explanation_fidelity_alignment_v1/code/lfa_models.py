"""Frozen model loading and *batched* gated forwards for both families.

Nothing is trained here.  Every parameter is frozen, every module is in
``eval()``, and the only differentiable leaf during a discovery pass is the
per-physical-bond gate matrix.

The two interventions are the ones already verified by the source rounds:

* LCRC -> ``CONDITIONAL_S1_S2_EVIDENCE_MASK`` (``as_gate.gated_parts``);
* GINE -> one shared physical-bond message gate in every existing layer.

The GINE batched forward mirrors :func:`xr_models.gine_gated_parts` exactly and
is cross-checked against it (``gates.py``) so the new batching cannot change the
intervention semantics.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import lfa_common as C

XM = None
XD = None
AS = None
AG = None


def _lazy() -> None:
    global XM, XD, AS, AG
    if XM is not None:
        return
    C.install_source_paths()
    import xr_models

    mods = C.source_modules()
    XM = xr_models
    XD = mods["gr_data"]
    AS = mods["as_search"]
    AG = mods["as_gate"]


def raw_bundle():
    _lazy()
    return XM.bundle(C.DATASET)


def build_occ(b, split: str):
    _lazy()
    mods = C.source_modules()
    return mods["as_data"].build_split_occ(b.caches[split], b.spec, b.dataset, split)


# ---------------------------------------------------------------------------
# checkpoint identity / loading
# ---------------------------------------------------------------------------


def _load_state(path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise RuntimeError(f"{path}: checkpoint has no model_state")
    return payload


def load_model(name: str, b, *, device: str | torch.device = "cpu"):
    _lazy()
    path = C.checkpoint_paths()[name]
    if not path.is_file():
        raise RuntimeError(f"CHECKPOINT_UNAVAILABLE {name}: {path} not found")
    actual = C.sha256_file(path)
    expected = C.EXPECTED_CHECKPOINT_SHA256[name]
    if actual != expected:
        raise RuntimeError(f"CHECKPOINT_UNAVAILABLE {name}: sha256 {actual} != {expected}")
    model = XM.build_untrained(name, b)
    payload = _load_state(path)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, payload


def checkpoint_identity(name: str, b) -> dict[str, Any]:
    _lazy()
    path = C.checkpoint_paths()[name]
    payload = _load_state(path) if path.is_file() else {}
    if path.is_file():
        model = XM.build_untrained(name, b)
        model.load_state_dict(payload["model_state"], strict=True)
        n_params = int(sum(p.numel() for p in model.parameters()))
        state_hash = C.sha256_state(model.state_dict())
        sha = C.sha256_file(path)
    else:
        n_params, state_hash, sha = None, None, None
    meta = C.EXPECTED_CHECKPOINT_META[name]
    return {
        "name": name,
        "path": C.relpath(path),
        "exists": bool(path.is_file()),
        "sha256": sha,
        "sha256_expected": C.EXPECTED_CHECKPOINT_SHA256[name],
        "sha256_match": bool(sha == C.EXPECTED_CHECKPOINT_SHA256[name]),
        "state_sha256": state_hash,
        "n_parameters": n_params,
        "n_parameters_expected": int(meta["n_parameters"]),
        "parameter_count_match": bool(n_params == int(meta["n_parameters"])),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_epoch_expected": int(meta["epoch"]),
        "epoch_match": bool(payload.get("epoch") == int(meta["epoch"])),
        "checkpoint_valid_rmse_raw": payload.get("valid_rmse_raw"),
        "valid_rmse_expected": meta["valid_rmse_raw"],
        "source_path": C.SOURCE_CHECKPOINT_PATHS[name],
    }


def parameter_hash(model) -> str:
    return C.sha256_state({k: v.detach().cpu() for k, v in model.state_dict().items()})


@torch.no_grad()
def predict_split(model, b_eval, split: str, *, device="cpu", batch_size: int = 256) -> dict[str, Any]:
    _lazy()
    order, row_ids, y_raw = XD.eval_order(b_eval, split)
    preds: list[np.ndarray] = []
    for start in range(0, order.size, int(batch_size)):
        chunk = order[start : start + int(batch_size)]
        batch = XD.collate(b_eval, split, chunk)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred_std = model(batch).detach().cpu().numpy().astype(np.float64).reshape(-1)
        preds.append(pred_std * float(b_eval.scaler["std"]) + float(b_eval.scaler["mean"]))
    pred_raw = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float64)
    return {
        "row_id": np.asarray(row_ids, dtype=np.int64),
        "y_raw": np.asarray(y_raw, dtype=np.float64),
        "pred_raw": pred_raw,
        "rmse_raw": C.rmse(pred_raw, y_raw),
        "mae_raw": C.mae(pred_raw, y_raw),
        "n_rows": int(row_ids.size),
    }


# ---------------------------------------------------------------------------
# single-molecule gradient passes (discovery fields)
# ---------------------------------------------------------------------------


def lcrc_gradient(model, cache, occ, row: int, b_task) -> dict[str, Any]:
    _lazy()
    arm = model.arm
    gb = C.source_modules()["as_data"].collate_gated(cache, [int(row)], occ)
    gp = AS.gradient_pass(arm, gb, sigma=float(b_task.scaler["std"]), mu=float(b_task.scaler["mean"]))
    return {
        "grad_transformed": np.asarray(gp["grad_transformed"][0], dtype=np.float64),
        "contrib_atom_std": np.asarray(gp["contrib_atom_std"], dtype=np.float64),
        "pred_raw": float(gp["pred_raw"][0]),
        "pred_std": float(gp["pred_std"][0]),
        "local_sum_raw": float(gp["local_sum_raw"][0]),
        "rho_raw": float(gp["rho_raw"][0]),
    }


def gine_gradient(model, batch, b_task) -> dict[str, Any]:
    _lazy()
    gp = XM.gine_gradient_pass(model, batch, b_task)
    return {
        "grad_transformed": np.asarray(gp["grad_transformed"][0], dtype=np.float64),
        "contrib_atom_std": None,
        "pred_raw": float(gp["pred_raw"][0]),
        "pred_std": float(gp["pred_std"][0]),
        "local_sum_raw": None,
        "rho_raw": None,
    }


def gine_single_batch(b, split: str, cache_row: int, *, device="cpu") -> dict[str, Any]:
    _lazy()
    batch = XD.collate(b, split, [int(cache_row)])
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# batched gated forwards (many mask variants of ONE molecule)
# ---------------------------------------------------------------------------


def lcrc_batch(cache, row: int, occ, n_graphs: int):
    _lazy()
    mods = C.source_modules()
    AD = mods["as_data"]
    rep_cache, rep_occ = AD.replicate_molecule(cache, int(row), int(n_graphs), occ)
    return AD.collate_gated(rep_cache, np.arange(int(n_graphs), dtype=np.int64), rep_occ)


def gate_matrix(gb, masks: Sequence[np.ndarray], *, device=None, dtype=torch.float32,
                requires_grad: bool = False) -> torch.Tensor:
    return gb.gate_matrix([np.asarray(m, dtype=np.float64).reshape(-1) for m in masks],
                          device=device, dtype=dtype, requires_grad=requires_grad)


def bond_count_of(batch_or_gb) -> int:
    if hasattr(batch_or_gb, "bond_counts"):
        return int(batch_or_gb.bond_counts[0])
    return int(np.asarray(batch_or_gb["edge_cat"]).shape[0] // 2)


def lcrc_forward(arm, gb, M: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.no_grad():
        parts = AG.gated_parts(arm, gb, M)
    return {key: np.asarray(value.detach().cpu().numpy(), dtype=np.float64)
            for key, value in parts.items()
            if key in ("pred_std", "local_sum", "rho", "contrib_atom")}


def gine_batch_replicated(b, split: str, cache_row: int, n_graphs: int, *, device="cpu") -> dict[str, Any]:
    _lazy()
    g = XD.gine_cache(b.dataset)
    rows = [int(cache_row)] * int(n_graphs)
    batch = g.batch(split, rows)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def gine_forward_parts(model, batch: Mapping[str, Any], M: torch.Tensor, n_graphs: int) -> dict[str, Any]:
    if model.training:
        raise RuntimeError("gated GINE forward requires eval() mode")
    trunk = model.trunk
    x = trunk.encode_nodes(batch)
    e = trunk.encode_edges(batch)
    total_bonds = int(batch["edge_cat"].shape[0]) // 2
    if total_bonds % int(n_graphs):
        raise ValueError(f"{total_bonds} physical bond rows is not divisible by {n_graphs} graphs")
    nb = total_bonds // int(n_graphs)
    if int(M.numel()) != int(total_bonds):
        raise ValueError(f"gate matrix has {int(M.numel())} entries for {total_bonds} physical bonds")
    local = torch.arange(nb, device=M.device, dtype=torch.long)
    # GineCache.batch appends, per graph, first the (u,v) then the (v,u) copy;
    # both copies address the same physical bond of that graph.
    per_graph = [local + g * nb for g in range(int(n_graphs))]
    edge_bond = torch.cat([torch.cat([x, x]) for x in per_graph], dim=0)
    gate = M.reshape(-1)[edge_bond]
    h = x
    for conv, norm in zip(trunk.convs, trunk.norms):
        residual = h
        src, dst = batch["edge_index"][0], batch["edge_index"][1]
        msg = (h[src] + e).relu() * gate.view(-1, 1)
        agg = h.new_zeros((h.shape[0], h.shape[1])).index_add_(0, dst, msg)
        update = conv.nn((1.0 + conv.eps) * h + agg)
        update = F.silu(norm(update))
        update = trunk.drop(update)
        h = update + residual
    return model.head.forward_parts(h, batch["gine_batch"], batch["gine_n_atoms"])


def gine_forward(model, batch: Mapping[str, Any], M: torch.Tensor, n_graphs: int) -> dict[str, np.ndarray]:
    """Batched GINE forward with one shared gate per physical bond."""
    parts = gine_forward_parts(model, batch, M, n_graphs)
    return {
        "pred_std": np.asarray(parts["pred_std"].detach().cpu().numpy(), dtype=np.float64),
        "pooled": np.asarray(parts["pooled"].detach().cpu().numpy(), dtype=np.float64),
    }


def lcrc_forward_grad(arm, gb, M: torch.Tensor) -> dict[str, Any]:
    """Gated LCRC forward keeping the mask differentiable (dose-path gate)."""
    parts = AG.gated_parts(arm, gb, M)
    goal = parts["pred_std"].sum()
    grad = torch.autograd.grad(goal, M)[0]
    return {
        "pred_std": np.asarray(parts["pred_std"].detach().cpu().numpy(), dtype=np.float64),
        "local_sum": np.asarray(parts["local_sum"].detach().cpu().numpy(), dtype=np.float64),
        "rho": np.asarray(parts["rho"].detach().cpu().numpy(), dtype=np.float64),
        "grad": np.asarray(grad.detach().cpu().numpy(), dtype=np.float64),
    }


def gine_forward_grad(model, batch: Mapping[str, Any], M: torch.Tensor, n_graphs: int) -> dict[str, Any]:
    """Batched GINE forward keeping the mask differentiable (dose-path gate)."""
    parts = gine_forward_parts(model, batch, M, n_graphs)
    goal = parts["pred_std"].sum()
    grad = torch.autograd.grad(goal, M)[0]
    return {
        "pred_std": np.asarray(parts["pred_std"].detach().cpu().numpy(), dtype=np.float64),
        "grad": np.asarray(grad.detach().cpu().numpy(), dtype=np.float64),
    }


def masks_to_flat(masks: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(m, dtype=np.float64).reshape(-1) for m in masks])


__all__ = [
    "raw_bundle", "build_occ", "load_model", "checkpoint_identity", "parameter_hash",
    "predict_split", "lcrc_gradient", "gine_gradient", "gine_single_batch",
    "lcrc_batch", "gate_matrix", "bond_count_of", "lcrc_forward", "lcrc_forward_grad",
    "gine_batch_replicated", "gine_forward", "gine_forward_grad", "masks_to_flat",
]
