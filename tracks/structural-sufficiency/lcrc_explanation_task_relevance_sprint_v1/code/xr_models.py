"""Frozen model loading, gated forwards, gradients and P1 model gates.

The two model families are *not* reimplemented from scratch:

* LCRC is the frozen ``cr_models.ArmA`` used by the route round, with the
  inherited ``CONDITIONAL_S1_S2_EVIDENCE_MASK`` forward from
  ``lcrc_auto_subgraph_faithfulness_pilot_v1/code/as_gate.py``.
* GINE is the frozen route-round ``GineTrunk``/``DenseGraphModel``; only the
  message-level shared physical-bond gate is added here, by multiplying the GINE
  message before neighbourhood aggregation in every existing layer.

Nothing is trained or fine-tuned in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import xr_common as C


# ---------------------------------------------------------------------------
# bundle / split occurrence maps
# ---------------------------------------------------------------------------


def bundle(dataset: str = C.DATASET):
    mods = C.source_modules()
    return mods["gr_data"].bundle(dataset, verify=True)


def build_split_occs(b: Any, splits: Sequence[str] = C.SPLITS) -> dict[str, Any]:
    mods = C.source_modules()
    AD = mods["as_data"]
    return {split: AD.build_split_occ(b.caches[split], b.spec, b.dataset, split) for split in splits}


def cache_row_of(b: Any, split: str, row_id: int) -> int:
    ids = np.asarray(b.global_row_ids[split], dtype=np.int64)
    hits = np.flatnonzero(ids == int(row_id))
    if hits.size != 1:
        raise KeyError(f"{split}/{row_id}: found {hits.size} cache rows")
    return int(hits[0])


# ---------------------------------------------------------------------------
# checkpoint loading and replay
# ---------------------------------------------------------------------------


def _load_state(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise RuntimeError(f"{path}: checkpoint has no model_state")
    return payload


def build_untrained(name: str, b: Any):
    mods = C.source_modules()
    ms = mods["gr_models"].ModelSet(b)
    if name in ("L_lipophilicity", "L_WC_LOGP"):
        return ms.lcrc()
    if name in ("G0_lipophilicity", "T_WC_LOGP"):
        return ms.dense("G0")
    raise KeyError(name)


def load_model(name: str, b: Any, *, device: str | torch.device = "cpu"):
    path = C.CHECKPOINTS[name]
    expected = C.EXPECTED_CHECKPOINT_SHA256[name]
    actual = C.sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"CHECKPOINT_UNAVAILABLE {name}: sha256 {actual} != {expected}")
    model = build_untrained(name, b)
    payload = _load_state(path)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, payload


def checkpoint_identity(name: str, b: Any) -> dict[str, Any]:
    path = C.CHECKPOINTS[name]
    payload = _load_state(path)
    model = build_untrained(name, b)
    model.load_state_dict(payload["model_state"], strict=True)
    n_params = int(sum(p.numel() for p in model.parameters()))
    expected_kind = C.EXPECTED_CHECKPOINT_META[name]["kind"]
    if name.startswith("L"):
        actual_kind = "L"
        n_expected = 228674
    elif name.startswith("T"):
        actual_kind = "T"
        n_expected = 377350
    else:
        actual_kind = "G0"
        n_expected = 377350
    state_hash = C.sha256_state(model.state_dict())
    return {
        "name": name,
        "path": C.relpath(path),
        "sha256": C.sha256_file(path),
        "sha256_expected": C.EXPECTED_CHECKPOINT_SHA256[name],
        "sha256_match": bool(C.sha256_file(path) == C.EXPECTED_CHECKPOINT_SHA256[name]),
        "state_sha256": state_hash,
        "n_parameters": n_params,
        "n_parameters_expected": n_expected,
        "parameter_count_match": bool(n_params == n_expected),
        "kind": actual_kind,
        "kind_expected": expected_kind,
        "kind_match": bool(actual_kind == expected_kind),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_valid_rmse_raw": payload.get("valid_rmse_raw"),
        "source_meta": C.EXPECTED_CHECKPOINT_META[name],
        "source_path": C.SOURCE_CHECKPOINT_PATHS[name],
    }


@torch.no_grad()
def predict_split(model, b: Any, split: str, *, device: str | torch.device = "cpu", batch_size: int = 256) -> dict[str, np.ndarray]:
    mods = C.source_modules()
    D = mods["gr_data"]
    order, row_ids, y_raw = D.eval_order(b, split)
    preds: list[np.ndarray] = []
    for start in range(0, order.size, int(batch_size)):
        chunk = order[start : start + int(batch_size)]
        batch = D.collate(b, split, chunk)
        batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
        pred_std = model(batch).detach().cpu().numpy().astype(np.float64).reshape(-1)
        preds.append(pred_std * float(b.scaler["std"]) + float(b.scaler["mean"]))
    pred_raw = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float64)
    return {
        "row_id": np.asarray(row_ids, dtype=np.int64),
        "y_raw": np.asarray(y_raw, dtype=np.float64),
        "pred_raw": pred_raw,
        "rmse_raw": C.rmse(pred_raw, y_raw),
        "mae_raw": C.mae(pred_raw, y_raw),
        "n_rows": int(row_ids.size),
    }


def replay_checkpoint(name: str, b: Any, *, device: str | torch.device = "cpu", batch_size: int = 256) -> dict[str, Any]:
    """Compare the reloaded checkpoint against the source round's stored predictions."""
    model, _payload = load_model(name, b, device=device)
    replay_path = C.RESULTS_DIR / "replay" / {
        "L_lipophilicity": "L_lipophilicity_predictions.npz",
        "G0_lipophilicity": "G0_lipophilicity_predictions.npz",
        "L_WC_LOGP": "L_WC_LOGP_predictions.npz",
        "T_WC_LOGP": "T_WC_LOGP_predictions.npz",
    }[name]
    stored = np.load(replay_path)
    mods = C.source_modules()
    eval_bundle = mods["gr_data"].oracle_bundle(b.dataset, "WC_LOGP") if name.endswith("WC_LOGP") else b
    out: dict[str, Any] = {"name": name, "path": C.relpath(replay_path), "splits": {}}
    passed = True
    for split in C.SPLITS:
        got = predict_split(model, eval_bundle, split, device=device, batch_size=batch_size)
        row_key = f"{split}_row_id"
        pred_key = f"{split}_pred_raw"
        y_key = f"{split}_y_raw"
        if row_key not in stored:
            raise RuntimeError(f"{replay_path}: missing {row_key}")
        if not np.array_equal(got["row_id"], np.asarray(stored[row_key], dtype=np.int64)):
            raise RuntimeError(f"{name}/{split}: row ids differ from stored replay")
        ref_pred = np.asarray(stored[pred_key], dtype=np.float64)
        ref_y = np.asarray(stored[y_key], dtype=np.float64)
        if not np.allclose(ref_y, got["y_raw"], atol=0.0, rtol=0.0):
            raise RuntimeError(f"{name}/{split}: labels differ from stored replay")
        tol = C.trace_tolerance(ref_pred)
        err = np.abs(got["pred_raw"] - ref_pred)
        split_pass = bool(np.all(err <= tol))
        passed = passed and split_pass
        out["splits"][split] = {
            "n_rows": int(got["n_rows"]),
            "rmse_raw": got["rmse_raw"],
            "mae_raw": got["mae_raw"],
            "max_abs_pred_error": float(err.max()) if err.size else 0.0,
            "tolerance_max": float(np.max(tol)) if np.size(tol) else 0.0,
            "passed": split_pass,
            "stored_rmse_raw": C.rmse(ref_pred, ref_y),
        }
    out["passed"] = bool(passed)
    return out


# ---------------------------------------------------------------------------
# LCRC gated forward / gradient / gates
# ---------------------------------------------------------------------------


def lcrc_arm(model):
    if not hasattr(model, "arm"):
        raise TypeError("expected the LCRC adapter with .arm")
    return model.arm


def lcrc_gradient_pass(model, cache: Mapping[str, Any], occ: Any, row: int, b: Any) -> dict[str, Any]:
    mods = C.source_modules()
    AD, AG, AS = mods["as_data"], mods["as_gate"], mods["as_search"]
    arm = lcrc_arm(model)
    gb = AD.collate_gated(cache, [int(row)], occ)
    gp = AS.gradient_pass(arm, gb, sigma=float(b.scaler["std"]), mu=float(b.scaler["mean"]))
    gp["batch"] = gb
    gp["n_bonds"] = int(gb.bond_counts[0])
    gp["n_atoms"] = int(gb.atom_jump.numel() if False else np.asarray(cache["atom_ptr"])[row + 1] - np.asarray(cache["atom_ptr"])[row])
    return gp


def lcrc_all_ones_check(model, cache: Mapping[str, Any], occ: Any, row: int, b: Any) -> dict[str, Any]:
    mods = C.source_modules()
    AD, AG = mods["as_data"], mods["as_gate"]
    arm = lcrc_arm(model)
    gb = AD.collate_gated(cache, [int(row)], occ)
    return AG.equivalence_check(arm, gb, sigma=float(b.scaler["std"]), mu=float(b.scaler["mean"]))


def lcrc_finite_difference(model, cache: Mapping[str, Any], occ: Any, row: int, b: Any, bond_ids: Sequence[int], *, eps: float = C.FD_EPS) -> list[dict[str, Any]]:
    mods = C.source_modules()
    AD, AG = mods["as_data"], mods["as_gate"]
    arm = lcrc_arm(model)
    gb = AD.collate_gated(cache, [int(row)], occ)
    gp = lcrc_gradient_pass(model, cache, occ, row, b)
    grad = gp["grad_transformed"][0]
    sigma, mu = float(b.scaler["std"]), float(b.scaler["mean"])
    out = []
    with torch.no_grad():
        for bond in bond_ids:
            Mp = gb.gate_ones()
            Mm = gb.gate_ones()
            Mp[0, int(bond)] = 1.0 + float(eps)
            Mm[0, int(bond)] = 1.0 - float(eps)
            pp = AG.gated_parts(arm, gb, Mp)["pred_std"].item() * sigma + mu
            pm = AG.gated_parts(arm, gb, Mm)["pred_std"].item() * sigma + mu
            fd = (pp - pm) / (2.0 * float(eps))
            ref = float(grad[int(bond)])
            err = abs(fd - ref)
            tol = 1e-2 + 1e-2 * max(1.0, abs(fd))
            out.append(
                {
                    "row": int(row),
                    "bond": int(bond),
                    "gradient_raw": ref,
                    "finite_difference_raw": float(fd),
                    "abs_error": float(err),
                    "tolerance": float(tol),
                    "passed": bool(err <= tol),
                }
            )
    return out


def lcrc_renumbering_check(model, cache: Mapping[str, Any], occ: Any, rows: Sequence[int], b: Any) -> dict[str, Any]:
    mods = C.source_modules()
    return mods["as_worker"].renumbering_check(
        lcrc_arm(model), cache, occ, list(rows),
        sigma=float(b.scaler["std"]), mu=float(b.scaler["mean"]),
    )


# ---------------------------------------------------------------------------
# GINE gated forward / gradient / gates
# ---------------------------------------------------------------------------


def _as_tensor_batch(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def gine_batch_for_row(b: Any, split: str, row: int, *, device="cpu") -> dict[str, Any]:
    mods = C.source_modules()
    D = mods["gr_data"]
    batch = D.collate(b, split, [int(row)])
    return _as_tensor_batch(batch, device)


def gine_gated_parts(model, batch: Mapping[str, torch.Tensor], M: torch.Tensor) -> dict[str, Any]:
    """Frozen GINE with one shared physical-bond gate on every message.

    ``M`` is ``(n_graphs, max_bonds)``.  This implementation is deliberately
    single-graph (the explanation worker processes one endpoint at a time) and
    mirrors the frozen ``GineTrunk.layers`` update exactly, adding only the
    message multiplier.
    """
    if model.training:
        raise RuntimeError("gated GINE forward requires eval() mode")
    if int(M.shape[0]) != 1:
        raise ValueError("gine_gated_parts supports one graph per call")
    trunk = model.trunk
    x = trunk.encode_nodes(batch)
    e = trunk.encode_edges(batch)
    nb = int(e.shape[0]) // 2
    edge_bond = torch.cat(
        [torch.arange(nb, device=M.device, dtype=torch.long), torch.arange(nb, device=M.device, dtype=torch.long)],
        dim=0,
    )
    if int(M.shape[1]) < nb:
        raise ValueError(f"gate matrix width {int(M.shape[1])} < physical bonds {nb}")
    gate = M[0, edge_bond]
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
    parts = model.head.forward_parts(h, batch["gine_batch"], batch["gine_n_atoms"])
    parts["h"] = h
    parts["gate"] = gate
    return parts


def gine_gradient_pass(model, batch: Mapping[str, torch.Tensor], b: Any) -> dict[str, Any]:
    nb = int(batch["edge_cat"].shape[0]) // 2
    M = torch.ones((1, nb), device=batch["edge_index"].device, dtype=torch.float32, requires_grad=True)
    parts = gine_gated_parts(model, batch, M)
    grad = torch.autograd.grad(parts["pred_std"].sum(), M)[0].detach().cpu().numpy().astype(np.float64)
    pred_std = parts["pred_std"].detach().cpu().numpy().astype(np.float64)
    sigma, mu = float(b.scaler["std"]), float(b.scaler["mean"])
    return {
        "grad_std": grad,
        "grad_transformed": grad * sigma,
        "pred_std": pred_std,
        "pred_raw": pred_std * sigma + mu,
        "n_bonds": int(nb),
        "n_atoms": int(batch["edge_index"].max().item()) + 1 if batch["edge_index"].numel() else 0,
    }


def gine_all_ones_check(model, batch: Mapping[str, torch.Tensor], b: Any) -> dict[str, Any]:
    nb = int(batch["edge_cat"].shape[0]) // 2
    M = torch.ones((1, nb), device=batch["edge_index"].device, dtype=torch.float32)
    with torch.no_grad():
        ref = model.parts(batch)["pred_std"].detach().cpu().numpy().astype(np.float64)
        got = gine_gated_parts(model, batch, M)["pred_std"].detach().cpu().numpy().astype(np.float64)
    sigma, mu = float(b.scaler["std"]), float(b.scaler["mean"])
    ref_raw, got_raw = ref * sigma + mu, got * sigma + mu
    tol = C.trace_tolerance(ref_raw)
    err = np.abs(ref_raw - got_raw)
    return {
        "n_graphs": int(ref.size),
        "pred_raw_max_abs_err": float(err.max()) if err.size else 0.0,
        "pred_raw_tolerance_max": float(np.max(tol)) if np.size(tol) else 0.0,
        "passed": bool(np.all(err <= tol)),
    }


def gine_finite_difference(model, batch: Mapping[str, torch.Tensor], b: Any, bond_ids: Sequence[int], *, eps: float = C.FD_EPS) -> list[dict[str, Any]]:
    nb = int(batch["edge_cat"].shape[0]) // 2
    gp = gine_gradient_pass(model, batch, b)
    grad = gp["grad_transformed"][0]
    sigma, mu = float(b.scaler["std"]), float(b.scaler["mean"])
    out = []
    with torch.no_grad():
        for bond in bond_ids:
            Mp = torch.ones((1, nb), device=batch["edge_index"].device, dtype=torch.float32)
            Mm = torch.ones((1, nb), device=batch["edge_index"].device, dtype=torch.float32)
            Mp[0, int(bond)] = 1.0 + float(eps)
            Mm[0, int(bond)] = 1.0 - float(eps)
            pp = gine_gated_parts(model, batch, Mp)["pred_std"].item() * sigma + mu
            pm = gine_gated_parts(model, batch, Mm)["pred_std"].item() * sigma + mu
            fd = (pp - pm) / (2.0 * float(eps))
            ref = float(grad[int(bond)])
            err = abs(fd - ref)
            tol = 1e-2 + 1e-2 * max(1.0, abs(fd))
            out.append(
                {
                    "bond": int(bond),
                    "gradient_raw": ref,
                    "finite_difference_raw": float(fd),
                    "abs_error": float(err),
                    "tolerance": float(tol),
                    "passed": bool(err <= tol),
                }
            )
    return out


def _gine_arrays_for_row(b: Any, split: str, row: int) -> dict[str, np.ndarray]:
    mods = C.source_modules()
    D = mods["gr_data"]
    g = D.gine_cache(b.dataset)
    gid = int(np.asarray(b.global_row_ids[split], dtype=np.int64)[int(row)])
    a0, a1 = int(g.node_ptr[gid]), int(g.node_ptr[gid + 1])
    e0, e1 = int(g.bond_ptr[gid]), int(g.bond_ptr[gid + 1])
    return {
        "x_cat": np.asarray(g.atom_feat[a0:a1], dtype=np.int64),
        "deg": np.asarray(g.degree[a0:a1], dtype=np.int64),
        "ring": np.asarray(g.min_ring[a0:a1], dtype=np.int64),
        "u": np.asarray(g.bond_u[e0:e1], dtype=np.int64),
        "v": np.asarray(g.bond_v[e0:e1], dtype=np.int64),
        "bf": np.asarray(g.bond_feat[e0:e1], dtype=np.int64),
    }


def _gine_batch_from_arrays(arrays: Mapping[str, np.ndarray], *, device="cpu") -> dict[str, Any]:
    n = int(arrays["x_cat"].shape[0])
    u = np.asarray(arrays["u"], dtype=np.int64)
    v = np.asarray(arrays["v"], dtype=np.int64)
    bf = np.asarray(arrays["bf"], dtype=np.int64)
    eidx = np.concatenate([np.stack([u, v], axis=1), np.stack([v, u], axis=1)], axis=0)
    ecat = np.concatenate([bf, bf], axis=0)
    return {
        "x_cat": torch.as_tensor(arrays["x_cat"], dtype=torch.long, device=device),
        "deg": torch.as_tensor(arrays["deg"], dtype=torch.long, device=device),
        "ring": torch.as_tensor(arrays["ring"], dtype=torch.long, device=device),
        "edge_index": torch.as_tensor(eidx.T, dtype=torch.long, device=device),
        "edge_cat": torch.as_tensor(ecat, dtype=torch.long, device=device),
        "gine_batch": torch.zeros(n, dtype=torch.long, device=device),
        "gine_n_atoms": torch.as_tensor([n], dtype=torch.long, device=device),
    }


def gine_renumbering_check(model, b: Any, split: str, rows: Sequence[int], *, seed: int = C.SAMPLE_SEED) -> dict[str, Any]:
    """Atom renumbering + physical-bond renumbering equivalence for GINE."""
    rng = np.random.default_rng([int(seed), 992])
    records = []
    ok = True
    for row in rows:
        arr = _gine_arrays_for_row(b, split, int(row))
        n = int(arr["x_cat"].shape[0])
        nb = int(arr["u"].size)
        perm = rng.permutation(n)          # new index = perm[old index]
        inv = np.argsort(perm)
        u_new = inv[arr["u"]]
        v_new = inv[arr["v"]]
        pair = np.stack([np.minimum(u_new, v_new), np.maximum(u_new, v_new)], axis=1)
        order = np.lexsort((pair[:, 1], pair[:, 0]))
        arr_perm = {
            "x_cat": arr["x_cat"][perm],
            "deg": arr["deg"][perm],
            "ring": arr["ring"][perm],
            "u": pair[order, 0],
            "v": pair[order, 1],
            "bf": arr["bf"][order],
        }
        batch0 = _gine_batch_from_arrays(arr)
        batch1 = _gine_batch_from_arrays(arr_perm)
        gp0 = gine_gradient_pass(model, batch0, b)
        gp1 = gine_gradient_pass(model, batch1, b)
        pos = {tuple(pair[k].tolist()): k for k in range(pair.shape[0])}
        inv_order = np.argsort(order)
        remap = np.zeros(nb, dtype=np.int64)
        for old_id, (i, j) in enumerate(zip(arr["u"].tolist(), arr["v"].tolist())):
            a, bb = int(inv[i]), int(inv[j])
            remap[old_id] = inv_order[pos[(min(a, bb), max(a, bb))]]
        pred_err = abs(float(gp0["pred_raw"][0]) - float(gp1["pred_raw"][0]))
        pred_tol = float(C.trace_tolerance(float(gp0["pred_raw"][0])))
        grad0 = gp0["grad_transformed"][0][:nb]
        grad1_back = gp1["grad_transformed"][0][:nb][remap]
        grad_err = np.abs(grad0 - grad1_back)
        grad_tol = 1e-5 + 1e-5 * np.abs(grad0)
        passed = bool(pred_err <= pred_tol and np.all(grad_err <= grad_tol))
        ok = ok and passed
        records.append(
            {
                "row": int(row),
                "n_atoms": n,
                "n_bonds": nb,
                "pred_raw_original": float(gp0["pred_raw"][0]),
                "pred_raw_renumbered": float(gp1["pred_raw"][0]),
                "pred_abs_err": float(pred_err),
                "pred_tolerance": pred_tol,
                "grad_max_abs_err": float(grad_err.max()) if grad_err.size else 0.0,
                "grad_max_tolerance": float(grad_tol.max()) if grad_tol.size else 0.0,
                "passed": passed,
            }
        )
    return {"n_molecules": len(records), "records": records, "passed": bool(ok)}


__all__ = [
    "bundle",
    "build_split_occs",
    "cache_row_of",
    "load_model",
    "checkpoint_identity",
    "predict_split",
    "replay_checkpoint",
    "lcrc_gradient_pass",
    "lcrc_all_ones_check",
    "lcrc_finite_difference",
    "lcrc_renumbering_check",
    "gine_batch_for_row",
    "gine_gated_parts",
    "gine_gradient_pass",
    "gine_all_ones_check",
    "gine_finite_difference",
    "gine_renumbering_check",
]
