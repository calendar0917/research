"""Model builders and mechanical audits for the two controls.

The S2-off branch is the historical ``v2_models.LCRCCapacityMatched`` class,
imported unchanged.  No alternative architecture is designed here.

The PNA-small width is selected *before training* by a CPU-only parameter-count
rule over ``20, 25, ..., 100``.  No Validation label or score is read by that
rule.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import c2_common as C

import v2_models as V2M  # noqa: E402  (v2 code is on sys.path via c2_common)
import r42_upstream as U  # noqa: E402
import r42_native_train as NT  # noqa: E402
import r42_data as D  # noqa: E402


# ---------------------------------------------------------------------------
# S2-off / capacity-matched
# ---------------------------------------------------------------------------


def build_s2off(spec, norm: Mapping[str, Any], seed: int, device: torch.device | str = "cpu"):
    # Build and audit on CPU: v2's ``build_e`` returns the comparison LCRC-A on
    # CPU while moving only the capacity-matched model to ``device``.  Auditing
    # before the move preserves the historical shared-init check bit-for-bit
    # and avoids a CPU/GPU comparison.  The model is then moved to the requested
    # device; no parameter value changes.
    model, base, shared_state = V2M.build_e(spec, norm, int(seed), "cpu")
    audit = V2M.e_capacity_audit(model, spec)
    shared = V2M.shared_init_audit(base, model)
    if audit["nominal_parameters"] != C.S2OFF_NOMINAL_REFERENCE:
        raise PermissionError(f"S2-off nominal parameter count drifted: {audit['nominal_parameters']}")
    if audit["effective_parameters"] != C.S2OFF_EFFECTIVE_REFERENCE:
        raise PermissionError(f"S2-off effective parameter count drifted: {audit['effective_parameters']}")
    if audit["dead_localmlp_s2_input_columns"] != C.S2OFF_DEAD_COLUMNS:
        raise PermissionError("S2-off dead-column count drifted")
    if audit["replacement_parameters"] != C.S2OFF_REPLACEMENT_PARAMS:
        raise PermissionError("S2-off replacement parameter count drifted")
    if audit["typed_s2_inputs_used"] is not False:
        raise PermissionError("S2-off implementation reports typed-S2 use")
    if shared["all_shared_bitwise_identical"] is not True:
        raise PermissionError(f"S2-off shared init differs from LCRC-A: {shared['mismatched']}")
    model.to(device)
    return model, base, {
        "capacity": audit,
        "shared_init": shared,
        "shared_state_keys": sorted(shared_state),
        "parameters_total": V2M.parameter_count(model),
        "parameters_requires_grad": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "parameters_effective": V2M.effective_parameters_e(model),
    }


def s2off_invariance(model: nn.Module, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that masked / perturbed typed-S2 inputs cannot change the output.

    On CPU the repeated forward is bitwise deterministic and the equality flags
    are exact.  On CUDA, ``index_add_`` uses atomics and repeated identical
    forwards can differ by ~1e-7; the gate therefore also measures the
    run-to-run noise from an identical repeat and requires the masked/perturbed
    differences to stay inside that noise (with a 1e-6 floor).
    """
    model.eval()
    with torch.no_grad():
        y0 = model(batch)
        y_repeat = model(batch)
        y_restricted = model(V2M.restricted_batch(batch))
        perturbed = dict(batch)
        for key, hi in (
            ("path_cat", int(model.spec.n_path_codes)),
            ("path_end63", int(model.spec.n_nbr_codes)),
            ("path_end", int(batch["atom_feat"].shape[0])),
            ("path_closure", 2),
        ):
            if key in perturbed:
                perturbed[key] = torch.randint_like(perturbed[key], 0, hi)
        y_perturbed = model(perturbed)
    def _max_diff(a, b) -> float:
        return float((a - b).abs().max().item()) if a.numel() else 0.0

    repeat_max = _max_diff(y0, y_repeat)
    restricted_max = _max_diff(y0, y_restricted)
    perturbed_max = _max_diff(y0, y_perturbed)
    tolerance = max(1e-6, 4.0 * repeat_max)
    return {
        "identical_restricted_batch": bool(torch.equal(y0, y_restricted)),
        "identical_perturbed_s2": bool(torch.equal(y0, y_perturbed)),
        "identical_repeated_forward": bool(torch.equal(y0, y_repeat)),
        "repeat_max_abs_diff": repeat_max,
        "restricted_max_abs_diff": restricted_max,
        "perturbed_max_abs_diff": perturbed_max,
        "max_abs_diff_perturbed": perturbed_max,
        "noise_tolerance": tolerance,
        "within_gpu_noise_restricted": bool(restricted_max <= tolerance),
        "within_gpu_noise_perturbed": bool(perturbed_max <= tolerance),
        "typed_s2_access_raises": _restricted_access_raises(batch),
    }


def _restricted_access_raises(batch: Mapping[str, Any]) -> bool:
    try:
        V2M.restricted_batch(batch)["path_cat"]
    except KeyError:
        return True
    return False


def s2off_gradient_gate(model: nn.Module, batch: Mapping[str, Any], loss_fn=None) -> dict[str, Any]:
    """One real-batch backward: replacement live, dead S2 input columns zero."""
    model.train()
    model.zero_grad(set_to_none=True)
    pred = model(batch)
    loss = pred.pow(2).mean() if loss_fn is None else loss_fn(pred)
    loss.backward()
    replacement = []
    for name, param in model.local.replacement.named_parameters():
        if param.grad is None:
            replacement.append({"name": name, "grad_max_abs": None, "nonzero": False})
            continue
        replacement.append(
            {
                "name": name,
                "grad_max_abs": float(param.grad.abs().max().item()),
                "grad_norm": float(param.grad.norm().item()),
                "nonzero": bool(float(param.grad.abs().max().item()) > 0.0),
            }
        )
    dead = model.local.mlp.lins[0].weight.grad[:, 320:520]
    live_cols = model.local.mlp.lins[0].weight.grad[:, :320]
    dead_max = float(dead.abs().max().item()) if dead is not None else None
    live_max = float(live_cols.abs().max().item()) if live_cols is not None else None
    model.zero_grad(set_to_none=True)
    return {
        "replacement_parameters": replacement,
        "all_replacement_receive_nonzero_grad": all(x["nonzero"] for x in replacement),
        "dead_s2_input_columns_max_abs_grad": dead_max,
        "dead_s2_input_columns_zero_grad": dead_max == 0.0,
        "live_localmlp_columns_max_abs_grad": live_max,
        "live_localmlp_columns_receive_grad": bool(live_max is not None and live_max > 0.0),
        "loss": float(loss.detach().item()),
    }


# ---------------------------------------------------------------------------
# PNA degree histogram and width search
# ---------------------------------------------------------------------------


def pna_degree_histogram(train_data: Sequence[Any]) -> dict[str, Any]:
    from torch_geometric.utils import degree

    deg = torch.zeros(5000, dtype=torch.long)
    for data in train_data:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    n_atoms = int(sum(int(d.num_nodes) for d in train_data))
    total_degree = int(sum(int(d.edge_index.shape[1]) for d in train_data))
    if int(deg.sum()) != n_atoms:
        raise RuntimeError("PNA degree histogram does not cover every training atom")
    if sum(i * int(v) for i, v in enumerate(deg.tolist())) != total_degree:
        raise RuntimeError("PNA degree histogram total degree does not match the training edge count")
    return {
        "tensor": deg,
        "sha256": C.sha256_bytes(np.ascontiguousarray(deg.numpy()).tobytes()),
        "length": int(deg.numel()),
        "n_atoms": n_atoms,
        "total_degree": total_degree,
        "nonzero": [[int(i), int(v)] for i, v in enumerate(deg.tolist()) if int(v)],
        "source": "train3360_graphs_only",
        "valid_or_test_used": False,
    }


def _pna_width_dir(tmp: Path, deg: torch.Tensor) -> Path:
    (tmp / "saved_PNA_degrees").mkdir(parents=True, exist_ok=True)
    torch.save(deg, tmp / "saved_PNA_degrees" / "degrees.pt")
    return tmp


def pna_estimator(
    *,
    width: int,
    num_features: int,
    edge_dim: int,
    out_path: Path,
    seed: int = C.TRAIN_SEED,
):
    mods = U.install()
    torch.manual_seed(int(seed))
    return mods["graph_models"].Estimator(
        task_type="regression",
        num_features=int(num_features),
        gnn_intermediate_dim=int(width),
        output_node_dim=int(width),
        num_layers=4,
        conv_type="PNA",
        edge_dim=int(edge_dim),
        batch_size=128,
        lr=1e-4,
        linear_output_size=1,
        scaler=None,
        monitor_loss_name="val_loss",
        out_path=str(out_path),
        train_dataset_for_PNA=[],
        regression_loss_fn="mse",
        early_stopping_patience=30,
        optimiser_weight_decay=1e-10,
        use_cpu=True,
    )


def pna_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def run_pna_width_search(train_rows: Sequence[int]) -> dict[str, Any]:
    """CPU-only width search.  This function never loads labels."""
    train_data = D.load_structure_data(rows=[int(r) for r in train_rows])
    deg_info = pna_degree_histogram(train_data)
    # Verify the Train3360 histogram against the frozen v1 tensor.
    v1_deg = torch.load(C.V1_PNA_DEGREES, map_location="cpu", weights_only=False)
    if not torch.equal(v1_deg, deg_info["tensor"]):
        raise PermissionError("Train3360 PNA degree histogram differs from the frozen v1 tensor")
    v1_sha = C.sha256_bytes(np.ascontiguousarray(v1_deg.numpy()).tobytes())
    if v1_sha != "56bc94c1186a1a471b9af7885313ed914f51469f3e40fa99266a20c65721fc40":
        raise PermissionError("frozen v1 PNA histogram tensor hash mismatch")
    dims = {
        "num_features": int(train_data[0].x.shape[1]),
        "edge_dim": int(train_data[0].edge_attr.shape[1]),
    }
    candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pna_width_search_") as tmp_name:
        tmp = _pna_width_dir(Path(tmp_name), deg_info["tensor"])
        for width in C.PNA_WIDTHS:
            model = pna_estimator(
                width=width,
                num_features=dims["num_features"],
                edge_dim=dims["edge_dim"],
                out_path=tmp,
            )
            params = pna_parameter_count(model)
            candidates.append(
                {
                    "width": int(width),
                    "gnn_intermediate_dim": int(width),
                    "output_node_dim": int(width),
                    "parameters_total": int(params),
                    "absolute_difference": int(abs(params - C.PNA_TARGET_PARAMS)),
                    "relative_difference": float(abs(params - C.PNA_TARGET_PARAMS) / C.PNA_TARGET_PARAMS),
                }
            )
            del model
    best = min(candidates, key=lambda row: (row["absolute_difference"], row["width"]))
    return {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "scope": "cpu_parameter_counting_only",
        "labels_read": False,
        "validation_scores_read": False,
        "target_parameters": C.PNA_TARGET_PARAMS,
        "target_source": "LCRC-A-R3 seed0, random42 v1",
        "width_rule": "instantiate PNA widths 20,25,...,100; choose min |params-247106|; freeze before training",
        "widths": list(C.PNA_WIDTHS),
        "candidates": candidates,
        "selected_width": int(best["width"]),
        "selected_parameters_total": int(best["parameters_total"]),
        "absolute_difference": int(best["absolute_difference"]),
        "relative_difference": float(best["relative_difference"]),
        "tightly_size_matched": bool(best["relative_difference"] <= C.PNA_SIZE_TOLERANCE),
        "size_tolerance": C.PNA_SIZE_TOLERANCE,
        "status_label": "TIGHTLY_SIZE_MATCHED" if best["relative_difference"] <= C.PNA_SIZE_TOLERANCE else "NOT_TIGHTLY_SIZE_MATCHED",
        "degree_histogram": {
            "sha256": deg_info["sha256"],
            "length": deg_info["length"],
            "n_atoms": deg_info["n_atoms"],
            "total_degree": deg_info["total_degree"],
            "nonzero": deg_info["nonzero"],
            "source": deg_info["source"],
            "matches_frozen_v1_tensor": True,
            "v1_file": C.track_rel(C.V1_PNA_DEGREES),
            "v1_file_sha256": C.sha256_file(C.V1_PNA_DEGREES),
        },
        "data": {
            "n_train": len(train_rows),
            "train_rows_sha256": C.rows_sha256_int64(train_rows),
            "num_features": dims["num_features"],
            "edge_dim": dims["edge_dim"],
            "test_labels_read": False,
        },
        "env": C.env_info(),
    }


def selected_width() -> int:
    if not C.WIDTH_SEARCH.exists():
        raise PermissionError("PNA width search is missing; run width-search before training")
    payload = C.read_json(C.WIDTH_SEARCH)
    width = int(payload["selected_width"])
    if width not in C.PNA_WIDTHS:
        raise PermissionError("selected PNA width is outside the frozen candidate grid")
    if int(payload["selected_parameters_total"]) != int(
        next(row["parameters_total"] for row in payload["candidates"] if row["width"] == width)
    ):
        raise PermissionError("PNA width-search payload is internally inconsistent")
    if payload.get("labels_read") is not False or payload.get("validation_scores_read") is not False:
        raise PermissionError("PNA width search is not certified label/validation free")
    return width


def pna_small_config(width: int | None = None) -> dict[str, Any]:
    width = selected_width() if width is None else int(width)
    cfg = dict(C.PNA_BASE_CONFIG)
    cfg["gnn_intermediate_dim"] = int(width)
    cfg["output_node_dim"] = int(width)
    cfg["width_source"] = C.track_rel(C.WIDTH_SEARCH)
    cfg["width_selection_rule"] = "min |params-247106| over 20..100 step 5, CPU only, frozen before training"
    return cfg


def pna_gradient_coverage(model: nn.Module, batch) -> dict[str, Any]:
    """Count parameter tensors/elements that receive a nonzero gradient on one batch.

    ``eval`` mode is used so that BatchRenorm running statistics are not
    modified before formal training begins.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    _, _, pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    loss = pred.pow(2).mean()
    loss.backward()
    tensors_total = 0
    tensors_with_grad = 0
    elements_total = 0
    elements_with_nonzero_grad = 0
    for param in model.parameters():
        tensors_total += 1
        elements_total += int(param.numel())
        if param.grad is not None:
            tensors_with_grad += 1
            elements_with_nonzero_grad += int((param.grad != 0).sum().item())
    model.zero_grad(set_to_none=True)
    return {
        "parameter_tensors_total": tensors_total,
        "parameter_tensors_with_grad": tensors_with_grad,
        "parameters_total": elements_total,
        "parameters_with_nonzero_grad_on_batch": elements_with_nonzero_grad,
        "all_parameter_tensors_receive_grad": tensors_with_grad == tensors_total,
        "loss": float(loss.detach().item()),
    }


__all__ = [
    "build_s2off",
    "s2off_invariance",
    "s2off_gradient_gate",
    "pna_degree_histogram",
    "pna_estimator",
    "pna_parameter_count",
    "run_pna_width_search",
    "selected_width",
    "pna_small_config",
    "pna_gradient_coverage",
]
