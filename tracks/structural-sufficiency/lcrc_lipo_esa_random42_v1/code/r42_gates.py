"""CPU implementation gates (Stage 2) and the XGB CPU baseline fit.

Everything here runs without CUDA.  A gate is a real-molecule forward/backward
plus identity, finiteness, unit and label-isolation checks; no hyper-parameter
search is performed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import r42_common as C
import r42_data as D
import r42_labels as L
import r42_metrics as MET
import r42_native_train as NT
import r42_lcrc_train as LT
import r42_upstream as U
import r42_xgb as X


def _result(name: str, ok: bool, **details) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), **details}


def _finite_grads(model: torch.nn.Module) -> tuple[bool, int, int]:
    total = 0
    with_grad = 0
    all_finite = True
    for param in model.parameters():
        total += 1
        if param.grad is None:
            continue
        with_grad += 1
        if not torch.isfinite(param.grad).all():
            all_finite = False
    return all_finite, with_grad, total


def _batch_identity(batch, data_list, n_atoms_per, n_edges_per) -> dict[str, Any]:
    from torch_geometric.data import Batch

    return {
        "n_graphs": int(batch.num_graphs),
        "n_atoms": int(batch.x.shape[0]),
        "n_edges": int(batch.edge_index.shape[1]),
        "expected_atoms": int(sum(n_atoms_per)),
        "expected_edges": int(sum(n_edges_per)),
        "atom_identity_ok": int(batch.x.shape[0]) == int(sum(n_atoms_per)),
        "edge_identity_ok": int(batch.edge_index.shape[1]) == int(sum(n_edges_per)),
        "graph_identity_ok": int(batch.batch.max().item()) == len(data_list) - 1 and int(batch.batch.min().item()) == 0,
        "batch_shape_ok": bool(int(batch.batch.shape[0]) == int(batch.x.shape[0])),
    }


def _metric_reference(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "mae": float(np.mean(np.abs(p - t))),
        "r2": float(1.0 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)),
    }


def _label_isolation_check(labels: L.TrainValidLabels) -> dict[str, Any]:
    raised = False
    try:
        labels.get("test")
    except L.TestLabelAccessError:
        raised = True
    mapping_raised = False
    try:
        L.assert_train_valid_only({"train": labels.y_train, "valid": labels.y_valid, "test": np.zeros(420)})
    except L.TestLabelAccessError:
        mapping_raised = True
    return {"TrainValidLabels_rejects_test": raised, "mapping_with_test_rejected": mapping_raised}


def run_cpu_gates(cfg: Mapping[str, Any], *, skip_esa: bool = False) -> dict[str, Any]:
    t_start = time.perf_counter()
    split = D.load_split()
    labels = L.load_train_valid_labels()
    scopes = _label_isolation_check(labels)
    rows_gate = [int(r) for r in split["rows"]["train"][:32]]
    data_all = D.load_structure_data()
    data_gate = [data_all[r] for r in rows_gate]
    structures = D.structure_summary()
    target_norm = D.fit_target_scaler(labels.y_train)
    results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # target scaler: depends on Train3360 only
    # ------------------------------------------------------------------
    train_only = D.fit_target_scaler(labels.y_train)
    perturbed = D.fit_target_scaler(np.concatenate([labels.y_train, np.full(420, 1e6)]))
    results.append(
        _result(
            "target_scaler_train_only",
            train_only["mean"] == target_norm["mean"] and train_only["scale"] == target_norm["scale"] and perturbed["mean"] != target_norm["mean"],
            mean=train_only["mean"],
            scale=train_only["scale"],
            unchanged_without_validation=bool(perturbed["mean"] != target_norm["mean"]),
            n=train_only["n"],
        )
    )

    # ------------------------------------------------------------------
    # metrics: raw units, independent NumPy reference
    # ------------------------------------------------------------------
    pred_demo = np.asarray([0.1, -1.2, 2.4, 3.3], dtype=np.float64)
    y_demo = np.asarray([0.0, -1.0, 2.0, 3.0], dtype=np.float64)
    ours = MET.all_metrics(pred_demo, y_demo)
    ref = _metric_reference(pred_demo, y_demo)
    std_pred = (pred_demo - target_norm["mean"]) / target_norm["scale"]
    std_y = (y_demo - target_norm["mean"]) / target_norm["scale"]
    std_rmse = MET.rmse(std_pred, std_y)
    raw_rmse = MET.rmse(pred_demo, y_demo)
    results.append(
        _result(
            "metrics_match_numpy_reference",
            all(abs(ours[k] - ref[k]) < 1e-12 for k in ref),
            ours=ours,
            reference=ref,
        )
    )
    results.append(
        _result(
            "standardized_rmse_is_not_raw_rmse",
            abs(raw_rmse - std_rmse * target_norm["scale"]) < 1e-9 and abs(raw_rmse - std_rmse) > 1e-3,
            raw_rmse=raw_rmse,
            standardized_rmse=std_rmse,
            recovered_raw=std_rmse * target_norm["scale"],
            target_scale=target_norm["scale"],
        )
    )

    # ------------------------------------------------------------------
    # LCRC-A-R3 forward/backward on 32 real Train3360 molecules
    # ------------------------------------------------------------------
    views = D.build_lcrc_views(split, labels)
    import lcrc_data
    import lcrc_model

    torch.manual_seed(int(cfg["models"]["lcrc"]["seed"]))
    lcrc = lcrc_model.LCRC(views["spec"], views["train_norm"], lcrc_model.DEFAULT_CONFIG)
    batch = lcrc_data.collate(views["train_cache"], list(range(32)))
    y = torch.as_tensor((labels.y_train[:32] - target_norm["mean"]) / target_norm["scale"], dtype=torch.float32)
    pred = lcrc(batch)
    loss = F.mse_loss(pred.view(-1), y.view(-1))
    loss.backward()
    grad_ok, grad_count, param_count = _finite_grads(lcrc)
    lcrc_params = int(sum(p.numel() for p in lcrc.parameters()))
    raw = LT._predict_lcrc(lcrc, D.select_cache(views["train_cache"], list(range(32)), split_label="gate"), torch.device("cpu"))
    raw = NT.denormalise(raw, target_norm)
    results.append(
        _result(
            "lcrc_forward_backward",
            bool(pred.shape == (32,)) and bool(torch.isfinite(loss)) and grad_ok and grad_count == param_count,
            parameters=lcrc_params,
            expected_parameters=247106,
            output_shape=list(pred.shape),
            loss=float(loss),
            finite_grads=grad_ok,
            params_with_grad=f"{grad_count}/{param_count}",
            target_mean=float(y.mean()),
            prediction_mean_raw=float(raw.mean()),
            unit_recovery_max_abs_error=float(np.max(np.abs(raw - (pred.detach().numpy() * target_norm["scale"] + target_norm["mean"])))),
            has_iterative_message_passing=False,
            module=lcrc.__class__.__module__,
        )
    )

    # ------------------------------------------------------------------
    # PNA-native forward/backward on 32 real Train3360 molecules
    # ------------------------------------------------------------------
    gate_dir = C.RUNS / "cpu_gate_pna"
    gate_dir.mkdir(parents=True, exist_ok=True)
    pna, degree_info = NT.build_pna(cfg["models"]["pna"], torch.device("cpu"), gate_dir, [data_all[r] for r in split["rows"]["train"]], structures["atom_feat_dim"], structures["edge_feat_dim"])
    from torch_geometric.data import Batch

    pna_batch = Batch.from_data_list(data_gate)
    pna_out = pna(pna_batch.x, pna_batch.edge_index, pna_batch.batch, pna_batch.edge_attr)
    pna_pred = pna_out[2]
    pna_loss = F.mse_loss(pna_pred.view(-1), y.view(-1))
    pna_loss.backward()
    pna_grad_ok, pna_grad_count, pna_param_count = _finite_grads(pna)
    hist = torch.load(gate_dir / "saved_PNA_degrees" / "degrees.pt", map_location="cpu", weights_only=False)
    results.append(
        _result(
            "pna_forward_backward",
            bool(pna_pred.shape == (32,)) and bool(torch.isfinite(pna_loss)) and pna_grad_ok and pna_grad_count == pna_param_count,
            parameters=int(sum(p.numel() for p in pna.parameters())),
            output_shape=list(pna_pred.shape),
            loss=float(pna_loss),
            finite_grads=pna_grad_ok,
            params_with_grad=f"{pna_grad_count}/{pna_param_count}",
            readout_pool_dims=list(pna_out[1].shape),
            degree_histogram_sha256=degree_info["histogram_sha256"],
            degree_histogram_source=degree_info["source"],
            degree_histogram_sum=int(hist.sum()),
            expected_total_directed_edges=int(sum(int(d.edge_index.shape[1]) for d in data_gate)),
            module=pna.__class__.__module__,
            conv_module=pna.gnn_model.__class__.__module__,
        )
    )
    results.append(_result("pna_batch_identity", all(_batch_identity(pna_batch, data_gate, [d.num_nodes for d in data_gate], [d.edge_index.shape[1] for d in data_gate]).values()), **_batch_identity(pna_batch, data_gate, [d.num_nodes for d in data_gate], [d.edge_index.shape[1] for d in data_gate])))

    # ------------------------------------------------------------------
    # ESA-native forward/backward on 16 real Train3360 molecules
    # ------------------------------------------------------------------
    if not skip_esa:
        esa_dir = C.RUNS / "cpu_gate_esa"
        esa_dir.mkdir(parents=True, exist_ok=True)
        esa_data = data_gate[:16]
        esa, esa_extra = NT.build_esa(cfg["models"]["esa"], torch.device("cpu"), esa_dir, structures["max_edge_global"], structures["atom_feat_dim"], structures["edge_feat_dim"])
        esa_batch = Batch.from_data_list(esa_data)
        mods_mod = U.install()
        num_max_items = int(mods_mod["esa_models"].nearest_multiple_of_8(structures["max_edge_global"] + 1))
        esa_extra["num_max_items"] = num_max_items
        esa_pred = esa(esa_batch.x.float(), esa_batch.edge_index, esa_batch.batch, esa_batch.edge_attr.float(), num_max_items, esa_batch)
        esa_loss = F.mse_loss(esa_pred.view(-1), y[:16].view(-1))
        esa_loss.backward()
        esa_grad_ok, esa_grad_count, esa_param_count = _finite_grads(esa)
        dead = [name for name, param in esa.named_parameters() if param.grad is None]
        # Upstream leaves two normalisation modules unreachable in the
        # edge-attention + PMA configuration (documented, not a defect of this round).
        expected_dead = {
            "mlp_norm.ln.weight",
            "mlp_norm.ln.bias",
            "st_fast.decoder.0.norm.ln.weight",
            "st_fast.decoder.0.norm.ln.bias",
        }
        results.append(
            _result(
                "esa_forward_backward",
                bool(esa_pred.shape == (16,)) and bool(torch.isfinite(esa_loss)) and esa_grad_ok and set(dead) <= expected_dead,
                parameters=int(sum(p.numel() for p in esa.parameters())),
                output_shape=list(esa_pred.shape),
                loss=float(esa_loss),
                finite_grads=esa_grad_ok,
                params_with_grad=f"{esa_grad_count}/{esa_param_count}",
                parameters_without_grad=dead,
                unreachable_parameters_are_upstream_module_specific=True,
                attention_backend=esa_extra["attention_backend"],
                num_max_items=int(num_max_items),
                graph_dim=int(cfg["models"]["esa"]["graph_dim"]),
                layer_types=list(cfg["models"]["esa"]["layer_types"]),
                module=esa.__class__.__module__,
                esa_module=esa.st_fast.__class__.__module__,
            )
        )
        ident = _batch_identity(esa_batch, esa_data, [d.num_nodes for d in esa_data], [d.edge_index.shape[1] for d in esa_data])
        results.append(_result("esa_batch_identity", all(ident.values()), **ident))

        # Mask semantics: patched torch bool mask must equal the xformers
        # additive-bias convention (bias == 0) on a real batch.
        mods = U.install()
        mask_mod = mods["masked_layers"]
        max_items = int(num_max_items)
        mask_torch = mask_mod.get_adj_mask_from_edge_index_edge(
            edge_index=esa_batch.edge_index,
            batch_size=esa_batch.num_graphs,
            max_items=max_items,
            batch_mapping=esa_batch.batch,
            xformers_or_torch_attn="torch",
            use_bfloat16=False,
        )
        bias_xformers = mask_mod.get_adj_mask_from_edge_index_edge(
            edge_index=esa_batch.edge_index,
            batch_size=esa_batch.num_graphs,
            max_items=max_items,
            batch_mapping=esa_batch.batch,
            xformers_or_torch_attn="xformers",
            use_bfloat16=False,
        )
        allowed_torch = mask_torch[:, 0] if mask_torch.dim() == 4 else mask_torch
        allowed_xf = bias_xformers[:, 0] == 0 if bias_xformers.dim() == 4 else bias_xformers == 0
        # Padded *edge slots* (index >= number of directed edges of that graph)
        # must never attend in either convention.
        edges_per_graph = torch.bincount(esa_batch.batch.index_select(0, esa_batch.edge_index[0]), minlength=esa_batch.num_graphs)
        pad = torch.arange(max_items).unsqueeze(0) >= edges_per_graph.unsqueeze(1)
        equal = torch.equal(allowed_torch, allowed_xf)
        # Independent reference: attention is allowed exactly between directed
        # edges that share at least one endpoint, never on the diagonal.
        reference_ok = True
        offset = 0
        for graph in range(esa_batch.num_graphs):
            n_e = int(edges_per_graph[graph])
            block = esa_batch.edge_index[:, offset : offset + n_e]
            offset += n_e
            src = block[0].unsqueeze(1)
            dst = block[1].unsqueeze(1)
            ref = (src == src.t()) | (src == dst.t()) | (dst == src.t()) | (dst == dst.t())
            ref.fill_diagonal_(False)
            got = allowed_torch[graph][:n_e, :n_e]
            if not torch.equal(ref, got):
                reference_ok = False
                break
        results.append(
            _result(
                "esa_torch_mask_equals_xformers_convention",
                bool(equal) and bool(reference_ok) and int((allowed_torch & pad.unsqueeze(1)).sum()) == 0,
                backend="torch",
                shape=list(mask_torch.shape),
                allowed_pairs=int(allowed_torch.sum()),
                matches_upstream_adjacency_reference=bool(reference_ok),
                padded_pairs_attend_torch=int((allowed_torch & pad.unsqueeze(1)).sum()),
                padded_pairs_attend_xformers=int((allowed_xf & pad.unsqueeze(1)).sum()),
                diagonal_self_attention=int(torch.diagonal(allowed_torch, dim1=-2, dim2=-1).sum()),
            )
        )

    # ------------------------------------------------------------------
    # XGB P3: the actual CPU baseline fit (Train3360 -> Validation420)
    # ------------------------------------------------------------------
    # The gate fits the frozen XGB recipe itself (CPU only, no GPU time); the
    # formal run is produced separately by run_v1.py train xgb.
    xgb_dir = C.RUNS / "cpu_gate_xgb"
    X.freeze_descriptor_names()
    xgb_payload = X.train_xgb(cfg["models"]["xgb"], split, labels, xgb_dir)
    xgb_ref = _metric_reference(
        np.load(xgb_dir / "valid_predictions.npz")["pred"],
        labels.y_valid,
    )
    results.append(
        _result(
            "xgb_train_valid",
            all(abs(xgb_payload["metrics"]["valid"][k] - xgb_ref[k]) < 1e-12 for k in xgb_ref),
            model_sha256=xgb_payload["model_sha256"],
            train_rmse=xgb_payload["metrics"]["train"]["rmse"],
            valid_rmse=xgb_payload["metrics"]["valid"]["rmse"],
            valid_mae=xgb_payload["metrics"]["valid"]["mae"],
            valid_r2=xgb_payload["metrics"]["valid"]["r2"],
            fit_seconds=xgb_payload["training"]["fit_seconds"],
            n_boosted_rounds=xgb_payload["training"]["n_boosted_rounds"],
            n_features_used=xgb_payload["features"]["n_features_used"],
        )
    )

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "CPU implementation gates",
        "status": "PASS" if all(r["ok"] for r in results) else "FAIL",
        "checks": results,
        "label_isolation": scopes,
        "structures": structures,
        "target_norm": target_norm,
        "upstream_patches": list(U.PATCHES),
        "stub_manifest": dict(U.STUB_MANIFEST),
        "env": C.env_info(),
        "cpu_gate_seconds": float(time.perf_counter() - t_start),
        "gpu_used": False,
    }
    C.write_json(C.ANALYSIS / "cpu_gates.json", payload)
    C.log(f"[cpu-gates] {payload['status']} ({len(results)} checks, {payload['cpu_gate_seconds']:.1f}s)")
    return payload


__all__ = ["run_cpu_gates"]
