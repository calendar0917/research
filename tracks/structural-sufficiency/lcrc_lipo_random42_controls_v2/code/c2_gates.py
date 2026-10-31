"""CPU gates for controls v2.

Every gate is a mechanical check on the frozen data, the historical references,
the imported capacity-matched implementation, or the pre-registered PNA width
rule.  No GPU is touched.  No Validation score is used to choose anything.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import c2_common as C
import c2_data as CD
import c2_models as CM

import r42_data as D  # noqa: E402
import r42_labels as L  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402
import lcrc_data  # noqa: E402


def _result(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), **details}


def _metric_reference(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "mae": float(np.mean(np.abs(p - t))),
        "r2": float(1.0 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)),
    }


def _scan_no_validation_scores(obj: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = str(key).lower()
            if "valid" in key_s and any(tok in key_s for tok in ("rmse", "mae", "r2", "loss")):
                hits.append(f"{path}.{key}")
            hits.extend(_scan_no_validation_scores(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(_scan_no_validation_scores(value, f"{path}[{i}]"))
    return hits


def run_cpu_gates(*, write: bool = True) -> dict[str, Any]:
    t_start = time.perf_counter()
    results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. byte-level reuse and historical references
    # ------------------------------------------------------------------
    reuse = C.verify_history()
    sources = C.verify_source_hashes()
    artifact_guard = C.assert_no_forbidden_artifacts_read()
    results.append(
        _result(
            "byte_level_reuse_and_history",
            reuse["status"] == "PASS" and reuse["test_labels_read"] is False,
            split_file_sha256=reuse["split"]["split_file_sha256"],
            split_index_sha256=reuse["split"]["split_index_sha256"],
            label_file_sha256=reuse["split"]["label_file_sha256"],
            structures_sha256=reuse["structures"]["sha256"],
            historical_models={
                name: {
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "valid_predictions_sha256": row["valid_predictions_sha256"],
                    "valid_metrics_recomputed": row["valid_metrics_recomputed"],
                }
                for name, row in reuse["models"].items()
            },
            source_hashes=sources,
        )
    )
    results.append(_result("test_artifact_static_guard", artifact_guard["status"] == "PASS", **artifact_guard))

    split, labels = CD.load_frozen_split_and_labels()
    target_norm = CD.load_target_norm(labels)
    results.append(
        _result(
            "label_isolation_and_target_scaler",
            True,
            train_valid_labels_reject_test=_labels_reject_test(labels),
            target_norm=target_norm,
            frozen_mean=C.TARGET_MEAN,
            frozen_scale=C.TARGET_SCALE,
            exact_match=target_norm["mean"] == C.TARGET_MEAN and target_norm["scale"] == C.TARGET_SCALE,
            test_labels_read=False,
        )
    )

    # ------------------------------------------------------------------
    # 2. metric evaluator: independent NumPy/sklearn reference
    # ------------------------------------------------------------------
    pred_demo = np.asarray([0.1, -1.2, 2.4, 3.3, 0.0], dtype=np.float64)
    y_demo = np.asarray([0.0, -1.0, 2.0, 3.0, 0.5], dtype=np.float64)
    ours = MET.all_metrics(pred_demo, y_demo)
    ref = _metric_reference(pred_demo, y_demo)
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        sk = {
            "rmse": float(np.sqrt(mean_squared_error(y_demo, pred_demo))),
            "mae": float(mean_absolute_error(y_demo, pred_demo)),
            "r2": float(r2_score(y_demo, pred_demo)),
        }
    except Exception:
        sk = None
    metric_ok = all(abs(ours[k] - ref[k]) <= 1e-15 for k in ref)
    if sk is not None:
        metric_ok = metric_ok and all(abs(ours[k] - sk[k]) <= 1e-14 for k in sk)
    results.append(
        _result(
            "metrics_raw_logd_reference",
            metric_ok,
            ours=ours,
            numpy_reference=ref,
            sklearn_reference=sk,
            historical_lcrc_valid=reuse["models"]["LCRC-A-R3-scratch"]["valid_metrics_recomputed"],
            historical_pna_valid=reuse["models"]["PNA-native"]["valid_metrics_recomputed"],
        )
    )

    # ------------------------------------------------------------------
    # 3. LCRC views / normalisation / cache identity
    # ------------------------------------------------------------------
    views = CD.build_lcrc_views(split, labels)
    spec = views["spec"]
    train_norm = views["train_norm"]
    results.append(
        _result(
            "lcrc_views_and_train_only_normalisation",
            views["train_norm_fingerprint"] == C.LCRC_TRAIN_NORM_FINGERPRINT
            and views["built"]["labels_present"] is False
            and (
                views["built"]["v1_content_fingerprint"] is None
                or views["built"]["v1_content_fingerprint"] == views["built"]["content_fingerprint"]
            ),
            cache_path=C.track_rel(views["built"]["path"]),
            cache_sha256=views["built"]["file_sha256"],
            cache_content_fingerprint=views["built"]["content_fingerprint"],
            v1_content_fingerprint=views["built"]["v1_content_fingerprint"],
            cache_source=views["built"]["source"],
            labels_present=False,
            train_norm_fingerprint=views["train_norm_fingerprint"],
            v1_train_norm_fingerprint=C.LCRC_TRAIN_NORM_FINGERPRINT,
            n_train=int(views["train_cache"]["n_molecules"]),
            n_valid=int(views["valid_cache"]["n_molecules"]),
            train_rows_sha256=C.rows_sha256_int64(views["rows_train"]),
            valid_rows_sha256=C.rows_sha256_int64(views["rows_valid"]),
        )
    )

    # ------------------------------------------------------------------
    # 4. S2-off capacity arithmetic and shared initialisation
    # ------------------------------------------------------------------
    a_model = __import__("v2_models").build_a(spec, train_norm, C.TRAIN_SEED)
    e_model, e_base, model_audit = CM.build_s2off(spec, train_norm, C.TRAIN_SEED)
    capacity = model_audit["capacity"]
    shared = model_audit["shared_init"]
    a_params = int(sum(p.numel() for p in a_model.parameters()))
    e_params = int(sum(p.numel() for p in e_model.parameters()))
    capacity_ok = (
        a_params == C.LCRC_BASELINE_PARAMS
        and e_params == C.S2OFF_NOMINAL_REFERENCE
        and int(capacity["effective_parameters"]) == C.S2OFF_EFFECTIVE_REFERENCE
        and int(capacity["dead_localmlp_s2_input_columns"]) == C.S2OFF_DEAD_COLUMNS
        and int(capacity["replacement_parameters"]) == C.S2OFF_REPLACEMENT_PARAMS
        and bool(capacity["effective_equals_B_plus_replacement"])
        and bool(capacity["typed_s2_inputs_used"]) is False
        and bool(capacity["iterative_hidden_message_passing"]) is False
        and bool(shared["all_shared_bitwise_identical"])
        and int(shared["n_shared_entries"]) == 27
        and torch.equal(a_model.local.mlp.lins[0].weight, e_model.local.mlp.lins[0].weight)
    )
    results.append(
        _result(
            "s2off_capacity_and_shared_init",
            capacity_ok,
            lcrc_a_parameters=a_params,
            s2off_nominal_parameters=e_params,
            s2off_effective_parameters=int(capacity["effective_parameters"]),
            target_effective=C.S2OFF_EFFECTIVE_REFERENCE,
            effective_delta_vs_lcrc=int(capacity["effective_parameters"]) - a_params,
            dead_s2_input_columns=int(capacity["dead_localmlp_s2_input_columns"]),
            replacement_parameters=int(capacity["replacement_parameters"]),
            effective_equals_B_plus_replacement=bool(capacity["effective_equals_B_plus_replacement"]),
            literal_529_input_live_parameters=int(capacity["literal_529_input_live_parameters_if_slot_not_dead"]),
            shared_init=shared,
            shared_localmlp_bitwise_equal=bool(torch.equal(a_model.local.mlp.lins[0].weight, e_model.local.mlp.lins[0].weight)),
            s2_module_keys_present_in_e=sorted(k for k in e_model.state_dict() if "path_" in k),
            a_only_state_keys=sorted(set(a_model.state_dict()) - set(e_model.state_dict()))[:20],
            e_only_state_keys=sorted(set(e_model.state_dict()) - set(a_model.state_dict()))[:20],
        )
    )

    # ------------------------------------------------------------------
    # 5. S2-off invariance to masked/perturbed typed-S2 inputs
    # ------------------------------------------------------------------
    gate_rows = list(range(16))
    batch = lcrc_data.collate(views["train_cache"], gate_rows)
    invariance = CM.s2off_invariance(e_model, batch)
    restricted_access = False
    try:
        __import__("v2_models").restricted_batch(batch)["path_cat"]
    except KeyError:
        restricted_access = True
    results.append(
        _result(
            "s2off_typed_s2_invariance",
            bool(invariance["identical_restricted_batch"])
            and bool(invariance["identical_perturbed_s2"])
            and float(invariance["max_abs_diff_perturbed"]) == 0.0
            and restricted_access,
            invariance=invariance,
            restricted_batch_access_raises=restricted_access,
            typed_s2_keys=("path_cat", "path_end63", "path_end", "path_closure"),
        )
    )

    # ------------------------------------------------------------------
    # 6. S2-off gradient placement
    # ------------------------------------------------------------------
    grad_gate = CM.s2off_gradient_gate(e_model, batch)
    results.append(
        _result(
            "s2off_gradient_placement",
            bool(grad_gate["all_replacement_receive_nonzero_grad"])
            and bool(grad_gate["dead_s2_input_columns_zero_grad"])
            and bool(grad_gate["live_localmlp_columns_receive_grad"]),
            **grad_gate,
        )
    )
    a_model.zero_grad(set_to_none=True)
    e_model.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # 7. historical parameter counts / requires_grad
    # ------------------------------------------------------------------
    from v2_models import build_a as _unused  # noqa: F401

    pna_native = _build_native_pna(views["rows_train"], width=256)
    lcrc_ok = a_params == C.LCRC_BASELINE_PARAMS and all(p.requires_grad for p in a_model.parameters())
    pna_native_params = int(sum(p.numel() for p in pna_native.parameters()))
    pna_native_ok = pna_native_params == 7_767_349 and all(p.requires_grad for p in pna_native.parameters())
    results.append(
        _result(
            "historical_parameter_counts",
            lcrc_ok and pna_native_ok,
            lcrc_a_parameters=a_params,
            lcrc_a_all_require_grad=all(p.requires_grad for p in a_model.parameters()),
            pna_native_parameters=pna_native_params,
            pna_native_all_require_grad=all(p.requires_grad for p in pna_native.parameters()),
            frozen_lcrc_parameters=C.LCRC_BASELINE_PARAMS,
            frozen_pna_parameters=7_767_349,
        )
    )
    del pna_native

    # ------------------------------------------------------------------
    # 8. PNA width search: frozen, CPU-only, no Validation scores
    # ------------------------------------------------------------------
    width_payload = C.read_json(C.WIDTH_SEARCH)
    score_hits = _scan_no_validation_scores(width_payload)
    recomputed = _recompute_width_counts(views["rows_train"], width_payload["candidates"])
    best = min(recomputed, key=lambda row: (row["absolute_difference"], row["width"]))
    width_ok = (
        width_payload.get("labels_read") is False
        and width_payload.get("validation_scores_read") is False
        and not score_hits
        and int(best["width"]) == int(width_payload["selected_width"])
        and int(best["parameters_total"]) == int(width_payload["selected_parameters_total"])
        and abs(float(best["relative_difference"]) - float(width_payload["relative_difference"])) <= 1e-12
        and bool(width_payload["tightly_size_matched"])
        and int(width_payload["selected_parameters_total"]) == 256_041
    )
    results.append(
        _result(
            "pna_width_search_frozen_cpu_only",
            width_ok,
            selected_width=int(width_payload["selected_width"]),
            selected_parameters_total=int(width_payload["selected_parameters_total"]),
            target_parameters=int(width_payload["target_parameters"]),
            absolute_difference=int(width_payload["absolute_difference"]),
            relative_difference=float(width_payload["relative_difference"]),
            tightly_size_matched=bool(width_payload["tightly_size_matched"]),
            status_label=width_payload["status_label"],
            labels_read=False,
            validation_scores_read=False,
            validation_score_like_keys=score_hits,
            candidates_recomputed=recomputed,
            candidate_count=len(recomputed),
            rule=width_payload["width_rule"],
        )
    )

    # ------------------------------------------------------------------
    # 9. PNA-small parameter count and gradient coverage (architecture only)
    # ------------------------------------------------------------------
    pna_small = _build_native_pna(views["rows_train"], width=int(width_payload["selected_width"]))
    from torch_geometric.data import Batch

    small_batch = Batch.from_data_list(D.load_structure_data(rows=views["rows_train"][:16]))
    small_cov = CM.pna_gradient_coverage(pna_small, small_batch)
    small_params = int(sum(p.numel() for p in pna_small.parameters()))
    results.append(
        _result(
            "pna_small_parameters_and_gradient_coverage",
            small_params == int(width_payload["selected_parameters_total"])
            and bool(small_cov["all_parameter_tensors_receive_grad"])
            and int(small_cov["parameters_with_nonzero_grad_on_batch"]) > 0,
            parameters_total=small_params,
            frozen_width_search_parameters=int(width_payload["selected_parameters_total"]),
            parameters_requires_grad=int(sum(p.numel() for p in pna_small.parameters() if p.requires_grad)),
            gradient_coverage=small_cov,
            only_width_changed=True,
        )
    )
    del pna_small, small_batch

    # ------------------------------------------------------------------
    # 10. no Test prediction artifacts, no accidental Test files
    # ------------------------------------------------------------------
    results.append(_result("prediction_scope_clean", artifact_guard["status"] == "PASS", **artifact_guard))

    failed = [row for row in results if not row["ok"]]
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "PASS" if not failed else "FAIL",
        "n_gates": len(results),
        "n_failed": len(failed),
        "failed": [row["name"] for row in failed],
        "scope": "cpu_only_train_validation_and_parameter_audits",
        "test_labels_read": False,
        "test_predictions_read": False,
        "gpu_used": False,
        "seconds": float(time.perf_counter() - t_start),
        "reuse_audit": reuse,
        "source_hashes": sources,
        "gates": results,
        "env": C.env_info(),
    }
    if write:
        C.write_json(C.CPU_GATES, payload)
    return payload


def _labels_reject_test(labels: L.TrainValidLabels) -> bool:
    try:
        labels.get("test")
    except L.TestLabelAccessError:
        pass
    else:
        return False
    try:
        L.assert_train_valid_only({"train": labels.y_train, "valid": labels.y_valid, "test": np.zeros(420)})
    except L.TestLabelAccessError:
        return True
    return False


def _build_native_pna(train_rows: Sequence[int], *, width: int):
    dims = CD.structure_dimensions()
    train_data = D.load_structure_data(rows=[int(r) for r in train_rows])
    with tempfile.TemporaryDirectory(prefix="gate_pna_") as tmp_name:
        tmp = Path(tmp_name)
        deg_info = CM.pna_degree_histogram(train_data)
        (tmp / "saved_PNA_degrees").mkdir(parents=True, exist_ok=True)
        torch.save(deg_info["tensor"], tmp / "saved_PNA_degrees" / "degrees.pt")
        model = CM.pna_estimator(
            width=int(width),
            num_features=dims["num_features"],
            edge_dim=dims["edge_dim"],
            out_path=tmp,
            seed=C.TRAIN_SEED,
        )
    return model


def _recompute_width_counts(train_rows: Sequence[int], candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dims = CD.structure_dimensions()
    train_data = D.load_structure_data(rows=[int(r) for r in train_rows])
    deg_info = CM.pna_degree_histogram(train_data)
    out: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="gate_width_") as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "saved_PNA_degrees").mkdir(parents=True, exist_ok=True)
        torch.save(deg_info["tensor"], tmp / "saved_PNA_degrees" / "degrees.pt")
        for row in candidates:
            width = int(row["width"])
            model = CM.pna_estimator(
                width=width,
                num_features=dims["num_features"],
                edge_dim=dims["edge_dim"],
                out_path=tmp,
                seed=C.TRAIN_SEED,
            )
            params = int(sum(p.numel() for p in model.parameters()))
            out.append(
                {
                    "width": width,
                    "parameters_total": params,
                    "payload_parameters_total": int(row["parameters_total"]),
                    "matches_payload": params == int(row["parameters_total"]),
                    "absolute_difference": int(abs(params - C.PNA_TARGET_PARAMS)),
                    "relative_difference": float(abs(params - C.PNA_TARGET_PARAMS) / C.PNA_TARGET_PARAMS),
                }
            )
            del model
    if not all(row["matches_payload"] for row in out):
        raise PermissionError("recomputed PNA candidate parameter counts differ from the frozen width search")
    return out


__all__ = ["run_cpu_gates"]
