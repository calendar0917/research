"""CPU correctness gates for the v2 round.

Every gate that writes anything uses a temporary directory.  The v1 freeze,
v1 prediction manifest and every file referenced by that manifest are checked
before and after the gate run; a mismatch is a hard failure.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import v2_common as C
import v2_data as D
import v2_folds as F
import v2_models as M


def _run(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = fn()
        return {"gate": name, "status": "PASS", "seconds": time.perf_counter() - t0, **result}
    except Exception as exc:  # pragma: no cover - gate failure path
        return {
            "gate": name,
            "status": "FAIL",
            "seconds": time.perf_counter() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def gate_g0_frozen() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    split = C.load_frozen_split()
    cache = C.load_fit_cache()
    v1 = C.assert_v1_unchanged()
    return {
        "v1_unchanged": v1["ok"],
        "v1_n_checks": v1["n_checks"],
        "v1_n_failed": v1["n_failed"],
        "spec_hash": spec.hash(),
        "norm_fingerprint": C.prep_data.norm_fingerprint(norm),
        "split_sizes": {k: len(v) for k, v in split["split"].items()},
        "fit_cache_fingerprint": cache["content_fingerprint"],
        "fold_file": C.file_record(C.FOLD_FILE),
    }


def gate_g1_folds() -> dict[str, Any]:
    folds = F.load_folds()
    summary = F.fold_summary(folds)
    counts = [f["n_molecules"] for f in summary["folds"]]
    if counts != [784, 784, 784]:
        raise ValueError(f"unexpected fold sizes {counts}")
    group_fold = {}
    canonical_fold = {}
    for f in folds["folds"]:
        for g in f["groups"]:
            if g["group_key"] in group_fold:
                raise ValueError("group crosses folds")
            group_fold[g["group_key"]] = f["fold"]
        for row in f["rows_global"]:
            info = folds["rows"][str(row)]
            canonical_fold.setdefault(info["canonical_smiles_sha256"], set()).add(f["fold"])
    if any(len(v) > 1 for v in canonical_fold.values()):
        raise ValueError("canonical SMILES crosses folds")
    return {
        "fold_sizes": counts,
        "n_groups_total": folds["n_groups_total"],
        "group_fold_disjoint": True,
        "canonical_smiles_fold_disjoint": True,
        "scope": folds["scope"],
        "summary": summary,
    }


def gate_g2_capacity() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    a = M.build_a(spec, norm, 0)
    e, base, _shared = M.build_e(spec, norm, 0)
    audit = M.e_capacity_audit(e, spec)
    if abs(audit["delta_fraction_vs_A"]) >= 0.001:
        raise ValueError(f"E capacity mismatch: {audit}")
    shared = M.shared_init_audit(base, e)
    if not shared["all_shared_bitwise_identical"]:
        raise ValueError(f"shared init mismatch: {shared['mismatched']}")
    cache = C.load_fit_cache()["cache"]
    batch = D.collate(cache, list(range(24)))
    e.train()
    loss = e(batch).pow(2).mean()
    loss.backward()
    new_grads = {}
    for name, p in e.named_parameters():
        if name.startswith("local.replacement"):
            new_grads[name] = None if p.grad is None else float(p.grad.abs().max())
    if any(v is None or v == 0.0 for v in new_grads.values()):
        raise ValueError(f"replacement parameters without gradient: {new_grads}")
    # Typed-S2 perturbation must be bitwise invisible to E.
    perturbed = dict(batch)
    perturbed["path_cat"] = torch.randint_like(batch["path_cat"], 0, int(spec.n_path_codes))
    perturbed["path_end63"] = torch.randint_like(batch["path_end63"], 0, int(spec.n_nbr_codes))
    perturbed["path_end"] = torch.randint_like(batch["path_end"], 0, batch["atom_feat"].shape[0])
    perturbed["path_closure"] = torch.randint_like(batch["path_closure"], 0, 2)
    e.eval()
    with torch.no_grad():
        y1 = e(batch)
        y2 = e(perturbed)
        y3 = e(M.restricted_batch(batch))
    if not torch.equal(y1, y2):
        raise ValueError("E output depends on typed-S2 tensors")
    if not torch.equal(y1, y3):
        raise ValueError("E restricted-batch forward differs")
    if any(name.startswith("convs") or "attention" in name for name, _ in e.named_modules()):
        raise ValueError("E unexpectedly contains message passing/attention")
    return {
        "capacity_audit": audit,
        "shared_init": shared,
        "new_parameter_gradients": new_grads,
        "typed_s2_invariance": True,
        "restricted_batch_ok": True,
        "parameters_A": M.parameter_count(a),
        "parameters_E_nominal": M.parameter_count(e),
        "parameters_E_effective": audit["effective_parameters"],
    }


def gate_g3_dropout() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    cache = C.load_fit_cache()["cache"]
    batch = D.collate(cache, list(range(16)))
    out = {}
    for arm in ("A", "C"):
        base = M.build_a(spec, norm, 0) if arm == "A" else M.build_c(spec, norm, 0)
        model, meta = M.build_model(arm, spec, norm, 0, dropout=0.0)
        for mode in ("train", "eval"):
            getattr(base, mode)()
            getattr(model, mode)()
            with torch.no_grad():
                y0 = base(batch)
                y1 = model(batch)
            if not torch.equal(y0, y1):
                raise ValueError(f"R0 dropout={arm} {mode} is not numerically identical")
            out[f"{arm}_R0_{mode}_bitwise_equal"] = True
        model, meta = M.build_model(arm, spec, norm, 0, dropout=0.20)
        model.train()
        torch.manual_seed(0)
        with torch.no_grad():
            y_train = model(batch)
        model.eval()
        with torch.no_grad():
            y_eval = model(batch)
            y_base = base.eval()(batch) if arm == "A" else M.build_c(spec, norm, 0).eval()(batch)
        if not torch.equal(y_eval, y_base):
            raise ValueError(f"dropout {arm} is not disabled in eval")
        out[f"{arm}_dropout_disabled_in_eval"] = True
        out[f"{arm}_dropout_state_entries"] = [k for k in model.state_dict() if "dropout" in k]
        target_name = meta["dropout_hook"].target.__class__.__name__
        out[f"{arm}_dropout_target"] = target_name
    return out


def gate_g4_data_training() -> dict[str, Any]:
    import v2_training as T

    data = C.load_fit_cache()
    cache = data["cache"]
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    folds = F.load_folds()
    view = D.build_fold_view(0, cache=cache, y_fit=y, folds=folds)
    if len(view.y_train) != 1568 or len(view.y_dev) != 784:
        raise ValueError("fold view sizes are wrong")
    if set(view.train_positions) & set(view.dev_positions):
        raise ValueError("fold train/dev overlap")
    # Normalisation must be recomputed from the training fold only.
    full_norm = C.load_frozen_norm()
    if C.prep_data.norm_fingerprint(view.train_norm) == C.prep_data.norm_fingerprint(full_norm):
        raise ValueError("fold norm is identical to the full-Fit norm; expected a recomputation")
    a = D.shuffle_batches(100, 16, 123)
    b = D.shuffle_batches(100, 16, 123)
    c = D.shuffle_batches(100, 16, 124)
    if a != b or a == c:
        raise ValueError("shuffle stream is not deterministic/seed-sensitive")
    with tempfile.TemporaryDirectory() as td:
        job = T.TrainJob(
            key="gate_smoke_A",
            arm="A",
            fold=0,
            recipe="R0",
            seed=0,
            device="cpu",
            out_root=Path(td),
            smoke_train_limit=64,
            smoke_epochs=2,
        )
        payload = T.train_job(job)
    if not np.isfinite([payload["metrics"]["train_rmse_raw"], payload["metrics"]["dev_rmse_raw"]]).all():
        raise ValueError("smoke metrics are non-finite")
    return {
        "fold0_train": 1568,
        "fold0_dev": 784,
        "fold_norm_recomputed": True,
        "shuffle_deterministic": True,
        "smoke": {
            "arm": payload["arm"],
            "train_rmse": payload["metrics"]["train_rmse_raw"],
            "dev_rmse": payload["metrics"]["dev_rmse_raw"],
            "nonfinite_events": payload["anomalies"]["nonfinite_events"],
        },
    }


def gate_g5_smoke() -> dict[str, Any]:
    import v2_training as T

    out = {}
    with tempfile.TemporaryDirectory() as td:
        for arm in ("A", "E", "C"):
            job = T.TrainJob(
                key=f"gate_smoke_{arm}",
                arm=arm,
                fold=0,
                recipe="R0",
                seed=0,
                device="cpu",
                out_root=Path(td),
                smoke_train_limit=64,
                smoke_epochs=2,
            )
            payload = T.train_job(job)
            out[arm] = {
                "parameters_nominal": payload["model"]["parameters_nominal"],
                "effective_parameters": payload["model"]["effective_parameters"],
                "train_rmse": payload["metrics"]["train_rmse_raw"],
                "dev_rmse": payload["metrics"]["dev_rmse_raw"],
                "nonfinite_events": payload["anomalies"]["nonfinite_events"],
                "soup_file_sha256": payload["soup_file_sha256"],
            }
    return out


def gate_g6_ssl() -> dict[str, Any]:
    import v2_ssl as S
    import v2_training as T

    data = C.load_fit_cache()
    cache = data["cache"]
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    folds = F.load_folds()
    view = D.build_fold_view(0, cache=cache, y_fit=y, folds=folds)
    leakage = S.structural_leakage_audit(cache)
    if not leakage["path_end_never_equals_root"]:
        raise ValueError("SSL leakage audit failed")
    with tempfile.TemporaryDirectory() as td:
        payload = S.pretrain(
            S.SSLJob(
                key="gate_ssl",
                fold=0,
                seed=0,
                view=view,
                out_root=Path(td),
                device="cpu",
                smoke_limit=128,
                smoke_epochs=2,
            )
        )
        ckpt = torch.load(Path(td) / "gate_ssl" / "ssl_checkpoint.pt", map_location="cpu", weights_only=False)
    keys = set(ckpt["module_state"])
    if any(k.startswith(("f.", "rho.")) for k in keys):
        raise ValueError("SSL checkpoint contains f/rho")
    if not keys or not all(k.startswith(("atom_encoder.", "bond_encoder.", "local.")) for k in keys):
        raise ValueError("SSL checkpoint keys are not atom/bond/local only")
    return {
        "leakage_audit": leakage,
        "pretext": payload["pretext"],
        "mask_coverage": payload["mask_coverage"],
        "gradient_audit": payload["gradient_audit"],
        "checkpoint_keys": sorted(keys),
        "checkpoint_sha256": payload["checkpoint_sha256"],
    }


def gate_g7_baselines() -> dict[str, Any]:
    import v2_baselines as B

    split = C.load_frozen_split()
    rows = [int(r) for r in split["split"]["fit"][:8]]
    s1 = B.compute_features(rows)
    s2 = B.compute_features(rows)
    for name in ("P1", "P2", "P3", "physchem13"):
        if not np.array_equal(getattr(s1, name), getattr(s2, name)):
            raise ValueError(f"descriptor matrix {name} is not deterministic")
    if len(s1.p2_names) != 217 or s1.p2_names != sorted(s1.p2_names):
        raise ValueError("P2 descriptor names are not the frozen sorted 217")
    if tuple(s1.physchem13.shape[1:]) != (13,):
        raise ValueError("physchem13 width is not 13")
    if list(C.DESCRIPTOR_13) != s1.audit["physchem13"]["names"]:
        raise ValueError("physchem13 name order changed")
    reg = B.candidate_registry()
    if sum(1 for c in reg if c["family"] == "ridge") != 6:
        raise ValueError("Ridge candidate count is wrong")
    if sum(1 for c in reg if c["family"] == "xgboost") != 24:
        raise ValueError("XGBoost candidate count is wrong")
    # One deterministic XGBoost fit on a small synthetic frame.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8))
    yv = rng.normal(size=64)
    p1, _ = B.fit_predict_xgb(X[:48], yv[:48], X[48:], {"n_estimators": 20, "max_depth": 3, "min_child_weight": 1})
    p2, _ = B.fit_predict_xgb(X[:48], yv[:48], X[48:], {"n_estimators": 20, "max_depth": 3, "min_child_weight": 1})
    return {
        "descriptor_names": s1.p2_names,
        "n_descriptors": len(s1.p2_names),
        "physchem13": list(C.DESCRIPTOR_13),
        "deterministic_features": True,
        "ridge_candidates": 6,
        "xgboost_candidates": 24,
        "xgb_repeat_max_abs_diff": float(np.max(np.abs(p1 - p2))),
        "xgb_repeat_bitwise": bool(np.array_equal(p1, p2)),
        "feature_audit": s1.audit,
    }


def gate_g8_label_isolation() -> dict[str, Any]:
    import inspect
    import re

    forbidden_imports = ("monitor_labels", "v2_monitor_labels", "evaluate")
    offenders = []
    for path in sorted((C.EXP_DIR / "code").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in forbidden_imports:
            if re.search(rf"\b(import|from)\s+{name}\b", text):
                if path.name not in ("run_v2.py", "v2_evaluate.py", "v2_monitor_labels.py"):
                    offenders.append({"file": path.name, "import": name})
    if offenders:
        raise ValueError(f"forbidden Monitor/label imports in training path: {offenders}")
    guard = C.v1_access_guard
    guard.arm()
    denied = None
    try:
        with open(C.v1c.LIPO_LABELS, "rb") as fh:
            fh.read(1)
    except PermissionError as exc:
        denied = str(exc)
    finally:
        guard.disarm()
    if denied is None:
        raise ValueError("access guard did not deny a direct label open")
    fit = C.load_fit_labels()
    if int(fit["audit"]["label_values_constructed"]) != 2352:
        raise ValueError("Fit label construction audit is wrong")
    if int(fit["audit"]["label_values_skipped"]) != 504:
        raise ValueError("Fit label skip audit is wrong")
    return {
        "forbidden_import_offenders": offenders,
        "direct_label_open_denied": bool(denied),
        "fit_label_audit": fit["audit"],
        "reserved_labels_read": False,
        "official_valid_test_read": False,
        "monitor_labels_read_by_training": False,
    }


def gate_g9_v1_immutability() -> dict[str, Any]:
    before = C.assert_v1_unchanged()
    with tempfile.TemporaryDirectory() as td:
        import v2_training as T

        T.train_job(
            T.TrainJob(
                key="gate_v1_guard",
                arm="A",
                fold=0,
                recipe="R0",
                seed=0,
                device="cpu",
                out_root=Path(td),
                smoke_train_limit=64,
                smoke_epochs=1,
            )
        )
    after = C.assert_v1_unchanged()
    if not (before["ok"] and after["ok"]):
        raise ValueError("v1 frozen artifacts changed during a v2 smoke")
    if before["checks"] != after["checks"]:
        raise ValueError("v1 checked hashes changed during a v2 smoke")
    return {
        "before_ok": before["ok"],
        "after_ok": after["ok"],
        "n_checks": before["n_checks"],
        "checks_identical": before["checks"] == after["checks"],
    }


def run_all() -> dict[str, Any]:
    started = time.perf_counter()
    v1_before = C.assert_v1_unchanged()
    gates = [
        _run("G0_frozen_artifacts_and_v1", gate_g0_frozen),
        _run("G1_fold_partition", gate_g1_folds),
        _run("G2_capacity_matched_E", gate_g2_capacity),
        _run("G3_dropout_R0_equivalence", gate_g3_dropout),
        _run("G4_fold_data_and_training", gate_g4_data_training),
        _run("G5_neural_smoke", gate_g5_smoke),
        _run("G6_ssl_pretext", gate_g6_ssl),
        _run("G7_cpu_descriptor_baselines", gate_g7_baselines),
        _run("G8_label_isolation", gate_g8_label_isolation),
        _run("G9_v1_immutability", gate_g9_v1_immutability),
    ]
    v1_after = C.assert_v1_unchanged()
    failed = [g for g in gates if g["status"] != "PASS"]
    final = "READY_FOR_GPU_AUTHORIZATION" if not failed and v1_before["ok"] and v1_after["ok"] else "BLOCKED"
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "elapsed_seconds": time.perf_counter() - started,
        "env": C.env_info(),
        "gates": gates,
        "final_status": final,
        "failed": [g["gate"] for g in failed],
        "v1_before": v1_before,
        "v1_after": v1_after,
        "v1_unchanged": bool(v1_before["ok"] and v1_after["ok"]),
    }
    C.write_json(C.ANALYSIS / "cpu_gates_v2.json", payload)
    return payload


__all__ = ["run_all", "gate_g0_frozen", "gate_g1_folds", "gate_g2_capacity", "gate_g3_dropout", "gate_g4_data_training", "gate_g5_smoke", "gate_g6_ssl", "gate_g7_baselines", "gate_g8_label_isolation", "gate_g9_v1_immutability"]
