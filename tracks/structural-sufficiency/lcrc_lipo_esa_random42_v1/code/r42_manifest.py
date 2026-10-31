"""Freeze the pre-test manifest and the per-model Test prediction files.

Order of operations (each step refuses if the previous one is missing):

1. every model that finished has a ``train.json`` with a validation metric and a
   checkpoint hash;
2. the validation predictions and the frozen configuration hashes are read from
   disk (nothing is re-trained here);
3. the Test420 structure order is read (structures only - no labels);
4. a frozen Test prediction is written per completed model, using the frozen
   checkpoint, and hashed;
5. ``analysis/pre_test_manifest.json`` lists models, checkpoints, prediction
   paths, hashes, the split hash and the ledger total;
6. ``analysis/prediction_manifest.json`` is the per-file hash list.

After this file exists the models may not change: the scorer only reads it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

import r42_common as C
import r42_data as D
import r42_labels as L
import r42_lcrc_train as LT
import r42_native_train as NT

MODEL_ORDER = ("LCRC-A-R3-scratch", "PNA-native", "ESA-native", "XGB-P3")
RUN_DIRS = {
    "LCRC-A-R3-scratch": "lcrc_a_r3_s0",
    "PNA-native": "pna_native_s0",
    "ESA-native": "esa_native_s0",
    "XGB-P3": "xgb_p3_s0",
}


def _views_for_norm():
    split = D.load_split()
    labels = L.load_train_valid_labels()
    return split, labels


def _lcrc_predictor() -> Callable[[list[int]], np.ndarray]:
    split, labels = _views_for_norm()
    views = D.build_lcrc_views(split, labels)
    spec = views["spec"]
    train_norm = views["train_norm"]
    built = views["built"]
    target_norm = views["target_norm"]
    state = torch.load(C.CHECKPOINTS / "lcrc_a_r3_s0.pt", map_location="cpu", weights_only=False)
    import lcrc_model

    model = lcrc_model.LCRC(spec, train_norm, lcrc_model.DEFAULT_CONFIG)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    cache_all = built["cache"]

    def predict(rows: list[int]) -> np.ndarray:
        cache = D.select_cache(cache_all, rows, split_label="test420")
        raw_std = LT._predict_lcrc(model, cache, torch.device("cpu"), batch_size=256)
        return NT.denormalise(raw_std, target_norm)

    info = {
        "cache_sha256": built["file_sha256"],
        "cache_content_fingerprint": built["content_fingerprint"],
        "train_norm_fingerprint": views["train_norm_fingerprint"],
        "target_norm": target_norm,
        "soup_state_hash": state.get("soup_state_hash"),
        "soup_members": state.get("soup_members"),
    }
    return predict, info


def _native_predictor(model: str) -> Callable[[list[int]], np.ndarray]:
    cfg = C.load_config()
    run_dir = C.RUNS / RUN_DIRS[model]
    payload = C.read_json(run_dir / "train.json")
    ckpt = C.CHECKPOINTS / ("pna_native_s0.pt" if model == "PNA-native" else "esa_native_s0.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    structures = D.structure_summary()
    kind = "pna" if model == "PNA-native" else "esa"
    model_cfg = cfg["models"][kind]
    device = torch.device("cpu")
    if kind == "pna":
        net, extra = NT.build_pna(model_cfg, device, run_dir, D.load_structure_data([int(r) for r in D.load_split()["rows"]["train"]]), structures["atom_feat_dim"], structures["edge_feat_dim"])
        num_max_items = None
    else:
        net, extra = NT.build_esa(model_cfg, device, run_dir, structures["max_edge_global"], structures["atom_feat_dim"], structures["edge_feat_dim"])
        mods = NT.U.install()
        num_max_items = int(mods["esa_models"].nearest_multiple_of_8(structures["max_edge_global"] + 1))
    net.load_state_dict(state["state_dict"], strict=True)
    net.eval()
    target_norm = payload["data"]["target_norm"]
    precision = str(model_cfg.get("precision", "bf16"))

    def predict(rows: list[int]) -> np.ndarray:
        data = D.load_structure_data(rows)
        raw_std = NT._predict(net, data, kind=kind, device=device, batch_size=int(model_cfg["batch_size"]), precision="fp32", num_max_items=num_max_items)
        return NT.denormalise(raw_std, target_norm)

    info = {
        "checkpoint_best_epoch": state.get("best_epoch"),
        "val_loss_standardized": state.get("val_loss_standardized"),
        "target_norm": target_norm,
        "extra": extra,
        "formal_precision": precision,
        "test_inference_precision": "fp32",
    }
    return predict, info


def _xgb_predictor() -> Callable[[list[int]], np.ndarray]:
    import r42_xgb as X

    run_dir = C.RUNS / RUN_DIRS["XGB-P3"]
    payload = C.read_json(run_dir / "train.json")

    def predict(rows: list[int]) -> np.ndarray:
        return X.predict_rows(C.REPO / payload["model_path"], rows)

    info = {
        "model_sha256": payload["model_sha256"],
        "model_path": payload["model_path"],
        "n_features_used": payload["features"]["n_features_used"],
        "preprocessor_fit_scope": payload["features"]["preprocessor"]["fit_scope"],
    }
    return predict, info


def build_pre_test_manifest() -> dict[str, Any]:
    split = D.load_split()
    labels = L.load_train_valid_labels()
    test_rows = [int(r) for r in split["rows"]["test"]]
    if len(test_rows) != C.SPLIT_SIZES["test"]:
        raise ValueError("Test split size mismatch")
    if C.rows_sha256_int64(test_rows) != C.SPLIT_SHA256["test"]:
        raise ValueError("Test split hash mismatch")

    C.ensure_dir(C.CHECKPOINTS)
    C.ensure_dir(C.PREDICTIONS)
    entries: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        run_dir = C.RUNS / RUN_DIRS[model]
        train_json = run_dir / "train.json"
        if not train_json.exists():
            missing.append({"model": model, "reason": "not_run", "expected": C.track_rel(train_json)})
            continue
        payload = C.read_json(train_json)
        # Freeze the run checkpoint into the experiment's own checkpoint store.
        if model == "XGB-P3":
            src = C.REPO / payload["model_path"]
            dst = C.CHECKPOINTS / "xgb_p3_s0.json"
        else:
            src = C.REPO / payload["checkpoint_path"]
            dst = C.CHECKPOINTS / {"LCRC-A-R3-scratch": "lcrc_a_r3_s0.pt", "PNA-native": "pna_native_s0.pt", "ESA-native": "esa_native_s0.pt"}[model]
        if not src.exists():
            missing.append({"model": model, "reason": "checkpoint_missing", "expected": C.track_rel(src)})
            continue
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        if C.sha256_file(dst) != C.sha256_file(src):
            raise PermissionError(f"{model}: frozen checkpoint copy differs from the run artifact")
        entries.append(
            {
                "model": model,
                "seed": int(payload.get("seed", 0)),
                "run_dir": C.track_rel(run_dir),
                "train_json_sha256": C.sha256_file(train_json),
                "checkpoint_source": C.track_rel(src),
                "checkpoint_frozen": C.track_rel(dst),
                "checkpoint_sha256": C.sha256_file(dst),
                "parameters_total": payload.get("model_audit", {}).get("parameters_total"),
                "config_hash": payload.get("config_hash"),
                "checkpoint_rule": payload.get("training", {}).get("checkpoint_rule"),
                "validation": payload.get("metrics", {}).get("valid"),
                "train_metrics": payload.get("metrics", {}).get("train"),
            }
        )

    builders = {
        "LCRC-A-R3-scratch": _lcrc_predictor,
        "PNA-native": lambda: _native_predictor("PNA-native"),
        "ESA-native": lambda: _native_predictor("ESA-native"),
        "XGB-P3": _xgb_predictor,
    }
    predictions: list[dict[str, Any]] = []
    for entry in entries:
        model = entry["model"]
        predict, info = builders[model]()
        values = np.asarray(predict(test_rows), dtype=np.float64).reshape(-1)
        if values.shape != (len(test_rows),) or not np.isfinite(values).all():
            raise RuntimeError(f"{model}: invalid Test prediction vector")
        path = C.PREDICTIONS / f"{RUN_DIRS[model]}_test420.npz"
        np.savez(path, rows=np.asarray(test_rows, dtype=np.int64), pred=values)
        predictions.append(
            {
                "model": model,
                "path": C.track_rel(path),
                "sha256": C.sha256_file(path),
                "n": int(values.size),
                "pred_mean": float(values.mean()),
                "pred_std": float(values.std()),
                "pred_min": float(values.min()),
                "pred_max": float(values.max()),
                "rows_sha256": C.rows_sha256_int64(test_rows),
                "predictor_audit": info,
            }
        )

    import r42_ledger

    ledger = r42_ledger.build_ledger()
    manifest = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "purpose": "freeze every model, checkpoint and Test prediction before the single Test evaluation",
        "split": {
            "path": C.track_rel(C.SPLIT_FILE),
            "file_sha256": C.sha256_file(C.SPLIT_FILE),
            "test_rows_sha256": C.rows_sha256_int64(test_rows),
            "n_test": len(test_rows),
        },
        "label_artifact": labels.audit,
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "config_canonical_sha256": C.sha256_json(C.load_config()),
        "code_hashes": {str(p.relative_to(C.EXP_DIR)): C.sha256_file(p) for p in sorted(C.CODE_DIR.glob("r42_*.py"))},
        "models": entries,
        "predictions": predictions,
        "missing_models": missing,
        "gpu_hours_charged": ledger["gpu_hours_charged"],
        "test_scoring_allowed": len(entries) > 0,
        "rules": {
            "single_evaluation": True,
            "retraining_after_test": False,
            "checkpoint_reselection_after_test": False,
            "metrics_reported_in": "raw_logD",
        },
    }
    C.write_json(C.ANALYSIS / "prediction_manifest.json", {"protocol_id": C.PROTOCOL_ID, "predictions": predictions})
    C.write_json(C.ANALYSIS / "pre_test_manifest.json", manifest)
    manifest["manifest_sha256"] = C.sha256_file(C.ANALYSIS / "pre_test_manifest.json")
    C.write_json(C.ANALYSIS / "pre_test_manifest.sha256", {"pre_test_manifest_sha256": manifest["manifest_sha256"]})
    return manifest


def build_implementation_manifest() -> dict[str, Any]:
    """Record what was actually implemented, before any GPU time is spent."""
    import subprocess

    import r42_gates
    import r42_upstream as U

    cfg = C.load_config()
    split = D.load_split()
    labels = L.load_train_valid_labels()
    structures = D.structure_summary()

    # Parameter counts, built exactly as the trainers build them.
    views = D.build_lcrc_views(split, labels)
    import lcrc_model

    torch.manual_seed(0)
    lcrc = lcrc_model.LCRC(views["spec"], views["train_norm"], lcrc_model.DEFAULT_CONFIG)
    lcrc_params = int(sum(p.numel() for p in lcrc.parameters()))

    pna_dir = C.RUNS / "manifest_pna"
    pna_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [int(r) for r in split["rows"]["train"]]
    pna, pna_deg = NT.build_pna(cfg["models"]["pna"], torch.device("cpu"), pna_dir, D.load_structure_data(train_rows), structures["atom_feat_dim"], structures["edge_feat_dim"])
    esa_dir = C.RUNS / "manifest_esa"
    esa_dir.mkdir(parents=True, exist_ok=True)
    esa, esa_extra = NT.build_esa(cfg["models"]["esa"], torch.device("cpu"), esa_dir, structures["max_edge_global"], structures["atom_feat_dim"], structures["edge_feat_dim"])

    code_hashes = {str(p.relative_to(C.EXP_DIR)): C.sha256_file(p) for p in sorted(C.CODE_DIR.glob("r42_*.py"))}
    test_hashes = {str(p.relative_to(C.EXP_DIR)): C.sha256_file(p) for p in sorted((C.EXP_DIR / "tests").glob("*.py"))}
    upstream_hashes = U.upstream_file_hashes()

    pytest_run = subprocess.run(
        ["python", "-m", "pytest", str(C.EXP_DIR / "tests"), "-q", "-p", "no:cacheprovider"],
        cwd=str(C.REPO),
        capture_output=True,
        text=True,
    )
    tail = [ln for ln in pytest_run.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
    tests_ok = pytest_run.returncode == 0

    manifest = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "config": {"path": C.track_rel(C.CONFIG_PATH), "sha256": C.sha256_file(C.CONFIG_PATH), "canonical_sha256": C.sha256_json(cfg)},
        "code_hashes": code_hashes,
        "test_hashes": test_hashes,
        "upstream": {
            "repository": C.UPSTREAM_REPO,
            "commit": C.UPSTREAM_COMMIT,
            "vendored_files": upstream_hashes,
            "runtime_stubs": dict(U.STUB_MANIFEST),
            "runtime_patches": list(U.PATCHES),
        },
        "models": {
            "LCRC-A-R3-scratch": {
                "implementation": "frozen local LCRC (tracks/.../lcrc_model.py, lcrc_data.py) with the lipo_v1 graph spec",
                "parameters_total": lcrc_params,
                "recipe": cfg["models"]["lcrc"],
                "checkpoint_rule": "average of the last 5 epoch-end state_dicts",
                "early_stopping": False,
                "ssl": False,
                "extra_modules": False,
            },
            "PNA-native": {
                "implementation": "vendored upstream gnn/graph_models.py:PNA via Estimator",
                "parameters_total": int(sum(p.numel() for p in pna.parameters())),
                "gnn_parameters_total": int(sum(p.numel() for p in pna.gnn_model.parameters())),
                "degree_histogram": {k: v for k, v in pna_deg.items() if k != "nonzero"},
                "recipe": cfg["models"]["pna"],
                "checkpoint_rule": "best Validation loss",
                "early_stopping_patience": int(cfg["models"]["pna"]["early_stopping_patience"]),
            },
            "ESA-native": {
                "implementation": "vendored upstream esa/models.py:FullGraphModel over masked_layers",
                "parameters_total": int(sum(p.numel() for p in esa.parameters())),
                "num_max_items": int(esa_extra["num_max_items"]),
                "attention_backend": esa_extra["attention_backend"],
                "recipe": cfg["models"]["esa"],
                "checkpoint_rule": "best Validation loss",
                "early_stopping_patience": int(cfg["models"]["esa"]["early_stopping_patience"]),
            },
            "XGB-P3": {
                "implementation": "xgboost 2.1.3 hist regressor on the frozen v2 P2 RDKit descriptor list",
                "recipe": cfg["models"]["xgb"],
                "checkpoint_rule": "single fit on Train3360",
            },
        },
        "feature_sources": {
            "lcrc": "lipo_v1 atom/bond/path tensors from lcrc_data (Train3360-fitted normalisation only)",
            "pna": "upstream Chemprop atom (79) / bond (13) features, explicit hydrogens",
            "esa": "identical Chemprop features + EdgeSetAttention masked layers, torch attention backend",
            "xgb": "RDKit descriptors only, preprocessor fitted on Train3360",
        },
        "hyperparameter_search": False,
        "seeds": {"split": C.SPLIT_SEED, "train": [0]},
        "tests": {"command": "python -m pytest tests -q", "returncode": pytest_run.returncode, "summary": tail[0], "passed": tests_ok},
        "gpu_used": False,
        "env": C.env_info(),
    }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "model_implementations": ["LCRC-A-R3-scratch", "PNA-native", "ESA-native", "XGB-P3"],
        "checkpoint_rules": cfg["checkpoint_rules"],
        "gpu_budget": cfg["budget"],
        "status": "PASS" if tests_ok else "FAIL",
        **manifest,
    }
    C.write_json(C.ANALYSIS / "implementation_manifest.json", payload)
    C.log(f"[implementation] tests={tail[0]!r} params: LCRC={lcrc_params} PNA={payload['models']['PNA-native']['parameters_total']} ESA={payload['models']['ESA-native']['parameters_total']}")
    if not tests_ok:
        raise SystemExit("unit tests failed; refusing to record a passing implementation manifest")
    return payload


__all__ = ["build_pre_test_manifest", "build_implementation_manifest", "MODEL_ORDER", "RUN_DIRS"]
