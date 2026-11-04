"""Label-blind Monitor prediction and the frozen prediction manifest.

Prediction writes only structural inputs -> raw predictions.  It never imports
``monitor_labels``.  The historical A predictions are hash-checked and reused
read-only; they are not retrained.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import common as C
import data as D
import model as M


class PredictionError(RuntimeError):
    pass


def prediction_paths(key: str) -> tuple[Path, Path]:
    return C.PREDICTIONS / f"{key}.npz", C.PREDICTIONS / f"{key}.json"


def counterfactual_paths(key: str) -> tuple[Path, Path]:
    root = C.PREDICTIONS / "diagnostics"
    return root / f"{key}.npz", root / f"{key}.json"


def _load_soup_and_model(arm: str, seed: int, device):
    run_dir = C.RUNS / f"{arm}_s{seed}"
    train_json = run_dir / "train.json"
    soup_path = run_dir / "soup.pt"
    if not train_json.exists() or not soup_path.exists():
        raise PredictionError(f"missing training artifacts for {arm}_s{seed}")
    train = C.read_json(train_json)
    if train.get("protocol_id") != C.PROTOCOL_ID or train.get("arm") != arm or int(train.get("seed")) != seed:
        raise PredictionError("training manifest does not match the requested arm/seed")
    if train.get("monitor_labels_read") is not False:
        raise PredictionError("training manifest is not certified Monitor-label-free")
    if train.get("recipe", {}).get("soup_members") != list(range(C.EPOCHS - C.SOUP_LAST + 1, C.EPOCHS + 1)):
        raise PredictionError("soup members are not the frozen last 5 epochs")
    soup = torch.load(soup_path, map_location="cpu", weights_only=True)
    if M.state_dict_fingerprint(soup) != train["soup_state_hash"]:
        raise PredictionError("soup tensor hash does not match the training manifest")
    if C.sha256_file(soup_path) != train.get("soup_sha256"):
        raise PredictionError("soup file hash does not match the training manifest")
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    model, info = M.build_model(arm, spec, norm, seed, device)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    model.eval()
    return model, info, train, soup_path


def _predict(
    model,
    cache,
    device,
    target_norm,
    *,
    arm: str,
    mode: str,
    key: str,
    train_meta: Mapping[str, Any],
    out_npz: Path,
    out_json: Path,
    purpose: str,
) -> dict[str, Any]:
    out = D.predict_cache(model, cache, device, return_aux=True)
    # ``predict_cache`` calls model(batch) with the model's current mode; for the
    # counterfactual we force the operator by temporarily switching ``mode``.
    if mode != getattr(model, "mode"):
        saved = model.mode
        model.mode = mode
        out = D.predict_cache(model, cache, device, return_aux=True)
        model.mode = saved
    rows = [int(r) for r in cache["rows"]]
    pred_raw = D.denormalise(out["pred"], target_norm)
    base_raw = D.denormalise(out["pred_base"], target_norm)
    delta_raw = out["delta"] * float(target_norm["scale"])
    diag = D.bond_diagnostics(cache)
    payload = {
        "rows": np.asarray(rows, dtype=np.int64),
        "pred": np.asarray(pred_raw, dtype=np.float64),
        "pred_base": np.asarray(base_raw, dtype=np.float64),
        "delta": np.asarray(delta_raw, dtype=np.float64),
        "n_bonds": np.asarray(diag["n_bonds"], dtype=np.int64),
        "n_bonds_by_type": np.asarray(diag["n_bonds_by_type"], dtype=np.int64),
    }
    C.ensure_dir(out_npz.parent)
    np.savez(out_npz, **payload)
    sha = C.sha256_file(out_npz)
    meta = {
        "protocol_id": C.PROTOCOL_ID,
        "key": key,
        "purpose": purpose,
        "arm": arm,
        "mode": mode,
        "seed": int(train_meta["seed"]),
        "parameters": train_meta.get("model", {}).get("parameters"),
        "parameters_trunk": train_meta.get("model", {}).get("parameters_trunk"),
        "parameters_new": train_meta.get("model", {}).get("parameters_new"),
        "device": str(device),
        "file": C.track_rel(out_npz),
        "sha256": sha,
        "rows": rows,
        "rows_sha256": D.monitor_rows_sha256({"monitor": cache}),
        "n_predictions": int(pred_raw.shape[0]),
        "prediction_mean": float(pred_raw.mean()),
        "prediction_std": float(pred_raw.std()),
        "delta_mean": float(delta_raw.mean()),
        "delta_std": float(delta_raw.std()),
        "delta_abs_max": float(np.abs(delta_raw).max()),
        "pred_base_equals_pred_minus_delta_max_abs": float(np.abs(pred_raw - base_raw - delta_raw).max()),
        "soup_state_hash": train_meta["soup_state_hash"],
        "init_state_hash": train_meta["init_state_hash"],
        "trunk_init_state_hash": train_meta["trunk_init_state_hash"],
        "train_manifest_sha256": C.sha256_file(C.RUNS / f"{arm}_s{train_meta['seed']}" / "train.json"),
        "config_hash": C.config_hash(C.load_config()),
        "target_norm": {
            "mean": float(target_norm["mean"]),
            "scale": float(target_norm["scale"]),
            "source": target_norm["source"],
        },
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
    }
    C.write_json(out_json, meta)
    C.log(f"[predict] {key} ({purpose}) -> {C.track_rel(out_npz)} sha={sha[:12]}")
    return meta


def predict_nn(arm: str, seed: int, device="cpu") -> dict[str, Any]:
    if arm not in C.ARMS:
        raise ValueError(f"arm must be one of {C.ARMS}")
    if seed not in C.SEEDS:
        raise ValueError(f"seed must be one of {C.SEEDS}")
    C.verify_protocol_freeze(require_clean=False)
    model, info, train, _soup = _load_soup_and_model(arm, seed, device)
    caches = D.load_caches(C.load_frozen_spec())
    target_norm = D.load_target_norm(verify=False)
    npz_path, json_path = prediction_paths(f"{arm}_s{seed}")
    meta = _predict(
        model, caches["monitor"]["cache"], device, target_norm,
        arm=arm, mode=arm, key=f"{arm}_s{seed}", train_meta=train,
        out_npz=npz_path, out_json=json_path, purpose="primary",
    )
    if arm == "R":
        cf_npz, cf_json = counterfactual_paths(f"R_s{seed}_nullswap")
        _predict(
            model, caches["monitor"]["cache"], device, target_norm,
            arm=arm, mode="N", key=f"R_s{seed}_nullswap", train_meta=train,
            out_npz=cf_npz, out_json=cf_json, purpose="mechanism_counterfactual_frozen_weights",
        )
    return meta


def freeze_predictions() -> dict[str, Any]:
    frozen = C.verify_protocol_freeze(require_clean=False)
    rows = [int(r) for r in C.load_frozen_split()["split"]["monitor"]]
    rows_sha = hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()

    def _entry(key: str, npz: Path, sidecar: Path) -> dict[str, Any]:
        if not npz.exists() or not sidecar.exists():
            raise PredictionError(f"cannot freeze: missing prediction for {key}")
        meta = C.read_json(sidecar)
        if meta.get("key") != key:
            raise PredictionError(f"{key}: sidecar key mismatch")
        actual = C.sha256_file(npz)
        if actual != meta.get("sha256"):
            raise PredictionError(f"{key}: prediction file changed after its sidecar was written")
        if [int(r) for r in meta.get("rows", [])] != rows or meta.get("rows_sha256") != rows_sha:
            raise PredictionError(f"{key}: predicted rows are not the frozen Monitor rows")
        if meta.get("monitor_labels_read") is not False:
            raise PredictionError(f"{key}: prediction was not label-blind")
        if meta.get("config_hash") != C.config_hash(C.load_config()):
            raise PredictionError(f"{key}: config hash changed between predict and freeze")
        with np.load(npz) as data:
            pred = np.asarray(data["pred"], dtype=np.float64)
        if pred.shape != (len(rows),) or not np.isfinite(pred).all():
            raise PredictionError(f"{key}: invalid prediction vector")
        return {
            "file": C.track_rel(npz),
            "sha256": actual,
            "rows_sha256": rows_sha,
            "arm": meta["arm"],
            "mode": meta["mode"],
            "seed": int(meta["seed"]),
            "purpose": meta.get("purpose"),
            "parameters": meta.get("parameters"),
            "soup_state_hash": meta.get("soup_state_hash"),
            "init_state_hash": meta.get("init_state_hash"),
            "train_manifest_sha256": meta.get("train_manifest_sha256"),
            "sidecar": C.track_rel(sidecar),
            "sidecar_sha256": C.sha256_file(sidecar),
            "prediction_sha256_vector": hashlib.sha256(np.ascontiguousarray(pred, dtype="<f8").tobytes()).hexdigest(),
        }

    entries = {key: _entry(key, *prediction_paths(key)) for key in C.RUN_KEYS}
    counterfactuals = {}
    for seed in C.SEEDS:
        key = f"R_s{seed}_nullswap"
        counterfactuals[key] = _entry(key, *counterfactual_paths(key))
    hist = C.verify_historical_a_predictions()
    historical = {}
    for key, entry in hist["predictions"].items():
        historical[key] = {
            "file": entry["file"],
            "sha256": entry["sha256"],
            "rows_sha256": entry["rows_sha256"],
            "arm": "A",
            "seed": int(key.split("_s")[1]),
            "soup_state_hash": entry["soup_state_hash"],
            "prediction_sha256_vector": entry["pred_sha256"],
            "reference_only": True,
        }
    manifest = {
        "protocol_id": C.PROTOCOL_ID,
        "status": "FROZEN",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_branch": C.git_branch(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.config_hash(C.load_config()),
        "preflight_sha256": C.sha256_file(C.PROTOCOL_FREEZE),
        "code_hashes": C.code_file_hashes(),
        "recipe": frozen["recipe"],
        "rows": rows,
        "rows_sha256": rows_sha,
        "predictions": entries,
        "counterfactuals": counterfactuals,
        "historical_A": historical,
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
        "freeze_order": "six R/N predictions and three historical A predictions hashed before any Monitor label access",
    }
    if C.PREDICTION_MANIFEST.exists() or C.PREDICTION_LOCK.exists():
        existing = _load_existing_manifest(manifest)
        C.log("[freeze] predictions already frozen; existing manifest verified, not rewritten")
        return existing
    C.write_json(C.PREDICTION_MANIFEST, manifest)
    sha = C.sha256_file(C.PREDICTION_MANIFEST)
    C.write_json(C.PREDICTION_LOCK, {"sha256": sha, "path": C.track_rel(C.PREDICTION_MANIFEST), "created_utc": C.utcnow()})
    C.log(f"[freeze] frozen {len(entries)} R/N predictions, {len(historical)} A predictions, manifest sha={sha[:12]}")
    manifest["manifest_sha256"] = sha
    return manifest


def _load_existing_manifest(rebuilt: Mapping[str, Any]) -> dict[str, Any]:
    if not C.PREDICTION_MANIFEST.exists() or not C.PREDICTION_LOCK.exists():
        raise PredictionError("partial prediction freeze: manifest and lock must both exist")
    lock = C.read_json(C.PREDICTION_LOCK)
    actual = C.sha256_file(C.PREDICTION_MANIFEST)
    if lock.get("sha256") != actual:
        raise PredictionError("existing prediction manifest does not match its lock")
    existing = C.read_json(C.PREDICTION_MANIFEST)
    if existing.get("status") != "FROZEN":
        raise PredictionError("existing prediction manifest is not FROZEN")
    if existing.get("rows") != rebuilt["rows"]:
        raise PredictionError("existing prediction manifest rows differ from the frozen split")
    for bucket in ("predictions", "counterfactuals", "historical_A"):
        if set(existing.get(bucket, {})) != set(rebuilt.get(bucket, {})):
            raise PredictionError(f"existing prediction registry bucket {bucket} is incomplete")
    for bucket in ("predictions", "counterfactuals"):
        for key, entry in existing[bucket].items():
            path = C.REPO / entry["file"]
            if not path.exists() or C.sha256_file(path) != entry["sha256"]:
                raise PredictionError(f"{key}: prediction file no longer matches the frozen manifest")
            if entry["sha256"] != rebuilt[bucket][key]["sha256"]:
                raise PredictionError(f"{key}: prediction content differs from the current run; refusing to overwrite a freeze")
    existing["manifest_sha256"] = actual
    return existing


__all__ = ["PredictionError", "prediction_paths", "counterfactual_paths", "predict_nn", "freeze_predictions"]
