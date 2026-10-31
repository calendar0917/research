"""Label-blind Monitor504 prediction and manifest freeze for v2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import v2_common as C
import v2_data as D
import v2_models as M
import v2_registry as R


def monitor_cache() -> dict[str, Any]:
    return C.v1c.load_frozen_cache("monitor")


def monitor_rows() -> list[int]:
    return [int(r) for r in C.load_frozen_split()["split"]["monitor"]]


def predict_nn(key: str) -> dict[str, Any]:
    row = R.run_by_key(key)
    if row["kind"] != "train":
        raise ValueError(f"{key} is not a supervised run")
    run_dir = C.RUNS / key
    train_path = run_dir / "train.json"
    soup_path = run_dir / "soup.pt"
    if not train_path.exists() or not soup_path.exists():
        raise FileNotFoundError(f"missing training artifacts for {key}")
    train = C.read_json(train_path)
    if train.get("key") != key:
        raise ValueError("training manifest key mismatch")
    if train.get("monitor_labels_read") is not False:
        raise PermissionError("training run is not certified Monitor-label-blind")
    recipe_name = str(train["recipe"])
    if recipe_name not in C.RECIPES:
        raise ValueError(f"unknown recipe in training manifest: {recipe_name}")
    recipe = C.RECIPES[recipe_name]
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    model, _meta = M.build_model(row["arm"], spec, norm, int(row["seed"]), "cpu", dropout=float(recipe["dropout"]))
    state = torch.load(soup_path, map_location="cpu", weights_only=False)
    model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    model.eval()
    cache = monitor_cache()["cache"]
    rows = [int(x) for x in cache["molecule_ids"]]
    expected_rows = monitor_rows()
    if rows != expected_rows:
        raise ValueError("Monitor cache order does not match the frozen split")
    target_norm = C.read_json(C.NORM_TARGET_V2)
    pred = D.denormalise(D.predict_view(model, cache, device="cpu"), target_norm)
    if pred.shape != (len(rows),) or not np.isfinite(pred).all():
        raise RuntimeError("invalid Monitor prediction vector")
    # The Monitor labels are on the raw Lipophilicity scale.  A prediction
    # vector centred near zero when the frozen target mean is ~2.16 is the
    # signature of a forgotten denormalisation; fail closed rather than freeze it.
    if abs(float(pred.mean()) - float(target_norm["mean"])) > 3.0 * float(target_norm["scale"]):
        raise RuntimeError(
            f"Monitor prediction mean {pred.mean():.4f} is implausibly far from the frozen target mean "
            f"{target_norm['mean']:.4f}"
        )
    out_dir = C.PREDICTIONS
    C.ensure_dir(out_dir)
    npz_path = out_dir / f"{key}.npz"
    np.savez(npz_path, rows=np.asarray(rows, dtype=np.int64), pred=pred)
    meta = {
        "protocol_id": C.PROTOCOL_ID,
        "key": key,
        "arm": row["arm"],
        "recipe": recipe_name,
        "seed": int(row["seed"]),
        "variant": train["variant"],
        "file": C.track_rel(npz_path),
        "sha256": C.sha256_file(npz_path),
        "rows": rows,
        "rows_sha256": C.row_sha256_bytes(rows),
        "n_predictions": int(pred.shape[0]),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "target_norm_mean": float(target_norm["mean"]),
        "target_norm_scale": float(target_norm["scale"]),
        "train_manifest": C.track_rel(train_path),
        "train_manifest_sha256": C.sha256_file(train_path),
        "soup_file": C.track_rel(soup_path),
        "soup_file_sha256": C.sha256_file(soup_path),
        "soup_state_hash": train["model"]["soup_state_hash"],
        "parameters_nominal": train["model"]["parameters_nominal"],
        "effective_parameters": train["model"]["effective_parameters"],
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
    }
    C.write_json(out_dir / f"{key}.json", meta)
    return meta


def prediction_registry() -> list[dict[str, Any]]:
    recipe_lock = C.read_json(C.RECIPE_LOCK)
    ssl_gate = C.read_json(C.SSL_GATE)
    return R.planned_predictions(recipe_lock, ssl_gate)


def freeze_predictions() -> dict[str, Any]:
    if (C.PREDICTIONS / "MANIFEST.json").exists():
        raise PermissionError("v2 predictions are already frozen; preserve the manifest")
    registry = prediction_registry()
    expected = {row["key"] for row in registry}
    files = sorted(p.stem for p in C.PREDICTIONS.glob("*.npz"))
    missing = sorted(expected - set(files))
    extra = sorted(set(files) - expected)
    if missing:
        raise PermissionError(f"missing predictions: {missing}")
    if extra:
        raise PermissionError(f"unregistered predictions present: {extra}")
    rows = monitor_rows()
    rows_sha = C.row_sha256_bytes(rows)
    preds = {}
    for row in registry:
        key = row["key"]
        npz_path = C.PREDICTIONS / f"{key}.npz"
        json_path = C.PREDICTIONS / f"{key}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"missing sidecar for {key}")
        sidecar = C.read_json(json_path)
        if sidecar.get("monitor_labels_read") is not False:
            raise PermissionError(f"{key} sidecar is not label-blind")
        data = np.load(npz_path)
        if list(map(int, data["rows"].tolist())) != rows:
            raise ValueError(f"{key} row order mismatch")
        if data["pred"].shape != (len(rows),):
            raise ValueError(f"{key} prediction length mismatch")
        if not np.isfinite(data["pred"]).all():
            raise ValueError(f"{key} has a non-finite prediction")
        if C.sha256_file(npz_path) != sidecar["sha256"]:
            raise ValueError(f"{key} file hash differs from its sidecar")
        preds[key] = {
            "arm": row["arm"],
            "recipe": row["recipe"],
            "seed": row["seed"],
            "variant": row["variant"],
            "file": C.track_rel(npz_path),
            "sha256": C.sha256_file(npz_path),
            "sidecar": C.track_rel(json_path),
            "sidecar_sha256": C.sha256_file(json_path),
            "n_predictions": len(rows),
            "prediction_mean": float(data["pred"].mean()),
            "prediction_std": float(data["pred"].std()),
        }
    manifest = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "rows": rows,
        "rows_sha256": rows_sha,
        "manifest_sha256_base": hashlib.sha256(
            C.stable_json(
                {
                    "protocol_id": C.PROTOCOL_ID,
                    "rows_sha256": rows_sha,
                    "predictions": preds,
                }
            ).encode()
        ).hexdigest(),
        "predictions": preds,
        "n_predictions": len(preds),
        "monitor_labels_read": False,
        "label_blind": True,
    }
    C.ensure_dir(C.PREDICTIONS)
    C.write_json(C.PREDICTIONS / "MANIFEST.json", manifest)
    lock = hashlib.sha256((C.PREDICTIONS / "MANIFEST.json").read_bytes()).hexdigest()
    (C.PREDICTIONS / "MANIFEST.sha256").write_text(lock + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = lock
    return manifest


def verify_manifest() -> dict[str, Any]:
    manifest_path = C.PREDICTIONS / "MANIFEST.json"
    lock_path = C.PREDICTIONS / "MANIFEST.sha256"
    if not manifest_path.exists() or not lock_path.exists():
        raise PermissionError("v2 prediction manifest/lock is missing")
    lock = lock_path.read_text(encoding="utf-8").strip()
    actual = C.sha256_file(manifest_path)
    if lock != actual:
        raise PermissionError("v2 prediction manifest sha256 does not match its lock")
    manifest = C.read_json(manifest_path)
    manifest["manifest_sha256"] = lock
    if manifest.get("label_blind") is not True:
        raise PermissionError("v2 manifest is not certified label-blind")
    rows = monitor_rows()
    if manifest.get("rows") != rows:
        raise ValueError("v2 manifest row order mismatch")
    for key, row in manifest["predictions"].items():
        if C.sha256_file(Path(C.REPO) / row["file"]) != row["sha256"]:
            raise PermissionError(f"prediction {key} changed after the freeze")
        if C.sha256_file(Path(C.REPO) / row["sidecar"]) != row["sidecar_sha256"]:
            raise PermissionError(f"prediction sidecar {key} changed after the freeze")
    return manifest


__all__ = ["monitor_cache", "monitor_rows", "predict_nn", "prediction_registry", "freeze_predictions", "verify_manifest"]
