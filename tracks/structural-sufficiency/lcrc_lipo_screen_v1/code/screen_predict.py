"""Label-blind prediction and the prediction freeze.

Order of operations enforced by this module:

1. ``predict-nn`` / ``predict-baseline`` build predictions from structure only.
   No Monitor label is opened, and the modules they import cannot read one.
2. ``freeze-predictions`` verifies that every registered arm x seed prediction
   file exists, hashes it, and writes ``MANIFEST.json`` + ``MANIFEST.sha256``.
3. Only then may :mod:`monitor_labels` (and therefore :mod:`evaluate`) open the
   Monitor label values.

The manifest is immutable: a second freeze attempt with different content is
refused rather than silently overwritten.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import screen_common as C
import screen_data as D
import screen_models as M


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be frozen safely."""


def _expected_keys() -> list[str]:
    keys = [f"{arm}_s{seed}" for arm in C.NN_ARMS for seed in C.SEEDS]
    keys += [f"{arm}_r{r}" for arm in C.BASELINE_ARMS for r in C.BASELINE_REPLICATES]
    return keys


def prediction_paths(key: str) -> tuple[Path, Path]:
    return C.PREDICTIONS / f"{key}.npz", C.PREDICTIONS / f"{key}.json"


# ---------------------------------------------------------------------------
# neural arms
# ---------------------------------------------------------------------------


def predict_nn(arm: str, seed: int, device="cpu") -> dict[str, Any]:
    if arm not in C.NN_ARMS:
        raise ValueError(f"arm must be one of {C.NN_ARMS}")
    if seed not in C.SEEDS:
        raise ValueError(f"seed must be one of {C.SEEDS}")
    C.verify_protocol_freeze(require_clean=False)

    run_dir = C.RUNS / f"{arm}_s{seed}"
    train_json_path = run_dir / "train.json"
    soup_path = run_dir / "soup.pt"
    if not train_json_path.exists() or not soup_path.exists():
        raise PredictionError(f"missing training artifacts for {arm} seed {seed}: {C.track_rel(run_dir)}")
    train = C.read_json(train_json_path)
    if train.get("protocol_id") != C.PROTOCOL_ID or train.get("arm") != arm or int(train.get("seed")) != seed:
        raise PredictionError("training manifest does not match the requested arm/seed")
    if train.get("monitor_labels_read") is not False:
        raise PredictionError("training manifest is not certified Monitor-label-free")
    soup = torch.load(soup_path, map_location="cpu", weights_only=True)
    if M.state_dict_fingerprint(soup) != train["soup_state_hash"]:
        raise PredictionError("soup tensor hash does not match the training manifest")
    if train["recipe"]["soup_members"] != list(range(C.EPOCHS - C.SOUP_LAST + 1, C.EPOCHS + 1)):
        raise PredictionError("soup members are not the frozen last 5 epochs")

    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    model, model_info = M.build_model(arm, spec, norm, seed, device)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    caches = D.load_caches(spec)
    target_norm = D.load_target_norm(verify=False)
    out = D.predict_cache(model, caches["monitor"]["cache"], device, return_aux=(arm == "A"))
    pred_raw = D.denormalise(out["pred"], target_norm)

    rows = [int(r) for r in caches["monitor"]["rows"]]
    npz_path, json_path = prediction_paths(f"{arm}_s{seed}")
    C.ensure_dir(npz_path.parent)
    payload = {
        "rows": np.asarray(rows, dtype=np.int64),
        "pred": np.asarray(pred_raw, dtype=np.float64),
    }
    if arm == "A":
        contrib_raw, rho_raw = D.raw_contribution_terms(out["contrib"], out["rho"], target_norm)
        payload["contrib"] = contrib_raw
        payload["rho"] = rho_raw
        payload["contrib_atom_sum_check"] = contrib_raw + rho_raw - pred_raw
    np.savez(npz_path, **payload)
    sha = C.sha256_file(npz_path)
    meta = {
        "protocol_id": C.PROTOCOL_ID,
        "key": f"{arm}_s{seed}",
        "arm": arm,
        "seed": int(seed),
        "device": str(device),
        "file": C.track_rel(npz_path),
        "sha256": sha,
        "rows": rows,
        "rows_sha256": D.monitor_rows_sha256(caches),
        "soup_state_hash": train["soup_state_hash"],
        "init_state_hash": train["init_state_hash"],
        "train_manifest_sha256": C.sha256_file(train_json_path),
        "config_hash": C.config_hash(C.load_config()),
        "model": model_info,
        "prediction_space": "raw_logD",
        "aux_space": "raw_logD; fit_mean absorbed into rho" if arm == "A" else None,
        "target_norm": {
            "mean": float(target_norm["mean"]),
            "scale": float(target_norm["scale"]),
            "source": target_norm["source"],
        },
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
        "n_predictions": int(payload["pred"].shape[0]),
        "prediction_mean": float(payload["pred"].mean()),
        "prediction_std": float(payload["pred"].std()),
        "aux": (
            {
                "contrib_mean": float(payload["contrib"].mean()),
                "rho_mean": float(payload["rho"].mean()),
                "decomposition_max_abs_error": float(np.abs(payload["contrib_atom_sum_check"]).max()),
            }
            if arm == "A"
            else None
        ),
    }
    C.write_json(json_path, meta)
    C.log(f"[predict] {arm} s{seed} -> {C.track_rel(npz_path)} sha={sha[:12]}")
    return meta


# ---------------------------------------------------------------------------
# prediction freeze
# ---------------------------------------------------------------------------


def freeze_predictions() -> dict[str, Any]:
    frozen = C.verify_protocol_freeze(require_clean=False)
    rows = [int(r) for r in C.load_frozen_split()["split"]["monitor"]]
    rows_sha = hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()
    entries: dict[str, Any] = {}
    for key in _expected_keys():
        npz_path, json_path = prediction_paths(key)
        if not npz_path.exists() or not json_path.exists():
            raise PredictionError(f"cannot freeze: missing prediction for {key}")
        meta = C.read_json(json_path)
        if meta.get("key") != key:
            raise PredictionError(f"{key}: sidecar key mismatch")
        if C.sha256_file(npz_path) != meta.get("sha256"):
            raise PredictionError(f"{key}: prediction file changed after its sidecar was written")
        if [int(r) for r in meta.get("rows", [])] != rows:
            raise PredictionError(f"{key}: predicted rows are not the frozen Monitor rows")
        if meta.get("rows_sha256") != rows_sha:
            raise PredictionError(f"{key}: rows hash mismatch")
        if meta.get("monitor_labels_read") is not False:
            raise PredictionError(f"{key}: prediction was not label-blind")
        if meta.get("config_hash") != C.config_hash(C.load_config()):
            raise PredictionError(f"{key}: config hash changed between predict and freeze")
        entries[key] = {
            "file": meta["file"],
            "sha256": meta["sha256"],
            "rows_sha256": meta["rows_sha256"],
            "arm": meta["arm"],
            "seed": meta["seed"],
            "model": meta["model"].get("model"),
            "parameters": meta["model"].get("parameters"),
            "soup_state_hash": meta.get("soup_state_hash"),
            "train_manifest_sha256": meta.get("train_manifest_sha256"),
            "sidecar": C.track_rel(json_path),
            "sidecar_sha256": C.sha256_file(json_path),
        }
    manifest = {
        "protocol_id": C.PROTOCOL_ID,
        "status": "FROZEN",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.config_hash(C.load_config()),
        "protocol_freeze_sha256": C.sha256_file(C.PROTOCOL_FREEZE),
        "code_hashes": C.code_file_hashes(),
        "recipe": frozen["recipe"],
        "rows": rows,
        "rows_sha256": rows_sha,
        "predictions": entries,
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
        "freeze_order": "predictions_written_and_hashed_before_any_monitor_label_access",
    }
    if C.PREDICTION_MANIFEST.exists() or C.PREDICTION_LOCK.exists():
        existing = _load_existing_manifest(manifest)
        C.log("[freeze] predictions already frozen; existing manifest verified, not rewritten")
        return existing
    C.write_json(C.PREDICTION_MANIFEST, manifest)
    sha = C.sha256_file(C.PREDICTION_MANIFEST)
    C.write_json(C.PREDICTION_LOCK, {"sha256": sha, "path": C.track_rel(C.PREDICTION_MANIFEST), "created_utc": C.utcnow()})
    C.log(f"[freeze] frozen {len(entries)} prediction files, manifest sha={sha[:12]}")
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
    if set(existing.get("predictions", {})) != set(rebuilt["predictions"]):
        raise PredictionError("existing prediction registry is incomplete")
    for key, entry in existing["predictions"].items():
        path = C.REPO / entry["file"]
        if not path.exists() or C.sha256_file(path) != entry["sha256"]:
            raise PredictionError(f"{key}: prediction file no longer matches the frozen manifest")
        if entry["sha256"] != rebuilt["predictions"][key]["sha256"]:
            raise PredictionError(f"{key}: prediction content differs from the current run; refusing to overwrite a freeze")
    existing["manifest_sha256"] = actual
    return existing


__all__ = ["PredictionError", "predict_nn", "freeze_predictions", "prediction_paths"]
