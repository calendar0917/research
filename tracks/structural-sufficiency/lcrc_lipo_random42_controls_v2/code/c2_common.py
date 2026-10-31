"""Shared frozen constants, reuse guards and integrity helpers for controls v2.

This round is a *read-only downstream consumer* of the frozen random42 v1
artifacts and of the historical ``v2_models.LCRCCapacityMatched`` class.  No
historical file is written, no Test label is read, and no Test prediction or
Test metric is opened.

The two new models are:

* ``LCRC-S2off-capacity-matched`` -- the historical capacity-matched S2-off
  implementation, unchanged, trained with the *executed* v1 LCRC-A recipe;
* ``PNA-small`` -- the upstream PNA-native stack with only
  ``gnn_intermediate_dim == output_node_dim`` reduced to the value selected by
  the CPU-only parameter-count rule.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

V1 = REPO / "tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1"
V1_CODE = V1 / "code"
V1_DATA = V1 / "data"
V1_ANALYSIS = V1 / "analysis"
V2 = REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2"
V2_CODE = V2 / "code"
PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"

for _path in (str(V1_CODE), str(PREP_CODE), str(V2_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

CONFIG_PATH = EXP_DIR / "configs/controls_v2.yaml"
DATA_DIR = EXP_DIR / "data"
CACHE_DIR = EXP_DIR / "cache"
ANALYSIS = EXP_DIR / "analysis"
CHECKPOINTS = EXP_DIR / "checkpoints"
PREDICTIONS = EXP_DIR / "predictions"
RUNS = ANALYSIS / "runs"

PROTOCOL_PATH = EXP_DIR / "PROTOCOL.md"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
PROTOCOL_SHA256 = ANALYSIS / "protocol_sha256.json"
REUSE_AUDIT = ANALYSIS / "reuse_audit.json"
WIDTH_SEARCH = ANALYSIS / "pna_width_search.json"
CPU_GATES = ANALYSIS / "cpu_gates.json"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"
VALIDATION_TABLE = ANALYSIS / "validation_table.json"
PAIRED_BOOTSTRAP = ANALYSIS / "paired_bootstrap.json"
INFERENCE_BENCHMARK = ANALYSIS / "inference_benchmark.json"
PREFLIGHT_REMOTE = ANALYSIS / "preflight_remote.json"

ROUND_ID = "lcrc-lipo-random42-controls-v2"
PROTOCOL_ID = ROUND_ID

# ---------------------------------------------------------------------------
# frozen data constants (identical to random42 v1)
# ---------------------------------------------------------------------------

CSV_PATH = REPO / "data/lipophilicity/Lipophilicity.csv"
CSV_SHA256 = "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
N_MOLECULES = 4200
SPLIT_SEED = 42
TRAIN_SEED = 0
SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}
SPLIT_SHA256 = {
    "train": "0df18c72753974d502f8bede1cd2aa5c0ebdc1df450b8871fffdb1f67d2648ee",
    "valid": "8f9c35171ac2125e678d2ad44d87b293719bb9806a528810ac271e9aa39f9b44",
    "test": "4bc449106a64833973b8e8d3d16097bf5d2906c788d7dfcdf02a4b3a30ffa49b",
}
SPLIT_FIRST_ROWS = {
    "train": [3942, 3415, 1426, 2365, 3462],
    "valid": [501, 1649, 3579, 2873, 106],
    "test": [967, 3557, 3492, 462, 2258],
}
V1_SPLIT_FILE = V1_DATA / "split_esa_random42_v1.json"
V1_SPLIT_SHA256 = "1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4"
V1_LABELS_FILE = V1_DATA / "labels_train_valid_v1.json"
V1_LABELS_SHA256 = "1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca"
V1_STRUCTURES_FILE = V1_DATA / "structures_esa_random42_v1.npz"
V1_STRUCTURES_SHA256 = "a098d27fa11c5d718d2f0a373636cfc9eeac9604ea5db8ae50a36d05fe035a2c"
V1_STRUCTURES_META = V1_DATA / "structures_esa_random42_v1_meta.json"
V1_STRUCTURES_META_SHA256 = "e3866a483f56d1cc8291122219cffd0949e167673e5bc1455f1cc697643e0a39"
V1_CONFIG = V1 / "configs/benchmark_v1.yaml"
V1_CONFIG_SHA256 = "0d8d58eae2351294f09da21056b4d193e94cefccc8b79b94ec54ca247a2d8402"
V1_METRICS_VALID = V1_ANALYSIS / "metrics_validation.json"
V1_METRICS_VALID_SHA256 = "c540f9ac89a391ea13c8235b7b14cb7faf10ae2a9d60616d2e4716659c576fdd"
V1_GPU_LEDGER = V1_ANALYSIS / "gpu_ledger.json"
V1_GPU_LEDGER_SHA256 = "9d4174124f9ac6ef6015d6e450bc974dc97d67e7a514f7106e78aa40b9c227cc"
V1_PROTOCOL_FREEZE = V1_ANALYSIS / "protocol_freeze.json"

TARGET_MEAN = 2.2081011904761905
TARGET_SCALE = 1.1906281212426577
TARGET_VECTOR_SHA256 = "85a4c7c559f4b7fd58ad78b1b05d5a4160e0f60c351b7fa842eb0354aa47abd8"
VALID_VECTOR_SHA256 = "09bc1a1e1756bbd09dd17a5fc77a1b3c33e7010aef55fc352a4bba1d9822636f"
LCRC_TRAIN_NORM_FINGERPRINT = "4d6e5bf456ff9d2217eb11a2c4a29d3f7cbbcb95665bf56e62b93c3ae93d0e26"

# ---------------------------------------------------------------------------
# frozen historical artifacts / source hashes
# ---------------------------------------------------------------------------

V1_LCRC_CKPT = V1_ANALYSIS / "runs/lcrc_a_r3_s0/checkpoint.pt"
V1_LCRC_CKPT_SHA256 = "eca1486102faed79ed4c233332644d8767dc76193adc16523fc115bb33a0e683"
V1_LCRC_VALID = V1_ANALYSIS / "runs/lcrc_a_r3_s0/valid_predictions.npz"
V1_LCRC_VALID_SHA256 = "7f9ad3919d57755d5adc08b99f1209b18ab6629c14eb182592a858183d88e894"
V1_LCRC_TRAIN_JSON = V1_ANALYSIS / "runs/lcrc_a_r3_s0/train.json"
V1_PNA_CKPT = V1_ANALYSIS / "runs/pna_native_s0/checkpoint.pt"
V1_PNA_CKPT_SHA256 = "3eebd089a6db49f504bc7de769ce73cbafe55d575028523560577b74e139524b"
V1_PNA_VALID = V1_ANALYSIS / "runs/pna_native_s0/valid_predictions.npz"
V1_PNA_VALID_SHA256 = "735df74915d43803b63323e09d85089c6c2d284e04acffcf242eb587ae1ac905"
V1_PNA_TRAIN_JSON = V1_ANALYSIS / "runs/pna_native_s0/train.json"
V1_PNA_DEGREES = V1_ANALYSIS / "runs/pna_native_s0/saved_PNA_degrees/degrees.pt"

V2_MODELS_SHA256 = "00b3d3c3657cb7320affd79d8829110ad8d34ca8327191781882f225853c2988"
LCRC_MODEL_SHA256 = "d5af76cb9b5998e6bfa068afd28c483c93dde579cfb85344ed928582b05c6d37"
V1_DATA_CODE_SHA256 = "cd08aec7134cf0c31a5e323c269e319164de59432416d8fcb91a399c671f2c3f"
V1_LABELS_CODE_SHA256 = "78bf0adf3a24a182a490c2fd1bff5f6eb13aaa393d5e2421e0899d6262bb1214"
V1_METRICS_CODE_SHA256 = "8535816d8bd8bbfd46f7a8571f0cb318e6e349709be42150c62a7eea207d7228"
V1_LCRC_TRAIN_SHA256 = "0e9949264f2133137a770099f59605552b6c014751bacbf11be88cc3f3a7f8f5"
V1_NATIVE_TRAIN_SHA256 = "98f7583d704b06594f63978e5cea2367c0a79014d68bdbf2f0f7b983804b6d7f"
V1_UPSTREAM_SHA256 = "b8a7e70a2fde8adc813f0a49482788c3b28daf4c2543bac8e96800e14f3e7e3a"

# ---------------------------------------------------------------------------
# frozen model / training constants
# ---------------------------------------------------------------------------

LCRC_BASELINE_PARAMS = 247_106
S2OFF_EFFECTIVE_REFERENCE = 247_049
S2OFF_NOMINAL_REFERENCE = 285_449
S2OFF_DEAD_COLUMNS = 38_400
S2OFF_REPLACEMENT_PARAMS = 70_535

LCRC_RECIPE = {
    "model": "LCRC-S2off-capacity-matched",
    "seed": TRAIN_SEED,
    "batch_size": 128,
    "epochs": 360,
    "optimizer": "AdamW",
    "peak_lr": 0.0003,
    "min_lr": 0.00001,
    "weight_decay": 0.001,
    "scheduler": "per_step_cosine",
    "dropout_configured": 0.10,
    # random42 v1's executed LCRC loop never wired the configured dropout into
    # the forward pass.  The control mirrors the executed historical recipe
    # exactly; retrofitting dropout to only one arm would confound the contrast.
    "dropout_applied": 0.0,
    "gradient_clip": 5.0,
    "loss": "standardized_mse",
    "checkpoint": "average_last_5_epoch_end_state_dicts",
    "early_stopping": False,
    "ssl": False,
    "steps_per_epoch": 27,
    "optimizer_steps": 9_720,
}

PNA_BASE_CONFIG = {
    "model": "PNA-small",
    "seed": TRAIN_SEED,
    "conv_type": "PNA",
    "num_layers": 4,
    "batch_size": 128,
    "optimizer": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.0000000001,
    "gradient_clip": 0.5,
    "loss": "standardized_mse",
    "max_epochs": 360,
    "early_stopping_patience": 30,
    "lr_scheduler": "ReduceLROnPlateau",
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 15,
    "checkpoint": "best_validation_loss",
    "precision": "bf16",
    "aggregation": ["mean", "min", "max", "std"],
    "scalers": ["identity", "amplification", "attenuation"],
    "towers": 5,
}

PNA_WIDTHS = tuple(range(20, 101, 5))
PNA_SIZE_TOLERANCE = 0.15
PNA_TARGET_PARAMS = LCRC_BASELINE_PARAMS

GPU_HARD_CAP_HOURS = 0.5
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
SHUFFLE_OFFSET = 917_000
SOUP_LAST = 5

# Files that must never be opened by this round.
FORBIDDEN_TEST_ARTIFACTS = (
    "metrics_test.json",
    "test_read.lock",
    "secondary_analysis.json",
    "prediction_manifest.json",
    "pre_test_manifest.json",
    "pre_test_manifest.sha256",
    "*_test420.npz",
)

# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return sha256_bytes(stable_json(obj).encode("utf-8"))


def rows_sha256_int64(rows: Iterable[int]) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(list(rows), dtype="<i8")).tobytes())


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_sha256(path: Path | str, expected: str, label: str) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise PermissionError(f"{label} sha256 {actual} != frozen {expected}")
    return actual


def track_rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return True


def env_info() -> dict[str, Any]:
    import torch

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "git_commit": git_commit(),
    }


def log(message: str) -> None:
    print(message, flush=True)


def load_config() -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("controls_v2.yaml must be a mapping")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("config protocol_id mismatch")
    return payload


def config_hash(config: Mapping[str, Any]) -> str:
    return sha256_json(dict(config))


def code_file_hashes() -> dict[str, str]:
    paths = sorted(list((EXP_DIR / "code").glob("*.py")) + list((EXP_DIR / "tests").glob("*.py")))
    paths += [CONFIG_PATH, PROTOCOL_PATH]
    return {
        str(p.relative_to(EXP_DIR)): sha256_file(p)
        for p in paths
        if p.exists()
    }


# ---------------------------------------------------------------------------
# frozen-artifact / history verification
# ---------------------------------------------------------------------------


def verify_split_and_labels() -> dict[str, Any]:
    """Verify the byte-level reuse of the random42 v1 split and labels."""
    import r42_data as D
    import r42_labels as L

    verify_sha256(V1_SPLIT_FILE, V1_SPLIT_SHA256, "v1 split file")
    verify_sha256(V1_LABELS_FILE, V1_LABELS_SHA256, "v1 labels file")
    split = D.load_split()
    # Independent re-derivation of the exact random_split call.
    regenerated = D.compute_split()
    for name in ("train", "valid", "test"):
        if [int(x) for x in split["rows"][name]] != [int(x) for x in regenerated[name]]:
            raise PermissionError(f"{name} split rows differ from the re-derived random_split")
        if rows_sha256_int64(split["rows"][name]) != SPLIT_SHA256[name]:
            raise PermissionError(f"{name} split hash mismatch")
        if [int(x) for x in split["rows"][name][:5]] != SPLIT_FIRST_ROWS[name]:
            raise PermissionError(f"{name} split first rows drifted")
    labels = L.load_train_valid_labels()
    if labels.audit["scope"] != "train_and_valid_only" or labels.audit["test_labels_included"] is not False:
        raise PermissionError("label artifact is not certified Train/Valid only")
    if int(labels.y_train.size) != SPLIT_SIZES["train"] or int(labels.y_valid.size) != SPLIT_SIZES["valid"]:
        raise PermissionError("label vector size mismatch")
    train_vec = sha256_bytes(np.ascontiguousarray(labels.y_train, dtype="<f8").tobytes())
    valid_vec = sha256_bytes(np.ascontiguousarray(labels.y_valid, dtype="<f8").tobytes())
    if train_vec != TARGET_VECTOR_SHA256 or valid_vec != VALID_VECTOR_SHA256:
        raise PermissionError("label vector hash mismatch")
    # Target scaler is frozen and reconstructed only from Train3360.
    mean = float(np.mean(labels.y_train))
    scale = float(np.std(labels.y_train, ddof=0))
    if mean != TARGET_MEAN or scale != TARGET_SCALE:
        raise PermissionError(f"target scaler drifted: mean={mean!r} scale={scale!r}")
    return {
        "split_file": track_rel(V1_SPLIT_FILE),
        "split_file_sha256": V1_SPLIT_SHA256,
        "split_index_sha256": dict(SPLIT_SHA256),
        "split_sizes": dict(SPLIT_SIZES),
        "split_seed": SPLIT_SEED,
        "label_file": track_rel(V1_LABELS_FILE),
        "label_file_sha256": V1_LABELS_SHA256,
        "label_scope": labels.audit["scope"],
        "test_labels_included": False,
        "train_vector_sha256": train_vec,
        "valid_vector_sha256": valid_vec,
        "target_scaler": {
            "mean": mean,
            "scale": scale,
            "n": int(labels.y_train.size),
            "source": "train3360_labels_only",
            "scale_definition": "population_std_float64",
        },
        "test_labels_read": False,
    }


def verify_structures() -> dict[str, Any]:
    import r42_data as D

    verify_sha256(V1_STRUCTURES_FILE, V1_STRUCTURES_SHA256, "v1 structures")
    verify_sha256(V1_STRUCTURES_META, V1_STRUCTURES_META_SHA256, "v1 structures meta")
    summary = D.structure_summary()
    if summary["n_molecules"] != N_MOLECULES:
        raise PermissionError("structure artifact molecule count mismatch")
    if summary["atom_feat_dim"] != 79 or summary["edge_feat_dim"] != 13:
        raise PermissionError("upstream Chemprop feature dimensions drifted")
    if summary["max_node_global"] != 216 or summary["max_edge_global"] != 438:
        raise PermissionError("frozen global maxima drifted")
    return {
        "path": summary["path"],
        "sha256": summary["sha256"],
        "meta_sha256": V1_STRUCTURES_META_SHA256,
        "n_molecules": summary["n_molecules"],
        "n_atoms": summary["n_atoms"],
        "n_directed_edges": summary["n_directed_edges"],
        "atom_feat_dim": summary["atom_feat_dim"],
        "edge_feat_dim": summary["edge_feat_dim"],
        "max_node_global": summary["max_node_global"],
        "max_edge_global": summary["max_edge_global"],
        "labels_included": False,
    }


def verify_csv_if_present() -> dict[str, Any]:
    if not CSV_PATH.exists():
        return {"present": False, "sha256": None, "verified": False}
    actual = sha256_file(CSV_PATH)
    if actual != CSV_SHA256:
        raise PermissionError("Lipophilicity.csv hash mismatch")
    return {"present": True, "sha256": actual, "verified": True}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def verify_history() -> dict[str, Any]:
    """Hash-verify the two historical references and reproduce their metrics.

    Only Train/Validation artifacts are opened.  Test predictions, Test labels,
    Test metrics, the pre-test manifest and the post-lock analysis are never
    touched.
    """
    import r42_common as V1C
    import r42_data as D
    import r42_labels as L
    import r42_metrics as MET

    split = verify_split_and_labels()
    labels_mean = TARGET_MEAN
    labels_scale = TARGET_SCALE
    labels = L.load_train_valid_labels()
    y_valid = labels.y_valid
    rows_valid = np.asarray(D.load_split()["rows"]["valid"], dtype=np.int64)

    entries = {}
    for name, ckpt, pred_path, pred_sha, train_json in (
        ("LCRC-A-R3-scratch", V1_LCRC_CKPT, V1_LCRC_VALID, V1_LCRC_VALID_SHA256, V1_LCRC_TRAIN_JSON),
        ("PNA-native", V1_PNA_CKPT, V1_PNA_VALID, V1_PNA_VALID_SHA256, V1_PNA_TRAIN_JSON),
    ):
        verify_sha256(ckpt, V1_LCRC_CKPT_SHA256 if name.startswith("LCRC") else V1_PNA_CKPT_SHA256, f"historical {name} checkpoint")
        verify_sha256(pred_path, pred_sha, f"historical {name} validation predictions")
        payload = _load_npz(pred_path)
        if "rows" not in payload or "pred" not in payload:
            raise PermissionError(f"{name} validation prediction file is malformed")
        rows = np.asarray(payload["rows"], dtype=np.int64)
        pred = np.asarray(payload["pred"], dtype=np.float64)
        if rows.shape != (SPLIT_SIZES["valid"],) or pred.shape != (SPLIT_SIZES["valid"],):
            raise PermissionError(f"{name} validation prediction length mismatch")
        if rows_sha256_int64(rows) != SPLIT_SHA256["valid"]:
            raise PermissionError(f"{name} validation prediction rows are not the frozen Validation420")
        if not np.array_equal(rows, rows_valid):
            raise PermissionError(f"{name} validation prediction row order differs")
        metrics = MET.all_metrics(pred, y_valid)
        entries[name] = {
            "checkpoint": track_rel(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "valid_predictions": track_rel(pred_path),
            "valid_predictions_sha256": pred_sha,
            "valid_rows_sha256": rows_sha256_int64(rows),
            "n_valid": int(rows.size),
            "valid_metrics_recomputed": metrics,
            "train_manifest": track_rel(train_json),
            "train_manifest_sha256": sha256_file(train_json),
            "test_artifacts_read": False,
        }
    # Cross-check against the frozen v1 Validation report.
    verify_sha256(V1_METRICS_VALID, V1_METRICS_VALID_SHA256, "v1 metrics_validation.json")
    frozen = read_json(V1_METRICS_VALID)
    if frozen.get("scope") != "train_and_validation_only" or frozen.get("test_labels_read") is not False:
        raise PermissionError("v1 metrics_validation.json is not Train/Valid only")
    by_model = {row["model"]: row for row in frozen["results"]}
    for name, entry in entries.items():
        target = by_model[name]["valid_metrics"]
        for key in ("rmse", "mae", "r2"):
            if abs(float(entry["valid_metrics_recomputed"][key]) - float(target[key])) > 1e-12:
                raise PermissionError(f"{name} recomputed {key} does not match the frozen v1 report")
        entry["frozen_v1_valid_metrics"] = target
        entry["frozen_v1_parameters_total"] = int(by_model[name]["parameters_total"])
    # Config/recipe identity.
    verify_sha256(V1_CONFIG, V1_CONFIG_SHA256, "v1 benchmark config")
    if sha256_file(V1_LCRC_TRAIN_JSON) == "":
        raise PermissionError("missing LCRC train manifest")
    lcrc_train = read_json(V1_LCRC_TRAIN_JSON)
    pna_train = read_json(V1_PNA_TRAIN_JSON)
    if lcrc_train.get("config") != V1C.LCRC_CONFIG or lcrc_train.get("config_hash") != V1C.config_hash(V1C.LCRC_CONFIG):
        raise PermissionError("historical LCRC config differs from the frozen v1 recipe")
    if pna_train.get("config") != V1C.PNA_CONFIG or pna_train.get("config_hash") != V1C.config_hash(V1C.PNA_CONFIG):
        raise PermissionError("historical PNA config differs from the frozen v1 recipe")
    if int(lcrc_train["model_audit"]["parameters_total"]) != LCRC_BASELINE_PARAMS:
        raise PermissionError("historical LCRC parameter count drifted")
    if int(pna_train["model_audit"]["parameters_total"]) != 7_767_349:
        raise PermissionError("historical PNA parameter count drifted")
    if lcrc_train["data"]["train_norm_fingerprint"] != LCRC_TRAIN_NORM_FINGERPRINT:
        raise PermissionError("historical LCRC train-normalisation fingerprint drifted")
    if int(lcrc_train["training"]["optimizer_steps"]) != 9_720:
        raise PermissionError("historical LCRC optimizer step count drifted")
    if int(pna_train["training"]["best_epoch"]) != 84:
        raise PermissionError("historical PNA best epoch drifted")
    return {
        "status": "PASS",
        "scope": "train_and_validation_only",
        "test_labels_read": False,
        "test_predictions_read": False,
        "test_metrics_read": False,
        "split": split,
        "structures": verify_structures(),
        "csv": verify_csv_if_present(),
        "models": entries,
        "v1_metrics_validation": {
            "path": track_rel(V1_METRICS_VALID),
            "sha256": V1_METRICS_VALID_SHA256,
            "scope": frozen["scope"],
        },
        "v1_gpu_ledger": {
            "path": track_rel(V1_GPU_LEDGER),
            "sha256": V1_GPU_LEDGER_SHA256,
        },
        "target_scaler": {
            "mean": labels_mean,
            "scale": labels_scale,
            "train_vector_sha256": TARGET_VECTOR_SHA256,
        },
    }


def verify_source_hashes() -> dict[str, Any]:
    expected = {
        "v2_models.py": (V2_CODE / "v2_models.py", V2_MODELS_SHA256),
        "lcrc_model.py": (PREP_CODE / "lcrc_model.py", LCRC_MODEL_SHA256),
        "r42_data.py": (V1_CODE / "r42_data.py", V1_DATA_CODE_SHA256),
        "r42_labels.py": (V1_CODE / "r42_labels.py", V1_LABELS_CODE_SHA256),
        "r42_metrics.py": (V1_CODE / "r42_metrics.py", V1_METRICS_CODE_SHA256),
        "r42_lcrc_train.py": (V1_CODE / "r42_lcrc_train.py", V1_LCRC_TRAIN_SHA256),
        "r42_native_train.py": (V1_CODE / "r42_native_train.py", V1_NATIVE_TRAIN_SHA256),
        "r42_upstream.py": (V1_CODE / "r42_upstream.py", V1_UPSTREAM_SHA256),
    }
    out = {}
    for name, (path, digest) in expected.items():
        verify_sha256(path, digest, f"historical source {name}")
        out[name] = {"path": track_rel(path), "sha256": digest}
    return out


def assert_no_forbidden_artifacts_read() -> dict[str, Any]:
    """Static guard: this round's code must not reference Test artifacts."""
    forbidden = ("metrics_test", "test420", "test_read.lock", "secondary_analysis", "prediction_manifest", "pre_test_manifest")
    hits: list[dict[str, str]] = []
    for path in sorted((EXP_DIR / "code").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text and path.name != "c2_common.py":
                hits.append({"file": str(path.relative_to(EXP_DIR)), "token": token})
    # c2_common only lists the names as forbidden; it never opens them.
    if hits:
        raise PermissionError(f"forbidden Test artifact references found: {hits}")
    produced = sorted(p.name for p in PREDICTIONS.glob("*")) if PREDICTIONS.exists() else []
    bad = [name for name in produced if "test" in name.lower()]
    if bad:
        raise PermissionError(f"Test-named prediction artifacts exist: {bad}")
    return {"status": "PASS", "forbidden_references": [], "prediction_files": produced, "test_artifacts_read": False}


def verify_frozen(require_clean: bool = False) -> dict[str, Any]:
    if not PROTOCOL_FREEZE.exists() or not PROTOCOL_SHA256.exists():
        raise PermissionError("protocol freeze is missing; run freeze before training")
    lock = read_json(PROTOCOL_SHA256)
    if sha256_file(PROTOCOL_FREEZE) != lock.get("protocol_freeze_sha256"):
        raise PermissionError("protocol_freeze.json changed after its sha256 lock")
    frozen = read_json(PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("protocol freeze has the wrong protocol id")
    cfg = load_config()
    if frozen.get("config_sha256") != sha256_file(CONFIG_PATH) or frozen.get("config_canonical_sha256") != config_hash(cfg):
        raise PermissionError("config changed after the protocol freeze")
    current = code_file_hashes()
    if current != frozen.get("code_hashes"):
        changed = sorted(k for k in set(current) | set(frozen.get("code_hashes", {})) if current.get(k) != frozen["code_hashes"].get(k))
        raise PermissionError(f"code changed after the protocol freeze: {changed}")
    for key in ("width_search_sha256", "cpu_gates_sha256"):
        path = WIDTH_SEARCH if key.startswith("width") else CPU_GATES
        if sha256_file(path) != frozen.get(key):
            raise PermissionError(f"{path.name} changed after the protocol freeze")
    if require_clean and git_dirty():
        raise PermissionError("formal GPU run requires a clean committed worktree")
    return frozen


def test_artifact_guard(path: Path | str) -> None:
    """Refuse to open any historical Test artifact."""
    text = str(path).lower()
    if any(token in text for token in ("metrics_test", "test420", "test_read", "secondary_analysis", "prediction_manifest", "pre_test_manifest")):
        raise PermissionError(f"refusing to open a Test artifact: {path}")


__all__ = [name for name in dir() if not name.startswith("_")]
