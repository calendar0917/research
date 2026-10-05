"""zinc_cycle_prototype_transfer_cpu_v1 — source identity verification.

Reuses the frozen source round's artifacts read-only and verifies:
  * COMP h_raw + q_raw + b_y reproduces published COMP y
  * COMP h_raw + b_g reproduces published g
  * y = g + c, g = ell + s
  * Q train/valid predictions reproduce from the frozen raw_soup state on the
    freshly built current T25

No training, no optimizer, no backward. CPU FP32 forward only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
SRC = REPO / "tracks/ksvd/results/zinc_component_supervision_fulltrain_confirmation_seed0_v1"
RUNNER = REPO / "tracks/ksvd/experiments/luyin16/zinc_component_supervision_fulltrain_confirmation_seed0_v1.py"
OUT = Path(__file__).resolve().parent


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return bool(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def main() -> int:
    torch.set_num_threads(8)
    runner = load_module(RUNNER, "proto_runner")

    manifest = json.loads((SRC / "frozen_eval_manifest.json").read_text())
    source_sha = {
        name: sha256_file(SRC / fname)
        for name, fname in [
            ("targets", "full_train_targets.npz"),
            ("prep", "full_train_prep.npz"),
            ("COMP_soup", "COMP_raw_soup_state.pt"),
            ("Q_soup", "Q_raw_soup_state.pt"),
            ("Q_train_predictions", "Q_train_predictions.npz"),
            ("COMP_raw_predictions", "COMP_raw_predictions.npz"),
            ("valid_frozen_predictions", "valid_frozen_predictions.npz"),
            ("calibration", "calibration.json"),
        ]
    }
    manifest_sha = {
        name: entry["sha256"]
        for name, entry in manifest["artifacts"].items()
        if name in {"targets", "prep", "Q_soup_state"}
    }
    manifest_sha["COMP_soup"] = manifest["models"]["COMP"]["raw_soup_state"]["sha256"]

    targets = np.load(SRC / "full_train_targets.npz", allow_pickle=False)
    c_train = targets["c"].astype(np.float64)
    k_train = targets["k"].astype(np.int64)
    y_train = targets["y"].astype(np.float64)
    g_train = targets["g"].astype(np.float64)
    ell_train = targets["ell"].astype(np.float64)
    s_train = targets["s"].astype(np.float64)

    cal = json.loads((SRC / "calibration.json").read_text())
    b_g_comp = float(cal["per_arm"]["COMP"]["b_g"])
    b_y_comp = float(cal["per_arm"]["COMP"]["b_y"])
    b_g_sum = float(cal["per_arm"]["SUM"]["b_g"])
    b_y_sum = float(cal["per_arm"]["SUM"]["b_y"])

    comp_train = np.load(SRC / "COMP_raw_predictions.npz", allow_pickle=False)
    sum_train = np.load(SRC / "SUM_raw_predictions.npz", allow_pickle=False)
    q_train_cache = np.load(SRC / "Q_train_predictions.npz", allow_pickle=False)
    h_comp_train = comp_train["raw_soup_train"].astype(np.float64)
    h_sum_train = sum_train["raw_soup_train"].astype(np.float64)
    q_train_cache_raw = q_train_cache["q_raw"].astype(np.float64)

    vp = np.load(SRC / "valid_frozen_predictions.npz", allow_pickle=False)

    checks: dict[str, Any] = {}

    # --- source label identities ---
    checks["y_equals_g_plus_c_max_abs"] = float(np.max(np.abs(y_train - (g_train + c_train))))
    checks["g_equals_ell_plus_s_max_abs"] = float(np.max(np.abs(g_train - (ell_train + s_train))))

    # --- train published scores reproduced ---
    y_cal_comp = h_comp_train + q_train_cache_raw + b_y_comp
    g_cal_comp = h_comp_train + b_g_comp
    checks["train_COMP_y_raw_mae"] = float(np.mean(np.abs((h_comp_train + q_train_cache_raw) - y_train)))
    checks["train_COMP_y_cal_mae"] = float(np.mean(np.abs(y_cal_comp - y_train)))
    checks["train_COMP_g_cal_mae"] = float(np.mean(np.abs(g_cal_comp - g_train)))
    checks["train_COMP_y_cal_mae_expected"] = manifest["calibration"]["per_arm"]["COMP"]["y_train_mae_cal"]
    checks["train_COMP_g_cal_mae_expected"] = manifest["calibration"]["per_arm"]["COMP"]["h_train_mae_g_cal"]
    checks["train_replay_ok"] = bool(
        abs(checks["train_COMP_y_cal_mae"] - checks["train_COMP_y_cal_mae_expected"]) <= 1e-9
        and abs(checks["train_COMP_g_cal_mae"] - checks["train_COMP_g_cal_mae_expected"]) <= 1e-9
    )

    # --- valid published scores reproduced ---
    y_v = vp["y"].astype(np.float64)
    g_v = vp["g"].astype(np.float64)
    c_v = vp["c"].astype(np.float64)
    k_v = vp["k"].astype(np.int64)
    h_comp_v = vp["COMP_h_raw"].astype(np.float64)
    q_raw_v = vp["q_raw"].astype(np.float64)
    checks["valid_COMP_y_cal_mae"] = float(np.mean(np.abs((h_comp_v + q_raw_v + b_y_comp) - y_v)))
    checks["valid_COMP_g_cal_mae"] = float(np.mean(np.abs((h_comp_v + b_g_comp) - g_v)))
    checks["valid_SUM_y_cal_mae"] = float(np.mean(np.abs((vp["SUM_h_raw"].astype(np.float64) + q_raw_v + b_y_sum) - y_v)))
    checks["valid_COMP_y_cal_reconstruct_max_abs"] = float(
        np.max(np.abs((h_comp_v + q_raw_v + b_y_comp) - vp["COMP_y_cal"].astype(np.float64)))
    )
    checks["valid_COMP_y_cal_mae_expected"] = 0.11740618350630393
    checks["valid_COMP_g_cal_mae_expected"] = 0.08930046045603672
    checks["valid_replay_ok"] = bool(
        abs(checks["valid_COMP_y_cal_mae"] - checks["valid_COMP_y_cal_mae_expected"]) <= 1e-9
        and abs(checks["valid_COMP_g_cal_mae"] - checks["valid_COMP_g_cal_mae_expected"]) <= 1e-9
    )
    checks["valid_identity_y_eq_g_plus_c"] = float(np.max(np.abs(y_v - (g_v + c_v))))

    # --- Q frozen replay on current T25 (train + valid) ---
    prep_meta = np.load(SRC / "full_train_prep.npz", allow_pickle=False)
    prep_meta_dict = {key: prep_meta[key] for key in prep_meta.files}
    train_data = runner.build_fulltrain_data(prep_meta_dict)
    T_train = runner.topology_matrix(train_data)

    bias_value = float(np.median(c_train))
    q_head = runner.build_q_head(0, bias_value)
    q_head.load_state_dict(torch.load(SRC / "Q_raw_soup_state.pt", map_location="cpu", weights_only=True))
    q_head.eval()
    with torch.no_grad():
        q_train_replay = runner.q_forward(q_head, torch.as_tensor(T_train, dtype=torch.float32)).double().numpy()
    checks["q_train_replay_max_abs_vs_cache"] = float(np.max(np.abs(q_train_replay - q_train_cache_raw)))
    checks["q_train_replay_ok"] = bool(checks["q_train_replay_max_abs_vs_cache"] <= 1e-5)

    valid_data, vmeta = runner.load_valid_data(SRC)
    T_valid = runner.topology_matrix(valid_data)
    with torch.no_grad():
        q_valid_replay = runner.q_forward(q_head, torch.as_tensor(T_valid, dtype=torch.float32)).double().numpy()
    checks["q_valid_replay_max_abs_vs_cache"] = float(np.max(np.abs(q_valid_replay - q_raw_v)))
    checks["q_valid_replay_ok"] = bool(checks["q_valid_replay_max_abs_vs_cache"] <= 1e-5)
    checks["valid_n_rows"] = int(T_valid.shape[0])
    checks["valid_meta"] = vmeta

    # --- hashes ---
    checks["source_file_sha256"] = source_sha
    checks["source_manifest_sha_match"] = {
        "targets": source_sha["targets"] == manifest_sha.get("targets"),
        "prep": source_sha["prep"] == manifest_sha.get("prep"),
        "Q_soup": source_sha["Q_soup"] == manifest_sha.get("Q_soup_state"),
        "COMP_soup": source_sha["COMP_soup"] == manifest_sha.get("COMP_soup"),
    }
    checks["execution_commit"] = manifest["execution_commit"]
    checks["protocol_version"] = manifest["protocol_version"]

    # --- T25 vector hash (frozen identity of this round's key source) ---
    checks["T25_train_sha256_f32"] = hashlib.sha256(np.ascontiguousarray(T_train, np.float32).tobytes()).hexdigest()
    checks["T25_valid_sha256_f32"] = hashlib.sha256(np.ascontiguousarray(T_valid, np.float32).tobytes()).hexdigest()
    checks["T25_train_shape"] = list(T_train.shape)
    checks["T25_valid_shape"] = list(T_valid.shape)
    checks["negative_zero_T25_train"] = int(np.sum(np.signbit(T_train) & (T_train == 0)))
    checks["negative_zero_T25_valid"] = int(np.sum(np.signbit(T_valid) & (T_valid == 0)))
    checks["nan_inf_T25_train"] = bool(np.isnan(T_train).any() or np.isinf(T_train).any())
    checks["nan_inf_T25_valid"] = bool(np.isnan(T_valid).any() or np.isinf(T_valid).any())

    checks["bias"] = {
        "b_g_COMP": b_g_comp,
        "b_y_COMP": b_y_comp,
        "b_g_SUM": b_g_sum,
        "b_y_SUM": b_y_sum,
    }

    checks["all_source_identity_ok"] = bool(
        checks["train_replay_ok"]
        and checks["valid_replay_ok"]
        and checks["q_train_replay_ok"]
        and checks["q_valid_replay_ok"]
        and checks["y_equals_g_plus_c_max_abs"] <= 1e-9
        and checks["g_equals_ell_plus_s_max_abs"] <= 1e-9
        and all(checks["source_manifest_sha_match"].values())
    )

    write_json(OUT / "source_identity.json", checks)
    print(json.dumps(jsonable({k: v for k, v in checks.items() if k not in ("source_file_sha256", "valid_meta")}), indent=2))
    print("SOURCE_IDENTITY_OK:", checks["all_source_identity_ok"])
    return 0 if checks["all_source_identity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
