"""Deterministic replay of the CPU tail branch (P_U / P_B) from saved states.

Rebuilds each 25->64->32->1 head from ``cpu_<arm>_head_soup_state.pt`` alone
(plus the recorded fit-median bias value), recomputes ``q`` on the frozen 8000
fit / 2000 dev topology25 matrices, re-derives ``b_P`` and the calibrated
``P = h_raw + q + b_P`` endpoint, and checks it against the saved prediction
arrays and the analysis JSON.  Official-valid/test are never loaded.

Run from the repo root:
    PYTHONPATH=. uv run python tracks/ksvd/results/zinc_overnight_interface_and_tail_seed0_v1/replay_cpu.py
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_overnight_interface_and_tail_seed0_v1_cpu as R

RESULTS_DIR = R.RESULTS_DIR
TOL = 1e-5


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()


def main() -> int:
    torch.set_num_threads(4)
    t25 = np.load(R.FROZEN_CYCLE / "T25_all.npz", allow_pickle=True)
    T = np.asarray(t25["T"], np.float32)
    o = np.load(R.FROZEN_O / "O_seed0_predictions.npz", allow_pickle=True)
    fit_idx = np.asarray(o["fit_idx"], np.int64)
    dev_idx = np.asarray(o["dev_idx"], np.int64)
    y_fit, h_fit = np.asarray(o["fit_y"], np.float64), np.asarray(o["fit_raw"], np.float64)
    y_dev, h_dev = np.asarray(o["dev_y"], np.float64), np.asarray(o["dev_raw"], np.float64)
    T_fit, T_dev = T[fit_idx], T[dev_idx]

    with np.load(RESULTS_DIR / "cpu_analysis_8k.json" if False else RESULTS_DIR / "cpu_P_U_predictions.npz") as z:
        keys = list(z.keys())
    analysis = json.loads((RESULTS_DIR / "cpu_analysis_8k.json").read_text())
    out: dict[str, Any] = {
        "protocol_version": R.PROTOCOL_VERSION,
        "T25_sha256": sha_arr(T),
        "fit_idx_sha256": hashlib.sha256(fit_idx.tobytes()).hexdigest(),
        "dev_idx_sha256": hashlib.sha256(dev_idx.tobytes()).hexdigest(),
        "prediction_npz_keys": keys,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "arms": {},
    }
    ok = True
    for name in ("P_U", "P_B"):
        meta = json.loads((RESULTS_DIR / f"cpu_{name}.json").read_text())
        with np.load(RESULTS_DIR / f"cpu_{name}_predictions.npz") as z:
            saved_q_fit = np.asarray(z["q_fit"], np.float64)
            saved_q_dev = np.asarray(z["q_dev"], np.float64)
        head = R.build_head(int(meta["seed"]), float(meta["bias_value_median_fit_c"]))
        head.load_state_dict(torch.load(RESULTS_DIR / f"cpu_{name}_head_soup_state.pt", map_location="cpu", weights_only=False))
        with torch.no_grad():
            q_fit = R.head_forward(head, torch.as_tensor(T_fit, dtype=torch.float32)).numpy().astype(np.float64)
            q_dev = R.head_forward(head, torch.as_tensor(T_dev, dtype=torch.float32)).numpy().astype(np.float64)
        p_raw_fit = h_fit + q_fit
        b = float(np.median(y_fit - p_raw_fit))
        p_cal_dev = h_dev + q_dev + b
        mae_dev = float(np.mean(np.abs(p_cal_dev - y_dev)))
        ref = analysis["arms"][name]
        entry = {
            "q_fit_max_abs_diff": float(np.max(np.abs(q_fit - saved_q_fit))),
            "q_dev_max_abs_diff": float(np.max(np.abs(q_dev - saved_q_dev))),
            "b_P": b,
            "b_P_saved": ref["b_P"],
            "b_P_abs_diff": float(abs(b - ref["b_P"])),
            "dev_cal_mae": mae_dev,
            "dev_cal_mae_saved": ref["P_cal_dev"]["mae"],
            "dev_cal_mae_abs_diff": float(abs(mae_dev - ref["P_cal_dev"]["mae"])),
            "soup_hash_saved": meta["soup_hash"],
        }
        entry["pass"] = bool(
            entry["q_fit_max_abs_diff"] <= TOL
            and entry["q_dev_max_abs_diff"] <= TOL
            and entry["b_P_abs_diff"] <= TOL
            and entry["dev_cal_mae_abs_diff"] <= TOL
        )
        ok = ok and entry["pass"]
        out["arms"][name] = entry
        print(f"[replay-cpu] {name}: q_dev_diff={entry['q_dev_max_abs_diff']:.2e} "
              f"b_diff={entry['b_P_abs_diff']:.2e} mae_diff={entry['dev_cal_mae_abs_diff']:.2e} pass={entry['pass']}")
    out["pass"] = bool(ok)
    (RESULTS_DIR / "cpu_replay_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[replay-cpu] pass={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
