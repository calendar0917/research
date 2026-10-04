"""Old R_SJ / R_DJ J->M inference flip: per-graph delta and error contributions.

CPU-only, no training, train-only (old 8000 fit / 2000 internal dev).  Loads the
released old checkpoints `R_SJ_raw_soup_state.pt` / `R_DJ_raw_soup_state.pt`,
runs the old internal dev twice per arm (native joint block, then marginal block
with the trained weights held fixed), and keeps each arm's original fit bias
(`calibration.b`).  No recalibration, no main-fold interaction.

Answers section-10 of the round: a small overall MAE change is not the same as
"predictions did not move".

Run:
    uv run python tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/jm_flip_replay.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_overnight_interface_and_tail_seed0_v1 as ov

RESULTS_DIR = Path(__file__).resolve().parent
OLD_DIR = ov.RESULTS_DIR
ARMS = ("R_SJ", "R_DJ")


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    return {
        "mean_abs": float(np.mean(np.abs(values))),
        "p50_abs": float(np.percentile(np.abs(values), 50)),
        "p95_abs": float(np.percentile(np.abs(values), 95)),
        "max_abs": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
        "changed_fraction": float(np.mean(np.abs(values) > 1e-6)),
    }


def main() -> int:
    torch.set_num_threads(8)
    started = time.perf_counter()
    _blob, _decomp, _train_data, _fit_data, dev_data, _fit_idx, dev_idx, y = ov.load_data()
    structural = ov.load_dictionary_objects()
    stats = ov.load_interface_stats()
    y_dev = np.asarray(y[dev_idx], np.float64)
    target = torch.as_tensor(y_dev, dtype=torch.float32)
    device = torch.device("cpu")
    out: dict[str, object] = {
        "protocol_version": "zinc-task-dictionary-and-cycle-witness-seed0-v1-jm-flip",
        "source": str(OLD_DIR),
        "n_dev": int(len(dev_idx)),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "arms": {},
    }
    for arm in ARMS:
        state_path = OLD_DIR / f"{arm}_raw_soup_state.pt"
        meta_path = OLD_DIR / f"{arm}.json"
        if not state_path.exists() or not meta_path.exists():
            out["arms"][arm] = {"status": "state_missing"}
            continue
        meta = json.loads(meta_path.read_text())
        code_mode, block_mode = ov.ARM_SPEC[arm]
        if block_mode != "joint":
            out["arms"][arm] = {"status": f"unexpected block_mode={block_mode}"}
            continue
        b = float(meta["calibration"]["b"])
        model = ov.build_interface_model(structural, stats, code_mode=code_mode, block_mode=block_mode, device=device)
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
        model.eval()
        native, _ = ov._predict(model, dev_data, target, device)
        model.block_mode = "marginal"
        flipped, _ = ov._predict(model, dev_data, target, device)
        model.block_mode = block_mode
        native = np.asarray(native, np.float64)
        flipped = np.asarray(flipped, np.float64)
        err_native = np.abs(y_dev - (native + b))
        err_flipped = np.abs(y_dev - (flipped + b))
        delta = flipped - native
        improve = np.maximum(err_native - err_flipped, 0.0)
        worsen = np.maximum(err_flipped - err_native, 0.0)
        k_all = np.asarray(np.load(ov.zftd.RESULTS_DIR / "target_decomposition.npz")["k"], np.int64)
        k_dev = k_all[dev_idx]
        groups = {
            "k=0": k_dev == 0,
            "k=-1": k_dev == -1,
            "k<=-2": k_dev <= -2,
        }
        out["arms"][arm] = {
            "code_mode": code_mode,
            "native_block": block_mode,
            "flipped_block": "marginal",
            "b_fit": b,
            "mae_native_cal": float(err_native.mean()),
            "mae_flipped_cal": float(err_flipped.mean()),
            "mae_change_cal": float(err_flipped.mean() - err_native.mean()),
            "delta_pred": _summary(delta),
            "error_contribution": {
                "improved_rows": int((err_flipped < err_native).sum()),
                "worsened_rows": int((err_flipped > err_native).sum()),
                "improve_contribution_per_row": float(improve.sum() / len(dev_idx)),
                "worsen_contribution_per_row": float(worsen.sum() / len(dev_idx)),
                "net_contribution_per_row": float((improve.sum() - worsen.sum()) / len(dev_idx)),
            },
            "groups": {
                name: {
                    "n": int(mask.sum()),
                    "mae_native_cal": float(err_native[mask].mean()) if mask.any() else None,
                    "mae_flipped_cal": float(err_flipped[mask].mean()) if mask.any() else None,
                    "delta_mean_abs": float(np.mean(np.abs(delta[mask]))) if mask.any() else None,
                }
                for name, mask in groups.items()
            },
        }
        print(
            f"[{arm}] MAE {err_native.mean():.6f} -> {err_flipped.mean():.6f} "
            f"(delta={err_flipped.mean() - err_native.mean():+.6f}); |delta_pred| mean={np.mean(np.abs(delta)):.6f} "
            f"p95={np.percentile(np.abs(delta), 95):.6f} max={np.max(np.abs(delta)):.6f}",
            flush=True,
        )
    out["seconds"] = float(time.perf_counter() - started)
    path = RESULTS_DIR / "jm_flip_replay.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[jm_flip] wrote {path} in {out['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
