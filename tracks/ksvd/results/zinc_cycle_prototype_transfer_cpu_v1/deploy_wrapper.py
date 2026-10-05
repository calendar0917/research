"""zinc_cycle_prototype_transfer_cpu_v1 — single-file frozen H wrapper (reloadable).

Loads the frozen deploy package (COMP body + frozen Q + train prototype table + key
codec + b_H) and exposes a standard forward returning (n,) y, plus an optional
diagnostic forward returning q and routing.

The wrapper never queries a query sample's true c/k and never reads valid labels.
official-test is never touched.
"""

from __future__ import annotations

import importlib.util
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

KEY_DIM = 25


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FrozenHWrapper:
    """COMP + frozen Q + train prototype lookup + single b_H."""

    def __init__(self, package_path: Path | None = None, src: Path | None = None):
        self.src = Path(src or SRC)
        package = torch.load(package_path or (OUT / "H_model_package.pt"), map_location="cpu", weights_only=False)
        self.b_H = float(package["b_H"])
        self.key_dim = int(package["key_dim"])
        self.key_map = {bytes(bytes.fromhex(k)): int(s) for k, s in zip(package["key_order"], package["key_slot"])}
        self.proto_val = np.asarray(package["proto_val"], np.float64)
        self.consistent = np.asarray(package["consistent"], bool)
        self.inference_rule = package["inference_rule"]
        # body + Q
        runner = _load_module(RUNNER, "frozen_h_runner")
        targets = np.load(self.src / "full_train_targets.npz", allow_pickle=False)
        self.bias_value = float(np.median(targets["c"].astype(np.float64)))
        self.q_head = runner.build_q_head(0, self.bias_value)
        self.q_head.load_state_dict(torch.load(self.src / "Q_raw_soup_state.pt", map_location="cpu", weights_only=True))
        self.q_head.eval()
        self._runner = runner
        self._q_forward = runner.q_forward

    # ---- key codec ----
    def key_of(self, row: np.ndarray) -> bytes:
        row = np.ascontiguousarray(row, np.float32)
        if row.shape != (self.key_dim,):
            raise RuntimeError(f"key shape mismatch: {row.shape}")
        if np.isnan(row).any() or np.isinf(row).any():
            raise RuntimeError("key contains NaN/Inf")
        neg = np.signbit(row) & (row == 0.0)
        if neg.any():
            row = row.copy()
            row[neg] = 0.0
        return row.tobytes()

    def route_of(self, row: np.ndarray) -> tuple[str, float | None]:
        slot = self.key_map.get(self.key_of(row))
        if slot is None:
            return "UNSEEN_FALLBACK", None
        if not self.consistent[slot]:
            return "TRAIN_CONFLICT_FALLBACK", None
        return "CONSISTENT_HIT", float(self.proto_val[slot])

    def q_frozen(self, T: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._q_forward(self.q_head, torch.as_tensor(np.asarray(T, np.float32))).double().numpy()

    def q_hybrid(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.q_frozen(T)
        route = np.empty(len(T), dtype=object)
        for i in range(len(T)):
            r, v = self.route_of(T[i])
            route[i] = r
            if v is not None:
                q[i] = v
        return q, route

    def forward(self, h_raw: np.ndarray, T: np.ndarray, want_diag: bool = False):
        """Standard forward returns (n,) y_cal; diagnostic adds q and routing."""
        h_raw = np.asarray(h_raw, np.float64).reshape(-1)
        T = np.asarray(T, np.float32)
        if T.shape[0] != h_raw.shape[0]:
            raise RuntimeError("length mismatch h_raw vs T25")
        q, route = self.q_hybrid(T)
        y = h_raw + q + self.b_H
        if want_diag:
            return {"y": y, "q": q, "route": route, "b_H": self.b_H}
        return y


def build_valid_T25() -> np.ndarray:
    runner = _load_module(RUNNER, "frozen_h_runner_valid")
    valid_data, _ = runner.load_valid_data(SRC)
    return runner.topology_matrix(valid_data)


if __name__ == "__main__":
    torch.set_num_threads(8)
    wrapper = FrozenHWrapper()
    vp = np.load(SRC / "valid_frozen_predictions.npz", allow_pickle=False)
    h_raw = vp["COMP_h_raw"].astype(np.float64)
    y = vp["y"].astype(np.float64)
    T = build_valid_T25()
    out = wrapper.forward(h_raw, T, want_diag=True)
    mae = float(np.mean(np.abs(y - out["y"])))
    print(f"[wrapper] n={out['y'].shape[0]} b_H={wrapper.b_H:.12f} valid H y_cal MAE={mae:.10f}")
    print(f"[wrapper] routes={ {r: int((out['route'] == r).sum()) for r in set(out['route'])} }")
    assert out["y"].shape == (1000,)
