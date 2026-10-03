"""Deployable Y / P wrappers for zinc-full-decomposition-valid-test-confirmation-v1.

``Y_s(x)`` = ``f_s(x) + b_Y,s``
``P_s(x)`` = ``h_s(x) + q_s(T) + b_P,s``  (topology25 ``T`` comes from the graph)

Both wrappers take a graph batch; the fixed train-fit transforms that produce
the canonical fields are applied by the caller's adapter (see the runner's
``load_valid_data`` / ``load_test_data``).  The wrappers never receive or look up
``y`` / ``g`` / ``c`` / ``k`` / group / molecule id.

Run: ``PYTHONPATH=. uv run python tracks/ksvd/results/<dir>/deploy_wrapper.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import zinc_full_decomposition_valid_test_confirmation_v1 as R

RESULTS_DIR = R.RESULTS_DIR


def load_prep() -> dict[str, np.ndarray]:
    return R.load_prep()


def build_predictor(seed: int, mode: str = "P", *, blob=None) -> "nn.Module":
    """Return a callable ``p(batch, topo25=None) -> [n]`` in y units."""
    assert mode in ("Y", "P"), mode
    blob = load_prep() if blob is None else blob
    cal = R.read_json(RESULTS_DIR / "calibration.json")
    median_c = float(cal["median_train_c"])
    cs = cal["per_seed"][str(seed)]
    if mode == "Y":
        full = R._full_model(blob, seed)
        full.load_state_dict(torch.load(RESULTS_DIR / f"Y_seed{seed}_raw_soup_state.pt",
                                        map_location="cpu"))
        return R.FullDecompositionWrapper(full, None, cs["b_Y"], mode="Y")
    full = R._full_model(blob, seed)
    full.load_state_dict(torch.load(RESULTS_DIR / f"H_seed{seed}_raw_soup_state.pt",
                                    map_location="cpu"))
    head = R.build_head(seed, median_c)
    head.load_state_dict(torch.load(RESULTS_DIR / f"Q_seed{seed}_head_soup_state.pt",
                                    map_location="cpu"))
    return R.FullDecompositionWrapper(full, head, cs["b_P"], mode="P")


@torch.no_grad()
def predict(model: nn.Module, data, topo25: torch.Tensor | None = None) -> np.ndarray:
    return R.predict_full(model, data, torch.zeros(len(data)), torch.device("cpu")) if False else _wrapper_predict(model, data, topo25)


def _wrapper_predict(model: nn.Module, data, topo25=None) -> np.ndarray:
    T = topo25
    if T is None:
        T = torch.as_tensor(np.concatenate(
            [d.topology_features.reshape(1, -1).numpy().astype(np.float32) for d in data], 0))
    out = []
    with torch.no_grad():
        for indices in R._batches(len(data), R.BATCH_SIZE, torch.Generator().manual_seed(0), False):
            batch = R._batch(data, indices, torch.zeros(len(data)), torch.device("cpu"))
            out.append(model(batch, T[indices]).view(-1))
    return torch.cat(out).numpy().astype(np.float64)


def _sanity() -> int:
    blob = load_prep()
    cal = R.read_json(RESULTS_DIR / "calibration.json")
    train = R.zftd.load_train_only()[:128]
    R.apply_prep(train, blob)
    T = torch.as_tensor(np.concatenate(
        [d.topology_features.reshape(1, -1).numpy().astype(np.float32) for d in train], 0))
    checks = {}
    for seed in R.SEEDS:
        yp = np.load(RESULTS_DIR / f"Y_seed{seed}_fit_pred.npz")
        hp = np.load(RESULTS_DIR / f"H_seed{seed}_fit_pred.npz")
        qp = np.load(RESULTS_DIR / f"Q_seed{seed}_predictions.npz")
        pred_y = predict(build_predictor(seed, "Y", blob=blob), train, T)
        pred_p = predict(build_predictor(seed, "P", blob=blob), train, T)
        cached_y = yp["fit_raw"][:128] + cal["per_seed"][str(seed)]["b_Y"]
        cached_p = (hp["fit_raw"][:128] + qp["fit_q"][:128]
                    + cal["per_seed"][str(seed)]["b_P"])
        # label invariance: shuffle the batch .y (unused by the wrapper)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(train))
        for i, j in enumerate(perm):
            train[i].y = train[j].y
        pred_p2 = predict(build_predictor(seed, "P", blob=blob), train, T)
        checks[f"seed{seed}"] = {
            "Y_vs_cached_max_abs": float(np.max(np.abs(pred_y - cached_y))),
            "P_vs_cached_max_abs": float(np.max(np.abs(pred_p - cached_p))),
            "label_permutation_delta": float(np.max(np.abs(pred_p - pred_p2))),
        }
    checks["note"] = "wrapper accepts graph features + topology25 only"
    R.write_json(RESULTS_DIR / "wrapper_checks.json", checks)
    print(checks)
    ok = all(v["Y_vs_cached_max_abs"] <= R.REPLAY_TOL and v["P_vs_cached_max_abs"] <= R.REPLAY_TOL
             and v["label_permutation_delta"] == 0.0 for k, v in checks.items() if k.startswith("seed"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_sanity())