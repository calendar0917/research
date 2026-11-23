"""Prediction helpers shared by training, checks and analysis."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import sr_common as C
import sr_data as D
import sr_gnn as G
import sr_model as M
import sr_tensor as T


def collate(arm: str, rows: Sequence[int], structs: Mapping[int, Any],
            raw: Mapping[str, Any], device: Any) -> dict[str, Any]:
    if arm in M.ARMS:
        return T.collate_ef(rows, structs, raw, device)
    return G.collate(rows, structs, G.mode_of(arm), device)


def predict_raw(model: Any, arm: str, rows: Sequence[int],
                structs: Mapping[int, Any], raw: Mapping[str, Any], device: Any,
                batch_size: int, scaler: Mapping[str, float]
                ) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows = np.asarray(rows, np.int64)
    pred_std = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = collate(arm, part.tolist(), structs, raw, device)
            out = model(batch)
            pred_std[start:start + part.size] = out["yhat_std"].double().cpu().numpy()
    pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
    return pred_raw, pred_std


def load_trained(run_dir: str, arm: str, bundle: Any, cfg: Mapping[str, Any],
                 device: Any = "cpu", registry: Any = None) -> tuple[Any, dict[str, Any]]:
    """Rebuild the architecture, load the saved state and return (model, blob).

    The graph-scalar normalisers are model buffers, so they come back with the
    state dict; a zero placeholder norm is used only during construction.
    """
    blob = torch.load(C.ROUND_DIR / run_dir / "state_best.pt", map_location=device,
                      weights_only=False)
    atom_dims, bond_dims = D.feature_dims()
    placeholder = {"graph_mean": np.zeros(8), "graph_scale": np.ones(8)}
    if arm in M.ARMS:
        model, _ = M.build_family_models(cfg, atom_dims, bond_dims, 0, placeholder, arm)
        if arm[0] == "F":
            if registry is None:
                raise ValueError("F model needs a registry")
            model.set_registry(registry)
    else:
        model, _ = G.build_model(arm, atom_dims, bond_dims, 0, placeholder, cfg)
    model.load_state_dict(blob["state"], strict=True)
    model.to(device)
    model.eval()
    return model, blob
