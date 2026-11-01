"""Attribution diagnostics for the strongest Sprint-04 signal.

The round did not produce a multi-task winner, so this script is deliberately
narrow: it explains the one robust signal (the chem8+contrib combo on
Lipophilicity) by intervening on the two learned extensions of a frozen
checkpoint.  It never trains and never selects a checkpoint.

Interventions
-------------
``full``        contrib + rho + delta
``no_delta``    contrib + rho            (removes MULTICONTRIB8)
``no_chem8``    rho + delta with the 8 atom channels zeroed (removes CHEMLOCAL8)
``base_like``   rho with chem8 zeroed and delta zero (BASE path)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cl_common as C
import cl_data as CD
import cl_chem8 as CH
import cl_model as CM
import mb2_data as D
import lcrc_data as PREP_DATA

CHEM8_NAMES = (
    "CIP_R",
    "CIP_S",
    "CIP_UNASSIGNED_TETRA",
    "HBD",
    "HBA",
    "HYB_SP",
    "HYB_SP2",
    "HYB_SP3",
)


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _forward_cache(model, cache, device, *, zero_chem8: bool, batch_size: int = 256) -> dict[str, np.ndarray]:
    n = int(cache["n_molecules"])
    work = dict(cache)
    if zero_chem8:
        atom_feat = np.array(work["atom_feat"], copy=True)
        base_width = int(work.get("base_atom_feat_dim", 51))
        atom_feat[:, base_width:] = 0.0
        work["atom_feat"] = atom_feat
    contrib, rho, delta, q_graph, y = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = D.to_device(PREP_DATA.collate(work, rows), device)
            pred, aux = model(batch, return_aux=True)
            contrib.append(aux["contrib"].detach().float().cpu().numpy().reshape(-1))
            rho.append(aux["rho"].detach().float().cpu().numpy().reshape(-1))
            delta.append(aux["delta"].detach().float().cpu().numpy().reshape(-1))
            q_graph.append(aux["q_graph"].detach().float().cpu().numpy())
            y.append(pred.detach().float().cpu().numpy().reshape(-1))
    return {
        "contrib": np.concatenate(contrib),
        "rho": np.concatenate(rho),
        "delta": np.concatenate(delta),
        "q_graph": np.concatenate(q_graph),
        "y": np.concatenate(y),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="attribution for a frozen combo checkpoint")
    parser.add_argument("--run", default="p2b_combo_chem_contrib_lipo")
    parser.add_argument("--split", default="published")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(C.ANALYSIS / "attribution_combo_lipo.json"))
    args = parser.parse_args(argv)

    run_dir = C.run_dir(args.run)
    train = json.loads((run_dir / "train.json").read_text())
    arm = train["arm"]
    if arm != "combo_chem_contrib":
        raise SystemExit(f"expected combo_chem_contrib, got {arm}")
    device = torch.device(args.device)
    view, base_spec, _uses = CD.view_for_arm("Lipophilicity", args.split, arm)
    model = CM.load_variant_model(
        view.spec,
        view.train_norm,
        torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)["state_dict"],
        device,
        arm=arm,
    )
    test_cache, y_test, test_rows, _base = CD.test_view("Lipophilicity", args.split)
    test_cache = CH.extend_cache(test_cache, view.spec, dataset="Lipophilicity")
    test_cache = dict(test_cache)
    test_cache["base_atom_feat_dim"] = int(base_spec.atom_feat_dim)

    full = _forward_cache(model, test_cache, device, zero_chem8=False)
    no_chem8 = _forward_cache(model, test_cache, device, zero_chem8=True)

    y = np.asarray(y_test, dtype=np.float64).reshape(-1)
    scale = float(view.target_norm["scale"])
    mean = float(view.target_norm["mean"])
    pred_full = full["y"] * scale + mean
    pred_no_delta = (full["contrib"] + full["rho"]) * scale + mean
    pred_no_chem8 = (no_chem8["contrib"] + no_chem8["rho"] + no_chem8["delta"]) * scale + mean
    pred_base_like = (no_chem8["contrib"] + no_chem8["rho"]) * scale + mean

    # Chem8 channel activity on the test cache.
    atom_feat = np.asarray(test_cache["atom_feat"])
    ptr = np.asarray(test_cache["atom_ptr"], dtype=np.int64)
    chem8 = atom_feat[:, int(base_spec.atom_feat_dim) :]
    activity = {}
    for j, name in enumerate(CHEM8_NAMES):
        col = chem8[:, j]
        per_mol_any = np.array([bool(col[ptr[i] : ptr[i + 1]].any()) for i in range(len(ptr) - 1)])
        activity[name] = {
            "atom_mean": float(col.mean()),
            "atom_fraction_active": float((col > 0).mean()),
            "molecules_with_any": int(per_mol_any.sum()),
            "molecule_fraction_with_any": float(per_mol_any.mean()),
        }

    # Weight usage.
    first = model.atom_encoder.lins[0].weight.detach().float().cpu().numpy()
    base_width = int(base_spec.atom_feat_dim)
    old_norm = float(np.linalg.norm(first[:, :base_width]))
    new_norm = float(np.linalg.norm(first[:, base_width:]))
    q_w = model.q.lins[-1].weight.detach().float().cpu().numpy()
    delta_w = model.delta.lins[-1].weight.detach().float().cpu().numpy()

    # MULTICONTRIB8 latent channels.
    q = full["q_graph"]
    q_mean = q.mean(0)
    q_std = q.std(0)
    corr = np.corrcoef(q, rowvar=False)
    collapsed = [i for i, s in enumerate(q_std) if float(s) < 1e-4]
    delta = full["delta"] * scale
    chem8_effect = pred_full - pred_no_chem8
    per_mol = {}
    for j, name in enumerate(CHEM8_NAMES):
        col = chem8[:, j]
        per_mol[name] = np.array([bool(col[ptr[i] : ptr[i + 1]].any()) for i in range(len(ptr) - 1)])
    groups = {
        "chirality": per_mol["CIP_R"] | per_mol["CIP_S"] | per_mol["CIP_UNASSIGNED_TETRA"],
        "HBD": per_mol["HBD"],
        "HBA": per_mol["HBA"],
        "hybridisation": per_mol["HYB_SP"] | per_mol["HYB_SP2"] | per_mol["HYB_SP3"],
    }
    group_effect = {}
    for name, mask in groups.items():
        group_effect[name] = {
            "n_with": int(mask.sum()),
            "n_without": int((~mask).sum()),
            "mean_abs_effect_with": float(np.abs(chem8_effect[mask]).mean()) if mask.any() else None,
            "mean_abs_effect_without": float(np.abs(chem8_effect[~mask]).mean()) if (~mask).any() else None,
            "mean_effect_with": float(chem8_effect[mask].mean()) if mask.any() else None,
            "mean_effect_without": float(chem8_effect[~mask].mean()) if (~mask).any() else None,
        }
    top_chem8 = np.argsort(-np.abs(chem8_effect))[:10]
    top_delta = np.argsort(-np.abs(delta))[:10]

    payload: dict[str, Any] = {
        "schema": "lcrc-attribution/v1",
        "label": "TEST_USED_FOR_DEVELOPMENT",
        "run": args.run,
        "dataset": "Lipophilicity",
        "split": args.split,
        "checkpoint": str(run_dir / "checkpoint.pt"),
        "checkpoint_sha256": C.sha256_file(run_dir / "checkpoint.pt"),
        "n_test": int(len(y)),
        "metrics": {
            "full_rmse": _rmse(y, pred_full),
            "no_delta_rmse": _rmse(y, pred_no_delta),
            "no_chem8_rmse": _rmse(y, pred_no_chem8),
            "base_like_rmse": _rmse(y, pred_base_like),
        },
        "prediction_change": {
            "delta_abs_mean": float(np.abs(delta).mean()),
            "delta_abs_max": float(np.abs(delta).max()),
            "chem8_abs_mean": float(np.abs(pred_full - pred_no_chem8).mean()),
            "chem8_abs_max": float(np.abs(pred_full - pred_no_chem8).max()),
        },
        "chem8": {
            "base_atom_feat_dim": base_width,
            "extended_atom_feat_dim": int(view.spec.atom_feat_dim),
            "weight_norm_original_columns": old_norm,
            "weight_norm_new_columns": new_norm,
            "new_over_old_norm_ratio": new_norm / max(old_norm, 1e-12),
            "channel_activity": activity,
            "group_effect": group_effect,
        },
        "multicontrib8": {
            "q_mean": q_mean.tolist(),
            "q_std": q_std.tolist(),
            "q_var": (q_std**2).tolist(),
            "channel_correlation": corr.tolist(),
            "max_abs_offdiag_correlation": float(np.max(np.abs(corr - np.eye(8)))),
            "collapsed_channels_std_lt_1e_4": collapsed,
            "q_last_layer_weight_norm": float(np.linalg.norm(q_w)),
            "delta_last_layer_weight_norm": float(np.linalg.norm(delta_w)),
        },
        "top_chem8_molecules": [
            {"row": int(test_rows[i]), "chem8_effect": float(chem8_effect[i])} for i in top_chem8
        ],
        "top_delta_molecules": [
            {"row": int(test_rows[i]), "delta": float(delta[i])} for i in top_delta
        ],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["metrics"], indent=2))
    print(json.dumps(payload["chem8"]["channel_activity"], indent=2))
    print("q_std", [round(float(x), 6) for x in q_std])
    print("collapsed", collapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
