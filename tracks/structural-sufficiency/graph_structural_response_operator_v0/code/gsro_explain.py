"""Section-10 explanation and frozen-intervention diagnostics for GSRO v0.

Only the best-Valid FULL checkpoint is used.  Everything runs in fp64 on CPU,
so the finite-difference / autograd checks are not polluted by fp32 rounding;
the stored fp32 predictions are reported alongside and their difference is
recorded.

    python code/gsro_explain.py --checkpoint runs/full/best.pt --out runs/full
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

import gsro_common as C
import gsro_data as D
from gsro_model import (
    GSROModel,
    full_response,
    local2_response,
)
from gsro_preflight import prediction_decomposition

TOL = 1e-9


def _inverse_vocab(mapping: dict[int, int]) -> dict[int, int]:
    return {index: raw for raw, index in mapping.items()}


def select_valid(config: dict[str, Any], prep) -> list[D.Graph]:
    n_wanted = int(config["explain_molecules"])
    graphs = prep.splits["valid"].graphs
    ordered = sorted(graphs, key=lambda g: C.stable_hash("gsro-v0-valid-mol", g.row_id))
    return ordered[:n_wanted]


def select_edges(config: dict[str, Any], graph: D.Graph) -> list[int]:
    order = sorted(
        range(graph.n_bonds),
        key=lambda e: C.stable_hash(
            "gsro-v0-edge", graph.row_id, int(graph.bu[e]), int(graph.bv[e])
        ),
    )
    return order[: int(config["explain_edges_per_molecule"])]


def load_model(checkpoint_path: Path, prep, config, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GSROModel(prep, config, "full").to(device=device, dtype=torch.float64)
    model.load_state_dict({k: v.double() for k, v in checkpoint["state_dict"].items()})
    model.eval()
    return model, checkpoint


def _grad_q(model, s, r, w, batch, channels):
    """dQ_k/dw for all channels (autograd through the exact solve)."""
    w_var = w.detach().clone().requires_grad_(True)
    q = full_response(
        s.detach(), r.detach(), w_var, batch["bu"], batch["bv"],
        batch["node_mask"], batch["edge_mask"],
    )
    grads = []
    for k in range(channels):
        grad = torch.autograd.grad(
            q[0, k], w_var, retain_graph=(k < channels - 1), create_graph=False
        )[0]
        grads.append(grad[0, :, k].detach())
    return torch.stack(grads, dim=-1), q.detach()  # [E,K]


def _fd_q(s, r, w, batch, edge, channel, eps):
    plus = w.clone()
    minus = w.clone()
    plus[0, edge, channel] += eps
    minus[0, edge, channel] -= eps
    q_plus = full_response(
        s, r, plus, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
    )
    q_minus = full_response(
        s, r, minus, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
    )
    return float(((q_plus - q_minus) / (2.0 * eps))[0, channel])


def run_explanation(config: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    prep = D.load_prepared(config)
    device = torch.device("cpu")
    model, checkpoint = load_model(checkpoint_path, prep, config, device)
    channels = int(config["channels"])
    eps = float(config["fd_epsilon"])

    selected = select_valid(config, prep)
    valid_graphs = prep.splits["valid"].graphs
    stored_pred_path = checkpoint_path.parent / "predictions.npz"
    stored_fp32 = None
    if stored_pred_path.is_file():
        with np.load(stored_pred_path) as handle:
            stored_fp32 = {
                "row_ids": np.asarray(handle["valid_row_ids"]),
                "y": np.asarray(handle["valid_y"]),
                "pred": np.asarray(handle["valid_pred"]),
            }

    atom_inverse = _inverse_vocab(prep.atom_vocab[0])
    bond_inverse = _inverse_vocab(prep.bond_vocab[0])

    # -- 10.1 exact decomposition on 64 hashed Valid molecules -------------
    decomposition_rows = []
    reconstruction_errors = []
    fp32_vs_fp64 = []
    for graph in selected:
        batch = D.single_graph(graph, device, dtype=torch.float64)
        with torch.no_grad():
            out = model(batch, return_x=True)
            decomp = prediction_decomposition(model, batch)
        pred64 = float(out["pred"][0])
        recon = float(decomp["reconstructed"][0])
        reconstruction_errors.append(abs(recon - pred64))
        if stored_fp32 is not None:
            index = int(np.where(stored_fp32["row_ids"] == graph.row_id)[0][0])
            fp32_vs_fp64.append(abs(float(stored_fp32["pred"][index]) - pred64))
        decomposition_rows.append(
            {
                "row_id": graph.row_id,
                "n_atoms": graph.n,
                "n_edges": graph.n_bonds,
                "prediction_fp64": pred64,
                "prediction_stored_fp32": (
                    float(stored_fp32["pred"][int(np.where(stored_fp32["row_ids"] == graph.row_id)[0][0])])
                    if stored_fp32 is not None
                    else None
                ),
                "reconstruction_fp64": recon,
                "reconstruction_abs_error": abs(recon - pred64),
                "bias": float(decomp["bias"]),
                "node_sum": float(decomp["node_contribution"].sum()),
                "edge_sum": float(decomp["edge_contribution"].sum()),
                "node_contribution": decomp["node_contribution"][0].tolist(),
                "edge_contribution": decomp["edge_contribution"][0].tolist(),
                "edge_u": graph.bu.tolist(),
                "edge_v": graph.bv.tolist(),
            }
        )

    # Full ledger for the first three molecules, with raw atom/bond types.
    ledgers = []
    for row in decomposition_rows[:3]:
        graph = next(g for g in selected if g.row_id == row["row_id"])
        atom_z = [atom_inverse[int(idx)] for idx in graph.atom_cat[:, 0]]
        bond_t = [bond_inverse[int(idx)] for idx in graph.bond_cat[:, 0]]
        ledger = {
            "row_id": row["row_id"],
            "prediction_fp64": row["prediction_fp64"],
            "bias": row["bias"],
            "reconstruction_fp64": row["reconstruction_fp64"],
            "reconstruction_abs_error": row["reconstruction_abs_error"],
            "nodes": [
                {
                    "atom_index": int(i),
                    "atomic_number": int(atom_z[i]),
                    "contribution": float(row["node_contribution"][i]),
                }
                for i in range(len(row["node_contribution"]))
            ],
            "edges": [
                {
                    "edge_index": int(e),
                    "u": int(row["edge_u"][e]),
                    "v": int(row["edge_v"][e]),
                    "bond_type": int(bond_t[e]),
                    "contribution": float(row["edge_contribution"][e]),
                }
                for e in range(len(row["edge_contribution"]))
            ],
            "node_total": float(sum(row["node_contribution"])),
            "edge_total": float(sum(row["edge_contribution"])),
        }
        ledgers.append(ledger)

    # -- 10.2 coupling sensitivity (autograd + FD spot checks) -------------
    sensitivity_rows = []
    analytic_vs_autograd = []
    analytic_vs_fd = []
    prediction_derivative_errors = []
    interventions = []
    edge_energy_vs_delta = []
    fd_spot = 0
    for graph in selected:
        batch = D.single_graph(graph, device, dtype=torch.float64)
        with torch.no_grad():
            s, r, w = model.generate(batch)
            q, x = full_response(
                s, r, w, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"],
                return_x=True,
            )
            pred, _ = model.predict_from_q(q, batch["n"])
        coeff = model.readout.coefficients(batch["n"], model.n_bar)[0].detach()
        sigma = float(model.sigma_y)
        hidden = x.shape[-1]
        index_u = batch["bu"].unsqueeze(-1).expand(-1, -1, hidden)
        index_v = batch["bv"].unsqueeze(-1).expand(-1, -1, hidden)
        x_u = x.gather(1, index_u)[0]
        x_v = x.gather(1, index_v)[0]
        analytic = -(x_u - x_v) ** 2  # [E,K]
        autograd_grad, _ = _grad_q(model, s, r, w, batch, channels)
        grad_err = (autograd_grad - analytic).abs()
        entropy_scale = analytic.abs().clamp_min(1e-8)
        ana_vs_auto_rel = (grad_err / entropy_scale).max().item()
        analytic_vs_autograd.append(
            {"max_abs": grad_err.max().item(), "max_rel": ana_vs_auto_rel}
        )

        pred_var = w.detach().clone().requires_grad_(True)
        q_var = full_response(
            s.detach(), r.detach(), pred_var, batch["bu"], batch["bv"],
            batch["node_mask"], batch["edge_mask"],
        )
        pred_from_q, _ = model.predict_from_q(q_var, batch["n"])
        pred_grad = torch.autograd.grad(pred_from_q[0], pred_var)[0][0]  # [E,K]
        pred_analytic = sigma * coeff.unsqueeze(0) * analytic  # [E,K]
        prediction_derivative_errors.append(
            float((pred_grad - pred_analytic).abs().max())
        )

        molecule_decomp = prediction_decomposition(model, batch)
        for edge in select_edges(config, graph):
            for k in range(channels):
                if fd_spot < 8:
                    fd_value = _fd_q(s, r, w, batch, edge, k, eps)
                    analytic_value = float(analytic[edge, k])
                    analytic_vs_fd.append(
                        {
                            "row_id": graph.row_id,
                            "edge": int(edge),
                            "channel": int(k),
                            "analytic": analytic_value,
                            "finite_difference": fd_value,
                            "abs_error": abs(fd_value - analytic_value),
                        }
                    )
                    fd_spot += 1
            w_zero = w.clone()
            w_zero[0, edge, :] = 0.0
            with torch.no_grad():
                q_zero = full_response(
                    s, r, w_zero, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
                )
                pred_zero, _ = model.predict_from_q(q_zero, batch["n"])
            delta = float(pred_zero[0] - pred[0])
            edge_energy = float(molecule_decomp["edge_contribution"][0, edge])
            interventions.append(float(delta))
            edge_energy_vs_delta.append(
                {
                    "row_id": graph.row_id,
                    "edge": int(edge),
                    "delta_prediction": float(delta),
                    "edge_energy_contribution": edge_energy,
                }
            )
            sensitivity_rows.append(
                {
                    "row_id": graph.row_id,
                    "edge": int(edge),
                    "u": int(graph.bu[edge]),
                    "v": int(graph.bv[edge]),
                    "max_abs_analytic": float(analytic[edge].abs().max()),
                    "zero_all_channels_delta_prediction": float(delta),
                }
            )

    # -- 10.3 frozen-parameter substitution on the full Valid split --------
    baseline = np.zeros(len(valid_graphs))
    local2_sub = np.zeros(len(valid_graphs))
    w0_sub = np.zeros(len(valid_graphs))
    y_valid = np.zeros(len(valid_graphs))
    for i, graph in enumerate(valid_graphs):
        batch = D.single_graph(graph, device, dtype=torch.float64)
        with torch.no_grad():
            s, r, w = model.generate(batch)
            q_full = full_response(
                s, r, w, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
            )
            q_local = local2_response(
                s, r, w, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
            )
            w_zero = torch.zeros_like(w)
            q_zero = full_response(
                s, r, w_zero, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
            )
            baseline[i] = float(model.predict_from_q(q_full, batch["n"])[0][0])
            local2_sub[i] = float(model.predict_from_q(q_local, batch["n"])[0][0])
            w0_sub[i] = float(model.predict_from_q(q_zero, batch["n"])[0][0])
        y_valid[i] = graph.y

    def rmse(pred):
        return float(np.sqrt(np.mean((pred - y_valid) ** 2)))

    substitution = {
        "n_valid": int(len(valid_graphs)),
        "full_fp64_rmse": rmse(baseline),
        "replaced_by_local2_rmse": rmse(local2_sub),
        "replaced_by_w0_node_response_rmse": rmse(w0_sub),
        "mean_abs_prediction_change_local2": float(np.mean(np.abs(local2_sub - baseline))),
        "max_abs_prediction_change_local2": float(np.max(np.abs(local2_sub - baseline))),
        "mean_abs_prediction_change_w0": float(np.mean(np.abs(w0_sub - baseline))),
        "max_abs_prediction_change_w0": float(np.max(np.abs(w0_sub - baseline))),
        "stored_fp32_valid_rmse": None,
        "max_abs_fp64_vs_stored_fp32_prediction": None,
    }
    if stored_fp32 is not None:
        order = np.argsort(stored_fp32["row_ids"])
        stored_sorted = stored_fp32["pred"][order]
        substitution["stored_fp32_valid_rmse"] = float(
            np.sqrt(np.mean((stored_sorted - y_valid) ** 2))
        )
        substitution["max_abs_fp64_vs_stored_fp32_prediction"] = float(
            np.max(np.abs(stored_sorted - baseline))
        )

    return {
        "round": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": C.sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "arm": checkpoint.get("arm"),
            "best": checkpoint.get("best"),
        },
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "environment": C.environment(),
        "data": prep.identity(),
        "dtype": "float64",
        "selection": {
            "valid_molecules": int(len(selected)),
            "valid_selection": "sha256('gsro-v0-valid-mol', row_id) ascending",
            "edges_per_molecule": int(config["explain_edges_per_molecule"]),
            "edge_selection": "sha256('gsro-v0-edge', row_id, u, v) ascending",
            "fd_epsilon": eps,
        },
        "prediction_decomposition": {
            "n_molecules": int(len(decomposition_rows)),
            "max_abs_reconstruction_error": float(max(reconstruction_errors)),
            "mean_abs_reconstruction_error": float(np.mean(reconstruction_errors)),
            "tolerance": TOL,
            "passed": bool(max(reconstruction_errors) < TOL),
            "max_abs_fp64_vs_stored_fp32_prediction": (
                float(max(fp32_vs_fp64)) if fp32_vs_fp64 else None
            ),
            "formula": "prediction = C_bias + sum_v C_v + sum_e C_e "
            "with C_v = sigma_y * sum_k c_k(n) r_vk x_vk^2 and "
            "C_e = sigma_y * sum_k c_k(n) w_ek (x_uk - x_vk)^2",
            "note": "exact energy decomposition of the current computation; "
            "not a unique chemical attribution",
            "molecules": decomposition_rows,
            "three_molecule_ledgers": ledgers,
        },
        "coupling_sensitivity": {
            "n_edges": int(len(sensitivity_rows)),
            "n_entries_channels": int(len(sensitivity_rows) * channels),
            "analytic_formula": "dQ_k/dw_ek = -(x_uk - x_vk)^2",
            "max_abs_analytic_vs_autograd": float(
                max(row["max_abs"] for row in analytic_vs_autograd)
            ),
            "max_rel_analytic_vs_autograd": float(
                max(row["max_rel"] for row in analytic_vs_autograd)
            ),
            "max_abs_prediction_derivative_error": float(
                max(prediction_derivative_errors)
            ),
            "finite_difference_spot_checks": analytic_vs_fd,
            "max_abs_analytic_vs_fd": float(
                max(row["abs_error"] for row in analytic_vs_fd)
            )
            if analytic_vs_fd
            else None,
            "zero_coupling_interventions": {
                "n": int(len(interventions)),
                "mean_abs_delta_prediction": float(np.mean(np.abs(interventions))),
                "max_abs_delta_prediction": float(np.max(np.abs(interventions))),
                "per_edge": edge_energy_vs_delta,
                "note": "setting all channels of one edge's w to zero and re-solving is "
                "NOT chemical bond deletion and its prediction change is NOT the edge "
                "energy contribution; no chemical-causal claim is made",
            },
            "per_edge": sensitivity_rows,
        },
        "frozen_substitution_full_valid": substitution,
        "interpretation_limits": [
            "The decomposition is exact for the computed response, not a unique chemical attribution.",
            "Coupling intervention (w -> 0) is not deleting a chemical bond.",
            "Edge energy contribution is not the prediction change after deleting the edge.",
            "Frozen substitution shows dependence on the operator for the current parameters; "
            "whether the global solve improves learning is decided by the independently trained FULL vs LOCAL2 arms.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = C.load_config()
    result = run_explanation(config, Path(args.checkpoint))
    C.write_json(Path(args.out) / "explanation.json", result)
    print(
        "GSRO_EXPLAIN reconstruction_max_error={:.3e} sensitivity_max={:.3e} "
        "sub_full={:.5f} sub_local2={:.5f} sub_w0={:.5f}".format(
            result["prediction_decomposition"]["max_abs_reconstruction_error"],
            result["coupling_sensitivity"]["max_abs_analytic_vs_autograd"],
            result["frozen_substitution_full_valid"]["full_fp64_rmse"],
            result["frozen_substitution_full_valid"]["replaced_by_local2_rmse"],
            result["frozen_substitution_full_valid"]["replaced_by_w0_node_response_rmse"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
