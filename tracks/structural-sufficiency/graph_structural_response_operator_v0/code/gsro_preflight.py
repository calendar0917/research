"""Preflight for GSRO v0: the eight required numerical/engineering checks and a
short throughput pilot that fixes the common training horizon.

    python code/gsro_preflight.py --mode both --out-dir runs/preflight

Checks always run in fp64 on CPU (independent reference) and, when available,
exercise the fp32 CUDA training path as well.  The pilot only measures speed;
it never looks at a metric, and the formal runs re-initialise from seed 0.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

import gsro_common as C
import gsro_data as D
from gsro_model import (
    GSROModel,
    full_response,
    full_residual,
    local2_response,
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def components(model: GSROModel, batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    s, r, w = model.generate(batch)
    return s, r, w


def _numpy_reference(s: np.ndarray, r: np.ndarray, w: np.ndarray, bu: np.ndarray, bv: np.ndarray):
    """Dense fp64 reference for one molecule, per channel, plus LOCAL2 parts."""
    n, channels = s.shape
    d = r.copy()
    np.add.at(d, bu, w)
    np.add.at(d, bv, w)
    k_mat = np.zeros((channels, n, n))
    k_mat[:, np.arange(n), np.arange(n)] = d.T
    for e in range(bu.size):
        k_mat[:, bu[e], bv[e]] -= w[e]
        k_mat[:, bv[e], bu[e]] -= w[e]
    x = np.linalg.solve(k_mat, s.T[..., None])[..., 0].T  # [n, K]
    q_full = (s * x).sum(axis=0)

    # Explicit 0/1/2 polynomial: Q = s^T (I + M + M^2) D^{-1} s, M = D^{-1} A.
    t = s / d
    q0 = (s * t).sum(axis=0)
    q1 = np.zeros(channels)
    q2 = np.zeros(channels)
    for k in range(channels):
        a_k = np.zeros((n, n))
        for e in range(bu.size):
            a_k[bu[e], bv[e]] += w[e, k]
            a_k[bv[e], bu[e]] += w[e, k]
        m = a_k / d[:, k][:, None]
        q1[k] = s[:, k] @ (m @ t[:, k])
        q2[k] = s[:, k] @ (m @ (m @ t[:, k]))
    q_poly = q0 + q1 + q2
    return {
        "K": k_mat,
        "d": d,
        "A": None,
        "x": x,
        "q_full": q_full,
        "q_local2": q_poly,
        "Q0": q0,
        "Q1": q1,
        "Q2": q2,
    }


def _components_numpy(model: GSROModel, graph: D.Graph, device="cpu", dtype=torch.float64):
    batch = D.single_graph(graph, device, dtype=dtype)
    model_eval = model.to(device=device, dtype=dtype)
    model_eval.eval()
    with torch.no_grad():
        s, r, w = model_eval.generate(batch)
        out = model_eval(batch, return_x=True)
    return batch, model_eval, s.detach().cpu().numpy()[0], r.detach().cpu().numpy()[0], w.detach().cpu().numpy()[0], out


def _max_rel_error(a: np.ndarray, b: np.ndarray) -> float:
    scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return float(np.max(np.abs(a - b) / scale)) if a.size else 0.0


def prediction_decomposition(model: GSROModel, batch: dict[str, Any]) -> dict[str, Any]:
    """Exact energy decomposition of the FULL response for one padded batch."""
    with torch.enable_grad():
        s, r, w = model.generate(batch)
    q, x = full_response(
        s, r, w, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"],
        return_x=True,
    )
    pred, pred_std = model.predict_from_q(q, batch["n"])
    coeff = model.readout.coefficients(batch["n"], model.n_bar)  # [B,K]
    sigma = float(model.sigma_y)
    node_mask = batch["node_mask"]
    edge_mask = batch["edge_mask"]
    node_energy = (coeff.unsqueeze(1) * r * x**2 * node_mask.unsqueeze(-1)).sum(dim=-1)  # [B,N]
    hidden = x.shape[-1]
    index_u = batch["bu"].unsqueeze(-1).expand(-1, -1, hidden)
    index_v = batch["bv"].unsqueeze(-1).expand(-1, -1, hidden)
    x_u = x.gather(1, index_u)
    x_v = x.gather(1, index_v)
    edge_energy = (
        coeff.unsqueeze(1) * w * (x_u - x_v) ** 2 * edge_mask.unsqueeze(-1)
    ).sum(dim=-1)  # [B,E]
    bias = float(model.mu_y) + sigma * float(model.readout.bias.detach())
    node_original = sigma * node_energy
    edge_original = sigma * edge_energy
    recon = bias + node_original.sum(dim=1) + edge_original.sum(dim=1)
    return {
        "q": q,
        "x": x,
        "pred": pred,
        "pred_std": pred_std,
        "bias": bias,
        "node_contribution": node_original,
        "edge_contribution": edge_original,
        "reconstructed": recon,
        "reconstruction_abs_error": (recon - pred).abs(),
        "node_mask": node_mask,
        "edge_mask": edge_mask,
    }


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def run_checks(config: dict[str, Any], cli_device: str) -> dict[str, Any]:
    seed = int(config["seed"])
    prep = D.load_prepared(config)

    train_graphs = sorted(prep.splits["train"].graphs, key=lambda g: (g.n, g.row_id))
    pick = [train_graphs[0], train_graphs[3], train_graphs[10], train_graphs[20]]
    check_device = torch.device("cpu")

    checks: dict[str, Any] = {
        "round": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "environment": C.environment(),
        "data": prep.identity(),
        "selected_graph_row_ids": [g.row_id for g in pick],
        "checks": {},
    }

    # -- shared initialisation and parameter count -------------------------
    C.set_seed(seed)
    model_full = GSROModel(prep, config, "full").to(check_device)
    C.set_seed(seed)
    model_local = GSROModel(prep, config, "local2").to(check_device)
    state_full = {k: v.detach().clone() for k, v in model_full.state_dict().items()}
    state_local = {k: v.detach().clone() for k, v in model_local.state_dict().items()}
    same_keys = sorted(state_full) == sorted(state_local)
    identical = same_keys and all(torch.equal(state_full[k], state_local[k]) for k in state_full)
    pad_graphs = [train_graphs[i] for i in (0, 1, 2, 5, 8, 13)]
    padded = D.collate(pad_graphs, check_device)
    model_full.eval()
    with torch.no_grad():
        cf, rf, wf = model_full.generate(padded)
    model_local.eval()
    with torch.no_grad():
        cl, rl, wl = model_local.generate(padded)
    checks["checks"]["shared_initialisation"] = {
        "param_count_full": int(model_full.parameter_count()),
        "param_count_local2": int(model_local.parameter_count()),
        "state_dict_keys_equal": bool(same_keys),
        "initial_state_bitwise_equal": bool(identical),
        "initial_state_sha256": model_full.initial_state_sha(),
        "generator_outputs_bitwise_equal": {
            "s": bool(torch.equal(cf, cl)),
            "r": bool(torch.equal(rf, rl)),
            "w": bool(torch.equal(wf, wl)),
        },
        "passed": bool(
            model_full.parameter_count() == model_local.parameter_count()
            and same_keys
            and identical
            and torch.equal(cf, cl)
            and torch.equal(rf, rl)
            and torch.equal(wf, wl)
        ),
    }

    # -- 1. atom renumbering + edge order invariance (FULL and LOCAL2) -----
    model_full = model_full.to(torch.float64)
    graph = pick[1]
    rng = np.random.default_rng(0)
    perm = rng.permutation(graph.n)
    inverse = np.argsort(perm)
    permuted_edges = rng.permutation(graph.n_bonds)
    new_u = inverse[graph.bu[permuted_edges]]
    new_v = inverse[graph.bv[permuted_edges]]
    graph_perm = replace(
        graph,
        atom_cat=graph.atom_cat[perm],
        nb_counts=graph.nb_counts[perm],
        bond_cat=graph.bond_cat[permuted_edges],
        bu=np.minimum(new_u, new_v),
        bv=np.maximum(new_u, new_v),
    )
    inv = {}
    for name, target in (("original", graph), ("permuted", graph_perm)):
        batch = D.single_graph(target, check_device, dtype=torch.float64)
        model_full.eval()
        with torch.no_grad():
            out = model_full(batch, return_x=False)
            q_local = model_full.response(
                out["s"], out["r"], out["w"], batch, arm="local2"
            )
        inv[name] = {
            "q_full": out["q"].cpu().numpy()[0],
            "q_local2": q_local.cpu().numpy()[0],
            "pred": out["pred"].cpu().numpy()[0],
        }
    err_full = _max_rel_error(inv["original"]["q_full"], inv["permuted"]["q_full"])
    err_local = _max_rel_error(inv["original"]["q_local2"], inv["permuted"]["q_local2"])
    err_pred = abs(float(inv["original"]["pred"]) - float(inv["permuted"]["pred"]))
    checks["checks"]["renumbering_and_edge_order"] = {
        "row_id": graph.row_id,
        "n_atoms": graph.n,
        "n_edges": graph.n_bonds,
        "max_rel_q_full": err_full,
        "max_rel_q_local2": err_local,
        "abs_pred_diff": err_pred,
        "tolerance": 1e-10,
        "passed": bool(err_full < 1e-10 and err_local < 1e-10 and err_pred < 1e-10),
    }

    # -- 2. batch vs single graph -----------------------------------------
    batch_graphs = [train_graphs[i] for i in (0, 2, 4, 6, 9, 12, 16)]
    batch = D.collate(batch_graphs, check_device)
    model_full.eval()
    with torch.no_grad():
        out_batch = model_full(batch, return_x=False)
        q_local_batch = model_full.response(
            out_batch["s"], out_batch["r"], out_batch["w"], batch, arm="local2"
        )
    single_q = []
    single_local = []
    single_pred = []
    for g in batch_graphs:
        b = D.single_graph(g, check_device, dtype=torch.float64)
        with torch.no_grad():
            o = model_full(b)
            ql = model_full.response(o["s"], o["r"], o["w"], b, arm="local2")
        single_q.append(o["q"].cpu().numpy()[0])
        single_local.append(ql.cpu().numpy()[0])
        single_pred.append(float(o["pred"].cpu().numpy()[0]))
    batch_q = out_batch["q"].cpu().numpy()
    batch_local = q_local_batch.cpu().numpy()
    batch_pred = out_batch["pred"].cpu().numpy()
    err_batch_full = _max_rel_error(np.stack(single_q), batch_q)
    err_batch_local = _max_rel_error(np.stack(single_local), batch_local)
    err_batch_pred = _max_rel_error(np.asarray(single_pred), batch_pred)
    checks["checks"]["batch_vs_single"] = {
        "row_ids": [g.row_id for g in batch_graphs],
        "n_atoms_max": int(batch["node_mask"].sum(dim=1).max()),
        "batch_size": len(batch_graphs),
        "max_rel_q_full": err_batch_full,
        "max_rel_q_local2": err_batch_local,
        "max_rel_pred": err_batch_pred,
        "tolerance": 1e-10,
        "passed": bool(
            err_batch_full < 1e-10 and err_batch_local < 1e-10 and err_batch_pred < 1e-10
        ),
    }

    # -- 3. K symmetric positive definite + small residual -----------------
    spd: dict[str, Any] = {}
    model_full.eval()
    model32 = GSROModel(prep, config, "full").to(check_device)
    model32.load_state_dict({k: v.float() for k, v in state_full.items()})
    model32.eval()
    for dtype, name, model_dtype in (
        (torch.float64, "fp64", model_full),
        (torch.float32, "fp32", model32),
    ):
        b = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            s, r, w = model_dtype.generate(b)
            k_mat, x, residual, info = full_residual(
                s, r, w, b["bu"], b["bv"], b["node_mask"], b["edge_mask"]
            )
            eig = torch.linalg.eigvalsh(k_mat.double())
        sym = float((k_mat - k_mat.transpose(-1, -2)).abs().max())
        min_eig = float(eig.min())
        max_eig = float(eig.max())
        rhs = (s * b["node_mask"].unsqueeze(-1)).transpose(1, 2).unsqueeze(-1)
        rhs_inf = float(rhs.abs().max())
        res_inf = float(residual.abs().max())
        spd[name] = {
            "dtype": str(dtype),
            "max_abs_asymmetry": sym,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": max_eig / max(min_eig, 1e-30),
            "max_abs_residual": res_inf,
            "rhs_inf": rhs_inf,
            "relative_residual": res_inf / max(rhs_inf, 1e-30),
            "cholesky_info_nonzero": int((info != 0).sum()),
        }
    tolerance = {"fp64": 1e-9, "fp32": 5e-4}
    spd["passed"] = bool(
        spd["fp64"]["max_abs_asymmetry"] < 1e-12
        and spd["fp64"]["min_eigenvalue"] > 0
        and spd["fp64"]["relative_residual"] < tolerance["fp64"]
        and spd["fp32"]["max_abs_asymmetry"] < 1e-6
        and spd["fp32"]["min_eigenvalue"] > 0
        and spd["fp32"]["relative_residual"] < tolerance["fp32"]
        and spd["fp64"]["cholesky_info_nonzero"] == 0
        and spd["fp32"]["cholesky_info_nonzero"] == 0
    )
    checks["checks"]["symmetric_positive_definite_residual"] = spd

    # -- 4/5/6/8. fp64 reference comparisons on real molecules -------------
    ref_full_errors = []
    ref_local_errors = []
    w0_errors = []
    decomp_errors = []
    numpy_checks = []
    for g in pick:
        b, mdl, s, r, w, out = _components_numpy(model_full, g)
        ref = _numpy_reference(s, r, w, g.bu, g.bv)
        with torch.no_grad():
            q_full = out["q"].cpu().numpy()[0]
            q_local = local2_response(
                out["s"], out["r"], out["w"], b["bu"], b["bv"], b["node_mask"], b["edge_mask"]
            ).cpu().numpy()[0]
            # w = 0: FULL and LOCAL2 must coincide (independent node response).
            w_zero = torch.zeros_like(out["w"])
            q_full_zero = full_response(
                out["s"], out["r"], w_zero, b["bu"], b["bv"], b["node_mask"], b["edge_mask"]
            ).cpu().numpy()[0]
            q_local_zero = local2_response(
                out["s"], out["r"], w_zero, b["bu"], b["bv"], b["node_mask"], b["edge_mask"]
            ).cpu().numpy()[0]
        decomp = prediction_decomposition(model_full, b)
        ref_full_errors.append(_max_rel_error(q_full, ref["q_full"]))
        ref_local_errors.append(_max_rel_error(q_local, ref["q_local2"]))
        w0_errors.append(_max_rel_error(q_full_zero, q_local_zero))
        decomp_errors.append(float(decomp["reconstruction_abs_error"].abs().max()))
        numpy_checks.append(
            {
                "row_id": g.row_id,
                "n_atoms": g.n,
                "max_rel_full_vs_numpy": ref_full_errors[-1],
                "max_rel_local2_vs_polynomial": ref_local_errors[-1],
                "max_rel_local2_parts_vs_polynomial": _max_rel_error(
                    np.asarray(
                        [
                            q_local.sum(),
                        ]
                    ),
                    np.asarray([ref["q_local2"].sum()]),
                ),
                "max_rel_w0_full_vs_local2": w0_errors[-1],
                "decomposition_abs_error": decomp_errors[-1],
                "condition_number": float(max(np.linalg.cond(mat) for mat in ref["K"])),
            }
        )
    checks["checks"]["full_vs_fp64_reference"] = {
        "per_graph": numpy_checks,
        "max_rel_error": max(ref_full_errors),
        "tolerance": 1e-9,
        "passed": bool(max(ref_full_errors) < 1e-9),
    }
    checks["checks"]["local2_vs_matrix_polynomial"] = {
        "per_graph": [row["max_rel_local2_vs_polynomial"] for row in numpy_checks],
        "max_rel_error": max(ref_local_errors),
        "tolerance": 1e-9,
        "passed": bool(max(ref_local_errors) < 1e-9),
    }
    checks["checks"]["w_zero_full_equals_local2"] = {
        "per_graph": [row["max_rel_w0_full_vs_local2"] for row in numpy_checks],
        "max_rel_error": max(w0_errors),
        "tolerance": 1e-12,
        "passed": bool(max(w0_errors) < 1e-12),
    }
    checks["checks"]["exact_decomposition_recomputes_prediction"] = {
        "per_graph_abs_error": decomp_errors,
        "max_abs_error": max(decomp_errors),
        "tolerance": 1e-9,
        "passed": bool(max(decomp_errors) < 1e-9),
    }

    # -- 7. every trainable path has finite, non-degenerate gradients ------
    dev = torch.device(cli_device)
    C.set_seed(seed)
    grad_model = GSROModel(prep, config, "full").to(dev)
    grad_batch = D.collate([train_graphs[i] for i in (0, 1, 2, 3, 4, 5)], dev)
    grad_model.train()
    grad_model.zero_grad(set_to_none=True)
    out = grad_model(grad_batch)
    loss = torch.nn.functional.mse_loss(out["pred_std"], grad_batch["y_std"])
    loss.backward()
    per_param = {}
    finite = True
    missing = []
    for name, param in grad_model.named_parameters():
        if param.grad is None:
            missing.append(name)
            finite = False
            continue
        grad = param.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all())
        per_param[name] = {
            "finite": bool(torch.isfinite(grad).all()),
            "max_abs": float(grad.abs().max()),
            "nonzero": bool(grad.abs().max() > 0),
        }
    head_grads = {
        name: per_param[name]["max_abs"]
        for name in ("source_head.weight", "source_head.bias", "self_head.weight",
                     "self_head.bias", "edge_head.weight", "edge_head.bias",
                     "readout.alpha", "readout.beta")
    }
    with torch.no_grad():
        s, r, w = grad_model.generate(grad_batch)
    response_grad = {
        "s_std": float(s.std()),
        "r_std": float(r.std()),
        "w_std": float(w.std()),
    }
    checks["checks"]["finite_effective_gradients"] = {
        "device": str(dev),
        "loss_std": float(loss.detach()),
        "params_with_missing_grad": missing,
        "all_grads_finite": bool(finite),
        "head_grad_max_abs": head_grads,
        "all_heads_nonzero_grad": bool(all(v > 0 for v in head_grads.values())),
        "response_output_spread": response_grad,
        "all_response_paths_active": bool(
            response_grad["s_std"] > 0 and response_grad["r_std"] > 0 and response_grad["w_std"] > 0
        ),
        "passed": bool(
            finite
            and not missing
            and all(v > 0 for v in head_grads.values())
            and response_grad["s_std"] > 0
            and response_grad["r_std"] > 0
            and response_grad["w_std"] > 0
        ),
    }

    # -- response path check on the training device (fp32) -----------------
    with torch.no_grad():
        q_full_dev = full_response(
            s, r, w, grad_batch["bu"], grad_batch["bv"],
            grad_batch["node_mask"], grad_batch["edge_mask"],
        )
        q_local_dev = local2_response(
            s, r, w, grad_batch["bu"], grad_batch["bv"],
            grad_batch["node_mask"], grad_batch["edge_mask"],
        )
    checks["checks"]["device_response_consistency"] = {
        "device": str(dev),
        "max_abs_q_diff_full_vs_local2": float((q_full_dev - q_local_dev).abs().max()),
        "note": "not expected to vanish; recorded to prove both paths execute on device",
        "passed": bool(torch.isfinite(q_full_dev).all() and torch.isfinite(q_local_dev).all()),
    }

    checks["all_passed"] = all(
        bool(entry.get("passed", False)) for entry in checks["checks"].values()
    )
    return checks


# ---------------------------------------------------------------------------
# throughput pilot
# ---------------------------------------------------------------------------


def _time_train_steps(model, batches, repeats: int) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.mse_loss
    for batch in batches[:2]:
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = loss_fn(out["pred_std"], batch["y_std"])
        loss.backward()
        optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(repeats):
        batch = batches[i % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = loss_fn(out["pred_std"], batch["y_std"])
        loss.backward()
        optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def _time_eval(model, batches) -> float:
    model.eval()
    with torch.no_grad():
        for batch in batches:
            model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def run_pilot(config: dict[str, Any], device_name: str) -> dict[str, Any]:
    seed = int(config["seed"])
    prep = D.load_prepared(config)
    device = torch.device(device_name)
    batch_size = int(config["batch_size"])
    eval_batch_size = int(config["eval_batch_size"])
    train_batches = D.size_sorted_batches(prep.splits["train"], batch_size, device)
    eval_batches = {
        name: D.size_sorted_batches(prep.splits[name], eval_batch_size, device)
        for name in ("train", "valid", "test")
    }
    per_arm: dict[str, Any] = {}
    for arm in ("full", "local2"):
        C.set_seed(seed)
        model = GSROModel(prep, config, arm).to(device)
        train_model = GSROModel(prep, config, arm).to(device)
        step_seconds = _time_train_steps(train_model, train_batches, repeats=8)
        eval_seconds = _time_eval(model, train_batches)
        eval_seconds += sum(_time_eval(model, eval_batches[name]) for name in ("valid", "test"))
        per_epoch = step_seconds * len(train_batches) + eval_seconds
        per_arm[arm] = {
            "train_step_seconds": step_seconds,
            "train_batches": len(train_batches),
            "eval_seconds_all_splits": eval_seconds,
            "projected_epoch_seconds": per_epoch,
            "projected_total_seconds": {
                str(epochs): per_epoch * epochs for epochs in config["epochs_candidates"]
            },
        }
        del model, train_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    slowest = max(
        per_arm, key=lambda arm: per_arm[arm]["projected_total_seconds"][
            str(int(config["epochs_candidates"][0]))
        ]
    )
    target = float(config["target_wall_minutes"]) * 60.0
    hard = float(config["hard_cap_minutes"]) * 60.0
    projections = {
        str(epochs): max(per_arm[arm]["projected_total_seconds"][str(epochs)] for arm in per_arm)
        for epochs in config["epochs_candidates"]
    }
    if projections[str(int(config["epochs_candidates"][0]))] <= target:
        horizon = int(config["epochs_candidates"][0])
        reason = "projected 120 epochs within target wall time"
    else:
        viable = [
            int(epochs)
            for epochs in config["epochs_candidates"][1:]
            if projections[str(epochs)] <= target
        ]
        if viable:
            horizon = max(viable)
            reason = "120 epochs exceed target; largest candidate within target selected"
        else:
            smallest = int(config["epochs_candidates"][-1])
            if projections[str(smallest)] <= hard:
                horizon = smallest
                reason = "no candidate within target; 60 epochs within hard cap selected"
            else:
                horizon = None
                reason = "even 60 epochs exceed the hard cap: COMPUTE_BOTTLENECK"
    return {
        "round": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "environment": C.environment(),
        "device": device_name,
        "per_arm": per_arm,
        "slowest_arm": slowest,
        "projected_total_seconds": projections,
        "target_seconds": target,
        "hard_cap_seconds": hard,
        "chosen_horizon": horizon,
        "reason": reason,
        "note": "pilot only measured speed; formal runs re-initialise from seed 0",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both", choices=("checks", "pilot", "both"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    config = C.load_config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("checks", "both"):
        result = run_checks(config, args.device)
        C.write_json(out_dir / "checks.json", result)
        print(f"GSRO_CHECKS all_passed={result['all_passed']}", flush=True)
        for name, entry in result["checks"].items():
            print(f"GSRO_CHECK {name} passed={entry.get('passed')}", flush=True)
    if args.mode in ("pilot", "both"):
        pilot = run_pilot(config, args.device)
        C.write_json(out_dir / "pilot.json", pilot)
        print(
            f"GSRO_PILOT horizon={pilot['chosen_horizon']} "
            f"projections_seconds={pilot['projected_total_seconds']} reason={pilot['reason']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
