"""FGCR-v0 minimum engineering verification.

Runs the six required pre-training checks plus the single-belief-pass check and
a small initial non-degeneracy report.  All checks are exact/brute-force or
real-data gradient checks; this is not a scientific audit.

Usage:
    uv run python tracks/structural-sufficiency/full_graph_cavity_response_pilot_v0/code/fgcr_checks.py
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

import fgcr_common as C
import fgcr_data as D
import fgcr_inference as I
import fgcr_model as M


# ---------------------------------------------------------------------------
# brute-force reference
# ---------------------------------------------------------------------------


def bruteforce_logz_logodds(u: np.ndarray, j: np.ndarray,
                            edges: Sequence[tuple[int, int]]) -> tuple[float, np.ndarray]:
    """Dense enumeration of all 2^n states in float64.

    ``u`` [n,2], ``j`` [m,2,2] symmetric.  Returns logZ and the exact posterior
    log-odds log p(s_v=1)/p(s_v=0) for every node.
    """
    u = np.asarray(u, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    n = int(u.shape[0])
    states = np.arange(1 << n, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(n, dtype=np.int64)) & 1)
    energy = np.zeros(1 << n, dtype=np.float64)
    for v in range(n):
        energy = energy + u[v][bits[:, v]]
    for e, (a, b) in enumerate(edges):
        energy = energy + j[e][bits[:, a], bits[:, b]]
    logz = float(np.logaddexp.reduce(energy))
    logodds = np.zeros(n, dtype=np.float64)
    for v in range(n):
        one = bits[:, v] == 1
        logodds[v] = float(np.logaddexp.reduce(energy[one])
                           - np.logaddexp.reduce(energy[~one]))
    return logz, logodds


def exact_logz_logodds(u: np.ndarray, j: np.ndarray,
                       edges: Sequence[tuple[int, int]], *,
                       smax: int) -> tuple[float, np.ndarray]:
    n = int(u.shape[0])
    plan = I.build_corpus_plan([(n, list(edges))], smax=int(smax))
    bp = plan.select([0], [n], [len(edges)]).to("cpu")
    log_phi = torch.as_tensor(np.asarray(u, dtype=np.float64))[:, None, :]
    log_psi = torch.as_tensor(np.asarray(j, dtype=np.float64))[:, None, :, :]
    with torch.no_grad():
        resp = I.exact_responses(log_phi, log_psi, bp)
    logz = float(resp["logz"][0, 0].item())
    logodds = (resp["logw"][:, 0, 1] - resp["logw"][:, 0, 0]).cpu().numpy()
    return logz, logodds


def _symmetric_j(rng: np.random.Generator, m: int) -> np.ndarray:
    raw = rng.normal(size=(m, 3))
    return np.stack([
        np.stack([raw[:, 0], raw[:, 1]], axis=-1),
        np.stack([raw[:, 1], raw[:, 2]], axis=-1),
    ], axis=-2)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cast_batch(batch: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    for key in ("graph_n_atoms", "graph_n_bonds", "y_std", "y_raw"):
        if key in batch and torch.is_tensor(batch[key]) and batch[key].is_floating_point():
            batch[key] = batch[key].to(dtype)
    return batch


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().double().pow(2).sum().item())
    return float(np.sqrt(total))


def _finite_all_grads(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def _predict_single_graph(model: M.FGCRModel, graph: dict[str, Any], *,
                          smax: int = 4) -> dict[str, np.ndarray]:
    n = int(graph["n_atoms"])
    edges = list(zip(np.asarray(graph["bond_u"]).tolist(),
                     np.asarray(graph["bond_v"]).tolist()))
    plan = I.build_corpus_plan([(n, edges)], smax=int(smax))
    batch = D.collate_graphs([graph], plan, "cpu")
    _cast_batch(batch, torch.float64)
    model.eval()
    with torch.no_grad():
        out = model(batch, return_aux=True)
    return {
        "yhat_std": out["yhat_std"].double().cpu().numpy(),
        "H": out["H"].double().cpu().numpy(),
        "eta": out["eta"].double().cpu().numpy(),
        "logz": out["logz"].double().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_bruteforce() -> dict[str, Any]:
    cases: list[tuple[str, int, list[tuple[int, int]]]] = [
        ("isolated_1", 1, []),
        ("single_edge_2", 2, [(0, 1)]),
        ("path_3", 3, [(0, 1), (1, 2)]),
        ("chain_4", 4, [(0, 1), (1, 2), (2, 3)]),
        ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("branch_5", 5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
        ("cycle_5", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
        ("multi_component_6", 6,
         [(0, 1), (1, 2), (2, 0), (3, 4)]),          # triangle + edge + isolated 5
        ("two_components_7", 7,
         [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6)]),  # square + path + isolated
        ("tree_7", 7,
         [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
        ("cyclic_7", 7,
         [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 2), (0, 6)]),
    ]
    rng = np.random.default_rng(20270115)
    rows: list[dict[str, Any]] = []
    worst_abs = 0.0
    worst_logz_scale = 1.0
    for name, n, edges in cases:
        u = rng.normal(size=(n, 2))
        j = _symmetric_j(rng, len(edges))
        bf_logz, bf_odds = bruteforce_logz_logodds(u, j, edges)
        ex_logz, ex_odds = exact_logz_logodds(u, j, edges, smax=n)
        logz_err = abs(ex_logz - bf_logz)
        odds_err = float(np.max(np.abs(ex_odds - bf_odds))) if n else 0.0
        scale = max(1.0, abs(bf_logz))
        passed = logz_err <= 1e-8 * scale and odds_err <= 1e-8 * max(1.0, float(np.max(np.abs(bf_odds)))) if n else logz_err <= 1e-8 * scale
        worst_abs = max(worst_abs, logz_err, odds_err)
        worst_logz_scale = max(worst_logz_scale, scale)
        rows.append({
            "name": name, "n": n, "m": len(edges),
            "logz_exact": ex_logz, "logz_bruteforce": bf_logz,
            "logz_abs_error": logz_err,
            "logodds_max_abs_error": odds_err,
            "pass": bool(passed),
        })
    return {
        "pass": all(r["pass"] for r in rows),
        "tolerance": "1e-8 (absolute for log-odds; relative scale for logZ)",
        "max_abs_error": worst_abs,
        "cases": rows,
    }


def check_zero_potential() -> dict[str, Any]:
    # Direct inference on a cycle with J identically zero.
    rng = np.random.default_rng(7)
    n, edges = 4, [(0, 1), (1, 2), (2, 3), (3, 0)]
    u = rng.normal(size=(n, 2))
    j = np.zeros((len(edges), 2, 2))
    logz, logodds = exact_logz_logodds(u, j, edges, smax=n)
    unary_lse = float(np.logaddexp(u[:, 0], u[:, 1]).sum())
    H = logz - unary_lse
    eta = logodds - (u[:, 1] - u[:, 0])
    direct = {"H_max_abs": abs(H), "eta_max_abs": float(np.max(np.abs(eta)))}

    # Real train molecules through the full model (float64) with the bond MLP
    # output layer zeroed, so every J is exactly 0.
    bundle = D.load_bundle()
    rows = bundle.train_idx[:8]
    graphs_all = D.graphs_for_rows(bundle, range(bundle.n_graphs))
    pairs = [(int(g["n_atoms"]), list(zip(g["bond_u"].tolist(), g["bond_v"].tolist())))
             for g in graphs_all]
    plan = I.build_corpus_plan(pairs, smax=4)
    model, _ = M.build_model(bundle.atom_dims, bundle.bond_dims, 0)
    model.double()
    with torch.no_grad():
        model.bond_mlp[-1].weight.zero_()
        model.bond_mlp[-1].bias.zero_()
    batch = D.collate_rows(bundle, rows, plan, "cpu")
    _cast_batch(batch, torch.float64)
    model.eval()
    with torch.no_grad():
        out = model(batch, return_aux=True)
        j_max = float(out["j"].abs().max().item())
        H_max = float(out["H"].abs().max().item())
        eta_max = float(out["eta"].abs().max().item())
    tol = 1e-8
    passed = direct["H_max_abs"] <= tol and direct["eta_max_abs"] <= tol \
        and j_max == 0.0 and H_max <= tol and eta_max <= tol
    return {
        "pass": bool(passed),
        "tolerance": tol,
        "direct_cycle": direct,
        "real_batch_model": {
            "n_molecules": int(len(rows)), "j_max_abs": j_max,
            "H_max_abs": H_max, "eta_max_abs": eta_max,
        },
    }


def _permuted_graph(graph: dict[str, Any], perm: np.ndarray,
                    edge_order: np.ndarray | None = None) -> dict[str, Any]:
    n = int(graph["n_atoms"])
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)
    u = inv[np.asarray(graph["bond_u"], dtype=np.int64)]
    v = inv[np.asarray(graph["bond_v"], dtype=np.int64)]
    uu = np.minimum(u, v)
    vv = np.maximum(u, v)
    feat = np.asarray(graph["bond_feat"], dtype=np.int64)
    order = np.lexsort((vv, uu))
    if edge_order is not None:
        order = order[np.asarray(edge_order, dtype=np.int64)]
    return {
        "row": int(graph["row"]),
        "n_atoms": n,
        "n_bonds": int(len(order)),
        "atom_feat": np.asarray(graph["atom_feat"], dtype=np.int64)[perm],
        "bond_u": uu[order],
        "bond_v": vv[order],
        "bond_feat": feat[order],
    }


def _reversed_graph(graph: dict[str, Any], order: np.ndarray | None = None) -> dict[str, Any]:
    u = np.asarray(graph["bond_u"], dtype=np.int64).copy()
    v = np.asarray(graph["bond_v"], dtype=np.int64).copy()
    feat = np.asarray(graph["bond_feat"], dtype=np.int64).copy()
    if order is not None:
        u, v, feat = u[order], v[order], feat[order]
    return {
        "row": int(graph["row"]),
        "n_atoms": int(graph["n_atoms"]),
        "n_bonds": int(len(u)),
        "atom_feat": np.asarray(graph["atom_feat"], dtype=np.int64),
        "bond_u": v,              # reversed endpoint order, deliberately unsorted
        "bond_v": u,
        "bond_feat": feat,
    }


def _reordered_graph(graph: dict[str, Any], order: np.ndarray) -> dict[str, Any]:
    return {
        "row": int(graph["row"]),
        "n_atoms": int(graph["n_atoms"]),
        "n_bonds": int(len(order)),
        "atom_feat": np.asarray(graph["atom_feat"], dtype=np.int64),
        "bond_u": np.asarray(graph["bond_u"], dtype=np.int64)[order],
        "bond_v": np.asarray(graph["bond_v"], dtype=np.int64)[order],
        "bond_feat": np.asarray(graph["bond_feat"], dtype=np.int64)[order],
    }


def check_invariance() -> dict[str, Any]:
    bundle = D.load_bundle()
    # A real molecule small enough for repeated exact plans.
    candidates = [int(r) for r in bundle.train_idx
                  if 8 <= int(bundle.graph_n_atoms[r]) <= 16]
    row = candidates[0]
    graph = D.graphs_for_rows(bundle, [row])[0]
    model, _ = M.build_model(bundle.atom_dims, bundle.bond_dims, 0)
    model.double()
    base = _predict_single_graph(model, graph)

    n = int(graph["n_atoms"])
    rng = np.random.default_rng(4242)
    perm = rng.permutation(n)
    permuted = _predict_single_graph(model, _permuted_graph(graph, perm))

    reverse = _predict_single_graph(model, _reversed_graph(graph))

    edge_order = rng.permutation(int(graph["n_bonds"]))
    reordered = _predict_single_graph(model, _reordered_graph(graph, edge_order))

    def maxdiff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    checks = {
        "atom_permutation": {
            "yhat": maxdiff(base["yhat_std"], permuted["yhat_std"]),
            "H": maxdiff(base["H"], permuted["H"]),
            "eta_permuted": maxdiff(base["eta"], permuted["eta"][np.argsort(perm)]),
        },
        "bond_direction_reversal": {
            "yhat": maxdiff(base["yhat_std"], reverse["yhat_std"]),
            "H": maxdiff(base["H"], reverse["H"]),
            "eta": maxdiff(base["eta"], reverse["eta"]),
        },
        "bond_storage_order": {
            "yhat": maxdiff(base["yhat_std"], reordered["yhat_std"]),
            "H": maxdiff(base["H"], reordered["H"]),
            "eta": maxdiff(base["eta"], reordered["eta"]),
        },
    }
    tol = 1e-7
    passed = all(v <= tol for group in checks.values() for v in group.values())
    return {
        "pass": bool(passed), "tolerance": tol, "row": int(row),
        "n_atoms": n, "n_bonds": int(graph["n_bonds"]),
        "max_diffs": checks,
    }


def check_train_eval_operator() -> dict[str, Any]:
    bundle = D.load_bundle()
    rows = bundle.train_idx[:16]
    graphs_all = D.graphs_for_rows(bundle, range(bundle.n_graphs))
    pairs = [(int(g["n_atoms"]), list(zip(g["bond_u"].tolist(), g["bond_v"].tolist())))
             for g in graphs_all]
    plan = I.build_corpus_plan(pairs, smax=4)
    model, _ = M.build_model(bundle.atom_dims, bundle.bond_dims, 0)
    model.eval()
    batch = D.collate_rows(bundle, rows, plan, "cpu")
    with torch.no_grad():
        eval_out = model(batch, return_aux=True)
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()   # keep the operator identical; dropout is the only mode branch
    with torch.no_grad():
        train_out = model(batch, return_aux=True)
    diffs = {
        "H": float((eval_out["H"] - train_out["H"]).abs().max().item()),
        "eta": float((eval_out["eta"] - train_out["eta"]).abs().max().item()),
        "logz": float((eval_out["logz"] - train_out["logz"]).abs().max().item()),
        "yhat": float((eval_out["yhat_std"] - train_out["yhat_std"]).abs().max().item()),
    }
    passed = all(v == 0.0 for v in diffs.values())
    return {"pass": bool(passed), "max_diffs": diffs,
            "note": "forward has no training-mode branch other than dropout"}


def check_real_batch_gradients() -> dict[str, Any]:
    bundle = D.load_bundle()
    rows = bundle.train_idx[:64]
    graphs_all = D.graphs_for_rows(bundle, range(bundle.n_graphs))
    pairs = [(int(g["n_atoms"]), list(zip(g["bond_u"].tolist(), g["bond_v"].tolist())))
             for g in graphs_all]
    plan = I.build_corpus_plan(pairs, smax=4)
    model, info = M.build_model(bundle.atom_dims, bundle.bond_dims, 0)
    model.train()
    batch = D.collate_rows(bundle, rows, plan, "cpu")
    out = model(batch, return_aux=True)
    out["u"].retain_grad()
    out["j"].retain_grad()
    loss = torch.mean((out["yhat_std"] - batch["y_std"]) ** 2)
    loss.backward()

    groups = {
        "atom_emb": model.atom_emb,
        "bond_emb": model.bond_emb,
        "atom_mlp": model.atom_mlp,
        "bond_mlp": model.bond_mlp,
        "eta_norm": model.eta_norm,
        "sentence_mlp": model.sentence_mlp,
        "readout_mlp": model.readout_mlp,
    }
    grads = {name: _module_grad_norm(module) for name, module in groups.items()}
    grads["beta"] = float(abs(model.beta.grad.item())) if model.beta.grad is not None else 0.0
    u_grad = float(out["u"].grad.double().pow(2).sum().sqrt().item())
    j_grad = float(out["j"].grad.double().pow(2).sum().sqrt().item())
    finite = bool(np.isfinite(loss.item())) and _finite_all_grads(model) \
        and bool(np.isfinite(u_grad)) and bool(np.isfinite(j_grad))
    nonzero = all(v > 0.0 for v in grads.values()) and u_grad > 0.0 and j_grad > 0.0

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    opt.step()
    deltas = {name: float((p.detach() - before[name]).abs().max().item())
              for name, p in model.named_parameters()}
    step_moved = all(v > 0.0 for v in deltas.values())
    passed = bool(finite and nonzero and step_moved)
    return {
        "pass": passed,
        "loss": float(loss.item()),
        "grad_norms": grads,
        "u_grad_norm": u_grad,
        "j_grad_norm": j_grad,
        "finite": finite,
        "nonzero": nonzero,
        "optimizer_step_moved_all_parameters": step_moved,
        "min_parameter_delta": float(min(deltas.values())),
        "parameters": info["parameters"],
    }


def check_data() -> dict[str, Any]:
    bundle = D.load_bundle()
    train = bundle.train_idx
    labels = bundle.labels.astype(np.float64)
    mean = float(labels[train].mean())
    std = float(labels[train].std(ddof=0))
    scaler_ok = (abs(mean - bundle.scaler["mean"]) <= 1e-12
                 and abs(std - bundle.scaler["std"]) <= 1e-12)
    y_std = bundle.labels_std[train].astype(np.float64)
    std_ok = abs(float(y_std.mean())) <= 1e-6 and abs(float(y_std.std(ddof=0)) - 1.0) <= 1e-6
    rng = np.random.default_rng(11)
    pred_std = rng.normal(size=37)
    target = rng.normal(size=37) * 1.7 - 0.2
    metrics = C.metrics_raw(pred_std, target, bundle.scaler)
    pred_raw = pred_std * bundle.scaler["std"] + bundle.scaler["mean"]
    manual = {
        "rmse": float(np.sqrt(np.mean((pred_raw - target) ** 2))),
        "mae": float(np.mean(np.abs(pred_raw - target))),
    }
    inversion_ok = abs(metrics["rmse"] - manual["rmse"]) <= 1e-12 \
        and abs(metrics["mae"] - manual["mae"]) <= 1e-12
    # Optional cross-check against the official OGB evaluator (same RMSE
    # definition, but an independent implementation).
    try:
        sq_data, _ = C.import_sq_data()
        official = sq_data.official_metric(
            C.DATASET, pred_raw, target, unit="raw")
        official_ok = (
            abs(official["raw_rmse"] - metrics["rmse"]) <= 1e-12
            and abs(official["raw_mae"] - metrics["mae"]) <= 1e-12
        )
    except Exception as exc:  # noqa: BLE001 - provenance, not a scientific gate
        official = {"error": repr(exc)}
        official_ok = None
    split_stats = {}
    for split in ("train", "valid", "test"):
        rows = bundle.split_rows(split)
        values = labels[rows]
        split_stats[split] = {
            "n": int(rows.size),
            "label_min": float(values.min()),
            "label_max": float(values.max()),
            "label_mean": float(values.mean()),
            "label_std_ddof0": float(values.std(ddof=0)),
        }
    passed = bool(scaler_ok and std_ok and inversion_ok
                  and (official_ok is not False))
    return {
        "pass": passed,
        "audit": bundle.audit,
        "scaler_matches_train_only": scaler_ok,
        "train_standardized_mean_std": {"mean": float(y_std.mean()),
                                         "std_ddof0": float(y_std.std(ddof=0))},
        "metric_inversion": {"computed": metrics, "manual": manual,
                             "pass": inversion_ok},
        "official_ogb_metric": {"computed": official,
                                "matches_rmse_definition": official_ok},
        "split_label_stats": split_stats,
    }


def check_single_belief_and_nondegeneracy() -> dict[str, Any]:
    bundle = D.load_bundle()
    rows = bundle.train_idx[:32]
    graphs_all = D.graphs_for_rows(bundle, range(bundle.n_graphs))
    pairs = [(int(g["n_atoms"]), list(zip(g["bond_u"].tolist(), g["bond_v"].tolist())))
             for g in graphs_all]
    plan = I.build_corpus_plan(pairs, smax=4)
    model, _ = M.build_model(bundle.atom_dims, bundle.bond_dims, 0)
    model.eval()
    batch = D.collate_rows(bundle, rows, plan, "cpu")
    I.BELIEF_CALLS["count"] = 0
    with torch.no_grad():
        out = model(batch, return_aux=True)
    calls = int(I.BELIEF_CALLS["count"])
    H = out["H"].double().cpu().numpy()
    eta = out["eta"].double().cpu().numpy()
    finite = bool(np.isfinite(H).all() and np.isfinite(eta).all())
    nondegenerate = (float(np.std(H)) > 1e-6 and float(np.std(eta)) > 1e-6
                     and float(np.max(np.abs(H))) > 1e-4
                     and float(np.max(np.abs(eta))) > 1e-4)
    return {
        "pass": bool(calls == 1 and finite and nondegenerate),
        "belief_calls_for_one_forward": calls,
        "H": {"mean": float(H.mean()), "std": float(H.std()),
              "min": float(H.min()), "max": float(H.max()),
              "max_abs": float(np.max(np.abs(H)))},
        "eta": {"mean": float(eta.mean()), "std": float(eta.std()),
                "min": float(eta.min()), "max": float(eta.max()),
                "max_abs": float(np.max(np.abs(eta))),
                "fraction_abs_gt_1e-3": float(np.mean(np.abs(eta) > 1e-3))},
        "finite": finite,
        "nondegenerate": nondegenerate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "correctness.json"))
    args = parser.parse_args()
    t0 = time.perf_counter()
    C.seed_everything(0)

    print("[1/7] brute-force logZ / posterior log-odds", flush=True)
    bruteforce = check_bruteforce()
    print(f"      pass={bruteforce['pass']} max_abs_error={bruteforce['max_abs_error']:.3e}",
          flush=True)

    print("[2/7] zero edge potentials", flush=True)
    zero = check_zero_potential()
    print(f"      pass={zero['pass']}", flush=True)

    print("[3/7] relabel / bond-direction / bond-order invariance", flush=True)
    invariance = check_invariance()
    print(f"      pass={invariance['pass']} max={max(v for g in invariance['max_diffs'].values() for v in g.values()):.3e}",
          flush=True)

    print("[4/7] train/eval operator identity", flush=True)
    operator = check_train_eval_operator()
    print(f"      pass={operator['pass']}", flush=True)

    print("[5/7] real Train batch gradients + optimizer step", flush=True)
    gradients = check_real_batch_gradients()
    print(f"      pass={gradients['pass']} loss={gradients['loss']:.6f}", flush=True)

    print("[6/7] data / split / standardization / metric inversion", flush=True)
    data = check_data()
    print(f"      pass={data['pass']}", flush=True)

    print("[7/7] single belief pass + initial non-degeneracy", flush=True)
    single = check_single_belief_and_nondegeneracy()
    print(f"      pass={single['pass']} belief_calls={single['belief_calls_for_one_forward']}",
          flush=True)

    checks = {
        "bruteforce": bruteforce,
        "zero_potential": zero,
        "invariance": invariance,
        "train_eval_operator": operator,
        "real_batch_gradients": gradients,
        "data": data,
        "single_belief_and_nondegeneracy": single,
    }
    status = "PASS" if all(v["pass"] for v in checks.values()) else "FAIL"
    payload = {
        "status": status,
        "seconds": float(time.perf_counter() - t0),
        "checks": checks,
        "provenance": {
            "commit": C.git_commit(),
            "dirty": C.git_dirty(),
            "environment": C.environment_snapshot(),
            "config": C.load_config(),
            "config_sha256": C.sha256_file(C.CONFIG_PATH),
            "vendored_inference_sha256": C.sha256_file(C.CODE_DIR / "bamf_inference.py"),
        },
    }
    C.write_json(args.out, payload)
    print(f"STATUS={status} wrote {args.out}", flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())