"""Read-only mechanism audit of the frozen Full (SCALE-FULL seed0) ZINC soup.

This script never trains, never touches the official test split, and never
writes to any historical result directory.  It only:

  * rebuilds the frozen ``LatentScaleSEM108`` soup from its published sha256,
  * replays eval-mode predictions on official train (10000) / valid (1000),
  * fits the train-median output bias (train only) and reports raw + calibrated,
  * dumps per-graph residuals and descriptive error-budget tables,
  * measures per-branch activation scale / effective rank / dead fractions,
  * measures the two dictionaries (upstream structural vs task latent) health.

Run:
  uv run python tracks/ksvd/results/zinc_full_bottleneck_audit_v1/audit_full.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

OUT = REPO_ROOT / "tracks/ksvd/results/zinc_full_bottleneck_audit_v1"
CKPT = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt"
EXPECTED_SHA = "17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb"
SEED = 0
THREADS = 8
BATCH = 128


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_model():
    subspace = base._load_parent_subspace()
    dictionary = base._dictionary_tensor()
    model = sc.build_scale_model(dictionary, SEED, subspace, sc.FULL, scale_seed=0)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = {k: v.float() for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, state, subspace


@torch.no_grad()
def eval_predictions(model, data, mask):
    loader = p1run.p1.make_env_loader(data, BATCH, False, SEED)
    preds, ys = [], []
    for batch in loader:
        preds.append(model(batch, mask=mask).view(-1).cpu())
        ys.append(batch.y.view(-1).cpu())
    return torch.cat(preds).numpy().astype(np.float64), torch.cat(ys).numpy().astype(np.float64)


@torch.no_grad()
def branch_stats(model, data, mask, n_graphs_sample=256):
    """Occurrence-level activation scale for the main branches, graph-weighted."""
    loader = p1run.p1.make_env_loader(data[:n_graphs_sample], BATCH, False, SEED)
    acc = {k: [] for k in [
        "node_slot", "edge_slot", "node_out", "edge_out", "fused", "E", "coord_res",
        "pair_value", "unary", "relation_readout",
    ]}
    for batch in loader:
        captured = {}

        def _hook(name):
            def fn(_m, _inp, out):
                captured[name] = out.detach()
            return fn

        hs = [
            model.node_encoder.register_forward_hook(_hook("node_out")),
            model.edge_encoder.register_forward_hook(_hook("edge_out")),
            model.fusion.register_forward_hook(_hook("fused")),
            model.local_dictionary_bridge.register_forward_hook(_hook("E")),
            model.pair_encoder.register_forward_hook(_hook("pair_value")),
        ]
        # node/edge slot tensors are built inside _environment_from_parts; capture via monkeypatch
        orig_node_slots = model._node_factor_tensors if hasattr(model, "_node_factor_tensors") else None

        out = model(batch, mask=mask, return_aux=True)
        _pred, aux = out
        for h in hs:
            h.remove()
        # recompute slots explicitly
        coord = aux["coord"]
        # node slots
        q = F.one_hot(batch.dict_atom, num_classes=cm.p2.ATOM_CATEGORIES).to(coord.dtype)
        occ = batch.env_occ_node.to(coord.device)
        c = coord[occ]
        qc = q[occ]
        u = (c @ model.W_A_S) * (qc @ model.W_A_C) / (cm.p2.D_A ** 0.5)
        flat = torch.zeros((int(coord.shape[0]) * cm.p2.N_SHELLS, cm.p2.D_A), dtype=u.dtype)
        flat.index_add_(0, batch.env_occ_root * cm.p2.N_SHELLS + batch.env_occ_shell, u)
        node_slots = flat.view(int(coord.shape[0]), cm.p2.N_SHELLS, cm.p2.D_A)
        # edge slots
        cu = coord[batch.env_bond_u]
        cv = coord[batch.env_bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        b = F.one_hot(batch.env_bond_type, num_classes=cm.p2.BOND_CATEGORIES).to(coord.dtype)
        ue = (g @ model.W_E_S) * (b @ model.W_E_C) / (model.config.d_e ** 0.5)
        fe = torch.zeros((int(coord.shape[0]) * cm.p2.SHELLPAIR_CLASSES, model.config.d_e), dtype=ue.dtype)
        fe.index_add_(0, batch.env_bond_root * cm.p2.SHELLPAIR_CLASSES + batch.env_bond_shellpair, ue)
        edge_slots = fe.view(int(coord.shape[0]), cm.p2.SHELLPAIR_CLASSES, model.config.d_e)

        acc["node_slot"].append(node_slots.reshape(-1).numpy())
        acc["edge_slot"].append(edge_slots.reshape(-1).numpy())
        acc["coord_res"].append(aux["coord"][:, model.common_dim:].numpy())
        for key in ["node_out", "edge_out", "fused", "E", "pair_value"]:
            if key in captured:
                acc[key].append(captured[key].reshape(-1).numpy())
        acc["unary"].append(aux["unary"].numpy())
        acc["relation_readout"].append(aux["relation_readout"].numpy())

    def _summ(name):
        x = np.concatenate(acc[name]) if acc[name] else np.zeros(1)
        return {
            "name": name,
            "n": int(x.size),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "absmean": float(np.abs(x).mean()),
            "absmax": float(np.abs(x).max()) if x.size else 0.0,
            "zero_frac": float((x == 0).mean()),
            "nan": bool(np.isnan(x).any()),
        }

    return {k: _summ(k) for k in acc}


def effective_rank(matrix: np.ndarray) -> dict:
    """Graph-weighted block matrix -> normalized singular spectrum / participation."""
    x = np.asarray(matrix, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        return {"n": int(x.shape[0])}
    s = np.linalg.svd(x, compute_uv=False)
    s2 = s ** 2
    total = s2.sum()
    if total <= 0:
        return {"n": int(x.shape[0]), "stable_rank": 0.0}
    p = s2 / total
    return {
        "n": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "stable_rank": float(total / (s2.max() + 1e-30)),
        "participation_ratio": float(1.0 / (p ** 2).sum()),
        "top1_fraction": float(p[0]),
        "top5_fraction": float(p[:5].sum()),
        "n_singular_gt_1e-9": int((s > 1e-9).sum()),
    }


def main():
    assert _sha256(CKPT) == EXPECTED_SHA, "soup checkpoint sha mismatch"
    torch.set_num_threads(THREADS)
    OUT.mkdir(parents=True, exist_ok=True)

    model, state, subspace = build_model()
    mask = cm.C6_MASK
    print("loading splits ...", flush=True)
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    print("eval valid ...", flush=True)
    pv, yv = eval_predictions(model, valid_data, mask)
    print("eval train ...", flush=True)
    pt, yt = eval_predictions(model, train_data, mask)

    bias = float(np.median(yt - pt))
    report = {
        "checkpoint_sha256": EXPECTED_SHA,
        "subspace_kind": subspace.kind,
        "official_test_loaded": False,
        "split_fingerprint": "58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a",
        "raw": {
            "train_mae_evalmode": float(np.abs(yt - pt).mean()),
            "valid_mae_evalmode": float(np.abs(yv - pv).mean()),
        },
        "bias_fit_train_median": bias,
        "calibrated": {
            "train_mae": float(np.abs(yt - (pt + bias)).mean()),
            "valid_mae": float(np.abs(yv - (pv + bias)).mean()),
        },
    }

    # per-graph valid table
    res = yv - pv
    cal_res = yv - (pv + bias)
    order = np.argsort(-np.abs(cal_res))
    absr = np.abs(cal_res)
    total_mae = absr.mean()
    n = absr.size

    def share(k):
        return float(absr[order[:k]].sum() / (absr.sum() + 1e-30))

    report["valid_error_budget_calibrated"] = {
        "mae": float(total_mae),
        "median_abs": float(np.median(absr)),
        "mean_signed": float(cal_res.mean()),
        "max_abs": float(absr.max()),
        "max_id": int(order[0]),
        "top1_share_of_sum": share(1),
        "top5_share": share(max(1, int(round(0.05 * n)))),
        "top10_share": share(max(1, int(round(0.10 * n)))),
        "central90_mae": float(np.abs(np.sort(absr)[: int(round(0.90 * n))]).mean()),
        "central95_mae": float(np.abs(np.sort(absr)[: int(round(0.95 * n))]).mean()),
    }
    report["valid_error_budget_raw"] = {
        "mae": float(np.abs(res).mean()),
        "median_abs": float(np.median(np.abs(res))),
        "mean_signed": float(res.mean()),
    }
    # id172
    i172 = int(np.where(yv < -20.0)[0][0]) if (yv < -20).any() else -1
    if i172 >= 0:
        report["id172"] = {
            "index": i172,
            "y": float(yv[i172]),
            "p_raw": float(pv[i172]),
            "abs_err_raw": float(abs(res[i172])),
            "p_cal": float(pv[i172] + bias),
            "abs_err_cal": float(abs(cal_res[i172])),
            "share_of_cal_sum": float(absr[i172] / (absr.sum() + 1e-30)),
            "mae_contribution": float(absr[i172] / n),
            "mae_excluding": float(absr.sum() - absr[i172]) / (n - 1),
        }
    # size grouping from R count block (log1p number of occurrences)
    Rv = np.load(OUT.parent / "zinc_dictionary_real_data_handoff/valid_reader.npz")["R"] if (
        OUT.parent / "zinc_dictionary_real_data_handoff/valid_reader.npz"
    ).exists() else None
    if Rv is not None:
        counts = np.expm1(Rv[:, 2 * sc.FULL.d]).astype(np.int64)
        qs = np.quantile(counts, [0.25, 0.5, 0.75])
        groups = np.digitize(counts, qs)
        report["valid_size_quartile"] = {}
        for g in range(4):
            m = groups == g
            if m.sum() == 0:
                continue
            report["valid_size_quartile"][str(g)] = {
                "n": int(m.sum()),
                "node_count_median": float(np.median(counts[m])),
                "mae_cal": float(absr[m].mean()),
                "mae_raw": float(np.abs(res[m]).mean()),
                "mean_target": float(yv[m].mean()),
            }
        # target quantiles from train-fitted boundaries
        tq = np.quantile(yt, [0.1, 0.25, 0.5, 0.75, 0.9])
        tg = np.digitize(yv, tq)
        report["valid_target_quantile"] = {}
        for g in range(len(tq) + 1):
            m = tg == g
            if m.sum() == 0:
                continue
            report["valid_target_quantile"][str(g)] = {
                "n": int(m.sum()),
                "y_range": [float(yv[m].min()), float(yv[m].max())],
                "mae_cal": float(absr[m].mean()),
                "mae_raw": float(np.abs(res[m]).mean()),
            }

    # branch statistics
    print("branch stats ...", flush=True)
    report["branches"] = branch_stats(model, valid_data, mask)
    # graph-level effective rank of the reader input R (from handoff, valid)
    if Rv is not None:
        report["reader_input_R_spectrum_valid"] = effective_rank(Rv)
        report["reader_input_R_blocks_valid"] = {}
        d = sc.FULL.d
        p = sc.FULL.p
        blocks = {
            "unary_first": (0, d),
            "unary_second": (d, 2 * d),
            "unary_count": (2 * d, 2 * d + 1),
            "pair_all": (2 * d + 1, 2 * d + 1 + 5 * (2 * p + 1)),
            "global": (2 * d + 1 + 5 * (2 * p + 1), 2 * d + 1 + 5 * (2 * p + 1) + 32),
            "topology": (2 * d + 1 + 5 * (2 * p + 1) + 32, 814),
        }
        for name, (lo, hi) in blocks.items():
            report["reader_input_R_blocks_valid"][name] = effective_rank(Rv[:, lo:hi])

    # dictionaries (upstream structural)
    D = model.D.detach().double()
    coln = D.norm(dim=0)
    report["structural_dictionary"] = {
        "K": int(D.shape[1]),
        "phi_dim": int(D.shape[0]),
        "col_norm_mean": float(coln.mean()),
        "col_norm_min": float(coln.min()),
        "col_norm_max": float(coln.max()),
    }
    # task dictionary health on valid
    with torch.no_grad():
        loader = p1run.p1.make_env_loader(valid_data, BATCH, False, SEED)
        batch = next(iter(loader))
        h = sc._capture_bridge_input(model, batch, mask)
        _E, aux = model.local_dictionary_bridge(h, return_aux=True)
        alpha = aux["alpha"].double()
        nz = (alpha != 0).sum(dim=1)
        E = _E
        x = aux["normalized"].double()
        Dbar = model.local_dictionary_bridge.normalized_dictionary().double()
        recon = (x - alpha @ Dbar.t())
        rel = (recon ** 2).sum(1) / ((x ** 2).sum(1) + 1e-12)
        report["task_dictionary"] = {
            "d": int(model.local_dictionary_bridge.dim),
            "k": int(model.local_dictionary_bridge.n_atoms),
            "mean_nonzero": float(nz.float().mean()),
            "mean_nonzero_fraction": float(nz.float().mean() / model.local_dictionary_bridge.n_atoms),
            "l0_p50": float(nz.float().median()),
            "l0_p95": float(nz.float().quantile(0.95)),
            "alpha_absmean": float(alpha.abs().mean()),
            "alpha_absmax": float(alpha.abs().max()),
            "x_absmean": float(x.abs().mean()),
            "recon_rel_mean": float(rel.mean()),
            "E_absmean": float(E.abs().mean()),
            "E_std_per_dim_mean": float(E.std(dim=0).mean()),
            "E_std_per_dim_min": float(E.std(dim=0).min()),
            "E_std_per_dim_max": float(E.std(dim=0).max()),
        }
        # per-dim scale of alpha and E
        report["task_dictionary"]["alpha_std_per_dim_mean"] = float(alpha.std(dim=0).mean())

    # reader affine replay check (H2 -> prediction) using handoff reader
    reader_npz = OUT.parent / "zinc_dictionary_real_data_handoff/reader.npz"
    if reader_npz.exists() and Rv is not None:
        rn = np.load(reader_npz)
        W1, b1 = rn["W1"], rn["b1"]
        W2, b2 = rn["W2"], rn["b2"]
        W3, b3 = rn["W3"], rn["b3"]
        h1 = np.maximum(Rv @ W1.T + b1, 0.0)
        h2 = np.maximum(h1 @ W2.T + b2, 0.0)
        pred_replay = (h2 @ W3.T + b3).reshape(-1)
        p_base = np.load(OUT.parent / "zinc_dictionary_real_data_handoff/valid_reader.npz")["p_base"]
        report["reader_replay_max_abs_diff"] = float(np.abs(pred_replay - p_base).max())
        # H2 graph-level spectrum
        report["H2_spectrum_valid"] = effective_rank(h2)
        report["H2_dead_units_valid"] = float((h2 == 0).mean())

    (OUT / "audit_full.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # per-graph valid CSV
    import csv

    with open(OUT / "valid_per_graph.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "graph_id", "y", "pred_raw", "train_fitted_bias",
                    "pred_calibrated", "abs_error", "fixed_group"])
        for i in range(n):
            w.writerow(["valid", i, f"{yv[i]:.6f}", f"{pv[i]:.6f}", f"{bias:.6f}",
                        f"{pv[i]+bias:.6f}", f"{absr[i]:.6f}", i % 5])
    print(json.dumps(report["raw"], indent=2), flush=True)
    print("calibrated", report["calibrated"], flush=True)
    print("bias", bias, flush=True)
    print("budget", json.dumps(report["valid_error_budget_calibrated"], indent=2), flush=True)
    print("node_slot", report["branches"]["node_slot"], flush=True)
    print("edge_slot", report["branches"]["edge_slot"], flush=True)
    print("node_out", report["branches"]["node_out"], flush=True)
    print("edge_out", report["branches"]["edge_out"], flush=True)
    print("task_dict", json.dumps(report["task_dictionary"], indent=2), flush=True)


if __name__ == "__main__":
    main()