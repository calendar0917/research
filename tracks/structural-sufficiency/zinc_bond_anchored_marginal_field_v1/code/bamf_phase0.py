"""Phase-0 CPU gates for BAMF-v1.

No GPU, no official valid, no official test, no Locked read.  The script
produces ``analysis/phase0_cpu.json`` and exits non-zero if any hard gate fails.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_DCC = _SUFF / "zinc_dual_channel_composer_v1" / "code"
_DCB = _SUFF / "zinc_dccl_bottleneck_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DCC), str(_DCB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import bamf_inference as B  # noqa: E402
import bamf_models as BM  # noqa: E402
import bamf_runner as R  # noqa: E402
import dcb_train as DT  # noqa: E402
import dcc_models as DCC  # noqa: E402

torch.set_default_dtype(torch.float32)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def _state_equal(a, b) -> bool:
    if set(a) != set(b):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def _ridge_r2(Xtr, Ytr, Xte, Yte, alpha=1.0):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)
    Yte = np.asarray(Yte, dtype=np.float64)
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    ym = Ytr.mean(0)
    Yc = Ytr - ym
    A = Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1])
    w = np.linalg.solve(A, Ztr.T @ Yc)
    pred = Zte @ w + ym
    sse = ((Yte - pred) ** 2).sum(0)
    sst = ((Yte - Yte.mean(0)) ** 2).sum(0)
    sst[sst < 1e-12] = 1.0
    return 1.0 - sse / sst


def _batch_features(model, collate, cache, rows, device, arm, plan=None):
    b = T1._to_device(collate(cache, rows, "BAMF", valid=True), device)
    with torch.no_grad():
        a, bb, w = model.encoders(b)
        Wg, Vg, n_b = model.group_to_bonds(w, b)
        Wl, Vl, _ = model.local_grouping(w, b)
        log_phi = torch.tanh(model.atom_potential(a)).reshape(-1, BM.CHANNELS, BM.STATES)
        log_psi = B.edge_log_potential(
            torch.tanh(model.bond_potential(bb)).reshape(-1, BM.CHANNELS, 3)
        )
        if arm == "G":
            _, em = B.exact_marginals(log_phi, log_psi, b["bamf_plan"])
        else:
            _, em = B.independent_marginals(log_phi, log_psi, b["bond_index"])
        f = B.marginal_features(em)
        local = torch.cat(
            [bb, Wl, Vl, torch.log1p(n_b).unsqueeze(1), b["bond_feat"]], dim=1
        )
    return {
        "features": f.detach().cpu().numpy(),
        "local": local.detach().cpu().numpy(),
        "bond_batch": b["bond_batch"].detach().cpu().numpy(),
        "bond_index": b["bond_index"].detach().cpu().numpy(),
        "marginal": em.detach().cpu().numpy(),
        "n_bonds": int(em.shape[0]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--out", default=str(_ROUND / "analysis" / "phase0_cpu.json"))
    args = p.parse_args(argv)
    device = torch.device(args.device)
    t_start = time.perf_counter()
    checks: dict = {}
    failures: list[str] = []

    split = DT.split_rows()
    train_cache = D.load_cache("train")
    norm = DT.fit_only_norm(train_cache, split["fit8k"])
    y = np.asarray(train_cache["labels"], dtype=np.float64)

    # 1. topology / induced width on the real corpus
    plan = R.build_plan(train_cache)
    topo = B.topological_checks(plan)
    width_counts = [int((plan.widths == w).sum()) for w in range(int(plan.max_width) + 1)]
    checks["topology"] = {
        **topo,
        "width_counts": width_counts,
        "plan_sha256": R.plan_hash(plan),
        "build_seconds": None,
        "note": "width recomputed by min-fill on all 10000 train graphs in this run",
    }
    if not topo["all_valid"] or plan.max_width > 3 or int(plan.max_clique_size) > 4:
        failures.append("topology_or_width")

    # 2. parameter audit and G/I initial-state identity
    t0 = time.perf_counter()
    g_model, g_info = BM.build_initial_state(norm, 0, "G")
    i_model, i_info = BM.build_initial_state(norm, 0, "I")
    checks["parameters"] = {
        "G": g_info["parameters"],
        "I": i_info["parameters"],
        "breakdown": g_info["parameters_breakdown"],
        "limit": 500_000,
        "parameter_audit": {
            "named_sum": int(sum(p.numel() for p in g_model.parameters())),
            "tensors": sum(1 for _ in g_model.parameters()),
            "all_require_grad": all(p.requires_grad for p in g_model.parameters()),
        },
    }
    checks["initial_state"] = {
        "G_sha256": g_info["initial_state_sha256"],
        "I_sha256": i_info["initial_state_sha256"],
        "identical": _state_equal(C.cpu_state(g_model), C.cpu_state(i_model)),
        "dcc_initial_state_sha256": g_info["dcc_initial_state_sha256"],
        "dcc_parameters": g_info["dcc_parameter_count"],
    }
    if g_info["parameters"] != 497_093 or i_info["parameters"] != 497_093:
        failures.append("parameter_count")
    if not checks["initial_state"]["identical"]:
        failures.append("G_I_initial_state_not_identical")

    # 3. step-0 replay against the trained historical DCC-L checkpoint
    ck_path = (_SUFF / "zinc_dccl_bottleneck_v1" / "runs" / "fit8k_s0" / "best_monitor.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    dcc = DCC.DualChannel(norm)
    dcc.load_state_dict(ck["state_dict"])
    dcc.eval()
    rows0 = np.arange(128, dtype=np.int64)
    ap = np.asarray(train_cache["atom_ptr"])
    bp = np.asarray(train_cache["bond_ptr"])
    plan0 = plan.select(rows0, ap[rows0 + 1] - ap[rows0], bp[rows0 + 1] - bp[rows0])
    batch0 = T1.collate(train_cache, rows0, "P3N")
    with torch.no_grad():
        y_dcc = dcc(batch0)
    replay = {}
    for arm in ("G", "I"):
        model = BM.BAMFModel(norm, arm=arm)
        model.load_dcc_state(ck["state_dict"])
        model.eval()
        if arm == "G":
            model.set_plan(plan)
        b = dict(batch0)
        if arm == "G":
            b["bamf_plan"] = plan0
        with torch.no_grad():
            pred = model(b)
        replay[arm] = float((pred - y_dcc).abs().max())
    checks["step0_replay"] = {
        "max_abs_diff_G_vs_DCC": replay["G"],
        "max_abs_diff_I_vs_DCC": replay["I"],
        "tolerance": 1e-6,
        "checkpoint_sha256": C.sha256_file(ck_path),
        "dcc_monitor_mae_checkpoint": float(ck["monitor_mae"]),
    }
    if max(replay.values()) > 1e-6:
        failures.append("step0_replay")

    # 4. train/eval mode identical forward
    model = g_model
    model.set_plan(copy.deepcopy(plan).to(device))
    collate = R.make_collate("G", plan, device, {})
    b = T1._to_device(collate(train_cache, rows0, "BAMF", valid=True), device)
    model.train()
    with torch.no_grad():
        p_train = model(b).detach().clone()
    model.eval()
    with torch.no_grad():
        p_eval = model(b).detach().clone()
    checks["train_eval_identical"] = {
        "max_abs_diff": float((p_train - p_eval).abs().max()),
        "dropout": 0.0,
    }
    if checks["train_eval_identical"]["max_abs_diff"] > 1e-7:
        failures.append("train_eval_forward_diff")

    # 5. first optimizer steps: nonzero gradients into the new potential nets
    model = g_model
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(101)
    order = torch.randperm(len(split["fit8k"]), generator=gen).numpy()
    grad_steps = []
    for step in range(5):
        batch_rows = split["fit8k"][order[step * 128:(step + 1) * 128]]
        b = T1._to_device(collate(train_cache, batch_rows, "BAMF", valid=False), device)
        target = torch.as_tensor(y[batch_rows], dtype=torch.float32, device=device)
        pred = model(b)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        ga = model.atom_potential.weight.grad
        gb = model.bond_potential.weight.grad
        grad_steps.append({
            "zero_based_step": int(step),
            "atom_potential_grad_max": float(ga.abs().max()) if ga is not None else None,
            "bond_potential_grad_max": float(gb.abs().max()) if gb is not None else None,
            "new_local_columns_grad_max": float(
                model.local_composer.lins[0].weight.grad[:, 296:].abs().max()
            ),
            "loss": float(loss.detach()),
        })
        opt.step()
    first_nonzero = next(
        (e["zero_based_step"] for e in grad_steps
         if (e["atom_potential_grad_max"] or 0.0) > 0.0
         and (e["bond_potential_grad_max"] or 0.0) > 0.0),
        None,
    )
    checks["gradient_audit"] = {
        "steps": grad_steps,
        "first_step_with_nonzero_potential_grads": first_nonzero,
        "note": "both the reader's new columns and the local composer's new columns are zero at init, so potentials first receive gradient on the third forward (0-based step 2).",
    }
    if first_nonzero is None or first_nonzero > 3:
        failures.append("potential_gradients_not_reached")

    # 6. six-node chain: distant node affects G but not I at edge (0,1)
    torch.set_default_dtype(torch.float64)
    n = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    chain_plan = B.build_global_plan([(n, edges)], smax=8)
    chain_bp = chain_plan.select([0], [n], [len(edges)])
    torch.manual_seed(0)
    phi = torch.randn(n, 4, 2) * 0.8
    psi = B.edge_log_potential(torch.randn(len(edges), 4, 3) * 0.8)
    _, em_g = B.exact_marginals(phi, psi, chain_bp)
    _, em_i = B.independent_marginals(phi, psi, torch.tensor(edges))
    phi2 = phi.clone()
    phi2[5, 0, 0] += 5.0
    _, em_g2 = B.exact_marginals(phi2, psi, chain_bp)
    _, em_i2 = B.independent_marginals(phi2, psi, torch.tensor(edges))
    checks["six_node_chain"] = {
        "edge_0_1_G_max_abs_change": float((em_g2[0] - em_g[0]).abs().max()),
        "edge_0_1_I_max_abs_change": float((em_i2[0] - em_i[0]).abs().max()),
        "edge_4_5_G_max_abs_change": float((em_g2[4] - em_g[4]).abs().max()),
        "edge_4_5_I_max_abs_change": float((em_i2[4] - em_i[4]).abs().max()),
        "note": "change is phi[5] only; edge (0,1) is five hops away",
    }
    if not (checks["six_node_chain"]["edge_0_1_G_max_abs_change"] > 0.0
            and checks["six_node_chain"]["edge_0_1_I_max_abs_change"] == 0.0):
        failures.append("six_node_nonlocality")

    # 7. real Fit8k G/I difference statistics and redundancy vs DCC local input
    torch.set_default_dtype(torch.float32)
    model = g_model
    model.set_plan(copy.deepcopy(plan).to(device))
    model.eval()
    collate = R.make_collate("G", plan, device, {})
    n_rows = int(min(args.rows, len(split["fit8k"])))
    stats_rows = split["fit8k"][:n_rows]
    fg_parts, fi_parts, local_parts, mb_parts = [], [], [], []
    for start in range(0, n_rows, 128):
        rows = stats_rows[start:start + 128]
        fg = _batch_features(model, collate, train_cache, rows, device, "G", plan)
        fi = _batch_features(model, collate, train_cache, rows, device, "I", plan)
        fg_parts.append(fg["features"])
        fi_parts.append(fi["features"])
        local_parts.append(fg["local"])
        mb_parts.append(fg["bond_batch"])
    FG = np.concatenate(fg_parts, 0)
    FI = np.concatenate(fi_parts, 0)
    LOCAL = np.concatenate(local_parts, 0)
    BATCH = np.concatenate(mb_parts, 0)
    diff = FG - FI
    per_col = []
    for j in range(FG.shape[1]):
        d = np.abs(diff[:, j])
        per_col.append({
            "column": int(j),
            "mean_abs": float(d.mean()),
            "p95_abs": float(np.quantile(d, 0.95)),
            "max_abs": float(d.max()),
            "fraction_gt_1e_6": float((d > 1e-6).mean()),
            "fraction_gt_1e_3": float((d > 1e-3).mean()),
        })
    # molecule-disjoint train/test split for redundancy
    uniq = np.unique(BATCH)
    rng = np.random.default_rng(2024)
    rng.shuffle(uniq)
    ntr = len(uniq) // 2
    tr_set = set(int(v) for v in uniq[:ntr])
    te_set = set(int(v) for v in uniq[ntr:])
    tr_mask = np.array([int(v) in tr_set for v in BATCH])
    te_mask = ~tr_mask
    r2_g = _ridge_r2(LOCAL[tr_mask], FG[tr_mask], LOCAL[te_mask], FG[te_mask])
    r2_d = _ridge_r2(LOCAL[tr_mask], diff[tr_mask], LOCAL[te_mask], diff[te_mask])
    checks["real_fit8k_operator"] = {
        "rows": int(n_rows),
        "bonds": int(FG.shape[0]),
        "per_column_G_minus_I": per_col,
        "all_feature_mean_abs": float(np.abs(diff).mean()),
        "all_feature_max_abs": float(np.abs(diff).max()),
        "fraction_gt_1e_6": float((np.abs(diff) > 1e-6).mean()),
        "redundancy_ridge_R2": {
            "predictors": "DCC local input [b, Wl, Vl, log1p(n_b), bond_feat] (296 dims)",
            "G_features_test_R2": r2_g.tolist(),
            "G_minus_I_test_R2": r2_d.tolist(),
            "G_features_mean_R2": float(np.mean(r2_g)),
            "G_minus_I_mean_R2": float(np.mean(r2_d)),
            "alpha": 1.0,
            "split": "molecule-disjoint half/half of the evaluated rows",
        },
    }

    checks["environment"] = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_loaded": False,
    }
    checks["status"] = "PASS" if not failures else "FAIL"
    checks["failures"] = failures
    checks["wall_seconds"] = float(time.perf_counter() - t_start)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(checks), indent=2) + "\n")
    print(json.dumps(_jsonable({k: checks[k] for k in (
        "status", "failures", "topology", "parameters", "initial_state",
        "step0_replay", "train_eval_identical", "gradient_audit",
        "six_node_chain", "real_fit8k_operator", "wall_seconds")}), indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
