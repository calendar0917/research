#!/usr/bin/env python
"""A2: algebraic counterexample + frozen-model state rebinding.

A2.1 is a pure activation-level construction: two bond-state assignments with
identical sum and sum of squares (the only c_loc blocks the DCC-L reader sees)
but different correspondence-dependent statistics.  A2.2 rebinds real c_loc
rows within real train molecules and checks the frozen reader response.

Neither is a legal molecule counterfactual or a label counterfactual.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audit_common as AC  # noqa: E402
import joint_stats as JS  # noqa: E402

N_REBIND_MOLECULES = 200
REBIND_ROWS_SEED = 20261027


def _line_path_relations(n_bonds: int = 6) -> dict[str, list[tuple[int, int]]]:
    edges = [(i, i + 1) for i in range(n_bonds - 1)]
    return {"R1": edges, "R2": [], "R3": [], "R4": []}


def _correspondence_stat(x: np.ndarray, relations: dict) -> dict:
    x = np.asarray(x, dtype=np.float64)
    gram = x @ x.T
    out = {}
    for r in JS.RELATIONS:
        pairs = relations.get(r, [])
        if pairs:
            out[r] = float(np.mean([gram[a, b] for a, b in pairs]) / x.shape[1])
        else:
            out[r] = 0.0
    # adjacency weighted sum (not mean)
    pairs = relations["R1"]
    out["line_graph_adjacent_sum"] = float(sum(gram[a, b] for a, b in pairs))
    return out


def algebraic_counterexample(reader, seed: int = 20261026) -> dict:
    rng = np.random.default_rng(seed)
    # Small integer vectors: sums and sums of squares are exactly representable.
    u = rng.integers(-2, 3, size=96).astype(np.float64)
    v = rng.integers(-2, 3, size=96).astype(np.float64)
    w = rng.integers(-2, 3, size=96).astype(np.float64)
    X = np.stack([u, v, -u, -v, w, -w])
    Y = np.stack([u, -u, v, -v, w, -w])
    rel = _line_path_relations(6)
    sx, sy = X.sum(axis=0), Y.sum(axis=0)
    qx, qy = (X * X).sum(axis=0), (Y * Y).sum(axis=0)
    stat_x = _correspondence_stat(X, rel)
    stat_y = _correspondence_stat(Y, rel)
    # Independent float64 reference for the sums.
    ref_sx = np.array([sum(float(X[i, j]) for i in range(6)) for j in range(96)])
    ref_sy = np.array([sum(float(Y[i, j]) for i in range(6)) for j in range(96)])
    ref_qx = np.array([sum(float(X[i, j]) ** 2 for i in range(6)) for j in range(96)])
    ref_qy = np.array([sum(float(Y[i, j]) ** 2 for i in range(6)) for j in range(96)])

    # Build two 513-d reader inputs that differ only in the c_loc rows.
    # The first 321 columns are fixed; zl/zl2 are the pooled moments.
    base = np.zeros((1, 513), dtype=np.float32)
    base[0, :321] = rng.normal(0.0, 1.0, size=321).astype(np.float32)
    zx = base.copy()
    zy = base.copy()
    zx[0, 321:417] = sx.astype(np.float32)
    zy[0, 321:417] = sy.astype(np.float32)
    zx[0, 417:513] = qx.astype(np.float32)
    zy[0, 417:513] = qy.astype(np.float32)
    with torch.no_grad():
        px = float(reader(torch.as_tensor(zx)).view(-1)[0])
        py = float(reader(torch.as_tensor(zy)).view(-1)[0])
    return {
        "construction": {
            "n_bonds": 6,
            "line_graph": "path 0-1-2-3-4-5",
            "X": "u, v, -u, -v, w, -w",
            "Y": "u, -u, v, -v, w, -w",
            "u_v_w_dtype": "float64 small integers",
            "only_c_loc_rows_differ": True,
            "other_reader_blocks": "identical fixed 321-d prefix",
        },
        "sum_max_abs_diff": float(np.max(np.abs(sx - sy))),
        "sumsq_max_abs_diff": float(np.max(np.abs(qx - qy))),
        "independent_sum_max_abs_diff": float(max(np.max(np.abs(sx - ref_sx)), np.max(np.abs(sy - ref_sy)))),
        "independent_sumsq_max_abs_diff": float(max(np.max(np.abs(qx - ref_qx)), np.max(np.abs(qy - ref_qy)))),
        "correspondence_stat_X": stat_x,
        "correspondence_stat_Y": stat_y,
        "correspondence_stat_delta": {k: stat_x[k] - stat_y[k] for k in stat_x},
        "reader_output_X": px,
        "reader_output_Y": py,
        "reader_output_abs_diff": abs(px - py),
        "scope": "activation-level only; not a real-molecule identifiability proof",
    }


def rebinding_audit(model, cache, norm, name: str, rows: np.ndarray) -> dict:
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    out_rows = {}
    pred_delta = []
    z_delta = []
    zl_delta = []
    zl2_delta = []
    jl_delta = {r: [] for r in JS.RELATIONS}
    adj_delta = []
    variation = []
    for row in rows:
        row = int(row)
        batch = AC.collate_rows(cache, [row], device="cpu")
        with torch.no_grad():
            cap = AC.capture_forward(model, batch)
        c = cap["c_loc"].detach().cpu().numpy()  # [n_b, 96]
        n_b = c.shape[0]
        rel = JS.molecule_relations(np.asarray(cache["bond_index"])[bp[row]:bp[row + 1]],
                                    int(ap[row + 1] - ap[row]))
        perm = np.random.default_rng(np.random.SeedSequence([JS.SEED_BASE, row])).permutation(n_b)
        c_perm = c[perm]
        # natural pooled recomputation by the same index_add path
        jb = batch["junc_bond"].long()
        c_t = torch.as_tensor(c_perm, dtype=cap["c_loc"].dtype)
        zl = c_t.new_zeros((1, c_t.shape[1])).index_add_(0, torch.zeros(n_b, dtype=torch.long), c_t)
        zl2 = c_t.new_zeros((1, c_t.shape[1])).index_add_(0, torch.zeros(n_b, dtype=torch.long), c_t * c_t)
        z_real = cap["z"].clone()
        z_perm = torch.cat([cap["z_glob"], cap["a_sum"], cap["a_sq"], cap["g"], zl, zl2], dim=1)
        # exact multiset pooled inputs: copy the real zl/zl2 (mathematically identical)
        z_perm_exact = torch.cat([cap["z_glob"], cap["a_sum"], cap["a_sq"], cap["g"], cap["zl"], cap["zl2"]], dim=1)
        with torch.no_grad():
            p_real = float(model.reader(z_real).view(-1)[0])
            p_perm = float(model.reader(z_perm).view(-1)[0])
            p_exact = float(model.reader(z_perm_exact).view(-1)[0])
        jl_real = JS.j_latent(c, rel)
        jl_perm = JS.j_latent(c_perm, rel)
        st_real = _correspondence_stat(c, rel)
        st_perm = _correspondence_stat(c_perm, rel)
        pred_delta.append(abs(p_perm - p_real))
        z_delta.append(float(torch.max(torch.abs(z_perm - z_real))))
        zl_delta.append(float(torch.max(torch.abs(zl - cap["zl"]))))
        zl2_delta.append(float(torch.max(torch.abs(zl2 - cap["zl2"]))))
        for r_i, r in enumerate(JS.RELATIONS):
            jl_delta[r].append(abs(float(jl_perm[r_i] - jl_real[r_i])))
        adj_delta.append(abs(st_perm["line_graph_adjacent_sum"] - st_real["line_graph_adjacent_sum"]))
        # within-molecule variation
        h = c - c.mean(axis=0, keepdims=True)
        cov = (h.T @ h) / max(n_b - 1, 1)
        variation.append({
            "n_bonds": int(n_b),
            "pr": AC.participation_ratio(cov),
            "cv": float(np.sqrt(np.mean(np.sum(h * h, axis=1))) / max(np.sqrt(np.mean(np.sum(c * c, axis=1))), 1e-12)),
        })
    out = {
        "checkpoint": name,
        "n_molecules": int(len(rows)),
        "prediction_abs_delta": {
            "mean": float(np.mean(pred_delta)), "max": float(np.max(pred_delta)),
            "median": float(np.median(pred_delta)),
        },
        "pooled_input_abs_delta": {
            "z_max_max": float(np.max(z_delta)),
            "zl_max_max": float(np.max(zl_delta)),
            "zl2_max_max": float(np.max(zl2_delta)),
        },
        "exact_multiset_reader_abs_delta": abs(p_exact - p_real),
        "j_latent_abs_delta": {
            r: {"mean": float(np.mean(v)), "max": float(np.max(v)),
                "frac_gt_1e-6": float(np.mean(np.asarray(v) > 1e-6))}
            for r, v in jl_delta.items()
        },
        "line_graph_adjacent_inner_product_abs_delta": {
            "mean": float(np.mean(adj_delta)), "max": float(np.max(adj_delta)),
            "frac_gt_1e-6": float(np.mean(np.asarray(adj_delta) > 1e-6)),
        },
        "within_molecule_c_loc_variation": {
            "pr_median": float(np.median([v["pr"] for v in variation])),
            "pr_p10": float(np.percentile([v["pr"] for v in variation], 10)),
            "pr_p90": float(np.percentile([v["pr"] for v in variation], 90)),
            "cv_median": float(np.median([v["cv"] for v in variation])),
            "cv_p10": float(np.percentile([v["cv"] for v in variation], 10)),
            "cv_p90": float(np.percentile([v["cv"] for v in variation], 90)),
        },
        "scope": "computational dependency only; not a legal molecule or label counterfactual",
    }
    return out


def main() -> int:
    AC.install_load_guard()
    cfg = AC.load_audit_config()
    cache = AC.load_train_cache()
    norm = AC.load_norm()
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    rng = np.random.default_rng(REBIND_ROWS_SEED)
    candidate = np.where((ap[1:] - ap[:-1]) >= 6)[0]
    rows = np.sort(rng.choice(candidate, N_REBIND_MOLECULES, replace=False)).astype(np.int64)

    results = {}
    for name in ("D0", "D1"):
        model, info = AC.load_checkpoint(name, norm, device="cpu", organized_local=True)
        model.eval()
        results[name] = rebinding_audit(model, cache, norm, name, rows)
        results[name]["checkpoint"] = info
    # algebraic counterexample uses D0's reader
    model0, _ = AC.load_checkpoint("D0", norm, device="cpu", organized_local=True)
    alg = algebraic_counterexample(model0.reader)
    out = {
        "protocol": cfg["protocol_id"],
        "algebraic_counterexample": alg,
        "rebinding": results,
        "rows_seed": REBIND_ROWS_SEED,
        "rows": rows.tolist(),
    }
    AC.write_json(AC.ANALYSIS / "algebraic_counterexample.json", AC.jsonable(out))
    print(json.dumps({
        "algebraic_reader_abs_diff": alg["reader_output_abs_diff"],
        "algebraic_sum_diff": alg["sum_max_abs_diff"],
        "algebraic_sumsq_diff": alg["sumsq_max_abs_diff"],
        "algebraic_pair_delta": alg["correspondence_stat_delta"],
        "rebinding": {k: {"pred_mean_abs_delta": v["prediction_abs_delta"]["mean"],
                          "zl_max": v["pooled_input_abs_delta"]["zl_max_max"],
                          "T_R1_mean_abs_delta": v["j_latent_abs_delta"]["R1"]["mean"],
                          "T_R1_frac_change": v["j_latent_abs_delta"]["R1"]["frac_gt_1e-6"],
                          "c_loc_PR_median": v["within_molecule_c_loc_variation"]["pr_median"]}
                      for k, v in results.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
