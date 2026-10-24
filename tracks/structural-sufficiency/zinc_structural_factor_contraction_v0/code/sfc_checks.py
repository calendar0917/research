"""T1 — correctness gate for SFC-v0.

Every check is written against an *independent* reference:

* ``log_partition_dense`` enumerates all ``2^n`` joint states in float64; it
  shares only the factor tables with the elimination path, never the elimination
  logic or the descriptor algebra.
* gradient checks compare autograd (through the elimination) against float64
  central finite differences of that dense reference.
* invariance checks re-run the full pipeline on a relabelled graph.
* the marginal checks compare elimination-derived marginals (autograd of
  ``logZ``) against dense enumeration marginals.

Checks (all must pass before any T2/T3 number is quoted):

 1. ``exact``        elimination ``logZ`` == dense enumeration on random typed
                     graphs (n in 1..8, trees, rings, complete, multi-component,
                     isolated nodes), max |delta| <= 1e-10.
 2. ``relabel``      a random vertex permutation leaves ``logZ``, ``H`` and ``T``
                     unchanged.
 3. ``grad_A``       d logZ / dA  (autograd, elimination) == central finite
                     differences of the dense reference.
 4. ``grad_S``       d logZ / dS  likewise.
 5. ``marginals``    elimination marginals == dense marginals; they sum to one.
 6. ``zero_coupling`` K == 1 (tanh(S) == 0)  =>  H == 0 and T == 0, every graph.
 7. ``tree_zero``    every forest has T == 0 exactly (with and without a node
                     field); rings at the same parameter point have T != 0.
 8. ``correspond``   graphs with identical atom-category multiset and identical
                     bond-type multiset but different real connections give
                     different H and different T.
 9. ``order``        every elimination order gives the same ``logZ``; only the
                     reported width depends on the order.
10. ``determinism``  identical inputs give bit-identical outputs on repeat.
11. ``subgraph``     induced-subgraph elimination == dense enumeration on it.
12. ``saturation``   the documented regime boundary: at extreme parameter scale
                     tanh saturates and T -> 0.  Recorded, not hidden.

Run:
  uv run python tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/code/sfc_checks.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sfc_core as SC  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
N_ATOM = 6        # small category space keeps the checks fast; 28 used elsewhere
N_BOND = 4        # slots 0..3, slot 0 unused
TOL_EXACT = 1e-10
TOL_GRAD = 1e-6


# ---------------------------------------------------------------------------
# graph builders
# ---------------------------------------------------------------------------


def g_path(n, atom=None, bond=None):
    atom = np.zeros(n, np.int64) if atom is None else atom
    bond = np.ones(max(n - 1, 0), np.int64) if bond is None else bond
    return SC.TypedGraph(n, atom, np.arange(n - 1), np.arange(1, n), bond)


def g_ring(n, atom=None, bond=None):
    atom = np.zeros(n, np.int64) if atom is None else atom
    bond = np.ones(n, np.int64) if bond is None else bond
    return SC.TypedGraph(n, atom, np.arange(n), (np.arange(n) + 1) % n, bond)


def g_complete(n, atom=None):
    atom = np.zeros(n, np.int64) if atom is None else atom
    eu, ev = [], []
    for i in range(n):
        for j in range(i + 1, n):
            eu.append(i)
            ev.append(j)
    return SC.TypedGraph(n, atom, np.asarray(eu), np.asarray(ev),
                         np.ones(len(eu), np.int64))


def g_random(n, rng, atom=None, p=0.35, bond=None):
    atom = rng.integers(0, N_ATOM, n).astype(np.int64) if atom is None else atom
    eu, ev, eb = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                eu.append(i)
                ev.append(j)
                eb.append(int(rng.integers(1, N_BOND)))
    if bond is not None:
        eb = list(bond)
    return SC.TypedGraph(n, atom, np.asarray(eu, np.int64), np.asarray(ev, np.int64),
                         np.asarray(eb, np.int64))


def g_disjoint(n, rng):
    """Several components + isolated nodes."""
    eu, ev, eb, atom = [], [], [], []
    off = 0
    while off < n:
        k = int(min(int(rng.integers(1, 4)), n - off))
        for i in range(off, off + k):
            atom.append(int(rng.integers(0, N_ATOM)))
        for i in range(off, off + k - 1):
            eu.append(i)
            ev.append(i + 1)
            eb.append(int(rng.integers(1, N_BOND)))
        off += k
        if rng.random() < 0.4 and off < n:
            atom.append(int(rng.integers(0, N_ATOM)))
            off += 1
    m = min(len(atom), n)
    return SC.TypedGraph(m, np.asarray(atom[:m], np.int64), np.asarray(eu, np.int64),
                         np.asarray(ev, np.int64), np.asarray(eb, np.int64))


def random_params(C, seed, dtype=torch.float64, scale=1.0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    A = (torch.randn(N_ATOM, C, 2, generator=g, dtype=dtype) * scale).requires_grad_(requires_grad)
    S = (torch.randn(N_BOND, C, 3, generator=g, dtype=dtype) * scale).requires_grad_(requires_grad)
    return A, S


def permute_graph(g: SC.TypedGraph, perm: np.ndarray) -> SC.TypedGraph:
    """Relabel: the new index of old vertex ``v`` is ``perm[v]``."""
    atom = np.zeros_like(g.atom)
    atom[perm] = g.atom
    return SC.TypedGraph(g.n, atom, perm[g.edge_u], perm[g.edge_v], g.edge_b.copy())


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_exact() -> dict:
    rng = np.random.default_rng(0)
    cases = []
    for n in range(1, 9):
        cases.append((f"path{n}", g_path(n)))
        if n >= 3:
            cases.append((f"ring{n}", g_ring(n)))
    for n in range(3, 8):
        cases.append((f"complete{n}", g_complete(n)))
    for k in range(14):
        cases.append((f"random{k}", g_random(int(rng.integers(2, 9)), rng)))
    for k in range(6):
        cases.append((f"disjoint{k}", g_disjoint(int(rng.integers(2, 9)), rng)))

    worst, per_case = 0.0, {}
    for C, scale in ((1, 1.0), (3, 1.0), (5, 2.5)):
        A, S = random_params(C, seed=11 + C, scale=scale)
        for name, g in cases:
            out = SC.sfc_forward(g, A, S)
            ref = SC.log_partition_dense(g, A, S)
            err = float((out.logZ - ref).abs().max())
            per_case[f"{name}_C{C}"] = err
            worst = max(worst, err)
    return {"max_abs_err": worst, "n_cases": len(per_case), "tol": TOL_EXACT,
            "pass": bool(worst <= TOL_EXACT),
            "worst_cases": sorted(per_case.items(), key=lambda kv: -kv[1])[:5]}


def check_relabel() -> dict:
    rng = np.random.default_rng(1)
    worst = 0.0
    n_tests = 0
    for C in (1, 4):
        A, S = random_params(C, seed=100 + C)
        for _ in range(25):
            g = g_random(int(rng.integers(3, 9)), rng)
            base = SC.sfc_forward(g, A, S)
            for _ in range(4):
                perm = rng.permutation(g.n)
                hp = SC.sfc_forward(permute_graph(g, perm), A, S)
                worst = max(worst, float((base.logZ - hp.logZ).abs().max()),
                            float((base.H - hp.H).abs().max()),
                            float((base.T - hp.T).abs().max()))
                n_tests += 1
    return {"max_abs_diff": worst, "n_relabel": n_tests, "pass": bool(worst <= 1e-12)}


def check_grad(param: str) -> dict:
    """autograd (elimination) vs float64 central differences of the dense reference."""
    C = 3
    A, S = random_params(C, seed=7, requires_grad=True)
    rng = np.random.default_rng(2)
    graphs = [g_path(4), g_ring(4), g_ring(5), g_complete(4),
              g_random(6, rng), g_random(7, rng), g_random(5, rng, p=0.6)]
    target = A if param == "A" else S
    total = sum(SC.sfc_forward(g, A, S).logZ.sum() for g in graphs)
    total.backward()
    autograd = target.grad.detach().clone()
    target.grad = None

    eps = 1e-6
    with torch.no_grad():
        num = torch.zeros_like(target)
        for idx in np.ndindex(*target.shape):
            old = float(target[idx])
            target[idx] = old + eps
            fp = sum(float(SC.log_partition_dense(g, A, S).sum()) for g in graphs)
            target[idx] = old - eps
            fm = sum(float(SC.log_partition_dense(g, A, S).sum()) for g in graphs)
            target[idx] = old
            num[idx] = (fp - fm) / (2 * eps)

    denom = num.abs().clamp_min(1e-8)
    rel = float(((autograd - num).abs() / denom).max())
    abs_err = float((autograd - num).abs().max())
    return {"param": param, "max_abs_err": abs_err, "max_rel_err": rel,
            "fd_scale": float(num.abs().max()), "n_entries": int(num.numel()),
            "pass": bool(rel <= TOL_GRAD and abs_err <= 1e-4)}


def check_marginals() -> dict:
    """Elimination marginals (autograd of logZ) == dense marginals; they normalise."""
    rng = np.random.default_rng(8)
    worst_node, worst_edge, worst_sum, worst_lz = 0.0, 0.0, 0.0, 0.0
    for k in range(6):
        g = g_random(int(rng.integers(2, 9)), rng, p=0.4)
        A, S = random_params(3, seed=600 + k)
        out = SC.sfc_forward(g, A, S)
        logZ, node_p, edge_p = SC.marginals_dense(g, A, S)
        worst_node = max(worst_node, float((out.node_p - node_p).abs().max()))
        if g.m:
            worst_edge = max(worst_edge, float((out.edge_p - edge_p).abs().max()))
            worst_sum = max(worst_sum,
                            float((out.edge_p.sum(dim=(-1, -2)) - 1).abs().max()))
        worst_sum = max(worst_sum, float((out.node_p.sum(-1) - 1).abs().max()))
        worst_lz = max(worst_lz, float((out.logZ - logZ).abs().max()))
    return {"max_node_marginal_err": worst_node, "max_edge_marginal_err": worst_edge,
            "max_normalisation_err": worst_sum, "max_logZ_err": worst_lz,
            "pass": bool(max(worst_node, worst_edge, worst_sum, worst_lz) <= 1e-10)}


def check_zero_coupling() -> dict:
    rng = np.random.default_rng(3)
    worst_H, worst_T = 0.0, 0.0
    for C in (1, 4):
        A = torch.randn(N_ATOM, C, 2, dtype=torch.float64)
        S = torch.zeros(N_BOND, C, 3, dtype=torch.float64)      # tanh(0)=0 -> K=1
        for _ in range(20):
            g = g_random(int(rng.integers(1, 9)), rng, p=0.5)
            out = SC.sfc_forward(g, A, S)
            worst_H = max(worst_H, float(out.H.abs().max()))
            worst_T = max(worst_T, float(out.T.abs().max()))
    return {"max_abs_H": worst_H, "max_abs_T": worst_T,
            "pass": bool(worst_H <= 1e-12 and worst_T <= 1e-12)}


def check_tree_zero() -> dict:
    """Every forest has T == 0 exactly; rings at the same parameters do not."""
    rng = np.random.default_rng(4)
    res = {}
    for C, scale in ((1, 1.0), (4, 1.5)):
        A, S = random_params(C, seed=200 + C, scale=scale)
        A_nofield = torch.zeros_like(A)        # no node field at all
        forests, rings = [], []
        for n in range(2, 10):
            forests.append(g_path(n, atom=rng.integers(0, N_ATOM, n).astype(np.int64),
                                  bond=rng.integers(1, N_BOND, max(n - 1, 0)).astype(np.int64)))
        for _ in range(12):
            n = int(rng.integers(2, 11))
            eu = np.array([int(rng.integers(0, i)) for i in range(1, n)])
            ev = np.arange(1, n)
            forests.append(SC.TypedGraph(
                n, rng.integers(0, N_ATOM, n).astype(np.int64), eu, ev,
                rng.integers(1, N_BOND, n - 1).astype(np.int64)))
        for n in range(3, 10):
            rings.append(g_ring(n, atom=rng.integers(0, N_ATOM, n).astype(np.int64),
                                bond=rng.integers(1, N_BOND, n).astype(np.int64)))

        def forest_stats(graphs, AA):
            ts, hs = [], []
            for g in graphs:
                assert g.is_forest(), (g.n, g.m)
                o = SC.sfc_forward(g, AA, S)
                ts.append(float(o.T.abs().max()))
                hs.append(float(o.H.abs().max()))
            return max(ts), max(hs)

        t_f, h_f = forest_stats(forests, A)
        t_f_nf, h_f_nf = forest_stats(forests, A_nofield)
        # a single channel may vanish by coincidence, so the requirement is that
        # the per-graph T *vector* is non-zero: max over channels
        ring_T = [float(SC.sfc_forward(g, A, S).T.abs().max()) for g in rings]
        ring_T_nf = [float(SC.sfc_forward(g, A_nofield, S).T.abs().max()) for g in rings]
        ring_H = [float(SC.sfc_forward(g, A, S).H.abs().max()) for g in rings]
        res[f"C{C}"] = {
            "n_forests": len(forests), "n_rings": len(rings),
            "forest_max_abs_T_node_field": t_f,
            "forest_max_abs_T_no_node_field": t_f_nf,
            "forest_max_abs_H_node_field": h_f,
            "forest_max_abs_H_no_node_field": h_f_nf,
            "ring_min_over_graphs_abs_T_node_field": min(ring_T),
            "ring_min_over_graphs_abs_T_no_node_field": min(ring_T_nf),
            "ring_max_abs_H_node_field": max(ring_H),
        }
    ok = all(v["forest_max_abs_T_node_field"] <= 1e-10
             and v["forest_max_abs_T_no_node_field"] <= 1e-10
             and v["ring_min_over_graphs_abs_T_node_field"] > 1e-8
             and v["ring_min_over_graphs_abs_T_no_node_field"] > 1e-8
             for v in res.values())
    return {"per_C": res, "pass": bool(ok)}


def check_correspondence() -> dict:
    """Same atom multiset, same bond multiset, different real connections."""
    atoms = np.array([0, 1, 2, 0, 1, 2], np.int64)      # identical multiset
    bonds = np.ones(6, np.int64)                        # identical bond multiset
    variants = {
        "two_triangles": SC.TypedGraph(
            6, atoms, np.array([0, 1, 2, 3, 4, 5]), np.array([1, 2, 0, 4, 5, 3]), bonds),
        "hexagon": SC.TypedGraph(
            6, atoms, np.array([0, 1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5, 0]), bonds),
        "pentagon_pendant": SC.TypedGraph(
            6, atoms, np.array([0, 1, 2, 3, 4, 0]), np.array([1, 2, 3, 4, 0, 5]), bonds),
        "k4_plus_2_isolated": SC.TypedGraph(
            6, atoms, np.array([0, 0, 0, 1, 1, 2]),
            np.array([1, 2, 3, 2, 3, 3]), bonds),
        "square_two_pendants": SC.TypedGraph(
            6, atoms, np.array([0, 1, 2, 3, 0, 2]),
            np.array([1, 2, 3, 0, 4, 5]), bonds),
    }
    for name, g in variants.items():
        assert g.n == 6 and g.m == 6, (name, g.n, g.m)
        assert sorted(g.atom.tolist()) == sorted(atoms.tolist()), name
        assert sorted(g.edge_b.tolist()) == sorted(bonds.tolist()), name

    out = {}
    for C, scale in ((2, 1.0), (4, 1.0)):
        A, S = random_params(C, seed=300 + C, scale=scale)
        rows = {}
        for name, g in variants.items():
            o = SC.sfc_forward(g, A, S)
            rows[name] = {"H": o.H.tolist(), "T": o.T.tolist(), "logZ": o.logZ.tolist(),
                          "cyclomatic": g.cyclomatic()}
        keys = list(rows)
        vecs = {k: np.array(rows[k]["H"] + rows[k]["T"]) for k in keys}
        pairs = list(itertools.combinations(keys, 2))
        d_full = min(float(np.linalg.norm(vecs[a] - vecs[b])) for a, b in pairs)
        d_T = min(float(np.linalg.norm(np.array(rows[a]["T"]) - np.array(rows[b]["T"])))
                  for a, b in pairs)
        mag_full = max(float(np.linalg.norm(v)) for v in vecs.values())
        mag_T = max(float(np.linalg.norm(np.array(rows[k]["T"]))) for k in keys)
        n_distinct = len({tuple(np.round(v, 10)) for v in (rows[k]["T"] for k in keys)})
        out[f"C{C}"] = {
            "rows": rows,
            "min_pairwise_l2_full": d_full,
            "min_pairwise_l2_T_only": d_T,
            "max_norm_full": mag_full,
            "max_norm_T_only": mag_T,
            "relative_separation_full": d_full / mag_full,
            "relative_separation_T_only": d_T / mag_T,
            "n_distinct_T": n_distinct,
            "n_pairs": len(pairs),
        }
    ok = all(v["relative_separation_full"] > 1e-3
             and v["relative_separation_T_only"] > 1e-3
             and v["n_distinct_T"] == 5 for v in out.values())
    return {"per_C": out, "pass": bool(ok)}


def check_order() -> dict:
    rng = np.random.default_rng(5)
    worst = 0.0
    widths: dict[str, list[int]] = {}
    n_cmp = 0
    for k in range(8):
        n = int(rng.integers(4, 9))
        g = g_random(n, rng, p=0.45)
        A, S = random_params(2, seed=400 + k)
        vals = []
        for oname, ofn in SC.ORDERS.items():
            o = SC.sfc_forward(g, A, S, order=ofn(g.n, g.edges()))
            vals.append(o.logZ)
            widths.setdefault(oname, []).append(o.stats["induced_width"])
        ref = SC.log_partition_dense(g, A, S)
        for v in vals:
            worst = max(worst, float((v - ref).abs().max()))
            n_cmp += 1
        worst = max(worst, float((vals[0] - vals[1]).abs().max()))
    return {"max_abs_diff": worst, "n_comparisons": n_cmp,
            "mean_width": {k: float(np.mean(v)) for k, v in widths.items()},
            "pass": bool(worst <= TOL_EXACT)}


def check_determinism() -> dict:
    rng = np.random.default_rng(6)
    g = g_random(9, rng, p=0.4)
    A, S = random_params(5, seed=7)
    a = SC.sfc_forward(g, A, S)
    b = SC.sfc_forward(g, A, S)
    same = (torch.equal(a.logZ, b.logZ) and torch.equal(a.H, b.H)
            and torch.equal(a.T, b.T))
    return {"bit_identical": bool(same), "pass": bool(same)}


def check_subgraph() -> dict:
    rng = np.random.default_rng(7)
    worst = 0.0
    n_sub = 0
    for k in range(6):
        g = g_random(int(rng.integers(6, 11)), rng, p=0.3)
        A, S = random_params(3, seed=500 + k)
        for _ in range(5):
            k2 = int(rng.integers(2, g.n + 1))
            nodes = sorted(rng.choice(g.n, size=k2, replace=False).tolist())
            sub = g.subgraph(nodes)
            o = SC.sfc_forward(sub, A, S)
            r = SC.log_partition_dense(sub, A, S)
            worst = max(worst, float((o.logZ - r).abs().max()))
            n_sub += 1
    return {"max_abs_err": worst, "n_subgraphs": n_sub, "pass": bool(worst <= TOL_EXACT)}


def check_saturation() -> dict:
    """Documented regime boundary: at extreme scale tanh saturates and T -> 0."""
    C = 4
    rows = {}
    for scale in (0.5, 1.0, 2.0, 4.0, 8.0):
        g = torch.Generator().manual_seed(900)
        A = torch.randn(N_ATOM, C, 2, generator=g, dtype=torch.float64) * scale
        S = torch.randn(N_BOND, C, 3, generator=g, dtype=torch.float64) * scale
        o = SC.sfc_forward(g_ring(6), A, S)
        rows[str(scale)] = {"max_abs_T": float(o.T.abs().max()),
                            "max_abs_H": float(o.H.abs().max()),
                            "mean_abs_tanh": float(torch.tanh(S).abs().mean())}
    return {"per_scale": rows,
            "note": "T vanishes when tanh saturates (deterministic limit); "
                    "the operator is informative only in the non-saturated regime",
            "pass": True}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ANALYSIS / "t1_correctness.json"),
                    help="output JSON (default keeps the historical file name)")
    args = ap.parse_args()
    out_path = Path(args.out)
    t0 = time.perf_counter()
    checks = {
        "exact": check_exact(),
        "relabel": check_relabel(),
        "grad_A": check_grad("A"),
        "grad_S": check_grad("S"),
        "marginals": check_marginals(),
        "zero_coupling": check_zero_coupling(),
        "tree_zero": check_tree_zero(),
        "correspondence": check_correspondence(),
        "order": check_order(),
        "determinism": check_determinism(),
        "subgraph": check_subgraph(),
        "saturation": check_saturation(),
    }
    n_pass = sum(1 for v in checks.values() if v["pass"])
    payload = {
        "protocol": "sfc-v0-cpu-audit",
        "stage": "T1-correctness",
        "n_atom_categories": N_ATOM,
        "n_bond_slots": N_BOND,
        "dtype": "float64",
        "checks": checks,
        "n_pass": n_pass,
        "n_total": len(checks),
        "all_pass": bool(n_pass == len(checks)),
        "seconds": time.perf_counter() - t0,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["pass"] for k, v in checks.items()}, indent=2))
    print(f"{n_pass}/{len(checks)} checks pass -> {out_path}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
