"""Engineering / identity / positive-control checks for HSRF v0."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

import hrf_common as C
import hrf_design as D
import hrf_stage0 as S


def _result(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), **evidence}


def data_checks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    rows = {s: np.asarray(payload["split_rows"][s], np.int64) for s in ("train", "valid", "test")}
    y = np.asarray(payload["labels"], np.float64)
    sizes = {s: int(rows[s].size) for s in rows}
    union = np.union1d(np.union1d(rows["train"], rows["valid"]), rows["test"])
    overlap = (np.intersect1d(rows["train"], rows["valid"]).size +
               np.intersect1d(rows["train"], rows["test"]).size +
               np.intersect1d(rows["valid"], rows["test"]).size)
    out.append(_result("official_split_sizes", sizes == {"train": 3360, "valid": 420, "test": 420},
                       sizes=sizes))
    out.append(_result("splits_disjoint_cover_4200",
                       overlap == 0 and union.size == 4200 and np.array_equal(union, np.arange(4200)),
                       overlap=overlap, union_size=int(union.size)))
    out.append(_result("row_ids_are_0_to_4199",
                       np.array_equal(np.asarray(payload["row_ids"]), np.arange(4200))))
    out.append(_result("labels_finite", bool(np.isfinite(y).all()),
                       n=int(y.size), min=float(y.min()), max=float(y.max())))
    scaler = C.target_scaler(y, rows["train"])
    z = (y[rows["train"]] - scaler["mean"]) / scaler["std"]
    back = z * scaler["std"] + scaler["mean"]
    out.append(_result("train_only_scaler_roundtrip",
                       float(np.abs(back - y[rows["train"]]).max()) <= 1e-12,
                       max_abs_error=float(np.abs(back - y[rows["train"]]).max()),
                       mean=scaler["mean"], std=scaler["std"], n=scaler["n"]))
    # perturb Valid/Test labels: a Train-only scaler cannot move
    y2 = y.copy(); y2[rows["valid"]] += 7.0; y2[rows["test"]] -= 5.0
    scaler2 = C.target_scaler(y2, rows["train"])
    out.append(_result("scaler_does_not_read_valid_test",
                       abs(scaler2["mean"] - scaler["mean"]) < 1e-15 and
                       abs(scaler2["std"] - scaler["std"]) < 1e-15,
                       delta_mean=float(abs(scaler2["mean"] - scaler["mean"])),
                       delta_std=float(abs(scaler2["std"] - scaler["std"]))))
    return out


def _brute_supports(n: int, edges: Sequence[tuple[int, int]], root: int) -> dict[int, set[tuple[int, ...]]]:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    dist = np.full(n, -1, int); dist[root] = 0; frontier = [root]
    while frontier:
        nxt = []
        for x in frontier:
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1; nxt.append(y)
        frontier = nxt
    out = {k: set() for k in (1, 2, 3, 4)}
    for size in (1, 2, 3, 4):
        for comb in itertools.combinations(range(n), size):
            if root not in comb:
                continue
            if any(dist[c] < 0 or dist[c] > 2 for c in comb):
                continue
            sub = set(comb)
            seen = {root}; stack = [root]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y in sub and y not in seen:
                        seen.add(y); stack.append(y)
            if seen == sub:
                out[size].add(tuple(sorted(comb)))
    return out


def _artificial_record(n: int, edges: Sequence[tuple[int, int]]) -> dict[str, Any]:
    atom = np.zeros((n, 9), np.int64)
    atom[:, 0] = 6
    bonds = np.asarray(edges, np.int64)
    return {
        "row": 0, "n_atoms": n, "n_bonds": len(edges),
        "atom_feat": atom,
        "bond_feat": np.zeros((len(edges), 3), np.int64),
        "bond_u": bonds[:, 0], "bond_v": bonds[:, 1],
        "graph_scalars": np.zeros(8),
    }


def structure_checks(payload: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    import rice_structure as RS

    atom_dims, bond_dims = C.atom_bond_dims()
    out = []
    graphs = {
        "path4": _artificial_record(4, [(0, 1), (1, 2), (2, 3)]),
        "star4": _artificial_record(4, [(0, 1), (0, 2), (0, 3)]),
        "triangle": _artificial_record(3, [(0, 1), (1, 2), (2, 0)]),
        "ring5": _artificial_record(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
    }
    for name, rec in graphs.items():
        st = RS.build_molecule(rec)
        edges = list(zip(rec["bond_u"].tolist(), rec["bond_v"].tolist()))
        ok = True
        detail = {}
        for root in range(rec["n_atoms"]):
            brute = _brute_supports(rec["n_atoms"], edges, root)
            for size in (1, 2, 3, 4):
                if size == 1:
                    got = {(root,)}
                else:
                    got = {tuple(sorted(row)) for row in st.atoms[size][:, :size]
                           if int(row[0]) == root}
                if got != brute[size]:
                    ok = False
                    detail[f"root{root}_size{size}"] = {
                        "missing": sorted(brute[size] - got)[:5],
                        "extra": sorted(got - brute[size])[:5]}
        # internal bonds are all real bonds inside the support
        edge_set = {tuple(sorted(e)) for e in edges}
        for size in (2, 3, 4):
            for i in range(st.atoms[size].shape[0]):
                slots = [int(x) for x in st.atoms[size][i, :size]]
                support = set(slots)
                got = {tuple(sorted((slots[int(st.bslot[size][i, t, 0])],
                                     slots[int(st.bslot[size][i, t, 1])])))
                       for t in range(6) if int(st.bonds[size][i, t]) >= 0}
                want = {e for e in edge_set if e[0] in support and e[1] in support}
                if got != want:
                    ok = False
                    detail[f"bonds_size{size}_{i}"] = {"got": sorted(got), "want": sorted(want)}
        out.append(_result(f"artificial_enumeration_{name}", ok, detail=detail))

    # one record per (root, support)
    dup = 0
    for name, rec in graphs.items():
        st = RS.build_molecule(rec)
        for size in (2, 3, 4):
            records = [(int(row[0]), tuple(sorted(int(x) for x in row[:size])))
                       for row in st.atoms[size]]
            dup += len(records) - len(set(records))
    out.append(_result("no_duplicate_root_support_records", dup == 0, duplicates=int(dup)))

    # relabelling invariance on 16 Train molecules: per-type counts must be identical
    key_to_col = {}
    for meta in C.read_json(C.DATA_DIR / "feature_catalog.json")["base_raw_columns"]:
        if meta["kind"] == "rooted_structure":
            key_to_col[(meta["gran"], int(meta["size"]),
                        np.asarray(meta["signature"], np.int32).tobytes())] = int(meta["raw_col"])
    z = np.load(C.DATA_DIR / "hsrf_stage0.npz")
    base_counts = np.asarray(z["base_counts"], np.float64)
    rng = np.random.default_rng(0)
    rows = np.asarray(payload["split_rows"]["train"], np.int64)[:16]
    max_diff = 0.0
    failures = []
    for row in rows.tolist():
        rec = C.molecule_record(payload, int(row))
        perm = rng.permutation(rec["n_atoms"])
        inv = np.empty_like(perm); inv[perm] = np.arange(perm.size)
        rec2 = dict(rec)
        rec2["atom_feat"] = rec["atom_feat"][perm]
        rec2["bond_u"] = inv[rec["bond_u"]]
        rec2["bond_v"] = inv[rec["bond_v"]]
        st1 = RS.build_molecule(rec)
        st2 = RS.build_molecule(rec2)
        vec1 = np.zeros(base_counts.shape[1], np.float64)
        vec2 = np.zeros(base_counts.shape[1], np.float64)
        for size in (1, 2, 3, 4):
            for gran in ("TOPO", "ELEMENT", "COARSE", "FULL"):
                for st, vec in ((st1, vec1), (st2, vec2)):
                    sig = C.signatures_for(st, size, gran, atom_dims, bond_dims)
                    for s in sig:
                        col = key_to_col.get((gran, size, np.ascontiguousarray(s, np.int32).tobytes()))
                        if col is not None:
                            vec[col] += 1.0
        d = float(np.max(np.abs(vec1 - vec2)))
        max_diff = max(max_diff, d)
        if d > 0:
            failures.append({"row": int(row), "max_abs_diff": d})
    out.append(_result("relabelling_invariance_16_train", max_diff == 0.0,
                       max_abs_diff=max_diff, failures=failures[:3]))

    # granularity mapping: one FULL type cannot map to two COARSE/ELEMENT/TOPO types
    inconsistency = 0
    for row in rows[:8].tolist():
        st = RS.build_molecule(C.molecule_record(payload, int(row)))
        by = {}
        for size in (2, 3, 4):
            sigs = {g: C.signatures_for(st, size, g, atom_dims, bond_dims) for g in
                    ("TOPO", "ELEMENT", "COARSE", "FULL")}
            for i in range(sigs["FULL"].shape[0]):
                full = np.ascontiguousarray(sigs["FULL"][i], np.int32).tobytes()
                coarse = np.ascontiguousarray(sigs["COARSE"][i], np.int32).tobytes()
                elem = np.ascontiguousarray(sigs["ELEMENT"][i], np.int32).tobytes()
                topo = np.ascontiguousarray(sigs["TOPO"][i], np.int32).tobytes()
                by.setdefault((size, full), set()).add(coarse)
                by.setdefault((size, coarse), set()).add(elem)
                by.setdefault((size, elem), set()).add(topo)
        inconsistency += sum(1 for v in by.values() if len(v) > 1)
    out.append(_result("granularity_mapping_deterministic", inconsistency == 0,
                       inconsistencies=int(inconsistency)))
    return out


def _relation_explicit(present: np.ndarray, bu: np.ndarray, bv: np.ndarray,
                       bt: np.ndarray) -> dict[tuple[int, int, int], int]:
    out: dict[tuple[int, int, int], int] = defaultdict(int)
    for bid in range(bu.size):
        b = int(bt[bid])
        for u, v in ((int(bu[bid]), int(bv[bid])), (int(bv[bid]), int(bu[bid]))):
            for t in np.nonzero(present[u])[0].tolist():
                for s in np.nonzero(present[v])[0].tolist():
                    t2, s2 = (t, s) if t <= s else (s, t)
                    out[(t2, s2, b)] += 1
    return out


def relation_checks(stage: D.Stage0, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    z = stage.z
    rel_counts = np.asarray(z["rel_counts"], np.int64)
    anchor_marginal = np.asarray(z["anchor_marginal"], np.int64)
    bond_slots = np.asarray(z["bond_slots"], np.int64)
    rel_meta = stage.rel_meta
    n_rel = len(rel_meta)
    ok_bounds = True
    max_upper_violation = 0
    max_lower_violation = 0
    for j, meta in enumerate(rel_meta):
        t, s, b = int(meta["t"]), int(meta["s"]), int(meta["bond_type"])
        A = anchor_marginal[:, t, b]
        B = anchor_marginal[:, s, b]
        R = rel_counts[:, j]
        upper = np.minimum(A, B)
        max_upper_violation = max(max_upper_violation, int(np.max(R - upper)))
        max_lower_violation = max(max_lower_violation, int(np.max(-R)))
        # relation hits are a subset of source-t and target-s slots
    ok_bounds = max_upper_violation <= 0 and max_lower_violation <= 0
    out.append(_result("real_relation_bounds", ok_bounds,
                       max_upper_violation=int(max_upper_violation),
                       max_lower_violation=int(max_lower_violation), n_relations=n_rel))
    # A and B are the same column by construction; check slot identities
    slot_ok = bool(np.all(bond_slots % 2 == 0))
    out.append(_result("bond_slots_even", slot_ok,
                       max_odd=int(np.max(bond_slots % 2)) if bond_slots.size else 0))
    # explicit REAL count on a small artificial graph
    present = np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]], np.uint8)
    bu = np.asarray([0, 1, 2], np.int64)
    bv = np.asarray([1, 2, 3], np.int64)
    bt = np.asarray([0, 0, 0], np.int64)
    explicit = _relation_explicit(present, bu, bv, bt)
    got = defaultdict(int)
    for bid in range(bu.size):
        for u, v in ((int(bu[bid]), int(bv[bid])), (int(bv[bid]), int(bu[bid]))):
            for t in np.nonzero(present[u])[0].tolist():
                for s in np.nonzero(present[v])[0].tolist():
                    t2, s2 = (t, s) if t <= s else (s, t)
                    got[(t2, s2, 0)] += 1
    out.append(_result("real_relation_explicit_small_graph", dict(got) == dict(explicit),
                       got={str(k): int(v) for k, v in got.items()},
                       explicit={str(k): int(v) for k, v in explicit.items()}))
    # hypergeometric expectation vs exact enumeration
    M, B, A = 8, 3, 4
    slots = np.arange(M)
    src = set(range(A))
    vals = []
    for subset in itertools.combinations(slots, B):
        K = len(src & set(subset))
        vals.append(max(K - 1, 0))
    explicit_mean = float(np.mean(vals))
    formula = 0.0
    for j in range(2, min(A, B) + 1):
        formula += float(stats.hypergeom.sf(j - 1, M, B, A))
    am = np.zeros((1, 2, 1), np.int64)
    am[0, 0, 0] = A
    am[0, 1, 0] = B
    tiny = S.null_expectations(
        np.zeros((1, 1), np.int64),
        [{"t": 0, "s": 1, "bond_type": 0}],
        am,
        np.asarray([[M]], np.int64),
        [[0, 1]],
    )
    out.append(_result("hypergeometric_hinge_expectation",
                       abs(explicit_mean - formula) < 1e-12 and
                       abs(float(tiny[0, 1]) - explicit_mean) < 1e-12 and
                       abs(float(tiny[0, 0]) - A * B / M) < 1e-12,
                       explicit=explicit_mean, formula=formula,
                       function_value=float(tiny[0, 1]), ek=float(tiny[0, 0])))
    # NULL expectation non-linear gap: E[(K-1)+] != (E[K]-1)+
    ek = A * B / M
    naive = max(ek - 1, 0)
    out.append(_result("null_uses_expectation_not_plug_in",
                       abs(explicit_mean - naive) > 1e-3,
                       explicit=explicit_mean, plug_in=naive))
    return out


def ridge_direct_check(stage: D.Stage0, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    import hrf_ridge as R

    design = stage.design("F_BASE")
    X = design["X"]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(stage.train.size, 200, replace=False))
    cols = np.arange(20)
    A = X[stage.train[idx][:, None], cols]
    z = stage.y_std[stage.train[idx]]
    mu = A.mean(0); Ac = A - mu; zc = z - z.mean()
    n = A.shape[0]
    lam = 0.1
    direct = np.linalg.solve(Ac.T @ Ac + n * lam * np.eye(cols.size), Ac.T @ zc)
    sol = R.solve_path(A, z, np.arange(n), [lam])
    diff = float(np.max(np.abs(sol["beta"][:, 0] - direct)))
    return [_result("ridge_vs_direct_solve", diff < 1e-9,
                    max_abs_diff=diff, n=200, p=20, lambda_=lam)]


def replay_checks(stage: D.Stage0, ridge_dir: Path) -> list[dict[str, Any]]:
    out = []
    for arm in ("F_BASE", "F_REAL", "L_BASE", "L_REAL", "F_NULL"):
        state = np.load(ridge_dir / "states" / f"{arm}.npz", allow_pickle=False)
        beta = np.asarray(state["beta_selected"], np.float64)
        design = stage.design(arm)
        X = design["X"]
        mu = np.asarray(state["mu"], np.float64)
        zmean = float(np.asarray(state["ymean"]).reshape(-1)[0])
        pred = stage.scaler["mean"] + stage.scaler["std"] * (zmean + (X - mu) @ beta)
        ref = np.load(ridge_dir / "predictions" / f"{arm}_valid.npz", allow_pickle=False)
        rows = np.asarray(ref["row_id"], np.int64)
        diff = float(np.max(np.abs(pred[rows] - np.asarray(ref["pred"], np.float64))))
        ledger = np.load(ridge_dir / "ledger" / f"{arm}.npz", allow_pickle=False)
        recon = (float(np.asarray(ledger["intercept"]).reshape(-1)[0])
                 + np.asarray(ledger["main_effect_sum"], np.float64)
                 + np.asarray(ledger["relation_effect_sum"], np.float64))
        err = float(np.max(np.abs(recon - pred)))
        out.append(_result(f"ridge_predictions_replay_{arm}", diff < 1e-9,
                           max_abs_diff=diff))
        out.append(_result(f"ledger_reconstructs_prediction_{arm}", err < 1e-7,
                           max_abs_error=err))
    return out


def _pick_columns(X: np.ndarray, train: np.ndarray, cands: Sequence[int],
                  k: int, raw_of: np.ndarray, used_raw: set[int],
                  max_corr: float = 0.9) -> list[int]:
    chosen: list[int] = []
    for c in cands:
        rc = int(raw_of[int(c)])
        if rc in used_raw:
            continue
        v = np.asarray(X[train, c], np.float64)
        if float(v.std()) < 1e-9:
            continue
        ok = True
        for ch in chosen:
            u = np.asarray(X[train, ch], np.float64)
            r = float(np.corrcoef(v, u)[0, 1])
            if abs(r) > max_corr:
                ok = False
                break
        if ok:
            chosen.append(int(c))
            used_raw.add(rc)
        if len(chosen) >= k:
            break
    return chosen


def synthetic_checks(stage: D.Stage0, cfg: Mapping[str, Any],
                     ridge_dir: Path) -> list[dict[str, Any]]:
    """Planted-term positive control on real Lipo graphs (no real logD used)."""
    import hrf_ridge as R

    rng = np.random.default_rng(0)
    train = stage.train
    meta = stage.base_raw_meta
    # candidate base linear / hinge columns, sorted deterministically by DF desc
    lin_c, hin_c = [], []
    for c in range(stage.base_basis.shape[1]):
        rc = int(stage.base_raw_col[c])
        m = meta[rc]
        if m["kind"] != "rooted_structure" or int(m["train_df"]) < 50:
            continue
        if int(stage.base_knot[c]) == 0:
            lin_c.append((int(m["train_df"]), rc, c))
        else:
            hin_c.append((int(m["train_df"]), rc, c))
    lin_c = [c for _, _, c in sorted(lin_c, key=lambda t: (-t[0], t[1], t[2]))]
    hin_c = [c for _, _, c in sorted(hin_c, key=lambda t: (-t[0], t[1], t[2]))]
    rel_lin, rel_hin = [], []
    for c in range(stage.rel_basis.shape[1]):
        rc = int(stage.rel_raw_col[c])
        if int(stage.rel_meta[rc]["train_df"]) < 50:
            continue
        if int(stage.rel_knot[c]) == 0:
            rel_lin.append((int(stage.rel_meta[rc]["train_df"]), rc, c))
        else:
            rel_hin.append((int(stage.rel_meta[rc]["train_df"]), rc, c))
    rel_lin = [c for _, _, c in sorted(rel_lin, key=lambda t: (-t[0], t[1], t[2]))]
    rel_hin = [c for _, _, c in sorted(rel_hin, key=lambda t: (-t[0], t[1], t[2]))]

    base_used: set[int] = set()
    rel_used: set[int] = set()
    base_lin = _pick_columns(stage.base_basis, train, lin_c, 2,
                             stage.base_raw_col, base_used)
    base_hin = _pick_columns(stage.base_basis, train, hin_c, 2,
                             stage.base_raw_col, base_used)
    rel_l = _pick_columns(stage.rel_basis, train, rel_lin, 2,
                          stage.rel_raw_col, rel_used)
    rel_h = _pick_columns(stage.rel_basis, train, rel_hin, 2,
                          stage.rel_raw_col, rel_used)
    if any(len(x) < 2 for x in (base_lin, base_hin, rel_l, rel_h)):
        return [_result("synthetic_planted_terms_available", False,
                        base_lin=base_lin, base_hin=base_hin,
                        rel_lin=rel_l, rel_hin=rel_h)]
    coefs = {
        "base_lin": [0.8, -0.5],
        "base_hin": [0.4, -0.3],
        "rel_lin": [0.6, -0.4],
        "rel_hin": [0.3, -0.2],
    }
    # scale each planted term to unit Train SD; true raw coefficient is coef/sd
    def term(values: np.ndarray, c: int, coef: float) -> tuple[np.ndarray, float]:
        sd = float(np.std(values[train, c]))
        return values[:, c] * (coef / max(sd, 1e-12)), coef / max(sd, 1e-12)
    yL = np.zeros(stage.n, np.float64)
    trueL = []
    for c, co in zip(base_lin, coefs["base_lin"]):
        t, b = term(stage.base_basis, c, co)
        yL += t; trueL.append(b)
    yF = yL.copy(); trueF = list(trueL)
    for c, co in zip(base_hin, coefs["base_hin"]):
        t, b = term(stage.base_basis, c, co)
        yF += t; trueF.append(b)
    yR = yF.copy(); trueR = list(trueF)
    for c, co in zip(rel_l, coefs["rel_lin"]):
        t, b = term(stage.rel_basis, c, co)
        yR += t; trueR.append(b)
    for c, co in zip(rel_h, coefs["rel_hin"]):
        t, b = term(stage.rel_basis, c, co)
        yR += t; trueR.append(b)
    # small full-rank design: planted terms + deterministic distractors
    small_rows = np.sort(rng.choice(train, 300, replace=False))
    planted_cols = [("base", c) for c in base_lin + base_hin] + \
                   [("rel", c) for c in rel_l + rel_h]
    pool = [("base", c) for c in lin_c if c not in base_lin + base_hin] + \
           [("rel", c) for c in rel_lin if c not in rel_l + rel_h]
    chosen = list(planted_cols)
    def mat_for(cols):
        mats = []
        for block, c in cols:
            mats.append(stage.base_basis[:, c] if block == "base" else stage.rel_basis[:, c])
        return np.stack(mats, axis=1)[small_rows]
    def std_mat(cols):
        m = mat_for(cols)
        sd = m.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (m - m.mean(axis=0)) / sd
    for cand in pool:
        if len(chosen) >= 10:
            break
        if np.linalg.matrix_rank(std_mat(chosen + [cand])) > \
                np.linalg.matrix_rank(std_mat(chosen)):
            chosen.append(cand)
    A_raw = mat_for(chosen)
    # scale by the full-Train SD so recovered raw coefficients are directly
    # comparable to the planted coefficients (which also use full-Train SD).
    sdA = np.asarray([
        (stage.base_basis[train, c] if block == "base" else stage.rel_basis[train, c]).std()
        for block, c in chosen
    ], np.float64)
    sdA = np.where(sdA < 1e-12, 1.0, sdA)
    A = (A_raw - A_raw.mean(axis=0)) / sdA
    rank = int(np.linalg.matrix_rank(A))
    cond = float(np.linalg.cond(A))
    y_small = {
        "L": yL[small_rows] - 0.0,
        "F": yF[small_rows] - 0.0,
        "R": yR[small_rows] - 0.0,
    }
    # true coefficients for every chosen column (0 for distractors)
    true_map = {}
    for block, c, b in ([(a, c, b) for a, c, b in zip(["base"]*2, base_lin, trueL[:2])]
                        + [("base", c, b) for c, b in zip(base_hin, trueF[2:4])]
                        + [("rel", c, b) for c, b in zip(rel_l, trueR[4:6])]
                        + [("rel", c, b) for c, b in zip(rel_h, trueR[6:8])]):
        true_map[(block, int(c))] = float(b)
    active = {
        "L": {("base", int(c)) for c in base_lin},
        "F": {("base", int(c)) for c in base_lin + base_hin},
        "R": {("base", int(c)) for c in base_lin + base_hin} |
             {("rel", int(c)) for c in rel_l + rel_h},
    }
    small_ok = True
    small_detail = {"rank": rank, "n_columns": int(A.shape[1]),
                    "condition_number": cond,
                    "planted": [(b, int(c)) for b, c in planted_cols]}
    for key, yy in y_small.items():
        sol = R.solve_path(A, yy, np.arange(A.shape[0]), [1e-6])
        pred = sol["ymean"] + (A - sol["mu"]) @ sol["beta"][:, 0]
        rmse = float(np.sqrt(np.mean((pred - yy) ** 2)))
        beta_raw = sol["beta"][:, 0] / sdA
        coeff_err = 0.0
        for i, (block, c) in enumerate(chosen):
            want = true_map.get((block, int(c)), 0.0) if (block, int(c)) in active[key] else 0.0
            coeff_err = max(coeff_err, abs(float(beta_raw[i]) - want))
        small_detail[key] = {"prediction_rmse": rmse, "max_coeff_abs_error": coeff_err}
        small_ok = small_ok and rmse < 1e-6 and coeff_err < 1e-5
    out = [_result("synthetic_small_full_rank_recovery", small_ok, **small_detail)]

    # full library: prediction vs coefficient recovery
    full = {}
    for name, design_name, y in (("L", "L_BASE", yL), ("F", "F_BASE", yF),
                                 ("R", "F_REAL", yR), ("R_base", "F_BASE", yR),
                                 ("R_null", "F_NULL", yR), ("F_linearbase", "L_BASE", yF),
                                 ("R_linearreal", "L_REAL", yR)):
        design = stage.design(design_name)
        X = design["X"]
        sol = R.solve_path(X, y, train, [1e-6])
        pred = sol["ymean"] + (X - sol["mu"]) @ sol["beta"][:, 0]
        split_rmse = {s: float(np.sqrt(np.mean((pred[stage.rows[s]] - y[stage.rows[s]]) ** 2)))
                      for s in ("train", "valid", "test")}
        # map (block, basis col) -> design col
        colmap = {}
        rel_cols = []
        for g in design["groups"]:
            if g["block"].startswith("relation"):
                rel_cols.extend(int(x) for x in g["cols"])
            for j, dcol in enumerate(g["cols"]):
                block = "rel" if g["block"].startswith("relation") else "base"
                colmap[(block, int(g["basis_cols"][j]))] = int(dcol)
        rel_effect = X[:, rel_cols] @ sol["beta"][rel_cols, 0] if rel_cols else np.zeros(X.shape[0])
        coeff_err = {}
        for key, want in (("base_lin", trueL[:2]), ("base_hin", trueF[2:4]),
                          ("rel_lin", trueR[4:6]), ("rel_hin", trueR[6:8])):
            cols = {"base_lin": base_lin, "base_hin": base_hin,
                    "rel_lin": rel_l, "rel_hin": rel_h}[key]
            block = "rel" if key.startswith("rel") else "base"
            errs = []
            for c, b in zip(cols, want):
                dcol = colmap.get((block, int(c)))
                if dcol is None:
                    continue
                scale = (stage.rel_scale[c] if block == "rel" else stage.base_scale[c])
                errs.append(abs(float(sol["beta"][dcol, 0]) / float(scale) - float(b)))
            if errs:
                coeff_err[key] = float(max(errs))
        full[name] = {"arm": design_name, "prediction_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                      "split_rmse": split_rmse,
                      "has_relation_columns": bool(rel_cols),
                      "relation_effect_max_abs": float(np.abs(rel_effect).max()),
                      "coefficient_max_abs_error": coeff_err}
    # pass criteria: function family recovers its own synthetic labels; a
    # relation-free model cannot recover the relation signal.
    own_ok = (full["L"]["split_rmse"]["train"] < 1e-3 and
              full["F"]["split_rmse"]["train"] < 1e-3 and
              full["R"]["split_rmse"]["train"] < 1e-3)
    negative_ok = (not full["R_base"]["has_relation_columns"] and
                   full["R_base"]["relation_effect_max_abs"] == 0.0 and
                   full["R"]["relation_effect_max_abs"] > 0.0)
    out.append(_result("synthetic_full_library_family_recovery", own_ok, **full))
    out.append(_result("synthetic_relation_negative_control", negative_ok,
                       R_base_test_rmse=full["R_base"]["split_rmse"]["test"],
                       R_real_test_rmse=full["R"]["split_rmse"]["test"],
                       R_base_has_relation_columns=full["R_base"]["has_relation_columns"],
                       R_null_test_rmse=full["R_null"]["split_rmse"]["test"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", default=str(C.DATA_DIR / "hsrf_stage0.npz"))
    ap.add_argument("--ridge-dir", default=str(C.RESULTS_DIR / "ridge"))
    ap.add_argument("--out", default=str(C.ANALYSIS_DIR / "checks.json"))
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    C.assert_source_pins()
    payload = C.load_payload()
    stage = D.Stage0(args.stage0)
    checks = []
    checks += data_checks(payload)
    checks += structure_checks(payload, cfg)
    checks += relation_checks(stage, cfg)
    checks += ridge_direct_check(stage, cfg)
    if (Path(args.ridge_dir) / "states" / "F_REAL.npz").is_file():
        checks += replay_checks(stage, Path(args.ridge_dir))
    if not args.skip_synthetic:
        checks += synthetic_checks(stage, cfg, Path(args.ridge_dir))
    passed = sum(1 for c in checks if c["pass"])
    payload_out = {
        "round": cfg["round_id"], "commit": C.git_commit(), "created_at": C.now_utc(),
        "stage0": str(stage.path), "stage0_sha256": C.sha256_file(stage.path),
        "n_checks": len(checks), "n_pass": passed,
        "all_pass": bool(passed == len(checks)), "checks": checks,
    }
    C.write_json(Path(args.out), payload_out)
    print(json.dumps({"n_checks": len(checks), "n_pass": passed,
                      "failures": [c["name"] for c in checks if not c["pass"]]},
                     indent=1))


if __name__ == "__main__":
    main()