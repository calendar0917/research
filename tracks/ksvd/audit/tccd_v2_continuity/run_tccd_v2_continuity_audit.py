#!/usr/bin/env python
"""TCCD-v2 representation continuity audit (read-only diagnostic).

Central question:
    Does the TCCD-v2 learned representation have graded chemical continuity?

This script is analysis-only:
  * never trains, never touches official test, never writes formal result files;
  * reuses the frozen Prototype-REL best checkpoint
    tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt
    and the TCCD-v0/v1 radius-2 canonical patch record cache.

It builds deterministic chemical-similarity tiers on internal-dev patches,
audits Z-space (latent) and C-space (prototype assignment), runs monotonicity
tests, prototype-coherence tests and case studies, and writes a JSON report.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import pathlib
import sys
import time

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V
from tracks.ksvd.code.run_tccd_v0 import internal_split

REPO = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parent
CKPT = REPO / "tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt"
CACHE = REPO / "tracks/ksvd/results/tccd_v0/cache/tccd_v0_train_r2_M14_A21_B3_n10000.pkl"

K_PROTO = V.K_PROTO
D_LOCAL = V.D_LOCAL
N_TARGET = 120_000
BOOT_B = 400
SEED = 20260922

CATALOG = json.loads((OUT / "catalog.json").read_text())
CAT_LABEL = {int(k): v for k, v in CATALOG["catalog_labels"].items()}
BOND_LABEL = {0: "single", 1: "double", 2: "triple"}


# ---------------------------------------------------------------------------
# pair metrics
# ---------------------------------------------------------------------------
def js_distance(p, q):
    """Jensen-Shannon distance (base-2) between probability rows, vectorized."""
    m = 0.5 * (p + q)
    lp = np.log2((p + 1e-12) / (m + 1e-12))
    lq = np.log2((q + 1e-12) / (m + 1e-12))
    kl = 0.5 * np.sum(p * lp, axis=1) + 0.5 * np.sum(q * lq, axis=1)
    return np.sqrt(np.maximum(kl, 0.0))


def cos_rows(A, i, B, j):
    return np.einsum("ij,ij->i", A[i], B[j])


def summarize(vals: np.ndarray) -> dict:
    vals = np.asarray(vals, dtype=np.float64)
    q = np.percentile(vals, [5, 25, 50, 75, 95])
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(q[2]),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "q75": float(q[3]),
        "q95": float(q[4]),
        "iqr": float(q[3] - q[1]),
    }


def cluster_boot_mean(vals, ga, gb, n_graphs, rng, B=BOOT_B):
    """Bootstrap CI of the mean by resampling molecules (both endpoints present)."""
    vals = np.asarray(vals, dtype=np.float64)
    ga = np.asarray(ga)
    gb = np.asarray(gb)
    out = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, n_graphs, n_graphs), minlength=n_graphs)
        w = cnt[ga].astype(np.float64) * cnt[gb].astype(np.float64)
        sw = w.sum()
        if sw > 0:
            out.append(float((vals * w).sum() / sw))
    if not out:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def cluster_boot_stat(levels, vals, ga, gb, n_graphs, rng, stat, B=100):
    """Cluster bootstrap for a statistic computed on pair lists (graph resample)."""
    out = []
    G = n_graphs
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G)
        w = cnt[ga] * cnt[gb]
        sel = w > 0
        if sel.sum() < 50:
            continue
        out.append(stat(levels[sel], vals[sel]))
    if not out:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def spearman(x, y):
    from scipy.stats import spearmanr

    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------
def sample_within_groups(groups, target, rng):
    """Approximately uniform-over-pairs sample within a list of index groups."""
    groups = [np.asarray(g, dtype=np.int64) for g in groups if len(g) >= 2]
    if not groups:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    sizes = np.array([len(g) for g in groups])
    w = sizes * (sizes - 1) // 2
    W = int(w.sum())
    I, J = [], []
    for g, wg in zip(groups, w):
        if wg < 1:
            continue
        n_g = int(round(target * float(wg) / W))
        n_g = max(1, min(n_g, int(wg)))
        m = len(g)
        cand = n_g + max(20, n_g // 4)
        a = rng.integers(0, m, cand)
        b = rng.integers(0, m, cand)
        ok = a != b
        a, b = a[ok], b[ok]
        if a.size == 0:
            continue
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        key = lo.astype(np.int64) * m + hi
        _, u = np.unique(key, return_index=True)
        lo, hi = lo[u][:n_g], hi[u][:n_g]
        I.append(g[lo])
        J.append(g[hi])
    if not I:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(I), np.concatenate(J)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(CKPT), help="checkpoint to audit (default: frozen TCCD-v2 BASE)")
    ap.add_argument("--out-json", default=None, help="output JSON path (default: frozen BASE audit path)")
    ap.add_argument("--label", default="tccd_v2", help="checkpoint label recorded in the report")
    cli = ap.parse_args(argv)
    ckpt_path = pathlib.Path(cli.checkpoint)
    out_json = pathlib.Path(cli.out_json) if cli.out_json else (OUT / "tccd_v2_continuity_audit.json")
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    # ---------------- load records & checkpoint ----------------
    rec = T.load_records(CACHE)["records"]
    tr_idx, dev_idx = internal_split(len(rec))
    dev = [int(i) for i in dev_idx]
    print(f"[load] graphs={len(rec)} internal_train={len(tr_idx)} internal_dev={len(dev)}")

    X, gid, keys, ys = [], [], [], []
    for gi in dev:
        r = rec[gi]
        X.append(np.asarray(r["X"], dtype=np.float32))
        gid.append(np.full(int(r["n"]), gi, dtype=np.int64))
        keys.extend(r["keys"])
        ys.append(np.full(int(r["n"]), float(r["y"]), dtype=np.float64))
    X = np.concatenate(X, 0)
    gid = np.concatenate(gid)
    y_patch = np.concatenate(ys)
    Pn, F = X.shape
    print(f"[load] dev patches={Pn} F={F}")

    key_id = np.empty(Pn, dtype=np.int64)
    kmap = {}
    for i, k in enumerate(keys):
        if k not in kmap:
            kmap[k] = len(kmap)
        key_id[i] = kmap[k]
    n_keys = len(kmap)

    ckpt = T._torch().load(ckpt_path, map_location="cpu", weights_only=False)
    W = ckpt["W"].numpy().astype(np.float32)
    Pmat = ckpt["P"].numpy().astype(np.float32)
    tau = float(V.TEMP_MIN + V.TEMP_SPAN * (1.0 / (1.0 + math.exp(-float(ckpt["temp_logit"])))))
    Z = X @ W
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), V.EPS)
    Pbar = Pmat / np.maximum(np.linalg.norm(Pmat, axis=0, keepdims=True), V.EPS)
    logits = (Zn @ Pbar) / tau
    logits -= logits.max(1, keepdims=True)
    C = np.exp(logits)
    C /= C.sum(1, keepdims=True)
    argmax = C.argmax(1)
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    print(f"[model] tau={tau:.6f} commit_ckpt_keys={list(ckpt.keys())}")

    # validation against the frozen vocabulary numbers
    cbar = C.mean(0)
    eff = float(1.0 / np.sum(cbar * cbar))
    active = int(np.sum(cbar >= V.DEAD_THRESHOLD))
    print(f"[validate] active={active} effective={eff:.4f} (expect 64 / 62.5755)")

    # ---------------- decode patch chemistry from X ----------------
    cap = V.frozen_layout().capacity
    na = V.frozen_layout().n_atom
    nb = V.frozen_layout().n_bond
    sl = V.frozen_layout().block_slices()
    pi, pj = V.frozen_layout().pair_i, V.frozen_layout().pair_j
    pid = np.full((cap, cap), -1, dtype=np.int64)
    pid[pi, pj] = np.arange(len(pi))

    n_atoms = np.zeros(Pn, dtype=np.int64)
    root = np.zeros(Pn, dtype=np.int64)
    degree = np.zeros(Pn, dtype=np.int64)
    # heavy-element view: collapse "C H1", "N H2 +", "O -" ... to base element
    import re as _re

    base_of = {}
    for c in range(na):
        m = _re.match(r"[A-Z][a-z]?", CAT_LABEL[c])
        base_of[c] = m.group(0) if m else CAT_LABEL[c]
    bases = sorted(set(base_of.values()))
    heavy_index = {b: i for i, b in enumerate(bases)}
    heavy_of = {c: heavy_index[base_of[c]] for c in range(na)}
    nheavy = len(bases)
    root_heavy = np.zeros(Pn, dtype=np.int64)
    s1heavy = np.zeros((Pn, nheavy), dtype=np.int64)
    n2 = np.zeros(Pn, dtype=np.int64)
    s1atom = np.zeros((Pn, na), dtype=np.int64)
    rbond = np.zeros((Pn, nb), dtype=np.int64)
    s2atom = np.zeros((Pn, na), dtype=np.int64)
    attach = np.zeros((Pn, na * nb), dtype=np.int64)
    s2bond = np.zeros((Pn, nb), dtype=np.int64)
    l1id = np.zeros(Pn, dtype=np.int64)
    l1map: dict = {}
    decoded = [None] * Pn

    def l1_key(atom, shell, edges):
        sh1 = [s for s in range(len(shell)) if shell[s] == 1]
        rb, inner = {}, {}
        for a, b, bt in edges:
            if a == 0 and b in sh1:
                rb[b] = bt
            elif b == 0 and a in sh1:
                rb[a] = bt
            elif a in sh1 and b in sh1:
                inner[(min(a, b), max(a, b))] = bt
        best = None
        for perm in itertools.permutations(sh1):
            pos = {v: i for i, v in enumerate(perm)}
            ent = []
            for v in perm:
                sub = tuple(
                    sorted(
                        (inner[(min(v, w), max(v, w))], pos[w])
                        for w in sh1
                        if w != v and (min(v, w), max(v, w)) in inner
                    )
                )
                ent.append((int(atom[v]), int(rb.get(v, -1)), sub))
            cand = (int(atom[0]), tuple(ent))
            if best is None or cand < best:
                best = cand
        return best

    t0 = time.time()
    for i in range(Pn):
        row = X[i]
        mask = row[sl["mask"]] > 0.5
        n = int(mask.sum())
        atom = row[sl["atom"]].reshape(cap, na).argmax(1)[:n].astype(np.int64)
        shell = row[sl["shell"]].reshape(cap, 3).argmax(1)[:n].astype(np.int64)
        topo = row[sl["topology"]] > 0.5
        bond = row[sl["bond"]].reshape(nb, len(pi)).argmax(0)
        edges = []
        for s in range(n):
            for t in range(s + 1, n):
                q = pid[s, t]
                if topo[q]:
                    edges.append((s, t, int(bond[q])))
        decoded[i] = (n, atom, shell, edges)
        n_atoms[i] = n
        root[i] = atom[0]
        root_heavy[i] = heavy_of[int(atom[0])]
        for s in range(n):
            if shell[s] == 1:
                s1atom[i, atom[s]] += 1
                s1heavy[i, heavy_of[int(atom[s])]] += 1
            elif shell[s] == 2:
                n2[i] += 1
                s2atom[i, atom[s]] += 1
        for a, b, bt in edges:
            if a == 0 and shell[b] == 1:
                rbond[i, bt] += 1
            elif b == 0 and shell[a] == 1:
                rbond[i, bt] += 1
            elif (shell[a] == 1 and shell[b] == 2) or (shell[a] == 2 and shell[b] == 1):
                s1t = int(atom[a if shell[a] == 1 else b])
                attach[i, s1t * nb + bt] += 1
            elif shell[a] == 2 and shell[b] == 2:
                s2bond[i, bt] += 1
        degree[i] = int(s1atom[i].sum())
        k = l1_key(atom, shell, edges)
        if k not in l1map:
            l1map[k] = len(l1map)
        l1id[i] = l1map[k]
    print(f"[decode] {time.time()-t0:.1f}s unique_keys={n_keys} unique_l1={len(l1map)}")
    # sanity: catalog slots
    assert np.all(degree == s1atom.sum(1))

    # ---------------- chemical distance helpers ----------------
    def c1_dist(i, j):
        return 0.5 * np.abs(s1atom[i] - s1atom[j]).sum(1) + 0.5 * np.abs(rbond[i] - rbond[j]).sum(1)

    def d2_dist(i, j):
        return (
            np.abs(n2[i] - n2[j])
            + 0.5 * np.abs(s2atom[i] - s2atom[j]).sum(1)
            + 0.5 * np.abs(attach[i] - attach[j]).sum(1)
            + 0.5 * np.abs(s2bond[i] - s2bond[j]).sum(1)
        )

    def metrics(i, j):
        return {
            "zcos": cos_rows(Zn, i, Zn, j),
            "ccos": cos_rows(Cn, i, Cn, j),
            "cjs": js_distance(C[i], C[j]),
            "agree": (argmax[i] == argmax[j]).astype(np.float64),
            "xcos": cos_rows(Xn, i, Xn, j),
        }

    def keep_cross(i, j):
        return gid[i] != gid[j]

    # ---------------- tier sampling ----------------
    tiers = {}
    t0 = time.time()

    # Exact: same canonical key
    groups = collections.defaultdict(list)
    for i in range(Pn):
        groups[int(key_id[i])].append(i)
    i, j = sample_within_groups(list(groups.values()), N_TARGET * 6, rng)
    m = keep_cross(i, j)
    i, j = i[m], j[m]
    sel = rng.permutation(len(i))[:N_TARGET]
    tiers["Exact"] = (i[sel], j[sel])

    # Very near: identical radius-1 rooted labeled subgraph, different full key
    groups = collections.defaultdict(list)
    for i2 in range(Pn):
        groups[int(l1id[i2])].append(i2)
    i, j = sample_within_groups(list(groups.values()), N_TARGET * 6, rng)
    m = keep_cross(i, j) & (key_id[i] != key_id[j])
    i, j = i[m], j[m]
    sel = rng.permutation(len(i))[:N_TARGET]
    tiers["VeryNear"] = (i[sel], j[sel])

    # matched (root, degree) candidate pool -> Moderate / HardNegative
    groups = collections.defaultdict(list)
    for i2 in range(Pn):
        groups[(int(root[i2]), int(degree[i2]))].append(i2)
    ci, cj = sample_within_groups(list(groups.values()), 2_500_000, rng)
    m = keep_cross(ci, cj)
    ci, cj = ci[m], cj[m]
    c1 = c1_dist(ci, cj)
    l1eq = l1id[ci] == l1id[cj]
    # Moderate: same root/degree, radius-1 differs, exactly one peripheral edit
    mm = (c1 == 1) & (~l1eq)
    mi, mj = ci[mm], cj[mm]
    sel = rng.permutation(len(mi))[:N_TARGET]
    tiers["Moderate"] = (mi[sel], mj[sel])
    # Hard negative: same root/degree, >=2 peripheral edits, size within +-1
    mh = (c1 >= 2) & (np.abs(n_atoms[ci] - n_atoms[cj]) <= 1)
    hi, hj = ci[mh], cj[mh]
    sel = rng.permutation(len(hi))[:N_TARGET]
    tiers["HardNegative"] = (hi[sel], hj[sel])
    print(
        f"[sample] moderate_pool={int(mm.sum())} hard_pool={int(mh.sum())} "
        f"c1_hist={collections.Counter(c1.tolist())}"
    )

    # Random: uniform cross-graph pairs
    ri = rng.integers(0, Pn, N_TARGET * 3)
    rj = rng.integers(0, Pn, N_TARGET * 3)
    m = keep_cross(ri, rj)
    ri, rj = ri[m], rj[m]
    tiers["Random"] = (ri[:N_TARGET], rj[:N_TARGET])

    print(f"[sample] {time.time()-t0:.1f}s tiers=" + str({k: len(v[0]) for k, v in tiers.items()}))

    # ---------------- per-tier metrics ----------------
    levels = {"Exact": 4, "VeryNear": 3, "Moderate": 2, "HardNegative": 1, "Random": 0}
    report = {"setup": {}, "tiers": {}, "monotonicity": {}, "graded_axis": {}, "coherence": {}, "cases": {}}
    pooled_lvl, pooled_z, pooled_c, pooled_js, pooled_x, pooled_g1, pooled_g2 = [], [], [], [], [], [], []
    per_tier_raw = {}
    for name, (i, j) in tiers.items():
        met = metrics(i, j)
        per_tier_raw[name] = met
        s = {
            "n_pairs": int(len(i)),
            "z_cosine": summarize(met["zcos"]),
            "c_cosine": summarize(met["ccos"]),
            "c_js_distance": summarize(met["cjs"]),
            "argmax_agreement": float(met["agree"].mean()),
            "raw_x_cosine": summarize(met["xcos"]),
            "z_cosine_mean_ci": cluster_boot_mean(met["zcos"], gid[i], gid[j], len(rec), rng),
            "c_cosine_mean_ci": cluster_boot_mean(met["ccos"], gid[i], gid[j], len(rec), rng),
            "c_js_mean_ci": cluster_boot_mean(met["cjs"], gid[i], gid[j], len(rec), rng),
        }
        report["tiers"][name] = s
        pooled_lvl.append(np.full(len(i), levels[name]))
        pooled_z.append(met["zcos"])
        pooled_c.append(met["ccos"])
        pooled_js.append(met["cjs"])
        pooled_x.append(met["xcos"])
        pooled_g1.append(gid[i])
        pooled_g2.append(gid[j])
    pooled_lvl = np.concatenate(pooled_lvl)
    pooled_z = np.concatenate(pooled_z)
    pooled_c = np.concatenate(pooled_c)
    pooled_js = np.concatenate(pooled_js)
    pooled_x = np.concatenate(pooled_x)
    pooled_g1 = np.concatenate(pooled_g1)
    pooled_g2 = np.concatenate(pooled_g2)

    # ---------------- monotonicity ----------------
    mono = {}
    for label, mask in [("all5", pooled_lvl >= 0), ("no_exact", pooled_lvl < 4)]:
        lv, zz, cc, jj = pooled_lvl[mask], pooled_z[mask], pooled_c[mask], pooled_js[mask]
        rz, pz = spearman(lv, zz)
        rc, pc = spearman(lv, cc)
        rj, pj_ = spearman(lv, jj)
        rx, px = spearman(lv, pooled_x[mask])
        mono[label] = {
            "spearman_level_vs_zcos": rz,
            "spearman_level_vs_ccos": rc,
            "spearman_level_vs_cjs": rj,
            "spearman_level_vs_raw_x_cosine": rx,
            "p_value_iid_note": "scipy iid p-values; use cluster CI below",
            "z_ci": cluster_boot_stat(
                lv, zz, pooled_g1[mask], pooled_g2[mask], len(rec), rng,
                lambda a, b: spearman(a, b)[0], B=120),
            "c_ci": cluster_boot_stat(
                lv, cc, pooled_g1[mask], pooled_g2[mask], len(rec), rng,
                lambda a, b: spearman(a, b)[0], B=120),
            "js_ci": cluster_boot_stat(
                lv, jj, pooled_g1[mask], pooled_g2[mask], len(rec), rng,
                lambda a, b: spearman(a, b)[0], B=120),
        }
    # adjacent-tier differences with cluster bootstrap
    adj = []
    order = ["Exact", "VeryNear", "Moderate", "HardNegative", "Random"]
    from scipy.stats import mannwhitneyu

    for a, b in zip(order[:-1], order[1:]):
        ia, ja = tiers[a]
        ib, jb = tiers[b]
        va = per_tier_raw[a]["zcos"]
        vb = per_tier_raw[b]["zcos"]
        u, pu = mannwhitneyu(va, vb, alternative="greater")
        adj.append({
            "high": a, "low": b,
            "z_mean_delta": float(va.mean() - vb.mean()),
            "c_mean_delta": float(per_tier_raw[a]["ccos"].mean() - per_tier_raw[b]["ccos"].mean()),
            "js_mean_delta": float(per_tier_raw[a]["cjs"].mean() - per_tier_raw[b]["cjs"].mean()),
            "mannwhitney_greater_p": float(pu),
            "auc_prob_high_gt_low": float(u / (len(va) * len(vb))),
        })
    report["monotonicity"] = {"pooled": mono, "adjacent": adj}

    # ---------------- graded continuous axes ----------------
    ga = {}
    # axis 1: same (root,degree), non-exact, bin by radius-1 edit count
    ci, cj = ci, cj
    m = (key_id[ci] != key_id[cj])
    ci, cj = ci[m], cj[m]
    c1 = c1_dist(ci, cj)
    l1eq = l1id[ci] == l1id[cj]
    axis1 = {}
    for lab, sel in [
        ("c1=0_l1_identical", l1eq),
        ("c1=1", c1 == 1),
        ("c1=2", c1 == 2),
        ("c1>=3", c1 >= 3),
    ]:
        if sel.sum() < 100:
            axis1[lab] = {"n": int(sel.sum())}
            continue
        ii, jj = ci[sel], cj[sel]
        met = metrics(ii, jj)
        axis1[lab] = {
            "n": int(sel.sum()),
            "z_cosine": summarize(met["zcos"]),
            "c_cosine": summarize(met["ccos"]),
            "c_js_distance": summarize(met["cjs"]),
            "argmax_agreement": float(met["agree"].mean()),
        }
    ga["radius1_edit_axis"] = axis1

    # axis 2: radius-1 identical (VeryNear), bin by radius-2 distance
    vi, vj = tiers["VeryNear"]
    d2 = d2_dist(vi, vj)
    key_eq = key_id[vi] == key_id[vj]
    axis2 = {}
    for lab, sel in [
        ("d2=1", d2 == 1),
        ("d2=2", d2 == 2),
        ("d2=3-4", (d2 >= 3) & (d2 <= 4)),
        ("d2>=5", d2 >= 5),
    ]:
        if sel.sum() < 100:
            axis2[lab] = {"n": int(sel.sum())}
            continue
        ii, jj = vi[sel], vj[sel]
        met = metrics(ii, jj)
        axis2[lab] = {
            "n": int(sel.sum()),
            "z_cosine": summarize(met["zcos"]),
            "c_cosine": summarize(met["ccos"]),
            "c_js_distance": summarize(met["cjs"]),
            "argmax_agreement": float(met["agree"].mean()),
        }
    ga["radius2_edit_axis_within_l1_identical"] = axis2
    # axis 3: within same (root,degree), non-exact, spearman of radius-1 edit count
    #          against Z/C similarity (a pure within-family graded test)
    from scipy.stats import spearmanr as _spr

    met_ci = metrics(ci, cj)
    c1v = c1_dist(ci, cj)
    axis3 = {
        "n": int(len(ci)),
        "spearman_c1_vs_zcos": [float(_spr(c1v, met_ci["zcos"]).statistic), float(_spr(c1v, met_ci["zcos"]).pvalue)],
        "spearman_c1_vs_ccos": [float(_spr(c1v, met_ci["ccos"]).statistic), float(_spr(c1v, met_ci["ccos"]).pvalue)],
        "note": "negative rho means more chemical edits -> less similar representation",
    }
    ga["within_family_radius1_edit_spearman"] = axis3

    # heavy-atom (element-level) robustness axis: same heavy root + degree,
    # bin by number of heavy-atom radius-1 substitutions.
    groups_h = collections.defaultdict(list)
    for i2 in range(Pn):
        groups_h[(int(root_heavy[i2]), int(degree[i2]))].append(i2)
    hci, hcj = sample_within_groups(list(groups_h.values()), 2_000_000, rng)
    mh = keep_cross(hci, hcj) & (key_id[hci] != key_id[hcj])
    hci, hcj = hci[mh], hcj[mh]
    c1h = 0.5 * np.abs(s1heavy[hci] - s1heavy[hcj]).sum(1) + 0.5 * np.abs(rbond[hci] - rbond[hcj]).sum(1)
    axis4 = {}
    for lab, sel in [("heavy_c1=0", c1h == 0), ("heavy_c1=1", c1h == 1),
                     ("heavy_c1=2", c1h == 2), ("heavy_c1>=3", c1h >= 3)]:
        if sel.sum() < 100:
            axis4[lab] = {"n": int(sel.sum())}
            continue
        ii, jj = hci[sel], hcj[sel]
        met = metrics(ii, jj)
        axis4[lab] = {
            "n": int(sel.sum()),
            "z_cosine": summarize(met["zcos"]),
            "c_cosine": summarize(met["ccos"]),
            "c_js_distance": summarize(met["cjs"]),
            "argmax_agreement": float(met["agree"].mean()),
        }
    ga["heavy_atom_edit_axis"] = axis4
    report["graded_axis"] = ga

    # ---------------- exact vs near gap ----------------
    z_exact = per_tier_raw["Exact"]["zcos"].mean()
    z_vn = per_tier_raw["VeryNear"]["zcos"].mean()
    z_mod = per_tier_raw["Moderate"]["zcos"].mean()
    z_hard = per_tier_raw["HardNegative"]["zcos"].mean()
    z_rand = per_tier_raw["Random"]["zcos"].mean()
    report["exact_vs_near_gap"] = {
        "z_exact": float(z_exact),
        "z_very_near": float(z_vn),
        "z_moderate": float(z_mod),
        "z_hard_negative": float(z_hard),
        "z_random": float(z_rand),
        "gap_exact_to_very_near": float(z_exact - z_vn),
        "gap_very_near_to_moderate": float(z_vn - z_mod),
        "gap_moderate_to_hard": float(z_mod - z_hard),
        "gap_hard_to_random": float(z_hard - z_rand),
        "fraction_of_exact_gap_lost_at_first_nonidentical": float((z_exact - z_vn) / max(z_exact - z_rand, 1e-9)),
        "c_exact": float(per_tier_raw["Exact"]["ccos"].mean()),
        "c_very_near": float(per_tier_raw["VeryNear"]["ccos"].mean()),
        "c_moderate": float(per_tier_raw["Moderate"]["ccos"].mean()),
        "c_hard": float(per_tier_raw["HardNegative"]["ccos"].mean()),
        "c_random": float(per_tier_raw["Random"]["ccos"].mean()),
        "agree_exact": float(per_tier_raw["Exact"]["agree"].mean()),
        "agree_very_near": float(per_tier_raw["VeryNear"]["agree"].mean()),
        "agree_moderate": float(per_tier_raw["Moderate"]["agree"].mean()),
        "agree_hard": float(per_tier_raw["HardNegative"]["agree"].mean()),
        "agree_random": float(per_tier_raw["Random"]["agree"].mean()),
    }

    # ---------------- prototype coherence ----------------
    def modal_share(arr):
        if len(arr) == 0:
            return 0.0
        c = np.bincount(np.asarray(arr).astype(np.int64))
        return float(c.max() / len(arr))

    gidx = {g: k for k, g in enumerate(dev)}
    ga_id = np.array([gidx[int(g)] for g in gid], dtype=np.int64)
    pos_to_local = {}
    for kk, gi in enumerate(dev):
        pass

    def within_pairs(idxs, n_sample, rng):
        idxs = np.asarray(idxs)
        m = len(idxs)
        if m < 2:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        t = min(n_sample, m * (m - 1) // 2)
        a = rng.integers(0, m, t * 2)
        b = rng.integers(0, m, t * 2)
        ok = a != b
        a, b = a[ok], b[ok]
        lo = idxs[np.minimum(a, b)]
        hi = idxs[np.maximum(a, b)]
        key = lo.astype(np.int64) * Pn + hi
        _, u = np.unique(key, return_index=True)
        return lo[u][:t], hi[u][:t]

    groups_rd = collections.defaultdict(list)
    for i2 in range(Pn):
        groups_rd[(int(root[i2]), int(degree[i2]))].append(i2)
    groups_rd = {k: np.asarray(v) for k, v in groups_rd.items()}

    coh_rows = []
    for k in range(K_PROTO):
        idx = np.flatnonzero(argmax == k)
        if len(idx) == 0:
            continue
        ii, jj = within_pairs(idx, 1500, rng)
        if len(ii) > 0:
            key_eq = float((key_id[ii] == key_id[jj]).mean())
            l1_eq = float((l1id[ii] == l1id[jj]).mean())
            c1m = float(c1_dist(ii, jj).mean())
            ccos_w = float(cos_rows(Cn, ii, Cn, jj).mean())
        else:
            key_eq = l1_eq = c1m = ccos_w = float("nan")
        # matched random baseline: same (root,degree) draw per assigned patch
        ridx = np.empty(0, dtype=np.int64)
        for i2 in idx:
            pool = groups_rd[(int(root[i2]), int(degree[i2]))]
            pool = pool[gid[pool] != gid[i2]]
            if len(pool) == 0:
                pool = groups_rd[(int(root[i2]), int(degree[i2]))]
            ridx = np.append(ridx, int(rng.choice(pool)))
        coh_rows.append({
            "prototype": k,
            "n": int(len(idx)),
            "key_conc": modal_share(key_id[idx]),
            "root_conc": modal_share(root[idx]),
            "degree_conc": modal_share(degree[idx]),
            "l1_conc": modal_share(l1id[idx]),
            "within_key_eq": key_eq,
            "within_l1_eq": l1_eq,
            "within_c1_mean": c1m,
            "within_ccos": ccos_w,
            "matched_key_conc": modal_share(key_id[ridx]),
            "matched_root_conc": modal_share(root[ridx]),
            "matched_degree_conc": modal_share(degree[ridx]),
            "matched_l1_conc": modal_share(l1id[ridx]),
        })
    coh = {kk: np.array([r[kk] for r in coh_rows]) for kk in coh_rows[0]}
    random_sets = []
    for _ in range(5):
        ridx = rng.choice(Pn, size=2000, replace=False)
        random_sets.append({
            "key_conc": modal_share(key_id[ridx]),
            "root_conc": modal_share(root[ridx]),
            "degree_conc": modal_share(degree[ridx]),
            "l1_conc": modal_share(l1id[ridx]),
        })
    # random within-pair radius-1 / key equality baseline
    ri0 = rng.integers(0, Pn, 200000)
    rj0 = rng.integers(0, Pn, 200000)
    mrm = (gid[ri0] != gid[rj0])
    ri0, rj0 = ri0[mrm], rj0[mrm]
    random_pair_baseline = {
        "l1_eq_rate": float((l1id[ri0] == l1id[rj0]).mean()),
        "key_eq_rate": float((key_id[ri0] == key_id[rj0]).mean()),
        "mean_c1_edits": float(c1_dist(ri0, rj0).mean()),
    }
    report["coherence"] = {
        "prototypes": coh_rows,
        "aggregate": {
            "mean_key_conc": float(np.nanmean(coh["key_conc"])),
            "mean_matched_key_conc": float(np.nanmean(coh["matched_key_conc"])),
            "mean_root_conc": float(np.nanmean(coh["root_conc"])),
            "mean_matched_root_conc": float(np.nanmean(coh["matched_root_conc"])),
            "mean_degree_conc": float(np.nanmean(coh["degree_conc"])),
            "mean_matched_degree_conc": float(np.nanmean(coh["matched_degree_conc"])),
            "mean_l1_conc": float(np.nanmean(coh["l1_conc"])),
            "mean_matched_l1_conc": float(np.nanmean(coh["matched_l1_conc"])),
            "mean_within_key_eq": float(np.nanmean(coh["within_key_eq"])),
            "mean_within_l1_eq": float(np.nanmean(coh["within_l1_eq"])),
            "mean_within_c1": float(np.nanmean(coh["within_c1_mean"])),
            "mean_within_ccos": float(np.nanmean(coh["within_ccos"])),
            "uniform_random_baseline": {
                kk: float(np.mean([r[kk] for r in random_sets]))
                for kk in random_sets[0]
            },
            "random_pair_baseline": random_pair_baseline,
        },
    }

    # ---------------- failure case studies ----------------
    def describe(i):
        n, atom, shell, edges = decoded[int(i)]
        # collapse shell-1: multiset of (bond, atom)
        s1 = collections.Counter()
        for a, b, bt in edges:
            if a == 0 and shell[b] == 1:
                s1[(BOND_LABEL[bt], CAT_LABEL[int(atom[b])])] += 1
            elif b == 0 and shell[a] == 1:
                s1[(BOND_LABEL[bt], CAT_LABEL[int(atom[a])])] += 1
        s2 = collections.Counter(CAT_LABEL[int(atom[s])] for s in range(n) if shell[s] == 2)
        return {
            "graph_id": int(gid[i]),
            "patch_index": int(i),
            "y": float(y_patch[i]),
            "root": CAT_LABEL[int(root[i])],
            "degree": int(degree[i]),
            "n_atoms": int(n_atoms[i]),
            "shell1": [f"{bt}-{at}" for (bt, at), c in sorted(s1.items()) for _ in range(c)],
            "shell2_atoms": [f"{at}" for at, c in sorted(s2.items()) for _ in range(c)],
            "key": keys[int(i)].hex()[:16],
            "l1_id": int(l1id[i]),
            "argmax_proto": int(argmax[i]),
            "z_norm": float(np.linalg.norm(Z[i])),
        }

    cases = {}
    # (a) very-near pairs that split prototypes with low C cosine
    vi, vj = tiers["VeryNear"]
    split = argmax[vi] != argmax[vj]
    cc = per_tier_raw["VeryNear"]["ccos"]
    order = np.argsort(cc)
    picked = []
    for t in order:
        if split[t] and len(picked) < 5:
            picked.append(int(t))
    cases["very_near_split_prototypes"] = [
        {
            "i": describe(vi[t]), "j": describe(vj[t]),
            "z_cosine": float(per_tier_raw["VeryNear"]["zcos"][t]),
            "c_cosine": float(cc[t]),
            "c_js": float(per_tier_raw["VeryNear"]["cjs"][t]),
        }
        for t in picked
    ]
    # (b) chemically different (heavy-atom radius-1 differs by >=2 edits)
    #     but representation very close
    hv = c1h
    bsel = np.flatnonzero(hv >= 2)
    bmet = metrics(hci[bsel], hcj[bsel])
    border = np.argsort(bmet["ccos"])[::-1][:5]
    cases["hard_negative_but_close"] = [
        {
            "i": describe(hci[bsel[p]]), "j": describe(hcj[bsel[p]]),
            "heavy_c1_edits": float(hv[bsel[p]]),
            "z_cosine": float(bmet["zcos"][p]),
            "c_cosine": float(bmet["ccos"][p]),
            "same_argmax": bool(argmax[hci[bsel[p]]] == argmax[hcj[bsel[p]]]),
        }
        for p in border
    ]
    # (c) same prototype but chemically different assigned patches (Failure A)
    worst = sorted(coh_rows, key=lambda r: r["l1_conc"])[:5]
    fa = []
    for r in worst:
        k = r["prototype"]
        idx = np.flatnonzero(argmax == k)
        if len(idx) < 2:
            continue
        ii, jj = within_pairs(idx, 4000, rng)
        if len(ii) == 0:
            continue
        c1v = c1_dist(ii, jj)
        t = int(np.argmax(c1v))
        fa.append({
            "prototype": k,
            "n": int(len(idx)),
            "l1_conc": r["l1_conc"],
            "key_conc": r["key_conc"],
            "pair_c1_edits": float(c1v[t]),
            "i": describe(ii[t]), "j": describe(jj[t]),
        })
    cases["failure_A_same_prototype_different_chemistry"] = fa
    report["cases"] = cases

    # ---------------- task organization light diagnostics ----------------
    def r2_partition(labels):
        lab = np.asarray(labels)
        y = y_patch
        gm = y.mean()
        sst = ((y - gm) ** 2).sum()
        if sst <= 0:
            return 0.0
        order = np.argsort(lab, kind="stable")
        lab_s = lab[order]
        y_s = y[order]
        # group boundaries
        uniq, start = np.unique(lab_s, return_index=True)
        ssb = 0.0
        for gi_, st in enumerate(start):
            en = start[gi_ + 1] if gi_ + 1 < len(start) else len(lab_s)
            seg = y_s[st:en]
            ssb += len(seg) * (seg.mean() - gm) ** 2
        return float(ssb / sst)

    r2_argmax = r2_partition(argmax)
    r2_l1 = r2_partition(l1id)
    r2_joint = r2_partition(l1id.astype(np.int64) * (K_PROTO + 1) + argmax)
    task = {
        "r2_y_from_argmax_prototype": r2_argmax,
        "r2_y_from_l1_key": r2_l1,
        "r2_y_from_canonical_key": r2_partition(key_id),
        "r2_y_from_root_degree": r2_partition(np.array([int(root[i]) * 100 + int(degree[i]) for i in range(Pn)])),
        "r2_y_from_joint_l1_and_prototype": r2_joint,
        "prototype_increment_beyond_l1_chemistry": float(r2_joint - r2_l1),
        "note": "patches of one molecule share y; R2 is descriptive, not causal",
    }
    # split vs non-split very-near pairs: |dy| between the two molecule targets
    dy = np.abs(y_patch[vi] - y_patch[vj])
    task["very_near_split_dy_mean"] = float(dy[split].mean()) if split.any() else None
    task["very_near_nonsplit_dy_mean"] = float(dy[~split].mean()) if (~split).any() else None
    report["task_organization"] = task

    # ---------------- provenance ----------------
    report["setup"] = {
        "protocol": "tccd_v2_continuity_audit",
        "checkpoint_label": cli.label,
        "checkpoint": str(ckpt_path.relative_to(REPO)) if str(ckpt_path).startswith(str(REPO)) else str(ckpt_path),
        "checkpoint_keys": list(ckpt.keys()),
        "records_cache": str(CACHE.relative_to(REPO)),
        "split": "TCCD internal split seed 20260922 (8000 train / 2000 dev)",
        "n_dev_graphs": len(dev),
        "n_dev_patches": int(Pn),
        "n_unique_canonical_keys": int(n_keys),
        "n_unique_radius1_keys": int(len(l1map)),
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "training_performed": False,
        "tuning_performed": False,
        "seed": SEED,
        "n_target_pairs_per_tier": N_TARGET,
        "tau": tau,
        "validation_active_prototypes": active,
        "validation_effective_prototype_count": eff,
        "commit_local_head": T.git("rev-parse", "HEAD"),
        "wall_s": time.time() - t_start,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1))
    print(f"[done] wrote {out_json} in {time.time()-t_start:.0f}s")
    return report


if __name__ == "__main__":
    main()
