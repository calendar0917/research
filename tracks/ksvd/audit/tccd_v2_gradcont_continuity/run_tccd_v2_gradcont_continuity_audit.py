#!/usr/bin/env python
"""TCCD-v2 GRAD-CONT ordinal-geometry continuity audit (read-only diagnostic).

Central question:
    After an ordinal-only local-geometry regularizer, does ``z`` become a truly
    graded local chemical geometry, or does it merely move a discrete cliff?

The audit is deliberately NOT a single Spearman number. For each checkpoint it
reports, on a *paired* internal-dev pair set:

  * the full ``(d1, d2)`` geometry curve (mean / median Z and C cosine, N);
  * fixed-axis slices (fix ``d1`` sweep ``d2``; fix ``d2`` sweep ``d1``);
  * adjacent-bin jump statistics and the largest-boundary share
    (largest adjacent drop / total near-to-far drop);
  * held-out ordinal-triplet accuracy ``P(cos(i,close) > cos(i,far))``,
    stratified into same_d1 / same_d2 / both;
  * a random-pair baseline and far-bin cosines (the "no abnormal negative
    cosine" guardrail);
  * Z and C separately, plus top-1 prototype agreement.

This script never trains, never refits, never loads the official test or the
official valid split, and never writes formal result files. It reuses the
frozen checkpoints and the canonical TCCD-v0/v1 record cache. The shared
chemistry decode and ordinal definitions live in
``tracks/ksvd/code/tccd_v2_local_chem.py``.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import time

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V
from tracks.ksvd.code import tccd_v2_local_chem as LC
from tracks.ksvd.code.run_tccd_v0 import internal_split

REPO = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parent
CACHE = REPO / "tracks/ksvd/results/tccd_v0/cache/tccd_v0_train_r2_M14_A21_B3_n10000.pkl"
GRADCONT_DIR = REPO / "tracks/ksvd/results/tccd_v2_gradcont"
DEV_TRIPLETS = GRADCONT_DIR / "dev_ordinal_triplets.npz"

SEED = 20260922
BOOT_B = 300
CTYPE_NAMES = {0: "same_d1", 1: "same_d2", 2: "both"}

CHECKPOINTS = {
    "BASE": REPO / "tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt",
    "CHEM-CONT": REPO / "tracks/ksvd/results/tccd_v2_chemcont/prototype_chemcont_seed0_best.pt",
    "GRAD-CONT": GRADCONT_DIR / "prototype_gradcont_seed0_best.pt",
}

# fixed-axis slices for the cliff / smoothness analysis
D1_SLICES = [0, 1, 2]
D2_SLICES = [1, 2]


def spearman(x, y):
    from scipy.stats import spearmanr

    r = spearmanr(x, y)
    return float(r.statistic)


def summarize(vals: np.ndarray) -> dict:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return {"n": 0}
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


def cluster_boot_mean(vals, w1, w2, n_graphs, rng, B=BOOT_B):
    vals = np.asarray(vals, dtype=np.float64)
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    out = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, n_graphs, n_graphs), minlength=n_graphs)
        w = cnt[w1].astype(np.float64) * cnt[w2].astype(np.float64)
        sw = w.sum()
        if sw > 0:
            out.append(float((vals * w).sum() / sw))
    if not out:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def triplet_weights(ga, gc, gk, n_graphs, rng):
    cnt = np.bincount(rng.integers(0, n_graphs, n_graphs), minlength=n_graphs)
    return cnt[ga].astype(np.float64) * cnt[gc].astype(np.float64) * cnt[gk].astype(np.float64)


def cluster_boot_stat_triplet(vals, ga, gc, gk, n_graphs, rng, B=BOOT_B):
    vals = np.asarray(vals, dtype=np.float64)
    out = []
    for _ in range(B):
        w = triplet_weights(ga, gc, gk, n_graphs, rng)
        sw = w.sum()
        if sw > 0:
            out.append(float((vals * w).sum() / sw))
    if not out:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def paired_cluster_boot_delta(correct_a, correct_b, ga, gc, gk, n_graphs, rng, B=BOOT_B):
    d = np.asarray(correct_a, dtype=np.float64) - np.asarray(correct_b, dtype=np.float64)
    out = []
    for _ in range(B):
        w = triplet_weights(ga, gc, gk, n_graphs, rng)
        sw = w.sum()
        if sw > 0:
            out.append(float((d * w).sum() / sw))
    if not out:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


# ---------------------------------------------------------------------------
def load_checkpoint(path: pathlib.Path, X, device_label="cpu"):
    torch = T._torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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
    return {
        "W": W, "P": Pmat, "tau": tau, "Z": Z, "Zn": Zn, "C": C, "Cn": Cn,
        "argmax": argmax, "checkpoint_keys": list(ckpt.keys()),
    }


def cos_rows(A, i, B, j):
    return np.einsum("ij,ij->i", A[i], B[j])


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", default=str(OUT / "tccd_v2_gradcont_continuity_audit.json"))
    ap.add_argument("--target-pairs", type=int, default=1_500_000)
    ap.add_argument("--exact-target", type=int, default=200_000)
    cli = ap.parse_args(argv)
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    rec = T.load_records(CACHE)["records"]
    tr_idx, dev_idx = internal_split(len(rec))
    dev = [int(i) for i in dev_idx]
    print(f"[load] graphs={len(rec)} internal_train={len(tr_idx)} internal_dev={len(dev)}")

    X = np.concatenate([np.asarray(rec[gi]["X"], dtype=np.float32) for gi in dev], axis=0)
    ys = np.concatenate(
        [np.full(int(rec[gi]["n"]), float(rec[gi]["y"]), dtype=np.float64) for gi in dev]
    )
    gid = np.concatenate([np.full(int(rec[gi]["n"]), k, dtype=np.int64) for k, gi in enumerate(dev)])
    Pn = X.shape[0]
    print(f"[load] dev patches={Pn} F={X.shape[1]}")

    # ---------------- chemistry decode & paired dev pair set ----------------
    chem = LC.decode_local_chem(rec, dev, log=print)
    assert int(chem["n_patches"]) == Pn

    pairs = LC.sample_bin_pairs(
        chem, seed=SEED, target_pairs=cli.target_pairs,
        exact_target=cli.exact_target, log=print,
    )
    pa, pj, psame = pairs["anchor"], pairs["partner"], pairs["same_key"]
    d1 = LC.d1_pairs(chem, pa, pj)
    d2 = LC.d2_pairs(chem, pa, pj)
    b1 = LC.bin_d1(d1)
    b2 = LC.bin_d2(d2)
    gpair1 = gid[pa]
    gpair2 = gid[pj]
    print(f"[pairs] n={pa.size} same_key={int(psame.sum())} cross={float((gpair1 != gpair2).mean()):.4f}")

    # ---------------- held-out dev ordinal triplets ----------------
    trip = LC.load_triplets(DEV_TRIPLETS)
    if trip is None:
        print("[triplets] dev triplet cache missing; rebuilding")
        trip = LC.build_ordinal_triplets(chem, seed=LC.ORDINAL_SEED, log=print)
    ta, tc, tk = trip["anchor"], trip["closer"], trip["farther"]
    ctype = trip["ctype"]
    gt_a = gid[ta]
    gt_c = gid[tc]
    gt_k = gid[tk]
    d1c = LC.d1_pairs(chem, ta, tc)
    d2c = LC.d2_pairs(chem, ta, tc)
    d1f = LC.d1_pairs(chem, ta, tk)
    d2f = LC.d2_pairs(chem, ta, tk)
    print(f"[triplets] n={ta.size} types={collections.Counter(CTYPE_NAMES[int(x)] for x in ctype.tolist())}")

    # ---------------- random cross-molecule baseline pair set ----------------
    n_rand = 200_000
    ri = rng.integers(0, Pn, n_rand * 3)
    rj = rng.integers(0, Pn, n_rand * 3)
    ok = gid[ri] != gid[rj]
    ri, rj = ri[ok][:n_rand], rj[ok][:n_rand]

    # ---------------- evaluate every checkpoint ----------------
    results = {}
    per_triplet_correct = {}
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            print(f"[skip] {name}: checkpoint missing at {path}")
            results[name] = {"missing": True, "checkpoint": str(path)}
            continue
        t0 = time.time()
        ck = load_checkpoint(path, X)
        Zn, Cn, argmax = ck["Zn"], ck["Cn"], ck["argmax"]
        zcos_p = cos_rows(Zn, pa, Zn, pj)
        ccos_p = cos_rows(Cn, pa, Cn, pj)
        agree_p = (argmax[pa] == argmax[pj]).astype(np.float64)
        zcos_tc = cos_rows(Zn, ta, Zn, tc)
        zcos_tf = cos_rows(Zn, ta, Zn, tk)
        ccos_tc = cos_rows(Cn, ta, Cn, tc)
        ccos_tf = cos_rows(Cn, ta, Cn, tk)
        zcos_r = cos_rows(Zn, ri, Zn, rj)
        ccos_r = cos_rows(Cn, ri, Cn, rj)
        z_correct = (zcos_tc > zcos_tf).astype(np.float64) + 0.5 * (zcos_tc == zcos_tf)
        c_correct = (ccos_tc > ccos_tf).astype(np.float64) + 0.5 * (ccos_tc == ccos_tf)
        per_triplet_correct[name] = (z_correct, c_correct)
        results[name] = {
            "checkpoint": str(path.relative_to(REPO)),
            "tau": ck["tau"],
            "pairs": {
                "zcos": zcos_p, "ccos": ccos_p, "agree": agree_p,
            },
            "triplets": {
                "zclose": zcos_tc, "zfar": zcos_tf,
                "cclose": ccos_tc, "cfar": ccos_tf,
                "z_correct": z_correct, "c_correct": c_correct,
                "z_margin": zcos_tc - zcos_tf,
                "c_margin": ccos_tc - ccos_tf,
            },
            "random": {"zcos": zcos_r, "ccos": ccos_r},
            "wall_s": time.time() - t0,
        }
        print(f"[eval] {name} tau={ck['tau']:.6f} in {time.time()-t0:.1f}s")

    # ---------------- geometry curve ----------------
    cell_key = b1.astype(np.int64) * 100 + b2.astype(np.int64)
    cells = sorted(set(cell_key.tolist()))
    curve = {}
    for name, r in results.items():
        if r.get("missing"):
            continue
        zc, cc, ag = r["pairs"]["zcos"], r["pairs"]["ccos"], r["pairs"]["agree"]
        row = {}
        for c in cells:
            m = cell_key == c
            if int(m.sum()) == 0:
                continue
            row[f"{c // 100},{c % 100}"] = {
                "n": int(m.sum()),
                "z_mean": float(zc[m].mean()),
                "z_median": float(np.median(zc[m])),
                "c_mean": float(cc[m].mean()),
                "c_median": float(np.median(cc[m])),
                "top1_agree": float(ag[m].mean()),
                "z_mean_ci": cluster_boot_mean(zc[m], gpair1[m], gpair2[m], len(dev), rng),
            }
        curve[name] = row

    # ---------------- fixed-axis slices ----------------
    def bin_label(bb1: int, bb2: int) -> str:
        return f"{bb1},{bb2}"

    slices = {}
    for name, row in curve.items():
        s = {}
        for d1v in D1_SLICES:
            seq = []
            for d2v in range(0, LC.D2_MAX_BIN + 1):
                k = bin_label(d1v, d2v)
                if k in row and row[k]["n"] >= 200:
                    seq.append((k, row[k]))
            s[f"d1={d1v}"] = seq
        for d2v in D2_SLICES:
            seq = []
            for d1v in range(0, LC.D1_MAX_BIN + 1):
                k = bin_label(d1v, d2v)
                if k in row and row[k]["n"] >= 200:
                    seq.append((k, row[k]))
            s[f"d2={d2v}"] = seq
        slices[name] = s

    # ---------------- sub-curves comparable to the frozen audit -----------
    # l1_identical = radius-1 canonical key equal (the audit "VeryNear" family)
    # radius1 = not l1-identical, binned by radius-1 edit count (audit "c1 axis")
    l1eq = chem["l1id"][pa] == chem["l1id"][pj]
    print(f"[pairs] l1_identical={int(l1eq.sum())} ({float(l1eq.mean()):.4f})")

    def _cell_stats(mask, zc, cc, ag):
        return {
            "n": int(mask.sum()),
            "z_mean": float(zc[mask].mean()),
            "z_median": float(np.median(zc[mask])),
            "c_mean": float(cc[mask].mean()),
            "c_median": float(np.median(cc[mask])),
            "top1_agree": float(ag[mask].mean()),
            "z_mean_ci": cluster_boot_mean(zc[mask], gpair1[mask], gpair2[mask], len(dev), rng),
        }

    subcurves = {}
    for name, r in results.items():
        if r.get("missing"):
            continue
        zc, cc, ag = r["pairs"]["zcos"], r["pairs"]["ccos"], r["pairs"]["agree"]
        lid = {}
        for d2v in range(0, LC.D2_MAX_BIN + 1):
            m = l1eq & (b2 == d2v)
            if int(m.sum()) >= 100:
                lid[f"d2={d2v}"] = _cell_stats(m, zc, cc, ag)
        r1 = {}
        for d1v in range(0, LC.D1_MAX_BIN + 1):
            m = (~l1eq) & (b1 == d1v)
            if int(m.sum()) >= 100:
                r1[f"c1={d1v}"] = _cell_stats(m, zc, cc, ag)
        subcurves[name] = {"l1_identical_by_d2": lid, "radius1_by_c1": r1}

    # ---------------- adjacent-bin jumps & largest-boundary share --------
    def jump_metrics(seq, value_key="z_mean"):
        vals = [b[value_key] for _, b in seq]
        labels = [k for k, _ in seq]
        if len(seq) < 2:
            return {"n_bins": len(seq), "labels": labels, "values": vals}
        vals_a = np.asarray(vals, dtype=np.float64)
        drops = vals_a[:-1] - vals_a[1:]
        total = float(vals_a[0] - vals_a[-1])
        largest = float(np.max(drops))
        largest_idx = int(np.argmax(drops))
        return {
            "n_bins": len(seq),
            "labels": labels,
            "values": [float(v) for v in vals],
            "adjacent_drops": [float(x) for x in drops],
            "adjacent_jumps_abs": [float(abs(x)) for x in drops],
            "total_drop": total,
            "largest_drop": largest,
            "largest_drop_boundary": f"{labels[largest_idx]}->{labels[largest_idx+1]}",
            "median_abs_jump": float(np.median(np.abs(drops))),
            "max_abs_jump": float(np.max(np.abs(drops))),
            "largest_boundary_share": float(largest / total) if total > 1e-9 else None,
            "monotone_nonincreasing": bool(np.all(drops >= -1e-9)),
            "spearman_bin_vs_value": spearman(np.arange(len(vals_a)), vals_a) if len(seq) >= 3 else None,
        }

    jumps = {}
    for name, s in slices.items():
        jm = {}
        for sk, seq in s.items():
            jm[sk] = {"z": jump_metrics(seq, "z_mean"), "c": jump_metrics(seq, "c_mean")}
        for sub_name in ("l1_identical_by_d2", "radius1_by_c1"):
            sub = subcurves.get(name, {}).get(sub_name, {})
            keys = sorted(sub.keys(), key=lambda k: int(k.split("=")[1]))
            seq = [(k, sub[k]) for k in keys]
            jm[f"sub::{sub_name}"] = {"z": jump_metrics(seq, "z_mean"), "c": jump_metrics(seq, "c_mean")}
        # aggregate across slices (equal weight, >=2 bins)
        allz = [m["z"]["adjacent_jumps_abs"] for m in jm.values() if m["z"].get("adjacent_jumps_abs")]
        shares = [m["z"]["largest_boundary_share"] for m in jm.values() if m["z"].get("largest_boundary_share") is not None]
        flat = [x for xs in allz for x in xs]
        jm["_aggregate"] = {
            "n_slices": len(jm) - 1,
            "median_abs_jump": float(np.median(flat)) if flat else None,
            "max_abs_jump": float(np.max(flat)) if flat else None,
            "mean_largest_boundary_share": float(np.mean(shares)) if shares else None,
            "max_largest_boundary_share": float(np.max(shares)) if shares else None,
        }
        jumps[name] = jm

    # ---------------- ordinal accuracy ----------------
    ordinal = {}
    for name, r in results.items():
        if r.get("missing"):
            continue
        zc = r["triplets"]["z_correct"]
        cc = r["triplets"]["c_correct"]
        grp = {"all": np.ones(ta.size, dtype=bool)}
        for code, nm in CTYPE_NAMES.items():
            grp[nm] = ctype == code
        # a few top (closer_bin -> farther_bin) comparisons
        bc1 = LC.bin_d1(d1c) * 100 + LC.bin_d2(d2c)
        bf1 = LC.bin_d1(d1f) * 100 + LC.bin_d2(d2f)
        cmp_key = [f"{a//100},{a%100}->{b//100},{b%100}" for a, b in zip(bc1.tolist(), bf1.tolist())]
        cmp_counts = collections.Counter(cmp_key)
        top_cmp = [k for k, _ in cmp_counts.most_common(6)]
        entry = {}
        for gname, mask in grp.items():
            entry[gname] = {
                "n": int(mask.sum()),
                "z_accuracy": float(zc[mask].mean()) if mask.any() else None,
                "c_accuracy": float(cc[mask].mean()) if mask.any() else None,
                "z_accuracy_ci": cluster_boot_stat_triplet(zc[mask], gt_a[mask], gt_c[mask], gt_k[mask], len(dev), rng)
                if mask.sum() > 100 else None,
                "c_accuracy_ci": cluster_boot_stat_triplet(cc[mask], gt_a[mask], gt_c[mask], gt_k[mask], len(dev), rng)
                if mask.sum() > 100 else None,
                "z_margin_mean": float(np.mean(r["triplets"]["z_margin"][mask])) if mask.any() else None,
                "c_margin_mean": float(np.mean(r["triplets"]["c_margin"][mask])) if mask.any() else None,
            }
        per_cmp = {}
        for k in top_cmp:
            mask = np.asarray([x == k for x in cmp_key])
            per_cmp[k] = {
                "n": int(mask.sum()),
                "z_accuracy": float(zc[mask].mean()),
                "c_accuracy": float(cc[mask].mean()),
            }
        entry["_top_comparisons"] = per_cmp
        entry["_spearman"] = {
            "d1_plus_d2_vs_zcos_cos": spearman((d1c + d2c).astype(np.float64), r["triplets"]["zclose"]),
            "d2_within_d1zero_vs_zcos": spearman(d2c[d1c == 0].astype(np.float64), r["triplets"]["zclose"][d1c == 0])
            if int((d1c == 0).sum()) > 10 else None,
            "d1_within_d2one_vs_zcos": spearman(d1c[d2c == 1].astype(np.float64), r["triplets"]["zclose"][d2c == 1])
            if int((d2c == 1).sum()) > 10 else None,
        }
        ordinal[name] = entry

    # paired deltas: GRAD-CONT vs others
    paired = {}
    if "GRAD-CONT" in per_triplet_correct and not results["GRAD-CONT"].get("missing"):
        gz, gc_ = per_triplet_correct["GRAD-CONT"]
        for other in ("BASE", "CHEM-CONT"):
            if other not in per_triplet_correct:
                continue
            oz, oc = per_triplet_correct[other]
            paired[f"GRAD-CONT_vs_{other}"] = {
                "z_accuracy_delta": float(gz.mean() - oz.mean()),
                "z_accuracy_delta_ci": paired_cluster_boot_delta(gz, oz, gt_a, gt_c, gt_k, len(dev), rng),
                "c_accuracy_delta": float(gc_.mean() - oc.mean()),
                "c_accuracy_delta_ci": paired_cluster_boot_delta(gc_, oc, gt_a, gt_c, gt_k, len(dev), rng),
                "z_by_type": {
                    nm: float((gz[ctype == code] - oz[ctype == code]).mean())
                    for code, nm in CTYPE_NAMES.items()
                },
            }

    # ---------------- far-bin / random guardrail ----------------
    guardrail = {}
    for name, r in results.items():
        if r.get("missing"):
            continue
        zc, cc = r["pairs"]["zcos"], r["pairs"]["ccos"]
        far = (b1 >= max(1, LC.D1_MAX_BIN - 1)) | (b2 >= LC.D2_MAX_BIN - 1)
        vnear = (b1 == 0) & (b2 == 1)
        big_cells = [c for c in cells if int((cell_key == c).sum()) >= 200]
        cell_means = [(f"{c//100},{c%100}", float(zc[cell_key == c].mean())) for c in big_cells]
        guardrail[name] = {
            "random_z_mean": float(r["random"]["zcos"].mean()),
            "random_c_mean": float(r["random"]["ccos"].mean()),
            "far_bin_n": int(far.sum()),
            "far_bin_z_mean": float(zc[far].mean()) if far.any() else None,
            "far_bin_c_mean": float(cc[far].mean()) if far.any() else None,
            "verynear_d1_0_d2_1_z_mean": float(zc[vnear].mean()) if vnear.any() else None,
            "min_bin_z_mean": min(cell_means, key=lambda x: x[1])[1] if cell_means else None,
            "min_bin_label": min(cell_means, key=lambda x: x[1])[0] if cell_means else None,
            "n_bins_with_negative_z_mean": int(sum(1 for _, m in cell_means if m < 0)),
            "n_bins_ge_200": len(big_cells),
        }

    # ---------------- exact / verynear / verynear-ish summary ----------------
    summary = {}
    for name, r in results.items():
        if r.get("missing"):
            continue
        zc, cc, ag = r["pairs"]["zcos"], r["pairs"]["ccos"], r["pairs"]["agree"]
        ex = psame == 1
        z00 = (b1 == 0) & (b2 == 0)
        vn = (b1 == 0) & (b2 == 1)
        summary[name] = {
            "exact_same_key_z_mean": float(zc[ex].mean()) if ex.any() else None,
            "exact_same_key_c_mean": float(cc[ex].mean()) if ex.any() else None,
            "bin_0_0_z_mean": float(zc[z00].mean()) if z00.any() else None,
            "bin_0_0_n": int(z00.sum()),
            "verynear_z_mean": float(zc[vn].mean()) if vn.any() else None,
            "verynear_c_mean": float(cc[vn].mean()) if vn.any() else None,
            "verynear_top1_agree": float(ag[vn].mean()) if vn.any() else None,
            "exact_to_verynear_cliff_z_samekey": float(zc[ex].mean() - zc[vn].mean())
            if ex.any() and vn.any() else None,
            "exact_to_verynear_cliff_z_bin00": float(zc[z00].mean() - zc[vn].mean())
            if z00.any() and vn.any() else None,
            "exact_to_verynear_cliff_c_samekey": float(cc[ex].mean() - cc[vn].mean())
            if ex.any() and vn.any() else None,
        }

    # ---------------- MAE guardrail (read from existing result JSONs) -------
    mae = {}
    base_train = json.loads((REPO / "tracks/ksvd/results/tccd_v2/gateA_seed0.json").read_text())
    mae["BASE"] = {
        "best_valid": base_train.get("best_valid"),
        "soup_valid": base_train.get("soup_valid"),
        "source": "tracks/ksvd/results/tccd_v2/gateA_seed0.json",
    }
    chem_train = json.loads((REPO / "tracks/ksvd/results/tccd_v2_chemcont/train_seed0.json").read_text())
    mae["CHEM-CONT"] = {
        "best_valid": chem_train.get("best_valid"),
        "soup_valid": chem_train.get("soup_valid"),
        "source": "tracks/ksvd/results/tccd_v2_chemcont/train_seed0.json",
    }
    gc_train_path = GRADCONT_DIR / "train_seed0.json"
    if gc_train_path.exists():
        gc_train = json.loads(gc_train_path.read_text())
        mae["GRAD-CONT"] = {
            "best_valid": gc_train.get("best_valid"),
            "soup_valid": gc_train.get("soup_valid"),
            "source": "tracks/ksvd/results/tccd_v2_gradcont/train_seed0.json",
        }

    # ---------------- provenance ----------------
    out = {
        "setup": {
            "protocol": "tccd_v2_gradcont_continuity_audit",
            "records_cache": str(CACHE.relative_to(REPO)),
            "split": "TCCD internal split seed 20260922 (8000 train / 2000 dev)",
            "n_dev_graphs": len(dev),
            "n_dev_patches": int(Pn),
            "n_unique_canonical_keys": int(chem["n_unique_keys"]),
            "n_unique_radius1_keys": int(chem["n_unique_l1"]),
            "official_test_loaded": False,
            "official_valid_loaded": False,
            "training_performed": False,
            "tuning_performed": False,
            "seed": SEED,
            "per_anchor_pairs": None,
            "target_pairs": cli.target_pairs,
            "exact_target": cli.exact_target,
            "pair_weighting": "uniform_over_pairs",
            "checkpoints": {k: str(v.relative_to(REPO)) for k, v in CHECKPOINTS.items()},
            "dev_triplets_source": str(DEV_TRIPLETS.relative_to(REPO)),
            "commit_local_head": T.git("rev-parse", "HEAD"),
            "wall_s": time.time() - t_start,
        },
        "pair_set": pairs["stats"],
        "geometry_curve": curve,
        "subcurves": subcurves,
        "slices": {
            name: {sk: [{"bin": k, **b} for k, b in seq] for sk, seq in s.items()}
            for name, s in slices.items()
        },
        "jumps": jumps,
        "ordinal": ordinal,
        "paired": paired,
        "guardrail": guardrail,
        "summary": summary,
        "mae": mae,
    }
    pathlib.Path(cli.out_json).write_text(json.dumps(out, indent=1, default=float))
    print(f"[done] wrote {cli.out_json} in {time.time()-t_start:.0f}s")
    return out


if __name__ == "__main__":
    main()
