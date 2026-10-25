"""Why is T redundant?  Label-free probes.

1. How much of T is explained by the size block alone, and does the bond sum z
   still add after the size block is regressed out?
2. Is the redundancy an artefact of one random ``z_proj``?  Re-draw the
   projection (composer + reader fixed) and repeat.
3. Are the eigenvectors of B nearly molecule-independent?  If they are, then
   T = sum_k lambda_k (u_k . Z)^2 is close to a *fixed* quadratic function of
   z = sum_e h, which is exactly what the linear probe finds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rcc_common as C  # noqa: E402
import rcc_kernels as K  # noqa: E402
import rcc_models as M  # noqa: E402
import ad_data as D  # noqa: E402


def lin_r2(Xtr, Ytr, Xte, Yte, lam=1e-6):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Atr = torch.cat([(Xtr - mu) / sd, torch.ones(len(Xtr), 1, dtype=torch.float64)], 1)
    Ate = torch.cat([(Xte - mu) / sd, torch.ones(len(Xte), 1, dtype=torch.float64)], 1)
    W = torch.linalg.solve(Atr.T @ Atr + lam * torch.eye(Atr.shape[1], dtype=torch.float64),
                           Atr.T @ Ytr)
    pred = Ate @ W
    per_entry = 1.0 - ((Yte - pred) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
    frob = 1.0 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
    return float(per_entry.mean()), float(frob)


def main() -> int:
    cache = C.load_train_cache()
    split = C.load_split()
    kpad, struct = C.load_bank()
    collate = C.make_collate((kpad, struct))
    norm = D.load_norm()
    fit = split["cache_rows"]["fit"]
    model, info = M.build_initial_state(norm, 0)

    rows = fit[:5000]
    X, Y, YN = [], [], []
    with torch.no_grad():
        for s in range(0, len(rows), 128):
            b = collate(cache, rows[s:s + 128])
            cap = model.compute(b)
            X.append(cap["x"][:, :422].double())
            Y.append(cap["T"].double())
            YN.append(model.compute(b, mode="null")["T"].double())
    X, Y, YN = torch.cat(X), torch.cat(Y), torch.cat(YN)
    ntr = 4000
    out: dict = {"protocol": "rcc-v0", "git_commit": C.git_commit(),
                 "config_sha256": C.cfg_sha256(), "git_dirty": C.git_dirty(),
                 "n_train": ntr, "n_test": len(X) - ntr,
                 "labels_read": False, "official_valid_loaded": False,
                 "official_test_loaded": False}

    def blocks(cols):
        return X[:, cols]

    sets = {
        "struct_only": list(range(417, 422)),
        "z_only": list(range(0, 96)),
        "z2_only": list(range(96, 192)),
        "z_plus_z2": list(range(0, 192)),
        "all_reader_except_T": list(range(0, 422)),
    }
    out["linear_r2"] = {}
    for name, cols in sets.items():
        Xb = blocks(cols)
        for tag, T in (("real", Y), ("null", YN)):
            mean_r2, frob = lin_r2(Xb[:ntr], T[:ntr], Xb[ntr:], T[ntr:])
            out["linear_r2"].setdefault(name, {})[tag] = {
                "mean_per_entry": mean_r2, "frobenius": frob}

    # residualise both sides on the size block, then ask what z adds
    S = X[:, 417:422]
    def resid(A, Z):
        mu, sd = S[:ntr].mean(0), S[:ntr].std(0).clamp(min=1e-6)
        Mx = torch.cat([(S - mu) / sd, torch.ones(len(S), 1, dtype=torch.float64)], 1)
        W = torch.linalg.solve(Mx[:ntr].T @ Mx[:ntr] + 1e-6 * torch.eye(6, dtype=torch.float64),
                               Mx[:ntr].T @ Z[:ntr])
        return Z - Mx @ W
    res = {}
    for tag, T in (("real", Y), ("null", YN)):
        Tz = resid(X, T)
        Zz = resid(X, X[:, :96])
        mean_r2, frob = lin_r2(Zz[:ntr], Tz[:ntr], Zz[ntr:], Tz[ntr:])
        res[tag] = {"mean_per_entry": mean_r2, "frobenius": frob}
    out["after_removing_size_block"] = {"predictor": "z_only", **res}

    # the only thing the real kernel adds over the null is T_real - T_null:
    # can the existing reader blocks reproduce *that*?
    diff = {}
    for name, cols in (("z_only", list(range(0, 96))),
                       ("z2_only", list(range(96, 192))),
                       ("z_plus_z2", list(range(0, 192))),
                       ("all_reader_except_T", list(range(0, 422)))):
        Xb = X[:, cols]
        Dt = Y - YN
        a, b_ = lin_r2(Xb[:ntr], Dt[:ntr], Xb[ntr:], Dt[ntr:])
        diff[name] = {"mean_per_entry": a, "frobenius": b_}
    out["real_minus_null_explained"] = diff

    # re-draw z_proj only
    redraw = {}
    for seed in (11, 12, 13):
        m2, _ = M.build_initial_state(norm, 0)
        torch.manual_seed(seed)
        with torch.no_grad():
            m2.z_proj.weight.normal_(0.0, 1.0 / np.sqrt(96))
            m2.z_proj.bias.zero_()
        X2, Y2, Y2n = [], [], []
        with torch.no_grad():
            for s in range(0, len(rows), 128):
                b = collate(cache, rows[s:s + 128])
                cap = m2.compute(b)
                X2.append(cap["x"][:, :422].double())
                Y2.append(cap["T"].double())
                Y2n.append(m2.compute(b, mode="null")["T"].double())
        X2, Y2, Y2n = torch.cat(X2), torch.cat(Y2), torch.cat(Y2n)
        e = {}
        for tag, T in (("real", Y2), ("null", Y2n)):
            a, b_ = lin_r2(X2[:ntr], T[:ntr], X2[ntr:], T[ntr:])
            e[tag] = {"mean_per_entry": a, "frobenius": b_}
        a, b_ = lin_r2(X2[:ntr, :96], Y2[:ntr], X2[ntr:, :96], Y2[ntr:])
        e["real_from_z_only"] = {"mean_per_entry": a, "frobenius": b_}
        redraw[str(seed)] = e
    out["zproj_redraw"] = redraw

    # how universal are the eigenvectors of B?
    kpad_np = kpad
    bptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    vecs = {}
    rng = np.random.default_rng(0)
    for r in rng.choice(fit[:3000], size=400, replace=False):
        n = int(bptr[r + 1] - bptr[r])
        Kr = K.build_kernel(
            np.asarray(cache["bond_index"], dtype=np.int64)[int(bptr[r]):int(bptr[r + 1])],
            int(cache["atom_ptr"][r + 1] - cache["atom_ptr"][r]),
        )[0]
        Kd, _, _ = K.null_from_row_sums(Kr)
        Bm = Kr - Kd
        ev, evec = np.linalg.eigh(0.5 * (Bm + Bm.T))
        order = np.argsort(-np.abs(ev))
        for k in range(3):
            v = evec[:, order[k]]
            v = v / (np.linalg.norm(v) + 1e-30)
            vecs.setdefault(k, []).append((n, v))
    uni = {}
    for k, lst in vecs.items():
        coss = []
        for _ in range(2000):
            i, j = rng.integers(0, len(lst), 2)
            ni, vi = lst[i]
            nj, vj = lst[j]
            L = min(ni, nj)
            if L < 4:
                continue
            c = abs(float(vi[:L] @ vj[:L]) / (np.linalg.norm(vi[:L]) * np.linalg.norm(vj[:L]) + 1e-30))
            coss.append(c)
        uni[f"eigvec_{k}"] = {"n_pairs": len(coss), "mean_abs_cos": float(np.mean(coss)),
                              "median_abs_cos": float(np.median(coss)),
                              "p05": float(np.percentile(coss, 5))}
    out["eigenvector_universality"] = uni

    # --- mechanism: does the contrast operator collapse to the centring
    # operator (trace / second-moment) up to one scalar per molecule?
    collapse = []
    coher = []
    for r in rng.choice(fit[:3000], size=400, replace=False):
        n = int(bptr[r + 1] - bptr[r])
        Kr = K.build_kernel(
            np.asarray(cache["bond_index"], dtype=np.int64)[int(bptr[r]):int(bptr[r + 1])],
            int(cache["atom_ptr"][r + 1] - cache["atom_ptr"][r]),
        )[0]
        Kd, _, _ = K.null_from_row_sums(Kr)
        Bm = Kr - Kd
        Cc = np.eye(n) - np.ones((n, n)) / n
        kappa = float((Bm * Cc).sum() / (Cc * Cc).sum())
        Bap = kappa * Cc
        Zg = np.random.default_rng(int(r)).normal(size=(n, C.RANK))
        Tt = K.qform(Zg, Bm)
        Ta = K.qform(Zg, Bap)
        collapse.append(float(np.linalg.norm(Tt - Ta) / max(np.linalg.norm(Tt), 1e-30)))
        # coherence of a random per-bond vector across adjacent bonds
        u = np.random.default_rng(int(r) + 1).normal(size=n)
        uc = u - u.mean()
        A = K.line_graph_adjacency(
            np.asarray(cache["bond_index"], dtype=np.int64)[int(bptr[r]):int(bptr[r + 1])])
        num = float(uc @ A @ uc)
        den = float((A.sum() / 2.0) * (uc @ uc) / max(n, 1))
        coher.append(num / max(den, 1e-30))
    # how much of the per-molecule bond scatter is *deviation* rather than the
    # common molecule-level offset?  This is the quantity that decides how much
    # independent information a quadratic readout can carry.
    dev_share = []
    with torch.no_grad():
        for s_ in range(0, 1280, 128):
            b = collate(cache, rows[s_:s_ + 128])
            cap = model.compute(b)
            hh = cap["h"]
            bb = b["bond_batch"]
            Bn = int(b["n_graphs"])
            m1 = hh.new_zeros((Bn, 96)).index_add_(0, bb, hh)
            cnt = hh.new_zeros(Bn).index_add_(0, bb, torch.ones_like(bb, dtype=hh.dtype))
            mean = m1 / cnt.clamp(min=1).unsqueeze(1)
            dev = hh - mean[bb]
            s_tot = hh.new_zeros(Bn).index_add_(0, bb, hh.pow(2).sum(1))
            s_dev = hh.new_zeros(Bn).index_add_(0, bb, dev.pow(2).sum(1))
            for i in range(Bn):
                dev_share.append(float(s_dev[i] / max(s_tot[i], 1e-30)))
    out["bond_scatter_deviation_share"] = {
        "n_molecules": len(dev_share),
        "mean": float(np.mean(dev_share)),
        "median": float(np.median(dev_share)),
        "p05": float(np.percentile(dev_share, 5)),
        "p95": float(np.percentile(dev_share, 95)),
        "note": "sum_e |h_e - mean_e(h)|^2 / sum_e |h_e|^2 at the shared initial "
                "state; a quadratic form of the bond vectors can only see the "
                "deviation part, so this bounds how independent a graph-level "
                "second-order channel can be from the bond first moment",
    }
    # effective dimension of the per-bond content (why bilinear functionals of h
    # carry almost nothing beyond its first two moments)
    with torch.no_grad():
        hs = []
        for s_ in range(0, 2560, 128):
            b = collate(cache, rows[s_:s_ + 128])
            hs.append(model.compute(b)["h"].double())
        H = torch.cat(hs)
    Hc = H - H.mean(0)
    cov = (Hc.T @ Hc) / max(len(Hc) - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    ev = torch.flip(ev, (0,))
    part = float((ev.sum() ** 2) / (ev.pow(2).sum()))
    share = ev.cumsum(0) / ev.sum()
    out["bond_content_spectrum"] = {
        "n_bonds": int(H.shape[0]),
        "participation_ratio": part,
        "top1_share": float(share[0]),
        "top5_share": float(share[4]),
        "top10_share": float(share[9]),
        "top20_share": float(share[19]),
        "note": "PCA of the shared bond representation h at the shared initial "
                "state; a low participation ratio means bilinear functionals of h "
                "(which is what T is) are nearly determined by the first two "
                "moments of a handful of directions",
    }
    out["trace_collapse"] = {
        "n": len(collapse),
        "median_rel_fro_diff_T_true_vs_kappa_centring": float(np.median(collapse)),
        "mean": float(np.mean(collapse)),
        "p95": float(np.percentile(collapse, 95)),
        "note": "T computed with the real B versus T computed with the best "
                "single-scalar multiple of the centring operator I - 11^T/n; a "
                "small number means the kernel's off-diagonal structure barely "
                "matters and the contraction is a trace/second-moment operator",
    }
    out["bond_deviation_coherence"] = {
        "n": len(coher),
        "median": float(np.median(coher)),
        "mean": float(np.mean(coher)),
        "note": "normalised quadratic form u_c^T A_L u_c of a centred random "
                "per-bond vector over the line graph; 0 = incoherent deviations, "
                "1 = as coherent as possible.  Incoherent deviations are exactly "
                "the regime in which a dense quadratic form collapses to a scalar "
                "times the second moment.",
    }
    out["eigenvector_universality_note"] = (
        "raw cosine of the same-index eigenvector of B across two molecules, "
        "truncated to the shorter bond list; high values mean the contrast "
        "operator acts along nearly molecule-independent directions, which makes "
        "T close to a fixed quadratic function of the bond sum"
    )
    C.write_json(C.ANALYSIS / "redundancy_probe.json", out)
    print(json.dumps(out["linear_r2"], indent=1))
    print(json.dumps(out["after_removing_size_block"], indent=1))
    print(json.dumps(out["zproj_redraw"], indent=1))
    print(json.dumps(uni, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
