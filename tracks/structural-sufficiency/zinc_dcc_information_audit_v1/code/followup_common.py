"""Shared loaders and diagnostics for the DCC H-C follow-up.

Only official-train structures are touched.  The label accessor below is
restricted to ProbeFit rows; AuditEval is forbidden and official valid/test are
never opened.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_REPO = _TRACK.parents[2]
_V1 = _REPO / "tracks" / "structural-sufficiency" / "zinc_arch_discovery_v1" / "code"
_P1E = _REPO / "tracks" / "structural-sufficiency" / "code" / "phase1e"
for _p in (str(_HERE), str(_V1), str(_P1E), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_data as D  # noqa: E402


class ProbeLabels:
    """Label accessor that allows only the frozen ProbeFit row ids."""

    def __init__(self, arr, allowed):
        self.arr = np.asarray(arr, dtype=np.float64)
        self.allowed = set(int(x) for x in allowed)
        self.log = []

    def __getitem__(self, idx):
        rows = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        bad = [int(x) for x in rows if int(x) not in self.allowed]
        if bad:
            raise RuntimeError(f"forbidden label access (not ProbeFit): {bad[:5]}")
        self.log.append(rows.copy())
        return self.arr[rows]


def load_split():
    return json.loads((_TRACK / "configs" / "split_fit8k_probe_audit.json").read_text())


def load_config():
    return json.loads((_TRACK / "configs" / "followup_diagnostic_v1.json").read_text())


def sha_array(a) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def load_jhat_predictions(seed: int, z: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name in ("raw", "latent"):
        rec = np.load(_TRACK / "analysis" / "b" / f"s{seed}" / f"jhat_{name}.npz")
        X = (np.asarray(z, dtype=np.float64) - rec["xmu"]) / rec["xsd"]
        Y = X @ rec["coef"].T + rec["intercept"]
        out[name] = Y * rec["ysd"] + rec["ymu"]
    return out


def load_probe_context(seed: int, cfg: dict) -> dict:
    split = load_split()
    probe = np.asarray(split["rows"]["probe"], dtype=np.int64)
    fit = np.asarray(split["rows"]["fit"], dtype=np.int64)
    audit = np.asarray(split["rows"]["audit"], dtype=np.int64)
    cache = D.load_cache("train")
    labels = np.asarray(cache.pop("labels"), dtype=np.float64)
    guard = ProbeLabels(labels, allowed=probe)
    y = guard[probe]
    bdir = _TRACK / "analysis" / "b" / f"s{seed}"
    z = np.load(bdir / "z_all.npy")[probe].astype(np.float64)
    raw = np.load(bdir / "pred_all.npy")[probe].astype(np.float64)
    jlat = np.load(bdir / "j_latent_all.npy")[probe].astype(np.float64)
    jraw = np.load(_TRACK / "analysis" / "j_raw.npy")[probe].astype(np.float64)
    jhat = load_jhat_predictions(seed, z)
    mask = np.asarray(cfg["j_raw"]["valid_indices"], dtype=np.int64)
    jperp_raw = (jraw - jhat["raw"])[:, mask]
    jperp_lat = jlat - jhat["latent"]
    support = np.load(bdir / "support_audit.npz", allow_pickle=False)
    # support_audit.npz is in AuditEval order; not used for ProbeFit strata.
    return {
        "seed": int(seed),
        "probe": probe,
        "fit": fit,
        "audit": audit,
        "y": y,
        "z": z,
        "raw": raw,
        "jraw": jraw,
        "jlat": jlat,
        "jhat_raw": jhat["raw"],
        "jhat_latent": jhat["latent"],
        "jperp_raw": jperp_raw,
        "jperp_latent": jperp_lat,
        "g": z[:, 288:321],
        "guard": guard,
        "cache": cache,
        "support_audit": support,
    }


def std_fit(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(0)
    sd = x.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def std_apply(x, mu, sd):
    return (np.asarray(x, dtype=np.float64) - mu) / sd


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def medae(y, p):
    return float(np.median(np.abs(np.asarray(y) - np.asarray(p))))


def top_error_rows(y, p, rows, k=10):
    e = np.abs(np.asarray(y) - np.asarray(p))
    order = np.argsort(e)[::-1]
    top = order[:k]
    return {
        "top_rows": [int(rows[i]) for i in top],
        "top_errors": [float(e[i]) for i in top],
        "top_share_of_mae": float(e[top].sum() / max(e.sum(), 1e-300)),
        "mae_without_top": float(np.delete(e, top).mean()),
        "mae_all": float(e.mean()),
    }


def top_gain_rows(y, pa, pb, rows, k=10):
    """Gain of pb over pa, positive means pb is better."""
    ga = np.abs(np.asarray(y) - np.asarray(pa))
    gb = np.abs(np.asarray(y) - np.asarray(pb))
    g = ga - gb
    order = np.argsort(g)[::-1]
    top = order[:k]
    return {
        "top_rows": [int(rows[i]) for i in top],
        "top_gains": [float(g[i]) for i in top],
        "top10_gain_contribution": float(g[top].sum() / len(g)),
        "gain_without_top10": float(np.delete(g, top).mean()),
        "gain_all": float(g.mean()),
    }


def boot_ci(values, n_boot=10000, seed=20261103):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    means = v[idx].mean(1)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "fraction_positive": float(np.mean(means > 0)),
    }


def boot_gain(y, pa, pb, n_boot=10000, seed=20261103):
    y = np.asarray(y)
    ea = np.abs(y - np.asarray(pa))
    eb = np.abs(y - np.asarray(pb))
    return boot_ci(ea - eb, n_boot=n_boot, seed=seed)


def huber_fit(X, y, alpha, cfg):
    from sklearn.linear_model import HuberRegressor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = HuberRegressor(
            epsilon=float(cfg["huber_epsilon"]),
            alpha=float(alpha),
            max_iter=int(cfg["huber_max_iter"]),
            tol=float(cfg["huber_tol"]),
        )
        m.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    n_iter = int(getattr(m, "n_iter_", -1))
    return m, {
        "n_iter": n_iter,
        "max_iter": int(cfg["huber_max_iter"]),
        "converged": bool(n_iter >= 0 and n_iter < int(cfg["huber_max_iter"])),
        "warnings": [str(w.message)[:240] for w in caught],
    }


def lad_fit(X, y, alpha, cfg):
    from sklearn.linear_model import QuantileRegressor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = QuantileRegressor(
            quantile=float(cfg.get("quantile", 0.5)),
            alpha=float(alpha),
            solver=str(cfg.get("solver", "highs")),
        )
        m.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    n_iter = int(getattr(m, "n_iter_", -1))
    pred = np.asarray(m.predict(np.asarray(X, dtype=np.float64)))
    return m, {
        "n_iter": n_iter,
        "solver": str(cfg.get("solver", "highs")),
        "quantile": float(cfg.get("quantile", 0.5)),
        "finite": bool(np.all(np.isfinite(pred))),
        "warnings": [str(w.message)[:240] for w in caught],
    }


def lad_predict(m, X, xmu, xsd, ymu, ysd, qlo, qhi, clip=True):
    Xs = std_apply(X, xmu, xsd)
    pred = (Xs @ m.coef_ + m.intercept_) * ysd + ymu
    if clip:
        pred = np.clip(pred, qlo, qhi)
    return np.asarray(pred, dtype=np.float64)


def fit_lad_with_standardization(X, y, alpha, cfg):
    xmu, xsd = std_fit(X)
    ymu, ysd = std_fit(y)
    Xs = std_apply(X, xmu, xsd)
    ys = std_apply(y, ymu, ysd)
    m, info = lad_fit(Xs, ys, alpha, cfg)
    return m, {"xmu": xmu, "xsd": xsd, "ymu": ymu, "ysd": ysd, "fit": info}


def huber_predict(m, X, xmu, xsd, ymu, ysd, qlo, qhi, clip=True):
    Xs = std_apply(X, xmu, xsd)
    pred = (Xs @ m.coef_ + m.intercept_) * ysd + ymu
    if clip:
        pred = np.clip(pred, qlo, qhi)
    return np.asarray(pred, dtype=np.float64)


def fit_huber_with_standardization(X, y, alpha, cfg):
    xmu, xsd = std_fit(X)
    ymu, ysd = std_fit(y)
    Xs = std_apply(X, xmu, xsd)
    ys = std_apply(y, ymu, ysd)
    m, info = huber_fit(Xs, ys, alpha, cfg)
    return m, {"xmu": xmu, "xsd": xsd, "ymu": ymu, "ysd": ysd, "fit": info}


def select_alpha_inner(X, y, cfg, base_seed):
    """Nested selection: only inner training folds choose alpha."""
    from sklearn.model_selection import KFold

    alphas = [float(a) for a in cfg["alphas"]]
    inner = KFold(
        n_splits=int(cfg["inner_folds"]),
        shuffle=True,
        random_state=int(base_seed) % (2**32 - 1),
    )
    scores = {a: [] for a in alphas}
    for tr, va in inner.split(X):
        for a in alphas:
            m, pack = fit_lad_with_standardization(X[tr], y[tr], a, cfg)
            qlo, qhi = np.quantile(y[tr], cfg["clip_quantiles"])
            pred = lad_predict(m, X[va], pack["xmu"], pack["xsd"], pack["ymu"],
                               pack["ysd"], qlo, qhi)
            scores[a].append(mae(y[va], pred))
    means = {a: float(np.mean(v)) for a, v in scores.items()}
    # Tie-break toward more regularization (larger alpha).
    best = min(alphas, key=lambda a: (means[a], -a))
    return float(best), {"alpha_scores_mae": means, "inner_folds": int(cfg["inner_folds"])}


def fit_predict_outer(Xtr, ytr, Xva, cfg, base_seed):
    alpha, info = select_alpha_inner(Xtr, ytr, cfg, base_seed)
    m, pack = fit_lad_with_standardization(Xtr, ytr, alpha, cfg)
    qlo, qhi = np.quantile(ytr, cfg["clip_quantiles"])
    pred = lad_predict(m, Xva, pack["xmu"], pack["xsd"], pack["ymu"],
                       pack["ysd"], qlo, qhi)
    pred_unclipped = lad_predict(m, Xva, pack["xmu"], pack["xsd"], pack["ymu"],
                                 pack["ysd"], qlo, qhi, clip=False)
    return pred, {
        "alpha": alpha,
        "alpha_selection": info,
        "fit": pack["fit"],
        "clip": [float(qlo), float(qhi)],
        "clip_fraction": float(np.mean((pred_unclipped < qlo) | (pred_unclipped > qhi))),
        "correction": {
            "mean": float(pred.mean()),
            "std": float(pred.std()),
            "min": float(pred.min()),
            "max": float(pred.max()),
        },
        "correction_unclipped": {
            "mean": float(pred_unclipped.mean()),
            "std": float(pred_unclipped.std()),
            "min": float(pred_unclipped.min()),
            "max": float(pred_unclipped.max()),
        },
        "coef_norm": float(np.linalg.norm(m.coef_)),
    }


def strata_for_probe(ctx, cfg):
    cache = ctx["cache"]
    probe = ctx["probe"]
    ap = np.asarray(cache["atom_ptr"])
    g = np.asarray(cache["global_feat"])
    y = ctx["y"]
    # ProbeFit support is not in support_audit.npz (that file is AuditEval).
    # Build a label-free fit-support map for ProbeFit with the same radius-2 WL
    # definition used in B3.  This is structure-only, no labels.
    import networkx as nx

    def graph(row):
        at = np.asarray(cache["atom_type"])
        bi = np.asarray(cache["bond_index"])
        bt = np.asarray(cache["bond_type"])
        bp = np.asarray(cache["bond_ptr"])
        gg = nx.Graph()
        for j, a in enumerate(at[ap[row]:ap[row + 1]]):
            gg.add_node(int(j), atom=int(a))
        for (u, v), t in zip(bi[bp[row]:bp[row + 1]], bt[bp[row]:bp[row + 1]]):
            gg.add_edge(int(u), int(v), bond=int(t))
        return gg

    def keys(gg):
        h = nx.weisfeiler_lehman_subgraph_hashes(
            gg, node_attr="atom", edge_attr="bond", iterations=2,
            digest_size=16, include_initial_labels=True,
        )
        return [h[int(n)][-1] for n in range(gg.number_of_nodes())]

    support = {}
    for r in np.asarray(load_split()["rows"]["fit"], dtype=np.int64):
        for k in keys(graph(int(r))):
            support.setdefault(k, set()).add(int(r))
    classes = []
    for r in probe:
        counts = [len(support.get(k, ())) for k in keys(graph(int(r)))]
        if any(c == 0 for c in counts):
            classes.append("unseen")
        elif any(1 <= c <= 4 for c in counts):
            classes.append("low_support")
        else:
            classes.append("full_support")
    classes = np.asarray(classes)
    n_atoms = ap[probe + 1] - ap[probe]
    rings = g[probe, 6]
    return {
        "support": classes,
        "tail": np.where(y <= float(cfg["tail_threshold"]), "tail_y_le_-3", "ordinary_y_gt_-3"),
        "size": np.asarray([f"size_{b}" for b in np.digitize(n_atoms, [21, 25, 29])]),
        "ring_rich": np.where(rings >= float(cfg["ring_rich_threshold"]),
                              "ring_rich_ge4", "ring_lt4"),
    }


def cycle_penalty_for_rows(rows, y_by_row):
    """Phase-1e label audit definition, diagnostic only.

    Uses the frozen Phase-1e fitted cycle scale and the GVAE logP/SA/cycle
    formula.  No cycle component is ever used as a model input.
    """
    import phase1e_common as P1E

    rows = [int(r) for r in rows]
    smiles = P1E.load_canonical_smiles()
    comp = P1E.reconstruct_label_components([smiles[r] for r in rows])
    audit = json.loads((P1E.RESULTS_1E / "label_audit.json").read_text())
    scale = audit["formula"]["fitted_cycle_scale"]
    sigma = float(scale["sigma_cycle"])
    mu = float(scale["mu_cycle"])
    y = np.asarray([y_by_row[r] for r in rows], dtype=np.float64)
    c = P1E.COMMUNITY_CONSTANTS
    y_chem = ((comp["logP"] - c["logP_mean"]) / c["logP_std"]
              + (comp["SA"] - c["SA_mean"]) / c["SA_std"])
    eff = (y - y_chem) * sigma + mu
    snapped = np.clip(np.round(eff), -20, 0)
    cycle_component = (snapped - mu) / sigma
    out = []
    csv_path = P1E.KSVD_AUDIT / "train_cycle_audit_label.csv"
    table = {}
    if csv_path.exists():
        with open(csv_path) as fh:
            for rec in csv.DictReader(fh):
                table[int(rec["subset_index"])] = rec
    for i, r in enumerate(rows):
        rec = {
            "row": r,
            "y": float(y[i]),
            "logP": float(comp["logP"][i]),
            "SA": float(comp["SA"][i]),
            "cycle_excess": int(-snapped[i]),
            "cycle_component": float(cycle_component[i]),
            "y_without_cycle": float(y[i] - cycle_component[i]),
            "snap_error": float(eff[i] - snapped[i]),
            "snap_error_y_units": float(abs(eff[i] - snapped[i]) / sigma),
        }
        if r in table:
            k = table[r]
            rec["phase1e_csv_crosscheck"] = {
                "label_excess": int(float(k["label_excess"])),
                "final_cycle_component": float(k["final_cycle_component"]),
                "y_without_cycle_label": float(k["y_without_cycle_label"]),
                "matched": k["matched"],
            }
        out.append(rec)
    return out
