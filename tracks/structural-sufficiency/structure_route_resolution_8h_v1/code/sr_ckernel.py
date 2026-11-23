"""Route C: frozen real-edit pairs, WL kernels, weighted ridge and CV.

All label use happens here, only after ``sr_catalogue.py`` has written and hashed
the pair freeze.  Fold vocabulary, centering, scalar scales and target scales are
rebuilt from the fold's training pairs; evaluation predictions are restored to
original label units before any error is computed.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import sr_common as C
import sr_catalogue as CAT
import sr_data as D
import sr_wl as WL

import rd_fragment as RF

LAMBDAS = (1e-4, 1e-2, 1.0)
N_FOLDS = 3


# ---------------------------------------------------------------------------
# pair graph / components
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(int(x), int(x))
        root = int(x)
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[int(x)] != root:
            nxt = self.parent[int(x)]
            self.parent[int(x)] = root
            x = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra != rb:
            self.parent[rb] = ra


def pair_components(row_a: np.ndarray, row_b: np.ndarray) -> np.ndarray:
    uf = UnionFind()
    for a, b in zip(row_a.tolist(), row_b.tolist()):
        uf.union(int(a), int(b))
    return np.asarray([uf.find(int(a)) for a in row_a.tolist()], np.int64)


def sub_components(idx: np.ndarray, row_a: np.ndarray, row_b: np.ndarray
                   ) -> np.ndarray:
    uf = UnionFind()
    for a, b in zip(row_a[idx].tolist(), row_b[idx].tolist()):
        uf.union(int(a), int(b))
    return np.asarray([uf.find(int(row_a[i])) for i in idx.tolist()], np.int64)


# ---------------------------------------------------------------------------
# endpoint features
# ---------------------------------------------------------------------------
@dataclass
class Endpoint:
    row: int
    edge: int
    s_port: int
    c_port: int
    cut: tuple[int, int, int]
    s_size: int
    c_size: int
    s_feat: np.ndarray
    c2_feat: np.ndarray
    c_full_feat: np.ndarray | None = None


class FeatureCache:
    def __init__(self, bundle: Any, arrays: Mapping[str, np.ndarray],
                 want_full: bool = False) -> None:
        self.bundle = bundle
        self.arrays = arrays
        self.want_full = bool(want_full)
        self.rows: dict[int, dict[str, Any]] = {}
        self.cache: dict[tuple[int, int], Endpoint] = {}

    def row_graph(self, row: int) -> dict[str, Any]:
        if row not in self.rows:
            v = D.view(self.bundle, int(row))
            self.rows[row] = {
                "atom_feat": np.asarray(v["atom_feat"], np.int64),
                "bond_u": np.asarray(v["bond_u"], np.int64),
                "bond_v": np.asarray(v["bond_v"], np.int64),
                "bond_feat": np.asarray(v["bond_feat"], np.int64),
                "n_atoms": int(v["n_atoms"]),
            }
        return self.rows[row]

    def get(self, row: int, edge: int, s_port: int, c_port: int,
            cut: Sequence[int], s_size: int, c_size: int) -> Endpoint:
        key = (int(row), int(edge))
        if key in self.cache:
            return self.cache[key]
        g = self.row_graph(int(row))
        n = g["n_atoms"]
        labels = RF.component_labels(n, g["bond_u"], g["bond_v"], skip_edge=int(edge))
        if labels[int(s_port)] == labels[int(c_port)]:
            raise ValueError("ports stayed on the same side after cutting")
        s_atoms = np.flatnonzero(labels == labels[int(s_port)])
        c_atoms = np.flatnonzero(labels == labels[int(c_port)])
        if int(s_atoms.size) != int(s_size) or int(c_atoms.size) != int(c_size):
            raise ValueError("frozen fragment sizes do not match the graph")
        s_frag = WL.induce(g["atom_feat"], g["bond_u"], g["bond_v"],
                           g["bond_feat"], s_atoms, int(s_port), cut)
        c_frag = WL.induce(g["atom_feat"], g["bond_u"], g["bond_v"],
                           g["bond_feat"], c_atoms, int(c_port), cut)
        c2_frag = WL.radius_ball(c_frag, 2)
        ep = Endpoint(row=int(row), edge=int(edge), s_port=int(s_port),
                      c_port=int(c_port), cut=tuple(int(x) for x in cut),
                      s_size=int(s_size), c_size=int(c_size),
                      s_feat=WL.wl_features(s_frag),
                      c2_feat=WL.wl_features(c2_frag),
                      c_full_feat=WL.wl_features(c_frag) if self.want_full else None)
        self.cache[key] = ep
        return ep


# ---------------------------------------------------------------------------
# design: pair-level raw features + target
# ---------------------------------------------------------------------------
@dataclass
class Design:
    split: str
    bundle: Any
    arrays: dict[str, np.ndarray]
    freeze_meta: dict[str, Any]
    s_a: np.ndarray
    s_b: np.ndarray
    c2: np.ndarray
    c_full: np.ndarray | None
    delta: np.ndarray
    row_fit: np.ndarray
    row_eval: np.ndarray
    global_component: np.ndarray
    train_idx: np.ndarray
    dev_idx: np.ndarray
    fitdev_idx: np.ndarray
    transform_id: np.ndarray
    core_key: np.ndarray
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    train_component: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def scope_rows(self, name: str) -> np.ndarray:
        return {"Fit-Fit": self.train_idx, "Dev-Dev": self.dev_idx,
                "Fit-Dev": self.fitdev_idx}[name]

    def pair_weights(self) -> np.ndarray:
        comp = sub_components(self.train_idx, self.arrays["row_a"],
                              self.arrays["row_b"])
        sizes = {c: int((comp == c).sum()) for c in np.unique(comp).tolist()}
        w = np.asarray([1.0 / sizes[int(c)] for c in comp], np.float64)
        w = w / w.mean()
        self.weights = w
        self.train_component = comp
        return w


def build_design(split: str, want_full: bool = False,
                 feature_cache: FeatureCache | None = None) -> Design:
    bundle = D.load_bundle(split)
    arrays, meta = CAT.load_frozen()
    atom_dims, bond_dims = D.feature_dims()
    cache = feature_cache or FeatureCache(bundle, arrays, want_full=want_full)
    n = int(arrays["row_a"].size)
    s_a = np.zeros((n, WL.FEATURE_DIM), np.float64)
    s_b = np.zeros((n, WL.FEATURE_DIM), np.float64)
    c2 = np.zeros((n, WL.FEATURE_DIM), np.float64)
    c_full = np.zeros((n, WL.FEATURE_DIM), np.float64) if want_full else None
    for i in range(n):
        ea = cache.get(int(arrays["row_a"][i]), int(arrays["edge_a"][i]),
                       int(arrays["s_port_a"][i]), int(arrays["c_port_a"][i]),
                       arrays["cut_a"][i], int(arrays["s_size_a"][i]),
                       int(arrays["c_size"][i]))
        eb = cache.get(int(arrays["row_b"][i]), int(arrays["edge_b"][i]),
                       int(arrays["s_port_b"][i]), int(arrays["c_port_b"][i]),
                       arrays["cut_b"][i], int(arrays["s_size_b"][i]),
                       int(arrays["c_size"][i]))
        s_a[i] = ea.s_feat
        s_b[i] = eb.s_feat
        c2[i] = ea.c2_feat
        if c_full is not None:
            c_full[i] = ea.c_full_feat
    labels = bundle.labels
    delta = labels[arrays["row_b"]] - labels[arrays["row_a"]]
    fit = np.zeros(int(labels.size), bool)
    fit[arrays["row_a"]] = np.isin(arrays["row_a"], bundle.fit_rows)
    # build explicit membership masks over the two split sides
    fit_set = set(int(x) for x in bundle.fit_rows.tolist())
    eval_set = set(int(x) for x in bundle.eval_rows.tolist())
    ra, rb = arrays["row_a"], arrays["row_b"]
    in_fit_a = np.asarray([int(r) in fit_set for r in ra.tolist()])
    in_fit_b = np.asarray([int(r) in fit_set for r in rb.tolist()])
    in_ev_a = np.asarray([int(r) in eval_set for r in ra.tolist()])
    in_ev_b = np.asarray([int(r) in eval_set for r in rb.tolist()])
    train_idx = np.flatnonzero(in_fit_a & in_fit_b)
    dev_idx = np.flatnonzero(in_ev_a & in_ev_b)
    fitdev_idx = np.flatnonzero((in_fit_a & in_ev_b) | (in_ev_a & in_fit_b))
    global_comp = pair_components(ra, rb)
    transform = arrays["transform_id"].astype(str)
    core = arrays["c_key"].astype(str)
    design = Design(split=split, bundle=bundle, arrays=arrays,
                    freeze_meta=meta, s_a=s_a, s_b=s_b, c2=c2, c_full=c_full,
                    delta=delta, row_fit=bundle.fit_rows, row_eval=bundle.eval_rows,
                    global_component=global_comp, train_idx=train_idx,
                    dev_idx=dev_idx, fitdev_idx=fitdev_idx,
                    transform_id=transform, core_key=core)
    design.pair_weights()
    return design


# ---------------------------------------------------------------------------
# scaling / kernels
# ---------------------------------------------------------------------------
def _scaled(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = np.array(features, copy=True)
    out[:, -2:] = (out[:, -2:] - mean) / scale
    return out


def scalar_stats(design: Design, train_idx: np.ndarray, w: np.ndarray,
                 which: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if which == "S":
        values = np.concatenate([design.s_a[train_idx, -2:],
                                 design.s_b[train_idx, -2:]], axis=0)
        weights = np.concatenate([w / 2.0, w / 2.0])
    elif which == "C":
        values = np.asarray(design.c2[train_idx, -2:])
        weights = np.asarray(w, np.float64)
    elif which == "CF":
        if design.c_full is None:
            raise ValueError("C_FULL features not built")
        values = np.asarray(design.c_full[train_idx, -2:])
        weights = np.asarray(w, np.float64)
    else:
        raise KeyError(which)
    return WL.weighted_scalar_stats(values, weights)


def _weighted_diag_mean(K: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * np.diag(K)) / np.sum(w))


@dataclass
class KernelSet:
    Kt: np.ndarray
    Kc: np.ndarray
    Ktc: np.ndarray
    Kt_off: np.ndarray
    Kc_off: np.ndarray
    Ktc_off: np.ndarray
    diag_t: float
    diag_c: float
    diag_tc: float
    scalars: dict[str, Any]


def build_kernels(dtr: Design, itr: np.ndarray, iev: np.ndarray,
                  w: np.ndarray, *, with_id: bool = False,
                  with_full: bool = False) -> KernelSet:
    mean_s, scale_s, meta_s = scalar_stats(dtr, itr, w, "S")
    mean_c, scale_c, meta_c = scalar_stats(dtr, itr, w, "C")
    sa = _scaled(dtr.s_a[itr], mean_s, scale_s)
    sb = _scaled(dtr.s_b[itr], mean_s, scale_s)
    x_tr = sb - sa
    ea = _scaled(dtr.s_a[iev], mean_s, scale_s)
    eb = _scaled(dtr.s_b[iev], mean_s, scale_s)
    x_ev = eb - ea
    chi_tr = _scaled(dtr.c2[itr], mean_c, scale_c)
    chi_ev = _scaled(dtr.c2[iev], mean_c, scale_c)
    center = (w[:, None] * chi_tr).sum(axis=0) / w.sum()
    chi_tr = chi_tr - center
    chi_ev = chi_ev - center

    Kt = x_tr @ x_tr.T
    Kc = chi_tr @ chi_tr.T
    diag_t = _weighted_diag_mean(Kt, w)
    diag_c = _weighted_diag_mean(Kc, w)
    diag_t = diag_t if diag_t > 1e-12 else 1.0
    diag_c = diag_c if diag_c > 1e-12 else 1.0
    Kt_n = Kt / diag_t
    Kc_n = Kc / diag_c
    Ktc_n = Kt_n * (1.0 + Kc_n)
    diag_tc = _weighted_diag_mean(Ktc_n, w)
    diag_tc = diag_tc if diag_tc > 1e-12 else 1.0
    Ktc_n = Ktc_n / diag_tc

    Kt_off = (x_ev @ x_tr.T) / diag_t
    Kc_off = (chi_ev @ chi_tr.T) / diag_c
    Ktc_off = (Kt_off * (1.0 + Kc_off)) / diag_tc
    scalars = {"S": meta_s, "C": meta_c,
               "diag_t": diag_t, "diag_c": diag_c, "diag_tc": diag_tc}
    out = KernelSet(Kt_n, Kc_n, Ktc_n, Kt_off, Kc_off, Ktc_off,
                    diag_t, diag_c, diag_tc, scalars)
    if with_id:
        # one-hot core identity, Fit-pair vocabulary only; unseen core -> exact 0
        vocab = sorted({str(dtr.core_key[i]) for i in itr.tolist()})
        index = {key: j for j, key in enumerate(vocab)}
        one_tr = np.zeros((itr.size, len(vocab)), np.float64)
        for row, i in enumerate(itr.tolist()):
            key = str(dtr.core_key[i])
            if key in index:
                one_tr[row, index[key]] = 1.0
        mean_id = (w[:, None] * one_tr).sum(axis=0) / w.sum()
        chi_id_tr = one_tr - mean_id
        chi_id_ev = np.zeros((iev.size, len(vocab)), np.float64)
        for row, i in enumerate(iev.tolist()):
            key = str(dtr.core_key[i])
            if key in index:
                chi_id_ev[row, index[key]] = 1.0
        Kid = chi_id_tr @ chi_id_tr.T
        diag_id = _weighted_diag_mean(Kid, w)
        diag_id = diag_id if diag_id > 1e-12 else 1.0
        Kid_n = Kid / diag_id
        Ktid_n = Kt_n * (1.0 + Kid_n)
        diag_tid = _weighted_diag_mean(Ktid_n, w)
        diag_tid = diag_tid if diag_tid > 1e-12 else 1.0
        Ktid_n = Ktid_n / diag_tid
        Kid_off = (chi_id_ev @ chi_id_tr.T) / diag_id
        Ktid_off = (Kt_off * (1.0 + Kid_off)) / diag_tid
        out.K_context_id = Kid_n
        out.Ktc_id = Ktid_n
        out.Ktc_id_off = Ktid_off
        out.diag_id = diag_id
        out.diag_tid = diag_tid
        out.id_vocab_size = len(vocab)
    if with_full:
        mean_f, scale_f, meta_f = scalar_stats(dtr, itr, w, "CF")
        cf_tr = _scaled(dtr.c_full[itr], mean_f, scale_f)
        cf_ev = _scaled(dtr.c_full[iev], mean_f, scale_f)
        center_f = (w[:, None] * cf_tr).sum(axis=0) / w.sum()
        cf_tr = cf_tr - center_f
        cf_ev = cf_ev - center_f
        Kf = cf_tr @ cf_tr.T
        diag_f = _weighted_diag_mean(Kf, w)
        diag_f = diag_f if diag_f > 1e-12 else 1.0
        Kf_n = Kf / diag_f
        Ktf_n = Kt_n * (1.0 + Kf_n)
        diag_tf = _weighted_diag_mean(Ktf_n, w)
        diag_tf = diag_tf if diag_tf > 1e-12 else 1.0
        Ktf_n = Ktf_n / diag_tf
        Kf_off = (cf_ev @ cf_tr.T) / diag_f
        Ktf_off = (Kt_off * (1.0 + Kf_off)) / diag_tf
        out.Ktc_full = Ktf_n
        out.Ktc_full_off = Ktf_off
        out.diag_f = diag_f
        out.diag_tf = diag_tf
        out.scalars["CF"] = meta_f
    return out


# ---------------------------------------------------------------------------
# weighted kernel ridge
# ---------------------------------------------------------------------------
def fit_krr(K: np.ndarray, K_off: np.ndarray, y: np.ndarray, w: np.ndarray,
            lam: float) -> np.ndarray:
    n = int(K.shape[0])
    s = np.sqrt(np.maximum(w, 0.0))
    A = (s[:, None] * K) * s[None, :] + n * float(lam) * np.eye(n)
    beta = np.linalg.solve(A, s * y)
    return (K_off * s[None, :]) @ beta


# ---------------------------------------------------------------------------
# exact-transform lookup
# ---------------------------------------------------------------------------
def exact_mean_predict(design: Design, itr: np.ndarray, iev: np.ndarray,
                       comp_train: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    by_transform: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, c in zip(itr.tolist(), comp_train.tolist()):
        by_transform[str(design.transform_id[i])][int(c)].append(int(i))
    pred = np.zeros(iev.size, np.float64)
    support = np.zeros(iev.size, np.int64)
    seen = np.zeros(iev.size, bool)
    for row, i in enumerate(iev.tolist()):
        groups = by_transform.get(str(design.transform_id[i]))
        if not groups:
            continue
        means = [float(np.mean(design.delta[np.asarray(idx, np.int64)]))
                 for idx in groups.values()]
        pred[row] = float(np.sum(means) / (len(means) + 1.0))
        support[row] = len(means)
        seen[row] = True
    return pred, {"coverage": float(seen.mean()) if iev.size else None,
                  "n_seen": int(seen.sum()), "n_eval": int(iev.size),
                  "support_counts": {str(int(k)): int(v) for k, v in
                                     zip(*np.unique(support, return_counts=True))}}


# ---------------------------------------------------------------------------
# cross validation
# ---------------------------------------------------------------------------
def assign_folds(design: Design, comp: np.ndarray) -> dict[int, int]:
    sizes = {int(c): int((comp == c).sum()) for c in np.unique(comp).tolist()}
    keys = sorted(sizes, key=lambda c: hashlib.sha256(
        f"route8h-cv:{c}".encode()).hexdigest())
    load = [0] * N_FOLDS
    assigned: dict[int, int] = {}
    for c in keys:
        f = int(np.argmin(load))
        assigned[c] = f
        load[f] += sizes[c]
    return assigned


def _fold_sigma(design: Design, itr: np.ndarray) -> float:
    rows = np.unique(np.concatenate([design.arrays["row_a"][itr],
                                     design.arrays["row_b"][itr]]))
    sigma = float(np.std(design.bundle.labels[rows]))
    return sigma if sigma > 0 else 1.0


KERNEL_NAMES = ("K_TRANSFORM", "K_CONDITIONAL_R2", "K_CONTEXT_ID",
                "K_CONDITIONAL_FULL")


def _kernel_pair(ks: KernelSet, name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    return {
        "K_TRANSFORM": (ks.Kt, ks.Kt_off),
        "K_CONDITIONAL_R2": (ks.Ktc, ks.Ktc_off),
        "K_CONTEXT_ID": (getattr(ks, "Ktc_id", None), getattr(ks, "Ktc_id_off", None)),
        "K_CONDITIONAL_FULL": (getattr(ks, "Ktc_full", None),
                               getattr(ks, "Ktc_full_off", None)),
    }[name]


def cv_select(design: Design, *, with_id: bool, with_full: bool = False
              ) -> dict[str, Any]:
    itr = design.train_idx
    comp = sub_components(itr, design.arrays["row_a"], design.arrays["row_b"])
    design.train_component = np.zeros(design.arrays["row_a"].size, np.int64)
    design.train_component[itr] = comp
    w = design.weights
    n_comp = int(len(np.unique(comp)))
    info: dict[str, Any] = {"n_train_pairs": int(itr.size),
                            "n_train_components": n_comp,
                            "lambdas": list(LAMBDAS)}
    if n_comp < N_FOLDS:
        info["cv"] = "NOT_POSSIBLE_LT3_COMPONENTS"
        info["lambda"] = {name: 1e-2 for name in KERNEL_NAMES}
        return info
    folds = assign_folds(design, comp)
    fold_ids = np.asarray([folds[int(c)] for c in comp.tolist()], np.int64)
    oof = {name: {lam: np.full(itr.size, np.nan) for lam in LAMBDAS}
           for name in KERNEL_NAMES}
    oof_exact = np.full(itr.size, np.nan)
    oof_zero = np.zeros(itr.size, np.float64)
    fold_rows = []
    for fold in range(N_FOLDS):
        tr = np.flatnonzero(fold_ids != fold)
        va = np.flatnonzero(fold_ids == fold)
        wf = w[tr]
        wf = wf / wf.mean()
        sigma = _fold_sigma(design, itr[tr])
        y = design.delta[itr[tr]] / sigma
        ks = build_kernels(design, itr[tr], itr[va], wf, with_id=with_id,
                           with_full=with_full)
        em, em_info = exact_mean_predict(design, itr[tr], itr[va], comp[tr])
        oof_exact[va] = em
        fold_rows.append({"fold": fold, "n_train": int(tr.size),
                          "n_eval": int(va.size),
                          "n_train_components": int(len(np.unique(comp[tr]))),
                          "sigma": sigma, "exact_mean": em_info})
        for name in KERNEL_NAMES:
            K, K_off = _kernel_pair(ks, name)
            if K is None or K_off is None:
                continue
            for lam in LAMBDAS:
                oof[name][lam][va] = fit_krr(K, K_off, y, wf, lam) * sigma
    err_exact = np.abs(oof_exact - design.delta[itr])
    err_zero = np.abs(oof_zero - design.delta[itr])
    scores: dict[str, float] = {"EXACT_MEAN": component_equal_mae(err_exact, comp),
                                "ZERO": component_equal_mae(err_zero, comp)}
    pair_scores: dict[str, float] = {"EXACT_MEAN": float(np.mean(err_exact)),
                                     "ZERO": float(np.mean(err_zero))}
    lambdas: dict[str, float] = {}
    errors: dict[str, np.ndarray] = {"EXACT_MEAN": err_exact, "ZERO": err_zero}
    for name in KERNEL_NAMES:
        best = None
        for lam in LAMBDAS:
            pred = oof[name][lam]
            if np.isnan(pred).any():
                continue
            err = np.abs(pred - design.delta[itr])
            score = component_equal_mae(err, comp)
            if best is None or score < best[0] - 1e-15 or (
                    abs(score - best[0]) <= 1e-15 and lam > best[1]):
                best = (score, lam, err)
        if best is None:
            continue
        scores[name] = float(best[0])
        lambdas[name] = float(best[1])
        errors[name] = best[2]
        pair_scores[name] = float(np.mean(best[2]))
    info["cv"] = "THREE_FOLD_COMPONENT"
    info["fold_rows"] = fold_rows
    info["lambda"] = lambdas
    info["oof_component_equal_mae"] = scores
    info["oof_pair_equal_mae"] = pair_scores
    info["oof_errors"] = {k: v.tolist() for k, v in errors.items()}
    return info


def component_equal_mae(err: np.ndarray, comp: np.ndarray) -> float:
    err = np.asarray(err, np.float64)
    comp = np.asarray(comp, np.int64)
    values = [float(np.mean(np.abs(err[comp == c]))) for c in np.unique(comp).tolist()]
    return float(np.mean(values)) if values else float("nan")


def pair_equal_mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(err, np.float64))))


def pair_rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(err, np.float64) ** 2)))
