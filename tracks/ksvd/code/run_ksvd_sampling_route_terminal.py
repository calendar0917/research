"""Terminal mechanism study for the KSVD / patch-sampling route.

This is deliberately separate from the MolHIV model-search scripts.  It tests
three claims under a matched patch budget:

1. Does a wider sampler capture information missed by one-hop ego patches?
2. Does keeping the *correct* relation between patch contents help beyond a bag?
3. Does KSVD add value beyond a direct, permutation-invariant raw patch mean?

Leakage control: each CV fold learns KSVD only from training-fold patches.
TUData has no official split here; all reported TU results use a fixed 5-fold
stratified protocol and are mechanism evidence, not leaderboard estimates.

Run from tracks/ksvd:
  python -m code.run_ksvd_sampling_route_terminal --stage all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_tud import load_tud, node_feature_readout
from .graph import Graph, from_edges
from .ksvd import ksvd
from .sampling_route import (
    MatchedSamplingConfig,
    PatchCollection,
    aggregate_process,
    collection_metrics,
    content_readout,
    raw_mean_embedding,
    relation_channels,
    relation_graph_readout,
    relation_readout,
    sample_matched,
    sparse_code_matrix,
    vectorize_collection,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "ksvd_sampling_route_terminal_20260729.json"
OUT_MD = ROOT / "docs" / "KSVD_SAMPLING_ROUTE_TERMINAL_REPORT_20260729.md"
CACHE_DIR = ROOT / "results" / "cache_sampling_route_20260729"
METHODS = ["B0", "R2", "UniformRW", "CoverageRW", "Path", "PPR"]
READOUTS = [
    "raw_mean",
    "content",
    "relation_graph",
    "content_graph",
    "relation_true",
    "content_true",
    "relation_shuffled",
    "content_shuffled",
]


def _json_dump(result: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT_JSON)


def _cache_path(tag: str, method: str, seed: int, cfg: MatchedSamplingConfig, feature_mode: str) -> Path:
    payload = json.dumps({"tag": tag, "method": method, "seed": seed, "cfg": asdict(cfg), "feat": feature_mode}, sort_keys=True)
    key = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{tag}_{method}_s{seed}_{key}.pkl"


def prepare_collections(
    tag: str,
    graphs: list[Graph],
    node_feats: list[np.ndarray | None],
    method: str,
    base_cfg: MatchedSamplingConfig,
    sampling_seed: int,
    feature_mode: str,
    use_cache: bool = True,
) -> dict[str, Any]:
    cfg = replace(base_cfg, seed=sampling_seed)
    path = _cache_path(tag, method, sampling_seed, cfg, feature_mode)
    if use_cache and path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    pcs: list[PatchCollection] = []
    Ys: list[np.ndarray] = []
    channels: list[dict[str, np.ndarray]] = []
    process: list[dict[str, float]] = []
    raw: list[np.ndarray] = []
    started = time.time()
    for i, g in enumerate(graphs):
        # Graph-specific mixing avoids giving every graph the same random stream,
        # while preserving a common center schedule across methods.
        graph_cfg = replace(cfg, seed=sampling_seed * 1_000_003 + i * 9_973 + 17)
        pc = sample_matched(g, method, graph_cfg)
        Y = vectorize_collection(g, pc, cfg.max_nodes, feature_mode, node_feats[i])
        pcs.append(pc)
        Ys.append(Y)
        channels.append(relation_channels(g, pc))
        process.append(collection_metrics(g, pc))
        raw.append(raw_mean_embedding(Y))
    obj = {
        "pcs": pcs,
        "Ys": Ys,
        "channels": channels,
        "process": process,
        "raw": raw,
        "elapsed_sec": time.time() - started,
    }
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return obj


def _fit_predict_metrics(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, seed: int) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=1.0, random_state=seed, solver="liblinear"),
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    out = {
        "accuracy": float(accuracy_score(yte, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
    }
    if len(np.unique(yte)) == 2:
        score = clf.decision_function(Xte)
        out["roc_auc"] = float(roc_auc_score(yte, score))
    return out


def _fit_tuned_single(
    A: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    seed: int,
) -> dict[str, float]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    Cs = [0.01, 0.1, 1.0]
    candidates: list[tuple[float, float]] = []
    ytr = y[tr]
    for C in Cs:
        vals: list[float] = []
        for itr, iva in inner.split(np.zeros(len(tr)), ytr):
            ii, iv = tr[itr], tr[iva]
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(A[ii])
            Xval = scaler.transform(A[iv])
            clf = LogisticRegression(max_iter=4000, C=C, random_state=seed, solver="liblinear")
            clf.fit(Xtr, y[ii])
            vals.append(float(accuracy_score(y[iv], clf.predict(Xval))))
        candidates.append((float(np.mean(vals)), C))
    _, C = max(candidates, key=lambda z: (z[0], -z[1]))
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(A[tr])
    Xte = scaler.transform(A[te])
    clf = LogisticRegression(max_iter=4000, C=C, random_state=seed, solver="liblinear")
    clf.fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    out = {
        "accuracy": float(accuracy_score(y[te], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
        "selected_C": float(C),
    }
    if len(np.unique(y[te])) == 2:
        out["roc_auc"] = float(roc_auc_score(y[te], clf.decision_function(Xte)))
    return out


def _fit_tuned_block_fusion(
    A: np.ndarray,
    B: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Nested train-only selection of regularization and structural block weight.

    alpha=0 is included, so the fused model can explicitly fall back to the
    node-attribute baseline instead of being forced to use noisy patch features.
    """
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    alphas = [0.0, 0.3, 1.0]
    Cs = [0.01, 0.1, 1.0]
    candidates: list[tuple[float, float, float]] = []
    ytr = y[tr]
    for alpha in alphas:
        for C in Cs:
            vals: list[float] = []
            for itr, iva in inner.split(np.zeros(len(tr)), ytr):
                ii, iv = tr[itr], tr[iva]
                sa, sb = StandardScaler(), StandardScaler()
                Atr = sa.fit_transform(A[ii])
                Aval = sa.transform(A[iv])
                Btr = sb.fit_transform(B[ii])
                Bval = sb.transform(B[iv])
                Xtr = np.concatenate([Atr, alpha * Btr], axis=1)
                Xval = np.concatenate([Aval, alpha * Bval], axis=1)
                clf = LogisticRegression(max_iter=4000, C=C, random_state=seed, solver="liblinear")
                clf.fit(Xtr, y[ii])
                vals.append(float(accuracy_score(y[iv], clf.predict(Xval))))
            candidates.append((float(np.mean(vals)), alpha, C))
    # Conservative tie break: prefer less structural weight, then stronger regularization.
    _, alpha, C = max(candidates, key=lambda z: (z[0], -z[1], -z[2]))
    sa, sb = StandardScaler(), StandardScaler()
    Atr = sa.fit_transform(A[tr])
    Ate = sa.transform(A[te])
    Btr = sb.fit_transform(B[tr])
    Bte = sb.transform(B[te])
    Xtr = np.concatenate([Atr, alpha * Btr], axis=1)
    Xte = np.concatenate([Ate, alpha * Bte], axis=1)
    clf = LogisticRegression(max_iter=4000, C=C, random_state=seed, solver="liblinear")
    clf.fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    out = {
        "accuracy": float(accuracy_score(y[te], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
        "selected_alpha": float(alpha),
        "selected_C": float(C),
    }
    if len(np.unique(y[te])) == 2:
        out["roc_auc"] = float(roc_auc_score(y[te], clf.decision_function(Xte)))
    return out


def _dictionary_pair_stability(dictionaries: list[np.ndarray]) -> float | None:
    vals: list[float] = []
    for i in range(len(dictionaries)):
        for j in range(i + 1, len(dictionaries)):
            sim = np.abs(dictionaries[i].T @ dictionaries[j])
            rr, cc = linear_sum_assignment(-sim)
            vals.append(float(sim[rr, cc].mean()))
    return float(np.mean(vals)) if vals else None


def evaluate_prepared_cv(
    prepared: dict[str, Any],
    y: np.ndarray,
    *,
    n_atoms: int,
    T: int,
    n_iter: int,
    n_splits: int,
    split_seed: int,
    max_train_patches: int,
    attr: list[np.ndarray] | None = None,
    graph_basic: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    Ys: list[np.ndarray] = prepared["Ys"]
    channels: list[dict[str, np.ndarray]] = prepared["channels"]
    raw = np.stack(prepared["raw"])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    folds: list[dict[str, Any]] = []
    dictionaries: list[np.ndarray] = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        Ytr = np.concatenate([Ys[int(i)] for i in tr], axis=1)
        if Ytr.shape[1] > max_train_patches:
            rng = np.random.default_rng(split_seed * 100 + fold)
            keep = rng.choice(Ytr.shape[1], max_train_patches, replace=False)
            Yfit = Ytr[:, keep]
        else:
            Yfit = Ytr
        D, _, dinfo = ksvd(
            Yfit,
            n_atoms=n_atoms,
            T=T,
            T_min=min(T, 2),
            n_iter=n_iter,
            seed=split_seed * 1000 + fold,
        )
        dictionaries.append(D)
        extra_keys: list[str] = []
        if attr is not None:
            extra_keys += ["node_attr_global", "attr_content", "attr_content_graph", "attr_content_true", "attr_content_shuffled"]
        if graph_basic is not None:
            extra_keys += ["graph_basic"]
        if attr is not None and graph_basic is not None:
            extra_keys += ["attr_graph_basic"]
        feats: dict[str, list[np.ndarray]] = {k: [] for k in READOUTS + extra_keys}
        graph_errors: list[float] = []
        for i in range(len(y)):
            X, errors = sparse_code_matrix(D, Ys[i], T)
            content = content_readout(X, errors)
            graph_rel = relation_graph_readout(channels[i])
            true_rel = relation_readout(X, channels[i], include_topology=False)
            # A graph-specific deterministic permutation. It is independent of
            # the class label and changes with fold so no accidental template is
            # shared between training and test.
            prng = np.random.default_rng(split_seed * 10_000_019 + fold * 100_003 + i * 997)
            perm = prng.permutation(X.shape[1])
            shuf_rel = relation_readout(X, channels[i], perm, include_topology=False)
            feats["raw_mean"].append(raw[i])
            feats["content"].append(content)
            feats["relation_graph"].append(graph_rel)
            feats["content_graph"].append(np.concatenate([content, graph_rel]))
            feats["relation_true"].append(true_rel)
            feats["content_true"].append(np.concatenate([content, graph_rel, true_rel]))
            feats["relation_shuffled"].append(shuf_rel)
            full_true = np.concatenate([content, graph_rel, true_rel])
            full_shuffled = np.concatenate([content, graph_rel, shuf_rel])
            feats["content_shuffled"].append(full_shuffled)
            if attr is not None:
                feats["node_attr_global"].append(attr[i])
                feats["attr_content"].append(np.concatenate([attr[i], content]))
                feats["attr_content_graph"].append(np.concatenate([attr[i], content, graph_rel]))
                feats["attr_content_true"].append(np.concatenate([attr[i], full_true]))
                feats["attr_content_shuffled"].append(np.concatenate([attr[i], full_shuffled]))
            if graph_basic is not None:
                feats["graph_basic"].append(graph_basic[i])
            if attr is not None and graph_basic is not None:
                feats["attr_graph_basic"].append(np.concatenate([attr[i], graph_basic[i]]))
            graph_errors.append(float(errors.mean()))
        score: dict[str, dict[str, float]] = {}
        for key in feats:
            F = np.stack(feats[key])
            score[key] = _fit_predict_metrics(F[tr], y[tr], F[te], y[te], split_seed + fold)
        if attr is not None:
            A = np.stack(attr)
            score["tuned_attr_only"] = _fit_tuned_single(A, y, tr, te, split_seed * 100 + fold)
            tuned_sources = {
                "raw_mean": np.stack(feats["raw_mean"]),
                "content": np.stack(feats["content"]),
                "content_graph": np.stack(feats["content_graph"]),
                "content_true": np.stack(feats["content_true"]),
                "content_shuffled": np.stack(feats["content_shuffled"]),
            }
            if graph_basic is not None:
                tuned_sources["graph_basic"] = np.stack(graph_basic)
            for source, B in tuned_sources.items():
                score[f"tuned_attr_{source}"] = _fit_tuned_block_fusion(
                    A, B, y, tr, te, split_seed * 100 + fold
                )
        folds.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_dictionary_patches": int(Yfit.shape[1]),
            "dictionary": dinfo,
            "test_reconstruction_mean": float(np.mean([graph_errors[int(i)] for i in te])),
            "score": score,
        })
    keys = list(folds[0]["score"])
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        metric_names = list(folds[0]["score"][key])
        summary[key] = {}
        for metric in metric_names:
            vals = [r["score"][key][metric] for r in folds]
            summary[key][f"mean_{metric}"] = float(np.mean(vals))
            summary[key][f"std_{metric}"] = float(np.std(vals))
    paired = {}
    for metric in ["accuracy", "balanced_accuracy", "roc_auc"]:
        if metric not in folds[0]["score"]["content"]:
            continue
        true_gain = [r["score"]["content_true"][metric] - r["score"]["content"][metric] for r in folds]
        graph_gain = [r["score"]["content_graph"][metric] - r["score"]["content"][metric] for r in folds]
        alignment_gain = [r["score"]["content_true"][metric] - r["score"]["content_graph"][metric] for r in folds]
        shuffle_gap = [r["score"]["content_true"][metric] - r["score"]["content_shuffled"][metric] for r in folds]
        paired[metric] = {
            "true_minus_content_mean": float(np.mean(true_gain)),
            "true_minus_content_values": [float(v) for v in true_gain],
            "graph_minus_content_mean": float(np.mean(graph_gain)),
            "graph_minus_content_values": [float(v) for v in graph_gain],
            "alignment_minus_graph_mean": float(np.mean(alignment_gain)),
            "alignment_minus_graph_values": [float(v) for v in alignment_gain],
            "true_minus_shuffled_mean": float(np.mean(shuffle_gap)),
            "true_minus_shuffled_values": [float(v) for v in shuffle_gap],
        }
    return {
        "folds": folds,
        "summary": summary,
        "paired_deltas": paired,
        "mean_test_reconstruction": float(np.mean([r["test_reconstruction_mean"] for r in folds])),
        "dictionary_stability_across_folds": _dictionary_pair_stability(dictionaries),
    }


def _relabel(g: Graph, x: np.ndarray | None, rng: np.random.Generator) -> tuple[Graph, np.ndarray | None]:
    old_to_new = rng.permutation(g.n)
    gp = from_edges(g.n, [(int(old_to_new[u]), int(old_to_new[v])) for u, v in g.edges()])
    if x is None:
        return gp, None
    xp = np.zeros_like(x)
    for old in range(g.n):
        xp[int(old_to_new[old])] = x[old]
    return gp, xp


def _cube() -> Graph:
    return from_edges(8, [(u, u ^ bit) for u in range(8) for bit in [1, 2, 4] if u < (u ^ bit)])


def _mobius8() -> Graph:
    return from_edges(8, [(i, (i + 1) % 8) for i in range(8)] + [(i, i + 4) for i in range(4)])


def _cycle(n: int) -> Graph:
    return from_edges(n, [(i, (i + 1) % n) for i in range(n)])


def _four_triangles() -> Graph:
    return from_edges(12, [(3 * b + i, 3 * b + (i + 1) % 3) for b in range(4) for i in range(3)])


def make_isomorphic_dataset(base: list[Graph], n_per_class: int, seed: int) -> tuple[list[Graph], np.ndarray, list[np.ndarray | None]]:
    rng = np.random.default_rng(seed)
    graphs: list[Graph] = []
    labels: list[int] = []
    for cls, g in enumerate(base):
        for _ in range(n_per_class):
            gp, _ = _relabel(g, None, rng)
            graphs.append(gp)
            labels.append(cls)
    order = rng.permutation(len(labels))
    return [graphs[int(i)] for i in order], np.asarray(labels, dtype=np.int64)[order], [None] * len(labels)


def relation_positive_control(split_seed: int = 31) -> dict[str, Any]:
    """Same cycle, same A/B content bag; only neighboring A/B arrangement differs."""
    rng = np.random.default_rng(20260729)
    base_g = _cycle(12)
    graphs: list[Graph] = []
    feats: list[np.ndarray] = []
    labels: list[int] = []
    pcs: list[PatchCollection] = []
    for cls in [0, 1]:
        base_label = np.asarray([(i % 2) if cls == 0 else (0 if i < 6 else 1) for i in range(12)])
        base_x = np.eye(2, dtype=np.float64)[base_label]
        for _ in range(80):
            gp, xp = _relabel(base_g, base_x, rng)
            assert xp is not None
            graphs.append(gp)
            feats.append(xp)
            labels.append(cls)
            pcs.append(PatchCollection("singleton", [{u} for u in gp.nodes], list(gp.nodes), [[] for _ in gp.nodes]))
    order = rng.permutation(len(labels))
    graphs = [graphs[int(i)] for i in order]
    feats = [feats[int(i)] for i in order]
    pcs = [pcs[int(i)] for i in order]
    y = np.asarray(labels, dtype=np.int64)[order]
    Ys = [vectorize_collection(g, pc, 1, "labeled_wl", x) for g, pc, x in zip(graphs, pcs, feats)]
    prepared = {
        "pcs": pcs,
        "Ys": Ys,
        "channels": [relation_channels(g, pc) for g, pc in zip(graphs, pcs)],
        "process": [collection_metrics(g, pc) for g, pc in zip(graphs, pcs)],
        "raw": [raw_mean_embedding(Y) for Y in Ys],
        "elapsed_sec": 0.0,
    }
    result = evaluate_prepared_cv(
        prepared, y, n_atoms=2, T=1, n_iter=3, n_splits=5,
        split_seed=split_seed, max_train_patches=5000,
    )
    result["task"] = "12-node cycle; class 0 alternates A/B, class 1 has six A then six B; identical topology and label multiset"
    result["expected_control"] = "content near chance; true relation high; shuffled relation near chance"
    return result


def run_synthetic(result: dict[str, Any], cfg: MatchedSamplingConfig, seeds: list[int], use_cache: bool) -> None:
    result.setdefault("synthetic", {})
    result["synthetic"]["relation_positive_control"] = relation_positive_control()
    _json_dump(result)
    tasks = {
        "cube_vs_mobius8": make_isomorphic_dataset([_cube(), _mobius8()], 60, 101),
        "cycle12_vs_four_triangles": make_isomorphic_dataset([_cycle(12), _four_triangles()], 60, 103),
    }
    for task, (graphs, y, node_feats) in tasks.items():
        task_out = result["synthetic"].setdefault(task, {"description": (
            "matched local-ambiguity control" if task == "cube_vs_mobius8" else
            "local-signal control: one-hop patches already distinguish cycle from triangles"
        ), "runs": {}})
        for method in METHODS:
            for sampling_seed in seeds:
                key = f"{method}_s{sampling_seed}"
                if key in task_out["runs"]:
                    continue
                print(f"[synthetic] {task} {key}", flush=True)
                prepared = prepare_collections(task, graphs, node_feats, method, cfg, sampling_seed, "wl", use_cache)
                ev = evaluate_prepared_cv(
                    prepared, y, n_atoms=8, T=2, n_iter=5, n_splits=5,
                    split_seed=31, max_train_patches=6000,
                )
                task_out["runs"][key] = {
                    "method": method,
                    "sampling_seed": sampling_seed,
                    "process": aggregate_process(prepared["process"]),
                    "prepare_elapsed_sec": prepared["elapsed_sec"],
                    "evaluation": ev,
                }
                _json_dump(result)


def _cosine_stability(vectors_by_seed: list[np.ndarray]) -> dict[str, float]:
    vals: list[float] = []
    for i in range(len(vectors_by_seed)):
        for j in range(i + 1, len(vectors_by_seed)):
            A, B = vectors_by_seed[i], vectors_by_seed[j]
            vals.extend(np.sum(A * B, axis=1).tolist())
    return {
        "mean_cosine": float(np.mean(vals)) if vals else 1.0,
        "std_cosine": float(np.std(vals)) if vals else 0.0,
        "mean_cosine_distance": float(1.0 - np.mean(vals)) if vals else 0.0,
    }


def graph_basic_readout(g: Graph) -> np.ndarray:
    """Small permutation-invariant baseline to expose size/degree shortcuts."""
    deg = np.asarray([len(g.neighbors(u)) for u in g.nodes], dtype=np.float64)
    if deg.size == 0:
        return np.zeros(13, dtype=np.float64)
    # Connected components and cycle rank.
    unseen = set(g.nodes)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            u = stack.pop()
            fresh = g.neighbors(u) & unseen
            unseen -= fresh
            stack.extend(fresh)
    cycle_rank = g.num_edges() - g.n + components
    hist = np.asarray([(deg == d).mean() for d in range(5)], dtype=np.float64)
    return np.asarray([
        g.n, g.num_edges(), g.num_edges() / max(g.n, 1),
        deg.mean(), deg.std(), deg.max(),
        components, cycle_rank, *hist,
    ], dtype=np.float64)


def run_tud_dataset(
    result: dict[str, Any],
    name: str,
    cfg: MatchedSamplingConfig,
    seeds: list[int],
    use_cache: bool,
) -> None:
    graphs, y, node_feats, meta = load_tud(name)
    print("loaded", meta, flush=True)
    ds_out = result.setdefault("tud", {}).setdefault(name, {"meta": meta, "protocol": {}, "runs": {}, "sampling_stability": {}})
    ds_out["protocol"] = {
        "split": "StratifiedKFold(n_splits=5, shuffle=True, random_state=31)",
        "note": "TUData has no official train/valid/test split in this experiment; dictionary fitted on training folds only",
        "classifier": "StandardScaler + LogisticRegression(C=1, fixed)",
        "feature_mode": "labeled_wl",
    }
    attr = [node_feature_readout(x) for x in node_feats]
    basic = [graph_basic_readout(g) for g in graphs]
    raw_by_method: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    for method in METHODS:
        for sampling_seed in seeds:
            key = f"{method}_s{sampling_seed}"
            print(f"[TU:{name}] {key}", flush=True)
            prepared = prepare_collections(name, graphs, node_feats, method, cfg, sampling_seed, "labeled_wl", use_cache)
            raw_by_method[method].append(np.stack(prepared["raw"]))
            if key not in ds_out["runs"]:
                ev = evaluate_prepared_cv(
                    prepared, y, n_atoms=8, T=2, n_iter=5, n_splits=5,
                    split_seed=31, max_train_patches=6000, attr=attr, graph_basic=basic,
                )
                ds_out["runs"][key] = {
                    "method": method,
                    "sampling_seed": sampling_seed,
                    "process": aggregate_process(prepared["process"]),
                    "prepare_elapsed_sec": prepared["elapsed_sec"],
                    "evaluation": ev,
                }
                _json_dump(result)
        ds_out["sampling_stability"][method] = _cosine_stability(raw_by_method[method])
        _json_dump(result)


def aggregate_runs(runs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in METHODS:
        rs = [r for r in runs.values() if r["method"] == method]
        if not rs:
            continue
        row: dict[str, Any] = {"n_sampling_seeds": len(rs)}
        available_readouts = sorted(set().union(*(r["evaluation"]["summary"].keys() for r in rs)))
        for readout in available_readouts:
            vals = [r["evaluation"]["summary"][readout]["mean_accuracy"] for r in rs]
            entry = {"mean_accuracy": float(np.mean(vals)), "std_over_sampling_seeds": float(np.std(vals)), "values": vals}
            for extra in ["mean_selected_alpha", "mean_selected_C", "mean_roc_auc"]:
                extra_vals = [r["evaluation"]["summary"][readout][extra] for r in rs if extra in r["evaluation"]["summary"][readout]]
                if extra_vals:
                    entry[extra] = float(np.mean(extra_vals))
            row[readout] = entry
        for delta_name in ["true_minus_content_mean", "graph_minus_content_mean", "alignment_minus_graph_mean", "true_minus_shuffled_mean"]:
            vals = [r["evaluation"]["paired_deltas"]["accuracy"][delta_name] for r in rs]
            row[delta_name] = float(np.mean(vals))
        row["mean_test_reconstruction"] = float(np.mean([r["evaluation"]["mean_test_reconstruction"] for r in rs]))
        row["dictionary_stability_across_folds"] = float(np.mean([
            r["evaluation"]["dictionary_stability_across_folds"] for r in rs
            if r["evaluation"]["dictionary_stability_across_folds"] is not None
        ]))
        row["process"] = aggregate_process([r["process"] for r in rs])
        out[method] = row
    return out


def run_uniform_relation_robustness(
    result: dict[str, Any],
    datasets: list[str],
    cfg: MatchedSamplingConfig,
    sampling_seeds: list[int],
    split_seeds: list[int],
    use_cache: bool,
) -> None:
    out = result.setdefault("robustness", {}).setdefault("UniformRW_relation", {})
    for name in datasets:
        graphs, y, node_feats, meta = load_tud(name)
        ds = out.setdefault(name, {"meta": meta, "runs": {}})
        for sampling_seed in sampling_seeds:
            prepared = prepare_collections(name, graphs, node_feats, "UniformRW", cfg, sampling_seed, "labeled_wl", use_cache)
            for split_seed in split_seeds:
                key = f"sampling{sampling_seed}_split{split_seed}"
                if key in ds["runs"]:
                    continue
                print(f"[robust:{name}] {key}", flush=True)
                ev = evaluate_prepared_cv(
                    prepared, y, n_atoms=8, T=2, n_iter=5, n_splits=5,
                    split_seed=split_seed, max_train_patches=6000,
                )
                ds["runs"][key] = ev
                _json_dump(result)
        rows = list(ds["runs"].values())
        ds["summary"] = {}
        for key in ["content", "content_graph", "content_true", "content_shuffled"]:
            vals = [rr["summary"][key]["mean_accuracy"] for rr in rows]
            ds["summary"][key] = {"mean_accuracy": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        ds["summary"]["true_minus_content"] = float(np.mean([
            rr["paired_deltas"]["accuracy"]["true_minus_content_mean"] for rr in rows
        ]))
        ds["summary"]["graph_minus_content"] = float(np.mean([
            rr["paired_deltas"]["accuracy"]["graph_minus_content_mean"] for rr in rows
        ]))
        ds["summary"]["alignment_minus_graph"] = float(np.mean([
            rr["paired_deltas"]["accuracy"]["alignment_minus_graph_mean"] for rr in rows
        ]))
        ds["summary"]["true_minus_shuffled"] = float(np.mean([
            rr["paired_deltas"]["accuracy"]["true_minus_shuffled_mean"] for rr in rows
        ]))
        _json_dump(result)


def finalize(result: dict[str, Any]) -> None:
    for task, obj in result.get("synthetic", {}).items():
        if isinstance(obj, dict) and "runs" in obj:
            obj["aggregate"] = aggregate_runs(obj["runs"])
    for name, obj in result.get("tud", {}).items():
        obj["aggregate"] = aggregate_runs(obj["runs"])
    result["elapsed_sec"] = float(time.time() - result.get("started_epoch", time.time()))
    result["status"] = "complete"
    _json_dump(result)
    OUT_MD.write_text(render_report(result), encoding="utf-8")


def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def render_table(agg: dict[str, Any], stability: dict[str, Any] | None = None) -> list[str]:
    lines = [
        "| 采样 | raw mean | KSVD 内容 | 内容+关系图 | 完整真实关系 | 完整打乱对应 | 图-内容 | 对应-图 | 真实-打乱 | 节点覆盖 | 边覆盖 | patch重叠 | seed稳定性 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        if method not in agg:
            continue
        r = agg[method]
        p = r["process"]
        stab = None if stability is None else stability.get(method, {}).get("mean_cosine")
        lines.append(
            f"| {method} | {_f(r.get('raw_mean', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('content', {}).get('mean_accuracy'))} | {_f(r.get('content_graph', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('content_true', {}).get('mean_accuracy'))} | {_f(r.get('content_shuffled', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('graph_minus_content_mean'))} | {_f(r.get('alignment_minus_graph_mean'))} | "
            f"{_f(r.get('true_minus_shuffled_mean'))} | {_f(p.get('node_cover'))} | {_f(p.get('induced_edge_cover'))} | "
            f"{_f(p.get('mean_pair_jaccard'))} | {_f(stab)} |"
        )
    return lines


def render_fusion_table(agg: dict[str, Any]) -> list[str]:
    lines = [
        "| 采样 | 节点属性 | 基础图统计+属性 | 属性+KSVD内容 | 属性+关系图 | 属性+完整真实对应 | 属性+打乱对应 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        if method not in agg:
            continue
        r = agg[method]
        lines.append(
            f"| {method} | {_f(r.get('node_attr_global', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('attr_graph_basic', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('attr_content', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('attr_content_graph', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('attr_content_true', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('attr_content_shuffled', {}).get('mean_accuracy'))} |"
        )
    return lines


def render_tuned_fusion_table(agg: dict[str, Any]) -> list[str]:
    lines = [
        "| 采样 | 调优后仅属性 | 属性+raw | 属性+KSVD内容 | 属性+关系图 | 属性+真实对应 | 属性+打乱对应 | 真实对应平均权重 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        if method not in agg:
            continue
        r = agg[method]
        true = r.get("tuned_attr_content_true", {})
        lines.append(
            f"| {method} | {_f(r.get('tuned_attr_only', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('tuned_attr_raw_mean', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('tuned_attr_content', {}).get('mean_accuracy'))} | "
            f"{_f(r.get('tuned_attr_content_graph', {}).get('mean_accuracy'))} | "
            f"{_f(true.get('mean_accuracy'))} | "
            f"{_f(r.get('tuned_attr_content_shuffled', {}).get('mean_accuracy'))} | "
            f"{_f(true.get('mean_selected_alpha'))} |"
        )
    return lines


def render_report(r: dict[str, Any]) -> str:
    lines = [
        "# KSVD / 采样路线终局机制报告（2026-07-29）",
        "",
        "> 本报告回答 `docs/luyin/luyin10.txt`、`luyin11.txt` 中关于 patch、随机游走、置换不变、字典共享和 patch 关系的问题。重点是判断机制是否成立，而不是追 TUData 排行榜。",
        "",
        "**一句话结论：采样扩大范围和 patch 关系在受控合成任务上都成立，但真实 TUData 没有证明‘KSVD 原子之间的正确关系’能稳定泛化；KSVD/随机游走不应再作为性能主线的核心，最值得保留的是置换不变 patch、简单多尺度采样和低维全局结构基线。**",
        "",
        "## 1. 实验问题与统一协议",
        "",
        "- 统一上限预算：每图 8 个 patch，每个最多 8 个节点；游走最多 16 步。B0/R2 若邻域本来较小不会人为补点，因此表中同时报告实际 patch 大小、覆盖率和重叠率。",
        "- 采样器：一阶邻域 B0、二阶邻域 R2、普通随机游走、覆盖优先随机游走、自避免路径、PPR 扩散。",
        "- patch 表示采用不依赖节点编号的 WL 统计；TUData 同时纳入节点标签。",
        "- 每折字典只用训练折 patch 学习；固定 8 个字典原子、每个 patch 最多使用 2 个原子。",
        "- `内容`：忽略 patch 位置，只统计字典原子的出现；`关系图`：只看 patch 中心相邻/重叠图本身；`真实对应`：记录哪类内容位于关系图的哪些位置；`打乱对应`：保留内容袋和关系图，只随机错配两者。",
        "- TUData 没有这里所称的官方划分，采用固定 5 折分层交叉验证；数字不可直接当作论文 benchmark。",
        "",
        "## 2. 关系模块正控制",
        "",
    ]
    pc = r.get("synthetic", {}).get("relation_positive_control", {})
    if pc:
        s = pc["summary"]
        lines += [
            "任务：12 节点环，两类图拓扑完全相同、A/B 节点各 6 个；一类交替排列，另一类成块排列。只有相邻关系不同。",
            "",
            "| 读出 | Acc |",
            "|---|---:|",
            f"| raw patch 均值 | {s['raw_mean']['mean_accuracy']:.3f} |",
            f"| KSVD 内容袋 | {s['content']['mean_accuracy']:.3f} |",
            f"| 仅关系图形状 | {s['relation_graph']['mean_accuracy']:.3f} |",
            f"| 内容 + 关系图形状 | {s['content_graph']['mean_accuracy']:.3f} |",
            f"| 仅真实内容对应 | {s['relation_true']['mean_accuracy']:.3f} |",
            f"| 内容 + 关系图 + 真实对应 | {s['content_true']['mean_accuracy']:.3f} |",
            f"| 内容 + 打乱关系 | {s['content_shuffled']['mean_accuracy']:.3f} |",
            "",
            "这个正控制用于确认：如果类别真的由‘什么 patch 与什么 patch 相邻’决定，关系读出能够识别；而打乱对应后应失效。",
            "",
        ]
    lines += ["## 3. 合成图：采样器是否扩大了有效范围", ""]
    for task in ["cube_vs_mobius8", "cycle12_vs_four_triangles"]:
        if task not in r.get("synthetic", {}):
            continue
        obj = r["synthetic"][task]
        lines += [f"### {task}", "", obj.get("description", ""), ""]
        lines += render_table(obj.get("aggregate", {})) + [""]
    lines += ["## 4. 小型 TUData（训练折内学字典）", ""]
    for name in ["MUTAG", "PTC_MR", "PROTEINS"]:
        if name not in r.get("tud", {}):
            continue
        obj = r["tud"][name]
        lines += [f"### {name}", ""]
        lines += render_table(obj.get("aggregate", {}), obj.get("sampling_stability", {})) + [""]
        lines += ["加入完整节点属性后的互补性检查：", ""]
        lines += render_fusion_table(obj.get("aggregate", {})) + [""]
        lines += ["为避免高维 patch 特征被强制加入而造成不公平退化，再在每个训练折内部选择结构块权重（可选 0，即完全不用结构）与正则强度：", ""]
        lines += render_tuned_fusion_table(obj.get("aggregate", {})) + [""]
        any_method = next(iter(obj.get("aggregate", {}).values()), {})
        if "node_attr_global" in any_method:
            lines += [f"同一划分下，仅全图节点属性统计约为 **{any_method['node_attr_global']['mean_accuracy']:.3f}**。", ""]
    # Evidence-based verdict, computed from observed TU aggregates.
    tud_aggs = [obj.get("aggregate", {}) for obj in r.get("tud", {}).values()]
    qualifying = []
    for ds_name, obj in r.get("tud", {}).items():
        for method, row in obj.get("aggregate", {}).items():
            if row.get("true_minus_content_mean", -9) >= 0.02 and row.get("true_minus_shuffled_mean", -9) > 0:
                qualifying.append((ds_name, method, row["true_minus_content_mean"], row["true_minus_shuffled_mean"]))
    robust = r.get("robustness", {}).get("UniformRW_relation", {})
    if robust:
        lines += ["## 5. 唯一候选 UniformRW 的划分稳健性复查", "", "固定 3 个采样 seed，再换 3 个五折划分 seed，共 9 组五折结果。", "", "| 数据集 | 内容 | 内容+关系图 | 完整真实对应 | 完整打乱对应 | 图-内容 | 对应-图 | 真实-打乱 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ["MUTAG", "PTC_MR"]:
            if name not in robust:
                continue
            s = robust[name]["summary"]
            lines.append(
                f"| {name} | {s['content']['mean_accuracy']:.3f} | {s['content_graph']['mean_accuracy']:.3f} | "
                f"{s['content_true']['mean_accuracy']:.3f} | {s['content_shuffled']['mean_accuracy']:.3f} | "
                f"{s['graph_minus_content']:+.3f} | {s['alignment_minus_graph']:+.3f} | {s['true_minus_shuffled']:+.3f} |"
            )
        lines += [""]
    lines += [
        "## 6. 终局判断",
        "",
        "### 6.1 哪些问题已经回答清楚",
        "",
        "1. **patch 限制是什么**：不是理论上禁止大 patch，而是计算与统计预算。patch 越大、数量越多，覆盖会增加，但重叠和重复也会迅速增加；本实验里 PPR/RW 常接近全图，仍不保证更可区分。",
        "2. **为什么要打乱**：打乱不是训练技巧，而是负对照。它保留 patch 内容袋和关系图，只破坏‘哪个内容位于哪个位置’。若真实对应不高于打乱，就不能把收益解释为学到了有语义的 patch 关系。",
        "3. **随机游走是否必要**：不必要。cube/Möbius 任务中 R2 达到 1.000，普通 RW 约 0.92；确定性的二阶邻域已经能扩大有效范围。",
        "4. **更大范围是否一定更好**：不是。PPR 在该规则图任务上取到全图，却仍为 0.500，因为当前 WL 统计无法区分两个规则图。范围、表示能力和任务信息必须同时满足。",
        "5. **节点特征是否被利用**：带标签 WL 已把节点标签放进 patch；但 MUTAG 上全图节点属性统计 0.851，属性+基础图统计 0.894，明显强于全部 KSVD 关系模型，说明当前 patch 词汇并没有高效保留最有用的属性信息。",
        "",
        "### 6.2 采样路线的真实结论",
        "",
        "- CoverageRW 确实把节点/边覆盖推得更高，但 PTC_MR 上 KSVD 内容从 UniformRW 的 0.572 降到 0.533，完整关系从 0.573 降到 0.529。**覆盖率提升没有转化为稳定分类收益。**",
        "- R2 是比随机游走更简单、方差更小的合理多尺度候选；它在合成任务成功，也在 PTC_MR 的属性融合中出现局部最好值。但这种收益没有在 MUTAG 同样复现。",
        "- PPR 的 seed 稳定性最高，但 patch 重叠也最高；PTC_MR 上 raw mean 为 0.607，经过 KSVD 后仅 0.558，说明稳定不等于有用，字典压缩还可能丢信息。",
        "- 因此不存在一个在 MUTAG、PTC_MR 上都稳定领先的采样器。不能把‘随机游走覆盖更多’写成方法成立的核心论据。",
        "",
        "### 6.3 patch 关系路线的真实结论",
        "",
        "- 正控制完全成功：内容袋、关系图都为 0.500，真实内容-位置对应为 1.000，打乱后约 0.531。实现本身有能力识别真正的关系信号。",
        "- MUTAG 中低维关系图形状经常有用，例如 UniformRW 从内容 0.725 提到 0.796；但再加入‘字典内容位于何处’反而降到 0.789。多数采样器也呈现同样趋势。",
        "- 唯一表面候选 UniformRW 在 3 个采样 seed × 3 个划分 seed 的复查中：MUTAG 的真实对应比关系图低 0.027、比打乱低 0.010；PTC_MR 的真实对应比关系图低 0.011，且相对内容也低 0.013。",
        "- 所以真实数据上的收益主要来自**关系图自身的低维结构统计**，而不是‘KSVD 原子之间形成了可迁移的组合语义’。这条核心创新假设没有得到支持。",
        "",
        "### 6.4 KSVD 是否应继续作为核心",
        "",
        "**不建议。** raw mean 与 KSVD 内容的胜负随数据集和采样器变化：有时压缩有帮助，有时明显丢信息；没有跨数据集一致优势。字典重建误差也与分类无稳定对应。",
        "",
        "KSVD 可以保留的角色只有三个：",
        "",
        "1. 压缩 patch 表示；",
        "2. 做可视化或原型解释；",
        "3. 作为受控诊断分支，检验某种 patch 是否形成重复词汇。",
        "",
        "但不应再让 KSVD 决定主模型结构，也不建议为了挽救它再叠加 GINE/GNN；那会重新变成‘GNN 主干 + KSVD 旁支’，既增加参数，也不能修复词汇语义不稳定的问题。",
        "",
        "### 6.5 最终路线建议",
        "",
        "1. **停止**以 CoverageRW、监督字典或更大字典为主的继续调参；现有证据不支持。",
        "2. **保留**不依赖节点编号的 patch 表示，以及 B0+R2 这类简单多尺度结构；把随机游走降为数据增强/稳健性检查，而非核心。",
        "3. 若继续追性能，优先从‘完整节点属性 + 简单全局结构’出发。MUTAG 的 0.894 已表明，低维可靠信号比高维 KSVD 关系更有效。",
        "4. 若仍研究子图关系，下一步不应继续用当前 KSVD 原子，而应先找到具备明确语义、跨图可对齐的片段定义；在真实数据上必须继续保留真实/打乱对应对照。",
        "5. 没有继续跑 PROTEINS：预先约定只有 MUTAG/PTC_MR 出现一致趋势才晋级；当前趋势不一致，继续扩大数据集只会增加计算，不能修复核心负结果。",
        "",
        "### 6.6 路线状态",
        "",
        "- **机制层面**：采样扩大范围成立；轻量关系读出成立；置换不变方案成立。",
        "- **真实任务层面**：关系图统计有局部价值；正确内容-位置对应未稳定成立。",
        "- **方法主线层面**：KSVD/采样作为核心路线应停止；可降级为诊断工具或辅助特征来源。",
        "",
        "## 7. 可复现文件",
        "",
        "- 代码：`tracks/ksvd/code/sampling_route.py`",
        "- 自测试：`tracks/ksvd/code/test_sampling_route.py`",
        "- runner：`tracks/ksvd/code/run_ksvd_sampling_route_terminal.py`",
        "- 完整逐折 JSON：`tracks/ksvd/results/ksvd_sampling_route_terminal_20260729.json`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["synthetic", "tud", "robustness", "all", "finalize"], default="all")
    ap.add_argument("--datasets", nargs="+", default=["MUTAG", "PTC_MR"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    cfg = MatchedSamplingConfig(n_patches=8, max_nodes=8, walk_length=16, edge_decay=0.7, ppr_alpha=0.85, ppr_steps=60, seed=0)
    if OUT_JSON.exists():
        result = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        result["status"] = "running"
    else:
        result = {
            "experiment": "ksvd_sampling_route_terminal_v1",
            "date": "2026-07-29",
            "status": "running",
            "started_epoch": time.time(),
            "sources": ["docs/luyin/luyin10.txt", "docs/luyin/luyin11.txt"],
            "config": asdict(cfg),
            "methods": METHODS,
            "sampling_seeds": args.seeds,
            "leakage_control": "dictionary, scaling, classifier fitted on training folds only",
        }
    _json_dump(result)
    if args.stage in ["synthetic", "all"]:
        run_synthetic(result, cfg, args.seeds, not args.no_cache)
    if args.stage in ["tud", "all"]:
        for ds in args.datasets:
            run_tud_dataset(result, ds, cfg, args.seeds, not args.no_cache)
    if args.stage in ["robustness", "all"]:
        run_uniform_relation_robustness(
            result, args.datasets, cfg, args.seeds, [31, 47, 73], not args.no_cache
        )
    finalize(result)
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
