"""Paired KSVD readout/classifier study on an official MolHIV subset.

The dictionary/sampler are held fixed while graph-level sparse-code statistics
and small downstream classifiers are varied.  Every KSVD result has a same-seed
random-patch control built from the identical training-patch pool.  Only the
OGB official validation split is evaluated; test graphs are not encoded.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from .data_molhiv import load_molhiv
from .graph_level import (
    GraphLevelConfig,
    bundle_to_Y,
    collect_train_Y,
    sample_patches_graph_level,
    sparse_code_patch_matrix,
    sparse_code_readouts,
)
from .ksvd import ksvd
from .run_molhiv_next_round import size_feat


def _csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _seeds(raw: str) -> list[int]:
    return [int(x) for x in _csv(raw)]


def _random_patch_dictionary(Y: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(Y.shape[1], size=n_atoms, replace=Y.shape[1] < n_atoms)
    D = Y[:, idx].astype(np.float64).copy()
    return D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)


def _readouts(X: np.ndarray, patch_errors: np.ndarray) -> dict[str, np.ndarray]:
    return sparse_code_readouts(X, patch_errors)


def _predict_score(clf: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        classes = list(clf.classes_) if hasattr(clf, "classes_") else list(clf[-1].classes_)
        return p[:, classes.index(1.0 if 1.0 in classes else 1)]
    return np.asarray(clf.decision_function(X)).reshape(-1)


def _fit_auc(kind: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, seed: int) -> dict[str, float]:
    if kind == "logistic":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000, random_state=seed, class_weight="balanced", C=1.0
            ),
        )
        clf.fit(Xtr, ytr)
    elif kind == "rbf":
        clf = make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
        )
        clf.fit(Xtr, ytr)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=seed,
        )
        clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    else:
        raise ValueError(f"unknown classifier {kind!r}")
    return {
        "train_auc": float(roc_auc_score(ytr, _predict_score(clf, Xtr))),
        "valid_auc": float(roc_auc_score(yva, _predict_score(clf, Xva))),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["readout"], row["classifier"])].append(row)
    out: dict[str, Any] = {}
    for (family, readout, classifier), rs in sorted(groups.items()):
        key = f"{family}/{readout}/{classifier}"
        out[key] = {
            "n": len(rs),
            "mean_valid_auc": float(np.mean([r["valid_auc"] for r in rs])),
            "std_valid_auc": float(np.std([r["valid_auc"] for r in rs], ddof=1)) if len(rs) > 1 else 0.0,
            "mean_delta_vs_size": float(np.mean([r["delta_valid_vs_size"] for r in rs])),
            "seeds": [r["seed"] for r in rs],
        }
    paired: dict[str, Any] = {}
    lookup = {(r["family"], r["readout"], r["classifier"], r["seed"]): r for r in rows}
    combos = sorted({(r["readout"], r["classifier"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    for readout, classifier in combos:
        deltas = []
        for seed in seeds:
            a = lookup.get(("ksvd", readout, classifier, seed))
            b = lookup.get(("random_patch", readout, classifier, seed))
            if a is not None and b is not None:
                deltas.append(float(a["valid_auc"] - b["valid_auc"]))
        if deltas:
            paired[f"{readout}/{classifier}"] = {
                "n": len(deltas),
                "ksvd_minus_random_patch": deltas,
                "mean": float(np.mean(deltas)),
                "median": float(np.median(deltas)),
                "std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "wins": int(np.sum(np.asarray(deltas) > 0)),
            }
    return {"groups": out, "paired": paired}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=8000, help="fixed stratified total subset; 0 means full")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--dict-seeds", default="0,1,2,3,4")
    ap.add_argument("--n-atoms", type=int, default=8)
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument("--max-train-patches", type=int, default=4000)
    ap.add_argument("--max-patches-per-graph", type=int, default=8)
    ap.add_argument("--readouts", default="max,basic,tail,moments,recon,rich_no_recon,rich")
    ap.add_argument("--classifiers", default="logistic,rbf,hgb")
    ap.add_argument("--families", default="random_patch,ksvd")
    ap.add_argument("--output", default=None)
    ap.add_argument("--save-features", default=None, help="optional NPZ containing graph-level readout matrices")
    args = ap.parse_args()

    t0 = time.time()
    seeds = _seeds(args.dict_seeds)
    readout_names = _csv(args.readouts)
    classifiers = _csv(args.classifiers)
    families = _csv(args.families)
    allowed_readouts = {"max", "basic", "tail", "moments", "recon", "rich_no_recon", "rich"}
    if set(readout_names) - allowed_readouts:
        raise ValueError(f"unknown readouts: {set(readout_names) - allowed_readouts}")
    if set(classifiers) - {"logistic", "rbf", "hgb"}:
        raise ValueError("classifiers must be logistic,rbf,hgb")
    if set(families) - {"random_patch", "pca", "ksvd"}:
        raise ValueError("families must be random_patch,pca,ksvd")

    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    bundle = load_molhiv(max_graphs=max_graphs, seed=args.data_seed, with_features=True)
    graphs, y = bundle.graphs, bundle.y
    tr, va = np.asarray(bundle.split["train"]), np.asarray(bundle.split["valid"])
    encode_indices = np.concatenate([tr, va])
    size = size_feat(graphs)

    cfg = GraphLevelConfig(
        n_atoms=args.n_atoms,
        T=args.sparsity,
        T_min=1,
        ksvd_iter=args.ksvd_iter,
        seed=args.sampler_seed,
        max_train_patches=args.max_train_patches,
        max_patches_per_graph=args.max_patches_per_graph,
        patch_feat="wl_chem_ring",
        normalize_patches=True,
        readout_mode="pool",
        pool="max",
    )
    Ytr, patch_stats = collect_train_Y(
        graphs, tr, cfg, node_feats=bundle.node_feats, edge_feats=bundle.edge_feats
    )
    dictionaries: list[tuple[str, int, np.ndarray, dict[str, Any]]] = []
    archive: dict[str, np.ndarray] = {}
    if "pca" in families:
        U, _, _ = np.linalg.svd(Ytr, full_matrices=False)
        Dpca = U[:, : args.n_atoms]
        dictionaries.append(("pca", 0, Dpca, {"deterministic": True}))
        archive["pca"] = Dpca
    for seed in seeds:
        if "random_patch" in families:
            D = _random_patch_dictionary(Ytr, args.n_atoms, seed)
            dictionaries.append(("random_patch", seed, D, {}))
            archive[f"random_patch_seed{seed}"] = D
        if "ksvd" in families:
            D, _, info = ksvd(
                Ytr,
                n_atoms=args.n_atoms,
                T=args.sparsity,
                T_min=1,
                n_iter=args.ksvd_iter,
                seed=seed,
            )
            dictionaries.append(("ksvd", seed, D, info))
            archive[f"ksvd_seed{seed}"] = D

    # Graph patches are sampled/vectorized once, then reused across every D.
    features: dict[tuple[str, int, str], np.ndarray] = {}
    collectors: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    for count, raw_i in enumerate(encode_indices, 1):
        i = int(raw_i)
        patches, _ = sample_patches_graph_level(graphs[i], cfg, seed=cfg.seed + i * 13)
        Y, _ = bundle_to_Y(
            graphs[i],
            patches,
            cfg.max_nodes,
            cfg.order_mode,
            patch_feat=cfg.patch_feat,
            node_feat=bundle.node_feats[i],
            edge_feat=bundle.edge_feats[i],
        )
        for family, seed, D, _ in dictionaries:
            Yn, X = sparse_code_patch_matrix(Y, D, cfg)
            denom = np.maximum(np.linalg.norm(Yn, axis=0), 1e-12)
            errors = np.linalg.norm(Yn - D @ X, axis=0) / denom
            values = _readouts(X, errors)
            for name in readout_names:
                collectors[(family, seed, name)].append(values[name])
        if count % 500 == 0 or count == len(encode_indices):
            print(f"encoded {count}/{len(encode_indices)}", flush=True)
    for key, values in collectors.items():
        features[key] = np.stack(values, axis=0)

    # encode_indices is train then valid, so local slices are deterministic.
    ntr = len(tr)
    local_tr = np.arange(ntr)
    local_va = np.arange(ntr, ntr + len(va))
    size_local = size[encode_indices]
    baselines = {
        classifier: _fit_auc(
            classifier,
            size_local[local_tr],
            y[tr],
            size_local[local_va],
            y[va],
            args.data_seed,
        )
        for classifier in classifiers
    }
    rows: list[dict[str, Any]] = []
    dictionary_info: dict[str, Any] = {}
    for family, seed, _, info in dictionaries:
        dictionary_info[f"{family}_seed{seed}"] = info
        for readout in readout_names:
            X = features[(family, seed, readout)]
            Xplus = np.hstack([X, size_local])
            for classifier in classifiers:
                result = _fit_auc(
                    classifier,
                    Xplus[local_tr],
                    y[tr],
                    Xplus[local_va],
                    y[va],
                    seed,
                )
                row = {
                    "family": family,
                    "seed": seed,
                    "readout": readout,
                    "classifier": classifier,
                    "feature_dim": int(Xplus.shape[1]),
                    **result,
                    "size_valid_auc": baselines[classifier]["valid_auc"],
                    "delta_valid_vs_size": result["valid_auc"] - baselines[classifier]["valid_auc"],
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parents[1]
        / "results"
        / "molhiv"
        / f"ksvd_readout_n{len(graphs)}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **archive)
    result = {
        "protocol_id": "molhiv-ksvd-readout-v1",
        "test_policy": "official test graphs were not encoded or evaluated",
        "meta": bundle.meta,
        "config": vars(args),
        "patch_stats": patch_stats,
        "baselines": baselines,
        "dictionary_info": dictionary_info,
        "rows": rows,
        "summary": _summary(rows),
        "elapsed_sec": time.time() - t0,
        "dictionary_archive": str(output.with_suffix(".npz")),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_features:
        feature_payload: dict[str, np.ndarray] = {
            "encode_indices": np.asarray(encode_indices, dtype=np.int64),
            "labels": np.asarray(y[encode_indices], dtype=np.int64),
            "train_count": np.asarray([len(tr)], dtype=np.int64),
        }
        for (family, seed, readout), matrix in features.items():
            feature_payload[f"{family}_seed{seed}_{readout}"] = np.asarray(matrix, dtype=np.float32)
        save_path = Path(args.save_features)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **feature_payload)
        print(f"wrote feature cache {save_path}", flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
