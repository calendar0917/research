"""First real-data screen for a content-dictionary + relation-dictionary route.

This is deliberately a small, label-clean classification screen on the
MolHIV 8k development folds.  Both dictionaries see only molecules from the
training part of each fold.  The classifier is kept linear so that a gain is
attributable to the representation rather than a powerful downstream network.

The main control shuffles the binding between patch pairs and their observed
relations while preserving, inside every molecule, patch count, relation-event
count, and the multiset of relation attributes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_molhiv import load_molhiv
from .dual_dictionary import (
    relation_event_vectors,
    relation_readout,
    shuffled_patch_code_binding,
)
from .graph_level import GraphLevelConfig, bundle_to_Y, sample_patches_graph_level
from .ksvd import _omp, ksvd


RELATION_ATTRIBUTE_NAMES = (
    "patch_overlap",
    "exclusive_cross_patch_bond",
    "shared_node_fraction",
    "cross_bond_fraction",
)


@dataclass
class PatchGraph:
    index: int
    label: float
    Y: np.ndarray
    pair_index: np.ndarray
    relation_attributes: np.ndarray


def _relation_events(g, patches: list[set[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Observed overlap / cross-bond events between sampled patches."""
    pairs: list[tuple[int, int]] = []
    attributes: list[np.ndarray] = []
    for i, left in enumerate(patches):
        for j in range(i + 1, len(patches)):
            right = patches[j]
            shared = left & right
            left_only, right_only = left - right, right - left
            cross_bonds = sum(
                1
                for u in left_only
                for v in g.neighbors(u)
                if v in right_only
            )
            if not shared and not cross_bonds:
                continue
            shared_fraction = len(shared) / max(1.0, float(min(len(left), len(right))))
            possible_cross = max(1.0, float(len(left_only) * len(right_only)))
            attributes.append(
                np.array(
                    [
                        float(bool(shared)),
                        float(bool(cross_bonds)),
                        shared_fraction,
                        cross_bonds / possible_cross,
                    ],
                    dtype=np.float64,
                )
            )
            pairs.append((i, j))
    if not pairs:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, len(RELATION_ATTRIBUTE_NAMES)), dtype=np.float64),
        )
    return np.asarray(pairs, dtype=np.int64), np.stack(attributes, axis=0)


def _patch_graph(bundle, index: int, cfg: GraphLevelConfig) -> PatchGraph | None:
    g = bundle.graphs[index]
    sampled, _ = sample_patches_graph_level(g, cfg, seed=cfg.seed + index)
    vectors: list[np.ndarray] = []
    patches: list[set[int]] = []
    for patch in sampled.node_sets[: cfg.max_patches_per_graph]:
        Y, meta = bundle_to_Y(
            g,
            SimpleNamespace(node_sets=[patch]),
            cfg.max_nodes,
            cfg.order_mode,
            patch_feat=cfg.patch_feat,
            node_feat=bundle.node_feats[index],
            edge_feat=bundle.edge_feats[index],
        )
        if not meta["n_cols"]:
            continue
        y = Y[:, 0]
        norm = float(np.linalg.norm(y))
        if norm <= 1e-12:
            continue
        vectors.append(y / norm)
        patches.append(set(patch))
    if len(vectors) < 2:
        return None
    pair_index, attributes = _relation_events(g, patches)
    return PatchGraph(
        index=index,
        label=float(bundle.y[index]),
        Y=np.stack(vectors, axis=1),
        pair_index=pair_index,
        relation_attributes=attributes,
    )


def _stratified_pick(indices: np.ndarray, labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Use a small, reproducible development screen without changing class prior."""
    indices = np.asarray(indices, dtype=np.int64)
    if n >= len(indices):
        return indices.copy()
    rng = np.random.default_rng(seed)
    pos = indices[labels[indices] > 0.5]
    neg = indices[labels[indices] <= 0.5]
    take_pos = min(len(pos), int(round(n * len(pos) / len(indices))))
    take_neg = min(len(neg), n - take_pos)
    if take_neg < n - take_pos:
        take_pos = min(len(pos), n - take_neg)
    out = np.concatenate(
        [
            rng.choice(pos, size=take_pos, replace=False),
            rng.choice(neg, size=take_neg, replace=False),
        ]
    )
    rng.shuffle(out)
    return out.astype(np.int64)


def _collect(bundle, indices, cfg, max_graphs: int, seed: int) -> list[PatchGraph]:
    picked = _stratified_pick(np.asarray(indices), bundle.y, max_graphs, seed)
    records = []
    for index in picked.tolist():
        item = _patch_graph(bundle, int(index), cfg)
        if item is not None:
            records.append(item)
    if len(records) < 2:
        raise RuntimeError("too few graphs with at least two nonempty patches")
    return records


def _concat_patch_columns(records: list[PatchGraph]) -> np.ndarray:
    return np.concatenate([r.Y for r in records], axis=1)


def _fit_random_dictionary(Y: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picked = rng.choice(Y.shape[1], size=n_atoms, replace=Y.shape[1] < n_atoms)
    D = Y[:, picked].copy()
    return D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)


def _fit_pca(Y: np.ndarray, n_components: int, seed: int) -> PCA:
    n = min(int(n_components), Y.shape[0], Y.shape[1])
    if n < 1:
        raise RuntimeError("PCA needs at least one training event")
    return PCA(n_components=n, random_state=seed).fit(Y.T)


def _random_projector(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((input_dim, output_dim)))
    return q[:, : min(input_dim, output_dim)]


def _graph_events(
    record: PatchGraph,
    content_encoder,
    *,
    shuffled: bool,
    seed: int,
) -> np.ndarray:
    Z = np.asarray(content_encoder(record.Y), dtype=np.float64)
    pairs, attrs = record.pair_index, record.relation_attributes
    if shuffled:
        Z = shuffled_patch_code_binding(Z, np.random.default_rng(seed + record.index))
    return relation_event_vectors(Z, pairs, attrs)


def _concat_events(records: list[PatchGraph], content_encoder, *, shuffled: bool, seed: int) -> np.ndarray:
    pieces = [_graph_events(r, content_encoder, shuffled=shuffled, seed=seed) for r in records]
    pieces = [x for x in pieces if x.shape[1]]
    if not pieces:
        # The dimensions below are determined by one content code; no relation
        # events means this route is inapplicable to the requested sample.
        example = np.asarray(content_encoder(records[0].Y))
        return np.zeros((3 * example.shape[0] + len(RELATION_ATTRIBUTE_NAMES), 0))
    return np.concatenate(pieces, axis=1)


def _features(
    records: list[PatchGraph],
    content_encoder,
    relation_encoder=None,
    *,
    shuffled: bool = False,
    seed: int = 0,
) -> np.ndarray:
    rows = []
    for record in records:
        Z = np.asarray(content_encoder(record.Y), dtype=np.float64)
        row = relation_readout(Z)
        if relation_encoder is not None:
            events = _graph_events(record, content_encoder, shuffled=shuffled, seed=seed)
            row = np.concatenate([row, relation_readout(relation_encoder(events))])
        rows.append(row)
    return np.stack(rows, axis=0)


def _score(train_X, train_y, valid_X, valid_y, C: float) -> dict[str, float | int]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
            solver="liblinear",
        ),
    )
    model.fit(train_X, train_y)
    prob = model.predict_proba(valid_X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(valid_y, prob)),
        "average_precision": float(average_precision_score(valid_y, prob)),
        "n_features": int(train_X.shape[1]),
    }


def _labels(records: list[PatchGraph]) -> np.ndarray:
    return np.asarray([r.label for r in records], dtype=np.float64)


def _resolve_repo_relative_path(value: str) -> Path:
    """Accept a repository-relative path from either repo or track cwd."""
    path = Path(value)
    if path.exists():
        return path
    candidate = Path(__file__).resolve().parents[3] / path
    if candidate.exists():
        return candidate
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold-cache",
        default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fit-graphs", type=int, default=600)
    ap.add_argument("--eval-graphs", type=int, default=600)
    ap.add_argument("--n-content-atoms", type=int, default=12)
    ap.add_argument("--content-sparsity", type=int, default=3)
    ap.add_argument("--n-relation-atoms", type=int, default=12)
    ap.add_argument("--relation-sparsity", type=int, default=3)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument("--classifier-c", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    started = time.time()
    bundle = load_molhiv(with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV atom and bond features are required")
    folds = np.load(_resolve_repo_relative_path(args.fold_cache))
    train_idx = folds[f"fold_{args.fold}_train_indices"]
    valid_idx = folds[f"fold_{args.fold}_valid_indices"]
    cfg = GraphLevelConfig(
        seed=args.seed,
        max_patches_per_graph=8,
        patch_feat="wl_chem_ring",
        normalize_patches=True,
    )
    train = _collect(bundle, train_idx, cfg, args.fit_graphs, args.seed + 100)
    valid = _collect(bundle, valid_idx, cfg, args.eval_graphs, args.seed + 200)
    y_train, y_valid = _labels(train), _labels(valid)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
        raise RuntimeError("both train and validation screens require both labels")

    Y_train = _concat_patch_columns(train)
    content_D, _, content_info = ksvd(
        Y_train,
        n_atoms=args.n_content_atoms,
        T=args.content_sparsity,
        T_min=1,
        n_iter=args.ksvd_iter,
        seed=args.seed,
    )
    random_content_D = _fit_random_dictionary(Y_train, args.n_content_atoms, args.seed + 1)
    content_pca = _fit_pca(Y_train, args.n_content_atoms, args.seed)
    content_encoders = {
        "ksvd": lambda Y: np.stack(
            [_omp(content_D, Y[:, i], args.content_sparsity) for i in range(Y.shape[1])], axis=1
        ),
        "random_dictionary": lambda Y: np.stack(
            [_omp(random_content_D, Y[:, i], args.content_sparsity) for i in range(Y.shape[1])], axis=1
        ),
        "pca": lambda Y: content_pca.transform(Y.T).T,
    }

    # Relation dictionaries are fitted separately for each content route; this
    # prevents a PCA/KSVD comparison from silently sharing a favored encoder.
    relation_models: dict[str, dict[str, object]] = {}
    for content_name, content_encoder in content_encoders.items():
        events = _concat_events(train, content_encoder, shuffled=False, seed=args.seed + 300)
        if events.shape[1] < 2:
            raise RuntimeError("the selected train graphs contain no relation events")
        pca = _fit_pca(events, args.n_relation_atoms, args.seed + 10)
        D, _, info = ksvd(
            events,
            n_atoms=args.n_relation_atoms,
            T=args.relation_sparsity,
            T_min=1,
            n_iter=args.ksvd_iter,
            seed=args.seed + 10,
        )
        random_D = _fit_random_dictionary(events, args.n_relation_atoms, args.seed + 11)
        relation_models[content_name] = {
            "pca": pca,
            "ksvd_D": D,
            "ksvd_info": info,
            "random_D": random_D,
            "n_train_events": int(events.shape[1]),
            "event_dim": int(events.shape[0]),
        }

    # Matched negative control: its second dictionary is itself fitted on
    # relation events whose endpoint binding has been shuffled in the training
    # graphs.  Reusing the true-relation dictionary here would confound a
    # changed input distribution with the effect of relation binding.
    shuffled_events = _concat_events(
        train, content_encoders["ksvd"], shuffled=True, seed=args.seed + 400
    )
    shuffled_D, _, shuffled_info = ksvd(
        shuffled_events,
        n_atoms=args.n_relation_atoms,
        T=args.relation_sparsity,
        T_min=1,
        n_iter=args.ksvd_iter,
        seed=args.seed + 20,
    )
    shuffled_pca = _fit_pca(shuffled_events, args.n_relation_atoms, args.seed + 20)
    relation_models["ksvd_shuffled"] = {
        "pca": shuffled_pca,
        "ksvd_D": shuffled_D,
        "ksvd_info": shuffled_info,
        "n_train_events": int(shuffled_events.shape[1]),
        "event_dim": int(shuffled_events.shape[0]),
    }

    def relation_pca(name: str):
        pca = relation_models[name]["pca"]
        return lambda E: pca.transform(E.T).T if E.shape[1] else np.zeros((pca.n_components_, 0))

    def relation_ksvd(name: str):
        D = relation_models[name]["ksvd_D"]
        return lambda E: np.stack(
            [_omp(D, E[:, i], args.relation_sparsity) for i in range(E.shape[1])], axis=1
        ) if E.shape[1] else np.zeros((D.shape[1], 0))

    def relation_random(name: str):
        D = relation_models[name]["random_D"]
        return lambda E: np.stack(
            [_omp(D, E[:, i], args.relation_sparsity) for i in range(E.shape[1])], axis=1
        ) if E.shape[1] else np.zeros((D.shape[1], 0))

    method_features = {
        "ksvd_content_only": (
            _features(train, content_encoders["ksvd"]),
            _features(valid, content_encoders["ksvd"]),
        ),
        "ksvd_raw_relation": (
            _features(train, content_encoders["ksvd"], lambda E: E),
            _features(valid, content_encoders["ksvd"], lambda E: E),
        ),
        "ksvd_pca_relation": (
            _features(train, content_encoders["ksvd"], relation_pca("ksvd")),
            _features(valid, content_encoders["ksvd"], relation_pca("ksvd")),
        ),
        "ksvd_pca_relation_shuffled": (
            _features(train, content_encoders["ksvd"], relation_pca("ksvd_shuffled"), shuffled=True, seed=args.seed + 400),
            _features(valid, content_encoders["ksvd"], relation_pca("ksvd_shuffled"), shuffled=True, seed=args.seed + 400),
        ),
        "dual_ksvd": (
            _features(train, content_encoders["ksvd"], relation_ksvd("ksvd")),
            _features(valid, content_encoders["ksvd"], relation_ksvd("ksvd")),
        ),
        "dual_ksvd_shuffled_relation": (
            _features(train, content_encoders["ksvd"], relation_ksvd("ksvd_shuffled"), shuffled=True, seed=args.seed + 400),
            _features(valid, content_encoders["ksvd"], relation_ksvd("ksvd_shuffled"), shuffled=True, seed=args.seed + 400),
        ),
        "pca_content_pca_relation": (
            _features(train, content_encoders["pca"], relation_pca("pca")),
            _features(valid, content_encoders["pca"], relation_pca("pca")),
        ),
        "random_dictionary_relation_dictionary": (
            _features(train, content_encoders["random_dictionary"], relation_random("random_dictionary")),
            _features(valid, content_encoders["random_dictionary"], relation_random("random_dictionary")),
        ),
    }
    results = {
        name: _score(Xtr, y_train, Xva, y_valid, args.classifier_c)
        for name, (Xtr, Xva) in method_features.items()
    }
    result = {
        "protocol_id": "molhiv-dual-content-relation-dictionary-screen-v1",
        "config": vars(args),
        "design": {
            "patch_features": "wl_chem_ring (atom and bond attributes retained)",
            "content_dictionary": "K-SVD on individual chemistry-aware patches",
            "relation_dictionary": "K-SVD on observed unordered patch-pair events",
            "classifier": "same scaled logistic classifier for every representation",
            "leakage_guard": "content and relation dictionaries fit on train-fold graphs only",
            "shuffle_control": "within-graph patch-code to relation-endpoint binding deranged; patch-code bag, relation graph, and relation attributes preserved",
        },
        "relation_attribute_names": RELATION_ATTRIBUTE_NAMES,
        "n_train_graphs": len(train),
        "n_valid_graphs": len(valid),
        "train_positive_rate": float(y_train.mean()),
        "valid_positive_rate": float(y_valid.mean()),
        "n_train_patches": int(Y_train.shape[1]),
        "content_ksvd_fit": content_info,
        "relation_models": {
            name: {
                key: value
                for key, value in model.items()
                if key not in {"pca", "ksvd_D", "random_D"}
            }
            for name, model in relation_models.items()
        },
        "methods": results,
        "paired_summary": {
            "dual_minus_content_only_roc_auc": results["dual_ksvd"]["roc_auc"] - results["ksvd_content_only"]["roc_auc"],
            "dual_minus_pca_relation_roc_auc": results["dual_ksvd"]["roc_auc"] - results["ksvd_pca_relation"]["roc_auc"],
            "dual_minus_shuffled_roc_auc": results["dual_ksvd"]["roc_auc"] - results["dual_ksvd_shuffled_relation"]["roc_auc"],
            "pca_relation_minus_shuffled_roc_auc": results["ksvd_pca_relation"]["roc_auc"] - results["ksvd_pca_relation_shuffled"]["roc_auc"],
            "dual_minus_pca_content_pca_relation_roc_auc": results["dual_ksvd"]["roc_auc"] - results["pca_content_pca_relation"]["roc_auc"],
        },
        "elapsed_sec": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["methods"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
