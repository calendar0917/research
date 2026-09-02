"""Feature-exact rooted motif vocabulary with nested scaffold stability selection.

This is an independent, GNN-free and KSVD-free MolHIV development probe.
Each atom receives deterministic rooted local-environment signatures at radii
0..3 from all nine OGB atom features and all three OGB bond features.  Full
128-bit signatures are mapped to observed vocabulary entries; they are never
folded into a fixed hash table, so ordinary fingerprint collisions are absent.

For every outer official-train scaffold fold, the usable vocabulary is built
from outer-fit graphs only.  Supervised stability selection uses only three
nested scaffold partitions of the outer-fit data.  Official valid/test graphs
are never accessed by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import _patch_torch_load_weights_only


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def digest(payload: bytes) -> bytes:
    return hashlib.blake2b(payload, digest_size=16, person=b"molhiv-motif-v1").digest()


def feature_bytes(values: np.ndarray) -> bytes:
    a = np.asarray(values, dtype=np.int16).reshape(-1)
    return struct.pack("<H", len(a)) + a.tobytes()


def rooted_signatures(node_feat: np.ndarray, edge_index: np.ndarray, edge_feat: np.ndarray, max_radius: int = 3) -> list[list[bytes]]:
    n = int(node_feat.shape[0])
    nbrs: list[list[tuple[int, bytes]]] = [[] for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        eb = feature_bytes(edge_feat[k])
        nbrs[u].append((v, eb))
        nbrs[v].append((u, eb))
    labels = [digest(b"A" + feature_bytes(node_feat[u])) for u in range(n)]
    out = [labels.copy()]
    for radius in range(1, max_radius + 1):
        nxt: list[bytes] = []
        for u in range(n):
            messages = sorted(eb + labels[v] for v, eb in nbrs[u])
            payload = b"R" + bytes([radius]) + labels[u] + struct.pack("<H", len(messages)) + b"".join(messages)
            nxt.append(digest(payload))
        labels = nxt
        out.append(labels.copy())
    return out


def z_association(x_binary: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    # Jeffreys smoothing avoids infinite scores for rare motifs.
    c1 = np.asarray(x_binary[pos].sum(axis=0)).ravel()
    c0 = np.asarray(x_binary[neg].sum(axis=0)).ravel()
    p1 = (c1 + 0.5) / (len(pos) + 1.0)
    p0 = (c0 + 0.5) / (len(neg) + 1.0)
    pooled = (c1 + c0 + 1.0) / (len(pos) + len(neg) + 2.0)
    se = np.sqrt(np.maximum(pooled * (1.0 - pooled) * (1.0 / max(1, len(pos)) + 1.0 / max(1, len(neg))), 1e-12))
    return (p1 - p0) / se


def stable_scores(x_binary: sparse.csr_matrix, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    z_rows = []
    meta = []
    for inner, (_, held) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        held = np.asarray(held, dtype=np.int64)
        z_rows.append(z_association(x_binary[held], y[held]))
        meta.append({"inner_partition": inner, "n": int(len(held)), "n_positive": int(y[held].sum()), "n_groups": int(len(set(groups[held].tolist())))})
    z = np.stack(z_rows, axis=0)
    consistent = np.all(z > 0, axis=0) | np.all(z < 0, axis=0)
    score = np.where(consistent, np.min(np.abs(z), axis=0), 0.0)
    direction = np.sign(np.sum(z, axis=0))
    return score, direction, meta


def transformed(x: sparse.csr_matrix, mode: str) -> sparse.csr_matrix:
    z = x.astype(np.float32, copy=True)
    if mode == "binary":
        z.data[:] = 1.0
    elif mode == "logcount":
        z.data = np.log1p(z.data)
    else:
        raise ValueError(mode)
    return z


def fit_eval(x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_held: sparse.csr_matrix, y_held: np.ndarray, c: float) -> tuple[dict[str, Any], np.ndarray]:
    model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0)
    model.fit(x_fit, y_fit)
    pf = model.predict_proba(x_fit)[:, 1]
    ph = model.predict_proba(x_held)[:, 1]
    return {
        "trainable_parameters": int(x_fit.shape[1] + 1),
        "fit_auc": float(roc_auc_score(y_fit, pf)),
        "heldout_auc": float(roc_auc_score(y_held, ph)),
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "coefficient_sha256": sha(np.asarray(model.coef_, dtype=np.float64)),
        "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
    }, ph


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a = np.asarray([r["heldout_auc"] for r in rows], dtype=float)
    return {"fold_auc": a.tolist(), "mean_auc": float(a.mean()), "sample_std_auc": float(a.std(ddof=1)), "min_fold_auc": float(a.min()), "max_fold_auc": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/stable_exact_motif_fulltrain_probe_20260729.json")
    args = ap.parse_args()
    t0 = time.time()
    repo = Path(__file__).resolve().parents[3]

    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    split = dataset.get_idx_split()
    train_graph_indices = np.asarray(split["train"], dtype=np.int64)
    forbidden = np.concatenate([np.asarray(split["valid"], dtype=np.int64), np.asarray(split["test"], dtype=np.int64)])
    labels_all = np.asarray(dataset.labels).reshape(-1).astype(np.int64)
    y = labels_all[train_graph_indices]
    position = np.full(len(dataset), -1, dtype=np.int64)
    position[train_graph_indices] = np.arange(len(train_graph_indices), dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["official_train_indices"], dtype=np.int64), train_graph_indices):
            raise ValueError("official train mismatch")
        groups = np.asarray(z["train_scaffold_groups"]).astype(str)
        outer = []
        for fold in range(3):
            fit_graph = np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)
            held_graph = np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)
            outer.append((position[fit_graph], position[held_graph], fit_graph, held_graph))
    if np.any(position[forbidden] >= 0):
        raise AssertionError("forbidden split mapped into train rows")

    # Extract only official-train graphs.  A global column map is merely a
    # storage index for deterministic signatures; each outer vocabulary below
    # is still restricted by outer-fit document frequency.
    token_to_col: dict[tuple[int, bytes], int] = {}
    token_radius: list[int] = []
    token_hex: list[str] = []
    token_example: list[tuple[int, int]] = []
    row_maps: list[Counter[int]] = []
    n_atoms = []
    for row, graph_index in enumerate(train_graph_indices.tolist()):
        g, _ = dataset[int(graph_index)]
        node_feat = np.asarray(g["node_feat"], dtype=np.int64)
        edge_index = np.asarray(g["edge_index"], dtype=np.int64)
        edge_feat = np.asarray(g["edge_feat"], dtype=np.int64)
        signatures = rooted_signatures(node_feat, edge_index, edge_feat, 3)
        counts: Counter[int] = Counter()
        for radius, values in enumerate(signatures):
            for atom, value in enumerate(values):
                key = (radius, value)
                col = token_to_col.get(key)
                if col is None:
                    col = len(token_to_col)
                    token_to_col[key] = col
                    token_radius.append(radius)
                    token_hex.append(value.hex())
                    token_example.append((int(graph_index), int(atom)))
                counts[col] += 1
        row_maps.append(counts)
        n_atoms.append(int(node_feat.shape[0]))
        if (row + 1) % 5000 == 0:
            print(f"extracted {row + 1}/{len(train_graph_indices)} graphs vocab={len(token_to_col)}", flush=True)

    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    for counts in row_maps:
        for col, count in sorted(counts.items()):
            indices.append(col)
            data.append(count)
        indptr.append(len(indices))
    x_all = sparse.csr_matrix((np.asarray(data, dtype=np.int16), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)), shape=(len(train_graph_indices), len(token_to_col)))
    radius_arr = np.asarray(token_radius, dtype=np.int8)
    del row_maps

    out: dict[str, Any] = {
        "protocol_id": "molhiv-feature-exact-rooted-motif-stability-v1",
        "date": "2026-07-29",
        "scope": "exact official-train only; three outer scaffold folds with nested scaffold stability selection",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "isolation": "dataset items were accessed only for official-train graph indices",
        "config": vars(args),
        "n_train": int(len(train_graph_indices)), "n_train_positive": int(y.sum()),
        "n_unique_motifs": int(x_all.shape[1]),
        "unique_motifs_by_radius": {str(r): int(np.sum(radius_arr == r)) for r in range(4)},
        "n_atoms_total": int(np.sum(n_atoms)), "mean_atoms": float(np.mean(n_atoms)),
        "train_indices_sha256": sha(train_graph_indices),
        "feature_matrix_shape": list(x_all.shape), "feature_matrix_nnz": int(x_all.nnz),
        "feature_matrix_data_sha256": sha(x_all.data), "feature_matrix_indices_sha256": sha(x_all.indices), "feature_matrix_indptr_sha256": sha(x_all.indptr),
        "folds": [], "aggregate": {},
    }

    radius_sets = {"r0": (0,), "r0_1": (0, 1), "r0_2": (0, 1, 2), "r0_3": (0, 1, 2, 3)}
    topks = (512, 1024, 2048, 4096)
    all_results: dict[str, list[dict[str, Any]]] = {}
    for fold, (fit, held, fit_graph, held_graph) in enumerate(outer):
        print(f"outer fold {fold}: fit={len(fit)} held={len(held)}", flush=True)
        xfit_full = x_all[fit]
        xheld_full = x_all[held]
        df = np.asarray((xfit_full > 0).sum(axis=0)).ravel()
        eligible_all = np.flatnonzero(df >= args.min_df)
        fold_out: dict[str, Any] = {
            "fold": fold, "n_fit": int(len(fit)), "n_heldout": int(len(held)),
            "n_fit_positive": int(y[fit].sum()), "n_heldout_positive": int(y[held].sum()),
            "fit_graph_indices_sha256": sha(fit_graph), "heldout_graph_indices_sha256": sha(held_graph),
            "n_eligible_min_df": int(len(eligible_all)), "results": {},
        }
        # Radius ablation baselines with vocabulary fitted by outer-fit DF only.
        for rname, radii in radius_sets.items():
            cols = np.flatnonzero((df >= args.min_df) & np.isin(radius_arr, radii))
            for mode in ("binary", "logcount"):
                for c in (0.003, 0.01):
                    key = f"baseline_{rname}_{mode}_c{c:g}"
                    row, ph = fit_eval(transformed(xfit_full[:, cols], mode), y[fit], transformed(xheld_full[:, cols], mode), y[held], c)
                    row.update({"selection": "outer-fit min document frequency only", "radii": list(radii), "mode": mode, "C": c, "n_features": int(len(cols)), "heldout_graph_indices": held_graph.astype(int).tolist(), "heldout_y": y[held].astype(int).tolist(), "heldout_probabilities": ph.tolist(), "heldout_probability_sha256": sha(ph)})
                    fold_out["results"][key] = row
                    all_results.setdefault(key, []).append(row)

        # Nested selection is performed over all radii 0..3 after min-DF.
        xb_eligible = transformed(xfit_full[:, eligible_all], "binary")
        stable, direction, inner_meta = stable_scores(xb_eligible, y[fit], groups[fit], 20260729 + fold)
        global_z = np.abs(z_association(xb_eligible, y[fit]))
        frequency = df[eligible_all].astype(float)
        rng = np.random.default_rng(20260729 + fold)
        shuffled_y = y[fit].copy(); rng.shuffle(shuffled_y)
        shuffled_stable, _, shuffled_meta = stable_scores(xb_eligible, shuffled_y, groups[fit], 20260829 + fold)
        fold_out["nested_partitions"] = inner_meta
        fold_out["shuffled_nested_partitions"] = shuffled_meta
        selectors = {"stable": stable, "global_assoc": global_z}
        for selector, scores in selectors.items():
            order = np.lexsort((eligible_all, -scores))
            positive = order[scores[order] > 0]
            for k in topks:
                chosen_local = positive[: min(k, len(positive))]
                cols = eligible_all[chosen_local]
                for c in ((0.003, 0.01) if selector == "stable" else (0.01,)):
                    key = f"{selector}_top{k}_logcount_c{c:g}"
                    row, ph = fit_eval(transformed(xfit_full[:, cols], "logcount"), y[fit], transformed(xheld_full[:, cols], "logcount"), y[held], c)
                    row.update({"selection": selector, "mode": "logcount", "C": c, "requested_topk": k, "n_features": int(len(cols)), "selected_score_min": float(scores[chosen_local[-1]]) if len(chosen_local) else None, "selected_positive_direction": int(np.sum(direction[chosen_local] > 0)) if selector == "stable" else None, "heldout_graph_indices": held_graph.astype(int).tolist(), "heldout_y": y[held].astype(int).tolist(), "heldout_probabilities": ph.tolist(), "heldout_probability_sha256": sha(ph)})
                    fold_out["results"][key] = row
                    all_results.setdefault(key, []).append(row)
        for selector, scores in {"frequency": frequency, "shuffled_stable": shuffled_stable}.items():
            order = np.lexsort((eligible_all, -scores)); positive = order[scores[order] > 0]; chosen_local = positive[: min(2048, len(positive))]; cols = eligible_all[chosen_local]
            key = f"{selector}_top2048_logcount_c0.01"
            row, ph = fit_eval(transformed(xfit_full[:, cols], "logcount"), y[fit], transformed(xheld_full[:, cols], "logcount"), y[held], 0.01)
            row.update({"selection": selector, "mode": "logcount", "C": 0.01, "requested_topk": 2048, "n_features": int(len(cols)), "heldout_graph_indices": held_graph.astype(int).tolist(), "heldout_y": y[held].astype(int).tolist(), "heldout_probabilities": ph.tolist(), "heldout_probability_sha256": sha(ph)})
            fold_out["results"][key] = row
            all_results.setdefault(key, []).append(row)
        out["folds"].append(fold_out)

    out["aggregate"] = {key: summarize(rows) for key, rows in all_results.items() if len(rows) == 3}
    out["ranking"] = sorted(({"candidate": key, **value} for key, value in out["aggregate"].items()), key=lambda row: (row["mean_auc"], row["min_fold_auc"]), reverse=True)
    out["token_examples"] = [{"column": i, "radius": int(token_radius[i]), "signature": token_hex[i], "example_graph_index": token_example[i][0], "example_root_atom": token_example[i][1]} for i in range(min(200, len(token_radius)))]
    out["elapsed_sec"] = time.time() - t0
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": args.output, "vocab": out["unique_motifs_by_radius"], "top20": out["ranking"][:20], "elapsed_sec": out["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
