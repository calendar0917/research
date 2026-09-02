"""Exact additive path evidence plus a compact low-rank path-pair residual.

The base model is the successful exact binary full-path logistic regression.
A factorization-machine (FM) residual then models which path words occur in the
same molecule.  Exact additive path coefficients are retained; only pairwise
interactions are compressed into deterministic hash buckets.  This is not a
GNN: no node messages are exchanged, and the graph is represented as a set of
canonical typed paths.

Outer held-out folds are never used for epoch selection.  For every outer fold,
a scaffold-separated split inside the outer-fit portion selects the number of
FM optimization steps.  The base model and FM are then refit on the complete
outer-fit portion for exactly that number of steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from code.run_molhiv_stable_exact_motif_probe import digest, feature_bytes
from code.run_molhiv_exact_motif_path_vocabulary_probe import (
    enumerate_simple_paths,
    path_signature,
    sparse_from_counters,
)
import torch
import torch.nn.functional as F


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def binary(x: sparse.csr_matrix) -> sparse.csr_matrix:
    z = x.astype(np.float32, copy=True)
    z.data.fill(1.0)
    return z


def hashed_binary(x: sparse.csr_matrix, col_buckets: np.ndarray, n_buckets: int, normalization: str) -> sparse.csr_matrix:
    coo = x.tocoo()
    z = sparse.csr_matrix(
        (np.ones(len(coo.data), dtype=np.float32), (coo.row, col_buckets[coo.col])),
        shape=(x.shape[0], n_buckets),
    )
    z.sum_duplicates()
    z.data.fill(1.0)
    if normalization == "sqrt":
        nnz = np.diff(z.indptr).astype(np.float32)
        z = sparse.diags(1.0 / np.sqrt(np.maximum(nnz, 1.0)), format="csr") @ z
    elif normalization != "raw":
        raise ValueError(normalization)
    return z.tocsr()


def torch_csr(x: sparse.csr_matrix) -> torch.Tensor:
    x = x.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(
        torch.from_numpy(x.indptr.astype(np.int64, copy=False)),
        torch.from_numpy(x.indices.astype(np.int64, copy=False)),
        torch.from_numpy(x.data.astype(np.float32, copy=False)),
        size=x.shape,
        dtype=torch.float32,
    )


def class_weights(y: np.ndarray) -> np.ndarray:
    out = np.empty(len(y), dtype=np.float32)
    for cls in (0, 1):
        mask = y == cls
        out[mask] = len(y) / (2.0 * max(int(mask.sum()), 1))
    return out


def sparse_values_squared(x: torch.Tensor) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        x.crow_indices(), x.col_indices(), x.values().square(), size=x.shape, dtype=x.dtype
    )


def fm_interaction(x: torch.Tensor, x_squared: torch.Tensor, factors: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    summed = torch.sparse.mm(x, factors)
    squared_sum = summed.square()
    sum_squared = torch.sparse.mm(x_squared, factors.square())
    return scale * 0.5 * (squared_sum - sum_squared).sum(dim=1)


def train_fm_residual(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    base_train: np.ndarray,
    x_eval: sparse.csr_matrix,
    y_eval: np.ndarray,
    base_eval: np.ndarray,
    rank: int,
    seed: int,
    max_steps: int,
    eval_every: int,
    selected_steps: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train FM residual; select steps on eval iff selected_steps is None."""
    torch.manual_seed(seed)
    xt = torch_csr(x_train)
    xe = torch_csr(x_eval)
    xt_squared = sparse_values_squared(xt)
    xe_squared = sparse_values_squared(xe)
    yt = torch.from_numpy(y_train.astype(np.float32))
    bt = torch.from_numpy(base_train.astype(np.float32))
    be = torch.from_numpy(base_eval.astype(np.float32))
    weights = torch.from_numpy(class_weights(y_train))
    factors = torch.nn.Parameter(torch.empty(x_train.shape[1], rank, dtype=torch.float32))
    torch.nn.init.normal_(factors, mean=0.0, std=0.02)
    scale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [
            {"params": [factors], "lr": 0.03, "weight_decay": 2e-3},
            {"params": [scale], "lr": 0.03, "weight_decay": 0.0},
        ]
    )
    target_steps = max_steps if selected_steps is None else int(selected_steps)
    history: list[dict[str, float | int]] = []
    best_auc = float(roc_auc_score(y_eval, base_eval))
    best_step = 0
    best_pred = 1.0 / (1.0 + np.exp(-base_eval))
    # Step 0 is exactly the additive path baseline because scale=0.
    history.append({"step": 0, "eval_auc": best_auc, "scale": 0.0})
    for step in range(1, target_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = bt + fm_interaction(xt, xt_squared, factors, scale)
        loss = (F.binary_cross_entropy_with_logits(logits, yt, reduction="none") * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([factors, scale], max_norm=5.0)
        optimizer.step()
        should_eval = step % eval_every == 0 or step == target_steps
        if should_eval:
            with torch.no_grad():
                eval_logits = be + fm_interaction(xe, xe_squared, factors, scale)
                pred = torch.sigmoid(eval_logits).cpu().numpy()
            auc = float(roc_auc_score(y_eval, pred))
            history.append({"step": step, "eval_auc": auc, "train_loss": float(loss.detach()), "scale": float(scale.detach())})
            if selected_steps is None and auc > best_auc + 1e-12:
                best_auc, best_step, best_pred = auc, step, pred.copy()
            elif selected_steps is not None and step == target_steps:
                best_auc, best_step, best_pred = auc, step, pred.copy()
    return best_pred, {
        "selected_steps": int(best_step),
        "eval_auc": float(best_auc),
        "history": history,
        "final_factor_l2": float(torch.linalg.vector_norm(factors).detach()),
        "final_scale": float(scale.detach()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a = np.asarray([r["heldout_auc"] for r in rows], dtype=np.float64)
    return {
        "fold_auc": a.tolist(),
        "mean_auc": float(a.mean()),
        "sample_std_auc": float(a.std(ddof=1)),
        "min_fold_auc": float(a.min()),
        "mean_total_parameters": float(np.mean([r["total_parameters"] for r in rows])),
        "selected_steps": [int(r.get("selected_steps", 0)) for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--buckets", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/path_pair_interaction_fulltrain_probe_20260729.json")
    args = ap.parse_args()
    t0 = time.time()
    repo = Path(__file__).resolve().parents[3]
    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset

    ds = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    split = ds.get_idx_split()
    train = np.asarray(split["train"], dtype=np.int64)
    forbidden = np.concatenate([np.asarray(split["valid"], dtype=np.int64), np.asarray(split["test"], dtype=np.int64)])
    y = np.asarray(ds.labels).reshape(-1).astype(np.int64)[train]
    pos = np.full(len(ds), -1, dtype=np.int64)
    pos[train] = np.arange(len(train))
    with np.load(args.fold_cache, allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["official_train_indices"], dtype=np.int64), train):
            raise ValueError("official train mismatch")
        groups = np.asarray(z["train_scaffold_groups"]).astype(str)
        outer = []
        for fold in range(3):
            fit_global = np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)
            held_global = np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)
            outer.append((pos[fit_global], pos[held_global]))
    if np.any(pos[forbidden] >= 0):
        raise AssertionError("forbidden split contamination")

    vocab: dict[tuple[int, bytes], int] = {}
    buckets_all: list[int] = []
    rows: list[Counter[int]] = []
    for row_i, graph_i in enumerate(train.tolist()):
        graph, _ = ds[int(graph_i)]
        nf = np.asarray(graph["node_feat"], dtype=np.int64)
        ei = np.asarray(graph["edge_index"], dtype=np.int64)
        ef = np.asarray(graph["edge_feat"], dtype=np.int64)
        paths, edge_pos = enumerate_simple_paths(ei, len(nf), 4)
        atom_labels = [digest(b"A" + feature_bytes(nf[u])) for u in range(len(nf))]
        edge_labels = {key: digest(b"B" + feature_bytes(ef[j])) for key, j in edge_pos.items()}
        counts: Counter[int] = Counter()
        for length in range(1, 5):
            for nodes in paths[length]:
                sig = path_signature(nodes, atom_labels, edge_labels)
                key = (length, sig)
                col = vocab.get(key)
                if col is None:
                    col = len(vocab)
                    vocab[key] = col
                    # Include path length in the deterministic bucket mapping.
                    h = hashlib.blake2b(bytes([length]) + sig, digest_size=8, person=b"path-fm").digest()
                    buckets_all.append(int.from_bytes(h, "little") % args.buckets)
                counts[col] += 1
        rows.append(counts)
        if (row_i + 1) % 5000 == 0:
            print(f"extracted {row_i + 1}/{len(train)} vocab={len(vocab)}", flush=True)
    x_all = sparse_from_counters(rows, len(vocab))
    buckets_all_np = np.asarray(buckets_all, dtype=np.int32)
    del rows, vocab, buckets_all

    configs = (
        {"name": "fm_pair_raw_b2048_r4", "normalization": "raw"},
        {"name": "fm_pair_sqrtnorm_b2048_r4", "normalization": "sqrt"},
    )
    if args.buckets != 2048 or args.rank != 4:
        configs = tuple({**c, "name": c["name"].replace("b2048", f"b{args.buckets}").replace("r4", f"r{args.rank}")} for c in configs)

    out: dict[str, Any] = {
        "protocol_id": "molhiv-exact-path-plus-lowrank-pair-residual-v1",
        "date": "2026-07-29",
        "scope": "official-train only; outer scaffold evaluation; inner scaffold epoch selection",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "isolation": "dataset graph items accessed only for official train",
        "config": vars(args),
        "base_model": "exact full path length1-4 binary logistic C=0.03",
        "interaction_model": "hashed factorization-machine residual; additive exact path score retained",
        "raw_vocab": int(x_all.shape[1]),
        "candidates": [dict(c) for c in configs],
        "folds": [],
        "aggregate": {},
    }
    aggregate_rows: dict[str, list[dict[str, Any]]] = {c["name"]: [] for c in configs}
    baseline_rows: list[dict[str, Any]] = []

    for fold, (fit, held) in enumerate(outer):
        df_outer = np.asarray((x_all[fit] > 0).sum(axis=0), dtype=np.int64).ravel()
        cols_outer = np.flatnonzero(df_outer >= args.min_df)
        xof = binary(x_all[fit][:, cols_outer])
        xoh = binary(x_all[held][:, cols_outer])
        base_outer = LogisticRegression(C=0.03, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0)
        base_outer.fit(xof, y[fit])
        base_fit_logits = base_outer.decision_function(xof).astype(np.float32)
        base_held_logits = base_outer.decision_function(xoh).astype(np.float32)
        base_held_prob = base_outer.predict_proba(xoh)[:, 1]
        base_auc = float(roc_auc_score(y[held], base_held_prob))
        baseline_rows.append({"heldout_auc": base_auc, "total_parameters": int(len(cols_outer) + 1)})

        # One deterministic scaffold-separated inner split; only this split can
        # choose the optimization length.  The outer held fold stays untouched.
        inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260729 + fold)
        inner_train_pos, inner_val_pos = next(inner_splitter.split(np.zeros(len(fit)), y[fit], groups[fit]))
        inner_train = fit[np.asarray(inner_train_pos, dtype=np.int64)]
        inner_val = fit[np.asarray(inner_val_pos, dtype=np.int64)]
        df_inner = np.asarray((x_all[inner_train] > 0).sum(axis=0), dtype=np.int64).ravel()
        cols_inner = np.flatnonzero(df_inner >= args.min_df)
        xit = binary(x_all[inner_train][:, cols_inner])
        xiv = binary(x_all[inner_val][:, cols_inner])
        base_inner = LogisticRegression(C=0.03, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0)
        base_inner.fit(xit, y[inner_train])
        inner_train_logits = base_inner.decision_function(xit).astype(np.float32)
        inner_val_logits = base_inner.decision_function(xiv).astype(np.float32)
        inner_base_auc = float(roc_auc_score(y[inner_val], base_inner.predict_proba(xiv)[:, 1]))

        fold_out: dict[str, Any] = {
            "fold": fold,
            "n_outer_fit": int(len(fit)), "n_outer_held": int(len(held)),
            "n_inner_train": int(len(inner_train)), "n_inner_valid": int(len(inner_val)),
            "n_outer_features": int(len(cols_outer)), "n_inner_features": int(len(cols_inner)),
            "outer_baseline_auc": base_auc, "inner_baseline_auc": inner_base_auc,
            "results": {},
        }
        print(f"outer fold {fold}: baseline={base_auc:.6f} inner_base={inner_base_auc:.6f}", flush=True)

        for ci, cfg in enumerate(configs):
            norm = str(cfg["normalization"])
            hit = hashed_binary(xit, buckets_all_np[cols_inner], args.buckets, norm)
            hiv = hashed_binary(xiv, buckets_all_np[cols_inner], args.buckets, norm)
            _, selection = train_fm_residual(
                hit, y[inner_train], inner_train_logits,
                hiv, y[inner_val], inner_val_logits,
                rank=args.rank, seed=20260729 + 100 * fold + ci,
                max_steps=args.max_steps, eval_every=args.eval_every,
                selected_steps=None,
            )
            selected_steps = int(selection["selected_steps"])
            hof = hashed_binary(xof, buckets_all_np[cols_outer], args.buckets, norm)
            hoh = hashed_binary(xoh, buckets_all_np[cols_outer], args.buckets, norm)
            pred, refit = train_fm_residual(
                hof, y[fit], base_fit_logits,
                hoh, y[held], base_held_logits,
                rank=args.rank, seed=20260729 + 100 * fold + ci,
                max_steps=args.max_steps, eval_every=max(args.eval_every, selected_steps if selected_steps else 1),
                selected_steps=selected_steps,
            )
            auc = float(roc_auc_score(y[held], pred))
            total_parameters = int(len(cols_outer) + 1 + args.buckets * args.rank + 1)
            rr = {
                "heldout_auc": auc,
                "baseline_auc": base_auc,
                "delta_auc": auc - base_auc,
                "selected_steps": selected_steps,
                "inner_selected_auc": float(selection["eval_auc"]),
                "inner_baseline_auc": inner_base_auc,
                "selection": selection,
                "refit": refit,
                "normalization": norm,
                "buckets": args.buckets,
                "rank": args.rank,
                "additive_path_parameters": int(len(cols_outer) + 1),
                "interaction_parameters": int(args.buckets * args.rank + 1),
                "total_parameters": total_parameters,
                "heldout_probabilities": pred.tolist(),
                "heldout_probability_sha256": sha(pred),
            }
            fold_out["results"][cfg["name"]] = rr
            aggregate_rows[cfg["name"]].append(rr)
            print(f"  {cfg['name']}: steps={selected_steps} inner={selection['eval_auc']:.6f} outer={auc:.6f} delta={auc-base_auc:+.6f}", flush=True)
        out["folds"].append(fold_out)

    out["baseline"] = summarize(baseline_rows)
    out["aggregate"] = {k: summarize(v) for k, v in aggregate_rows.items()}
    out["ranking"] = sorted(({"candidate": k, **v} for k, v in out["aggregate"].items()), key=lambda r: (r["mean_auc"], r["min_fold_auc"]), reverse=True)
    out["elapsed_sec"] = time.time() - t0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": str(output), "baseline": out["baseline"], "ranking": out["ranking"], "elapsed_sec": out["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
