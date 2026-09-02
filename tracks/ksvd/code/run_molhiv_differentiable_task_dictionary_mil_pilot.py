"""Small MolHIV internal-fold pilot for a jointly task-aware patch dictionary.

Patch extraction follows the existing graph-level chemical KSVD substrate.  A
dictionary is initialized by unsupervised K-SVD on inner-train patches, then
updated together with a graph-level MIL head.  Each graph is represented by
the per-atom maximum absolute sparse activation over its patches.  This is a
deliberately simple downstream readout: it tests the dictionary objective
before adding GINE, overlap relations, or a Transformer.

Only a subset of official-train is used by default and the evaluation is an
internal scaffold holdout.  Official valid/test are never loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig
from .ksvd import _omp, ksvd
from .run_molhiv_label_aware_ksvd import _collect_patch_pool


def _group_pool(Y_pool: np.ndarray, records: list[object]) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = {}
    for column, record in enumerate(records):
        graph_idx = int(record.graph_idx)  # type: ignore[attr-defined]
        groups.setdefault(graph_idx, []).append(column)
    return {graph: np.asarray(cols, dtype=np.int64) for graph, cols in groups.items()}


def _omp_graph_features(
    Y_pool: np.ndarray,
    groups: dict[int, np.ndarray],
    graph_indices: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
) -> np.ndarray:
    rows = []
    for graph_idx in graph_indices:
        cols = groups[int(graph_idx)]
        codes = np.stack([_omp(dictionary, Y_pool[:, int(c)], sparsity) for c in cols], axis=0)
        rows.append(np.max(np.abs(codes), axis=0))
    return np.stack(rows, axis=0)


def _fit_logistic(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
    )
    model.fit(X_train, y_train)
    return float(roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]))


def _train_joint(
    graph_arrays: dict[int, np.ndarray],
    labels: dict[int, float],
    train_indices: np.ndarray,
    init_dictionary: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    reconstruction_weight: float,
    incoherence_weight: float,
    seed: int,
    sparse: bool = True,
) -> tuple[np.ndarray, float, dict[str, float]]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    max_patches = max(graph_arrays[int(i)].shape[1] for i in train_indices)
    dim = init_dictionary.shape[0]
    atom_count = init_dictionary.shape[1]
    values = np.zeros((len(train_indices), max_patches, dim), dtype=np.float32)
    valid = np.zeros((len(train_indices), max_patches), dtype=np.float32)
    target = np.zeros(len(train_indices), dtype=np.float32)
    for row, graph_idx in enumerate(train_indices):
        array = graph_arrays[int(graph_idx)].T.astype(np.float32)
        values[row, : array.shape[0]] = array
        valid[row, : array.shape[0]] = 1.0
        target[row] = labels[int(graph_idx)]
    x = torch.from_numpy(values)
    mask = torch.from_numpy(valid)
    y = torch.from_numpy(target)
    atom_param = nn.Parameter(torch.from_numpy(init_dictionary.T.astype(np.float32)).clone())
    head = nn.Linear(atom_count, 1)
    nn.init.normal_(head.weight, mean=0.0, std=0.01)
    nn.init.zeros_(head.bias)
    log_tau = nn.Parameter(torch.tensor(-3.0))
    optimizer = torch.optim.Adam([atom_param, *head.parameters(), log_tau], lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1.0 - y.mean()) / max(float(y.mean()), 1e-6)))
    order_rng = np.random.default_rng(seed + 1)
    last = (0.0, 0.0, 0.0)
    for _epoch in range(epochs):
        order = order_rng.permutation(len(train_indices))
        for start in range(0, len(order), batch_size):
            batch = torch.from_numpy(order[start : start + batch_size])
            xb, mb, yb = x[batch], mask[batch], y[batch]
            optimizer.zero_grad(set_to_none=True)
            atoms = torch.nn.functional.normalize(atom_param, dim=1)
            corr = xb @ atoms.T
            tau = torch.sigmoid(log_tau) * 0.5
            code = (
                torch.sign(corr) * torch.relu(torch.abs(corr) - tau)
                if sparse else corr
            )
            code = code * mb[:, :, None]
            graph_feature = torch.amax(torch.abs(code), dim=1)
            logits = head(graph_feature).squeeze(1)
            recon = code @ atoms
            recon_loss = ((recon - xb) ** 2 * mb[:, :, None]).sum() / mb.sum().clamp_min(1.0) / dim
            gram = atoms @ atoms.T
            offdiag = gram - torch.diag(torch.diag(gram))
            incoherence = torch.mean(offdiag ** 2)
            task_loss = bce(logits, yb)
            loss = task_loss + reconstruction_weight * recon_loss + incoherence_weight * incoherence
            loss.backward()
            torch.nn.utils.clip_grad_norm_([atom_param, *head.parameters(), log_tau], 5.0)
            optimizer.step()
            last = (float(task_loss.detach()), float(recon_loss.detach()), float(incoherence.detach()))
    with torch.no_grad():
        atoms = torch.nn.functional.normalize(atom_param, dim=1)
        return atoms.T.cpu().numpy().astype(np.float64), (float((torch.sigmoid(log_tau) * 0.5).cpu()) if sparse else 0.0), {
            "final_task_bce": last[0],
            "final_reconstruction_mse": last[1],
            "final_incoherence": last[2],
        }


def _soft_graph_features(
    graph_arrays: dict[int, np.ndarray],
    graph_indices: np.ndarray,
    dictionary: np.ndarray,
    tau: float,
) -> np.ndarray:
    rows = []
    for graph_idx in graph_indices:
        patches = graph_arrays[int(graph_idx)].T
        corr = patches @ dictionary
        code = np.sign(corr) * np.maximum(np.abs(corr) - tau, 0.0)
        rows.append(np.max(np.abs(code), axis=0))
    return np.stack(rows, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--inner-seed", type=int, default=20260824)
    ap.add_argument("--model-seed", type=int, default=None)
    ap.add_argument("--fold-cache", type=str, default="")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--inner-valid-fraction", type=float, default=0.2)
    ap.add_argument("--n-atoms", type=int, default=16)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--ksvd-iter", type=int, default=3)
    ap.add_argument("--max-train-patches", type=int, default=12000)
    ap.add_argument("--max-patches-per-graph", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reconstruction-weight", type=float, default=0.2)
    ap.add_argument("--incoherence-weight", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    model_seed = args.inner_seed if args.model_seed is None else args.model_seed
    bundle = load_molhiv(max_graphs=args.max_graphs, seed=args.data_seed, with_features=True)
    cfg = GraphLevelConfig(
        n_atoms=args.n_atoms, T=args.sparsity, T_min=1, ksvd_iter=args.ksvd_iter,
        seed=0, max_train_patches=args.max_train_patches,
        max_patches_per_graph=args.max_patches_per_graph,
        patch_feat="wl_chem_ring", normalize_patches=True,
    )
    Y_pool, records, pool_stats = _collect_patch_pool(bundle, cfg)
    groups = _group_pool(Y_pool, records)
    train_all = np.asarray(bundle.split["train"], dtype=np.int64)
    if args.fold_cache:
        if args.fold not in (0, 1, 2):
            raise ValueError("--fold must be 0, 1 or 2 when --fold-cache is used")
        with np.load(args.fold_cache, allow_pickle=False) as folds:
            cached_original = np.asarray(folds["original_indices"], dtype=np.int64)
            current_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
            if not np.array_equal(cached_original, current_original):
                raise ValueError("fold cache original_indices do not match loaded MolHIV subset")
            fit_original = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
            valid_original = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
            # The n8000 cache stores fold rows as local subset positions, even
            # though ``original_indices`` stores OGB global IDs.
            if max(int(fit_original.max()), int(valid_original.max())) < len(bundle.graphs):
                fit_graphs = fit_original
                valid_graphs = valid_original
            else:
                original_to_local = {int(value): index for index, value in enumerate(current_original)}
                fit_graphs = np.asarray([original_to_local[int(value)] for value in fit_original], dtype=np.int64)
                valid_graphs = np.asarray([original_to_local[int(value)] for value in valid_original], dtype=np.int64)
        split_protocol = "scaffold_fold_cache"
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_valid_fraction, random_state=args.inner_seed)
        fit_pos, valid_pos = next(splitter.split(train_all, bundle.y[train_all]))
        fit_graphs, valid_graphs = train_all[fit_pos], train_all[valid_pos]
        split_protocol = "stratified_random_inner_split"
    fit_cols = np.concatenate([groups[int(i)] for i in fit_graphs])
    if len(fit_cols) > args.max_train_patches:
        rng = np.random.default_rng(args.inner_seed)
        fit_cols = np.sort(rng.choice(fit_cols, size=args.max_train_patches, replace=False))
    Y_fit = Y_pool[:, fit_cols]
    dictionary, _, ksvd_info = ksvd(Y_fit, n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=model_seed)
    task_dictionary, tau, task_info = _train_joint(
        {int(i): Y_pool[:, groups[int(i)]] for i in fit_graphs},
        {int(i): float(bundle.y[int(i)] > 0.5) for i in fit_graphs},
        fit_graphs, dictionary,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        reconstruction_weight=args.reconstruction_weight,
        incoherence_weight=args.incoherence_weight, seed=model_seed,
    )
    dense_dictionary, dense_tau, dense_info = _train_joint(
        {int(i): Y_pool[:, groups[int(i)]] for i in fit_graphs},
        {int(i): float(bundle.y[int(i)] > 0.5) for i in fit_graphs},
        fit_graphs, dictionary,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        reconstruction_weight=args.reconstruction_weight,
        incoherence_weight=args.incoherence_weight, seed=model_seed + 700001,
        sparse=False,
    )
    graph_arrays = {int(i): Y_pool[:, groups[int(i)]] for i in np.concatenate([fit_graphs, valid_graphs])}
    labels = {int(i): float(bundle.y[int(i)] > 0.5) for i in np.concatenate([fit_graphs, valid_graphs])}
    ksvd_train = _omp_graph_features(Y_pool, groups, fit_graphs, dictionary, args.sparsity)
    ksvd_valid = _omp_graph_features(Y_pool, groups, valid_graphs, dictionary, args.sparsity)
    task_train = _soft_graph_features(graph_arrays, fit_graphs, task_dictionary, tau)
    task_valid = _soft_graph_features(graph_arrays, valid_graphs, task_dictionary, tau)
    dense_train = _soft_graph_features(graph_arrays, fit_graphs, dense_dictionary, dense_tau)
    dense_valid = _soft_graph_features(graph_arrays, valid_graphs, dense_dictionary, dense_tau)
    y_fit = np.asarray([labels[int(i)] for i in fit_graphs])
    y_valid = np.asarray([labels[int(i)] for i in valid_graphs])
    raw_train = np.stack([np.max(np.abs(Y_pool[:, groups[int(i)]]), axis=1) for i in fit_graphs])
    raw_valid = np.stack([np.max(np.abs(Y_pool[:, groups[int(i)]]), axis=1) for i in valid_graphs])
    results = {
        "raw_patch_max": _fit_logistic(raw_train, y_fit, raw_valid, y_valid),
        "ksvd_omp_maxabs": _fit_logistic(ksvd_train, y_fit, ksvd_valid, y_valid),
        "differentiable_task_dictionary_maxabs": _fit_logistic(task_train, y_fit, task_valid, y_valid),
        "dense_task_projection_maxabs": _fit_logistic(dense_train, y_fit, dense_valid, y_valid),
    }
    payload = {
        "protocol_id": "molhiv-differentiable-task-dictionary-mil-pilot-v1",
        "scope": "subset of official-train; one internal holdout; official-valid/test not loaded",
        "split_protocol": split_protocol,
        "config": vars(args),
        "pool_stats": pool_stats,
        "fit_graphs": int(len(fit_graphs)),
        "valid_graphs": int(len(valid_graphs)),
        "ksvd_fit": ksvd_info,
        "task_fit": task_info,
        "dense_control_fit": dense_info,
        "results": results,
        "deltas": {
            "task_minus_ksvd": results["differentiable_task_dictionary_maxabs"] - results["ksvd_omp_maxabs"],
            "task_minus_raw": results["differentiable_task_dictionary_maxabs"] - results["raw_patch_max"],
            "task_minus_dense_control": results["differentiable_task_dictionary_maxabs"] - results["dense_task_projection_maxabs"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
