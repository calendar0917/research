"""Strict scaffold-fold pilot for prototype-constrained task-aware MIL.

The central hypothesis is that graph-label MIL benefits from atoms that remain
actual observed local-context prototypes, rather than freely moving synthetic
KSVD reconstruction directions.  A fold-fit candidate bank is made only from
real masked-context node latents.  Graph-level labels rank candidates through a
linear occurrence model; a diversity-constrained selector chooses 32 frozen
real prototypes.  Matched random, farthest-point, label-shuffled, cached-random,
and KSVD controls use the same downstream prototype-occurrence MIL classifier.

Only official-train scaffold folds are accepted.  The held-out scaffold fold is
evaluated once after a fixed epoch budget; official-valid/test are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _graph_candidate_features(
    latents: np.ndarray,
    offsets: np.ndarray,
    graph_indices: np.ndarray,
    candidates: np.ndarray,
    top_nodes: int,
) -> np.ndarray:
    """Per candidate: max positive cosine, top-node mean, all-node mean."""
    out = np.empty((len(graph_indices), 3 * len(candidates)), dtype=np.float32)
    candidates_t = candidates.T
    for out_i, raw_i in enumerate(graph_indices):
        i = int(raw_i)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        sim = np.maximum(latents[lo:hi] @ candidates_t, 0.0)
        max_sim = sim.max(axis=0)
        k = min(top_nodes, sim.shape[0])
        if k == sim.shape[0]:
            top_mean = sim.mean(axis=0)
        else:
            top_mean = np.partition(sim, sim.shape[0] - k, axis=0)[-k:].mean(axis=0)
        mean_sim = sim.mean(axis=0)
        out[out_i] = np.concatenate([max_sim, top_mean, mean_sim])
    return out


def _fit_candidate_importance(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    c_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
        max_iter=3000,
    )
    model.fit(x, labels.astype(np.int64))
    scores = model.decision_function(x)
    n_candidates = features.shape[1] // 3
    coef = model.coef_.reshape(-1)
    grouped = np.stack([
        coef[:n_candidates],
        coef[n_candidates:2 * n_candidates],
        coef[2 * n_candidates:],
    ], axis=1)
    importance = np.linalg.norm(grouped, axis=1)
    return importance.astype(np.float32), {
        "fit_auc": float(roc_auc_score(labels, scores)),
        "n_iter": int(model.n_iter_[0]),
        "coefficient_l2": float(np.linalg.norm(coef)),
        "nonzero_coefficients": int(np.count_nonzero(np.abs(coef) > 1e-10)),
        "score_sha256": _sha256(scores.astype(np.float64)),
    }


def _select_mmr(
    candidates: np.ndarray,
    importance: np.ndarray,
    n_select: int,
    diversity_weight: float,
) -> np.ndarray:
    lo = float(importance.min())
    hi = float(importance.max())
    relevance = (importance - lo) / max(hi - lo, 1e-12)
    selected: list[int] = []
    available = np.ones(len(candidates), dtype=bool)
    for _ in range(n_select):
        if not selected:
            choice = int(np.argmax(relevance))
        else:
            max_similarity = (candidates @ candidates[np.asarray(selected)].T).max(axis=1)
            diversity = 1.0 - np.clip(max_similarity, -1.0, 1.0)
            score = (1.0 - diversity_weight) * relevance + diversity_weight * diversity
            score[~available] = -np.inf
            choice = int(np.argmax(score))
        selected.append(choice)
        available[choice] = False
    return np.asarray(selected, dtype=np.int64)


def _select_farthest(candidates: np.ndarray, n_select: int) -> np.ndarray:
    centroid = _normalize_rows(candidates.mean(axis=0, keepdims=True))[0]
    selected = [int(np.argmax(candidates @ centroid))]
    min_distance = 1.0 - candidates @ candidates[selected[0]]
    for _ in range(1, n_select):
        min_distance[np.asarray(selected)] = -np.inf
        choice = int(np.argmax(min_distance))
        selected.append(choice)
        min_distance = np.minimum(min_distance, 1.0 - candidates @ candidates[choice])
    return np.asarray(selected, dtype=np.int64)


def _prototype_diagnostics(prototypes: np.ndarray) -> dict[str, float]:
    gram = prototypes @ prototypes.T
    mask = ~np.eye(len(prototypes), dtype=bool)
    offdiag = gram[mask]
    return {
        "mean_signed_coherence": float(offdiag.mean()),
        "mean_abs_coherence": float(np.abs(offdiag).mean()),
        "max_abs_coherence": float(np.abs(offdiag).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument(
        "--controls",
        default="random,farthest,taskaware,shuffled_taskaware,cached_random,ksvd",
    )
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate-bank-size", type=int, default=256)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--selector-top-nodes", type=int, default=3)
    ap.add_argument("--selector-c", type=float, default=0.1)
    ap.add_argument("--diversity-weight", type=float, default=0.35)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--task-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    known_controls = {
        "random", "farthest", "taskaware", "shuffled_taskaware",
        "cached_random", "ksvd",
    }
    controls = [x.strip() for x in args.controls.split(",") if x.strip()]
    unknown = sorted(set(controls).difference(known_controls))
    if not controls or unknown or len(set(controls)) != len(controls):
        raise ValueError(f"invalid controls={controls}, unknown={unknown}")
    if args.fold < 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("invalid fold/training configuration")
    if not (0.0 <= args.diversity_weight <= 1.0):
        raise ValueError("--diversity-weight must be in [0,1]")
    if min(args.candidate_bank_size, args.n_prototypes, args.sparsity) <= 0:
        raise ValueError("bank/prototype/sparsity sizes must be positive")
    if args.n_prototypes > args.candidate_bank_size:
        raise ValueError("n-prototypes exceeds candidate-bank-size")
    if args.sparsity > args.n_prototypes:
        raise ValueError("sparsity exceeds n-prototypes")
    if args.temperature <= 0 or args.selector_c <= 0:
        raise ValueError("temperature and selector C must be positive")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.utils.features import get_atom_feature_dims
        from torch.utils.data import DataLoader, Dataset
        from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
        from torch_geometric.utils import softmax as graph_softmax
    except ImportError as exc:
        raise RuntimeError(f"missing training dependencies: {exc}") from exc

    t0 = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None:
        raise AssertionError("center atom features were not loaded")
    labels = np.asarray(bundle.y, dtype=np.float32)
    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        if "original_indices" in folds.files and not np.array_equal(
            np.asarray(folds["original_indices"], dtype=np.int64), expected_original
        ):
            raise ValueError("fold cache original_indices do not match dataset")
    official_train_set = set(official_train.tolist())
    if not set(fit_indices.tolist()).issubset(official_train_set):
        raise AssertionError("fit contains non-official-train graphs")
    if not set(heldout_indices.tolist()).issubset(official_train_set):
        raise AssertionError("heldout contains non-official-train graphs")
    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("fit and heldout overlap")
    if set(np.concatenate([fit_indices, heldout_indices]).tolist()) != official_train_set:
        raise AssertionError("outer scaffold fold does not partition official train")

    with np.load(args.latent_cache, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        latent_original = np.asarray(source["original_indices"], dtype=np.int64)
        latent_train = np.asarray(source["train_indices"], dtype=np.int64)
        latent_valid = np.asarray(source["valid_indices"], dtype=np.int64)
        latent_test = np.asarray(source["test_indices"], dtype=np.int64)
        latents_np = np.asarray(source["latents"], dtype=np.float32)
    with np.load(args.token_cache, allow_pickle=False) as source:
        token_arrays = {key: np.asarray(source[key]) for key in source.files}

    checks = [
        (latent_original, expected_original, "latent original_indices"),
        (latent_train, official_train, "latent train_indices"),
        (latent_valid, official_valid, "latent valid_indices"),
        (latent_test, official_test, "latent test_indices"),
        (np.asarray(token_arrays["offsets"], dtype=np.int64), offsets, "token offsets"),
        (np.asarray(token_arrays["original_indices"], dtype=np.int64), expected_original, "token original_indices"),
    ]
    for actual, expected, name in checks:
        if not np.array_equal(actual, expected):
            raise ValueError(f"misaligned {name}")
    if latents_np.shape[0] != int(offsets[-1]):
        raise ValueError("latent row count and offsets disagree")
    official_nontrain_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in np.concatenate([official_valid, official_test])
    ])
    if np.any(latents_np[official_nontrain_rows] != 0):
        raise AssertionError("official-valid/test latent rows are nonzero")

    fit_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in fit_indices
    ])
    heldout_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in heldout_indices
    ])
    if np.mean(np.linalg.norm(latents_np[np.concatenate([fit_rows, heldout_rows])], axis=1) > 0.99) < 0.999:
        raise ValueError("fit/heldout scaffold latents are missing or unnormalized")

    rng = np.random.default_rng(args.seed + 910_003 * (args.fold + 1))
    candidate_rows = rng.choice(
        fit_rows, size=args.candidate_bank_size, replace=False
    ).astype(np.int64)
    candidate_bank = _normalize_rows(latents_np[candidate_rows])
    candidate_features = _graph_candidate_features(
        latents_np, offsets, fit_indices, candidate_bank, args.selector_top_nodes
    )
    task_importance, task_selector = _fit_candidate_importance(
        candidate_features, labels[fit_indices], args.seed, args.selector_c
    )
    shuffled_labels = labels[fit_indices].copy()
    shuffled_rng = np.random.default_rng(args.seed + 71_017)
    shuffled_rng.shuffle(shuffled_labels)
    shuffled_importance, shuffled_selector = _fit_candidate_importance(
        candidate_features, shuffled_labels, args.seed + 1, args.selector_c
    )

    selection_indices: dict[str, np.ndarray] = {}
    selection_indices["random"] = rng.choice(
        args.candidate_bank_size, size=args.n_prototypes, replace=False
    ).astype(np.int64)
    selection_indices["farthest"] = _select_farthest(candidate_bank, args.n_prototypes)
    selection_indices["taskaware"] = _select_mmr(
        candidate_bank, task_importance, args.n_prototypes, args.diversity_weight
    )
    selection_indices["shuffled_taskaware"] = _select_mmr(
        candidate_bank, shuffled_importance, args.n_prototypes, args.diversity_weight
    )

    prototype_sets: dict[str, np.ndarray] = {
        name: candidate_bank[idx] for name, idx in selection_indices.items()
    }
    cached_random = np.asarray(token_arrays["dictionary_random_patch"], dtype=np.float32).T
    cached_ksvd = np.asarray(token_arrays["dictionary_ksvd"], dtype=np.float32).T
    if cached_random.shape != (args.n_prototypes, latents_np.shape[1]):
        raise ValueError(f"cached random shape mismatch: {cached_random.shape}")
    if cached_ksvd.shape != cached_random.shape:
        raise ValueError(f"cached KSVD shape mismatch: {cached_ksvd.shape}")
    prototype_sets["cached_random"] = _normalize_rows(cached_random)
    prototype_sets["ksvd"] = _normalize_rows(cached_ksvd)

    atom_dims = [int(x) for x in get_atom_feature_dims()]
    node_features = bundle.node_feats

    class GraphIndexDataset(Dataset):
        def __init__(self, indices: np.ndarray):
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self) -> int:
            return int(len(self.indices))

        def __getitem__(self, item: int) -> int:
            return int(self.indices[item])

    def collate_graphs(graph_indices: list[int]) -> dict[str, Any]:
        rows: list[np.ndarray] = []
        center_fields: list[np.ndarray] = []
        graph_ids: list[np.ndarray] = []
        for batch_i, raw_i in enumerate(graph_indices):
            i = int(raw_i)
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            n = hi - lo
            rows.append(np.arange(lo, hi, dtype=np.int64))
            center = np.asarray(node_features[i], dtype=np.int64)
            if center.shape != (n, len(atom_dims)):
                raise ValueError(f"center feature mismatch for graph {i}: {center.shape}")
            center_fields.append(center)
            graph_ids.append(np.full(n, batch_i, dtype=np.int64))
        return {
            "indices": torch.tensor(graph_indices, dtype=torch.long),
            "rows": torch.from_numpy(np.concatenate(rows)),
            "center": torch.from_numpy(np.concatenate(center_fields, axis=0)),
            "node_graph": torch.from_numpy(np.concatenate(graph_ids)),
            "labels": torch.from_numpy(labels[np.asarray(graph_indices, dtype=np.int64)]),
        }

    latents = torch.from_numpy(latents_np)

    class PrototypeMIL(nn.Module):
        def __init__(self, prototypes: np.ndarray):
            super().__init__()
            p = torch.tensor(prototypes, dtype=torch.float32)
            p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("prototypes", p)
            self.center_embeddings = nn.ModuleList([
                nn.Embedding(dim, args.hidden) for dim in atom_dims
            ])
            self.identity_embeddings = nn.Parameter(
                torch.empty(args.n_prototypes, args.hidden)
            )
            self.affinity_embeddings = nn.Parameter(
                torch.empty(args.n_prototypes, args.hidden)
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(3 * args.hidden + 2, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, args.hidden),
                nn.SiLU(),
            )
            self.attention = nn.Sequential(
                nn.Linear(args.hidden, args.hidden // 2),
                nn.Tanh(),
                nn.Linear(args.hidden // 2, 1),
            )
            self.graph_head = nn.Sequential(
                nn.Linear(3 * args.hidden, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, 1),
            )
            self.reset_parameters()

        def reset_parameters(self) -> None:
            for emb in self.center_embeddings:
                nn.init.xavier_uniform_(emb.weight)
            nn.init.xavier_uniform_(self.identity_embeddings)
            nn.init.xavier_uniform_(self.affinity_embeddings)
            modules = list(self.node_mlp.modules()) + list(self.attention.modules()) + list(self.graph_head.modules())
            for module in modules:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def occurrence_codes(self, z):
            cosine = z @ self.prototypes.T
            positive = torch.relu(cosine)
            if args.sparsity < positive.shape[1]:
                values, indices = positive.topk(args.sparsity, dim=1)
                codes = torch.zeros_like(positive).scatter_(1, indices, values)
            else:
                codes = positive
            return cosine, codes

        def forward(self, z, center, node_graph):
            cosine, codes = self.occurrence_codes(z)
            mass = codes.sum(dim=1, keepdim=True)
            normalized = codes / mass.clamp_min(1e-6)
            sharpened = torch.softmax(codes / args.temperature, dim=1)
            sharpened = sharpened * (codes > 0).float()
            sharpened = sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(1e-6)
            center_h = 0.0
            for field, emb in enumerate(self.center_embeddings):
                center_h = center_h + emb(center[:, field])
            identity_h = normalized @ self.identity_embeddings
            affinity_h = sharpened @ self.affinity_embeddings
            top_similarity = cosine.max(dim=1, keepdim=True).values
            node_h = self.node_mlp(torch.cat([
                center_h,
                identity_h,
                affinity_h,
                torch.log1p(mass),
                top_similarity,
            ], dim=1))
            attention = graph_softmax(self.attention(node_h).view(-1), node_graph)
            attention_pool = global_add_pool(node_h * attention[:, None], node_graph)
            mean_pool = global_mean_pool(node_h, node_graph)
            max_pool = global_max_pool(node_h, node_graph)
            logits = self.graph_head(torch.cat([
                attention_pool, mean_pool, max_pool
            ], dim=1)).view(-1)
            reconstruction = normalized @ self.prototypes
            return logits, codes, cosine, reconstruction

    device = torch.device(args.device)

    def make_loader(indices: np.ndarray, shuffle: bool, seed_offset: int):
        generator = torch.Generator().manual_seed(args.seed + seed_offset)
        return DataLoader(
            GraphIndexDataset(indices),
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=args.num_workers,
            collate_fn=collate_graphs,
        )

    def move_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(device) for key, value in batch.items()}

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        rec_sum = 0.0
        node_count = 0
        support_sum = 0.0
        mass_sum = 0.0
        best_similarity: list[np.ndarray] = []
        for batch in loader:
            batch = move_batch(batch)
            z = latents[batch["rows"].cpu()].to(device)
            logits, codes, cosine, reconstruction = model(
                z, batch["center"], batch["node_graph"]
            )
            all_scores.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            rec_sum += float((z - reconstruction).square().sum())
            node_count += int(z.shape[0])
            support_sum += float((codes > 1e-8).sum())
            mass_sum += float(codes.sum())
            best_similarity.append(cosine.max(dim=1).values.cpu().numpy())
        scores = np.concatenate(all_scores)
        y = np.concatenate(all_labels)
        best = np.concatenate(best_similarity)
        return {
            "auc": float(roc_auc_score(y, scores)),
            "n_graphs": int(len(y)),
            "n_positive": int(y.sum()),
            "reconstruction_mse_per_node": rec_sum / max(node_count, 1),
            "mean_active_prototypes": support_sum / max(node_count, 1),
            "mean_occurrence_mass": mass_sum / max(node_count, 1),
            "mean_best_signed_cosine": float(best.mean()),
            "q10_best_signed_cosine": float(np.quantile(best, 0.1)),
            "score_sha256": _sha256(scores.astype(np.float64)),
        }

    results: dict[str, Any] = {}
    for control in controls:
        control_start = time.time()
        _seed_everything(args.seed, torch)
        prototypes = prototype_sets[control]
        model = PrototypeMIL(prototypes).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.task_lr, weight_decay=args.weight_decay
        )
        train_loader = make_loader(fit_indices, True, 1000)
        history: list[dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            seen = 0
            for batch in train_loader:
                batch = move_batch(batch)
                z = latents[batch["rows"].cpu()].to(device)
                logits, _, _, _ = model(z, batch["center"], batch["node_graph"])
                loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                n = int(batch["labels"].shape[0])
                total += float(loss.detach()) * n
                seen += n
            row = {"epoch": float(epoch), "task": total / max(seen, 1)}
            history.append(row)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"{control} epoch={epoch:02d} task={row['task']:.5f}", flush=True)

        fit_metrics = evaluate(model, make_loader(fit_indices, False, 2000))
        heldout_metrics = evaluate(model, make_loader(heldout_indices, False, 3000))
        selection: dict[str, Any] = {
            **_prototype_diagnostics(prototypes),
            "prototype_sha256": _sha256(prototypes),
            "all_prototypes_are_real_fit_rows": bool(control not in {"ksvd"}),
        }
        if control in selection_indices:
            chosen = selection_indices[control]
            selection.update({
                "candidate_indices": chosen.tolist(),
                "source_node_rows": candidate_rows[chosen].tolist(),
                "mean_task_importance": float(task_importance[chosen].mean()),
                "mean_shuffled_importance": float(shuffled_importance[chosen].mean()),
            })
        results[control] = {
            "fit": fit_metrics,
            "heldout": heldout_metrics,
            "selection": selection,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "history": history,
            "elapsed_sec": time.time() - control_start,
        }
        print(
            f"{control} heldout_auc={heldout_metrics['auc']:.4f} "
            f"coverage={heldout_metrics['mean_best_signed_cosine']:.4f} "
            f"q10={heldout_metrics['q10_best_signed_cosine']:.4f}",
            flush=True,
        )

    comparisons: dict[str, float | bool] = {}
    if "taskaware" in results and "random" in results:
        delta = results["taskaware"]["heldout"]["auc"] - results["random"]["heldout"]["auc"]
        comparisons["taskaware_minus_random_auc"] = float(delta)
        comparisons["taskaware_beats_random_by_005"] = bool(delta >= 0.005)
    if "taskaware" in results and "farthest" in results:
        comparisons["taskaware_minus_farthest_auc"] = float(
            results["taskaware"]["heldout"]["auc"] - results["farthest"]["heldout"]["auc"]
        )
    if "taskaware" in results and "shuffled_taskaware" in results:
        comparisons["taskaware_minus_shuffled_auc"] = float(
            results["taskaware"]["heldout"]["auc"] - results["shuffled_taskaware"]["heldout"]["auc"]
        )
    if "taskaware" in results and "cached_random" in results:
        comparisons["taskaware_minus_cached_random_auc"] = float(
            results["taskaware"]["heldout"]["auc"] - results["cached_random"]["heldout"]["auc"]
        )
    if "taskaware" in results and "ksvd" in results:
        comparisons["taskaware_minus_ksvd_auc"] = float(
            results["taskaware"]["heldout"]["auc"] - results["ksvd"]["heldout"]["auc"]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": "molhiv-prototype-constrained-task-aware-mil-scaffold-v1",
        "date": "2026-07-28",
        "hypothesis": (
            "graph-label MIL benefits more from task-selected real patch prototypes "
            "than from synthetic reconstruction directions or unselected real patches"
        ),
        "architecture": {
            "candidate_bank": "random fold-fit real masked-context node latents",
            "selector": "graph-level candidate occurrence logistic model plus MMR diversity",
            "prototype_constraint": "selected task-aware atoms remain exact observed fit-node latents",
            "graph_model": "positive top-k prototype occurrence embeddings plus attention/mean/max MIL",
            "supervised_message_passing_layers": 0,
            "upstream_context_encoder": "frozen label-free two-layer masked-context GINE",
            "graph_labels_assigned_to_individual_nodes": False,
        },
        "selection_policy": {
            "data": "official-train only",
            "outer_validation": "one held-out Bemis-Murcko scaffold fold",
            "epoch_policy": f"single evaluation after fixed epoch {args.epochs}",
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "seed0_promotion_rule_across_3_folds": {
                "taskaware_mean_gain_over_random_at_least": 0.005,
                "minimum_taskaware_random_fold_wins": 2,
                "taskaware_must_beat_label_shuffled_mean": True,
                "taskaware_must_beat_cached_random_mean": True,
            },
        },
        "config": vars(args),
        "fold": int(args.fold),
        "n_fit": int(len(fit_indices)),
        "n_fit_positive": int(labels[fit_indices].sum()),
        "n_heldout": int(len(heldout_indices)),
        "n_heldout_positive": int(labels[heldout_indices].sum()),
        "fit_indices_sha256": _sha256(fit_indices),
        "heldout_indices_sha256": _sha256(heldout_indices),
        "candidate_rows_sha256": _sha256(candidate_rows),
        "candidate_bank_sha256": _sha256(candidate_bank),
        "task_selector": task_selector,
        "shuffled_selector": shuffled_selector,
        "shuffled_labels_sha256": _sha256(shuffled_labels),
        "results": results,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - t0,
        "output": str(output),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "fold": args.fold,
        "heldout_auc": {name: row["heldout"]["auc"] for name, row in results.items()},
        "comparisons": comparisons,
        "elapsed_sec": report["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
