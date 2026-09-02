"""Strict scaffold-fold screen for a task-adapted KSVD-primary graph model.

The model deliberately removes the strong multi-layer GINE classifier used by
previous fusion experiments.  Its graph prediction is built from sparse local
context codes, center-atom identities, and MIL/DeepSets pooling.  A fold-fitted
masked-context latent is coded by a differentiable, OMP-warm-started iterative
hard-thresholding (IHT) layer.  The dictionary can be frozen or adapted by the
graph-label objective while reconstruction and anchor penalties preserve the
KSVD formulation.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv


@dataclass(frozen=True)
class ControlSpec:
    name: str
    family: str
    adapt_dictionary: bool
    use_reconstruction: bool


def _parse_controls(raw: str) -> list[ControlSpec]:
    known = {
        "frozen_ksvd": ControlSpec("frozen_ksvd", "ksvd", False, True),
        "adapt_ksvd": ControlSpec("adapt_ksvd", "ksvd", True, True),
        "adapt_random": ControlSpec("adapt_random", "random_patch", True, True),
        "adapt_pca": ControlSpec("adapt_pca", "pca", True, True),
        "frozen_pca": ControlSpec("frozen_pca", "pca", False, True),
        "frozen_random": ControlSpec("frozen_random", "random_patch", False, True),
        "adapt_ksvd_no_recon": ControlSpec(
            "adapt_ksvd_no_recon", "ksvd", True, False
        ),
    }
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("at least one control is required")
    unknown = sorted(set(names).difference(known))
    if unknown:
        raise ValueError(f"unknown controls: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate controls are not allowed")
    return [known[name] for name in names]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument(
        "--controls",
        default="frozen_ksvd,adapt_ksvd,adapt_random,adapt_ksvd_no_recon",
    )
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--code-steps", type=int, default=2)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--iht-step-scale", type=float, default=0.9)
    ap.add_argument("--task-lr", type=float, default=1e-3)
    ap.add_argument("--dictionary-lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--recon-weight", type=float, default=0.1)
    ap.add_argument("--anchor-weight", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    controls = _parse_controls(args.controls)
    if args.fold < 0:
        raise ValueError("--fold must be non-negative")
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden <= 0:
        raise ValueError("epochs, batch size, and hidden must be positive")
    if args.code_steps < 0 or args.sparsity <= 0:
        raise ValueError("invalid code refinement configuration")
    if not 0.0 < args.iht_step_scale <= 1.0:
        raise ValueError("--iht-step-scale must be in (0,1]")
    if min(args.task_lr, args.dictionary_lr) <= 0.0:
        raise ValueError("learning rates must be positive")
    if min(args.recon_weight, args.anchor_weight) < 0.0:
        raise ValueError("regularization weights must be non-negative")

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
        train_key = f"fold_{args.fold}_train_indices"
        valid_key = f"fold_{args.fold}_valid_indices"
        if train_key not in folds.files or valid_key not in folds.files:
            raise KeyError(f"fold {args.fold} is absent from fold cache")
        fit_indices = np.asarray(folds[train_key], dtype=np.int64)
        heldout_indices = np.asarray(folds[valid_key], dtype=np.int64)
        if "original_indices" in folds.files and not np.array_equal(
            np.asarray(folds["original_indices"], dtype=np.int64), expected_original
        ):
            raise ValueError("fold cache original_indices do not match dataset")
        if "official_train_indices" in folds.files and not np.array_equal(
            np.asarray(folds["official_train_indices"], dtype=np.int64), official_train
        ):
            raise ValueError("fold cache official train does not match dataset")
    official_train_set = set(official_train.tolist())
    if not set(fit_indices.tolist()).issubset(official_train_set):
        raise ValueError("fit split leaves official train")
    if not set(heldout_indices.tolist()).issubset(official_train_set):
        raise ValueError("heldout split leaves official train")
    if set(fit_indices.tolist()).intersection(heldout_indices.tolist()):
        raise ValueError("fit/heldout graph leakage")
    if set(fit_indices.tolist()).union(heldout_indices.tolist()) != official_train_set:
        raise ValueError("fit/heldout split does not partition official train")

    with np.load(args.latent_cache, allow_pickle=False) as latent_source:
        required = {
            "offsets", "original_indices", "train_indices", "valid_indices",
            "test_indices", "latents",
        }
        missing = required.difference(latent_source.files)
        if missing:
            raise KeyError(f"latent cache is missing {sorted(missing)}")
        offsets = np.asarray(latent_source["offsets"], dtype=np.int64)
        latent_original = np.asarray(latent_source["original_indices"], dtype=np.int64)
        latent_train = np.asarray(latent_source["train_indices"], dtype=np.int64)
        latent_valid = np.asarray(latent_source["valid_indices"], dtype=np.int64)
        latent_test = np.asarray(latent_source["test_indices"], dtype=np.int64)
        latents_np = np.asarray(latent_source["latents"], dtype=np.float32)

    with np.load(args.token_cache, allow_pickle=False) as token_source:
        token_arrays = {key: np.asarray(token_source[key]) for key in token_source.files}

    alignment_checks = [
        (latent_original, expected_original, "latent original_indices"),
        (latent_train, official_train, "latent train_indices"),
        (latent_valid, official_valid, "latent valid_indices"),
        (latent_test, official_test, "latent test_indices"),
        (np.asarray(token_arrays["offsets"], dtype=np.int64), offsets, "token offsets"),
        (np.asarray(token_arrays["original_indices"], dtype=np.int64), expected_original,
         "token original_indices"),
    ]
    for actual, expected, name in alignment_checks:
        if not np.array_equal(actual, expected):
            raise ValueError(f"misaligned {name}")
    if offsets.shape != (len(bundle.graphs) + 1,):
        raise ValueError(f"invalid offsets shape {offsets.shape}")
    if latents_np.shape[0] != int(offsets[-1]):
        raise ValueError("latent row count and offsets disagree")
    latent_dim = int(latents_np.shape[1])
    if latent_dim <= 0:
        raise ValueError("empty latent dimension")

    official_nontrain_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in np.concatenate([official_valid, official_test])
    ])
    if np.any(latents_np[official_nontrain_rows] != 0):
        raise AssertionError("official-valid/test latent rows are nonzero")
    heldout_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in heldout_indices
    ])
    heldout_norms = np.linalg.norm(latents_np[heldout_rows].astype(np.float64), axis=1)
    if np.mean(heldout_norms > 0.99) < 0.999:
        raise ValueError("heldout scaffold latents are missing or not normalized")

    families = sorted({control.family for control in controls})
    dictionaries: dict[str, np.ndarray] = {}
    initial_codes: dict[str, np.ndarray] = {}
    for family in families:
        d_key = f"dictionary_{family}"
        x_key = f"tokens_{family}"
        if d_key not in token_arrays or x_key not in token_arrays:
            raise KeyError(f"token cache lacks family {family}")
        dictionary = np.asarray(token_arrays[d_key], dtype=np.float32)
        codes = np.asarray(token_arrays[x_key], dtype=np.float32)
        if dictionary.shape[0] != latent_dim:
            raise ValueError(
                f"{family} dictionary dimension {dictionary.shape} != latent {latent_dim}"
            )
        if codes.shape != (int(offsets[-1]), dictionary.shape[1]):
            raise ValueError(f"invalid {family} code shape {codes.shape}")
        if np.any(codes[official_nontrain_rows] != 0):
            raise AssertionError(f"{family} encoded official-valid/test rows")
        dictionaries[family] = dictionary
        initial_codes[family] = codes
    n_atoms = int(next(iter(dictionaries.values())).shape[1])
    if any(d.shape[1] != n_atoms for d in dictionaries.values()):
        raise ValueError("controls have unmatched dictionary sizes")
    if args.sparsity > n_atoms:
        raise ValueError("sparsity exceeds dictionary size")

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
    code_tensors = {family: torch.from_numpy(array) for family, array in initial_codes.items()}

    class DictionaryMIL(nn.Module):
        def __init__(self, dictionary_init: np.ndarray, adapt_dictionary: bool):
            super().__init__()
            d0 = torch.tensor(dictionary_init, dtype=torch.float32)
            d0 = d0 / d0.norm(dim=0, keepdim=True).clamp_min(1e-12)
            self.register_buffer("dictionary_initial", d0.clone())
            self.dictionary_raw = nn.Parameter(d0.clone(), requires_grad=adapt_dictionary)
            self.center_embeddings = nn.ModuleList([
                nn.Embedding(dim, args.hidden) for dim in atom_dims
            ])
            self.signed_atom_embeddings = nn.Parameter(
                torch.empty(n_atoms, args.hidden)
            )
            self.absolute_atom_embeddings = nn.Parameter(
                torch.empty(n_atoms, args.hidden)
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(3 * args.hidden + 1, args.hidden),
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
            nn.init.xavier_uniform_(self.signed_atom_embeddings)
            nn.init.xavier_uniform_(self.absolute_atom_embeddings)
            for module in list(self.node_mlp.modules()) + list(self.attention.modules()) + list(self.graph_head.modules()):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def dictionary(self):
            return self.dictionary_raw / self.dictionary_raw.norm(
                dim=0, keepdim=True
            ).clamp_min(1e-12)

        def refine_codes(self, z, x0, dictionary):
            x = x0
            # Safe IHT step for ||z - D x||^2.  The spectral step is detached:
            # it is a stability bound, not an extra route for task gradients.
            spectral_sq = torch.linalg.matrix_norm(dictionary, ord=2).square().detach()
            step = args.iht_step_scale / spectral_sq.clamp_min(1e-6)
            for _ in range(args.code_steps):
                residual = z - x @ dictionary.T
                proposal = x + step * (residual @ dictionary)
                if args.sparsity < proposal.shape[1]:
                    keep = proposal.abs().topk(args.sparsity, dim=1).indices
                    mask = torch.zeros_like(proposal).scatter_(1, keep, 1.0)
                    proposal = proposal * mask
                x = proposal
            return x

        def forward(self, z, x0, center, node_graph):
            dictionary = self.dictionary()
            codes = self.refine_codes(z, x0, dictionary)
            center_h = 0.0
            for field, emb in enumerate(self.center_embeddings):
                center_h = center_h + emb(center[:, field])
            mass = codes.abs().sum(dim=1, keepdim=True)
            normalized_signed = codes / mass.clamp_min(1e-6)
            normalized_abs = codes.abs() / mass.clamp_min(1e-6)
            signed_h = normalized_signed @ self.signed_atom_embeddings
            absolute_h = normalized_abs @ self.absolute_atom_embeddings
            node_h = self.node_mlp(torch.cat([
                center_h, signed_h, absolute_h, torch.log1p(mass)
            ], dim=1))
            attention = graph_softmax(self.attention(node_h).view(-1), node_graph)
            attention_pool = global_add_pool(node_h * attention[:, None], node_graph)
            mean_pool = global_mean_pool(node_h, node_graph)
            max_pool = global_max_pool(node_h, node_graph)
            logits = self.graph_head(torch.cat([
                attention_pool, mean_pool, max_pool
            ], dim=1)).view(-1)
            reconstruction = codes @ dictionary.T
            return logits, codes, reconstruction, dictionary

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
    def evaluate(model, loader, code_tensor):
        model.eval()
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        rec_sum = 0.0
        node_count = 0
        support_sum = 0.0
        mass_sum = 0.0
        for batch in loader:
            batch = move_batch(batch)
            rows_cpu = batch["rows"].cpu()
            z = latents[rows_cpu].to(device)
            x0 = code_tensor[rows_cpu].to(device)
            logits, codes, reconstruction, _ = model(
                z, x0, batch["center"], batch["node_graph"]
            )
            all_scores.append(logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].detach().cpu().numpy())
            rec_sum += float((z - reconstruction).square().sum())
            node_count += int(z.shape[0])
            support_sum += float((codes.abs() > 1e-8).sum())
            mass_sum += float(codes.abs().sum())
        scores = np.concatenate(all_scores)
        y = np.concatenate(all_labels)
        return {
            "auc": float(roc_auc_score(y, scores)),
            "n_graphs": int(len(y)),
            "n_positive": int(y.sum()),
            "reconstruction_mse_per_node": rec_sum / max(node_count, 1),
            "mean_active_coefficients": support_sum / max(node_count, 1),
            "mean_code_l1": mass_sum / max(node_count, 1),
            "score_sha256": _sha256(scores.astype(np.float64)),
        }

    results: dict[str, Any] = {}
    for control_i, control in enumerate(controls):
        control_start = time.time()
        _seed_everything(args.seed, torch)
        model = DictionaryMIL(
            dictionaries[control.family], control.adapt_dictionary
        ).to(device)
        task_parameters = [
            parameter for name, parameter in model.named_parameters()
            if name != "dictionary_raw" and parameter.requires_grad
        ]
        groups = [{
            "params": task_parameters,
            "lr": args.task_lr,
            "weight_decay": args.weight_decay,
        }]
        if control.adapt_dictionary:
            groups.append({
                "params": [model.dictionary_raw],
                "lr": args.dictionary_lr,
                "weight_decay": 0.0,
            })
        optimizer = torch.optim.AdamW(groups)
        train_loader = make_loader(fit_indices, True, 1000)
        code_tensor = code_tensors[control.family]
        history: list[dict[str, float]] = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            sums = {"total": 0.0, "task": 0.0, "reconstruction": 0.0, "anchor": 0.0}
            seen_graphs = 0
            for batch in train_loader:
                batch = move_batch(batch)
                rows_cpu = batch["rows"].cpu()
                z = latents[rows_cpu].to(device)
                x0 = code_tensor[rows_cpu].to(device)
                logits, _, reconstruction, dictionary = model(
                    z, x0, batch["center"], batch["node_graph"]
                )
                task_loss = F.binary_cross_entropy_with_logits(
                    logits, batch["labels"].float()
                )
                reconstruction_loss = (z - reconstruction).square().sum(dim=1).mean()
                anchor_loss = (
                    dictionary - model.dictionary_initial
                ).square().sum(dim=0).mean()
                active_recon_weight = (
                    args.recon_weight if control.use_reconstruction else 0.0
                )
                loss = (
                    task_loss
                    + active_recon_weight * reconstruction_loss
                    + args.anchor_weight * anchor_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                batch_graphs = int(batch["labels"].shape[0])
                seen_graphs += batch_graphs
                sums["total"] += float(loss.detach()) * batch_graphs
                sums["task"] += float(task_loss.detach()) * batch_graphs
                sums["reconstruction"] += float(reconstruction_loss.detach()) * batch_graphs
                sums["anchor"] += float(anchor_loss.detach()) * batch_graphs
            row = {key: value / max(seen_graphs, 1) for key, value in sums.items()}
            row["epoch"] = float(epoch)
            history.append(row)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(
                    f"{control.name} epoch={epoch:02d} total={row['total']:.5f} "
                    f"task={row['task']:.5f} rec={row['reconstruction']:.5f} "
                    f"anchor={row['anchor']:.6f}",
                    flush=True,
                )

        fit_loader = make_loader(fit_indices, False, 2000)
        heldout_loader = make_loader(heldout_indices, False, 3000)
        fit_metrics = evaluate(model, fit_loader, code_tensor)
        heldout_metrics = evaluate(model, heldout_loader, code_tensor)
        with torch.no_grad():
            dictionary = model.dictionary().detach().cpu()
            d0 = model.dictionary_initial.detach().cpu()
            atom_cosine = (dictionary * d0).sum(dim=0)
            gram = dictionary.T @ dictionary
            offdiag = gram - torch.eye(n_atoms)
            dictionary_metrics = {
                "mean_atom_l2_movement": float(
                    (dictionary - d0).norm(dim=0).mean()
                ),
                "max_atom_l2_movement": float(
                    (dictionary - d0).norm(dim=0).max()
                ),
                "mean_aligned_atom_cosine": float(atom_cosine.mean()),
                "min_aligned_atom_cosine": float(atom_cosine.min()),
                "mean_abs_coherence": float(
                    offdiag.abs().sum() / max(n_atoms * (n_atoms - 1), 1)
                ),
                "max_abs_coherence": float(offdiag.abs().max()),
                "dictionary_sha256": _sha256(dictionary.numpy()),
            }
        results[control.name] = {
            "family": control.family,
            "adapt_dictionary": control.adapt_dictionary,
            "use_reconstruction": control.use_reconstruction,
            "trainable_parameters": int(sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
            "fit": fit_metrics,
            "heldout": heldout_metrics,
            "dictionary": dictionary_metrics,
            "history": history,
            "elapsed_sec": time.time() - control_start,
        }
        print(
            f"{control.name} fixed_epoch={args.epochs} "
            f"fit_auc={fit_metrics['auc']:.4f} heldout_auc={heldout_metrics['auc']:.4f} "
            f"heldout_rec={heldout_metrics['reconstruction_mse_per_node']:.5f} "
            f"dict_move={dictionary_metrics['mean_atom_l2_movement']:.5f}",
            flush=True,
        )

    comparisons: dict[str, Any] = {}
    if "frozen_ksvd" in results and "adapt_ksvd" in results:
        delta = (
            results["adapt_ksvd"]["heldout"]["auc"]
            - results["frozen_ksvd"]["heldout"]["auc"]
        )
        comparisons["adapt_ksvd_minus_frozen_ksvd_auc"] = float(delta)
        comparisons["per_fold_plus_005"] = bool(delta >= 0.005)
    if "adapt_random" in results and "adapt_ksvd" in results:
        comparisons["adapt_ksvd_minus_adapt_random_auc"] = float(
            results["adapt_ksvd"]["heldout"]["auc"]
            - results["adapt_random"]["heldout"]["auc"]
        )
    if "adapt_ksvd_no_recon" in results and "adapt_ksvd" in results:
        comparisons["reconstruction_ablation_delta_auc"] = float(
            results["adapt_ksvd"]["heldout"]["auc"]
            - results["adapt_ksvd_no_recon"]["heldout"]["auc"]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": "molhiv-task-adapted-dictionary-primary-mil-scaffold-v1",
        "date": "2026-07-28",
        "hypothesis": (
            "graph-label adaptation of a reconstruction-anchored sparse dictionary "
            "is more useful when KSVD is the primary local representation rather "
            "than an ignorable side channel of a strong 3-layer GINE"
        ),
        "architecture": {
            "local_input": "fold-specific masked-context SSL latent plus center atom fields",
            "coder": "OMP warm start plus unrolled exact-top-k iterative hard thresholding",
            "graph_model": "dictionary-atom occurrence embeddings plus attention/mean/max MIL pooling",
            "message_passing_layers": 0,
            "graph_labels_assigned_to_individual_nodes": False,
        },
        "selection_policy": {
            "data": "official-train only",
            "outer_validation": "one held-out Bemis-Murcko scaffold fold",
            "epoch_policy": f"single evaluation after fixed epoch {args.epochs}",
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "promotion_rule_across_3_folds": {
                "mean_adapt_minus_frozen_auc_at_least": 0.005,
                "minimum_fold_wins": 2,
                "must_beat_adapt_random_mean": True,
                "heldout_reconstruction_must_not_collapse": True,
            },
        },
        "config": vars(args),
        "fold": int(args.fold),
        "fit_indices_sha256": _sha256(fit_indices),
        "heldout_indices_sha256": _sha256(heldout_indices),
        "n_fit": int(len(fit_indices)),
        "n_fit_positive": int(labels[fit_indices].sum()),
        "n_heldout": int(len(heldout_indices)),
        "n_heldout_positive": int(labels[heldout_indices].sum()),
        "latent_dim": latent_dim,
        "n_atoms": n_atoms,
        "results": results,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - t0,
        "output": str(output),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "fold": args.fold,
        "comparisons": comparisons,
        "heldout_auc": {
            name: row["heldout"]["auc"] for name, row in results.items()
        },
        "elapsed_sec": report["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
