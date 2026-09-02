"""KSVD gate on topology-only masked-chemistry context embeddings.

The pretrained balanced-cover reconstructor is frozen.  Each atom is encoded
only on the deterministic pass where its own categorical features are masked.
Dictionaries and downstream classifiers use one official-train internal fold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv
from code.ksvd import _omp, ksvd
from code.molhiv_beam8_incidence import typed_bond_adjacency
from code.run_beam8_nci1_chain_classification import _adjacency
from code.run_molhiv_beam8_masked_chemistry_gate import (
    _limit,
    balanced_bfs_cover_slots,
    build_patch_context_data,
)


DEFAULT_OUTPUT = Path(
    "tracks/ksvd/results/molhiv/molhiv_ssl_context_ksvd_gate_fold0_20260816.json"
)


def _random_dictionary(values: np.ndarray, atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(values), size=atoms, replace=False)
    dictionary = values[rows].T.copy()
    dictionary /= np.maximum(np.linalg.norm(dictionary, axis=0), 1e-12)
    return dictionary


def _graph_features(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [values.mean(axis=0), values.std(axis=0), np.max(np.abs(values), axis=0)]
    )


def _code_features(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            values.mean(axis=0),
            np.mean(np.abs(values), axis=0),
            np.max(np.abs(values), axis=0),
            np.mean(np.abs(values) > 1e-12, axis=0),
        ]
    )


def _evaluate(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    heldout_x: np.ndarray,
    heldout_y: np.ndarray,
    seed: int,
) -> dict[str, float]:
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        ),
    )
    classifier.fit(fit_x, fit_y)
    fit_probability = classifier.predict_proba(fit_x)[:, 1]
    probability = classifier.predict_proba(heldout_x)[:, 1]
    return {
        "fit_roc_auc": float(roc_auc_score(fit_y, fit_probability)),
        "heldout_roc_auc": float(roc_auc_score(heldout_y, probability)),
        "heldout_average_precision": float(
            average_precision_score(heldout_y, probability)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-cache",
        default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=8000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int, default=1000)
    parser.add_argument("--limit-seed", type=int, default=20260816)
    parser.add_argument("--cover-seeds", type=int, nargs="+", default=[20260813, 20260814, 20260815, 20260816])
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--context-groups", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reservoir", type=int, default=6000)
    parser.add_argument("--dictionary-atoms", type=int, default=32)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--ksvd-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred.mol_encoder import BondEncoder
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
        from torch_geometric.loader import DataLoader
        from torch_geometric.utils import scatter
    except ImportError as exc:
        raise RuntimeError(f"SSL context KSVD dependencies are missing: {exc}") from exc

    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "multi_balanced_bfs":
        raise ValueError("checkpoint must be a balanced-BFS masked reconstructor")
    if checkpoint.get("cover_canonicalization") != "topology_only":
        raise ValueError("checkpoint must use topology-only covers")
    if int(checkpoint.get("fold", -1)) != args.fold:
        raise ValueError("checkpoint fold mismatch")
    hidden = int(checkpoint["hidden"])

    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise AssertionError("categorical chemistry features are required")
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)
    with np.load(args.fold_cache, allow_pickle=False) as folds:
        full_fit = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        full_heldout = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
    if set(full_fit) | set(full_heldout) != set(official_train):
        raise AssertionError("internal fold does not partition official train")
    forbidden = set(official_valid) | set(official_test)
    if (set(full_fit) | set(full_heldout)) & forbidden:
        raise AssertionError("official valid/test leakage")
    checkpoint_fit = np.asarray(checkpoint["fit_indices"], dtype=np.int64)
    if not set(checkpoint_fit).issubset(set(full_fit)):
        raise ValueError("checkpoint SSL fit indices leave the fold-fit split")
    fit_indices = _limit(full_fit, args.limit_per_split, args.limit_seed + 1)
    heldout_indices = _limit(full_heldout, args.limit_per_split, args.limit_seed + 2)
    selected = np.concatenate([fit_indices, heldout_indices])

    atom_dims = tuple(int(value) for value in get_atom_feature_dims())
    bond_dims = tuple(int(value) for value in get_bond_feature_dims())
    items: dict[int, Any] = {}
    for count, raw_index in enumerate(selected, 1):
        graph_index = int(raw_index)
        graph = bundle.graphs[graph_index]
        atoms = np.asarray(bundle.node_feats[graph_index], dtype=np.int64)
        edge_features = bundle.edge_feats[graph_index]
        adjacency = _adjacency(graph)
        cover_typed = (adjacency != 0).astype(np.int16)
        canonical_types = np.zeros(graph.n, dtype=np.int64)
        covers = [
            balanced_bfs_cover_slots(
                adjacency,
                cover_typed,
                canonical_types,
                seed=int(cover_seed),
                patch_size=args.patch_size,
                overlap=args.overlap,
                edge_capacity_multiplier=args.edge_capacity_multiplier,
            )
            for cover_seed in args.cover_seeds
        ]
        item = build_patch_context_data(
            graph_index,
            atoms,
            edge_features,
            covers,
            patch_size=args.patch_size,
            context_groups=args.context_groups,
            shuffle_seed=args.seed + 314159,
        )
        item.graph_index = torch.tensor([graph_index], dtype=torch.long)
        items[graph_index] = item
        if count == 1 or count % 200 == 0 or count == len(selected):
            print(f"prepared balanced-cover SSL items {count}/{len(selected)}", flush=True)

    pair_count = args.patch_size * (args.patch_size - 1) // 2

    class MaskableAtomEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(dim + 1, hidden) for dim in atom_dims]
            )

        def forward(self, values: Any, mask: Any) -> Any:
            output = 0.0
            for field, (dim, embedding) in enumerate(zip(atom_dims, self.embeddings)):
                indices = values[:, field].clone()
                indices[mask] = dim
                output = output + embedding(indices)
            return output

    class Reconstructor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atom_encoder = MaskableAtomEncoder()
            self.bond_encoder = BondEncoder(hidden)
            self.slot_embedding = nn.Embedding(args.patch_size, hidden)
            self.pair_embedding = nn.Embedding(pair_count, hidden)
            self.patch_update = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.message = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.heads = nn.ModuleList([nn.Linear(hidden, dim) for dim in atom_dims])

        def encode(self, data: Any, mask: Any) -> Any:
            atom_h = self.atom_encoder(data["atom"].x, mask)
            contains = data["patch", "contains", "atom"]
            patch_index, atom_index = contains.edge_index
            slot_h = self.slot_embedding(contains.slot)
            patch_count = int(data["patch"].num_nodes)
            patch_h = scatter(
                atom_h[atom_index] + slot_h,
                patch_index,
                dim=0,
                dim_size=patch_count,
                reduce="sum",
            )
            degree = scatter(
                torch.ones_like(patch_index, dtype=atom_h.dtype), patch_index,
                dim=0, dim_size=patch_count, reduce="sum"
            ).clamp_min(1.0)
            patch_h = patch_h / torch.sqrt(degree).unsqueeze(-1)
            internal = data["patch", "internal", "patch"]
            if internal.edge_index.shape[1] > 0:
                internal_patch = internal.edge_index[0]
                internal_h = self.bond_encoder(internal.edge_attr) + self.pair_embedding(internal.pair)
                internal_sum = scatter(internal_h, internal_patch, dim=0, dim_size=patch_count, reduce="sum")
                internal_degree = scatter(
                    torch.ones_like(internal_patch, dtype=atom_h.dtype), internal_patch,
                    dim=0, dim_size=patch_count, reduce="sum"
                ).clamp_min(1.0)
                patch_h = patch_h + internal_sum / torch.sqrt(internal_degree).unsqueeze(-1)
            patch_h = self.patch_update(patch_h)
            messages = self.message(torch.cat([patch_h[patch_index], slot_h], dim=1))
            return scatter(messages, atom_index, dim=0, dim_size=len(atom_h), reduce="mean")

    device = torch.device(args.device)
    model = Reconstructor().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader([items[int(index)] for index in selected], batch_size=args.batch_size, shuffle=False)
    latents: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            values = torch.zeros((data["atom"].num_nodes, hidden), device=device)
            assigned = torch.zeros(data["atom"].num_nodes, dtype=torch.bool, device=device)
            for group in range(args.context_groups):
                mask = data["atom"].context_group == group
                context = model.encode(data, mask)
                values[mask] = context[mask]
                assigned |= mask
            if not bool(assigned.all()):
                raise AssertionError("not every atom received a masked-context latent")
            ptr = data["atom"].ptr.detach().cpu().numpy().astype(np.int64)
            graph_indices = data.graph_index.detach().cpu().numpy().astype(np.int64)
            values_np = values.detach().cpu().numpy()
            for offset, graph_index in enumerate(graph_indices):
                latents[int(graph_index)] = values_np[ptr[offset]:ptr[offset + 1]]

    fit_rows = np.concatenate([latents[int(index)] for index in fit_indices])
    mean = fit_rows.mean(axis=0)
    def normalize(values: np.ndarray) -> np.ndarray:
        centered = values.astype(np.float64) - mean
        return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    normalized = {index: normalize(values) for index, values in latents.items()}
    reservoir_rng = np.random.default_rng(args.seed)
    reservoir_count = min(args.reservoir, len(fit_rows))
    fit_normalized = np.concatenate([normalized[int(index)] for index in fit_indices])
    reservoir_rows = reservoir_rng.choice(len(fit_normalized), reservoir_count, replace=False)
    reservoir = fit_normalized[reservoir_rows]

    pca_u, singular_values, _ = randomized_svd(
        reservoir.T, n_components=args.dictionary_atoms, n_iter=5, random_state=args.seed
    )
    dictionaries = {
        "random_dictionary": _random_dictionary(reservoir, args.dictionary_atoms, args.seed),
        "ksvd": ksvd(
            reservoir.T,
            n_atoms=args.dictionary_atoms,
            T=args.sparsity,
            T_min=1,
            n_iter=args.ksvd_iterations,
            seed=args.seed,
        )[0],
    }

    features: dict[str, dict[int, np.ndarray]] = {name: {} for name in ("raw", "pca", *dictionaries)}
    reconstruction_error: dict[str, list[float]] = {name: [] for name in dictionaries}
    for graph_index, values in normalized.items():
        features["raw"][graph_index] = _graph_features(values)
        pca_values = values @ pca_u
        features["pca"][graph_index] = _graph_features(pca_values)
        for name, dictionary in dictionaries.items():
            codes = np.stack([_omp(dictionary, row, args.sparsity) for row in values])
            features[name][graph_index] = _code_features(codes)
            reconstruction = codes @ dictionary.T
            reconstruction_error[name].extend(np.mean((values - reconstruction) ** 2, axis=1).tolist())

    labels = np.asarray(bundle.y, dtype=np.int64)
    results = {}
    for name, values in features.items():
        fit_x = np.stack([values[int(index)] for index in fit_indices])
        heldout_x = np.stack([values[int(index)] for index in heldout_indices])
        results[name] = {
            **_evaluate(fit_x, labels[fit_indices], heldout_x, labels[heldout_indices], args.seed),
            "feature_dimension": int(fit_x.shape[1]),
        }
        if name in reconstruction_error:
            results[name]["node_reconstruction_mse"] = float(np.mean(reconstruction_error[name]))

    best = max(results, key=lambda name: results[name]["heldout_roc_auc"])
    payload = {
        "protocol_id": "molhiv-fold-only-balanced-cover-ssl-context-ksvd-gate-v1",
        "scope": "official-train internal scaffold fold only",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "config": {**vars(args), "checkpoint": str(args.checkpoint), "output": str(args.output)},
        "split": {"fit_graphs": len(fit_indices), "heldout_graphs": len(heldout_indices)},
        "latent": {"dimension": hidden, "reservoir": reservoir_count},
        "pca_singular_values": singular_values.tolist(),
        "results": results,
        "best_heldout_auc_family": best,
        "ksvd_auc_delta_vs_raw": results["ksvd"]["heldout_roc_auc"] - results["raw"]["heldout_roc_auc"],
        "ksvd_auc_delta_vs_pca": results["ksvd"]["heldout_roc_auc"] - results["pca"]["heldout_roc_auc"],
        "ksvd_auc_delta_vs_random_dictionary": results["ksvd"]["heldout_roc_auc"] - results["random_dictionary"]["heldout_roc_auc"],
        "elapsed_seconds": float(time.time() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": results, "best": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
