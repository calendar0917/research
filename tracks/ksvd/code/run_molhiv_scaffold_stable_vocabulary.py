"""Label-free scaffold-balanced real-patch vocabulary pilot for MolHIV.

The vocabulary is selected only from each outer-fit split.  A deterministic,
graph-balanced observed-patch candidate pool is scored by facility-location
coverage over multiple deterministic target patches per fit graph.  Uniform,
true-scaffold-balanced, and shuffled-scaffold objectives are matched controls.
Official heldout encoding/evaluation is enabled only through an explicit frozen split protocol.
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
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv
from code.molhiv_evaluation_protocol import (
    evaluation_description, forbidden_encoded_indices, resolve_evaluation_protocol,
    validate_latent_sidecar,
)


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
    ap.add_argument("--token-cache", default="", help="optional legacy token-cache audit; direct fold-fit latent caches may self-audit")
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--evaluation-split", choices=("scaffold_fold", "official_valid", "official_test"), default="scaffold_fold")
    ap.add_argument(
        "--controls",
        default="fixed_random,farthest,uniform_facility,scaffold_facility,shuffled_scaffold_facility",
    )
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prototype-seed", type=int, default=20260728)
    ap.add_argument("--candidate-bank-size", type=int, default=0, help="0 means one candidate source graph per outer-fit graph")
    ap.add_argument("--facility-seed", type=int, default=20260728)
    ap.add_argument("--facility-targets-per-graph", type=int, default=3)
    ap.add_argument("--facility-memmap-dir", default="/tmp", help="storage only; does not affect selection")
    ap.add_argument("--n-prototypes", type=int, default=32)
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
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    known_controls = {
        "fixed_random", "farthest", "uniform_facility",
        "scaffold_facility", "shuffled_scaffold_facility", "hybrid_half",
    }
    controls = [x.strip() for x in args.controls.split(",") if x.strip()]
    unknown = sorted(set(controls).difference(known_controls))
    if not controls or unknown or len(set(controls)) != len(controls):
        raise ValueError(f"invalid controls={controls}, unknown={unknown}")
    if args.fold < 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("invalid fold/training configuration")
    if min(args.n_prototypes, args.sparsity, args.facility_targets_per_graph) <= 0:
        raise ValueError("prototype/sparsity/target sizes must be positive")
    if args.candidate_bank_size < 0:
        raise ValueError("candidate-bank-size must be nonnegative")
    if args.sparsity > args.n_prototypes:
        raise ValueError("sparsity exceeds n-prototypes")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

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

    protocol = resolve_evaluation_protocol(
        evaluation_split=args.evaluation_split,
        fold_cache=args.fold_cache,
        fold=args.fold,
        expected_original=expected_original,
        official_train=official_train,
        official_valid=official_valid,
        official_test=official_test,
    )
    fit_indices = protocol.fit_indices
    heldout_indices = protocol.heldout_indices
    scaffold_groups = protocol.scaffold_groups

    with np.load(args.latent_cache, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        latent_original = np.asarray(source["original_indices"], dtype=np.int64)
        latent_train = np.asarray(source["train_indices"], dtype=np.int64)
        latent_valid = np.asarray(source["valid_indices"], dtype=np.int64)
        latent_test = np.asarray(source["test_indices"], dtype=np.int64)
        latents_np = np.asarray(source["latents"], dtype=np.float32)
    token_arrays: dict[str, np.ndarray] | None = None
    if args.token_cache:
        with np.load(args.token_cache, allow_pickle=False) as source:
            token_arrays = {key: np.asarray(source[key]) for key in source.files}

    checks = [
        (latent_original, expected_original, "latent original_indices"),
        (latent_train, official_train, "latent train_indices"),
        (latent_valid, official_valid, "latent valid_indices"),
        (latent_test, official_test, "latent test_indices"),
    ]
    if token_arrays is not None:
        checks.extend([
            (np.asarray(token_arrays["offsets"], dtype=np.int64), offsets, "token offsets"),
            (np.asarray(token_arrays["original_indices"], dtype=np.int64), expected_original, "token original_indices"),
        ])
    for actual, expected, name in checks:
        if not np.array_equal(actual, expected):
            raise ValueError(f"misaligned {name}")
    if latents_np.shape[0] != int(offsets[-1]):
        raise ValueError("latent row count and offsets disagree")
    forbidden = forbidden_encoded_indices(args.evaluation_split, official_valid, official_test)
    forbidden_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in forbidden
    ])
    if np.any(latents_np[forbidden_rows] != 0):
        raise AssertionError("forbidden official split latent rows are nonzero")

    fit_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in fit_indices
    ])
    heldout_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in heldout_indices
    ])
    if np.mean(np.linalg.norm(latents_np[np.concatenate([fit_rows, heldout_rows])], axis=1) > 0.99) < 0.999:
        raise ValueError("fit/heldout latents are missing or unnormalized")

    expected_fit_sha = _sha256(fit_indices)
    token_meta: dict[str, Any] | None = None
    if args.token_cache:
        token_sidecar = Path(args.token_cache).with_suffix(".json")
        if not token_sidecar.exists():
            raise FileNotFoundError(f"missing token-cache audit sidecar: {token_sidecar}")
        token_meta = json.loads(token_sidecar.read_text(encoding="utf-8"))
        if token_meta.get("dictionary_fit_indices_sha256") != expected_fit_sha:
            raise ValueError("token dictionary was not fit on this fit split")
        if bool(token_meta.get("encoded_official_valid", True)):
            raise ValueError("token cache unexpectedly encoded official-valid graphs")

    latent_meta = validate_latent_sidecar(args.latent_cache, args.evaluation_split, expected_fit_sha)
    if args.token_cache:
        latent_token_name = Path(latent_meta.get("config", {}).get("token_cache", "")).name
        if latent_token_name != Path(args.token_cache).name:
            raise ValueError("latent cache was not derived from the requested token cache")
    group_by_graph = {int(i): str(g) for i, g in zip(official_train, scaffold_groups)}
    fit_groups = np.asarray([group_by_graph[int(i)] for i in fit_indices])

    # One deterministic observed patch per source graph gives a much broader and
    # less seed-sensitive candidate pool than the earlier 256-graph reservoir.
    # An optional cap is deterministic and scaffold-round-robin rather than random.
    if args.candidate_bank_size and args.candidate_bank_size < len(fit_indices):
        order = np.lexsort((fit_indices, fit_groups))
        by_group: dict[str, list[int]] = {}
        for pos in order:
            by_group.setdefault(str(fit_groups[int(pos)]), []).append(int(pos))
        source_positions: list[int] = []
        depth = 0
        group_names = sorted(by_group)
        while len(source_positions) < args.candidate_bank_size:
            added = False
            for group in group_names:
                rows = by_group[group]
                if depth < len(rows):
                    source_positions.append(rows[depth])
                    added = True
                    if len(source_positions) == args.candidate_bank_size:
                        break
            if not added:
                break
            depth += 1
        candidate_source_positions = np.asarray(source_positions, dtype=np.int64)
    else:
        candidate_source_positions = np.arange(len(fit_indices), dtype=np.int64)
    candidate_source_graphs = fit_indices[candidate_source_positions]

    def deterministic_rows(graph_indices: np.ndarray, count: int, salt: int) -> tuple[np.ndarray, np.ndarray]:
        selected_rows: list[int] = []
        selected_graphs: list[int] = []
        for raw_i in graph_indices:
            i = int(raw_i)
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            n = hi - lo
            take = min(count, n)
            rng = np.random.default_rng(args.facility_seed + 1_000_003 * i + salt)
            local = np.sort(rng.choice(n, size=take, replace=False))
            selected_rows.extend((lo + local).tolist())
            selected_graphs.extend([i] * take)
        return np.asarray(selected_rows, dtype=np.int64), np.asarray(selected_graphs, dtype=np.int64)

    candidate_rows, candidate_row_graphs = deterministic_rows(candidate_source_graphs, 1, 17)
    if not np.array_equal(candidate_row_graphs, candidate_source_graphs):
        raise AssertionError("one-candidate-per-graph construction failed")
    candidate_bank = _normalize_rows(latents_np[candidate_rows])
    if args.n_prototypes > len(candidate_bank):
        raise ValueError("n-prototypes exceeds candidate bank")

    target_rows, target_graphs = deterministic_rows(
        fit_indices, args.facility_targets_per_graph, 104729
    )
    target_bank = _normalize_rows(latents_np[target_rows])
    target_groups = np.asarray([group_by_graph[int(i)] for i in target_graphs])

    def target_weights(groups: np.ndarray) -> np.ndarray:
        graph_count: dict[int, int] = {}
        scaffold_graphs: dict[str, set[int]] = {}
        for graph, group in zip(target_graphs, groups):
            graph_count[int(graph)] = graph_count.get(int(graph), 0) + 1
            scaffold_graphs.setdefault(str(group), set()).add(int(graph))
        weights = np.empty(len(target_graphs), dtype=np.float64)
        n_scaffolds = len(scaffold_graphs)
        for j, (graph, group) in enumerate(zip(target_graphs, groups)):
            weights[j] = (1.0 / n_scaffolds) / len(scaffold_graphs[str(group)]) / graph_count[int(graph)]
        weights /= weights.sum()
        return weights

    graph_counts = {int(i): int(np.sum(target_graphs == int(i))) for i in fit_indices}
    uniform_weights = np.asarray([1.0 / len(fit_indices) / graph_counts[int(i)] for i in target_graphs], dtype=np.float64)
    uniform_weights /= uniform_weights.sum()
    scaffold_weights = target_weights(target_groups)
    shuffled_fit_groups = fit_groups.copy()
    shuffled_rng = np.random.default_rng(args.facility_seed + 71_017 * (args.fold + 1))
    shuffled_rng.shuffle(shuffled_fit_groups)
    shuffled_group_by_graph = {int(i): str(g) for i, g in zip(fit_indices, shuffled_fit_groups)}
    shuffled_target_groups = np.asarray([shuffled_group_by_graph[int(i)] for i in target_graphs])
    shuffled_scaffold_weights = target_weights(shuffled_target_groups)

    # Positive cosine matches the downstream occurrence code.  The 8k pilot can
    # keep the complete matrix in RAM.  On full official-train it is several GB,
    # so build an exact float32 disk-backed matrix in target chunks and compute
    # greedy gains in candidate chunks.  This changes storage, not the objective.
    facility_names = {"uniform_facility", "scaffold_facility", "shuffled_scaffold_facility"}
    needed = set(controls)
    if "hybrid_half" in needed:
        needed.update({"farthest", "scaffold_facility"})
    needed_facility = needed.intersection(facility_names)
    facility_similarity: np.ndarray | np.memmap | None = None
    facility_memmap_path: Path | None = None
    if needed_facility:
        n_similarity = len(target_bank) * len(candidate_bank)
        if n_similarity <= 250_000_000:
            facility_similarity = np.maximum(target_bank @ candidate_bank.T, 0.0).astype(np.float32)
        else:
            facility_memmap_dir = Path(args.facility_memmap_dir)
            facility_memmap_dir.mkdir(parents=True, exist_ok=True)
            facility_memmap_path = facility_memmap_dir / (
                f"ksvd_facility_{args.evaluation_split}_fold{args.fold}_pid{__import__('os').getpid()}.f32"
            )
            facility_similarity = np.memmap(
                facility_memmap_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(target_bank), len(candidate_bank)),
            )
            target_chunk = 512
            for lo in range(0, len(target_bank), target_chunk):
                hi = min(lo + target_chunk, len(target_bank))
                block = target_bank[lo:hi] @ candidate_bank.T
                np.maximum(block, 0.0, out=block)
                facility_similarity[lo:hi] = block.astype(np.float32, copy=False)
                if lo == 0 or hi == len(target_bank) or (lo // target_chunk) % 20 == 0:
                    print(
                        f"facility matrix rows {hi}/{len(target_bank)}; "
                        f"candidates={len(candidate_bank)}",
                        flush=True,
                    )
            facility_similarity.flush()

    def select_facility(weights: np.ndarray) -> tuple[np.ndarray, list[float]]:
        if facility_similarity is None:
            raise AssertionError("facility matrix was not constructed")
        current = np.zeros(len(target_bank), dtype=np.float32)
        available = np.ones(len(candidate_bank), dtype=bool)
        chosen: list[int] = []
        objective: list[float] = []
        candidate_chunk = 256 if isinstance(facility_similarity, np.memmap) else len(candidate_bank)
        for step in range(args.n_prototypes):
            gains = np.empty(len(candidate_bank), dtype=np.float64)
            for lo in range(0, len(candidate_bank), candidate_chunk):
                hi = min(lo + candidate_chunk, len(candidate_bank))
                delta = np.asarray(facility_similarity[:, lo:hi], dtype=np.float32) - current[:, None]
                np.maximum(delta, 0.0, out=delta)
                gains[lo:hi] = np.asarray(weights @ delta, dtype=np.float64)
            gains[~available] = -np.inf
            pick = int(np.argmax(gains))
            chosen.append(pick)
            available[pick] = False
            current = np.maximum(current, np.asarray(facility_similarity[:, pick], dtype=np.float32))
            objective.append(float(weights @ current))
            print(
                f"facility step {step + 1:02d}/{args.n_prototypes}: pick={pick} "
                f"objective={objective[-1]:.6f}",
                flush=True,
            )
        return np.asarray(chosen, dtype=np.int64), objective

    prototype_rng = np.random.default_rng(args.prototype_seed + 910_003 * (args.fold + 1))
    selection_indices: dict[str, np.ndarray] = {}
    selection_objective_trace: dict[str, list[float]] = {}
    if "fixed_random" in needed:
        selection_indices["fixed_random"] = prototype_rng.choice(
            len(candidate_bank), size=args.n_prototypes, replace=False
        ).astype(np.int64)
    if "farthest" in needed:
        selection_indices["farthest"] = _select_farthest(candidate_bank, args.n_prototypes)
    facility_weight_map = {
        "uniform_facility": uniform_weights,
        "scaffold_facility": scaffold_weights,
        "shuffled_scaffold_facility": shuffled_scaffold_weights,
    }
    for name in ("uniform_facility", "scaffold_facility", "shuffled_scaffold_facility"):
        if name in needed:
            selection_indices[name], selection_objective_trace[name] = select_facility(
                facility_weight_map[name]
            )

    # Single-bank compression of the two complementary deterministic selectors:
    # retain the first 16 farthest and first 16 scaffold-facility prototypes,
    # then deterministically fill any overlap from their remaining rankings.
    if "hybrid_half" in needed:
        hybrid: list[int] = []
        for source in (selection_indices["farthest"][:16], selection_indices["scaffold_facility"][:16]):
            for idx in source:
                if int(idx) not in hybrid:
                    hybrid.append(int(idx))
        depth = 16
        while len(hybrid) < args.n_prototypes:
            added = False
            for source in (selection_indices["farthest"], selection_indices["scaffold_facility"]):
                if depth < len(source) and int(source[depth]) not in hybrid:
                    hybrid.append(int(source[depth]))
                    added = True
                    if len(hybrid) == args.n_prototypes:
                        break
            if not added and depth >= max(len(selection_indices["farthest"]), len(selection_indices["scaffold_facility"])):
                break
            depth += 1
        if len(hybrid) != args.n_prototypes:
            raise AssertionError("hybrid selector did not produce n-prototypes unique rows")
        selection_indices["hybrid_half"] = np.asarray(hybrid, dtype=np.int64)

    prototype_sets: dict[str, np.ndarray] = {
        name: candidate_bank[idx] for name, idx in selection_indices.items()
    }

    def facility_scores(chosen: np.ndarray) -> dict[str, float]:
        # Recompute only the selected 32 columns.  This is small and lets us
        # release the full disk-backed similarity matrix before model training.
        coverage = np.maximum(target_bank @ candidate_bank[chosen].T, 0.0).max(axis=1)
        return {
            "uniform_graph_coverage_objective": float(uniform_weights @ coverage),
            "true_scaffold_coverage_objective": float(scaffold_weights @ coverage),
            "shuffled_scaffold_coverage_objective": float(shuffled_scaffold_weights @ coverage),
            "target_uncovered_fraction_lt025": float(np.mean(coverage < 0.25)),
            "target_uncovered_fraction_lt050": float(np.mean(coverage < 0.50)),
        }

    facility_score_cache = {
        name: facility_scores(indices) for name, indices in selection_indices.items()
    }
    facility_similarity_audit = {
        "shape": [int(len(target_bank)), int(len(candidate_bank))],
        "dtype": "float32",
        "storage": "disk_backed_memmap" if isinstance(facility_similarity, np.memmap) else "in_memory",
        "construction": "chunked exact positive-cosine matrix" if isinstance(facility_similarity, np.memmap) else "single exact positive-cosine matrix",
        "greedy_gain_candidate_chunk": 256 if isinstance(facility_similarity, np.memmap) else int(len(candidate_bank)),
    }
    facility_similarity_sha256 = (
        None if isinstance(facility_similarity, np.memmap) else _sha256(facility_similarity)
    )
    if isinstance(facility_similarity, np.memmap):
        facility_similarity.flush()
    facility_similarity = None
    if facility_memmap_path is not None:
        try:
            facility_memmap_path.unlink()
        except FileNotFoundError:
            pass

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
        all_indices: list[np.ndarray] = []
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
            all_indices.append(batch["indices"].cpu().numpy())
            rec_sum += float((z - reconstruction).square().sum())
            node_count += int(z.shape[0])
            support_sum += float((codes > 1e-8).sum())
            mass_sum += float(codes.sum())
            best_similarity.append(cosine.max(dim=1).values.cpu().numpy())
        scores = np.concatenate(all_scores)
        y = np.concatenate(all_labels)
        best = np.concatenate(best_similarity)
        graph_indices = np.concatenate(all_indices).astype(np.int64)
        metrics = {
            "auc": float(roc_auc_score(y, scores)),
            "n_graphs": int(len(y)),
            "n_positive": int(y.sum()),
            "reconstruction_mse_per_node": rec_sum / max(node_count, 1),
            "mean_active_prototypes": support_sum / max(node_count, 1),
            "mean_occurrence_mass": mass_sum / max(node_count, 1),
            "mean_best_signed_cosine": float(best.mean()),
            "q10_best_signed_cosine": float(np.quantile(best, 0.1)),
            "graph_indices_sha256": _sha256(graph_indices),
            "score_sha256": _sha256(scores.astype(np.float64)),
        }
        if args.save_predictions:
            metrics["graph_indices"] = graph_indices.tolist()
            metrics["labels"] = y.astype(np.float32).tolist()
            metrics["scores"] = scores.astype(np.float32).tolist()
        return metrics

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
            "all_prototypes_are_real_fit_rows": True,
        }
        if control in selection_indices:
            chosen = selection_indices[control]
            selection.update({
                "candidate_indices": chosen.tolist(),
                "source_graph_indices": candidate_source_graphs[chosen].tolist(),
                "source_node_rows": candidate_rows[chosen].tolist(),
                **facility_scores(chosen),
                "facility_objective_trace": selection_objective_trace.get(control, []),
                "source_scaffold_count": int(len(set(fit_groups[candidate_source_positions[chosen]].tolist()))),
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
    if "scaffold_facility" in results:
        target_auc = results["scaffold_facility"]["heldout"]["auc"]
        for baseline in ("fixed_random", "farthest", "uniform_facility", "shuffled_scaffold_facility"):
            if baseline in results:
                comparisons[f"scaffold_facility_minus_{baseline}_auc"] = float(
                    target_auc - results[baseline]["heldout"]["auc"]
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": "molhiv-label-free-scaffold-facility-real-vocabulary-v1",
        "date": "2026-07-28",
        "hypothesis": (
            "a deterministic scaffold-balanced facility-location vocabulary can replace "
            "high-variance random real-prototype bank ensembling"
        ),
        "architecture": {
            "candidate_bank": "one deterministic observed outer-fit PCA64 patch per source graph",
            "selector": "greedy positive-cosine facility location with graph/scaffold matched controls",
            "prototype_constraint": "all selected prototypes remain exact observed outer-fit node latents",
            "graph_model": "positive top-k prototype occurrence embeddings plus attention/mean/max MIL",
            "supervised_message_passing_layers": 0,
            "upstream_context_encoder": "none; raw permutation-invariant radius-2 descriptor plus fold-fit PCA64",
            "graph_labels_assigned_to_individual_nodes": False,
        },
        "selection_policy": {
            "data": "official-train only",
            "outer_validation": evaluation_description(args.evaluation_split, args.max_graphs),
            "epoch_policy": f"single evaluation after fixed epoch {args.epochs}",
            "official_valid_evaluations": protocol.official_valid_evaluations,
            "official_test_evaluations": protocol.official_test_evaluations,
            "seed0_promotion_rule_across_3_folds": {
                "scaffold_facility_mean_gain_over_fixed_random_at_least": 0.005,
                "minimum_scaffold_facility_random_fold_wins": 2,
                "must_beat_farthest_uniform_and_shuffled_scaffold_means": True,
                "reference_three_random_bank_ensemble_mean_auc": 0.7282186686866933,
            },
        },
        "config": vars(args),
        "evaluation_split": args.evaluation_split,
        "fold": int(args.fold),
        "n_fit": int(len(fit_indices)),
        "n_fit_positive": int(labels[fit_indices].sum()),
        "n_heldout": int(len(heldout_indices)),
        "n_heldout_positive": int(labels[heldout_indices].sum()),
        "fit_indices_sha256": _sha256(fit_indices),
        "heldout_indices_sha256": _sha256(heldout_indices),
        "token_dictionary_fit_indices_sha256": expected_fit_sha,
        "candidate_source_positions_sha256": _sha256(candidate_source_positions),
        "candidate_source_graphs_sha256": _sha256(candidate_source_graphs),
        "candidate_rows_sha256": _sha256(candidate_rows),
        "candidate_bank_sha256": _sha256(candidate_bank),
        "target_rows_sha256": _sha256(target_rows),
        "target_graphs_sha256": _sha256(target_graphs),
        "facility_similarity_sha256": facility_similarity_sha256,
        "facility_similarity_audit": facility_similarity_audit,
        "uniform_weights_sha256": _sha256(uniform_weights),
        "scaffold_weights_sha256": _sha256(scaffold_weights),
        "shuffled_scaffold_weights_sha256": _sha256(shuffled_scaffold_weights),
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
