"""GNN-free prototype occurrence relation pilot on MolHIV scaffold folds.

Frozen broad-pool farthest and scaffold-facility vocabularies are loaded from the
completed label-free vocabulary pilot.  For each molecular graph, sparse top-k
node-to-prototype assignments Z are composed through exact shortest-path bins:

    C[d] = Z.T @ M[d] @ Z

with bins self/1/2/3+/disconnected.  A low-capacity gated relation residual is
added to the unchanged occurrence-independent PrototypeMIL baseline.  A matched
control deterministically permutes Z over the fixed molecular nodes before the
same distance composition, preserving prototype marginals and graph distances
while destroying their correspondence.  Only official-train scaffold folds are
accepted; official valid/test remain unencoded and unevaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv

DISTANCE_BINS = ("self", "distance_1", "distance_2", "distance_3plus", "disconnected")


@dataclass(frozen=True)
class ControlSpec:
    name: str
    family: str
    relation_mode: str  # none, real, shuffled


def parse_controls(raw: str) -> list[ControlSpec]:
    known = {
        "farthest_occurrence": ControlSpec("farthest_occurrence", "farthest", "none"),
        "farthest_relation": ControlSpec("farthest_relation", "farthest", "real"),
        "farthest_relation_shuffled": ControlSpec("farthest_relation_shuffled", "farthest", "shuffled"),
        "scaffold_occurrence": ControlSpec("scaffold_occurrence", "scaffold_facility", "none"),
        "scaffold_relation": ControlSpec("scaffold_relation", "scaffold_facility", "real"),
        "scaffold_relation_shuffled": ControlSpec("scaffold_relation_shuffled", "scaffold_facility", "shuffled"),
    }
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(names).difference(known))
    if not names or unknown or len(names) != len(set(names)):
        raise ValueError(f"invalid controls={names}, unknown={unknown}")
    return [known[x] for x in names]


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def all_pairs_distance_bins(graph) -> np.ndarray:
    """Return n x n bin ids: 0,1,2,3+,4(disconnected)."""
    n = graph.n
    distances = np.full((n, n), -1, dtype=np.int16)
    for source in range(n):
        distances[source, source] = 0
        queue: deque[int] = deque([source])
        while queue:
            u = queue.popleft()
            for v in graph.neighbors(u):
                if distances[source, v] < 0:
                    distances[source, v] = distances[source, u] + 1
                    queue.append(v)
    bins = np.full((n, n), 4, dtype=np.int8)
    bins[distances == 0] = 0
    bins[distances == 1] = 1
    bins[distances == 2] = 2
    bins[distances >= 3] = 3
    return bins


def sparse_assignment(z: np.ndarray, prototypes: np.ndarray, sparsity: int) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.asarray(z @ prototypes.T, dtype=np.float32)
    positive = np.maximum(cosine, 0.0)
    if sparsity < positive.shape[1]:
        indices = np.argpartition(positive, -sparsity, axis=1)[:, -sparsity:]
        codes = np.zeros_like(positive)
        rows = np.arange(len(positive))[:, None]
        codes[rows, indices] = positive[rows, indices]
    else:
        codes = positive
    mass = codes.sum(axis=1, keepdims=True)
    assignment = np.divide(codes, mass, out=np.zeros_like(codes), where=mass > 1e-8)
    return assignment, cosine


def relation_feature(
    assignment: np.ndarray,
    distance_bins: np.ndarray,
    triangle: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(assignment)
    vectors: list[np.ndarray] = []
    pair_fractions: list[float] = []
    active_pair_mass: list[float] = []
    for bin_id in range(len(DISTANCE_BINS)):
        mask = np.asarray(distance_bins == bin_id, dtype=np.float32)
        count = float(mask.sum())
        if count > 0:
            matrix = assignment.T @ mask @ assignment / count
        else:
            matrix = np.zeros((assignment.shape[1], assignment.shape[1]), dtype=np.float32)
        vectors.append(np.asarray(matrix[triangle], dtype=np.float32))
        pair_fractions.append(count / max(float(n * n), 1.0))
        active_pair_mass.append(float(matrix.sum()))
    feature = np.concatenate([
        *vectors,
        np.asarray(pair_fractions, dtype=np.float32),
        np.asarray(active_pair_mass, dtype=np.float32),
    ])
    return feature.astype(np.float32), np.asarray(pair_fractions), np.asarray(active_pair_mass)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--vocabulary-result", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument(
        "--controls",
        default=(
            "farthest_occurrence,farthest_relation,farthest_relation_shuffled,"
            "scaffold_occurrence,scaffold_relation,scaffold_relation_shuffled"
        ),
    )
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relation-seed", type=int, default=20260728)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--relation-gate-scale", type=float, default=1.0)
    ap.add_argument("--task-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    controls = parse_controls(args.controls)
    if args.fold < 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("invalid fold/training configuration")
    if args.n_prototypes <= 1 or not (0 < args.sparsity <= args.n_prototypes):
        raise ValueError("invalid prototype/sparsity configuration")
    if args.temperature <= 0 or args.relation_gate_scale <= 0:
        raise ValueError("temperature and relation-gate-scale must be positive")

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
    if set(np.concatenate([fit_indices, heldout_indices]).tolist()) != official_train_set:
        raise AssertionError("outer scaffold fold does not partition official train")
    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("outer fit/heldout overlap")

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
        raise AssertionError("official valid/test latent rows are nonzero")

    token_sidecar = Path(args.token_cache).with_suffix(".json")
    latent_sidecar = Path(args.latent_cache).with_suffix(".json")
    if not token_sidecar.exists() or not latent_sidecar.exists():
        raise FileNotFoundError("missing token/latent audit sidecar")
    token_meta = json.loads(token_sidecar.read_text())
    latent_meta = json.loads(latent_sidecar.read_text())
    expected_fit_sha = sha256(fit_indices)
    if token_meta.get("dictionary_fit_indices_sha256") != expected_fit_sha:
        raise ValueError("token dictionary was not fit on this outer-fit fold")
    if bool(token_meta.get("encoded_official_valid", True)):
        raise ValueError("token cache unexpectedly encoded official valid")
    if bool(latent_meta.get("official_valid_encoded", True)) or bool(latent_meta.get("official_test_encoded", True)):
        raise ValueError("latent cache unexpectedly encoded official valid/test")
    if Path(latent_meta.get("config", {}).get("token_cache", "")).name != Path(args.token_cache).name:
        raise ValueError("latent cache was not derived from requested token cache")

    vocabulary_doc = json.loads(Path(args.vocabulary_result).read_text())
    if int(vocabulary_doc.get("fold", -1)) != args.fold:
        raise ValueError("vocabulary result fold mismatch")
    if vocabulary_doc.get("fit_indices_sha256") != expected_fit_sha:
        raise ValueError("vocabulary result fit split mismatch")
    if Path(vocabulary_doc.get("config", {}).get("latent_cache", "")).name != Path(args.latent_cache).name:
        raise ValueError("vocabulary result latent cache mismatch")

    fit_row_mask = np.zeros(len(latents_np), dtype=bool)
    for i in fit_indices:
        fit_row_mask[int(offsets[int(i)]):int(offsets[int(i) + 1])] = True
    prototype_sets: dict[str, np.ndarray] = {}
    prototype_rows: dict[str, np.ndarray] = {}
    for family in sorted({c.family for c in controls}):
        result = vocabulary_doc.get("results", {}).get(family)
        if result is None:
            raise ValueError(f"vocabulary result lacks family={family}")
        rows = np.asarray(result["selection"]["source_node_rows"], dtype=np.int64)
        if rows.shape != (args.n_prototypes,) or not np.all(fit_row_mask[rows]):
            raise ValueError(f"invalid or non-fit prototype rows for family={family}")
        prototypes = normalize_rows(latents_np[rows])
        expected_hash = result["selection"].get("prototype_sha256")
        if expected_hash is not None and sha256(prototypes) != expected_hash:
            raise ValueError(f"prototype hash mismatch for family={family}")
        prototype_rows[family] = rows
        prototype_sets[family] = prototypes

    triangle = np.triu_indices(args.n_prototypes)
    relation_dim = len(DISTANCE_BINS) * len(triangle[0]) + 2 * len(DISTANCE_BINS)
    n_graphs = len(bundle.graphs)
    required_families = sorted(prototype_sets)
    relation_real = {
        family: np.zeros((n_graphs, relation_dim), dtype=np.float32)
        for family in required_families
    }
    relation_shuffled = {
        family: np.zeros((n_graphs, relation_dim), dtype=np.float32)
        for family in required_families
    }
    assignment_audit = {
        family: {"nodes": 0, "zero_mass_nodes": 0, "best_cosine": [], "real_shuffled_l1": []}
        for family in required_families
    }
    distance_pair_fraction_sum = np.zeros(len(DISTANCE_BINS), dtype=np.float64)
    disconnected_graphs = 0
    relation_graphs = np.sort(np.concatenate([fit_indices, heldout_indices]))
    precompute_start = time.time()
    for count_graph, raw_i in enumerate(relation_graphs, start=1):
        i = int(raw_i)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        bins = all_pairs_distance_bins(bundle.graphs[i])
        if np.any(bins == 4):
            disconnected_graphs += 1
        permutation_rng = np.random.default_rng(args.relation_seed + 1_000_003 * i + 104729 * (args.fold + 1))
        permutation = permutation_rng.permutation(hi - lo)
        first_family = True
        for family in required_families:
            assignment, cosine = sparse_assignment(
                latents_np[lo:hi], prototype_sets[family], args.sparsity
            )
            real, pair_fraction, _ = relation_feature(assignment, bins, triangle)
            shuffled, _, _ = relation_feature(assignment[permutation], bins, triangle)
            relation_real[family][i] = real
            relation_shuffled[family][i] = shuffled
            audit = assignment_audit[family]
            audit["nodes"] += int(hi - lo)
            audit["zero_mass_nodes"] += int(np.sum(assignment.sum(axis=1) == 0))
            audit["best_cosine"].append(float(cosine.max(axis=1).mean()))
            audit["real_shuffled_l1"].append(float(np.mean(np.abs(real - shuffled))))
            if first_family:
                distance_pair_fraction_sum += pair_fraction
                first_family = False
        if count_graph % 1000 == 0:
            print(f"relation precompute graphs={count_graph}/{len(relation_graphs)}", flush=True)
    relation_precompute_sec = time.time() - precompute_start
    distance_pair_fraction_mean = distance_pair_fraction_sum / len(relation_graphs)
    assignment_summary = {}
    for family, audit in assignment_audit.items():
        assignment_summary[family] = {
            "n_nodes": int(audit["nodes"]),
            "zero_mass_node_fraction": float(audit["zero_mass_nodes"] / max(audit["nodes"], 1)),
            "mean_graph_best_signed_cosine": float(np.mean(audit["best_cosine"])),
            "mean_real_shuffled_feature_l1": float(np.mean(audit["real_shuffled_l1"])),
            "real_relation_sha256": sha256(relation_real[family][relation_graphs]),
            "shuffled_relation_sha256": sha256(relation_shuffled[family][relation_graphs]),
        }

    atom_dims = [int(x) for x in get_atom_feature_dims()]
    node_features = bundle.node_feats

    class GraphIndexDataset(Dataset):
        def __init__(self, indices: np.ndarray):
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self) -> int:
            return int(len(self.indices))

        def __getitem__(self, item: int) -> int:
            return int(self.indices[item])

    def make_collate(family: str, relation_mode: str):
        relation_source = relation_real[family] if relation_mode != "shuffled" else relation_shuffled[family]

        def collate(graph_indices: list[int]) -> dict[str, Any]:
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
            indices = np.asarray(graph_indices, dtype=np.int64)
            return {
                "indices": torch.from_numpy(indices),
                "rows": torch.from_numpy(np.concatenate(rows)),
                "center": torch.from_numpy(np.concatenate(center_fields, axis=0)),
                "node_graph": torch.from_numpy(np.concatenate(graph_ids)),
                "relation": torch.from_numpy(relation_source[indices]),
                "labels": torch.from_numpy(labels[indices]),
            }

        return collate

    latents = torch.from_numpy(latents_np)

    class PrototypeRelationMIL(nn.Module):
        def __init__(self, prototypes: np.ndarray, use_relation: bool):
            super().__init__()
            p = torch.tensor(prototypes, dtype=torch.float32)
            p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("prototypes", p)
            self.use_relation = use_relation
            self.center_embeddings = nn.ModuleList([
                nn.Embedding(dim, args.hidden) for dim in atom_dims
            ])
            self.identity_embeddings = nn.Parameter(torch.empty(args.n_prototypes, args.hidden))
            self.affinity_embeddings = nn.Parameter(torch.empty(args.n_prototypes, args.hidden))
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
            self.reset_base_parameters()
            # Construct relation modules after resetting the base modules, but do
            # not let their initialization advance the training RNG stream.  The
            # latter is required for the occurrence-only control to reproduce the
            # frozen PrototypeMIL baseline exactly (not merely its parameters),
            # because dropout consumes the global torch RNG during training.
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            self.relation_encoder = nn.Sequential(
                nn.LayerNorm(relation_dim),
                nn.Linear(relation_dim, args.hidden),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, 1),
            )
            self.relation_gate = nn.Parameter(torch.zeros(()))
            for module in self.relation_encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        def reset_base_parameters(self) -> None:
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

        def forward(self, z, center, node_graph, relation):
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
                center_h, identity_h, affinity_h, torch.log1p(mass), top_similarity
            ], dim=1))
            attention = graph_softmax(self.attention(node_h).view(-1), node_graph)
            attention_pool = global_add_pool(node_h * attention[:, None], node_graph)
            mean_pool = global_mean_pool(node_h, node_graph)
            max_pool = global_max_pool(node_h, node_graph)
            base_logits = self.graph_head(torch.cat([
                attention_pool, mean_pool, max_pool
            ], dim=1)).view(-1)
            if self.use_relation:
                relation_logits = self.relation_encoder(relation).view(-1)
                residual = args.relation_gate_scale * torch.tanh(self.relation_gate) * relation_logits
                logits = base_logits + residual
            else:
                # Do not execute the relation encoder for the matched occurrence
                # control: its dropout would advance the RNG stream even though
                # the branch has no effect on the logit.
                residual = torch.zeros_like(base_logits)
                logits = base_logits
            reconstruction = normalized @ self.prototypes
            return logits, codes, cosine, reconstruction, base_logits, residual

    device = torch.device(args.device)

    def make_loader(indices: np.ndarray, shuffle: bool, seed_offset: int, collate_fn):
        generator = torch.Generator().manual_seed(args.seed + seed_offset)
        return DataLoader(
            GraphIndexDataset(indices),
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    def move_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(device) for key, value in batch.items()}

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_scores: list[np.ndarray] = []
        all_base_scores: list[np.ndarray] = []
        all_residuals: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        reconstruction_sum = 0.0
        node_count = 0
        support_sum = 0.0
        mass_sum = 0.0
        best_values: list[np.ndarray] = []
        for batch in loader:
            batch = move_batch(batch)
            z = latents[batch["rows"].cpu()].to(device)
            logits, codes, cosine, reconstruction, base_logits, residual = model(
                z, batch["center"], batch["node_graph"], batch["relation"]
            )
            all_scores.append(logits.cpu().numpy())
            all_base_scores.append(base_logits.cpu().numpy())
            all_residuals.append(residual.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            all_indices.append(batch["indices"].cpu().numpy())
            reconstruction_sum += float(F.mse_loss(reconstruction, z, reduction="sum"))
            node_count += int(z.shape[0])
            support_sum += float((codes > 0).float().sum())
            mass_sum += float(codes.sum())
            best_values.append(cosine.max(dim=1).values.cpu().numpy())
        scores = np.concatenate(all_scores)
        base_scores = np.concatenate(all_base_scores)
        residuals = np.concatenate(all_residuals)
        y = np.concatenate(all_labels)
        graph_indices = np.concatenate(all_indices).astype(np.int64)
        best = np.concatenate(best_values)
        metrics = {
            "auc": float(roc_auc_score(y, scores)),
            "base_auc": float(roc_auc_score(y, base_scores)),
            "n_graphs": int(len(y)),
            "n_positive": int(y.sum()),
            "reconstruction_mse": reconstruction_sum / max(node_count * latents_np.shape[1], 1),
            "mean_active_prototypes": support_sum / max(node_count, 1),
            "mean_occurrence_mass": mass_sum / max(node_count, 1),
            "mean_best_signed_cosine": float(best.mean()),
            "q10_best_signed_cosine": float(np.quantile(best, 0.1)),
            "mean_abs_relation_residual": float(np.mean(np.abs(residuals))),
            "graph_indices_sha256": sha256(graph_indices),
            "score_sha256": sha256(scores.astype(np.float64)),
            "base_score_sha256": sha256(base_scores.astype(np.float64)),
        }
        if args.save_predictions:
            metrics.update({
                "graph_indices": graph_indices.tolist(),
                "labels": y.astype(np.float32).tolist(),
                "scores": scores.astype(np.float32).tolist(),
                "base_scores": base_scores.astype(np.float32).tolist(),
                "relation_residuals": residuals.astype(np.float32).tolist(),
            })
        return metrics

    results: dict[str, Any] = {}
    for control in controls:
        control_start = time.time()
        seed_everything(args.seed, torch)
        collate_fn = make_collate(control.family, control.relation_mode)
        model = PrototypeRelationMIL(
            prototype_sets[control.family], use_relation=control.relation_mode != "none"
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.task_lr, weight_decay=args.weight_decay
        )
        train_loader = make_loader(fit_indices, True, 1000, collate_fn)
        history: list[dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            seen = 0
            for batch in train_loader:
                batch = move_batch(batch)
                z = latents[batch["rows"].cpu()].to(device)
                logits, _, _, _, _, _ = model(
                    z, batch["center"], batch["node_graph"], batch["relation"]
                )
                loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                n = int(batch["labels"].shape[0])
                total += float(loss.detach()) * n
                seen += n
            history.append({
                "epoch": float(epoch),
                "task": total / max(seen, 1),
                "relation_gate": float(torch.tanh(model.relation_gate).detach().cpu()),
            })
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                row = history[-1]
                print(
                    f"{control.name} epoch={epoch:02d} task={row['task']:.5f} "
                    f"gate={row['relation_gate']:+.4f}",
                    flush=True,
                )

        fit_metrics = evaluate(model, make_loader(fit_indices, False, 2000, collate_fn))
        heldout_metrics = evaluate(model, make_loader(heldout_indices, False, 3000, collate_fn))
        results[control.name] = {
            "family": control.family,
            "relation_mode": control.relation_mode,
            "fit": fit_metrics,
            "heldout": heldout_metrics,
            "prototype_rows": prototype_rows[control.family].tolist(),
            "prototype_sha256": sha256(prototype_sets[control.family]),
            "relation_gate_tanh": float(torch.tanh(model.relation_gate).detach().cpu()),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "history": history,
            "elapsed_sec": time.time() - control_start,
        }
        print(
            f"{control.name} heldout_auc={heldout_metrics['auc']:.4f} "
            f"base_auc={heldout_metrics['base_auc']:.4f} "
            f"residual={heldout_metrics['mean_abs_relation_residual']:.5f}",
            flush=True,
        )

    comparisons: dict[str, float] = {}
    for family, prefix in (("farthest", "farthest"), ("scaffold_facility", "scaffold")):
        baseline = f"{prefix}_occurrence"
        real = f"{prefix}_relation"
        shuffled = f"{prefix}_relation_shuffled"
        if real in results and baseline in results:
            comparisons[f"{prefix}_relation_minus_occurrence_auc"] = float(
                results[real]["heldout"]["auc"] - results[baseline]["heldout"]["auc"]
            )
        if real in results and shuffled in results:
            comparisons[f"{prefix}_relation_minus_shuffled_auc"] = float(
                results[real]["heldout"]["auc"] - results[shuffled]["heldout"]["auc"]
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": "molhiv-prototype-exact-distance-relations-scaffold-v1",
        "date": "2026-07-28",
        "hypothesis": (
            "exact shortest-path composition of frozen prototype assignments adds graph-level "
            "signal beyond occurrence-independent MIL and shuffled node-distance correspondence"
        ),
        "architecture": {
            "vocabularies": "frozen broad-pool farthest and scaffold-facility observed PCA64 patches",
            "assignment": f"positive cosine top-{args.sparsity}, per-node mass normalization",
            "relation": "C[d] = Z.T M[d] Z with self/1/2/3+/disconnected bins",
            "relation_readout": "upper-triangle normalized relation matrices plus bin pair/mass scalars",
            "relation_integration": "low-capacity gated scalar residual on unchanged PrototypeMIL logit",
            "shuffled_control": "deterministic node permutation of Z over fixed molecular distance matrices",
            "supervised_message_passing_layers": 0,
            "graph_labels_assigned_to_nodes": False,
        },
        "selection_policy": {
            "data": "official-train only",
            "outer_validation": "one held-out Bemis-Murcko scaffold fold",
            "fixed_epoch": args.epochs,
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "seed0_promotion_rule_across_3_folds": {
                "relation_mean_gain_over_occurrence_at_least": 0.005,
                "minimum_relation_occurrence_fold_wins": 2,
                "relation_must_beat_distance_shuffled_mean": True,
            },
        },
        "config": vars(args),
        "fold": int(args.fold),
        "n_fit": int(len(fit_indices)),
        "n_fit_positive": int(labels[fit_indices].sum()),
        "n_heldout": int(len(heldout_indices)),
        "n_heldout_positive": int(labels[heldout_indices].sum()),
        "fit_indices_sha256": expected_fit_sha,
        "heldout_indices_sha256": sha256(heldout_indices),
        "relation_graph_indices_sha256": sha256(relation_graphs),
        "distance_bins": list(DISTANCE_BINS),
        "distance_pair_fraction_mean": distance_pair_fraction_mean.tolist(),
        "n_disconnected_graphs": int(disconnected_graphs),
        "relation_dim": int(relation_dim),
        "relation_precompute_sec": relation_precompute_sec,
        "assignment_audit": assignment_summary,
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
        "relation_precompute_sec": relation_precompute_sec,
        "elapsed_sec": report["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
