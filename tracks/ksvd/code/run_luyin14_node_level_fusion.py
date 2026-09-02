"""Strict node-level KSVD/attribute fusion on MUTAG and PTC_MR."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from .canonical_slots import exact_canonical_order
from .data_tud import load_tud
from .from_scratch_unplanted_dictionary import deterministic_maximin_initialization
from .global_stable_ids import compute_global_wl_ids
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .run_luyin14_route import (
    DEFAULT_ROOT,
    _adjacency,
    _atomic_json,
    _atomic_text,
    _pad_upper,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/node_level_fusion_stage_a_20260812.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/NODE_LEVEL_FUSION_STAGE_A_20260812.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_LUYIN14_NODE_LEVEL_FUSION_PROTOCOL_20260812.md"
DEFAULT_DATASETS = ("MUTAG", "PTC_MR")
VARIANTS = (
    "GIN_ONLY",
    "FINAL_CONCAT_TRUE",
    "FINAL_FILM_TRUE",
    "FINAL_FILM_SHUFFLED",
    "INIT_FILM_TRUE",
)


def _component_stable_rank(adjacency: np.ndarray, center: int) -> np.ndarray:
    """Stable, label-free ranks inside the component containing ``center``."""
    values = np.asarray(adjacency, dtype=np.int8)
    seen = {int(center)}
    queue = [int(center)]
    for node in queue:
        for raw_neighbor in np.flatnonzero(values[node]):
            neighbor = int(raw_neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    component = tuple(sorted(seen))
    local = values[np.ix_(component, component)]
    stable = compute_global_wl_ids(local)
    rank = np.full(values.shape[0], values.shape[0], dtype=np.int64)
    for order, local_index in enumerate(stable.order):
        rank[component[int(local_index)]] = order
    return rank


def _node_patch_vectors(
    graph, *, patch_size: int, radius: int = 1
) -> np.ndarray:
    if radius < 1:
        raise ValueError("radius must be positive")
    adjacency = _adjacency(graph)
    degree = np.sum(adjacency, axis=1)
    rows = []
    for center in range(graph.n):
        distances = np.full(graph.n, graph.n + 1, dtype=np.int64)
        distances[center] = 0
        queue = [center]
        for node in queue:
            if distances[node] >= radius:
                continue
            for raw_neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(raw_neighbor)
                if distances[neighbor] > distances[node] + 1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        candidates = [node for node in range(graph.n) if distances[node] <= radius]
        stable_rank = (
            _component_stable_rank(adjacency, center)
            if len(candidates) > patch_size
            else np.arange(graph.n, dtype=np.int64)
        )
        neighbors = [node for node in candidates if node != center]
        neighbors.sort(
            key=lambda node: (
                int(distances[node]),
                -int(degree[node]),
                int(stable_rank[node]),
                int(node),
            )
        )
        nodes = (center, *neighbors[: max(patch_size - 1, 0)])
        others = tuple(node for node in nodes if node != center)
        local_adjacency = adjacency[np.ix_(nodes, nodes)]
        result = exact_canonical_order(
            local_adjacency,
            nodes,
            color_cells=((center,), others) if others else ((center,),),
        )
        local = {node: index for index, node in enumerate(nodes)}
        order = [local[node] for node in result.node_ids]
        induced = local_adjacency[np.ix_(order, order)]
        rows.append(_pad_upper(induced, patch_size))
    return np.stack(rows)


def _fit_dictionary(
    node_patches: Sequence[np.ndarray],
    train_indices: Sequence[int],
    *,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    max_train_patches: int,
    seed: int,
) -> dict[str, Any]:
    raw = np.concatenate([node_patches[int(index)] for index in train_indices], axis=0).T
    raw_count = raw.shape[1]
    if raw_count > max_train_patches:
        rng = np.random.default_rng(seed)
        selected = np.sort(
            rng.choice(raw_count, size=max_train_patches, replace=False)
        )
        raw = raw[:, selected]
    mean = raw.mean(axis=1, keepdims=True)
    centered = raw - mean
    nonzero = np.flatnonzero(np.linalg.norm(centered, axis=0) > 1e-12)
    atoms = min(n_atoms, len(nonzero), centered.shape[0])
    if atoms < 2:
        raise RuntimeError("too few nonzero node patches")
    initial, initialization = deterministic_maximin_initialization(
        centered[:, nonzero], atoms
    )
    final, _codes, training = ksvd(
        centered,
        n_atoms=atoms,
        T=min(sparsity, atoms),
        T_min=1,
        n_iter=iterations,
        seed=0,
        initial_dictionary=initial,
    )
    return {
        "mean": mean,
        "initial": initial,
        "final": final,
        "training": training,
        "initialization": initialization,
        "raw_train_patches": int(raw_count),
        "used_train_patches": int(raw.shape[1]),
    }


def _encode_nodes(
    patches: np.ndarray,
    dictionary: np.ndarray,
    mean: np.ndarray,
    sparsity: int,
) -> np.ndarray:
    codes = encode_with_minimum_sparsity(
        patches.T - mean,
        dictionary,
        sparsity=min(sparsity, dictionary.shape[1]),
        minimum_sparsity=1,
    )
    return np.abs(codes.T)


def _normalize_tokens(
    tokens: Sequence[np.ndarray], train_indices: Sequence[int]
) -> list[np.ndarray]:
    train = np.concatenate([tokens[int(index)] for index in train_indices], axis=0)
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    active = scale > 1e-8
    scale = np.where(active, scale, 1.0)
    return [(values - mean) / scale for values in tokens]


def _shuffle_tokens(
    tokens: Sequence[np.ndarray], *, seed: int
) -> list[np.ndarray]:
    output = []
    for graph_index, values in enumerate(tokens):
        rng = np.random.default_rng(seed + graph_index * 1009)
        output.append(values[rng.permutation(len(values))])
    return output


def _load_pyg(name: str, root: Path) -> tuple[list[Data], np.ndarray, int]:
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root=str(root), name=name)
    raw = []
    labels = []
    for data in dataset:
        x = data.x.float() if data.x is not None else torch.ones((data.num_nodes, 1))
        labels.append(int(data.y.view(-1)[0].item()))
        raw.append(
            Data(
                x=x,
                edge_index=data.edge_index,
                y=data.y.view(1).long(),
                num_nodes=int(data.num_nodes),
            )
        )
    unique = sorted(set(labels))
    remap = {value: index for index, value in enumerate(unique)}
    labels_array = np.asarray([remap[value] for value in labels], dtype=np.int64)
    for index, data in enumerate(raw):
        data.y = torch.tensor([int(labels_array[index])], dtype=torch.long)
    return raw, labels_array, len(unique)


def _mlp(input_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
    )


class NodeFusionGIN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        classes: int,
        *,
        layers: int,
        dropout: float,
        mode: str,
        struct_dim: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            layer_input = input_dim if layer == 0 else hidden
            self.convs.append(GINConv(_mlp(layer_input, hidden), train_eps=False))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.predictors = nn.ModuleList(
            [nn.Linear(input_dim, classes)]
            + [nn.Linear(hidden, classes) for _ in range(layers)]
        )
        self.film = nn.ModuleList()
        if mode == "film":
            for _ in range(layers):
                projection = nn.Linear(struct_dim, 2 * hidden)
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
                self.film.append(projection)

    def forward(self, data: Data) -> torch.Tensor:
        h = data.x
        representations = [h]
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = F.relu(norm(conv(h, data.edge_index)))
            if self.mode == "film":
                gamma, beta = self.film[layer](data.s).chunk(2, dim=1)
                h = h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
            representations.append(h)
        score = 0.0
        for predictor, representation in zip(self.predictors, representations):
            pooled = global_add_pool(representation, data.batch)
            score = score + F.dropout(
                predictor(pooled), p=self.dropout, training=self.training
            )
        return score


def _attach(
    raw: Sequence[Data],
    tokens: Sequence[np.ndarray] | None,
    *,
    mode: str,
) -> list[Data]:
    output = []
    for index, data in enumerate(raw):
        fields: dict[str, Any] = {
            "edge_index": data.edge_index,
            "y": data.y,
            "num_nodes": data.num_nodes,
        }
        if mode == "concat":
            token = torch.tensor(tokens[index], dtype=torch.float32)
            fields["x"] = torch.cat([data.x, token], dim=1)
        else:
            fields["x"] = data.x
            if mode == "film":
                fields["s"] = torch.tensor(tokens[index], dtype=torch.float32)
        output.append(Data(**fields))
    return output


@torch.no_grad()
def _evaluate(
    model: NodeFusionGIN,
    data: Sequence[Data],
    indices: Sequence[int],
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader([data[int(index)] for index in indices], batch_size=128)
    prediction = []
    labels = []
    for batch in loader:
        batch = batch.to(device)
        prediction.extend(model(batch).argmax(dim=1).cpu().numpy().tolist())
        labels.extend(batch.y.view(-1).cpu().numpy().tolist())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
    }


def _train_selected(
    data: Sequence[Data],
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    input_dim: int,
    struct_dim: int,
    classes: int,
    mode: str,
    hidden: int,
    layers: int,
    dropout: float,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    inner_train, validation = train_test_split(
        train_indices,
        test_size=0.2,
        stratify=labels[train_indices],
        random_state=1729 + seed,
    )

    def make_model() -> NodeFusionGIN:
        torch.manual_seed(seed)
        return NodeFusionGIN(
            input_dim,
            hidden,
            classes,
            layers=layers,
            dropout=dropout,
            mode=mode,
            struct_dim=struct_dim,
        ).to(device)

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 11)
    loader = DataLoader(
        [data[int(index)] for index in inner_train],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best_epoch = 1
    best_validation = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
        validation_score = _evaluate(
            model, data, validation, device=device
        )["balanced_accuracy"]
        if validation_score > best_validation + 1e-12:
            best_validation = validation_score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 11)
    loader = DataLoader(
        [data[int(index)] for index in train_indices],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    for _epoch in range(best_epoch):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
    score = _evaluate(model, data, test_indices, device=device)
    return {
        **score,
        "selected_epoch": int(best_epoch),
        "inner_validation_balanced_accuracy": float(best_validation),
    }


def _paired(folds: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray(
        [
            fold["scores"][left]["balanced_accuracy"]
            - fold["scores"][right]["balanced_accuracy"]
            for fold in folds
        ]
    )
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
    }


def run_dataset(name: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, node_features, metadata = load_tud(name, args.dataset_root)
    raw, raw_labels, classes = _load_pyg(name, args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("Graph and PyG loaders disagree on labels")
    node_patches = [
        _node_patch_vectors(graph, patch_size=args.patch_size) for graph in graphs
    ]
    folds = []
    splitter = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.split_seed
    )
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        dictionary = _fit_dictionary(
            node_patches,
            train_indices,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            iterations=args.iterations,
            max_train_patches=args.max_train_patches,
            seed=args.split_seed * 100 + fold_index,
        )
        init_tokens = [
            _encode_nodes(
                patches, dictionary["initial"], dictionary["mean"], args.sparsity
            )
            for patches in node_patches
        ]
        final_tokens = [
            _encode_nodes(
                patches, dictionary["final"], dictionary["mean"], args.sparsity
            )
            for patches in node_patches
        ]
        init_tokens = _normalize_tokens(init_tokens, train_indices)
        final_tokens = _normalize_tokens(final_tokens, train_indices)
        final_shuffled = _shuffle_tokens(
            final_tokens, seed=314159 + fold_index
        )
        datasets = {
            "GIN_ONLY": _attach(raw, None, mode="only"),
            "FINAL_CONCAT_TRUE": _attach(raw, final_tokens, mode="concat"),
            "FINAL_FILM_TRUE": _attach(raw, final_tokens, mode="film"),
            "FINAL_FILM_SHUFFLED": _attach(raw, final_shuffled, mode="film"),
            "INIT_FILM_TRUE": _attach(raw, init_tokens, mode="film"),
        }
        scores = {}
        for variant in VARIANTS:
            mode = (
                "only"
                if variant == "GIN_ONLY"
                else "concat"
                if "CONCAT" in variant
                else "film"
            )
            data = datasets[variant]
            scores[variant] = _train_selected(
                data,
                labels,
                train_indices,
                test_indices,
                input_dim=int(data[0].x.shape[1]),
                struct_dim=0 if mode != "film" else int(data[0].s.shape[1]),
                classes=classes,
                mode=mode,
                hidden=args.hidden,
                layers=args.layers,
                dropout=args.dropout,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                seed=args.model_seed * 1000 + fold_index,
                device=torch.device(args.device),
            )
            print(
                f"[{name}] fold={fold_index} {variant} "
                f"bacc={scores[variant]['balanced_accuracy']:.3f} "
                f"epoch={scores[variant]['selected_epoch']}",
                flush=True,
            )
        folds.append(
            {
                "fold_index": fold_index,
                "scores": scores,
                "dictionary_reconstruction": dictionary["training"].get("recon_rel"),
            }
        )
    paired = {
        "film_true_minus_gin": _paired(folds, "FINAL_FILM_TRUE", "GIN_ONLY"),
        "film_true_minus_shuffled": _paired(
            folds, "FINAL_FILM_TRUE", "FINAL_FILM_SHUFFLED"
        ),
        "film_final_minus_init": _paired(
            folds, "FINAL_FILM_TRUE", "INIT_FILM_TRUE"
        ),
        "concat_true_minus_gin": _paired(
            folds, "FINAL_CONCAT_TRUE", "GIN_ONLY"
        ),
    }
    return {
        "dataset": name,
        "metadata": metadata,
        "folds": folds,
        "summary": {
            variant: {
                "balanced_accuracy_mean": float(
                    np.mean([fold["scores"][variant]["balanced_accuracy"] for fold in folds])
                ),
                "accuracy_mean": float(
                    np.mean([fold["scores"][variant]["accuracy"] for fold in folds])
                ),
                "selected_epoch_mean": float(
                    np.mean([fold["scores"][variant]["selected_epoch"] for fold in folds])
                ),
            }
            for variant in VARIANTS
        },
        "paired": paired,
        "seconds": time.time() - started,
    }


def classify(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for result in results:
        pairs = result["paired"]
        pass_dataset = (
            pairs["film_true_minus_gin"]["mean"] >= 0.01
            and pairs["film_true_minus_gin"]["wins"] >= 2
            and pairs["film_true_minus_shuffled"]["mean"] >= 0.005
            and pairs["film_true_minus_shuffled"]["wins"] >= 2
            and pairs["film_final_minus_init"]["mean"] > 0.0
        )
        rows[result["dataset"]] = {"pass": pass_dataset, **pairs}
    deltas = [row["film_true_minus_gin"]["mean"] for row in rows.values()]
    advance = any(row["pass"] for row in rows.values()) and min(deltas) >= -0.01
    return {
        "classification": (
            "NODE_LEVEL_FUSION_ADVANCES_TO_MULTI_SPLIT"
            if advance
            else "NODE_LEVEL_FUSION_STAGE_A_NO_GO"
        ),
        "advance": advance,
        "datasets": rows,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# luyin14 节点级 KSVD–属性融合 strict Stage A",
        "",
        "> 日期：2026-08-12  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']['classification']}`",
        "",
        "## 1. Balanced accuracy",
        "",
        "| dataset | GIN | concat TRUE | FiLM TRUE | FiLM SHUFFLED | FiLM INIT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        get = lambda name: result["summary"][name]["balanced_accuracy_mean"]
        lines.append(
            f"| {result['dataset']} | {get('GIN_ONLY'):.3f} | "
            f"{get('FINAL_CONCAT_TRUE'):.3f} | {get('FINAL_FILM_TRUE'):.3f} | "
            f"{get('FINAL_FILM_SHUFFLED'):.3f} | {get('INIT_FILM_TRUE'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 2. Paired Stage-A deltas",
            "",
            "| dataset | FiLM-GIN | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L | concat-GIN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        pairs = result["paired"]
        cells = []
        for key in (
            "film_true_minus_gin",
            "film_true_minus_shuffled",
            "film_final_minus_init",
        ):
            pair = pairs[key]
            cells.extend(
                [f"{pair['mean']:+.3f}", f"{pair['wins']}/{pair['ties']}/{pair['losses']}"]
            )
        lines.append(
            f"| {result['dataset']} | "
            + " | ".join(cells)
            + f" | {pairs['concat_true_minus_gin']['mean']:+.3f} |"
        )
    lines.extend(["", "## 3. 结论", ""])
    if payload["decision"]["advance"]:
        lines.append(
            "节点级 FiLM 通过 Stage A；按原协议扩展 split seeds 1/2，再决定是否进入更复杂 cross-attention。"
        )
    else:
        lines.append(
            "严格 checkpoint 下节点级 concat/FiLM 未通过。图级融合过粗不是唯一问题；当前局部 adjacency KSVD token 本身仍缺少稳定任务互补性。"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-train-patches", type=int, default=3000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    results = []
    for dataset in args.datasets:
        print(f"[{dataset}] start", flush=True)
        result = run_dataset(dataset, args)
        print(f"[{dataset}] done in {result['seconds']:.1f}s", flush=True)
        results.append(result)
    payload = {
        "protocol": PROTOCOL,
        "config": vars(args)
        | {
            "dataset_root": str(args.dataset_root),
            "json": str(args.json),
            "report": str(args.report),
        },
        "datasets": results,
        "decision": classify(results),
    }
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render_report(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
