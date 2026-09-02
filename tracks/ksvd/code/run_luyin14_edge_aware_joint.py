"""Strict edge-aware joint dictionary and GINE screen."""
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
from torch_geometric.nn import GINEConv, global_add_pool

from .canonical_slots import exact_canonical_order
from .data_tud import load_tud
from .global_stable_ids import compute_global_wl_ids
from .run_luyin14_joint_multiview_dictionary import (
    _attribute_context,
    _encode,
    _fit_dictionary,
    _fit_scaler,
    _joint_vectors,
    _normalize_tokens,
    _shuffle_attribute_context,
)
from .run_luyin14_node_level_fusion import (
    DEFAULT_ROOT,
    _adjacency,
    _component_stable_rank,
    _mlp,
    _paired,
)
from .run_luyin14_route import _atomic_json, _atomic_text


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/edge_aware_joint_stage_a_20260812.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/EDGE_AWARE_JOINT_STAGE_A_20260812.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_LUYIN14_EDGE_AWARE_JOINT_PROTOCOL_20260812.md"
VARIANTS = (
    "GINE_ONLY",
    "TYPED_JOINT_FINAL_TRUE",
    "TYPED_JOINT_FINAL_SHUFFLED",
    "TYPED_JOINT_INIT_TRUE",
)


def _load_edge_pyg(
    name: str, root: Path
) -> tuple[list[Data], np.ndarray, int, int]:
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root=str(root), name=name)
    raw = []
    labels = []
    edge_dim = 0
    for data in dataset:
        if data.x is None or data.edge_attr is None:
            raise RuntimeError(f"{name} requires node and edge attributes")
        edge_dim = int(data.edge_attr.shape[1])
        labels.append(int(data.y.view(-1)[0].item()))
        raw.append(
            Data(
                x=data.x.float(),
                edge_index=data.edge_index,
                edge_attr=data.edge_attr.float(),
                y=data.y.view(1).long(),
                num_nodes=int(data.num_nodes),
            )
        )
    unique = sorted(set(labels))
    remap = {value: index for index, value in enumerate(unique)}
    labels_array = np.asarray([remap[value] for value in labels], dtype=np.int64)
    for index, data in enumerate(raw):
        data.y = torch.tensor([int(labels_array[index])], dtype=torch.long)
    return raw, labels_array, len(unique), edge_dim


def _typed_node_patch_vectors(
    graph, data: Data, *, patch_size: int, edge_dim: int, radius: int = 1
) -> np.ndarray:
    if radius < 1:
        raise ValueError("radius must be positive")
    adjacency = _adjacency(graph)
    typed = np.zeros((graph.n, graph.n), dtype=np.int16)
    edges = data.edge_index.cpu().numpy()
    types = data.edge_attr.argmax(dim=1).cpu().numpy() + 1
    typed[edges[0], edges[1]] = types
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
        if len(candidates) > patch_size:
            stable_rank = _component_stable_rank(adjacency, center)
        else:
            stable_rank = np.arange(graph.n, dtype=np.int64)
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
        local = typed[np.ix_(nodes, nodes)]
        result = exact_canonical_order(
            local,
            nodes,
            color_cells=((center,), others) if others else ((center,),),
        )
        lookup = {node: index for index, node in enumerate(nodes)}
        order = [lookup[node] for node in result.node_ids]
        canonical = local[np.ix_(order, order)]
        vector = []
        for left in range(patch_size):
            for right in range(left + 1, patch_size):
                value = (
                    int(canonical[left, right])
                    if left < len(nodes) and right < len(nodes)
                    else 0
                )
                vector.extend(float(value == edge_type) for edge_type in range(1, edge_dim + 1))
        rows.append(vector)
    return np.asarray(rows, dtype=np.float64)


def _attach_edge(
    raw: Sequence[Data], tokens: Sequence[np.ndarray] | None
) -> list[Data]:
    output = []
    for index, data in enumerate(raw):
        fields: dict[str, Any] = {
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr,
            "y": data.y,
            "num_nodes": data.num_nodes,
        }
        if tokens is not None:
            fields["s"] = torch.tensor(tokens[index], dtype=torch.float32)
        output.append(Data(**fields))
    return output


class EdgeFusionGINE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden: int,
        classes: int,
        *,
        layers: int,
        dropout: float,
        struct_dim: int,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            layer_input = input_dim if layer == 0 else hidden
            self.convs.append(
                GINEConv(_mlp(layer_input, hidden), train_eps=False, edge_dim=edge_dim)
            )
            self.norms.append(nn.BatchNorm1d(hidden))
        self.predictors = nn.ModuleList(
            [nn.Linear(input_dim, classes)]
            + [nn.Linear(hidden, classes) for _ in range(layers)]
        )
        self.film = nn.ModuleList()
        if struct_dim > 0:
            for _ in range(layers):
                projection = nn.Linear(struct_dim, 2 * hidden)
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
                self.film.append(projection)

    def forward(self, data: Data) -> torch.Tensor:
        h = data.x
        representations = [h]
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = F.relu(norm(conv(h, data.edge_index, data.edge_attr)))
            if self.film:
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


@torch.no_grad()
def _evaluate(
    model: EdgeFusionGINE,
    data: Sequence[Data],
    indices: Sequence[int],
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader([data[int(index)] for index in indices], batch_size=128)
    prediction, labels = [], []
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
    edge_dim: int,
    struct_dim: int,
    classes: int,
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

    def make_model() -> EdgeFusionGINE:
        torch.manual_seed(seed)
        return EdgeFusionGINE(
            input_dim,
            edge_dim,
            hidden,
            classes,
            layers=layers,
            dropout=dropout,
            struct_dim=struct_dim,
        ).to(device)

    def make_loader(indices: Sequence[int]) -> DataLoader:
        generator = torch.Generator().manual_seed(seed + 11)
        return DataLoader(
            [data[int(index)] for index in indices],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = make_loader(inner_train)
    best_epoch, best_validation, stale = 1, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
        score = _evaluate(model, data, validation, device=device)["balanced_accuracy"]
        if score > best_validation + 1e-12:
            best_epoch, best_validation, stale = epoch, score, 0
        else:
            stale += 1
        if stale >= patience:
            break

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = make_loader(train_indices)
    for _epoch in range(best_epoch):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
    return {
        **_evaluate(model, data, test_indices, device=device),
        "selected_epoch": int(best_epoch),
        "inner_validation_balanced_accuracy": float(best_validation),
    }


def run_dataset(name: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, node_features, metadata = load_tud(name, args.dataset_root)
    raw, raw_labels, classes, edge_dim = _load_edge_pyg(name, args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("Graph and PyG loaders disagree on labels")
    structures = [
        _typed_node_patch_vectors(
            graph,
            data,
            patch_size=args.patch_size,
            edge_dim=edge_dim,
            radius=args.radius,
        )
        for graph, data in zip(graphs, raw)
    ]
    attributes = [
        _attribute_context(graph, features)
        for graph, features in zip(graphs, node_features)
    ]
    shuffled_attributes = _shuffle_attribute_context(attributes, seed=271828)
    splitter = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.split_seed
    )
    folds = []
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        scaler = _fit_scaler(structures, attributes, train_indices)
        true_vectors = _joint_vectors(structures, attributes, scaler)
        shuffled_vectors = _joint_vectors(structures, shuffled_attributes, scaler)
        true_dictionary = _fit_dictionary(
            true_vectors,
            train_indices,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            iterations=args.iterations,
            max_train_patches=args.max_train_patches,
            seed=args.split_seed * 100 + fold_index,
        )
        shuffled_dictionary = _fit_dictionary(
            shuffled_vectors,
            train_indices,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            iterations=args.iterations,
            max_train_patches=args.max_train_patches,
            seed=args.split_seed * 100 + fold_index,
        )
        true_final = _normalize_tokens(
            [_encode(v, true_dictionary["final"], args.sparsity) for v in true_vectors],
            train_indices,
        )
        true_init = _normalize_tokens(
            [_encode(v, true_dictionary["initial"], args.sparsity) for v in true_vectors],
            train_indices,
        )
        shuffled_final = _normalize_tokens(
            [_encode(v, shuffled_dictionary["final"], args.sparsity) for v in shuffled_vectors],
            train_indices,
        )
        datasets = {
            "GINE_ONLY": _attach_edge(raw, None),
            "TYPED_JOINT_FINAL_TRUE": _attach_edge(raw, true_final),
            "TYPED_JOINT_FINAL_SHUFFLED": _attach_edge(raw, shuffled_final),
            "TYPED_JOINT_INIT_TRUE": _attach_edge(raw, true_init),
        }
        scores = {}
        for variant in VARIANTS:
            data = datasets[variant]
            scores[variant] = _train_selected(
                data,
                labels,
                train_indices,
                test_indices,
                input_dim=int(data[0].x.shape[1]),
                edge_dim=edge_dim,
                struct_dim=0 if variant == "GINE_ONLY" else int(data[0].s.shape[1]),
                classes=classes,
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
                "initial_reconstruction": true_dictionary["initial_reconstruction"],
                "final_reconstruction": true_dictionary["final_reconstruction"],
            }
        )
    paired = {
        "final_minus_gine": _paired(folds, "TYPED_JOINT_FINAL_TRUE", "GINE_ONLY"),
        "true_minus_shuffled": _paired(
            folds, "TYPED_JOINT_FINAL_TRUE", "TYPED_JOINT_FINAL_SHUFFLED"
        ),
        "final_minus_init": _paired(
            folds, "TYPED_JOINT_FINAL_TRUE", "TYPED_JOINT_INIT_TRUE"
        ),
    }
    return {
        "dataset": name,
        "metadata": metadata | {"edge_feat_dim": edge_dim},
        "unique_typed_structure_patches": int(
            len(np.unique(np.concatenate(structures), axis=0))
        ),
        "folds": folds,
        "summary": {
            variant: {
                "balanced_accuracy_mean": float(np.mean([
                    fold["scores"][variant]["balanced_accuracy"] for fold in folds
                ])),
                "selected_epoch_mean": float(np.mean([
                    fold["scores"][variant]["selected_epoch"] for fold in folds
                ])),
            }
            for variant in VARIANTS
        },
        "paired": paired,
        "reconstruction": {
            "initial_mean": float(np.mean([fold["initial_reconstruction"] for fold in folds])),
            "final_mean": float(np.mean([fold["final_reconstruction"] for fold in folds])),
        },
        "seconds": time.time() - started,
    }


def classify(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for result in results:
        pairs = result["paired"]
        reconstruction_improved = (
            result["reconstruction"]["final_mean"]
            < result["reconstruction"]["initial_mean"] - 1e-8
        )
        passed = (
            pairs["final_minus_gine"]["mean"] >= 0.01
            and pairs["final_minus_gine"]["wins"] >= 2
            and pairs["true_minus_shuffled"]["mean"] >= 0.005
            and pairs["true_minus_shuffled"]["wins"] >= 2
            and pairs["final_minus_init"]["mean"] > 0.0
            and pairs["final_minus_init"]["wins"] >= 2
            and reconstruction_improved
        )
        rows[result["dataset"]] = {
            "pass": passed,
            "reconstruction_improved": reconstruction_improved,
            **pairs,
        }
    deltas = [row["final_minus_gine"]["mean"] for row in rows.values()]
    advance = any(row["pass"] for row in rows.values()) and min(deltas) >= -0.01
    return {
        "classification": (
            "EDGE_AWARE_JOINT_ADVANCES"
            if advance else "EDGE_AWARE_JOINT_STAGE_A_NO_GO"
        ),
        "advance": advance,
        "datasets": rows,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# luyin14 edge-aware joint dictionary strict Stage A",
        "",
        "> 日期：2026-08-12  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']['classification']}`",
        "",
        "## 1. 诊断",
        "",
        "| dataset | typed patch types | INIT recon | FINAL recon |",
        "|---|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        lines.append(
            f"| {result['dataset']} | {result['unique_typed_structure_patches']} | "
            f"{result['reconstruction']['initial_mean']:.4f} | "
            f"{result['reconstruction']['final_mean']:.4f} |"
        )
    lines += [
        "",
        "## 2. Balanced accuracy",
        "",
        "| dataset | GINE | FINAL TRUE | FINAL SHUFFLED | INIT TRUE |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        get = lambda variant: result["summary"][variant]["balanced_accuracy_mean"]
        lines.append(
            f"| {result['dataset']} | {get('GINE_ONLY'):.3f} | "
            f"{get('TYPED_JOINT_FINAL_TRUE'):.3f} | "
            f"{get('TYPED_JOINT_FINAL_SHUFFLED'):.3f} | "
            f"{get('TYPED_JOINT_INIT_TRUE'):.3f} |"
        )
    lines += [
        "",
        "## 3. Paired deltas",
        "",
        "| dataset | FINAL-GINE | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        cells = []
        for key in ("final_minus_gine", "true_minus_shuffled", "final_minus_init"):
            pair = result["paired"][key]
            cells += [f"{pair['mean']:+.3f}", f"{pair['wins']}/{pair['ties']}/{pair['losses']}"]
        lines.append(f"| {result['dataset']} | " + " | ".join(cells) + " |")
    lines += ["", "## 4. 结论", ""]
    if payload["decision"]["advance"]:
        lines.append("edge-aware shared code 通过 Stage A；扩展 split seeds 1/2。")
    else:
        lines.append(
            "edge-aware shared code 未通过 Stage A；不在这两个小数据集上继续增加融合容量。"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--datasets", nargs="+", default=["MUTAG", "PTC_MR"])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
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
    results = [run_dataset(dataset, args) for dataset in args.datasets]
    payload = {
        "protocol": args.protocol,
        "config": vars(args) | {
            "dataset_root": str(args.dataset_root),
            "json": str(args.json),
            "report": str(args.report),
        },
        "datasets": results,
    }
    payload["decision"] = classify(results)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render_report(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
