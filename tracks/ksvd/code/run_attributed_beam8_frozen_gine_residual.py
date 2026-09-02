"""Frozen-GINE residual calibration using orbit-safe localized Beam8 INIT features."""
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
from torch_geometric.nn import global_add_pool

from .data_tud import load_tud
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_classification import _normalize_nodes, _orbit_shuffle
from .run_attributed_beam8_node_incidence_feasibility import node_incidence_features, orbit_safe_features
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_beam8_nci1_chain_classification import DEFAULT_ROOT, ROOT, _atomic_json, _atomic_text, _encode, _fit_dictionary
from .run_luyin14_edge_aware_joint import EdgeFusionGINE, _attach_edge, _load_edge_pyg


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_GINE_RESIDUAL_PROTOCOL_20260814.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/attributed_beam8_frozen_gine_residual_20260814.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_FROZEN_GINE_RESIDUAL_20260814.md"
VARIANTS = ("GINE_FROZEN", "TRUE_RESIDUAL", "SHUFFLED_RESIDUAL", "BAG_RESIDUAL")


def _model(seed: int, input_dim: int, edge_dim: int, classes: int, device: torch.device) -> EdgeFusionGINE:
    torch.manual_seed(seed)
    return EdgeFusionGINE(input_dim, edge_dim, 64, classes, layers=3, dropout=0.5, struct_dim=0).to(device)


def _forward_base(model: EdgeFusionGINE, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    h = data.x
    representations = [h]
    for conv, norm in zip(model.convs, model.norms):
        h = F.relu(norm(conv(h, data.edge_index, data.edge_attr)))
        representations.append(h)
    score = 0.0
    for predictor, representation in zip(model.predictors, representations):
        score = score + predictor(global_add_pool(representation, data.batch))
    return score, h


@torch.no_grad()
def _base_score(model: EdgeFusionGINE, data: Sequence[Data], indices: Sequence[int], device: torch.device) -> dict[str, float]:
    model.eval()
    prediction, labels = [], []
    for batch in DataLoader([data[int(i)] for i in indices], batch_size=128):
        batch = batch.to(device)
        logits, _states = _forward_base(model, batch)
        prediction.extend(logits.argmax(1).cpu().numpy().tolist())
        labels.extend(batch.y.view(-1).cpu().numpy().tolist())
    return {"balanced_accuracy": float(balanced_accuracy_score(labels, prediction)), "accuracy": float(accuracy_score(labels, prediction))}


def _train_base(
    data: Sequence[Data], indices: Sequence[int], *, epochs: int, seed: int,
    input_dim: int, edge_dim: int, classes: int, device: torch.device,
) -> EdgeFusionGINE:
    model = _model(seed, input_dim, edge_dim, classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 11)
    loader = DataLoader([data[int(i)] for i in indices], batch_size=64, shuffle=True, generator=generator)
    for _epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
    return model


def _select_base_epoch(
    data: Sequence[Data], train: np.ndarray, validation: np.ndarray, *, seed: int,
    input_dim: int, edge_dim: int, classes: int, device: torch.device,
) -> int:
    model = _model(seed, input_dim, edge_dim, classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 11)
    loader = DataLoader([data[int(i)] for i in train], batch_size=64, shuffle=True, generator=generator)
    best_epoch, best_score, stale = 1, -1.0, 0
    for epoch in range(1, 81):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
        score = _base_score(model, data, validation, device)["balanced_accuracy"]
        if score > best_score + 1e-12:
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        if stale >= 20:
            break
    return best_epoch


class ResidualHead(nn.Module):
    def __init__(self, state_dim: int, struct_dim: int, classes: int, rank: int = 16) -> None:
        super().__init__()
        self.state = nn.Linear(state_dim, rank, bias=False)
        self.struct = nn.Linear(struct_dim, rank, bias=False)
        self.output = nn.Linear(rank, classes, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, states: torch.Tensor, struct: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        interaction = torch.tanh(self.state(states)) * torch.tanh(self.struct(struct))
        return self.output(global_add_pool(interaction, batch))


@torch.no_grad()
def _residual_score(
    base: EdgeFusionGINE, head: ResidualHead, data: Sequence[Data], indices: Sequence[int], device: torch.device,
) -> dict[str, float]:
    base.eval(); head.eval(); prediction, labels = [], []
    for batch in DataLoader([data[int(i)] for i in indices], batch_size=128):
        batch = batch.to(device)
        logits, states = _forward_base(base, batch)
        output = logits + head(states, batch.s, batch.batch)
        prediction.extend(output.argmax(1).cpu().numpy().tolist())
        labels.extend(batch.y.view(-1).cpu().numpy().tolist())
    return {"balanced_accuracy": float(balanced_accuracy_score(labels, prediction)), "accuracy": float(accuracy_score(labels, prediction))}


def _train_head(
    base: EdgeFusionGINE, data: Sequence[Data], indices: Sequence[int], *, epochs: int, seed: int,
    struct_dim: int, classes: int, device: torch.device,
) -> ResidualHead:
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()
    torch.manual_seed(seed + 70000)
    head = ResidualHead(64, struct_dim, classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.003, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 23)
    loader = DataLoader([data[int(i)] for i in indices], batch_size=64, shuffle=True, generator=generator)
    for _epoch in range(epochs):
        head.train()
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                logits, states = _forward_base(base, batch)
            optimizer.zero_grad()
            output = logits + head(states, batch.s, batch.batch)
            loss = F.cross_entropy(output, batch.y.view(-1))
            loss.backward()
            optimizer.step()
    return head


def _select_head_epoch(
    base: EdgeFusionGINE, data: Sequence[Data], train: np.ndarray, validation: np.ndarray, *, seed: int,
    struct_dim: int, classes: int, device: torch.device,
) -> int:
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval(); torch.manual_seed(seed + 70000)
    head = ResidualHead(64, struct_dim, classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.003, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 23)
    loader = DataLoader([data[int(i)] for i in train], batch_size=64, shuffle=True, generator=generator)
    best_epoch, best_score, stale = 1, -1.0, 0
    for epoch in range(1, 61):
        head.train()
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                logits, states = _forward_base(base, batch)
            optimizer.zero_grad()
            loss = F.cross_entropy(logits + head(states, batch.s, batch.batch), batch.y.view(-1))
            loss.backward(); optimizer.step()
        score = _residual_score(base, head, data, validation, device)["balanced_accuracy"]
        if score > best_score + 1e-12:
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        if stale >= 15:
            break
    return best_epoch


def _paired(folds: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray([fold["scores"][left]["balanced_accuracy"] - fold["scores"][right]["balanced_accuracy"] for fold in folds])
    return {"mean": float(values.mean()), "wins": int(np.sum(values > 1e-12)), "ties": int(np.sum(np.abs(values) <= 1e-12)), "losses": int(np.sum(values < -1e-12)), "values": values.tolist()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time(); device = torch.device(args.device)
    graphs, labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, raw_labels, classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    if not np.array_equal(labels, raw_labels): raise RuntimeError("the graph and PyG loaders disagree on labels")
    items, typed_graphs = [], []
    for index, (graph, label, node_features, data) in enumerate(zip(graphs, labels, features, raw)):
        typed = _typed_adjacency(data, edge_dim); typed_graphs.append(typed)
        items.append(prepare_attributed_beam_graph(index, graph, int(label), node_features, typed, edge_dim=edge_dim))
        if (index + 1) % 500 == 0 or index + 1 == len(graphs): print(f"prepare {index + 1}/{len(graphs)}", flush=True)
    base_data = _attach_edge(raw, None); folds = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.split_seed)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        split_fold_seed = args.split_seed * 1000 + fold
        inner_train, validation = train_test_split(
            train,
            test_size=0.2,
            stratify=labels[train],
            random_state=1729 + split_fold_seed,
        )
        dictionary = _fit_dictionary(
            items,
            train,
            n_atoms=24,
            sparsity=3,
            iterations=5,
            max_train_patches=3000,
            seed=split_fold_seed,
        )
        patch_tokens = [np.concatenate([_encode(item, dictionary["initial"], dictionary["mean"], 3), item.node_histograms], axis=1) for item in items]
        true_rows = []
        for item, typed, node_features, tokens, graph in zip(items, typed_graphs, features, patch_tokens, graphs):
            direct, _diagnostic = node_incidence_features(item, graph.n, tokens=tokens, include_relation=False)
            safe, _orbits = orbit_safe_features(direct, typed, np.asarray(node_features)); true_rows.append(safe)
        true_rows = _normalize_nodes(true_rows, train)
        shuffled_rows = [
            _orbit_shuffle(
                values,
                typed,
                np.asarray(node_features),
                seed=161803 + split_fold_seed * 100000 + index * 1009,
            )
            for index, (values, typed, node_features) in enumerate(zip(true_rows, typed_graphs, features))
        ]
        bag_rows = [np.repeat(values.mean(0, keepdims=True), len(values), axis=0) for values in true_rows]
        residual_data = {"TRUE_RESIDUAL": _attach_edge(raw, true_rows), "SHUFFLED_RESIDUAL": _attach_edge(raw, shuffled_rows), "BAG_RESIDUAL": _attach_edge(raw, bag_rows)}
        seed = args.model_seed * 1000 + fold; input_dim = int(raw[0].x.shape[1])
        base_epoch = _select_base_epoch(base_data, inner_train, validation, seed=seed, input_dim=input_dim, edge_dim=edge_dim, classes=classes, device=device)
        inner_base = _train_base(base_data, inner_train, epochs=base_epoch, seed=seed, input_dim=input_dim, edge_dim=edge_dim, classes=classes, device=device)
        full_base = _train_base(base_data, train, epochs=base_epoch, seed=seed, input_dim=input_dim, edge_dim=edge_dim, classes=classes, device=device)
        scores = {"GINE_FROZEN": _base_score(full_base, base_data, test, device)}; selected = {"base": base_epoch}
        print(f"fold={fold} GINE_FROZEN bacc={scores['GINE_FROZEN']['balanced_accuracy']:.4f} base_epoch={base_epoch}", flush=True)
        for variant, data in residual_data.items():
            struct_dim = int(data[0].s.shape[1])
            residual_epoch = _select_head_epoch(inner_base, data, inner_train, validation, seed=seed, struct_dim=struct_dim, classes=classes, device=device)
            head = _train_head(full_base, data, train, epochs=residual_epoch, seed=seed, struct_dim=struct_dim, classes=classes, device=device)
            scores[variant] = _residual_score(full_base, head, data, test, device); selected[variant] = residual_epoch
            print(f"fold={fold} {variant} bacc={scores[variant]['balanced_accuracy']:.4f} residual_epoch={residual_epoch}", flush=True)
        folds.append({"fold_index": fold, "scores": scores, "selected_epochs": selected})
    variants = {variant: {"balanced_accuracy_mean": float(np.mean([fold["scores"][variant]["balanced_accuracy"] for fold in folds])), "balanced_accuracy_std": float(np.std([fold["scores"][variant]["balanced_accuracy"] for fold in folds])), "accuracy_mean": float(np.mean([fold["scores"][variant]["accuracy"] for fold in folds]))} for variant in VARIANTS}
    paired = {"increment": _paired(folds, "TRUE_RESIDUAL", "GINE_FROZEN"), "binding": _paired(folds, "TRUE_RESIDUAL", "SHUFFLED_RESIDUAL"), "localization": _paired(folds, "TRUE_RESIDUAL", "BAG_RESIDUAL")}
    checks = {name: row["mean"] >= 0.005 and row["wins"] >= 2 for name, row in paired.items()}
    decision = "FROZEN_GINE_RESIDUAL_ADVANCES_TO_MULTI_MODEL_SEED" if all(checks.values()) else "FROZEN_GINE_RESIDUAL_BELOW_GATE_STOP_MUTAGENICITY_CLASSIFICATION"
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "config": {"model_seed": args.model_seed, "split_seed": args.split_seed},
        "folds": folds,
        "summary": {"variants": variants, "paired": paired, "checks": checks, "decision": decision},
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]; lines = ["# Attributed Beam8 frozen-GINE residual calibration", "", f"> 协议：`{payload['protocol']}`  ", f"> 判定：`{summary['decision']}`", "", "| variant | balanced accuracy | accuracy |", "|---|---:|---:|"]
    for variant in VARIANTS:
        row = summary["variants"][variant]; lines.append(f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | {row['accuracy_mean']:.4f} |")
    lines.extend(["", "## Paired attribution", "", "| comparison | mean | W/T/L |", "|---|---:|---:|"])
    for name, row in summary["paired"].items(): lines.append(f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items(): lines.append(f"- {name}：`{value}`；")
    lines.extend(["", "## Boundary", "", "- residual training never updates GINE parameters or BatchNorm state.", "- TRUE/SHUFFLED/BAG share the same frozen base model, checkpoint epochs, seeds and residual capacity."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT); parser.add_argument("--model-seed", type=int, default=0); parser.add_argument("--split-seed", type=int, default=0); parser.add_argument("--device", default="cpu"); parser.add_argument("--json", type=Path, default=DEFAULT_JSON); parser.add_argument("--report", type=Path, default=DEFAULT_REPORT); args = parser.parse_args()
    payload = run(args); _atomic_json(args.json, payload); _atomic_text(args.report, render(payload)); print(args.report); return 0


if __name__ == "__main__": raise SystemExit(main())
