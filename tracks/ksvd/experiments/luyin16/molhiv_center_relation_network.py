"""MolHIV evaluation of the centre-level conditional-fusion prototype.

This is the MolHIV counterpart of the ZINC runner's conditional_fusion mode.
It keeps the information flow fixed:

* one topology-only rooted-WL row per atom centre;
* one chemistry-only OGB atom/bond row for the same centre;
* [s_v, a_v, s_v * a_v] fusion before graph pooling;
* sum/mean/std readout over centres, without relation propagation or attention.

MolHIV is binary classification, so optimization uses BCE-with-logits and
reporting uses ROC-AUC. The official validation split selects the epoch. A
fresh model is then trained on official train+validation for that fixed number
of epochs and evaluated once on official test.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.center_relation_network import (
    CenterRelationNetwork,
    RELATION_WIDTH,
)
from tracks.ksvd.experiments.luyin16.structural_role_fusion_screen import (
    BOND_DIM,
    REPO_ROOT,
    STRICT_ATOM_DIM,
    compact_bond_semantics,
    strict_atom_semantics,
)
from tracks.ksvd.experiments.luyin16.zinc_s_marginal import (
    _ego_distances,
    _rooted_wl_roles,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "tracks/ksvd/configs/luyin16/molhiv_center_relation_network.yaml"
)
NODE_ROLE_BINS = 64
EDGE_ROLE_BINS = 32
TOPOLOGY_WIDTH = NODE_ROLE_BINS * 2 + EDGE_ROLE_BINS + 8
ATTRIBUTE_WIDTH = STRICT_ATOM_DIM * 2 + BOND_DIM * 2


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _one_hot(value: int, width: int) -> np.ndarray:
    output = np.zeros(int(width), dtype=np.float32)
    output[int(value)] = 1.0
    return output


def _normalized_histogram(values: Sequence[int], width: int) -> np.ndarray:
    if not values:
        return np.zeros(int(width), dtype=np.float32)
    output = np.bincount(
        np.asarray(list(values), dtype=np.int64), minlength=int(width)
    ).astype(np.float32)
    return output / float(output.sum())


def _mean_rows(rows: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.size == 0:
        return np.zeros(int(width), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != int(width):
        raise ValueError(f"unexpected row matrix shape {values.shape}; width={width}")
    return values.mean(axis=0, dtype=np.float32)


def _center_features(
    graph: Any,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    representation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned topology and chemistry rows for every atom centre."""
    atom_semantics, _ = strict_atom_semantics(
        np.asarray(node_features, dtype=np.int64)
    )
    bond_semantics = {
        graph.edge_key(int(left), int(right)): compact_bond_semantics(values)
        for (left, right), values in edge_features.items()
    }
    radius = int(representation["radius"])
    wl_rounds = int(representation["wl_rounds"])
    node_role_bins = int(representation["node_role_bins"])
    edge_role_bins = int(representation["edge_role_bins"])
    degree_scale = float(representation.get("degree_scale", 4.0))

    topology_rows: list[np.ndarray] = []
    attribute_rows: list[np.ndarray] = []
    for center in graph.nodes:
        nodes, edges, node_roles, edge_roles = _rooted_wl_roles(
            graph,
            int(center),
            radius,
            typed_wl=False,
            wl_rounds=wl_rounds,
            node_role_bins=node_role_bins,
            edge_role_bins=edge_role_bins,
        )
        shell_distances = _ego_distances(graph, int(center), radius)
        shell = np.asarray(
            [
                float(
                    sum(
                        distance == shell_id
                        for distance in shell_distances.values()
                    )
                )
                / max(float(len(nodes)), 1.0)
                for shell_id in range(radius + 1)
            ],
            dtype=np.float32,
        )
        center_position = nodes.index(int(center))
        local_cycle_rank = max(len(edges) - len(nodes) + 1, 0)
        topology_rows.append(
            np.concatenate(
                [
                    _one_hot(int(node_roles[center_position]), node_role_bins),
                    _normalized_histogram(node_roles.tolist(), node_role_bins),
                    _normalized_histogram(edge_roles.tolist(), edge_role_bins),
                    shell,
                    np.asarray(
                        [
                            float(len(nodes)) / max(float(graph.n), 1.0),
                            float(len(edges)) / max(float(graph.n), 1.0),
                            float(len(graph.neighbors(int(center))))
                            / max(degree_scale, 1.0),
                            float(local_cycle_rank) / max(float(graph.n), 1.0),
                        ],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32, copy=False)
        )

        local_bonds = (
            np.stack(
                [
                    bond_semantics[graph.edge_key(left, right)]
                    for left, right in edges
                ],
                axis=0,
            )
            if edges
            else np.zeros((0, BOND_DIM), dtype=np.float32)
        )
        incident_bonds = [
            bond_semantics[graph.edge_key(int(center), int(neighbor))]
            for neighbor in graph.neighbors(int(center))
        ]
        incident_bond = (
            np.stack(incident_bonds, axis=0)
            if incident_bonds
            else np.zeros((0, BOND_DIM), dtype=np.float32)
        )
        attribute_rows.append(
            np.concatenate(
                [
                    atom_semantics[int(center)],
                    _mean_rows(incident_bond, BOND_DIM),
                    _mean_rows(
                        atom_semantics[np.asarray(nodes, dtype=np.int64)],
                        STRICT_ATOM_DIM,
                    ),
                    _mean_rows(local_bonds, BOND_DIM),
                ]
            ).astype(np.float32, copy=False)
        )

    topology = np.stack(topology_rows, axis=0).astype(np.float32, copy=False)
    attributes = np.stack(attribute_rows, axis=0).astype(np.float32, copy=False)
    expected_topology_width = (
        node_role_bins * 2 + edge_role_bins + (radius + 1) + 4
    )
    if topology.shape[1] != expected_topology_width:
        raise RuntimeError(
            f"topology width changed: {topology.shape}; "
            f"expected {expected_topology_width}"
        )
    if topology.shape[1] != TOPOLOGY_WIDTH:
        raise RuntimeError(
            "configured topology schema is not the current radius-3 schema: "
            f"{topology.shape[1]}"
        )
    if attributes.shape[1] != ATTRIBUTE_WIDTH:
        raise RuntimeError(
            f"attribute width changed: {attributes.shape}; "
            f"expected {ATTRIBUTE_WIDTH}"
        )
    return topology, attributes


def _to_center_data(
    bundle: Any,
    index: int,
    representation: Mapping[str, Any],
) -> tuple[Data, dict[str, Any]]:
    topology, attributes = _center_features(
        bundle.graphs[int(index)],
        bundle.node_feats[int(index)],
        bundle.edge_feats[int(index)],
        representation,
    )
    item = Data(
        node_topology=torch.from_numpy(topology),
        node_attributes=torch.from_numpy(attributes),
        y=torch.tensor([float(bundle.y[int(index)])], dtype=torch.float32),
    )
    item.num_nodes = int(topology.shape[0])
    return item, {
        "n_nodes": int(topology.shape[0]),
        "positive": int(float(bundle.y[int(index)]) > 0.5),
    }


def _cache_signature(
    config_path: Path,
    config: Mapping[str, Any],
    split_indices: Mapping[str, np.ndarray],
) -> str:
    payload = {
        "schema": "molhiv_center_conditional_fusion_v1",
        "config": dict(config["representation"]),
        "splits": {
            name: np.asarray(values, dtype=np.int64).tolist()
            for name, values in split_indices.items()
        },
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_path),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _save_cache(
    path: Path,
    graph_sets: Mapping[str, Sequence[Data]],
    signature: str,
) -> None:
    arrays: dict[str, np.ndarray] = {"signature": np.asarray([signature])}
    for split in ("train", "valid", "test"):
        graphs = list(graph_sets[split])
        arrays[f"{split}_topology"] = np.concatenate(
            [graph.node_topology.numpy() for graph in graphs], axis=0
        ).astype(np.float32, copy=False)
        arrays[f"{split}_attributes"] = np.concatenate(
            [graph.node_attributes.numpy() for graph in graphs], axis=0
        ).astype(np.float32, copy=False)
        arrays[f"{split}_offsets"] = np.asarray(
            [0, *np.cumsum([int(graph.num_nodes) for graph in graphs])],
            dtype=np.int64,
        )
        arrays[f"{split}_labels"] = np.asarray(
            [float(graph.y.view(-1)[0]) for graph in graphs],
            dtype=np.float32,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_cache(path: Path, signature: str) -> dict[str, list[Data]]:
    graph_sets: dict[str, list[Data]] = {}
    with np.load(path, allow_pickle=False) as archive:
        cached_signature = str(np.asarray(archive["signature"]).reshape(-1)[0])
        if cached_signature != signature:
            raise ValueError(f"MolHIV centre cache signature mismatch: {path}")
        for split in ("train", "valid", "test"):
            topology = np.asarray(
                archive[f"{split}_topology"], dtype=np.float32
            )
            attributes = np.asarray(
                archive[f"{split}_attributes"], dtype=np.float32
            )
            offsets = np.asarray(archive[f"{split}_offsets"], dtype=np.int64)
            labels = np.asarray(archive[f"{split}_labels"], dtype=np.float32)
            graphs: list[Data] = []
            for index, label in enumerate(labels):
                start, stop = int(offsets[index]), int(offsets[index + 1])
                item = Data(
                    node_topology=torch.from_numpy(topology[start:stop]),
                    node_attributes=torch.from_numpy(attributes[start:stop]),
                    y=torch.tensor([float(label)], dtype=torch.float32),
                )
                item.num_nodes = stop - start
                graphs.append(item)
            graph_sets[split] = graphs
    return graph_sets


def _build_features(
    bundle: Any,
    split_indices: Mapping[str, np.ndarray],
    representation: Mapping[str, Any],
) -> tuple[dict[str, list[Data]], dict[str, dict[str, float | int]]]:
    graph_sets: dict[str, list[Data]] = {}
    metadata: dict[str, dict[str, float | int]] = {}
    for split in ("train", "valid", "test"):
        indices = np.asarray(split_indices[split], dtype=np.int64)
        values: list[Data] = []
        details: list[dict[str, Any]] = []
        for position, index in enumerate(indices):
            item, row = _to_center_data(bundle, int(index), representation)
            values.append(item)
            details.append(row)
            if (position + 1) % 500 == 0 or position + 1 == len(indices):
                print(
                    f"MolHIV centre features {split}: "
                    f"{position + 1}/{len(indices)}",
                    flush=True,
                )
        graph_sets[split] = values
        metadata[split] = {
            "n_graphs": int(len(values)),
            "mean_centres": float(
                np.mean([row["n_nodes"] for row in details])
            )
            if details
            else 0.0,
            "positive_graphs": int(
                sum(int(row["positive"]) for row in details)
            ),
        }
    return graph_sets, metadata


def _make_loader(
    graphs: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        graphs,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(torch.sigmoid(model(batch)).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float64, copy=False)
    target = np.concatenate(targets).astype(np.float64, copy=False)
    return float(roc_auc_score(target, prediction)), prediction


def _new_model(hidden: int) -> CenterRelationNetwork:
    return CenterRelationNetwork(
        "conditional_fusion",
        TOPOLOGY_WIDTH,
        ATTRIBUTE_WIDTH,
        RELATION_WIDTH,
        int(hidden),
        1,
    )


def _train_to_valid(
    model: CenterRelationNetwork,
    train_graphs: Sequence[Data],
    valid_graphs: Sequence[Data],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    evaluate_every: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_loader = _make_loader(train_graphs, batch_size, True, seed + 91011)
    valid_loader = _make_loader(valid_graphs, batch_size, False, seed + 91012)
    best_auc = -float("inf")
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []
    losses: list[float] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            target = batch.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        train_loss = total_loss / max(seen, 1)
        losses.append(float(train_loss))
        if (
            epoch == 1
            or epoch % max(int(evaluate_every), 1) == 0
            or epoch == int(epochs)
        ):
            valid_auc, _ = _evaluate(model, valid_loader, device)
            history.append(
                {
                    "epoch": int(epoch),
                    "train_bce": float(train_loss),
                    "valid_auc": float(valid_auc),
                }
            )
            if valid_auc > best_auc:
                best_auc = float(valid_auc)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
            if (
                epoch == 1
                or epoch % max(1, int(epochs) // 10) == 0
                or valid_auc == best_auc
            ):
                print(
                    f"molhiv phase=train-to-valid epoch={epoch:03d}/{epochs} "
                    f"bce={train_loss:.6f} valid_auc={valid_auc:.6f}",
                    flush=True,
                )
    if best_state is None:
        raise RuntimeError("validation phase did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_auc, _ = _evaluate(model, valid_loader, device)
    return {
        "epochs_run": int(len(losses)),
        "best_epoch": int(best_epoch),
        "best_valid_auc": float(best_auc),
        "final_valid_auc": float(final_auc),
        "history": history,
        "train_bce": losses,
    }


def _refit_and_test(
    model: CenterRelationNetwork,
    train_graphs: Sequence[Data],
    test_graphs: Sequence[Data],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    _seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_loader = _make_loader(train_graphs, batch_size, True, seed + 91011)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            target = batch.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        if (
            epoch == 1
            or epoch == int(epochs)
            or epoch % max(1, int(epochs) // 10) == 0
        ):
            print(
                f"molhiv phase=train+valid-to-test epoch={epoch:03d}/{epochs} "
                f"bce={total_loss / max(seen, 1):.6f}",
                flush=True,
            )
    test_auc, _ = _evaluate(
        model,
        _make_loader(test_graphs, batch_size, False, seed + 91013),
        device,
    )
    return {"epochs_run": int(epochs), "test_auc": float(test_auc)}


def _render_markdown(result: Mapping[str, Any]) -> str:
    row = result["model"]
    lines = [
        f"# {result['protocol_id']}",
        "",
        "MolHIV evaluation of the centre-level conditional-fusion prototype.",
        "",
        f"- split sizes: {result['data']['sizes']}",
        f"- representation: topology-only rooted-WL radius "
        f"{result['representation']['radius']}, "
        f"{result['representation']['wl_rounds']} WL rounds",
        f"- device: {result['training']['device']}; fixed budget "
        f"{result['training']['epochs']} epochs",
        "- test policy: best official-valid epoch, then train+valid refit; "
        "test was not used for selection",
        "",
        "| phase | ROC-AUC | epoch |",
        "|---|---:|---:|",
        f"| official train -> valid (best) | "
        f"{row['valid']['best_auc']:.6f} | "
        f"{row['valid']['selected_epoch']} |",
        f"| official train+valid -> test | "
        f"{row['test']['auc']:.6f} | "
        f"{row['test']['epochs_run']} |",
        "",
        "The model uses same-centre [s_v, a_v, s_v*a_v] fusion and "
        "sum/mean/std centre readout; relation propagation and attention "
        "are disabled.",
        "",
        f"Runtime: {result['runtime_seconds']:.1f}s.",
        "",
    ]
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    representation = dict(config["representation"])
    model_config = dict(config["model"])
    output_config = config["output"]
    start = time.perf_counter()

    device_name = str(model_config.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    if model_config.get("num_threads") is not None:
        torch.set_num_threads(int(model_config["num_threads"]))

    bundle = load_molhiv(
        root=REPO_ROOT / data_config["root"],
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV OGB node/edge features were not loaded")
    split_indices = {
        name: np.asarray(bundle.split[name], dtype=np.int64)
        for name in ("train", "valid", "test")
    }
    signature = _cache_signature(config_path, config, split_indices)
    cache_path = REPO_ROOT / output_config["feature_cache"]
    if cache_path.exists():
        print(f"loading MolHIV centre feature cache: {cache_path}", flush=True)
        graph_sets = _load_cache(cache_path, signature)
        feature_metadata = {
            split: {
                "n_graphs": int(len(graph_sets[split])),
                "mean_centres": float(
                    np.mean([graph.num_nodes for graph in graph_sets[split]])
                ),
                "positive_graphs": int(
                    sum(
                        float(graph.y.item()) > 0.5
                        for graph in graph_sets[split]
                    )
                ),
                "cache_hit": True,
            }
            for split in ("train", "valid", "test")
        }
    else:
        print("building MolHIV centre feature cache", flush=True)
        graph_sets, feature_metadata = _build_features(
            bundle,
            split_indices,
            representation,
        )
        _save_cache(cache_path, graph_sets, signature)
        print(f"saved MolHIV centre feature cache: {cache_path}", flush=True)

    seed_values = [int(value) for value in model_config.get("model_seeds", [0])]
    if len(seed_values) != 1:
        raise ValueError(
            f"this requested run is single-seed; got model_seeds={seed_values}"
        )
    seed = int(seed_values[0])
    hidden = int(model_config["hidden"])

    _seed_everything(seed)
    valid_model = _new_model(hidden)
    valid_phase = _train_to_valid(
        valid_model,
        graph_sets["train"],
        graph_sets["valid"],
        epochs=int(model_config["epochs"]),
        batch_size=int(model_config["batch_size"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
        seed=seed,
        device=device,
        evaluate_every=int(model_config.get("evaluate_every", 1)),
    )
    selected_epoch = int(valid_phase["best_epoch"])

    _seed_everything(seed)
    refit_model = _new_model(hidden)
    refit_phase = _refit_and_test(
        refit_model,
        graph_sets["train"] + graph_sets["valid"],
        graph_sets["test"],
        epochs=selected_epoch,
        batch_size=int(model_config["batch_size"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
        seed=seed,
        device=device,
    )

    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "data": {
            "dataset": str(data_config["dataset"]),
            "root": str(REPO_ROOT / data_config["root"]),
            "sizes": {
                name: int(values.size)
                for name, values in split_indices.items()
            },
            "positive": {
                name: int(bundle.y[values].sum())
                for name, values in split_indices.items()
            },
            "positive_rate": {
                name: float(bundle.y[values].mean())
                for name, values in split_indices.items()
            },
            "official_split": "OGB ogbg-molhiv scaffold train/valid/test",
            "test_used_for_selection": False,
        },
        "representation": {
            **representation,
            "topology_width": TOPOLOGY_WIDTH,
            "attribute_width": ATTRIBUTE_WIDTH,
            "topology_definition": (
                "untyped rooted-WL root role + topology role histograms + "
                "shell/size/cycle statistics"
            ),
            "attribute_definition": (
                "strict OGB atom semantics (degree/ring omitted) + compact "
                "OGB bond semantics, centre/local/incident means"
            ),
            "fusion": (
                "[structural, attribute, structural * attribute] before "
                "sum/mean/std readout"
            ),
        },
        "feature_build": feature_metadata,
        "training": {
            **model_config,
            "device": str(device),
            "loss": "binary cross entropy with logits",
            "metric": "ROC-AUC",
            "model": "conditional_fusion only",
            "relation_propagation": False,
            "attention": False,
            "parameters": int(
                sum(
                    parameter.numel()
                    for parameter in valid_model.parameters()
                    if parameter.requires_grad
                )
            ),
        },
        "model": {
            "seed": seed,
            "valid": {
                "best_auc": float(valid_phase["best_valid_auc"]),
                "final_auc_at_selected_checkpoint": float(
                    valid_phase["final_valid_auc"]
                ),
                "selected_epoch": selected_epoch,
                "epochs_run": int(valid_phase["epochs_run"]),
                "trace": valid_phase["history"],
            },
            "test": {
                "auc": float(refit_phase["test_auc"]),
                "epochs_run": int(refit_phase["epochs_run"]),
            },
        },
        "runtime_seconds": float(time.perf_counter() - start),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    output_json = REPO_ROOT / output_config["json"]
    output_markdown = REPO_ROOT / output_config["markdown"]
    _write_json_atomic(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path = (
        args.config if args.config.is_absolute() else REPO_ROOT / args.config
    ).resolve()
    result = run(config_path)
    print(
        json.dumps(
            {
                "valid_auc": result["model"]["valid"]["best_auc"],
                "selected_epoch": result["model"]["valid"]["selected_epoch"],
                "test_auc": result["model"]["test"]["auc"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
