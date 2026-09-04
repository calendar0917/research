"""MolHIV MLP controls for the frozen ``S + marginal`` representation.

This experiment separates the downstream learner from the representation:

* ``s_marginal_mlp`` maps the graph-level ``S + marginal + context`` vector
  to a logit with a small MLP;
* ``center_fusion_plus_s_marginal_mlp`` encodes the existing centre-level
  conditional fusion, pools it, and combines that representation with the
  same graph-level vector.

The default loss is balanced BCE-with-logits, using the fitting-scope
``N_negative / N_positive`` as ``pos_weight``.  The centre feature cache is
the current radius-3 conditional-fusion cache.
The graph-level marginal block is the frozen radius-2 cross-centre base used
by the existing ``S + marginal`` MolHIV route.  Standardization statistics
are fitted on the fitting split only.  The official validation split selects
the epoch; test is evaluated after a train+validation refit at that frozen
epoch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.center_relation_network import (
    CenterRelationNetwork,
    _MLP,
)
from tracks.ksvd.experiments.luyin16.molhiv_center_relation_network import (
    ATTRIBUTE_WIDTH,
    REPO_ROOT,
    TOPOLOGY_WIDTH,
    _cache_signature,
    _load_cache,
    _make_loader,
    _seed_everything,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/molhiv_mlp_controls.yaml"

GLOBAL_BLOCKS = {
    "S": 205,
    "marginal": 298,
    "context": 5,
}
GLOBAL_WIDTH = sum(GLOBAL_BLOCKS.values())


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _rows_for_indices(
    available: np.ndarray,
    requested: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    mapping = {
        int(index): position
        for position, index in enumerate(np.asarray(available, dtype=np.int64))
    }
    try:
        return np.asarray(
            [mapping[int(index)] for index in np.asarray(requested, dtype=np.int64)],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"{name} lacks dataset index {exc}") from exc


class FeatureStandardizer:
    """Train-scope standardization for dense graph-level inputs."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise ValueError("invalid standardizer shapes")

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureStandardizer":
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError(f"standardizer expects non-empty 2-D input, got {matrix.shape}")
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
        if not np.isfinite(mean).all():
            raise ValueError("standardizer mean is not finite")
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.size:
            raise ValueError(
                f"standardizer input shape {matrix.shape} does not match width {self.mean.size}"
            )
        output = ((matrix - self.mean) / self.scale).astype(np.float32, copy=False)
        if not np.isfinite(output).all():
            raise ValueError("standardized features are not finite")
        return output


class GlobalMLP(nn.Module):
    """MLP operating only on graph-level S+marginal features."""

    def __init__(self, global_width: int, hidden: int) -> None:
        super().__init__()
        self.encoder = _MLP(int(global_width), int(hidden))
        self.head = nn.Linear(int(hidden), 1)

    def forward(self, data: Data) -> torch.Tensor:
        features = data.global_features
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return self.head(self.encoder(features)).view(-1)


class CenterFusionGlobalMLP(nn.Module):
    """Existing centre fusion augmented with the graph-level base vector."""

    def __init__(self, global_width: int, hidden: int) -> None:
        super().__init__()
        self.center_encoder = CenterRelationNetwork(
            "conditional_fusion",
            TOPOLOGY_WIDTH,
            ATTRIBUTE_WIDTH,
            relation_width=9,
            hidden=int(hidden),
            relation_layers=1,
        )
        self.global_encoder = _MLP(int(global_width), int(hidden))
        self.head = nn.Sequential(
            nn.Linear(4 * int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        centre = self.center_encoder.encode(data)
        features = data.global_features
        if features.ndim == 1:
            features = features.unsqueeze(0)
        global_hidden = self.global_encoder(features)
        return self.head(torch.cat([centre, global_hidden], dim=1)).view(-1)


def _load_center_graphs(
    config: Mapping[str, Any],
    split_indices: Mapping[str, np.ndarray],
) -> dict[str, list[Data]]:
    center_config_path = _resolve(config["data"]["center_feature_config"])
    center_config = yaml.safe_load(center_config_path.read_text(encoding="utf-8"))
    center_cache_path = _resolve(config["data"]["center_feature_cache"])
    if not center_cache_path.exists():
        raise FileNotFoundError(f"required centre feature cache is missing: {center_cache_path}")
    signature = _cache_signature(center_config_path, center_config, split_indices)
    print(f"loading MolHIV centre feature cache: {center_cache_path}", flush=True)
    return _load_cache(center_cache_path, signature)


def _load_global_matrix(
    config: Mapping[str, Any],
    split_indices: Mapping[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assemble the historical S+marginal base indexed by original graph id."""
    data_config = config["data"]
    train = np.asarray(split_indices["train"], dtype=np.int64)
    valid = np.asarray(split_indices["valid"], dtype=np.int64)
    test = np.asarray(split_indices["test"], dtype=np.int64)
    dev = np.concatenate([train, valid]).astype(np.int64, copy=False)

    frozen_path = _resolve(data_config["global_frozen_features"])
    dev_path = _resolve(data_config["global_dev_interaction_cache"])
    test_path = _resolve(data_config["global_test_interaction_cache"])
    full_path = _resolve(data_config["global_full_features"])

    with np.load(frozen_path, allow_pickle=False) as archive:
        frozen_indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
        frozen_s = np.asarray(archive["s"], dtype=np.float32)
    with np.load(dev_path, allow_pickle=False) as archive:
        dev_indices_cache = np.asarray(archive["dataset_indices"], dtype=np.int64)
        dev_labels = np.asarray(archive["labels"], dtype=np.int64)
        dev_marginal = np.asarray(archive["marginal"], dtype=np.float32)
        dev_context = np.asarray(archive["context"], dtype=np.float32)
    with np.load(test_path, allow_pickle=False) as archive:
        test_indices_cache = np.asarray(archive["dataset_indices"], dtype=np.int64)
        test_labels = np.asarray(archive["labels"], dtype=np.int64)
        test_marginal = np.asarray(archive["marginal"], dtype=np.float32)
        test_context = np.asarray(archive["context"], dtype=np.float32)
    with np.load(full_path, allow_pickle=False) as archive:
        composition = np.asarray(archive["composition"], dtype=np.float32)

    if not np.array_equal(np.sort(dev_indices_cache), np.sort(dev)):
        raise RuntimeError("global dev interaction cache does not cover train+valid exactly")
    if not np.array_equal(np.sort(test_indices_cache), np.sort(test)):
        raise RuntimeError("global test interaction cache does not cover test exactly")
    if not np.array_equal(dev_labels, labels[dev_indices_cache].astype(np.int64)):
        raise RuntimeError("global dev cache labels do not match MolHIV labels")
    if not np.array_equal(test_labels, labels[test_indices_cache].astype(np.int64)):
        raise RuntimeError("global test cache labels do not match MolHIV labels")
    if frozen_s.shape[1] != GLOBAL_BLOCKS["S"]:
        raise RuntimeError(f"unexpected frozen S width: {frozen_s.shape}")
    if dev_marginal.shape[1] != GLOBAL_BLOCKS["marginal"]:
        raise RuntimeError(f"unexpected marginal width: {dev_marginal.shape}")
    if dev_context.shape[1] != GLOBAL_BLOCKS["context"]:
        raise RuntimeError(f"unexpected context width: {dev_context.shape}")
    if composition.shape[1] != GLOBAL_BLOCKS["S"]:
        raise RuntimeError(f"unexpected full composition width: {composition.shape}")

    dev_s = frozen_s[
        _rows_for_indices(frozen_indices, dev, name="frozen S")
    ]
    dev_marginal_ordered = dev_marginal[
        _rows_for_indices(dev_indices_cache, dev, name="global dev marginal")
    ]
    dev_context_ordered = dev_context[
        _rows_for_indices(dev_indices_cache, dev, name="global dev context")
    ]
    test_marginal_ordered = test_marginal[
        _rows_for_indices(test_indices_cache, test, name="global test marginal")
    ]
    test_context_ordered = test_context[
        _rows_for_indices(test_indices_cache, test, name="global test context")
    ]
    dev_values = np.concatenate(
        [dev_s, dev_marginal_ordered, dev_context_ordered], axis=1
    ).astype(np.float32, copy=False)
    test_values = np.concatenate(
        [composition[test], test_marginal_ordered, test_context_ordered], axis=1
    ).astype(np.float32, copy=False)

    max_index = int(max(np.max(dev), np.max(test)))
    full_values = np.full((max_index + 1, GLOBAL_WIDTH), np.nan, dtype=np.float32)
    full_values[dev] = dev_values
    full_values[test] = test_values
    if not np.isfinite(full_values).all():
        raise RuntimeError("global feature assembly left missing or non-finite rows")

    # The frozen train+valid S and the full composition file should be the
    # same label-free feature block.  This catches accidental cache mixing.
    s_composition_dev = composition[dev]
    max_s_drift = float(np.max(np.abs(s_composition_dev - dev_s)))
    if max_s_drift > 1.0e-6:
        raise RuntimeError(f"frozen S differs from full composition by {max_s_drift}")

    return full_values, {
        "width": int(full_values.shape[1]),
        "blocks": dict(GLOBAL_BLOCKS),
        "definition": "S_v1 + radius-2 centre marginal mean/std + five graph context values",
        "dev_rows": int(dev.size),
        "test_rows": int(test.size),
        "sources": {
            "frozen_S": str(frozen_path),
            "dev_marginal": str(dev_path),
            "test_marginal": str(test_path),
            "full_S": str(full_path),
        },
        "max_dev_S_drift_vs_full_composition": max_s_drift,
    }


def _set_global_features(
    graph_sets: Mapping[str, Sequence[Data]],
    split_indices: Mapping[str, np.ndarray],
    full_features: np.ndarray,
) -> None:
    for split in ("train", "valid", "test"):
        graphs = graph_sets[split]
        indices = np.asarray(split_indices[split], dtype=np.int64)
        if len(graphs) != indices.size:
            raise ValueError(f"centre/global row count mismatch for {split}")
        for graph, index in zip(graphs, indices, strict=True):
            graph.global_features = torch.from_numpy(
                np.asarray(full_features[int(index)], dtype=np.float32).reshape(1, -1)
            )


def _evaluate(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> float:
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
    return float(roc_auc_score(target, prediction))


def _positive_weight(labels: np.ndarray) -> float:
    """Return the fitting-scope positive-class weight for balanced BCE."""
    values = np.asarray(labels, dtype=np.float32).reshape(-1)
    positives = int(np.sum(values > 0.5))
    negatives = int(values.size - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError(
            "balanced BCE needs both classes; "
            f"positives={positives}, negatives={negatives}"
        )
    return float(negatives / positives)


def _graph_labels(graphs: Sequence[Data]) -> np.ndarray:
    return np.asarray(
        [float(graph.y.view(-1)[0]) for graph in graphs], dtype=np.float32
    )


def _train_to_valid(
    model: nn.Module,
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
    pos_weight: float,
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
    weight = torch.as_tensor(float(pos_weight), dtype=torch.float32, device=device)
    best_auc = -float("inf")
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(
                model(batch), target, pos_weight=weight
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        train_bce = total_loss / max(seen, 1)
        if (
            epoch == 1
            or epoch % max(int(evaluate_every), 1) == 0
            or epoch == int(epochs)
        ):
            valid_auc = _evaluate(model, valid_loader, device)
            history.append(
                {
                    "epoch": int(epoch),
                    "train_bce": float(train_bce),
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
                    f"mlp phase=train-to-valid epoch={epoch:03d}/{epochs} "
                    f"bce={train_bce:.6f} valid_auc={valid_auc:.6f}",
                    flush=True,
                )
    if best_state is None:
        raise RuntimeError("validation phase did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_auc = _evaluate(model, valid_loader, device)
    return {
        "epochs_run": int(epochs),
        "best_epoch": int(best_epoch),
        "best_valid_auc": float(best_auc),
        "final_valid_auc": float(final_auc),
        "history": history,
    }


def _refit_and_test(
    model: nn.Module,
    train_graphs: Sequence[Data],
    test_graphs: Sequence[Data],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    pos_weight: float,
) -> dict[str, Any]:
    _seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_loader = _make_loader(train_graphs, batch_size, True, seed + 91011)
    weight = torch.as_tensor(float(pos_weight), dtype=torch.float32, device=device)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(
                model(batch), target, pos_weight=weight
            )
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
                f"mlp phase=train+valid-to-test epoch={epoch:03d}/{epochs} "
                f"bce={total_loss / max(seen, 1):.6f}",
                flush=True,
            )
    test_auc = _evaluate(
        model,
        _make_loader(test_graphs, batch_size, False, seed + 91013),
        device,
    )
    return {"epochs_run": int(epochs), "test_auc": float(test_auc)}


def _build_model(name: str, hidden: int) -> nn.Module:
    if name == "s_marginal_mlp":
        return GlobalMLP(GLOBAL_WIDTH, hidden)
    if name == "center_fusion_plus_s_marginal_mlp":
        return CenterFusionGlobalMLP(GLOBAL_WIDTH, hidden)
    raise ValueError(f"unknown model {name!r}")


def _run_one_model(
    name: str,
    *,
    center_graphs: Mapping[str, Sequence[Data]],
    split_indices: Mapping[str, np.ndarray],
    raw_global: np.ndarray,
    model_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    train = np.asarray(split_indices["train"], dtype=np.int64)
    train_graphs = center_graphs["train"]
    valid_graphs = center_graphs["valid"]
    refit_graphs = list(train_graphs) + list(valid_graphs)
    dev = np.concatenate(
        [train, np.asarray(split_indices["valid"], dtype=np.int64)]
    ).astype(np.int64, copy=False)

    train_scaler = FeatureStandardizer.fit(raw_global[train])
    train_scaled = train_scaler.transform(raw_global)
    _set_global_features(center_graphs, split_indices, train_scaled)
    validation_pos_weight = _positive_weight(_graph_labels(train_graphs))

    _seed_everything(seed)
    valid_model = _build_model(name, int(model_config["hidden"]))
    valid_phase = _train_to_valid(
        valid_model,
        center_graphs["train"],
        center_graphs["valid"],
        epochs=int(model_config["epochs"]),
        batch_size=int(model_config["batch_size"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
        seed=seed,
        device=device,
        evaluate_every=int(model_config.get("evaluate_every", 1)),
        pos_weight=validation_pos_weight,
    )
    selected_epoch = int(valid_phase["best_epoch"])

    refit_scaler = FeatureStandardizer.fit(raw_global[dev])
    refit_scaled = refit_scaler.transform(raw_global)
    _set_global_features(center_graphs, split_indices, refit_scaled)
    refit_pos_weight = _positive_weight(_graph_labels(refit_graphs))
    _seed_everything(seed)
    refit_model = _build_model(name, int(model_config["hidden"]))
    refit_phase = _refit_and_test(
        refit_model,
        refit_graphs,
        center_graphs["test"],
        epochs=selected_epoch,
        batch_size=int(model_config["batch_size"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
        seed=seed,
        device=device,
        pos_weight=refit_pos_weight,
    )
    parameters = int(
        sum(parameter.numel() for parameter in valid_model.parameters() if parameter.requires_grad)
    )
    return {
        "name": name,
        "seed": int(seed),
        "parameters": parameters,
        "valid": {
            "best_auc": float(valid_phase["best_valid_auc"]),
            "final_auc_at_selected_checkpoint": float(valid_phase["final_valid_auc"]),
            "selected_epoch": selected_epoch,
            "epochs_run": int(valid_phase["epochs_run"]),
            "trace": valid_phase["history"],
        },
        "test": {
            "auc": float(refit_phase["test_auc"]),
            "epochs_run": int(refit_phase["epochs_run"]),
        },
        "standardization": {
            "fit_for_validation": "official train only",
            "fit_for_test_refit": "official train + official valid",
            "constant_or_near_constant_dimensions": int(
                np.sum(train_scaler.scale == 1.0)
            ),
        },
        "balanced_bce": {
            "definition": (
                "F.binary_cross_entropy_with_logits(..., "
                "pos_weight=N_negative/N_positive)"
            ),
            "validation_fit_train_positive": int(
                np.sum(_graph_labels(train_graphs) > 0.5)
            ),
            "validation_fit_train_negative": int(
                np.sum(_graph_labels(train_graphs) <= 0.5)
            ),
            "validation_pos_weight": validation_pos_weight,
            "test_refit_positive": int(
                np.sum(_graph_labels(refit_graphs) > 0.5)
            ),
            "test_refit_negative": int(
                np.sum(_graph_labels(refit_graphs) <= 0.5)
            ),
            "test_refit_pos_weight": refit_pos_weight,
        },
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"# {result['protocol_id']}",
        "",
        "MolHIV downstream-control experiment for the frozen S+marginal base.",
        "",
        f"- split sizes: {result['data']['sizes']}",
        f"- global input: {result['global_features']['width']}D "
        f"({result['global_features']['definition']})",
        f"- device: {result['training']['device']}; budget: "
        f"{result['training']['epochs']} epochs; seed: {result['training']['model_seeds']}",
        "- loss: balanced BCE-with-logits (`pos_weight=N_negative/N_positive`); "
        "metric: ROC-AUC",
        "- test policy: best official-valid epoch, then train+valid refit; "
        "test was not used for selection",
        "",
        "| model | valid best AUC | selected epoch | test AUC |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["models"].items():
        lines.append(
            f"| `{name}` | {row['valid']['best_auc']:.6f} | "
            f"{row['valid']['selected_epoch']} | {row['test']['auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The second model uses centre-level `[s_v, a_v, s_v*a_v]` fusion "
            "before sum/mean/std pooling, then concatenates the pooled centre "
            "state with the graph-level S+marginal representation.",
            "",
            f"Runtime: {result['runtime_seconds']:.1f}s.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    model_config = config["model"]
    output_config = config["output"]
    start = time.perf_counter()

    device_name = str(model_config.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    if model_config.get("num_threads") is not None:
        torch.set_num_threads(int(model_config["num_threads"]))

    bundle = load_molhiv(root=_resolve(data_config["root"]), with_features=False)
    split_indices = {
        name: np.asarray(bundle.split[name], dtype=np.int64)
        for name in ("train", "valid", "test")
    }
    labels = np.asarray(bundle.y, dtype=np.float64)
    center_graphs = _load_center_graphs(config, split_indices)
    raw_global, global_meta = _load_global_matrix(config, split_indices, labels)

    seed_values = [int(value) for value in model_config.get("model_seeds", [0])]
    if len(seed_values) != 1:
        raise ValueError(f"this requested run is single-seed; got model_seeds={seed_values}")

    model_names = (
        "s_marginal_mlp",
        "center_fusion_plus_s_marginal_mlp",
    )
    models: dict[str, Any] = {}
    for name in model_names:
        print(f"starting model={name} seed={seed_values[0]}", flush=True)
        models[name] = _run_one_model(
            name,
            center_graphs=center_graphs,
            split_indices=split_indices,
            raw_global=raw_global,
            model_config=model_config,
            device=device,
            seed=seed_values[0],
        )

    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "data": {
            "dataset": str(data_config["dataset"]),
            "root": str(_resolve(data_config["root"])),
            "sizes": {name: int(values.size) for name, values in split_indices.items()},
            "positive": {
                name: int(labels[values].sum()) for name, values in split_indices.items()
            },
            "positive_rate": {
                name: float(labels[values].mean()) for name, values in split_indices.items()
            },
            "official_split": "OGB ogbg-molhiv scaffold train/valid/test",
            "test_used_for_selection": False,
        },
        "center_representation": {
            "source_config": str(_resolve(data_config["center_feature_config"])),
            "cache": str(_resolve(data_config["center_feature_cache"])),
            "radius": 3,
            "fusion": "[s_v, a_v, s_v*a_v] before sum/mean/std pooling",
        },
        "global_features": global_meta,
        "training": {
            **dict(model_config),
            "device": str(device),
            "loss": "balanced binary cross entropy with logits",
            "loss_definition": (
                "pos_weight=N_negative/N_positive, computed on each fitting split"
            ),
            "metric": "ROC-AUC",
            "model_seeds": seed_values,
        },
        "models": models,
        "runtime_seconds": float(time.perf_counter() - start),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    output_json = _resolve(output_config["json"])
    output_markdown = _resolve(output_config["markdown"])
    _write_json_atomic(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    result = run(config_path)
    print(
        json.dumps(
            {
                name: {
                    "valid_auc": row["valid"]["best_auc"],
                    "selected_epoch": row["valid"]["selected_epoch"],
                    "test_auc": row["test"]["auc"],
                }
                for name, row in result["models"].items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
