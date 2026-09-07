"""Structured, task-trained patch/relationship model for luyin16.

The model is deliberately smaller and more explicit than the earlier
``unified_typed_relational_patch_encoder`` experiment.  A rooted typed patch
is kept as its complete canonical descriptor and is split into two channels:

* ``T``: topology/canonical incidence coordinates;
* ``A``: atom and bond attribute coordinates.

The channels are encoded separately.  Their only within-centre interaction is
a low-rank Hadamard product ``(U T) * (V A)``.  Centre pairs use the same
construction, separately for topology, attributes and the binding channel,
and are pooled by shortest-path distance.  A single *linear* graph head then
maps invariant moment features to the task output.  Consequently the output
can be decomposed into structure, attribute, binding, geometry, relation and
global-context contributions without adding a second predictor or a residual
model.

WL/nauty certificates are used upstream only to construct a canonical rooted
descriptor in the cached records; no WL colour counts, K-SVD dictionary or
PCA coordinates enter this model.  The same model class is usable for ZINC
regression and MolHIV classification; only the loss/metric and output
activation differ in the training runner.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import yaml

from tracks.ksvd.experiments.luyin16 import unified_typed_relational_patch_encoder as base


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/structured_patch_relational_model_zinc.yaml"
)


@dataclass
class StructuredTransforms:
    structure: base.Standardizer
    attribute: base.Standardizer
    auxiliary: base.Standardizer
    relation: base.Standardizer
    context: base.Standardizer


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _fit_transforms(
    records: Sequence[Any],
    *,
    dataset: str,
    spec: base.DatasetSpec,
) -> StructuredTransforms:
    """Fit all label-free transforms on one protocol split only."""
    structure = base._fit_streaming(
        records,
        lambda record: base._parts_for_record(record, dataset)[0],
        spec.structure_width,
    )
    attribute = base._fit_streaming(
        records,
        lambda record: base._parts_for_record(record, dataset)[1],
        spec.attribute_width,
    )
    auxiliary = base._fit_streaming(
        records,
        lambda record: base._patch_auxiliary(record, dataset),
        4,
    )
    relation = base._fit_streaming(
        records,
        lambda record: record.pair_relation,
        spec.relation_width,
    )
    context = base._fit_standardizer(
        np.stack(
            [np.asarray(record.global_context, dtype=np.float32) for record in records],
            axis=0,
        )
    )
    return StructuredTransforms(
        structure=structure,
        attribute=attribute,
        auxiliary=auxiliary,
        relation=relation,
        context=context,
    )


def _encode_records(
    records: Sequence[Any],
    *,
    dataset: str,
    transforms: StructuredTransforms,
) -> list[Data]:
    encoded: list[Data] = []
    for record in records:
        structure_raw, attribute_raw = base._parts_for_record(record, dataset)
        structure = transforms.structure.transform(structure_raw)
        attribute = transforms.attribute.transform(attribute_raw)
        auxiliary = transforms.auxiliary.transform(
            base._patch_auxiliary(record, dataset)
        )
        context = transforms.context.transform(
            np.asarray(record.global_context, dtype=np.float32)[None, :]
        )
        encoded.append(
            Data(
                patch_structure=torch.from_numpy(structure),
                patch_attribute=torch.from_numpy(attribute),
                patch_auxiliary=torch.from_numpy(auxiliary),
                pair_index=torch.from_numpy(
                    np.asarray(record.pair_index, dtype=np.int64)
                ),
                pair_relation=torch.from_numpy(
                    transforms.relation.transform(
                        np.asarray(record.pair_relation, dtype=np.float32)
                    )
                ),
                pair_bucket=torch.from_numpy(
                    np.asarray(record.pair_bucket, dtype=np.int64)
                ),
                global_context=torch.from_numpy(context),
                y=torch.tensor([float(record.y)], dtype=torch.float32),
                num_nodes=len(record.patches),
            )
        )
    return encoded


class _Encoder(nn.Module):
    """A signed two-layer adapter; the final code has no forced ReLU."""

    def __init__(self, width: int, hidden: int, output: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(width), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(output)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class StructuredPatchRelationalModel(nn.Module):
    """One semantic patch/pair encoder and one additive graph head."""

    def __init__(
        self,
        *,
        spec: base.DatasetSpec,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        model_config = config["model"]
        self.structure_code = int(model_config.get("structure_code_width", 32))
        self.attribute_code = int(model_config.get("attribute_code_width", 32))
        self.binding_code = int(model_config.get("binding_width", 16))
        self.geometry_code = int(model_config.get("geometry_code_width", 8))
        self.pair_structure = int(model_config.get("pair_structure_width", 16))
        self.pair_attribute = int(model_config.get("pair_attribute_width", 16))
        self.pair_binding = int(model_config.get("pair_binding_width", 8))
        self.pair_relation = int(model_config.get("pair_relation_width", 16))
        self.distance_buckets = int(spec.distance_buckets)
        dropout = float(model_config.get("dropout", 0.05))
        adapter_hidden = int(model_config.get("adapter_hidden", 64))

        self.structure_encoder = _Encoder(
            spec.structure_width,
            max(adapter_hidden, self.structure_code),
            self.structure_code,
            dropout,
        )
        self.attribute_encoder = _Encoder(
            spec.attribute_width,
            max(adapter_hidden, self.attribute_code),
            self.attribute_code,
            dropout,
        )
        self.geometry_encoder = _Encoder(4, max(16, self.geometry_code), self.geometry_code, dropout)
        self.binding_structure = nn.Linear(
            self.structure_code, self.binding_code, bias=False
        )
        self.binding_attribute = nn.Linear(
            self.attribute_code, self.binding_code, bias=False
        )

        # The pair object is unordered.  A shared projection on both centres
        # keeps the elementwise product invariant to swapping centre order.
        self.pair_structure_projection = nn.Linear(
            self.structure_code, self.pair_structure, bias=False
        )
        self.pair_attribute_projection = nn.Linear(
            self.attribute_code, self.pair_attribute, bias=False
        )
        self.pair_binding_projection = nn.Linear(
            self.binding_code, self.pair_binding, bias=False
        )
        self.relation_encoder = _Encoder(
            spec.relation_width,
            max(32, self.pair_relation),
            self.pair_relation,
            dropout,
        )

        context_hidden = int(model_config.get("context_code_width", 16))
        self.context_encoder = _Encoder(
            spec.context_width,
            max(32, context_hidden),
            context_hidden,
            dropout,
        )
        self.context_code = context_hidden

        # Every moment block is sum, sum-of-squares, max and log-count.
        self._block_order = (
            "structure",
            "attribute",
            "binding",
            "geometry",
            "pair_structure",
            "pair_attribute",
            "pair_binding",
            "relation",
            "global_context",
        )
        widths = {
            "structure": 3 * self.structure_code + 1,
            "attribute": 3 * self.attribute_code + 1,
            "binding": 3 * self.binding_code + 1,
            "geometry": 3 * self.geometry_code + 1,
            "pair_structure": self.distance_buckets * (3 * self.pair_structure + 1),
            "pair_attribute": self.distance_buckets * (3 * self.pair_attribute + 1),
            "pair_binding": self.distance_buckets * (3 * self.pair_binding + 1),
            "relation": self.distance_buckets * (3 * self.pair_relation + 1),
            "global_context": self.context_code,
        }
        self.block_widths = widths
        self.head_input_width = sum(widths.values())
        self.head = nn.Linear(self.head_input_width, 1)

    @staticmethod
    def _grouped_max(
        values: torch.Tensor,
        groups: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        maximum = torch.full(
            (int(n_groups), values.shape[1]),
            -float("inf"),
            dtype=values.dtype,
            device=values.device,
        )
        if values.numel():
            maximum.scatter_reduce_(
                0,
                groups.view(-1, 1).expand(-1, values.shape[1]),
                values,
                reduce="amax",
                include_self=True,
            )
        return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))

    @classmethod
    def _pool(
        cls,
        values: torch.Tensor,
        groups: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        total = torch.zeros(
            (int(n_groups), values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        if values.numel():
            total.index_add_(0, groups, values)
        squared = torch.zeros_like(total)
        if values.numel():
            squared.index_add_(0, groups, values * values)
        counts = torch.bincount(groups, minlength=int(n_groups)).to(values.dtype)
        maximum = cls._grouped_max(values, groups, n_groups)
        return torch.cat(
            [total, squared, maximum, torch.log1p(counts).unsqueeze(1)], dim=1
        )

    def _pool_pairs(
        self,
        values: torch.Tensor,
        pair_batch: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        for bucket in range(self.distance_buckets):
            mask = pair_bucket == int(bucket)
            blocks.append(
                self._pool(values[mask], pair_batch[mask], int(n_graphs))
            )
        return torch.cat(blocks, dim=1)

    def _readout(self, data: Data) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n_graphs = int(data.y.view(-1).shape[0])
        structure = self.structure_encoder(data.patch_structure)
        attribute = self.attribute_encoder(data.patch_attribute)
        binding = self.binding_structure(structure) * self.binding_attribute(attribute)
        geometry = self.geometry_encoder(data.patch_auxiliary)
        blocks: dict[str, torch.Tensor] = {
            "structure": self._pool(structure, data.batch, n_graphs),
            "attribute": self._pool(attribute, data.batch, n_graphs),
            "binding": self._pool(binding, data.batch, n_graphs),
            "geometry": self._pool(geometry, data.batch, n_graphs),
        }

        source = data.pair_index[0]
        target = data.pair_index[1]
        pair_structure = self.pair_structure_projection(structure[source]) * self.pair_structure_projection(
            structure[target]
        )
        pair_attribute = self.pair_attribute_projection(attribute[source]) * self.pair_attribute_projection(
            attribute[target]
        )
        pair_binding = self.pair_binding_projection(binding[source]) * self.pair_binding_projection(
            binding[target]
        )
        relation = self.relation_encoder(data.pair_relation)
        pair_batch = data.batch[source]
        pair_bucket = data.pair_bucket.long()
        blocks["pair_structure"] = self._pool_pairs(
            pair_structure, pair_batch, pair_bucket, n_graphs
        )
        blocks["pair_attribute"] = self._pool_pairs(
            pair_attribute, pair_batch, pair_bucket, n_graphs
        )
        blocks["pair_binding"] = self._pool_pairs(
            pair_binding, pair_batch, pair_bucket, n_graphs
        )
        blocks["relation"] = self._pool_pairs(
            relation, pair_batch, pair_bucket, n_graphs
        )
        context = data.global_context
        if context.ndim == 1:
            context = context.unsqueeze(0)
        blocks["global_context"] = self.context_encoder(context)
        ordered = {name: blocks[name] for name in self._block_order}
        return torch.cat(list(ordered.values()), dim=1), ordered

    def forward(
        self,
        data: Data,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features, blocks = self._readout(data)
        prediction = self.head(features).view(-1)
        if not return_components:
            return prediction
        # The linear head makes each semantic block's contribution exact.
        offset = 0
        contributions: dict[str, torch.Tensor] = {}
        for name in self._block_order:
            width = int(self.block_widths[name])
            weight = self.head.weight[:, offset : offset + width]
            contributions[name] = (blocks[name] @ weight.t()).view(-1)
            offset += width
        contributions["bias"] = self.head.bias.view(1).expand_as(prediction)
        return prediction, contributions

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> float:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            values = model(batch)
            if isinstance(values, tuple):
                values = values[0]
            predictions.append(
                values.cpu().numpy()
                if task == "zinc"
                else torch.sigmoid(values).cpu().numpy()
            )
            targets.append(batch.y.view(-1).cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    if task == "zinc":
        return float(mean_absolute_error(target, prediction))
    return float(roc_auc_score(target, prediction))


def _train_phase(
    train_data: Sequence[Data],
    eval_data: Sequence[Data],
    *,
    spec: base.DatasetSpec,
    config: Mapping[str, Any],
    seed: int,
    task: str,
    select_best: bool,
    epochs: int,
) -> dict[str, Any]:
    model_config = config["model"]
    _seed_everything(seed)
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but CUDA is unavailable")
    model = StructuredPatchRelationalModel(spec=spec, config=config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(model_config.get("weight_decay", 1.0e-5)),
    )
    batch_size = int(model_config.get("batch_size", 128))
    loader = DataLoader(
        list(train_data),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed) + 91011),
        num_workers=0,
    )
    eval_loader = DataLoader(
        list(eval_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    best_value = float("inf") if task == "zinc" else -float("inf")
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    stale = 0
    patience = int(model_config.get("patience", 12))
    losses: list[float] = []
    trace: list[dict[str, float | int]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            target = batch.y.view(-1)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            loss = (
                F.l1_loss(prediction, target)
                if task == "zinc"
                else F.binary_cross_entropy_with_logits(prediction, target)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * int(target.shape[0])
            count += int(target.shape[0])
        epoch_loss = total / max(count, 1)
        losses.append(float(epoch_loss))
        current = _evaluate(model, eval_loader, device, task) if select_best else None
        if current is not None:
            metric_key = "mae" if task == "zinc" else "auc"
            trace.append({"epoch": int(epoch), "loss": float(epoch_loss), metric_key: float(current)})
            improved = current < best_value if task == "zinc" else current > best_value
            if improved:
                best_value = float(current)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        if epoch == 1 or epoch == int(epochs) or epoch % max(1, int(epochs) // 10) == 0:
            suffix = "" if current is None else f" valid_{'mae' if task == 'zinc' else 'auc'}={current:.6f}"
            print(
                f"structured task={task} phase={'select' if select_best else 'refit'} "
                f"epoch={epoch:03d}/{epochs} loss={epoch_loss:.6f}{suffix}",
                flush=True,
            )
        if select_best and stale >= patience:
            print(
                f"structured early_stop task={task} epoch={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    final_metric = _evaluate(model, eval_loader, device, task)
    return {
        "model": model,
        "metric": float(final_metric),
        "best_metric": None if not select_best else float(best_value),
        "selected_epoch": int(best_epoch if select_best else epochs),
        "epochs_run": int(len(losses)),
        "losses": losses,
        "trace": trace,
        "parameters": model.parameter_count(),
    }


def _load_graphs(config: Mapping[str, Any], dataset: str, split: str):
    return base._load_graphs(config, dataset, split)


def _record_sizes(meta: Mapping[str, Any], records: Sequence[Any]) -> int:
    value = meta.get("n_graphs")
    return int(value) if value is not None else int(len(records))


def _render_markdown(result: Mapping[str, Any]) -> str:
    metric = "MAE" if result["task"] == "zinc" else "ROC-AUC"
    evaluation = result["evaluation"]
    lines = [
        f"# {result['protocol_id']}",
        "",
        "Structured rooted typed patch model: separate topology/attribute encoders, low-rank within-centre binding, distance-conditioned pair products, invariant moments, and one linear graph head.",
        "",
        f"- dataset: `{result['dataset']}`; split: `{result['data']['split']}`",
        f"- patch input: structure `{result['representation']['structure_input_width']}D` + attribute `{result['representation']['attribute_input_width']}D` + geometry `4D`",
        f"- pair input: relation `{result['representation']['relation_input_width']}D`, `{result['representation']['distance_buckets']}` distance buckets",
        f"- readout: `{result['representation']['readout']}`",
        "- WL/K-SVD/PCA upstream statistics: `none` / `none` / `none`",
        "- message passing/attention: `False` / `False`",
        "",
        f"| model | valid {metric} | test {metric} after train+valid refit | selected epoch |",
        "|---|---:|---:|---:|",
        f"| `structured_single_head` | {evaluation['valid']['metric']:.6f} | {evaluation['test_after_train_valid_refit']['metric']:.6f} | {evaluation['valid']['selected_epoch']} |",
        "",
        f"- trainable parameters: `{evaluation['parameters']}`",
        f"- runtime: `{result['runtime']['seconds']:.1f}s`",
        "",
        "The linear head makes the prediction an exact additive decomposition over structure, attribute, within-patch binding, geometry, cross-centre structure/attribute/binding products, relation descriptors and global context.",
        "",
    ]
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = str(config["dataset"])
    if dataset not in {"zinc", "molhiv"}:
        raise ValueError(f"unknown dataset {dataset!r}")
    task = dataset
    seed = int(config.get("seed", 0))
    started = time.perf_counter()
    torch.set_num_threads(max(int(config.get("runtime", {}).get("torch_threads", 4)), 1))

    train_records, train_meta, train_hit = _load_graphs(config, dataset, "train")
    valid_records, valid_meta, valid_hit = _load_graphs(config, dataset, "valid")
    train_size = _record_sizes(train_meta, train_records)
    valid_size = _record_sizes(valid_meta, valid_records)
    spec = base._spec_from_records(dataset, train_records)
    valid_transforms = _fit_transforms(train_records, dataset=dataset, spec=spec)
    valid_train_data = _encode_records(
        train_records, dataset=dataset, transforms=valid_transforms
    )
    valid_eval_data = _encode_records(
        valid_records, dataset=dataset, transforms=valid_transforms
    )
    valid_phase = _train_phase(
        valid_train_data,
        valid_eval_data,
        spec=spec,
        config=config,
        seed=seed,
        task=task,
        select_best=True,
        epochs=int(config["model"].get("epochs", 50)),
    )
    selected_epoch = int(valid_phase["selected_epoch"])
    valid_metric = float(valid_phase["metric"])
    parameters = int(valid_phase["parameters"])
    del valid_phase["model"], valid_train_data, valid_eval_data, valid_transforms
    gc.collect()

    refit_records = [*train_records, *valid_records]
    refit_transforms = _fit_transforms(refit_records, dataset=dataset, spec=spec)
    refit_train_data = _encode_records(
        refit_records, dataset=dataset, transforms=refit_transforms
    )
    del refit_records, train_records, valid_records
    gc.collect()
    test_records, test_meta, test_hit = _load_graphs(config, dataset, "test")
    test_size = _record_sizes(test_meta, test_records)
    refit_test_data = _encode_records(
        test_records, dataset=dataset, transforms=refit_transforms
    )
    refit_phase = _train_phase(
        refit_train_data,
        refit_test_data,
        spec=spec,
        config=config,
        seed=seed,
        task=task,
        select_best=False,
        epochs=selected_epoch,
    )
    test_metric = float(refit_phase["metric"])
    del refit_phase["model"], refit_train_data, refit_test_data
    gc.collect()

    result: dict[str, Any] = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data": {
            "root": str(_resolve(config["data"]["root"])),
            "split": (
                "PyG ZINC subset=True official train/valid/test"
                if dataset == "zinc"
                else "OGB ogbg-molhiv official scaffold train/valid/test"
            ),
            "sizes": {
                "train": train_size,
                "valid": valid_size,
                "test": test_size,
            },
            "record_cache_hits": {
                "train": bool(train_hit),
                "valid": bool(valid_hit),
                "test": bool(test_hit),
            },
            "test_labels_used_for_selection": False,
            "representation_fit_scope": "official train only for valid; official train+valid for terminal test refit",
        },
        "representation": {
            "structure_input_width": int(spec.structure_width),
            "attribute_input_width": int(spec.attribute_width),
            "relation_input_width": int(spec.relation_width),
            "context_input_width": int(spec.context_width),
            "distance_buckets": int(spec.distance_buckets),
            "readout": "separate T/A encoders + low-rank binding; pair products by distance; sum/sumsq/max/log-count",
            "wl_statistics": False,
            "ksvd": False,
            "pca": False,
            "message_passing": False,
            "attention": False,
            "one_predictive_head": True,
            "head_type": "single linear additive head",
        },
        "training": {
            **dict(config["model"]),
            "loss": "L1 / mean absolute error" if task == "zinc" else "binary cross entropy with logits",
            "metric": "MAE" if task == "zinc" else "ROC-AUC",
            "selected_epoch": selected_epoch,
            "one_predictive_head": True,
        },
        "evaluation": {
            "valid": {
                "metric": valid_metric,
                "selected_epoch": selected_epoch,
                "epochs_run": int(valid_phase["epochs_run"]),
                "trace": valid_phase["trace"],
            },
            "test_after_train_valid_refit": {
                "metric": test_metric,
                "epochs_run": selected_epoch,
            },
            "parameters": parameters,
        },
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "torch_geometric": importlib.metadata.version("torch-geometric"),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    output_json = _resolve(config["output"]["json"])
    output_markdown = _resolve(config["output"]["markdown"])
    _write_json_atomic(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    key = "mae" if result["task"] == "zinc" else "auc"
    print(
        json.dumps(
            {
                "valid_" + key: result["evaluation"]["valid"]["metric"],
                "test_" + key: result["evaluation"]["test_after_train_valid_refit"]["metric"],
                "selected_epoch": result["evaluation"]["valid"]["selected_epoch"],
                "parameters": result["evaluation"]["parameters"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
