"""A resumable, sufficiently trained GINE baseline for the official ZINC-12K split.

This is intentionally the same matched GINE used by ``zinc_motif_count.py``:
four edge-aware GINE layers, hidden width 64, global-add pooling at every
layer, and an average of the layer-wise graph states.  The only purpose of
this runner is to remove the previous 60-epoch/128-sample training-budget
confound from the comparison with the explicit WL-count representation.

The validation phase selects one best epoch per seed without looking at test.
The frozen test phase refits on official train+valid for that selected number
of epochs.  Each seed is written atomically, so a long CPU run can be resumed
without discarding completed seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error
import yaml

from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _load_zinc,
    _resolve,
    source_audit,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_gine_baseline.yaml"


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
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _make_model(torch: Any, nn: Any, functional: Any, hidden: int, layers: int) -> Any:
    from torch_geometric.nn import GINEConv

    class MatchedGINE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atom_embedding = nn.Embedding(28, hidden)
            self.bond_embedding = nn.Embedding(4, hidden)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.readout_projections = nn.ModuleList()
            for _ in range(int(layers)):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(hidden))
                self.readout_projections.append(nn.Linear(hidden, hidden))
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, data: Any) -> Any:
            from torch_geometric.nn import global_add_pool

            x = self.atom_embedding(data.x.view(-1).long())
            edge_attr = self.bond_embedding(data.edge_attr.view(-1).long())
            states = []
            for conv, batch_norm, projection in zip(
                self.convs, self.bns, self.readout_projections, strict=True
            ):
                x = functional.relu(batch_norm(conv(x, data.edge_index, edge_attr)))
                states.append(projection(global_add_pool(x, data.batch)))
            graph_state = torch.stack(states, dim=0).sum(dim=0) / max(float(len(states)), 1.0)
            return self.head(graph_state).view(-1)

    return MatchedGINE


def _make_data_list(dataset: Any, labels: np.ndarray, torch: Any) -> list[Any]:
    result = []
    for data, label in zip(dataset, labels, strict=True):
        copied = data.clone()
        copied.y = torch.tensor([float(label)], dtype=torch.float32)
        result.append(copied)
    return result


def _make_loader(data: Sequence[Any], batch_size: int, shuffle: bool, seed: int, torch: Any) -> Any:
    from torch_geometric.loader import DataLoader

    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        list(data),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def _evaluate(model: Any, loader: Any, device: Any, torch: Any) -> tuple[float, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu().numpy())
            targets.append(batch.y.view(-1).float().cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float64, copy=False)
    target = np.concatenate(targets).astype(np.float64, copy=False)
    return float(mean_absolute_error(target, prediction)), prediction


def _train_phase(
    *,
    train_data: Sequence[Any],
    eval_data: Sequence[Any],
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    seed: int,
    phase: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden: int,
    layers: int,
    device_name: str,
    torch: Any,
    checkpoint_path: Path | None,
    evaluate_every: int,
    select_best: bool,
    start_epoch: int = 0,
    resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch.nn as nn
    import torch.nn.functional as functional

    _seed_everything(seed, torch)
    device = torch.device(device_name)
    model_class = _make_model(torch, nn, functional, hidden, layers)
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        # Optimizer tensors are created on CPU in the checkpoint; move them to
        # the target device explicitly for the CUDA case.
        for state in optimizer.state.values():
            for key, value in state.items():
                if hasattr(value, "to"):
                    state[key] = value.to(device)

    train_loader = _make_loader(train_data, batch_size, True, seed + 91011, torch)
    eval_loader = _make_loader(eval_data, batch_size, False, seed + 91012, torch)
    losses: list[float] = []
    validation_trace: list[dict[str, float | int]] = []
    best_epoch: int | None = None
    best_mae = float("inf")
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    latest_epoch = int(start_epoch)

    for epoch in range(int(start_epoch) + 1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            prediction = model(batch)
            target = batch.y.view(-1).float()
            loss = functional.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        epoch_loss = total_loss / max(seen, 1)
        losses.append(float(epoch_loss))
        latest_epoch = epoch

        should_evaluate = select_best and (
            epoch == 1 or epoch % max(int(evaluate_every), 1) == 0 or epoch == int(epochs)
        )
        current_mae = None
        if should_evaluate:
            current_mae, _ = _evaluate(model, eval_loader, device, torch)
            validation_trace.append({"epoch": int(epoch), "mae": float(current_mae)})
            if current_mae < best_mae:
                best_mae = float(current_mae)
                best_epoch = int(epoch)
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        if epoch == 1 or epoch % max(1, int(epochs) // 10) == 0 or epoch == int(epochs):
            suffix = "" if current_mae is None else f" valid_mae={current_mae:.6f}"
            print(
                f"gine phase={phase} seed={seed} epoch={epoch:04d}/{epochs} "
                f"l1={epoch_loss:.6f}{suffix}",
                flush=True,
            )
        if checkpoint_path is not None:
            _write_checkpoint(
                checkpoint_path,
                {
                    "phase": phase,
                    "seed": int(seed),
                    "epoch": int(latest_epoch),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "losses": losses,
                    "validation_trace": validation_trace,
                    "best_epoch": best_epoch,
                    "best_mae": best_mae,
                },
                torch,
            )

    if select_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_mae, prediction = _evaluate(model, eval_loader, device, torch)
    return {
        "phase": phase,
        "seed": int(seed),
        "epochs_run": int(latest_epoch),
        "selected_epoch": int(best_epoch if select_best and best_epoch is not None else epochs),
        "best_valid_mae": None if not select_best else float(best_mae),
        "final_eval_mae": float(final_mae),
        "prediction": prediction,
        "losses": losses,
        "validation_trace": validation_trace,
        "model": model,
        "optimizer": optimizer,
        "best_optimizer_state": best_optimizer_state,
    }


def _write_checkpoint(path: Path, payload: Mapping[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _run_seed(
    *,
    seed: int,
    config: Mapping[str, Any],
    train_dataset: Any,
    valid_dataset: Any,
    test_dataset: Any,
    train_y: np.ndarray,
    valid_y: np.ndarray,
    test_y: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    model_config = config["model"]
    epochs = int(model_config["epochs"])
    batch_size = int(model_config["batch_size"])
    hidden = int(model_config["hidden"])
    layers = int(model_config["layers"])
    learning_rate = float(model_config["learning_rate"])
    device_name = str(model_config.get("device", "cpu"))
    evaluate_every = int(model_config.get("evaluate_every", 1))

    train_data = _make_data_list(train_dataset, train_y, torch)
    valid_data = _make_data_list(valid_dataset, valid_y, torch)
    test_data = _make_data_list(test_dataset, test_y, torch)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    phase_checkpoint = checkpoint_dir / f"seed{seed}_valid.pt"

    # The current implementation only resumes whole phase checkpoints at the
    # next invocation.  A completed seed JSON is the stronger resume boundary;
    # a partial checkpoint is retained for future extension but is not loaded
    # silently because its validation best-state metadata may be incomplete.
    valid_phase = _train_phase(
        train_data=train_data,
        eval_data=valid_data,
        train_labels=train_y,
        eval_labels=valid_y,
        seed=seed,
        phase="official-train-to-valid",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden=hidden,
        layers=layers,
        device_name=device_name,
        torch=torch,
        checkpoint_path=phase_checkpoint,
        evaluate_every=evaluate_every,
        select_best=True,
    )
    selected_epoch = int(valid_phase["selected_epoch"])

    combined_data = train_data + valid_data
    combined_y = np.concatenate([train_y, valid_y]).astype(np.float32, copy=False)
    test_phase = _train_phase(
        train_data=combined_data,
        eval_data=test_data,
        train_labels=combined_y,
        eval_labels=test_y,
        seed=seed,
        phase="official-train-plus-valid-to-test",
        epochs=selected_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden=hidden,
        layers=layers,
        device_name=device_name,
        torch=torch,
        checkpoint_path=None,
        evaluate_every=selected_epoch + 1,
        select_best=False,
    )
    return {
        "seed": int(seed),
        "valid": {
            "mae_at_selected_epoch": float(valid_phase["final_eval_mae"]),
            "selected_epoch": selected_epoch,
            "best_valid_mae": float(valid_phase["best_valid_mae"]),
            "validation_trace": valid_phase["validation_trace"],
            "train_l1": valid_phase["losses"],
        },
        "test_after_train_valid_refit": {
            "mae": float(test_phase["final_eval_mae"]),
            "train_l1": test_phase["losses"],
        },
        "training": {
            "hidden": hidden,
            "layers": layers,
            "epochs_budget": epochs,
            "selected_epoch": selected_epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss": "L1 / mean absolute error",
            "readout": "mean of layer-wise global-add states",
            "trainable_parameters": int(sum(p.numel() for p in valid_phase["model"].parameters() if p.requires_grad)),
            "device": device_name,
        },
    }


def _summary(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = np.asarray([float(row["valid"]["mae_at_selected_epoch"]) for row in seed_results])
    test = np.asarray([float(row["test_after_train_valid_refit"]["mae"]) for row in seed_results])
    return {
        "n_seeds": int(len(seed_results)),
        "valid": {
            "mean_mae": float(valid.mean()),
            "std_mae": float(valid.std()),
            "seed_values": valid.tolist(),
        },
        "test_after_train_valid_refit": {
            "mean_mae": float(test.mean()),
            "std_mae": float(test.std()),
            "seed_values": test.tolist(),
        },
        "selected_epochs": [int(row["valid"]["selected_epoch"]) for row in seed_results],
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# ZINC sufficiently trained matched GINE baseline",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "The architecture matches the GINE used in the Step-A report; this run changes only the ZINC training budget to 2000 epochs and batch size 32, with validation-based epoch selection.",
        "",
        "| phase | MAE |",
        "|---|---:|",
        f"| official train → valid (mean ± std) | {summary['valid']['mean_mae']:.6f} ± {summary['valid']['std_mae']:.6f} |",
        f"| official train+valid → test (mean ± std) | {summary['test_after_train_valid_refit']['mean_mae']:.6f} ± {summary['test_after_train_valid_refit']['std_mae']:.6f} |",
        "",
        f"Selected epochs: `{summary['selected_epochs']}`.",
        "",
        "Per-seed results:",
        "",
        "| seed | selected epoch | valid MAE | test MAE |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["seeds"]:
        lines.append(
            f"| {row['seed']} | {row['valid']['selected_epoch']} | "
            f"{row['valid']['mae_at_selected_epoch']:.6f} | "
            f"{row['test_after_train_valid_refit']['mae']:.6f} |"
        )
    lines.extend(["", f"Runtime: `{result['runtime_seconds']:.1f}s`.", ""])
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(config["data"]["root"])
    output_config = config["output"]
    output_dir = _resolve(output_config["directory"])
    result_path = _resolve(output_config["json"])
    markdown_path = _resolve(output_config["markdown"])
    start = time.perf_counter()

    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    model_seeds = [int(value) for value in config["model"]["model_seeds"]]
    seed_results: list[dict[str, Any]] = []
    seed_json_dir = output_dir / "seeds"
    for seed in model_seeds:
        seed_path = seed_json_dir / f"seed{seed}.json"
        if seed_path.exists():
            seed_results.append(json.loads(seed_path.read_text(encoding="utf-8")))
            print(f"resume: found completed seed={seed} at {seed_path}", flush=True)
            continue
        print(f"starting seed={seed}", flush=True)
        result = _run_seed(
            seed=seed,
            config=config,
            train_dataset=datasets[0],
            valid_dataset=datasets[1],
            test_dataset=datasets[2],
            train_y=labels[0],
            valid_y=labels[1],
            test_y=labels[2],
            output_dir=output_dir,
        )
        _write_json_atomic(seed_path, result)
        seed_results.append(result)
        print(
            f"completed seed={seed} valid={result['valid']['mae_at_selected_epoch']:.6f} "
            f"test={result['test_after_train_valid_refit']['mae']:.6f} "
            f"epoch={result['valid']['selected_epoch']}",
            flush=True,
        )

    seed_results.sort(key=lambda row: int(row["seed"]))
    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {name: len(dataset) for name, dataset in zip(("train", "valid", "test"), datasets, strict=True)},
            "source": source_audit(data_root),
            "target": {
                name: {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
                for name, values in zip(("train", "valid", "test"), labels, strict=True)
            },
        },
        "architecture": {
            "atom_embedding_categories": 28,
            "bond_embedding_categories": 4,
            "layers": int(config["model"]["layers"]),
            "hidden": int(config["model"]["hidden"]),
            "readout": "mean of layer-wise global-add states",
            "head": "ReLU + linear",
        },
        "protocol": {
            "train_budget": dict(config["model"]),
            "selection": "best official-valid MAE per seed; test evaluated once after train+valid refit",
            "test_labels_used_for_tuning": False,
        },
        "seeds": seed_results,
        "summary": _summary(seed_results),
        "runtime_seconds": float(time.perf_counter() - start),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _write_json_atomic(result_path, result)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
