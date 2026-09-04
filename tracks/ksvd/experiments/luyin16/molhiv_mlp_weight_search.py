"""Search lightweight positive-class weights for the MolHIV MLP controls.

For each of the two downstream models, candidate ``pos_weight`` values are
trained and selected using only official validation.  Only the selected
candidate is then refit on official train+validation and evaluated on test.
This keeps the test split outside the weight search.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.molhiv_mlp_controls import (
    FeatureStandardizer,
    _build_model,
    _graph_labels,
    _load_center_graphs,
    _load_global_matrix,
    _positive_weight,
    _refit_and_test,
    _resolve,
    _seed_everything,
    _set_global_features,
    _sha256,
    _train_to_valid,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/molhiv_mlp_weight_search.yaml"


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


def _validation_candidate(
    name: str,
    pos_weight: float,
    *,
    center_graphs: Mapping[str, Sequence[Any]],
    split_indices: Mapping[str, np.ndarray],
    raw_global: np.ndarray,
    model_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    train_indices = np.asarray(split_indices["train"], dtype=np.int64)
    train_graphs = center_graphs["train"]
    train_scaler = FeatureStandardizer.fit(raw_global[train_indices])
    _set_global_features(
        center_graphs,
        split_indices,
        train_scaler.transform(raw_global),
    )

    _seed_everything(seed)
    model = _build_model(name, int(model_config["hidden"]))
    phase = _train_to_valid(
        model,
        train_graphs,
        center_graphs["valid"],
        epochs=int(model_config["epochs"]),
        batch_size=int(model_config["batch_size"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
        seed=seed,
        device=device,
        evaluate_every=int(model_config.get("evaluate_every", 1)),
        pos_weight=float(pos_weight),
    )
    parameters = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return {
        "pos_weight": float(pos_weight),
        "parameters": parameters,
        "valid": {
            "best_auc": float(phase["best_valid_auc"]),
            "final_auc_at_selected_checkpoint": float(phase["final_valid_auc"]),
            "selected_epoch": int(phase["best_epoch"]),
            "epochs_run": int(phase["epochs_run"]),
            "trace": phase["history"],
        },
    }


def _select_candidate(candidates: Mapping[str, Mapping[str, Any]]) -> tuple[str, Mapping[str, Any]]:
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    return max(
        candidates.items(),
        key=lambda item: (
            float(item[1]["valid"]["best_auc"]),
            -float(item[1]["pos_weight"]),
        ),
    )


def _refit_selected(
    name: str,
    selected: Mapping[str, Any],
    *,
    center_graphs: Mapping[str, Sequence[Any]],
    split_indices: Mapping[str, np.ndarray],
    raw_global: np.ndarray,
    model_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    train_indices = np.asarray(split_indices["train"], dtype=np.int64)
    dev_indices = np.concatenate(
        [train_indices, np.asarray(split_indices["valid"], dtype=np.int64)]
    ).astype(np.int64, copy=False)
    refit_scaler = FeatureStandardizer.fit(raw_global[dev_indices])
    _set_global_features(
        center_graphs,
        split_indices,
        refit_scaler.transform(raw_global),
    )
    _seed_everything(seed)
    model = _build_model(name, int(model_config["hidden"]))
    return _refit_and_test(
        model,
        list(center_graphs["train"]) + list(center_graphs["valid"]),
        center_graphs["test"],
        epochs=int(selected["valid"]["selected_epoch"]),
        batch_size=int(model_config["batch_size"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
        seed=seed,
        device=device,
        pos_weight=float(selected["pos_weight"]),
    )


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"# {result['protocol_id']}",
        "",
        "MolHIV search over lightweight positive-class weights for the MLP controls.",
        "",
        f"- split sizes: {result['data']['sizes']}",
        f"- candidates: `{result['search']['candidate_pos_weights']}`",
        f"- device: {result['training']['device']}; budget: "
        f"{result['training']['epochs']} epochs; seed: {result['training']['model_seeds']}",
        "- loss: BCE-with-logits with the candidate `pos_weight`; metric: ROC-AUC",
        "- selection: highest official-valid AUC per model; test is run only for the selected weight",
        "",
        "| model | weight | valid best AUC | epoch | test AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in result["models"].items():
        selected = row["selected"]
        lines.append(
            f"| `{name}` | {selected['pos_weight']:.1f} | "
            f"{selected['valid']['best_auc']:.6f} | "
            f"{selected['valid']['selected_epoch']} | "
            f"{row['test']['test_auc']:.6f} |"
        )
    lines.extend(["", "## Validation search", ""])
    lines.extend(
        [
            "| model | weight | valid best AUC | epoch |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in result["models"].items():
        for candidate in row["candidates"].values():
            lines.append(
                f"| `{name}` | {candidate['pos_weight']:.1f} | "
                f"{candidate['valid']['best_auc']:.6f} | "
                f"{candidate['valid']['selected_epoch']} |"
            )
    lines.extend(["", f"Runtime: {result['runtime_seconds']:.1f}s.", ""])
    return "\n".join(lines)


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    model_config = config["model"]
    search_config = config["search"]
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

    candidate_weights = [float(value) for value in search_config["pos_weights"]]
    if len(candidate_weights) != 3 or any(value <= 0.0 for value in candidate_weights):
        raise ValueError("this run requires exactly three positive pos_weights")
    if len(set(candidate_weights)) != len(candidate_weights):
        raise ValueError("candidate pos_weights must be distinct")
    seed_values = [int(value) for value in model_config.get("model_seeds", [0])]
    if len(seed_values) != 1:
        raise ValueError(f"this requested run is single-seed; got model_seeds={seed_values}")
    seed = seed_values[0]

    full_train_weight = _positive_weight(_graph_labels(center_graphs["train"]))
    refit_graphs = list(center_graphs["train"]) + list(center_graphs["valid"])
    full_refit_weight = _positive_weight(_graph_labels(refit_graphs))
    model_names = (
        "s_marginal_mlp",
        "center_fusion_plus_s_marginal_mlp",
    )
    models: dict[str, Any] = {}
    for name in model_names:
        candidates: dict[str, Any] = {}
        for pos_weight in candidate_weights:
            key = f"{pos_weight:g}"
            print(
                f"starting model={name} pos_weight={pos_weight:g} seed={seed}",
                flush=True,
            )
            candidates[key] = _validation_candidate(
                name,
                pos_weight,
                center_graphs=center_graphs,
                split_indices=split_indices,
                raw_global=raw_global,
                model_config=model_config,
                device=device,
                seed=seed,
            )
        selected_key, selected = _select_candidate(candidates)
        print(
            f"selected model={name} pos_weight={selected['pos_weight']:g} "
            f"valid_auc={selected['valid']['best_auc']:.6f}",
            flush=True,
        )
        test_phase = _refit_selected(
            name,
            selected,
            center_graphs=center_graphs,
            split_indices=split_indices,
            raw_global=raw_global,
            model_config=model_config,
            device=device,
            seed=seed,
        )
        models[name] = {
            "candidates": candidates,
            "selected_key": selected_key,
            "selected": selected,
            "test": test_phase,
        }

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
            "fusion": "[s_v, a_v, s_v*a_v] before sum/mean/std pooling",
        },
        "global_features": global_meta,
        "search": {
            "candidate_pos_weights": candidate_weights,
            "selection_rule": "highest official-valid AUC independently for each model; lower weight breaks exact ties",
            "full_train_balanced_pos_weight": full_train_weight,
            "full_train_valid_balanced_pos_weight": full_refit_weight,
            "test_evaluated_only_for_selected_candidate": True,
        },
        "training": {
            **dict(model_config),
            "device": str(device),
            "loss": "binary cross entropy with logits",
            "loss_definition": "candidate pos_weight on positive examples",
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
    result = run(_resolve(args.config))
    print(
        json.dumps(
            {
                name: {
                    "selected_pos_weight": row["selected"]["pos_weight"],
                    "valid_auc": row["selected"]["valid"]["best_auc"],
                    "selected_epoch": row["selected"]["valid"]["selected_epoch"],
                    "test_auc": row["test"]["test_auc"],
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
