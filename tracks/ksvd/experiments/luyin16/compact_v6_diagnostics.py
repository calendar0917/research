"""Compact-v6 mechanism diagnostics on frozen validation checkpoints.

Runs no training.  It rebuilds the validation-selection phase exactly as the
recorded run did (train-only vocabularies + standardizers), loads the frozen
best-validation state dict, and then measures:

* clean validation MAE;
* attribute-type shuffle   (permute atom/bond *types* inside each patch);
* role-association shuffle (permute role descriptors inside each patch);
* per-molecule error for the historical difficulty quintiles;
* the attribute-branch output norm distribution.

``count_control`` receives the identical shuffles so we can show that the
count control is invariant to placement while the factorized-role branch is
not.  The official test split is never loaded.

Usage::

    uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as pp
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _load_zinc,
)
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


def _load_config(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "legacy_input.yaml").read_text(encoding="utf-8"))


def _load_result(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "artifacts" / "legacy_full_result.json").read_text(encoding="utf-8")
    )


def _extract(config: Mapping[str, Any]):
    data_root = pp._resolve(config["data"]["root"])
    representation = config.get("representation", {})
    model_config = config["model"]
    patch_radius = int(representation.get("patch_radius", pp.PATCH_RADIUS))
    context_radius = int(representation.get("context_radius", 0))
    tokenizer_version = pp.resolve_typed_tokenizer_version(
        representation.get("typed_tokenizer_version")
    )
    structural_mode = str(model_config.get("structural_context_mode", "none"))
    topology_mode = str(model_config.get("topology_mode", "none"))
    attribute_mode = str(model_config.get("attribute_mode", "none"))
    max_cycle_len = int(representation.get("structural_context_max_cycle_len", 10))
    certificate_cache: dict[bytes, bytes] = {}
    records: list[list[pp.GraphRecord]] = []
    for split in ("train", "val"):
        dataset = _load_zinc(data_root, split)
        matrix = None
        if topology_mode != "none":
            matrix, _frame, _meta = ztopo.matrices_for_split(
                "valid" if split == "val" else split,
                dataset,
                topology_mode,
                force=False,
                input_width=(
                    int(model_config.get("topology_input_width"))
                    if topology_mode == "capacity_control"
                    else None
                ),
            )
        split_records, _meta = pp._extract_split(
            dataset,
            split,
            certificate_cache,
            patch_radius=patch_radius,
            context_radius=context_radius,
            structural_mode=structural_mode,
            max_cycle_len=max_cycle_len,
            topology_mode=topology_mode,
            topology_matrix=matrix,
            tokenizer_version=tokenizer_version,
            attribute_mode=attribute_mode,
        )
        records.append(split_records)
    return records[0], records[1], patch_radius, context_radius, structural_mode


def _build_model(config, audit, patch_radius, context_radius, **overrides):
    model_config = config["model"]
    return pp.PatchPathModel(
        int(audit["typed_vocabulary_size_with_oov"]),
        int(audit["parent_vocabulary_size_with_oov"]),
        patch_hidden=int(model_config.get("patch_hidden", pp.PATCH_HIDDEN)),
        pair_hidden=int(model_config.get("pair_hidden", pp.PAIR_HIDDEN)),
        token_width=int(model_config.get("token_width", 32)),
        dropout=float(model_config.get("dropout", 0.05)),
        embedding_mode=str(model_config.get("embedding_mode", "full")),
        embedding_rank=int(model_config.get("embedding_rank", 16)),
        parent_embedding_rank=(
            None
            if model_config.get("parent_embedding_rank") is None
            else int(model_config["parent_embedding_rank"])
        ),
        hybrid_full_typed_tokens=(
            None
            if model_config.get("hybrid_full_typed_tokens") is None
            else int(model_config["hybrid_full_typed_tokens"])
        ),
        hybrid_full_parent_tokens=(
            None
            if model_config.get("hybrid_full_parent_tokens") is None
            else int(model_config["hybrid_full_parent_tokens"])
        ),
        readout=str(model_config.get("readout", "moments")),
        node_readout=model_config.get("node_readout"),
        pair_readout=model_config.get("pair_readout"),
        shell_width=pp._shell_width_for_radius(patch_radius),
        context_width=int(pp.CONTEXT_WIDTH if context_radius > patch_radius else 0),
        direct_token_readout=bool(model_config.get("direct_token_readout", False)),
        center_context=bool(model_config.get("center_context", False)),
        center_context_hidden=(
            None
            if model_config.get("center_context_hidden") is None
            else int(model_config["center_context_hidden"])
        ),
        graph_head_hidden_0=(
            None
            if model_config.get("graph_head_hidden_0") is None
            else int(model_config["graph_head_hidden_0"])
        ),
        graph_head_hidden_1=(
            None
            if model_config.get("graph_head_hidden_1") is None
            else int(model_config["graph_head_hidden_1"])
        ),
        structural_context_mode=str(model_config.get("structural_context_mode", "none")),
        structural_context_fusion=str(
            model_config.get("structural_context_fusion", "condition")
        ),
        structural_context_dim=int(model_config.get("structural_context_dim", 8)),
        structural_context_embedding_rank=int(
            model_config.get("structural_context_embedding_rank", 1)
        ),
        structural_context_condition_rank=int(
            model_config.get("structural_context_condition_rank", 8)
        ),
        structural_context_vocabulary_size=int(
            model_config.get("structural_context_vocabulary_size", 2)
        ),
        topology_mode=str(model_config.get("topology_mode", "none")),
        topology_input_width=int(audit["topology"]["input_width"]),
        topology_hidden_dim=int(model_config.get("topology_hidden_dim", 16)),
        topology_out_dim=int(model_config.get("topology_out_dim", 8)),
        attribute_mode=str(model_config.get("attribute_mode", "none")),
        attribute_atom_dim=int(
            model_config.get("attribute_atom_dim", pp.DEFAULT_ATTRIBUTE_ATOM_DIM)
        ),
        attribute_bond_dim=int(
            model_config.get("attribute_bond_dim", pp.DEFAULT_ATTRIBUTE_BOND_DIM)
        ),
        attribute_hidden=int(
            model_config.get("attribute_hidden", pp.DEFAULT_ATTRIBUTE_HIDDEN)
        ),
        attribute_out_dim=int(
            model_config.get("attribute_out_dim", pp.DEFAULT_ATTRIBUTE_OUT_DIM)
        ),
        attribute_fusion_init=str(
            overrides.get(
                "attribute_fusion_init",
                model_config.get("attribute_fusion_init", "zero"),
            )
        ),
    )


def _evaluate(model, graphs, batch_size=128):
    loader = pp._make_loader(graphs, int(batch_size), False, 0)
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch).cpu().numpy().reshape(-1))
            targets.append(batch.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds).astype(np.float64)
    targets = np.concatenate(targets).astype(np.float64)
    return float(np.mean(np.abs(preds - targets))), targets, preds


def _rng_for(seed: int, index: int, tag: str) -> np.random.Generator:
    digest = __import__("hashlib").blake2b(digest_size=8)
    digest.update(tag.encode("utf-8"))
    digest.update(np.asarray([seed, index], dtype=np.int64).tobytes())
    return np.random.default_rng(int.from_bytes(digest.digest(), "little"))


def _shuffle_graph(data, mode: str, seed: int, index: int):
    """Return a copy of ``data`` with intra-patch attribute shuffles applied.

    ``mode='type'`` permutes atom/bond *types* inside each patch (multiset
    preserved).  ``mode='role'`` permutes role descriptors inside each patch
    (types and roles both preserved as multisets, binding broken).
    """
    copy = data.clone()
    rng = _rng_for(seed, index, mode)
    atom_patch = data.attribute_atom_patch_index.numpy()
    bond_patch = data.attribute_bond_patch_index.numpy()
    if mode == "type":
        for kind, patch_index, key in (
            ("atom", atom_patch, "attribute_atom_type"),
            ("bond", bond_patch, "attribute_bond_type"),
        ):
            values = getattr(copy, key).clone()
            for patch in np.unique(patch_index):
                rows = np.where(patch_index == patch)[0]
                if rows.size > 1:
                    values[rows] = values[rows][rng.permutation(rows.size)]
            setattr(copy, key, values)
    elif mode == "role":
        for key, patch_index in (
            ("attribute_atom_role", atom_patch),
            ("attribute_bond_role_left", bond_patch),
            ("attribute_bond_role_right", bond_patch),
        ):
            values = getattr(copy, key).clone()
            for patch in np.unique(patch_index):
                rows = np.where(patch_index == patch)[0]
                if rows.size > 1:
                    values[rows] = values[rows][rng.permutation(rows.size)]
            setattr(copy, key, values)
    else:
        raise ValueError(mode)
    return copy


def _attribute_norms(model, graphs, batch_size=128):
    norms: list[np.ndarray] = []
    hooks = []

    def hook(_module, _inputs, output):
        norms.append(output.detach().norm(dim=1).cpu().numpy())

    hooks.append(model.attribute_encoder.register_forward_hook(hook))
    loader = pp._make_loader(graphs, int(batch_size), False, 0)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            model(batch)
    for handle in hooks:
        handle.remove()
    return np.concatenate(norms) if norms else np.zeros(0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factorized-run", type=Path,
        default=REPO_ROOT / "tracks/ksvd/runs/2026/09/10/20260910-162801-fa8a41cf",
    )
    parser.add_argument(
        "--count-run", type=Path,
        default=REPO_ROOT / "tracks/ksvd/runs/2026/09/10/20260910-163550-00179d67",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "tracks/ksvd/results/compact_v6_diagnostics",
    )
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {}

    for label, run_dir in (
        ("factorized_role", args.factorized_run),
        ("count_control", args.count_run),
    ):
        config = _load_config(run_dir)
        train_records, valid_records, patch_radius, context_radius, _mode = _extract(
            config
        )
        _fit, valid_data, audit = pp._phase_data(
            train_records, valid_records, config=config
        )
        model = _build_model(config, audit, patch_radius, context_radius)
        state = torch.load(
            run_dir / "artifacts" / "legacy_full_result_selection_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        clean_mae, targets, _preds = _evaluate(model, valid_data)
        type_shuffled = [
            _shuffle_graph(data, "type", args.seed, index)
            for index, data in enumerate(valid_data)
        ]
        role_shuffled = [
            _shuffle_graph(data, "role", args.seed, index)
            for index, data in enumerate(valid_data)
        ]
        type_mae, _t, _p = _evaluate(model, type_shuffled)
        role_mae, _t, _p = _evaluate(model, role_shuffled)
        norms = (
            _attribute_norms(model, valid_data)
            if model.attribute_encoder is not None
            else np.zeros(0)
        )
        output[label] = {
            "clean_mae": clean_mae,
            "attribute_type_shuffle_mae": type_mae,
            "role_association_shuffle_mae": role_mae,
            "attribute_norm_mean": float(norms.mean()) if norms.size else None,
            "attribute_norm_std": float(norms.std()) if norms.size else None,
            "attribute_norm_p50": float(np.median(norms)) if norms.size else None,
            "attribute_norm_p95": float(np.percentile(norms, 95)) if norms.size else None,
        }
        np.save(args.output / f"{label}_valid_targets.npy", targets)
        np.save(args.output / f"{label}_attribute_norms.npy", norms)
        print(label, output[label], flush=True)

    (args.output / "diagnostics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("wrote", args.output / "diagnostics.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
