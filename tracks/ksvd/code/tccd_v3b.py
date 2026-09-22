"""TCCD-v3b: correct frozen B-full pre-pair local-state bridge.

This round differs from historical TCCD-v3 in one and only one place: the
local input is the full B-full ``patch_encoder`` output (64D), captured before
any pair projection, pair encoder, centre-context update, global context, or
topology operation.  All downstream TCCD-v2 composition and reader mechanics
are reused unchanged through ``tccd_v3`` helpers.

Official ZINC test is never loaded.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v3 as V3
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as ZPP
from tracks.ksvd.experiments.luyin16 import zinc_shared_structural_patch_encoder as SSPE
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ZTOPO

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tccd_v3b"
CACHE_DIR = RESULTS_DIR / "cache"
PROTOCOL_VERSION = "tccd_v3b_correct_prepair_bridge_v1"

# Frozen B-full provenance.
STRONG_CHECKPOINT = SSPE.RESULTS_DIR / "soup_states/sspe_seed0_top5_soup.pt"
STRONG_SOURCE_COMMIT = "0aa71c81e845fd334c8ecd2bf7904c149277c289"
STRONG_CHECKPOINT_SHA256 = "785dff866fa8d952d6e9764549761a7019a4e46ffd70d625110b618f51e6a2c4"
STRONG_GRAPH_CACHE_SCHEMA = SSPE.CACHE_SCHEMA_VERSION

SELECTED_TENSOR = "PatchPathRecurrentPairCentreModel._encode_core -> patch_encoder output"
SELECTED_LAYER = "patch_encoder"
SELECTED_WIDTH = 64
PATCH_ENCODER_INPUT_WIDTH = 170
PATCH_ENCODER_INPUT_BLOCKS = {
    "patch_cont": 146,
    "patch_context": 0,
    "e_struct": 16,
    "parent_embedding": 8,
    "other_pure_local_blocks": 0,
}

# Historical references, frozen by the preregistration.
TCCD_V2_PROTO_BEST = 0.287337
TCCD_V2_PROTO_SOUP = 0.261988
TCCD_V3_ISOLATED_E_STRUCT_PROTO = 0.982502
CANONICAL_B_FULL = 0.119818

D_LOCAL = 64
K_PROTO = V3.K_PROTO
N_REL = V3.N_REL
LOCAL_PASS_TOL = 1.0e-6
PERM_PASS_TOL = 1.0e-5
BATCH_PASS_TOL = 1.0e-5
DETERMINISM_PASS_TOL = 0.0
CACHE_MATCH_TOL = 2.0e-6

L1_MAX = 0.20
L1_DELTA = 0.07
L2_MAX = 0.24
L2_DELTA = 0.04
PROTO_PRESERVE = 0.015
PROTO_HARMFUL = 0.03
COMP_MIN = 0.01


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"correct_prepair_{split}.pkl.gz"


def _cache_metadata_path(split: str) -> Path:
    return CACHE_DIR / f"correct_prepair_{split}_metadata.json"


def _source_model(device: str | torch.device = "cpu") -> nn.Module:
    if not STRONG_CHECKPOINT.exists():
        raise FileNotFoundError(f"missing frozen B-full checkpoint: {STRONG_CHECKPOINT}")
    actual_sha = _sha256_file(STRONG_CHECKPOINT)
    if actual_sha != STRONG_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"source checkpoint SHA256 mismatch: {actual_sha} != {STRONG_CHECKPOINT_SHA256}"
        )
    model = SSPE.build_candidate(0)
    state = torch.load(STRONG_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _model_inventory(model: nn.Module) -> dict[str, Any]:
    sequence = model.patch_encoder.layers
    out_width = int(sequence[-2].out_features)
    in_width = int(sequence[0].in_features)
    if out_width != SELECTED_WIDTH:
        raise RuntimeError(f"patch_encoder output width changed: {out_width}")
    if in_width != PATCH_ENCODER_INPUT_WIDTH:
        raise RuntimeError(f"patch_encoder input width changed: {in_width}")
    if model.pair_projection.in_features != out_width:
        raise RuntimeError("selected patch state is not the pair_projection input")
    return {
        "model_class": type(model).__name__,
        "selected_tensor": SELECTED_TENSOR,
        "selected_layer": SELECTED_LAYER,
        "is_patch_encoder_output": True,
        "is_patch_encoder_input_block": False,
        "patch_encoder_input_width": in_width,
        "patch_encoder_output_width": out_width,
        "pair_projection_input_width": int(model.pair_projection.in_features),
        "pair_projection_output_width": int(model.pair_projection.out_features),
        "patch_hidden": int(model.patch_hidden),
        "structural_encoder_output_width": int(model.structural_encoder.output_dim),
        "input_blocks": dict(PATCH_ENCODER_INPUT_BLOCKS),
        "input_blocks_sum": int(sum(PATCH_ENCODER_INPUT_BLOCKS.values())),
        "next_operation": "pair_projection / pair encoder",
    }


def _capture_patch_encoder(model: nn.Module, batch: Data) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured.append(output.detach())

    handle = model.patch_encoder.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.encode(batch)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected exactly one patch_encoder call, got {len(captured)}")
    output = captured[0]
    if output.ndim != 2 or int(output.shape[1]) != SELECTED_WIDTH:
        raise RuntimeError(f"captured patch state has unexpected shape {tuple(output.shape)}")
    return output


def _split_batch_rows(output: torch.Tensor, batch: Data) -> list[np.ndarray]:
    if not hasattr(batch, "batch"):
        raise RuntimeError("batched source data has no graph batch index")
    counts = torch.bincount(batch.batch.detach().cpu()).tolist()
    if sum(int(x) for x in counts) != int(output.shape[0]):
        raise RuntimeError("patch state row count does not match graph batch")
    rows: list[np.ndarray] = []
    offset = 0
    for count in counts:
        count = int(count)
        rows.append(output[offset : offset + count].cpu().numpy().astype(np.float32, copy=True))
        offset += count
    return rows


def _cache_metadata(
    split: str,
    n_graphs: int,
    embeddings: Sequence[np.ndarray],
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema": "tccd_v3b_correct_prepair_embeddings_v1",
        "split": split,
        "n_graphs": int(n_graphs),
        "n_patches": int(sum(int(x.shape[0]) for x in embeddings)),
        "graph_patch_counts": [int(x.shape[0]) for x in embeddings],
        "graph_ids": list(range(int(n_graphs))),
        "centre_ids": [list(range(int(x.shape[0]))) for x in embeddings],
        "strong_checkpoint": str(STRONG_CHECKPOINT.relative_to(REPO_ROOT)),
        "strong_checkpoint_sha256": _sha256_file(STRONG_CHECKPOINT),
        "strong_source_commit": STRONG_SOURCE_COMMIT,
        "strong_source_valid_mae_seed0_soup": 0.11981802638241788,
        "strong_source_model_selection": "official-valid-selected Top-5 soup",
        "strong_graph_cache_schema": STRONG_GRAPH_CACHE_SCHEMA,
        "selected_tensor": SELECTED_TENSOR,
        "selected_layer": SELECTED_LAYER,
        "is_patch_encoder_output": True,
        "is_patch_encoder_input_block": False,
        "raw_width": SELECTED_WIDTH,
        "patch_encoder_input_width": PATCH_ENCODER_INPUT_WIDTH,
        "input_blocks": dict(PATCH_ENCODER_INPUT_BLOCKS),
        "relation_metadata_reference": f"tccd_v3b/{split}/graph_index_and_centre_id",
        "official_test_loaded": False,
        "source_audit": dict(source_audit),
    }


def extract_prepair_embeddings(
    split: str,
    device: str,
    *,
    batch_size: int = 128,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if split not in {"train", "valid"}:
        raise ValueError("official test is forbidden")
    train_data, valid_data, _audit = SSPE.build_encoded_records()
    data = train_data if split == "train" else valid_data
    model = _source_model(device)
    source_audit = _model_inventory(model)
    loader = SSPE._make_struct_loader(data, int(batch_size), False, 0)
    embeddings: list[np.ndarray | None] = [None] * len(data)
    graph_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = _capture_patch_encoder(model, batch)
            rows = _split_batch_rows(output, batch)
            for local_graph_id, row in enumerate(rows):
                embeddings[graph_offset + int(local_graph_id)] = row
            graph_offset += len(rows)
    if any(row is None for row in embeddings):
        raise RuntimeError(f"missing cached rows for split={split}")
    final = [row for row in embeddings if row is not None]
    if any(row.shape[1] != SELECTED_WIDTH for row in final):
        raise RuntimeError("pre-pair cache width mismatch")
    metadata = _cache_metadata(split, len(final), final, source_audit)
    return final, metadata


def load_or_build_prepair_cache(
    split: str,
    device: str,
    *,
    batch_size: int = 128,
    force: bool = False,
    require_existing: bool = False,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    path = _cache_path(split)
    if path.exists() and not force:
        with gzip.open(path, "rb") as handle:
            payload = json_or_pickle_load(handle)
        metadata = dict(payload["metadata"])
        valid = (
            metadata.get("cache_schema") == "tccd_v3b_correct_prepair_embeddings_v1"
            and metadata.get("strong_checkpoint_sha256") == _sha256_file(STRONG_CHECKPOINT)
            and metadata.get("raw_width") == SELECTED_WIDTH
            and metadata.get("official_test_loaded") is False
            and len(payload["embeddings"]) == int(metadata.get("n_graphs", -1))
        )
        if valid:
            return [np.asarray(row, dtype=np.float32) for row in payload["embeddings"]], metadata
    if require_existing:
        raise FileNotFoundError(
            f"valid frozen pre-pair cache required before formal training: {path}"
        )
    embeddings, metadata = extract_prepair_embeddings(split, device, batch_size=batch_size)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        json_or_pickle_dump({"metadata": metadata, "embeddings": embeddings}, handle)
    _write_json(_cache_metadata_path(split), metadata)
    return embeddings, metadata


def json_or_pickle_load(handle):
    import pickle

    return pickle.load(handle)


def json_or_pickle_dump(payload, handle) -> None:
    import pickle

    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _official_records(data_root: Path):
    return V3._official_records(data_root)


def build_bridge_records(
    data_root: Path,
    device: str,
    *,
    force_embeddings: bool = False,
    require_existing_cache: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_records, valid_records, protocol_meta = _official_records(data_root)
    train_e, train_meta = load_or_build_prepair_cache(
        "train",
        device,
        force=force_embeddings,
        require_existing=require_existing_cache,
    )
    valid_e, valid_meta = load_or_build_prepair_cache(
        "valid",
        device,
        force=force_embeddings,
        require_existing=require_existing_cache,
    )
    if len(train_records) != len(train_e) or len(valid_records) != len(valid_e):
        raise RuntimeError("pre-pair cache / TCCD record graph count mismatch")
    outputs: list[list[dict[str, Any]]] = [[], []]
    for records, embeddings, split, target in (
        (train_records, train_e, "train", outputs[0]),
        (valid_records, valid_e, "valid", outputs[1]),
    ):
        for graph_id, (record, embedding) in enumerate(zip(records, embeddings)):
            if int(record["n"]) != int(embedding.shape[0]):
                raise RuntimeError(f"{split} graph {graph_id}: centre count mismatch")
            copied = dict(record)
            copied["X"] = np.asarray(embedding, dtype=np.float32)
            copied["graph_index"] = int(graph_id)
            copied["split"] = split
            copied["centre_ids"] = list(range(int(record["n"])))
            target.append(copied)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": protocol_meta,
        "train_embedding_cache": train_meta,
        "valid_embedding_cache": valid_meta,
        "input_width": SELECTED_WIDTH,
        "adapter_width": D_LOCAL,
        "official_test_loaded": False,
    }
    return outputs[0], outputs[1], metadata


class CorrectStrongLocalDenseREL(V3.DenseBridgeModel):
    """Correct pre-pair state -> one Linear(64,64) -> DenseREL."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(input_width=SELECTED_WIDTH, seed=seed)
        self.graph_feature_kind = "dense_latent_composition_only"
        self.bridge_kind = "correct_fused_prepair_patch_encoder_output"


class CorrectStrongLocalPrototypeREL(V3.PrototypeBridgeModel):
    """Correct pre-pair state -> one Linear(64,64) -> frozen TCCD-v2 PrototypeREL."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(input_width=SELECTED_WIDTH, seed=seed)
        self.graph_feature_kind = "assignment_composition_only"
        self.bridge_kind = "correct_fused_prepair_patch_encoder_output"


def build_model(arm: str, seed: int = 0) -> nn.Module:
    if arm == "dense":
        return CorrectStrongLocalDenseREL(seed=seed)
    if arm == "prototype":
        return CorrectStrongLocalPrototypeREL(seed=seed)
    raise ValueError(arm)


def _synthetic_raw_graph(exterior_atom: int, exterior_bond: int) -> Data:
    edges = [(0, 1), (1, 2), (2, 3)]
    directed = edges + [(b, a) for a, b in edges]
    bonds = [0, 1, exterior_bond, 0, 1, exterior_bond]
    return Data(
        edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
        x=torch.tensor([0, 1, 2, int(exterior_atom)], dtype=torch.long),
        edge_attr=torch.tensor(bonds, dtype=torch.long),
        num_nodes=4,
        y=torch.tensor([0.0]),
    )


def _synthetic_full_data(raw: Data) -> Data:
    graphs = SSPE._patch_graphs_from_dataset([raw])
    graph = graphs[0]
    node_mask = graph.patch == 0
    edge_mask = graph.edge_patch == 0
    old_nodes = np.flatnonzero(node_mask)
    remap = {int(old): i for i, old in enumerate(old_nodes.tolist())}
    src = np.asarray([remap[int(x)] for x in graph.src[edge_mask]], dtype=np.int64)
    dst = np.asarray([remap[int(x)] for x in graph.dst[edge_mask]], dtype=np.int64)
    node_count = int(old_nodes.size)
    return Data(
        patch_cont=torch.zeros((1, ZPP.SHELL_WIDTH), dtype=torch.float32),
        patch_context=torch.zeros((1, 0), dtype=torch.float32),
        typed_token=torch.zeros((1,), dtype=torch.long),
        parent_token=torch.zeros((1,), dtype=torch.long),
        structural_token=torch.zeros((1,), dtype=torch.long),
        structural_coarse=torch.zeros((1, ZPP.COARSE_WIDTH), dtype=torch.float32),
        pair_index=torch.zeros((2, 0), dtype=torch.long),
        pair_relation=torch.zeros((0, ZPP.RELATION_WIDTH), dtype=torch.float32),
        pair_bucket=torch.zeros((0,), dtype=torch.long),
        global_context=torch.zeros((1, ZPP.GLOBAL_WIDTH), dtype=torch.float32),
        y=torch.tensor([0.0], dtype=torch.float32),
        num_nodes=1,
        topology_features=torch.zeros((1, ZTOPO.raw_width("hinge")), dtype=torch.float32),
        struct_atom=torch.from_numpy(graph.atom[node_mask]),
        struct_root=torch.from_numpy(graph.root[node_mask]),
        struct_dist=torch.from_numpy(graph.dist[node_mask]),
        struct_patch=torch.zeros((node_count,), dtype=torch.long),
        struct_src=torch.from_numpy(src),
        struct_dst=torch.from_numpy(dst),
        struct_bond=torch.from_numpy(graph.bond[edge_mask]),
    )


def _batch_one(data: Data) -> Batch:
    return SSPE.struct_collate([data])


def _synthetic_relabel(data: Data) -> Data:
    out = copy.deepcopy(data)
    n = int(data.struct_atom.numel())
    perm = torch.tensor(list(range(n))[::-1], dtype=torch.long)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n)
    out.struct_atom = data.struct_atom[perm]
    out.struct_root = data.struct_root[perm]
    out.struct_dist = data.struct_dist[perm]
    out.struct_patch = data.struct_patch[perm]
    out.struct_src = inverse[data.struct_src]
    out.struct_dst = inverse[data.struct_dst]
    out.struct_bond = data.struct_bond.clone()
    return out


def gate0_checks(device: str = "cpu", *, require_cache: bool = False) -> dict[str, Any]:
    """Correctness gate for the selected full fused pre-pair state."""
    if not STRONG_CHECKPOINT.exists():
        return {"all_pass": False, "checkpoint_present": False, "official_test_loaded": False}
    model = _source_model(device)
    inventory = _model_inventory(model)
    raw_a = _synthetic_raw_graph(3, 2)
    raw_b = _synthetic_raw_graph(7, 3)
    data_a = _synthetic_full_data(raw_a)
    data_b = _synthetic_full_data(raw_b)
    with torch.no_grad():
        out_a = _capture_patch_encoder(model, _batch_one(data_a).to(device))
        out_b = _capture_patch_encoder(model, _batch_one(data_b).to(device))
        out_batch = _capture_patch_encoder(
            model, SSPE.struct_collate([data_a, data_b]).to(device)
        )[:1]
        reload_model = _source_model(device)
        out_reload = _capture_patch_encoder(
            model=reload_model, batch=_batch_one(data_a).to(device)
        )
        out_perm = _capture_patch_encoder(
            model, _batch_one(_synthetic_relabel(data_a)).to(device)
        )

    # Pair/global/topology intervention on one actual local patch.  The hook is
    # captured before all pair/global operations, so this is also a code-path
    # independence check rather than a post-hoc feature comparison.
    intervention = copy.deepcopy(data_a)
    intervention.pair_relation = torch.full_like(intervention.pair_relation, 7.0)
    intervention.global_context = torch.ones_like(intervention.global_context)
    intervention.topology_features = torch.ones_like(intervention.topology_features)
    with torch.no_grad():
        out_intervention = _capture_patch_encoder(
            model, _batch_one(intervention).to(device)
        )

    checks: dict[str, Any] = {
        "official_test_loaded": False,
        "checkpoint_present": True,
        "is_patch_encoder_output": bool(inventory["is_patch_encoder_output"]),
        "is_patch_encoder_input_block": bool(inventory["is_patch_encoder_input_block"]),
        "selected_tensor_is_not_input_block": not bool(inventory["is_patch_encoder_input_block"]),
        "selected_width": int(inventory["patch_encoder_output_width"]) == SELECTED_WIDTH,
        "input_blocks_sum_matches_width": int(inventory["input_blocks_sum"]) == PATCH_ENCODER_INPUT_WIDTH,
        "pair_projection_consumes_selected_state": int(inventory["pair_projection_input_width"]) == SELECTED_WIDTH,
        "exterior_max_delta": float((out_a - out_b).abs().max()),
        "exterior_invariant": float((out_a - out_b).abs().max()) <= LOCAL_PASS_TOL,
        "batch_max_delta": float((out_a - out_batch).abs().max()),
        "batch_invariant": float((out_a - out_batch).abs().max()) <= BATCH_PASS_TOL,
        "reload_max_delta": float((out_a - out_reload).abs().max()),
        "deterministic_reload": float((out_a - out_reload).abs().max()) <= DETERMINISM_PASS_TOL,
        "relabel_max_delta": float((out_a - out_perm).abs().max()),
        "permutation_invariant": float((out_a - out_perm).abs().max()) <= PERM_PASS_TOL,
        "pair_global_intervention_max_delta": float((out_a - out_intervention).abs().max()),
        "pair_global_independent": float((out_a - out_intervention).abs().max()) <= LOCAL_PASS_TOL,
        "frozen_parameters": all(not parameter.requires_grad for parameter in model.parameters()),
        "source_model_width": int(out_a.shape[1]),
    }

    # Task gradients on the TCCD-v2 bridge modules using a synthetic batch.
    records = _synthetic_records()
    dense = CorrectStrongLocalDenseREL(0)
    proto = CorrectStrongLocalPrototypeREL(0)
    batch = V3.make_batch(records, [0, 1], "cpu")
    dense_loss = (dense.forward_padded(batch)[0] - batch["y"]).abs().mean()
    dense_loss.backward()
    proto_loss = (proto.forward_padded(batch)[0] - batch["y"]).abs().mean()
    proto_loss.backward()
    checks.update(
        {
            "dense_adapter_gradient": dense.adapter.weight.grad is not None and float(dense.adapter.weight.grad.abs().sum()) > 0.0,
            "dense_reader_gradient": dense.head.weight.grad is not None and float(dense.head.weight.grad.abs().sum()) > 0.0,
            "prototype_adapter_gradient": proto.adapter.weight.grad is not None and float(proto.adapter.weight.grad.abs().sum()) > 0.0,
            "prototype_gradient": proto.P.grad is not None and float(proto.P.grad.abs().sum()) > 0.0,
            "temperature_gradient": proto.temp_logit.grad is not None and float(proto.temp_logit.grad.abs().sum()) > 0.0,
            "prototype_reader_gradient": proto.head.weight.grad is not None and float(proto.head.weight.grad.abs().sum()) > 0.0,
            "prototype_no_raw_bypass": bool(proto.uses_latent_bypass is False and proto.graph_feature_kind == "assignment_composition_only"),
            "source_parameters_receive_no_gradient": all(parameter.grad is None for parameter in model.parameters()),
        }
    )

    cache_match = _cache_match_check(model, device)
    checks.update(cache_match)
    if not require_cache and not checks.get("cache_present", False):
        checks["cache_not_required"] = True
    else:
        checks["cache_not_required"] = False
    bool_checks = [
        value
        for key, value in checks.items()
        if isinstance(value, bool)
        and key not in {
            "official_test_loaded",
            "cache_not_required",
            "is_patch_encoder_input_block",
        }
    ]
    if not require_cache and not checks.get("cache_present", False):
        bool_checks = [
            value
            for key, value in checks.items()
            if isinstance(value, bool)
            and key not in {
                "official_test_loaded",
                "cache_not_required",
                "cache_present",
                "cache_matches_fresh",
                "is_patch_encoder_input_block",
            }
        ]
    checks["all_pass"] = bool(all(bool_checks))
    checks["inventory"] = inventory
    return checks


def _cache_match_check(model: nn.Module, device: str) -> dict[str, Any]:
    max_delta_by_split: dict[str, float] = {}
    for split in ("train", "valid"):
        path = _cache_path(split)
        if not path.exists():
            return {
                "cache_present": False,
                "cache_fresh_max_delta": None,
                "cache_fresh_max_delta_by_split": max_delta_by_split,
                "cache_matches_fresh": False,
            }
        with gzip.open(path, "rb") as handle:
            import pickle

            payload = pickle.load(handle)
        metadata = dict(payload["metadata"])
        if metadata.get("official_test_loaded") is not False:
            return {
                "cache_present": False,
                "cache_fresh_max_delta": None,
                "cache_fresh_max_delta_by_split": max_delta_by_split,
                "cache_matches_fresh": False,
            }
        embeddings = [np.asarray(x, dtype=np.float32) for x in payload["embeddings"]]
        data = SSPE.build_encoded_records()[0 if split == "train" else 1]
        loader = SSPE._make_struct_loader(list(data)[:2], 2, False, 0)
        batch = next(iter(loader)).to(device)
        fresh = _split_batch_rows(_capture_patch_encoder(model, batch), batch)
        max_delta = max(
            float(np.max(np.abs(cached - current)))
            for cached, current in zip(embeddings[: len(fresh)], fresh)
        )
        max_delta_by_split[split] = max_delta
    max_delta = max(max_delta_by_split.values())
    return {
        "cache_present": True,
        "cache_fresh_max_delta": max_delta,
        "cache_fresh_max_delta_by_split": max_delta_by_split,
        "cache_matches_fresh": max_delta <= CACHE_MATCH_TOL,
    }


def _synthetic_records(n_graphs: int = 4) -> list[dict[str, Any]]:
    layout = T.PatchLayout(capacity=8, n_atom=5, n_bond=3)
    atom_index = {c: i for i, c in enumerate(range(5))}
    bond_index = {c: i for i, c in enumerate(range(3))}
    records: list[dict[str, Any]] = []
    for seed in range(n_graphs):
        record = T.build_mol_record(
            synthetic_zinc_like(seed, 10 + seed), layout, atom_index, bond_index
        )
        rng = np.random.default_rng(100 + seed)
        record["X"] = rng.standard_normal((int(record["n"]), SELECTED_WIDTH)).astype(np.float32)
        record["y"] = float(seed) / 3.0
        records.append(record)
    return records


def classify_decision(strong_proto: float) -> str:
    delta_local = float(TCCD_V2_PROTO_BEST - strong_proto)
    if strong_proto <= L1_MAX and delta_local >= L1_DELTA:
        return "L1"
    if strong_proto <= L2_MAX and delta_local >= L2_DELTA:
        return "L2"
    return "L3"


def interpretation_proto_gap(gap: float) -> str:
    if gap <= 0.0:
        return "prototype has non-positive gap and positive inductive-bias evidence"
    if gap <= PROTO_PRESERVE:
        return "prototype vocabulary essentially preserves dense information"
    if gap > PROTO_HARMFUL:
        return "prototype vocabulary is harmful under the correct strong local state"
    return "prototype gap is intermediate / unresolved under registered thresholds"


def interpretation_composition(gain: float) -> str:
    if gain > COMP_MIN:
        return "assignment-sensitive C^T R C composition remains materially useful"
    return "explicit assignment-sensitive C^T R C composition no longer contributes materially"


def vocabulary_diagnostics(model, records, indices, device: str) -> dict[str, Any]:
    return V3.vocabulary_diagnostics(model, records, indices, device)


def evaluate_mae(model, records, indices, device: str, **kwargs: Any) -> float:
    return V3.evaluate_mae(model, records, indices, device, **kwargs)


def train_bridge(model, *args: Any, **kwargs: Any):
    return V3.train_bridge(model, *args, **kwargs)
