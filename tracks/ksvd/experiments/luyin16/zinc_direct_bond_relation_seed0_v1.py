"""ZINC direct chemical-bond relation entry, same-GPU-regime paired experiment (seed 0).

One round, exactly two formal trajectories on the frozen fresh fold of the
``zinc_local_tuple_fresh_fold_replication_seed0_v1`` source round:

* ``O`` — matched control: static pair relation stays the 15-D source input and
  the four direct-bond coordinates are exact zeros (capacity matched).
* ``T`` — candidate: the same 15-D source input concatenated with the 4-D
  primitive one-hot of the *direct* bond type connecting the two environments
  (all-zero for non-adjacent pairs).

Everything else — fit/dev split, ``g = y - c`` target, model skeleton,
optimizer, schedule, epochs, soup window, mask (C6) — is inherited from the
frozen source round.  The only new object is ``data.pair_beta``, the
pre-standardisation 4-D adjacent-bond one-hot taken from the train-only encoded
cache and independently re-derived from the raw ZINC ``edge_index`` /
``edge_attr``.

This module never loads official valid/test data and never inspects dev scores
during training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_local_tuple_fresh_fold_replication_seed0_v1 as fresh
from tracks.ksvd.experiments.luyin16 import zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as tuplemod
from tracks.ksvd.experiments.luyin16 import zinc_local_tuple_dictionary_vs_mlp_seed0_v1 as mlpmod
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_pooling_scale_count_seed0_v1 as pool
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw

PROTOCOL_VERSION = "zinc-direct-bond-relation-seed0-v1"
RESULT_SLUG = "zinc_direct_bond_relation_seed0_v1"
RESULTS_DIR = Path("tracks/ksvd/results") / RESULT_SLUG
SOURCE_DIR = fresh.RESULTS_DIR
ARMS = ("O", "T")
SEED = 0
EPOCHS = 240
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
TRAIN_SHUFFLE_OFFSET = fresh.TRAIN_SHUFFLE_OFFSET
N_FIT = fresh.N_FIT
N_DEV = fresh.N_DEV
LOG_EPOCHS = fresh.LOG_EPOCHS
BOOT_SEED = 20261007
N_BOOT = 1000
DELTA = 0.003
REPLAY_TOL = 1e-5
FIT_SHA = fresh.NEW_FIT_SHA
DEV_SHA = fresh.NEW_DEV_SHA
SCHEDULE_SHA = fresh.FROZEN_SCHEDULE_SHA
GID_STREAM_SHA = "69187f13fba5def82ee2f28765c1c46ac4129844d3f76ac769e018d18ae097d2"
BUILD_RNG_SHA = "a2e8a8ab56091d98d239b81fd15117cb818f1b88d95d2fc1bcff036eee5a8853"
TRAIN_RNG_SHA = "1ccf17250133dec5e96478acd63e13d728d0dbbabb30389415a302c6f872f81f"
SOURCE_INIT_HASH = "e1ed793ae85b632f31525aa3d67334b6caf40fce9dc95c4294ab9e199829cad2"
SOURCE_PARAMS = 297499
EXPANDED_PARAMS = SOURCE_PARAMS + 4 * 96
RELATION_INDICES = tuple(p1.P1_RELATION_INDICES)
RELATION_WIDTH = int(p1.RELATION_WIDTH)
BOND_CATEGORIES = int(p1.BOND_CATEGORIES)
GROUP_NAMES = ("k=0", "k=-1", "k=-2", "k<=-3")

seed_everything = fresh.seed_everything
write_json = fresh.write_json
file_sha256 = fresh.file_sha256
state_hash = fresh.state_hash
array_sha256 = fresh.array_sha256
_hash_bytes = fresh._hash_bytes


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



# ---------------------------------------------------------------------------
# runtime provenance
# ---------------------------------------------------------------------------


def _runtime(device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {
        "hostname": socket.gethostname(), "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE"),
        "SLURM_JOB_NODELIST": os.environ.get("SLURM_JOB_NODELIST"),
        "SLURM_NODELIST": os.environ.get("SLURM_NODELIST"),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(index)
        out.update({
            "device_name": prop.name, "device_index": index,
            "device_uuid": str(getattr(prop, "uuid", "")),
            "device_pci_bus_id": str(getattr(prop, "pci_bus_id", "")),
            "device_total_memory": int(prop.total_memory),
        })
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,driver_version", "--format=csv,noheader"],
                text=True, capture_output=True, timeout=30, check=False)
            out["nvidia_smi_gpu_inventory"] = q.stdout.strip()
            out["nvidia_smi_driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, capture_output=True, timeout=30, check=False).stdout.strip()
            out["nvidia_smi_compute_apps"] = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                text=True, capture_output=True, timeout=30, check=False).stdout.strip()
        except Exception as exc:  # pragma: no cover - provenance only
            out["nvidia_smi_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# frozen inputs
# ---------------------------------------------------------------------------


def prepare_inputs(*, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_manifest = json.loads((SOURCE_DIR / "fresh_manifest.json").read_text())
    mapping = {
        "fresh_fold.npz": "fold_npz", "fresh_targets.npz": "targets_npz",
        "fresh_tuple_payload.npz": "payload_npz", "fresh_prep.npz": "prep_npz",
    }
    copied: dict[str, Any] = {}
    for name, key in mapping.items():
        src = SOURCE_DIR / name
        expected = src_manifest["artifacts"][key]["sha256"]
        actual = file_sha256(src)
        if actual != expected:
            raise RuntimeError(f"source artifact hash mismatch {name}: {actual} != {expected}")
        dst = out_dir / name
        if not dst.exists() or file_sha256(dst) != expected:
            shutil.copy2(src, dst)
        copied[name] = {"source": str(src), "sha256": expected, "copy_sha256": file_sha256(dst)}
    for name in ("fresh_kappa.json", "fresh_manifest.json", "M_init_state.pt"):
        src = SOURCE_DIR / name
        dst = out_dir / name
        if not dst.exists() or file_sha256(dst) != file_sha256(src):
            shutil.copy2(src, dst)
        copied[name] = {"source": str(src), "sha256": file_sha256(src), "copy_sha256": file_sha256(dst)}
    write_json(out_dir / "frozen_input_copy.json", copied)
    fresh.load_fresh_objects(out_dir)
    return {"copied": copied, "source_manifest": src_manifest}


def load_source_init(out_dir: Path = RESULTS_DIR) -> dict[str, torch.Tensor]:
    path = out_dir / "M_init_state.pt"
    if not path.exists():
        path = SOURCE_DIR / "M_init_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state_hash(state) != SOURCE_INIT_HASH:
        raise RuntimeError("source M init state hash mismatch")
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def expanded_source_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {key: value.detach().cpu().clone() for key, value in state.items()}
    weight = out.pop("relation_encoder.layers.0.weight")
    out["relation_encoder.layers.0.weight"] = torch.cat(
        (weight, torch.zeros(weight.shape[0], BOND_CATEGORIES)), dim=1)
    return out


# ---------------------------------------------------------------------------
# beta construction and verification
# ---------------------------------------------------------------------------


def _edge_map(raw: Any) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    """Undirected typed edges from one official-train graph's raw edge_index/attr."""
    ei = raw.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    ea = raw.edge_attr.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    if ei.shape[1] != ea.size:
        raise RuntimeError("edge_index / edge_attr length mismatch")
    directed: dict[tuple[int, int], int] = {}
    for j in range(ei.shape[1]):
        u, v = int(ei[0, j]), int(ei[1, j])
        if u == v:
            continue
        key = (u, v)
        typ = int(ea[j])
        if key in directed and directed[key] != typ:
            raise RuntimeError(f"inconsistent directed edge types for {key}")
        directed[key] = typ
    edges: dict[tuple[int, int], int] = {}
    for (u, v), typ in directed.items():
        if directed.get((v, u)) != typ:
            raise RuntimeError(f"directed reverse edge mismatch {(u, v)} type={typ}")
        key = (min(u, v), max(u, v))
        if key in edges and edges[key] != typ:
            raise RuntimeError(f"undirected edge type conflict {key}")
        edges[key] = typ
    counts = np.unique(ea, return_counts=True)
    return edges, {
        "n_directed_edges": len(directed), "n_undirected_edges": len(edges),
        "reverse_edge_types_equal": True,
        "edge_type_counts": {str(int(k)): int(v) for k, v in zip(*counts)},
    }


def build_pair_beta(train_data: Sequence[Any], *, verify_raw_edges: bool = True) -> dict[str, Any]:
    """Per-graph [n_pairs, 4] raw adjacent-bond one-hot; label fields are never read."""
    raw_train = None
    if verify_raw_edges:
        raw_train = zlr._load_zinc(Path(zpp.REPO_ROOT) / "data/ZINC", "train")
        if len(raw_train) != len(train_data):
            raise RuntimeError(f"raw/encoded train size mismatch: {len(raw_train)} vs {len(train_data)}")
    betas: list[torch.Tensor] = []
    category_mass = np.zeros(BOND_CATEGORIES, np.int64)
    n_adjacent = 0
    per_graph: list[dict[str, Any]] = []
    for gid, data in enumerate(train_data):
        relation = data.pair_relation.detach().cpu().numpy()
        if relation.ndim != 2 or relation.shape[1] != 23:
            raise RuntimeError(f"train graph {gid} pair_relation layout changed: {relation.shape}")
        cached = relation[:, 19:23].astype(np.float32, copy=False)
        if not np.isfinite(cached).all() or not np.isin(cached, (0.0, 1.0)).all():
            raise RuntimeError(f"graph {gid} direct-bond relation is not primitive 0/1")
        if (cached.sum(axis=1) > 1).any():
            raise RuntimeError(f"graph {gid} direct-bond relation is not one-hot-or-zero")
        pair_index = data.pair_index.detach().cpu().numpy().astype(np.int64, copy=False)
        meta: dict[str, Any] = {"train_global_graph_id": int(gid)}
        if raw_train is not None:
            raw = raw_train[gid]
            if int(raw.num_nodes) != int(data.num_nodes):
                raise RuntimeError(f"graph {gid}: node count mismatch encoded/raw")
            edges, edge_meta = _edge_map(raw)
            independent = np.zeros((pair_index.shape[1], BOND_CATEGORIES), np.float32)
            for j, (u0, v0) in enumerate(pair_index.T):
                u, v = int(u0), int(v0)
                if u == v:
                    raise RuntimeError(f"graph {gid}: self pair in relation cache")
                typ = edges.get((min(u, v), max(u, v)))
                if typ is not None:
                    if typ < 0 or typ >= BOND_CATEGORIES:
                        raise RuntimeError(f"raw bond category out of range: {typ}")
                    independent[j, typ] = 1.0
            if not np.array_equal(cached, independent):
                rows = np.flatnonzero(np.any(cached != independent, axis=1))[:5].tolist()
                raise RuntimeError(
                    f"graph {gid}: cached pair_relation[:,19:23] != raw edge_index/edge_attr beta rows={rows}")
            meta["edge_validation"] = edge_meta
        else:
            independent = cached.copy()
            meta["edge_validation"] = "not_run"
        beta = torch.from_numpy(np.ascontiguousarray(independent, dtype=np.float32))
        betas.append(beta)
        adjacent = beta.sum(1) > 0
        n_adjacent += int(adjacent.sum())
        category_mass += beta.sum(0).to(torch.int64).numpy()
        meta.update({
            "n_pairs": int(beta.shape[0]), "n_adjacent_pairs": int(adjacent.sum()),
            "n_nonadjacent_pairs": int((~adjacent).sum()),
            "nonadjacent_beta_zero": bool((beta[~adjacent] == 0).all()),
            "adjacent_beta_onehot": bool(torch.all(beta[adjacent].sum(1) == 1)),
        })
        per_graph.append(meta)
    return {"betas": betas, "meta": {
        "n_train_graphs": len(betas), "n_adjacent_pairs": int(n_adjacent),
        "category_mass": category_mass.tolist(),
        "category_counts_positive": int((category_mass > 0).sum()),
        "source": "train-only raw ZINC edge_index/edge_attr, cross-checked against pair_relation[:,19:23]",
        "relation_provenance": "zinc_patch_path_pooling._pair_relation emits float32 0/1 one-hot; no relation scaler",
        "raw_split": "PyG ZINC subset=True split=train only",
        "graph_checks": per_graph,
    }}


def attach_beta(data_list: Sequence[Any], betas: Sequence[torch.Tensor]) -> None:
    if len(data_list) != len(betas):
        raise RuntimeError("beta/data length mismatch")
    for i, (data, beta) in enumerate(zip(data_list, betas)):
        if tuple(beta.shape) != (int(data.pair_index.shape[1]), BOND_CATEGORIES):
            raise RuntimeError(f"beta shape mismatch graph {i}")
        data.pair_beta = beta.clone()


def _collate_beta(data_list: Sequence[Any]) -> torch.Tensor:
    return torch.cat([d.pair_beta for d in data_list], dim=0)


def load_fit_dev(*, out_dir: Path = RESULTS_DIR, verify_raw_edges: bool = True):
    if not (out_dir / "fresh_fold.npz").exists():
        prepare_inputs(out_dir=out_dir)
    objects = fresh.load_fresh_objects(out_dir)
    fold, targets = objects["fold"], objects["targets"]
    train_data = zftd.load_train_only()
    if len(train_data) != 10_000:
        raise RuntimeError("train-only cache length mismatch")
    # Reproduce the source round's fresh-fit preprocessing. The stored encoded
    # arrays are an all-train standardized cache; only the frozen 8000 fit rows
    # fit the replacement scalers. No validation/test cache or labels are read.
    prep_meta = fresh.build_fresh_prep(train_data, fold)
    for key, value in objects["prep"].items():
        if key not in prep_meta or not np.array_equal(np.asarray(prep_meta[key], np.float32), np.asarray(value, np.float32)):
            raise RuntimeError(f"fresh input preprocessing differs from frozen source at {key}")
    beta_info = build_pair_beta(train_data, verify_raw_edges=verify_raw_edges)
    attach_beta(train_data, beta_info["betas"])
    fit_data = [train_data[int(i)] for i in fold["fit_idx"]]
    dev_data = [train_data[int(i)] for i in fold["dev_idx"]]
    for position, index in enumerate(fold["fit_idx"].tolist()):
        fit_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    for position, index in enumerate(fold["dev_idx"].tolist()):
        dev_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    if len(fit_data) != N_FIT or len(dev_data) != N_DEV:
        raise RuntimeError("fit/dev rebuild length mismatch")
    return objects, fit_data, dev_data, beta_info


# ---------------------------------------------------------------------------
# model: source skeleton + 19-D static relation (only beta differs by arm)
# ---------------------------------------------------------------------------


class DirectBondFull(pool.ScaledLocalTupleFull):
    """``forward`` is ``audit.AuditModel.forward`` with one inserted concat."""

    bond_arm: str

    def forward(
        self,
        data: Any,
        *,
        mask: audit.AuditMask | None = None,
        fill: Mapping[str, torch.Tensor] | None = None,
        readout_perm: tuple[Sequence[str], torch.Tensor] | None = None,
        return_aux: bool = False,
        **_kwargs: Any,
    ):
        if mask is None:
            raise RuntimeError("DirectBondFull requires the explicit C6 mask (19-D relation path)")

        coord = self.code(data.dict_phi)
        if mask.coord_zero:
            coord = torch.zeros_like(coord)
        E = self.environments_masked(coord, data, mask, fill)

        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = audit.pool_moments_masked(E, batch, n_graphs, mask.unary_zero_blocks, fill)

        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        if mask.pair_projection_zero:
            u = torch.zeros_like(u)
        left = u[source]
        right = u[target]
        relation_input = data.pair_relation[:, list(RELATION_INDICES)]
        if mask.relation_zero_groups:
            relation_input = audit._replace_grouped_columns(
                relation_input, audit.RELATION_GROUPS, mask.relation_zero_groups, fill, "relation:"
            )
        if not hasattr(data, "pair_beta"):
            raise RuntimeError("explicit pair_beta missing")
        native_beta = data.pair_beta.to(device=relation_input.device, dtype=relation_input.dtype)
        if tuple(native_beta.shape) != (int(relation_input.shape[0]), BOND_CATEGORIES):
            raise RuntimeError(f"pair_beta shape {tuple(native_beta.shape)} incompatible with pair relation")
        beta_input = torch.zeros_like(native_beta) if self.bond_arm == "O" else native_beta
        relation_input = torch.cat((relation_input, beta_input), dim=1)
        if int(relation_input.shape[1]) != RELATION_WIDTH + BOND_CATEGORIES:
            raise RuntimeError("relation path was unexpectedly sliced away from 19-D")
        relation = self.relation_encoder(relation_input)
        product = left * right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        if mask.gate_off:
            gate = torch.ones_like(gate)
        pair_input = torch.cat([left + right, torch.abs(left - right), product * gate, relation], dim=1)
        pair_value = self.pair_encoder(pair_input)
        pair_batch = batch[source]
        relation_readout = audit.pool_pair_moments_masked(
            pair_value, pair_batch, data.pair_bucket, n_graphs, mask.pair_zero_blocks, fill
        )
        if readout_perm is not None:
            blocks, permutation = readout_perm
            unary, relation_readout = audit.apply_readout_permutation(
                unary, relation_readout, blocks, permutation
            )

        global_input = data.global_context
        if mask.global_zero_groups:
            global_input = audit._replace_grouped_columns(
                global_input, audit.GLOBAL_GROUPS, mask.global_zero_groups, fill, "global:"
            )
        graph_hidden = self.global_encoder(global_input)
        if mask.graph_hidden_zero:
            graph_hidden = torch.zeros_like(graph_hidden)
        topology_input = data.topology_features
        if mask.topology_zero:
            if fill is not None and "topology" in fill:
                topology_input = (
                    fill["topology"].to(topology_input.device, topology_input.dtype)
                    .reshape(1, -1).expand_as(topology_input).contiguous()
                )
            else:
                topology_input = torch.zeros_like(topology_input)
        topology = self.topology_encoder(topology_input)
        unified = torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)
        prediction = self.reader(unified).view(-1)
        if return_aux:
            return prediction, {
                "E": E, "coord": coord, "phi": data.dict_phi, "unary": unary,
                "relation_readout": relation_readout, "pair_value": pair_value,
                "pair_batch": pair_batch, "relation_input": relation_input,
                "beta_input": beta_input, "native_beta": native_beta,
            }
        return prediction


def build_arm(arm: str, payload: tuplemod.TuplePayload, kappa: float) -> DirectBondFull:
    """Byte-identical construction to source M, then the exact 15->19 expansion.

    The first ``Linear`` is replaced by a 19-column copy (original 15 columns
    and bias preserved, 4 new columns exactly zero).  No RNG draw is consumed by
    the expansion: the global torch RNG state is saved and restored.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    seed_everything(SEED)
    model = DirectBondFull(payload, "joint", "sum")
    d_init = model.local_tuple.D_loc_raw.detach().clone()
    model.local_tuple = mlpmod.LocalTupleEncoderM(payload, d_init.t().contiguous())
    model.local_tuple.kappa = float(kappa)
    old_rng = torch.get_rng_state()
    old = model.relation_encoder.layers[0]
    if not isinstance(old, torch.nn.Linear) or (old.in_features, old.out_features) != (RELATION_WIDTH, 96):
        raise RuntimeError(f"unexpected source relation first layer: {old}")
    expanded = torch.nn.Linear(RELATION_WIDTH + BOND_CATEGORIES, 96, bias=True)
    torch.set_rng_state(old_rng)
    with torch.no_grad():
        expanded.weight[:, :RELATION_WIDTH].copy_(old.weight)
        expanded.weight[:, RELATION_WIDTH:].zero_()
        expanded.bias.copy_(old.bias)
    model.relation_encoder.layers[0] = expanded
    model.bond_arm = arm
    parameters = int(sum(p.numel() for p in model.parameters()))
    if parameters != EXPANDED_PARAMS:
        raise RuntimeError(f"expected {EXPANDED_PARAMS} parameters, found {parameters}")
    return model


def load_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()}, strict=True)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_state(
    model: torch.nn.Module,
    data_list: Sequence[Any],
    y_local: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    predictions = np.empty(len(data_list), np.float64)
    target = torch.as_tensor(y_local, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(data_list), batch_size):
            indices = list(range(start, min(start + batch_size, len(data_list))))
            batch = zftd.make_batch(data_list, indices, target, device)
            predictions[start:start + len(indices)] = model(batch, mask=cm.C6_MASK).view(-1).detach().cpu().numpy()
    return predictions


# ---------------------------------------------------------------------------
# pre-checks (all fit-only; no dev score is read)
# ---------------------------------------------------------------------------


def pre_checks(*, out_dir: Path = RESULTS_DIR, verify_raw_edges: bool = True) -> dict[str, Any]:
    torch.set_num_threads(8)
    device = torch.device("cpu")
    objects, fit_data, dev_data, beta_info = load_fit_dev(out_dir=out_dir, verify_raw_edges=verify_raw_edges)
    payload = tuplemod.TuplePayload(objects["payload_arrays"])
    kappa = float(objects["kappa"]["kappa_M"])
    fold, targets = objects["fold"], objects["targets"]
    g_fit = np.asarray(targets["g"], np.float64)[fold["fit_idx"]]

    # (1) source M replay on CPU + O/T init equivalence with the source function.
    source = mlpmod.build_arm_mj(payload, kappa_M=kappa).to(device).eval()
    load_state(source, load_source_init(out_dir), device)
    models = {arm: build_arm(arm, payload, kappa).to(device).eval() for arm in ARMS}
    expected_state = expanded_source_state(load_source_init(out_dir))
    for arm, model in models.items():
        load_state(model, expected_state, device)
    probe_ids = list(range(128))
    target_fit_all = torch.as_tensor(g_fit, dtype=torch.float32)
    batch = zftd.make_batch(fit_data, probe_ids, target_fit_all, device)
    with torch.no_grad():
        p_source = source(batch, mask=cm.C6_MASK).view(-1)
        outs = {arm: models[arm](batch, mask=cm.C6_MASK, return_aux=True) for arm in ARMS}
    base = batch.pair_relation[:, list(RELATION_INDICES)]
    base = audit._replace_grouped_columns(base, audit.RELATION_GROUPS, ("path_count",), None, "relation:")
    expected_o = torch.cat((base, torch.zeros(base.shape[0], BOND_CATEGORIES, device=device)), dim=1)
    expected_t = torch.cat((base, batch.pair_beta.to(device)), dim=1)
    init = {
        "source_init_state_hash": state_hash(load_source_init(out_dir)),
        "source_init_hash_matches_frozen": state_hash(load_source_init(out_dir)) == SOURCE_INIT_HASH,
        "expanded_state_hashes": {arm: state_hash({k: v.detach().cpu() for k, v in models[arm].state_dict().items()}) for arm in ARMS},
        "source_prediction_vs_O_max_abs": float((p_source - outs["O"][0]).abs().max()),
        "source_prediction_vs_T_max_abs": float((p_source - outs["T"][0]).abs().max()),
        "O_T_prediction_max_abs": float((outs["O"][0] - outs["T"][0]).abs().max()),
        "O_relation_input_exact": bool(torch.equal(outs["O"][1]["relation_input"], expected_o)),
        "T_relation_input_exact": bool(torch.equal(outs["T"][1]["relation_input"], expected_t)),
        "relation_input_width": int(outs["T"][1]["relation_input"].shape[1]),
        "parameters": {arm: int(sum(p.numel() for p in models[arm].parameters())) for arm in ARMS},
        "O_T_state_identical": state_hash({k: v.detach().cpu() for k, v in models["O"].state_dict().items()})
                              == state_hash({k: v.detach().cpu() for k, v in models["T"].state_dict().items()}),
        "path_bond_mean_columns_absent": int(outs["T"][1]["relation_input"].shape[1]) == RELATION_WIDTH + BOND_CATEGORIES,
    }
    init["all_ok"] = bool(
        init["source_init_hash_matches_frozen"]
        and init["source_prediction_vs_O_max_abs"] <= REPLAY_TOL
        and init["source_prediction_vs_T_max_abs"] <= REPLAY_TOL
        and init["O_T_prediction_max_abs"] <= REPLAY_TOL
        and init["O_relation_input_exact"] and init["T_relation_input_exact"]
        and init["relation_input_width"] == 19 and init["O_T_state_identical"]
        and all(v == EXPANDED_PARAMS for v in init["parameters"].values())
    )
    if not init["all_ok"]:
        raise RuntimeError(f"init/function pre-check failed: {init}")

    # (2)(3) beta consistency: per-graph vs batched, reverse edges, non-adjacent zero,
    #       independent raw-edge verification (already enforced in build_pair_beta).
    first_ids = list(range(8))
    target_fit_all = torch.as_tensor(g_fit, dtype=torch.float32)
    bat = zftd.make_batch(fit_data, first_ids, target_fit_all, device)
    single_beta = _collate_beta([fit_data[i] for i in first_ids])
    bat_beta = bat.pair_beta.clone()
    reordered = [fit_data[i] for i in reversed(first_ids)]
    rev = zftd.make_batch(reordered, list(range(8)), torch.as_tensor(g_fit[:8][::-1].copy(), dtype=torch.float32), device)
    with torch.no_grad():
        p_bat = models["T"](bat, mask=cm.C6_MASK).view(-1)
        p_rev = models["T"](rev, mask=cm.C6_MASK).view(-1).flip(0)
        p_single = torch.cat([
            models["T"](zftd.make_batch(fit_data, [i], target_fit_all, device), mask=cm.C6_MASK).view(-1)
            for i in first_ids
        ])
        categories = torch.unique(torch.argmax(bat_beta[bat_beta.sum(1) > 0], dim=1))
    beta_checks = {
        "beta_concat_equals_per_graph": bool(torch.equal(bat_beta, single_beta)),
        "batch_vs_single_prediction_max_abs": float((p_bat - p_single).abs().max()),
        "reordered_batch_prediction_max_abs": float((p_bat - p_rev).abs().max()),
        "nonadjacent_beta_zero": bool((bat_beta[bat_beta.sum(1) == 0] == 0).all()),
        "adjacent_beta_onehot": bool(torch.all(bat_beta.sum(1)[bat_beta.sum(1) > 0] == 1)),
        "distinct_bond_categories_in_probe": int(categories.numel()),
        "independent_raw_edge_verification": "enforced inside build_pair_beta over all 10000 train graphs",
    }
    if not (beta_checks["beta_concat_equals_per_graph"]
            and beta_checks["batch_vs_single_prediction_max_abs"] <= REPLAY_TOL
            and beta_checks["reordered_batch_prediction_max_abs"] <= REPLAY_TOL
            and beta_checks["nonadjacent_beta_zero"] and beta_checks["adjacent_beta_onehot"]
            and beta_checks["distinct_bond_categories_in_probe"] >= 2):
        raise RuntimeError(f"beta batching pre-check failed: {beta_checks}")

    # (4) real backward: T's new columns get non-zero task gradient, O's are exactly zero.
    grad_checks: dict[str, Any] = {}
    for arm in ARMS:
        model = build_arm(arm, payload, kappa).to(device)
        load_state(model, expected_state, device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        b = zftd.make_batch(fit_data, list(range(64)), target_fit_all, device)
        loss = F.l1_loss(model(b, mask=cm.C6_MASK).view(-1), b.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        grad = model.relation_encoder.layers[0].weight.grad
        grad_checks[arm] = {
            "loss": float(loss.detach()),
            "new_columns_grad_norm": float(grad[:, RELATION_WIDTH:].norm()),
            "old_columns_grad_norm": float(grad[:, :RELATION_WIDTH].norm()),
            "new_columns_exact_zero": bool(float(grad[:, RELATION_WIDTH:].abs().sum()) == 0.0),
        }
        del optimizer, model
    grad_ok = bool(grad_checks["O"]["new_columns_exact_zero"]
                   and grad_checks["T"]["new_columns_grad_norm"] > 0.0
                   and grad_checks["T"]["old_columns_grad_norm"] > 0.0)
    if not grad_ok:
        raise RuntimeError(f"task-gradient pre-check failed: {grad_checks}")

    # (5) pair-relation mutation must not change E; environments/pair composition run once.
    model = build_arm("T", payload, kappa).to(device).eval()
    load_state(model, expected_state, device)
    calls = {"local_environment": 0, "pair_encoder": 0}
    handles = [
        model.local_tuple.register_forward_hook(lambda *_: calls.__setitem__("local_environment", calls["local_environment"] + 1)),
        model.pair_encoder.register_forward_hook(lambda *_: calls.__setitem__("pair_encoder", calls["pair_encoder"] + 1)),
    ]
    try:
        with torch.no_grad():
            pred1, aux1 = model(bat, mask=cm.C6_MASK, return_aux=True)
            changed = copy.copy(bat)
            changed.pair_beta = bat.pair_beta.clone()
            changed.pair_beta[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            pred2, aux2 = model(changed, mask=cm.C6_MASK, return_aux=True)
    finally:
        for handle in handles:
            handle.remove()
    invariance = {
        "environment_once_per_forward": calls["local_environment"] == 2,
        "pair_composition_once_per_forward": calls["pair_encoder"] == 2,
        "E_invariant_to_beta_mutation": bool(torch.equal(aux1["E"], aux2["E"])),
        "beta_input_columns_differ_after_mutation": bool(
            not torch.equal(aux1["relation_input"][:, RELATION_WIDTH:], aux2["relation_input"][:, RELATION_WIDTH:])
        ),
        "prediction_change_at_init_max_abs": float((pred1 - pred2).abs().max()),
        "prediction_change_at_init_is_zero_because_new_columns_are_zero": bool(
            float((pred1 - pred2).abs().max()) == 0.0
        ),
    }
    if not all(value for key, value in invariance.items() if key in (
            "environment_once_per_forward", "pair_composition_once_per_forward",
            "E_invariant_to_beta_mutation", "beta_input_columns_differ_after_mutation",
            "prediction_change_at_init_is_zero_because_new_columns_are_zero")):
        raise RuntimeError(f"pair/environment isolation pre-check failed: {invariance}")

    # (6)(7) recipe witnesses: schedule, construction/training RNG streams, exact fit-only
    #       data order witnesses, and the frozen source fit/dev hashes.
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    seed_everything(SEED)
    probe_model = build_arm("O", payload, kappa)
    build_rng = torch.get_rng_state()
    del probe_model
    seed_everything(SEED)
    train_rng = torch.get_rng_state()
    recipe = {
        "schedule_sha256": schedule_hash,
        "schedule_matches_frozen": schedule_hash == SCHEDULE_SHA,
        "build_rng_sha256": _hash_bytes(build_rng.numpy().tobytes()),
        "build_rng_matches_frozen": _hash_bytes(build_rng.numpy().tobytes()) == BUILD_RNG_SHA,
        "train_rng_sha256": _hash_bytes(train_rng.numpy().tobytes()),
        "train_rng_matches_frozen": _hash_bytes(train_rng.numpy().tobytes()) == TRAIN_RNG_SHA,
        "fit_sha256": _hash_bytes(fold["fit_idx"].astype(np.int64).tobytes()),
        "dev_sha256": _hash_bytes(fold["dev_idx"].astype(np.int64).tobytes()),
        "fit_sha_matches_frozen": _hash_bytes(fold["fit_idx"].astype(np.int64).tobytes()) == FIT_SHA,
        "dev_sha_matches_frozen": _hash_bytes(fold["dev_idx"].astype(np.int64).tobytes()) == DEV_SHA,
        "steps_expected": 15120,
    }
    if not all(recipe[key] for key in (
            "schedule_matches_frozen", "build_rng_matches_frozen", "train_rng_matches_frozen",
            "fit_sha_matches_frozen", "dev_sha_matches_frozen")):
        raise RuntimeError(f"recipe witness failed: {recipe}")

    # Bootstrap implementation witnesses, frozen before O/T dev scores.
    witness_o = np.asarray([0.2, 0.9, 1.1, 0.1, 0.7, 0.4], np.float64)
    witness_t = np.asarray([0.1, 1.0, 0.8, 0.3, 0.5, 0.6], np.float64)
    witness_mask = np.ones(6, dtype=bool)
    identical_witness = _paired(witness_o, witness_o.copy(), witness_mask, BOOT_SEED)
    forward_witness = _paired(witness_o, witness_t, witness_mask, BOOT_SEED)
    reverse_witness = _paired(witness_t, witness_o, witness_mask, BOOT_SEED)
    bootstrap_checks = {
        "identical_gain_zero": identical_witness["point"] == 0.0,
        "identical_ci_exact_zero": identical_witness["ci_low"] == 0.0 and identical_witness["ci_high"] == 0.0,
        "swap_point_mirror": np.isclose(forward_witness["point"], -reverse_witness["point"], atol=1e-15, rtol=0.0),
        "swap_ci_mirror": np.isclose(forward_witness["ci_low"], -reverse_witness["ci_high"], atol=1e-15, rtol=0.0)
                           and np.isclose(forward_witness["ci_high"], -reverse_witness["ci_low"], atol=1e-15, rtol=0.0),
    }
    if not all(bootstrap_checks.values()):
        raise RuntimeError(f"paired bootstrap witnesses failed: {bootstrap_checks}")

    result = {
        "init": init, "beta_checks": beta_checks, "gradient_checks": grad_checks,
        "invariance_checks": invariance, "recipe_witnesses": recipe,
        "beta_meta": {k: v for k, v in beta_info["meta"].items() if k != "graph_checks"},
        "relation_indices": list(RELATION_INDICES),
        "source_M_init_hash": SOURCE_INIT_HASH,
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    write_json(out_dir / "beta_provenance.json", beta_info["meta"])
    encoded_train = Path(p1run.sdp.CACHE_DIR) / "encoded_train.pt"
    env_train = p1run.CACHE_DIR / "env_train.pt"
    raw_files = sorted((Path(zpp.REPO_ROOT) / "data/ZINC/raw").glob("train*"))
    source_files = [
        "M_init_state.pt", "M_raw_soup_state.pt", "M_meta.json", "fresh_fold.npz",
        "fresh_targets.npz", "fresh_tuple_payload.npz", "fresh_prep.npz", "fresh_manifest.json",
    ]
    input_manifest = {
        "round": PROTOCOL_VERSION,
        "source_round": str(SOURCE_DIR), "source_training_commit": "a5400de",
        "source_prereg_commit": "2e4ee3f",
        "source_files": {name: {"path": str(SOURCE_DIR / name), "sha256": file_sha256(SOURCE_DIR / name)}
                         for name in source_files},
        "fresh_fold_sha256": {"fit": FIT_SHA, "dev": DEV_SHA},
        "position_schedule_sha256": SCHEDULE_SHA, "global_gid_stream_sha256": GID_STREAM_SHA,
        "train_only_cache_files": {
            str(encoded_train): file_sha256(encoded_train),
            str(env_train): file_sha256(env_train),
        },
        "raw_zinc_data_files": {str(path): file_sha256(path) for path in raw_files},
        "beta_sha256": array_sha256(np.concatenate([b.numpy() for b in beta_info["betas"]]), np.float32),
        "relation_indices": list(RELATION_INDICES), "beta_raw_columns": [19, 23],
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    write_json(out_dir / "input_manifest.json", input_manifest)
    write_json(out_dir / "input_checks.json", result)
    return result


# ---------------------------------------------------------------------------
# smoke (fit-only; states discarded)
# ---------------------------------------------------------------------------


def smoke(*, device: torch.device, out_dir: Path = RESULTS_DIR, steps: int = 3) -> dict[str, Any]:
    torch.set_num_threads(4 if device.type == "cuda" else 8)
    objects, fit_data, _, _ = load_fit_dev(out_dir=out_dir)
    fold, targets = objects["fold"], objects["targets"]
    payload = tuplemod.TuplePayload(objects["payload_arrays"])
    kappa = float(objects["kappa"]["kappa_M"])
    g_fit = np.asarray(targets["g"], np.float64)[fold["fit_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    source_state = load_source_init(out_dir)
    results: dict[str, Any] = {}
    for arm in ARMS:
        seed_everything(SEED)
        model = build_arm(arm, payload, kappa)
        build_rng = torch.get_rng_state()
        load_state(model, expanded_source_state(source_state), torch.device("cpu"))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        model.train()
        losses, grad_new, grad_old, norms = [], [], [], []
        indices = [int(i) for i in schedule[0][:BATCH_SIZE].tolist()]
        for _ in range(int(steps)):
            batch = zftd.make_batch(fit_data, indices, target_fit, device)
            prediction = model(batch, mask=cm.C6_MASK)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            grad = model.relation_encoder.layers[0].weight.grad
            grad_new.append(float(grad[:, RELATION_WIDTH:].norm()))
            grad_old.append(float(grad[:, :RELATION_WIDTH].norm()))
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)))
            optimizer.step()
            losses.append(float(loss.detach()))
        checks = {
            "losses": losses, "finite": all(math.isfinite(v) for v in losses + norms),
            "new_columns_grad_norms": grad_new, "old_columns_grad_norms": grad_old,
            "O_new_columns_exact_zero": bool(arm != "O" or all(v == 0.0 for v in grad_new)),
            "T_new_columns_nonzero": bool(arm != "T" or all(v > 0.0 for v in grad_new)),
            "new_columns_weight_norm_after_smoke": float(model.relation_encoder.layers[0].weight[:, RELATION_WIDTH:].detach().norm()),
            "build_rng_sha256": _hash_bytes(build_rng.numpy().tobytes()),
            "build_rng_matches_frozen": _hash_bytes(build_rng.numpy().tobytes()) == BUILD_RNG_SHA,
            "states_discarded": True, "runtime": _runtime(device),
        }
        if (not checks["finite"] or not checks["O_new_columns_exact_zero"]
                or not checks["T_new_columns_nonzero"] or not checks["build_rng_matches_frozen"]):
            raise RuntimeError(f"smoke failed for arm {arm}: {checks}")
        results[arm] = checks
        del optimizer, model
    result = {
        "arms": results, "device": str(device), "fit_only": True,
        "schedule_sha256": schedule_hash, "all_finite": True,
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    write_json(out_dir / "smoke_checks.json", result)
    return result


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def train_arm(arm: str, *, device: torch.device, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(4 if device.type == "cuda" else 8)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects, fit_data, dev_data, beta_info = load_fit_dev(out_dir=out_dir)
    fold, targets = objects["fold"], objects["targets"]
    payload = tuplemod.TuplePayload(objects["payload_arrays"])
    kappa = float(objects["kappa"]["kappa_M"])
    g = np.asarray(targets["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)

    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    if schedule_hash != SCHEDULE_SHA:
        raise RuntimeError(f"schedule hash {schedule_hash} != frozen {SCHEDULE_SHA}")

    seed_everything(SEED)
    model = build_arm(arm, payload, kappa)
    build_rng = torch.get_rng_state()
    model = model.to(device)
    seed_everything(SEED)
    train_rng = torch.get_rng_state()
    if _hash_bytes(build_rng.numpy().tobytes()) != BUILD_RNG_SHA:
        raise RuntimeError("construction RNG differs from the frozen source recipe")
    if _hash_bytes(train_rng.numpy().tobytes()) != TRAIN_RNG_SHA:
        raise RuntimeError("training RNG differs from the frozen source recipe")

    source_state = load_source_init(out_dir)
    expected_state = expanded_source_state(source_state)
    load_state(model, expected_state, torch.device("cpu"))
    model = model.to(device)
    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if any(not torch.equal(init_state[key], value) for key, value in expected_state.items()):
        raise RuntimeError(f"{arm} init state is not source M init with exact zero new columns")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probe_log: list[dict[str, Any]] = []
    id_stream = hashlib.sha256()
    gid_stream = hashlib.sha256()
    steps_done = 0
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, n_mol, n_steps, gnorm_sum, clip_hits = 0.0, 0, 0, 0.0, 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = schedule[epoch - 1][start:start + BATCH_SIZE]
            index_list = [int(i) for i in indices.tolist()]
            id_stream.update(np.asarray(index_list, np.int64).tobytes())
            gid_stream.update(np.asarray(targets["gid"][fold["fit_idx"][index_list]], np.int64).tobytes())
            batch = zftd.make_batch(fit_data, index_list, target_fit, device)
            prediction = model(batch, mask=cm.C6_MASK)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{arm} non-finite loss at epoch {epoch} step {n_steps}")
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                grad = model.relation_encoder.layers[0].weight.grad
                probe_log.append({
                    "epoch": int(epoch), "step_in_epoch": 1, "clip_total_norm": total_norm,
                    "new_relation_columns_grad_norm": float(grad[:, RELATION_WIDTH:].norm()),
                    "old_relation_columns_grad_norm": float(grad[:, :RELATION_WIDTH].norm()),
                    "A_grad_norm": float(model.local_tuple.A_raw.grad.norm()) if model.local_tuple.A_raw.grad is not None else None,
                    "W_loc_grad_norm": float(model.local_tuple.W_loc.grad.norm()) if model.local_tuple.W_loc.grad is not None else None,
                })
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
        curve.append({
            "epoch": epoch, "train_task_mae": task_sum / max(n_mol, 1),
            "grad_norm": gnorm_sum / max(n_steps, 1), "clip_fraction": clip_hits / max(n_steps, 1),
            "seconds": time.perf_counter() - epoch_started,
        })
        if epoch in soup_epochs:
            soup[epoch] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS):
            log(f"[{arm}] epoch={epoch:03d} train_mae={curve[-1]['train_task_mae']:.6f} "
                f"sec={curve[-1]['seconds']:.1f}")
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION, "arm": arm, "seed": SEED,
        "epochs": EPOCHS, "steps_done": steps_done, "steps_expected": 15120,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE, "soup_epochs": members, "n_fit": N_FIT, "n_dev": N_DEV,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "supervision": "g = y - c (fit-only frozen constants)",
        "fold_sha256": {"fit": FIT_SHA, "dev": DEV_SHA},
        "init_state_hash": state_hash(init_state),
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "source_M_init_state_hash": SOURCE_INIT_HASH,
        "build_rng_sha256": _hash_bytes(build_rng.numpy().tobytes()),
        "train_rng_sha256": _hash_bytes(train_rng.numpy().tobytes()),
        "schedule_sha256": schedule_hash,
        "position_stream_sha256": id_stream.hexdigest(),
        "global_gid_stream_sha256": gid_stream.hexdigest(),
        "curve": curve, "probe_log": probe_log,
        "wall_clock_s": float(time.perf_counter() - started),
        "runtime": _runtime(device),
        "beta_sha256": array_sha256(np.concatenate([b.numpy() for b in beta_info["betas"]]), np.float32),
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    if steps_done != 15120:
        raise RuntimeError(f"{arm} steps {steps_done} != 15120")
    if gid_stream.hexdigest() != GID_STREAM_SHA:
        raise RuntimeError(f"{arm} global graph-id stream {gid_stream.hexdigest()} != frozen {GID_STREAM_SHA}")

    torch.save(init_state, out_dir / f"{arm}_init_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")

    raw_predictions: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = build_arm(arm, payload, kappa)
        load_state(replay, state, torch.device("cpu"))
        replay = replay.to(device)
        raw_predictions[f"{state_name}_fit"] = evaluate_state(replay, fit_data, g_fit, device)
        raw_predictions[f"{state_name}_dev"] = evaluate_state(replay, dev_data, g[fold["dev_idx"]], device)
    np.savez_compressed(
        out_dir / f"{arm}_raw_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_predictions.items()},
    )
    result["calibration_b"] = {
        "init": float(np.median(g_fit - raw_predictions["init_fit"])),
        "last": float(np.median(g_fit - raw_predictions["last_fit"])),
        "raw_soup": float(np.median(g_fit - raw_predictions["raw_soup_fit"])),
    }

    replay = build_arm(arm, payload, kappa)
    load_state(replay, soup_state, torch.device("cpu"))
    cpu_pred = evaluate_state(replay, fit_data[:128], g_fit[:128], torch.device("cpu"))
    if device.type == "cuda":
        gpu_pred = evaluate_state(replay.to(device), fit_data[:128], g_fit[:128], device)
        result["cpu_gpu_replay_128_max_abs"] = float(np.max(np.abs(cpu_pred - gpu_pred)))
        if result["cpu_gpu_replay_128_max_abs"] > REPLAY_TOL:
            raise RuntimeError(f"{arm} CPU/GPU replay maxdiff {result['cpu_gpu_replay_128_max_abs']} > {REPLAY_TOL}")
    else:
        result["cpu_gpu_replay_128_max_abs"] = 0.0
    result["fit_path_diagnostic"] = _fit_path_diagnostic(replay.to(device), fit_data, g_fit, device)
    write_json(out_dir / f"{arm}_meta.json", result)
    write_json(out_dir / f"{arm}_curve.json", curve)
    write_json(out_dir / f"{arm}_probe.json", probe_log)
    write_json(out_dir / f"{arm}_gpu_runtime.json", result["runtime"])
    log(f"[{arm}] done steps={steps_done} wall={result['wall_clock_s']:.1f}s "
        f"b_raw_soup={result['calibration_b']['raw_soup']:.6f}")
    return result


def _fit_path_diagnostic(model: torch.nn.Module, data_list: Sequence[Any], y_local: np.ndarray,
                         device: torch.device) -> dict[str, Any]:
    """Fit-only structural witness: E isolated from beta, environments/pairs composed once."""
    model.eval()
    batch = zftd.make_batch(data_list, list(range(128)), torch.as_tensor(y_local, dtype=torch.float32), device)
    calls = {"local_environment": 0, "pair_encoder": 0}
    handles = [
        model.local_tuple.register_forward_hook(lambda *_: calls.__setitem__("local_environment", calls["local_environment"] + 1)),
        model.pair_encoder.register_forward_hook(lambda *_: calls.__setitem__("pair_encoder", calls["pair_encoder"] + 1)),
    ]
    try:
        with torch.no_grad():
            _p1, aux1 = model(batch, mask=cm.C6_MASK, return_aux=True)
            changed = copy.copy(batch)
            changed.pair_beta = batch.pair_beta.clone()
            if changed.pair_beta.numel():
                changed.pair_beta[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            _p2, aux2 = model(changed, mask=cm.C6_MASK, return_aux=True)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "environment_once_per_forward": calls["local_environment"] == 2,
        "pair_composition_once_per_forward": calls["pair_encoder"] == 2,
        "E_invariant_to_pair_beta": bool(torch.equal(aux1["E"], aux2["E"])),
        "relation_input_width": int(aux1["relation_input"].shape[1]),
        "beta_input_matches_data_for_T": bool(torch.equal(aux1["beta_input"], batch.pair_beta)),
    }


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _metrics(pred_fit: np.ndarray, pred_dev: np.ndarray, g_fit: np.ndarray,
             g_dev: np.ndarray, k_fit: np.ndarray, k_dev: np.ndarray) -> dict[str, Any]:
    bias = float(np.median(g_fit - pred_fit))
    cal_fit = pred_fit + bias
    cal_dev = pred_dev + bias
    fit_g0 = k_fit == 0
    dev_g0 = k_dev == 0
    return {
        "bias": bias,
        "fit_overall_raw_mae": float(np.mean(np.abs(g_fit - pred_fit))),
        "fit_overall_cal_mae": float(np.mean(np.abs(g_fit - cal_fit))),
        "dev_overall_raw_mae": float(np.mean(np.abs(g_dev - pred_dev))),
        "dev_overall_cal_mae": float(np.mean(np.abs(g_dev - cal_dev))),
        "fit_G0_raw_mae": float(np.mean(np.abs((g_fit - pred_fit)[fit_g0]))),
        "fit_G0_cal_mae": float(np.mean(np.abs((g_fit - cal_fit)[fit_g0]))),
        "dev_G0_raw_mae": float(np.mean(np.abs((g_dev - pred_dev)[dev_g0]))),
        "dev_G0_cal_mae": float(np.mean(np.abs((g_dev - cal_dev)[dev_g0]))),
        "fit_overall_raw_to_cal_change": float(np.mean(np.abs(g_fit - cal_fit)) - np.mean(np.abs(g_fit - pred_fit))),
        "fit_G0_raw_to_cal_change": float(np.mean(np.abs((g_fit - cal_fit)[fit_g0])) - np.mean(np.abs((g_fit - pred_fit)[fit_g0]))),
        "dev_overall_raw_to_cal_change": float(np.mean(np.abs(g_dev - cal_dev)) - np.mean(np.abs(g_dev - pred_dev))),
        "dev_G0_raw_to_cal_change": float(np.mean(np.abs((g_dev - cal_dev)[dev_g0])) - np.mean(np.abs((g_dev - pred_dev)[dev_g0]))),
        "dev_fit_overall_raw_gap": float(np.mean(np.abs(g_dev - pred_dev)) - np.mean(np.abs(g_fit - pred_fit))),
        "dev_fit_overall_cal_gap": float(np.mean(np.abs(g_dev - cal_dev)) - np.mean(np.abs(g_fit - cal_fit))),
        "dev_fit_G0_raw_gap": float(np.mean(np.abs((g_dev - pred_dev)[dev_g0])) - np.mean(np.abs((g_fit - pred_fit)[fit_g0]))),
        "dev_fit_G0_cal_gap": float(np.mean(np.abs((g_dev - cal_dev)[dev_g0])) - np.mean(np.abs((g_fit - cal_fit)[fit_g0]))),
        "err_fit_raw": np.abs(g_fit - pred_fit), "err_fit_cal": np.abs(g_fit - cal_fit),
        "err_dev_raw": np.abs(g_dev - pred_dev), "err_dev_cal": np.abs(g_dev - cal_dev),
    }


def _group_rows(err: np.ndarray, k: np.ndarray) -> list[dict[str, Any]]:
    masks = (k == 0, k == -1, k == -2, k <= -3)
    rows = []
    for name, mask in zip(GROUP_NAMES, masks):
        rows.append({
            "group": name, "n": int(mask.sum()),
            "mae": float(err[mask].mean()) if mask.any() else None,
            "sum_abs_error": float(err[mask].sum()),
            "contribution": float(err[mask].sum() / len(k)),
        })
    if not math.isclose(sum(x["contribution"] for x in rows), float(err.mean()), abs_tol=1e-10):
        raise RuntimeError("group contributions do not sum to the total MAE")
    return rows


def _paired(err_o: np.ndarray, err_t: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
    a, b = err_o[mask], err_t[mask]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)), endpoint=False)
    draws = a[idx].mean(1) - b[idx].mean(1)
    point = float(a.mean() - b.mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    identical = a - a
    swap_draws = b[idx].mean(1) - a[idx].mean(1)
    swap_lo, swap_hi = np.percentile(swap_draws, [2.5, 97.5])
    zero_draws = np.zeros(N_BOOT, np.float64)
    mirror_ok = bool(np.isclose(swap_lo, -hi, rtol=0.0, atol=1e-15)
                     and np.isclose(swap_hi, -lo, rtol=0.0, atol=1e-15))
    return {
        "point": point, "ci_low": float(lo), "ci_high": float(hi), "n": int(len(a)),
        "O_mae": float(a.mean()), "T_mae": float(b.mean()),
        "bootstrap_witnesses": {
            "identical_zero_point": float(identical.mean()),
            "identical_zero_ci": [float(np.percentile(zero_draws, 2.5)), float(np.percentile(zero_draws, 97.5))],
            "identical_zero_ci_exact": bool(np.array_equal(zero_draws, np.zeros_like(zero_draws))),
            "swap_mirror_point": float(-point),
            "swap_mirror_ci": [float(swap_lo), float(swap_hi)],
            "swap_mirror_matches_negated_ci": mirror_ok,
        },
    }


def analyze(*, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    torch.set_num_threads(8)
    objects = fresh.load_fresh_objects(out_dir)
    fold, targets = objects["fold"], objects["targets"]
    g = np.asarray(targets["g"], np.float64)
    k = np.asarray(targets["k"], np.int64)
    g_fit = g[fold["fit_idx"]]
    g_dev = g[fold["dev_idx"]]
    k_fit = k[fold["fit_idx"]]
    k_dev = k[fold["dev_idx"]]
    arms: dict[str, Any] = {}
    for arm in ARMS:
        meta = json.loads((out_dir / f"{arm}_meta.json").read_text())
        with np.load(out_dir / f"{arm}_raw_predictions.npz", allow_pickle=False) as z:
            predictions = {key: z[key].astype(np.float64) for key in z.files}
        arms[arm] = {
            "meta": meta, "predictions": predictions,
            "metrics": _metrics(predictions["raw_soup_fit"], predictions["raw_soup_dev"], g_fit, g_dev, k_fit, k_dev),
        }

    endpoints = (
        ("G0_cal", "err_dev_cal", k_dev == 0),
        ("overall_cal", "err_dev_cal", np.ones(len(k_dev), bool)),
        ("G0_raw", "err_dev_raw", k_dev == 0),
        ("overall_raw", "err_dev_raw", np.ones(len(k_dev), bool)),
    )
    gains = {
        name: _paired(arms["O"]["metrics"][key], arms["T"]["metrics"][key], mask, BOOT_SEED)
        for name, key, mask in endpoints
    }
    gate = {
        "G0_cal_gain_ge_0.003": gains["G0_cal"]["point"] >= DELTA,
        "overall_cal_gain_ge_0.003": gains["overall_cal"]["point"] >= DELTA,
        "G0_cal_CI_lower_gt_0": gains["G0_cal"]["ci_low"] > 0.0,
    }
    passed = all(gate.values())
    if passed:
        classification = "DIRECT_BOND_INPUT_CANDIDATE"
    elif gains["G0_cal"]["point"] > 0.0 and gains["overall_cal"]["point"] > 0.0:
        classification = "DIRECTIONAL_NOT_CONFIRMED"
    elif gains["G0_cal"]["point"] < 0.0 and gains["overall_cal"]["point"] < 0.0:
        classification = "NO_GAIN_FOR_THIS_INTERFACE"
    else:
        classification = "DIRECTIONAL_NOT_CONFIRMED"
    calibration_qualification = (
        "CALIBRATION_DEPENDENT" if passed and
        (gains["G0_raw"]["point"] <= 0.0 or gains["overall_raw"]["point"] <= 0.0)
        else "RAW_DIRECTIONAL_CONSISTENT" if passed else "NOT_APPLICABLE"
    )
    practical_equivalence = bool(
        -DELTA <= gains["G0_cal"]["ci_low"] and gains["G0_cal"]["ci_high"] <= DELTA
        and -DELTA <= gains["overall_cal"]["ci_low"] and gains["overall_cal"]["ci_high"] <= DELTA
    )
    
    group = {arm: _group_rows(arms[arm]["metrics"]["err_dev_cal"], k_dev) for arm in ARMS}
    group_gain = []
    for index, name in enumerate(GROUP_NAMES):
        o_row, t_row = group["O"][index], group["T"][index]
        group_gain.append({
            "group": name, "n": o_row["n"],
            "gain_contribution": o_row["contribution"] - t_row["contribution"],
            "O_sum_abs_error": o_row["sum_abs_error"], "T_sum_abs_error": t_row["sum_abs_error"],
        })

    cal_gain = gains["overall_cal"]["point"]
    raw_gain = gains["overall_raw"]["point"]
    identity = raw_gain + (
        (arms["T"]["metrics"]["dev_overall_raw_mae"] - arms["T"]["metrics"]["dev_overall_cal_mae"])
        - (arms["O"]["metrics"]["dev_overall_raw_mae"] - arms["O"]["metrics"]["dev_overall_cal_mae"])
    )
    worst = int(np.argmax(arms["O"]["metrics"]["err_dev_cal"]))
    keep = np.arange(len(k_dev)) != worst
    sensitivity = {
        "removed_dev_position": worst, "removed_global_train_id": int(fold["dev_idx"][worst]),
        "O_cal_mae": float(arms["O"]["metrics"]["err_dev_cal"][keep].mean()),
        "T_cal_mae": float(arms["T"]["metrics"]["err_dev_cal"][keep].mean()),
        "gain": float(arms["O"]["metrics"]["err_dev_cal"][keep].mean() - arms["T"]["metrics"]["err_dev_cal"][keep].mean()),
    }

    mechanism: dict[str, Any] = {"status": "NOT_RUN_NO_CANDIDATE", "reason": "frozen three-part gate not met"}
    if passed:
        mechanism = _marginal_beta_diagnostic(out_dir, objects, g_dev, arms["T"]["predictions"]["raw_soup_dev"])

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "classification": classification,
        "calibration_qualification": calibration_qualification,
        "practical_equivalence_within_0.003": practical_equivalence,
        "candidate_gate": {"conditions": gate, "pass": passed},
        "primary_gain": gains["G0_cal"], "overall_cal_gain": gains["overall_cal"],
        "raw_gain": {"G0": gains["G0_raw"], "overall": gains["overall_raw"]},
        "arms": {
            arm: {key: value for key, value in arms[arm]["metrics"].items() if not isinstance(value, np.ndarray)}
            for arm in ARMS
        },
        "calibration_qualification": calibration_qualification,
        "practical_equivalence_within_0.003": practical_equivalence,
        "gains": gains, "group_table": group, "group_gain_contributions": group_gain,
        "group_gain_sum": float(sum(x["gain_contribution"] for x in group_gain)),
        "raw_cal_accounting_identity": float(identity),
        "calibration_dependence_accounting_only": float(cal_gain - raw_gain),
        "sensitivity_drop_O_worst": sensitivity,
        "mechanism_diagnostic": mechanism,
        "background_source_M": {
            "dev_G0_cal": 0.0973826389, "dev_overall_cal": 0.0993037399,
            "note": "source round M (15-D relation, no direct-bond input) on the same frozen fold",
        },
        "metric_semantics": "internal g = y - c MAE; NOT the official ZINC y score",
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    if abs(summary["group_gain_sum"] - cal_gain) > 1e-9:
        raise RuntimeError("group gain contributions do not equal the overall calibrated gain")
    if abs(identity - cal_gain) > 1e-9:
        raise RuntimeError("raw/cal accounting identity failed")
    summary["bootstrap_witnesses_pass"] = all(
        gains[name]["bootstrap_witnesses"]["identical_zero_point"] == 0.0
        and gains[name]["bootstrap_witnesses"]["identical_zero_ci_exact"]
        and gains[name]["bootstrap_witnesses"]["swap_mirror_matches_negated_ci"]
        for name in gains
    )
    if not summary["bootstrap_witnesses_pass"]:
        raise RuntimeError("paired bootstrap zero/swap witness failed")
    write_json(out_dir / "analysis.json", summary)
    write_json(out_dir / "gains.json", gains)
    write_json(out_dir / "gate.json", {
        "classification": classification, "candidate_gate": summary["candidate_gate"], "gains": gains,
    })

    rows = []
    fit_rows = {arm: _group_rows(arms[arm]["metrics"]["err_fit_cal"], k_fit) for arm in ARMS}
    for arm in ARMS:
        m = arms[arm]["metrics"]
        rows.append({
            "arm": arm, "fit_overall_raw": m["fit_overall_raw_mae"], "fit_overall_cal": m["fit_overall_cal_mae"],
            "fit_G0_raw": m["fit_G0_raw_mae"], "fit_G0_cal": m["fit_G0_cal_mae"],
            "dev_G0_raw": m["dev_G0_raw_mae"], "dev_G0_cal": m["dev_G0_cal_mae"],
            "dev_overall_raw": m["dev_overall_raw_mae"], "dev_overall_cal": m["dev_overall_cal_mae"],
            "dev_overall_raw_gap": m["dev_fit_overall_raw_gap"], "dev_overall_cal_gap": m["dev_fit_overall_cal_gap"],
            "dev_G0_raw_gap": m["dev_fit_G0_raw_gap"], "dev_G0_cal_gap": m["dev_fit_G0_cal_gap"],
            "raw_to_cal_fit_overall": m["fit_overall_raw_to_cal_change"], "raw_to_cal_fit_G0": m["fit_G0_raw_to_cal_change"],
            "raw_to_cal_dev_overall": m["dev_overall_raw_to_cal_change"], "raw_to_cal_dev_G0": m["dev_G0_raw_to_cal_change"],
            "bias": m["bias"],
        })
    _write_csv(out_dir / "main_table.csv", rows)
    _write_csv(out_dir / "fit_group_table.csv", [{"arm": arm, **row} for arm in ARMS for row in fit_rows[arm]])
    _write_csv(out_dir / "group_table.csv", [{"arm": arm, **row} for arm in ARMS for row in group[arm]])
    _write_csv(out_dir / "group_gain_table.csv", group_gain)
    _write_csv(out_dir / "per_graph_dev.csv", [
        {
            "dev_position": i, "global_train_id": int(fold["dev_idx"][i]), "k": int(k_dev[i]),
            "g": float(g_dev[i]), "O_raw": float(arms["O"]["predictions"]["raw_soup_dev"][i]),
            "T_raw": float(arms["T"]["predictions"]["raw_soup_dev"][i]),
            "O_cal_error": float(arms["O"]["metrics"]["err_dev_cal"][i]),
            "T_cal_error": float(arms["T"]["metrics"]["err_dev_cal"][i]),
        } for i in range(len(k_dev))
    ])
    return summary


def _marginal_beta_diagnostic(out_dir: Path, objects: Mapping[str, Any], g_dev: np.ndarray,
                              stored_native: np.ndarray) -> dict[str, Any]:
    """Label-free input-only intervention on T's soup (run only if the gate passes)."""
    fold = objects["fold"]
    payload = tuplemod.TuplePayload(objects["payload_arrays"])
    kappa = float(objects["kappa"]["kappa_M"])
    _, _, dev_data, _ = load_fit_dev(out_dir=out_dir)
    device = torch.device("cpu")
    model = build_arm("T", payload, kappa)
    load_state(model, torch.load(out_dir / "T_raw_soup_state.pt", map_location="cpu", weights_only=False), device)
    native = evaluate_state(model, dev_data, g_dev, device)
    raw_train = zlr._load_zinc(Path(zpp.REPO_ROOT) / "data/ZINC", "train")
    marginal_data, edge_type_means = [], []
    for position, gid in enumerate(fold["dev_idx"]):
        edges, _ = _edge_map(raw_train[int(gid)])
        counts = np.zeros(BOND_CATEGORIES, np.float64)
        for typ in edges.values():
            counts[int(typ)] += 1.0
        mean = (counts / counts.sum()).astype(np.float32) if counts.sum() else np.zeros(BOND_CATEGORIES, np.float32)
        item = copy.copy(dev_data[position])
        beta = item.pair_beta.clone()
        adjacent = beta.sum(1) > 0
        beta[adjacent] = torch.as_tensor(mean).expand(int(adjacent.sum()), -1)
        item.pair_beta = beta
        marginal_data.append(item)
        edge_type_means.append(mean.tolist())
    marginal = evaluate_state(model, marginal_data, g_dev, device)
    result = {
        "status": "COMPLETED",
        "native_replay_max_abs_vs_stored": float(np.max(np.abs(native - stored_native))),
        "native_dev_mae": float(np.mean(np.abs(g_dev - native))),
        "graph_marginal_dev_mae": float(np.mean(np.abs(g_dev - marginal))),
        "prediction_change_mae": float(np.mean(np.abs(native - marginal))),
        "per_graph_edge_type_means": edge_type_means,
        "interpretation_boundary": (
            "label-free input-only forward intervention; it preserves each graph's undirected bond-type "
            "marginal and adjacent-pair count but is not an end-to-end matched-marginal retraining and "
            "does not by itself separate pair-specific from graph-marginal use of beta"
        ),
    }
    np.savez_compressed(
        out_dir / "beta_marginal_dev_diagnostic.npz",
        native=native.astype(np.float32), marginal=marginal.astype(np.float32),
        edge_type_means=np.asarray(edge_type_means, np.float32),
    )
    write_json(out_dir / "beta_marginal_dev_diagnostic.json", result)
    return result


# ---------------------------------------------------------------------------
# replay + manifest
# ---------------------------------------------------------------------------


def replay(*, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    objects, fit_data, dev_data, _ = load_fit_dev(out_dir=out_dir)
    fold, targets = objects["fold"], objects["targets"]
    g = np.asarray(targets["g"], np.float64)
    payload = tuplemod.TuplePayload(objects["payload_arrays"])
    kappa = float(objects["kappa"]["kappa_M"])
    result: dict[str, Any] = {}
    for arm, data_list, name, index in (
        ("O", fit_data, "fit", fold["fit_idx"]), ("T", fit_data, "fit", fold["fit_idx"]),
        ("O", dev_data, "dev", fold["dev_idx"]), ("T", dev_data, "dev", fold["dev_idx"]),
    ):
        model = build_arm(arm, payload, kappa)
        load_state(model, torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False),
                   torch.device("cpu"))
        predictions = evaluate_state(model, data_list[:128], g[index][:128], torch.device("cpu"))
        with np.load(out_dir / f"{arm}_raw_predictions.npz", allow_pickle=False) as z:
            stored = z[f"raw_soup_{name}"].astype(np.float64)[:128]
        diff = float(np.max(np.abs(predictions - stored)))
        result[f"{arm}_{name}"] = {"max_abs": diff, "within_tol": bool(diff <= REPLAY_TOL)}
    result["all_ok"] = all(v["within_tol"] for v in result.values() if isinstance(v, Mapping))
    write_json(out_dir / "replay_checks.json", result)
    if not result["all_ok"]:
        raise RuntimeError(f"replay failed: {result}")
    return result


def write_manifest(*, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    files = [p for p in out_dir.iterdir() if p.is_file() and p.name != "manifest.json"]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "artifacts": {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in sorted(files)},
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pre-checks", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--skip-raw-edge-verification", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.prepare:
        prepare_inputs(out_dir=args.out)
    if args.pre_checks:
        pre_checks(out_dir=args.out, verify_raw_edges=not args.skip_raw_edge_verification)
    if args.smoke:
        smoke(device=device, out_dir=args.out, steps=args.smoke_steps)
    if args.train:
        if not args.arm:
            parser.error("--train requires --arm O|T")
        train_arm(args.arm, device=device, out_dir=args.out)
    if args.analyze:
        analyze(out_dir=args.out)
    if args.replay:
        replay(out_dir=args.out)
    if args.manifest:
        write_manifest(out_dir=args.out)
    if not any((args.prepare, args.pre_checks, args.smoke, args.train, args.analyze, args.replay, args.manifest)):
        parser.error("select an action")
    print(json.dumps({"seconds": time.perf_counter() - started, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
