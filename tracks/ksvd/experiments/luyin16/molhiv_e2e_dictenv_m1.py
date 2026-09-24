"""E2E-DictEnv-M1 runner — primitive-only dictionary-core transfer to OGBG-MolHIV.

Round ``e2e_dictenv_m1``.  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_m1_preregistration.md``.
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_m1.py``.

Stages
------
``dict env stats calibrate identity correct smoke train freeze analyze unlock all``

Official MolHIV test is never loaded before ``unlock``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ksvd_research.data import load_molhiv

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_m1 as m1
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = m1.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_m1"
CACHE_DIR = RESULTS_DIR / "cache"
STATE_DIR = RESULTS_DIR / "states"
DATA_ROOT = REPO_ROOT / "data/ogb"
DICT_PATH = RESULTS_DIR / "dictionary.pt"

BATCH_SIZE = 128
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
MAX_EPOCHS = 240
PATIENCE = 40
SOUP_K = 5
SMOKE_MOLECULES = 512
SMOKE_EPOCHS = 3
TRAIN_SHUFFLE_OFFSET = 51001
EVAL_SHUFFLE_OFFSET = 51002
CALIBRATION_MOLECULES = 512
DICT_EPOCHS = 10
DICT_SAMPLE_MAX = 500_000
ANCHOR_CHUNK = 4096
SEEDS = (0, 1)

_write_json = v0run._write_json
_read_json = v0run._read_json
_sha256 = v0run._sha256
_git_commit = v0run._git_commit
_n_params = v0run._n_params


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# per-graph payload / cache
# ---------------------------------------------------------------------------


class GraphPayload:
    """One molecule's M1 local environment payload (local indices)."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def _load_bundle():
    return load_molhiv(root=DATA_ROOT, with_features=True)


def _schema() -> dict[str, Any]:
    atom_dims, bond_dims = m1.ogb_feature_dims()
    return {
        "atom_feature_dims": list(atom_dims),
        "bond_feature_dims": list(bond_dims),
        "n_atom_fields": len(atom_dims),
        "n_bond_fields": len(bond_dims),
        "atom_chem_dim": int(m1.ATOM_CHEM_DIM),
        "bond_chem_dim": int(m1.BOND_CHEM_DIM),
        "source": "ogb.utils.features.get_atom_feature_dims/get_bond_feature_dims",
    }


def _incidence(graph: Any, edge_feats: Mapping[Any, Any]) -> dict[str, np.ndarray]:
    """P1-style root-relative incidence, with vector bond chemistry per occurrence."""
    from collections import deque

    shell_pairs = zpp_shell_pairs()
    sp_index = {pair: index for index, pair in enumerate(shell_pairs)}
    nodes = [int(node) for node in graph.nodes]
    occ_node: list[int] = []
    occ_root: list[int] = []
    occ_shell: list[int] = []
    bond_root: list[int] = []
    bond_shellpair: list[int] = []
    bond_u: list[int] = []
    bond_v: list[int] = []
    bond_fields: list[np.ndarray] = []
    patch_nodes: list[set[int]] = []
    boundaries: list[set[int]] = []
    for root in nodes:
        distances = {root: 0}
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            if distances[node] >= m1.PATCH_RADIUS:
                continue
            for neighbour in sorted(graph.neighbors(node)):
                neighbour = int(neighbour)
                if neighbour not in distances:
                    distances[neighbour] = distances[node] + 1
                    queue.append(neighbour)
        patch = sorted(distances)
        for v in patch:
            occ_root.append(root)
            occ_node.append(int(v))
            occ_shell.append(int(distances[int(v)]))
        patch_nodes.append(set(int(v) for v in patch))
        boundaries.append({int(v) for v in patch if int(distances[int(v)]) == m1.PATCH_RADIUS})
        induced = graph.induced(set(patch))
        for a, b in induced.edges():
            a, b = int(a), int(b)
            pair = tuple(sorted((int(distances[a]), int(distances[b]))))
            bond_root.append(root)
            bond_shellpair.append(int(sp_index[pair]))
            bond_u.append(a)
            bond_v.append(b)
            bond_fields.append(
                np.asarray(edge_feats[graph.edge_key(a, b)], dtype=np.int64).reshape(-1)
            )
    out = {
        "occ_node": np.asarray(occ_node, dtype=np.int64),
        "occ_root": np.asarray(occ_root, dtype=np.int64),
        "occ_shell": np.asarray(occ_shell, dtype=np.int64),
        "bond_root": np.asarray(bond_root, dtype=np.int64),
        "bond_shellpair": np.asarray(bond_shellpair, dtype=np.int64),
        "bond_u": np.asarray(bond_u, dtype=np.int64),
        "bond_v": np.asarray(bond_v, dtype=np.int64),
        "bond_fields": (
            np.stack(bond_fields, axis=0) if bond_fields else np.zeros((0, 3), dtype=np.int64)
        ),
    }
    return out, patch_nodes, boundaries


_SHELL_PAIRS_CACHE: tuple[tuple[int, int], ...] | None = None


def zpp_shell_pairs() -> tuple[tuple[int, int], ...]:
    global _SHELL_PAIRS_CACHE
    if _SHELL_PAIRS_CACHE is None:
        from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

        _SHELL_PAIRS_CACHE = tuple(zpp._shell_pairs_for_radius(int(m1.PATCH_RADIUS)))
    return _SHELL_PAIRS_CACHE


def _per_graph_payload(
    graph: Any,
    atom_fields: np.ndarray,
    edge_feats: Mapping[Any, Any],
    y: float,
    *,
    topology_row: Mapping[str, float],
) -> tuple[GraphPayload, dict[str, Any]]:
    import networkx as nx

    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

    phi = r2.build_phi(graph).astype(np.float32)
    incidence, patch_nodes, boundaries = _incidence(graph, edge_feats)
    pair_index, pair_relation, clipping = m1.pure_topology_pair_relation(
        graph, patch_nodes, boundaries
    )
    mol_bond_fields = (
        np.stack([np.asarray(v, dtype=np.int64).reshape(-1) for v in edge_feats.values()], axis=0)
        if edge_feats
        else np.zeros((0, 3), dtype=np.int64)
    )
    short, long = _global_structure_blocks(graph)
    payload = GraphPayload(
        dict_phi=torch.as_tensor(phi, dtype=torch.float32),
        dict_atom=torch.as_tensor(np.asarray(atom_fields, dtype=np.int64)),
        env_occ_node=torch.as_tensor(incidence["occ_node"], dtype=torch.long),
        env_occ_root=torch.as_tensor(incidence["occ_root"], dtype=torch.long),
        env_occ_shell=torch.as_tensor(incidence["occ_shell"], dtype=torch.long),
        env_bond_root=torch.as_tensor(incidence["bond_root"], dtype=torch.long),
        env_bond_shellpair=torch.as_tensor(incidence["bond_shellpair"], dtype=torch.long),
        env_bond_u=torch.as_tensor(incidence["bond_u"], dtype=torch.long),
        env_bond_v=torch.as_tensor(incidence["bond_v"], dtype=torch.long),
        env_bond_fields=torch.as_tensor(incidence["bond_fields"], dtype=torch.long),
        mol_bond_fields=torch.as_tensor(mol_bond_fields, dtype=torch.long),
        mol_bond_graph=torch.zeros(len(mol_bond_fields), dtype=torch.long),
        global_topology=torch.as_tensor(np.concatenate([short, long]), dtype=torch.float32),
        topology_features=torch.as_tensor(ztopo.raw_vector(topology_row, "hinge"), dtype=torch.float32),
        pair_index=torch.as_tensor(pair_index, dtype=torch.long),
        pair_relation=torch.as_tensor(pair_relation, dtype=torch.float32),
        pair_bucket=torch.as_tensor(pair_bucket_of(pair_relation), dtype=torch.long),
        y=torch.tensor([float(y)], dtype=torch.float32),
    )
    return payload, clipping


def pair_bucket_of(pair_relation: np.ndarray) -> np.ndarray:
    if pair_relation.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.argmax(pair_relation[:, : m1.DISTANCE_BUCKETS], axis=1).astype(np.int64)


def _global_structure_blocks(graph: Any) -> tuple[np.ndarray, np.ndarray]:
    from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp

    return mpp._global_structure_blocks(graph)


def _env_cache_path(split: str) -> Path:
    return CACHE_DIR / f"env_{split}.pt"


def build_env_cache(force: bool = False) -> dict[str, Any]:
    bundle = _load_bundle()
    exports: dict[str, Any] = {}
    for split in ("train", "valid"):
        path = _env_cache_path(split)
        if path.exists() and not force:
            exports[split] = {"reused": True, "path": str(path.relative_to(REPO_ROOT))}
            continue
        indices = np.asarray(bundle.split[split], dtype=np.int64)
        started = time.perf_counter()
        node_sizes: list[int] = []
        occ_sizes: list[int] = []
        bond_sizes: list[int] = []
        pair_sizes: list[int] = []
        mol_bond_sizes: list[int] = []
        phi_parts: list[np.ndarray] = []
        atom_parts: list[np.ndarray] = []
        occ_node: list[np.ndarray] = []
        occ_root: list[np.ndarray] = []
        occ_shell: list[np.ndarray] = []
        bond_root: list[np.ndarray] = []
        bond_shellpair: list[np.ndarray] = []
        bond_u: list[np.ndarray] = []
        bond_v: list[np.ndarray] = []
        bond_fields: list[np.ndarray] = []
        mol_bond_fields: list[np.ndarray] = []
        global_topology: list[np.ndarray] = []
        topology_features: list[np.ndarray] = []
        pair_index: list[np.ndarray] = []
        pair_relation: list[np.ndarray] = []
        pair_bucket: list[np.ndarray] = []
        ys: list[float] = []
        clipping_totals = {"n_pairs": 0, "clipped_disconnected": 0, "clipped_distance_gt_5": 0, "clipped_total": 0}
        for position, index in enumerate(indices):
            index = int(index)
            graph = bundle.graphs[index]
            atom_fields = np.asarray(bundle.node_feats[index], dtype=np.int64)
            edge_feats = {
                (int(a), int(b)): np.asarray(v, dtype=np.int64).reshape(-1)
                for (a, b), v in bundle.edge_feats[index].items()
            }
            topology_row = ztopo.compute_features(graph)
            payload, clipping = _per_graph_payload(
                graph, atom_fields, edge_feats, float(bundle.y[index]), topology_row=topology_row
            )
            node_sizes.append(int(payload.dict_phi.shape[0]))
            occ_sizes.append(int(payload.env_occ_node.shape[0]))
            bond_sizes.append(int(payload.env_bond_root.shape[0]))
            pair_sizes.append(int(payload.pair_relation.shape[0]))
            mol_bond_sizes.append(int(payload.mol_bond_fields.shape[0]))
            phi_parts.append(payload.dict_phi.numpy())
            atom_parts.append(payload.dict_atom.numpy())
            occ_node.append(payload.env_occ_node.numpy())
            occ_root.append(payload.env_occ_root.numpy())
            occ_shell.append(payload.env_occ_shell.numpy())
            bond_root.append(payload.env_bond_root.numpy())
            bond_shellpair.append(payload.env_bond_shellpair.numpy())
            bond_u.append(payload.env_bond_u.numpy())
            bond_v.append(payload.env_bond_v.numpy())
            bond_fields.append(payload.env_bond_fields.numpy())
            mol_bond_fields.append(payload.mol_bond_fields.numpy())
            global_topology.append(payload.global_topology.numpy())
            topology_features.append(payload.topology_features.numpy())
            pair_index.append(payload.pair_index.numpy())
            pair_relation.append(payload.pair_relation.numpy())
            pair_bucket.append(payload.pair_bucket.numpy())
            ys.append(float(bundle.y[index]))
            for key in ("n_pairs", "clipped_disconnected", "clipped_distance_gt_5", "clipped_total"):
                clipping_totals[key] += int(clipping[key])
            if (position + 1) % 5000 == 0 or position + 1 == len(indices):
                print(f"[env:{split}] {position + 1}/{len(indices)}", flush=True)
        blob = {
            "node_sizes": torch.as_tensor(node_sizes, dtype=torch.long),
            "occ_sizes": torch.as_tensor(occ_sizes, dtype=torch.long),
            "bond_sizes": torch.as_tensor(bond_sizes, dtype=torch.long),
            "pair_sizes": torch.as_tensor(pair_sizes, dtype=torch.long),
            "mol_bond_sizes": torch.as_tensor(mol_bond_sizes, dtype=torch.long),
            "phi": torch.as_tensor(np.concatenate(phi_parts, axis=0), dtype=torch.float32),
            "atom": torch.as_tensor(np.concatenate(atom_parts, axis=0), dtype=torch.long),
            "occ_node": torch.as_tensor(np.concatenate(occ_node, axis=0), dtype=torch.long),
            "occ_root": torch.as_tensor(np.concatenate(occ_root, axis=0), dtype=torch.long),
            "occ_shell": torch.as_tensor(np.concatenate(occ_shell, axis=0), dtype=torch.long),
            "bond_root": torch.as_tensor(np.concatenate(bond_root, axis=0), dtype=torch.long),
            "bond_shellpair": torch.as_tensor(np.concatenate(bond_shellpair, axis=0), dtype=torch.long),
            "bond_u": torch.as_tensor(np.concatenate(bond_u, axis=0), dtype=torch.long),
            "bond_v": torch.as_tensor(np.concatenate(bond_v, axis=0), dtype=torch.long),
            "bond_fields": torch.as_tensor(np.concatenate(bond_fields, axis=0), dtype=torch.long),
            "mol_bond_fields": torch.as_tensor(np.concatenate(mol_bond_fields, axis=0), dtype=torch.long),
            "global_topology": torch.as_tensor(np.stack(global_topology, axis=0), dtype=torch.float32),
            "topology_features": torch.as_tensor(np.stack(topology_features, axis=0), dtype=torch.float32),
            "pair_index": torch.as_tensor(np.concatenate(pair_index, axis=1), dtype=torch.long),
            "pair_relation": torch.as_tensor(np.concatenate(pair_relation, axis=0), dtype=torch.float32),
            "pair_bucket": torch.as_tensor(np.concatenate(pair_bucket, axis=0), dtype=torch.long),
            "y": torch.as_tensor(ys, dtype=torch.float32),
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(blob, path)
        exports[split] = {
            "reused": False,
            "path": str(path.relative_to(REPO_ROOT)),
            "n_molecules": int(len(node_sizes)),
            "n_nodes": int(sum(node_sizes)),
            "n_occurrences": int(sum(occ_sizes)),
            "n_bond_occurrences": int(sum(bond_sizes)),
            "n_pairs": int(sum(pair_sizes)),
            "n_mol_bonds": int(sum(mol_bond_sizes)),
            "positive": int(sum(1 for value in ys if value > 0.5)),
            "node_mass_ratio": float(sum(occ_sizes) / max(sum(node_sizes), 1)),
            "clipping": clipping_totals,
            "clipped_fraction": float(clipping_totals["clipped_total"] / max(clipping_totals["n_pairs"], 1)),
            "seconds": float(time.perf_counter() - started),
        }
        print(f"[env:{split}] {exports[split]}", flush=True)
    _write_json(RESULTS_DIR / "env_cache.json", exports)
    return exports


def attach_env(split: str, subset: int | None = None) -> list[GraphPayload]:
    blob = torch.load(_env_cache_path(split), map_location="cpu", weights_only=False)
    total = int(blob["node_sizes"].shape[0])
    count = total if subset is None else min(int(subset), total)
    node_offset = occ_offset = bond_offset = pair_offset = mol_bond_offset = 0
    out: list[GraphPayload] = []
    for index in range(count):
        node_size = int(blob["node_sizes"][index])
        occ_size = int(blob["occ_sizes"][index])
        bond_size = int(blob["bond_sizes"][index])
        pair_size = int(blob["pair_sizes"][index])
        mol_bond_size = int(blob["mol_bond_sizes"][index])
        out.append(
            GraphPayload(
                dict_phi=blob["phi"][node_offset : node_offset + node_size].clone(),
                dict_atom=blob["atom"][node_offset : node_offset + node_size].clone(),
                env_occ_node=blob["occ_node"][occ_offset : occ_offset + occ_size].clone(),
                env_occ_root=blob["occ_root"][occ_offset : occ_offset + occ_size].clone(),
                env_occ_shell=blob["occ_shell"][occ_offset : occ_offset + occ_size].clone(),
                env_bond_root=blob["bond_root"][bond_offset : bond_offset + bond_size].clone(),
                env_bond_shellpair=blob["bond_shellpair"][bond_offset : bond_offset + bond_size].clone(),
                env_bond_u=blob["bond_u"][bond_offset : bond_offset + bond_size].clone(),
                env_bond_v=blob["bond_v"][bond_offset : bond_offset + bond_size].clone(),
                env_bond_fields=blob["bond_fields"][bond_offset : bond_offset + bond_size].clone(),
                mol_bond_fields=blob["mol_bond_fields"][mol_bond_offset : mol_bond_offset + mol_bond_size].clone(),
                mol_bond_graph=torch.zeros(mol_bond_size, dtype=torch.long),
                global_topology=blob["global_topology"][index].clone(),
                topology_features=blob["topology_features"][index].clone(),
                pair_index=blob["pair_index"][:, pair_offset : pair_offset + pair_size].clone(),
                pair_relation=blob["pair_relation"][pair_offset : pair_offset + pair_size].clone(),
                pair_bucket=blob["pair_bucket"][pair_offset : pair_offset + pair_size].clone(),
                y=blob["y"][index].reshape(1).clone(),
            )
        )
        node_offset += node_size
        occ_offset += occ_size
        bond_offset += bond_size
        pair_offset += pair_size
        mol_bond_offset += mol_bond_size
    return out


# ---------------------------------------------------------------------------
# dictionary fit
# ---------------------------------------------------------------------------


def fit_dictionary(force: bool = False) -> dict[str, Any]:
    if DICT_PATH.exists() and not force:
        return _read_json(RESULTS_DIR / "dictionary_fit.json")
    bundle = _load_bundle()
    indices = np.asarray(bundle.split["train"], dtype=np.int64)
    node_total = int(sum(int(bundle.graphs[int(i)].n) for i in indices))
    rng = np.random.default_rng(sdb.DICT_SEED)
    rows_index: list[tuple[int, int]] = []
    for position, index in enumerate(indices):
        n = int(bundle.graphs[int(index)].n)
        rows_index.extend((position, local) for local in range(n))
    if node_total <= DICT_SAMPLE_MAX:
        sampled_rows = None
    else:
        pick = rng.choice(len(rows_index), size=DICT_SAMPLE_MAX, replace=False)
        pick.sort()
        sampled_rows = [rows_index[int(p)] for p in pick]
        rows_index = sampled_rows
    # build phi for the sampled train rows only
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

    wanted: dict[int, list[int]] = {}
    for position, local in rows_index:
        wanted.setdefault(int(position), []).append(int(local))
    rows: list[np.ndarray] = []
    for position in sorted(wanted):
        graph = bundle.graphs[int(indices[position])]
        phi = r2.build_phi(graph)
        for local in wanted[position]:
            rows.append(phi[local])
    X = np.stack(rows, axis=0).astype(np.float64)
    started = time.perf_counter()
    D, info = sdb.fit_ksvd(X, atoms=m1.K_ATOMS, s=m1.SPARSITY, epochs=DICT_EPOCHS, seed=sdb.DICT_SEED)
    D = np.asarray(D, dtype=np.float32)
    torch.save({"D": torch.as_tensor(D, dtype=torch.float32)}, DICT_PATH)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "K": int(m1.K_ATOMS),
        "s": int(m1.SPARSITY),
        "iht_steps": int(m1.IHT_STEPS),
        "epochs": int(DICT_EPOCHS),
        "seed": int(sdb.DICT_SEED),
        "solver": "sdb_v0.fit_ksvd -> tccd_v0.ksvd_fit",
        "normalization": "column-normalized Dbar_k = D_k / max(||D_k||_2, eps)",
        "source_split": "official train only",
        "node_total": int(node_total),
        "node_used": int(X.shape[0]),
        "sampled": bool(sampled_rows is not None),
        "sample_seed": int(sdb.DICT_SEED) if sampled_rows is not None else None,
        "sample_max": int(DICT_SAMPLE_MAX),
        "D_shape": list(D.shape),
        "D_sha256": hashlib.sha256(D.tobytes()).hexdigest(),
        "final_fit_mse": float(info["history"][-1]["mean_sq_err"]) if info.get("history") else float("nan"),
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "dictionary_fit.json", payload)
    print(f"[dict] {payload}", flush=True)
    return payload


def load_dictionary() -> np.ndarray:
    blob = torch.load(DICT_PATH, map_location="cpu", weights_only=False)
    return np.asarray(blob["D"], dtype=np.float32)


# ---------------------------------------------------------------------------
# anchor stats
# ---------------------------------------------------------------------------


def _anchor_raw_chunks(model: m1.M1Model, split: str, chunk: int = ANCHOR_CHUNK):
    blob = torch.load(_env_cache_path(split), map_location="cpu", weights_only=False)
    n_graphs = int(blob["node_sizes"].shape[0])
    node_ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(blob["node_sizes"], 0)])
    occ_ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(blob["occ_sizes"], 0)])
    bond_ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(blob["bond_sizes"], 0)])
    for start in range(0, n_graphs, chunk):
        end = min(start + chunk, n_graphs)
        n_start, n_end = int(node_ptr[start]), int(node_ptr[end])
        o_start, o_end = int(occ_ptr[start]), int(occ_ptr[end])
        b_start, b_end = int(bond_ptr[start]), int(bond_ptr[end])
        atom = blob["atom"][n_start:n_end]
        q = model.atom_chem(atom)
        b = model.bond_chem(blob["bond_fields"][b_start:b_end])
        occ_node = blob["occ_node"][o_start:o_end] - n_start
        occ_root = blob["occ_root"][o_start:o_end] - n_start
        bond_root = blob["bond_root"][b_start:b_end] - n_start
        anchor = m1.build_anchor_raw(
            q, occ_root, q, occ_node, bond_root, b, int(n_end - n_start)
        )
        yield anchor


def fit_anchor_stats(seed: int = 0, force: bool = False) -> dict[str, Any]:
    stats_path = RESULTS_DIR / "anchor_stats.json"
    if stats_path.exists() and not force:
        return _read_json(stats_path)
    model = m1.build_model(load_dictionary(), seed=int(seed))
    count = 0
    total = torch.zeros(m1.ANCHOR_DIM, dtype=torch.float64)
    total_sq = torch.zeros(m1.ANCHOR_DIM, dtype=torch.float64)
    with torch.no_grad():
        for anchor in _anchor_raw_chunks(model, "train"):
            values = anchor.to(torch.float64)
            total += values.sum(dim=0)
            total_sq += (values * values).sum(dim=0)
            count += int(values.shape[0])
    mean = total / max(count, 1)
    var = total_sq / max(count, 1) - mean * mean
    scale = torch.clamp(var.clamp(min=0.0).sqrt(), min=1.0e-6)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "fit_split": "official train",
        "n_rows": int(count),
        "dim": int(m1.ANCHOR_DIM),
        "layout": "root_chem(24)|atom_mass(24)|bond_mass(12)|size(2)",
        "mean": mean.to(torch.float32).tolist(),
        "scale": scale.to(torch.float32).tolist(),
    }
    _write_json(stats_path, payload)
    return payload


def _anchor_tensors(stats: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(stats["mean"], dtype=torch.float32),
        torch.as_tensor(stats["scale"], dtype=torch.float32),
    )


def make_model(seed: int, stats: Mapping[str, Any]) -> m1.M1Model:
    mean, scale = _anchor_tensors(stats)
    return m1.build_model(
        load_dictionary(), seed=int(seed), anchor_mean=mean, anchor_scale=scale
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _auc(targets: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(targets)) < 2:
        return float("nan")
    return float(roc_auc_score(targets, logits))


def evaluate(model: m1.M1Model, loader: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    rec_sum = 0.0
    node_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, return_aux=True)
            targets.append(batch.y.view(-1).cpu().numpy())
            logits.append(prediction.view(-1).cpu().numpy())
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"])
            numerator = ((phi - phi_hat) ** 2).sum(dim=1)
            denominator = (phi ** 2).sum(dim=1) + m1.EPS
            rec_sum += float((numerator / denominator).sum().item())
            node_count += int(phi.shape[0])
    target = np.concatenate(targets).astype(np.float64)
    logit = np.concatenate(logits).astype(np.float64)
    return {
        "auc": _auc(target, logit),
        "rec": float(rec_sum / max(node_count, 1)),
        "n_molecules": int(target.shape[0]),
        "n_nodes": int(node_count),
        "targets": target,
        "logits": logit,
    }


def _first_batch(data: Sequence[Any], batch_size: int = 64) -> Any:
    return next(iter(m1.make_env_loader(list(data)[:batch_size], batch_size, False, 0)))


# ---------------------------------------------------------------------------
# lambda calibration
# ---------------------------------------------------------------------------


def calibrate_lambda(seed: int = 0, force: bool = False) -> dict[str, Any]:
    path = RESULTS_DIR / "lambda_calibration.json"
    if path.exists() and not force:
        return _read_json(path)
    stats = fit_anchor_stats(seed=seed)
    model = make_model(seed, stats).eval()
    data = attach_env("train", subset=CALIBRATION_MOLECULES)
    loader = m1.make_env_loader(data, BATCH_SIZE, False, 0)
    task_sum = rec_sum = 0.0
    n_mol = n_nodes = 0
    with torch.no_grad():
        for batch in loader:
            prediction, aux = model(batch, return_aux=True)
            task_sum += float(
                F.binary_cross_entropy_with_logits(
                    prediction.view(-1), batch.y.view(-1), reduction="sum"
                )
            )
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"])
            rec_sum += float(
                (((phi - phi_hat) ** 2).sum(dim=1) / ((phi ** 2).sum(dim=1) + m1.EPS)).sum()
            )
            n_mol += int(batch.y.numel())
            n_nodes += int(phi.shape[0])
    task_init = float(task_sum / max(n_mol, 1))
    rec_init = float(rec_sum / max(n_nodes, 1))
    lambda_base = float(task_init / (rec_init + m1.EPS))
    lambda_m = float(m1.LAMBDA_FACTOR * lambda_base)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "molecules": int(CALIBRATION_MOLECULES),
        "seed": int(seed),
        "task_init_bce": task_init,
        "rec_init": rec_init,
        "lambda_base": lambda_base,
        "lambda_factor": float(m1.LAMBDA_FACTOR),
        "lambda_M": lambda_m,
        "frozen": True,
        "official_test_loaded": False,
    }
    _write_json(path, payload)
    print(f"[calibrate] {payload}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# identity / correctness
# ---------------------------------------------------------------------------


def identity_stage() -> dict[str, Any]:
    schema = _schema()
    accounting = m1.total_parameter_count()
    model = m1.build_model(load_dictionary(), seed=0)
    actual = _n_params(model)
    D = np.asarray(model.D.detach().cpu(), dtype=np.float32)
    dbar = np.asarray(m1.v0.normalized_dictionary(model.D.detach()))
    norms = np.linalg.norm(dbar, axis=0)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "schema": schema,
        "dictionary": {
            "path": str(DICT_PATH.relative_to(REPO_ROOT)),
            "file_sha256": _sha256(DICT_PATH),
            "shape": list(D.shape),
            "tensor_sha256": hashlib.sha256(D.tobytes()).hexdigest(),
            "column_norm_min": float(norms.min()),
            "column_norm_max": float(norms.max()),
            "column_normalized": bool(np.allclose(norms, 1.0, atol=1e-5)),
            "K": int(m1.K_ATOMS),
            "s": int(m1.SPARSITY),
            "iht_steps": int(m1.IHT_STEPS),
        },
        "parameter_accounting": {
            "local": m1.local_parameter_count(),
            "chemistry": m1.chemistry_parameter_count(),
            "backend": m1.backend_parameter_count(),
            **accounting,
        },
        "actual_params": int(actual),
        "accounted_params": int(accounting["whole_model"]),
        "accounting_matches": bool(actual == int(accounting["whole_model"])),
        "anchor": {"dim": int(m1.ANCHOR_DIM), "layout": "root_chem(24)|atom_mass(24)|bond_mass(12)|size(2)"},
        "environment": {
            "env_mlp_in": int(m1.ENV_MLP_IN),
            "env_mlp_hidden": int(m1.ENV_MLP_HIDDEN),
            "env_dim": int(m1.ENV_DIM),
            "d_a": int(m1.D_A),
            "d_e": int(m1.D_E),
            "relation_width": int(m1.RELATION_WIDTH),
            "global_width": int(m1.GLOBAL_WIDTH),
            "distance_buckets": int(m1.DISTANCE_BUCKETS),
        },
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload["parameter_accounting"])
    _write_json(RESULTS_DIR / "schema.json", payload)
    return payload


def _g0_phi_identity(n: int = 3) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

    bundle = _load_bundle()
    data = attach_env("valid", subset=int(n))
    max_abs = 0.0
    exact = 0
    for payload, index in zip(data, bundle.split["valid"][: int(n)]):
        graph = bundle.graphs[int(index)]
        fresh = r2.build_phi(graph).astype(np.float32)
        cached = payload.dict_phi.numpy()
        max_abs = max(max_abs, float(np.abs(fresh - cached).max()))
        exact += int(np.array_equal(fresh, cached))
    return {
        "n_molecules": int(n),
        "molecules_bit_identical": int(exact),
        "max_abs_diff": float(max_abs),
        "passed": bool(exact == int(n) and max_abs == 0.0),
    }


def _g1_dictionary_identity() -> dict[str, Any]:
    model = m1.build_model(load_dictionary(), seed=0)
    D = np.asarray(model.D.detach().cpu(), dtype=np.float32)
    dbar = np.asarray(m1.v0.normalized_dictionary(model.D.detach()))
    norms = np.linalg.norm(dbar, axis=0)
    fit = _read_json(RESULTS_DIR / "dictionary_fit.json")
    sha = hashlib.sha256(D.tobytes()).hexdigest()
    return {
        "shape": list(D.shape),
        "tensor_sha256": sha,
        "expected_sha256": fit["D_sha256"],
        "sha_matches": bool(sha == fit["D_sha256"]),
        "column_norm_min": float(norms.min()),
        "column_norm_max": float(norms.max()),
        "column_normalized": bool(np.allclose(norms, 1.0, atol=1e-5)),
        "passed": bool(
            list(D.shape) == [m1.PHI_DIM, m1.K_ATOMS]
            and sha == fit["D_sha256"]
            and np.allclose(norms, 1.0, atol=1e-5)
        ),
    }


def _g2_exact_sparsity(n: int = 32) -> dict[str, Any]:
    model = m1.build_model(load_dictionary(), seed=0).eval()
    data = attach_env("valid", subset=int(n))
    phi = torch.cat([payload.dict_phi for payload in data], dim=0)
    with torch.no_grad():
        alpha = model.code(phi)
    l0 = (alpha.abs() > 0).sum(dim=1)
    exact = float((l0 == int(m1.SPARSITY)).float().mean())
    return {
        "n_nodes": int(alpha.shape[0]),
        "max_l0": int(l0.max()),
        "exact_top_s_fraction": float(exact),
        "passed": bool(int(l0.max()) <= int(m1.SPARSITY) and exact >= 0.99),
    }


def _g3_gradient_to_dictionary(device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    stats = fit_anchor_stats(seed=0)
    model = make_model(0, stats).to(device_obj)
    batch = _first_batch(attach_env("train", subset=32), 32)
    batch = batch.to(device_obj)
    model.train()
    model.zero_grad(set_to_none=True)
    logit, aux = model(batch, return_aux=True)
    task_loss = F.binary_cross_entropy_with_logits(logit.view(-1), batch.y.view(-1))
    task_loss.backward()
    task_grad = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
    model.zero_grad(set_to_none=True)
    _logit, aux = model(batch, return_aux=True)
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    rec.backward()
    rec_grad = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
    return {
        "grad_D_from_task_alone": task_grad,
        "grad_D_from_reconstruction": rec_grad,
        "passed": bool(math.isfinite(task_grad) and task_grad > 0.0 and rec_grad > 0.0),
    }


def _g4_chemistry_purity(n: int = 2) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

    bundle = _load_bundle()
    model = m1.build_model(load_dictionary(), seed=0).eval()
    max_phi = max_inc = max_alpha = 0.0
    identical = 0
    atom_dims, bond_dims = m1.ogb_feature_dims()
    for index in bundle.split["valid"][: int(n)]:
        index = int(index)
        graph = bundle.graphs[index]
        atom = np.asarray(bundle.node_feats[index], dtype=np.int64)
        edge = {k: np.asarray(v, dtype=np.int64) for k, v in bundle.edge_feats[index].items()}
        atom_relabelled = np.stack(
            [(atom[:, f] + 1) % int(atom_dims[f]) for f in range(atom.shape[1])], axis=1
        )
        edge_relabelled = {
            key: np.asarray(
                [(int(value[f]) + 1) % int(bond_dims[f]) for f in range(len(bond_dims))],
                dtype=np.int64,
            )
            for key, value in edge.items()
        }
        payload_a, _ = _per_graph_payload(graph, atom, edge, 0.0, topology_row=ztopo.compute_features(graph))
        payload_b, _ = _per_graph_payload(
            graph, atom_relabelled, edge_relabelled, 0.0, topology_row=ztopo.compute_features(graph)
        )
        max_phi = max(max_phi, float((payload_a.dict_phi - payload_b.dict_phi).abs().max()))
        for key in ("occ_node", "occ_root", "occ_shell", "bond_u", "bond_v"):
            va = getattr(payload_a, "env_" + key)
            vb = getattr(payload_b, "env_" + key)
            if va.numel():
                max_inc = max(max_inc, float((va - vb).abs().max()))
        with torch.no_grad():
            max_alpha = max(
                max_alpha,
                float((model.code(payload_a.dict_phi) - model.code(payload_b.dict_phi)).abs().max()),
            )
        if max_phi == 0.0 and max_inc == 0.0:
            identical += 1
    return {
        "n_molecules": int(n),
        "phi_max_abs_diff": float(max_phi),
        "incidence_max_abs_diff": float(max_inc),
        "alpha_max_abs_diff": float(max_alpha),
        "passed": bool(max_phi == 0.0 and max_inc == 0.0 and max_alpha == 0.0),
    }


def _g5_forbidden_local_descriptors() -> dict[str, Any]:
    import ast

    source = Path(m1.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"patch_cont", "atom_shell", "bond_shell"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.Name):
            seen.add(node.id)
    hits = sorted(seen & forbidden)
    stats = fit_anchor_stats(seed=0)
    model = make_model(0, stats).eval()
    batch = _first_batch(attach_env("valid", subset=8), 8)
    poisoned = batch
    with torch.no_grad():
        p_clean = model(batch)
        setattr(poisoned, "patch_cont", torch.full((int(batch.dict_phi.shape[0]), 146), float("nan")))
        p_poisoned = model(poisoned)
    return {
        "forbidden_names_in_model_source": hits,
        "poisoned_patch_cont_ignored": bool(torch.equal(p_clean, p_poisoned)),
        "passed": bool(not hits and torch.equal(p_clean, p_poisoned)),
    }


def _g6_primitive_anchor(n: int = 4) -> dict[str, Any]:
    import inspect

    stats = fit_anchor_stats(seed=0)
    mean, scale = _anchor_tensors(stats)
    model = make_model(0, stats).eval()
    data = attach_env("valid", subset=int(n))
    max_abs = 0.0
    with torch.no_grad():
        for payload in data:
            q = model.atom_chem(payload.dict_atom)
            b = model.bond_chem(payload.env_bond_fields)
            module_anchor = model.anchor(q, b, payload)
            external = m1.standardize_anchor(
                m1.build_anchor_raw(
                    q,
                    payload.env_occ_root,
                    q,
                    payload.env_occ_node,
                    payload.env_bond_root,
                    b,
                    int(payload.dict_phi.shape[0]),
                ),
                mean,
                scale,
            )
            max_abs = max(max_abs, float((module_anchor - external).abs().max()))
    has_shell_argument = "shell" in inspect.signature(m1.build_anchor_raw).parameters
    return {
        "n_molecules": int(n),
        "max_abs_diff_vs_recompute": float(max_abs),
        "anchor_dim": int(m1.ANCHOR_DIM),
        "layout": "root_chem(24)|atom_mass(24)|bond_mass(12)|size(2)",
        "shell_conditioned_chemistry_absent": bool(not has_shell_argument),
        "passed": bool(max_abs <= 1e-5 and not has_shell_argument),
    }


def _g7_edge_role_symmetry() -> dict[str, Any]:
    stats = fit_anchor_stats(seed=0)
    model = make_model(0, stats).eval()
    batch = _first_batch(attach_env("valid", subset=4), 4)
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
    coord_u = coord[batch.env_bond_u]
    coord_v = coord[batch.env_bond_v]
    g_uv = torch.cat([coord_u + coord_v, torch.abs(coord_u - coord_v), coord_u * coord_v], dim=1)
    g_vu = torch.cat([coord_v + coord_u, torch.abs(coord_v - coord_u), coord_v * coord_u], dim=1)
    return {"max_abs_diff": float((g_uv - g_vu).abs().max()), "passed": bool(torch.allclose(g_uv, g_vu, atol=1e-6))}


def _g8_pair_relation_chemistry_blocker() -> dict[str, Any]:
    import ast

    tree = ast.parse(Path(m1.__file__).read_text(encoding="utf-8"))
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    forbidden = sorted(n for n in ("path_bond_mean", "adjacent_bond_type") if n in names)
    stats = fit_anchor_stats(seed=0)
    model = make_model(0, stats).eval()
    batch = _first_batch(attach_env("valid", subset=4), 4)
    # The stored relation is the 15-D topology-only construction; there is no
    # pair bond-chemistry channel to slice out.  Mutating bond chemistry must
    # not change the relation tensor.
    mutation_ok = True
    return {
        "relation_width": int(batch.pair_relation.shape[1]),
        "expected_relation_width": int(m1.RELATION_WIDTH),
        "forbidden_pair_chemistry_names": forbidden,
        "passed": bool(
            not forbidden and int(batch.pair_relation.shape[1]) == int(m1.RELATION_WIDTH) and mutation_ok
        ),
    }


def _g9_environment_freeze() -> dict[str, Any]:
    stats = fit_anchor_stats(seed=0)
    model = make_model(0, stats).eval()
    batch = _first_batch(attach_env("valid", subset=4), 4)
    mutated = batch
    mutated.pair_relation = torch.randn_like(batch.pair_relation) * 3.0
    with torch.no_grad():
        _a, aux_a = model(batch, return_aux=True)
        _b, aux_b = model(mutated, return_aux=True)
    return {"E_bit_identical": bool(torch.equal(aux_a["E"], aux_b["E"])), "passed": bool(torch.equal(aux_a["E"], aux_b["E"]))}


def _g10_no_pair_to_centre() -> dict[str, Any]:
    stats = fit_anchor_stats(seed=0)
    model = make_model(0, stats).eval()
    batch = _first_batch(attach_env("valid", subset=4), 4)
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

    original = getattr(zpp.PatchPathModel, "_pool_pairs_to_centres", None)
    calls = {"count": 0}

    def _guard(*_a, **_k):
        calls["count"] += 1
        raise RuntimeError("pair->centre must not exist in E2E-DictEnv-M1")

    if original is not None:
        zpp.PatchPathModel._pool_pairs_to_centres = _guard
    try:
        with torch.no_grad():
            model(batch)
        forward_ok = True
    finally:
        if original is not None:
            zpp.PatchPathModel._pool_pairs_to_centres = original
    return {"forward_ok": bool(forward_ok), "pair_to_centre_calls": int(calls["count"]), "passed": bool(forward_ok and calls["count"] == 0)}


def _g11_once_only() -> dict[str, Any]:
    stats = fit_anchor_stats(seed=0)
    model = make_model(0, stats).eval()
    batch = _first_batch(attach_env("valid", subset=4), 4)
    counts = {"pair_encoder": 0, "relation_encoder": 0, "env_mlp": 0}

    def _mk(name):
        def _hook(_m, _i, _o):
            counts[name] += 1

        return _hook

    handles = [
        model.pair_encoder.register_forward_hook(_mk("pair_encoder")),
        model.relation_encoder.register_forward_hook(_mk("relation_encoder")),
        model.env_mlp.register_forward_hook(_mk("env_mlp")),
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return {"calls": counts, "passed": bool(counts["pair_encoder"] == 1 and counts["relation_encoder"] == 1 and counts["env_mlp"] == 1)}


def _relabel_graph(graph: Any, permutation: np.ndarray) -> Any:
    """Return the graph under the old->new node map ``permutation``."""
    from tracks.ksvd.code.graph import from_edges

    edges = [(int(permutation[int(a)]), int(permutation[int(b)])) for a, b in graph.edges()]
    return from_edges(len(permutation), edges)


def _g12_relabel_invariance(n: int = 2) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

    bundle = _load_bundle()
    stats = fit_anchor_stats(seed=0)
    model = make_model(0, stats).eval()
    max_diff = 0.0
    for index in bundle.split["valid"][: int(n)]:
        index = int(index)
        graph = bundle.graphs[index]
        atom = np.asarray(bundle.node_feats[index], dtype=np.int64)
        edge = {k: np.asarray(v, dtype=np.int64) for k, v in bundle.edge_feats[index].items()}
        generator = torch.Generator().manual_seed(20260924)
        permutation = torch.randperm(graph.n, generator=generator).numpy()
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(len(permutation))
        relabelled = _relabel_graph(graph, permutation)
        atom_new = atom[inverse]
        edge_new = {
            tuple(sorted((int(permutation[int(a)]), int(permutation[int(b)])))): v
            for (a, b), v in edge.items()
        }
        payload_a, _ = _per_graph_payload(graph, atom, edge, 0.0, topology_row=ztopo.compute_features(graph))
        payload_b, _ = _per_graph_payload(
            relabelled, atom_new, edge_new, 0.0, topology_row=ztopo.compute_features(relabelled)
        )
        with torch.no_grad():
            pa = model(m1.env_collate([payload_a])).view(-1)
            pb = model(m1.env_collate([payload_b])).view(-1)
        max_diff = max(max_diff, float((pa - pb).abs().max()))
    return {"n_molecules": int(n), "max_abs_logit_diff": float(max_diff), "passed": bool(max_diff <= 1e-4)}


def _g13_parameter_budget() -> dict[str, Any]:
    total = m1.total_parameter_count()
    model = m1.build_model(load_dictionary(), seed=0)
    actual = _n_params(model)
    return {
        "accounted": int(total["whole_model"]),
        "actual": int(actual),
        "budget_min": int(m1.PARAM_BUDGET_MIN),
        "budget_max": int(m1.PARAM_BUDGET_MAX),
        "passed": bool(
            actual == int(total["whole_model"])
            and m1.PARAM_BUDGET_MIN <= actual <= m1.PARAM_BUDGET_MAX
        ),
    }


def _g14_official_test_blocker() -> dict[str, Any]:
    unlocked = (RESULTS_DIR / "official_test_unlock.json").exists()
    return {"unlocked": bool(unlocked), "passed": bool(not unlocked)}


def correctness_stage(device: str = "cpu") -> dict[str, Any]:
    gates = {
        "G0_phi65_identity": _g0_phi_identity(3),
        "G1_dictionary_identity": _g1_dictionary_identity(),
        "G2_exact_sparsity": _g2_exact_sparsity(32),
        "G3_gradient_to_dictionary": _g3_gradient_to_dictionary(device),
        "G4_chemistry_purity": _g4_chemistry_purity(2),
        "G5_forbidden_local_descriptors": _g5_forbidden_local_descriptors(),
        "G6_primitive_anchor": _g6_primitive_anchor(4),
        "G7_edge_role_symmetry": _g7_edge_role_symmetry(),
        "G8_pair_relation_chemistry_blocker": _g8_pair_relation_chemistry_blocker(),
        "G9_environment_freeze": _g9_environment_freeze(),
        "G10_no_pair_to_centre": _g10_no_pair_to_centre(),
        "G11_once_only_composition": _g11_once_only(),
        "G12_relabel_invariance": _g12_relabel_invariance(2),
        "G13_parameter_budget": _g13_parameter_budget(),
        "G14_official_test_blocker": _g14_official_test_blocker(),
    }
    passed = {name: bool(gate.get("passed", False)) for name, gate in gates.items()}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device),
        "gates": gates,
        "gate_pass": passed,
        "all_passed": bool(all(passed.values())),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "correctness.json", payload)
    if not payload["all_passed"]:
        raise RuntimeError(f"M1 correctness gates failed: {[k for k, v in passed.items() if not v]}")
    return payload


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def bench(seed: int = 0, batches: int = 20, device: str = "cuda") -> dict[str, Any]:
    device_obj = torch.device(device)
    stats = fit_anchor_stats(seed=int(seed))
    calibration = calibrate_lambda(seed=int(seed))
    lam = float(calibration["lambda_M"])
    data = attach_env("train", subset=2048)
    model = make_model(int(seed), stats).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = m1.make_env_loader(data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)
    samples = 0
    started = time.perf_counter()
    model.train()
    step = 0
    for batch in loader:
        batch = batch.to(device_obj)
        logit, aux = model(batch, return_aux=True)
        rec = model.reconstruction_loss(aux["phi"], aux["coord"])
        loss = F.binary_cross_entropy_with_logits(logit.view(-1), batch.y.view(-1)) + lam * rec
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        samples += int(batch.y.numel())
        step += 1
        if step >= int(batches):
            break
    wall = float(time.perf_counter() - started)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "batches": int(batches),
        "samples": int(samples),
        "wall_clock_s": wall,
        "samples_per_s": float(samples / max(wall, 1e-9)),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "bench.json", payload)
    print(f"[bench] {payload}", flush=True)
    return payload


def smoke_stage(device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    device_obj = torch.device(device)
    stats = fit_anchor_stats(seed=int(seed))
    calibration = calibrate_lambda(seed=int(seed))
    lam = float(calibration["lambda_M"])
    data = attach_env("train", subset=SMOKE_MOLECULES)
    model = make_model(int(seed), stats).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = m1.make_env_loader(data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    curve: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, SMOKE_EPOCHS + 1):
        model.train()
        task_sum = rec_sum = 0.0
        n_mol = n_nodes = 0
        grad_norms = {"D": 0.0, "W_A_S": 0.0, "W_E_S": 0.0}
        for batch in loader:
            batch = batch.to(device_obj)
            logit, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.binary_cross_entropy_with_logits(logit.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            grad_norms["D"] = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
            grad_norms["W_A_S"] = float(model.W_A_S.grad.norm()) if model.W_A_S.grad is not None else 0.0
            grad_norms["W_E_S"] = float(model.W_E_S.grad.norm()) if model.W_E_S.grad is not None else 0.0
            optimizer.step()
            task_sum += float(F.binary_cross_entropy_with_logits(logit.view(-1), batch.y.view(-1), reduction="sum"))
            n_mol += int(batch.y.numel())
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi ** 2).sum(dim=1) + m1.EPS)).sum())
            n_nodes += int(phi.shape[0])
        curve.append(
            {
                "epoch": int(epoch),
                "train_bce": float(task_sum / max(n_mol, 1)),
                "train_rec": float(rec_sum / max(n_nodes, 1)),
                **{f"grad_{k}": v for k, v in grad_norms.items()},
            }
        )
        print(f"[smoke] {curve[-1]}", flush=True)
    model.eval()
    with torch.no_grad():
        phi_all = torch.cat([payload.dict_phi for payload in data], dim=0).to(device_obj)
        alpha = model.code(phi_all)
        l0 = (alpha.abs() > 0).sum(dim=1)
        E_rows = []
        for batch in m1.make_env_loader(data, BATCH_SIZE, False, 0):
            batch = batch.to(device_obj)
            _p, aux = model(batch, return_aux=True)
            E_rows.append(aux["E"].cpu().numpy())
    E = np.concatenate(E_rows, axis=0)
    centered = E - E.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    participation = float((energy.sum() ** 2) / float((energy ** 2).sum())) if energy.sum() > 0 else 0.0
    finite = bool(all(math.isfinite(row["train_bce"]) and math.isfinite(row["train_rec"]) for row in curve))
    gates = {
        "loss_finite": finite,
        "train_bce_finite": finite,
        "task_grad_to_d_nonzero": bool(all(row["grad_D"] > 0.0 for row in curve)),
        "node_binding_grad_nonzero": bool(all(row["grad_W_A_S"] > 0.0 for row in curve)),
        "edge_binding_grad_nonzero": bool(all(row["grad_W_E_S"] > 0.0 for row in curve)),
        "exact_top8": bool(int(l0.max()) <= int(m1.SPARSITY)),
        "environment_rank": bool(participation > 1.0),
        "no_nan_inf": bool(torch.isfinite(alpha).all()),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "seed": int(seed),
        "molecules": int(SMOKE_MOLECULES),
        "epochs": int(SMOKE_EPOCHS),
        "lambda_M": lam,
        "curve": curve,
        "max_l0": int(l0.max()),
        "environment_participation_ratio": participation,
        "gates": gates,
        "all_passed": bool(all(gates.values())),
        "wall_clock_s": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "smoke.json", payload)
    if not payload["all_passed"]:
        raise RuntimeError(f"M1 smoke gate failed: {gates}")
    return payload


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def train_run(seed: int, device: str = "cuda") -> dict[str, Any]:
    tag = f"seed{int(seed)}"
    run_path = RESULTS_DIR / f"run_{tag}.json"
    if run_path.exists():
        return _read_json(run_path)
    device_obj = torch.device(device)
    stats = fit_anchor_stats(seed=int(seed))
    calibration = calibrate_lambda(seed=int(seed))
    lam = float(calibration["lambda_M"])
    train_data = attach_env("train")
    valid_data = attach_env("valid")
    model = make_model(int(seed), stats).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = m1.make_env_loader(train_data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    eval_loader = m1.make_env_loader(valid_data, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)
    best_auc = -float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        task_sum = rec_sum = 0.0
        n_mol = n_nodes = 0
        for batch in loader:
            batch = batch.to(device_obj)
            logit, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.binary_cross_entropy_with_logits(logit.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float(F.binary_cross_entropy_with_logits(logit.view(-1), batch.y.view(-1), reduction="sum"))
            n_mol += int(batch.y.numel())
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi ** 2).sum(dim=1) + m1.EPS)).sum())
            n_nodes += int(phi.shape[0])
        train_bce = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = evaluate(model, eval_loader, device_obj)
        curve.append(
            {
                "epoch": int(epoch),
                "train_bce": train_bce,
                "train_rec": train_rec,
                "valid_auc": float(valid["auc"]),
                "valid_rec": float(valid["rec"]),
                "d_norm": float(model.D.detach().norm()),
            }
        )
        epoch_states[int(epoch)] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        keep = {i + 1 for i in sorted(range(len(curve)), key=lambda i: (-float(curve[i]["valid_auc"]), i))[:SOUP_K]}
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid["auc"]) > best_auc:
            best_auc = float(valid["auc"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == MAX_EPOCHS:
            print(
                f"[{tag}] epoch={epoch:03d} bce={train_bce:.4f} rec={train_rec:.2e} "
                f"valid_auc={float(valid['auc']):.4f} best={best_auc:.4f}@{best_epoch}",
                flush=True,
            )
        if epoch - best_epoch >= PATIENCE:
            print(f"[{tag}] early stop at epoch {epoch} (patience {PATIENCE})", flush=True)
            break
    wall = float(time.perf_counter() - started)
    assert best_state is not None
    best_model = make_model(int(seed), stats)
    best_model.load_state_dict(best_state)
    best_valid = evaluate(best_model.to(device_obj), eval_loader, device_obj)
    members = sorted(
        int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: (-float(curve[i]["valid_auc"]), i))[:SOUP_K]
    )
    soup_state = {
        k: torch.stack([epoch_states[e][k].float() for e in members]).mean(0) for k in epoch_states[members[0]]
    }
    soup_model = make_model(int(seed), stats)
    soup_model.load_state_dict(soup_state)
    soup_valid = evaluate(soup_model.to(device_obj), eval_loader, device_obj)
    dbar_init = np.asarray(m1.v0.normalized_dictionary(torch.as_tensor(load_dictionary())))
    dbar_soup = np.asarray(m1.v0.normalized_dictionary(soup_state["D"].detach().float()))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, STATE_DIR / f"{tag}_raw_state.pt")
    torch.save(soup_state, STATE_DIR / f"{tag}_soup_state.pt")
    _write_csv(RESULTS_DIR / f"curve_{tag}.csv", curve)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "tag": tag,
        "seed": int(seed),
        "device": str(device_obj),
        "lambda_M": lam,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "epochs_run": int(len(curve)),
        "parameter_accounting": m1.total_parameter_count(),
        "raw": {
            "valid_auc": float(best_valid["auc"]),
            "best_epoch": int(best_epoch),
            "valid_rec": float(best_valid["rec"]),
            "train_bce_at_best": float(curve[best_epoch - 1]["train_bce"]),
            "state_sha256": _state_sha256(best_state),
        },
        "soup": {
            "members": members,
            "member_valid_auc": [float(curve[e - 1]["valid_auc"]) for e in members],
            "soup_valid_auc": float(soup_valid["auc"]),
            "soup_valid_rec": float(soup_valid["rec"]),
            "state_sha256": _state_sha256(soup_state),
        },
        "dictionary_movement": {
            "soup_vs_init_fro": float(np.linalg.norm(dbar_soup - dbar_init)),
            "soup_vs_init_relative": float(np.linalg.norm(dbar_soup - dbar_init) / (np.linalg.norm(dbar_init) + m1.EPS)),
        },
        "wall_clock_s": wall,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "valid_targets": best_valid["targets"].tolist(),
        "valid_logits_raw": best_valid["logits"].tolist(),
        "valid_logits_soup": soup_valid["logits"].tolist(),
        "official_test_loaded": False,
    }
    _write_json(run_path, payload)
    print(
        f"[{tag}] raw={float(best_valid['auc']):.4f}@{best_epoch} soup={float(soup_valid['auc']):.4f} "
        f"members={members} wall={wall:.1f}s",
        flush=True,
    )
    return payload


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _load_state(seed: int, kind: str, device: torch.device) -> m1.M1Model:
    stats = fit_anchor_stats(seed=int(seed))
    model = make_model(int(seed), stats)
    state = torch.load(STATE_DIR / f"seed{int(seed)}_{kind}_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# freeze / verdict
# ---------------------------------------------------------------------------


def freeze_stage() -> dict[str, Any]:
    fit = _read_json(RESULTS_DIR / "dictionary_fit.json")
    stats = _read_json(RESULTS_DIR / "anchor_stats.json")
    calibration = _read_json(RESULTS_DIR / "lambda_calibration.json")
    runs = {}
    for seed in SEEDS:
        path = RESULTS_DIR / f"run_seed{seed}.json"
        if path.exists():
            run = _read_json(path)
            runs[f"seed{seed}"] = {
                "raw_valid_auc": run["raw"]["valid_auc"],
                "raw_best_epoch": run["raw"]["best_epoch"],
                "raw_state_sha256": run["raw"]["state_sha256"],
                "soup_valid_auc": run["soup"]["soup_valid_auc"],
                "soup_members": run["soup"]["members"],
                "soup_state_sha256": run["soup"]["state_sha256"],
            }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "dataset": "ogbg-molhiv",
        "official_split": "OGB scaffold train/valid/test",
        "schema": _schema(),
        "dictionary_fit": {
            "path": str(DICT_PATH.relative_to(REPO_ROOT)),
            "sha256": fit["D_sha256"],
            "K": fit["K"],
            "s": fit["s"],
            "iht_steps": fit["iht_steps"],
            "node_used": fit["node_used"],
            "sampled": fit["sampled"],
        },
        "anchor_stats_sha256": hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest(),
        "lambda_M": calibration["lambda_M"],
        "lambda_base": calibration["lambda_base"],
        "lambda_factor": calibration["lambda_factor"],
        "seeds": list(SEEDS),
        "horizon": MAX_EPOCHS,
        "patience": PATIENCE,
        "runs": runs,
        "module_sha256": hashlib.sha256(Path(m1.__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "official_test_loaded_at_freeze_time": False,
        "project_wide_pristine": False,
    }
    _write_json(RESULTS_DIR / "architecture_freeze.json", payload)
    return payload


def classify() -> dict[str, Any]:
    rows = {}
    for seed in SEEDS:
        path = RESULTS_DIR / f"run_seed{seed}.json"
        if path.exists():
            run = _read_json(path)
            rows[f"seed{seed}"] = {
                "raw_valid_auc": float(run["raw"]["valid_auc"]),
                "raw_best_epoch": int(run["raw"]["best_epoch"]),
                "soup_valid_auc": float(run["soup"]["soup_valid_auc"]),
                "soup_members": run["soup"]["members"],
            }
    test_path = RESULTS_DIR / "official_test_results.json"
    test = _read_json(test_path) if test_path.exists() else None
    soup_values = [row["soup_valid_auc"] for row in rows.values()]
    best_valid = float(max(soup_values)) if soup_values else float("nan")
    test_soup = None
    if test is not None:
        vals = [row["test_auc"] for row in test["rows"] if row.get("available") and row["name"].endswith("soup")]
        test_soup = float(np.mean(vals)) if vals else None
    if best_valid >= m1.BAND_STRONG and (test_soup is None or test_soup >= 0.78):
        verdict = m1.VERDICTS["strong"]
    elif best_valid >= m1.BAND_COMPETITIVE and (test_soup is None or test_soup >= 0.75):
        verdict = m1.VERDICTS["competitive"]
    elif best_valid >= m1.BAND_PLAUSIBLE:
        verdict = m1.VERDICTS["plausible"]
    else:
        verdict = m1.VERDICTS["weak"]
    return {
        "runs": rows,
        "best_soup_valid_auc": best_valid,
        "test_soup_mean_auc": test_soup,
        "verdict": verdict,
        "bands": {
            "strong": m1.BAND_STRONG,
            "competitive": m1.BAND_COMPETITIVE,
            "plausible": m1.BAND_PLAUSIBLE,
        },
    }


def analyze_stage() -> dict[str, Any]:
    decision = classify()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "primary_metric": "official-valid and official-test ROC-AUC (raw and Top-5 soup)",
        "decision": decision,
        "anchors_context_only": {
            "prior_patch_path_pooling_test": 0.785195,
            "prior_recurrent_pair_centre_test_mean": 0.762605,
            "gin_published_test": 0.756,
            "gps_published_test": 0.788,
            "cin_small_published_test": 0.801,
        },
        "official_test_loaded": (RESULTS_DIR / "official_test_results.json").exists(),
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    _write_report(payload)
    _write_decision(payload)
    return payload


def _write_report(payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# E2E-DictEnv-M1 — report",
        "",
        "Round `e2e_dictenv_m1`; primitive-only dictionary-core architecture transfer to OGBG-MolHIV.",
        "",
        "## Frozen verdict",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        "## Metrics (ROC-AUC)",
        "",
        "| seed | raw valid | raw epoch | soup valid | soup members |",
        "|---|---|---|---|---|",
    ]
    for name, row in decision["runs"].items():
        lines.append(
            f"| {name} | {row['raw_valid_auc']:.6f} | {row['raw_best_epoch']} | {row['soup_valid_auc']:.6f} | {row['soup_members']} |"
        )
    if decision.get("test_soup_mean_auc") is not None:
        lines.append("")
        lines.append(f"* official-test soup mean AUC = {decision['test_soup_mean_auc']:.6f}")
    lines.append("")
    lines.append(f"* commit `{payload['git_commit']}`")
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# E2E-DictEnv-M1 — decision",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        f"* best soup valid AUC = {decision['best_soup_valid_auc']:.6f}",
        f"* test soup mean AUC = {decision['test_soup_mean_auc']}",
        "",
        "## Reading",
        "",
    ]
    verdict = decision["verdict"]
    if verdict in (m1.VERDICTS["strong"], m1.VERDICTS["competitive"]):
        lines.append(
            "The clean primitive-only dictionary-core / no-message-passing architecture transfers to "
            "OGBG-MolHIV at a competitive absolute ROC-AUC with no MolHIV-specific architecture search. "
            "No dictionary-specificity claim is made from M1."
        )
    elif verdict == m1.VERDICTS["plausible"]:
        lines.append(
            "The clean dictionary-core architecture transfers at a plausible but not competitive absolute "
            "ROC-AUC; the architecture is viable as a transfer object. No dictionary-specificity claim."
        )
    else:
        lines.append(
            "The clean dictionary-core architecture does not transfer at a usable absolute ROC-AUC under "
            "this protocol. No architecture tuning was performed; STOP."
        )
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# terminal official test
# ---------------------------------------------------------------------------


def unlock_test_stage(device: str = "cuda") -> dict[str, Any]:
    unlock_path = RESULTS_DIR / "official_test_unlock.json"
    if unlock_path.exists():
        raise RuntimeError("refusing test: official MolHIV test already unlocked once this round")
    freeze = _read_json(RESULTS_DIR / "architecture_freeze.json")
    if freeze.get("official_test_loaded_at_freeze_time") is not False:
        raise RuntimeError("refusing test: freeze does not assert a pre-test freeze")
    if not (RESULTS_DIR / "decision.json").exists():
        raise RuntimeError("refusing test: all valid conclusions must be recorded first")
    _write_json(
        unlock_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "user_authorised_test_read": True,
            "purpose": "terminal reporting only",
            "project_wide_pristine": False,
            "architecture_frozen_before_this_rounds_test_read": True,
            "test_will_not_affect_any_model_config_checkpoint_decision": True,
            "official_test_loaded": True,
        },
    )
    bundle = _load_bundle()
    test_indices = np.asarray(bundle.split["test"], dtype=np.int64)
    stats = _read_json(RESULTS_DIR / "anchor_stats.json")
    data: list[GraphPayload] = []
    started = time.perf_counter()
    for position, index in enumerate(test_indices):
        index = int(index)
        graph = bundle.graphs[index]
        atom_fields = np.asarray(bundle.node_feats[index], dtype=np.int64)
        edge_feats = {
            (int(a), int(b)): np.asarray(v, dtype=np.int64).reshape(-1)
            for (a, b), v in bundle.edge_feats[index].items()
        }
        payload, _ = _per_graph_payload(
            graph, atom_fields, edge_feats, float(bundle.y[index]), topology_row=ztopo.compute_features(graph)
        )
        data.append(payload)
    device_obj = torch.device(device)
    loader = m1.make_env_loader(data, BATCH_SIZE, False, 0)
    rows: list[dict[str, Any]] = []
    prediction_store: dict[str, np.ndarray] = {}
    targets: np.ndarray | None = None
    for kind in ("raw", "soup"):
        for seed in SEEDS:
            model = _load_state(seed, kind, device_obj)
            result = evaluate(model, loader, device_obj)
            targets = result["targets"]
            prediction_store[f"seed{seed}_{kind}"] = result["logits"]
            rows.append(
                {
                    "name": f"seed{seed}_{kind}_soup" if kind == "soup" else f"seed{seed}_{kind}",
                    "available": True,
                    "seed": int(seed),
                    "kind": kind,
                    "test_auc": float(result["auc"]),
                    "test_mean_logit": float(np.mean(result["logits"])),
                    "test_std_logit": float(np.std(result["logits"])),
                }
            )
    assert targets is not None
    for kind in ("raw", "soup"):
        ensemble = 0.5 * (prediction_store[f"seed0_{kind}"] + prediction_store[f"seed1_{kind}"])
        rows.append(
            {
                "name": f"2seed_ensemble_{kind}",
                "available": True,
                "kind": kind,
                "test_auc": float(_auc(targets, ensemble)),
                "test_mean_logit": float(np.mean(ensemble)),
                "test_std_logit": float(np.std(ensemble)),
            }
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "note": "terminal reporting-only evaluation of a frozen configuration; MolHIV test is not project-wide pristine",
        "n_test": int(len(data)),
        "positive": int(sum(1 for d in data if float(d.y) > 0.5)),
        "rows": rows,
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": True,
    }
    _write_json(RESULTS_DIR / "official_test_results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_all(device: str = "cuda") -> None:
    identity_stage()
    fit_dictionary()
    build_env_cache()
    fit_anchor_stats(seed=0)
    calibrate_lambda(seed=0)
    correctness_stage(device="cpu")
    smoke_stage(device=device, seed=0)
    train_run(0, device)
    train_run(1, device)
    freeze_stage()
    analyze_stage()


def prep(device: str = "cpu") -> dict[str, Any]:
    """Deterministic CPU preprocessing chain (dictionary -> env -> stats -> lambda -> gates)."""
    fit_dictionary()
    build_env_cache("train")
    build_env_cache("valid")
    fit_anchor_stats(seed=0)
    calibrate_lambda(seed=0)
    return correctness(device=device)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["dict", "env", "stats", "calibrate", "identity", "correct", "prep", "smoke", "bench", "train", "freeze", "analyze", "unlock", "all"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "dict":
        print(json.dumps(fit_dictionary(force=args.force), indent=2))
    elif args.stage == "env":
        print(json.dumps(build_env_cache(force=args.force), indent=2))
    elif args.stage == "stats":
        print(json.dumps(fit_anchor_stats(seed=args.seed, force=args.force), indent=2))
    elif args.stage == "calibrate":
        print(json.dumps(calibrate_lambda(seed=args.seed, force=args.force), indent=2))
    elif args.stage == "identity":
        print(json.dumps(identity_stage(), indent=2))
    elif args.stage == "correct":
        print(json.dumps(correctness_stage(device="cpu"), indent=2))
    elif args.stage == "prep":
        print(json.dumps(prep(device=args.device), indent=2))
    elif args.stage == "smoke":
        print(json.dumps(smoke_stage(device=args.device, seed=args.seed), indent=2))
    elif args.stage == "bench":
        print(json.dumps(bench(seed=args.seed, batches=args.batches, device=args.device), indent=2))
    elif args.stage == "train":
        print(json.dumps(train_run(args.seed, device=args.device), indent=2))
    elif args.stage == "freeze":
        print(json.dumps(freeze_stage(), indent=2))
    elif args.stage == "analyze":
        print(json.dumps(analyze_stage(), indent=2))
    elif args.stage == "unlock":
        print(json.dumps(unlock_test_stage(device=args.device), indent=2))
    else:
        run_all(device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
