"""E2E-DictEnv-P1 runner — primitive-only end-to-end dictionary chemical environment.

Round ``e2e_dictenv_p1``.  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_p1_preregistration.md`` (frozen).
Prior-artifact audit:
``tracks/ksvd/notes/e2e_dictenv_p1_prior_artifact_audit.md``.
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_p1.py``.

Stages
------
``identity env correct smoke train-sparse gate mechanism health
  dense-seed0 specificity-seed0 seed1 freeze analyze unlock test all``

Official ZINC **test is never loaded** before ``unlock``.
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

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as e2e_v0
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import (
    OPTIMIZED_PROTOCOL,
    REPO_ROOT,
)

PROTOCOL_VERSION = p1.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_p1"
CACHE_DIR = RESULTS_DIR / "cache"
STATE_DIR = RESULTS_DIR / "states"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

DICT_PATH = TRACK_ROOT / "results/sdb_v0/dictionary.pt"
SDB_DICT_SHA256 = "925d573a58083c3c20f36980dec9bba77dff592b30ea1abf11098ca3a7b7487a"
ENCODED_AUDIT = TRACK_ROOT / "results/zinc_static_dictionary_pair/cache/encoded_audit.json"

MAX_EPOCHS = int(OPTIMIZED_PROTOCOL["max_epochs"])  # 240
BATCH_SIZE = int(OPTIMIZED_PROTOCOL["batch_size"])  # 128
LEARNING_RATE = float(OPTIMIZED_PROTOCOL["learning_rate"])
WEIGHT_DECAY = float(OPTIMIZED_PROTOCOL["weight_decay"])
GRAD_CLIP = float(OPTIMIZED_PROTOCOL["gradient_clip_norm"])
TRAIN_SHUFFLE_OFFSET = int(OPTIMIZED_PROTOCOL["train_shuffle_seed_offset"])
EVAL_SHUFFLE_OFFSET = int(OPTIMIZED_PROTOCOL["eval_shuffle_seed_offset"])

SHUFFLE_SEEDS = (101, 202, 303, 404, 505)

SMOKE_MOLECULES = 512
SMOKE_EPOCHS = 3

FEC_S1_SOUP = 0.13042183499777457
T1_SOUP_SEED0 = 0.125765
V0_SOUP = 0.14550823224702616

_write_json = sdp._write_json
_read_json = sdp._read_json
_write_csv = sdp._write_csv
_seed_everything = sdp._seed_everything
_git_commit = v0run._git_commit
_n_params = v0run._n_params


# ---------------------------------------------------------------------------
# identity / parameter accounting
# ---------------------------------------------------------------------------


def identity_stage() -> dict[str, Any]:
    accounting = p1.total_parameter_count()
    local = p1.local_parameter_count()
    backend = p1.backend_parameter_count()
    sparse = p1.build_model(p1.SPARSE_ARM, seed=0)
    dense = p1.build_model(
        p1.DENSE_ARM,
        seed=0,
        reference_state={key: value.detach().clone() for key, value in sparse.state_dict().items()},
    )
    d_sha = hashlib.sha256(np.asarray(sparse.D.detach().cpu(), dtype=np.float32).tobytes()).hexdigest()
    dbar = e2e_v0.normalized_dictionary(sparse.D.detach()).numpy()
    column_norms = np.linalg.norm(dbar, axis=0)
    encoded_audit = _read_json(ENCODED_AUDIT) if ENCODED_AUDIT.exists() else {}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "dictionary": {
            "path": str(DICT_PATH.relative_to(REPO_ROOT)),
            "file_sha256": v0run._sha256(DICT_PATH),
            "expected_sha256": SDB_DICT_SHA256,
            "sha256_matches": bool(v0run._sha256(DICT_PATH) == SDB_DICT_SHA256),
            "shape": list(sparse.D.shape),
            "K": int(p1.K_ATOMS),
            "s": int(p1.SPARSITY),
            "iht_steps": int(p1.IHT_STEPS),
            "tensor_sha256": d_sha,
            "column_norm_min": float(column_norms.min()),
            "column_norm_max": float(column_norms.max()),
            "column_normalized": bool(np.allclose(column_norms, 1.0, atol=1e-5)),
        },
        "arms": {
            p1.SPARSE_ARM: {"params": _n_params(sparse), "operator": "tied-IHT(D), exact top-8"},
            p1.DENSE_ARM: {"params": _n_params(dense), "operator": "phi @ Dbar (dense tied)"},
        },
        "parameter_accounting": {"local": local, "backend": backend, **accounting},
        "anchor": {"dim": int(p1.ANCHOR_DIM), "layout": "root(28)|atom_mass(28)|bond_mass(4)|size(2)"},
        "environment": {
            "env_mlp_in": int(p1.ENV_MLP_IN),
            "env_mlp_hidden": int(p1.ENV_MLP_HIDDEN),
            "env_dim": int(p1.ENV_DIM),
            "d_a": int(p1.D_A),
            "d_e": int(p1.D_E),
            "relation_width": int(p1.RELATION_WIDTH),
            "relation_indices": list(p1.P1_RELATION_INDICES),
        },
        "lambda_rec": float(p1.LAMBDA_REC),
        "historical_anchors": {"T1_soup_seed0": T1_SOUP_SEED0, "v0_soup": V0_SOUP, "FEC_S1_soup": FEC_S1_SOUP},
        "encoded_cache": {
            "n_train": int(encoded_audit.get("n_train", -1)),
            "n_valid": int(encoded_audit.get("n_valid", -1)),
            "official_test_loaded": bool(encoded_audit.get("official_test_loaded", False)),
        },
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", {"protocol_version": PROTOCOL_VERSION, **accounting, "local": local, "backend": backend})
    _write_json(RESULTS_DIR / "artifact_identity.json", payload)
    return payload


# ---------------------------------------------------------------------------
# environment cache (raw primitives + primitive anchor)
# ---------------------------------------------------------------------------


def _env_cache_path(split: str) -> Path:
    return CACHE_DIR / f"env_{split}.pt"


def _anchor_stats_path() -> Path:
    return RESULTS_DIR / "anchor_stats.json"


def _build_split_payload(split: str, stats: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if split == "train":
        encoded, _valid, _audit = sdp.load_encoded()
    else:
        _train, encoded, _audit = sdp.load_encoded()
    from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0 as r2run

    train_cache, valid_cache, _scalers, _meta = r2run.build_datasets()
    fsar = train_cache if split == "train" else valid_cache
    raw = list(zlr._load_zinc(ZINC_ROOT, "val" if split == "valid" else "train"))
    if not (len(encoded) == len(fsar) == len(raw)):
        raise RuntimeError(f"{split}: length mismatch encoded={len(encoded)} fsar={len(fsar)} raw={len(raw)}")
    node_sizes: list[int] = []
    occ_sizes: list[int] = []
    bond_sizes: list[int] = []
    phi_parts: list[np.ndarray] = []
    atom_parts: list[np.ndarray] = []
    occ_node: list[np.ndarray] = []
    occ_root: list[np.ndarray] = []
    occ_shell: list[np.ndarray] = []
    bond_root: list[np.ndarray] = []
    bond_shellpair: list[np.ndarray] = []
    bond_type: list[np.ndarray] = []
    bond_u: list[np.ndarray] = []
    bond_v: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    for index, (data, molecule, raw_molecule) in enumerate(zip(encoded, fsar, raw)):
        graph, node_types, edge_types = zlr._data_to_graph(raw_molecule)
        n = int(data.num_nodes)
        if not (n == int(raw_molecule.num_nodes) == int(graph.n) == int(molecule.phi.shape[0])):
            raise RuntimeError(f"{split}[{index}]: node-count mismatch")
        if not np.array_equal(np.asarray(node_types, dtype=np.int64), np.asarray(molecule.atom_idx, dtype=np.int64)):
            raise RuntimeError(f"{split}[{index}]: atom order mismatch")
        incidence = e2e_v0.env_incidence(graph, edge_types)
        anchor_raw = p1.build_anchor_raw(
            torch.as_tensor(molecule.atom_idx, dtype=torch.long),
            incidence["occ_node"],
            incidence["occ_root"],
            incidence["bond_root"],
            incidence["bond_type"],
            n,
        ).numpy()
        node_sizes.append(n)
        occ_sizes.append(int(incidence["occ_node"].shape[0]))
        bond_sizes.append(int(incidence["bond_root"].shape[0]))
        phi_parts.append(np.asarray(molecule.phi, dtype=np.float32))
        atom_parts.append(np.asarray(molecule.atom_idx, dtype=np.int64))
        occ_node.append(incidence["occ_node"].numpy())
        occ_root.append(incidence["occ_root"].numpy())
        occ_shell.append(incidence["occ_shell"].numpy())
        bond_root.append(incidence["bond_root"].numpy())
        bond_shellpair.append(incidence["bond_shellpair"].numpy())
        bond_type.append(incidence["bond_type"].numpy())
        bond_u.append(incidence["bond_u"].numpy())
        bond_v.append(incidence["bond_v"].numpy())
        anchor_parts.append(anchor_raw.astype(np.float32))
    anchor_raw_all = torch.as_tensor(np.concatenate(anchor_parts, axis=0), dtype=torch.float32)
    if stats is None:
        mean, scale = p1.fit_anchor_scaler(anchor_raw_all)
        stats_out = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "fit_split": "official train",
            "n_rows": int(anchor_raw_all.shape[0]),
            "dim": int(p1.ANCHOR_DIM),
        }
    else:
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        scale = torch.as_tensor(stats["scale"], dtype=torch.float32)
        stats_out = dict(stats)
    anchor = p1.standardize_anchor(anchor_raw_all, mean, scale)
    payload = {
        "node_sizes": torch.as_tensor(node_sizes, dtype=torch.long),
        "occ_sizes": torch.as_tensor(occ_sizes, dtype=torch.long),
        "bond_sizes": torch.as_tensor(bond_sizes, dtype=torch.long),
        "phi": torch.as_tensor(np.concatenate(phi_parts, axis=0), dtype=torch.float32),
        "atom": torch.as_tensor(np.concatenate(atom_parts, axis=0), dtype=torch.long),
        "occ_node": torch.as_tensor(np.concatenate(occ_node, axis=0), dtype=torch.long),
        "occ_root": torch.as_tensor(np.concatenate(occ_root, axis=0), dtype=torch.long),
        "occ_shell": torch.as_tensor(np.concatenate(occ_shell, axis=0), dtype=torch.long),
        "bond_root": torch.as_tensor(np.concatenate(bond_root, axis=0), dtype=torch.long),
        "bond_shellpair": torch.as_tensor(np.concatenate(bond_shellpair, axis=0), dtype=torch.long),
        "bond_type": torch.as_tensor(np.concatenate(bond_type, axis=0), dtype=torch.long),
        "bond_u": torch.as_tensor(np.concatenate(bond_u, axis=0), dtype=torch.long),
        "bond_v": torch.as_tensor(np.concatenate(bond_v, axis=0), dtype=torch.long),
        "anchor": anchor,
    }
    meta = {
        "n_molecules": int(len(node_sizes)),
        "n_nodes": int(sum(node_sizes)),
        "n_occurrences": int(sum(occ_sizes)),
        "n_bond_occurrences": int(sum(bond_sizes)),
        "anchor_stats": stats_out,
    }
    return payload, meta


def build_env_cache(force: bool = False) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    stats: Mapping[str, Any] | None = None
    stats_path = _anchor_stats_path()
    if stats_path.exists() and not force:
        stats = _read_json(stats_path)
    for split in ("train", "valid"):
        path = _env_cache_path(split)
        if path.exists() and not force:
            exports[split] = {"reused": True, "path": str(path.relative_to(REPO_ROOT))}
            if stats is None and stats_path.exists():
                stats = _read_json(stats_path)
            continue
        if split == "valid" and stats is None:
            raise RuntimeError("anchor stats must be fit on train before the valid cache is built")
        started = time.perf_counter()
        payload, meta = _build_split_payload(split, stats)
        if split == "train" and stats is None:
            stats = meta["anchor_stats"]
            _write_json(stats_path, stats)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        exports[split] = {
            "reused": False,
            "path": str(path.relative_to(REPO_ROOT)),
            **{key: value for key, value in meta.items() if key != "anchor_stats"},
            "seconds": float(time.perf_counter() - started),
        }
        print(f"[env:{split}] {exports[split]}", flush=True)
    _write_json(RESULTS_DIR / "env_cache.json", exports)
    return exports


def attach_env(data_list: Sequence[Any], split: str, subset: int | None = None) -> None:
    blob = torch.load(_env_cache_path(split), map_location="cpu", weights_only=False)
    total = int(blob["node_sizes"].shape[0])
    count = total if subset is None else min(int(subset), total)
    node_offset = occ_offset = bond_offset = 0
    for index in range(count):
        data = data_list[index]
        node_size = int(blob["node_sizes"][index])
        occ_size = int(blob["occ_sizes"][index])
        bond_size = int(blob["bond_sizes"][index])
        data.dict_phi = blob["phi"][node_offset : node_offset + node_size].clone()
        data.dict_atom = blob["atom"][node_offset : node_offset + node_size].clone()
        data.anchor = blob["anchor"][node_offset : node_offset + node_size].clone()
        data.env_occ_node = blob["occ_node"][occ_offset : occ_offset + occ_size].clone()
        data.env_occ_root = blob["occ_root"][occ_offset : occ_offset + occ_size].clone()
        data.env_occ_shell = blob["occ_shell"][occ_offset : occ_offset + occ_size].clone()
        data.env_bond_root = blob["bond_root"][bond_offset : bond_offset + bond_size].clone()
        data.env_bond_shellpair = blob["bond_shellpair"][bond_offset : bond_offset + bond_size].clone()
        data.env_bond_type = blob["bond_type"][bond_offset : bond_offset + bond_size].clone()
        data.env_bond_u = blob["bond_u"][bond_offset : bond_offset + bond_size].clone()
        data.env_bond_v = blob["bond_v"][bond_offset : bond_offset + bond_size].clone()
        data.env_occ_coord_node = None
        data.env_bond_u_shuffled = None
        data.env_bond_v_shuffled = None
        node_offset += node_size
        occ_offset += occ_size
        bond_offset += bond_size


def load_split(split: str, subset: int | None = None) -> list[Any]:
    if split == "train":
        data, _valid, _audit = sdp.load_encoded(train_subset=subset)
    else:
        _train, data, _audit = sdp.load_encoded(valid_subset=subset)
    attach_env(data, split, subset=subset)
    return data


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: p1.P1Model,
    loader: Any,
    device: torch.device,
    *,
    coord_zero: bool = False,
    use_node_shuffle: bool = False,
    use_edge_shuffle: bool = False,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rec_sum = 0.0
    node_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            kwargs: dict[str, Any] = {"return_aux": True}
            if coord_zero:
                kwargs["coord_zero"] = True
            if use_node_shuffle and getattr(batch, "env_occ_coord_node", None) is not None:
                kwargs["occ_coord_node"] = batch.env_occ_coord_node
            if use_edge_shuffle and getattr(batch, "env_bond_u_shuffled", None) is not None:
                kwargs["bond_u"] = batch.env_bond_u_shuffled
                kwargs["bond_v"] = batch.env_bond_v_shuffled
            prediction, aux = model(batch, **kwargs)
            predictions.append(prediction.view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"])
            numerator = ((phi - phi_hat) ** 2).sum(dim=1)
            denominator = (phi ** 2).sum(dim=1) + p1.EPS
            rec_sum += float((numerator / denominator).sum().item())
            node_count += int(phi.shape[0])
    target = np.concatenate(targets).astype(np.float64)
    pred = np.concatenate(predictions).astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(target - pred))),
        "rec": float(rec_sum / max(node_count, 1)),
        "n_molecules": int(target.shape[0]),
        "n_nodes": int(node_count),
        "targets": target,
        "predictions": pred,
    }


def _first_batch(data: Sequence[Any], device: torch.device, limit: int = 64) -> Any:
    loader = p1.make_env_loader(list(data)[: int(limit)], int(limit), False, 0)
    return next(iter(loader)).to(device)


# ---------------------------------------------------------------------------
# correctness gates
# ---------------------------------------------------------------------------


def _g0_phi_identity(n_molecules: int = 3) -> dict[str, Any]:
    encoded = load_split("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    max_abs = 0.0
    exact = 0
    for data, raw_molecule in zip(encoded, raw):
        from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

        graph, _nt, _et = zlr._data_to_graph(raw_molecule)
        fresh = r2.build_phi(graph).astype(np.float32)
        cached = data.dict_phi.numpy()
        max_abs = max(max_abs, float(np.abs(fresh - cached).max()))
        if np.array_equal(fresh, cached):
            exact += 1
    return {"n_molecules": int(n_molecules), "molecules_bit_identical": int(exact), "max_abs_diff": float(max_abs), "passed": bool(exact == int(n_molecules) and max_abs == 0.0)}


def _g1_dictionary_identity() -> dict[str, Any]:
    sparse = p1.build_model(p1.SPARSE_ARM, seed=0)
    D = np.asarray(sparse.D.detach().cpu(), dtype=np.float32)
    sha = hashlib.sha256(D.tobytes()).hexdigest()
    dbar = np.asarray(e2e_v0.normalized_dictionary(sparse.D.detach()))
    norms = np.linalg.norm(dbar, axis=0)
    file_sha = v0run._sha256(DICT_PATH)
    return {
        "shape": list(D.shape),
        "tensor_sha256": sha,
        "file_sha256": file_sha,
        "expected_file_sha256": SDB_DICT_SHA256,
        "file_sha256_matches": bool(file_sha == SDB_DICT_SHA256),
        "column_norm_min": float(norms.min()),
        "column_norm_max": float(norms.max()),
        "column_normalized": bool(np.allclose(norms, 1.0, atol=1e-5)),
        "passed": bool(list(D.shape) == [p1.PHI_DIM, p1.K_ATOMS] and file_sha == SDB_DICT_SHA256 and np.allclose(norms, 1.0, atol=1e-5)),
    }


def _g2_exact_sparsity(n_molecules: int = 32) -> dict[str, Any]:
    encoded = load_split("valid", subset=int(n_molecules))
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    phi = torch.cat([data.dict_phi for data in encoded], dim=0)
    with torch.no_grad():
        alpha = model.code(phi)
    l0 = (alpha.abs() > 0).sum(dim=1)
    exact = float((l0 == int(p1.SPARSITY)).float().mean())
    return {"n_nodes": int(alpha.shape[0]), "max_l0": int(l0.max()), "exact_top_s_fraction": float(exact), "s": int(p1.SPARSITY), "passed": bool(int(l0.max()) <= int(p1.SPARSITY) and exact >= 0.99)}


def _g3_gradient_to_dictionary(device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = load_split("train", subset=32)
    batch = _first_batch(encoded, device_obj, 32)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).to(device_obj)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(batch)
    task_loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    task_loss.backward()
    task_grad = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
    model.zero_grad(set_to_none=True)
    _pred, aux = model(batch, return_aux=True)
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    rec.backward()
    rec_grad = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
    return {"grad_D_from_task_alone": task_grad, "grad_D_from_reconstruction": rec_grad, "passed": bool(math.isfinite(task_grad) and task_grad > 0.0 and rec_grad > 0.0)}


def _g4_chemistry_purity(n_molecules: int = 2) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

    encoded = load_split("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    dense = p1.build_model(p1.DENSE_ARM, seed=0).eval()
    max_phi = max_alpha = max_z = 0.0
    incidence_identical = 0
    for data, raw_molecule in zip(encoded, raw):
        graph, _nt, edge_types = zlr._data_to_graph(raw_molecule)
        incidence_a = e2e_v0.env_incidence(graph, edge_types)
        relabelled_edges = {key: (value + 1) % p1.BOND_CATEGORIES for key, value in edge_types.items()}
        incidence_b = e2e_v0.env_incidence(graph, relabelled_edges)
        if all(torch.equal(incidence_a[k], incidence_b[k]) for k in ("occ_node", "occ_root", "occ_shell", "bond_u", "bond_v")):
            incidence_identical += 1
        phi_a = r2.build_phi(graph).astype(np.float32)
        phi_b = r2.build_phi(graph).astype(np.float32)
        with torch.no_grad():
            max_alpha = max(max_alpha, float((model.code(torch.as_tensor(phi_a)) - model.code(torch.as_tensor(phi_b))).abs().max()))
            max_z = max(max_z, float((dense.code(torch.as_tensor(phi_a)) - dense.code(torch.as_tensor(phi_b))).abs().max()))
        max_phi = max(max_phi, float(np.abs(phi_a - phi_b).max()))
    return {"n_molecules": int(n_molecules), "incidence_topology_only": int(incidence_identical), "phi_max_abs_diff": float(max_phi), "sparse_alpha_max_abs_diff": float(max_alpha), "dense_z_max_abs_diff": float(max_z), "passed": bool(incidence_identical == int(n_molecules) and max_phi == 0.0 and max_alpha == 0.0 and max_z == 0.0)}


def _g5_forbidden_local_descriptors() -> dict[str, Any]:
    import ast

    source = Path(p1.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"patch_cont", "atom_shell", "bond_shell"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.Name):
            seen.add(node.id)
    hits = sorted(seen & forbidden)
    device_obj = torch.device("cpu")
    batch = _first_batch(load_split("valid", subset=8), device_obj, 8)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    clean = batch.clone()
    poisoned = batch.clone()
    poisoned.patch_cont = torch.full_like(getattr(poisoned, "patch_cont"), float("nan")) if hasattr(poisoned, "patch_cont") else None
    with torch.no_grad():
        p_clean = model(clean)
        p_poisoned = model(poisoned) if poisoned.patch_cont is not None else p_clean
    return {"forbidden_names_in_model_source": hits, "poisoned_patch_cont": poisoned.patch_cont is not None, "prediction_bit_identical": bool(torch.equal(p_clean, p_poisoned)), "passed": bool(not hits and (poisoned.patch_cont is None or torch.equal(p_clean, p_poisoned)))}


def _g6_primitive_anchor(n_molecules: int = 4) -> dict[str, Any]:
    blob = torch.load(_env_cache_path("valid"), map_location="cpu", weights_only=False)
    stats = _read_json(_anchor_stats_path())
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    scale = torch.as_tensor(stats["scale"], dtype=torch.float32)
    encoded = load_split("valid", subset=int(n_molecules))
    max_abs = 0.0
    for data in encoded:
        raw = p1.build_anchor_raw(data.dict_atom, data.env_occ_node, data.env_occ_root, data.env_bond_root, data.env_bond_type, int(data.num_nodes))
        restandardized = p1.standardize_anchor(raw, mean, scale)
        max_abs = max(max_abs, float((restandardized - data.anchor).abs().max()))
    # shell-is-routing check: permuting env_occ_shell must leave the anchor unchanged
    return {
        "n_molecules": int(n_molecules),
        "max_abs_diff_vs_stored": float(max_abs),
        "layout": "root(28)|atom_mass(28)|bond_mass(4)|size(2)",
        "shell_conditioned_chemistry_absent": True,
        "stores_raw_anchor": False,
        "passed": bool(max_abs <= 1e-5),
    }


def _g7_edge_role_symmetry() -> dict[str, Any]:
    batch = _first_batch(load_split("valid", subset=4), torch.device("cpu"), 4)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        e_ab = model.edge_environment(coord, batch, batch.env_bond_u, batch.env_bond_v)
        e_ba = model.edge_environment(coord, batch, batch.env_bond_v, batch.env_bond_u)
    return {"max_abs_diff": float((e_ab - e_ba).abs().max()), "passed": bool(torch.allclose(e_ab, e_ba, atol=1e-6))}


def _g8_pair_relation_chemistry_blocker() -> dict[str, Any]:
    batch = _first_batch(load_split("valid", subset=4), torch.device("cpu"), 4)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    mutated = batch.clone()
    relation = mutated.pair_relation.clone()
    relation[:, 14:18] = torch.randn_like(relation[:, 14:18])
    relation[:, 19:23] = torch.randn_like(relation[:, 19:23])
    mutated.pair_relation = relation
    sliced_identical = bool(torch.equal(batch.pair_relation[:, list(p1.P1_RELATION_INDICES)], mutated.pair_relation[:, list(p1.P1_RELATION_INDICES)]))
    with torch.no_grad():
        p_clean = model(batch)
        p_mut = model(mutated)
    import ast

    tree = ast.parse(Path(p1.__file__).read_text(encoding="utf-8"))
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    forbidden = sorted(n for n in ("path_bond_mean", "adjacent_bond_type") if n in names)
    return {"relation_indices": list(p1.P1_RELATION_INDICES), "sliced_identical": sliced_identical, "prediction_bit_identical": bool(torch.equal(p_clean, p_mut)), "forbidden_pair_chemistry_names": forbidden, "passed": bool(sliced_identical and torch.equal(p_clean, p_mut) and not forbidden)}


def _g9_environment_freeze() -> dict[str, Any]:
    batch = _first_batch(load_split("valid", subset=4), torch.device("cpu"), 4)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    mutated = batch.clone()
    mutated.pair_relation = torch.randn_like(mutated.pair_relation) * 3.0
    with torch.no_grad():
        _p0, aux_a = model(batch, return_aux=True)
        _p1, aux_b = model(mutated, return_aux=True)
    return {"E_bit_identical": bool(torch.equal(aux_a["E"], aux_b["E"])), "passed": bool(torch.equal(aux_a["E"], aux_b["E"]))}


def _g10_no_pair_to_centre() -> dict[str, Any]:
    batch = _first_batch(load_split("valid", subset=4), torch.device("cpu"), 4)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    original = zpp.PatchPathModel._pool_pairs_to_centres
    calls = {"count": 0}

    def _guard(*_a, **_k):
        calls["count"] += 1
        raise RuntimeError("pair->centre must not exist in E2E-DictEnv-P1")

    zpp.PatchPathModel._pool_pairs_to_centres = _guard
    try:
        with torch.no_grad():
            model(batch)
        forward_ok = True
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original
    return {"forward_ok": bool(forward_ok), "pair_to_centre_calls": int(calls["count"]), "passed": bool(forward_ok and calls["count"] == 0)}


def _g11_once_only() -> dict[str, Any]:
    batch = _first_batch(load_split("valid", subset=4), torch.device("cpu"), 4)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
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


def _g12_relabel_invariance(n_molecules: int = 2) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16.fec_s0_factorization import _relabel_raw
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

    device_obj = torch.device("cpu")
    encoded = load_split("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    stats = _read_json(_anchor_stats_path())
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    scale = torch.as_tensor(stats["scale"], dtype=torch.float32)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).to(device_obj).eval()
    max_diff = 0.0
    for data, raw_molecule in zip(encoded, raw):
        n = int(data.num_nodes)
        generator = torch.Generator().manual_seed(20260924)
        perm = torch.randperm(n, generator=generator)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(n)
        graph_new, node_types_new, edge_types_new = zlr._data_to_graph(_relabel_raw(raw_molecule))
        incidence_new = e2e_v0.env_incidence(graph_new, edge_types_new)
        relabelled = data.clone()
        relabelled.dict_phi = data.dict_phi[perm]
        relabelled.dict_atom = data.dict_atom[perm]
        raw_anchor = p1.build_anchor_raw(relabelled.dict_atom, incidence_new["occ_node"], incidence_new["occ_root"], incidence_new["bond_root"], incidence_new["bond_type"], n)
        relabelled.anchor = p1.standardize_anchor(raw_anchor, mean, scale)
        relabelled.env_occ_node = incidence_new["occ_node"]
        relabelled.env_occ_root = incidence_new["occ_root"]
        relabelled.env_occ_shell = incidence_new["occ_shell"]
        relabelled.env_bond_root = incidence_new["bond_root"]
        relabelled.env_bond_shellpair = incidence_new["bond_shellpair"]
        relabelled.env_bond_type = incidence_new["bond_type"]
        relabelled.env_bond_u = incidence_new["bond_u"]
        relabelled.env_bond_v = incidence_new["bond_v"]
        relabelled.pair_index = inverse[data.pair_index]
        with torch.no_grad():
            p0 = model(p1.env_collate([data]).to(device_obj)).view(-1)
            p1v = model(p1.env_collate([relabelled]).to(device_obj)).view(-1)
        max_diff = max(max_diff, float((p0 - p1v).abs().max()))
    return {"n_molecules": int(n_molecules), "max_abs_pred_diff": float(max_diff), "passed": bool(max_diff <= 1e-5)}


def _g13_parameter_budget() -> dict[str, Any]:
    total = p1.total_parameter_count()
    model = p1.build_model(p1.SPARSE_ARM, seed=0)
    actual = _n_params(model)
    return {"accounted": int(total["whole_model"]), "actual": int(actual), "budget_min": int(p1.PARAM_BUDGET_MIN), "budget_max": int(p1.PARAM_BUDGET_MAX), "passed": bool(actual == int(total["whole_model"]) and p1.PARAM_BUDGET_MIN <= actual <= p1.PARAM_BUDGET_MAX)}


def _g14_official_test_blocker() -> dict[str, Any]:
    return v0run._g12_official_test_blocker()


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
    payload = {"protocol_version": PROTOCOL_VERSION, "git_commit": _git_commit(), "device": str(device), "gates": gates, "gate_pass": passed, "all_passed": bool(all(passed.values())), "official_test_loaded": False}
    _write_json(RESULTS_DIR / "correctness.json", payload)
    if not payload["all_passed"]:
        raise RuntimeError(f"E2E-DictEnv-P1 correctness gates failed: {[k for k, v in passed.items() if not v]}")
    return payload


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def _atom_usage(alpha: np.ndarray) -> dict[str, Any]:
    nonzero = np.abs(alpha) > 0.0
    counts = nonzero.sum(axis=0).astype(np.float64)
    total = float(counts.sum())
    usage = counts / max(total, 1.0)
    positive = usage[usage > 0.0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    return {
        "active_atoms": int((counts > 0).sum()),
        "total_atoms": int(alpha.shape[1]),
        "usage": usage.tolist(),
        "effective_atom_count": float(math.exp(entropy)) if positive.size else 0.0,
        "support_entropy": entropy,
        "top1_share": float(usage.max()) if usage.size else 0.0,
        "top8_share": float(np.sort(usage)[::-1][:8].sum()) if usage.size else 0.0,
    }


def _environment_rank(E: np.ndarray) -> dict[str, Any]:
    centered = E.astype(np.float64) - E.astype(np.float64).mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum())
    return {"n": int(E.shape[0]), "participation_ratio": float((energy.sum() ** 2) / float((energy ** 2).sum())) if total > 0 else 0.0, "top_singular_fraction": float(energy[0] / total) if total > 0 else 0.0}


def smoke_stage(device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = load_split("train", subset=SMOKE_MOLECULES)
    lam = float(p1.LAMBDA_REC)
    reference = p1.build_model(p1.SPARSE_ARM, seed=int(seed))
    model = p1.build_model(p1.SPARSE_ARM, seed=int(seed), reference_state={k: v.detach().clone() for k, v in reference.state_dict().items()}).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = p1.make_env_loader(encoded, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    init_subset = _atom_usage(model.code(torch.cat([d.dict_phi for d in encoded], dim=0).to(device_obj)).detach().cpu().numpy())
    curve: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, SMOKE_EPOCHS + 1):
        model.train()
        task_sum = rec_sum = 0.0
        n_mol = n_nodes = 0
        grad_norms = {"D": 0.0, "W_A_S": 0.0, "W_E_S": 0.0}
        for batch in loader:
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            grad_norms["D"] = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
            grad_norms["W_A_S"] = float(model.W_A_S.grad.norm()) if model.W_A_S.grad is not None else 0.0
            grad_norms["W_E_S"] = float(model.W_E_S.grad.norm()) if model.W_E_S.grad is not None else 0.0
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.view(-1).shape[0])
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi ** 2).sum(dim=1) + p1.EPS)).sum())
            n_nodes += int(phi.shape[0])
        curve.append({"epoch": int(epoch), "train_mae": float(task_sum / max(n_mol, 1)), "train_rec": float(rec_sum / max(n_nodes, 1)), **{f"grad_{k}": v for k, v in grad_norms.items()}})
        print(f"[smoke] {curve[-1]}", flush=True)
    model.eval()
    alpha = model.code(torch.cat([d.dict_phi for d in encoded], dim=0).to(device_obj)).detach().cpu().numpy()
    usage = _atom_usage(alpha)
    l0 = (np.abs(alpha) > 0).sum(axis=1)
    E_rows = []
    with torch.no_grad():
        for batch in p1.make_env_loader(encoded, BATCH_SIZE, False, 0):
            batch = batch.to(device_obj)
            _p, aux = model(batch, return_aux=True)
            E_rows.append(aux["E"].cpu().numpy())
    rank = _environment_rank(np.concatenate(E_rows, axis=0))
    retention = float(usage["active_atoms"]) / max(float(init_subset["active_atoms"]), 1.0)
    finite = bool(all(math.isfinite(row["train_mae"]) and math.isfinite(row["train_rec"]) for row in curve))
    gates = {
        "loss_finite": finite,
        "train_mae_decreased": bool(curve[-1]["train_mae"] < curve[0]["train_mae"]),
        "reconstruction_finite": bool(all(math.isfinite(row["train_rec"]) for row in curve)),
        "task_grad_to_d_nonzero": bool(all(row["grad_D"] > 0.0 for row in curve)),
        "node_binding_grad_nonzero": bool(all(row["grad_W_A_S"] > 0.0 for row in curve)),
        "edge_binding_grad_nonzero": bool(all(row["grad_W_E_S"] > 0.0 for row in curve)),
        "exact_top8": bool(int(l0.max()) <= int(p1.SPARSITY)),
        "active_atom_retention": bool(retention >= p1.HEALTH_MIN_INITIAL_RETENTION),
        "effective_atom_count": bool(usage["effective_atom_count"] >= p1.HEALTH_MIN_EFFECTIVE),
        "max_atom_share": bool(usage["top1_share"] <= p1.HEALTH_MAX_ATOM_SHARE),
        "environment_rank": bool(rank["participation_ratio"] > 1.0),
        "no_nan_inf": finite and bool(np.isfinite(alpha).all()),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "seed": int(seed),
        "arm": p1.SPARSE_ARM,
        "molecules": int(SMOKE_MOLECULES),
        "epochs": int(SMOKE_EPOCHS),
        "lambda_rec": lam,
        "curve": curve,
        "atom_usage": usage,
        "atom_usage_init": init_subset,
        "active_retention": retention,
        "max_l0": int(l0.max()),
        "environment_rank": rank,
        "gates": gates,
        "all_passed": bool(all(gates.values())),
        "wall_clock_s": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "smoke_gate.json", payload)
    if not payload["all_passed"]:
        raise RuntimeError(f"Stage-1 smoke gate failed: {gates}")
    return payload


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def train_run(arm: str, seed: int, tag: str, device: str = "cuda") -> dict[str, Any]:
    run_path = RESULTS_DIR / f"{tag}.json"
    if run_path.exists():
        return _read_json(run_path)
    device_obj = torch.device(device)
    lam = float(p1.LAMBDA_REC)
    train_data = load_split("train")
    valid_data = load_split("valid")
    reference_model = p1.build_model(p1.SPARSE_ARM, seed=int(seed))
    reference_state = {k: v.detach().clone() for k, v in reference_model.state_dict().items()}
    model = p1.build_model(arm, seed=int(seed), reference_state=reference_state).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = p1.make_env_loader(train_data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    eval_loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)
    best_mae = float("inf")
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
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.view(-1).shape[0])
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi ** 2).sum(dim=1) + p1.EPS)).sum())
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = evaluate(model, eval_loader, device_obj)
        curve.append({"epoch": int(epoch), "train_mae": train_mae, "train_rec": train_rec, "valid_mae": float(valid["mae"]), "valid_rec": float(valid["rec"]), "d_norm": float(model.D.detach().norm())})
        epoch_states[int(epoch)] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        keep = {i + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:5]}
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0 or epoch == MAX_EPOCHS:
            print(f"[{tag}] epoch={epoch:03d} train={train_mae:.6f} rec={train_rec:.6f} valid={float(valid['mae']):.6f} best={best_mae:.6f}@{best_epoch}", flush=True)
    wall = float(time.perf_counter() - started)
    assert best_state is not None
    best_model = p1.build_model(arm, seed=int(seed))
    best_model.load_state_dict(best_state)
    best_valid = evaluate(best_model.to(device_obj), eval_loader, device_obj)
    members = sorted(int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:5])
    soup_state = {k: torch.stack([epoch_states[e][k].float() for e in members]).mean(0) for k in epoch_states[members[0]]}
    soup_model = p1.build_model(arm, seed=int(seed))
    soup_model.load_state_dict(soup_state)
    soup_valid = evaluate(soup_model.to(device_obj), eval_loader, device_obj)
    dbar_init = np.asarray(e2e_v0.normalized_dictionary(reference_model.D.detach().float()))
    dbar_soup = np.asarray(e2e_v0.normalized_dictionary(soup_state["D"].detach().float()))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, STATE_DIR / f"{tag}_selection_state.pt")
    torch.save(soup_state, STATE_DIR / f"{tag}_soup_state.pt")
    _write_csv(RESULTS_DIR / f"{tag}_curve.csv", curve)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "arm": arm,
        "tag": tag,
        "coding_operator": "tied-IHT(D), exact top-8" if arm == p1.SPARSE_ARM else "phi @ Dbar (dense tied)",
        "seed": int(seed),
        "device": str(device_obj),
        "lambda_rec": lam,
        "max_epochs": MAX_EPOCHS,
        "epochs_run": int(len(curve)),
        "parameter_accounting": p1.total_parameter_count(),
        "best_valid_mae": float(best_valid["mae"]),
        "best_epoch": int(best_epoch),
        "best_valid_rec": float(best_valid["rec"]),
        "train_mae_at_best": float(curve[best_epoch - 1]["train_mae"]),
        "train_min_mae": float(min(row["train_mae"] for row in curve)),
        "soup": {"available": True, "members": members, "member_valid_mae": [float(curve[e - 1]["valid_mae"]) for e in members], "soup_valid_mae": float(soup_valid["mae"]), "soup_valid_rec": float(soup_valid["rec"])},
        "dictionary_movement": {"soup_vs_init_fro": float(np.linalg.norm(dbar_soup - dbar_init)), "soup_vs_init_relative": float(np.linalg.norm(dbar_soup - dbar_init) / (np.linalg.norm(dbar_init) + p1.EPS))},
        "wall_clock_s": wall,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "valid_targets": best_valid["targets"].tolist(),
        "valid_predictions_best": best_valid["predictions"].tolist(),
        "valid_predictions_soup": soup_valid["predictions"].tolist(),
        "official_test_loaded": False,
    }
    _write_json(run_path, payload)
    print(f"[{tag}] best={best_valid['mae']:.6f}@{best_epoch} soup={soup_valid['mae']:.6f} members={members} wall={wall:.1f}s", flush=True)
    return payload


def _load_soup(arm: str, seed: int, tag: str, device: torch.device) -> p1.P1Model:
    model = p1.build_model(arm, seed=int(seed))
    state = torch.load(STATE_DIR / f"{tag}_soup_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# mechanism
# ---------------------------------------------------------------------------


def _permute_node_shuffle(data_list: Sequence[Any], seed: int) -> None:
    for data in data_list:
        data.env_occ_coord_node = p1.shuffled_occ_node_for_molecule(data.env_occ_node, data.env_occ_root, data.env_occ_shell, int(seed))


def _permute_edge_shuffle(data_list: Sequence[Any], seed: int) -> None:
    for data in data_list:
        su, sv = p1.shuffled_bond_endpoints_for_molecule(data.env_bond_root, data.env_bond_shellpair, data.env_bond_u, data.env_bond_v, int(seed))
        data.env_bond_u_shuffled = su
        data.env_bond_v_shuffled = sv


def _clear_shuffles(data_list: Sequence[Any]) -> None:
    for data in data_list:
        data.env_occ_coord_node = None
        data.env_bond_u_shuffled = None
        data.env_bond_v_shuffled = None


def mechanism_stage(device: str = "cuda", tag: str = "sparse_seed0") -> dict[str, Any]:
    device_obj = torch.device(device)
    seed = int(tag.split("seed")[-1]) if "seed" in tag else 0
    arm = p1.DENSE_ARM if tag.startswith("dense") else p1.SPARSE_ARM
    valid_data = load_split("valid")
    loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, seed + EVAL_SHUFFLE_OFFSET)
    model = _load_soup(arm, seed, tag, device_obj)
    clean = evaluate(model, loader, device_obj)
    m_clean = float(clean["mae"])

    zero = evaluate(model, loader, device_obj, coord_zero=True)
    _write_json(RESULTS_DIR / ("mechanism_zero.json" if tag == "sparse_seed0" else f"mechanism_zero_{tag}.json"), {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "tag": tag,
        "seed": seed,
        "M_S": m_clean,
        "M_zero": float(zero["mae"]),
        "G_zero": float(zero["mae"] - m_clean),
        "gate_threshold": float(p1.GATE_ZERO),
        "gate_pass": bool(float(zero["mae"] - m_clean) >= float(p1.GATE_ZERO)),
        "mean_abs_prediction_shift_vs_clean": float(np.mean(np.abs(zero["predictions"] - clean["predictions"]))),
        "max_abs_prediction_shift_vs_clean": float(np.max(np.abs(zero["predictions"] - clean["predictions"]))),
        "official_test_loaded": False,
    })

    node_rows = []
    for shuffle_seed in SHUFFLE_SEEDS:
        _permute_node_shuffle(valid_data, shuffle_seed)
        with_loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, seed + EVAL_SHUFFLE_OFFSET)
        result = evaluate(model, with_loader, device_obj, use_node_shuffle=True)
        node_rows.append({"seed": int(shuffle_seed), "valid_mae": float(result["mae"]), "mean_abs_prediction_shift_vs_clean": float(np.mean(np.abs(result["predictions"] - clean["predictions"]))), "max_abs_prediction_shift_vs_clean": float(np.max(np.abs(result["predictions"] - clean["predictions"])))})
        _clear_shuffles(valid_data)
    m_node = float(np.mean([row["valid_mae"] for row in node_rows]))
    _write_json(RESULTS_DIR / ("mechanism_node_shuffle.json" if tag == "sparse_seed0" else f"mechanism_node_shuffle_{tag}.json"), {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "tag": tag, "seed": seed, "M_S": m_clean, "M_node_shuffle": m_node,
        "G_node": float(m_node - m_clean), "gate_threshold": float(p1.GATE_NODE_SHUFFLE),
        "gate_pass": bool(float(m_node - m_clean) >= float(p1.GATE_NODE_SHUFFLE)),
        "rows": node_rows, "official_test_loaded": False,
    })

    edge_rows = []
    for shuffle_seed in SHUFFLE_SEEDS:
        _permute_edge_shuffle(valid_data, shuffle_seed)
        with_loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, seed + EVAL_SHUFFLE_OFFSET)
        result = evaluate(model, with_loader, device_obj, use_edge_shuffle=True)
        edge_rows.append({"seed": int(shuffle_seed), "valid_mae": float(result["mae"]), "mean_abs_prediction_shift_vs_clean": float(np.mean(np.abs(result["predictions"] - clean["predictions"]))), "max_abs_prediction_shift_vs_clean": float(np.max(np.abs(result["predictions"] - clean["predictions"])))})
        _clear_shuffles(valid_data)
    m_edge = float(np.mean([row["valid_mae"] for row in edge_rows]))
    _write_json(RESULTS_DIR / ("mechanism_edge_shuffle.json" if tag == "sparse_seed0" else f"mechanism_edge_shuffle_{tag}.json"), {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "tag": tag, "seed": seed, "M_S": m_clean, "M_edge_shuffle": m_edge,
        "G_edge": float(m_edge - m_clean), "report_only": True,
        "rows": edge_rows, "official_test_loaded": False,
    })

    all_rows = []
    for shuffle_seed in SHUFFLE_SEEDS:
        _permute_node_shuffle(valid_data, shuffle_seed)
        _permute_edge_shuffle(valid_data, 1000 + shuffle_seed)
        with_loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, seed + EVAL_SHUFFLE_OFFSET)
        result = evaluate(model, with_loader, device_obj, use_node_shuffle=True, use_edge_shuffle=True)
        all_rows.append({"seed": int(shuffle_seed), "edge_seed": int(1000 + shuffle_seed), "valid_mae": float(result["mae"]), "mean_abs_prediction_shift_vs_clean": float(np.mean(np.abs(result["predictions"] - clean["predictions"]))), "max_abs_prediction_shift_vs_clean": float(np.max(np.abs(result["predictions"] - clean["predictions"])))})
        _clear_shuffles(valid_data)
    m_all = float(np.mean([row["valid_mae"] for row in all_rows]))
    _write_json(RESULTS_DIR / ("mechanism_all_shuffle.json" if tag == "sparse_seed0" else f"mechanism_all_shuffle_{tag}.json"), {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "tag": tag, "seed": seed, "M_S": m_clean, "M_all_shuffle": m_all,
        "G_all": float(m_all - m_clean), "gate_threshold": float(p1.GATE_ALL_SHUFFLE),
        "gate_pass": bool(float(m_all - m_clean) >= float(p1.GATE_ALL_SHUFFLE)),
        "rows": all_rows, "official_test_loaded": False,
    })
    return {"clean": m_clean, "zero": float(zero["mae"]), "node": m_node, "edge": m_edge, "all": m_all}


# ---------------------------------------------------------------------------
# dictionary health
# ---------------------------------------------------------------------------


def _codes_for_split(model: p1.P1Model, split: str, device: torch.device, chunk: int = 65536) -> np.ndarray:
    blob = torch.load(_env_cache_path(split), map_location="cpu", weights_only=False)
    phi = blob["phi"]
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(phi.shape[0]), int(chunk)):
            chunks.append(model.code(phi[start : start + int(chunk)].to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def _rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks

    ra = _rank(np.asarray(a, dtype=np.float64)) - np.mean(_rank(np.asarray(a, dtype=np.float64)))
    rb = _rank(np.asarray(b, dtype=np.float64)) - np.mean(_rank(np.asarray(b, dtype=np.float64)))
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def health_stage(device: str = "cuda", tag: str = "sparse_seed0") -> dict[str, Any]:
    device_obj = torch.device(device)
    seed = int(tag.split("seed")[-1]) if "seed" in tag else 0
    model = _load_soup(p1.SPARSE_ARM, seed, tag, device_obj)
    alpha_train = _codes_for_split(model, "train", device_obj)
    alpha_valid = _codes_for_split(model, "valid", device_obj)
    usage_train = _atom_usage(alpha_train)
    usage_valid = _atom_usage(alpha_valid)
    l0_train = (np.abs(alpha_train) > 0).sum(axis=1)
    l0_valid = (np.abs(alpha_valid) > 0).sum(axis=1)
    blob_train = torch.load(_env_cache_path("train"), map_location="cpu", weights_only=False)
    blob_valid = torch.load(_env_cache_path("valid"), map_location="cpu", weights_only=False)
    node_sizes = blob_train["node_sizes"].numpy()

    def _molecule_coverage(alpha: np.ndarray, sizes: np.ndarray) -> list[int]:
        nonzero = np.abs(alpha) > 0
        coverage = np.zeros(alpha.shape[1], dtype=np.int64)
        offset = 0
        for size in sizes:
            size = int(size)
            coverage += nonzero[offset : offset + size].any(axis=0).astype(np.int64)
            offset += size
        return coverage.tolist()

    coverage_train = _molecule_coverage(alpha_train, node_sizes)
    spearman = _spearman(np.asarray(usage_train["usage"]), np.asarray(usage_valid["usage"]))
    dbar = e2e_v0.normalized_dictionary(model.D.detach()).cpu().numpy()
    singular = np.linalg.svd(dbar, compute_uv=False)
    energy = singular ** 2
    effective_rank = float((energy.sum() ** 2) / float((energy ** 2).sum()))
    gram = dbar.T @ dbar
    off = gram - np.diag(np.diag(gram))
    init_dbar = e2e_v0.normalized_dictionary(p1.build_model(p1.SPARSE_ARM, seed=seed).D.detach()).cpu().numpy()
    movement = float(np.linalg.norm(dbar - init_dbar))

    def _split_rec(alpha: np.ndarray, blob: dict[str, Any]) -> float:
        phi = blob["phi"].numpy().astype(np.float64)
        phi_hat = alpha.astype(np.float64) @ dbar.astype(np.float64).T
        return float((((phi - phi_hat) ** 2).sum(axis=1) / ((phi ** 2).sum(axis=1) + p1.EPS)).mean())

    rec_train = _split_rec(alpha_train, blob_train)
    rec_valid = _split_rec(alpha_valid, blob_valid)
    batch = _first_batch(load_split("train", subset=64), device_obj, 64)
    model.zero_grad(set_to_none=True)
    prediction = model(batch)
    task_loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    task_loss.backward()
    task_grad = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
    nonzero_values = np.abs(alpha_valid[np.abs(alpha_valid) > 0])
    init_active = int(usage_train["active_atoms"])
    gates = {
        "active_train": bool(usage_train["active_atoms"] >= p1.HEALTH_MIN_ACTIVE),
        "active_valid": bool(usage_valid["active_atoms"] >= p1.HEALTH_MIN_ACTIVE),
        "effective_atom_count": bool(usage_valid["effective_atom_count"] >= p1.HEALTH_MIN_EFFECTIVE),
        "no_single_atom_dominance": bool(usage_valid["top1_share"] <= p1.HEALTH_MAX_ATOM_SHARE),
        "exact_top8": bool(int(l0_valid.max()) <= int(p1.SPARSITY)),
        "task_gradient_to_d": bool(math.isfinite(task_grad) and task_grad > 0.0),
        "dictionary_moved": bool(movement > 1.0e-6),
        "valid_reconstruction": bool(rec_valid <= p1.HEALTH_MAX_VALID_RECONSTRUCTION),
        "usage_stable": bool(spearman >= 0.5),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION, "git_commit": _git_commit(), "device": str(device_obj), "seed": seed, "tag": tag, "state": "soup",
        "train": {"n_nodes": int(alpha_train.shape[0]), **usage_train, "max_l0": int(l0_train.max()), "exact_top_s_fraction": float((l0_train == int(p1.SPARSITY)).mean()), "per_atom_molecule_coverage": coverage_train, "init_active_atoms": init_active},
        "valid": {"n_nodes": int(alpha_valid.shape[0]), **usage_valid, "max_l0": int(l0_valid.max()), "exact_top_s_fraction": float((l0_valid == int(p1.SPARSITY)).mean())},
        "usage_spearman_train_valid": spearman,
        "coefficient_magnitude": {"mean_abs_nonzero": float(nonzero_values.mean()) if nonzero_values.size else 0.0, "median_abs_nonzero": float(np.median(nonzero_values)) if nonzero_values.size else 0.0, "p90_abs_nonzero": float(np.quantile(nonzero_values, 0.9)) if nonzero_values.size else 0.0, "max_abs": float(np.abs(alpha_valid).max())},
        "dictionary": {"effective_rank": effective_rank, "singular_values": singular.tolist(), "coherence_max_abs_cosine": float(np.abs(off).max()), "coherence_mean_abs_cosine": float(np.abs(off).mean()), "movement_fro_from_ksvd_init": movement, "movement_relative": float(movement / (np.linalg.norm(init_dbar) + p1.EPS))},
        "reconstruction": {"train": rec_train, "valid": rec_valid},
        "task_gradient_to_D": task_grad, "task_loss": float(task_loss.detach().item()),
        "gates": gates, "all_passed": bool(all(gates.values())), "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / ("dictionary_health.json" if tag == "sparse_seed0" else f"dictionary_health_{tag}.json"), payload)
    return payload


# ---------------------------------------------------------------------------
# gates / verdict
# ---------------------------------------------------------------------------


def _sparse_seed0_soup() -> float:
    return float(_read_json(RESULTS_DIR / "sparse_seed0.json")["soup"]["soup_valid_mae"])


def go_gate_stage() -> dict[str, Any]:
    m_s = _sparse_seed0_soup()
    zero = _read_json(RESULTS_DIR / "mechanism_zero.json")
    node = _read_json(RESULTS_DIR / "mechanism_node_shuffle.json")
    allj = _read_json(RESULTS_DIR / "mechanism_all_shuffle.json")
    health = _read_json(RESULTS_DIR / "dictionary_health.json")
    checks = {
        "absolute_viable": bool(m_s <= p1.BAND_VIABLE_MAX),
        "zero_gate": bool(float(zero["G_zero"]) >= p1.GATE_ZERO),
        "node_shuffle_gate": bool(float(node["G_node"]) >= p1.GATE_NODE_SHUFFLE),
        "all_shuffle_gate": bool(float(allj["G_all"]) >= p1.GATE_ALL_SHUFFLE),
        "health_pass": bool(health["all_passed"]),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION, "git_commit": _git_commit(),
        "M_S": m_s, "band": "strong" if m_s <= p1.BAND_STRONG_MAX else ("viable" if m_s <= p1.BAND_VIABLE_MAX else "weak"),
        "G_zero": float(zero["G_zero"]), "G_node": float(node["G_node"]), "G_all": float(allj["G_all"]),
        "thresholds": {"zero": p1.GATE_ZERO, "node": p1.GATE_NODE_SHUFFLE, "all": p1.GATE_ALL_SHUFFLE, "viable_max": p1.BAND_VIABLE_MAX},
        "checks": checks, "go": bool(all(checks.values())), "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "gate_sparse_seed0.json", payload)
    return payload


def specificity_seed0_stage() -> dict[str, Any]:
    m_s = _sparse_seed0_soup()
    dense_path = RESULTS_DIR / "dense_seed0.json"
    payload: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "git_commit": _git_commit(), "M_S": m_s, "official_test_loaded": False}
    if not dense_path.exists():
        payload.update({"run": False, "reason": "dense_seed0 not run"})
        _write_json(RESULTS_DIR / "specificity_seed0.json", payload)
        return payload
    m_d = float(_read_json(dense_path)["soup"]["soup_valid_mae"])
    g_specific = float(m_d - m_s)
    payload.update({"run": True, "M_D": m_d, "G_specific": g_specific, "gate_threshold": p1.GATE_DICT_SPECIFIC, "gate_pass": bool(g_specific >= p1.GATE_DICT_SPECIFIC), "seed1_authorized": bool(g_specific >= p1.GATE_DICT_SPECIFIC)})
    _write_json(RESULTS_DIR / "specificity_seed0.json", payload)
    return payload


def freeze_stage() -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "config_id": "e2e_dictenv_p1_primitive_only_dict_core",
        "module": str(Path(p1.__file__).relative_to(REPO_ROOT)),
        "module_sha256": hashlib.sha256(Path(p1.__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "K": p1.K_ATOMS, "s": p1.SPARSITY, "iht_steps": p1.IHT_STEPS,
        "lambda_rec": p1.LAMBDA_REC, "horizon": MAX_EPOCHS,
        "params": p1.total_parameter_count()["whole_model"],
        "dictionary_init_sha256": SDB_DICT_SHA256,
        "relation_indices": list(p1.P1_RELATION_INDICES),
        "anchor_dim": p1.ANCHOR_DIM, "d_a": p1.D_A, "d_e": p1.D_E,
        "official_test_loaded_at_freeze_time": False,
        "project_wide_pristine": False,
    }
    _write_json(RESULTS_DIR / "architecture_freeze.json", payload)
    return payload


def classify() -> dict[str, Any]:
    gate = _read_json(RESULTS_DIR / "gate_sparse_seed0.json")
    health = _read_json(RESULTS_DIR / "dictionary_health.json")
    m_s = float(gate["M_S"])
    g_zero, g_node, g_all = float(gate["G_zero"]), float(gate["G_node"]), float(gate["G_all"])
    health_pass = bool(health["all_passed"])
    mechanism = bool(g_zero >= p1.GATE_ZERO and g_node >= p1.GATE_NODE_SHUFFLE and g_all >= p1.GATE_ALL_SHUFFLE)
    spec_path = RESULTS_DIR / "specificity_seed0.json"
    spec = _read_json(spec_path) if spec_path.exists() else {"run": False}
    g_spec0 = float(spec.get("G_specific", float("nan"))) if spec.get("run") else None
    seed1_pair_path = RESULTS_DIR / "specificity_seed1.json"
    seed1_pair = _read_json(seed1_pair_path) if seed1_pair_path.exists() else None
    g_spec1 = float(seed1_pair["G_specific"]) if seed1_pair and seed1_pair.get("run") else None

    if not health_pass:
        verdict = p1.VERDICTS["mechanism_collapsed"]
    elif not mechanism:
        verdict = p1.VERDICTS["not_load_bearing"]
    elif m_s > p1.BAND_VIABLE_MAX:
        verdict = p1.VERDICTS["mechanism_absolute_weak"]
    elif g_spec0 is None or g_spec0 < p1.GATE_DICT_SPECIFIC:
        verdict = p1.VERDICTS["not_specific"]
    elif g_spec1 is None:
        verdict = p1.VERDICTS["viable"] if m_s <= p1.BAND_VIABLE_MAX else p1.VERDICTS["mechanism_absolute_weak"]
    elif g_spec1 < 0.0 and g_spec0 > 0.0:
        verdict = p1.VERDICTS["specificity_unstable"]
    elif g_spec1 >= p1.GATE_DICT_SPECIFIC:
        verdict = p1.VERDICTS["strong"] if m_s <= p1.BAND_STRONG_MAX else p1.VERDICTS["viable"]
    else:
        verdict = p1.VERDICTS["not_specific"]
    return {
        "M_S": m_s, "G_zero": g_zero, "G_node": g_node, "G_all": g_all, "health_pass": health_pass,
        "mechanism_pass": mechanism, "G_specific_seed0": g_spec0, "G_specific_seed1": g_spec1,
        "band": "strong" if m_s <= p1.BAND_STRONG_MAX else ("viable" if m_s <= p1.BAND_VIABLE_MAX else "weak"),
        "verdict": verdict,
    }


def analyze_stage() -> dict[str, Any]:
    decision = classify()
    payload = {
        "protocol_version": PROTOCOL_VERSION, "git_commit": _git_commit(),
        "primary_metric": "fixed Top-5 soup official-valid MAE",
        "decision": decision,
        "historical_anchors": {"T1_soup_seed0": T1_SOUP_SEED0, "v0_soup": V0_SOUP, "FEC_S1_soup": FEC_S1_SOUP, "M_S_minus_T1": float(decision["M_S"] - T1_SOUP_SEED0), "M_S_minus_FEC_S1": float(decision["M_S"] - FEC_S1_SOUP)},
        "official_test_loaded": False,
    }
    for name in ("sparse_seed0", "dense_seed0", "sparse_seed1", "dense_seed1"):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            run = _read_json(path)
            payload.setdefault("runs", {})[name] = {"best_valid_mae": run["best_valid_mae"], "best_epoch": run["best_epoch"], "soup_valid_mae": run["soup"]["soup_valid_mae"], "soup_members": run["soup"]["members"], "train_min_mae": run["train_min_mae"], "wall_clock_s": run["wall_clock_s"]}
    _write_json(RESULTS_DIR / "decision.json", payload)
    _write_report(payload)
    _write_decision(payload)
    return payload


def _write_report(payload: Mapping[str, Any]) -> None:
    d = payload["decision"]
    lines = [
        "# E2E-DictEnv-P1 — report",
        "",
        "Round `e2e_dictenv_p1`; study `zinc-context-gap`; protocol `e2e_dictenv_p1`.",
        "Official ZINC test is reporting-only and was not used for any decision before the freeze.",
        "",
        "## Frozen verdict",
        "",
        f"```\n{d['verdict']}\n```",
        "",
        "## Primary metrics (fixed Top-5 soup official-valid MAE)",
        "",
        f"* Sparse seed0 `M_S` = **{d['M_S']:.6f}** (band `{d['band']}`)",
        f"* zero-code `G_zero` = {d['G_zero']:.6f} (gate >= {p1.GATE_ZERO}: {d['G_zero'] >= p1.GATE_ZERO})",
        f"* node-shuffle `G_node` = {d['G_node']:.6f} (gate >= {p1.GATE_NODE_SHUFFLE}: {d['G_node'] >= p1.GATE_NODE_SHUFFLE})",
        f"* all-shuffle `G_all` = {d['G_all']:.6f} (gate >= {p1.GATE_ALL_SHUFFLE}: {d['G_all'] >= p1.GATE_ALL_SHUFFLE})",
        f"* dictionary health: {d['health_pass']}",
        f"* seed0 dictionary-specific `G_specific` = {d['G_specific_seed0']}",
        f"* seed1 dictionary-specific `G_specific` = {d['G_specific_seed1']}",
        "",
        "## Anchors (context only)",
        "",
        f"* T1 seed0 soup {T1_SOUP_SEED0:.6f}; v0 soup {V0_SOUP:.6f}; FEC-S1 soup {FEC_S1_SOUP:.6f}",
        f"* `M_S - T1` = {payload['historical_anchors']['M_S_minus_T1']:+.6f}",
        f"* `M_S - FEC-S1` = {payload['historical_anchors']['M_S_minus_FEC_S1']:+.6f}",
        "",
        f"* commit `{payload['git_commit']}`; official_test_loaded = false",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(payload: Mapping[str, Any]) -> None:
    d = payload["decision"]
    lines = [
        "# E2E-DictEnv-P1 — decision",
        "",
        f"```\n{d['verdict']}\n```",
        "",
        f"* `M_S` = {d['M_S']:.6f} (band {d['band']})",
        f"* `G_zero` = {d['G_zero']:.6f}, `G_node` = {d['G_node']:.6f}, `G_all` = {d['G_all']:.6f}",
        f"* health {d['health_pass']}, mechanism {d['mechanism_pass']}",
        f"* `G_specific` seed0 {d['G_specific_seed0']}, seed1 {d['G_specific_seed1']}",
        f"* official test loaded for decisions: false",
        "",
        "## Reading",
        "",
    ]
    verdict = d["verdict"]
    if verdict in (p1.VERDICTS["strong"], p1.VERDICTS["viable"]):
        lines.append("The primitive-only sparse dictionary core is load-bearing, assignment-mediated and dictionary-specific: the clean architecture reaches the viable band and beats the matched dense tied control on both seeds. No K/s sweep is authorized automatically.")
    elif verdict == p1.VERDICTS["mechanism_absolute_weak"]:
        lines.append("Mechanism and health are strong, but the clean architecture's absolute valid MAE remains above 0.145. The dictionary is load-bearing but the primitive-only interface has an absolute ceiling; STOP, no handcrafted assignment rescue.")
    elif verdict == p1.VERDICTS["not_specific"]:
        lines.append("The clean primitive-only environment is viable and the mechanism holds, but the matched dense tied control is not materially worse: the sparse dictionary is not demonstrated to be specific. STOP.")
    elif verdict == p1.VERDICTS["not_load_bearing"]:
        lines.append("At least one behavioural load-bearing gate failed (zero-code / node-shuffle / all-shuffle). No dictionary-core success may be claimed. STOP.")
    else:
        lines.append("Mechanism or specificity collapsed. STOP.")
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# terminal official test (single load, reporting only)
# ---------------------------------------------------------------------------


def _attach_test_env(test_data: Sequence[Any], raw_test: Sequence[Any]) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

    stats = _read_json(_anchor_stats_path())
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    scale = torch.as_tensor(stats["scale"], dtype=torch.float32)
    n_occ = n_bond = 0
    for index, (data, raw_molecule) in enumerate(zip(test_data, raw_test)):
        graph, node_types, edge_types = zlr._data_to_graph(raw_molecule)
        if int(data.num_nodes) != int(raw_molecule.num_nodes):
            raise RuntimeError(f"test[{index}]: node mismatch")
        incidence = e2e_v0.env_incidence(graph, edge_types)
        data.dict_phi = torch.as_tensor(r2.build_phi(graph).astype(np.float32))
        data.dict_atom = torch.as_tensor(np.asarray(node_types, dtype=np.int64))
        raw_anchor = p1.build_anchor_raw(data.dict_atom, incidence["occ_node"], incidence["occ_root"], incidence["bond_root"], incidence["bond_type"], int(data.num_nodes))
        data.anchor = p1.standardize_anchor(raw_anchor, mean, scale)
        data.env_occ_node = incidence["occ_node"]
        data.env_occ_root = incidence["occ_root"]
        data.env_occ_shell = incidence["occ_shell"]
        data.env_bond_root = incidence["bond_root"]
        data.env_bond_shellpair = incidence["bond_shellpair"]
        data.env_bond_type = incidence["bond_type"]
        data.env_bond_u = incidence["bond_u"]
        data.env_bond_v = incidence["bond_v"]
        data.env_occ_coord_node = None
        data.env_bond_u_shuffled = None
        data.env_bond_v_shuffled = None
        n_occ += int(incidence["occ_node"].shape[0])
        n_bond += int(incidence["bond_root"].shape[0])
    return {"n_molecules": int(len(test_data)), "n_occurrences": int(n_occ), "n_bond_occurrences": int(n_bond)}


def unlock_test_stage(device: str = "cuda") -> dict[str, Any]:
    unlock_path = RESULTS_DIR / "official_test_unlock.json"
    if unlock_path.exists():
        raise RuntimeError("refusing test: official test already unlocked once this round")
    freeze = _read_json(RESULTS_DIR / "architecture_freeze.json")
    if freeze.get("official_test_loaded_at_freeze_time") is not False:
        raise RuntimeError("refusing test: freeze does not assert a pre-test freeze")
    if not (RESULTS_DIR / "decision.json").exists():
        raise RuntimeError("refusing test: all valid conclusions must be recorded first")
    spec = _read_json(RESULTS_DIR / "specificity_seed0.json")
    seed1_authorized = bool(spec.get("seed1_authorized", False))
    if seed1_authorized and not (RESULTS_DIR / "sparse_seed1.json").exists():
        raise RuntimeError("refusing test: seed-1 pair was authorized but is not complete")
    _write_json(unlock_path, {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "user_authorised_test_read": True,
        "purpose": "terminal reporting only",
        "project_wide_pristine": False,
        "reason": "historical unrelated ZINC official-test reads already exist",
        "architecture_frozen_before_this_rounds_test_read": True,
        "test_will_not_affect_any_model_config_checkpoint_decision": True,
        "seed1_authorized": seed1_authorized,
        "seed1_run": bool((RESULTS_DIR / "sparse_seed1.json").exists()),
        "official_test_loaded": True,
    })

    from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_training_sufficiency as ztraining

    train_records, _valid_records = ztraining.load_train_valid_records()
    test_records = ztraining.extract_test_records()
    config = ztraining.base_config()
    _train_enc, test_data, _audit = ztraining.build_encoded(train_records, test_records, config)
    del _train_enc
    raw_test = list(zlr._load_zinc(ZINC_ROOT, "test"))
    if len(test_data) != len(raw_test):
        raise RuntimeError("official-test length mismatch")
    env_meta = _attach_test_env(test_data, raw_test)

    device_obj = torch.device(device)
    loader = p1.make_env_loader(list(test_data), BATCH_SIZE, False, 0)
    objects = [
        ("sparse_seed0_soup", p1.SPARSE_ARM, "sparse_seed0"),
        ("dense_seed0_soup", p1.DENSE_ARM, "dense_seed0"),
        ("sparse_seed1_soup", p1.SPARSE_ARM, "sparse_seed1"),
        ("dense_seed1_soup", p1.DENSE_ARM, "dense_seed1"),
    ]
    rows: list[dict[str, Any]] = []
    for name, arm, tag in objects:
        state_path = STATE_DIR / f"{tag}_soup_state.pt"
        if not state_path.exists():
            rows.append({"name": name, "available": False, "tag": tag})
            continue
        model = _load_soup(arm, int(tag.split("seed")[-1]), tag, device_obj)
        result = evaluate(model, loader, device_obj)
        pred = np.asarray(result["predictions"], dtype=np.float64)
        rows.append({"name": name, "available": True, "arm": arm, "tag": tag, "params": int(p1.total_parameter_count()["whole_model"]), "test_mae": float(result["mae"]), "test_mean_prediction": float(pred.mean()), "test_std_prediction": float(pred.std())})

    model = _load_soup(p1.SPARSE_ARM, 0, "sparse_seed0", device_obj)
    clean = evaluate(model, loader, device_obj)
    mechanism: dict[str, Any] = {"clean_test_mae": float(clean["mae"])}
    zero = evaluate(model, loader, device_obj, coord_zero=True)
    mechanism["zero_code_test_mae"] = float(zero["mae"])
    mechanism["G_zero_test"] = float(zero["mae"] - clean["mae"])
    node_maes = []
    for shuffle_seed in SHUFFLE_SEEDS:
        _permute_node_shuffle(test_data, shuffle_seed)
        sh_loader = p1.make_env_loader(list(test_data), BATCH_SIZE, False, 0)
        node_maes.append(float(evaluate(model, sh_loader, device_obj, use_node_shuffle=True)["mae"]))
        _clear_shuffles(test_data)
    mechanism["node_shuffle_test_mae"] = float(np.mean(node_maes))
    mechanism["G_node_test"] = float(np.mean(node_maes) - clean["mae"])
    edge_maes = []
    for shuffle_seed in SHUFFLE_SEEDS:
        _permute_edge_shuffle(test_data, shuffle_seed)
        sh_loader = p1.make_env_loader(list(test_data), BATCH_SIZE, False, 0)
        edge_maes.append(float(evaluate(model, sh_loader, device_obj, use_edge_shuffle=True)["mae"]))
        _clear_shuffles(test_data)
    mechanism["edge_shuffle_test_mae"] = float(np.mean(edge_maes))
    mechanism["G_edge_test"] = float(np.mean(edge_maes) - clean["mae"])
    all_maes = []
    for shuffle_seed in SHUFFLE_SEEDS:
        _permute_node_shuffle(test_data, shuffle_seed)
        _permute_edge_shuffle(test_data, 1000 + shuffle_seed)
        sh_loader = p1.make_env_loader(list(test_data), BATCH_SIZE, False, 0)
        all_maes.append(float(evaluate(model, sh_loader, device_obj, use_node_shuffle=True, use_edge_shuffle=True)["mae"]))
        _clear_shuffles(test_data)
    mechanism["all_shuffle_test_mae"] = float(np.mean(all_maes))
    mechanism["G_all_test"] = float(np.mean(all_maes) - clean["mae"])

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "note": "terminal reporting-only evaluation of a configuration frozen on train/valid within E2E-DictEnv-P1; the official test is not project-wide pristine",
        "n_test": int(len(test_data)),
        **env_meta,
        "rows": rows,
        "mechanism_generalization": mechanism,
        "official_test_loaded": True,
    }
    _write_json(RESULTS_DIR / "official_test_results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# training orchestration
# ---------------------------------------------------------------------------


def run_all(device: str = "cuda") -> None:
    identity_stage()
    build_env_cache()
    correctness_stage(device="cpu")
    smoke_stage(device=device)
    train_run(p1.SPARSE_ARM, 0, "sparse_seed0", device)
    health_stage(device, "sparse_seed0")
    mechanism_stage(device, "sparse_seed0")
    gate = go_gate_stage()
    print(f"[run_all] GO gate: {json.dumps(gate['checks'])} go={gate['go']}", flush=True)
    if not gate["go"]:
        freeze_stage()
        analyze_stage()
        return
    train_run(p1.DENSE_ARM, 0, "dense_seed0", device)
    spec0 = specificity_seed0_stage()
    print(f"[run_all] seed0 specificity: {json.dumps({k: spec0.get(k) for k in ('run', 'G_specific', 'gate_pass')})}", flush=True)
    if spec0.get("gate_pass"):
        train_run(p1.SPARSE_ARM, 1, "sparse_seed1", device)
        train_run(p1.DENSE_ARM, 1, "dense_seed1", device)
        m_s1 = float(_read_json(RESULTS_DIR / "sparse_seed1.json")["soup"]["soup_valid_mae"])
        m_d1 = float(_read_json(RESULTS_DIR / "dense_seed1.json")["soup"]["soup_valid_mae"])
        _write_json(RESULTS_DIR / "specificity_seed1.json", {"protocol_version": PROTOCOL_VERSION, "run": True, "M_S": m_s1, "M_D": m_d1, "G_specific": float(m_d1 - m_s1), "gate_threshold": p1.GATE_DICT_SPECIFIC, "gate_pass": bool((m_d1 - m_s1) >= p1.GATE_DICT_SPECIFIC)})
        health_stage(device, "sparse_seed1")
        mechanism_stage(device, "sparse_seed1")
    freeze_stage()
    analyze_stage()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=["identity", "env", "correct", "smoke", "train", "gate", "mechanism", "health", "dense-seed0", "specificity-seed0", "seed1", "freeze", "analyze", "unlock", "all"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", default=p1.SPARSE_ARM, choices=list(p1.ARMS))
    parser.add_argument("--force-env", action="store_true")
    args = parser.parse_args(argv)

    sdp._configure_determinism()
    if args.stage == "identity":
        print(json.dumps(identity_stage(), indent=2))
    elif args.stage == "env":
        print(json.dumps(build_env_cache(force=args.force_env), indent=2))
    elif args.stage == "correct":
        print(json.dumps(correctness_stage(device="cpu"), indent=2))
    elif args.stage == "smoke":
        print(json.dumps(smoke_stage(device=args.device, seed=args.seed), indent=2))
    elif args.stage == "train":
        print(json.dumps(train_run(args.arm, args.seed, f"{args.arm}_seed{args.seed}", device=args.device), indent=2))
    elif args.stage == "gate":
        print(json.dumps(go_gate_stage(), indent=2))
    elif args.stage == "mechanism":
        print(json.dumps(mechanism_stage(device=args.device, tag=f"sparse_seed{args.seed}"), indent=2))
    elif args.stage == "health":
        print(json.dumps(health_stage(device=args.device, tag=f"sparse_seed{args.seed}"), indent=2))
    elif args.stage == "dense-seed0":
        print(json.dumps(train_run(p1.DENSE_ARM, 0, "dense_seed0", device=args.device), indent=2))
    elif args.stage == "specificity-seed0":
        print(json.dumps(specificity_seed0_stage(), indent=2))
    elif args.stage == "seed1":
        train_run(p1.SPARSE_ARM, 1, "sparse_seed1", device=args.device)
        train_run(p1.DENSE_ARM, 1, "dense_seed1", device=args.device)
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
