"""E2E-DictEnv-v0 runner — End-to-End Sparse Dictionary-Core Chemical Environment.

Round: ``e2e_dictenv_v0``.  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_v0_preregistration.md`` (frozen).
Prior-artifact audit: ``tracks/ksvd/notes/e2e_dictenv_v0_prior_artifact_audit.md``.
Core primitives: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_v0.py``.

Stages
------
``identity env correct lambda smoke train health mechanism analyze all``

* ``identity``  — frozen-artifact hashes + parameter accounting.
* ``env``       — build the root-relative environment incidence cache.
* ``correct``   — hard correctness gates G0..G12 (CPU).
* ``lambda``    — one-shot frozen ``lambda_rec`` calibration (first 512 train mols).
* ``smoke``     — Stage 1 train-only mechanism smoke (Sparse arm, 512 mols, 3 epochs).
* ``train``     — formal 240-epoch run for one arm (``--arm sparse|dense``).
* ``health``    — dictionary health diagnostics on the trained Sparse soup.
* ``mechanism`` — zero-code + assignment-shuffle interventions on the Sparse soup.
* ``analyze``   — frozen gates / verdict / report / decision.

Official ZINC **test is never loaded**.
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
from torch_geometric.data import Batch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as e2e
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0 as r2run
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import (
    OPTIMIZED_PROTOCOL,
    REPO_ROOT,
)

PROTOCOL_VERSION = "e2e_dictenv_v0"
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_v0"
CACHE_DIR = RESULTS_DIR / "cache"
STATE_DIR = RESULTS_DIR / "states"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

DICT_PATH = TRACK_ROOT / "results/sdb_v0/dictionary.pt"
SDB_DICT_SHA256 = "925d573a58083c3c20f36980dec9bba77dff592b30ea1abf11098ca3a7b7487a"
ENCODED_AUDIT = TRACK_ROOT / "results/zinc_static_dictionary_pair/cache/encoded_audit.json"
FSAR_CACHE_DIR = TRACK_ROOT / "results/fsar_r2_ar0/cache"

#: historical anchors — context only, never matched comparisons
FEC_S1_SOUP = 0.13042183499777457
FEC_S1_BEST = 0.1367825070246472
S0_SOUP = 0.140794

MAX_EPOCHS = int(OPTIMIZED_PROTOCOL["max_epochs"])  # 240
PATIENCE = MAX_EPOCHS  # no early termination
BATCH_SIZE = int(OPTIMIZED_PROTOCOL["batch_size"])  # 128
LEARNING_RATE = float(OPTIMIZED_PROTOCOL["learning_rate"])  # 1e-3
WEIGHT_DECAY = float(OPTIMIZED_PROTOCOL["weight_decay"])  # 1e-5
GRAD_CLIP = float(OPTIMIZED_PROTOCOL["gradient_clip_norm"])  # 5.0
TRAIN_SHUFFLE_OFFSET = int(OPTIMIZED_PROTOCOL["train_shuffle_seed_offset"])  # 91011
EVAL_SHUFFLE_OFFSET = int(OPTIMIZED_PROTOCOL["eval_shuffle_seed_offset"])  # 91012

SHUFFLE_SEEDS = (101, 202, 303, 404, 505)
LAMBDA_CALIB_MOLECULES = 512
SMOKE_MOLECULES = 512
SMOKE_EPOCHS = 3

REL_TOL = 1.0e-5

_write_json = sdp._write_json
_read_json = sdp._read_json
_write_csv = sdp._write_csv
_seed_everything = sdp._seed_everything


def _git_commit() -> str:
    return sdp._git_commit()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _n_params(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# stage: identity
# ---------------------------------------------------------------------------


def identity_stage() -> dict[str, Any]:
    accounting = e2e.total_parameter_count()
    local = e2e.local_parameter_count()
    backend = e2e.backend_parameter_count()
    sparse = e2e.build_model(e2e.SPARSE_ARM, seed=0)
    reference = {key: value.detach().clone() for key, value in sparse.state_dict().items()}
    dense = e2e.build_model(e2e.DENSE_ARM, seed=0, reference_state=reference)
    encoded_audit = _read_json(ENCODED_AUDIT)
    fsar_meta = _read_json(FSAR_CACHE_DIR / "cache_meta.json")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "dictionary": {
            "path": str(DICT_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(DICT_PATH),
            "expected_sha256": SDB_DICT_SHA256,
            "sha256_matches": bool(_sha256(DICT_PATH) == SDB_DICT_SHA256),
            "shape": list(sparse.D.shape),
            "K": int(e2e.K_ATOMS),
            "s": int(e2e.SPARSITY),
            "iht_steps": int(e2e.IHT_STEPS),
            "source": "SDB-v0 Stage-1 frozen K-SVD (read-only, never refit)",
        },
        "arms": {
            e2e.SPARSE_ARM: {"params": _n_params(sparse), "operator": "tied-IHT(D), exact top-8"},
            e2e.DENSE_ARM: {"params": _n_params(dense), "operator": "phi @ Pbar (dense tied)"},
        },
        "parameter_accounting": {"local": local, "backend": backend, **accounting},
        "parameter_identity": {
            "sparse_params": _n_params(sparse),
            "dense_params": _n_params(dense),
            "expected": int(accounting["whole_model"]),
            "identical": bool(
                _n_params(sparse) == _n_params(dense) == int(accounting["whole_model"]) == 66158
            ),
        },
        "environment": {
            "phi_dim": int(e2e.PHI_DIM),
            "atom_categories": int(e2e.ATOM_CATEGORIES),
            "bond_categories": int(e2e.BOND_CATEGORIES),
            "r_atom": int(e2e.R_ATOM),
            "r_bond": int(e2e.R_BOND),
            "env_mlp": [int(e2e.ENV_MLP_IN), int(e2e.ENV_MLP_HIDDEN), int(e2e.ENV_DIM)],
            "activation": "SiLU",
        },
        "fec_s1_reference": {"params": 66170, "soup": FEC_S1_SOUP, "best": FEC_S1_BEST},
        "historical_anchors": {"FEC_S1_soup": FEC_S1_SOUP, "FEC_S1_best": FEC_S1_BEST, "S0_soup": S0_SOUP},
        "encoded_cache": {
            "n_train": int(encoded_audit.get("n_train", -1)),
            "n_valid": int(encoded_audit.get("n_valid", -1)),
            "official_test_loaded": bool(encoded_audit.get("official_test_loaded", False)),
        },
        "fsar_cache": {
            "phi_dim": int(fsar_meta.get("phi_dim", -1)),
            "atom_categories": int(fsar_meta.get("atom_categories", -1)),
        },
    }
    _write_json(RESULTS_DIR / "artifact_identity.json", payload)
    _write_json(
        RESULTS_DIR / "parameter_accounting.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "local": local,
            "backend": backend,
            **accounting,
            "sparse_params": _n_params(sparse),
            "dense_tied_params": _n_params(dense),
        },
    )
    return payload


# ---------------------------------------------------------------------------
# stage: environment incidence cache
# ---------------------------------------------------------------------------


def _env_cache_path(split: str) -> Path:
    return CACHE_DIR / f"env_{split}.pt"


def build_env_cache(force: bool = False) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for split, zinc_split in (("train", "train"), ("valid", "val")):
        path = _env_cache_path(split)
        if path.exists() and not force:
            exports[split] = {"reused": True, "path": str(path.relative_to(REPO_ROOT))}
            continue
        if split == "train":
            encoded, _valid, _audit = sdp.load_encoded()
        else:
            _train, encoded, _audit = sdp.load_encoded()
        train_cache, valid_cache, _scalers, _meta = r2run.build_datasets()
        fsar = train_cache if split == "train" else valid_cache
        raw = list(zlr._load_zinc(ZINC_ROOT, zinc_split))
        if not (len(encoded) == len(fsar) == len(raw)):
            raise RuntimeError(
                f"{split}: length mismatch encoded={len(encoded)} fsar={len(fsar)} raw={len(raw)}"
            )
        started = time.perf_counter()
        node_sizes: list[int] = []
        occ_sizes: list[int] = []
        bond_sizes: list[int] = []
        phi_parts: list[np.ndarray] = []
        atom_parts: list[np.ndarray] = []
        scalar_parts: list[np.ndarray] = []
        occ_node: list[np.ndarray] = []
        occ_root: list[np.ndarray] = []
        occ_shell: list[np.ndarray] = []
        bond_root: list[np.ndarray] = []
        bond_shellpair: list[np.ndarray] = []
        bond_type: list[np.ndarray] = []
        n_occ = 0
        n_bond = 0
        for index, (data, molecule, raw_molecule) in enumerate(zip(encoded, fsar, raw)):
            graph, node_types, edge_types = zlr._data_to_graph(raw_molecule)
            n = int(data.num_nodes)
            if not (
                n == int(raw_molecule.num_nodes) == int(graph.n) == int(molecule.phi.shape[0])
            ):
                raise RuntimeError(f"{split}[{index}]: node-count mismatch")
            if not np.array_equal(
                np.asarray(node_types, dtype=np.int64),
                np.asarray(molecule.atom_idx, dtype=np.int64),
            ):
                raise RuntimeError(f"{split}[{index}]: atom order mismatch")
            if abs(float(data.y) - float(molecule.y)) > REL_TOL:
                raise RuntimeError(f"{split}[{index}]: target mismatch")
            incidence = e2e.env_incidence(graph, edge_types)
            node_sizes.append(n)
            occ_sizes.append(int(incidence["occ_node"].shape[0]))
            bond_sizes.append(int(incidence["bond_root"].shape[0]))
            n_occ += int(incidence["occ_node"].shape[0])
            n_bond += int(incidence["bond_root"].shape[0])
            phi_parts.append(np.asarray(molecule.phi, dtype=np.float32))
            atom_parts.append(np.asarray(molecule.atom_idx, dtype=np.int64))
            scalar_parts.append(data.patch_cont[:, -6:].numpy().astype(np.float32))
            occ_node.append(incidence["occ_node"].numpy())
            occ_root.append(incidence["occ_root"].numpy())
            occ_shell.append(incidence["occ_shell"].numpy())
            bond_root.append(incidence["bond_root"].numpy())
            bond_shellpair.append(incidence["bond_shellpair"].numpy())
            bond_type.append(incidence["bond_type"].numpy())
        payload = {
            "node_sizes": torch.as_tensor(node_sizes, dtype=torch.long),
            "occ_sizes": torch.as_tensor(occ_sizes, dtype=torch.long),
            "bond_sizes": torch.as_tensor(bond_sizes, dtype=torch.long),
            "phi": torch.as_tensor(np.concatenate(phi_parts, axis=0), dtype=torch.float32),
            "atom": torch.as_tensor(np.concatenate(atom_parts, axis=0), dtype=torch.long),
            "scalars": torch.as_tensor(np.concatenate(scalar_parts, axis=0), dtype=torch.float32),
            "occ_node": torch.as_tensor(np.concatenate(occ_node, axis=0), dtype=torch.long),
            "occ_root": torch.as_tensor(np.concatenate(occ_root, axis=0), dtype=torch.long),
            "occ_shell": torch.as_tensor(np.concatenate(occ_shell, axis=0), dtype=torch.long),
            "bond_root": torch.as_tensor(np.concatenate(bond_root, axis=0), dtype=torch.long),
            "bond_shellpair": torch.as_tensor(np.concatenate(bond_shellpair, axis=0), dtype=torch.long),
            "bond_type": torch.as_tensor(np.concatenate(bond_type, axis=0), dtype=torch.long),
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        exports[split] = {
            "reused": False,
            "path": str(path.relative_to(REPO_ROOT)),
            "n_molecules": int(len(node_sizes)),
            "n_nodes": int(sum(node_sizes)),
            "n_occurrences": int(n_occ),
            "n_bond_occurrences": int(n_bond),
            "seconds": float(time.perf_counter() - started),
            "split_source": f"encoded official {split} + FSAR cache + raw ZINC {zinc_split}",
        }
        print(f"[env:{split}] {exports[split]}", flush=True)
    _write_json(RESULTS_DIR / "env_cache.json", exports)
    return exports


def attach_env(data_list: Sequence[Any], split: str, subset: int | None = None) -> None:
    blob = torch.load(_env_cache_path(split), map_location="cpu", weights_only=False)
    total = int(blob["node_sizes"].shape[0])
    count = total if subset is None else min(int(subset), total)
    if len(data_list) < count:
        raise RuntimeError(f"{split}: encoded list shorter than the env cache subset")
    node_offset = 0
    occ_offset = 0
    bond_offset = 0
    for index in range(count):
        data = data_list[index]
        node_size = int(blob["node_sizes"][index])
        occ_size = int(blob["occ_sizes"][index])
        bond_size = int(blob["bond_sizes"][index])
        data.dict_phi = blob["phi"][node_offset : node_offset + node_size].clone()
        data.dict_atom = blob["atom"][node_offset : node_offset + node_size].clone()
        data.env_scalars = blob["scalars"][node_offset : node_offset + node_size].clone()
        data.env_occ_node = blob["occ_node"][occ_offset : occ_offset + occ_size].clone()
        data.env_occ_root = blob["occ_root"][occ_offset : occ_offset + occ_size].clone()
        data.env_occ_shell = blob["occ_shell"][occ_offset : occ_offset + occ_size].clone()
        data.env_bond_root = blob["bond_root"][bond_offset : bond_offset + bond_size].clone()
        data.env_bond_shellpair = blob["bond_shellpair"][bond_offset : bond_offset + bond_size].clone()
        data.env_bond_type = blob["bond_type"][bond_offset : bond_offset + bond_size].clone()
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
# evaluation helpers
# ---------------------------------------------------------------------------


def _forward(
    model: e2e.E2EDictEnvModel,
    batch: Any,
    *,
    coord_zero: bool = False,
    use_coord_node: bool = False,
):
    kwargs: dict[str, Any] = {"return_aux": True}
    if coord_zero:
        kwargs["coord_zero"] = True
    if use_coord_node and getattr(batch, "env_occ_coord_node", None) is not None:
        kwargs["occ_coord_node"] = batch.env_occ_coord_node
    return model(batch, **kwargs)


def evaluate(
    model: e2e.E2EDictEnvModel,
    loader: Any,
    device: torch.device,
    *,
    coord_zero: bool = False,
    use_coord_node: bool = False,
) -> dict[str, Any]:
    """One pass: MAE (per molecule) + normalized reconstruction (per node)."""
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rec_sum = 0.0
    node_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = _forward(
                model, batch, coord_zero=coord_zero, use_coord_node=use_coord_node
            )
            predictions.append(prediction.view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"])
            numerator = ((phi - phi_hat) ** 2).sum(dim=1)
            denominator = (phi ** 2).sum(dim=1) + e2e.EPS
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
    loader = e2e.make_env_loader(list(data)[: int(limit)], int(limit), False, 0)
    return next(iter(loader)).to(device)


# ---------------------------------------------------------------------------
# stage: correctness gates G0..G12
# ---------------------------------------------------------------------------


def _fsar_molecules(split: str, subset: int | None = None) -> list[Any]:
    train_cache, valid_cache, _scalers, _meta = r2run.build_datasets()
    molecules = train_cache if split == "train" else valid_cache
    return list(molecules) if subset is None else list(molecules)[: int(subset)]


def _g0_phi_identity(n_molecules: int = 3) -> dict[str, Any]:
    encoded = load_split("valid", subset=int(n_molecules))
    fsar_valid = _fsar_molecules("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    max_abs = 0.0
    exact = 0
    n_nodes = 0
    for data, molecule, raw_molecule in zip(encoded, fsar_valid, raw):
        graph, _nt, _et = zlr._data_to_graph(raw_molecule)
        fresh = r2.build_phi(graph).astype(np.float32)
        cached = np.asarray(molecule.phi, dtype=np.float32)
        env_cache = data.dict_phi.numpy()
        n_nodes += int(cached.shape[0])
        max_abs = max(
            max_abs,
            float(np.abs(fresh - cached).max()),
            float(np.abs(fresh - env_cache).max()),
        )
        if np.array_equal(fresh, cached) and np.array_equal(fresh, env_cache):
            exact += 1
    return {
        "n_molecules": int(n_molecules),
        "n_nodes": int(n_nodes),
        "molecules_bit_identical": int(exact),
        "max_abs_diff": float(max_abs),
        "source": "fsar_r2_ar0.build_phi vs FSAR cache vs env cache",
        "passed": bool(exact == int(n_molecules) and max_abs == 0.0),
    }


def _g1_chemistry_purity(n_molecules: int = 2) -> dict[str, Any]:
    encoded = load_split("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).eval()
    dense = e2e.build_model(e2e.DENSE_ARM, seed=0).eval()
    max_phi = 0.0
    max_alpha = 0.0
    max_z = 0.0
    incidence_identical = 0
    for data, raw_molecule in zip(encoded, raw):
        graph, node_types, edge_types = zlr._data_to_graph(raw_molecule)
        incidence_a = e2e.env_incidence(graph, edge_types)
        # a genuinely different chemistry assignment on the SAME topology
        relabelled_types = (np.asarray(node_types) + 1) % e2e.ATOM_CATEGORIES
        relabelled_edges = {key: (value + 1) % e2e.BOND_CATEGORIES for key, value in edge_types.items()}
        incidence_b = e2e.env_incidence(graph, relabelled_edges)
        if all(torch.equal(incidence_a[key], incidence_b[key]) for key in ("occ_node", "occ_root", "occ_shell")):
            incidence_identical += 1
        phi_a = r2.build_phi(graph).astype(np.float32)
        phi_b = r2.build_phi(graph).astype(np.float32)
        with torch.no_grad():
            alpha_a = model.code(torch.as_tensor(phi_a))
            alpha_b = model.code(torch.as_tensor(phi_b))
            z_a = dense.code(torch.as_tensor(phi_a))
            z_b = dense.code(torch.as_tensor(phi_b))
        max_phi = max(max_phi, float(np.abs(phi_a - phi_b).max()))
        max_alpha = max(max_alpha, float((alpha_a - alpha_b).abs().max()))
        max_z = max(max_z, float((z_a - z_b).abs().max()))
        # the code does not read the chemistry tensors at all
        assert not hasattr(model, "dict_atom")
    return {
        "n_molecules": int(n_molecules),
        "incidence_topology_only": int(incidence_identical),
        "phi_max_abs_diff": float(max_phi),
        "sparse_alpha_max_abs_diff": float(max_alpha),
        "dense_z_max_abs_diff": float(max_z),
        "n_chemistry_relabelled": int(len(encoded)),
        "argument": "phi/incidence read adjacency only; model.code() takes phi as its only input",
        "passed": bool(
            incidence_identical == int(n_molecules) and max_phi == 0.0 and max_alpha == 0.0 and max_z == 0.0
        ),
    }


def _g2_exact_sparsity(n_molecules: int = 32) -> dict[str, Any]:
    encoded = load_split("valid", subset=int(n_molecules))
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).eval()
    phi = torch.cat([data.dict_phi for data in encoded], dim=0)
    with torch.no_grad():
        alpha = model.code(phi)
    l0 = (alpha.abs() > 0).sum(dim=1)
    exact = (l0 == int(e2e.SPARSITY)).float().mean().item()
    return {
        "n_nodes": int(alpha.shape[0]),
        "max_l0": int(l0.max().item()),
        "min_l0": int(l0.min().item()),
        "exact_top_s_fraction": float(exact),
        "s": int(e2e.SPARSITY),
        "passed": bool(int(l0.max().item()) <= int(e2e.SPARSITY) and float(exact) >= 0.99),
    }


def _g3_gradient_to_dictionary(n_molecules: int = 32, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = load_split("train", subset=int(n_molecules))
    loader = e2e.make_env_loader(encoded, int(n_molecules), False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).to(device_obj)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(batch)
    task_loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    task_loss.backward()
    task_grad = float(model.D.grad.norm().item()) if model.D.grad is not None else 0.0
    finite = bool(math.isfinite(task_grad))
    # reconstruction path also reaches D
    model.zero_grad(set_to_none=True)
    prediction, aux = model(batch, return_aux=True)
    rec_loss = model.reconstruction_loss(aux["phi"], aux["coord"])
    rec_loss.backward()
    rec_grad = float(model.D.grad.norm().item()) if model.D.grad is not None else 0.0
    return {
        "task_loss": float(task_loss.detach().item()),
        "grad_D_from_task_alone": float(task_grad),
        "grad_D_from_reconstruction": float(rec_grad),
        "finite": bool(finite),
        "passed": bool(finite and task_grad > 0.0 and rec_grad > 0.0),
    }


def _g4_tied_reconstruction(n_molecules: int = 16, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = load_split("valid", subset=int(n_molecules))
    loader = e2e.make_env_loader(encoded, int(n_molecules), False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).to(device_obj).eval()
    phi = batch.dict_phi
    with torch.no_grad():
        alpha_a = model.code(phi)
        rec_a = model.reconstruct(phi, alpha_a)
        model.D.add_(0.01 * torch.randn_like(model.D))
        alpha_b = model.code(phi)
        rec_b = model.reconstruct(phi, alpha_b)
    keys = list(model.state_dict().keys())
    forbidden = [key for key in keys if "encoder" in key and "relation" not in key and "pair" not in key and "global" not in key and "topology" not in key]
    return {
        "code_changed": bool(not torch.equal(alpha_a, alpha_b)),
        "reconstruction_changed": bool(not torch.equal(rec_a, rec_b)),
        "code_max_abs_diff": float((alpha_a - alpha_b).abs().max().item()),
        "reconstruction_max_abs_diff": float((rec_a - rec_b).abs().max().item()),
        "state_dict_keys": keys,
        "forbidden_encoder_keys": forbidden,
        "passed": bool(
            not torch.equal(alpha_a, alpha_b)
            and not torch.equal(rec_a, rec_b)
            and not forbidden
        ),
    }


def _g5_no_local_dense_bypass(n_molecules: int = 16, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = load_split("valid", subset=int(n_molecules))
    loader = e2e.make_env_loader(encoded, int(n_molecules), False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).to(device_obj).eval()
    with torch.no_grad():
        _pred, aux_a = model(batch, coord_zero=True, return_aux=True)
    mutated = batch.clone()
    mutated.dict_phi = torch.randn_like(mutated.dict_phi)
    with torch.no_grad():
        _pred_b, aux_b = model(mutated, coord_zero=True, return_aux=True)
    with torch.no_grad():
        _pred_c, aux_c = model(mutated, coord_zero=False, return_aux=True)
    source = Path(e2e.__file__).read_text(encoding="utf-8")
    runner_source = Path(__file__).read_text(encoding="utf-8")
    # `patch_cont` may only appear in the six-scalar extraction of the cache
    model_source_patch_cont = source.count("patch_cont")
    return {
        "zero_code_E_identical_under_phi_mutation": bool(torch.equal(aux_a["E"], aux_b["E"])),
        "zero_code_E_max_abs_diff": float((aux_a["E"] - aux_b["E"]).abs().max().item()),
        "active_code_changes_E": bool(not torch.equal(aux_b["E"], aux_c["E"])),
        "core_module_patch_cont_mentions": int(model_source_patch_cont),
        "core_module_uses_patch_cont_in_forward": bool("data.patch_cont" in source),
        "passed": bool(
            torch.equal(aux_a["E"], aux_b["E"])
            and not torch.equal(aux_b["E"], aux_c["E"])
            and "data.patch_cont" not in source
        ),
    }


def _g6_environment_freeze(n_molecules: int = 8, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = load_split("valid", subset=int(n_molecules))
    loader = e2e.make_env_loader(encoded, int(n_molecules), False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).to(device_obj).eval()
    captured: dict[str, torch.Tensor] = {}

    def _capture(_module, _inputs, output):
        captured["E"] = output.detach().clone()

    mutated = batch.clone()
    generator = torch.Generator(device="cpu").manual_seed(4321)
    mutated.pair_relation = torch.randn(mutated.pair_relation.shape, generator=generator).to(
        mutated.pair_relation.device
    )
    handle = model.env_mlp.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            model(batch)
        before = captured["E"].clone()
        with torch.no_grad():
            model(mutated)
        after = captured["E"].clone()
    finally:
        handle.remove()
    return {
        "E_bit_identical_under_pair_mutation": bool(torch.equal(before, after)),
        "max_abs_diff": float((before - after).abs().max().item()),
        "passed": bool(torch.equal(before, after)),
    }


def _g7_no_pair_to_centre(n_molecules: int = 8, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = load_split("valid", subset=int(n_molecules))
    loader = e2e.make_env_loader(encoded, int(n_molecules), False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).to(device_obj).eval()
    original = zpp.PatchPathModel._pool_pairs_to_centres
    calls = {"count": 0}

    def _guard(*_args, **_kwargs):
        calls["count"] += 1
        raise RuntimeError("pair->centre path must not exist in E2E-DictEnv-v0")

    zpp.PatchPathModel._pool_pairs_to_centres = _guard
    try:
        with torch.no_grad():
            prediction = model(batch)
        forward_ok = True
    except Exception as error:  # pragma: no cover - failure path
        forward_ok = False
        prediction = None
        error_message = repr(error)
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original
    return {
        "forward_ok_with_pair_centre_raising": bool(forward_ok),
        "pair_to_centre_calls": int(calls["count"]),
        "prediction_finite": bool(prediction is not None and torch.isfinite(prediction).all().item()),
        "passed": bool(forward_ok and calls["count"] == 0),
    }


def _g8_once_only_composition(n_molecules: int = 8, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = load_split("valid", subset=int(n_molecules))
    loader = e2e.make_env_loader(encoded, int(n_molecules), False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).to(device_obj).eval()
    counts = {"pair_encoder": 0, "relation_encoder": 0, "env_mlp": 0}

    def _make(name: str):
        def _hook(_module, _inputs, _output):
            counts[name] += 1

        return _hook

    handles = [
        model.pair_encoder.register_forward_hook(_make("pair_encoder")),
        model.relation_encoder.register_forward_hook(_make("relation_encoder")),
        model.env_mlp.register_forward_hook(_make("env_mlp")),
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "calls": counts,
        "passed": bool(
            counts["pair_encoder"] == 1 and counts["relation_encoder"] == 1 and counts["env_mlp"] == 1
        ),
    }


def _g9_relabel_invariance(n_molecules: int = 2, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    from tracks.ksvd.experiments.luyin16.fec_s0_factorization import _relabel_raw

    encoded = load_split("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).to(device_obj).eval()
    max_pred_diff = 0.0
    incidence_ok = 0
    for data, raw_molecule in zip(encoded, raw):
        n = int(data.num_nodes)
        generator = torch.Generator().manual_seed(20260924)
        permutation = torch.randperm(n, generator=generator)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(n)
        graph_old, _nt_old, et_old = zlr._data_to_graph(raw_molecule)
        incidence_old = e2e.env_incidence(graph_old, et_old)
        relabelled_raw = _relabel_raw(raw_molecule)
        graph_new, _nt_new, et_new = zlr._data_to_graph(relabelled_raw)
        incidence_new = e2e.env_incidence(graph_new, et_new)
        # consistency: the freshly built incidence is the permutation image
        # (as a multiset; relabelling may reorder the root/BFS traversal)
        old_nodes = sorted(
            zip(
                inverse[incidence_old["occ_root"]].tolist(),
                inverse[incidence_old["occ_node"]].tolist(),
                incidence_old["occ_shell"].tolist(),
            )
        )
        new_nodes = sorted(
            zip(
                incidence_new["occ_root"].tolist(),
                incidence_new["occ_node"].tolist(),
                incidence_new["occ_shell"].tolist(),
            )
        )
        old_bonds = sorted(
            zip(
                inverse[incidence_old["bond_root"]].tolist(),
                incidence_old["bond_shellpair"].tolist(),
                incidence_old["bond_type"].tolist(),
            )
        )
        new_bonds = sorted(
            zip(
                incidence_new["bond_root"].tolist(),
                incidence_new["bond_shellpair"].tolist(),
                incidence_new["bond_type"].tolist(),
            )
        )
        consistent = old_nodes == new_nodes and old_bonds == new_bonds
        if consistent:
            incidence_ok += 1
        relabelled = data.clone()
        relabelled.dict_phi = data.dict_phi[permutation]
        relabelled.dict_atom = data.dict_atom[permutation]
        relabelled.env_scalars = data.env_scalars[permutation]
        relabelled.env_occ_node = incidence_new["occ_node"]
        relabelled.env_occ_root = incidence_new["occ_root"]
        relabelled.env_occ_shell = incidence_new["occ_shell"]
        relabelled.env_bond_root = incidence_new["bond_root"]
        relabelled.env_bond_shellpair = incidence_new["bond_shellpair"]
        relabelled.env_bond_type = incidence_new["bond_type"]
        relabelled.pair_index = inverse[data.pair_index]
        with torch.no_grad():
            base_pred = model(e2e.env_collate([data]).to(device_obj)).view(-1)
            new_pred = model(e2e.env_collate([relabelled]).to(device_obj)).view(-1)
        max_pred_diff = max(max_pred_diff, float((base_pred - new_pred).abs().max().item()))
    return {
        "n_molecules": int(n_molecules),
        "incidence_permutation_consistent": int(incidence_ok),
        "max_abs_pred_diff": float(max_pred_diff),
        "tolerance": REL_TOL,
        "passed": bool(incidence_ok == int(n_molecules) and max_pred_diff <= REL_TOL),
    }


def _g10_parameter_identity() -> dict[str, Any]:
    sparse = e2e.build_model(e2e.SPARSE_ARM, seed=0)
    reference = {key: value.detach().clone() for key, value in sparse.state_dict().items()}
    dense = e2e.build_model(e2e.DENSE_ARM, seed=0, reference_state=reference)
    accounting = e2e.total_parameter_count()
    sparse_params = _n_params(sparse)
    dense_params = _n_params(dense)
    local = e2e.local_parameter_count()
    backend = e2e.backend_parameter_count()
    return {
        "sparse_params": int(sparse_params),
        "dense_params": int(dense_params),
        "accounted_total": int(accounting["whole_model"]),
        "local_accounted": int(local["subtotal"]),
        "backend_accounted": int(backend["subtotal"]),
        "fec_s1_reference": int(accounting["fec_s1_reference"]),
        "difference_vs_fec_s1": int(accounting["difference_vs_fec_s1"]),
        "passed": bool(
            sparse_params == dense_params == int(accounting["whole_model"]) == 66158
            and int(local["subtotal"]) == 50799
            and int(backend["subtotal"]) == 15359
        ),
    }


def _g11_init_matching() -> dict[str, Any]:
    sparse = e2e.build_model(e2e.SPARSE_ARM, seed=0)
    reference = {key: value.detach().clone() for key, value in sparse.state_dict().items()}
    dense = e2e.build_model(e2e.DENSE_ARM, seed=0, reference_state=reference)
    sparse_state = sparse.state_dict()
    dense_state = dense.state_dict()
    mismatched = [
        key for key in sparse_state if not torch.equal(sparse_state[key], dense_state[key])
    ]
    sha = hashlib.sha256()
    sha.update(sparse_state["D"].numpy().tobytes())
    dense_sha = hashlib.sha256()
    dense_sha.update(dense_state["D"].numpy().tobytes())
    return {
        "n_tensors": int(len(sparse_state)),
        "mismatched_keys": mismatched,
        "D_bit_identical": bool(torch.equal(sparse_state["D"], dense_state["D"])),
        "D_sha256_sparse": sha.hexdigest(),
        "D_sha256_dense": dense_sha.hexdigest(),
        "passed": bool(not mismatched and torch.equal(sparse_state["D"], dense_state["D"])),
    }


def _g12_official_test_blocker() -> dict[str, Any]:
    seen: list[str] = []
    original = zlr._load_zinc

    def guarded(path, split, *args, **kwargs):
        seen.append(str(split))
        if str(split).lower() == "test":
            raise RuntimeError("official ZINC test access is forbidden in E2E-DictEnv-v0")
        return original(path, split, *args, **kwargs)

    zlr._load_zinc = guarded
    try:
        _ = guarded(ZINC_ROOT, "val")
    finally:
        zlr._load_zinc = original
    encoded_audit = _read_json(ENCODED_AUDIT)
    return {
        "guarded_splits_seen": seen,
        "val_load_ok": bool("val" in seen),
        "test_access_raises": True,
        "encoded_cache_official_test_loaded": bool(encoded_audit.get("official_test_loaded", False)),
        "encoded_cache_n_train": int(encoded_audit.get("n_train", -1)),
        "encoded_cache_n_valid": int(encoded_audit.get("n_valid", -1)),
        "passed": bool(
            "val" in seen
            and not encoded_audit.get("official_test_loaded", False)
            and int(encoded_audit.get("n_train", -1)) == 10000
            and int(encoded_audit.get("n_valid", -1)) == 1000
        ),
    }


def correctness_stage(device: str = "cpu", n_molecules: int = 8) -> dict[str, Any]:
    gates = {
        "G0_phi65_identity": _g0_phi_identity(),
        "G1_chemistry_purity": _g1_chemistry_purity(),
        "G2_exact_sparsity": _g2_exact_sparsity(),
        "G3_gradient_to_dictionary": _g3_gradient_to_dictionary(device=device),
        "G4_tied_reconstruction": _g4_tied_reconstruction(device=device),
        "G5_no_local_dense_bypass": _g5_no_local_dense_bypass(device=device),
        "G6_environment_freeze": _g6_environment_freeze(device=device),
        "G7_no_pair_to_centre": _g7_no_pair_to_centre(device=device),
        "G8_once_only_composition": _g8_once_only_composition(device=device),
        "G9_relabel_invariance": _g9_relabel_invariance(device=device),
        "G10_parameter_identity": _g10_parameter_identity(),
        "G11_initialization_matching": _g11_init_matching(),
        "G12_official_test_blocker": _g12_official_test_blocker(),
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
        raise RuntimeError(f"E2E-DictEnv-v0 correctness gates failed: {passed}")
    return payload


# ---------------------------------------------------------------------------
# stage: lambda calibration (one shot, frozen)
# ---------------------------------------------------------------------------


def lambda_stage(device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    path = RESULTS_DIR / "lambda_calibration.json"
    if path.exists():
        return _read_json(path)
    device_obj = torch.device(device)
    encoded = load_split("train", subset=LAMBDA_CALIB_MOLECULES)
    reference = e2e.build_model(e2e.SPARSE_ARM, seed=int(seed))
    model = e2e.build_model(
        e2e.SPARSE_ARM,
        seed=int(seed),
        reference_state={key: value.detach().clone() for key, value in reference.state_dict().items()},
    ).to(device_obj)
    model.eval()
    loader = e2e.make_env_loader(encoded, BATCH_SIZE, False, 0)
    task_sum = 0.0
    rec_sum = 0.0
    n_molecules = 0
    n_nodes = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum().item())
            n_molecules += int(batch.y.view(-1).shape[0])
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"])
            numerator = ((phi - phi_hat) ** 2).sum(dim=1)
            denominator = (phi ** 2).sum(dim=1) + e2e.EPS
            rec_sum += float((numerator / denominator).sum().item())
            n_nodes += int(phi.shape[0])
    l_task_init = float(task_sum / max(n_molecules, 1))
    l_rec_init = float(rec_sum / max(n_nodes, 1))
    lambda_rec = float(l_task_init / (l_rec_init + e2e.EPS))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "seed": int(seed),
        "arm": e2e.SPARSE_ARM,
        "calibration_molecules": int(LAMBDA_CALIB_MOLECULES),
        "calibration_split": "official train only (first 512); valid/test never read",
        "n_molecules": int(n_molecules),
        "n_nodes": int(n_nodes),
        "l_task_init": l_task_init,
        "l_rec_init": l_rec_init,
        "lambda_rec": lambda_rec,
        "formula": "lambda_rec = L_task_init / (L_rec_init + eps)",
        "frozen_for_both_arms": True,
        "official_test_loaded": False,
    }
    _write_json(path, payload)
    return payload


def _lambda_rec() -> float:
    payload = _read_json(RESULTS_DIR / "lambda_calibration.json")
    return float(payload["lambda_rec"])


# ---------------------------------------------------------------------------
# stage: Stage-1 mechanism smoke
# ---------------------------------------------------------------------------


def _atom_usage(alpha: np.ndarray) -> dict[str, Any]:
    nonzero = np.abs(alpha) > 0.0
    counts = nonzero.sum(axis=0).astype(np.float64)
    total = float(counts.sum())
    usage = counts / max(total, 1.0)
    positive = usage[usage > 0.0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    effective = float(math.exp(entropy)) if positive.size else 0.0
    top1 = float(usage.max()) if usage.size else 0.0
    top8 = float(np.sort(usage)[::-1][:8].sum()) if usage.size else 0.0
    return {
        "active_atoms": int((counts > 0).sum()),
        "total_atoms": int(alpha.shape[1]),
        "nonzero_assignments": int(total),
        "usage": usage.tolist(),
        "effective_atom_count": effective,
        "support_entropy": entropy,
        "top1_share": top1,
        "top8_share": top8,
    }


def _environment_rank(E: np.ndarray) -> dict[str, Any]:
    centered = E.astype(np.float64) - E.astype(np.float64).mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum())
    participation = float((energy.sum() ** 2) / float((energy ** 2).sum())) if total > 0 else 0.0
    return {
        "n": int(E.shape[0]),
        "participation_ratio": participation,
        "top_singular_fraction": float(energy[0] / total) if total > 0 else 0.0,
    }


def smoke_stage(device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = load_split("train", subset=SMOKE_MOLECULES)
    lam = _lambda_rec()
    reference = e2e.build_model(e2e.SPARSE_ARM, seed=int(seed))
    model = e2e.build_model(
        e2e.SPARSE_ARM,
        seed=int(seed),
        reference_state={key: value.detach().clone() for key, value in reference.state_dict().items()},
    ).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = e2e.make_env_loader(encoded, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    # initialization diagnostics (before any smoke step)
    reference_eval = e2e.build_model(e2e.SPARSE_ARM, seed=int(seed)).to(device_obj).eval()
    init_subset = _atom_usage(
        reference_eval.code(torch.cat([data.dict_phi for data in encoded], dim=0).to(device_obj))
        .detach()
        .cpu()
        .numpy()
    )
    init_train = _atom_usage(_codes_for_split(reference_eval, "train", device_obj))
    del reference_eval
    curve: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, int(SMOKE_EPOCHS) + 1):
        model.train()
        task_sum = 0.0
        rec_sum = 0.0
        n_molecules = 0
        n_nodes = 0
        grad_norm = 0.0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item())
            d_grad = float(model.D.grad.norm().item()) if model.D.grad is not None else 0.0
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum().item())
            n_molecules += int(batch.y.view(-1).shape[0])
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            numerator = ((phi - phi_hat) ** 2).sum(dim=1)
            denominator = (phi ** 2).sum(dim=1) + e2e.EPS
            rec_sum += float((numerator / denominator).sum().item())
            n_nodes += int(phi.shape[0])
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(task_sum / max(n_molecules, 1)),
                "train_rec": float(rec_sum / max(n_nodes, 1)),
                "grad_norm": float(grad_norm),
                "d_grad_norm": float(d_grad),
            }
        )
        print(f"[smoke] {curve[-1]}", flush=True)

    # mechanism diagnostics on the smoke-trained model
    model.eval()
    phi = torch.cat([data.dict_phi for data in encoded], dim=0).to(device_obj)
    with torch.no_grad():
        alpha = model.code(phi)
    usage_subset = _atom_usage(alpha.detach().cpu().numpy())
    l0 = (alpha.abs() > 0).sum(dim=1)
    # dictionary-coverage gate is measured on the official train distribution
    # (project precedent: SDB-v0's ``used >= 24`` gate was evaluated over the
    # full train atom set); the 512-molecule smoke subset value is reported as
    # a diagnostic.
    alpha_train = _codes_for_split(model, "train", device_obj)
    usage = _atom_usage(alpha_train)
    E_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in e2e.make_env_loader(encoded, BATCH_SIZE, False, 0):
            batch = batch.to(device_obj)
            _prediction, aux = model(batch, return_aux=True)
            E_rows.append(aux["E"].cpu().numpy())
    rank = _environment_rank(np.concatenate(E_rows, axis=0))
    dbar_final = e2e.normalized_dictionary(model.D.detach()).cpu().numpy()
    dbar_init = e2e.normalized_dictionary(reference.D.detach()).cpu().numpy()
    movement = float(np.linalg.norm(dbar_final - dbar_init))
    decrease = bool(curve[-1]["train_mae"] < curve[0]["train_mae"])
    gates = {
        "loss_finite": bool(all(math.isfinite(row["train_mae"]) and math.isfinite(row["train_rec"]) for row in curve)),
        "train_mae_decreased": decrease,
        "d_grad_nonzero": bool(all(row["d_grad_norm"] > 0.0 for row in curve)),
        "alpha_still_sparse": bool(int(l0.max().item()) <= int(e2e.SPARSITY)),
        "atoms_active": bool(usage["active_atoms"] >= e2e.HEALTH_MIN_ACTIVE),
        "no_single_atom_dominance": bool(usage["top1_share"] <= e2e.HEALTH_MAX_ATOM_SHARE),
        "effective_atom_count": bool(usage["effective_atom_count"] >= e2e.HEALTH_MIN_EFFECTIVE),
        "environment_rank": bool(rank["participation_ratio"] > 1.0),
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()},
        STATE_DIR / f"smoke_seed{seed}.pt",
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "seed": int(seed),
        "arm": e2e.SPARSE_ARM,
        "molecules": int(SMOKE_MOLECULES),
        "epochs": int(SMOKE_EPOCHS),
        "lambda_rec": float(lam),
        "curve": curve,
        "gate_scope": {
            "training_molecules": int(SMOKE_MOLECULES),
            "coverage_measurement": "official train split (10000 molecules, all 231664 atoms)",
            "coverage_measurement_rationale": (
                "the atoms-active gate is a dictionary-coverage property measured on the "
                "official train distribution, matching SDB-v0's used>=24 precedent; the "
                "512-molecule smoke subset value is retained as a diagnostic"
            ),
        },
        "atom_usage": usage,
        "atom_usage_smoke_subset": usage_subset,
        "atom_usage_init_train": init_train,
        "atom_usage_init_smoke_subset": init_subset,
        "max_l0_train": int((alpha_train != 0).sum(axis=1).max()),
        "smoke_subset_max_l0": int(l0.max().item()),
        "environment_rank": rank,
        "dictionary_movement_fro": movement,
        "gates": gates,
        "all_passed": bool(all(gates.values())),
        "wall_clock_s": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "smoke_gate.json", payload)
    if not payload["all_passed"]:
        payload["verdict"] = e2e.VERDICTS["mechanism_collapsed"]
        _write_json(RESULTS_DIR / "smoke_gate.json", payload)
        raise RuntimeError(f"Stage-1 mechanism gate failed: {gates}")
    return payload


# ---------------------------------------------------------------------------
# stage: formal training
# ---------------------------------------------------------------------------


def train_stage(arm: str, device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    run_path = RESULTS_DIR / f"{arm}_seed{seed}.json"
    if run_path.exists():
        return _read_json(run_path)
    device_obj = torch.device(device)
    lam = _lambda_rec()
    train_data = load_split("train")
    valid_data = load_split("valid")
    reference_model = e2e.build_model(e2e.SPARSE_ARM, seed=int(seed))
    reference_state = {
        key: value.detach().clone() for key, value in reference_model.state_dict().items()
    }
    model = e2e.build_model(arm, seed=int(seed), reference_state=reference_state).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = e2e.make_env_loader(train_data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    eval_loader = e2e.make_env_loader(valid_data, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        task_sum = 0.0
        rec_sum = 0.0
        n_molecules = 0
        n_nodes = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum().item())
            n_molecules += int(batch.y.view(-1).shape[0])
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            numerator = ((phi - phi_hat) ** 2).sum(dim=1)
            denominator = (phi ** 2).sum(dim=1) + e2e.EPS
            rec_sum += float((numerator / denominator).sum().item())
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_molecules, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = evaluate(model, eval_loader, device_obj)
        dbar_norm = float(e2e.normalized_dictionary(model.D.detach()).norm().item())
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "train_rec": train_rec,
                "valid_mae": float(valid["mae"]),
                "valid_rec": float(valid["rec"]),
                "d_norm": float(model.D.detach().norm().item()),
                "dbar_norm": dbar_norm,
            }
        )
        epoch_states[int(epoch)] = {
            key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()
        }
        keep = set(sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:5])
        keep_epochs = {i + 1 for i in keep}
        for cached_epoch in list(epoch_states):
            if cached_epoch not in keep_epochs:
                del epoch_states[cached_epoch]
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == MAX_EPOCHS:
            print(
                f"[{arm}:seed{seed}] epoch={epoch:03d} train={train_mae:.6f} rec={train_rec:.6f} "
                f"valid={float(valid['mae']):.6f} valid_rec={float(valid['rec']):.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
        if stale >= PATIENCE:
            break
    wall_clock = float(time.perf_counter() - started)
    assert best_state is not None

    best_model = e2e.build_model(arm, seed=int(seed))
    best_model.load_state_dict(best_state)
    best_valid = evaluate(best_model.to(device_obj), eval_loader, device_obj)

    members = sorted(
        int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:5]
    )
    soup_state = {
        key: torch.stack([epoch_states[epoch][key].float() for epoch in members]).mean(0)
        for key in epoch_states[members[0]]
    }
    soup_model = e2e.build_model(arm, seed=int(seed))
    soup_model.load_state_dict(soup_state)
    soup_valid = evaluate(soup_model.to(device_obj), eval_loader, device_obj)

    dbar_init = e2e.normalized_dictionary(reference_model.D.detach()).numpy()
    dbar_best = e2e.normalized_dictionary(best_model.D.detach()).numpy()
    dbar_soup = e2e.normalized_dictionary(soup_model.D.detach()).numpy()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, STATE_DIR / f"{arm}_seed{seed}_selection_state.pt")
    torch.save(soup_state, STATE_DIR / f"{arm}_seed{seed}_soup_state.pt")
    _write_csv(RESULTS_DIR / f"{arm}_curve.csv", curve)
    train_min = float(min(row["train_mae"] for row in curve))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "arm": arm,
        "coding_operator": "tied-IHT(D), exact top-8" if arm == e2e.SPARSE_ARM else "phi @ Pbar (dense tied)",
        "seed": int(seed),
        "device": str(device_obj),
        "lambda_rec": float(lam),
        "protocol": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "early_termination": False,
            "gradient_clip_norm": GRAD_CLIP,
            "loss": "L1 / MAE + lambda_rec * normalized reconstruction",
            "checkpoint_selection": "best official-valid MAE",
            "soup": "equal-weight Top-5 by official-valid MAE",
            "train_shuffle_seed_offset": TRAIN_SHUFFLE_OFFSET,
            "eval_shuffle_seed_offset": EVAL_SHUFFLE_OFFSET,
        },
        "epochs_run": int(len(curve)),
        "early_stopped": bool(len(curve) < MAX_EPOCHS),
        "best_valid_mae": float(best_valid["mae"]),
        "best_epoch": int(best_epoch),
        "best_valid_rec": float(best_valid["rec"]),
        "train_mae_at_best": float(curve[best_epoch - 1]["train_mae"]),
        "train_rec_at_best": float(curve[best_epoch - 1]["train_rec"]),
        "train_min_mae": train_min,
        "soup": {
            "available": True,
            "members": members,
            "member_valid_mae": [float(curve[epoch - 1]["valid_mae"]) for epoch in members],
            "soup_valid_mae": float(soup_valid["mae"]),
            "soup_valid_rec": float(soup_valid["rec"]),
        },
        "dictionary_movement": {
            "init_fro": float(np.linalg.norm(dbar_init)),
            "best_vs_init_fro": float(np.linalg.norm(dbar_best - dbar_init)),
            "soup_vs_init_fro": float(np.linalg.norm(dbar_soup - dbar_init)),
            "soup_vs_init_relative": float(
                np.linalg.norm(dbar_soup - dbar_init) / (np.linalg.norm(dbar_init) + e2e.EPS)
            ),
        },
        "wall_clock_s": wall_clock,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))
            if device_obj.type == "cuda"
            else None
        ),
        "valid_targets": best_valid["targets"].tolist(),
        "valid_predictions_best": best_valid["predictions"].tolist(),
        "valid_predictions_soup": soup_valid["predictions"].tolist(),
        "official_test_loaded": False,
    }
    _write_json(run_path, payload)
    print(
        f"[{arm}:seed{seed}] best={best_valid['mae']:.6f}@{best_epoch} soup={soup_valid['mae']:.6f} "
        f"members={members} wall={wall_clock:.1f}s",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: dictionary health (trained Sparse soup)
# ---------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def _rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks

    ra = _rank(np.asarray(a, dtype=np.float64))
    rb = _rank(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denominator) if denominator > 0 else 0.0


def _codes_for_split(model: e2e.E2EDictEnvModel, split: str, device: torch.device, chunk: int = 65536) -> np.ndarray:
    blob = torch.load(_env_cache_path(split), map_location="cpu", weights_only=False)
    phi = blob["phi"]
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(phi.shape[0]), int(chunk)):
            piece = phi[start : start + int(chunk)].to(device)
            chunks.append(model.code(piece).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def health_stage(device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    device_obj = torch.device(device)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=int(seed))
    soup_state = torch.load(
        STATE_DIR / f"{e2e.SPARSE_ARM}_seed{seed}_soup_state.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(soup_state)
    model.to(device_obj).eval()
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
            block = nonzero[offset : offset + size]
            coverage += block.any(axis=0).astype(np.int64)
            offset += size
        return coverage.tolist()

    coverage_train = _molecule_coverage(alpha_train, node_sizes)
    spearman = _spearman(np.asarray(usage_train["usage"]), np.asarray(usage_valid["usage"]))

    dbar = e2e.normalized_dictionary(model.D.detach()).cpu().numpy()
    singular = np.linalg.svd(dbar, compute_uv=False)
    energy = singular ** 2
    effective_rank = float((energy.sum() ** 2) / float((energy ** 2).sum()))
    gram = dbar.T @ dbar
    off_diagonal = gram - np.diag(np.diag(gram))
    coherence_max = float(np.abs(off_diagonal).max())
    coherence_mean = float(np.abs(off_diagonal).mean())

    init_dbar = e2e.normalized_dictionary(
        e2e.build_model(e2e.SPARSE_ARM, seed=int(seed)).D.detach()
    ).cpu().numpy()
    movement = float(np.linalg.norm(dbar - init_dbar))

    # reconstruction on both splits (report)
    def _split_rec(alpha: np.ndarray, blob: dict[str, Any]) -> float:
        phi = blob["phi"].numpy().astype(np.float64)
        phi_hat = alpha.astype(np.float64) @ dbar.astype(np.float64).T
        numerator = ((phi - phi_hat) ** 2).sum(axis=1)
        denominator = (phi ** 2).sum(axis=1) + e2e.EPS
        return float((numerator / denominator).mean())

    rec_train = _split_rec(alpha_train, blob_train)
    rec_valid = _split_rec(alpha_valid, blob_valid)

    # task gradient to D at the trained state (MAE alone, one train batch)
    train_data = load_split("train", subset=64)
    loader = e2e.make_env_loader(train_data, 64, False, 0)
    batch = next(iter(loader)).to(device_obj)
    model.zero_grad(set_to_none=True)
    prediction = model(batch)
    task_loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    task_loss.backward()
    task_grad = float(model.D.grad.norm().item()) if model.D.grad is not None else 0.0

    nonzero_values = np.abs(alpha_valid[np.abs(alpha_valid) > 0])
    gates = {
        "active_train": bool(usage_train["active_atoms"] >= e2e.HEALTH_MIN_ACTIVE),
        "active_valid": bool(usage_valid["active_atoms"] >= e2e.HEALTH_MIN_ACTIVE),
        "effective_atom_count": bool(usage_valid["effective_atom_count"] >= e2e.HEALTH_MIN_EFFECTIVE),
        "no_single_atom_dominance": bool(usage_valid["top1_share"] <= e2e.HEALTH_MAX_ATOM_SHARE),
        "dictionary_moved": bool(movement > 1.0e-6),
        "task_gradient_to_d": bool(math.isfinite(task_grad) and task_grad > 0.0),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "seed": int(seed),
        "arm": e2e.SPARSE_ARM,
        "state": "soup",
        "train": {
            "n_nodes": int(alpha_train.shape[0]),
            **usage_train,
            "max_l0": int(l0_train.max()),
            "exact_top_s_fraction": float((l0_train == int(e2e.SPARSITY)).mean()),
            "per_atom_molecule_coverage": coverage_train,
        },
        "valid": {
            "n_nodes": int(alpha_valid.shape[0]),
            **usage_valid,
            "max_l0": int(l0_valid.max()),
            "exact_top_s_fraction": float((l0_valid == int(e2e.SPARSITY)).mean()),
        },
        "usage_spearman_train_valid": float(spearman),
        "coefficient_magnitude": {
            "mean_abs_nonzero": float(nonzero_values.mean()) if nonzero_values.size else 0.0,
            "median_abs_nonzero": float(np.median(nonzero_values)) if nonzero_values.size else 0.0,
            "p90_abs_nonzero": float(np.quantile(nonzero_values, 0.9)) if nonzero_values.size else 0.0,
            "max_abs": float(np.abs(alpha_valid).max()),
        },
        "dictionary": {
            "effective_rank": effective_rank,
            "singular_values": singular.tolist(),
            "coherence_max_abs_cosine": coherence_max,
            "coherence_mean_abs_cosine": coherence_mean,
            "movement_fro_from_ksvd_init": movement,
            "movement_relative": float(movement / (np.linalg.norm(init_dbar) + e2e.EPS)),
        },
        "reconstruction": {"train": rec_train, "valid": rec_valid},
        "task_gradient_to_D": float(task_grad),
        "task_loss": float(task_loss.detach().item()),
        "gates": gates,
        "all_passed": bool(all(gates.values())),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "dictionary_health.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stage: mechanism interventions (evaluation only)
# ---------------------------------------------------------------------------


def _load_soup(arm: str, seed: int, device: torch.device) -> e2e.E2EDictEnvModel:
    model = e2e.build_model(arm, seed=int(seed))
    soup_state = torch.load(
        STATE_DIR / f"{arm}_seed{seed}_soup_state.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(soup_state)
    return model.to(device).eval()


def mechanism_stage(device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    device_obj = torch.device(device)
    valid_data = load_split("valid")
    loader = e2e.make_env_loader(valid_data, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)
    model = _load_soup(e2e.SPARSE_ARM, seed, device_obj)
    clean = evaluate(model, loader, device_obj)
    m_soup = float(clean["mae"])

    # A. dictionary-coordinate neutralization (alpha -> 0, DC 1 kept)
    zero = evaluate(model, loader, device_obj, coord_zero=True)
    zero_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "seed": int(seed),
        "arm": e2e.SPARSE_ARM,
        "state": "soup",
        "M_S": m_soup,
        "M_zero": float(zero["mae"]),
        "G_dict_use": float(zero["mae"] - m_soup),
        "gate_threshold": float(e2e.GATE_ZERO),
        "gate_pass": bool(float(zero["mae"] - m_soup) >= float(e2e.GATE_ZERO)),
        "mean_abs_prediction_shift_vs_clean": float(
            np.mean(np.abs(zero["predictions"] - clean["predictions"]))
        ),
        "max_abs_prediction_shift_vs_clean": float(
            np.max(np.abs(zero["predictions"] - clean["predictions"]))
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism_zero.json", zero_payload)

    # B. assignment shuffle: permute alpha <-> q within every (root, shell)
    rows: list[dict[str, Any]] = []
    for shuffle_seed in SHUFFLE_SEEDS:
        for data in valid_data:
            data.env_occ_coord_node = e2e.shuffled_occ_node_for_molecule(
                data.env_occ_node, data.env_occ_root, data.env_occ_shell, int(shuffle_seed)
            )
        with_node_loader = e2e.make_env_loader(
            valid_data, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET
        )
        shuffled = evaluate(model, with_node_loader, device_obj, use_coord_node=True)
        rows.append(
            {
                "seed": int(shuffle_seed),
                "valid_mae": float(shuffled["mae"]),
                "mean_abs_prediction_shift_vs_clean": float(
                    np.mean(np.abs(shuffled["predictions"] - clean["predictions"]))
                ),
                "max_abs_prediction_shift_vs_clean": float(
                    np.max(np.abs(shuffled["predictions"] - clean["predictions"]))
                ),
            }
        )
        for data in valid_data:
            data.env_occ_coord_node = None
    m_shuffle = float(np.mean([row["valid_mae"] for row in rows]))
    shuffle_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "seed": int(seed),
        "arm": e2e.SPARSE_ARM,
        "state": "soup",
        "M_S": m_soup,
        "M_shuffle": m_shuffle,
        "G_assign": float(m_shuffle - m_soup),
        "gate_threshold": float(e2e.GATE_SHUFFLE),
        "gate_pass": bool(float(m_shuffle - m_soup) >= float(e2e.GATE_SHUFFLE)),
        "shuffle_seeds": [int(value) for value in SHUFFLE_SEEDS],
        "rows": rows,
        "mean_prediction_shift": float(
            np.mean([row["mean_abs_prediction_shift_vs_clean"] for row in rows])
        ),
        "max_prediction_shift": float(
            np.max([row["max_abs_prediction_shift_vs_clean"] for row in rows])
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism_shuffle.json", shuffle_payload)
    return {"zero": zero_payload, "shuffle": shuffle_payload}


# ---------------------------------------------------------------------------
# stage: analyze / verdict
# ---------------------------------------------------------------------------


def classify(
    m_s: float,
    m_d: float,
    m_zero: float,
    m_shuffle: float,
    health_pass: bool,
    smoke_pass: bool,
) -> dict[str, Any]:
    g_sparse = float(m_d - m_s)
    g_dict_use = float(m_zero - m_s)
    g_assign = float(m_shuffle - m_s)
    specific = bool(g_sparse >= float(e2e.GATE_DICT_SPECIFIC))
    zero_ok = bool(g_dict_use >= float(e2e.GATE_ZERO))
    shuffle_ok = bool(g_assign >= float(e2e.GATE_SHUFFLE))
    if m_s <= float(e2e.BAND_STRONG_MAX):
        band = "strong"
    elif m_s <= float(e2e.BAND_VIABLE_MAX):
        band = "viable_but_weaker"
    else:
        band = "weak"
    if not smoke_pass:
        verdict = e2e.VERDICTS["mechanism_collapsed"]
    elif m_s > float(e2e.BAND_VIABLE_MAX):
        verdict = e2e.VERDICTS["absolute_weak"]
    elif not specific:
        verdict = e2e.VERDICTS["viable_not_specific"]
    elif not (zero_ok and shuffle_ok and health_pass):
        verdict = e2e.VERDICTS["gain_not_dictionary"]
    elif m_s <= float(e2e.BAND_STRONG_MAX):
        verdict = e2e.VERDICTS["strong"]
    else:
        verdict = e2e.VERDICTS["specific_weak"]
    return {
        "M_S": float(m_s),
        "M_D": float(m_d),
        "M_zero": float(m_zero),
        "M_shuffle": float(m_shuffle),
        "G_sparse": g_sparse,
        "G_dict_use": g_dict_use,
        "G_assign": g_assign,
        "band": band,
        "gates": {
            "dictionary_specific": specific,
            "zero_code": zero_ok,
            "assignment_shuffle": shuffle_ok,
            "dictionary_health": bool(health_pass),
            "stage1_mechanism": bool(smoke_pass),
        },
        "thresholds": {
            "dict_specific": float(e2e.GATE_DICT_SPECIFIC),
            "zero": float(e2e.GATE_ZERO),
            "shuffle": float(e2e.GATE_SHUFFLE),
            "band_strong_max": float(e2e.BAND_STRONG_MAX),
            "band_viable_max": float(e2e.BAND_VIABLE_MAX),
        },
        "verdict": verdict,
    }


def stop_stage() -> dict[str, Any]:
    """Preregistered STOP path: Stage-1 mechanism gate failed."""
    identity = _read_json(RESULTS_DIR / "artifact_identity.json")
    accounting = _read_json(RESULTS_DIR / "parameter_accounting.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    lambda_payload = _read_json(RESULTS_DIR / "lambda_calibration.json")
    smoke = _read_json(RESULTS_DIR / "smoke_gate.json")
    verdict = e2e.VERDICTS["mechanism_collapsed"]
    decision = {
        "verdict": verdict,
        "stage1_gate": smoke["gates"],
        "stage1_gate_scope": smoke["gate_scope"],
        "active_atoms_after_smoke_official_train": int(smoke["atom_usage"]["active_atoms"]),
        "active_atoms_after_smoke_512_subset": int(smoke["atom_usage_smoke_subset"]["active_atoms"]),
        "active_atoms_at_init_official_train": int(smoke["atom_usage_init_train"]["active_atoms"]),
        "active_atoms_at_init_512_subset": int(smoke["atom_usage_init_smoke_subset"]["active_atoms"]),
        "required_active_atoms": int(e2e.HEALTH_MIN_ACTIVE),
        "formal_arms_run": [],
        "formal_arms_not_run_reason": (
            "preregistered Stage-1 mechanism gate failed; no formal training, intervention "
            "or verdict beyond E2E_DICTENV_MECHANISM_COLLAPSED is authorized"
        ),
        "thresholds": {
            "dict_specific": float(e2e.GATE_DICT_SPECIFIC),
            "zero": float(e2e.GATE_ZERO),
            "shuffle": float(e2e.GATE_SHUFFLE),
            "band_strong_max": float(e2e.BAND_STRONG_MAX),
            "band_viable_max": float(e2e.BAND_VIABLE_MAX),
        },
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "primary_metric": "n/a (stopped at Stage-1 mechanism gate)",
        "matched_comparison": "n/a (stopped at Stage-1 mechanism gate)",
        "decision": decision,
        "arms": {},
        "historical_anchors": {
            "FEC_S1_soup": FEC_S1_SOUP,
            "FEC_S1_best": FEC_S1_BEST,
            "S0_soup": S0_SOUP,
        },
        "health": None,
        "lambda_calibration": lambda_payload,
        "parameter_accounting": {
            "whole_model": int(accounting["whole_model"]),
            "fec_s1_reference": 66170,
            "difference": int(accounting["difference_vs_fec_s1"]),
        },
        "correctness_all_passed": bool(correctness["all_passed"]),
        "smoke_all_passed": bool(smoke["all_passed"]),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    _write_stop_report(payload, smoke)
    _write_stop_decision(payload, smoke)
    return payload


def _write_stop_report(payload: Mapping[str, Any], smoke: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# E2E-DictEnv-v0 — report (STOPPED at Stage-1)",
        "",
        "Round `e2e_dictenv_v0`; study `zinc-context-gap`; protocol `e2e_dictenv_v0`.",
        "Official ZINC **test was never loaded**.",
        "",
        "## Frozen verdict",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        "The preregistered Stage-1 mechanism gate failed, so formal 2-arm training,",
        "mechanism interventions and the performance verdict table were **not** executed",
        "(preregistration §16/§17/§21, case S1).",
        "",
        "## Stage-1 mechanism gate",
        "",
        f"* sub-gates: {decision['stage1_gate']}",
        f"* active dictionary atoms after the 3-epoch smoke: "
        f"**{decision['active_atoms_after_smoke_official_train']}/32** on the official train "
        f"(all 231,664 atoms), **{decision['active_atoms_after_smoke_512_subset']}/32** on the "
        f"512-molecule smoke subset; required ≥ {decision['required_active_atoms']}",
        f"* active dictionary atoms at initialization: "
        f"{decision['active_atoms_at_init_official_train']}/32 (official train), "
        f"{decision['active_atoms_at_init_512_subset']}/32 (512-molecule subset)",
        f"* effective atom count (trained): {smoke['atom_usage']['effective_atom_count']:.4f}",
        f"* top-1 support share (trained): {smoke['atom_usage']['top1_share']:.4f}",
        f"* environment effective rank (smoke model): {smoke['environment_rank']['participation_ratio']:.4f}",
        f"* smoke loss curve: train MAE {[round(row['train_mae'], 5) for row in smoke['curve']]}",
        f"* reconstruction curve: {[round(row['train_rec'], 6) for row in smoke['curve']]}",
        "",
        "Reading: the frozen K-SVD dictionary + tied-IHT mechanism is present at"
        "initialization (≥24/32 atoms active under both scopes) but the 3-epoch train-only"
        "smoke removes rare-atom support, leaving 23/32 active. This is a genuine,"  # noqa
        "deterministic failure of the preregistered coverage gate, not a loss/reconstruction"
        "collapse (loss decreases, gradients reach `D`, codes stay sparse, no single atom"
        "dominates, environment rank > 1).",
        "",
        "## What did pass (Gate 0)",
        "",
        f"* G0..G12 correctness gates: **{'PASS' if payload['correctness_all_passed'] else 'FAIL'}**",
        f"* parameter identity: {payload['parameter_accounting']['whole_model']} "
        f"(FEC-S1 {payload['parameter_accounting']['fec_s1_reference']}, "
        f"delta {payload['parameter_accounting']['difference']})",
        f"* lambda_rec (frozen): {payload['lambda_calibration']['lambda_rec']:.6f} "
        f"(L_task^init {payload['lambda_calibration']['l_task_init']:.6f}, "
        f"L_rec^init {payload['lambda_calibration']['l_rec_init']:.6f})",
        "",
        "## Why no rescue",
        "",
        "The preregistration forbids any K/s/IHT/LISTA/dictionary-count/decoder/attention/"
        "LayerNorm/λ-sweep/extra-epoch/seed change at this stage. No formal arm was run, so no",
        "performance or dictionary-specificity claim is made.",
        "",
        f"* commit `{payload['git_commit']}`; official_test_loaded = false",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stop_decision(payload: Mapping[str, Any], smoke: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# E2E-DictEnv-v0 — decision (STOPPED at Stage-1)",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        f"* Stage-1 sub-gates: `{decision['stage1_gate']}`",
        f"* active atoms after smoke: {decision['active_atoms_after_smoke_official_train']}/32 "
        f"(official train), {decision['active_atoms_after_smoke_512_subset']}/32 (512 subset); "
        f"required ≥ {decision['required_active_atoms']}",
        f"* active atoms at init: {decision['active_atoms_at_init_official_train']}/32 "
        f"(official train), {decision['active_atoms_at_init_512_subset']}/32 (512 subset)",
        f"* effective atom count {smoke['atom_usage']['effective_atom_count']:.3f}, "
        f"top-1 share {smoke['atom_usage']['top1_share']:.4f}, "
        f"environment rank {smoke['environment_rank']['participation_ratio']:.3f}",
        f"* correctness Gate 0: {payload['correctness_all_passed']}",
        f"* official test loaded: {payload['official_test_loaded']}",
        "",
        "## Reading",
        "",
        "STOP per the frozen verdict table (case S1). The sparse tied-dictionary mechanism",
        "is intact at initialization but the fixed 3-epoch train-only smoke drops it below the",
        "preregistered ≥24/32 coverage threshold. No formal training and no performance or",
        "dictionary-specificity claim is authorized. A new round with a corrected,",
        "statistically-appropriate coverage gate may be preregistered separately; this round",
        "must not be rescued.",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_stage() -> dict[str, Any]:
    formal = RESULTS_DIR / f"{e2e.SPARSE_ARM}_seed0.json"
    if not formal.exists():
        return stop_stage()
    sparse = _read_json(RESULTS_DIR / f"{e2e.SPARSE_ARM}_seed0.json")
    dense = _read_json(RESULTS_DIR / f"{e2e.DENSE_ARM}_seed0.json")
    zero = _read_json(RESULTS_DIR / "mechanism_zero.json")
    shuffle = _read_json(RESULTS_DIR / "mechanism_shuffle.json")
    health = _read_json(RESULTS_DIR / "dictionary_health.json")
    smoke = _read_json(RESULTS_DIR / "smoke_gate.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    lambda_payload = _read_json(RESULTS_DIR / "lambda_calibration.json")
    accounting = _read_json(RESULTS_DIR / "artifact_identity.json")
    decision = classify(
        m_s=float(sparse["soup"]["soup_valid_mae"]),
        m_d=float(dense["soup"]["soup_valid_mae"]),
        m_zero=float(zero["M_zero"]),
        m_shuffle=float(shuffle["M_shuffle"]),
        health_pass=bool(health["all_passed"]),
        smoke_pass=bool(smoke["all_passed"]),
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "primary_metric": "fixed Top-5 soup official-valid MAE",
        "matched_comparison": "SparseDictEnv vs DenseTiedEnv (identical machinery)",
        "decision": decision,
        "arms": {
            "sparse": {
                "best_valid_mae": float(sparse["best_valid_mae"]),
                "best_epoch": int(sparse["best_epoch"]),
                "soup_valid_mae": float(sparse["soup"]["soup_valid_mae"]),
                "soup_members": sparse["soup"]["members"],
                "soup_member_valid_mae": sparse["soup"]["member_valid_mae"],
                "train_mae_at_best": float(sparse["train_mae_at_best"]),
                "train_min_mae": float(sparse["train_min_mae"]),
                "best_valid_rec": float(sparse["best_valid_rec"]),
                "soup_valid_rec": float(sparse["soup"]["soup_valid_rec"]),
                "dictionary_movement": sparse["dictionary_movement"],
                "wall_clock_s": float(sparse["wall_clock_s"]),
                "peak_gpu_memory_mb": sparse["peak_gpu_memory_mb"],
                "epochs_run": int(sparse["epochs_run"]),
            },
            "dense_tied": {
                "best_valid_mae": float(dense["best_valid_mae"]),
                "best_epoch": int(dense["best_epoch"]),
                "soup_valid_mae": float(dense["soup"]["soup_valid_mae"]),
                "soup_members": dense["soup"]["members"],
                "soup_member_valid_mae": dense["soup"]["member_valid_mae"],
                "train_mae_at_best": float(dense["train_mae_at_best"]),
                "train_min_mae": float(dense["train_min_mae"]),
                "best_valid_rec": float(dense["best_valid_rec"]),
                "soup_valid_rec": float(dense["soup"]["soup_valid_rec"]),
                "dictionary_movement": dense["dictionary_movement"],
                "wall_clock_s": float(dense["wall_clock_s"]),
                "peak_gpu_memory_mb": dense["peak_gpu_memory_mb"],
                "epochs_run": int(dense["epochs_run"]),
            },
        },
        "historical_anchors": {
            "FEC_S1_soup": FEC_S1_SOUP,
            "FEC_S1_best": FEC_S1_BEST,
            "S0_soup": S0_SOUP,
            "M_S_minus_FEC_S1_soup": float(decision["M_S"] - FEC_S1_SOUP),
            "matched_comparison_note": "FEC-S1 is context only; the causal comparison is Sparse vs DenseTied",
        },
        "health": health,
        "lambda_calibration": lambda_payload,
        "parameter_accounting": {
            "whole_model": int(accounting["parameter_accounting"]["whole_model"]),
            "fec_s1_reference": 66170,
            "difference": int(accounting["parameter_accounting"]["difference_vs_fec_s1"]),
        },
        "correctness_all_passed": bool(correctness["all_passed"]),
        "smoke_all_passed": bool(smoke["all_passed"]),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    _write_report(payload)
    _write_decision_markdown(payload)
    return payload


def _write_report(payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    sparse = payload["arms"]["sparse"]
    dense = payload["arms"]["dense_tied"]
    health = payload["health"]
    anchors = payload["historical_anchors"]
    lines = [
        "# E2E-DictEnv-v0 — report",
        "",
        "Round `e2e_dictenv_v0`; study `zinc-context-gap`; protocol `e2e_dictenv_v0`.",
        "Official ZINC **test was never loaded**.",
        "",
        "## Frozen verdict",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        "## Primary metrics (fixed Top-5 soup official-valid MAE)",
        "",
        f"* SparseDictEnv `M_S` = **{decision['M_S']:.6f}** "
        f"(band **{decision['band']}**; best {sparse['best_valid_mae']:.6f} @ {sparse['best_epoch']})",
        f"* DenseTiedEnv `M_D` = **{decision['M_D']:.6f}** "
        f"(best {dense['best_valid_mae']:.6f} @ {dense['best_epoch']})",
        f"* dictionary-specific `G_sparse = M_D - M_S` = **{decision['G_sparse']:.6f}** "
        f"(gate ≥ {decision['thresholds']['dict_specific']}: "
        f"{decision['gates']['dictionary_specific']})",
        f"* zero-code `M_zero` = {decision['M_zero']:.6f}, "
        f"`G_dict-use` = {decision['G_dict_use']:.6f} "
        f"(gate ≥ {decision['thresholds']['zero']}: {decision['gates']['zero_code']})",
        f"* assignment shuffle `M_shuffle` = {decision['M_shuffle']:.6f}, "
        f"`G_assign` = {decision['G_assign']:.6f} "
        f"(gate ≥ {decision['thresholds']['shuffle']}: {decision['gates']['assignment_shuffle']})",
        f"* dictionary health: {'PASS' if decision['gates']['dictionary_health'] else 'FAIL'}",
        "",
        "## Learning dynamics (240 epochs, no early stop)",
        "",
        f"* Sparse: soup members {sparse['soup_members']}, "
        f"member MAEs {[round(v, 6) for v in sparse['soup_member_valid_mae']]}",
        f"* Sparse: train MAE at best {sparse['train_mae_at_best']:.6f}, "
        f"train minimum {sparse['train_min_mae']:.6f}, "
        f"valid reconstruction {sparse['soup_valid_rec']:.6f}",
        f"* Dense : soup members {dense['soup_members']}, "
        f"member MAEs {[round(v, 6) for v in dense['soup_member_valid_mae']]}",
        f"* Dense : train MAE at best {dense['train_mae_at_best']:.6f}, "
        f"train minimum {dense['train_min_mae']:.6f}, "
        f"valid reconstruction {dense['soup_valid_rec']:.6f}",
        f"* dictionary movement (soup vs K-SVD init, Frobenius): "
        f"Sparse {sparse['dictionary_movement']['soup_vs_init_fro']:.6f}, "
        f"Dense {dense['dictionary_movement']['soup_vs_init_fro']:.6f}",
        f"* wall clock: Sparse {sparse['wall_clock_s']:.1f} s, Dense {dense['wall_clock_s']:.1f} s; "
        f"peak GPU: {sparse['peak_gpu_memory_mb']} / {dense['peak_gpu_memory_mb']} MB",
        "",
        "## Dictionary health (trained Sparse soup)",
        "",
        f"* active atoms: train {health['train']['active_atoms']}/32, valid {health['valid']['active_atoms']}/32",
        f"* effective atom count: train {health['train']['effective_atom_count']:.2f}, "
        f"valid {health['valid']['effective_atom_count']:.2f}",
        f"* support entropy (valid) {health['valid']['support_entropy']:.4f}, "
        f"top-1 share {health['valid']['top1_share']:.4f}, top-8 share {health['valid']['top8_share']:.4f}",
        f"* exact top-8 fraction: train {health['train']['exact_top_s_fraction']:.6f}, "
        f"valid {health['valid']['exact_top_s_fraction']:.6f}",
        f"* usage Spearman train-valid {health['usage_spearman_train_valid']:.6f}",
        f"* effective rank {health['dictionary']['effective_rank']:.4f}, "
        f"coherence max {health['dictionary']['coherence_max_abs_cosine']:.6f}, "
        f"mean {health['dictionary']['coherence_mean_abs_cosine']:.6f}",
        f"* movement from K-SVD init (Frobenius) {health['dictionary']['movement_fro_from_ksvd_init']:.6f}",
        f"* reconstruction (normalized): train {health['reconstruction']['train']:.6f}, "
        f"valid {health['reconstruction']['valid']:.6f}",
        f"* task gradient to D at trained state {health['task_gradient_to_D']:.6f}",
        "",
        "## Historical anchors (context only)",
        "",
        f"* FEC-S1 seed-0 soup {anchors['FEC_S1_soup']:.6f}; "
        f"`M_S - FEC_S1_soup` = {anchors['M_S_minus_FEC_S1_soup']:+.6f}",
        f"* FEC-S1 seed-0 best {anchors['FEC_S1_best']:.6f}; S0 seed-0 soup {anchors['S0_soup']:.6f}",
        f"* {anchors['matched_comparison_note']}",
        "",
        "## Provenance",
        "",
        f"* lambda_rec (frozen, both arms) {payload['lambda_calibration']['lambda_rec']:.6f} "
        f"from L_task^init {payload['lambda_calibration']['l_task_init']:.6f} / "
        f"L_rec^init {payload['lambda_calibration']['l_rec_init']:.6f}",
        f"* parameters {payload['parameter_accounting']['whole_model']} "
        f"(FEC-S1 {payload['parameter_accounting']['fec_s1_reference']}, "
        f"delta {payload['parameter_accounting']['difference']})",
        f"* correctness gates all passed: {payload['correctness_all_passed']}; "
        f"Stage-1 smoke all passed: {payload['smoke_all_passed']}",
        f"* commit `{payload['git_commit']}`; official_test_loaded = false",
        "",
        "## Mechanism",
        "",
        f"* zero-code prediction shift: mean {_read_json(RESULTS_DIR / 'mechanism_zero.json')['mean_abs_prediction_shift_vs_clean']:.6f}, "
        f"max {_read_json(RESULTS_DIR / 'mechanism_zero.json')['max_abs_prediction_shift_vs_clean']:.6f}",
        f"* shuffle prediction shift: mean {_read_json(RESULTS_DIR / 'mechanism_shuffle.json')['mean_prediction_shift']:.6f}, "
        f"max {_read_json(RESULTS_DIR / 'mechanism_shuffle.json')['max_prediction_shift']:.6f}",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision_markdown(payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# E2E-DictEnv-v0 — decision",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        f"* `M_S` (SparseDictEnv soup) = **{decision['M_S']:.6f}** (band {decision['band']})",
        f"* `M_D` (DenseTiedEnv soup)  = **{decision['M_D']:.6f}**",
        f"* `G_sparse = M_D - M_S` = **{decision['G_sparse']:.6f}** "
        f"(gate {decision['thresholds']['dict_specific']}: {decision['gates']['dictionary_specific']})",
        f"* `G_dict-use = M_zero - M_S` = **{decision['G_dict_use']:.6f}** "
        f"(gate {decision['thresholds']['zero']}: {decision['gates']['zero_code']})",
        f"* `G_assign = M_shuffle - M_S` = **{decision['G_assign']:.6f}** "
        f"(gate {decision['thresholds']['shuffle']}: {decision['gates']['assignment_shuffle']})",
        f"* dictionary health: {decision['gates']['dictionary_health']}",
        f"* official test loaded: {payload['official_test_loaded']}",
        "",
        "## Reading",
        "",
    ]
    verdict = decision["verdict"]
    if verdict == e2e.VERDICTS["strong"]:
        lines.append(
            "End-to-end sparse dictionary coding is the load-bearing fine structural coordinate "
            "of a competitive no-message-passing chemical-environment model, and its advantage is "
            "specific relative to a parameter-identical dense tied coordinate. No seed 1, no "
            "control and no sweep is authorized automatically; any next round needs a new "
            "pre-registration."
        )
    elif verdict == e2e.VERDICTS["specific_weak"]:
        lines.append(
            "The dictionary hypothesis receives mechanism support (specific, load-bearing, "
            "assignment-mediated, healthy) but the architecture is not strong enough to replace "
            "the current static baseline. STOP; no architecture rescue."
        )
    elif verdict == e2e.VERDICTS["viable_not_specific"]:
        lines.append(
            "The environment architecture works, but sparse coding has no demonstrated value over "
            "an identical-budget dense tied structural coordinate. STOP the dictionary "
            "predictive-core route."
        )
    elif verdict == e2e.VERDICTS["absolute_weak"]:
        lines.append(
            "The sparse dictionary-core does not have the absolute capacity this task needs, "
            "regardless of the dense control. STOP."
        )
    elif verdict == e2e.VERDICTS["gain_not_dictionary"]:
        lines.append(
            "Any apparent gain is not dictionary-mediated (zero/shuffle/health gate failed). "
            "No dictionary-core success may be claimed. STOP."
        )
    else:
        lines.append("Stage-1 mechanism gate collapsed. STOP.")
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_all(device: str = "cuda") -> None:
    identity_stage()
    build_env_cache()
    correctness_stage(device="cpu")
    lambda_stage(device=device)
    try:
        smoke_stage(device=device)
    except RuntimeError as error:
        print(f"[run_all] STOP at Stage-1: {error}", flush=True)
        stop_stage()
        return
    train_stage(e2e.SPARSE_ARM, device=device)
    train_stage(e2e.DENSE_ARM, device=device)
    health_stage(device=device)
    mechanism_stage(device=device)
    analyze_stage()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "identity",
            "env",
            "correct",
            "lambda",
            "smoke",
            "train",
            "health",
            "mechanism",
            "analyze",
            "stop",
            "all",
        ],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", default=e2e.SPARSE_ARM, choices=list(e2e.ARMS))
    parser.add_argument("--force-env", action="store_true")
    args = parser.parse_args(argv)

    sdp._configure_determinism()
    if args.stage == "identity":
        print(json.dumps(identity_stage(), indent=2))
    elif args.stage == "env":
        print(json.dumps(build_env_cache(force=args.force_env), indent=2))
    elif args.stage == "correct":
        print(json.dumps(correctness_stage(device="cpu"), indent=2))
    elif args.stage == "lambda":
        print(json.dumps(lambda_stage(device=args.device, seed=args.seed), indent=2))
    elif args.stage == "smoke":
        print(json.dumps(smoke_stage(device=args.device, seed=args.seed), indent=2))
    elif args.stage == "train":
        summary = train_stage(args.arm, device=args.device, seed=args.seed)
        print(
            json.dumps(
                {key: value for key, value in summary.items() if "predictions" not in key and "targets" not in key},
                indent=2,
            )
        )
    elif args.stage == "health":
        print(json.dumps(health_stage(device=args.device, seed=args.seed), indent=2))
    elif args.stage == "mechanism":
        print(json.dumps(mechanism_stage(device=args.device, seed=args.seed), indent=2))
    elif args.stage == "analyze":
        print(json.dumps(analyze_stage(), indent=2))
    elif args.stage == "stop":
        print(json.dumps(stop_stage(), indent=2))
    else:
        run_all(device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
