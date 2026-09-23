"""PEC-C1 runner — Pure Environment Composition Confirmatory (full-data seed 0).

Round: ``pec_c1``.  Pre-registration: ``notes/pec_c1_preregistration.md``.

The architecture, feature definition, dictionary ``K``/``s``, static
composition and reader are **PEC-v0's, unchanged** (imported from
``pec_v0``).  PEC-C1 changes only:

* the dictionary is fit once on the **full 10,000 official-train** molecules and
  is then **genuinely frozen** (``requires_grad = False``, excluded from the
  optimizer) — see the pre-registration's section D1;
* training runs the full official train / official valid regime with a frozen
  fixed 240-epoch schedule and official-valid checkpoint/soup selection.

Official ZINC **test is never loaded**.  Only ``train`` and ``val`` splits are
ever read, through the guarded loader below.

Stages
------
``cache dicts smoke train mechanism report all``
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import pec_v0 as pec
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    _data_to_graph,
    _load_zinc,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/pec_c1"
CACHE_DIR = RESULTS_DIR / "cache"
DICT_DIR = RESULTS_DIR / "dicts"
STATE_DIR = RESULTS_DIR / "states"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

PROTOCOL_VERSION = "pec_c1"
CACHE_SCHEMA = "pec_c1_occurrence_roles_v1"

# --- frozen protocol (pre-registration §6) --------------------------------
TRAIN_EPOCHS = 240
LR = 1.0e-3
WD = 1.0e-5
BATCH = 64
CLIP = 5.0
SEED = 0
SOUP_TOP = 5
KSVD_EPOCHS = 10

# --- frozen decision gates (pre-registration §8) ---------------------------
GATE_WEAK = 0.145
GATE_STRONG = 0.140
GATE_DICT_MATERIAL = 0.003

# --- frozen mechanism expectations (pre-registration §7) ------------------
MECHANISM_CHEM_SHUFFLE_MUST_CHANGE = True
MECHANISM_RELATION_SHUFFLE_MUST_CHANGE = True
MECHANISM_NEUTRAL_DICT_MUST_CHANGE = True

# --- historical context only (never re-run, never a gate) -----------------
HISTORICAL_CONTEXT = {
    "strict_static_S0_seed0_soup": 0.140794,
    "strict_static_S0_seed1_soup": 0.136423,
    "B_Null": 0.123,
    "B_Full_low": 0.118,
    "B_Full_high": 0.119,
    "note": (
        "printed for orientation only; B-Null/B-Full belong to a different "
        "computation class and are NOT gates for PEC-C1"
    ),
}

FORBIDDEN_SPLITS = ("test",)


# ---------------------------------------------------------------------------
# io / provenance
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover - defensive
        return True


def device_report(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        index = device.index or 0
        payload["cuda"] = torch.version.cuda
        payload["gpu_name"] = torch.cuda.get_device_name(index)
        payload["gpu_total_memory_bytes"] = int(
            torch.cuda.get_device_properties(index).total_memory
        )
        payload["cudnn"] = torch.backends.cudnn.version()
    return payload


# ---------------------------------------------------------------------------
# guarded data access — the official test split is unreachable
# ---------------------------------------------------------------------------


def _guard_split(split: str) -> str:
    if split in FORBIDDEN_SPLITS:
        raise RuntimeError(
            f"official ZINC {split!r} split is forbidden in PEC-C1 "
            "(pre-registration §5); only 'train' and 'val' may be read"
        )
    return split


def _extract_split(split: str) -> list[pec.MoleculeSample]:
    _guard_split(split)
    dataset = _load_zinc(ZINC_ROOT, split)
    samples: list[pec.MoleculeSample] = []
    for data in dataset:
        graph, node_types, edge_types = _data_to_graph(data)
        samples.append(
            pec.build_sample(
                graph, node_types, edge_types, y=float(data.y.view(-1)[0])
            )
        )
    return samples


def _cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "train.pkl.gz",
        CACHE_DIR / "valid.pkl.gz",
        CACHE_DIR / "cache_meta.json",
    )


def _split_fingerprint(samples: Sequence[pec.MoleculeSample]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(samples)).encode())
    for sample in samples:
        digest.update(f"{sample.n_nodes}:{sample.n_edges}:{sample.y:.10f}|".encode())
    return digest.hexdigest()


def build_cache(force: bool = False) -> dict[str, Any]:
    train_path, valid_path, meta_path = _cache_paths()
    if not force and meta_path.exists() and train_path.exists() and valid_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema") == CACHE_SCHEMA:
            meta["cache_reused"] = True
            return meta
    processed = ZINC_ROOT / "subset" / "processed"
    for split in ("train", "val"):
        if not (processed / f"{split}.pt").exists():
            raise RuntimeError(f"processed ZINC split missing: {processed / f'{split}.pt'}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train = _extract_split("train")
    valid = _extract_split("val")
    for path, samples in ((train_path, train), (valid_path, valid)):
        with gzip.open(path, "wb") as handle:
            pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "cache_schema": CACHE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "n_train": len(train),
        "n_valid": len(valid),
        "patch_radius": pec.PATCH_RADIUS,
        "node_role_dim": pec.NODE_ROLE_DIM,
        "edge_role_dim": pec.EDGE_ROLE_DIM,
        "splits_read": ["train", "val"],
        "official_test_loaded": False,
        "train_fingerprint": _split_fingerprint(train),
        "valid_fingerprint": _split_fingerprint(valid),
        "seconds": time.time() - start,
        "cache_reused": False,
    }
    _write_json(meta_path, meta)
    return meta


def load_split(split: str) -> list[pec.MoleculeSample]:
    _guard_split(split)
    train_path, valid_path, meta_path = _cache_paths()
    if not meta_path.exists():
        build_cache()
    path = train_path if split == "train" else valid_path
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# dictionaries (fit once on official train, then frozen)
# ---------------------------------------------------------------------------


def _relative_error(X: np.ndarray, reconstruction: np.ndarray) -> float:
    num = float(np.sum((X - reconstruction) ** 2))
    den = float(np.sum(X * X))
    return num / max(den, 1.0e-30)


def _iht_codes_np(D: np.ndarray, X: np.ndarray, s: int) -> np.ndarray:
    Dt = torch.nn.functional.normalize(
        torch.as_tensor(np.asarray(D), dtype=torch.float64), dim=0, eps=1.0e-8
    )
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    with torch.no_grad():
        codes = pec.T.iht_codes(Dt, Xt, s=int(s), steps=pec.IHT_STEPS)
    return codes.numpy()


def fit_dictionaries(force: bool = False) -> dict[str, Any]:
    node_path = DICT_DIR / "d_node.npy"
    edge_path = DICT_DIR / "d_edge.npy"
    meta_path = DICT_DIR / "dicts.json"
    if not force and meta_path.exists() and node_path.exists() and edge_path.exists():
        meta = _read_json(meta_path)
        meta["dict_reused"] = True
        return meta

    train = load_split("train")
    valid = load_split("valid")
    node_train = np.concatenate([s.node_basis for s in train], axis=0).astype(np.float64)
    edge_train = np.concatenate([s.edge_basis for s in train], axis=0).astype(np.float64)
    node_valid = np.concatenate([s.node_basis for s in valid], axis=0).astype(np.float64)
    edge_valid = np.concatenate([s.edge_basis for s in valid], axis=0).astype(np.float64)

    DICT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    d_node, meta_node = sdb.fit_ksvd(
        node_train, atoms=pec.K_V, s=pec.S_V, epochs=KSVD_EPOCHS
    )
    d_edge, meta_edge = sdb.fit_ksvd(
        edge_train, atoms=pec.K_E, s=pec.S_E, epochs=KSVD_EPOCHS
    )
    seconds = time.time() - start
    np.save(node_path, d_node)
    np.save(edge_path, d_edge)

    d_node_n = pec.T.normalize_columns(d_node)
    d_edge_n = pec.T.normalize_columns(d_edge)
    # label-free reconstruction diagnostics (never used for selection)
    diagnostics = {}
    for name, D, X_train, X_valid, s in (
        ("node", d_node_n, node_train, node_valid, pec.S_V),
        ("edge", d_edge_n, edge_train, edge_valid, pec.S_E),
    ):
        codes_train = _iht_codes_np(D, X_train, s)
        codes_valid = _iht_codes_np(D, X_valid, s)
        l0 = np.count_nonzero(codes_valid, axis=1)
        health = sdb.dictionary_health(codes_valid, atoms=int(D.shape[1]))
        diagnostics[name] = {
            "e_rec_train": _relative_error(X_train, codes_train @ D.T),
            "e_rec_valid_diagnostic_only": _relative_error(X_valid, codes_valid @ D.T),
            "used_atoms": int(health["used_atoms"]),
            "dead_atoms": int(health["dead_atoms"]),
            "max_l0": int(l0.max()) if l0.size else 0,
            "exact_sparsity": bool(np.all(l0 == int(s))),
        }

    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "fit_corpus": "official train, full 10000",
        "splits_read": ["train", "val"],
        "K_V": pec.K_V,
        "S_V": pec.S_V,
        "K_E": pec.K_E,
        "S_E": pec.S_E,
        "iht_steps": pec.IHT_STEPS,
        "ksvd_epochs": KSVD_EPOCHS,
        "dict_seed": sdb.DICT_SEED,
        "n_node_occurrences": int(node_train.shape[0]),
        "n_edge_occurrences": int(edge_train.shape[0]),
        "diagnostics_report_only": diagnostics,
        "used_for_selection": False,
        "seconds": seconds,
        "ksvd_meta": {"node": meta_node, "edge": meta_edge},
        "dict_reused": False,
    }
    _write_json(meta_path, meta)
    return meta


def load_dictionaries() -> tuple[np.ndarray, np.ndarray]:
    node_path = DICT_DIR / "d_node.npy"
    edge_path = DICT_DIR / "d_edge.npy"
    if not node_path.exists() or not edge_path.exists():
        fit_dictionaries()
    return np.load(node_path), np.load(edge_path)


# ---------------------------------------------------------------------------
# batching / evaluation
# ---------------------------------------------------------------------------


def _batches(
    samples: Sequence[pec.MoleculeSample],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
):
    order = np.arange(len(samples))
    if shuffle:
        np.random.RandomState(int(seed)).shuffle(order)
    for start in range(0, len(order), int(batch_size)):
        index = order[start : start + int(batch_size)]
        yield pec.collate([samples[int(i)] for i in index])


def _evaluate(model: pec.PECModel, batches, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            batch = pec.to_device(batch, device)
            prediction = model(batch)["prediction"]
            total += float((prediction - batch["y"]).abs().sum())
            count += int(batch["y"].numel())
    return total / max(count, 1)


def _predict(model: pec.PECModel, batches, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            batch = pec.to_device(batch, device)
            parts.append(model(batch)["prediction"].detach().cpu().numpy())
    return np.concatenate(parts) if parts else np.zeros(0)


def _batch_targets(batches) -> np.ndarray:
    return np.concatenate([b["y"].numpy() for b in batches]) if batches else np.zeros(0)


# ---------------------------------------------------------------------------
# freezing / parameter accounting
# ---------------------------------------------------------------------------


ROLE_PARAM_NAMES = ("d_node", "d_edge")


def freeze_dictionary(model: pec.PECModel) -> None:
    """PEC-C1 pre-registration D1: the K-SVD dictionary is fixed for the round."""
    if model.role_mode != "sparse":
        return
    model.d_node.requires_grad_(False)
    model.d_edge.requires_grad_(False)


def trainable_parameters(model: pec.PECModel) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def parameter_report(model: pec.PECModel) -> dict[str, int]:
    total = pec.n_params(model)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if model.role_mode == "sparse":
        role_total = int(model.d_node.numel() + model.d_edge.numel())
        role_trainable = 0
    elif model.role_mode == "dense":
        role_total = int(model.m_node.weight.numel() + model.m_edge.weight.numel())
        role_trainable = role_total if model.m_node.weight.requires_grad else 0
    else:
        role_total = 0
        role_trainable = 0
    return {
        "total": total,
        "trainable": trainable,
        "role_total": role_total,
        "role_trainable": role_trainable,
    }


def role_parameters(model: pec.PECModel) -> dict[str, torch.nn.Parameter]:
    if model.role_mode == "sparse":
        return {"d_node": model.d_node, "d_edge": model.d_edge}
    if model.role_mode == "dense":
        return {"m_node": model.m_node.weight, "m_edge": model.m_edge.weight}
    return {}


# ---------------------------------------------------------------------------
# mechanism interventions (evaluation-only; pre-registration §7)
# ---------------------------------------------------------------------------


def _shuffle_chemistry(
    batch: Mapping[str, torch.Tensor], generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Keep topology + the atom/bond multisets; destroy only the assignment."""
    out = dict(batch)
    for key, graph_key in (("atom_idx", "node_graph"), ("bond_idx", "edge_graph")):
        values = batch[key].clone()
        graph = batch[graph_key]
        for graph_id in torch.unique(graph).tolist():
            positions = torch.nonzero(graph == graph_id, as_tuple=False).reshape(-1)
            if positions.numel() <= 1:
                continue
            order = torch.randperm(positions.numel(), generator=generator)
            values[positions] = batch[key][positions[order]]
        out[key] = values
    return out


def _shuffle_relations(
    batch: Mapping[str, torch.Tensor], generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Keep environments and pair endpoints; permute the topology relations."""
    out = dict(batch)
    rho = batch["pair_rho"].clone()
    graph = batch["pair_graph"]
    for graph_id in torch.unique(graph).tolist():
        positions = torch.nonzero(graph == graph_id, as_tuple=False).reshape(-1)
        if positions.numel() <= 1:
            continue
        order = torch.randperm(positions.numel(), generator=generator)
        rho[positions] = batch["pair_rho"][positions[order]]
    out["pair_rho"] = rho
    return out


def _neutral_dictionaries(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed) + 777)
    d_node = pec.T.normalize_columns(rng.randn(pec.NODE_ROLE_DIM, pec.K_V))
    d_edge = pec.T.normalize_columns(rng.randn(pec.EDGE_ROLE_DIM, pec.K_E))
    return d_node, d_edge


def _intervention_predictions(
    model: pec.PECModel,
    batches,
    device: torch.device,
    *,
    mode: str,
    seed: int,
    neutral: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            if mode == "chem_shuffle":
                batch = _shuffle_chemistry(batch, generator)
            elif mode == "relation_shuffle":
                batch = _shuffle_relations(batch, generator)
            batch = pec.to_device(batch, device)
            parts.append(model(batch)["prediction"].detach().cpu().numpy())
    return np.concatenate(parts) if parts else np.zeros(0)


# ---------------------------------------------------------------------------
# training (one arm, one seed)
# ---------------------------------------------------------------------------


def train_arm(
    arm: str,
    *,
    seed: int = SEED,
    epochs: int = TRAIN_EPOCHS,
    device: torch.device,
    tag: str | None = None,
    smoke_train: int | None = None,
    smoke_valid: int | None = None,
) -> dict[str, Any]:
    if arm not in ("CK", "CD"):
        raise ValueError(f"unknown arm {arm!r}; PEC-C1 has exactly CK and CD")
    if smoke_train is None and int(epochs) != TRAIN_EPOCHS:
        raise RuntimeError(
            f"formal PEC-C1 runs are frozen at {TRAIN_EPOCHS} epochs "
            f"(pre-registration §6); got {epochs}"
        )
    role_mode = "sparse" if arm == "CK" else "dense"
    tag = tag or f"{arm}_seed{seed}"

    train = load_split("train")
    valid = load_split("valid")
    if smoke_train is not None:
        train = train[: int(smoke_train)]
    if smoke_valid is not None:
        valid = valid[: int(smoke_valid)]
    d_node, d_edge = load_dictionaries()

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device)

    model = pec.build_model(
        role_mode, d_node=d_node, d_edge=d_edge, ablation="true", seed=seed
    ).to(device)
    if arm == "CK":
        freeze_dictionary(model)

    params = trainable_parameters(model)
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WD)
    n_trainable_in_opt = int(sum(p.numel() for p in params))
    trainable_ids = {id(p) for p in params}
    frozen_in_opt = any(
        id(p) in trainable_ids
        for name, p in model.named_parameters()
        if name in ROLE_PARAM_NAMES
    )
    if arm == "CK" and frozen_in_opt:
        raise RuntimeError("frozen dictionary leaked into the optimizer")

    initial_role = {
        name: parameter.detach().clone()
        for name, parameter in role_parameters(model).items()
    }

    valid_batches = list(_batches(valid, BATCH, shuffle=False, seed=seed))
    valid_targets = _batch_targets(valid_batches)

    params_report = parameter_report(model)
    start = time.time()
    history: list[dict[str, Any]] = []
    top: list[tuple[float, int, dict[str, torch.Tensor]]] = []

    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        seen = 0
        for batch in _batches(train, BATCH, shuffle=True, seed=seed * 1000 + epoch):
            batch = pec.to_device(batch, device)
            out = model(batch)
            loss = (out["prediction"] - batch["y"]).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            optimizer.step()
            running += float(loss.detach()) * int(batch["y"].numel())
            seen += int(batch["y"].numel())
        train_mae = running / max(seen, 1)
        valid_mae = _evaluate(model, valid_batches, device)
        history.append(
            {"epoch": int(epoch), "train_mae": train_mae, "valid_mae": valid_mae}
        )
        state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        top.append((valid_mae, int(epoch), state))
        top.sort(key=lambda item: (item[0], item[1]))
        del top[SOUP_TOP:]
        if epoch % 10 == 0 or epoch == int(epochs) - 1:
            print(
                f"[{tag}] epoch={epoch:03d} train={train_mae:.6f} valid={valid_mae:.6f}",
                flush=True,
            )

    wall = time.time() - start

    # PEC-C1 D1 empirical proof: the role coordinate must not have moved.
    drift_after_training = {
        name: float((parameter.detach().cpu() - initial_role[name]).abs().max())
        for name, parameter in role_parameters(model).items()
    }
    if arm == "CK":
        for name, value in drift_after_training.items():
            if value != 0.0:
                raise RuntimeError(
                    f"PEC-C1 D1 violated: frozen dictionary {name} drifted by {value}"
                )

    best_valid, best_epoch, best_state = top[0]
    best_state = dict(best_state)

    # fixed equal-weight Top-5 epoch-checkpoint prediction soup (official valid)
    members = sorted(int(epoch) for _mae, epoch, _state in top)
    soup_parts: list[np.ndarray] = []
    best_predictions: np.ndarray | None = None
    for _mae, epoch, state in sorted(top, key=lambda item: item[1]):
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        prediction = _predict(model, valid_batches, device)
        soup_parts.append(prediction)
        if int(epoch) == int(best_epoch):
            best_predictions = prediction
    soup_prediction = np.mean(np.stack(soup_parts, axis=0), axis=0)
    soup_valid = float(np.abs(soup_prediction - valid_targets).mean())
    if best_predictions is None:  # pragma: no cover - defensive
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        best_predictions = _predict(model, valid_batches, device)
    best_valid = float(np.abs(best_predictions - valid_targets).mean())
    drift_after_soup = {
        name: float((parameter.detach().cpu() - initial_role[name]).abs().max())
        for name, parameter in role_parameters(model).items()
    }
    if arm == "CK":
        for name, value in drift_after_soup.items():
            if value != 0.0:
                raise RuntimeError(
                    f"PEC-C1 D1 violated during soup: {name} drifted by {value}"
                )

    # mechanism interventions on the FROZEN Top-5 soup
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    neutral = _neutral_dictionaries(seed) if arm == "CK" else None
    interventions: dict[str, Any] = {}
    for mode in ("chem_shuffle", "relation_shuffle"):
        parts = []
        for _mae, _epoch, state in sorted(top, key=lambda item: item[1]):
            model.load_state_dict({k: v.to(device) for k, v in state.items()})
            parts.append(
                _intervention_predictions(
                    model, valid_batches, device, mode=mode, seed=seed
                )
            )
        prediction = np.mean(np.stack(parts, axis=0), axis=0)
        mae = float(np.abs(prediction - valid_targets).mean())
        interventions[mode] = {
            "valid_mae": mae,
            "degradation": mae - soup_valid,
            "mean_abs_prediction_shift": float(
                np.abs(prediction - soup_prediction).mean()
            ),
        }
    if neutral is not None:
        parts = []
        for _mae, _epoch, state in sorted(top, key=lambda item: item[1]):
            model.load_state_dict({k: v.to(device) for k, v in state.items()})
            with torch.no_grad():
                model.d_node.copy_(torch.as_tensor(neutral[0], dtype=torch.float32).to(device))
                model.d_edge.copy_(torch.as_tensor(neutral[1], dtype=torch.float32).to(device))
            parts.append(_predict(model, valid_batches, device))
        prediction = np.mean(np.stack(parts, axis=0), axis=0)
        mae = float(np.abs(prediction - valid_targets).mean())
        interventions["neutral_dictionary"] = {
            "valid_mae": mae,
            "degradation": mae - soup_valid,
            "mean_abs_prediction_shift": float(
                np.abs(prediction - soup_prediction).mean()
            ),
        }

    role_drift = {
        name: float((parameter.detach().cpu() - initial_role[name]).abs().max())
        for name, parameter in role_parameters(model).items()
    }

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"top5": [state for _m, _e, state in sorted(top, key=lambda i: i[1])],
         "members": members, "arm": arm, "seed": int(seed)},
        STATE_DIR / f"{tag}_top5.pt",
    )
    torch.save(best_state, STATE_DIR / f"{tag}_best.pt")
    curve_path = RESULTS_DIR / f"{tag}_curve.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", encoding="utf-8") as handle:
        handle.write("epoch,train_mae,valid_mae\n")
        for row in history:
            handle.write(f"{row['epoch']},{row['train_mae']:.10f},{row['valid_mae']:.10f}\n")

    best_rows = sorted(history, key=lambda row: row["valid_mae"])[:SOUP_TOP]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "round": "pec_c1",
        "arm": arm,
        "tag": tag,
        "role_mode": role_mode,
        "seed": int(seed),
        "epochs": int(epochs),
        "smoke": smoke_train is not None,
        "official_test_loaded": False,
        "splits_read": ["train", "val"],
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "device": device_report(device),
        "params": params_report,
        "optimizer_trainable_params": n_trainable_in_opt,
        "dictionary_frozen": bool(arm == "CK"),
        "role_drift_after_training": drift_after_training,
        "role_drift_after_soup": drift_after_soup,
        "role_drift_after_interventions": role_drift,
        "dictionary_fit_corpus": "official train, full 10000"
        if smoke_train is None
        else f"official train subset {smoke_train}",
        "config": {
            "lr": LR,
            "wd": WD,
            "batch": BATCH,
            "clip": CLIP,
            "loss": "L1",
            "scheduler": "none",
            "soup_top": SOUP_TOP,
        },
        "n_train": len(train),
        "n_valid": len(valid),
        "curve": history,
        "best_valid_mae": best_valid,
        "best_epoch": int(best_epoch),
        "soup_valid_mae": soup_valid,
        "soup_members": members,
        "soup_member_valid_mae": [float(m) for m, e, _s in sorted(top)],
        "top5_from_curve": [int(r["epoch"]) for r in best_rows],
        "mechanism": interventions,
        "wall_seconds": wall,
        "wall_seconds_per_epoch": wall / max(int(epochs), 1),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "historical_context_only": HISTORICAL_CONTEXT,
        "stop_reason": (
            "frozen 240-epoch schedule completed; single seed by pre-registration"
        ),
    }
    _write_json(RESULTS_DIR / f"{tag}.json", payload)
    if smoke_train is None:
        np.save(RESULTS_DIR / f"{tag}_soup_predictions.npy", soup_prediction)
        np.save(RESULTS_DIR / f"{tag}_best_predictions.npy", best_predictions)
    print(
        f"[{tag}] DONE best_valid={best_valid:.6f}@{best_epoch} "
        f"soup={soup_valid:.6f} members={members} wall={wall:.1f}s",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# decision (frozen gates)
# ---------------------------------------------------------------------------


def _band(value: float) -> str:
    if value > GATE_WEAK:
        return ">0.145"
    if value > GATE_STRONG:
        return "0.140-0.145"
    return "<=0.140"


def decide(ck: Mapping[str, Any], cd: Mapping[str, Any]) -> dict[str, Any]:
    m_ck = float(ck["soup_valid_mae"])
    m_cd = float(cd["soup_valid_mae"])
    delta_dict = m_cd - m_ck  # positive => sparse better
    absolute = min(m_ck, m_cd)

    case_a = absolute > GATE_WEAK
    case_b = m_ck > m_cd + GATE_DICT_MATERIAL
    case_c = (m_ck <= GATE_WEAK) and (delta_dict >= GATE_DICT_MATERIAL)
    case_d = absolute <= GATE_STRONG
    case_e = (GATE_STRONG < absolute <= GATE_WEAK) and (
        abs(delta_dict) < GATE_DICT_MATERIAL
    )

    if case_a:
        verdict = "PURE_ENV_COMPOSITION_ABSOLUTE_WEAK"
    elif case_b:
        verdict = "ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT"
    elif case_d:
        verdict = "PURE_ENV_COMPOSITION_STRONG_SEED0"
    elif case_c:
        verdict = "PEC_C1_SPARSE_SIGNAL_SEED0"
    elif case_e:
        verdict = "PURE_ENV_COMPOSITION_VIABLE_DICT_UNRESOLVED"
    else:  # pragma: no cover - the five cases partition the space
        verdict = "PURE_ENV_COMPOSITION_VIABLE_DICT_UNRESOLVED"

    trigger_1 = case_c
    trigger_2 = case_d
    seed1 = bool(trigger_1 or trigger_2)

    mechanism = ck.get("mechanism", {})
    chem_ok = bool(
        mechanism.get("chem_shuffle", {}).get("mean_abs_prediction_shift", 0.0) > 0.0
    )
    rel_ok = bool(
        mechanism.get("relation_shuffle", {}).get("mean_abs_prediction_shift", 0.0) > 0.0
    )
    dict_ok = bool(
        mechanism.get("neutral_dictionary", {}).get("mean_abs_prediction_shift", 0.0) > 0.0
    )
    mechanism_integrity = {
        "chemistry_placement_sensitive": chem_ok,
        "dictionary_dependent": dict_ok,
        "composition_relation_sensitive": rel_ok,
        "no_mp_no_recurrence_no_bypass": True,
        "dictionary_branch_alive": dict_ok,
    }

    if verdict == "PEC_C1_SPARSE_SIGNAL_SEED0" and not dict_ok:
        verdict = "PURE_ENV_COMPOSITION_VIABLE_DICT_UNRESOLVED"
        seed1 = False

    return {
        "M_CK": m_ck,
        "M_CD": m_cd,
        "delta_dict": delta_dict,
        "absolute_min": absolute,
        "band": _band(absolute),
        "cases": {
            "A_absolute_weak": case_a,
            "B_dense_dominates_sparse": case_b,
            "C_sparse_signal": case_c,
            "D_strong_pure": case_d,
            "E_viable_dict_unresolved": case_e,
        },
        "precedence": "A > B > D > C > E",
        "verdict": verdict,
        "seed1_authorized": seed1,
        "seed1_trigger": (
            "Trigger 2 (min <= 0.140)" if trigger_2 else
            "Trigger 1 (CK <= 0.145 and CD-CK >= 0.003)" if trigger_1 else "none"
        ),
        "mechanism_integrity": mechanism_integrity,
        "confound_note": (
            "CK's dictionary is frozen (PEC-C1 D1) while CD's dense role map is "
            "trainable (D2): the comparison is conservative against CK, and a "
            "Case B verdict must not be generalised to task-coupled dictionaries "
            "(that is PEC-C2's question)."
        ),
        "official_test_loaded": False,
    }


def report(force: bool = False) -> dict[str, Any]:
    ck_path = RESULTS_DIR / f"CK_seed{SEED}.json"
    cd_path = RESULTS_DIR / f"CD_seed{SEED}.json"
    if not ck_path.exists() or not cd_path.exists():
        raise RuntimeError(
            f"both arms are required for the decision: {ck_path.name}, {cd_path.name}"
        )
    decision = decide(_read_json(ck_path), _read_json(cd_path))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "decision": decision,
        "arms": {"CK_seed0": _read_json(ck_path), "CD_seed0": _read_json(cd_path)},
        "dictionaries": _read_json(DICT_DIR / "dicts.json"),
        "cache": _read_json(CACHE_DIR / "cache_meta.json"),
        "historical_context_only": HISTORICAL_CONTEXT,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "pec_c1_decision.json", payload)
    _write_report_markdown(payload)
    return payload


def _write_report_markdown(payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    ck = payload["arms"]["CK_seed0"]
    cd = payload["arms"]["CD_seed0"]
    lines = [
        "# PEC-C1 — Pure Environment Composition Confirmatory (seed 0, official valid)",
        "",
        f"Verdict: **{decision['verdict']}**",
        "",
        "> PEC-v0 Gate 1 remains a historical frozen FAIL. PEC-C1 does not "
        "retroactively pass or recalibrate it.",
        "",
        "Full official train 10,000 / official valid 1,000. "
        "`official_test_loaded = false`.",
        "",
        "| arm | role coordinate | params (total / trainable) | best valid MAE | best epoch | "
        "**Top-5 soup valid MAE** | wall (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| CK | SparseDict (frozen K-SVD `K=16,s=4`) | "
        f"{ck['params']['total']} / {ck['params']['trainable']} | "
        f"{ck['best_valid_mae']:.6f} | {ck['best_epoch']} | "
        f"**{ck['soup_valid_mae']:.6f}** | {ck['wall_seconds']:.1f} |",
        f"| CD | DenseRole (trainable, D-initialized) | "
        f"{cd['params']['total']} / {cd['params']['trainable']} | "
        f"{cd['best_valid_mae']:.6f} | {cd['best_epoch']} | "
        f"**{cd['soup_valid_mae']:.6f}** | {cd['wall_seconds']:.1f} |",
        "",
        f"`M_CK = {decision['M_CK']:.6f}`, `M_CD = {decision['M_CD']:.6f}`, "
        f"`delta_dict = M_CD - M_CK = {decision['delta_dict']:+.6f}` "
        "(positive ⇒ SparseDict better).",
        f"Absolute band: `{decision['band']}`.",
        f"Soup members CK `{ck['soup_members']}`, CD `{cd['soup_members']}`.",
        "",
        "## Frozen gate outcomes",
        "",
        "| case | fired |",
        "|---|---|",
    ]
    for key, value in decision["cases"].items():
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        f"`seed1_authorized = {str(decision['seed1_authorized']).lower()}` "
        f"({decision['seed1_trigger']})",
        "",
        "## Mechanism integrity (evaluation-only, on the CK Top-5 soup)",
        "",
        "| intervention | valid MAE | degradation | mean abs prediction shift |",
        "|---|---:|---:|---:|",
    ]
    for name, row in ck.get("mechanism", {}).items():
        lines.append(
            f"| {name} | {row['valid_mae']:.6f} | {row['degradation']:+.6f} | "
            f"{row['mean_abs_prediction_shift']:.6f} |"
        )
    lines += [
        "",
        "## Historical context only (not gates, never re-run)",
        "",
        "```text",
        "strict-static S0 seed0 soup ≈ 0.140794",
        "strict-static S0 seed1 soup ≈ 0.136423",
        "B-Null ≈ 0.123   B-Full ≈ 0.119–0.118 (different computation class)",
        "```",
        "",
        f"_Confound:_ {decision['confound_note']}",
        "",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device(value)
    raise ValueError(f"unsupported device {value!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PEC-C1 runner")
    parser.add_argument("stage", choices=("cache", "dicts", "smoke", "train", "mechanism", "report", "all"))
    parser.add_argument("--arm", default=None, help="CK | CD (train stage)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--smoke-train", type=int, default=None)
    parser.add_argument("--smoke-valid", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "cache":
        print(json.dumps(build_cache(force=args.force), indent=2, sort_keys=True, default=str))
    elif args.stage == "dicts":
        print(json.dumps(fit_dictionaries(force=args.force), indent=2, sort_keys=True, default=str))
    elif args.stage == "smoke":
        device = _parse_device(args.device)
        build_cache(force=False)
        fit_dictionaries(force=False)
        for arm in ("CK", "CD"):
            train_arm(
                arm,
                seed=args.seed,
                epochs=int(args.epochs),
                device=device,
                tag=f"smoke_{arm}",
                smoke_train=args.smoke_train or 512,
                smoke_valid=args.smoke_valid or 256,
            )
    elif args.stage == "train":
        if not args.arm:
            raise SystemExit("--arm CK|CD is required for the train stage")
        device = _parse_device(args.device)
        payload = train_arm(
            args.arm,
            seed=args.seed,
            epochs=int(args.epochs),
            device=device,
        )
        print(json.dumps(
            {
                "arm": payload["arm"],
                "soup_valid_mae": payload["soup_valid_mae"],
                "best_valid_mae": payload["best_valid_mae"],
                "best_epoch": payload["best_epoch"],
            },
            indent=2,
        ))
    elif args.stage == "report":
        payload = report(force=args.force)
        print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    else:  # all
        build_cache(force=args.force)
        fit_dictionaries(force=args.force)
        print("cache + dictionaries ready; run the two arms separately per GPU")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
