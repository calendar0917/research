"""Optimized-manifold broad frozen-state sufficiency screen (compact-v4-hinge).

This is a **cheap screening gate**, not a new architecture, not an OOF
training run and not a re-run of the historical endpoint / covariance / triad
witness tree.  It asks exactly one question:

    On the already fully-trained *optimized* compact-v4-hinge latent manifold,
    do the frozen node/patch/pair states ``{h'_i, q_ij}`` still carry a
    low-complexity increment of predictive signal beyond the final 302D graph
    vector ``R``?

The historical broad-readout audit was run on the *under-trained* 60-epoch
manifold and returned a clear NO-GO (a parameter-matched R-only residual head
matched or beat a set/state adapter).  Because the optimized manifold drifts
materially from the historical one, the historical NO-GO is a valid statement
about the *old* manifold but cannot be upgraded to the optimized manifold for
free.  This module re-screens the broad question with **zero new backbone
training**: it reuses the two existing optimized compact-v4-hinge checkpoints
(``Pstar_A2_long_seed{0,1}``) whose training cost was already paid.

Reuse policy (mandate section 9/10):
* the B0 definition, the parameter-matched R-only adapter (B1), the historical
  permutation-invariant Set/State adapter (E), the shared preprocessing, the
  L1/Adam/full-batch/400-epoch protocol and the deterministic adapter init are
  imported *unchanged* from ``zinc_frozen_readout_sufficiency``;
* only the backbone manifold and the evaluation protocol change.

Measurement discipline (mandate sections 20/21):
* adapters are trained on an official-train-only deterministic hash split
  (7200 adapter-fit / 800 adapter-selection / 2000 train-probe);
* official valid is touched exactly once, after the adapter checkpoint is
  locked, and only for B0/B1/E evaluation -- never for training / selection /
  threshold / architecture tuning;
* because the optimized backbone checkpoint was itself valid-selected, the
  valid result is a **canonical-checkpoint screening diagnostic**, not clean
  OOF mechanism evidence;
* official test is never loaded.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_optimized_manifold_broad_state_screen <stage>

Stages: ``inventory export gate0 splits stage1 stage1b decision figures all``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16 import zinc_frozen_readout_sufficiency as frs
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    TYPED_TOKENIZER_V1_HISTORICAL,
    typed_tokenizer_fingerprint,
)
from tracks.ksvd.experiments.luyin16.zinc_information_gap_audit import _train_counter
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc
from tracks.ksvd.experiments.luyin16.zinc_oof_difficulty_audit import (
    _model_config_with_clamps,
)
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
    _build_v4_model,
    _extract_v4_records,
)

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/optimized_manifold_broad_state_screen"
EXPORT_DIR = RESULTS_DIR / "state_exports"
FIG_DIR = RESULTS_DIR / "figures"

CONFIG_PATH = TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
CONFIG_SHA256 = "b9205b05421589967018a85a3118ef96e218796dc04a57fbab49d0f83886ae63"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

# --- canonical optimized checkpoints (training cost already paid) ----------
OPTIMIZED_MODEL_COMMIT = "586542e1e03b8c4517a20b1ac24520f4e7a3ffed"
STATE_PATHS = {
    0: TRACK_ROOT
    / "results/compact_v4_training_sufficiency/states/Pstar_A2_long_seed0_selection_state.pt",
    1: TRACK_ROOT
    / "results/compact_v4_training_sufficiency/states/Pstar_A2_long_seed1_selection_state.pt",
}
EXPECTED_VALID = {
    0: 0.14642022556537995,
    1: 0.1493322635096847,
}
EXPECTED_BEST_EPOCH = {0: 169, 1: 104}
EXPECTED_PARAMS = 99613
BACKBONE_SEEDS = (0, 1)
TOKENIZER_VERSION = TYPED_TOKENIZER_V1_HISTORICAL

OPTIMIZED_PROTOCOL: dict[str, Any] = {
    "optimizer": "Adam",
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "batch_size": 128,
    "max_epochs": 240,
    "patience": 40,
    "scheduler": "none",
    "loss": "L1 / mean absolute error",
    "checkpoint_selection": "best official-valid MAE",
    "single_stage": True,
    "gradient_clip_norm": 5.0,
}

# --- export ----------------------------------------------------------------
EXPORT_VERSION = "optimized_frozen_state_export_v1"
LEGACY_EXPORT_VERSIONS = {
    "head_input_v1",
    "frozen_state_export_v1",
    "frozen_state_export_v2_corrected",
    "frozen_state_export_v3_pair_endpoint",
    "frozen_state_export_v4_centre_incidence",
}

R_DIM = frs.R_DIM  # 302
PATCH_DIM = frs.PATCH_DIM  # 48
PAIR_DIM = frs.PAIR_DIM  # 16
N_BUCKETS = frs.N_BUCKETS  # 5

# --- official-train adapter split ------------------------------------------
SPLIT_SEED = "optimized-manifold-broad-state-screen-v1-20260919"
SPLIT_SIZES = (7200, 800, 2000)  # adapter-fit / adapter-selection / train-probe
SPLIT_ROLES = ("adapter_fit", "adapter_selection", "train_probe")
N_TRAIN = 10000
N_VALID = 1000

# --- adapter protocol (inherited from the historical corrected audit) ------
PRIMARY_ADAPTER_SEED = 0
SECOND_ADAPTER_SEED = 1

# --- decision thresholds (mandate sections 31/33/35/37/38) -----------------
ADVANCE_PER_SEED = 0.002
ADVANCE_MEAN = 0.003
CLEAR_NOGO_MEAN = 0.0005
CLEAR_NOGO_SINGLE = -0.0015
BULK_MAX_DEGRADATION = 0.002
BULK_RARE_PERCENTILE = 80.0
STATE_DEPENDENCY_MIN = 1.0e-3

# --- gate 0 tolerances -----------------------------------------------------
R_RECON_ATOL = 1.0e-4
PRED_RECON_ATOL = 1.0e-5
BATCH_INV_ATOL = 1.0e-4
PERM_INV_ATOL = 1.0e-4
VALID_ANCHOR_ATOL = 1.0e-5
B0_RECON_ATOL = 1.0e-5


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json_any(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_json(value: Any) -> str:
    blob = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    head = REPO_ROOT / ".git/HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = REPO_ROOT / ".git" / text.split(" ", 1)[1].strip()
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()
        return text
    except OSError:  # pragma: no cover
        return "unknown"


def _frozen_config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# in-process encoded-data / model cache (encoding is the expensive part)
# ---------------------------------------------------------------------------

_ENCODED_CACHE: dict[str, Any] = {}


def _build_encoded() -> tuple[list[Any], list[Any], dict[str, Any], list[Any], list[Any], dict[str, Any]]:
    if "encoded" not in _ENCODED_CACHE:
        started = time.perf_counter()
        train_records, valid_records = _extract_v4_records()
        config = _frozen_config()
        torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", 4)))
        encoded_train, encoded_valid, audit = zpp._phase_data(
            train_records, valid_records, config=config
        )
        _ENCODED_CACHE["encoded"] = (
            train_records,
            valid_records,
            config,
            encoded_train,
            encoded_valid,
            audit,
        )
        print(f"[encoded] train={len(train_records)} valid={len(valid_records)} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    return _ENCODED_CACHE["encoded"]


def _load_optimized_model(seed: int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Build the frozen v4-hinge model, load the optimized checkpoint."""
    (
        _train_records,
        _valid_records,
        config,
        _encoded_train,
        _encoded_valid,
        audit,
    ) = _build_encoded()
    config_mod = dict(config)
    config_mod["model"] = _model_config_with_clamps(config, audit)
    model = _build_v4_model(config_mod, audit)
    state_path = STATE_PATHS[int(seed)]
    if not state_path.exists():
        raise FileNotFoundError(f"optimized checkpoint missing: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = int(sum(p.numel() for p in model.parameters()))
    if n_params != EXPECTED_PARAMS:
        raise AssertionError(
            f"seed {seed}: model parameters {n_params} != expected {EXPECTED_PARAMS}"
        )
    if int(model.head[0].in_features) != R_DIM:
        raise AssertionError(
            f"seed {seed}: pre-head R width {model.head[0].in_features} != {R_DIM}"
        )
    return model, config, audit


# ---------------------------------------------------------------------------
# export fingerprints
# ---------------------------------------------------------------------------

def _vocabulary_fingerprint(train_records: Sequence[Any], config: Mapping[str, Any]) -> dict[str, Any]:
    representation = config["representation"]
    typed = zpp._fit_vocabulary(
        train_records,
        "typed_certificate",
        int(representation.get("max_typed_tokens", 8192)),
        int(representation.get("minimum_typed_frequency", 1)),
    )
    parent = zpp._fit_vocabulary(
        train_records,
        "parent_certificate",
        int(representation.get("max_parent_tokens", 2048)),
        int(representation.get("minimum_parent_frequency", 1)),
    )
    return {
        "typed_vocabulary_size": int(len(typed)),
        "typed_vocabulary_size_with_oov": int(len(typed) + 1),
        "parent_vocabulary_size": int(len(parent)),
        "parent_vocabulary_size_with_oov": int(len(parent) + 1),
        "typed_sha256": hashlib.sha256(
            b"\n".join(k for k, _ in sorted(typed.items(), key=lambda kv: kv[1]))
        ).hexdigest(),
        "parent_sha256": hashlib.sha256(
            b"\n".join(k for k, _ in sorted(parent.items(), key=lambda kv: kv[1]))
        ).hexdigest(),
    }


def _topology_fingerprint(config: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    topology = audit.get("topology", {})
    return {
        "mode": str(config["model"].get("topology_mode", "none")),
        "input_width": int(topology.get("input_width", 0)),
        "feature_names": ztopo.feature_names("hinge"),
        "standardizer_mean_sha256": _hash_json(topology.get("mean")),
        "standardizer_scale_sha256": _hash_json(topology.get("scale")),
    }


def _checkpoint_fingerprint(seed: int) -> dict[str, Any]:
    path = STATE_PATHS[int(seed)]
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _split_fingerprint(split: str, roles: Mapping[str, np.ndarray] | None) -> dict[str, Any]:
    if split == "valid":
        return {"scope": "official_valid", "n_valid": N_VALID}
    assert roles is not None
    return {
        "scope": "official_train",
        "split_seed": SPLIT_SEED,
        "sizes": dict(zip(SPLIT_ROLES, SPLIT_SIZES)),
        "assignment_sha256": hashlib.sha256(
            "".join(str(r) for r in roles.tolist()).encode()
        ).hexdigest(),
    }


def _export_fingerprint(split: str, seed: int, config: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    train_records, _valid_records = _extract_v4_records()
    roles = _role_assignment() if split == "train" else None
    payload = {
        "export_version": EXPORT_VERSION,
        "model_commit": OPTIMIZED_MODEL_COMMIT,
        "repo_commit": _git_commit(),
        "optimized_protocol_lock_hash": _hash_json(OPTIMIZED_PROTOCOL),
        "backbone_seed": int(seed),
        "checkpoint": _checkpoint_fingerprint(seed),
        "config_sha256": _sha256_file(CONFIG_PATH),
        "tokenizer_version": TOKENIZER_VERSION,
        "tokenizer_fingerprint": typed_tokenizer_fingerprint(TOKENIZER_VERSION, zpp.PATCH_RADIUS),
        "vocabulary_fingerprint": _vocabulary_fingerprint(train_records, config),
        "topology_fingerprint": _topology_fingerprint(config, audit),
        "split_fingerprint": _split_fingerprint(split, roles),
        "expected_valid_mae": EXPECTED_VALID[int(seed)],
        "expected_best_epoch": EXPECTED_BEST_EPOCH[int(seed)],
        "expected_parameters": EXPECTED_PARAMS,
        "patch_dim": PATCH_DIM,
        "pair_dim": PAIR_DIM,
        "r_dim": R_DIM,
        "n_buckets": N_BUCKETS,
    }
    payload["fingerprint"] = _hash_json(payload)
    return payload


# ---------------------------------------------------------------------------
# Stage 0: export
# ---------------------------------------------------------------------------

def _export_path(split: str, seed: int) -> Path:
    return EXPORT_DIR / f"{EXPORT_VERSION}_{split}_seed{seed}.npz"


def _direct_forward_predictions(model: Any, graphs: Sequence[Any], batch_size: int = 64) -> np.ndarray:
    """Independent (hook-free) forward, natural order, smaller batch."""
    from torch_geometric.loader import DataLoader

    loader = DataLoader(list(graphs), batch_size=int(batch_size), shuffle=False)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def _build_export(split: str, seed: int, batch_size: int = 128) -> dict[str, Any]:
    (
        train_records,
        valid_records,
        config,
        encoded_train,
        encoded_valid,
        audit,
    ) = _build_encoded()
    model, _config, _audit = _load_optimized_model(seed)

    if split == "train":
        records = train_records
        graphs = encoded_train
    elif split == "valid":
        records = valid_records
        graphs = encoded_valid
    else:  # pragma: no cover
        raise ValueError(f"unknown split {split!r}")

    capture = frs._capture_states(model, graphs, batch_size=batch_size)
    target = np.asarray([float(record.y) for record in records], dtype=np.float64)
    subset_index = np.arange(len(records), dtype=np.int64)
    reference = _direct_forward_predictions(model, graphs, batch_size=64)
    b0_recon = float(np.abs(reference - capture["yhat_0"]).max())

    fingerprint = _export_fingerprint(split, seed, config, audit)
    fingerprint["b0_reference_max_diff"] = b0_recon
    if split == "valid":
        anchor = float(np.abs(np.abs(target - capture["yhat_0"]).mean() - EXPECTED_VALID[int(seed)]))
        fingerprint["valid_anchor_abs_error"] = anchor
    export = {
        "subset_index": subset_index,
        "target": target,
        "reference_yhat_0": reference.astype(np.float64),
        "fingerprint": fingerprint,
        **capture,
    }
    return export


def _save_export(path: Path, export: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, value in export.items():
        if key == "fingerprint":
            arrays["fingerprint_json"] = np.asarray(
                json.dumps(_jsonable(value), sort_keys=True)
            )
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_export(path: Path) -> dict[str, Any]:
    """Load an optimized-manifold export, refusing any legacy / foreign cache."""
    with np.load(path, allow_pickle=False) as data:
        if "fingerprint_json" not in data.files:
            raise RuntimeError(f"cache {path} has no fingerprint; refusing to load")
        fingerprint = json.loads(str(data["fingerprint_json"]))
        if fingerprint.get("export_version") != EXPORT_VERSION:
            raise RuntimeError(
                f"cache {path} export_version={fingerprint.get('export_version')!r} "
                f"!= required {EXPORT_VERSION!r}; refusing to load"
            )
        export: dict[str, Any] = {"fingerprint": fingerprint}
        for key in data.files:
            if key == "fingerprint_json":
                continue
            export[key] = data[key]
    return export


def checkpoint_inventory() -> dict[str, Any]:
    entries = []
    for seed in BACKBONE_SEEDS:
        path = STATE_PATHS[int(seed)]
        entry: dict[str, Any] = {
            "seed": int(seed),
            "status": "present" if path.exists() else "missing",
            "state_path": str(path),
            "protocol": dict(OPTIMIZED_PROTOCOL),
            "expected_valid_mae": EXPECTED_VALID[int(seed)],
            "expected_best_epoch": EXPECTED_BEST_EPOCH[int(seed)],
            "parameters_expected": EXPECTED_PARAMS,
            "tokenizer_version": TOKENIZER_VERSION,
            "model_commit": OPTIMIZED_MODEL_COMMIT,
        }
        if path.exists():
            entry["sha256"] = _sha256_file(path)
        entries.append(entry)
    return {
        "protocol_version": EXPORT_VERSION,
        "note": (
            "Reuses the two existing optimized compact-v4-hinge checkpoints; "
            "no backbone is trained in this stage."
        ),
        "backbone_seeds": list(BACKBONE_SEEDS),
        "config_path": str(CONFIG_PATH),
        "config_sha256": _sha256_file(CONFIG_PATH),
        "entries": entries,
    }


def stage_export(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    built: dict[str, Any] = {}
    for seed in BACKBONE_SEEDS:
        for split in ("train", "valid"):
            path = _export_path(split, seed)
            if path.exists() and not force:
                built[f"{split}_seed{seed}"] = {"path": str(path), "cached": True}
                continue
            t0 = time.perf_counter()
            export = _build_export(split, seed)
            _save_export(path, export)
            built[f"{split}_seed{seed}"] = {
                "path": str(path),
                "cached": False,
                "seconds": time.perf_counter() - t0,
                "n": int(len(export["subset_index"])),
                "n_patches_total": int(export["n_patches"].sum()),
                "n_pairs_total": int(export["n_pairs"].sum()),
                "b0_reference_max_diff": float(
                    export["fingerprint"]["b0_reference_max_diff"]
                ),
                "valid_anchor_abs_error": export["fingerprint"].get(
                    "valid_anchor_abs_error"
                ),
            }
            print(
                f"[export] split={split} seed={seed} n={len(export['subset_index'])} "
                f"b0diff={export['fingerprint']['b0_reference_max_diff']:.2e} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    summary = {
        "export_version": EXPORT_VERSION,
        "seeds": list(BACKBONE_SEEDS),
        "exports": built,
        "checkpoint_inventory": checkpoint_inventory(),
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", summary["checkpoint_inventory"])
    fingerprints = {
        key: json.loads(str(np.load(info["path"], allow_pickle=False)["fingerprint_json"]))
        for key, info in built.items()
    }
    _write_json_any(RESULTS_DIR / "export_fingerprints.json", fingerprints)
    _write_json_any(RESULTS_DIR / "stage_export.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Gate 0 (export integrity)
# ---------------------------------------------------------------------------

def _gate_valid_anchor(export: Mapping[str, Any], seed: int) -> dict[str, Any]:
    anchor = float(export["fingerprint"].get("valid_anchor_abs_error", float("nan")))
    return {
        "gate": "G0.3b_valid_prediction_anchor",
        "seed": int(seed),
        "expected_valid_mae": EXPECTED_VALID[int(seed)],
        "anchor_abs_error": anchor,
        "atol": VALID_ANCHOR_ATOL,
        "passed": bool(np.isfinite(anchor) and anchor <= VALID_ANCHOR_ATOL),
    }


def _gate_b0_reference(export: Mapping[str, Any]) -> dict[str, Any]:
    diff = float(np.abs(export["reference_yhat_0"] - export["yhat_0"]).max())
    return {
        "gate": "G0.3a_stored_exported_yhat_consistency",
        "max_abs_diff": diff,
        "atol": B0_RECON_ATOL,
        "passed": bool(diff <= B0_RECON_ATOL),
    }


def _gate_pair_grouping(export: Mapping[str, Any]) -> dict[str, Any]:
    return frs._gate_pair_grouping(export)


def _gate_pair_endpoint_identity(export: Mapping[str, Any]) -> dict[str, Any]:
    return frs._gate_pair_endpoint_identity(export)


def _gate_pair_coverage(export: Mapping[str, Any]) -> dict[str, Any]:
    return frs._gate_pair_coverage(export)


def _gate_reconstruct_R(export: Mapping[str, Any]) -> dict[str, Any]:
    return frs._gate_reconstruct_R(export)


def _gate_reconstruct_prediction(export: Mapping[str, Any], model: Any) -> dict[str, Any]:
    return frs._gate_reconstruct_prediction(export, model)


def _gate_node_permutation(seed: int, n_molecules: int = 4) -> dict[str, Any]:
    """Relabel nodes on a few valid molecules; R and prediction must not move."""
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import global_feature_views
    from torch_geometric.data import Data as PyGData

    (
        _train_records,
        valid_records,
        config,
        _encoded_train,
        _encoded_valid,
        _audit,
    ) = _build_encoded()
    model, _config, _audit = _load_optimized_model(seed)
    dataset = _load_zinc(ZINC_ROOT, "val")
    topology_matrix = ztopo.matrices_for_split("val", dataset, "hinge")[0]

    # transforms fitted on all official train (matches the optimized model)
    train_records = _train_records
    representation = config["representation"]
    transforms = {
        "typed_vocabulary": zpp._fit_vocabulary(
            train_records, "typed_certificate",
            int(representation.get("max_typed_tokens", 8192)),
            int(representation.get("minimum_typed_frequency", 1)),
        ),
        "parent_vocabulary": zpp._fit_vocabulary(
            train_records, "parent_certificate",
            int(representation.get("max_parent_tokens", 2048)),
            int(representation.get("minimum_parent_frequency", 1)),
        ),
        "patch_standardizer": zpp.Standardizer.fit(zpp._patch_matrix(train_records)),
        "context_standardizer": zpp.Standardizer.fit(zpp._context_matrix(train_records)),
        "topology_standardizer": zpp.Standardizer.fit(
            np.stack([r.topology_features for r in train_records], axis=0).astype(np.float32)
        ),
    }

    rc = np.random.RandomState(0)
    chosen = sorted(
        np.random.RandomState(7).choice(len(valid_records), size=n_molecules, replace=False).tolist()
    )
    diffs = {"R": 0.0, "yhat_0": 0.0, "summary": 0.0}
    for position in chosen:
        original = valid_records[position]
        data = dataset[position]
        n = int(data.num_nodes)
        perm = rc.permutation(n)
        inv = np.argsort(perm)
        edge_index = data.edge_index.detach().cpu().numpy()
        permuted_data = PyGData(
            x=data.x[torch.tensor(inv, dtype=torch.long)],
            edge_index=torch.tensor(perm[edge_index], dtype=torch.long),
            edge_attr=data.edge_attr,
            y=data.y,
            num_nodes=n,
        )
        global_context = global_feature_views([permuted_data])["global_all"][0]
        permuted_record = zpp._graph_record(
            permuted_data,
            global_context,
            {},
            patch_radius=zpp.PATCH_RADIUS,
            context_radius=0,
            structural_mode="none",
            topology_features=topology_matrix[position],
            tokenizer_version=TOKENIZER_VERSION,
            attribute_mode="none",
        )
        encoded_original = zpp._encode_records(
            [original], transforms["typed_vocabulary"], transforms["parent_vocabulary"],
            transforms["patch_standardizer"], transforms["context_standardizer"], None,
            structural_mode="none", structural_vocabulary=None, topology_mode="hinge",
            topology_standardizer=transforms["topology_standardizer"], attribute_mode="none",
        )[0]
        encoded_permuted = zpp._encode_records(
            [permuted_record], transforms["typed_vocabulary"], transforms["parent_vocabulary"],
            transforms["patch_standardizer"], transforms["context_standardizer"], None,
            structural_mode="none", structural_vocabulary=None, topology_mode="hinge",
            topology_standardizer=transforms["topology_standardizer"], attribute_mode="none",
        )[0]
        cap_o = frs._capture_states(model, [encoded_original], batch_size=1)
        cap_p = frs._capture_states(model, [encoded_permuted], batch_size=1)
        diffs["R"] = max(diffs["R"], float(np.abs(cap_o["R"] - cap_p["R"]).max()))
        diffs["yhat_0"] = max(diffs["yhat_0"], float(np.abs(cap_o["yhat_0"] - cap_p["yhat_0"]).max()))
        diffs["summary"] = max(
            diffs["summary"],
            float(np.abs(cap_o["patch_states"].mean(axis=0) - cap_p["patch_states"].mean(axis=0)).max()),
        )
    return {
        "gate": "G0.8_graph_node_pair_ordering_invariance",
        "seed": int(seed),
        "n_molecules": n_molecules,
        "max_abs_diff": diffs,
        "atol": PERM_INV_ATOL,
        "passed": bool(max(diffs.values()) <= PERM_INV_ATOL),
    }


def _gate_batch_invariance(seed: int) -> dict[str, Any]:
    (_train_records, _valid_records, _config, _encoded_train, encoded_valid, _audit) = _build_encoded()
    model, _config2, _audit2 = _load_optimized_model(seed)
    baseline = frs._capture_states(model, encoded_valid, batch_size=128)
    order = np.arange(len(encoded_valid))[::-1].copy()
    reordered = [encoded_valid[int(i)] for i in order]
    shifted = frs._capture_states(model, reordered, batch_size=97)
    inv_order = np.argsort(order)
    diffs = {
        "R": float(np.abs(baseline["R"] - shifted["R"][inv_order]).max()),
        "yhat_0": float(np.abs(baseline["yhat_0"] - shifted["yhat_0"][inv_order]).max()),
    }
    return {
        "gate": "G0.6_batch_ordering_invariance",
        "seed": int(seed),
        "batch_sizes": [128, 97],
        "max_abs_diff": diffs,
        "atol": BATCH_INV_ATOL,
        "passed": bool(max(diffs.values()) <= BATCH_INV_ATOL),
    }


def run_gate0(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = RESULTS_DIR / "export_integrity.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    per_export: dict[str, Any] = {}
    for seed in BACKBONE_SEEDS:
        model, _config, _audit = _load_optimized_model(seed)
        train_export = load_export(_export_path("train", seed))
        valid_export = load_export(_export_path("valid", seed))
        gates: dict[str, Any] = {
            "G0.1_R_dim": {
                "gate": "G0.1_R_dimension",
                "seed": int(seed),
                "R_dim": int(train_export["R"].shape[1]),
                "expected": R_DIM,
                "passed": bool(train_export["R"].shape[1] == R_DIM and valid_export["R"].shape[1] == R_DIM),
            },
            "G0.2_fingerprint": {
                "gate": "G0.2_checkpoint_config_tokenizer_fingerprint",
                "seed": int(seed),
                "config_sha256": valid_export["fingerprint"]["config_sha256"],
                "config_matches": valid_export["fingerprint"]["config_sha256"] == CONFIG_SHA256,
                "tokenizer_fingerprint": valid_export["fingerprint"]["tokenizer_fingerprint"],
                "checkpoint_sha256": valid_export["fingerprint"]["checkpoint"]["sha256"],
                "protocol_lock": valid_export["fingerprint"]["optimized_protocol_lock_hash"],
                "passed": bool(
                    valid_export["fingerprint"]["config_sha256"] == CONFIG_SHA256
                    and valid_export["fingerprint"]["checkpoint"]["sha256"]
                    == _sha256_file(STATE_PATHS[int(seed)])
                ),
            },
            "G0.3a_b0_reference_train": _gate_b0_reference(train_export),
            "G0.3a_b0_reference_valid": _gate_b0_reference(valid_export),
            "G0.3b_valid_anchor": _gate_valid_anchor(valid_export, seed),
            "G0.4_reconstruct_R_valid": _gate_reconstruct_R(valid_export),
            "G0.5_reconstruct_prediction_valid": _gate_reconstruct_prediction(valid_export, model),
            "G0.6_pair_graph_grouping_valid": _gate_pair_grouping(valid_export),
            "G0.7_hprime_graph_grouping_train": _gate_pair_grouping(train_export),
            "G0.9_pair_endpoint_identity_valid": _gate_pair_endpoint_identity(valid_export),
            "G0.10_pair_endpoint_identity_train": _gate_pair_endpoint_identity(train_export),
            "G0.11_pair_coverage_valid": _gate_pair_coverage(valid_export),
            "G0.12_pair_coverage_train": _gate_pair_coverage(train_export),
        }
        per_export[f"seed{seed}"] = gates
        print(
            f"[gate0] seed={seed} "
            + " ".join(f"{k}={'PASS' if v['passed'] else 'FAIL'}" for k, v in gates.items()),
            flush=True,
        )

    invariances = {
        "batch_invariance_seed0": _gate_batch_invariance(0),
        "node_permutation_seed0": _gate_node_permutation(0),
    }

    all_gates: list[dict[str, Any]] = []
    for gates in per_export.values():
        all_gates.extend(gates.values())
    all_gates.extend(invariances.values())
    failures = [g["gate"] for g in all_gates if not g.get("passed", False)]
    report = {
        "export_version": EXPORT_VERSION,
        "per_export": per_export,
        "invariances": invariances,
        "n_gates": len(all_gates),
        "failures": failures,
        "all_passed": len(failures) == 0,
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_path, report)
    print(f"[gate0] all_passed={report['all_passed']} failures={failures}", flush=True)
    return report


# ---------------------------------------------------------------------------
# official-train deterministic 7200/800/2000 hash split
# ---------------------------------------------------------------------------

def _role_assignment() -> np.ndarray:
    keys = np.asarray(
        [
            int(hashlib.sha256(f"{SPLIT_SEED}|train:{mid:04d}".encode()).hexdigest()[:16], 16)
            for mid in range(N_TRAIN)
        ],
        dtype=np.float64,
    )
    order = np.argsort(keys, kind="stable")
    roles = np.empty(N_TRAIN, dtype=object)
    cursor = 0
    for label, size in zip(SPLIT_ROLES, SPLIT_SIZES):
        roles[order[cursor: cursor + size]] = label
        cursor += size
    return roles


def stage_splits(force: bool = False) -> dict[str, Any]:
    roles = _role_assignment()
    counts = {label: int((roles == label).sum()) for label in SPLIT_ROLES}
    manifest = {
        "split_seed": SPLIT_SEED,
        "source": "official train only; target-independent deterministic molecule-id hash",
        "sizes": dict(zip(SPLIT_ROLES, SPLIT_SIZES)),
        "counts": counts,
        "assignment_sha256": hashlib.sha256("".join(roles.tolist()).encode()).hexdigest(),
        "roles": {
            label: np.flatnonzero(roles == label).astype(np.int64).tolist()
            for label in SPLIT_ROLES
        },
    }
    _write_json_any(RESULTS_DIR / "split_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# adapter protocol lock
# ---------------------------------------------------------------------------

def adapter_protocol_lock() -> dict[str, Any]:
    ronly = frs._count_parameters(frs._make_ronly(PRIMARY_ADAPTER_SEED))
    setter = frs._count_parameters(frs._make_set(PRIMARY_ADAPTER_SEED))
    mismatch = abs(ronly - setter) / ronly
    lock = {
        "export_version": EXPORT_VERSION,
        "backbone_seeds": list(BACKBONE_SEEDS),
        "primary_adapter_seed": PRIMARY_ADAPTER_SEED,
        "second_adapter_seed": SECOND_ADAPTER_SEED,
        "optimizer": "Adam",
        "learning_rate": frs.ADAPTER_LR,
        "weight_decay": 0.0,
        "objective": "L1 / MAE",
        "batch": "full (all adapter-fit molecules)",
        "epochs": frs.ADAPTER_EPOCHS,
        "patience": frs.ADAPTER_PATIENCE,
        "checkpoint_selection": "best adapter-selection MAE",
        "standardisation": "fit-only (adapter-fit molecule statistics)",
        "reader_definitions": {
            "B1_R_only": "R -> 13 -> 13 -> 1; yhat = yhat_0 + rho_R(R)",
            "E_state": (
                "R + [mean_i phi_h(h'_i); mean_{bucket b} phi_q(q_ij)] -> 8 -> 1; "
                "yhat = yhat_0 + rho([R; s])"
            ),
        },
        "set_reader_components": {
            "phi_h": "48 -> 16 -> 8",
            "phi_q": "16 -> 16 -> 8",
            "rho": f"{R_DIM + 8 + N_BUCKETS * 8} -> 8 -> 1",
        },
        "ronly_params": ronly,
        "state_params": setter,
        "parameter_mismatch": mismatch,
        "parameter_mismatch_within_2pct": bool(mismatch <= 0.02),
        "adapter_fit": SPLIT_SIZES[0],
        "adapter_selection": SPLIT_SIZES[1],
        "train_probe": SPLIT_SIZES[2],
        "primary_metric": "official-valid MAE",
        "secondary_metric": "train-probe MAE (descriptive only; backbone saw all train labels)",
        "bulk_gate": BULK_MAX_DEGRADATION,
        "advance_per_seed": ADVANCE_PER_SEED,
        "advance_mean": ADVANCE_MEAN,
    }
    _write_json_any(RESULTS_DIR / "adapter_protocol_lock.json", lock)
    return lock


# ---------------------------------------------------------------------------
# common-input bulk (target-independent)
# ---------------------------------------------------------------------------

def _rare_ratio_rows(records: Sequence[Any], counter: Any) -> np.ndarray:
    rows = []
    for record in records:
        freqs = np.asarray(
            [counter.get(p.typed_certificate, 0) for p in record.patches], dtype=np.float64
        )
        n = float(len(freqs))
        rows.append(float(np.sum(freqs <= 5)) / max(n, 1.0))
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Stage 1: adapter training / evaluation
# ---------------------------------------------------------------------------

def _all_positions(states: frs.FoldStates) -> np.ndarray:
    return np.arange(len(states.subset_index), dtype=np.int64)


def _run_seed(
    seed: int,
    adapter_seed: int,
    train_export: Mapping[str, Any],
    valid_export: Mapping[str, Any],
    manifest: Mapping[str, Any],
    train_rare: np.ndarray,
    valid_rare: np.ndarray,
) -> dict[str, Any]:
    roles = np.empty(N_TRAIN, dtype=object)
    for label in SPLIT_ROLES:
        roles[np.asarray(manifest["roles"][label], dtype=np.int64)] = label
    fit_positions = np.flatnonzero(roles == "adapter_fit").astype(np.int64)
    sel_positions = np.flatnonzero(roles == "adapter_selection").astype(np.int64)
    probe_positions = np.flatnonzero(roles == "train_probe").astype(np.int64)

    train_states = frs.FoldStates(train_export)
    valid_states = frs.FoldStates(valid_export)
    fit = train_states.tensors(fit_positions)
    selection = train_states.tensors(sel_positions)
    probe = train_states.tensors(probe_positions)
    valid = valid_states.tensors(_all_positions(valid_states))

    standardizers = frs._standardize_all(fit, [selection, probe, valid])

    # --- B1 (R-only control) ---
    ronly = frs._make_ronly(adapter_seed)
    ronly.set_standardizer(*standardizers["R"])
    ronly_train = frs._train_adapter(ronly, fit, selection, adapter_seed)

    # --- E (broad frozen-state reader) ---
    state = frs._make_set(adapter_seed)
    state.set_standardizers(standardizers["R"], standardizers["patch"], standardizers["pair"])
    state_train = frs._train_adapter(state, fit, selection, adapter_seed)

    with torch.no_grad():
        b1_valid = ronly(valid).numpy().astype(np.float64)
        e_valid = state(valid).numpy().astype(np.float64)
        b1_probe = ronly(probe).numpy().astype(np.float64)
        e_probe = state(probe).numpy().astype(np.float64)

    y_valid = valid["y"].numpy().astype(np.float64)
    yhat0_valid = valid["yhat_0"].numpy().astype(np.float64)
    y_probe = probe["y"].numpy().astype(np.float64)
    yhat0_probe = probe["yhat_0"].numpy().astype(np.float64)

    mae_b0_v = float(np.abs(y_valid - yhat0_valid).mean())
    mae_b1_v = float(np.abs(y_valid - b1_valid).mean())
    mae_e_v = float(np.abs(y_valid - e_valid).mean())
    mae_b0_p = float(np.abs(y_probe - yhat0_probe).mean())
    mae_b1_p = float(np.abs(y_probe - b1_probe).mean())
    mae_e_p = float(np.abs(y_probe - e_probe).mean())

    # common-input bulk: target-independent rare-patch prevalence threshold
    threshold = float(np.percentile(train_rare[fit_positions], BULK_RARE_PERCENTILE))
    bulk_mask = valid_rare < threshold
    if int(bulk_mask.sum()) >= 10:
        bulk_b1 = float(np.abs(y_valid - b1_valid)[bulk_mask].mean())
        bulk_e = float(np.abs(y_valid - e_valid)[bulk_mask].mean())
        bulk_b0 = float(np.abs(y_valid - yhat0_valid)[bulk_mask].mean())
        bulk_delta = bulk_e - bulk_b1
        bulk_delta_b0 = bulk_e - bulk_b0
    else:
        bulk_b1 = bulk_e = bulk_b0 = float("nan")
        bulk_delta = bulk_delta_b0 = float("nan")
    probe_bulk_mask = train_rare[probe_positions] < threshold
    if int(probe_bulk_mask.sum()) >= 10:
        probe_bulk_delta = float(
            np.abs(y_probe - e_probe)[probe_bulk_mask].mean()
            - np.abs(y_probe - b1_probe)[probe_bulk_mask].mean()
        )
    else:
        probe_bulk_delta = float("nan")

    # adapter dependency / collapse check (mandatory section 38 / Test 12)
    dependency = _state_dependency(state, valid, adapter_seed)

    return {
        "backbone_seed": int(seed),
        "adapter_seed": int(adapter_seed),
        "valid": {
            "mae_b0": mae_b0_v,
            "mae_b1": mae_b1_v,
            "mae_e": mae_e_v,
            "delta0": mae_b0_v - mae_e_v,
            "delta_state": mae_b1_v - mae_e_v,
            "n_eval": int(len(y_valid)),
        },
        "train_probe": {
            "mae_b0": mae_b0_p,
            "mae_b1": mae_b1_p,
            "mae_e": mae_e_p,
            "delta0": mae_b0_p - mae_e_p,
            "delta_state": mae_b1_p - mae_e_p,
            "n_eval": int(len(y_probe)),
            "probe_bulk_delta_state": probe_bulk_delta,
        },
        "bulk": {
            "rare_le5_threshold": threshold,
            "n_bulk_valid": int(bulk_mask.sum()),
            "mae_b0": bulk_b0,
            "mae_b1": bulk_b1,
            "mae_e": bulk_e,
            "delta_state": bulk_delta,
            "delta0": bulk_delta_b0,
        },
        "training": {
            "ronly_selection_mae": ronly_train["best_selection_mae"],
            "state_selection_mae": state_train["best_selection_mae"],
            "ronly_epochs": ronly_train["epochs"],
            "state_epochs": state_train["epochs"],
            "ronly_grad_alive": ronly_train["grad_alive"],
            "state_grad_alive": state_train["grad_alive"],
        },
        "dependency": dependency,
        "params": {
            "ronly": frs._count_parameters(ronly),
            "state": frs._count_parameters(state),
        },
        "_molecules": {
            "err_b0": np.abs(y_valid - yhat0_valid),
            "err_b1": np.abs(y_valid - b1_valid),
            "err_e": np.abs(y_valid - e_valid),
        },
    }


def _state_dependency(model: Any, tensors: Mapping[str, Any], adapter_seed: int) -> dict[str, Any]:
    with torch.no_grad():
        baseline = model(tensors).numpy()
        summary_off = model(tensors, use_summary=False).numpy()
    zero = dict(tensors)
    zero["patch_states"] = torch.zeros_like(tensors["patch_states"])
    zero["pair_states"] = torch.zeros_like(tensors["pair_states"])
    with torch.no_grad():
        zero_pred = model(zero).numpy()
    rng = np.random.RandomState(20260919 + int(adapter_seed))
    shuffled = frs._shuffle_states(tensors, rng)
    with torch.no_grad():
        shuffled_pred = model(shuffled).numpy()
    return {
        "summary_off_max_abs_diff": float(np.abs(baseline - summary_off).max()),
        "state_zero_max_abs_diff": float(np.abs(baseline - zero_pred).max()),
        "state_permuted_max_abs_diff": float(np.abs(baseline - shuffled_pred).max()),
        "prediction_std": float(baseline.std()),
    }


def run_stage1(
    adapter_seed: int,
    tag: str,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = stage_splits()
    adapter_protocol_lock()
    train_records, valid_records = _extract_v4_records()
    counter = _train_counter(train_records)
    train_rare = _rare_ratio_rows(train_records, counter)
    valid_rare = _rare_ratio_rows(valid_records, counter)

    rows = []
    for seed in BACKBONE_SEEDS:
        train_export = load_export(_export_path("train", seed))
        valid_export = load_export(_export_path("valid", seed))
        result = _run_seed(
            seed, adapter_seed, train_export, valid_export, manifest, train_rare, valid_rare
        )
        molecules = result.pop("_molecules")
        fold_ids = np.zeros(len(molecules["err_e"]), dtype=np.int64)
        result["bootstrap_delta_state"] = frs._stratified_bootstrap(
            molecules["err_b1"] - molecules["err_e"], fold_ids
        )
        result["bootstrap_delta0"] = frs._stratified_bootstrap(
            molecules["err_b0"] - molecules["err_e"], fold_ids
        )
        rows.append(result)
        _write_json_any(RESULTS_DIR / f"seed{seed}_{tag}.json", result)
        print(
            f"[{tag}] seed={seed} valid B0={result['valid']['mae_b0']:.5f} "
            f"B1={result['valid']['mae_b1']:.5f} E={result['valid']['mae_e']:.5f} "
            f"dState={result['valid']['delta_state']:+.5f}",
            flush=True,
        )

    frame = pd.DataFrame(
        [
            {
                "backbone_seed": r["backbone_seed"],
                "adapter_seed": r["adapter_seed"],
                "valid_mae_b0": r["valid"]["mae_b0"],
                "valid_mae_b1": r["valid"]["mae_b1"],
                "valid_mae_e": r["valid"]["mae_e"],
                "valid_delta_state": r["valid"]["delta_state"],
                "valid_delta0": r["valid"]["delta0"],
                "probe_mae_b0": r["train_probe"]["mae_b0"],
                "probe_mae_b1": r["train_probe"]["mae_b1"],
                "probe_mae_e": r["train_probe"]["mae_e"],
                "probe_delta_state": r["train_probe"]["delta_state"],
                "bulk_delta_state": r["bulk"]["delta_state"],
                "n_bulk_valid": r["bulk"]["n_bulk_valid"],
                "state_zero_max_abs_diff": r["dependency"]["state_zero_max_abs_diff"],
                "state_permuted_max_abs_diff": r["dependency"]["state_permuted_max_abs_diff"],
            }
            for r in rows
        ]
    )
    csv_path = RESULTS_DIR / ("valid_summary.csv" if tag == "stage1" else f"{tag}_results.csv")
    frame.to_csv(csv_path, index=False)
    return {"tag": tag, "rows": rows, "csv": str(csv_path), "seconds": time.perf_counter() - started}


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

def _decide(rows: Sequence[Mapping[str, Any]], gate0_passed: bool) -> dict[str, Any]:
    deltas = {int(r["backbone_seed"]): float(r["valid"]["delta_state"]) for r in rows}
    bulk = {int(r["backbone_seed"]): float(r["bulk"]["delta_state"]) for r in rows}
    dependencies = {
        int(r["backbone_seed"]): max(
            float(r["dependency"]["state_zero_max_abs_diff"]),
            float(r["dependency"]["state_permuted_max_abs_diff"]),
        )
        for r in rows
    }
    mean_delta = float(np.mean(list(deltas.values())))
    min_delta = float(np.min(list(deltas.values())))
    max_delta = float(np.max(list(deltas.values())))
    sign_consistent = len({np.sign(v) for v in deltas.values() if v != 0.0}) <= 1
    bulk_values = [v for v in bulk.values() if np.isfinite(v)]
    bulk_safe = bool(bulk_values) and max(bulk_values) <= BULK_MAX_DEGRADATION
    state_alive = all(v > STATE_DEPENDENCY_MIN for v in dependencies.values())

    advance = (
        all(v >= ADVANCE_PER_SEED for v in deltas.values())
        and mean_delta >= ADVANCE_MEAN
        and bulk_safe
        and state_alive
    )
    # CLEAR NO-GO clauses (mandate section 33)
    clear_nogo = (
        mean_delta <= CLEAR_NOGO_MEAN
        or all(v <= 0.0 for v in deltas.values())
        or (min_delta <= CLEAR_NOGO_SINGLE and mean_delta < ADVANCE_MEAN)
    )

    if not gate0_passed:
        verdict = "INVALID"
    elif not state_alive:
        verdict = "INVALID"
    elif advance:
        verdict = "BROAD_STATE_SIGNAL_ADVANCE"
    elif clear_nogo:
        verdict = "BROAD_STATE_NO_GO"
    else:
        verdict = "INCONCLUSIVE_BORDERLINE"

    return {
        "deltas": {str(k): v for k, v in deltas.items()},
        "mean_delta_state": mean_delta,
        "min_delta_state": min_delta,
        "max_delta_state": max_delta,
        "sign_consistent": bool(sign_consistent),
        "bulk_delta_valid": {str(k): v for k, v in bulk.items()},
        "bulk_safe": bulk_safe,
        "state_dependency": {str(k): v for k, v in dependencies.items()},
        "state_alive": state_alive,
        "criteria": {
            "per_seed_ge_+0.002": all(v >= ADVANCE_PER_SEED for v in deltas.values()),
            "mean_ge_+0.003": mean_delta >= ADVANCE_MEAN,
            "bulk_safe_le_+0.002": bulk_safe,
            "state_branch_alive": state_alive,
        },
        "verdict": verdict,
    }


def _combine_inits(
    rows: Sequence[Mapping[str, Any]], second_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Average the two adapter inits per backbone (mandate section 35/36)."""
    combined: list[dict[str, Any]] = []
    for s0, s1 in zip(rows, second_rows):
        merged = json.loads(json.dumps(s0))
        for block in ("valid", "train_probe"):
            for key in ("mae_b0", "mae_b1", "mae_e", "delta0", "delta_state"):
                merged[block][key] = float(np.mean([s0[block][key], s1[block][key]]))
        merged["bulk"] = dict(s0["bulk"])
        merged["bulk"]["delta_state"] = float(
            np.mean([s0["bulk"]["delta_state"], s1["bulk"]["delta_state"]])
        )
        merged["bulk"]["delta0"] = float(
            np.mean([s0["bulk"]["delta0"], s1["bulk"]["delta0"]])
        )
        combined.append(merged)
    return combined


def _strategic_implication(verdict: str) -> str:
    if verdict == "BROAD_STATE_SIGNAL_ADVANCE":
        return "Authorise only a two-fold optimized OOF pilot (mandate section 41/42); do not build an architecture."
    if verdict == "INVALID":
        return "Measurement invalid; not a scientific NO-GO. Fix the export/integrity issue before re-running."
    return (
        "DO NOT rebuild the old witness tree (no optimized endpoint / covariance / triad "
        "revalidation) and do NOT buy optimized OOF compute.  The broad state increment is "
        "sub-threshold and non-robust; the next move must be paradigm-level, not another "
        "missing-local-statistic probe."
    )


def _questions(rows: Sequence[Mapping[str, Any]], decision: Mapping[str, Any], historical: Mapping[str, Any]) -> dict[str, str]:
    d = decision["deltas"]
    r0 = rows[0]
    r1 = rows[1] if len(rows) > 1 else rows[0]
    return {
        "Q1": (
            "optimized compact-v4-hinge seed0 (valid "
            f"{EXPECTED_VALID[0]:.6f}, best epoch {EXPECTED_BEST_EPOCH[0]}) and seed1 (valid "
            f"{EXPECTED_VALID[1]:.6f}, best epoch {EXPECTED_BEST_EPOCH[1]})"
        ),
        "Q2": "true pre-head R = 302D = unary 97 + pair 165 + global 32 + topology 8",
        "Q3": "the updated patch/centre states h'_i (48D) and the frozen pair states q_ij (16D)",
        "Q4": (
            "historical corrected broad audit (under-trained 60-epoch manifold): "
            "B1 R-only 0.19050, set/state 0.19229, delta_state < 0 -> NO-GO"
        ),
        "Q5": (
            f"seed0 valid B0={r0['valid']['mae_b0']:.5f} B1={r0['valid']['mae_b1']:.5f} "
            f"E={r0['valid']['mae_e']:.5f}"
        ),
        "Q6": f"seed0 delta_state = B1 - E = {d['0']:+.5f}",
        "Q7": (
            f"seed1 valid B0={r1['valid']['mae_b0']:.5f} B1={r1['valid']['mae_b1']:.5f} "
            f"E={r1['valid']['mae_e']:.5f}"
        ),
        "Q8": f"seed1 delta_state = B1 - E = {d['1']:+.5f}",
        "Q9": f"mean delta_state = {decision['mean_delta_state']:+.5f} "
              f"(min {decision['min_delta_state']:+.5f}, max {decision['max_delta_state']:+.5f})",
        "Q10": f"sign consistent across the two backbones: {decision['sign_consistent']}",
        "Q11": f"common-input bulk safe (<= +0.002): {decision['bulk_safe']}",
        "Q12": f"state branch alive (dependency > {STATE_DEPENDENCY_MIN}): {decision['state_alive']}",
        "Q13": "see final_decision.json 'second_init'",
        "Q14": f"optimized OOF pilot authorised: {decision['verdict'] == 'BROAD_STATE_SIGNAL_ADVANCE'}",
        "Q15": decision["verdict"],
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _make_figures(rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(r["backbone_seed"]) for r in rows]
    b0 = [r["valid"]["mae_b0"] for r in rows]
    b1 = [r["valid"]["mae_b1"] for r in rows]
    e = [r["valid"]["mae_e"] for r in rows]
    delta = [r["valid"]["delta_state"] for r in rows]

    x = np.arange(len(seeds))
    width = 0.25
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(x - width, b0, width, label="B0 frozen optimized")
    ax.bar(x, b1, width, label="B1 R-only")
    ax.bar(x + width, e, width, label="E broad state")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed{s}" for s in seeds])
    ax.set_ylabel("official-valid MAE")
    ax.set_title("Figure 1: B0 / B1 / E on official valid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_valid_mae.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    colors = ["tab:green" if v > 0 else "tab:red" for v in delta]
    ax.bar([f"seed{s}" for s in seeds], delta, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(ADVANCE_PER_SEED, color="tab:blue", linewidth=0.8, linestyle="--",
               label=f"per-seed gate +{ADVANCE_PER_SEED}")
    ax.set_ylabel(r"$\Delta_{state} = MAE(B1) - MAE(E)$")
    ax.set_title("Figure 2: broad-state increment per backbone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_delta_state.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_stage1_artifacts(stage1: Mapping[str, Any]) -> None:
    if len(stage1["rows"]) != 2:
        return
    _write_json_any(RESULTS_DIR / "seed0_results.json", stage1["rows"][0])
    _write_json_any(RESULTS_DIR / "seed1_results.json", stage1["rows"][1])
    probe_frame = pd.DataFrame(
        [
            {
                "backbone_seed": r["backbone_seed"],
                "adapter_seed": r["adapter_seed"],
                "probe_mae_b0": r["train_probe"]["mae_b0"],
                "probe_mae_b1": r["train_probe"]["mae_b1"],
                "probe_mae_e": r["train_probe"]["mae_e"],
                "probe_delta_state": r["train_probe"]["delta_state"],
            }
            for r in stage1["rows"]
        ]
    )
    probe_frame.to_csv(RESULTS_DIR / "train_probe_summary.csv", index=False)


def _finalize_decision(
    rows: Sequence[Mapping[str, Any]],
    gate0_passed: bool,
    second_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    primary_decision = _decide(rows, gate0_passed)
    decision = primary_decision
    rows_for_questions: Sequence[Mapping[str, Any]] = rows
    second_init = None
    if second_rows is not None:
        second_init = "ran (adapter_seed=%d)" % SECOND_ADAPTER_SEED
        combined = _combine_inits(rows, second_rows)
        decision["second_init_rows"] = list(second_rows)
        decision["combined_after_second_init"] = combined
        if primary_decision["verdict"] == "INCONCLUSIVE_BORDERLINE":
            decision = _decide(combined, gate0_passed)
            rows_for_questions = combined
            if decision["verdict"] != "BROAD_STATE_SIGNAL_ADVANCE":
                # Mandate section 36 / Decision Case C: a borderline primary
                # that does not become an unambiguous advance after the second
                # adapter init is INCONCLUSIVE -- the budget is NOT expanded.
                decision["verdict"] = "INCONCLUSIVE_DO_NOT_BUILD_OOF"
    decision["primary_init_decision"] = primary_decision["verdict"]
    decision["export_version"] = EXPORT_VERSION
    decision["gate0_all_passed"] = gate0_passed
    decision["historical_reference"] = _load_historical_reference()
    decision["seed_rows"] = {"seed0": rows[0], "seed1": rows[1]}
    decision["second_init"] = second_init or "not triggered"
    decision["primary_backbones"] = list(BACKBONE_SEEDS)
    decision["strategic_implication"] = _strategic_implication(decision["verdict"])
    decision["questions"] = _questions(rows_for_questions, decision, decision["historical_reference"])
    decision["scope_caveat"] = (
        "This is a canonical-checkpoint screening diagnostic used only to decide "
        "whether optimized OOF reconstruction is worth the compute.  The optimized "
        "backbone checkpoints were valid-selected, so a positive result is NOT clean "
        "OOF mechanism evidence.  A negative result terminates the local "
        "endpoint/covariance/triad revalidation tree; it does NOT prove that no "
        "information exists beyond R."
    )
    return decision


def _load_historical_reference() -> dict[str, Any]:
    return {
        "source": "results/frozen_readout_sufficiency/stage1_fold_results.csv",
        "b1_r_only_mean": 0.190499,
        "set_state_mean": 0.192289,
        "delta_state_sign": "negative",
        "manifold": "historical under-trained 60-epoch compact-v4-hinge",
        "note": "absolute MAE not comparable (different backbone / protocol / split)",
    }


def _run_all(force: bool = False) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== checkpoint inventory ===", flush=True)
    inventory = checkpoint_inventory()
    _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    if any(e["status"] != "present" for e in inventory["entries"]):
        print("missing optimized checkpoint; STOP.", flush=True)
        return 2

    print("=== export ===", flush=True)
    stage_export(force=force)

    print("=== gate0 ===", flush=True)
    gate0 = run_gate0(force=force)
    if not gate0["all_passed"]:
        _write_json_any(
            RESULTS_DIR / "final_decision.json",
            {
                "verdict": "INVALID",
                "reason": "export integrity gate failed",
                "failures": gate0["failures"],
            },
        )
        print("GATE 0 FAILED - measurement invalid. STOP.", flush=True)
        return 3

    print("=== stage1 (primary adapter init) ===", flush=True)
    stage1 = run_stage1(PRIMARY_ADAPTER_SEED, "stage1", force=force)
    _write_stage1_artifacts(stage1)
    second_init = None
    if _decide(stage1["rows"], gate0["all_passed"])["verdict"] == "INCONCLUSIVE_BORDERLINE":
        print("=== stage1b (second adapter init) ===", flush=True)
        second_init = run_stage1(SECOND_ADAPTER_SEED, "stage1b", force=force)
    decision = _finalize_decision(stage1["rows"], gate0["all_passed"], None if second_init is None else second_init["rows"])
    _write_json_any(RESULTS_DIR / "final_decision.json", decision)
    _write_json_any(
        RESULTS_DIR / "common_input_bulk.json",
        {f"seed{r['backbone_seed']}": r["bulk"] for r in stage1["rows"]},
    )
    _write_json_any(
        RESULTS_DIR / "adapter_dependency.json",
        {f"seed{r['backbone_seed']}": r["dependency"] for r in stage1["rows"]},
    )
    if second_init is not None:
        import shutil

        shutil.copyfile(RESULTS_DIR / "stage1b_results.csv", RESULTS_DIR / "second_init_results.csv")
    _make_figures(stage1["rows"])
    print(json.dumps(decision, indent=2, sort_keys=True, default=str), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "inventory", "export", "gate0", "splits", "stage1", "stage1b",
            "decision", "figures", "all",
        ],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "inventory":
        inventory = checkpoint_inventory()
        _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0
    if args.stage == "export":
        stage_export(force=args.force)
        return 0
    if args.stage == "gate0":
        report = run_gate0(force=args.force)
        _write_json_any(RESULTS_DIR / "export_integrity.json", report)
        return 0 if report.get("all_passed") else 1
    if args.stage == "splits":
        stage_splits(force=args.force)
        adapter_protocol_lock()
        return 0
    if args.stage == "stage1":
        result = run_stage1(PRIMARY_ADAPTER_SEED, "stage1", force=args.force)
        _write_stage1_artifacts(result)
        _make_figures(result["rows"])
        return 0
    if args.stage == "stage1b":
        run_stage1(SECOND_ADAPTER_SEED, "stage1b", force=args.force)
        return 0
    if args.stage == "figures":
        if (RESULTS_DIR / "seed0_results.json").exists() and (RESULTS_DIR / "seed1_results.json").exists():
            rows = [
                json.loads((RESULTS_DIR / "seed0_results.json").read_text()),
                json.loads((RESULTS_DIR / "seed1_results.json").read_text()),
            ]
            _make_figures(rows)
        return 0
    if args.stage == "decision":
        gate0 = json.loads((RESULTS_DIR / "export_integrity.json").read_text())
        rows = [
            json.loads((RESULTS_DIR / "seed0_results.json").read_text()),
            json.loads((RESULTS_DIR / "seed1_results.json").read_text()),
        ]
        second_rows = None
        stage1b_paths = [RESULTS_DIR / "seed0_stage1b.json", RESULTS_DIR / "seed1_stage1b.json"]
        if all(p.exists() for p in stage1b_paths):
            second_rows = [json.loads(p.read_text()) for p in stage1b_paths]
        decision = _finalize_decision(rows, gate0.get("all_passed", False), second_rows)
        _write_json_any(RESULTS_DIR / "final_decision.json", decision)
        _write_json_any(
            RESULTS_DIR / "common_input_bulk.json",
            {f"seed{r['backbone_seed']}": r["bulk"] for r in rows},
        )
        _write_json_any(
            RESULTS_DIR / "adapter_dependency.json",
            {f"seed{r['backbone_seed']}": r["dependency"] for r in rows},
        )
        if second_rows is not None:
            import shutil

            shutil.copyfile(
                RESULTS_DIR / "stage1b_results.csv", RESULTS_DIR / "second_init_results.csv"
            )
        print(json.dumps(decision, indent=2, sort_keys=True, default=str))
        return 0
    if args.stage == "all":
        return _run_all(force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
