"""E2E-DictEnv-BondAnchoredTriple-v1 — frozen-representation triple screen (core).

Round ``e2e_dictenv_bond_anchored_triple_v1`` (study ``zinc-context-gap``).
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_bond_anchored_triple_v1_preregistration.md``.

One question, one candidate, one seed, 80 epochs: with the original
``CSSD-Sem108 + C6`` parent **frozen**, does keeping the common-endpoint
correspondence of three frozen pair tokens — one environment per unique real
bond ``(i, j)``, one environment per third node ``k`` — give a valid-MAE signal
before graph-level aggregation?

The module contains:

* frozen-parent loading + provenance (checkpoint sha256, canonical state hash);
* the frozen feature cache (``p_ij`` 16-D, ``z_old`` 302-D) with a
  provenance key that forces a rebuild on any parent/code/split change;
* the exact triple-object construction (unique ``i < j`` real-key anchors,
  ``m*(n-2)`` triples per graph, zero summary for degenerate graphs);
* the new trainable path ``F: 48->64->32`` (shared, both orderings) and the
  re-initialised ``Reader: 366->13->13->1`` with fixed train-only
  standardizers;
* the 80-epoch training/soup protocol and the focused correctness helpers.

CPU-first (the repository execution regime); official valid is evaluation only
and the official test split is never instantiated.  ``official_test_loaded`` is
``False`` in every payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = "e2e_dictenv_bond_anchored_triple_v1"
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_bond_anchored_triple_v1"
CACHE_DIR = RESULTS_DIR / "cache"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_bond_anchored_triple_v1_preregistration.md"

SEM108_RESULTS = TRACK_ROOT / "results/e2e_dictenv_sem108_v1"
PARENT_SOUP_PATH = SEM108_RESULTS / "checkpoints/SEM108-seed0_soup_state.pt"
PARENT_SOUP_SHA256 = "7a721984c5579dd76f9bcaff13a2055b8efba71da553eefb3645d94207f5050a"
PARENT_HISTORICAL_SOUP_MAE = 0.123704927947314
PARENT_SOURCE_RUN = "tracks/ksvd/results/e2e_dictenv_sem108_v1/"
PARENT_SOURCE_COMMIT = "f836b04"
PARENT_READER_PARAMS = 4135
PARENT_TOTAL_PARAMS = 97709

SUBSPACE_PATH = (
    TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json"
)

CACHE_FORMAT_VERSION = 1

PAIR_DIM = int(p2.PAIR_HIDDEN)  # 16
Z_OLD_DIM = int(audit.READER_IN_DIM)  # 302
TRIPLE_SUMMARY_DIM = 64
F_HIDDEN = 64
F_OUT = 32
F_IN = 3 * PAIR_DIM
READER_HIDDEN = (13, 13)
READER_IN = Z_OLD_DIM + TRIPLE_SUMMARY_DIM
STD_EPS = 1.0e-6

TRAIN_SIZE = 10_000
VALID_SIZE = 1_000

EPOCHS = 80
BATCH_SIZE = int(p2run.BATCH_SIZE)  # 128
LEARNING_RATE = float(p2run.LEARNING_RATE)  # 1e-3
WEIGHT_DECAY = float(p2run.WEIGHT_DECAY)  # 1e-5 coupled L2
GRAD_CLIP = float(p2run.GRAD_CLIP)  # 5.0
TRAIN_SHUFFLE_OFFSET = int(p2run.TRAIN_SHUFFLE_OFFSET)  # 91011
EVAL_SHUFFLE_OFFSET = int(p2run.EVAL_SHUFFLE_OFFSET)  # 91012
SOUP_K = int(p2run.SOUP_K)  # 5
SOUP_EPOCH_LO = 41
SOUP_EPOCH_HI = 80
MASK = cm.C6_MASK  # type: ignore[assignment]

EXPECTED_F_PARAMS = (F_IN + 1) * F_HIDDEN + (F_HIDDEN + 1) * F_OUT  # 5216
EXPECTED_READER_PARAMS = (
    (READER_IN + 1) * READER_HIDDEN[0]
    + (READER_HIDDEN[0] + 1) * READER_HIDDEN[1]
    + (READER_HIDDEN[1] + 1) * 1
)  # 4967
EXPECTED_TRAINABLE = EXPECTED_F_PARAMS + EXPECTED_READER_PARAMS  # 10183
EXPECTED_FULL_MODEL = PARENT_TOTAL_PARAMS - PARENT_READER_PARAMS + EXPECTED_TRAINABLE  # 103757

THREADS = 8
FLOAT32 = torch.float32


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_sha256() -> str:
    return _sha256_file(Path(__file__))


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


def cpu_guard(device: Any) -> None:
    if torch.device(device).type != "cpu":
        raise RuntimeError(f"BondAnchoredTriple-v1 core is CPU-only, got {device}")


# ---------------------------------------------------------------------------
# frozen parent
# ---------------------------------------------------------------------------


def load_dictionary() -> tuple[np.ndarray, str]:
    return p2run.load_dictionary(cm.H1_CONFIG.dict_kind)


def load_subspace() -> cssd.CommonSubspace:
    entry = _read_json(SUBSPACE_PATH)["q1"]
    return cssd.CommonSubspace(
        components=np.asarray(entry["components"], dtype=np.float64),
        rms=np.asarray(entry["rms"], dtype=np.float64),
        kind=str(entry["kind"]),
    )


def load_parent_state() -> dict[str, torch.Tensor]:
    state = torch.load(PARENT_SOUP_PATH, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError("Sem108 soup checkpoint is not a state mapping")
    return {str(key): value for key, value in state.items()}


def build_parent(seed: int = 0) -> sem.SEM108Model:
    dictionary, _sha = load_dictionary()
    subspace = load_subspace()
    model = sem.build_sem108_model(dictionary, int(seed), subspace)
    model.load_state_dict(load_parent_state())
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def parent_provenance(model: sem.SEM108Model | None = None) -> dict[str, Any]:
    state = load_parent_state()
    if model is None:
        model = build_parent()
    measured = {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "reader_params": int(sum(p.numel() for p in model.reader.parameters())),
        "reader_input_dim": int(model.reader.net[0].in_features),
        "pair_token_dim": int(model.pair_encoder.layers[-2].out_features),
        "environment_dim": int(p2.ENV_DIM),
        "state_keys": int(len(state)),
        "canonical_state_sha256": audit.state_sha256(state),
        "checkpoint_path": str(PARENT_SOUP_PATH.relative_to(REPO_ROOT)),
        "checkpoint_sha256": _sha256_file(PARENT_SOUP_PATH),
        "historical_valid_soup_mae": float(PARENT_HISTORICAL_SOUP_MAE),
        "source_run": str(PARENT_SOURCE_RUN),
        "source_commit": str(PARENT_SOURCE_COMMIT),
        "mask": MASK.as_dict(),
        "config": cm.H1_CONFIG.as_dict(),
        "dictionary_sha256": load_dictionary()[1],
        "subspace_kind": load_subspace().kind,
        "all_params_requires_grad_false": bool(
            all(not p.requires_grad for p in model.parameters())
        ),
        "official_test_loaded": False,
    }
    if measured["total_params"] != PARENT_TOTAL_PARAMS:
        raise RuntimeError(
            f"parent total params {measured['total_params']} != {PARENT_TOTAL_PARAMS}"
        )
    if measured["reader_params"] != PARENT_READER_PARAMS:
        raise RuntimeError(
            f"parent reader params {measured['reader_params']} != {PARENT_READER_PARAMS}"
        )
    if measured["canonical_state_sha256"] != PARENT_SOUP_SHA256:
        raise RuntimeError("parent canonical state hash mismatch")
    official_test_blocker(measured)
    return measured


def parent_predictions(
    model: sem.SEM108Model,
    loaders: Sequence[Any],
    capture: dict[str, list[torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Prediction-only parent evaluation on a list of batches (CPU, C6 mask)."""
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loaders:
            if capture is not None:
                import contextlib

                handles = [
                    model.pair_encoder.register_forward_hook(
                        lambda _m, _i, o, c=capture: c["pair_value"].append(o.detach().clone())
                    ),
                    model.reader.register_forward_hook(
                        lambda _m, i, _o, c=capture: c["z_old"].append(i[0].detach().clone())
                    ),
                ]
                try:
                    prediction = model(batch, mask=MASK)
                finally:
                    for handle in handles:
                        handle.remove()
            else:
                prediction = model(batch, mask=MASK)
            predictions.append(prediction.view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    target = np.concatenate(targets).astype(np.float64)
    pred = np.concatenate(predictions).astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(target - pred))),
        "n_molecules": int(target.shape[0]),
        "predictions": pred,
        "targets": target,
    }


def replay_parent_valid_mae(model: sem.SEM108Model | None = None) -> dict[str, Any]:
    parent = build_parent() if model is None else model
    valid_data = p1run.load_split("valid")
    loader = p1.make_env_loader(
        valid_data, BATCH_SIZE, False, 0 + EVAL_SHUFFLE_OFFSET
    )
    result = parent_predictions(parent, loader)
    replay = float(result["mae"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "device": "cpu",
        "mask": MASK.as_dict(),
        "n_valid": int(result["n_molecules"]),
        "replay_valid_mae": replay,
        "historical_valid_soup_mae": float(PARENT_HISTORICAL_SOUP_MAE),
        "abs_diff": float(abs(replay - PARENT_HISTORICAL_SOUP_MAE)),
        "tolerance": 1.0e-7,
        "passed": bool(abs(replay - PARENT_HISTORICAL_SOUP_MAE) <= 1.0e-7),
        "parent_state_sha256": audit.state_sha256(load_parent_state()),
    }
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# frozen feature cache
# ---------------------------------------------------------------------------


def _cache_key(split: str) -> str:
    payload = {
        "cache_format_version": int(CACHE_FORMAT_VERSION),
        "protocol_version": PROTOCOL_VERSION,
        "parent_canonical_state_sha256": PARENT_SOUP_SHA256,
        "parent_checkpoint_sha256": _sha256_file(PARENT_SOUP_PATH),
        "mask_signature": MASK.signature(),
        "module_sha256": module_sha256(),
        "split": str(split),
        "order_policy": "official_split_order",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_path(split: str) -> Path:
    return CACHE_DIR / f"features_{split}_{_cache_key(split)[:16]}.pt"


def cache_manifest_path() -> Path:
    return CACHE_DIR / "cache_manifest.json"


def _split_data(split: str) -> list[Any]:
    if split not in ("train", "valid"):
        raise ValueError(f"unknown split {split!r}")
    return p1run.load_split(split)


@dataclass
class FeatureCache:
    pair_tokens: torch.Tensor      # [P, 16]
    z_old: torch.Tensor            # [G, 302]
    y: torch.Tensor                # [G]
    n_nodes: torch.Tensor          # [G]
    pair_ptr: torch.Tensor         # [G+1]
    pair_bucket: torch.Tensor      # [P]
    triple_ij: torch.Tensor        # [T] global pair row ids
    triple_ik: torch.Tensor        # [T]
    triple_jk: torch.Tensor        # [T]
    triple_graph: torch.Tensor     # [T]
    triple_ptr: torch.Tensor       # [G+1]
    n_bonds: torch.Tensor          # [G] unique real bonds
    provenance: dict[str, Any]

    @property
    def n_graphs(self) -> int:
        return int(self.z_old.shape[0])

    @property
    def n_pairs(self) -> int:
        return int(self.pair_tokens.shape[0])

    @property
    def n_triples(self) -> int:
        return int(self.triple_ij.shape[0])


def _empty_triples(pair_count: int, n_nodes: int) -> tuple[torch.Tensor, ...]:
    zero = torch.zeros(0, dtype=torch.int64)
    return zero, zero.clone(), zero.clone()


def graph_triple_rows(
    pair_index: torch.Tensor, pair_bucket: torch.Tensor, n_nodes: int
) -> dict[str, torch.Tensor]:
    """Exact per-graph triple construction from the (triangular) pair cache.

    ``pair_index`` is ``[2, P]`` with ``source < target`` and rows in triangular
    order.  Anchors are the ``pair_bucket == 0`` rows (the unique real bonds).
    Returns local pair-row ids ``ij / ik / jk`` for every unique triple
    ``(i, j; k)`` with ``k != i, j`` in anchor order then ascending ``k``.
    """
    n = int(n_nodes)
    if int(pair_index.shape[1]) == 0:
        empty = _empty_triples(0, n)
        return {"ij": empty[0], "ik": empty[1], "jk": empty[2], "anchors": empty[0]}
    lookup = torch.full((n, n), -1, dtype=torch.int64)
    rows = torch.arange(int(pair_index.shape[1]), dtype=torch.int64)
    lookup[pair_index[0].long(), pair_index[1].long()] = rows
    lookup[pair_index[1].long(), pair_index[0].long()] = rows
    anchor_mask = pair_bucket.long() == 0
    anchor_rows = torch.nonzero(anchor_mask, as_tuple=False).view(-1)
    ai = pair_index[0].long()[anchor_rows]
    aj = pair_index[1].long()[anchor_rows]
    m = int(ai.numel())
    if n < 3 or m == 0:
        empty = _empty_triples(int(pair_index.shape[1]), n)
        return {"ij": empty[0], "ik": empty[1], "jk": empty[2], "anchors": anchor_rows}
    k = torch.arange(n, dtype=torch.int64).unsqueeze(0).expand(m, n)
    valid = (k != ai.unsqueeze(1)) & (k != aj.unsqueeze(1))
    i_rep = ai.unsqueeze(1).expand(m, n)[valid]
    j_rep = aj.unsqueeze(1).expand(m, n)[valid]
    k_rep = k[valid]
    ij = lookup[i_rep, j_rep]
    ik = lookup[i_rep, k_rep]
    jk = lookup[j_rep, k_rep]
    if bool(((ij < 0) | (ik < 0) | (jk < 0)).any()):
        raise RuntimeError("triple construction produced an out-of-cache pair row")
    return {"ij": ij, "ik": ik, "jk": jk, "anchors": anchor_rows}


def extract_split_features(
    parent: sem.SEM108Model, split: str, batch_size: int = BATCH_SIZE
) -> dict[str, Any]:
    """One frozen-parent pass per graph: cache ``p_ij``, ``z_old`` and bookkeeping."""
    data_list = _split_data(split)
    expected = TRAIN_SIZE if split == "train" else VALID_SIZE
    if len(data_list) != expected:
        raise RuntimeError(f"{split} split size {len(data_list)} != {expected}")

    pair_parts: list[torch.Tensor] = []
    z_old_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    n_nodes: list[int] = []
    pair_ptr = [0]
    pair_bucket_parts: list[torch.Tensor] = []
    n_bonds: list[int] = []
    triple_ij: list[torch.Tensor] = []
    triple_ik: list[torch.Tensor] = []
    triple_jk: list[torch.Tensor] = []
    triple_graph: list[torch.Tensor] = []
    triple_ptr = [0]

    started = time.perf_counter()
    for start in range(0, len(data_list), int(batch_size)):
        chunk = list(data_list[start : start + int(batch_size)])
        batch = p1.env_collate(chunk)
        capture: dict[str, list[torch.Tensor]] = {"pair_value": [], "z_old": []}
        handles = [
            parent.pair_encoder.register_forward_hook(
                lambda _m, _i, o, c=capture: c["pair_value"].append(o.detach().clone())
            ),
            parent.reader.register_forward_hook(
                lambda _m, i, _o, c=capture: c["z_old"].append(i[0].detach().clone())
            ),
        ]
        try:
            with torch.no_grad():
                parent(batch, mask=MASK)
        finally:
            for handle in handles:
                handle.remove()
        pair_value = capture["pair_value"][0].detach().cpu()
        z_old = capture["z_old"][0].detach().cpu()
        if pair_value.shape[1] != PAIR_DIM or z_old.shape[1] != Z_OLD_DIM:
            raise RuntimeError(
                f"captured widths {tuple(pair_value.shape)} / {tuple(z_old.shape)}"
                f" != ({PAIR_DIM}) / ({Z_OLD_DIM})"
            )
        if int(z_old.shape[0]) != len(chunk):
            raise RuntimeError("captured z_old row count != chunk size")

        pair_graph = batch.batch[batch.pair_index[0].long()].cpu()
        pair_bucket = batch.pair_bucket.cpu()
        # validate the batch pair rows are exactly the per-graph concatenation
        offset = 0
        for local_index, data in enumerate(chunk):
            n = int(data.dict_phi.shape[0])
            npairs = int(data.pair_index.shape[1])
            if npairs != n * (n - 1) // 2:
                raise RuntimeError(
                    f"{split}[{start + local_index}] pair count {npairs} != n(n-1)/2"
                )
            expected_pairs = torch.tensor(
                [(a, b) for a in range(n) for b in range(a + 1, n)], dtype=torch.int64
            ).t()
            if not torch.equal(data.pair_index.to(torch.int64), expected_pairs):
                raise RuntimeError(
                    f"{split}[{start + local_index}] pair cache is not the triangular i<j enumeration"
                )
            rows = pair_graph == local_index
            if int(rows.sum()) != npairs or not bool(rows[offset : offset + npairs].all()):
                raise RuntimeError("batched pair rows are not the per-graph concatenation")
            chunk_pair = pair_value[offset : offset + npairs]
            chunk_bucket = pair_bucket[offset : offset + npairs]
            triples = graph_triple_rows(data.pair_index, chunk_bucket, n)
            m = int(triples["anchors"].numel())
            expected_triples = m * (n - 2) if n >= 3 else 0
            if int(triples["ij"].numel()) != expected_triples:
                raise RuntimeError(
                    f"{split}[{start + local_index}] triples {int(triples['ij'].numel())}"
                    f" != m*(n-2) = {expected_triples}"
                )
            base = pair_ptr[-1]
            pair_parts.append(chunk_pair)
            pair_bucket_parts.append(chunk_bucket)
            z_old_parts.append(z_old[local_index : local_index + 1])
            y_parts.append(data.y.view(-1).float())
            n_nodes.append(n)
            n_bonds.append(m)
            pair_ptr.append(base + npairs)
            triple_ij.append(triples["ij"] + base)
            triple_ik.append(triples["ik"] + base)
            triple_jk.append(triples["jk"] + base)
            triple_graph.append(torch.full((int(triples["ij"].numel()),), len(y_parts) - 1, dtype=torch.int64))
            triple_ptr.append(triple_ptr[-1] + int(triples["ij"].numel()))
            offset += npairs
        if offset != int(pair_value.shape[0]):
            raise RuntimeError("pair offset did not consume the batch")

    payload = {
        "pair_tokens": torch.cat(pair_parts, dim=0).to(torch.float32),
        "z_old": torch.cat(z_old_parts, dim=0).to(torch.float32),
        "y": torch.stack(y_parts).view(-1).to(torch.float32),
        "n_nodes": torch.as_tensor(n_nodes, dtype=torch.int64),
        "pair_ptr": torch.as_tensor(pair_ptr, dtype=torch.int64),
        "pair_bucket": torch.cat(pair_bucket_parts, dim=0).to(torch.int64),
        "triple_ij": torch.cat(triple_ij, dim=0).to(torch.int32),
        "triple_ik": torch.cat(triple_ik, dim=0).to(torch.int32),
        "triple_jk": torch.cat(triple_jk, dim=0).to(torch.int32),
        "triple_graph": torch.cat(triple_graph, dim=0).to(torch.int32),
        "triple_ptr": torch.as_tensor(triple_ptr, dtype=torch.int64),
        "n_bonds": torch.as_tensor(n_bonds, dtype=torch.int64),
    }
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_format_version": int(CACHE_FORMAT_VERSION),
        "official_test_loaded": False,
        "split": str(split),
        "split_size": int(len(data_list)),
        "n_pairs": int(payload["pair_tokens"].shape[0]),
        "n_triples": int(payload["triple_ij"].shape[0]),
        "parent_canonical_state_sha256": PARENT_SOUP_SHA256,
        "parent_checkpoint_sha256": _sha256_file(PARENT_SOUP_PATH),
        "mask_signature": MASK.signature(),
        "module_sha256": module_sha256(),
        "cache_key": _cache_key(split),
        "order_policy": "official_split_order",
        "seconds": float(time.perf_counter() - started),
    }
    payload["provenance"] = provenance
    return payload


def build_or_load_cache(split: str, force: bool = False) -> FeatureCache:
    path = cache_path(split)
    if path.exists() and not force:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        provenance = dict(blob["provenance"])
        if provenance.get("cache_key") == _cache_key(split):
            return FeatureCache(**{k: blob[k] for k in (
                "pair_tokens", "z_old", "y", "n_nodes", "pair_ptr", "pair_bucket",
                "triple_ij", "triple_ik", "triple_jk", "triple_graph", "triple_ptr",
                "n_bonds",
            )}, provenance=provenance)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parent = build_parent()
    blob = extract_split_features(parent, split)
    torch.save(blob, path)
    manifest = _read_json(cache_manifest_path()) if cache_manifest_path().exists() else {}
    manifest[split] = blob["provenance"]
    _write_json(cache_manifest_path(), manifest)
    return FeatureCache(**{k: blob[k] for k in (
        "pair_tokens", "z_old", "y", "n_nodes", "pair_ptr", "pair_bucket",
        "triple_ij", "triple_ik", "triple_jk", "triple_graph", "triple_ptr",
        "n_bonds",
    )}, provenance=blob["provenance"])


# ---------------------------------------------------------------------------
# initialisation and normalisation
# ---------------------------------------------------------------------------


@dataclass
class Standardizers:
    mu_p: torch.Tensor
    scale_p: torch.Tensor
    mu_old: torch.Tensor
    scale_old: torch.Tensor
    mu_3: torch.Tensor
    scale_3: torch.Tensor
    near_constant: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "mu_p": self.mu_p.tolist(),
            "scale_p": self.scale_p.tolist(),
            "mu_old": self.mu_old.tolist(),
            "scale_old": self.scale_old.tolist(),
            "mu_3": self.mu_3.tolist(),
            "scale_3": self.scale_3.tolist(),
            "near_constant": self.near_constant,
            "fit_split": "official train",
            "valid_used": False,
            "std_eps": float(STD_EPS),
        }


def fit_standardizer(raw: torch.Tensor, eps: float = STD_EPS) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Per-channel population standardizer; ``std < eps`` -> ``scale = 1``."""
    values = raw.detach().to(torch.float64)
    mu = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    near = std < float(eps)
    scale = torch.where(near, torch.ones_like(std), std)
    info = {
        "n_rows": int(values.shape[0]),
        "n_channels": int(values.shape[1]),
        "near_constant_count": int(near.sum()),
        "near_constant_dims": [int(v) for v in torch.nonzero(near, as_tuple=False).view(-1).tolist()],
        "min_std": float(std.min()) if std.numel() else float("nan"),
        "max_std": float(std.max()) if std.numel() else float("nan"),
    }
    return mu.to(torch.float32), scale.to(torch.float32), info


def fit_standardizers(train_cache: FeatureCache, seed: int = 0) -> Standardizers:
    """Train-only statistics.

    ``p`` and ``z_old`` come from the cache; the initial ``z3`` is computed with
    the freshly initialised F and the *fixed* ``p`` statistics, under
    ``no_grad`` and without touching the valid split.
    """
    mu_p, scale_p, info_p = fit_standardizer(train_cache.pair_tokens)
    mu_old, scale_old, info_old = fit_standardizer(train_cache.z_old)

    torch.manual_seed(int(seed))
    probe = BondAnchoredTripleModel(seed=int(seed))
    probe.mu_p.copy_(mu_p)
    probe.scale_p.copy_(scale_p)
    z3_raw = compute_raw_z3(probe, train_cache)
    mu_3, scale_3, info_3 = fit_standardizer(z3_raw)
    payload = Standardizers(
        mu_p=mu_p,
        scale_p=scale_p,
        mu_old=mu_old,
        scale_old=scale_old,
        mu_3=mu_3,
        scale_3=scale_3,
        near_constant={"pair": info_p, "z_old": info_old, "z3_init": info_3},
    )
    official_test_blocker({"official_test_loaded": False})
    return payload


def compute_raw_z3(
    model: "BondAnchoredTripleModel",
    cache: FeatureCache,
    chunk: int = 2048,
) -> torch.Tensor:
    """Raw ``z3`` for every graph (mean / second moment over that graph's triples)."""
    outputs: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, cache.n_graphs, int(chunk)):
            stop = min(start + int(chunk), cache.n_graphs)
            batch = collate_graphs(cache, list(range(start, stop)))
            p_ij, p_ik, p_jk = model.standardize_pair_rows(
                batch["pair_ij"], batch["pair_ik"], batch["pair_jk"]
            )
            t = model.triple_embedding(p_ij, p_ik, p_jk)
            z3 = model.pool_triples(t, batch["triple_graph"], int(stop - start))
            outputs.append(z3)
    if was_training:
        model.train()
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# the new model
# ---------------------------------------------------------------------------


class BondAnchoredTripleModel(nn.Module):
    """F (shared, both orderings) + re-initialised Reader + fixed buffers."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.F = nn.Sequential(
            nn.Linear(F_IN, F_HIDDEN, bias=True),
            nn.SiLU(),
            nn.Linear(F_HIDDEN, F_OUT, bias=True),
        )
        self.reader = nn.Sequential(
            nn.Linear(READER_IN, READER_HIDDEN[0], bias=True),
            nn.ReLU(),
            nn.Linear(READER_HIDDEN[0], READER_HIDDEN[1], bias=True),
            nn.ReLU(),
            nn.Linear(READER_HIDDEN[1], 1, bias=True),
        )
        self.register_buffer("mu_p", torch.zeros(PAIR_DIM))
        self.register_buffer("scale_p", torch.ones(PAIR_DIM))
        self.register_buffer("mu_old", torch.zeros(Z_OLD_DIM))
        self.register_buffer("scale_old", torch.ones(Z_OLD_DIM))
        self.register_buffer("mu_3", torch.zeros(TRIPLE_SUMMARY_DIM))
        self.register_buffer("scale_3", torch.ones(TRIPLE_SUMMARY_DIM))

    # -- parameter groups ------------------------------------------------------

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.F.parameters()) + list(self.reader.parameters())

    def load_standardizers(self, stats: Standardizers | Mapping[str, Any]) -> None:
        if isinstance(stats, Standardizers):
            payload = stats
        else:
            payload = Standardizers(
                mu_p=torch.as_tensor(stats["mu_p"], dtype=torch.float32),
                scale_p=torch.as_tensor(stats["scale_p"], dtype=torch.float32),
                mu_old=torch.as_tensor(stats["mu_old"], dtype=torch.float32),
                scale_old=torch.as_tensor(stats["scale_old"], dtype=torch.float32),
                mu_3=torch.as_tensor(stats["mu_3"], dtype=torch.float32),
                scale_3=torch.as_tensor(stats["scale_3"], dtype=torch.float32),
                near_constant=dict(stats.get("near_constant", {})),
            )
        with torch.no_grad():
            self.mu_p.copy_(payload.mu_p)
            self.scale_p.copy_(payload.scale_p)
            self.mu_old.copy_(payload.mu_old)
            self.scale_old.copy_(payload.scale_old)
            self.mu_3.copy_(payload.mu_3)
            self.scale_3.copy_(payload.scale_3)

    # -- forward pieces ---------------------------------------------------------

    @staticmethod
    def standardize(value: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (value - mu) / scale

    def standardize_pair_rows(
        self, p_ij: torch.Tensor, p_ik: torch.Tensor, p_jk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.standardize(p_ij, self.mu_p, self.scale_p),
            self.standardize(p_ik, self.mu_p, self.scale_p),
            self.standardize(p_jk, self.mu_p, self.scale_p),
        )

    def triple_embedding(
        self, p_ij: torch.Tensor, p_ik: torch.Tensor, p_jk: torch.Tensor
    ) -> torch.Tensor:
        a = torch.cat([p_ij, p_ik, p_jk], dim=-1)
        b = torch.cat([p_ij, p_jk, p_ik], dim=-1)
        return 0.5 * (self.F(a) + self.F(b))

    def pool_triples(
        self, t: torch.Tensor, triple_graph: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        width = int(t.shape[1]) if t.dim() > 1 else F_OUT
        total = torch.zeros((int(n_graphs), width), dtype=t.dtype, device=t.device)
        squared = torch.zeros_like(total)
        counts = torch.zeros((int(n_graphs), 1), dtype=t.dtype, device=t.device)
        if int(t.numel()):
            index = triple_graph.long()
            total.index_add_(0, index, t)
            squared.index_add_(0, index, t * t)
            counts.index_add_(
                0,
                index,
                torch.ones((int(index.numel()), 1), dtype=t.dtype, device=t.device),
            )
        denominator = counts.clamp_min(1.0)
        return torch.cat([total / denominator, squared / denominator], dim=-1)

    def forward(
        self,
        z_old: torch.Tensor,
        pair_ij: torch.Tensor,
        pair_ik: torch.Tensor,
        pair_jk: torch.Tensor,
        triple_graph: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        p_ij, p_ik, p_jk = self.standardize_pair_rows(pair_ij, pair_ik, pair_jk)
        t = self.triple_embedding(p_ij, p_ik, p_jk)
        z3 = self.pool_triples(t, triple_graph, int(n_graphs))
        z3_bar = self.standardize(z3, self.mu_3, self.scale_3)
        z_old_bar = self.standardize(z_old, self.mu_old, self.scale_old)
        unified = torch.cat([z_old_bar, z3_bar], dim=-1)
        return self.reader(unified).view(-1)


# ---------------------------------------------------------------------------
# batching over the cached per-graph features
# ---------------------------------------------------------------------------


def collate_graphs(cache: FeatureCache, indices: Sequence[int]) -> dict[str, torch.Tensor]:
    z_old = cache.z_old[list(indices)]
    y = cache.y[list(indices)]
    pair_ij: list[torch.Tensor] = []
    pair_ik: list[torch.Tensor] = []
    pair_jk: list[torch.Tensor] = []
    triple_graph: list[torch.Tensor] = []
    for local, index in enumerate(indices):
        lo = int(cache.triple_ptr[index])
        hi = int(cache.triple_ptr[index + 1])
        pair_ij.append(cache.triple_ij[lo:hi].long())
        pair_ik.append(cache.triple_ik[lo:hi].long())
        pair_jk.append(cache.triple_jk[lo:hi].long())
        triple_graph.append(torch.full((hi - lo,), local, dtype=torch.int64))
    return {
        "z_old": z_old,
        "y": y,
        "pair_ij": cache.pair_tokens[torch.cat(pair_ij, dim=0).long()] if pair_ij else torch.zeros(0, PAIR_DIM),
        "pair_ik": cache.pair_tokens[torch.cat(pair_ik, dim=0).long()] if pair_ik else torch.zeros(0, PAIR_DIM),
        "pair_jk": cache.pair_tokens[torch.cat(pair_jk, dim=0).long()] if pair_jk else torch.zeros(0, PAIR_DIM),
        "triple_graph": torch.cat(triple_graph, dim=0) if triple_graph else torch.zeros(0, dtype=torch.int64),
        "n_graphs": int(len(indices)),
    }


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, cache: FeatureCache) -> None:
        self.cache = cache

    def __len__(self) -> int:
        return self.cache.n_graphs

    def __getitem__(self, index: int) -> int:
        return int(index)


def make_collate(cache: FeatureCache):
    def _collate(indices: Sequence[int]) -> dict[str, torch.Tensor]:
        return collate_graphs(cache, [int(i) for i in indices])

    return _collate


def make_loader(
    cache: FeatureCache, batch_size: int, shuffle: bool, seed: int
) -> Any:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return torch.utils.data.DataLoader(
        FeatureDataset(cache),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=make_collate(cache),
    )


def evaluate(
    model: BondAnchoredTripleModel, loader: Any, device: Any
) -> dict[str, Any]:
    cpu_guard(device)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["z_old"].to(device),
                batch["pair_ij"].to(device),
                batch["pair_ik"].to(device),
                batch["pair_jk"].to(device),
                batch["triple_graph"].to(device),
                int(batch["n_graphs"]),
            )
            predictions.append(prediction.view(-1).cpu().numpy())
            targets.append(batch["y"].view(-1).cpu().numpy())
    target = np.concatenate(targets).astype(np.float64)
    pred = np.concatenate(predictions).astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(target - pred))),
        "n_molecules": int(target.shape[0]),
        "predictions": pred,
        "targets": target,
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_mainline(
    train_cache: FeatureCache,
    valid_cache: FeatureCache,
    stats: Mapping[str, Any],
    *,
    seed: int = 0,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    threads: int = THREADS,
    checkpoint_dir: Path | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Fixed 80-epoch mainline run; Top-5 soup over epochs 41-80."""
    torch.set_num_threads(int(threads))
    device = torch.device("cpu")
    cpu_guard(device)
    out_dir = Path(CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = BondAnchoredTripleModel(seed=int(seed))
    model.load_standardizers(stats)
    model.to(device)
    trainable = model.trainable_parameters()
    optimizer = torch.optim.Adam(
        trainable, lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY)
    )
    train_loader = make_loader(
        train_cache, batch_size, True, int(seed) + TRAIN_SHUFFLE_OFFSET
    )
    eval_loader = make_loader(
        valid_cache, batch_size, False, int(seed) + EVAL_SHUFFLE_OFFSET
    )

    curve: list[dict[str, Any]] = []
    top_states: dict[int, dict[str, torch.Tensor]] = {}
    best_mae = float("inf")
    best_epoch = 1
    rss_start = audit._rss_mb()
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_started = time.perf_counter()
        abs_sum = 0.0
        n_mol = 0
        for batch in train_loader:
            prediction = model(
                batch["z_old"].to(device),
                batch["pair_ij"].to(device),
                batch["pair_ik"].to(device),
                batch["pair_jk"].to(device),
                batch["triple_graph"].to(device),
                int(batch["n_graphs"]),
            )
            target = batch["y"].to(device)
            loss = F.l1_loss(prediction.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(GRAD_CLIP))
            optimizer.step()
            abs_sum += float((prediction.view(-1) - target.view(-1)).abs().sum())
            n_mol += int(target.numel())
        train_mae = float(abs_sum / max(n_mol, 1))
        valid = evaluate(model, eval_loader, device)
        epoch_seconds = float(time.perf_counter() - epoch_started)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "valid_mae": float(valid["mae"]),
                "epoch_seconds": epoch_seconds,
            }
        )
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
        eligible = int(epoch) >= SOUP_EPOCH_LO
        if eligible:
            top_states[int(epoch)] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                if k in {name for name, _ in model.named_parameters()}
            }
            ranked = sorted(
                top_states,
                key=lambda e: (float(next(r for r in curve if r["epoch"] == e)["valid_mae"]), e),
            )
            for stale in ranked[SOUP_K:]:
                del top_states[stale]
        if log and (epoch == 1 or epoch % 5 == 0 or epoch == int(epochs)):
            print(
                f"[BAT-v1-seed0] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={float(valid['mae']):.6f} best={best_mae:.6f}@{best_epoch} "
                f"({epoch_seconds:.1f}s)",
                flush=True,
            )
    wall = float(time.perf_counter() - started)
    if len(top_states) < SOUP_K:
        raise RuntimeError(
            f"only {len(top_states)} soup-eligible epoch states (need {SOUP_K})"
        )
    members = sorted(
        top_states,
        key=lambda e: (float(next(r for r in curve if r["epoch"] == e)["valid_mae"]), e),
    )
    param_names = [name for name, _ in model.named_parameters()]
    soup_params = {
        name: torch.stack([top_states[e][name].float() for e in members], dim=0).mean(dim=0)
        for name in param_names
    }
    soup_model = BondAnchoredTripleModel(seed=int(seed))
    soup_model.load_standardizers(stats)
    with torch.no_grad():
        for name, parameter in soup_model.named_parameters():
            parameter.copy_(soup_params[name])
    soup_model.eval()
    soup_valid = evaluate(soup_model, eval_loader, device)
    final_train_mae = float(curve[-1]["train_mae"])
    member_valid = [
        float(next(r for r in curve if r["epoch"] == e)["valid_mae"]) for e in members
    ]

    for e in members:
        torch.save(top_states[e], out_dir / f"BAT-v1-seed0_epoch{e:03d}_trainable_state.pt")
    torch.save(soup_params, out_dir / "BAT-v1-seed0_soup_trainable_state.pt")
    torch.save(soup_model.state_dict(), out_dir / "BAT-v1-seed0_soup_full_state.pt")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": "BAT-v1-seed0",
        "device": "cpu",
        "threads": int(threads),
        "seed": int(seed),
        "epoch_budget": int(epochs),
        "epochs_run": int(len(curve)),
        "completed": bool(len(curve) == int(epochs)),
        "batch_size": int(batch_size),
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "grad_clip": float(GRAD_CLIP),
        "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        "mask": MASK.as_dict(),
        "curve": curve,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "soup": {
            "members": [int(e) for e in members],
            "member_valid_mae": member_valid,
            "soup_valid_mae": float(soup_valid["mae"]),
            "soup_epoch_window": [int(SOUP_EPOCH_LO), int(SOUP_EPOCH_HI)],
        },
        "last_20_valid_mean": float(
            np.mean([row["valid_mae"] for row in curve[-20:]])
        ) if len(curve) >= 20 else float("nan"),
        "final_train_mae": final_train_mae,
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve), 1)),
        "rss_start_mb": float(rss_start),
        "rss_end_mb": float(audit._rss_mb()),
        "peak_rss_mb": float(audit._peak_rss_mb()),
        "official_test_loaded": False,
    }
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# correctness helpers
# ---------------------------------------------------------------------------


def parent_freeze_report(parent: sem.SEM108Model) -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "all_requires_grad_false": bool(all(not p.requires_grad for p in parent.parameters())),
        "all_grad_none": bool(all(p.grad is None for p in parent.parameters())),
        "parent_state_sha256": audit.state_sha256(load_parent_state()),
        "checkpoint_sha256": _sha256_file(PARENT_SOUP_PATH),
    }
    official_test_blocker(payload)
    return payload


def hand_pooling_check() -> dict[str, Any]:
    """Tiny hand-computable mean / second moment, including empty tuples."""
    t = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64
    )
    graph = torch.tensor([0, 0, 1], dtype=torch.int64)
    model = BondAnchoredTripleModel(seed=0).double()
    z3 = model.pool_triples(t, graph, 3)
    hand_mean_0 = (t[0] + t[1]) / 2
    hand_sq_0 = (t[0] * t[0] + t[1] * t[1]) / 2
    max_err = max(
        float((z3[0, :2] - hand_mean_0).abs().max()),
        float((z3[0, 2:] - hand_sq_0).abs().max()),
        float((z3[1, :2] - t[2]).abs().max()),
        float((z3[1, 2:] - t[2] * t[2]).abs().max()),
        float(z3[2].abs().max()),
    )
    empty_raw = model.pool_triples(torch.zeros(0, 2, dtype=torch.float64), torch.zeros(0, dtype=torch.int64), 1)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "max_abs_error": float(max_err),
        "empty_summary_is_zero": bool(float(empty_raw.abs().max()) == 0.0),
        "passed": bool(max_err <= 1.0e-12 and float(empty_raw.abs().max()) == 0.0),
    }


def endpoint_swap_check(model: BondAnchoredTripleModel | None = None) -> dict[str, Any]:
    """Swapping the anchor endpoints swaps ``a`` and ``b``; ``t`` is invariant."""
    torch.manual_seed(7)
    p_ij = torch.rand(64, PAIR_DIM)
    p_ik = torch.rand(64, PAIR_DIM)
    p_jk = torch.rand(64, PAIR_DIM)
    m = BondAnchoredTripleModel(seed=0) if model is None else model
    m.eval()
    with torch.no_grad():
        t = m.triple_embedding(p_ij, p_ik, p_jk)
        t_swap = m.triple_embedding(p_ij, p_jk, p_ik)
    diff = float((t - t_swap).abs().max())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "max_abs_diff": diff,
        "passed": bool(diff == 0.0),
    }
    official_test_blocker(payload)
    return payload


def batch_pooling_check(cache: FeatureCache, model: BondAnchoredTripleModel | None = None) -> dict[str, Any]:
    """Batched graph pooling must equal a per-graph loop."""
    m = BondAnchoredTripleModel(seed=0) if model is None else model
    m.eval()
    indices = list(range(min(16, cache.n_graphs)))
    with torch.no_grad():
        batched = collate_graphs(cache, indices)
        p_ij, p_ik, p_jk = m.standardize_pair_rows(
            batched["pair_ij"], batched["pair_ik"], batched["pair_jk"]
        )
        t = m.triple_embedding(p_ij, p_ik, p_jk)
        z3_batch = m.pool_triples(t, batched["triple_graph"], len(indices))
        per_graph: list[torch.Tensor] = []
        for local, index in enumerate(indices):
            single = collate_graphs(cache, [index])
            a, b, c = m.standardize_pair_rows(
                single["pair_ij"], single["pair_ik"], single["pair_jk"]
            )
            tt = m.triple_embedding(a, b, c)
            per_graph.append(m.pool_triples(tt, single["triple_graph"], 1)[0])
        per_graph_tensor = torch.stack(per_graph, dim=0)
    diff = float((z3_batch - per_graph_tensor).abs().max())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "n_graphs": len(indices),
        "max_abs_diff": diff,
        "tolerance": 1.0e-5,
        "passed": bool(diff <= 1.0e-5),
    }
    official_test_blocker(payload)
    return payload


def triple_audit(cache: FeatureCache, brute_force_graphs: int = 32) -> dict[str, Any]:
    """Exact counts, cross-graph safety and a brute-force re-enumeration."""
    n_graphs = cache.n_graphs
    ptr = cache.triple_ptr
    tcount = ptr[1:] - ptr[:-1]
    n = cache.n_nodes
    m = cache.n_bonds
    expected = torch.where(n >= 3, m * (n - 2).clamp_min(0), torch.zeros_like(m))
    count_ok = bool(torch.equal(tcount, expected))
    pair_ptr = cache.pair_ptr
    pair_lo = pair_ptr[:-1][cache.triple_graph.long()]
    pair_hi = pair_ptr[1:][cache.triple_graph.long()]
    rows = torch.stack([cache.triple_ij, cache.triple_ik, cache.triple_jk], dim=1).long()
    in_range = bool(((rows >= pair_lo.unsqueeze(1)) & (rows < pair_hi.unsqueeze(1))).all())
    anchor_ok = True
    brute_force_ok = True
    for index in range(min(int(brute_force_graphs), n_graphs)):
        lo = int(pair_ptr[index])
        hi = int(pair_ptr[index + 1])
        local_index = torch.arange(lo, hi, dtype=torch.int64)
        local_index = local_index - lo
        # triangular reconstruction (all graphs in the cache are complete)
        nn = int(n[index])
        src, dst = [], []
        for a in range(nn):
            for b in range(a + 1, nn):
                src.append(a)
                dst.append(b)
        pair_index = torch.tensor([src, dst], dtype=torch.int64)
        bucket = cache.pair_bucket[lo:hi].to(torch.int64)
        anchors = torch.nonzero(bucket == 0, as_tuple=False).view(-1)
        if int(anchors.numel()) != int(m[index]):
            anchor_ok = False
        reference = graph_triple_rows(pair_index, bucket, nn)
        got_lo = int(ptr[index])
        got_hi = int(ptr[index + 1])
        got = torch.stack(
            [
                cache.triple_ij[got_lo:got_hi].long() - lo,
                cache.triple_ik[got_lo:got_hi].long() - lo,
                cache.triple_jk[got_lo:got_hi].long() - lo,
            ],
            dim=1,
        )
        expected_rows = torch.stack(
            [reference["ij"], reference["ik"], reference["jk"]], dim=1
        )
        if not torch.equal(got, expected_rows):
            brute_force_ok = False
        del local_index
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "n_graphs": int(n_graphs),
        "n_pairs": int(cache.n_pairs),
        "n_triples": int(cache.n_triples),
        "triple_count_rule_ok": count_ok,
        "no_out_of_range_rows": in_range,
        "anchor_counts_ok": bool(anchor_ok),
        "brute_force_enumeration_ok": bool(brute_force_ok),
        "zero_tuple_graphs": int((tcount == 0).sum()),
        "passed": bool(count_ok and in_range and anchor_ok and brute_force_ok),
    }
    official_test_blocker(payload)
    return payload


def one_batch_gradient_check(
    train_cache: FeatureCache, stats: Mapping[str, Any], seed: int = 0
) -> dict[str, Any]:
    """Finite loss/gradients; non-zero task gradients on every F and Reader weight."""
    model = BondAnchoredTripleModel(seed=int(seed))
    model.load_standardizers(stats)
    trainable = model.trainable_parameters()
    loader = make_loader(train_cache, BATCH_SIZE, False, 0)
    batch = next(iter(loader))
    model.train()
    prediction = model(
        batch["z_old"],
        batch["pair_ij"],
        batch["pair_ik"],
        batch["pair_jk"],
        batch["triple_graph"],
        int(batch["n_graphs"]),
    )
    target = batch["y"]
    loss = F.l1_loss(prediction.view(-1), target.view(-1))
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    all_finite = bool(
        torch.isfinite(loss).all()
        and torch.isfinite(prediction).all()
        and all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
    )
    nonzero = {
        name: float(grad.norm()) for name, grad in grads.items() if grad is not None
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "n_graphs": int(batch["n_graphs"]),
        "n_triples": int(batch["triple_graph"].numel()),
        "loss": float(loss.detach()),
        "prediction_rms": float(prediction.detach().pow(2).mean().sqrt()),
        "pair_bar_rms": float(
            model.standardize(batch["pair_ij"], model.mu_p, model.scale_p).pow(2).mean().sqrt()
        ),
        "z_old_bar_rms": float(
            model.standardize(batch["z_old"], model.mu_old, model.scale_old).pow(2).mean().sqrt()
        ),
        "grad_norms": nonzero,
        "min_grad_norm": float(min(nonzero.values())) if nonzero else 0.0,
        "max_grad_norm": float(max(nonzero.values())) if nonzero else 0.0,
        "total_grad_norm": float(
            torch.sqrt(sum(g.pow(2).sum() for g in grads.values() if g is not None))
        ),
        "all_finite": all_finite,
        "all_grads_nonzero": bool(nonzero and min(nonzero.values()) > 0.0),
        "optimizer_numel": int(sum(p.numel() for p in trainable)),
        "optimizer_names": [name for name, _ in model.named_parameters()],
    }
    payload["passed"] = bool(
        all_finite and payload["all_grads_nonzero"] and payload["optimizer_numel"] == EXPECTED_TRAINABLE
    )
    official_test_blocker(payload)
    return payload


def parameter_accounting(model: BondAnchoredTripleModel | None = None) -> dict[str, Any]:
    m = BondAnchoredTripleModel(seed=0) if model is None else model
    f_params = int(sum(p.numel() for p in m.F.parameters()))
    reader_params = int(sum(p.numel() for p in m.reader.parameters()))
    trainable = int(sum(p.numel() for p in m.trainable_parameters()))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "F_params": f_params,
        "reader_params": reader_params,
        "reader_old_params": int(PARENT_READER_PARAMS),
        "reader_increment": int(reader_params - PARENT_READER_PARAMS),
        "new_trainable_params": int(f_params + reader_params),
        "total_increment_vs_parent": int(f_params + reader_params - PARENT_READER_PARAMS),
        "total_increment": int(f_params + reader_params - PARENT_READER_PARAMS),
        "trainable_params": trainable,
        "parent_params": int(PARENT_TOTAL_PARAMS),
        "full_model_params": int(PARENT_TOTAL_PARAMS - PARENT_READER_PARAMS + f_params + reader_params),
        "expected": {
            "F_params": int(EXPECTED_F_PARAMS),
            "reader_params": int(EXPECTED_READER_PARAMS),
            "trainable_params": int(EXPECTED_TRAINABLE),
            "full_model_params": int(EXPECTED_FULL_MODEL),
        },
        "passed": bool(
            f_params == EXPECTED_F_PARAMS
            and reader_params == EXPECTED_READER_PARAMS
            and trainable == EXPECTED_TRAINABLE
            and int(PARENT_TOTAL_PARAMS - PARENT_READER_PARAMS + f_params + reader_params)
            == EXPECTED_FULL_MODEL
        ),
    }
    official_test_blocker(payload)
    return payload


def cache_alignment_check(
    parent: sem.SEM108Model, cache: FeatureCache, split: str = "valid", n_graphs: int = 32
) -> dict[str, Any]:
    """Cached ``z_old`` through the original Reader vs the direct parent forward."""
    data_list = _split_data(split)[: int(n_graphs)]
    fresh: list[dict[str, Any]] = []
    for data in data_list:
        batch = p1.env_collate([data])
        capture: dict[str, list[torch.Tensor]] = {"pair_value": [], "z_old": []}
        handles = [
            parent.pair_encoder.register_forward_hook(
                lambda _m, _i, o, c=capture: c["pair_value"].append(o.detach().clone())
            ),
            parent.reader.register_forward_hook(
                lambda _m, i, _o, c=capture: c["z_old"].append(i[0].detach().clone())
            ),
        ]
        try:
            with torch.no_grad():
                prediction = parent(batch, mask=MASK)
        finally:
            for handle in handles:
                handle.remove()
        fresh.append(
            {
                "pair": capture["pair_value"][0].cpu(),
                "z_old": capture["z_old"][0].cpu(),
                "prediction": float(prediction.view(-1)[0]),
                "y": float(data.y.view(-1)[0]),
            }
        )
    max_pair_diff = 0.0
    max_z_old_diff = 0.0
    max_pred_diff = 0.0
    for index, row in enumerate(fresh):
        lo = int(cache.pair_ptr[index])
        hi = int(cache.pair_ptr[index + 1])
        max_pair_diff = max(
            max_pair_diff, float((cache.pair_tokens[lo:hi] - row["pair"]).abs().max())
        )
        max_z_old_diff = max(
            max_z_old_diff, float((cache.z_old[index] - row["z_old"]).abs().max())
        )
        cached_reader_pred = float(
            parent.reader(cache.z_old[index : index + 1].clone()).view(-1)[0]
        )
        max_pred_diff = max(max_pred_diff, abs(cached_reader_pred - row["prediction"]))
    target = np.asarray([row["y"] for row in fresh], dtype=np.float64)
    cached_y = cache.y[: len(fresh)].detach().cpu().numpy().astype(np.float64)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "split": split,
        "n_graphs": int(len(fresh)),
        "pair_tokens_bit_identical": bool(max_pair_diff == 0.0),
        "max_pair_token_diff": max_pair_diff,
        "max_z_old_diff": max_z_old_diff,
        "max_reader_prediction_diff": max_pred_diff,
        "targets_bit_identical": bool(np.array_equal(target, cached_y)),
        "passed": bool(
            max_pair_diff <= 2.0e-6
            and max_z_old_diff <= 1.0e-5
            and max_pred_diff <= 1.0e-6
            and np.array_equal(target, cached_y)
        ),
    }
    official_test_blocker(payload)
    return payload


def run_correctness_suite(
    *,
    train_cache: FeatureCache | None = None,
    valid_cache: FeatureCache | None = None,
    stats: Mapping[str, Any] | None = None,
    parent: sem.SEM108Model | None = None,
) -> dict[str, Any]:
    """All pre-registered focused checks on real data (cache required)."""
    parent = build_parent() if parent is None else parent
    checks: dict[str, Any] = {
        "parent_replay": replay_parent_valid_mae(parent),
        "parent_freeze": parent_freeze_report(parent),
        "parameter_accounting": parameter_accounting(),
        "hand_pooling": hand_pooling_check(),
        "endpoint_swap": endpoint_swap_check(),
    }
    if train_cache is not None and valid_cache is not None and stats is not None:
        checks["cache_alignment"] = cache_alignment_check(parent, valid_cache, "valid", n_graphs=16)
        checks["triple_audit_train"] = triple_audit(train_cache)
        checks["triple_audit_valid"] = triple_audit(valid_cache)
        checks["batch_pooling"] = batch_pooling_check(train_cache)
        checks["one_batch_gradient"] = one_batch_gradient_check(train_cache, stats)
    passed = all(
        bool(value.get("passed", False))
        for key, value in checks.items()
        if key not in ("parent_freeze",)
    ) and bool(checks["parent_freeze"]["all_requires_grad_false"]) and bool(
        checks["parent_freeze"]["all_grad_none"]
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "checks": checks,
        "all_passed": bool(passed),
    }
    official_test_blocker(payload)
    return payload


__all__ = [
    "PROTOCOL_VERSION",
    "RESULTS_DIR",
    "CACHE_DIR",
    "CHECKPOINT_DIR",
    "PREREG_PATH",
    "PARENT_SOUP_PATH",
    "PARENT_SOUP_SHA256",
    "PARENT_HISTORICAL_SOUP_MAE",
    "PAIR_DIM",
    "Z_OLD_DIM",
    "TRIPLE_SUMMARY_DIM",
    "EXPECTED_F_PARAMS",
    "EXPECTED_READER_PARAMS",
    "EXPECTED_TRAINABLE",
    "EXPECTED_FULL_MODEL",
    "EPOCHS",
    "SOUP_EPOCH_LO",
    "SOUP_EPOCH_HI",
    "MASK",
    "FeatureCache",
    "Standardizers",
    "BondAnchoredTripleModel",
    "graph_triple_rows",
    "extract_split_features",
    "build_or_load_cache",
    "cache_path",
    "cache_manifest_path",
    "fit_standardizer",
    "fit_standardizers",
    "compute_raw_z3",
    "collate_graphs",
    "make_loader",
    "evaluate",
    "train_mainline",
    "load_parent_state",
    "build_parent",
    "parent_provenance",
    "parent_predictions",
    "replay_parent_valid_mae",
    "parent_freeze_report",
    "hand_pooling_check",
    "endpoint_swap_check",
    "batch_pooling_check",
    "triple_audit",
    "one_batch_gradient_check",
    "parameter_accounting",
    "cache_alignment_check",
    "run_correctness_suite",
    "official_test_blocker",
    "module_sha256",
]
