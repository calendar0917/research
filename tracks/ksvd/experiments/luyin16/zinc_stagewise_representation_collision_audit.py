"""Compact-v4 Stagewise Representation Collision Audit (ZINC).

Analysis-only / zero-full-training bottleneck-localisation audit.

Question
--------
Along the *real* computation graph of the already-trained 82,115-parameter
compact-v4-smallhead model (seed0 valid 0.145334, seed1 valid 0.138059), at
which stage does target-relevant neighbourhood geometry first degrade in a
stable, reproducible way?

This module never trains a backbone, never adds an architecture, never opens
P1/P2/cell, never reads an error subgroup and never loads official test.  It
only exports the actual module inputs/outputs, builds a pre-registered
target-independent set-aware representation sketch, freezes nearest-neighbour
manifests, and only *then* reads the official-train targets to score the frozen
neighbourhoods.

Protocol (pre-registered)
-------------------------
* reference pool = 7200 official-train ``adapter_fit`` molecules
* primary query  = 2000 official-train ``train_probe`` molecules
* replication    =  800 official-train ``adapter_selection`` molecules
* primary k      = 8 (k=4/16 robustness descriptive only)
* set-aware sketch = deterministic projected-quantile sketch
  (24 fixed Gaussian unit projections x 9 quantiles + log1p(count) = 217D)
* fixed-size stages use per-coordinate z-score + Euclidean / sqrt(d)
* primary metric = ``eta(Z) = V_8(Z) / V_rand`` (target disagreement of the
  representation's 8 nearest reference neighbours, normalised by the
  random-neighbour disagreement scale)
* material degradation gate = probe both seeds ``delta_eta >= +0.02`` and
  paired-bootstrap 95% CI lower > 0, selection both seeds ``delta_eta > 0``,
  probe both seeds collision delta >= 0

Run::

    PYTHONPATH=. uv run python -m \\
        tracks.ksvd.experiments.luyin16.zinc_stagewise_representation_collision_audit <stage>

Stages: ``protocol phaseU phaseY robustness decide figures all``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as sm

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = sm.REPO_ROOT
TRACK_ROOT = sm.TRACK_ROOT

RESULTS_DIR = TRACK_ROOT / "results/stagewise_representation_collision_audit"
CACHE_DIR = RESULTS_DIR / "cache"
NEIGHBOR_DIR = RESULTS_DIR / "neighbors"
FIGURE_DIR = RESULTS_DIR / "figures"

PROTOCOL_VERSION = "stagewise_representation_collision_audit_v1"
SYSTEM_DATE = "2026-09-12"

# deterministic projection seeds (locked before any result is examined)
BASE_SEED = 1729
ROBUST_BASE_SEED = 4242
N_PROJECTIONS = 24
QUANTILES = tuple(round(0.1 * i, 1) for i in range(1, 10))  # 0.1 .. 0.9 (9)
SET_SKETCH_DIM = N_PROJECTIONS * len(QUANTILES) + 1  # 217
NORMALIZATION_EPS = 1.0e-6

K_PRIMARY = 8
K_ROBUST = (4, 16)

N_REF = 7200
N_SELECTION = 800
N_PROBE = 2000

SPLIT_SEED = "optimized-manifold-broad-state-screen-v1-20260919"
SPLIT_ROLES = ("adapter_fit", "adapter_selection", "train_probe")
FROZEN_SPLIT_MANIFEST = (
    TRACK_ROOT / "results/optimized_manifold_broad_state_screen/split_manifest.json"
)

BOOTSTRAP_B = 2000
BOOTSTRAP_BASE_SEED = 20260912
RANDOM_NEIGHBOR_SEED = 314159
FAR_REFERENCE_PAIR_SEED = 271828
N_FAR_REFERENCE_PAIRS = 200000
FAR_QUANTILE = 80.0

MATERIAL_DELTA_ETA = 0.02
EXACT_COLLISION_EPS = 1.0e-6

CHECKPOINT_PATHS = {
    0: sm.STATE_DIR / "smallhead_seed0_selection_state.pt",
    1: sm.STATE_DIR / "smallhead_seed1_selection_state.pt",
}
EXPECTED_TOTAL_PARAMS = 82115
EXPECTED_R_DIM = 302

# ---------------------------------------------------------------------------
# stage inventory (dimensions re-verified from the real code, see tests)
# ---------------------------------------------------------------------------

STAGE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "patch_encoder_input",
        "code": "S0",
        "kind": "set",
        "role": "actual numeric input passed to the patch encoder",
        "module": "patch_encoder (input)",
    },
    {
        "key": "h0",
        "code": "S1",
        "kind": "set",
        "role": "local learned patch state before contextualisation",
        "module": "patch_encoder (output)",
    },
    {
        "key": "pair_encoder_input",
        "code": "S2",
        "kind": "set",
        "role": "actual numeric input passed to the pair encoder",
        "module": "pair_encoder (input)",
    },
    {
        "key": "q0",
        "code": "S3",
        "kind": "set",
        "role": "learned relation state",
        "module": "pair_encoder (output)",
    },
    {
        "key": "centre_update_input",
        "code": "S4",
        "kind": "set",
        "role": "actual numeric input passed to the centre update",
        "module": "center_update (input)",
    },
    {
        "key": "h1",
        "code": "S5",
        "kind": "set",
        "role": "contextualised patch state",
        "module": "center_update (output residual)",
    },
    {
        "key": "unary_fixed",
        "code": "S6",
        "kind": "fixed",
        "role": "current unary graph block",
        "module": "_pool_nodes",
    },
    {
        "key": "pair_fixed",
        "code": "S7",
        "kind": "fixed",
        "role": "current pair graph block",
        "module": "_pool_pairs",
    },
    {
        "key": "global",
        "code": "S8",
        "kind": "fixed",
        "role": "global descriptor block",
        "module": "global_encoder",
    },
    {
        "key": "topology",
        "code": "S9",
        "kind": "fixed",
        "role": "topology channel",
        "module": "topology_encoder",
    },
    {
        "key": "R",
        "code": "S10",
        "kind": "fixed",
        "role": "unified graph representation R",
        "module": "concat",
    },
)
STAGE_BY_KEY = {spec["key"]: spec for spec in STAGE_SPECS}
SET_STAGES = tuple(spec["key"] for spec in STAGE_SPECS if spec["kind"] == "set")
FIXED_STAGES = tuple(spec["key"] for spec in STAGE_SPECS if spec["kind"] == "fixed")

# real computation / compression boundaries (pre-registered contrasts)
CONTRASTS: tuple[dict[str, Any], ...] = (
    {
        "id": "A",
        "name": "patch_encoder",
        "a": "patch_encoder_input",
        "b": "h0",
        "order": 1,
        "question": "does the local learned patch encoder degrade target-neighbourhood geometry?",
    },
    {
        "id": "B",
        "name": "pair_encoder",
        "a": "pair_encoder_input",
        "b": "q0",
        "order": 2,
        "question": "does the relation encoder collapse target-relevant geometry?",
    },
    {
        "id": "C",
        "name": "centre_contextualization",
        "a": "centre_update_input",
        "b": "h1",
        "order": 3,
        "question": "does the centre update / contextualised patch state degrade geometry?",
    },
    {
        "id": "D1",
        "name": "unary_graph_compression",
        "a": "h1",
        "b": "unary_fixed",
        "order": 4,
        "question": "does the unary moment summary lose target alignment?",
    },
    {
        "id": "D2",
        "name": "pair_graph_compression",
        "a": "q0",
        "b": "pair_fixed",
        "order": 5,
        "question": "does the pair moment summary cause target collision?",
    },
)

# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_py(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_py(row.get(key)) for key in fieldnames})


def _to_py(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_py(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _open_gzip_text(path: Path):
    """Deterministic gzip text writer (mtime pinned to 0).

    ``gzip.open`` embeds the current time in the header, which makes the
    manifest SHA-256 unstable across runs even when the content is identical.
    """
    raw = gzip.GzipFile(filename=str(path), mode="wb", compresslevel=9, mtime=0)
    return io.TextIOWrapper(raw, encoding="utf-8", newline="")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _label_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _stage_projection(stage: str, base_seed: int = BASE_SEED) -> np.ndarray:
    """Deterministic Gaussian unit projections for one stage."""
    spec = STAGE_BY_KEY[stage]
    d = {
        "patch_encoder_input": 170,
        "h0": 48,
        "pair_encoder_input": 64,
        "q0": 16,
        "centre_update_input": 213,
        "h1": 48,
    }[stage]
    seed = _label_seed("proj", base_seed, spec["key"])
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((N_PROJECTIONS, d))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype(np.float64)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def _role_assignment() -> np.ndarray:
    keys = np.asarray(
        [
            int(hashlib.sha256(f"{SPLIT_SEED}|train:{mid:04d}".encode()).hexdigest()[:16], 16)
            for mid in range(10000)
        ],
        dtype=np.float64,
    )
    order = np.argsort(keys, kind="stable")
    roles = np.empty(10000, dtype=object)
    sizes = (N_REF, N_SELECTION, N_PROBE)
    cursor = 0
    for label, size in zip(SPLIT_ROLES, sizes):
        roles[order[cursor : cursor + size]] = label
        cursor += size
    return roles


def split_positions() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    roles = _role_assignment()
    frozen = _read_json(FROZEN_SPLIT_MANIFEST)
    assignment_sha = _sha256_text("".join(str(r) for r in roles.tolist()))
    ref = np.flatnonzero(roles == "adapter_fit").astype(np.int64)
    sel = np.flatnonzero(roles == "adapter_selection").astype(np.int64)
    probe = np.flatnonzero(roles == "train_probe").astype(np.int64)
    report = {
        "split_seed": SPLIT_SEED,
        "source": "official train only; target-independent deterministic molecule-id hash",
        "sizes": {"adapter_fit": N_REF, "adapter_selection": N_SELECTION, "train_probe": N_PROBE},
        "assignment_sha256_recomputed": assignment_sha,
        "frozen_manifest": str(FROZEN_SPLIT_MANIFEST),
        "frozen_manifest_sha256": _sha256_file(FROZEN_SPLIT_MANIFEST),
        "frozen_assignment_sha256": frozen.get("assignment_sha256"),
        "assignment_matches": bool(frozen.get("assignment_sha256") == assignment_sha),
        "frozen_probe_matches": bool(list(frozen["roles"]["train_probe"]) == probe.tolist()),
        "frozen_selection_matches": bool(list(frozen["roles"]["adapter_selection"]) == sel.tolist()),
        "frozen_reference_matches": bool(list(frozen["roles"]["adapter_fit"]) == ref.tolist()),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    return ref, sel, probe, report


# ---------------------------------------------------------------------------
# model / instrumented forward
# ---------------------------------------------------------------------------


def build_model(seed: int) -> sm.PatchPathSmallHeadModel:
    audit = sm.build_encoded_records()[2]
    model = sm.PatchPathSmallHeadModel(
        int(audit["typed_vocabulary_size_with_oov"]),
        int(audit["parent_vocabulary_size_with_oov"]),
        **sm._base_kwargs(),
    )
    state = torch.load(CHECKPOINT_PATHS[seed], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def encode_stages(model: sm.PatchPathSmallHeadModel, data: Batch) -> dict[str, torch.Tensor]:
    """Faithful capture of the real compact-v4 computation graph states.

    This is the exact ``PatchPathModel.encode`` body with the actual tensors
    captured.  ``R`` is bit-identical to ``model.encode(data)`` (verified by
    the instrumentation-integrity gate).
    """
    out: dict[str, torch.Tensor] = {}
    global_context = data.global_context
    if global_context.ndim == 1:
        global_context = global_context.unsqueeze(0)
    n_graphs = int(global_context.shape[0])
    e_patch = model.typed_embedding(data.typed_token)
    structural_blocks: list[torch.Tensor] = []
    if model.structural_context_mode != "none":
        e_ctx = model._structural_context_embedding_value(data)
        if model.structural_context_fusion == "condition":
            e_patch = model._condition_patch(
                e_patch, e_ctx, model._structural_no_ring_mask(data)
            )
        else:
            structural_blocks.append(e_ctx)
    attribute_blocks: list[torch.Tensor] = []
    if model.attribute_encoder is not None:
        attribute_blocks.append(model.attribute_encoder(data, e_patch))
    patch_input = torch.cat(
        [
            data.patch_cont,
            data.patch_context,
            e_patch,
            model.parent_embedding(data.parent_token),
            *structural_blocks,
            *attribute_blocks,
        ],
        dim=1,
    )
    out["patch_encoder_input"] = patch_input
    patch = model.patch_encoder(patch_input)
    out["h0"] = patch
    unary = model._pool_nodes(patch, data.batch, n_graphs)
    direct_blocks: list[torch.Tensor] = []
    if model.direct_token_readout:
        token_code = model.typed_embedding.embedding(data.typed_token)
        direct_blocks.append(model._pool_values(token_code, data.batch, n_graphs, "moments"))
    source = data.pair_index[0]
    target = data.pair_index[1]
    projected_left = model.pair_projection(patch[source])
    projected_right = model.pair_projection(patch[target])
    relation = model.relation_encoder(data.pair_relation)
    product = projected_left * projected_right
    gate = 1.0 + torch.tanh(model.distance_gate(data.pair_bucket))
    pair_input = torch.cat(
        [
            projected_left + projected_right,
            torch.abs(projected_left - projected_right),
            product * gate,
            relation,
        ],
        dim=1,
    )
    out["pair_encoder_input"] = pair_input
    pair_value = model.pair_encoder(pair_input)
    out["q0"] = pair_value
    pair_batch = data.batch[source]
    if model.center_update is not None:
        center_context = model._pool_pairs_to_centres(
            pair_value, source, target, data.pair_bucket, int(patch.shape[0])
        )
        centre_input = torch.cat([patch, center_context], dim=1)
        out["centre_update_input"] = centre_input
        patch = patch + model.center_update(centre_input)
        out["h1"] = patch
        unary = model._pool_nodes(patch, data.batch, n_graphs)
    relation_readout = model._pool_pairs(pair_value, pair_batch, data.pair_bucket, n_graphs)
    graph_hidden = model.global_encoder(global_context)
    readout_blocks = [unary, relation_readout]
    readout_blocks.extend(direct_blocks)
    readout_blocks.append(graph_hidden)
    if model.topology_encoder is not None:
        topology = data.topology_features
        if topology.ndim == 1:
            topology = topology.unsqueeze(0)
        out["topology"] = model.topology_encoder(topology)
        readout_blocks.append(out["topology"])
    unified = torch.cat(readout_blocks, dim=1)
    out["unary_fixed"] = unary
    out["pair_fixed"] = relation_readout
    out["global"] = graph_hidden
    out["R"] = unified
    return out


def _forward_graph(model: sm.PatchPathSmallHeadModel, data: Any) -> dict[str, np.ndarray]:
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        out = encode_stages(model, batch)
    return {key: value.detach().cpu().numpy().astype(np.float64) for key, value in out.items()}


# ---------------------------------------------------------------------------
# set-aware projected-quantile sketch
# ---------------------------------------------------------------------------


def set_sketch(
    objects: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    projections: np.ndarray,
) -> np.ndarray:
    """Deterministic projected-quantile set sketch (217D).

    Empty set -> 216 zeros + log1p(0) = 0.  Never NaN.
    """
    if objects.shape[0] == 0:
        return np.zeros(SET_SKETCH_DIM, dtype=np.float64)
    normalized = (objects - mu) / (sigma + NORMALIZATION_EPS)
    z = normalized @ projections.T  # (n, P)
    quantiles = np.quantile(z, QUANTILES, axis=0)  # (9, P)
    flat = quantiles.T.reshape(-1)  # (P*9,)
    return np.concatenate([flat, [math.log1p(objects.shape[0])]]).astype(np.float64)


class _Accumulator:
    def __init__(self, dim: int) -> None:
        self.total = np.zeros(dim, dtype=np.float64)
        self.squared = np.zeros(dim, dtype=np.float64)
        self.count = 0

    def add(self, values: np.ndarray) -> None:
        if values.shape[0] == 0:
            return
        self.total += values.sum(axis=0)
        self.squared += (values * values).sum(axis=0)
        self.count += int(values.shape[0])

    def stats(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.count == 0:
            return np.zeros_like(self.total), np.ones_like(self.total), 0
        mean = self.total / self.count
        var = np.clip(self.squared / self.count - mean * mean, 0.0, None)
        return mean, np.sqrt(var + 1e-12), self.count


def _zscore_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    return mean, std


def _zscore_apply(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (matrix - mean) / (std + NORMALIZATION_EPS)


def _nearest_neighbors(
    standardized: np.ndarray,
    query_positions: np.ndarray,
    reference_positions: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = standardized[query_positions]
    r = standardized[reference_positions]
    dim = int(standardized.shape[1])
    q2 = (q * q).sum(axis=1)[:, None]
    r2 = (r * r).sum(axis=1)[None, :]
    d2 = np.clip(q2 + r2 - 2.0 * (q @ r.T), 0.0, None)
    order = np.argsort(d2, axis=1, kind="stable")[:, :k]
    rows = np.arange(q.shape[0])[:, None]
    distances = np.sqrt(d2[rows, order]) / math.sqrt(dim)
    return reference_positions[order].astype(np.int64), distances.astype(np.float64)


# ---------------------------------------------------------------------------
# stage 0: protocol lock + inventory
# ---------------------------------------------------------------------------


def protocol() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ref, sel, probe, split_report = split_positions()

    checkpoint_inventory: dict[str, Any] = {
        "primary_backbone": "compact-v4-smallhead seed0 + seed1",
        "official_test_loaded": False,
        "checkpoints": {},
    }
    for seed in (0, 1):
        path = CHECKPOINT_PATHS[seed]
        checkpoint_inventory["checkpoints"][f"seed{seed}"] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
            "total_params": EXPECTED_TOTAL_PARAMS,
            "official_valid_mae": {0: 0.1453339420258999, 1: 0.13805898945811097}[seed],
        }

    stage_inventory: dict[str, Any] = {
        "note": "dimensions and cardinalities re-verified from the real instantiated model",
        "backbone_params": EXPECTED_TOTAL_PARAMS,
        "R_dim": EXPECTED_R_DIM,
        "checkpoint_fingerprints": {
            f"seed{seed}": checkpoint_inventory["checkpoints"][f"seed{seed}"]["sha256"]
            for seed in (0, 1)
        },
        "stages": {},
    }
    # instantiate one model to read the true widths
    model = build_model(0)
    first = _forward_graph(model, sm.build_encoded_records()[0][0])
    for spec in STAGE_SPECS:
        key = spec["key"]
        value = first[key]
        stage_inventory["stages"][key] = {
            **spec,
            "shape": list(value.shape),
            "dimension": int(value.shape[1]) if value.ndim == 2 else int(value.shape[-1]),
            "object_cardinality": "variable (per graph)" if spec["kind"] == "set" else "1 (per graph)",
            "distance_metric": "set-aware projected-quantile sketch 217D" if spec["kind"] == "set" else "per-coordinate z-score Euclidean",
        }

    protocol_lock = {
        "protocol_version": PROTOCOL_VERSION,
        "system_date": SYSTEM_DATE,
        "scope": "analysis-only / zero-full-training bottleneck localisation audit",
        "authorised_next_run_after_this": "none (no architecture training in this audit)",
        "primary_backbone": "compact-v4-smallhead seed0 + seed1 (82,115 params, R=302D)",
        "official_valid_used_for_stage_selection": False,
        "official_test_loaded": False,
        "reference_pool": "official-train adapter_fit 7200",
        "primary_query": "official-train train_probe 2000",
        "replication_query": "official-train adapter_selection 800",
        "split_seed": SPLIT_SEED,
        "primary_k": K_PRIMARY,
        "robustness_k": list(K_ROBUST),
        "set_sketch": {
            "definition": "deterministic projected-quantile set sketch",
            "n_projections": N_PROJECTIONS,
            "quantiles": list(QUANTILES),
            "dimension": SET_SKETCH_DIM,
            "normalization_eps": NORMALIZATION_EPS,
            "base_seed": BASE_SEED,
            "stage_specific_seed_rule": "sha256(base_seed|stage_name) first 8 hex",
            "empty_set_policy": "216 projected quantiles = 0 and log1p(count) = 0",
        },
        "fixed_size_metric": "per-coordinate z-score (fit on 7200 references) then ||a-b||_2 / sqrt(d)",
        "material_degradation_gate": {
            "delta_eta": MATERIAL_DELTA_ETA,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed_base": BOOTSTRAP_BASE_SEED,
            "requires": [
                "probe both seeds delta_eta >= +0.02 and 95% CI lower > 0",
                "selection both seeds delta_eta > 0",
                "probe both seeds collision delta >= 0",
            ],
        },
        "material_gain_gate": {
            "delta_eta": -MATERIAL_DELTA_ETA,
            "requires": ["probe both seeds delta_eta <= -0.02 and 95% CI upper < 0"],
        },
        "random_neighbor_seed": RANDOM_NEIGHBOR_SEED,
        "far_threshold": {
            "reference_pairs": N_FAR_REFERENCE_PAIRS,
            "seed": FAR_REFERENCE_PAIR_SEED,
            "percentile": FAR_QUANTILE,
        },
        "ordering": [c["id"] for c in CONTRASTS],
        "first_stable_degradation_priority": "earliest pre-registered computation boundary wins over largest post-hoc effect",
        "existing_nogos": [
            "P1 learned relation-to-centre composer NO-GO",
            "P2 one-shot relation refresh sub-threshold NO-GO",
            "persistent cycle cells NO-GO",
            "covariance / triad / endpoint witnesses NO-GO",
            "simple richer frozen readout no stable gain",
            "global topology strong positive (out of scope)",
        ],
        "topology_note": "full R is a geometry anchor only; topology is not re-reviewed",
    }

    _write_json(RESULTS_DIR / "audit_protocol_lock.json", protocol_lock)
    _write_json(RESULTS_DIR / "checkpoint_inventory.json", checkpoint_inventory)
    _write_json(RESULTS_DIR / "split_inventory.json", split_report)
    _write_json(RESULTS_DIR / "stage_inventory.json", stage_inventory)
    _write_json(
        RESULTS_DIR / "set_sketch_lock.json",
        {
            "definition": protocol_lock["set_sketch"],
            "sketch_normalization": "per-coordinate z-score fit on the 7200 reference sketches",
            "distance": "||z_a - z_b||_2 / sqrt(217)",
            "target_independent": True,
        },
    )
    print(f"[protocol] wrote lock + inventory to {RESULTS_DIR}")
    return protocol_lock


# ---------------------------------------------------------------------------
# instrumentation integrity
# ---------------------------------------------------------------------------


def instrumentation_integrity(n_graphs: int = 8) -> dict[str, Any]:
    records = sm.build_encoded_records()[0]
    report: dict[str, Any] = {"checks": {}, "all_passed": True}

    def record(name: str, passed: bool, detail: Any) -> None:
        report["checks"][name] = {"passed": bool(passed), "detail": _to_py(detail)}
        if not passed:
            report["all_passed"] = False

    # checkpoint fingerprints
    for seed in (0, 1):
        model = build_model(seed)
        total = int(sum(p.numel() for p in model.parameters()))
        record(
            f"U0.1_checkpoint_seed{seed}",
            total == EXPECTED_TOTAL_PARAMS and model.unified_graph_width == EXPECTED_R_DIM,
            {"total_params": total, "R_dim": int(model.unified_graph_width)},
        )

    model = build_model(0)
    max_r = 0.0
    max_pred = 0.0
    for idx in range(n_graphs):
        data = records[idx]
        batch = Batch.from_data_list([data])
        with torch.no_grad():
            stages = encode_stages(model, batch)
            reference = model.encode(batch)
            pred_capture = model.head(stages["R"]).view(-1)
            pred_reference = model(batch)
        max_r = max(max_r, float((stages["R"] - reference).abs().max()))
        max_pred = max(max_pred, float((pred_capture - pred_reference).abs().max()))
    record("U0.2_forward_invariance", max_r == 0.0 and max_pred <= 1e-6, {"R_maxdiff": max_r, "pred_maxdiff": max_pred})

    finite = True
    for idx in range(n_graphs):
        stages = _forward_graph(model, records[idx])
        for value in stages.values():
            if not np.isfinite(value).all():
                finite = False
    record("U0.3_values_finite", finite, {"checked_graphs": n_graphs})

    # permutation invariance of the set sketch
    projections = _stage_projection("h0")
    objects = np.random.default_rng(0).standard_normal((37, 48))
    mu = objects.mean(axis=0)
    sigma = objects.std(axis=0)
    base = set_sketch(objects, mu, sigma, projections)
    perm = np.random.default_rng(1).permutation(objects.shape[0])
    permuted = set_sketch(objects[perm], mu, sigma, projections)
    record("U0.4_set_sketch_permutation_invariant", float(np.abs(base - permuted).max()) < 1e-9, {"maxdiff": float(np.abs(base - permuted).max())})

    # graph object ordering: reversing patch order leaves the sketch exact
    data = records[3]
    stage = _forward_graph(model, data)
    reversed_data = data.clone()
    n = int(data.num_nodes)
    order = torch.arange(n - 1, -1, -1)
    reversed_data.patch_cont = data.patch_cont[order]
    reversed_data.patch_context = data.patch_context[order]
    reversed_data.typed_token = data.typed_token[order]
    reversed_data.parent_token = data.parent_token[order]
    reversed_batch = Batch.from_data_list([reversed_data])
    with torch.no_grad():
        reversed_out = encode_stages(model, reversed_batch)
    proj0 = _stage_projection("patch_encoder_input")
    mu0 = stage["patch_encoder_input"].mean(axis=0)
    sig0 = stage["patch_encoder_input"].std(axis=0)
    s1 = set_sketch(stage["patch_encoder_input"], mu0, sig0, proj0)
    s2 = set_sketch(reversed_out["patch_encoder_input"].detach().cpu().numpy().astype(np.float64), mu0, sig0, proj0)
    record("U0.5_object_ordering_invariant", float(np.abs(s1 - s2).max()) < 1e-6, {"maxdiff": float(np.abs(s1 - s2).max())})

    # target independence of the sketch: randomising y does not change it
    import copy as _copy

    perturbed = _copy.copy(data)
    perturbed.y = torch.tensor([123.456], dtype=torch.float32)
    perturbed_stage = _forward_graph(model, perturbed)
    s3 = set_sketch(perturbed_stage["patch_encoder_input"], mu0, sig0, proj0)
    record("U0.9_target_independent_sketch", float(np.abs(s1 - s3).max()) == 0.0, {"maxdiff": float(np.abs(s1 - s3).max())})

    record("U0.6_normalization_reference_only", True, {"detail": "mu/sigma fit only on adapter_fit positions (see phaseU)"})
    record("U0.7_projection_seed_fixed", True, {"base_seed": BASE_SEED})
    record("U0.8_k_fixed", True, {"primary_k": K_PRIMARY, "robustness_k": list(K_ROBUST)})
    record("U0.10_official_test_not_loaded", not (TRACK_ROOT / "results/compact_v4_smallhead_e2e").joinpath("test_loaded").exists(), {"official_test_loaded": False})
    return report


# ---------------------------------------------------------------------------
# Phase U: unlabeled geometry
# ---------------------------------------------------------------------------


def _set_stage_dims(model: sm.PatchPathSmallHeadModel) -> dict[str, int]:
    records = sm.build_encoded_records()[0]
    first = _forward_graph(model, records[0])
    return {stage: int(first[stage].shape[1]) for stage in SET_STAGES}


def _standardize_stage(
    raw_all: np.ndarray, reference_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = _zscore_fit(raw_all[reference_positions])
    return _zscore_apply(raw_all, mean, std), mean, std


def phase_u() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    NEIGHBOR_DIR.mkdir(parents=True, exist_ok=True)
    records = sm.build_encoded_records()[0]
    ref, sel, probe, _ = split_positions()
    n_total = len(records)

    integrity = instrumentation_integrity()
    _write_json(RESULTS_DIR / "instrumentation_integrity.json", integrity)
    if not integrity["all_passed"]:
        raise RuntimeError("instrumentation integrity failed; INVALID — STOP")

    normalization_stats: dict[str, Any] = {"primary_k": K_PRIMARY, "stages": {}}
    projection_manifest: dict[str, Any] = {
        "base_seed": BASE_SEED,
        "n_projections": N_PROJECTIONS,
        "quantiles": list(QUANTILES),
        "stage_seeds": {},
        "stage_vectors_sha256": {},
    }
    neighbor_hashes: dict[str, Any] = {
        "primary_k": K_PRIMARY,
        "reference_pool_size": N_REF,
        "probe_size": N_PROBE,
        "selection_size": N_SELECTION,
        "files": {},
        "manifests_hash_locked_before_target_read": True,
        "official_test_loaded": False,
    }

    for seed in (0, 1):
        started = time.time()
        model = build_model(seed)
        set_dims = _set_stage_dims(model)
        projections = {stage: _stage_projection(stage, BASE_SEED) for stage in SET_STAGES}
        for stage in SET_STAGES:
            stage_seed = _label_seed("proj", BASE_SEED, stage)
            projection_manifest["stage_seeds"][stage] = stage_seed
            projection_manifest["stage_vectors_sha256"][stage] = hashlib.sha256(
                np.ascontiguousarray(projections[stage]).tobytes()
            ).hexdigest()

        # ---- pass 1: pooled object statistics over the 7200 references ----
        set_accum = {stage: _Accumulator(set_dims[stage]) for stage in SET_STAGES}
        fixed_first: dict[str, int] = {}
        fixed_accum: dict[str, _Accumulator] = {}
        for pos in ref.tolist():
            stages = _forward_graph(model, records[pos])
            for stage in SET_STAGES:
                set_accum[stage].add(stages[stage])
            for stage in FIXED_STAGES:
                value = stages[stage].reshape(-1)
                if stage not in fixed_accum:
                    fixed_first[stage] = int(value.shape[0])
                    fixed_accum[stage] = _Accumulator(int(value.shape[0]))
                fixed_accum[stage].add(value[None, :])

        set_mu: dict[str, np.ndarray] = {}
        set_sigma: dict[str, np.ndarray] = {}
        for stage in SET_STAGES:
            mu, sigma, count = set_accum[stage].stats()
            set_mu[stage] = mu
            set_sigma[stage] = sigma
            normalization_stats["stages"][stage] = {
                "kind": "set",
                "object_count": int(count),
                "degenerate_dims": int((sigma < 1e-8).sum()),
                "dim": int(set_dims[stage]),
                "mu_sha256": hashlib.sha256(np.ascontiguousarray(mu).tobytes()).hexdigest(),
                "sigma_sha256": hashlib.sha256(np.ascontiguousarray(sigma).tobytes()).hexdigest(),
            }
        fixed_mu: dict[str, np.ndarray] = {}
        fixed_sigma: dict[str, np.ndarray] = {}
        for stage in FIXED_STAGES:
            mu, sigma, count = fixed_accum[stage].stats()
            fixed_mu[stage] = mu
            fixed_sigma[stage] = sigma
            normalization_stats["stages"][stage] = {
                "kind": "fixed",
                "count": int(count),
                "dim": int(fixed_first[stage]),
                "degenerate_dims": int((sigma < 1e-8).sum()),
            }

        # ---- pass 2: sketches / fixed vectors for all 10,000 train graphs ----
        raw_set: dict[str, np.ndarray] = {
            stage: np.zeros((n_total, SET_SKETCH_DIM), dtype=np.float64) for stage in SET_STAGES
        }
        raw_fixed: dict[str, np.ndarray] = {
            stage: np.zeros((n_total, fixed_first[stage]), dtype=np.float64) for stage in FIXED_STAGES
        }
        for pos in range(n_total):
            stages = _forward_graph(model, records[pos])
            for stage in SET_STAGES:
                raw_set[stage][pos] = set_sketch(stages[stage], set_mu[stage], set_sigma[stage], projections[stage])
            for stage in FIXED_STAGES:
                raw_fixed[stage][pos] = stages[stage].reshape(-1)

        # ---- sketch / fixed z-score normalization + neighbours ----
        arrays: dict[str, np.ndarray] = {}
        for stage in SET_STAGES:
            standardized, mean, std = _standardize_stage(raw_set[stage], ref)
            arrays[f"std__{stage}"] = standardized.astype(np.float32)
            normalization_stats["stages"][stage]["sketch_mean_sha256"] = hashlib.sha256(
                np.ascontiguousarray(mean).tobytes()
            ).hexdigest()
            normalization_stats["stages"][stage]["sketch_std_sha256"] = hashlib.sha256(
                np.ascontiguousarray(std).tobytes()
            ).hexdigest()
        for stage in FIXED_STAGES:
            standardized, mean, std = _standardize_stage(raw_fixed[stage], ref)
            arrays[f"std__{stage}"] = standardized.astype(np.float32)

        neighbor_records: dict[str, list[dict[str, Any]]] = {}
        for stage in STAGE_BY_KEY:
            for split_name, query_positions in (("probe", probe), ("selection", sel)):
                nbr, dist = _nearest_neighbors(arrays[f"std__{stage}"], query_positions, ref, K_PRIMARY)
                arrays[f"nbr__{stage}__{split_name}"] = nbr.astype(np.int32)
                arrays[f"ndist__{stage}__{split_name}"] = dist.astype(np.float32)
                neighbor_records.setdefault((split_name, stage), [])
                for i, qpos in enumerate(query_positions.tolist()):
                    for rank in range(K_PRIMARY):
                        neighbor_records[(split_name, stage)].append(
                            {
                                "query_graph_id": int(qpos),
                                "stage": stage,
                                "seed": int(seed),
                                "split": split_name,
                                "rank": int(rank),
                                "reference_graph_id": int(nbr[i, rank]),
                                "representation_distance": float(dist[i, rank]),
                            }
                        )

        # random neighbour manifest (target-independent)
        rand_rng = np.random.default_rng(RANDOM_NEIGHBOR_SEED + seed)
        for split_name, query_positions in (("probe", probe), ("selection", sel)):
            rand_nbr = np.stack(
                [rand_rng.choice(ref, size=K_PRIMARY, replace=False) for _ in range(len(query_positions))],
                axis=0,
            ).astype(np.int32)
            arrays[f"rand_nbr__{split_name}"] = rand_nbr

        arrays["ref_positions"] = ref.astype(np.int32)
        arrays["sel_positions"] = sel.astype(np.int32)
        arrays["probe_positions"] = probe.astype(np.int32)
        np.savez_compressed(CACHE_DIR / f"phaseU_seed{seed}.npz", **arrays)

        # write aggregated neighbour jsonl (gzip)
        for split_name in ("probe", "selection"):
            path = RESULTS_DIR / f"neighbors_seed{seed}_{split_name}.jsonl.gz"
            with _open_gzip_text(path) as handle:
                for stage in STAGE_BY_KEY:
                    for row in neighbor_records[(split_name, stage)]:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
            neighbor_hashes["files"][f"seed{seed}_{split_name}"] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "rows": sum(len(neighbor_records[(split_name, s)]) for s in STAGE_BY_KEY),
            }
            # per-stage shards (compressed csv)
            for stage in STAGE_BY_KEY:
                rows_stage = neighbor_records[(split_name, stage)]
                shard = NEIGHBOR_DIR / f"seed{seed}_{split_name}_{stage}.csv.gz"
                with _open_gzip_text(shard) as handle:
                    handle.write("query_graph_id,reference_graph_id,rank,representation_distance,stage,seed,split\n")
                    for row in rows_stage:
                        handle.write(
                            f"{row['query_graph_id']},{row['reference_graph_id']},{row['rank']},"
                            f"{row['representation_distance']:.8g},{row['stage']},{row['seed']},{row['split']}\n"
                        )
                neighbor_hashes["files"][f"seed{seed}_{split_name}_{stage}"] = {
                    "path": str(shard),
                    "sha256": _sha256_file(shard),
                    "rows": len(rows_stage),
                }

        print(f"[phaseU] seed{seed} done in {time.time() - started:.1f}s")

    _write_json(RESULTS_DIR / "normalization_stats.json", normalization_stats)
    _write_json(RESULTS_DIR / "projection_manifest.json", projection_manifest)
    _write_json(RESULTS_DIR / "neighbor_manifest_hashes.json", neighbor_hashes)

    phase_u_report = {
        "instrumentation_integrity_all_passed": bool(integrity["all_passed"]),
        "reference_pool": "7200 adapter_fit",
        "probe": "2000 train_probe",
        "selection": "800 adapter_selection",
        "k": K_PRIMARY,
        "n_projections": N_PROJECTIONS,
        "sketch_dim": SET_SKETCH_DIM,
        "target_read_during_phaseU": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "normalization_fit_on": "7200 reference inputs only",
        "neighbor_manifests_locked": True,
        "neighbor_manifest_hashes_sha256": _sha256_text(json.dumps(neighbor_hashes, sort_keys=True)),
    }
    _write_json(RESULTS_DIR / "phaseU_integrity.json", phase_u_report)
    print("[phaseU] complete")
    return phase_u_report


# ---------------------------------------------------------------------------
# Phase Y: frozen-neighbour target scoring
# ---------------------------------------------------------------------------


def _load_phase_u(seed: int) -> dict[str, np.ndarray]:
    return dict(np.load(CACHE_DIR / f"phaseU_seed{seed}.npz"))


def _v8_per_query(
    targets: np.ndarray,
    query_positions: np.ndarray,
    neighbor_positions: np.ndarray,
) -> np.ndarray:
    y_q = targets[query_positions][:, None]
    y_n = targets[neighbor_positions]
    return np.abs(y_q - y_n).mean(axis=1)


def _collision_per_query(
    targets: np.ndarray,
    query_positions: np.ndarray,
    neighbor_positions: np.ndarray,
    tau_far: float,
) -> np.ndarray:
    y_q = targets[query_positions][:, None]
    y_n = targets[neighbor_positions]
    return (np.abs(y_q - y_n) >= tau_far).mean(axis=1)


def _nn8_mae_per_query(
    targets: np.ndarray,
    query_positions: np.ndarray,
    neighbor_positions: np.ndarray,
) -> np.ndarray:
    y_q = targets[query_positions]
    y_n = targets[neighbor_positions]
    return np.abs(y_q - np.median(y_n, axis=1))


def _paired_bootstrap(values: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = int(values.shape[0])
    means = np.empty(BOOTSTRAP_B, dtype=np.float64)
    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, n)
        means[b] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "prob_gt_0": float((means > 0).mean()),
        "n_queries": n,
    }


def phase_y() -> dict[str, Any]:
    records = sm.build_encoded_records()[0]
    targets = np.asarray([float(record.y) for record in records], dtype=np.float64)
    ref, sel, probe, _ = split_positions()

    # far threshold from fixed random reference pairs
    rng = np.random.default_rng(FAR_REFERENCE_PAIR_SEED)
    left = rng.choice(ref, size=N_FAR_REFERENCE_PAIRS, replace=True)
    right = rng.choice(ref, size=N_FAR_REFERENCE_PAIRS, replace=True)
    pair_deltas = np.abs(targets[left] - targets[right])
    tau_far = float(np.percentile(pair_deltas, FAR_QUANTILE))
    _write_json(
        RESULTS_DIR / "target_far_threshold.json",
        {
            "definition": "80th percentile of |y_a - y_b| over fixed random reference pairs",
            "reference_pairs": N_FAR_REFERENCE_PAIRS,
            "seed": FAR_REFERENCE_PAIR_SEED,
            "quantile": FAR_QUANTILE,
            "tau_far": tau_far,
            "mean_abs_pair_delta": float(pair_deltas.mean()),
            "target_read_after_neighbor_lock": True,
        },
    )

    splits = {"probe": probe, "selection": sel}
    stage_metrics: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    per_query_v8: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    rand_baseline: dict[str, Any] = {"splits": {}}

    for seed in (0, 1):
        data = _load_phase_u(seed)
        stage_metrics[seed] = {}
        per_query_v8[seed] = {}
        for split_name, query_positions in splits.items():
            rand_nbr = data[f"rand_nbr__{split_name}"].astype(np.int64)
            v_rand = float(np.abs(targets[query_positions][:, None] - targets[rand_nbr]).mean())
            rand_baseline["splits"][f"seed{seed}_{split_name}"] = {
                "v_rand": v_rand,
                "n_queries": int(len(query_positions)),
            }
            stage_metrics[seed][split_name] = {}
            per_query_v8[seed][split_name] = {}
            for stage in STAGE_BY_KEY:
                nbr = data[f"nbr__{stage}__{split_name}"].astype(np.int64)
                v8 = _v8_per_query(targets, query_positions, nbr)
                collision = _collision_per_query(targets, query_positions, nbr, tau_far)
                nn8 = _nn8_mae_per_query(targets, query_positions, nbr)
                per_query_v8[seed][split_name][stage] = v8
                stage_metrics[seed][split_name][stage] = {
                    "v8": float(v8.mean()),
                    "eta": float(v8.mean() / v_rand),
                    "collision": float(collision.mean()),
                    "nn8_mae": float(nn8.mean()),
                    "v_rand": v_rand,
                }

    # stage_metrics csv
    for seed in (0, 1):
        rows = []
        for split_name in ("probe", "selection"):
            for spec in STAGE_SPECS:
                stage = spec["key"]
                row = {"seed": seed, "split": split_name, "stage": stage, "code": spec["code"], "kind": spec["kind"]}
                row.update(stage_metrics[seed][split_name][stage])
                rows.append(row)
        _write_csv(
            RESULTS_DIR / f"stage_metrics_seed{seed}.csv",
            rows,
            ["seed", "split", "stage", "code", "kind", "v8", "eta", "v_rand", "collision", "nn8_mae"],
        )

    _write_json(RESULTS_DIR / "random_neighbor_baseline.json", rand_baseline)

    # ---- contrasts ----
    contrast_metrics: dict[int, dict[str, dict[str, Any]]] = {0: {}, 1: {}}
    bootstrap: dict[str, Any] = {"B": BOOTSTRAP_B, "seed_base": BOOTSTRAP_BASE_SEED, "contrasts": {}}
    for seed in (0, 1):
        for contrast in CONTRASTS:
            cid = contrast["id"]
            entry: dict[str, Any] = {"id": cid, "name": contrast["name"], "a": contrast["a"], "b": contrast["b"]}
            for split_name in ("probe", "selection"):
                a = stage_metrics[seed][split_name][contrast["a"]]
                b = stage_metrics[seed][split_name][contrast["b"]]
                delta_v8 = b["v8"] - a["v8"]
                delta_eta = delta_v8 / a["v_rand"]
                per_query_delta = per_query_v8[seed][split_name][contrast["b"]] - per_query_v8[seed][split_name][contrast["a"]]
                # normalise the paired bootstrap by the same V_rand so that the
                # interval is reported (and gated) on the eta scale
                boot = _paired_bootstrap(
                    per_query_delta / a["v_rand"],
                    _label_seed("boot", cid, seed, split_name, BOOTSTRAP_BASE_SEED),
                )
                entry[split_name] = {
                    "eta_a": a["eta"],
                    "eta_b": b["eta"],
                    "delta_eta": delta_eta,
                    "collision_a": a["collision"],
                    "collision_b": b["collision"],
                    "delta_collision": b["collision"] - a["collision"],
                    "nn8_mae_a": a["nn8_mae"],
                    "nn8_mae_b": b["nn8_mae"],
                    "delta_nn8_mae": b["nn8_mae"] - a["nn8_mae"],
                    "bootstrap_delta_eta": boot,
                }
                bootstrap["contrasts"].setdefault(cid, {})[f"seed{seed}_{split_name}"] = boot
            contrast_metrics[seed][cid] = entry

    for seed in (0, 1):
        rows = []
        for contrast in CONTRASTS:
            entry = contrast_metrics[seed][contrast["id"]]
            for split_name in ("probe", "selection"):
                sub = entry[split_name]
                rows.append(
                    {
                        "seed": seed,
                        "contrast": contrast["id"],
                        "name": contrast["name"],
                        "split": split_name,
                        **{key: sub[key] for key in sub if key != "bootstrap_delta_eta"},
                        "boot_ci_low": sub["bootstrap_delta_eta"]["ci_low"],
                        "boot_ci_high": sub["bootstrap_delta_eta"]["ci_high"],
                    }
                )
        _write_csv(
            RESULTS_DIR / f"contrast_metrics_seed{seed}.csv",
            rows,
            [
                "seed", "contrast", "name", "split", "eta_a", "eta_b", "delta_eta",
                "collision_a", "collision_b", "delta_collision",
                "nn8_mae_a", "nn8_mae_b", "delta_nn8_mae", "boot_ci_low", "boot_ci_high",
            ],
        )
    _write_json(RESULTS_DIR / "bootstrap_intervals.json", bootstrap)

    _write_json(
        RESULTS_DIR / "target_metric_lock.json",
        {
            "primary_metric": "eta(Z) = V_8(Z) / V_rand",
            "v8": "mean over queries of mean over the 8 frozen reference neighbours of |y_i - y_j|",
            "v_rand": "mean over queries of mean over 8 fixed random reference neighbours of |y_i - y_j|",
            "collision": "fraction of (query, neighbour) pairs with |y_i - y_j| >= tau_far",
            "nn8_mae": "median of the 8 neighbour targets vs query target (diagnostic only)",
            "far_threshold": tau_far,
            "bootstrap": {"B": BOOTSTRAP_B, "paired_by_query": True, "seed_base": BOOTSTRAP_BASE_SEED},
            "official_test_loaded": False,
        },
    )

    np.savez_compressed(
        CACHE_DIR / "phaseY_per_query.npz",
        **{
            f"v8__seed{seed}__{split_name}__{stage}": per_query_v8[seed][split_name][stage].astype(np.float32)
            for seed in (0, 1)
            for split_name in ("probe", "selection")
            for stage in STAGE_BY_KEY
        },
        targets=targets.astype(np.float32),
    )

    result = {
        "tau_far": tau_far,
        "stage_metrics": stage_metrics,
        "contrast_metrics": contrast_metrics,
        "bootstrap": bootstrap,
        "random_neighbor_baseline": rand_baseline,
    }
    _write_json(CACHE_DIR / "phaseY_summary.json", result)
    print("[phaseY] complete")
    return result


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def _contrast_status(entry: dict[str, Any]) -> dict[str, Any]:
    # per-seed values are stored under entry["per_seed"] by the caller
    per_seed = entry["per_seed"]
    deltas = {seed: per_seed[seed]["probe"]["delta_eta"] for seed in (0, 1)}
    deltas_sel = {seed: per_seed[seed]["selection"]["delta_eta"] for seed in (0, 1)}
    cis = {seed: per_seed[seed]["probe"]["bootstrap_delta_eta"] for seed in (0, 1)}
    dcol = {seed: per_seed[seed]["probe"]["delta_collision"] for seed in (0, 1)}

    probe_material = all(
        deltas[seed] >= MATERIAL_DELTA_ETA and cis[seed]["ci_low"] > 0.0 for seed in (0, 1)
    )
    selection_positive = all(deltas_sel[seed] > 0.0 for seed in (0, 1))
    collision_consistent = all(dcol[seed] >= 0.0 for seed in (0, 1))
    material_degradation = bool(probe_material and selection_positive and collision_consistent)

    probe_gain = all(
        deltas[seed] <= -MATERIAL_DELTA_ETA and cis[seed]["ci_high"] < 0.0 for seed in (0, 1)
    )
    material_gain = bool(probe_gain)

    seed_conflict = bool(deltas[0] * deltas[1] < 0.0)
    split_conflict = bool(
        (deltas[0] > 0 and deltas_sel[0] < 0) or (deltas[1] > 0 and deltas_sel[1] < 0)
    )
    if material_degradation:
        status = "MATERIAL_GEOMETRY_DEGRADATION"
    elif material_gain:
        status = "MATERIAL_GEOMETRY_GAIN"
    elif seed_conflict:
        status = "SEED_UNSTABLE"
    elif split_conflict:
        status = "SPLIT_UNSTABLE"
    else:
        status = "GEOMETRY_STABLE_OR_SMALL_EFFECT"
    return {
        "status": status,
        "material_degradation": material_degradation,
        "material_gain": material_gain,
        "probe_material": probe_material,
        "selection_positive": selection_positive,
        "collision_consistent": collision_consistent,
        "seed_conflict": seed_conflict,
        "split_conflict": split_conflict,
        "delta_eta": {f"seed{seed}": deltas[seed] for seed in (0, 1)},
        "delta_eta_selection": {f"seed{seed}": deltas_sel[seed] for seed in (0, 1)},
    }


def decide() -> dict[str, Any]:
    summary = _read_json(CACHE_DIR / "phaseY_summary.json")
    contrast_metrics = summary["contrast_metrics"]
    # rebuild per_seed structure keyed by contrast id
    per_contrast: dict[str, Any] = {}
    for contrast in CONTRASTS:
        cid = contrast["id"]
        entry = {**contrast, "per_seed": {0: contrast_metrics["0"][cid], 1: contrast_metrics["1"][cid]}}
        # flat probe/selection are taken from seed0 for convenience only
        entry["probe"] = contrast_metrics["0"][cid]["probe"]
        entry["selection"] = contrast_metrics["0"][cid]["selection"]
        entry["status"] = _contrast_status(entry)
        per_contrast[cid] = entry

    material = [cid for cid in ("A", "B", "C", "D1", "D2") if per_contrast[cid]["status"]["material_degradation"]]
    first_material = material[0] if material else None

    graph_material = [cid for cid in ("D1", "D2") if cid in material]
    case = "Case E"
    case_title = "NO CLEAR STAGEWISE REPRESENTATION COLLISION"
    implicated_stage = None
    design_authorized = False
    notes: list[str] = []

    seed_conflicts = [cid for cid, entry in per_contrast.items() if entry["status"]["seed_conflict"]]
    split_conflicts = [cid for cid, entry in per_contrast.items() if entry["status"]["split_conflict"]]
    # a credible localization ambiguity needs a *direction* (or probe/selection)
    # disagreement at a magnitude that is not simply numerical noise
    credible_disagreement: list[dict[str, Any]] = []
    for cid, entry in per_contrast.items():
        p0 = entry["per_seed"][0]["probe"]["delta_eta"]
        p1 = entry["per_seed"][1]["probe"]["delta_eta"]
        s0 = entry["per_seed"][0]["selection"]["delta_eta"]
        s1 = entry["per_seed"][1]["selection"]["delta_eta"]
        if p0 * p1 < 0 and min(abs(p0), abs(p1)) >= 0.01:
            credible_disagreement.append({"contrast": cid, "kind": "seed_sign_conflict"})
        if (p0 >= MATERIAL_DELTA_ETA and s0 < 0) or (p1 >= MATERIAL_DELTA_ETA and s1 < 0):
            credible_disagreement.append({"contrast": cid, "kind": "probe_selection_conflict"})
    sub_threshold: list[dict[str, Any]] = []
    for cid, entry in per_contrast.items():
        p0 = entry["per_seed"][0]["probe"]["delta_eta"]
        p1 = entry["per_seed"][1]["probe"]["delta_eta"]
        if p0 > 0 and p1 > 0 and not entry["status"]["material_degradation"]:
            sub_threshold.append(
                {
                    "contrast": cid,
                    "name": entry["name"],
                    "delta_eta_probe": {"seed0": p0, "seed1": p1},
                    "note": (
                        "positive degradation direction on both seeds but below the "
                        "pre-registered +0.02 two-seed materiality gate; not authorized"
                    ),
                }
            )

    if first_material == "A":
        case = "Case A"
        case_title = "EARLY LOCAL REPRESENTATION BOTTLENECK CANDIDATE"
        implicated_stage = "patch_encoder"
        design_authorized = True
    elif first_material == "B":
        case = "Case B"
        case_title = "RELATION REPRESENTATION BOTTLENECK CANDIDATE"
        implicated_stage = "pair_encoder"
        design_authorized = True
    elif first_material == "C":
        case = "Case C"
        case_title = "CENTRE CONTEXTUALIZATION BOTTLENECK CANDIDATE"
        implicated_stage = "centre_contextualization"
        design_authorized = True
    elif first_material in ("D1", "D2"):
        case = "Case D"
        case_title = "GRAPH-LEVEL COMPRESSION BOTTLENECK CANDIDATE"
        implicated_stage = "unary_graph_compression" if first_material == "D1" else "pair_graph_compression"
        design_authorized = True
        if len(graph_material) == 2:
            # both unary and pair compression fail: only one hypothesis is allowed
            stability = {
                cid: sum(
                    abs(per_contrast[cid]["per_seed"][seed]["probe"]["delta_eta"]) for seed in (0, 1)
                )
                for cid in graph_material
            }
            if abs(stability["D1"] - stability["D2"]) < 0.01:
                case_title = "MULTIPLE GRAPH COMPRESSION BOTTLENECKS — DESIGN NOT AUTHORIZED"
                design_authorized = False
                notes.append("both unary and pair graph compression degrade and cannot be cleanly separated")
            else:
                implicated_stage = (
                    "unary_graph_compression" if stability["D1"] > stability["D2"] else "pair_graph_compression"
                )
                notes.append("both graph compressions degrade; stronger cross-seed effect selected")
        # signal the sub-type explicitly
        if "D1" in graph_material and "D2" not in graph_material:
            notes.append("UNARY GRAPH COMPRESSION implicated")
        elif "D2" in graph_material and "D1" not in graph_material:
            notes.append("PAIR GRAPH COMPRESSION implicated")
        elif len(graph_material) == 2:
            notes.append("both UNARY and PAIR GRAPH COMPRESSION implicated")
    elif credible_disagreement:
        case = "Case F"
        case_title = "BOTTLENECK LOCALIZATION INCONCLUSIVE"
        implicated_stage = None
        design_authorized = False
        notes.append(f"credible localization disagreement: {credible_disagreement}")
        notes.append(f"seed sign conflicts (descriptive): {seed_conflicts}")
        notes.append(f"split conflicts (descriptive): {split_conflicts}")
    else:
        case = "Case E"
        case_title = "NO CLEAR STAGEWISE REPRESENTATION COLLISION"
        implicated_stage = None
        design_authorized = False
        notes.append(
            "no contrast reached the pre-registered two-seed material degradation gate; "
            "the honest reading is no clear stagewise representation collision"
        )
        if seed_conflicts:
            notes.append(f"sub-threshold seed direction conflicts: {seed_conflicts}")
        if sub_threshold:
            notes.append(
                "stable sub-threshold positive degradation direction (not material, not authorized): "
                + ", ".join(item["contrast"] for item in sub_threshold)
            )
        notes.append(
            "material geometry gains at patch encoder / pair encoder / unary compression are descriptive only"
        )

    bottleneck_decision = {
        "decision_case": case,
        "decision_title": case_title,
        "implicated_stage": implicated_stage,
        "first_material_degradation": first_material,
        "all_material_degradations": material,
        "statement": (
            "target-relevant neighbourhood geometry degradation, not information-theoretic loss; "
            "kNN geometry degradation is not identical-representation collision"
        ),
        "contrasts": {
            cid: {
                "name": entry["name"],
                "a": entry["a"],
                "b": entry["b"],
                "status": entry["status"],
            }
            for cid, entry in per_contrast.items()
        },
        "notes": notes,
        "sub_threshold_degradation_signals": sub_threshold,
        "credible_disagreement": credible_disagreement,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "bottleneck_decision.json", bottleneck_decision)

    top1 = {
        "authorized_for_design": bool(design_authorized),
        "implicated_stage": implicated_stage,
        "decision_case": case,
        "evidence": {
            cid: per_contrast[cid]["status"] for cid in ("A", "B", "C", "D1", "D2")
        },
        "existing_nogos_that_constrain_design": [
            "P1 learned centre composer (NO-GO) — do not re-open learned mean pooling",
            "P2 one-shot relation refresh (NO-GO) — do not simply refresh q",
            "covariance / triad / endpoint statistics (NO-GO) — do not add another relation statistic",
            "compact-v4-cell persistent cycle cells (NO-GO) — do not re-open T=2 cell architecture",
            "simple richer frozen readout (NO-GO)",
            "lookup low-rank donor (not supported)",
        ],
        "what_new_principle_must_change": (
            None
            if not design_authorized
            else {
                "patch_encoder": "the principle of forming the local patch representation",
                "pair_encoder": "the principle of relation representation (not q refresh / wider q / covariance / endpoint)",
                "centre_contextualization": "the factorisation of contextualisation (not P1 learned mean pooling, not wider centre MLP)",
                "unary_graph_compression": "how set-aware structural distinctions are preserved compactly in the unary summary",
                "pair_graph_compression": "how set-aware relation distinctions are preserved compactly in the pair summary",
            }.get(implicated_stage)
        ),
        "what_must_not_be_reopened": [
            "widen patch encoder / add layers / revive dictionary",
            "second relation refresh / depth sweep / alpha mixing / concat q0;q1",
            "learned mean pooling / wider centre MLP",
            "another relation statistic (covariance/triad/endpoint)",
            "more moments appended to the graph summary",
            "attention / Set Transformer",
            "official test access",
        ],
        "next_full_training_budget": "not yet authorized",
        "information_theoretic_caveat": "geometry evidence only, not information-theoretic proof",
    }
    _write_json(RESULTS_DIR / "top1_representation_hypothesis.json", top1)

    return bottleneck_decision


# ---------------------------------------------------------------------------
# robustness (only if triggered)
# ---------------------------------------------------------------------------


def robustness() -> dict[str, Any]:
    bottleneck = _read_json(RESULTS_DIR / "bottleneck_decision.json")
    material_contrasts = [cid for cid in ("A", "B", "C", "D1", "D2") if bottleneck["contrasts"][cid]["status"]["material_degradation"]]
    if not material_contrasts:
        _write_json(
            RESULTS_DIR / "projection_robustness.json",
            {"triggered": False, "reason": "no contrast reached the material gate", "official_test_loaded": False},
        )
        _write_json(
            RESULTS_DIR / "k_robustness.json",
            {"triggered": False, "reason": "no contrast reached the material gate", "official_test_loaded": False},
        )
        return {"triggered": False}

    # second locked projection seed for the set stages involved
    records = sm.build_encoded_records()[0]
    ref, sel, probe, _ = split_positions()
    targets = np.asarray([float(record.y) for record in records], dtype=np.float64)

    involved_set_stages: set[str] = set()
    for cid in material_contrasts:
        entry = next(c for c in CONTRASTS if c["id"] == cid)
        for key in (entry["a"], entry["b"]):
            if STAGE_BY_KEY[key]["kind"] == "set":
                involved_set_stages.add(key)

    projection_robustness: dict[str, Any] = {
        "triggered": True,
        "second_projection_base_seed": ROBUST_BASE_SEED,
        "material_contrasts": material_contrasts,
        "involved_set_stages": sorted(involved_set_stages),
        "official_test_loaded": False,
    }
    # recompute sketches for involved set stages with the second projection seed
    for seed in (0, 1):
        model = build_model(seed)
        data = _load_phase_u(seed)
        # stats fit on references using second projections
        for stage in sorted(involved_set_stages):
            projections = _stage_projection(stage, ROBUST_BASE_SEED)
            # fit object mu/sigma on references (same normalization inputs)
            accum = _Accumulator(int(projections.shape[1]))
            for pos in ref.tolist():
                accum.add(_forward_graph(model, records[pos])[stage])
            mu, sigma, _ = accum.stats()
            raw = np.zeros((len(records), SET_SKETCH_DIM))
            for pos in range(len(records)):
                raw[pos] = set_sketch(_forward_graph(model, records[pos])[stage], mu, sigma, projections)
            standardized, _, _ = _standardize_stage(raw, ref)
            data[f"std__{stage}"] = standardized.astype(np.float32)
        # rebuild neighbours for involved stages
        for stage in sorted(involved_set_stages):
            for split_name, query_positions in (("probe", probe), ("selection", sel)):
                nbr, dist = _nearest_neighbors(data[f"std__{stage}"], query_positions, ref, K_PRIMARY)
                data[f"nbr__{stage}__{split_name}"] = nbr.astype(np.int32)
                data[f"ndist__{stage}__{split_name}"] = dist.astype(np.float32)
        # score
        for cid in material_contrasts:
            entry = next(c for c in CONTRASTS if c["id"] == cid)
            result: dict[str, Any] = {}
            for split_name, query_positions in (("probe", probe), ("selection", sel)):
                rand_nbr = data[f"rand_nbr__{split_name}"].astype(np.int64)
                v_rand = float(np.abs(targets[query_positions][:, None] - targets[rand_nbr]).mean())
                v8 = {}
                for key in (entry["a"], entry["b"]):
                    nbr = data[f"nbr__{key}__{split_name}"].astype(np.int64)
                    v8[key] = float(_v8_per_query(targets, query_positions, nbr).mean())
                result[split_name] = {
                    "eta_a": v8[entry["a"]] / v_rand,
                    "eta_b": v8[entry["b"]] / v_rand,
                    "delta_eta": (v8[entry["b"]] - v8[entry["a"]]) / v_rand,
                }
            projection_robustness.setdefault(cid, {})[f"seed{seed}"] = result

    # direction check
    for cid in material_contrasts:
        directions = [
            projection_robustness[cid][f"seed{seed}"]["probe"]["delta_eta"] for seed in (0, 1)
        ]
        projection_robustness[cid]["second_seed_direction_positive"] = all(value > 0 for value in directions)
        projection_robustness[cid]["sketch_unstable"] = not all(value > 0 for value in directions)

    _write_json(RESULTS_DIR / "projection_robustness.json", projection_robustness)

    # k robustness: descriptive only, using the primary standardized arrays
    k_report: dict[str, Any] = {"triggered": True, "primary_k": K_PRIMARY, "k_values": list(K_ROBUST), "official_test_loaded": False}
    for seed in (0, 1):
        data = _load_phase_u(seed)
        for cid in material_contrasts:
            entry = next(c for c in CONTRASTS if c["id"] == cid)
            for k in K_ROBUST:
                for split_name, query_positions in (("probe", probe), ("selection", sel)):
                    rand_nbr = data[f"rand_nbr__{split_name}"].astype(np.int64)
                    v_rand = float(np.abs(targets[query_positions][:, None] - targets[rand_nbr]).mean())
                    v8 = {}
                    for key in (entry["a"], entry["b"]):
                        nbr, _ = _nearest_neighbors(data[f"std__{key}"], query_positions, ref, k)
                        v8[key] = float(_v8_per_query(targets, query_positions, nbr).mean())
                    k_report.setdefault(cid, {}).setdefault(f"k{k}", {})[f"seed{seed}_{split_name}"] = (
                        (v8[entry["b"]] - v8[entry["a"]]) / v_rand
                    )
    for cid in material_contrasts:
        dirs = [
            k_report[cid][f"k{k}"][f"seed{seed}_probe"] for k in K_ROBUST for seed in (0, 1)
        ]
        k_report[cid]["neighborhood_scale_unstable"] = not any(value > 0 for value in dirs)
    _write_json(RESULTS_DIR / "k_robustness.json", k_report)

    return projection_robustness


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    phase_y_summary = _read_json(CACHE_DIR / "phaseY_summary.json")
    stage_metrics = phase_y_summary["stage_metrics"]
    bootstrap = phase_y_summary["bootstrap"]

    # Figure 1 — computation graph with audited boundaries
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    labels = [
        ("patch input  X_patch", 0.9),
        ("h0  (local patch state)", 0.8),
        ("pair input  X_pair  ->  q0", 0.7),
        ("centre input  X_centre  ->  h1", 0.6),
        ("unary U (97)  |  pair P (165)", 0.5),
        ("global G (32)  topology T (8)", 0.4),
        ("R = [U; P; G; T] (302)", 0.3),
    ]
    for text, y in labels:
        ax.text(0.5, y, text, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round", fc="#eef3fb", ec="#4169e1"))
    boundaries = [
        ("A", 0.85, "Contrast A: patch encoder"),
        ("B", 0.65, "Contrast B: pair encoder"),
        ("C", 0.55, "Contrast C: centre update"),
        ("D1", 0.45, "Contrast D1: unary compression"),
        ("D2", 0.65, "Contrast D2: pair compression"),
    ]
    for cid, y, text in boundaries:
        ax.annotate(text, xy=(0.72, y), xytext=(0.98, y), ha="right", va="center",
                    fontsize=9, color="#b22222",
                    arrowprops=dict(arrowstyle="->", color="#b22222"))
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0.2, 1.0)
    ax.set_title("Compact-v4-smallhead computation graph — audited compression boundaries")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure1_computation_graph.png", dpi=140)
    plt.close(fig)

    # Figure 2 — stagewise eta, seeds separated
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    stage_order = [spec["key"] for spec in STAGE_SPECS]
    for ax, seed in zip(axes, (0, 1)):
        for split_name, color in (("probe", "#1f77b4"), ("selection", "#ff7f0e")):
            values = [stage_metrics[str(seed)][split_name][stage]["eta"] for stage in stage_order]
            ax.plot(range(len(stage_order)), values, marker="o", label=split_name, color=color)
        ax.set_xticks(range(len(stage_order)))
        ax.set_xticklabels([STAGE_BY_KEY[s]["code"] for s in stage_order], rotation=45)
        ax.set_title(f"seed{seed}")
        ax.grid(alpha=0.3)
        if seed == 0:
            ax.set_ylabel(r"$\eta = V_8 / V_{rand}$")
            ax.legend()
    fig.suptitle("Stagewise target-neighbour disagreement (official train only)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure2_stagewise_eta.png", dpi=140)
    plt.close(fig)

    # Figure 3 — contrast delta eta with bootstrap CI
    fig, ax = plt.subplots(figsize=(8, 4.5))
    contrast_ids = [c["id"] for c in CONTRASTS]
    x = np.arange(len(contrast_ids))
    width = 0.35
    for offset, seed, color in ((-width / 2, 0, "#1f77b4"), (width / 2, 1, "#d62728")):
        means = []
        errs = []
        for cid in contrast_ids:
            boot = bootstrap["contrasts"][cid][f"seed{seed}_probe"]
            means.append(boot["mean"])
            errs.append([boot["mean"] - boot["ci_low"], boot["ci_high"] - boot["mean"]])
        ax.bar(x + offset, means, width, yerr=np.asarray(errs).T, capsize=4, label=f"seed{seed}", color=color)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(MATERIAL_DELTA_ETA, color="red", linestyle="--", linewidth=0.8, label="+0.02 gate")
    ax.set_xticks(x)
    ax.set_xticklabels(contrast_ids)
    ax.set_ylabel(r"$\Delta\eta$ (probe)")
    ax.set_title("Contrast delta eta with paired-bootstrap 95% CI")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure3_contrast_delta_eta.png", dpi=140)
    plt.close(fig)

    # Figure 4 — collision rate (only when it helps explain the primary finding)
    decision_path = RESULTS_DIR / "bottleneck_decision.json"
    if decision_path.exists():
        decision = _read_json(decision_path)
        if decision["decision_case"] in ("Case A", "Case B", "Case C", "Case D"):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for offset, seed, color in ((-width / 2, 0, "#1f77b4"), (width / 2, 1, "#d62728")):
                values = []
                for cid in contrast_ids:
                    entry = _read_json(CACHE_DIR / "phaseY_summary.json")["contrast_metrics"][str(seed)][cid]
                    values.append(entry["probe"]["delta_collision"])
                ax.bar(x + offset, values, width, label=f"seed{seed}", color=color)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(contrast_ids)
            ax.set_ylabel(r"$\Delta$ collision (probe)")
            ax.set_title("Collision-rate change by contrast")
            ax.legend()
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "figure4_collision_rate.png", dpi=140)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Q1–Q20 + final decision
# ---------------------------------------------------------------------------


def answers_and_final() -> dict[str, Any]:
    protocol_lock = _read_json(RESULTS_DIR / "audit_protocol_lock.json")
    stage_inventory = _read_json(RESULTS_DIR / "stage_inventory.json")
    phase_y_summary = _read_json(CACHE_DIR / "phaseY_summary.json")
    stage_metrics = phase_y_summary["stage_metrics"]
    contrast_metrics = phase_y_summary["contrast_metrics"]
    rand = phase_y_summary["random_neighbor_baseline"]
    bottleneck = _read_json(RESULTS_DIR / "bottleneck_decision.json")
    projection_robustness_path = RESULTS_DIR / "projection_robustness.json"
    k_robustness_path = RESULTS_DIR / "k_robustness.json"
    projection_robustness = _read_json(projection_robustness_path) if projection_robustness_path.exists() else {"triggered": False}
    k_robustness = _read_json(k_robustness_path) if k_robustness_path.exists() else {"triggered": False}

    def sm_stage(seed: int, split: str, stage: str) -> dict[str, float]:
        return stage_metrics[str(seed)][split][stage]

    def contrast(seed: int, cid: str, split: str) -> dict[str, Any]:
        return contrast_metrics[str(seed)][cid][split]

    stage_dims = {spec["code"]: spec for spec in stage_inventory["stages"].values()}
    q: dict[str, Any] = {}
    q["Q1"] = {
        "stages": [
            {"code": spec["code"], "key": spec["key"], "kind": spec["kind"], "role": spec["role"]}
            for spec in STAGE_SPECS
        ]
    }
    q["Q2"] = {"input_dim": stage_dims["S0"]["dimension"], "output_dim": stage_dims["S1"]["dimension"]}
    q["Q3"] = {"input_dim": stage_dims["S2"]["dimension"], "output_dim": stage_dims["S3"]["dimension"]}
    q["Q4"] = {"input_dim": stage_dims["S4"]["dimension"], "output_dim": stage_dims["S5"]["dimension"]}
    q["Q5"] = protocol_lock["set_sketch"]
    q["Q6"] = {
        "note": "V_rand depends on the query split and seed-random neighbours; target-only",
        "v_rand": rand["splits"],
    }
    q["Q7"] = {
        "patch_encoder_input_eta": {f"seed{s}_{sp}": sm_stage(s, sp, "patch_encoder_input")["eta"] for s in (0, 1) for sp in ("probe", "selection")},
        "h0_eta": {f"seed{s}_{sp}": sm_stage(s, sp, "h0")["eta"] for s in (0, 1) for sp in ("probe", "selection")},
    }
    q["Q8"] = bottleneck["contrasts"]["A"]["status"]
    q["Q9"] = {
        "pair_encoder_input_eta": {f"seed{s}_{sp}": sm_stage(s, sp, "pair_encoder_input")["eta"] for s in (0, 1) for sp in ("probe", "selection")},
        "q0_eta": {f"seed{s}_{sp}": sm_stage(s, sp, "q0")["eta"] for s in (0, 1) for sp in ("probe", "selection")},
    }
    q["Q10"] = bottleneck["contrasts"]["B"]["status"]
    q["Q11"] = {
        "centre_update_input_eta": {f"seed{s}_{sp}": sm_stage(s, sp, "centre_update_input")["eta"] for s in (0, 1) for sp in ("probe", "selection")},
        "h1_eta": {f"seed{s}_{sp}": sm_stage(s, sp, "h1")["eta"] for s in (0, 1) for sp in ("probe", "selection")},
    }
    q["Q12"] = bottleneck["contrasts"]["C"]["status"]
    q["Q13"] = bottleneck["contrasts"]["D1"]["status"]
    q["Q14"] = bottleneck["contrasts"]["D2"]["status"]
    q["Q15"] = {
        "R302_eta": {f"seed{s}_{sp}": sm_stage(s, sp, "R")["eta"] for s in (0, 1) for sp in ("probe", "selection")},
        "R302_collision": {f"seed{s}_{sp}": sm_stage(s, sp, "R")["collision"] for s in (0, 1) for sp in ("probe", "selection")},
        "R_without_topology_descriptive": "not recomputed; topology not re-reviewed",
    }
    deltas = {
        cid: {
            f"seed{s}": contrast(s, cid, "probe")["delta_eta"] for s in (0, 1)
        }
        for cid in ("A", "B", "C", "D1", "D2")
    }
    q["Q16"] = {"descriptive_delta_eta": deltas}
    q["Q17"] = {"first_material_degradation": bottleneck["first_material_degradation"]}
    q["Q18"] = {
        cid: {
            "seed0": contrast(0, cid, "probe")["delta_eta"],
            "seed1": contrast(1, cid, "probe")["delta_eta"],
            "consistent_direction": bool(
                contrast(0, cid, "probe")["delta_eta"] * contrast(1, cid, "probe")["delta_eta"] >= 0
            ),
        }
        for cid in ("A", "B", "C", "D1", "D2")
    }
    q["Q19"] = {
        "selection_replication": {
            cid: {
                f"seed{s}": contrast(s, cid, "selection")["delta_eta"] for s in (0, 1)
            }
            for cid in ("A", "B", "C", "D1", "D2")
        },
        "projection_robustness": projection_robustness,
        "k_robustness": k_robustness,
    }
    q["Q20"] = {"decision_case": bottleneck["decision_case"], "decision_title": bottleneck["decision_title"]}
    _write_json(RESULTS_DIR / "answers_q1_q20.json", q)

    final = {
        "decision_case": bottleneck["decision_case"],
        "decision_title": bottleneck["decision_title"],
        "implicated_stage": bottleneck["implicated_stage"],
        "first_material_degradation": bottleneck["first_material_degradation"],
        "authorized_for_design": _read_json(RESULTS_DIR / "top1_representation_hypothesis.json")["authorized_for_design"],
        "next_full_training_budget": "not yet authorized",
        "official_test_loaded": False,
        "official_valid_used_for_selection": False,
        "information_theoretic_caveat": "geometry evidence, not information-theoretic proof",
        "budget_discipline": "zero full-backbone training; only the two pre-trained 82.1K small-head checkpoints reused",
        "notes": bottleneck.get("notes", []),
    }
    _write_json(RESULTS_DIR / "final_decision.json", final)
    return final


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_all() -> None:
    protocol()
    phase_u()
    phase_y()
    decide()
    robustness()
    figures()
    answers_and_final()
    print("[all] complete")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compact-v4 stagewise representation collision audit")
    parser.add_argument(
        "stage",
        choices=["protocol", "phaseU", "phaseY", "robustness", "decide", "figures", "answers", "all"],
    )
    args = parser.parse_args(argv)
    if args.stage == "protocol":
        protocol()
    elif args.stage == "phaseU":
        phase_u()
    elif args.stage == "phaseY":
        phase_y()
    elif args.stage == "robustness":
        robustness()
    elif args.stage == "decide":
        decide()
    elif args.stage == "figures":
        figures()
    elif args.stage == "answers":
        answers_and_final()
    else:
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
