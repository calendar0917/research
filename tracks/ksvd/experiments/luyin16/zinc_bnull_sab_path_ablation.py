"""ZINC B-Null S/A/B path ablation (frozen screen + <=3 retrained seed0 tickets).

Scientific question
-------------------
The B-Null backbone (49,343 params, local 16-D patch token forced to an exact
zero, no local-token generator) reaches official-valid Top-5 soup MAE
``0.123028`` at seed0 and proved that the molecule-dependent local patch token
is not required.  This experiment asks which of the *remaining* information
paths carries the performance.

The B-Null backbone is decomposed into four families:

``S`` structure
    distance bucket / log distance / patch overlap / boundary relations /
    shortest-path count / shell masses / size / cycle / degree scalars.

``A`` attribute marginals
    per-patch and per-molecule atom-type and bond-type composition.

``B`` structure--attribute binding
    which atom sits in which shell, which bond sits in which shell-pair, the
    radius-1 typed parent joint context, the pair-specific path/adjacent bond
    assignment.

``C`` composition / computation
    the shared ``P``/``Q`` pair encoder, the pair->centre aggregation, the
    centre update ``U`` and the T=2 weight-tied pair<->centre feedback.

Every intervention measures the **incremental value of one path while the
others are kept**.  Nothing here is a Shapley attribution and the deltas must
never be summed.

Stage A (eval-only frozen screen, no retraining)
------------------------------------------------
The frozen B-Null seed0 Top-5 soup is scored under four interventions:

* ``A1 parent-null``           -- the radius-1 typed parent embedding output is
                                  forced to an exact zero;
* ``A2 pair-B marginalized``   -- pure-S pair columns are bit-identical; the
                                  pair-specific bond assignment (shortest-path
                                  bond composition + adjacent bond type) is
                                  replaced by the molecule's graph-level
                                  bond-type marginal;
* ``A3 patch-B marginalized``  -- shell / root / incident chemistry is replaced
                                  by the patch-level marginals with the shell
                                  and shell-pair masses preserved; pure-S
                                  scalars are bit-identical;
* ``A4 T=1 frozen recurrence`` -- exactly one pair->centre round
                                  (``h0 -> q0 -> h1 -> readout``) with the same
                                  weights.

Stage B (<= 3 retrained seed0 tickets)
--------------------------------------
``NoParent`` / ``Pair-B-Marginal`` / ``Patch-B-Marginal`` retrain the identical
B-Null backbone from scratch on the control representation selected by the
pre-registered ticket rule.  No parameter re-investment, no width/optimizer
sweep, seed0 only, official test never loaded.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_bnull_sab_path_ablation <stage>
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_pair_centre as rec
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_variance_diagnosis as vd
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_training_sufficiency as ztraining
from tracks.ksvd.experiments.luyin16 import zinc_local_token_null as ltn
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/bnull_sab_path_ablation"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
EXAMPLES_DIR = RESULTS_DIR / "examples"

PROTOCOL_VERSION = "bnull_sab_path_ablation_v1"

# frozen B-Null seed0 reference (results/local_token_null)
BNULL_RESULTS_DIR = TRACK_ROOT / "results/local_token_null"
BNULL_SOUP_PATH = BNULL_RESULTS_DIR / "soup_states/lt_null_seed0_top5_soup.pt"
BNULL_SOUP_JSON = BNULL_RESULTS_DIR / "soup_lt_null_seed0.json"
BNULL_PARAMS = ltn.NULL_TOTAL_PARAMS  # 49,343
BNULL_SOUP_VALID_MAE = 0.1230275  # recorded reference (recomputed at run time)

SEED = 0

# ---------------------------------------------------------------------------
# exact feature map (read off the real code, not guessed)
# ---------------------------------------------------------------------------

ATOM_CATEGORIES = zpp.ATOM_CATEGORIES  # 28
BOND_CATEGORIES = zpp.BOND_CATEGORIES  # 4
DISTANCE_BUCKETS = zpp.DISTANCE_BUCKETS  # 5
SHELL_PAIRS = zpp.SHELL_PAIRS  # ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
PATCH_CONT_WIDTH = zpp.SHELL_WIDTH  # 146
RELATION_WIDTH = zpp.RELATION_WIDTH  # 23
GLOBAL_WIDTH = zpp.GLOBAL_WIDTH  # 62

# patch_cont layout (raw, before the train-fit per-column standardizer)
ATOM_SHELL_SLICES = ((0, 28), (28, 56), (56, 84))  # 3 shells x 28 atom types
BOND_SHELL_BLOCKS = tuple(
    (84 + 4 * index, 84 + 4 * (index + 1)) for index in range(len(SHELL_PAIRS))
)  # 6 shell-pairs x 4 bond types
BOND_SHELL_SLICE = (84, 108)
ROOT_ATOM_SLICE = (108, 136)  # centre atom one-hot
INCIDENT_BONDS_SLICE = (136, 140)  # centre incident-bond proportions
SCALARS_SLICE = (140, 146)  # log n_nodes, log n_edges, boundary frac,
#                             cycle_rank/n, centre degree/4, mean degree/4

# pair_relation layout (raw / never standardized)
DIST_ONEHOT_SLICE = (0, 5)
LOG_DIST_SLICE = (5, 6)
OVERLAP_SLICE = (6, 11)
BOUNDARY_SLICE = (11, 14)
PATH_BOND_SLICE = (14, 18)  # B: bond composition averaged over shortest paths
LOG_PATH_SLICE = (18, 19)
ADJACENT_SLICE = (19, 23)  # B: adjacent bond type (zero when not adjacent)

PAIR_S_SLICES = ((0, 14), (18, 19))
PAIR_B_SLICES = (PATH_BOND_SLICE, ADJACENT_SLICE)

TICKETS = ("NoParent", "Pair-B-Marginal", "Patch-B-Marginal")
TICKET_TAGS = {
    "NoParent": "sab_noparent",
    "Pair-B-Marginal": "sab_pairbmarg",
    "Patch-B-Marginal": "sab_patchbmarg",
    "T1": "sab_t1",
}

# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    head = REPO_ROOT / ".git" / "HEAD"
    if head.exists():
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = REPO_ROOT / ".git" / text.split(" ", 1)[1].strip()
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()
        return text
    return "unknown"


def _environment_fingerprint(device: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _mae(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - predictions)))


# ---------------------------------------------------------------------------
# feature map (pre-registration artefact)
# ---------------------------------------------------------------------------


def feature_map() -> dict[str, Any]:
    def block(name: str, slices, family: str, note: str) -> dict[str, Any]:
        return {
            "name": name,
            "columns": [list(map(int, pair)) for pair in slices],
            "family": family,
            "note": note,
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "patch_cont": {
            "width": int(PATCH_CONT_WIDTH),
            "standardized": True,
            "standardizer": "per-column train-fit (mean/scale, scale<1e-6 -> 1)",
            "blocks": [
                block(
                    "atom_shell",
                    ATOM_SHELL_SLICES,
                    "A/B",
                    "atom-type histogram per distance shell 0/1/2 (normalized by "
                    "patch node count); the histogram itself is A, the shell "
                    "assignment is B",
                ),
                block(
                    "bond_shell",
                    BOND_SHELL_BLOCKS,
                    "A/B",
                    "bond-type histogram per shell-pair (0,0)..(2,2) "
                    "(normalized by patch edge count); the histogram itself is A, "
                    "the shell-pair assignment is B",
                ),
                block(
                    "root_atom",
                    [ROOT_ATOM_SLICE],
                    "B",
                    "centre atom one-hot (root structural role <-> atom type)",
                ),
                block(
                    "incident_bonds",
                    [INCIDENT_BONDS_SLICE],
                    "B",
                    "centre incident-bond proportions (root role <-> bond type)",
                ),
                block(
                    "scalars",
                    [SCALARS_SLICE],
                    "S",
                    "log1p(n_nodes), log1p(n_edges), radius-boundary fraction, "
                    "cycle_rank/n, centre degree/4, mean degree/4",
                ),
            ],
        },
        "pair_relation": {
            "width": int(RELATION_WIDTH),
            "standardized": False,
            "blocks": [
                block("distance_onehot", [DIST_ONEHOT_SLICE], "S", "min(d,5)-1 one-hot"),
                block("log_distance", [LOG_DIST_SLICE], "S", "log1p(shortest path)"),
                block(
                    "overlap",
                    [OVERLAP_SLICE],
                    "S",
                    "patch intersection / union / min / max and size diff",
                ),
                block(
                    "boundary",
                    [BOUNDARY_SLICE],
                    "S",
                    "radius-boundary overlap and centre containment",
                ),
                block(
                    "path_bond_mean",
                    [PATH_BOND_SLICE],
                    "B",
                    "mean bond-type composition over shortest paths",
                ),
                block("log_path_count", [LOG_PATH_SLICE], "S", "log1p(#shortest paths)"),
                block(
                    "adjacent_bond",
                    [ADJACENT_SLICE],
                    "S/B",
                    "adjacent bond type one-hot for distance-1 pairs (block mass "
                    "is a pure-S adjacency indicator; the type is B)",
                ),
            ],
        },
        "global_context": {
            "width": int(GLOBAL_WIDTH),
            "standardized": True,
            "note": "S + graph-level A marginals (untouched by every intervention)",
        },
        "topology_features": {
            "mode": "hinge",
            "note": "pure-S topology channel (untouched)",
        },
        "parent_token": {
            "embedding_rows": 32,
            "embedding_width": 8,
            "note": "radius-1 typed joint context (parent-B)",
        },
        "local_token": {
            "width": 16,
            "note": "exact zero in B-Null; no generator instantiated",
        },
        "recurrent_computation": {
            "rounds": 2,
            "weight_tied": True,
            "P": "pair_projection Linear(patch_hidden, q_dim, bias=False)",
            "Q": "pair_encoder MLP(4*q_dim -> max(2*q_dim,64) -> q_dim)",
            "distance_gate": "Embedding(5, q_dim), gate = 1 + tanh(...)",
            "pair_pool": "distance-conditioned mean/std/log-count per centre",
            "U": "center_update Linear(patch_hidden+center_context_width, hidden) "
            "-> LayerNorm -> ReLU -> Dropout -> Linear(..., patch_hidden)",
            "readout": "unary moments + distance-conditioned pair moments + "
            "global encoder + topology encoder",
        },
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# raw-record access
# ---------------------------------------------------------------------------


def load_raw_records():
    return ztraining.load_train_valid_records()


def load_encoded():
    return rec.load_encoded()


# ---------------------------------------------------------------------------
# raw-semantic control transforms (no new feature family)
# ---------------------------------------------------------------------------


def graph_bond_marginal(record: Any) -> np.ndarray:
    """Graph-level bond-type marginal from the adjacent-bond block.

    Every graph edge is exactly one distance-1 centre pair, so summing the
    (0/1) adjacent-bond one-hots over all pairs of a ``GraphRecord`` gives the
    exact per-molecule bond-type counts.  No graph object is needed.
    """
    relation = np.asarray(record.pair_relation, dtype=np.float64)
    if relation.shape[0] == 0:
        return np.zeros(BOND_CATEGORIES, dtype=np.float32)
    counts = relation[:, ADJACENT_SLICE[0]:ADJACENT_SLICE[1]].sum(axis=0)
    total = float(counts.sum())
    if total <= 0.0:
        return np.zeros(BOND_CATEGORIES, dtype=np.float32)
    return (counts / total).astype(np.float32)


def marginalize_pair_relation(relation: np.ndarray, bond_marginal: np.ndarray) -> np.ndarray:
    """Preserve pure-S columns; destroy the pair-specific bond assignment.

    * shortest-path bond composition -> molecule bond marginal;
    * adjacent bond type -> ``(adjacency mass) * molecule bond marginal`` so the
      pure-S adjacency indicator (the block mass) survives while the bond type
      assignment is destroyed.
    """
    relation = np.asarray(relation, dtype=np.float32)
    out = relation.copy()
    out[PATH_BOND_SLICE[0]:PATH_BOND_SLICE[1]] = bond_marginal
    adjacency_mass = float(relation[ADJACENT_SLICE[0]:ADJACENT_SLICE[1]].sum())
    out[ADJACENT_SLICE[0]:ADJACENT_SLICE[1]] = adjacency_mass * bond_marginal
    return out


def marginalize_patch_descriptor(descriptor: np.ndarray) -> np.ndarray:
    """Replace shell/root chemistry by patch marginals, keep S and masses.

    * atom_shell[s, a] -> shell_mass[s] * patch_atom_marginal[a]
    * bond_shell[p, b] -> shell_pair_mass[p] * patch_bond_marginal[b]
    * root_atom         -> patch_atom_marginal
    * incident_bonds    -> patch_bond_marginal
    * scalars           -> bit-identical

    Row sums (shell mass / shell-pair mass) and column sums (patch atom / bond
    marginal) are both preserved exactly, so no new information family is
    created and no marginal information is lost.
    """
    descriptor = np.asarray(descriptor, dtype=np.float32)
    out = descriptor.copy()
    atom = descriptor[0:84].reshape(len(ATOM_SHELL_SLICES), ATOM_CATEGORIES).astype(
        np.float64
    )
    shell_mass = atom.sum(axis=1, keepdims=True)
    atom_marginal = atom.sum(axis=0, keepdims=True)
    out[0:84] = (shell_mass * atom_marginal).reshape(-1).astype(np.float32)

    bond = (
        descriptor[BOND_SHELL_SLICE[0]:BOND_SHELL_SLICE[1]]
        .reshape(len(SHELL_PAIRS), BOND_CATEGORIES)
        .astype(np.float64)
    )
    shell_pair_mass = bond.sum(axis=1, keepdims=True)
    bond_marginal = bond.sum(axis=0, keepdims=True)
    out[BOND_SHELL_SLICE[0]:BOND_SHELL_SLICE[1]] = (
        (shell_pair_mass * bond_marginal).reshape(-1).astype(np.float32)
    )

    out[ROOT_ATOM_SLICE[0]:ROOT_ATOM_SLICE[1]] = atom_marginal.reshape(-1).astype(
        np.float32
    )
    out[INCIDENT_BONDS_SLICE[0]:INCIDENT_BONDS_SLICE[1]] = bond_marginal.reshape(
        -1
    ).astype(np.float32)
    return out


def pair_b_marginal_records(records: Sequence[Any]) -> list[Any]:
    output: list[Any] = []
    for record in records:
        marginal = graph_bond_marginal(record)
        relation = np.asarray(record.pair_relation, dtype=np.float32)
        if relation.shape[0]:
            controlled = np.stack(
                [marginalize_pair_relation(row, marginal) for row in relation], axis=0
            ).astype(np.float32, copy=False)
        else:
            controlled = relation.reshape(0, RELATION_WIDTH)
        output.append(replace(record, pair_relation=controlled))
    return output


def patch_b_marginal_records(records: Sequence[Any]) -> list[Any]:
    output: list[Any] = []
    for record in records:
        patches = tuple(
            replace(
                patch,
                shell_descriptor=marginalize_patch_descriptor(patch.shell_descriptor),
            )
            for patch in record.patches
        )
        output.append(replace(record, patches=patches))
    return output


def control_records(kind: str):
    train_records, valid_records = load_raw_records()
    kind = str(kind)
    if kind == "none":
        return list(train_records), list(valid_records)
    if kind == "pair_b_marginal":
        return pair_b_marginal_records(train_records), pair_b_marginal_records(
            valid_records
        )
    if kind == "patch_b_marginal":
        return patch_b_marginal_records(train_records), patch_b_marginal_records(
            valid_records
        )
    raise ValueError(f"unknown control kind {kind!r}")


# ---------------------------------------------------------------------------
# encoded control data
# ---------------------------------------------------------------------------


def control_encoded(kind: str, *, raw_records=None):
    """Encoded (train, valid, audit) for a control representation.

    ``none`` / ``pair_b_marginal`` reuse the canonical encoded split and only
    replace the raw (never standardized) ``pair_relation`` tensor, so every
    other field is bit-identical.  ``patch_b_marginal`` re-fits the patch
    standardizer on the marginalized train split through the canonical
    ``_phase_data`` path (the representation *is* the experimental variable).
    """
    kind = str(kind)
    if kind == "none":
        train, valid, audit = rec.load_encoded()
        return list(train), list(valid), dict(audit)

    if kind == "pair_b_marginal":
        train_raw, valid_raw = (
            load_raw_records() if raw_records is None else raw_records
        )
        train, valid, audit = rec.load_encoded()
        return (
            _replace_pair_relation(train, train_raw),
            _replace_pair_relation(valid, valid_raw),
            dict(audit),
        )

    if kind == "patch_b_marginal":
        train_raw, valid_raw = (
            control_records("patch_b_marginal")
            if raw_records is None
            else raw_records
        )
        config = shead.base_config()
        config["test_policy"] = "no_test"
        config["model"]["device"] = "cpu"
        train, valid, audit = ztraining.build_encoded(
            train_raw, valid_raw, config
        )
        return list(train), list(valid), dict(audit)

    raise ValueError(f"unknown control kind {kind!r}")


def _replace_pair_relation(encoded: Sequence[Data], raw_records: Sequence[Any]):
    if len(encoded) != len(raw_records):
        raise RuntimeError(
            f"encoded/raw split size mismatch: {len(encoded)} vs {len(raw_records)}"
        )
    output: list[Data] = []
    for data, record in zip(encoded, raw_records):
        marginal = graph_bond_marginal(record)
        relation = np.asarray(record.pair_relation, dtype=np.float32)
        if relation.shape[0]:
            controlled = np.stack(
                [marginalize_pair_relation(row, marginal) for row in relation], axis=0
            ).astype(np.float32, copy=False)
        else:
            controlled = relation.reshape(0, RELATION_WIDTH)
        clone = data.clone()
        clone.pair_relation = torch.from_numpy(controlled)
        output.append(clone)
    return output


def patch_b_valid_original_scale():
    """Stage-A frozen patch-B control: original train-fit standardizer.

    Preserved columns stay bit-identical because the *original* per-column
    scaler is reused instead of refitting on the marginalized split.
    """
    train_raw, valid_raw = load_raw_records()
    standardizer = zpp.Standardizer.fit(zpp._patch_matrix(train_raw))
    _train, valid, _audit = rec.load_encoded()
    controlled: list[Data] = []
    for data, record in zip(valid, valid_raw):
        original_matrix = zpp._patch_matrix([record])
        reconstructed = standardizer.transform(original_matrix)
        original = data.patch_cont.detach().cpu().numpy()
        max_reconstruction_error = float(np.abs(reconstructed - original).max())
        if max_reconstruction_error > 1.0e-5:
            raise RuntimeError(
                "patch standardizer reconstruction drifted: "
                f"{max_reconstruction_error}"
            )
        marginal_matrix = np.stack(
            [
                marginalize_patch_descriptor(patch.shell_descriptor)
                for patch in record.patches
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        clone = data.clone()
        clone.patch_cont = torch.from_numpy(standardizer.transform(marginal_matrix))
        controlled.append(clone)
    return controlled


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def load_bnull_soup(seed: int = SEED) -> nn.Module:
    if int(seed) != SEED:
        raise ValueError("only seed0 B-Null soup is available")
    if not BNULL_SOUP_PATH.exists():
        raise FileNotFoundError(
            f"frozen B-Null soup not found at {BNULL_SOUP_PATH}; sync it first"
        )
    model = ltn.build_null(SEED)
    state = torch.load(BNULL_SOUP_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def build_noparent(seed: int = 0) -> nn.Module:
    """B-Null backbone whose radius-1 parent embedding is exactly zero/dead.

    The embedding module is kept so the patch-encoder input shape and total
    parameter count are unchanged (49,343); its weight is zeroed and frozen, so
    the parent slot is an exact zero for every patch and no released parameter
    is re-invested anywhere.
    """
    model = ltn.build_null(int(seed))
    with torch.no_grad():
        for parameter in model.parent_embedding.parameters():
            parameter.zero_()
    for parameter in model.parent_embedding.parameters():
        parameter.requires_grad_(False)
    return model


def build_t1(seed: int = 0) -> nn.Module:
    """Same B-Null backbone with a single weight-tied pair-centre round."""
    baseline = ltn.build_null(int(seed))
    baseline_state = {
        key: value.detach().clone() for key, value in baseline.state_dict().items()
    }
    shead._seed_everything(int(seed))
    kwargs = shead._base_kwargs()
    kwargs["patch_hidden"] = ltn.H_DIM
    kwargs["patch_encoder_hidden"] = ltn.PATCH_ENCODER_HIDDEN
    kwargs["global_encoder_hidden"] = ltn.GLOBAL_ENCODER_HIDDEN
    kwargs.pop("pair_hidden", None)
    from tracks.ksvd.experiments.luyin16 import (
        zinc_compact_v4_recurrent_pair_centre_capacity as cap,
    )

    with cap._construction_guards(ltn.H_DIM, ltn.Q_DIM):
        model = rec.PatchPathRecurrentPairCentreModel(
            ltn.TYPED_VOCABULARY_SIZE,
            ltn.PARENT_VOCABULARY_SIZE,
            pair_hidden=ltn.Q_DIM,
            recurrence_rounds=1,
            recurrence_enabled=True,
            recurrence_mode="refresh",
            patch_representation="null",
            **kwargs,
        )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in baseline_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)
    torch.manual_seed(int(shead.SMALL_HEAD_SEED))
    model.head = shead.GenericReader(
        int(model.unified_graph_width), shead.SMALL_HEAD_HIDDEN
    )
    return model


BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    "B-Null": ltn.build_null,
    "NoParent": build_noparent,
    "Pair-B-Marginal": ltn.build_null,
    "Patch-B-Marginal": ltn.build_null,
    "T1": build_t1,
}

TRAIN_KINDS: dict[str, str] = {
    "B-Null": "none",
    "NoParent": "none",
    "Pair-B-Marginal": "pair_b_marginal",
    "Patch-B-Marginal": "patch_b_marginal",
    "T1": "none",
}


# ---------------------------------------------------------------------------
# forward interventions
# ---------------------------------------------------------------------------


class zero_parent_embedding:
    """Context manager that forces ``parent_embedding`` output to exact zeros."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handle = None

    def __enter__(self):
        def hook(_module, _inputs, output):
            return torch.zeros_like(output)

        self.handle = self.model.parent_embedding.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        return False


def _evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    parent_null: bool = False,
    t1: bool = False,
    patch_cont_override: torch.Tensor | None = None,
    pair_relation_override: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    patch_seen = 0
    pair_seen = 0
    with torch.no_grad():
        for batch in loader:
            n_patches = int(batch.patch_cont.shape[0])
            n_pairs = int(batch.pair_relation.shape[0])
            if patch_cont_override is not None or pair_relation_override is not None:
                batch = batch.clone()
                if patch_cont_override is not None:
                    batch.patch_cont = patch_cont_override[
                        patch_seen:patch_seen + n_patches
                    ]
                if pair_relation_override is not None:
                    batch.pair_relation = pair_relation_override[
                        pair_seen:pair_seen + n_pairs
                    ]
            patch_seen += n_patches
            pair_seen += n_pairs
            batch = batch.to(device)
            targets.append(batch.y.view(-1).cpu().numpy())
            if t1:
                prediction = model.head(model.encode_original(batch)).view(-1)
            elif parent_null:
                with zero_parent_embedding(model):
                    prediction = model(batch).view(-1)
            else:
                prediction = model(batch).view(-1)
            predictions.append(prediction.cpu().numpy())
    return (
        np.concatenate(targets).astype(np.float64),
        np.concatenate(predictions).astype(np.float64),
    )


def _predictions_sha256(predictions: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(predictions, dtype=np.float64).tobytes()
    ).hexdigest()


def evaluate_state(
    state: Mapping[str, torch.Tensor],
    valid_data: Sequence[Data],
    device: str = "cpu",
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    model = ltn.build_null(SEED)
    model.load_state_dict(dict(state), strict=True)
    loader = zpp._make_loader(
        list(valid_data), int(shead.OPTIMIZED_PROTOCOL["batch_size"]), False, 0
    )
    return _evaluate(model, loader, torch.device(device), **kwargs)


def _valid_loader(valid_data: Sequence[Data]):
    return zpp._make_loader(
        list(valid_data), int(shead.OPTIMIZED_PROTOCOL["batch_size"]), False, 0
    )


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------


def stage_a(device: str = "cpu") -> dict[str, Any]:
    torch.set_num_threads(4)
    device_obj = torch.device(device)
    _train, valid_original, _audit = rec.load_encoded()
    soup_state = torch.load(BNULL_SOUP_PATH, map_location="cpu", weights_only=True)

    model = ltn.build_null(SEED).to(device_obj)
    model.load_state_dict(soup_state, strict=True)
    model.eval()
    loader = _valid_loader(valid_original)

    targets, original_preds = _evaluate(model, loader, device_obj)
    original_mae = _mae(targets, original_preds)

    # A1 parent-null
    _t, parent_preds = _evaluate(model, loader, device_obj, parent_null=True)
    parent_mae = _mae(targets, parent_preds)

    # A2 pair-B marginalized (raw pair_relation, original encoded split)
    pair_valid = control_encoded("pair_b_marginal")[1]
    pair_override = torch.cat(
        [data.pair_relation.detach().cpu() for data in pair_valid], dim=0
    )
    _t, pair_preds = _evaluate(
        model, loader, device_obj, pair_relation_override=pair_override
    )
    pair_mae = _mae(targets, pair_preds)

    # A3 patch-B marginalized (original train-fit standardizer)
    patch_valid = patch_b_valid_original_scale()
    patch_override = torch.cat(
        [data.patch_cont.detach().cpu() for data in patch_valid], dim=0
    )
    _t, patch_preds = _evaluate(
        model, loader, device_obj, patch_cont_override=patch_override
    )
    patch_mae = _mae(targets, patch_preds)

    # A4 T=1 frozen recurrence
    _t, t1_preds = _evaluate(model, loader, device_obj, t1=True)
    t1_mae = _mae(targets, t1_preds)

    recorded = (
        _read_json(BNULL_SOUP_JSON) if BNULL_SOUP_JSON.exists() else {}
    )
    interventions = {
        "parent_null": parent_mae,
        "pair_b_marginal": pair_mae,
        "patch_b_marginal": patch_mae,
        "t1_frozen": t1_mae,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "A_frozen_screen",
        "seed": SEED,
        "parameters": BNULL_PARAMS,
        "official_test_loaded": False,
        "git_commit": _git_commit(),
        "device": str(device),
        "environment": _environment_fingerprint(device),
        "original_valid_mae": float(original_mae),
        "recorded_soup_valid_mae": (
            None
            if not recorded
            else float(recorded["top5_soup_valid_mae"])
        ),
        "recomputed_minus_recorded": (
            None
            if not recorded
            else float(original_mae - float(recorded["top5_soup_valid_mae"]))
        ),
        "valid_targets_sha256": _predictions_sha256(targets),
        "original_predictions_sha256": _predictions_sha256(original_preds),
        "interventions": {
            name: {
                "valid_mae": float(value),
                "delta_vs_original": float(value - original_mae),
                "predictions_sha256": _predictions_sha256(
                    {
                        "parent_null": parent_preds,
                        "pair_b_marginal": pair_preds,
                        "patch_b_marginal": patch_preds,
                        "t1_frozen": t1_preds,
                    }[name]
                ),
            }
            for name, value in interventions.items()
        },
    }
    _write_json(RESULTS_DIR / "stage_a_bnull_seed0.json", payload)
    return payload


def stage_a_integrity() -> dict[str, Any]:
    """Bit-level integrity checks + saved before/after tensor examples."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    _train_raw, valid_raw = load_raw_records()
    _train, valid_original, _audit = rec.load_encoded()

    checks: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "A_integrity",
        "official_test_loaded": False,
    }

    model = load_bnull_soup(SEED)
    loader = _valid_loader(valid_original)

    # ---- A1 parent-null: output exactly zero; other blocks identical --------
    batch = next(iter(loader))
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, inputs, _output):
        captured["input"] = inputs[0].detach().clone()

    handle = model.patch_encoder.register_forward_hook(capture)
    with torch.no_grad():
        model(batch)
    handle.remove()
    base_input = captured["input"].clone()

    handle = model.patch_encoder.register_forward_hook(capture)
    with torch.no_grad(), zero_parent_embedding(model):
        model(batch)
    handle.remove()
    parent_input = captured["input"].clone()

    parent_width = int(model.parent_embedding.embedding_dim)
    parent_start = int(model.shell_width) + int(model.context_width) + int(
        model.token_width
    )
    parent_slice = slice(parent_start, parent_start + parent_width)
    others = [
        index
        for index in range(base_input.shape[1])
        if not (parent_start <= index < parent_start + parent_width)
    ]
    checks["parent_null"] = {
        "parent_width": parent_width,
        "parent_columns": [int(parent_start), int(parent_start + parent_width)],
        "parent_block_exactly_zero": bool(
            torch.equal(
                parent_input[:, parent_slice],
                torch.zeros_like(parent_input[:, parent_slice]),
            )
        ),
        "non_parent_columns_bit_identical": bool(
            torch.equal(base_input[:, others], parent_input[:, others])
        ),
        "non_parent_max_abs_diff": float(
            (base_input[:, others] - parent_input[:, others]).abs().max()
        ),
    }
    torch.save(
        {
            "before_non_parent": base_input[:, others][:8].detach().cpu(),
            "before_parent_block": base_input[:, parent_slice][:8].detach().cpu(),
            "after_parent_block": parent_input[:, parent_slice][:8].detach().cpu(),
        },
        EXAMPLES_DIR / "a1_parent_null_example.pt",
    )

    # ---- A2 pair-B: pure-S bit-identical; binding destroyed; marginal kept --
    pair_valid = control_encoded("pair_b_marginal")[1]
    pure_s_identical = True
    for record, data in zip(valid_raw, pair_valid):
        original = np.asarray(record.pair_relation, dtype=np.float32)
        controlled = data.pair_relation.detach().cpu().numpy().astype(np.float32)
        for start, stop in PAIR_S_SLICES:
            if not np.array_equal(original[:, start:stop], controlled[:, start:stop]):
                pure_s_identical = False
    destroyed = 0
    marginal_preserved = 0
    adjacency_mass_preserved = True
    graphs_checked = 0
    for record, data in zip(valid_raw, pair_valid):
        relation = data.pair_relation.detach().cpu().numpy().astype(np.float64)
        if relation.shape[0] == 0:
            continue
        graphs_checked += 1
        marginal = graph_bond_marginal(record).astype(np.float64)
        path_blocks = relation[:, PATH_BOND_SLICE[0]:PATH_BOND_SLICE[1]]
        unique_path = np.unique(np.round(path_blocks, 6), axis=0)
        if unique_path.shape[0] == 1 and np.allclose(
            unique_path[0], marginal, atol=1e-5
        ):
            destroyed += 1
        original = np.asarray(record.pair_relation, dtype=np.float64)
        if np.allclose(
            original[:, ADJACENT_SLICE[0]:ADJACENT_SLICE[1]].sum(axis=0),
            relation[:, ADJACENT_SLICE[0]:ADJACENT_SLICE[1]].sum(axis=0),
            atol=1e-5,
        ):
            marginal_preserved += 1
        if not np.isclose(
            original[:, ADJACENT_SLICE[0]:ADJACENT_SLICE[1]].sum(),
            relation[:, ADJACENT_SLICE[0]:ADJACENT_SLICE[1]].sum(),
        ):
            adjacency_mass_preserved = False
    checks["pair_b_marginal"] = {
        "pure_s_columns_bit_identical": bool(pure_s_identical),
        "graphs_checked": int(graphs_checked),
        "graphs_with_single_path_bond_block_equal_to_marginal": int(destroyed),
        "graphs_with_graph_level_adjacent_counts_preserved": int(marginal_preserved),
        "adjacency_mass_preserved": bool(adjacency_mass_preserved),
    }
    torch.save(
        {
            "before_pair_relation": torch.from_numpy(
                np.asarray(valid_raw[0].pair_relation, dtype=np.float32)
            ),
            "after_pair_relation": pair_valid[0].pair_relation.detach().cpu(),
        },
        EXAMPLES_DIR / "a2_pair_b_example.pt",
    )

    # ---- A3 patch-B: marginals + scalars preserved, assignment destroyed ----
    patch_valid = patch_b_valid_original_scale()
    atom_marg_ok = True
    bond_marg_ok = True
    row_mass_ok = True
    raw_scalars_ok = True
    assignment_changed = 0
    total_patches = 0
    for record in valid_raw:
        for patch in record.patches:
            before_raw = np.asarray(patch.shell_descriptor, dtype=np.float64)
            after_raw = marginalize_patch_descriptor(before_raw).astype(np.float64)
            before_atom = before_raw[0:84].reshape(3, ATOM_CATEGORIES)
            after_atom = after_raw[0:84].reshape(3, ATOM_CATEGORIES)
            before_bond = before_raw[84:108].reshape(6, BOND_CATEGORIES)
            after_bond = after_raw[84:108].reshape(6, BOND_CATEGORIES)
            total_patches += 1
            if not np.allclose(before_atom.sum(axis=0), after_atom.sum(axis=0), atol=1e-5):
                atom_marg_ok = False
            if not np.allclose(before_bond.sum(axis=0), after_bond.sum(axis=0), atol=1e-5):
                bond_marg_ok = False
            if not np.allclose(
                before_atom.sum(axis=1), after_atom.sum(axis=1), atol=1e-5
            ) or not np.allclose(
                before_bond.sum(axis=1), after_bond.sum(axis=1), atol=1e-5
            ):
                row_mass_ok = False
            if not np.array_equal(before_raw[140:146], after_raw[140:146]):
                raw_scalars_ok = False
            if not np.allclose(before_atom, after_atom, atol=1e-6):
                assignment_changed += 1

    # standardized pure-S scalars must be bit-identical (original scaler reused)
    standardized_scalars_identical = True
    for original, controlled in zip(valid_original, patch_valid):
        before = original.patch_cont.detach().cpu().numpy()[:, SCALARS_SLICE[0]:SCALARS_SLICE[1]]
        after = controlled.patch_cont.detach().cpu().numpy()[:, SCALARS_SLICE[0]:SCALARS_SLICE[1]]
        if not np.array_equal(before, after):
            standardized_scalars_identical = False
    checks["patch_b_marginal"] = {
        "per_patch_atom_marginal_preserved": bool(atom_marg_ok),
        "per_patch_bond_marginal_preserved": bool(bond_marg_ok),
        "shell_and_shell_pair_mass_preserved": bool(row_mass_ok),
        "raw_scalars_bit_identical": bool(raw_scalars_ok),
        "standardized_scalars_bit_identical": bool(standardized_scalars_identical),
        "input_width_unchanged": int(PATCH_CONT_WIDTH),
        "patches_checked": int(total_patches),
        "patches_with_shell_assignment_changed": int(assignment_changed),
    }
    torch.save(
        {
            "before_patch_cont": valid_original[0].patch_cont.detach().cpu(),
            "after_patch_cont": patch_valid[0].patch_cont.detach().cpu(),
        },
        EXAMPLES_DIR / "a3_patch_b_example.pt",
    )

    # ---- A4 T=1: same params, exactly one Q/U round -------------------------
    model_t1 = load_bnull_soup(SEED)
    t2_counts = model_t1.module_call_counts(batch)
    t1_counts = _t1_module_call_counts(model_t1, batch)
    checks["t1_frozen"] = {
        "parameters_unchanged": int(_n_params(model_t1)) == BNULL_PARAMS,
        "t2_reference_counts": {key: int(value) for key, value in t2_counts.items()},
        "t1_counts": {key: int(value) for key, value in t1_counts.items()},
        "exactly_one_q_round": int(t1_counts["pair_encoder"]) == 1,
        "exactly_one_u_round": int(t1_counts["center_update"]) == 1,
        "t2_had_two_rounds": int(t2_counts["center_update"]) == 2
        and int(t2_counts["pair_encoder"]) == 2,
    }

    # ---- determinism / finiteness -------------------------------------------
    soup_state = torch.load(BNULL_SOUP_PATH, map_location="cpu", weights_only=True)
    targets_a, preds_a = evaluate_state(soup_state, valid_original, device="cpu")
    targets_b, preds_b = evaluate_state(soup_state, valid_original, device="cpu")
    checks["determinism"] = {
        "repeat_predictions_bit_identical": bool(np.array_equal(preds_a, preds_b)),
        "finite": bool(np.isfinite(preds_a).all()),
        "valid_targets_identical": bool(np.array_equal(targets_a, targets_b)),
    }
    checks["all_pass"] = bool(
        checks["parent_null"]["parent_block_exactly_zero"]
        and checks["parent_null"]["non_parent_columns_bit_identical"]
        and checks["pair_b_marginal"]["pure_s_columns_bit_identical"]
        and checks["pair_b_marginal"]["adjacency_mass_preserved"]
        and checks["patch_b_marginal"]["per_patch_atom_marginal_preserved"]
        and checks["patch_b_marginal"]["per_patch_bond_marginal_preserved"]
        and checks["patch_b_marginal"]["shell_and_shell_pair_mass_preserved"]
        and checks["patch_b_marginal"]["raw_scalars_bit_identical"]
        and checks["patch_b_marginal"]["standardized_scalars_bit_identical"]
        and checks["t1_frozen"]["exactly_one_q_round"]
        and checks["t1_frozen"]["exactly_one_u_round"]
        and checks["t1_frozen"]["t2_had_two_rounds"]
        and checks["determinism"]["repeat_predictions_bit_identical"]
        and checks["determinism"]["finite"]
    )
    _write_json(RESULTS_DIR / "stage_a_integrity.json", checks)
    return checks
def _t1_module_call_counts(model: nn.Module, data: Data) -> dict[str, int]:
    counts = {"pair_projection": 0, "pair_encoder": 0, "center_update": 0}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, _output):
            counts[name] += 1

        return hook

    for name in counts:
        module = getattr(model, name, None)
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model.encode_original(data)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return counts


# ---------------------------------------------------------------------------
# Stage A interpretation / ticket selection (pre-registered rule)
# ---------------------------------------------------------------------------

TINY_EFFECT = 0.003
T1_LARGE_DEGRADATION = 0.020


def stage_a_gate() -> dict[str, Any]:
    payload = _read_json(RESULTS_DIR / "stage_a_bnull_seed0.json")
    interventions = payload["interventions"]
    deltas = {
        "parent_null": float(interventions["parent_null"]["delta_vs_original"]),
        "pair_b_marginal": float(interventions["pair_b_marginal"]["delta_vs_original"]),
        "patch_b_marginal": float(
            interventions["patch_b_marginal"]["delta_vs_original"]
        ),
        "t1_frozen": float(interventions["t1_frozen"]["delta_vs_original"]),
    }

    def band(value: float) -> str:
        if abs(value) < TINY_EFFECT:
            return "NO_CLEAR_INCREMENTAL_SIGNAL"
        if value >= 0.030:
            return "VERY_IMPORTANT"
        if value >= 0.010:
            return "MATERIAL"
        if value >= TINY_EFFECT:
            return "SMALL_TO_MODERATE"
        return "CANDIDATE_BETTER_NEGATIVE"

    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "A_gate",
        "original_valid_mae": float(payload["original_valid_mae"]),
        "deltas": deltas,
        "bands": {name: band(value) for name, value in deltas.items()},
        "tiny_effect_threshold": TINY_EFFECT,
        "t1_large_degradation_threshold": T1_LARGE_DEGRADATION,
        "official_test_loaded": False,
    }


def select_tickets() -> dict[str, Any]:
    """Apply the pre-registered training-ticket selection rule to Stage A."""
    gate = stage_a_gate()
    deltas = gate["deltas"]
    frozen = {
        "NoParent": deltas["parent_null"],
        "Pair-B-Marginal": deltas["pair_b_marginal"],
        "Patch-B-Marginal": deltas["patch_b_marginal"],
    }
    tickets = list(TICKETS)
    replaced = None
    reason = "default_tickets"
    if deltas["t1_frozen"] >= T1_LARGE_DEGRADATION and min(
        abs(value) for value in frozen.values()
    ) < TINY_EFFECT:
        weakest = min(frozen, key=lambda name: abs(frozen[name]))
        tickets = [name for name in tickets if name != weakest] + ["T1"]
        replaced = weakest
        reason = "t1_exception_triggered"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "B_ticket_selection",
        "rule": (
            "default tickets NoParent/Pair-B-Marginal/Patch-B-Marginal; if "
            "frozen T=1 delta >= 0.020 AND some frozen |delta| < 0.003, replace "
            "the smallest-|delta| ticket by a T=1 retrain; still <= 3 runs"
        ),
        "frozen_deltas": frozen,
        "t1_frozen_delta": deltas["t1_frozen"],
        "tickets": tickets,
        "replaced": replaced,
        "reason": reason,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage_b_ticket_selection.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage B -- retraining
# ---------------------------------------------------------------------------


def tag_for(ticket: str) -> str:
    return TICKET_TAGS[str(ticket)]


def _train_ticket(ticket: str, seed: int = SEED, device: str = "cpu") -> dict[str, Any]:
    ticket = str(ticket)
    if ticket not in TICKET_TAGS:
        raise ValueError(f"unknown ticket {ticket!r}")
    train_data, valid_data, _audit = control_encoded(TRAIN_KINDS[ticket])
    build_fn = BUILDERS[ticket]
    tag = tag_for(ticket)
    original = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = (
        CURVE_DIR,
        STATE_DIR,
        RUNS_DIR,
    )
    snapshot_dir = SNAPSHOT_DIR / f"{tag}_seed{int(seed)}"
    started = time.perf_counter()
    try:
        summary = shead.train_model(
            build_fn=lambda s: build_fn(int(s)),
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=tag,
            save_state=True,
            real_batch_identity=False,
            expected_total=ltn.NULL_TOTAL_PARAMS,
            snapshot_dir=snapshot_dir,
            device=device,
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original
    summary = dict(summary)
    summary["ticket"] = ticket
    summary["tag"] = tag
    summary["seed"] = int(seed)
    summary["control_kind"] = TRAIN_KINDS[ticket]
    summary["epoch_time_s"] = float(summary["wall_clock_s"]) / max(
        int(summary["epochs_run"]), 1
    )
    summary["peak_gpu_memory_mb"] = (
        float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if torch.cuda.is_available()
        else 0.0
    )
    summary["outer_wall_clock_s"] = float(time.perf_counter() - started)
    summary["official_test_loaded"] = False
    _write_json(RUNS_DIR / f"{tag}_seed{int(seed)}.json", summary)
    soup_ticket(ticket, int(seed))
    return summary


def soup_ticket(ticket: str, seed: int = SEED) -> dict[str, Any]:
    ticket = str(ticket)
    tag = tag_for(ticket)
    summary = _read_json(RUNS_DIR / f"{tag}_seed{int(seed)}.json")
    top = vd._top5_epochs(summary)
    soup_state = vd.build_soup_state(top)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = SOUP_DIR / f"{tag}_seed{int(seed)}_top5_soup.pt"
    torch.save(soup_state, soup_path)

    train_data, valid_data, _audit = control_encoded(TRAIN_KINDS[ticket])
    loader = _valid_loader(valid_data)
    model = BUILDERS[ticket](int(seed))
    selection_path = STATE_DIR / f"{tag}_seed{int(seed)}_selection_state.pt"
    targets, best_preds = vd._predict_state(
        model, torch.load(selection_path, map_location="cpu", weights_only=True), loader
    )
    _t2, soup_preds = vd._predict_state(model, soup_state, loader)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "ticket": ticket,
        "control_kind": TRAIN_KINDS[ticket],
        "tag": tag,
        "seed": int(seed),
        "top5_epochs": [int(row["epoch"]) for row in top],
        "top5_valid_mae": [float(row["valid_mae"]) for row in top],
        "best_epoch": int(summary["best_epoch"]),
        "best_checkpoint_valid_mae": _mae(targets, best_preds),
        "top5_soup_valid_mae": _mae(targets, soup_preds),
        "train_loss_at_best": float(summary.get("train_loss_at_best", float("nan"))),
        "epochs_run": int(summary.get("epochs_run", 0)),
        "wall_clock_s": float(summary.get("wall_clock_s", float("nan"))),
        "peak_gpu_memory_mb": float(summary.get("peak_gpu_memory_mb", 0.0)),
        "parameters": int(_n_params(model)),
        "official_test_loaded": False,
        "valid_targets": targets.tolist(),
        "best_predictions": best_preds.tolist(),
        "soup_predictions": soup_preds.tolist(),
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{int(seed)}.json", payload)
    return payload


def train_queue(tickets: Sequence[str], seed: int = SEED, device: str = "cpu") -> None:
    for ticket in tickets:
        tag = tag_for(ticket)
        if (RUNS_DIR / f"{tag}_seed{int(seed)}.json").exists() and (
            SOUP_DIR / f"{tag}_seed{int(seed)}_top5_soup.pt"
        ).exists():
            print(f"skip existing {ticket} seed{seed}", flush=True)
            continue
        print(f"=== train {ticket} seed{seed} device={device} ===", flush=True)
        _train_ticket(ticket, int(seed), device=device)


# ---------------------------------------------------------------------------
# witness
# ---------------------------------------------------------------------------


_MODULE_GROUPS = (
    "patch_encoder",
    "parent_embedding",
    "pair_projection",
    "relation_encoder",
    "distance_gate",
    "pair_encoder",
    "center_update",
    "global_encoder",
    "topology_encoder",
    "head",
)


def _gradient_audit(model: nn.Module, batch: Data, device: torch.device) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    batch = batch.to(device)
    prediction = model(batch).view(-1)
    target = batch.y.view(-1)
    loss = (prediction - target).abs().mean()
    loss.backward()
    result: dict[str, Any] = {"loss": float(loss.detach()), "groups": {}}
    for name in _MODULE_GROUPS:
        module = getattr(model, name, None)
        parameters = [] if module is None else list(module.parameters())
        norm = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                norm += float(parameter.grad.detach().norm() ** 2)
        result["groups"][name] = {
            "params": int(sum(p.numel() for p in parameters)),
            "grad_norm": float(norm**0.5),
        }
    result["all_groups_nonzero"] = all(
        row["grad_norm"] > 0.0
        for name, row in result["groups"].items()
        if row["params"] > 0 and name != "parent_embedding"
    )
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


def witness(ticket: str, seed: int = SEED, device: str = "cpu") -> dict[str, Any]:
    ticket = str(ticket)
    tag = tag_for(ticket)
    device_obj = torch.device(device)
    model = BUILDERS[ticket](int(seed)).to(device_obj)
    selection_path = STATE_DIR / f"{tag}_seed{int(seed)}_selection_state.pt"
    soup_path = SOUP_DIR / f"{tag}_seed{int(seed)}_top5_soup.pt"
    selection = torch.load(selection_path, map_location="cpu", weights_only=True)
    model.load_state_dict(selection, strict=True)
    model.eval()

    _train, valid_data, _audit = control_encoded(TRAIN_KINDS[ticket])
    loader = _valid_loader(valid_data)
    first_batch = next(iter(loader)).to(device_obj)
    with torch.no_grad():
        prediction = model(first_batch)
        token = model._patch_token_value(first_batch)
    gradient_audit = _gradient_audit(
        BUILDERS[ticket](int(seed)).to(device_obj), first_batch, device_obj
    )
    parent_parameters = list(model.parent_embedding.parameters())
    parent_norm = float(
        torch.sqrt(sum(float(p.detach().pow(2).sum()) for p in parent_parameters))
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "ticket": ticket,
        "control_kind": TRAIN_KINDS[ticket],
        "seed": int(seed),
        "total_params": int(_n_params(model)),
        "patch_encoder_input_width": int(model.patch_encoder.layers[0].in_features),
        "local_token_is_exactly_zero": bool(torch.equal(token, torch.zeros_like(token))),
        "parent_embedding_weight_norm": parent_norm,
        "parent_embedding_requires_grad": bool(
            any(p.requires_grad for p in parent_parameters)
        ),
        "recurrence_rounds": int(getattr(model, "recurrence_rounds", 1)),
        "forward_finite": bool(torch.isfinite(prediction).all()),
        "selection_state_path": str(selection_path),
        "soup_state_path": (str(soup_path) if soup_path.exists() else None),
        "gradient_audit": gradient_audit,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"witness_{tag}_seed{int(seed)}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# camera / report
# ---------------------------------------------------------------------------


def camera() -> dict[str, Any]:
    reference = None
    if BNULL_SOUP_JSON.exists():
        reference = float(_read_json(BNULL_SOUP_JSON)["top5_soup_valid_mae"])
    rows: list[dict[str, Any]] = [
        {
            "condition": "B-Null",
            "params": BNULL_PARAMS,
            "soup_valid_mae": reference,
            "delta_vs_bnull": 0.0 if reference is not None else None,
        }
    ]
    for ticket in TICKETS:
        path = RESULTS_DIR / f"soup_{tag_for(ticket)}_seed0.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        soup = float(payload["top5_soup_valid_mae"])
        rows.append(
            {
                "condition": ticket,
                "params": int(payload["parameters"]),
                "soup_valid_mae": soup,
                "delta_vs_bnull": (None if reference is None else soup - reference),
            }
        )
    t1_path = RESULTS_DIR / f"soup_{tag_for('T1')}_seed0.json"
    if t1_path.exists():
        payload = _read_json(t1_path)
        soup = float(payload["top5_soup_valid_mae"])
        rows.append(
            {
                "condition": "T1-retrained",
                "params": int(payload["parameters"]),
                "soup_valid_mae": soup,
                "delta_vs_bnull": (None if reference is None else soup - reference),
            }
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "rows": rows,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "camera.json", payload)
    return payload


def report() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "files": {},
        "official_test_loaded": False,
    }
    for name in (
        "stage_a_feature_map.json",
        "stage_a_bnull_seed0.json",
        "stage_a_integrity.json",
        "stage_a_gate.json",
        "stage_b_ticket_selection.json",
        "camera.json",
    ):
        path = RESULTS_DIR / name
        if path.exists():
            payload["files"][name] = _read_json(path)
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "feature_map",
            "sanity",
            "stage_a",
            "stage_a_integrity",
            "stage_a_gate",
            "select_tickets",
            "train",
            "train_queue",
            "soup",
            "witness",
            "camera",
            "report",
        ],
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ticket", type=str, default="NoParent")
    parser.add_argument("--tickets", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    torch.set_num_threads(4)

    if args.stage == "feature_map":
        payload = feature_map()
        _write_json(RESULTS_DIR / "stage_a_feature_map.json", payload)
        print(json.dumps(payload, indent=2, default=str), flush=True)
    if args.stage == "sanity":
        result = {"feature_map": feature_map(), "params": {}}
        for name, builder in BUILDERS.items():
            if name == "T1":
                continue
            model = builder(0)
            result["params"][name] = int(_n_params(model))
        result["soup_path"] = str(BNULL_SOUP_PATH)
        result["official_test_loaded"] = False
        _write_json(RESULTS_DIR / "sanity.json", result)
        print(json.dumps(result, indent=2, default=str), flush=True)
    if args.stage == "stage_a":
        print(json.dumps(stage_a(device=args.device), indent=2, default=str), flush=True)
    if args.stage == "stage_a_integrity":
        print(json.dumps(stage_a_integrity(), indent=2, default=str), flush=True)
    if args.stage == "stage_a_gate":
        print(json.dumps(stage_a_gate(), indent=2, default=str), flush=True)
    if args.stage == "select_tickets":
        print(json.dumps(select_tickets(), indent=2, default=str), flush=True)
    if args.stage == "train":
        print(
            json.dumps(
                _train_ticket(args.ticket, int(args.seed), device=args.device),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "train_queue":
        tickets = (
            [row for row in args.tickets.split(",") if row]
            if args.tickets
            else list(TICKETS)
        )
        train_queue(tickets, int(args.seed), args.device)
    if args.stage == "soup":
        print(
            json.dumps(soup_ticket(args.ticket, int(args.seed)), indent=2, default=str),
            flush=True,
        )
    if args.stage == "witness":
        print(
            json.dumps(
                witness(args.ticket, int(args.seed), device=args.device),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "camera":
        print(json.dumps(camera(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
