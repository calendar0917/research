"""E2E-DictEnv-BondAnchoredTriple-JointTune-v1 — joint representation + composition fine-tune.

Round ``e2e_dictenv_bond_anchored_triple_joint_tune_v1`` (study
``zinc-context-gap``).  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_bond_anchored_triple_joint_tune_v1_preregistration.md``.

One question, one candidate, one seed, 40 epochs: with the shared dictionary
coordinates (``D``, ``U``, ``common_rms``), the semantic interface and the C6
readout rule **frozen**, does jointly fine-tuning the active edge
structure-semantics binding, the local environment fusion, the pair path and
the three-environment composition + Reader push the complete predictor to
``valid soup MAE <= 0.120``?

The M1 soup of ``e2e_dictenv_bond_anchored_triple_v1`` supplies the hot start
(F, extended 366-D Reader, six fixed normalisation buffers); the original
Sem108 soup supplies the backbone.  The dynamic chain (edge binding -> fusion
-> E -> pair value -> C6 pooling -> z_old -> three-environment z3 ->
standardisation -> Reader) is recomputed **online with autograd** every step;
cached BAT-v1 ``E`` / ``p_ij`` / ``z_old`` are reference-only.

Local CPU (8 threads, float32); official valid is evaluation only and the
official test split is never instantiated.  ``official_test_loaded`` is
``False`` in every payload.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_bond_anchored_triple_v1 as bat
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = "e2e_dictenv_bond_anchored_triple_joint_tune_v1"
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_bond_anchored_triple_joint_tune_v1"
STRUCTURE_DIR = RESULTS_DIR / "structure"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_bond_anchored_triple_joint_tune_v1_preregistration.md"

# -- frozen sources -----------------------------------------------------------

BAT_RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_bond_anchored_triple_v1"
BAT_SOUP_PATH = BAT_RESULTS_DIR / "checkpoints/BAT-v1-seed0_soup_full_state.pt"
BAT_SOUP_JSON = BAT_RESULTS_DIR / "soup.json"
BAT_STANDARDIZERS_JSON = BAT_RESULTS_DIR / "standardizers.json"

SEM108_RESULTS = TRACK_ROOT / "results/e2e_dictenv_sem108_v1"
SEM108_SOUP_PATH = SEM108_RESULTS / "checkpoints/SEM108-seed0_soup_state.pt"

BAT_SOUP_MEMBERS = (50, 57, 43, 49, 71)
M_START_EXPECTED = 0.12300291641423246
M_PARENT_EXPECTED = 0.123704927947314
HOT_START_TOLERANCE = 1.0e-6
ONLINE_CACHE_TOLERANCE = 2.0e-5

TAG = "BATJ-v1-seed0"

# -- architecture (identical to the BAT-v1 triple path) -----------------------

PAIR_DIM = int(p2.PAIR_HIDDEN)                 # 16
Z_OLD_DIM = int(audit.READER_IN_DIM)           # 302
TRIPLE_SUMMARY_DIM = 64
F_IN = 3 * PAIR_DIM
F_HIDDEN = 64
F_OUT = 32
READER_IN = Z_OLD_DIM + TRIPLE_SUMMARY_DIM     # 366
READER_HIDDEN = (13, 13)
ENV_DIM = int(p2.ENV_DIM)                      # 48

#: the exact joint fine-tune whitelist (pre-registration section 2).
TRAINABLE_PREFIXES: tuple[str, ...] = (
    "backbone.W_E_S",
    "backbone.W_E_C",
    "backbone.edge_encoder.",
    "backbone.fusion.",
    "backbone.pair_projection.",
    "backbone.relation_encoder.",
    "backbone.distance_gate.",
    "backbone.pair_encoder.",
    "F.",
    "reader.",
)

EXPECTED_TRAINABLE_TENSORS = 34
EXPECTED_TRAINABLE_PARAMS = 82_805
EXPECTED_FULL_PARAMS = 103_757
EXPECTED_PARENT_PARAMS = 97_709
EXPECTED_PARENT_READER_PARAMS = 4_135
EXPECTED_REMOVED_READER_PARAMS = 4_135

# -- training budget (fixed) ---------------------------------------------------

TRAIN_SIZE = 10_000
VALID_SIZE = 1_000
EPOCHS = 40
BATCH_SIZE = int(p2run.BATCH_SIZE)             # 128
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
TRAIN_SHUFFLE_OFFSET = int(p2run.TRAIN_SHUFFLE_OFFSET)   # 91011
EVAL_SHUFFLE_OFFSET = int(p2run.EVAL_SHUFFLE_OFFSET)     # 91012
SOUP_K = 5
SOUP_EPOCH_LO = 21
SOUP_EPOCH_HI = 40
PROMISING_THRESHOLD = 0.120
THREADS = 8
FLOAT32 = torch.float32

STRUCTURE_FORMAT_VERSION = 1
EXPECTED_STRUCTURE_COUNTS = {
    "train": {"graphs": TRAIN_SIZE, "pairs": 2_668_346, "triples": 5_511_568},
    "valid": {"graphs": VALID_SIZE, "pairs": 264_776, "triples": 546_830},
}

MASK = cm.C6_MASK  # type: ignore[assignment]

VERDICT_PROMISING = "JOINT_TUNE_SCREEN_PROMISING"
VERDICT_NO_SIGNAL = "JOINT_TUNE_SCREEN_NO_STRONG_SIGNAL"
VERDICT_INVALID = "INVALID_INCOMPLETE"

SCOPE_NOTE = (
    "Single-candidate 40/40-epoch joint-adaptation performance screen with fixed "
    "dictionary coordinates.  Delta_vs_start describes this round's overall joint "
    "adaptation; it is not the triple operator's independent effect and cannot "
    "establish that the three-environment mechanism or the dictionary coordinates "
    "are irreplaceable.  No M0 retraining, matched control, shuffle, ablation, "
    "mechanism arm, extra seed or added training; the official test was never read."
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


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
        raise RuntimeError(f"BondAnchoredTriple-JointTune-v1 core is CPU-only, got {device}")


def freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def load_dictionary_and_subspace() -> tuple[np.ndarray, Any, str]:
    """The exact frozen source chain: sdb32 dictionary + q1 common subspace."""
    dictionary, dictionary_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return dictionary, bat.load_subspace(), dictionary_sha


def parent_provenance() -> dict[str, Any]:
    """Re-measure the frozen Sem108 source (local feature extractor)."""
    dictionary, subspace, dictionary_sha = load_dictionary_and_subspace()
    state = bat.load_parent_state()
    model = sem.build_sem108_model(dictionary, 0, subspace)
    result = model.load_state_dict(state)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Sem108 warm start mismatch: {result}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "path": str(SEM108_SOUP_PATH.relative_to(REPO_ROOT)),
        "checkpoint_sha256": _sha256_file(SEM108_SOUP_PATH),
        "canonical_state_sha256": audit.state_sha256(state),
        "canonical_state_sha256_expected": bat.PARENT_SOUP_SHA256,
        "state_keys": int(len(state)),
        "params": int(sum(p.numel() for p in model.parameters())),
        "reader_params": int(sum(p.numel() for p in model.reader.parameters())),
        "historical_valid_soup_mae": float(bat.PARENT_HISTORICAL_SOUP_MAE),
        "mask": MASK.as_dict(),
        "config": cm.H1_CONFIG.as_dict(),
        "dictionary_sha256": dictionary_sha,
        "subspace_kind": subspace.kind,
    }
    if payload["canonical_state_sha256"] != bat.PARENT_SOUP_SHA256:
        raise RuntimeError("Sem108 canonical state hash mismatch")
    if payload["params"] != EXPECTED_PARENT_PARAMS:
        raise RuntimeError(f"Sem108 params {payload['params']} != {EXPECTED_PARENT_PARAMS}")
    if payload["reader_params"] != EXPECTED_PARENT_READER_PARAMS:
        raise RuntimeError(
            f"Sem108 reader params {payload['reader_params']} != {EXPECTED_PARENT_READER_PARAMS}"
        )
    official_test_blocker(payload)
    return payload


def m1_soup_provenance() -> dict[str, Any]:
    """Re-measure the BAT-v1 M1 soup state (hot start) and its member metadata."""
    state = torch.load(BAT_SOUP_PATH, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError("BAT-v1 soup checkpoint is not a state mapping")
    state = {str(key): value for key, value in state.items()}
    soup_json = _read_json(BAT_SOUP_JSON)
    members = [int(v) for v in soup_json["members"]]
    if list(members) != list(BAT_SOUP_MEMBERS):
        raise RuntimeError(f"BAT-v1 soup members {members} != {list(BAT_SOUP_MEMBERS)}")
    soup_mae = float(soup_json["soup_valid_mae"])
    if abs(soup_mae - M_START_EXPECTED) > 1.0e-12:
        raise RuntimeError(f"BAT-v1 soup MAE {soup_mae} != {M_START_EXPECTED}")
    standardizers = _read_json(BAT_STANDARDIZERS_JSON)["statistics"]
    buffer_match: dict[str, bool] = {}
    for name in ("mu_p", "scale_p", "mu_old", "scale_old", "mu_3", "scale_3"):
        buffer_match[name] = bool(
            torch.equal(state[name].float(), torch.as_tensor(standardizers[name], dtype=torch.float32))
        )
    if not all(buffer_match.values()):
        raise RuntimeError(f"BAT-v1 soup buffers differ from standardizers.json: {buffer_match}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "path": str(BAT_SOUP_PATH.relative_to(REPO_ROOT)),
        "checkpoint_sha256": _sha256_file(BAT_SOUP_PATH),
        "canonical_state_sha256": audit.state_sha256(state),
        "state_keys": sorted(state),
        "soup_members": members,
        "soup_member_valid_mae": [float(v) for v in soup_json["member_valid_mae"]],
        "soup_valid_mae": float(soup_mae),
        "soup_epoch_window": [int(v) for v in soup_json["soup_epoch_window"]],
        "buffers_match_standardizers_json": buffer_match,
        "expected_M_start": float(M_START_EXPECTED),
    }
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_extended_reader() -> nn.Sequential:
    """The BAT-v1 Reader: 366 -> 13 -> 13 -> 1 (ReLU, no dropout)."""
    return nn.Sequential(
        nn.Linear(READER_IN, READER_HIDDEN[0]),
        nn.ReLU(),
        nn.Linear(READER_HIDDEN[0], READER_HIDDEN[1]),
        nn.ReLU(),
        nn.Linear(READER_HIDDEN[1], 1),
    )


class JointTuneModel(nn.Module):
    """Sem108 backbone (frozen except the whitelist) + joint F + extended Reader.

    The old 302-D Reader is removed from the module registry (``backbone.reader
    = None``), so it cannot enter ``parameters()``, the optimizer or the clip
    set.  The extended Reader and F live at wrapper level, matching the exact
    key names of the BAT-v1 M1 soup state.

    ``train()`` is overridden: the frozen feature extractor always returns to
    ``eval()`` so the original backend dropout (0.05 in ``relation_encoder`` /
    ``pair_encoder`` / ``global_encoder``) stays off, exactly as during BAT-v1
    cache extraction.  ``eval()`` does not disable autograd.
    """

    def __init__(self, dictionary: np.ndarray, seed: int = 0, subspace: Any = None) -> None:
        super().__init__()
        self.backbone = sem.build_sem108_model(dictionary, int(seed), subspace)
        self.F = nn.Sequential(
            nn.Linear(F_IN, F_HIDDEN, bias=True),
            nn.SiLU(),
            nn.Linear(F_HIDDEN, F_OUT, bias=True),
        )
        self.reader = build_extended_reader()
        self.register_buffer("mu_p", torch.zeros(PAIR_DIM))
        self.register_buffer("scale_p", torch.ones(PAIR_DIM))
        self.register_buffer("mu_old", torch.zeros(Z_OLD_DIM))
        self.register_buffer("scale_old", torch.ones(Z_OLD_DIM))
        self.register_buffer("mu_3", torch.zeros(TRIPLE_SUMMARY_DIM))
        self.register_buffer("scale_3", torch.ones(TRIPLE_SUMMARY_DIM))

    # -- mode discipline -------------------------------------------------------

    def train(self, mode: bool = True) -> "JointTuneModel":
        super().train(mode)
        self.backbone.eval()  # frozen extractor: dropout off, like cache extraction
        return self

    # -- parameter bookkeeping -------------------------------------------------

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def trainable_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    # -- forward pieces ---------------------------------------------------------

    @staticmethod
    def standardize(value: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (value - mu) / scale

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
                0, index, torch.ones((int(index.numel()), 1), dtype=t.dtype, device=t.device)
            )
        denominator = counts.clamp_min(1.0)
        return torch.cat([total / denominator, squared / denominator], dim=-1)

    # -- online forward (autograd through every trainable module) --------------

    def forward(
        self,
        data: Any,
        triple_ij: torch.Tensor,
        triple_ik: torch.Tensor,
        triple_jk: torch.Tensor,
        triple_graph: torch.Tensor,
        n_graphs: int,
        *,
        return_aux: bool = False,
    ):
        backbone = self.backbone
        with torch.no_grad():
            # fixed dictionary coordinates; D/U/common_rms are frozen
            coord = backbone.code(data.dict_phi)
        E = backbone.environments_masked(coord, data, MASK)

        batch = data.batch
        unary = audit.pool_moments_masked(E, batch, int(n_graphs), MASK.unary_zero_blocks)

        source = data.pair_index[0]
        target = data.pair_index[1]
        projected = backbone.pair_projection(E)
        left = projected[source]
        right = projected[target]
        relation_input = data.pair_relation[:, list(p1.P1_RELATION_INDICES)]
        if MASK.relation_zero_groups:
            relation_input = audit._replace_grouped_columns(
                relation_input, audit.RELATION_GROUPS, MASK.relation_zero_groups, None, "relation:"
            )
        relation = backbone.relation_encoder(relation_input)
        product = left * right
        gate = 1.0 + torch.tanh(backbone.distance_gate(data.pair_bucket))
        if MASK.gate_off:
            gate = torch.ones_like(gate)
        pair_input = torch.cat(
            [left + right, torch.abs(left - right), product * gate, relation], dim=1
        )
        pair_value = backbone.pair_encoder(pair_input)
        pair_batch = batch[source]
        relation_readout = audit.pool_pair_moments_masked(
            pair_value, pair_batch, data.pair_bucket, int(n_graphs), MASK.pair_zero_blocks
        )

        global_input = data.global_context
        if MASK.global_zero_groups:
            global_input = audit._replace_grouped_columns(
                global_input, audit.GLOBAL_GROUPS, MASK.global_zero_groups, None, "global:"
            )
        graph_hidden = backbone.global_encoder(global_input)
        topology_input = data.topology_features
        if MASK.topology_zero:
            topology_input = torch.zeros_like(topology_input)
        topology = backbone.topology_encoder(topology_input)

        z_old = torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)

        p_bar = self.standardize(pair_value, self.mu_p, self.scale_p)
        t = self.triple_embedding(
            p_bar[triple_ij.long()], p_bar[triple_ik.long()], p_bar[triple_jk.long()]
        )
        z3 = self.pool_triples(t, triple_graph, int(n_graphs))

        z3_bar = self.standardize(z3, self.mu_3, self.scale_3)
        z_old_bar = self.standardize(z_old, self.mu_old, self.scale_old)
        unified = torch.cat([z_old_bar, z3_bar], dim=1)
        prediction = self.reader(unified).view(-1)
        if return_aux:
            return prediction, {
                "E": E,
                "coord": coord,
                "pair_value": pair_value,
                "z_old": z_old,
                "z3": z3,
                "unified": unified,
            }
        return prediction


def build_joint_model(seed: int = 0, warm_start: bool = True) -> JointTuneModel:
    """Build the joint model, load the Sem108 backbone and (optionally) M1 soup."""
    dictionary, subspace, _ = load_dictionary_and_subspace()
    model = JointTuneModel(dictionary, seed=int(seed), subspace=subspace)
    result = model.backbone.load_state_dict(bat.load_parent_state())
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Sem108 state load mismatch: {result}")
    model.backbone.reader = None  # old 302-D Reader leaves the module registry
    if warm_start:
        state = torch.load(BAT_SOUP_PATH, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise RuntimeError("BAT-v1 soup checkpoint is not a state mapping")
        state = {str(key): value for key, value in state.items()}
        model_keys = set(model.state_dict())
        missing = set(state) - model_keys
        if missing:
            raise RuntimeError(f"BAT-v1 keys missing from the joint model: {sorted(missing)}")
        result = model.load_state_dict({key: value.float() for key, value in state.items()}, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"unexpected BAT-v1 keys: {result.unexpected_keys}")
        current = model.state_dict()
        for key, value in state.items():
            if not torch.equal(current[key], value.float()):
                raise RuntimeError(f"BAT-v1 warm start not applied value-for-value: {key}")
    freeze_all(model)
    return model


def configure_trainable(model: JointTuneModel) -> dict[str, Any]:
    """Unfreeze exactly the whitelist; return the measured classification table."""
    freeze_all(model)
    rows: list[dict[str, Any]] = []
    trainable_numel = 0
    trainable_tensors = 0
    for name, parameter in model.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)
        parameter.requires_grad_(trainable)
        rows.append(
            {
                "name": str(name),
                "shape": [int(v) for v in parameter.shape],
                "numel": int(parameter.numel()),
                "trainable": bool(trainable),
            }
        )
        if trainable:
            trainable_numel += int(parameter.numel())
            trainable_tensors += 1
    total_params = int(sum(row["numel"] for row in rows))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "whitelist_prefixes": list(TRAINABLE_PREFIXES),
        "rows": rows,
        "frozen_tensors": int(len(rows) - trainable_tensors),
        "frozen_params": int(total_params - trainable_numel),
        "trainable_tensors": int(trainable_tensors),
        "trainable_params": int(trainable_numel),
        "total_registered_params": int(total_params),
        "old_reader_removed_params": int(EXPECTED_REMOVED_READER_PARAMS),
        "expected": {
            "trainable_tensors": int(EXPECTED_TRAINABLE_TENSORS),
            "trainable_params": int(EXPECTED_TRAINABLE_PARAMS),
            "total_registered_params": int(EXPECTED_FULL_PARAMS),
        },
    }
    payload["passed"] = bool(
        trainable_tensors == EXPECTED_TRAINABLE_TENSORS
        and trainable_numel == EXPECTED_TRAINABLE_PARAMS
        and total_params == EXPECTED_FULL_PARAMS
    )
    official_test_blocker(payload)
    return payload


def parameter_breakdown(model: JointTuneModel | None = None) -> dict[str, Any]:
    model = build_joint_model() if model is None else model
    groups = {
        "edge_binding": ("backbone.W_E_S", "backbone.W_E_C"),
        "edge_encoder": ("backbone.edge_encoder.",),
        "fusion": ("backbone.fusion.",),
        "pair_projection": ("backbone.pair_projection.",),
        "relation_encoder": ("backbone.relation_encoder.",),
        "distance_gate": ("backbone.distance_gate.",),
        "pair_encoder": ("backbone.pair_encoder.",),
        "F": ("F.",),
        "reader": ("reader.",),
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "groups": {},
    }
    for group, prefixes in groups.items():
        numel = 0
        tensors: list[str] = []
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                numel += int(parameter.numel())
                tensors.append(name)
        payload["groups"][group] = {"numel": int(numel), "tensors": tensors}
    payload["trainable_params"] = int(sum(v["numel"] for v in payload["groups"].values()))
    payload["full_model_params"] = int(sum(p.numel() for p in model.parameters()))
    payload["total_registered_params"] = payload["full_model_params"]
    payload["parameter_ratio_trainable_over_full"] = float(
        payload["trainable_params"] / max(payload["full_model_params"], 1)
    )
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# structure cache (data-only; no trainable weights)
# ---------------------------------------------------------------------------


@dataclass
class SplitStructure:
    pair_ptr: torch.Tensor      # [G+1] cumulative pair rows
    pair_bucket: torch.Tensor   # [P] distance bucket per pair row
    n_nodes: torch.Tensor       # [G]
    n_bonds: torch.Tensor       # [G] unique real bonds (anchor rows)
    y: torch.Tensor             # [G]
    triple_ij: torch.Tensor     # [T] split-global pair-row ids
    triple_ik: torch.Tensor     # [T]
    triple_jk: torch.Tensor     # [T]
    triple_graph: torch.Tensor  # [T]
    triple_ptr: torch.Tensor    # [G+1]
    provenance: dict[str, Any]

    @property
    def n_graphs(self) -> int:
        return int(self.n_nodes.shape[0])

    @property
    def n_pairs(self) -> int:
        return int(self.pair_bucket.shape[0])

    @property
    def n_triples(self) -> int:
        return int(self.triple_ij.shape[0])


def _structure_key(split: str) -> str:
    payload = {
        "structure_format_version": int(STRUCTURE_FORMAT_VERSION),
        "protocol_version": PROTOCOL_VERSION,
        "parent_canonical_state_sha256": bat.PARENT_SOUP_SHA256,
        "mask_signature": MASK.signature(),
        "module_sha256": module_sha256(),
        "split": str(split),
        "order_policy": "official_split_order",
        "triple_rule": "unique_bond_anchor_x_third_node_m*(n-2)",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def structure_path(split: str) -> Path:
    return STRUCTURE_DIR / f"structure_{split}_{_structure_key(split)[:16]}.pt"


def build_structure(split: str) -> dict[str, Any]:
    """One data-only pass: pair buckets, anchors and exact triples per graph."""
    data_list = p1run.load_split(split)
    expected = TRAIN_SIZE if split == "train" else VALID_SIZE
    if len(data_list) != expected:
        raise RuntimeError(f"{split} split size {len(data_list)} != {expected}")
    started = time.perf_counter()
    pair_parts: list[torch.Tensor] = []
    pair_ptr = [0]
    n_nodes: list[int] = []
    n_bonds: list[int] = []
    y: list[float] = []
    ij: list[torch.Tensor] = []
    ik: list[torch.Tensor] = []
    jk: list[torch.Tensor] = []
    triple_graph: list[torch.Tensor] = []
    triple_ptr = [0]
    for graph_index, data in enumerate(data_list):
        n = int(data.dict_phi.shape[0])
        pair_index = data.pair_index.to(torch.int64)
        npairs = int(pair_index.shape[1])
        if npairs != n * (n - 1) // 2:
            raise RuntimeError(f"{split}[{graph_index}] pair count {npairs} != n(n-1)/2")
        expected_pairs = torch.tensor(
            [(a, b) for a in range(n) for b in range(a + 1, n)], dtype=torch.int64
        ).t()
        if not torch.equal(pair_index, expected_pairs):
            raise RuntimeError(f"{split}[{graph_index}] pair cache is not triangular i<j")
        bucket = data.pair_bucket.to(torch.int64)
        rows = bat.graph_triple_rows(pair_index, bucket, n)
        base = pair_ptr[-1]
        pair_parts.append(bucket)
        pair_ptr.append(base + npairs)
        n_nodes.append(n)
        n_bonds.append(int(rows["anchors"].numel()))
        y.append(float(data.y.view(-1)[0]))
        ij.append(rows["ij"] + base)
        ik.append(rows["ik"] + base)
        jk.append(rows["jk"] + base)
        triple_graph.append(torch.full((int(rows["ij"].numel()),), graph_index, dtype=torch.int64))
        triple_ptr.append(triple_ptr[-1] + int(rows["ij"].numel()))
    payload = {
        "pair_ptr": torch.as_tensor(pair_ptr, dtype=torch.int64),
        "pair_bucket": torch.cat(pair_parts, dim=0).to(torch.int64),
        "n_nodes": torch.as_tensor(n_nodes, dtype=torch.int64),
        "n_bonds": torch.as_tensor(n_bonds, dtype=torch.int64),
        "y": torch.as_tensor(y, dtype=torch.float32),
        "triple_ij": torch.cat(ij, dim=0).to(torch.int32),
        "triple_ik": torch.cat(ik, dim=0).to(torch.int32),
        "triple_jk": torch.cat(jk, dim=0).to(torch.int32),
        "triple_graph": torch.cat(triple_graph, dim=0).to(torch.int32),
        "triple_ptr": torch.as_tensor(triple_ptr, dtype=torch.int64),
        "provenance": {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "split": str(split),
            "split_size": int(len(data_list)),
            "n_pairs": int(sum(pair_ptr[-1:])),
            "n_triples": int(triple_ptr[-1]),
            "cache_key": _structure_key(split),
            "module_sha256": module_sha256(),
            "parent_canonical_state_sha256": bat.PARENT_SOUP_SHA256,
            "mask_signature": MASK.signature(),
            "order_policy": "official_split_order",
            "seconds": float(time.perf_counter() - started),
        },
    }
    return payload


def build_or_load_structure(split: str, force: bool = False) -> SplitStructure:
    path = structure_path(split)
    if path.exists() and not force:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        provenance = dict(blob["provenance"])
        if provenance.get("cache_key") == _structure_key(split):
            return SplitStructure(
                **{
                    key: blob[key]
                    for key in (
                        "pair_ptr",
                        "pair_bucket",
                        "n_nodes",
                        "n_bonds",
                        "y",
                        "triple_ij",
                        "triple_ik",
                        "triple_jk",
                        "triple_graph",
                        "triple_ptr",
                    )
                },
                provenance=provenance,
            )
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    blob = build_structure(split)
    torch.save(blob, path)
    return SplitStructure(
        **{
            key: blob[key]
            for key in (
                "pair_ptr",
                "pair_bucket",
                "n_nodes",
                "n_bonds",
                "y",
                "triple_ij",
                "triple_ik",
                "triple_jk",
                "triple_graph",
                "triple_ptr",
            )
        },
        provenance=blob["provenance"],
    )


def batch_triples(
    structure: SplitStructure, indices: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map split-global triple rows of the selected graphs into batch pair rows."""
    order = [int(v) for v in indices]
    if not order:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty.clone(), empty.clone(), empty.clone()
    pair_counts = structure.pair_ptr[1:] - structure.pair_ptr[:-1]
    sel_counts = pair_counts[torch.as_tensor(order, dtype=torch.int64)]
    pair_base = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(sel_counts, dim=0)[:-1]]
    )
    ij: list[torch.Tensor] = []
    ik: list[torch.Tensor] = []
    jk: list[torch.Tensor] = []
    triple_graph: list[torch.Tensor] = []
    for local, graph_index in enumerate(order):
        lo = int(structure.triple_ptr[graph_index])
        hi = int(structure.triple_ptr[graph_index + 1])
        offset = int(pair_base[local]) - int(structure.pair_ptr[graph_index])
        ij.append(structure.triple_ij[lo:hi].long() + offset)
        ik.append(structure.triple_ik[lo:hi].long() + offset)
        jk.append(structure.triple_jk[lo:hi].long() + offset)
        triple_graph.append(torch.full((hi - lo,), local, dtype=torch.int64))
    return (
        torch.cat(ij, dim=0),
        torch.cat(ik, dim=0),
        torch.cat(jk, dim=0),
        torch.cat(triple_graph, dim=0),
    )


def make_index_loader(
    data_list: Sequence[Any],
    structure: SplitStructure,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    def _collate(indices: Any) -> dict[str, Any]:
        order = [int(v) for v in indices]
        data = p1.env_collate([data_list[index] for index in order])
        triple_ij, triple_ik, triple_jk, triple_graph = batch_triples(structure, order)
        return {
            "data": data,
            "indices": torch.as_tensor(order, dtype=torch.int64),
            "triple_ij": triple_ij,
            "triple_ik": triple_ik,
            "triple_jk": triple_jk,
            "triple_graph": triple_graph,
        }

    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return torch.utils.data.DataLoader(
        list(range(len(data_list))),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=_collate,
    )


# ---------------------------------------------------------------------------
# evaluation and training
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_online(model: JointTuneModel, loader: Any, device: Any) -> dict[str, Any]:
    cpu_guard(device)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        data = batch["data"].to(device)
        prediction = model(
            data,
            batch["triple_ij"],
            batch["triple_ik"],
            batch["triple_jk"],
            batch["triple_graph"],
            int(data.y.numel()),
        )
        predictions.append(prediction.view(-1).cpu().numpy())
        targets.append(data.y.view(-1).cpu().numpy())
    target = np.concatenate(targets).astype(np.float64)
    pred = np.concatenate(predictions).astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(target - pred))),
        "n_molecules": int(target.shape[0]),
        "predictions": pred,
        "targets": target,
    }


def fixed_coordinate_hash(model: JointTuneModel) -> dict[str, str]:
    """Hash the frozen dictionary / common basis / common scaling buffers."""
    digest = hashlib.sha256()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name in ("backbone.D", "backbone.U", "backbone.common_rms"):
        tensor = buffers.get(name)
        if tensor is None:
            tensor = params[name]
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return {"fixed_coordinate_sha256": digest.hexdigest()}


def frozen_subset_hash(model: JointTuneModel) -> dict[str, Any]:
    frozen = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    buffers = {name: value for name, value in model.named_buffers()}
    return {
        "frozen_param_tensors": int(len(frozen)),
        "frozen_param_numel": int(sum(p.numel() for p in frozen.values())),
        "frozen_params_sha256": audit.state_sha256(frozen),
        "buffers_sha256": audit.state_sha256(buffers),
        "fixed_coordinate_sha256": fixed_coordinate_hash(model)["fixed_coordinate_sha256"],
    }


def train_joint_tune(
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    train_structure: SplitStructure,
    valid_structure: SplitStructure,
    *,
    seed: int = 0,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    threads: int = THREADS,
    checkpoint_dir: Path | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Fixed 40-epoch joint fine-tune; Top-5 soup over epochs 21-40."""
    if int(epochs) != EPOCHS:
        raise RuntimeError(f"this round is fixed at {EPOCHS} epochs, got {epochs}")
    torch.set_num_threads(int(threads))
    device = torch.device("cpu")
    cpu_guard(device)
    out_dir = Path(CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_joint_model(seed=int(seed))
    accounting = configure_trainable(model)
    if not accounting["passed"]:
        raise RuntimeError(f"parameter accounting failed: {accounting['passed']}")
    trainable = model.trainable_parameters()
    trainable_ids = [id(parameter) for parameter in trainable]
    if len(set(trainable_ids)) != len(trainable_ids):
        raise RuntimeError("duplicate parameters in the optimizer list")
    optimizer = torch.optim.Adam(
        trainable, lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY)
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if optimizer_ids != set(trainable_ids):
        raise RuntimeError("optimizer parameter set != trainable whitelist")

    hot_start_path = out_dir / f"{TAG}_epoch000_hotstart_full_state.pt"
    torch.save(model.state_dict(), hot_start_path)
    hashes_before = frozen_subset_hash(model)
    source_hashes_before = {
        "sem108_checkpoint_sha256": _sha256_file(SEM108_SOUP_PATH),
        "bat_checkpoint_sha256": _sha256_file(BAT_SOUP_PATH),
    }

    train_loader = make_index_loader(
        train_data, train_structure, batch_size, True, int(seed) + TRAIN_SHUFFLE_OFFSET
    )
    eval_loader = make_index_loader(
        valid_data, valid_structure, batch_size, False, int(seed) + EVAL_SHUFFLE_OFFSET
    )
    start_valid = evaluate_online(model, eval_loader, device)

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
        n_molecules = 0
        for batch in train_loader:
            data = batch["data"].to(device)
            prediction = model(
                data,
                batch["triple_ij"],
                batch["triple_ik"],
                batch["triple_jk"],
                batch["triple_graph"],
                int(data.y.numel()),
            )
            target = data.y.view(-1)
            loss = F.l1_loss(prediction.view(-1), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(GRAD_CLIP))
            optimizer.step()
            abs_sum += float((prediction.view(-1) - target).abs().sum())
            n_molecules += int(target.numel())
        train_mae = float(abs_sum / max(n_molecules, 1))
        valid = evaluate_online(model, eval_loader, device)
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
        if int(epoch) >= SOUP_EPOCH_LO:
            top_states[int(epoch)] = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            ranked = sorted(
                top_states,
                key=lambda e: (float(next(r for r in curve if r["epoch"] == e)["valid_mae"]), e),
            )
            for stale in ranked[SOUP_K:]:
                del top_states[stale]
        if log and (epoch == 1 or epoch % 5 == 0 or epoch == int(epochs)):
            remaining = (
                f" est_remaining={epoch_seconds * (int(epochs) - epoch):.0f}s"
                if epoch == 1
                else ""
            )
            print(
                f"[{TAG}] epoch={epoch:03d} train={train_mae:.6f} valid={float(valid['mae']):.6f} "
                f"best={best_mae:.6f}@{best_epoch} ({epoch_seconds:.1f}s){remaining}",
                flush=True,
            )
    wall = float(time.perf_counter() - started)
    if len(top_states) < SOUP_K:
        raise RuntimeError(f"only {len(top_states)} soup-eligible epoch states")

    members = sorted(
        top_states,
        key=lambda e: (float(next(r for r in curve if r["epoch"] == e)["valid_mae"]), e),
    )
    member_valid = [float(next(r for r in curve if r["epoch"] == e)["valid_mae"]) for e in members]
    trainable_names = [name for name, _ in model.named_parameters() if _.requires_grad]
    soup_params = {
        name: torch.stack([top_states[e][name].float() for e in members], dim=0).mean(dim=0)
        for name in trainable_names
    }
    frozen_names = [name for name, _ in model.named_parameters() if not _.requires_grad]
    frozen_identical: dict[str, bool] = {
        name: all(torch.equal(top_states[e][name], top_states[members[0]][name]) for e in members)
        for name in frozen_names
    }
    buffer_identical: dict[str, bool] = {
        name: all(torch.equal(top_states[e][name], top_states[members[0]][name]) for e in members)
        for name in model.state_dict()
        if name not in trainable_names and name not in frozen_names
    }
    if not all(frozen_identical.values()) or not all(buffer_identical.values()):
        raise RuntimeError("frozen parameters / buffers differ across soup members")

    hot_start_state = torch.load(hot_start_path, map_location="cpu", weights_only=False)
    soup_full = {key: value.detach().cpu().clone() for key, value in hot_start_state.items()}
    for name, value in soup_params.items():
        soup_full[name] = value
    soup_model = build_joint_model(seed=int(seed))
    configure_trainable(soup_model)
    result = soup_model.load_state_dict(soup_full)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"soup assembly mismatch: {result}")
    soup_valid = evaluate_online(soup_model, eval_loader, device)

    for epoch in members:
        torch.save(top_states[epoch], out_dir / f"{TAG}_epoch{epoch:03d}_full_state.pt")
    torch.save(
        {name: soup_params[name] for name in trainable_names},
        out_dir / f"{TAG}_soup_trainable_state.pt",
    )
    torch.save(soup_model.state_dict(), out_dir / f"{TAG}_soup_full_state.pt")

    hashes_after = frozen_subset_hash(soup_model)
    source_hashes_after = {
        "sem108_checkpoint_sha256": _sha256_file(SEM108_SOUP_PATH),
        "bat_checkpoint_sha256": _sha256_file(BAT_SOUP_PATH),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(TAG),
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
        "eval_shuffle_seed": int(seed) + EVAL_SHUFFLE_OFFSET,
        "optimizer": "Adam",
        "optimizer_numel": int(sum(parameter.numel() for parameter in trainable)),
        "optimizer_tensors": int(len(trainable)),
        "mask": MASK.as_dict(),
        "parameter_accounting": accounting,
        "curve": curve,
        "start_valid_mae": float(start_valid["mae"]),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "soup": {
            "members": [int(e) for e in members],
            "member_valid_mae": member_valid,
            "soup_valid_mae": float(soup_valid["mae"]),
            "soup_epoch_window": [int(SOUP_EPOCH_LO), int(SOUP_EPOCH_HI)],
            "soup_k": int(SOUP_K),
            "averaged_trainable_tensors": int(len(trainable_names)),
            "averaged_trainable_params": int(sum(p.numel() for p in soup_params.values())),
            "frozen_members_identical": bool(all(frozen_identical.values())),
            "buffers_members_identical": bool(all(buffer_identical.values())),
        },
        "last_10_valid_mean": float(np.mean([row["valid_mae"] for row in curve[-10:]])),
        "final_train_mae": float(curve[-1]["train_mae"]),
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve), 1)),
        "first_epoch_seconds": float(curve[0]["epoch_seconds"]),
        "estimated_remaining_s_after_epoch1": float(curve[0]["epoch_seconds"] * (int(epochs) - 1)),
        "rss_start_mb": float(rss_start),
        "rss_end_mb": float(audit._rss_mb()),
        "peak_rss_mb": float(audit._peak_rss_mb()),
        "frozen_subset_before": hashes_before,
        "frozen_subset_after": hashes_after,
        "frozen_subset_unchanged": bool(hashes_before == hashes_after),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "source_files_unchanged": bool(source_hashes_before == source_hashes_after),
        "hot_start_checkpoint": str(hot_start_path.relative_to(REPO_ROOT)),
        "hot_start_excluded_from_soup": True,
        "official_test_loaded": False,
    }
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# correctness helpers
# ---------------------------------------------------------------------------


def structure_audit(structure: SplitStructure, brute_force_graphs: int = 32) -> dict[str, Any]:
    counts = structure.triple_ptr[1:] - structure.triple_ptr[:-1]
    expected = torch.where(
        structure.n_nodes >= 3,
        structure.n_bonds * (structure.n_nodes - 2).clamp_min(0),
        torch.zeros_like(structure.n_bonds),
    )
    count_ok = bool(torch.equal(counts, expected))
    pair_lo = structure.pair_ptr[:-1][structure.triple_graph.long()]
    pair_hi = structure.pair_ptr[1:][structure.triple_graph.long()]
    rows = torch.stack(
        [structure.triple_ij, structure.triple_ik, structure.triple_jk], dim=1
    ).long()
    in_range = bool(((rows >= pair_lo.unsqueeze(1)) & (rows < pair_hi.unsqueeze(1))).all())
    anchor_ok = True
    brute_force_ok = True
    for graph_index in range(min(int(brute_force_graphs), structure.n_graphs)):
        n = int(structure.n_nodes[graph_index])
        lo = int(structure.pair_ptr[graph_index])
        hi = int(structure.pair_ptr[graph_index + 1])
        bucket = structure.pair_bucket[lo:hi]
        anchors = torch.nonzero(bucket == 0, as_tuple=False).view(-1)
        if int(anchors.numel()) != int(structure.n_bonds[graph_index]):
            anchor_ok = False
        pair_index = torch.tensor(
            [(a, b) for a in range(n) for b in range(a + 1, n)], dtype=torch.int64
        ).t()
        reference = bat.graph_triple_rows(pair_index, bucket, n)
        got_lo = int(structure.triple_ptr[graph_index])
        got_hi = int(structure.triple_ptr[graph_index + 1])
        got = torch.stack(
            [
                structure.triple_ij[got_lo:got_hi].long() - lo,
                structure.triple_ik[got_lo:got_hi].long() - lo,
                structure.triple_jk[got_lo:got_hi].long() - lo,
            ],
            dim=1,
        )
        expected_rows = torch.stack(
            [reference["ij"], reference["ik"], reference["jk"]], dim=1
        )
        if not torch.equal(got, expected_rows):
            brute_force_ok = False
    zero_graphs = int((counts == 0).sum())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "split": structure.provenance.get("split"),
        "n_graphs": int(structure.n_graphs),
        "n_pairs": int(structure.n_pairs),
        "n_triples": int(structure.n_triples),
        "triple_count_rule_ok": count_ok,
        "no_out_of_range_rows": in_range,
        "anchor_counts_ok": bool(anchor_ok),
        "brute_force_enumeration_ok": bool(brute_force_ok),
        "zero_tuple_graphs": zero_graphs,
        "passed": bool(count_ok and in_range and anchor_ok and brute_force_ok),
    }
    official_test_blocker(payload)
    return payload


def batch_mapping_check(
    structure: SplitStructure, data_list: Sequence[Any], batch_size: int = 32
) -> dict[str, Any]:
    """A real batch: cached pair buckets match, every triple row stays in-graph."""
    order = list(range(min(int(batch_size), structure.n_graphs)))
    data = p1.env_collate([data_list[i] for i in order])
    triple_ij, triple_ik, triple_jk, triple_graph = batch_triples(structure, order)
    pair_counts = structure.pair_ptr[1:] - structure.pair_ptr[:-1]
    sel = pair_counts[torch.as_tensor(order, dtype=torch.int64)]
    pair_base = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(sel, 0)[:-1]])
    bucket_ok = True
    for local, graph_index in enumerate(order):
        lo = int(structure.pair_ptr[graph_index])
        hi = int(structure.pair_ptr[graph_index + 1])
        block = data.pair_bucket[pair_base[local] : pair_base[local] + (hi - lo)]
        if not torch.equal(block.to(torch.int64), structure.pair_bucket[lo:hi]):
            bucket_ok = False
    pair_lo = pair_base[triple_graph]
    pair_hi = pair_base + torch.as_tensor(
        [int(structure.pair_ptr[g + 1] - structure.pair_ptr[g]) for g in order], dtype=torch.int64
    )
    pair_hi = pair_hi[triple_graph]
    rows = torch.stack([triple_ij, triple_ik, triple_jk], dim=1)
    in_range = bool(((rows >= pair_lo.unsqueeze(1)) & (rows < pair_hi.unsqueeze(1))).all())
    expected_triples = int(
        sum(
            int(structure.n_bonds[g])
            * max(int(structure.n_nodes[g]) - 2, 0)
            for g in order
        )
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "n_graphs": len(order),
        "n_triples": int(triple_ij.numel()),
        "n_triples_expected": expected_triples,
        "pair_buckets_match_cache": bool(bucket_ok),
        "triple_rows_in_graph_pair_range": in_range,
        "no_cross_graph_rows": bool(in_range),
        "passed": bool(
            bucket_ok and in_range and int(triple_ij.numel()) == expected_triples
        ),
    }
    official_test_blocker(payload)
    return payload


def online_vs_cached_check(
    model: JointTuneModel, valid_structure: SplitStructure, n_graphs: int = 32
) -> dict[str, Any]:
    """Online autograd path vs the BAT-v1 cached path on the same small batch."""
    cached = bat.build_or_load_cache("valid")
    cached_model = bat.BondAnchoredTripleModel(seed=0)
    state = torch.load(BAT_SOUP_PATH, map_location="cpu", weights_only=False)
    cached_model.load_state_dict({key: value for key, value in state.items()})
    cached_model.load_standardizers(_read_json(BAT_STANDARDIZERS_JSON)["statistics"])
    cached_model.eval()
    model.eval()

    indices = list(range(min(int(n_graphs), cached.n_graphs)))
    batch = bat.collate_graphs(cached, indices)
    with torch.no_grad():
        p_cached = cached_model(
            batch["z_old"],
            batch["pair_ij"],
            batch["pair_ik"],
            batch["pair_jk"],
            batch["triple_graph"],
            len(indices),
        )
    valid_data = p1run.load_split("valid")
    data = p1.env_collate([valid_data[i] for i in indices])
    triple_ij, triple_ik, triple_jk, triple_graph = batch_triples(valid_structure, indices)
    with torch.no_grad():
        # cached reference intermediates from the BAT-v1 module pieces
        p_ij, p_ik, p_jk = cached_model.standardize_pair_rows(
            batch["pair_ij"], batch["pair_ik"], batch["pair_jk"]
        )
        t_cached = cached_model.triple_embedding(p_ij, p_ik, p_jk)
        z3_cached = cached_model.pool_triples(t_cached, batch["triple_graph"], len(indices))
        prediction, aux = model(data, triple_ij, triple_ik, triple_jk, triple_graph, len(indices), return_aux=True)
    max_pair = float((aux["pair_value"][triple_ij.long()] - batch["pair_ij"]).abs().max())
    max_z_old = float((aux["z_old"] - batch["z_old"]).abs().max())
    max_z3 = float((aux["z3"] - z3_cached).abs().max())
    max_prediction = float((prediction - p_cached).abs().max())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "n_graphs": len(indices),
        "max_pair_value_diff": max_pair,
        "max_z_old_diff": max_z_old,
        "max_z3_diff": max_z3,
        "max_prediction_diff": max_prediction,
        "prediction_tolerance": ONLINE_CACHE_TOLERANCE,
        "passed": bool(max_prediction <= ONLINE_CACHE_TOLERANCE),
    }
    official_test_blocker(payload)
    return payload


def mode_forward_check(model: JointTuneModel, train_structure: SplitStructure, train_data: Sequence[Any]) -> dict[str, Any]:
    order = [int(v) for v in torch.randperm(8, generator=torch.Generator().manual_seed(0)).tolist()]
    data = p1.env_collate([train_data[i] for i in order])
    triple_ij, triple_ik, triple_jk, triple_graph = batch_triples(train_structure, order)
    model.eval()
    with torch.no_grad():
        eval_prediction = model(data, triple_ij, triple_ik, triple_jk, triple_graph, len(order))
    model.train()
    backbone_training = [name for name, module in model.backbone.named_modules() if module.training]
    dropout_training = [
        name for name, module in model.named_modules() if isinstance(module, nn.Dropout) and module.training
    ]
    with torch.no_grad():
        train_prediction = model(data, triple_ij, triple_ik, triple_jk, triple_graph, len(order))
    max_diff = float((train_prediction - eval_prediction).abs().max())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "backbone_modules_in_train_mode": backbone_training,
        "dropout_modules_in_train_mode": dropout_training,
        "max_abs_diff_train_vs_eval": max_diff,
        "passed": bool(not backbone_training and not dropout_training and max_diff == 0.0),
    }
    official_test_blocker(payload)
    return payload


GRADIENT_GROUPS: dict[str, tuple[str, ...]] = {
    "edge_binding": ("backbone.W_E_S", "backbone.W_E_C"),
    "edge_encoder": ("backbone.edge_encoder.",),
    "fusion": ("backbone.fusion.",),
    "pair_projection": ("backbone.pair_projection.",),
    "relation_encoder": ("backbone.relation_encoder.",),
    "distance_gate": ("backbone.distance_gate.",),
    "pair_encoder": ("backbone.pair_encoder.",),
    "F": ("F.",),
    "reader": ("reader.",),
}


def one_batch_gradient_check(model: JointTuneModel, train_structure: SplitStructure, train_data: Sequence[Any]) -> dict[str, Any]:
    """Backward on one normal training batch without an optimizer step."""
    order = list(range(128))
    data = p1.env_collate([train_data[i] for i in order])
    triple_ij, triple_ik, triple_jk, triple_graph = batch_triples(train_structure, order)
    model.train()
    prediction = model(data, triple_ij, triple_ik, triple_jk, triple_graph, len(order))
    target = data.y.view(-1)
    loss = F.l1_loss(prediction.view(-1), target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    group_norms: dict[str, float] = {}
    for group, prefixes in GRADIENT_GROUPS.items():
        total = 0.0
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                if parameter.grad is None:
                    raise RuntimeError(f"trainable parameter without gradient: {name}")
                total += float(parameter.grad.detach().pow(2).sum())
        group_norms[group] = float(np.sqrt(total))
    frozen_grads_none = all(
        parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad
    )
    all_finite = bool(
        torch.isfinite(loss).all()
        and torch.isfinite(prediction).all()
        and all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters() if parameter.requires_grad)
    )
    trainable = model.trainable_parameters()
    optimizer = torch.optim.Adam(trainable, lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY))
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable_ids = {id(p) for p in trainable}
    model.zero_grad(set_to_none=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "n_graphs": len(order),
        "n_triples": int(triple_ij.numel()),
        "loss": float(loss.detach()),
        "prediction_rms": float(prediction.detach().pow(2).mean().sqrt()),
        "group_grad_norms": group_norms,
        "min_group_grad_norm": float(min(group_norms.values())),
        "all_finite": all_finite,
        "frozen_grads_none": bool(frozen_grads_none),
        "optimizer_numel": int(sum(p.numel() for p in trainable)),
        "optimizer_tensors": int(len(trainable)),
        "optimizer_set_equals_whitelist": bool(optimizer_ids == trainable_ids),
        "grads_cleared": all(parameter.grad is None for parameter in model.parameters()),
        "passed": bool(
            all_finite
            and all(value > 0.0 for value in group_norms.values())
            and frozen_grads_none
            and optimizer_ids == trainable_ids
            and int(sum(p.numel() for p in trainable)) == EXPECTED_TRAINABLE_PARAMS
        ),
    }
    official_test_blocker(payload)
    return payload


def hot_start_check(model: JointTuneModel, valid_data: Sequence[Any], valid_structure: SplitStructure) -> dict[str, Any]:
    loader = make_index_loader(valid_data, valid_structure, BATCH_SIZE, False, EVAL_SHUFFLE_OFFSET)
    result = evaluate_online(model, loader, torch.device("cpu"))
    mae = float(result["mae"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "n_valid": int(result["n_molecules"]),
        "M_start_online": mae,
        "M_start_expected": float(M_START_EXPECTED),
        "abs_diff": float(abs(mae - M_START_EXPECTED)),
        "tolerance": float(HOT_START_TOLERANCE),
        "passed": bool(abs(mae - M_START_EXPECTED) <= HOT_START_TOLERANCE),
    }
    official_test_blocker(payload)
    return payload


def run_correctness_suite(
    *,
    train_structure: SplitStructure,
    valid_structure: SplitStructure,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
) -> dict[str, Any]:
    model = build_joint_model(seed=0)
    checks: dict[str, Any] = {
        "parameter_accounting": _read_json(RESULTS_DIR / "references.json")["parameter_accounting"],
        "hot_start": hot_start_check(model, valid_data, valid_structure),
        "online_vs_cached": online_vs_cached_check(model, valid_structure),
        "mode_forward": mode_forward_check(model, train_structure, train_data),
        "structure_audit_train": structure_audit(train_structure),
        "structure_audit_valid": structure_audit(valid_structure),
        "batch_mapping": batch_mapping_check(valid_structure, valid_data),
    }
    # The gradient probe must not perturb the hot start used by training.
    probe = build_joint_model(seed=0)
    configure_trainable(probe)
    checks["one_batch_gradient"] = one_batch_gradient_check(probe, train_structure, train_data)
    del probe
    passed = all(
        bool(value.get("passed", value.get("all_passed", False))) for value in checks.values()
    ) and bool(checks["parameter_accounting"]["passed"])
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
    "STRUCTURE_DIR",
    "CHECKPOINT_DIR",
    "PREREG_PATH",
    "BAT_SOUP_PATH",
    "SEM108_SOUP_PATH",
    "M_START_EXPECTED",
    "M_PARENT_EXPECTED",
    "TAG",
    "TRAINABLE_PREFIXES",
    "EPOCHS",
    "SOUP_EPOCH_LO",
    "SOUP_EPOCH_HI",
    "SOUP_K",
    "PROMISING_THRESHOLD",
    "VERDICT_PROMISING",
    "VERDICT_NO_SIGNAL",
    "VERDICT_INVALID",
    "SCOPE_NOTE",
    "MASK",
    "SplitStructure",
    "JointTuneModel",
    "build_joint_model",
    "configure_trainable",
    "parameter_breakdown",
    "parent_provenance",
    "m1_soup_provenance",
    "build_structure",
    "build_or_load_structure",
    "batch_triples",
    "make_index_loader",
    "evaluate_online",
    "frozen_subset_hash",
    "fixed_coordinate_hash",
    "train_joint_tune",
    "structure_audit",
    "batch_mapping_check",
    "online_vs_cached_check",
    "mode_forward_check",
    "one_batch_gradient_check",
    "hot_start_check",
    "run_correctness_suite",
    "module_sha256",
    "official_test_blocker",
]
