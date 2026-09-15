"""ZINC Cell-A identity / capacity control: learned lookup vs shared capacity.

Question
--------
The frozen deterministic-GPU ZINC winner is cell A (the T=2 recurrent
pair--centre ``H64`` configuration, 85,763 trainable params, 2-seed fixed
Top-5 soup valid MAE ``0.126368``).  Its per-patch initial representation is

    patch_cont  +  learned typed lookup (16D, hybrid full/low-rank)
                +  learned parent lookup (8D)
        -> patch_encoder -> h0 -> ...

This experiment asks a single question:

    Does the current performance actually depend on *per-identity learned
    embedding memory*, or only on identity distinguishability / generic shared
    capacity?

Naively deleting the typed lookup and seeing the metric drop is **not**
evidence, because deletion also removes a large block of trainable parameters.
So three parameter-matched conditions are compared:

* **A0** -- reference.  Unchanged frozen cell A: ``typed_embedding`` is the
  historical learned hybrid lookup (36,420 trainable params).  A0 is *not*
  retrained; its frozen results are reused.
* **A1** -- fixed identity code + shared capacity.  The learned lookup is
  replaced by a deterministic, seed-independent, non-trainable 16D code per
  typed token (OOV id 0 included) followed by a *token-shared* trainable
  adapter that also reads ``patch_cont``.  The adapter is sized so the total
  trainable parameter count matches A0.
* **A2** -- no typed identity + the same shared capacity.  The adapter is
  byte-for-byte the A1 adapter; only the identity slot content differs
  (exact zeros instead of the fixed code).  Total params match A0.

A1 and A2 therefore differ in exactly one thing: whether the shared adapter
sees the deterministic identity code.

Frozen / forbidden
------------------
``h=64``, ``q=16``, ``T=2`` weight-tied refresh, relation descriptor, pair
projection / encoder, distance gate, centre aggregation / update, recurrent
refresh, global branch, readout semantics, small raw head, optimizer
(Adam lr 1e-3, wd 1e-5, batch 128, max_epochs 240, patience 40, no scheduler),
fixed Top-5 soup rule and the deterministic A100 execution regime are all
inherited unchanged.  The parent lookup is untouched.  The official ZINC test
split is **never** loaded by this module.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_identity_capacity_control <stage>

Stages: ``params sanity train train_queue repro soup diagnostics decide report``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity_decomposition as cd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_hwidth as hw,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_identity_capacity_control"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
SOUP_DIR = RESULTS_DIR / "soup_states"
BASELINE_DIR = RESULTS_DIR / "baseline_guard"

A0_RESULTS_DIR = (
    TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_capacity_decomposition"
)

PROTOCOL_VERSION = "compact_v4_identity_capacity_control_v1"

# Frozen cell A definition (identical to the capacity-decomposition cell A).
Q_DIM = 16
CELL_A_H = 64
CELL_A_PATCH_ENCODER_HIDDEN = 64
CELL_A_GLOBAL_ENCODER_HIDDEN = 32
CELL_A_PARAMS = 85763
CELL_A_HEAD_PARAMS = 4551
CELL_A_UNIFIED_WIDTH = 334
TYPED_LOOKUP_PARAMS = 36420
TYPED_VOCABULARY_SIZE = 6785
PARENT_VOCABULARY_SIZE = 32
IDENTITY_DIM = 16
UNIFIED_TOLERANCE = 0.01  # +/- 1% total-parameter match to A0

# Adapter = Linear(identity_dim + shell_width, hidden) -> LayerNorm -> ReLU
#           -> Dropout -> Linear(hidden, identity_dim).
# Parameters = (1 + identity_dim + shell_width) * hidden + 2*hidden + identity_dim + 16.
# With identity_dim=16 and shell_width=146 (frozen SHELL_WIDTH) the exact
# formula is 181*hidden + 16.  hidden=201 gives 36,397 adapter params, i.e.
# 85,740 total (A0 - 23 params, 0.027%).
ADAPTER_HIDDEN = 201

# Deterministic fixed-code / adapter initialisation seeds (no search).
IDENTITY_CODE_SEED = 20260915
ADAPTER_INIT_SEED = 20260916

CONDITIONS = ("A1", "A2")
SEEDS = (0, 1)

# Reference soup numbers reused verbatim (never retrained).
A0_REFERENCE_SOUP_PER_SEED = {0: 0.12470379155874252, 1: 0.12803224420547485}
A0_REFERENCE_SOUP_2SEED_MEAN = 0.1263680279762484
A0_REFERENCE_RAW_PER_SEED = {0: 0.12970963285310427, 1: 0.13141501784324646}
A0_REFERENCE_RAW_2SEED_MEAN = 0.1305623254594393

MEANINGFUL_THRESHOLD = 0.002

DETERMINISTIC = False


def _set_deterministic(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# deterministic fixed identity codes (no RNG, seed-independent, target-free)
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    """Stable 64-bit integer mixer (public-domain splitmix64 finalizer)."""
    x = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def fixed_identity_codes(
    vocabulary_size: int,
    dim: int = IDENTITY_DIM,
    base_seed: int = IDENTITY_CODE_SEED,
) -> torch.Tensor:
    """Deterministic unit-norm 16D code per typed-token id.

    The code for token ``t`` is derived only from ``base_seed`` and ``t`` via
    the stable ``splitmix64`` integer hash: no RNG, no target, no training.  ID
    0 (the reserved OOV token) gets its own fixed row, so every OOV occurrence
    maps to exactly the same code.  Codes are normalised to unit L2 norm so the
    shared adapter sees a scale-controlled input.
    """
    vocabulary_size = int(vocabulary_size)
    dim = int(dim)
    rows = np.zeros((vocabulary_size, dim), dtype=np.float64)
    for token in range(vocabulary_size):
        values = np.empty(dim, dtype=np.float64)
        for j in range(dim):
            word = _splitmix64(((int(base_seed) + token) * dim) + j)
            values[j] = (word >> 11) / float(1 << 53)  # uniform [0, 1)
        values = values - 0.5
        norm = float(np.linalg.norm(values))
        if norm < 1.0e-12:
            values = np.zeros(dim, dtype=np.float64)
            values[0] = 1.0
            norm = 1.0
        rows[token] = values / norm
    return torch.from_numpy(rows.astype(np.float32))


def _code_statistics(codes: torch.Tensor, sample: int = 512) -> dict[str, Any]:
    array = codes.double().cpu().numpy()
    n = int(array.shape[0])
    norms = np.linalg.norm(array, axis=1)
    take = min(int(sample), n)
    index = np.linspace(0, n - 1, take).astype(int)
    sub = array[index]
    sub = sub / np.linalg.norm(sub, axis=1, keepdims=True)
    gram = sub @ sub.T
    off_diagonal = gram[~np.eye(take, dtype=bool)]
    return {
        "vocabulary_size": n,
        "dim": int(array.shape[1]),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "oov_code_norm": float(norms[0]),
        "sampled_pairs": int(take),
        "abs_cosine_min": float(np.abs(off_diagonal).min()),
        "abs_cosine_mean": float(np.abs(off_diagonal).mean()),
        "abs_cosine_median": float(np.median(np.abs(off_diagonal))),
        "abs_cosine_max": float(np.abs(off_diagonal).max()),
        "exact_duplicate_rows": int(
            n - len(np.unique(np.round(array, 6), axis=0))
        ),
    }


# ---------------------------------------------------------------------------
# token channel: fixed code (+ frozen Code adapter shared over all tokens)
# ---------------------------------------------------------------------------


class IdentityTokenChannel(nn.Module):
    """Drop-in replacement for the learned typed lookup.

    ``forward(token)`` returns ``adapter([slot, patch_cont])`` where:

    * ``slot`` is the deterministic fixed code for ``token`` (``mode="fixed"``)
      or the exact zero vector (``mode="none"``, condition A2);
    * ``patch_cont`` is the frozen continuous shell descriptor of the current
      batch (set by ``set_context`` just before the forward pass).

    The adapter has one shared set of weights for all tokens.  It is the only
    trainable module here; the code table is a non-persistent buffer.
    """

    def __init__(
        self,
        code: torch.Tensor,
        context_width: int,
        hidden: int,
        dropout: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"fixed", "none"}:
            raise ValueError(f"unknown identity mode {mode!r}; expected fixed|none")
        self.mode = str(mode)
        self.dim = int(code.shape[1])
        self.context_width = int(context_width)
        self.hidden = int(hidden)
        self.register_buffer("code", code, persistent=False)
        self.adapter = nn.Sequential(
            nn.Linear(self.dim + self.context_width, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden, self.dim),
        )
        self._context: torch.Tensor | None = None

    def set_context(self, patch_cont: torch.Tensor) -> None:
        if int(patch_cont.shape[1]) != self.context_width:
            raise RuntimeError(
                f"patch_cont width {int(patch_cont.shape[1])} != adapter context "
                f"width {self.context_width}"
            )
        self._context = patch_cont

    def identity_slot(self, token: torch.Tensor) -> torch.Tensor:
        slot = self.code[token.long()]
        if self.mode == "none":
            slot = torch.zeros_like(slot)
        return slot

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        slot = self.identity_slot(token)
        if self._context is None:
            context = torch.zeros(
                (slot.shape[0], self.context_width),
                dtype=slot.dtype,
                device=slot.device,
            )
        else:
            context = self._context
        return self.adapter(torch.cat([slot, context], dim=1))


class IdentityCapacityModel(rec.PatchPathRecurrentPairCentreModel):
    """Frozen cell-A recurrent pair--centre with the typed lookup replaced.

    All inherited modules (parent lookup, patch/pair/centre encoders, readout,
    small head) are the untouched frozen cell-A modules.  Only ``typed_embedding``
    is swapped for an :class:`IdentityTokenChannel`.
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        identity_mode: str = "fixed",
        adapter_hidden: int = ADAPTER_HIDDEN,
        identity_code_seed: int = IDENTITY_CODE_SEED,
        **kwargs: Any,
    ) -> None:
        super().__init__(typed_vocabulary_size, parent_vocabulary_size, **kwargs)
        if bool(getattr(self, "direct_token_readout", False)):
            raise ValueError(
                "direct_token_readout is incompatible with the identity control"
            )
        original = self.typed_embedding
        vocabulary_size = int(original.vocabulary_size)
        code = fixed_identity_codes(
            vocabulary_size, IDENTITY_DIM, int(identity_code_seed)
        )
        self.identity_mode = str(identity_mode)
        self.adapter_hidden = int(adapter_hidden)
        self.identity_context_width = int(self.shell_width)
        self.typed_embedding = IdentityTokenChannel(
            code=code,
            context_width=int(self.shell_width),
            hidden=int(adapter_hidden),
            dropout=float(kwargs.get("dropout", 0.05)),
            mode=self.identity_mode,
        )

    @property
    def identity_code(self) -> torch.Tensor:
        return self.typed_embedding.code

    @property
    def identity_adapter(self) -> nn.Module:
        return self.typed_embedding.adapter

    def _encode_core(self, data: Data, use_recurrence: bool) -> torch.Tensor:
        # The inherited ``_encode_core`` calls ``self.typed_embedding(token)``;
        # the shared adapter additionally needs the continuous shell descriptor
        # of the same batch, so publish it immediately before the forward.
        self.typed_embedding.set_context(data.patch_cont)
        return super()._encode_core(data, use_recurrence)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _reinit_adapter(adapter: nn.Module, seed: int) -> None:
    """Deterministic adapter init that does not touch the global RNG.

    Reproduces the default ``nn.Linear`` reset (kaiming-uniform with
    ``a=sqrt(5)`` gives ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``) and the default
    ``nn.LayerNorm`` reset (weight 1, bias 0) from a local generator.
    """
    generator = torch.Generator().manual_seed(int(seed))
    for module in adapter.modules():
        if isinstance(module, nn.Linear):
            fan_in = int(module.weight.shape[1])
            bound = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                module.weight.uniform_(-bound, bound, generator=generator)
                if module.bias is not None:
                    module.bias.uniform_(-bound, bound, generator=generator)
        elif isinstance(module, nn.LayerNorm):
            with torch.no_grad():
                module.weight.fill_(1.0)
                module.bias.zero_()


def build_condition(
    condition: str, seed: int = 0
) -> tuple[IdentityCapacityModel, list[str], dict[str, torch.Tensor]]:
    """Build A1/A2 with the frozen cell-A init for every shared tensor.

    The reference is ``cd.build_cell("A", seed)`` (bit-exact frozen cell A).  A
    fresh identity model is constructed, its adapter is initialised
    deterministically, and then *every* shape-matching tensor is copied from
    the reference.  As a result A1 and A2 share the exact same shared weights
    and only differ in the identity slot content.
    """
    condition = str(condition)
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    reference = cd.build_cell("A", int(seed))
    reference_state = {
        key: value.detach().clone() for key, value in reference.state_dict().items()
    }
    kwargs = shead._base_kwargs()
    kwargs["patch_hidden"] = CELL_A_H
    kwargs["patch_encoder_hidden"] = CELL_A_PATCH_ENCODER_HIDDEN
    kwargs["global_encoder_hidden"] = CELL_A_GLOBAL_ENCODER_HIDDEN
    kwargs.pop("pair_hidden", None)
    shead._seed_everything(int(seed))
    mode = "fixed" if condition == "A1" else "none"
    with cap._construction_guards(CELL_A_H, Q_DIM):
        model = IdentityCapacityModel(
            TYPED_VOCABULARY_SIZE,
            PARENT_VOCABULARY_SIZE,
            pair_hidden=Q_DIM,
            recurrence_rounds=rec.RECURRENCE_ROUNDS,
            recurrence_enabled=True,
            recurrence_mode="refresh",
            identity_mode=mode,
            adapter_hidden=ADAPTER_HIDDEN,
            **kwargs,
        )
    _reinit_adapter(model.identity_adapter, ADAPTER_INIT_SEED + int(seed))
    copied: list[str] = []
    with torch.no_grad():
        state = model.state_dict()
        for key, value in reference_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)
                copied.append(key)
    return model, copied, reference_state


def build_model(condition: str, seed: int = 0) -> IdentityCapacityModel:
    model, _copied, _reference = build_condition(condition, seed)
    return model


def tag_for(condition: str) -> str:
    return f"identity_{condition}"


def expected_total(condition: str) -> int:
    return int(hw._n_params(build_model(condition, 0)))


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


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    a1 = build_model("A1", 0)
    a2 = build_model("A2", 0)
    a0_model = cd.build_cell("A", 0)

    def row(condition: str, model: nn.Module, identity_mode: str) -> dict[str, Any]:
        total = hw._n_params(model)
        head = hw._n_params(model.head)
        adapter = (
            hw._n_params(model.identity_adapter)
            if condition != "A0"
            else 0
        )
        lookup = (
            hw._n_params(model.typed_embedding)
            if condition == "A0"
            else 0
        )
        return {
            "condition": condition,
            "identity_mode": identity_mode,
            "total_params": int(total),
            "head_params": int(head),
            "backbone_params": int(total - head),
            "typed_lookup_trainable_params": int(lookup),
            "shared_adapter_trainable_params": int(adapter),
            "parent_lookup_params": int(hw._n_params(model.parent_embedding)),
            "patch_state_width_h": int(model.patch_hidden),
            "pair_width_q": int(model.pair_hidden),
            "token_width": int(model.token_width),
            "adapter_hidden": int(getattr(model, "adapter_hidden", 0)),
            "identity_dim": int(
                model.identity_code.shape[1]
                if hasattr(model, "identity_code")
                else 0
            ),
            "unified_graph_width": int(model.unified_graph_width),
            "total_matches_A0_within_1pct": bool(
                abs(int(total) - CELL_A_PARAMS) / CELL_A_PARAMS <= UNIFIED_TOLERANCE
            ),
        }

    rows = [
        row("A0", a0_model, "learned_hybrid_lookup"),
        row("A1", a1, "fixed_code_plus_shared_adapter"),
        row("A2", a2, "zero_slot_plus_shared_adapter"),
    ]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_cell_a_params": CELL_A_PARAMS,
        "typed_lookup_released_params": TYPED_LOOKUP_PARAMS,
        "adapter_hidden": int(ADAPTER_HIDDEN),
        "identity_dim": int(IDENTITY_DIM),
        "unified_tolerance": float(UNIFIED_TOLERANCE),
        "rows": rows,
        "a1_a2_total_identical": bool(
            rows[1]["total_params"] == rows[2]["total_params"]
        ),
        "a1_a2_adapter_identical": bool(
            rows[1]["shared_adapter_trainable_params"]
            == rows[2]["shared_adapter_trainable_params"]
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# sanity
# ---------------------------------------------------------------------------


def _synthetic_batch() -> Data:
    """Small synthetic batch mirroring the capacity-decomposition test helper."""
    from tracks.ksvd.code.graph import from_edges, ring_chords
    from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

    topo_width = int(ztopo.raw_width("hinge"))
    datasets: list[Data] = []
    for graph in (
        ring_chords(6, []),
        ring_chords(5, []),
        from_edges(6, [(0, 1), (1, 2), (2, 0)]),
    ):
        n = len(graph.nodes)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs
            else torch.zeros(2, 0, dtype=torch.long)
        )
        buckets = torch.tensor(
            [min(max(i % 5, 0), 4) for i in range(len(pairs))], dtype=torch.long
        )
        datasets.append(
            Data(
                patch_cont=torch.zeros(n, int(zpp.SHELL_WIDTH)),
                patch_context=torch.zeros(n, 0),
                typed_token=torch.arange(n, dtype=torch.long),
                parent_token=torch.zeros(n, dtype=torch.long),
                structural_token=torch.zeros(n, dtype=torch.long),
                structural_coarse=torch.zeros(n, 0),
                pair_index=pair_index,
                pair_relation=torch.zeros(len(pairs), int(zpp.RELATION_WIDTH)),
                pair_bucket=buckets,
                global_context=torch.zeros(1, int(zpp.GLOBAL_WIDTH)),
                topology_features=torch.zeros(1, topo_width),
                y=torch.zeros(1),
                num_nodes=n,
            )
        )
    return next(iter(zpp._make_loader(datasets, 128, False, 0)))


def _shared_tensor_names(reference_state: Mapping[str, torch.Tensor]) -> list[str]:
    return sorted(
        key
        for key in reference_state
        if not key.startswith("typed_embedding.")
    )


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    batch = _synthetic_batch()

    a1, copied1, ref_state = build_condition("A1", 0)
    a2, copied2, _ = build_condition("A2", 0)
    a0 = cd.build_cell("A", 0)

    expected_adapter = None
    expected_total_value = None
    checks: dict[str, bool] = {}

    param_payload = params()
    checks["param_counts_match_A0_within_1pct"] = all(
        row["total_matches_A0_within_1pct"] for row in param_payload["rows"]
    )
    checks["A1_A2_total_identical"] = bool(param_payload["a1_a2_total_identical"])
    checks["A1_A2_adapter_identical"] = bool(
        param_payload["a1_a2_adapter_identical"]
    )
    expected_total_value = int(param_payload["rows"][1]["total_params"])
    expected_adapter = int(param_payload["rows"][1]["shared_adapter_trainable_params"])

    # 1. Every non-typed tensor is byte-identical to the frozen cell A.
    shared = _shared_tensor_names(ref_state)
    a1_state = a1.state_dict()
    a2_state = a2.state_dict()
    shared1 = max(
        float((a1_state[key] - ref_state[key]).abs().max()) for key in shared
    )
    shared2 = max(
        float((a2_state[key] - ref_state[key]).abs().max()) for key in shared
    )
    checks["A1_shared_tensors_bit_identical_to_A0"] = shared1 == 0.0
    checks["A2_shared_tensors_bit_identical_to_A0"] = shared2 == 0.0
    checks["parent_embedding_unchanged"] = (
        float(
            (a1_state["parent_embedding.full.weight"]
             - ref_state["parent_embedding.full.weight"]).abs().max()
        )
        == 0.0
    )

    # 2. A1/A2 adapters are bit-identical to each other.
    adapter_keys = [
        key for key in a1_state if key.startswith("typed_embedding.")
    ]
    adapter_diff = max(
        float((a1_state[key] - a2_state[key]).abs().max()) for key in adapter_keys
    )
    checks["A1_A2_adapter_bit_identical"] = adapter_diff == 0.0

    # 3. Fixed code has no gradient / is not trainable; deterministic.
    code_a = a1.identity_code
    code_b = build_model("A1", 0).identity_code
    checks["fixed_code_not_trainable"] = not bool(code_a.requires_grad)
    checks["fixed_code_seed_independent"] = bool(
        torch.equal(code_a, build_model("A1", 1).identity_code)
    )
    checks["fixed_code_deterministic_across_builds"] = bool(torch.equal(code_a, code_b))
    checks["fixed_code_oov_row_fixed"] = bool(
        torch.equal(code_a[0], build_model("A1", 0).identity_code[0])
    )
    code_stats = _code_statistics(code_a)
    checks["fixed_code_distinguishable"] = bool(
        code_stats["abs_cosine_max"] < 0.999
        and code_stats["exact_duplicate_rows"] == 0
    )
    checks["fixed_code_unit_norm"] = bool(
        abs(code_stats["norm_min"] - 1.0) < 1e-5
        and abs(code_stats["norm_max"] - 1.0) < 1e-5
    )

    # 4. A2 has no typed identity information: a typed-token permutation must
    #    leave the output bit-identical.  A1 must react.
    def _permute(batch_data: Data) -> Data:
        clone = batch_data.clone()
        token = clone.typed_token.clone()
        # deterministic derangement-ish permutation
        clone.typed_token = token.flip(0)
        return clone

    a1.eval()
    a2.eval()
    with torch.no_grad():
        out_a1 = a1(batch)
        out_a2 = a2(batch)
        perm = _permute(batch)
        out_a1_perm = a1(perm)
        out_a2_perm = a2(perm)
    a1_flip = float((out_a1 - out_a1_perm).abs().max())
    a2_flip = float((out_a2 - out_a2_perm).abs().max())
    checks["A2_typed_token_permutation_invariant"] = a2_flip == 0.0
    checks["A1_typed_token_permutation_sensitive"] = a1_flip > 0.0

    # 5. A2 identity slot is exactly zero.
    with torch.no_grad():
        slot_a2 = a2.typed_embedding.identity_slot(batch.typed_token)
    checks["A2_identity_slot_exactly_zero"] = bool(
        float(slot_a2.abs().max()) == 0.0
    )
    with torch.no_grad():
        slot_a1 = a1.typed_embedding.identity_slot(batch.typed_token)
    checks["A1_identity_slot_nonzero"] = bool(float(slot_a1.abs().max()) > 0.0)

    # 6. Weight-tied T=2 recurrence unchanged.
    a1.train()
    counts = a1.module_call_counts(batch)
    checks["two_round_weight_tying"] = bool(
        counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}
    )
    checks["recurrence_flags_refresh"] = bool(
        int(a1.recurrence_rounds) == 2
        and bool(a1.recurrence_enabled)
        and str(a1.recurrence_mode) == "refresh"
    )

    # 7. Finite forward / backward, finite gradients, code gradient-free.
    a1.zero_grad(set_to_none=True)
    out = a1(batch)
    loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    grads = [
        p.grad
        for p in a1.parameters()
        if p.grad is not None
    ]
    checks["forward_finite"] = bool(torch.isfinite(out).all())
    checks["backward_finite"] = bool(
        all(bool(torch.isfinite(g).all()) for g in grads)
    )
    adapter_grad_nonzero = any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in a1.identity_adapter.parameters()
    )
    checks["adapter_gradient_alive"] = bool(adapter_grad_nonzero)
    checks["fixed_code_has_no_grad"] = bool(a1.identity_code.grad is None)
    parent_grad_nonzero = any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in a1.parent_embedding.parameters()
    )
    checks["parent_gradient_alive"] = bool(parent_grad_nonzero)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "adapter_params": int(expected_adapter),
        "total_params": int(expected_total_value),
        "shared_tensors_checked": len(shared),
        "A1_shared_max_abs_diff_vs_A0": float(shared1),
        "A2_shared_max_abs_diff_vs_A0": float(shared2),
        "A1_A2_adapter_max_abs_diff": float(adapter_diff),
        "A1_permutation_max_abs_diff": float(a1_flip),
        "A2_permutation_max_abs_diff": float(a2_flip),
        "code_statistics": code_stats,
        "module_call_counts": counts,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"identity capacity control sanity failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def _valid_loader():
    _train_data, valid_data, _ = rec.load_encoded()
    return zpp._make_loader(list(valid_data), 128, False, 0)


def train(condition: str, seed: int, device: str = "cpu") -> dict[str, Any]:
    tag = tag_for(condition)
    train_data, valid_data, _ = rec.load_encoded()
    original = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = CURVE_DIR, STATE_DIR, RUNS_DIR
    snapshot_dir = SNAPSHOT_DIR / f"{tag}_seed{seed}"
    rss_before = hw._rss_peak_kb()
    started = time.perf_counter()
    total = expected_total(condition)
    build_fn = lambda s: build_model(condition, s)  # noqa: E731
    try:
        summary = shead.train_model(
            build_fn=build_fn,
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=tag,
            save_state=True,
            real_batch_identity=False,
            expected_total=total,
            snapshot_dir=snapshot_dir,
            device=device,
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original
    elapsed = float(time.perf_counter() - started)

    summary = dict(summary)
    summary["config_name"] = f"identity_{condition}"
    summary["condition"] = condition
    summary["identity_mode"] = "fixed" if condition == "A1" else "none"
    summary["adapter_hidden"] = int(ADAPTER_HIDDEN)
    summary["q_dim"] = Q_DIM
    summary["epoch_time_s"] = float(summary["wall_clock_s"]) / max(
        int(summary["epochs_run"]), 1
    )
    summary["peak_rss_kb"] = hw._rss_peak_kb()
    summary["peak_rss_delta_kb"] = hw._rss_peak_kb() - int(rss_before)
    summary["outer_wall_clock_s"] = elapsed
    summary["platform"] = platform.platform()
    summary["deterministic_algorithms"] = bool(
        torch.are_deterministic_algorithms_enabled()
    )
    summary["peak_gpu_memory_mb"] = (
        float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if torch.cuda.is_available()
        else 0.0
    )
    hw._write_json(RUNS_DIR / f"{tag}_seed{seed}.json", summary)

    soup(condition, seed)
    return summary


def soup(condition: str, seed: int) -> dict[str, Any]:
    tag = tag_for(condition)
    summary = _read_json(RUNS_DIR / f"{tag}_seed{seed}.json")
    top = vd._top5_epochs(summary)
    soup_state = vd.build_soup_state(top)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
    torch.save(soup_state, soup_path)

    loader = _valid_loader()
    model = build_model(condition, seed)
    selection_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    targets, best_preds = vd._predict_state(
        model,
        torch.load(selection_path, map_location="cpu", weights_only=True),
        loader,
    )
    _t2, soup_preds = vd._predict_state(model, soup_state, loader)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "condition": condition,
        "identity_mode": "fixed" if condition == "A1" else "none",
        "tag": tag,
        "seed": int(seed),
        "adapter_hidden": int(ADAPTER_HIDDEN),
        "top5_epochs": [int(row["epoch"]) for row in top],
        "top5_valid_mae": [float(row["valid_mae"]) for row in top],
        "best_epoch": int(summary["best_epoch"]),
        "best_checkpoint_valid_mae": vd._mae(targets, best_preds),
        "top5_soup_valid_mae": vd._mae(targets, soup_preds),
        "soup_improvement_over_best": vd._mae(targets, best_preds)
        - vd._mae(targets, soup_preds),
        "parameters": int(sum(p.numel() for p in soup_state.values())),
        "epoch_time_s": float(summary.get("epoch_time_s", float("nan"))),
        "wall_clock_s": float(summary.get("wall_clock_s", float("nan"))),
        "peak_gpu_memory_mb": float(summary.get("peak_gpu_memory_mb", 0.0)),
        "official_test_loaded": False,
        "best_predictions": best_preds.tolist(),
        "soup_predictions": soup_preds.tolist(),
        "valid_targets": targets.tolist(),
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json", payload)
    return payload


def train_queue(conditions: Sequence[str], seeds: Sequence[int], device: str) -> None:
    for condition in conditions:
        for seed in seeds:
            if (RUNS_DIR / f"{tag_for(condition)}_seed{seed}.json").exists():
                print(f"skip existing {condition} seed{seed}", flush=True)
                continue
            print(f"=== train {condition} seed{seed} device={device} ===", flush=True)
            train(str(condition), int(seed), device=device)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().to(torch.float32).cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def repro(
    condition: str, seed: int, epochs: int, device: str, run_tag: str = ""
) -> dict[str, Any]:
    """Short GPU reproducibility sanity in an isolated result subtree."""
    global CURVE_DIR, STATE_DIR, RUNS_DIR, SNAPSHOT_DIR, SOUP_DIR
    tag = str(run_tag or "default").replace("/", "_")
    saved = (CURVE_DIR, STATE_DIR, RUNS_DIR, SNAPSHOT_DIR, SOUP_DIR)
    root = RESULTS_DIR / "repro" / tag
    CURVE_DIR, STATE_DIR, RUNS_DIR, SNAPSHOT_DIR, SOUP_DIR = (
        root / "curves",
        root / "states",
        root / "runs",
        root / "snapshots",
        root / "soup_states",
    )
    original = dict(shead.OPTIMIZED_PROTOCOL)
    shead.OPTIMIZED_PROTOCOL = {
        **original,
        "max_epochs": int(epochs),
        "patience": int(epochs),
    }
    try:
        summary = train(condition, seed, device=device)
        state = torch.load(
            STATE_DIR / f"{tag_for(condition)}_seed{seed}_selection_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        state_hash = _state_sha256(state)
        curve_path = CURVE_DIR / f"{tag_for(condition)}_seed{seed}_curve.csv"
        valid_curve: list[float] = []
        if curve_path.exists():
            for line in curve_path.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split(",")
                if len(parts) > 2:
                    valid_curve.append(float(parts[2]))
    finally:
        shead.OPTIMIZED_PROTOCOL = original
        CURVE_DIR, STATE_DIR, RUNS_DIR, SNAPSHOT_DIR, SOUP_DIR = saved
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "gpu_reproducibility_sanity",
        "condition": condition,
        "seed": int(seed),
        "epochs": int(epochs),
        "run_tag": tag,
        "device": str(device),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "best_valid_mae": float(summary["best_valid_mae"]),
        "best_epoch": int(summary["best_epoch"]),
        "epochs_run": int(summary["epochs_run"]),
        "wall_clock_s": float(summary["wall_clock_s"]),
        "valid_mae_curve": valid_curve,
        "selection_state_sha256": state_hash,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"repro_{tag}_{condition}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# baseline guard
# ---------------------------------------------------------------------------

# Frozen cell-A (seed 0) checkpoint forward references, produced on the remote
# deterministic-GPU regime and pulled into ``baseline_guard/``.
BASELINE_GUARD_REFERENCE = {
    "best_checkpoint_valid_mae": 0.12970963285310427,
    "top5_soup_valid_mae": 0.12470379155874252,
}


def baseline_guard(tol: float = 1.0e-6) -> dict[str, Any]:
    """Re-run the stored frozen cell-A checkpoint forward on the local CPU.

    Confirms the frozen builder still reproduces the stored selection/soup
    states exactly (bit-level parameter counts and MAE) before any new
    experiment is interpreted.
    """
    model = cd.build_cell("A", 0)
    total = hw._n_params(model)
    head = hw._n_params(model.head)
    loader = _valid_loader()
    selection = torch.load(
        BASELINE_DIR / "cell_A_seed0_selection_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    soup_state = torch.load(
        BASELINE_DIR / "cell_A_seed0_top5_soup.pt",
        map_location="cpu",
        weights_only=True,
    )
    targets, best_preds = vd._predict_state(model, selection, loader)
    _t2, soup_preds = vd._predict_state(model, soup_state, loader)
    best_mae = vd._mae(targets, best_preds)
    soup_mae = vd._mae(targets, soup_preds)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "builder_total_params": int(total),
        "builder_head_params": int(head),
        "builder_unified_width": int(model.unified_graph_width),
        "stored_best_checkpoint_valid_mae": best_mae,
        "stored_top5_soup_valid_mae": soup_mae,
        "reference_best_checkpoint_valid_mae": BASELINE_GUARD_REFERENCE[
            "best_checkpoint_valid_mae"
        ],
        "reference_top5_soup_valid_mae": BASELINE_GUARD_REFERENCE[
            "top5_soup_valid_mae"
        ],
        "best_abs_diff": abs(
            best_mae - BASELINE_GUARD_REFERENCE["best_checkpoint_valid_mae"]
        ),
        "soup_abs_diff": abs(
            soup_mae - BASELINE_GUARD_REFERENCE["top5_soup_valid_mae"]
        ),
        "reproduced": bool(
            abs(best_mae - BASELINE_GUARD_REFERENCE["best_checkpoint_valid_mae"])
            <= tol
            and abs(soup_mae - BASELINE_GUARD_REFERENCE["top5_soup_valid_mae"])
            <= tol
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "baseline_guard.json", payload)
    return payload


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def _a0_reference() -> dict[str, Any]:
    per_seed_soup: dict[int, float] = {}
    per_seed_raw: dict[int, float] = {}
    for seed in SEEDS:
        path = A0_RESULTS_DIR / f"soup_cell_A_seed{seed}.json"
        if path.exists():
            payload = _read_json(path)
            per_seed_soup[int(seed)] = float(payload["top5_soup_valid_mae"])
            per_seed_raw[int(seed)] = float(payload["best_checkpoint_valid_mae"])
        else:
            per_seed_soup[int(seed)] = float(A0_REFERENCE_SOUP_PER_SEED[seed])
            per_seed_raw[int(seed)] = float(A0_REFERENCE_RAW_PER_SEED[seed])
    return {
        "source": str(A0_RESULTS_DIR),
        "soup_per_seed": {str(k): v for k, v in per_seed_soup.items()},
        "raw_per_seed": {str(k): v for k, v in per_seed_raw.items()},
        "soup_2seed_mean": float(np.mean(list(per_seed_soup.values()))),
        "raw_2seed_mean": float(np.mean(list(per_seed_raw.values()))),
        "frozen_reference_soup_2seed_mean": A0_REFERENCE_SOUP_2SEED_MEAN,
        "params": CELL_A_PARAMS,
    }


def _subgroup_valid_error(
    targets: np.ndarray,
    predictions: np.ndarray,
    valid_data: Sequence[Data],
    occurrence: Mapping[int, int],
) -> dict[str, Any]:
    """Rare/common typed-token subgroup valid MAE (explanatory only)."""
    if len(valid_data) != len(targets):
        return {"available": False, "reason": "valid length mismatch"}
    scores = np.empty(len(valid_data), dtype=np.float64)
    for index, graph in enumerate(valid_data):
        tokens = graph.typed_token.tolist()
        if tokens:
            scores[index] = float(
                np.mean([occurrence.get(int(t), 0) for t in tokens])
            )
        else:
            scores[index] = 0.0
    errors = np.abs(targets - predictions)
    order = np.argsort(scores)
    n = len(order)
    if n < 8:
        return {"available": False, "reason": "too few valid molecules"}
    q = max(1, n // 4)
    rare = order[:q]
    common = order[-q:]
    return {
        "available": True,
        "definition": "per-molecule mean train occurrence of its typed tokens",
        "n_valid": int(n),
        "rare_quartile_mean_mae": float(errors[rare].mean()),
        "common_quartile_mean_mae": float(errors[common].mean()),
        "common_minus_rare": float(errors[common].mean() - errors[rare].mean()),
    }


def diagnostics() -> dict[str, Any]:
    train_data, valid_data, _ = rec.load_encoded()
    batch = hw._first_batch(valid_data, torch.device("cpu"))
    occurrence: dict[int, int] = {}
    for graph in train_data:
        for token in graph.typed_token.tolist():
            occurrence[int(token)] = occurrence.get(int(token), 0) + 1

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "a0_reference": _a0_reference(),
        "code": _code_statistics(fixed_identity_codes(TYPED_VOCABULARY_SIZE)),
        "conditions": {},
        "prediction_disagreement": {},
        "subgroups": {},
        "official_test_loaded": False,
    }

    soup_preds: dict[str, dict[int, np.ndarray]] = {}
    for condition in CONDITIONS:
        payload["conditions"][condition] = {}
        soup_preds[condition] = {}
        for seed in SEEDS:
            sp = soup(condition, seed)
            model = build_model(condition, seed)
            model.load_state_dict(
                torch.load(
                    STATE_DIR / f"{tag_for(condition)}_seed{seed}_selection_state.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            payload["conditions"][condition][str(seed)] = hw._centre_diagnostics(
                model, batch
            )
            payload["conditions"][condition][str(seed)]["adapter_output_norm"] = (
                _adapter_output_norm(model, batch)
            )
            soup_preds[condition][int(seed)] = np.asarray(sp["soup_predictions"])
            payload["subgroups"][f"{condition}_seed{seed}"] = _subgroup_valid_error(
                np.asarray(sp["valid_targets"]),
                np.asarray(sp["soup_predictions"]),
                valid_data,
                occurrence,
            )

        payload["prediction_disagreement"][condition] = {
            "soup_seed0_seed1": float(
                np.mean(
                    np.abs(
                        soup_preds[condition][0] - soup_preds[condition][1]
                    )
                )
            ),
        }
    # Cross-condition disagreement at matched seed.
    for seed in SEEDS:
        payload["prediction_disagreement"][f"A1_vs_A2_seed{seed}"] = float(
            np.mean(np.abs(soup_preds["A1"][seed] - soup_preds["A2"][seed]))
        )
    hw._write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


def _adapter_output_norm(model: IdentityCapacityModel, batch: Data) -> float:
    model.eval()
    with torch.no_grad():
        model.typed_embedding.set_context(batch.patch_cont)
        value = model.typed_embedding(batch.typed_token)
    return float(value.norm(dim=1).mean())


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide() -> dict[str, Any]:
    a0 = _a0_reference()
    per_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        soups = [soup(condition, seed) for seed in SEEDS]
        soup_values = [float(x["top5_soup_valid_mae"]) for x in soups]
        raw_values = [float(x["best_checkpoint_valid_mae"]) for x in soups]
        run0 = _read_json(RUNS_DIR / f"{tag_for(condition)}_seed0.json")
        per_condition[condition] = {
            "params": int(expected_total(condition)),
            "soup_per_seed": {
                str(s): float(soups[i]["top5_soup_valid_mae"])
                for i, s in enumerate(SEEDS)
            },
            "raw_per_seed": {
                str(s): float(soups[i]["best_checkpoint_valid_mae"])
                for i, s in enumerate(SEEDS)
            },
            "soup_2seed_mean": float(np.mean(soup_values)),
            "raw_2seed_mean": float(np.mean(raw_values)),
            "best_epochs": [int(x["best_epoch"]) for x in soups],
            "epoch_time_s": float(run0.get("epoch_time_s", float("nan"))),
            "peak_gpu_memory_mb": float(
                run0.get("peak_gpu_memory_mb", 0.0)
            ),
        }

    a0_mean = float(a0["soup_2seed_mean"])
    a1_mean = per_condition["A1"]["soup_2seed_mean"]
    a2_mean = per_condition["A2"]["soup_2seed_mean"]
    d10 = a1_mean - a0_mean  # positive => A1 worse than A0
    d20 = a2_mean - a0_mean  # positive => A2 worse than A0
    d21 = a2_mean - a1_mean  # positive => A2 worse than A1

    def _close(x: float, y: float) -> bool:
        return abs(x - y) < MEANINGFUL_THRESHOLD

    if _close(a0_mean, a1_mean) and (a2_mean - min(a0_mean, a1_mean)) > MEANINGFUL_THRESHOLD:
        case = "I"
        reading = (
            "identity distinguishability matters, per-id learned memory does not"
        )
    elif (a1_mean - a0_mean) > MEANINGFUL_THRESHOLD and _close(a1_mean, a2_mean):
        case = "II"
        reading = (
            "gain comes from task-specific learned categorical memory, not from "
            "knowing that tokens differ"
        )
    elif _close(a0_mean, a1_mean) and _close(a1_mean, a2_mean):
        case = "III"
        reading = "typed lookup largely dispensable; shared capacity suffices"
    elif a0_mean < a1_mean < a2_mean:
        case = "IV"
        reading = "both identity and learned per-token semantics contribute"
    else:
        case = "UNRESOLVED"
        reading = "ordering does not match a pre-registered case cleanly"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "a0_reference": a0,
        "conditions": per_condition,
        "deltas": {
            "A1_minus_A0_soup": float(d10),
            "A2_minus_A0_soup": float(d20),
            "A2_minus_A1_soup": float(d21),
        },
        "meaningful_threshold": float(MEANINGFUL_THRESHOLD),
        "case": case,
        "reading": reading,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "parameter_accounting": _maybe("parameter_accounting.json"),
        "sanity": (lambda d: None if d is None else d["checks"])(_maybe("sanity.json")),
        "baseline_guard": _maybe("baseline_guard.json"),
        "diagnostics": _maybe("diagnostics.json"),
        "decision": _maybe("decision.json"),
        "official_test_loaded": False,
    }
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
            "params",
            "sanity",
            "baseline_guard",
            "train",
            "train_queue",
            "repro",
            "soup",
            "diagnostics",
            "decide",
            "report",
        ],
    )
    parser.add_argument("--condition", type=str, default="A1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", type=str, default="A1,A2")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    global DETERMINISTIC
    DETERMINISTIC = bool(args.deterministic)
    torch.set_num_threads(int(shead.TORCH_THREADS))
    _set_deterministic(DETERMINISTIC)

    if args.stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    if args.stage == "sanity":
        print(json.dumps(sanity()["checks"], indent=2, default=str), flush=True)
    if args.stage == "baseline_guard":
        print(json.dumps(baseline_guard(), indent=2, default=str), flush=True)
    if args.stage == "train":
        print(
            json.dumps(
                train(args.condition, args.seed, args.device),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "train_queue":
        train_queue(
            args.conditions.split(","),
            [int(s) for s in args.seeds.split(",")],
            args.device,
        )
    if args.stage == "repro":
        print(
            json.dumps(
                repro(
                    args.condition, args.seed, args.epochs, args.device, args.run_tag
                ),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "soup":
        print(
            json.dumps(soup(args.condition, args.seed), indent=2, default=str),
            flush=True,
        )
    if args.stage == "diagnostics":
        print(json.dumps(diagnostics(), indent=2, default=str), flush=True)
    if args.stage == "decide":
        print(json.dumps(decide(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
