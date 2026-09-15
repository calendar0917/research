"""Compact-v4-smallhead + 2-round weight-tied recurrent pair--centre variant.

Single falsification experiment
--------------------------------
The current best compact-v4 / small-head model computes the structural pair
relation ``q_ij`` exactly once:

    h^0 -> q^0 -> centre aggregate -> h^1 -> readout/head.

The hypothesis under test is *not* that a new statistic is missing, but that
the pair / structural relation does not persist into the later computation:

    h^0 -> q^0 -> A -> h^1 -> q^1 -> A -> h^2 -> readout/head

where ``q^1 = Q(P(h^1_i), P(h^1_j), r_ij)`` is recomputed from the *updated*
centre states and where ``h^2 = h^1 + U([h^1, A(q^1)])`` reuses the **same**
shared pair encoder ``P``/``Q`` and the **same** centre update ``U``.

Two additional composition controls share the exact same parameters:

* ``stale`` (depth-only): ``h^2 = h^1 + U([h^1, A(q^0)])`` -- two centre updates
  but the relation is never refreshed.
* ``late_refresh`` (one-shot): ``A^0 = A(q^0)`` is replayed into both centre
  updates (``h^1 = h^0 + U(h^0, A^0)``, ``h^2 = h^1 + U(h^1, A^0)``); only then
  is ``q^1 = Q(h^2)`` recomputed and fed *exclusively* to the pair readout.
  This isolates the value of a refreshed final pair representation from the
  value of feeding that refresh back into the centre.

Design constraints (what is deliberately *not* here)
----------------------------------------------------
* Exactly two rounds, weight-tied: no new trainable tensor is added.  The
  candidate has precisely the same parameter count as the matched baseline.
* Round 2 uses the updated centre state ``h^1`` -- it is not a replay of q^0.
* Tokenizer / radius / patch construction / relation descriptor / distance
  buckets / centre pooling / readout / global descriptors / topology branch /
  small head / optimizer / scheduler / training protocol are all inherited
  unchanged.
* No attention, no path features, no extra statistics, no per-round weights.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_pair_centre <stage>

Stages: ``sanity sanity_late smoke baseline recurrent stale late_refresh control
control_late decision recurrent_seed1 terminal_test final all``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_training_sufficiency as ztraining
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_recurrent_pair_centre"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
FIGURE_DIR = RESULTS_DIR / "figures"
RUNS_DIR = RESULTS_DIR / "runs"

PROTOCOL_VERSION = "compact_v4_recurrent_pair_centre_v1"

# Matched baseline = compact-v4-smallhead end-to-end (82,115 trainable params).
R_DIM = 302
EXPECTED_SMALL_TOTAL = 82115
EXPECTED_HEAD = 4135
Q_DIM = 16
RECURRENCE_ROUNDS = 2

# Pre-registered continuation gate on official-valid MAE:
#   delta = baseline_valid - recurrent_valid  (positive => recurrence helps)
ARCH_GATE = 0.003
ARCH_GATE_SEED1 = 0.003
ARCH_MEAN_GATE = 0.003

CANONICAL_V4_CONFIG = shead.CANONICAL_V4_CONFIG
V4_SEED0_VALID = shead.V4_SEED0_VALID
V4_SEED1_VALID = shead.V4_SEED1_VALID
V4_PARAMS = shead.V4_PARAMS
SMALLHEAD_SEED0_VALID = 0.14533376283763208  # existing matched smallhead run
SMALLHEAD_SEED1_VALID = 0.13805893784115325

# P2 context only (NOT matched to small-head; 99,613-param hinge backbone).
P2_ONE_SHOT_REFRESH_DELTA = 0.001963

EPS = 1.0e-8

# ---------------------------------------------------------------------------
# io helpers (mirrors the surrounding experiments)
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _environment_fingerprint() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "device": "cpu",
    }


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class PatchPathRecurrentPairCentreModel(shead.PatchPathSmallHeadModel):
    """compact-v4-smallhead with a 2-round weight-tied pair--centre recursion.

    The inherited modules are byte-for-byte the matched small-head baseline.
    This class adds **zero** trainable tensors: round 2 only re-applies the
    existing ``pair_projection`` / ``relation_encoder`` / ``distance_gate`` /
    ``pair_encoder`` and the existing ``center_update`` to the updated centre
    state ``h^1``.

    Forward semantics (``recurrence_enabled=True``)
    -----------------------------------------------
    ``h^0``  : patch states from the inherited patch encoder.
    ``q^0``  : shared pair encoder applied to ``h^0`` endpoints.
    ``h^1``  : ``h^0 + center_update([h^0, pool(q^0)])``.
    ``q^1``  : shared pair encoder applied to ``h^1`` endpoints.
    ``h^2``  : ``h^1 + center_update([h^1, pool(q^1)])``.
    readout  : unary moments of ``h^2`` + keyed pair moments of ``q^1`` +
               global + topology, exactly as the baseline readout.

    With ``recurrence_enabled=False`` (or ``rounds=1``) this reduces exactly to
    the matched baseline computation (unary from ``h^1``, pair moments from
    ``q^0``).
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        recurrence_rounds: int = RECURRENCE_ROUNDS,
        recurrence_enabled: bool = True,
        recurrence_mode: str = "refresh",
        pair_to_pair_enabled: bool = False,
        pair_to_pair_hidden: int = 24,
        gate_update_magnitude: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(typed_vocabulary_size, parent_vocabulary_size, **kwargs)
        if int(self.pair_hidden) != Q_DIM:
            raise ValueError(
                f"recurrence requires pair_hidden == {Q_DIM}; got {self.pair_hidden}"
            )
        if self.center_update is None:
            raise ValueError("recurrence requires center_context=True (a centre update)")
        if int(recurrence_rounds) < 1:
            raise ValueError("recurrence_rounds must be >= 1")
        if recurrence_mode not in {"refresh", "stale", "late_refresh"}:
            raise ValueError(
                f"unknown recurrence_mode={recurrence_mode!r}; expected "
                "refresh|stale|late_refresh"
            )
        self.recurrence_rounds = int(recurrence_rounds)
        self.recurrence_enabled = bool(recurrence_enabled)
        # "refresh": round 2 recomputes q from the updated h^(1).
        # "stale":  round 2 reuses q^(0) -> a depth-only control with the same
        #           number of centre updates and the same parameter count.
        # "late_refresh": A^(0) is replayed into both centre updates; q^(1) is
        #           recomputed from h^(2) after refinement and read out only.
        self.recurrence_mode = str(recurrence_mode)
        # -- direct pair-to-pair composition (shared intermediate k) ---------
        # Applied to the refreshed relation q^(1) *before* it enters the second
        # centre aggregation.  Disabled for the matched recurrent baseline so
        # that parameter count is unchanged when ``pair_to_pair_enabled=False``.
        self.pair_to_pair_enabled = bool(pair_to_pair_enabled)
        self.pair_to_pair_hidden = int(pair_to_pair_hidden)
        if self.pair_to_pair_enabled:
            # psi: shared composition MLP on [q_ik + q_kj ; |q_ik - q_kj| ; q_ik*q_kj]
            self.pair_to_pair_psi = nn.Sequential(
                nn.Linear(3 * Q_DIM, self.pair_to_pair_hidden),
                nn.ReLU(),
                nn.Linear(self.pair_to_pair_hidden, Q_DIM),
            )
            # phi: residual update on [q_ij ; mean_k psi]
            self.pair_to_pair_phi = nn.Sequential(
                nn.Linear(2 * Q_DIM, self.pair_to_pair_hidden),
                nn.ReLU(),
                nn.Linear(self.pair_to_pair_hidden, Q_DIM),
            )
            # Zero-init the residual branch so the composed model is exactly
            # the matched recurrent function at initialisation.
            nn.init.zeros_(self.pair_to_pair_phi[-1].weight)
            nn.init.zeros_(self.pair_to_pair_phi[-1].bias)
        else:
            self.pair_to_pair_psi = None
            self.pair_to_pair_phi = None
        # -- minimal shared residual-magnitude gate ---------------------------
        # One learnable scalar ``a`` gives ``alpha = sigmoid(a)``, applied to
        # *every* centre update.  The T=2 recurrence therefore shares a single
        # ``alpha`` across both rounds:
        #     h^(t+1) = h^(t) + alpha * U(h^(t), A^(t)).
        # ``a`` is initialised to 0 so ``alpha == 0.5``.  With
        # ``gate_update_magnitude=False`` **no parameter is added** and the
        # forward path is bit-identical to the canonical ungated model.
        self.gate_update_magnitude = bool(gate_update_magnitude)
        if self.gate_update_magnitude:
            self.update_gate_logit = nn.Parameter(torch.zeros(()))
        else:
            self.update_gate_logit = None
        # Local triple patterns depend only on the graph size; cache them.
        self._triple_pattern_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # Diagnostic / mechanism-probe switches (never parameters, never
        # buffers; they only change the forward composition).
        self.force_zero_center_update = False
        self.force_zero_pair_to_pair = False
        self.capture_diagnostics = False
        self._reset_diagnostics()

    def update_alpha(self) -> torch.Tensor:
        """Learned scalar ``sigmoid(a)`` (``1.0`` when the gate is disabled)."""
        if self.update_gate_logit is None:
            return torch.ones((), dtype=torch.float32)
        return torch.sigmoid(self.update_gate_logit)

    # -- diagnostics ---------------------------------------------------------
    def _reset_diagnostics(self) -> None:
        self.last_h0: torch.Tensor | None = None
        self.last_h1: torch.Tensor | None = None
        self.last_h2: torch.Tensor | None = None
        self.last_q0: torch.Tensor | None = None
        self.last_q1: torch.Tensor | None = None
        self.last_q1_raw: torch.Tensor | None = None
        self.last_pair_to_pair_left: torch.Tensor | None = None
        self.last_pair_to_pair_right: torch.Tensor | None = None
        self.last_pair_to_pair_target: torch.Tensor | None = None
        self.last_center_context0: torch.Tensor | None = None
        self.last_center_context1: torch.Tensor | None = None
        self.last_pair_source: torch.Tensor | None = None
        self.last_pair_target: torch.Tensor | None = None
        self.last_pair_bucket: torch.Tensor | None = None
        self.last_pair_batch: torch.Tensor | None = None
        self.last_n_graphs: int = 0
        self.diag_q_cos_sum = 0.0
        self.diag_q_drift_sum = 0.0
        self.diag_batches = 0

    def module_call_counts(self, data: Data) -> dict[str, int]:
        """Count forward invocations of the shared pair/centre modules.

        A weight-tied 2-round forward must call each shared module exactly
        twice as often as the 1-round baseline (pair projection twice per
        relation, pair encoder once per relation, centre update once per
        round).
        """
        counts = {"pair_projection": 0, "pair_encoder": 0, "center_update": 0}
        handles = []

        def make_hook(name: str):
            def hook(_module, _inputs, _output):
                counts[name] += 1

            return hook

        for name in counts:
            module = getattr(self, name, None)
            if module is not None:
                handles.append(module.register_forward_hook(make_hook(name)))
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                self.encode(data)
        finally:
            for handle in handles:
                handle.remove()
            self.train(was_training)
        return counts

    # -- shared, weight-tied pair value --------------------------------------
    def _pair_value(
        self,
        patch: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """``q = Q(P(h)_i, P(h)_j, r_ij)`` reusing the baseline modules."""
        left = self.pair_projection(patch[source])
        right = self.pair_projection(patch[target])
        product = left * right
        pair_input = torch.cat(
            [
                left + right,
                torch.abs(left - right),
                product * gate,
                relation,
            ],
            dim=1,
        )
        return self.pair_encoder(pair_input)

    # -- direct pair-to-pair composition (shared intermediate k) --------------
    def _triple_pattern(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Local ``(left_rel, right_rel, target)`` triple indices for ``n`` patches.

        Pairs inside one graph are enumerated in the canonical order
        ``(a, b)`` with ``a < b`` (see ``_graph_record``), so the local pair
        index is ``p(a, b) = a*n - a*(a+1)//2 + (b - a - 1)``.  For every
        ordered triple ``(i, k, j)`` with ``i < j`` and ``k not in {i, j}`` the
        message ``psi(q_{i,k}, q_{k,j})`` contributes to target pair ``(i, j)``.
        The pattern depends only on ``n`` and is cached across forward passes.
        """
        cached = self._triple_pattern_cache.get(int(n))
        if cached is not None:
            return cached
        arange = torch.arange(int(n), dtype=torch.long)
        eye_i = arange[:, None, None]
        eye_k = arange[None, :, None]
        eye_j = arange[None, None, :]
        i = eye_i.expand(int(n), int(n), int(n))
        k = eye_k.expand(int(n), int(n), int(n))
        j = eye_j.expand(int(n), int(n), int(n))
        mask = (i < j) & (k != i) & (k != j)
        i = i[mask]
        k = k[mask]
        j = j[mask]
        lo_ik = torch.minimum(i, k)
        hi_ik = torch.maximum(i, k)
        lo_kj = torch.minimum(k, j)
        hi_kj = torch.maximum(k, j)
        left = lo_ik * int(n) - lo_ik * (lo_ik + 1) // 2 + (hi_ik - lo_ik - 1)
        right = lo_kj * int(n) - lo_kj * (lo_kj + 1) // 2 + (hi_kj - lo_kj - 1)
        target = i * int(n) - i * (i + 1) // 2 + (j - i - 1)
        pattern = (left, right, target)
        self._triple_pattern_cache[int(n)] = pattern
        return pattern

    def _pair_to_pair_refine(
        self,
        q: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        node_batch: torch.Tensor,
        pair_batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """``q_ij* = q_ij + phi(q_ij, mean_k psi(q_ik, q_kj))``.

        Only real ``(i, k)``/``(k, j)`` pairs inside the same graph are
        enumerated; no dense ``(i, j, k)`` tensor is materialised.  The local
        triple patterns are cached by graph size, so per-step work is a single
        gather/scatter over the existing pair rows.
        """
        if not self.pair_to_pair_enabled or self.force_zero_pair_to_pair:
            return q
        if self.pair_to_pair_psi is None or self.pair_to_pair_phi is None:
            return q
        device = q.device
        node_counts = torch.bincount(node_batch, minlength=int(n_graphs))
        pair_counts = torch.bincount(pair_batch, minlength=int(n_graphs))
        pair_offset = torch.zeros(int(n_graphs) + 1, dtype=torch.long, device=device)
        pair_offset[1:] = torch.cumsum(pair_counts, 0)
        left_parts: list[torch.Tensor] = []
        right_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        for graph in range(int(n_graphs)):
            n = int(node_counts[graph])
            if n < 3:
                continue
            local_left, local_right, local_target = self._triple_pattern(n)
            offset = pair_offset[graph]
            left_parts.append(local_left.to(device) + offset)
            right_parts.append(local_right.to(device) + offset)
            target_parts.append(local_target.to(device) + offset)
        if not left_parts:
            return q
        left_index = torch.cat(left_parts)
        right_index = torch.cat(right_parts)
        target_index = torch.cat(target_parts)
        # Diagnostics only; these are views into the index arithmetic above.
        if self.capture_diagnostics or not torch.is_grad_enabled():
            self.last_pair_to_pair_left = left_index
            self.last_pair_to_pair_right = right_index
            self.last_pair_to_pair_target = target_index
        q_left = q[left_index]
        q_right = q[right_index]
        composition_input = torch.cat(
            [q_left + q_right, torch.abs(q_left - q_right), q_left * q_right],
            dim=1,
        )
        message = self.pair_to_pair_psi(composition_input)
        aggregated = torch.zeros_like(q)
        aggregated.index_add_(0, target_index, message)
        counts = torch.zeros(q.shape[0], device=device, dtype=q.dtype)
        counts.index_add_(
            0,
            target_index,
            torch.ones(target_index.shape[0], device=device, dtype=q.dtype),
        )
        aggregated = aggregated / counts.clamp_min(1.0).unsqueeze(1)
        update = self.pair_to_pair_phi(torch.cat([q, aggregated], dim=1))
        return q + update

    # -- forward core --------------------------------------------------------
    def _encode_core(self, data: Data, use_recurrence: bool) -> torch.Tensor:
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])
        e_patch = self._patch_token_value(data)
        structural_blocks: list[torch.Tensor] = []
        if self.structural_context_mode != "none":
            e_ctx = self._structural_context_embedding_value(data)
            if self.structural_context_fusion == "condition":
                e_patch = self._condition_patch(
                    e_patch, e_ctx, self._structural_no_ring_mask(data)
                )
            else:
                structural_blocks.append(e_ctx)
        attribute_blocks: list[torch.Tensor] = []
        if self.attribute_encoder is not None:
            attribute_blocks.append(self.attribute_encoder(data, e_patch))
        patch = self.patch_encoder(
            torch.cat(
                [
                    data.patch_cont,
                    data.patch_context,
                    e_patch,
                    self.parent_embedding(data.parent_token),
                    *structural_blocks,
                    *attribute_blocks,
                ],
                dim=1,
            )
        )
        direct_blocks: list[torch.Tensor] = []
        if self.direct_token_readout:
            token_code = self.typed_embedding.embedding(data.typed_token)
            direct_blocks.append(
                self._pool_values(token_code, data.batch, n_graphs, "moments")
            )

        source = data.pair_index[0]
        target = data.pair_index[1]
        relation = self.relation_encoder(data.pair_relation)
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_batch = data.batch[source]

        rounds = self.recurrence_rounds if use_recurrence else 1

        h = patch
        q = self._pair_value(h, source, target, relation, gate)  # q^(0)
        h0 = h
        q0 = q
        center_contexts: list[torch.Tensor] = []
        centre_states: list[torch.Tensor] = []
        q1_raw: torch.Tensor | None = None
        # "late_refresh": A^(0) is formed exactly once from q^(0) and replayed
        # into *both* centre updates.  The relation is refreshed only after the
        # centre refinement is finished (q^(1) = Q(h^(2))) and enters the pair
        # readout alone -- it is never aggregated back into a centre.
        reuse_center_context = self.recurrence_mode == "late_refresh"
        for round_index in range(rounds):
            if self.center_update is not None:
                if reuse_center_context and center_contexts:
                    # Same A^(0) tensor is reused; no recomputation and no new
                    # aggregate is formed from any refreshed relation.
                    center_context = center_contexts[0]
                else:
                    center_context = self._pool_pairs_to_centres(
                        q,
                        source,
                        target,
                        data.pair_bucket,
                        int(h.shape[0]),
                    )
                center_contexts.append(center_context)
                delta = self.center_update(torch.cat([h, center_context], dim=1))
                if self.force_zero_center_update:
                    delta = torch.zeros_like(delta)
                if self.update_gate_logit is not None:
                    delta = self.update_alpha() * delta
                h = h + delta
                centre_states.append(h)
            if round_index + 1 < rounds:
                # refresh: q^(round+1) MUST be recomputed from the updated
                # centre state.  stale/late_refresh: keep q^(round) inside the
                # centre loop (depth-only / one-shot late relation).
                if self.recurrence_mode == "refresh":
                    q = self._pair_value(h, source, target, relation, gate)
                    q1_raw = q
                    if self.pair_to_pair_enabled:
                        # Direct shared-k composition of q^(1) *before* it enters
                        # the second centre aggregation and the pair readout.
                        q = self._pair_to_pair_refine(
                            q, source, target, data.batch, pair_batch, n_graphs
                        )

        if self.recurrence_mode == "late_refresh" and rounds >= 2:
            # One-shot late relation refresh from the final centre state.  This
            # value is consumed only by ``relation_readout`` below.
            q = self._pair_value(h, source, target, relation, gate)

        unary = self._pool_nodes(h, data.batch, n_graphs)
        relation_readout = self._pool_pairs(
            q, pair_batch, data.pair_bucket, n_graphs
        )

        # h^(1) is the centre state after the first round; h^(2) after the
        # second.  The final pair state is q^(1) for a 2-round forward.
        h1 = centre_states[0] if len(centre_states) >= 1 else None
        h2 = centre_states[1] if len(centre_states) >= 2 else None
        q1 = q if rounds >= 2 else None

        # Persist the last forward for mechanism probes only when asked, so the
        # training loop does not retain the autograd graph unnecessarily.
        if self.capture_diagnostics or not torch.is_grad_enabled():
            self.last_h0 = h0
            self.last_h1 = h1
            self.last_h2 = h2
            self.last_q0 = q0
            self.last_q1 = q1
            self.last_q1_raw = q1_raw
            self.last_center_context0 = (
                center_contexts[0] if center_contexts else None
            )
            self.last_center_context1 = (
                center_contexts[1] if len(center_contexts) > 1 else None
            )
            self.last_pair_source = source
            self.last_pair_target = target
            self.last_pair_bucket = data.pair_bucket
            self.last_pair_batch = pair_batch
            self.last_n_graphs = n_graphs

        graph_hidden = self.global_encoder(global_context)
        readout_blocks = [unary, relation_readout]
        readout_blocks.extend(direct_blocks)
        readout_blocks.append(graph_hidden)
        if self.topology_encoder is not None:
            topology = data.topology_features
            if topology.ndim == 1:
                topology = topology.unsqueeze(0)
            if int(topology.shape[1]) != self.topology_input_width:
                raise RuntimeError(
                    f"topology width mismatch: data={int(topology.shape[1])} "
                    f"model={self.topology_input_width}"
                )
            readout_blocks.append(self.topology_encoder(topology))
        unified = torch.cat(readout_blocks, dim=1)
        return unified

    def encode(self, data: Data) -> torch.Tensor:  # type: ignore[override]
        return self._encode_core(data, use_recurrence=bool(self.recurrence_enabled))

    def encode_original(self, data: Data) -> torch.Tensor:
        """1-round compact-v4-smallhead pathway (must be bit-identical)."""
        return self._encode_core(data, use_recurrence=False)


# ---------------------------------------------------------------------------
# builders (exact init matching with the matched baseline)
# ---------------------------------------------------------------------------


def build_baseline(seed: int = 0) -> shead.PatchPathSmallHeadModel:
    """Matched baseline: compact-v4-smallhead (82,115 params)."""
    return shead.build_smallhead(seed)


def build_recurrent(
    seed: int = 0,
    *,
    recurrence_enabled: bool = True,
    recurrence_rounds: int = RECURRENCE_ROUNDS,
    recurrence_mode: str = "refresh",
    pair_to_pair_enabled: bool = False,
    pair_to_pair_hidden: int = 24,
    gate_update_magnitude: bool = False,
    typed_vocabulary_size: int = 6785,
    parent_vocabulary_size: int = 32,
    head_seed: int = shead.SMALL_HEAD_SEED,
) -> PatchPathRecurrentPairCentreModel:
    """Build the recurrent model with bit-identical shared init to baseline."""
    baseline = shead.build_baseline(seed)
    baseline_state = {
        key: value.detach().clone() for key, value in baseline.state_dict().items()
    }
    shead._seed_everything(seed)
    model = PatchPathRecurrentPairCentreModel(
        int(typed_vocabulary_size),
        int(parent_vocabulary_size),
        recurrence_rounds=int(recurrence_rounds),
        recurrence_enabled=bool(recurrence_enabled),
        recurrence_mode=str(recurrence_mode),
        pair_to_pair_enabled=bool(pair_to_pair_enabled),
        pair_to_pair_hidden=int(pair_to_pair_hidden),
        gate_update_magnitude=bool(gate_update_magnitude),
        **shead._base_kwargs(),
    )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in baseline_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)
    # Same head-init convention as the matched small-head baseline.
    torch.manual_seed(int(head_seed))
    model.head = GenericReader(int(model.unified_graph_width), shead.SMALL_HEAD_HIDDEN)
    return model


def build_stale(seed: int = 0) -> PatchPathRecurrentPairCentreModel:
    """Depth-only control: 2 centre updates, but q^(0) is reused in round 2."""
    return build_recurrent(seed, recurrence_mode="stale")


def build_late_refresh(seed: int = 0) -> PatchPathRecurrentPairCentreModel:
    """One-shot late relation refresh from the final centre state ``h^(2)``.

    Both centre updates replay the same stale aggregate ``A^(0)`` (so the
    centre path is bit-identical to the ``stale`` control); only the final
    pair readout consumes ``q^(1) = Q(h^(2))``.
    """
    return build_recurrent(seed, recurrence_mode="late_refresh")


def build_gated(seed: int = 0) -> PatchPathRecurrentPairCentreModel:
    """Canonical T=2 recurrent model + one shared scalar update-magnitude gate.

    Adds exactly one parameter (``update_gate_logit``, initialised to 0 so
    ``alpha == 0.5``); all other structure and initialisation are identical to
    :func:`build_recurrent`.
    """
    return build_recurrent(seed, gate_update_magnitude=True)


def build_pair_to_pair(
    seed: int = 0, *, pair_to_pair_hidden: int = 24
) -> PatchPathRecurrentPairCentreModel:
    """Recurrent T=2 model plus direct shared-``k`` pair-to-pair composition.

    The composed relation ``q^(1)*`` replaces ``q^(1)`` immediately before the
    second centre aggregation and the pair readout.  All original shared
    tensors keep their exact baseline initialisation; only the new ``psi``/
    ``phi`` composition MLPs add parameters.
    """
    return build_recurrent(
        seed,
        recurrence_mode="refresh",
        pair_to_pair_enabled=True,
        pair_to_pair_hidden=int(pair_to_pair_hidden),
    )


# ---------------------------------------------------------------------------
# data / training plumbing
# ---------------------------------------------------------------------------


def load_encoded() -> tuple[list[Data], list[Data], dict[str, Any]]:
    return shead.build_encoded_records()


def train(
    build_fn: Callable[[int], nn.Module],
    train_data: Sequence[Data],
    valid_data: Sequence[Data],
    *,
    seed: int,
    tag: str,
    real_batch_identity: bool = True,
    protocol_override: Mapping[str, Any] | None = None,
    expected_total: int | None = None,
    snapshot_dir: Path | None = None,
    data_seed: int | None = None,
) -> dict[str, Any]:
    """Run the inherited training loop, redirecting outputs to this result dir."""
    original = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    original_protocol = shead.OPTIMIZED_PROTOCOL
    shead.CURVE_DIR = CURVE_DIR
    shead.STATE_DIR = STATE_DIR
    shead.RUNS_DIR = RUNS_DIR
    if protocol_override:
        protocol = dict(original_protocol)
        protocol.update(protocol_override)
        shead.OPTIMIZED_PROTOCOL = protocol
    try:
        return shead.train_model(
            build_fn=build_fn,
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=str(tag),
            save_state=True,
            real_batch_identity=bool(real_batch_identity),
            expected_total=expected_total,
            snapshot_dir=snapshot_dir,
            data_seed=data_seed,
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original
        shead.OPTIMIZED_PROTOCOL = original_protocol


def _first_batch(valid_data: Sequence[Data], device: torch.device) -> Data:
    batch = next(
        iter(zpp._make_loader(list(valid_data)[:128], 128, False, 0))
    ).to(device)
    return batch


# ---------------------------------------------------------------------------
# static / mechanism sanity
# ---------------------------------------------------------------------------


def _state_shapes(module: nn.Module) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    train_data, valid_data, audit = load_encoded()
    batch = _first_batch(valid_data, device)

    baseline = build_baseline(0).to(device).eval()
    recurrent = build_recurrent(0).to(device).eval()

    # --- parameter / architecture identity --------------------------------
    params_baseline = _n_params(baseline)
    params_recurrent = _n_params(recurrent)
    shapes_baseline = _state_shapes(baseline)
    shapes_recurrent = _state_shapes(recurrent)
    head_baseline = _n_params(baseline.head)
    head_recurrent = _n_params(recurrent.head)

    with torch.no_grad():
        # 1-round recurrent pathway must equal the parent encode bit-exactly.
        r_parent = baseline.encode(batch)
        r_orig = recurrent.encode_original(batch)
        diff_original = float((r_parent - r_orig).abs().max())

        # At initialisation the centre update is zero, so the 2-round model is
        # also mathematically identical to the baseline function.
        r_init = recurrent.encode(batch)
        diff_init = float((r_parent - r_init).abs().max())

    # --- weight tying: shared modules invoked once per round --------------
    counts_recurrent = recurrent.module_call_counts(batch)
    counts_baseline = baseline.module_call_counts(batch) if hasattr(
        baseline, "module_call_counts"
    ) else None
    # baseline is a plain PatchPathSmallHeadModel; count manually via hooks.
    if counts_baseline is None:
        counts: dict[str, int] = {"pair_projection": 0, "pair_encoder": 0, "center_update": 0}

        def _hook(name):
            def hook(_module, _inputs, _output):
                counts[name] += 1

            return hook

        handles = [
            baseline.pair_projection.register_forward_hook(_hook("pair_projection")),
            baseline.pair_encoder.register_forward_hook(_hook("pair_encoder")),
            baseline.center_update.register_forward_hook(_hook("center_update")),
        ]
        with torch.no_grad():
            baseline.encode(batch)
        for handle in handles:
            handle.remove()
        counts_baseline = counts

    # --- round-2 genuinely depends on the updated centre state ------------
    perturbed = build_recurrent(0).to(device).eval()
    perturbed.capture_diagnostics = True
    with torch.no_grad():
        for parameter in perturbed.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    out_perturbed = perturbed.encode(batch)
    h0, h1, h2 = perturbed.last_h0, perturbed.last_h1, perturbed.last_h2
    q0, q1 = perturbed.last_q0, perturbed.last_q1
    assert h0 is not None and h1 is not None and h2 is not None and q0 is not None and q1 is not None
    center_moves = float((h1 - h0).abs().max())
    second_update_moves = float((h2 - h1).abs().max())
    q_drift = float((q1 - q0).abs().max())
    q_cosine = float(F.cosine_similarity(q0, q1, dim=1, eps=EPS).mean())
    # q1 semantic dependency on h1 (autograd, not just numerical drift).
    grad_q1_h1 = torch.autograd.grad(
        q1.sum(), h1, retain_graph=True, allow_unused=False
    )[0]
    grad_q1_h1_max = float(grad_q1_h1.abs().max())

    # If the centre update is forced to zero, both rounds collapse to the
    # baseline function exactly -> proves round 2 rides on the update.
    collapsed = build_recurrent(0).to(device).eval()
    collapsed.capture_diagnostics = True
    with torch.no_grad():
        for parameter in collapsed.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    collapsed.force_zero_center_update = True
    with torch.no_grad():
        r_collapsed = collapsed.encode(batch)
        r_baseline_same_init = baseline.encode(batch)
    diff_collapsed = float((r_collapsed - r_baseline_same_init).abs().max())

    # --- mask / bucket / batch indexing consistency across rounds ---------
    ctx0 = perturbed.last_center_context0
    ctx1 = perturbed.last_center_context1
    assert ctx0 is not None and ctx1 is not None
    width = 2 * Q_DIM + 1
    count_columns = []
    for bucket in range(zpp.DISTANCE_BUCKETS):
        start = bucket * width + 2 * Q_DIM
        count_columns.append(start)
    # The trailing log-count coordinate of every (centre, bucket) block is a
    # deterministic function of the incidence mask only, so it must be
    # bit-identical between round 1 and round 2.
    count_diff = float(
        (ctx0[:, count_columns] - ctx1[:, count_columns]).abs().max()
    )
    # Centre-context support must be finite everywhere and have exactly the
    # expected width.
    ctx_width_ok = int(ctx0.shape[1]) == zpp.DISTANCE_BUCKETS * width
    ctx_finite = bool(torch.isfinite(ctx0).all() and torch.isfinite(ctx1).all())

    # Reconstruct the per-bucket pair counts from pair_index directly and
    # compare with the trailing log-count block.
    source = perturbed.last_pair_source
    target = perturbed.last_pair_target
    bucket = perturbed.last_pair_bucket
    pair_batch = perturbed.last_pair_batch
    n_centres = int(ctx0.shape[0])
    counts_ok = True
    for b in range(zpp.DISTANCE_BUCKETS):
        mask = bucket == b
        manual = torch.zeros(n_centres)
        if int(mask.sum()) > 0:
            endpoints = torch.cat([source[mask], target[mask]], dim=0)
            manual.index_add_(0, endpoints, torch.ones(endpoints.shape[0]))
        logged = torch.expm1(ctx0[:, count_columns[b]])
        if float((manual - logged).abs().max()) > 1.0e-4:
            counts_ok = False
    # Pair readout counts must match the number of pairs per graph/bucket.
    n_graphs = int(perturbed.last_n_graphs)
    pair_counts = torch.bincount(pair_batch, minlength=n_graphs).to(torch.float64)
    pair_count_ok = int(pair_counts.sum()) == int(source.shape[0])

    index_range_ok = bool(
        int(source.min()) >= 0
        and int(target.min()) >= 0
        and int(source.max()) < int(batch.num_nodes)
        and int(target.max()) < int(batch.num_nodes)
    )
    bucket_range_ok = bool(int(bucket.min()) >= 0 and int(bucket.max()) < zpp.DISTANCE_BUCKETS)

    checks = {
        "param_recurrent_equals_baseline": params_recurrent == params_baseline,
        "param_expected_82115": params_recurrent == EXPECTED_SMALL_TOTAL,
        "head_recurrent_equals_baseline": head_recurrent == head_baseline,
        "head_expected_4135": head_recurrent == EXPECTED_HEAD,
        "state_dict_shapes_identical": shapes_baseline == shapes_recurrent,
        "unified_graph_width_302": int(recurrent.unified_graph_width) == R_DIM,
        "one_round_bit_identical_to_baseline": float(diff_original) == 0.0,
        "init_function_identical_to_baseline": float(diff_init) == 0.0,
        "weight_tying_pair_projection_doubles": (
            counts_recurrent["pair_projection"]
            == 2 * counts_baseline["pair_projection"]
        ),
        "weight_tying_pair_encoder_doubles": (
            counts_recurrent["pair_encoder"] == 2 * counts_baseline["pair_encoder"]
        ),
        "weight_tying_center_update_doubles": (
            counts_recurrent["center_update"] == 2 * counts_baseline["center_update"]
        ),
        "centre_update_moves_h1": center_moves > 1.0e-6,
        "second_round_moves_h2": second_update_moves > 1.0e-6,
        "q1_differs_from_q0": q_drift > 1.0e-6,
        "q1_autograd_depends_on_h1": grad_q1_h1_max > 0.0,
        "zero_update_collapses_to_baseline": float(diff_collapsed) == 0.0,
        "round_counts_match_incidence": bool(counts_ok),
        "round_mask_count_block_identical": count_diff == 0.0,
        "ctx_width_expected": bool(ctx_width_ok),
        "ctx_finite": bool(ctx_finite),
        "pair_count_consistency": bool(pair_count_ok),
        "pair_index_range_ok": bool(index_range_ok),
        "bucket_range_ok": bool(bucket_range_ok),
        "no_nan_perturbed_output": bool(torch.isfinite(out_perturbed).all()),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "parameters": {
            "baseline_total": params_baseline,
            "recurrent_total": params_recurrent,
            "baseline_head": head_baseline,
            "recurrent_head": head_recurrent,
            "delta": params_recurrent - params_baseline,
        },
        "module_call_counts": {
            "baseline": counts_baseline,
            "recurrent": counts_recurrent,
        },
        "mechanism": {
            "center_update_moves_h1_max": center_moves,
            "second_update_moves_h2_max": second_update_moves,
            "q1_q0_max_abs_drift": q_drift,
            "q1_q0_cosine": q_cosine,
            "grad_q1_wrt_h1_max": grad_q1_h1_max,
            "one_round_parent_vs_original_max_diff": diff_original,
            "init_parent_vs_recurrent_max_diff": diff_init,
            "zero_update_vs_baseline_max_diff": diff_collapsed,
            "round_mask_count_block_max_diff": count_diff,
        },
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(),
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"recurrent sanity checks failed: {failed}")
    return payload


def late_refresh_sanity() -> dict[str, Any]:
    """Sanity checks specific to the one-shot ``late_refresh`` variant.

    Verifies (i) parameter neutrality, (ii) both centre updates consume the
    *same* stale aggregate ``A^(0)``, (iii) ``q^(1)`` is genuinely recomputed
    from the final centre state ``h^(2)``, (iv) ``q^(1)`` never re-enters
    centre aggregation, (v) weight tying is unchanged, and (vi) the centre
    path is bit-identical to the ``stale`` depth-only control, so the only
    quantity that changes is the pair readout.
    """
    device = torch.device("cpu")
    _train_data, valid_data, _audit = load_encoded()
    batch = _first_batch(valid_data, device)

    baseline = build_baseline(0).to(device).eval()
    stale = build_stale(0).to(device).eval()
    recurrent = build_recurrent(0).to(device).eval()
    late = build_late_refresh(0).to(device).eval()
    stale.capture_diagnostics = True
    late.capture_diagnostics = True

    params = {
        "baseline": _n_params(baseline),
        "stale": _n_params(stale),
        "recurrent": _n_params(recurrent),
        "late_refresh": _n_params(late),
    }
    shapes_ok = _state_shapes(baseline) == _state_shapes(late)

    # --- deterministic shared perturbation of only the centre update -------
    # All builders share the baseline initialisation, so applying the *same*
    # noise to ``stale`` and ``late_refresh`` makes their centre paths
    # directly comparable.
    shared = {key: value.detach().clone() for key, value in stale.state_dict().items()}
    late.load_state_dict(shared)
    torch.manual_seed(20260912)
    noise = {
        key: 0.01 * torch.randn_like(value)
        for key, value in shared.items()
        if "center_update" in key
    }
    for model in (stale, late):
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if key in noise:
                    value.add_(noise[key])

    # --- run the two controls ---------------------------------------------
    with torch.no_grad():
        stale.encode(batch)
        stale_q1 = stale.last_q1
        stale_h1 = stale.last_h1
        stale_h2 = stale.last_h2
        stale_ctx0 = stale.last_center_context0
        stale_ctx1 = stale.last_center_context1

    late.zero_grad(set_to_none=True)
    late.encode(batch)
    late_h0, late_h1, late_h2 = late.last_h0, late.last_h1, late.last_h2
    late_q0, late_q1 = late.last_q0, late.last_q1
    late_ctx0, late_ctx1 = late.last_center_context0, late.last_center_context1
    assert None not in (
        late_h0,
        late_h1,
        late_h2,
        late_q0,
        late_q1,
        late_ctx0,
        late_ctx1,
        stale_q1,
        stale_h1,
        stale_h2,
        stale_ctx0,
        stale_ctx1,
    )

    # q^(1) must be differentiable w.r.t. the *final* centre state h^(2).
    grad_q1_h2 = torch.autograd.grad(late_q1.sum(), late_h2, retain_graph=True)[0]
    grad_q1_h2_max = float(grad_q1_h2.abs().max())

    # Manual recomputation confirms the stored q^(1) is exactly Q(h^(2)).
    relation = late.relation_encoder(batch.pair_relation)
    gate = 1.0 + torch.tanh(late.distance_gate(batch.pair_bucket))
    with torch.no_grad():
        q1_manual = late._pair_value(
            late_h2, late.last_pair_source, late.last_pair_target, relation, gate
        )
    q1_manual_diff = float((q1_manual - late_q1).abs().max())

    # --- centre aggregation call counts -----------------------------------
    def _pool_calls(model: nn.Module) -> int:
        counter = {"n": 0}
        original = model._pool_pairs_to_centres  # type: ignore[attr-defined]

        def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
            counter["n"] += 1
            return original(*args, **kwargs)

        model._pool_pairs_to_centres = wrapped  # type: ignore[method-assign]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model.encode(batch)
        finally:
            try:
                del model._pool_pairs_to_centres
            except AttributeError:
                pass
            model.train(was_training)
        return int(counter["n"])

    pool_calls_late = _pool_calls(build_late_refresh(0))
    pool_calls_stale = _pool_calls(build_stale(0))
    pool_calls_recurrent = _pool_calls(build_recurrent(0))
    pool_calls_baseline = _pool_calls(build_baseline(0))

    # --- weight tying: every shared module is invoked per round -----------
    counts_late = late.module_call_counts(batch)

    # --- mode switch isolates the readout change --------------------------
    mode_switch = build_late_refresh(0).to(device).eval()
    mode_switch.load_state_dict(shared)
    with torch.no_grad():
        for key, value in mode_switch.state_dict().items():
            if key in noise:
                value.add_(noise[key])
    mode_switch.recurrence_mode = "stale"
    with torch.no_grad():
        r_stale = stale.encode(batch)
        r_as_stale = mode_switch.encode(batch)
    mode_switch.recurrence_mode = "late_refresh"
    with torch.no_grad():
        r_as_late = mode_switch.encode(batch)
    mode_switch_diff = float((r_as_stale - r_stale).abs().max())
    late_output_diff = float((r_as_late - r_stale).abs().max())

    # --- forced-zero update collapses to the baseline function ------------
    collapsed = build_late_refresh(0).to(device).eval()
    collapsed.load_state_dict(shared)
    with torch.no_grad():
        for key, value in collapsed.state_dict().items():
            if key in noise:
                value.add_(noise[key])
    collapsed.force_zero_center_update = True
    with torch.no_grad():
        r_collapsed = collapsed.encode(batch)
        r_baseline = baseline.encode(batch)
    diff_collapsed = float((r_collapsed - r_baseline).abs().max())

    checks = {
        "params_neutral_late_refresh": (
            params["late_refresh"]
            == params["baseline"]
            == params["stale"]
            == params["recurrent"]
            == EXPECTED_SMALL_TOTAL
        ),
        "head_expected_4135_late": _n_params(late.head) == EXPECTED_HEAD,
        "state_dict_shapes_identical_late": bool(shapes_ok),
        "late_refresh_mode_set": late.recurrence_mode == "late_refresh",
        "centre_aggregation_called_once_late": pool_calls_late == 1,
        "centre_aggregation_called_twice_stale": pool_calls_stale == 2,
        "centre_aggregation_called_twice_recurrent": pool_calls_recurrent == 2,
        "centre_aggregation_called_once_baseline": pool_calls_baseline == 1,
        "both_centre_updates_share_same_A0": (
            late_ctx1 is late_ctx0
            and float((late_ctx1 - late_ctx0).abs().max()) == 0.0
        ),
        "centre_path_identical_to_stale_h1": float((late_h1 - stale_h1).abs().max()) == 0.0,
        "centre_path_identical_to_stale_h2": float((late_h2 - stale_h2).abs().max()) == 0.0,
        "centre_context_identical_to_stale": (
            float((late_ctx0 - stale_ctx0).abs().max()) == 0.0
            and float((late_ctx1 - stale_ctx1).abs().max()) == 0.0
        ),
        "q1_recomputed_from_h2": (
            float((late_q1 - late_q0).abs().max()) > 1.0e-6
            and grad_q1_h2_max > 0.0
            and q1_manual_diff == 0.0
        ),
        "q1_never_aggregated_into_centre": late_ctx1 is late_ctx0,
        "weight_tying_pair_projection_2x_per_round_late": (
            counts_late["pair_projection"] == 4
        ),
        "weight_tying_pair_encoder_2x_per_round_late": (
            counts_late["pair_encoder"] == 2
        ),
        "weight_tying_center_update_2x_per_round_late": (
            counts_late["center_update"] == 2
        ),
        "mode_switch_late_to_stale_matches_stale": mode_switch_diff == 0.0,
        "late_output_differs_from_stale": late_output_diff > 0.0,
        "late_q1_differs_from_stale_q1": (
            float((late_q1 - stale_q1).abs().max()) > 1.0e-6
        ),
        "zero_update_collapses_late_to_baseline": diff_collapsed == 0.0,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "parameters": {
            **params,
            "delta_late_minus_baseline": params["late_refresh"] - params["baseline"],
        },
        "mechanism": {
            "late_h1_moves_from_h0_max": float((late_h1 - late_h0).abs().max()),
            "late_h2_moves_from_h1_max": float((late_h2 - late_h1).abs().max()),
            "late_q1_q0_max_abs_drift": float((late_q1 - late_q0).abs().max()),
            "grad_q1_wrt_h2_max": grad_q1_h2_max,
            "q1_manual_recompute_max_diff": q1_manual_diff,
            "late_vs_stale_output_max_diff": late_output_diff,
            "mode_switch_vs_stale_max_diff": mode_switch_diff,
            "zero_update_vs_baseline_max_diff": diff_collapsed,
        },
        "module_call_counts": {"late_refresh": counts_late},
        "centre_aggregation_calls": {
            "baseline": pool_calls_baseline,
            "stale": pool_calls_stale,
            "recurrent": pool_calls_recurrent,
            "late_refresh": pool_calls_late,
        },
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(),
    }
    _write_json(RESULTS_DIR / "sanity_late_refresh.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"late-refresh sanity checks failed: {failed}")
    return payload


def pair_to_pair_sanity() -> dict[str, Any]:
    """Sanity checks for the direct shared-``k`` pair-to-pair composition.

    Confirms (i) exact collapse to the current recurrent model when the
    composition update is disabled/zero-initialised, (ii) ``q*`` genuinely
    depends on the gathered ``q_{ik}`` / ``q_{kj}``, (iii) shared-``k``
    indexing never crosses graph boundaries, (iv) endpoints reconstruct a valid
    ``(i, k, j)`` triple, (v) per-pair counts and shapes are correct, and
    (vi) the added parameter count is small and accounted for.
    """
    device = torch.device("cpu")
    _train_data, valid_data, _audit = load_encoded()
    batch = _first_batch(valid_data, device)

    model = build_pair_to_pair(0).to(device).eval()
    model.capture_diagnostics = True
    baseline = build_baseline(0).to(device).eval()
    recurrent = build_recurrent(0).to(device).eval()

    # --- initialisation collapse: phi is zero-initialised ------------------
    # --- make the shared centre update and the composition active ----------
    torch.manual_seed(20260913)
    with torch.no_grad():
        for parameter in model.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        for parameter in model.pair_to_pair_psi.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        for parameter in model.pair_to_pair_phi.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    # Give the recurrent reference exactly the same (perturbed) shared tensors.
    recurrent.load_state_dict(model.state_dict(), strict=False)

    # --- initialisation identity (fresh models, phi = 0) -------------------
    fresh_pair = build_pair_to_pair(0).to(device).eval()
    fresh_recurrent = build_recurrent(0).to(device).eval()
    with torch.no_grad():
        diff_init = float(
            (fresh_pair.encode(batch) - fresh_recurrent.encode(batch)).abs().max()
        )

    # --- active composition forward ---------------------------------------
    model.zero_grad(set_to_none=True)
    out = model.encode(batch)
    q_raw = model.last_q1_raw
    q_star = model.last_q1
    assert q_raw is not None and q_star is not None
    delta = q_star - q_raw
    grad_delta = torch.autograd.grad(delta.sum(), q_raw, retain_graph=True)[0]
    grad_delta_max = float(grad_delta.abs().max())
    composition_delta_max = float(delta.abs().max())

    # --- forced-zero collapse to the matched recurrent model ----------------
    model.force_zero_pair_to_pair = True
    with torch.no_grad():
        out_zero = model.encode(batch)
        out_rec = recurrent.encode(batch)
    diff_zero = float((out_zero - out_rec).abs().max())
    model.force_zero_pair_to_pair = False

    left = model.last_pair_to_pair_left
    right = model.last_pair_to_pair_right
    target = model.last_pair_to_pair_target
    pair_batch = model.last_pair_batch
    source = model.last_pair_source
    pair_target = model.last_pair_target
    assert (
        left is not None
        and right is not None
        and target is not None
        and pair_batch is not None
        and source is not None
        and pair_target is not None
    )

    # --- graph isolation ---------------------------------------------------
    leak = int((pair_batch[target] != pair_batch[left]).sum()) + int(
        (pair_batch[target] != pair_batch[right]).sum()
    )

    # --- endpoint consistency: left = {i,k}, right = {k,j}, target = {i,j} --
    a = source[target]
    b = pair_target[target]
    ls, lt = source[left], pair_target[left]
    rs, rt = source[right], pair_target[right]
    left_has_a = (ls == a) | (lt == a)
    right_has_b = (rs == b) | (rt == b)
    left_other = torch.where(ls == a, lt, ls)
    right_other = torch.where(rs == b, rt, rs)
    endpoint_ok = bool(
        (
            left_has_a
            & right_has_b
            & (left_other == right_other)
            & (left_other != a)
            & (left_other != b)
        ).all()
    )

    # --- per-pair composition counts must equal n_graph - 2 ----------------
    node_counts = torch.bincount(batch.batch)
    expected_counts = (node_counts[pair_batch] - 2).to(q_raw.dtype)
    actual_counts = torch.zeros(q_raw.shape[0], device=device, dtype=q_raw.dtype)
    actual_counts.index_add_(
        0, target, torch.ones(target.shape[0], device=device, dtype=q_raw.dtype)
    )
    count_ok = bool(
        torch.allclose(actual_counts, expected_counts.clamp_min(0.0), atol=1e-5)
    )

    # --- parameter accounting ---------------------------------------------
    psi_params = _n_params(model.pair_to_pair_psi)
    phi_params = _n_params(model.pair_to_pair_phi)
    added = psi_params + phi_params
    total = _n_params(model)
    baseline_total = _n_params(baseline)
    recurrent_total = _n_params(recurrent)
    added_reported = pair_to_pair_param_count(0)
    psi_first = model.pair_to_pair_psi[0]
    psi_last = model.pair_to_pair_psi[-1]

    checks = {
        "params_pair_to_pair_expected": (
            total == baseline_total + added
            and total == recurrent_total + added
            and added == added_reported
        ),
        "added_params_in_2k_5k": 2000 <= added <= 5000,
        "total_params_below_90k": total < 90000,
        "head_expected_4135": _n_params(model.head) == EXPECTED_HEAD,
        "psi_phi_present": model.pair_to_pair_psi is not None
        and model.pair_to_pair_phi is not None,
        "psi_input_width_3q": int(psi_first.in_features) == 3 * Q_DIM,
        "psi_output_width_q": int(psi_last.out_features) == Q_DIM,
        "init_collapses_to_recurrent": diff_init == 0.0,
        "zero_flag_collapses_to_recurrent": diff_zero == 0.0,
        "composition_delta_nonzero": composition_delta_max > 1.0e-6,
        "q_star_depends_on_gathered_q": grad_delta_max > 0.0,
        "no_cross_graph_leak": leak == 0,
        "triple_endpoints_consistent": endpoint_ok,
        "pair_counts_match_n_minus_2": count_ok,
        "residual_shape_correct": tuple(q_star.shape) == tuple(q_raw.shape)
        and int(q_raw.shape[1]) == Q_DIM,
        "no_nan_inf_forward": bool(torch.isfinite(out).all())
        and bool(torch.isfinite(q_star).all()),
        "recurrent_path_unchanged": recurrent_total == EXPECTED_SMALL_TOTAL,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "parameters": {
            "baseline_total": baseline_total,
            "recurrent_total": recurrent_total,
            "pair_to_pair_total": total,
            "added": added,
            "psi": psi_params,
            "phi": phi_params,
        },
        "mechanism": {
            "composition_delta_max": composition_delta_max,
            "grad_delta_wrt_gathered_q_max": grad_delta_max,
            "n_triples": int(target.shape[0]),
            "n_pairs": int(q_raw.shape[0]),
            "triples_per_pair": float(target.shape[0]) / max(q_raw.shape[0], 1),
            "init_pair_vs_recurrent_max_diff": diff_init,
            "zero_flag_vs_recurrent_max_diff": diff_zero,
            "cross_graph_leak": leak,
        },
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(),
    }
    _write_json(RESULTS_DIR / "sanity_pair_to_pair.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"pair-to-pair sanity checks failed: {failed}")
    return payload


def smoke() -> dict[str, Any]:
    """Cheap 3-epoch training on a subset to confirm there is no obvious bug."""
    train_data, valid_data, _ = load_encoded()
    train_subset = list(train_data)[:1000]
    valid_subset = list(valid_data)[:256]
    protocol = {"max_epochs": 3, "patience": 3}
    baseline = train(
        build_baseline,
        train_subset,
        valid_subset,
        seed=0,
        tag="smoke_baseline",
        real_batch_identity=True,
        protocol_override=protocol,
    )
    recurrent = train(
        build_recurrent,
        train_subset,
        valid_subset,
        seed=0,
        tag="smoke_recurrent",
        real_batch_identity=True,
        protocol_override=protocol,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "subset": {"train": len(train_subset), "valid": len(valid_subset)},
        "baseline": {
            "best_valid_mae": float(baseline["best_valid_mae"]),
            "parameters": int(baseline["parameters"]),
            "train_loss_at_best": float(baseline["train_loss_at_best"]),
            "epochs_run": int(baseline["epochs_run"]),
            "finite": bool(
                np.isfinite(baseline["best_valid_mae"])
                and np.isfinite(baseline["train_loss_at_best"])
            ),
        },
        "recurrent": {
            "best_valid_mae": float(recurrent["best_valid_mae"]),
            "parameters": int(recurrent["parameters"]),
            "train_loss_at_best": float(recurrent["train_loss_at_best"]),
            "epochs_run": int(recurrent["epochs_run"]),
            "finite": bool(
                np.isfinite(recurrent["best_valid_mae"])
                and np.isfinite(recurrent["train_loss_at_best"])
            ),
        },
        "official_test_loaded": False,
    }
    payload["passed"] = bool(
        payload["baseline"]["finite"] and payload["recurrent"]["finite"]
    )
    _write_json(RESULTS_DIR / "smoke.json", payload)
    if not payload["passed"]:
        raise RuntimeError("smoke training produced non-finite losses")
    return payload


# ---------------------------------------------------------------------------
# full matched runs
# ---------------------------------------------------------------------------


def _read_run(tag: str, seed: int) -> dict[str, Any]:
    return _read_json(RUNS_DIR / f"{tag}_seed{seed}.json")


def run_baseline(seed: int = 0) -> dict[str, Any]:
    train_data, valid_data, _ = load_encoded()
    summary = train(
        build_baseline, train_data, valid_data, seed=int(seed), tag="baseline"
    )
    _write_json(RESULTS_DIR / f"baseline_seed{seed}.json", summary)
    return summary


def run_recurrent(seed: int = 0) -> dict[str, Any]:
    train_data, valid_data, _ = load_encoded()
    summary = train(
        build_recurrent, train_data, valid_data, seed=int(seed), tag="recurrent"
    )
    _write_json(RESULTS_DIR / f"recurrent_seed{seed}.json", summary)
    return summary


def run_stale(seed: int = 0) -> dict[str, Any]:
    """Depth-only control: 2 centre updates with a stale relation q^(0)."""
    train_data, valid_data, _ = load_encoded()
    summary = train(
        build_stale, train_data, valid_data, seed=int(seed), tag="stale"
    )
    _write_json(RESULTS_DIR / f"stale_seed{seed}.json", summary)
    return summary


def run_late_refresh(seed: int = 0) -> dict[str, Any]:
    """One-shot late relation refresh: stale centres + refreshed pair readout."""
    train_data, valid_data, _ = load_encoded()
    summary = train(
        build_late_refresh,
        train_data,
        valid_data,
        seed=int(seed),
        tag="late_refresh",
    )
    _write_json(RESULTS_DIR / f"late_refresh_seed{seed}.json", summary)
    return summary


def pair_to_pair_param_count(seed: int = 0, *, pair_to_pair_hidden: int = 24) -> int:
    """Added trainable parameters of the pair-to-pair composition MLPs."""
    model = build_pair_to_pair(seed, pair_to_pair_hidden=pair_to_pair_hidden)
    return _n_params(model) - EXPECTED_SMALL_TOTAL


def run_pair_to_pair(
    seed: int = 0, *, pair_to_pair_hidden: int = 24
) -> dict[str, Any]:
    """Train the recurrent T=2 model + direct shared-k pair-to-pair composition."""
    train_data, valid_data, _ = load_encoded()
    added = pair_to_pair_param_count(seed, pair_to_pair_hidden=pair_to_pair_hidden)
    summary = train(
        lambda s: build_pair_to_pair(s, pair_to_pair_hidden=pair_to_pair_hidden),
        train_data,
        valid_data,
        seed=int(seed),
        tag="pair_to_pair",
        expected_total=EXPECTED_SMALL_TOTAL + int(added),
    )
    _write_json(RESULTS_DIR / f"pair_to_pair_seed{seed}.json", summary)
    return summary


def decision_pair_to_pair(seed: int = 0) -> dict[str, Any]:
    """Compare pair-to-pair against the matched recurrent reference."""
    recurrent = _read_json(RESULTS_DIR / f"recurrent_seed{seed}.json")
    candidate = _read_json(RESULTS_DIR / f"pair_to_pair_seed{seed}.json")
    r = float(recurrent["best_valid_mae"])
    c = float(candidate["best_valid_mae"])
    delta = r - c  # positive => pair-to-pair improves over recurrent
    if delta >= 0.003:
        verdict = "STRONG_POSITIVE"
        next_step = "run_seed1"
    elif delta >= 0.001:
        verdict = "WEAK_POSITIVE"
        next_step = "run_seed1_to_check_replication"
    elif delta > -0.001:
        verdict = "NO_INDEPENDENT_SIGNAL"
        next_step = "stop_no_sweep"
    else:
        verdict = "DEGRADED"
        next_step = "stop_operator"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "recurrent_valid_mae": r,
        "pair_to_pair_valid_mae": c,
        "delta_recurrent_minus_pair_to_pair": delta,
        "strong_gate": 0.003,
        "weak_gate": 0.001,
        "recurrent_best_epoch": int(recurrent["best_epoch"]),
        "pair_to_pair_best_epoch": int(candidate["best_epoch"]),
        "recurrent_params": int(recurrent["parameters"]),
        "pair_to_pair_params": int(candidate["parameters"]),
        "verdict": verdict,
        "next_step": next_step,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"decision_pair_to_pair_seed{seed}.json", payload)
    return payload


def decision_pair_to_pair_replication() -> dict[str, Any]:
    """Two-seed replication verdict for the direct pair-to-pair operator."""
    seed0 = _read_json(RESULTS_DIR / "decision_pair_to_pair_seed0.json")
    seed1 = _read_json(RESULTS_DIR / "decision_pair_to_pair_seed1.json")
    d0 = float(seed0["delta_recurrent_minus_pair_to_pair"])
    d1 = float(seed1["delta_recurrent_minus_pair_to_pair"])
    mean = 0.5 * (d0 + d1)
    if d0 >= 0.003 and d1 >= 0.003 and mean >= 0.003:
        verdict = "REPLICATED_STRONG"
        next_step = "keep_operator"
    elif mean >= 0.001 and d0 > 0.0 and d1 > 0.0:
        verdict = "WEAK_DIRECTION_CONSISTENT"
        next_step = "optional_seed2_not_required"
    elif abs(mean) <= 0.001 or (d0 * d1 < 0.0):
        verdict = "NO_INDEPENDENT_SIGNAL"
        next_step = "stop_no_sweep"
    else:
        verdict = "DEGRADED"
        next_step = "stop_operator"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed0_delta": d0,
        "seed1_delta": d1,
        "mean_delta": float(mean),
        "strong_gate": 0.003,
        "weak_gate": 0.001,
        "verdict": verdict,
        "next_step": next_step,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision_pair_to_pair_replication.json", payload)
    return payload


def decision_seed0() -> dict[str, Any]:
    baseline = _read_json(RESULTS_DIR / "baseline_seed0.json")
    recurrent = _read_json(RESULTS_DIR / "recurrent_seed0.json")
    b = float(baseline["best_valid_mae"])
    r = float(recurrent["best_valid_mae"])
    delta = b - r
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "baseline_valid_mae": b,
        "recurrent_valid_mae": r,
        "delta_baseline_minus_recurrent": delta,
        "gate": ARCH_GATE,
        "passes_gate": bool(delta >= ARCH_GATE),
        "baseline_best_epoch": int(baseline["best_epoch"]),
        "recurrent_best_epoch": int(recurrent["best_epoch"]),
        "baseline_params": int(baseline["parameters"]),
        "recurrent_params": int(recurrent["parameters"]),
        "both_horizon_ok": bool(
            not baseline["horizon_boundary_warning"]
            and not recurrent["horizon_boundary_warning"]
        ),
        "p2_one_shot_refresh_context_delta": P2_ONE_SHOT_REFRESH_DELTA,
        "official_test_loaded": False,
    }
    if payload["passes_gate"]:
        payload["verdict"] = "RECURRENCE_SIGNAL"
        payload["next"] = "buy seed1 replication"
    elif delta > 0:
        payload["verdict"] = "SUB_THRESHOLD_POSITIVE"
        payload["next"] = "do not spend multi-seed budget; treat as no evidence"
    else:
        payload["verdict"] = "NO_GO"
        payload["next"] = "recurrence hypothesis not supported"
    _write_json(RESULTS_DIR / "decision_seed0.json", payload)
    return payload


def decision_seed1() -> dict[str, Any]:
    baseline = _read_json(RESULTS_DIR / "baseline_seed1.json")
    recurrent = _read_json(RESULTS_DIR / "recurrent_seed1.json")
    s0 = _read_json(RESULTS_DIR / "decision_seed0.json")
    b1 = float(baseline["best_valid_mae"])
    r1 = float(recurrent["best_valid_mae"])
    d1 = b1 - r1
    d0 = float(s0["delta_baseline_minus_recurrent"])
    mean = 0.5 * (d0 + d1)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed1": {
            "baseline_valid_mae": b1,
            "recurrent_valid_mae": r1,
            "delta": d1,
            "gate": ARCH_GATE_SEED1,
            "passes": bool(d1 >= ARCH_GATE_SEED1),
        },
        "seed0_delta": d0,
        "mean_delta": float(mean),
        "mean_gate": ARCH_MEAN_GATE,
        "mean_passes": bool(mean >= ARCH_MEAN_GATE),
        "verdict": (
            "RECURRENCE_REPLICATED"
            if (d0 >= ARCH_GATE and d1 >= ARCH_GATE_SEED1 and mean >= ARCH_MEAN_GATE)
            else "NOT_REPLICATED"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision_seed1.json", payload)
    return payload


# ---------------------------------------------------------------------------
# depth-only control decision
# ---------------------------------------------------------------------------


def decision_control(seed: int = 0) -> dict[str, Any]:
    """Is the gain from the refreshed relation or from the extra centre depth?

    ``stale`` has the same parameter count and the same two centre updates as
    ``recurrent``; only round 2 reuses ``q^(0)`` instead of recomputing
    ``q^(1)``.  If ``refresh`` beats ``stale``, the relation persistence (not
    mere depth) is doing the work.
    """
    baseline = _read_json(RESULTS_DIR / f"baseline_seed{seed}.json")
    stale = _read_json(RESULTS_DIR / f"stale_seed{seed}.json")
    recurrent = _read_json(RESULTS_DIR / f"recurrent_seed{seed}.json")
    b = float(baseline["best_valid_mae"])
    s = float(stale["best_valid_mae"])
    r = float(recurrent["best_valid_mae"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "baseline_valid_mae": b,
        "stale_valid_mae": s,
        "recurrent_valid_mae": r,
        "delta_baseline_minus_stale": b - s,
        "delta_baseline_minus_recurrent": b - r,
        "delta_stale_minus_recurrent": s - r,
        "stale_params": int(stale["parameters"]),
        "recurrent_params": int(recurrent["parameters"]),
        "stale_best_epoch": int(stale["best_epoch"]),
        "recurrent_best_epoch": int(recurrent["best_epoch"]),
        "refresh_beats_depth": bool(r < s),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"decision_control_seed{seed}.json", payload)
    return payload


def decision_late(seed: int = 0) -> dict[str, Any]:
    """Four-way seed0 comparison: baseline / stale / late-refresh / recurrent.

    ``late_refresh`` keeps the exact stale centre path and only refreshes the
    final pair representation.  Comparing it against ``recurrent`` therefore
    isolates whether feeding the refreshed relation back into the centre adds
    independent value beyond having a better final pair state.
    """
    baseline = _read_json(RESULTS_DIR / f"baseline_seed{seed}.json")
    stale = _read_json(RESULTS_DIR / f"stale_seed{seed}.json")
    late = _read_json(RESULTS_DIR / f"late_refresh_seed{seed}.json")
    recurrent = _read_json(RESULTS_DIR / f"recurrent_seed{seed}.json")
    b = float(baseline["best_valid_mae"])
    s = float(stale["best_valid_mae"])
    lo = float(late["best_valid_mae"])
    r = float(recurrent["best_valid_mae"])
    late_minus_stale = s - lo  # positive => late-refresh better than stale
    recurrent_minus_late = lo - r  # positive => recurrent better than late-refresh
    noise_band = 0.002
    if late_minus_stale <= 0.001 and recurrent_minus_late <= noise_band:
        verdict = "LATE_REFRESH_NOT_SEPARABLE_FROM_STALE"
        implication = (
            "relation-refresh benefit appears to depend on the full second-round "
            "feedback loop rather than a better final pair representation alone"
        )
        next_step = "cannot_distinguish_without_further_controls"
    elif recurrent_minus_late <= noise_band:
        verdict = "LATE_REFRESH_MATCHES_RECURRENT"
        implication = (
            "refreshed relation acts mainly as a better final representation; "
            "feedback into the centre is not required"
        )
        next_step = "centre_refinement_depth"
    else:
        verdict = "RECURRENT_LEADS_LATE_REFRESH"
        implication = (
            "refreshed information re-entering the centre likely has independent "
            "value"
        )
        next_step = "persistent_relation_or_pair_to_pair"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "baseline_valid_mae": b,
        "stale_valid_mae": s,
        "late_refresh_valid_mae": lo,
        "recurrent_valid_mae": r,
        "late_refresh_params": int(late["parameters"]),
        "recurrent_params": int(recurrent["parameters"]),
        "stale_params": int(stale["parameters"]),
        "baseline_params": int(baseline["parameters"]),
        "late_refresh_best_epoch": int(late["best_epoch"]),
        "stale_best_epoch": int(stale["best_epoch"]),
        "recurrent_best_epoch": int(recurrent["best_epoch"]),
        "baseline_best_epoch": int(baseline["best_epoch"]),
        "delta_baseline_minus_stale": b - s,
        "delta_baseline_minus_late_refresh": b - lo,
        "delta_baseline_minus_recurrent": b - r,
        "delta_stale_minus_late_refresh": late_minus_stale,
        "delta_late_refresh_minus_recurrent": recurrent_minus_late,
        "noise_band": noise_band,
        "late_refresh_beats_stale": bool(lo < s),
        "recurrent_beats_late_refresh": bool(r < lo),
        "verdict": verdict,
        "implication": implication,
        "next_step": next_step,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"decision_late_refresh_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# terminal official-test evaluation (authorised only after a positive verdict)
# ---------------------------------------------------------------------------


def _terminal_authorized() -> bool:
    seed1 = RESULTS_DIR / "decision_seed1.json"
    if seed1.exists():
        if _read_json(seed1).get("verdict") == "RECURRENCE_REPLICATED":
            return True
    seed0 = RESULTS_DIR / "decision_seed0.json"
    if seed0.exists():
        return bool(_read_json(seed0).get("passes_gate"))
    return False


def terminal_test() -> dict[str, Any]:
    """Frozen one-shot official-test evaluation of the selection checkpoints.

    This is the repository's canonical terminal test protocol: transforms are
    fit on official train only (identical to the selection encoding) and each
    selected checkpoint is evaluated once on the official test split.  No test
    label is ever used for any decision.  The stage refuses to run unless a
    positive validation verdict was already written.
    """
    if not _terminal_authorized():
        raise RuntimeError(
            "refusing official test: no positive recurrence verdict on validation"
        )
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "authorised_by": [
            path.name
            for path in (
                RESULTS_DIR / "decision_seed0.json",
                RESULTS_DIR / "decision_seed1.json",
            )
            if path.exists()
        ],
        "encoding": "fit on official train only (identical to selection)",
        "seeds": [0, 1],
        "checkpoints": "selection states (best official-valid MAE)",
        "official_test_loaded": True,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "terminal_test_lock.json", lock)

    train_records, _valid_records = ztraining.load_train_valid_records()
    test_records = ztraining.extract_test_records()
    config = ztraining.base_config()
    _train_data, test_data, audit = ztraining.build_encoded(
        train_records, test_records, config
    )
    del _train_data
    loader = zpp._make_loader(test_data, 128, False, 0)
    targets = np.asarray(
        [float(graph.y.view(-1)[0]) for graph in test_data], dtype=np.float64
    )
    rows: list[dict[str, Any]] = []
    for seed in (0, 1):
        for tag, builder in (
            ("baseline", build_baseline),
            ("recurrent", build_recurrent),
        ):
            state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
            if not state_path.exists():
                continue
            model = builder(int(seed)).eval()
            model.load_state_dict(
                torch.load(state_path, map_location="cpu", weights_only=True)
            )
            predictions: list[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    predictions.append(model(batch).view(-1).cpu().numpy())
            prediction = np.concatenate(predictions).astype(np.float64)
            rows.append(
                {
                    "seed": int(seed),
                    "tag": str(tag),
                    "test_mae": float(mean_absolute_error(targets, prediction)),
                    "n_test": int(targets.shape[0]),
                }
            )
    by_key = {(row["seed"], row["tag"]): row["test_mae"] for row in rows}
    deltas = {
        int(seed): by_key[(seed, "baseline")] - by_key[(seed, "recurrent")]
        for seed in (0, 1)
        if (seed, "baseline") in by_key and (seed, "recurrent") in by_key
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "rows": rows,
        "delta_baseline_minus_recurrent": deltas,
        "mean_delta": float(np.mean(list(deltas.values()))) if deltas else None,
        "baseline_test_mean": float(
            np.mean([by_key[(s, "baseline")] for s in (0, 1) if (s, "baseline") in by_key])
        )
        if any((s, "baseline") in by_key for s in (0, 1))
        else None,
        "recurrent_test_mean": float(
            np.mean([by_key[(s, "recurrent")] for s in (0, 1) if (s, "recurrent") in by_key])
        )
        if any((s, "recurrent") in by_key for s in (0, 1))
        else None,
        "audit": {key: audit[key] for key in sorted(audit) if key != "topology"},
        "official_test_loaded": True,
    }
    _write_json(RESULTS_DIR / "terminal_test.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "sanity",
            "sanity_late",
            "sanity_pair",
            "smoke",
            "baseline",
            "recurrent",
            "stale",
            "late_refresh",
            "pair_to_pair",
            "decision",
            "control",
            "control_late",
            "control_pair",
            "control_pair_both",
            "recurrent_seed1",
            "terminal_test",
            "final",
            "all",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    stage = args.stage
    torch.set_num_threads(4)
    if stage in {"sanity", "all"}:
        result = sanity()
        print(json.dumps(result["checks"], indent=2), flush=True)
    if stage == "sanity_late":
        result = late_refresh_sanity()
        print(json.dumps(result["checks"], indent=2), flush=True)
    if stage == "sanity_pair":
        result = pair_to_pair_sanity()
        print(json.dumps(result["checks"], indent=2), flush=True)
    if stage in {"smoke", "all"}:
        result = smoke()
        print(json.dumps(result, indent=2), flush=True)
    if stage in {"baseline", "all"}:
        run_baseline(args.seed)
    if stage in {"recurrent", "all"}:
        run_recurrent(args.seed)
    if stage == "stale":
        run_stale(seed=args.seed if args.seed != 0 else 0)
    if stage == "late_refresh":
        run_late_refresh(seed=args.seed if args.seed != 0 else 0)
    if stage == "pair_to_pair":
        run_pair_to_pair(seed=args.seed)
    if stage == "control":
        result = decision_control(int(args.seed))
        print(json.dumps(result, indent=2), flush=True)
    if stage == "control_late":
        result = decision_late(int(args.seed))
        print(json.dumps(result, indent=2), flush=True)
    if stage == "control_pair":
        result = decision_pair_to_pair(int(args.seed))
        print(json.dumps(result, indent=2), flush=True)
    if stage == "control_pair_both":
        result = decision_pair_to_pair_replication()
        print(json.dumps(result, indent=2), flush=True)
    if stage in {"decision", "all"}:
        result = decision_seed0()
        print(json.dumps(result, indent=2), flush=True)
    if stage == "recurrent_seed1":
        run_baseline(1)
        run_recurrent(1)
        result = decision_seed1()
        print(json.dumps(result, indent=2), flush=True)
    if stage == "terminal_test":
        terminal_test()
    if stage in {"decision", "final", "all"}:
        print("official test is loaded only by the explicit terminal_test stage", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
