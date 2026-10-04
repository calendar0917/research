"""ZINC pooling-scale / count-accessibility round (seed 0): arms C and N.

Round: ``zinc_pooling_scale_count_seed0_v1``.

Architecture question (frozen before any formal trajectory): with the same
information, same parameter count and same training recipe, does replacing the
unary/pair *sum* moments by *count-normalised mean* moments improve the
chemical component ``g = y - c``?  Secondary question: does merely making the
existing pooling count coordinates visible already help?

Exactly two formal seed-0 trajectories are trained, both on the fresh
8000/2000 fold and the fresh-fit-only targets/prep/payload of
``zinc_local_tuple_fresh_fold_replication_seed0_v1`` (source commit
``a5400de``; pre-registration ``2e4ee3f``):

* ``C`` — sum/count control: unary/pair blocks ``[sum z, sum z^2, log1p(n)]``
  (the existing pooling with the C6 count zeroing removed; global
  atom/bond histogram and relation ``path_count`` stay masked exactly as C6).
* ``N`` — mean/count candidate: ``[sum z / max(n,1), sum z^2 / max(n,1),
  log1p(n)]``; every other line identical to C.

The latest fresh-fold ``M`` soup is read-only reference (never retrained).

Usage::

    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_pooling_scale_count_seed0_v1 --prepare
    ... --pre-checks              (local CPU)
    ... --smoke --arm C --device cpu
    ... --train --arm C --device cuda --out <dir>
    ... --analyze
    ... --budget --manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_vs_mlp_seed0_v1 as mlpmod,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_fresh_fold_replication_seed0_v1 as fresh,
)
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw

PROTOCOL_VERSION = "zinc-pooling-scale-count-seed0-v1"
RESULT_SLUG = "zinc_pooling_scale_count_seed0_v1"
RESULTS_DIR = Path("tracks/ksvd/results") / RESULT_SLUG
SOURCE_DIR = fresh.RESULTS_DIR

# ---- frozen recipe (identical for C and N, copied from the source round) ---
SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
TRAIN_SHUFFLE_OFFSET = 101
LOG_EPOCHS = (1, 40, 120, 240)
N_FIT = 8000
N_DEV = 2000

#: frozen source constants (witness targets)
FROZEN_SCHEDULE_SHA = fresh.FROZEN_SCHEDULE_SHA
FROZEN_GID_STREAM_SHA = "69187f13fba5def82ee2f28765c1c46ac4129844d3f76ac769e018d18ae097d2"
FROZEN_TRAIN_RNG_SHA = "1ccf17250133dec5e96478acd63e13d728d0dbbabb30389415a302c6f872f81f"
FROZEN_BUILD_RNG_SHA = "a2e8a8ab56091d98d239b81fd15117cb818f1b88d95d2fc1bcff036eee5a8853"
SOURCE_M_INIT_STATE_HASH = "e1ed793ae85b632f31525aa3d67334b6caf40fce9dc95c4294ab9e199829cad2"
NEW_FIT_SHA = fresh.NEW_FIT_SHA
NEW_DEV_SHA = fresh.NEW_DEV_SHA
EXPECTED_PARAMETERS = 297499
REPLAY_TOL = 1.0e-5

#: statistics / gates (frozen before the two formal runs)
BOOT_SEED = 20261006
N_BOOT = 1000
PERF_DELTA = 0.003
GROUP_NAMES = ("k=0", "k=-1", "k=-2", "k<=-3")
ARMS = ("C", "N")
POOL_MODES = {"C": "sum", "N": "mean"}

#: runtime pooling widths (asserted against real forwards; the audit-module
#: ``UNARY_BLOCKS``/``PAIR_BLOCKS`` constants describe the v0 layout, not this
#: 144/48-wide M bridge, so the round derives the six count coordinates at
#: runtime and freezes the width witness here).
UNARY_MOMENT_DIM = 144
PAIR_MOMENT_DIM = 48
UNARY_DIM_CN = 2 * UNARY_MOMENT_DIM + 1          # 289
PAIR_BLOCK_DIM_CN = 2 * PAIR_MOMENT_DIM + 1      # 97
RELATION_DIM_CN = int(p2.DISTANCE_BUCKETS) * PAIR_BLOCK_DIM_CN  # 485
GRAPH_HIDDEN_DIM = 32
TOPOLOGY_DIM = 8
READER_DIM_CN = UNARY_DIM_CN + RELATION_DIM_CN + GRAPH_HIDDEN_DIM + TOPOLOGY_DIM  # 814
UNARY_COUNT_INDEX = 2 * UNARY_MOMENT_DIM          # 288
PAIR_COUNT_INDEX_IN_BLOCK = 2 * PAIR_MOMENT_DIM   # 96

#: the round mask: exactly C6 minus the unary/pair count zeroing.
#: global atom/bond histogram and relation path_count remain masked.
ROUND_MASK = audit.AuditMask(
    global_zero_groups=("atom_histogram", "bond_histogram"),
    relation_zero_groups=("path_count",),
)
if ROUND_MASK.unary_zero_blocks or ROUND_MASK.pair_zero_blocks:
    raise RuntimeError("round mask must re-enable unary/pair count coordinates")

#: fixed tolerances for the pooling reference checks
POOL_RTOL = 1.0e-4
POOL_ATOL = 1.0e-5

jsonable = prev.jsonable
write_json = prev.write_json
file_sha256 = prev.file_sha256
state_hash = prev.state_hash
seed_everything = prev.seed_everything
array_sha256 = prev._array_sha256
group_masks = zw.group_masks


# ---------------------------------------------------------------------------
# 1. pooling: sum/count (C) and mean/count (N)
# ---------------------------------------------------------------------------


def pool_moments_scaled(
    value: torch.Tensor,
    batch: torch.Tensor,
    n_graphs: int,
    mode: str = "sum",
) -> torch.Tensor:
    """Unary pool ``[moment z, mean-square z, log1p(n)]`` per graph.

    ``mode == "sum"`` is bit-identical to
    ``audit.pool_moments_masked(..., zero_blocks=())`` (asserted by
    :func:`pool_reference_checks`).  ``mode == "mean"`` divides the first two
    blocks by the actual number of rows entering the pool, ``max(n, 1)``.
    There is no post-hoc standardisation, LayerNorm or extra branch.
    """
    if mode not in ("sum", "mean"):
        raise ValueError(mode)
    counts = torch.bincount(batch.to(torch.long), minlength=int(n_graphs)).to(value.dtype).unsqueeze(1)
    total = torch.zeros((int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype)
    squared = torch.zeros_like(total)
    total.index_add_(0, batch.to(torch.long), value)
    squared.index_add_(0, batch.to(torch.long), value * value)
    if mode == "mean":
        denom = counts.clamp_min(1.0)
        total = total / denom
        squared = squared / denom
    return torch.cat([total, squared, torch.log1p(counts)], dim=1)


def pool_pair_moments_scaled(
    value: torch.Tensor,
    pair_batch: torch.Tensor,
    pair_bucket: torch.Tensor,
    n_graphs: int,
    mode: str = "sum",
) -> torch.Tensor:
    """Per-distance-bucket pair pool ``[moment z, mean-square z, log1p(n)]``.

    Empty buckets output two zero moments and ``count = 0`` (denominator
    ``max(n, 1)``), never NaN.
    """
    if mode not in ("sum", "mean"):
        raise ValueError(mode)
    pair_batch = pair_batch.to(torch.long)
    blocks: list[torch.Tensor] = []
    for bucket in range(int(p2.DISTANCE_BUCKETS)):
        mask = pair_bucket == int(bucket)
        current = value[mask]
        current_batch = pair_batch[mask]
        total = torch.zeros((int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype)
        squared = torch.zeros_like(total)
        counts = torch.zeros((int(n_graphs), 1), device=value.device, dtype=value.dtype)
        if current.numel():
            total.index_add_(0, current_batch, current)
            squared.index_add_(0, current_batch, current * current)
            counts.index_add_(
                0,
                current_batch,
                torch.ones((current_batch.shape[0], 1), device=value.device, dtype=value.dtype),
            )
        if mode == "mean":
            denom = counts.clamp_min(1.0)
            total = total / denom
            squared = squared / denom
        blocks.append(torch.cat([total, squared, torch.log1p(counts)], dim=1))
    return torch.cat(blocks, dim=1)


# ---------------------------------------------------------------------------
# 2. independent (float64, per-graph) reference for the pooling checks
# ---------------------------------------------------------------------------


def _reference_blocks(vals: torch.Tensor, mode: str) -> torch.Tensor:
    n = int(vals.shape[0])
    if n:
        total = vals.sum(0)
        squared = (vals * vals).sum(0)
    else:
        total = torch.zeros(vals.shape[1], dtype=vals.dtype)
        squared = torch.zeros(vals.shape[1], dtype=vals.dtype)
    if mode == "mean":
        total = total / max(n, 1)
        squared = squared / max(n, 1)
    count = torch.log1p(torch.tensor(float(n), dtype=vals.dtype)).reshape(1)
    return torch.cat([total, squared, count])


def reference_pool_unary(
    value: torch.Tensor, batch: torch.Tensor, n_graphs: int, mode: str
) -> torch.Tensor:
    v = value.double()
    rows = []
    for g in range(int(n_graphs)):
        rows.append(_reference_blocks(v[batch == g], mode))
    return torch.stack(rows)


def reference_pool_pair(
    value: torch.Tensor,
    pair_batch: torch.Tensor,
    pair_bucket: torch.Tensor,
    n_graphs: int,
    mode: str,
) -> torch.Tensor:
    v = value.double()
    blocks = []
    for bucket in range(int(p2.DISTANCE_BUCKETS)):
        rows = []
        for g in range(int(n_graphs)):
            rows.append(_reference_blocks(v[(pair_bucket == bucket) & (pair_batch == g)], mode))
        blocks.append(torch.stack(rows))
    return torch.cat(blocks, dim=1)


# ---------------------------------------------------------------------------
# 3. model: same skeleton/constructor as source M, pooling scale switched by
#    one instance attribute; the forward body mirrors audit.AuditModel.forward
#    line by line, only the two pooling calls are routed to our functions.
# ---------------------------------------------------------------------------


class ScaledLocalTupleFull(prev.LocalTupleFull):
    """``LocalTupleFull`` with a ``pool_mode`` switch on the pooling moments.

    Construction is identical to ``prev.LocalTupleFull``; ``pool_mode`` is a
    plain attribute (not a buffer / parameter), so the state dict is exactly
    the source M state dict and construction consumes no extra RNG.
    """

    def __init__(self, payload: prev.TuplePayload, weight_key: str, pool_mode: str) -> None:
        super().__init__(payload, weight_key)
        if pool_mode not in ("sum", "mean"):
            raise ValueError(pool_mode)
        self.pool_mode = pool_mode

    def forward(
        self,
        data: Any,
        *,
        mask: audit.AuditMask | None = None,
        fill: Mapping[str, torch.Tensor] | None = None,
        readout_perm: tuple[Sequence[str], torch.Tensor] | None = None,
        return_aux: bool = False,
        **_kwargs: Any,
    ):
        if mask is None:
            return super().forward(data, return_aux=return_aux)

        coord = self.code(data.dict_phi)
        if mask.coord_zero:
            coord = torch.zeros_like(coord)
        E = self.environments_masked(coord, data, mask, fill)

        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = pool_moments_scaled(E, batch, n_graphs, self.pool_mode)

        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        if mask.pair_projection_zero:
            u = torch.zeros_like(u)
        left = u[source]
        right = u[target]
        relation_input = data.pair_relation[:, list(audit.p1.P1_RELATION_INDICES)]
        if mask.relation_zero_groups:
            relation_input = audit._replace_grouped_columns(
                relation_input, audit.RELATION_GROUPS, mask.relation_zero_groups, fill, "relation:"
            )
        relation = self.relation_encoder(relation_input)
        product = left * right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        if mask.gate_off:
            gate = torch.ones_like(gate)
        pair_input = torch.cat([left + right, torch.abs(left - right), product * gate, relation], dim=1)
        pair_value = self.pair_encoder(pair_input)
        pair_batch = batch[source]
        relation_readout = pool_pair_moments_scaled(
            pair_value, pair_batch, data.pair_bucket, n_graphs, self.pool_mode
        )
        if readout_perm is not None:
            blocks, permutation = readout_perm
            unary, relation_readout = audit.apply_readout_permutation(
                unary, relation_readout, blocks, permutation
            )

        global_input = data.global_context
        if mask.global_zero_groups:
            global_input = audit._replace_grouped_columns(
                global_input, audit.GLOBAL_GROUPS, mask.global_zero_groups, fill, "global:"
            )
        graph_hidden = self.global_encoder(global_input)
        if mask.graph_hidden_zero:
            graph_hidden = torch.zeros_like(graph_hidden)
        topology_input = data.topology_features
        if mask.topology_zero:
            if fill is not None and "topology" in fill:
                topology_input = (
                    fill["topology"]
                    .to(topology_input.device, topology_input.dtype)
                    .reshape(1, -1)
                    .expand_as(topology_input)
                    .contiguous()
                )
            else:
                topology_input = torch.zeros_like(topology_input)
        topology = self.topology_encoder(topology_input)
        unified = torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)
        if int(unified.shape[1]) != READER_DIM_CN:
            raise RuntimeError(f"reader input width {int(unified.shape[1])} != {READER_DIM_CN}")
        prediction = self.reader(unified).view(-1)
        if return_aux:
            return prediction, {
                "E": E,
                "coord": coord,
                "phi": data.dict_phi,
                "unary": unary,
                "relation_readout": relation_readout,
                "pair_value": pair_value,
                "pair_batch": pair_batch,
                "graph_hidden": graph_hidden,
                "topology": topology,
            }
        return prediction


def build_arm_cn(
    arm: str,
    payload: prev.TuplePayload | None = None,
    *,
    kappa_M: float | None = None,
) -> torch.nn.Module:
    """Fresh C/N arm: byte-identical construction to ``mlpmod.build_arm_mj``.

    The only difference from the source M constructor is the model subclass
    (whose construction body is ``prev.LocalTupleFull.__init__`` plus a plain
    ``pool_mode`` attribute) and the swap of the local encoder.  The
    pre-training identity is checked element-wise against the frozen source
    ``M_init_state.pt``.
    """
    if arm not in POOL_MODES:
        raise ValueError(arm)
    if payload is None:
        payload = prev.load_tuple_payload()
    seed_everything(SEED)
    model = ScaledLocalTupleFull(payload, "joint", POOL_MODES[arm])
    d_init = model.local_tuple.D_loc_raw.detach().clone()
    encoder = mlpmod.LocalTupleEncoderM(payload, d_init.t().contiguous())
    if kappa_M is not None:
        encoder.kappa = float(kappa_M)
    model.local_tuple = encoder
    audit_ = mlpmod.parameter_audit_mj(model)
    expected = {
        "total_parameters": EXPECTED_PARAMETERS,
        "base_body_parameters": 184667,
        "bridge_parameters": 82944,
        "local_tuple_parameters": 29888,
        "A_parameters": 8000,
        "W_loc_parameters": 21888,
    }
    if audit_ != expected:
        raise RuntimeError(f"parameter audit failed: {audit_} != {expected}")
    return model


def _model_for_cn(arm: str, payload: prev.TuplePayload, kappa_M: float) -> torch.nn.Module:
    if arm not in POOL_MODES:
        raise ValueError(arm)
    return build_arm_cn(arm, payload, kappa_M=float(kappa_M))


def evaluate_state_cn(
    model: torch.nn.Module,
    data_list: Sequence[Any],
    y_local: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Raw predictions with the round mask (count coordinates visible)."""
    model.eval()
    predictions = np.empty(len(data_list), np.float64)
    target = torch.as_tensor(y_local, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(data_list), batch_size):
            indices = list(range(start, min(start + batch_size, len(data_list))))
            batch = zftd.make_batch(data_list, indices, target, device)
            prediction = model(batch, mask=ROUND_MASK)
            predictions[start : start + len(indices)] = prediction.view(-1).detach().cpu().numpy()
    return predictions


# ---------------------------------------------------------------------------
# 4. frozen inputs: copy the source round artifacts, verify, load
# ---------------------------------------------------------------------------

FROZEN_INPUT_NAMES = (
    "fresh_fold.npz",
    "fresh_targets.npz",
    "fresh_tuple_payload.npz",
    "fresh_prep.npz",
    "fresh_kappa.json",
    "fresh_manifest.json",
)


def prepare_frozen_inputs(
    *, source_dir: Path = SOURCE_DIR, out_dir: Path = RESULTS_DIR, log: Any = print
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_src = json.loads((source_dir / "fresh_manifest.json").read_text())
    copied: dict[str, Any] = {}
    for name in FROZEN_INPUT_NAMES:
        src = source_dir / name
        dst = out_dir / name
        actual = file_sha256(src)
        expected = manifest_src["artifacts"].get(name, {}).get("sha256")
        if expected is not None and actual != expected:
            raise RuntimeError(f"source artifact {name} sha {actual} != frozen {expected}")
        if dst.exists() and file_sha256(dst) == actual:
            pass
        else:
            shutil.copy2(src, dst)
        copied[name] = {
            "source": str(src),
            "sha256": actual,
            "manifest_sha256": expected,
            "copied_sha256": file_sha256(dst),
        }
    if copied["fresh_fold.npz"]["sha256"] != manifest_src["artifacts"]["fold_npz"]["sha256"]:
        raise RuntimeError("fold manifest key mismatch")
    write_json(out_dir / "frozen_input_copy.json", copied)
    log(f"[prepare] frozen inputs copied/verified from {source_dir}")
    return copied


def load_cn_objects(out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Load the frozen fresh-fold objects from the round directory."""
    return fresh.load_fresh_objects(out_dir)


# ---------------------------------------------------------------------------
# 5. pre-training checks (local CPU, before any formal trajectory)
# ---------------------------------------------------------------------------


def pool_reference_checks() -> dict[str, Any]:
    """Independent reference checks of the two pooling functions.

    Covers: two different graph sizes, an empty pair bucket, per-graph vs
    concatenated batch, graph-order shuffle/restore, the sum<->mean recovery
    relation, sum-mode bit-identity with the existing masked pooling, and
    finiteness for every block.
    """
    torch.manual_seed(0)
    value = torch.randn(6, 3, dtype=torch.float64).float()
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)  # equal sizes: add imbalance below
    batch = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
    value = value[:6]
    n_graphs = 2
    pair_value = torch.randn(5, 4, dtype=torch.float64).float()
    pair_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    pair_bucket = torch.tensor([0, 0, 3, 4, 4], dtype=torch.long)  # bucket 1,2 empty

    checks: dict[str, Any] = {}
    for mode in ("sum", "mean"):
        got_u = pool_moments_scaled(value, batch, n_graphs, mode)
        ref_u = reference_pool_unary(value, batch, n_graphs, mode)
        checks[f"unary_{mode}_matches_reference"] = bool(
            torch.allclose(got_u.double(), ref_u, rtol=POOL_RTOL, atol=POOL_ATOL)
        )
        checks[f"unary_{mode}_finite"] = bool(torch.isfinite(got_u).all())

        got_p = pool_pair_moments_scaled(pair_value, pair_batch, pair_bucket, n_graphs, mode)
        ref_p = reference_pool_pair(pair_value, pair_batch, pair_bucket, n_graphs, mode)
        checks[f"pair_{mode}_matches_reference"] = bool(
            torch.allclose(got_p.double(), ref_p, rtol=POOL_RTOL, atol=POOL_ATOL)
        )
        checks[f"pair_{mode}_finite"] = bool(torch.isfinite(got_p).all())

    # sum mode must be bit-identical to the existing C6-pooling with no zeroed block
    got_u_sum = pool_moments_scaled(value, batch, n_graphs, "sum")
    ref_u_sum = audit.pool_moments_masked(value, batch, n_graphs, (), None)
    checks["unary_sum_bitwise_existing"] = bool(torch.equal(got_u_sum, ref_u_sum))
    got_p_sum = pool_pair_moments_scaled(pair_value, pair_batch, pair_bucket, n_graphs, "sum")
    ref_p_sum = audit.pool_pair_moments_masked(pair_value, pair_batch, pair_bucket, n_graphs, (), None)
    checks["pair_sum_bitwise_existing"] = bool(torch.equal(got_p_sum, ref_p_sum))

    # per-graph computation vs concatenated batch (same mode each side)
    for mode in ("sum", "mean"):
        batched_u = pool_moments_scaled(value, batch, n_graphs, mode)
        singles_u = torch.cat(
            [
                pool_moments_scaled(
                    value[batch == g], torch.zeros(int((batch == g).sum()), dtype=torch.long), 1, mode
                )
                for g in range(n_graphs)
            ],
            dim=0,
        )
        checks[f"per_graph_equals_batched_unary_{mode}"] = bool(torch.equal(batched_u, singles_u))
        batched_p = pool_pair_moments_scaled(pair_value, pair_batch, pair_bucket, n_graphs, mode)
        singles_p = torch.cat(
            [
                pool_pair_moments_scaled(
                    pair_value[pair_batch == g],
                    torch.zeros(int((pair_batch == g).sum()), dtype=torch.long),
                    pair_bucket[pair_batch == g],
                    1,
                    mode,
                )
                for g in range(n_graphs)
            ],
            dim=0,
        )
        checks[f"per_graph_equals_batched_pair_{mode}"] = bool(torch.equal(batched_p, singles_p))

    # graph-order shuffle / restore (stable ids): relabel 0->1, 1->0
    perm_batch = torch.where(batch == 0, torch.ones_like(batch), torch.zeros_like(batch))
    perm_pair_batch = torch.where(pair_batch == 0, torch.ones_like(pair_batch), torch.zeros_like(pair_batch))
    for mode in ("sum", "mean"):
        shuffled_u = pool_moments_scaled(value, perm_batch, n_graphs, mode)[[1, 0]]
        checks[f"shuffle_restore_unary_{mode}"] = bool(
            torch.allclose(shuffled_u, pool_moments_scaled(value, batch, n_graphs, mode), rtol=0.0, atol=0.0)
        )
        shuffled_p = pool_pair_moments_scaled(pair_value, perm_pair_batch, pair_bucket, n_graphs, mode)[[1, 0]]
        checks[f"shuffle_restore_pair_{mode}"] = bool(
            torch.allclose(shuffled_p, pool_pair_moments_scaled(pair_value, pair_batch, pair_bucket, n_graphs, mode), rtol=0.0, atol=0.0)
        )

    # math recovery: sum == mean * n; count == log1p(n); empty bucket zero
    mean_u = pool_moments_scaled(value, batch, n_graphs, "mean")
    sum_u = pool_moments_scaled(value, batch, n_graphs, "sum")
    counts_u = torch.bincount(batch, minlength=n_graphs).to(mean_u.dtype).unsqueeze(1)
    checks["sum_equals_mean_times_n_unary"] = bool(
        torch.allclose(sum_u[:, : 2 * value.shape[1]], mean_u[:, : 2 * value.shape[1]] * counts_u, rtol=POOL_RTOL, atol=POOL_ATOL)
    )
    checks["count_is_log1p_n_unary"] = bool(
        torch.allclose(mean_u[:, 2 * value.shape[1] :], torch.log1p(counts_u), rtol=0.0, atol=0.0)
    )
    mean_p = pool_pair_moments_scaled(pair_value, pair_batch, pair_bucket, n_graphs, "mean")
    for bucket in (1, 2):
        lo = bucket * (2 * pair_value.shape[1] + 1)
        block = mean_p[:, lo : lo + 2 * pair_value.shape[1] + 1]
        checks[f"empty_bucket_{bucket}_all_zero"] = bool(torch.equal(block, torch.zeros_like(block)))
        checks[f"empty_bucket_{bucket}_count_zero"] = bool(float(block[:, -1].abs().sum()) == 0.0)

    checks["all_ok"] = bool(all(v for k, v in checks.items() if k != "all_ok"))
    return checks


def forward_equivalence_check(
    payload: prev.TuplePayload,
    kappa_M: float,
    batch: Any,
) -> dict[str, Any]:
    """C's forward body must equal the inherited audit forward bit-for-bit.

    With ``ROUND_MASK`` (no count zeroed) our forward only routes the two pool
    calls through :func:`pool_moments_scaled` / :func:`pool_pair_moments_scaled`;
    the inherited ``audit.AuditModel.forward`` uses the existing masked pooling
    with empty zero-blocks, which is bit-identical for sum mode.
    """
    model = build_arm_cn("C", payload, kappa_M=kappa_M)
    model.eval()
    with torch.no_grad():
        ours = model(batch, mask=ROUND_MASK).view(-1).clone()
        inherited = audit.AuditModel.forward(model, batch, mask=ROUND_MASK).view(-1).clone()
        ours_n = model(batch, mask=ROUND_MASK).view(-1).clone()
    return {
        "our_forward_vs_inherited_max_abs": float((ours - inherited).abs().max()),
        "our_forward_deterministic": bool(torch.equal(ours, ours_n)),
        "bitwise_equal": bool(torch.equal(ours, inherited)),
    }


def init_identity_check(
    payload: prev.TuplePayload,
    kappa_M: float,
    *,
    source_dir: Path = SOURCE_DIR,
) -> dict[str, Any]:
    """C/N init state must be byte-identical to the frozen source M init."""
    source_init = torch.load(source_dir / "M_init_state.pt", map_location="cpu")
    results: dict[str, Any] = {}
    for arm in ARMS:
        model = build_arm_cn(arm, payload, kappa_M=kappa_M)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if set(state) != set(source_init):
            raise RuntimeError(f"{arm} state key set differs from source M init")
        max_abs = 0.0
        shape_mismatch = [key for key in state if tuple(state[key].shape) != tuple(source_init[key].shape)]
        if shape_mismatch:
            raise RuntimeError(f"{arm} shape mismatch: {shape_mismatch}")
        for key in state:
            max_abs = max(max_abs, float((state[key] - source_init[key]).abs().max()))
        results[arm] = {
            "all_state_keys_equal": True,
            "max_abs_diff_vs_source_M_init": max_abs,
            "state_hash": state_hash(state),
            "source_M_init_state_hash": state_hash(source_init),
            "byte_identical": bool(max_abs == 0.0 and state_hash(state) == state_hash(source_init)),
            "pool_mode": POOL_MODES[arm],
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }
    if not results["C"]["byte_identical"] or not results["N"]["byte_identical"]:
        raise RuntimeError(f"C/N init identity failed: {jsonable(results)}")
    return results


def rng_stream_checks() -> dict[str, Any]:
    """Schedule and training RNG stream must match the frozen source recipe."""
    schedule, schedule_hash = zw.build_schedule(N_FIT, EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    seed_everything(SEED)
    model = build_arm_cn("C")
    build_rng = torch.get_rng_state()
    model = model.to(torch.device("cpu"))
    seed_everything(SEED)
    train_rng = torch.get_rng_state()
    return {
        "schedule_sha256": schedule_hash,
        "schedule_matches_frozen": bool(schedule_hash == FROZEN_SCHEDULE_SHA),
        "build_rng_sha256": fresh._hash_bytes(build_rng.numpy().tobytes()),
        "build_rng_matches_frozen": bool(fresh._hash_bytes(build_rng.numpy().tobytes()) == FROZEN_BUILD_RNG_SHA),
        "train_rng_sha256": fresh._hash_bytes(train_rng.numpy().tobytes()),
        "train_rng_matches_frozen": bool(fresh._hash_bytes(train_rng.numpy().tobytes()) == FROZEN_TRAIN_RNG_SHA),
    }


def _count_columns(aux: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    unary = aux["unary"]
    rel = aux["relation_readout"]
    pair_dim = int(aux["pair_value"].shape[1])
    block = 2 * pair_dim + 1
    out = {"unary": unary[:, 2 * (int(unary.shape[1]) - 1) // 2]}
    for bucket in range(int(p2.DISTANCE_BUCKETS)):
        out[f"pair_{bucket}"] = rel[:, bucket * block + 2 * pair_dim]
    return out


def count_witness_and_snapshot(
    fresh_objects: Mapping[str, Any],
    *,
    n_graphs: int = 128,
    out_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Six count coordinates: same for C/N, non-constant, matching real counts.

    Also verifies the N moments are the per-bucket/per-graph sums divided by
    the actual n, saves the representative fit-batch input snapshot and the
    pre-reader block RMS at init.
    """
    fold = fresh_objects["fold"]
    targets = fresh_objects["targets"]
    payload = prev.TuplePayload(fresh_objects["payload_arrays"])
    kappa_M = float(fresh_objects["kappa"]["kappa_M"])
    g = np.asarray(targets["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    _, fit_data, _ = fresh.build_prepared_data(fresh_objects, verify_prep=True)
    indices = list(range(int(n_graphs)))
    device = torch.device("cpu")
    batch = zftd.make_batch(fit_data, indices, target_fit[:n_graphs], device)

    models = {arm: build_arm_cn(arm, payload, kappa_M=kappa_M).to(device).eval() for arm in ARMS}
    aux = {}
    with torch.no_grad():
        for arm in ARMS:
            _pred, a = models[arm](batch, mask=ROUND_MASK, return_aux=True)
            aux[arm] = a
    for arm in ARMS:
        if int(aux[arm]["E"].shape[1]) != UNARY_MOMENT_DIM:
            raise RuntimeError(f"{arm} E width {int(aux[arm]['E'].shape[1])} != {UNARY_MOMENT_DIM}")
        if int(aux[arm]["pair_value"].shape[1]) != PAIR_MOMENT_DIM:
            raise RuntimeError(f"{arm} pair width {int(aux[arm]['pair_value'].shape[1])} != {PAIR_MOMENT_DIM}")

    batch_long = batch.batch.to(torch.long)
    unary_n = torch.bincount(batch_long, minlength=n_graphs).to(torch.float64)
    pair_counts = []
    pair_src = batch.pair_index[0].to(torch.long)
    for bucket in range(int(p2.DISTANCE_BUCKETS)):
        mask = batch.pair_bucket == bucket
        cnt = torch.zeros(n_graphs, dtype=torch.float64)
        if bool(mask.any()):
            cnt.index_add_(0, batch_long[pair_src[mask]], torch.ones(int(mask.sum()), dtype=torch.float64))
        pair_counts.append(cnt)
    pair_counts_t = torch.stack(pair_counts)

    checks: dict[str, Any] = {}
    for arm in ARMS:
        cc = _count_columns(aux[arm])
        checks[f"{arm}_unary_count_matches_bincount"] = bool(
            torch.allclose(cc["unary"].view(-1).double(), torch.log1p(unary_n), rtol=0.0, atol=1e-5)
        )
        ok = all(
            torch.allclose(cc[f"pair_{b}"].view(-1).double(), torch.log1p(pair_counts_t[b]), rtol=0.0, atol=1e-5)
            for b in range(int(p2.DISTANCE_BUCKETS))
        )
        checks[f"{arm}_pair_counts_match_reference"] = bool(ok)
        checks[f"{arm}_counts_finite"] = bool(all(torch.isfinite(v).all() for v in cc.values()))

    # C and N count coordinates identical; non-constant where n > 0
    for name in _count_columns(aux["C"]):
        same = bool(torch.equal(_count_columns(aux["C"])[name], _count_columns(aux["N"])[name]))
        checks[f"count_coordinate_identical_C_N_{name}"] = same
    checks["unary_count_nonconstant"] = bool(float(_count_columns(aux["C"])["unary"].std()) > 0.0)
    for b in range(int(p2.DISTANCE_BUCKETS)):
        cc = _count_columns(aux["C"])[f"pair_{b}"]
        checks[f"pair_{b}_count_consistent_with_n"] = bool(
            int((cc > 0).sum()) == int((pair_counts_t[b] > 0).sum())
        )

    # moment reference: sum and mean*sq per graph, per bucket
    E64 = aux["C"]["E"].double()
    unary_sum_ref = torch.zeros(n_graphs, E64.shape[1], dtype=torch.float64)
    unary_sq_ref = torch.zeros_like(unary_sum_ref)
    unary_sum_ref.index_add_(0, batch_long, E64)
    unary_sq_ref.index_add_(0, batch_long, E64 * E64)
    n_denom = unary_n.clamp_min(1.0).unsqueeze(1)
    checks["C_unary_first_equals_sum"] = bool(
        torch.allclose(aux["C"]["unary"][:, : E64.shape[1]].double(), unary_sum_ref, rtol=POOL_RTOL, atol=POOL_ATOL)
    )
    checks["C_unary_second_equals_sumsq"] = bool(
        torch.allclose(aux["C"]["unary"][:, E64.shape[1] : 2 * E64.shape[1]].double(), unary_sq_ref, rtol=POOL_RTOL, atol=POOL_ATOL)
    )
    checks["N_unary_first_equals_sum_over_n"] = bool(
        torch.allclose(aux["N"]["unary"][:, : E64.shape[1]].double(), unary_sum_ref / n_denom, rtol=POOL_RTOL, atol=POOL_ATOL)
    )
    checks["N_unary_second_equals_sumsq_over_n"] = bool(
        torch.allclose(aux["N"]["unary"][:, E64.shape[1] : 2 * E64.shape[1]].double(), unary_sq_ref / n_denom, rtol=POOL_RTOL, atol=POOL_ATOL)
    )

    pair_value64 = aux["C"]["pair_value"].double()
    pair_denom = pair_counts_t.t().clamp_min(1.0).unsqueeze(2)  # [n_graphs, buckets, 1]
    block = 2 * PAIR_MOMENT_DIM + 1
    for bucket in range(int(p2.DISTANCE_BUCKETS)):
        mask = batch.pair_bucket == bucket
        src = pair_src[mask]
        total = torch.zeros(n_graphs, PAIR_MOMENT_DIM, dtype=torch.float64)
        squared = torch.zeros_like(total)
        if bool(mask.any()):
            vals = pair_value64[mask]
            total.index_add_(0, batch_long[src], vals)
            squared.index_add_(0, batch_long[src], vals * vals)
        lo = bucket * block
        checks[f"C_pair_bucket_{bucket}_first_equals_sum"] = bool(
            torch.allclose(aux["C"]["relation_readout"][:, lo : lo + PAIR_MOMENT_DIM].double(), total, rtol=POOL_RTOL, atol=POOL_ATOL)
        )
        checks[f"C_pair_bucket_{bucket}_second_equals_sumsq"] = bool(
            torch.allclose(aux["C"]["relation_readout"][:, lo + PAIR_MOMENT_DIM : lo + 2 * PAIR_MOMENT_DIM].double(), squared, rtol=POOL_RTOL, atol=POOL_ATOL)
        )
        checks[f"N_pair_bucket_{bucket}_first_equals_sum_over_n"] = bool(
            torch.allclose(
                aux["N"]["relation_readout"][:, lo : lo + PAIR_MOMENT_DIM].double(),
                total / pair_denom[:, bucket],
                rtol=POOL_RTOL,
                atol=POOL_ATOL,
            )
        )

    # pre-reader block RMS at init (same batch for all arms)
    block_rms = {arm: _pre_reader_block_rms(aux[arm]) for arm in ARMS}
    snapshot = {
        "batch": batch_long.numpy(),
        "pair_batch": batch.pair_index[0].numpy(),
        "pair_bucket": batch.pair_bucket.numpy(),
        "unary_n": unary_n.numpy(),
        "pair_n": pair_counts_t.numpy(),
        "C_unary": aux["C"]["unary"].numpy().astype(np.float32),
        "C_relation": aux["C"]["relation_readout"].numpy().astype(np.float32),
        "N_unary": aux["N"]["unary"].numpy().astype(np.float32),
        "N_relation": aux["N"]["relation_readout"].numpy().astype(np.float32),
        "E": aux["C"]["E"].numpy().astype(np.float32),
        "pair_value": aux["C"]["pair_value"].numpy().astype(np.float32),
    }
    np.savez_compressed(out_dir / "count_snapshot.npz", **snapshot)
    witness = {
        "n_graphs": int(n_graphs),
        "unary_n_min": float(unary_n.min()),
        "unary_n_max": float(unary_n.max()),
        "pair_n": pair_counts_t.sum(dim=1).tolist(),
        "checks": checks,
        "block_rms_init": block_rms,
        "count_coordinate_positions": {
            "unary_count_index": UNARY_COUNT_INDEX,
            "pair_count_index_in_block": PAIR_COUNT_INDEX_IN_BLOCK,
            "pair_block_dim": PAIR_BLOCK_DIM_CN,
        },
        "snapshot_npz": "count_snapshot.npz",
        "snapshot_sha256": file_sha256(out_dir / "count_snapshot.npz"),
        "all_ok": bool(all(v for k, v in checks.items() if isinstance(v, bool))),
    }
    if not witness["all_ok"]:
        raise RuntimeError(f"count witness failed: {jsonable(checks)}")
    write_json(out_dir / "count_witness.json", witness)
    return witness


def _pre_reader_block_rms(aux: Mapping[str, Any]) -> dict[str, Any]:
    unary = aux["unary"].double()
    m = int(aux["E"].shape[1])
    p = int(aux["pair_value"].shape[1])
    block = 2 * p + 1
    rel = aux["relation_readout"].double()
    out: dict[str, Any] = {
        "unary_first": _rms(unary[:, :m]),
        "unary_second": _rms(unary[:, m : 2 * m]),
        "unary_count": _rms(unary[:, 2 * m : 2 * m + 1]),
        "graph_hidden": _rms(aux["graph_hidden"].double()),
        "topology": _rms(aux["topology"].double()),
    }
    for bucket in range(int(p2.DISTANCE_BUCKETS)):
        lo = bucket * block
        out[f"pair_{bucket}_first"] = _rms(rel[:, lo : lo + p])
        out[f"pair_{bucket}_second"] = _rms(rel[:, lo + p : lo + 2 * p])
        out[f"pair_{bucket}_count"] = _rms(rel[:, lo + 2 * p : lo + 2 * p + 1])
    return out


def _rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


def pre_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh_objects = load_cn_objects(out_dir)
    payload = prev.TuplePayload(fresh_objects["payload_arrays"])
    kappa_M = float(fresh_objects["kappa"]["kappa_M"])

    fold = fresh_objects["fold"]
    targets = fresh_objects["targets"]
    g = np.asarray(targets["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    _, fit_data, _ = fresh.build_prepared_data(fresh_objects, verify_prep=True)
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    batch = zftd.make_batch(fit_data, [0, 1, 2, 3], target_fit[:4], torch.device("cpu"))

    pool_checks = pool_reference_checks()
    forward_equiv = forward_equivalence_check(payload, kappa_M, batch)
    identity = init_identity_check(payload, kappa_M)
    rng = rng_stream_checks()
    witness = count_witness_and_snapshot(fresh_objects, out_dir=out_dir)

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "pool_reference_checks": pool_checks,
        "forward_equivalence": forward_equiv,
        "init_identity": identity,
        "rng_stream": rng,
        "count_witness": witness,
        "source_dir": str(SOURCE_DIR),
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    all_ok = (
        bool(pool_checks["all_ok"])
        and bool(forward_equiv["bitwise_equal"])
        and bool(identity["C"]["byte_identical"])
        and bool(identity["N"]["byte_identical"])
        and bool(rng["schedule_matches_frozen"])
        and bool(rng["train_rng_matches_frozen"])
        and bool(rng["build_rng_matches_frozen"])
        and bool(witness["all_ok"])
    )
    result["all_ok"] = all_ok
    write_json(out_dir / "pre_checks.json", result)
    log(f"[pre-checks] all_ok={all_ok} seconds={result['seconds']:.1f}")
    if not all_ok:
        raise RuntimeError("pre-checks failed")
    return result


# ---------------------------------------------------------------------------
# 6. smoke / probes / diagnostics / formal training
# ---------------------------------------------------------------------------


def gpu_runtime_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "device": str(device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE"),
        "SLURM_JOB_NODELIST": os.environ.get("SLURM_JOB_NODELIST"),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "cuda_device_count": int(torch.cuda.device_count()),
                "device_name": str(props.name),
                "device_index": int(torch.cuda.current_device()),
                "device_uuid": str(getattr(props, "uuid", "")),
                "device_total_memory": int(getattr(props, "total_memory", 0)),
            }
        )
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            info["nvidia_smi_compute_apps"] = query.stdout.strip()
        except Exception as exc:  # pragma: no cover - environment dependent
            info["nvidia_smi_error"] = str(exc)
    return info


def reader_block_grads(model: torch.nn.Module) -> dict[str, float]:
    first = model.reader.net[0]
    grad = first.weight.grad
    if grad is None:
        return {}
    blocks = {
        "unary": (0, UNARY_DIM_CN),
        "relation": (UNARY_DIM_CN, UNARY_DIM_CN + RELATION_DIM_CN),
        "graph_hidden": (UNARY_DIM_CN + RELATION_DIM_CN, UNARY_DIM_CN + RELATION_DIM_CN + GRAPH_HIDDEN_DIM),
        "topology": (
            UNARY_DIM_CN + RELATION_DIM_CN + GRAPH_HIDDEN_DIM,
            UNARY_DIM_CN + RELATION_DIM_CN + GRAPH_HIDDEN_DIM + TOPOLOGY_DIM,
        ),
    }
    return {name: float(grad[:, lo:hi].norm()) for name, (lo, hi) in blocks.items()}


def _probe_cn(model: torch.nn.Module, epoch: int, step: int, total_norm: float) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "epoch": int(epoch),
        "step_in_epoch": int(step),
        "clip_total_norm": float(total_norm),
        "W_loc_grad_norm": float(model.local_tuple.W_loc.grad.norm())
        if model.local_tuple.W_loc.grad is not None
        else None,
        "A_grad_norm": float(model.local_tuple.A_raw.grad.norm())
        if model.local_tuple.A_raw.grad is not None
        else None,
        "W_loc_norm": float(model.local_tuple.W_loc.detach().norm()),
        "A_norm": float(model.local_tuple.A_raw.detach().norm()),
        "reader_first_layer_block_grad_norm": reader_block_grads(model),
        "health": {key: value for key, value in model.local_tuple.last_stats.items()},
    }
    return entry


def pre_reader_block_rms_state(model: torch.nn.Module, diag_batch: Any) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _pred, aux = model(diag_batch, mask=ROUND_MASK, return_aux=True)
    rms = _pre_reader_block_rms(aux)
    if was_training:
        model.train()
    return rms


def train_arm(
    arm: str,
    *,
    device: torch.device,
    out_dir: Path = RESULTS_DIR,
    epochs: int = EPOCHS,
    log: Any = print,
) -> dict[str, Any]:
    if arm not in POOL_MODES:
        raise ValueError(arm)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh_objects = load_cn_objects(out_dir)
    fold, targets = fresh_objects["fold"], fresh_objects["targets"]
    payload = prev.TuplePayload(fresh_objects["payload_arrays"])
    kappa_M = float(fresh_objects["kappa"]["kappa_M"])
    g = np.asarray(targets["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)

    prep_meta, fit_data, dev_data = fresh.build_prepared_data(fresh_objects, verify_prep=True)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)
    if int(epochs) == EPOCHS and schedule_hash != FROZEN_SCHEDULE_SHA:
        raise RuntimeError(f"schedule hash {schedule_hash} != frozen {FROZEN_SCHEDULE_SHA}")

    seed_everything(SEED)
    model = _model_for_cn(arm, payload, kappa_M)
    build_rng = torch.get_rng_state()
    model = model.to(device)
    seed_everything(SEED)
    train_rng = torch.get_rng_state()
    if fresh._hash_bytes(build_rng.numpy().tobytes()) != FROZEN_BUILD_RNG_SHA:
        raise RuntimeError("build RNG stream differs from the frozen source recipe")
    if fresh._hash_bytes(train_rng.numpy().tobytes()) != FROZEN_TRAIN_RNG_SHA:
        raise RuntimeError("train RNG stream differs from the frozen source recipe")

    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    source_init = torch.load(SOURCE_DIR / "M_init_state.pt", map_location="cpu")
    init_max_abs = max(float((init_state[k] - source_init[k]).abs().max()) for k in init_state)
    if init_max_abs != 0.0 or state_hash(init_state) != SOURCE_M_INIT_STATE_HASH:
        raise RuntimeError(f"{arm} init state differs from source M init (max abs {init_max_abs})")

    diag_batch = zftd.make_batch(fit_data, list(range(128)), target_fit[:128], device)
    diagnostics = {
        "epoch0": pre_reader_block_rms_state(model, diag_batch),
        "probe_epochs": [int(e) for e in LOG_EPOCHS],
        "reader_blocks": {
            "unary": [0, UNARY_DIM_CN],
            "relation": [UNARY_DIM_CN, UNARY_DIM_CN + RELATION_DIM_CN],
            "graph_hidden": [UNARY_DIM_CN + RELATION_DIM_CN, UNARY_DIM_CN + RELATION_DIM_CN + GRAPH_HIDDEN_DIM],
            "topology": [
                UNARY_DIM_CN + RELATION_DIM_CN + GRAPH_HIDDEN_DIM,
                UNARY_DIM_CN + RELATION_DIM_CN + GRAPH_HIDDEN_DIM + TOPOLOGY_DIM,
            ],
        },
        "mode": metadata_mode(device),
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probe_log: list[dict[str, Any]] = []
    epoch40_state: dict[str, torch.Tensor] | None = None
    steps_done = 0
    gid_stream = hashlib.sha256()
    id_stream = hashlib.sha256()
    started = time.perf_counter()
    stopped_reason = "completed"

    for epoch in range(1, int(epochs) + 1):
        model.train()
        task_sum, n_mol, n_steps, gnorm_sum, clip_hits = 0.0, 0, 0, 0.0, 0
        epoch_started = time.perf_counter()
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = schedule[epoch - 1][start : start + BATCH_SIZE]
            index_list = [int(i) for i in indices.tolist()]
            id_stream.update(np.asarray(index_list, np.int64).tobytes())
            gid_stream.update(np.asarray(targets["gid"][fold["fit_idx"][index_list]], np.int64).tobytes())
            batch = zftd.make_batch(fit_data, index_list, target_fit, device)
            prediction = model(batch, mask=ROUND_MASK)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{arm} non-finite loss at epoch {epoch} step {n_steps}")
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                probe_log.append(_probe_cn(model, epoch, 1, total_norm))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "clip_fraction": float(clip_hits / max(n_steps, 1)),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if epoch == 40:
            epoch40_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            diagnostics["epoch40"] = pre_reader_block_rms_state(model, diag_batch)
        if epoch == int(epochs):
            diagnostics[f"epoch{int(epochs)}"] = pre_reader_block_rms_state(model, diag_batch)
        if epoch in soup_epochs:
            soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[{arm}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"gnorm={curve[-1]['grad_norm']:.3g} clip={curve[-1]['clip_fraction']:.3f} "
                f"{curve[-1]['seconds']:.1f}s"
            )

    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    diagnostics["soup"] = pre_reader_block_rms_state_dict(model, soup_state, diag_batch)

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "pool_mode": POOL_MODES[arm],
        "supervision_target": "g = y - c (fresh-fold fit-only constants)",
        "seed": SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "steps_expected": int(epochs) * math.ceil(len(fit_data) / BATCH_SIZE),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "soup_epochs": members,
        "n_fit": int(len(fit_data)),
        "n_dev": int(len(dev_data)),
        "kappa_M": kappa_M,
        "fold": {
            "definition": fold.get("definition", "fresh fold npz"),
            "fit_idx_sha256": fresh._hash_bytes(fold["fit_idx"].tobytes()),
            "dev_idx_sha256": fresh._hash_bytes(fold["dev_idx"].tobytes()),
        },
        "target": {
            "source": str(out_dir / "fresh_targets.npz"),
            "g_sha256": array_sha256(g, np.float64),
            "definition": "g = y - c; constants fitted on the new 8000 fit rows only",
        },
        "parameter_audit": mlpmod.parameter_audit_mj(model),
        "init_state_hash": state_hash(init_state),
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "source_M_init_state_hash": SOURCE_M_INIT_STATE_HASH,
        "init_identity_max_abs_vs_source_M": init_max_abs,
        "build_rng_sha256": fresh._hash_bytes(build_rng.numpy().tobytes()),
        "train_rng_sha256": fresh._hash_bytes(train_rng.numpy().tobytes()),
        "schedule_sha256": schedule_hash,
        "position_stream_sha256": id_stream.hexdigest(),
        "global_gid_stream_sha256": gid_stream.hexdigest(),
        "curve": curve,
        "probe_log": probe_log,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    torch.save(init_state, out_dir / f"{arm}_init_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")
    if epoch40_state is not None:
        torch.save(epoch40_state, out_dir / f"{arm}_epoch40_state.pt")

    raw_predictions: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = _model_for_cn(arm, payload, kappa_M)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        raw_predictions[f"{state_name}_fit"] = evaluate_state_cn(replay, fit_data, g_fit, device)
        raw_predictions[f"{state_name}_dev"] = evaluate_state_cn(replay, dev_data, g[fold["dev_idx"]], device)
    np.savez_compressed(
        out_dir / f"{arm}_raw_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_predictions.items()},
    )

    b_soup = float(np.median(g_fit - raw_predictions["raw_soup_fit"]))
    b_init = float(np.median(g_fit - raw_predictions["init_fit"]))
    b_last = float(np.median(g_fit - raw_predictions["last_fit"]))
    result["calibration_b"] = {"init": b_init, "last": b_last, "raw_soup": b_soup}

    replay = _model_for_cn(arm, payload, kappa_M)
    replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in soup_state.items()}, strict=True)
    batch_data = fit_data[:128]
    gpu_pred = evaluate_state_cn(replay.to(device), batch_data, g_fit[:128], device)
    cpu_pred = evaluate_state_cn(replay.to(torch.device("cpu")), batch_data, g_fit[:128], torch.device("cpu"))
    max_diff = float(np.max(np.abs(gpu_pred - cpu_pred))) if device.type != "cpu" else 0.0
    result["replay_max_abs_diff"] = max_diff
    if device.type != "cpu" and max_diff > REPLAY_TOL:
        raise RuntimeError(f"{arm} CPU/GPU replay maxdiff {max_diff} > {REPLAY_TOL}")

    write_json(out_dir / f"{arm}_meta.json", result)
    write_json(out_dir / f"{arm}_curve.json", curve)
    write_json(out_dir / f"{arm}_probe.json", probe_log)
    write_json(out_dir / f"{arm}_diagnostics.json", diagnostics)
    write_json(out_dir / f"{arm}_gpu_runtime.json", gpu_runtime_info(device))
    log(f"[{arm}] done steps={steps_done} wall={result['wall_clock_s']:.1f}s b={b_soup:.6f} replay={max_diff:.2e}")
    return result


def pre_reader_block_rms_state_dict(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor], diag_batch: Any
) -> dict[str, Any]:
    current = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
    rms = pre_reader_block_rms_state(model, diag_batch)
    model.load_state_dict(current, strict=True)
    return rms


def metadata_mode(device: torch.device) -> dict[str, Any]:
    return {
        "fp32": True,
        "amp": False,
        "ddp": False,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": os.sys.version.split()[0],
    }


def run_smoke(arm: str, *, device: torch.device, out_dir: Path = RESULTS_DIR, steps: int = 3, log: Any = print) -> dict[str, Any]:
    """Fit-only smoke: a few steps, finite loss/grads, states discarded."""
    if arm not in POOL_MODES:
        raise ValueError(arm)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    fresh_objects = load_cn_objects(out_dir)
    fold, targets = fresh_objects["fold"], fresh_objects["targets"]
    payload = prev.TuplePayload(fresh_objects["payload_arrays"])
    kappa_M = float(fresh_objects["kappa"]["kappa_M"])
    g = np.asarray(targets["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    _, fit_data, _ = fresh.build_prepared_data(fresh_objects, verify_prep=True)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)

    seed_everything(SEED)
    model = _model_for_cn(arm, payload, kappa_M).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    losses: list[float] = []
    grad_norms: list[float] = []
    indices = [int(i) for i in schedule[0][:BATCH_SIZE].tolist()]
    for _ in range(int(steps)):
        batch = zftd.make_batch(fit_data, indices, target_fit, device)
        prediction = model(batch, mask=ROUND_MASK)
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        optimizer.step()
        losses.append(float(loss))
        grad_norms.append(total_norm)
    finite = bool(all(math.isfinite(v) for v in losses + grad_norms))
    finite_grads = bool(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    mean_count_block_rms = 0.0
    with torch.no_grad():
        model.eval()
        _pred, aux = model(batch, mask=ROUND_MASK, return_aux=True)
        cc = _count_columns(aux)
        mean_count_block_rms = float(torch.cat([v.view(-1) for v in cc.values()]).pow(2).mean().sqrt())
        finite_forward = bool(torch.isfinite(_pred).all())
    # save/reload forward identity (single file)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(state, out_dir / f"smoke_{arm}_state_discard.pt")
    reloaded = _model_for_cn(arm, payload, kappa_M).to(device)
    reloaded.load_state_dict({key: value.to(device) for key, value in state.items()}, strict=True)
    reloaded.eval()
    with torch.no_grad():
        pred2 = reloaded(batch, mask=ROUND_MASK).view(-1)
        pred1 = model(batch, mask=ROUND_MASK).view(-1)
    reload_max = float((pred1 - pred2).abs().max())
    (out_dir / f"smoke_{arm}_state_discard.pt").unlink()
    result = {
        "arm": arm,
        "pool_mode": POOL_MODES[arm],
        "device": str(device),
        "steps": int(steps),
        "losses": losses,
        "grad_norms": grad_norms,
        "all_finite": finite,
        "all_param_grads_finite": finite_grads,
        "forward_finite": finite_forward,
        "count_block_rms": mean_count_block_rms,
        "W_loc_grad_nonzero_step_last": float(model.local_tuple.W_loc.grad.norm()) > 0.0,
        "A_grad_nonzero_step_last": float(model.local_tuple.A_raw.grad.norm()) > 0.0,
        "save_reload_max_abs": reload_max,
        "schedule_sha256": schedule_hash,
        "states_discarded": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "smoke" / f"smoke_checks_{arm}.json", result)
    log(f"[smoke-{arm}] losses={['%.5f' % v for v in losses]} grad={['%.3f' % v for v in grad_norms]}")
    return result


# ---------------------------------------------------------------------------
# 7. analysis: main table, groups, paired bootstrap, gates, classification
# ---------------------------------------------------------------------------


def load_arm(out_dir: Path, arm: str) -> dict[str, Any]:
    meta = json.loads((out_dir / f"{arm}_meta.json").read_text())
    with np.load(out_dir / f"{arm}_raw_predictions.npz", allow_pickle=False) as z:
        preds = {key: z[key].astype(np.float64) for key in z.files}
    return {"meta": meta, "preds": preds}


def load_source_arm(source_dir: Path, arm: str) -> dict[str, Any]:
    meta = json.loads((source_dir / f"{arm}_meta.json").read_text())
    with np.load(source_dir / f"{arm}_raw_predictions.npz", allow_pickle=False) as z:
        preds = {key: z[key].astype(np.float64) for key in z.files}
    return {"meta": meta, "preds": preds}


def arm_metrics(
    pred_fit: np.ndarray, pred_dev: np.ndarray, g_fit: np.ndarray, g_dev: np.ndarray, k_dev: np.ndarray
) -> dict[str, Any]:
    bias = float(np.median(g_fit - pred_fit))
    cal_fit = pred_fit + bias
    cal_dev = pred_dev + bias
    g0 = k_dev == 0
    return {
        "bias": bias,
        "fit_overall_cal_mae": float(np.mean(np.abs(g_fit - cal_fit))),
        "dev_overall_raw_mae": float(np.mean(np.abs(g_dev - pred_dev))),
        "dev_overall_cal_mae": float(np.mean(np.abs(g_dev - cal_dev))),
        "dev_G0_raw_mae": float(np.mean(np.abs((g_dev - pred_dev)[g0]))),
        "dev_G0_cal_mae": float(np.mean(np.abs((g_dev - cal_dev)[g0]))),
        "err_dev_raw": np.abs(g_dev - pred_dev),
        "err_dev_cal": np.abs(g_dev - cal_dev),
        "err_fit_cal": np.abs(g_fit - cal_fit),
    }


def group_metrics(
    err: np.ndarray, k_dev: np.ndarray, n_dev: int
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": int(k_dev.size),
        "mae": float(np.mean(err)),
        "contribution": float(err.sum() / n_dev),
    }
    for name in GROUP_NAMES:
        mask = {"k=0": k_dev == 0, "k=-1": k_dev == -1, "k=-2": k_dev == -2, "k<=-3": k_dev <= -3}[name]
        out[name] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(err[mask])) if bool(mask.any()) else None,
            "contribution": float(err[mask].sum() / n_dev),
        }
    return out


def paired_gain(
    err_control: np.ndarray,
    err_candidate: np.ndarray,
    row_mask: np.ndarray,
    boot_indices: np.ndarray,
) -> dict[str, Any]:
    c = err_control[row_mask]
    d = err_candidate[row_mask]
    draws = boot_indices
    gains = c[draws].mean(axis=1) - d[draws].mean(axis=1)
    point = float(c.mean() - d.mean())
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_rows": int(row_mask.sum()),
        "control_mae": float(c.mean()),
        "candidate_mae": float(d.mean()),
    }


def gate_conditions(g_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    conds = {
        "G0_cal_gain_ge_0.003": bool(g_metrics["G0_cal"]["point"] >= PERF_DELTA),
        "overall_cal_gain_ge_0.003": bool(g_metrics["overall_cal"]["point"] >= PERF_DELTA),
        "G0_cal_CI_lower_gt_0": bool(g_metrics["G0_cal"]["ci_low"] > 0.0),
        "G0_raw_gain_gt_0": bool(g_metrics["G0_raw"]["point"] > 0.0),
        "overall_raw_gain_gt_0": bool(g_metrics["overall_raw"]["point"] > 0.0),
    }
    return {"conditions": conds, "pass_count": int(sum(conds.values())), "pass": bool(all(conds.values()))}


def classify(gate_nc: Mapping[str, Any], gate_nm: Mapping[str, Any], gate_cm: Mapping[str, Any]) -> str:
    if gate_nc["pass"] and gate_nm["pass"]:
        return "NORMALIZATION_PERFORMANCE_CANDIDATE"
    if gate_nm["pass"] and not gate_nc["pass"]:
        return "INTERFACE_CANDIDATE_ATTRIBUTION_UNRESOLVED"
    if (not gate_nm["pass"]) and gate_cm["pass"]:
        return "COUNT_ACCESS_CANDIDATE"
    if (not gate_nm["pass"]) and (not gate_cm["pass"]) and gate_nc["pass"]:
        return "NORMALIZATION_SIGNAL_NO_PRACTICAL_GAIN"
    return "NO_CANDIDATE"


def replay_source_m(
    fresh_objects: Mapping[str, Any], source_dir: Path, *, tol: float = REPLAY_TOL
) -> dict[str, Any]:
    """Reload the source M soup with the original C6 path and re-predict."""
    payload = prev.TuplePayload(fresh_objects["payload_arrays"])
    kappa_M = float(fresh_objects["kappa"]["kappa_M"])
    fold = fresh_objects["fold"]
    targets = fresh_objects["targets"]
    g = np.asarray(targets["g"], np.float64)
    prep_meta, fit_data, dev_data = fresh.build_prepared_data(fresh_objects, verify_prep=True)
    model = mlpmod.build_arm_mj(payload, kappa_M=kappa_M)
    state = torch.load(source_dir / "M_raw_soup_state.pt", map_location="cpu")
    model.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
    fit128 = evaluate_state_c6(model, fit_data[:128], g[fold["fit_idx"]][:128])
    dev128 = evaluate_state_c6(model, dev_data[:128], g[fold["dev_idx"]][:128])
    with np.load(source_dir / "M_raw_predictions.npz", allow_pickle=False) as z:
        stored_fit = z["raw_soup_fit"].astype(np.float64)[:128]
        stored_dev = z["raw_soup_dev"].astype(np.float64)[:128]
    diff_fit = np.max(np.abs(fit128.reshape(-1) - stored_fit.reshape(-1)))
    diff_dev = np.max(np.abs(dev128.reshape(-1) - stored_dev.reshape(-1)))
    return {
        "replay_fit128_max_abs": float(diff_fit),
        "replay_dev128_max_abs": float(diff_dev),
        "within_tol": bool(diff_fit <= tol and diff_dev <= tol),
        "source_predictions_sha256": file_sha256(source_dir / "M_raw_predictions.npz"),
    }


def evaluate_state_c6(model: torch.nn.Module, data_list: Sequence[Any], y_local: np.ndarray) -> np.ndarray:
    model.eval()
    predictions = np.empty(len(data_list), np.float64)
    target = torch.as_tensor(y_local, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(data_list), BATCH_SIZE):
            indices = list(range(start, min(start + BATCH_SIZE, len(data_list))))
            batch = zftd.make_batch(data_list, indices, target, torch.device("cpu"))
            prediction = model(batch, mask=cm.C6_MASK)
            predictions[start : start + len(indices)] = prediction.view(-1).detach().cpu().numpy()
    return predictions


def state_replay_check(
    fresh_objects: Mapping[str, Any], out_dir: Path, arm: str, *, tol: float = REPLAY_TOL
) -> dict[str, Any]:
    """Reload one arm state from its single file and re-predict stored rows."""
    payload = prev.TuplePayload(fresh_objects["payload_arrays"])
    kappa_M = float(fresh_objects["kappa"]["kappa_M"])
    fold = fresh_objects["fold"]
    targets = fresh_objects["targets"]
    g = np.asarray(targets["g"], np.float64)
    _, fit_data, dev_data = fresh.build_prepared_data(fresh_objects, verify_prep=True)
    model = _model_for_cn(arm, payload, kappa_M)
    state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu")
    model.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
    fit128 = evaluate_state_cn(model, fit_data[:128], g[fold["fit_idx"]][:128], torch.device("cpu"))
    dev128 = evaluate_state_cn(model, dev_data[:128], g[fold["dev_idx"]][:128], torch.device("cpu"))
    with np.load(out_dir / f"{arm}_raw_predictions.npz", allow_pickle=False) as z:
        stored_fit = z["raw_soup_fit"].astype(np.float64)[:128].reshape(-1)
        stored_dev = z["raw_soup_dev"].astype(np.float64)[:128].reshape(-1)
    diff_fit = float(np.max(np.abs(fit128.reshape(-1) - stored_fit)))
    diff_dev = float(np.max(np.abs(dev128.reshape(-1) - stored_dev)))
    return {
        "fit128_max_abs": diff_fit,
        "dev128_max_abs": diff_dev,
        "within_tol": bool(diff_fit <= tol and diff_dev <= tol),
        "single_file": f"{arm}_raw_soup_state.pt",
    }


def scale_table(
    fresh_objects: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    g_fit: np.ndarray,
    g_dev: np.ndarray,
    k_dev: np.ndarray,
    fold: Mapping[str, Any],
) -> list[dict[str, Any]]:
    node_sizes = np.asarray(fresh_objects["payload_arrays"]["node_sizes"], dtype=np.float64)
    fit_nodes = node_sizes[fold["fit_idx"]]
    dev_nodes = node_sizes[fold["dev_idx"]]
    edges = np.quantile(fit_nodes, [0.0, 0.25, 0.5, 0.75, 1.0])
    inner = edges[1:-1]
    fit_bin = np.searchsorted(inner, fit_nodes, side="right")
    dev_bin = np.searchsorted(inner, dev_nodes, side="right")
    rows: list[dict[str, Any]] = []
    for b in range(4):
        m_fit = fit_bin == b
        row: dict[str, Any] = {
            "bin": b,
            "fit_edges": f"{float(edges[b])};{float(edges[b + 1])}",
            "n_fit": int(m_fit.sum()),
            "n_dev": int((dev_bin == b).sum()),
        }
        m = dev_bin == b
        for name, item in arms.items():
            cal_dev = item["preds"]["raw_soup_dev"] + item["bias"]
            err_cal = np.abs(g_dev - cal_dev)
            fit_cal = item["preds"]["raw_soup_fit"] + item["bias"]
            row[f"{name}_fit_cal_mae"] = (
                float(np.mean(np.abs(g_fit - fit_cal)[m_fit])) if bool(m_fit.any()) else None
            )
            row[f"{name}_dev_cal_mae"] = float(np.mean(err_cal[m])) if bool(m.any()) else None
            row[f"{name}_dev_cal_signed_error"] = float(np.mean((g_dev - cal_dev)[m])) if bool(m.any()) else None
        for control, candidate in (("C", "N"), ("M", "N"), ("M", "C")):
            a = arms[control]
            cnd = arms[candidate]
            e_control = np.abs(g_dev - (a["preds"]["raw_soup_dev"] + a["bias"]))
            e_cand = np.abs(g_dev - (cnd["preds"]["raw_soup_dev"] + cnd["bias"]))
            row[f"{candidate}_minus_{control}_dev_cal_gain"] = (
                float(np.mean(e_control[m]) - np.mean(e_cand[m])) if bool(m.any()) else None
            )
        rows.append(row)
    return rows


def analyze(*, out_dir: Path = RESULTS_DIR, source_dir: Path = SOURCE_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    started = time.perf_counter()
    fresh_objects = load_cn_objects(out_dir)
    fold = fresh_objects["fold"]
    targets = fresh_objects["targets"]
    g = np.asarray(targets["g"], np.float64)
    k = np.asarray(targets["k"], np.int64)
    g_fit = g[fold["fit_idx"]]
    g_dev = g[fold["dev_idx"]]
    k_dev = k[fold["dev_idx"]]
    n_dev = int(g_dev.size)
    g0_mask = k_dev == 0

    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        item = load_arm(out_dir, arm)
        item["bias"] = float(item["meta"]["calibration_b"]["raw_soup"])
        item["metrics"] = arm_metrics(item["preds"]["raw_soup_fit"], item["preds"]["raw_soup_dev"], g_fit, g_dev, k_dev)
        arms[arm] = item
    for name, source_arm in (("M", "M"), ("B", "B")):
        item = load_source_arm(source_dir, source_arm)
        item["bias"] = float(item["meta"]["calibration_b"]["raw_soup"])
        item["metrics"] = arm_metrics(item["preds"]["raw_soup_fit"], item["preds"]["raw_soup_dev"], g_fit, g_dev, k_dev)
        arms[name] = item

    rng = np.random.default_rng(BOOT_SEED)
    idx_overall = rng.integers(0, n_dev, size=(N_BOOT, n_dev))
    g0_rows = np.nonzero(g0_mask)[0]
    idx_g0 = rng.integers(0, g0_rows.size, size=(N_BOOT, g0_rows.size))

    comparisons = {
        "N_vs_C": ("C", "N"),
        "N_vs_M": ("M", "N"),
        "C_vs_M": ("M", "C"),
    }
    gains: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for name, (control, candidate) in comparisons.items():
        err_control = np.abs(g_dev - (arms[control]["preds"]["raw_soup_dev"] + arms[control]["bias"]))
        err_cand = np.abs(g_dev - (arms[candidate]["preds"]["raw_soup_dev"] + arms[candidate]["bias"]))
        g_metrics = {
            "G0_cal": paired_gain(err_control, err_cand, g0_mask, idx_g0),
            "overall_cal": paired_gain(err_control, err_cand, np.ones(n_dev, dtype=bool), idx_overall),
            "G0_raw": paired_gain(
                np.abs(g_dev - arms[control]["preds"]["raw_soup_dev"]),
                np.abs(g_dev - arms[candidate]["preds"]["raw_soup_dev"]),
                g0_mask,
                idx_g0,
            ),
            "overall_raw": paired_gain(
                np.abs(g_dev - arms[control]["preds"]["raw_soup_dev"]),
                np.abs(g_dev - arms[candidate]["preds"]["raw_soup_dev"]),
                np.ones(n_dev, dtype=bool),
                idx_overall,
            ),
        }
        gains[name] = {"control": control, "candidate": candidate, "metrics": g_metrics}
        gates[name] = gate_conditions(g_metrics)

    classification = classify(gates["N_vs_C"], gates["N_vs_M"], gates["C_vs_M"])

    # witnesses: identical predictions -> zero; swapped -> mirrored
    err_c = np.abs(g_dev - (arms["C"]["preds"]["raw_soup_dev"] + arms["C"]["bias"]))
    err_n = np.abs(g_dev - (arms["N"]["preds"]["raw_soup_dev"] + arms["N"]["bias"]))
    same = paired_gain(err_c, err_c, g0_mask, idx_g0)
    swapped = paired_gain(err_n, err_c, g0_mask, idx_g0)
    witnesses = {
        "same_predictions_zero": bool(same["point"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0),
        "swap_mirrors_point": bool(abs(swapped["point"] + gains["N_vs_C"]["metrics"]["G0_cal"]["point"]) < 1e-12),
        "swap_mirrors_ci": bool(
            abs(swapped["ci_low"] + gains["N_vs_C"]["metrics"]["G0_cal"]["ci_high"]) < 1e-12
            and abs(swapped["ci_high"] + gains["N_vs_C"]["metrics"]["G0_cal"]["ci_low"]) < 1e-12
        ),
        "same_ci": [same["ci_low"], same["ci_high"]],
        "swapped_G0_cal": swapped,
    }

    sensitivity: dict[str, Any] = {}
    for name, (control, candidate) in comparisons.items():
        err_control = np.abs(g_dev - (arms[control]["preds"]["raw_soup_dev"] + arms[control]["bias"]))
        err_cand = np.abs(g_dev - (arms[candidate]["preds"]["raw_soup_dev"] + arms[candidate]["bias"]))
        mean_err = 0.5 * (err_control + err_cand)
        worst = int(np.argmax(mean_err))
        keep = np.ones(n_dev, dtype=bool)
        keep[worst] = False
        sensitivity[name] = {
            "worst_row": worst,
            "worst_gid": int(np.asarray(targets["gid"])[fold["dev_idx"]][worst]),
            "worst_k": int(k_dev[worst]),
            "G0_cal_gain_drop": float(err_control[g0_mask & keep].mean() - err_cand[g0_mask & keep].mean()),
            "overall_cal_gain_drop": float(err_control[keep].mean() - err_cand[keep].mean()),
        }

    # group table + gains per group
    group_rows: list[dict[str, Any]] = []
    group_gain_rows: list[dict[str, Any]] = []
    for name, item in arms.items():
        gm = group_metrics(item["metrics"]["err_dev_cal"], k_dev, n_dev)
        for gname in ("k=0", "k=-1", "k=-2", "k<=-3"):
            group_rows.append(
                {
                    "arm": name,
                    "group": gname,
                    "n": gm[gname]["n"],
                    "mae_cal": gm[gname]["mae"],
                    "contribution_cal": gm[gname]["contribution"],
                }
            )
    for name, (control, candidate) in comparisons.items():
        e_control = np.abs(g_dev - (arms[control]["preds"]["raw_soup_dev"] + arms[control]["bias"]))
        e_cand = np.abs(g_dev - (arms[candidate]["preds"]["raw_soup_dev"] + arms[candidate]["bias"]))
        for gname, mask in (
            ("k=0", k_dev == 0),
            ("k=-1", k_dev == -1),
            ("k=-2", k_dev == -2),
            ("k<=-3", k_dev <= -3),
        ):
            group_gain_rows.append(
                {
                    "comparison": name,
                    "group": gname,
                    "n": int(mask.sum()),
                    "control_mae_cal": float(e_control[mask].mean()) if bool(mask.any()) else None,
                    "candidate_mae_cal": float(e_cand[mask].mean()) if bool(mask.any()) else None,
                    "gain_cal": float(e_control[mask].mean() - e_cand[mask].mean()) if bool(mask.any()) else None,
                    "contribution_gain_cal": float((e_control[mask].sum() - e_cand[mask].sum()) / n_dev),
                }
            )

    scale_rows = scale_table(fresh_objects, {k: arms[k] for k in ("C", "N", "M", "B")}, g_fit, g_dev, k_dev, fold)

    # mechanism evidence
    mechanism: dict[str, Any] = {"block_rms": {}, "reader_block_grads": {}, "pair_count_scale": {}}
    for arm in ARMS:
        diagnostics = json.loads((out_dir / f"{arm}_diagnostics.json").read_text())
        probe = json.loads((out_dir / f"{arm}_probe.json").read_text())
        mechanism["block_rms"][arm] = {
            key: diagnostics[key] for key in diagnostics if key.startswith("epoch") or key == "soup"
        }
        mechanism["reader_block_grads"][arm] = [
            {"epoch": p["epoch"], **p["reader_first_layer_block_grad_norm"]} for p in probe
        ]
    with np.load(out_dir / "count_snapshot.npz", allow_pickle=False) as z:
        snap = {key: z[key] for key in z.files}
    for bucket in range(int(p2.DISTANCE_BUCKETS)):
        lo = bucket * PAIR_BLOCK_DIM_CN
        c_first = snap["C_relation"][:, lo : lo + PAIR_MOMENT_DIM].astype(np.float64)
        n_first = snap["N_relation"][:, lo : lo + PAIR_MOMENT_DIM].astype(np.float64)
        counts = snap["pair_n"][bucket].astype(np.float64)
        mechanism["pair_count_scale"][f"bucket_{bucket}"] = {
            "total_pairs": float(counts.sum()),
            "mean_pairs_per_graph": float(counts.mean()),
            "rms_sum_moment": float(np.sqrt((c_first ** 2).mean())),
            "rms_mean_moment": float(np.sqrt((n_first ** 2).mean())),
            "rms_mean_over_sum": float(
                np.sqrt((n_first ** 2).mean()) / max(np.sqrt((c_first ** 2).mean()), 1e-30)
            ),
        }
    u_first = snap["C_unary"].astype(np.float64)[:, :UNARY_MOMENT_DIM]
    un_first = snap["N_unary"].astype(np.float64)[:, :UNARY_MOMENT_DIM]
    u_counts = snap["unary_n"].astype(np.float64)
    mechanism["pair_count_scale"]["unary"] = {
        "total_env_rows": float(u_counts.sum()),
        "mean_env_rows_per_graph": float(u_counts.mean()),
        "rms_sum_moment": float(np.sqrt((u_first ** 2).mean())),
        "rms_mean_moment": float(np.sqrt((un_first ** 2).mean())),
    }

    replay = {
        "source_M": replay_source_m(fresh_objects, source_dir),
        "C": state_replay_check(fresh_objects, out_dir, "C"),
        "N": state_replay_check(fresh_objects, out_dir, "N"),
    }

    # source M main-metric reproduction against the source analysis
    source_analysis = json.loads((source_dir / "analysis.json").read_text())
    source_m_expected = source_analysis["arms"]["M"]
    source_m_repro = {
        "G0_cal": float(abs(arms["M"]["metrics"]["dev_G0_cal_mae"] - source_m_expected["dev_G0_cal_mae"])),
        "overall_cal": float(abs(arms["M"]["metrics"]["dev_overall_cal_mae"] - source_m_expected["dev_overall_cal_mae"])),
        "G0_raw": float(abs(arms["M"]["metrics"]["dev_G0_raw_mae"] - source_m_expected["dev_G0_raw_mae"])),
        "overall_raw": float(abs(arms["M"]["metrics"]["dev_overall_raw_mae"] - source_m_expected["dev_overall_raw_mae"])),
    }

    identity = {
        "C_N_position_stream_equal": bool(
            arms["C"]["meta"]["position_stream_sha256"] == arms["N"]["meta"]["position_stream_sha256"]
        ),
        "C_N_gid_stream_equal": bool(
            arms["C"]["meta"]["global_gid_stream_sha256"] == arms["N"]["meta"]["global_gid_stream_sha256"]
        ),
        "C_N_gid_stream_is_frozen_source": bool(
            arms["C"]["meta"]["global_gid_stream_sha256"] == FROZEN_GID_STREAM_SHA
        ),
        "C_N_train_rng_equal": bool(arms["C"]["meta"]["train_rng_sha256"] == arms["N"]["meta"]["train_rng_sha256"]),
        "C_N_init_state_equal": bool(arms["C"]["meta"]["init_state_hash"] == arms["N"]["meta"]["init_state_hash"]),
        "steps_C": int(arms["C"]["meta"]["steps_done"]),
        "steps_N": int(arms["N"]["meta"]["steps_done"]),
        "schedule_sha256": arms["C"]["meta"]["schedule_sha256"],
        "gid_stream_sha256": arms["C"]["meta"]["global_gid_stream_sha256"],
    }

    main_rows = []
    for name, item in arms.items():
        m = item["metrics"]
        main_rows.append(
            {
                "arm": name,
                "bias": item["bias"],
                "fit_overall_cal_mae": m["fit_overall_cal_mae"],
                "dev_overall_raw_mae": m["dev_overall_raw_mae"],
                "dev_overall_cal_mae": m["dev_overall_cal_mae"],
                "dev_G0_raw_mae": m["dev_G0_raw_mae"],
                "dev_G0_cal_mae": m["dev_G0_cal_mae"],
            }
        )

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "classification": classification,
        "comparisons": comparisons,
        "gains": gains,
        "gates": gates,
        "gate_summary": {name: gates[name]["pass_count"] for name in gates},
        "bootstrap": {"seed": BOOT_SEED, "n_boot": N_BOOT, "endpoint": "paired dev-row resample"},
        "witnesses": witnesses,
        "sensitivity_drop_worst": sensitivity,
        "group_table": group_rows,
        "group_gain_table": group_gain_rows,
        "scale_table": scale_rows,
        "mechanism": mechanism,
        "replay": replay,
        "source_M_metric_reproduction_max_abs": source_m_repro,
        "identity_checks": identity,
        "main_table": main_rows,
        "n_dev": n_dev,
        "n_G0": int(g0_mask.sum()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - started),
    }
    write_json(out_dir / "analysis.json", result)
    write_json(out_dir / "gains.json", gains)
    write_json(out_dir / "gate.json", {"classification": classification, "gates": gates})
    write_json(out_dir / "bootstrap.json", {"seed": BOOT_SEED, "n_boot": N_BOOT, "gains": gains, "witnesses": witnesses})
    write_json(out_dir / "classification.json", {"classification": classification, "gate_summary": result["gate_summary"]})
    write_json(out_dir / "mechanism_evidence.json", mechanism)
    write_json(out_dir / "replay_checks.json", replay)
    write_main_csv(out_dir / "main_table.csv", main_rows)
    write_group_csv(out_dir / "group_table.csv", group_rows)
    write_group_gain_csv(out_dir / "group_gain_table.csv", group_gain_rows)
    write_scale_csv(out_dir / "scale_table.csv", scale_rows)
    log(f"[analyze] classification={classification} gate_summary={result['gate_summary']} seconds={result['seconds']:.1f}")
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join("" if row.get(key) is None else str(row.get(key)) for key in keys))
    path.write_text("\n".join(lines) + "\n")


write_main_csv = _write_csv
write_group_csv = _write_csv
write_group_gain_csv = _write_csv
write_scale_csv = _write_csv


# ---------------------------------------------------------------------------
# 8. budget / manifest / CLI
# ---------------------------------------------------------------------------


def make_budget(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    budget: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "wall_minutes_max": 120,
        "compute_stop_minute": 90,
        "gpu_hours_max": 0.8,
        "gpu_concurrency_max": 2,
        "local_cpu_threads_max": 8,
    }
    walls = {}
    for arm in ARMS:
        meta_path = out_dir / f"{arm}_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            walls[arm] = float(meta["wall_clock_s"])
            budget[f"{arm}_steps"] = int(meta["steps_done"])
            budget[f"{arm}_gpu_runtime"] = (
                json.loads((out_dir / f"{arm}_gpu_runtime.json").read_text())
                if (out_dir / f"{arm}_gpu_runtime.json").exists()
                else None
            )
            run_meta_path = out_dir / f"{arm}_run_meta.json"
            if run_meta_path.exists():
                budget[f"{arm}_run_meta"] = json.loads(run_meta_path.read_text())
    budget["training_wall_seconds"] = walls
    budget["training_gpu_hours_upper_bound"] = float(sum(walls.values()) / 3600.0)
    for arm in ARMS:
        smoke_path = out_dir / "smoke" / f"smoke_checks_{arm}.json"
        budget[f"{arm}_smoke"] = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    budget["official_valid_loaded"] = False
    budget["official_test_loaded"] = False
    write_json(out_dir / "budget.json", budget)
    log(f"[budget] training_gpu_hours_upper_bound={budget['training_gpu_hours_upper_bound']:.4f}")
    return budget


def make_manifest(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    files: dict[str, Any] = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(out_dir))] = {"sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "result_slug": RESULT_SLUG,
        "execution_commit": "ddd5bd394fe2c2e71b3b0017131eb39f593469e",
        "manifest_commit": commit,
        "source_round": "zinc_local_tuple_fresh_fold_replication_seed0_v1",
        "source_commit": "a5400de",
        "prereg_commit": "2e4ee3f",
        "arms": {"C": "sum/count", "N": "mean/count"},
        "frozen_constants": {
            "new_fit_sha256": NEW_FIT_SHA,
            "new_dev_sha256": NEW_DEV_SHA,
            "schedule_sha256": FROZEN_SCHEDULE_SHA,
            "source_gid_stream_sha256": FROZEN_GID_STREAM_SHA,
            "bootstrap_seed": BOOT_SEED,
            "n_boot": N_BOOT,
            "perf_delta": PERF_DELTA,
        },
        "files": files,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "manifest.json", manifest)
    log(f"[manifest] {len(files)} files, commit={commit[:12]}")
    return manifest


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="copy+verify frozen source inputs")
    parser.add_argument("--pre-checks", action="store_true", help="local CPU pre-training checks")
    parser.add_argument("--smoke", action="store_true", help="fit-only smoke (states discarded)")
    parser.add_argument("--train", action="store_true", help="formal trajectory")
    parser.add_argument("--analyze", action="store_true", help="analysis from stored predictions")
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--arm", choices=list(ARMS), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--smoke-steps", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    if args.prepare:
        prepare_frozen_inputs(out_dir=out_dir)
    if args.pre_checks:
        pre_checks(out_dir=out_dir)
    if args.smoke:
        if args.arm is None:
            raise SystemExit("--smoke requires --arm")
        run_smoke(args.arm, device=fresh.resolve_device(args.device), out_dir=out_dir, steps=args.smoke_steps)
    if args.train:
        if args.arm is None:
            raise SystemExit("--train requires --arm")
        train_arm(args.arm, device=fresh.resolve_device(args.device), out_dir=out_dir)
    if args.analyze:
        analyze(out_dir=out_dir)
    if args.budget:
        make_budget(out_dir=out_dir)
    if args.manifest:
        make_manifest(out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
