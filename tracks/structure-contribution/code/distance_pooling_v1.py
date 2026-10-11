"""cscl-distance-pooling-bottleneck-v1 — fixed continuous-distance (Soft5) vs
hard-bucket (Hard5) graph-level pair-moment pooling.

Round: ``cscl-distance-pooling-bottleneck-v1``
Track: ``structure-contribution`` (CSCL).

Research question
-----------------
Full-Y's pair state ``h_ij in R^48`` is summarised by five *hard* shortest-path
distance buckets (1, 2, 3, 4, >=5), each contributing ``sum(h)``, ``sum(h^2)``
and ``log1p(count)`` to the 814-D reader input.  The pair encoder itself
already receives the continuous ``log1p(distance)`` coordinate.  This round
asks whether replacing only that graph-level hard grouping by one **fixed**
continuous distance basis (Soft5: five normalised Gaussian RBF weights in
log-distance space) changes held-out raw-y prediction.

Frozen Soft5 recipe (pre-registered, never tuned on y)
------------------------------------------------------
``s(d) = log1p(d)``, centers ``[1, 2, 3, 4, 8]`` represented as
``mu_k = log1p(center_k)``, fixed bandwidth ``sigma = 0.35``,
``a_k(d) = exp(-(s(d)-mu_k)^2 / (2 sigma^2))``, ``w_k(d) = a_k / sum_j a_j``.

Weighted moment pooling (the ONLY change relative to Hard5)::

    first_k  = sum_ij w_k(d_ij) * h_ij
    second_k = sum_ij w_k(d_ij) * h_ij^2      # square of h, then weighted
    count_k  = sum_ij w_k(d_ij)               # stored log1p, C6-masked to 0

Everything else (K-SVD / CSSD coding, environment encoder, pair projection,
15-D relation encoder input including ``log1p(distance)``, the discrete
``distance_gate`` embedding, pair encoder, global/topology encoders, reader,
C6_MASK semantics, reconstruction loss, parameter count 408,651, reader width
814) is byte-for-byte the frozen Full-Y computation.

Data discipline
---------------
* The Soft5 weights are read from ``data.pair_relation[:, 5]`` — the exact
  ``log1p(shortest path distance)`` coordinate the relation encoder already
  consumes.  No new input channel is introduced.
* No trainable parameters are added; the pooling mode is a plain attribute.
* The module never reads labels.

"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

PROTOCOL_ID = "cscl-distance-pooling-bottleneck-v1"

#: frozen Soft5 kernel (never tuned; no y may enter its construction)
SOFT5_CENTERS: tuple[int, ...] = (1, 2, 3, 4, 8)
SOFT5_SIGMA: float = 0.35
N_BUCKETS: int = int(p2.DISTANCE_BUCKETS)  # 5

#: frozen Full-Y shape invariants
FULL_PARAMETERS: int = 408_651
READER_INPUT_DIM: int = 814
PAIR_VALUE_DIM: int = 48
UNARY_DIM: int = 289
RELATION_READOUT_DIM: int = 485
GRAPH_HIDDEN_DIM: int = 32
TOPOLOGY_DIM: int = 8

PAIR_POOLING_MODES: tuple[str, ...] = ("hard5", "soft5")


# ---------------------------------------------------------------------------
# distance basis
# ---------------------------------------------------------------------------


def soft5_weights_from_log_distance(log_distance: torch.Tensor) -> torch.Tensor:
    """Normalised fixed Gaussian RBF weights in log-distance space.

    ``log_distance`` is ``s(d) = log1p(d)`` (the stored
    ``pair_relation[:, 5]``).  Returns ``[N, 5]`` with non-negative finite
    entries and row sums one (to float32 rounding).
    """
    if log_distance.ndim != 1:
        raise ValueError(f"log_distance must be 1-D, got shape {tuple(log_distance.shape)}")
    value = log_distance.to(torch.float32)
    centers = torch.tensor(
        [math.log1p(float(c)) for c in SOFT5_CENTERS],
        dtype=value.dtype,
        device=value.device,
    )
    diff = value.unsqueeze(1) - centers.unsqueeze(0)
    amplitude = torch.exp(-(diff * diff) / (2.0 * float(SOFT5_SIGMA) ** 2))
    denominator = amplitude.sum(dim=1, keepdim=True)
    if not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any()):
        raise RuntimeError("Soft5 basis produced a non-positive / non-finite denominator")
    return amplitude / denominator


def hard5_weights_from_bucket(pair_bucket: torch.Tensor) -> torch.Tensor:
    """One-hot weights ``[N, 5]`` for the historical hard distance buckets.

    Diagnostic / test helper only: the Hard5 training path dispatches to the
    frozen :func:`audit.pool_pair_moments_masked` so it is bit-identical to the
    original forward.
    """
    bucket = pair_bucket.to(torch.long)
    if bucket.ndim != 1:
        raise ValueError(f"pair_bucket must be 1-D, got shape {tuple(bucket.shape)}")
    if bool(((bucket < 0) | (bucket >= N_BUCKETS)).any()):
        raise ValueError("pair_bucket outside [0, 5)")
    weights = torch.zeros((int(bucket.shape[0]), N_BUCKETS), dtype=torch.float32, device=bucket.device)
    weights.scatter_(1, bucket.unsqueeze(1), 1.0)
    return weights


def distance_pooling_weights(
    mode: str,
    pair_bucket: torch.Tensor,
    log_distance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch the frozen weights for one pooling mode."""
    if mode == "hard5":
        return hard5_weights_from_bucket(pair_bucket)
    if mode == "soft5":
        if log_distance is None:
            raise ValueError("soft5 pooling requires the log-distance coordinate")
        return soft5_weights_from_log_distance(log_distance)
    raise ValueError(f"unknown pair pooling mode {mode!r}; expected {PAIR_POOLING_MODES}")


# ---------------------------------------------------------------------------
# weighted moment pooling
# ---------------------------------------------------------------------------


def pool_pair_moments_weighted(
    value: torch.Tensor,
    pair_batch: torch.Tensor,
    weights: torch.Tensor,
    n_graphs: int,
    *,
    zero_count_blocks: bool = True,
    fill: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Weighted per-basis pair moments.

    ``first_k = sum_i w_ik h_i``, ``second_k = sum_i w_ik h_i^2`` (elementwise
    square of ``h``, then weighted) and ``count_k = sum_i w_ik`` stored as
    ``log1p``.  The count block is zeroed when ``zero_count_blocks`` is set
    (the C6_MASK semantics) — the soft weights must not silently restore the
    masked count channel.
    """
    value = value.to(torch.float32)
    pair_batch = pair_batch.to(torch.long)
    if weights.shape != (int(value.shape[0]), N_BUCKETS):
        raise ValueError(
            f"weights shape {tuple(weights.shape)} != ({int(value.shape[0])}, {N_BUCKETS})"
        )
    blocks: list[torch.Tensor] = []
    for basis in range(N_BUCKETS):
        weight = weights[:, basis: basis + 1]
        total = torch.zeros(
            (int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype
        )
        squared = torch.zeros_like(total)
        counts = torch.zeros((int(n_graphs), 1), device=value.device, dtype=value.dtype)
        if value.numel():
            total.index_add_(0, pair_batch, value * weight)
            squared.index_add_(0, pair_batch, (value * value) * weight)
            counts.index_add_(0, pair_batch, weight)
        count_block = torch.log1p(counts)
        if zero_count_blocks:
            count_block = audit._bucket_block_fill_or_zero(count_block, "pair:count", fill, 1, basis)
        blocks.append(torch.cat([total, squared, count_block], dim=1))
    return torch.cat(blocks, dim=1)


def pool_pair_moments_hard5(
    value: torch.Tensor,
    pair_batch: torch.Tensor,
    pair_bucket: torch.Tensor,
    n_graphs: int,
    *,
    zero_count_blocks: bool = True,
    fill: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Historical five-bucket moments (bit-identical to ``pool_pair_moments_masked``)."""
    zero_blocks: tuple[str, ...] = ("count",) if zero_count_blocks else ()
    return audit.pool_pair_moments_masked(
        value, pair_batch, pair_bucket, int(n_graphs), zero_blocks, fill
    )


def pool_pair_moments_soft5(
    value: torch.Tensor,
    pair_batch: torch.Tensor,
    log_distance: torch.Tensor,
    n_graphs: int,
    *,
    zero_count_blocks: bool = True,
    fill: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Soft5 continuous-distance weighted moments."""
    weights = soft5_weights_from_log_distance(log_distance)
    return pool_pair_moments_weighted(
        value,
        pair_batch,
        weights,
        int(n_graphs),
        zero_count_blocks=zero_count_blocks,
        fill=fill,
    )


def pooled_pair_features(
    pair_value: torch.Tensor,
    pair_batch: torch.Tensor,
    pair_bucket: torch.Tensor,
    log_distance: torch.Tensor,
    n_graphs: int,
    mode: str,
    *,
    zero_count_blocks: bool = True,
) -> torch.Tensor:
    """Unified audit entry point (same shapes for both modes)."""
    if mode == "hard5":
        return pool_pair_moments_hard5(
            pair_value,
            pair_batch,
            pair_bucket,
            n_graphs,
            zero_count_blocks=zero_count_blocks,
        )
    if mode == "soft5":
        return pool_pair_moments_soft5(
            pair_value,
            pair_batch,
            log_distance,
            n_graphs,
            zero_count_blocks=zero_count_blocks,
        )
    raise ValueError(f"unknown pair pooling mode {mode!r}")


# ---------------------------------------------------------------------------
# the model: frozen Full-Y with a pooling-mode switch
# ---------------------------------------------------------------------------


class DistancePoolingFullY(sc.LatentScaleSEM108):
    """Frozen Full-Y (``LatentScaleSEM108`` FULL) with a pair-pooling switch.

    ``pair_pooling`` is a plain string attribute, never a parameter or buffer,
    so the state dict is exactly the Full-Y state dict and construction
    consumes no extra RNG.  ``forward`` mirrors ``audit.AuditModel.forward``
    line by line; only the pair readout call is routed to
    :func:`pool_pair_moments_hard5` / :func:`pool_pair_moments_soft5`.
    """

    def __init__(self, *args: Any, pair_pooling: str = "hard5", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if pair_pooling not in PAIR_POOLING_MODES:
            raise ValueError(f"unknown pair_pooling {pair_pooling!r}; expected {PAIR_POOLING_MODES}")
        self.pair_pooling = str(pair_pooling)

    # -- forward (verbatim AuditModel.forward with one substituted call) -----

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
        unary = audit.pool_moments_masked(E, batch, n_graphs, mask.unary_zero_blocks, fill)

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
        zero_count = "count" in tuple(mask.pair_zero_blocks)
        if self.pair_pooling == "soft5":
            relation_readout = pool_pair_moments_soft5(
                pair_value,
                pair_batch,
                data.pair_relation[:, 5].to(pair_value.dtype),
                n_graphs,
                zero_count_blocks=zero_count,
                fill=fill,
            )
        else:
            relation_readout = pool_pair_moments_hard5(
                pair_value,
                pair_batch,
                data.pair_bucket,
                n_graphs,
                zero_count_blocks=zero_count,
                fill=fill,
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
        if int(unified.shape[1]) != READER_INPUT_DIM:
            raise RuntimeError(
                f"reader input width {int(unified.shape[1])} != frozen {READER_INPUT_DIM}"
            )
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
            }
        return prediction


def build_pooling_model(
    blob: Mapping[str, Any],
    seed: int = 0,
    *,
    mode: str = "hard5",
    scale_seed: int = 0,
    clean_spec: Any = None,
) -> DistancePoolingFullY:
    """Build a Full-Y model with the frozen RNG stream for the chosen pooling.

    Byte-for-byte the construction call of ``sc.build_scale_model`` +
    ``sc.FULL`` with ``clean_spec=cssd.CSSD_SPEC``; the only addition is the
    ``pair_pooling`` attribute, which consumes no RNG and adds no parameters.
    """
    clean_spec = cssd.CSSD_SPEC if clean_spec is None else clean_spec
    torch.manual_seed(int(seed))
    model = DistancePoolingFullY(
        cm.H1_CONFIG,
        np.asarray(blob["D_fit"], np.float32),
        subspace=zjd.fold_subspace(blob),
        spec=sc.FULL,
        node_binding=clean_spec.node_binding,
        edge_binding=clean_spec.edge_binding,
        coding=clean_spec.coding,
        scale_seed=int(scale_seed),
        pair_pooling=mode,
    )
    parameter_audit = sc.scale_parameter_audit(model)
    if int(parameter_audit["actual_parameters"]) != FULL_PARAMETERS or not parameter_audit["parameter_exact"]:
        raise RuntimeError(f"parameter audit failed: {parameter_audit}")
    return model


__all__ = [
    "PROTOCOL_ID",
    "SOFT5_CENTERS",
    "SOFT5_SIGMA",
    "N_BUCKETS",
    "FULL_PARAMETERS",
    "READER_INPUT_DIM",
    "PAIR_VALUE_DIM",
    "UNARY_DIM",
    "RELATION_READOUT_DIM",
    "GRAPH_HIDDEN_DIM",
    "TOPOLOGY_DIM",
    "PAIR_POOLING_MODES",
    "soft5_weights_from_log_distance",
    "hard5_weights_from_bucket",
    "distance_pooling_weights",
    "pool_pair_moments_weighted",
    "pool_pair_moments_hard5",
    "pool_pair_moments_soft5",
    "pooled_pair_features",
    "DistancePoolingFullY",
    "build_pooling_model",
]
