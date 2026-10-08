"""ZINC CSSD relation projection v1: does the relation side read the endpoint
projection BEFORE the local representation is compressed?

Round: ``zinc_cssd_relation_projection_v1`` (single body seed 0, exploratory).

The frozen CSSD basis and its effective local access are held fixed (the
verified DICT consumer of ``zinc_cssd_consumer_replacement_v1``).  In that
consumer every environment pair first shares ONE projection
``P: E_i -> 48`` (``pair_projection``, Linear 144->48, no bias); only then are
the endpoint sum / absolute difference / gated product computed, the relation
encoded, and the whole 192-D pair input pushed through the pair encoder and
pooled by the original distance buckets.  Hypothesis under test (a budgeted
discovery probe, NOT a proven bottleneck): different relations may need
different local structure-semantic information, so letting the EXISTING
distance buckets select the endpoint projection may beat compressing first.

Three arms, one body seed (0), one from-scratch 240-epoch training each:

* ``BASE``          — the untouched historical DICT consumer (shared no-bias
  Linear(144,48); 6,912 projection parameters, 297,539 total).  Its init and
  schedule must reproduce the historical seed-0 run bit-for-bit.
* ``SHARED_MLP`` (C) — ONLY the endpoint projection replaced by a shared
  no-bias 144->180->48 MLP with SiLU (34,560 projection parameters, 325,187
  total).  Still one projection for every distance bucket; it reads no
  distance, no relation and no graph id.  Fixed function-preserving init
  (private seed 20261008 under ``fork_rng``): W1[0:48]=P0, W1[48:96]=-P0,
  W1[96:180]=default init; W2[:,0:48]=I48, W2[:,48:96]=-I48, W2[:,96:180]=0,
  so the initial function is P0·E (SiLU(x)-SiLU(-x)=x) up to float rounding.
* ``BUCKET_LINEAR`` (R) — five no-bias projections P_b[48,144], one per
  EXISTING distance bucket (34,560 projection parameters, 325,187 total).
  For a pair (i,j) the endpoint views are P_b(E_i), P_b(E_j) with b the
  pair's projection ROUTE (normally ``data.pair_bucket``); both endpoints
  share P_b (endpoint-swap symmetric).  Every P_b starts as a bitwise copy
  of P0, so step 0 is function-identical to BASE.

The relation input, the distance gate, the pair readout pooling and Q keep
using ``data.pair_bucket`` unchanged; only the endpoint views are routed.
The route override (terminal intervention 2) never writes ``data.pair_bucket``
and never touches E_i: the projection output is only a relation-side view.

Everything else is the historical recipe: CSSD phi_hat decode frozen, Sem108 +
size2 interface, real J incidence, kappa, fusion, the 144->288->144 posterior
MLP bridge, reader, C6 mask, COMP supervision, Adam lr 1e-3 / coupled wd 1e-5 /
batch 128 / clip 5 / FP32, 240 epochs = 15,120 steps, estimator = the
equal-weight FP32 mean of the full member states 236..240.  Fit = the source
round's 8001-row fold; dev = the 1999 historical development rows (a
development comparison, never a new confirm); official valid/test never
instantiated.  Terminal (one-shot): full dev y_raw MAE (primary) and g_raw MAE
(secondary) with the three paired contrasts R-BASE, C-BASE, R-C, the
group-paired bootstrap (2000 draws, seed 20261008, shared picks across the
three arms and both metrics), the noise bound, the unique alpha-mean
dictionary intervention on all three soups, and the route-shuffle
intervention with one fixed override set shared by all three arms.  Single
seed: a directional signal only — never a repeatability or method-stability
claim; the historical B72 A-E gates are NOT inherited.

Usage (local CPU for source/pretrain checks; res-2 res2-cu124 for the GPU
stages, all through the registered runner)::

    python -m tracks.ksvd.experiments.luyin16.\\
zinc_cssd_relation_projection_v1 --stage source-checks
    ... --stage pretrain-checks
    ... --stage smoke --device cuda:0
    ... --stage train --arm BASE --seed 0 --device cuda:0
    ... --stage terminal-eval --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import (
    zinc_chemistry_component_supervision_seed0_v1 as zcs,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_basis_reuse_v1 as zreuse,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_consumer_generalization_v1 as gen,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_consumer_replacement_v1 as repl,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_nonlinear_binding_v1 as nb,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_full_cycle_target_decomposition_v1 as zftd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_vs_mlp_seed0_v1 as mlpmod,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw,
)

PROTOCOL_VERSION = "zinc-cssd-relation-projection-v1"
RESULT_SLUG = "zinc_cssd_relation_projection_v1"
TRACK_ROOT = repl.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG
SOURCE_DIR = repl.SOURCE_DIR

ARMS = ("BASE", "SHARED_MLP", "BUCKET_LINEAR")
CONTROL = "BASE"
CANDIDATES = ("SHARED_MLP", "BUCKET_LINEAR")
SEEDS = (0,)
SEED = 0

EPOCHS = repl.EPOCHS                    # 240
LR = repl.LR                            # 1e-3
WEIGHT_DECAY = repl.WEIGHT_DECAY        # 1e-5 (coupled)
GRAD_CLIP = repl.GRAD_CLIP              # 5.0
BATCH_SIZE = repl.BATCH_SIZE            # 128
SOUP_EPOCHS = repl.SOUP_EPOCHS          # 236..240 fixed five-epoch soup
LOG_EPOCHS = repl.LOG_EPOCHS            # (1, 40, 120, 240)
TRAIN_SHUFFLE_OFFSET = repl.TRAIN_SHUFFLE_OFFSET  # 101
COMPONENT_LOSS_WEIGHT = repl.COMPONENT_LOSS_WEIGHT  # 0.5
STEPS_EXPECTED = 15120

#: the endpoint projection lives on pair_projection (module name preserved);
#: the historical shared Linear is 144->48 no-bias = 6,912 parameters
PROJECTION_IN = int(sc.FULL.d)          # 144
PROJECTION_OUT = int(sc.FULL.p)         # 48
N_BUCKETS = int(p2.DISTANCE_BUCKETS)    # 5 (the EXISTING buckets, never remapped)
SHARED_MLP_HIDDEN = 180
BASE_PROJECTION_PARAMETERS = PROJECTION_IN * PROJECTION_OUT                  # 6,912
SHARED_MLP_PROJECTION_PARAMETERS = PROJECTION_IN * SHARED_MLP_HIDDEN + SHARED_MLP_HIDDEN * PROJECTION_OUT  # 34,560
BUCKET_PROJECTION_PARAMETERS = N_BUCKETS * PROJECTION_IN * PROJECTION_OUT    # 34,560

#: parameter contract of THIS round (repl component audit + per-arm projection)
EXPECTED_BASE_AUDIT = {
    "total_parameters": 297_539,
    "base_body_parameters": 184_707,
    "bridge_parameters": 82_944,
    "local_tuple_parameters": 29_888,
    "reader_output_parameters": 2 * 39 + 2,
}
DELTA_PROJECTION = SHARED_MLP_PROJECTION_PARAMETERS - BASE_PROJECTION_PARAMETERS  # +27,648
EXPECTED_CANDIDATE_AUDIT = {
    "total_parameters": 297_539 + DELTA_PROJECTION,                     # 325,187
    "base_body_parameters": 184_707 + DELTA_PROJECTION,                 # 212,355
    "bridge_parameters": 82_944,
    "local_tuple_parameters": 29_888,
    "reader_output_parameters": 2 * 39 + 2,
}

#: fixed, function-preserving private init stream for the C/R modules
PROJ_INIT_SEED = 20261008
#: the terminal route-shuffle stream (one fixed override set for all arms)
ROUTE_SHUFFLE_SEED = 20261008

N_BOOT = 2000
BOOT_SEED = 20261008
REPLAY_TOL = 1e-4
CROSS_DEVICE_TOL = 1e-5
#: CPU step-0 bands: R is mathematically identical to BASE (bit-exact on CPU);
#: C differs only by SiLU/addition rounding
STEP0_BAND_C = 1e-4
STEP0_BAND_C_GPU = 5e-5 * 10  # GPU band for C (kernel variation), still tiny
FLOAT64_SUM_TOL = 1e-4
#: historical descriptive anchor (replacement round, DICT seed 0 dev y_raw)
HISTORICAL_DICT_S0_Y_RAW = 0.12151750804669596

write_json = repl.write_json
read_json = repl.read_json
file_sha256 = repl.file_sha256
array_sha256 = repl.array_sha256
state_hash = repl.state_hash
seed_everything = repl.seed_everything
resolve_device = repl.resolve_device
load_round_objects = repl.load_round_objects
build_round_data = repl.build_round_data
frozen_basis_parts = repl.frozen_basis_parts
fit_mean_codes = repl.fit_mean_codes


def arm_parameter_audit(model: nn.Module) -> dict[str, int]:
    return zcs.component_parameter_audit(model)


def arm_expected_audit(arm: str) -> dict[str, int]:
    if arm == "BASE":
        return dict(EXPECTED_BASE_AUDIT)
    return dict(EXPECTED_CANDIDATE_AUDIT)


def projection_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k.startswith("pair_projection.")}


def non_projection_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if not k.startswith("pair_projection.")}


# ---------------------------------------------------------------------------
# 1. the endpoint projections and the round-local model class
# ---------------------------------------------------------------------------


class SharedEndpointMLP(nn.Module):
    """SHARED_MLP endpoint projection: 144 -> 180 -> 48, SiLU, no bias.

    One projection for every distance bucket; it reads no distance, no
    relation and no graph id.  Fixed function-preserving init: the first two
    48-row blocks of W1 carry ``+P0`` and ``-P0`` and the matching W2 columns
    carry ``+I`` and ``-I``, so ``W2 silu(W1 x) = silu(P0 x) - silu(-P0 x) =
    P0 x`` (the supplementary rows W1[96:180] keep the private-stream default
    init and are multiplied by zero W2 columns at step 0; their first-step
    gradient is zero BY CONSTRUCTION — after one optimizer step W2's zero
    columns move and the rows start training, checked in the gradient smoke,
    never misread as dead).
    """

    def __init__(self, w0: torch.Tensor, *, hidden: int = SHARED_MLP_HIDDEN) -> None:
        super().__init__()
        if tuple(w0.shape) != (PROJECTION_OUT, PROJECTION_IN):
            raise ValueError(f"P0 shape {tuple(w0.shape)} != {(PROJECTION_OUT, PROJECTION_IN)}")
        self.w1 = nn.Linear(PROJECTION_IN, int(hidden), bias=False)
        self.w2 = nn.Linear(int(hidden), PROJECTION_OUT, bias=False)
        with torch.no_grad():
            # rows 0:96 are overwritten; rows 96:180 keep the private-stream
            # default reset_parameters draw; w2 is fully overwritten
            self.w1.weight[:PROJECTION_OUT].copy_(w0)
            self.w1.weight[PROJECTION_OUT : 2 * PROJECTION_OUT].copy_(-w0)
            identity = torch.eye(PROJECTION_OUT)
            self.w2.weight[:, :PROJECTION_OUT].copy_(identity)
            self.w2.weight[:, PROJECTION_OUT : 2 * PROJECTION_OUT].copy_(-identity)
            self.w2.weight[:, 2 * PROJECTION_OUT :].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


class BucketEndpointProjection(nn.Module):
    """BUCKET_LINEAR: one no-bias P_b[48,144] per EXISTING distance bucket.

    Every P_b starts as a bitwise copy of the historical shared P0, so the
    step-0 function equals BASE regardless of the route.  The buckets are the
    original five (never added, removed or remapped); the route is read by the
    model's forward, never by this module.
    """

    def __init__(self, w0: torch.Tensor, n_buckets: int = N_BUCKETS) -> None:
        super().__init__()
        if tuple(w0.shape) != (PROJECTION_OUT, PROJECTION_IN):
            raise ValueError(f"P0 shape {tuple(w0.shape)} != {(PROJECTION_OUT, PROJECTION_IN)}")
        self.n_buckets = int(n_buckets)
        self.layers = nn.ModuleList(
            [nn.Linear(PROJECTION_IN, PROJECTION_OUT, bias=False) for _ in range(int(n_buckets))]
        )
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(w0)


class RelationProjectionFullM(mlpmod.LocalTupleFullM):
    """The historical DICT consumer with an explicit endpoint-view interface.

    Construction is EXACTLY ``repl.build_arm("DICT", ...)`` (this round's
    factory promotes the instance's class and, for the candidate arms, swaps
    ``model.pair_projection`` under a private ``fork_rng`` stream).  The
    masked forward below is the historical ``AuditModel.forward`` path with
    ONLY the pair-branch endpoint views replaced by :meth:`endpoint_views`;
    every other operation, the mask/fill semantics and all module names are
    unchanged.  ``route_override`` (terminal route intervention only) reroutes
    which P_b reads each pair's endpoints; the relation input, the distance
    gate and the pair readout ALWAYS read ``data.pair_bucket`` (never the
    override), ``data.pair_bucket`` is never written, and E_i is generated
    once, before any pair conditioning, and is never updated by the pairs.
    """

    #: "shared" (BASE/SHARED_MLP: one projection for all pairs) or "bucket" (R)
    endpoint_mode: str = "shared"
    #: terminal route intervention only; a [P_batch] LongTensor or None
    route_override: torch.Tensor | None = None
    #: aggregate stats of the last route-overridden bucket forward (detached)
    last_route_stats: dict[str, Any] | None = None

    def pair_route(self, data: Any) -> torch.Tensor:
        if self.route_override is None:
            return data.pair_bucket
        return self.route_override.to(
            device=data.pair_bucket.device, dtype=data.pair_bucket.dtype
        )

    def endpoint_views(
        self,
        E: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        data: Any,
        mask: audit.AuditMask,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The pair-branch endpoint views — the ONLY routed part of the model."""
        if self.endpoint_mode == "bucket":
            views = torch.stack([layer(E) for layer in self.pair_projection.layers], dim=0)
            if mask.pair_projection_zero:
                views = torch.zeros_like(views)
            route = self.pair_route(data)
            left = views[route, source]
            right = views[route, target]
            self._record_route_stats(views, route, data, source, target, views_zeroed=bool(mask.pair_projection_zero))
            return left, right
        u = self.pair_projection(E)
        if mask.pair_projection_zero:
            u = torch.zeros_like(u)
        return u[source], u[target]

    @torch.no_grad()
    def _record_route_stats(
        self,
        views: torch.Tensor,
        route: torch.Tensor,
        data: Any,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        views_zeroed: bool,
    ) -> None:
        """Detach-only diagnostics while an override is active (eval use)."""
        if self.route_override is None:
            self.last_route_stats = None
            return
        natural = data.pair_bucket
        changed = route != natural
        n_pairs = int(route.numel())
        n_changed = int(changed.sum())
        stats: dict[str, Any] = {
            "n_pairs": n_pairs,
            "n_pairs_route_changed": n_changed,
            "fraction_route_changed": float(n_changed) / float(max(n_pairs, 1)),
        }
        if n_changed and not views_zeroed:
            sel = changed.nonzero(as_tuple=False).view(-1)
            v_nat = views[natural[sel], source[sel]]
            v_new = views[route[sel], source[sel]]
            rel = (v_new - v_nat).norm(dim=1) / (v_nat.norm(dim=1) + 1e-12)
            stats.update({
                "source_view_rel_change_median": float(rel.median()),
                "source_view_rel_change_mean": float(rel.mean()),
                "source_view_rel_change_max": float(rel.max()),
            })
        self.last_route_stats = stats

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
        left, right = self.endpoint_views(E, source, target, data, mask)
        relation_input = data.pair_relation[:, list(p1.P1_RELATION_INDICES)]
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
        relation_readout = audit.pool_pair_moments_masked(
            pair_value, pair_batch, data.pair_bucket, n_graphs, mask.pair_zero_blocks, fill
        )
        if readout_perm is not None:
            blocks, permutation = readout_perm
            unary, relation_readout = audit.apply_readout_permutation(unary, relation_readout, blocks, permutation)

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
        prediction = self.reader(unified).view(-1)
        if return_aux:
            return prediction, {
                "E": E,
                "coord": coord,
                "phi": data.dict_phi,
                "unary": unary,
                "relation_readout": relation_readout,
            }
        return prediction


_ENDPOINT_MODES = {"BASE": "shared", "SHARED_MLP": "shared", "BUCKET_LINEAR": "bucket"}


def build_arm_round(
    arm: str,
    payload: prev.TuplePayload,
    kappa_M: float,
    basis_parts: Mapping[str, torch.Tensor],
    seed: int,
    *,
    decode: bool = True,
) -> RelationProjectionFullM:
    """Fresh untrained arm of THIS round, body seed explicitly penetrated.

    Every arm starts from the ORIGINAL untrained DICT factory
    ``repl.build_arm("DICT", ...)``, which first completes its own historical
    297,539-parameter audit and consumes the historical constructor stream
    with the body seed scoped exactly like the source round.  BASE then only
    changes the instance's class (no module, no tensor).  SHARED_MLP and
    BUCKET_LINEAR additionally swap ``model.pair_projection`` on CPU under
    ``torch.random.fork_rng(devices=[])`` with the private seed 20261008; the
    fork restores the global CPU RNG, so the post-construction global RNG
    state is identical across arms for the same seed and every NON-projection
    initial tensor is bit-identical by construction.  R's five P_b are bitwise
    copies of the factory's P0.  The per-arm parameter contract of THIS round
    is asserted afterwards (the historical all-arms-297,539 assertion is NOT
    inherited).
    """
    if arm not in ARMS:
        raise ValueError(arm)
    model = repl.build_arm("DICT", payload, kappa_M, basis_parts, int(seed), decode=decode)
    model.__class__ = RelationProjectionFullM
    model.endpoint_mode = _ENDPOINT_MODES[arm]
    model.route_override = None
    model.last_route_stats = None
    if arm != "BASE":
        w0 = model.pair_projection.weight.detach().clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(PROJ_INIT_SEED)
            if arm == "SHARED_MLP":
                model.pair_projection = SharedEndpointMLP(w0)
            else:
                model.pair_projection = BucketEndpointProjection(w0)
    audit_row = arm_parameter_audit(model)
    expected = arm_expected_audit(arm)
    if audit_row != expected:
        raise RuntimeError(f"{arm} parameter audit failed: {audit_row} != {expected}")
    if not isinstance(model.local_tuple, repl.LocalTupleEncoderMCSSD):
        raise RuntimeError(f"{arm}: the CSSD encoder is missing")
    if arm == "BASE" and not isinstance(model.pair_projection, nn.Linear):
        raise RuntimeError("BASE pair_projection must remain the historical shared Linear")
    if arm == "SHARED_MLP" and not isinstance(model.pair_projection, SharedEndpointMLP):
        raise RuntimeError("SHARED_MLP pair_projection must be the SharedEndpointMLP")
    if arm == "BUCKET_LINEAR" and not isinstance(model.pair_projection, BucketEndpointProjection):
        raise RuntimeError("BUCKET_LINEAR pair_projection must be the BucketEndpointProjection")
    return model


def run_dir(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    if arm not in ARMS:
        raise ValueError(arm)
    return Path(out_dir) / "runs" / f"{arm}_s{int(seed)}"


def load_run(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """One run's manifest + soup state (hash-checked; ALWAYS reloaded through
    THIS round's factory — never the historical DICT/CTRL288/B72 loaders)."""
    rdir = run_dir(arm, seed, out_dir)
    manifest = read_json(rdir / "manifest.json")
    if manifest["arm"] != arm or int(manifest["seed"]) != int(seed):
        raise RuntimeError(f"run dir {rdir} is not {arm} s{seed}")
    soup_state = torch.load(rdir / "soup_state.pt", map_location="cpu", weights_only=False)
    if state_hash(soup_state) != manifest["soup_state_sha256"]:
        raise RuntimeError(f"soup state hash mismatch for {arm} s{seed}")
    return {"manifest": manifest, "soup_state": soup_state, "run_dir": rdir}


# ---------------------------------------------------------------------------
# 2. stage: source-checks (CPU, read-only)
# ---------------------------------------------------------------------------


def source_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Verify every reused object, the historical seed-0 run, the dimension /
    bucket contracts and the per-arm init contracts (fit-only, read-only)."""
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ctx = gen.load_round_context()
    objects = ctx["objects"]
    basis_parts = ctx["basis_parts"]
    payload, kappa_M = ctx["payload"], ctx["kappa_M"]
    fit_idx, dev_idx = ctx["fit_idx"], ctx["dev_idx"]
    n_fit = int(fit_idx.size)

    # (a) every stage-0 artifact hash against the replacement round's
    #     committed source manifest (the read-only source of this line)
    src = read_json(repl.RESULTS_DIR / "source_manifest.json")
    artifact_hashes = {name: file_sha256(SOURCE_DIR / name) for name in src["reused_artifacts"]}
    if artifact_hashes != src["reused_artifacts"]:
        raise RuntimeError("stage-0 artifact hashes differ from the committed source manifest")
    if src["cssd_basis"]["U_sha256"] != array_sha256(ctx["basis"]["U"]) or \
        src["cssd_basis"]["D_sha256"] != array_sha256(ctx["basis"]["D"]):
        raise RuntimeError("CSSD basis hashes differ from the committed source manifest")
    if src["fold_view"]["fit"]["sha256"] != array_sha256(fit_idx, np.int64) or \
        src["fold_view"]["dev"]["sha256"] != array_sha256(dev_idx, np.int64):
        raise RuntimeError("round fold view differs from the committed source manifest")

    # (b) the historical DICT seed-0 run (read-only, hash-checked)
    old = gen.load_old_run(0)

    # (c) schedule reproduction: this round's 240-epoch plan (seed 0) must be
    #     bit-identical to the historical DICT schedule
    schedule, schedule_hash = zw.build_schedule(n_fit, EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    if not all(np.array_equal(schedule[e], old["schedule_order"][e]) for e in range(EPOCHS)):
        raise RuntimeError("schedule does not reproduce the historical plan (seed 0)")
    if schedule_hash != old["manifest"]["schedule_sha256"]:
        raise RuntimeError("schedule hash mismatch vs historical run (seed 0)")

    # (d) dimension / bucket contracts of the fixed interface
    sample_row = ctx["fit_data"][0]
    dim_contract = {
        "d_environments": int(PROJECTION_IN),
        "p_endpoints": int(PROJECTION_OUT),
        "distance_buckets": int(N_BUCKETS),
        "pair_projection_weight_shape": [PROJECTION_OUT, PROJECTION_IN],
        "relation_width": int(p1.RELATION_WIDTH),
        "reader_input": int(sc.FULL.reader_input),
        "fit_n": n_fit,
        "dev_n": int(dev_idx.size),
        "row_pair_bucket_values": sorted(int(v) for v in set(sample_row.pair_bucket.tolist())),
    }
    if int(sc.FULL.d) != 144 or int(sc.FULL.p) != 48 or int(p2.DISTANCE_BUCKETS) != 5:
        raise RuntimeError("the fixed interface dimensions changed (d/p/buckets)")
    if int(sc.FULL.reader_input) != 814:
        raise RuntimeError("reader input width changed")
    if int(sample_row.pair_bucket.max()) >= N_BUCKETS or int(sample_row.pair_bucket.min()) < 0:
        raise RuntimeError("row pair buckets outside the five existing buckets")

    # (e) init contracts: BASE reproduces the historical DICT s0 init
    #     bit-for-bit; C/R share every non-projection tensor with BASE and
    #     their projection swap is fork_rng RNG-neutral; the fixed identity
    #     init of C and the P0 copies of R hold
    init_contracts: dict[str, Any] = {}
    seed_everything(SEED)
    base = build_arm_round("BASE", payload, kappa_M, basis_parts, SEED)
    base_state = gen._capture_state(base)
    base_hash = state_hash(base_state)
    if base_hash != old["manifest"]["init_state_sha256"]:
        raise RuntimeError("BASE init does not reproduce the historical DICT s0 init")
    rng_after_base = torch.get_rng_state().clone()
    states: dict[str, dict[str, torch.Tensor]] = {"BASE": base_state}
    for arm in CANDIDATES:
        seed_everything(SEED)
        cand = build_arm_round(arm, payload, kappa_M, basis_parts, SEED)
        if not torch.equal(rng_after_base, torch.get_rng_state()):
            raise RuntimeError(f"{arm} construction disturbed the global CPU RNG")
        cand_state = gen._capture_state(cand)
        states[arm] = cand_state
        nb_base, nb_cand = non_projection_state(base_state), non_projection_state(cand_state)
        if sorted(nb_base) != sorted(nb_cand):
            raise RuntimeError(f"{arm}: non-projection state key sets differ vs BASE")
        mismatched = [k for k in nb_base if not torch.equal(nb_base[k], nb_cand[k])]
        if mismatched:
            raise RuntimeError(f"{arm}: non-projection init tensors differ vs BASE at {mismatched}")
    proj = projection_state(states["SHARED_MLP"])
    w0 = states["BASE"]["pair_projection.weight"]
    ident = torch.eye(PROJECTION_OUT)
    c_checks = {
        "w1_rows_plus_p0": bool(torch.equal(proj["pair_projection.w1.weight"][:PROJECTION_OUT], w0)),
        "w1_rows_minus_p0": bool(
            torch.equal(proj["pair_projection.w1.weight"][PROJECTION_OUT : 2 * PROJECTION_OUT], -w0)
        ),
        "w2_cols_plus_identity": bool(torch.equal(proj["pair_projection.w2.weight"][:, :PROJECTION_OUT], ident)),
        "w2_cols_minus_identity": bool(
            torch.equal(proj["pair_projection.w2.weight"][:, PROJECTION_OUT : 2 * PROJECTION_OUT], -ident)
        ),
        "w2_supplementary_zero": bool(
            torch.equal(proj["pair_projection.w2.weight"][:, 2 * PROJECTION_OUT :], torch.zeros(
                PROJECTION_OUT, SHARED_MLP_HIDDEN - 2 * PROJECTION_OUT))
        ),
        "no_bias": True,
    }
    if not all(c_checks.values()):
        raise RuntimeError(f"SHARED_MLP init identity failed: {c_checks}")
    proj_r = projection_state(states["BUCKET_LINEAR"])
    r_checks = {
        f"bucket_{b}_bitwise_p0": bool(torch.equal(proj_r[f"pair_projection.layers.{b}.weight"], w0))
        for b in range(N_BUCKETS)
    }
    if not all(r_checks.values()):
        raise RuntimeError(f"BUCKET_LINEAR P0 copies failed: {r_checks}")
    init_contracts = {
        "base_init_sha256": base_hash,
        "base_matches_historical_dict_s0": True,
        "non_projection_bitwise_equal_across_arms": True,
        "shared_mlp_identity": c_checks,
        "bucket_p0_copies": r_checks,
        "shared_mlp_projection_sha256": state_hash(projection_state(states["SHARED_MLP"])),
        "bucket_projection_sha256": state_hash(projection_state(states["BUCKET_LINEAR"])),
        "rng_stream_identical_after_construction": True,
        "projection_init_private_seed": PROJ_INIT_SEED,
    }
    if state_hash(projection_state(states["SHARED_MLP"])) == state_hash(projection_state(states["BUCKET_LINEAR"])):
        raise RuntimeError("C and R projection inits collide (expected different modules)")

    # (f) frozen basis / Q consistency with the historical run
    if old["manifest"]["frozen_basis_hashes"] != basis_parts["hashes"]:
        raise RuntimeError("frozen basis hashes differ from the historical run")
    if state_hash(ctx["q_soup"]) != old["manifest"]["q_soup_sha256"]:
        raise RuntimeError("Q soup hash differs from the historical run")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "source-checks",
        "source_dir": str(SOURCE_DIR.relative_to(TRACK_ROOT)),
        "source_read_only": True,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "dim_contract": dim_contract,
        "historical_schedule_sha256": old["manifest"]["schedule_sha256"],
        "rebuilt_schedule_sha256": schedule_hash,
        "schedule_bit_identical_240": True,
        "init_contracts": init_contracts,
        "parameter_contracts": {arm: arm_expected_audit(arm) for arm in ARMS},
        "projection_parameters": {
            "BASE": BASE_PROJECTION_PARAMETERS,
            "SHARED_MLP": SHARED_MLP_PROJECTION_PARAMETERS,
            "BUCKET_LINEAR": BUCKET_PROJECTION_PARAMETERS,
        },
        "frozen_basis_hashes": basis_parts["hashes"],
        "q_soup_sha256": state_hash(ctx["q_soup"]),
        "old_run": {
            "run_dir": str(old["rdir"].relative_to(TRACK_ROOT)),
            "manifest_hashes_verified": True,
            "init_state_sha256": old["manifest"]["init_state_sha256"],
            "schedule_sha256": old["manifest"]["schedule_sha256"],
        },
        "reuse_note": (
            "no basis/Q/prep/target refit; the historical DICT s0 trajectory is a "
            "read-only anchor; official valid/test never loaded"
        ),
        "all_passed": True,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "source_manifest.json", manifest)
    log(f"[source-checks] fit={n_fit} dev={dev_idx.size} verified in {manifest['seconds']:.1f}s")
    return manifest


# ---------------------------------------------------------------------------
# 3. stage: pretrain-checks (CPU, fit rows only — never dev, never a score)
# ---------------------------------------------------------------------------


def _projection_grad_norms(model: RelationProjectionFullM) -> dict[str, Any]:
    proj = model.pair_projection
    if isinstance(proj, SharedEndpointMLP):
        g = proj.w1.weight.grad
        h = proj.w2.weight.grad
        out: dict[str, Any] = {
            "w1_grad_norm_total": float(g.norm()) if g is not None else None,
            "w1_rows_0_48_grad_norm": float(g[:PROJECTION_OUT].norm()) if g is not None else None,
            "w1_rows_48_96_grad_norm": float(
                g[PROJECTION_OUT : 2 * PROJECTION_OUT].norm()) if g is not None else None,
            "w1_rows_96_180_grad_norm": float(
                g[2 * PROJECTION_OUT :].norm()) if g is not None else None,
            "w2_grad_norm": float(h.norm()) if h is not None else None,
            "w2_supplementary_cols_grad_norm": float(
                h[:, 2 * PROJECTION_OUT :].norm()) if h is not None else None,
        }
        return out
    if isinstance(proj, BucketEndpointProjection):
        rows: dict[str, Any] = {}
        for b, layer in enumerate(proj.layers):
            gb = layer.weight.grad
            rows[f"bucket_{b}_grad_norm"] = float(gb.norm()) if gb is not None else None
        return rows
    g = proj.weight.grad
    return {"linear_grad_norm": float(g.norm()) if g is not None else None}


def pretrain_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """The meaningful fit-only pre-training checks of this round.

    (a) BASE parity: this round's BASE forward/component path is bit-identical
        to the untouched replacement factory on fit batches;
    (b) projection contracts (shapes/no-bias/counts) + the fixed identity
        init of C (C(E) ~= P0 E on real fit-batch environments) and the P0
        bitwise copies of R;
    (c) step-0 functional agreement of all three arms on fit batches
        (R bit-exact vs BASE on CPU; C within the SiLU-rounding band) with the
        endpoint view tensors checked directly;
    (d) same-forward component identity (torch exact, float64 sum in band);
    (e) endpoint-swap symmetry of the pair input; pair-row permutation, node
        relabelling and batch-composition/label invariance on fit molecules;
    (f) a small-batch fit-only optimization smoke: finite outputs/loss,
        gradients reaching the bridge, W_loc, A_raw, reader and the projection
        modules (C's supplementary W1 rows are zero-grad at step 0 BY
        CONSTRUCTION and must receive gradient by step 3; R's per-bucket
        gradients are read together with the batch's bucket coverage), CSSD
        buffers never in the optimizer/state dict, basis hashes unchanged;
    (g) save -> correct-factory reload -> replay (CPU bit-exact) and
        wrong-factory strict load blocked;
    (h) route-override wiring: BASE/C do not read the route (bit-exact under
        override), R with equal P_b is override-invariant, R with an
        artificially different P does respond at the endpoint views, and
        data.pair_bucket is never modified.
    """
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cpu")
    ctx = gen.load_round_context()
    objects = ctx["objects"]
    basis_parts = ctx["basis_parts"]
    payload, kappa_M = ctx["payload"], ctx["kappa_M"]
    fit_idx, fit_data = ctx["fit_idx"], ctx["fit_data"]
    g_all = np.asarray(objects["targets"]["g"], np.float64)
    ell_all = np.asarray(objects["targets"]["ell"], np.float64)
    s_all = np.asarray(objects["targets"]["s"], np.float64)
    target_g = torch.as_tensor(g_all[fit_idx], dtype=torch.float32)
    target_ell = torch.as_tensor(ell_all[fit_idx], dtype=torch.float32)
    target_s = torch.as_tensor(s_all[fit_idx], dtype=torch.float32)
    checks: dict[str, Any] = {}
    smoke_rows = list(range(16))

    def build(arm: str, seed: int = SEED) -> RelationProjectionFullM:
        seed_everything(int(seed))
        return build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))

    # (a) BASE parity vs the untouched replacement factory (bit-exact)
    base = build("BASE")
    raw_factory = repl.build_arm("DICT", payload, kappa_M, basis_parts, SEED, decode=True)
    base_state = gen._capture_state(base)
    raw_state = gen._capture_state(raw_factory)
    if sorted(base_state) != sorted(raw_state):
        raise RuntimeError("BASE state keys differ from the replacement factory")
    if state_hash(base_state) != state_hash(raw_state):
        raise RuntimeError("BASE init tensors differ from the replacement factory")
    batch16 = zftd.make_batch(fit_data, smoke_rows, target_g, device)
    base.eval(); raw_factory.eval()
    with torch.no_grad():
        p_base = base(batch16, mask=cm.C6_MASK)
        c_base = base.reader.components()
        p_raw = raw_factory(batch16, mask=cm.C6_MASK)
        c_raw = raw_factory.reader.components()
    parity = {
        "prediction_max_abs": float((p_base - p_raw).abs().max()),
        "components_max_abs": float((c_base - c_raw).abs().max()),
    }
    if parity["prediction_max_abs"] != 0.0 or parity["components_max_abs"] != 0.0:
        raise RuntimeError(f"BASE forward is not bit-identical to the historical path: {parity}")
    checks["base_parity_vs_historical_forward"] = parity

    # (b) projection contracts + C(E) ~= P0 E on real environments
    contracts: dict[str, Any] = {}
    with torch.no_grad():
        pred, aux = base(batch16, mask=cm.C6_MASK, return_aux=True)
        E_fit = aux["E"]
        p0_view = F.linear(E_fit, base_state["pair_projection.weight"])
        c_model = build("SHARED_MLP")
        c_model.eval()
        c_view = c_model.pair_projection(E_fit)
        d_c = float((c_view - p0_view).abs().max())
        rel_c = float(((c_view - p0_view).norm(dim=1) / (p0_view.norm(dim=1) + 1e-12)).max())
        rand_x = torch.randn(256, PROJECTION_IN, generator=torch.Generator().manual_seed(7)) * 3.0
        d_rand = float(
            (c_model.pair_projection(rand_x) - F.linear(rand_x, base_state["pair_projection.weight"])).abs().max()
        )
    contracts["shared_mlp"] = {
        "shapes": [list(c_model.pair_projection.w1.weight.shape), list(c_model.pair_projection.w2.weight.shape)],
        "no_bias": bool(c_model.pair_projection.w1.bias is None and c_model.pair_projection.w2.bias is None),
        "parameters": int(sum(p.numel() for p in c_model.pair_projection.parameters())),
        "fit_E_max_abs_vs_p0": d_c,
        "fit_E_max_rel_vs_p0": rel_c,
        "random_x_max_abs_vs_p0": d_rand,
        "w1_rows_96_180_default_init_nonzero": bool(
            float(c_model.pair_projection.w1.weight[2 * PROJECTION_OUT :].abs().sum()) > 0.0
        ),
    }
    if d_c > STEP0_BAND_C or rel_c > STEP0_BAND_C:
        raise RuntimeError(f"SHARED_MLP initial function deviates from P0: {d_c}, {rel_c}")
    if d_rand > STEP0_BAND_C:
        raise RuntimeError(f"SHARED_MLP initial function deviates from P0 on random x: {d_rand}")
    r_model = build("BUCKET_LINEAR")
    contracts["bucket_linear"] = {
        "n_buckets": int(r_model.pair_projection.n_buckets),
        "layer_shape": list(r_model.pair_projection.layers[0].weight.shape),
        "no_bias": bool(all(l.bias is None for l in r_model.pair_projection.layers)),
        "parameters": int(sum(p.numel() for p in r_model.pair_projection.parameters())),
        "layers_bitwise_p0": bool(
            all(torch.equal(l.weight, base_state["pair_projection.weight"]) for l in r_model.pair_projection.layers)
        ),
    }
    if contracts["shared_mlp"]["parameters"] != SHARED_MLP_PROJECTION_PARAMETERS:
        raise RuntimeError("SHARED_MLP projection parameter count wrong")
    if contracts["bucket_linear"]["parameters"] != BUCKET_PROJECTION_PARAMETERS:
        raise RuntimeError("BUCKET_LINEAR projection parameter count wrong")
    checks["projection_contracts"] = contracts

    # (c) step-0 functional agreement of the three arms (+ endpoint views)
    step0: dict[str, Any] = {}
    outs: dict[str, torch.Tensor] = {}
    comps: dict[str, torch.Tensor] = {}
    for arm in ARMS:
        model = build(arm)
        model.eval()
        with torch.no_grad():
            outs[arm] = model(batch16, mask=cm.C6_MASK).detach().clone()
            comps[arm] = model.reader.components().detach().clone()
    step0["r_vs_base_max_abs"] = float((outs["BUCKET_LINEAR"] - outs["BASE"]).abs().max())
    step0["c_vs_base_max_abs"] = float((outs["SHARED_MLP"] - outs["BASE"]).abs().max())
    if step0["r_vs_base_max_abs"] != 0.0:
        raise RuntimeError(f"BUCKET_LINEAR step 0 is not bit-identical to BASE on CPU: {step0}")
    if step0["c_vs_base_max_abs"] > STEP0_BAND_C:
        raise RuntimeError(f"SHARED_MLP step 0 exceeds the rounding band vs BASE: {step0}")
    with torch.no_grad():
        u_base = F.linear(E_fit, base_state["pair_projection.weight"])
        views = torch.stack([l(E_fit) for l in r_model.pair_projection.layers], dim=0)
        route = batch16.pair_bucket
        source = batch16.pair_index[0]
        target = batch16.pair_index[1]
        r_left = views[route, source]
        r_right = views[route, target]
    step0["endpoint_views"] = {
        "left_max_abs_vs_base": float((r_left - u_base[source]).abs().max()),
        "right_max_abs_vs_base": float((r_right - u_base[target]).abs().max()),
        "n_pairs_checked": int(route.numel()),
    }
    if step0["endpoint_views"]["left_max_abs_vs_base"] != 0.0 or \
        step0["endpoint_views"]["right_max_abs_vs_base"] != 0.0:
        raise RuntimeError("BUCKET_LINEAR endpoint views differ from BASE at step 0")
    checks["step0_agreement"] = step0

    # (d) same-forward component identity (fit rows, one batch, all arms)
    identity: dict[str, Any] = {}
    for arm in ARMS:
        model = build(arm)
        model.eval()
        with torch.no_grad():
            prediction = model(batch16, mask=cm.C6_MASK)
            components = model.reader.components()
        torch_gap = float((components.sum(-1) - prediction).abs().max())
        f64_gap = float(np.max(np.abs(
            prediction.numpy().astype(np.float64)
            - (components[:, 0].numpy().astype(np.float64) + components[:, 1].numpy().astype(np.float64))
        )))
        if torch_gap != 0.0:
            raise RuntimeError(f"{arm}: torch-internal component identity broken")
        if f64_gap > FLOAT64_SUM_TOL:
            raise RuntimeError(f"{arm}: exported float64 component sum off h")
        identity[arm] = {"torch_identity_max": torch_gap, "float64_sum_max_abs": f64_gap}
    checks["same_forward_components"] = identity

    # (e) invariances on fit molecules
    invariance: dict[str, Any] = {}
    subset = fit_data[:24]
    for arm in ARMS:
        model = build(arm)
        model.eval()

        def predict(rows: Sequence[Any], indices: Sequence[int] | None = None) -> np.ndarray:
            b = zftd.make_batch(rows, list(range(len(rows))) if indices is None else list(indices),
                                torch.zeros(len(rows)), device)
            with torch.no_grad():
                return model(b, mask=cm.C6_MASK).view(-1).numpy().astype(np.float64)

        # batch composition / row order
        order = list(range(len(subset)))
        singles = np.concatenate([predict(subset[i : i + 1]) for i in order])
        shuffled = [order[(i * 7 + 3) % len(order)] for i in order]
        grouped = predict(subset, shuffled)
        d_order = float(np.abs(singles[np.asarray(shuffled)] - grouped).max())
        # label independence
        b_y0 = zftd.make_batch(fit_data, list(range(8)), torch.zeros(8), device)
        b_y1 = zftd.make_batch(fit_data, list(range(8)), torch.ones(8), device)
        with torch.no_grad():
            d_label = float((model(b_y0, mask=cm.C6_MASK) - model(b_y1, mask=cm.C6_MASK)).abs().max())
        # pair-row permutation within two molecules (fp-order noise allowed)
        d_pairs = 0.0
        for row in subset[:2]:
            n_pairs = int(row.pair_bucket.shape[0])
            perm = torch.randperm(n_pairs, generator=torch.Generator().manual_seed(20261008))
            cloned = row.clone()
            cloned.pair_index = row.pair_index[:, perm].clone()
            cloned.pair_relation = row.pair_relation[perm].clone()
            cloned.pair_bucket = row.pair_bucket[perm].clone()
            p_ref = predict([row])[0]
            p_perm = predict([cloned])[0]
            d_pairs = max(d_pairs, float(abs(p_ref - p_perm)))
        # node relabelling (the positional tuple payload is swapped for the one
        # molecule exactly like the nonlinear-binding round's proven pattern)
        row0 = subset[0]
        n_nodes = int(row0.dict_phi.shape[0])
        relabel = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(20261015))
        row_r = nb._relabel_nodes(row0, relabel.numpy())
        mol0 = int(torch.as_tensor(row0.local_mol_id).reshape(-1)[0].item())
        permuted_payload = prev.TuplePayload(
            nb._permuted_tuple_payload(objects["payload_arrays"], mol0, relabel.numpy())
        )
        original_payload = model.local_tuple._payload
        original_cache = model.local_tuple._device_cache
        model.local_tuple._payload = permuted_payload
        model.local_tuple._device_cache = {}
        try:
            p_ref = predict([row0])[0]
            p_rel = predict([row_r])[0]
        finally:
            model.local_tuple._payload = original_payload
            model.local_tuple._device_cache = original_cache
        d_relabel = float(abs(p_ref - p_rel))
        # endpoint-swap symmetry of the pair input (the routed views are the
        # same P_b on both endpoints, so swapping roles must not change it)
        model_row = zftd.make_batch(subset[:1], [0], torch.zeros(1), device)
        with torch.no_grad():
            pred_aux = model(model_row, mask=cm.C6_MASK, return_aux=True)
            E1 = pred_aux[1]["E"]
            src = model_row.pair_index[0]
            tgt = model_row.pair_index[1]
            left, right = model.endpoint_views(E1, src, tgt, model_row, cm.C6_MASK)
            rel_in = model_row.pair_relation[:, list(p1.P1_RELATION_INDICES)]
            relation = model.relation_encoder(rel_in)
            gate = 1.0 + torch.tanh(model.distance_gate(model_row.pair_bucket))
            pi = torch.cat([left + right, torch.abs(left - right), left * right * gate, relation], dim=1)
            pi_swap = torch.cat([right + left, torch.abs(right - left), right * left * gate, relation], dim=1)
        d_swap = float((pi - pi_swap).abs().max())
        invariance[arm] = {
            "batch_order_max_abs": d_order,
            "label_independence_max_abs": d_label,
            "pair_row_permutation_max_abs": d_pairs,
            "node_relabel_max_abs": d_relabel,
            "endpoint_swap_pair_input_max_abs": d_swap,
        }
        if d_order > 1e-5:
            raise RuntimeError(f"{arm}: batch/order invariance failed ({d_order})")
        if d_label != 0.0:
            raise RuntimeError(f"{arm}: labels entered the forward path")
        if d_pairs > 1e-5:
            raise RuntimeError(f"{arm}: pair-row order changed predictions ({d_pairs})")
        if d_relabel > 1e-5:
            raise RuntimeError(f"{arm}: node relabelling changed predictions ({d_relabel})")
        if d_swap != 0.0:
            raise RuntimeError(f"{arm}: endpoint swap changed the pair input ({d_swap})")
    checks["invariances"] = invariance

    # (f) small-batch fit-only optimization smoke (3 optimizer steps per arm)
    grads: dict[str, Any] = {}
    for arm in ARMS:
        seed_everything(0)
        model = build(arm)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        n_opt = int(sum(p.numel() for group in optimizer.param_groups for p in group["params"]))
        if n_opt != arm_expected_audit(arm)["total_parameters"]:
            raise RuntimeError(f"{arm}: optimizer parameters {n_opt} != contract total")
        state_keys = [k for k in model.state_dict() if k.startswith("local_tuple.cssd_")]
        if state_keys:
            raise RuntimeError(f"{arm}: CSSD basis leaked into the state dict: {state_keys}")
        encoder_hashes_before = model.local_tuple.frozen_basis_hashes()
        finite_ok = True
        loss_values: list[float] = []
        proj_grad_trace: list[dict[str, Any]] = []
        for step in range(3):
            b = zftd.make_batch(fit_data, smoke_rows, target_g, device)
            prediction = model(b, mask=cm.C6_MASK)
            components = model.reader.components()
            loss = F.l1_loss(prediction.view(-1), b.y.view(-1)) + COMPONENT_LOSS_WEIGHT * (
                F.l1_loss(components[:, 0], target_ell[smoke_rows])
                + F.l1_loss(components[:, 1], target_s[smoke_rows])
            )
            optimizer.zero_grad()
            loss.backward()
            if step == 0:
                proj_grad_trace.append({"step": 0, **_projection_grad_norms(model)})
            optimizer.step()
            finite_ok = finite_ok and bool(torch.isfinite(prediction).all() and torch.isfinite(loss))
            loss_values.append(float(loss.detach()))
        proj_grad_trace.append({"step": 2, **_projection_grad_norms(model)})
        bridge = model.local_dictionary_bridge
        g1 = float(bridge.fc1.weight.grad.norm()) if bridge.fc1.weight.grad is not None else None
        g2 = float(bridge.fc2.weight.grad.norm()) if bridge.fc2.weight.grad is not None else None
        wloc_g = float(model.local_tuple.W_loc.grad.norm()) if model.local_tuple.W_loc.grad is not None else None
        a_g = float(model.local_tuple.A_raw.grad.norm()) if model.local_tuple.A_raw.grad is not None else None
        reader_g = float(model.reader.net[-1].weight.grad.norm()) if model.reader.net[-1].weight.grad is not None else None
        encoder_hashes_after = model.local_tuple.frozen_basis_hashes()
        if encoder_hashes_after != encoder_hashes_before:
            raise RuntimeError(f"{arm}: frozen basis changed during the optimization smoke")
        if not finite_ok:
            raise RuntimeError(f"{arm}: non-finite outputs/loss in the optimization smoke")
        if not (g1 and g1 > 0.0 and g2 and g2 > 0.0 and wloc_g and wloc_g > 0.0
                and a_g and a_g > 0.0 and reader_g and reader_g > 0.0):
            raise RuntimeError(f"{arm}: body gradients not healthy after 3 steps: {g1}, {g2}, {wloc_g}, {a_g}, {reader_g}")
        grads[arm] = {
            "loss_values": loss_values,
            "bridge_fc1_grad_norm_after_3_steps": g1,
            "bridge_fc2_grad_norm_after_3_steps": g2,
            "W_loc_grad_norm_after_3_steps": wloc_g,
            "A_raw_grad_norm_after_3_steps": a_g,
            "reader_grad_norm_after_3_steps": reader_g,
            "projection_grad_trace": proj_grad_trace,
            "cssd_buffers_not_in_optimizer": True,
            "frozen_basis_hashes_unchanged": True,
        }
    # C's supplementary W1 rows are zero-grad at step 0 BY CONSTRUCTION
    # (W2's zero columns) and must have moved by step 3; R's missing buckets
    # may be zero-grad on a 16-molecule batch — read with the bucket coverage
    c_trace = grads["SHARED_MLP"]["projection_grad_trace"]
    if c_trace[0]["w1_rows_96_180_grad_norm"] != 0.0:
        raise RuntimeError("SHARED_MLP supplementary rows should be zero-grad at step 0")
    if not c_trace[1]["w1_rows_96_180_grad_norm"] > 0.0 or not c_trace[1]["w2_supplementary_cols_grad_norm"] > 0.0:
        raise RuntimeError("SHARED_MLP supplementary block never received gradient by step 3")
    if not grads["SHARED_MLP"]["projection_grad_trace"][1]["w1_grad_norm_total"] > 0.0:
        raise RuntimeError("SHARED_MLP W1 never received gradient")
    if not grads["BUCKET_LINEAR"]["projection_grad_trace"][1]["bucket_0_grad_norm"] > 0.0:
        raise RuntimeError("BUCKET_LINEAR bucket 0 never received gradient")
    smoke_bucket_counts = np.bincount(
        np.concatenate([np.asarray(fit_data[i].pair_bucket, np.int64) for i in smoke_rows]),
        minlength=N_BUCKETS,
    ).tolist()
    grads["fit_smoke_bucket_coverage"] = {
        "bucket_pair_counts_16_rows": smoke_bucket_counts,
        "note": "a short batch may miss buckets (zero grad for those P_b is expected, not deadness)",
    }
    checks["gradient_smoke"] = grads

    # (g) save -> correct-factory reload -> replay (CPU bit-exact)
    replay: dict[str, Any] = {}
    tmp = out_dir / "pretrain_scratch"
    tmp.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        model = build(arm)
        state = gen._capture_state(model)
        replay_path = tmp / f"{arm}_replay_state.pt"
        torch.save(state, replay_path)
        reloaded = torch.load(replay_path, map_location="cpu", weights_only=False)
        replay_path.unlink()
        if state_hash(reloaded) != state_hash(state):
            raise RuntimeError(f"{arm}: save/reload hash mismatch")
        model_r = build(arm)
        model_r.load_state_dict({k: v for k, v in reloaded.items()}, strict=True)
        model.eval(); model_r.eval()
        with torch.no_grad():
            p_a = model(batch16, mask=cm.C6_MASK)
            p_b = model_r(batch16, mask=cm.C6_MASK)
        d = float((p_a - p_b).abs().max())
        if d != 0.0:
            raise RuntimeError(f"{arm}: reload replay differs by {d} on CPU")
        wrong_arm = next(a for a in ARMS if a != arm)
        wrong = build(wrong_arm)
        cross_blocked = False
        try:
            wrong.load_state_dict({k: v for k, v in reloaded.items()}, strict=True)
        except RuntimeError:
            cross_blocked = True
        if cross_blocked == (arm == wrong_arm):
            raise RuntimeError(f"{arm}: cross-factory strict load behaviour unexpected")
        replay[arm] = {"reload_replay_max_abs": d, "cross_factory_strict_load_blocked": cross_blocked}
    checks["save_reload_replay"] = replay

    # (h) route-override wiring (fit rows, fit-only, no training)
    route_checks: dict[str, Any] = {}
    natural_bucket = batch16.pair_bucket
    # one fixed override for the whole batch: reverse each molecule's pair-row
    # ORDER and read the bucket labels through it (a value permutation, exactly
    # like the terminal plan — the labels stay in 0..4)
    override_parts = []
    n_graphs = int(batch16.global_context.shape[0])
    pair_counts = torch.bincount(batch16.batch[batch16.pair_index[0]], minlength=n_graphs)
    offset = 0
    for count in pair_counts.tolist():
        override_parts.append(torch.arange(offset + int(count) - 1, offset - 1, -1))
        offset += int(count)
    order = torch.cat(override_parts)
    if int(order.numel()) != int(natural_bucket.numel()):
        raise RuntimeError("route override length mismatch")
    route_override = natural_bucket[order].to(natural_bucket.dtype)
    for arm in ("BASE", "SHARED_MLP"):
        model = build(arm)
        model.eval()
        with torch.no_grad():
            p_nat = model(batch16, mask=cm.C6_MASK).detach().clone()
            model.route_override = route_override
            p_ovr = model(batch16, mask=cm.C6_MASK).detach().clone()
            model.route_override = None
        route_checks[f"{arm}_override_ineffective_max_abs"] = float((p_nat - p_ovr).abs().max())
        if route_checks[f"{arm}_override_ineffective_max_abs"] != 0.0:
            raise RuntimeError(f"{arm}: the route override leaked into a shared-mode forward")
    # R with EQUAL P_b: the override must be a no-op
    model_r = build("BUCKET_LINEAR")
    model_r.eval()
    with torch.no_grad():
        p_nat = model_r(batch16, mask=cm.C6_MASK).detach().clone()
        model_r.route_override = route_override
        p_ovr = model_r(batch16, mask=cm.C6_MASK).detach().clone()
        stats_equal = model_r.last_route_stats
        model_r.route_override = None
    route_checks["bucket_equal_p_override_max_abs"] = float((p_nat - p_ovr).abs().max())
    if route_checks["bucket_equal_p_override_max_abs"] != 0.0:
        raise RuntimeError("R with equal P_b must be override-invariant")
    if not stats_equal or stats_equal["n_pairs_route_changed"] <= 0:
        raise RuntimeError("the override did not change any route label (test invalid)")
    # R with an ARTIFICIALLY different P_1 (fit-only in-memory perturbation):
    # the route must now select different endpoint views and change predictions
    with torch.no_grad():
        model_r.pair_projection.layers[1].weight.add_(0.05)
        p_nat2 = model_r(batch16, mask=cm.C6_MASK).detach().clone()
        model_r.route_override = route_override
        p_ovr2 = model_r(batch16, mask=cm.C6_MASK).detach().clone()
        stats_diff = dict(model_r.last_route_stats or {})
        model_r.route_override = None
    route_checks["bucket_different_p_natural_vs_override_max_abs"] = float((p_nat2 - p_ovr2).abs().max())
    route_checks["bucket_different_p_route_stats"] = stats_diff
    if route_checks["bucket_different_p_natural_vs_override_max_abs"] <= 0.0:
        raise RuntimeError("R did not respond to the route with different P_b (routing not wired)")
    # data.pair_bucket is never written
    if not torch.equal(batch16.pair_bucket, natural_bucket):
        raise RuntimeError("the route override modified data.pair_bucket")
    route_checks["pair_bucket_unmodified"] = True
    checks["route_override_wiring"] = route_checks

    checks["all_passed"] = True
    checks["seconds"] = float(time.perf_counter() - started)
    checks["official_valid_loaded"] = False
    checks["official_test_loaded"] = False
    checks["dev_rows_touched"] = False
    write_json(out_dir / "pretrain_checks.json", checks)
    log(f"[pretrain-checks] all passed in {checks['seconds']:.1f}s")
    return checks


# ---------------------------------------------------------------------------
# 4. stage: one formal body training (240 epochs, from scratch, fit-only)
# ---------------------------------------------------------------------------


def _probe_channels(model: RelationProjectionFullM, arm: str, epoch: int, step: int, total_norm: float) -> dict[str, Any]:
    """The historical local-channel probe + this round's projection grads."""
    entry = repl._probe_local_channel(model, "DICT", epoch, step, total_norm)
    entry["arm"] = arm
    entry["projection"] = _projection_grad_norms(model)
    return entry


def train_run(
    arm: str,
    seed: int,
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    smoke: bool = False,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """One from-scratch 240-epoch training of one arm x seed 0 (fit rows only).

    Clones the source round's train loop exactly (schedule, init, optimizer,
    batch order, loss, probes, curve, soup semantics) with only this round's
    per-arm factory and deterministic state capture at the member/checkpoint
    epochs.  During training nothing is built or scored: only detach/clone
    captures; dev is never touched; the route override stays None.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir = Path(out_dir)
    rdir = run_dir(arm, seed, out_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "members").mkdir(exist_ok=True)
    ctx = gen.load_round_context()
    objects = ctx["objects"]
    basis_parts = ctx["basis_parts"]
    payload, kappa_M = ctx["payload"], ctx["kappa_M"]
    fit_idx, fit_data = ctx["fit_idx"], ctx["fit_data"]
    targets = objects["targets"]
    y_all = np.asarray(targets["y"], np.float64)
    g_all = np.asarray(targets["g"], np.float64)
    ell_all = np.asarray(targets["ell"], np.float64)
    s_all = np.asarray(targets["s"], np.float64)
    gid_all = np.asarray(targets["gid"], np.int64)
    y_fit, g_fit = y_all[fit_idx], g_all[fit_idx]
    ell_fit, s_fit = ell_all[fit_idx], s_all[fit_idx]
    if float(np.max(np.abs(g_fit - (ell_fit + s_fit)))) > 1e-12:
        raise RuntimeError("g != ell + s on fit rows")
    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)

    epochs = 1 if smoke else int(EPOCHS)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), epochs, int(seed) + TRAIN_SHUFFLE_OFFSET)
    old = gen.load_old_run(int(seed))
    for e in range(epochs):
        if not np.array_equal(schedule[e], old["schedule_order"][e]):
            raise RuntimeError(f"schedule mismatch vs the historical plan at epoch {e + 1}")

    seed_everything(int(seed))
    model = build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))
    build_rng = torch.get_rng_state().clone()
    init_state = gen._capture_state(model)
    init_hash = state_hash(init_state)
    # cross-arm non-projection identity (fresh CPU constructions, fit-free)
    base_state = gen._capture_state(
        build_arm_round("BASE", payload, kappa_M, basis_parts, int(seed))
    )
    nb_init, nb_base = non_projection_state(init_state), non_projection_state(base_state)
    if sorted(nb_init) != sorted(nb_base):
        raise RuntimeError(f"{arm} s{seed}: non-projection state key sets differ vs BASE")
    mismatched = [k for k in nb_init if not torch.equal(nb_init[k], nb_base[k])]
    if mismatched:
        raise RuntimeError(f"{arm} s{seed}: non-projection init differs vs BASE at {mismatched}")
    if arm == "BASE" and init_hash != old["manifest"]["init_state_sha256"]:
        raise RuntimeError(f"BASE s{seed} init does not reproduce the historical DICT init")
    projection_init_sha256 = state_hash(projection_state(init_state))
    del base_state, nb_init, nb_base

    model = model.to(device)
    expected_encoder_hashes = {
        "U": array_sha256(basis_parts["U"].numpy()),
        "common_rms": array_sha256(basis_parts["common_rms"].numpy()),
        "Dbar": array_sha256(basis_parts["Dbar"].numpy()),
    }
    encoder_hashes_before = model.local_tuple.frozen_basis_hashes()
    if encoder_hashes_before != expected_encoder_hashes:
        raise RuntimeError(f"{arm} s{seed}: frozen basis buffers != loaded basis package")
    seed_everything(int(seed))
    train_rng = torch.get_rng_state().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_opt = int(sum(p.numel() for group in optimizer.param_groups for p in group["params"]))
    if n_opt != arm_expected_audit(arm)["total_parameters"]:
        raise RuntimeError(f"{arm} s{seed}: optimizer parameters {n_opt} != contract total")
    member_epochs = [1] if smoke else [int(e) for e in SOUP_EPOCHS]
    checkpoint_epochs = [1] if smoke else [120, 240]
    capture_epochs = sorted(set(member_epochs) | set(checkpoint_epochs))
    members: dict[int, dict[str, torch.Tensor]] = {}
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    steps_done = 0
    stopped_reason = "completed"
    started = time.perf_counter()
    peak_mb = 0.0
    position_stream = hashlib.sha256()
    gid_stream = hashlib.sha256()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        g_sum = ell_sum = s_sum = total_sum = 0.0
        n_mol = n_steps = 0
        gnorm_sum = 0.0
        clip_hits = 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = [int(i) for i in schedule[epoch - 1][start:start + BATCH_SIZE]]
            index_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            position_stream.update(np.asarray(indices, np.int64).tobytes())
            gid_stream.update(np.asarray(gid_all[fit_idx[np.asarray(indices, np.int64)]], np.int64).tobytes())
            batch = zftd.make_batch(fit_data, indices, target_g, device)
            prediction = model(batch, mask=cm.C6_MASK)
            if tuple(prediction.shape) != (len(indices),):
                raise RuntimeError("standard forward must return (batch,) total g")
            components = model.reader.components()
            if tuple(components.shape) != (len(indices), 2):
                raise RuntimeError("reader component output is not (batch, 2)")
            identity_gap = float(torch.max(torch.abs(components.sum(-1) - prediction.detach())).item())
            if identity_gap != 0.0:
                raise RuntimeError(f"g_hat != ell_hat + s_hat exactly (gap {identity_gap})")
            l_g = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            l_ell = F.l1_loss(components[:, 0], target_ell[index_t])
            l_s = F.l1_loss(components[:, 1], target_s[index_t])
            loss = l_g + COMPONENT_LOSS_WEIGHT * (l_ell + l_s)
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                probes.append(_probe_channels(model, arm, epoch, steps_done + 1, total_norm))
            optimizer.step()
            g_sum += float((prediction.detach().view(-1) - batch.y.view(-1)).abs().sum())
            ell_sum += float((components[:, 0].detach() - target_ell[index_t]).abs().sum())
            s_sum += float((components[:, 1].detach() - target_s[index_t]).abs().sum())
            total_sum += float(loss.detach()) * int(len(indices))
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        if device.type == "cuda":
            peak_mb = max(peak_mb, float(torch.cuda.max_memory_allocated() / (1 << 20)))
        curve.append({
            "epoch": int(epoch),
            "train_L_g": float(g_sum / max(n_mol, 1)),
            "train_L_ell": float(ell_sum / max(n_mol, 1)),
            "train_L_s": float(s_sum / max(n_mol, 1)),
            "train_loss": float(total_sum / max(n_mol, 1)),
            "grad_norm": float(gnorm_sum / max(n_steps, 1)),
            "clip_fraction": float(clip_hits / max(n_steps, 1)),
            "seconds": float(time.perf_counter() - epoch_started),
        })
        if epoch in capture_epochs:
            state = gen._capture_state(model)          # deep copy: no live references
            if epoch in member_epochs:
                members[int(epoch)] = state
                torch.save(state, rdir / "members" / f"epoch{epoch}_state.pt")
            if epoch in checkpoint_epochs:
                checkpoints[int(epoch)] = state
        if log and (epoch in LOG_EPOCHS or epoch % 40 == 0):
            c = curve[-1]
            log(
                f"[train {arm} s{seed}] ep={epoch:03d} Lg={c['train_L_g']:.5f} "
                f"Lell={c['train_L_ell']:.5f} Ls={c['train_L_s']:.5f} "
                f"gn={c['grad_norm']:.3g} {c['seconds']:.1f}s",
                flush=True,
            )
        if stopped_reason == "max_steps":
            break

    if not smoke:
        if stopped_reason != "completed":
            raise RuntimeError(f"formal run stopped early: {stopped_reason}")
        if steps_done != STEPS_EXPECTED:
            raise RuntimeError(f"{arm} s{seed}: steps {steps_done} != {STEPS_EXPECTED}")
        if sorted(members) != [int(e) for e in SOUP_EPOCHS]:
            raise RuntimeError(f"soup epochs missing: {sorted(members)}")
        if sorted(checkpoints) != [120, 240]:
            raise RuntimeError(f"trace checkpoints missing: {sorted(checkpoints)}")

    last_state = gen._capture_state(model)
    member_hashes = {int(e): state_hash(members[e]) for e in sorted(members)}
    soup_state = gen.average_states([members[e] for e in member_epochs])
    torch.save(soup_state, rdir / "soup_state.pt")
    soup_hash = state_hash(soup_state)

    # estimator strict-load + save/reload prediction assertion (fit rows only)
    probe_rows = list(range(8))
    probe_batch = zftd.make_batch(fit_data, probe_rows, torch.zeros(8), device)
    model_e = build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))
    model_e.load_state_dict({k: v for k, v in soup_state.items()}, strict=True)
    model_e = model_e.to(device).eval()
    with torch.no_grad():
        p_mem = model_e(probe_batch, mask=cm.C6_MASK).detach().cpu().clone()
    reloaded = torch.load(rdir / "soup_state.pt", map_location="cpu", weights_only=False)
    if state_hash(reloaded) != soup_hash:
        raise RuntimeError(f"{arm} s{seed}: soup save/reload hash mismatch")
    model_r = build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))
    model_r.load_state_dict({k: v for k, v in reloaded.items()}, strict=True)
    model_r = model_r.to(device).eval()
    with torch.no_grad():
        p_disk = model_r(probe_batch, mask=cm.C6_MASK).detach().cpu().clone()
    # CPU constructions replay bit-exact; separately allocated GPU instances may
    # pick different cuBLAS kernels, so the GPU band is the cross-device FP32
    # tolerance (the located fix of the source rounds, kept here from the start)
    reload_tol = 0.0 if device.type == "cpu" else CROSS_DEVICE_TOL
    d_reload = float((p_mem - p_disk).abs().max())
    if d_reload > reload_tol:
        raise RuntimeError(f"{arm} s{seed}: save/reload predictions differ by {d_reload}")
    del model_e, model_r
    if device.type == "cuda":
        torch.cuda.empty_cache()

    encoder_hashes_after = model.local_tuple.frozen_basis_hashes()
    if encoder_hashes_after != encoder_hashes_before:
        raise RuntimeError(f"{arm} s{seed}: frozen basis changed during training")

    # exact-resume capability record (never used for the round's numbers)
    torch.save(
        {
            "model_state": last_state,
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state().clone(),
            "steps_done": int(steps_done),
            "epochs_done": int(curve[-1]["epoch"]) if curve else 0,
            "schedule_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        },
        rdir / "resume_state.pt",
    )
    np.savez_compressed(
        rdir / "schedule.npz",
        order=np.stack([np.asarray(e, np.int64) for e in schedule]),
    )
    torch.save(init_state, rdir / "init_state.pt")
    torch.save(last_state, rdir / "last_state.pt")
    for epoch, state in sorted(checkpoints.items()):
        torch.save(state, rdir / f"epoch{epoch}_state.pt")

    # fit-only diagnostics from the soup (never dev, never a selection)
    replay = build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))
    replay.load_state_dict({k: v for k, v in soup_state.items()}, strict=True)
    replay = replay.to(device).eval()
    h, comps = zcs.evaluate_state_components(replay, fit_data, device)
    ell_hat, s_hat = comps[:, 0], comps[:, 1]
    T_topology = nb.topology_matrix(fit_data)
    q_raw = nb._q_predictions(ctx["q_soup"], T_topology, device)
    y_raw = ell_hat + s_hat + q_raw
    if float(np.max(np.abs(y_raw - (h + q_raw)))) > 1e-5:
        raise RuntimeError("y_raw != h + Q_raw on fit rows")
    b_y = float(np.median(y_fit - y_raw))
    if arm_parameter_audit(replay) != arm_expected_audit(arm):
        raise RuntimeError(f"{arm} s{seed}: post-training parameter audit failed")
    if replay.local_tuple.frozen_basis_hashes() != expected_encoder_hashes:
        raise RuntimeError(f"{arm} s{seed}: replay encoder basis hashes changed")
    post_audit = arm_parameter_audit(replay)
    del replay
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.savez_compressed(
        rdir / "fit_predictions.npz",
        row_index=np.asarray(fit_idx, np.int64), gid=gid_all[fit_idx],
        y=y_fit, g=g_fit, ell=ell_fit, s=s_fit,
        h=h, ell_hat=ell_hat, s_hat=s_hat, q_raw=q_raw, y_raw=y_raw,
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "train",
        "arm": arm,
        "seed": int(seed),
        "smoke": bool(smoke),
        "run_dir": str(rdir.relative_to(out_dir)),
        "recipe": {
            "epochs": int(epochs), "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "soup_epochs": list(member_epochs), "trace_checkpoints": list(checkpoint_epochs),
            "log_epochs": list(LOG_EPOCHS),
            "component_loss_weight": COMPONENT_LOSS_WEIGHT,
            "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
            "amp": None, "ddp": None, "scheduler": None, "early_stopping": None,
        },
        "supervision": (
            "COMP: L = MAE(ell_hat+s_hat, g) + 0.5*(MAE(ell_hat,ell)+MAE(s_hat,s)) "
            "(the unchanged disclosed auxiliary condition)"
        ),
        "steps_done": int(steps_done),
        "steps_expected": int(sum((len(e) + BATCH_SIZE - 1) // BATCH_SIZE for e in schedule)),
        "stopped_reason": stopped_reason,
        "schedule_sha256": schedule_hash,
        "schedule_bit_identical_to_historical": True,
        "position_stream_sha256": position_stream.hexdigest(),
        "global_gid_stream_sha256": gid_stream.hexdigest(),
        "init_state_sha256": init_hash,
        "init_non_projection_sha256": state_hash(non_projection_state(init_state)),
        "init_projection_sha256": projection_init_sha256,
        "non_projection_bitwise_equal_base": True,
        "member_state_sha256": {str(int(e)): member_hashes[int(e)] for e in sorted(members)},
        "soup_state_sha256": soup_hash,
        "last_state_sha256": state_hash(last_state),
        "build_rng_sha256": hashlib.sha256(build_rng.numpy().tobytes()).hexdigest(),
        "train_rng_sha256": hashlib.sha256(train_rng.numpy().tobytes()).hexdigest(),
        "frozen_basis_hashes": basis_parts["hashes"],
        "encoder_frozen_basis_hashes": encoder_hashes_after,
        "frozen_basis_unchanged": True,
        "q_soup_sha256": state_hash(ctx["q_soup"]),
        "parameter_audit": post_audit,
        "calibration": {
            "b_y": b_y,
            "note": "single fit-median b_y = median(y_fit - y_raw_fit); never stacked with any old bias",
        },
        "fit_diagnostics": {
            "fit_y_raw_mae": float(np.mean(np.abs(y_fit - y_raw))),
            "fit_y_cal_mae": float(np.mean(np.abs(y_fit - (y_raw + b_y)))),
            "fit_g_raw_mae": float(np.mean(np.abs(g_fit - h))),
            "fit_ell_mae": float(np.mean(np.abs(ell_fit - ell_hat))),
            "fit_s_mae": float(np.mean(np.abs(s_fit - s_hat))),
            "fit_q_mae": float(np.mean(np.abs(objects["targets"]["c"][fit_idx] - q_raw))),
        },
        "curve_seconds_total": float(sum(c["seconds"] for c in curve)),
        "wall_clock_s": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": float(peak_mb),
        "device": str(device),
        "allocation_probe": repl.allocation_probe(),
        "dev_scores_computed_during_training": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(rdir / "curve.json", curve)
    write_json(rdir / "probes.json", probes)
    write_json(rdir / "manifest.json", manifest)
    log(
        f"[train {arm} s{seed}] done: {steps_done} steps, {manifest['curve_seconds_total']:.0f}s train, "
        f"fit y_raw MAE {manifest['fit_diagnostics']['fit_y_raw_mae']:.5f} "
        f"b_y={b_y:.5f} soup sha={soup_hash[:12]}…"
    )
    return manifest


# ---------------------------------------------------------------------------
# 5. stage: GPU smoke (short end-to-end plumbing check, fit-only)
# ---------------------------------------------------------------------------


def run_smoke(
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    """One 1-epoch trajectory per arm (seed 0) through the FULL train path
    (capture/average/strict-load/save-reload), fit rows only; no dev read."""
    device = resolve_device(device_name)
    out_dir = Path(out_dir)
    smoke_dir = out_dir / "smoke"
    started = time.perf_counter()
    manifests = {}
    for arm in ARMS:
        manifests[arm] = train_run(
            arm, 0, device_name=device_name, out_dir=smoke_dir, smoke=True, log=log,
        )
    smoke = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "smoke",
        "device": str(device),
        "arms": {
            arm: {
                "steps_done": m["steps_done"],
                "stopped_reason": m["stopped_reason"],
                "parameter_audit": m["parameter_audit"],
                "schedule_bit_identical_to_historical": m["schedule_bit_identical_to_historical"],
                "soup_state_sha256": m["soup_state_sha256"],
            }
            for arm, m in manifests.items()
        },
        "checks": {
            "steps_1_epoch_63": all(m["steps_done"] == 63 for m in manifests.values()),
            "all_arms_completed": all(m["stopped_reason"] == "completed" for m in manifests.values()),
            "parameter_contracts_ok": all(
                m["parameter_audit"] == arm_expected_audit(arm) for arm, m in manifests.items()
            ),
            "frozen_basis_unchanged": all(m["frozen_basis_unchanged"] for m in manifests.values()),
            "aggregation_strict_load": True,
            "save_reload_predictions_within_band": True,
            "dev_never_touched": True,
        },
        "all_passed": True,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if not all(smoke["checks"].values()):
        raise RuntimeError(f"smoke checks failed: { {k: v for k, v in smoke['checks'].items() if not v} }")
    write_json(out_dir / "smoke.json", smoke)
    log(f"[smoke] all three arms ok on {device} in {smoke['seconds']:.1f}s")
    return smoke


# ---------------------------------------------------------------------------
# 6. terminal helpers (all accept THIS round's arm/factory, never the old ones)
# ---------------------------------------------------------------------------


def _load_soup_model(
    arm: str, seed: int, ctx: Mapping[str, Any], *, out_dir: Path,
) -> tuple[RelationProjectionFullM, dict[str, torch.Tensor]]:
    run = load_run(arm, seed, out_dir)
    model = build_arm_round(arm, ctx["payload"], ctx["kappa_M"], ctx["basis_parts"], int(seed))
    model.load_state_dict({k: v for k, v in run["soup_state"].items()}, strict=True)
    return model, run["soup_state"]


def _score_run(
    arm: str, seed: int, ctx: Mapping[str, Any], *,
    out_dir: Path, device: torch.device, q_fit: np.ndarray, q_dev: np.ndarray,
) -> dict[str, Any]:
    """Full fit+dev scoring of one soup (b_y from this run's own fit median)."""
    model, state = _load_soup_model(arm, seed, ctx, out_dir=out_dir)
    rdir = run_dir(arm, seed, out_dir)
    manifest = read_json(rdir / "manifest.json")
    if state_hash(state) != manifest["soup_state_sha256"]:
        raise RuntimeError(f"{arm} s{seed}: soup state hash mismatch vs manifest")
    model = model.to(device).eval()
    fitp = gen._predict_rows_checked(model, ctx["fit_data"], device)
    devp = gen._predict_rows_checked(model, ctx["dev_data"], device)
    y_raw_fit = fitp["ell_hat"] + fitp["s_hat"] + q_fit
    y_raw_dev = devp["ell_hat"] + devp["s_hat"] + q_dev
    b_y = float(np.median(ctx["y_fit"] - y_raw_fit))
    return {
        "arm": arm, "seed": int(seed), "model": model, "state": state,
        "fit": {"h": fitp["h"], "ell_hat": fitp["ell_hat"], "s_hat": fitp["s_hat"],
                "y_raw": y_raw_fit, "b_y": b_y},
        "dev": {"h": devp["h"], "ell_hat": devp["ell_hat"], "s_hat": devp["s_hat"],
                "y_raw": y_raw_dev, "b_y": b_y},
    }


CONTRASTS = (("BUCKET_LINEAR", "BASE"), ("SHARED_MLP", "BASE"), ("BUCKET_LINEAR", "SHARED_MLP"))
CONTRAST_LABELS = {"BUCKET_LINEAR|BASE": "R-BASE", "SHARED_MLP|BASE": "C-BASE", "BUCKET_LINEAR|SHARED_MLP": "R-C"}


def paired_bootstrap_three_contrasts(
    errs_y: Mapping[str, np.ndarray],
    errs_g: Mapping[str, np.ndarray],
    group_of_row: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Canonical-SMILES-group-paired bootstrap of Delta_y AND Delta_g for the
    three contrasts R-BASE, C-BASE, R-C.

    2000 draws, fixed seed 20261008; the SAME group picks are shared by all
    three arms and BOTH metrics within a draw (y and g share the sampling).
    Molecular sampling only — no training-seed/basis uncertainty; single body
    seed, so a CI excluding 0 is still NOT a repeatability claim.
    """
    keys = [f"{arm}_s{SEED}" for arm in ARMS]
    for table, tag in ((errs_y, "y"), (errs_g, "g")):
        for key in keys:
            if key not in table:
                raise RuntimeError(f"bootstrap expects {keys}, missing {key} ({tag})")
            if np.asarray(table[key]).shape != np.asarray(group_of_row).shape:
                raise RuntimeError(f"row misalignment at {key} ({tag})")
    labels = np.asarray([str(v) for v in np.asarray(group_of_row, dtype=object).tolist()], dtype=object)
    order = {label: index for index, label in enumerate(sorted(set(labels.tolist())))}
    g_index = np.asarray([order[label] for label in labels.tolist()], np.int64)
    n_groups = len(order)
    cnt = np.zeros(n_groups, np.int64)
    np.add.at(cnt, g_index, 1)
    sums = {tag: {k: np.zeros(n_groups, np.float64) for k in keys} for tag in ("y", "g")}
    for tag, table in (("y", errs_y), ("g", errs_g)):
        for k in keys:
            np.add.at(sums[tag][k], g_index, np.asarray(table[k], np.float64))
    rng = np.random.default_rng(int(seed))
    draws = {f"{cand}|{ctrl}": {tag: np.empty(int(n_boot)) for tag in ("y", "g")} for cand, ctrl in CONTRASTS}
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_groups, n_groups)   # shared across arms AND metrics
        n = int(cnt[pick].sum())
        m_y = {k: float(sums["y"][k][pick].sum() / n) for k in keys}
        m_g = {k: float(sums["g"][k][pick].sum() / n) for k in keys}
        for cand, ctrl in CONTRASTS:
            key = f"{cand}|{ctrl}"
            draws[key]["y"][b] = m_y[f"{cand}_s{SEED}"] - m_y[f"{ctrl}_s{SEED}"]
            draws[key]["g"][b] = m_g[f"{cand}_s{SEED}"] - m_g[f"{ctrl}_s{SEED}"]
    return {
        "n_rows": int(cnt.sum()),
        "n_groups": int(n_groups),
        "n_boot": int(n_boot),
        "boot_seed": int(seed),
        "shared_group_resampling": True,
        "shared_picks_across_arms_and_metrics": True,
        "contrasts": {
            CONTRAST_LABELS[key]: {
                "delta_y_ci95": [float(np.percentile(draws[key]["y"], 2.5)), float(np.percentile(draws[key]["y"], 97.5))],
                "delta_y_mean": float(draws[key]["y"].mean()),
                "delta_g_ci95": [float(np.percentile(draws[key]["g"], 2.5)), float(np.percentile(draws[key]["g"], 97.5))],
                "delta_g_mean": float(draws[key]["g"].mean()),
            }
            for key in draws
        },
        "reading": (
            "CI covers molecule resampling only — not training-seed or basis "
            "uncertainty; a single body seed is trained, so no contrast here is "
            "a repeatability or method-stability claim"
        ),
    }


def alpha_mean_intervention(
    arm: str,
    seed: int,
    base: Mapping[str, Any],
    mean_alpha: np.ndarray,
    *,
    ctx: Mapping[str, Any],
    out_dir: Path,
    device: torch.device,
    marker: float,
    log: Any = print,
) -> dict[str, Any]:
    """The unique dictionary intervention on one soup (any arm of THIS round).

    alpha_v := the fit-root mean alpha at the single local structure input via
    ``model.local_tuple.alpha_replace`` (the CSSD ENCODER — never the wrapper);
    c and every other input untouched; re-decoded and re-forwarded; no
    retraining, no recalibration.  Dependence evidence, never benefit.
    """
    model, _state = _load_soup_model(arm, seed, ctx, out_dir=out_dir)
    model = model.to(device).eval()
    model.local_tuple.alpha_replace = torch.as_tensor(np.asarray(mean_alpha, np.float64), dtype=torch.float32)
    preds = gen._predict_rows_checked(model, ctx["dev_data"], device)
    model.local_tuple.alpha_replace = None

    basis = ctx["basis"]
    dev_idx = ctx["dev_idx"]
    rows, _checks = repl._phi_rows_for(dev_idx.tolist())
    z, phi_hat, _rel = zreuse._root_codes(basis, rows)
    alpha_dev = z[:, repl.COMMON_DIM:]
    mean = np.asarray(mean_alpha, np.float64)
    U = np.asarray(basis["U"], np.float64)
    Dbar = np.asarray(zreuse._frozen_code_parts(basis)[2].numpy(), np.float64)
    c = rows.astype(np.float64) @ U
    phi_hat_int = c @ U.T + mean[None, :] @ Dbar.T
    alpha_dev_from_mean = np.linalg.norm(alpha_dev - mean[None, :], axis=1)
    phi_change = np.abs(phi_hat_int - phi_hat)

    q_raw = np.asarray(base["q_raw"], np.float64)
    y_raw_i = preds["ell_hat"] + preds["s_hat"] + q_raw
    y = np.asarray(base["y"], np.float64)
    g = np.asarray(base["g"], np.float64)
    base_y_raw = np.asarray(base["y_raw"], np.float64)
    base_h = np.asarray(base["h"], np.float64)
    d_pred = np.abs(y_raw_i - base_y_raw)
    row = {
        "arm": arm, "seed": int(seed),
        "mechanism": (
            "alpha_v := the fit-root mean alpha at the single local structure input "
            "(model.local_tuple.alpha_replace on the CSSD encoder); c kept; chemistry "
            "one-hots, J incidence, Sem108, reader and the frozen Q untouched; "
            "re-decoded and re-forwarded; no retraining, no recalibration"
        ),
        "input_change": {
            "alpha_vs_mean_l2_median": float(np.median(alpha_dev_from_mean)),
            "alpha_vs_mean_l2_p95": float(np.percentile(alpha_dev_from_mean, 95)),
            "alpha_vs_mean_l2_max": float(alpha_dev_from_mean.max()),
            "phi_hat_abs_change_median": float(np.median(phi_change)),
            "phi_hat_abs_change_max": float(np.max(phi_change)),
        },
        "abs_dpred": {
            "mean": float(d_pred.mean()),
            "median": float(np.median(d_pred)),
            "p95": float(np.percentile(d_pred, 95)),
            "max": float(d_pred.max()),
        },
        "response_fraction_gt_marker": float((d_pred > float(marker)).mean()),
        "delta_y_raw_mae": float(np.mean(np.abs(y - y_raw_i)) - np.mean(np.abs(y - base_y_raw))),
        "delta_g_mae": float(np.mean(np.abs(g - preds["h"])) - np.mean(np.abs(g - base_h))),
        "g_note": "Q input unchanged by construction: q_raw reused bitwise from the base evaluation",
        "reading": "response shows the consumer reads the sparse code; never incremental benefit",
    }
    if log:
        log(
            f"[alpha-intervention {arm} s{seed}] |dPred| p95 {row['abs_dpred']['p95']:.2e} "
            f"resp {row['response_fraction_gt_marker']:.3f} dMAE {row['delta_y_raw_mae']:+.2e}"
        )
    return row


def build_route_plan(dev_data: Sequence[Any], dev_gid: np.ndarray) -> dict[str, Any]:
    """The ONE fixed route-shuffle override set shared by all three arms.

    Within each dev molecule the EXISTING projection-route bucket labels are
    permuted (usage counts per bucket preserved exactly), using a single
    deterministic stream (seed 20261008) over the molecules in the saved dev
    order, keyed by nothing but the fixed row order — never by batch
    composition.  Natural and permuted bucket sequences are returned aligned
    to the flattened molecule order so any batch assembly gathers them.
    """
    rng = np.random.default_rng(ROUTE_SHUFFLE_SEED)
    natural_parts: list[np.ndarray] = []
    permuted_parts: list[np.ndarray] = []
    per_molecule: list[dict[str, Any]] = []
    for position, row in enumerate(dev_data):
        buckets = np.asarray(row.pair_bucket, np.int64)
        perm = rng.permutation(int(buckets.size))
        permuted = buckets[perm]
        changed = int((permuted != buckets).sum())
        natural_parts.append(buckets)
        permuted_parts.append(permuted)
        per_molecule.append({
            "dev_position": int(position),
            "gid": int(dev_gid[position]),
            "n_pairs": int(buckets.size),
            "n_pairs_changed": changed,
        })
    natural = np.concatenate(natural_parts) if natural_parts else np.zeros(0, np.int64)
    permuted = np.concatenate(permuted_parts) if permuted_parts else np.zeros(0, np.int64)
    return {
        "seed": int(ROUTE_SHUFFLE_SEED),
        "n_molecules": len(dev_data),
        "natural": natural,
        "permuted": permuted,
        "per_molecule": per_molecule,
        "n_pairs_total": int(natural.size),
        "n_pairs_changed": int((natural != permuted).sum()),
        "molecules_covered": int(sum(1 for m in per_molecule if m["n_pairs_changed"] > 0)),
    }


def _route_override_for_chunk(plan: Mapping[str, Any], chunk: Sequence[int]) -> torch.Tensor:
    parts = []
    offset = 0
    for position in chunk:
        n = int(plan["per_molecule"][position]["n_pairs"])
        parts.append(torch.as_tensor(plan["permuted"][offset : offset + n], dtype=torch.long))
        offset += n
    return torch.cat(parts)


def route_shuffle_intervention(
    arm: str,
    seed: int,
    base: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    ctx: Mapping[str, Any],
    out_dir: Path,
    device: torch.device,
    marker: float,
    log: Any = print,
) -> dict[str, Any]:
    """Intervention 2: shuffle the projection-route buckets, nothing else.

    The fixed per-molecule permutation of the EXISTING route labels (usage
    counts preserved) is applied at the endpoint-view gather ONLY; the
    relation input, the distance gate, the pair readout pooling, the pair
    endpoints, E_i and the frozen Q are untouched and ``data.pair_bucket`` is
    never written.  BASE/C do not read the route (their predictions may move
    only by numerical noise).  For R the response is only meaningful if the
    five P_b actually diverged in training; a no-response with identical P_b
    would mean no different relation viewpoints were learned (recorded as
    such, never rescued by re-initialization).
    """
    model, _state = _load_soup_model(arm, seed, ctx, out_dir=out_dir)
    model = model.to(device).eval()
    dev_data = ctx["dev_data"]
    reads_route = model.endpoint_mode == "bucket"
    weights = model.pair_projection
    permuted_preds: dict[str, np.ndarray] = {"h": [], "ell_hat": [], "s_hat": []}
    route_stats_agg: dict[str, Any] = {
        "n_pairs": 0, "n_pairs_route_changed": 0,
        "source_view_rel_change_median": [], "source_view_rel_change_mean": [],
    }
    pair_bucket_untouched = True
    hs, ells, ss = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dev_data), BATCH_SIZE):
            chunk = list(range(start, min(start + BATCH_SIZE, len(dev_data))))
            batch = zftd.make_batch(dev_data, chunk, torch.zeros(len(dev_data)), device)
            natural = batch.pair_bucket.detach().clone()
            model.route_override = _route_override_for_chunk(plan, chunk).to(device)
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            model.route_override = None
            pair_bucket_untouched = pair_bucket_untouched and bool(
                torch.equal(batch.pair_bucket.detach().cpu(), natural.cpu())
            )
            hs.append(prediction.detach().cpu().numpy().astype(np.float64))
            ells.append(components[:, 0].detach().cpu().numpy().astype(np.float64))
            ss.append(components[:, 1].detach().cpu().numpy().astype(np.float64))
            if reads_route and model.last_route_stats is not None:
                stats = model.last_route_stats
                route_stats_agg["n_pairs"] += int(stats["n_pairs"])
                route_stats_agg["n_pairs_route_changed"] += int(stats["n_pairs_route_changed"])
                if "source_view_rel_change_median" in stats:
                    route_stats_agg["source_view_rel_change_median"].append(float(stats["source_view_rel_change_median"]))
                    route_stats_agg["source_view_rel_change_mean"].append(float(stats["source_view_rel_change_mean"]))
            model.last_route_stats = None
    permuted_preds = {"h": np.concatenate(hs), "ell_hat": np.concatenate(ells), "s_hat": np.concatenate(ss)}
    q_raw = np.asarray(base["q_raw"], np.float64)
    y_raw_i = permuted_preds["ell_hat"] + permuted_preds["s_hat"] + q_raw
    y = np.asarray(base["y"], np.float64)
    g = np.asarray(base["g"], np.float64)
    base_y_raw = np.asarray(base["y_raw"], np.float64)
    base_h = np.asarray(base["h"], np.float64)
    d_pred = np.abs(y_raw_i - base_y_raw)
    d_h = np.abs(permuted_preds["h"] - base_h)

    if reads_route:
        changed = route_stats_agg["n_pairs_route_changed"]
        agg = {
            "n_pairs": route_stats_agg["n_pairs"],
            "n_pairs_route_changed": changed,
            "fraction_route_changed": float(changed) / float(max(route_stats_agg["n_pairs"], 1)),
            "source_view_rel_change_median_across_batches": (
                float(np.median(route_stats_agg["source_view_rel_change_median"]))
                if route_stats_agg["source_view_rel_change_median"] else None
            ),
            "source_view_rel_change_mean_across_batches": (
                float(np.mean(route_stats_agg["source_view_rel_change_mean"]))
                if route_stats_agg["source_view_rel_change_mean"] else None
            ),
        }
    else:
        agg = {"note": "shared-mode endpoint projection: the route is not read"}

    if isinstance(weights, BucketEndpointProjection):
        stacked = torch.stack([l.weight.detach() for l in weights.layers])
        mean_w = stacked.mean(dim=0)
        dev_rel = (stacked - mean_w).norm(dim=(1, 2)) / (mean_w.norm() + 1e-12)
        p_divergence = {
            "per_bucket_rel_dev_from_mean": [float(v) for v in dev_rel],
            "mean_rel_dev_from_mean": float(dev_rel.mean()),
            "max_pairwise_rel_diff": float(max(
                (stacked[i] - stacked[j]).norm() / (stacked[i].norm() + 1e-12)
                for i in range(N_BUCKETS) for j in range(N_BUCKETS) if i != j
            )),
        }
    else:
        p_divergence = {"note": "not a bucket projection"}

    row = {
        "arm": arm, "seed": int(seed),
        "mechanism": (
            "within each dev molecule the EXISTING projection-route bucket labels are "
            "permuted (per-bucket usage counts preserved; fixed stream seed "
            f"{ROUTE_SHUFFLE_SEED}); ONLY the endpoint-view gather follows the "
            "permuted route — relation input, distance gate, pair readout pooling, "
            "pair endpoints, E_i and Q untouched; data.pair_bucket never written"
        ),
        "reads_route": bool(reads_route),
        "plan": {
            "n_pairs_total": int(plan["n_pairs_total"]),
            "n_pairs_changed": int(plan["n_pairs_changed"]),
            "fraction_pairs_changed": float(plan["n_pairs_changed"]) / float(max(plan["n_pairs_total"], 1)),
            "molecules_covered": int(plan["molecules_covered"]),
            "n_molecules": int(plan["n_molecules"]),
        },
        "route_forward_stats": agg,
        "p_divergence": p_divergence,
        "abs_dpred": {
            "mean": float(d_pred.mean()),
            "median": float(np.median(d_pred)),
            "p95": float(np.percentile(d_pred, 95)),
            "max": float(d_pred.max()),
        },
        "abs_dpred_h": {
            "mean": float(d_h.mean()),
            "p95": float(np.percentile(d_h, 95)),
            "max": float(d_h.max()),
        },
        "response_fraction_gt_marker": float((d_pred > float(marker)).mean()),
        "delta_y_raw_mae": float(np.mean(np.abs(y - y_raw_i)) - np.mean(np.abs(y - base_y_raw))),
        "delta_g_mae": float(np.mean(np.abs(g - permuted_preds["h"])) - np.mean(np.abs(g - base_h))),
        "pair_bucket_unmodified": bool(pair_bucket_untouched),
        "reading": (
            "for R: response beyond the noise marker AND diverged P_b means different "
            "relation viewpoints are actually used; a no-response with equal P_b "
            "means no different viewpoints were learned (recorded, never rescued); "
            "for BASE/C: predictions may move only by numerical noise"
        ),
    }
    if log:
        log(
            f"[route-intervention {arm}] changed {row['plan']['fraction_pairs_changed']:.3f} | "
            f"|dPred| p95 {row['abs_dpred']['p95']:.2e} resp {row['response_fraction_gt_marker']:.3f} "
            f"dMAE_y {row['delta_y_raw_mae']:+.2e}"
        )
    return row


def exploratory_reading(
    deltas: Mapping[str, float],
    bootstrap: Mapping[str, Any],
    alpha_rows: Mapping[str, Mapping[str, Any]],
    route_rows: Mapping[str, Mapping[str, Any]],
    marker: float,
) -> dict[str, Any]:
    """The frozen single-seed reading rubric (protocol section 7).

    "Improves" always means the paired dev Delta (candidate - reference) < 0
    on that metric; point estimates decide the branch, the CIs are reported
    alongside and can only annotate uncertainty (a CI crossing 0 is explicitly
    recorded — never evidence of absence, never repeatability).
    """
    dy_rb, dg_rb = deltas["R-BASE_y"], deltas["R-BASE_g"]
    dy_cb, dg_cb = deltas["C-BASE_y"], deltas["C-BASE_g"]
    dy_rc, dg_rc = deltas["R-C_y"], deltas["R-C_g"]
    ci = bootstrap["contrasts"]
    ci_rc_y = ci["R-C"]["delta_y_ci95"]
    ci_rc_g = ci["R-C"]["delta_g_ci95"]
    r_alpha_ok = bool(alpha_rows["BUCKET_LINEAR"]["abs_dpred"]["p95"] > marker)
    r_route = route_rows["BUCKET_LINEAR"]
    p_diverged = (
        isinstance(r_route.get("p_divergence", {}).get("mean_rel_dev_from_mean", None), float)
        and float(r_route["p_divergence"]["mean_rel_dev_from_mean"]) > 0.0
    )
    route_responds = bool(
        r_route["abs_dpred"]["p95"] > marker and p_diverged and r_route["plan"]["fraction_pairs_changed"] > 0
    )
    r_improves = dy_rb < 0 and dg_rb < 0
    c_improves = dy_cb < 0 and dg_cb < 0
    r_vs_c_better_y = dy_rc < 0
    r_vs_c_not_worse_g = dg_rc <= 0

    if not r_alpha_ok:
        branch = "dictionary-readout-not-supported"
        note = "R's alpha-mean intervention p95 <= marker: R stopped reading the dictionary; no upgrade."
    elif (not p_diverged) and route_rows["BUCKET_LINEAR"]["plan"]["fraction_pairs_changed"] > 0:
        branch = "route-viewpoints-not-learned"
        note = "R's five P_b stayed (near-)identical: no different relation viewpoints were learned."
    elif r_improves and r_vs_c_better_y and r_vs_c_not_worse_g and route_responds:
        branch = "relational-conditional-projection-directional-signal"
        note = (
            "R improves both y and g vs BASE, beats C on y and is not worse on g, "
            "keeps the dictionary readout and really uses different routes "
            "(single-seed directional signal; the R-C CI decides how separated it is)."
        )
    elif r_improves and c_improves:
        branch = "stronger-endpoint-projection-directional-signal"
        note = (
            "Both C and R improve y and g vs BASE without a separated R-C difference: "
            "a stronger endpoint projection carries the signal; the relational split "
            "did not separate at this seed."
        )
    elif c_improves and not r_improves:
        branch = "shared-nonlinear-projection-directional-signal"
        note = (
            "C improves while R does not: the shared nonlinear projection carries a "
            "directional signal; the distance-split projection is not supported."
        )
    elif (dy_rb < 0) != (dg_rb < 0) or (dy_cb < 0) != (dg_cb < 0):
        branch = "mixed-signals"
        note = "y and g move in opposite directions for at least one contrast: no clear signal."
    else:
        branch = "no-clear-signal"
        note = (
            "Neither candidate reliably improves the full y AND g of the same-round "
            "BASE: no clear/mixed signal; this round's extra budget is closed and no "
            "fourth candidate is invented."
        )
    return {
        "branch": branch,
        "note": note,
        "point_estimates": {
            "R-BASE_y": dy_rb, "R-BASE_g": dg_rb,
            "C-BASE_y": dy_cb, "C-BASE_g": dg_cb,
            "R-C_y": dy_rc, "R-C_g": dg_rc,
        },
        "r_improves_both_vs_base": bool(r_improves),
        "c_improves_both_vs_base": bool(c_improves),
        "r_vs_c_y_better": bool(r_vs_c_better_y),
        "r_vs_c_g_not_worse": bool(r_vs_c_not_worse_g),
        "r_ci_crossings": {
            "R-BASE_y_ci_contains_zero": bool(ci["R-BASE"]["delta_y_ci95"][0] <= 0.0 <= ci["R-BASE"]["delta_y_ci95"][1]),
            "R-BASE_g_ci_contains_zero": bool(ci["R-BASE"]["delta_g_ci95"][0] <= 0.0 <= ci["R-BASE"]["delta_g_ci95"][1]),
            "R-C_y_ci_contains_zero": bool(ci_rc_y[0] <= 0.0 <= ci_rc_y[1]),
            "R-C_g_ci_contains_zero": bool(ci_rc_g[0] <= 0.0 <= ci_rc_g[1]),
        },
        "dictionary_readout_responds": r_alpha_ok,
        "route_viewpoints_learned_and_used": bool(p_diverged and route_responds),
        "single_seed_caveat": (
            "a directional signal is this seed's screening result only — not a "
            "statistical-significance or repeatability claim; C is a parameter-matched "
            "different function recipe, so R>C is never a universal causal mechanism claim"
        ),
    }


# ---------------------------------------------------------------------------
# 7. stage: ONE terminal evaluation (perf + the two interventions)
# ---------------------------------------------------------------------------


def terminal_eval(
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    log: Any = print,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if (out_dir / "terminal_eval.json").exists():
        raise RuntimeError("terminal_eval.json already exists: the terminal stage is one-shot")
    started = time.perf_counter()
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    ctx = gen.load_round_context()
    objects = ctx["objects"]
    basis_parts = ctx["basis_parts"]
    fit_idx, dev_idx = ctx["fit_idx"], ctx["dev_idx"]
    targets = objects["targets"]
    y = np.asarray(targets["y"], np.float64)
    g = np.asarray(targets["g"], np.float64)
    ell = np.asarray(targets["ell"], np.float64)
    s_t = np.asarray(targets["s"], np.float64)
    k_all = np.asarray(targets["k"], np.int64)
    gid_all = np.asarray(targets["gid"], np.int64)
    y_dev, g_dev, ell_dev, s_dev = y[dev_idx], g[dev_idx], ell[dev_idx], s_t[dev_idx]
    k_dev = k_all[dev_idx]
    _phi, _atom, node_sizes = prev._env_phi_atom()
    n_nodes_fit = np.asarray(node_sizes, np.int64)[fit_idx]
    n_nodes_dev = np.asarray(node_sizes, np.int64)[dev_idx]
    g_p90 = float(np.percentile(np.abs(g[fit_idx]), 90))
    smiles = zreuse._canonical_smiles()
    dev_group_labels, dev_group_ids = np.unique(smiles[dev_idx], return_inverse=True)
    fit_group_labels, fit_group_ids = np.unique(smiles[fit_idx], return_inverse=True)
    q_fit = nb._q_predictions(ctx["q_soup"], nb.topology_matrix(ctx["fit_data"]), device)
    q_dev = nb._q_predictions(ctx["q_soup"], nb.topology_matrix(ctx["dev_data"]), device)
    old = gen.load_old_run(0)
    expected_encoder_hashes = {
        "U": array_sha256(basis_parts["U"].numpy()),
        "common_rms": array_sha256(basis_parts["common_rms"].numpy()),
        "Dbar": array_sha256(basis_parts["Dbar"].numpy()),
    }

    # (1) roster verification (before any scoring)
    checks = {
        "runs_completed": True, "steps_ok": True, "schedule_ok": True,
        "init_ok": True, "non_projection_ok": True, "param_audit_ok": True,
        "frozen_basis_ok": True, "members_ok": True, "soup_ok": True,
        "q_soup_ok": True,
    }
    manifests: dict[str, dict[str, Any]] = {}
    init_states: dict[str, dict[str, torch.Tensor]] = {}
    for arm in ARMS:
        m = read_json(run_dir(arm, SEED, out_dir) / "manifest.json")
        if m["stopped_reason"] != "completed" or m.get("smoke"):
            checks["runs_completed"] = False
        if int(m["steps_done"]) != STEPS_EXPECTED or int(m["steps_expected"]) != STEPS_EXPECTED:
            checks["steps_ok"] = False
        if m["parameter_audit"] != arm_expected_audit(arm):
            checks["param_audit_ok"] = False
        if m["frozen_basis_hashes"] != basis_parts["hashes"]:
            checks["frozen_basis_ok"] = False
        if m["encoder_frozen_basis_hashes"] != expected_encoder_hashes:
            checks["frozen_basis_ok"] = False
        if not m["frozen_basis_unchanged"] or not m["schedule_bit_identical_to_historical"]:
            checks["frozen_basis_ok"] = checks["schedule_ok"] = False
        if m["schedule_sha256"] != old["manifest"]["schedule_sha256"]:
            checks["schedule_ok"] = False
        if state_hash(ctx["q_soup"]) != m["q_soup_sha256"]:
            checks["q_soup_ok"] = False
        if arm == "BASE" and m["init_state_sha256"] != old["manifest"]["init_state_sha256"]:
            checks["init_ok"] = False
        for e, h in m["member_state_sha256"].items():
            member = torch.load(
                run_dir(arm, SEED, out_dir) / "members" / f"epoch{int(e)}_state.pt",
                map_location="cpu", weights_only=False,
            )
            if state_hash(member) != h:
                checks["members_ok"] = False
        soup = torch.load(run_dir(arm, SEED, out_dir) / "soup_state.pt", map_location="cpu", weights_only=False)
        if state_hash(soup) != m["soup_state_sha256"]:
            checks["soup_ok"] = False
        init_states[arm] = torch.load(
            run_dir(arm, SEED, out_dir) / "init_state.pt", map_location="cpu", weights_only=False
        )
        if state_hash(init_states[arm]) != m["init_state_sha256"]:
            checks["init_ok"] = False
        manifests[arm] = m
    base_nb = non_projection_state(init_states["BASE"])
    for arm in CANDIDATES:
        nb_c = non_projection_state(init_states[arm])
        if sorted(nb_c) != sorted(base_nb) or not all(torch.equal(nb_c[k], base_nb[k]) for k in nb_c):
            checks["non_projection_ok"] = False
    if not all(checks.values()):
        raise RuntimeError(f"roster verification failed: { {k: v for k, v in checks.items() if not v} }")

    # (2) scoring of the three soups (fit + dev, full per-row exports)
    results: dict[str, dict[str, Any]] = {}
    main_table: dict[str, Any] = {}
    for arm in ARMS:
        res = _score_run(arm, SEED, ctx, out_dir=out_dir, device=device, q_fit=q_fit, q_dev=q_dev)
        results[arm] = res
        for split, rows_idx, preds, tgt, nnodes, gids, q in (
            ("fit", fit_idx, res["fit"],
             (ctx["y_fit"], ctx["g_fit"], ctx["ell_fit"], ctx["s_fit"], k_all[fit_idx]),
             n_nodes_fit, fit_group_ids, q_fit),
            ("dev", dev_idx, res["dev"], (y_dev, g_dev, ell_dev, s_dev, k_dev),
             n_nodes_dev, dev_group_ids, q_dev),
        ):
            np.savez_compressed(
                run_dir(arm, SEED, out_dir) / f"{split}_predictions.npz",
                row_index=np.asarray(rows_idx, np.int64), gid=gid_all[rows_idx],
                smiles_group=np.asarray(gids, np.int64),
                n_nodes=np.asarray(nnodes, np.int64), k=tgt[4],
                y=tgt[0], g=tgt[1], ell=tgt[2], s=tgt[3],
                h=preds["h"], ell_hat=preds["ell_hat"], s_hat=preds["s_hat"],
                q_raw=q, y_raw=preds["y_raw"],
                b_y=np.float64(preds["b_y"]),
            )
        mae_dev = {
            "y_raw": float(np.mean(np.abs(y_dev - res["dev"]["y_raw"]))),
            "y_cal": float(np.mean(np.abs(y_dev - (res["dev"]["y_raw"] + res["dev"]["b_y"])))),
            "g_raw": float(np.mean(np.abs(g_dev - res["dev"]["h"]))),
            "ell": float(np.mean(np.abs(ell_dev - res["dev"]["ell_hat"]))),
            "s": float(np.mean(np.abs(s_dev - res["dev"]["s_hat"]))),
        }
        mae_fit = {
            "y_raw": float(np.mean(np.abs(ctx["y_fit"] - res["fit"]["y_raw"]))),
            "g_raw": float(np.mean(np.abs(ctx["g_fit"] - res["fit"]["h"]))),
            "ell": float(np.mean(np.abs(ctx["ell_fit"] - res["fit"]["ell_hat"]))),
            "s": float(np.mean(np.abs(ctx["s_fit"] - res["fit"]["s_hat"]))),
        }
        e_y = res["dev"]["y_raw"] - y_dev
        e_g = res["dev"]["h"] - g_dev
        res["mae_dev"], res["mae_fit"] = mae_dev, mae_fit
        res["component_stats_dev"] = gen._component_stats(
            res["dev"]["ell_hat"], res["dev"]["s_hat"], res["dev"]["h"], ell_dev, s_dev, g_dev)
        res["k_groups_dev"] = gen._k_group_table(e_y, e_g, k_dev)
        res["k0_subgroups_dev"] = gen._k0_subgroup_table(
            e_y, e_g, k_dev, n_nodes_dev, np.abs(g_dev), g_p90)
        main_table[arm] = {
            "mae_dev": mae_dev,
            "mae_fit": mae_fit,
            "b_y": res["dev"]["b_y"],
            "gap_dev_minus_fit_g": float(mae_dev["g_raw"] - mae_fit["g_raw"]),
            "gap_dev_minus_fit_y_raw": float(mae_dev["y_raw"] - mae_fit["y_raw"]),
            "component_stats_dev": res["component_stats_dev"],
            "k_groups_dev": res["k_groups_dev"],
            "k0_subgroups_dev": res["k0_subgroups_dev"],
            "steps": STEPS_EXPECTED,
            "curve_seconds_total": manifests[arm]["curve_seconds_total"],
            "wall_clock_s": manifests[arm]["wall_clock_s"],
            "peak_gpu_memory_mb": manifests[arm]["peak_gpu_memory_mb"],
            "soup_state_sha256": manifests[arm]["soup_state_sha256"],
        }
        log(
            f"[eval {arm} s{SEED}] dev y_raw {mae_dev['y_raw']:.5f} g {mae_dev['g_raw']:.5f} | "
            f"fit y_raw {mae_fit['y_raw']:.5f} g {mae_fit['g_raw']:.5f} | "
            f"gap(g) {main_table[arm]['gap_dev_minus_fit_g']:.5f}"
        )

    # (3) paired deltas (dev + fit) for the three contrasts + the anchor
    deltas: dict[str, Any] = {}
    for cand, ctrl in CONTRASTS:
        label = CONTRAST_LABELS[f"{cand}|{ctrl}"]
        for metric, table_key in (("y", "y_raw"), ("g", "g_raw")):
            deltas[f"{label}_{metric}_dev"] = float(
                results[cand]["mae_dev"][table_key] - results[ctrl]["mae_dev"][table_key])
            deltas[f"{label}_{metric}_fit"] = float(
                results[cand]["mae_fit"][table_key] - results[ctrl]["mae_fit"][table_key])
    deltas["definition"] = (
        "Delta = MAE(candidate) - MAE(reference), same-seed paired, full dev "
        "(negative = improvement); fit side reported separately, never as a ratio "
        "in place of the dev error"
    )
    deltas["gap_table"] = {
        arm: main_table[arm]["gap_dev_minus_fit_g"] for arm in ARMS
    }
    historical_anchor = {
        "historical_dict_s0_dev_y_raw": HISTORICAL_DICT_S0_Y_RAW,
        "historical_dict_s0_dev_g_raw": 0.09390312910616606,
        "base_dev_y_raw": results["BASE"]["mae_dev"]["y_raw"],
        "base_dev_g_raw": results["BASE"]["mae_dev"]["g_raw"],
        "note": (
            "descriptive anchor ONLY: the historical seed-0 soup is not the same-round "
            "control and carries GPU training nondeterminism; the recipe effect is "
            "judged against the paired same-round BASE"
        ),
    }

    # (4) group-paired bootstrap (2000 draws, seed 20261008, shared picks)
    errs_y = {f"{arm}_s{SEED}": np.abs(y_dev - results[arm]["dev"]["y_raw"]) for arm in ARMS}
    errs_g = {f"{arm}_s{SEED}": np.abs(g_dev - results[arm]["dev"]["h"]) for arm in ARMS}
    bootstrap = paired_bootstrap_three_contrasts(errs_y, errs_g, smiles[dev_idx])

    # (5) noise bound eta: repeated forwards + same-weight save/reload replay
    eta = 0.0
    noise: dict[str, Any] = {}
    for arm in ARMS:
        repeat = _score_run(arm, SEED, ctx, out_dir=out_dir, device=device, q_fit=q_fit, q_dev=q_dev)
        d_repeat = float(np.max(np.abs(
            np.asarray(repeat["dev"]["y_raw"], np.float64)
            - np.asarray(results[arm]["dev"]["y_raw"], np.float64))))
        del repeat["model"], repeat
        model_disk, _state = _load_soup_model(arm, SEED, ctx, out_dir=out_dir)
        model_disk = model_disk.to(device).eval()
        preds_disk = gen._predict_rows_checked(model_disk, ctx["dev_data"], device)
        y_raw_disk = preds_disk["ell_hat"] + preds_disk["s_hat"] + q_dev  # frozen Q reused bitwise
        d_reload = float(np.max(np.abs(
            np.asarray(y_raw_disk, np.float64)
            - np.asarray(results[arm]["dev"]["y_raw"], np.float64))))
        noise[arm] = {
            "repeat_max_abs_dpred": d_repeat,
            "save_reload_max_abs_dpred": d_reload,
        }
        eta = max(eta, d_repeat, d_reload)
        del model_disk
        if device.type == "cuda":
            torch.cuda.empty_cache()
    marker = float(max(REPLAY_TOL, 10.0 * eta))

    # (6a) the unique alpha-mean dictionary intervention on all THREE soups
    mean_codes = repl.fit_mean_codes(ctx["basis"], fit_idx)
    alpha_rows: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        base = {
            "y": y_dev, "g": g_dev,
            "y_raw": results[arm]["dev"]["y_raw"],
            "h": results[arm]["dev"]["h"],
            "q_raw": q_dev, "b_y": results[arm]["dev"]["b_y"],
        }
        row = alpha_mean_intervention(
            arm, SEED, base, mean_codes["mean_alpha"], ctx=ctx, out_dir=out_dir,
            device=device, marker=marker, log=log,
        )
        row["response_beyond_noise"] = bool(row["abs_dpred"]["p95"] > marker)
        alpha_rows[arm] = row
        del results[arm]["model"]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # (6b) intervention 2: the ONE fixed route shuffle, shared by all arms
    plan = build_route_plan(ctx["dev_data"], gid_all[dev_idx])
    np.savez_compressed(
        out_dir / "route_shuffle_plan.npz",
        natural=plan["natural"], permuted=plan["permuted"],
        gid=np.asarray([m["gid"] for m in plan["per_molecule"]], np.int64),
        n_pairs=np.asarray([m["n_pairs"] for m in plan["per_molecule"]], np.int64),
        n_pairs_changed=np.asarray([m["n_pairs_changed"] for m in plan["per_molecule"]], np.int64),
    )
    route_rows: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        base = {
            "y": y_dev, "g": g_dev,
            "y_raw": results[arm]["dev"]["y_raw"],
            "h": results[arm]["dev"]["h"],
            "q_raw": q_dev, "b_y": results[arm]["dev"]["b_y"],
        }
        route_rows[arm] = route_shuffle_intervention(
            arm, SEED, base, plan, ctx=ctx, out_dir=out_dir, device=device,
            marker=marker, log=log,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # (7) the frozen single-seed exploratory reading
    point_deltas = {
        "R-BASE_y": deltas["R-BASE_y_dev"], "R-BASE_g": deltas["R-BASE_g_dev"],
        "C-BASE_y": deltas["C-BASE_y_dev"], "C-BASE_g": deltas["C-BASE_g_dev"],
        "R-C_y": deltas["R-C_y_dev"], "R-C_g": deltas["R-C_g_dev"],
    }
    reading = exploratory_reading(
        point_deltas, bootstrap,
        {arm: alpha_rows[arm] for arm in ARMS},
        {arm: route_rows[arm] for arm in ARMS},
        marker,
    )

    # (8) cost (actual, no inference from parameter counts)
    cost = {
        "per_run": {
            arm: {
                "steps": STEPS_EXPECTED,
                "curve_seconds_total": manifests[arm]["curve_seconds_total"],
                "wall_clock_s": manifests[arm]["wall_clock_s"],
                "peak_gpu_memory_mb": manifests[arm]["peak_gpu_memory_mb"],
                "device": manifests[arm]["device"],
                "allocation_probe": manifests[arm]["allocation_probe"],
            }
            for arm in ARMS
        },
        "total_wall_clock_s": float(sum(manifests[arm]["wall_clock_s"] for arm in ARMS)),
        "parameter_cost": {
            "BASE_total": int(arm_expected_audit("BASE")["total_parameters"]),
            "SHARED_MLP_total": int(arm_expected_audit("SHARED_MLP")["total_parameters"]),
            "BUCKET_LINEAR_total": int(arm_expected_audit("BUCKET_LINEAR")["total_parameters"]),
            "projection": {
                "BASE": BASE_PROJECTION_PARAMETERS,
                "SHARED_MLP": SHARED_MLP_PROJECTION_PARAMETERS,
                "BUCKET_LINEAR": BUCKET_PROJECTION_PARAMETERS,
            },
            "note": "cost reads the MEASURED seconds/memory; parameter deltas never imply time deltas",
        },
    }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "terminal-eval",
        "one_shot": True,
        "roster_checks": checks,
        "main_table": main_table,
        "paired_deltas": deltas,
        "historical_anchor": historical_anchor,
        "bootstrap": bootstrap,
        "noise_bound": {
            "eta": eta,
            "marker": marker,
            "per_run": noise,
            "method": "repeated forwards + same-weight save/reload replay",
        },
        "alpha_mean_intervention": alpha_rows,
        "route_shuffle_intervention": route_rows,
        "route_shuffle_plan": {
            "seed": int(plan["seed"]),
            "n_pairs_total": int(plan["n_pairs_total"]),
            "n_pairs_changed": int(plan["n_pairs_changed"]),
            "molecules_covered": int(plan["molecules_covered"]),
            "plan_file": "route_shuffle_plan.npz",
        },
        "exploratory_reading": reading,
        "mean_codes": {
            "mean_alpha_sha256": hashlib.sha256(
                np.asarray(mean_codes["mean_alpha"], np.float64).tobytes()).hexdigest(),
            "fit_rel_err_median": float(mean_codes["fit_rel_err_median"]),
            "fit_alpha_nnz": float(mean_codes["fit_alpha_nnz"]),
        },
        "cost": cost,
        "g_abs_p90_fit": g_p90,
        "scope": (
            "single body seed (0), development comparison on the historical dev rows "
            "(union of the old select/confirm), repeatedly used by this line — never "
            "an independent confirm, never official valid/test; the reading is a "
            "single-seed directional screen, not significance or repeatability"
        ),
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "terminal_eval.json", payload)
    log(f"[terminal-eval] reading={reading['branch']} R-BASE y {point_deltas['R-BASE_y']:+.5f} "
        f"g {point_deltas['R-BASE_g']:+.5f} in {payload['seconds']:.0f}s")
    return payload


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="source-checks", choices=(
        "source-checks", "pretrain-checks", "smoke", "train", "terminal-eval"))
    parser.add_argument("--arm", default=CONTROL, choices=ARMS)
    parser.add_argument("--seed", type=int, default=0, choices=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.stage == "source-checks":
        source_checks(out_dir=out_dir)
    elif args.stage == "pretrain-checks":
        pretrain_checks(out_dir=out_dir)
    elif args.stage == "smoke":
        run_smoke(device_name=args.device, out_dir=out_dir)
    elif args.stage == "train":
        train_run(args.arm, args.seed, device_name=args.device, out_dir=out_dir)
    else:
        terminal_eval(device_name=args.device, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
