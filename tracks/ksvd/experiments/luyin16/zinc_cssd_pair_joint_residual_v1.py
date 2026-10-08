"""ZINC CSSD pair joint residual v1: does a SHARED, endpoint-swap-symmetric
joint residual reading BOTH full endpoints and the existing relation improve
the complete y_raw and g of the C consumer?

Round: ``zinc_cssd_pair_joint_residual_v1`` (single body seed 0, exploratory;
study ``zinc-context-gap``; design basis 5d33c9a — the
``zinc_cssd_relation_projection_v1`` terminal results).

The working base is C (``SHARED_MLP`` of the relation-projection round): the
verified DICT consumer whose relation side compresses every pair to the shared
nonlinear endpoint projection ``C: E_i -> 48`` (no-bias 144->180->48 SiLU) and
then computes the per-coordinate symmetric statistics
``[left+right, |left-right|, left*right*gate, relation]``.  C is a directional
single-seed lead that has NOT been repeat-confirmed; it is kept as the control
of THIS round and the recipe effect is judged against the same-round CTRL_C,
never against the historical anchor value.

Purchase reason (an INTERFACE property, not a proven error mechanism): the
per-coordinate statistics cannot distinguish the endpoint pair
``{(1,0),(0,1)}`` from ``{(1,1),(0,0)}`` (sum = abs-diff = (1,1), product =
(0,0)) although the co-occurrence on one endpoint differs.  The pair input can
therefore collide; C's learning, the unary/other branches or the relation
itself may compensate — nothing here claims the whole model is non-identifiable
or that the current g error is localised to this interface.

Two arms, one body seed (0), one from-scratch 240-epoch training each:

* ``CTRL_C``         — exactly ``proj.build_arm_round("SHARED_MLP", ...)``
  (the untrained C factory of the relation-projection round, 325,187
  parameters) promoted to this round's model class with NO joint module; its
  init must hash-match the previous round's SHARED_MLP s0 init bit-for-bit.
* ``JOINT_RESIDUAL`` — the SAME untrained factory and every common init
  tensor, plus ONE new module ``joint_residual`` (12,288 parameters,
  337,475 total, +3.78%): ``psi`` = no-bias Linear(144,64) -> SiLU -> no-bias
  Linear(64,48), input ``[left, right, relation]`` (48+48+48 = 144), the last
  layer an ORDINARY Linear followed by nothing (no activation, norm, dropout,
  gamma or extra learnable scale), and

      delta_ij = 0.5 * (psi([left,right,relation]) + psi([right,left,relation]))
      h_pair   = h_old + delta_ij

  Both orderings are evaluated by the SAME psi (concatenated to [2P,144] for
  one call, split and averaged — a STATIC function evaluation of the two
  orders, not two propagation rounds or a relation refresh).  The old h_old
  path is kept verbatim; h_pair enters the ORIGINAL
  ``pool_pair_moments_masked``; there is no new pool, no second environment
  generation and no pair->node write-back.  ``relation`` is the SAME
  post-mask ``relation_encoder`` output the pair input already uses.  The
  pair_encoder / relation_encoder are called once per batch as before; the
  distance gate, ``data.pair_bucket`` and E_i are untouched.

Private init (CPU, ``torch.random.fork_rng(devices=[])``, seed 202610081):
W1 keeps the ``nn.Linear`` default draw, W2 is EXACTLY zero — step-0 delta is
exactly zero, both arms start function-identical, and the common init tensors
are bitwise identical to CTRL_C.  W1's first-backward gradient is zero BY
CONSTRUCTION (never "rescued"); W2 has a reachable non-zero gradient and W1
must be active after a few optimizer steps (fit-only smoke check).

Everything else is the historical recipe: frozen CSSD phi_hat decode (q=1,
32 atoms, top8 tied-IHT10, common kept, phi scaler applied once), Sem108 +
size2 interface, real J incidence, kappa, fusion, the 144->288->144 posterior
bridge, reader, C6 mask, COMP supervision (L = L1(g_hat,g) +
0.5*(L1(ell_hat,ell)+L1(s_hat,s))), Adam lr 1e-3 / coupled wd 1e-5 / batch 128
/ clip 5 / FP32, 240 epochs = 15,120 steps, estimator = the equal-weight FP32
mean of the full member states 236..240, no AMP/DDP/scheduler/early stopping.
All original consumers stay jointly trainable; the basis and Q stay frozen;
E_i is generated before any pair computation and never written back.

Terminal (one-shot, dev = the 1999 historical development rows — a
development comparison, never a new confirm): full dev y_raw MAE (primary)
and g_raw MAE, the single paired contrast JOINT-CTRL_C on y/g (point deltas,
fit deltas, gaps), the canonical-SMILES-group-paired bootstrap (2000 draws,
seed 20261008, shared group picks across BOTH arms and BOTH metrics), the
noise bound eta, and EXACTLY TWO interventions: (A) alpha_v := fit-root mean
alpha on both soups (the verified dictionary-readout entry), (B) the JOINT
soup with ``joint_enabled=False`` (delta := 0, nothing else touched; CTRL_C's
switch is a no-op verified on a small batch).  Single seed: a performance
LEAD screening only — never a repeatability, mechanism or paired-recovery
claim; the historical 0.003 gates and A-E classes are NOT inherited.

Usage (local CPU for source/pretrain checks; res-2 res2-cu124 A100 for the
GPU stages, all through the registered runner)::

    python -m tracks.ksvd.experiments.luyin16.zinc_cssd_pair_joint_residual_v1 \\
        --stage source-checks
    ... --stage pretrain-checks
    ... --stage smoke --device cuda:0
    ... --stage train --arm CTRL_C --seed 0 --device cuda:0
    ... --stage train --arm JOINT_RESIDUAL --seed 0 --device cuda:0
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
    zinc_cssd_relation_projection_v1 as proj,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_full_cycle_target_decomposition_v1 as zftd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw,
)

PROTOCOL_VERSION = "zinc-cssd-pair-joint-residual-v1"
RESULT_SLUG = "zinc_cssd_pair_joint_residual_v1"
TRACK_ROOT = repl.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG
SOURCE_DIR = repl.SOURCE_DIR
#: the previous round's results (READ-ONLY anchor: the C factory, its SHARED_MLP
#: s0 run manifest/init and its terminal numbers as a descriptive anchor)
PREV_RESULT_DIR = TRACK_ROOT / "results" / "zinc_cssd_relation_projection_v1"

ARMS = ("CTRL_C", "JOINT_RESIDUAL")
CONTROL = "CTRL_C"
CANDIDATES = ("JOINT_RESIDUAL",)
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

PROJECTION_IN = int(sc.FULL.d)          # 144 (also the psi input width: 3x48)
PROJECTION_OUT = int(sc.FULL.p)         # 48
N_BUCKETS = int(p2.DISTANCE_BUCKETS)    # 5 (the EXISTING buckets, never remapped)
JOINT_HIDDEN = 64
JOINT_PARAMETERS = PROJECTION_IN * JOINT_HIDDEN + JOINT_HIDDEN * PROJECTION_OUT  # 12,288
CTRL_C_TOTAL_PARAMETERS = 325_187       # proj EXPECTED_CANDIDATE_AUDIT total
JOINT_TOTAL_PARAMETERS = CTRL_C_TOTAL_PARAMETERS + JOINT_PARAMETERS              # 337,475
CTRL_C_BASE_BODY_PARAMETERS = 212_355
JOINT_BASE_BODY_PARAMETERS = CTRL_C_BASE_BODY_PARAMETERS + JOINT_PARAMETERS      # 224,643

EXPECTED_CTRL_C_AUDIT = {
    "total_parameters": CTRL_C_TOTAL_PARAMETERS,
    "base_body_parameters": CTRL_C_BASE_BODY_PARAMETERS,
    "bridge_parameters": 82_944,
    "local_tuple_parameters": 29_888,
    "reader_output_parameters": 2 * 39 + 2,
}
EXPECTED_JOINT_AUDIT = {
    "total_parameters": JOINT_TOTAL_PARAMETERS,
    "base_body_parameters": JOINT_BASE_BODY_PARAMETERS,
    "bridge_parameters": 82_944,
    "local_tuple_parameters": 29_888,
    "reader_output_parameters": 2 * 39 + 2,
}

#: private init stream for the joint residual (distinct from C's 20261008)
JOINT_INIT_SEED = 202610081

N_BOOT = 2000
BOOT_SEED = 20261008
REPLAY_TOL = 1e-4
CROSS_DEVICE_TOL = 1e-5
FLOAT64_SUM_TOL = 1e-4
#: step-0 band: JOINT is mathematically IDENTICAL to CTRL_C (delta exactly 0);
#: the check is bitwise on CPU and uses the cross-device band on GPU
STEP0_BAND_GPU = CROSS_DEVICE_TOL
#: descriptive anchors from the PREVIOUS round (never the same-round control)
HISTORICAL_ANCHORS = {
    "prev_round_C_dev_y_raw": 0.11659,
    "prev_round_C_dev_g_raw": 0.08935,
    "prev_round_BASE_dev_y_raw": 0.11920,
    "prev_round_BASE_dev_g_raw": 0.09169,
}

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
    """The historical component audit — the new top-level ``joint_residual``
    module is a direct child of the model, so it is counted in
    ``total_parameters`` AND ``base_body_parameters``; nothing escapes."""
    return zcs.component_parameter_audit(model)


def arm_expected_audit(arm: str) -> dict[str, int]:
    if arm == CONTROL:
        return dict(EXPECTED_CTRL_C_AUDIT)
    return dict(EXPECTED_JOINT_AUDIT)


def joint_residual_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k.startswith("joint_residual.")}


def non_joint_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if not k.startswith("joint_residual.")}


def prev_shared_mlp_manifest() -> dict[str, Any]:
    """The previous round's SHARED_MLP_s0 manifest (read-only init anchors)."""
    return read_json(PREV_RESULT_DIR / "runs" / "SHARED_MLP_s0" / "manifest.json")


# ---------------------------------------------------------------------------
# 1. the joint residual and the round-local model class
# ---------------------------------------------------------------------------


class JointEndpointResidual(nn.Module):
    """psi: no-bias Linear(144,64) -> SiLU -> no-bias Linear(64,48).

    Reads ``[left, right, relation]`` (48+48+48 = 144).  The last layer is an
    ordinary Linear followed by NOTHING (no activation, norm, dropout, gamma
    or learnable scale).  Private init (seed 202610081 under ``fork_rng``):
    W1 keeps the ``nn.Linear`` default draw, W2 is EXACTLY zero, so the
    step-0 delta is exactly zero and W1's first-backward gradient is zero BY
    CONSTRUCTION (expected, never a defect); W2's gradient is reachable at
    step 0 and W1 must be active after a few optimizer steps (smoke-checked,
    never rescued by re-initialization).

    ``forward`` evaluates the SAME psi on both endpoint orderings — the two
    orderings are concatenated to [2P, 144] for one call and the outputs are
    split and averaged: a STATIC function evaluation of the two orders, not
    two propagation rounds and not a relation refresh.  The module has no
    dropout, so the shared function is exact.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(PROJECTION_IN, JOINT_HIDDEN, bias=False)
        self.w2 = nn.Linear(JOINT_HIDDEN, PROJECTION_OUT, bias=False)
        with torch.no_grad():
            self.w2.weight.zero_()

    def forward(self, left: torch.Tensor, right: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.shape != relation.shape:
            raise RuntimeError(
                f"joint residual inputs must align: {tuple(left.shape)} {tuple(right.shape)} {tuple(relation.shape)}"
            )
        both = torch.cat(
            [torch.cat([left, right, relation], dim=1), torch.cat([right, left, relation], dim=1)],
            dim=0,
        )
        out = self.w2(F.silu(self.w1(both)))
        p = out.shape[0] // 2
        return 0.5 * (out[:p] + out[p:])


class PairJointResidualFullM(proj.RelationProjectionFullM):
    """The C consumer (shared endpoint projection, untouched pair path) with
    an OPTIONAL shared symmetric joint residual.

    Construction is EXACTLY ``proj.build_arm_round("SHARED_MLP", ...)`` — the
    class is promoted afterwards and, for JOINT_RESIDUAL only, the new module
    is attached under the private fork_rng stream.  The masked forward below
    keeps the historical ``RelationProjectionFullM.forward`` order, module
    names, fill semantics and ``readout_perm`` support; the ONLY addition is
    after ``h_old = pair_encoder(pair_input)`` and BEFORE the pair pooling:

        delta_ij = 0.5 * (psi([left,right,relation]) + psi([right,left,relation]))
        pair_value = h_old + delta_ij          (joint_residual present & enabled)

    ``relation`` is the SAME post-mask ``relation_encoder`` output the pair
    input uses; ``left``/``right`` are the SAME masked endpoint views (so a
    ``pair_projection_zero`` mask reaches psi through them).  There is no new
    pool, no second environment generation, no pair->node write-back, no
    attention/MP/recurrence/environment update; ``data.pair_bucket`` is never
    written; E_i is generated once, before any pair computation.

    Mask handling: an explicit mask is the formal path (``cm.C6_MASK``); a
    call WITHOUT a mask defaults to ``cm.C6_MASK`` rather than falling back to
    the unmasked historical super() forward — the joint branch can never be
    silently bypassed.  ``joint_enabled=False`` is the pure terminal switch:
    ONLY delta becomes zero; E, u, relation, the old h_old path, the bucket
    pooling, the reader and every weight are untouched.  On CTRL_C (no
    joint_residual module) the switch is a no-op.
    """

    #: terminal evaluation switch (default ON; False => delta := 0)
    joint_enabled: bool = True
    #: detached eval-time diagnostics (never written in train mode)
    last_joint_delta: torch.Tensor | None = None
    last_h_old: torch.Tensor | None = None
    DEFAULT_MASK = cm.C6_MASK

    @property
    def has_joint_residual(self) -> bool:
        return getattr(self, "joint_residual", None) is not None

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
            mask = self.DEFAULT_MASK  # NEVER bypass the joint branch silently
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
        if self.has_joint_residual and self.joint_enabled:
            delta = self.joint_residual(left, right, relation)
            if not self.training:
                self.last_h_old = pair_value.detach().clone()
                self.last_joint_delta = delta.detach().clone()
            pair_value = pair_value + delta
        elif not self.training:
            self.last_h_old = None
            self.last_joint_delta = None
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


_ENDPOINT_MODE = "shared"  # this round never routes; C's shared projection only


def build_arm_round(
    arm: str,
    payload: prev.TuplePayload,
    kappa_M: float,
    basis_parts: Mapping[str, torch.Tensor],
    seed: int,
    *,
    decode: bool = True,
) -> PairJointResidualFullM:
    """Fresh untrained arm of THIS round, body seed explicitly penetrated.

    Every arm starts from the PRIMARY factory
    ``proj.build_arm_round("SHARED_MLP", payload, kappa_M, basis_parts, seed,
    decode=decode)`` (which itself starts from the original untrained DICT
    factory ``repl.build_arm("DICT", ...)``, completes the historical
    297,539-parameter audit, swaps the endpoint projection to the
    function-preserving SharedEndpointMLP under the private seed 20261008 and
    asserts C's 325,187-parameter contract).  The class is then promoted to
    :class:`PairJointResidualFullM` and ``joint_residual`` stays ``None``
    (CTRL_C) or is attached under ``torch.random.fork_rng(devices=[])`` with
    the private seed 202610081 (JOINT_RESIDUAL) — the fork restores the
    global CPU RNG, so the post-construction RNG state is identical across
    arms and every NON-joint initial tensor is bit-identical to CTRL_C (and
    to the previous round's SHARED_MLP init) by construction.  No historical
    trained soup weight is ever read as an initialisation.  The per-arm
    parameter contract of THIS round is asserted afterwards (the previous
    round's audit objects are NOT inherited).
    """
    if arm not in ARMS:
        raise ValueError(arm)
    model = proj.build_arm_round("SHARED_MLP", payload, kappa_M, basis_parts, int(seed), decode=decode)
    model.__class__ = PairJointResidualFullM
    if getattr(model, "endpoint_mode", None) != _ENDPOINT_MODE:
        raise RuntimeError("this round never routes: the endpoint mode must stay shared")
    model.route_override = None
    model.last_route_stats = None
    model.joint_enabled = True
    model.last_joint_delta = None
    model.last_h_old = None
    model.joint_residual = None
    if arm == "JOINT_RESIDUAL":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(JOINT_INIT_SEED)
            model.joint_residual = JointEndpointResidual()
    audit_row = arm_parameter_audit(model)
    expected = arm_expected_audit(arm)
    if audit_row != expected:
        raise RuntimeError(f"{arm} parameter audit failed: {audit_row} != {expected}")
    if not isinstance(model.local_tuple, repl.LocalTupleEncoderMCSSD):
        raise RuntimeError(f"{arm}: the CSSD encoder is missing")
    if not isinstance(model.pair_projection, proj.SharedEndpointMLP):
        raise RuntimeError(f"{arm}: the shared endpoint projection must be the SharedEndpointMLP")
    if arm == "CTRL_C" and model.joint_residual is not None:
        raise RuntimeError("CTRL_C must not carry a joint residual")
    if arm == "JOINT_RESIDUAL" and not isinstance(model.joint_residual, JointEndpointResidual):
        raise RuntimeError("JOINT_RESIDUAL must carry the JointEndpointResidual")
    return model


def run_dir(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    if arm not in ARMS:
        raise ValueError(arm)
    return Path(out_dir) / "runs" / f"{arm}_s{int(seed)}"


def load_run(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """One run's manifest + soup state (hash-checked; ALWAYS reloaded through
    THIS round's factory — never a historical loader)."""
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
    """Verify every reused object, the historical seed-0 run, the previous
    round's C anchor, the schedule and the per-arm init contracts
    (fit-only, read-only)."""
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ctx = gen.load_round_context()
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
    # the previous round's SHARED_MLP_s0 run (read-only init anchors)
    prev_manifest = prev_shared_mlp_manifest()
    if prev_manifest.get("arm") != "SHARED_MLP" or int(prev_manifest.get("seed", -1)) != 0:
        raise RuntimeError("the previous C anchor is not the SHARED_MLP s0 run")
    prev_init = torch.load(
        PREV_RESULT_DIR / "runs" / "SHARED_MLP_s0" / "init_state.pt",
        map_location="cpu", weights_only=False,
    )
    if state_hash(prev_init) != prev_manifest["init_state_sha256"]:
        raise RuntimeError("the previous C anchor init hash mismatch")

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
        "relation_width_after_encoder": 48,
        "psi_input_width": 3 * 48,
        "distance_buckets": int(N_BUCKETS),
        "pair_projection_weight_shape": [2 * PROJECTION_OUT, PROJECTION_IN],  # SharedEndpointMLP w1
        "relation_input_width": int(p1.RELATION_WIDTH),
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

    # (e) init contracts: CTRL_C must reproduce the previous round's
    #     SHARED_MLP s0 init bit-for-bit; JOINT shares every non-joint tensor
    #     with CTRL_C and its module attach is fork_rng RNG-neutral with
    #     W2 exactly zero and W1 a live default draw
    init_contracts: dict[str, Any] = {}
    seed_everything(SEED)
    ctrl = build_arm_round(CONTROL, payload, kappa_M, basis_parts, SEED)
    ctrl_state = gen._capture_state(ctrl)
    ctrl_hash = state_hash(ctrl_state)
    if ctrl_hash != prev_manifest["init_state_sha256"]:
        raise RuntimeError("CTRL_C init does not reproduce the previous round's SHARED_MLP s0 init")
    if not all(torch.equal(ctrl_state[k], prev_init[k]) for k in sorted(ctrl_state)):
        raise RuntimeError("CTRL_C init tensors differ from the previous round's SHARED_MLP s0 init")
    rng_after_ctrl = torch.get_rng_state().clone()
    seed_everything(SEED)
    joint = build_arm_round("JOINT_RESIDUAL", payload, kappa_M, basis_parts, SEED)
    if not torch.equal(rng_after_ctrl, torch.get_rng_state()):
        raise RuntimeError("JOINT_RESIDUAL construction disturbed the global CPU RNG")
    joint_state = gen._capture_state(joint)
    nb_ctrl, nb_joint = non_joint_state(ctrl_state), non_joint_state(joint_state)
    if sorted(nb_ctrl) != sorted(nb_joint):
        raise RuntimeError("non-joint state key sets differ vs CTRL_C")
    mismatched = [k for k in nb_ctrl if not torch.equal(nb_ctrl[k], nb_joint[k])]
    if mismatched:
        raise RuntimeError(f"non-joint init tensors differ vs CTRL_C at {mismatched}")
    jr = joint_residual_state(joint_state)
    joint_checks = {
        "w1_shape": list(jr["joint_residual.w1.weight"].shape),
        "w2_shape": list(jr["joint_residual.w2.weight"].shape),
        "no_bias": True,
        "w2_exactly_zero": bool(torch.equal(jr["joint_residual.w2.weight"], torch.zeros_like(jr["joint_residual.w2.weight"]))),
        "w1_default_draw_nonzero": bool(float(jr["joint_residual.w1.weight"].abs().sum()) > 0.0),
        "private_seed": JOINT_INIT_SEED,
        "joint_residual_parameters": int(sum(v.numel() for v in jr.values())),
    }
    if joint_checks["joint_residual_parameters"] != JOINT_PARAMETERS:
        raise RuntimeError("joint residual parameter count wrong")
    if not joint_checks["w2_exactly_zero"] or not joint_checks["w1_default_draw_nonzero"]:
        raise RuntimeError(f"joint residual init contract failed: {joint_checks}")
    # a second JOINT construction is bit-identical (private stream, no leakage)
    seed_everything(SEED)
    joint_again = build_arm_round("JOINT_RESIDUAL", payload, kappa_M, basis_parts, SEED)
    if state_hash(gen._capture_state(joint_again)) != state_hash(joint_state):
        raise RuntimeError("JOINT_RESIDUAL construction is not deterministic")
    init_contracts = {
        "ctrl_c_init_sha256": ctrl_hash,
        "ctrl_c_matches_prev_shared_mlp_s0": True,
        "prev_shared_mlp_s0_init_sha256": prev_manifest["init_state_sha256"],
        "non_joint_bitwise_equal_across_arms": True,
        "joint_residual": joint_checks,
        "joint_residual_sha256": state_hash(joint_residual_state(joint_state)),
        "rng_stream_identical_after_construction": True,
        "joint_init_private_seed": JOINT_INIT_SEED,
    }

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
        "prev_anchor_dir": str(PREV_RESULT_DIR.relative_to(TRACK_ROOT)),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "dim_contract": dim_contract,
        "historical_schedule_sha256": old["manifest"]["schedule_sha256"],
        "rebuilt_schedule_sha256": schedule_hash,
        "schedule_bit_identical_240": True,
        "init_contracts": init_contracts,
        "parameter_contracts": {arm: arm_expected_audit(arm) for arm in ARMS},
        "joint_residual_parameters": JOINT_PARAMETERS,
        "total_parameters": {arm: arm_expected_audit(arm)["total_parameters"] for arm in ARMS},
        "frozen_basis_hashes": basis_parts["hashes"],
        "q_soup_sha256": state_hash(ctx["q_soup"]),
        "old_run": {
            "run_dir": str(old["rdir"].relative_to(TRACK_ROOT)),
            "manifest_hashes_verified": True,
            "init_state_sha256": old["manifest"]["init_state_sha256"],
            "schedule_sha256": old["manifest"]["schedule_sha256"],
        },
        "reuse_note": (
            "no basis/Q/prep/target refit; the previous round's C s0 init is the "
            "bitwise anchor of CTRL_C; official valid/test never loaded"
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


def _joint_grad_norms(model: PairJointResidualFullM) -> dict[str, Any]:
    jr = model.joint_residual
    if jr is None:
        return {"note": "no joint residual"}
    g1 = jr.w1.weight.grad
    g2 = jr.w2.weight.grad
    return {
        "w1_grad_norm": float(g1.norm()) if g1 is not None else None,
        "w2_grad_norm": float(g2.norm()) if g2 is not None else None,
    }


def pretrain_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """The meaningful fit-only pre-training checks of this round.

    (a) CTRL_C parity: this round's CTRL_C forward/component/aux path is
        bit-identical to the previous round's untouched SHARED_MLP factory
        under cm.C6_MASK;
    (b) joint module contracts (shapes/no-bias/counts, W2 exactly zero, W1
        live), endpoint-swap symmetry of delta BY CONSTRUCTION, and the small
        2-D collision demo: the per-coordinate pair statistics cannot
        distinguish {(1,0),(0,1)} from {(1,1),(0,0)} while a psi-like shared
        function on [left,right] can (function-family expressiveness only —
        never a claim that training will separate them);
    (c) step-0 agreement: JOINT vs CTRL_C bitwise on CPU (delta exactly 0)
        and JOINT vs the previous round's SHARED_MLP factory;
    (d) same-forward component identity (torch exact, float64 sum in band);
    (e) invariances on fit molecules: batch composition/order, label
        independence, pair-row permutation, node relabelling, endpoint swap
        of the pair input AND of delta (real inputs);
    (f) mask/fill wiring: the joint branch reads the MASKED views and the
        MASKED relation (forward-hook evidence), pair_zero_blocks pooling is
        not bypassed, and with joint_enabled=False the JOINT responses to
        every probe mask are bitwise identical to CTRL_C's;
    (g) a small-batch fit-only optimization smoke (3 steps per arm): CTRL_C
        gradients healthy; JOINT W1 zero-grad at step 0 BY CONSTRUCTION, W2
        grad > 0 at step 0, W1 active and delta non-zero by step 3; CSSD
        buffers never in the optimizer/state dict; basis hashes unchanged;
    (h) save -> correct-factory reload -> replay (CPU bit-exact) and
        wrong-factory strict load blocked;
    (i) the joint_enabled=False switch with an artificially non-zero W2 (a
        temporary fit-only copy, never a training arm, never written to the
        formal init): ONLY delta is affected — predictions/aux bitwise equal
        CTRL_C when off and different when on; CTRL_C's switch is a no-op.
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
    targets = gen.load_round_context()["objects"]["targets"]
    g_all = np.asarray(targets["g"], np.float64)
    ell_all = np.asarray(targets["ell"], np.float64)
    s_all = np.asarray(targets["s"], np.float64)
    target_g = torch.as_tensor(g_all[fit_idx], dtype=torch.float32)
    target_ell = torch.as_tensor(ell_all[fit_idx], dtype=torch.float32)
    target_s = torch.as_tensor(s_all[fit_idx], dtype=torch.float32)
    checks: dict[str, Any] = {}
    smoke_rows = list(range(16))

    def build(arm: str, seed: int = SEED) -> PairJointResidualFullM:
        seed_everything(int(seed))
        return build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))

    batch16 = zftd.make_batch(fit_data, smoke_rows, target_g, device)

    # (a) CTRL_C parity vs the previous round's untouched SHARED_MLP factory
    ctrl = build(CONTROL)
    raw_c = proj.build_arm_round("SHARED_MLP", payload, kappa_M, basis_parts, SEED, decode=True)
    ctrl_state, raw_state = gen._capture_state(ctrl), gen._capture_state(raw_c)
    if sorted(ctrl_state) != sorted(raw_state) or state_hash(ctrl_state) != state_hash(raw_state):
        raise RuntimeError("CTRL_C init tensors differ from the previous round's SHARED_MLP factory")
    ctrl.eval(); raw_c.eval()
    with torch.no_grad():
        p_ctrl, aux_ctrl = ctrl(batch16, mask=cm.C6_MASK, return_aux=True)
        p_raw, aux_raw = raw_c(batch16, mask=cm.C6_MASK, return_aux=True)
        c_ctrl, c_raw = ctrl.reader.components(), raw_c.reader.components()
    parity = {
        "prediction_max_abs": float((p_ctrl - p_raw).abs().max()),
        "components_max_abs": float((c_ctrl - c_raw).abs().max()),
        "E_max_abs": float((aux_ctrl["E"] - aux_raw["E"]).abs().max()),
        "unary_max_abs": float((aux_ctrl["unary"] - aux_raw["unary"]).abs().max()),
        "relation_readout_max_abs": float(
            (aux_ctrl["relation_readout"] - aux_raw["relation_readout"]).abs().max()),
    }
    if any(v != 0.0 for v in parity.values()):
        raise RuntimeError(f"CTRL_C forward is not bit-identical to the previous C path: {parity}")
    checks["ctrl_c_parity_vs_prev_factory"] = parity

    # (b) joint module contracts + symmetry + the 2-D collision demo
    joint = build("JOINT_RESIDUAL")
    joint.eval()
    jr = joint.joint_residual
    contracts = {
        "w1_shape": list(jr.w1.weight.shape),
        "w2_shape": list(jr.w2.weight.shape),
        "no_bias": bool(jr.w1.bias is None and jr.w2.bias is None),
        "parameters": int(sum(p.numel() for p in jr.parameters())),
        "w2_init_zero": bool(float(jr.w2.weight.abs().sum()) == 0.0),
        "w1_init_nonzero": bool(float(jr.w1.weight.abs().sum()) > 0.0),
    }
    if contracts["parameters"] != JOINT_PARAMETERS or contracts["parameters"] != 12_288:
        raise RuntimeError("joint residual parameter count wrong")
    lft = torch.randn(7, PROJECTION_OUT, generator=torch.Generator().manual_seed(11))
    rgt = torch.randn(7, PROJECTION_OUT, generator=torch.Generator().manual_seed(12))
    rel = torch.randn(7, PROJECTION_OUT, generator=torch.Generator().manual_seed(13))
    with torch.no_grad():
        d0 = jr(lft, rgt, rel)
        d_swap = jr(rgt, lft, rel)
    contracts["delta_endpoint_swap_symmetric"] = bool(torch.equal(d0, d_swap))
    contracts["delta_step0_exactly_zero"] = bool(float(d0.abs().sum()) == 0.0)
    if not contracts["delta_endpoint_swap_symmetric"] or not contracts["delta_step0_exactly_zero"]:
        raise RuntimeError(f"joint module contracts failed: {contracts}")
    # 2-D collision demo (interface expressiveness only)
    e1, e2 = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    e3, e4 = torch.tensor([[1.0, 1.0]]), torch.tensor([[0.0, 0.0]])
    stats_a = [e1 + e2, torch.abs(e1 - e2), e1 * e2]
    stats_b = [e3 + e4, torch.abs(e3 - e4), e3 * e4]
    collision = {
        "pair_a": [[1.0, 0.0], [0.0, 1.0]],
        "pair_b": [[1.0, 1.0], [0.0, 0.0]],
        "sum_absdiff_product_identical": bool(
            all(torch.equal(sa, sb) for sa, sb in zip(stats_a, stats_b))),
        "concat_distinguishable": bool(not torch.equal(
            torch.cat([e1, e2], dim=1), torch.cat([e3, e4], dim=1))),
        "note": (
            "per-coordinate sum/abs-diff/product collide on {(1,0),(0,1)} vs "
            "{(1,1),(0,0)}; a shared function on [left,right] separates them — "
            "an interface-property demo, never a trained-behaviour claim"
        ),
    }
    if not collision["sum_absdiff_product_identical"] or not collision["concat_distinguishable"]:
        raise RuntimeError("the 2-D collision demo is broken")
    checks["joint_module_contracts"] = contracts
    checks["collision_demo"] = collision

    # (c) step-0 agreement (bitwise on CPU)
    joint.eval()
    with torch.no_grad():
        p_joint, aux_joint = joint(batch16, mask=cm.C6_MASK, return_aux=True)
        c_joint = joint.reader.components()
    step0 = {
        "joint_vs_ctrl_c_max_abs": float((p_joint - p_ctrl).abs().max()),
        "joint_vs_prev_factory_max_abs": float((p_joint - p_raw).abs().max()),
        "components_max_abs": float((c_joint - c_ctrl).abs().max()),
        "unary_max_abs": float((aux_joint["unary"] - aux_ctrl["unary"]).abs().max()),
        "relation_readout_max_abs": float(
            (aux_joint["relation_readout"] - aux_ctrl["relation_readout"]).abs().max()),
    }
    if any(v != 0.0 for v in step0.values()):
        raise RuntimeError(f"JOINT step 0 is not bit-identical to CTRL_C on CPU: {step0}")
    checks["step0_agreement"] = step0

    # (d) same-forward component identity (fit rows, one batch, both arms)
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
    subset = fit_data[:24]
    invariance: dict[str, Any] = {}
    for arm in ARMS:
        model = build(arm)
        model.eval()

        def predict(rows: Sequence[Any], indices: Sequence[int] | None = None) -> np.ndarray:
            b = zftd.make_batch(rows, list(range(len(rows))) if indices is None else list(indices),
                                torch.zeros(len(rows)), device)
            with torch.no_grad():
                return model(b, mask=cm.C6_MASK).view(-1).numpy().astype(np.float64)

        order = list(range(len(subset)))
        singles = np.concatenate([predict(subset[i : i + 1]) for i in order])
        shuffled = [order[(i * 7 + 3) % len(order)] for i in order]
        grouped = predict(subset, shuffled)
        d_order = float(np.abs(singles[np.asarray(shuffled)] - grouped).max())
        b_y0 = zftd.make_batch(fit_data, list(range(8)), torch.zeros(8), device)
        b_y1 = zftd.make_batch(fit_data, list(range(8)), torch.ones(8), device)
        with torch.no_grad():
            d_label = float((model(b_y0, mask=cm.C6_MASK) - model(b_y1, mask=cm.C6_MASK)).abs().max())
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
        # node relabelling (payload swapped exactly like the proven pattern)
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
        # endpoint swap: the old pair input AND the joint delta are symmetric
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
            d_delta_swap = 0.0
            if model.has_joint_residual:
                d_delta_swap = float((model.joint_residual(left, right, relation)
                                      - model.joint_residual(right, left, relation)).abs().max())
        invariance[arm] = {
            "batch_order_max_abs": d_order,
            "label_independence_max_abs": d_label,
            "pair_row_permutation_max_abs": d_pairs,
            "node_relabel_max_abs": d_relabel,
            "endpoint_swap_pair_input_max_abs": float((pi - pi_swap).abs().max()),
            "endpoint_swap_joint_delta_max_abs": d_delta_swap,
        }
        if d_order > 1e-5:
            raise RuntimeError(f"{arm}: batch/order invariance failed ({d_order})")
        if d_label != 0.0:
            raise RuntimeError(f"{arm}: labels entered the forward path")
        if d_pairs > 1e-5:
            raise RuntimeError(f"{arm}: pair-row order changed predictions ({d_pairs})")
        if d_relabel > 1e-5:
            raise RuntimeError(f"{arm}: node relabelling changed predictions ({d_relabel})")
        if float((pi - pi_swap).abs().max()) != 0.0:
            raise RuntimeError(f"{arm}: endpoint swap changed the pair input")
        if model.has_joint_residual and d_delta_swap != 0.0:
            raise RuntimeError(f"{arm}: endpoint swap changed the joint delta")
    checks["invariances"] = invariance

    # (f) mask/fill wiring of the new branch
    mask_wiring: dict[str, Any] = {}
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        left_i, right_i, rel_i = (t.detach().clone() for t in inputs[:3])
        captured["psi_input"] = torch.cat([left_i, right_i, rel_i], dim=1)

    pair_input_capture: dict[str, torch.Tensor] = {}

    def _pair_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        pair_input_capture["pair_input"] = inputs[0].detach().clone()

    model = build("JOINT_RESIDUAL")
    model.eval()
    h1 = model.joint_residual.register_forward_hook(_hook)
    h2 = model.pair_encoder.register_forward_hook(_pair_hook)
    try:
        with torch.no_grad():
            model(batch16, mask=cm.C6_MASK)
            psi_nat = captured["psi_input"].clone()
            pin_nat = pair_input_capture["pair_input"].clone()
            model(batch16, mask=audit.AuditMask(pair_projection_zero=True))
            psi_zp = captured["psi_input"].clone()
            model(batch16, mask=audit.AuditMask(relation_zero_groups=("overlap", "path_count")))
            psi_zr = captured["psi_input"].clone()
            pin_zr = pair_input_capture["pair_input"].clone()
    finally:
        h1.remove()
        h2.remove()
    mask_wiring["psi_reads_masked_views"] = {
        "natural_input_nonzero": bool(float(psi_nat.abs().sum()) > 0.0),
        "pair_projection_zero_endpoints_exactly_zero": bool(float(psi_zp[:, :96].abs().sum()) == 0.0),
        "pair_projection_zero_relation_intact": bool(float(psi_zp[:, 96:].abs().sum()) > 0.0),
    }
    if not all(mask_wiring["psi_reads_masked_views"].values()):
        raise RuntimeError("the joint branch does not read the masked endpoint views")
    mask_wiring["relation_shared_with_pair_input"] = {
        "natural_relation_blocks_equal": bool(torch.equal(psi_nat[:, 96:], pin_nat[:, 144:])),
        "masked_relation_blocks_equal": bool(torch.equal(psi_zr[:, 96:], pin_zr[:, 144:])),
        "masked_relation_changed": bool(not torch.equal(psi_zr[:, 96:], psi_nat[:, 96:])),
    }
    if not all(mask_wiring["relation_shared_with_pair_input"].values()):
        raise RuntimeError("psi and the pair input do not share the same masked relation tensor")
    # pair_zero_blocks pooling not bypassed (behavioral, both arms)
    for arm in ARMS:
        model_m = build(arm)
        model_m.eval()
        with torch.no_grad():
            p_nat = model_m(batch16, mask=cm.C6_MASK).detach().clone()
            p_zb = model_m(batch16, mask=audit.AuditMask(pair_zero_blocks=("first",))).detach().clone()
        d = float((p_nat - p_zb).abs().max())
        mask_wiring[f"{arm}_pair_zero_blocks_changes_predictions"] = d
        if arm == CONTROL:
            ctrl_zb = d
    if float(mask_wiring[f"{CONTROL}_pair_zero_blocks_changes_predictions"]) <= 0.0:
        raise RuntimeError("pair_zero_blocks does not reach the pooled readout")
    # with joint_enabled=False, every probe-mask response is CTRL_C-bitwise
    switch_equivalence: dict[str, Any] = {}
    model_off = build("JOINT_RESIDUAL")
    with torch.no_grad():
        model_off.joint_residual.w2.weight.copy_(
            torch.randn_like(model_off.joint_residual.w2.weight) * 0.05
        )  # artificial non-zero W2: the branch is LIVE in this temporary copy
    model_off.eval()
    ctrl_model = build(CONTROL)
    ctrl_model.eval()
    probe_masks = {
        "c6": cm.C6_MASK,
        "pair_projection_zero": audit.AuditMask(pair_projection_zero=True),
        "relation_zero_groups": audit.AuditMask(relation_zero_groups=("overlap", "path_count")),
        "pair_zero_blocks": audit.AuditMask(pair_zero_blocks=("first",)),
        "gate_off": audit.AuditMask(gate_off=True),
    }
    for name, m in probe_masks.items():
        with torch.no_grad():
            p_ctrl_m = ctrl_model(batch16, mask=m).detach().clone()
            model_off.joint_enabled = True
            p_on = model_off(batch16, mask=m).detach().clone()
            model_off.joint_enabled = False
            p_off = model_off(batch16, mask=m).detach().clone()
            _, aux_ctrl_m = ctrl_model(batch16, mask=m, return_aux=True)
            model_off.joint_enabled = False
            _, aux_off = model_off(batch16, mask=m, return_aux=True)
            model_off.joint_enabled = True
        switch_equivalence[name] = {
            "off_vs_ctrl_c_max_abs": float((p_off - p_ctrl_m).abs().max()),
            "on_vs_ctrl_c_max_abs": float((p_on - p_ctrl_m).abs().max()),
            "off_aux_E_max_abs": float((aux_off["E"] - aux_ctrl_m["E"]).abs().max()),
            "off_aux_unary_max_abs": float((aux_off["unary"] - aux_ctrl_m["unary"]).abs().max()),
            "off_aux_relation_readout_max_abs": float(
                (aux_off["relation_readout"] - aux_ctrl_m["relation_readout"]).abs().max()),
        }
        row = switch_equivalence[name]
        if row["off_vs_ctrl_c_max_abs"] != 0.0 or row["on_vs_ctrl_c_max_abs"] == 0.0:
            raise RuntimeError(f"joint_enabled switch behaviour wrong under mask {name}: {row}")
        if any(row[k] != 0.0 for k in ("off_aux_E_max_abs", "off_aux_unary_max_abs", "off_aux_relation_readout_max_abs")):
            raise RuntimeError(f"joint_enabled=False changed an upstream output under mask {name}")
    # CTRL_C switch no-op
    with torch.no_grad():
        p_ctrl_on = ctrl_model(batch16, mask=cm.C6_MASK).detach().clone()
        ctrl_model.joint_enabled = False
        p_ctrl_off = ctrl_model(batch16, mask=cm.C6_MASK).detach().clone()
        ctrl_model.joint_enabled = True
    switch_equivalence["ctrl_c_switch_noop_max_abs"] = float((p_ctrl_on - p_ctrl_off).abs().max())
    if switch_equivalence["ctrl_c_switch_noop_max_abs"] != 0.0:
        raise RuntimeError("the CTRL_C switch is not a no-op")
    mask_wiring["joint_enabled_switch_equivalence"] = switch_equivalence
    checks["mask_wiring"] = mask_wiring

    # (g) small-batch fit-only optimization smoke (3 optimizer steps per arm)
    grads: dict[str, Any] = {}
    for arm in ARMS:
        seed_everything(0)
        model = build(arm)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        n_opt = int(sum(p.numel() for group in optimizer.param_groups for p in group["params"]))
        if n_opt != arm_expected_audit(arm)["total_parameters"]:
            raise RuntimeError(f"{arm}: optimizer parameters {n_opt} != contract total")
        named_total = int(sum(p.numel() for _, p in model.named_parameters()))
        if named_total != n_opt:
            raise RuntimeError(f"{arm}: named_parameters total {named_total} != optimizer total {n_opt}")
        state_keys = [k for k in model.state_dict() if k.startswith("local_tuple.cssd_")]
        if state_keys:
            raise RuntimeError(f"{arm}: CSSD basis leaked into the state dict: {state_keys}")
        encoder_hashes_before = model.local_tuple.frozen_basis_hashes()
        finite_ok = True
        loss_values: list[float] = []
        joint_grad_trace: list[dict[str, Any]] = []
        delta_active = False
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
                joint_grad_trace.append({"step": 0, **_joint_grad_norms(model)})
            optimizer.step()
            finite_ok = finite_ok and bool(torch.isfinite(prediction).all() and torch.isfinite(loss))
            loss_values.append(float(loss.detach()))
        # delta activity after 3 steps (fresh functional check on the live module)
        delta_active = False
        if model.has_joint_residual:
            model.eval()
            with torch.no_grad():
                aux = model(b, mask=cm.C6_MASK, return_aux=True)
                E1 = aux[1]["E"]
                left, right = model.endpoint_views(
                    E1, b.pair_index[0], b.pair_index[1], b, cm.C6_MASK)
                rel_in = b.pair_relation[:, list(p1.P1_RELATION_INDICES)]
                relation = model.relation_encoder(rel_in)
                delta_now = model.joint_residual(left, right, relation)
            delta_active = bool(float(delta_now.abs().sum()) > 0.0)
            joint_grad_trace.append({"step": 2, **_joint_grad_norms(model)})
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
            "joint_grad_trace": joint_grad_trace,
            "delta_active_after_3_steps": delta_active,
            "cssd_buffers_not_in_optimizer": True,
            "optimizer_matches_named_parameters": True,
            "frozen_basis_hashes_unchanged": True,
        }
    c_trace = grads["JOINT_RESIDUAL"]["joint_grad_trace"]
    if c_trace[0]["w1_grad_norm"] != 0.0:
        raise RuntimeError("JOINT W1 should be zero-grad at step 0 BY CONSTRUCTION")
    if not c_trace[0]["w2_grad_norm"] > 0.0:
        raise RuntimeError("JOINT W2 has no reachable gradient at step 0")
    if not c_trace[1]["w1_grad_norm"] > 0.0:
        raise RuntimeError("JOINT W1 never received gradient by step 3")
    if not grads["JOINT_RESIDUAL"]["delta_active_after_3_steps"]:
        raise RuntimeError("JOINT delta is not active after 3 steps")
    checks["gradient_smoke"] = grads

    # (h) save -> correct-factory reload -> replay (CPU bit-exact)
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
        if not cross_blocked:
            raise RuntimeError(f"{arm}: cross-factory strict load was NOT blocked")
        replay[arm] = {"reload_replay_max_abs": d, "cross_factory_strict_load_blocked": cross_blocked}
    checks["save_reload_replay"] = replay

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


def _probe_channels(model: PairJointResidualFullM, arm: str, epoch: int, step: int, total_norm: float) -> dict[str, Any]:
    """The historical local-channel probe + this round's joint grads."""
    entry = repl._probe_local_channel(model, "DICT", epoch, step, total_norm)
    entry["arm"] = arm
    entry["joint"] = _joint_grad_norms(model)
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

    Clones the source rounds' train loop exactly (schedule, init, optimizer,
    batch order, loss, probes, curve, soup semantics) with only this round's
    per-arm factory and deterministic state capture at the member/checkpoint
    epochs.  During training nothing is built or scored: only detach/clone
    captures; dev is never touched; joint_enabled stays True; no route
    override exists.
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
    # cross-arm non-joint identity (fresh CPU constructions, fit-free)
    ctrl_state = gen._capture_state(
        build_arm_round(CONTROL, payload, kappa_M, basis_parts, int(seed))
    )
    nb_init, nb_ctrl = non_joint_state(init_state), non_joint_state(ctrl_state)
    if sorted(nb_init) != sorted(nb_ctrl):
        raise RuntimeError(f"{arm} s{seed}: non-joint state key sets differ vs CTRL_C")
    mismatched = [k for k in nb_init if not torch.equal(nb_init[k], nb_ctrl[k])]
    if mismatched:
        raise RuntimeError(f"{arm} s{seed}: non-joint init differs vs CTRL_C at {mismatched}")
    prev_manifest = prev_shared_mlp_manifest()
    if arm == CONTROL and init_hash != prev_manifest["init_state_sha256"]:
        raise RuntimeError(f"{arm} s{seed}: init does not reproduce the previous round's SHARED_MLP s0 init")
    if arm != CONTROL and state_hash(non_joint_state(init_state)) != prev_manifest["init_state_sha256"]:
        raise RuntimeError(f"{arm} s{seed}: non-joint init does not reproduce the previous round's SHARED_MLP s0 init")
    joint_init_sha256 = state_hash(joint_residual_state(init_state))
    del ctrl_state, nb_init, nb_ctrl

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
    named_total = int(sum(p.numel() for _, p in replay.named_parameters()))
    if named_total != post_audit["total_parameters"]:
        raise RuntimeError(f"{arm} s{seed}: named_parameters total {named_total} != audit total")
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
        "init_matches_prev_shared_mlp_s0": bool(arm == CONTROL),
        "init_non_joint_matches_prev_shared_mlp_s0": True,
        "init_non_joint_sha256": state_hash(non_joint_state(init_state)),
        "init_joint_residual_sha256": joint_init_sha256,
        "joint_enabled_during_training": True,
        "non_joint_bitwise_equal_ctrl_c": True,
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
        "joint_residual_parameters": JOINT_PARAMETERS if arm == "JOINT_RESIDUAL" else 0,
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
    log(f"[smoke] both arms ok on {device} in {smoke['seconds']:.1f}s")
    return smoke


# ---------------------------------------------------------------------------
# 6. terminal helpers (this round's factory/dirs only)
# ---------------------------------------------------------------------------


def _load_soup_model(
    arm: str, seed: int, ctx: Mapping[str, Any], *, out_dir: Path,
) -> tuple[PairJointResidualFullM, dict[str, torch.Tensor]]:
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


def _collect_delta_stats(
    model: PairJointResidualFullM,
    rows: Sequence[Any],
    device: torch.device,
) -> dict[str, Any]:
    """Eval-mode delta/h_old scale statistics (detached diagnostics only).

    Replays the same batching as ``gen._predict_rows_checked`` and reads the
    detached ``last_joint_delta`` / ``last_h_old`` recorded by the model in
    eval mode; no RNG is consumed and no weight is touched.
    """
    deltas: list[np.ndarray] = []
    h_olds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            chunk = list(range(start, min(start + BATCH_SIZE, len(rows))))
            batch = zftd.make_batch(rows, chunk, torch.zeros(len(rows)), device)
            model(batch, mask=cm.C6_MASK)
            if model.last_joint_delta is None or model.last_h_old is None:
                raise RuntimeError("eval forward did not record the joint diagnostics")
            deltas.append(model.last_joint_delta.detach().cpu().numpy().astype(np.float64))
            h_olds.append(model.last_h_old.detach().cpu().numpy().astype(np.float64))
    delta = np.concatenate(deltas)
    h_old = np.concatenate(h_olds)
    d_norm = np.linalg.norm(delta, axis=1)
    h_norm = np.linalg.norm(h_old, axis=1)
    rel = d_norm / (h_norm + 1e-12)
    return {
        "n_pairs": int(delta.shape[0]),
        "delta_abs_mean": float(np.abs(delta).mean()),
        "delta_norm_mean": float(d_norm.mean()),
        "delta_norm_median": float(np.median(d_norm)),
        "delta_norm_p95": float(np.percentile(d_norm, 95)),
        "delta_norm_max": float(d_norm.max()),
        "h_old_norm_mean": float(h_norm.mean()),
        "delta_rel_to_h_old_mean": float(rel.mean()),
        "delta_rel_to_h_old_median": float(np.median(rel)),
        "delta_rel_to_h_old_p95": float(np.percentile(rel, 95)),
    }


def paired_bootstrap_contrast(
    errs_y: Mapping[str, np.ndarray],
    errs_g: Mapping[str, np.ndarray],
    group_of_row: np.ndarray,
    *,
    cand: str,
    ctrl: str,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Canonical-SMILES-group-paired bootstrap of Delta_y AND Delta_g for the
    ONE contrast of this round (cand - ctrl).

    2000 draws, fixed seed 20261008; the SAME group picks are shared by both
    arms and BOTH metrics within a draw (y and g share the sampling).
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
    key = f"{cand}|{ctrl}"
    draws = {key: {tag: np.empty(int(n_boot)) for tag in ("y", "g")}}
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_groups, n_groups)   # shared across arms AND metrics
        n = int(cnt[pick].sum())
        m_y = {k: float(sums["y"][k][pick].sum() / n) for k in keys}
        m_g = {k: float(sums["g"][k][pick].sum() / n) for k in keys}
        draws[key]["y"][b] = m_y[f"{cand}_s{SEED}"] - m_y[f"{ctrl}_s{SEED}"]
        draws[key]["g"][b] = m_g[f"{cand}_s{SEED}"] - m_g[f"{ctrl}_s{SEED}"]
    return {
        "n_rows": int(cnt.sum()),
        "n_groups": int(n_groups),
        "n_boot": int(n_boot),
        "boot_seed": int(seed),
        "shared_group_resampling": True,
        "shared_picks_across_arms_and_metrics": True,
        "contrast": key,
        "delta_y_ci95": [float(np.percentile(draws[key]["y"], 2.5)), float(np.percentile(draws[key]["y"], 97.5))],
        "delta_y_mean": float(draws[key]["y"].mean()),
        "delta_g_ci95": [float(np.percentile(draws[key]["g"], 2.5)), float(np.percentile(draws[key]["g"], 97.5))],
        "delta_g_mean": float(draws[key]["g"].mean()),
        "delta_y_ci_contains_zero": bool(
            np.percentile(draws[key]["y"], 2.5) <= 0.0 <= np.percentile(draws[key]["y"], 97.5)),
        "delta_g_ci_contains_zero": bool(
            np.percentile(draws[key]["g"], 2.5) <= 0.0 <= np.percentile(draws[key]["g"], 97.5)),
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
    """The dictionary intervention on one soup (any arm of THIS round).

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


def joint_toggle_intervention(
    seed: int,
    base: Mapping[str, Any],
    *,
    ctx: Mapping[str, Any],
    out_dir: Path,
    device: torch.device,
    marker: float,
    log: Any = print,
) -> dict[str, Any]:
    """Intervention B: the JOINT soup with ``joint_enabled=False`` (delta:=0).

    All old paths and weights untouched; no retraining, no recalibration; the
    E, u, relation, old h_old path, bucket pooling and reader are exactly the
    enabled-path ones.  This is ADDITIONAL-path dependence / direction of
    action — never a substitute for the paired performance result, and a large
    response is never good news by itself.  Also records the delta-scale
    summary relative to h_old and the soup's joint-parameter norms.
    """
    arm = "JOINT_RESIDUAL"
    model, state = _load_soup_model(arm, seed, ctx, out_dir=out_dir)
    model = model.to(device).eval()
    jr_w1 = model.joint_residual.w1.weight.detach()
    jr_w2 = model.joint_residual.w2.weight.detach()
    param_norms = {
        "w1_fro_norm": float(jr_w1.norm()),
        "w2_fro_norm": float(jr_w2.norm()),
        "w2_exactly_zero": bool(float(jr_w2.abs().sum()) == 0.0),
        "w1_row_norm_p95": float(np.percentile(jr_w1.norm(dim=1).cpu().numpy(), 95)),
    }
    delta_stats = _collect_delta_stats(model, ctx["dev_data"], device)
    model.joint_enabled = False
    preds_off = gen._predict_rows_checked(model, ctx["dev_data"], device)
    model.joint_enabled = True
    model.cpu()
    del model

    q_raw = np.asarray(base["q_raw"], np.float64)
    y = np.asarray(base["y"], np.float64)
    g = np.asarray(base["g"], np.float64)
    base_y_raw = np.asarray(base["y_raw"], np.float64)
    base_h = np.asarray(base["h"], np.float64)
    y_raw_off = preds_off["ell_hat"] + preds_off["s_hat"] + q_raw
    d_pred = np.abs(y_raw_off - base_y_raw)
    d_mae_y = float(np.mean(np.abs(y - y_raw_off)) - np.mean(np.abs(y - base_y_raw)))
    d_mae_g = float(np.mean(np.abs(g - preds_off["h"])) - np.mean(np.abs(g - base_h)))
    row = {
        "arm": arm, "seed": int(seed),
        "mechanism": (
            "the JOINT soup re-forwarded with joint_enabled=False (delta := 0); every "
            "old path and weight untouched; no retraining, no recalibration; Q reused "
            "bitwise from the base evaluation"
        ),
        "joint_parameter_norms": param_norms,
        "delta_vs_h_old": delta_stats,
        "abs_dpred": {
            "mean": float(d_pred.mean()),
            "median": float(np.median(d_pred)),
            "p95": float(np.percentile(d_pred, 95)),
            "max": float(d_pred.max()),
        },
        "response_fraction_gt_marker": float((d_pred > float(marker)).mean()),
        "delta_y_raw_mae_disable": d_mae_y,
        "delta_g_mae_disable": d_mae_g,
        "disable_improves_y": bool(d_mae_y < 0.0),
        "disable_improves_g": bool(d_mae_g < 0.0),
        "reading": (
            "branch dependence/direction only: disabling improving the dev MAE means "
            "the additional path is net harmful for THIS fixed checkpoint (a large "
            "response is never good news by itself); never a substitute for the "
            "paired performance comparison"
        ),
    }
    if log:
        log(
            f"[joint-toggle {arm} s{seed}] |dPred| p95 {row['abs_dpred']['p95']:.2e} "
            f"resp {row['response_fraction_gt_marker']:.3f} "
            f"dMAE_y(disable) {d_mae_y:+.2e} dMAE_g(disable) {d_mae_g:+.2e}"
        )
    return row


def ctrl_c_switch_noop_check(
    seed: int,
    *,
    ctx: Mapping[str, Any],
    out_dir: Path,
    device: torch.device,
    n_rows: int = 64,
) -> dict[str, Any]:
    """CTRL_C's joint_enabled switch is a no-op (small-batch verification)."""
    model, _state = _load_soup_model(CONTROL, seed, ctx, out_dir=out_dir)
    model = model.to(device).eval()
    rows = ctx["dev_data"][: int(n_rows)]
    batch = zftd.make_batch(rows, list(range(len(rows))), torch.zeros(len(rows)), device)
    with torch.no_grad():
        p_on = model(batch, mask=cm.C6_MASK).detach().clone()
        model.joint_enabled = False
        p_off = model(batch, mask=cm.C6_MASK).detach().clone()
        model.joint_enabled = True
    model.cpu()
    del model
    d = float((p_on - p_off).abs().max())
    if d != 0.0:
        raise RuntimeError(f"CTRL_C switch is not a no-op (d={d})")
    return {"n_rows": int(n_rows), "switch_noop_max_abs": d, "no_op": True}


def exploratory_reading(
    deltas: Mapping[str, float],
    bootstrap: Mapping[str, Any],
    alpha_rows: Mapping[str, Mapping[str, Any]],
    toggle_row: Mapping[str, Any],
    marker: float,
) -> dict[str, Any]:
    """The frozen single-seed exploratory reading rubric (protocol section 8).

    "Improves" always means the paired dev Delta (JOINT - CTRL_C) < 0 on that
    metric; point estimates decide the branch, the CIs are reported alongside
    and can only annotate uncertainty (a CI crossing 0 is explicitly recorded
    as weak/uncertain — never evidence of absence, never repeatability).  The
    historical 0.003 gate and A-E classes are NOT inherited; this is a
    performance-lead screen, not a mechanism confirmation.
    """
    dy, dg = float(deltas["JOINT-CTRL_C_y"]), float(deltas["JOINT-CTRL_C_g"])
    ci = bootstrap
    ci_y = ci["delta_y_ci95"]
    ci_g = ci["delta_g_ci95"]
    ci_y_cross = bool(ci_y[0] <= 0.0 <= ci_y[1])
    ci_g_cross = bool(ci_g[0] <= 0.0 <= ci_g[1])
    alpha_ok = bool(alpha_rows["JOINT_RESIDUAL"]["abs_dpred"]["p95"] > marker)
    toggle_responds = bool(toggle_row["abs_dpred"]["p95"] > marker)
    improves_both = dy < 0.0 and dg < 0.0
    signs_differ = (dy < 0.0) != (dg < 0.0)
    effect_below_noise = bool(abs(dy) <= marker and abs(dg) <= marker)

    if not alpha_ok:
        branch = "dictionary-readout-lost-no-dictionary-mainline-upgrade"
        note = (
            "JOINT's alpha-mean intervention p95 <= marker: the soup no longer reads "
            "the sparse code substantially — performance is recorded but JOINT is NOT "
            "upgraded to this round's dictionary mainline candidate."
        )
    elif improves_both:
        if not toggle_responds:
            branch = "performance-lead-without-branch-response"
            note = (
                "JOINT improves both dev y and g, but the toggle intervention shows "
                "almost no response: the lead is retained as a training-parameterisation/"
                "trajectory observation and is NOT upgraded to a verified endpoint-joint "
                "readout; the initialization is never re-purposed as a rescue."
            )
        elif ci_y_cross or ci_g_cross:
            branch = "joint-residual-single-seed-lead-weak-uncertain"
            note = (
                "JOINT improves both dev y and g vs the same-round CTRL_C (the shared "
                "symmetric endpoint joint residual), the branch genuinely responds and "
                "the dictionary readout is intact — but at least one CI crosses 0: a "
                "WEAK/UNCERTAIN single-seed performance lead (candidate saved, no "
                "automatic upgrade, no extra seeds purchased; added-capacity and "
                "trajectory explanations stay open)."
            )
        else:
            branch = "joint-residual-single-seed-lead"
            note = (
                "JOINT improves both dev y and g vs the same-round CTRL_C with both CIs "
                "excluding 0: a single-seed performance lead of the shared symmetric "
                "endpoint joint residual (candidate saved, no automatic upgrade; still "
                "ONE seed — never a repeatability or paired-information-recovery claim)."
            )
    elif signs_differ:
        branch = "mixed-inconclusive"
        note = (
            "y and g move in opposite directions: a mixed/inconclusive result, "
            "reported as such — never packaged as a success."
        )
    elif effect_below_noise:
        branch = "mixed-inconclusive"
        note = (
            "Both |deltas| are at or below the noise marker: no usable signal this "
            "round (mixed/inconclusive)."
        )
    else:
        branch = "joint-residual-configuration-not-retained"
        note = (
            "JOINT does not improve both dev metrics (or is clearly worse): THIS "
            "configuration is not retained and its additional budget is closed — no "
            "width/depth/residual-scale sweeps, and no verdict on relation fusion in "
            "general."
        )
    return {
        "branch": branch,
        "note": note,
        "point_estimates": {"JOINT-CTRL_C_y": dy, "JOINT-CTRL_C_g": dg},
        "joint_improves_both_vs_ctrl_c": bool(improves_both),
        "ci_y_contains_zero": ci_y_cross,
        "ci_g_contains_zero": ci_g_cross,
        "signs_agree": not signs_differ,
        "dictionary_readout_responds": alpha_ok,
        "joint_branch_responds": toggle_responds,
        "disable_improves_y": bool(toggle_row.get("disable_improves_y", False)),
        "disable_improves_g": bool(toggle_row.get("disable_improves_g", False)),
        "single_seed_caveat": (
            "a directional performance lead is this seed's screening result only — not "
            "a statistical-significance, repeatability or mechanism claim; JOINT has "
            "+12,288 parameters (+3.78%), so capacity/implicit-regularisation/trajectory "
            "explanations are NEVER excluded by this round"
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
    prev_manifest = prev_shared_mlp_manifest()
    expected_encoder_hashes = {
        "U": array_sha256(basis_parts["U"].numpy()),
        "common_rms": array_sha256(basis_parts["common_rms"].numpy()),
        "Dbar": array_sha256(basis_parts["Dbar"].numpy()),
    }

    # (1) roster verification (before any scoring)
    checks = {
        "runs_completed": True, "steps_ok": True, "schedule_ok": True,
        "init_ok": True, "non_joint_ok": True, "param_audit_ok": True,
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
        if arm == CONTROL and m["init_state_sha256"] != prev_manifest["init_state_sha256"]:
            checks["init_ok"] = False
        if arm != CONTROL and m["init_non_joint_sha256"] != prev_manifest["init_state_sha256"]:
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
    base_nb = non_joint_state(init_states[CONTROL])
    nb_joint = non_joint_state(init_states["JOINT_RESIDUAL"])
    if sorted(nb_joint) != sorted(base_nb) or not all(torch.equal(nb_joint[k], base_nb[k]) for k in nb_joint):
        checks["non_joint_ok"] = False
    jr0 = joint_residual_state(init_states["JOINT_RESIDUAL"])
    if state_hash(jr0) != manifests["JOINT_RESIDUAL"]["init_joint_residual_sha256"]:
        checks["init_ok"] = False
    if not all(checks.values()):
        raise RuntimeError(f"roster verification failed: { {k: v for k, v in checks.items() if not v} }")

    # (2) scoring of the two soups (fit + dev, full per-row exports)
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

    # (3) paired deltas (dev + fit) for the single contrast + the anchor
    deltas: dict[str, Any] = {}
    cand, ctrl = CANDIDATES[0], CONTROL
    for metric, table_key in (("y", "y_raw"), ("g", "g_raw")):
        deltas[f"JOINT-CTRL_C_{metric}_dev"] = float(
            results[cand]["mae_dev"][table_key] - results[ctrl]["mae_dev"][table_key])
        deltas[f"JOINT-CTRL_C_{metric}_fit"] = float(
            results[cand]["mae_fit"][table_key] - results[ctrl]["mae_fit"][table_key])
    dy_dev, dg_dev = deltas["JOINT-CTRL_C_y_dev"], deltas["JOINT-CTRL_C_g_dev"]
    deltas["JOINT-CTRL_C_y_dev_relative"] = float(dy_dev / results[ctrl]["mae_dev"]["y_raw"])
    deltas["JOINT-CTRL_C_g_dev_relative"] = float(dg_dev / results[ctrl]["mae_dev"]["g_raw"])
    deltas["gap_reduction_g_vs_ctrl_c"] = float(
        main_table[ctrl]["gap_dev_minus_fit_g"] - main_table[cand]["gap_dev_minus_fit_g"])
    deltas["gap_reduction_y_vs_ctrl_c"] = float(
        main_table[ctrl]["gap_dev_minus_fit_y_raw"] - main_table[cand]["gap_dev_minus_fit_y_raw"])
    deltas["fit_change_note"] = (
        "fit numbers are the FIXED SOUP's fit MAE (never a training optimum or a "
        "training lower bound); gap changes are reported separately and never "
        "substitute for the dev error"
    )
    deltas["definition"] = (
        "Delta = MAE(JOINT_RESIDUAL) - MAE(CTRL_C), same-seed paired, full dev "
        "(negative = improvement); the end-of-training online train loss is a "
        "different quantity (curve.json train_loss) and never ranks the arms"
    )
    historical_anchor = {
        **HISTORICAL_ANCHORS,
        "ctrl_c_dev_y_raw": results[CONTROL]["mae_dev"]["y_raw"],
        "ctrl_c_dev_g_raw": results[CONTROL]["mae_dev"]["g_raw"],
        "note": (
            "descriptive anchors ONLY: the previous round's C/BASE numbers come from "
            "a different same-recipe execution and carry GPU training nondeterminism; "
            "the recipe effect of THIS round is judged against the paired same-round "
            "CTRL_C"
        ),
    }

    # (4) group-paired bootstrap (2000 draws, seed 20261008, shared picks)
    errs_y = {f"{arm}_s{SEED}": np.abs(y_dev - results[arm]["dev"]["y_raw"]) for arm in ARMS}
    errs_g = {f"{arm}_s{SEED}": np.abs(g_dev - results[arm]["dev"]["h"]) for arm in ARMS}
    bootstrap = paired_bootstrap_contrast(errs_y, errs_g, smiles[dev_idx], cand=cand, ctrl=ctrl)

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

    # (6a) intervention A: the alpha-mean dictionary readout on BOTH soups
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

    # (6b) intervention B: the JOINT soup with joint_enabled=False (delta := 0)
    base_joint = {
        "y": y_dev, "g": g_dev,
        "y_raw": results["JOINT_RESIDUAL"]["dev"]["y_raw"],
        "h": results["JOINT_RESIDUAL"]["dev"]["h"],
        "q_raw": q_dev, "b_y": results["JOINT_RESIDUAL"]["dev"]["b_y"],
    }
    toggle_row = joint_toggle_intervention(
        SEED, base_joint, ctx=ctx, out_dir=out_dir, device=device, marker=marker, log=log,
    )
    toggle_row["response_beyond_noise"] = bool(toggle_row["abs_dpred"]["p95"] > marker)
    noop_row = ctrl_c_switch_noop_check(SEED, ctx=ctx, out_dir=out_dir, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # (7) the frozen single-seed exploratory reading
    point_deltas = {
        "JOINT-CTRL_C_y": deltas["JOINT-CTRL_C_y_dev"],
        "JOINT-CTRL_C_g": deltas["JOINT-CTRL_C_g_dev"],
    }
    reading = exploratory_reading(
        point_deltas, bootstrap,
        {arm: alpha_rows[arm] for arm in ARMS},
        toggle_row,
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
            "CTRL_C_total": int(arm_expected_audit(CONTROL)["total_parameters"]),
            "JOINT_RESIDUAL_total": int(arm_expected_audit("JOINT_RESIDUAL")["total_parameters"]),
            "joint_residual_parameters": JOINT_PARAMETERS,
            "relative_increase": float(JOINT_PARAMETERS / CTRL_C_TOTAL_PARAMETERS),
            "note": "cost reads the MEASURED seconds/memory; the +3.78% parameter delta never implies a time delta",
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
        "joint_toggle_intervention": toggle_row,
        "ctrl_c_switch_noop": noop_row,
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
            "single-seed performance-lead screen, not significance or repeatability"
        ),
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "terminal_eval.json", payload)
    log(f"[terminal-eval] reading={reading['branch']} JOINT-CTRL_C y {point_deltas['JOINT-CTRL_C_y']:+.5f} "
        f"g {point_deltas['JOINT-CTRL_C_g']:+.5f} in {payload['seconds']:.0f}s")
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
