"""Repaired warm-adaptation capacity screen on the closed ZINC dictionary line.

Round ``e2e_dictenv_capacity_localization_v2`` (Workstream Z, ZINC).
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_capacity_localization_v2_preregistration.md``.

The frozen capacity base is **CAP-BASE = CSSD-q1** (C6 clean mask + common
structural coordinate ``c1`` + q1-orthogonal residual sparse dictionary +
paired node/edge structure-semantic binding + full relation + first/second
moment readout; seed-0 Top-5 soup valid MAE ``0.130028``).

v1 (``e2e_dictenv_capacity_localization_v1``) ran the same three candidates
under a fresh ``Adam(lr = 1e-3)`` full-parameter warm restart and returned
``NO_CLEAR_CAPACITY_LOCALIZATION`` **without power**: the M0 continuation
control itself degraded from ``0.130028`` (step 0) to ``0.135621`` (best) and
``0.156383`` (epoch 40), so the screen never ran in a stable baseline
neighbourhood.  This round repairs the adapter, not the candidates:

1. **Phase A — M0 calibration** (20 epochs, fresh ``Adam(lr = 1e-4)``) must
   pass a frozen stability gate; otherwise the round stops immediately
   (``WARM_ADAPTATION_PROTOCOL_UNSTABLE``) and no candidate is run.
2. **Phase B — repaired screen** (40 epochs): the v1 F/R/G architectures are
   reused *unchanged* but their residual projections are re-scaled so that the
   residual augmentation is near-zero at initialisation
   (``mean |prediction shift| <= 0.002``, preferred ``<= 0.001``) while every
   new parameter still receives finite non-zero gradient.  The optimizer is
   split into ``base`` (lr ``1e-4``) and ``new capacity`` (lr ``1e-3``) groups.
3. **Phase C — at most one from-scratch 320-epoch winner run** (the frozen
   CSSD protocol, Adam ``1e-3``) if and only if a candidate passes the frozen
   S1-S4 gate against both the matched M0 control *and* the absolute anchor
   ``M_start = 0.130028``.

CPU only; the official ZINC test split is never loaded
(``official_test_loaded = false`` everywhere).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_capacity_localization_v1 as cl
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

PROTOCOL_VERSION = "e2e_dictenv_capacity_localization_v2"

#: frozen capacity base and historical references (identical to v1).
CAP_BASE = cl.CAP_BASE
CAP_BASE_SOUP_MAE = cl.CAP_BASE_SOUP_MAE
FINAL_CLEAN_SPARSE_SOUP_MAE = cl.FINAL_CLEAN_SPARSE_SOUP_MAE

KINDS: tuple[str, ...] = cl.KINDS

# ---------------------------------------------------------------------------
# warm-adaptation protocol (v2 sections 3, 11, 17)
# ---------------------------------------------------------------------------

#: M0 calibration (Phase A): fresh Adam, low rate, exactly 20 epochs.
CALIBRATION_EPOCHS = 20
CALIBRATION_LR = 1.0e-4
CALIB_SOUP_WINDOW: tuple[int, int] = (1, 20)
CALIB_SOUP_K = 5
CALIB_LAST_K = 5
CALIB_LATE_WINDOW: tuple[int, int] = (16, 20)

#: differential warm-adaptation screen (Phase B): base vs new-capacity rates.
BASE_LR = 1.0e-4
NEW_LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0

SCREEN_EPOCHS = 40
SCREEN_SOUP_WINDOW: tuple[int, int] = (21, 40)
SCREEN_SOUP_K = 5
SCREEN_LAST10 = 10

#: Full from-scratch winner run (Phase C), frozen CSSD protocol.
FULL_EPOCHS = 320
FULL_SEED = 0
FULL_LR = 1.0e-3

# ---------------------------------------------------------------------------
# Phase A calibration gate (v2 section 13)
# ---------------------------------------------------------------------------

CALIB_SOUP_MAX = 0.1320
CALIB_LATE_MEAN_MAX = 0.1350
CALIB_EPOCH20_MAX = 0.137
CALIB_LATE_SLOPE_MAX = 5.0e-4

# ---------------------------------------------------------------------------
# Phase B screen gate (v2 sections 30-34)
# ---------------------------------------------------------------------------

S1_DELTA_VS_M0 = -0.003
S2_ABS_MAX = 0.1270
S3_DELTA_LAST10_VS_M0 = -0.003
STRONG_SCREEN_MAX = 0.123
TIE_TOLERANCE = 0.002
TIE_ORDER: tuple[str, ...] = cl.TIE_ORDER  # F > R > G

# ---------------------------------------------------------------------------
# Phase C interpretation bands (v2 section 40)
# ---------------------------------------------------------------------------

FULL_DID_NOT_TRANSFER_ABOVE = 0.127
FULL_USEFUL_AT = 0.125
FULL_STRONG_AT = 0.120
FULL_MAJOR_AT = 0.110

# ---------------------------------------------------------------------------
# residual-augmentation initialisation contract (v2 sections 20-25)
# ---------------------------------------------------------------------------

#: hard bound and preferred bound for the step-0 mean |prediction shift|.
INIT_SHIFT_HARD_MAX = 0.002
INIT_SHIFT_PREFERRED_MAX = 0.001

#: residual projection scales.  v1 used ``FUSION_PROJ_INIT = 0.01`` (F) and
#: ``RELATION_RESIDUAL_INIT = 0.05`` (R), which produced step-0 mean shifts of
#: 0.0105 / 0.0195 — 5-10x the hard bound.  Only the *scale* of the residual
#: projection changes; every architecture constant, shape and parameter name
#: stays identical to v1.
FUSION_RESIDUAL_SCALE = 0.0008
RELATION_RESIDUAL_SCALE = 0.0016
#: candidate G keeps the v1 appended-reader-column scale (measured 0.00113).
READOUT_SUMMARY_SCALE = cl.READOUT_READER_INIT

BUDGET_PREFERRED = cl.BUDGET_PREFERRED
BUDGET_HARD_CEILING = cl.BUDGET_HARD_CEILING
BUDGET_RATIO_MAX = cl.BUDGET_RATIO_MAX

official_test_blocker = cl.official_test_blocker
cpu_only_guard = cl.cpu_only_guard
predictions_for = cl.predictions_for
evaluate_mae = cl.evaluate_mae
prediction_shift = cl.prediction_shift
new_module_gradient_norms = cl.new_module_gradient_norms
load_capacity_warm_state = cl.load_capacity_warm_state


# ---------------------------------------------------------------------------
# candidates: the v1 architectures with re-scaled residual projections
# ---------------------------------------------------------------------------


class ResidualFusionModel(cl.FusionCapacityModel):
    """Candidate F — v1 multi-rank fusion with a near-zero residual projection.

    Architecture, parameter names, shapes, head count, head width and the
    factor draws are exactly v1's.  Only ``F_NP`` / ``F_EP`` are multiplied by
    ``FUSION_RESIDUAL_SCALE / FUSION_PROJ_INIT``, so the residual multi-rank
    branch is near-zero at initialisation while the head factors keep finite
    non-zero gradients.
    """

    CAPACITY_KIND = "F"
    CAPACITY_NEW_PREFIXES = cl.FusionCapacityModel.CAPACITY_NEW_PREFIXES

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        heads: int = cl.FUSION_HEADS,
        head_dim: int = cl.FUSION_HEAD_DIM,
        init_seed: int = cl.FUSION_INIT_SEED,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
            heads=heads,
            head_dim=head_dim,
            init_seed=init_seed,
        )
        self.fusion_residual_scale = float(FUSION_RESIDUAL_SCALE)
        factor = float(FUSION_RESIDUAL_SCALE) / float(cl.FUSION_PROJ_INIT)
        with torch.no_grad():
            self.F_NP.weight.mul_(factor)
            self.F_EP.weight.mul_(factor)

    def zero_residual(self) -> None:
        with torch.no_grad():
            self.F_NP.weight.zero_()
            self.F_EP.weight.zero_()


class ResidualRelationModel(cl.RelationCapacityModel):
    """Candidate R — v1 pair-composition blocks with a near-zero residual.

    Same blocks, hidden width, FiLM design and zero ``gamma``/``beta``
    initialisation as v1; only the two ``fc2`` maps (weight and bias) are
    multiplied by ``RELATION_RESIDUAL_SCALE / RELATION_RESIDUAL_INIT``.
    """

    CAPACITY_KIND = "R"
    CAPACITY_NEW_PREFIXES = cl.RelationCapacityModel.CAPACITY_NEW_PREFIXES

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        blocks: int = cl.RELATION_BLOCKS,
        hidden: int = cl.RELATION_HIDDEN,
        init_seed: int = cl.RELATION_INIT_SEED,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
            blocks=blocks,
            hidden=hidden,
            init_seed=init_seed,
        )
        self.relation_residual_scale = float(RELATION_RESIDUAL_SCALE)
        factor = float(RELATION_RESIDUAL_SCALE) / float(cl.RELATION_RESIDUAL_INIT)
        with torch.no_grad():
            for block in self.pair_blocks:
                block.fc2.weight.mul_(factor)
                block.fc2.bias.mul_(factor)

    def zero_residual(self) -> None:
        with torch.no_grad():
            for block in self.pair_blocks:
                block.fc2.weight.zero_()
                block.fc2.bias.zero_()


class SummaryProjectionReader(nn.Module):
    """Reader with the appended summary columns as an explicit additive map.

    The CAP-BASE reader stack (``net``) is kept exactly as it is — same
    parameter names and shapes as CAP-BASE — and the appended summary columns
    enter through a separate bias-free ``summary_proj``:

    ``h = net[0](base) + summary_proj(summary)``, then the frozen remaining
    reader layers.

    This is functionally identical to v1's widened first reader layer (its
    appended columns are the same tensor) but keeps the base columns and the
    new columns separable, which is what the differential-LR optimizer groups
    and the residual-zero contract require.
    """

    def __init__(self, base_reader: Any, base_in_dim: int, summary_dim: int) -> None:
        super().__init__()
        self.base_in_dim = int(base_in_dim)
        self.summary_dim = int(summary_dim)
        self.net = base_reader.net
        hidden = int(self.net[0].out_features)
        self.summary_proj = nn.Linear(2 * self.summary_dim, hidden, bias=False)

    def forward(self, unified: torch.Tensor) -> torch.Tensor:
        base = unified[:, : self.base_in_dim]
        summary = unified[:, self.base_in_dim :]
        hidden = self.net[0](base) + self.summary_proj(summary)
        for layer in self.net[1:]:
            hidden = layer(hidden)
        return hidden.squeeze(-1)


class SummaryReadoutModel(cl.ReadoutCapacityModel):
    """Candidate G — v1 gated DeepSets summaries with a split reader.

    Same summary width, same gate/value parameterisations and the same frozen
    residual (now ``reader.summary_proj``) as v1's appended reader columns;
    ``capacity_graph_repr`` is inherited from v1's ``ReadoutCapacityModel``, so
    the concatenation contract and the per-summary disable semantics are
    unchanged.  Only the reader's first layer is split into the frozen base
    columns plus the appended summary projection.
    """

    CAPACITY_KIND = "G"
    CAPACITY_NEW_PREFIXES = (
        "summary_gate_env",
        "summary_value_env",
        "summary_gate_pair",
        "summary_value_pair",
        "reader.summary_proj",
    )

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        dim: int = cl.READOUT_SUMMARY_DIM,
        init_seed: int = cl.READOUT_INIT_SEED,
    ) -> None:
        # deliberately bypass ``cl.ReadoutCapacityModel.__init__`` (which widens
        # the reader); the CAP-BASE reader is built unchanged and the appended
        # summary columns become an explicit additive projection.
        cl.CapacityModel.__init__(
            self,
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        self.summary_dim = int(dim)
        self.disabled_summaries: set[str] = set()
        generator = torch.Generator().manual_seed(int(init_seed))
        with cl.preserved_global_rng():
            self.summary_gate_env = nn.Linear(int(p2.ENV_DIM), self.summary_dim)
            self.summary_value_env = nn.Linear(int(p2.ENV_DIM), self.summary_dim)
            self.summary_gate_pair = nn.Linear(int(p2.PAIR_HIDDEN), self.summary_dim)
            self.summary_value_pair = nn.Linear(int(p2.PAIR_HIDDEN), self.summary_dim)
            self.reader = SummaryProjectionReader(
                self.reader, int(audit.READER_IN_DIM), self.summary_dim
            )
        with torch.no_grad():
            for module in (
                self.summary_gate_env,
                self.summary_value_env,
                self.summary_gate_pair,
                self.summary_value_pair,
            ):
                cl._linear_default_generator(module, generator)
            cl._small_uniform_generator(
                self.reader.summary_proj.weight, generator, float(READOUT_SUMMARY_SCALE)
            )

    def zero_residual(self) -> None:
        with torch.no_grad():
            self.reader.summary_proj.weight.zero_()


MODEL_BY_KIND: dict[str, type[cl.CapacityModel]] = {
    "M0": cl.CapacityModel,
    "F": ResidualFusionModel,
    "R": ResidualRelationModel,
    "G": SummaryReadoutModel,
}


def build_v2_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    kind: str,
    *,
    spec: cm.CleanMechSpec | None = None,
) -> cl.CapacityModel:
    """Build a v2 arm with the exact CAP-BASE initialization stream."""
    kind = str(kind).upper()
    if kind not in MODEL_BY_KIND:
        raise ValueError(f"unknown capacity kind {kind!r}")
    spec = cssd.CSSD_SPEC if spec is None else spec
    torch.manual_seed(int(seed))
    model = MODEL_BY_KIND[kind](
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        node_binding=spec.node_binding,
        edge_binding=spec.edge_binding,
        coding=spec.coding,
    )
    if model.CAPACITY_KIND != kind:
        raise RuntimeError(f"model kind mismatch: built {model.CAPACITY_KIND!r} for {kind!r}")
    return model


# ---------------------------------------------------------------------------
# optimizer parameter partition (v2 sections 17-19)
# ---------------------------------------------------------------------------


def _is_capacity_parameter(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def parameter_partition(model: cl.CapacityModel) -> dict[str, list[str]]:
    """Split every trainable parameter into exactly one of ``base`` / ``new``.

    ``base`` = every trainable parameter that already exists in the CAP-BASE
    checkpoint; ``new`` = the added capacity module (for G: the gated summary
    modules plus the strictly necessary appended reader projection).  The
    partition is exact: duplicates, missing parameters and unknown names all
    raise.
    """
    prefixes = tuple(model.CAPACITY_NEW_PREFIXES)
    base: list[str] = []
    new: list[str] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"parameter {name!r} appears twice in named_parameters()")
        seen.add(id(parameter))
        (new if _is_capacity_parameter(name, prefixes) else base).append(name)
    known = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if len(base) + len(new) != len(known):
        raise RuntimeError("parameter partition does not cover every trainable parameter")
    return {"base": base, "new": new, "all": base + new}


def build_warm_optimizer(
    model: cl.CapacityModel,
    *,
    base_lr: float = BASE_LR,
    new_lr: float = NEW_LR,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple[torch.optim.Adam, dict[str, Any]]:
    """Fresh Adam with one group for ``base`` parameters and one for ``new``.

    The groups partition the parameter list exactly (asserted); M0 has an empty
    ``new`` group.  No optimizer state is inherited from any earlier run.
    """
    partition = parameter_partition(model)
    parameters = dict(model.named_parameters())
    groups = []
    if partition["base"]:
        groups.append({"params": [parameters[name] for name in partition["base"]], "lr": float(base_lr)})
    if partition["new"]:
        groups.append({"params": [parameters[name] for name in partition["new"]], "lr": float(new_lr)})
    if not groups:
        raise RuntimeError("no trainable parameters")
    optimizer = torch.optim.Adam(groups, lr=float(base_lr), weight_decay=float(weight_decay))
    report = {
        "base_lr": float(base_lr),
        "new_lr": float(new_lr),
        "weight_decay": float(weight_decay),
        "base_parameters": list(partition["base"]),
        "new_parameters": list(partition["new"]),
        "base_parameter_count": int(sum(parameters[name].numel() for name in partition["base"])),
        "new_parameter_count": int(sum(parameters[name].numel() for name in partition["new"])),
        "group_count": len(groups),
    }
    return optimizer, report


# ---------------------------------------------------------------------------
# the warm-adaptation loop (frozen CAP-BASE training semantics)
# ---------------------------------------------------------------------------


def _flat_norm(tensors: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        total += float(tensor.detach().pow(2).sum())
    return float(math.sqrt(max(total, 0.0)))


def _delta_norm(parameters: Sequence[torch.Tensor], before: Sequence[torch.Tensor]) -> float:
    total = 0.0
    for parameter, reference in zip(parameters, before):
        total += float((parameter.detach() - reference).pow(2).sum())
    return float(math.sqrt(max(total, 0.0)))


def soup_state_from(
    states: Mapping[int, Mapping[str, torch.Tensor]], members: Sequence[int]
) -> dict[str, torch.Tensor]:
    members = list(members)
    return {
        key: torch.stack([states[epoch][key].float() for epoch in members], dim=0).mean(0)
        for key in states[members[0]]
    }


def train_warm(
    *,
    tag: str,
    model: cl.CapacityModel,
    dictionary: np.ndarray,
    subspace: cssd.CommonSubspace,
    epochs: int,
    threads: int,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    seed: int = 0,
    base_lr: float = BASE_LR,
    new_lr: float = NEW_LR,
    soup_window: tuple[int, int] = SCREEN_SOUP_WINDOW,
    soup_k: int = SCREEN_SOUP_K,
    last_k: int = SCREEN_LAST10,
    log: bool = True,
    return_soup_state: bool = False,
) -> dict[str, Any]:
    """Warm-adapt one arm with the frozen differential-LR protocol.

    Identical to ``cssd.train_cssd`` on the training path (fresh Adam, wd
    1e-5, batch 128, clip 5.0, same shuffle seeds, same objective) except that
    the parameters are split into a base group (``base_lr``) and a new-capacity
    group (``new_lr``), the retained weight soup is the Top-``soup_k`` of the
    window ``soup_window``, and every epoch logs base/new gradient and update
    norms.  The model must already be warm-loaded; the caller has seeded the
    global RNG exactly as ``train_cssd`` does.
    """
    import time

    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    device = audit.attach_cpu(int(threads))
    cpu_only_guard(device)
    model = model.to(device)
    optimizer, group_report = build_warm_optimizer(
        model, base_lr=float(base_lr), new_lr=float(new_lr)
    )
    parameters = dict(model.named_parameters())
    base_parameters = [parameters[name] for name in group_report["base_parameters"]]
    new_parameters = [parameters[name] for name in group_report["new_parameters"]]
    loader = p1.make_env_loader(
        train_data, int(p2run.BATCH_SIZE), True, int(seed) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    eval_loader = p1.make_env_loader(
        valid_data, int(p2run.BATCH_SIZE), False, int(seed) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    mask = cssd.CSSD_MASK
    lam = float(cm.H1_LAMBDA)
    low, high = (int(soup_window[0]), int(soup_window[1]))
    window_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum = rec_sum = rec_term_sum = total_loss_sum = 0.0
        grad_base_sum = grad_new_sum = upd_base_sum = upd_new_sum = 0.0
        n_mol = n_nodes = n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            grad_base_sum += _flat_norm([parameter.grad for parameter in base_parameters])
            grad_new_sum += _flat_norm([parameter.grad for parameter in new_parameters])
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
            before_base = [parameter.detach().clone() for parameter in base_parameters]
            before_new = [parameter.detach().clone() for parameter in new_parameters]
            optimizer.step()
            upd_base_sum += _delta_norm(base_parameters, before_base)
            upd_new_sum += _delta_norm(new_parameters, before_new)
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            total_loss_sum += float(loss.detach())
            rec_term_sum += float(rec.detach())
            n_batches += 1
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float(
                (((phi - phi_hat) ** 2).sum(dim=1) / ((phi**2).sum(dim=1) + float(v0.EPS))).sum()
            )
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = audit._evaluate_model(model, eval_loader, device, mask)
        steps = max(n_batches, 1)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "train_rec": train_rec,
                "train_rec_term": float(rec_term_sum / steps),
                "train_total_loss": float(total_loss_sum / steps),
                "valid_mae": float(valid["mae"]),
                "d_norm": float(model.D.detach().norm()),
                "grad_norm_base": float(grad_base_sum / steps),
                "grad_norm_new": float(grad_new_sum / steps),
                "grad_norm_total": float((grad_base_sum + grad_new_sum) / steps),
                "update_norm_base": float(upd_base_sum / steps),
                "update_norm_new": float(upd_new_sum / steps),
                "update_norm_total": float((upd_base_sum + upd_new_sum) / steps),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if low <= epoch <= high:
            window_states[int(epoch)] = {
                key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()
            }
        if log and (epoch == 1 or epoch % 5 == 0 or epoch == int(epochs)):
            print(
                f"[{tag}] epoch={epoch:03d} train={train_mae:.6f} valid={float(valid['mae']):.6f} "
                f"grad_base={curve[-1]['grad_norm_base']:.4f} grad_new={curve[-1]['grad_norm_new']:.6f}",
                flush=True,
            )
    wall = float(time.perf_counter() - started)
    if not window_states:
        raise RuntimeError(f"soup window {soup_window} produced no retained states")
    window_rows = [row for row in curve if low <= int(row["epoch"]) <= high]
    members = sorted(
        int(row["epoch"])
        for row in sorted(window_rows, key=lambda row: float(row["valid_mae"]))[: int(soup_k)]
    )
    soup_state = soup_state_from(window_states, members)
    soup_model = build_v2_model(dictionary, int(seed), subspace, model.CAPACITY_KIND)
    soup_model.load_state_dict(soup_state)
    soup_model = soup_model.to(device)
    soup_mae = evaluate_mae(soup_model, eval_loader, device, mask)
    last_rows = curve[-int(last_k) :]
    grad_new_values = [float(row["grad_norm_new"]) for row in curve]
    update_new_values = [float(row["update_norm_new"]) for row in curve]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "kind": model.CAPACITY_KIND,
        "device": "cpu",
        "threads": int(threads),
        "seed": int(seed),
        "epochs_budget": int(epochs),
        "epochs_run": int(len(curve)),
        "official_test_loaded": False,
        "optimizer_groups": group_report,
        "soup_window": [low, high],
        "soup_members": members,
        "soup_member_valid_mae": [float(curve[epoch - 1]["valid_mae"]) for epoch in members],
        "soup_valid_mae": float(soup_mae),
        "best_valid_mae": float(min(row["valid_mae"] for row in curve)),
        "best_epoch": int(min(curve, key=lambda row: float(row["valid_mae"]))["epoch"]),
        "last_mean_valid_mae": float(np.mean([row["valid_mae"] for row in last_rows])),
        "last_valid_mae": float(curve[-1]["valid_mae"]),
        "final_train_mae": float(curve[-1]["train_mae"]),
        "branch_grad_mean": float(np.mean(grad_new_values)) if new_parameters else 0.0,
        "branch_grad_max": float(np.max(grad_new_values)) if new_parameters else 0.0,
        "branch_grad_finite": bool(all(np.isfinite(value) for value in grad_new_values)),
        "branch_update_mean": float(np.mean(update_new_values)) if new_parameters else 0.0,
        "branch_update_max": float(np.max(update_new_values)) if new_parameters else 0.0,
        "branch_update_finite": bool(all(np.isfinite(value) for value in update_new_values)),
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve), 1)),
        "official_test_loaded_guard": False,
        "curve": curve,
    }
    if return_soup_state:
        payload["soup_state"] = soup_state
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# Phase A — calibration decision (v2 sections 12-15)
# ---------------------------------------------------------------------------


def late_window_slope(curve: Sequence[Mapping[str, Any]], window: tuple[int, int] = CALIB_LATE_WINDOW) -> float:
    rows = [row for row in curve if int(window[0]) <= int(row["epoch"]) <= int(window[1])]
    if len(rows) < 2:
        raise ValueError("late window needs at least two epochs")
    x = np.asarray([float(row["epoch"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["valid_mae"]) for row in rows], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def calibration_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen Phase-A calibration record (preregistration section 48)."""
    curve = result["curve"]
    late_rows = [
        row for row in curve if CALIB_LATE_WINDOW[0] <= int(row["epoch"]) <= CALIB_LATE_WINDOW[1]
    ]
    last5 = float(np.mean([float(row["valid_mae"]) for row in late_rows]))
    epoch20 = float(curve[-1]["valid_mae"])
    slope = late_window_slope(curve)
    start = float(CAP_BASE_SOUP_MAE)
    best = float(result["best_valid_mae"])
    soup = float(result["soup_valid_mae"])
    c1 = bool(soup <= CALIB_SOUP_MAX)
    c2 = bool(last5 <= CALIB_LATE_MEAN_MAX)
    c3 = bool(epoch20 <= CALIB_EPOCH20_MAX and slope <= CALIB_LATE_SLOPE_MAX)
    passed = bool(c1 and c2 and c3)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "start_mae": start,
        "calibration_best": best,
        "calibration_soup": soup,
        "calibration_last5": last5,
        "epoch20": epoch20,
        "delta_best": float(best - start),
        "delta_soup": float(soup - start),
        "delta_last5": float(last5 - start),
        "late_slope": slope,
        "late_window": list(CALIB_LATE_WINDOW),
        "soup_window": list(CALIB_SOUP_WINDOW),
        "soup_members": list(result["soup_members"]),
        "thresholds": {
            "soup_max": CALIB_SOUP_MAX,
            "late_mean_max": CALIB_LATE_MEAN_MAX,
            "epoch20_max": CALIB_EPOCH20_MAX,
            "late_slope_max": CALIB_LATE_SLOPE_MAX,
        },
        "C1_pass": c1,
        "C2_pass": c2,
        "C3_pass": c3,
        "overall_pass": passed,
        "verdict": "WARM_ADAPTATION_PROTOCOL_VALIDATED"
        if passed
        else "WARM_ADAPTATION_PROTOCOL_UNSTABLE",
    }


# ---------------------------------------------------------------------------
# Phase B — screen decision (v2 sections 29-34)
# ---------------------------------------------------------------------------


def screening_deltas(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Deltas vs the matched M0 control AND vs the frozen absolute anchor."""
    control = rows["M0"]
    deltas: dict[str, dict[str, float]] = {}
    for kind in ("F", "R", "G"):
        row = rows[kind]
        deltas[kind] = {
            "delta_soup_vs_M0": float(row["soup_valid_mae"]) - float(control["soup_valid_mae"]),
            "delta_last10_vs_M0": float(row["last_mean_valid_mae"])
            - float(control["last_mean_valid_mae"]),
            "delta_soup_vs_start": float(row["soup_valid_mae"]) - float(CAP_BASE_SOUP_MAE),
            "delta_last10_vs_start": float(row["last_mean_valid_mae"]) - float(CAP_BASE_SOUP_MAE),
        }
    return deltas


def capacity_gate(row: Mapping[str, Any], delta: Mapping[str, float]) -> dict[str, Any]:
    """Frozen S1-S4 screen gate (all four must hold)."""
    s1 = bool(float(delta["delta_soup_vs_M0"]) <= S1_DELTA_VS_M0)
    s2 = bool(float(row["soup_valid_mae"]) <= S2_ABS_MAX)
    s3 = bool(float(delta["delta_last10_vs_M0"]) <= S3_DELTA_LAST10_VS_M0)
    s4 = bool(
        np.isfinite(float(row["branch_grad_max"]))
        and np.isfinite(float(row["branch_update_max"]))
        and float(row["branch_grad_max"]) > 0.0
        and float(row["branch_update_max"]) > 0.0
        and bool(row["branch_grad_finite"])
        and bool(row["branch_update_finite"])
    )
    stable = bool(np.isfinite(float(row["soup_valid_mae"]))) and bool(
        np.isfinite(float(row["last_mean_valid_mae"]))
    )
    passed = bool(s1 and s2 and s3 and s4 and stable)
    verdict = "CAPACITY_SIGNAL" if passed else "NO_CAPACITY_SIGNAL"
    if passed and float(row["soup_valid_mae"]) <= STRONG_SCREEN_MAX:
        verdict = "STRONG_CAPACITY_SIGNAL"
    return {
        "S1_matched_control": s1,
        "S2_absolute": s2,
        "S3_late_window": s3,
        "S4_branch_usage": s4,
        "stable": stable,
        "passed": passed,
        "verdict": verdict,
        "thresholds": {
            "S1_delta_vs_M0": S1_DELTA_VS_M0,
            "S2_abs_max": S2_ABS_MAX,
            "S3_delta_last10_vs_M0": S3_DELTA_LAST10_VS_M0,
            "strong_screen_max": STRONG_SCREEN_MAX,
        },
    }


def select_winner(
    rows: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Frozen winner rule: S1-S4 survivors, then soup MAE, then F > R > G."""
    passing = [kind for kind in TIE_ORDER if bool(gate[kind]["passed"])]
    if not passing:
        return {
            "winner": None,
            "reason": "LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED",
            "passing": [],
            "frozen_tie_order": list(TIE_ORDER),
        }
    ranked = sorted(
        passing, key=lambda kind: (float(rows[kind]["soup_valid_mae"]), TIE_ORDER.index(kind))
    )
    best = ranked[0]
    tied = [
        kind
        for kind in ranked
        if abs(float(rows[kind]["soup_valid_mae"]) - float(rows[best]["soup_valid_mae"]))
        <= TIE_TOLERANCE
    ]
    if len(tied) > 1:
        best = min(tied, key=lambda kind: TIE_ORDER.index(kind))
        reason = "TIE_BREAK_F_R_G"
    else:
        reason = "SCREENING_SOUP_MAE"
    return {
        "winner": best,
        "reason": reason,
        "passing": passing,
        "ranked": ranked,
        "tied_within_tolerance": tied,
        "frozen_tie_order": list(TIE_ORDER),
    }


# ---------------------------------------------------------------------------
# Phase C — full-run interpretation (v2 section 40)
# ---------------------------------------------------------------------------


def full_interpretation(soup_mae: float) -> dict[str, Any]:
    if soup_mae <= FULL_MAJOR_AT:
        band = "MAJOR_CAPACITY_BOTTLENECK_IDENTIFIED"
    elif soup_mae <= FULL_STRONG_AT:
        band = "NEW_PERFORMANCE_BAND_SINGLE_SEED"
    elif soup_mae <= FULL_USEFUL_AT:
        band = "CAPACITY_DIRECTION_SUPPORTED_SINGLE_SEED"
    elif soup_mae <= FULL_DID_NOT_TRANSFER_ABOVE:
        band = "FULL_CAPACITY_GAIN_NOT_ESTABLISHED"
    else:
        band = "SHORT_SCREEN_SIGNAL_DID_NOT_TRANSFER"
    return {
        "band": band,
        "soup_valid_mae": float(soup_mae),
        "thresholds": {
            "did_not_transfer_above": FULL_DID_NOT_TRANSFER_ABOVE,
            "useful_at": FULL_USEFUL_AT,
            "strong_at": FULL_STRONG_AT,
            "major_at": FULL_MAJOR_AT,
        },
        "delta_vs_cap_base": float(soup_mae) - float(CAP_BASE_SOUP_MAE),
        "delta_vs_final_clean_sparse": float(soup_mae) - float(FINAL_CLEAN_SPARSE_SOUP_MAE),
    }


# ---------------------------------------------------------------------------
# frozen architecture fingerprint (preregistration section 9)
# ---------------------------------------------------------------------------


def architecture_fingerprint(kind: str, model: cl.CapacityModel) -> dict[str, Any]:
    """Name/shape fingerprint recorded in the preregistration snapshot."""
    kind = str(kind).upper()
    lines = [f"{name}:{tuple(shape)}" for name, shape in _named_shapes(model)]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    partition = parameter_partition(model)
    return {
        "kind": kind,
        "tensor_count": len(lines),
        "params": int(sum(parameter.numel() for parameter in model.parameters())),
        "added_params": int(
            sum(dict(model.named_parameters())[name].numel() for name in partition["new"])
        ),
        "shape_fingerprint_sha256": digest,
        "new_parameter_names": list(partition["new"]),
        "constants": {
            "fusion_heads": cl.FUSION_HEADS,
            "fusion_head_dim": cl.FUSION_HEAD_DIM,
            "fusion_residual_scale": FUSION_RESIDUAL_SCALE,
            "relation_blocks": cl.RELATION_BLOCKS,
            "relation_hidden": cl.RELATION_HIDDEN,
            "relation_residual_scale": RELATION_RESIDUAL_SCALE,
            "readout_summary_dim": cl.READOUT_SUMMARY_DIM,
            "readout_summary_scale": READOUT_SUMMARY_SCALE,
        },
    }


def _named_shapes(model: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(name, tuple(tensor.shape)) for name, tensor in model.state_dict().items()]


__all__ = [
    "PROTOCOL_VERSION",
    "CAP_BASE",
    "CAP_BASE_SOUP_MAE",
    "FINAL_CLEAN_SPARSE_SOUP_MAE",
    "KINDS",
    "CALIBRATION_EPOCHS",
    "CALIBRATION_LR",
    "CALIB_SOUP_WINDOW",
    "CALIB_SOUP_K",
    "CALIB_LAST_K",
    "CALIB_LATE_WINDOW",
    "BASE_LR",
    "NEW_LR",
    "WEIGHT_DECAY",
    "GRAD_CLIP",
    "SCREEN_EPOCHS",
    "SCREEN_SOUP_WINDOW",
    "SCREEN_SOUP_K",
    "SCREEN_LAST10",
    "FULL_EPOCHS",
    "FULL_SEED",
    "FULL_LR",
    "CALIB_SOUP_MAX",
    "CALIB_LATE_MEAN_MAX",
    "CALIB_EPOCH20_MAX",
    "CALIB_LATE_SLOPE_MAX",
    "S1_DELTA_VS_M0",
    "S2_ABS_MAX",
    "S3_DELTA_LAST10_VS_M0",
    "STRONG_SCREEN_MAX",
    "TIE_TOLERANCE",
    "TIE_ORDER",
    "FULL_DID_NOT_TRANSFER_ABOVE",
    "FULL_USEFUL_AT",
    "FULL_STRONG_AT",
    "FULL_MAJOR_AT",
    "INIT_SHIFT_HARD_MAX",
    "INIT_SHIFT_PREFERRED_MAX",
    "FUSION_RESIDUAL_SCALE",
    "RELATION_RESIDUAL_SCALE",
    "READOUT_SUMMARY_SCALE",
    "official_test_blocker",
    "cpu_only_guard",
    "ResidualFusionModel",
    "ResidualRelationModel",
    "SummaryProjectionReader",
    "SummaryReadoutModel",
    "MODEL_BY_KIND",
    "build_v2_model",
    "parameter_partition",
    "build_warm_optimizer",
    "soup_state_from",
    "train_warm",
    "late_window_slope",
    "calibration_summary",
    "screening_deltas",
    "capacity_gate",
    "select_winner",
    "full_interpretation",
    "architecture_fingerprint",
]
