"""Junction sufficiency / structural information audit models (v1).

Protocol: ``configs/bjoc_junction_sufficiency_audit_v1.yaml``.
Track: ``tracks/structure-contribution``.

This module defines **only** a functional forward-mode switch on top of the
frozen BJOC-Null model.  No parameter, module, loss or training primitive is
added:

* ``A2`` — the historical BJOC-Null grouping (``W_e = n_e * mu_G``,
  ``V_e = n_e * nu_G``, ``q_e = log1p(n_e)``); delegates literally to the
  parent implementation, so a forward pass is bit-identical to
  :class:`structural_relation_composer_v1.BJOC` in null mode;
* ``A1`` — graph Junction moments broadcast uniformly to every bond
  (``W_e = mu_G``, ``V_e = nu_G``, ``q_e = 0``);
* ``A0`` — no learned Junction content reaches the composer
  (``W_e = 0``, ``V_e = 0``, ``q_e = 0``); the Junction encoder is still
  executed (identical forward shape / model spec) but its output is discarded,
  so it receives no supervised gradient.

The three modes share the exact same allocated parameter tensors.  They do
**not** have the same effective trained capacity: ``A0``'s Junction encoder and
the composer input columns for ``W``/``V``/``q`` receive zero (or missing)
gradients.  That distinction is measured by :func:`gradient_audit` and must be
reported; the round is a functional ablation, not a strict
active-capacity-matched causal experiment.

Also contains the supervised loop copy used for A0/A1.  The loop body is a
line-for-line copy of :func:`bjoc_arch_breakthrough_v1.train_supervised_arch`
(itself the frozen ``bjoc_perfmech_long_v1.train_supervised`` body); the
equivalence test asserts an ``A2`` smoke run reproduces
``bjoc_perfmech_long_v1.train_supervised`` within the machine noise floor.
"""

from __future__ import annotations

import hashlib
import math
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-junction-sufficiency-audit-v1"

MODE_A0 = "A0"
MODE_A1 = "A1"
MODE_A2 = "A2"
MODES = (MODE_A0, MODE_A1, MODE_A2)
MODE_NAMES = {
    MODE_A0: "NO_LEARNED_J_CONTENT",
    MODE_A1: "GLOBAL_J_MOMENTS",
    MODE_A2: "P3_NULL_REFERENCE",
}
ARM_MODES = {"A0": MODE_A0, "A1": MODE_A1, "A2": MODE_A2}

#: composer input layout (frozen): [b (96), W (96), V (96), q (1), raw bond (7)]
_COMPOSE_B = 96
_COMPOSE_W_OFF = 96
_COMPOSE_V_OFF = 192
_COMPOSE_Q_IDX = 288
_COMPOSE_RAW_OFF = 289


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class BJOCSufficiency(src.BJOC):
    """BJOC-Null with a functional switch over the Junction composer inputs."""

    def __init__(
        self,
        config: src.BJOCConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        mode: str = MODE_A2,
    ) -> None:
        super().__init__(config, global_mean, global_scale, organization_mode=src.NULL)
        if mode not in MODES:
            raise ValueError(f"unknown sufficiency mode {mode!r}")
        self.mode = mode

    # -- grouping switch -----------------------------------------------------
    def group_to_bonds_suff(
        self,
        w: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (W_b, V_b, n_b) for one of the three modes."""
        n_bonds = int(batch["bond_feat"].shape[0])
        if mode == MODE_A2:
            return src.BJOC.group_to_bonds(self, w, batch, src.NULL)
        n_graphs = int(batch["n_graphs"])
        junc_batch = batch["junc_batch"]
        w_sum = torch.zeros((n_graphs, w.shape[1]), device=w.device, dtype=w.dtype)
        w2_sum = torch.zeros_like(w_sum)
        w_sum.index_add_(0, junc_batch, w)
        w2_sum.index_add_(0, junc_batch, w * w)
        counts = torch.bincount(junc_batch, minlength=n_graphs).to(w.dtype).clamp(min=1.0).unsqueeze(1)
        if mode == MODE_A1:
            mean_w = w_sum / counts
            mean_w2 = w2_sum / counts
            bond_batch = batch["bond_batch"]
            W = mean_w[bond_batch]
            V = mean_w2[bond_batch]
            n_b = torch.zeros((n_bonds,), device=w.device, dtype=w.dtype)
            return W, V, n_b
        if mode == MODE_A0:
            W = torch.zeros((n_bonds, w.shape[1]), device=w.device, dtype=w.dtype)
            V = torch.zeros_like(W)
            n_b = torch.zeros((n_bonds,), device=w.device, dtype=w.dtype)
            return W, V, n_b
        raise ValueError(f"unknown sufficiency mode {mode!r}")

    # -- forward -------------------------------------------------------------
    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        # An explicit organization_mode always wins and uses the historical
        # parent path (needed by the ownership-intervention diagnostics).
        if organization_mode is not None:
            return src.BJOC.forward_features(self, batch, organization_mode)
        grp = self.mode if mode is None else mode
        if grp not in MODES:
            raise ValueError(f"unknown sufficiency mode {grp!r}")
        if grp == MODE_A2:
            return src.BJOC.forward_features(self, batch)
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)
        W, V, n_b = self.group_to_bonds_suff(w, batch, grp)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]],
            dim=1,
        )
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        return {
            "a": a,
            "b": b,
            "w": w,
            "W": W,
            "V": V,
            "n_b": n_b,
            "c": c,
            "z": z,
            "reader_in": reader_in,
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        return_features: bool = False,
        mode: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = self.forward_features(batch, organization_mode, mode=mode)
        pred = self.reader(feats["reader_in"]).view(-1)
        if return_features:
            return pred, feats
        return pred


def build_sufficiency_from_init(
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str,
    mode: str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[src.BJOC, BJOCSufficiency, dict[str, Any]]:
    """Build (scratch base holder, sufficiency model) from one shared init.

    The sufficiency model receives the *exact* base state dict, so the
    allocated parameter tensors and the scratch init are identical to the
    historical P3-D-s0 construction for the seed; this is asserted by hash.
    """
    if mode not in MODES:
        raise SystemExit(f"unknown sufficiency mode {mode!r}")
    base = bp.build_bjoc_null(seed, norm, device)
    scratch_full = src.state_dict_hash(base)
    scratch_group = mp.group_state_hash(base)
    scratch_encoder = bp.encoder_group_hash(base)
    model = BJOCSufficiency(
        src.BJOCConfig(),
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        mode=mode,
    )
    model.load_state_dict({k: v.detach().clone().to(device) for k, v in base.state_dict().items()})
    model.to(device)
    if src.state_dict_hash(model) != scratch_full:
        raise SystemExit("sufficiency init does not match the shared scratch init")
    if ssl_state is not None:
        mp.load_encoder_state(model, ssl_state, device)
    encoder_used = bp.encoder_group_hash(model)
    audit = src.parameter_audit(src.BJOCConfig(), model)
    if not audit["parameter_exact"] or audit["actual_parameters"] != bp.EXPECTED_PARAMETERS:
        raise SystemExit(f"sufficiency parameter audit failed: {audit}")
    info = {
        "mode": mode,
        "mode_name": MODE_NAMES[mode],
        "base_scratch_init_full_sha256": scratch_full,
        "base_scratch_init_group_state_sha256": scratch_group,
        "base_scratch_init_encoder_sha256": scratch_encoder,
        "encoder_used_sha256": encoder_used,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "new_parameters": 0,
    }
    return base, model, info


# ---------------------------------------------------------------------------
# gradient / capacity audit
# ---------------------------------------------------------------------------


def _grad_norm(p: torch.Tensor) -> float | None:
    return None if p.grad is None else float(p.grad.detach().norm())


def _grad_max_abs(p: torch.Tensor) -> float | None:
    return None if p.grad is None else float(p.grad.detach().abs().max())


def gradient_audit(
    model: BJOCSufficiency,
    *,
    objs: Sequence[sro.MolObjects],
    y: np.ndarray,
    device: torch.device | str,
    mode: str | None = None,
    n_molecules: int = 128,
) -> dict[str, Any]:
    """One forward/backward pass; per-module gradients and active capacity."""
    dev = torch.device(device)
    chunk = list(objs[: int(n_molecules)])
    y_t = torch.as_tensor(np.asarray(y[: len(chunk)]), dtype=torch.float32, device=dev)
    model.train()
    batch = bp._mv(sro.batch_objects(chunk), dev)
    model.zero_grad(set_to_none=True)
    pred = model(batch, mode=mode)
    loss = F.l1_loss(pred.view(-1), y_t.view(-1))
    loss.backward()
    modules = {
        "atom_encoder": model.atom_encoder,
        "bond_encoder": model.bond_encoder,
        "junction_encoder": model.junction_encoder,
        "group_composer": model.group_composer,
        "reader": model.reader,
    }
    per_module: dict[str, Any] = {}
    active_total = 0
    allocated_total = 0
    active_entries = 0
    for name, module in modules.items():
        params = list(module.parameters())
        allocated = int(sum(p.numel() for p in params))
        active = int(
            sum(p.numel() for p in params if p.grad is not None and float(p.grad.detach().abs().max()) > 0.0)
        )
        entries = int(sum(int((p.grad.detach() != 0).sum()) for p in params if p.grad is not None))
        norms = [_grad_norm(p) for p in params]
        maxabs = [_grad_max_abs(p) for p in params]
        per_module[name] = {
            "allocated_parameters": allocated,
            "parameters_with_nonzero_grad": active,
            "active_entries_nonzero_grad": entries,
            "grad_norm_total": None if all(n is None for n in norms) else float(sum(n for n in norms if n is not None)),
            "grad_max_abs": None if all(m is None for m in maxabs) else float(max(m for m in maxabs if m is not None)),
            "has_grad": any(g is not None for g in norms),
        }
        active_total += active
        active_entries += entries
        allocated_total += allocated
    # composer first-layer column blocks: [b | W | V | q | raw]
    w0 = model.group_composer.lins[0].weight
    if w0.grad is None:
        col_blocks = {k: None for k in ("b", "W", "V", "q", "raw")}
    else:
        g = w0.grad.detach()
        col_blocks = {
            "b": float(g[:, :_COMPOSE_W_OFF].abs().max()),
            "W": float(g[:, _COMPOSE_W_OFF:_COMPOSE_V_OFF].abs().max()),
            "V": float(g[:, _COMPOSE_V_OFF:_COMPOSE_Q_IDX].abs().max()),
            "q": float(g[:, _COMPOSE_Q_IDX:_COMPOSE_RAW_OFF].abs().max()),
            "raw": float(g[:, _COMPOSE_RAW_OFF:].abs().max()),
        }
    return {
        "mode": model.mode if mode is None else mode,
        "n_molecules": len(chunk),
        "loss_l1": float(loss.detach()),
        "modules": per_module,
        "composer_first_layer_column_grad_max_abs": col_blocks,
        "allocated_parameters": allocated_total,
        "parameters_with_nonzero_grad": active_total,
        "active_entries_nonzero_grad": active_entries,
        "effective_active_entry_fraction": float(active_entries) / float(max(allocated_total, 1)),
        "note": (
            "active_entries counts scalar parameter entries whose gradient is present and non-zero in this "
            "one batch; it is an effective-capacity indicator, not a capacity-matching guarantee."
        ),
    }


@torch.no_grad()
def activation_audit(
    model: BJOCSufficiency,
    *,
    objs: Sequence[sro.MolObjects],
    device: torch.device | str,
    mode: str | None = None,
    n_molecules: int = 256,
) -> dict[str, Any]:
    """Dead-channel check on the fixed audit subset (post-SiLU activations)."""
    dev = torch.device(device)
    chunk = list(objs[: int(n_molecules)])
    batch = bp._mv(sro.batch_objects(chunk), dev)
    model.eval()
    feats = model.forward_features(batch, mode=mode)

    def hidden_stats(module: src._MLP, x: torch.Tensor) -> dict[str, Any]:
        h = F.silu(module.lins[0](x))
        per_unit = h.abs().mean(0)
        return {
            "units": int(h.shape[1]),
            "dead_units_mean_abs_lt_1e-8": int((per_unit < 1e-8).sum().item()),
            "mean_abs_unit_activation_min": float(per_unit.min()),
            "mean_abs_unit_activation_max": float(per_unit.max()),
        }

    a, b, w = feats["a"], feats["b"], feats["w"]
    idx = batch["bond_index"]
    au, av = a[idx[:, 0]], a[idx[:, 1]]
    bond_in = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
    jb = batch["junc_bond"]
    ji = batch["junc_index"]
    ai, aj, ak = a[ji[:, 0]], a[ji[:, 1]], a[ji[:, 2]]
    b1, b2 = b[jb[:, 0]], b[jb[:, 1]]
    junc_in = torch.cat(
        [ai + ak, (ai - ak).abs(), ai * ak, aj, b1 + b2, (b1 - b2).abs(), batch["junc_feat"]], dim=1
    )
    compose_in = torch.cat(
        [b, feats["W"], feats["V"], torch.log1p(feats["n_b"]).unsqueeze(1), batch["bond_feat"]], dim=1
    )
    return {
        "mode": model.mode if mode is None else mode,
        "n_molecules": len(chunk),
        "atom_encoder_hidden": hidden_stats(model.atom_encoder, batch["atom_feat"]),
        "bond_encoder_hidden": hidden_stats(model.bond_encoder, bond_in),
        "junction_encoder_hidden": hidden_stats(model.junction_encoder, junc_in),
        "composer_hidden": hidden_stats(model.group_composer, compose_in),
    }


# ---------------------------------------------------------------------------
# supervised training loop (frozen bp body, model construction parameterised)
# ---------------------------------------------------------------------------


def train_supervised_suff(
    *,
    arm: str,
    seed: int,
    recipe_key: str,
    init: str,
    train_objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    device: torch.device,
    mode: str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    smoke_epochs: int | None = None,
    log_fn: Callable[[str], None] = log,
) -> dict[str, Any]:
    """One supervised trajectory; loop body identical to ``bp.train_supervised``."""
    if recipe_key not in bp.RECIPES:
        raise SystemExit(f"unknown recipe {recipe_key!r}")
    recipe = bp.RECIPES[recipe_key]
    if init not in ("scratch", "ssl"):
        raise SystemExit(f"unknown init {init!r}")
    if init == "ssl" and ssl_state is None:
        raise SystemExit("init=ssl requires a pretrained encoder state")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")

    _base, model, build_info = build_sufficiency_from_init(
        seed, norm, device, mode, ssl_state if init == "ssl" else None
    )
    if init == "ssl" and build_info["encoder_used_sha256"] == build_info["base_scratch_init_encoder_sha256"]:
        raise SystemExit("init=ssl did not change the encoder state")
    init_full_sha = build_info["base_scratch_init_full_sha256"]
    init_group_sha = build_info["base_scratch_init_group_state_sha256"]
    init_encoder_sha = build_info["base_scratch_init_encoder_sha256"]
    encoder_used_sha = build_info["encoder_used_sha256"]
    audit = src.parameter_audit(src.BJOCConfig(), model)
    if not audit["parameter_exact"] or audit["actual_parameters"] != bp.EXPECTED_PARAMETERS:
        raise SystemExit(f"parameter audit failed: {audit}")

    epochs = int(recipe["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(rows))
    steps_per_epoch = math.ceil(n / bp.BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = None
    if recipe["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"])
        )
    elif recipe["schedule"] != "constant":
        raise SystemExit(f"unknown schedule {recipe['schedule']!r}")

    gen = torch.Generator().manual_seed(int(seed) + bp.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=bp.SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    total_mol = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, bp.BATCH_SIZE, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = bp._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), bp.GRAD_CLIP))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > bp.GRAD_CLIP:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                expected_lr = bp.schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
                lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        train_mae = total / max(n_mol, 1)
        total_mol += n_mol
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")
    wall = time.perf_counter() - started

    members, soup = bp._last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = bp.predict_model(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "mode": mode,
        "mode_name": MODE_NAMES[mode],
        "seed": int(seed),
        "recipe_key": recipe_key,
        "recipe": dict(recipe),
        "init": init,
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {bp.SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_state": soup_state,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "schedule": {
            "type": recipe["schedule"],
            "per": "step",
            "warmup": "none",
            "lr0": float(recipe["lr"]),
            "lr_min": None if recipe["lr_min"] is None else float(recipe["lr_min"]),
            "t_max_updates": total_updates,
            "lr_first_step": curve[0]["lr_first"],
            "lr_last_step": curve[-1]["lr_last"],
            "cosine_formula_max_abs_deviation": lr_deviation_max,
        },
        "init_full_sha256": init_full_sha,
        "init_group_state_sha256": init_group_sha,
        "init_encoder_sha256": init_encoder_sha,
        "encoder_used_sha256": encoder_used_sha,
        "smoke": bool(smoke_epochs is not None),
        "full_recipe_epochs": int(recipe["epochs"]),
        "build_info": build_info,
        "parameter_audit": audit,
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "full_hash_independent_recomputation": bp.batch_order_hash(seed, n, bp.BATCH_SIZE, epochs),
            "full_matches": order_hash.hexdigest() == bp.batch_order_hash(seed, n, bp.BATCH_SIZE, epochs),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": grad_norm_max,
            "lr_schedule_deviation_max": lr_deviation_max,
        },
        "lr_curve": curve,
        "wall_seconds": wall,
        "molecules_seen": int(total_mol),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_used": "train Fit rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# synthetic / object-level ownership diagnostics (shared with the audit driver)
# ---------------------------------------------------------------------------


def object_batch_with_incidence(
    bond_tokens: Sequence[int],
    junc_tokens: Sequence[int],
    incidence: Sequence[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Build a BJOC batch for one object-level incidence system."""
    inst = src.ObjectInstance(list(bond_tokens), list(junc_tokens), list(incidence), 0.0, 0.0)
    inst.target_incidence, inst.target_marginal = src.object_targets(
        inst.bond_tokens, inst.junc_tokens, inst.incidence
    )
    return src.batch_object_instances([inst])


def marginal_coarsening(
    incidence: Sequence[tuple[int, int]],
    n_bonds: int,
    n_junc: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Random reassignment of junction rows over the same per-bond counts.

    The junction multiset and the per-bond junction count are preserved by
    construction: it is a random matching of the same endpoint-count sequence.
    The junction identities are then assigned to the slots in random order.
    """
    slots: list[int] = []
    for j, (a, b) in enumerate(incidence):
        slots.extend([a, b])
    rng.shuffle(slots)
    return [(int(slots[2 * j]), int(slots[2 * j + 1])) for j in range(n_junc)]
