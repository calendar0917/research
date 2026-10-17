"""P3-SAM v1 — standard two-step SAM on top of the frozen BJOC-P3 recipe.

Protocol ``cscl-p3-sam-v1`` (``results/p3_sam_v1/PROTOCOL.md``).

The model, the Fit-only masked SSL objective, the supervised samples, the Adam
optimiser, the cosine schedule, the raw-y L1 loss, the 360-epoch budget and the
last-5 parameter soup are **byte-for-byte the frozen P3 recipe**.  The only
change in the SAM arm is the optimiser outer step:

    w_t      = current parameters
    g_t      = grad L(w_t)                         (first backward)
    eps_t    = rho * g_t / (||g_t||_2 + 1e-12)     (global L2 over all params)
    w_t'     = w_t + eps_t                         (perturbation)
    g_t'     = grad L(w_t')                        (second backward, same batch)
    w_t      = w_t                                 (exact restore)
    clip(g_t', 5.0)                                (only after 2nd backward)
    Adam.step(g_t')

The first gradient is **not** clipped; clipping is applied exactly once per
update, after the second backward, using the frozen ``GRAD_CLIP=5.0`` rule.
The first gradient is used only for the perturbation.  No rho / ASAM /
regulariser / dropout / loss / architecture search is part of this module.

This module is label-free except :func:`train_supervised_sam`, which reads Fit
``y`` only.
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
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-p3-sam-v1"
SPLIT_TAG = "T"
SPLIT_SALT = "cscl-p3-sam-v1"

SEEDS = (0, 1)
ARMS = ("P3", "P3SAM")
RECIPE_KEY = "R3"
SAM_RHO = 0.05
SAM_STEPS = 2

BASE_PARAMETERS = 413_505
PARAM_CAP = 500_000

MEANINGFUL_DELTA = 0.004
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
BOOTSTRAP_N = 5_000
BOOTSTRAP_SEED = 0
RMSE_TOL = 0.005
TAIL_TOL = 0.005
GPU_BUDGET_CAP_HOURS = 3.0

ZONES = ("monitor", "locked")
PRIMARY_PHASE_ZONE = "monitor"


class P3SAMError(RuntimeError):
    """Hard protocol violation (raises instead of silently continuing)."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise P3SAMError(message)


# ---------------------------------------------------------------------------
# model / hashes
# ---------------------------------------------------------------------------


def build_p3(seed: int, norm: Mapping[str, Any], device: torch.device | str) -> src.BJOC:
    """The exact frozen P3 scratch init (413,505 parameters)."""
    model = bp.build_bjoc_null(seed, norm, device)
    require(
        int(sum(p.numel() for p in model.parameters())) == BASE_PARAMETERS,
        "P3 parameter audit failed",
    )
    return model


def full_state_hash(model: nn.Module) -> str:
    return src.state_dict_hash(model)


def encoder_hash(model: src.BJOC) -> str:
    return bp.encoder_group_hash(model)


def shared_group_hash(model: src.BJOC) -> str:
    return mp.group_state_hash(model)


def model_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_model(model: nn.Module, objs: Sequence[sro.MolObjects], device: torch.device | str) -> np.ndarray:
    return bp.predict_model(model, objs, device)


def _grad_norm(parameters: Sequence[nn.Parameter] | Any) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return float(math.sqrt(total))


# ---------------------------------------------------------------------------
# two-step SAM primitive
# ---------------------------------------------------------------------------


class TwoStepSAM:
    """Standard two-step SAM perturbation with exact parameter restoration.

    ``perturb()`` requires the first-backward gradients to be present.  It
    stores an exact clone of every parameter, adds the global-L2-normalised
    gradient perturbation, and leaves the first gradients in place for the
    caller to clear.  ``restore()`` copies the exact originals back and returns
    the residual error (0.0 for a correct implementation).
    """

    def __init__(self, parameters: Sequence[nn.Parameter], rho: float = SAM_RHO) -> None:
        self.parameters = [p for p in parameters if p.requires_grad]
        require(self.parameters, "SAM received no trainable parameters")
        self.rho = float(rho)
        self._originals: list[torch.Tensor] | None = None
        self._last: dict[str, Any] | None = None

    @torch.no_grad()
    def perturb(self) -> dict[str, Any]:
        require(self._originals is None, "SAM perturb() called twice without restore()")
        total_sq = 0.0
        for p in self.parameters:
            if p.grad is not None:
                total_sq += float(p.grad.detach().pow(2).sum())
        grad_norm = float(math.sqrt(total_sq))
        scale = self.rho / (grad_norm + 1e-12)
        self._originals = [p.detach().clone() for p in self.parameters]
        perturb_sq = 0.0
        n_perturbed = 0
        for p in self.parameters:
            if p.grad is None:
                continue
            e = p.grad.detach() * scale
            p.add_(e)
            perturb_sq += float(e.pow(2).sum())
            n_perturbed += 1
        self._last = {
            "grad_norm": grad_norm,
            "perturbation_norm": float(math.sqrt(perturb_sq)),
            "n_perturbed_parameters": int(n_perturbed),
            "rho": self.rho,
        }
        return dict(self._last)

    @torch.no_grad()
    def restore(self) -> dict[str, Any]:
        require(self._originals is not None, "SAM restore() called without perturb()")
        before = 0.0
        for p, orig in zip(self.parameters, self._originals):
            before = max(before, float((p.detach() - orig).abs().max()))
        for p, orig in zip(self.parameters, self._originals):
            p.copy_(orig)
        after = 0.0
        for p, orig in zip(self.parameters, self._originals):
            after = max(after, float((p.detach() - orig).abs().max()))
        self._originals = None
        return {
            "max_abs_delta_before_restore": before,
            "max_abs_error_after_restore": after,
        }


# ---------------------------------------------------------------------------
# last-5 soup
# ---------------------------------------------------------------------------


def _last_soup(states: Sequence[Mapping[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    require(0 < len(states) <= bp.SOUP_LAST, f"unexpected soup state count {len(states)}")
    members = list(range(int(last_epoch) - len(states) + 1, int(last_epoch) + 1))
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0].keys()}
    return members, soup


# ---------------------------------------------------------------------------
# supervised SAM loop (mirror of bp.train_supervised + two-step SAM)
# ---------------------------------------------------------------------------


def train_supervised_sam(
    *,
    arm: str = "P3SAM",
    seed: int,
    model: nn.Module,
    train_objs: Sequence[sro.MolObjects],
    y_fit: np.ndarray,
    device: torch.device,
    recipe_key: str = RECIPE_KEY,
    epochs: int | None = None,
    batch_size: int | None = None,
    rho: float = SAM_RHO,
    grad_clip: float = bp.GRAD_CLIP,
    log_fn: Callable[[str], None] = print,
    collect_epoch_states: bool = False,
    clip_fn: Callable[[Sequence[nn.Parameter], float], Any] | None = None,
    _allow_test_rho_zero: bool = False,
    _allow_test_grad_clip: bool = False,
) -> dict[str, Any]:
    """The frozen R3 loop with a standard two-step SAM outer step.

    The clean first forward/backward is the same computation as one P3 update;
    the second forward/backward is the SAM cost.  Training MAE is accumulated
    from the clean forward so A/B fit curves share a definition.
    """
    require(arm == "P3SAM", f"SAM trainer only accepts P3SAM, got {arm!r}")
    require(recipe_key in bp.RECIPES, f"unknown recipe {recipe_key!r}")
    recipe = bp.RECIPES[recipe_key]
    require(epochs is None or 1 <= int(epochs), "epochs must be positive")
    total_epochs = int(epochs if epochs is not None else recipe["epochs"])
    bs = int(batch_size if batch_size is not None else bp.BATCH_SIZE)
    require(bs > 0, "batch size must be positive")
    n = int(len(train_objs))
    require(n == int(len(y_fit)), "train objects and labels differ in length")
    steps_per_epoch = math.ceil(n / bs)
    total_updates = total_epochs * steps_per_epoch
    require(
        float(rho) == float(SAM_RHO) or (_allow_test_rho_zero and float(rho) == 0.0),
        f"SAM rho is frozen at {SAM_RHO}, got {rho}",
    )
    require(
        float(grad_clip) == float(bp.GRAD_CLIP) or (_allow_test_grad_clip and float(grad_clip) > 0),
        f"grad clip is frozen at {bp.GRAD_CLIP}, got {grad_clip}",
    )

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise P3SAMError("cuda requested but torch.cuda.is_available() is False")

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = None
    if recipe["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"])
        )
    elif recipe["schedule"] != "constant":
        raise P3SAMError(f"unknown schedule {recipe['schedule']!r}")

    sam = TwoStepSAM(params, rho=float(rho))
    clip = clip_fn if clip_fn is not None else torch.nn.utils.clip_grad_norm_
    gen = torch.Generator().manual_seed(int(seed) + bp.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=bp.SOUP_LAST)
    epoch_states: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    first_norm_max = 0.0
    second_norm_max = 0.0
    perturb_norm_min = float("inf")
    perturb_norm_max = 0.0
    restore_error_max = 0.0
    clip_call_count = 0
    clip_event_count = 0
    clip_return_minus_computed_max = 0.0
    optimizer_step_count = 0
    scheduler_step_count = 0
    total_clean = 0.0
    total_adv = 0.0
    total_mol = 0
    started = time.perf_counter()

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_clean = 0.0
        epoch_adv = 0.0
        epoch_mol = 0
        steps = 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, bs, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = move_batch(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)

            # -- clean first forward/backward (the P3 half of the cost) ------
            pred_clean = model(batch)
            loss_clean = F.l1_loss(pred_clean.view(-1), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss_clean.backward()
            pinfo = sam.perturb()
            first_norm_max = max(first_norm_max, float(pinfo["grad_norm"]))
            perturb_norm_min = min(perturb_norm_min, float(pinfo["perturbation_norm"]))
            perturb_norm_max = max(perturb_norm_max, float(pinfo["perturbation_norm"]))

            # -- second forward/backward at the perturbed point ---------------
            optimizer.zero_grad(set_to_none=True)
            pred_adv = model(batch)
            loss_adv = F.l1_loss(pred_adv.view(-1), y.view(-1))
            loss_adv.backward()
            second_norm = _grad_norm(params)
            second_norm_max = max(second_norm_max, second_norm)
            rinfo = sam.restore()
            restore_error_max = max(restore_error_max, float(rinfo["max_abs_error_after_restore"]))

            # -- clip exactly once, after the second backward -----------------
            clip_norm = float(clip(params, float(grad_clip)))
            clip_call_count += 1
            clip_return_minus_computed_max = max(
                clip_return_minus_computed_max, abs(clip_norm - second_norm)
            )
            if second_norm > float(grad_clip):
                clip_event_count += 1
            if not math.isfinite(second_norm):
                raise P3SAMError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")

            actual_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                expected_lr = bp.schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
                lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr

            optimizer.step()
            optimizer_step_count += 1
            if scheduler is not None:
                scheduler.step()
                scheduler_step_count += 1

            epoch_clean += float((pred_clean.view(-1) - y.view(-1)).abs().sum())
            epoch_adv += float((pred_adv.view(-1) - y.view(-1)).abs().sum())
            epoch_mol += len(indices)
            total_mol += len(indices)
            steps += 1

        clean_mae = epoch_clean / max(epoch_mol, 1)
        adv_mae = epoch_adv / max(epoch_mol, 1)
        total_clean += epoch_clean
        total_adv += epoch_adv
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": clean_mae,
                "train_mae_adversarial": adv_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        states.append(state)
        if collect_epoch_states:
            epoch_states.append({k: v.detach().cpu().clone() for k, v in state.items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == total_epochs:
            log_fn(
                f"[{arm} s{seed}] ep={epoch:03d} train={clean_mae:.6f} "
                f"adv={adv_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s"
            )
        if not math.isfinite(clean_mae):
            raise P3SAMError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")

    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), total_epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_model(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    independent_order = bp.batch_order_hash(seed, n, bs, total_epochs)
    info: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "recipe_key": recipe_key,
        "recipe": dict(recipe),
        "epochs": total_epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "batch_size": bs,
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
        "anomalies": {
            "grad_clip_events": int(clip_event_count),
            "grad_clip_calls": int(clip_call_count),
            "first_grad_norm_max": float(first_norm_max),
            "second_grad_norm_max": float(second_norm_max),
            "perturbation_norm_min": float(perturb_norm_min),
            "perturbation_norm_max": float(perturb_norm_max),
            "restore_max_abs_error": float(restore_error_max),
            "clip_return_minus_computed_max": float(clip_return_minus_computed_max),
            "optimizer_step_count": int(optimizer_step_count),
            "scheduler_step_count": int(scheduler_step_count),
            "scheduler_last_epoch": int(scheduler.last_epoch) if scheduler is not None else 0,
        },
        "sam": {
            "variant": "standard_two_step_global_l2",
            "rho": float(rho),
            "steps": int(SAM_STEPS),
            "same_minibatch_for_both_steps": True,
            "first_gradient_clipped": False,
            "second_gradient_clipped": True,
            "clip_after_second_backward": True,
            "restore": "exact parameter clone copied back before clip/Adam step",
        },
        "cost": {
            "optimizer_updates": int(total_updates),
            "forward_calls": int(2 * total_updates),
            "backward_calls": int(2 * total_updates),
            "relative_forward_backward_multiplier_vs_p3": 2.0,
        },
        "curve": curve,
        "wall_seconds": float(wall),
        "molecules_seen": int(total_mol),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "full_hash_independent_recomputation": independent_order,
            "full_matches": order_hash.hexdigest() == independent_order,
        },
        "parameter_audit": {"actual_parameters": model_parameters(model)},
        "labels_used": "train rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if collect_epoch_states:
        info["epoch_states"] = epoch_states
    return info


# ---------------------------------------------------------------------------
# synthetic CPU gates
# ---------------------------------------------------------------------------


def _mol(idx: int, atom_types: Sequence[int], bonds: Sequence[tuple[int, int, int]]) -> sro.MolObjects:
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    edge_index, edge_attr = [], []
    for u, v, bt in bonds:
        edge_index.extend([(u, v), (v, u)])
        edge_attr.extend([bt, bt])
    return sro.build_mol_objects(
        idx,
        x,
        np.asarray(edge_index, dtype=np.int64).T.reshape(2, -1),
        np.asarray(edge_attr, dtype=np.int64),
    )


def synthetic_objects() -> list[sro.MolObjects]:
    """Eight tiny label-free molecules used only by CPU gates."""
    return [
        _mol(0, range(6), [(i, (i + 1) % 6, 1) for i in range(6)]),
        _mol(1, range(7), [(i, (i + 1) % 7, 1) for i in range(7)]),
        _mol(2, range(5), [(i, i + 1, 1) for i in range(4)]),
        _mol(3, range(8), [(i, i + 1, 1) for i in range(7)]),
        _mol(
            4,
            range(10),
            [
                (0, 1, 2),
                (1, 2, 1),
                (2, 3, 2),
                (3, 4, 1),
                (4, 5, 2),
                (5, 0, 1),
                (4, 9, 1),
                (9, 8, 2),
                (8, 7, 1),
                (7, 6, 2),
                (6, 5, 1),
            ],
        ),
        _mol(
            5,
            range(7),
            [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1), (0, 4, 1), (4, 5, 1), (5, 6, 1), (6, 0, 1)],
        ),
        _mol(6, range(6), [(i, (i + 1) % 6, 2 if i % 2 == 0 else 1) for i in range(6)]),
        _mol(7, range(4), [(i, i + 1, 1) for i in range(3)]),
    ]


def _gate(name: str, ok: Any, **detail: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), **detail}


def _tiny_norm() -> dict[str, Any]:
    return {
        "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    }


def _batch(objs: Sequence[sro.MolObjects]) -> dict[str, Any]:
    return move_batch(sro.batch_objects(objs), torch.device("cpu"))


def run_sam_unit_gates(n_fit: int | None = None) -> dict[str, Any]:
    """CPU-only correctness gates for the SAM primitive and frozen loop.

    No ZINC data, no checkpoints, no GPU, no labels from any split.  The
    synthetic molecules and synthetic targets are fixed in this module.
    """
    gates: list[dict[str, Any]] = []
    objs = synthetic_objects()
    norm = _tiny_norm()
    y = np.asarray([0.5, -0.3, 0.1, 0.9, -0.7, 0.2, 0.4, -0.1], dtype=np.float64)
    torch.set_num_threads(1)

    # G1: perturbation + exact restore on a tiny linear model.
    lin = nn.Linear(3, 1)
    x = torch.as_tensor([[1.0, 2.0, -1.0], [0.5, -0.2, 0.3]], dtype=torch.float32)
    target = torch.as_tensor([1.0, -1.0], dtype=torch.float32)
    loss = F.l1_loss(lin(x).view(-1), target)
    loss.backward()
    sam = TwoStepSAM(lin.parameters(), rho=SAM_RHO)
    before = {k: v.detach().clone() for k, v in lin.state_dict().items()}
    p = sam.perturb()
    moved = max(float((lin.state_dict()[k] - before[k]).abs().max()) for k in before)
    r = sam.restore()
    after = max(float((lin.state_dict()[k] - before[k]).abs().max()) for k in before)
    gates.append(
        _gate(
            "G1_parameter_perturbation_restore_exact",
            moved > 0.0 and after == 0.0 and r["max_abs_error_after_restore"] == 0.0,
            max_abs_move=float(moved),
            max_abs_error_after_restore=float(after),
            perturbation_norm=float(p["perturbation_norm"]),
        )
    )

    # G2: the perturbation has global L2 norm rho (all grads non-zero).
    gates.append(
        _gate(
            "G2_perturbation_norm_equals_rho",
            abs(float(p["perturbation_norm"]) - SAM_RHO) <= 1e-6,
            perturbation_norm=float(p["perturbation_norm"]),
            rho=float(SAM_RHO),
        )
    )

    # G3/G4: real SAM loop, update count + clipping after the second backward.
    model = build_p3(123, norm, "cpu")
    clip_calls: list[float] = []

    def _recording_clip(parameters: Sequence[nn.Parameter], max_norm: float) -> Any:
        clip_calls.append(_grad_norm(parameters))
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    info_clip = train_supervised_sam(
        arm="P3SAM",
        seed=123,
        model=model,
        train_objs=objs,
        y_fit=y,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=4,
        grad_clip=1e-8,
        clip_fn=_recording_clip,
        _allow_test_grad_clip=True,
        log_fn=lambda _m: None,
    )
    updates = 2 * math.ceil(len(objs) / 4)
    anomalies = info_clip["anomalies"]
    gates.append(
        _gate(
            "G3_optimizer_step_count_matches_update_budget",
            info_clip["updates"] == updates
            and anomalies["optimizer_step_count"] == updates
            and anomalies["scheduler_step_count"] == updates
            and anomalies["scheduler_last_epoch"] == updates,
            expected_updates=int(updates),
            updates=int(info_clip["updates"]),
            optimizer_step_count=int(anomalies["optimizer_step_count"]),
            scheduler_step_count=int(anomalies["scheduler_step_count"]),
            scheduler_last_epoch=int(anomalies["scheduler_last_epoch"]),
        )
    )
    gates.append(
        _gate(
            "G4_clip_only_after_second_backward",
            len(clip_calls) == updates
            and anomalies["grad_clip_calls"] == updates
            and anomalies["grad_clip_events"] == updates
            and all(v > 1e-8 for v in clip_calls)
            and anomalies["clip_return_minus_computed_max"] <= 1e-5,
            clip_calls=int(len(clip_calls)),
            grad_clip_calls=int(anomalies["grad_clip_calls"]),
            grad_clip_events=int(anomalies["grad_clip_events"]),
            second_grad_norm_max=float(anomalies["second_grad_norm_max"]),
            first_grad_norm_max=float(anomalies["first_grad_norm_max"]),
            clip_return_minus_computed_max=float(anomalies["clip_return_minus_computed_max"]),
        )
    )

    # G5: last-5 soup is exactly the mean of the last five epoch states.
    model_soup = build_p3(124, norm, "cpu")
    info_soup = train_supervised_sam(
        arm="P3SAM",
        seed=124,
        model=model_soup,
        train_objs=objs,
        y_fit=y,
        device=torch.device("cpu"),
        epochs=6,
        batch_size=4,
        log_fn=lambda _m: None,
        collect_epoch_states=True,
    )
    epoch_states = info_soup.pop("epoch_states")
    expected_members = [2, 3, 4, 5, 6]
    expected_soup = {
        k: torch.stack([epoch_states[i - 1][k].float() for i in expected_members]).mean(0)
        for k in epoch_states[0]
    }
    soup_diff = max(
        float((info_soup["soup_state"][k] - expected_soup[k]).abs().max()) for k in expected_soup
    )
    gates.append(
        _gate(
            "G5_last5_soup_rule",
            info_soup["soup_members"] == expected_members and soup_diff <= 1e-7,
            soup_members=info_soup["soup_members"],
            expected_members=expected_members,
            max_abs_soup_diff=float(soup_diff),
        )
    )

    # G6: rho=0 SAM is bitwise/equivalently the frozen P3 trainer on the same
    # batch order, optimiser, schedule and soup rule (dropout is frozen at 0).
    model_a = build_p3(125, norm, "cpu")
    info_a = bp.train_supervised(
        arm="P3",
        seed=125,
        recipe_key=RECIPE_KEY,
        init="scratch",
        train_objs=objs,
        rows=[o.mol_index for o in objs],
        y_fit=y,
        norm=norm,
        device=torch.device("cpu"),
        smoke_epochs=2,
        log_fn=lambda _m: None,
    )
    model_b = build_p3(125, norm, "cpu")
    info_b = train_supervised_sam(
        arm="P3SAM",
        seed=125,
        model=model_b,
        train_objs=objs,
        y_fit=y,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=bp.BATCH_SIZE,
        rho=0.0,
        _allow_test_rho_zero=True,
        log_fn=lambda _m: None,
    )
    soup_diff_eq = max(
        float((info_a["soup_state"][k] - info_b["soup_state"][k]).abs().max()) for k in info_a["soup_state"]
    )
    gates.append(
        _gate(
            "G6_rho0_equals_frozen_p3_trainer",
            soup_diff_eq <= 1e-7
            and info_a["batch_order"]["full_hash"] == info_b["batch_order"]["full_hash"]
            and info_a["updates"] == info_b["updates"],
            max_abs_soup_diff=float(soup_diff_eq),
            batch_order_match=bool(info_a["batch_order"]["full_hash"] == info_b["batch_order"]["full_hash"]),
            updates_a=int(info_a["updates"]),
            updates_b=int(info_b["updates"]),
        )
    )

    # G7: actual Fit-size update budget and batch-order hash (if supplied).
    if n_fit is not None:
        expected = int(bp.RECIPES[RECIPE_KEY]["epochs"]) * math.ceil(int(n_fit) / bp.BATCH_SIZE)
        h0 = bp.batch_order_hash(0, int(n_fit), bp.BATCH_SIZE, int(bp.RECIPES[RECIPE_KEY]["epochs"]))
        h1 = bp.batch_order_hash(1, int(n_fit), bp.BATCH_SIZE, int(bp.RECIPES[RECIPE_KEY]["epochs"]))
        gates.append(
            _gate(
                "G7_fit_size_update_budget",
                expected == 22_680 and h0 != h1,
                n_fit=int(n_fit),
                expected_updates=int(expected),
                seed0_order_sha256=h0,
                seed1_order_sha256=h1,
            )
        )

    all_passed = all(g["ok"] for g in gates)
    return {
        "protocol": PROTOCOL_ID,
        "suite": "cpu_sam_two_step_and_frozen_loop",
        "device": "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gates": gates,
        "all_passed": bool(all_passed),
        "labels_read": False,
        "zinc_loaded": False,
        "gpu_used": False,
    }
