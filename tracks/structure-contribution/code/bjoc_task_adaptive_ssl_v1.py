"""Task-adaptive masked SSL for cscl-bjoc-task-adaptive-ssl-v1.

Protocol: ``configs/bjoc_task_adaptive_ssl_v1.yaml``.
Track: ``tracks/structure-contribution``.

This module adds exactly one new training procedure on top of the frozen
BJOC-Null + masked-SSL + R3 system:

**TA-SSL** keeps a low-frequency masked-atom reconstruction objective alive
during supervised training.  Every 4th minibatch (0-based global update index
``t`` with ``t % 4 == 0``) the loss becomes

    L_t = L_property + 0.1 * CE(h_atom(E_theta(masked view)), hidden atom type)

while all other minibatches keep ``L_t = L_property`` (raw-y L1).  The masked
view is built from the *same* Fit molecules of the current batch with the
frozen ``mp.build_masked_sample`` path (15% atoms / 15% bonds, full rebuild),
the atom context is computed by the *same* static-incidence formula as the
frozen ``mp.PretextModel``, and the auxiliary head starts from the *same
seed's* Fit-only SSL atom head.  There is exactly one joint optimizer step per
minibatch; the R3 schedule, batch order and last-5 soup rule are those of the
frozen P3 loop.

Everything else in this module is audit / diagnostics code.  Monitor and
Locked ``.y`` are never read here; supervised training reads Fit raw ``y``
only.
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

PROTOCOL_ID = "cscl-bjoc-task-adaptive-ssl-v1"

#: frozen candidate constants (no search after this freeze)
AUX_EVERY = 4
AUX_LAMBDA = 0.1
AUX_LOSS = "atom CE only (bond CE is not used during the supervised phase)"
AUX_VISIT_RULE = "visit_index = epoch - 1"

#: split D fingerprints (copied from configs/bjoc_architecture_breakthrough_v1.yaml)
SPLIT_SALT = "cscl-bjoc-arch-breakthrough-v1-repl"
SPLIT_FINGERPRINTS: dict[str, str] = {
    "fit_sha256_rows": "9468656affcc2b7c69409f03698b1b6287b3a42dc6e164efd19dfb0190254525",
    "monitor_sha256_rows": "d196bc75ce898c7cfd61efa133372f25803d5bc1198210f60b6e0f981a400393",
    "locked_sha256_rows": "f3525ae882114350a57f7b0ddb6889656f40c05cebda21b2611a2c99cdfa22d5",
}

#: approximate compute-control stretch (5670 aux forward/backward passes)
CONTROL_EPOCHS = 450
CONTROL_UPDATES = CONTROL_EPOCHS * math.ceil(8000 / bp.BATCH_SIZE)  # 28,350


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# hashes
# ---------------------------------------------------------------------------


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return bp.state_group_sha256(state)


def atom_head_parameters(cfg: src.BJOCConfig | None = None) -> int:
    cfg = src.BJOCConfig() if cfg is None else cfg
    head = build_atom_head(cfg)
    return int(sum(p.numel() for p in head.parameters()))


def build_atom_head(cfg: src.BJOCConfig | None = None) -> nn.Module:
    """The frozen SSL atom pretext head (same width/activation as mp.PretextModel)."""
    cfg = src.BJOCConfig() if cfg is None else cfg
    return src._MLP(4 * int(cfg.atom_out), mp.ATOM_HEAD_HIDDEN, sro.ATOM_CATEGORIES, 1, 0.0)


# ---------------------------------------------------------------------------
# SSL wrapper that captures the atom head without touching the frozen loop
# ---------------------------------------------------------------------------


def pretrain_masked_with_head(
    *,
    seed: int,
    fit_objs: Sequence[sro.MolObjects],
    fit_rows: Sequence[int],
    raw_dataset,
    device: torch.device,
    mode: str = "atom_bond",
    smoke_updates: int | None = None,
    log_fn: Callable[[str], None] = bp.log,
) -> dict[str, Any]:
    """Run the frozen ``bp.pretrain_masked`` loop and additionally return the
    last-step atom pretext head state.

    The frozen loop is executed *unchanged*: this function only substitutes a
    trivial subclass of ``mp.PretextModel`` for the duration of the call so
    that the pretext instance can be captured after the last optimizer step.
    Mask RNG, shuffle RNG, optimizer, clipping, encoder updates and the
    last-step checkpoint rule are exactly those of ``bp.pretrain_masked``.
    """
    captured: dict[str, Any] = {}
    original = mp.PretextModel

    class _HeadCapture(original):  # type: ignore[misc, valid-type]
        def __init__(self, backbone: nn.Module) -> None:
            super().__init__(backbone)
            captured["pretext"] = self

    mp.PretextModel = _HeadCapture
    try:
        payload = bp.pretrain_masked(
            seed=seed,
            fit_objs=fit_objs,
            fit_rows=fit_rows,
            raw_dataset=raw_dataset,
            device=device,
            mode=mode,
            smoke_updates=smoke_updates,
            log_fn=log_fn,
        )
    finally:
        mp.PretextModel = original
    if "pretext" not in captured:
        raise RuntimeError("pretext head capture failed: bp.pretrain_masked did not build a PretextModel")
    head_state = {
        k: v.detach().cpu().clone() for k, v in captured["pretext"].atom_head.state_dict().items()
    }
    payload["atom_head_state"] = head_state
    payload["atom_head_sha256"] = state_sha256(head_state)
    payload["atom_head_parameters"] = int(sum(v.numel() for v in head_state.values()))
    payload["atom_head_source"] = (
        "same-run last-step masked-SSL atom pretext head; the SSL loop itself is the frozen "
        "bp.pretrain_masked implemented in bjoc_perfmech_long_v1.py"
    )
    payload["wrapper"] = "subclass capture around bp.pretrain_masked; no loop code changed"
    return payload


# ---------------------------------------------------------------------------
# auxiliary branch
# ---------------------------------------------------------------------------


def atom_pretext_logits(
    model: src.BJOC,
    batch: Mapping[str, torch.Tensor],
    atom_ids: torch.Tensor,
    atom_head: nn.Module,
) -> torch.Tensor:
    """Atom logits with the exact context of ``mp.PretextModel.forward``.

    Uses the same three encoders and the same static incidence gathers; only
    the atom branch is computed (bond head not needed during the supervised
    auxiliary step).
    """
    a = model.encode_atoms(batch)
    b = model.encode_bonds(a, batch)
    w = model.encode_junctions(a, b, batch)
    inc = mp.PretextModel._incidence(batch)
    n_atoms = int(a.shape[0])
    sum_b = torch.zeros((n_atoms, b.shape[1]), device=a.device, dtype=a.dtype)
    sum_b = sum_b.index_add(0, inc["hole_atom"], b[inc["hole_bond"]])
    sum_wc = torch.zeros((n_atoms, w.shape[1]), device=a.device, dtype=a.dtype)
    sum_wc = sum_wc.index_add(0, inc["junc_center_atom"], w)
    sum_we = torch.zeros((n_atoms, w.shape[1]), device=a.device, dtype=a.dtype)
    sum_we = sum_we.index_add(0, inc["junc_endpoint_atom"], w[inc["junc_endpoint_row"]])
    cat_atom = torch.cat([a, sum_b, sum_wc, sum_we], dim=1)
    return atom_head(cat_atom[atom_ids])


def aux_masked_batch(
    objs: Sequence[sro.MolObjects],
    indices: Sequence[int],
    raw_dataset,
    *,
    seed: int,
    visit_index: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Masked view of the *same* molecules (full rebuild, structure fields only)."""
    samples = []
    for j in indices:
        o = objs[int(j)]
        d = raw_dataset[int(o.mol_index)]
        samples.append(
            mp.build_masked_sample(
                o,
                d.x.numpy(),
                d.edge_index.numpy(),
                d.edge_attr.numpy(),
                seed=int(seed),
                visit_index=int(visit_index),
            )
        )
    batch, targets_a, targets_b, _, _ = mp.batch_pretext_samples(samples)
    return batch, targets_a, targets_b


# ---------------------------------------------------------------------------
# task-adaptive supervised loop
# ---------------------------------------------------------------------------


def train_task_adaptive(
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
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    atom_head_state: Mapping[str, torch.Tensor] | None = None,
    aux_every: int | None = None,
    aux_lambda: float = AUX_LAMBDA,
    raw_dataset=None,
    smoke_epochs: int | None = None,
    recipe_override: Mapping[str, Any] | None = None,
    log_fn: Callable[[str], None] = bp.log,
) -> dict[str, Any]:
    """One supervised trajectory; the property branch is the frozen P3 loop.

    ``aux_every=None`` reproduces ``bp.train_supervised`` (model_kind="base")
    bit for bit; the frozen loop is used for the reported P3 arms, and this
    function is the TA-SSL arm with ``aux_every=4``.
    """
    recipe = dict(bp.RECIPES[recipe_key] if recipe_override is None else recipe_override)
    if init not in ("scratch", "ssl"):
        raise SystemExit(f"unknown init {init!r}")
    if init == "ssl" and ssl_state is None:
        raise SystemExit("init=ssl requires a pretrained encoder state")
    use_aux = aux_every is not None
    if use_aux:
        if int(aux_every) < 1:
            raise SystemExit("aux_every must be >= 1")
        if atom_head_state is None:
            raise SystemExit("aux branch requires the SSL atom head state")
        if raw_dataset is None:
            raise SystemExit("aux branch requires a raw structure dataset")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")

    model = bp.build_bjoc_null(seed, norm, device)
    init_full_sha = src.state_dict_hash(model)
    init_group_sha = mp.group_state_hash(model)
    init_encoder_sha = bp.encoder_group_hash(model)
    if init == "ssl":
        mp.load_encoder_state(model, ssl_state, device)
    encoder_used_sha = bp.encoder_group_hash(model)
    if init == "ssl" and encoder_used_sha == init_encoder_sha:
        raise SystemExit("init=ssl did not change the encoder state")
    audit = src.parameter_audit(src.BJOCConfig(), model)
    if not audit["parameter_exact"] or audit["actual_parameters"] != bp.EXPECTED_PARAMETERS:
        raise SystemExit(f"parameter audit failed: {audit}")

    atom_head = None
    head_info: dict[str, Any] = {}
    train_params: list[nn.Parameter] = list(model.parameters())
    if use_aux:
        atom_head = build_atom_head()
        atom_head.load_state_dict({k: v.to(device) for k, v in atom_head_state.items()})
        atom_head.to(device)
        head_info = {
            "atom_head_parameters": int(sum(p.numel() for p in atom_head.parameters())),
            "atom_head_sha256": state_sha256(atom_head.state_dict()),
            "training_parameters": int(
                sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in atom_head.parameters())
            ),
            "inference_parameters": int(sum(p.numel() for p in model.parameters())),
        }
        train_params = train_params + list(atom_head.parameters())

    epochs = int(recipe["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(rows))
    steps_per_epoch = math.ceil(n / bp.BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(
        train_params, lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"])
    )
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
    aux_updates_total = 0
    aux_molecules_total = 0
    aux_ce_all: list[float] = []
    aux_acc_all: list[float] = []
    aux_seconds_total = 0.0
    head_grad_norm_max = 0.0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        if atom_head is not None:
            atom_head.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        ep_ce: list[float] = []
        ep_acc: list[float] = []
        ep_aux_batches = 0
        ep_aux_seconds = 0.0
        for indices in bp.shuffle_batches(n, bp.BATCH_SIZE, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = bp._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            t = (epoch - 1) * steps_per_epoch + steps
            if use_aux and (t % int(aux_every) == 0):
                t_aux = time.perf_counter()
                masked_batch, targets_a, _ = aux_masked_batch(
                    train_objs, indices, raw_dataset, seed=seed, visit_index=epoch - 1
                )
                masked_batch = bp._mv(masked_batch, device)
                targets_a = targets_a.to(device)
                atom_logits = atom_pretext_logits(
                    model, masked_batch, masked_batch["masked_atom_ids"], atom_head
                )
                aux_ce = F.cross_entropy(atom_logits, targets_a)
                loss = loss + float(aux_lambda) * aux_ce
                ep_aux_seconds += time.perf_counter() - t_aux
                ep_aux_batches += 1
                with torch.no_grad():
                    acc = float((atom_logits.argmax(1) == targets_a).float().mean())
                    ep_ce.append(float(aux_ce.detach()))
                    ep_acc.append(acc)
                aux_updates_total += 1
                aux_molecules_total += len(indices)
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(train_params, bp.GRAD_CLIP))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > bp.GRAD_CLIP:
                clip_events += 1
            if atom_head is not None:
                hg = math.sqrt(
                    sum(
                        float((p.grad.detach() ** 2).sum())
                        for p in atom_head.parameters()
                        if p.grad is not None
                    )
                )
                if not math.isfinite(hg):
                    raise RuntimeError(f"non-finite auxiliary head gradient at epoch {epoch}")
                head_grad_norm_max = max(head_grad_norm_max, hg)
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
        ep_ce_mean = float(np.mean(ep_ce)) if ep_ce else None
        ep_acc_mean = float(np.mean(ep_acc)) if ep_acc else None
        if ep_ce_mean is not None:
            aux_ce_all.append(ep_ce_mean)
            aux_acc_all.append(float(ep_acc_mean))
        aux_seconds_total += ep_aux_seconds
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "aux_ce": ep_ce_mean,
                "aux_acc": ep_acc_mean,
                "aux_batches": int(ep_aux_batches),
                "aux_seconds": float(ep_aux_seconds),
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        )
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            extra = (
                f" aux_ce={ep_ce_mean:.4f} aux_acc={ep_acc_mean:.4f} aux_batches={ep_aux_batches}"
                if ep_ce_mean is not None
                else ""
            )
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e}{extra} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")
    wall = time.perf_counter() - started

    members, soup = bp._last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = bp.predict_model(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_head_state = None
    if use_aux:
        final_head_state = {k: v.detach().cpu().clone() for k, v in atom_head.state_dict().items()}
    return {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "recipe_key": recipe_key,
        "recipe": dict(recipe),
        "init": init,
        "model_kind": "base",
        "aux_every": None if not use_aux else int(aux_every),
        "aux_lambda": None if not use_aux else float(aux_lambda),
        "aux_loss": AUX_LOSS if use_aux else None,
        "aux_visit_rule": AUX_VISIT_RULE if use_aux else None,
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {bp.SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_state": soup_state,
        "final_atom_head_state": final_head_state,
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
        "head_info": head_info,
        "parameter_audit": audit,
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "full_hash_independent_recomputation": bp.batch_order_hash(seed, n, bp.BATCH_SIZE, epochs),
            "full_matches": order_hash.hexdigest() == bp.batch_order_hash(seed, n, bp.BATCH_SIZE, epochs),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": grad_norm_max,
            "aux_head_grad_norm_max": head_grad_norm_max,
            "lr_schedule_deviation_max": lr_deviation_max,
        },
        "aux_stats": {
            "aux_updates": int(aux_updates_total),
            "aux_molecules": int(aux_molecules_total),
            "aux_updates_expected": int((total_updates + int(aux_every) - 1) // int(aux_every)) if use_aux else 0,
            "aux_ce_first": aux_ce_all[0] if aux_ce_all else None,
            "aux_ce_last": aux_ce_all[-1] if aux_ce_all else None,
            "aux_acc_first": aux_acc_all[0] if aux_acc_all else None,
            "aux_acc_last": aux_acc_all[-1] if aux_acc_all else None,
            "aux_seconds_total": float(aux_seconds_total),
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
# gradient diagnostics (Fit-only; no Monitor/Locked labels)
# ---------------------------------------------------------------------------


def _group_norms(
    grads: Mapping[str, Sequence[torch.Tensor | None]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, group in grads.items():
        total = 0.0
        for g in group:
            if g is None:
                continue
            total += float((g.detach() ** 2).sum())
        out[name] = math.sqrt(total)
    return out


def _flat_grads(grads: Sequence[torch.Tensor | None]) -> torch.Tensor:
    return torch.cat([g.detach().reshape(-1) for g in grads if g is not None])


def gradient_diagnostics(
    *,
    seed: int,
    train_objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    device: torch.device,
    raw_dataset,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None,
    atom_head_state: Mapping[str, torch.Tensor],
    n_batches: int = 4,
    batch_size: int = 64,
    visit_index: int = 0,
    model: src.BJOC | None = None,
    atom_head: nn.Module | None = None,
) -> dict[str, Any]:
    """Measure g_y (property L1) and g_a (aux atom CE) on the shared encoders.

    The auxiliary head is *not* part of the property graph; its property
    gradient is expected to be ``None`` (no inference bypass).
    """
    if model is None:
        model = bp.build_bjoc_null(seed, norm, device)
        if ssl_state is not None:
            mp.load_encoder_state(model, ssl_state, device)
    if atom_head is None:
        atom_head = build_atom_head()
        atom_head.load_state_dict({k: v.to(device) for k, v in atom_head_state.items()})
        atom_head.to(device)
    groups: dict[str, list[nn.Parameter]] = {
        "atom_encoder": list(model.atom_encoder.parameters()),
        "bond_encoder": list(model.bond_encoder.parameters()),
        "junction_encoder": list(model.junction_encoder.parameters()),
    }
    enc_params = [p for g in groups.values() for p in g]
    head_params = list(atom_head.parameters())
    rng = np.random.default_rng(int(seed) + 4242)
    order = rng.permutation(len(rows))
    per_batch: list[dict[str, Any]] = []
    model.train()
    atom_head.train()
    for b in range(int(n_batches)):
        chunk = [int(i) for i in order[b * int(batch_size) : (b + 1) * int(batch_size)]]
        batch = bp._mv(sro.batch_objects([train_objs[i] for i in chunk]), device)
        y = torch.as_tensor(y_fit[chunk], dtype=torch.float32, device=device)
        pred = model(batch)
        loss_y = F.l1_loss(pred.view(-1), y.view(-1))
        gy = torch.autograd.grad(loss_y, enc_params, allow_unused=True, retain_graph=True)
        head_on_property = torch.autograd.grad(loss_y, head_params, allow_unused=True, retain_graph=True)
        masked_batch, targets_a, _ = aux_masked_batch(
            train_objs, chunk, raw_dataset, seed=seed, visit_index=visit_index
        )
        masked_batch = bp._mv(masked_batch, device)
        targets_a = targets_a.to(device)
        logits = atom_pretext_logits(model, masked_batch, masked_batch["masked_atom_ids"], atom_head)
        loss_a = F.cross_entropy(logits, targets_a)
        ga_all = torch.autograd.grad(loss_a, enc_params + head_params, allow_unused=True)
        ga = ga_all[: len(enc_params)]
        ga_head = ga_all[len(enc_params) :]
        gy_groups = {
            name: list(gs) for name, gs in zip(groups, _chunked(gy, [len(g) for g in groups.values()]))
        }
        ga_groups = {
            name: list(gs) for name, gs in zip(groups, _chunked(ga, [len(g) for g in groups.values()]))
        }
        ny = _group_norms(gy_groups)
        na = _group_norms(ga_groups)
        vy, va = _flat_grads(gy), _flat_grads(ga)
        ny_total, na_total = float(vy.norm()), float(va.norm())
        cosine = float((vy @ va) / max(ny_total * na_total, 1e-12))
        weighted = vy + AUX_LAMBDA * va
        per_batch.append(
            {
                "batch": b,
                "n_molecules": len(chunk),
                "loss_property": float(loss_y.detach()),
                "loss_atom_ce": float(loss_a.detach()),
                "grad_norm_property": ny_total,
                "grad_norm_atom": na_total,
                "grad_norm_ratio_atom_over_property": na_total / max(ny_total, 1e-12),
                "cosine_property_atom": cosine,
                "grad_norm_weighted": float(weighted.norm()),
                "weighted_over_property": float(weighted.norm()) / max(ny_total, 1e-12),
                "per_group_property": ny,
                "per_group_atom": na,
                "head_grad_norm_atom": float(
                    math.sqrt(sum(float((g.detach() ** 2).sum()) for g in ga_head if g is not None))
                ),
                "head_grad_on_property_is_none": bool(all(g is None for g in head_on_property)),
                "finite": bool(
                    all(math.isfinite(v) for v in [ny_total, na_total, cosine])
                ),
            }
        )
    return {
        "protocol": PROTOCOL_ID,
        "seed": int(seed),
        "n_batches": int(n_batches),
        "batch_size": int(batch_size),
        "visit_index": int(visit_index),
        "aux_lambda": AUX_LAMBDA,
        "per_batch": per_batch,
        "mean_grad_norm_ratio_atom_over_property": float(
            np.mean([r["grad_norm_ratio_atom_over_property"] for r in per_batch])
        ),
        "mean_cosine_property_atom": float(np.mean([r["cosine_property_atom"] for r in per_batch])),
        "head_grad_on_property_is_none_all": bool(
            all(r["head_grad_on_property_is_none"] for r in per_batch)
        ),
        "head_receives_atom_gradient": bool(all(r["head_grad_norm_atom"] > 0 for r in per_batch)),
        "all_finite": bool(all(r["finite"] for r in per_batch)),
        "labels_used": "Fit rows raw y only; atom targets are structure fields",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


def _chunked(seq: Sequence[Any], sizes: Sequence[int]) -> list[list[Any]]:
    out, i = [], 0
    for size in sizes:
        out.append(list(seq[i : i + size]))
        i += size
    return out


# ---------------------------------------------------------------------------
# masked-view audits on real Fit rows (structure only)
# ---------------------------------------------------------------------------


def _sample_fields_equal(a: mp.PretextSample, b: mp.PretextSample) -> dict[str, bool]:
    fields = {}
    for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
        fields[name] = bool(np.array_equal(getattr(a.masked, name), getattr(b.masked, name)))
    fields["bond_index"] = bool(np.array_equal(a.masked.bond_index, b.masked.bond_index))
    fields["junc_index"] = bool(np.array_equal(a.masked.junc_index, b.masked.junc_index))
    fields["junc_bond"] = bool(np.array_equal(a.masked.junc_bond, b.masked.junc_bond))
    fields["mask_ids"] = bool(
        np.array_equal(a.atom_ids, b.atom_ids) and np.array_equal(a.bond_ids, b.bond_ids)
    )
    return fields


def _pretext_logits_equal(a: mp.PretextSample, b: mp.PretextSample) -> dict[str, bool]:
    cfg = src.BJOCConfig()
    model = src.build_bjoc(
        cfg,
        seed=123,
        organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )
    pretext = mp.PretextModel(model)
    ba, _, _, _, _ = mp.batch_pretext_samples([a])
    bb, _, _, _, _ = mp.batch_pretext_samples([b])
    with torch.no_grad():
        oa = pretext(ba, ba["masked_atom_ids"], ba["masked_bond_ids"])
        ob = pretext(bb, bb["masked_atom_ids"], bb["masked_bond_ids"])
    return {
        "atom_logits_equal": bool(torch.equal(oa["atom_logits"], ob["atom_logits"])),
        "atom_ctx_equal": bool(torch.equal(oa["atom_ctx"], ob["atom_ctx"])),
    }


def mask_swap_audit(
    raw_rows: Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    *,
    n_trials: int = 100,
    positive_cases: int = 20,
    seed: int = 7,
    audit_seed: int = 20261013,
) -> dict[str, Any]:
    """Changing the hidden truth at a masked position must not change any model
    input; changing a visible atom must change only dependency cells."""
    rng = np.random.default_rng(int(audit_seed))
    atom_pass = bond_pass = atom_cases = bond_cases = 0
    positive_checked = 0
    positive_ok = True
    for trial in range(int(n_trials)):
        mol_index, x_r, ei_r, ea_r = raw_rows[int(rng.integers(0, len(raw_rows)))]
        obj_r = sro.build_mol_objects(int(mol_index), x_r, ei_r, ea_r)
        if obj_r.n_atoms < 2 or obj_r.bond_index.shape[0] < 1:
            continue
        s0 = mp.build_masked_sample(obj_r, x_r, ei_r, ea_r, seed=int(seed), visit_index=trial)
        if len(s0.atom_ids):
            x_b = x_r.copy().reshape(-1)
            aid = int(s0.atom_ids[0])
            old = int(x_b[aid])
            new = (old + 1) % 27
            if new == mp.ATOM_MASK_TOKEN:
                new = (old + 2) % 27
            x_b[aid] = new
            s1 = mp.build_masked_sample(
                obj_r, x_b.reshape(x_r.shape), ei_r, ea_r, seed=int(seed), visit_index=trial
            )
            atom_cases += 1
            if all(_sample_fields_equal(s0, s1).values()) and all(_pretext_logits_equal(s0, s1).values()):
                atom_pass += 1
        if len(s0.bond_ids):
            vid = int(s0.bond_ids[0])
            old = int(obj_r.bond_type[vid])
            new = 3 if old != 3 else 1
            ea_b = ea_r.copy()
            u, v = int(obj_r.bond_index[vid][0]), int(obj_r.bond_index[vid][1])
            for col in range(ei_r.shape[1]):
                a, c = int(ei_r[0, col]), int(ei_r[1, col])
                if (min(a, c), max(a, c)) == (min(u, v), max(u, v)):
                    ea_b[col] = new
            obj_variant = sro.build_mol_objects(int(mol_index), x_r, ei_r, ea_b)
            s2 = mp.build_masked_sample(
                obj_variant, x_r, ei_r, ea_b, seed=int(seed), visit_index=trial
            )
            bond_cases += 1
            if all(_sample_fields_equal(s0, s2).values()) and all(_pretext_logits_equal(s0, s2).values()):
                bond_pass += 1
        if trial < int(positive_cases):
            visible = [a for a in range(obj_r.n_atoms) if a not in set(map(int, s0.atom_ids))]
            if visible:
                vv = int(rng.choice(visible))
                x_v = x_r.copy().reshape(-1)
                x_v[vv] = (int(x_v[vv]) + 1) % 27
                if int(x_v[vv]) == mp.ATOM_MASK_TOKEN:
                    x_v[vv] = (int(x_v[vv]) + 1) % 27
                s3 = mp.build_masked_sample(
                    obj_r, x_v.reshape(x_r.shape), ei_r, ea_r, seed=int(seed), visit_index=trial
                )
                dep = mp.dependent_cell_mask(s0.masked, [vv], [])
                diff = np.asarray(s0.masked.atom_feat) != np.asarray(s3.masked.atom_feat)
                changed = diff.any(axis=1)
                unexpected = int((changed & ~dep["atom"].any(axis=1)).sum())
                if unexpected or int(changed.sum()) == 0:
                    positive_ok = False
                positive_checked += 1
    verdict = (
        "MASK_SWAP_INVARIANT"
        if (
            atom_cases > 0
            and bond_cases > 0
            and atom_pass == atom_cases
            and bond_pass == bond_cases
            and positive_ok
        )
        else "MASK_TARGET_LEAKAGE_BLOCKED"
    )
    return {
        "protocol": PROTOCOL_ID,
        "n_trials": int(n_trials),
        "atom_swap_cases": int(atom_cases),
        "atom_swap_passed": int(atom_pass),
        "bond_swap_cases": int(bond_cases),
        "bond_swap_passed": int(bond_pass),
        "positive_control_cases": int(positive_checked),
        "positive_control_confined_to_dependency_cells": bool(positive_ok),
        "verdict": verdict,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


def rebuild_audit(
    raw_rows: Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    *,
    n_cases: int = 50,
    seed: int = 3,
    audit_seed: int = 11,
) -> dict[str, Any]:
    """Deterministic rebuild, dependency-confined diffs, own slot holds only the
    mask token, hidden truth absent, both directed PyG records replaced."""
    rng = np.random.default_rng(int(audit_seed))
    determinism_ok = subset_ok = own_ok = absent_ok = bidi_ok = True
    checked = 0
    max_ratio_gap = 0.0
    for trial in range(int(n_cases)):
        mol_index, x_r, ei_r, ea_r = raw_rows[int(rng.integers(0, len(raw_rows)))]
        base = sro.build_mol_objects(int(mol_index), x_r, ei_r, ea_r)
        s = mp.build_masked_sample(base, x_r, ei_r, ea_r, seed=int(seed), visit_index=trial)
        s_again = mp.build_masked_sample(base, x_r, ei_r, ea_r, seed=int(seed), visit_index=trial)
        for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
            if not np.array_equal(getattr(s.masked, name), getattr(s_again.masked, name)):
                determinism_ok = False
        dep = mp.dependent_cell_mask(base, s.atom_ids, s.bond_ids)
        for name, before, after, allowed in (
            ("atom", base.atom_feat, s.masked.atom_feat, dep["atom"]),
            ("bond", base.bond_feat, s.masked.bond_feat, dep["bond"]),
            ("junc", base.junc_feat, s.masked.junc_feat, dep["junc"]),
            ("global", base.global_feat, s.masked.global_feat, dep["global"]),
        ):
            diff = np.asarray(before) != np.asarray(after)
            if not np.all(diff <= allowed):
                subset_ok = False
        for a, t in zip(s.atom_ids, s.atom_types):
            if int(s.masked.atom_feat[int(a), : sro.ATOM_CATEGORIES].argmax()) != mp.ATOM_MASK_TOKEN:
                own_ok = False
            if s.masked.atom_feat[int(a), int(t)] != 0.0:
                absent_ok = False
        for b, t in zip(s.bond_ids, s.bond_types):
            if int(s.masked.bond_feat[int(b), : sro.BOND_CATEGORIES].argmax()) != mp.BOND_MASK_TOKEN:
                own_ok = False
            if s.masked.bond_feat[int(b), int(t)] != 0.0:
                absent_ok = False
        _, ea_masked = mp.mask_raw_arrays(x_r, ei_r, ea_r, base.bond_index, s.atom_ids, s.bond_ids)
        mapping = mp.directed_bond_index(ei_r)
        masked_bonds = set(int(z) for z in s.bond_ids)
        for k in range(base.bond_index.shape[0]):
            u, v = int(base.bond_index[k][0]), int(base.bond_index[k][1])
            for col in mapping[(min(u, v), max(u, v))]:
                expected = mp.BOND_MASK_TOKEN if k in masked_bonds else int(ea_r.reshape(-1)[col])
                if int(ea_masked.reshape(-1)[col]) != expected:
                    bidi_ok = False
        if base.n_atoms:
            max_ratio_gap = max(max_ratio_gap, abs(len(s.atom_ids) / base.n_atoms - mp.ATOM_MASK_RATIO))
        checked += 1
    verdict = (
        "MASKED_REBUILD_LEAK_FREE"
        if (determinism_ok and subset_ok and own_ok and absent_ok and bidi_ok)
        else "MASKED_REBUILD_INCONSISTENT"
    )
    return {
        "protocol": PROTOCOL_ID,
        "n_cases": int(checked),
        "deterministic_reproduction": bool(determinism_ok),
        "differences_confined_to_dependency_map": bool(subset_ok),
        "masked_own_slots_hold_token": bool(own_ok),
        "hidden_truth_absent_from_own_slot": bool(absent_ok),
        "both_directed_records_masked": bool(bidi_ok),
        "max_abs_observed_atom_mask_ratio_gap": float(max_ratio_gap),
        "verdict": verdict,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# invariance helpers used by Stage A4
# ---------------------------------------------------------------------------


def relabel_objects(objs: Sequence[sro.MolObjects], raw_dataset, seed: int = 0) -> list[sro.MolObjects]:
    """Pure atom-order permutation rebuilt through the raw builder."""
    rng = np.random.default_rng(int(seed))
    out = []
    for o in objs:
        d = raw_dataset[int(o.mol_index)]
        x = d.x.numpy().reshape(-1).copy()
        ei = d.edge_index.numpy().copy()
        ea = d.edge_attr.numpy().reshape(-1).copy()
        n = len(x)
        perm = rng.permutation(n)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(n)
        x2 = x[perm]
        ei2 = inv[ei]
        out.append(sro.build_mol_objects(int(o.mol_index), x2, ei2, ea))
    return out


def endpoint_swap_objects(objs: Sequence[sro.MolObjects], raw_dataset) -> list[sro.MolObjects]:
    """Swap both directed records of every physical bond (same bond types)."""
    out = []
    for o in objs:
        d = raw_dataset[int(o.mol_index)]
        x = d.x.numpy()
        ei = d.edge_index.numpy().copy()
        ea = d.edge_attr.numpy()
        ei[0], ei[1] = ei[1].copy(), ei[0].copy()
        out.append(sro.build_mol_objects(int(o.mol_index), x, ei, ea))
    return out


def batch_invariance(model: src.BJOC, objs: Sequence[sro.MolObjects], device: torch.device) -> dict[str, Any]:
    def forward(chunk):
        batch = bp._mv(sro.batch_objects(chunk), device)
        with torch.no_grad():
            return model(batch).detach().cpu().numpy()

    y_all = forward(list(objs))
    y_single = np.asarray([forward([o])[0] for o in objs])
    return {
        "batch_membership_max_abs_diff": float(np.abs(y_all - y_single).max()),
        "all_finite": bool(np.isfinite(y_all).all()),
    }


def model_state_keys(model: src.BJOC) -> list[str]:
    return sorted(model.state_dict().keys())
