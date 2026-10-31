"""Training, prediction and frozen-checkpoint diagnostics for the pilot round.

Recipe (locked, identical for all five arms)::

    AdamW(betas=0.9/0.999, eps=1e-8), peak LR 3e-4, per-step cosine to 1e-5,
    weight decay 1e-3, batch 128, 360 epochs, grad clip 5.0,
    dropout 0.10 on the final atom-level h (v2 ``DropoutAtLocal`` position),
    MSE on the fold-Train-standardised target,
    soup of the last five epoch-end state dicts, no early stopping,
    no dev-based checkpoint selection.

Target standardisation, feature standardisation and every batch come from the
frozen v2 Fold-2 view: 1568 Train molecules / 784 Dev molecules.  No Monitor,
Reserved or random42 label is ever read.
"""

from __future__ import annotations

import copy
import csv
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import pilot_common as C
import pilot_models as M
import pilot_pairs as P

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _grad_sq_sum(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total


def _average_states(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("empty soup")
    keys = list(states[0].keys())
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        out[key] = torch.stack([s[key].detach().float().cpu() for s in states], dim=0).mean(dim=0)
    return out


def resolve_device(device: str) -> torch.device:
    text = str(device)
    if text == "cpu":
        return torch.device("cpu")
    if text in ("cuda", "cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        # rr sets CUDA_VISIBLE_DEVICES per pool; inside the process the pool GPU
        # is always device 0.
        return torch.device("cuda")
    raise ValueError("device must be 'cpu', 'cuda' or 'cuda:0'")


def pair_semantics(arm: str) -> str | None:
    if arm == "PAIR_TRUE":
        return "true"
    if arm == "PAIR_NULL":
        return "null"
    return None


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    *,
    device: torch.device,
    pairs: Mapping[str, Any] | None = None,
    batch_size: int = 256,
    limit_batches: int | None = None,
    return_delta: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Label-blind prediction over a cache in cache order (normalised units)."""
    was_training = model.training
    model.eval()
    n = int(cache["n_molecules"])
    preds: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    n_batches = 0
    for start in range(0, n, int(batch_size)):
        if limit_batches is not None and n_batches >= int(limit_batches):
            break
        rows = list(range(start, min(start + int(batch_size), n)))
        batch = P.make_batch(cache, rows, pairs)
        batch = C.V2D.to_device(batch, device)
        if return_delta:
            y, aux = model(batch, return_aux=True)
            deltas.append(aux["delta"].detach().float().cpu().numpy().reshape(-1))
        else:
            y = model(batch)
        preds.append(y.detach().float().cpu().numpy().reshape(-1))
        n_batches += 1
    if was_training:
        model.train()
    pred = np.concatenate(preds).astype(np.float64)
    delta = np.concatenate(deltas).astype(np.float64) if return_delta else None
    return pred, delta


def r2_score(pred: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def metric_block(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "rmse": C.V2D.rmse(pred, y),
        "mae": C.V2D.mae(pred, y),
        "r2": r2_score(pred, y),
        "n": int(np.asarray(y).size),
    }


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def proto_usage(model: torch.nn.Module, cache: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Top-1 / top-4 prototype usage over a cache (eval mode, no dropout)."""
    module = model.f
    if not isinstance(module, M.ProtoReadout):
        raise TypeError("proto_usage requires a ProtoReadout head")
    was_training = model.training
    model.eval()
    top1 = torch.zeros(module.n_prototypes, dtype=torch.float64)
    top4 = torch.zeros(module.n_prototypes, dtype=torch.float64)
    n_atoms = 0
    with torch.no_grad():
        for start in range(0, int(cache["n_molecules"]), 256):
            rows = list(range(start, min(start + 256, int(cache["n_molecules"]))))
            batch = C.V2D.to_device(C.V2D.collate(cache, rows), device)
            _, _, h = model._encode(batch)
            weights, top_idx = module.route(h)
            if not bool(torch.all(torch.isfinite(weights))):
                raise RuntimeError("non-finite prototype routing weights")
            top1 += torch.bincount(top_idx[:, 0].cpu(), minlength=module.n_prototypes).double()
            top4 += torch.bincount(top_idx.reshape(-1).cpu(), minlength=module.n_prototypes).double()
            n_atoms += int(h.shape[0])
    if was_training:
        model.train()
    if n_atoms == 0:
        raise ValueError("empty cache")
    top1_share = (top1 / float(n_atoms)).numpy()
    top4_share = (top4 / float(max(module.topk * n_atoms, 1))).numpy()
    nonzero = top1_share[top1_share > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum()) if nonzero.size else 0.0
    return {
        "n_atoms": int(n_atoms),
        "n_prototypes": int(module.n_prototypes),
        "topk": int(module.topk),
        "tau": float(module.tau),
        "top1_counts": [int(x) for x in top1.tolist()],
        "top4_counts": [int(x) for x in top4.tolist()],
        "top1_share": [float(x) for x in top1_share.tolist()],
        "top4_share": [float(x) for x in top4_share.tolist()],
        "n_prototypes_used_as_top1": int((top1 > 0).sum()),
        "n_prototypes_entering_top4": int((top4 > 0).sum()),
        "n_prototypes_never_top1": int((top1 == 0).sum()),
        "n_prototypes_never_top4": int((top4 == 0).sum()),
        "effective_prototypes_perplexity": float(math.exp(entropy)),
        "top1_share_max": float(top1_share.max()),
        "top1_share_min": float(top1_share.min()),
    }


def proto_key_shuffle_diagnostic(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int = 0,
) -> dict[str, Any]:
    """Shuffle prototype *keys* only (experts untouched) and re-predict Dev."""
    module = model.f
    if not isinstance(module, M.ProtoReadout):
        raise TypeError("proto_key_shuffle_diagnostic requires a ProtoReadout head")
    baseline, _ = predict(model, cache, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(module.n_prototypes, generator=generator)
    saved = module.prototypes.detach().clone()
    experts_saved = module.experts.detach().clone()
    try:
        with torch.no_grad():
            module.prototypes.copy_(saved[perm].to(module.prototypes.device))
        shuffled, _ = predict(model, cache, device=device)
    finally:
        with torch.no_grad():
            module.prototypes.copy_(saved)
    unchanged_experts = bool(torch.equal(experts_saved, module.experts.detach()))
    diff = np.abs(shuffled - baseline)
    return {
        "scope": "dev predictions, prototype keys D permuted, experts W/b untouched",
        "seed": int(seed),
        "key_permutation": [int(x) for x in perm.tolist()],
        "experts_untouched": unchanged_experts,
        "mean_abs_change_normalised": float(diff.mean()),
        "max_abs_change_normalised": float(diff.max()),
        "n_predictions_changed": int((diff > 0).sum()),
        "n_predictions": int(diff.size),
        "note": "diagnostic only; not used for model selection",
    }


def pair_diagnostics(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    pairs: Mapping[str, Any],
    *,
    device: torch.device,
    target_scale: float,
) -> dict[str, Any]:
    pred, delta = predict(model, cache, device=device, pairs=pairs, return_delta=True)
    delta = np.asarray(delta, dtype=np.float64)
    changed = np.abs(delta) > 0
    return {
        "delta_units": "raw logD (normalised output multiplied by the fold Train target scale)",
        "delta_mean_abs": float(np.abs(delta).mean() * target_scale),
        "delta_max_abs": float(np.abs(delta).max() * target_scale),
        "delta_mean_signed": float(delta.mean() * target_scale),
        "fraction_graphs_with_nonzero_delta": float(changed.mean()),
        "n_graphs": int(delta.size),
        "prediction_mean_abs_normalised": float(np.abs(pred).mean()),
    }


def gradient_probe(
    model: torch.nn.Module,
    view: Any,
    pairs: Mapping[str, Any] | None,
    *,
    device: torch.device,
    prefixes: Sequence[str],
    n_batches: int = 2,
    batch_size: int = C.BATCH_SIZE,
) -> dict[str, Any]:
    """One forward/backward pass pair; report per-tensor gradient health."""
    was_training = model.training
    model.train()
    params = {
        name: p
        for name, p in model.named_parameters()
        if any(name.startswith(pref) for pref in prefixes)
    }
    if not params:
        raise ValueError(f"no parameters match {list(prefixes)}")
    for p in model.parameters():
        p.grad = None
    total_abs = {name: 0.0 for name in params}
    total_sq = {name: 0.0 for name in params}
    seen = 0
    rng = np.random.default_rng(0)
    n = int(view.train_cache["n_molecules"])
    for _ in range(int(n_batches)):
        rows = sorted(int(x) for x in rng.choice(n, size=min(batch_size, n), replace=False).tolist())
        batch = P.make_batch(view.train_cache, rows, pairs)
        batch = C.V2D.to_device(batch, device)
        target = torch.as_tensor(
            (view.y_train[rows] - float(view.target_norm["mean"])) / float(view.target_norm["scale"]),
            dtype=torch.float32,
            device=device,
        )
        pred = model(batch)
        loss = F.mse_loss(pred.view(-1), target.view(-1))
        if not math.isfinite(float(loss)):
            raise RuntimeError("non-finite loss in gradient probe")
        loss.backward()
        for name, p in params.items():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            total_abs[name] += float(g.abs().sum())
            total_sq[name] += float(g.pow(2).sum())
        seen += 1
    out = {"n_batches": seen, "batch_size": batch_size, "parameters": {}}
    all_finite = True
    all_nonzero = True
    for name in params:
        grad = params[name].grad
        finite = grad is not None and bool(torch.isfinite(grad).all())
        nonzero = bool(total_abs[name] > 0.0)
        out["parameters"][name] = {
            "grad_l1": float(total_abs[name]),
            "grad_l2": float(math.sqrt(total_sq[name])),
            "finite": bool(finite),
            "nonzero": bool(nonzero),
            "shape": list(params[name].shape),
        }
        all_finite = all_finite and finite
        all_nonzero = all_nonzero and nonzero
    out["all_finite"] = bool(all_finite)
    out["all_nonzero"] = bool(all_nonzero)
    for p in model.parameters():
        p.grad = None
    if was_training:
        model.train()
    else:
        model.eval()
    return out


def zero_residual_check(
    model: torch.nn.Module, cache: Mapping[str, Any], pairs: Mapping[str, Any], *, device: torch.device
) -> dict[str, Any]:
    """Zero the residual head of a trained PAIR model; compare with a pure BASE."""
    if not isinstance(model, M.PairLCRC):
        raise TypeError("zero_residual_check requires a PairLCRC model")
    spec = C.V2C.load_frozen_spec()
    base = C.prep_model.LCRC(spec, _norm_from_buffers(model), dict(model.cfg)).to(device)
    state = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith("pair_")}
    base.load_state_dict(state, strict=True)
    base.eval()

    zeroed = copy.deepcopy(model).to(device)
    last = zeroed.pair_head[2]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    zeroed.eval()

    pred_base, _ = predict(base, cache, device=device)
    pred_zero, _ = predict(zeroed, cache, device=device, pairs=pairs)
    pred_pair, _ = predict(model, cache, device=device, pairs=pairs)
    return {
        "max_abs_diff_zeroed_vs_base": float(np.abs(pred_zero - pred_base).max()),
        "max_abs_diff_zeroed_vs_pair": float(np.abs(pred_zero - pred_pair).max()),
        "pair_residual_max_abs": float(np.abs(pred_pair - pred_base).max()),
    }


def _norm_from_buffers(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "global_mean": model.global_mean.detach().cpu().numpy().tolist(),
        "global_scale": model.global_scale.detach().cpu().numpy().tolist(),
        "scalar_mean": model.scalar_mean.detach().cpu().numpy().tolist(),
        "scalar_scale": model.scalar_scale.detach().cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# training job
# ---------------------------------------------------------------------------


@dataclass
class PilotJob:
    arm: str
    seed: int = C.SEED
    device: str = "cpu"
    pool: str = "local"
    out_root: Path = field(default_factory=lambda: C.RUNS)
    epochs: int = C.EPOCHS
    smoke: bool = False
    smoke_batches: int | None = None
    dropout: float = C.DROPOUT
    write: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.arm not in C.ARMS:
            raise ValueError(f"arm must be one of {C.ARMS}")
        if int(self.seed) != C.SEED:
            raise ValueError(f"seed must be {C.SEED}")
        if self.smoke:
            if self.epochs < 1 or self.epochs > 5:
                raise ValueError("smoke epochs must be in [1, 5]")
        else:
            if int(self.epochs) != C.EPOCHS:
                raise ValueError(f"formal training must use {C.EPOCHS} epochs")
            if self.smoke_batches is not None:
                raise ValueError("formal training must evaluate the full folds")
            if float(self.dropout) != C.DROPOUT:
                raise ValueError("formal training must use the R3 dropout 0.10")
            if str(self.pool) != C.ARM_POOL[self.arm]:
                raise ValueError(f"arm {self.arm} must run on pool {C.ARM_POOL[self.arm]}, got {self.pool}")

    def key(self) -> str:
        return f"{self.arm}_s{self.seed}_fold{C.FOLD}"


def train_job(job: PilotJob) -> dict[str, Any]:
    job.validate()
    device = resolve_device(job.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    view = C.load_fold_view()
    spec = C.V2C.load_frozen_spec()
    norm = view.train_norm
    target_norm = view.target_norm
    semantics = pair_semantics(job.arm)

    t_pairs = time.perf_counter()
    train_pairs = P.build_pair_cache(view.train_cache, semantics=semantics) if semantics else None
    dev_pairs = P.build_pair_cache(view.dev_cache, semantics=semantics) if semantics else None
    pairs_seconds = time.perf_counter() - t_pairs

    model, meta = M.build_arm(
        job.arm, spec, norm, job.seed, device, dropout=float(job.dropout)
    )
    init_hash = C.state_dict_fingerprint(model.state_dict())
    n_params = M.parameter_count(model)
    breakdown = M.module_breakdown(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=C.PEAK_LR,
        weight_decay=C.WEIGHT_DECAY,
        betas=C.ADAM_BETAS,
        eps=C.ADAM_EPS,
    )
    n_train = int(view.train_cache["n_molecules"])
    steps_per_epoch = C.steps_per_epoch(n_train)
    total_updates = steps_per_epoch * int(job.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=C.MIN_LR)
    generator = torch.Generator().manual_seed(int(job.seed) + C.SHUFFLE_OFFSET)

    soup_last = min(C.SOUP_LAST, int(job.epochs))
    snapshots: list[dict[str, torch.Tensor]] = []
    member_hashes: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    grad_norm_max = 0.0
    clip_events = 0
    started = time.perf_counter()
    cuda_step_seconds = 0.0
    for epoch in range(1, int(job.epochs) + 1):
        model.train()
        t_epoch = time.perf_counter()
        epoch_abs = 0.0
        epoch_sq = 0.0
        seen = 0
        lr_last = None
        perm = torch.randperm(n_train, generator=generator).tolist()
        for start in range(0, len(perm), C.BATCH_SIZE):
            rows = perm[start : start + C.BATCH_SIZE]
            batch = P.make_batch(view.train_cache, rows, train_pairs)
            batch = C.V2D.to_device(batch, device)
            y = torch.as_tensor(
                (view.y_train[rows] - float(target_norm["mean"])) / float(target_norm["scale"]),
                dtype=torch.float32,
                device=device,
            )
            step_start = time.perf_counter() if device.type == "cuda" else 0.0
            pred = model(batch)
            loss = F.mse_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP))
            if not math.isfinite(gnorm) or not math.isfinite(float(loss)):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            grad_norm_max = max(grad_norm_max, gnorm)
            if gnorm > C.GRAD_CLIP:
                clip_events += 1
            lr_last = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                cuda_step_seconds += time.perf_counter() - step_start
            error = (pred.view(-1) - y.view(-1)).detach()
            epoch_abs += float(error.abs().sum())
            epoch_sq += float((error * error).sum())
            seen += int(error.numel())
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps_per_epoch,
                "train_mae_normalised": epoch_abs / max(seen, 1),
                "train_rmse_normalised": math.sqrt(epoch_sq / max(seen, 1)),
                "lr_last": lr_last,
                "seconds": time.perf_counter() - t_epoch,
            }
        )
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        member_hashes.append(
            {
                "epoch": epoch,
                "state_hash": C.state_dict_fingerprint(snapshots[-1]),
            }
        )
        if len(snapshots) > soup_last:
            snapshots.pop(0)

    train_seconds = time.perf_counter() - started
    members = list(range(int(job.epochs) - len(snapshots) + 1, int(job.epochs) + 1))
    expected_members = list(range(max(1, int(job.epochs) - soup_last + 1), int(job.epochs) + 1))
    if members != expected_members:
        raise RuntimeError("soup members are not the last epochs")
    soup = _average_states(snapshots)
    soup_hash = C.state_dict_fingerprint(soup)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})

    t_eval = time.perf_counter()
    train_pred_norm, _ = predict(
        model, view.train_cache, device=device, pairs=train_pairs, limit_batches=job.smoke_batches
    )
    dev_pred_norm, _ = predict(
        model, view.dev_cache, device=device, pairs=dev_pairs, limit_batches=job.smoke_batches
    )
    train_pred = C.V2D.denormalise(train_pred_norm, target_norm)
    dev_pred = C.V2D.denormalise(dev_pred_norm, target_norm)
    train_n = int(train_pred.size)
    dev_n = int(dev_pred.size)
    train_metrics = metric_block(train_pred, view.y_train[:train_n])
    dev_metrics = metric_block(dev_pred, view.y_dev[:dev_n])
    eval_seconds = time.perf_counter() - t_eval

    diagnostics: dict[str, Any] = {}
    if semantics is not None and dev_pairs is not None:
        diagnostics["pair"] = {
            "semantics": semantics,
            "delta_train": pair_diagnostics(
                model,
                view.train_cache,
                train_pairs,
                device=device,
                target_scale=float(target_norm["scale"]),
            ),
            "delta_dev": pair_diagnostics(
                model,
                view.dev_cache,
                dev_pairs,
                device=device,
                target_scale=float(target_norm["scale"]),
            ),
            "coverage_train": P.coverage(train_pairs),
            "coverage_dev": P.coverage(dev_pairs),
            "pair_content_vs_true": (
                P.same_pair_content(
                    P.build_pair_cache(view.dev_cache, semantics="true"), dev_pairs
                )
                if semantics == "null"
                else None
            ),
            "zero_residual": zero_residual_check(model, view.dev_cache, dev_pairs, device=device),
            "gradients": gradient_probe(
                model, view, train_pairs, device=device, prefixes=("pair_z.", "pair_head.")
            ),
        }
    if job.arm in C.PROTO_ARMS:
        diagnostics["proto"] = {
            "model_name": C.PROTO_NAME,
            "train": proto_usage(model, view.train_cache, device),
            "dev": proto_usage(model, view.dev_cache, device),
            "key_shuffle": proto_key_shuffle_diagnostic(model, view.dev_cache, device=device),
            "gradients": gradient_probe(
                model, view, None, device=device, prefixes=("f.prototypes", "f.experts", "f.bias")
            ),
        }

    wall_seconds = time.perf_counter() - PROCESS_T0
    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "key": job.key(),
        "arm": job.arm,
        "smoke": bool(job.smoke),
        "seed": int(job.seed),
        "fold": C.FOLD,
        "pool": str(job.pool),
        "device": str(device),
        "device_note": "rr sets CUDA_VISIBLE_DEVICES per pool; the pool GPU is cuda:0 in-process",
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.config_hash(C.load_config()),
        "data_view": view.audit(),
        "recipe": {
            "optimizer": "AdamW",
            "adam_betas": list(C.ADAM_BETAS),
            "adam_eps": C.ADAM_EPS,
            "peak_lr": C.PEAK_LR,
            "min_lr": C.MIN_LR,
            "scheduler": "per_step_cosine",
            "weight_decay": C.WEIGHT_DECAY,
            "batch_size": C.BATCH_SIZE,
            "epochs": int(job.epochs),
            "steps_per_epoch": steps_per_epoch,
            "total_updates": total_updates,
            "grad_clip": C.GRAD_CLIP,
            "dropout": float(job.dropout),
            "dropout_position": "final atom-level local representation h (v2 DropoutAtLocal on model.local)",
            "dropout_hook_calls": int(meta["hook"].calls),
            "loss": "mse_on_fold_train_standardised_target",
            "soup_last": soup_last,
            "soup_members": members,
            "soup_member_state_hashes": [row for row in member_hashes if row["epoch"] in members],
            "early_stopping": False,
            "dev_checkpoint_selection": False,
            "ssl": False,
            "pretraining": False,
            "extra_descriptors": False,
            "external_data": False,
        },
        "model": {
            "arm": job.arm,
            "parameters": n_params,
            "parameters_breakdown": breakdown,
            "init_state_hash": init_hash,
            "soup_state_hash": soup_hash,
            "spec": spec.name,
            "width": int(model.cfg["width"]),
            "prototypes": (
                {
                    "n_prototypes": C.N_PROTOTYPES,
                    "topk": C.PROTO_TOPK,
                    "tau": C.PROTO_TAU,
                    "auxiliary_loss": None,
                    "mlp_residual_bypass": False,
                }
                if job.arm in C.PROTO_ARMS
                else None
            ),
            "pair": (
                {
                    "distances": list(C.PAIR_DISTANCES),
                    "z_dim": C.PAIR_Z_DIM,
                    "hidden": C.PAIR_HIDDEN,
                    "aggregation": "mean over atom pairs per distance bin",
                    "head": "Linear(48,32) -> SiLU -> Linear(32,1)",
                    "last_layer_init": "weight ~ Normal(0,1e-3), bias = 0",
                    "semantics": semantics,
                    "null_salt": C.PAIR_NULL_SALT if semantics == "null" else None,
                }
                if semantics is not None
                else None
            ),
        },
        "metrics": {
            "train_rmse_raw": train_metrics["rmse"],
            "train_mae_raw": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "dev_rmse_raw": dev_metrics["rmse"],
            "dev_mae_raw": dev_metrics["mae"],
            "dev_r2": dev_metrics["r2"],
            "generalisation_gap": dev_metrics["rmse"] - train_metrics["rmse"],
            "train_n": train_n,
            "dev_n": dev_n,
            "metric_note": "RMSE/MAE in original logD units (denormalised with the fold Train target std)",
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": 0,
        },
        "diagnostics": diagnostics,
        "timing": {
            "process_wall_seconds": float(wall_seconds),
            "train_loop_seconds": float(train_seconds),
            "eval_seconds": float(eval_seconds),
            "pair_precompute_seconds": float(pairs_seconds),
            "cuda_step_seconds": float(cuda_step_seconds),
        },
        "pair_precompute": (
            {
                "train_pairs": int(train_pairs["n_pairs_total"]),
                "dev_pairs": int(dev_pairs["n_pairs_total"]),
            }
            if semantics is not None
            else None
        ),
        "labels_scope": {
            "fit2352_fold2_only": True,
            "monitor_labels_read": False,
            "reserved_labels_read": False,
            "official_valid_test_read": False,
            "random42_splits_read": False,
        },
    }

    if job.extra:
        payload.update(job.extra)
    if job.write:
        out_dir = Path(job.out_root) / job.key()
        C.ensure_dir(out_dir)
        torch.save(soup, out_dir / "soup.pt")
        _write_predictions(
            out_dir,
            split="dev",
            table=C.dev_row_table(view),
            y=view.y_dev[:dev_n],
            pred=dev_pred,
            cache_rows=view.dev_positions[:dev_n],
        )
        _write_predictions(
            out_dir,
            split="train",
            table=C.train_row_table(view),
            y=view.y_train[:train_n],
            pred=train_pred,
            cache_rows=view.train_positions[:train_n],
        )
        _write_curve(out_dir / "curve.csv", curve)
        payload["artifacts"] = {
            "soup": C.file_record(out_dir / "soup.pt"),
            "predictions_dev": C.file_record(out_dir / "predictions_dev.npz"),
            "predictions_dev_csv": C.file_record(out_dir / "predictions_dev.csv"),
            "predictions_train": C.file_record(out_dir / "predictions_train.npz"),
            "curve": C.file_record(out_dir / "curve.csv"),
        }
        C.write_json(out_dir / "train.json", payload)
        if diagnostics:
            C.write_json(out_dir / "diagnostics.json", diagnostics)
        C.log(
            f"[{job.key()}] arm={job.arm} pool={job.pool} train={train_metrics['rmse']:.5f} "
            f"dev={dev_metrics['rmse']:.5f} params={n_params} wall={wall_seconds:.1f}s"
        )
    return payload


def _write_predictions(
    out_dir: Path,
    *,
    split: str,
    table: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    pred: np.ndarray,
    cache_rows: Sequence[int],
) -> None:
    table = list(table)[: int(np.asarray(pred).size)]
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if not (len(table) == y.size == pred.size):
        raise ValueError("prediction table length mismatch")
    arrays = {
        "global_row_id": np.asarray([row["global_row_id"] for row in table], dtype=np.int64),
        "fit_cache_position": np.asarray([row["fit_cache_position"] for row in table], dtype=np.int64),
        "cache_row": np.asarray(list(cache_rows)[: pred.size], dtype=np.int64),
        "y_true": y,
        "pred": pred,
        "residual": pred - y,
        "abs_error": np.abs(pred - y),
        "group_key_sha256": np.asarray([row["group_key_sha256"] for row in table]),
        "scaffold": np.asarray([row["scaffold"] for row in table]),
    }
    np.savez(out_dir / f"predictions_{split}.npz", **arrays)
    if split == "dev":
        with open(out_dir / "predictions_dev.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["global_row_id", "fit_cache_position", "cache_row", "scaffold", "y_true", "pred", "residual"]
            )
            for i in range(y.size):
                writer.writerow(
                    [
                        int(arrays["global_row_id"][i]),
                        int(arrays["fit_cache_position"][i]),
                        int(arrays["cache_row"][i]),
                        str(arrays["scaffold"][i]),
                        f"{arrays['y_true'][i]:.6f}",
                        f"{arrays['pred'][i]:.6f}",
                        f"{arrays['residual'][i]:.6f}",
                    ]
                )


def _write_curve(path: Path, curve: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["epoch", "updates", "train_mae_normalised", "train_rmse_normalised", "lr_last", "seconds"]
        )
        for row in curve:
            writer.writerow(
                [
                    row["epoch"],
                    row["updates"],
                    f"{row['train_mae_normalised']:.8f}",
                    f"{row['train_rmse_normalised']:.8f}",
                    f"{row['lr_last']:.10f}",
                    f"{row['seconds']:.4f}",
                ]
            )


__all__ = [
    "PROCESS_T0",
    "PilotJob",
    "train_job",
    "predict",
    "metric_block",
    "proto_usage",
    "proto_key_shuffle_diagnostic",
    "pair_diagnostics",
    "gradient_probe",
    "zero_residual_check",
    "resolve_device",
    "pair_semantics",
]
