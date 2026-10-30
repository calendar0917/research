"""Fold-aware supervised training for the v2 recipes.

The training loop is a v2 rewrite of the frozen v1 loop.  It keeps the same
index discipline (shuffle position -> cache position -> molecule id), the same
per-step cosine schedule and last-5 epoch soup, and adds only:

* fold-specific train/dev caches and normalisation;
* the fixed recipe grid R0--R3;
* dropout at the final atom-level local representation;
* optional SSL initialisation of atom/bond/local;
* optional 13-dimensional physicochemical global augmentation.

No Monitor label is read here.  No dev-based early stopping or checkpoint
selection is implemented.
"""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import v2_common as C
import v2_data as D
import v2_models as M


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return math.sqrt(total)


def _append_global(batch: Mapping[str, Any], extra: np.ndarray | None, rows) -> dict[str, Any]:
    if extra is None:
        return dict(batch)
    out = dict(batch)
    out["global"] = torch.cat(
        [
            out["global"],
            torch.as_tensor(np.asarray(extra)[np.asarray(rows, dtype=np.int64)], dtype=torch.float32),
        ],
        dim=1,
    )
    return out


def _average_states(states: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("empty soup")
    keys = list(states[0].keys())
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        out[key] = torch.stack([s[key].detach().float().cpu() for s in states], dim=0).mean(dim=0)
    return out


def load_ssl_encoder_state(model: torch.nn.Module, checkpoint: Path | str) -> dict[str, Any]:
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    module_state = state.get("module_state", state)
    allowed = ("atom_encoder.", "bond_encoder.", "local.")
    encoder = {k: v for k, v in module_state.items() if k.startswith(allowed)}
    if not encoder:
        raise ValueError("SSL checkpoint contains no atom/bond/local parameters")
    own = model.state_dict()
    for key, value in encoder.items():
        if key not in own or tuple(own[key].shape) != tuple(value.shape):
            raise ValueError(f"SSL encoder key {key!r} does not fit the supervised model")
    model.load_state_dict({**own, **encoder}, strict=True)
    return {
        "checkpoint": C.track_rel(checkpoint),
        "checkpoint_sha256": C.sha256_file(checkpoint),
        "loaded_keys": sorted(encoder),
        "n_loaded": len(encoder),
        "f_rho_from_scratch": True,
    }


@dataclass
class TrainJob:
    key: str
    arm: str
    fold: int | str
    recipe: str
    seed: int
    variant: str = "scratch"
    ssl_checkpoint: Path | str | None = None
    device: str = "cpu"
    out_root: Path = field(default_factory=lambda: C.RUNS)
    # Only for CPU smoke / gates; formal jobs must leave these None.
    smoke_train_limit: int | None = None
    smoke_epochs: int | None = None
    global_extra_train: np.ndarray | None = None
    global_extra_dev: np.ndarray | None = None
    global_extra_audit: dict[str, Any] | None = None
    spec_override: Any | None = None
    norm_override: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if self.arm not in C.ARMS:
            raise ValueError(f"arm must be one of {C.ARMS}")
        if self.recipe not in C.RECIPES:
            raise ValueError(f"recipe must be one of {tuple(C.RECIPES)}")
        if self.seed not in C.SEEDS:
            raise ValueError(f"seed must be one of {C.SEEDS}")
        if self.fold not in (*C.FOLDS, "full"):
            raise ValueError("fold must be 0, 1, 2 or 'full'")
        if self.variant not in ("scratch", "ssl"):
            raise ValueError("variant must be scratch or ssl")
        if self.variant == "ssl" and self.ssl_checkpoint is None:
            raise ValueError("ssl variant requires an ssl_checkpoint")
        if self.variant == "ssl" and self.arm != "A":
            raise ValueError("SSL downstream is only defined for arm A")
        if self.smoke_train_limit is None and self.smoke_epochs is None:
            if self.fold == "full":
                pass
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

    def recipe_cfg(self) -> dict[str, float]:
        return dict(C.RECIPES[self.recipe])

    def resolved_fold(self) -> int | str:
        return self.fold


def _build_view(fold: int | str) -> D.DataView:
    data = C.load_fit_cache()
    cache = data["cache"]
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    if fold == "full":
        return D.build_full_view(cache=cache, y_fit=y)
    import v2_folds as F

    folds = F.load_folds()
    return D.build_fold_view(int(fold), cache=cache, y_fit=y, folds=folds)


def _smoke_view(view: D.DataView, limit: int) -> D.DataView:
    """Prefix subset of the training cache; dev is left intact."""
    n = min(int(limit), int(view.train_cache["n_molecules"]))
    train_cache = D.select_cache(view.train_cache, list(range(n)), split_label=f"{view.name}_smoke")
    return D.DataView(
        name=f"{view.name}_smoke{n}",
        train_cache=train_cache,
        dev_cache=view.dev_cache,
        y_train=view.y_train[:n],
        y_dev=view.y_dev,
        train_norm=C.prep_data.fit_norm(train_cache, C.load_frozen_spec(), source=f"{view.name}_smoke_structures"),
        target_norm=view.target_norm,
        train_positions=tuple(range(n)),
        dev_positions=view.dev_positions,
        fold=view.fold,
    )


def train_job(job: TrainJob) -> dict[str, Any]:
    job.validate()
    recipe = job.recipe_cfg()
    view = _build_view(job.fold)
    if job.smoke_train_limit is not None:
        view = _smoke_view(view, int(job.smoke_train_limit))
    epochs = int(job.smoke_epochs) if job.smoke_epochs is not None else C.EPOCHS
    if job.smoke_epochs is None and epochs != C.EPOCHS:
        raise ValueError("formal training must use 360 epochs")
    if job.smoke_train_limit is None and view.train_cache["n_molecules"] != C.SPLIT_SIZES["fit"] and job.fold == "full":
        raise ValueError("full training must use Fit2352")
    if job.smoke_train_limit is None and job.fold != "full" and view.train_cache["n_molecules"] != 2 * C.SPLIT_SIZES["fit"] // 3:
        raise ValueError("fold training must use two full folds")

    spec = job.spec_override if job.spec_override is not None else C.load_frozen_spec()
    norm = job.norm_override if job.norm_override is not None else view.train_norm
    target_norm = view.target_norm
    device = torch.device(job.device)

    model, model_meta = M.build_model(job.arm, spec, norm, job.seed, device, dropout=float(recipe["dropout"]))
    init_hash = M.model_state_hash(model)
    f_rho_init = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("f.", "rho."))}
    ssl_info = None
    if job.variant == "ssl":
        ssl_info = load_ssl_encoder_state(model, job.ssl_checkpoint)
        f_rho_after = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("f.", "rho."))}
        for key in f_rho_init:
            if not torch.equal(f_rho_init[key], f_rho_after[key]):
                raise RuntimeError(f"SSL load changed f/rho parameter {key}")
        ssl_info["f_rho_unchanged_by_ssl_load"] = True
        if job.arm == "A":
            scratch_model, _ = M.build_model("A", spec, norm, job.seed, device, dropout=float(recipe["dropout"]))
            scratch_f_rho = {k: v.detach().clone() for k, v in scratch_model.state_dict().items() if k.startswith(("f.", "rho."))}
            ssl_info["f_rho_matches_scratch_init"] = all(
                torch.equal(scratch_f_rho[k], f_rho_init[k]) for k in f_rho_init
            )
            if not ssl_info["f_rho_matches_scratch_init"]:
                raise RuntimeError("SSL downstream f/rho do not match the scratch initialization")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(recipe["peak_lr"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    n_train = int(view.train_cache["n_molecules"])
    steps_per_epoch = math.ceil(n_train / C.BATCH_SIZE)
    total_updates = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=C.LR_MIN)
    generator = torch.Generator().manual_seed(int(job.seed) + C.SHUFFLE_OFFSET)

    snapshots: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    grad_norm_max = 0.0
    clip_events = 0
    nonfinite_events = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t_epoch = time.perf_counter()
        epoch_abs = 0.0
        epoch_sq = 0.0
        seen = 0
        lr_last = None
        perm = torch.randperm(n_train, generator=generator).tolist()
        for start in range(0, len(perm), C.BATCH_SIZE):
            rows = perm[start : start + C.BATCH_SIZE]
            batch = _append_global(D.collate(view.train_cache, rows), job.global_extra_train, rows)
            batch = D.to_device(batch, device)
            y = torch.as_tensor(
                (view.y_train[rows] - float(target_norm["mean"])) / float(target_norm["scale"]),
                dtype=torch.float32,
                device=device,
            )
            pred = model(batch)
            loss = F.mse_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP))
            if not math.isfinite(gnorm) or not math.isfinite(float(loss)):
                nonfinite_events += 1
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            grad_norm_max = max(grad_norm_max, gnorm)
            if gnorm > C.GRAD_CLIP:
                clip_events += 1
            lr_last = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            error = (pred.view(-1) - y.view(-1)).detach()
            epoch_abs += float(error.abs().sum())
            epoch_sq += float((error * error).sum())
            seen += int(error.numel())
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps_per_epoch,
                "train_mae_normalised": epoch_abs / max(seen, 1),
                "train_mse_normalised": epoch_sq / max(seen, 1),
                "lr_last": lr_last,
                "seconds": time.perf_counter() - t_epoch,
            }
        )
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if len(snapshots) > C.SOUP_LAST:
            snapshots.pop(0)

    wall = time.perf_counter() - started
    members = list(range(epochs - len(snapshots) + 1, epochs + 1))
    expected_members = list(range(max(1, epochs - C.SOUP_LAST + 1), epochs + 1))
    if members != expected_members:
        raise RuntimeError("soup members are not the last five epochs")
    soup = _average_states(snapshots)
    soup_hash = M.state_dict_fingerprint(soup)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})

    train_pred = D.denormalise(
        D.predict_view(model, view.train_cache, device=device, global_extra=job.global_extra_train), target_norm
    )
    train_rmse = D.rmse(train_pred, view.y_train)
    train_mae = D.mae(train_pred, view.y_train)
    if job.fold == "full" and job.smoke_train_limit is None:
        dev_rmse, dev_mae, dev_is_train = train_rmse, train_mae, True
    else:
        dev_pred = D.denormalise(
            D.predict_view(model, view.dev_cache, device=device, global_extra=job.global_extra_dev), target_norm
        )
        dev_rmse = D.rmse(dev_pred, view.y_dev)
        dev_mae = D.mae(dev_pred, view.y_dev)
        dev_is_train = False

    out_dir = Path(job.out_root) / job.key
    C.ensure_dir(out_dir)
    torch.save(soup, out_dir / "soup.pt")
    if job.fold == "full" and job.smoke_train_limit is None:
        pred_rows = np.asarray(view.train_positions, dtype=np.int64)
        pred_values = np.asarray(train_pred, dtype=np.float64)
        pred_split = "fit"
    else:
        pred_rows = np.asarray(view.dev_positions, dtype=np.int64)
        pred_values = np.asarray(dev_pred, dtype=np.float64)
        pred_split = "dev"
    pred_path = out_dir / "predictions.npz"
    np.savez(pred_path, rows=pred_rows, pred=pred_values)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "key": job.key,
        "arm": job.arm,
        "fold": job.fold,
        "recipe": job.recipe,
        "recipe_config": recipe,
        "seed": int(job.seed),
        "variant": job.variant,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.config_hash(C.load_config()),
        "data_view": view.audit(),
        "recipe_audit": {
            "optimizer": "AdamW",
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
            "peak_lr": float(recipe["peak_lr"]),
            "min_lr": C.LR_MIN,
            "weight_decay": float(recipe["weight_decay"]),
            "dropout": float(recipe["dropout"]),
            "dropout_position": "final atom-level local representation h, before graph readout",
            "scheduler": "per_step_cosine",
            "batch_size": C.BATCH_SIZE,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_updates": total_updates,
            "grad_clip": C.GRAD_CLIP,
            "loss": "mse_on_fold_standardised_target",
            "soup_last": C.SOUP_LAST,
            "soup_members": members,
            "early_stopping": False,
            "dev_checkpoint_selection": False,
        },
        "model": {
            "arm": job.arm,
            "parameters_nominal": M.parameter_count(model),
            "parameters_breakdown": M.module_breakdown(model),
            "effective_parameters": (
                M.effective_parameters_e(model) if job.arm == "E" else M.parameter_count(model)
            ),
            "capacity_audit": M.e_capacity_audit(model, spec) if job.arm == "E" else None,
            "init_state_hash": init_hash,
            "soup_state_hash": soup_hash,
            "spec_global_dim": int(spec.global_dim),
            "dropout_hook_calls": int(model_meta["dropout_hook"].calls),
        },
        "ssl": ssl_info,
        "global_extra": job.global_extra_audit,
        "metrics": {
            "train_rmse_raw": float(train_rmse),
            "train_mae_raw": float(train_mae),
            "dev_rmse_raw": float(dev_rmse),
            "dev_mae_raw": float(dev_mae),
            "dev_is_train": bool(dev_is_train),
            "generalisation_gap": float(dev_rmse - train_rmse),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": int(nonfinite_events),
        },
        "curve": curve,
        "wall_seconds": float(wall),
        "soup_file": C.track_rel(out_dir / "soup.pt"),
        "soup_file_sha256": C.sha256_file(out_dir / "soup.pt"),
        "predictions_file": C.track_rel(pred_path),
        "predictions_sha256": C.sha256_file(pred_path),
        "predictions_split": pred_split,
        "prediction_rows": [int(x) for x in pred_rows],
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(
        f"[{job.key}] arm={job.arm} fold={job.fold} {job.recipe} s{job.seed} {job.variant} "
        f"train={train_rmse:.5f} dev={dev_rmse:.5f} wall={wall:.1f}s"
    )
    return payload


__all__ = ["TrainJob", "train_job", "load_ssl_encoder_state"]
