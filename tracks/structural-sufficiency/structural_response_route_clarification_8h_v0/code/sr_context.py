"""Context builders and family adapters (one place that knows each family)."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import sr_common as C
import sr_data as SD
import sr_gine as SG
import sr_pairs as SP
import sr_resp as SR
import sr_train as ST
import sr_fgcr as SF

if str(C.FGCR_CODE) not in sys.path:
    sys.path.insert(0, str(C.FGCR_CODE))
import fgcr_inference as I  # noqa: E402


@dataclass
class Ctx:
    dataset: str
    bundle: Any
    audit: dict[str, Any]
    plan: Any
    row_to_plan: dict[int, int]
    plan_stats: dict[str, Any]
    d0: dict[str, Any] | None
    pairs: SP.PairIndex | None
    scaler: dict[str, float]
    split_rows: dict[str, np.ndarray]


def _plan(bundle, rows: Sequence[int], smax: int | None) -> tuple[Any, dict[int, int], dict[str, Any]]:
    graphs = SD.graph_pairs(bundle, rows)
    if smax is None:
        widths = [I.min_fill_width(n, e) for n, e in graphs]
        smax = int(max(widths)) + 1
    plan = I.build_corpus_plan(graphs, smax=int(smax))
    stats = I.plan_stats(plan)
    row_to_plan = {int(r): i for i, r in enumerate(rows)}
    return plan, row_to_plan, stats


def build_lipo(*, include_valid: bool, smax: int = 4) -> Ctx:
    bundle, audit = SD.load_bundle("ogbg-mollipo")
    train = np.asarray(bundle.train_idx, dtype=np.int64)
    valid = np.asarray(bundle.valid_idx, dtype=np.int64)
    rows = np.concatenate([train, valid]) if include_valid else train
    plan, row_to_plan, stats = _plan(bundle, rows, smax)
    d0 = SD.load_d0(bundle)
    pairs = SP.build_pair_index(bundle, d0)
    scaler = dict(bundle.scaler)
    return Ctx(dataset="ogbg-mollipo", bundle=bundle, audit=audit, plan=plan,
               row_to_plan=row_to_plan, plan_stats=stats, d0=d0, pairs=pairs,
               scaler=scaler,
               split_rows={"train": train, "valid": valid})


def build_esol(*, smax_ceiling: int = 9) -> Ctx:
    bundle, audit = SD.load_bundle("ogbg-molesol")
    train = np.asarray(bundle.train_idx, dtype=np.int64)
    valid = np.asarray(bundle.valid_idx, dtype=np.int64)
    rows = np.concatenate([train, valid])
    graphs = SD.graph_pairs(bundle, rows)
    widths = [I.min_fill_width(n, e) for n, e in graphs]
    needed = int(max(widths)) + 1
    if needed > int(smax_ceiling):
        raise RuntimeError(f"ESOL needs clique size {needed} > ceiling {smax_ceiling}")
    plan, row_to_plan, stats = _plan(bundle, rows, needed)
    if stats["width_max"] > 8:
        raise RuntimeError(f"ESOL induced width {stats['width_max']} > 8")
    return Ctx(dataset="ogbg-molesol", bundle=bundle, audit=audit, plan=plan,
               row_to_plan=row_to_plan, plan_stats=stats, d0=None, pairs=None,
               scaler=dict(bundle.scaler),
               split_rows={"train": train, "valid": valid})


# ---------------------------------------------------------------------------
# FGCR family
# ---------------------------------------------------------------------------


def fgcr_init_state(ctx: Ctx) -> dict[str, torch.Tensor]:
    model, info = SF.build_variant(ctx.bundle.atom_dims, ctx.bundle.bond_dims,
                                   h_mode="FULL", eta_mode="FULL", seed=0)
    if info["n_parameters"] != 100926:
        raise RuntimeError(f"FGCR parameter count {info['n_parameters']} != 100926")
    if info["init_hash"] != _fgcr_init_hash():
        raise RuntimeError("FGCR init hash does not match the historical seed-0 state")
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


_FGCR_INIT_HASH: dict[str, str] = {}


def _fgcr_init_hash() -> str:
    if "value" not in _FGCR_INIT_HASH:
        ckpt = torch.load(C.FGCR_CKPT, map_location="cpu", weights_only=False)
        # the historical checkpoint records its own init hash only implicitly;
        # rebuild from the frozen init path and compare against the checkpoint
        model, info = SF.build_variant(ckpt["atom_dims"], ckpt["bond_dims"],
                                       h_mode="FULL", eta_mode="FULL", seed=0)
        _FGCR_INIT_HASH["value"] = info["init_hash"]
        _FGCR_INIT_HASH["ckpt_atom_dims"] = list(ckpt["atom_dims"])
        _FGCR_INIT_HASH["ckpt_epoch"] = int(ckpt["epoch"])
    return _FGCR_INIT_HASH["value"]


def make_fgcr_arm(ctx: Ctx, *, h_mode: str, eta_mode: str,
                  init_state: Mapping[str, torch.Tensor], device: Any,
                  recipe: Mapping[str, Any] | None = None,
                  pair_rows: tuple[np.ndarray, np.ndarray] | None = None,
                  eval_labels_raw: np.ndarray | None = None,
                  meta: Mapping[str, Any] | None = None) -> ST.Arm:
    recipe = dict(recipe or C.FGCR_RECIPE)
    model, info = SF.build_variant(ctx.bundle.atom_dims, ctx.bundle.bond_dims,
                                   h_mode=h_mode, eta_mode=eta_mode,
                                   init_state=init_state, device=device)
    labels_std = ctx.bundle.labels_std
    plan, row_to_plan = ctx.plan, ctx.row_to_plan

    def base_loss(rows: np.ndarray) -> torch.Tensor:
        batch = SF.collate_rows(ctx.bundle, rows.tolist(), plan, row_to_plan, device)
        pred = model(batch)["yhat_std"]
        target = torch.as_tensor(labels_std[np.asarray(rows, dtype=np.int64)],
                                 dtype=torch.float32, device=device)
        return torch.mean((pred - target) ** 2)

    def aux_predict(rows: np.ndarray) -> torch.Tensor:
        batch = SF.collate_rows(ctx.bundle, rows.tolist(), plan, row_to_plan, device)
        return model(batch)["yhat_std"]

    def predict(rows: np.ndarray) -> np.ndarray:
        return SF.predict_std(model, ctx.bundle, plan, row_to_plan, rows,
                              batch_size=int(recipe["eval_batch_molecules"]),
                              device=device)

    extra = {"meta": dict(meta or {}), "arm_id": None, "h_mode": h_mode, "eta_mode": eta_mode,
             "model_info": info}
    if pair_rows is not None:
        extra["pair_row_a"], extra["pair_row_b"] = pair_rows
    if eval_labels_raw is not None:
        extra["eval_labels_raw"] = np.asarray(eval_labels_raw, dtype=np.float64)
    return ST.Arm(family="FGCR", model=model, device=device, base_loss=base_loss,
                  aux_predict=aux_predict, predict=predict,
                  lr_at=ST.lr_at_fgcr, recipe=recipe, extra=extra)


# ---------------------------------------------------------------------------
# RESP family
# ---------------------------------------------------------------------------


def make_resp_arm(ctx: Ctx, *, rows: Sequence[int], split_name: str,
                  scaler: Mapping[str, float], operator: str,
                  init_state: Mapping[str, torch.Tensor], device: Any,
                  recipe: Mapping[str, Any] | None = None,
                  pair_rows: tuple[np.ndarray, np.ndarray] | None = None,
                  eval_rows: Sequence[int] | None = None,
                  eval_labels_raw: np.ndarray | None = None,
                  fit_cache: Any = None,
                  meta: Mapping[str, Any] | None = None) -> tuple[ST.Arm, SR.RespBundle]:
    recipe = dict(recipe or C.RESP_RECIPE)
    bundle = fit_cache or SR.RespBundle(ctx.dataset, rows, scaler, split_name=split_name)
    model = SR.build_model(bundle.dims, init_state, operator, recipe).to(device)
    fit_subset = bundle.fit

    def base_loss(chunk: np.ndarray) -> torch.Tensor:
        return SR.base_loss(model, bundle, fit_subset, fit_subset.local(chunk), device)

    def aux_predict(chunk: np.ndarray) -> torch.Tensor:
        return SR.predict_tensor(model, bundle, fit_subset, fit_subset.local(chunk), device)

    def predict(chunk: np.ndarray) -> np.ndarray:
        return SR.predict_std(model, bundle, fit_subset, fit_subset.local(chunk), device,
                              batch_size=int(recipe["eval_batch_molecules"]))

    extra = {"meta": dict(meta or {}), "operator": operator,
             "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    if pair_rows is not None:
        extra["pair_row_a"], extra["pair_row_b"] = pair_rows
    if eval_labels_raw is not None:
        extra["eval_labels_raw"] = np.asarray(eval_labels_raw, dtype=np.float64)
    arm = ST.Arm(family="RESP", model=model, device=device, base_loss=base_loss,
                 aux_predict=aux_predict, predict=predict,
                 lr_at=ST.lr_at_linear_warmup, recipe=recipe, extra=extra)
    return arm, bundle


# ---------------------------------------------------------------------------
# GINE G0 family
# ---------------------------------------------------------------------------


def make_gine_arm(ctx: Ctx, *, init_state: Mapping[str, torch.Tensor], device: Any,
                  recipe: Mapping[str, Any] | None = None,
                  pair_rows: tuple[np.ndarray, np.ndarray] | None = None,
                  eval_labels_raw: np.ndarray | None = None,
                  meta: Mapping[str, Any] | None = None) -> ST.Arm:
    recipe = dict(recipe or C.GINE_RECIPE)
    model = SG.build_model(ctx.bundle.gine_categories, init_state).to(device)
    labels_std = ctx.bundle.labels_std

    def base_loss(rows: np.ndarray) -> torch.Tensor:
        return SG.base_loss(model, ctx.bundle, rows, device)

    def aux_predict(rows: np.ndarray) -> torch.Tensor:
        batch = {k: v.to(device) for k, v in SG.pack(ctx.bundle, np.asarray(rows).tolist()).items()}
        return model(batch)

    def predict(rows: np.ndarray) -> np.ndarray:
        return SG.predict_std(model, ctx.bundle, rows, device,
                              batch_size=int(recipe["eval_batch_molecules"]))

    extra = {"meta": dict(meta or {}), "n_parameters": model.parameter_count()}
    if pair_rows is not None:
        extra["pair_row_a"], extra["pair_row_b"] = pair_rows
    if eval_labels_raw is not None:
        extra["eval_labels_raw"] = np.asarray(eval_labels_raw, dtype=np.float64)
    return ST.Arm(family="GINE_G0", model=model, device=device, base_loss=base_loss,
                  aux_predict=aux_predict, predict=predict,
                  lr_at=ST.lr_at_linear_warmup, recipe=recipe, extra=extra)


# ---------------------------------------------------------------------------
# frozen per-epoch batch orders and the frozen pair schedule
# ---------------------------------------------------------------------------


def batch_order(n_rows: int, epochs: int, seed: int = C.TRAIN_SEED) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.stack([rng.permutation(int(n_rows)) for _ in range(int(epochs))]).astype(np.int64)


def load_or_build_schedule(ctx: Ctx, n_steps: int, path) -> np.ndarray:
    from pathlib import Path

    path = Path(path)
    if path.is_file():
        with np.load(path) as handle:
            saved = handle["schedule"]
            seed = int(handle["seed"])
        if seed != C.PAIR_SEED:
            raise RuntimeError("frozen pair schedule has a different seed")
        if saved.shape[0] < n_steps:
            raise RuntimeError("frozen pair schedule is shorter than requested")
        return saved[:n_steps]
    schedule = SP.pair_schedule(ctx.pairs, n_steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schedule=schedule, seed=np.int64(C.PAIR_SEED),
                        shape=np.asarray(schedule.shape, dtype=np.int64),
                        sha256=np.asarray(C.sha256_array(schedule)))
    return schedule


def pair_rows(ctx: Ctx) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray([r["row_a"] for r in ctx.pairs.rows], dtype=np.int64)
    b = np.asarray([r["row_b"] for r in ctx.pairs.rows], dtype=np.int64)
    return a, b
