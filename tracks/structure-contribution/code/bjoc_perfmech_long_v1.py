"""Training / model primitives for cscl-bjoc-performance-mechanism-long-v1.

Protocol: ``configs/bjoc_performance_mechanism_long_v1.yaml``.
Track: ``tracks/structure-contribution``.

This module contains everything that must stay frozen once GPU training has
started:

* the deterministic group-aware split builder (canonical-SMILES groups);
* the object cache + Fit-only standardiser builders;
* the supervised recipe loop (R0 240-constant / R3 360-cosine), copied
  line-for-line from the frozen ``run_bjoc_recipe_tuning_v1.train_recipe``
  loop with only the initialisation source parameterised;
* the masked atom/bond pretraining loop (Fit rows only, no labels), copied
  from ``run_bjoc_masked_pretraining_v1.pretrain_one_seed``;
* the Stage-C ``BJOCSecondMoment`` readout model and its P3-identity
  construction.

The CLI driver ``run_bjoc_performance_mechanism_long_v1.py`` imports this
module.  Analysis / scoring code lives in the driver, not here.
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
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-performance-mechanism-long-v1"

# ---- frozen constants -------------------------------------------------------
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
LR = 1.0e-3
LR_MIN = 1.0e-5
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
EXPECTED_PARAMETERS = 413505
C1_EXPECTED_PARAMETERS = 438081
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0

#: R0 and R3 are the frozen recipes of cscl-bjoc-recipe-tuning-v1.
RECIPES: dict[str, dict[str, Any]] = {
    "R0": {"epochs": 240, "schedule": "constant", "lr": LR, "lr_min": None, "weight_decay": WEIGHT_DECAY},
    "R3": {"epochs": 360, "schedule": "cosine", "lr": LR, "lr_min": LR_MIN, "weight_decay": WEIGHT_DECAY},
}

SPLIT_TARGETS = {"fit": 8000, "monitor": 1000, "locked": 1000}

#: masked pretraining constants (copied from the frozen v1 module)
PRETRAIN_UPDATES = mp.PRETRAIN_UPDATES
PRETRAIN_BATCH = mp.PRETRAIN_BATCH
PRETRAIN_LR = mp.PRETRAIN_LR
PRETRAIN_WD = mp.PRETRAIN_WD
PRETRAIN_CLIP = mp.PRETRAIN_CLIP
PRETRAIN_SHUFFLE_OFFSET = mp.PRETRAIN_SHUFFLE_OFFSET


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# hashing / io
# ---------------------------------------------------------------------------


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_sha256(rows: Sequence[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()).hexdigest()


def state_group_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def encoder_group_hash(model: src.BJOC) -> str:
    st = mp.encoder_state(model)
    combined = (
        {f"atom:{k}": v for k, v in st["atom_encoder"].items()}
        | {f"bond:{k}": v for k, v in st["bond_encoder"].items()}
        | {f"junc:{k}": v for k, v in st["junction_encoder"].items()}
    )
    return state_group_sha256(combined)


# ---------------------------------------------------------------------------
# split / objects / standardiser
# ---------------------------------------------------------------------------


def build_split(smiles: Sequence[str], salt: str, fit_target: int = 8000, monitor_target: int = 1000) -> dict[str, Any]:
    """Deterministic canonical-SMILES group split (same rule as split A)."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(str(s), []).append(i)
    keys = sorted(groups, key=lambda s: hashlib.sha256(f"{salt}|{s}".encode()).hexdigest())
    fit: list[int] = []
    monitor: list[int] = []
    locked: list[int] = []
    for key in keys:
        rows = sorted(groups[key])
        if len(fit) < fit_target:
            fit.extend(rows)
        elif len(monitor) < monitor_target:
            monitor.extend(rows)
        else:
            locked.extend(rows)
    side_of: dict[int, str] = {}
    for name, rows in (("fit", fit), ("monitor", monitor), ("locked", locked)):
        for r in rows:
            side_of[int(r)] = name
    cross = sum(1 for rows in groups.values() if len({side_of[int(r)] for r in rows}) > 1)
    return {
        "salt": salt,
        "n_groups": len(groups),
        "fit": np.asarray(sorted(fit), dtype=np.int64),
        "monitor": np.asarray(sorted(monitor), dtype=np.int64),
        "locked": np.asarray(sorted(locked), dtype=np.int64),
        "duplicate_groups": {s: sorted(r) for s, r in groups.items() if len(r) > 1},
        "duplicate_groups_cross_sides": int(cross),
    }


def build_objects_for_rows(rows: Sequence[int], ds) -> list[sro.MolObjects]:
    objs = []
    for i in rows:
        d = ds[int(i)]
        objs.append(sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
    return objs


def fit_standardiser(objs: Sequence[sro.MolObjects]) -> dict[str, Any]:
    gfit = np.stack([o.global_feat for o in objs], 0).astype(np.float64)
    mean = gfit.mean(0)
    scale = gfit.std(0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {
        "fit_only": True,
        "n_fit": int(len(objs)),
        "global_mean": mean.tolist(),
        "global_scale": scale.tolist(),
        "names": list(sro.GLOBAL_FEAT_NAMES),
    }


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_bjoc_null(seed: int, norm: Mapping[str, Any], device: torch.device | str) -> src.BJOC:
    """Scratch BJOC-Null init.  Must be the exact P-round init for the seed."""
    cfg = src.BJOCConfig()
    seed_everything(seed)
    return src.build_bjoc(
        cfg,
        seed=seed,
        organization_mode=src.NULL,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device=device,
    )


class C1Config(src.BJOCConfig):
    """Reader input widened by one compose_out block (``c_sq``)."""

    @property
    def reader_in(self) -> int:
        return 2 * self.compose_out + 2 * self.atom_out + sro.GLOBAL_DIM


class BJOCSecondMoment(src.BJOC):
    """BJOC-Null with the reader input extended by ``sum_b c_b^2``.

    The reader input becomes ``[z, c_sq, a_sum, a_sq, g]`` (the original
    [z, a_sum, a_sq, g] with the new 96-dim block inserted after ``z``).
    All computation is one-shot (single ``index_add_``); nothing is iterated.
    """

    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        feats = super().forward_features(batch, organization_mode)
        c = feats["c"]
        n_graphs = int(batch["n_graphs"])
        c_sq = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        c_sq.index_add_(0, batch["bond_batch"], c * c)
        co = int(self.config.compose_out)
        ao = int(self.config.atom_out)
        old = feats["reader_in"]
        a_sum = old[:, co : co + ao]
        a_sq = old[:, co + ao : co + 2 * ao]
        g = old[:, co + 2 * ao :]
        feats["c_sq"] = c_sq
        feats["reader_in"] = torch.cat([feats["z"], c_sq, a_sum, a_sq, g], dim=1)
        return feats


def _expand_reader_first_layer(base_reader: src._MLP, c1_reader: src._MLP) -> None:
    """Copy base reader into the widened C1 reader with zero new columns."""
    cfg = src.BJOCConfig()
    co, ao = int(cfg.compose_out), int(cfg.atom_out)
    w_old = base_reader.lins[0].weight.detach()
    w_new = torch.zeros_like(c1_reader.lins[0].weight)
    w_new[:, 0:co] = w_old[:, 0:co]  # z unchanged
    w_new[:, co + ao : co + 2 * ao] = w_old[:, co : co + ao]  # a_sum shifted by one ao block
    w_new[:, co + 2 * ao :] = w_old[:, co + ao :]  # a_sq, g shifted by one ao block
    with torch.no_grad():
        c1_reader.lins[0].weight.copy_(w_new)
        c1_reader.lins[0].bias.copy_(base_reader.lins[0].bias.detach())
        for i in range(1, len(base_reader.lins)):
            c1_reader.lins[i].load_state_dict(base_reader.lins[i].state_dict())


def build_c1_from_init(
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[src.BJOC, BJOCSecondMoment, dict[str, Any]]:
    """Build (base P3-init model, widened C1 model) from one shared scratch init.

    ``base`` is the exact P3 supervision-phase initialisation (scratch init,
    optionally SSL encoder loaded).  ``c1`` shares every base parameter and
    widens only the reader first weight matrix with zero columns, so its
    initial function is identical to ``base``.
    """
    base = build_bjoc_null(seed, norm, device)
    scratch_full = src.state_dict_hash(base)
    scratch_group = mp.group_state_hash(base)
    scratch_encoder = encoder_group_hash(base)
    if ssl_state is not None:
        mp.load_encoder_state(base, ssl_state, device)
    base_audit = src.parameter_audit(src.BJOCConfig(), base)
    if not base_audit["parameter_exact"] or base_audit["actual_parameters"] != EXPECTED_PARAMETERS:
        raise SystemExit(f"base parameter audit failed: {base_audit}")
    encoder_used = encoder_group_hash(base)

    cfg = C1Config()
    c1 = BJOCSecondMoment(
        cfg,
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        src.NULL,
    )
    c1.atom_encoder.load_state_dict(base.atom_encoder.state_dict())
    c1.bond_encoder.load_state_dict(base.bond_encoder.state_dict())
    c1.junction_encoder.load_state_dict(base.junction_encoder.state_dict())
    c1.group_composer.load_state_dict(base.group_composer.state_dict())
    _expand_reader_first_layer(base.reader, c1.reader)
    c1.to(device)
    c1_audit = src.parameter_audit(cfg, c1)
    if not c1_audit["parameter_exact"] or c1_audit["actual_parameters"] != C1_EXPECTED_PARAMETERS:
        raise SystemExit(f"C1 parameter audit failed: {c1_audit}")
    info = {
        "base_scratch_init_full_sha256": scratch_full,
        "base_scratch_init_group_state_sha256": scratch_group,
        "base_scratch_init_encoder_sha256": scratch_encoder,
        "base_encoder_used_sha256": encoder_used,
        "c1_parameters": int(sum(p.numel() for p in c1.parameters())),
        "c1_new_parameters": int(C1_EXPECTED_PARAMETERS - EXPECTED_PARAMETERS),
        "zero_columns": True,
    }
    return base, c1, info


# ---------------------------------------------------------------------------
# batch / prediction helpers
# ---------------------------------------------------------------------------


def shuffle_batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    order = torch.randperm(int(n), generator=generator).tolist()
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def batch_order_hash(seed: int, n: int, batch_size: int, epochs: int) -> str:
    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    h = hashlib.sha256()
    for _ in range(int(epochs)):
        order = torch.randperm(int(n), generator=gen)
        h.update(np.ascontiguousarray(order.numpy(), dtype=np.int64).tobytes())
    return h.hexdigest()


def _mv(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_model(model: src.BJOC, objs: Sequence[sro.MolObjects], device: torch.device | str) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), 256):
            batch = _mv(sro.batch_objects(list(objs[i : i + 256])), torch.device(device))
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0)


def load_y_rows(encoded_path: Path, rows: Sequence[int]) -> np.ndarray:
    enc = torch.load(encoded_path, map_location="cpu", weights_only=False)
    values = np.asarray([float(enc[int(i)].y.reshape(-1)[0]) for i in rows], dtype=np.float64)
    del enc
    return values


# ---------------------------------------------------------------------------
# frozen learning-rate schedule (identical to cscl-bjoc-recipe-tuning-v1)
# ---------------------------------------------------------------------------


def cosine_lr(step: int, total: int, lr0: float, lr_min: float) -> float:
    if total <= 0:
        return float(lr0)
    return float(lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * float(step) / float(total))))


def schedule_lr(recipe: Mapping[str, Any], step: int, total: int) -> float:
    if recipe["schedule"] == "constant":
        return float(recipe["lr"])
    if recipe["schedule"] == "cosine":
        return cosine_lr(step, total, float(recipe["lr"]), float(recipe["lr_min"]))
    raise SystemExit(f"unknown schedule {recipe['schedule']!r}")


# ---------------------------------------------------------------------------
# supervised training (frozen loop)
# ---------------------------------------------------------------------------


def train_supervised(
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
    model_kind: str = "base",
    smoke_epochs: int | None = None,
    log_fn: Callable[[str], None] = log,
) -> dict[str, Any]:
    """One supervised trajectory.

    ``init`` is ``scratch`` or ``ssl``.  ``model_kind`` is ``base`` or ``c1``.
    The loop body (optimizer, schedule, shuffle, clip, soup, fit MAE) is the
    frozen recipe-tuning loop; the only difference is the initialisation
    source and the optional C1 reader width.
    """
    if recipe_key not in RECIPES:
        raise SystemExit(f"unknown recipe {recipe_key!r}")
    recipe = RECIPES[recipe_key]
    if init not in ("scratch", "ssl"):
        raise SystemExit(f"unknown init {init!r}")
    if init == "ssl" and ssl_state is None:
        raise SystemExit("init=ssl requires a pretrained encoder state")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")

    if model_kind == "base":
        model = build_bjoc_null(seed, norm, device)
        c1_info: dict[str, Any] = {}
    elif model_kind == "c1":
        _, model, c1_info = build_c1_from_init(seed, norm, device, ssl_state if init == "ssl" else None)
    else:
        raise SystemExit(f"unknown model_kind {model_kind!r}")

    if model_kind == "base":
        init_full_sha = src.state_dict_hash(model)
        init_group_sha = mp.group_state_hash(model)
        init_encoder_sha = encoder_group_hash(model)
    else:
        init_full_sha = c1_info["base_scratch_init_full_sha256"]
        init_group_sha = c1_info["base_scratch_init_group_state_sha256"]
        init_encoder_sha = c1_info["base_scratch_init_encoder_sha256"]
    encoder_used_sha = init_encoder_sha
    if init == "ssl" and model_kind == "base":
        mp.load_encoder_state(model, ssl_state, device)
        encoder_used_sha = encoder_group_hash(model)
    elif model_kind == "c1" and init == "ssl":
        encoder_used_sha = c1_info["base_encoder_used_sha256"]
    audit = src.parameter_audit(C1Config() if model_kind == "c1" else src.BJOCConfig(), model)
    expected = EXPECTED_PARAMETERS if model_kind == "base" else C1_EXPECTED_PARAMETERS
    if not audit["parameter_exact"] or audit["actual_parameters"] != expected:
        raise SystemExit(f"parameter audit failed: {audit}")

    epochs = int(recipe["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(rows))
    steps_per_epoch = math.ceil(n / BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = None
    if recipe["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"]))
    elif recipe["schedule"] != "constant":
        raise SystemExit(f"unknown schedule {recipe['schedule']!r}")

    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=SOUP_LAST)
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
        for indices in shuffle_batches(n, BATCH_SIZE, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = _mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > GRAD_CLIP:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                expected_lr = schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
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

    members, soup = _last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_model(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "recipe_key": recipe_key,
        "recipe": dict(recipe),
        "init": init,
        "model_kind": model_kind,
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {SOUP_LAST} epoch states",
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
        "c1_info": c1_info,
        "parameter_audit": audit,
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "full_hash_independent_recomputation": batch_order_hash(seed, n, BATCH_SIZE, epochs),
            "full_matches": order_hash.hexdigest() == batch_order_hash(seed, n, BATCH_SIZE, epochs),
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
        "labels_used": "train rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    assert 0 < len(states) <= SOUP_LAST, len(states)
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0].keys()}
    return members, soup


# ---------------------------------------------------------------------------
# masked pretraining (frozen loop; Fit rows only, no labels)
# ---------------------------------------------------------------------------


def pretrain_masked(
    *,
    seed: int,
    fit_objs: Sequence[sro.MolObjects],
    fit_rows: Sequence[int],
    raw_dataset,
    device: torch.device,
    mode: str = "atom_bond",
    smoke_updates: int | None = None,
    log_fn: Callable[[str], None] = log,
) -> dict[str, Any]:
    """Masked atom/bond pretraining.  Reads no labels at all.

    ``mode`` is ``atom_bond`` (frozen v1 objective) or ``atom_only``
    (Stage-E ablation: bond CE is disabled, atom CE unchanged).
    """
    if mode not in ("atom_bond", "atom_only"):
        raise SystemExit(f"unknown pretrain mode {mode!r}")
    train_rows = set(int(i) for i in fit_rows)
    cfg = src.BJOCConfig()
    seed_everything(seed)
    backbone = src.build_bjoc(
        cfg,
        seed=seed,
        organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
        device=device,
    )
    init_encoder_sha = encoder_group_hash(backbone)
    pretext = mp.PretextModel(backbone).to(device)
    optimizer = torch.optim.Adam(pretext.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
    gen = torch.Generator().manual_seed(int(seed) + PRETRAIN_SHUFFLE_OFFSET)

    curve: list[dict[str, Any]] = []
    steps = 0
    visit = 0
    loss_at_steps: dict[int, float] = {}
    started = time.perf_counter()
    steps_limit = PRETRAIN_UPDATES if smoke_updates is None else int(smoke_updates)
    if smoke_updates is not None and not (1 <= steps_limit <= 100):
        raise SystemExit("smoke updates must be in [1, 100]")
    while steps < steps_limit:
        pretext.train()
        ce_a_sum = ce_b_sum = acc_a_sum = acc_b_sum = 0.0
        n_a = n_b = n_steps = 0
        t_ep = time.perf_counter()
        stop = False
        for chunk in shuffle_batches(len(fit_objs), PRETRAIN_BATCH, gen):
            pre = []
            for j in chunk:
                o = fit_objs[j]
                if int(o.mol_index) not in train_rows:
                    raise SystemExit("pretraining row outside Fit")
                d = raw_dataset[int(o.mol_index)]
                pre.append(
                    mp.build_masked_sample(
                        o, d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy(),
                        seed=seed, visit_index=visit,
                    )
                )
            batch, targets_a, targets_b, _, _ = mp.batch_pretext_samples(pre)
            batch = _mv(batch, device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)
            if mode == "atom_bond":
                loss, parts = pretext.loss(batch, targets_a, targets_b)
            else:
                logits = pretext(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
                loss = F.cross_entropy(logits["atom_logits"], targets_a)
                parts = {"ce_atom": loss.detach()}
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretext.parameters(), PRETRAIN_CLIP)
            optimizer.step()
            steps += 1
            n_steps += 1
            with torch.no_grad():
                logits = pretext(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
                if targets_a.numel():
                    ce_a_sum += float(parts["ce_atom"])
                    acc_a_sum += float((logits["atom_logits"].argmax(1) == targets_a).float().mean())
                    n_a += 1
                if mode == "atom_bond" and targets_b.numel():
                    ce_b_sum += float(parts["ce_bond"])
                    acc_b_sum += float((logits["bond_logits"].argmax(1) == targets_b).float().mean())
                    n_b += 1
            loss_at_steps[steps] = float(loss)
            if steps >= steps_limit:
                stop = True
                break
        curve.append(
            {
                "visit": visit,
                "steps_total": steps,
                "ce_atom": ce_a_sum / max(n_a, 1),
                "ce_bond": ce_b_sum / max(n_b, 1) if n_b else None,
                "acc_atom": acc_a_sum / max(n_a, 1),
                "acc_bond": acc_b_sum / max(n_b, 1) if n_b else None,
                "seconds": time.perf_counter() - t_ep,
            }
        )
        if visit % 10 == 0 or stop:
            log_fn(
                f"[ssl s{seed}] visit={visit} steps={steps}/{PRETRAIN_UPDATES} "
                f"ce_a={curve[-1]['ce_atom']:.4f} acc_a={curve[-1]['acc_atom']:.4f} "
                f"ce_b={curve[-1]['ce_bond']} acc_b={curve[-1]['acc_bond']}"
            )
        visit += 1
    wall = time.perf_counter() - started

    enc_state = mp.encoder_state(backbone)
    final_encoder_sha = encoder_group_hash(backbone)
    return {
        "protocol": PROTOCOL_ID,
        "arm": "SSL",
        "seed": int(seed),
        "mode": mode,
        "updates": int(steps),
        "smoke": bool(smoke_updates is not None),
        "visits": visit,
        "batch_size": PRETRAIN_BATCH,
        "hyper": {
            "optimizer": "Adam",
            "lr": PRETRAIN_LR,
            "weight_decay": PRETRAIN_WD,
            "grad_clip": PRETRAIN_CLIP,
            "loss": "CE(masked atoms) + CE(masked physical bonds), weights 1.0/1.0" if mode == "atom_bond" else "CE(masked atoms) only",
            "checkpoint_rule": "last step only",
        },
        "mask": {
            "atom_ratio": mp.ATOM_MASK_RATIO,
            "bond_ratio": mp.BOND_MASK_RATIO,
            "atom_token": mp.ATOM_MASK_TOKEN,
            "bond_token": mp.BOND_MASK_TOKEN,
            "rng": "SeedSequence([seed, molecule_id, visit_index])",
        },
        "fit_rows_only": True,
        "n_fit_rows": len(fit_objs),
        "wall_seconds": wall,
        "encoder_init_sha256": init_encoder_sha,
        "encoder_final_sha256": final_encoder_sha,
        "encoder_parameters_changed": bool(init_encoder_sha != final_encoder_sha),
        "encoder_state": enc_state,
        "curve": curve,
        "final_ce_atom": curve[-1]["ce_atom"],
        "final_ce_bond": curve[-1]["ce_bond"],
        "final_acc_atom": curve[-1]["acc_atom"],
        "final_acc_bond": curve[-1]["acc_bond"],
        "loss_at_steps": {str(k): v for k, v in sorted(loss_at_steps.items())},
        "params": int(sum(p.numel() for p in pretext.parameters())),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
