"""P3-Slim v1 — reader-width capacity test on top of the frozen BJOC-P3 recipe.

Protocol ``cscl-p3-slim-v1`` (``results/p3_slim_v1/PROTOCOL.md``).

The two arms are the same one-shot BJOC-Null model, the same masked SSL
checkpoint, the same Fit rows, the same Adam/cosine/L1/last-5-soup recipe and
the same minibatch order.  The **only** difference is the graph-level reader:

    P3     : 321 -> 256 -> 256 -> 1   (148,481 reader parameters; 413,505 total)
    P3Slim : 321 -> 128 -> 128 -> 1   ( 57,857 reader parameters; 322,881 total)

Common modules (atom / bond / junction encoders and the Null composer) are
copied bit-for-bit from one canonical P3 initialisation.  The Slim reader is
initialised from a separate, fixed RNG seed (``seed + 10_000``); this cannot
change the already-created common modules because they are copied afterwards.

No GNN, attention, Transformer, message passing, ensemble, extra pretraining,
external data, target decomposition or penalised-logP formula is used.  This
module is label-free except :func:`train_supervised_model`, which reads Fit
raw ``y`` only.
"""

from __future__ import annotations

import hashlib
import io
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

import bjoc_arch_breakthrough_v1 as ab  # noqa: E402
import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-p3-slim-v1"
SPLIT_TAG = "U"
SPLIT_SALT = "cscl-p3-slim-v1"

SEEDS = (0, 1)
ARMS = ("P3", "P3Slim")
RECIPE_KEY = "R3"

P3_READER_HIDDEN = 256
SLIM_READER_HIDDEN = 128
READER_LAYERS = 2
SLIM_READER_SEED_OFFSET = 10_000

P3_PARAMETERS = 413_505
SLIM_PARAMETERS = 322_881
P3_READER_PARAMETERS = 148_481
SLIM_READER_PARAMETERS = 57_857
SHARED_PARAMETERS = 265_024
PARAM_CAP = 500_000

MEANINGFUL_DELTA = 0.004
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
BOOTSTRAP_N = 5_000
BOOTSTRAP_SEED = 0
RMSE_TOL = 0.005
TAIL_TOL = 0.005
GPU_BUDGET_CAP_HOURS = 2.0

ZONES = ("monitor", "locked")
PRIMARY_PHASE_ZONE = "monitor"

SHARED_MODULE_NAMES = ("atom_encoder", "bond_encoder", "junction_encoder", "group_composer")


class P3SlimError(RuntimeError):
    """Hard protocol violation (raises instead of silently continuing)."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise P3SlimError(message)


# ---------------------------------------------------------------------------
# hashes / parameter audit
# ---------------------------------------------------------------------------


def module_state_hash(module: nn.Module) -> str:
    return bp.state_group_sha256({k: v for k, v in module.state_dict().items()})


def shared_hash(model: nn.Module) -> str:
    combined: dict[str, torch.Tensor] = {}
    for name in SHARED_MODULE_NAMES:
        for k, v in getattr(model, name).state_dict().items():
            combined[f"{name}:{k}"] = v
    return bp.state_group_sha256(combined)


def encoder_hash(model: nn.Module) -> str:
    """Encoder-group hash using the frozen P3/SSL convention (``atom:``...)."""
    return bp.encoder_group_hash(model)


def encoder_state_hash(enc: Mapping[str, Mapping[str, torch.Tensor]]) -> str:
    combined: dict[str, torch.Tensor] = {}
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        prefix = {"atom_encoder": "atom", "bond_encoder": "bond", "junction_encoder": "junc"}[name]
        for k, v in enc[name].items():
            combined[f"{prefix}:{k}"] = v
    return bp.state_group_sha256(combined)


def reader_hash(model: nn.Module) -> str:
    return module_state_hash(model.reader)


def full_hash(model: nn.Module) -> str:
    return src.state_dict_hash(model)


def model_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def module_parameter_counts(model: nn.Module) -> dict[str, int]:
    out = {name: int(sum(p.numel() for p in getattr(model, name).parameters())) for name in SHARED_MODULE_NAMES}
    out["reader"] = int(sum(p.numel() for p in model.reader.parameters()))
    out["total"] = model_parameters(model)
    return out


def parameter_inventory() -> dict[str, Any]:
    p3_cfg = src.BJOCConfig(reader_hidden=P3_READER_HIDDEN, reader_layers=READER_LAYERS)
    slim_cfg = src.BJOCConfig(reader_hidden=SLIM_READER_HIDDEN, reader_layers=READER_LAYERS)
    p3 = p3_cfg.inventory()
    slim = slim_cfg.inventory()
    return {
        "protocol": PROTOCOL_ID,
        "p3": {
            "reader_hidden": P3_READER_HIDDEN,
            "reader_layers": READER_LAYERS,
            "reader_in": p3_cfg.reader_in,
            "inventory": p3,
            "reader_parameters": p3["reader"],
            "total_parameters": p3["total"],
        },
        "slim": {
            "reader_hidden": SLIM_READER_HIDDEN,
            "reader_layers": READER_LAYERS,
            "reader_in": slim_cfg.reader_in,
            "inventory": slim,
            "reader_parameters": slim["reader"],
            "total_parameters": slim["total"],
        },
        "shared_parameters": p3["total"] - p3["reader"],
        "reader_reduction": p3["reader"] - slim["reader"],
        "total_reduction": p3["total"] - slim["total"],
        "expected": {
            "p3_total": P3_PARAMETERS,
            "slim_total": SLIM_PARAMETERS,
            "p3_reader": P3_READER_PARAMETERS,
            "slim_reader": SLIM_READER_PARAMETERS,
            "shared": SHARED_PARAMETERS,
        },
        "parameter_cap": PARAM_CAP,
        "all_expected": bool(
            p3["total"] == P3_PARAMETERS
            and slim["total"] == SLIM_PARAMETERS
            and p3["reader"] == P3_READER_PARAMETERS
            and slim["reader"] == SLIM_READER_PARAMETERS
            and p3["total"] - p3["reader"] == SHARED_PARAMETERS
        ),
    }


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_p3(seed: int, norm: Mapping[str, Any], device: torch.device | str) -> src.BJOC:
    """Exact frozen P3 scratch init (413,505 parameters)."""
    model = bp.build_bjoc_null(int(seed), norm, device)
    require(model_parameters(model) == P3_PARAMETERS, "P3 parameter audit failed")
    require(int(model.config.reader_hidden) == P3_READER_HIDDEN, "P3 reader width changed")
    return model


def build_slim(
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str,
    *,
    canonical: src.BJOC | None = None,
) -> src.BJOC:
    """Slim model with common modules copied bit-for-bit from canonical P3.

    The Slim reader is created by a separate deterministic seed
    ``seed + SLIM_READER_SEED_OFFSET``.  Only after that construction are the
    common modules overwritten from ``canonical``; therefore the reader RNG
    cannot perturb any shared parameter.
    """
    canonical = build_p3(seed, norm, device) if canonical is None else canonical
    cfg = src.BJOCConfig(reader_hidden=SLIM_READER_HIDDEN, reader_layers=READER_LAYERS)
    slim = src.build_bjoc(
        cfg,
        seed=int(seed) + SLIM_READER_SEED_OFFSET,
        organization_mode=src.NULL,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device=device,
    )
    for name in SHARED_MODULE_NAMES:
        getattr(slim, name).load_state_dict(getattr(canonical, name).state_dict())
    slim.global_mean.copy_(canonical.global_mean)
    slim.global_scale.copy_(canonical.global_scale)
    require(model_parameters(slim) == SLIM_PARAMETERS, "Slim parameter audit failed")
    require(int(slim.config.reader_hidden) == SLIM_READER_HIDDEN, "Slim reader width changed")
    require(shared_hash(slim) == shared_hash(canonical), "Slim common modules differ from canonical P3")
    require(reader_hash(slim) != reader_hash(canonical), "Slim reader unexpectedly equals P3 reader")
    return slim


def build_pair(seed: int, norm: Mapping[str, Any], device: torch.device | str) -> tuple[src.BJOC, src.BJOC]:
    canonical = build_p3(seed, norm, device)
    slim = build_slim(seed, norm, device, canonical=canonical)
    return canonical, slim


def initialization_record(seed: int, norm: Mapping[str, Any], device: torch.device | str = "cpu") -> dict[str, Any]:
    p3, slim = build_pair(seed, norm, device)
    shared_equal = shared_hash(p3) == shared_hash(slim)
    return {
        "seed": int(seed),
        "canonical_p3": {
            "reader_hidden": P3_READER_HIDDEN,
            "reader_in": int(p3.config.reader_in),
            "parameters": model_parameters(p3),
            "module_parameters": module_parameter_counts(p3),
            "full_sha256": full_hash(p3),
            "shared_sha256": shared_hash(p3),
            "encoder_sha256": encoder_hash(p3),
            "reader_sha256": reader_hash(p3),
        },
        "slim": {
            "reader_hidden": SLIM_READER_HIDDEN,
            "reader_in": int(slim.config.reader_in),
            "reader_seed": int(seed) + SLIM_READER_SEED_OFFSET,
            "parameters": model_parameters(slim),
            "module_parameters": module_parameter_counts(slim),
            "full_sha256": full_hash(slim),
            "shared_sha256": shared_hash(slim),
            "encoder_sha256": encoder_hash(slim),
            "reader_sha256": reader_hash(slim),
        },
        "shared_modules_equal": bool(shared_equal),
        "encoder_equal": bool(encoder_hash(p3) == encoder_hash(slim)),
        "reader_distinct": bool(reader_hash(p3) != reader_hash(slim)),
        "reader_input_equal": bool(p3.config.reader_in == slim.config.reader_in == 321),
        "shared_parameter_count": model_parameters(p3) - module_parameter_counts(p3)["reader"],
        "reader_parameter_reduction": module_parameter_counts(p3)["reader"] - module_parameter_counts(slim)["reader"],
        "total_parameter_reduction": model_parameters(p3) - model_parameters(slim),
        "init_rule": (
            "canonical P3 built first; Slim reader built from seed+10000; then all four shared "
            "modules and the two buffers copied bit-for-bit from canonical P3"
        ),
    }


# ---------------------------------------------------------------------------
# frozen supervised loop, taking a prebuilt model
# ---------------------------------------------------------------------------


def train_supervised_model(
    *,
    arm: str,
    seed: int,
    model: src.BJOC,
    train_objs: Sequence[sro.MolObjects],
    y_fit: np.ndarray,
    device: torch.device,
    epochs: int = 360,
    batch_size: int = bp.BATCH_SIZE,
    grad_clip: float = bp.GRAD_CLIP,
    collect_epoch_states: bool = False,
    log_fn: Callable[[str], None] = bp.log,
) -> dict[str, Any]:
    """Frozen R3 loop (same body as ``bp.train_supervised``) on a built model."""
    require(arm in ARMS, f"unknown arm {arm!r}")
    recipe = bp.RECIPES[RECIPE_KEY]
    require(int(epochs) >= 1, "epochs must be positive")
    require(int(batch_size) >= 1, "batch_size must be positive")
    require(float(grad_clip) == float(bp.GRAD_CLIP), "gradient clip changed")
    require(int(len(train_objs)) == int(len(y_fit)), "train objects / targets length mismatch")

    n = int(len(train_objs))
    steps_per_epoch = math.ceil(n / int(batch_size))
    total_updates = int(epochs) * steps_per_epoch
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(recipe["lr"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"])
    )

    gen = torch.Generator().manual_seed(int(seed) + bp.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=bp.SOUP_LAST)
    epoch_states: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    total_mol = 0
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, int(batch_size), gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = bp._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            if not math.isfinite(gn):
                raise P3SlimError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > grad_clip:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            expected_lr = bp.schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
            lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
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
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        states.append(state)
        if collect_epoch_states:
            epoch_states.append({k: v.detach().cpu().clone() for k, v in state.items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == int(epochs):
            log_fn(
                f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} "
                f"lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s"
            )
        if not math.isfinite(train_mae):
            raise P3SlimError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")

    wall = time.perf_counter() - started
    members, soup = bp._last_soup(list(states), int(epochs))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = bp.predict_model(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    independent_order = bp.batch_order_hash(int(seed), n, int(batch_size), int(epochs))
    expected_parameters = P3_PARAMETERS if arm == "P3" else SLIM_PARAMETERS
    info: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "recipe_key": RECIPE_KEY,
        "recipe": dict(recipe),
        "reader_hidden": int(model.config.reader_hidden),
        "epochs": int(epochs),
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "batch_size": int(batch_size),
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
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "lr_schedule_deviation_max": float(lr_deviation_max),
        },
        "lr_curve": curve,
        "wall_seconds": float(wall),
        "molecules_seen": int(total_mol),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "full_hash_independent_recomputation": independent_order,
            "full_matches": order_hash.hexdigest() == independent_order,
        },
        "parameter_audit": {
            "actual_parameters": model_parameters(model),
            "expected_parameters": int(expected_parameters),
            "parameter_exact": bool(model_parameters(model) == expected_parameters),
            "reader_parameters": int(sum(p.numel() for p in model.reader.parameters())),
        },
        "cost": {
            "optimizer_updates": int(total_updates),
            "forward_calls": int(total_updates),
            "backward_calls": int(total_updates),
            "relative_forward_backward_multiplier_vs_p3": 1.0,
        },
        "labels_used": "train rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    require(info["parameter_audit"]["parameter_exact"], f"{arm} parameter audit failed")
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
                (0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2),
                (5, 0, 1), (4, 9, 1), (9, 8, 2), (8, 7, 1), (7, 6, 2), (6, 5, 1),
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
    return bp._mv(sro.batch_objects(objs), torch.device("cpu"))


def _grad_norm_for(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return float(math.sqrt(total))


def _atom_types(obj: sro.MolObjects) -> list[int]:
    return [int(np.argmax(obj.atom_feat[i, : sro.ATOM_CATEGORIES])) for i in range(int(obj.n_atoms))]


def _bond_list(obj: sro.MolObjects) -> list[tuple[int, int, int]]:
    return [(int(u), int(v), int(bt)) for (u, v), bt in zip(np.asarray(obj.bond_index), np.asarray(obj.bond_type))]


def _relabeled(obj: sro.MolObjects, perm: Sequence[int]) -> sro.MolObjects:
    """New atom order ``new = perm[old]``; graph is otherwise unchanged."""
    types = _atom_types(obj)
    bonds = _bond_list(obj)
    new_types = [0] * len(types)
    for old, new in enumerate(perm):
        new_types[int(new)] = types[old]
    new_bonds = [(int(perm[u]), int(perm[v]), bt) for u, v, bt in bonds]
    return _mol(int(obj.mol_index), new_types, new_bonds)


def _source_isolation_scan() -> dict[str, Any]:
    """G7: no official valid/test loader and no guarded-label bypass in code."""
    split_tokens = ["split=" + q + name + q for q in ('"', "'") for name in ("valid", "test")]
    path_tokens = ["processed/" + name for name in ("valid", "test")]
    hits: list[str] = []
    for name in ("p3_slim_v1.py", "run_p3_slim_v1.py"):
        path = _CODE_DIR / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in split_tokens + path_tokens:
            if token in text:
                hits.append(f"{name}:{token}")
    return {"hits": hits, "clean": not hits, "label_guard": ab.label_guard_selfcheck()}


def run_slim_unit_gates(n_fit: int | None = None) -> dict[str, Any]:
    """CPU-only G1-G8 gates; no ZINC, no labels, no GPU."""
    gates: list[dict[str, Any]] = []
    norm = _tiny_norm()
    objs = synthetic_objects()
    y = np.asarray([0.5, -0.3, 0.1, 0.9, -0.7, 0.2, 0.4, -0.1], dtype=np.float64)
    torch.set_num_threads(1)
    p3, slim = build_pair(0, norm, "cpu")

    # G1: exact parameter counts and reader-only difference.
    p3_counts = module_parameter_counts(p3)
    slim_counts = module_parameter_counts(slim)
    inv = parameter_inventory()
    gate1_ok = (
        p3_counts["total"] == P3_PARAMETERS
        and slim_counts["total"] == SLIM_PARAMETERS
        and p3_counts["reader"] == P3_READER_PARAMETERS
        and slim_counts["reader"] == SLIM_READER_PARAMETERS
        and p3_counts["total"] - p3_counts["reader"] == SHARED_PARAMETERS
        and slim_counts["total"] - slim_counts["reader"] == SHARED_PARAMETERS
        and p3_counts["total"] - slim_counts["total"] == P3_PARAMETERS - SLIM_PARAMETERS
        and inv["all_expected"]
    )
    gates.append(
        _gate(
            "G1_parameter_counts",
            gate1_ok,
            p3_total=p3_counts["total"],
            slim_total=slim_counts["total"],
            p3_reader=p3_counts["reader"],
            slim_reader=slim_counts["reader"],
            shared_p3=p3_counts["total"] - p3_counts["reader"],
            shared_slim=slim_counts["total"] - slim_counts["reader"],
            total_reduction=p3_counts["total"] - slim_counts["total"],
            expected_reduction=P3_PARAMETERS - SLIM_PARAMETERS,
        )
    )

    # G2: common-module identity and no inactive shared pathway.
    shared_modules = {
        name: {
            "p3_sha256": module_state_hash(getattr(p3, name)),
            "slim_sha256": module_state_hash(getattr(slim, name)),
            "equal": module_state_hash(getattr(p3, name)) == module_state_hash(getattr(slim, name)),
        }
        for name in SHARED_MODULE_NAMES
    }
    gate2_ok = (
        all(v["equal"] for v in shared_modules.values())
        and p3.config.atom_hidden == slim.config.atom_hidden
        and p3.config.atom_out == slim.config.atom_out
        and p3.config.bond_hidden == slim.config.bond_hidden
        and p3.config.bond_out == slim.config.bond_out
        and p3.config.junc_hidden == slim.config.junc_hidden
        and p3.config.junc_out == slim.config.junc_out
        and p3.config.compose_hidden == slim.config.compose_hidden
        and p3.config.compose_out == slim.config.compose_out
        and p3.organization_mode == slim.organization_mode == src.NULL
        and p3.config.dropout == slim.config.dropout == 0.0
    )
    gates.append(_gate("G2_shared_modules_identical", gate2_ok, modules=shared_modules))

    # G3: the new loop's P3 path equals the frozen P3 trainer on synthetic data.
    frozen_model = build_p3(125, norm, "cpu")
    info_frozen = bp.train_supervised(
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
    new_model = build_p3(125, norm, "cpu")
    info_new = train_supervised_model(
        arm="P3",
        seed=125,
        model=new_model,
        train_objs=objs,
        y_fit=y,
        device=torch.device("cpu"),
        epochs=2,
        log_fn=lambda _m: None,
    )
    soup_diff = max(
        float((info_frozen["soup_state"][k] - info_new["soup_state"][k]).abs().max())
        for k in info_frozen["soup_state"]
    )
    frozen_model.load_state_dict(info_frozen["soup_state"])
    new_model.load_state_dict(info_new["soup_state"])
    with torch.no_grad():
        frozen_model.eval()
        new_model.eval()
        pred_frozen = frozen_model(_batch(objs)).cpu().numpy()
        pred_new = new_model(_batch(objs)).cpu().numpy()
    forward_diff = float(np.abs(pred_frozen - pred_new).max())
    gate3_ok = (
        soup_diff <= 1e-7
        and forward_diff <= 1e-6
        and info_frozen["batch_order"]["full_hash"] == info_new["batch_order"]["full_hash"]
        and info_frozen["updates"] == info_new["updates"]
    )
    gates.append(
        _gate(
            "G3_p3_replay_frozen_forward_and_loop",
            gate3_ok,
            max_abs_soup_diff=float(soup_diff),
            max_abs_forward_diff=float(forward_diff),
            batch_order_match=bool(info_frozen["batch_order"]["full_hash"] == info_new["batch_order"]["full_hash"]),
            updates_frozen=int(info_frozen["updates"]),
            updates_new=int(info_new["updates"]),
        )
    )

    # G4: both readers receive the same 321-d reader input before training.
    with torch.no_grad():
        f1 = p3.forward_features(_batch(objs[:3]))
        f2 = slim.forward_features(_batch(objs[:3]))
    reader_diff = float((f1["reader_in"] - f2["reader_in"]).abs().max())
    gate4_ok = (
        tuple(f1["reader_in"].shape) == (3, 321)
        and tuple(f2["reader_in"].shape) == (3, 321)
        and reader_diff == 0.0
    )
    gates.append(
        _gate(
            "G4_reader_input_identical",
            gate4_ok,
            p3_shape=list(f1["reader_in"].shape),
            slim_shape=list(f2["reader_in"].shape),
            max_abs_diff=reader_diff,
        )
    )

    # G5: Slim reader and all shared modules receive finite, non-zero gradients.
    slim.train()
    slim.zero_grad(set_to_none=True)
    pred = slim(_batch(objs))
    target = torch.as_tensor(y, dtype=torch.float32)
    loss = F.l1_loss(pred.view(-1), target)
    loss.backward()
    grad_norms = {name: _grad_norm_for(getattr(slim, name)) for name in (*SHARED_MODULE_NAMES, "reader")}
    gate5_ok = all(math.isfinite(v) and v > 0.0 for v in grad_norms.values())
    gates.append(_gate("G5_finite_nonzero_gradients", gate5_ok, grad_norms=grad_norms))

    # G6: invariance / batch / save-load replay for both arms.
    invariance: dict[str, Any] = {}
    for label, model in (("P3", p3), ("P3Slim", slim)):
        model.eval()
        obj = objs[4]
        with torch.no_grad():
            base = model(_batch([obj])).cpu().numpy()[0]
            perm = list(reversed(range(int(obj.n_atoms))))
            relabeled = _relabeled(obj, perm)
            relabel_pred = model(_batch([relabeled])).cpu().numpy()[0]
            swapped = _mol(int(obj.mol_index), _atom_types(obj), [(v, u, bt) for u, v, bt in _bond_list(obj)])
            swap_pred = model(_batch([swapped])).cpu().numpy()[0]
            batch_pred = model(_batch(objs[:3])).cpu().numpy()
            single_pred = np.asarray([model(_batch([o])).cpu().numpy()[0] for o in objs[:3]])
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        loaded = build_p3(999, norm, "cpu") if label == "P3" else build_slim(999, norm, "cpu")
        loaded.load_state_dict(torch.load(buf, map_location="cpu", weights_only=False))
        loaded.eval()
        with torch.no_grad():
            replay = loaded(_batch([obj])).cpu().numpy()[0]
        invariance[label] = {
            "atom_relabel_max_abs_diff": float(np.abs(base - relabel_pred).max()),
            "bond_swap_max_abs_diff": float(np.abs(base - swap_pred).max()),
            "batch_composition_max_abs_diff": float(np.abs(batch_pred - single_pred).max()),
            "save_load_max_abs_diff": float(np.abs(base - replay).max()),
        }
    gate6_ok = all(
        v["atom_relabel_max_abs_diff"] <= 1e-6
        and v["bond_swap_max_abs_diff"] <= 1e-6
        and v["batch_composition_max_abs_diff"] <= 1e-6
        and v["save_load_max_abs_diff"] == 0.0
        for v in invariance.values()
    )
    gates.append(_gate("G6_invariance_and_replay", gate6_ok, arms=invariance))

    # G7: protocol isolation.
    scan = _source_isolation_scan()
    gate7_ok = (
        scan["clean"]
        and scan["label_guard"]["guarded_y_blocked"]
        and scan["label_guard"]["structure_passthrough"]
        and scan["label_guard"]["unguarded_y_allowed"]
    )
    gates.append(_gate("G7_protocol_isolation", gate7_ok, source_scan=scan["hits"], label_guard=scan["label_guard"]))

    # G8: same update budget / batch order and exact last-5 soup for both arms.
    soup_info: dict[str, Any] = {}
    for label, model in (("P3", build_p3(126, norm, "cpu")), ("P3Slim", build_slim(126, norm, "cpu"))):
        info = train_supervised_model(
            arm=label,
            seed=126,
            model=model,
            train_objs=objs,
            y_fit=y,
            device=torch.device("cpu"),
            epochs=6,
            collect_epoch_states=True,
            log_fn=lambda _m: None,
        )
        epoch_states = info.pop("epoch_states")
        expected_members = [2, 3, 4, 5, 6]
        expected_soup = {
            k: torch.stack([epoch_states[i - 1][k].float() for i in expected_members]).mean(0)
            for k in epoch_states[0]
        }
        soup_diff = max(float((info["soup_state"][k] - expected_soup[k]).abs().max()) for k in expected_soup)
        soup_info[label] = {
            "members": info["soup_members"],
            "updates": info["updates"],
            "batch_order_sha256": info["batch_order"]["full_hash"],
            "soup_max_abs_diff": float(soup_diff),
        }
    expected_updates = int(bp.RECIPES[RECIPE_KEY]["epochs"]) * math.ceil(int(n_fit or 7996) / bp.BATCH_SIZE)
    h0 = bp.batch_order_hash(0, int(n_fit or 7996), bp.BATCH_SIZE, int(bp.RECIPES[RECIPE_KEY]["epochs"]))
    h1 = bp.batch_order_hash(1, int(n_fit or 7996), bp.BATCH_SIZE, int(bp.RECIPES[RECIPE_KEY]["epochs"]))
    gate8_ok = (
        soup_info["P3"]["members"] == [2, 3, 4, 5, 6]
        and soup_info["P3Slim"]["members"] == [2, 3, 4, 5, 6]
        and soup_info["P3"]["soup_max_abs_diff"] <= 1e-7
        and soup_info["P3Slim"]["soup_max_abs_diff"] <= 1e-7
        and soup_info["P3"]["updates"] == soup_info["P3Slim"]["updates"]
        and soup_info["P3"]["batch_order_sha256"] == soup_info["P3Slim"]["batch_order_sha256"]
        and expected_updates == 22_680
        and h0 != h1
    )
    gates.append(
        _gate(
            "G8_soup_updates_batch_order",
            gate8_ok,
            arms=soup_info,
            expected_fit_updates=int(expected_updates),
            seed0_order_sha256=h0,
            seed1_order_sha256=h1,
        )
    )

    all_passed = all(g["ok"] for g in gates)
    return {
        "protocol": PROTOCOL_ID,
        "suite": "cpu_slim_capacity_and_frozen_loop",
        "device": "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gates": gates,
        "all_passed": bool(all_passed),
        "labels_read": False,
        "zinc_loaded": False,
        "gpu_used": False,
    }
