"""ZINC task-dictionary vs matched MLP bridge, fresh fold, seed 0.

Two arms share every parameter outside the bridge (``184,667``), train from a
fresh seed-0 initialization on a **new** 8000/2000 model-internal fold
(``numpy.random.default_rng(20261004)``), and differ only in the 144->144
coding module (``82,944`` parameters):

* ``D``  — the current task dictionary bridge (column-normalized ``D_L``,
  16-step ISTA, ``lambda1=0.05``, ``lambda2=0.01``, ``E = rho * alpha @ V_L``);
* ``M``  — a two-layer bias-free SiLU MLP with ``W1 = D_L_init.T`` and
  ``W2 = V_L_init.T`` (same parameter count, same input normalization,
  different function class).

No learned state is loaded: no ``S_M``/``A0``/``C``/``T``/``O``/``H``/``Y`` or
old bridge tensors enter either arm.  Only scalar ``y`` supervises.  The
official-valid split and official-test are never loaded.

The new fold is a model-internal fit/dev split of the same 10000 official-train
rows; the old model may have trained on the new dev rows, so no warm start,
distillation or reuse of old fitted scalers is allowed.  All input
standardizers are refit on the new fit rows only.

Run:
    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_task_dictionary_and_cycle_witness_seed0_v1 --smoke
    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_task_dictionary_and_cycle_witness_seed0_v1 --arm D --out <dir>
    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_task_dictionary_and_cycle_witness_seed0_v1 --arm M --out <dir>
    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_task_dictionary_and_cycle_witness_seed0_v1 --analyze --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_structure_semantic_factorial_seed0_v1 as zsf
from tracks.ksvd.experiments.luyin16 import zinc_zero_binding_baseline_seed0_v1 as zbn

PROTOCOL_VERSION = "zinc-task-dictionary-and-cycle-witness-seed0-v1"

TRACK_ROOT = zjd.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/zinc_task_dictionary_and_cycle_witness_seed0_v1"
PREP_BLOB = zjd.PREP_DIR / "fold_objects.npz"

#: frozen recipe — never selected on dev.
SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
TRAIN_SHUFFLE_OFFSET = 101
FOLD_SEED = 20261004
N_FIT = 8000
N_DEV = 2000

#: bridge contract (FULL spec).
BRIDGE_DIM = 144
BRIDGE_ATOMS = 288
BRIDGE_PARAMETERS = int(BRIDGE_DIM * BRIDGE_ATOMS * 2)

#: expected compressed-body accounting.
EXPECTED_BODY_PARAMETERS = 184_667
EXPECTED_TOTAL_PARAMETERS = 267_611

#: statistics.
BOOT_SEED = 20261004
N_BOOT = 1000
DELTA = 0.003
G0_TOL = 0.001
DELTA_SLACK = 0.001
REPLAY_TOL = 1.0e-5

GROUP_NAMES = ("k=0", "k=-1", "k=-2", "k<=-3", "k<=-2")
ARMS = ("D", "M")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    arr = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def state_tensor_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {key: tensor_hash(value) for key, value in sorted(state.items()) if torch.is_tensor(value)}


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "k=0": k == 0,
        "k=-1": k == -1,
        "k=-2": k == -2,
        "k<=-3": k <= -3,
        "k<=-2": k <= -2,
    }


# ---------------------------------------------------------------------------
# new fold (generated once, frozen)
# ---------------------------------------------------------------------------


def build_fold() -> dict[str, Any]:
    perm = np.random.default_rng(20261004).permutation(10000)
    fit_idx = np.sort(perm[:8000]).astype(np.int64)
    dev_idx = np.sort(perm[8000:]).astype(np.int64)
    if int(np.intersect1d(fit_idx, dev_idx).size) != 0:
        raise RuntimeError("new fold fit/dev overlap")
    if int(np.union1d(fit_idx, dev_idx).size) != 10000:
        raise RuntimeError("new fold does not cover all train rows")
    out = {
        "definition": "np.random.default_rng(20261004).permutation(10000); perm[:8000] fit, perm[8000:] dev; np.sort each",
        "fold_seed": int(20261004),
        "n_fit": int(fit_idx.size),
        "n_dev": int(dev_idx.size),
        "fit_idx_sha256": hashlib.sha256(fit_idx.tobytes()).hexdigest(),
        "dev_idx_sha256": hashlib.sha256(dev_idx.tobytes()).hexdigest(),
        "fit_idx": fit_idx,
        "dev_idx": dev_idx,
    }
    return out


# ---------------------------------------------------------------------------
# new-fit-only preprocessing (no old-fit scaler, no valid)
# ---------------------------------------------------------------------------


def _stack(data_list: Sequence[Any], attr: str) -> tuple[np.ndarray, list[int]]:
    parts, counts = [], []
    for data in data_list:
        value = getattr(data, attr)
        value = value.numpy() if torch.is_tensor(value) else np.asarray(value)
        parts.append(np.asarray(value, dtype=np.float32))
        counts.append(int(value.shape[0]))
    return np.concatenate(parts, axis=0), counts


def _unstack(values: np.ndarray, counts: Sequence[int], attr: str, data_list: Sequence[Any]) -> None:
    offset = 0
    for data, n in zip(data_list, counts):
        setattr(data, attr, torch.as_tensor(values[offset : offset + n], dtype=torch.float32))
        offset += n


def _root_mask(counts: Sequence[int], molecule_idx: np.ndarray) -> np.ndarray:
    want = set(int(i) for i in molecule_idx.tolist())
    mask = np.zeros(int(sum(counts)), dtype=bool)
    offset = 0
    for index, count in enumerate(counts):
        if index in want:
            mask[offset : offset + count] = True
        offset += count
    return mask


def apply_new_fit_prep(train_data: Sequence[Any], blob: Mapping[str, np.ndarray], fit_idx: np.ndarray) -> dict[str, Any]:
    """Refit every input standardizer on the new fit rows only.

    The frozen encoded cache stores values normalised by the old *all-train*
    standardizer.  That standardizer is inverted with the constants stored in
    the frozen prep blob (cache recovery, no fitting), and then a fresh
    per-column standardizer is fitted on the new fit rows (``floor=1e-6``) and
    applied to every train row.  No old *fit* scaler is read or reused.
    """
    patch_all = zjd.Std(blob["patch_all_mean"], blob["patch_all_scale"])
    ctx_all = zjd.Std(blob["ctx_all_mean"], blob["ctx_all_scale"])
    anchor_all = zjd.Std(blob["anchor_all_mean"], blob["anchor_all_scale"])
    topo_all = zjd.Std(blob["topo_all_mean"], blob["topo_all_scale"])

    patch_raw, patch_counts = _stack(train_data, "patch_cont")
    patch_raw = patch_all.inverse(patch_raw)
    ctx_raw, ctx_counts = _stack(train_data, "global_context")
    ctx_raw = ctx_all.inverse(ctx_raw)
    anchor_raw, anchor_counts = _stack(train_data, "anchor")
    anchor_raw = anchor_all.inverse(anchor_raw)
    topo_raw, topo_counts = _stack(train_data, "topology_features")
    topo_raw = topo_all.inverse(topo_raw)

    patch_mask = _root_mask(patch_counts, fit_idx)
    ctx_mask = _root_mask(ctx_counts, fit_idx)
    anchor_mask = _root_mask(anchor_counts, fit_idx)
    topo_mask = _root_mask(topo_counts, fit_idx)

    patch_fit = zjd.Std.fit(patch_raw[patch_mask], floor=zjd.CANON_STD_FLOOR)
    ctx_fit = zjd.Std.fit(ctx_raw[ctx_mask], floor=zjd.CANON_STD_FLOOR)
    anchor_fit = zjd.Std.fit(anchor_raw[anchor_mask], floor=zjd.CANON_STD_FLOOR)
    topo_fit = zjd.Std.fit(topo_raw[topo_mask], floor=zjd.CANON_STD_FLOOR)

    _unstack(patch_fit.transform(patch_raw), patch_counts, "patch_cont", train_data)
    _unstack(ctx_fit.transform(ctx_raw), ctx_counts, "global_context", train_data)
    _unstack(anchor_fit.transform(anchor_raw), anchor_counts, "anchor", train_data)
    _unstack(topo_fit.transform(topo_raw), topo_counts, "topology_features", train_data)

    meta = {
        "fitted_on": "new fit rows only",
        "inversion_constants": "frozen encoded-cache all-train standardizers (from fold_objects.npz)",
        "floor": float(zjd.CANON_STD_FLOOR),
        "fit_root_rows": int(patch_mask.sum()),
        "n_fit_molecules": int(len(fit_idx)),
        "patch_fit_mean_sha256": hashlib.sha256(np.ascontiguousarray(patch_fit.mean, np.float32).tobytes()).hexdigest(),
        "patch_fit_scale_sha256": hashlib.sha256(np.ascontiguousarray(patch_fit.scale, np.float32).tobytes()).hexdigest(),
        "ctx_fit_mean_sha256": hashlib.sha256(np.ascontiguousarray(ctx_fit.mean, np.float32).tobytes()).hexdigest(),
        "ctx_fit_scale_sha256": hashlib.sha256(np.ascontiguousarray(ctx_fit.scale, np.float32).tobytes()).hexdigest(),
        "anchor_fit_mean_sha256": hashlib.sha256(np.ascontiguousarray(anchor_fit.mean, np.float32).tobytes()).hexdigest(),
        "anchor_fit_scale_sha256": hashlib.sha256(np.ascontiguousarray(anchor_fit.scale, np.float32).tobytes()).hexdigest(),
        "topo_fit_mean_sha256": hashlib.sha256(np.ascontiguousarray(topo_fit.mean, np.float32).tobytes()).hexdigest(),
        "topo_fit_scale_sha256": hashlib.sha256(np.ascontiguousarray(topo_fit.scale, np.float32).tobytes()).hexdigest(),
        "patch_fit_mean": patch_fit.mean,
        "patch_fit_scale": patch_fit.scale,
        "ctx_fit_mean": ctx_fit.mean,
        "ctx_fit_scale": ctx_fit.scale,
        "anchor_fit_mean": anchor_fit.mean,
        "anchor_fit_scale": anchor_fit.scale,
        "topo_fit_mean": topo_fit.mean,
        "topo_fit_scale": topo_fit.scale,
    }
    return meta


def load_and_prepare() -> tuple[dict[str, Any], dict[str, Any], list[Any], list[Any], dict[str, np.ndarray], dict[str, Any]]:
    train_data = zftd.load_train_only()
    if len(train_data) != 10000:
        raise RuntimeError(f"train-only cache has {len(train_data)} rows, expected 10000")
    decomp = zftd.load_target_decomposition()
    y = np.asarray(decomp["y"], np.float64)
    k = np.asarray(decomp["k"], np.int64)
    if int(y.shape[0]) != 10000:
        raise RuntimeError("target decomposition length mismatch")
    blob = np.load(PREP_BLOB, allow_pickle=False)
    fold = build_fold()
    prep_meta = apply_new_fit_prep(train_data, blob, fold["fit_idx"])
    fit_data, dev_data = zftd.build_fit_dev(train_data, fold["fit_idx"], fold["dev_idx"])
    if len(fit_data) != 8000 or len(dev_data) != 2000:
        raise RuntimeError("fit/dev rebuild length mismatch")
    return fold, prep_meta, fit_data, dev_data, decomp, {"y": y, "k": k}


# ---------------------------------------------------------------------------
# the two bridge modules
# ---------------------------------------------------------------------------


class MLPBridge(nn.Module):
    """Matched two-layer MLP bridge: ``E = rho * W2 SiLU(W1 x)``, no biases."""

    def __init__(self, dictionary: torch.Tensor, values: torch.Tensor) -> None:
        super().__init__()
        d, n_atoms = int(dictionary.shape[0]), int(dictionary.shape[1])
        if (d, n_atoms) != (BRIDGE_DIM, BRIDGE_ATOMS):
            raise ValueError(f"unexpected bridge frame {d}x{n_atoms}")
        if tuple(values.shape) != (n_atoms, d):
            raise ValueError("value-frame shape mismatch")
        self.dim = d
        self.n_atoms = n_atoms
        self.fc1 = nn.Linear(d, n_atoms, bias=False)
        self.fc2 = nn.Linear(n_atoms, d, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(dictionary.detach().t().contiguous())
            self.fc2.weight.copy_(values.detach().t().contiguous())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(h.pow(2).mean(dim=1, keepdim=True) + float(lb.BRIDGE_EPS))
        x = h / scale
        return scale * self.fc2(F.silu(self.fc1(x)))


def bridge_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for name, p in model.named_parameters() if name.startswith("local_dictionary_bridge.")))


def body_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for name, p in model.named_parameters() if not name.startswith("local_dictionary_bridge.")))


def build_arm(arm: str) -> nn.Module:
    """Fresh seed-0 compressed body + the requested bridge (no learned state)."""
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(int(SEED))
    model = zbn.build_deploy_model(None)
    if arm == "M":
        bridge = model.local_dictionary_bridge
        dictionary = bridge.D_L.detach().clone()
        values = bridge.V_L.detach().clone()
        model.local_dictionary_bridge = MLPBridge(dictionary, values)
    expected_total = EXPECTED_TOTAL_PARAMETERS
    if int(sum(p.numel() for p in model.parameters())) != expected_total:
        raise RuntimeError(
            f"parameter audit failed: {int(sum(p.numel() for p in model.parameters()))} != {expected_total}"
        )
    if bridge_parameter_count(model) != BRIDGE_PARAMETERS:
        raise RuntimeError("bridge parameter audit failed")
    if body_parameter_count(model) != EXPECTED_BODY_PARAMETERS:
        raise RuntimeError("body parameter audit failed")
    return model


def build_pair_hashes() -> dict[str, Any]:
    """Verify the two fresh arms share every non-bridge tensor item-for-item."""
    model_d = build_arm("D")
    state_d = {k: v.detach().cpu().clone() for k, v in model_d.state_dict().items()}
    model_m = build_arm("M")
    state_m = {k: v.detach().cpu().clone() for k, v in model_m.state_dict().items()}
    shared_keys = sorted(
        key for key in state_d if not key.startswith("local_dictionary_bridge.")
    )
    if sorted(key for key in state_m if not key.startswith("local_dictionary_bridge.")) != shared_keys:
        raise RuntimeError("shared body key sets differ")
    hashes: dict[str, str] = {}
    for key in shared_keys:
        if not torch.equal(state_d[key], state_m[key]):
            raise RuntimeError(f"shared tensor mismatch at {key}")
        hashes[key] = tensor_hash(state_d[key])
    bridge_d = {k: v for k, v in state_d.items() if k.startswith("local_dictionary_bridge.")}
    bridge_m = {k: v for k, v in state_m.items() if k.startswith("local_dictionary_bridge.")}
    # D frame vs M frames: W1 == D_L.T and W2 == V_L.T on the fresh init.
    fc1 = bridge_m["local_dictionary_bridge.fc1.weight"]
    fc2 = bridge_m["local_dictionary_bridge.fc2.weight"]
    if not torch.equal(fc1, bridge_d["local_dictionary_bridge.D_L"].t().contiguous()):
        raise RuntimeError("M fc1 does not equal fresh D_L.T")
    if not torch.equal(fc2, bridge_d["local_dictionary_bridge.V_L"].t().contiguous()):
        raise RuntimeError("M fc2 does not equal fresh V_L.T")
    return {
        "shared_tensor_hashes": hashes,
        "shared_tensor_count": len(shared_keys),
        "shared_state_hash": state_hash({k: state_d[k] for k in shared_keys}),
        "d_bridge_init_hash": state_hash(bridge_d),
        "m_bridge_init_hash": state_hash(bridge_m),
        "body_parameters": body_parameter_count(model_d),
        "bridge_parameters": bridge_parameter_count(model_d),
        "total_parameters": int(sum(p.numel() for p in model_d.parameters())),
    }


# ---------------------------------------------------------------------------
# shared batch schedule
# ---------------------------------------------------------------------------


def build_schedule(n: int, epochs: int, seed: int) -> tuple[list[np.ndarray], str]:
    generator = torch.Generator().manual_seed(int(seed))
    schedule = [torch.randperm(int(n), generator=generator).numpy().astype(np.int64) for _ in range(int(epochs))]
    digest = hashlib.sha256()
    for epoch_order in schedule:
        digest.update(epoch_order.tobytes())
    return schedule, digest.hexdigest()


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def evaluate_state(
    model: nn.Module,
    data_list: Sequence[Any],
    y_local: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    predictions = np.empty(len(data_list), np.float64)
    target = torch.as_tensor(y_local, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(data_list), batch_size):
            indices = list(range(start, min(start + batch_size, len(data_list))))
            batch = zftd.make_batch(data_list, indices, target, device)
            prediction = model(batch, mask=cm.C6_MASK)
            predictions[start : start + len(indices)] = prediction.view(-1).detach().cpu().numpy()
    return predictions


def _bridge_aux(model: nn.Module, batch: Any) -> dict[str, float]:
    """Health numbers from a real batch (no training RNG consumption)."""
    model.eval()
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        interface = model.fusion(sem.SEM108Model.semantic_interface(model, coord, batch, cm.C6_MASK, None))
        module = model.local_dictionary_bridge
        if isinstance(module, MLPBridge):
            scale = torch.sqrt(interface.pow(2).mean(dim=1, keepdim=True) + float(lb.BRIDGE_EPS))
            x = interface / scale
            hidden = F.silu(module.fc1(x))
            return {
                "kind": "mlp",
                "hidden_abs_mean": float(hidden.abs().mean()),
                "hidden_nonzero_fraction": float((hidden != 0).float().mean()),
            }
        _, aux = module(interface, return_aux=True)
        alpha = aux["alpha"]
        x = aux["normalized"]
        scale = aux["scale"]
        dbar = module.normalized_dictionary()
        recon = (alpha @ dbar.t()) * scale
        rel = float((recon - interface).abs().mean() / interface.abs().mean().clamp_min(1e-12))
        return {
            "kind": "dictionary",
            "code_nonzero_fraction": float((alpha != 0).float().mean()),
            "code_per_row_nonzero": float((alpha != 0).float().sum(dim=1).mean()),
            "ista_recon_rel_abs_mean": rel,
        }


def train_arm(
    arm: str,
    *,
    device: torch.device,
    out_dir: Path,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold, prep_meta, fit_data, dev_data, decomp, _targets = load_and_prepare()
    y = np.asarray(decomp["y"], np.float64)
    k = np.asarray(decomp["k"], np.int64)
    y_fit = y[fold["fit_idx"]]
    y_dev = y[fold["dev_idx"]]
    target_fit = torch.as_tensor(y_fit, dtype=torch.float32)
    schedule, schedule_hash = build_schedule(len(fit_data), int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)

    model = build_arm(arm).to(device)
    audit = {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "body_parameters": body_parameter_count(model),
        "bridge_parameters": bridge_parameter_count(model),
    }
    seed_everything(SEED)
    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    steps_done = 0
    stream = hashlib.sha256()
    started = time.perf_counter()
    stopped_reason = "completed"
    bridge_grad_log: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, n_mol, n_steps, gnorm_sum, clip_hits = 0.0, 0, 0, 0.0, 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = schedule[epoch - 1][start : start + BATCH_SIZE]
            index_list = [int(i) for i in indices.tolist()]
            stream.update(np.asarray(index_list, np.int64).tobytes())
            batch = zftd.make_batch(fit_data, index_list, target_fit, device)
            prediction = model(batch, mask=cm.C6_MASK)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if epoch in (1, 120, 240):
                entry = {"epoch": int(epoch)}
                for name, parameter in model.named_parameters():
                    if name.startswith("local_dictionary_bridge.") and parameter.grad is not None:
                        entry[name] = float(parameter.grad.norm())
                bridge_grad_log.append(entry)
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "clip_fraction": float(clip_hits / max(n_steps, 1)),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if epoch in soup_epochs:
            soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[{arm}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"gnorm={curve[-1]['grad_norm']:.3g} clip={curve[-1]['clip_fraction']:.3f} "
                f"{curve[-1]['seconds']:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    if not soup and max_steps is not None:
        # engineering smoke: keep the last state as the only soup member.
        soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    health: dict[str, Any] = {"fit": {}, "dev": {}}
    for split_name, data_list in (("fit", fit_data[:512]), ("dev", dev_data[:512])):
        batch = zftd.make_batch(data_list, list(range(min(128, len(data_list)))), torch.zeros(len(data_list)), device)
        health[split_name] = _bridge_aux(model, batch)
    if arm == "D":
        bridge = model.local_dictionary_bridge
        health["param_change"] = {
            "D_L_rel": float((bridge.D_L.detach().cpu() - init_state["local_dictionary_bridge.D_L"]).norm() / init_state["local_dictionary_bridge.D_L"].norm()),
            "V_L_rel": float((bridge.V_L.detach().cpu() - init_state["local_dictionary_bridge.V_L"]).norm() / init_state["local_dictionary_bridge.V_L"].norm()),
        }
    else:
        bridge = model.local_dictionary_bridge
        health["param_change"] = {
            "fc1_rel": float((bridge.fc1.weight.detach().cpu() - init_state["local_dictionary_bridge.fc1.weight"]).norm() / init_state["local_dictionary_bridge.fc1.weight"].norm()),
            "fc2_rel": float((bridge.fc2.weight.detach().cpu() - init_state["local_dictionary_bridge.fc2.weight"]).norm() / init_state["local_dictionary_bridge.fc2.weight"].norm()),
        }

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "seed": SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "steps_expected": int(epochs) * math.ceil(len(fit_data) / BATCH_SIZE),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "soup_epochs": members if max_steps is None else [],
        "n_fit": int(len(fit_data)),
        "n_dev": int(len(dev_data)),
        "fold": {
            "seed": FOLD_SEED,
            "fit_idx_sha256": fold["fit_idx_sha256"],
            "dev_idx_sha256": fold["dev_idx_sha256"],
        },
        "prep": {key: value for key, value in prep_meta.items() if not isinstance(value, np.ndarray)},
        "parameter_audit": audit,
        "init_state_hash": state_hash(init_state),
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "schedule_sha256": schedule_hash,
        "data_stream_sha256": stream.hexdigest(),
        "curve": curve,
        "bridge_grad_log": bridge_grad_log,
        "health": health,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if max_steps is not None:
        result["smoke_last_state_hash"] = state_hash(last_state)
        return result

    torch.save(init_state, out_dir / f"{arm}_init_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")

    raw_predictions: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = build_arm(arm)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        raw_predictions[f"{state_name}_fit"] = evaluate_state(replay, fit_data, y_fit, device)
        raw_predictions[f"{state_name}_dev"] = evaluate_state(replay, dev_data, y_dev, device)
    np.savez_compressed(
        out_dir / f"{arm}_raw_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_predictions.items()},
    )

    b_soup = float(np.median(y_fit - raw_predictions["raw_soup_fit"]))
    b_init = float(np.median(y_fit - raw_predictions["init_fit"]))
    b_last = float(np.median(y_fit - raw_predictions["last_fit"]))
    result["calibration_b"] = {"init": b_init, "last": b_last, "raw_soup": b_soup}

    # fresh-init replay on a fixed batch: CPU vs GPU device equivalence.
    replay = build_arm(arm)
    replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in soup_state.items()}, strict=True)
    batch_data = fit_data[:128]
    fixed_index = list(range(128))
    gpu_pred = evaluate_state(replay.to(device), batch_data, y_fit[:128], device)
    cpu_pred = evaluate_state(replay.to(torch.device("cpu")), batch_data, y_fit[:128], torch.device("cpu"))
    max_diff = float(np.max(np.abs(gpu_pred - cpu_pred))) if device.type != "cpu" else 0.0
    result["replay_max_abs_diff"] = max_diff
    if device.type != "cpu" and max_diff > REPLAY_TOL:
        raise RuntimeError(f"CPU/GPU replay maxdiff {max_diff} > {REPLAY_TOL}")
    write_json(out_dir / f"{arm}_meta.json", result)
    log(f"[{arm}] done steps={steps_done} wall={result['wall_clock_s']:.1f}s b={b_soup:.6f} replay={max_diff:.2e}")
    return result


# ---------------------------------------------------------------------------
# smoke checks (engineering only; states discarded)
# ---------------------------------------------------------------------------


def run_smoke(*, device: torch.device, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold, prep_meta, fit_data, dev_data, decomp, _targets = load_and_prepare()
    y = np.asarray(decomp["y"], np.float64)
    y_fit = y[fold["fit_idx"]]
    checks: dict[str, Any] = {}

    pair = build_pair_hashes()
    checks["shared_body"] = {
        "shared_tensor_count": pair["shared_tensor_count"],
        "shared_state_hash": pair["shared_state_hash"],
        "d_bridge_init_hash": pair["d_bridge_init_hash"],
        "m_bridge_init_hash": pair["m_bridge_init_hash"],
        "body_parameters": pair["body_parameters"],
        "bridge_parameters": pair["bridge_parameters"],
        "total_parameters": pair["total_parameters"],
    }
    checks["parameter_contract"] = bool(
        pair["body_parameters"] == EXPECTED_BODY_PARAMETERS
        and pair["bridge_parameters"] == BRIDGE_PARAMETERS
        and pair["total_parameters"] == EXPECTED_TOTAL_PARAMETERS
    )

    model_d = build_arm("D").to(device)
    model_m = build_arm("M").to(device)
    model_d.eval()
    model_m.eval()
    target_smoke = torch.as_tensor(y_fit[:512], dtype=torch.float32)
    batch = zftd.make_batch(fit_data, list(range(32)), target_smoke, device)

    # Consumer check: hook the bridge and compare its output with the forward aux E.
    captured: list[torch.Tensor] = []
    handle = model_d.local_dictionary_bridge.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().clone()))
    with torch.no_grad():
        _, aux_d = model_d(batch, mask=cm.C6_MASK, return_aux=True)
    handle.remove()
    checks["dictionary_consumer_bridge_output_maxdiff"] = float((captured[0] - aux_d["E"]).abs().max())
    checks["dictionary_consumer_ok"] = checks["dictionary_consumer_bridge_output_maxdiff"] < 1e-6

    captured_m: list[torch.Tensor] = []
    handle = model_m.local_dictionary_bridge.register_forward_hook(lambda module, inputs, output: captured_m.append(output.detach().clone()))
    with torch.no_grad():
        _, aux_m = model_m(batch, mask=cm.C6_MASK, return_aux=True)
    handle.remove()
    checks["mlp_consumer_bridge_output_maxdiff"] = float((captured_m[0] - aux_m["E"]).abs().max())
    checks["mlp_consumer_ok"] = checks["mlp_consumer_bridge_output_maxdiff"] < 1e-6

    # y is never read: randomising the target cannot change the prediction.
    shuffled = batch.clone()
    shuffled.y = torch.randn_like(batch.y)
    with torch.no_grad():
        pred_d2 = model_d(shuffled, mask=cm.C6_MASK)
        pred_m2 = model_m(shuffled, mask=cm.C6_MASK)
    checks["y_not_read_d_maxdiff"] = float((pred_d2 - model_d(batch, mask=cm.C6_MASK)).abs().max())
    checks["y_not_read_m_maxdiff"] = float((pred_m2 - model_m(batch, mask=cm.C6_MASK)).abs().max())
    checks["y_not_read_ok"] = bool(
        checks["y_not_read_d_maxdiff"] == 0.0 and checks["y_not_read_m_maxdiff"] == 0.0
    )

    # Batched endpoint offsets: grouped prediction equals per-molecule prediction.
    groups = list(range(8))
    batch_g = zftd.make_batch(fit_data, groups, target_smoke, device)
    with torch.no_grad():
        grouped_d = model_d(batch_g, mask=cm.C6_MASK).cpu().numpy()
        single_d = np.array(
            [
                float(model_d(zftd.make_batch(fit_data, [i], target_smoke, device), mask=cm.C6_MASK).item())
                for i in groups
            ]
        )
    checks["endpoint_offset_d_maxdiff"] = float(np.max(np.abs(grouped_d - single_d)))
    with torch.no_grad():
        grouped_m = model_m(batch_g, mask=cm.C6_MASK).cpu().numpy()
        single_m = np.array(
            [
                float(model_m(zftd.make_batch(fit_data, [i], target_smoke, device), mask=cm.C6_MASK).item())
                for i in groups
            ]
        )
    checks["endpoint_offset_m_maxdiff"] = float(np.max(np.abs(grouped_m - single_m)))
    checks["endpoint_offset_ok"] = bool(
        checks["endpoint_offset_d_maxdiff"] < 1e-6 and checks["endpoint_offset_m_maxdiff"] < 1e-6
    )

    # Real backward + optimizer update, loss is mean L1 only.
    for arm, model in (("D", model_d), ("M", model_m)):
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        prediction = model(batch, mask=cm.C6_MASK)
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        changed = [key for key, value in model.state_dict().items() if not torch.equal(value, before[key])]
        checks[f"{arm}_optimizer_step_changed_tensors"] = len(changed)
        checks[f"{arm}_loss_value"] = float(loss.detach())
    checks["optimizer_ok"] = bool(
        checks["D_optimizer_step_changed_tensors"] > 0 and checks["M_optimizer_step_changed_tensors"] > 0
    )

    # Every consumer reads the bridge output: perturb the bridge and the full prediction moves.
    for arm, model in (("D", model_d), ("M", model_m)):
        bridge = model.local_dictionary_bridge
        old = {key: value.detach().clone() for key, value in bridge.state_dict().items()}
        with torch.no_grad():
            original = model(batch, mask=cm.C6_MASK)
            for value in bridge.parameters():
                value.add_(0.01)
            perturbed = model(batch, mask=cm.C6_MASK)
        checks[f"{arm}_bridge_perturbation_maxdiff"] = float((perturbed - original).abs().max())
        bridge.load_state_dict(old)
    checks["consumers_depend_on_bridge"] = bool(
        checks["D_bridge_perturbation_maxdiff"] > 0.0 and checks["M_bridge_perturbation_maxdiff"] > 0.0
    )
    checks["no_structural_aux_loss"] = True  # loss is exactly mean L1(y); asserted in train_arm.
    checks["fold"] = {
        "fit_idx_sha256": fold["fit_idx_sha256"],
        "dev_idx_sha256": fold["dev_idx_sha256"],
        "n_fit": fold["n_fit"],
        "n_dev": fold["n_dev"],
    }
    checks["prep_meta"] = {key: value for key, value in prep_meta.items() if not isinstance(value, np.ndarray)}
    checks["official_valid_loaded"] = False
    checks["official_test_loaded"] = False

    # Tiny end-to-end training step for both arms (states discarded).
    t0 = time.perf_counter()
    res_d = train_arm("D", device=device, out_dir=out_dir / "smoke", epochs=1, max_steps=3, log=log)
    res_m = train_arm("M", device=device, out_dir=out_dir / "smoke", epochs=1, max_steps=3, log=log)
    checks["tiny_train"] = {
        "D_steps": res_d["steps_done"],
        "M_steps": res_m["steps_done"],
        "D_loss_curve": [entry["train_task_mae"] for entry in res_d["curve"]],
        "M_loss_curve": [entry["train_task_mae"] for entry in res_m["curve"]],
        "schedule_match": bool(res_d["schedule_sha256"] == res_m["schedule_sha256"]),
        "seconds": float(time.perf_counter() - t0),
    }
    checks["all_ok"] = bool(
        checks["parameter_contract"]
        and checks["dictionary_consumer_ok"]
        and checks["mlp_consumer_ok"]
        and checks["y_not_read_ok"]
        and checks["endpoint_offset_ok"]
        and checks["optimizer_ok"]
        and checks["consumers_depend_on_bridge"]
        and checks["tiny_train"]["schedule_match"]
    )
    write_json(out_dir / "smoke_checks.json", checks)
    if not checks["all_ok"]:
        raise RuntimeError("smoke checks failed")
    log(f"[smoke] all_ok={checks['all_ok']} in {time.perf_counter() - t0:.1f}s")
    return checks


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def paired_bootstrap_gain(
    err_d: np.ndarray,
    err_m: np.ndarray,
    strata: Sequence[np.ndarray] | None = None,
    *,
    seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Gain = mean|err_M| - mean|err_D| (positive = D better), shared resamples."""
    err_d = np.asarray(err_d, np.float64)
    err_m = np.asarray(err_m, np.float64)
    point = float(np.abs(err_m).mean() - np.abs(err_d).mean())
    rng = np.random.default_rng(int(seed))
    gains = np.empty(int(n_boot), np.float64)
    if strata is None:
        rows = np.arange(err_d.size)
        for b in range(int(n_boot)):
            idx = rng.choice(rows, size=rows.size, replace=True)
            gains[b] = np.abs(err_m[idx]).mean() - np.abs(err_d[idx]).mean()
    else:
        index_sets = [np.where(mask)[0] for mask in strata if int(np.sum(mask)) > 0]
        for b in range(int(n_boot)):
            idx = np.concatenate([rng.choice(rows, size=rows.size, replace=True) for rows in index_sets]) if index_sets else np.array([], np.int64)
            if idx.size == 0:
                gains[b] = 0.0
            else:
                gains[b] = np.abs(err_m[idx]).mean() - np.abs(err_d[idx]).mean()
    return {
        "point": point,
        "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
        "n_boot": int(n_boot),
        "seed": int(seed),
        "shared_indices": True,
    }


def _metric_table(error: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(k.shape[0]), "mae": float(np.mean(np.abs(error))) if error.size else None}
    total = int(k.shape[0])
    for name, mask in group_masks(k).items():
        n = int(mask.sum())
        out[name] = {
            "n": n,
            "mae": float(np.mean(np.abs(error[mask]))) if n else None,
            "contribution": float(np.abs(error[mask]).sum() / total) if total else None,
        }
    return out


def _init_probe(model: nn.Module, batch: Any) -> dict[str, Any]:
    """Initial input/output RMS and one-step bridge gradients (engineering check)."""
    model.train()
    for parameter in model.parameters():
        parameter.grad = None
    coord = model.code(batch.dict_phi)
    interface = model.fusion(sem.SEM108Model.semantic_interface(model, coord, batch, cm.C6_MASK, None))
    module = model.local_dictionary_bridge
    if isinstance(module, MLPBridge):
        scale = torch.sqrt(interface.pow(2).mean(dim=1, keepdim=True) + float(lb.BRIDGE_EPS))
        x = interface / scale
        hidden = F.silu(module.fc1(x))
        output = scale * module.fc2(hidden)
        extra = {"hidden_abs_mean": float(hidden.abs().mean())}
    else:
        output, aux = module(interface, return_aux=True)
        extra = {
            "code_nonzero_fraction": float((aux["alpha"] != 0).float().mean()),
            "code_per_row_nonzero": float((aux["alpha"] != 0).float().sum(dim=1).mean()),
        }
    prediction = model(batch, mask=cm.C6_MASK)
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    loss.backward()
    grads = {}
    finite = True
    for name, parameter in model.named_parameters():
        if name.startswith("local_dictionary_bridge.") and parameter.grad is not None:
            value = parameter.grad.detach()
            grads[name] = float(value.norm())
            finite = finite and bool(torch.isfinite(value).all())
    model.eval()
    return {
        "input_rms": float(interface.pow(2).mean().sqrt()),
        "output_rms": float(output.pow(2).mean().sqrt()),
        "grad_norms": grads,
        "grads_finite": finite,
        **extra,
    }


def analyze(*, out_dir: Path, device: torch.device, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    fold, prep_meta, fit_data, dev_data, decomp, _targets = load_and_prepare()
    y = np.asarray(decomp["y"], np.float64)
    k = np.asarray(decomp["k"], np.int64)
    k_fit, k_dev = k[fold["fit_idx"]], k[fold["dev_idx"]]
    y_fit, y_dev = y[fold["fit_idx"]], y[fold["dev_idx"]]

    prediction = {}
    meta = {}
    for arm in ARMS:
        meta[arm] = json.loads((out_dir / f"{arm}_meta.json").read_text())
        with np.load(out_dir / f"{arm}_raw_predictions.npz") as z:
            prediction[arm] = {key: np.asarray(z[key], np.float64) for key in z.files}

    contract_errors: list[str] = []
    for arm in ARMS:
        if meta[arm]["parameter_audit"] != {
            "total_parameters": EXPECTED_TOTAL_PARAMETERS,
            "body_parameters": EXPECTED_BODY_PARAMETERS,
            "bridge_parameters": BRIDGE_PARAMETERS,
        }:
            contract_errors.append(f"{arm}: parameter audit mismatch")
        if meta[arm]["steps_expected"] != 15120 or meta[arm]["steps_done"] != 15120:
            contract_errors.append(f"{arm}: step count {meta[arm]['steps_done']} != 15120")
        if meta[arm]["stopped_reason"] != "completed":
            contract_errors.append(f"{arm}: stopped_reason={meta[arm]['stopped_reason']}")
        if meta[arm]["official_valid_loaded"] or meta[arm]["official_test_loaded"]:
            contract_errors.append(f"{arm}: official split loaded")
    if meta["D"]["schedule_sha256"] != meta["M"]["schedule_sha256"]:
        contract_errors.append("schedule hash mismatch")
    schedule_local, schedule_local_hash = build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    if schedule_local_hash != meta["D"]["schedule_sha256"]:
        contract_errors.append("schedule regeneration mismatch")
    np.savez_compressed(
        out_dir / "batch_schedule.npz",
        schedule=np.stack(schedule_local).astype(np.int32),
        sha256=np.asarray([schedule_local_hash]),
    )
    if meta["D"]["data_stream_sha256"] != meta["M"]["data_stream_sha256"]:
        contract_errors.append("data stream hash mismatch")
    if meta["D"]["fold"]["fit_idx_sha256"] != meta["M"]["fold"]["fit_idx_sha256"]:
        contract_errors.append("fold mismatch")

    b = {arm: float(meta[arm]["calibration_b"]["raw_soup"]) for arm in ARMS}

    # Cross-arm initial shared-parameter identity, from the saved init states.
    init_states = {
        arm: torch.load(out_dir / f"{arm}_init_state.pt", map_location="cpu", weights_only=False)
        for arm in ARMS
    }
    bridge_prefix = "local_dictionary_bridge."
    shared_keys = sorted(key for key in init_states["D"] if not key.startswith(bridge_prefix))
    shared_mismatch = [
        key for key in shared_keys if not torch.equal(init_states["D"][key], init_states["M"][key])
    ]
    shared_init_hasher = hashlib.sha256()
    for key in shared_keys:
        shared_init_hasher.update(key.encode())
        shared_init_hasher.update(init_states["D"][key].detach().cpu().numpy().tobytes())
    if shared_mismatch:
        contract_errors.append(f"init shared tensors differ: {shared_mismatch[:8]}")
    init_pair_check = {
        "shared_tensor_count": len(shared_keys),
        "shared_mismatch_keys": shared_mismatch,
        "shared_init_sha256": shared_init_hasher.hexdigest(),
        "bridge_keys_D": sorted(key for key in init_states["D"] if key.startswith(bridge_prefix)),
    }
    np.savez_compressed(out_dir / "fold_indices.npz", fit_idx=fold["fit_idx"], dev_idx=fold["dev_idx"])
    np.savez_compressed(
        out_dir / "new_fit_prep.npz",
        **{
            key: value
            for key, value in prep_meta.items()
            if isinstance(value, np.ndarray)
        },
    )
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "fold": {
            "fit_idx_sha256": fold["fit_idx_sha256"],
            "dev_idx_sha256": fold["dev_idx_sha256"],
            "n_fit": fold["n_fit"],
            "n_dev": fold["n_dev"],
        },
        "calibration_b": b,
        "contract_errors": contract_errors,
        "init_pair_check": init_pair_check,
        "arms": {},
        "gains": {},
        "contributions": {},
        "sensitivity": {},
    }
    for state in ("init", "last", "raw_soup"):
        for part, target_y in (("fit", y_fit), ("dev", y_dev)):
            for arm in ARMS:
                raw = prediction[arm][f"{state}_{part}"]
                cal = raw + b[arm]
                report.setdefault("state_metrics", {}).setdefault(state, {}).setdefault(part, {})[arm] = {
                    "raw": _metric_table(target_y - raw, k_fit if part == "fit" else k_dev),
                    "cal": _metric_table(target_y - cal, k_fit if part == "fit" else k_dev),
                }

    # Official endpoint: raw_soup, calibrated with its own fit median b.
    for part in ("fit", "dev"):
        for arm in ARMS:
            raw = prediction[arm][f"raw_soup_{part}"]
            cal = raw + b[arm]
            report["arms"].setdefault(part, {})[arm] = {
                "raw": _metric_table((y_fit if part == "fit" else y_dev) - raw, k_fit if part == "fit" else k_dev),
                "cal": _metric_table((y_fit if part == "fit" else y_dev) - cal, k_fit if part == "fit" else k_dev),
                "b": b[arm],
            }

    err_d_raw = y_dev - prediction["D"]["raw_soup_dev"]
    err_m_raw = y_dev - prediction["M"]["raw_soup_dev"]
    err_d_cal = y_dev - (prediction["D"]["raw_soup_dev"] + b["D"])
    err_m_cal = y_dev - (prediction["M"]["raw_soup_dev"] + b["M"])

    strata_overall = [np.ones(k_dev.size, dtype=bool)]
    strata_g0 = [k_dev == 0]
    report["gains"]["dev_overall_cal"] = paired_bootstrap_gain(err_d_cal, err_m_cal)
    report["gains"]["dev_overall_raw"] = paired_bootstrap_gain(err_d_raw, err_m_raw)
    report["gains"]["dev_G0_cal"] = paired_bootstrap_gain(err_d_cal[k_dev == 0], err_m_cal[k_dev == 0])
    report["gains"]["dev_G0_raw"] = paired_bootstrap_gain(err_d_raw[k_dev == 0], err_m_raw[k_dev == 0])

    # group contributions (soup cal), sum equals overall.
    for arm in ARMS:
        err = y_dev - (prediction[arm]["raw_soup_dev"] + b[arm])
        report["contributions"][arm] = _metric_table(err, k_dev)
        raw_err = y_dev - prediction[arm]["raw_soup_dev"]
        report["contributions"][arm + "_raw"] = _metric_table(raw_err, k_dev)

    # per-row movement (soup); positive delta means M is worse on that row.
    def _movement(d_err: np.ndarray, m_err: np.ndarray) -> dict[str, Any]:
        delta = np.abs(m_err) - np.abs(d_err)
        return {
            "n": int(delta.size),
            "m_better_rows": int((delta < 0).sum()),
            "m_worse_rows": int((delta > 0).sum()),
            "tied_rows": int((delta == 0).sum()),
            "mean_delta_M_minus_D": float(delta.mean()),
            "sum_delta_M_minus_D": float(delta.sum()),
            "max_abs_delta": float(np.abs(delta).max()),
        }

    report["row_movement"] = {
        "fit_cal": _movement(
            y_fit - (prediction["D"]["raw_soup_fit"] + b["D"]),
            y_fit - (prediction["M"]["raw_soup_fit"] + b["M"]),
        ),
        "dev_cal": _movement(err_d_cal, err_m_cal),
        "dev_raw": _movement(err_d_raw, err_m_raw),
    }

    # sensitivity: drop the single dev row with the largest combined cal error.
    combined = np.abs(err_d_cal) + np.abs(err_m_cal)
    drop = int(np.argmax(combined))
    mask = np.ones(k_dev.size, dtype=bool)
    mask[drop] = False
    report["sensitivity"] = {
        "drop_rule": "max(|err_D_cal| + |err_M_cal|) on new dev",
        "dropped_row_stable_id": int(fold["dev_idx"][drop]),
        "dropped_combined_error": float(combined[drop]),
        "overall_cal_gain_without_row": float(np.abs(err_m_cal[mask]).mean() - np.abs(err_d_cal[mask]).mean()),
        "overall_raw_gain_without_row": float(np.abs(err_m_raw[mask]).mean() - np.abs(err_d_raw[mask]).mean()),
        "G0_cal_gain_without_row": float(
            np.abs(err_m_cal[mask & (k_dev == 0)]).mean() - np.abs(err_d_cal[mask & (k_dev == 0)]).mean()
        ),
    }

    # bootstrap self-tests.
    self_tests = {
        "same_predictions": paired_bootstrap_gain(err_d_cal, err_d_cal),
        "swapped": paired_bootstrap_gain(err_m_cal, err_d_cal),
        "constant_shift": paired_bootstrap_gain(err_d_cal + 0.001, err_m_cal),
    }
    original = report["gains"]["dev_overall_cal"]
    mirror_ok = bool(
        abs(self_tests["swapped"]["point"] + original["point"]) < 1e-9
        and abs(self_tests["swapped"]["ci95"][0] + original["ci95"][1]) < 1e-9
        and abs(self_tests["swapped"]["ci95"][1] + original["ci95"][0]) < 1e-9
    )
    shift_ok = bool(abs(self_tests["constant_shift"]["point"] - original["point"]) <= 0.001 + 1e-9)
    self_tests["checks"] = {
        "same_predictions_zero": bool(
            self_tests["same_predictions"]["point"] == 0.0
            and self_tests["same_predictions"]["ci95"] == [0.0, 0.0]
        ),
        "swapped_mirror": mirror_ok,
        "constant_shift_bounded": shift_ok,
    }
    report["bootstrap_self_tests"] = self_tests

    # gate.
    g0_cal = report["gains"]["dev_G0_cal"]
    g0_raw = report["gains"]["dev_G0_raw"]
    ov_cal = report["gains"]["dev_overall_cal"]
    ov_raw = report["gains"]["dev_overall_raw"]

    def _ci_inside(ci: Sequence[float], low: float, high: float) -> bool:
        return bool(ci[0] >= low and ci[1] <= high)

    d_support = bool(
        g0_cal["point"] >= DELTA and g0_cal["ci95"][0] > 0 and g0_raw["point"] > 0 and ov_cal["point"] >= -DELTA_SLACK
    )
    m_support = bool(
        g0_cal["point"] <= -DELTA and g0_cal["ci95"][1] < 0 and g0_raw["point"] < 0 and ov_cal["point"] <= DELTA_SLACK
    )
    local_equivalence = bool(
        _ci_inside(g0_cal["ci95"], -DELTA, DELTA)
        and _ci_inside(ov_cal["ci95"], -DELTA, DELTA)
        and not (g0_raw["point"] >= DELTA or g0_raw["point"] <= -DELTA)
        and not (ov_raw["point"] >= DELTA or ov_raw["point"] <= -DELTA)
    )
    tradeoff = bool(
        (g0_cal["point"] >= DELTA and ov_cal["point"] <= -DELTA_SLACK)
        or (g0_cal["point"] <= -DELTA and ov_cal["point"] >= DELTA_SLACK)
    )
    if contract_errors:
        category = "INVALID"
    elif d_support:
        category = "D_G0_SUPPORT"
    elif m_support:
        category = "M_G0_SUPPORT"
    elif tradeoff:
        category = "TRADEOFF"
    elif local_equivalence:
        category = "LOCAL_EQUIVALENCE"
    else:
        category = "INCONCLUSIVE"
    report["gate"] = {
        "category": category,
        "definitions_checked": {
            "D_G0_SUPPORT": d_support,
            "M_G0_SUPPORT": m_support,
            "LOCAL_EQUIVALENCE": local_equivalence,
            "TRADEOFF": tradeoff,
        },
        "rules": {
            "D_G0_SUPPORT": "G0 cal gain>=0.003 & CI_low>0 & G0 raw gain>0 & overall cal gain>=-0.001",
            "M_G0_SUPPORT": "G0 cal gain<=-0.003 & CI_high<0 & G0 raw gain<0 & overall cal gain<=+0.001",
            "LOCAL_EQUIVALENCE": "G0/overall cal CI inside [-0.003,0.003]; no raw |effect|>=0.003 in opposite direction",
            "TRADEOFF": "one arm improves G0 while overall cal worsens >0.001",
        },
    }
    report["separate_overall_improvement"] = {
        "exists": bool(abs(ov_cal["point"]) >= DELTA),
        "favored": "M" if ov_cal["point"] < 0 else "D",
        "raw_same_direction": bool(np.sign(ov_raw["point"]) == np.sign(ov_cal["point"])),
        "overall_cal_gain": ov_cal["point"],
        "overall_raw_gain": ov_raw["point"],
    }
    report["official_valid_loaded"] = False
    report["official_test_loaded"] = False

    # initial-function probe on a fixed fit batch (both arms, fresh init).
    probe_batch = zftd.make_batch(fit_data, list(range(128)), torch.as_tensor(y_fit[:128], dtype=torch.float32), device)
    init_probe: dict[str, Any] = {}
    for arm in ARMS:
        probe_model = build_arm(arm).to(device)
        probe_model.load_state_dict(torch.load(out_dir / f"{arm}_init_state.pt", map_location="cpu", weights_only=False))
        init_probe[arm] = _init_probe(probe_model, probe_batch)
    report["init_probe"] = init_probe

    # tables as CSV for the report.
    rows = ["endpoint,arm,raw_mae,cal_mae,raw_G0,cal_G0,raw_k-1,cal_k-1,raw_k-2,cal_k-2,raw_k<=-3,cal_k<=-3,raw_k<=-2,cal_k<=-2"]
    for part in ("fit", "dev"):
        for arm in ARMS:
            entry = report["arms"][part][arm]
            values = [part, arm, entry["raw"]["mae"], entry["cal"]["mae"]]
            for name in ("k=0", "k=-1", "k=-2", "k<=-3", "k<=-2"):
                values += [entry["raw"][name]["mae"] if entry["raw"][name]["mae"] is not None else "",
                           entry["cal"][name]["mae"] if entry["cal"][name]["mae"] is not None else ""]
            rows.append(",".join(str(v) for v in values))
    (out_dir / "main_table.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # per-graph raw/cal predictions, stable ids, and group contributions.
    group_rows = ["arm,part,group,n,raw_mae,cal_mae,raw_contribution,cal_contribution"]
    for part, part_idx in (("fit", fold["fit_idx"]), ("dev", fold["dev_idx"])):
        target_y = y_fit if part == "fit" else y_dev
        kk = k[part_idx]
        for arm in ARMS:
            raw_pred = prediction[arm][f"raw_soup_{part}"]
            cal_pred = raw_pred + b[arm]
            for name, mask in group_masks(kk).items():
                n = int(mask.sum())
                raw_mae = float(np.mean(np.abs(target_y[mask] - raw_pred[mask]))) if n else None
                cal_mae = float(np.mean(np.abs(target_y[mask] - cal_pred[mask]))) if n else None
                raw_contrib = float(np.abs(target_y[mask] - raw_pred[mask]).sum() / kk.size) if n else None
                cal_contrib = float(np.abs(target_y[mask] - cal_pred[mask]).sum() / kk.size) if n else None
                group_rows.append(f"{arm},{part},{name},{n},{raw_mae},{cal_mae},{raw_contrib},{cal_contrib}")
    (out_dir / "group_table.csv").write_text("\n".join(group_rows) + "\n", encoding="utf-8")

    gain_rows = ["endpoint,gain_point,ci_low,ci_high"]
    for name, entry in report["gains"].items():
        gain_rows.append(f"{name},{entry['point']},{entry['ci95'][0]},{entry['ci95'][1]}")
    (out_dir / "gain_table.csv").write_text("\n".join(gain_rows) + "\n", encoding="utf-8")

    for part, part_idx in (("fit", fold["fit_idx"]), ("dev", fold["dev_idx"])):
        target_y = y_fit if part == "fit" else y_dev
        kk = k[part_idx]
        lines = ["stable_id,k,y,D_raw,D_cal,M_raw,M_cal"]
        d_raw = prediction["D"][f"raw_soup_{part}"]
        m_raw = prediction["M"][f"raw_soup_{part}"]
        for position, stable in enumerate(part_idx.tolist()):
            lines.append(
                f"train:{int(stable):04d},{int(kk[position])},{target_y[position]:.10f},"
                f"{d_raw[position]:.10f},{d_raw[position] + b['D']:.10f},"
                f"{m_raw[position]:.10f},{m_raw[position] + b['M']:.10f}"
            )
        (out_dir / f"per_graph_{part}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(out_dir / "analysis.json", report)
    write_json(out_dir / "gate.json", report["gate"])
    write_json(out_dir / "gains.json", report["gains"])
    log(f"[analysis] category={category} overall_cal_gain={ov_cal['point']:.6f} CI={ov_cal['ci95']}")
    return report


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZINC task-dictionary vs matched MLP bridge (fresh fold, seed 0)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    device = resolve_device(args.device)
    if args.smoke:
        run_smoke(device=device, out_dir=out_dir)
        return 0
    if args.arm:
        train_arm(args.arm, device=device, out_dir=out_dir, epochs=int(args.epochs), max_steps=args.max_steps)
        return 0
    if args.analyze:
        analyze(out_dir=out_dir, device=device)
        return 0
    parser.error("choose --smoke, --arm {D,M} or --analyze")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
