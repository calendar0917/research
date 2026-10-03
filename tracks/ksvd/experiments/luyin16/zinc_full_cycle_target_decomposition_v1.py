"""ZINC Full — cycle-target decomposition diagnostic (round v1).

Question
--------
If the canonical ``Full`` (``e2e_dictenv_scale_v1``, 408,651 params) is **not**
required to learn the rare cycle penalty at the same time as the rest of the
chemical objective, do the representation and the static composer generalise
better on the remaining chemical targets?

Two paired arms, identical architecture and identical fresh seed-specific init:

* ``Y`` (control)  — output ``f(x)``, task loss ``L1(f(x), y)``,
  evaluation prediction ``p_Y = f(x)``.
* ``O`` (oracle diagnostic) — output ``h(x)``, task loss ``L1(h(x), g)`` with
  ``g = y - c``, evaluation prediction ``p_O = h(x) + c``.

``c`` is the label-derived cycle component (``label_cycle_component`` from the
frozen ``zinc_long_cycle_audit`` decomposition).  It is **never** a model input;
it is only used to build the ``O`` supervision target and, at diagnostic
evaluation time, added back externally.  ``O`` is therefore an oracle
diagnostic, not a deployable model.

Both arms keep the full canonical recipe: Sem108, structural dictionary, task
dictionary, structural-semantic binding, fusion, static unary/pair composer,
topology25 -> 8, reader, C6 mask; the complete loss stays
``task L1 + cm.H1_LAMBDA * structural reconstruction``.  No message passing,
no transformer, no scheduler/AMP/DDP.

Train-only: only ``encoded_train.pt`` + the train env cache are loaded.  The
official ZINC test split is never instantiated and the official validation split
is never loaded.  The 2000-row dev diagnostic is drawn from the official train
split.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp

PROTOCOL_VERSION = "zinc-full-cycle-target-decomposition-v1"

RESULTS_DIR = zjd.TRACK_ROOT / "results/zinc_full_cycle_target_decomposition_v1"
TARGET_DECOMP_PATH = RESULTS_DIR / "target_decomposition.npz"
PREP_BLOB = zjd.PREP_DIR / "fold_objects.npz"
CYCLE_LABEL_CSV = zjd.CYCLE / "train_cycle_audit_label.csv"
REFINE_JSON = zjd.CYCLE / "stage_refine.json"
HANDOFF_TRAIN = zjd.HANDOFF / "train.npz"

#: frozen recipe (canonical Full), never selected on dev.
SEEDS = (0, 1)
#: the canonical Full's own task-path widening seed; fixed and recorded.
SCALE_SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
TRAIN_SHUFFLE_OFFSET = 101
FULL_PARAMETERS = 408_651

ARMS = ("Y", "O")
GROUP_NAMES = {"k0": "k=0", "k-1": "k=-1", "kle-2": "k<=-2"}
SEVERE_MAX = -2
BOOT_SEED = 20261003
N_BOOT = 1000
REPLAY_TOL = 2.0e-6

#: pre-registered route gate.
GATE_MEAN_GAIN = 0.003
GATE_G0_WORSEN_MAX = 0.001
GATE_G0_MEAN_GAIN = 0.002


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key in ("cuda", "cuda:0", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device requests CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    raise ValueError(f"unsupported device {name!r}")


def parameter_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# train-only data loading
# ---------------------------------------------------------------------------


def load_train_only() -> list[Any]:
    """Load only the official-train encoded cache + the train env cache.

    ``p1run.load_split("train")`` also deserialises ``encoded_valid.pt``; it is
    deliberately not called.
    """
    train = list(torch.load(sdp.CACHE_DIR / "encoded_train.pt", map_location="cpu", weights_only=False))
    p1run.attach_env(train, "train")
    return train


def apply_prep_train_only(train_data: Sequence[Any], blob: Mapping[str, Any]) -> np.ndarray:
    """Apply the frozen 8000-fit prep blob to the train list only.

    Byte-for-byte the train branch of ``zinc_joint_dictionary_decision_v1
    .apply_prep_blob``; the official-valid branch is not executed.  Returns the
    all-train normalised X175 matrix (only used for the frozen-hash check).
    """
    patch_all = zjd.Std(blob["patch_all_mean"], blob["patch_all_scale"])
    ctx_all = zjd.Std(blob["ctx_all_mean"], blob["ctx_all_scale"])
    anchor_all = zjd.Std(blob["anchor_all_mean"], blob["anchor_all_scale"])
    topo_all = zjd.Std(blob["topo_all_mean"], blob["topo_all_scale"])
    patch_fit = zjd.Std(blob["patch_fit_mean"], blob["patch_fit_scale"])
    ctx_fit = zjd.Std(blob["ctx_fit_mean"], blob["ctx_fit_scale"])
    topo_fit = zjd.Std(blob["topo_fit_mean"], blob["topo_fit_scale"])
    anchor_fit = zjd.Std(blob["anchor_fit_mean"], blob["anchor_fit_scale"])
    x_mean = np.asarray(blob["x_mean"], np.float32)
    x_std = np.asarray(blob["x_std"], np.float32)
    block_scale = np.asarray(blob["block_scale"], np.float32)
    blocks = [
        (zjd.PHI_BLOCK[0], zjd.PHI_BLOCK[1]),
        (zjd.SEM_BLOCK[0], zjd.SEM_BLOCK[1]),
        (zjd.SIZE_BLOCK[0], zjd.SIZE_BLOCK[1]),
    ]

    p, pc = zjd._stack(train_data, "patch_cont")
    c, cc = zjd._stack(train_data, "global_context")
    a, ac = zjd._stack(train_data, "anchor")
    t, tc = zjd._stack(train_data, "topology_features")
    zjd._unstack(patch_fit.transform(patch_all.inverse(p)), pc, "patch_cont", train_data)
    zjd._unstack(ctx_fit.transform(ctx_all.inverse(c)), cc, "global_context", train_data)
    zjd._unstack(anchor_fit.transform(anchor_all.inverse(a)), ac, "anchor", train_data)
    zjd._unstack(topo_fit.transform(topo_all.inverse(t)), tc, "topology_features", train_data)

    p, _ = zjd._stack(train_data, "patch_cont")
    p_raw = patch_all.inverse(p)
    a, _ = zjd._stack(train_data, "anchor")
    a_raw = anchor_all.inverse(a)
    phi = np.concatenate([d.dict_phi.numpy() for d in train_data], 0)
    x = np.concatenate([phi[:, :65], p_raw[:, :108], a_raw[:, 60:62]], axis=1).astype(np.float32)
    z = (x - x_mean) / x_std
    out = z.copy()
    for (lo, hi), s in zip(blocks, block_scale.tolist()):
        out[:, lo:hi] = z[:, lo:hi] / s
    out = out.astype(np.float32)
    offset = 0
    for d in train_data:
        n = int(d.dict_phi.shape[0])
        d.x175_raw = torch.as_tensor(x[offset : offset + n], dtype=torch.float32)
        d.x175 = torch.as_tensor(out[offset : offset + n], dtype=torch.float32)
        offset += n
    return out


# ---------------------------------------------------------------------------
# frozen target decomposition (label-derived oracle c)
# ---------------------------------------------------------------------------


def build_target_decomposition() -> dict[str, Any]:
    """Rebuild ``y``, ``c``, ``g``, ``k`` positionally from the frozen sources."""
    import pandas as pd

    lab = pd.read_csv(CYCLE_LABEL_CSV)
    lab["molecule_id"] = lab["molecule_id"].astype(str)
    handoff = np.load(HANDOFF_TRAIN, allow_pickle=True)
    ids = np.asarray(handoff["ids"]).astype(str)
    y_handoff = np.asarray(handoff["y"], np.float64)
    gid = np.asarray(handoff["canonical_group_id"], np.int64)

    n = int(len(y_handoff))
    subset = lab["subset_index"].to_numpy(np.int64)
    checks: dict[str, Any] = {
        "n_rows": n,
        "label_rows": int(len(lab)),
        "handoff_rows": n,
        "ids_unique": bool(len(set(ids.tolist())) == n),
        "label_ids_unique": bool(lab["molecule_id"].is_unique),
        "subset_index_is_arange": bool(np.array_equal(subset, np.arange(n))),
        "handoff_ids_are_positional": bool(list(ids) == [str(i) for i in range(n)]),
        "official_test_loaded": False,
    }
    if not (checks["subset_index_is_arange"] and checks["handoff_ids_are_positional"]):
        raise RuntimeError("label/handoff rows are not in positional order")

    y_csv = lab["target"].to_numpy(np.float64)
    c = lab["label_cycle_component"].to_numpy(np.float64)
    g_csv = lab["y_without_cycle_label"].to_numpy(np.float64)
    k = np.round(lab["label_effective_cycle_snapped"].to_numpy(np.float64)).astype(np.int64)
    snapped = lab["label_effective_cycle_snapped"].to_numpy(np.float64)

    refine = json.loads(REFINE_JSON.read_text())
    mu_c = float(refine["constants_fitted"]["mu_cycle"])
    sigma_c = float(refine["constants_fitted"]["sigma_cycle"])
    c_expected = (snapped - mu_c) / sigma_c

    checks.update(
        {
            "y_handoff_equals_csv_max_abs": float(np.max(np.abs(y_handoff - y_csv))),
            "c_finite": bool(np.isfinite(c).all()),
            "g_csv_finite": bool(np.isfinite(g_csv).all()),
            "k_finite": bool(np.isfinite(k).all()),
            "c_equals_label_formula_max_abs": float(np.max(np.abs(c - c_expected))),
            "mu_cycle": mu_c,
            "sigma_cycle": sigma_c,
        }
    )
    y = y_handoff.copy()
    g = y - c
    checks["y_minus_c_equals_g_csv_max_abs"] = float(np.max(np.abs(g - g_csv)))
    checks["k_group_counts"] = {
        "k0": int((k == 0).sum()),
        "k-1": int((k == -1).sum()),
        "kle-2": int((k <= -2).sum()),
    }
    checks["c_summary"] = {
        "mean": float(c.mean()),
        "std": float(c.std()),
        "min": float(c.min()),
        "max": float(c.max()),
        "n_negative": int((c < 0).sum()),
    }
    checks["g_summary"] = {"mean": float(g.mean()), "std": float(g.std()),
                           "min": float(g.min()), "max": float(g.max())}
    checks["y_summary"] = {"mean": float(y.mean()), "std": float(y.std()),
                           "min": float(y.min()), "max": float(y.max())}

    payload = {
        "y": y.astype(np.float64),
        "c": c.astype(np.float64),
        "g": g.astype(np.float64),
        "k": k.astype(np.int64),
        "gid": gid.astype(np.int64),
        "ids": ids,
    }
    return {"arrays": payload, "checks": checks}


def save_target_decomposition(decomp: Mapping[str, Any]) -> None:
    a = decomp["arrays"]
    TARGET_DECOMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TARGET_DECOMP_PATH,
        y=np.asarray(a["y"], np.float64),
        c=np.asarray(a["c"], np.float64),
        g=np.asarray(a["g"], np.float64),
        k=np.asarray(a["k"], np.int64),
        gid=np.asarray(a["gid"], np.int64),
    )


def load_target_decomposition() -> dict[str, np.ndarray]:
    with np.load(TARGET_DECOMP_PATH, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


# ---------------------------------------------------------------------------
# fold objects / model construction
# ---------------------------------------------------------------------------


def load_prep_blob() -> dict[str, np.ndarray]:
    with np.load(PREP_BLOB, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def build_full_model(blob: Mapping[str, Any], seed: int) -> torch.nn.Module:
    """One canonical fresh Full: 408,651 params, ``scale_seed = 0``."""
    model = sc.build_scale_model(
        np.asarray(blob["D_fit"], np.float32),
        int(seed),
        zjd.fold_subspace(blob),
        sc.FULL,
        scale_seed=SCALE_SEED,
    )
    audit = sc.scale_parameter_audit(model)
    if int(audit["actual_parameters"]) != FULL_PARAMETERS or not audit["parameter_exact"]:
        raise RuntimeError(f"canonical Full parameter audit failed: {audit}")
    return model


def build_fit_dev(train_data: Sequence[Any], fit_idx: np.ndarray, dev_idx: np.ndarray):
    fit_set = set(int(i) for i in np.asarray(fit_idx).tolist())
    dev_set = set(int(i) for i in np.asarray(dev_idx).tolist())
    fit = [d for i, d in enumerate(train_data) if i in fit_set]
    dev = [d for i, d in enumerate(train_data) if i in dev_set]
    if len(fit) != len(fit_idx) or len(dev) != len(dev_idx):
        raise RuntimeError("fit/dev rebuild length mismatch")
    return fit, dev


# ---------------------------------------------------------------------------
# batching (private generators only; never the DataLoader global stream)
# ---------------------------------------------------------------------------


def epoch_batches(n: int, batch_size: int, generator: torch.Generator, shuffle: bool) -> list[list[int]]:
    if shuffle:
        order = torch.randperm(int(n), generator=generator).tolist()
    else:
        order = list(range(int(n)))
    return [order[start : start + batch_size] for start in range(0, int(n), batch_size)]


def make_batch(data_list: Sequence[Any], indices: Sequence[int], targets: torch.Tensor, device: torch.device):
    batch = p1.env_collate([data_list[i] for i in indices])
    batch.y = targets[torch.as_tensor(list(indices), dtype=torch.long)].view(-1).clone()
    return batch.to(device)


# ---------------------------------------------------------------------------
# training one arm
# ---------------------------------------------------------------------------


def train_arm(
    arm: str,
    seed: int,
    blob: Mapping[str, Any],
    fit_data: Sequence[Any],
    dev_data: Sequence[Any],
    decomp: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    out_dir: Path,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    save_states: bool = True,
    log: Any = print,
) -> dict[str, Any]:
    assert arm in ARMS, arm
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    y_all = np.asarray(decomp["y"], np.float64)
    g_all = np.asarray(decomp["g"], np.float64)
    c_all = np.asarray(decomp["c"], np.float64)
    y_fit_np = y_all[fit_idx]
    g_fit_np = g_all[fit_idx]
    c_fit_np = c_all[fit_idx]
    # every target tensor is *local to its list position*, never a global index.
    target_fit = torch.as_tensor(y_fit_np if arm == "Y" else g_fit_np, dtype=torch.float32)
    dev_y_local = torch.as_tensor(y_all[dev_idx], dtype=torch.float32)
    dev_g_local = torch.as_tensor(g_all[dev_idx], dtype=torch.float32)
    y_t = torch.as_tensor(y_all.astype(np.float32))
    g_t = torch.as_tensor(g_all.astype(np.float32))

    seed_everything(seed)
    model = build_full_model(blob, seed).to(device)
    init_hash = parameter_state_hash(model)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    soup_epochs = set(range(max(1, int(epochs) - 4), int(epochs) + 1))
    curve: list[dict[str, Any]] = []
    epoch_seconds: list[float] = []
    grad_log: list[dict[str, Any]] = []
    peak_memory = 0.0
    steps_done = 0
    started = time.perf_counter()
    stopped_reason = "completed"

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, rec_sum, n_mol, n_steps, gnorm_sum = 0.0, 0.0, 0, 0, 0.0
        for indices in epoch_batches(len(fit_data), BATCH_SIZE, train_gen, True):
            batch = make_batch(fit_data, indices, target_fit, device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = task + float(cm.H1_LAMBDA) * rec
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(rec.detach())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            if len(grad_log) < 3 and epoch == 1:
                grad_log.append({"epoch": 1, "step": n_steps, "grad_norm": total_norm})
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        epoch_seconds.append(float(time.perf_counter() - epoch_started))
        with torch.no_grad():
            state = rng_state()
            model.eval()
            preds, targets = [], []
            for indices in epoch_batches(len(dev_data), BATCH_SIZE, train_gen, False):
                batch = make_batch(dev_data, indices, dev_y_local, device)
                preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
                targets.append(batch.y.view(-1).detach().cpu())
            restore_rng(state)
        dev_mae = float((torch.cat(preds) - torch.cat(targets)).abs().mean())
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "train_rec": float(rec_sum / max(n_steps, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "dev_mae_y": dev_mae,
                "seconds": epoch_seconds[-1],
            }
        )
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if device.type == "cuda":
            peak_memory = max(peak_memory, float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)))
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[{arm}/s{seed}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"dev_y={dev_mae:.6f} rec={curve[-1]['train_rec']:.4g} "
                f"gnorm={curve[-1]['grad_norm']:.3g} {epoch_seconds[-1]:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "seed": int(seed),
        "scale_seed": SCALE_SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "h1_lambda": float(cm.H1_LAMBDA),
        "soup_epochs": list(soup_epochs) if max_steps is None else [],
        "init_state_sha256": init_hash,
        "curve": curve,
        "epoch_seconds": epoch_seconds,
        "wall_clock_s": float(time.perf_counter() - started),
        "seconds_per_epoch": float(np.mean(epoch_seconds)) if epoch_seconds else float("nan"),
        "peak_gpu_memory_mb": peak_memory,
        "device": str(device),
        "grad_log": grad_log,
        "fit_target_is_y": bool(arm == "Y"),
        "other_target_abs_gap_sum": float(np.abs(y_fit_np - g_fit_np).sum()),
        "official_test_loaded": False,
    }

    if max_steps is not None:
        # smoke path: just record the last state hash and return without soup.
        result["last_state_sha256"] = parameter_state_hash(model)
        return result

    members = sorted(soup)
    soup_state = {k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]}
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if save_states:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(soup_state, out_dir / f"{arm}_seed{seed}_raw_soup_state.pt")
        torch.save(last_state, out_dir / f"{arm}_seed{seed}_last_state.pt")
        torch.save(init_state, out_dir / f"{arm}_seed{seed}_init_state.pt")

    # --- raw-soup reload + replay check -------------------------------------
    fit_y_local = torch.as_tensor(y_fit_np, dtype=torch.float32)
    fit_g_local = torch.as_tensor(g_fit_np, dtype=torch.float32)
    replay = build_full_model(blob, seed).to(device)
    replay.load_state_dict({k: v.to(device) for k, v in soup_state.items()})
    replay.eval()
    fit_raw, _ = predict(replay, fit_data, fit_y_local, device)
    dev_raw, _ = predict(replay, dev_data, dev_y_local, device)
    dev_g_raw, _ = predict(replay, dev_data, dev_g_local, device)
    fit_g_raw, _ = predict(replay, fit_data, fit_g_local, device)

    # calibration is always external; the raw soup state is never bias-folded.
    b = float(np.median(y_fit_np - fit_raw)) if arm == "Y" else float(np.median(g_fit_np - fit_raw))
    b_from_y = float(np.median(y_fit_np - (fit_raw + c_fit_np))) if arm == "O" else None

    # reload from the exported file to certify the released artifact.
    reloaded = build_full_model(blob, seed).to(device)
    reloaded.load_state_dict(torch.load(out_dir / f"{arm}_seed{seed}_raw_soup_state.pt", map_location=device))
    reloaded.eval()
    dev_replay, _ = predict(reloaded, dev_data, dev_y_local, device)

    result.update(
        {
            "soup_members": members,
            "soup_state_sha256": parameter_state_hash_of_state(soup_state),
            "last_state_sha256": parameter_state_hash_of_state(last_state),
            "calibration": {"b": b, "b_from_y": b_from_y, "folded_into_state": False},
            "fit_raw_mae_y": float(np.mean(np.abs(fit_raw - y_fit_np))),
            "fit_cal_mae_y": float(np.mean(np.abs(fit_raw + b - y_fit_np))),
            "fit_raw_mae_g": float(np.mean(np.abs(fit_g_raw - g_fit_np))),
            "dev_raw_mae_y": float(np.mean(np.abs(dev_raw - y_all[dev_idx]))),
            "dev_cal_mae_y": float(np.mean(np.abs(dev_raw + b - y_all[dev_idx]))),
            "replay_max_abs_diff": float(np.max(np.abs(dev_raw - dev_replay))),
            "replay_ok": bool(float(np.max(np.abs(dev_raw - dev_replay))) <= REPLAY_TOL),
            "dev_predictions_replay_sha256": _sha256_array(dev_replay),
            "official_test_loaded": False,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{arm}_seed{seed}_predictions.npz",
        fit_raw=fit_raw,
        fit_y=y_fit_np,
        fit_g=g_fit_np,
        fit_c=c_fit_np,
        dev_raw=dev_raw,
        dev_y=y_all[dev_idx],
        dev_g=g_all[dev_idx],
        dev_c=c_all[dev_idx],
        fit_idx=fit_idx,
        dev_idx=dev_idx,
    )
    _write_json(out_dir / f"{arm}_seed{seed}.json", result)
    log(
        f"[{arm}/s{seed}] DONE dev_raw={result['dev_raw_mae_y']:.6f} "
        f"dev_cal={result['dev_cal_mae_y']:.6f} fit_cal={result['fit_cal_mae_y']:.6f} "
        f"b={b:.6f} wall={result['wall_clock_s']:.0f}s replay={result['replay_max_abs_diff']:.2e}"
    )
    return result


def parameter_state_hash_of_state(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def predict(
    model: torch.nn.Module,
    data_list: Sequence[Any],
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, tgts = [], []
    state = rng_state()
    with torch.no_grad():
        for indices in epoch_batches(len(data_list), BATCH_SIZE, torch.Generator().manual_seed(0), False):
            batch = make_batch(data_list, indices, targets, device)
            preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
            tgts.append(batch.y.view(-1).detach().cpu())
    restore_rng(state)
    return torch.cat(preds).numpy().astype(np.float64), torch.cat(tgts).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def run_smoke(
    blob: Mapping[str, Any],
    fit_data: Sequence[Any],
    dev_data: Sequence[Any],
    decomp: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    out_dir: Path,
    max_steps: int = 8,
    log: Any = print,
) -> dict[str, Any]:
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    y = np.asarray(decomp["y"], np.float64)
    c = np.asarray(decomp["c"], np.float64)
    g = np.asarray(decomp["g"], np.float64)
    fit_y_local = torch.as_tensor(y[fit_idx], dtype=torch.float32)
    fit_g_local = torch.as_tensor(g[fit_idx], dtype=torch.float32)
    checks: dict[str, Any] = {"official_test_loaded": False}

    # 1. paired identity of the two arms at initialisation.
    seed_everything(0)
    m_y = build_full_model(blob, 0).to(device)
    h_y = parameter_state_hash(m_y)
    seed_everything(0)
    m_o = build_full_model(blob, 0).to(device)
    h_o = parameter_state_hash(m_o)
    checks["init_identity"] = {
        "Y_sha256": h_y,
        "O_sha256": h_o,
        "identical": bool(h_y == h_o),
        "parameters": int(sum(p.numel() for p in m_y.parameters())),
    }
    # seed1 init must differ from seed0.
    seed_everything(1)
    m_y1 = build_full_model(blob, 1)
    checks["init_identity"]["seed1_sha256"] = parameter_state_hash(m_y1)
    checks["init_identity"]["seed1_differs_from_seed0"] = bool(
        checks["init_identity"]["seed1_sha256"] != h_y
    )

    # identical inputs -> identical initial predictions; target gap == c.
    indices = list(range(min(32, len(fit_data))))
    batch = p1.env_collate([fit_data[i] for i in indices])
    m_y.eval()
    m_o.eval()
    with torch.no_grad():
        p_y, _ = m_y(batch, mask=cm.C6_MASK, return_aux=True)
        p_o, _ = m_o(batch, mask=cm.C6_MASK, return_aux=True)
    checks["initial_forward_identity"] = {
        "max_abs_pred_diff": float((p_y - p_o).abs().max()),
        "identical": bool(torch.equal(p_y, p_o)),
    }
    target_gap = (fit_y_local[indices] - fit_g_local[indices]).numpy()
    checks["target_gap_is_c"] = {
        "max_abs": float(np.max(np.abs(target_gap - c[fit_idx[indices]]))),
        "ok": bool(np.allclose(target_gap, c[fit_idx[indices]], atol=1e-6)),
    }

    # 2. O gradient path: h -> task loss, and both dictionaries participate.
    seed_everything(0)
    model = build_full_model(blob, 0).to(device)
    model.train()
    batch = make_batch(fit_data, indices, fit_g_local, device)
    pred, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
    task = F.l1_loss(pred.view(-1), batch.y.view(-1))
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = task + float(cm.H1_LAMBDA) * rec
    model.zero_grad()
    loss.backward()
    grads = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None}
    reader_g = float(model.reader.net[4].weight.grad.norm())
    dl_g = float(model.local_dictionary_bridge.D_L.grad.norm())
    vl_g = float(model.local_dictionary_bridge.V_L.grad.norm())
    structural_g = float(model.D.grad.norm())
    checks["gradient_path"] = {
        "loss_finite": bool(math.isfinite(float(loss))),
        "n_params_with_grad": int(len(grads)),
        "reader_last_weight_grad": reader_g,
        "task_dictionary_D_L_grad": dl_g,
        "task_dictionary_V_L_grad": vl_g,
        "structural_dictionary_D_grad": structural_g,
        "h1_lambda": float(cm.H1_LAMBDA),
        "ok": bool(reader_g > 0 and dl_g > 0 and vl_g > 0 and structural_g > 0),
    }
    # reconstruction term still contributes: a fresh rec-only forward/backward.
    model.zero_grad()
    _, aux_rec = model(batch, mask=cm.C6_MASK, return_aux=True)
    rec_only = model.reconstruction_loss(aux_rec["phi"], aux_rec["coord"])
    rec_only.backward()
    rec_only_structural = float(model.D.grad.norm())
    bridge_grad_present = model.local_dictionary_bridge.D_L.grad is not None
    checks["reconstruction_path"] = {
        "rec_value": float(rec_only.detach()),
        "rec_only_structural_D_grad": rec_only_structural,
        "bridge_D_L_touched_by_rec": bool(bridge_grad_present),
        "rec_does_not_flow_into_task_dictionary": bool(not bridge_grad_present),
        "ok": bool(rec_only_structural > 0),
    }

    # 3. oracle residual identity.
    model.zero_grad()
    model.eval()
    with torch.no_grad():
        h_raw = model(batch, mask=cm.C6_MASK).view(-1).double().cpu().numpy()
    yy = y[fit_idx[indices]]
    cc = c[fit_idx[indices]]
    gg = g[fit_idx[indices]]
    lhs = yy - (h_raw + cc)
    rhs = gg - h_raw
    checks["oracle_residual_identity"] = {
        "max_abs": float(np.max(np.abs(lhs - rhs))),
        "mae_lhs": float(np.mean(np.abs(lhs))),
        "mae_rhs": float(np.mean(np.abs(rhs))),
        "ok": bool(np.allclose(lhs, rhs, atol=1e-6)),
    }
    # Y: y - f == g - (f - c); subtracting c from prediction and target changes nothing.
    f_raw = h_raw
    checks["control_identity"] = {
        "max_abs": float(np.max(np.abs((yy - f_raw) - (gg - (f_raw - cc))))),
        "ok": bool(np.allclose(yy - f_raw, gg - (f_raw - cc), atol=1e-6)),
    }

    # 4. single-calibration semantics.
    pred_same = np.linspace(-1.0, 2.0, len(dev_data))
    same = median_gain(pred_same, pred_same, g, c, np.asarray(blob["dev_idx"], np.int64))
    swapped = median_gain(pred_same, pred_same + 0.5, g, c, np.asarray(blob["dev_idx"], np.int64))
    checks["single_calibration"] = {
        "identical_predictions_gain": same,
        "identical_is_zero": bool(same == 0.0),
        "swapped_sign": swapped,
        "ok": bool(same == 0.0 and swapped < 0.0),
    }

    # 5. sources untouched: encoded y still equals the frozen decomposition.
    pos = list(range(min(64, len(fit_data))))
    enc_y = np.array([float(fit_data[i].y.item()) for i in pos])
    checks["cleanup_check"] = {
        "encoded_y_matches_decomposition_max_abs": float(np.max(np.abs(enc_y - y[fit_idx[pos]]))),
        "decomposition_sha256": _sha256_array(np.concatenate([y, c, g])),
        "ok": bool(np.max(np.abs(enc_y - y[fit_idx[pos]])) < 1e-6),
    }
    checks["mechanism_ok"] = bool(
        checks["init_identity"]["identical"]
        and checks["initial_forward_identity"]["identical"]
        and checks["target_gap_is_c"]["ok"]
        and checks["gradient_path"]["ok"]
        and checks["reconstruction_path"]["ok"]
        and checks["oracle_residual_identity"]["ok"]
        and checks["control_identity"]["ok"]
        and checks["single_calibration"]["ok"]
        and checks["cleanup_check"]["ok"]
    )

    # optional: run <= max_steps optimizer steps to exercise the real loop.
    smoke_run = train_arm(
        "O",
        0,
        blob,
        fit_data,
        dev_data,
        decomp,
        device=device,
        out_dir=out_dir,
        epochs=1,
        max_steps=int(max_steps),
        save_states=False,
        log=log,
    )
    checks["smoke_steps"] = {
        "steps_done": smoke_run["steps_done"],
        "loss_curve": smoke_run["curve"],
        "grad_log": smoke_run["grad_log"],
    }
    _write_json(out_dir / "smoke_checks.json", checks)
    log(f"[smoke] mechanism_ok={checks['mechanism_ok']} steps={smoke_run['steps_done']}")
    return checks


def median_gain(pred_y: np.ndarray, pred_o: np.ndarray, g: np.ndarray, c: np.ndarray, dev_idx: np.ndarray) -> float:
    y_dev = np.asarray(g, np.float64)[dev_idx] + np.asarray(c, np.float64)[dev_idx]
    return float(np.mean(np.abs(pred_y - y_dev)) - np.mean(np.abs(pred_o - y_dev)))


# ---------------------------------------------------------------------------
# phase 0 CLI
# ---------------------------------------------------------------------------


def phase0(*, device: str = "cpu", log: Any = print) -> dict[str, Any]:
    t0 = time.perf_counter()
    decomp = build_target_decomposition()
    save_target_decomposition(decomp)
    checks = decomp["checks"]

    blob = load_prep_blob()
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    k = np.asarray(decomp["arrays"]["k"], np.int64)
    checks["split"] = {
        "n_fit": int(len(fit_idx)),
        "n_dev": int(len(dev_idx)),
        "fit_idx_sha256": hashlib.sha256(fit_idx.tobytes()).hexdigest(),
        "dev_idx_sha256": hashlib.sha256(dev_idx.tobytes()).hexdigest(),
        "expected_fit_idx_sha256": "165e87ef4398ba8ef57411c2f118c4cca4ea74007f2485611b84f0c09bbdd9ea",
        "expected_dev_idx_sha256": "fb8b78063e7c8a5c759bfa7d553738068cee1738a2d67b210e5d9dc492331376",
        "dev_group_counts": {
            "k0": int((k[dev_idx] == 0).sum()),
            "k-1": int((k[dev_idx] == -1).sum()),
            "kle-2": int((k[dev_idx] <= -2).sum()),
        },
        "dev_no_group_straddle": bool(no_group_straddle(blob)),
    }
    checks["split"]["dev_group_counts_expected"] = {"k0": 1926, "k-1": 65, "kle-2": 9}

    # train-only encoded check.
    train = load_train_only()
    checks["encoded"] = {
        "n_rows": int(len(train)),
        "official_test_loaded": False,
    }
    enc_y = np.array([float(d.y.item()) for d in train])
    y = np.asarray(decomp["arrays"]["y"], np.float64)
    checks["encoded"]["y_matches_decomposition_max_abs"] = float(np.max(np.abs(enc_y - y)))
    checks["encoded"]["y_matches"] = bool(np.max(np.abs(enc_y - y)) < 1e-5)

    x175 = apply_prep_train_only(train, blob)
    checks["x175_all_train_sha256"] = _sha256_array(x175)
    checks["x175_expected_all_train_sha256"] = "1ca3163cc2ddcec8c25ff1350e963d31105edc081a6bd8d79d5fcd001a1f81f7"
    checks["x175_reproduced"] = bool(checks["x175_all_train_sha256"] == checks["x175_expected_all_train_sha256"])
    checks["prep"] = {
        "D_fit_sha256": _sha256_array(np.asarray(blob["D_fit"], np.float32)),
        "U_components_sha256": _sha256_array(np.asarray(blob["U_components"], np.float32)),
        "fold_objects_file_sha256": _file_sha256(PREP_BLOB),
    }
    checks["sources"] = {
        "cycle_label_csv": {"path": str(CYCLE_LABEL_CSV.relative_to(zjd.REPO_ROOT)), "sha256": _file_sha256(CYCLE_LABEL_CSV)},
        "refine_json": {"path": str(REFINE_JSON.relative_to(zjd.REPO_ROOT)), "sha256": _file_sha256(REFINE_JSON)},
        "handoff_train": {"path": str(HANDOFF_TRAIN.relative_to(zjd.REPO_ROOT)), "sha256": _file_sha256(HANDOFF_TRAIN)},
        "generator_source": "tracks/ksvd/experiments/luyin16/zinc_long_cycle_audit.py::stage_refine + zinc_long_cycle_summary.py",
        "c_definition": "label_cycle_component = (label_effective_cycle_snapped - mu_cycle)/sigma_cycle; label-derived oracle",
    }
    checks["target_decomposition_sha256"] = _sha256_array(np.concatenate([y, np.asarray(decomp["arrays"]["c"]), np.asarray(decomp["arrays"]["g"])]))
    checks["seconds"] = float(time.perf_counter() - t0)
    checks["official_test_loaded"] = False

    manifest = build_input_manifest(checks, decomp)
    _write_json(RESULTS_DIR / "checks.json", checks)
    _write_json(RESULTS_DIR / "input_manifest.json", manifest)
    log(
        f"[phase0] dev k0={checks['split']['dev_group_counts']['k0']} "
        f"k-1={checks['split']['dev_group_counts']['k-1']} "
        f"kle-2={checks['split']['dev_group_counts']['kle-2']} "
        f"x175_ok={checks['x175_reproduced']}"
    )
    return checks


def no_group_straddle(blob: Mapping[str, Any]) -> bool:
    gid = np.asarray(load_target_decomposition()["gid"], np.int64)
    k = np.asarray(load_target_decomposition()["k"], np.int64)
    fit_set = set(int(i) for i in np.asarray(blob["fit_idx"]).tolist())
    by_group: dict[int, set] = collections.defaultdict(set)
    for i, g in enumerate(gid.tolist()):
        by_group[g].add("fit" if i in fit_set else "dev")
    return all(len(v) == 1 for v in by_group.values())


def build_input_manifest(checks: Mapping[str, Any], decomp: Mapping[str, Any]) -> dict[str, Any]:
    y = np.asarray(decomp["arrays"]["y"])
    c = np.asarray(decomp["arrays"]["c"])
    g = np.asarray(decomp["arrays"]["g"])
    k = np.asarray(decomp["arrays"]["k"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "train_source": "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt + results/e2e_dictenv_p1/cache/env_train.pt (train-only)",
        "split": checks["split"],
        "prep": checks["prep"],
        "x175": {
            "all_train_sha256": checks["x175_all_train_sha256"],
            "expected_all_train_sha256": checks["x175_expected_all_train_sha256"],
            "reproduced": checks["x175_reproduced"],
        },
        "targets": {
            "y": "handoff train.npz y (= encoded .y)",
            "c": "train_cycle_audit_label.csv label_cycle_component (label-derived oracle)",
            "g": "y - c (= label y_without_cycle_label)",
            "k": "round(label_effective_cycle_snapped)",
            "shapes": {"y": list(y.shape), "c": list(c.shape), "g": list(g.shape), "k": list(k.shape)},
            "k_distribution": {str(int(v)): int((k == v).sum()) for v in sorted(set(k.tolist()))},
            "c_summary": checks["c_summary"],
            "g_summary": checks["g_summary"],
            "task_definitions": {"Y": "L1(f(x), y)", "O": "L1(h(x), g), pred=h(x)+c"},
        },
        "sources": checks["sources"],
        "flags": {
            "c_is_model_input": False,
            "c_used_for": "O supervision target + diagnostic evaluation offset only",
            "oracle_diagnostic": True,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="ZINC Full cycle-target decomposition v1")
    parser.add_argument("--mode", required=True, choices=("phase0", "smoke", "train"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--subset", type=int, default=None, help="limit fit/dev molecules (smoke only)")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)

    if args.mode == "phase0":
        phase0(device="cpu")
        return 0

    blob = load_prep_blob()
    decomp = load_target_decomposition()
    train_data = load_train_only()
    apply_prep_train_only(train_data, blob)
    fit_data, dev_data = build_fit_dev(train_data, blob["fit_idx"], blob["dev_idx"])

    if args.mode == "smoke":
        if args.subset is not None:
            fit_data = fit_data[: int(args.subset)]
            dev_data = dev_data[: int(args.subset)]
        checks = run_smoke(
            blob, fit_data, dev_data, decomp,
            device=resolve_device(args.device), out_dir=out_dir, max_steps=int(args.max_steps or 8),
        )
        return 0 if checks["mechanism_ok"] else 1

    if args.mode == "train":
        assert args.arm is not None, "--arm required for train"
        train_arm(
            args.arm, args.seed, blob, fit_data, dev_data, decomp,
            device=resolve_device(args.device), out_dir=out_dir, epochs=int(args.epochs),
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())