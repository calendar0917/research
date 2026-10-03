"""ZINC full-decomposition valid/test confirmation v1.

Round ``zinc_full_decomposition_valid_test_confirmation_v1``.

Purpose
-------
Re-examine, at the **full official-train scale (all 10,000 rows)** and with two
paired seeds, the frozen "dictionary-chemistry branch + independent learned
cycle head" method that was previously only run on an 8,000-fit / 2,000-reused
-diagnostic split.  After every selection is frozen, run **one** authorised
official-valid and official-test evaluation and report the real MAE.

Four Full trajectories, two seeds (paired Y/H), one independent small cycle head
per seed:

* ``Y_s``  — canonical Full ``f(x)``, loss ``L1(f(x), y)``       (matched control)
* ``H_s``  — same canonical Full init, loss ``L1(h(x), g)``, g = y - c   (chemistry branch)
* ``Q_s``  — small MLP on the actual Full topology25 input ``T``, ``L1(q(T), c)``
* ``P_s``  — deployable ``h_s(x) + q_s(T) + b_P,s``            (no true ``c`` at inference)

``c`` is the label-derived cycle component from the frozen train-only
decomposition; it is used as the ``H`` supervision target and the ``Q`` target,
and (post-prediction only) for grouping.  It is never a model input.

Training is strictly train-only: the official-valid and official-test splits are
not loaded until ``--mode heldout`` is invoked **after** the frozen manifest is
written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import fec_s0_factorization as fec
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16 import zinc_post_v4_residual_audit as pva

PROTOCOL_VERSION = "zinc-full-decomposition-valid-test-confirmation-v1"

RESULTS_DIR = zjd.TRACK_ROOT / "results/zinc_full_decomposition_valid_test_confirmation_v1"
PROTOCOL_PATH = zjd.TRACK_ROOT / "protocols/zinc-full-decomposition-valid-test-confirmation-v1.yaml"
OLD_8K_PREP = zjd.PREP_DIR / "fold_objects.npz"
DECOMP_PATH = zftd.TARGET_DECOMP_PATH
PREP_PATH = RESULTS_DIR / "all_train_prep.npz"
T25_PATH = RESULTS_DIR / "T25_all.npz"

SEEDS = (0, 1)
SCALE_SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
FULL_PARAMETERS = 408_651

HEAD_EPOCHS = 300
HEAD_HIDDEN = (64, 32)
TOPOLOGY_IN = 25
HEAD_PARAMETERS = 3777
TRAIN_GEN_BASE = 20261003

N_TRAIN = 10000
DIAG_ROWS = 512
FIXED_DEV_BATCH = 32
REPLAY_TOL = 2.0e-6

ARMS = ("Y", "H")
SEVERE_MAX = -2
BOOT_SEED = 20261003
N_BOOT = 1000

# pre-registered gate (valid-only interpretation; test is unconditional reporting)
GATE_MEAN_GAIN = 0.003
GATE_G0_WORSEN_MAX = 0.001


# ---------------------------------------------------------------------------
# generic helpers
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


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
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    raise ValueError(f"unsupported device {name!r}")


def median(values: np.ndarray) -> float:
    return float(np.median(np.asarray(values, np.float64)))


def parameter_state_hash(model: nn.Module) -> str:
    return state_hash({k: v for k, v in model.state_dict().items()})


# ---------------------------------------------------------------------------
# target decomposition (train-only label source, frozen)
# ---------------------------------------------------------------------------


def load_decomp() -> dict[str, np.ndarray]:
    with np.load(DECOMP_PATH, allow_pickle=False) as z:
        return {k: z[k] for k in ("y", "c", "g", "k", "gid")}


# ---------------------------------------------------------------------------
# all-10k train-only preprocessing
# ---------------------------------------------------------------------------


def build_all_train_prep(*, log: Any = print) -> dict[str, Any]:
    """Build the single canonical all-10k train-fit prep object.

    Reuses the historical algorithm (``zinc_joint_dictionary_decision_v1
    .build_fold_inputs``) with ``fit_idx = all 10,000 rows``.  Because the
    per-column ``*_all`` standardisers are already fitted on all official-train
    rows, the row-fit standardisers are the same objects; the genuinely new
    quantities are the structural dictionary ``D_fit`` and the common subspace
    ``U`` refit over all 10,000 molecules' ``phi`` rows.
    """
    started = time.perf_counter()
    train = zftd.load_train_only()
    if len(train) != N_TRAIN:
        raise RuntimeError(f"expected {N_TRAIN} train rows, got {len(train)}")

    records, _valid_records = pva._extract_v4_records()
    fits_all = fec.build_fits(records)
    patch_all = zjd.Std(fits_all.patch_standardizer.mean, fits_all.patch_standardizer.scale)
    ctx_all = zjd.Std(fits_all.context_standardizer.mean, fits_all.context_standardizer.scale)
    topo_all = zjd.Std(fits_all.topology_standardizer.mean, fits_all.topology_standardizer.scale)
    anchor_stats = read_json(zjd.TRACK_ROOT / "results/e2e_dictenv_p1/anchor_stats.json")
    anchor_all = zjd.Std(np.asarray(anchor_stats["mean"], np.float32),
                         np.asarray(anchor_stats["scale"], np.float32))

    phi = np.concatenate([d.dict_phi.numpy() for d in train], 0).astype(np.float64)
    if phi.shape[1] != cssd.PHI_DIM:
        raise RuntimeError(f"phi dim {phi.shape[1]} != {cssd.PHI_DIM}")
    n_rows = int(phi.shape[0])

    # all rows are fit rows -> root_mask is all True.
    patch_raw, _ = zjd._stack(train, "patch_cont")
    patch_raw = patch_all.inverse(patch_raw)
    ctx_raw, _ = zjd._stack(train, "global_context")
    ctx_raw = ctx_all.inverse(ctx_raw)
    anchor_raw, _ = zjd._stack(train, "anchor")
    anchor_raw = anchor_all.inverse(anchor_raw)
    topo_raw, _ = zjd._stack(train, "topology_features")
    topo_raw = topo_all.inverse(topo_raw)

    patch_fit = zjd.Std.fit(patch_raw, floor=zjd.CANON_STD_FLOOR)
    ctx_fit = zjd.Std.fit(ctx_raw, floor=zjd.CANON_STD_FLOOR)
    anchor_fit = zjd.Std.fit(anchor_raw, floor=zjd.CANON_STD_FLOOR)
    topo_fit = zjd.Std.fit(topo_raw, floor=zjd.CANON_STD_FLOOR)

    log(f"[prep] fitting D_fit (K-SVD) on {n_rows} phi rows over {len(train)} molecules ...")
    t_ksvd = time.perf_counter()
    D_fit, ksvd_info = sdb.fit_ksvd(phi, atoms=sdb.K_ATOMS, s=int(sdb.SPARSITY),
                                    epochs=10, seed=sdb.DICT_SEED)
    ksvd_seconds = time.perf_counter() - t_ksvd
    log(f"[prep] K-SVD {ksvd_seconds:.1f}s")
    t_sub = time.perf_counter()
    subspace_fit = cssd.build_common_subspace(phi, q=1, kind="q1")
    subspace_seconds = time.perf_counter() - t_sub
    log(f"[prep] subspace {subspace_seconds:.1f}s")

    # all-train X175 normalisation (not consumed by the Full, kept for provenance).
    x = np.concatenate([phi[:, :65].astype(np.float32), patch_raw[:, :108], anchor_raw[:, 60:62]], 1)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[~np.isfinite(x_std) | (x_std < zjd.STD_FLOOR)] = 1.0
    z = (x - x_mean) / x_std
    blocks = [(zjd.PHI_BLOCK[0], zjd.PHI_BLOCK[1]), (zjd.SEM_BLOCK[0], zjd.SEM_BLOCK[1]),
              (zjd.SIZE_BLOCK[0], zjd.SIZE_BLOCK[1])]
    block_scale = np.asarray(
        [math.sqrt(max(float(np.mean(np.sum(z[:, lo:hi] ** 2, axis=1))), 1e-12)) for lo, hi in blocks],
        dtype=np.float32,
    )

    scaler_gap = {
        "patch": float(np.max(np.abs(patch_fit.mean - patch_all.mean)) + np.max(np.abs(patch_fit.scale - patch_all.scale))),
        "ctx": float(np.max(np.abs(ctx_fit.mean - ctx_all.mean)) + np.max(np.abs(ctx_fit.scale - ctx_all.scale))),
        "anchor": float(np.max(np.abs(anchor_fit.mean - anchor_all.mean)) + np.max(np.abs(anchor_fit.scale - anchor_all.scale))),
        "topo": float(np.max(np.abs(topo_fit.mean - topo_all.mean)) + np.max(np.abs(topo_fit.scale - topo_all.scale))),
    }
    # The frozen anchor standardiser keeps degenerate columns at scale 1e-6 while
    # ``Std.fit`` (floor 1e-6) maps them to 1.0; those columns are constant, so
    # the only permitted gap is on them.
    anchor_degenerate = np.where(anchor_all.scale <= 1e-5)[0].tolist()
    nondeg = np.ones(anchor_all.scale.shape[0], bool)
    nondeg[anchor_degenerate] = False
    anchor_nondeg_gap = float(
        np.max(np.abs(anchor_fit.mean[nondeg] - anchor_all.mean[nondeg]))
        + np.max(np.abs(anchor_fit.scale[nondeg] - anchor_all.scale[nondeg]))
    ) if nondeg.any() else 0.0
    if max(scaler_gap["patch"], scaler_gap["ctx"], scaler_gap["topo"], anchor_nondeg_gap) > 1e-5:
        raise RuntimeError(f"fit==all scaler identity failed: {scaler_gap} nondeg_anchor={anchor_nondeg_gap}")

    keep = dict(
        fit_idx=np.arange(N_TRAIN, dtype=np.int64),
        dev_idx=np.arange(N_TRAIN, dtype=np.int64),  # no held-out split in this prep
        patch_all_mean=patch_all.mean, patch_all_scale=patch_all.scale,
        ctx_all_mean=ctx_all.mean, ctx_all_scale=ctx_all.scale,
        anchor_all_mean=anchor_all.mean, anchor_all_scale=anchor_all.scale,
        topo_all_mean=topo_all.mean, topo_all_scale=topo_all.scale,
        patch_fit_mean=patch_fit.mean, patch_fit_scale=patch_fit.scale,
        ctx_fit_mean=ctx_fit.mean, ctx_fit_scale=ctx_fit.scale,
        anchor_fit_mean=anchor_fit.mean, anchor_fit_scale=anchor_fit.scale,
        topo_fit_mean=topo_fit.mean, topo_fit_scale=topo_fit.scale,
        D_fit=D_fit, U_components=subspace_fit.components, U_rms=subspace_fit.rms,
        x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32),
        block_scale=block_scale,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREP_PATH, **keep)
    prep_meta = {
        "protocol_version": PROTOCOL_VERSION,
        "n_rows": N_TRAIN,
        "phi_rows": n_rows,
        "algorithm": "zinc_joint_dictionary_decision_v1.build_fold_inputs semantics with fit_idx = all 10,000",
        "dictionary": {"atoms": int(sdb.K_ATOMS), "sparsity": int(sdb.SPARSITY),
                       "epochs": 10, "seed": int(sdb.DICT_SEED), "seconds": ksvd_seconds},
        "subspace": {"q": 1, "kind": "q1", "seconds": subspace_seconds},
        "scaler_gap_fit_vs_all_max": scaler_gap,
        "anchor_degenerate_columns": anchor_degenerate,
        "anchor_nondeg_gap": anchor_nondeg_gap,
        "prep_file_sha256": file_sha256(PREP_PATH),
        "D_fit_sha256": sha256_array(D_fit),
        "U_components_sha256": sha256_array(subspace_fit.components),
        "x_mean_sha256": sha256_array(x_mean.astype(np.float32)),
        "x_std_sha256": sha256_array(x_std.astype(np.float32)),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - started),
    }
    write_json(RESULTS_DIR / "prep_meta.json", prep_meta)
    log(f"[prep] done {prep_meta['seconds']:.1f}s D_fit_sha256={prep_meta['D_fit_sha256'][:16]}")
    return prep_meta


def load_prep() -> dict[str, np.ndarray]:
    with np.load(PREP_PATH, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build_all_train_topo25(blob: Mapping[str, Any], train: Sequence[Any]) -> np.ndarray:
    T = np.concatenate(
        [d.topology_features.reshape(1, -1).numpy().astype(np.float32) for d in train], axis=0
    )
    if T.shape != (len(train), TOPOLOGY_IN):
        raise RuntimeError(f"topology25 shape mismatch: {T.shape}")
    if blob is not None:
        pass
    return T


# ---------------------------------------------------------------------------
# Full training (paired Y/H)
# ---------------------------------------------------------------------------


def _full_model(blob: Mapping[str, Any], seed: int) -> nn.Module:
    model = sc.build_scale_model(
        np.asarray(blob["D_fit"], np.float32),
        int(seed),
        zjd.fold_subspace(blob),
        sc.FULL,
        scale_seed=SCALE_SEED,
    )
    audit = sc.scale_parameter_audit(model)
    if int(audit["actual_parameters"]) != FULL_PARAMETERS or not audit.get("parameter_exact", audit.get("parameter_audit", False)):
        raise RuntimeError(f"canonical Full parameter audit failed: {audit}")
    return model


def _batches(n: int, batch_size: int, generator: torch.Generator, shuffle: bool) -> list[list[int]]:
    if shuffle:
        order = torch.randperm(int(n), generator=generator).tolist()
    else:
        order = list(range(int(n)))
    return [order[s:s + batch_size] for s in range(0, int(n), batch_size)]


def _batch(data_list: Sequence[Any], indices: Sequence[int], targets: torch.Tensor, device: torch.device):
    batch = p1.env_collate([data_list[i] for i in indices])
    batch.y = targets[torch.as_tensor(list(indices), dtype=torch.long)].view(-1).clone()
    return batch.to(device)


def predict_full(model: nn.Module, data_list: Sequence[Any], targets: torch.Tensor,
                 device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[torch.Tensor] = []
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    with torch.no_grad():
        for indices in _batches(len(data_list), BATCH_SIZE, torch.Generator().manual_seed(0), False):
            batch = _batch(data_list, indices, targets, device)
            preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
    torch.set_rng_state(rng)
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)
    return torch.cat(preds).numpy().astype(np.float64)


def train_full_arm(arm: str, seed: int, blob: Mapping[str, Any], fit_data: Sequence[Any],
                   diag_data: Sequence[Any], decomp: Mapping[str, np.ndarray], *,
                   device: torch.device, out_dir: Path, epochs: int = EPOCHS,
                   max_steps: int | None = None, save_states: bool = True, log: Any = print) -> dict[str, Any]:
    assert arm in ARMS, arm
    y_all = np.asarray(decomp["y"], np.float64)
    g_all = np.asarray(decomp["g"], np.float64)
    target_fit_np = y_all if arm == "Y" else g_all
    fit_target = torch.as_tensor(target_fit_np, dtype=torch.float32)
    diag_target = torch.as_tensor(target_fit_np[: len(diag_data)], dtype=torch.float32)
    y_t = torch.as_tensor(y_all.astype(np.float32))

    seed_everything(seed)
    model = _full_model(blob, seed).to(device)
    init_hash = parameter_state_hash(model)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)

    soup: dict[int, dict[str, torch.Tensor]] = {}
    soup_epochs = set(range(max(1, int(epochs) - SOUP_LAST + 1), int(epochs) + 1))
    curve: list[dict[str, Any]] = []
    epoch_seconds: list[float] = []
    steps_done = 0
    stopped_reason = "completed"
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, rec_sum, n_mol, n_steps = 0.0, 0.0, 0, 0
        for indices in _batches(len(fit_data), BATCH_SIZE, train_gen, True):
            batch = _batch(fit_data, indices, fit_target, device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = task + float(cm.H1_LAMBDA) * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(rec.detach())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        epoch_seconds.append(float(time.perf_counter() - epoch_started))
        with torch.no_grad():
            model.eval()
            diag_pred = predict_full(model, diag_data, diag_target, device)
        diag_mae = float(np.mean(np.abs(diag_pred - target_fit_np[: len(diag_data)])))
        curve.append({
            "epoch": int(epoch),
            "train_task_mae": float(task_sum / max(n_mol, 1)),
            "train_rec": float(rec_sum / max(n_steps, 1)),
            "diag_mae": diag_mae,
            "seconds": epoch_seconds[-1],
        })
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(f"[{arm}/s{seed}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"diag={diag_mae:.6f} rec={curve[-1]['train_rec']:.4g} {epoch_seconds[-1]:.1f}s")
        if max_steps is not None and steps_done >= int(max_steps):
            break

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION, "arm": arm, "seed": int(seed),
        "scale_seed": SCALE_SEED, "epochs": int(epochs), "steps_done": int(steps_done),
        "stopped_reason": stopped_reason, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE, "h1_lambda": float(cm.H1_LAMBDA),
        "soup_epochs": sorted(soup_epochs) if max_steps is None else [],
        "init_state_sha256": init_hash, "curve": curve,
        "wall_clock_s": float(time.perf_counter() - started),
        "seconds_per_epoch": float(np.mean(epoch_seconds)) if epoch_seconds else float("nan"),
        "device": str(device),
        "fit_target_is_y": bool(arm == "Y"),
        "train_generator_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    if max_steps is not None:
        result["last_state_sha256"] = parameter_state_hash(model)
        return result

    members = sorted(soup)
    soup_state = {k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]}
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_states:
        torch.save(soup_state, out_dir / f"{arm}_seed{seed}_raw_soup_state.pt")
        torch.save(last_state, out_dir / f"{arm}_seed{seed}_last_state.pt")
        torch.save(init_state, out_dir / f"{arm}_seed{seed}_init_state.pt")

    # raw-soup reload + replay on all 10k train rows (eval mode, external bias)
    replay = _full_model(blob, seed).to(device)
    replay.load_state_dict({k: v.to(device) for k, v in soup_state.items()})
    fit_raw = predict_full(replay, fit_data, y_t, device)
    if arm == "Y":
        b = median(y_all - fit_raw)
    else:
        b = median(g_all - fit_raw)  # diagnostic b_H; never folded into P
    b_from_y = median(y_all - fit_raw) if arm == "H" else None

    # certify the released artifact (reload from disk)
    reloaded = _full_model(blob, seed).to(device)
    reloaded.load_state_dict(torch.load(out_dir / f"{arm}_seed{seed}_raw_soup_state.pt", map_location=device))
    fit_raw_replay = predict_full(reloaded, fit_data, y_t, device)
    replay_max = float(np.max(np.abs(fit_raw - fit_raw_replay)))

    result.update({
        "soup_members": members,
        "soup_state_sha256": state_hash(soup_state),
        "last_state_sha256": state_hash(last_state),
        "calibration": {"b": b, "b_from_y": b_from_y, "folded_into_state": False},
        "fit_raw_mae_y": float(np.mean(np.abs(fit_raw - y_all))),
        "fit_cal_mae_y": float(np.mean(np.abs(fit_raw + b - y_all))),
        "replay_max_abs_diff": replay_max,
        "replay_ok": bool(replay_max <= REPLAY_TOL),
    })
    np.savez_compressed(
        out_dir / f"{arm}_seed{seed}_fit_pred.npz",
        fit_raw=fit_raw, y=y_all, g=g_all, c=np.asarray(decomp["c"]),
        k=np.asarray(decomp["k"]), gid=np.asarray(decomp["gid"]),
    )
    write_json(out_dir / f"{arm}_seed{seed}.json", result)
    log(f"[{arm}/s{seed}] DONE fit_raw_mae_y={result['fit_raw_mae_y']:.6f} b={b:.6f} "
        f"replay={replay_max:.2e} wall={result['wall_clock_s']:.0f}s")
    return result


# ---------------------------------------------------------------------------
# learned cycle head (Q), CPU
# ---------------------------------------------------------------------------


def build_head(seed: int, bias_value: float) -> nn.Module:
    torch.manual_seed(int(seed))
    head = nn.Sequential(
        nn.Linear(TOPOLOGY_IN, HEAD_HIDDEN[0]), nn.SiLU(),
        nn.Linear(HEAD_HIDDEN[0], HEAD_HIDDEN[1]), nn.SiLU(),
        nn.Linear(HEAD_HIDDEN[1], 1),
    )
    n_params = int(sum(p.numel() for p in head.parameters()))
    if n_params != HEAD_PARAMETERS:
        raise RuntimeError(f"head parameter audit failed: {n_params}")
    nn.init.zeros_(head[4].weight)
    with torch.no_grad():
        head[4].bias.copy_(torch.tensor(float(bias_value)))
    return head


def head_forward(head: nn.Module, T: torch.Tensor) -> torch.Tensor:
    return head(T).view(-1)


def train_head(seed: int, T_fit: np.ndarray, c_fit: np.ndarray, bias_value: float,
               log: Any = print) -> dict[str, Any]:
    n = int(T_fit.shape[0])
    head = build_head(seed, bias_value)
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(int(TRAIN_GEN_BASE) + int(seed))
    T_t = torch.as_tensor(T_fit, dtype=torch.float32)
    c_t = torch.as_tensor(c_fit, dtype=torch.float32)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        order = torch.randperm(n, generator=generator)
        abs_sum, grad_norm = 0.0, 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            prediction = head_forward(head, T_t[idx])
            loss = (prediction - c_t[idx]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP))
            optimizer.step()
            abs_sum += float((prediction - c_t[idx]).abs().sum())
            steps += 1
        if epoch >= HEAD_EPOCHS - SOUP_LAST + 1:
            soup[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        curve.append({"epoch": epoch, "train_task_mae": abs_sum / n, "grad_norm": grad_norm,
                      "seconds": float(time.perf_counter() - started)})
    soup_mean = {key: torch.stack([soup[e][key] for e in sorted(soup)], dim=0).mean(dim=0)
                 for key in soup[HEAD_EPOCHS]}
    head.load_state_dict(soup_mean)
    head.eval()
    with torch.no_grad():
        q_fit = head_forward(head, T_t).numpy().astype(np.float64)
    return {"head": head, "init_state": init_state, "soup_mean": soup_mean,
            "init_hash": state_hash(init_state), "soup_hash": state_hash(head),
            "curve": curve, "steps": steps, "soup_members": sorted(soup), "q_fit": q_fit,
            "seconds": float(time.perf_counter() - started)}


# ---------------------------------------------------------------------------
# deploy wrapper
# ---------------------------------------------------------------------------


class FullDecompositionWrapper(nn.Module):
    """Deployable inference: ``f(x)`` (Y) or ``h(x) + q(T) + b_P`` (P).

    Accepts graph data plus the frozen Full topology25 input.  It never receives
    or looks up ``y``/``g``/``c``/``k``/group/molecule identifiers.
    """

    def __init__(self, full: nn.Module, head: nn.Module | None, bias: float, *, mode: str) -> None:
        super().__init__()
        assert mode in ("Y", "P")
        self.full = full
        self.head = head
        self.bias = float(bias)
        self.mode = mode
        self.full.eval()
        for parameter in self.full.parameters():
            parameter.requires_grad_(False)
        if head is not None:
            self.head.eval()
            for parameter in self.head.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch: Any, topo25: torch.Tensor | None = None) -> torch.Tensor:
        h = self.full(batch, mask=cm.C6_MASK).view(-1)
        if self.mode == "Y":
            return h + self.bias
        T = batch.topology_features if topo25 is None else topo25
        return h + head_forward(self.head, T) + self.bias


# ---------------------------------------------------------------------------
# held-out data adapters
# ---------------------------------------------------------------------------


def apply_prep(data: Sequence[Any], blob: Mapping[str, Any]) -> None:
    zftd.apply_prep_train_only(data, blob)


def load_valid_data(blob: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    valid = list(torch.load(sdp.CACHE_DIR / "encoded_valid.pt", map_location="cpu", weights_only=False))
    p1run.attach_env(valid, "valid")
    # optional anchor refresh is not needed (encoded cache already carries it)
    apply_prep(valid, blob)
    meta = {"n_rows": int(len(valid)), "source": "encoded_valid.pt + env_valid.pt",
            "official_valid_loaded": True}
    return valid, meta


def load_test_data(blob: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Build the official-test encoded cache from the raw ZINC subset.

    Uses the repository's terminal official-test construction path
    (``zinc_compact_v4_training_sufficiency``) but keeps the split mapping
    explicit: ``test`` is never silently dispatched to ``train``.
    """
    from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_training_sufficiency as ztraining

    train_records, _valid_records = ztraining.load_train_valid_records()
    test_records = ztraining.extract_test_records()
    config = ztraining.base_config()
    _train_enc, test_data, _audit = ztraining.build_encoded(train_records, test_records, config)
    test_data = list(test_data)
    raw_test = list(ztraining._load_zinc(ztraining.ZINC_ROOT, "test"))
    if len(test_data) != len(raw_test):
        raise RuntimeError(f"official-test length mismatch: {len(test_data)} vs {len(raw_test)}")
    env_meta = p1run._attach_test_env(test_data, raw_test)
    apply_prep(test_data, blob)
    meta = {"n_rows": int(len(test_data)), "source": "official ZINC subset test (PyG)",
            "raw_split": "test", "official_test_loaded": True, **env_meta}
    return test_data, meta


# ---------------------------------------------------------------------------
# engineering smoke + adapter parity (no held-out data)
# ---------------------------------------------------------------------------


def _load_train_all(blob: Mapping[str, Any], subset: int | None = None) -> list[Any]:
    train = zftd.load_train_only()
    if subset is not None:
        train = train[: int(subset)]
    apply_prep(train, blob)
    return list(train)


def run_smoke(out_dir: Path) -> int:
    blob = load_prep()
    decomp = load_decomp()
    y = np.asarray(decomp["y"], np.float64)
    g = np.asarray(decomp["g"], np.float64)
    c = np.asarray(decomp["c"], np.float64)

    checks: dict[str, Any] = {"official_valid_loaded": False, "official_test_loaded": False}

    seed_everything(0)
    m_y = _full_model(blob, 0)
    h_y = parameter_state_hash(m_y)
    seed_everything(0)
    m_h = _full_model(blob, 0)
    h_h = parameter_state_hash(m_h)
    seed_everything(1)
    m_y1 = _full_model(blob, 1)
    h_y1 = parameter_state_hash(m_y1)
    checks["full_init"] = {
        "Y_sha256": h_y, "H_sha256": h_h, "identical": bool(h_y == h_h),
        "seed1_sha256": h_y1, "seed1_differs": bool(h_y1 != h_y),
        "Y_parameters": int(sum(p.numel() for p in m_y.parameters())),
        "H_parameters": int(sum(p.numel() for p in m_h.parameters())),
        "expected_parameters": FULL_PARAMETERS,
    }
    # paired data stream + init
    gen_a = torch.Generator().manual_seed(0 + TRAIN_SHUFFLE_OFFSET)
    gen_b = torch.Generator().manual_seed(0 + TRAIN_SHUFFLE_OFFSET)
    same_order = bool(torch.equal(torch.randperm(N_TRAIN, generator=gen_a),
                                  torch.randperm(N_TRAIN, generator=gen_b)))
    checks["paired_stream"] = {"same_batch_order_YH": same_order,
                               "train_generator_seed": 0 + TRAIN_SHUFFLE_OFFSET}

    checks["target_identity"] = {
        "max_abs_g_plus_c_minus_y": float(np.max(np.abs(g + c - y))),
        "c_not_zeroed_k0": bool(np.any(np.abs(c[np.asarray(decomp["k"]) == 0]) > 0)),
    }

    median_c = median(c)
    head = build_head(0, median_c)
    checks["head"] = {
        "parameters": int(sum(p.numel() for p in head.parameters())),
        "expected_parameters": HEAD_PARAMETERS,
        "last_layer_weight_all_zero": bool(torch.all(head[4].weight == 0)),
        "last_layer_bias": float(head[4].bias.detach()),
        "median_train_c": median_c,
    }

    # <= 8-step full smoke on Y and H, states discarded
    train = _load_train_all(blob, subset=1024)
    diag = train[:128]
    smoke_runs: dict[str, Any] = {}
    for arm in ARMS:
        r = train_full_arm(arm, 0, blob, train, diag, decomp, device=torch.device("cpu"),
                           out_dir=out_dir, epochs=1, max_steps=8, save_states=False, log=None)
        smoke_runs[arm] = {"steps_done": r["steps_done"], "loss_curve": r["curve"],
                           "last_state_sha256": r["last_state_sha256"]}
    checks["full_smoke"] = smoke_runs

    # head gradient path on step 1 (last layer learns; layers 1-3 zero by design)
    T = build_all_train_topo25(blob, train)
    T_t = torch.as_tensor(T[:512], dtype=torch.float32)
    c_t = torch.as_tensor(c[:512], dtype=torch.float32)
    head = build_head(0, median_c)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    opt.zero_grad()
    loss = (head_forward(head, T_t) - c_t).abs().mean()
    loss.backward()
    grads = {k: float(p.grad.norm()) for k, p in head.named_parameters() if p.grad is not None}
    opt.step()
    opt.zero_grad()
    loss2 = (head_forward(head, T_t) - c_t).abs().mean()
    loss2.backward()
    opt.step()
    changed = {k: bool(not torch.equal(head.state_dict()[k], before[k])) for k in before}
    checks["head_smoke"] = {
        "first_step_grad_norms": grads,
        "layers_1_3_zero_on_step1": bool(all(v == 0.0 for k, v in grads.items()
                                               if not k.startswith("4."))),
        "all_params_changed_after_2_steps": bool(all(changed.values())),
        "loss_step1": float(loss.detach()), "loss_step2": float(loss2.detach()),
    }
    write_json(out_dir / "smoke_checks.json", checks)
    ok = bool(checks["full_init"]["identical"] and checks["full_init"]["seed1_differs"]
              and same_order and checks["target_identity"]["max_abs_g_plus_c_minus_y"] < 1e-9
              and checks["head"]["parameters"] == HEAD_PARAMETERS
              and checks["head_smoke"]["layers_1_3_zero_on_step1"]
              and checks["head_smoke"]["all_params_changed_after_2_steps"])
    checks["ok"] = ok
    write_json(out_dir / "smoke_checks.json", checks)
    print(f"[smoke] ok={ok}")
    return 0 if ok else 1


def run_adapter_parity(out_dir: Path) -> int:
    """Verify a fresh graph built by the adapter reproduces canonical fields."""
    blob = load_prep()
    records, _ = pva._extract_v4_records()
    fits_all = fec.build_fits(records)
    patch_all = zjd.Std(fits_all.patch_standardizer.mean, fits_all.patch_standardizer.scale)
    ctx_all = zjd.Std(fits_all.context_standardizer.mean, fits_all.context_standardizer.scale)
    topo_all = zjd.Std(fits_all.topology_standardizer.mean, fits_all.topology_standardizer.scale)
    train = list(torch.load(sdp.CACHE_DIR / "encoded_train.pt", map_location="cpu", weights_only=False))
    p1run.attach_env(train, "train")
    env = torch.load(p1run.CACHE_DIR / "env_train.pt", map_location="cpu", weights_only=False)
    n = 128
    checks: dict[str, Any] = {"n_rows": n, "official_valid_loaded": False, "official_test_loaded": False}

    def _stack(attr: str) -> np.ndarray:
        parts = []
        for d in train[:n]:
            v = getattr(d, attr)
            v = v.numpy() if torch.is_tensor(v) else np.asarray(v)
            parts.append(np.asarray(v, dtype=np.float32))
        return np.concatenate(parts, 0)

    rec_patch = np.asarray(zpp._patch_matrix(records[:n]), np.float32)
    rec_ctx = np.asarray(zpp._context_matrix(records[:n]), np.float32)
    rec_topo = np.stack([np.asarray(r.topology_features, np.float32) for r in records[:n]], 0)
    checks["patch_cont"] = float(np.max(np.abs(patch_all.transform(rec_patch) - _stack("patch_cont"))))
    checks["global_context"] = float(np.max(np.abs(ctx_all.transform(rec_ctx) - _stack("global_context"))))
    checks["topology_features"] = float(np.max(np.abs(topo_all.transform(rec_topo) - _stack("topology_features"))))

    # env cache (<-> encoded) structural consistency for the whole train split
    ns = np.asarray(env["node_sizes"])
    checks["env_phi_rows_match"] = bool(int(ns.sum()) == int(env["phi"].shape[0]))
    attached = torch.cat([d.dict_phi for d in train[:64]], 0).numpy()
    _m = min(4096, int(attached.shape[0]))
    checks["env_phi_equals_attached_dict_phi"] = float(np.max(np.abs(
        env["phi"].numpy()[:_m] - attached[:_m])))
    checks["env_anchor_offset_ok"] = bool(int(env["anchor"].shape[0]) == int(ns.sum()))

    # fresh phi from raw graphs for a few train molecules
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
    from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
    from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_training_sufficiency as ztraining
    raw = list(ztraining._load_zinc(ztraining.ZINC_ROOT, "train"))[:4]
    max_phi = 0.0
    for data, mol in zip(train[:4], raw):
        graph, _nt, _et = zlr._data_to_graph(mol)
        fresh = r2.build_phi(graph).astype(np.float32)
        max_phi = max(max_phi, float(np.abs(fresh - data.dict_phi.numpy()).max()))
    checks["fresh_phi_vs_encoded_max_abs"] = max_phi
    checks["pass"] = bool(
        checks["patch_cont"] <= 2e-6 and checks["global_context"] <= 2e-6
        and checks["topology_features"] <= 2e-6 and checks["env_phi_rows_match"]
        and checks["env_anchor_offset_ok"] and checks["env_phi_equals_attached_dict_phi"] <= 2e-6
        and max_phi == 0.0
    )
    write_json(out_dir / "adapter_checks.json", checks)
    print(f"[adapter] pass={checks['pass']} {checks}")
    return 0 if checks["pass"] else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=PROTOCOL_VERSION)
    parser.add_argument("--mode", required=True,
                        choices=("prep", "topo", "smoke", "train", "qhead", "adapter"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "prep":
        build_all_train_prep(log=print)
        return 0
    if args.mode == "topo":
        blob = load_prep()
        train = zftd.load_train_only()
        if args.subset:
            train = train[: int(args.subset)]
        T = build_all_train_topo25(blob, train)
        np.savez_compressed(T25_PATH, T=T, sha256=np.array([sha256_array(T)]))
        write_json(RESULTS_DIR / "topo_meta.json", {
            "shape": list(T.shape), "sha256": sha256_array(T),
            "source": "actual canonical Full topology25 model input (all-10000 official train)",
            "columns_order_preserved": True,
            "official_valid_loaded": False, "official_test_loaded": False,
        })
        print(f"[topo] {T.shape} sha256={sha256_array(T)}")
        return 0
    if args.mode == "smoke":
        return run_smoke(Path(args.out))
    if args.mode == "train":
        assert args.arm is not None, "--arm required"
        blob = load_prep()
        decomp = load_decomp()
        train = zftd.load_train_only()
        apply_prep(train, blob)
        diag = train[:DIAG_ROWS]
        train_full_arm(args.arm, args.seed, blob, train, diag, decomp,
                       device=resolve_device(args.device), out_dir=out_dir,
                       epochs=int(args.epochs), max_steps=args.max_steps, log=print)
        return 0
    if args.mode == "qhead":
        blob = load_prep()
        decomp = load_decomp()
        train = zftd.load_train_only()
        apply_prep(train, blob)
        T = build_all_train_topo25(blob, train)
        c = np.asarray(decomp["c"], np.float64)
        y = np.asarray(decomp["y"], np.float64)
        median_c = median(c)
        seed = int(args.seed)
        result = train_head(seed, T, c, median_c, log=print)
        torch.save(result["init_state"], out_dir / f"Q_seed{seed}_head_init_state.pt")
        torch.save(result["soup_mean"], out_dir / f"Q_seed{seed}_head_soup_state.pt")
        np.savez_compressed(out_dir / f"Q_seed{seed}_predictions.npz",
                            fit_q=result["q_fit"], c=c, y=y, T=T)
        write_json(out_dir / f"Q_seed{seed}.json", {
            "protocol_version": PROTOCOL_VERSION, "seed": seed, "epochs": HEAD_EPOCHS,
            "steps": result["steps"], "soup_members": result["soup_members"],
            "train_generator_seed": int(TRAIN_GEN_BASE) + seed,
            "init_hash": result["init_hash"], "soup_hash": result["soup_hash"],
            "median_train_c": median_c, "seconds": result["seconds"],
            "curve": result["curve"], "official_valid_loaded": False,
            "official_test_loaded": False,
        })
        print(f"[qhead/s{seed}] done {result['seconds']:.1f}s")
        return 0
    if args.mode == "adapter":
        return run_adapter_parity(Path(args.out))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())