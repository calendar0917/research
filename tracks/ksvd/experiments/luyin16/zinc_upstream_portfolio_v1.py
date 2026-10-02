"""Runner-side orchestration for the upstream portfolio round.

Round ``upstream_portfolio_v1`` (Workstream Z, ZINC).  Core science lives in
``tracks/ksvd/experiments/luyin16/e2e_dictenv_upstream_portfolio_v1.py``.
Pre-registration: ``tracks/ksvd/notes/zinc_upstream_portfolio_v1_preregistration.md``.

Stages
------
``stage0_code``  frozen-parent export of the native code moments + H39, followed
                 by the fixed convex conditional fit and the CODE purchase gate.
``pilot``        one shared 80-epoch warm-start pilot for one arm (control or a
                 candidate); the soup is the *fixed* last five epochs 76..80.
``screen``       one 320-epoch fresh-initialisation screen for one arm; the soup
                 is the parent Top-5 protocol.

The official ZINC **test** split is never instantiated; every payload records
``official_test_loaded = false``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_upstream_portfolio_v1 as up
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = up.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_upstream_portfolio_v1"
CACHE_DIR = RESULTS_DIR / "cache"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "zinc_upstream_portfolio_v1_preregistration.md"
REFERENCE_DIR = TRACK_ROOT / "experiments/luyin16/upstream_portfolio_v1_reference"
PARENT_SOUP = TRACK_ROOT / "results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt"
PARENT_SOUP_SHA256 = "17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb"

ZINC_ROOT = REPO_ROOT / "data/ZINC"

SEED = 0
SCALE_SEED = 0
BATCH_SIZE = int(p2run.BATCH_SIZE)
WEIGHT_DECAY = float(p2run.WEIGHT_DECAY)
GRAD_CLIP = float(p2run.GRAD_CLIP)
TRAIN_SHUFFLE_OFFSET = int(p2run.TRAIN_SHUFFLE_OFFSET)
EVAL_SHUFFLE_OFFSET = int(p2run.EVAL_SHUFFLE_OFFSET)
SOUP_K = 5

PILOT_EPOCHS = 80
PILOT_LR = 1.0e-4
PILOT_SOUP_EPOCHS = (76, 77, 78, 79, 80)
SCREEN_EPOCHS = 320
SCREEN_LR = 1.0e-3

#: fixed id%5 valid groups (row index 0..999) and the historically flagged row.
GROUP_MOD = 5
ID172_EXCLUDED = 172

#: CODE conditional gate (frozen before the export).
CODE_GATE_VALID_MAX = 0.112
CODE_GATE_GAIN_MIN = 0.003
CODE_GATE_GROUPS_MIN = 4
CONDITIONAL_LAMBDA = 1.0e-5
CONDITIONAL_GAP = 1.0e-6

#: warm-pilot automatic purchase gate (frozen before any pilot).
PILOT_GATE_VALID_MAX = 0.111
PILOT_GATE_GAIN_MIN = 0.004
PILOT_GATE_GROUPS_MIN = 4
PILOT_GATE_EX172_GAIN_MIN = 0.002

_write_json = base._write_json
_read_json = base._read_json
_write_csv = base._write_csv
_git_commit = base._git_commit
_sha256_file = base._sha256_file
_load_parent_subspace = base._load_parent_subspace
_dictionary_tensor = base._dictionary_tensor


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, CACHE_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_device(name: str) -> torch.device:
    """Explicit runtime device; ``cpu`` keeps the parent threads, cuda is opt-in."""
    key = str(name).strip().lower()
    if key in ("cpu",):
        return torch.device("cpu")
    if key in ("cuda", "cuda:0", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device requests CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    raise ValueError(f"unsupported runtime.device {name!r}")


def _seed_everything(seed: int) -> None:
    p2run._seed_everything(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_dictionary() -> tuple[np.ndarray, str]:
    return p2run.load_dictionary(cm.H1_CONFIG.dict_kind)


def r3_cache_path(split: str) -> Path:
    return CACHE_DIR / f"r3_{split}.pt"


def build_r3_cache(split: str, force: bool = False) -> dict[str, Any]:
    """Compute (once) the exact shell-3 statistics for one official split."""
    _ensure_dirs()
    path = r3_cache_path(split)
    if path.exists() and not force:
        return torch.load(path, map_location="cpu", weights_only=False)
    raw_split = "val" if split == "valid" else "train"
    raw = list(zlr._load_zinc(ZINC_ROOT, raw_split))
    parts: list[torch.Tensor] = []
    sizes: list[int] = []
    started = time.perf_counter()
    for molecule in raw:
        n = int(molecule.num_nodes)
        edge_index = molecule.edge_index.detach().cpu().numpy()
        edge_attr = molecule.edge_attr.detach().cpu().numpy().reshape(-1)
        atoms = molecule.x.detach().cpu().numpy().reshape(-1).astype(np.int64)
        edges = up.undirected_edges(edge_index, edge_attr)
        parts.append(torch.as_tensor(up.shell3_features(n, atoms, edges), dtype=torch.int16))
        sizes.append(n)
    payload = {
        "features": torch.cat(parts, dim=0),
        "node_sizes": torch.as_tensor(sizes, dtype=torch.long),
        "split": split,
        "n_molecules": int(len(sizes)),
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    torch.save(payload, path)
    return payload


def attach_r3(data_list: Sequence[Any], split: str, subset: int | None = None) -> None:
    blob = build_r3_cache(split)
    sizes = blob["node_sizes"]
    features = blob["features"]
    total = int(sizes.shape[0])
    count = total if subset is None else min(int(subset), total)
    offset = 0
    for index in range(count):
        n = int(sizes[index])
        data_list[index].r3_features = features[offset : offset + n].clone()
        offset += n


def load_split(arm: str, split: str, subset: int | None = None) -> list[Any]:
    data = p1run.load_split(split, subset)
    if arm == "r3":
        attach_r3(data, split, subset)
    return data


def load_parent_soup_state() -> dict[str, torch.Tensor]:
    if not PARENT_SOUP.exists():
        raise FileNotFoundError(f"parent soup checkpoint unavailable: {PARENT_SOUP}")
    digest = _sha256_file(PARENT_SOUP)
    if digest != PARENT_SOUP_SHA256:
        raise RuntimeError(f"parent soup sha256 mismatch: {digest}")
    state = torch.load(PARENT_SOUP, map_location="cpu", weights_only=False)
    return {key: value.float() for key, value in state.items()}


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------


def make_model(arm: str, dictionary: np.ndarray, subspace: cssd.CommonSubspace, *, r3_scales: tuple[float, float, float], drop_p: float) -> sc.LatentScaleSEM108:
    return up.build_arm_model(
        arm,
        dictionary,
        SEED,
        subspace,
        sc.FULL,
        scale_seed=SCALE_SEED,
        r3_scales=r3_scales,
        drop_p=drop_p,
    )


def collect_predictions(model: sc.LatentScaleSEM108, data_list: Sequence[Any], device: torch.device, mask: Any) -> tuple[np.ndarray, np.ndarray]:
    loader = p1.make_env_loader(data_list, BATCH_SIZE, False, SEED + EVAL_SHUFFLE_OFFSET)
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch, mask=mask).view(-1).detach().cpu())
            targets.append(batch.y.view(-1).detach().cpu())
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()


def train_median_bias(model: sc.LatentScaleSEM108, train_data: Sequence[Any], device: torch.device, mask: Any) -> dict[str, Any]:
    """Train-only residual median folded into the existing reader output bias."""
    predictions, targets = collect_predictions(model, train_data, device, mask)
    delta = float(np.median(targets - predictions))
    before = float(model.reader.net[4].bias.detach().cpu().item())
    with torch.no_grad():
        model.reader.net[4].bias.add_(delta)
    after = float(model.reader.net[4].bias.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        corrected = collect_predictions(model, train_data, device, mask)[0]
    return {
        "bias_before": before,
        "bias_after": after,
        "delta": delta,
        "train_median_residual": delta,
        "train_mae_before": float(np.mean(np.abs(targets - predictions))),
        "train_mae_after": float(np.mean(np.abs(targets - corrected))),
    }


def group_deltas(candidate: np.ndarray, control: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Per-id%5 MAE deltas and the descriptive non-id172 delta."""
    deltas: list[float] = []
    for group in range(GROUP_MOD):
        rows = np.arange(len(target)) % GROUP_MOD == group
        candidate_mae = float(np.mean(np.abs(candidate[rows] - target[rows])))
        control_mae = float(np.mean(np.abs(control[rows] - target[rows])))
        deltas.append(float(control_mae - candidate_mae))
    keep = np.arange(len(target)) != ID172_EXCLUDED
    ex172_gain = float(
        np.mean(np.abs(control[keep] - target[keep])) - np.mean(np.abs(candidate[keep] - target[keep]))
    )
    return {
        "groups_gain": deltas,
        "groups_improved": int(sum(value > 0.0 for value in deltas)),
        "ex172_gain": ex172_gain,
    }


def _keep_soup(store: dict[int, dict[str, torch.Tensor]], epoch: int, state: Mapping[str, torch.Tensor], curve: list[dict[str, Any]], mode: str) -> None:
    snapshot = {key: value.detach().to("cpu", copy=True) for key, value in state.items()}
    if mode == "last5":
        store[int(epoch)] = snapshot
        while len(store) > 5:
            del store[min(store)]
        return
    store[int(epoch)] = snapshot
    keep = {i + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:SOUP_K]}
    for cached in list(store):
        if cached not in keep:
            del store[cached]


def train_arm(
    arm: str,
    model: sc.LatentScaleSEM108,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    soup_mode: str,
    log: bool = True,
) -> dict[str, Any]:
    """One arm trajectory; loss / optimizer / masking follow the parent recipe."""
    _seed_everything(SEED)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=WEIGHT_DECAY)
    loader = p1.make_env_loader(train_data, BATCH_SIZE, True, SEED + TRAIN_SHUFFLE_OFFSET)
    eval_loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, SEED + EVAL_SHUFFLE_OFFSET)
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    epoch_seconds: list[float] = []
    peak_memory = 0.0
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum = 0.0
        n_molecules = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_molecules += int(batch.y.numel())
        model.eval()
        with torch.no_grad():
            predictions: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for batch in eval_loader:
                batch = batch.to(device)
                predictions.append(model(batch, mask=mask).view(-1).detach().cpu())
                targets.append(batch.y.view(-1).detach().cpu())
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        valid_mae = float((prediction - target).abs().mean())
        train_mae = float(task_sum / max(n_molecules, 1))
        epoch_seconds.append(float(time.perf_counter() - epoch_started))
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "valid_mae": valid_mae,
                "seconds": epoch_seconds[-1],
            }
        )
        _keep_soup(soup, epoch, model.state_dict(), curve, soup_mode)
        if device.type == "cuda":
            peak_memory = max(peak_memory, float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)))
        if log and (epoch == 1 or epoch % 10 == 0 or epoch == int(epochs)):
            print(f"[{arm}] epoch={epoch:03d} train={train_mae:.6f} valid={valid_mae:.6f} {epoch_seconds[-1]:.1f}s", flush=True)
    members = sorted(soup)
    soup_state = {
        key: torch.stack([soup[epoch][key].float() for epoch in members]).mean(0)
        for key in soup[members[0]]
    }
    fresh = make_model(arm, _dictionary_tensor(), _load_parent_subspace(), r3_scales=model.block_scales if arm == "r3" else (1.0, 1.0, 1.0), drop_p=getattr(model, "atom_dropout", up.DROP_P))
    fresh.to(device)
    if arm == "code":
        fresh.set_code_scales(model.code_scale_1, model.code_scale_2)
    fresh.load_state_dict(soup_state)
    raw_predictions, targets = collect_predictions(fresh, valid_data, device, mask)
    calibration = train_median_bias(fresh, train_data, device, mask)
    calibrated_predictions, _ = collect_predictions(fresh, valid_data, device, mask)
    return {
        "arm": arm,
        "epochs": int(epochs),
        "lr": float(lr),
        "soup_mode": soup_mode,
        "members": members,
        "member_valid_mae": [float(curve[epoch - 1]["valid_mae"]) for epoch in members],
        "curve": curve,
        "wall_clock_s": float(time.perf_counter() - started),
        "seconds_per_epoch": float(sum(epoch_seconds) / max(len(epoch_seconds), 1)),
        "peak_gpu_memory_mb": peak_memory,
        "raw_valid_mae": float(np.mean(np.abs(raw_predictions - targets))),
        "calibrated_valid_mae": float(np.mean(np.abs(calibrated_predictions - targets))),
        "calibration": calibration,
        "soup_state": soup_state,
        "valid_predictions_raw": raw_predictions.tolist(),
        "valid_predictions_calibrated": calibrated_predictions.tolist(),
        "valid_targets": targets.tolist(),
        "soup_state_sha256": audit.state_sha256(soup_state),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# stage 0: CODE conditional export + fixed convex fit
# ---------------------------------------------------------------------------


def _reader_input_and_code_moments(model: sc.LatentScaleSEM108, data_list: Sequence[Any], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen-parent forward; returns ``(H39, z, y)`` in fixed row order."""
    loader = p1.make_env_loader(data_list, BATCH_SIZE, False, SEED + EVAL_SHUFFLE_OFFSET)
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    handle = model.reader.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("reader_input", inputs[0].detach())
    )
    h39: list[np.ndarray] = []
    moments: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weight = model.reader.net[0].weight.detach()
    bias = model.reader.net[0].bias.detach()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            model(batch, mask=cm.C6_MASK)
            reader_input = captured.pop("reader_input")
            hidden = torch.clamp(reader_input @ weight.t() + bias, min=0.0)
            coord = model.code(batch.dict_phi)
            interface = model.semantic_interface(coord, batch, cm.C6_MASK, None)
            h = sem.SEM108Model._environment_from_parts(model, coord, batch, interface, mask=cm.C6_MASK)
            _e, aux = model.local_dictionary_bridge(h, return_aux=True)
            code = aux["scale"] * aux["alpha"]
            graph = batch.batch
            n_graphs = int(graph.max().item()) + 1 if int(graph.numel()) else 0
            first = code.new_zeros((n_graphs, code.shape[1]))
            second = code.new_zeros((n_graphs, code.shape[1]))
            first.index_add_(0, graph, code)
            second.index_add_(0, graph, code * code)
            h39.append(hidden.detach().cpu().numpy())
            moments.append(torch.cat([first, second], dim=1).detach().cpu().numpy())
            targets.append(batch.y.view(-1).detach().cpu().numpy())
    handle.remove()
    return np.concatenate(h39, 0), np.concatenate(moments, 0), np.concatenate(targets, 0)


def stage0_code(device: torch.device) -> dict[str, Any]:
    """Frozen Full export of ``z`` and the fixed ``[1, H39, z/scale]`` fit."""
    from tracks.ksvd.experiments.luyin16.upstream_portfolio_v1_reference.prototype_dictionary import fit_mae

    _ensure_dirs()
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    _seed_everything(SEED)
    model = up.build_arm_model("control", dictionary, SEED, subspace, sc.FULL, scale_seed=SCALE_SEED)
    model.load_state_dict(load_parent_soup_state())
    model.to(device).eval()

    train_data = load_split("control", "train")
    valid_data = load_split("control", "valid")
    h39_train, z_train, y_train = _reader_input_and_code_moments(model, train_data, device)
    h39_valid, z_valid, y_valid = _reader_input_and_code_moments(model, valid_data, device)

    scale_1 = float(np.sqrt(np.mean(z_train[:, : z_train.shape[1] // 2] ** 2)))
    scale_2 = float(np.sqrt(np.mean(z_train[:, z_train.shape[1] // 2 :] ** 2)))
    scale_1 = scale_1 if math.isfinite(scale_1) and scale_1 > 1e-12 else 1.0
    scale_2 = scale_2 if math.isfinite(scale_2) and scale_2 > 1e-12 else 1.0
    half = z_train.shape[1] // 2
    z_train_scaled = np.concatenate([z_train[:, :half] / scale_1, z_train[:, half:] / scale_2], axis=1)
    z_valid_scaled = np.concatenate([z_valid[:, :half] / scale_1, z_valid[:, half:] / scale_2], axis=1)

    baseline = np.load(REFERENCE_DIR / "H39_PHYSICAL.npz", allow_pickle=False)
    baseline_valid = np.asarray(baseline["p_valid"], dtype=np.float64)
    baseline_mae = float(np.mean(np.abs(baseline_valid - y_valid)))

    h39_design_train = np.column_stack([np.ones(len(h39_train)), h39_train])
    h39_design_valid = np.column_stack([np.ones(len(h39_valid)), h39_valid])
    refit_coef, refit_status = fit_mae(h39_design_train, y_train.astype(np.float64), CONDITIONAL_LAMBDA, max_iterations=12000, gap_tolerance=CONDITIONAL_GAP, time_budget_s=120)
    refit_valid_pred = h39_design_valid @ refit_coef
    refit_mae = float(np.mean(np.abs(refit_valid_pred - y_valid)))

    code_design_train = np.column_stack([np.ones(len(h39_train)), h39_train, z_train_scaled])
    code_design_valid = np.column_stack([np.ones(len(h39_valid)), h39_valid, z_valid_scaled])
    code_coef, code_status = fit_mae(code_design_train, y_train.astype(np.float64), CONDITIONAL_LAMBDA, max_iterations=12000, gap_tolerance=CONDITIONAL_GAP, time_budget_s=120)
    code_valid_pred = code_design_valid @ code_coef
    code_mae = float(np.mean(np.abs(code_valid_pred - y_valid)))

    gain = float(baseline_mae - code_mae)
    group_delta = group_deltas(code_valid_pred, baseline_valid, y_valid)
    converged = bool(code_status["status"] == "CONVERGED" and refit_status["status"] == "CONVERGED")
    passed = bool(
        converged
        and code_mae <= CODE_GATE_VALID_MAX
        and gain >= CODE_GATE_GAIN_MIN
        and group_delta["groups_improved"] >= CODE_GATE_GROUPS_MIN
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device),
        "official_test_loaded": False,
        "parent_soup_sha256": PARENT_SOUP_SHA256,
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "code_width": int(z_train.shape[1]),
        "code_scale_1": scale_1,
        "code_scale_2": scale_2,
        "baseline_h39_valid_mae": baseline_mae,
        "refit_h39_valid_mae": refit_mae,
        "refit_status": refit_status,
        "code_valid_mae": code_mae,
        "code_gain_vs_h39": gain,
        "code_status": code_status,
        "groups_gain_vs_h39": group_delta["groups_gain"],
        "groups_improved": group_delta["groups_improved"],
        "ex172_gain_vs_h39": group_delta["ex172_gain"],
        "gate": {
            "valid_max": CODE_GATE_VALID_MAX,
            "gain_min": CODE_GATE_GAIN_MIN,
            "groups_min": CODE_GATE_GROUPS_MIN,
        },
        "converged": converged,
        "passed": passed,
        "decision": "CODE_CONDITIONAL_PASS" if passed else "CODE_CONDITIONAL_STOP",
        "note": (
            "fixed frozen-parent proxy: a negative result closes this fixed proxy only, "
            "not the theoretical ceiling of native dictionary codes"
        ),
    }
    up.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "stage0_code.json", payload)
    np.savez_compressed(
        RESULTS_DIR / "stage0_code_fit.npz",
        code_coef=code_coef,
        code_valid_predictions=code_valid_pred,
        h39_valid_predictions=refit_valid_pred,
        valid_targets=y_valid,
        official_test_loaded=np.array(False),
    )
    print(f"[stage0] baseline={baseline_mae:.9f} refit={refit_mae:.9f} code={code_mae:.9f} gain={gain:+.6f} passed={passed}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: warm 80-epoch pilot
# ---------------------------------------------------------------------------


def _r3_scales_for(arm: str, train_data: Sequence[Any]) -> tuple[float, float, float]:
    if arm != "r3":
        return (1.0, 1.0, 1.0)
    features = torch.cat([getattr(item, "r3_features") for item in train_data], dim=0)
    return up.r3_block_scales(features)


def pilot(arm: str, device: torch.device, *, epochs: int = PILOT_EPOCHS, lr: float = PILOT_LR) -> dict[str, Any]:
    _ensure_dirs()
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    train_data = load_split(arm, "train")
    valid_data = load_split(arm, "valid")
    r3_scales = _r3_scales_for(arm, train_data)
    soup_state = load_parent_soup_state()

    _seed_everything(SEED)
    model = make_model(arm, dictionary, subspace, r3_scales=r3_scales, drop_p=up.DROP_P)
    loaded = up.load_parent_soup(model, arm, soup_state)
    audit_info = up.arm_parameter_audit(model, arm)

    # CODE: freeze the train-only moment scales *after* warm-starting on the parent weights.
    if arm == "code":
        model.to(device).eval()
        train_moments = _code_moments_only(model, train_data, device)
        scale_1 = float(np.sqrt(np.mean(train_moments[:, : train_moments.shape[1] // 2] ** 2)))
        scale_2 = float(np.sqrt(np.mean(train_moments[:, train_moments.shape[1] // 2 :] ** 2)))
        model.set_code_scales(scale_1 if math.isfinite(scale_1) and scale_1 > 1e-12 else 1.0, scale_2 if math.isfinite(scale_2) and scale_2 > 1e-12 else 1.0)

    result = train_arm(arm, model, train_data, valid_data, epochs=epochs, lr=lr, device=device, soup_mode="last5")
    torch.save(result["soup_state"], CHECKPOINT_DIR / f"PILOT-{arm.upper()}-seed0_soup_state.pt")
    result.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": str(device),
            "stage": "pilot",
            "seed": SEED,
            "parent_soup_sha256": PARENT_SOUP_SHA256,
            "parent_load": loaded,
            "parameter_audit": audit_info,
            "r3_scales": list(r3_scales),
            "split_sizes": {"train": len(train_data), "valid": len(valid_data)},
        }
    )
    up.official_test_blocker(result)
    _write_json(RESULTS_DIR / f"pilot_{arm}.json", result)
    print(f"[pilot:{arm}] raw={result['raw_valid_mae']:.6f} calibrated={result['calibrated_valid_mae']:.6f} wall={result['wall_clock_s']:.0f}s", flush=True)
    return result


def _code_moments_only(model: sc.LatentScaleSEM108, data_list: Sequence[Any], device: torch.device) -> np.ndarray:
    loader = p1.make_env_loader(data_list, BATCH_SIZE, False, SEED + EVAL_SHUFFLE_OFFSET)
    model.eval()
    moments: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            coord = model.code(batch.dict_phi)
            interface = model.semantic_interface(coord, batch, cm.C6_MASK, None)
            h = sem.SEM108Model._environment_from_parts(model, coord, batch, interface, mask=cm.C6_MASK)
            _e, aux = model.local_dictionary_bridge(h, return_aux=True)
            code = aux["scale"] * aux["alpha"]
            graph = batch.batch
            n_graphs = int(graph.max().item()) + 1 if int(graph.numel()) else 0
            first = code.new_zeros((n_graphs, code.shape[1]))
            second = code.new_zeros((n_graphs, code.shape[1]))
            first.index_add_(0, graph, code)
            second.index_add_(0, graph, code * code)
            moments.append(torch.cat([first, second], dim=1).detach().cpu().numpy())
    return np.concatenate(moments, 0)


# ---------------------------------------------------------------------------
# stage: fresh 320-epoch screen
# ---------------------------------------------------------------------------


def screen(arm: str, device: torch.device, *, epochs: int = SCREEN_EPOCHS, lr: float = SCREEN_LR) -> dict[str, Any]:
    _ensure_dirs()
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    train_data = load_split(arm, "train")
    valid_data = load_split(arm, "valid")
    r3_scales = _r3_scales_for(arm, train_data)

    _seed_everything(SEED)
    model = make_model(arm, dictionary, subspace, r3_scales=r3_scales, drop_p=up.DROP_P)
    audit_info = up.arm_parameter_audit(model, arm)
    if arm == "code":
        model.to(device).eval()
        train_moments = _code_moments_only(model, train_data, device)
        scale_1 = float(np.sqrt(np.mean(train_moments[:, : train_moments.shape[1] // 2] ** 2)))
        scale_2 = float(np.sqrt(np.mean(train_moments[:, train_moments.shape[1] // 2 :] ** 2)))
        model.set_code_scales(scale_1 if math.isfinite(scale_1) and scale_1 > 1e-12 else 1.0, scale_2 if math.isfinite(scale_2) and scale_2 > 1e-12 else 1.0)

    result = train_arm(arm, model, train_data, valid_data, epochs=epochs, lr=lr, device=device, soup_mode="top5")
    torch.save(result["soup_state"], CHECKPOINT_DIR / f"SCREEN-{arm.upper()}-seed0_soup_state.pt")
    result.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": str(device),
            "stage": "screen",
            "seed": SEED,
            "fresh_initialisation": True,
            "parameter_audit": audit_info,
            "r3_scales": list(r3_scales),
            "split_sizes": {"train": len(train_data), "valid": len(valid_data)},
        }
    )
    up.official_test_blocker(result)
    _write_json(RESULTS_DIR / f"screen_{arm}.json", result)
    print(f"[screen:{arm}] raw={result['raw_valid_mae']:.6f} calibrated={result['calibrated_valid_mae']:.6f} wall={result['wall_clock_s']:.0f}s", flush=True)
    return result


# ---------------------------------------------------------------------------
# small-preset parameter audit (construction only; Small is never trained)
# ---------------------------------------------------------------------------


def small_parameter_audit() -> dict[str, Any]:
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    rows: dict[str, Any] = {}
    for arm in up.ARMS:
        _seed_everything(SEED)
        if arm == "r3":
            model = up.build_arm_model(arm, dictionary, SEED, subspace, sc.SMALL, r3_scales=(1.0, 1.0, 1.0))
        else:
            model = up.build_arm_model(arm, dictionary, SEED, subspace, sc.SMALL)
        rows[arm] = {
            "expected": up.expected_parameters(arm, sc.SMALL),
            "actual": int(sum(parameter.numel() for parameter in model.parameters())),
            "reader_input": int(model.reader.net[0].weight.shape[1]),
            "fusion_input": int(model.fusion[0].weight.shape[1]),
        }
        rows[arm]["exact"] = bool(rows[arm]["expected"] == rows[arm]["actual"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "spec": sc.SMALL.as_dict(),
        "arms": rows,
        "all_exact": bool(all(row["exact"] for row in rows.values())),
    }


__all__ = [
    "PROTOCOL_VERSION",
    "RESULTS_DIR",
    "CACHE_DIR",
    "CHECKPOINT_DIR",
    "PREREG_PATH",
    "PARENT_SOUP",
    "PARENT_SOUP_SHA256",
    "PILOT_EPOCHS",
    "PILOT_LR",
    "PILOT_SOUP_EPOCHS",
    "SCREEN_EPOCHS",
    "SCREEN_LR",
    "CODE_GATE_VALID_MAX",
    "CODE_GATE_GAIN_MIN",
    "PILOT_GATE_VALID_MAX",
    "PILOT_GATE_GAIN_MIN",
    "PILOT_GATE_EX172_GAIN_MIN",
    "resolve_device",
    "load_dictionary",
    "build_r3_cache",
    "attach_r3",
    "load_split",
    "load_parent_soup_state",
    "collect_predictions",
    "train_median_bias",
    "group_deltas",
    "train_arm",
    "stage0_code",
    "pilot",
    "screen",
    "small_parameter_audit",
]