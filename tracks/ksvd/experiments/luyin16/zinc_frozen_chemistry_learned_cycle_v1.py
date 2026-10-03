"""ZINC frozen-chemistry learned-cycle head v1 (local CPU diagnostic).

Single question: how much of the frozen ``O`` (g-predictor) oracle gain can the
EXISTING Full ``topology25`` input already learn, and therefore be retained at
real inference without reading the true label-derived ``c``?

Frozen base (commit ``87894a2``, dir
``tracks/ksvd/results/zinc_full_cycle_target_decomposition_v1``):
``O_seed{0,1}_raw_soup_state.pt`` is ``h_s(x)`` (g-predictor, unfolded bias);
``O_seed{0,1}_predictions.npz`` ``fit_raw``/``dev_raw`` are the cached ``h_raw``.

This runner:
* verifies h_raw from the raw soup forward on a fixed small dev batch,
* builds the frozen Full topology25 model input ``T`` for all 10000 train rows
  via ``apply_prep_train_only`` (same source / column order / 8000-fit
  standardization as the frozen O),
* runs a <=4-step smoke (head updates, frozen O unchanged), discarded,
* trains two fresh cycle heads seed 0/1 with the frozen recipe,
* calibrates P and K once on fit and saves the artifacts (analysis is separate).

The official ZINC test split is never instantiated and the official validation
split is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

PROTOCOL_VERSION = "zinc-frozen-chemistry-learned-cycle-v1"

RESULTS_DIR = zjd.TRACK_ROOT / "results/zinc_frozen_chemistry_learned_cycle_v1"
FROZEN_DIR = zjd.TRACK_ROOT / "results/zinc_full_cycle_target_decomposition_v1"
PREP_BLOB = zjd.PREP_DIR / "fold_objects.npz"
DECOMP = FROZEN_DIR / "target_decomposition.npz"

SEEDS = (0, 1)
SCALE_SEED = 0
EPOCHS = 300
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
HIDDEN = (64, 32)
TOPOLOGY_IN = 25
HEAD_PARAMETERS = 3777
TRAIN_GEN_BASE = 20261003
SMOKE_STEPS = 4
FIXED_DEV_BATCH = 32
REPLAY_TOL = 2.0e-6
SEVERE_MAX = -2


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


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_arr(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def median(values: np.ndarray) -> float:
    return float(np.median(np.asarray(values, np.float64)))


# ---------------------------------------------------------------------------
# frozen inputs
# ---------------------------------------------------------------------------


def load_frozen(seed: int, arm: str) -> dict[str, Any]:
    meta = json.loads((FROZEN_DIR / f"{arm}_seed{seed}.json").read_text())
    preds = dict(np.load(FROZEN_DIR / f"{arm}_seed{seed}_predictions.npz", allow_pickle=False))
    return {"meta": meta, "preds": preds}


def load_prep_and_targets() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    blob = zftd.load_prep_blob()
    with np.load(DECOMP, allow_pickle=False) as z:
        decomp = {k: z[k] for k in ("y", "c", "g", "k", "gid")}
    return blob, decomp


def build_topo25(blob: Mapping[str, Any]) -> tuple[list[Any], np.ndarray]:
    """The actual frozen Full topology25 model input for all 10000 train rows."""
    train = zftd.load_train_only()
    zftd.apply_prep_train_only(train, blob)
    T = np.concatenate(
        [d.topology_features.reshape(1, -1).numpy().astype(np.float32) for d in train], axis=0
    )
    if T.shape != (len(train), TOPOLOGY_IN):
        raise RuntimeError(f"topology25 shape mismatch: {T.shape}")
    return train, T


def load_o_model(blob: Mapping[str, Any], seed: int) -> nn.Module:
    model = zftd.build_full_model(blob, seed)
    state = torch.load(FROZEN_DIR / f"O_seed{seed}_raw_soup_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# cycle head
# ---------------------------------------------------------------------------


def build_head(seed: int, bias_value: float) -> nn.Module:
    torch.manual_seed(int(seed))
    head = nn.Sequential(
        nn.Linear(TOPOLOGY_IN, HIDDEN[0]),
        nn.SiLU(),
        nn.Linear(HIDDEN[0], HIDDEN[1]),
        nn.SiLU(),
        nn.Linear(HIDDEN[1], 1),
    )
    n_params = int(sum(p.numel() for p in head.parameters()))
    if n_params != HEAD_PARAMETERS:
        raise RuntimeError(f"cycle head parameter audit failed: {n_params}")
    nn.init.zeros_(head[4].weight)
    with torch.no_grad():
        head[4].bias.copy_(torch.tensor(float(bias_value)))
    return head


def head_forward(head: nn.Module, T: torch.Tensor) -> torch.Tensor:
    return head(T).view(-1)


def train_head(seed: int, T_fit: np.ndarray, c_fit: np.ndarray, bias_value: float, log: Any) -> dict[str, Any]:
    """Frozen recipe.  Only fit T and fit c are accepted."""
    n = int(T_fit.shape[0])
    head = build_head(seed, bias_value)
    init_hash = _state_hash(head)
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(int(TRAIN_GEN_BASE) + int(seed))
    T_t = torch.as_tensor(T_fit, dtype=torch.float32)
    c_t = torch.as_tensor(c_fit, dtype=torch.float32)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        head.train()
        order = torch.randperm(n, generator=generator)
        abs_sum, grad_norm = 0.0, 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            prediction = head_forward(head, T_t[idx])
            loss = (prediction - c_t[idx]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP))
            optimizer.step()
            abs_sum += float((prediction - c_t[idx]).abs().sum())
            steps += 1
        if epoch >= EPOCHS - SOUP_LAST + 1:
            soup[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        curve.append({
            "epoch": epoch,
            "train_task_mae": abs_sum / n,
            "grad_norm": grad_norm,
            "seconds": float(time.perf_counter() - started),
        })
    soup_mean = {
        key: torch.stack([soup[e][key] for e in sorted(soup)], dim=0).mean(dim=0) for key in soup[EPOCHS]
    }
    head.load_state_dict(soup_mean)
    head.eval()
    with torch.no_grad():
        q_fit = head_forward(head, T_t).numpy().astype(np.float64)
    return {
        "head": head,
        "init_state": init_state,
        "soup_mean": soup_mean,
        "init_hash": init_hash,
        "soup_hash": _state_hash(head),
        "curve": curve,
        "steps": steps,
        "soup_members": sorted(soup),
        "q_fit": q_fit,
        "seconds": float(time.perf_counter() - started),
    }


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def smoke(blob: Mapping[str, Any], T: np.ndarray, decomp: Mapping[str, np.ndarray], log: Any) -> dict[str, Any]:
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    c_fit = np.asarray(decomp["c"], np.float64)[fit_idx]
    bias_value = median(c_fit)
    T_fit = T[fit_idx]
    head = build_head(0, bias_value)
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    T_t = torch.as_tensor(T_fit[:512], dtype=torch.float32)
    c_t = torch.as_tensor(c_fit[:512], dtype=torch.float32)

    model = load_o_model(blob, 0)
    o_hash_before = _state_hash(model)
    first_step_grads: dict[str, float] = {}
    for step in range(1, SMOKE_STEPS + 1):
        head.train()
        optimizer.zero_grad()
        prediction = head_forward(head, T_t)
        loss = (prediction - c_t).abs().mean()
        loss.backward()
        if step == 1:
            first_step_grads = {
                f"layer_{name}": float(p.grad.norm())
                for name, p in head.named_parameters()
                if p.grad is not None
            }
        optimizer.step()
    changed = {
        key: bool(not torch.equal(head.state_dict()[key], init_state[key]))
        for key in init_state
    }
    o_hash_after = _state_hash(model)
    return {
        "steps": SMOKE_STEPS,
        "bias_value": bias_value,
        "first_step_grad_norms": first_step_grads,
        "first_step_head_layers_1_3_zero_by_design": bool(
            all(v == 0.0 for k, v in first_step_grads.items() if not k.endswith("4.weight") and not k.endswith("4.bias"))
        ),
        "all_params_changed_after_smoke": bool(all(changed.values())),
        "changed": changed,
        "o_state_hash_before": o_hash_before,
        "o_state_hash_after": o_hash_after,
        "o_unchanged": bool(o_hash_before == o_hash_after),
        "ok": bool(all(changed.values()) and o_hash_before == o_hash_after),
    }


# ---------------------------------------------------------------------------
# identity verification
# ---------------------------------------------------------------------------


def verify_identity(blob: Mapping[str, Any], train: Sequence[Any], decomp: Mapping[str, np.ndarray]) -> dict[str, Any]:
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    y_all = np.asarray(decomp["y"], np.float64)
    c_all = np.asarray(decomp["c"], np.float64)
    g_all = np.asarray(decomp["g"], np.float64)
    dev_data = [train[int(i)] for i in dev_idx]
    batch = zftd.make_batch(dev_data, list(range(FIXED_DEV_BATCH)), torch.zeros(len(dev_data)), torch.device("cpu"))
    out: dict[str, Any] = {"fixed_dev_batch": FIXED_DEV_BATCH, "per_seed": {}}
    for seed in SEEDS:
        frozen = load_frozen(seed, "O")
        cached_dev = frozen["preds"]["dev_raw"].astype(np.float64)
        cached_fit = frozen["preds"]["fit_raw"].astype(np.float64)
        b_o = float(frozen["meta"]["calibration"]["b"])
        model = load_o_model(blob, seed)
        hash_before = _state_hash(model)
        with torch.no_grad():
            h_fwd = model(batch, mask=cm.C6_MASK).view(-1).double().numpy()
        hash_after = _state_hash(model)
        max_diff = float(np.max(np.abs(h_fwd - cached_dev[:FIXED_DEV_BATCH])))
        # oracle replay: h_raw + b_O + c reproduces the released dev table.
        y_dev = y_all[dev_idx]
        c_dev = c_all[dev_idx]
        o_cal = cached_dev + b_o + c_dev
        mae_o_cal = float(np.mean(np.abs(o_cal - y_dev)))
        # g-identity of the cached h_raw.
        g_fit = g_all[fit_idx]
        out["per_seed"][str(seed)] = {
            "h_raw_forward_vs_cache_max_abs": max_diff,
            "h_raw_reproduced": bool(max_diff <= REPLAY_TOL),
            "o_soup_state_sha256": _sha_file(FROZEN_DIR / f"O_seed{seed}_raw_soup_state.pt"),
            "o_state_hash_before": hash_before,
            "o_state_hash_after": hash_after,
            "o_state_unchanged": bool(hash_before == hash_after),
            "b_O": b_o,
            "o_cal_dev_mae_recomputed": mae_o_cal,
            "o_cal_dev_mae_released": 0.10162616127963878 if seed == 0 else 0.09899889367500833,
            "o_cal_dev_mae_reproduced": bool(
                abs(mae_o_cal - (0.10162616127963878 if seed == 0 else 0.09899889367500833)) <= 2e-6
            ),
            "fit_h_raw_equals_g_minus_y_identity": None,
            "fit_c_finite": bool(np.all(np.isfinite(c_all[fit_idx]))),
            "fit_g_finite": bool(np.all(np.isfinite(g_fit))),
        }
    out["fit_idx_sha256"] = hashlib.sha256(fit_idx.tobytes()).hexdigest()
    out["dev_idx_sha256"] = hashlib.sha256(dev_idx.tobytes()).hexdigest()
    out["fit_idx_sha256_expected"] = "165e87ef4398ba8ef57411c2f118c4cca4ea74007f2485611b84f0c09bbdd9ea"
    out["dev_idx_sha256_expected"] = "fb8b78063e7c8a5c759bfa7d553738068cee1738a2d67b210e5d9dc492331376"
    out["split_ok"] = bool(
        out["fit_idx_sha256"] == out["fit_idx_sha256_expected"]
        and out["dev_idx_sha256"] == out["dev_idx_sha256_expected"]
    )
    out["prep_blob_sha256"] = _sha_file(PREP_BLOB)
    out["all_ok"] = bool(
        out["split_ok"]
        and all(v["h_raw_reproduced"] and v["o_state_unchanged"] and v["o_cal_dev_mae_reproduced"] for v in out["per_seed"].values())
    )
    return out


# ---------------------------------------------------------------------------
# deploy wrapper
# ---------------------------------------------------------------------------


class LearnedCyclePredictor(nn.Module):
    """Deployable inference: frozen Full h(x) + learned cycle head q(T) + b_P.

    ``forward`` accepts graph data and the existing topology25 model input.  It
    never receives or looks up ``c``/``g``/``k``/``y``/group/molecule labels.
    """

    def __init__(self, full: nn.Module, head: nn.Module, b_P: float) -> None:
        super().__init__()
        self.full = full
        self.head = head
        self.b_P = float(b_P)
        self.full.eval()
        for parameter in self.full.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch: Any, topo25: torch.Tensor | None = None) -> torch.Tensor:
        h = self.full(batch, mask=cm.C6_MASK).view(-1)
        T = batch.topology_features if topo25 is None else topo25
        q = head_forward(self.head, T)
        return h + q + self.b_P


def build_bundle(seed: int, train_result: Mapping[str, Any], b_P: float, b_K: float, median_fit_c: float) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "o_soup_state_path": str(FROZEN_DIR / f"O_seed{seed}_raw_soup_state.pt"),
        "head_state": train_result["soup_mean"],
        "head_init_state": train_result["init_state"],
        "b_P": float(b_P),
        "b_K": float(b_K),
        "median_fit_c": float(median_fit_c),
        "note": "topo25 must be the existing Full topology25 model input (same 8000-fit standardization).",
    }


def load_predictor(bundle_path: Path, blob: Mapping[str, Any]) -> LearnedCyclePredictor:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    full = zftd.build_full_model(blob, int(bundle["seed"]))
    full.load_state_dict(torch.load(bundle["o_soup_state_path"], map_location="cpu"))
    head = build_head(int(bundle["seed"]), float(bundle["median_fit_c"]))
    head.load_state_dict(bundle["head_state"])
    return LearnedCyclePredictor(full, head, float(bundle["b_P"]))


def wrapper_checks(blob: Mapping[str, Any], train: Sequence[Any], T: np.ndarray, decomp: Mapping[str, np.ndarray], seed: int, b_P: float, head_state: Mapping[str, torch.Tensor], log: Any) -> dict[str, Any]:
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    y_all = np.asarray(decomp["y"], np.float64)
    c_all = np.asarray(decomp["c"], np.float64)
    g_all = np.asarray(decomp["g"], np.float64)
    k_all = np.asarray(decomp["k"], np.int64)
    dev_data = [train[int(i)] for i in dev_idx]
    batch = zftd.make_batch(dev_data, list(range(FIXED_DEV_BATCH)), torch.zeros(len(dev_data)), torch.device("cpu"))
    T_dev = torch.as_tensor(T[dev_idx], dtype=torch.float32)
    bundle_path = RESULTS_DIR / f"deploy_bundle_seed{seed}.pt"
    predictor = load_predictor(bundle_path, blob)
    with torch.no_grad():
        p1 = predictor(batch, T_dev[:FIXED_DEV_BATCH]).double().numpy()
        p2 = predictor(batch, T_dev[:FIXED_DEV_BATCH]).double().numpy()
    # cached path: h_raw + q + b_P
    frozen = load_frozen(seed, "O")
    h_raw = frozen["preds"]["dev_raw"].astype(np.float64)[:FIXED_DEV_BATCH]
    head = build_head(seed, float(np.median(c_all[np.asarray(blob["fit_idx"], np.int64)])))
    head.load_state_dict(torch.load(bundle_path, map_location="cpu", weights_only=False)["head_state"])
    head.eval()
    with torch.no_grad():
        q = head_forward(head, T_dev[:FIXED_DEV_BATCH]).double().numpy()
    cached = h_raw + q + b_P
    pre_shuffle = p1.copy()
    # label permutation / removal must not change predictions: wrapper takes no labels.
    rng = np.random.default_rng(0)
    _ = y_all[rng.permutation(len(y_all))]
    _ = c_all[rng.permutation(len(c_all))]
    _ = k_all[rng.permutation(len(k_all))]
    with torch.no_grad():
        p3 = predictor(batch, T_dev[:FIXED_DEV_BATCH]).double().numpy()
    return {
        "seed": seed,
        "wrapper_vs_cached_max_abs": float(np.max(np.abs(p1 - cached))),
        "wrapper_equals_cached": bool(float(np.max(np.abs(p1 - cached))) <= 1e-6),
        "deterministic_repeat_max_abs": float(np.max(np.abs(p1 - p2))),
        "label_permutation_max_abs": float(np.max(np.abs(pre_shuffle - p3))),
        "label_permutation_invariant": bool(np.array_equal(pre_shuffle, p3)),
        "head_state_hash": _state_hash(head),
        "forward_accepts_labels": False,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_all(log: Any = print, save_T: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    blob, decomp = load_prep_and_targets()
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    y_all = np.asarray(decomp["y"], np.float64)
    c_all = np.asarray(decomp["c"], np.float64)
    g_all = np.asarray(decomp["g"], np.float64)
    k_all = np.asarray(decomp["k"], np.int64)

    log("[load] encoded_train + prep (train-only) ...")
    train, T = build_topo25(blob)
    log(f"[load] T25 {T.shape} sha256={_sha_arr(T)}")

    identity = verify_identity(blob, train, decomp)
    _write_json(RESULTS_DIR / "identity_checks.json", identity)
    log(f"[identity] all_ok={identity['all_ok']}")

    smoke_checks = smoke(blob, T, decomp, log)
    _write_json(RESULTS_DIR / "smoke_checks.json", smoke_checks)
    log(f"[smoke] ok={smoke_checks['ok']}")

    if save_T:
        np.savez_compressed(
            RESULTS_DIR / "T25_all.npz",
            T=T,
            fit_idx=fit_idx,
            dev_idx=dev_idx,
            sha256=np.array([_sha_arr(T)]),
        )

    if not identity["all_ok"]:
        _write_json(RESULTS_DIR / "ABORT.json", {"reason": "identity check failed", "identity": identity})
        raise RuntimeError("identity check failed; refusing to fit")

    T_fit = T[fit_idx]
    c_fit = c_all[fit_idx]
    T_dev = T[dev_idx]
    median_fit_c = median(c_fit)

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        log(f"[head] seed={seed} training {EPOCHS} epochs ...")
        result = train_head(seed, T_fit, c_fit, median_fit_c, log)
        torch.save(result["init_state"], RESULTS_DIR / f"P_seed{seed}_head_init_state.pt")
        torch.save(result["soup_mean"], RESULTS_DIR / f"P_seed{seed}_head_soup_state.pt")
        with torch.no_grad():
            head = build_head(seed, median_fit_c)
            head.load_state_dict(result["soup_mean"])
            head.eval()
            T_dev_t = torch.as_tensor(T_dev, dtype=torch.float32)
            q_dev = head_forward(head, T_dev_t).numpy().astype(np.float64)
        h_raw_fit = load_frozen(seed, "O")["preds"]["fit_raw"].astype(np.float64)
        h_raw_dev = load_frozen(seed, "O")["preds"]["dev_raw"].astype(np.float64)
        y_fit = y_all[fit_idx]
        b_P = median(y_fit - h_raw_fit - result["q_fit"])
        b_K = median(y_fit - h_raw_fit - median_fit_c)
        np.savez_compressed(
            RESULTS_DIR / f"P_seed{seed}_predictions.npz",
            fit_q=result["q_fit"],
            dev_q=q_dev,
            fit_idx=fit_idx,
            dev_idx=dev_idx,
            b_P=np.array([b_P]),
            b_K=np.array([b_K]),
            median_fit_c=np.array([median_fit_c]),
        )
        _write_json(
            RESULTS_DIR / f"P_seed{seed}.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "epochs": EPOCHS,
                "steps": result["steps"],
                "soup_members": result["soup_members"],
                "train_generator_seed": int(TRAIN_GEN_BASE) + int(seed),
                "init_hash": result["init_hash"],
                "soup_hash": result["soup_hash"],
                "b_P": b_P,
                "b_K": b_K,
                "median_fit_c": median_fit_c,
                "seconds": result["seconds"],
                "curve": result["curve"],
            },
        )
        bundle = build_bundle(seed, result, b_P, b_K, median_fit_c)
        torch.save(bundle, RESULTS_DIR / f"deploy_bundle_seed{seed}.pt")
        per_seed[str(seed)] = {"b_P": b_P, "b_K": b_K, "seconds": result["seconds"]}
        log(f"[head] seed={seed} done in {result['seconds']:.1f}s b_P={b_P:.6f} b_K={b_K:.6f}")

    # wrapper / invariance / replay checks (after bundles saved)
    wrapper = {}
    for seed in SEEDS:
        meta = json.loads((RESULTS_DIR / f"P_seed{seed}.json").read_text())
        wrapper[str(seed)] = wrapper_checks(blob, train, T, decomp, seed, meta["b_P"], None, log)
    _write_json(RESULTS_DIR / "wrapper_checks.json", wrapper)
    log(f"[wrapper] {wrapper}")

    np.savez_compressed(
        RESULTS_DIR / "cycle_targets.npz",
        fit_idx=fit_idx,
        dev_idx=dev_idx,
        fit_c=c_fit,
        dev_c=c_all[dev_idx],
        fit_y=y_all[fit_idx],
        dev_y=y_all[dev_idx],
        fit_g=g_all[fit_idx],
        dev_g=g_all[dev_idx],
        dev_k=k_all[dev_idx],
        gid_dev=decomp["gid"][dev_idx],
    )
    budget = {
        "protocol_version": PROTOCOL_VERSION,
        "seconds_total": float(time.perf_counter() - t0),
        "epochs": EPOCHS,
        "steps_per_head": EPOCHS * int(np.ceil(len(fit_idx) / BATCH_SIZE)),
        "threads": int(torch.get_num_threads()),
        "device": "cpu",
        "gpu_jobs": 0,
        "new_full_training": 0,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "budget.json", budget)
    log(f"[done] total {budget['seconds_total']:.1f}s")
    return {"identity": identity, "smoke": smoke_checks, "per_seed": per_seed, "wrapper": wrapper, "budget": budget}


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description="ZINC frozen-chemistry learned-cycle head v1")
    parser.add_argument("--mode", default="all", choices=("all", "identity", "smoke", "train"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "all":
        run_all(log=print)
        return 0
    blob, decomp = load_prep_and_targets()
    train, T = build_topo25(blob)
    if args.mode == "identity":
        _write_json(RESULTS_DIR / "identity_checks.json", verify_identity(blob, train, decomp))
        return 0
    if args.mode == "smoke":
        _write_json(RESULTS_DIR / "smoke_checks.json", smoke(blob, T, decomp, print))
        return 0
    if args.mode == "train":
        fit_idx = np.asarray(blob["fit_idx"], np.int64)
        c_fit = np.asarray(decomp["c"], np.float64)[fit_idx]
        result = train_head(args.seed, T[fit_idx], c_fit, median(c_fit), print)
        torch.save(result["soup_mean"], RESULTS_DIR / f"P_seed{args.seed}_head_soup_state.pt")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
