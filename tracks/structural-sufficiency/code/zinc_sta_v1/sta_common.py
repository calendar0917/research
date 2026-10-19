"""ZINC structural-token attention v1 — frozen inputs, model construction, metrics.

Round: ``zinc-structural-token-attention-v1``
Protocol: ``results/zinc_structural_token_attention_v1/PROTOCOL.md``

Data and P3 provenance are reused read-only from the completed Phase 2b /
bond-conditioned attention rounds.  This module adds only the token-task
construction, prediction and round-local statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (
    HERE,
    REPO / "tracks/structural-sufficiency/code/phase2b",
    REPO / "tracks/structure-contribution/code",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase2b_common as P2B  # noqa: E402
import sta_data as D  # noqa: E402
import sta_models as M  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "zinc-structural-token-attention-v1"
ROUND = "zinc_structural_token_attention_v1"
TRACK = REPO / "tracks/structural-sufficiency"
OUT = TRACK / f"results/{ROUND}"
CONFIG_PATH = TRACK / f"configs/{ROUND}.yaml"
PROTOCOL_PATH = OUT / "PROTOCOL.md"

SEEDS = (0, 1)
ARMS = ("S", "T", "R")
ALL_MODELS = ("P3",) + ARMS

#: frozen recipe (R3, identical to Phase 2b / C-G round)
RECIPE_KEY = "R3"
EPOCHS = 360
BATCH_SIZE = 128
STEPS_PER_EPOCH = 79
UPDATES = EPOCHS * STEPS_PER_EPOCH
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
GRAD_CLIP = 5.0
LR = 1.0e-3
LR_MIN = 1.0e-5
WEIGHT_DECAY = 1.0e-5

#: round-local statistics (frozen before the score)
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20261019
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
EXTREME_VALID_ROW = 172
PRACTICAL_MEAN_GAIN = 0.003

N_TRAIN = 10000
N_VALID = 1000
EXPECTED_PARAMETERS = M.EXPECTED_PARAMETERS

# re-exports
sha256_file = P2B.sha256_file
sha256_array = P2B.sha256_array
read_json = P2B.read_json
write_json = P2B.write_json
write_once_json = P2B.write_once_json
abs_metrics = P2B.abs_metrics
mse_delta = P2B.mse_delta
subgroup_metrics = P2B.subgroup_metrics
cycle_proxy = P2B.cycle_proxy


# ---------------------------------------------------------------------------
# config / frozen inputs
# ---------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    import yaml

    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def verify_file(name: str, path: Path, expected: str) -> dict[str, Any]:
    return P2B.verify_file(name, path, expected)


def frozen_input(name: str) -> tuple[Path, dict[str, Any]]:
    spec = load_config()["frozen_inputs"][name]
    return REPO / spec["path"], spec


def load_objects(name: str) -> list[sro.MolObjects]:
    path, spec = frozen_input(name)
    verify_file(name, path, spec["sha256"])
    return P2B.torch_load(path)


def load_norm() -> dict[str, Any]:
    path, spec = frozen_input("prep_train10k")
    verify_file("prep_train10k", path, spec["sha256"])
    norm = read_json(path)
    if not norm.get("fit_only") or int(norm.get("n_fit", 0)) != N_TRAIN:
        raise SystemExit("standardiser is not the frozen train-10k-only one")
    return norm


def load_train_y() -> np.ndarray:
    y = P2B.load_train_y()
    expected = load_config()["frozen_inputs"]["encoded_train"]["train_y_sha256_float64"]
    if sha256_array(y, "float64") != expected:
        raise SystemExit("official-train raw y does not match the frozen hash")
    return y


def load_valid_y() -> np.ndarray:
    y = P2B.load_valid_y()
    expected = load_config()["label_provenance"]["valid_y"]["array_sha256_float32"]
    if sha256_array(y) != expected:
        raise SystemExit("official-valid labels do not match the frozen array hash")
    return y


def ssl_state(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path, spec = frozen_input(f"pretrain_encoder_s{seed}")
    rec = verify_file(f"pretrain_encoder_s{seed}", path, spec["sha256"])
    return torch_load(path), rec


def torch_load(path: Path | str):
    return P2B.torch_load(path)


def load_soup(model: Any, path: Path) -> dict[str, Any]:
    return P2B.load_soup(model, path)


def p3_soup_path(seed: int) -> Path:
    path, _ = frozen_input(f"soup_R3_seed{seed}")
    return path


def p3_soup_sha_expected(seed: int) -> str:
    return load_config()["frozen_inputs"][f"soup_R3_seed{seed}"]["sha256"]


def p3_run(seed: int) -> dict[str, Any]:
    return read_json(P2B.RECIPE / f"train_R3_seed{seed}.json")


def historical_valid_pred(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen historical P3 valid prediction, verified against this round's config."""
    key = f"valid_P3{seed}"
    prov = load_config()["historical_prediction_provenance"][key]
    path = REPO / prov["historical_path"]
    verify_file(key, path, prov["historical_file_sha256"])
    with np.load(path) as z:
        pred = np.asarray(z["pred"], dtype=np.float64)
    if pred.shape != (N_VALID,):
        raise SystemExit(f"unexpected historical P3 valid shape {pred.shape}")
    if sha256_array(pred) != prov["array_sha256_float32"]:
        raise SystemExit("historical P3 valid array hash mismatch")
    return pred, {
        "path": prov["historical_path"],
        "file_sha256": prov["historical_file_sha256"],
        "array_sha256_float32": prov["array_sha256_float32"],
    }


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_model(variant: str, seed: int, device: str | Any = "cpu") -> tuple[Any, dict[str, Any]]:
    """Exact P3 supervised start + seed-matched full-10k SSL + this arm."""
    import torch

    if variant not in ARMS:
        raise SystemExit(f"unknown arm {variant!r}")
    if seed not in SEEDS:
        raise SystemExit(f"unknown seed {seed}")
    cfg = load_config()
    norm = load_norm()
    state, ssl_rec = ssl_state(seed)
    base, model, info = M.build_sta_model(variant, seed=seed, norm=norm, ssl_state=state, device="cpu")
    rec = p3_run(seed)["init"]
    exp = cfg["audit_expectations"]
    checks = {
        "shared_matches_base": bool(info["shared_matches_base"]),
        "scratch_full_matches_r3": info["scratch"]["init_full_sha256"] == exp["init_full_sha256"][str(seed)],
        "scratch_group_matches_r3": info["scratch"]["init_group_state_sha256"]
        == exp["init_group_state_sha256"][str(seed)],
        "scratch_encoder_matches_r3": info["scratch"]["init_encoder_sha256"]
        == exp["init_encoder_sha256"][str(seed)],
        "encoder_used_matches_r3": info["encoder_used_sha256"] == exp["encoder_used_sha256"][str(seed)],
        "encoder_file_matches_record": ssl_rec["sha256"] == rec["pretrain_encoder_file_sha256"],
        "shared_rng_untouched": bool(info["global_rng_untouched"]),
    }
    if not all(checks.values()):
        raise SystemExit(f"init/SSL provenance check failed for {variant} s{seed}: {checks}")
    audit = M.parameter_audit(model)
    if not audit["parameter_exact"]:
        raise SystemExit(f"parameter audit failed for {variant}: {audit}")
    mod = M.module_audit(model)
    if not mod["pass"]:
        raise SystemExit(f"module audit failed for {variant}: {mod}")
    model = model.to(torch.device(device))
    return model, {
        "arm": variant,
        "seed": int(seed),
        "parameters": audit["actual_parameters"],
        "parameter_audit": audit,
        "module_audit": mod,
        "build": info,
        "checks": checks,
        "ssl": ssl_rec,
        "encoder_used_sha256": info["encoder_used_sha256"],
    }


def build_p3_reference(seed: int, device: str | Any = "cpu") -> tuple[Any, dict[str, Any]]:
    """Historical frozen P3 soup, rebuilt from the same P3 start + SSL encoder."""
    import bjoc_masked_pretraining_v1 as mp
    import bjoc_perfmech_long_v1 as bp

    norm = load_norm()
    state, ssl_rec = ssl_state(seed)
    base = bp.build_bjoc_null(seed, norm, "cpu")
    scratch = {
        "init_full_sha256": M.src.state_dict_hash(base),
        "init_group_state_sha256": mp.group_state_hash(base),
        "init_encoder_sha256": bp.encoder_group_hash(base),
    }
    mp.load_encoder_state(base, state, "cpu")
    encoder_used = bp.encoder_group_hash(base)
    exp = load_config()["audit_expectations"]
    rec = p3_run(seed)["init"]
    checks = {
        "scratch_full_matches_r3": scratch["init_full_sha256"] == exp["init_full_sha256"][str(seed)],
        "scratch_group_matches_r3": scratch["init_group_state_sha256"]
        == exp["init_group_state_sha256"][str(seed)],
        "scratch_encoder_matches_r3": scratch["init_encoder_sha256"] == exp["init_encoder_sha256"][str(seed)],
        "encoder_used_matches_r3": encoder_used == exp["encoder_used_sha256"][str(seed)],
        "encoder_file_matches_record": ssl_rec["sha256"] == rec["pretrain_encoder_file_sha256"],
    }
    if not all(checks.values()):
        raise SystemExit(f"P3 rebuild provenance failed for seed {seed}: {checks}")
    path = p3_soup_path(seed)
    soup = load_soup(base, path)
    if soup["sha256"] != p3_soup_sha_expected(seed):
        raise SystemExit(f"P3 soup hash mismatch for seed {seed}")
    n_params = int(sum(p.numel() for p in base.parameters()))
    if n_params != M.P3_TOTAL:
        raise SystemExit(f"P3 parameter count {n_params} != {M.P3_TOTAL}")
    return base.to(device), {
        "seed": int(seed),
        "parameters": n_params,
        "scratch": scratch,
        "encoder_used_sha256": encoder_used,
        "soup": soup,
        "checks": checks,
        "ssl": ssl_rec,
    }


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


def predict_sta(
    model: Any,
    objects: Sequence[sro.MolObjects],
    cache: Sequence[Mapping[str, np.ndarray]],
    device: str | Any,
    batch_size: int = 128,
) -> tuple[np.ndarray, float]:
    import time

    import torch

    model.eval()
    out = []
    started = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(objects), batch_size):
            rows = list(range(i, min(i + batch_size, len(objects))))
            batch = D.batch_objects_sta(objects, rows, cache)
            batch = D.move_batch(batch, device)
            out.append(model(batch).detach().cpu().numpy())
    pred = np.concatenate(out, 0).astype(np.float64)
    return pred, time.perf_counter() - started


def predict_p3(
    seed: int, objects: Sequence[sro.MolObjects], device: str | Any = "cpu"
) -> tuple[np.ndarray, float]:
    import time

    import bjoc_perfmech_long_v1 as bp

    model, _info = build_p3_reference(seed, device)
    started = time.perf_counter()
    pred = bp.predict_model(model, objects, device)
    return pred.astype(np.float64), time.perf_counter() - started


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def seed_averaged_delta(
    pred_cand: Mapping[int, np.ndarray],
    pred_ref: Mapping[int, np.ndarray],
    y: np.ndarray,
) -> np.ndarray:
    """d_i = mean_s(|e_ref,s,i| - |e_cand,s,i|); same molecule resample for both seeds."""
    y = np.asarray(y, dtype=np.float64)
    parts = [
        np.abs(np.asarray(pred_ref[s], dtype=np.float64) - y)
        - np.abs(np.asarray(pred_cand[s], dtype=np.float64) - y)
        for s in SEEDS
    ]
    return np.mean(np.stack(parts, 0), axis=0)


def paired_bootstrap(delta: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(n), d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(d.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "better_fraction": float((d > 0).mean()),
        "n_resamples": int(n),
        "bootstrap_seed": int(seed),
        "per_molecule_unit": True,
        "two_seed_synchronised": True,
    }


def per_seed_gain(
    pred_cand: Mapping[int, np.ndarray],
    pred_ref: Mapping[int, np.ndarray],
    y: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    gains = {}
    for s in SEEDS:
        ea = np.abs(np.asarray(pred_ref[s], dtype=np.float64) - y)
        eb = np.abs(np.asarray(pred_cand[s], dtype=np.float64) - y)
        gains[f"seed{s}"] = float(ea.mean() - eb.mean())
    gains["mean"] = float(np.mean(list(gains.values())))
    return gains


def worst_indices(pred: Mapping[int, np.ndarray], y: np.ndarray, k: int = TOP_K_REMOVED) -> list[int]:
    y = np.asarray(y, dtype=np.float64)
    err = np.mean([np.abs(np.asarray(pred[s], dtype=np.float64) - y) for s in SEEDS], axis=0)
    return [int(i) for i in np.argsort(-err)[: int(k)]]


def subset_gain(delta: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    return float(np.mean(delta[idx])) if idx.size else float("nan")


def contrast_payload(
    cand: str,
    ref: str,
    preds: Mapping[str, Mapping[int, np.ndarray]],
    y: np.ndarray,
    p3_worst: Sequence[int],
) -> dict[str, Any]:
    delta = seed_averaged_delta(preds[cand], preds[ref], y)
    boot = paired_bootstrap(delta)
    gains = per_seed_gain(preds[cand], preds[ref], y)
    drop = np.ones(delta.size, dtype=bool)
    drop[list(p3_worst)] = False
    ordinary = np.asarray(y, dtype=np.float64) > TAIL_THRESHOLD
    no172 = np.ones(delta.size, dtype=bool)
    if EXTREME_VALID_ROW < delta.size:
        no172[EXTREME_VALID_ROW] = False
    return {
        "candidate": cand,
        "reference": ref,
        "gain": gains,
        "paired_bootstrap": boot,
        "robustness": {
            "ordinary_gt_-3": subset_gain(delta, ordinary),
            "drop_p3_worst10": subset_gain(delta, drop),
            "without_row172": subset_gain(delta, no172),
            "tail_le_-3": subset_gain(delta, ~ordinary),
        },
        "p3_worst10_rows": [int(i) for i in p3_worst],
    }


def practical_pass(contrast: Mapping[str, Any]) -> dict[str, Any]:
    gain = contrast["gain"]
    rob = contrast["robustness"]
    ci = contrast["paired_bootstrap"]
    checks = {
        "mean_gain_ge_0.003": bool(gain["mean"] >= PRACTICAL_MEAN_GAIN),
        "both_seeds_positive": bool(gain["seed0"] > 0.0 and gain["seed1"] > 0.0),
        "ordinary_positive": bool(rob["ordinary_gt_-3"] > 0.0),
        "drop_p3_worst10_positive": bool(rob["drop_p3_worst10"] > 0.0),
        "paired_ci_low_gt_0": bool(ci["ci95_low"] > 0.0),
    }
    return {"pass": all(checks.values()), "checks": checks}


def consistent_positive(contrast: Mapping[str, Any]) -> bool:
    gain = contrast["gain"]
    ci = contrast["paired_bootstrap"]
    return bool(gain["seed0"] > 0.0 and gain["seed1"] > 0.0 and ci["ci95_low"] > 0.0)
