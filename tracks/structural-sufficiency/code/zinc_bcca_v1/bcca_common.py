"""ZINC bond-conditioned context attention v1 — shared paths, frozen-data access
and contrast metrics.

Round: ``zinc-bond-conditioned-context-attention-v1``
Protocol: ``results/zinc_bond_conditioned_context_attention_v1/PROTOCOL.md``

The frozen data paths, hashing rules, metric primitives and label loaders are
reused from the completed Phase 2b module (read-only).  This module only adds
the per-run record paths, the arm/seed matrix and the generic paired contrasts
(C vs P3 / G / F etc.).
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

import bcca_models as BM  # noqa: E402
import phase2b_common as P2B  # noqa: E402

PROTOCOL_ID = "zinc-bond-conditioned-context-attention-v1"
ROUND = "zinc_bond_conditioned_context_attention_v1"
TRACK = REPO / "tracks/structural-sufficiency"
OUT = TRACK / f"results/{ROUND}"
CONFIG_PATH = TRACK / f"configs/{ROUND}.yaml"
PROTOCOL_PATH = OUT / "PROTOCOL.md"

SEEDS = (0, 1)
NEW_ARMS = ("F", "G", "C")
CONTEXT_ARMS = ("G", "C")
ALL_MODELS = ("P3", "F", "G", "C")

#: frozen recipe (identical to R3 / Phase 2b)
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

#: frozen statistics
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20261019
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = (1, 10)
EXTREME_VALID_ROW = 172

N_TRAIN = 10000
N_VALID = 1000

EXPECTED_PARAMETERS = BM.EXPECTED_PARAMETERS

# re-exports used by the runner/tests
sha256_file = P2B.sha256_file
sha256_array = P2B.sha256_array
read_json = P2B.read_json
write_json = P2B.write_json
write_once_json = P2B.write_once_json
abs_metrics = P2B.abs_metrics
paired_bootstrap = P2B.paired_bootstrap
seed_averaged_delta = P2B.seed_averaged_delta
mse_delta = P2B.mse_delta
error_rank = P2B.error_rank
top_removed_gain = P2B.top_removed_gain
subgroup_metrics = P2B.subgroup_metrics
cycle_proxy = P2B.cycle_proxy
predict = P2B.predict
load_soup = P2B.load_soup
torch_load = P2B.torch_load
p3_run = P2B.p3_run
p3_soup_path = P2B.p3_soup_path


# ---------------------------------------------------------------------------
# configuration / frozen inputs
# ---------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    import yaml

    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def verify_file(name: str, path: Path, expected: str) -> dict[str, Any]:
    return P2B.verify_file(name, path, expected)


def frozen_input(name: str) -> tuple[Path, dict[str, Any]]:
    spec = load_config()["frozen_inputs"][name]
    path = REPO / spec["path"]
    return path, spec


def load_objects(name: str) -> list[Any]:
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


def p3_soup_path_expected(seed: int) -> tuple[Path, str]:
    path, spec = frozen_input(f"soup_R3_seed{seed}")
    return path, spec["sha256"]


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_model(arm: str, seed: int, device: str | Any = "cpu") -> tuple[Any, dict[str, Any]]:
    """Exact P3 supervised init + seed-matched full-10k SSL + this round's arm."""
    import torch

    if arm not in BM.VARIANTS:
        raise SystemExit(f"unknown arm {arm!r}")
    if seed not in SEEDS:
        raise SystemExit(f"unknown seed {seed}")
    cfg = load_config()
    norm = load_norm()
    state, ssl_rec = ssl_state(seed)
    base, model, info = BM.build_variant(arm, seed=seed, norm=norm, ssl_state=state, device="cpu")
    rec = p3_run(seed)["init"]
    exp = cfg["audit_expectations"]
    checks = {
        "trunk_matches_base_init": bool(info["shared_matches_base"]),
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
        raise SystemExit(f"init/SSL provenance check failed for {arm} s{seed}: {checks}")
    audit = BM.parameter_audit(model)
    if not audit["parameter_exact"]:
        raise SystemExit(f"parameter audit failed for {arm}: {audit}")
    mod = BM.module_audit(model)
    if not mod["pass"]:
        raise SystemExit(f"module audit failed for {arm}: {mod}")
    model = model.to(torch.device(device))
    return model, {
        "arm": arm,
        "seed": int(seed),
        "parameters": audit["actual_parameters"],
        "parameter_audit": audit,
        "module_audit": mod,
        "build": info,
        "checks": checks,
        "ssl": ssl_rec,
        "encoder_used_sha256": info["encoder_used_sha256"],
    }


def build_p3(seed: int, device: str | Any = "cpu") -> tuple[Any, dict[str, Any]]:
    """Historical frozen P3 soup (only needed for the audit/replay or a baseline)."""
    path, expected_sha = p3_soup_path_expected(seed)
    model, info = build_model("P3", seed, "cpu")
    soup = load_soup(model, path)
    if soup["sha256"] != expected_sha:
        raise SystemExit(f"P3 soup hash mismatch for seed {seed}")
    return model.to(device), {
        **info,
        "soup": soup,
    }


# ---------------------------------------------------------------------------
# historical P3 valid predictions
# ---------------------------------------------------------------------------


def p3_valid_pred(seed: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Frozen historical valid prediction (file+array hash verified) or None.

    ``None`` means the gitignored array file is absent on this checkout; the
    audit then performs a label-free replay from the hash-verified soup and
    freezes that array (PROTOCOL §5 branch B).
    """
    cfg = load_config()
    prov = dict(cfg["historical_prediction_provenance"][f"valid_P3{seed}"])
    path = REPO / prov["historical_path"]
    if not path.exists():
        return None, {**prov, "present": False}
    verify_file(f"valid_P3{seed}", path, prov["historical_file_sha256"])
    with np.load(path) as z:
        pred = np.asarray(z["pred"], dtype=np.float64)
    if pred.shape != (N_VALID,):
        raise SystemExit(f"unexpected P3 valid prediction shape {pred.shape}")
    if sha256_array(pred) != prov["array_sha256_float32"]:
        raise SystemExit("historical P3 valid prediction array hash mismatch")
    return pred, {**prov, "present": True}


# ---------------------------------------------------------------------------
# per-run record paths
# ---------------------------------------------------------------------------


def train_path(arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_s{seed}.json"


def soup_path(arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_s{seed}.pt"


def valid_pred_path(arm: str, seed: int) -> Path:
    return OUT / f"valid_pred_{arm}_s{seed}.npz"


def valid_freeze_path(arm: str, seed: int) -> Path:
    return OUT / f"valid_freeze_{arm}_s{seed}.json"


def mech_pred_path(name: str, seed: int, mode: str) -> Path:
    return OUT / f"mech_pred_{name}_s{seed}_{mode}.npz"


# ---------------------------------------------------------------------------
# contrasts
# ---------------------------------------------------------------------------


def per_seed_gain(cand: Mapping[int, np.ndarray], ref: Mapping[int, np.ndarray], y: np.ndarray) -> dict[str, float]:
    """Positive = candidate better (lower MAE)."""
    return {
        str(s): float(np.abs(np.asarray(ref[s]) - y).mean() - np.abs(np.asarray(cand[s]) - y).mean())
        for s in SEEDS
    }


def contrast(
    cand_name: str,
    ref_name: str,
    preds: Mapping[str, Mapping[int, np.ndarray]],
    y: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Frozen contrast block: per seed, seed mean, paired CI, robustness."""
    cand, ref = preds[cand_name], preds[ref_name]
    gains = per_seed_gain(cand, ref, y)
    d = seed_averaged_delta(cand, ref, y)
    d_mse = mse_delta(cand, ref, y)
    p3_err_order = error_rank(preds["P3"], y)
    abs_contrib = np.abs(d)
    top = int(np.argmax(abs_contrib))
    out = {
        "candidate": cand_name,
        "reference": ref_name,
        "definition": f"MAE({ref_name}) - MAE({cand_name}); positive = {cand_name} better",
        "per_seed_gain": gains,
        "mean_gain": float(np.mean(list(gains.values()))),
        "mae_seed_mean": {name: float(np.mean([np.abs(preds[name][s] - y).mean() for s in SEEDS]))
                          for name in (cand_name, ref_name)},
        "bootstrap_mae_delta": paired_bootstrap(d, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED),
        "bootstrap_mse_delta": paired_bootstrap(d_mse, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED),
        "better_fraction": float((d > 0).mean()),
        "drop_top1_favourable_gain": top_removed_gain(d, [top]),
        "drop_top10_p3_error_gain": top_removed_gain(d, p3_err_order[:10]),
        "max_single_molecule_contribution_share": float(abs_contrib[top] / max(abs_contrib.sum(), 1e-12)),
        "max_single_molecule_row": top,
        "ordinary_gain": float(d[masks["ordinary_y_gt_minus3"]].mean()),
        "tail_gain": float(d[masks["tail_y_le_minus3"]].mean()),
        "penalty_complement_gain": float(d[masks["penalty_complement"]].mean()),
        "without_extreme_row_gain": float(d[masks["without_extreme_valid172"]].mean()),
        "per_seed_mse_delta": {
            str(s): float(np.mean((np.asarray(ref[s]) - y) ** 2) - np.mean((np.asarray(cand[s]) - y) ** 2))
            for s in SEEDS
        },
    }
    return out


def performance_promising(c: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_gain_ge_threshold": c["mean_gain"] >= float(thresholds["performance_mean_gain"]),
        "both_seeds_positive": all(v > 0 for v in c["per_seed_gain"].values()),
        "paired_ci_lower_gt_zero": c["bootstrap_mae_delta"]["ci95_low"] > 0.0,
        "ordinary_gain_positive": c["ordinary_gain"] > 0.0,
        "drop_top10_p3_error_positive": c["drop_top10_p3_error_gain"] > 0.0,
    }
    return {"checks": checks, "pass": all(checks.values()), "thresholds": dict(thresholds)}


def conditioning_promising(c: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_gain_ge_threshold": c["mean_gain"] >= float(thresholds["conditioning_mean_gain"]),
        "both_seeds_positive": all(v > 0 for v in c["per_seed_gain"].values()),
        "paired_ci_lower_gt_zero": c["bootstrap_mae_delta"]["ci95_low"] > 0.0,
    }
    return {"checks": checks, "pass": all(checks.values()), "thresholds": dict(thresholds)}


def capacity_claim(c: Mapping[str, Any]) -> dict[str, Any]:
    """C vs F: only the pre-registered conjunction may be called 'better'."""
    checks = {
        "both_seeds_positive": all(v > 0 for v in c["per_seed_gain"].values()),
        "paired_ci_lower_gt_zero": c["bootstrap_mae_delta"]["ci95_low"] > 0.0,
    }
    return {"checks": checks, "pass": all(checks.values())}


def seed_mean_mae(preds: Mapping[str, Mapping[int, np.ndarray]], name: str, y: np.ndarray) -> float:
    return float(np.mean([np.abs(preds[name][s] - y).mean() for s in SEEDS]))
