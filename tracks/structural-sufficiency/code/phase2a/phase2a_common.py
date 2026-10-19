"""SSRC Phase 2a shared loaders, frozen-artifact verification and metrics.

Scope discipline
----------------
* Reads the SSRC Fit/Monitor split and the Fit+Monitor label vault only.
* Reuses the frozen Phase 1e P3E artefacts (soup, SSL encoder, predictions) by
  hash; never retrains them.
* The internal Reserved split and the official ZINC valid/test splits are never
  loaded.
* Monitor labels are read once, only in the ``score`` stage, after every
  candidate prediction array has been frozen and hashed.

Group definitions are *descriptive* and post-hoc (the Monitor was used by
Phases 1b-1e); they must never be used to select a configuration.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STRUCT_CODE = REPO / "tracks/structure-contribution/code"
for _p in (HERE, STRUCT_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS_1B = REPO / "tracks/structural-sufficiency/results/phase1b"
RESULTS_1E = REPO / "tracks/structural-sufficiency/results/phase1e"
RESULTS_2A = REPO / "tracks/structural-sufficiency/results/phase2a"
SPLIT_PATH = RESULTS_1B / "split_zinc_phase1b.json"
VAULT_PATH = RESULTS_1B / "zinc_labels_fit_monitor.json"
LABEL_AUDIT = RESULTS_1E / "label_audit.json"
ERROR_ATTR = RESULTS_1E / "error_attribution.json"
STRONG_BASELINE = RESULTS_1E / "strong_baseline.json"
BASELINE_DIR = RESULTS_1E / "baseline"
SSL_STATE = BASELINE_DIR / "pretrain_s0.pt"
P3E_SOUP = BASELINE_DIR / "train_P3E_s0_soup.pt"
P3E_RUN = BASELINE_DIR / "train_P3E_s0.json"
P3E_PRED = BASELINE_DIR / "monitor_pred_s0.npz"
P3E_FREEZE = BASELINE_DIR / "monitor_prediction_freeze.json"
P3E_PREP_MANIFEST = BASELINE_DIR / "prep_manifest.json"

PROTOCOL_ID = "ssrc-phase2a-zinc-v1"
P3E_MAE = 0.09107104176818394
P3E_RMSE = 0.2703102664227065
P3E_PARAMS = 413505
P3E_ENCODER_SHA = "37282569525d1e69a6172c23f8f2e357ba7f35892af6116ce980628c54c4d4d6"
P3E_SSL_FILE_SHA = "0d045e6090062ac6e919c79c82c2cbc9a223c7974122c91d5b68a244d265bc0c"
P3E_SOUP_SHA = "94ba0f5f14d87169610af8f3abdbc577ca30b0685e322ac4af896fb52eb530fd"
P3E_PRED_SHA = "fc698ccc82dba024f73baa4604e27443bc25d74c452ccd38b8685f2a6aa4d1b1"
P3E_BATCH_ORDER_SHA = "d142e307339b2869f4ad2d3ff8472e28249f1dd1ff887a4ae55f5b837d61c81f"

BOOT_N = 5000
BOOT_SEED = 20261023
PRACTICAL_BAND = 0.004
MP_SEPARATION_BAND = 0.002
RMSE_TOLERANCE = 0.005
COMPLEMENT_BAND = 0.002


# ---------------------------------------------------------------------------
# io / hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray, dtype: str = "float32") -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    return hashlib.sha256(a.tobytes()).hexdigest()


def rows_sha256(rows: Sequence[int]) -> str:
    return sha256_array(np.asarray(rows, dtype=np.int64), "int64")


def row_text_sha256(rows: Sequence[int]) -> str:
    """Phase 1a/1b row-index hash convention (comma-separated decimals)."""
    return hashlib.sha256(",".join(str(int(r)) for r in rows).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# split / labels / objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    fit: np.ndarray
    monitor: np.ndarray
    reserved: np.ndarray
    sha256: str


def load_split() -> Split:
    split = read_json(SPLIT_PATH)
    rows = split["rows"]
    fit = np.asarray(rows["fit"], dtype=np.int64)
    monitor = np.asarray(rows["monitor"], dtype=np.int64)
    reserved = np.asarray(rows["reserved"], dtype=np.int64)
    if (fit.size, monitor.size, reserved.size) != (8000, 1000, 1000):
        raise SystemExit("unexpected SSRC split sizes")
    declared = split.get("sha256", {})
    for name, arr in (("fit", fit), ("monitor", monitor), ("reserved", reserved)):
        if name in declared and declared[name] != row_text_sha256(arr):
            raise SystemExit(f"split row hash mismatch for {name}")
    f, m, r = set(fit.tolist()), set(monitor.tolist()), set(reserved.tolist())
    if f & m or f & r or m & r:
        raise SystemExit("SSRC split sides overlap")
    return Split(fit=fit, monitor=monitor, reserved=reserved, sha256=sha256_file(SPLIT_PATH))


def load_vault() -> dict[int, float]:
    vault = read_json(VAULT_PATH)
    scope = vault.get("scope", {})
    if scope.get("reserved_labels_present") or scope.get("official_valid_test_labels_present"):
        raise SystemExit("label vault is not Fit/Monitor-only")
    return {int(k): float(v) for k, v in vault["labels"].items()}


def row_labels(rows: Sequence[int], labels: Mapping[int, float]) -> np.ndarray:
    missing = [int(r) for r in rows if int(r) not in labels]
    if missing:
        raise SystemExit(f"labels missing for rows {missing[:5]}")
    return np.asarray([labels[int(r)] for r in rows], dtype=np.float64)


def load_objects(path: Path, rows: Sequence[int], *, expected_sha: str | None = None, allow_rebuild: bool = True):
    """Load the cached MolObjects for ``rows`` (a prefix/subset is allowed)."""
    import torch

    want = [int(r) for r in rows]
    if path.exists():
        if expected_sha is not None:
            got = sha256_file(path)
            if got != expected_sha:
                raise SystemExit(f"object cache hash mismatch for {path.name}: {got[:16]} != {expected_sha[:16]}")
        objs = torch.load(path, map_location="cpu", weights_only=False)
        have = [int(o.mol_index) for o in objs]
        if have == want:
            sel = objs
        else:
            pos = {r: i for i, r in enumerate(have)}
            if not set(want) <= set(have):
                raise SystemExit(f"requested rows are not in the object cache {path.name}")
            sel = [objs[pos[r]] for r in want]
        if [int(o.mol_index) for o in sel] != want:
            raise SystemExit(f"object cache row order mismatch in {path.name}")
        return sel, {
            "source": "cache" if sel is objs else "cache_subset",
            "path": str(path.relative_to(REPO)),
            "sha256": sha256_file(path),
            "n_requested": len(want),
        }
    if not allow_rebuild:
        raise SystemExit(f"object cache missing: {path}")
    import cscl_features as cf
    import bjoc_perfmech_long_v1 as bp

    ds = cf.load_official_train()
    objs = bp.build_objects_for_rows(want, ds)
    if [int(o.mol_index) for o in objs] != want:
        raise SystemExit("rebuilt object order mismatch")
    return objs, {"source": "rebuilt", "rows_sha256": rows_sha256(want)}


def load_ssl_state() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import torch

    import bjoc_masked_pretraining_v1 as mp

    got = sha256_file(SSL_STATE)
    if got != P3E_SSL_FILE_SHA:
        raise SystemExit(f"SSL state file hash mismatch: {got} != {P3E_SSL_FILE_SHA}")
    state = torch.load(SSL_STATE, map_location="cpu", weights_only=False)
    return state, {"file_sha256": got, "keys": sorted(state.keys())}


def verify_encoder_state(model, state) -> dict[str, Any]:
    import bjoc_perfmech_long_v1 as bp
    import bjoc_masked_pretraining_v1 as mp

    mp.load_encoder_state(model, state, "cpu")
    got = bp.encoder_group_hash(model)
    if got != P3E_ENCODER_SHA:
        raise SystemExit(f"encoder hash mismatch after SSL load: {got} != {P3E_ENCODER_SHA}")
    return {
        "encoder_group_sha256": got,
        "ssl_state_sha256": P3E_SSL_FILE_SHA,
        "matches_frozen_p3e_encoder": True,
    }


# ---------------------------------------------------------------------------
# frozen P3E reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class P3EReference:
    pred: np.ndarray
    sha256: str


def load_p3e_predictions() -> P3EReference:
    with np.load(P3E_PRED) as z:
        pred = np.asarray(z["pred"], dtype=np.float64)
    freeze = read_json(P3E_FREEZE)
    if sha256_file(P3E_PRED) != freeze["prediction_sha256"]:
        raise SystemExit("P3E prediction file changed since its freeze")
    if sha256_array(pred) != freeze["prediction_array_sha256"]:
        raise SystemExit("P3E prediction array hash mismatch")
    return P3EReference(pred=pred, sha256=freeze["prediction_sha256"])


# ---------------------------------------------------------------------------
# label mechanism groups (descriptive only)
# ---------------------------------------------------------------------------


def label_groups(monitor_rows: Sequence[int]) -> dict[str, Any]:
    """Reconstruct the label-mechanism groups from the frozen Phase 1e audit.

    Returns per-row penalty levels plus the fixed group masks.  Nothing here is a
    model input; the masks are only used to *describe* errors.
    """
    audit = read_json(LABEL_AUDIT)
    scale = audit["formula"]["fitted_cycle_scale"]
    sigma = float(scale["sigma_cycle"])
    mu = float(scale["mu_cycle"])
    records = audit["monitor_penalised_molecules"]
    row_pos = {int(r): i for i, r in enumerate(monitor_rows)}
    n = len(monitor_rows)
    excess = np.zeros(n, dtype=np.int64)
    rdkit_large = np.zeros(n, dtype=bool)
    perm_sensitive_rows = set(int(r) for r in audit["permutation_stability"]["unstable_penalty_monitor_rows"])
    perm_sensitive = np.zeros(n, dtype=bool)
    for rec in records:
        r = int(rec["row"])
        if r not in row_pos:
            raise SystemExit(f"penalised audit row {r} is not a Monitor row")
        i = row_pos[r]
        excess[i] = int(rec["label_excess"])
        rdkit_large[i] = bool(rec["rdkit_large_ring"])
        perm_sensitive[i] = r in perm_sensitive_rows
    cycle_component = (-excess.astype(np.float64) - mu) / sigma
    penalised = excess > 0
    expected = {"label_penalised": 43, "rdkit_large": 40, "penalty_only_rdkit_missed": 3, "neither": 957}
    counts = {
        "label_penalised": int(penalised.sum()),
        "rdkit_large": int(rdkit_large.sum()),
        "penalty_only_rdkit_missed": int((penalised & ~rdkit_large).sum()),
        "neither": int((~penalised).sum()),
        "permutation_sensitive": int(perm_sensitive.sum()),
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise SystemExit(f"group size mismatch for {key}: {counts[key]} != {value}")
    if counts["permutation_sensitive"] != 4:
        raise SystemExit(f"permutation sensitive group changed: {counts['permutation_sensitive']}")
    # consistency: recorded y_without_cycle must be reproducible from the scale
    for rec in records:
        i = row_pos[int(rec["row"])]
        if abs(((-float(rec["label_excess"]) - mu) / sigma) - float(rec["cycle_component"])) > 1e-6:
            raise SystemExit("reconstructed cycle component disagrees with the frozen audit")
    return {
        "excess": excess,
        "cycle_component": cycle_component,
        "masks": {
            "all": np.ones(n, dtype=bool),
            "label_complement": ~penalised,
            "label_penalised": penalised,
            "rdkit_large": rdkit_large,
            "penalty_only_rdkit_missed": penalised & ~rdkit_large,
            "permutation_sensitive": perm_sensitive,
        },
        "counts": counts,
        "sigma_cycle": sigma,
        "mu_cycle": mu,
        "caveat": (
            "post-hoc descriptive groups on a Monitor reused by Phases 1b-1e; "
            "never used for configuration selection or training"
        ),
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def paired_bootstrap_mean(values: np.ndarray, *, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, vals.size, size=(int(n_boot), vals.size))
    means = vals[idx].mean(axis=1)
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return {"mean": float(vals.mean()), "ci95": [lo, hi], "n": int(vals.size)}


def topk_stats(ae: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    """Tail concentration of a signed per-molecule difference (positive = worse)."""
    order = np.argsort(-np.abs(delta), kind="stable")
    out: dict[str, Any] = {}
    for frac in (0.01, 0.05):
        k = max(1, int(np.ceil(frac * delta.size)))
        top = order[:k]
        out[f"top{int(frac * 100)}pct_share"] = (
            float(delta[top].sum() / delta.sum()) if delta.sum() != 0 else float("nan")
        )
        rest = np.delete(delta, top)
        out[f"mean_drop_top{int(frac * 100)}pct"] = float(rest.mean())
    out["abs_error_share_top1pct"] = float(np.sort(ae)[::-1][: max(1, delta.size // 100)].sum() / ae.sum())
    return out


def arm_metrics(
    pred: np.ndarray,
    y: np.ndarray,
    *,
    y_nocycle: np.ndarray,
    cycle_component: np.ndarray,
    groups: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    resid = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    ae = np.abs(resid)
    out: dict[str, Any] = {
        "n": int(ae.size),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((ae**2).mean())),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "p99_ae": float(np.percentile(ae, 99)),
        "max_ae": float(ae.max()),
        "mean_signed_residual": float(resid.mean()),
        "groups": {},
        "tail": topk_stats(ae, resid),
    }
    for name, mask in groups.items():
        m = np.asarray(mask, dtype=bool)
        g: dict[str, Any] = {
            "n": int(m.sum()),
            "mae": float(ae[m].mean()),
            "rmse": float(np.sqrt((ae[m] ** 2).mean())),
            "median_ae": float(np.median(ae[m])),
            "p90_ae": float(np.percentile(ae[m], 90)),
            "mean_signed_residual": float(resid[m].mean()),
            "abs_error_sum": float(ae[m].sum()),
            "share_of_total_abs_error": float(ae[m].sum() / ae.sum()),
            "mae_vs_cycle_free_target": float(np.abs(pred[m] - y_nocycle[m]).mean()),
        }
        if m.any() and abs(float(cycle_component[m].mean())) > 1e-9:
            g["cycle_offset_capture"] = float(
                (pred[m] - y_nocycle[m]).mean() / cycle_component[m].mean()
            )
        else:
            g["cycle_offset_capture"] = None
        out["groups"][name] = g
    return out


def contrast(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y: np.ndarray,
    groups: Mapping[str, np.ndarray],
    *,
    definition: str,
) -> dict[str, Any]:
    """mean |e_a| - |e_b|; positive means ``a`` is worse than ``b``."""
    delta = np.abs(np.asarray(pred_a) - y) - np.abs(np.asarray(pred_b) - y)
    row: dict[str, Any] = {"definition": definition, "groups": {}}
    for name, mask in groups.items():
        m = np.asarray(mask, dtype=bool)
        vals = delta[m]
        st = paired_bootstrap_mean(vals)
        st["win_rate_a_better"] = float(np.mean(vals < 0)) if vals.size else None
        row["groups"][name] = st
    row["tail"] = topk_stats(np.abs(np.asarray(pred_a) - y), delta)
    return row


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(
    *,
    mae_f: float,
    mae_m: float,
    rmse_m: float,
    complement_gain: float,
    flops_ratio_f: float,
    flops_ratio_m: float,
    optimization_ok: bool,
) -> dict[str, Any]:
    """Mechanical application of the frozen Phase 2a rules (protocol §7)."""
    gain_mp = mae_f - mae_m  # positive: M better than F
    gain_f = P3E_MAE - mae_f  # positive: F better than P3E
    gain_p3e = P3E_MAE - mae_m  # positive: M better than P3E
    rmse_delta = rmse_m - P3E_RMSE
    reasons: list[str] = []

    if not optimization_ok:
        status = "OPTIMIZATION_UNRESOLVED"
        reasons.append("optimization pathology detected; see arm records")
    elif (
        gain_mp >= MP_SEPARATION_BAND
        and gain_p3e >= PRACTICAL_BAND
        and rmse_delta <= RMSE_TOLERANCE
        and complement_gain >= COMPLEMENT_BAND
    ):
        status = "STRONG_BACKBONE_MP_INCREMENT"
        reasons.append("M beats F, beats P3E by >= 0.004, RMSE stable, gain survives on the 957 complement")
    elif gain_mp < MP_SEPARATION_BAND and gain_f >= PRACTICAL_BAND and gain_p3e >= PRACTICAL_BAND:
        status = "CAPACITY_ONLY_INCREMENT"
        reasons.append("F and M both beat P3E but are not separated: gain is capacity/training structure")
    elif gain_p3e >= PRACTICAL_BAND and gain_p3e < 0.010 and max(flops_ratio_f, flops_ratio_m) >= 1.5:
        status = "NONCOMPETITIVE_COMPUTE_TRADEOFF"
        reasons.append("small gain at >= 1.5x forward FLOPs")
    elif gain_p3e < PRACTICAL_BAND and gain_f < PRACTICAL_BAND:
        status = "NO_PRACTICAL_GAIN_OVER_P3E"
        reasons.append("neither arm reaches the 0.004 practical band against P3E")
    elif gain_mp < 0:
        status = "NO_PRACTICAL_GAIN_OVER_P3E"
        reasons.append("F is better than M; the local-propagation prior is lowered")
    else:
        status = "INSUFFICIENT_EVIDENCE"
        reasons.append("no rule fired; inspect the recorded contrasts")

    buy_seed1 = bool(
        gain_p3e >= PRACTICAL_BAND
        and gain_mp >= MP_SEPARATION_BAND
        and rmse_delta <= RMSE_TOLERANCE
        and complement_gain >= COMPLEMENT_BAND
        and optimization_ok
        and max(flops_ratio_f, flops_ratio_m) <= 1.5
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "reasons": reasons,
        "gain_mp_mae_f_minus_mae_m": float(gain_mp),
        "gain_p3e_mae_p3e_minus_mae_m": float(gain_p3e),
        "gain_f_mae_p3e_minus_mae_f": float(gain_f),
        "rmse_m_minus_p3e": float(rmse_delta),
        "complement_gain_m_vs_p3e": float(complement_gain),
        "flops_ratio_f": float(flops_ratio_f),
        "flops_ratio_m": float(flops_ratio_m),
        "optimization_ok": bool(optimization_ok),
        "thresholds": {
            "practical_band": PRACTICAL_BAND,
            "mp_separation_band": MP_SEPARATION_BAND,
            "rmse_tolerance": RMSE_TOLERANCE,
            "complement_band": COMPLEMENT_BAND,
        },
        "buy_seed1": buy_seed1,
        "single_seed_bootstrap_note": (
            "molecule bootstrap is discovery-only; it is not training-seed uncertainty"
        ),
    }
