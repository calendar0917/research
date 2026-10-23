"""Shared constants, provenance, metrics and fixed valid groups for

    tracks/structural-sufficiency/zinc_native_relation_v1

Protocol: ``ssrc-zinc-native-relation-v1``.

Discipline
----------
* Inputs are the frozen, label-blind P3 atom/bond/junction objects plus
  structure-only relations between the tokens that already exist in P3.
* ``y`` is read only by the supervised trainer and the evaluator.
* The official ZINC test split is not touched before the terminal stage and
  only if the frozen valid trigger fires.
* No preprocessing statistic is fit on valid/test: the global-33 standardiser
  is the original P3 train-only one, and the relation construction has no
  fitted parameters.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
REPO = TRACK.parents[2]
V1_CODE = REPO / "tracks/structural-sufficiency/zinc_arch_discovery_v1/code"
SS_CODE = REPO / "tracks/structural-sufficiency/code"
for _p in (HERE, V1_CODE, SS_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ad_common as AC  # noqa: E402

CONFIG_PATH = TRACK / "configs/nr_v1.json"
CACHE_DIR = TRACK / "cache"
ANALYSIS_DIR = TRACK / "analysis"
RUNS_DIR = TRACK / "runs"

PROTOCOL_ID = "ssrc-zinc-native-relation-v1"
PROTOCOL_PATH = TRACK / "PROTOCOL.md"
DESIGN_PATH = TRACK / "DESIGN.md"

N_TRAIN = 10000
N_VALID = 1000
ARMS = ("P", "R", "N")
SEEDS = (0, 1)

#: token role order used by the P3 object layout and by every relation array
ROLE_JUNCTION = 0
ROLE_BOND = 1
ROLE_NAMES = {ROLE_JUNCTION: "J", ROLE_BOND: "B"}
BASE_ROLE_PAIRS = ((0, 0), (0, 1), (1, 1))

#: fixed valid group tables from earlier rounds (not changed in this round)
RESTRICTED_REF = (
    REPO
    / "tracks/structural-sufficiency/zinc_restricted_adaptation_v1"
    / "analysis/reference_preds_valid.csv"
)
AUDIT_PER_MOL = (
    REPO
    / "tracks/structural-sufficiency/zinc_generalization_audit_v1"
    / "analysis/per_molecule.csv"
)
P3B_VALID = (
    REPO
    / "tracks/structural-sufficiency/zinc_recipe_trunk_v2/runs/xBssl_P3N_s0"
    / "predictions/valid_best.npz"
)
P3B_SHA256 = "8dc5ba2005f9a76bbb995f75dba6760ab471fc0eb57155797bad6d3087aca33a"
P3B_VALID_MAE = 0.07622375367576023

#: fixed molecule bootstrap seed (new for this round; no optional stopping)
BOOTSTRAP_SEED = 20261108
BOOTSTRAP_N = 10000


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


def sha256_bytes(chunks: Sequence[bytes]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def sha256_state_dict(state: Mapping[str, Any]) -> str:
    import torch

    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        arr = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(arr.shape)).encode())
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path | str, payload: Mapping[str, Any] | Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def write_once_json(path: Path | str, payload: Mapping[str, Any] | Any) -> None:
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen evidence {path}")
    write_json(path, payload)


def load_config() -> dict[str, Any]:
    return read_json(CONFIG_PATH)


def git_commit() -> str:
    return AC.git_commit()


def environment() -> dict[str, Any]:
    return AC.environment()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {y.shape}")
    return float(np.abs(pred - y).mean())


def metric_block(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ae = np.abs(pred - y)
    return {
        "n": int(len(ae)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.square(pred - y).mean())),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "p95_ae": float(np.quantile(ae, 0.95)),
        "p99_ae": float(np.quantile(ae, 0.99)),
        "max_ae": float(ae.max()),
        "bias_pred_minus_y": float((pred - y).mean()),
        "sd_pred": float(pred.std()),
    }


def paired_bootstrap(
    ae_first: np.ndarray,
    ae_second: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired molecule bootstrap of ``mean(ae_first - ae_second)``.

    Negative means the first model is better.  Molecule sampling only: it does
    not estimate training-seed variance, and k2 co-occurrence makes valid rows
    positively correlated (see report caveat).
    """
    first = np.asarray(ae_first, dtype=np.float64)
    second = np.asarray(ae_second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("paired arrays must have the same shape")
    rng = np.random.default_rng(seed)
    n = len(first)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = (first[idx] - second[idx]).mean()
    return {
        "n": int(n),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "mean_delta": float((first - second).mean()),
        "median_delta": float(np.median(first - second)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "prob_first_better": float((deltas < 0).mean()),
        "better_fraction": float((first < second).mean()),
    }


def window_metrics(curve: Sequence[Mapping[str, Any]], *, window: int = 20, tail: int = 100) -> dict[str, Any]:
    """W20 = minimum median over every contiguous 20-epoch window.

    T100 = median of the last 100 epochs.  Neither replaces the best-valid
    checkpoint; they are pre-declared persistence diagnostics.
    """
    v = np.asarray([float(e["valid_mae"]) for e in curve], dtype=np.float64)
    epochs = np.asarray([int(e["epoch"]) for e in curve], dtype=np.int64)
    n = len(v)
    if n == 0:
        raise ValueError("empty curve")
    w20 = float("inf")
    w20_start = int(epochs[0]) if n else -1
    if n >= window:
        meds = np.asarray([np.median(v[s : s + window]) for s in range(n - window + 1)])
        k = int(np.argmin(meds))
        w20 = float(meds[k])
        w20_start = int(epochs[k])
    t100 = float(np.median(v[-min(tail, n) :]))
    return {
        "W20": w20 if np.isfinite(w20) else float(np.median(v)),
        "W20_window_start_epoch": w20_start,
        "W20_window_epochs": int(window),
        "T100": t100,
        "T100_window_epochs": int(min(tail, n)),
        "valid_min": float(v.min()),
        "valid_min_epoch": int(epochs[int(np.argmin(v))]),
        "valid_last": float(v[-1]),
        "n_epochs": int(n),
        "epochs": epochs.tolist(),
        "valid_curve": v.tolist(),
    }


# ---------------------------------------------------------------------------
# fixed valid groups
# ---------------------------------------------------------------------------


def load_valid_groups() -> dict[str, Any]:
    """Old four groups (710/225/39/26), tail/ordinary, ring-rich, y and P3B.

    These masks were fixed by earlier rounds and are copied here by value; they
    are not re-derived from any model in this round.
    """
    import csv

    if not RESTRICTED_REF.exists():
        raise SystemExit(f"missing fixed valid table: {RESTRICTED_REF}")
    with open(RESTRICTED_REF, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != N_VALID:
        raise SystemExit(f"fixed valid table has {len(rows)} rows, expected {N_VALID}")
    rows.sort(key=lambda r: int(r["row"]))
    if not np.array_equal(np.asarray([int(r["row"]) for r in rows]), np.arange(N_VALID)):
        raise SystemExit("fixed valid table rows are not 0..999")

    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    unseen_225 = np.asarray([int(r["signature_unseen"]) for r in rows], dtype=bool)
    ordinary = np.asarray([int(r["ordinary_y_gt_minus3"]) for r in rows], dtype=bool)
    tail = np.asarray([int(r["tail_y_le_minus3"]) for r in rows], dtype=bool)
    p3b = np.asarray([float(r["pred_P3B_s0"]) for r in rows], dtype=np.float64)
    if not np.array_equal(tail, ~ordinary):
        raise SystemExit("tail and ordinary masks are not complements")

    with open(AUDIT_PER_MOL, newline="", encoding="utf-8") as fh:
        audit = list(csv.DictReader(fh))
    audit.sort(key=lambda r: int(r["row_id"]))
    if len(audit) != N_VALID:
        raise SystemExit("audit per-molecule table is not 1000 rows")
    n_rings = np.asarray([int(r["n_rings"]) for r in audit], dtype=np.int64)
    audit_unseen = np.asarray([float(r["r2_unseen_atom_frac"]) > 0.0 for r in audit], dtype=bool)
    audit_y = np.asarray([float(r["y_raw"]) for r in audit], dtype=np.float64)
    if not np.array_equal(audit_unseen, unseen_225):
        raise SystemExit("audit unseen mask disagrees with the fixed four-group table")
    if not np.allclose(audit_y, y, rtol=0.0, atol=0.0):
        raise SystemExit("audit y disagrees with the fixed four-group table")

    p3b_ae = np.abs(p3b - y)
    groups = {
        "seen_ordinary": ~unseen_225 & ordinary,
        "unseen_ordinary": unseen_225 & ordinary,
        "seen_tail": ~unseen_225 & tail,
        "unseen_tail": unseen_225 & tail,
    }
    expected = {"seen_ordinary": 710, "unseen_ordinary": 225, "seen_tail": 39, "unseen_tail": 26}
    got = {k: int(v.sum()) for k, v in groups.items()}
    if got != expected:
        raise SystemExit(f"old four-group sizes {got} != {expected}")
    groups.update(
        {
            "all": np.ones(N_VALID, dtype=bool),
            "ordinary_935": ordinary,
            "tail_65": tail,
            "unseen_251": unseen_225,
            "seen_749": ~unseen_225,
            "ring_rich_213": n_rings >= 4,
            "ring_rich_unseen_47": (n_rings >= 4) & unseen_225,
            "drop_row172": np.arange(N_VALID) != 172,
            "tail_drop_row172": tail & (np.arange(N_VALID) != 172),
        }
    )
    return {
        "y": y,
        "p3b_pred": p3b,
        "p3b_ae": p3b_ae,
        "p3b_valid_mae": float(p3b_ae.mean()),
        "groups": groups,
        "n_rings": n_rings,
        "radius2_unseen": unseen_225,
        "ordinary": ordinary,
        "tail": tail,
        "top10_p3b": np.argsort(-p3b_ae, kind="stable")[:10],
        "source": str(RESTRICTED_REF.relative_to(REPO)),
        "source_audit": str(AUDIT_PER_MOL.relative_to(REPO)),
    }


def group_table(ae: np.ndarray, groups: Mapping[str, np.ndarray], *, total_ae: float | None = None) -> dict[str, Any]:
    ae = np.asarray(ae, dtype=np.float64)
    tot = float(ae.sum()) if total_ae is None else float(total_ae)
    out: dict[str, Any] = {}
    for name, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        out[name] = {
            "n": int(mask.sum()),
            "mae": float(ae[mask].mean()),
            "sum_ae": float(ae[mask].sum()),
            "share_of_total_ae": float(ae[mask].sum() / max(tot, 1e-12)),
        }
    return out


def delta_stats(
    ae_candidate: np.ndarray,
    ae_reference: np.ndarray,
    groups: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Candidate minus reference diagnostics; negative = candidate better."""
    d = np.asarray(ae_candidate, dtype=np.float64) - np.asarray(ae_reference, dtype=np.float64)
    out = {
        "mean_delta_ae": float(d.mean()),
        "median_delta_ae": float(np.median(d)),
        "better_fraction": float((d < 0).mean()),
        "equal_fraction": float((d == 0).mean()),
        "total_delta_ae_contribution": float(d.mean()),
        "groups": {},
    }
    for name, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        out["groups"][name] = {
            "n": int(mask.sum()),
            "mean_delta_ae": float(d[mask].mean()),
            "median_delta_ae": float(np.median(d[mask])),
            "better_fraction": float((d[mask] < 0).mean()),
            "delta_sum": float(d[mask].sum()),
            "contribution_to_overall_mae": float(d[mask].sum() / len(d)),
        }
    return out


def tail_calibration(pred: np.ndarray, y: np.ndarray, tail: np.ndarray, *, drop_row172: bool = False) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(tail, dtype=bool).copy()
    if drop_row172:
        mask[172] = False
    yy, pp = y[mask], pred[mask]
    slope = float(np.polyfit(yy, pp, 1)[0]) if len(yy) >= 2 else float("nan")
    return {
        "n": int(mask.sum()),
        "mae": float(np.abs(pp - yy).mean()),
        "bias_pred_minus_y": float((pp - yy).mean()),
        "slope_pred_on_y": slope,
        "mean_y": float(yy.mean()),
        "mean_pred": float(pp.mean()),
    }


def topk_removed_gain(
    ae_candidate: np.ndarray,
    ae_reference: np.ndarray,
    *,
    k: int = 10,
    fixed_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    ae_c = np.asarray(ae_candidate, dtype=np.float64)
    ae_r = np.asarray(ae_reference, dtype=np.float64)
    if fixed_rows is None:
        fixed_rows = np.argsort(-ae_r, kind="stable")[:k].tolist()
    keep = np.ones(len(ae_c), dtype=bool)
    keep[np.asarray(list(fixed_rows), dtype=int)] = False
    return {
        "k": int(k),
        "fixed_rows": [int(i) for i in fixed_rows],
        "gain_all": float(ae_r.mean() - ae_c.mean()),
        "gain_drop_topk": float(ae_r[keep].mean() - ae_c[keep].mean()),
        "mae_all": float(ae_c.mean()),
        "mae_drop_topk": float(ae_c[keep].mean()),
    }
