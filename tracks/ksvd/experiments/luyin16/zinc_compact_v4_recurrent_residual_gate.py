"""Minimal recurrent residual-magnitude gate for the canonical compact-v4 T=2 Q16.

Question
--------
The canonical T=2 weight-tied recurrent pair--centre model (82,115 params) is
functional, but its single-model training trajectory is known to be sensitive
to initialization / data-order / execution noise (Top-5 checkpoint soup helps
on 3/3 seeds).  Can the *simplest possible* limit on the recurrent update
magnitude stabilise the dynamics and improve generalisation **without adding
capacity**?

Change (exactly one parameter)
------------------------------
The centre update becomes a gated residual

    alpha = sigmoid(a)
    h^(t+1) = h^(t) + alpha * U(h^(t), A^(t))

with a single model-wide learnable scalar ``a`` (initialised to ``0`` so that
``alpha == 0.5``), shared by **both** T=2 rounds.  Everything else -- tokenizer,
patch encoder, relation descriptor, distance buckets, pair encoder, centre
pooling / aggregation, readout, global/topology descriptors, small head,
optimizer and training protocol -- is inherited unchanged.

Deliberately *not* here: channel-wise gate, context-dependent gate, LayerNorm,
attention, dropout, new relation feature, new width/depth.  No official test.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_residual_gate <stage>

Stages: ``reference sanity train soup alpha stability decision report all``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_recurrent_residual_gate"
RUNS_DIR = RESULTS_DIR / "runs"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
SOUP_DIR = RESULTS_DIR / "soup_states"

PROTOCOL_VERSION = "compact_v4_recurrent_residual_gate_v1"
GATED_TAG = "gated"

EXPECTED_UNGATED_TOTAL = 82115
EXPECTED_GATED_TOTAL = 82116
EXPECTED_HEAD = 4135

SOUP_K = vd.SOUP_K  # 5, locked rule reused verbatim
PHASE1_SEEDS = (0, 1)
ALL_SEEDS = (0, 1, 2)

# --- pre-registered decision gates (2-seed Top-5 soup mean) ----------------
STRONG_POSITIVE_GATE = 0.002
WEAK_GATE = 0.001
OPPOSITE_DIRECTION_GATE = 0.001

# --- canonical ungated reference (compact_v4_recurrent_variance_diagnosis) --
UNGATED_RAW_REF: dict[int, float] = {
    0: 0.1406094916117727,
    1: 0.13343997858563672,
    2: 0.1427223045033752,
}
UNGATED_SOUP_REF: dict[int, float] = {
    0: 0.13707755148684372,
    1: 0.13220956423232566,
    2: 0.13806160257657757,
}
UNGATED_2SEED_SOUP_MEAN = float(
    np.mean([UNGATED_SOUP_REF[0], UNGATED_SOUP_REF[1]])
)
UNGATED_2SEED_RAW_MEAN = float(np.mean([UNGATED_RAW_REF[0], UNGATED_RAW_REF[1]]))

# canonical selection states / soups produced by the variance-diagnosis run.
_CANONICAL_STATE = {0: "canonical_seed0_selection_state.pt",
                    1: "canonical_seed1_selection_state.pt",
                    2: "canonical_seed2_selection_state.pt"}


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()
    except OSError:
        return "unknown"


def _run_path(tag: str, seed: int) -> Path:
    return RUNS_DIR / f"{tag}_seed{seed}.json"


def _load_run(tag: str, seed: int) -> dict[str, Any]:
    return _read_json(_run_path(tag, seed))


def _soup_path(tag: str, seed: int) -> Path:
    return SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"


def _load_soup_payload(tag: str, seed: int) -> dict[str, Any]:
    return _read_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json")


def _read_curve(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# evaluation plumbing (reused from the variance-diagnosis module)
# ---------------------------------------------------------------------------


def _valid_data() -> list[Any]:
    _train, valid, _audit = rec.load_encoded()
    return list(valid)


def _eval_loader(valid_data: Sequence[Any]):
    return zpp._make_loader(
        list(valid_data), int(shead.OPTIMIZED_PROTOCOL["batch_size"]), False, 0
    )


def _alpha_of_state(state: Mapping[str, torch.Tensor]) -> float:
    logit = state["update_gate_logit"]
    return float(torch.sigmoid(torch.as_tensor(logit, dtype=torch.float32)))


def _load_selection_state(tag: str, seed: int) -> dict[str, torch.Tensor]:
    path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------


def reference() -> dict[str, Any]:
    per_seed = {}
    for seed in ALL_SEEDS:
        run = _read_json(vd._run_path("canonical", seed))
        soup = _read_json(vd.RESULTS_DIR / f"soup_canonical_seed{seed}.json")
        per_seed[int(seed)] = {
            "raw_valid_mae": float(run["best_valid_mae"]),
            "best_epoch": int(run["best_epoch"]),
            "train_at_best": float(run["train_loss_at_best"]),
            "soup_valid_mae": float(soup["top5_soup_valid_mae"]),
            "soup_improvement": float(soup["soup_improvement_over_best"]),
            "top5_epochs": [int(e) for e in soup["top5_epochs"]],
        }
    raw_vals = np.array([per_seed[s]["raw_valid_mae"] for s in ALL_SEEDS])
    soup_vals = np.array([per_seed[s]["soup_valid_mae"] for s in ALL_SEEDS])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "description": "canonical ungated T=2 recurrent Q16 reference",
        "architecture": {
            "recurrence_rounds": rec.RECURRENCE_ROUNDS,
            "q_dim": rec.Q_DIM,
            "R_dim": rec.R_DIM,
            "params": EXPECTED_UNGATED_TOTAL,
        },
        "per_seed": per_seed,
        "raw_2seed_mean": UNGATED_2SEED_RAW_MEAN,
        "soup_2seed_mean": UNGATED_2SEED_SOUP_MEAN,
        "raw_3seed_mean": float(raw_vals.mean()),
        "raw_3seed_std": float(raw_vals.std(ddof=1)),
        "soup_3seed_mean": float(soup_vals.mean()),
        "soup_3seed_std": float(soup_vals.std(ddof=1)),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "reference.json", payload)
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def run_training(
    tag: str,
    seed: int,
    *,
    build_fn: Callable[[int], nn.Module] | None = None,
    snapshots: bool = True,
    protocol_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one gated seed under the inherited canonical protocol, sequential."""
    train_data, valid_data, _ = rec.load_encoded()
    build_fn = rec.build_gated if build_fn is None else build_fn
    override = dict(protocol_override or {})
    snap_dir = SNAPSHOT_DIR / f"{tag}_seed{seed}" if snapshots else None
    started = time.perf_counter()
    summary = rec.train(
        build_fn,
        train_data,
        valid_data,
        seed=int(seed),
        tag=str(tag),
        protocol_override=override or None,
        expected_total=EXPECTED_GATED_TOTAL,
        snapshot_dir=snap_dir,
        data_seed=None,
    )
    summary["residual_gate_wall_clock_s"] = float(time.perf_counter() - started)
    # copy the selection state / curve next to this experiment's own outputs.
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(str(summary["state_path"]))
    dst = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    dst.write_bytes(src.read_bytes())
    summary["residual_gate_state_path"] = str(dst)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    curve_src = Path(str(summary["curve_path"]))
    curve_dst = CURVE_DIR / f"{tag}_seed{seed}_curve.csv"
    curve_dst.write_bytes(curve_src.read_bytes())
    summary["residual_gate_curve_path"] = str(curve_dst)
    state = torch.load(dst, map_location="cpu", weights_only=True)
    summary["best_alpha"] = _alpha_of_state(state)
    summary["best_gate_logit"] = float(state["update_gate_logit"])
    _write_json(_run_path(tag, seed), summary)
    return summary


def _top5_epochs(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return vd._top5_epochs(summary)


def build_soup_state(epochs: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    return vd.build_soup_state(epochs)


def soup(tag: str, seed: int) -> dict[str, Any]:
    """Locked fixed Top-5 soup: lowest selection MAE, ties -> earliest epoch."""
    summary = _load_run(tag, seed)
    top = _top5_epochs(summary)
    soup_state = build_soup_state(top)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    path = _soup_path(tag, seed)
    torch.save(soup_state, path)

    valid_data = _valid_data()
    loader = _eval_loader(valid_data)
    model = rec.build_gated(seed)
    best_state = _load_selection_state(tag, seed)
    targets, best_preds = vd._predict_state(model, best_state, loader)
    _targets, soup_preds = vd._predict_state(model, soup_state, loader)
    best_mae = vd._mae(targets, best_preds)
    soup_mae = vd._mae(targets, soup_preds)
    top_alphas = []
    for row in top:
        snap = torch.load(Path(str(row["path"])), map_location="cpu", weights_only=True)
        top_alphas.append(_alpha_of_state(snap))
    payload = {
        "tag": str(tag),
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "soup_rule": {
            "K": SOUP_K,
            "ranking": "lowest official-valid MAE",
            "tie_rule": "earliest epoch",
            "weight_aggregation": "arithmetic mean",
            "no_k_search": True,
            "no_weight_search": True,
        },
        "top5_epochs": [int(row["epoch"]) for row in top],
        "top5_valid_mae": [float(row["valid_mae"]) for row in top],
        "top5_alphas": top_alphas,
        "soup_alpha": _alpha_of_state(soup_state),
        "best_epoch": int(summary["best_epoch"]),
        "best_checkpoint_valid_mae": best_mae,
        "top5_soup_valid_mae": soup_mae,
        "soup_improvement_over_best": best_mae - soup_mae,
        "soup_state_path": str(path),
        "soup_sha256": _sha256_file(path),
        "parameters": int(sum(p.numel() for p in soup_state.values())),
        "official_test_loaded": False,
        "best_predictions": best_preds.tolist(),
        "soup_predictions": soup_preds.tolist(),
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# learned-alpha reporting
# ---------------------------------------------------------------------------


def alpha_report() -> dict[str, Any]:
    per_seed: dict[int, Any] = {}
    for seed in ALL_SEEDS:
        run_path = _run_path(GATED_TAG, seed)
        if not run_path.exists():
            continue
        run = _read_json(run_path)
        alphas: list[dict[str, float]] = []
        for row in run.get("snapshot_manifest") or []:
            snap = torch.load(
                Path(str(row["path"])), map_location="cpu", weights_only=True
            )
            alphas.append(
                {
                    "epoch": int(row["epoch"]),
                    "valid_mae": float(row["valid_mae"]),
                    "alpha": _alpha_of_state(snap),
                }
            )
        final_alpha = _alpha_of_state(_load_selection_state(GATED_TAG, seed))
        soup_info = None
        soup_json = RESULTS_DIR / f"soup_{GATED_TAG}_seed{seed}.json"
        if soup_json.exists():
            soup_info = _read_json(soup_json)
        per_seed[int(seed)] = {
            "best_alpha": float(run["best_alpha"]),
            "best_gate_logit": float(run["best_gate_logit"]),
            "selection_state_alpha": float(final_alpha),
            "final_epoch_alpha": alphas[-1]["alpha"] if alphas else None,
            "alpha_min": float(min(a["alpha"] for a in alphas)) if alphas else None,
            "alpha_max": float(max(a["alpha"] for a in alphas)) if alphas else None,
            "top5_alphas": None if soup_info is None else soup_info["top5_alphas"],
            "top5_alpha_min": (
                None if soup_info is None else float(min(soup_info["top5_alphas"]))
            ),
            "top5_alpha_max": (
                None if soup_info is None else float(max(soup_info["top5_alphas"]))
            ),
            "soup_alpha": None if soup_info is None else float(soup_info["soup_alpha"]),
            "trajectory": alphas,
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "gated_tag": GATED_TAG,
        "per_seed": per_seed,
        "best_alpha_range": [
            float(min(v["best_alpha"] for v in per_seed.values())),
            float(max(v["best_alpha"] for v in per_seed.values())),
        ]
        if per_seed
        else None,
        "top5_alpha_range": [
            float(
                min(
                    a
                    for v in per_seed.values()
                    for a in (v["top5_alphas"] or [v["best_alpha"]])
                )
            ),
            float(
                max(
                    a
                    for v in per_seed.values()
                    for a in (v["top5_alphas"] or [v["best_alpha"]])
                )
            ),
        ]
        if per_seed
        else None,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "alpha_report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stability diagnostics (function consistency across seeds)
# ---------------------------------------------------------------------------


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return vd._corr(a, b)


def _predictions(run: Mapping[str, Any], soup: Mapping[str, Any] | None, key: str):
    """Return (targets, predictions) from a stored run / soup payload."""
    targets = np.asarray(run["valid_targets"], dtype=np.float64)
    if key == "best":
        preds = np.asarray(run["valid_predictions"], dtype=np.float64)
    elif key == "soup":
        if soup is None:
            raise ValueError("soup payload required")
        preds = np.asarray(soup["soup_predictions"], dtype=np.float64)
    else:
        raise ValueError(key)
    return targets, preds


def _pairwise(a: np.ndarray, b: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return {
        "mean_abs_prediction_disagreement": float(np.mean(np.abs(a - b))),
        "prediction_correlation": _corr(a, b),
        "residual_correlation": _corr(targets - a, targets - b),
    }


def stability() -> dict[str, Any]:
    ungated_canonical = {}
    gated = {}
    targets_ref = None
    for seed in ALL_SEEDS:
        run_path = vd._run_path("canonical", seed)
        soup_json = vd.RESULTS_DIR / f"soup_canonical_seed{seed}.json"
        if not run_path.exists():
            continue
        run = _read_json(run_path)
        soup_payload = _read_json(soup_json) if soup_json.exists() else None
        t, best = _predictions(run, soup_payload, "best")
        targets_ref = t if targets_ref is None else targets_ref
        _t, soup = _predictions(run, soup_payload, "soup")
        ungated_canonical[int(seed)] = {"best": best, "soup": soup}
        gated_path = _run_path(GATED_TAG, seed)
        gated_soup_path = RESULTS_DIR / f"soup_{GATED_TAG}_seed{seed}.json"
        if gated_path.exists():
            grun = _read_json(gated_path)
            gsoup = _read_json(gated_soup_path) if gated_soup_path.exists() else None
            _gt, gbest = _predictions(grun, gsoup, "best")
            _gt, gsoup_pred = _predictions(grun, gsoup, "soup")
            gated[int(seed)] = {"best": gbest, "soup": gsoup_pred}

    def _summarise(states: Mapping[int, Mapping[str, np.ndarray]], kind: str):
        if len(states) < 2:
            return None
        seeds = sorted(states)
        a = states[seeds[0]][kind]
        b = states[seeds[1]][kind]
        metrics = _pairwise(a, b, targets_ref)
        per_seed_mae = {
            str(s): vd._mae(targets_ref, states[s][kind]) for s in seeds
        }
        mae_values = np.array([per_seed_mae[str(s)] for s in seeds])
        return {
            **metrics,
            "per_seed_mae": per_seed_mae,
            "mae_spread": float(mae_values.max() - mae_values.min()),
            "seed_mean_mae": float(mae_values.mean()),
        }

    def _best_epoch_stats(tag: str) -> dict[str, int]:
        epochs = {}
        for seed in (0, 1):
            path = (
                vd._run_path("canonical", seed)
                if tag == "ungated"
                else _run_path(GATED_TAG, seed)
            )
            if path.exists():
                epochs[str(int(seed))] = int(_read_json(path)["best_epoch"])
        return {
            "per_seed": epochs,
            "spread": int(max(epochs.values()) - min(epochs.values()))
            if len(epochs) > 1
            else 0,
        }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "comparison": {
            "ungated_canonical_best": _summarise(ungated_canonical, "best"),
            "gated_best": _summarise(gated, "best"),
            "ungated_canonical_soup": _summarise(ungated_canonical, "soup"),
            "gated_soup": _summarise(gated, "soup"),
        },
        "best_epoch": {
            "ungated": _best_epoch_stats("ungated"),
            "gated": _best_epoch_stats("gated"),
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stability.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def _gated_result(tag: str, seed: int) -> dict[str, Any] | None:
    run_path = _run_path(tag, seed)
    if not run_path.exists():
        return None
    run = _read_json(run_path)
    soup_path = RESULTS_DIR / f"soup_{tag}_seed{seed}.json"
    soup = _read_json(soup_path) if soup_path.exists() else None
    return {
        "seed": int(seed),
        "raw_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "train_at_best": float(run["train_loss_at_best"]),
        "train_valid_gap": float(run["best_valid_mae"] - run["train_loss_at_best"]),
        "epochs_run": int(run["epochs_run"]),
        "horizon_warning": bool(run.get("horizon_boundary_warning", False)),
        "best_alpha": float(run["best_alpha"]),
        "soup_valid_mae": None if soup is None else float(soup["top5_soup_valid_mae"]),
        "soup_improvement": (
            None if soup is None else float(soup["soup_improvement_over_best"])
        ),
        "top5_epochs": None if soup is None else [int(e) for e in soup["top5_epochs"]],
        "top5_alphas": None if soup is None else list(soup["top5_alphas"]),
    }


def decision() -> dict[str, Any]:
    present = [s for s in PHASE1_SEEDS if _run_path(GATED_TAG, s).exists()]
    gated = {s: _gated_result(GATED_TAG, s) for s in present}
    gated = {s: v for s, v in gated.items() if v is not None}
    if len(gated) < 2:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "incomplete",
            "seeds_present": present,
            "official_test_loaded": False,
        }
        _write_json(RESULTS_DIR / "decision.json", payload)
        return payload

    seeds = sorted(gated)
    per_seed_delta = {
        int(s): float(UNGATED_SOUP_REF[s] - gated[s]["soup_valid_mae"]) for s in seeds
    }
    raw_delta = {
        int(s): float(UNGATED_RAW_REF[s] - gated[s]["raw_valid_mae"]) for s in seeds
    }
    gated_soup_mean = float(np.mean([gated[s]["soup_valid_mae"] for s in seeds]))
    gated_raw_mean = float(np.mean([gated[s]["raw_valid_mae"] for s in seeds]))
    ungated_soup_mean = float(np.mean([UNGATED_SOUP_REF[s] for s in seeds]))
    ungated_raw_mean = float(np.mean([UNGATED_RAW_REF[s] for s in seeds]))
    soup_improvement = float(ungated_soup_mean - gated_soup_mean)
    raw_improvement = float(ungated_raw_mean - gated_raw_mean)
    deltas = np.array([per_seed_delta[s] for s in seeds])
    sign_flip = bool(
        deltas.min() < -OPPOSITE_DIRECTION_GATE
        and deltas.max() > OPPOSITE_DIRECTION_GATE
    )
    if sign_flip or soup_improvement <= WEAK_GATE:
        case = "no_signal"
    elif (
        soup_improvement >= STRONG_POSITIVE_GATE
        and deltas.min() > -OPPOSITE_DIRECTION_GATE
    ):
        case = "strong_positive"
    else:
        case = "weak_ambiguous"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_metric": "2-seed Top-5 soup mean valid MAE",
        "seeds": [int(s) for s in seeds],
        "gated": {str(s): gated[s] for s in seeds},
        "reference": {
            "ungated_raw": {str(s): UNGATED_RAW_REF[s] for s in seeds},
            "ungated_soup": {str(s): UNGATED_SOUP_REF[s] for s in seeds},
        },
        "per_seed_soup_improvement": {
            str(s): per_seed_delta[s] for s in seeds
        },
        "per_seed_raw_improvement": {
            str(s): raw_delta[s] for s in seeds
        },
        "gated_2seed_soup_mean": gated_soup_mean,
        "ungated_2seed_soup_mean": ungated_soup_mean,
        "soup_mean_improvement": soup_improvement,
        "gated_2seed_raw_mean": gated_raw_mean,
        "ungated_2seed_raw_mean": ungated_raw_mean,
        "raw_mean_improvement": raw_improvement,
        "gated_soup_spread": float(
            np.max([gated[s]["soup_valid_mae"] for s in seeds])
            - np.min([gated[s]["soup_valid_mae"] for s in seeds])
        ),
        "ungated_soup_spread": float(
            np.max([UNGATED_SOUP_REF[s] for s in seeds])
            - np.min([UNGATED_SOUP_REF[s] for s in seeds])
        ),
        "gate": {
            "strong_positive": STRONG_POSITIVE_GATE,
            "weak": WEAK_GATE,
            "opposite_direction": OPPOSITE_DIRECTION_GATE,
        },
        "sign_flip": sign_flip,
        "case": case,
        "run_seed2_replication": bool(case == "strong_positive"),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


# ---------------------------------------------------------------------------
# sanity / mechanism checks
# ---------------------------------------------------------------------------


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    torch.set_num_threads(int(shead.TORCH_THREADS))
    _train, valid_data, _audit = rec.load_encoded()
    batch = rec._first_batch(valid_data, device)

    baseline = rec.build_baseline(0).to(device).eval()
    ungated = rec.build_recurrent(0).to(device).eval()
    gated = rec.build_gated(0).to(device).eval()

    def n_params(module: nn.Module) -> int:
        return int(sum(p.numel() for p in module.parameters()))

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # 1. exactly one new parameter
    p_base = n_params(baseline)
    p_ungated = n_params(ungated)
    p_gated = n_params(gated)
    base_names = {n for n, _ in baseline.named_parameters()}
    ungated_names = {n for n, _ in ungated.named_parameters()}
    gated_names = {n for n, _ in gated.named_parameters()}
    checks["params_ungated_is_canonical"] = p_ungated == EXPECTED_UNGATED_TOTAL
    checks["params_gated_plus_one"] = p_gated == p_ungated + 1 == EXPECTED_GATED_TOTAL
    checks["ungated_named_params_equal_baseline"] = base_names == ungated_names
    checks["gated_adds_only_update_gate_logit"] = (
        gated_names - ungated_names == {"update_gate_logit"}
        and not (ungated_names - gated_names)
    )
    details["params"] = {
        "baseline": p_base,
        "ungated": p_ungated,
        "gated": p_gated,
        "head": n_params(gated.head),
    }

    # 2. one shared scalar; both rounds gated
    alpha0 = float(gated.update_alpha())
    checks["init_alpha_is_half"] = abs(alpha0 - 0.5) < 1e-12
    checks["single_gate_parameter"] = (
        sum(1 for n, _ in gated.named_parameters() if n == "update_gate_logit") == 1
    )
    source = inspect.getsource(rec.PatchPathRecurrentPairCentreModel._encode_core)
    checks["gate_applied_once_inside_round_loop"] = (
        source.count("self.update_alpha()") == 1
        and "for round_index in range(rounds)" in source
    )
    # numerical proof that both rounds are gated: activate the centre update
    # (it is zero-initialised), then alpha -> 0 must collapse to baseline.  A
    # round-2-only ungated path would leave an O(U) difference, far above tol.
    collapsed = rec.build_gated(0).to(device).eval()
    with torch.no_grad():
        for parameter in collapsed.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        collapsed.update_gate_logit.fill_(-30.0)
        assert float(collapsed.update_alpha()) < 1e-12
        r_collapsed = collapsed.encode(batch)
        r_baseline = baseline.encode(batch)
    collapse_diff = float((r_collapsed - r_baseline).abs().max())
    checks["alpha_to_zero_collapses_to_baseline"] = collapse_diff < 1e-9
    details["alpha_to_zero_max_abs_diff"] = collapse_diff

    # round-1 update scales exactly with alpha.  The centre update is
    # zero-initialised, so perturb it first (otherwise U == 0 and the ratio is
    # undefined).
    probe = rec.build_gated(0).to(device).eval()
    probe.capture_diagnostics = True
    with torch.no_grad():
        for parameter in probe.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        probe.update_gate_logit.fill_(0.0)
        probe.encode(batch)
        d_a0 = (probe.last_h1 - probe.last_h0).clone()
        probe.update_gate_logit.fill_(1.0)
        probe.encode(batch)
        d_a1 = (probe.last_h1 - probe.last_h0).clone()
    ratio_expected = float(
        torch.sigmoid(torch.tensor(0.0)) / torch.sigmoid(torch.tensor(1.0))
    )
    ratio_observed = float(d_a0.abs().mean() / d_a1.abs().mean())
    checks["round1_update_scales_with_alpha"] = (
        abs(ratio_observed - ratio_expected) < 1e-5
    )
    details["round1_alpha_ratio"] = {
        "expected": ratio_expected,
        "observed": ratio_observed,
    }

    # 4. alpha receives a non-zero, finite gradient.  The centre update is
    # zero-initialised (so is its contribution and hence the gate's gradient);
    # activate it first, mirroring the mechanism probes above.
    trainable = rec.build_gated(0).to(device)
    with torch.no_grad():
        for parameter in trainable.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    trainable.train()
    prediction = trainable(batch).view(-1)
    target = batch.y.view(-1)
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    grad = trainable.update_gate_logit.grad
    grad_ok = grad is not None and bool(torch.isfinite(grad)) and float(grad.abs()) > 0.0
    checks["alpha_gradient_nonzero_finite"] = grad_ok
    details["alpha_grad"] = None if grad is None else float(grad.detach())

    # 3/7. finite forward + backward for the gated model (no NaN/Inf)
    with torch.no_grad():
        out = gated.encode(batch)
    checks["gated_forward_finite"] = bool(torch.isfinite(out).all())

    # 6. relation refresh / aggregation / readout unchanged
    #    (drop the gate parameter -> must be bit-identical to ungated recurrent)
    probe2 = rec.build_gated(0).to(device).eval()
    ungated2 = rec.build_recurrent(0).to(device).eval()
    gated_shared = {
        k: v for k, v in probe2.state_dict().items() if k != "update_gate_logit"
    }
    ungated_state = ungated2.state_dict()
    keys_match = set(gated_shared) == set(ungated_state)
    bit_same_init = keys_match and all(
        torch.equal(gated_shared[k], ungated_state[k]) for k in ungated_state
    )
    probe2.update_gate_logit = None  # deregister; forward skips the gate
    with torch.no_grad():
        diff_no_gate = float((probe2.encode(batch) - ungated2.encode(batch)).abs().max())
    checks["gated_minus_gate_equals_ungated_bitwise"] = diff_no_gate == 0.0
    checks["gated_shared_init_equals_ungated"] = bool(bit_same_init)
    details["gated_minus_gate_max_abs_diff"] = diff_no_gate

    # 8. original ungated builder / checkpoint path intact
    canonical_state_path = vd.STATE_DIR / _CANONICAL_STATE[0]
    checkpoint_ok = False
    recomputed_mae = None
    if canonical_state_path.exists():
        state = torch.load(
            canonical_state_path, map_location="cpu", weights_only=True
        )
        loader = _eval_loader(valid_data)
        targets, preds = vd._predict_state(ungated2, state, loader)
        recomputed_mae = vd._mae(targets, preds)
        checkpoint_ok = (
            abs(recomputed_mae - UNGATED_RAW_REF[0]) < 1e-9
            and int(sum(v.numel() for v in state.values())) == EXPECTED_UNGATED_TOTAL
        )
    checks["ungated_canonical_checkpoint_intact"] = bool(checkpoint_ok)
    details["ungated_checkpoint_recomputed_mae"] = recomputed_mae

    checks["no_nan_inf"] = bool(
        details["alpha_grad"] is not None
        and np.isfinite(details["alpha_grad"])
        and np.isfinite(details["round1_alpha_ratio"]["observed"])
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "details": details,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    return payload


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def report() -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reference": _read_json(RESULTS_DIR / "reference.json")
        if (RESULTS_DIR / "reference.json").exists()
        else None,
        "sanity": _read_json(RESULTS_DIR / "sanity.json")
        if (RESULTS_DIR / "sanity.json").exists()
        else None,
        "alpha": _read_json(RESULTS_DIR / "alpha_report.json")
        if (RESULTS_DIR / "alpha_report.json").exists()
        else None,
        "stability": _read_json(RESULTS_DIR / "stability.json")
        if (RESULTS_DIR / "stability.json").exists()
        else None,
        "decision": _read_json(RESULTS_DIR / "decision.json")
        if (RESULTS_DIR / "decision.json").exists()
        else None,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "final_report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "reference",
            "sanity",
            "train",
            "soup",
            "alpha",
            "stability",
            "decision",
            "report",
            "all",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default=GATED_TAG)
    parser.add_argument("--no-snapshots", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(int(shead.TORCH_THREADS))
    # canonical stochastic training mode: deterministic algorithms OFF for the
    # official runs.  Only needed as an explicit statement; default is False.
    torch.use_deterministic_algorithms(False)

    if args.stage == "reference":
        print(json.dumps(reference(), indent=2, default=str), flush=True)
    if args.stage == "sanity":
        print(json.dumps(sanity(), indent=2, default=str), flush=True)
    if args.stage == "train":
        print(
            json.dumps(
                run_training(
                    args.tag,
                    args.seed,
                    snapshots=not args.no_snapshots,
                ),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "soup":
        print(json.dumps(soup(args.tag, args.seed), indent=2, default=str), flush=True)
    if args.stage == "alpha":
        print(json.dumps(alpha_report(), indent=2, default=str), flush=True)
    if args.stage == "stability":
        print(json.dumps(stability(), indent=2, default=str), flush=True)
    if args.stage == "decision":
        print(json.dumps(decision(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    if args.stage == "all":
        reference()
        sanity()
        for seed in PHASE1_SEEDS:
            run_training(GATED_TAG, seed, snapshots=True)
            soup(GATED_TAG, seed)
        alpha_report()
        stability()
        decision()
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
