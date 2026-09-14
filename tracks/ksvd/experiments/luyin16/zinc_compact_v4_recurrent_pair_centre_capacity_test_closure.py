"""Official-test closure for the frozen ZINC capacity-decomposition winner (cell A).

Context
-------
The deterministic-A100 2x2 capacity decomposition
(``compact_v4_recurrent_pair_centre_capacity_decomposition_v1``) selected
**cell A** as the frozen configuration:

    h_dim = 64, q_dim = 16, patch_encoder_hidden = 64,
    global_encoder_hidden = 32, T = 2 weight-tied refresh, 85,763 params.

Cell A is validation-optimal, parameter-minimal, and architecturally frozen
before this module exists.  This module performs the *official-test closure*:
it never re-runs architecture selection, never searches K, and never changes the
frozen Top-5 soup weights.

Two pre-registered stages:

* ``freeze``  -- write ``architecture_freeze.json`` recording the frozen cell A
  and its validation evidence while the official test is still unloaded
  (``test_loaded_at_freeze_time: false``).
* ``test``    -- load the official ZINC test split exactly once, evaluate the
  pre-existing seed0/seed1 selection states (raw) and the pre-existing fixed
  Top-5 soups, and report raw/soup mean +/- std.  The 2-seed equal-weight
  prediction average is reported **only** as a diagnostic ensemble.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_pair_centre_capacity_test_closure <stage>

Stages: ``params sanity freeze test report``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity_decomposition as cd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_training_sufficiency as ztraining,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = (
    TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_capacity_test_closure"
)
SOURCE_DIR = TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_capacity_decomposition"

PROTOCOL_VERSION = "compact_v4_recurrent_pair_centre_capacity_test_closure_v1"

SELECTED_CELL = "A"
SELECTED_H = 64
SELECTED_Q = 16
SELECTED_PATCH_HIDDEN = 64
SELECTED_GLOBAL_HIDDEN = 32
SELECTED_ROUNDS = 2

SELECTION_METRIC = "2-seed fixed Top-5 soup valid MAE"
SELECTION_VALUE = 0.1263680279762484
SEEDS = (0, 1)

# Historical CPU-regime references (different execution regime; context only).
HISTORICAL_H48_TEST_RAW_MEAN = 0.114951
HISTORICAL_H96_TEST_RAW_MEAN = 0.107958
HISTORICAL_H96_TEST_RAW_STD = 0.000249
HISTORICAL_H96_TEST_SOUP_MEAN = 0.104990
HISTORICAL_H96_TEST_SOUP_STD = 0.002373

TARGET = 0.10

H_DEPENDENT_STATE = dict(cd.CELLS[SELECTED_CELL])

DETERMINISTIC = False


def _set_deterministic(enabled: bool) -> None:
    """Match the frozen deterministic-A100 execution regime."""
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


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
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    head = REPO_ROOT / ".git" / "HEAD"
    if head.exists():
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = REPO_ROOT / ".git" / text.split(" ", 1)[1].strip()
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()
        return text
    return "unknown"


def _state_dir() -> Path:
    return SOURCE_DIR / "states"


def _soup_dir() -> Path:
    return SOURCE_DIR / "soup_states"


def _selection_path(seed: int) -> Path:
    return _state_dir() / f"cell_{SELECTED_CELL}_seed{seed}_selection_state.pt"


def _soup_path(seed: int) -> Path:
    return _soup_dir() / f"cell_{SELECTED_CELL}_seed{seed}_top5_soup.pt"


# ---------------------------------------------------------------------------
# parameter accounting (frozen cell A)
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    model = cd.build_cell(SELECTED_CELL, 0)
    total = int(sum(p.numel() for p in model.parameters()))
    head = int(sum(p.numel() for p in model.head.parameters()))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_architecture": f"cell_{SELECTED_CELL}",
        "h_dim": SELECTED_H,
        "q_dim": SELECTED_Q,
        "patch_encoder_hidden": SELECTED_PATCH_HIDDEN,
        "global_encoder_hidden": SELECTED_GLOBAL_HIDDEN,
        "recurrence_rounds": SELECTED_ROUNDS,
        "weight_tied": True,
        "total_params": total,
        "head_params": head,
        "backbone_params": total - head,
        "unified_graph_width": int(model.unified_graph_width),
        "matches_expected": total == cd.EXPECTED_PARAMS[SELECTED_CELL],
        "expected_params": cd.EXPECTED_PARAMS[SELECTED_CELL],
        "official_test_loaded": False,
    }
    if not payload["matches_expected"]:
        raise RuntimeError(f"cell A parameter count changed: {total}")
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# sanity -- frozen assets exist and are exactly the validated ones
# ---------------------------------------------------------------------------


def sanity() -> dict[str, Any]:
    decision = _read_json(SOURCE_DIR / "decision.json")
    soup_rows = {
        seed: _read_json(SOURCE_DIR / f"soup_cell_{SELECTED_CELL}_seed{seed}.json")
        for seed in SEEDS
    }
    run_rows = {
        seed: _read_json(SOURCE_DIR / "runs" / f"cell_{SELECTED_CELL}_seed{seed}.json")
        for seed in SEEDS
    }

    per_seed_soup = [float(soup_rows[s]["top5_soup_valid_mae"]) for s in SEEDS]
    per_seed_raw = [float(soup_rows[s]["best_checkpoint_valid_mae"]) for s in SEEDS]
    soup_mean = float(np.mean(per_seed_soup))

    checks = {
        "cell_A_selected_by_decision": decision["best_cell"] == SELECTED_CELL,
        "selection_value_reproduced": abs(
            soup_mean - SELECTION_VALUE
        )
        <= 1.0e-9,
        "selection_matches_decision": abs(
            soup_mean - float(decision["soup_2seed_mean"][SELECTED_CELL])
        )
        <= 1.0e-12,
        "selection_state_seed0_exists": _selection_path(0).exists(),
        "selection_state_seed1_exists": _selection_path(1).exists(),
        "soup_state_seed0_exists": _soup_path(0).exists(),
        "soup_state_seed1_exists": _soup_path(1).exists(),
        "five_top5_checkpoints_per_seed": all(
            len(soup_rows[s]["top5_epochs"]) == vd.SOUP_K for s in SEEDS
        ),
        "soup_param_count_matches_cell_A": all(
            int(soup_rows[s]["parameters"]) == cd.EXPECTED_PARAMS[SELECTED_CELL]
            for s in SEEDS
        ),
        "run_param_count_matches_cell_A": all(
            int(run_rows[s]["parameters"]) == cd.EXPECTED_PARAMS[SELECTED_CELL]
            for s in SEEDS
        ),
        "soup_states_not_yet_read_for_test": True,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "selected_cell": SELECTED_CELL,
        "per_seed_soup_valid_mae": {str(s): per_seed_soup[i] for i, s in enumerate(SEEDS)},
        "per_seed_raw_valid_mae": {str(s): per_seed_raw[i] for i, s in enumerate(SEEDS)},
        "soup_2seed_mean": soup_mean,
        "top5_epochs": {str(s): soup_rows[s]["top5_epochs"] for s in SEEDS},
        "best_epochs": {str(s): int(run_rows[s]["best_epoch"]) for s in SEEDS},
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"capacity test-closure sanity failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# freeze -- BEFORE any official-test read
# ---------------------------------------------------------------------------


def freeze() -> dict[str, Any]:
    unlock_path = RESULTS_DIR / "official_test_unlock.json"
    if unlock_path.exists():
        raise RuntimeError("refusing to re-freeze: official test already unlocked")
    if (RESULTS_DIR / "architecture_freeze.json").exists():
        raise RuntimeError("refusing to overwrite an existing architecture freeze")

    decision = _read_json(SOURCE_DIR / "decision.json")
    soup_rows = {
        seed: _read_json(SOURCE_DIR / f"soup_cell_{SELECTED_CELL}_seed{seed}.json")
        for seed in SEEDS
    }
    run_rows = {
        seed: _read_json(SOURCE_DIR / "runs" / f"cell_{SELECTED_CELL}_seed{seed}.json")
        for seed in SEEDS
    }
    accounting = params()

    per_seed_soup = {
        str(s): float(soup_rows[s]["top5_soup_valid_mae"]) for s in SEEDS
    }
    per_seed_raw = {
        str(s): float(soup_rows[s]["best_checkpoint_valid_mae"]) for s in SEEDS
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_architecture": f"cell_{SELECTED_CELL}",
        "h_dim": SELECTED_H,
        "q_dim": SELECTED_Q,
        "patch_encoder_hidden": SELECTED_PATCH_HIDDEN,
        "global_encoder_hidden": SELECTED_GLOBAL_HIDDEN,
        "recurrence_rounds": SELECTED_ROUNDS,
        "weight_tied": True,
        "params": int(accounting["total_params"]),
        "selection_metric": SELECTION_METRIC,
        "selection_value": SELECTION_VALUE,
        "seeds": [int(s) for s in SEEDS],
        "execution_regime": "deterministic A100",
        "cell_spec": dict(H_DEPENDENT_STATE),
        "validation": {
            "soup_valid_mae": per_seed_soup,
            "soup_2seed_mean": float(decision["soup_2seed_mean"][SELECTED_CELL]),
            "raw_valid_mae": per_seed_raw,
            "raw_2seed_mean": float(decision["raw_2seed_mean"][SELECTED_CELL]),
            "best_epochs": {str(s): int(run_rows[s]["best_epoch"]) for s in SEEDS},
            "top5_epochs": {str(s): soup_rows[s]["top5_epochs"] for s in SEEDS},
        },
        "soup_rule": {
            "K": int(vd.SOUP_K),
            "ranking": "lowest official-valid MAE",
            "tie_rule": "earliest epoch",
            "weight_aggregation": "equal-weight arithmetic parameter mean",
            "no_k_search": True,
            "no_weight_search": True,
            "frozen_before_test": True,
        },
        "selection_state_sha256": {
            str(s): _sha256_file(_selection_path(s)) for s in SEEDS
        },
        "soup_state_sha256": {str(s): _sha256_file(_soup_path(s)) for s in SEEDS},
        "test_loaded_at_freeze_time": False,
        "test_status": "not yet loaded",
        "target_reference": TARGET,
        "historical_cpu_reference": {
            "h48_test_raw_mean": HISTORICAL_H48_TEST_RAW_MEAN,
            "h96_test_raw_mean": HISTORICAL_H96_TEST_RAW_MEAN,
            "h96_test_soup_mean": HISTORICAL_H96_TEST_SOUP_MEAN,
            "execution_regime": "CPU (historical, different regime)",
        },
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "architecture_freeze.json", payload)
    return payload


# ---------------------------------------------------------------------------
# one-shot official test
# ---------------------------------------------------------------------------


def test_eval() -> dict[str, Any]:
    freeze_path = RESULTS_DIR / "architecture_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("refusing test: architecture_freeze.json missing")
    freeze_record = _read_json(freeze_path)
    if freeze_record.get("test_loaded_at_freeze_time") is not False:
        raise RuntimeError("refusing test: freeze record does not assert a pre-test freeze")
    unlock_path = RESULTS_DIR / "official_test_unlock.json"
    if unlock_path.exists():
        raise RuntimeError("refusing test: official test already unlocked once")

    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_architecture": freeze_record["selected_architecture"],
        "frozen_before_test": True,
        "encoding": "transforms fit on official train only (identical to selection)",
        "checkpoints": "pre-existing seed0/seed1 selection states and fixed Top-5 soups",
        "official_test_loaded": True,
        "git_commit": _git_commit(),
    }
    _write_json(unlock_path, lock)

    # --- first and only official-test load ---------------------------------
    train_records, _valid_records = ztraining.load_train_valid_records()
    test_records = ztraining.extract_test_records()
    config = ztraining.base_config()
    _train_data, test_data, audit = ztraining.build_encoded(
        train_records, test_records, config
    )
    del _train_data
    loader = zpp._make_loader(test_data, 128, False, 0)
    targets = np.asarray(
        [float(graph.y.view(-1)[0]) for graph in test_data], dtype=np.float64
    )

    rows: list[dict[str, Any]] = []
    raw_preds: dict[int, np.ndarray] = {}
    soup_preds: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        model = cd.build_cell(SELECTED_CELL, int(seed)).eval()
        t_raw, raw_pred = vd._predict_state(
            model, torch.load(_selection_path(seed), map_location="cpu", weights_only=True), loader
        )
        t_soup, soup_pred = vd._predict_state(
            model, torch.load(_soup_path(seed), map_location="cpu", weights_only=True), loader
        )
        if not np.array_equal(t_raw, targets) or not np.array_equal(t_soup, targets):
            raise RuntimeError("official-test target order mismatch")
        raw_preds[seed] = raw_pred
        soup_preds[seed] = soup_pred
        rows.append(
            {
                "seed": int(seed),
                "raw_selection_test_mae": float(np.mean(np.abs(targets - raw_pred))),
                "soup_test_mae": float(np.mean(np.abs(targets - soup_pred))),
            }
        )

    raw_values = np.array([row["raw_selection_test_mae"] for row in rows])
    soup_values = np.array([row["soup_test_mae"] for row in rows])

    def _mean_std(values: np.ndarray) -> tuple[float, float]:
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
        return mean, std

    raw_mean, raw_std = _mean_std(raw_values)
    soup_mean, soup_std = _mean_std(soup_values)

    raw_ensemble = np.mean(np.stack([raw_preds[s] for s in SEEDS], axis=0), axis=0)
    soup_ensemble = np.mean(np.stack([soup_preds[s] for s in SEEDS], axis=0), axis=0)
    raw_ensemble_mae = float(np.mean(np.abs(targets - raw_ensemble)))
    soup_ensemble_mae = float(np.mean(np.abs(targets - soup_ensemble)))

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_architecture": freeze_record["selected_architecture"],
        "parameter_count": int(freeze_record["params"]),
        "execution_regime": freeze_record["execution_regime"],
        "n_test": int(targets.shape[0]),
        "rows": rows,
        "raw_test_per_seed": {str(r["seed"]): r["raw_selection_test_mae"] for r in rows},
        "soup_test_per_seed": {str(r["seed"]): r["soup_test_mae"] for r in rows},
        "raw_test_mean": raw_mean,
        "raw_test_std": raw_std,
        "soup_test_mean": soup_mean,
        "soup_test_std": soup_std,
        "diagnostic_raw_2seed_ensemble_test_mae": raw_ensemble_mae,
        "diagnostic_soup_2seed_ensemble_test_mae": soup_ensemble_mae,
        "diagnostic_note": (
            "the 2-seed prediction ensemble is a diagnostic, NOT a single-model result"
        ),
        "distance_to_0.10": {
            "raw": raw_mean - TARGET,
            "soup": soup_mean - TARGET,
            "diagnostic_raw_ensemble": raw_ensemble_mae - TARGET,
            "diagnostic_soup_ensemble": soup_ensemble_mae - TARGET,
        },
        "historical_cpu_reference": {
            "h48_test_raw_mean": HISTORICAL_H48_TEST_RAW_MEAN,
            "h96_test_raw_mean": HISTORICAL_H96_TEST_RAW_MEAN,
            "h96_test_raw_std": HISTORICAL_H96_TEST_RAW_STD,
            "h96_test_soup_mean": HISTORICAL_H96_TEST_SOUP_MEAN,
            "h96_test_soup_std": HISTORICAL_H96_TEST_SOUP_STD,
            "regime_mismatch": True,
        },
        "raw_minus_historical_h48": raw_mean - HISTORICAL_H48_TEST_RAW_MEAN,
        "raw_minus_historical_h96": raw_mean - HISTORICAL_H96_TEST_RAW_MEAN,
        "soup_minus_historical_h96_soup": soup_mean - HISTORICAL_H96_TEST_SOUP_MEAN,
        "official_test_loaded": True,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "audit": {key: audit[key] for key in sorted(audit) if key != "topology"},
    }
    _write_json(RESULTS_DIR / "official_test_results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "parameter_accounting": _maybe("parameter_accounting.json"),
        "sanity": (lambda d: None if d is None else d["checks"])(_maybe("sanity.json")),
        "architecture_freeze": _maybe("architecture_freeze.json"),
        "official_test_results": _maybe("official_test_results.json"),
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["params", "sanity", "freeze", "test", "report"]
    )
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    global DETERMINISTIC
    DETERMINISTIC = bool(args.deterministic)
    torch.set_num_threads(4)
    _set_deterministic(DETERMINISTIC)
    if args.stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    if args.stage == "sanity":
        print(json.dumps(sanity(), indent=2, default=str), flush=True)
    if args.stage == "freeze":
        print(json.dumps(freeze(), indent=2, default=str), flush=True)
    if args.stage == "test":
        print(json.dumps(test_eval(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
