"""Official-test closure for the ZINC local-token-null screen (B-Null / Constant).

Authorised by the user after the ZINC round: report B-Null and Constant-16 on
the official ZINC test split, with **both** the best-valid ``raw`` checkpoint
and the fixed Top-5 ``soup`` frozen before any test read.

Discipline (mirrors
``zinc_compact_v4_recurrent_pair_centre_capacity_test_closure.py``):

* ``freeze`` -- write ``architecture_freeze.json`` recording the already-selected
  valid-only states (raw selection state + fixed equal-weight Top-5 soup) and
  their SHA-256, while the official test is still unloaded
  (``test_loaded_at_freeze_time: false``).
* ``test``   -- write ``official_test_unlock.json``, then load the official ZINC
  test split exactly once and evaluate the pre-frozen raw and soup states.

This is a **post-hoc terminal read of a non-promoted screen**: the numbers must
not be used for promotion or any further architecture selection.  The ZINC
cell-A typed-lookup test closure (0.106717 soup / 0.109361 raw, 2-seed) is
reported as context only.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_local_token_null_test_closure <stage>

Stages: ``freeze test report``.
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
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_training_sufficiency as ztraining,
)
from tracks.ksvd.experiments.luyin16 import zinc_local_token_null as ltn
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/local_token_null_test_closure"
SOURCE_DIR = ltn.RESULTS_DIR

CELLA_CLOSURE_DIR = (
    TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_capacity_test_closure"
)

PROTOCOL_VERSION = "zinc_local_token_null_test_closure_v1"
SEED = 0

#: (representation, source tag)
CANDIDATES = (("null", "lt_null"), ("constant", "lt_constant"))

EXPECTED_TOTAL = {
    "null": ltn.NULL_TOTAL_PARAMS,
    "constant": ltn.NULL_TOTAL_PARAMS + ltn.CONSTANT_EXTRA_PARAMS,
}


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
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover - best effort provenance
        return "unknown"


def _set_deterministic(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def _state_path(tag: str) -> Path:
    return SOURCE_DIR / "states" / f"{tag}_seed{SEED}_selection_state.pt"


def _soup_path(tag: str) -> Path:
    return SOURCE_DIR / "soup_states" / f"{tag}_seed{SEED}_top5_soup.pt"


def _candidate_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for representation, tag in CANDIDATES:
        rows[representation] = {
            "tag": tag,
            "run": _read_json(ltn.RUNS_DIR / f"{tag}_seed{SEED}.json"),
            "soup": _read_json(ltn.RESULTS_DIR / f"soup_{tag}_seed{SEED}.json"),
        }
    return rows


# ---------------------------------------------------------------------------
# freeze -- BEFORE any official-test read
# ---------------------------------------------------------------------------


def freeze() -> dict[str, Any]:
    if (RESULTS_DIR / "official_test_unlock.json").exists():
        raise RuntimeError("refusing to re-freeze: official test already unlocked")
    if (RESULTS_DIR / "architecture_freeze.json").exists():
        raise RuntimeError("refusing to overwrite an existing architecture freeze")

    rows = _candidate_rows()
    checks: dict[str, bool] = {}
    for representation, tag in CANDIDATES:
        run = rows[representation]["run"]
        soup = rows[representation]["soup"]
        checks[f"{representation}_params_match_expected"] = bool(
            int(run["parameters"]) == EXPECTED_TOTAL[representation]
        )
        checks[f"{representation}_raw_valid_matches_run"] = bool(
            abs(
                float(soup["best_checkpoint_valid_mae"])
                - float(run["best_valid_mae"])
            )
            < 1e-6
        )
        checks[f"{representation}_source_test_never_loaded"] = bool(
            run.get("official_test_loaded") is False
            and soup.get("official_test_loaded") is False
        )
        checks[f"{representation}_states_exist"] = bool(
            _state_path(tag).exists() and _soup_path(tag).exists()
        )
    cella_path = CELLA_CLOSURE_DIR / "official_test_results.json"
    checks["cell_a_reference_test_present"] = bool(cella_path.exists())

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "post-hoc terminal read of the non-promoted ZINC local-token-null "
            "screen (B-Null / Constant-16); not usable for promotion"
        ),
        "selection_metric": "official-valid fixed Top-5 soup MAE (already selected)",
        "seed": int(SEED),
        "candidates": {
            representation: {
                "representation": representation,
                "params": int(rows[representation]["run"]["parameters"]),
                "best_epoch": int(rows[representation]["run"]["best_epoch"]),
                "raw_valid_mae": float(
                    rows[representation]["soup"]["best_checkpoint_valid_mae"]
                ),
                "soup_valid_mae": float(
                    rows[representation]["soup"]["top5_soup_valid_mae"]
                ),
                "top5_epochs": [
                    int(e) for e in rows[representation]["soup"]["top5_epochs"]
                ],
            }
            for representation, _ in CANDIDATES
        },
        "soup_rule": {
            "K": int(vd.SOUP_K),
            "ranking": "lowest official-valid MAE",
            "tie_rule": "earliest epoch",
            "weight_aggregation": "equal-weight arithmetic parameter mean",
            "frozen_before_test": True,
        },
        "state_sha256": {
            tag: {
                "selection_state": _sha256_file(_state_path(tag)),
                "top5_soup": _sha256_file(_soup_path(tag)),
            }
            for _, tag in CANDIDATES
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "cell_a_typed_reference_test": (
            _read_json(cella_path) if cella_path.exists() else None
        ),
        "test_loaded_at_freeze_time": False,
        "official_test_loaded": False,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "architecture_freeze.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"freeze checks failed: {failed}")
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

    _write_json(
        unlock_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "frozen_before_test": True,
            "encoding": "transforms fit on official train only (identical to selection)",
            "checkpoints": "pre-existing seed0 raw selection state and fixed Top-5 soup",
            "official_test_loaded": True,
            "git_commit": _git_commit(),
        },
    )

    # --- first and only official-test load ---------------------------------
    train_records, _valid_records = ztraining.load_train_valid_records()
    test_records = ztraining.extract_test_records()
    config = ztraining.base_config()
    _train_data, test_data, audit = ztraining.build_encoded(
        train_records, test_records, config
    )
    del _train_data
    loader = zpp._make_loader(list(test_data), 128, False, 0)
    targets = np.asarray(
        [float(graph.y.view(-1)[0]) for graph in test_data], dtype=np.float64
    )

    rows: list[dict[str, Any]] = []
    for representation, tag in CANDIDATES:
        model = (
            ltn.build_null(SEED) if representation == "null" else ltn.build_constant(SEED)
        )
        t_raw, raw_pred = vd._predict_state(
            model, torch.load(_state_path(tag), map_location="cpu", weights_only=True), loader
        )
        t_soup, soup_pred = vd._predict_state(
            model, torch.load(_soup_path(tag), map_location="cpu", weights_only=True), loader
        )
        if not np.array_equal(t_raw, targets) or not np.array_equal(t_soup, targets):
            raise RuntimeError("official-test target order mismatch")
        rows.append(
            {
                "representation": representation,
                "params": int(freeze_record["candidates"][representation]["params"]),
                "raw_valid_mae": float(
                    freeze_record["candidates"][representation]["raw_valid_mae"]
                ),
                "soup_valid_mae": float(
                    freeze_record["candidates"][representation]["soup_valid_mae"]
                ),
                "raw_test_mae": float(vd._mae(targets, raw_pred)),
                "soup_test_mae": float(vd._mae(targets, soup_pred)),
            }
        )

    cella_path = CELLA_CLOSURE_DIR / "official_test_results.json"
    cella = _read_json(cella_path) if cella_path.exists() else None
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "note": (
            "post-hoc terminal read of a non-promoted screen; soup and raw are "
            "both reported and neither may be used for selection"
        ),
        "seed": int(SEED),
        "n_test": int(targets.shape[0]),
        "rows": rows,
        "raw_test_mae": {r["representation"]: r["raw_test_mae"] for r in rows},
        "soup_test_mae": {r["representation"]: r["soup_test_mae"] for r in rows},
        "valid_to_test_delta": {
            r["representation"]: {
                "raw": r["raw_test_mae"] - r["raw_valid_mae"],
                "soup": r["soup_test_mae"] - r["soup_valid_mae"],
            }
            for r in rows
        },
        "cell_a_typed_reference_test": (
            None
            if cella is None
            else {
                "params": 85763,
                "raw_test_mean_2seed": float(cella["raw_test_mean"]),
                "soup_test_mean_2seed": float(cella["soup_test_mean"]),
                "raw_test_std_2seed": float(cella["raw_test_std"]),
                "soup_test_std_2seed": float(cella["soup_test_std"]),
                "note": "2-seed mean, different execution regime (deterministic A100)",
            }
        ),
        "official_test_loaded": True,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "audit": {key: audit[key] for key in sorted(audit) if key != "topology"},
    }
    _write_json(RESULTS_DIR / "official_test_results.json", payload)
    return payload


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "architecture_freeze": _maybe("architecture_freeze.json"),
        "official_test_results": _maybe("official_test_results.json"),
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "test", "report"])
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    _set_deterministic(bool(args.deterministic))
    if args.stage == "freeze":
        result = freeze()
    elif args.stage == "test":
        result = test_eval()
    else:
        result = report()
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
