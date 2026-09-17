"""Official-test closure for the MolHIV local-token-null screen (B-Null / Constant-32).

Authorised by the user after the MolHIV round: report B-Null and Constant-32 on
the official OGBG-MolHIV test split, with **both** the best-valid ``raw``
checkpoint and the fixed Top-5 ``soup`` frozen before any test read.

Discipline (mirrors ``molhiv_recurrent_pair_centre.py`` freeze/test):

* ``freeze`` -- write ``architecture_freeze.json`` recording the already-selected
  valid-only states and their SHA-256, while the official test is still unloaded
  (``test_loaded_at_freeze_time: false``).
* ``test``   -- write ``official_test_unlock.json``, then load the official test
  split exactly once and evaluate the pre-frozen raw and soup states.

This is a **post-hoc terminal read of a non-promoted screen**: the numbers must
not be used for promotion or any further architecture selection.  The frozen
typed-lookup test result (0.762095 soup / 0.762605 raw, 2-seed) is reported as
context only.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.molhiv_local_token_null_test_closure <stage>

Stages: ``freeze test report``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import molhiv_local_token_null as mltn
from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc

# ---------------------------------------------------------------------------
# layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = rpc.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/molhiv_local_token_null_test_closure"
SOURCE_DIR = mltn.RESULTS_DIR

PROTOCOL_VERSION = "molhiv_local_token_null_test_closure_v1"
SEED = 0
CANDIDATES = ("null", "constant")


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
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover - best effort provenance
        return "unknown"


def _expected_total(representation: str, typed_vocab: int, parent_vocab: int) -> int:
    model = rpc.build_model(
        int(typed_vocab), int(parent_vocab), 0, patch_representation=representation
    )
    return int(rpc._n_params(model))


def _raw_path(representation: str) -> Path:
    return SOURCE_DIR / f"raw_{representation}_seed{SEED}.pt"


def _soup_path(representation: str) -> Path:
    return SOURCE_DIR / f"soup_{representation}_seed{SEED}.pt"


# ---------------------------------------------------------------------------
# freeze -- BEFORE any official-test read
# ---------------------------------------------------------------------------


def freeze() -> dict[str, Any]:
    if (RESULTS_DIR / "official_test_unlock.json").exists():
        raise RuntimeError("refusing to re-freeze: official test already unlocked")
    if (RESULTS_DIR / "architecture_freeze.json").exists():
        raise RuntimeError("refusing to overwrite an existing architecture freeze")

    runs = {
        representation: _read_json(
            SOURCE_DIR / f"run_{representation}_seed{SEED}.json"
        )
        for representation in CANDIDATES
    }
    typed_vocab = int(runs[CANDIDATES[0]]["typed_vocabulary_size_with_oov"])
    parent_vocab = int(runs[CANDIDATES[0]]["parent_vocabulary_size_with_oov"])
    checks: dict[str, bool] = {}
    for representation in CANDIDATES:
        run = runs[representation]
        checks[f"{representation}_params_match_expected"] = bool(
            int(run["parameters"])
            == _expected_total(representation, typed_vocab, parent_vocab)
        )
        checks[f"{representation}_source_test_never_loaded"] = bool(
            run.get("official_test_loaded") is False
        )
        checks[f"{representation}_states_exist"] = bool(
            _raw_path(representation).exists() and _soup_path(representation).exists()
        )
    typed_path = rpc.RESULTS_DIR / "official_test_results.json"
    checks["typed_reference_test_present"] = bool(typed_path.exists())

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "post-hoc terminal read of the non-promoted MolHIV local-token-null "
            "screen (B-Null / Constant-32); not usable for promotion"
        ),
        "selection_metric": "official-valid fixed Top-5 soup ROC-AUC (already selected)",
        "seed": int(SEED),
        "candidates": {
            representation: {
                "representation": representation,
                "params": int(runs[representation]["parameters"]),
                "best_epoch": int(runs[representation]["best_epoch"]),
                "raw_valid_auc": float(
                    runs[representation]["raw_valid_auc_recomputed"]
                ),
                "soup_valid_auc": float(runs[representation]["soup_valid_auc"]),
                "top5_epochs": [
                    int(e) for e in runs[representation]["top5_epochs"]
                ],
            }
            for representation in CANDIDATES
        },
        "soup_rule": {
            "K": int(rpc.SOUP_K),
            "ranking": "highest official-valid ROC-AUC",
            "tie_rule": "earliest epoch",
            "weight_aggregation": "equal-weight arithmetic parameter mean",
            "frozen_before_test": True,
        },
        "state_sha256": {
            representation: {
                "raw_state": _sha256_file(_raw_path(representation)),
                "top5_soup": _sha256_file(_soup_path(representation)),
            }
            for representation in CANDIDATES
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "typed_reference_test": (
            _read_json(typed_path) if typed_path.exists() else None
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


def test_eval(device: str = "cuda") -> dict[str, Any]:
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
            "checkpoints": "pre-existing raw / fixed Top-5 soup selection states",
            "official_test_loaded": True,
        },
    )

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("test requested CUDA but CUDA is unavailable")

    # --- first and only official-test load ---------------------------------
    train_records, _ = rpc._load_records("train")
    transforms = rpc.fit_train_transforms(train_records)
    typed_size = len(transforms["typed_vocabulary"]) + 1
    parent_size = len(transforms["parent_vocabulary"]) + 1
    del train_records
    gc.collect()

    test_records, metadata = rpc._load_records("test")
    test_data = rpc.encode_records(test_records, transforms)
    del test_records
    gc.collect()
    test_loader = mpp._make_loader(list(test_data), rpc.BATCH_SIZE, False, 0)
    targets = np.asarray(
        [float(g.y.view(-1)[0]) for g in test_data], dtype=np.float64
    )

    rows: list[dict[str, Any]] = []
    for representation in CANDIDATES:
        model = rpc.build_model(
            typed_size, parent_size, SEED, patch_representation=representation
        ).to(dev)
        t_raw, raw_logits = rpc._predict(
            model,
            torch.load(_raw_path(representation), map_location="cpu", weights_only=True),
            test_loader,
            dev,
        )
        t_soup, soup_logits = rpc._predict(
            model,
            torch.load(_soup_path(representation), map_location="cpu", weights_only=True),
            test_loader,
            dev,
        )
        if not np.array_equal(t_raw, targets) or not np.array_equal(t_soup, targets):
            raise RuntimeError("official-test target order mismatch")
        rows.append(
            {
                "representation": representation,
                "params": int(freeze_record["candidates"][representation]["params"]),
                "raw_valid_auc": float(
                    freeze_record["candidates"][representation]["raw_valid_auc"]
                ),
                "soup_valid_auc": float(
                    freeze_record["candidates"][representation]["soup_valid_auc"]
                ),
                "raw_test_auc": rpc._auc(targets, raw_logits),
                "soup_test_auc": rpc._auc(targets, soup_logits),
            }
        )

    typed_path = rpc.RESULTS_DIR / "official_test_results.json"
    typed = _read_json(typed_path) if typed_path.exists() else None
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "note": (
            "post-hoc terminal read of a non-promoted screen; soup and raw are "
            "both reported and neither may be used for selection"
        ),
        "seed": int(SEED),
        "n_test": int(targets.shape[0]),
        "rows": rows,
        "raw_test_auc": {r["representation"]: r["raw_test_auc"] for r in rows},
        "soup_test_auc": {r["representation"]: r["soup_test_auc"] for r in rows},
        "valid_to_test_delta": {
            r["representation"]: {
                "raw": r["raw_test_auc"] - r["raw_valid_auc"],
                "soup": r["soup_test_auc"] - r["soup_valid_auc"],
            }
            for r in rows
        },
        "typed_reference_test": (
            None
            if typed is None
            else {
                "params": 1076589,
                "raw_test_mean_2seed": float(typed["raw_test_mean"]),
                "soup_test_mean_2seed": float(typed["soup_test_mean"]),
                "raw_test_std_2seed": float(typed["raw_test_std"]),
                "soup_test_std_2seed": float(typed["soup_test_std"]),
                "note": "2-seed mean from the frozen typed-lookup model",
            }
        ),
        "test_metadata": {k: v for k, v in metadata.items() if k != "records"},
        "official_test_loaded": True,
        "git_commit": _git_commit(),
        "audit_note": "raw and Top-5 soup states frozen in architecture_freeze.json",
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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.stage == "freeze":
        result = freeze()
    elif args.stage == "test":
        result = test_eval(device=args.device)
    else:
        result = report()
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
