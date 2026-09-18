"""Official-test closure for the MolHIV B-Null frozen small-head probe.

Authorised by the user because MolHIV valid->test generalisation is difficult
and the valid-only probe cannot answer whether the small head transfers.

Reported on the official OGBG-MolHIV test split (one read):

* frozen B-Null backbone (joint head), raw + fixed Top-5 soup -- matched
  reference / reproduction of the earlier local-token-null closure;
* ``H_refit``  (423->192->96->1, 100,417 params) raw + soup;
* ``H_small32`` (423->32->16->1, 14,177 params) raw + soup.

Discipline (mirrors ``molhiv_local_token_null_test_closure.py`` and
``molhiv_recurrent_pair_centre.py``):

* ``freeze`` -- materialise the already-valid-selected head states and write
  ``architecture_freeze.json`` while the official test is still unloaded
  (``test_loaded_at_freeze_time: false``);
* ``test``   -- write ``official_test_unlock.json``, then load the official test
  split exactly once, extract the test representation ``R`` from the *frozen*
  backbone and evaluate the pre-frozen states.

This is a terminal read of a non-promoted probe: no number from it may be used
for promotion or further architecture selection.  Test labels are used only to
score the frozen predictions.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.molhiv_bnull_small_head_probe_test_closure <stage>

Stages: ``freeze test report``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import molhiv_bnull_small_head_probe as probe
from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc

REPO_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/molhiv_bnull_small_head_probe_test_closure"
SOURCE_DIR = probe.RESULTS_DIR
BNULL_DIR = probe.BNULL_RESULTS_DIR

PROTOCOL_VERSION = "molhiv_bnull_small_head_probe_test_closure_v1"
SEED = probe.SEED
HEAD_NAMES = ("H_refit", "H_small32")


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
    head = REPO_ROOT / ".git" / "HEAD"
    if head.exists():
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = REPO_ROOT / ".git" / text.split(" ", 1)[1].strip()
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()
        return text
    return "unknown"


def _run_path(name: str) -> Path:
    return SOURCE_DIR / f"run_{name}_seed{SEED}.json"


def _snapshot_dir(name: str) -> Path:
    return SOURCE_DIR / f"snapshots_{name}"


def _selected_head_states(name: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    run = _read_json(_run_path(name))
    best_epoch = int(run["best_epoch"])
    top5 = [int(e) for e in run["top5_epochs"]]
    raw = torch.load(
        _snapshot_dir(name) / f"epoch_{best_epoch:03d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    rows = [
        {"path": str(_snapshot_dir(name) / f"epoch_{epoch:03d}.pt"), "epoch": epoch}
        for epoch in top5
    ]
    soup = rpc._soup_state(rows)
    return raw, soup


def _head_logits(head: torch.nn.Module, R: torch.Tensor, device: torch.device) -> np.ndarray:
    head.to(device)
    head.eval()
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, R.shape[0], probe.BATCH_SIZE):
            out.append(
                head(R[start:start + probe.BATCH_SIZE].to(device)).view(-1).cpu()
            )
    return torch.cat(out).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# freeze -- BEFORE any official-test read
# ---------------------------------------------------------------------------


def freeze() -> dict[str, Any]:
    if (RESULTS_DIR / "official_test_unlock.json").exists():
        raise RuntimeError("refusing to re-freeze: official test already unlocked")
    if (RESULTS_DIR / "architecture_freeze.json").exists():
        raise RuntimeError("refusing to overwrite an existing architecture freeze")

    extraction = _read_json(probe.R_META_PATH)
    checks: dict[str, bool] = {
        "R_dim_is_423": bool(int(extraction["R_dim"]) == 423),
        "no_test_representation_extracted_during_probe": bool(
            extraction["official_test_representation_extracted"] is False
        ),
        "probe_never_loaded_test": bool(extraction["official_test_loaded"] is False),
        "frozen_head_reproduced_logits_exactly": bool(
            float(extraction["frozen_head_reproduces_logits_max_abs"]) == 0.0
        ),
    }

    valid_payload = torch.load(
        probe.R_VALID_PATH, map_location="cpu", weights_only=True
    )
    R_valid, y_valid = valid_payload["R"].float(), valid_payload["y"].float()
    y_valid_np = y_valid.numpy().astype(np.float64)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    head_records: dict[str, Any] = {}
    state_hashes: dict[str, Any] = {}
    for name in HEAD_NAMES:
        raw_state, soup_state = _selected_head_states(name)
        raw_path = RESULTS_DIR / f"selected_{name}_raw.pt"
        soup_path = RESULTS_DIR / f"selected_{name}_soup.pt"
        torch.save(raw_state, raw_path)
        torch.save(soup_state, soup_path)

        head = probe.HEADS[name]()
        head.load_state_dict(raw_state, strict=True)
        raw_valid_auc = probe._auc(y_valid_np, _head_logits(head, R_valid, torch.device("cpu")))
        head.load_state_dict(soup_state, strict=True)
        soup_valid_auc = probe._auc(y_valid_np, _head_logits(head, R_valid, torch.device("cpu")))

        run = _read_json(_run_path(name))
        checks[f"{name}_raw_valid_reproduces"] = bool(
            abs(raw_valid_auc - float(run["raw_valid_auc_recomputed"])) < 1e-9
        )
        checks[f"{name}_soup_valid_reproduces"] = bool(
            abs(soup_valid_auc - float(run["soup_valid_auc"])) < 1e-9
        )
        checks[f"{name}_params_match"] = bool(
            int(run["parameters"]) == int(probe._n_params(probe.HEADS[name]()))
        )
        head_records[name] = {
            "params": int(run["parameters"]),
            "best_epoch": int(run["best_epoch"]),
            "top5_epochs": [int(e) for e in run["top5_epochs"]],
            "raw_valid_auc": float(raw_valid_auc),
            "soup_valid_auc": float(soup_valid_auc),
        }
        state_hashes[name] = {
            "raw": _sha256_file(raw_path),
            "soup": _sha256_file(soup_path),
        }

    # matched backbone reference re-scored on valid at freeze time
    valid_targets = _read_json(probe.BNULL_RUN_PATH)
    backbone_record = {
        "params": int(valid_targets["parameters"]),
        "best_epoch": int(valid_targets["best_epoch"]),
        "raw_valid_auc": float(valid_targets["raw_valid_auc_recomputed"]),
        "soup_valid_auc": float(valid_targets["soup_valid_auc"]),
        "prior_test_closure": {
            "raw_test_auc": 0.743411,
            "soup_test_auc": 0.760501,
            "source": str(
                TRACK_ROOT / "results/molhiv_local_token_null_test_closure/official_test_results.json"
            ),
            "note": "previous one-shot read; reproduced here only as a matched reference",
        },
    }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "terminal one-shot official-test read of the non-promoted MolHIV "
            "B-Null frozen small-head probe; not usable for promotion/selection"
        ),
        "authorisation": "explicit user request (MolHIV generalisation)",
        "selection_metric": "official-valid fixed Top-5 soup ROC-AUC (already selected)",
        "seed": int(SEED),
        "R_dim": int(extraction["R_dim"]),
        "backbone": backbone_record,
        "heads": head_records,
        "head_state_sha256": state_hashes,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "test_loaded_at_freeze_time": False,
        "official_test_loaded": False,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "architecture_freeze.json", payload)
    if not payload["passed"]:
        failed = [key for key, ok in checks.items() if not ok]
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
            "checkpoints": "pre-existing valid-selected backbone soup and head states",
            "official_test_loaded": True,
        },
    )

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("test requested CUDA but CUDA is unavailable")

    train_records, _ = rpc._load_records("train")
    transforms = rpc.fit_train_transforms(train_records)
    typed_size = len(transforms["typed_vocabulary"]) + 1
    parent_size = len(transforms["parent_vocabulary"]) + 1
    del train_records
    gc.collect()

    # --- first and only official-test load ---------------------------------
    test_records, metadata = rpc._load_records("test")
    test_data = rpc.encode_records(test_records, transforms)
    del test_records
    gc.collect()
    test_loader = mpp._make_loader(list(test_data), rpc.BATCH_SIZE, False, 0)
    targets = np.asarray(
        [float(graph.y.view(-1)[0]) for graph in test_data], dtype=np.float64
    )

    # frozen backbone + test representation (labels are never used for R)
    model = rpc.build_model(
        typed_size, parent_size, SEED, patch_representation="null"
    ).to(dev)
    backbone_dir = BNULL_DIR
    soup_state = torch.load(
        probe.BNULL_SOUP_PATH, map_location="cpu", weights_only=True
    )
    raw_state = torch.load(
        backbone_dir / "raw_null_seed0.pt", map_location="cpu", weights_only=True
    )
    _t, backbone_raw_logits = rpc._predict(model, raw_state, test_loader, dev)
    _t, backbone_soup_logits = rpc._predict(model, soup_state, test_loader, dev)
    if not np.array_equal(_t, targets):
        raise RuntimeError("official-test target order mismatch (backbone)")

    R_test, y_test = probe._extract_R_and_y(model, test_loader, dev)
    if not np.array_equal(y_test.numpy().astype(np.float64), targets):
        raise RuntimeError("official-test target order mismatch (R extraction)")

    rows: list[dict[str, Any]] = [
        {
            "head": "B-Null-backbone",
            "params": int(freeze_record["backbone"]["params"]),
            "soup_valid_auc": float(freeze_record["backbone"]["soup_valid_auc"]),
            "raw_valid_auc": float(freeze_record["backbone"]["raw_valid_auc"]),
            "raw_test_auc": rpc._auc(targets, backbone_raw_logits),
            "soup_test_auc": rpc._auc(targets, backbone_soup_logits),
        }
    ]
    for name in HEAD_NAMES:
        head = probe.HEADS[name]()
        raw_head = torch.load(
            RESULTS_DIR / f"selected_{name}_raw.pt", map_location="cpu", weights_only=True
        )
        soup_head = torch.load(
            RESULTS_DIR / f"selected_{name}_soup.pt", map_location="cpu", weights_only=True
        )
        head.load_state_dict(raw_head, strict=True)
        raw_logits = _head_logits(head, R_test, dev)
        head.load_state_dict(soup_head, strict=True)
        soup_logits = _head_logits(head, R_test, dev)
        rows.append(
            {
                "head": name,
                "params": int(freeze_record["heads"][name]["params"]),
                "soup_valid_auc": float(freeze_record["heads"][name]["soup_valid_auc"]),
                "raw_valid_auc": float(freeze_record["heads"][name]["raw_valid_auc"]),
                "raw_test_auc": rpc._auc(targets, raw_logits),
                "soup_test_auc": rpc._auc(targets, soup_logits),
            }
        )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "note": (
            "terminal read of a non-promoted probe; raw and soup are both "
            "reported and neither may be used for selection"
        ),
        "seed": int(SEED),
        "n_test": int(targets.shape[0]),
        "n_test_positive": int(targets.sum()),
        "R_dim": int(R_test.shape[1]),
        "rows": rows,
        "raw_test_auc": {row["head"]: row["raw_test_auc"] for row in rows},
        "soup_test_auc": {row["head"]: row["soup_test_auc"] for row in rows},
        "valid_to_test_delta_soup": {
            row["head"]: row["soup_test_auc"] - row["soup_valid_auc"] for row in rows
        },
        "R_test_sha256": hashlib.sha256(
            np.ascontiguousarray(R_test.numpy()).tobytes()
        ).hexdigest(),
        "test_metadata": {key: value for key, value in metadata.items() if key != "records"},
        "official_test_loaded": True,
        "git_commit": _git_commit(),
        "audit_note": "valid-selected states frozen in architecture_freeze.json",
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
