"""Frozen official-valid evaluation for zinc-overnight-interface-and-tail-seed0-v1.

Two-step discipline:

1. ``--mode freeze`` writes ``frozen_eval_manifest.json`` **before** any
   official-valid row is read.  It binds: git commit, prep blob, interface
   stats, the model-selection decision file and every model state (sha256) plus
   its train-only calibration bias and inference spec.
2. ``--mode heldout`` (a separate invocation) reads the 1000 official-valid
   rows exactly once, verifies the positional ID/label correspondence, emits
   raw and calibrated predictions for every frozen model, logs the actual
   access time and refuses to continue if the manifest is missing/not frozen.

Official-test is never loaded.  No training, calibration or model selection
happens in this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_overnight_interface_and_tail_seed0_v1 as ov
from tracks.ksvd.experiments.luyin16 import zinc_structure_semantic_factorial_seed0_v1 as zsf

TRACK_ROOT = zjd.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/zinc_overnight_interface_and_tail_seed0_v1"
DECISION_PATH = RESULTS_DIR / "final_decision.json"
MANIFEST_PATH = RESULTS_DIR / "frozen_eval_manifest.json"
ACCESS_PATH = RESULTS_DIR / "heldout_access.json"
CYCLE_LABEL = TRACK_ROOT / "results/zinc_long_cycle_audit/valid_cycle_audit_label.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> dict[str, str]:
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": head, "branch": branch, "dirty": bool(status)}


def freeze() -> dict[str, Any]:
    decision = read_json(DECISION_PATH)
    models: dict[str, Any] = {}
    for entry in decision["frozen_models"]:
        arm = entry["arm"]
        state_path = RESULTS_DIR / f"{arm}_raw_soup_state.pt"
        meta_path = RESULTS_DIR / f"{arm}.json"
        meta = read_json(meta_path)
        models[arm] = {
            "state_path": str(state_path.relative_to(TRACK_ROOT)),
            "state_sha256": file_sha256(state_path),
            "meta_path": str(meta_path.relative_to(TRACK_ROOT)),
            "meta_sha256": file_sha256(meta_path),
            "code_mode": entry["code_mode"],
            "block_mode": entry["block_mode"],
            "adapter_zero": bool(entry.get("adapter_zero", False)),
            "b": float(meta["calibration"]["b"]),
            "b_source": "eval-mode fit median(y - raw) of this model (fit8k or train10k per training stage)",
            "train_mode": meta.get("train_mode"),
            "soup_members": meta.get("soup_members"),
        }
        if entry.get("family") == "cpu_tail":
            models[arm] = dict(entry)
            models[arm]["state_sha256"] = file_sha256(RESULTS_DIR / entry["state_path"])
    manifest = {
        "protocol_version": ov.PROTOCOL_VERSION,
        "freeze_before_heldout": True,
        "frozen_at_utc": _now_utc(),
        "git": git_head(),
        "decision": str(DECISION_PATH.relative_to(TRACK_ROOT)),
        "decision_sha256": file_sha256(DECISION_PATH),
        "prep_blob": str((zjd.PREP_DIR / "fold_objects.npz").relative_to(TRACK_ROOT)),
        "prep_blob_sha256": file_sha256(zjd.PREP_DIR / "fold_objects.npz"),
        "interface_stats": str((RESULTS_DIR / "interface_stats.json").relative_to(TRACK_ROOT)),
        "interface_stats_sha256": file_sha256(RESULTS_DIR / "interface_stats.json"),
        "models": models,
        "wrapper": "InterfaceDeploy.forward (eval); raw=h0+delta -> bridge; cal=raw+b",
        "valid_loader": "encoded_valid.pt + env_valid.pt + frozen 8k train-only prep (apply_prep_train_only)",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "official_test_will_not_be_loaded": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(f"[freeze] manifest written {MANIFEST_PATH} at {manifest['frozen_at_utc']}")
    return manifest


def load_valid_data(blob: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp

    valid = list(torch.load(sdp.CACHE_DIR / "encoded_valid.pt", map_location="cpu", weights_only=False))
    p1run.attach_env(valid, "valid")
    zftd.apply_prep_train_only(valid, blob)
    return valid, {"n_rows": int(len(valid)), "source": "encoded_valid.pt + env_valid.pt", "official_valid_loaded": True}


def predict_model(model: torch.nn.Module, data: Sequence[Any], device: torch.device) -> np.ndarray:
    model.eval()
    preds, targets = [], []
    state = torch.get_rng_state()
    with torch.no_grad():
        for indices in zftd.epoch_batches(len(data), ov.BATCH_SIZE, torch.Generator().manual_seed(0), False):
            batch = zftd.make_batch(data, indices, torch.zeros(len(indices)), device)
            preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
            targets.append(batch.y.view(-1).detach().cpu())
    torch.set_rng_state(state)
    return torch.cat(preds).numpy().astype(np.float64)


def heldout() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise RuntimeError("no frozen_eval_manifest.json; run --mode freeze first")
    manifest = read_json(MANIFEST_PATH)
    if not manifest.get("freeze_before_heldout"):
        raise RuntimeError("manifest is not marked frozen")
    if ACCESS_PATH.exists():
        raise RuntimeError("official-valid was already read in this round; refusing a second read")

    blob = zftd.load_prep_blob()
    valid, meta = load_valid_data(blob)
    y = np.asarray([float(d.y) for d in valid], np.float64)
    label_check: dict[str, Any] | None = None
    ids = None
    if CYCLE_LABEL.exists():
        import pandas as pd

        df = pd.read_csv(CYCLE_LABEL).sort_values("subset_index")
        ids = df["molecule_id"].astype(str).to_numpy()
        n = min(len(y), len(df))
        label_check = {
            "n_label": int(len(df)),
            "y_max_abs_diff": float(np.max(np.abs(y[:n] - df["target"].to_numpy(np.float64)[:n]))),
            "label_source_sha256": file_sha256(CYCLE_LABEL),
        }
    access = {
        "protocol_version": ov.PROTOCOL_VERSION,
        "first_read_utc": _now_utc(),
        "split": "official-valid",
        "n_rows": meta["n_rows"],
        "source": meta["source"],
        "label_check": label_check,
        "official_valid_loaded": True,
        "official_test_loaded": False,
        "note": "single frozen evaluation read; predictions cached before any statistics",
    }
    write_json(ACCESS_PATH, access)
    print(f"[heldout] valid n={meta['n_rows']} label_check={label_check}")

    device = torch.device("cpu")
    torch.set_num_threads(8)
    structural = ov.load_dictionary_objects()
    stats = ov.load_interface_stats()
    out_arrays: dict[str, np.ndarray] = {"y": y}
    if ids is not None:
        out_arrays["ids"] = ids
    table = []
    for arm, m in manifest["models"].items():
        if m.get("family") == "cpu_tail":
            h_raw = None  # not used in this round's main line
            raise RuntimeError("cpu_tail heldout is handled by the CPU runner")
        model = ov.build_interface_model(
            structural, stats, code_mode=m["code_mode"], block_mode=m["block_mode"], device=device
        )
        model.load_state_dict(torch.load(RESULTS_DIR / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False))
        model.adapter_zero = bool(m["adapter_zero"])
        raw = predict_model(model, valid, device)
        raw2 = predict_model(model, valid, device)
        b = float(m["b"])
        cal = raw + b
        out_arrays[f"{arm}_raw"] = raw
        out_arrays[f"{arm}_cal"] = cal
        row = {
            "arm": arm,
            "n": int(len(y)),
            "raw_mae": float(np.mean(np.abs(raw - y))),
            "cal_mae": float(np.mean(np.abs(cal - y))),
            "b": b,
            "replay_max_abs": float(np.max(np.abs(raw - raw2))),
            "state_sha256": m["state_sha256"],
        }
        table.append(row)
        print(f"[heldout] {arm} raw={row['raw_mae']:.6f} cal={row['cal_mae']:.6f} replay={row['replay_max_abs']:.2e}")

    np.savez_compressed(RESULTS_DIR / "final_valid_predictions.npz", **out_arrays)
    with (RESULTS_DIR / "final_valid_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "n", "raw_mae", "cal_mae", "b", "replay_max_abs", "state_sha256"])
        writer.writeheader()
        writer.writerows(table)
    summary = {
        "protocol_version": ov.PROTOCOL_VERSION,
        "table": table,
        "manifest": str(MANIFEST_PATH.relative_to(TRACK_ROOT)),
        "access": access,
        "official_valid_loaded": True,
        "official_test_loaded": False,
        "target_reached": {row["arm"]: bool(row["cal_mae"] < 0.09) for row in table},
    }
    write_json(RESULTS_DIR / "final_valid_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("freeze", "heldout"))
    args = parser.parse_args(argv)
    if args.mode == "freeze":
        freeze()
    else:
        heldout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
