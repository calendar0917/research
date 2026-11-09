"""Valid-only channel-zeroing sensitivity for the BOC best-Validation models.

For each arm, evaluate the selected best-Valid checkpoint on the official
Validation split with (a) both channels, (b) ring channel zeroed, (c) bond
channel zeroed, (d) both zeroed.  This is a diagnostic, never a model-selection
signal, and it never reads an official Test label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import torch  # noqa: E402

import boc_common as C  # noqa: E402
import boc_model as M  # noqa: E402
import boc_structures as S  # noqa: E402
import boc_test as T  # noqa: E402

MODES = {
    "both_channels": {"ring_zero": False, "bond_zero": False},
    "ring_zero": {"ring_zero": True, "bond_zero": False},
    "bond_zero": {"ring_zero": False, "bond_zero": True},
    "both_zero": {"ring_zero": True, "bond_zero": True},
}


def evaluate(
    model: torch.nn.Module,
    task: Any,
    struct: Mapping[str, Any],
    rows: np.ndarray,
    device: torch.device,
    *,
    ring_zero: bool,
    bond_zero: bool,
    batch_size: int = 256,
) -> dict[str, Any]:
    model.eval()
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, rows.size, int(batch_size)):
            chunk = rows[start : start + int(batch_size)]
            batch = S.collate_boc(task, struct, chunk.tolist())
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch, ring_zero=ring_zero, bond_zero=bond_zero)
            logits.append(out.detach().cpu().numpy().astype(np.float32))
    logits_np = np.concatenate(logits, axis=0)
    probs = C.BC.PC.sigmoid(logits_np)
    report = C.MET.validation_report(np.asarray(task.labels[rows]), probs)
    report["bce"] = float(C.MET.masked_bce_numpy(logits_np, np.asarray(task.labels[rows])))
    report["logits_sha256"] = C.sha256_bytes(np.ascontiguousarray(logits_np, dtype=np.float32).tobytes())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default=str(T.SELECTION_PATH))
    parser.add_argument("--struct-cache", default=str(C.STRUCT_CACHE_DIR))
    parser.add_argument("--out", default=str(C.ANALYSIS / "zeroing_sensitivity.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selection = C.read_json(Path(args.selection))
    if selection.get("test_read") is not False:
        raise RuntimeError("selection manifest must be pre-Test")
    task = C.BC.SD.load_task()
    struct = S.load_cache(args.struct_cache)
    rows = np.asarray(task.valid_rows, dtype=np.int64)
    out: dict[str, Any] = {
        "schema": "lcrc-pcba-boc-zeroing-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "split": "official Validation only",
        "test_read": False,
        "selection": C.track_rel(Path(args.selection)),
        "selection_sha256": C.sha256_file(Path(args.selection)),
        "structure_cache": S.cache_identity(struct),
        "arms": {},
    }
    for arm in ("FULL", "BOND"):
        entry = selection["models"][arm]
        model, payload, audit, sha = T.restore_model(
            {
                "label": f"BOC_{arm}_best_valid",
                "model": arm,
                "epoch": int(entry["best_epoch"]),
                "path": entry["best_checkpoint"],
                "sha256": entry["best_checkpoint_sha256"],
            },
            task,
            struct,
            device,
        )
        modes: dict[str, Any] = {}
        for name, flags in MODES.items():
            rep = evaluate(model, task, struct, rows, device, **flags)
            modes[name] = {
                "valid_ap": float(rep["ap"]),
                "valid_bce": float(rep["bce"]),
                "n_tasks_scored": int(rep["n_tasks_scored"]),
                "logits_sha256": rep["logits_sha256"],
                "delta_vs_both_channels": None,
            }
        base = modes["both_channels"]["valid_ap"]
        for name in MODES:
            modes[name]["delta_vs_both_channels"] = float(modes[name]["valid_ap"] - base)
        out["arms"][arm] = {
            "checkpoint": entry["best_checkpoint"],
            "checkpoint_sha256": sha,
            "epoch": int(entry["best_epoch"]),
            "best_valid_ap_recorded": float(entry["best_valid_ap"]),
            "modes": modes,
            "full_vs_ring_zero_delta": float(modes["both_channels"]["valid_ap"] - modes["ring_zero"]["valid_ap"])
            if arm == "FULL"
            else 0.0,
            "full_vs_bond_zero_delta": float(modes["both_channels"]["valid_ap"] - modes["bond_zero"]["valid_ap"])
            if arm == "FULL"
            else 0.0,
        }
        del model
    C.write_json(args.out, out)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
