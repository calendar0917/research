"""Frozen-checkpoint inference: Validation (and the single one-shot Test read).

Reads ``run.json`` of a finished run, rebuilds the *exact* view and model
configuration, loads ``checkpoint_best.pt`` and writes one ``.npz`` per run with
the raw checkpoint output plus the official metrics for that split.

    python esr_eval.py --runs runs/lipo-base-s0 runs/lipo-sren-add-s0 \
        --split valid --out-dir analysis/preds --device cuda
    python esr_eval.py --runs runs/... --split test --out-dir analysis/test_preds --allow-test

Test is refused unless ``--allow-test`` is given *and*
``analysis/protocol_freeze.json`` exists; the caller is responsible for having
frozen every decision before that call (see REPORT.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import esr_common as C
import esr_data as D
import esr_model as M
import esr_train as T


def _config_from_run(run: dict[str, Any]) -> dict[str, Any]:
    info = run["model_info"]
    return {
        "dataset": run["dataset"],
        "arm": run["arm"],
        "seed": int(run["seed"]),
        "hidden_width": info.get("hidden_width"),
        "relation_mode": str(run.get("relation_mode", "real")),
        "max_dist": int(run.get("max_dist", 2)),
    }


def evaluate_run(run_dir: Path, split: str, out_dir: Path, device: torch.device) -> dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text())
    cfg = _config_from_run(run)
    view = D.load_view(
        cfg["dataset"],
        max_dist=cfg["max_dist"],
        relation_mode=cfg["relation_mode"],
        include_test=split == "test",
        workers=4,
    )
    if split == "test" and cfg["dataset"] == C.PCBA:
        raise RuntimeError("PCBA Test labels are unavailable; Test cannot be scored")
    model = M.build_model(
        arm=cfg["arm"],
        view=view,
        seed=cfg["seed"],
        device=device,
        hidden_width=cfg["hidden_width"],
    )[0]
    payload = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    metrics, logits = D.evaluate(model, view, split, device, batch_size=256)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_dir.name
    y = np.asarray(view.y[split], dtype=np.float64)
    ids = np.asarray(view.ids[split], dtype=np.int64)
    extra: dict[str, Any] = {}
    if view.task == "regression":
        raw = logits.reshape(-1) * float(view.target_norm["scale"]) + float(view.target_norm["mean"])
        extra["raw_pred"] = raw.astype(np.float64)
    np.savez_compressed(
        out_dir / f"{name}__{split}.npz",
        logits=np.asarray(logits, dtype=np.float32),
        y=y.astype(np.float32),
        ids=ids,
        task_names=np.asarray(view.task_names),
        **extra,
    )
    record = {
        "run": name,
        "dataset": cfg["dataset"],
        "arm": cfg["arm"],
        "seed": cfg["seed"],
        "split": split,
        "n": int(y.shape[0]),
        "metrics": metrics,
        "checkpoint": str(run_dir / "checkpoint_best.pt"),
        "checkpoint_epoch": int(payload.get("best_epoch") or payload.get("epoch") or 0),
        "parameters": int(run.get("parameters") or 0),
        "git": C.git_commit(),
    }
    (out_dir / f"{name}__{split}.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    C.log(f"eval {name} [{split}] primary={view.primary(metrics):.6f}")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="frozen-checkpoint inference")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    if args.split == "test":
        if not args.allow_test:
            raise SystemExit("refusing the Test split without --allow-test (one-shot rule)")
        if not (C.ANALYSIS / "protocol_freeze.json").exists():
            raise SystemExit("refusing Test before the protocol freeze exists")
        C.log("EXPOSED_TEST_ENGINEERING_COMPARISON_SET: one-shot Test read")
    device = torch.device(args.device)
    out = Path(args.out_dir)
    for run_dir in args.runs:
        evaluate_run(Path(run_dir), args.split, out, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
