"""Aggregate the CPU compute ledger for the frozen-transfer audit.

No GPU is used in this round.  The ledger records wall seconds per stage as
reported by the scripts themselves plus the per-cell probe timings.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import fta_common as K


def cpu_model() -> str:
    try:
        txt = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
        for line in txt.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # pragma: no cover
        pass
    return platform.processor() or "unknown"


def main() -> int:
    entries = []
    ledger = K.RUNS / "ledger.jsonl"
    if ledger.exists():
        entries = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    probe = json.loads((K.ANALYSIS / "probe_freeze.json").read_text())
    metrics = json.loads((K.ANALYSIS / "probe_monitor_metrics.json").read_text())

    by_stage: dict[str, float] = {}
    for e in entries:
        by_stage[e["stage"]] = by_stage.get(e["stage"], 0.0) + float(e["wall_seconds"])
    probe_cells = {c["id"]: float(c["wall_seconds"]) for c in probe["cells"]}
    main_cells = [c for c in probe["cells"] if c["kind"] == "main"]
    control_cells = [c for c in probe["cells"] if c["kind"] != "main"]

    freeze = json.loads((K.ANALYSIS / "representation_freeze.json").read_text())
    stats = json.loads((K.ANALYSIS / "representation_stats.json").read_text())
    artifact_seconds = {
        "export": float(freeze["wall_seconds"]),
        "stats": float(stats["wall_seconds"]),
        "probe": float(probe["wall_seconds_total"]),
        "score": float(metrics["wall_seconds"]),
    }

    payload = {
        "round": K.ROUND_ID,
        "device": "cpu",
        "gpu_used": False,
        "gpu_hours": 0.0,
        "a100_hours": 0.0,
        "machine": {
            "cpu_model": cpu_model(),
            "cpu_count": int(os.cpu_count() or 0),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_threads_probe": int(probe["reader_cfg"].get("torch_threads", 1)),
            "platform": platform.platform(),
        },
        "wall_seconds_by_stage_ledger": by_stage,
        "wall_seconds_by_stage_from_artifacts": artifact_seconds,
        "wall_seconds_total": float(sum(artifact_seconds.values())) + float(by_stage.get("analyze", 0.0))
        + float(by_stage.get("checks", 0.0)),
        "note_ledger_reset": ("runs/ledger.jsonl was reset once after an instrument-only fix "
                              "(head-state hash recorded from the tensor dict instead of the "
                              "pickled file); export/stats wall times are read from their own "
                              "artefacts, so the ledger covers every stage of the producing run."),
        "probe_cells": probe_cells,
        "probe_main_cells_total_s": float(sum(float(c["wall_seconds"]) for c in main_cells)),
        "probe_control_cells_total_s": float(sum(float(c["wall_seconds"]) for c in control_cells)),
        "cpu_hours_total_single_core_equivalent": float(
            (sum(artifact_seconds.values()) + float(by_stage.get("analyze", 0.0))
             + float(by_stage.get("checks", 0.0))) / 3600.0),
        "n_probe_cells": len(probe["cells"]),
        "monitor_scoring_seconds": float(metrics["wall_seconds"]),
        "peak_rss_mb_export": float(json.loads(
            (K.ANALYSIS / "representation_freeze.json").read_text())["peak_rss_mb"]),
        "note": ("probe cells run single-threaded and sequentially for bitwise "
                 "reproducibility; the export forward runs on the same CPU. "
                 "No A100 hour is consumed by this round."),
    }
    K.write_json(K.ANALYSIS / "cpu_ledger.json", payload)
    K.log(f"ledger: {payload['wall_seconds_total']:.1f}s total, 0 GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
