"""The single, one-shot Test read after every decision has been frozen.

Refuses to run unless
  * ``analysis/protocol_freeze.json`` exists,
  * ``analysis/phase_decisions.json`` exists (the decision file is written),
  * the caller passes ``--acknowledge-exposure``.

Lipo / ESOL / Tox21 only: the frozen PCBA label artifact is NaN on every Test row
and the loader refuses it, so no PCBA Test number exists in this round.

Every output file is stamped ``EXPOSED_TEST_ENGINEERING_COMPARISON_SET``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import esr_common as C
import esr_eval as E

LABEL = "EXPOSED_TEST_ENGINEERING_COMPARISON_SET"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="one-shot Test scoring")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", default=str(C.ANALYSIS / "test_preds"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--acknowledge-exposure", action="store_true")
    args = ap.parse_args(argv)
    if not args.acknowledge_exposure:
        raise SystemExit("refusing the Test read without --acknowledge-exposure")
    for required in ("protocol_freeze.json", "phase_decisions.json", "main_table.csv"):
        if not (C.ANALYSIS / required).exists():
            raise SystemExit(f"refusing Test before {required} exists")
    device = torch.device(args.device)
    out = Path(args.out_dir)
    records = []
    for run_dir in args.runs:
        run = json.loads((Path(run_dir) / "run.json").read_text())
        if run["dataset"] == C.PCBA:
            C.log(f"skipping {run_dir}: PCBA Test labels are NaN in the frozen artifact")
            continue
        record = E.evaluate_run(Path(run_dir), "test", out, device)
        record["label"] = LABEL
        records.append(record)
        C.log(f"{LABEL}: {record['run']} test primary={record['metrics'].get('rmse', record['metrics'].get('macro_roc_auc'))}")
    payload: dict[str, Any] = {
        "label": LABEL,
        "read_utc": C.utcnow(),
        "n_runs": len(records),
        "test_policy": "single read after the Validation-only freeze; no Test number influenced any decision",
        "pcba_test": "not scoreable: frozen label artifact is NaN on Test",
        "records": records,
    }
    C.write_json(C.ANALYSIS / "test_scores.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
