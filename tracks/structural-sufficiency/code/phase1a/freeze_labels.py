"""Freeze a Phase 1a label artifact containing only Fit + Monitor labels.

This is the only file the GPU host needs for supervised training.  It excludes
the internal Reserved 504 rows and the official valid/test rows.  The source
CSV is streamed; disallowed labels are not retained in the output mapping.

Run:
    uv run python tracks/structural-sufficiency/code/phase1a/freeze_labels.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phase1a_data import load_frozen_split  # noqa: E402

SPLIT = REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"
CSV = REPO / "data/lipophilicity/Lipophilicity.csv"
OUT = REPO / "tracks/structural-sufficiency/results/phase1a/labels_fit_monitor.json"


def main() -> int:
    split = load_frozen_split(SPLIT)
    allowed = sorted(set(split.fit) | set(split.monitor))
    allowed_set = set(allowed)
    reserved = set(split.reserved)
    if allowed_set & reserved:
        raise SystemExit("allowed/reserved overlap")

    labels: dict[int, float] = {}
    with open(CSV, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        exp_index = header.index("exp")
        for i, row in enumerate(reader):
            if i in allowed_set:
                labels[i] = float(row[exp_index])
    if set(labels) != allowed_set:
        raise SystemExit(f"label coverage mismatch: {len(labels)} vs {len(allowed_set)}")
    if set(labels) & reserved:
        raise SystemExit("reserved label leaked into artifact")

    payload = {
        "protocol_id": "ssrc-phase1a-lipo-v1",
        "status": "FROZEN_FIT_MONITOR_ONLY",
        "source_csv_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
        "rows": allowed,
        "labels": {str(k): labels[k] for k in allowed},
        "row_sha256": hashlib.sha256(
            ",".join(str(r) for r in allowed).encode("utf-8")
        ).hexdigest(),
        "scope": {
            "fit_rows": len(split.fit),
            "monitor_rows": len(split.monitor),
            "reserved_rows_excluded": len(split.reserved),
            "reserved_labels_present": False,
            "official_valid_test_labels_present": False,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  rows: {len(allowed)} (fit {len(split.fit)} + monitor {len(split.monitor)})")
    print(f"  reserved excluded: {len(split.reserved)}")
    print(f"  artifact sha256: {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
