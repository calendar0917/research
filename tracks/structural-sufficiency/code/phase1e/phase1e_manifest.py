"""Emit the Phase 1e run manifest: one row per artefact, with its sha256.

The manifest is the reproducibility record for the round: every committed JSON
and markdown artefact, its hash, the command that produced it, and the commit
the code was frozen at.  Only hashes of tracked files are recorded, so the
manifest is stable across machines.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import phase1e_common as C  # noqa: E402

OUT = C.RESULTS_1E / "run_manifest.tsv"

_COMMON = "PYTHONPATH=. .venv/bin/python"
_ROWS = [

    ("label_audit.json", "A1", f"{_COMMON} code/phase1e/phase1e_label_audit.py", "CPU"),
    ("error_attribution.json", "A2", f"{_COMMON} code/phase1e/phase1e_analysis.py", "CPU"),
    ("zero_training_routing.json", "A3", f"{_COMMON} code/phase1e/phase1e_analysis.py", "CPU"),
    ("reference_audit.json", "B1", f"{_COMMON} code/phase1e/phase1e_reference_audit.py", "CPU"),
    (
        "strong_baseline.json",
        "B2",
        "rr run res ssrc-p1e-zinc-baseline-s0 --pool res-gpu0 -- "
        ".venv/bin/python code/phase1e/phase1e_baseline.py run --device cuda",
        "GPU a100-2",
    ),
    ("decision.json", "A4", f"{_COMMON} code/phase1e/phase1e_decision.py", "CPU"),
    ("phase_c_design.md", "C", "(authored; design only, no run)", "-"),
    ("REPORT.md", "report", "(authored)", "-"),
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=None, help="code commit to record")
    args = ap.parse_args(argv)

    commit = args.commit or _commit()
    lines = ["stage\tartefact\tsha256\tcommit\tbackend\tcommand"]
    missing: list[str] = []
    for name, stage, cmd, backend in _ROWS:
        path = C.RESULTS_1E / name
        if not path.exists():
            missing.append(name)
            continue
        lines.append(
            f"{stage}\t{path.relative_to(REPO)}\t{_sha(path)}\t{commit}\t{backend}\t{cmd}"
        )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"[manifest] {len(lines) - 1} artefacts -> {OUT.relative_to(REPO)}")
    if missing:
        print(f"[manifest] MISSING: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
