"""Single entry point for the support-conditioned shrinkage round.

    uv run --group lipo python code/run.py prepare       # structure prep + gates + freeze
    uv run --group lipo python code/run.py preflight     # Train-only feasibility report
    uv run python code/run.py smoke                      # <=8 short implementation jobs
    uv run python code/run.py status                     # fail-closed ledger state
    uv run python code/sc_dispatch.py --backend rr ...   # formal runs (two A100 pools)
    uv run python code/run.py analyze                    # tables + bootstrap + groups
    uv run python code/run.py report                      # REPORT.md + figures
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = argv[0] if argv else "help"
    rest = argv[1:]
    if stage in ("prepare", "preflight"):
        import sc_prepare
        if stage == "preflight":
            return sc_prepare.main(["select"])
        return sc_prepare.main(rest or ["all"])
    if stage == "smoke":
        import sc_smoke
        return sc_smoke.main(rest)
    if stage == "status":
        import sc_dispatch
        return sc_dispatch.main(["--dry-run", *rest])
    if stage == "dispatch":
        import sc_dispatch
        return sc_dispatch.main(rest)
    if stage == "analyze":
        import sc_analysis
        return sc_analysis.main(rest)
    if stage == "report":
        import sc_report
        return sc_report.main(rest)
    if stage == "tests":
        import subprocess
        return subprocess.run(["uv", "run", "--group", "lipo", "pytest", "-q", "tests"],
                              cwd=str(CODE.parent)).returncode
    print(__doc__ or "", file=sys.stderr)
    print("stages: prepare | preflight | smoke | status | dispatch | analyze | report | tests",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
