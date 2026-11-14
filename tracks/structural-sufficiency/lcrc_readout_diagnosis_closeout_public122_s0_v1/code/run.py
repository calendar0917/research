"""Entry point for the readout-diagnosis closeout round.

Usage::

    uv run python code/run.py source-audit --device cpu
    uv run python code/run.py smoke   --device cpu
    uv run python code/run.py probe   --rule P_WARM --device cuda --out-dir ...
    uv run python code/run.py analyze
    uv run python code/run.py report
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import rc_common as R  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "source-audit":
        import rc_source_audit
        return rc_source_audit.main(rest)
    if cmd == "probe":
        import rc_probe
        return rc_probe.main(rest)
    if cmd == "smoke":
        # smoke = 4 updates per probe into a scratch dir, never a result
        import rc_probe
        import tempfile
        device = "cpu"
        if "--device" in rest:
            device = rest[rest.index("--device") + 1]
        out = Path(tempfile.mkdtemp(prefix="rc_smoke_"))
        for rule in ("P_WARM", "P_POST", "P_PRE"):
            rc_probe.train_probe(rule=rule, device=__import__("torch").device(device),
                                 out_dir=out / rule, smoke=True)
        R.log(f"smoke done -> {out}")
        return 0
    if cmd == "analyze":
        import rc_analysis
        return rc_analysis.main(rest)
    if cmd == "report":
        import rc_report
        return rc_report.main(rest)
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
