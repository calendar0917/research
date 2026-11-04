"""Run the preflight gates and write ``analysis/gates.json``."""

from __future__ import annotations

import argparse
from pathlib import Path

import wb_checks as CH
import wb_common as C


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS / "gates.json"))
    args = parser.parse_args(argv)
    gates = CH.run_all()
    C.write_json(Path(args.out), gates)
    C.log(f"gates all_passed={gates['all_passed']} -> {C.track_rel(Path(args.out))}")
    for name, ok in gates["checks"].items():
        C.log(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if gates["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
